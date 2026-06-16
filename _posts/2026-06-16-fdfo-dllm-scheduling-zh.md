---
layout: post
title: "FDFO: 让 dLLM 推理从「整批等最慢」变成「谁先完成谁先走」"
date: 2026-06-16 12:00:00 +0800
categories: [systems]
lang: zh
translation: /2026/06/16/fdfo-dllm-scheduling/
permalink: /2026/06/16/fdfo-dllm-scheduling-zh/
excerpt: "把 FDFO 抽象成 dLLM 调度器的框架级执行模式：完成即释放，不再整批等最慢。基于向 SGLang 提交的 #27551 与 #27877 两个 PR。"
---

# FDFO: 让 dLLM 推理从「整批等最慢」变成「谁先完成谁先走」

> 本文基于作者向 SGLang 提交的两个关于扩散语言模型调度器优化的 PR：[#27551](https://github.com/sgl-project/sglang/pull/27551) 和 [#27877](https://github.com/sgl-project/sglang/pull/27877)。前者把 FDFO 抽象成 dLLM 的框架级调度能力，后者进一步复用未完成 block 的 KV/req slot，减少 FDFO 轮次里的内存分配和调度往返开销。

## 先从几个基础概念说起

大模型推理服务本质上是在做一件事：不断接收用户请求，把多个请求合成 batch 放到 GPU 上跑，然后把生成结果返回给用户。这里的 scheduler 可以理解成推理系统里的「调度员」：它决定哪些请求能进 GPU、哪些请求要等、什么时候释放资源、什么时候把新请求补进来。

在传统自回归 LLM 里，模型通常一次生成一个 token。每个请求会经历 prefill 和 decode 两个阶段：

- prefill：把 prompt 送进模型，建立上下文。
- decode：基于已有上下文，一个 token 一个 token 地往后生成。

为了让后续 decode 更快，推理框架会保存每层 attention 的 KV cache。KV cache 可以理解成「模型已经看过的上下文缓存」。没有它，每生成一个 token 都要重新计算整个历史上下文，成本会非常高。

dLLM，也就是 diffusion language model，生成方式不太一样。它不是严格从左到右一个 token 一个 token 地生成，而是先准备一段带 mask 的 block，然后通过多轮 denoise 把 mask 位置逐步填出来。比如 block size 是 32 时，一个请求可能先拿到 32 个 mask token；每一轮 forward 后，算法根据模型预测的置信度选择若干位置填成真实 token；多轮之后，这个 block 全部解决，再进入下一个 block。

所以 dLLM 推理里有一个很重要的概念：一个 block 不一定一轮就完成。简单 prompt、高置信度的 block 可能几步就结束；复杂 prompt、低置信度的 block 可能要跑很多步。

## 原来的同步调度问题在哪里

原来的 dLLM batch 更接近同步执行：同一个 batch 里的请求一起进入 denoise loop，然后一起等到整个 batch 的 block 都完成，scheduler 才重新拿到控制权。

这会导致一个典型问题：head-of-line blocking，也就是队头阻塞。它不是说某个请求真的排在队头，而是说一批请求里最慢的那个会拖住所有已经完成的请求。

看一个小例子。假设同一个 batch 有三个请求：

| 请求 | 完成当前 block 需要的 denoise 步数 |
| --- | --- |
| A | 3 |
| B | 8 |
| C | 2 |

同步模式下，这个 batch 要跑到第 8 步才整体结束。C 在第 2 步已经完成，却还要等到第 8 步；A 在第 3 步已经完成，也要等到第 8 步。用一个很粗糙但直观的公式看，batch 的完成时间由 `max(T_i)` 决定，其中 `T_i` 是每个请求自己的完成步数。多出来的等待是：

```text
(8 - 3) + (8 - 8) + (8 - 2) = 11 个 request-step
```

这 11 个 request-step 背后对应几类实际损失：

- 已经完成的 block 仍然占着 batch 位置，新的请求不能及时补进来。
- 已经完成的 block 可能还会参与后续 forward，GPU 计算被花在「不再产生新结果」的位置上。
- 用户侧看见的是尾部请求把整批请求的延迟都拉高了。
- 算法实现里，每个 dLLM algorithm 都要自己写完整 `run()` 循环，调度策略和 token 选择策略耦合在一起。

最后一点尤其影响工程演进。LowConfidence、JointThreshold 这类名字描述的是「这一轮应该解开哪些 token」的策略；FDFO 描述的是「完成的请求什么时候离开 batch」的执行模式。它们本来是两个正交问题。如果把 FDFO 写死到某个算法类里，就会出现两个问题：其他算法无法复用 FDFO，有状态算法也很难把跨轮状态安全地传回 scheduler。

## FDFO 的核心思想

FDFO 是 First-Done-First-Out。直白地说，就是：谁的当前 block 先完成，谁就先离开这个 batch。

这不是改变 dLLM 的 token 选择算法。LowConfidence 还是 LowConfidence，JointThreshold 还是 JointThreshold。FDFO 改的是执行节奏：

1. scheduler 组一个 dLLM batch。
2. worker 对这个 batch 做一轮或少量几轮 `forward -> step`。
3. 每个请求独立判断当前 block 是否已经完成。
4. 完成的请求立刻提交生成结果，释放或复用资源，继续进入下一个阶段。
5. 没完成的请求保存当前 block 的中间 token 和算法状态，下次继续 denoise。

回到前面的 A/B/C 例子。FDFO 下，C 第 2 步完成就可以离开，A 第 3 步完成就可以离开，B 自己跑到第 8 步。更重要的是，C 和 A 离开后，scheduler 可以把新的请求补进来，而不是让它们陪 B 等到最后。

从排队系统的角度看，同步 dLLM batching 把一批请求绑成了一个大任务，完成时间取决于最慢者；FDFO 则把每个请求的完成事件暴露给 scheduler，让 scheduler 能更细粒度地回收资源、补充新工作。它降低的是 batch 内部的尾部拖累，而不是改变模型本身的计算语义。

## 为什么 FDFO 对 dLLM 特别自然

自回归 decode 每轮通常只前进一个 token，请求之间的「完成差异」主要体现在总生成长度不同。而 dLLM 的一个 block 内部就有多轮 denoise，不同请求在同一个 block 上的收敛速度可能差很多。

这意味着 dLLM batch 天然存在「短任务」和「长任务」混在一起的问题：

- 置信度高的请求可能很快把 mask 全部解开。
- 置信度低的请求每轮只能解开少量 token。
- 有些算法还会在 mask 全部消失后做 token-to-token edit，进一步拉开步数差异。

同步 batch 会把这些差异全部折叠成最慢请求的步数。FDFO 则承认这些请求本来就应该有不同的完成时间，并把这个完成时间交还给 scheduler。

## 工程设计一：把「算法策略」和「执行模式」拆开

#27551 的第一个关键设计，是重构 `DllmAlgorithm`。重构之后，子类不再需要各自实现完整的 denoise loop，而是只实现一个单步函数：

```text
step(forward_batch, full_logits, states) -> done_flags
```

这个 `step()` 做三件事：

1. 读取当前 forward 的 logits。
2. 按算法策略更新 `forward_batch.input_ids`。
3. 返回每个 block 是否已经拥有可提交的最终 KV。

执行循环则由基类统一拥有：

- 同步模式：`_run_sync()` 会持续 `forward -> step`，直到 batch 内所有 block 完成。
- FDFO 模式：`_run_fdfo()` 会执行 `forward -> step`，拿到每个请求的 done flag，然后把控制权交还给 scheduler。

这样拆完之后，LowConfidence 和 JointThreshold 都变得更像「纯算法」。

LowConfidence 的逻辑是：对每个 mask 位置看模型预测 token 的置信度，置信度超过阈值就解开；如果一轮里没有任何位置超过阈值，就至少选择置信度最高的一个位置，保证过程能向前推进。

JointThreshold 更复杂一些。它不仅做 mask-to-token，也就是把 mask 填成 token，还会做 token-to-token edit，也就是对已经生成出的 token 继续修正。它需要保存跨 step 状态，比如已经做了多少轮 post-edit、哪些位置属于原始 prompt、不应该被改动。因此它是一个有状态算法。

重构后的接口同时支持这两类算法：

- 无状态算法：`init_step_state()` 默认返回 `None` 状态即可。
- 有状态算法：在 `init_step_state()` 里创建每个请求自己的状态，并在 FDFO 轮次之间由框架保存和恢复。

这个边界很干净：算法只关心「这一轮怎么改 token」，框架只关心「这一轮之后谁完成、谁继续」。

## 工程设计二：用 request 字段承载 FDFO 生命周期

FDFO 的难点不在于知道谁完成了，而在于「没完成的请求下轮怎么接着跑」。这要求 scheduler 能保存两个东西：

- 当前 block 已经 denoise 到什么样子。
- 当前算法有没有私有的跨轮状态。

#27551 在 `Req` 上引入了两个核心字段：

```text
req.dllm_incomplete_ids
req.dllm_algo_state
```

`dllm_incomplete_ids` 保存未完成 block 当前的 token 序列。它是框架通用状态，不管用 LowConfidence 还是 JointThreshold，只要 block 没完成，都需要把这个半成品带到下一轮。

`dllm_algo_state` 是算法私有状态。对 LowConfidence 来说它通常是 `None`；对 JointThreshold 来说，它会保存 `post_edit_steps`、`prompt_mask` 等信息。scheduler 不理解这些状态的内部含义，只负责原样带回 worker。

worker 侧会在 FDFO 模式下从 batch 里的 req 收集 `dllm_algo_state`，传给 `DllmAlgorithm.run()`。算法执行完一轮后，再通过 `GenerationBatchResult` 返回：

```text
accept_length_per_req_cpu
dllm_algo_state
```

其中 `accept_length_per_req_cpu` 可以理解成每个请求这轮接受了多少 token。对当前 block 来说，值为 0 表示还没完成，需要继续 denoise；值为 `block_size` 表示当前 block 已完成，可以提交。

scheduler 收到结果后按请求逐个处理：

- 未完成：把本轮的半成品写入 `dllm_incomplete_ids`，把算法状态写入 `dllm_algo_state`，等待下一轮继续。
- 已完成：清空 `dllm_incomplete_ids` 和 `dllm_algo_state`，把完整 block 写入输出，更新 finish state，并在请求结束时释放 KV cache。

这套机制的好处是，FDFO 没有侵入具体算法，也没有要求 scheduler 理解算法细节。scheduler 只看「完成或未完成」，算法状态则作为 opaque state 被框架搬运。

## 工程设计三：保留同步模式，FDFO 变成独立开关

另一个很重要的 API 设计是 `--dllm-fdfo`。

它独立于 `--dllm-algorithm`。也就是说，算法和执行模式可以自由组合：

```text
--dllm-algorithm LowConfidence  --dllm-fdfo
--dllm-algorithm LowConfidence  --no-dllm-fdfo
--dllm-algorithm JointThreshold --dllm-fdfo
--dllm-algorithm JointThreshold --no-dllm-fdfo
```

这比新增一个 `LowConfidenceFDFO` 类更可维护。因为 FDFO 不是 LowConfidence 的一部分，它应该是所有 dLLM algorithm 都能选择的执行模式。

这个设计也让 correctness 更容易验证。在 batch size 为 1 时，FDFO 和同步模式看到的 forward shape 是一样的。如果实现正确，两者应该产出完全一致的结果。PR 中也用 LLaDA2.0-mini 在 temperature 0 下验证了 LowConfidence 和 JointThreshold 的 batch size 1 byte-identical parity。

## 工程设计四：#27877 里的 in-place KV/req slot 复用

#27551 解决了「完成的请求要尽快离开」的调度问题，但 FDFO 也引入了一个新的工程成本：未完成的 block 会跨多个 scheduler round 继续 denoise。如果每一轮都释放 KV slot 和 req slot，下一轮再重新分配，就会产生 allocator churn，也会增加 HBM 写入和调度往返开销。

#27877 做的优化是：对未完成的 FDFO block，原地复用它已经占用的 req slot 和 KV slot。

具体识别条件很直接：

```text
r.req_pool_idx is not None and bool(r.dllm_incomplete_ids)
```

这表示这个请求还保留着 request pool 位置，并且有一个未完成 block。此时 `alloc_for_extend()` 不再为这个 block 重新申请 KV，而是从 `req_to_token` 里取回已经映射好的那段 KV slot。只有真正的新请求，或者已经进入新 block 的请求，才会走 fresh allocation。

这个优化有两个实际收益：

- 减少未完成 block 在 FDFO 轮次之间的 free/alloc 循环。
- 当一轮 FDFO 没有任何 block 完成时，可以直接在同一个 batch 上继续 `forward -> step`，不必返回 scheduler 再绕一圈。

第二点就是 #27877 里的 in-place re-loop。它的规则也很克制：如果这一轮没有任何 block 完成，就继续在当前 batch 上 denoise；一旦有任意 block 完成，立刻 yield 给 scheduler，让 FDFO 及时释放完成请求。这样既减少了无意义的 scheduler round-trip，又保留了 FDFO 的核心语义。

这里还有一个内存分页上的细节。dLLM scheduler 会确保 `page_size` 是 `block_size` 的倍数，因此一个 dLLM block 是 page-aligned 的。第一次创建这个 block 时，它已经是合法分页分配；后续原地复用这段 block KV，在 paging 语义上是安全的。

## 性能结果说明了什么

#27551 在 H200、TP=1、LLaDA2.0-mini、GSM8K 200 examples 上给出的 LowConfidence 结果是：

| max-running-requests | 同步模式 | FDFO | 提升 |
| --- | --- | --- | --- |
| 4 | 525 tok/s | 680 tok/s | 1.30× |
| 16 | 918 ～ 959 tok/s | 1276 ～ 1378 tok/s | 约 1.45× |

准确率在两个模式下保持在相同区间。batch size 大于 1 时，不同运行之间本身会有大约 0.01 的分数波动，PR 中把它归因于 kernel batch non-invariance 和 MoE routing 等运行噪声；FDFO 和同步模式的分布是重叠的。

#27877 进一步和 #27551 做 controlled A/B，对 FDFO 模式本身做内存复用优化：

| batch size | #27551 | #27877 | 提升 |
| --- | --- | --- | --- |
| 4 | 614.9 tok/s | 660.4 tok/s | +7.4% |
| 16 | 1288 tok/s | 1305 tok/s | +1.3% |

这个结果也符合直觉。小 batch 下，scheduler round-trip 和 allocator churn 占比更高，所以 KV/req slot 复用收益更明显；大 batch 下，forward compute 本身占主导，优化幅度会被计算成本稀释。

## 正确性为什么能站住

FDFO 看起来改变了 batch 调度顺序，所以最容易被问到的问题是：它会不会改变结果？

可以分两层看。

第一层是 batch size 1。此时没有其他请求参与，FDFO 和同步模式的 forward shape 一致，执行语义也应该一致。因此 batch size 1 parity 是很强的 correctness check。PR 中 LowConfidence 和 JointThreshold 都通过了 byte-identical 验证，说明单请求语义没有被 FDFO 改坏，也说明 JointThreshold 的跨 step 状态搬运是正确的。

第二层是 batch size 大于 1。GPU kernel、MoE routing、batch 形状变化等因素可能让不同 batch 编排下的数值细节出现轻微差异。因此这里更合理的验证方式不是要求 byte-identical，而是看准确率分布是否落在同一波动区间，以及是否有 OOM、KV leak、状态错乱等系统问题。PR 中的 GSM8K 结果和无泄漏验证覆盖的就是这一层。

## 这次设计最值得借鉴的地方

这次 FDFO 的设计价值不只是性能提升，更在于抽象边界足够清楚。

第一，token selection 和 execution mode 被拆开了。算法类只实现 `step()`，不用关心自己是在同步模式还是 FDFO 模式下运行。

第二，跨轮状态有明确载体。通用半成品 token 放在 `dllm_incomplete_ids`，算法私有状态放在 `dllm_algo_state`。scheduler 不窥探算法状态，算法也不直接操纵 scheduler 内部结构。

第三，完成状态被表达成简单的 per-request accept length。对 scheduler 来说，0 就是继续，`block_size` 就是提交。这让 FDFO 的生命周期非常直接。

第四，优化分层很自然。#27551 先把能力抽象正确，#27877 再在这个抽象上做 HBM 复用和 in-place re-loop。先把语义做干净，再优化路径，这是服务系统里很稳的演进方式。

## 致谢

本文介绍的 FDFO 框架化设计与后续 KV/req slot 原地复用优化，来自小红书（RedNote）引擎基础架构部 Engine Architecture Group 5 的工作。由 Engine Architecture Group 5, Engine Infrastructure Department, Xiaohongshu (RedNote) 的 Huayi Jin、Zhaokai Luo、Junxiang Wu、Bing Zhang、Chenchen Hong、Bing Tian 共同贡献；感谢小红书团队和同学把 FDFO 从一个算法内的特殊实现，设计并打磨成 dLLM scheduler 可以复用的框架能力。

## 小结

原来的同步 dLLM scheduling 把一个 batch 绑成了一个整体，完成时间受最慢请求支配。对于 denoise 步数差异明显的 dLLM 来说，这会造成明显的 batch 内部队头阻塞。

FDFO 的思想很简单：当前 block 谁先完成，谁先离开。它让 scheduler 能更快提交结果、释放资源、补充新请求，从而提升吞吐并降低不必要等待。

工程上，这个 feature 的关键不是多写一个 FDFO 算法类，而是把 FDFO 抽象成框架级执行模式。`step()` 机制让算法策略保持单纯，`dllm_incomplete_ids` 和 `dllm_algo_state` 让跨轮状态有地方放，`accept_length_per_req_cpu` 让 scheduler 用统一方式理解完成与否。再加上 #27877 的 KV/req slot 原地复用，FDFO 不仅语义清晰，也开始具备更好的系统效率。

如果用一句话概括：FDFO 让 dLLM 推理从「整批等最慢」变成「完成即释放」，而这次工程设计把这个思想变成了所有 dLLM algorithm 都能复用的框架能力。
