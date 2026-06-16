---
layout: post
title: "FDFO: First Done, First Out — Rethinking dLLM Inference Scheduling"
date: 2026-06-16 07:02:00 +0800
categories: [systems]
lang: en
translation: /2026/06/16/fdfo-dllm-scheduling-zh/
excerpt: "How a framework-level First-Done-First-Out execution mode lets diffusion-LLM serving stop waiting for the slowest request in a batch — released to SGLang in PRs #27551 and #27877."
---

# FDFO: First Done, First Out — Rethinking dLLM Inference Scheduling

> This post is based on two PRs I contributed to SGLang for diffusion-LLM scheduler optimization: [#27551](https://github.com/sgl-project/sglang/pull/27551) and [#27877](https://github.com/sgl-project/sglang/pull/27877). The first lifts **FDFO** into a framework-level scheduling capability for dLLMs; the second reuses the KV/req slots of unfinished blocks in place, cutting the memory-allocation and scheduling round-trip overhead inside FDFO rounds.

## Some background first

LLM inference serving is, at its core, doing one thing over and over: accept user requests, pack several of them into a batch to run on the GPU, and return the generated results. The **scheduler** is the dispatcher of the inference system — it decides which requests get onto the GPU, which ones wait, when to release resources, and when to admit new work.

In a traditional autoregressive LLM, the model usually generates one token at a time. Each request goes through two phases:

- **prefill**: feed the prompt into the model and build up its context.
- **decode**: generate one token after another on top of the existing context.

To make later decode steps faster, the framework keeps a per-layer **KV cache** for attention — think of it as a cache of "the context the model has already seen." Without it, every new token would require recomputing the entire history, which is enormously expensive.

A **dLLM** — diffusion language model — generates differently. It doesn't strictly go left to right one token at a time. Instead it prepares a masked **block**, then fills in the masked positions through multiple rounds of **denoise**. With a block size of 32, for example, a request might start with 32 masked tokens; after each forward pass the algorithm picks some positions to commit to real tokens based on the model's predicted confidence; after several rounds the whole block is resolved, and the request moves on to the next block.

That gives dLLM inference an important property: **a block does not necessarily finish in a single round.** A simple, high-confidence block may finish in a few steps; a complex, low-confidence block may take many.

## Where the old synchronous scheduling falls short

The original dLLM batch was closer to synchronous execution: all requests in a batch enter the denoise loop together, and the scheduler only regains control once *every* block in the batch is done.

This causes a classic problem — **head-of-line blocking**. It's not that some request is literally stuck at the front of a queue; it's that the slowest request in the batch holds back every request that has already finished.

Consider a small example. Suppose one batch has three requests:

| Request | Denoise steps to finish the current block |
| --- | --- |
| A | 3 |
| B | 8 |
| C | 2 |

In synchronous mode, the batch doesn't end until step 8. C finished at step 2 but still waits until step 8; A finished at step 3 but also waits until step 8. With a crude but intuitive formula, the batch's completion time is governed by `max(T_i)`, where `T_i` is each request's own finishing step count. The extra waiting is:

```text
(8 - 3) + (8 - 8) + (8 - 2) = 11 request-steps
```

Those 11 request-steps map to several real costs:

- A finished block still occupies its batch slot, so new requests can't be admitted in time.
- A finished block may still participate in subsequent forwards, spending GPU compute on positions that no longer produce new results.
- On the user side, the tail request inflates the latency of the whole batch.
- In the algorithm code, every dLLM algorithm has to write its own full `run()` loop, coupling the scheduling policy with the token-selection policy.

That last point especially hurts long-term evolution. Names like **LowConfidence** and **JointThreshold** describe *which tokens to unmask this round*; **FDFO** describes *when a finished request leaves the batch*. These are two orthogonal concerns. Hard-coding FDFO into a specific algorithm class creates two problems: other algorithms can't reuse FDFO, and stateful algorithms struggle to hand their cross-round state safely back to the scheduler.

## The core idea of FDFO

FDFO stands for **First-Done-First-Out**. Plainly: whoever's current block finishes first leaves the batch first.

This does not change the dLLM token-selection algorithm. LowConfidence is still LowConfidence; JointThreshold is still JointThreshold. What FDFO changes is the **execution rhythm**:

1. The scheduler forms a dLLM batch.
2. The worker runs one — or a few — `forward -> step` rounds on that batch.
3. Each request independently checks whether its current block is done.
4. Finished requests immediately commit their results, release or reuse resources, and move on to the next stage.
5. Unfinished requests save the in-progress tokens of the current block plus the algorithm state, and continue denoising next time.

Back to the A/B/C example. Under FDFO, C can leave once it finishes at step 2, A can leave at step 3, and B runs on its own to step 8. More importantly, once C and A leave, the scheduler can admit new requests instead of making them wait out B until the end.

From a queueing-system perspective, synchronous dLLM batching binds a group of requests into one big task whose completion time is dictated by the slowest member; FDFO instead exposes each request's completion event to the scheduler, letting it reclaim resources and inject new work at a finer granularity. What it reduces is the tail drag *inside* a batch — not the model's computational semantics.

## Why FDFO fits dLLMs especially well

An autoregressive decode step usually advances only one token, so the "completion difference" between requests mainly comes from differing total generation lengths. But a single dLLM block already contains multiple denoise rounds, and different requests can converge at very different speeds within the same block.

That means a dLLM batch naturally mixes "short tasks" and "long tasks":

- A high-confidence request may unmask all positions quickly.
- A low-confidence request may unmask only a few tokens per round.
- Some algorithms even perform token-to-token edits after all masks are gone, widening the step-count gap further.

A synchronous batch folds all of this variance into the slowest request's step count. FDFO instead acknowledges that these requests were always meant to finish at different times, and hands that completion timing back to the scheduler.

## Engineering design 1: separate "algorithm strategy" from "execution mode"

The first key design in #27551 is a refactor of `DllmAlgorithm`. Afterward, subclasses no longer implement a full denoise loop each; they implement a single step function:

```text
step(forward_batch, full_logits, states) -> done_flags
```

This `step()` does three things:

1. Read the logits of the current forward.
2. Update `forward_batch.input_ids` according to the algorithm's strategy.
3. Return, per block, whether it now has a committable final KV.

The execution loop is owned uniformly by the base class:

- **Synchronous mode**: `_run_sync()` keeps doing `forward -> step` until every block in the batch is done.
- **FDFO mode**: `_run_fdfo()` runs `forward -> step`, collects each request's done flag, and then hands control back to the scheduler.

After this split, LowConfidence and JointThreshold both become much more like "pure algorithms."

**LowConfidence**'s logic: for each masked position, look at the confidence of the model's predicted token, and unmask it if confidence clears a threshold; if no position clears the threshold in a round, unmask at least the single highest-confidence position to guarantee forward progress.

**JointThreshold** is more involved. It does both mask-to-token (filling masks with tokens) and token-to-token edits (revising already-generated tokens). It needs to keep cross-step state — how many post-edit rounds it has done, which positions belong to the original prompt and must not be touched — so it is a **stateful algorithm**.

The refactored interface supports both kinds:

- Stateless algorithms: `init_step_state()` simply returns `None`.
- Stateful algorithms: create per-request state in `init_step_state()`, and let the framework save and restore it between FDFO rounds.

The boundary is clean: the algorithm only cares about "how to change tokens this round," and the framework only cares about "who's done and who continues after this round."

## Engineering design 2: carry the FDFO lifecycle on request fields

The hard part of FDFO isn't knowing *who* finished — it's "how does an unfinished request resume next round." That requires the scheduler to persist two things:

- How far the current block has been denoised.
- Whether the current algorithm has private cross-round state.

#27551 introduces two core fields on `Req`:

```text
req.dllm_incomplete_ids
req.dllm_algo_state
```

`dllm_incomplete_ids` holds the current token sequence of an unfinished block. It's **framework-generic** state — whether you use LowConfidence or JointThreshold, as long as a block is unfinished, this work-in-progress has to be carried to the next round.

`dllm_algo_state` is **algorithm-private** state. For LowConfidence it's usually `None`; for JointThreshold it stores things like `post_edit_steps` and `prompt_mask`. The scheduler doesn't understand the internals of this state — it just carries it back to the worker verbatim.

On the worker side, FDFO mode collects `dllm_algo_state` from the reqs in the batch and passes it into `DllmAlgorithm.run()`. After running a round, the algorithm returns, via `GenerationBatchResult`:

```text
accept_length_per_req_cpu
dllm_algo_state
```

Here `accept_length_per_req_cpu` is how many tokens each request accepted this round. For the current block, `0` means it isn't done yet and needs more denoising; `block_size` means the current block is complete and can be committed.

When the scheduler receives the result, it handles requests one by one:

- **Unfinished**: write this round's work-in-progress into `dllm_incomplete_ids`, write the algorithm state into `dllm_algo_state`, and wait for the next round.
- **Finished**: clear `dllm_incomplete_ids` and `dllm_algo_state`, write the complete block to output, update the finish state, and release the KV cache when the request ends.

The benefit: FDFO doesn't intrude on any specific algorithm, and doesn't require the scheduler to understand algorithm details. The scheduler only sees "done or not done," while the algorithm state is shuttled around as opaque state by the framework.

## Engineering design 3: keep synchronous mode, make FDFO an independent switch

Another important API decision is `--dllm-fdfo`.

It's independent of `--dllm-algorithm`. In other words, algorithm and execution mode combine freely:

```text
--dllm-algorithm LowConfidence  --dllm-fdfo
--dllm-algorithm LowConfidence  --no-dllm-fdfo
--dllm-algorithm JointThreshold --dllm-fdfo
--dllm-algorithm JointThreshold --no-dllm-fdfo
```

This is more maintainable than adding a `LowConfidenceFDFO` class — because FDFO isn't part of LowConfidence; it should be an execution mode that *every* dLLM algorithm can opt into.

This design also makes correctness easier to verify. At batch size 1, FDFO and synchronous mode see the same forward shape, so a correct implementation should produce identical results. The PR verifies exactly this: byte-identical batch-size-1 parity for both LowConfidence and JointThreshold on LLaDA2.0-mini at temperature 0.

## Engineering design 4: in-place KV/req-slot reuse in #27877

#27551 solved the scheduling problem of "finished requests should leave as soon as possible," but FDFO also introduces a new engineering cost: an unfinished block keeps denoising across multiple scheduler rounds. If every round frees its KV slot and req slot and then re-allocates them next round, you get allocator churn, extra HBM writes, and more scheduling round-trips.

What #27877 optimizes: for an unfinished FDFO block, reuse the req slot and KV slot it already holds, **in place**.

The identification condition is direct:

```text
r.req_pool_idx is not None and bool(r.dllm_incomplete_ids)
```

This means the request still holds its request-pool slot and has an unfinished block. In that case `alloc_for_extend()` no longer requests fresh KV for the block; it retrieves the already-mapped KV slots back from `req_to_token`. Only genuinely new requests, or requests that have moved on to a new block, take the fresh-allocation path.

This optimization has two concrete payoffs:

- Fewer free/alloc cycles for unfinished blocks between FDFO rounds.
- When a round finishes *no* blocks at all, you can keep doing `forward -> step` on the same batch instead of returning to the scheduler and looping back around.

That second point is the **in-place re-loop** in #27877. Its rule is also restrained: if no block finishes this round, keep denoising on the current batch; the moment any block finishes, immediately yield to the scheduler so FDFO can release the finished request promptly. This cuts pointless scheduler round-trips while preserving FDFO's core semantics.

There's also a memory-paging detail here. The dLLM scheduler ensures `page_size` is a multiple of `block_size`, so a dLLM block is page-aligned. When the block is first created it's already a valid paged allocation; reusing that block's KV in place afterward is therefore safe under paging semantics.

## What the performance results tell us

On H200, TP=1, LLaDA2.0-mini, GSM8K (200 examples), #27551 reports the following LowConfidence numbers:

| max-running-requests | Synchronous | FDFO | Speedup |
| --- | --- | --- | --- |
| 4 | 525 tok/s | 680 tok/s | 1.30× |
| 16 | 918 – 959 tok/s | 1276 – 1378 tok/s | ~1.45× |

Accuracy stays in the same band across both modes. At batch size > 1, runs differ by roughly 0.01 in score on their own; the PR attributes this to kernel batch non-invariance, MoE routing, and similar runtime noise — and the FDFO and synchronous distributions overlap.

#27877 then runs a controlled A/B against #27551, optimizing the memory reuse of the FDFO mode itself:

| batch size | #27551 | #27877 | Gain |
| --- | --- | --- | --- |
| 4 | 614.9 tok/s | 660.4 tok/s | +7.4% |
| 16 | 1288 tok/s | 1305 tok/s | +1.3% |

This matches intuition. At small batch, scheduler round-trips and allocator churn make up a larger share, so KV/req-slot reuse pays off more; at large batch, forward compute dominates, so the optimization's effect is diluted by compute cost.

## Why correctness holds

FDFO appears to change batch scheduling order, so the most natural question is: does it change results?

Look at it in two layers.

**Layer one is batch size 1.** With no other requests in the batch, FDFO and synchronous mode see the same forward shape, and execution semantics should match. So **batch-size-1 parity** is a strong correctness check. In the PR, both LowConfidence and JointThreshold pass byte-identical verification — showing FDFO doesn't break single-request semantics, and that JointThreshold's cross-step state shuttling is correct.

**Layer two is batch size > 1.** GPU kernels, MoE routing, and batch-shape changes can introduce minor numerical differences across batch arrangements. So the more reasonable verification here isn't to demand byte-identical output, but to check whether accuracy distributions land in the same noise band, and whether there are systemic issues like OOM, KV leaks, or corrupted state. The GSM8K results and the no-leak verification in the PR cover this layer.

## What's most worth taking away

The value of this FDFO design isn't just the throughput gain — it's that the abstraction boundaries are clear enough.

1. **Token selection and execution mode are separated.** An algorithm class only implements `step()` and doesn't have to know whether it's running under synchronous or FDFO mode.
2. **Cross-round state has explicit carriers.** Generic work-in-progress tokens go in `dllm_incomplete_ids`; algorithm-private state goes in `dllm_algo_state`. The scheduler doesn't peek at algorithm state, and the algorithm doesn't manipulate the scheduler's internals.
3. **Completion is expressed as a simple per-request accept length.** To the scheduler, `0` means continue and `block_size` means commit. That makes the FDFO lifecycle very direct.
4. **The optimization is naturally layered.** #27551 gets the abstraction right first; #27877 then does HBM reuse and in-place re-loop on top of it. Get the semantics clean, then optimize the path — a very stable way to evolve a serving system.

## Acknowledgements

The framework-level FDFO design described here, and the subsequent in-place KV/req-slot reuse optimization, come from the work of Engine Architecture Group 5, Engine Infrastructure Department, Xiaohongshu (RedNote). It was contributed jointly by Huayi Jin, Zhaokai Luo, Junxiang Wu, Bing Zhang, Chenchen Hong, and Bing Tian. Thanks to the RedNote team and colleagues for designing and polishing FDFO from a special-case implementation inside one algorithm into a reusable framework capability for the dLLM scheduler.

## Summary

The original synchronous dLLM scheduling bound a batch into a single whole whose completion time was dominated by the slowest request. For dLLMs, where denoise step counts vary a lot, this causes obvious head-of-line blocking inside a batch.

FDFO's idea is simple: whoever's current block finishes first leaves first. It lets the scheduler commit results sooner, release resources sooner, and admit new requests sooner — raising throughput and cutting needless waiting.

Engineering-wise, the key to this feature isn't writing one more FDFO algorithm class, but abstracting FDFO into a framework-level execution mode. The `step()` mechanism keeps the algorithm strategy pure, `dllm_incomplete_ids` and `dllm_algo_state` give cross-round state a home, and `accept_length_per_req_cpu` lets the scheduler reason about completion uniformly. Add #27877's in-place KV/req-slot reuse, and FDFO is not only semantically clean but also starting to gain better system efficiency.

In one sentence: FDFO turns dLLM inference from "the whole batch waits for the slowest" into "release as soon as done" — and this engineering design turns that idea into a framework capability every dLLM algorithm can reuse.
