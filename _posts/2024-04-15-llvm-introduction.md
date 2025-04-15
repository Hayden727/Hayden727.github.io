---
layout: post
title: "LLVM编译框架入门"
date: 2024-04-15
categories: [compiler]
subcategories: [llvm]
---

# LLVM编译框架入门

LLVM是一个模块化和可重用的编译器和工具链技术的集合。本文将介绍LLVM的基本架构和核心概念。

## LLVM的三段式设计

1. 前端（Frontend）
   - Clang：C/C++/Objective-C编译器
   - 其他语言前端

2. 优化器（Optimizer）
   - LLVM IR优化
   - 平台无关优化

3. 后端（Backend）
   - 代码生成
   - 目标平台相关优化

## LLVM IR

LLVM IR（中间表示）是LLVM的核心，它具有以下特点：

- 类似汇编的低级表示
- 强类型系统
- 显式的控制流图
- SSA（静态单赋值）形式

## 基本概念

- Module（模块）
- Function（函数）
- BasicBlock（基本块）
- Instruction（指令）

后续文章将深入探讨LLVM的各个组件和实际应用。 