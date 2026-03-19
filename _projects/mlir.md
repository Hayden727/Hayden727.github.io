---
layout: project
title: "MLIR 编译优化框架"
title_en: "MLIR Compilation Framework"
date: 2025-01-15
status: "开发中"
status_en: "Active"
excerpt: "基于 MLIR 的编译优化框架"
excerpt_en: "A production MLIR-based compiler with cost-model-driven tiling, multi-level memory management, and PyTorch frontend integration"
---

# MLIR Compilation Framework

A production-grade compiler built on LLVM/MLIR that compiles PyTorch models and C/C++ code to domain-specific hardware.

## Architecture

The compiler follows a three-layer design:

- **Frontend**: Clang-based C/C++ compilation + PyTorch TorchDynamo capture with 350+ op lowerings
- **Midend**: 18-pass MLIR optimization pipeline with cost-model-driven tiling, DMA insertion, and register optimization
- **Backend**: LLVM SelectionDAG instruction selection and assembly generation

## Key Technical Contributions

- **Cost-model-driven tiling**: Automatic vectorization dimension selection based on access pattern analysis and instruction latency costs
- **Multi-level memory tiling**: Two-level tiling for explicit memory hierarchy management with auto-inserted DMA operations
- **Accumulator promotion**: Automatic conversion of load/store patterns to register-carried values across reduction loops
- **Store-to-load forwarding**: Epilogue fusion that eliminates intermediate memory round-trips
- **Pure Python PyTorch frontend**: TorchDynamo integration with TOSA/Linalg lowering, no C++ dependencies

## Scale

- 25,000+ lines of code across frontend, midend, and backend
- 205+ custom MLIR dialect operations
- 141 tests covering backend instruction selection and midend pass validation
