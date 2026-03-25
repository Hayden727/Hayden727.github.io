---
layout: project
title: "MLIR 编译优化框架"
title_en: "MLIR Compilation Framework for FT-Matrix"
date: 2026-01-15
status: "开发中"
status_en: "Active"
excerpt: "基于 MLIR 的 FT-Matrix 编译优化框架与基准测试平台"
excerpt_en: "A production MLIR compiler targeting FT-Matrix with cost-model-driven optimization, PyTorch frontend, and a comprehensive benchmark framework achieving ~57x kernel speedup"
---

# MLIR Compilation Framework for FT-Matrix

A production-grade compiler built on LLVM/MLIR that compiles PyTorch models and C/C++ code to FT-Matrix processors, paired with a systematic benchmark framework for evaluating optimization effectiveness.

## Architecture

The compiler follows a three-layer design:

- **Frontend**: Clang-based C/C++ compilation + PyTorch TorchDynamo capture with 350+ op lowerings and three compilation modes (MLIR inspection / assembly generation / JIT execution)
- **Midend**: 18-pass MLIR optimization pipeline with two custom dialects — FTM dialect (205 operations mapping hardware intrinsics) and CSL dialect (25 system-level operations for DMA, barrier sync, and core management)
- **Backend**: LLVM SelectionDAG instruction selection with post-increment addressing optimization and assembly generation

## Key Technical Contributions

- **Cost-model-driven tiling**: Three-phase framework — access pattern analysis, instruction latency cost evaluation, and optimal vectorization dimension selection
- **Multi-level memory tiling**: Two-level tiling for on-chip memory hierarchy with auto-inserted multi-channel DMA (double buffering, scatter-gather support)
- **Register blocking & accumulator promotion**: M-dimension blocking for matmul with accumulator reuse; automatic conversion of load/store patterns to register-carried values across reduction loops
- **Store-to-load forwarding**: Epilogue fusion that eliminates intermediate memory round-trips between producer-consumer operations
- **Elementwise fusion**: Fusing chains of elementwise operations for improved data locality
- **Conv2D im2col decomposition**: Transforms convolution into matrix multiplication for efficient vectorization
- **Padding optimization**: Border-only padding strategy achieving ~96% reduction compared to full-buffer initialization
- **Pure Python PyTorch frontend**: TorchDynamo integration with Linalg lowering, supporting model capture → MLIR → assembly end-to-end (demonstrated with Qwen3-0.6B LLM)

## Benchmark Framework

A systematic evaluation platform that measures compiler optimization effectiveness through:

- **Three-way comparison**: Scalar baseline vs. compiler-optimized vs. hand-written assembly, reporting speedup ratios and percentage of peak performance
- **Ablation studies**: 8 incremental optimization variants per kernel (scalar → pre-optimization → memory tiling → DMA insertion → vectorization → register blocking → loop unrolling → full pipeline), isolating each pass's contribution
- **End-to-end simulation**: Cycle-accurate simulation with automatic test data generation and NumPy reference verification for correctness

**Benchmarks include**:
- 5 BLAS kernels (GEMM, GEMV, dot product, axpby, scal) with hand-written reference implementations
- 5 additional kernels (softmax, layernorm, GELU, elementwise fusion, broadcast add)
- Model-level benchmarks (LeNet CNN, Transformer encoder)

## Performance Results

- **~57x speedup** on matrix kernels vs. scalar baseline
- **~44x speedup** on transformer inference vs. scalar baseline
- Compiler output achieves **~75% of hand-tuned assembly** performance
- End-to-end LLM compilation demonstrated (Qwen3-0.6B)

## Scale

- 25,000+ lines of code across frontend, midend, and backend
- 205+ custom MLIR operations + 25 system-level operations
- 146 compiler tests + comprehensive benchmark suite covering 12 kernels and 2 models
