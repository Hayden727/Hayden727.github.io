---
layout: post
title: "Building a Production MLIR Compiler: Architecture and Design Decisions"
date: 2025-03-01
categories: [compiler]
excerpt: "Lessons learned from building a full MLIR-based compilation pipeline — from dialect design to pass orchestration."
---

# Building a Production MLIR Compiler: Architecture and Design Decisions

After spending months building a production-grade MLIR-based compiler for a domain-specific architecture, I want to share the architectural decisions and patterns that shaped the project. This post focuses on the high-level design — how to structure an MLIR compiler that goes from PyTorch models and C/C++ code all the way down to assembly.

## The Three-Layer Architecture

Our compiler is structured as three distinct layers, each with clear responsibilities:

```
Frontend Layer (Clang + PyTorch)
    ↓
Midend Layer (MLIR Dialects + Passes)
    ↓
Backend Layer (LLVM CodeGen)
```

This separation is more than organizational — it enables independent development velocity. The midend can evolve without rebuilding LLVM. The frontend team can add new op coverage without touching optimization passes.

### Why Separate the Midend?

One critical decision was building the midend as a **standalone CMake project** that links against a pre-built LLVM/MLIR installation. The alternative — embedding everything into the LLVM tree — has a steep cost: every change requires rebuilding a large chunk of LLVM.

Our approach:

```bash
# Step 1: Build LLVM + MLIR + Clang (once, rarely)
cmake -G Ninja -S llvm/llvm -B llvm/build \
  -DLLVM_ENABLE_PROJECTS="clang;mlir"
ninja -C llvm/build

# Step 2: Build midend standalone (fast iteration)
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=$PWD/llvm/build/lib/cmake/mlir
ninja -C build
```

Midend rebuild takes seconds instead of minutes. This matters enormously during pass development.

## Dialect Design Philosophy

We defined two custom MLIR dialects, following fundamentally different design patterns:

### Pattern 1: LLVMIR Sub-Dialect (for compute ops)

Our primary compute dialect follows the NVVM/ROCDL model — it lives as a sub-dialect of the LLVM dialect. Each operation maps 1:1 to an LLVM intrinsic:

```
ftm.vload_w  →  llvm.ftm.vload.w
ftm.vfmadd   →  llvm.ftm.vfmadd
ftm.vstore_w →  llvm.ftm.vstore.w
```

This design has a key advantage: **lowering to LLVM IR is trivial**. Each op translates directly to an intrinsic call. No complex lowering logic needed.

We used TableGen helper classes to keep definitions consistent:

```tablegen
// Pure ops (no side effects)
class FTM_PureUnaryOp<...>  // 1 in, 1 out
class FTM_PureBinaryOp<...> // 2 in, 1 out

// Memory ops (with side effects)
class FTM_LoadOp<...>  // reads memory
class FTM_StoreOp<...> // writes memory
```

### Pattern 2: Standalone Dialect (for system ops)

System-level operations (DMA, synchronization, clock management) use a different pattern — a standalone dialect where each op lowers to an `llvm.call` targeting a pre-built library function:

```mlir
// Before lowering
csl.dma_start(%config, %src, %dst, %size)

// After CSL → LLVM lowering
llvm.call @csl_dma_start(%config, %src, %dst, %size)
```

This encapsulates hardware details behind a stable function interface. When the hardware library changes, only the library needs updating — the compiler passes remain stable.

### The 205-Op Challenge

Our compute dialect defines 205+ operations. Managing this at scale requires discipline:

1. **Consistent naming convention**: `v` prefix for vector, operation name, type suffix (`d` for f64, `s32` for f32)
2. **Category-based organization**: Memory ops, FP compare, precision conversion, dot products, etc.
3. **Exhaustive testing**: Every op has at least one FileCheck test validating the lowering path

## Pass Pipeline Orchestration

The midend chains 18 passes in a specific order. Getting this order right is the hardest part of compiler design. Our pipeline:

```
Pre-optimization (cleanup)
  → eliminate-memref-copy
  → optimize-padding
  → simplify-linalg-ops

Kernel decomposition
  → convert-conv-to-im2col
  → linalg-fusion

Memory hierarchy tiling (outer)
  → linalg-tiling (AM locality)
  → insert-dma

Vectorization (inner)
  → linalg-tiling (vector width)

Register optimization
  → simplify-vector
  → hoist-vector-transfers
  → unroll-loops
  → promote-accumulators
  → forward-store-to-load
  → merge-vector-loads

Dialect lowering
  → vector → ftm
  → math → ftm
  → arith → ftm
  → csl → llvm
```

**Key insight**: The order between memory hierarchy tiling and vectorization tiling is critical. You must tile for memory locality *first*, insert DMA operations, and *then* tile for vectorization within each memory-local tile. Reversing this order produces incorrect DMA boundaries.

## Build System Integration

We use LLVM as a git submodule with a **patch-based integration** strategy:

1. Patches stored as unified diffs in `/patches/`
2. Applied automatically via `apply-patches.sh`
3. Can be regenerated from working tree changes

This avoids maintaining a separate LLVM fork while keeping our changes version-controlled. When upstream LLVM updates, we rebase patches — much lighter than a full fork merge.

## Testing Strategy

With 18 passes and 205+ ops, testing is non-negotiable:

- **Per-pass directories**: Each pass has its own test directory with `implemented/` and `unimplemented/` subdirectories
- **FileCheck + lit**: Standard MLIR testing infrastructure
- **Kernel integration tests**: Full end-to-end compilation of BLAS kernels, FFT, etc.
- **141 tests total**: 48 backend + 93 midend

The `unimplemented/` directories with `XFAIL` markers serve as living documentation of known limitations — when you fix something, you move the test and change the expectation.

## Takeaways

1. **Separate your midend build** — fast iteration on passes is worth the initial setup cost
2. **Choose dialect patterns deliberately** — LLVMIR sub-dialect for compute, standalone for system ops
3. **Pass ordering is the hard problem** — document why each pass precedes the next
4. **Patch-based LLVM integration** beats fork maintenance for smaller teams
5. **Test per-pass, not just end-to-end** — localized failures are infinitely easier to debug

In the next post, I'll dive into the cost-model-driven tiling framework — how we automatically decide which dimensions to vectorize based on memory access patterns and hardware instruction costs.
