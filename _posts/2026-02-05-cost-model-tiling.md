---
layout: post
title: "Cost-Model-Driven Tiling in MLIR: Automating Vectorization Decisions"
date: 2026-02-05
categories: [compiler]
excerpt: "How to build a cost model that automatically selects optimal vectorization dimensions based on memory access patterns and instruction latencies."
---

# Cost-Model-Driven Tiling in MLIR: Automating Vectorization Decisions

One of the hardest problems in compiler optimization is tiling — deciding how to partition computation into chunks that fit hardware constraints. When you have vector registers, the question becomes: *which dimension should we vectorize?*

Manual tuning is fragile. Change the kernel shape, element type, or memory layout, and your hand-tuned tiling breaks. We built a cost-model-driven framework that automatically makes these decisions. Here's how it works.

## The Three-Phase Framework

Our tiling framework operates in three phases:

```
Phase 1: Access Pattern Analysis
  → Classify how each operand accesses each dimension

Phase 2: Cost Evaluation
  → Score each candidate dimension using instruction costs

Phase 3: Tiling & Vectorization
  → Apply the winning dimension as the vector axis
```

## Phase 1: Access Pattern Classification

For each operand and each dimension of a Linalg operation, we classify the memory access pattern into one of five categories:

| Pattern | Definition | Example |
|---------|-----------|---------|
| **Contiguous** | Maps to innermost memref dim (unit stride) | `A[i, j]` vectorize on `j` |
| **Strided** | Maps to non-innermost dim | `A[i, j]` vectorize on `i` |
| **Broadcast** | Dimension absent from indexing map | `bias[j]` on dim `i` |
| **Reduction** | Reduction iterator | `k` in matmul `C[i,j] += A[i,k] * B[k,j]` |
| **None** | Non-affine or complex | Conservative fallback |

The classification uses MLIR's affine map infrastructure. For a `linalg.generic` with indexing maps, we inspect which result dimensions each operand's map projects to:

```mlir
// matmul: C[i,j] += A[i,k] * B[k,j]
#map_a = affine_map<(i, j, k) -> (i, k)>  // A
#map_b = affine_map<(i, j, k) -> (k, j)>  // B
#map_c = affine_map<(i, j, k) -> (i, j)>  // C

// Vectorize on j:
//   A: dim j absent → Broadcast
//   B: dim j is innermost → Contiguous load
//   C: dim j is innermost → Contiguous load/store
```

## Phase 2: Cost Evaluation

Each access pattern maps to a hardware instruction with a known cycle cost. Per-dimension cost is the sum across all operands:

```
Contiguous load:   9 cycles  (vector load)
Contiguous store:  4 cycles  (vector store)
Strided load:      9 cycles  (strided vector load)
Strided store:     4 cycles  (strided vector store)
Broadcast:         4 cycles  (scalar broadcast to vector)
Gather:           32 cycles  (scalar loop fallback)
Reduction:         0 cycles  (orthogonal to vectorization)
```

For a matmul `C[i,j] += A[i,k] * B[k,j]`:

**Vectorize on `j`:**
- A: broadcast (4) + B: contiguous load (9) + C: contiguous load (9) + store (4) = **26 cycles**

**Vectorize on `i`:**
- A: contiguous load (9) + B: broadcast (4) + C: strided load (9) + store (4) = **26 cycles**

**Vectorize on `k` (reduction):**
- Skipped — reduction dimensions require horizontal reduction, handled separately.

Tie-break rule: prefer the last (highest-index) dimension, as it aligns with row-major memory layout conventions.

## Phase 3: Tiling

Once we've selected the optimal dimension, we tile:

1. **Selected dimension** → tile size = vector register width (16 for f64, 32 for f32, 64 for f16)
2. **All other parallel dimensions** → tile size = 1 (creates scalar loops via `scf.for`)
3. **Reduction dimensions** → untouched (remain as inner loops)

```mlir
// Before tiling (linalg.matmul on 64x64x64)
linalg.matmul ins(%A, %B) outs(%C)

// After tiling on j with vector width 32 (f32):
scf.for %i = 0 to 64 step 1 {
  scf.for %j = 0 to 64 step 32 {
    // Vectorized matmul tile: 1x32xK
    linalg.matmul ins(%A_tile, %B_tile) outs(%C_tile)
  }
}
```

### Handling Irregular Sizes

When tensor dimensions aren't multiples of the vector width, we use **loop peeling**:

```mlir
// Dimension = 100, vector width = 32
scf.for %j = 0 to 96 step 32 {
  // Main loop: full vector operations
}
scf.for %j = 96 to 100 step 1 {
  // Remainder: scalar operations
}
```

We chose peeling over masked loads/stores because our target hardware doesn't support predicated vector memory operations. This is a hardware-specific tradeoff — if your target supports masking, you can avoid the remainder loop entirely.

## Multi-Level Tiling for Memory Hierarchy

Real hardware has memory hierarchies. Our framework supports **two-level tiling**:

1. **Outer level**: Tile all dimensions to fit the local memory capacity (e.g., 768KB scratchpad)
2. **Inner level**: Vectorization tiling within each memory-local tile

```
Outer tile: fit working set into local memory
  → Insert DMA to move data from main memory to scratchpad
  → Inner tile: vectorize within scratchpad-resident data
```

The outer tile sizes are computed from the memory capacity constraint:

```
element_size × Π(tile_sizes) × num_operands ≤ memory_capacity
```

We solve this by iteratively halving the largest tile dimension until the constraint is satisfied.

## Register Blocking

For compute-intensive kernels like matmul, we add a third tiling level — **register blocking**. This tiles one dimension (typically M in `C[M,N] += A[M,K] * B[K,N]`) to create multiple independent accumulator chains:

```mlir
// Without register blocking: 1 accumulator
scf.for %k = 0 to K {
  %c = fma(%a, %b, %c)  // single dependency chain
}

// With register blocking (M_block=4): 4 accumulators
scf.for %k = 0 to K {
  %c0 = fma(%a0, %b, %c0)  // 4 independent chains
  %c1 = fma(%a1, %b, %c1)  // → better pipeline utilization
  %c2 = fma(%a2, %b, %c2)
  %c3 = fma(%a3, %b, %c3)
}
```

The block factor is bounded by the number of available vector registers (typically 6-8 accumulators for a 32-register file, leaving room for operands and temporaries).

## Post-Tiling Optimizations

After tiling, several cleanup passes are essential:

1. **Hoist loop-invariant loads**: Vector loads that don't depend on the loop IV are hoisted out
2. **Accumulator promotion**: Load/store pairs around reduction loops become `scf.for` iter_args
3. **Store-to-load forwarding**: When a store is immediately followed by a load of the same address, forward the value directly
4. **Adjacent load merging**: Two consecutive vector loads to adjacent addresses become a single double-width load

These are individually simple transforms, but together they can eliminate 40-60% of memory traffic in compute-intensive kernels.

## Results

The cost model consistently selects the same dimensions that expert programmers would choose manually, while handling edge cases (irregular sizes, non-standard layouts) that manual tuning often misses.

More importantly, when kernel shapes change — which happens constantly during model development — the cost model adapts automatically. No human intervention needed.

## Takeaways

1. **Classify access patterns first** — the rest follows from knowing contiguous vs. strided vs. broadcast
2. **Use real instruction costs** — abstract "fast/slow" labels aren't precise enough for cost models
3. **Multi-level tiling is essential** for hardware with explicit memory hierarchies
4. **Post-tiling cleanup passes are as important as the tiling itself** — without accumulator promotion and store forwarding, you leave significant performance on the table
5. **Automate what humans forget** — the cost model handles irregular sizes, unusual layouts, and new kernel shapes without manual tuning

Next post: how we built a PyTorch frontend that captures models via TorchDynamo and lowers them through our MLIR pipeline.
