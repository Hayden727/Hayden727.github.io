---
layout: post
title: "From PyTorch to MLIR: Building a TorchDynamo-Based Compiler Frontend"
date: 2025-03-15
categories: [compiler]
excerpt: "How we built a pure-Python frontend that captures PyTorch models via TorchDynamo and lowers them to MLIR for domain-specific compilation."
---

# From PyTorch to MLIR: Building a TorchDynamo-Based Compiler Frontend

Most ML compiler projects start with the backend — instruction selection, register allocation, code generation. But the frontend is equally critical: if you can't efficiently capture and lower PyTorch models, your compiler doesn't matter.

We built a pure-Python frontend that integrates with PyTorch's `torch.compile` infrastructure via TorchDynamo. Here's the architecture, the tricky parts, and what we learned.

## The Capture Problem

PyTorch models are imperative Python code. To compile them, you need to capture their computational graph. There are several approaches:

| Approach | Pros | Cons |
|----------|------|------|
| Tracing (torch.jit.trace) | Simple | Misses control flow |
| Scripting (torch.jit.script) | Handles control flow | Requires type annotations, limited Python |
| **TorchDynamo** (torch.compile) | **Handles real Python**, guards on shapes | More complex integration |
| Export (torch.export) | Clean graph guarantee | Stricter than Dynamo |

We chose TorchDynamo because it handles real-world PyTorch code — dynamic control flow, third-party library calls, and all the Python patterns that researchers actually use.

## Architecture Overview

```
PyTorch Model
    ↓
TorchDynamo (FX Graph capture)
    ↓
Graph IR (our intermediate representation)
    ↓
TOSA / Linalg MLIR generation
    ↓
Bufferization (tensor → memref)
    ↓
Midend Pipeline (subprocess)
    ↓
Assembly Output
```

### Three Compilation Modes

Our frontend exposes three modes through a single `FTMCompiler` class:

```python
compiler = FTMCompiler()

# Mode 1: Inspect MLIR (debugging)
mlir_str = compiler.compile_to_mlir(model, *sample_inputs)

# Mode 2: Compile to assembly
compiler.compile_to_lasm(model, *sample_inputs, output_path="kernel.lan")

# Mode 3: JIT execution on CPU (validation)
result = compiler.jit_run(model, *sample_inputs)
```

Mode 3 uses MLIR's `ExecutionEngine` to run on CPU — invaluable for validating correctness before running on target hardware.

## The Graph IR Layer

Between FX capture and MLIR generation, we maintain our own Graph IR. This might seem redundant, but it serves three purposes:

1. **Decouples FX changes from MLIR generation** — FX's internal representation evolves across PyTorch versions
2. **Carries tensor metadata** — dtype, shape, stride information propagated from PyTorch
3. **Enables pre-MLIR transformations** — op fusion, quantization handling, weight format conversion

Each node in our Graph IR holds:

```python
@dataclass
class Node:
    op_type: str          # "conv2d", "relu", "matmul", ...
    inputs: List[Edge]    # input edges with tensor metadata
    outputs: List[Edge]   # output edges
    attributes: Dict      # kernel_size, stride, padding, ...
    metadata: TensorMeta  # dtype, shape, strides
```

## Op Lowering: 350+ Operations

The largest component (3,400+ lines) is `operation.py` — the registry mapping PyTorch ops to MLIR dialect operations. We support two target dialects:

### TOSA Ops (Tensor Operator Set Architecture)
For standard neural network operations:
```python
# torch.nn.functional.relu → tosa.clamp
def lower_relu(node):
    return tosa.ClampOp(
        input=node.inputs[0],
        min_fp=0.0,
        max_fp=float('inf')
    )
```

### Linalg Ops (for compute-intensive kernels)
For operations that need tiling and vectorization:
```python
# torch.matmul → linalg.matmul
def lower_matmul(node):
    return linalg.MatmulOp(
        inputs=[node.inputs[0], node.inputs[1]],
        outputs=[init_tensor]
    )
```

The choice between TOSA and Linalg depends on the downstream optimization: TOSA ops have well-defined semantics but limited optimization opportunity. Linalg ops expose loop structure for our cost-model-driven tiling.

## Handling Conv2d: The Im2Col Approach

Convolution is the most complex lowering. Rather than emitting a monolithic `linalg.conv_2d`, we decompose convolutions using the im2col pattern:

```
Input [N, C, H, W]
  → im2col → Matrix [N*OH*OW, C*KH*KW]
  → matmul with Weight [C*KH*KW, OC]
  → reshape → Output [N, OC, OH, OW]
```

This decomposition is done in a dedicated midend pass (`ConvertConvToIm2Col`) rather than the frontend, because:

1. The frontend emits clean `linalg.conv_2d` that's easy to validate
2. The midend can choose whether to decompose based on kernel size and stride
3. Small convolutions (1x1) are better handled as direct matmul without im2col overhead

## The Pipeline Subprocess Pattern

After generating MLIR, the midend runs as a **subprocess**:

```python
class FTMPipeline:
    def run(self, mlir_path, output_path):
        # Step 1: MLIR optimization
        subprocess.run([
            "ftm-opt",
            "--ftm-pipeline",  # chains all 18 passes
            mlir_path,
            "-o", optimized_path
        ])

        # Step 2: MLIR → LLVM IR
        subprocess.run([
            "ftm-translate",
            "--mlir-to-llvmir",
            optimized_path,
            "-o", llvm_ir_path
        ])

        # Step 3: LLVM IR → Assembly
        subprocess.run([
            "llc",
            "-mtriple=matrix",
            llvm_ir_path,
            "-o", output_path
        ])
```

Why subprocess instead of in-process? Two reasons:

1. **Isolation**: A crash in the C++ compiler doesn't kill the Python process
2. **No C++ Python bindings needed for the midend**: The midend is a standalone CMake project. Embedding it in Python would require maintaining pybind11 wrappers for every pass

We do offer Python bindings (via upstream MLIR's Python API) for users who want tighter integration, but the subprocess path is the default.

## Integration with torch.compile

The cleanest integration point is `torch.compile`:

```python
from frontend import TorchCompileBackend

backend = TorchCompileBackend(output_dir="./compiled")

@torch.compile(backend=backend)
def my_model(x):
    return torch.matmul(x, weight) + bias

# First call compiles; subsequent calls reuse
result = my_model(input_tensor)
```

Under the hood, `TorchCompileBackend` wraps our `DynamoCompiler`, handles graph breaks (when Dynamo can't capture a subgraph), and caches compiled results.

## Lessons Learned

### 1. Start with TOSA, add Linalg incrementally
TOSA provides a clean, well-specified starting point. You can lower most ops to TOSA first, validate correctness, then selectively lower compute-intensive ops to Linalg for optimization.

### 2. The op registry is a maintenance burden
350+ ops means constant work tracking PyTorch API changes. We mitigate this with:
- Decomposition: complex ops are decomposed into primitives (e.g., `batch_norm` → multiply + add)
- AOT Autograd: let PyTorch decompose backward ops before we see them
- Exhaustive testing: every op has a round-trip test (PyTorch → MLIR → execute → compare)

### 3. Subprocess isolation is worth the overhead
The ~50ms subprocess startup cost is negligible compared to compilation time. And the first time the C++ compiler segfaults during development (it will), you'll be grateful your Python process survived.

### 4. Keep the Graph IR thin
Our initial Graph IR was too rich — it tried to encode optimization hints, target constraints, and scheduling decisions. This created coupling between frontend and midend. We stripped it back to pure data flow + tensor metadata. Let the midend make optimization decisions.

### 5. Validate at every boundary
- FX → Graph IR: check shapes match
- Graph IR → MLIR: round-trip through `mlir-opt --verify-each`
- MLIR → Assembly: run on CPU via ExecutionEngine, compare with PyTorch eager output

Three validation points catch bugs at the layer where they originate, rather than surfacing as mysterious assembly-level failures.

## What's Next

The frontier is **dynamic shapes**. TorchDynamo supports shape guards, but our midend pipeline assumes static shapes for tiling and DMA insertion. Bridging this gap — compiling shape-polymorphic kernels with specialization at runtime — is the next major challenge.
