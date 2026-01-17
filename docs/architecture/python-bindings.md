# Python Bindings (xlog-gpu-py)

This document describes the Python bindings for XLOG, implemented using PyO3 and exposing GPU tensors via DLPack for zero-copy interoperability.

## Overview

The `xlog_gpu` Python module provides:

- Deterministic Datalog execution via `LogicProgram`
- Probabilistic inference via `Program`
- Zero-copy GPU tensor exchange via DLPack

## Installation

```bash
cd crates/xlog-gpu-py
pip install maturin
maturin develop --release
```

## Package Details

| Attribute | Value |
|-----------|-------|
| Package name | `xlog-gpu` (PyPI: `xlog_gpu`) |
| Build system | PyO3 + maturin |
| Platform | Linux x86_64 + CUDA only |
| Interop | DLPack capsules (framework-agnostic) |

## API Reference

### LogicProgram (Deterministic)

```python
import xlog_gpu

# Compile a deterministic program
program = xlog_gpu.LogicProgram.compile("""
    pred edge(u32, u32).
    pred reach(u32, u32).

    edge(1, 2). edge(2, 3). edge(3, 4).

    reach(X, Y) :- edge(X, Y).
    reach(X, Z) :- reach(X, Y), edge(Y, Z).

    ?- reach(1, N).
""")

# Execute and get results
results = program.evaluate()

# Results are DLPack capsules
for name, capsule in results.items():
    import torch
    tensor = torch.from_dlpack(capsule)
    print(f"{name}: {tensor}")
```

### Program (Probabilistic)

```python
import xlog_gpu

# Compile with exact inference
program = xlog_gpu.Program.compile("""
    0.3::rain.
    0.7::sprinkler.

    wet :- rain.
    wet :- sprinkler.

    evidence(sprinkler, false).
    query(wet).
""", prob_engine="exact_ddnnf")

# Evaluate
result = program.evaluate()
print(f"P(wet | not sprinkler) = {result.prob}")

# With gradients (for learning)
result = program.evaluate(return_grads=True)
print(f"Gradients: {result.gradients}")
```

### Monte Carlo Inference

```python
program = xlog_gpu.Program.compile(source, prob_engine="mc")

result = program.evaluate(
    samples=10000,
    seed=42,
    confidence=0.95
)

print(f"P(query) = {result.prob} ± {result.stderr}")
print(f"95% CI: [{result.ci_low}, {result.ci_high}]")
```

## DLPack Integration

All GPU data is exchanged via DLPack capsules, enabling zero-copy interop with:

- PyTorch
- CuPy
- JAX
- TensorFlow
- Any DLPack-compatible library

### Input via DLPack

```python
import torch
import xlog_gpu

# Create GPU tensor
edge_tensor = torch.tensor([[1, 2], [2, 3], [3, 4]], device='cuda')

# Pass as input
program = xlog_gpu.LogicProgram.compile(source)
results = program.evaluate(dlpack_inputs={
    'edge': edge_tensor.__dlpack__()
})
```

### Output via DLPack

```python
results = program.evaluate()

# Convert to PyTorch
import torch
for name, capsule in results.items():
    tensor = torch.from_dlpack(capsule)

# Convert to CuPy
import cupy
for name, capsule in results.items():
    array = cupy.from_dlpack(capsule)
```

### dlpack_roundtrip Helper

For testing interop:

```python
# Verify DLPack roundtrip works
capsule = results['reach']
tensor = torch.from_dlpack(capsule)
capsule2 = tensor.__dlpack__()
# capsule2 can be passed back to xlog
```

## Compile Options

### LogicProgram.compile()

```python
program = xlog_gpu.LogicProgram.compile(
    source,                    # str: Datalog source code
    device=0,                  # int: CUDA device index
    memory_mb=1024,           # int: GPU memory limit
)
```

### Program.compile() (Probabilistic)

```python
program = xlog_gpu.Program.compile(
    source,                    # str: Probabilistic Datalog source
    prob_engine="exact_ddnnf", # str: "exact_ddnnf" or "mc"
    device=0,                  # int: CUDA device index
    memory_mb=1024,           # int: GPU memory limit
)
```

## Result Objects

### Deterministic Results

```python
results = program.evaluate()
# dict[str, PyCapsule]: relation name → DLPack capsule
```

### Probabilistic Results

```python
result = program.evaluate()
result.prob          # float: query probability
result.log_prob      # float: log probability
result.gradients     # dict[str, float]: gradients w.r.t. weights (if return_grads=True)

# Monte Carlo only:
result.stderr        # float: standard error
result.ci_low        # float: confidence interval lower bound
result.ci_high       # float: confidence interval upper bound
result.samples       # int: number of samples used
result.seed          # int: random seed used
```

## Error Handling

Python exceptions are raised for errors:

```python
try:
    program = xlog_gpu.LogicProgram.compile(invalid_source)
except xlog_gpu.ParseError as e:
    print(f"Parse error: {e}")
except xlog_gpu.CompilationError as e:
    print(f"Compilation error: {e}")
except xlog_gpu.ExecutionError as e:
    print(f"Execution error: {e}")
except xlog_gpu.ResourceExhaustedError as e:
    print(f"OOM: {e}")
```

## Memory Management

- DLPack capsules own their GPU memory
- Memory is freed when the capsule is garbage collected
- Converting to PyTorch/CuPy shares memory (no copy)
- Explicit cleanup: `del capsule`

## Thread Safety

- `compile()` is thread-safe
- `evaluate()` is NOT thread-safe on the same program instance
- Use separate program instances for concurrent execution

## Examples

### Integration with PyTorch

```python
import torch
import xlog_gpu

# Define a neural-symbolic model
class XlogLayer(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.program = xlog_gpu.Program.compile(source, prob_engine="exact_ddnnf")
        self.weights = torch.nn.Parameter(torch.tensor([0.3, 0.7]))

    def forward(self, evidence):
        result = self.program.evaluate(
            weights={'rain': self.weights[0], 'sprinkler': self.weights[1]},
            evidence=evidence,
            return_grads=True
        )
        return result.prob
```

### Batch Processing

```python
# Process multiple inputs
for batch in data_loader:
    edge_tensor = batch['edges'].cuda()
    results = program.evaluate(dlpack_inputs={
        'edge': edge_tensor.__dlpack__()
    })
    # Process results...
```

## Limitations

Current limitations:
- Linux x86_64 + CUDA only
- No PyPI distribution (build from source)
- No async evaluation API
- No per-call memory limit configuration

## See Also

- [Data Interoperability](cudf-interop.md) — DLPack and Arrow details
- [Probabilistic Tier](xlog-prob.md) — Inference engine details
- [CLI Reference](cli-reference.md) — Command-line alternative
