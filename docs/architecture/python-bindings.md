# Python Bindings (pyxlog)

This document describes the Python bindings for XLOG, implemented using PyO3 and exposing GPU tensors via DLPack for zero-copy interoperability.

## Overview

The `pyxlog` Python module provides:

- Deterministic Datalog execution via `LogicProgram`
- Probabilistic inference via `Program`
- Zero-copy GPU tensor exchange via DLPack

## Installation

```bash
cd crates/pyxlog
pip install maturin
maturin develop --release
```

## Package Details

| Attribute | Value |
|-----------|-------|
| Package name | `pyxlog` |
| Build system | PyO3 + maturin |
| Platform | Linux x86_64 + CUDA only |
| Interop | DLPack capsules (framework-agnostic) |

## API Reference

### LogicProgram (Deterministic)

```python
import pyxlog

# Compile a deterministic program
program = pyxlog.LogicProgram.compile("""
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
import pyxlog

# Compile with exact inference
program = pyxlog.Program.compile("""
    0.3::rain.
    0.7::sprinkler.

    wet :- rain.
    wet :- sprinkler.

    evidence(sprinkler, false).
    query(wet).
""", prob_engine="exact_ddnnf")

# Evaluate (probabilities are returned as DLPack capsules)
result = program.evaluate()
import torch
prob = torch.from_dlpack(result.prob)       # f64 CUDA tensor, shape [num_queries]
log_prob = torch.from_dlpack(result.log_prob)
print(list(zip(result.atoms, prob.tolist())))  # host read for printing

# If you need a single host scalar (e.g., for logging), read it explicitly:
p0 = float(prob[0].item())  # host read
print(f"P(wet | not sprinkler) = {p0}")

# With gradients (exact engine only; per-query grad vectors are DLPack too)
result = program.evaluate(return_grads=True)
grad_true0 = torch.from_dlpack(result.grad_true[0])   # f64 CUDA tensor, shape [num_vars]
grad_false0 = torch.from_dlpack(result.grad_false[0]) # f64 CUDA tensor, shape [num_vars]
```

### Monte Carlo Inference

```python
program = pyxlog.Program.compile(source, prob_engine="mc")

result = program.evaluate(
    samples=10000,
    seed=42,
    confidence=0.95
)

import torch
prob = torch.from_dlpack(result.prob)
stderr = torch.from_dlpack(result.stderr)
ci_low = torch.from_dlpack(result.ci_low)
ci_high = torch.from_dlpack(result.ci_high)
print(f"P(query) = {float(prob[0].item())} ± {float(stderr[0].item())}")  # host reads
print(f"95% CI: [{float(ci_low[0].item())}, {float(ci_high[0].item())}]") # host reads
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
import pyxlog

# Create GPU tensor
edge_tensor = torch.tensor([[1, 2], [2, 3], [3, 4]], device='cuda')

# Pass as input
program = pyxlog.LogicProgram.compile(source)
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
program = pyxlog.LogicProgram.compile(
    source,                    # str: Datalog source code
    device=0,                  # int: CUDA device index
    memory_mb=1024,           # int: GPU memory limit
)
```

### Program.compile() (Probabilistic)

```python
program = pyxlog.Program.compile(
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
result.atoms         # list[str]: query atoms (stringified)
result.prob          # PyCapsule: DLPack f64 vector of probabilities (len = num_queries)
result.log_prob      # PyCapsule: DLPack f64 vector of log-probabilities (len = num_queries)
result.num_vars      # int: number of CNF variables in the compiled program

# Exact-only (when return_grads=True):
result.grad_true     # Optional[list[PyCapsule]]: per-query DLPack f64 vector (len = num_vars)
result.grad_false    # Optional[list[PyCapsule]]: per-query DLPack f64 vector (len = num_vars)

# Monte Carlo only:
result.stderr        # Optional[PyCapsule]: DLPack f64 vector (len = num_queries)
result.ci_low        # Optional[PyCapsule]: DLPack f64 vector (len = num_queries)
result.ci_high       # Optional[PyCapsule]: DLPack f64 vector (len = num_queries)
result.samples       # Optional[int]
result.evidence_samples # Optional[int]
result.seed          # Optional[int]
result.confidence    # Optional[float]
```

## Error Handling

Python exceptions are raised for errors:

```python
try:
    program = pyxlog.LogicProgram.compile(invalid_source)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"XLOG error: {e}")
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
import pyxlog

# Neural-symbolic training loop (v0.4.0-alpha):
# - neural predicate outputs (CUDA tensors) are imported via DLPack
# - XLOG computes NLL gradients on GPU and calls output.backward(grad) internally

source = """
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
"""
program = pyxlog.Program.compile(source, prob_engine="exact_ddnnf")

net = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(28 * 28, 10),
    torch.nn.Softmax(dim=-1),
).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
program.register_network("mnist_net", net, optimizer)

images = torch.randn(128, 1, 28, 28, device="cuda")
program.add_tensor_source("train", images)

program.zero_grad()
loss = program.forward_backward_tensor("addition(0, 1, 7)")  # CUDA scalar tensor (no host reads required)
program.optimizer_step()

# Optional host read for logging:
print(float(loss.item()))
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
