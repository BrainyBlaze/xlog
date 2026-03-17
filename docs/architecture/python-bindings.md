# Python Bindings (pyxlog)

This document describes the Python bindings for XLOG, implemented using PyO3 and exposing GPU tensors via DLPack for zero-copy interoperability.

## Overview

The `pyxlog` Python module provides:

- Deterministic Datalog execution via `LogicProgram`
- Probabilistic inference via `Program`
- Term embedding registration and lookup via `register_embedding` / `forward_embedding`
- Differentiable ILP training via `pyxlog.ilp` (rule learning from examples)
- Zero-copy GPU tensor exchange via DLPack (primary interop boundary)
- Optional experimental Arrow C Device interop (feature-gated)

Host-read convenience outputs (probabilities, gradients, confidence intervals) are behind a `host-io`
Cargo feature so GPU-native call sites can enforce a "no DTOH for results" contract.

## Installation

```bash
cd crates/pyxlog
pip install maturin
maturin develop --release
```

### Build Features

- `host-io`: enable host-read convenience APIs (e.g. `CompiledProgram.evaluate(...)`)
- `arrow-device-import`: enable experimental Arrow C Device export/import helpers

Example:

```bash
maturin develop --release -m crates/pyxlog/Cargo.toml --features host-io
maturin develop --release -m crates/pyxlog/Cargo.toml --features arrow-device-import
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
import torch

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
result = program.evaluate()

# Results are a list of query outputs (relations) with per-column DLPack tensors
for q in result.queries:
    print(q.relation_name, q.columns, q.num_rows, q.is_true)
    cols = [torch.from_dlpack(t) for t in q.tensors]
    print(cols)
```

#### Supplying Input Relations (DLPack)

`CompiledLogicProgram.evaluate(dlpack_inputs=...)` accepts a dict mapping relation name to a
sequence of DLPack columns.

```python
import pyxlog
import torch

program = pyxlog.LogicProgram.compile("""
    pred edge(u32, u32).
    pred reach(u32, u32).
    reach(X, Y) :- edge(X, Y).
    ?- reach(1, N).
""")

# Two 1D columns, not a 2D tensor.
edge_a = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
edge_b = torch.tensor([2, 3, 4], device="cuda", dtype=torch.int32)

result = program.evaluate(dlpack_inputs={"edge": [edge_a, edge_b]})
```

#### Persistent Named Relations (DLPack)

For repeated evaluation with long-lived GPU relations, create a persistent session instead of
re-supplying `dlpack_inputs` on every call.

```python
import pyxlog
import torch

program = pyxlog.LogicProgram.compile("""
    pred edge(i32, i32).
    pred reach(i32, i32).
    reach(X, Y) :- edge(X, Y).
    ?- reach(X, Y).
""")

session = program.session()

edge_a = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
edge_b = torch.tensor([2, 3, 4], device="cuda", dtype=torch.int32)

session.put_relation("edge", [edge_a, edge_b])   # register or replace
result = session.evaluate()                      # reuse stored relations
exported = session.export_relation("edge")       # DLPack columns

session.remove_relation("edge")
session.clear_relations()
```

The persistent session path is additive:

- `evaluate(dlpack_inputs=...)` remains the stateless one-shot API
- `session()` exposes a mutable named relation store with schema-checked DLPack import/export

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

```

#### Host Outputs (Requires `host-io`)

When built with `--features host-io`, you can call `CompiledProgram.evaluate(...)` to get host-derived
probability outputs as device tensors (DLPack):

```python
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

#### Monte Carlo Inference (Device-Only)

For GPU-native workflows, prefer `CompiledProgram.evaluate_device(...)` (no host reads for results).

```python
program = pyxlog.Program.compile(source, prob_engine="mc")

device_result = program.evaluate_device(
    samples=10000,
    seed=42,
    confidence=0.95,
)

import torch
query_counts = torch.from_dlpack(device_result.query_counts)       # int32 CUDA tensor, shape [num_queries]
evidence_count = torch.from_dlpack(device_result.evidence_count)   # int32 CUDA tensor, shape [1]
print(device_result.total_samples, device_result.seed, device_result.confidence)
```

#### Monte Carlo Inference (Host Outputs, Requires `host-io`)

When built with `--features host-io`, `CompiledProgram.evaluate(...)` computes probabilities and
confidence intervals and uploads them as device tensors (DLPack):

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

### Experimental Arrow C Device Interop (Feature `arrow-device-import`)

When built with `--features arrow-device-import`, `pyxlog` exposes:

- `pyxlog.export_arrow_device(...) -> PyCapsule` (name `arrow_device_array`)
- `pyxlog.import_arrow_device(...) -> (dlpack_tensors, names, num_rows)`

These helpers exist to bridge between DLPack columns and Arrow's C Device interface without host
copies. This is experimental and currently rejects nulls; import does not yet support bit-packed
`Bool`.

## Term Embeddings (v0.5.0)

The `register_embedding` / `forward_embedding` API enables explicit PyTorch-side embedding training
through the logic program. Embedding predicates use the label-free `nn/3` declaration form.

### Embedding Registration

```python
program = pyxlog.Program.compile("""
    nn(entity_embed, [X], E) :: embed(X, E).
""")

# Trainable nn.Embedding — autograd graph preserved
embedding = torch.nn.Embedding(100, 64).cuda()
program.register_embedding("entity_embed", embedding, trainable=True)

# Frozen torch.Tensor — detached at registration, no gradient flow
weights = torch.randn(100, 64).cuda()
program.register_embedding("entity_embed", weights, trainable=False)
```

### Forward Lookup

```python
# Returns [n, dim] tensor on same device as embedding
vectors = program.forward_embedding("entity_embed", [0, 5, 42])

# For trainable nn.Embedding: vectors.requires_grad == True
# For frozen torch.Tensor: vectors.requires_grad == False
```

### Cross-Registration Validation

- Embedding declarations (`nn/3`, no labels) reject `register_network()` — error directs to `register_embedding()`
- Classification declarations (`nn/4`, with labels) reject `register_embedding()` — error directs to `register_network()`
- Same network name as both embedding and classification → compile-time error

### Constraints

- `trainable=True` requires `nn.Embedding`; raw `torch.Tensor` with `trainable=True` raises `ValueError`
- Raw tensors with `requires_grad=True` are detached at registration (frozen contract enforced)
- Integer IDs only (symbol/string lookup keys deferred)
- Optimizer ownership is user-managed (P2b APIs do not cover embeddings)
- Inference through rules (dot/cosine evaluation, grounded query API) deferred to v0.5.1+

---

## ILP Training (dILP Beta)

The `pyxlog.ilp` subpackage provides differentiable ILP (Inductive Logic Programming) for learning
Datalog rules from examples via gradient descent.

### Training API

```python
from pyxlog.ilp import train_only, train_and_promote, TrainConfig, LearnedArtifact

# Define a learnable program
source = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5).
    learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
pos = [("reach", [1, 3]), ("reach", [2, 4])]
neg = [("reach", [1, 1])]

# Configure training
config = TrainConfig(
    step_budget_per_attempt=150,   # steps per attempt
    max_attempts=5,                # multi-start attempts
    tau_start=2.0,                 # initial temperature
    tau_floor=0.05,                # minimum temperature
    seed=42,                       # reproducibility
)

# Train only (no promotion gates)
result = train_only(source, "W", pos, neg, config)
assert result.converged
print(result.discovered_rule)      # e.g., "reach(X,Y) :- edge(X,Z), edge(Z,Y)."

# Train and promote (with gates)
config = TrainConfig(check_ambiguity=True, max_novel_rate=0.05)
promotion = train_and_promote(source, "W", pos, neg, config)
print(promotion.status)            # PromotionStatus.PROMOTED
```

### Artifact Persistence

```python
# Save learned artifact
result.artifact.save("artifact.json")

# Load with hash verification
loaded = LearnedArtifact.load("artifact.json", verify_hash=True)
print(loaded.discovered_rule)
print(loaded.logits)
```

### TrainConfig Fields

| Field | Default | Description |
|-------|---------|-------------|
| `step_budget_per_attempt` | 150 | Max gradient steps per attempt |
| `max_attempts` | 5 | Multi-start attempts |
| `tau_start` | 2.0 | Initial Gumbel-softmax temperature |
| `tau_floor` | 0.05 | Minimum temperature |
| `allow_recursive_candidates` | False | Enable body-references-head candidates |
| `check_ambiguity` | False | Run ambiguity scan on convergence |
| `max_novel_rate` | 0.0 | Max fraction of novel (non-example) derivations |
| `debug_dense_mask` | False | Force dense mask backend (for parity testing) |
| `seed` | None | Random seed for reproducibility |
| `device` | 0 | CUDA device index |
| `memory_mb` | 512 | GPU memory limit |

### Result Types

```python
# TrainResult
result.converged          # bool
result.discovered_rule    # str | None
result.attempt_count      # int
result.total_steps        # int
result.precision          # float
result.recall             # float
result.holdout_f1         # float | None
result.artifact           # LearnedArtifact

# PromotionResult
promotion.status          # PromotionStatus (PROMOTED, GATE_FAILED, etc.)
promotion.gates           # list[GateResult]
promotion.novel_count     # int | None
promotion.novel_rate      # float | None
promotion.committed_source # str | None
```

### Device Query APIs

For GPU-native ILP workflows, `CompiledIlpProgram` now exposes device-resident query helpers in
addition to the existing host-returning helpers:

```python
import torch

prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)

# Device membership: bool CUDA tensor, one row per queried fact.
mask = torch.from_dlpack(
    prog.batch_fact_membership_device("edge", [[1, 2], [9, 9], [2, 3]])
)
assert mask.device.type == "cuda"
assert mask.dtype == torch.bool

# Device tagged credit: CSR-style CUDA outputs.
credit = prog.batch_tagged_credit_device("reach", [[1, 3], [2, 4]])
row_offsets = torch.from_dlpack(credit.fact_row_offsets)   # int32 CUDA tensor
entry_indices = torch.from_dlpack(credit.entry_indices)    # int32 CUDA tensor
entry_i = torch.from_dlpack(credit.entry_i)                # int32 CUDA tensor
entry_j = torch.from_dlpack(credit.entry_j)                # int32 CUDA tensor
entry_k = torch.from_dlpack(credit.entry_k)                # int32 CUDA tensor
```

Contract notes:

- `batch_fact_membership()` and `batch_tagged_credit()` remain available for host-materialized Python outputs
- `batch_fact_membership_device()` returns a DLPack bool tensor on CUDA
- `batch_tagged_credit_device()` returns CSR-style device outputs:
  `fact_row_offsets`, `entry_indices`, `entry_i`, `entry_j`, `entry_k`
- The device query path avoids semantic-loop DTOH transfers; inspect
  `host_transfer_stats()` / `reset_host_transfer_stats()` when enforcing that contract in tests
- Unsigned metadata/count tensors are exported as DLPack `int32` for broad framework compatibility

### Sparse Mask APIs

`CompiledIlpProgram` exposes two sparse mask setters:

- `set_rule_mask_sparse(name, candidate_ids, soft_probs, budget, allow_recursive=False)`
  is the legacy compatibility path. Rust receives the full candidate soft-probability vector and
  ranks it internally.
- `set_rule_mask_sparse_selected(name, selected_candidate_ids, selected_soft_probs, allow_recursive=False)`
  is the preferred hot-loop path. Python/Torch performs ranking on CUDA, then Rust consumes only
  the selected subset and preserves that order as the sparse active-rule list.

The selected-candidate path is the one to prefer when enforcing zero provider-side DTOH during
mask setup.

### GPU-Native Contract

For Python consumers that need an auditable GPU-native ILP hot loop, the intended contract is:

- Zero provider-tracked semantic-loop DTOH:
  `set_rule_mask_sparse_selected(...)`,
  `batch_fact_membership_device(...)`,
  `batch_tagged_credit_device(...)`,
  and `compute_ilp_loss_grad_gpu(...)`
- Metadata/control-plane reads may still occur behind public runtime/provider helpers such as
  cached row-count access; these are not relation-column materializations
- Compatibility paths that are not suitable for a strict GPU-native hot loop:
  `set_rule_mask_sparse(...)`,
  `batch_fact_membership(...)`,
  `batch_tagged_credit(...)`,
  and any host-output API gated behind `host-io`
- Use `host_transfer_stats()` / `reset_host_transfer_stats()` to audit the provider-tracked
  transfer behavior of the chosen path

---

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

# Create GPU columns
edge_a = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
edge_b = torch.tensor([2, 3, 4], device="cuda", dtype=torch.int32)

# Pass as input (relation name -> sequence of columns)
program = pyxlog.LogicProgram.compile(source)
result = program.evaluate(dlpack_inputs={"edge": [edge_a, edge_b]})
```

### Output via DLPack

```python
result = program.evaluate()

# Convert to PyTorch
import torch
for q in result.queries:
    cols = [torch.from_dlpack(t) for t in q.tensors]

# Convert to CuPy
import cupy
for q in result.queries:
    cols = [cupy.from_dlpack(t) for t in q.tensors]
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
result = program.evaluate()

result.queries             # list[LogicQueryResult]
result.queries[0].tensors  # list[PyCapsule] (DLPack), one per column
result.queries[0].columns  # list[str]
result.queries[0].num_rows # int
result.queries[0].is_true  # bool
```

### Probabilistic Results

```python
result = program.evaluate()  # requires host-io
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

### Device-Only MC Results

```python
device_result = program.evaluate_device(...)
device_result.query_counts    # PyCapsule: DLPack int32 vector (len = num_queries)
device_result.evidence_count  # PyCapsule: DLPack int32 vector (len = 1)
device_result.total_samples   # int
device_result.seed            # int
device_result.confidence      # float
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

- [dILP Training Architecture](dilp-training.md) — System design, mask backends, promotion pipeline
- [Data Interoperability](cudf-interop.md) — DLPack and Arrow details
- [Probabilistic Tier](xlog-prob.md) — Inference engine details
- [CLI Reference](cli-reference.md) — Command-line alternative
