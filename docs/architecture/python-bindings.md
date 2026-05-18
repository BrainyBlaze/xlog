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

For the latest published release:

```bash
pip install pyxlog
```

On import, `pyxlog` checks for bundled CUDA kernel artifacts under
`pyxlog/kernels/` and, when present, exports that directory to
`XLOG_CUBIN_DIR` automatically. Any pilot script, probe harness, or artifact
replay that runs outside the packaged wheel layout should set
`XLOG_CUBIN_DIR` explicitly before importing `pyxlog`, for example:

```bash
export XLOG_CUBIN_DIR=/home/dev/projects/xlog/crates/pyxlog/python/pyxlog/kernels
python your_probe.py
```

This is especially important for `pipeline_run`-style execution on saved
artifacts: cold starts without `XLOG_CUBIN_DIR` can fail if the active install
does not contain `pyxlog/kernels/`.

For unreleased `main` branch features or local development:

```bash
python scripts/install_pyxlog_for_python.py --python /usr/local/bin/python --user
```

Use the Python executable from the downstream project, not necessarily the
Python from the xlog checkout. The helper stages generated CUDA artifacts,
builds a wheel for that interpreter with `maturin build -i`, installs the wheel
with the same interpreter's `pip`, and verifies that the installed package has
`pyxlog/kernels/`. Generated `.ptx` and `.cubin` files remain build artifacts
and are not tracked in git.

### Build Features

- `host-io`: enable host-read convenience APIs (e.g. `CompiledProgram.evaluate(...)`)
- `arrow-device-import`: enable experimental Arrow C Device export/import helpers

Example:

```bash
python scripts/install_pyxlog_for_python.py --python /usr/local/bin/python
python scripts/install_pyxlog_for_python.py --python /usr/local/bin/python \
  --features extension-module,host-io,arrow-device-import
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

#### Persistent Relation Deltas

Persistent sessions also support DLPack-backed relation deltas for DTS-DLM
Stage-4 update loops. `insert_relation(...)`, `delete_relation(...)`, and
`apply_relation_delta(...)` update the session relation store through the
runtime `RelationDelta` / `apply_deltas_and_recompute` path. Insert-only
monotone SCCs keep prior materialized output where the execution plan permits
it; delete-containing deltas clear and recompute affected SCCs for correctness.

```python
session.put_relation("wmir_committed", [row_id, parent_id])
session.evaluate()

delta = session.insert_relation("wmir_committed", [new_row_id, new_parent_id])
result = session.evaluate()          # returns the delta-updated cached store
print(session.delta_stats(), delta)

session.apply_relation_delta(
    "wmir_committed",
    insert_columns=[added_row_id, added_parent_id],
    delete_columns=[removed_row_id, removed_parent_id],
)
```

The delta stats dictionary contains `changed_relations`, `insert_rows`,
`delete_rows`, `affected_sccs`, `recomputed_sccs`, and `incremental_sccs`.
Direct `put_relation`, `remove_relation`, or `clear_relations` calls invalidate
the cached runtime store and make the next `evaluate()` perform a full plan
run before later deltas can reuse it.

#### v0.8.0 Runtime Controls And Diagnostics

Long-running DTS-DLM callers can submit logic or probabilistic evaluations to a
background Python worker with `evaluate_async(...)`. The returned
`AsyncEvaluation` is awaitable and also exposes `done()`, `cancel()`,
`exception()`, and `result(timeout=None)` for synchronous orchestration.

```python
handle = session.evaluate_async(memory_mb=512)
result = handle.result(timeout=30)
```

Large logic outputs can be consumed as DLPack-compatible CUDA tensor chunks:

```python
for chunk in session.evaluate_stream(memory_mb=512, chunk_rows=1024):
    cols = chunk.tensors  # torch CUDA tensor views, DLPack-compatible
    print(chunk.relation_name, chunk.offset, chunk.num_rows, cols)
```

The same chunking is available from an already materialized result:

```python
result = session.evaluate()
for chunk in result.iter_query_chunks(chunk_rows=1024):
    ...
```

Per-call `memory_mb` is accepted by `CompiledLogicProgram.evaluate`,
`LogicRelationSession.evaluate`, `CompiledProgram.evaluate`, and
`CompiledProgram.evaluate_device`. A zero limit raises `ValueError`; a limit
below the provider's current tracked allocation raises `MemoryError` before the
evaluation starts. The provider-level compile-time budget remains the hard GPU
allocator budget.

Runtime progress and diagnostics are exposed as stable dictionaries:

```python
session.progress_stats()
session.memory_stats()
session.host_transfer_stats()
session.cuda_graph_stats()

program.progress_stats()
program.memory_stats()
program.host_transfer_stats()
program.cuda_graph_stats()
```

`memory_stats()` reports `allocated_bytes`, `memory_limit_bytes`,
`peak_memory_bytes`, and `status`. CUDA Graph stats report
`csm_cuda_graph_captures`, `csm_cuda_graph_launches`,
`csm_cuda_graph_fallbacks`, and `csm_cuda_graph_cache_hits`. Environments that
cannot provide a future diagnostic must report an explicit unavailable status or
error rather than fabricating a zero-valued probe.

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

## Training Loop API (Neural-Symbolic)

For neural-symbolic training with `nn/k` predicates, `Program` exposes loss
computation, optimizer stepping, gradient clipping, learning-rate control, and
batched training loops in addition to the single-query
`forward_backward*` helpers.

### Loss computation

```python
loss = program.nll_loss("addition(0, 1, 7)")
loss = program.nll_loss_batch(queries)
loss = program.nll_loss_mean(queries)

loss_t = program.nll_loss_tensor("addition(0, 1, 7)")
batch_t = program.nll_loss_batch_tensor(queries)
avg_loss = program.evaluate_loss(queries)
```

### v0.8.0 DTS-DLM Bridge Helpers

M37-A+B bridge training keeps Belnap pro/contra/quarantine semantics in the
Python/ML layer. Stage-4 structural kernels remain oblivious to those channels.
The helper surfaces operate on PyTorch tensors and preserve autograd unless the
caller explicitly detaches inputs.

```python
top = program.deterministic_topk(scores, k=4)
stats = program.neural_cache_stats()

terms = program.belnap_loss(
    pro=pro_scores,
    contra=contra_scores,
    quarantine=quarantine_scores,
    pro_reward=1.0,
    contra_penalty=2.0,
    quarantine_penalty=0.5,
)

semantic = program.semantic_loss_tensor(violations, weight=1.5)
mse = program.mse_loss_tensor(pred, target)
info = program.infoloss_tensor(prob)
```

`deterministic_topk(...)` resolves ties by lower input index. `neural_cache_stats()`
reports circuit-cache size, hit/miss counters, template compile count,
query-signature cache size, and registered-network cache/top-k/deterministic
configuration. `belnap_loss(...)` returns a dictionary containing `loss`,
`pro_reward`, `contra_penalty`, `quarantine_penalty`, `cfr_regret_proxy`, and
the formula string.

### Optimizer and scheduler control

```python
program.zero_grad()
program.optimizer_step()
program.clip_grad_norms(max_norm=1.0)

program.scheduler_step()
program.scheduler_step(network_name="mnist_net")

lr = program.get_lr("mnist_net")
program.set_lr("mnist_net", 1e-4)
```

### Batched training epoch

```python
stats = program.train_epoch(queries, batch_size=32, max_grad_norm=1.0)
stats = program.train_epoch_tensor(queries, batch_size=32, max_grad_norm=1.0)
```

### Profiling

```python
profile = program.warmup_breakdown()
```

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
    memory_mb=32768,          # int: GPU memory limit in megabytes
)
```

### Program.compile() (Probabilistic)

```python
program = pyxlog.Program.compile(
    source,                    # str: Probabilistic Datalog source
    prob_engine="exact_ddnnf", # str: "exact_ddnnf" or "mc"
    device=0,                  # int: CUDA device index
    memory_mb=32768,          # int: GPU memory limit in megabytes
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
- Published PyPI wheels follow tagged releases and may lag the current `main` branch workspace version
- No async evaluation API
- No per-call memory limit configuration

## See Also

- [dILP Training Architecture](dilp-training.md) — System design, mask backends, promotion pipeline
- [Data Interoperability](cudf-interop.md) — DLPack and Arrow details
- [Probabilistic Tier](xlog-prob.md) — Inference engine details
- [CLI Reference](cli-reference.md) — Command-line alternative
