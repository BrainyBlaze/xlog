# Differentiable ILP (dILP) Training Architecture

This document describes the dILP training subsystem: a GPU-accelerated differentiable Inductive Logic
Programming engine that learns Datalog rules from positive/negative examples via gradient descent.

## Design Goals

1. **Learn rules, not weights** — discover symbolic Datalog clauses (e.g., `reach(X,Y) :- edge(X,Z), edge(Z,Y).`) from data
2. **GPU-resident hot loop** — no semantic column downloads in the training step loop (P0 contract)
3. **Sparse by default** — candidate-indexed soft-probs instead of materializing N³ tensors
4. **Transactional promotion** — learned rules pass gate checks before entering the knowledge base

## Core Idea: Tensorized Super-Graph Masking

Traditional ILP systems compile candidate rules into executable programs — impossible at millisecond
timescales. XLOG's approach pre-compiles a "super-graph" of all candidate rules and activates them
via continuous mask tensors optimized with Gumbel-Softmax:

```
Candidate rules          Logit tensor W (C floats)
  ┌──────────┐              ┌───────────┐
  │ r1: A←B,C│              │ w1  w2 .. │ ─── Gumbel-Softmax(τ) ──►  soft mask
  │ r2: A←B,D│              │           │                               │
  │ ...       │              └───────────┘                               ▼
  └──────────┘                                              set_rule_mask_sparse()
                                                                        │
                                                                        ▼
                                                            xlog evaluate (GPU)
                                                                        │
                                                                        ▼
                                                              BCE loss + ∇W
```

At convergence, `argmax(W)` picks the winning rule. Temperature annealing (τ → τ_floor) drives the
soft mask toward a one-hot selection.

For the full mathematical treatment and resolved design decisions (RD-1 through RD-27), see [`rfc-tensorized-ilp.md`](rfc-tensorized-ilp.md).

## Architecture Overview

```
                           Python (pyxlog.ilp)
     ┌─────────────────────────────────────────────────────┐
     │  train_only()                                       │
     │    ├─ valid_candidates()  → candidate map           │
     │    ├─ MaskBackend.init_weights()                    │
     │    ├─ AdaptiveTempController                        │
     │    └─ step loop ──────────────────────────┐         │
     │         ├─ MaskBackend.apply_mask()       │         │
     │         ├─ program.evaluate_device()      │ GPU     │
     │         ├─ BCE loss (torch)               │ only    │
     │         ├─ loss.backward()                │         │
     │         └─ optimizer.step()               │         │
     │                                           │         │
     │  train_and_promote()                      │         │
     │    ├─ train_only()                        │         │
     │    ├─ trial compile (Rust)                │         │
     │    └─ promotion gates                     │         │
     └───────────────────────────────────────────┘         │
                        │                                   │
                        ▼                                   │
     ┌────────────────────────────────────┐                │
     │  Rust (xlog-runtime, xlog-cuda)   │                │
     │    ├─ set_rule_mask_sparse()       │◄───────────────┘
     │    ├─ IlpRegistry (mask storage)  │
     │    ├─ TensorMaskedJoin (executor) │
     │    ├─ batch_fact_membership()     │
     │    ├─ batch_fact_membership_device() │
     │    ├─ batch_tagged_credit()       │
     │    └─ batch_tagged_credit_device()│
     └────────────────────────────────────┘
                        │
                        ▼
     ┌────────────────────────────────────┐
     │  CUDA (kernels/ilp.cu)            │
     │    └─ extract_nonzero_indices()   │
     └────────────────────────────────────┘
```

## Key Entry Points

### Python (pyxlog.ilp)

| File | Purpose |
|------|---------|
| `pyxlog/ilp/trainer.py` | `train_only()` — multi-start training loop |
| `pyxlog/ilp/promoter.py` | `train_and_promote()` — training + gate pipeline |
| `pyxlog/ilp/backend.py` | `MaskBackend` protocol, `SparseMaskBackend`, `DenseMaskBackend` |
| `pyxlog/ilp/temperature.py` | `AdaptiveTempController` — cosine-annealed τ schedule |
| `pyxlog/ilp/entropy.py` | Entropy regularization helpers |
| `pyxlog/ilp/holdout.py` | `holdout_f1_and_variance()` — LOO (`<=20`) and k-fold (`>20`) F1 scoring |
| `pyxlog/ilp/types.py` | `TrainConfig`, `TrainResult`, `PromotionResult`, `LearnedArtifact`, etc. |
| `pyxlog/ilp/exceptions.py` | `IlpConfigError`, `IlpCandidateError`, `IlpTrainingError` |

### Rust (xlog-runtime, xlog-cuda)

| File | Purpose |
|------|---------|
| `crates/xlog-runtime/src/ilp_registry.rs` | `IlpRegistry` — mask storage, `IlpTaggedResult` metadata |
| `crates/xlog-runtime/tests/ilp_integration_tests.rs` | Rust-side integration tests for mask round-trips |
| `crates/xlog-cuda/tests/ilp_kernel_tests.rs` | CUDA kernel unit tests (extract_nonzero_indices) |

### CUDA Kernels

| File | Purpose |
|------|---------|
| `kernels/ilp.cu` | `extract_nonzero_indices()` — N³ mask → sparse index extraction |

## Mask Backends

The `MaskBackend` protocol abstracts how the learnable tensor W is applied to the XLOG executor:

### SparseMaskBackend (default)

```
W (C logits) → Gumbel-Softmax(τ) → candidate_soft_probs (C,)
                                          │
                    argsort/top-k on CUDA in Python/Torch
                                          │
                    set_rule_mask_sparse_selected(selected_ids, selected_soft_probs)
                                          │
                                          ▼
                              Rust builds executor mask internally
                              (no N³ tensor materialized, no full soft-vector DTOH)
```

- Learnable params: `C` floats (one per candidate rule)
- Memory: O(C) — typically C < 100
- Preferred hot-loop path calls `set_rule_mask_sparse_selected()` on the compiled program
- Legacy compatibility path `set_rule_mask_sparse()` remains available when Rust-side ranking is desired

### DenseMaskBackend (alpha-compatible, debug)

```
W (N×N×N logits) → Gumbel-Softmax(τ) → 3D soft mask
                                              │
                           flatten → DLPack → set_rule_mask()
                                              │
                                              ▼
                                    Rust imports N³ tensor
```

- Learnable params: N³ floats (N = schema size)
- Memory: O(N³) — expensive for large schemas
- Enabled via `TrainConfig(debug_dense_mask=True)` for parity testing

## Training Pipeline

### train_only()

1. **Candidate enumeration** — `valid_candidates(source, mask_name)` returns all syntactically legal body-pair assignments
2. **Multi-start** — up to `max_attempts` independent restarts with fresh logits
3. **Step loop** (per attempt, up to `step_budget_per_attempt`):
   - Apply mask via backend
   - Forward pass: `program.evaluate_device()` (GPU-only, no host reads)
   - BCE loss between predicted and target fact membership
   - Backward pass: `loss.backward()` through PyTorch autograd
   - Optimizer step on W
   - Temperature anneal: τ_start → τ_floor (cosine schedule)
   - Optional deterministic controls (`deterministic=True`) for reproducible attempt seeding
   - Early stopping: when argmax is stable and loss < threshold
4. **Decode** — `argmax(W)` maps to winning candidate → discovered rule string

### train_and_promote()

1. Call `train_only()` — get `TrainResult`
2. If not converged → `PromotionStatus.NOT_CONVERGED`
3. **Trial compile** — substitute discovered rule into source, compile via Rust
4. **Promotion gates** (all must pass for `PROMOTED`):
   - **Convergence gate** — training converged (already checked)
   - **Novel-rate gate** — fraction of non-example derivations ≤ `max_novel_rate`
   - **Protected-relation gate** — no unwanted relation side-effects
   - **Holdout F1 gate** — F1 on held-out examples ≥ threshold
   - **Ambiguity gate** — top-M scan (or exhaustive mode) detects no alternative winning candidates
   - **Typed-schema gate** — optional hard gate requiring relation type metadata (or waiver-driven manual review)
5. All pass → `PromotionStatus.PROMOTED` with `committed_source`

## Artifact Persistence

`LearnedArtifact` captures the full training result for reproducibility:

```python
artifact.save("artifact.json")   # JSON with SHA-256 candidate-map hash
loaded = LearnedArtifact.load("artifact.json", verify_hash=True)
```

Schema version: `beta-v1`. Fields: discovered rule, logits, candidate map, config, telemetry,
precision/recall, metadata (timestamp, schema version, candidate map hash).

## GPU Contract

The training step loop obeys XLOG's P0 GPU-resident contract:

- `evaluate_device()` — no host reads for semantic results
- `batch_fact_membership_device()` — returns a CUDA bool mask via DLPack with zero semantic-loop DTOH
- `batch_tagged_credit_device()` — returns CSR-style CUDA credit data via DLPack with zero semantic-loop DTOH
- `batch_fact_membership()` / `batch_tagged_credit()` remain available when host materialization is desired
- `AtomicU64` D2H counter on `CudaKernelProvider` — hard gate raises if `download_column_*` is observed during step loop
- `host_transfer_stats()` / `reset_host_transfer_stats()` expose broader host transfer accounting for profiling
- Legacy `set_rule_mask_sparse()` still performs a control-plane soft-probability download; the selected-candidate sparse path avoids it

## Testing

- **86+ static test functions** across ILP Python test files (expanded by parametrized GA/beta gates)
- **Reliability gate**: 20 consecutive `train_only()` runs must all converge (20/20 pass)
- **GA reliability gate**: default 50-seed statistical run (`test_ilp_ga_reliability.py`)
- **GA performance/transfer tests**: `forward_p95_us` + host transfer accounting (`test_ilp_performance.py`)
- **Dense/sparse parity**: every sparse-path test has a `debug_dense_mask=True` variant
- Rust-side: `ilp_integration_tests.rs`, `ilp_kernel_tests.rs`
- CUDA certification: `extract_nonzero_indices` covered by kernel test suite

## Design Documents

The dILP design went through ten internal iterations before converging on the RFC. Those numbered working documents were archived during the v0.5.0 cleanup; their resolved decisions (RD-1 through RD-27) are consolidated in the RFC. The live design references are:

| Document | Content |
|----------|---------|
| [`rfc-tensorized-ilp.md`](rfc-tensorized-ilp.md) | Full RFC: mathematical foundation (Gumbel-softmax, tensorized semi-naive), hardware rationale, implementation map, resolved design decisions |
| [`dilp-showcase-report.md`](dilp-showcase-report.md) | Validation: 4-stage showcase run analysis (reach/grandparent/colleague/plus2) |

## See Also

- [Python Bindings — ILP Training API](python-bindings.md#ilp-training-dilp-beta) — user-facing API reference
- [GPU Execution](gpu-execution.md) — mask DAG evaluation, stream compaction
- [Probabilistic Tier](xlog-prob.md) — XGCF circuits, provenance (shared infrastructure)
- [Data Interoperability](cudf-interop.md) — DLPack details
