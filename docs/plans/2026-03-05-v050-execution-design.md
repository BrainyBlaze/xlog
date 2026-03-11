# v0.5.0 Execution Plan — Design

**Date:** 2026-03-05
**Status:** Approved
**Depends on:** v0.4.0-ga (tagged 2026-03-04, commit 48027c18)

## Goal

Close the remaining D2H transfer gap in the ILP compute path, harden
artifact/telemetry persistence, add term embeddings and training controls,
and introduce incremental verifier reuse — with release gates scoped to
`compute_ilp_loss_grad_gpu`.

## Architecture Summary

Five priority tiers:

| Tier | Scope | Depends on |
|------|-------|------------|
| P0 | Bounded-memory zero-D2H chunk merge | Nothing |
| P1 | Artifact schema migration + telemetry persistence | Nothing |
| P2a | Term embeddings | P0 (file sequencing in lib.rs) |
| P2b | Extended training controls | P0 (file sequencing in lib.rs) |
| P3 | Incremental verifier interface | Nothing |
| P4 | Release gates | P0–P3 |

Sequencing: P0 first (modifies chunked path in `lib.rs`), then P2a and P2b
(touch different functions in `lib.rs`). P1 and P3 are file-independent and
can run in parallel with everything.

```
Week 1-2: P0 (chunk merge) || P1 (artifact schema) || P3 (verifier)
Week 3-4: P2a (embeddings) || P2b (training controls)
Week 5:   P4 gates, integration testing, release
```

---

## P0: Bounded-Memory Zero-D2H Chunk Merge

### Problem

The chunked COO fallback path (`lib.rs:4353–4514`) does D2H downloads of
`d_coo_facts`/`d_coo_cands` per chunk, host-side sentinel filtering, then
H2D re-upload. This is the only D2H in `compute_ilp_loss_grad_gpu`.

### Semantic change: `coo_memory_cap` → `coo_chunk_budget`

Current code treats `coo_memory_cap` as both the chunking trigger and an
allocation ceiling (`lib.rs:4236`). The new design redefines it:

- **`coo_chunk_budget`**: Maximum bytes for per-chunk temp allocations
  (masks, prefix sums, chunk-local COO scratch). Default 16 MB.
- The **final merged COO** buffer is exact-NNZ-sized and may exceed the
  chunk budget. This is safe because actual NNZ << upper_bound (masks are
  sparse, typically <1% dense).
- `set_coo_memory_cap()` remains as a deprecated compatibility alias for
  one release cycle, forwarding to `set_coo_chunk_budget()`.

### Algorithm: Two-pass bounded-memory streaming

**Pass 1 — Count NNZ per task (GPU-only, bounded temp):**

For each chunk of tasks fitting within `coo_chunk_budget / 8` queries:
1. For each task in chunk: run `count_mask_into_slot(mask, num_query,
   task_counts, task_idx)` — writes count directly into
   `task_counts[task_idx]` on device.
2. Drop chunk-local mask/prefix buffers (bounded by chunk budget).

After all chunks: `task_counts[num_tasks]` is fully populated on device.

Compute per-task write offsets:
- `exclusive_scan_u32_inplace(task_counts) → task_offsets[num_tasks]`
  (already exists)
- Read `total_nnz` = `task_offsets[last] + task_counts[last]` via
  `dtoh_scalar_untracked` (8-byte metadata read, not data-plane).

**Allocation decision:**

Allocate `d_global_facts[total_nnz]` + `d_global_cands[total_nnz]`.
This allocation is exact-sized and may exceed `coo_chunk_budget`.

**Pass 2 — Fill COO (GPU-only, bounded temp):**

Re-iterate the same chunks:
1. For each task: compute prefix scan (`scan_u8_mask_device`, bounded temp).
2. Call `ilp_coo_fill_from_mask` with `offset_idx = task_idx` into the
   global `task_offsets` device array. Writes directly into the global COO
   buffer at the correct task-level offset.
3. Drop chunk-local prefix buffers.

After all chunks: global COO is fully populated. Proceed to radix sort +
CSR build + credit forward/backward (all GPU-only, unchanged).

### New infrastructure

1. **`count_mask_into_slot(mask, n, task_counts, idx)`** — provider method
   that runs `count_mask` kernel and writes result into `task_counts[idx]`
   instead of allocating a fresh 1-element buffer per call. Avoids
   allocation churn for large task counts.

2. **`dtoh_scalar_untracked<T>(&self, src) → T`** — provider helper that
   reads a single scalar via `device().inner()` without touching
   `record_dtoh`. Documented as metadata-only, not data-plane. Makes the
   contract explicit rather than relying on the accidental bypass in
   `device().inner().dtoh_sync_copy()`.

### Strict gate update

`strict_zero_dtoh = true` now means: zero data-plane D2H in
`compute_ilp_loss_grad_gpu`. The 8-byte metadata read via
`dtoh_scalar_untracked` is excluded by design. Existing strict tests
(`dtoh_calls == 0`, `dtoh_bytes == 0`) continue to pass because
`dtoh_scalar_untracked` does not increment the tracker.

### Files

- Modify: `kernels/ilp.cu` + `.ptx` (add `ilp_count_mask_into_slot` or
  use existing `count_mask` with a device-pointer write target)
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (add `count_mask_into_slot`,
  `dtoh_scalar_untracked`)
- Modify: `crates/pyxlog/src/lib.rs:3987–3994` (rename field to
  `coo_chunk_budget`, add deprecated `set_coo_memory_cap` alias)
- Modify: `crates/pyxlog/src/lib.rs:4230–4514` (replace chunked path with
  two-pass streaming)
- Modify: `python/tests/test_ilp_credit_gpu.py` (update
  `test_strict_zero_dtoh_rejects_chunking` → verify chunked path now
  succeeds under strict mode with GPU-only merge; add test with tiny
  budget forcing chunking)

### Validation

`strict_zero_dtoh=True` + tiny `coo_chunk_budget` must pass all 4 ILP
stages (reach, grandparent, colleague, plus2).

---

## P1: Artifact Schema Migration + Telemetry Persistence

### Problem

Config restore from artifacts already works (`types.py:217`). What's
missing: schema versioning for forward evolution, and bounded telemetry
persistence.

### Design

**Schema migration `beta-v1` → `beta-v2`:**
- `load()` accepts both v1 and v2.
- `save()` writes v2.
- v2 adds `telemetry_snapshot` field (null when absent).
- Migration test: v1 artifact loads cleanly in v2 code.

**Bounded telemetry persistence:**
- New `TrainConfig` fields: `persist_telemetry: bool = False`,
  `telemetry_persist_limit: int = 100`.
- When `persist_telemetry=True`: `save()` includes
  `"telemetry_snapshot": {"steps": [last N StepRecords], "step_timings": {...}}`
- `load()` restores telemetry from snapshot if present, else empty.

**Compatibility tests:**
- Roundtrip: save v2 with telemetry → load → verify steps + timings.
- Backward compat: load v1 in v2 code → telemetry empty, config restored.
- Size bound: verify truncation respects `telemetry_persist_limit`.

### Files

- Modify: `crates/pyxlog/python/pyxlog/ilp/types.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `python/tests/test_ilp_artifact.py`

---

## P2a: Term Embeddings

### Problem

Embedding syntax parses (`nn(encoder, [Text], Embedding) ::
encode(Text, Embedding).`) but has no execution path. `labels.is_none()`
is recognized in the AST but the forward/backward path only handles
classification (softmax over a label list).

### Design

1. **`EmbeddingHandle`** variant in `crates/xlog-neural/src/handle.rs` —
   wraps `nn.Embedding` or frozen pretrained tensor.
2. **Python API:** `program.register_embedding(name, module_or_tensor,
   trainable=True)` on `CompiledProgram`.
3. **Execution path:** When `forward_backward_*` encounters a neural
   predicate with `labels.is_none()`, dispatch to embedding path: call
   `module(input_indices)` → get embedding vectors → store in a per-query
   term-vector table.
4. **Foreign tensor predicates:** `dot(E1, E2, Score)`,
   `cosine(E1, E2, Score)` — operate on embedding vectors from the
   term-vector table.
5. **Gradient flow:** Embeddings are differentiable PyTorch modules;
   gradients flow back through `backward()` as with classification
   networks.

### Files

- Modify: `crates/xlog-neural/src/handle.rs` (add EmbeddingHandle)
- Modify: `crates/xlog-neural/src/registry.rs` (add embedding registration)
- Modify: `crates/pyxlog/src/lib.rs` (register_embedding, embedding
  execution, foreign tensor preds)
- New: `python/tests/test_embeddings.py`
- New: `examples/neural/07_embeddings/`

---

## P2b: Extended Training Controls

### Problem

Training controls are minimal: no per-network LR access, no gradient
clipping, no early stopping.

### Design

1. `program.get_lr(network_name) → float` — reads
   `optimizer.param_groups[0]['lr']`
2. `program.set_lr(network_name, lr)` — writes to all param groups
3. `train_model(..., max_grad_norm=None)` — gradient clipping via
   `torch.nn.utils.clip_grad_norm_`
4. `train_model(..., val_queries=None, patience=None)` — early stopping:
   evaluate val loss each epoch, stop after `patience` epochs without
   improvement
5. `program.scheduler_step(network_name=None)` — per-network or all
   (None = all, backward compatible)

### Files

- Modify: `crates/pyxlog/src/lib.rs` (get_lr, set_lr, grad clipping,
  early stopping, per-network scheduler)
- Modify: `python/tests/test_training.py`

---

## P3: Incremental Verifier Interface

### Problem

The CDCL solver is stateless — every call allocates a fresh arena
(`gpu_cdcl.rs`). For the equivalence check pair (q1 = φ∧¬C, q2 = C∧¬φ),
which share the circuit CNF base, this wastes work re-deriving learned
clauses.

### Design

1. **`GpuCdclSession`** — persistent arena with base CNF:
   - `create_session(cnf) → GpuCdclSession` — allocates arena
   - `session.solve_under_assumptions(assumptions: &[i32]) → SolveResult`
   - `session.add_clauses(clauses)` — incrementally add clauses
   - `session.drop()` — release GPU memory

2. **Integration:** `validation.rs` reuses a session for the q1+q2 pair
   (shared circuit CNF base, different negation clauses).

3. **Opt-in:** `GpuCompileConfig.incremental_verify: bool` (default false).
   Existing single-shot path remains as fallback.

### Files

- New: `crates/xlog-solve/src/gpu_cdcl_session.rs`
- Modify: `crates/xlog-solve/src/gpu_cdcl.rs` (add session factory)
- New kernel variant: `kernels/sat_cdcl.cu` (assumption-aware solve)
- Modify: `crates/xlog-prob/src/compilation/validation.rs`
- New: solver integration tests in `crates/xlog-solve/tests/`

---

## P4: Release Gates

### Scope

The D2H gate applies to `compute_ilp_loss_grad_gpu` only, not the full
training step. `batch_fact_membership` (witness coverage check) performs
`dtoh_sync_copy_into` of mask bytes via `membership_mask`
(`provider/mod.rs (pre-Wave-2 line 8576)`). This is a correctness-critical control-plane
transfer that requires a GPU-resident witness coverage protocol to
eliminate (v0.6.0 scope).

### Gates

1. **Zero compute-path D2H** — `strict_zero_dtoh=True` passes on all 4
   ILP stages (reach, grandparent, colleague, plus2) with both default
   and tiny `coo_chunk_budget`.

2. **SLO harness** — `pytest python/tests/test_ilp_performance.py -v`
   passes.

3. **50-seed GA reliability** — `pytest python/tests/test_ilp_ga_reliability.py -v`
   passes (200/200, ≤600s).

4. **Regression suite** —
   `pytest python/tests/ -v --ignore=python/tests/test_ilp_ga_reliability.py`
   all pass;
   `cargo test --workspace --all-targets --exclude pyxlog --release`
   all pass.

---

## Design Decisions Log

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Rename `coo_memory_cap` → `coo_chunk_budget` | Old name implies hard ceiling on all COO allocations; new semantics allow exact-NNZ output buffer to exceed chunk budget |
| D2 | Keep deprecated `set_coo_memory_cap` alias | One-release backward compatibility for existing user code |
| D3 | Two-pass count+fill instead of global atomic | Deterministic write positions, no atomic contention, reuses existing kernels |
| D4 | `count_mask_into_slot` instead of per-call allocation | Avoids N fresh device allocations in pass 1 |
| D5 | `dtoh_scalar_untracked` for metadata reads | Makes the "metadata ≠ data-plane" contract explicit and auditable |
| D6 | P4 scoped to `compute_ilp_loss_grad_gpu` | `batch_fact_membership` D2H is control-plane; eliminating it requires GPU-resident witness protocol (v0.6.0) |
| D7 | P2 split into embeddings (P2a) and training controls (P2b) | Independent epics with no shared code |
| D8 | Incremental verifier is opt-in | Minimizes risk; single-shot path remains as fallback |
