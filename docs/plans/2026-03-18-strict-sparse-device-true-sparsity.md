# Strict SparseDevice True-Sparsity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current strict `SparseDevice` executor path with a genuinely sparse substrate that executes only selected rules without host materialization or full candidate-order scans.

**Architecture:** Today strict `SparseDevice` stores `candidate_order + active_flags` and the executor still iterates every candidate triple before filtering joined rows. The fix is to change the strict internal representation from a device flag mask to a device-selected rule list, then add a private runtime execution path that consumes that selected-rule list directly. Public generic `evaluate()` remains hard-gated until the generic runtime can use the same sparse substrate.

**Tech Stack:** Rust, PyO3, xlog-runtime, xlog-cuda, CUDA kernels, Python pytest, maturin.

## Design Summary

Recommended approach:
- Keep the public strict Python contract unchanged.
- Change strict internal `SparseDevice` storage to hold a compact selected-rule device buffer, not just `active_flags`.
- Add a private runtime entry point for strict ILP loss/grad evaluation that executes only selected rules.
- Leave public generic `evaluate()` gated until it can also route through the new sparse substrate.

Rejected approach:
- Download selected IDs to host and iterate only them.
Reason: it reintroduces hidden host materialization in the strict runtime path.

Partial-but-insufficient approach:
- Keep scanning all candidates but pre-group by `(i, j)` or other static metadata.
Reason: it may reduce cost, but it is still not truly sparse and leaves the substrate mismatch in place.

### Task 1: Lock in the regression target

**Files:**
- Modify: `python/tests/test_ilp_sparse_guard.py`
- Modify: `python/tests/test_ilp_credit_gpu.py`
- Modify: `python/tests/test_ilp_d2h_gate.py`

**Step 1: Write the failing tests**

Add tests that fail against the current dense-scan substrate:
- A structural test that the strict executor path no longer loops over `candidate_order.iter()` in the `SparseDevice` branch of `TensorMaskedJoin`.
- A structural test that the strict sparse-device registry variant stores a compact selected-rule device buffer, not only `active_flags`.
- A behavior test that strict loss/grad on a large candidate map does not use the generic public `evaluate()` path.

**Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest python/tests/test_ilp_sparse_guard.py -q -k sparse_device
python -m pytest python/tests/test_ilp_credit_gpu.py -q -k strict
python -m pytest python/tests/test_ilp_d2h_gate.py -q -k sparse_selected
```

Expected:
- At least one new assertion fails because the runtime still uses `candidate_order` scanning and the registry still stores only `active_flags`.

**Step 3: Commit**

```bash
git add python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_credit_gpu.py python/tests/test_ilp_d2h_gate.py
git commit -m "test: capture strict sparse-device true-sparsity gap"
```

### Task 2: Change strict mask storage to a selected-rule list substrate

**Files:**
- Modify: `crates/xlog-runtime/src/ilp_registry.rs`
- Modify: `crates/pyxlog/src/ilp.rs`
- Modify: `crates/xlog-cuda/src/provider/ilp.rs`
- Modify: `kernels/ilp.cu`

**Step 1: Write the failing test**

Add a Rust or Python-facing test for the new registry/debug state:
- strict selected-device setup should expose a mask kind that carries compact selected rules
- selected rule count must equal the actual compact device row count, not just a host `selected_count`

**Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest python/tests/test_ilp_sparse_guard.py -q -k runtime_sparse_device_variant
```

Expected:
- FAIL because the current variant stores `candidate_order + active_flags`.

**Step 3: Write minimal implementation**

Replace the strict internal representation:
- In `IlpMask::SparseDevice`, store:
  - `selected_rules: CudaBuffer` with columns `(i, j, k)` compacted to selected rows
  - `selected_count_device` or use the selected buffer row count metadata
  - keep host `schema_size`
- Remove `candidate_order` and `active_flags` from the strict execution-critical path
- In `set_rule_mask_sparse_selected_device_impl(...)`:
  - keep GPU validation
  - compact selected candidate IDs into selected `(i, j, k)` rows on GPU
  - build a `CudaBuffer` for those selected triples

Implementation detail:
- Upload candidate triples once as a device buffer when `set_candidate_map(...)` is called or lazily on first strict selected-device update.
- Add a provider helper that gathers rows from the candidate-triple buffer using the selected ID buffer.
- Use existing compaction/gather primitives where possible; only add a new kernel if the current gather path cannot consume the candidate-triple layout directly.

**Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest python/tests/test_ilp_sparse_guard.py -q -k 'runtime_sparse_device_variant or public_api_rejects'
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add crates/xlog-runtime/src/ilp_registry.rs crates/pyxlog/src/ilp.rs crates/xlog-cuda/src/provider/ilp.rs kernels/ilp.cu python/tests/test_ilp_sparse_guard.py
git commit -m "refactor: store strict sparse-device masks as selected-rule buffers"
```

### Task 3: Add a private strict executor path that executes only selected rules

**Files:**
- Modify: `crates/xlog-runtime/src/executor/node_dispatch.rs`
- Modify: `crates/xlog-runtime/src/executor/mod.rs`
- Modify: `crates/pyxlog/src/ilp.rs`

**Step 1: Write the failing test**

Add tests that prove the strict internal path is no longer candidate-order dense:
- structural test: `SparseDevice` branch must iterate selected-rule buffer rows, not `candidate_order.iter()`
- behavior test: strict `compute_ilp_loss_grad_gpu()` must reject if the new selected-rule buffer is absent

**Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest python/tests/test_ilp_credit_gpu.py -q -k strict
python -m pytest python/tests/test_ilp_sparse_guard.py -q -k candidate_order
```

Expected:
- FAIL because execution still scans `candidate_order`.

**Step 3: Write minimal implementation**

Add a private executor path, for example:
- `execute_tensor_masked_join_sparse_device_selected(...)`
- or a private branch inside `TensorMaskedJoin` dispatch used only by strict ILP credit/evaluation internals

Requirements:
- It must execute only selected rules.
- It must not route through public generic `evaluate()`.
- It must not download selected IDs or selected triples to host.

Practical shape:
- The host may still orchestrate the join stages, but only over the selected-rule buffer’s logical rows.
- If the existing join primitives require host-side rule tuples, then this task is not complete; add a batched join/project substrate that consumes selected `(i, j, k)` rows directly.

**Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest python/tests/test_ilp_credit_gpu.py -q -k strict
python -m pytest python/tests/test_ilp_sparse_guard.py -q -k 'candidate_order or evaluate_is_hard_gated'
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add crates/xlog-runtime/src/executor/node_dispatch.rs crates/xlog-runtime/src/executor/mod.rs crates/pyxlog/src/ilp.rs python/tests/test_ilp_credit_gpu.py python/tests/test_ilp_sparse_guard.py
git commit -m "feat: execute strict sparse-device ILP over selected rules only"
```

### Task 4: Route strict ILP loss/grad and strict reevaluation through the new path

**Files:**
- Modify: `crates/pyxlog/src/ilp.rs`
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`

**Step 1: Write the failing test**

Add tests that enforce the call graph:
- strict trainer path must use the new private sparse-device reevaluation path
- strict loss/grad must not depend on the public generic `evaluate()` gate

**Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest python/tests/test_ilp_trainer.py -q -k strict
python -m pytest python/tests/test_ilp_d2h_gate.py -q -k strict
```

Expected:
- FAIL until strict internal evaluation is wired to the new executor path.

**Step 3: Write minimal implementation**

In `compute_ilp_loss_grad_gpu(...)` and any strict internal reevaluation helper:
- require the new selected-rule sparse-device substrate
- call the new private executor path directly
- preserve the public `evaluate()` hard gate for generic strict callers

**Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest python/tests/test_ilp_trainer.py -q -k strict
python -m pytest python/tests/test_ilp_d2h_gate.py -q -k strict
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add crates/pyxlog/src/ilp.rs crates/pyxlog/python/pyxlog/ilp/trainer.py python/tests/test_ilp_trainer.py python/tests/test_ilp_d2h_gate.py
git commit -m "refactor: route strict ILP evaluation through selected-rule sparse executor"
```

### Task 5: Rebuild and run the focused verification matrix

**Files:**
- Modify: `docs/evidence/2026-03-17-strict-ilp-gpu-native-audit.md`

**Step 1: Rebuild bindings**

Run:
```bash
maturin develop --release -m crates/pyxlog/Cargo.toml
```

Expected:
- Build succeeds.

**Step 2: Run verification**

Run:
```bash
python -m pytest python/tests/test_ilp_sparse.py python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_types.py -q
python -m pytest python/tests/test_ilp_d2h_gate.py -q
python -m pytest python/tests/test_ilp_device_queries.py -q -x
python -m pytest python/tests/test_ilp_credit_gpu.py -q -x
python -m pytest python/tests/test_ilp_holdout.py -q -x
python -m pytest python/tests/test_ilp_promoter.py -q -x
python -m pytest python/tests/test_ilp_trainer.py -q
python -m pytest python/tests/test_ilp_backend.py python/tests/test_ilp_artifact.py python/tests/test_ilp_reset.py python/tests/test_ilp_multistart.py python/tests/test_ilp_robustness.py -q
cargo check -p pyxlog --quiet
```

Expected:
- All commands pass with fresh output.

**Step 3: Update evidence doc**

Document:
- issue
- root cause
- fix
- strict vs compatibility classification
- verification commands and results
- residual risks, if any, with exact file/line refs

**Step 4: Commit**

```bash
git add docs/evidence/2026-03-17-strict-ilp-gpu-native-audit.md
git commit -m "docs: record true-sparse strict sparse-device closure"
```

## Acceptance Criteria

- Strict `SparseDevice` no longer stores only `candidate_order + active_flags` as its execution substrate.
- The strict execution-critical path does not iterate `candidate_order.iter()` for `SparseDevice`.
- Strict ILP loss/grad executes only selected rules.
- Public generic `evaluate()` remains hard-gated in strict mode unless and until it also uses the new sparse substrate.
- No DTOH/download helper is added back into strict Python/runtime paths.
- No new host-shaped sparse API becomes callable under `strict_zero_dtoh`.

## Anti-Gaming Gates

- Reject any solution that downloads selected IDs or active flags to host in order to build a selected rule list.
- Reject any solution that keeps the candidate-order scan and only adds more filtering after join/project.
- Reject any solution that relaxes strict gating instead of fixing the substrate.
- Reject any solution that claims “true sparse” while still doing O(candidate_order) join dispatch.
- Reject grep-only proof; require structural tests plus behavior tests on the strict internal path.
