# Device-Side Sparse Mask API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the current Rust-side sparse soft-probability download from the preferred ILP sparse-mask hot path by adding a selected-candidate API and switching the Python sparse backend to it.

**Architecture:** Keep `set_rule_mask_sparse(...)` available as the legacy full-vector path, but add a new additive sparse setter that accepts already-selected candidates from the Python backend. The backend will perform deterministic ranking on GPU with PyTorch, pass only the selected candidate IDs into Rust, and Rust will map those IDs to existing candidate triples without downloading the full soft-probability vector.

**Tech Stack:** Rust (`pyxlog`, `xlog-runtime`), PyO3, Python `pytest`, PyTorch CUDA tensors

### Task 1: Add the direct selected-candidate sparse API

**Files:**
- Modify: `python/tests/test_ilp_sparse.py`
- Modify: `crates/pyxlog/src/ilp.rs`

**Step 1: Write the failing test**

Add a focused test for `set_rule_mask_sparse_selected("W", selected_ids, selected_soft, allow_recursive=False)`:
- accepts a small selected subset instead of a full `[0..C)` candidate list
- evaluates to the same tagged results as the equivalent dense single-rule setup
- `host_transfer_stats()` reports zero DTOH after the call

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_sparse.py::test_sparse_selected_mask_is_zero_dtoh_and_matches_dense -v`
Expected: FAIL because `set_rule_mask_sparse_selected` does not exist.

**Step 3: Write minimal implementation**

In `crates/pyxlog/src/ilp.rs`:
- add `set_rule_mask_sparse_selected(...)`
- validate selected ID range against `valid_candidates(...)`
- validate selected soft-prob tensor length against selected ID count
- map selected IDs to candidate triples on host without downloading the full soft vector
- insert the sparse mask through the existing `IlpRegistry::insert_mask_from_sparse(...)`

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_sparse.py::test_sparse_selected_mask_is_zero_dtoh_and_matches_dense -v`
Expected: PASS

**Step 5: Commit**

```bash
git add python/tests/test_ilp_sparse.py crates/pyxlog/src/ilp.rs
git commit -m "feat(ilp): add selected-candidate sparse mask API"
```

### Task 2: Switch the Python sparse backend to the new preferred path

**Files:**
- Modify: `python/tests/test_ilp_sparse_guard.py`
- Modify: `python/tests/test_ilp_sparse.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/backend.py`

**Step 1: Write the failing test**

Add tests that:
- `SparseMaskBackend.apply_mask(...)` calls the new selected-candidate sparse API instead of the legacy full-vector setter
- deterministic tie-breaking still picks the lower candidate ID first
- sparse backend setup leaves `prog.host_transfer_stats()["dtoh_calls"] == 0`

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_sparse.py -k 'selected or backend' -v`
Expected: FAIL because the backend still calls `set_rule_mask_sparse(...)`.

**Step 3: Write minimal implementation**

In `crates/pyxlog/python/pyxlog/ilp/backend.py`:
- compute live candidate scores on GPU
- use deterministic ranking on GPU (`argsort(..., stable=True)`) so ties prefer lower IDs
- pass only the selected IDs plus selected soft probabilities into `set_rule_mask_sparse_selected(...)`
- keep `selected_hard` mapping aligned to artifact candidate indices

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_sparse.py -k 'selected or backend' -v`
Expected: PASS

**Step 5: Commit**

```bash
git add python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_sparse.py crates/pyxlog/python/pyxlog/ilp/backend.py
git commit -m "feat(ilp): prefer selected-candidate sparse backend path"
```

### Task 3: Focused verification and docs

**Files:**
- Modify: `docs/architecture/dilp-training.md`
- Modify: `docs/architecture/python-bindings.md`
- Modify: `python/tests/test_ilp_performance.py`

**Step 1: Write the failing test**

Add/update transfer-accounting coverage so:
- the legacy `set_rule_mask_sparse(...)` path is explicitly documented/tested as the download-based compatibility path
- the new selected-candidate path is explicitly tested as zero-DTOH in provider transfer accounting

**Step 2: Run focused verification**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_sparse.py python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_performance.py -k 'sparse' -v`
Expected: PASS

**Step 3: Write minimal implementation**

Document:
- the preferred sparse hot-loop API boundary
- that legacy `set_rule_mask_sparse(...)` remains available for compatibility
- the deterministic tie-break contract for selected candidates

**Step 4: Re-run focused verification**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_sparse.py python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_performance.py -k 'sparse' -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/architecture/dilp-training.md docs/architecture/python-bindings.md python/tests/test_ilp_performance.py
git commit -m "docs(ilp): document preferred device-side sparse mask path"
```
