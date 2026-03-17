# Device-Native ILP Membership And Credit API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Python-facing device-native ILP membership and tagged-credit APIs that return DLPack-compatible CUDA outputs without semantic-loop DTOH transfers.

**Architecture:** Expose a direct device membership API as a 0/1 CUDA mask tensor and expose tagged credit as a CSR-like device result: per-fact row offsets plus device-resident entry indices and entry metadata `(i, j, k)`. Reuse the existing `membership_mask_device` primitive and the existing GPU COO/CSR builder already used by the ILP loss path.

**Tech Stack:** Rust (`pyxlog`, `xlog-cuda`), PyO3, DLPack, Python `pytest`, PyTorch CUDA tensors

### Task 1: Membership device API

**Files:**
- Create: `python/tests/test_ilp_device_queries.py`
- Modify: `crates/pyxlog/src/lib.rs`
- Modify: `crates/pyxlog/src/ilp.rs`

**Step 1: Write the failing test**

Add a focused test for `batch_fact_membership_device("edge", [[1,2], [9,9], [2,3]])`:
- returns a DLPack CUDA tensor
- decodes to `[1, 0, 1]`
- `host_transfer_stats()` shows `dtoh_calls == 0` and `dtoh_bytes == 0`

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_device_queries.py::test_batch_fact_membership_device_returns_gpu_mask -v`
Expected: FAIL because `batch_fact_membership_device` does not exist.

**Step 3: Write minimal implementation**

In `crates/pyxlog/src/ilp.rs`:
- factor query-buffer upload into a helper
- call `membership_mask_device(...)`
- wrap the device `u8` mask as a single-column `CudaBuffer`
- export column 0 as DLPack

In `crates/pyxlog/src/lib.rs`:
- no new result type needed for membership; return a capsule directly

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_device_queries.py::test_batch_fact_membership_device_returns_gpu_mask -v`
Expected: PASS

**Step 5: Commit**

```bash
git add python/tests/test_ilp_device_queries.py crates/pyxlog/src/ilp.rs crates/pyxlog/src/lib.rs
git commit -m "feat(ilp): add device-native membership query API"
```

### Task 2: Tagged credit device API

**Files:**
- Modify: `python/tests/test_ilp_device_queries.py`
- Modify: `crates/pyxlog/src/lib.rs`
- Modify: `crates/pyxlog/src/ilp.rs`
- Modify: `crates/pyxlog/src/ilp_gpu.rs`

**Step 1: Write the failing test**

Add a focused test for `batch_tagged_credit_device("reach", facts)`:
- returns a result object with DLPack tensors for `fact_row_offsets`, `entry_indices`, `entry_i`, `entry_j`, `entry_k`
- reconstructing credits on the Python side matches the existing host-returning `batch_tagged_credit(...)`
- `host_transfer_stats()` shows zero DTOH during the device-native call

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_device_queries.py::test_batch_tagged_credit_device_matches_host_credit -v`
Expected: FAIL because `batch_tagged_credit_device` and/or its result type do not exist.

**Step 3: Write minimal implementation**

In `crates/pyxlog/src/ilp.rs`:
- reuse the same relation-group upload pattern as `compute_ilp_loss_grad_gpu`
- build `CooTask` values with `cidx = entry_idx`
- call the existing COO/CSR helpers
- upload `(i, j, k)` metadata arrays once to device

In `crates/pyxlog/src/lib.rs`:
- add a `#[pyclass]` result type for the device credit outputs

In `crates/pyxlog/src/ilp_gpu.rs`:
- add small export helpers for `u32`/`u8` device slices to DLPack capsules

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_device_queries.py::test_batch_tagged_credit_device_matches_host_credit -v`
Expected: PASS

**Step 5: Commit**

```bash
git add python/tests/test_ilp_device_queries.py crates/pyxlog/src/lib.rs crates/pyxlog/src/ilp.rs crates/pyxlog/src/ilp_gpu.rs
git commit -m "feat(ilp): add device-native tagged credit query API"
```

### Task 3: Focused verification and docs

**Files:**
- Modify: `docs/architecture/dilp-training.md`
- Modify: `docs/architecture/python-bindings.md`
- Test: `python/tests/test_ilp_device_queries.py`

**Step 1: Write the failing test**

No new executable test; use the focused API tests as the contract.

**Step 2: Run focused verification**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_device_queries.py -v`
Expected: PASS

**Step 3: Write minimal implementation**

Document:
- which new ILP device-native APIs are zero semantic-loop DTOH
- that membership returns a 0/1 device mask
- that tagged credit returns CSR-style device outputs

**Step 4: Re-run focused verification**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_device_queries.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/architecture/dilp-training.md docs/architecture/python-bindings.md
git commit -m "docs(ilp): document device-native membership and credit APIs"
```
