# Strict ILP GPU-Native Audit Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove reachable host-sync and host-materialization escapes from strict Python ILP training/runtime paths, and hard-gate remaining compatibility-only APIs out of strict mode.

**Architecture:** Split strict and compatibility behavior explicitly. Keep the strict training hot loop on GPU by avoiding Python scalar syncs and host-selected sparse IDs, while rejecting compatibility-only telemetry/holdout/promotion/runtime helpers when strict mode is enabled.

**Tech Stack:** Python, PyTorch CUDA, PyO3, Rust, xlog-runtime, pytest, maturin

### Task 1: Add regression tests for strict runtime API gating

**Files:**
- Modify: `python/tests/test_ilp_d2h_gate.py`
- Modify: `python/tests/test_ilp_sparse_guard.py`

**Step 1: Write the failing tests**

Add tests that enable `prog.set_strict_zero_dtoh(True)` and assert host semantic APIs raise:
- `fact_exists(...)`
- `relation_facts(...)`
- `sample_false_positives(...)`
- `batch_fact_membership(...)`
- `batch_tagged_credit(...)`

Also add a test that strict mode rejects any remaining legacy sparse-mask entry point that still requires host materialization.

**Step 2: Run test to verify it fails**

Run: `python -m pytest python/tests/test_ilp_d2h_gate.py python/tests/test_ilp_sparse_guard.py -q`

Expected: FAIL on the new strict-gating cases.

**Step 3: Write minimal implementation**

Add strict-mode guards in `crates/pyxlog/src/ilp.rs` for compatibility-only host semantic APIs.

**Step 4: Run test to verify it passes**

Run: `python -m pytest python/tests/test_ilp_d2h_gate.py python/tests/test_ilp_sparse_guard.py -q`

Expected: PASS

### Task 2: Add regression tests for strict sparse selected-ID handling

**Files:**
- Modify: `python/tests/test_ilp_sparse.py`
- Modify: `python/tests/test_ilp_trainer.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/backend.py`

**Step 1: Write the failing tests**

Add tests that verify:
- strict sparse mask application does not use host `.cpu().tolist()` selected IDs
- strict sparse backend uses a device-ID API instead of host-selected lists
- strict trainer hot loop source does not use `torch.cuda.synchronize()` or host scalar downloads inside the strict loop helper

**Step 2: Run test to verify it fails**

Run: `python -m pytest python/tests/test_ilp_sparse.py python/tests/test_ilp_trainer.py -q`

Expected: FAIL on the new strict sparse-path assertions.

**Step 3: Write minimal implementation**

Add a new PyO3 entry point for selected sparse candidate IDs passed as a device tensor, store candidate order in `CompiledIlpProgram`, and route strict sparse mask setup through that path.

**Step 4: Run test to verify it passes**

Run: `python -m pytest python/tests/test_ilp_sparse.py python/tests/test_ilp_trainer.py -q`

Expected: PASS

### Task 3: Add regression tests for strict trainer compatibility gates

**Files:**
- Modify: `python/tests/test_ilp_trainer.py`
- Modify: `python/tests/test_ilp_holdout.py`
- Modify: `python/tests/test_ilp_promoter.py`

**Step 1: Write the failing tests**

Add tests that assert strict mode rejects or disables compatibility-only features:
- `telemetry_level > 0`
- `debug_dense_mask=True`
- holdout helpers under strict mode
- `train_and_promote(...)` under strict mode

Keep existing holdout/promoter coverage by setting `strict_gpu_native=False` where those compatibility paths are intentionally exercised.

**Step 2: Run test to verify it fails**

Run: `python -m pytest python/tests/test_ilp_trainer.py python/tests/test_ilp_holdout.py python/tests/test_ilp_promoter.py -q`

Expected: FAIL on the new strict-mode cases.

**Step 3: Write minimal implementation**

Validate and gate strict-incompatible features in `trainer.py`, `holdout.py`, and `promoter.py`. Split strict and compatibility training loops so the strict hot loop avoids Python scalar syncs and per-step CUDA timing.

**Step 4: Run test to verify it passes**

Run: `python -m pytest python/tests/test_ilp_trainer.py python/tests/test_ilp_holdout.py python/tests/test_ilp_promoter.py -q`

Expected: PASS

### Task 4: Rebuild native bindings and run focused verification

**Files:**
- Modify: `crates/pyxlog/src/ilp.rs`
- Modify: `crates/pyxlog/src/lib.rs`
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/backend.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/holdout.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/promoter.py`

**Step 1: Rebuild pyxlog**

Run: `maturin develop --release -m crates/pyxlog/Cargo.toml`

Expected: build succeeds

**Step 2: Run focused suites**

Run:
- `python -m pytest python/tests/test_ilp_trainer.py python/tests/test_ilp_sparse.py python/tests/test_ilp_sparse_guard.py -q`
- `python -m pytest python/tests/test_ilp_d2h_gate.py python/tests/test_ilp_device_queries.py python/tests/test_ilp_credit_gpu.py -q`
- `python -m pytest python/tests/test_ilp_holdout.py python/tests/test_ilp_promoter.py -q`

Expected: PASS

### Task 5: Write evidence document

**Files:**
- Create: `docs/evidence/2026-03-17-strict-ilp-gpu-native-audit.md`

**Step 1: Capture evidence**

For each fixed issue document:
- issue
- root cause
- fix
- strict vs compatibility classification
- verification commands and fresh results

**Step 2: Capture residual risks**

List any remaining strict-path risks with exact file and line references.
