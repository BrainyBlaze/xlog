# GPU-Native IO and MC Device Results Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add experimental Arrow device import, expose device-only MC results in Python, and tighten host-io gating while keeping GPU-native paths zero-copy.

**Architecture:** Introduce a feature-flagged Arrow C Device import path in `xlog-cuda`, add a device-results API in `pyxlog`, and gate host-read convenience APIs behind `host-io`. Keep existing host-copy Arrow IPC paths unchanged.

**Tech Stack:** Rust (xlog-cuda/xlog-prob/pyxlog), CUDA (PTX already present), Arrow C Data Interface, DLPack, PyO3.

---

### Task 1: Add experimental Arrow device import in xlog-cuda

**Files:**
- Modify: `crates/xlog-cuda/Cargo.toml`
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Modify: `crates/xlog-cuda/src/arrow_device.rs`
- Test: `crates/xlog-cuda/tests/arrow_device_import.rs`

**Step 1: Write the failing test** (@superpowers:test-driven-development)

```rust
#[test]
fn arrow_device_roundtrip_import_export() {
    // Export a tiny CudaBuffer to ArrowDeviceArray, then import it back.
    // Expect schema/row count to match.
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test arrow_device_import -- --nocapture`
Expected: FAIL with "from_arrow_device_record_batch not found".

**Step 3: Implement minimal import API**

- Add `arrow-device-import` feature in `crates/xlog-cuda/Cargo.toml`.
- Add `CudaKernelProvider::from_arrow_device_record_batch(...)` behind the feature.
- Validate `ARROW_DEVICE_CUDA`, device id, and schema type compatibility.
- Wrap Arrow buffers into `CudaColumn` without copies (keepalive like export).

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test arrow_device_import -- --nocapture`
Expected: PASS (or skipped if CUDA unavailable).

**Step 5: Commit**

```bash
git add crates/xlog-cuda/Cargo.toml crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/src/arrow_device.rs crates/xlog-cuda/tests/arrow_device_import.rs
git commit -m "cuda: add experimental Arrow device import"
```

---

### Task 2: Expose Arrow device import in pyxlog (experimental)

**Files:**
- Modify: `crates/pyxlog/Cargo.toml`
- Modify: `crates/pyxlog/src/lib.rs`
- Test: `python/tests/test_arrow_device_import.py`

**Step 1: Write the failing test**

```python
def test_arrow_device_import_roundtrip():
    # Create a CudaBuffer via pyxlog, export device Arrow, then import it back.
    # Validate rows/values.
    pass
```

**Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_arrow_device_import.py -v`
Expected: FAIL (API missing).

**Step 3: Implement minimal Python wrapper**

- Add `arrow-device-import` feature in `crates/pyxlog/Cargo.toml` that enables `xlog-cuda/arrow-device-import`.
- Expose `pyxlog.import_arrow_device(...)` in `crates/pyxlog/src/lib.rs` with explicit experimental docstring.

**Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_arrow_device_import.py -v`
Expected: PASS (or skip if CUDA unavailable).

**Step 5: Commit**

```bash
git add crates/pyxlog/Cargo.toml crates/pyxlog/src/lib.rs python/tests/test_arrow_device_import.py
git commit -m "pyxlog: add experimental Arrow device import"
```

---

### Task 3: Add device-only MC results API in pyxlog

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`
- Test: `python/tests/test_mc_device_results.py`

**Step 1: Write the failing test**

```python
def test_mc_device_results_returns_dlpack():
    # Program.evaluate_device should return DLPack for query_counts/evidence_count.
    pass
```

**Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_mc_device_results.py -v`
Expected: FAIL (evaluate_device missing).

**Step 3: Implement minimal API**

- Add `Program.evaluate_device(...)` using `McProgram::evaluate_gpu_device_with_provider`.
- Return DLPack capsules for device counts and metadata (samples/seed/confidence).

**Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_mc_device_results.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_mc_device_results.py
git commit -m "pyxlog: expose device-only MC results"
```

---

### Task 4: Tighten host-io gating for host-read convenience APIs

**Files:**
- Modify: `crates/xlog-prob/src/mc.rs`
- Modify: `crates/pyxlog/src/lib.rs`
- Test: `crates/xlog-prob/tests/mc_gpu_native.rs`

**Step 1: Write/extend failing guard test**

```rust
#[test]
fn mc_host_read_apis_gated() {
    // Ensure evaluate/evaluate_cpu/evaluate_gpu are behind host-io.
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob --test mc_gpu_native -- --nocapture`
Expected: FAIL if any host API not gated.

**Step 3: Implement gating and errors**

- Ensure host-read APIs are `#[cfg(feature = "host-io")]`.
- Python/CLI should return explicit errors when host-io disabled.

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob --test mc_gpu_native -- --nocapture`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/mc.rs crates/pyxlog/src/lib.rs crates/xlog-prob/tests/mc_gpu_native.rs
git commit -m "prob: tighten host-io gating for MC outputs"
```

---

### Task 5: Update documentation

**Files:**
- Modify: `docs/architecture/cudf-interop.md`
- Modify: `docs/architecture/python-bindings.md`
- Modify: `docs/design/2026-01-28-gpu-native-gaps.md`

**Step 1: Write doc updates**

- Note experimental Arrow device import and feature flag.
- Document `Program.evaluate_device` (device-only MC results).
- Mark resolved gaps (CPU D4, GPU CNF wired, SAT/CDCL coverage).

**Step 2: Quick doc check**

Run: `rg -n "device import|evaluate_device|host-io" docs/architecture docs/design`
Expected: entries updated.

**Step 3: Commit**

```bash
git add docs/architecture/cudf-interop.md docs/architecture/python-bindings.md docs/design/2026-01-28-gpu-native-gaps.md
git commit -m "docs: update GPU-native IO and MC device results"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-02-01-gpu-native-io-and-mc-device-implementation-plan.md`.

Two execution options:

1. **Subagent-Driven (this session)** - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
