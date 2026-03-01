# GPU Hot-Loop Transfer Elimination Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the three optimizations from the [design doc](2026-03-01-gpu-hotloop-transfer-elimination.md): row-count cache, pre-uploaded fact buffers, and complete transfer tracking.

**Architecture:** Bottom-up: first add the cache infrastructure (Rust/memory.rs), wire it into provider and fix bypasses, then add transfer tracking completeness, then add pre-upload API and wire into Python trainer. Each task is independently testable.

**Tech Stack:** Rust (xlog-cuda, xlog-runtime, pyxlog), Python (trainer.py), CUDA (no kernel changes)

---

### Task 1: Add AtomicU32 Row-Count Cache to CudaBuffer

**Files:**
- Modify: `crates/xlog-cuda/src/memory.rs:365-406` (struct + constructor)
- Modify: `crates/xlog-cuda/src/memory.rs:499-504` (test struct literal)
- Modify: `crates/xlog-cuda/src/provider.rs:1843-1848` (destructuring site)

**Step 1: Add the `cached_row_count` field and accessors to CudaBuffer**

In `crates/xlog-cuda/src/memory.rs`, add `use std::sync::atomic::{AtomicU32, Ordering};` to imports, then modify the struct and constructor:

```rust
// memory.rs — struct definition (around line 365)
pub struct CudaBuffer {
    pub columns: Vec<CudaColumn>,
    pub row_cap: u64,
    pub d_num_rows: TrackedCudaSlice<u32>,
    pub schema: Schema,
    cached_row_count: AtomicU32,  // u32::MAX = not yet cached
}
```

Add accessors and a new constructor variant after the existing `from_columns`:

```rust
impl CudaBuffer {
    // Existing from_columns — add cached_row_count: AtomicU32::new(u32::MAX) to Self { ... }
    pub fn from_columns(
        columns: Vec<CudaColumn>,
        row_cap: u64,
        d_num_rows: TrackedCudaSlice<u32>,
        schema: Schema,
    ) -> Self {
        assert_eq!(columns.len(), schema.arity(), ...);
        Self {
            columns,
            row_cap,
            d_num_rows,
            schema,
            cached_row_count: AtomicU32::new(u32::MAX),
        }
    }

    /// Like from_columns, but eagerly populates the row-count cache.
    /// Use when the host already knows the exact row count (e.g., buffer_from_columns).
    pub fn from_columns_with_host_count(
        columns: Vec<CudaColumn>,
        row_cap: u64,
        d_num_rows: TrackedCudaSlice<u32>,
        schema: Schema,
        host_row_count: u32,
    ) -> Self {
        assert_eq!(columns.len(), schema.arity(), ...);
        Self {
            columns,
            row_cap,
            d_num_rows,
            schema,
            cached_row_count: AtomicU32::new(host_row_count),
        }
    }

    /// Returns the cached row count if available (not sentinel u32::MAX).
    pub fn cached_row_count(&self) -> Option<u32> {
        let v = self.cached_row_count.load(Ordering::Relaxed);
        if v == u32::MAX { None } else { Some(v) }
    }

    /// Sets the cached row count if not already set (CAS from sentinel).
    /// No-op if already cached.
    pub fn set_cached_row_count_if_unset(&self, count: u32) {
        let _ = self.cached_row_count.compare_exchange(
            u32::MAX,
            count,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }
}
```

**Step 2: Fix struct-literal and destructuring sites**

In `crates/xlog-cuda/src/memory.rs` around line 499 (test code), change the direct struct literal to use `from_columns`:
```rust
// Before:
let buffer = CudaBuffer {
    columns: Vec::new(),
    row_cap: 100,
    d_num_rows,
    schema: schema.clone(),
};
// After:
let buffer = CudaBuffer::from_columns(Vec::new(), 100, d_num_rows, schema.clone());
```

In `crates/xlog-cuda/src/provider.rs` around line 1843, add `..` rest pattern to the destructuring:
```rust
// Before:
let CudaBuffer {
    columns: combined_columns,
    row_cap,
    d_num_rows,
    schema: _,
} = combined;
// After:
let CudaBuffer {
    columns: combined_columns,
    row_cap,
    d_num_rows,
    schema: _,
    ..
} = combined;
```

Search for any other `CudaBuffer {` struct literal sites with: `rg 'CudaBuffer \{' crates/` — fix each one.

**Step 3: Build and run Rust tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -30`
Expected: All tests pass (the new field has a default via `from_columns`, so existing code is unaffected).

**Step 4: Commit**

```bash
git add crates/xlog-cuda/src/memory.rs crates/xlog-cuda/src/provider.rs
git commit -m "feat(cuda): add AtomicU32 row-count cache to CudaBuffer"
```

---

### Task 2: Wire Row-Count Cache into Provider

**Files:**
- Modify: `crates/xlog-cuda/src/provider.rs:6969-6976` (device_row_count)
- Modify: `crates/xlog-cuda/src/provider.rs:7158-7174` (buffer_from_columns)
- Modify: `crates/xlog-cuda/src/provider.rs:6996-7007` (buffer_from_columns_with_device_count)

**Step 1: Modify `device_row_count` to check cache first**

In `crates/xlog-cuda/src/provider.rs`, replace the `device_row_count` method (lines 6969-6976):

```rust
fn device_row_count(&self, buffer: &CudaBuffer) -> Result<usize> {
    if let Some(cached) = buffer.cached_row_count() {
        return Ok(cached as usize);
    }
    let mut host_rows = [0u32];
    self.dtoh_sync_copy_into_tracked(buffer.num_rows_device(), &mut host_rows)?;
    buffer.set_cached_row_count_if_unset(host_rows[0]);
    Ok(host_rows[0] as usize)
}
```

Note: this also converts the D2H fallback from direct `device.inner().dtoh_sync_copy_into` to `dtoh_sync_copy_into_tracked` — completing part of Task 4's tracking work.

**Step 2: Modify `buffer_from_columns` to eagerly populate cache**

In `crates/xlog-cuda/src/provider.rs`, replace the `buffer_from_columns` method (lines 7158-7174):

```rust
pub(crate) fn buffer_from_columns(
    &self,
    columns: Vec<CudaColumn>,
    row_cap: u64,
    schema: Schema,
) -> Result<CudaBuffer> {
    let row_u32 = u32::try_from(row_cap)
        .map_err(|_| XlogError::Kernel(format!("Row capacity {} exceeds u32::MAX", row_cap)))?;
    let mut d_num_rows = self.memory.alloc::<u32>(1)?;
    self.device
        .inner()
        .htod_sync_copy_into(&[row_u32], &mut d_num_rows)
        .map_err(|e| XlogError::Kernel(format!("Failed to set row count: {}", e)))?;
    Ok(CudaBuffer::from_columns_with_host_count(
        columns, row_cap, d_num_rows, schema, row_u32,
    ))
}
```

**Step 3: Build and run Rust tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -30`
Expected: All tests pass.

**Step 4: Run Python ILP tests to verify no regressions**

Run: `cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py -v --timeout=120 2>&1 | tail -20`
Expected: All 10 tests pass.

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider.rs
git commit -m "feat(cuda): wire row-count cache into device_row_count + buffer_from_columns"
```

---

### Task 3: Fix External Row-Count Bypasses

**Files:**
- Modify: `crates/xlog-runtime/src/ilp_registry.rs:9-20` (read_device_row_count)
- Modify: `crates/xlog-runtime/src/executor.rs:2961-2969` (buffer_row_count)

**Step 1: Rewrite `read_device_row_count` in ilp_registry.rs**

Replace the function (lines 9-20) to use the buffer cache:

First, make `device_row_count` public on `CudaKernelProvider` so external crates can route through it (it already has cache + tracked D2H from Task 2):

In `crates/xlog-cuda/src/provider.rs`, change the method visibility (around line 6969):
```rust
// Before:
fn device_row_count(&self, buffer: &CudaBuffer) -> Result<usize> {
// After:
pub fn device_row_count(&self, buffer: &CudaBuffer) -> Result<usize> {
```

Now rewrite `read_device_row_count` to delegate to the provider's tracked+cached method:

```rust
pub fn read_device_row_count(
    provider: &CudaKernelProvider,
    buffer: &CudaBuffer,
) -> Result<usize, XlogError> {
    // Route through provider.device_row_count which uses
    // the cache (Task 1) and tracked D2H wrapper (Task 2).
    provider.device_row_count(buffer)
}
```

**Step 2: Rewrite `buffer_row_count` in executor.rs**

Replace the method (lines 2961-2969) to delegate to the provider:

```rust
fn buffer_row_count(&self, buffer: &CudaBuffer) -> Result<u32> {
    // Route through provider.device_row_count which uses
    // the cache (Task 1) and tracked D2H wrapper (Task 2).
    self.provider
        .device_row_count(buffer)
        .map(|n| n as u32)
}
```

**Step 3: Build and run Rust tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -30`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add crates/xlog-runtime/src/ilp_registry.rs crates/xlog-runtime/src/executor.rs
git commit -m "fix(runtime): route row-count bypasses through CudaBuffer cache"
```

---

### Task 4: Route Untracked D2H and H2D Through Tracked Wrappers

**Files:**
- Modify: `crates/xlog-cuda/src/provider.rs:8410` (membership_mask mask download — D2H)
- Modify: `crates/xlog-cuda/src/provider.rs:10716,10744-10756` (extract_active_rule_indices — D2H)
- Modify: `crates/xlog-cuda/src/provider.rs:10662` (extract_active_rule_indices — H2D)
- Modify: `crates/xlog-cuda/src/provider.rs:5806` (create_buffer_from_u32_columns — H2D)

**Step 1: Convert `membership_mask` mask download to tracked**

In `crates/xlog-cuda/src/provider.rs`, find the mask download in `membership_mask` (around line 8410):

```rust
// Before (around line 8406-8412):
let mut host_mask = vec![0u8; num_probe];
self.device
    .inner()
    .dtoh_sync_copy_into(&d_has_match, &mut host_mask)
    .map_err(|e| {
        XlogError::Kernel(format!("Failed to download membership mask: {}", e))
    })?;

// After:
let mut host_mask = vec![0u8; num_probe];
self.dtoh_sync_copy_into_tracked(&d_has_match, &mut host_mask)?;
```

**Step 2: Convert `extract_active_rule_indices` D2H calls to tracked**

In `crates/xlog-cuda/src/provider.rs`, find `extract_active_rule_indices` and convert all 5 D2H calls:

```rust
// Line ~10716: count download
// Before:
self.device()
    .inner()
    .dtoh_sync_copy_into(&count, &mut count_host)
    .map_err(|e| XlogError::Kernel(format!("ILP dtoh count: {}", e)))?;
// After:
self.dtoh_sync_copy_into_tracked(&count, &mut count_host)?;

// Lines ~10744-10756: i, j, k, p downloads (4 calls)
// Before (repeated for i, j, k, p):
self.device()
    .inner()
    .dtoh_sync_copy_into(&out_i_view, &mut i_host)
    .map_err(|e| XlogError::Kernel(format!("ILP dtoh i: {}", e)))?;
// After:
self.dtoh_sync_copy_into_tracked(&out_i_view, &mut i_host)?;
// (same pattern for j, k, p)
```

Note: `dtoh_sync_copy_into_tracked` uses `&self` (not `&self.device()`) and returns `Result<()>` with XlogError, so the call site simplifies. Check the existing tracked wrapper signature at line 1555 — it takes `&self` on `CudaKernelProvider`. But `extract_active_rule_indices` uses `self.device()` which returns the `CudaDevice`. The tracked wrapper is `self.dtoh_sync_copy_into_tracked()` on CudaKernelProvider. Verify the method receiver — if `extract_active_rule_indices` is on `CudaKernelProvider`, use `self.dtoh_sync_copy_into_tracked(...)`. If it's on a sub-struct, you may need to pass the tracker through.

**Step 3: Convert step-loop H2D calls to tracked**

`htod_sync_copy_into_tracked` already exists on `CudaKernelProvider` (provider.rs:1581). Route step-loop H2D paths through it:

In `extract_active_rule_indices` (line ~10662), convert the count-register initialization:
```rust
// Before:
self.device()
    .inner()
    .htod_sync_copy_into(&[0u32], &mut count)
    .map_err(|e| XlogError::Kernel(format!("ILP htod count: {}", e)))?;
// After:
self.htod_sync_copy_into_tracked(&[0u32], &mut count)?;
```

In `create_buffer_from_u32_columns` (line ~5806), convert the column upload:
```rust
// Before:
self.device
    .inner()
    .htod_sync_copy_into(&bytes, &mut col)
    .map_err(|e| XlogError::Kernel(format!("Failed to upload column: {}", e)))?;
// After:
self.htod_sync_copy_into_tracked(&bytes, &mut col)?;
```

Note: construction-time H2D (e.g., `buffer_from_columns` at line 7169 uploading row counts) does NOT need conversion — it runs during compile/init, not during the step loop. The H2D assertion is scoped to the post-evaluate window where these paths don't execute.

**Step 4: Verify no remaining untracked D2H in provider.rs**

Run: `rg 'device.*\.inner\(\).*\.dtoh_sync_copy_into' crates/xlog-cuda/src/provider.rs`
Expected: Zero matches (all converted to tracked wrapper). If any remain, convert them too.

Also check: `rg 'device\(\).*\.inner\(\).*\.dtoh_sync_copy_into' crates/xlog-cuda/src/provider.rs`
Expected: Zero matches.

**Step 5: Build and run tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -30`
Expected: All tests pass.

Run: `cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py -v --timeout=120 2>&1 | tail -20`
Expected: All 10 tests pass.

**Step 6: Commit**

```bash
git add crates/xlog-cuda/src/provider.rs
git commit -m "feat(cuda): route all provider D2H + step-loop H2D through tracked wrappers"
```

---

### Task 5: Expose host_transfer_stats via PyO3

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (CompiledIlpProgram impl block)
- Modify: `python/tests/test_ilp_d2h_gate.py` (add test)

**Step 1: Write the failing Python test**

Add to `python/tests/test_ilp_d2h_gate.py`:

```python
def test_host_transfer_stats_accessible():
    """host_transfer_stats returns a dict with expected keys."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    stats = prog.host_transfer_stats()
    assert isinstance(stats, dict)
    assert "dtoh_calls" in stats
    assert "dtoh_bytes" in stats
    assert "htod_calls" in stats
    assert "htod_bytes" in stats


def test_host_transfer_stats_reset():
    """reset clears all counters."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()
    stats_before = prog.host_transfer_stats()
    assert stats_before["dtoh_calls"] > 0 or stats_before["htod_calls"] > 0

    prog.reset_host_transfer_stats()
    stats_after = prog.host_transfer_stats()
    assert stats_after["dtoh_calls"] == 0
    assert stats_after["dtoh_bytes"] == 0
    assert stats_after["htod_calls"] == 0
    assert stats_after["htod_bytes"] == 0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py::test_host_transfer_stats_accessible -v --timeout=60 2>&1`
Expected: FAIL — `AttributeError: 'CompiledIlpProgram' object has no attribute 'host_transfer_stats'`

**Step 3: Add PyO3 methods to CompiledIlpProgram**

In `crates/pyxlog/src/lib.rs`, find the `#[pymethods]` impl block for `CompiledIlpProgram` and add:

```rust
/// Returns comprehensive host transfer statistics from the provider.
/// Dict keys: dtoh_calls, dtoh_bytes, htod_calls, htod_bytes
pub fn host_transfer_stats(&self) -> PyResult<HashMap<String, u64>> {
    let stats = self.provider.host_transfer_stats();
    let mut map = HashMap::new();
    map.insert("dtoh_calls".to_string(), stats.dtoh_calls);
    map.insert("dtoh_bytes".to_string(), stats.dtoh_bytes);
    map.insert("htod_calls".to_string(), stats.htod_calls);
    map.insert("htod_bytes".to_string(), stats.htod_bytes);
    Ok(map)
}

/// Resets the host transfer statistics counters to zero.
pub fn reset_host_transfer_stats(&self) {
    self.provider.reset_host_transfer_stats();
}
```

Ensure `use std::collections::HashMap;` is imported (may already be present).

Verify that `CudaKernelProvider` has public methods `host_transfer_stats()` and `reset_host_transfer_stats()`. The explore found references at provider.rs:1532-1538. If these methods exist but return `HostTransferStats` struct, check if `HostTransferStats` has `dtoh_calls`, `dtoh_bytes`, `htod_calls`, `htod_bytes` fields (it does, based on the `snapshot()` method at line 707).

**Step 4: Build and run test**

Run: `cd /home/dev/projects/xlog && cargo build -p pyxlog --release 2>&1 | tail -10 && .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py::test_host_transfer_stats_accessible python/tests/test_ilp_d2h_gate.py::test_host_transfer_stats_reset -v --timeout=60 2>&1`
Expected: Both tests PASS.

**Step 5: Commit**

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_ilp_d2h_gate.py
git commit -m "feat(pyxlog): expose host_transfer_stats via PyO3"
```

---

### Task 6: Add Pre-Upload Fact Buffer Storage and Lifecycle API

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (CompiledIlpProgram struct + methods)
- Modify: `python/tests/test_ilp_d2h_gate.py` (add tests)

**Step 1: Write the failing Python tests**

Add to `python/tests/test_ilp_d2h_gate.py`:

```python
def test_upload_fact_buffer_returns_handle():
    """upload_fact_buffer returns an opaque u64 handle."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    h = prog.upload_fact_buffer("edge", [[1, 2], [2, 3]])
    assert isinstance(h, int)
    assert h >= 0
    prog.drop_fact_buffer(h)


def test_drop_fact_buffer_invalid_handle():
    """drop_fact_buffer with invalid handle raises error."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    with pytest.raises(Exception):
        prog.drop_fact_buffer(99999)


def test_upload_fact_buffer_multiple_handles():
    """Multiple uploads return distinct handles."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    h1 = prog.upload_fact_buffer("edge", [[1, 2]])
    h2 = prog.upload_fact_buffer("edge", [[2, 3]])
    assert h1 != h2
    prog.drop_fact_buffer(h1)
    prog.drop_fact_buffer(h2)
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py::test_upload_fact_buffer_returns_handle -v --timeout=60 2>&1`
Expected: FAIL — `AttributeError: 'CompiledIlpProgram' object has no attribute 'upload_fact_buffer'`

**Step 3: Add fact buffer storage and lifecycle methods**

In `crates/pyxlog/src/lib.rs`, add to `CompiledIlpProgram` struct:

```rust
#[pyclass]
pub struct CompiledIlpProgram {
    // ... existing fields ...
    fact_buffers: HashMap<u64, FactBufferEntry>,
    next_fact_handle: u64,
}

struct FactBufferEntry {
    relation: String,
    buffer: CudaBuffer,
    fact_count: usize,
}
```

Initialize in the constructor (wherever `CompiledIlpProgram` is created):
```rust
fact_buffers: HashMap::new(),
next_fact_handle: 0,
```

Add methods to the `#[pymethods]` impl block:

```rust
/// Upload facts to GPU and return a reusable handle.
/// facts: list of fact tuples as lists of ints, e.g. [[1,2], [3,4]]
pub fn upload_fact_buffer(
    &mut self,
    relation: &str,
    facts: Vec<Vec<i64>>,
) -> PyResult<u64> {
    if facts.is_empty() {
        return Err(PyValueError::new_err("Cannot upload empty fact list"));
    }

    // Look up relation schema
    let schema = self.schemas.get(relation)
        .ok_or_else(|| PyValueError::new_err(
            format!("Relation '{}' not found in schema", relation)
        ))?
        .clone();
    let arity = schema.arity();

    // Validate fact arity
    for (idx, fact) in facts.iter().enumerate() {
        if fact.len() != arity {
            return Err(PyValueError::new_err(format!(
                "Fact {} has {} values, expected {} (arity of '{}')",
                idx, fact.len(), arity, relation,
            )));
        }
    }

    // Transpose to columnar layout and encode as u32
    let mut columns: Vec<Vec<u32>> = vec![Vec::with_capacity(facts.len()); arity];
    for fact in &facts {
        for (col_idx, &val) in fact.iter().enumerate() {
            columns[col_idx].push(val as u32);
        }
    }

    let col_slices: Vec<&[u32]> = columns.iter().map(|c| c.as_slice()).collect();
    let buffer = self.provider
        .create_buffer_from_u32_columns(&col_slices, schema)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let handle = self.next_fact_handle;
    self.next_fact_handle += 1;
    let fact_count = facts.len();
    self.fact_buffers.insert(handle, FactBufferEntry {
        relation: relation.to_string(),
        buffer,
        fact_count,
    });

    Ok(handle)
}

/// Drop a pre-uploaded buffer to free GPU memory.
pub fn drop_fact_buffer(&mut self, handle: u64) -> PyResult<()> {
    self.fact_buffers.remove(&handle)
        .ok_or_else(|| PyValueError::new_err(
            format!("Invalid fact buffer handle: {}", handle)
        ))?;
    Ok(())
}
```

**Step 4: Build and run tests**

Run: `cd /home/dev/projects/xlog && cargo build -p pyxlog --release 2>&1 | tail -10 && .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py::test_upload_fact_buffer_returns_handle python/tests/test_ilp_d2h_gate.py::test_drop_fact_buffer_invalid_handle python/tests/test_ilp_d2h_gate.py::test_upload_fact_buffer_multiple_handles -v --timeout=60 2>&1`
Expected: All 3 new tests PASS.

**Step 5: Run existing tests for regressions**

Run: `cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py -v --timeout=120 2>&1 | tail -20`
Expected: All tests pass (existing + new).

**Step 6: Commit**

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_ilp_d2h_gate.py
git commit -m "feat(pyxlog): add upload_fact_buffer/drop_fact_buffer lifecycle API"
```

---

### Task 7: Add Preloaded API Variants

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (add _preloaded methods)
- Modify: `python/tests/test_ilp_d2h_gate.py` (add tests)

**Step 1: Write the failing Python tests**

Add to `python/tests/test_ilp_d2h_gate.py`:

```python
def test_batch_fact_membership_preloaded():
    """Preloaded variant returns same results as non-preloaded."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()

    facts = [[1, 2], [5, 6], [2, 3]]
    expected = prog.batch_fact_membership("edge", facts)

    h = prog.upload_fact_buffer("edge", facts)
    try:
        result = prog.batch_fact_membership_preloaded(h)
        assert result == expected, f"preloaded={result} != expected={expected}"
    finally:
        prog.drop_fact_buffer(h)


def test_batch_tagged_credit_preloaded():
    """Preloaded variant returns same results as non-preloaded."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()
    k_reach = rel_names.index("reach")
    i_edge = rel_names.index("edge")

    M_hard = torch.zeros(n, n, n, device="cuda")
    M_soft = torch.zeros(n, n, n, device="cuda")
    M_hard[i_edge, i_edge, k_reach] = 1.0
    M_soft[i_edge, i_edge, k_reach] = 1.0
    prog.set_rule_mask("W_reach", M_hard.view(-1), M_soft.view(-1), n)
    prog.evaluate()

    facts = [[1, 3], [2, 4], [99, 99]]
    expected = prog.batch_tagged_credit("reach", facts)

    h = prog.upload_fact_buffer("reach", facts)
    try:
        result = prog.batch_tagged_credit_preloaded(h)
        assert result == expected, f"preloaded={result} != expected={expected}"
    finally:
        prog.drop_fact_buffer(h)


def test_preloaded_no_htod_uploads():
    """Preloaded variants must not trigger any H2D fact uploads."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()

    facts = [[1, 2], [2, 3]]
    h = prog.upload_fact_buffer("edge", facts)
    try:
        prog.reset_host_transfer_stats()
        _ = prog.batch_fact_membership_preloaded(h)
        stats = prog.host_transfer_stats()
        assert stats["htod_calls"] == 0, (
            f"preloaded membership triggered {stats['htod_calls']} H2D uploads"
        )
    finally:
        prog.drop_fact_buffer(h)
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py::test_batch_fact_membership_preloaded -v --timeout=60 2>&1`
Expected: FAIL — `AttributeError: 'CompiledIlpProgram' object has no attribute 'batch_fact_membership_preloaded'`

**Step 3: Add preloaded methods**

In `crates/pyxlog/src/lib.rs`, add to the `#[pymethods]` impl block:

```rust
/// Like batch_fact_membership, but uses a pre-uploaded buffer (no H2D upload).
pub fn batch_fact_membership_preloaded(&self, handle: u64) -> PyResult<Vec<bool>> {
    let entry = self.fact_buffers.get(&handle)
        .ok_or_else(|| PyValueError::new_err(
            format!("Invalid fact buffer handle: {}", handle)
        ))?;

    let buf = self.executor.store().get(&entry.relation)
        .ok_or_else(|| PyValueError::new_err(
            format!("Relation '{}' not found", entry.relation)
        ))?;

    let arity = entry.buffer.arity();
    if arity == 0 {
        return Ok(vec![false; entry.fact_count]);
    }

    let keys: Vec<usize> = (0..arity).collect();

    self.provider
        .membership_mask(&entry.buffer, buf, &keys, &keys)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Like batch_tagged_credit, but uses a pre-uploaded buffer (no H2D upload).
pub fn batch_tagged_credit_preloaded(
    &self,
    handle: u64,
) -> PyResult<Vec<Vec<(u32, u32, u32)>>> {
    let entry = self.fact_buffers.get(&handle)
        .ok_or_else(|| PyValueError::new_err(
            format!("Invalid fact buffer handle: {}", handle)
        ))?;

    if entry.fact_count == 0 {
        return Ok(Vec::new());
    }

    let k_idx = self.rel_index.iter()
        .position(|(_, name)| name == &entry.relation)
        .ok_or_else(|| PyValueError::new_err(
            format!("Relation '{}' not in ILP schema", entry.relation)
        ))? as u32;

    let tagged = match self.executor.ilp_last_result() {
        Some(t) => t,
        None => return Ok(vec![Vec::new(); entry.fact_count]),
    };

    let relevant_entries: Vec<&xlog_runtime::ilp_registry::IlpTagEntry> = tagged.entries.iter()
        .filter(|e| e.k == k_idx && e.num_rows > 0 && e.buffer.is_some())
        .collect();

    if relevant_entries.is_empty() {
        return Ok(vec![Vec::new(); entry.fact_count]);
    }

    let arity = entry.buffer.arity();
    let keys: Vec<usize> = (0..arity).collect();

    let mut per_fact_credits: Vec<Vec<(u32, u32, u32)>> = vec![Vec::new(); entry.fact_count];

    for tagged_entry in &relevant_entries {
        let entry_buf = tagged_entry.buffer.as_ref().unwrap();
        let mask = self.provider
            .membership_mask(&entry.buffer, entry_buf, &keys, &keys)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        for (fact_idx, &found) in mask.iter().enumerate() {
            if found {
                per_fact_credits[fact_idx].push((
                    tagged_entry.i, tagged_entry.j, tagged_entry.k,
                ));
            }
        }
    }

    Ok(per_fact_credits)
}
```

**Step 4: Build and run tests**

Run: `cd /home/dev/projects/xlog && cargo build -p pyxlog --release 2>&1 | tail -10 && .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py::test_batch_fact_membership_preloaded python/tests/test_ilp_d2h_gate.py::test_batch_tagged_credit_preloaded python/tests/test_ilp_d2h_gate.py::test_preloaded_no_htod_uploads -v --timeout=120 2>&1`
Expected: All 3 new tests PASS.

**Step 5: Run all D2H gate tests**

Run: `cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py -v --timeout=120 2>&1 | tail -25`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_ilp_d2h_gate.py
git commit -m "feat(pyxlog): add batch_fact_membership_preloaded + batch_tagged_credit_preloaded"
```

---

### Task 8: Update Trainer to Use Preloaded APIs

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py` (_run_single_attempt, _compute_loss, _check_convergence)

**Step 1: Modify `_run_single_attempt` to pre-upload facts and use RAII cleanup**

In `crates/pyxlog/python/pyxlog/ilp/trainer.py`, modify `_run_single_attempt`:

1. After `prog = pyxlog.IlpProgramFactory.compile(...)`, group facts by relation and pre-upload:

```python
def _run_single_attempt(...):
    prog = pyxlog.IlpProgramFactory.compile(...)

    # Pre-upload fact buffers once per attempt (outside step loop)
    pos_by_rel: dict[str, list[list[int]]] = {}
    pos_indices_by_rel: dict[str, list[int]] = {}
    for idx, (rel, vals) in enumerate(positives):
        pos_by_rel.setdefault(rel, []).append(vals)
        pos_indices_by_rel.setdefault(rel, []).append(idx)

    neg_by_rel: dict[str, list[list[int]]] = {}
    neg_indices_by_rel: dict[str, list[int]] = {}
    for idx, (rel, vals) in enumerate(negatives):
        neg_by_rel.setdefault(rel, []).append(vals)
        neg_indices_by_rel.setdefault(rel, []).append(idx)

    handles: list[tuple[str, int, str]] = []  # (kind, handle, relation)
    try:
        for rel_name, facts_list in pos_by_rel.items():
            h = prog.upload_fact_buffer(rel_name, facts_list)
            handles.append(("pos", h, rel_name))
        for rel_name, facts_list in neg_by_rel.items():
            h = prog.upload_fact_buffer(rel_name, facts_list)
            handles.append(("neg", h, rel_name))

        # Build handle lookup dicts
        pos_handles = {rel: h for kind, h, rel in handles if kind == "pos"}
        neg_handles = {rel: h for kind, h, rel in handles if kind == "neg"}

        # ... step loop uses pos_handles/neg_handles ...
    finally:
        for _, h, _ in handles:
            prog.drop_fact_buffer(h)
```

2. Inside the step loop, replace `batch_fact_membership` calls with `batch_fact_membership_preloaded`:

```python
# Witness coverage — replace batch_fact_membership with preloaded
witness_mask = [False] * len(positives)
for rel_name in pos_by_rel:
    h = pos_handles[rel_name]
    mask = prog.batch_fact_membership_preloaded(h)
    for local_idx, found in enumerate(mask):
        global_idx = pos_indices_by_rel[rel_name][local_idx]
        witness_mask[global_idx] = found
```

3. Modify `_compute_loss` to accept and use preloaded handles:

```python
def _compute_loss(
    prog,
    M_soft: torch.Tensor,
    pos_handles: dict[str, int],   # relation -> handle
    neg_handles: dict[str, int],   # relation -> handle
    rel_names: list[str],
    n: int,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device="cuda")

    for rel_name, h in pos_handles.items():
        credits = prog.batch_tagged_credit_preloaded(h)
        for fact_idx, contributing in enumerate(credits):
            if contributing:
                credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
                loss = loss + (-torch.log(credit.clamp(min=1e-8)))
            else:
                k_idx = rel_names.index(rel_name)
                loss = loss + (-M_soft[:, :, k_idx].sum() / (n * n))

    for rel_name, h in neg_handles.items():
        credits = prog.batch_tagged_credit_preloaded(h)
        for fact_idx, contributing in enumerate(credits):
            if contributing:
                credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
                loss = loss + (-torch.log((1.0 - credit).clamp(min=1e-8)))

    return loss
```

4. Modify `_check_convergence` to accept and use preloaded handles:

```python
def _check_convergence(
    prog,
    W: torch.Tensor,
    mask_name: str,
    pos_handles: dict[str, int],
    neg_handles: dict[str, int],
    rel_names: list[str],
    n: int,
    argmax_ijk: tuple[int, int, int],
) -> bool:
    # Gate 1: all positives derived
    for rel_name, h in pos_handles.items():
        mask = prog.batch_fact_membership_preloaded(h)
        if not all(mask):
            return False

    # Gate 2: no negatives derived
    for rel_name, h in neg_handles.items():
        mask = prog.batch_fact_membership_preloaded(h)
        if any(mask):
            return False

    # Gate 3: argmax-only re-evaluation
    i, j, k = argmax_ijk
    M_check = torch.zeros((n, n, n), device="cuda")
    M_check[i, j, k] = 1.0
    flat_check = M_check.contiguous().view(-1)
    prog.set_rule_mask(mask_name, flat_check, flat_check, n)
    prog.evaluate()
    prog.reset_d2h_transfer_count()

    for rel_name, h in pos_handles.items():
        mask = prog.batch_fact_membership_preloaded(h)
        if not all(mask):
            return False

    if neg_handles:
        for rel_name, h in neg_handles.items():
            mask = prog.batch_fact_membership_preloaded(h)
            if any(mask):
                return False

    return True
```

5. Update the call sites in `_run_single_attempt` to pass handles instead of fact lists to `_compute_loss` and `_check_convergence`.

**Step 2: Build and run Python tests**

Run: `cd /home/dev/projects/xlog && cargo build -p pyxlog --release 2>&1 | tail -5 && .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py -v --timeout=120 2>&1 | tail -25`
Expected: All tests pass.

Run: `cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/ -v --timeout=120 -k 'not slow' 2>&1 | tail -30`
Expected: All non-slow tests pass.

**Step 3: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/trainer.py
git commit -m "feat(trainer): use preloaded fact buffers — zero H2D in step loop"
```

---

### Task 9: Add Transfer Profiling Integration Test

**Files:**
- Modify: `python/tests/test_ilp_d2h_gate.py` (add profiling test)

**Step 1: Write the transfer profiling test**

Add to `python/tests/test_ilp_d2h_gate.py`:

```python
def test_transfer_profiling_post_evaluate():
    """Post-evaluate D2H/step must be within design-doc thresholds.

    For reach benchmark (2 positives, 0 negatives, 1 relation):
    - Post-evaluate D2H: E_post + R_pos + R_neg + 2 = 2 + 1 + 0 + 2 = 5
    - Post-evaluate H2D: 0 (all facts pre-uploaded)
    """
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()
    k_reach = rel_names.index("reach")
    i_edge = rel_names.index("edge")

    # Set mask so edge+edge->reach is active
    M_hard = torch.zeros(n, n, n, device="cuda")
    M_soft = torch.zeros(n, n, n, device="cuda")
    M_hard[i_edge, i_edge, k_reach] = 1.0
    M_soft[i_edge, i_edge, k_reach] = 1.0
    prog.set_rule_mask("W_reach", M_hard.view(-1), M_soft.view(-1), n)
    prog.evaluate()

    # Pre-upload facts
    pos_facts = [[1, 3], [2, 4]]
    h_pos = prog.upload_fact_buffer("reach", pos_facts)

    try:
        # Reset stats AFTER evaluate (post-evaluate window)
        prog.reset_host_transfer_stats()

        # Simulate step-loop post-evaluate work
        _ = prog.batch_tagged_credit_preloaded(h_pos)      # loss
        _ = prog.batch_fact_membership_preloaded(h_pos)     # witness

        stats = prog.host_transfer_stats()

        # Post-evaluate D2H threshold: E_post + R_pos + R_neg + margin
        # E_post ≈ 2 (2 positives × 1 contributing rule)
        # R_pos = 1, R_neg = 0, margin = 2
        POST_EVAL_D2H_THRESHOLD = 5
        assert stats["dtoh_calls"] <= POST_EVAL_D2H_THRESHOLD, (
            f"Post-evaluate D2H calls {stats['dtoh_calls']} exceeds threshold "
            f"{POST_EVAL_D2H_THRESHOLD}. Stats: {stats}"
        )

        # H2D must be zero (facts pre-uploaded)
        assert stats["htod_calls"] == 0, (
            f"Post-evaluate H2D calls {stats['htod_calls']} should be 0. Stats: {stats}"
        )
    finally:
        prog.drop_fact_buffer(h_pos)
```

**Step 2: Run the test**

Run: `cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py::test_transfer_profiling_post_evaluate -v --timeout=120 2>&1`
Expected: PASS

**Step 3: Run all D2H gate tests**

Run: `cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py -v --timeout=120 2>&1 | tail -30`
Expected: All tests pass.

**Step 4: Run all non-slow Python tests**

Run: `cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/ -v --timeout=120 -k 'not slow' 2>&1 | tail -30`
Expected: All pass.

**Step 5: Commit**

```bash
git add python/tests/test_ilp_d2h_gate.py
git commit -m "test(ilp): add transfer profiling integration test with concrete thresholds"
```

---

### Task 10: Run Reliability Suite

**Files:** None (verification only)

**Step 1: Run Rust workspace tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -30`
Expected: All tests pass.

**Step 2: Run all non-slow Python tests**

Run: `cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/ -v --timeout=120 -k 'not slow' 2>&1 | tail -30`
Expected: All pass.

**Step 3: Run 20/20 reliability suite**

Run: `cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/test_ilp_reliability.py -v --timeout=1800 2>&1 | tail -40`
Expected: 20/20 passed.

**Step 4: Final commit summary**

No code changes — this is verification only. If all pass, the milestone is complete.
