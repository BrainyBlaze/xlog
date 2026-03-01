# GPU Hot-Loop Transfer Elimination — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate remaining redundant D2H/H2D transfers in the ILP trainer step loop via row-count caching, pre-uploaded fact buffers, and complete transfer tracking.

**Architecture:** Three layered optimizations: (1) `AtomicU32` row-count cache on `CudaBuffer` removes ~130 of ~170 D2H/step, (2) pre-uploaded fact buffer handles remove ~3 H2D/step, (3) routing all D2H through `dtoh_sync_copy_into_tracked` gives complete profiling signal via existing `HostTransferTracker`.

**Tech Stack:** Rust (xlog-cuda `memory.rs`/`provider.rs`, xlog-runtime `executor.rs`/`ilp_registry.rs`, pyxlog `lib.rs`), Python (`trainer.py`), CUDA (no kernel changes)

---

### Task 1: Add `cached_row_count` field to CudaBuffer

**Context:** `CudaBuffer` (memory.rs:365-374) stores a device-resident row count (`d_num_rows: TrackedCudaSlice<u32>`) that is immutable after construction. Currently every read does a 4-byte D2H transfer. We add an `AtomicU32` cache with `u32::MAX` as the "unknown" sentinel, so the first D2H populates the cache and all subsequent reads are free.

**Files:**
- Modify: `crates/xlog-cuda/src/memory.rs:365-420`

**Step 1: Write the failing test**

Add to the existing test module in `crates/xlog-cuda/src/memory.rs` (tests are at the bottom of the file):

```rust
#[test]
fn cached_row_count_unset_by_default() {
    let device = CudaDevice::new(0).expect("CUDA device");
    let schema = Schema::new(vec![ScalarType::U32]);
    let budget = MemoryBudget::with_limit(1024 * 1024);
    let manager = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    let mut d_num_rows = manager.alloc::<u32>(1).unwrap();
    device.htod_sync_copy_into(&[42u32], &mut d_num_rows).unwrap();
    let buffer = CudaBuffer::from_columns(Vec::new(), 42, d_num_rows, schema);
    // Default construction does NOT eagerly cache
    assert_eq!(buffer.cached_row_count(), None);
}

#[test]
fn cached_row_count_set_and_read() {
    let device = CudaDevice::new(0).expect("CUDA device");
    let schema = Schema::new(vec![ScalarType::U32]);
    let budget = MemoryBudget::with_limit(1024 * 1024);
    let manager = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    let mut d_num_rows = manager.alloc::<u32>(1).unwrap();
    device.htod_sync_copy_into(&[42u32], &mut d_num_rows).unwrap();
    let buffer = CudaBuffer::from_columns(Vec::new(), 42, d_num_rows, schema);
    buffer.set_cached_row_count_if_unset(42);
    assert_eq!(buffer.cached_row_count(), Some(42));
}

#[test]
fn cached_row_count_no_overwrite() {
    let device = CudaDevice::new(0).expect("CUDA device");
    let schema = Schema::new(vec![ScalarType::U32]);
    let budget = MemoryBudget::with_limit(1024 * 1024);
    let manager = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    let mut d_num_rows = manager.alloc::<u32>(1).unwrap();
    device.htod_sync_copy_into(&[42u32], &mut d_num_rows).unwrap();
    let buffer = CudaBuffer::from_columns(Vec::new(), 42, d_num_rows, schema);
    buffer.set_cached_row_count_if_unset(42);
    buffer.set_cached_row_count_if_unset(999); // should NOT overwrite
    assert_eq!(buffer.cached_row_count(), Some(42));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --lib -- cached_row_count --release 2>&1 | tail -20`
Expected: FAIL — `cached_row_count` method doesn't exist.

**Step 3: Implement the cached_row_count field and methods**

In `crates/xlog-cuda/src/memory.rs`:

a) Add import at top:
```rust
use std::sync::atomic::{AtomicU32, Ordering};
```

b) Add field to `CudaBuffer` struct (after `schema`):
```rust
pub struct CudaBuffer {
    pub columns: Vec<CudaColumn>,
    pub row_cap: u64,
    pub d_num_rows: TrackedCudaSlice<u32>,
    pub schema: Schema,
    /// Host-cached row count. u32::MAX = not yet cached.
    /// Immutable after first set (CAS from sentinel).
    cached_row_count: AtomicU32,
}
```

c) Update `from_columns` constructor to initialize with sentinel:
```rust
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
```

d) Add accessor methods after existing methods:
```rust
/// Return the cached host-side row count, or None if not yet cached.
pub fn cached_row_count(&self) -> Option<u32> {
    let v = self.cached_row_count.load(Ordering::Relaxed);
    if v == u32::MAX { None } else { Some(v) }
}

/// Set the cached row count if not already set. Returns the value
/// that is stored (either `count` if this call won, or the existing value).
pub fn set_cached_row_count_if_unset(&self, count: u32) -> u32 {
    match self.cached_row_count.compare_exchange(
        u32::MAX, count, Ordering::Relaxed, Ordering::Relaxed,
    ) {
        Ok(_) => count,
        Err(existing) => existing,
    }
}
```

e) Fix the struct literal test at line 499. Change:
```rust
let buffer = CudaBuffer {
    columns: Vec::new(),
    row_cap: 100,
    d_num_rows,
    schema: schema.clone(),
};
```
to use `from_columns`:
```rust
let buffer = CudaBuffer::from_columns(Vec::new(), 100, d_num_rows, schema.clone());
```

f) Fix the destructuring at `crates/xlog-cuda/src/provider.rs:1843`. Change:
```rust
let CudaBuffer {
    columns: combined_columns,
    row_cap,
    d_num_rows,
    schema: _,
} = combined;
```
to:
```rust
let CudaBuffer {
    columns: combined_columns,
    row_cap,
    d_num_rows,
    schema: _,
    ..
} = combined;
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p xlog-cuda --lib -- cached_row_count --release 2>&1 | tail -20`
Expected: 3 tests PASS.

Run full build check: `cargo build --workspace --release 2>&1 | tail -10`
Expected: Build succeeds (no broken struct literal/destructure sites).

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/memory.rs crates/xlog-cuda/src/provider.rs
git commit -m "feat(xlog-cuda): add AtomicU32 row-count cache to CudaBuffer"
```

---

### Task 2: Eagerly populate cache in `buffer_from_columns`

**Context:** `buffer_from_columns` (provider.rs:7158-7174) is the main construction helper called by all `create_buffer_from_*` methods. It already knows the row count on the host (`row_cap`). After construction, we eagerly set the cache.

**Files:**
- Modify: `crates/xlog-cuda/src/provider.rs:7158-7174`

**Step 1: Write the failing test**

Create `crates/xlog-cuda/tests/test_row_count_cache.rs`:

```rust
//! Test that CudaBuffer row count cache is populated eagerly on host-known construction paths.
use xlog_cuda::{CudaKernelProvider, Schema, ScalarType};

#[test]
fn buffer_from_u32_columns_has_cached_count() {
    let provider = CudaKernelProvider::new(0, 64).unwrap();
    let schema = Schema::new(vec![ScalarType::U32, ScalarType::U32]);
    let buffer = provider
        .create_buffer_from_u32_columns(&[vec![1, 2, 3], vec![4, 5, 6]], schema)
        .unwrap();
    assert_eq!(buffer.cached_row_count(), Some(3));
}

#[test]
fn empty_buffer_has_cached_count_zero() {
    let provider = CudaKernelProvider::new(0, 64).unwrap();
    let schema = Schema::new(vec![ScalarType::U32]);
    let buffer = provider.create_empty_buffer(schema).unwrap();
    assert_eq!(buffer.cached_row_count(), Some(0));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test test_row_count_cache --release 2>&1 | tail -20`
Expected: FAIL — `cached_row_count()` returns `None`.

**Step 3: Implement eager cache population**

In `crates/xlog-cuda/src/provider.rs`, modify `buffer_from_columns` (line 7158):

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
    let buf = CudaBuffer::from_columns(columns, row_cap, d_num_rows, schema);
    buf.set_cached_row_count_if_unset(row_u32);
    Ok(buf)
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test test_row_count_cache --release 2>&1 | tail -20`
Expected: 2 tests PASS.

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider.rs crates/xlog-cuda/tests/test_row_count_cache.rs
git commit -m "feat(xlog-cuda): eagerly populate row-count cache in buffer_from_columns"
```

---

### Task 3: Make `device_row_count` use the cache

**Context:** `device_row_count` (provider.rs:6969-6976) currently does a D2H transfer on every call. With the cache, it checks the buffer first and only does D2H on cold miss (populating the cache for future reads). We also make it `pub` so external callers can use it instead of rolling their own D2H.

**Files:**
- Modify: `crates/xlog-cuda/src/provider.rs:6969-6976`
- Test: `crates/xlog-cuda/tests/test_row_count_cache.rs` (extend)

**Step 1: Write the failing test**

Add to `crates/xlog-cuda/tests/test_row_count_cache.rs`:

```rust
#[test]
fn device_row_count_populates_cache_on_miss() {
    let provider = CudaKernelProvider::new(0, 64).unwrap();
    let schema = Schema::new(vec![ScalarType::U32]);
    // create_buffer_from_u32_columns should eagerly cache (from Task 2)
    let buffer = provider
        .create_buffer_from_u32_columns(&[vec![10, 20, 30]], schema)
        .unwrap();
    // Already cached from construction
    assert_eq!(buffer.cached_row_count(), Some(3));
    // device_row_count should return same value without D2H
    assert_eq!(provider.device_row_count(&buffer).unwrap(), 3);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test test_row_count_cache --release 2>&1 | tail -20`
Expected: FAIL — `device_row_count` is private.

**Step 3: Implement cached device_row_count**

In `crates/xlog-cuda/src/provider.rs`, change `device_row_count` from private to public and add cache logic:

```rust
/// Read the device-resident row count, using the host cache when available.
/// On cache miss, performs a single D2H of 4 bytes and populates the cache.
pub fn device_row_count(&self, buffer: &CudaBuffer) -> Result<usize> {
    if let Some(cached) = buffer.cached_row_count() {
        return Ok(cached as usize);
    }
    let mut host_rows = [0u32];
    self.dtoh_sync_copy_into_tracked(buffer.num_rows_device(), &mut host_rows)?;
    buffer.set_cached_row_count_if_unset(host_rows[0]);
    Ok(host_rows[0] as usize)
}
```

Note: this also routes the fallback through `dtoh_sync_copy_into_tracked` instead of raw `device.inner().dtoh_sync_copy_into`, which feeds the `HostTransferTracker`.

**Step 4: Run tests**

Run: `cargo test -p xlog-cuda --test test_row_count_cache --release 2>&1 | tail -20`
Expected: 3 tests PASS.

Run full workspace: `cargo build --workspace --release 2>&1 | tail -10`
Expected: Build succeeds.

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider.rs crates/xlog-cuda/tests/test_row_count_cache.rs
git commit -m "feat(xlog-cuda): device_row_count uses cache, falls back with tracking"
```

---

### Task 4: Route external row-count readers through cache

**Context:** Two external functions bypass the provider's `device_row_count`:
- `read_device_row_count` in `crates/xlog-runtime/src/ilp_registry.rs:9-20`
- `buffer_row_count` in `crates/xlog-runtime/src/executor.rs:2961-2968`

Both do raw `dtoh_sync_copy_into` — no cache, no tracking. Now that `device_row_count` is public, these should route through it or use `buffer.cached_row_count()` directly.

**Files:**
- Modify: `crates/xlog-runtime/src/ilp_registry.rs:9-20`
- Modify: `crates/xlog-runtime/src/executor.rs:2961-2968`

**Step 1: Rewrite `read_device_row_count`**

In `crates/xlog-runtime/src/ilp_registry.rs`, replace:

```rust
pub fn read_device_row_count(
    provider: &CudaKernelProvider,
    buffer: &CudaBuffer,
) -> Result<usize, XlogError> {
    let mut host_rows = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
        .map_err(|e| XlogError::Kernel(format!("Failed to read row count: {}", e)))?;
    Ok(host_rows[0] as usize)
}
```

with:

```rust
pub fn read_device_row_count(
    provider: &CudaKernelProvider,
    buffer: &CudaBuffer,
) -> Result<usize, XlogError> {
    provider.device_row_count(buffer)
}
```

**Step 2: Rewrite `buffer_row_count` in executor.rs**

In `crates/xlog-runtime/src/executor.rs`, replace the `buffer_row_count` method (line 2961):

```rust
fn buffer_row_count(&self, buffer: &CudaBuffer) -> Result<u32> {
    let mut host_rows = [0u32];
    self.provider
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
        .map_err(|e| XlogError::Execution(format!("Failed to read row count: {}", e)))?;
    Ok(host_rows[0])
}
```

with:

```rust
fn buffer_row_count(&self, buffer: &CudaBuffer) -> Result<u32> {
    self.provider.device_row_count(buffer)
        .map(|n| n as u32)
        .map_err(|e| XlogError::Execution(format!("Failed to read row count: {}", e)))
}
```

**Step 3: Build and run tests**

Run: `cargo build --workspace --release 2>&1 | tail -10`
Expected: Build succeeds.

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -20`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add crates/xlog-runtime/src/ilp_registry.rs crates/xlog-runtime/src/executor.rs
git commit -m "refactor: route external row-count readers through cached device_row_count"
```

---

### Task 5: Route remaining untracked D2H calls through tracked wrapper

**Context:** Several D2H call sites in provider.rs still use `self.device.inner().dtoh_sync_copy_into()` directly, bypassing the `HostTransferTracker`. Convert them to `self.dtoh_sync_copy_into_tracked()`. The main sites are:

- `membership_mask` mask download (line 8408-8410)
- `extract_active_rule_indices` count + arrays (lines 10714-10757)
- `download_column_*` functions (lines 6686-6958)
- `download_raw_bytes` (line 6958)
- Line 1653 (standalone D2H)

**Files:**
- Modify: `crates/xlog-cuda/src/provider.rs`

**Step 1: Write the gate test**

Add to `crates/xlog-cuda/tests/test_row_count_cache.rs`:

```rust
#[test]
fn host_transfer_stats_tracks_membership_mask() {
    let provider = CudaKernelProvider::new(0, 64).unwrap();
    let schema = Schema::new(vec![ScalarType::U32, ScalarType::U32]);
    let relation = provider
        .create_buffer_from_u32_columns(&[vec![1, 2], vec![3, 4]], schema.clone())
        .unwrap();
    let query = provider
        .create_buffer_from_u32_columns(&[vec![1], vec![3]], schema)
        .unwrap();

    provider.reset_host_transfer_stats();
    let _mask = provider.membership_mask(&query, &relation, &[0, 1], &[0, 1]).unwrap();
    let stats = provider.host_transfer_stats();
    // Should track the mask download (1 byte) — dtoh_bytes > 0
    assert!(stats.dtoh_bytes > 0, "membership_mask D2H not tracked: {:?}", stats);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test test_row_count_cache -- host_transfer_stats_tracks --release 2>&1 | tail -20`
Expected: FAIL — `dtoh_bytes` is 0 because `membership_mask` uses untracked D2H.

**Step 3: Convert untracked D2H sites**

In `crates/xlog-cuda/src/provider.rs`, perform these replacements:

a) **`membership_mask` mask download** (line 8408-8413):
Replace:
```rust
self.device
    .inner()
    .dtoh_sync_copy_into(&d_has_match, &mut host_mask)
    .map_err(|e| {
        XlogError::Kernel(format!("Failed to download membership mask: {}", e))
    })?;
```
With:
```rust
self.dtoh_sync_copy_into_tracked(&d_has_match, &mut host_mask)?;
```

b) **`extract_active_rule_indices` count** (lines 10714-10717):
Replace:
```rust
self.device()
    .inner()
    .dtoh_sync_copy_into(&count, &mut count_host)
    .map_err(|e| XlogError::Kernel(format!("ILP dtoh count: {}", e)))?;
```
With:
```rust
self.dtoh_sync_copy_into_tracked(&count, &mut count_host)?;
```

c) **`extract_active_rule_indices` i,j,k,p arrays** (lines 10742-10757):
Replace each of the 4 blocks:
```rust
self.device()
    .inner()
    .dtoh_sync_copy_into(&out_X_view, &mut X_host)
    .map_err(|e| XlogError::Kernel(format!("ILP dtoh X: {}", e)))?;
```
With:
```rust
self.dtoh_sync_copy_into_tracked(&out_X_view, &mut X_host)?;
```
(for X in {i, j, k, p})

d) **`download_column_*` functions** (lines 6686-6958 — 8 functions):
Each contains a pattern like:
```rust
self.device
    .inner()
    .dtoh_sync_copy_into(&col_view, &mut bytes)
    .map_err(...)?;
```
Replace with:
```rust
self.dtoh_sync_copy_into_tracked(&col_view, &mut bytes)?;
```

e) **Line 1653** (if it exists as a standalone D2H — verify and convert if present).

**Step 4: Run tests**

Run: `cargo test -p xlog-cuda --test test_row_count_cache -- host_transfer_stats_tracks --release 2>&1 | tail -20`
Expected: PASS — `dtoh_bytes > 0`.

Run full build: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -20`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider.rs crates/xlog-cuda/tests/test_row_count_cache.rs
git commit -m "refactor(xlog-cuda): route all D2H through tracked wrapper for complete profiling"
```

---

### Task 6: Expose `host_transfer_stats` via PyO3

**Context:** The `HostTransferStats` are already available on `CudaKernelProvider`. We expose them through `CompiledIlpProgram` so the trainer and tests can profile total D2H/H2D per step.

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`
- Test: `python/tests/test_ilp_d2h_gate.py` (extend)

**Step 1: Write the failing test**

Add to `python/tests/test_ilp_d2h_gate.py`:

```python
def test_host_transfer_stats_accessible():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    stats = prog.host_transfer_stats()
    assert "dtoh_calls" in stats
    assert "dtoh_bytes" in stats
    assert "htod_calls" in stats
    assert "htod_bytes" in stats


def test_host_transfer_stats_reset():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()
    stats1 = prog.host_transfer_stats()
    assert stats1["dtoh_calls"] > 0 or stats1["dtoh_bytes"] > 0
    prog.reset_host_transfer_stats()
    stats2 = prog.host_transfer_stats()
    assert stats2["dtoh_calls"] == 0
    assert stats2["dtoh_bytes"] == 0
```

**Step 2: Run test to verify it fails**

Run: `cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml && LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py::test_host_transfer_stats_accessible -v --timeout=60 2>&1 | tail -20`
Expected: FAIL — `host_transfer_stats` method doesn't exist.

**Step 3: Add PyO3 methods**

In `crates/pyxlog/src/lib.rs`, add to the `#[pymethods] impl CompiledIlpProgram` block (near the existing `d2h_transfer_count` methods):

```rust
/// Return host transfer statistics: {dtoh_calls, dtoh_bytes, htod_calls, htod_bytes}.
fn host_transfer_stats(&self) -> HashMap<String, u64> {
    let stats = self.provider.host_transfer_stats();
    let mut map = HashMap::new();
    map.insert("dtoh_calls".to_string(), stats.dtoh_calls);
    map.insert("dtoh_bytes".to_string(), stats.dtoh_bytes);
    map.insert("htod_calls".to_string(), stats.htod_calls);
    map.insert("htod_bytes".to_string(), stats.htod_bytes);
    map
}

/// Reset host transfer statistics counters to zero.
fn reset_host_transfer_stats(&self) {
    self.provider.reset_host_transfer_stats();
}
```

**Step 4: Build and run tests**

Run: `cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml && LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py -v --timeout=120 2>&1 | tail -30`
Expected: All d2h_gate tests PASS.

**Step 5: Commit**

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_ilp_d2h_gate.py
git commit -m "feat(pyxlog): expose host_transfer_stats via PyO3"
```

---

### Task 7: Add pre-uploaded fact buffer infrastructure

**Context:** `CompiledIlpProgram` needs a handle map, epoch tracking, and upload/drop methods. Handles are opaque `u64` IDs. All existing handles are dropped on epoch bump (schema change/recompile) to prevent GPU memory leaks.

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`
- Test: `python/tests/test_ilp_d2h_gate.py` (extend)

**Step 1: Write the failing test**

Add to `python/tests/test_ilp_d2h_gate.py`:

```python
def test_upload_fact_buffer_and_drop():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()
    handle = prog.upload_fact_buffer("edge", [[1, 2], [2, 3]])
    assert isinstance(handle, int)
    prog.drop_fact_buffer(handle)


def test_upload_fact_buffer_invalid_handle():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()
    with pytest.raises(Exception):
        prog.batch_fact_membership_preloaded(99999)
```

**Step 2: Run test to verify it fails**

Run: `cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml && LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py::test_upload_fact_buffer_and_drop -v --timeout=60 2>&1 | tail -20`
Expected: FAIL — `upload_fact_buffer` method doesn't exist.

**Step 3: Implement handle infrastructure**

In `crates/pyxlog/src/lib.rs`:

a) Add fields to `CompiledIlpProgram` struct:
```rust
fact_buffers: HashMap<u64, FactBufferEntry>,
next_fact_handle: u64,
compile_epoch: u64,
```

b) Add the entry struct (outside the impl block):
```rust
struct FactBufferEntry {
    relation: String,
    buffer: CudaBuffer,
    fact_count: usize,
    creation_epoch: u64,
}
```

c) Initialize in the `compile` constructor:
```rust
fact_buffers: HashMap::new(),
next_fact_handle: 0,
compile_epoch: 0,
```

d) Add PyO3 methods:
```rust
/// Upload facts to GPU and return a reusable handle.
fn upload_fact_buffer(&mut self, relation: &str, facts: Vec<Vec<i64>>) -> PyResult<u64> {
    // Verify relation exists
    let buf = self.executor.store().get(relation)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown relation: {}", relation),
        ))?;
    let schema = buf.schema().clone();
    let arity = schema.arity();

    if facts.is_empty() {
        let empty = self.provider.create_empty_buffer(schema)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let handle = self.next_fact_handle;
        self.next_fact_handle += 1;
        self.fact_buffers.insert(handle, FactBufferEntry {
            relation: relation.to_string(),
            buffer: empty,
            fact_count: 0,
            creation_epoch: self.compile_epoch,
        });
        return Ok(handle);
    }

    // Transpose row-major facts to columnar
    let num_facts = facts.len();
    let mut columns: Vec<Vec<u32>> = vec![vec![]; arity];
    for fact in &facts {
        if fact.len() != arity {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Fact arity {} != relation arity {}", fact.len(), arity),
            ));
        }
        for (col_idx, &val) in fact.iter().enumerate() {
            columns[col_idx].push(val as u32);
        }
    }

    let col_slices: Vec<Vec<u32>> = columns;
    let col_refs: Vec<&[u32]> = col_slices.iter().map(|c| c.as_slice()).collect();
    // Need to pass as slice of vecs to create_buffer_from_u32_columns
    let query_buf = self.provider
        .create_buffer_from_u32_columns(
            &col_slices.iter().map(|c| c.clone()).collect::<Vec<_>>(),
            schema,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let handle = self.next_fact_handle;
    self.next_fact_handle += 1;
    self.fact_buffers.insert(handle, FactBufferEntry {
        relation: relation.to_string(),
        buffer: query_buf,
        fact_count: num_facts,
        creation_epoch: self.compile_epoch,
    });
    Ok(handle)
}

/// Drop a pre-uploaded fact buffer.
fn drop_fact_buffer(&mut self, handle: u64) -> PyResult<()> {
    self.fact_buffers.remove(&handle)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Invalid fact buffer handle: {}", handle),
        ))?;
    Ok(())
}
```

**Step 4: Build and run tests**

Run: `cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml && LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py -v --timeout=120 2>&1 | tail -30`
Expected: New tests PASS.

**Step 5: Commit**

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_ilp_d2h_gate.py
git commit -m "feat(pyxlog): add upload_fact_buffer/drop_fact_buffer handle infrastructure"
```

---

### Task 8: Add `batch_fact_membership_preloaded` and `batch_tagged_credit_preloaded`

**Context:** These are variants of the existing batch APIs that accept a handle instead of re-uploading facts. They look up the pre-uploaded buffer and call the same underlying `membership_mask` logic.

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`
- Test: `python/tests/test_ilp_d2h_gate.py` (extend)

**Step 1: Write the failing test**

Add to `python/tests/test_ilp_d2h_gate.py`:

```python
def test_batch_fact_membership_preloaded_basic():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()
    handle = prog.upload_fact_buffer("edge", [[1, 2], [5, 6], [2, 3]])
    mask = prog.batch_fact_membership_preloaded(handle)
    assert mask == [True, False, True]
    prog.drop_fact_buffer(handle)


def test_batch_tagged_credit_preloaded_basic():
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

    handle = prog.upload_fact_buffer("reach", [[1, 3], [2, 4], [99, 99]])
    credits = prog.batch_tagged_credit_preloaded(handle)
    assert len(credits) == 3
    assert len(credits[0]) > 0
    assert credits[2] == []
    prog.drop_fact_buffer(handle)


def test_preloaded_no_extra_h2d():
    """Preloaded variants should not upload facts (zero H2D for fact data)."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()
    handle = prog.upload_fact_buffer("edge", [[1, 2], [2, 3]])
    prog.reset_host_transfer_stats()
    _ = prog.batch_fact_membership_preloaded(handle)
    stats = prog.host_transfer_stats()
    # Should have zero H2D (facts already uploaded)
    assert stats["htod_calls"] == 0, f"Unexpected H2D calls: {stats}"
    prog.drop_fact_buffer(handle)
```

**Step 2: Run test to verify it fails**

Run: `cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml && LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py::test_batch_fact_membership_preloaded_basic -v --timeout=60 2>&1 | tail -20`
Expected: FAIL — method doesn't exist.

**Step 3: Implement preloaded variants**

In `crates/pyxlog/src/lib.rs`, add to the `#[pymethods]` block:

```rust
/// Like batch_fact_membership, but uses a pre-uploaded fact buffer.
fn batch_fact_membership_preloaded(&self, handle: u64) -> PyResult<Vec<bool>> {
    let entry = self.fact_buffers.get(&handle)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Invalid fact buffer handle: {}", handle),
        ))?;
    if entry.creation_epoch != self.compile_epoch {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Stale fact buffer handle (program recompiled since upload)",
        ));
    }
    if entry.fact_count == 0 {
        return Ok(vec![]);
    }
    let relation = &entry.relation;
    let buf = self.executor.store().get(relation)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Relation '{}' not found in store", relation),
        ))?;
    let arity = buf.schema().arity();
    let keys: Vec<usize> = (0..arity).collect();
    let mask = self.provider
        .membership_mask(&entry.buffer, buf, &keys, &keys)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(mask)
}

/// Like batch_tagged_credit, but uses a pre-uploaded fact buffer.
fn batch_tagged_credit_preloaded(&self, handle: u64) -> PyResult<Vec<Vec<(u32, u32, u32)>>> {
    let entry = self.fact_buffers.get(&handle)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Invalid fact buffer handle: {}", handle),
        ))?;
    if entry.creation_epoch != self.compile_epoch {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Stale fact buffer handle (program recompiled since upload)",
        ));
    }
    let num_facts = entry.fact_count;
    if num_facts == 0 {
        return Ok(vec![]);
    }
    let relation = &entry.relation;

    // Get the ILP tagged result
    let tagged = self.executor.ilp_last_result()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "No ILP result available — call evaluate() first",
        ))?;

    // Find the relation's index
    let rel_idx = self.rel_index.iter()
        .find(|(_, name)| name == relation)
        .map(|(id, _)| *id)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Relation '{}' not found in ILP schema", relation),
        ))?;

    // Collect relevant entries for this relation's head
    let arity = entry.buffer.schema().arity();
    let keys: Vec<usize> = (0..arity).collect();
    let mut per_fact_credits: Vec<Vec<(u32, u32, u32)>> = vec![vec![]; num_facts];

    for tag_entry in tagged.entries() {
        if tag_entry.k != rel_idx as u32 {
            continue;
        }
        if let Some(ref entry_buf) = tag_entry.buffer {
            let mask = self.provider
                .membership_mask(&entry.buffer, entry_buf, &keys, &keys)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            for (fact_idx, &found) in mask.iter().enumerate() {
                if found {
                    per_fact_credits[fact_idx].push((tag_entry.i, tag_entry.j, tag_entry.k));
                }
            }
        }
    }

    Ok(per_fact_credits)
}
```

**Step 4: Build and run tests**

Run: `cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml && LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py -v --timeout=120 2>&1 | tail -30`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_ilp_d2h_gate.py
git commit -m "feat(pyxlog): add batch_fact_membership_preloaded and batch_tagged_credit_preloaded"
```

---

### Task 9: Rewrite trainer.py to use preloaded handles

**Context:** The trainer currently re-uploads facts on every `batch_fact_membership` and `batch_tagged_credit` call. With pre-uploaded handles, we upload once per attempt and reuse across all steps. Uses `try/finally` for RAII cleanup.

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`

**Step 1: Refactor `_run_single_attempt` to pre-upload facts**

In `trainer.py`, modify `_run_single_attempt`:

a) After `prog = pyxlog.IlpProgramFactory.compile(...)` (line 147) and before the step loop, add fact grouping and pre-upload:

```python
    # Group facts by relation for batch API calls
    pos_by_rel: dict[str, list[list[int]]] = {}
    pos_indices_by_rel: dict[str, list[int]] = {}
    for idx, (rel, vals) in enumerate(positives):
        pos_by_rel.setdefault(rel, []).append(vals)
        pos_indices_by_rel.setdefault(rel, []).append(idx)

    neg_by_rel: dict[str, list[list[int]]] = {}
    for rel, vals in negatives:
        neg_by_rel.setdefault(rel, []).append(vals)

    # Pre-upload fact buffers to GPU (once per attempt, reused across all steps)
    pos_handles: dict[str, int] = {}
    neg_handles: dict[str, int] = {}
    all_handles: list[int] = []
```

b) Wrap the step loop and everything after in `try/finally`:

```python
    try:
        for rel_name, facts_list in pos_by_rel.items():
            h = prog.upload_fact_buffer(rel_name, facts_list)
            pos_handles[rel_name] = h
            all_handles.append(h)
        for rel_name, facts_list in neg_by_rel.items():
            h = prog.upload_fact_buffer(rel_name, facts_list)
            neg_handles[rel_name] = h
            all_handles.append(h)

        # ... existing step loop ...

    finally:
        for h in all_handles:
            try:
                prog.drop_fact_buffer(h)
            except Exception:
                pass
```

c) In the step loop, replace `batch_fact_membership` calls with `batch_fact_membership_preloaded`:

Witness coverage (around line 249-251):
```python
        witness_mask = [False] * len(positives)
        for rel_name in pos_by_rel:
            mask = prog.batch_fact_membership_preloaded(pos_handles[rel_name])
            for local_idx, found in enumerate(mask):
                global_idx = pos_indices_by_rel[rel_name][local_idx]
                witness_mask[global_idx] = found
```

d) Pass handles to `_compute_loss` and `_check_convergence`:

```python
loss = _compute_loss(prog, M_soft, pos_handles, neg_handles, rel_names, n)
```

```python
converged = _check_convergence(
    prog, W, mask_name, pos_handles, neg_handles,
    rel_names, n, argmax_ijk,
)
```

e) Rewrite `_compute_loss` to use preloaded:

```python
def _compute_loss(
    prog,
    M_soft: torch.Tensor,
    pos_handles: dict[str, int],
    neg_handles: dict[str, int],
    rel_names: list[str],
    n: int,
) -> torch.Tensor:
    """Per-fact surrogate loss using GPU-side batch credit assignment."""
    loss = torch.tensor(0.0, device="cuda")

    for rel_name, handle in pos_handles.items():
        credits = prog.batch_tagged_credit_preloaded(handle)
        for contributing in credits:
            if contributing:
                credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
                loss = loss + (-torch.log(credit.clamp(min=1e-8)))
            else:
                k_idx = rel_names.index(rel_name)
                loss = loss + (-M_soft[:, :, k_idx].sum() / (n * n))

    for rel_name, handle in neg_handles.items():
        credits = prog.batch_tagged_credit_preloaded(handle)
        for contributing in credits:
            if contributing:
                credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
                loss = loss + (-torch.log((1.0 - credit).clamp(min=1e-8)))

    return loss
```

f) Rewrite `_check_convergence` to use preloaded:

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
    """Three-gated convergence using batch GPU APIs with preloaded buffers."""
    # Gate 1: all positives derived under current mask
    for rel_name, handle in pos_handles.items():
        mask = prog.batch_fact_membership_preloaded(handle)
        if not all(mask):
            return False

    # Gate 2: no negatives derived
    for rel_name, handle in neg_handles.items():
        mask = prog.batch_fact_membership_preloaded(handle)
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

    # Gate 3a: argmax-only must derive all positives
    for rel_name, handle in pos_handles.items():
        mask = prog.batch_fact_membership_preloaded(handle)
        if not all(mask):
            return False

    # Gate 4: argmax-only must not derive negatives
    for rel_name, handle in neg_handles.items():
        mask = prog.batch_fact_membership_preloaded(handle)
        if any(mask):
            return False

    return True
```

**Step 2: Build and run all non-slow ILP tests**

```bash
cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_*.py -v --timeout=300 -m "not slow" 2>&1 | tail -40
```

Expected: All non-slow ILP tests PASS.

**Step 3: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/trainer.py
git commit -m "feat(ilp): trainer uses preloaded fact buffers — zero per-step H2D uploads"
```

---

### Task 10: Add transfer profiling telemetry to trainer

**Context:** Add `host_transfer_stats` snapshot to per-step telemetry (when `telemetry_level >= 2`), using the "full step" window (reset before set_rule_mask, snapshot at end of step).

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/types.py` (extend StepRecord)

**Step 1: Add transfer fields to StepRecord**

In `crates/pyxlog/python/pyxlog/ilp/types.py`, check if `StepRecord` is a dataclass or namedtuple and add optional fields `d2h_calls` and `d2h_bytes` (default 0).

**Step 2: Add profiling in trainer step loop**

In `_run_single_attempt`, at the START of each step (before `optimizer.zero_grad()`), add:
```python
        prog.reset_host_transfer_stats()
```

In the telemetry block (`if config.telemetry_level >= 2`), add:
```python
            stats = prog.host_transfer_stats()
            # Include in StepRecord or log
```

**Step 3: Build and run tests**

```bash
cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_*.py -v --timeout=300 -m "not slow" 2>&1 | tail -40
```

**Step 4: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/trainer.py crates/pyxlog/python/pyxlog/ilp/types.py
git commit -m "feat(ilp): add per-step D2H transfer profiling to telemetry"
```

---

### Task 11: Run full reliability suite and verify

**Context:** Final verification that all optimizations preserve convergence behavior.

**Files:**
- No new files — verification only

**Step 1: Run reliability suite**

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_reliability.py -v --timeout=600 2>&1 | tail -30
```

Expected: 20/20 PASS.

**Step 2: Run full non-slow suite**

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_*.py -v --timeout=300 -m "not slow" 2>&1 | tail -30
```

Expected: All PASS.

**Step 3: Run Rust workspace tests**

```bash
cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -20
```

Expected: All PASS.
