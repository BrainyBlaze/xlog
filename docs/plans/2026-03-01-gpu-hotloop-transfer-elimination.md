# GPU Hot-Loop Transfer Elimination Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate remaining redundant D2H/H2D transfers in the ILP trainer step loop, reducing per-step transfers from ~170 to ~40 and bytes from ~1,300 to ~600.

**Architecture:** Three optimizations: (1) lazy row-count cache on CudaBuffer eliminates ~130 redundant 4-byte D2H reads, (2) pre-uploaded fact buffers eliminate ~3 per-step H2D uploads, (3) route all D2H through tracked wrappers so existing HostTransferTracker provides complete profiling signal.

**Tech Stack:** Rust (xlog-cuda, xlog-runtime, pyxlog), Python (trainer.py), CUDA (no kernel changes)

---

## 1. Row Count Lazy Cache

### Problem
`device_row_count()` performs a synchronous `dtoh_sync_copy_into` of 4 bytes on every call. There are 49 call sites in provider.rs. During a training step with K=32 active rules, this produces ~130 D2H transfers — 76% of all per-step transfers.

The device-resident `d_num_rows: TrackedCudaSlice<u32>` on CudaBuffer is **immutable after construction**: no code ever writes to `d_num_rows` after `CudaBuffer::from_columns` returns (verified: no `.d_num_rows =` assignments post-construction, no `htod` writes to existing buffers' `d_num_rows`).

### Design

Add `cached_row_count: AtomicU32` to `CudaBuffer` with sentinel `u32::MAX` meaning "not yet cached":

```rust
// memory.rs
pub struct CudaBuffer {
    pub columns: Vec<CudaColumn>,
    pub row_cap: u64,
    pub d_num_rows: TrackedCudaSlice<u32>,
    pub schema: Schema,
    cached_row_count: AtomicU32,  // u32::MAX = unknown
}
```

**Accessors** (private field, public methods):
- `cached_row_count(&self) -> Option<u32>`: returns `Some(n)` if cached, `None` if sentinel
- `set_cached_row_count_if_unset(&self, count: u32)`: CAS from `u32::MAX` to `count`; no-op if already set

**Construction paths**:
- `from_columns(...)`: sets `cached_row_count: AtomicU32::new(u32::MAX)` (unknown)
- New `from_columns_with_host_count(...)`: sets `AtomicU32::new(count)` (eagerly populated)
- `buffer_from_columns` in provider.rs already knows `row_cap` → passes `Some(row_u32)` eagerly

**`device_row_count()` in provider.rs**:
```rust
fn device_row_count(&self, buffer: &CudaBuffer) -> Result<usize> {
    if let Some(cached) = buffer.cached_row_count() {
        return Ok(cached as usize);
    }
    let mut host_rows = [0u32];
    self.dtoh_sync_copy_into_tracked(&buffer.d_num_rows, &mut host_rows)?;
    buffer.set_cached_row_count_if_unset(host_rows[0]);
    Ok(host_rows[0] as usize)
}
```

**Bypass fixes** (all external row-count readers that bypass provider):
- `read_device_row_count` in `crates/xlog-runtime/src/ilp_registry.rs:9-20` — calls `provider.device().inner().dtoh_sync_copy_into()` directly. Must route through `buffer.cached_row_count()` with lazy fallback via public `provider.device_row_count()`.
- `buffer_row_count` in `crates/xlog-runtime/src/executor.rs:2961-2968` — same pattern, calls `provider.device().inner().dtoh_sync_copy_into()` directly. Must route through `buffer.cached_row_count()` with lazy fallback.

**Struct-literal and destructuring sites** that must be updated when adding the new field:
- `crates/xlog-cuda/src/memory.rs:499` — direct struct literal construction in tests (`CudaBuffer { columns, row_cap, d_num_rows, schema }`)
- `crates/xlog-cuda/src/provider.rs:1843` — destructuring pattern (`let CudaBuffer { columns, row_cap, d_num_rows, schema: _ } = combined`)
- All test files using struct literal syntax (search for `CudaBuffer {`)
- Strategy: add `cached_row_count: AtomicU32::new(u32::MAX)` in `from_columns`, then convert struct-literal sites to use `from_columns` or add `..` rest pattern for destructuring.

### Impact
- Eliminates ~130 of ~170 per-step D2H transfers
- Most buffers eagerly cached (host-known construction); device-computed buffers (compact, groupby) do 1 D2H on first read then cached forever
- Zero semantic behavior change

---

## 2. Pre-Uploaded Fact Buffers

### Problem
`batch_fact_membership` and `batch_tagged_credit` each re-upload query facts via `create_buffer_from_u32_columns` on every call. In a training step, the same fact sets are uploaded 3-5 times (witness coverage + loss + convergence checks), producing ~240 H2D bytes/step plus Python→Rust→GPU marshalling overhead.

### Design

**New Rust API on `CompiledIlpProgram`**:

```rust
/// Upload facts to GPU and return a reusable handle.
fn upload_fact_buffer(&mut self, relation: &str, facts: Vec<Vec<i64>>) -> PyResult<u64>

/// Like batch_fact_membership, but uses a pre-uploaded buffer.
fn batch_fact_membership_preloaded(&self, handle: u64) -> PyResult<Vec<bool>>

/// Like batch_tagged_credit, but uses a pre-uploaded buffer.
fn batch_tagged_credit_preloaded(&self, handle: u64) -> PyResult<Vec<Vec<(u32, u32, u32)>>>

/// Drop a pre-uploaded buffer to free GPU memory.
fn drop_fact_buffer(&mut self, handle: u64) -> PyResult<()>
```

**Storage**: `HashMap<u64, FactBufferEntry>` in `CompiledIlpProgram`:
```rust
struct FactBufferEntry {
    relation: String,
    buffer: CudaBuffer,
    fact_count: usize,
    creation_epoch: u64,  // for stale-state detection
}
```

**Handle validity**: Monotonic `next_handle: u64` counter. Handles are opaque u64 IDs.

**Epoch guard**: `compile_epoch: u64` on CompiledIlpProgram, incremented on recompile/schema-change/commit (NOT on `evaluate()` — fact buffers are valid across steps within the same attempt). `_preloaded` variants reject handles where `entry.creation_epoch != self.compile_epoch`. **On epoch bump, all existing handles are dropped** to prevent GPU memory leaks from accumulated stale buffers across repeated recompile cycles.

**Schema-aware upload**: Route through the relation's schema from `self.executor.store()`, not hardcoded u32. For alpha scope, the API accepts `Vec<Vec<i64>>` and encodes as integer-like columns only (U32, I32, I64) based on the relation schema. True symbol/term/float handling is deferred to beta when richer schemas are introduced — at that point the API type may need to change to support heterogeneous column types.

**RAII cleanup in Python** (`trainer.py`):
```python
def _run_single_attempt(...):
    prog = pyxlog.IlpProgramFactory.compile(...)
    handles = []
    try:
        # Pre-upload fact buffers once per attempt
        for rel_name, facts_list in pos_by_rel.items():
            h = prog.upload_fact_buffer(rel_name, facts_list)
            handles.append(h)
        ...
        # Use _preloaded variants in step loop
        ...
    finally:
        for h in handles:
            prog.drop_fact_buffer(h)
```

### Impact
- Eliminates ~3 H2D uploads per step (~240 bytes/step)
- Removes Python→Rust list→Vec→columnar marshalling from hot loop
- Explicit lifecycle with guaranteed cleanup

---

## 3. Complete D2H Transfer Tracking

### Problem
`HostTransferTracker` exists on CudaKernelProvider with `dtoh_calls`, `dtoh_bytes`, `htod_calls`, `htod_bytes` and reset/snapshot methods. Several call sites already use `dtoh_sync_copy_into_tracked` (join paths at lines 2344, 7931, 8107, 8802, 9171, etc.), but many hot-path sites still call `device.inner().dtoh_sync_copy_into()` directly, bypassing tracking — notably `device_row_count`, `membership_mask` mask download, `extract_active_rule_indices`, and the external `read_device_row_count` / `buffer_row_count` in xlog-runtime.

### Design

**Route all D2H through tracked wrapper**: Convert every `self.device.inner().dtoh_sync_copy_into()` call site in provider.rs to use `self.dtoh_sync_copy_into_tracked()` instead. Key sites:

| Path | Current | Change to |
|------|---------|-----------|
| `device_row_count` (line 6970) | `device.inner().dtoh_sync_copy_into` | `self.dtoh_sync_copy_into_tracked` (but most eliminated by cache) |
| `membership_mask` mask download (line 8406) | `device.inner().dtoh_sync_copy_into` | `self.dtoh_sync_copy_into_tracked` |
| `extract_active_rule_indices` (lines 10715, 10743-10757) | `device.inner().dtoh_sync_copy_into` | `self.dtoh_sync_copy_into_tracked` |
| `hash_join_inner_v2` probe count (line 7930) | `device.inner().dtoh_sync_copy_into` | `self.dtoh_sync_copy_into_tracked` |
| All other `device.inner().dtoh_sync_copy_into` sites | direct | tracked wrapper |

**External bypasses** (must also be routed through cache or tracked wrapper):
- `read_device_row_count` in `ilp_registry.rs:9-20` — after making `device_row_count` public (or using buffer cache), this becomes a thin wrapper or is eliminated.
- `buffer_row_count` in `executor.rs:2961-2968` — same pattern. Must route through `buffer.cached_row_count()` with lazy fallback.

**PyO3 exposure**:
```python
prog.host_transfer_stats()  # returns dict: {dtoh_calls, dtoh_bytes, htod_calls, htod_bytes}
prog.reset_host_transfer_stats()
```

**Trainer integration**:
- Existing `d2h_transfer_count` gate (column-only) stays for strict gating
- New `host_transfer_stats()` provides comprehensive profiling
- Log total D2H bytes in telemetry when `telemetry_level >= 2`

**Metric window definition**: The trainer currently resets `d2h_transfer_count` at line 206 (after `evaluate()`), so the column-download gate measures only the "post-evaluate hot loop" (loss + witness + convergence), not the evaluate phase itself. The `host_transfer_stats` profiling counter uses a different window: reset at the START of each step (before `set_rule_mask` + `evaluate`), snapshot at the END. This gives a "full step" view including evaluate-phase transfers. The expected-results table below uses the "full step" window for total D2H counts.

### Impact
- No new counters needed — reuse existing HostTransferTracker
- Complete visibility into all transfer types and bytes
- Enables data-driven optimization decisions for v0.5.0

---

## Expected Results

| Metric | Before P0 | After P0 | After This |
|--------|-----------|----------|------------|
| D2H transfers/step | ~500+ | ~170 | ~40 |
| D2H bytes/step | ~10,000+ | ~1,300 | ~600 |
| H2D uploads/step | ~15-25 | ~3 | 0 (pre-uploaded) |
| download_column_*/step | 15-25 | 0 (gated) | 0 (gated) |

Remaining ~40 D2H transfers in the "full step" window are from the evaluate phase (`extract_active_rule_indices` + per-rule probe counts + join output counts) and PyTorch scalar `.item()` calls. In the "post-evaluate" window (what the column-download gate measures), the remaining D2H is ~13 calls from `membership_mask` mask downloads only. Eliminating the evaluate-phase transfers requires a fully GPU-resident join dispatcher (v0.5.0).

---

## Testing Strategy

1. **Unit tests**: Row-count cache correctness (eager, lazy, device-computed paths)
2. **Integration tests**: Pre-uploaded buffer lifecycle (create, use, stale-handle rejection, cleanup)
3. **D2H gate**: Existing `d2h_transfer_count` gate unchanged (zero column downloads)
4. **Transfer profiling test**: Assert `host_transfer_stats().dtoh_calls` per step is ≤ expected threshold
5. **Reliability gate**: 20/20 alpha reliability suite passes unchanged
6. **Semantic parity**: Loss and convergence behavior identical (same test assertions)
