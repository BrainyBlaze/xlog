# GPU Hot-Loop Transfer Elimination Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce redundant D2H/H2D transfers in the ILP trainer step loop. This is an incremental transfer-reduction milestone, NOT a full GPU-resident step loop — the step loop still returns host data (`Vec<bool>`, `Vec<Vec<(i,j,k)>>`) and Python-side loss/convergence logic remains CPU-side. A fully GPU-resident step loop (zero host returns, GPU-side loss computation) is a v0.5.0 objective.

**Architecture:** Three optimizations: (1) lazy row-count cache on CudaBuffer eliminates ~130 redundant 4-byte D2H reads, (2) pre-uploaded fact buffers eliminate per-step H2D re-uploads, (3) route all host transfers (D2H and H2D) through tracked wrappers so existing HostTransferTracker provides complete profiling signal.

**Explicit non-goals for this milestone:**
- GPU-side loss computation (Python still loops over returned credit vectors)
- GPU-side convergence checking (Python still loops over returned membership masks)
- Eliminating evaluate-phase transfers (host-dispatched join loop stays)
- Eliminating PyTorch `.item()` scalar transfers (control-flow scalars stay)

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

**Route all transfers through tracked wrappers**: Convert every `self.device.inner().dtoh_sync_copy_into()` call site in provider.rs to use `self.dtoh_sync_copy_into_tracked()`. Similarly, add an `htod_sync_copy_into_tracked` wrapper (same pattern: increment `htod_calls`/`htod_bytes` on `HostTransferTracker`, then delegate) and route step-loop H2D paths through it. Key D2H sites:

| Path | Current | Change to |
|------|---------|-----------|
| `device_row_count` (line 6970) | `device.inner().dtoh_sync_copy_into` | `self.dtoh_sync_copy_into_tracked` (but most eliminated by cache) |
| `membership_mask` mask download (line 8406) | `device.inner().dtoh_sync_copy_into` | `self.dtoh_sync_copy_into_tracked` |
| `extract_active_rule_indices` (lines 10715, 10743-10757) | `device.inner().dtoh_sync_copy_into` | `self.dtoh_sync_copy_into_tracked` |
| `hash_join_inner_v2` probe count (line 7930) | `device.inner().dtoh_sync_copy_into` | `self.dtoh_sync_copy_into_tracked` |
| All other `device.inner().dtoh_sync_copy_into` sites | direct | tracked wrapper |

**REQUIRED: External bypass fixes** (these are blocking implementation work, not optional):
- `read_device_row_count` in `ilp_registry.rs:9-20` — calls `provider.device().inner().dtoh_sync_copy_into()` directly, bypassing both cache and tracking. MUST be rewritten to route through `buffer.cached_row_count()` with lazy fallback via provider. Without this fix, the row-count cache provides incomplete coverage and the "complete profiling signal" claim is invalid.
- `buffer_row_count` in `executor.rs:2961-2968` — same pattern, same requirement. MUST route through `buffer.cached_row_count()` with lazy fallback. Both bypasses must be fixed in the same task as the row-count cache and tracked-wrapper work.

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

**Full-step window** (reset before `set_rule_mask`, snapshot after convergence check):

| Metric | Before P0 | After P0 | After This |
|--------|-----------|----------|------------|
| D2H transfers/step | ~500+ | ~170 | E_eval + E_post + S |
| D2H bytes/step | ~10,000+ | ~1,300 | workload-dependent |
| H2D uploads/step | ~15-25 | ~3 | 0 (pre-uploaded) |
| download_column_*/step | 15-25 | 0 (gated) | 0 (gated) |

Where:
- **E_eval** = evaluate-phase D2H (fixed per step): `extract_active_rule_indices` (5 calls) + per-active-rule probe counts + join output counts. For K=32 active rules, E_eval ≈ 25.
- **E_post** = post-evaluate D2H (workload-dependent): `batch_tagged_credit` calls `membership_mask` once per contributing tagged entry. Each `membership_mask` call does 2× `device_row_count` (cached → 0 D2H) + 1× mask download. So E_post = number of contributing (i,j,k) entries across all positive facts. For the reach benchmark with 2 positives and 1 contributing rule: E_post ≈ 2. For the grandparent benchmark with 2 positives: E_post ≈ 2-4.
- **S** = PyTorch scalar `.item()` calls (fixed per step): ~10 calls for loss/entropy/temperature. **Note:** These are CUDA runtime transfers outside xlog's provider, so they are NOT counted by `host_transfer_stats()`. The S term appears in the "total transfers" model but must be excluded from `host_transfer_stats`-based test thresholds.

**Post-evaluate window** (what the column-download gate measures): E_post + witness `batch_fact_membership` mask downloads. The `batch_fact_membership` calls do 1× `membership_mask` per relation, so post-evaluate D2H ≈ E_post + R_pos + R_neg (where R_pos/R_neg = number of distinct relations with positives/negatives).

**Concrete benchmarks** (K=32 active rules):

| Workload | E_eval | E_post | Tracked D2H/step | S (untracked) |
|----------|--------|--------|-------------------|----------------|
| reach (2 pos, 0 neg) | ~25 | ~2 | ~27 | ~10 |
| grandparent (2 pos, 2 neg) | ~25 | ~4 | ~29 | ~10 |
| colleague (4 pos, 0 neg) | ~25 | ~4 | ~29 | ~10 |

"Tracked D2H/step" = E_eval + E_post (what `host_transfer_stats().dtoh_calls` reports). S is outside provider tracking.

Remaining D2H in the full-step window is dominated by the evaluate phase (~25 calls from `extract_active_rule_indices` + probe/output counts). Eliminating these requires a fully GPU-resident join dispatcher (v0.5.0). In the post-evaluate window, remaining D2H is E_post + R mask downloads only.

---

## Testing Strategy

1. **Unit tests**: Row-count cache correctness (eager, lazy, device-computed paths)
2. **Integration tests**: Pre-uploaded buffer lifecycle (create, use, stale-handle rejection, cleanup)
3. **D2H gate**: Existing `d2h_transfer_count` gate unchanged (zero column downloads)
4. **Transfer profiling test** (concrete thresholds — all thresholds use `host_transfer_stats()` which only counts provider-tracked transfers, NOT PyTorch `.item()` or other CUDA runtime transfers):
   - **Post-evaluate window**: Assert `host_transfer_stats().dtoh_calls` ≤ `E_post + R_pos + R_neg + 2` per step, where E_post = number of contributing tagged entries, R_pos/R_neg = distinct relations with positives/negatives. For the reach benchmark (2 pos, 0 neg, 1 relation): threshold = 2 + 1 + 0 + 2 = **5**.
   - **Full-step window**: Assert `host_transfer_stats().dtoh_calls` ≤ `E_eval + E_post + 5` (5 = margin for device-computed buffers' first-read lazy cache misses). S is excluded because PyTorch `.item()` calls are outside provider tracking. For reach: threshold = 25 + 2 + 5 = **32**.
   - **Tracked H2D uploads**: Assert `host_transfer_stats().htod_calls` == **0** in the step loop body (all facts pre-uploaded before the loop). This assertion covers only provider-tracked H2D paths. Direct `htod_sync_copy_into` calls that bypass the tracked wrapper (e.g., `buffer_from_columns` at provider.rs:7169) are outside tracking scope. To make this assertion sound, the H2D tracking wrapper (`htod_sync_copy_into_tracked`) must also be added as part of Section 3's "route all transfers through tracked wrappers" work — same pattern as the D2H wrapper conversion. Untracked H2D paths that are NOT called during the step loop (only during construction/compilation) do not affect the assertion.
   - Test uses the reach benchmark with known positive counts so thresholds are deterministic. If actual counts exceed thresholds, the test fails with a diagnostic message showing the breakdown.
5. **Reliability gate**: 20/20 alpha reliability suite passes unchanged
6. **Semantic parity**: Loss and convergence behavior identical (same test assertions)
