# Wave 2: Provider Decomposition + GpuScalar Migration

**Date**: 2026-03-10
**Status**: Approved
**Depends on**: Wave 1 (error seams + GpuScalar trait)

## Overview

Split `crates/xlog-cuda/src/provider.rs` (12,809 lines, 163 methods) into focused
submodules using distributed `impl CudaKernelProvider` blocks. Simultaneously collapse
type-specialized function families via the `GpuScalar` trait introduced in Wave 1.

## Constraints

- Green at wave boundary: workspace + 206/206 + `cargo check -p pyxlog`
- Break internal inter-crate APIs freely (turbofish migration)
- Preserve zero data-plane D2H / GPU-residency contracts
- Preserve determinism/reproducibility
- Reuse existing `kernel_manifest_data.rs` — do not invent a second manifest
- Bool is a special case in H2D transfer (no `create_buffer_from_bool_slice` exists today)
- DLPack already has its own module (`dlpack.rs`) — io.rs covers Arrow/IPC only
- Kernel const exports are used outside xlog-cuda (xlog-runtime, xlog-prob, xlog-cuda-tests) — coordinate visibility changes with consumer updates

## 1. File Split: Distributed impl Blocks

```
crates/xlog-cuda/src/
├── provider.rs          → provider/mod.rs
├──                        provider/kernel_loading.rs
├──                        provider/relational.rs
├──                        provider/filter.rs
├──                        provider/groupby.rs
├──                        provider/arithmetic.rs
├──                        provider/transfer.rs
├──                        provider/probabilistic.rs
├──                        provider/ilp.rs
└──                        provider/io.rs
```

### Module Responsibilities

| Module | Methods (approx) | LOC est. | Content |
|--------|-------------------|----------|---------|
| `mod.rs` | 8–10 | ~800 | Struct definition, field accessors, `new()` (refactored), memory budget, d2h counter, device_row_count |
| `kernel_loading.rs` | 2–3 | ~200 | Consumes/extends `kernel_manifest_data.rs`, `load_all_kernel_modules()` helper, warmup profiling |
| `relational.rs` | ~20 | ~2,200 | hash_join (provider.rs:1845), hash_join_with_limit (provider.rs:1873), hash_join_v2 (provider.rs:7447), hash_join_v2_with_limit (provider.rs:7473), build_join_index_v2 (provider.rs:7495), hash_join_v2_with_index (provider.rs:7544), dedup (provider.rs:1998), dedup_sorted (provider.rs:2033), union (provider.rs:2217), diff (provider.rs:2341), union_gpu (provider.rs:2536), diff_gpu (provider.rs:2592), sort (provider.rs:3793), build_hash_table_u64 (provider.rs:8025), membership_mask (provider.rs:8658), membership_mask_device (provider.rs:8534). Also includes extract_column (provider.rs:10022), extract_active_rule_indices (provider.rs:11247). |
| `filter.rs` | ~4 | ~400 | Generic `filter<T: GpuScalar>()`, `compare_columns<T: GpuScalar>()`, mask composition (collapsed from 18 type-specialized fns) |
| `groupby.rs` | ~5 | ~700 | groupby_agg, groupby_multi_agg, count_distinct |
| `arithmetic.rs` | ~12 | ~600 | add/sub/mul/div/mod/abs/negate/pow/cast columns, binary/unary dispatch |
| `transfer.rs` | ~6 | ~350 | Generic `download_column<T: GpuScalar>()`, `create_buffer_from_slice<T: GpuScalar>()`, `dtoh_scalar_untracked`, batch helpers |
| `probabilistic.rs` | ~15 | ~1,000 | mc_sample_bernoulli, mc_eval_circuit, forward_backward, circuit build/cache, evidence forcing |
| `ilp.rs` | ~12 | ~900 | ILP credit/loss kernels, COO fill, CSR histogram, reduce_sum, chunk merge, count_mask_into_slot, batch_fact_membership, batch_tagged_credit |
| `io.rs` | ~10 | ~700 | from_arrow_record_batch, from_arrow_device_record_batch, Arrow IPC paths, schema conversion. DLPack stays in its own existing `dlpack.rs` module. |

**Total**: ~7,550 LOC (vs 12,809 current — ~41% reduction from deduplication and cleanup)

## 2. GpuScalar Migration: Collapsing Type-Specialized Families

### D1: download_column_<T> → single generic (in transfer.rs)

Before: 8 functions (~280 LOC) with 99% identical bodies.

After: 1 function (~35 LOC):
```rust
pub fn download_column<T: GpuScalar>(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<T>> {
    self.d2h_transfer_count.fetch_add(1, Ordering::Relaxed);
    let col = buffer.column(col_idx)
        .ok_or_else(|| XlogError::kernel_ctx("download_column", "column index out of bounds", &col_idx))?;
    let num_rows = self.device_row_count(buffer)?;
    if num_rows == 0 { return Ok(vec![]); }
    let num_bytes = num_rows.checked_mul(T::BYTE_WIDTH)
        .ok_or_else(|| XlogError::kernel_ctx("download_column", "byte size overflow", &num_rows))?;
    let col_view = self.column_bytes_view(col, num_bytes)?;
    let mut bytes = vec![0u8; num_bytes];
    self.device.inner().dtoh_sync_copy_into(&col_view, &mut bytes)
        .map_err(|e| XlogError::kernel_ctx("download_column", "dtoh copy failed", &e))?;
    Ok(bytes.chunks_exact(T::BYTE_WIDTH).map(|c| T::from_le_bytes(c)).collect())
}
```

Call-site migration: `provider.download_column_u32(buf, idx)` → `provider.download_column::<u32>(buf, idx)`.

### D2: create_buffer_from_<T>_slice → single generic (in transfer.rs)

Before: 7 functions (~220 LOC). After: 1 function (~30 LOC).

Same pattern: serialize via `GpuScalar::to_le_bytes_into()`, alloc, htod_sync_copy, buffer_from_columns.

**Bool special case**: No `create_buffer_from_bool_slice` exists today. The generic H2D
collapse excludes bool initially. If bool H2D is needed later, define explicit 0/1 encoding
semantics in a follow-up.

### D3: filter_<T> + filter_<T>_eq/gt/lt → generic + enum dispatch (in filter.rs)

Before: 11 functions (~1,200 LOC). The filter functions dispatch to type-specific CUDA kernel
names.

After: Generic with kernel-name lookup added to `GpuScalar`:
```rust
pub(crate) trait GpuScalar: ... {
    // ... existing methods from Wave 1 ...
    fn filter_kernel_name() -> &'static str;  // added in Wave 2
}
```

The `filter_<T>_eq()` / `filter_<T>_gt()` wrappers become unnecessary — callers use
`filter::<u32>(buf, col, CompareOp::Eq, val)`.

### D4: compare_columns_<T> → generic (in filter.rs)

Already delegates to `compare_columns_mask::<T>()` internally. Collapse to single public
entry point.

## 3. Refactoring new() (814 → ~120 lines)

Reuse and extend the existing `kernel_manifest_data.rs` as the single source of truth.
Today it only exposes `KERNEL_CU_NAMES` (kernel_manifest_data.rs:10) — an array of `.cu`
file stems used by `build.rs`. Wave 2 extends this file to include module constants and
kernel function name lists, so both `build.rs` and `kernel_loading.rs` consume one manifest.

```rust
// In kernel_manifest_data.rs (extended by Wave 2):
pub struct KernelModuleSpec {
    pub name: &'static str,           // e.g., "join"
    pub module_const: &'static str,   // e.g., JOIN_MODULE
    pub kernels: &'static [&'static str],
}

pub const KERNEL_MODULES: &[KernelModuleSpec] = &[
    KernelModuleSpec { name: "join", module_const: "join_mod", kernels: &[...] },
    // ... derived from current KERNEL_CU_NAMES + provider.rs const blocks
];

// Existing KERNEL_CU_NAMES stays for build.rs backward compat,
// or is derived from KERNEL_MODULES.
```

Then in `kernel_loading.rs`:

```rust
pub(crate) fn load_all_kernel_modules(
    device: &CudaDevice,
    cc: (u32, u32),
    profiling: bool,
) -> Result<()> {
    for spec in kernel_manifest_data::KERNEL_MODULES {
        let t0 = if profiling { Some(Instant::now()) } else { None };
        let (ptx, _is_cubin) = load_module_from_file(spec.name, cc)?;
        device.inner().load_ptx(ptx, spec.module_const, spec.kernels)
            .map_err(|e| XlogError::kernel_ctx("load_module", spec.name, &e))?;
        if let Some(t0) = t0 {
            // single timing block
        }
    }
    Ok(())
}
```

`new()` calls `load_all_kernel_modules(...)` — one line replacing 750+ lines of boilerplate.

## 4. Ride-Along Improvements

Applied as methods are relocated to submodules (zero marginal cost):

| Ride-along | Scope |
|------------|-------|
| **Visibility** | Internal helpers become `pub(crate)` instead of `pub`. Kernel const names that are used by external crates (xlog-runtime, xlog-prob, xlog-cuda-tests) stay `pub` — coordinate any tightening with consumer updates. |
| **Error context** | Replace `XlogError::Kernel(format!("Failed to X: {}", e))` with `XlogError::kernel_ctx(...)` as each function is relocated. |
| **Unwrap fixes** | Fix dangerous `.unwrap()` in production-path methods when touching that function. ~30 of the 90 provider.rs unwraps are in production paths. |
| **Clone reduction** | `schema.clone()` on empty-buffer returns — evaluate whether reference or pre-built schema is feasible as each function moves. |

## 5. Facade Stability

**Internal crate API**: Breaks freely. All `download_column_u32()` call sites updated to
`download_column::<u32>()`.

**External surface**: `xlog-cuda/src/lib.rs` re-exports stay identical. The `provider` module
path changes from `provider.rs` to `provider/mod.rs` — Rust treats these identically for
`use` statements. No downstream `use` path changes.

**Provider module direct imports**: External crates import kernel submodule constants and
kernel function-name modules directly from `xlog_cuda::provider::{...}`. The `provider/mod.rs`
must re-export the same namespace so all existing `use` paths continue to resolve.

| Consumer | Import location | What they import from `provider::` |
|----------|----------------|-----------------------------------|
| xlog-runtime/executor.rs | executor.rs:13 | `arith_kernels`, `filter_kernels`, `ARITH_MODULE`, `FILTER_MODULE` |
| xlog-gpu/gpu.rs | gpu.rs:8–10 | `arith_kernels`, `d4_kernels`, `filter_kernels`, `ARITH_MODULE`, `D4_MODULE`, `FILTER_MODULE` |
| xlog-solve/gpu_cdcl.rs | gpu_cdcl.rs:7 | `sat_kernels`, `SAT_MODULE` |
| xlog-prob/compilation/gpu_d4.rs | gpu_d4.rs:14 | `d4_kernels`, `scan_kernels`, `D4_MODULE`, `SCAN_MODULE` |
| xlog-prob/compilation/gpu_cnf.rs | gpu_cnf.rs:9 | `cnf_kernels`, `CNF_MODULE` |
| xlog-prob/compilation/gpu_cache.rs | gpu_cache.rs:8 | `cache_kernels`, `CACHE_MODULE` |
| xlog-prob/compilation/gpu_pir_intern.rs | gpu_pir_intern.rs:11 | `pir_kernels`, `scan_kernels`, `RadixSortScratch`, `PIR_MODULE`, `SCAN_MODULE` |
| xlog-prob/compilation/gpu_weights.rs | gpu_weights.rs:8 | `weights_kernels`, `WEIGHTS_MODULE` |
| xlog-prob/compilation/validation.rs | validation.rs:10–11 | `sat_kernels`, `SAT_MODULE` |
| xlog-prob/exact.rs | exact.rs:24–26 | `arith_kernels`, `filter_kernels`, `neural_kernels`, `weights_kernels`, `ARITH_MODULE`, `FILTER_MODULE`, `NEURAL_MODULE`, `WEIGHTS_MODULE` |
| xlog-prob/mc.rs | mc.rs:22 | `mc_eval_kernels`, `MC_EVAL_MODULE` |
| xlog-prob/gpu.rs | gpu.rs:8–10 | `arith_kernels`, `d4_kernels`, `filter_kernels`, `ARITH_MODULE`, `D4_MODULE`, `FILTER_MODULE` |

Note: pyxlog/lib.rs does NOT import from `xlog_cuda::provider::` — it uses crate-root
re-exports (`xlog_cuda::CudaKernelProvider`, `xlog_cuda::JoinType`, etc.).

The full set of kernel submodule names that must remain re-exported from `provider/mod.rs`:
`arith_kernels`, `cache_kernels`, `circuit_kernels`, `cnf_kernels`, `d4_kernels`,
`dedup_kernels`, `filter_kernels`, `groupby_kernels`, `ilp_kernels`, `join_kernels`,
`mc_eval_kernels`, `neural_kernels`, `pir_kernels`, `sat_kernels`, `scan_kernels`,
`sort_kernels`, `weights_kernels`, plus their corresponding `*_MODULE` constants and
`RadixSortScratch`.

## 6. Call-Site Update Scope

| Consumer crate | Estimated changes | Nature |
|---------------|-------------------|--------|
| xlog-runtime/executor.rs | ~40 turbofish additions | `download_column_u32` → `download_column::<u32>` |
| xlog-prob (mc.rs, exact.rs, gpu_d4.rs) | ~25 turbofish additions | Same pattern |
| xlog-solve/gpu_cdcl.rs | ~10 | Same |
| pyxlog/lib.rs | ~30 | Same |
| xlog-cuda-tests | ~20 | Same |
| xlog-cuda internal (provider tests) | ~15 | Same |

Total: ~140 call-site updates, all mechanical (type-specialized name → generic with turbofish).

## 7. Gate

- `cargo test --workspace --all-targets --exclude pyxlog --release` — green
- `cargo test -p xlog-cuda-tests --test certification_suite --release` — 206/206
- `cargo check -p pyxlog` — compile gate (Wave 2 includes mechanical pyxlog call-site rewrites)

No Python gates — Wave 2 doesn't change pyxlog's Python-facing API, only its Rust internals.

## 8. Diff Profile (estimated)

| Change type | Files | Lines added | Lines removed |
|-------------|-------|-------------|---------------|
| New provider submodules (9 files) | 9 | ~6,500 | — |
| Delete provider.rs (replaced by provider/) | 1 | — | ~12,809 |
| kernel_loading.rs (consumes kernel_manifest_data.rs) | 1 | ~200 | — |
| type_seam.rs additions (GpuScalar filter_kernel_name) | 1 | ~20 | — |
| Call-site turbofish updates | ~8 | ~140 | ~140 |
| New tests for generic functions | 2–3 | ~100 | — |
| **Net** | ~15 files | ~6,960 | ~12,949 |

**Net reduction: ~5,990 lines**

## 9. Risks

| Risk | Mitigation |
|------|-----------|
| `GpuScalar::filter_kernel_name()` ties type dispatch to string names | Kernel names are already string-dispatched today. No regression. |
| 140 mechanical turbofish edits are error-prone | `cargo test` catches type mismatches at compile time. Wrong type won't compile. |
| Moving 12,809 lines across 9 files risks accidental logic changes | Move first (pure cut/paste), then apply ride-alongs as separate logical steps within the same commit. |
| `groupby_multi_agg()` (523 lines) remains long after move | Complex state machine. Move as-is; consider internal decomposition in Wave 5 if warranted. |
