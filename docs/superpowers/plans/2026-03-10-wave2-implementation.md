# Wave 2: Provider Decomposition + GpuScalar Migration — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `crates/xlog-cuda/src/provider/mod.rs` (12,809 lines, 163 methods) into focused submodules and collapse type-specialized function families into generics using the `GpuScalar` trait from Wave 1.

**Architecture:** Distributed `impl CudaKernelProvider` blocks across 10 submodules. Kernel loading consolidated via extended `kernel_manifest_data.rs`. Four type-specialized families (download_column, create_buffer_from_slice, filter, compare_columns) replaced with generic functions using turbofish syntax at call sites. Old functions coexist during migration, then are removed.

**Tech Stack:** Rust workspace, cudarc (DeviceRepr, LaunchAsync), xlog-core (ScalarType, Result, XlogError)

**Spec:** `docs/superpowers/specs/2026-03-10-wave2-provider-decomposition-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Rename | `provider.rs` → `provider/mod.rs` | Struct def, fields, new() (refactored), memory/d2h accessors, kernel consts+modules |
| Create | `provider/kernel_loading.rs` | `load_all_kernel_modules()` consuming extended `kernel_manifest_data.rs` |
| Create | `provider/transfer.rs` | Generic `download_column::<T>()`, `create_buffer_from_slice::<T>()`, untracked variant |
| Create | `provider/filter.rs` | Generic `filter::<T>()`, `compare_columns::<T>()`, mask helpers |
| Create | `provider/relational.rs` | Joins, dedup, union, diff, sort, extract_column, membership_mask |
| Create | `provider/groupby.rs` | groupby_agg, groupby_multi_agg, count_distinct |
| Create | `provider/arithmetic.rs` | Column arithmetic: add/sub/mul/div/mod/abs/negate/pow/cast |
| Create | `provider/probabilistic.rs` | MC sample/eval, forward/backward, circuit build/cache |
| Create | `provider/ilp.rs` | ILP credit/loss kernels, COO fill, CSR histogram, batch ops |
| Create | `provider/io.rs` | Arrow IPC, from_arrow_record_batch, schema conversion |
| Modify | `kernel_manifest_data.rs` | Add `KernelModuleSpec` struct + `KERNEL_MODULES` array |
| Modify | `type_seam.rs` | Add filter/compare kernel name methods to `GpuScalar` |
| Modify | `lib.rs` (xlog-cuda) | Module declaration update (provider.rs → provider/) |
| Modify | `executor.rs` (xlog-runtime) | ~1 turbofish migration |
| Modify | `lib.rs` (pyxlog) | ~28 turbofish migrations |
| Modify | xlog-prob test files | ~6 turbofish migrations |
| Modify | xlog-cuda test files | ~30+ turbofish migrations |
| Modify | xlog-cuda-tests src files | ~100+ turbofish migrations |

All paths relative to `crates/xlog-cuda/src/` unless otherwise noted.

---

## Chunk 1: Provider Directory Structure + Kernel Loading Refactor

### Task 1: Rename provider.rs → provider/mod.rs

Pure structural change. Convert the single file into a module directory. No code changes.

**Files:**
- Rename: `crates/xlog-cuda/src/provider/mod.rs` → `crates/xlog-cuda/src/provider/mod.rs`

- [ ] **Step 1: Create provider directory and move file**

```bash
cd crates/xlog-cuda/src
mkdir -p provider
git mv provider.rs provider/mod.rs
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p xlog-cuda`
Expected: PASS (Rust treats `provider.rs` and `provider/mod.rs` identically for module paths)

- [ ] **Step 3: Verify workspace**

Run: `cargo check --workspace --exclude pyxlog`
Expected: PASS (all `use xlog_cuda::provider::*` paths unchanged)

- [ ] **Step 4: Commit**

```bash
git add -A crates/xlog-cuda/src/provider/
git commit -m "refactor(xlog-cuda): rename provider.rs to provider/mod.rs for Wave 2 split"
```

---

### Task 2: Extend kernel_manifest_data.rs + create kernel_loading.rs + refactor new()

Replace 812-line `new()` with a data-driven loop over `KERNEL_MODULES`. This is the single largest LOC reduction in Wave 2.

**Files:**
- Modify: `crates/xlog-cuda/src/kernel_manifest_data.rs`
- Create: `crates/xlog-cuda/src/provider/kernel_loading.rs`
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (refactor `new()`, lines 763–1574)

**Context:**
- `kernel_manifest_data.rs` currently has `KERNEL_CU_NAMES: &[&str]` (21 entries) consumed by `build.rs` via `include!()`.
- `new()` (mod.rs:763–1574) loads 21 kernel modules with identical per-module boilerplate: resolve PTX/cubin → load → register kernels → optionally profile timing.
- The 21 `pub const *_MODULE` constants are at mod.rs:182–202.
- The 21 `pub mod *_kernels` blocks are at mod.rs:208–557.

- [ ] **Step 1: Write test for kernel loading**

Add a test in `crates/xlog-cuda/tests/` or within `kernel_loading.rs` (if feasible without GPU) that verifies the `KERNEL_MODULES` array matches `KERNEL_CU_NAMES` in count and names. This catches any data entry errors.

```rust
// In kernel_manifest_data.rs, add at bottom:
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_modules_matches_cu_names() {
        assert_eq!(
            KERNEL_MODULES.len(),
            KERNEL_CU_NAMES.len(),
            "KERNEL_MODULES and KERNEL_CU_NAMES must have same count"
        );
        for (i, spec) in KERNEL_MODULES.iter().enumerate() {
            assert_eq!(
                spec.cu_name, KERNEL_CU_NAMES[i],
                "Mismatch at index {}: spec.cu_name={} vs KERNEL_CU_NAMES={}",
                i, spec.cu_name, KERNEL_CU_NAMES[i]
            );
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --lib -- kernel_manifest_data::tests`
Expected: FAIL (KERNEL_MODULES not yet defined)

- [ ] **Step 3: Add KernelModuleSpec to kernel_manifest_data.rs**

Extend `crates/xlog-cuda/src/kernel_manifest_data.rs` with:

```rust
/// Specification for a single kernel module: PTX file stem, Rust module constant name,
/// and the list of kernel function names to register.
pub struct KernelModuleSpec {
    /// The .cu file stem (matches entries in KERNEL_CU_NAMES), e.g., "join"
    pub cu_name: &'static str,
    /// The module constant string used for `load_ptx()`, e.g., "xlog_join"
    pub module_name: &'static str,
    /// Kernel function names to register from this module
    pub kernels: &'static [&'static str],
}
```

Then add `KERNEL_MODULES: &[KernelModuleSpec]` with 21 entries. Derive entries from the existing `new()` function at mod.rs:763–1574. Each entry's `module_name` maps to the `*_MODULE` constant value, and `kernels` lists the function names from the corresponding `*_kernels` module.

**Derivation guide** (cross-reference mod.rs constants at lines 182–202 and kernel name modules at lines 208–557):

| cu_name | module_name | Kernel names source |
|---------|-------------|---------------------|
| `"join"` | `"xlog_join"` | `join_kernels` (mod.rs:370) |
| `"dedup"` | `"xlog_dedup"` | `dedup_kernels` (mod.rs:384) |
| `"groupby"` | `"xlog_groupby"` | `groupby_kernels` (mod.rs:392) |
| `"scan"` | `"xlog_scan"` | `scan_kernels` (mod.rs:409) |
| `"sort"` | `"xlog_sort"` | `sort_kernels` (mod.rs:422) |
| `"filter"` | `"xlog_filter"` | `filter_kernels` (mod.rs:447) |
| `"set_ops"` | `"xlog_set_ops"` | `set_ops_kernels` (mod.rs:482) |
| `"pack"` | `"xlog_pack"` | `pack_kernels` (mod.rs:489) |
| `"circuit"` | `"xlog_circuit"` | `circuit_kernels` (mod.rs:517) |
| `"mc_sample"` | `"xlog_mc_sample"` | `mc_sample_kernels` (mod.rs:208) |
| `"mc_eval"` | `"xlog_mc_eval"` | `mc_eval_kernels` (mod.rs:213) |
| `"arith"` | `"xlog_arith"` | `arith_kernels` (mod.rs:221) |
| `"sat"` | `"xlog_sat"` | `sat_kernels` (mod.rs:557) |
| `"d4"` | `"xlog_d4"` | `d4_kernels` (mod.rs:335) |
| `"neural"` | `"xlog_neural"` | `neural_kernels` (mod.rs:251) |
| `"pir"` | `"xlog_pir"` | `pir_kernels` (mod.rs:275) |
| `"cnf"` | `"xlog_cnf"` | `cnf_kernels` (mod.rs:295) |
| `"cache"` | `"xlog_cache"` | `cache_kernels` (mod.rs:545) |
| `"weights"` | `"xlog_weights"` | `weights_kernels` (mod.rs:312) |
| `"ilp"` | `"xlog_ilp"` | `ilp_kernels` (mod.rs:257) |
| `"ilp_credit"` | `"xlog_ilp_credit"` | `ilp_credit_kernels` (mod.rs:266) |

For each entry, read the corresponding `pub mod *_kernels` block in mod.rs to get the exact kernel function name constants. Collect them into the `kernels` array as string literals matching the constant *values* (not the constant names).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --lib -- kernel_manifest_data::tests`
Expected: PASS

- [ ] **Step 5: Create kernel_loading.rs**

Create `crates/xlog-cuda/src/provider/kernel_loading.rs`:

```rust
//! Kernel module loading — consumes kernel_manifest_data::KERNEL_MODULES to
//! load all PTX/cubin modules into the CUDA device context.

use std::sync::Arc;
use std::time::Instant;
use xlog_core::{Result, XlogError};
use crate::CudaDevice;
use crate::kernel_manifest_data::{KernelModuleSpec, KERNEL_MODULES};
use super::PtxLoadProfile;

impl super::CudaKernelProvider {
    /// Load all kernel modules from PTX/cubin files into the device context.
    /// Replaces the 812-line inline loading in the old new() constructor.
    pub(crate) fn load_all_kernel_modules(
        device: &Arc<CudaDevice>,
        profiling: bool,
    ) -> Result<Option<PtxLoadProfile>> {
        let total_t0 = if profiling { Some(Instant::now()) } else { None };
        let mut per_module_sec = Vec::new();
        let mut cubin_loaded = 0u32;
        let mut ptx_fallback = 0u32;

        let cc = super::detect_compute_capability(device)?;

        for spec in KERNEL_MODULES {
            let t0 = if profiling { Some(Instant::now()) } else { None };

            let (ptx, is_cubin) = super::load_module_from_file(spec.cu_name, cc)
                .map_err(|e| XlogError::kernel_ctx(
                    "load_all_kernel_modules",
                    &format!("failed to load module '{}'", spec.cu_name),
                    &e,
                ))?;

            if is_cubin {
                cubin_loaded += 1;
            } else {
                ptx_fallback += 1;
            }

            device
                .inner()
                .load_ptx(ptx, spec.module_name, spec.kernels)
                .map_err(|e| XlogError::kernel_ctx(
                    "load_all_kernel_modules",
                    &format!("failed to register module '{}'", spec.module_name),
                    &e,
                ))?;

            if let Some(t0) = t0 {
                per_module_sec.push((spec.cu_name.to_string(), t0.elapsed().as_secs_f64()));
            }
        }

        if let Some(total_t0) = total_t0 {
            Ok(Some(PtxLoadProfile {
                total_sec: total_t0.elapsed().as_secs_f64(),
                per_module_sec,
                cubin_loaded,
                ptx_fallback,
            }))
        } else {
            Ok(None)
        }
    }
}
```

**Note:** `detect_compute_capability` and `load_module_from_file` are private module-level functions in `provider/mod.rs` (lines 36 and 101 respectively). Since `kernel_loading.rs` is a child module of `provider/`, it can access them via `super::`. `load_module_from_file(name, cc)` returns `Result<(Ptx, bool)>` where `bool` indicates cubin (true) vs portable PTX (false). `detect_compute_capability(&device)` returns `Result<u32>` (two-digit SM number like 75, 80, 120).

- [ ] **Step 6: Declare kernel_loading submodule**

Add to `crates/xlog-cuda/src/provider/mod.rs` near the top (after the existing `use` block):

```rust
mod kernel_loading;
```

- [ ] **Step 7: Refactor new() to use load_all_kernel_modules()**

Replace the body of `new()` (mod.rs:763–1574) with:

```rust
pub fn new(device: Arc<CudaDevice>, memory: Arc<GpuMemoryManager>) -> Result<Self> {
    let profiling = warmup_profiling_enabled();
    let ptx_load_profile = Self::load_all_kernel_modules(&device, profiling)?;

    Ok(Self {
        device,
        memory,
        transfer_tracker: HostTransferTracker::default(),
        ptx_load_profile,
        d2h_transfer_count: AtomicU64::new(0),
    })
}
```

This replaces ~810 lines with ~10 lines. Verify that `HostTransferTracker` has a `default()` or equivalent constructor by checking the existing `new()` body.

- [ ] **Step 8: Verify compilation and tests**

Run: `cargo test -p xlog-cuda --release`
Expected: PASS

Run: `cargo check --workspace --exclude pyxlog`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add crates/xlog-cuda/src/kernel_manifest_data.rs crates/xlog-cuda/src/provider/kernel_loading.rs crates/xlog-cuda/src/provider/mod.rs
git commit -m "refactor(xlog-cuda): data-driven kernel loading via KERNEL_MODULES, collapse new() from 812 to ~10 lines"
```

---

### Task 3: Create transfer.rs with generic download_column\<T\> and create_buffer_from_slice\<T\>

Replace 8 `download_column_*` functions (~280 LOC) with one generic (~35 LOC) and 7 `create_buffer_from_*_slice` functions (~220 LOC) with one generic (~30 LOC).

**Files:**
- Create: `crates/xlog-cuda/src/provider/transfer.rs`
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (declare submodule)
- Modify: `crates/xlog-cuda/src/type_seam.rs` (no changes needed — existing GpuScalar methods suffice)

**Context:**
- All 8 `download_column_*` functions (mod.rs:6874–7210) follow identical structure: increment d2h_transfer_count → get column → get row count → allocate bytes → dtoh_sync_copy → deserialize via from_le_bytes chunks. Only the type and BYTE_WIDTH differ.
- `download_f64_untracked` (mod.rs:7020) differs: no d2h_transfer_count increment, uses `dtoh_sync_copy_into_tracked()` for stats recording.
- All 7 `create_buffer_from_*_slice` functions (mod.rs:5961–6147) follow identical structure: serialize via to_le_bytes → alloc → htod_sync_copy → buffer_from_columns. u8 variant is slightly simpler (no serialization).
- `create_buffer_from_u32_columns` (mod.rs:5983) is a multi-column variant — keep as-is or generalize separately.
- `create_buffer_from_slices` (mod.rs:6166) is already generic (byte-level) — keep as-is.
- GpuScalar trait already provides `BYTE_WIDTH`, `from_le_bytes()`, `to_le_bytes_into()` for all 8 types.

- [ ] **Step 1: Write tests for generic download_column and create_buffer_from_slice**

These need a CUDA device so should be `#[cfg(test)]` within transfer.rs or in an integration test. Write tests that exercise the generic with at least u32, f64, and bool types:

```rust
#[cfg(test)]
mod tests {
    // Tests require CUDA device — will be validated by the certification suite.
    // Verify that download_column::<u32>() returns the same result as
    // download_column_u32() for a known buffer. Same for f64 and bool.
}
```

Since these are GPU tests, the primary validation is that the workspace test suite and certification suite pass after migration. Add a focused test if a non-GPU unit test is feasible (e.g., testing the byte serialization path).

- [ ] **Step 2: Create transfer.rs with generic functions**

Create `crates/xlog-cuda/src/provider/transfer.rs`:

```rust
//! Host ↔ device transfer operations.
//!
//! Generic versions of download_column and create_buffer_from_slice,
//! replacing the 15 type-specialized functions with 2 generics.

use std::sync::atomic::Ordering;
use xlog_core::{Result, Schema, XlogError};
use crate::CudaBuffer;
use crate::type_seam::GpuScalar;

impl super::CudaKernelProvider {
    /// Download a single column from GPU to host as `Vec<T>`.
    ///
    /// Replaces: download_column_u32, download_column_u64, download_column_i32,
    /// download_column_i64, download_column_f32, download_column_f64,
    /// download_column_u8, download_column_bool.
    ///
    /// Increments the D2H transfer counter (gate-tracked).
    pub fn download_column<T: GpuScalar>(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<T>> {
        self.d2h_transfer_count.fetch_add(1, Ordering::Relaxed);
        self.download_column_inner::<T>(buffer, col_idx)
    }

    /// Download a column WITHOUT incrementing the D2H transfer counter.
    /// Records in transfer_tracker for profiling stats but not for the D2H gate.
    ///
    /// Replaces: download_f64_untracked (now generic over T).
    pub fn download_column_untracked<T: GpuScalar>(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<T>> {
        // Use dtoh_sync_copy_into_tracked for stats recording without d2h_transfer_count
        let col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::kernel_ctx("download_column_untracked", "column not found", &col_idx))?;

        let num_rows = self.device_row_count(buffer)?;
        if num_rows == 0 {
            return Ok(vec![]);
        }

        let num_bytes = num_rows
            .checked_mul(T::BYTE_WIDTH)
            .ok_or_else(|| XlogError::kernel_ctx("download_column_untracked", "byte size overflow", &num_rows))?;
        let col_view = self.column_bytes_view(col, num_bytes)?;
        let mut bytes = vec![0u8; num_bytes];
        self.dtoh_sync_copy_into_tracked(&col_view, &mut bytes)?;

        Ok(bytes.chunks_exact(T::BYTE_WIDTH).map(|c| T::from_le_bytes(c)).collect())
    }

    /// Shared implementation for both tracked and untracked column downloads.
    fn download_column_inner<T: GpuScalar>(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<T>> {
        let col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::kernel_ctx("download_column", "column not found", &col_idx))?;

        let num_rows = self.device_row_count(buffer)?;
        if num_rows == 0 {
            return Ok(vec![]);
        }

        let num_bytes = num_rows
            .checked_mul(T::BYTE_WIDTH)
            .ok_or_else(|| XlogError::kernel_ctx("download_column", "byte size overflow", &num_rows))?;
        let col_view = self.column_bytes_view(col, num_bytes)?;
        let mut bytes = vec![0u8; num_bytes];
        self.device
            .inner()
            .dtoh_sync_copy_into(&col_view, &mut bytes)
            .map_err(|e| XlogError::kernel_ctx("download_column", "dtoh copy failed", &e))?;

        Ok(bytes.chunks_exact(T::BYTE_WIDTH).map(|c| T::from_le_bytes(c)).collect())
    }

    /// Upload a typed slice as a single-column GPU buffer.
    ///
    /// Replaces: create_buffer_from_u32_slice, create_buffer_from_u64_slice,
    /// create_buffer_from_i32_slice, create_buffer_from_i64_slice,
    /// create_buffer_from_f32_slice, create_buffer_from_f64_slice,
    /// create_buffer_from_u8_slice.
    pub fn create_buffer_from_slice<T: GpuScalar>(&self, data: &[T], schema: Schema) -> Result<CudaBuffer> {
        let num_bytes = data.len()
            .checked_mul(T::BYTE_WIDTH)
            .ok_or_else(|| XlogError::kernel_ctx("create_buffer_from_slice", "byte size overflow", &data.len()))?;

        let mut bytes = vec![0u8; num_bytes];
        for (i, val) in data.iter().enumerate() {
            let offset = i * T::BYTE_WIDTH;
            val.to_le_bytes_into(&mut bytes[offset..offset + T::BYTE_WIDTH]);
        }

        let mut col = self.memory.alloc::<u8>(bytes.len())?;
        self.device
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .map_err(|e| XlogError::kernel_ctx("create_buffer_from_slice", "htod copy failed", &e))?;

        self.buffer_from_columns(vec![col.into()], data.len() as u64, schema)
    }
}
```

**Important:** Verify that `download_column_inner` matches the exact pattern of existing `download_column_u32` (mod.rs:6874–6899). The `column_bytes_view`, `device_row_count`, and `buffer_from_columns` helpers are private methods on `CudaKernelProvider` in mod.rs — they're accessible from transfer.rs because it uses `impl super::CudaKernelProvider`.

The `download_column_untracked` uses `dtoh_sync_copy_into_tracked` (mod.rs:1615) which records in transfer_tracker for stats but does NOT increment d2h_transfer_count. This matches the existing `download_f64_untracked` behavior.

- [ ] **Step 3: Declare transfer submodule**

Add to `crates/xlog-cuda/src/provider/mod.rs`:

```rust
mod transfer;
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p xlog-cuda`
Expected: PASS (new generics coexist with old type-specialized functions)

- [ ] **Step 5: Run tests**

Run: `cargo test -p xlog-cuda --release`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/xlog-cuda/src/provider/transfer.rs crates/xlog-cuda/src/provider/mod.rs
git commit -m "feat(xlog-cuda): add generic download_column<T> and create_buffer_from_slice<T> in transfer.rs"
```

---

## Chunk 2: Filter/Compare Generics + Method Relocation

### Task 4: Extend GpuScalar + create filter.rs with generic compare_columns\<T\> and filter\<T\>

Replace 7 `compare_columns_*` wrapper functions and 11 `filter_*` functions with generics.

**Files:**
- Modify: `crates/xlog-cuda/src/type_seam.rs` (add kernel name methods to GpuScalar)
- Create: `crates/xlog-cuda/src/provider/filter.rs`
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (declare submodule)

**Context:**
- All 7 `compare_columns_*` functions (mod.rs:5054–5177) are thin wrappers around `compare_columns_mask::<T>()` (mod.rs:5259–5353). Each passes type-specific (allowed_types, kernel_name) args. Collapse is clean.
- Filter functions have two structural variants:
  - **Fused-scan path** (filter_u32 at mod.rs:4844, filter_f64 at mod.rs:5413): ~115 lines each, use compare+scan+compact in one multi-phase kernel sequence.
  - **Mask+compact path** (filter_i32 at mod.rs:4961, filter_u64:4984, filter_f32:5007, filter_bool:5030): ~20 lines each, use `compare_const_mask<T>()` + `filter_by_device_mask()`.
- `compare_const_mask<T>` (mod.rs:5179–5257) is already generic internally, parameterized by `T: DeviceRepr + Copy`.
- `compare_columns_mask<T>` (mod.rs:5259–5353) is already generic internally.
- Wrapper functions `filter_u32_eq/gt`, `filter_f64_eq/gt/lt` (mod.rs:4814–4832, 5367–5412) are thin delegation to `filter_u32`/`filter_f64` with hardcoded `CompareOp`. Used only in test code — will be removed.

- [ ] **Step 1: Add kernel name methods to GpuScalar**

Extend `crates/xlog-cuda/src/type_seam.rs`. Add these methods to the trait:

```rust
pub(crate) trait GpuScalar: cudarc::driver::DeviceRepr + Copy + Send + 'static {
    // ... existing methods ...

    /// Kernel function name for const-compare mask generation.
    /// Used by the generic filter<T>() mask-based path.
    fn filter_compare_kernel() -> &'static str;

    /// Kernel function name for column-column comparison mask.
    fn compare_col_kernel() -> &'static str;

    /// ScalarType variants accepted for this type in filter/compare operations.
    fn allowed_scalar_types() -> &'static [ScalarType];

    /// Optional fused compare+scan kernel (phase 1). Only u32 and f64 have optimized
    /// fused-scan paths. Returns None for types that use the mask+compact path.
    fn filter_scan_phase1_kernel() -> Option<&'static str> { None }
}
```

Then implement for each type. Cross-reference the kernel name constants from the `filter_kernels` module (mod.rs:447–480) and the existing type-specialized filter functions:

| Type | filter_compare_kernel | compare_col_kernel | allowed_scalar_types | filter_scan_phase1_kernel |
|------|----------------------|-------------------|---------------------|--------------------------|
| u32 | `FILTER_COMPARE_U32` | `FILTER_COMPARE_U32_COL` | `[U32, Symbol]` | `Some(FILTER_COMPARE_U32_SCAN_PHASE1)` |
| u64 | `FILTER_COMPARE_U64` | `FILTER_COMPARE_U64_COL` | `[U64]` | `None` |
| i32 | `FILTER_COMPARE_I32` | `FILTER_COMPARE_I32_COL` | `[I32]` | `None` |
| i64 | `FILTER_COMPARE_I64` | `FILTER_COMPARE_I64_COL` | `[I64]` | `None` |
| f32 | `FILTER_COMPARE_F32` | `FILTER_COMPARE_F32_COL` | `[F32]` | `None` |
| f64 | `FILTER_COMPARE_F64` | `FILTER_COMPARE_F64_COL` | `[F64]` | `Some(FILTER_COMPARE_F64_SCAN_PHASE1)` |
| u8 | `FILTER_COMPARE_U8` | `FILTER_COMPARE_U8_COL` | `[Bool]` | `None` |
| bool | `FILTER_COMPARE_U8` | `FILTER_COMPARE_U8_COL` | `[Bool]` | `None` |

**Important:** Read the `filter_kernels` module (mod.rs:447–480) to get the exact constant names and verify which types have fused-scan kernels. The table above is derived from the existing filter function implementations. u32 uses `FILTER_COMPARE_U32_SCAN_PHASE1`; verify f64 has an analogous `FILTER_COMPARE_F64_SCAN_PHASE1` by reading the filter_kernels module.

**Note on f32:** A `FILTER_COMPARE_F32_SCAN_PHASE1` kernel constant exists in `filter_kernels` (mod.rs:457) and is loaded at provider/mod.rs (pre-Wave-2 line 963), but the current `filter_f32` function (mod.rs:5007) does NOT use the fused-scan path — it uses the mask+compact path via `compare_const_mask`. The plan maps f32 to `None` for `filter_scan_phase1_kernel()` to match the current runtime behavior. If a future optimization wants to wire up the fused-scan path for f32, that can be done by changing the trait impl.

**Note on i64:** No `filter_i64` function exists in the current codebase, but `FILTER_COMPARE_I64` and `FILTER_COMPARE_I64_COL` kernel constants exist. The generic `filter::<i64>()` will work via the mask+compact path, providing new functionality that wasn't previously exposed.

Note that `ScalarType` is from xlog-core — import it in type_seam.rs:
```rust
use xlog_core::ScalarType;
```

- [ ] **Step 2: Add GpuScalar tests for new methods**

Add tests to `type_seam.rs` verifying kernel names are non-empty strings and allowed_scalar_types is non-empty for each type.

- [ ] **Step 3: Create filter.rs with generic compare_columns\<T\>**

Create `crates/xlog-cuda/src/provider/filter.rs`. Start with the simpler collapse:

```rust
//! Filter and comparison operations.
//!
//! Generic versions of compare_columns and filter, replacing
//! the type-specialized function families.

use xlog_core::Result;
use crate::CudaBuffer;
use crate::memory::TrackedCudaSlice;
use crate::type_seam::GpuScalar;
use super::CompareOp;

impl super::CudaKernelProvider {
    /// Compare two columns of the same type, returning a device mask.
    ///
    /// Replaces: compare_columns_u32, compare_columns_i32, compare_columns_i64,
    /// compare_columns_u64, compare_columns_f32, compare_columns_f64, compare_columns_u8.
    pub fn compare_columns<T: GpuScalar>(
        &self,
        input: &CudaBuffer,
        left: usize,
        right: usize,
        op: CompareOp,
    ) -> Result<TrackedCudaSlice<u8>> {
        self.compare_columns_mask::<T>(
            input,
            left,
            right,
            op,
            T::allowed_scalar_types(),
            T::compare_col_kernel(),
        )
    }
}
```

- [ ] **Step 4: Add generic filter\<T\> to filter.rs**

The generic filter dispatches between fused-scan (u32, f64) and mask+compact paths:

```rust
impl super::CudaKernelProvider {
    /// Filter rows where column `col` compares to `value` per `op`.
    ///
    /// Replaces: filter_u32, filter_u64, filter_i32, filter_f32, filter_f64,
    /// filter_bool. Also replaces wrappers: filter_u32_eq, filter_u32_gt,
    /// filter_f64_eq, filter_f64_gt, filter_f64_lt.
    pub fn filter<T: GpuScalar>(
        &self,
        input: &CudaBuffer,
        col: usize,
        value: T,
        op: CompareOp,
    ) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema.clone());
        }

        if let Some(scan_kernel) = T::filter_scan_phase1_kernel() {
            // Fused compare+scan+compact path (u32, f64)
            self.filter_fused_scan::<T>(input, col, value, op, scan_kernel)
        } else {
            // Mask + compact path (i32, u64, f32, bool, u8)
            let mask = self.compare_const_mask::<T>(
                input,
                col,
                value,
                op,
                T::allowed_scalar_types(),
                T::filter_compare_kernel(),
            )?;
            self.filter_by_device_mask(input, &mask)
        }
    }
}
```

The `filter_fused_scan` helper is extracted from the current `filter_u32` body (mod.rs:4844–4958). It must be made generic over `T: GpuScalar`. Key changes from the u32-specific version:
- Use `T::BYTE_WIDTH` instead of `std::mem::size_of::<u32>()`
- Use `T::allowed_scalar_types()` for type validation
- The fused scan kernel receives the column and value as typed parameters — verify that `cudarc::driver::LaunchAsync` can handle generic `T: DeviceRepr` values

**Important:** Verify that `filter_f64` (mod.rs:5413) uses the same fused-scan structure as `filter_u32` before generalizing. If they differ structurally, keep them as separate specializations within filter.rs rather than trying to share a generic path.

- [ ] **Step 5: Move compare_const_mask and compare_columns_mask to filter.rs**

Move the private helpers `compare_const_mask` (mod.rs:5179–5257) and `compare_columns_mask` (mod.rs:5259–5353) from mod.rs to filter.rs. These are already generic (`<T: DeviceRepr + Copy>` and `<T: DeviceRepr>` respectively) and are only called by filter/compare methods.

Cut from mod.rs, paste into filter.rs within an `impl super::CudaKernelProvider` block.

- [ ] **Step 6: Declare filter submodule and verify**

Add to mod.rs: `mod filter;`

Run: `cargo check -p xlog-cuda`
Expected: PASS

Run: `cargo test -p xlog-cuda --release`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/xlog-cuda/src/type_seam.rs crates/xlog-cuda/src/provider/filter.rs crates/xlog-cuda/src/provider/mod.rs
git commit -m "feat(xlog-cuda): add generic filter<T> and compare_columns<T> in filter.rs"
```

---

### Task 5: Relocate remaining methods to submodules

Move methods from mod.rs to 6 domain submodules. This is pure relocation — no logic changes, no signature changes.

**Files:**
- Create: `crates/xlog-cuda/src/provider/relational.rs` (~2,200 LOC)
- Create: `crates/xlog-cuda/src/provider/groupby.rs` (~700 LOC)
- Create: `crates/xlog-cuda/src/provider/arithmetic.rs` (~600 LOC)
- Create: `crates/xlog-cuda/src/provider/probabilistic.rs` (~1,000 LOC)
- Create: `crates/xlog-cuda/src/provider/ilp.rs` (~900 LOC)
- Create: `crates/xlog-cuda/src/provider/io.rs` (~700 LOC)
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (declare submodules, shrink by ~6,100 LOC)

**Context — method-to-module mapping:**

Use the spec (Section 1) for the definitive mapping. Key method groups:

| Module | Methods to move | Source lines (approx) |
|--------|----------------|----------------------|
| `relational.rs` | `hash_join` (1845), `hash_join_with_limit` (1873), `hash_join_v2` (7447), `hash_join_v2_with_limit` (7473), `build_join_index_v2` (7495), `hash_join_v2_with_index` (7544), `dedup` (1998), `dedup_sorted` (2033), `union` (2217), `diff` (2341), `union_gpu` (2536), `diff_gpu` (2592), `sort` (3793), `build_hash_table_u64` (8025), `membership_mask` (8658), `membership_mask_device` (8534), `extract_column` (10022), `extract_active_rule_indices` (11247) | ~2,200 |
| `groupby.rs` | `groupby_agg`, `groupby_multi_agg` (523 lines — move as-is), `count_distinct` | ~700 |
| `arithmetic.rs` | Column arithmetic methods: add, sub, mul, div, mod, abs, negate, pow, cast | ~600 |
| `probabilistic.rs` | `mc_sample_bernoulli`, `mc_eval_circuit`, `forward_backward` (f32 + f64 variants), circuit build/cache methods | ~1,000 |
| `ilp.rs` | ILP credit/loss kernels, `coo_fill_*`, `csr_histogram`, `reduce_sum`, `chunk_merge_*`, `count_mask_into_slot`, `batch_fact_membership`, `batch_tagged_credit` | ~900 |
| `io.rs` | `from_arrow_record_batch`, `from_arrow_device_record_batch`, Arrow IPC, schema conversion | ~700 |

**Procedure for each submodule:**

1. Create the file with module doc comment
2. Add `use` imports (copy relevant imports from mod.rs top)
3. Open `impl super::CudaKernelProvider { ... }` block
4. Cut methods from mod.rs, paste into the new file
5. Move any helper functions that are ONLY used by methods in this submodule
6. Add `mod <name>;` to mod.rs

**What stays in mod.rs:**
- `CudaKernelProvider` struct definition (lines 687–698)
- Field accessor methods: `device()`, `memory()`, `d2h_transfer_count()`, etc.
- `device_row_count()` (line 7200) — used by many submodules
- `column_bytes_view()` (line 7244) — used by transfer + others
- `column_as_u32_view()` (line 7292) — used by filter + relational
- `buffer_from_columns()` (line 7393) — used by transfer + relational + io
- `buffer_from_columns_with_device_count()` (line 7231)
- `create_empty_buffer()` (line 7384)
- `filter_by_device_mask()` (line 5851) — used by filter + relational
- `compact_buffer_*` helpers (lines 5585–5851)
- `multiblock_scan_u32_inplace()` (line 3606) — used by filter + relational
- `dtoh_sync_copy_into_tracked()` (line 1615) — used by transfer
- `dtoh_scalar_untracked()` (line 1637) — used by probabilistic + ilp
- Kernel module constants (`pub const *_MODULE`, `pub mod *_kernels`)
- `PtxLoadProfile` struct definition
- `RawCudaView` struct (line 114) — internal helper used across submodules
- `RadixSortScratch` struct (line ~170) — `pub`, imported by external crates (xlog-prob)
- Module-level private functions: `warmup_profiling_enabled()`, `detect_compute_capability()`, `resolve_module_path()`, `load_module_from_file()`, `CompareOp` enum

- [ ] **Step 1: Create relational.rs**

Create the file, move the 18 relational methods listed above. Include any private helpers that are exclusively called by these methods. Grep for call sites of each private helper before moving — if a helper is shared across submodules, leave it in mod.rs.

- [ ] **Step 2: Create groupby.rs**

Move groupby methods. `groupby_multi_agg` is 523 lines — move as-is per spec.

- [ ] **Step 3: Create arithmetic.rs**

Move column arithmetic methods.

- [ ] **Step 4: Create probabilistic.rs**

Move MC sample/eval methods and forward/backward.

- [ ] **Step 5: Create ilp.rs**

Move ILP credit/loss methods.

- [ ] **Step 6: Create io.rs**

Move Arrow IPC and schema conversion methods.

- [ ] **Step 7: Declare all submodules in mod.rs**

Add to mod.rs:
```rust
mod relational;
mod groupby;
mod arithmetic;
mod probabilistic;
mod ilp;
mod io;
```

- [ ] **Step 8: Verify compilation**

Run: `cargo check -p xlog-cuda`
Expected: PASS

Run: `cargo check --workspace --exclude pyxlog`
Expected: PASS (external import paths unchanged — all methods still on `CudaKernelProvider`)

- [ ] **Step 9: Run tests**

Run: `cargo test -p xlog-cuda --release`
Expected: PASS

Run: `cargo test --workspace --all-targets --exclude pyxlog --release`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add crates/xlog-cuda/src/provider/
git commit -m "refactor(xlog-cuda): relocate provider methods to 6 domain submodules"
```

---

### Task 6: Re-export verification + provider/mod.rs cleanup

Verify that all external import paths still resolve after the structural split.

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (if re-exports needed)

**Context — external consumers of provider internals:**

These crates import kernel submodule constants directly from `xlog_cuda::provider::*`:
- xlog-runtime (arith_kernels, filter_kernels, ARITH_MODULE, FILTER_MODULE)
- xlog-solve (sat_kernels, SAT_MODULE)
- xlog-prob (d4_kernels, scan_kernels, cnf_kernels, cache_kernels, pir_kernels, weights_kernels, arith_kernels, filter_kernels, neural_kernels, mc_eval_kernels + corresponding MODULE constants + RadixSortScratch)

All kernel constants and `pub mod *_kernels` blocks remain in mod.rs, so existing imports should work unchanged. This task verifies that.

- [ ] **Step 1: Full workspace compilation check**

Run: `cargo check --workspace`
Expected: PASS (including pyxlog — no provider signature changes yet)

If any import path breaks, add re-exports to mod.rs:
```rust
// If submodule items need re-exporting (unlikely for this split):
pub use relational::SomeType;
```

- [ ] **Step 2: Verify certification suite compiles**

Run: `cargo check -p xlog-cuda-tests`
Expected: PASS

- [ ] **Step 3: Commit if any re-exports were added**

```bash
git add crates/xlog-cuda/src/provider/mod.rs
git commit -m "fix(xlog-cuda): add re-exports for provider submodule split"
```

---

## Chunk 3: Call-Site Migration + Cleanup

### Task 7: Migrate download_column and create_buffer_from call sites

Replace all type-specialized function calls with turbofish generic equivalents across the workspace.

**File discovery strategy:** Use grep to find ALL call sites. The spec estimates ~140 call sites but the actual count is higher — test files (xlog-cuda/tests/ and xlog-cuda-tests/) contain the majority. The grep command in Step 2 is the authoritative discovery mechanism.

**Known high-traffic files** (non-exhaustive — grep will find the complete list):
- `crates/xlog-runtime/src/executor.rs` (~1 call)
- `crates/pyxlog/src/lib.rs` (~30 calls)
- `crates/xlog-runtime/tests/executor_config_tests.rs` (~2 calls)
- `crates/xlog-prob/tests/neural_fast_path.rs` (~6 calls)
- `crates/xlog-cuda/tests/` — filter_tests.rs, join_collision_test.rs, arrow_tests.rs, type_coverage_tests.rs, groupby_tests.rs, set_ops_tests.rs, join_v2_tests.rs, sort_tests.rs, test_d2h_counter.rs
- `crates/xlog-cuda-tests/src/` — properties.rs + all 25 category files (c01 through c25)
- `crates/xlog-gpu/` — logic.rs and other files
- `crates/xlog-logic/tests/` — integration test files
- `examples/sort_debug.rs`

**Migration pattern:**

| Old call | New call |
|----------|----------|
| `provider.download_column_u32(buf, idx)` | `provider.download_column::<u32>(buf, idx)` |
| `provider.download_column_u64(buf, idx)` | `provider.download_column::<u64>(buf, idx)` |
| `provider.download_column_i32(buf, idx)` | `provider.download_column::<i32>(buf, idx)` |
| `provider.download_column_i64(buf, idx)` | `provider.download_column::<i64>(buf, idx)` |
| `provider.download_column_f32(buf, idx)` | `provider.download_column::<f32>(buf, idx)` |
| `provider.download_column_f64(buf, idx)` | `provider.download_column::<f64>(buf, idx)` |
| `provider.download_column_u8(buf, idx)` | `provider.download_column::<u8>(buf, idx)` |
| `provider.download_column_bool(buf, idx)` | `provider.download_column::<bool>(buf, idx)` |
| `provider.download_f64_untracked(buf, idx)` | `provider.download_column_untracked::<f64>(buf, idx)` |
| `provider.create_buffer_from_u32_slice(data, schema)` | `provider.create_buffer_from_slice::<u32>(data, schema)` |
| `provider.create_buffer_from_u64_slice(data, schema)` | `provider.create_buffer_from_slice::<u64>(data, schema)` |
| `provider.create_buffer_from_i32_slice(data, schema)` | `provider.create_buffer_from_slice::<i32>(data, schema)` |
| `provider.create_buffer_from_i64_slice(data, schema)` | `provider.create_buffer_from_slice::<i64>(data, schema)` |
| `provider.create_buffer_from_f32_slice(data, schema)` | `provider.create_buffer_from_slice::<f32>(data, schema)` |
| `provider.create_buffer_from_f64_slice(data, schema)` | `provider.create_buffer_from_slice::<f64>(data, schema)` |
| `provider.create_buffer_from_u8_slice(data, schema)` | `provider.create_buffer_from_slice::<u8>(data, schema)` |

**Note:** `create_buffer_from_u32_columns` and `create_buffer_from_slices` are NOT part of this migration — they have distinct semantics (multi-column and raw-byte respectively) and stay as-is.

- [ ] **Step 1: Migrate source code call sites**

Update `crates/xlog-runtime/src/executor.rs` (1 call at line ~3124) and `crates/pyxlog/src/lib.rs` (~28 calls — 20 download + 1 untracked + 7+ create_buffer_from).

Use find-and-replace per file. For pyxlog, the download_column calls appear in 4 blocks at lines ~5249, ~5327, ~6079, ~6138 (matching ScalarType arms). The create_buffer_from_f64_slice calls appear around lines ~3343–3513.

- [ ] **Step 2: Migrate test code call sites**

Update all test files listed above. Use `cargo check -p xlog-cuda --tests` and `cargo check -p xlog-cuda-tests` to find remaining old call sites.

**Strategy:** Run `cargo check` iteratively — each old function will generate a "method not found" error after removal, so the compiler is the migration checklist.

Wait — the old functions still exist at this point. Instead, grep for old function names across the workspace to find remaining call sites:

```bash
rg 'download_column_u32|download_column_u64|download_column_i32|download_column_i64|download_column_f32|download_column_f64|download_column_u8|download_column_bool|download_f64_untracked|create_buffer_from_u32_slice|create_buffer_from_u64_slice|create_buffer_from_i32_slice|create_buffer_from_i64_slice|create_buffer_from_f32_slice|create_buffer_from_f64_slice|create_buffer_from_u8_slice' --type rust -l
```

Then update each file. Exclude `provider/mod.rs` (where the old definitions live — they'll be removed in Task 9).

- [ ] **Step 3: Verify compilation**

Run: `cargo check --workspace`
Expected: PASS (old and new functions coexist)

- [ ] **Step 4: Run workspace tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release`
Expected: PASS

- [ ] **Step 5: Commit**

Stage all modified files found by the grep (exclude provider/mod.rs where old definitions live):
```bash
# Stage all crate source + test files that were modified (verify list with git diff --name-only)
git diff --name-only | xargs git add
git commit -m "refactor: migrate download_column and create_buffer_from call sites to turbofish generics"
```

---

### Task 8: Migrate filter and compare_columns call sites

**File discovery:** Use grep to find all call sites (same strategy as Task 7):
```bash
rg 'filter_u32|filter_u64|filter_i32|filter_f32|filter_f64|filter_bool|compare_columns_u32|compare_columns_i32|compare_columns_i64|compare_columns_u64|compare_columns_f32|compare_columns_f64|compare_columns_u8' --type rust -l
```

Exclude `provider/mod.rs` and `provider/filter.rs` (where old definitions live).

**Migration pattern:**

| Old call | New call |
|----------|----------|
| `provider.filter_u32(input, col, val, op)` | `provider.filter::<u32>(input, col, val, op)` |
| `provider.filter_u32_eq(input, col, val)` | `provider.filter::<u32>(input, col, val, CompareOp::Eq)` |
| `provider.filter_u32_gt(input, col, val)` | `provider.filter::<u32>(input, col, val, CompareOp::Gt)` |
| `provider.filter_f64(input, col, val, op)` | `provider.filter::<f64>(input, col, val, op)` |
| `provider.filter_f64_eq(input, col, val)` | `provider.filter::<f64>(input, col, val, CompareOp::Eq)` |
| `provider.filter_f64_gt(input, col, val)` | `provider.filter::<f64>(input, col, val, CompareOp::Gt)` |
| `provider.filter_f64_lt(input, col, val)` | `provider.filter::<f64>(input, col, val, CompareOp::Lt)` |
| `provider.filter_i32(input, col, val, op)` | `provider.filter::<i32>(input, col, val, op)` |
| `provider.filter_u64(input, col, val, op)` | `provider.filter::<u64>(input, col, val, op)` |
| `provider.filter_f32(input, col, val, op)` | `provider.filter::<f32>(input, col, val, op)` |
| `provider.filter_bool(input, col, val, op)` | `provider.filter::<bool>(input, col, val, op)` |
| `provider.compare_columns_u32(input, l, r, op)` | `provider.compare_columns::<u32>(input, l, r, op)` |
| `provider.compare_columns_i32(...)` | `provider.compare_columns::<i32>(...)` |
| (etc. for all 7 compare_columns variants) | |

**Important:** Check `crates/xlog-runtime/src/executor.rs` for filter/compare calls — the executor dispatches filter operations to the provider based on `ScalarType`. Look for `match scalar_type { ... }` blocks that call type-specific filter functions. These need ScalarType-based dispatch to the generic:

```rust
// Before:
match scalar_type {
    ScalarType::U32 | ScalarType::Symbol => provider.filter_u32(input, col, val_u32, op),
    ScalarType::I32 => provider.filter_i32(input, col, val_i32, op),
    // ...
}

// After:
match scalar_type {
    ScalarType::U32 | ScalarType::Symbol => provider.filter::<u32>(input, col, val_u32, op),
    ScalarType::I32 => provider.filter::<i32>(input, col, val_i32, op),
    // ...
}
```

The match arms still need to exist because the value type differs per arm. This is a mechanical rename within each arm.

- [ ] **Step 1: Grep for all filter/compare call sites**

```bash
rg 'filter_u32|filter_u64|filter_i32|filter_f32|filter_f64|filter_bool|compare_columns_u32|compare_columns_i32|compare_columns_i64|compare_columns_u64|compare_columns_f32|compare_columns_f64|compare_columns_u8' --type rust -l
```

- [ ] **Step 2: Migrate all call sites**

Update each file. Add `use xlog_cuda::provider::CompareOp;` if not already imported (needed for wrapper function replacements).

- [ ] **Step 3: Verify compilation and tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git diff --name-only | xargs git add
git commit -m "refactor: migrate filter and compare_columns call sites to turbofish generics"
```

---

### Task 9: Remove old type-specialized functions + ride-along improvements + final gate

Remove all old functions that have been replaced by generics. Apply ride-along improvements (visibility tightening, error context).

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (remove ~1,200 LOC of old functions)
- Modify: `crates/xlog-cuda/src/provider/transfer.rs` (move old functions here before deletion, or delete from mod.rs directly)

**Functions to remove from mod.rs:**

Download column family (8 tracked + 1 untracked):
- `download_column_u32` (line 6874)
- `download_column_u8` (line 6914)
- `download_column_u64` (line 6949)
- `download_column_f64` (line 6989)
- `download_f64_untracked` (line 7020)
- `download_column_bool` (line 7056)
- `download_column_i32` (line 7091)
- `download_column_i64` (line 7131)
- `download_column_f32` (line 7171)

Create buffer family (7 single-column):
- `create_buffer_from_u32_slice` (line 5961)
- `create_buffer_from_u64_slice` (line 6029)
- `create_buffer_from_i64_slice` (line 6051)
- `create_buffer_from_i32_slice` (line 6073)
- `create_buffer_from_f64_slice` (line 6095)
- `create_buffer_from_f32_slice` (line 6117)
- `create_buffer_from_u8_slice` (line 6139)

**Note:** Keep `create_buffer_from_u32_columns` (line 5983) and `create_buffer_from_slices` (line 6166) — these have distinct semantics not covered by the generic.

Filter family (6 primary + 5 wrappers):
- `filter_u32_eq` (line 4814)
- `filter_u32_gt` (line 4830)
- `filter_u32` (line 4844)
- `filter_i32` (line 4961)
- `filter_u64` (line 4984)
- `filter_f32` (line 5007)
- `filter_bool` (line 5030)
- `filter_f64_eq` (line 5367)
- `filter_f64_gt` (line 5383)
- `filter_f64_lt` (line 5399)
- `filter_f64` (line 5413)

Compare columns family (7 wrappers):
- `compare_columns_u32` (line 5054)
- `compare_columns_i32` (line 5072)
- `compare_columns_i64` (line 5090)
- `compare_columns_u64` (line 5108)
- `compare_columns_f32` (line 5126)
- `compare_columns_f64` (line 5144)
- `compare_columns_u8` (line 5162)

**Total removed: ~1,500 LOC**

- [ ] **Step 1: Remove old functions**

Delete all listed functions from mod.rs. Compile to verify no remaining call sites:

Run: `cargo check --workspace`
Expected: PASS (all call sites migrated in Tasks 7–8)

If any errors, fix remaining call sites first.

- [ ] **Step 2: Apply ride-along — visibility tightening**

In each new submodule, change internal helpers from `pub` to `pub(crate)` where they are not called outside xlog-cuda. Check cross-crate usage with grep before tightening.

Candidates (internal helpers that should be `pub(crate)` rather than `pub`):
- `compact_buffer_by_device_mask_counted` (mod.rs:5585) — verify callers
- `compact_buffer_by_device_mask` (mod.rs:5792) — verify callers
- Helper methods moved to submodules that are only called within xlog-cuda

- [ ] **Step 3: Apply ride-along — error context**

In the new submodule methods, replace instances of:
```rust
XlogError::Kernel(format!("Failed to X: {}", e))
```
with:
```rust
XlogError::kernel_ctx("method_name", "description", &e)
```

Apply opportunistically as methods are reviewed during deletion. Don't refactor methods that weren't touched.

- [ ] **Step 4: Verify dead_code warnings resolved**

The Wave 1 `error_helpers.rs` and `type_seam.rs` had `#[allow(dead_code)]` or expected dead_code warnings. After Wave 2 adoption:
- `type_seam::GpuScalar` is now used by transfer.rs, filter.rs — warning should be gone
- `error_helpers::driver_err()` — check if it's been adopted in kernel_loading.rs; if not, adopt it there

- [ ] **Step 5: Full gate verification**

Run the complete Wave 2 gate:

```bash
# Gate 1: Workspace tests
cargo test --workspace --all-targets --exclude pyxlog --release

# Gate 2: CUDA certification suite (206/206)
cargo test -p xlog-cuda-tests --test certification_suite --release

# Gate 3: pyxlog compile gate
cargo check -p pyxlog
```

All three must pass.

- [ ] **Step 6: Commit**

```bash
git add crates/xlog-cuda/src/provider/ crates/xlog-cuda/src/type_seam.rs crates/xlog-cuda/src/error_helpers.rs
git commit -m "$(cat <<'EOF'
refactor(xlog-cuda): remove old type-specialized functions, apply ride-along improvements

Wave 2 complete: provider.rs (12,809 LOC) decomposed into 10 submodules.
4 type-specialized families collapsed into generics via GpuScalar trait.
Call sites migrated to turbofish syntax across the workspace.
new() refactored from 812 to ~10 lines via data-driven kernel loading.
Net reduction: ~5,990 lines.
EOF
)"
```

---

## Gate Summary

| Gate | Command | Required |
|------|---------|----------|
| Rust workspace | `cargo test --workspace --all-targets --exclude pyxlog --release` | Yes |
| CUDA certification | `cargo test -p xlog-cuda-tests --test certification_suite --release` | Yes (206/206) |
| pyxlog compile | `cargo check -p pyxlog` | Yes |

No Python gates for Wave 2 — pyxlog's Python-facing API is unchanged, only its internal Rust call sites are rewritten.

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| GpuScalar kernel name methods return wrong string | Test that each returns a non-empty string; certification suite catches kernel dispatch failures |
| 180+ turbofish edits introduce type mismatches | Rust compiler catches wrong type at compile time — wrong turbofish won't compile |
| filter fused-scan path breaks when generalized | Keep existing filter_u32/f64 tests, add explicit test calling filter::\<u32\>() and filter::\<f64\>() |
| Private helpers in mod.rs not accessible from submodules | All submodules use `impl super::CudaKernelProvider` — private methods on the struct are accessible from any `impl` block in the same crate |
| kernel_manifest_data.rs KERNEL_MODULES array has wrong entries | Test validates KERNEL_MODULES matches KERNEL_CU_NAMES; certification suite validates all kernels load correctly |
