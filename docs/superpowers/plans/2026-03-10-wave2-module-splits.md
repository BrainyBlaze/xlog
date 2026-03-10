# Wave 2: God Module Splits

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split three god modules (provider.rs 12,809 LOC, pyxlog/lib.rs 6,202 LOC) into focused submodules, genericize type-specialized functions, extract profiling boilerplate, and consolidate the test harness.

**Architecture:** Move functions into submodules by category while keeping the public API stable via re-exports from the parent module. Use Rust generics with a `DeviceRepr` trait to replace type-specialized download/upload functions. Create a shared test helper module for CUDA tests.

**Tech Stack:** Rust (generics, module system), PyO3 (Python bindings). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-10-codebase-refactoring-design.md` (H1–H5, H8)

**Prerequisite:** Wave 1 (error unification + visibility lockdown) must be complete first.

---

## Chunk 1: provider.rs Split (H1)

### File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `crates/xlog-cuda/src/provider.rs` → `crates/xlog-cuda/src/provider/mod.rs` | Struct definition, lifecycle, core helpers |
| Create | `crates/xlog-cuda/src/provider/joins.rs` | hash_join family, key packing, hash tables |
| Create | `crates/xlog-cuda/src/provider/set_ops.rs` | union, diff, dedup, sort |
| Create | `crates/xlog-cuda/src/provider/groupby.rs` | groupby, aggregation |
| Create | `crates/xlog-cuda/src/provider/filters.rs` | filter_*, compare_columns_*, filter_by_mask |
| Create | `crates/xlog-cuda/src/provider/memory.rs` | buffer creation, column extraction, download, upload |
| Create | `crates/xlog-cuda/src/provider/arrow.rs` | Arrow/IPC/DLPack conversions |
| Create | `crates/xlog-cuda/src/provider/arithmetic.rs` | add/sub/mul/div/mod/abs/min/max/pow/cast columns |
| Create | `crates/xlog-cuda/src/provider/sampling.rs` | Bernoulli sampling, MC kernels |
| Create | `crates/xlog-cuda/src/provider/ilp.rs` | ILP kernel launches (credit, COO fill, reduce, histogram) |
| Create | `crates/xlog-cuda/src/provider/scan.rs` | prefix_sum, exclusive_scan, count_mask, gather, init_indices |
| Modify | `crates/xlog-cuda/src/lib.rs` | Update `mod provider` to `mod provider` (directory module) |

### Task 1: Convert provider.rs to a directory module

**Important context:** The `CudaKernelProvider` struct (defined at ~line 744) has 5 fields that all submodules need access to:
```rust
pub struct CudaKernelProvider {
    device: Arc<CudaDevice>,
    memory: Arc<GpuMemoryManager>,
    transfer_tracker: HostTransferTracker,
    ptx_load_profile: Option<PtxLoadProfile>,
    d2h_transfer_count: AtomicU64,
}
```

**Strategy:** Keep the struct definition and shared helpers in `mod.rs`. Each submodule file contains `impl CudaKernelProvider` blocks with the category-specific methods. Rust allows multiple `impl` blocks for the same type across modules within the same crate.

- [ ] **Step 1: Create the provider directory**

```bash
mkdir -p crates/xlog-cuda/src/provider
```

- [ ] **Step 2: Move provider.rs to provider/mod.rs**

```bash
git mv crates/xlog-cuda/src/provider.rs crates/xlog-cuda/src/provider/mod.rs
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p xlog-cuda --release`

Expected: Clean compilation (the module path is the same).

- [ ] **Step 4: Commit**

```bash
git add crates/xlog-cuda/src/provider/
git commit -m "refactor(cuda): convert provider.rs to directory module (no code changes)"
```

### Task 2: Extract joins.rs

**Functions to move** (from `provider/mod.rs`):

Public API:
- `hash_join` (line 1845)
- `hash_join_with_limit` (line 1873)
- `hash_join_v2` (line 7447)
- `hash_join_v2_with_limit` (line 7473)
- `build_join_index_v2` (line 7495)
- `hash_join_v2_with_index` (line 7544)
- `build_hash_table_u64` (line 8025)

Internal helpers (move as `pub(crate)` or keep accessible):
- `pack_keys_gpu` (line 7666)
- `build_hash_table_v2` (line 7919)
- `hash_join_inner_v2` (line 8047)
- `hash_join_inner_v2_indexed` (line 8253)

- [ ] **Step 1: Create joins.rs with the impl block**

Create `crates/xlog-cuda/src/provider/joins.rs`. Move the join functions into an `impl CudaKernelProvider` block. Add necessary `use` imports at the top.

The file structure should be:
```rust
use super::*;  // Imports from mod.rs (types, CudaKernelProvider, helpers)

impl CudaKernelProvider {
    // Paste all join functions here, preserving exact signatures and bodies
    pub fn hash_join(...) -> Result<CudaBuffer> { ... }
    // ... etc
}
```

**IMPORTANT:** Functions that call helpers still in `mod.rs` (like `device_row_count`, `buffer_from_columns`, `column_as_u32_view`) will work because `self` methods resolve across all `impl` blocks.

- [ ] **Step 2: Add `mod joins;` to mod.rs and remove moved functions**

In `provider/mod.rs`:
1. Add `mod joins;` near the top
2. Delete the moved function bodies (but NOT the helper functions they depend on)

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p xlog-cuda --release`

Expected: Clean compilation.

- [ ] **Step 4: Run CUDA tests**

Run: `cargo test -p xlog-cuda --release 2>&1 | tail -10`

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider/
git commit -m "refactor(cuda): extract join operations to provider/joins.rs"
```

### Task 3: Extract set_ops.rs

**Functions to move:**
- `union` (line 2217)
- `diff` (line 2341)
- `union_gpu` (line 2536)
- `diff_gpu` (line 2592)
- `dedup` (line 1998)
- `dedup_sorted` (line 2033)

- [ ] **Step 1–5:** Same pattern as Task 2. Create `provider/set_ops.rs`, move functions, add `mod set_ops;`, verify, commit.

```bash
git commit -m "refactor(cuda): extract set operations to provider/set_ops.rs"
```

### Task 4: Extract filters.rs

**Functions to move:**
- All `filter_*` functions (lines 4814–5585)
- All `compare_columns_*` functions (lines 5054–5162)
- `filter_by_mask` (line 5536)
- `compact_buffer_by_device_mask_counted` (line 5585)
- `filter_by_device_mask` (line 5851)
- `filter_f64_*` functions (lines 5367–5413)

- [ ] **Step 1–5:** Same pattern. Create `provider/filters.rs`, move, verify, commit.

```bash
git commit -m "refactor(cuda): extract filter operations to provider/filters.rs"
```

### Task 5: Extract groupby.rs

**Functions to move:**
- `groupby_agg` (line 2819)
- `groupby_multi_agg` (line 2856)

- [ ] **Step 1–5:** Same pattern. Create `provider/groupby.rs`, move, verify, commit.

```bash
git commit -m "refactor(cuda): extract groupby operations to provider/groupby.rs"
```

### Task 6: Extract scan.rs

**Functions to move:**
- `exclusive_scan_u32_inplace` (line 3472)
- `prefix_sum_mask` (line 3487)
- `prefix_sum_mask_cpu` (line 3510)
- `prefix_sum_mask_gpu_multiblock` (line 3525)
- `multiblock_scan_u32_inplace` (line 3606)
- `multiblock_scan_u32_view_inplace` (line 3689)
- `sort` (line 3793)
- `init_indices` (line 4240)
- `gather_u32_by_indices` (line 4278)
- `gather_u8_by_indices` (line 4320)
- `gather_u64_lo_by_indices` (line 4362)
- `gather_u64_hi_by_indices` (line 4395)
- `radix_sort_u32_pairs` (line 4428)
- `scan_u8_mask_device` (line 4454)
- `count_mask_device` (line 4524)
- `count_mask_into_slot` (line 4575)

- [ ] **Step 1–5:** Same pattern. Create `provider/scan.rs`, move, verify, commit.

```bash
git commit -m "refactor(cuda): extract scan/sort operations to provider/scan.rs"
```

### Task 7: Extract memory.rs

**Functions to move:**
- All `create_buffer_from_*_slice` functions (lines 5961–6166)
- `create_empty_buffer` (line 7384)
- `extract_column` (line 10022)
- `create_constant_column` (line 10070)
- `create_constant_column_with_device_count` (line 10156)
- `combine_columns` (line 11096)
- All `download_column_*` functions (lines 6874–7171)

Shared helpers to keep in mod.rs (used by multiple submodules):
- `device_row_count` (line 7200)
- `clone_device_row_count` (line 7213)
- `upload_device_row_count` (line 7222)
- `buffer_from_columns` (line 7393)
- `buffer_from_columns_with_device_count` (line 7231)
- `column_bytes_view` (line 7244)
- `bytes_as_u32_view` (line 7264)
- `column_as_u32_view` (line 7292)
- `column_as_u64_view` (line 7319)
- `column_as_f64_view` (line 7347)
- `schemas_type_compatible` (line 7422)
- `combine_schemas` (line 7412)
- `dtoh_sync_copy_into_tracked` (line 1615)
- `htod_sync_copy_into_tracked` (line 1666)

- [ ] **Step 1–5:** Same pattern. Create `provider/memory.rs`, move buffer/download functions, verify, commit.

```bash
git commit -m "refactor(cuda): extract buffer/memory operations to provider/memory.rs"
```

### Task 8: Extract arrow.rs

**Functions to move:**
- `to_arrow_device_record_batch` (line 6228)
- `from_arrow_device_record_batch` (line 6304)
- `to_arrow_record_batch` (line 6434)
- `from_arrow_record_batch` (line 6696)
- `to_arrow_ipc_stream` (line 6784)
- `from_arrow_ipc_stream` (line 6803)
- `write_arrow_ipc_stream_file` (line 6830)
- `read_arrow_ipc_stream_file` (line 6847)

- [ ] **Step 1–5:** Same pattern. Create `provider/arrow.rs`, move, verify, commit.

```bash
git commit -m "refactor(cuda): extract Arrow conversion to provider/arrow.rs"
```

### Task 9: Extract arithmetic.rs

**Functions to move:**
- `add_columns` (line 10270)
- `sub_columns` (line 10314)
- `mul_columns` (line 10358)
- `div_columns` (line 10404)
- `mod_columns` (line 10449)
- `abs_column` (line 10490)
- `min_columns` (line 10651)
- `max_columns` (line 10694)
- `pow_columns` (line 10738)
- `cast_column` (line 10946)
- `select_columns_bool` (line 10918)
- `binary_arith_op_device` (line 11018)

- [ ] **Step 1–5:** Same pattern. Create `provider/arithmetic.rs`, move, verify, commit.

```bash
git commit -m "refactor(cuda): extract arithmetic operations to provider/arithmetic.rs"
```

### Task 10: Extract sampling.rs

**Functions to move:**
- `sample_bernoulli_matrix` (line 1685)
- `sample_bernoulli_matrix_device` (line 1761)

- [ ] **Step 1–5:** Same pattern. Create `provider/sampling.rs`, move, verify, commit.

```bash
git commit -m "refactor(cuda): extract sampling operations to provider/sampling.rs"
```

### Task 11: Extract ilp.rs

**Functions to move:**
- `ilp_coo_fill_launch` (line 8684)
- `ilp_credit_forward_f32_launch` (line 8720)
- `ilp_credit_forward_f64_launch` (line 8778)
- `ilp_credit_backward_f32_launch` (line 8835)
- `ilp_credit_backward_f64_launch` (line 8888)
- `ilp_reduce_sum_f32_launch` (line 11166)
- `ilp_reduce_sum_f64_launch` (line 11208)
- `ilp_coo_fill_from_mask_launch` (line 11385)
- `ilp_csr_histogram_launch` (line 11450)

- [ ] **Step 1–5:** Same pattern. Create `provider/ilp.rs`, move, verify, commit.

```bash
git commit -m "refactor(cuda): extract ILP kernel launches to provider/ilp.rs"
```

### Task 12: Verify provider split is complete

- [ ] **Step 1: Check mod.rs size**

Run: `wc -l crates/xlog-cuda/src/provider/mod.rs`

Expected: ~2,000–2,500 lines (struct definition + lifecycle + shared helpers).

- [ ] **Step 2: Check all submodule sizes**

Run: `wc -l crates/xlog-cuda/src/provider/*.rs`

Expected: Each submodule under 2,500 lines.

- [ ] **Step 3: Full test suite**

Run: `cargo test -p xlog-cuda --release && cargo test -p xlog-cuda-tests --test certification_suite --release`

Expected: All pass including 206/206 certification.

- [ ] **Step 4: Commit any fixups**

If any tests failed, fix and commit.

---

## Chunk 2: Genericize Type-Specialized Functions (H2)

### Task 13: Create DeviceRepr trait and generic download_column

**Files:**
- Create: `crates/xlog-cuda/src/provider/device_repr.rs`
- Modify: `crates/xlog-cuda/src/provider/memory.rs`
- Create: `crates/xlog-cuda/tests/test_generic_download.rs`

**Context:** 8 `download_column_*` functions (lines 6874–7171 in original provider.rs, now in `memory.rs`) share 85% identical code. The only differences are:
1. The type `T` (u32, u64, i32, i64, f32, f64, bool, u8)
2. The byte-to-type conversion (`T::from_le_bytes` or `b != 0` for bool)
3. Whether D2H counter is incremented (all except `download_f64_untracked`)

- [ ] **Step 1: Write failing test for generic download**

Create `crates/xlog-cuda/tests/test_generic_download.rs`:

```rust
mod common;

#[test]
fn generic_download_u32_matches_typed() {
    let Some(provider) = common::setup_provider() else { return };

    let schema = xlog_core::Schema::new(vec![
        ("a".to_string(), xlog_core::ScalarType::U32),
    ]);
    let data: Vec<u32> = vec![1, 2, 3, 4, 5];
    let buf = provider.create_buffer_from_u32_slice(&data, schema).unwrap();

    // Generic version
    let generic_result: Vec<u32> = provider.download_column(&buf, 0).unwrap();
    // Typed version (existing)
    let typed_result = provider.download_column_u32(&buf, 0).unwrap();

    assert_eq!(generic_result, typed_result);
}

#[test]
fn generic_download_f64_matches_typed() {
    let Some(provider) = common::setup_provider() else { return };

    let schema = xlog_core::Schema::new(vec![
        ("a".to_string(), xlog_core::ScalarType::F64),
    ]);
    let data: Vec<f64> = vec![1.0, 2.5, 3.7];
    let buf = provider.create_buffer_from_f64_slice(&data, schema).unwrap();

    let generic_result: Vec<f64> = provider.download_column(&buf, 0).unwrap();
    let typed_result = provider.download_column_f64(&buf, 0).unwrap();

    assert_eq!(generic_result, typed_result);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test test_generic_download --release 2>&1 | head -20`

Expected: Compilation failure — `download_column` method doesn't exist.

- [ ] **Step 3: Define the DeviceRepr trait**

Create `crates/xlog-cuda/src/provider/device_repr.rs`:

```rust
/// Trait for types that can be transferred between host and GPU device memory.
/// Enables generic `download_column<T>` and `create_buffer_from_slice<T>`.
pub trait DeviceRepr: Copy + Sized + 'static {
    /// Number of bytes per element.
    const BYTE_SIZE: usize = std::mem::size_of::<Self>();

    /// Convert a byte chunk (little-endian) to this type.
    fn from_le_bytes(bytes: &[u8]) -> Self;

    /// Convert this type to little-endian bytes.
    fn to_le_bytes_vec(&self) -> Vec<u8>;
}

impl DeviceRepr for u32 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        u32::from_le_bytes(bytes.try_into().unwrap())
    }
    fn to_le_bytes_vec(&self) -> Vec<u8> { self.to_le_bytes().to_vec() }
}

impl DeviceRepr for u64 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        u64::from_le_bytes(bytes.try_into().unwrap())
    }
    fn to_le_bytes_vec(&self) -> Vec<u8> { self.to_le_bytes().to_vec() }
}

impl DeviceRepr for i32 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        i32::from_le_bytes(bytes.try_into().unwrap())
    }
    fn to_le_bytes_vec(&self) -> Vec<u8> { self.to_le_bytes().to_vec() }
}

impl DeviceRepr for i64 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        i64::from_le_bytes(bytes.try_into().unwrap())
    }
    fn to_le_bytes_vec(&self) -> Vec<u8> { self.to_le_bytes().to_vec() }
}

impl DeviceRepr for f32 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes(bytes.try_into().unwrap())
    }
    fn to_le_bytes_vec(&self) -> Vec<u8> { self.to_le_bytes().to_vec() }
}

impl DeviceRepr for f64 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        f64::from_le_bytes(bytes.try_into().unwrap())
    }
    fn to_le_bytes_vec(&self) -> Vec<u8> { self.to_le_bytes().to_vec() }
}

impl DeviceRepr for u8 {
    fn from_le_bytes(bytes: &[u8]) -> Self { bytes[0] }
    fn to_le_bytes_vec(&self) -> Vec<u8> { vec![*self] }
}

impl DeviceRepr for bool {
    const BYTE_SIZE: usize = 1;
    fn from_le_bytes(bytes: &[u8]) -> Self { bytes[0] != 0 }
    fn to_le_bytes_vec(&self) -> Vec<u8> { vec![if *self { 1 } else { 0 }] }
}
```

- [ ] **Step 4: Add generic download_column method**

In `crates/xlog-cuda/src/provider/memory.rs`, add:

```rust
use super::device_repr::DeviceRepr;

impl CudaKernelProvider {
    /// Generic column download — replaces all type-specialized download_column_* functions.
    pub fn download_column<T: DeviceRepr>(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<T>> {
        self.d2h_transfer_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let col = buffer.column(col_idx)?;
        let num_rows = self.device_row_count(buffer)?;
        if num_rows == 0 {
            return Ok(vec![]);
        }
        let num_bytes = num_rows * T::BYTE_SIZE;
        let col_view = self.column_bytes_view(col, num_bytes)?;
        let mut bytes = vec![0u8; num_bytes];
        self.device
            .inner()
            .dtoh_sync_copy_into(&col_view, &mut bytes)
            .map_err(|e| XlogError::Execution(format!("D2H copy failed: {}", e)))?;
        self.transfer_tracker.record_d2h(num_bytes);
        Ok(bytes.chunks_exact(T::BYTE_SIZE).map(|c| T::from_le_bytes(c)).collect())
    }
}
```

Add `mod device_repr;` and `pub use device_repr::DeviceRepr;` to `provider/mod.rs`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test test_generic_download --release`

Expected: PASS.

- [ ] **Step 6: Deprecate old typed functions**

Add `#[deprecated(note = "Use download_column::<T>() instead")]` to each of the 8 typed download functions. Do NOT remove them yet — external consumers may use them.

- [ ] **Step 7: Run full test suite**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -20`

Expected: All pass (deprecated functions still compile).

- [ ] **Step 8: Commit**

```bash
git add crates/xlog-cuda/src/provider/
git add crates/xlog-cuda/tests/test_generic_download.rs
git commit -m "feat(cuda): add generic download_column<T> via DeviceRepr trait, deprecate typed variants"
```

### Task 14: Add generic create_buffer_from_slice

Same pattern as Task 13 but for the 7 `create_buffer_from_*_slice` functions.

- [ ] **Step 1: Write failing test**

Test that `provider.create_buffer_from_slice::<u32>(&data, schema)` produces the same result as `provider.create_buffer_from_u32_slice(&data, schema)`.

- [ ] **Step 2: Implement generic version**

```rust
pub fn create_buffer_from_slice<T: DeviceRepr>(&self, data: &[T], schema: Schema) -> Result<CudaBuffer> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes_vec()).collect();
    // ... same logic as typed version using bytes
}
```

- [ ] **Step 3: Deprecate old typed functions**

- [ ] **Step 4: Run tests, commit**

```bash
git commit -m "feat(cuda): add generic create_buffer_from_slice<T>, deprecate typed variants"
```

---

## Chunk 3: pyxlog/lib.rs Split (H3)

### Task 15: Convert pyxlog/lib.rs to a directory module

- [ ] **Step 1: Create the directory**

```bash
mkdir -p crates/pyxlog/src/pyxlog_impl
```

**Note:** We can't use `pyxlog/` as the module name since it conflicts with the crate name. Use `pyxlog_impl/` or keep functions in separate files at the `src/` level.

**Alternative approach (simpler):** Keep `lib.rs` as the module root, create sibling files:
```
crates/pyxlog/src/
  lib.rs           — module declarations, registration, re-exports (~500 LOC)
  types.rs         — wrapper types, internal enums, conversions (~300 LOC)
  bindings.rs      — Program, CompiledProgram (~2900 LOC)
  logic.rs         — LogicProgram, CompiledLogicProgram (~150 LOC)
  results.rs       — EvalResult, McDeviceEvalResult, etc. (~100 LOC)
  training.rs      — EpochStats, TrainingHistory, train_model (~550 LOC)
  ilp.rs           — IlpProgramFactory, CompiledIlpProgram (~1900 LOC)
```

- [ ] **Step 2: Verify current lib.rs compiles**

Run: `cargo check -p pyxlog --release`

- [ ] **Step 3: Commit the structure plan**

No code changes yet — just verifying the starting state.

### Task 16: Extract types.rs from pyxlog

**Items to move** (from lib.rs):
- `scalar_type_name()` (line 49)
- `nll_loss_value()` (line 75)
- `create_torch_tensor()` (line 80)
- `dlpack_capsule_from_tensor()` (line 108)
- `arrow_device_capsule_from_device_array()` (line 157)
- `arrow_device_from_py()` (line 183)
- `atom_to_string()` (line 225)
- `provider_from_config()` (line 250)
- `parse_prob_engine_override()` (line 259)
- `dlpack_from_py()` (line 271)
- `InputSource` enum (line 559)
- `NeuralGroup` struct (line 565)
- `QuerySignature` enum (line 573)
- `CachedCircuit` struct (line 547)
- `CompiledProbProgram` enum (line 591)

- [ ] **Step 1: Create types.rs with moved items**

Create `crates/pyxlog/src/types.rs` with all utility functions and internal types. Use `pub(crate)` visibility since these are not exposed to Python.

- [ ] **Step 2: Update lib.rs imports**

Add `mod types;` and `use types::*;` (or specific imports) to lib.rs.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p pyxlog --release`

- [ ] **Step 4: Commit**

```bash
git add crates/pyxlog/src/
git commit -m "refactor(pyxlog): extract utility types and helpers to types.rs"
```

### Task 17: Extract results.rs from pyxlog

**Items to move:**
- `LogicQueryResult` struct (line 3695)
- `LogicEvalResult` struct (line 3709)
- `McDeviceEvalResult` struct (line 3715)
- `EvalResult` struct (line 3741)
- `EpochStats` struct (line 3789)
- `TrainingHistory` struct (line 3804)

All are `#[pyclass]` types — they need `#[pyclass]` annotations preserved and must be registered in the module init function.

- [ ] **Step 1–4:** Same pattern. Create `results.rs`, move, update imports, verify, commit.

```bash
git commit -m "refactor(pyxlog): extract result types to results.rs"
```

### Task 18: Extract logic.rs from pyxlog

**Items to move:**
- `LogicProgram` struct + impl (line 3570–3599)
- `CompiledLogicProgram` struct + impl (line 3599–3693)

- [ ] **Step 1–4:** Same pattern.

```bash
git commit -m "refactor(pyxlog): extract logic program bindings to logic.rs"
```

### Task 19: Extract training.rs from pyxlog

**Items to move:**
- `train_model()` function and `train_model_tensor()` function
- Supporting helpers: `push_term_bytes()` (line 4011), `pack_i64_columns_typed()` (line 4083), `load_facts_into_store()` (line 4178)

- [ ] **Step 1–4:** Same pattern.

```bash
git commit -m "refactor(pyxlog): extract training functions to training.rs"
```

### Task 20: Extract ilp.rs from pyxlog

**Items to move:**
- `IlpProgramFactory` struct + impl (line 4300–4387)
- `CompiledIlpProgram` struct + impl (line 4387–6174)
- Supporting helpers: `walk_tmj()`, `extract_tmj_meta()`, `extract_tmj_meta_for_mask()`, `strip_learnable_declarations()`, `extract_learnable_declarations()`

- [ ] **Step 1–4:** Same pattern.

```bash
git commit -m "refactor(pyxlog): extract ILP bindings to ilp.rs"
```

### Task 21: Verify pyxlog split is complete

- [ ] **Step 1: Check lib.rs size**

Run: `wc -l crates/pyxlog/src/lib.rs`

Expected: ~500–800 lines (module declarations + Program/CompiledProgram core + registration).

**Note:** The core `Program` and `CompiledProgram` classes (~2900 lines) may stay in lib.rs or move to `bindings.rs`. If lib.rs is still >1500 lines after the above extractions, extract bindings.rs too.

- [ ] **Step 2: Run Python tests**

```bash
cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/ -x -q 2>&1 | tail -20
```

Expected: All pass.

- [ ] **Step 3: Commit any fixups**

---

## Chunk 4: Profiling Extraction & Test Harness (H5, H8)

### Task 22: Extract profiling wrapper from execute_node (H5)

**Files:**
- Modify: `crates/xlog-runtime/src/executor.rs:694-882`

**Context:** The `execute_node` match (11 arms, 8 with profiling) repeats this pattern:
```rust
let start = self.profiler.start_op();
let result = self.execute_<op>(...)?;
if let Some(start) = start {
    let mem = self.provider.memory().allocated_bytes();
    self.profiler.record_op("<op>", input_rows, result.num_rows(), start, mem);
    self.profiler.record_peak_memory(mem);
}
Ok(result)
```

- [ ] **Step 1: Write the profiled_execute helper**

Add to `executor.rs`:

```rust
/// Execute an operation with profiling instrumentation.
fn profiled_execute<F>(
    &mut self,
    op_name: &str,
    input_rows: usize,
    f: F,
) -> Result<CudaBuffer>
where
    F: FnOnce(&mut Self) -> Result<CudaBuffer>,
{
    let start = self.profiler.start_op();
    let result = f(self)?;
    if let Some(start) = start {
        let mem = self.provider.memory().allocated_bytes();
        let out_rows = self.buffer_row_count(&result)? as usize;
        self.profiler.record_op(op_name, input_rows, out_rows, start, mem);
        self.profiler.record_peak_memory(mem);
    }
    Ok(result)
}
```

- [ ] **Step 2: Refactor each match arm to use the helper**

Example transformation for the Filter arm:
```rust
// Before
RirNode::Filter { input, predicate } => {
    let start = self.profiler.start_op();
    let input_buf = self.execute_node(input)?;
    let input_rows = self.buffer_row_count(&input_buf)? as usize;
    let result = self.execute_filter(&input_buf, predicate)?;
    if let Some(start) = start {
        let mem = self.provider.memory().allocated_bytes();
        self.profiler.record_op("filter", input_rows, self.buffer_row_count(&result)? as usize, start, mem);
        self.profiler.record_peak_memory(mem);
    }
    Ok(result)
}

// After
RirNode::Filter { input, predicate } => {
    let input_buf = self.execute_node(input)?;
    let input_rows = self.buffer_row_count(&input_buf)? as usize;
    self.profiled_execute("filter", input_rows, |this| {
        this.execute_filter(&input_buf, predicate)
    })
}
```

**CAUTION:** The closure borrows `self` mutably. You may need to adjust the signature to take `&Self` if the inner operation only needs shared access, or use a different pattern if borrow checker complains. Test each arm individually.

- [ ] **Step 3: Run executor tests**

Run: `cargo test -p xlog-runtime --release 2>&1 | tail -10`

Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add crates/xlog-runtime/src/executor.rs
git commit -m "refactor(runtime): extract profiled_execute helper from execute_node match arms"
```

### Task 23: Create shared test harness (H8)

**Files:**
- Create: `crates/xlog-cuda/tests/common/mod.rs`
- Modify: All 20+ test files in `crates/xlog-cuda/tests/`

- [ ] **Step 1: Create common/mod.rs**

Create `crates/xlog-cuda/tests/common/mod.rs`:

```rust
#![allow(dead_code)]

use std::sync::Arc;
use xlog_cuda::{
    CudaKernelProvider, CudaBuffer,
    memory::{GpuMemoryManager, MemoryBudget, TrackedCudaSlice},
};
use xlog_core::{Schema, ScalarType};
use cudarc::driver::CudaDevice;

/// Create a CudaKernelProvider for testing. Returns None if CUDA is unavailable.
pub fn setup_provider() -> Option<CudaKernelProvider> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GB
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    Some(CudaKernelProvider::new(device, memory).unwrap())
}

/// Read the device-side row count from a CudaBuffer.
pub fn read_device_row_count(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> u32 {
    let mut host_rows = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
        .expect("device row count copy");
    host_rows[0]
}

/// Allocate a device-side row count with a specific value.
pub fn alloc_device_row_count(
    provider: &CudaKernelProvider,
    rows: u32,
) -> TrackedCudaSlice<u32> {
    let mut d_num_rows = provider.memory().alloc::<u32>(1).expect("alloc");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[rows], &mut d_num_rows)
        .expect("htod row count");
    d_num_rows
}

/// Create a Schema from column name/type pairs.
pub fn make_schema(cols: &[(&str, ScalarType)]) -> Schema {
    Schema::new(cols.iter().map(|(n, t)| (n.to_string(), *t)).collect())
}
```

- [ ] **Step 2: Migrate one test file as proof-of-concept**

Pick `groupby_tests.rs`. Replace its local `setup_provider()` and `device_row_count()` with:

```rust
mod common;
use common::*;
```

Then replace all `setup_provider()` calls (already compatible) and `device_row_count(&provider, &buf)` calls with `read_device_row_count(&provider, &buf)`.

- [ ] **Step 3: Verify the migrated test file passes**

Run: `cargo test -p xlog-cuda --test groupby_tests --release`

Expected: PASS.

- [ ] **Step 4: Migrate remaining test files**

Repeat for all 20+ files. For each file:
1. Add `mod common; use common::*;`
2. Remove local `setup_provider()`, `device_row_count()`, `make_schema()` definitions
3. Adjust any call-site differences (e.g., variant B `device_row_count` that allocates rather than reads → use `alloc_device_row_count`)

- [ ] **Step 5: Run full CUDA test suite**

Run: `cargo test -p xlog-cuda --release && cargo test -p xlog-cuda-tests --test certification_suite --release`

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add crates/xlog-cuda/tests/
git commit -m "refactor(cuda): consolidate test helpers into common/mod.rs (removes 20+ duplicates)"
```

### Task 24: Final Wave 2 validation

- [ ] **Step 1: Full workspace test**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release`

Expected: All pass.

- [ ] **Step 2: CUDA certification**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release`

Expected: 206/206 PASS.

- [ ] **Step 3: Python tests**

```bash
cd /home/dev/projects/xlog && .venv/bin/python -m pytest python/tests/ -x -q
```

Expected: All pass.

- [ ] **Step 4: Check file sizes**

```bash
wc -l crates/xlog-cuda/src/provider/*.rs
wc -l crates/pyxlog/src/*.rs
wc -l crates/xlog-runtime/src/executor.rs
```

Expected:
- No provider submodule >2,500 LOC
- pyxlog/lib.rs <1,500 LOC
- executor.rs ~4,200 LOC (reduced by ~45 LOC from profiling extraction)
