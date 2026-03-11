# GPU Row-Count Correctness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure GPU set/join/groupby paths honor device row counts so buffers with `row_cap > d_num_rows` never leak garbage rows into results.

**Architecture:** Treat `row_cap` as capacity and `d_num_rows` as truth. All data-plane kernels must clamp to device row counts, and host-side orchestration must size buffers based on actual counts when required. Union/concat must never copy rows past `d_num_rows`. Join/groupby/pack must use device counts for sizes and emptiness decisions.

**Tech Stack:** Rust (xlog-cuda/xlog-runtime), CUDA kernels (cudarc/PTX), GPU memory pools.

---

### Task 1: Add a failing union test for device row counts (TDD)

**Files:**
- Modify: `crates/xlog-cuda/tests/set_ops_tests.rs`

**Step 1: Write the failing test**

Add a helper that builds a buffer with `row_cap > d_num_rows`, and a test that unions two such buffers:

```rust
fn device_row_count(provider: &CudaKernelProvider, rows: u32) -> xlog_cuda::memory::TrackedCudaSlice<u32> {
    let mut d_num_rows = provider.memory().alloc::<u32>(1).expect("alloc");
    provider.device().inner().htod_sync_copy_into(&[rows], &mut d_num_rows).expect("htod");
    d_num_rows
}

fn buffer_with_row_cap(
    provider: &CudaKernelProvider,
    data: &[u32],
    row_cap: u64,
    actual_rows: u32,
    schema: Schema,
) -> CudaBuffer {
    let mut bytes = Vec::with_capacity((row_cap as usize) * 4);
    for &v in data { bytes.extend_from_slice(&v.to_le_bytes()); }
    while bytes.len() < (row_cap as usize) * 4 {
        bytes.extend_from_slice(&0u32.to_le_bytes());
    }
    let mut col = provider.memory().alloc::<u8>(bytes.len()).expect("alloc");
    provider.device().inner().htod_sync_copy_into(&bytes, &mut col).expect("htod");
    let d_num_rows = device_row_count(provider, actual_rows);
    CudaBuffer::from_columns(vec![col.into()], row_cap, d_num_rows, schema)
}

#[test]
fn test_union_gpu_uses_device_row_count() {
    let Some(provider) = setup_provider() else { return; };
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let data = vec![1u32, 2, 99, 100];
    let buf_a = buffer_with_row_cap(&provider, &data, 4, 2, schema.clone());
    let buf_b = buffer_with_row_cap(&provider, &data, 4, 2, schema.clone());
    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column_u32(&result, 0).unwrap();
    assert_eq!(result_data, vec![1, 2]);
}
```

**Step 2: Run test to verify it fails**

Run:
```
cargo test -p xlog-cuda --test set_ops_tests test_union_gpu_uses_device_row_count -- --nocapture
```
Expected: FAIL, result contains `[1, 2, 99, 100]`.

---

### Task 2: Fix concat/union to respect device row counts

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`

**Step 1: Write minimal implementation**

Update `concat_buffers_gpu` to use device row counts for `a_rows`/`b_rows`, treat zero device rows as empty, and set `row_cap` to the actual total rows:

```rust
let a_rows = self.device_row_count(a)? as u64;
let b_rows = self.device_row_count(b)? as u64;
if a_rows == 0 && b_rows == 0 { ... }
if a_rows == 0 { return self.clone_buffer(b); }
if b_rows == 0 { return self.clone_buffer(a); }
let total_rows = a_rows + b_rows;
...
let a_bytes = (a_rows as usize) * elem_size;
let b_bytes = (b_rows as usize) * elem_size;
...
self.buffer_from_columns(result_columns, total_rows, schema)
```

Ensure union uses these semantics (no additional changes needed if concat is fixed). For empty checks in `union_gpu`, use device counts if needed.

**Step 2: Run union test**

Run:
```
cargo test -p xlog-cuda --test set_ops_tests test_union_gpu_uses_device_row_count -- --nocapture
```
Expected: PASS.

---

### Task 3: Fix join/groupby/pack to use device row counts

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`

**Step 1: Join path uses device counts**

Update `build_join_index_v2`, `hash_join_v2_with_index`, and all join variants to use `device_row_count` for `num_left/num_right`, and to treat `d_num_rows == 0` as empty:

```rust
let num_left = self.device_row_count(left)? as u32;
if num_left == 0 { return empty; }
let num_right = self.device_row_count(right)? as u32;
if num_right == 0 { return empty; }
```

Validate cached index against device counts (not `row_cap`).

**Step 2: Pack keys uses device counts**

In `pack_keys_gpu` and `pack_keys_gpu_generic`, replace `buffer.num_rows()` with `device_row_count(buffer)? as u32` and return empty outputs when it is 0.

**Step 3: Groupby uses device counts**

In `groupby_multi_agg`, replace `buffer.is_empty()` and `buffer.num_rows()` with the device row count for emptiness and `num_rows` sizing.

**Step 4: Run failing e2e tests**

Run:
```
cargo test -p xlog-logic --test e2e_integration_tests -- --nocapture
```
Expected: PASS.

---

### Task 4: Full verification

**Files:**
- None

**Step 1: Run full test suite**

```
cargo test
```
Expected: PASS.

---

### Task 5: Commit

**Step 1: Commit changes**

```
git add crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/set_ops_tests.rs kernels/set_ops.cu kernels/set_ops.ptx
# plus any other touched files

git commit -m "fix: respect device row counts in set/join/groupby"
```
