# P1 Critical Issues Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 4 P1 critical issues from the validation report to enable production use.

**Architecture:** Each fix is self-contained. Issues 1-3 require CUDA kernel and Rust provider changes. Issue 4 requires replacing CPU-based dedup with GPU sort.

**Tech Stack:** CUDA C++ kernels, Rust (cudarc), PTX compilation

---

## Issues Summary

| # | Issue | Files | Status |
|---|-------|-------|--------|
| 1 | Hash-only join comparison | `kernels/join.cu`, `provider.rs` | TODO |
| 2 | Sum truncation (schema mismatch) | `provider/mod.rs (pre-Wave-2 line 1747)-1748` | TODO |
| 3 | 256-element prefix sum limit | `kernels/scan.cu`, `provider.rs` | TODO |
| 4 | CPU sort in dedup | `provider/mod.rs (pre-Wave-2 line 507)-590` | TODO |
| 5 | Memory budget not enforced | `memory.rs` | ALREADY FIXED |

Note: Issue 5 (memory budget) was already fixed in `memory.rs:60-77` with proper atomic enforcement.

---

## Task 1: Fix Hash-Only Join Comparison

**Files:**
- Modify: `kernels/join.cu:167-196` (hash_join_probe_v2)
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (pass key data to probe)
- Test: `crates/xlog-cuda/tests/join_tests.rs` (new collision test)

**Problem:** The `hash_join_probe_v2` kernel only compares 64-bit hashes, not actual key bytes. While collision probability is extremely low (~2^-64), this is not mathematically correct.

**Step 1: Write failing test for hash collision**

Create `crates/xlog-cuda/tests/join_collision_test.rs`:

```rust
//! Test for join key verification (not just hash comparison)

use xlog_core::{MemoryBudget, Schema, ScalarType};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use std::sync::Arc;

fn create_test_provider() -> Option<CudaKernelProvider> {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return None;
    }
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let budget = MemoryBudget::with_limit(1024 * 1024 * 256);
    let memory = GpuMemoryManager::new(device.clone(), budget);
    CudaKernelProvider::new(device, memory).ok()
}

fn create_buffer(provider: &CudaKernelProvider, col1: &[u32], col2: &[u32]) -> CudaBuffer {
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);
    let num_rows = col1.len() as u64;
    let col1_bytes: Vec<u8> = col1.iter().flat_map(|v| v.to_le_bytes()).collect();
    let col2_bytes: Vec<u8> = col2.iter().flat_map(|v| v.to_le_bytes()).collect();

    let device = provider.device().inner();
    let d_col1 = device.htod_sync_copy(&col1_bytes).unwrap();
    let d_col2 = device.htod_sync_copy(&col2_bytes).unwrap();

    CudaBuffer::from_columns(vec![d_col1, d_col2], num_rows, schema)
}

/// Test that joins compare actual keys, not just hashes.
/// Even if two keys have the same hash, they should not match unless equal.
#[test]
fn test_join_verifies_keys_not_just_hash() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no CUDA device");
            return;
        }
    };

    // Create left table: key=100, val=1
    let left = create_buffer(&provider, &[100], &[1]);

    // Create right table: key=200, val=2 (different key, might have hash collision)
    let right = create_buffer(&provider, &[200], &[2]);

    // Join on key column - should produce ZERO results since 100 != 200
    let result = provider.hash_join_v2(&left, &right, &[0], &[0]).unwrap();

    assert_eq!(result.num_rows(), 0,
        "Join should produce no results when keys differ, even if hashes might collide");
}

/// Test that identical keys DO match
#[test]
fn test_join_matches_equal_keys() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no CUDA device");
            return;
        }
    };

    let left = create_buffer(&provider, &[100, 200], &[1, 2]);
    let right = create_buffer(&provider, &[200, 300], &[10, 20]);

    // Join on key column - should match on key=200
    let result = provider.hash_join_v2(&left, &right, &[0], &[0]).unwrap();

    assert_eq!(result.num_rows(), 1, "Join should match on equal keys");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test join_collision_test -- --nocapture`

Expected: Tests should PASS currently (hash comparison works for these simple cases). This test establishes correctness baseline.

**Step 3: Update join kernel to verify keys**

Modify `kernels/join.cu` - add key verification to `hash_join_probe_v2`:

```cuda
/**
 * Probe hash table (v2) - outputs matching row index pairs.
 *
 * Compares both hash values AND actual key bytes for correctness.
 */
extern "C" __global__ void hash_join_probe_v2(
    const uint64_t* __restrict__ probe_hashes,
    uint32_t num_probe,
    const uint32_t* __restrict__ hash_table,
    const uint64_t* __restrict__ build_hashes,
    const uint32_t* __restrict__ next_ptrs,
    uint32_t hash_table_size,
    uint32_t* __restrict__ output_left,
    uint32_t* __restrict__ output_right,
    uint32_t* __restrict__ output_count,
    uint32_t max_output,
    // New parameters for key verification
    const uint8_t* __restrict__ probe_data,
    const uint8_t* __restrict__ build_data,
    const uint32_t* __restrict__ key_col_offsets,
    const uint32_t* __restrict__ key_col_sizes,
    uint32_t num_key_cols,
    uint32_t probe_row_stride,
    uint32_t build_row_stride
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_probe) return;

    uint64_t hash = probe_hashes[tid];
    uint32_t bucket = (uint32_t)(hash % hash_table_size);

    uint32_t current = hash_table[bucket];
    while (current != 0xFFFFFFFF) {
        // First check hash match (fast path)
        if (build_hashes[current] == hash) {
            // Verify actual key bytes match
            bool keys_match = true;
            const uint8_t* probe_row = probe_data + (uint64_t)tid * probe_row_stride;
            const uint8_t* build_row = build_data + (uint64_t)current * build_row_stride;

            for (uint32_t c = 0; c < num_key_cols && keys_match; c++) {
                uint32_t offset = key_col_offsets[c];
                uint32_t size = key_col_sizes[c];
                for (uint32_t b = 0; b < size; b++) {
                    if (probe_row[offset + b] != build_row[offset + b]) {
                        keys_match = false;
                        break;
                    }
                }
            }

            if (keys_match) {
                uint32_t out_idx = atomicAdd(output_count, 1);
                if (out_idx < max_output) {
                    output_left[out_idx] = tid;
                    output_right[out_idx] = current;
                }
            }
        }
        current = next_ptrs[current];
    }
}
```

**Step 4: Recompile PTX**

Run: `nvcc -ptx --gpu-architecture=sm_90 kernels/join.cu -o kernels/join.ptx`

**Step 5: Update provider to pass key data**

Modify `crates/xlog-cuda/src/provider/mod.rs` in `hash_join_v2` method to pass the additional key verification parameters to the kernel.

**Step 6: Run tests to verify fix**

Run: `cargo test -p xlog-cuda --test join_collision_test -- --nocapture`

Expected: PASS

**Step 7: Commit**

```bash
git add kernels/join.cu kernels/join.ptx crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/join_collision_test.rs
git commit -m "fix(xlog-cuda): add key verification to hash join probe

Joins now compare actual key bytes after hash match, ensuring
mathematical correctness instead of relying on hash collision
probability."
```

---

## Task 2: Fix Sum Truncation (Schema Mismatch)

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs:1747-1748`
- Test: existing tests should pass

**Problem:** `groupby_multi_agg_result_schema` returns `ScalarType::U32` for Sum, but the actual implementation computes and returns u64 bytes. This schema/data mismatch causes incorrect interpretation of results.

**Step 1: Write failing test**

The existing tests may pass because they don't check schema types carefully. Add to `provider.rs` tests:

```rust
#[test]
fn test_groupby_multi_agg_sum_returns_u64_schema() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no CUDA device");
            return;
        }
    };

    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    let result_schema = provider.groupby_multi_agg_result_schema(
        &schema,
        &[0],
        &[(1, AggOp::Sum)],
    );

    // Sum should return U64 to prevent overflow
    assert_eq!(result_schema.column_type(1), Some(ScalarType::U64),
        "Sum aggregation should return U64 type, not U32");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda test_groupby_multi_agg_sum_returns_u64_schema -- --nocapture`

Expected: FAIL - schema returns U32 but should return U64

**Step 3: Fix the schema function**

Modify `crates/xlog-cuda/src/provider/mod.rs:1739-1749`:

```rust
for (i, &(_value_col, agg_op)) in aggs.iter().enumerate() {
    let agg_name = match agg_op {
        AggOp::Count => format!("count_{}", i),
        AggOp::Sum => format!("sum_{}", i),
        AggOp::Min => format!("min_{}", i),
        AggOp::Max => format!("max_{}", i),
        AggOp::LogSumExp => format!("logsumexp_{}", i),
    };
    // Return correct types for each aggregation
    let agg_type = match agg_op {
        AggOp::Count => ScalarType::U32,
        AggOp::Sum => ScalarType::U64,  // Sum uses u64 to prevent overflow
        AggOp::Min | AggOp::Max => ScalarType::U32,
        AggOp::LogSumExp => ScalarType::F64,
    };
    columns.push((agg_name, agg_type));
}
```

**Step 4: Run test to verify fix**

Run: `cargo test -p xlog-cuda test_groupby_multi_agg_sum_returns_u64_schema -- --nocapture`

Expected: PASS

**Step 5: Run all tests to ensure no regressions**

Run: `cargo test -p xlog-cuda -- --nocapture`

Expected: All tests pass

**Step 6: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs
git commit -m "fix(xlog-cuda): return U64 type for Sum aggregation schema

The groupby_multi_agg_result_schema now returns U64 for Sum operations,
matching the actual u64 computation in the kernel. This prevents silent
data corruption from schema/data type mismatch."
```

---

## Task 3: Implement Multi-Block Prefix Sum

**Files:**
- Modify: `kernels/scan.cu` (add multi-block scan)
- Modify: `crates/xlog-cuda/src/provider/mod.rs:1783-1867` (use multi-block scan)
- Test: `crates/xlog-cuda/tests/prefix_sum_tests.rs`

**Problem:** Current `prefix_sum_mask` is limited to 256 elements due to single-block scan. This breaks all filter operations on realistic data.

**Step 1: Write failing test for large prefix sum**

Create `crates/xlog-cuda/tests/large_prefix_sum_test.rs`:

```rust
//! Test prefix sum with more than 256 elements

use xlog_core::{MemoryBudget, Schema, ScalarType};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use std::sync::Arc;

fn create_test_provider() -> Option<CudaKernelProvider> {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return None;
    }
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let budget = MemoryBudget::with_limit(1024 * 1024 * 256);
    let memory = GpuMemoryManager::new(device.clone(), budget);
    CudaKernelProvider::new(device, memory).ok()
}

#[test]
fn test_prefix_sum_1000_elements() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no CUDA device");
            return;
        }
    };

    // Create mask with 1000 elements, alternating 0 and 1
    let mask: Vec<u8> = (0..1000).map(|i| (i % 2) as u8).collect();

    let result = provider.prefix_sum_mask(&mask);
    assert!(result.is_ok(), "prefix_sum_mask should work with 1000 elements");

    let (prefix_sum, count) = result.unwrap();
    assert_eq!(count, 500, "Should count 500 ones");
    assert_eq!(prefix_sum.len(), 1000);

    // Verify prefix sum values
    // mask[0]=0 -> prefix[0]=0
    // mask[1]=1 -> prefix[1]=0 (exclusive)
    // mask[2]=0 -> prefix[2]=1
    // mask[3]=1 -> prefix[3]=1
    // etc.
    let mut expected = 0u32;
    for i in 0..1000 {
        assert_eq!(prefix_sum[i], expected, "prefix_sum[{}] wrong", i);
        expected += mask[i] as u32;
    }
}

#[test]
fn test_prefix_sum_10000_elements() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no CUDA device");
            return;
        }
    };

    let mask: Vec<u8> = (0..10000).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();

    let result = provider.prefix_sum_mask(&mask);
    assert!(result.is_ok(), "prefix_sum_mask should work with 10000 elements");

    let (prefix_sum, count) = result.unwrap();
    let expected_count = (10000 + 2) / 3; // ceil(10000/3)
    assert_eq!(count, expected_count as u32);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test large_prefix_sum_test -- --nocapture`

Expected: FAIL with "prefix_sum_mask currently limited to 256 elements"

**Step 3: Add multi-block scan kernels to scan.cu**

Add to `kernels/scan.cu`:

```cuda
#define BLOCK_SIZE 256

/**
 * Multi-block exclusive scan - Phase 1: Block-level scans
 * Each block computes its local exclusive scan and outputs block totals.
 */
extern "C" __global__ void multiblock_scan_phase1(
    const uint8_t* __restrict__ mask,
    uint32_t* __restrict__ block_scans,
    uint32_t* __restrict__ block_totals,
    uint32_t n
) {
    __shared__ uint32_t temp[BLOCK_SIZE];

    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load mask as u32
    temp[tid] = (gid < n) ? (uint32_t)mask[gid] : 0;
    __syncthreads();

    // Inclusive scan within block using Hillis-Steele
    for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
        uint32_t val = 0;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Write block total
    if (tid == blockDim.x - 1) {
        block_totals[blockIdx.x] = temp[tid];
    }

    // Write exclusive scan (shift by 1)
    if (gid < n) {
        block_scans[gid] = (tid == 0) ? 0 : temp[tid - 1];
    }
}

/**
 * Multi-block exclusive scan - Phase 2: Scan block totals
 * Single block scans the block totals array.
 */
extern "C" __global__ void multiblock_scan_phase2(
    uint32_t* __restrict__ block_totals,
    uint32_t num_blocks
) {
    __shared__ uint32_t temp[BLOCK_SIZE];

    uint32_t tid = threadIdx.x;

    // Load block total (or 0 if out of range)
    temp[tid] = (tid < num_blocks) ? block_totals[tid] : 0;
    __syncthreads();

    // Inclusive scan
    for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
        uint32_t val = 0;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Write exclusive scan of block totals
    if (tid < num_blocks) {
        block_totals[tid] = (tid == 0) ? 0 : temp[tid - 1];
    }
}

/**
 * Multi-block exclusive scan - Phase 3: Add block offsets
 * Adds the scanned block totals to each element.
 */
extern "C" __global__ void multiblock_scan_phase3(
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ block_offsets,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && blockIdx.x > 0) {
        output[gid] += block_offsets[blockIdx.x];
    }
}
```

**Step 4: Recompile PTX**

Run: `nvcc -ptx --gpu-architecture=sm_90 kernels/scan.cu -o kernels/scan.ptx`

**Step 5: Update provider to use multi-block scan**

Modify `crates/xlog-cuda/src/provider/mod.rs` - replace `prefix_sum_mask` implementation:

```rust
/// Compute exclusive prefix sum of u8 mask using multi-block scan
pub fn prefix_sum_mask(&self, mask: &[u8]) -> Result<(Vec<u32>, u32)> {
    if mask.is_empty() {
        return Ok((vec![], 0));
    }

    let n = mask.len();
    let device = self.device.inner();

    // For small inputs, use simple CPU scan (faster than kernel launch overhead)
    if n <= 256 {
        let mut prefix_sum = Vec::with_capacity(n);
        let mut running = 0u32;
        for &m in mask {
            prefix_sum.push(running);
            running += m as u32;
        }
        return Ok((prefix_sum, running));
    }

    // Multi-block scan
    let block_size = 256u32;
    let num_blocks = ((n as u32) + block_size - 1) / block_size;

    // Upload mask
    let d_mask = device
        .htod_sync_copy(mask)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload mask: {}", e)))?;

    // Allocate buffers
    let d_block_scans = unsafe { device.alloc::<u32>(n) }
        .map_err(|e| XlogError::Kernel(format!("Failed to alloc block_scans: {}", e)))?;
    let d_block_totals = unsafe { device.alloc::<u32>(num_blocks as usize) }
        .map_err(|e| XlogError::Kernel(format!("Failed to alloc block_totals: {}", e)))?;

    // Phase 1: Block-level scans
    let phase1_fn = device
        .get_func(SCAN_MODULE, "multiblock_scan_phase1")
        .ok_or_else(|| XlogError::Kernel("multiblock_scan_phase1 not found".to_string()))?;

    unsafe {
        phase1_fn.clone().launch(
            LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (&d_mask, &d_block_scans, &d_block_totals, n as u32),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase1 failed: {}", e)))?;

    // Phase 2: Scan block totals (if more than one block)
    if num_blocks > 1 {
        let phase2_fn = device
            .get_func(SCAN_MODULE, "multiblock_scan_phase2")
            .ok_or_else(|| XlogError::Kernel("multiblock_scan_phase2 not found".to_string()))?;

        unsafe {
            phase2_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&d_block_totals, num_blocks),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase2 failed: {}", e)))?;

        // Phase 3: Add block offsets
        let phase3_fn = device
            .get_func(SCAN_MODULE, "multiblock_scan_phase3")
            .ok_or_else(|| XlogError::Kernel("multiblock_scan_phase3 not found".to_string()))?;

        unsafe {
            phase3_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&d_block_scans, &d_block_totals, n as u32),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase3 failed: {}", e)))?;
    }

    self.device.synchronize()?;

    // Download results
    let prefix_sum = device
        .dtoh_sync_copy(&d_block_scans)
        .map_err(|e| XlogError::Kernel(format!("Failed to download prefix_sum: {}", e)))?;

    // Compute total count
    let count = prefix_sum.last().copied().unwrap_or(0) + mask.last().map(|&m| m as u32).unwrap_or(0);

    Ok((prefix_sum, count))
}
```

**Step 6: Update kernel loading to include new functions**

Add new kernel names to the scan module loading.

**Step 7: Run tests to verify fix**

Run: `cargo test -p xlog-cuda --test large_prefix_sum_test -- --nocapture`

Expected: PASS

**Step 8: Run all tests**

Run: `cargo test -p xlog-cuda -- --nocapture`

Expected: All tests pass

**Step 9: Commit**

```bash
git add kernels/scan.cu kernels/scan.ptx crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/large_prefix_sum_test.rs
git commit -m "feat(xlog-cuda): implement multi-block prefix sum

Removes the 256-element limitation on prefix_sum_mask by implementing
a three-phase multi-block scan algorithm. Filter operations now work
on datasets of any practical size."
```

---

## Task 4: Replace CPU Sort in Dedup with GPU Sort

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs:507-590` (dedup function)
- Test: existing dedup tests should pass

**Problem:** The `dedup` function downloads data to CPU, sorts using Rust, and re-uploads. This is a performance bottleneck and unnecessary since GPU sort exists.

**Step 1: Review current dedup implementation**

The current implementation:
1. Downloads all columns to host
2. Sorts rows using Rust's sort_by
3. Removes consecutive duplicates
4. Re-uploads to GPU

**Step 2: Write performance test (optional, for benchmarking)**

The fix is to use existing `sort()` + GPU-based dedup marking.

**Step 3: Refactor dedup to use GPU sort**

Replace `crates/xlog-cuda/src/provider/mod.rs:507-590`:

```rust
/// Remove duplicate rows from buffer based on key columns
///
/// Uses GPU sort followed by GPU-based duplicate marking and compaction.
pub fn dedup(&self, input: &CudaBuffer, key_cols: &[usize]) -> Result<CudaBuffer> {
    if input.is_empty() {
        return self.create_empty_buffer(input.schema().clone());
    }

    if key_cols.is_empty() {
        return Err(XlogError::Kernel("Dedup requires at least one key column".to_string()));
    }

    // Step 1: Sort by key columns using GPU sort
    let sorted = self.sort(input, key_cols)?;

    if sorted.num_rows() <= 1 {
        return Ok(sorted);
    }

    let num_rows = sorted.num_rows() as usize;

    // Step 2: Mark boundaries (rows where key differs from previous)
    // For single U32 key column, use GPU kernel
    if key_cols.len() == 1 && sorted.schema().column_type(key_cols[0]) == Some(ScalarType::U32) {
        return self.dedup_sorted_single_key(&sorted, key_cols[0]);
    }

    // For multi-column or non-U32 keys, fall back to CPU-based boundary detection
    // (This is still faster than the previous full-CPU implementation since sort is GPU)
    self.dedup_sorted_cpu_boundary(&sorted, key_cols)
}

/// GPU-optimized dedup for sorted single U32 key column
fn dedup_sorted_single_key(&self, sorted: &CudaBuffer, key_col: usize) -> Result<CudaBuffer> {
    let num_rows = sorted.num_rows() as u32;
    let device = self.device.inner();

    // Get key column
    let key_data = sorted.column(key_col)
        .ok_or_else(|| XlogError::Kernel("Key column not found".to_string()))?;
    let keys_view = self.column_as_u32_view(key_data, num_rows as usize)?;

    // Allocate mask for unique rows
    let d_mask = self.memory.alloc::<u8>(num_rows as usize)?;

    // Launch mark_duplicates kernel (marks first occurrence of each key)
    let mark_fn = device
        .get_func(DEDUP_MODULE, dedup_kernels::MARK_DUPLICATES)
        .ok_or_else(|| XlogError::Kernel("mark_duplicates kernel not found".to_string()))?;

    let block_size = 256u32;
    let grid_size = (num_rows + block_size - 1) / block_size;

    unsafe {
        mark_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (&keys_view, num_rows, &d_mask),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("mark_duplicates failed: {}", e)))?;

    self.device.synchronize()?;

    // Download mask and use filter_by_mask
    let mut mask_host = vec![0u8; num_rows as usize];
    device
        .dtoh_sync_copy_into(&d_mask, &mut mask_host)
        .map_err(|e| XlogError::Kernel(format!("Failed to read mask: {}", e)))?;

    self.filter_by_mask(sorted, &mask_host)
}

/// CPU boundary detection for sorted multi-column dedup
fn dedup_sorted_cpu_boundary(&self, sorted: &CudaBuffer, key_cols: &[usize]) -> Result<CudaBuffer> {
    let num_rows = sorted.num_rows() as usize;

    // Download key columns
    let mut key_data: Vec<Vec<u8>> = Vec::new();
    for &col_idx in key_cols {
        let col = sorted.column(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;
        let mut host_data = vec![0u8; col.len()];
        self.device.inner()
            .dtoh_sync_copy_into(col, &mut host_data)
            .map_err(|e| XlogError::Kernel(format!("Failed to read column: {}", e)))?;
        key_data.push(host_data);
    }

    // Mark unique rows (first occurrence where key differs from previous)
    let mut mask = vec![0u8; num_rows];
    mask[0] = 1; // First row is always unique

    let col_sizes: Vec<usize> = key_cols.iter()
        .map(|&i| sorted.schema().column_type(i).map(|t| t.size_bytes()).unwrap_or(4))
        .collect();

    for i in 1..num_rows {
        let mut differs = false;
        for (col_idx, col_data) in key_data.iter().enumerate() {
            let size = col_sizes[col_idx];
            let prev_start = (i - 1) * size;
            let curr_start = i * size;
            if col_data[prev_start..prev_start + size] != col_data[curr_start..curr_start + size] {
                differs = true;
                break;
            }
        }
        if differs {
            mask[i] = 1;
        }
    }

    self.filter_by_mask(sorted, &mask)
}
```

**Step 4: Run existing dedup tests**

Run: `cargo test -p xlog-cuda dedup -- --nocapture`

Expected: All dedup tests pass

**Step 5: Run all tests**

Run: `cargo test -p xlog-cuda -- --nocapture`

Expected: All tests pass

**Step 6: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs
git commit -m "perf(xlog-cuda): use GPU sort in dedup instead of CPU sort

Dedup now uses the existing GPU radix sort followed by GPU-based
duplicate marking. This eliminates the CPU sorting bottleneck while
maintaining correctness for multi-column keys with CPU boundary detection."
```

---

## Final Verification

**Step 1: Run full test suite**

```bash
cargo test --workspace
```

Expected: All 275+ tests pass

**Step 2: Run E2E integration tests**

```bash
cargo test -p xlog-logic --test e2e_integration_tests -- --nocapture
```

Expected: All 11 E2E tests pass

**Step 3: Verify no regressions**

```bash
cargo clippy --workspace
cargo build --release
```

---

## Summary

| Task | Issue | Fix |
|------|-------|-----|
| 1 | Hash-only join | Add key byte verification in probe kernel |
| 2 | Sum truncation | Fix schema to return U64 for Sum |
| 3 | 256-element prefix sum | Implement multi-block scan |
| 4 | CPU sort in dedup | Use existing GPU sort + GPU dedup marking |
