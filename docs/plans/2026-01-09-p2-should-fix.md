# P2 Should-Fix Issues Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 4 P2 medium-priority fixes to improve xlog functionality and remove artificial limitations.

**Architecture:** Each task is independent. Tasks 1-2 are Rust-only changes (kernels exist). Tasks 3-4 require new CUDA kernels.

**Tech Stack:** CUDA C++, Rust (cudarc), PTX compilation

---

## Issues Summary

| # | Issue | Effort | Approach |
|---|-------|--------|----------|
| 6 | Join output 1M limit | Low | Make limit configurable via parameter |
| 7 | No float support | Low | Wire existing filter_compare_f64 kernel |
| 8 | No LogSumExp | Medium | Implement 3-pass numerically stable algorithm |
| 9 | U32-only set ops | Medium | Extend sort/dedup to handle multiple types |

---

## Task 1: Make Join Output Limit Configurable

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (lines 428, 3070, 3572)
- Test: Add test for custom limit

**Problem:** Join output is hardcoded to `min(1_000_000)`. Large joins silently truncate results.

**Step 1: Add default constant and parameter**

Add near the top of provider.rs (after imports):

```rust
/// Default maximum output rows for join operations
pub const DEFAULT_JOIN_MAX_OUTPUT: usize = 1_000_000;
```

**Step 2: Update hash_join to accept max_output parameter**

Modify `hash_join` signature (around line 350):

```rust
/// Perform hash join between two buffers
///
/// # Arguments
/// * `left` - Left (probe) buffer
/// * `right` - Right (build) buffer
/// * `left_key` - Key column index in left buffer
/// * `right_key` - Key column index in right buffer
/// * `max_output` - Maximum output rows (None = DEFAULT_JOIN_MAX_OUTPUT)
pub fn hash_join(
    &self,
    left: &CudaBuffer,
    right: &CudaBuffer,
    left_key: usize,
    right_key: usize,
    max_output: Option<usize>,
) -> Result<CudaBuffer> {
    let max_output = max_output.unwrap_or(DEFAULT_JOIN_MAX_OUTPUT);
    // ... rest of implementation using max_output
```

**Step 3: Update hash_join_v2 similarly**

Update `hash_join_v2` (around line 2773) and `hash_join_inner_v2` (around line 3020):

```rust
pub fn hash_join_v2(
    &self,
    left: &CudaBuffer,
    right: &CudaBuffer,
    left_keys: &[usize],
    right_keys: &[usize],
    max_output: Option<usize>,
) -> Result<CudaBuffer> {
    let max_output = max_output.unwrap_or(DEFAULT_JOIN_MAX_OUTPUT);
```

**Step 4: Update internal methods**

Update `hash_join_inner_v2`, `hash_join_left_outer_impl` to accept and use the parameter.

**Step 5: Add test for custom limit**

Add to provider.rs tests:

```rust
#[test]
fn test_join_custom_max_output() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => return,
    };

    // Create buffers that would produce more than 10 results
    let left = create_test_buffer(&provider, &[1, 1, 1, 1, 1], "key");
    let right = create_test_buffer(&provider, &[1, 1, 1], "key");

    // With limit of 10, should get at most 10 results (5*3=15 possible)
    let result = provider.hash_join_v2(&left, &right, &[0], &[0], Some(10)).unwrap();
    assert!(result.num_rows() <= 10);

    // With no limit (default 1M), should get all 15
    let result_full = provider.hash_join_v2(&left, &right, &[0], &[0], None).unwrap();
    assert_eq!(result_full.num_rows(), 15);
}
```

**Step 6: Run tests**

```bash
cargo test -p xlog-cuda test_join_custom_max_output -- --nocapture
cargo test -p xlog-cuda join -- --nocapture
```

**Step 7: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs
git commit -m "feat(xlog-cuda): make join output limit configurable

Adds max_output parameter to hash_join and hash_join_v2. Defaults to
1M rows but can be customized for larger joins or memory-constrained
environments."
```

---

## Task 2: Add Float (F64) Filter Support

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (add filter_f64 method)
- Test: `crates/xlog-cuda/tests/filter_tests.rs`

**Problem:** The `filter_compare_f64` CUDA kernel exists but has no Rust wrapper.

**Step 1: Add filter_f64 method**

Add after `filter_u32` method (around line 2400):

```rust
/// Filter buffer by comparing F64 column to constant
///
/// # Arguments
/// * `input` - Input buffer
/// * `col` - Column index to compare (must be F64 type)
/// * `value` - Constant to compare against
/// * `op` - Comparison operator
///
/// # Returns
/// Buffer with rows where comparison is true
pub fn filter_f64(
    &self,
    input: &CudaBuffer,
    col: usize,
    value: f64,
    op: CompareOp,
) -> Result<CudaBuffer> {
    if input.is_empty() {
        return self.create_empty_buffer(input.schema().clone());
    }

    // Validate column type
    let col_type = input.schema().column_type(col)
        .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col)))?;
    if col_type != ScalarType::F64 {
        return Err(XlogError::Kernel(format!(
            "filter_f64 requires F64 column, got {:?}", col_type
        )));
    }

    let n = input.num_rows as usize;
    let device = self.device.inner();

    // Get column data
    let col_data = input.column(col)
        .ok_or_else(|| XlogError::Kernel("Column not found".to_string()))?;

    // Allocate mask
    let d_mask = self.memory.alloc::<u8>(n)?;

    // Launch filter_compare_f64 kernel
    let block_size = 256u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    let config = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let filter_fn = device
        .get_func(FILTER_MODULE, filter_kernels::FILTER_COMPARE_F64)
        .ok_or_else(|| XlogError::Kernel("filter_compare_f64 kernel not found".to_string()))?;

    // Reinterpret column bytes as f64
    let col_view: CudaView<f64> = unsafe {
        std::mem::transmute(col_data.slice(..))
    };

    let op_code = op as u8;

    // SAFETY: filter_compare_f64(column, constant, num_rows, op, mask)
    unsafe {
        filter_fn.clone().launch(config, (&col_view, value, n as u32, op_code, &d_mask))
    }
    .map_err(|e| XlogError::Kernel(format!("filter_compare_f64 failed: {}", e)))?;

    // Download mask and use filter_by_mask
    self.device.synchronize()?;

    let mut mask_host = vec![0u8; n];
    device
        .dtoh_sync_copy_into(&d_mask, &mut mask_host)
        .map_err(|e| XlogError::Kernel(format!("Failed to download mask: {}", e)))?;

    self.filter_by_mask(input, &mask_host)
}

/// Filter F64 column for equality
pub fn filter_f64_eq(&self, input: &CudaBuffer, col: usize, value: f64) -> Result<CudaBuffer> {
    self.filter_f64(input, col, value, CompareOp::Eq)
}

/// Filter F64 column for greater than
pub fn filter_f64_gt(&self, input: &CudaBuffer, col: usize, value: f64) -> Result<CudaBuffer> {
    self.filter_f64(input, col, value, CompareOp::Gt)
}

/// Filter F64 column for less than
pub fn filter_f64_lt(&self, input: &CudaBuffer, col: usize, value: f64) -> Result<CudaBuffer> {
    self.filter_f64(input, col, value, CompareOp::Lt)
}
```

**Step 2: Add helper to create F64 buffer for testing**

Add test helper:

```rust
fn create_f64_buffer(provider: &CudaKernelProvider, values: &[f64], name: &str) -> CudaBuffer {
    let schema = Schema::new(vec![(name.to_string(), ScalarType::F64)]);
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let device = provider.device().inner();
    let d_col = device.htod_sync_copy(&bytes).unwrap();
    CudaBuffer::from_columns(vec![d_col], values.len() as u64, schema)
}
```

**Step 3: Add tests**

Add to `crates/xlog-cuda/tests/filter_tests.rs`:

```rust
#[test]
fn test_filter_f64_gt() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no CUDA device");
            return;
        }
    };

    let buffer = create_f64_buffer(&provider, &[1.0, 2.5, 3.0, 0.5, 4.0], "value");

    let result = provider.filter_f64_gt(&buffer, 0, 2.0).unwrap();
    assert_eq!(result.num_rows(), 3); // 2.5, 3.0, 4.0
}

#[test]
fn test_filter_f64_lt() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no CUDA device");
            return;
        }
    };

    let buffer = create_f64_buffer(&provider, &[1.0, 2.5, 3.0, 0.5, 4.0], "value");

    let result = provider.filter_f64_lt(&buffer, 0, 2.0).unwrap();
    assert_eq!(result.num_rows(), 2); // 1.0, 0.5
}

#[test]
fn test_filter_f64_eq() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no CUDA device");
            return;
        }
    };

    let buffer = create_f64_buffer(&provider, &[1.0, 2.5, 2.5, 0.5, 4.0], "value");

    let result = provider.filter_f64_eq(&buffer, 0, 2.5).unwrap();
    assert_eq!(result.num_rows(), 2); // two 2.5 values
}
```

**Step 4: Run tests**

```bash
cargo test -p xlog-cuda --test filter_tests -- --nocapture
```

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/filter_tests.rs
git commit -m "feat(xlog-cuda): add F64 filter support

Wires up existing filter_compare_f64 CUDA kernel with Rust wrapper.
Adds filter_f64, filter_f64_eq, filter_f64_gt, filter_f64_lt methods."
```

---

## Task 3: Implement LogSumExp Aggregation

**Files:**
- Modify: `kernels/groupby.cu` (add logsumexp kernels)
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (implement LogSumExp case)
- Test: Add LogSumExp tests

**Problem:** LogSumExp aggregation returns "not implemented" error. Needed for probabilistic reasoning.

**Background:** LogSumExp computes `log(sum(exp(x_i)))` in a numerically stable way using:
```
logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
```

This requires a 3-pass algorithm:
1. Find max per group
2. Compute sum of exp(x - max) per group
3. Compute log(sum) + max per group

**Step 1: Add LogSumExp kernels to groupby.cu**

Add to `kernels/groupby.cu`:

```cuda
// LogSumExp Pass 1: Find max value per group
extern "C" __global__ void groupby_logsumexp_max(
    const double* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    double* __restrict__ maxs  // Initialize to -INFINITY
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t group = group_ids[tid];
    double val = values[tid];

    // Atomic max for doubles using CAS
    unsigned long long* addr = (unsigned long long*)&maxs[group];
    unsigned long long old = *addr;
    unsigned long long assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (val <= old_val) break;
        old = atomicCAS(addr, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

// LogSumExp Pass 2: Compute sum of exp(x - max) per group
extern "C" __global__ void groupby_logsumexp_sumexp(
    const double* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    const double* __restrict__ maxs,
    uint32_t num_rows,
    double* __restrict__ sumexps  // Initialize to 0
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t group = group_ids[tid];
    double val = values[tid];
    double max_val = maxs[group];
    double exp_val = exp(val - max_val);

    // Atomic add for doubles using CAS
    unsigned long long* addr = (unsigned long long*)&sumexps[group];
    unsigned long long old = *addr;
    unsigned long long assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + exp_val;
        old = atomicCAS(addr, assumed, __double_as_longlong(sum));
    } while (assumed != old);
}

// LogSumExp Pass 3: Compute final result = max + log(sumexp)
extern "C" __global__ void groupby_logsumexp_final(
    const double* __restrict__ maxs,
    const double* __restrict__ sumexps,
    uint32_t num_groups,
    double* __restrict__ results
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_groups) return;

    results[gid] = maxs[gid] + log(sumexps[gid]);
}
```

**Step 2: Recompile PTX**

```bash
nvcc -ptx --gpu-architecture=sm_90 kernels/groupby.cu -o kernels/groupby.ptx
```

**Step 3: Add kernel names to provider.rs**

Add to `groupby_kernels` module:

```rust
pub mod groupby_kernels {
    // ... existing kernels ...
    pub const GROUPBY_LOGSUMEXP_MAX: &str = "groupby_logsumexp_max";
    pub const GROUPBY_LOGSUMEXP_SUMEXP: &str = "groupby_logsumexp_sumexp";
    pub const GROUPBY_LOGSUMEXP_FINAL: &str = "groupby_logsumexp_final";
}
```

**Step 4: Update kernel loading**

Add new kernels to load_ptx call for GROUPBY_MODULE.

**Step 5: Implement LogSumExp case in groupby_agg**

Replace the `AggOp::LogSumExp` error case (around line 1413) with:

```rust
AggOp::LogSumExp => {
    // LogSumExp requires F64 values
    let col_type = input.schema().column_type(value_col)
        .ok_or_else(|| XlogError::Kernel("Value column not found".to_string()))?;
    if col_type != ScalarType::F64 {
        return Err(XlogError::Kernel(
            "LogSumExp requires F64 column".to_string()
        ));
    }

    // Get values as f64 view
    let values_col = input.column(value_col)
        .ok_or_else(|| XlogError::Kernel("Value column not found".to_string()))?;
    let values_view: CudaView<f64> = unsafe {
        std::mem::transmute(values_col.slice(..))
    };

    // Allocate intermediate buffers
    let mut maxs = self.memory.alloc::<f64>(num_groups as usize)?;
    let mut sumexps = self.memory.alloc::<f64>(num_groups as usize)?;
    let mut output = self.memory.alloc::<f64>(num_groups as usize)?;

    // Initialize maxs to -INFINITY, sumexps to 0
    let neg_inf = vec![f64::NEG_INFINITY; num_groups as usize];
    let zeros = vec![0.0f64; num_groups as usize];
    device.htod_sync_copy_into(&neg_inf, &mut maxs)
        .map_err(|e| XlogError::Kernel(format!("Failed to init maxs: {}", e)))?;
    device.htod_sync_copy_into(&zeros, &mut sumexps)
        .map_err(|e| XlogError::Kernel(format!("Failed to init sumexps: {}", e)))?;

    // Pass 1: Find max per group
    let max_fn = device.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_LOGSUMEXP_MAX)
        .ok_or_else(|| XlogError::Kernel("groupby_logsumexp_max not found".to_string()))?;
    unsafe {
        max_fn.clone().launch(config, (&values_view, &group_ids, num_rows, &maxs))
    }.map_err(|e| XlogError::Kernel(format!("logsumexp_max failed: {}", e)))?;

    // Pass 2: Sum exp(x - max) per group
    let sumexp_fn = device.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_LOGSUMEXP_SUMEXP)
        .ok_or_else(|| XlogError::Kernel("groupby_logsumexp_sumexp not found".to_string()))?;
    unsafe {
        sumexp_fn.clone().launch(config, (&values_view, &group_ids, &maxs, num_rows, &sumexps))
    }.map_err(|e| XlogError::Kernel(format!("logsumexp_sumexp failed: {}", e)))?;

    // Pass 3: Compute final result
    let final_config = LaunchConfig {
        grid_dim: ((num_groups as u32 + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let final_fn = device.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_LOGSUMEXP_FINAL)
        .ok_or_else(|| XlogError::Kernel("groupby_logsumexp_final not found".to_string()))?;
    unsafe {
        final_fn.clone().launch(final_config, (&maxs, &sumexps, num_groups as u32, &output))
    }.map_err(|e| XlogError::Kernel(format!("logsumexp_final failed: {}", e)))?;

    self.device.synchronize()?;

    // Read back as bytes
    let mut host_output = vec![0.0f64; num_groups as usize];
    device.dtoh_sync_copy_into(&output, &mut host_output)
        .map_err(|e| XlogError::Kernel(format!("Failed to read logsumexp output: {}", e)))?;
    let bytes: Vec<u8> = host_output.iter().flat_map(|v| v.to_le_bytes()).collect();
    (bytes, ScalarType::F64)
}
```

**Step 6: Add download_column_f64 helper**

```rust
/// Download a column from GPU memory as f64 values
pub fn download_column_f64(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<f64>> {
    let col = buffer.column(col_idx).ok_or_else(|| {
        XlogError::Kernel(format!("Column {} not found", col_idx))
    })?;

    if buffer.num_rows == 0 {
        return Ok(vec![]);
    }

    let num_bytes = (buffer.num_rows as usize) * std::mem::size_of::<f64>();
    let mut bytes = vec![0u8; num_bytes];
    self.device
        .inner()
        .dtoh_sync_copy_into(col, &mut bytes)
        .map_err(|e| XlogError::Kernel(format!("Failed to download column: {}", e)))?;

    Ok(bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}
```

**Step 7: Add tests**

```rust
#[test]
fn test_groupby_logsumexp() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no CUDA device");
            return;
        }
    };

    // Create buffer with F64 values and U32 keys
    // Group 0: log(e^1 + e^2) = log(e + e^2) ≈ 2.31
    // Group 1: log(e^3 + e^4) ≈ 4.31
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::F64),
    ]);

    let keys: Vec<u8> = [0u32, 0, 1, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
    let vals: Vec<u8> = [1.0f64, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();

    let device = provider.device().inner();
    let d_keys = device.htod_sync_copy(&keys).unwrap();
    let d_vals = device.htod_sync_copy(&vals).unwrap();
    let buffer = CudaBuffer::from_columns(vec![d_keys, d_vals], 4, schema);

    // Sort by key first (groupby requires sorted input)
    let sorted = provider.sort(&buffer, &[0]).unwrap();

    let result = provider.groupby_agg(&sorted, &[0], AggOp::LogSumExp, 1).unwrap();

    assert_eq!(result.num_rows(), 2);

    let logsumexp_vals = provider.download_column_f64(&result, 1).unwrap();

    // Verify numerical results (with tolerance for floating point)
    let expected_0 = (1.0f64.exp() + 2.0f64.exp()).ln(); // ≈ 2.31
    let expected_1 = (3.0f64.exp() + 4.0f64.exp()).ln(); // ≈ 4.31

    assert!((logsumexp_vals[0] - expected_0).abs() < 0.01);
    assert!((logsumexp_vals[1] - expected_1).abs() < 0.01);
}
```

**Step 8: Run tests**

```bash
cargo test -p xlog-cuda test_groupby_logsumexp -- --nocapture
```

**Step 9: Commit**

```bash
git add kernels/groupby.cu kernels/groupby.ptx crates/xlog-cuda/src/provider/mod.rs
git commit -m "feat(xlog-cuda): implement LogSumExp aggregation

Adds numerically stable LogSumExp using 3-pass algorithm:
1. Find max per group
2. Sum exp(x - max) per group
3. Compute log(sum) + max

Required for probabilistic reasoning in xlog-prob tier."
```

---

## Task 4: Extend Set Operations to Multiple Types

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (union_gpu, diff_gpu)
- Test: Add multi-type set op tests

**Problem:** `union_gpu` and `diff_gpu` only work with single U32 columns. Should support U64, I64.

**Note:** This is complex because GPU sort only supports U32. For non-U32 types, we'll use CPU-based operations with GPU compaction.

**Step 1: Refactor union_gpu to handle multiple types**

Update `union_gpu` to check column type and dispatch appropriately:

```rust
pub fn union_gpu(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
    // Handle empty cases
    if a.is_empty() && b.is_empty() {
        return self.create_empty_buffer(a.schema().clone());
    }
    if a.is_empty() {
        return self.dedup(b, &[0]);
    }
    if b.is_empty() {
        return self.dedup(a, &[0]);
    }

    // Verify schemas match
    if a.schema() != b.schema() {
        return Err(XlogError::Kernel("Union requires matching schemas".to_string()));
    }

    // Check column type for optimized path
    let col_type = a.schema().column_type(0)
        .ok_or_else(|| XlogError::Kernel("No columns".to_string()))?;

    match col_type {
        ScalarType::U32 => self.union_gpu_u32(a, b),
        ScalarType::U64 | ScalarType::I64 | ScalarType::F64 => {
            // For non-U32 types, use CPU-based concat + sort + dedup
            self.union_cpu_sort(a, b)
        }
        _ => Err(XlogError::Kernel(format!(
            "Union not supported for type {:?}", col_type
        ))),
    }
}

/// U32-optimized union using GPU sort
fn union_gpu_u32(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
    // ... existing union_gpu implementation ...
}

/// CPU-sort based union for non-U32 types
fn union_cpu_sort(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
    // Download both buffers
    let mut a_data = self.download_buffer_rows(a)?;
    let mut b_data = self.download_buffer_rows(b)?;

    // Concatenate
    a_data.extend(b_data);

    // Sort by first column bytes
    let col_size = a.schema().column_type(0).map(|t| t.size_bytes()).unwrap_or(4);
    a_data.sort_by(|x, y| x[..col_size].cmp(&y[..col_size]));

    // Dedup consecutive equal rows
    a_data.dedup_by(|x, y| x[..col_size] == y[..col_size]);

    // Upload result
    self.upload_buffer_rows(&a_data, a.schema().clone())
}
```

**Step 2: Add helper methods for row-based operations**

```rust
/// Download buffer as row-major byte arrays
fn download_buffer_rows(&self, buffer: &CudaBuffer) -> Result<Vec<Vec<u8>>> {
    let num_rows = buffer.num_rows() as usize;
    let row_size: usize = buffer.schema().columns.iter()
        .map(|(_, t)| t.size_bytes())
        .sum();

    // Download all columns
    let mut col_data: Vec<Vec<u8>> = Vec::new();
    for col_idx in 0..buffer.arity() {
        let col = buffer.column(col_idx)
            .ok_or_else(|| XlogError::Kernel("Column not found".to_string()))?;
        let mut data = vec![0u8; col.len()];
        self.device.inner().dtoh_sync_copy_into(col, &mut data)
            .map_err(|e| XlogError::Kernel(format!("Download failed: {}", e)))?;
        col_data.push(data);
    }

    // Convert to row-major
    let mut rows = Vec::with_capacity(num_rows);
    for row_idx in 0..num_rows {
        let mut row = Vec::with_capacity(row_size);
        for (col_idx, col) in col_data.iter().enumerate() {
            let col_size = buffer.schema().column_type(col_idx)
                .map(|t| t.size_bytes()).unwrap_or(4);
            let start = row_idx * col_size;
            row.extend_from_slice(&col[start..start + col_size]);
        }
        rows.push(row);
    }

    Ok(rows)
}

/// Upload row-major byte arrays as buffer
fn upload_buffer_rows(&self, rows: &[Vec<u8>], schema: Schema) -> Result<CudaBuffer> {
    if rows.is_empty() {
        return self.create_empty_buffer(schema);
    }

    let num_rows = rows.len();
    let mut columns = Vec::new();
    let mut offset = 0;

    for (_, col_type) in &schema.columns {
        let col_size = col_type.size_bytes();
        let mut col_bytes = Vec::with_capacity(num_rows * col_size);

        for row in rows {
            col_bytes.extend_from_slice(&row[offset..offset + col_size]);
        }

        let d_col = self.device.inner().htod_sync_copy(&col_bytes)
            .map_err(|e| XlogError::Kernel(format!("Upload failed: {}", e)))?;
        columns.push(d_col);
        offset += col_size;
    }

    Ok(CudaBuffer::from_columns(columns, num_rows as u64, schema))
}
```

**Step 3: Similarly update diff_gpu**

Apply same pattern to `diff_gpu` for multi-type support.

**Step 4: Add tests**

```rust
#[test]
fn test_union_u64() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => return,
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    let a_bytes: Vec<u8> = [1u64, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = [2u64, 3, 4].iter().flat_map(|v| v.to_le_bytes()).collect();

    let device = provider.device().inner();
    let a = CudaBuffer::from_columns(
        vec![device.htod_sync_copy(&a_bytes).unwrap()],
        3, schema.clone()
    );
    let b = CudaBuffer::from_columns(
        vec![device.htod_sync_copy(&b_bytes).unwrap()],
        3, schema
    );

    let result = provider.union_gpu(&a, &b).unwrap();
    assert_eq!(result.num_rows(), 4); // 1, 2, 3, 4
}
```

**Step 5: Run tests**

```bash
cargo test -p xlog-cuda test_union_u64 -- --nocapture
cargo test -p xlog-cuda set_ops -- --nocapture
```

**Step 6: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs
git commit -m "feat(xlog-cuda): extend set operations to U64/I64/F64 types

Union and diff now support multiple scalar types using CPU sort
fallback for non-U32 columns. GPU sort path retained for U32."
```

---

## Final Verification

**Run full test suite:**

```bash
cargo test -p xlog-cuda -- --nocapture
```

**Expected:** All P2-related tests pass. Pre-existing union/diff bugs may still fail for large inputs (separate issue with GPU sort).

---

## Summary

| Task | Issue | Fix |
|------|-------|-----|
| 1 | Join output 1M limit | Add configurable max_output parameter |
| 2 | No float support | Wire existing filter_compare_f64 kernel |
| 3 | No LogSumExp | Implement 3-pass numerically stable algorithm |
| 4 | U32-only set ops | Add CPU sort fallback for other types |
