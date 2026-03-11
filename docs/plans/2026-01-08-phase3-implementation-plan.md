# Phase 3: Production-Grade GPU Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the MVP GPU Datalog engine into production-grade by implementing GPU sorting, multi-column joins, all join types, GPU filtering, complete aggregation, and eliminating host roundtrips.

**Architecture:** Bottom-up kernel implementation starting with prefix sum (foundation), then sort, filter, joins, set ops, and aggregation. Each kernel gets Rust wrappers and integration into the executor.

**Tech Stack:** CUDA kernels (.cu → .ptx via nvcc), cudarc for Rust bindings, xlog-cuda provider, xlog-runtime executor.

---

## Task 1: GPU Prefix Sum Kernel

**Files:**
- Create: `kernels/scan.cu`
- Create: `kernels/scan.ptx` (compiled)
- Modify: `crates/xlog-cuda/build.rs` (add scan.cu compilation)

**Step 1: Write scan.cu with block-level scan**

```cuda
// kernels/scan.cu
#include <cstdint>

#define BLOCK_SIZE 256

// Shared memory block-level inclusive scan (Blelloch)
extern "C" __global__ void block_inclusive_scan(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t* __restrict__ block_sums,
    uint32_t n
) {
    __shared__ uint32_t temp[BLOCK_SIZE * 2];

    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    temp[tid] = (gid < n) ? input[gid] : 0;
    __syncthreads();

    // Up-sweep (reduce)
    for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
        uint32_t index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Store block sum before down-sweep
    if (tid == blockDim.x - 1) {
        if (block_sums != nullptr) {
            block_sums[blockIdx.x] = temp[tid];
        }
    }

    // Down-sweep for exclusive scan, then shift for inclusive
    if (tid == blockDim.x - 1) {
        temp[tid] = 0;
    }
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        uint32_t index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            uint32_t t = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += t;
        }
        __syncthreads();
    }

    // Write output (shift by one for inclusive scan)
    if (gid < n) {
        output[gid] = temp[tid] + input[gid];
    }
}

// Add block offsets to convert block-local scans to global scan
extern "C" __global__ void add_block_offsets(
    uint32_t* __restrict__ data,
    const uint32_t* __restrict__ block_offsets,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && blockIdx.x > 0) {
        data[gid] += block_offsets[blockIdx.x - 1];
    }
}

// Exclusive prefix sum of u8 mask (for stream compaction)
extern "C" __global__ void exclusive_scan_mask(
    const uint8_t* __restrict__ mask,
    uint32_t* __restrict__ prefix_sum,
    uint32_t n
) {
    __shared__ uint32_t temp[BLOCK_SIZE];

    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load mask as u32
    temp[tid] = (gid < n) ? (uint32_t)mask[gid] : 0;
    __syncthreads();

    // Exclusive scan within block
    for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
        uint32_t val = 0;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Write exclusive scan (shift by 1)
    if (gid < n) {
        prefix_sum[gid] = (tid == 0) ? 0 : temp[tid - 1];
    }
}

// Count total 1s in mask
extern "C" __global__ void count_mask(
    const uint8_t* __restrict__ mask,
    uint32_t n,
    uint32_t* __restrict__ count
) {
    __shared__ uint32_t block_count;
    if (threadIdx.x == 0) block_count = 0;
    __syncthreads();

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && mask[gid]) {
        atomicAdd(&block_count, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, block_count);
    }
}
```

**Step 2: Update build.rs to compile scan.cu**

In `crates/xlog-cuda/build.rs`, add `"scan"` to the kernel list.

**Step 3: Compile and verify PTX generation**

Run: `cargo build -p xlog-cuda 2>&1 | head -20`
Expected: Build succeeds, `kernels/scan.ptx` generated

**Step 4: Commit**

```bash
git add kernels/scan.cu crates/xlog-cuda/build.rs
git commit -m "feat(xlog-cuda): add GPU prefix sum kernel"
```

---

## Task 2: Prefix Sum Rust Wrapper

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Create: `crates/xlog-cuda/tests/scan_tests.rs`

**Step 1: Write failing test for prefix_sum_mask**

```rust
// crates/xlog-cuda/tests/scan_tests.rs
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager, MemoryBudget};
use std::sync::Arc;

fn setup_provider() -> Option<CudaKernelProvider> {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return None;
    }
    let device = Arc::new(CudaDevice::new(0).unwrap());
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::default()));
    Some(CudaKernelProvider::new(device, memory).unwrap())
}

#[test]
fn test_prefix_sum_mask_simple() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // mask: [1, 0, 1, 1, 0, 1]
    // exclusive prefix sum: [0, 1, 1, 2, 3, 3]
    // total count: 4
    let mask = vec![1u8, 0, 1, 1, 0, 1];
    let (prefix_sum, count) = provider.prefix_sum_mask(&mask).unwrap();

    assert_eq!(count, 4);
    assert_eq!(prefix_sum, vec![0u32, 1, 1, 2, 3, 3]);
}

#[test]
fn test_prefix_sum_mask_empty() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let mask = vec![0u8; 10];
    let (prefix_sum, count) = provider.prefix_sum_mask(&mask).unwrap();

    assert_eq!(count, 0);
    assert_eq!(prefix_sum, vec![0u32; 10]);
}

#[test]
fn test_prefix_sum_mask_all_ones() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let mask = vec![1u8; 5];
    let (prefix_sum, count) = provider.prefix_sum_mask(&mask).unwrap();

    assert_eq!(count, 5);
    assert_eq!(prefix_sum, vec![0u32, 1, 2, 3, 4]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test scan_tests 2>&1 | tail -20`
Expected: FAIL with "method `prefix_sum_mask` not found"

**Step 3: Add scan module loading to CudaKernelProvider**

In `crates/xlog-cuda/src/provider/mod.rs`, add:
- Field: `scan_module: CudaModule`
- Load PTX in `new()`: `include_str!("../../../kernels/scan.ptx")`

**Step 4: Implement prefix_sum_mask**

```rust
impl CudaKernelProvider {
    /// Compute exclusive prefix sum of u8 mask, returns (prefix_sum_vec, total_count)
    pub fn prefix_sum_mask(&self, mask: &[u8]) -> Result<(Vec<u32>, u32)> {
        if mask.is_empty() {
            return Ok((vec![], 0));
        }

        let n = mask.len();
        let device = self.device.inner();

        // Upload mask to GPU
        let d_mask = device.htod_sync_copy(mask)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload mask: {}", e)))?;

        // Allocate output
        let mut d_prefix_sum = unsafe { device.alloc::<u32>(n) }
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc prefix_sum: {}", e)))?;

        // Allocate count
        let mut d_count = device.htod_sync_copy(&[0u32])
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc count: {}", e)))?;

        // Launch exclusive_scan_mask kernel
        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;

        let scan_fn = self.scan_module
            .get_fn("exclusive_scan_mask")
            .map_err(|e| XlogError::Kernel(format!("Failed to get exclusive_scan_mask: {}", e)))?;

        unsafe {
            scan_fn.launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&d_mask, &mut d_prefix_sum, n as u32),
            )
        }.map_err(|e| XlogError::Kernel(format!("Failed to launch exclusive_scan_mask: {}", e)))?;

        // Launch count_mask kernel
        let count_fn = self.scan_module
            .get_fn("count_mask")
            .map_err(|e| XlogError::Kernel(format!("Failed to get count_mask: {}", e)))?;

        unsafe {
            count_fn.launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&d_mask, n as u32, &mut d_count),
            )
        }.map_err(|e| XlogError::Kernel(format!("Failed to launch count_mask: {}", e)))?;

        // Synchronize and download results
        self.device.synchronize()?;

        let prefix_sum = device.dtoh_sync_copy(&d_prefix_sum)
            .map_err(|e| XlogError::Kernel(format!("Failed to download prefix_sum: {}", e)))?;

        let count_vec = device.dtoh_sync_copy(&d_count)
            .map_err(|e| XlogError::Kernel(format!("Failed to download count: {}", e)))?;

        Ok((prefix_sum, count_vec[0]))
    }
}
```

**Step 5: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test scan_tests 2>&1 | tail -20`
Expected: 3 tests PASS

**Step 6: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/scan_tests.rs
git commit -m "feat(xlog-cuda): add prefix_sum_mask kernel wrapper"
```

---

## Task 3: GPU Radix Sort Kernel

**Files:**
- Create: `kernels/sort.cu`
- Modify: `crates/xlog-cuda/build.rs`

**Step 1: Write sort.cu with radix sort**

```cuda
// kernels/sort.cu
#include <cstdint>

#define BLOCK_SIZE 256
#define RADIX_BITS 4
#define RADIX_SIZE (1 << RADIX_BITS)  // 16 buckets

// Compute histogram of radix digits for current pass
extern "C" __global__ void radix_histogram(
    const uint32_t* __restrict__ keys,
    uint32_t num_rows,
    uint32_t* __restrict__ histograms,  // [grid_size * RADIX_SIZE]
    uint32_t shift
) {
    __shared__ uint32_t local_hist[RADIX_SIZE];

    // Initialize local histogram
    if (threadIdx.x < RADIX_SIZE) {
        local_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_rows) {
        uint32_t digit = (keys[gid] >> shift) & (RADIX_SIZE - 1);
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    // Write local histogram to global
    if (threadIdx.x < RADIX_SIZE) {
        histograms[blockIdx.x * RADIX_SIZE + threadIdx.x] = local_hist[threadIdx.x];
    }
}

// Scatter keys to sorted positions based on prefix sums
extern "C" __global__ void radix_scatter(
    const uint32_t* __restrict__ keys_in,
    const uint32_t* __restrict__ indices_in,
    uint32_t* __restrict__ keys_out,
    uint32_t* __restrict__ indices_out,
    const uint32_t* __restrict__ prefix_sums,  // [RADIX_SIZE] global prefix sums
    uint32_t* __restrict__ local_offsets,      // [grid_size * RADIX_SIZE] for local counting
    uint32_t num_rows,
    uint32_t shift
) {
    __shared__ uint32_t local_prefix[RADIX_SIZE];

    // Load global prefix sums
    if (threadIdx.x < RADIX_SIZE) {
        local_prefix[threadIdx.x] = prefix_sums[threadIdx.x];
        // Add local offset from previous blocks
        for (uint32_t b = 0; b < blockIdx.x; b++) {
            local_prefix[threadIdx.x] += local_offsets[b * RADIX_SIZE + threadIdx.x];
        }
    }
    __syncthreads();

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_rows) {
        uint32_t key = keys_in[gid];
        uint32_t digit = (key >> shift) & (RADIX_SIZE - 1);
        uint32_t pos = atomicAdd(&local_prefix[digit], 1);
        keys_out[pos] = key;
        indices_out[pos] = indices_in[gid];
    }
}

// Initialize indices to identity permutation
extern "C" __global__ void init_indices(
    uint32_t* __restrict__ indices,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        indices[gid] = gid;
    }
}

// Apply permutation to reorder a column
extern "C" __global__ void apply_permutation_u32(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ permutation,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        output[gid] = input[permutation[gid]];
    }
}

// Apply permutation to reorder bytes (for any column type)
extern "C" __global__ void apply_permutation_bytes(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const uint32_t* __restrict__ permutation,
    uint32_t num_rows,
    uint32_t elem_size
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_rows) {
        uint32_t src_idx = permutation[gid];
        for (uint32_t b = 0; b < elem_size; b++) {
            output[gid * elem_size + b] = input[src_idx * elem_size + b];
        }
    }
}
```

**Step 2: Update build.rs**

Add `"sort"` to kernel list in `crates/xlog-cuda/build.rs`.

**Step 3: Compile and verify**

Run: `cargo build -p xlog-cuda 2>&1 | head -20`
Expected: Build succeeds, `kernels/sort.ptx` generated

**Step 4: Commit**

```bash
git add kernels/sort.cu crates/xlog-cuda/build.rs
git commit -m "feat(xlog-cuda): add GPU radix sort kernel"
```

---

## Task 4: Radix Sort Rust Wrapper

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Create: `crates/xlog-cuda/tests/sort_tests.rs`

**Step 1: Write failing test**

```rust
// crates/xlog-cuda/tests/sort_tests.rs
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager, MemoryBudget, CudaBuffer};
use xlog_ir::Schema;
use std::sync::Arc;

fn setup_provider() -> Option<CudaKernelProvider> {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return None;
    }
    let device = Arc::new(CudaDevice::new(0).unwrap());
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::default()));
    Some(CudaKernelProvider::new(device, memory).unwrap())
}

#[test]
fn test_sort_single_column() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Unsorted: [5, 2, 8, 1, 9, 3]
    // Sorted:   [1, 2, 3, 5, 8, 9]
    let keys = vec![5u32, 2, 8, 1, 9, 3];
    let schema = Schema::new(vec![xlog_ir::ScalarType::U32]);

    let buffer = provider.create_buffer_from_u32(&keys, schema.clone()).unwrap();
    let sorted = provider.sort(&buffer, &[0]).unwrap();

    let result = provider.download_column_u32(&sorted, 0).unwrap();
    assert_eq!(result, vec![1, 2, 3, 5, 8, 9]);
}

#[test]
fn test_sort_preserves_other_columns() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Sort by col0, col1 should follow
    // col0: [3, 1, 2] -> [1, 2, 3]
    // col1: [30, 10, 20] -> [10, 20, 30]
    let col0 = vec![3u32, 1, 2];
    let col1 = vec![30u32, 10, 20];
    let schema = Schema::new(vec![xlog_ir::ScalarType::U32, xlog_ir::ScalarType::U32]);

    let buffer = provider.create_buffer_from_u32_columns(&[&col0, &col1], schema).unwrap();
    let sorted = provider.sort(&buffer, &[0]).unwrap();

    let result0 = provider.download_column_u32(&sorted, 0).unwrap();
    let result1 = provider.download_column_u32(&sorted, 1).unwrap();

    assert_eq!(result0, vec![1, 2, 3]);
    assert_eq!(result1, vec![10, 20, 30]);
}

#[test]
fn test_sort_empty() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![xlog_ir::ScalarType::U32]);
    let buffer = provider.create_empty_buffer(schema).unwrap();
    let sorted = provider.sort(&buffer, &[0]).unwrap();

    assert_eq!(sorted.num_rows, 0);
}
```

**Step 2: Run test to verify failure**

Run: `cargo test -p xlog-cuda --test sort_tests 2>&1 | tail -20`
Expected: FAIL with "method `sort` not found"

**Step 3: Add sort module to provider and implement sort()**

In `crates/xlog-cuda/src/provider/mod.rs`:
- Add field: `sort_module: CudaModule`
- Load PTX in `new()`
- Implement `sort()` method using radix sort passes

```rust
impl CudaKernelProvider {
    /// Sort buffer by key columns using GPU radix sort
    pub fn sort(&self, input: &CudaBuffer, key_cols: &[usize]) -> Result<CudaBuffer> {
        if input.num_rows == 0 {
            return Ok(input.clone());
        }

        let n = input.num_rows as u32;
        let device = self.device.inner();

        // For now, support single u32 key column
        if key_cols.len() != 1 {
            return Err(XlogError::Kernel("Multi-column sort not yet implemented".into()));
        }

        let key_col = key_cols[0];

        // Download keys (TODO: keep on GPU)
        let keys = self.download_column_u32(input, key_col)?;

        // Upload keys
        let mut d_keys_in = device.htod_sync_copy(&keys)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload keys: {}", e)))?;
        let mut d_keys_out = unsafe { device.alloc::<u32>(n as usize) }
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc keys_out: {}", e)))?;

        // Initialize indices
        let mut d_indices_in = unsafe { device.alloc::<u32>(n as usize) }
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc indices: {}", e)))?;
        let mut d_indices_out = unsafe { device.alloc::<u32>(n as usize) }
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc indices_out: {}", e)))?;

        let block_size = 256u32;
        let grid_size = (n + block_size - 1) / block_size;

        // Initialize indices to [0, 1, 2, ...]
        let init_fn = self.sort_module.get_fn("init_indices")
            .map_err(|e| XlogError::Kernel(format!("Failed to get init_indices: {}", e)))?;

        unsafe {
            init_fn.launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut d_indices_in, n),
            )
        }.map_err(|e| XlogError::Kernel(format!("Failed to init indices: {}", e)))?;

        // Radix sort passes (8 passes for 32-bit keys, 4 bits per pass)
        for pass in 0..8 {
            let shift = pass * 4;

            // Compute histogram
            let hist_size = grid_size as usize * 16;
            let mut d_histograms = unsafe { device.alloc::<u32>(hist_size) }
                .map_err(|e| XlogError::Kernel(format!("Failed to alloc histograms: {}", e)))?;

            let hist_fn = self.sort_module.get_fn("radix_histogram")
                .map_err(|e| XlogError::Kernel(format!("Failed to get radix_histogram: {}", e)))?;

            unsafe {
                hist_fn.launch(
                    cudarc::driver::LaunchConfig {
                        grid_dim: (grid_size, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&d_keys_in, n, &mut d_histograms, shift),
                )
            }.map_err(|e| XlogError::Kernel(format!("Failed to compute histogram: {}", e)))?;

            // Compute prefix sums on CPU for now (TODO: GPU prefix sum)
            self.device.synchronize()?;
            let histograms = device.dtoh_sync_copy(&d_histograms)
                .map_err(|e| XlogError::Kernel(format!("Failed to download histograms: {}", e)))?;

            // Sum histograms across blocks
            let mut global_hist = [0u32; 16];
            for block in 0..grid_size as usize {
                for digit in 0..16 {
                    global_hist[digit] += histograms[block * 16 + digit];
                }
            }

            // Compute prefix sums
            let mut prefix_sums = [0u32; 16];
            let mut sum = 0u32;
            for digit in 0..16 {
                prefix_sums[digit] = sum;
                sum += global_hist[digit];
            }

            let d_prefix_sums = device.htod_sync_copy(&prefix_sums)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload prefix_sums: {}", e)))?;

            // Scatter
            let scatter_fn = self.sort_module.get_fn("radix_scatter")
                .map_err(|e| XlogError::Kernel(format!("Failed to get radix_scatter: {}", e)))?;

            unsafe {
                scatter_fn.launch(
                    cudarc::driver::LaunchConfig {
                        grid_dim: (grid_size, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&d_keys_in, &d_indices_in, &mut d_keys_out, &mut d_indices_out,
                     &d_prefix_sums, &d_histograms, n, shift),
                )
            }.map_err(|e| XlogError::Kernel(format!("Failed to scatter: {}", e)))?;

            // Swap buffers for next pass
            std::mem::swap(&mut d_keys_in, &mut d_keys_out);
            std::mem::swap(&mut d_indices_in, &mut d_indices_out);
        }

        // d_keys_in now contains sorted keys, d_indices_in contains permutation
        self.device.synchronize()?;
        let permutation = device.dtoh_sync_copy(&d_indices_in)
            .map_err(|e| XlogError::Kernel(format!("Failed to download permutation: {}", e)))?;

        // Apply permutation to all columns
        self.apply_permutation(input, &permutation)
    }

    /// Apply permutation to reorder all columns in buffer
    fn apply_permutation(&self, input: &CudaBuffer, permutation: &[u32]) -> Result<CudaBuffer> {
        // Implementation: reorder each column by permutation indices
        // For each column, download, reorder on CPU, upload
        // TODO: GPU-based permutation

        let mut new_columns = Vec::with_capacity(input.columns.len());

        for (col_idx, _col) in input.columns.iter().enumerate() {
            let data = self.download_column_u32(input, col_idx)?;
            let mut reordered = vec![0u32; data.len()];
            for (new_idx, &old_idx) in permutation.iter().enumerate() {
                reordered[new_idx] = data[old_idx as usize];
            }

            let device = self.device.inner();
            let d_col = device.htod_sync_copy(&reordered)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload reordered col: {}", e)))?;

            // Convert to u8 slice
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    reordered.as_ptr() as *const u8,
                    reordered.len() * 4,
                )
            };
            let d_bytes = device.htod_sync_copy(bytes)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload bytes: {}", e)))?;

            new_columns.push(d_bytes);
        }

        Ok(CudaBuffer {
            columns: new_columns,
            num_rows: input.num_rows,
            schema: input.schema.clone(),
        })
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test sort_tests 2>&1 | tail -20`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/sort_tests.rs
git commit -m "feat(xlog-cuda): add GPU radix sort wrapper"
```

---

## Task 5: GPU Filter Kernel

**Files:**
- Create: `kernels/filter.cu`
- Modify: `crates/xlog-cuda/build.rs`

**Step 1: Write filter.cu**

```cuda
// kernels/filter.cu
#include <cstdint>

// Comparison operators
#define OP_EQ 0
#define OP_NE 1
#define OP_LT 2
#define OP_LE 3
#define OP_GT 4
#define OP_GE 5

// Compare i64 column against constant
extern "C" __global__ void filter_compare_i64(
    const int64_t* __restrict__ column,
    int64_t constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    int64_t val = column[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (val == constant); break;
        case OP_NE: result = (val != constant); break;
        case OP_LT: result = (val < constant); break;
        case OP_LE: result = (val <= constant); break;
        case OP_GT: result = (val > constant); break;
        case OP_GE: result = (val >= constant); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

// Compare u32 column against constant
extern "C" __global__ void filter_compare_u32(
    const uint32_t* __restrict__ column,
    uint32_t constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint32_t val = column[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (val == constant); break;
        case OP_NE: result = (val != constant); break;
        case OP_LT: result = (val < constant); break;
        case OP_LE: result = (val <= constant); break;
        case OP_GT: result = (val > constant); break;
        case OP_GE: result = (val >= constant); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

// Compare f64 column against constant
extern "C" __global__ void filter_compare_f64(
    const double* __restrict__ column,
    double constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    double val = column[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (val == constant); break;
        case OP_NE: result = (val != constant); break;
        case OP_LT: result = (val < constant); break;
        case OP_LE: result = (val <= constant); break;
        case OP_GT: result = (val > constant); break;
        case OP_GE: result = (val >= constant); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

// Combine masks with AND
extern "C" __global__ void mask_and(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    uint8_t* __restrict__ out,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = (a[gid] && b[gid]) ? 1 : 0;
    }
}

// Combine masks with OR
extern "C" __global__ void mask_or(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    uint8_t* __restrict__ out,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = (a[gid] || b[gid]) ? 1 : 0;
    }
}

// Negate mask
extern "C" __global__ void mask_not(
    const uint8_t* __restrict__ a,
    uint8_t* __restrict__ out,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = a[gid] ? 0 : 1;
    }
}

// Stream compaction: gather rows where mask is 1
extern "C" __global__ void compact_by_mask(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ mask,
    const uint32_t* __restrict__ prefix_sum,
    uint32_t num_rows,
    uint32_t row_bytes,
    uint8_t* __restrict__ output
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    if (mask[gid]) {
        uint32_t out_idx = prefix_sum[gid];
        for (uint32_t b = 0; b < row_bytes; b++) {
            output[out_idx * row_bytes + b] = input[gid * row_bytes + b];
        }
    }
}

// Compact single u32 column by mask
extern "C" __global__ void compact_u32_by_mask(
    const uint32_t* __restrict__ input,
    const uint8_t* __restrict__ mask,
    const uint32_t* __restrict__ prefix_sum,
    uint32_t num_rows,
    uint32_t* __restrict__ output
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    if (mask[gid]) {
        uint32_t out_idx = prefix_sum[gid];
        output[out_idx] = input[gid];
    }
}
```

**Step 2: Update build.rs**

Add `"filter"` to kernel list.

**Step 3: Compile and verify**

Run: `cargo build -p xlog-cuda 2>&1 | head -20`
Expected: Build succeeds, `kernels/filter.ptx` generated

**Step 4: Commit**

```bash
git add kernels/filter.cu crates/xlog-cuda/build.rs
git commit -m "feat(xlog-cuda): add GPU filter and compaction kernels"
```

---

## Task 6: GPU Filter Rust Wrapper

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Create: `crates/xlog-cuda/tests/filter_tests.rs`

**Step 1: Write failing test**

```rust
// crates/xlog-cuda/tests/filter_tests.rs
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager, MemoryBudget};
use xlog_ir::Schema;
use std::sync::Arc;

fn setup_provider() -> Option<CudaKernelProvider> {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return None;
    }
    let device = Arc::new(CudaDevice::new(0).unwrap());
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::default()));
    Some(CudaKernelProvider::new(device, memory).unwrap())
}

#[test]
fn test_filter_u32_eq() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Filter col0 == 2
    // col0: [1, 2, 3, 2, 4] -> [2, 2]
    // col1: [10, 20, 30, 40, 50] -> [20, 40]
    let col0 = vec![1u32, 2, 3, 2, 4];
    let col1 = vec![10u32, 20, 30, 40, 50];
    let schema = Schema::new(vec![xlog_ir::ScalarType::U32, xlog_ir::ScalarType::U32]);

    let buffer = provider.create_buffer_from_u32_columns(&[&col0, &col1], schema).unwrap();
    let filtered = provider.filter_u32_eq(&buffer, 0, 2).unwrap();

    let result0 = provider.download_column_u32(&filtered, 0).unwrap();
    let result1 = provider.download_column_u32(&filtered, 1).unwrap();

    assert_eq!(result0, vec![2, 2]);
    assert_eq!(result1, vec![20, 40]);
}

#[test]
fn test_filter_u32_gt() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Filter col0 > 2
    let col0 = vec![1u32, 2, 3, 4, 5];
    let schema = Schema::new(vec![xlog_ir::ScalarType::U32]);

    let buffer = provider.create_buffer_from_u32(&col0, schema).unwrap();
    let filtered = provider.filter_u32_gt(&buffer, 0, 2).unwrap();

    let result = provider.download_column_u32(&filtered, 0).unwrap();
    assert_eq!(result, vec![3, 4, 5]);
}

#[test]
fn test_filter_no_matches() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let col0 = vec![1u32, 2, 3];
    let schema = Schema::new(vec![xlog_ir::ScalarType::U32]);

    let buffer = provider.create_buffer_from_u32(&col0, schema).unwrap();
    let filtered = provider.filter_u32_eq(&buffer, 0, 99).unwrap();

    assert_eq!(filtered.num_rows, 0);
}

#[test]
fn test_filter_by_mask() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let col0 = vec![10u32, 20, 30, 40, 50];
    let mask = vec![1u8, 0, 1, 0, 1];
    let schema = Schema::new(vec![xlog_ir::ScalarType::U32]);

    let buffer = provider.create_buffer_from_u32(&col0, schema).unwrap();
    let filtered = provider.filter_by_mask(&buffer, &mask).unwrap();

    let result = provider.download_column_u32(&filtered, 0).unwrap();
    assert_eq!(result, vec![10, 30, 50]);
}
```

**Step 2: Run test to verify failure**

Run: `cargo test -p xlog-cuda --test filter_tests 2>&1 | tail -20`
Expected: FAIL

**Step 3: Implement filter methods in provider.rs**

Add filter module loading and implement:
- `filter_u32_eq()`
- `filter_u32_gt()`
- `filter_by_mask()`

**Step 4: Run test to verify pass**

Run: `cargo test -p xlog-cuda --test filter_tests 2>&1 | tail -20`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/filter_tests.rs
git commit -m "feat(xlog-cuda): add GPU filter wrapper with stream compaction"
```

---

## Task 7: Multi-Column Join Kernel

**Files:**
- Modify: `kernels/join.cu`

**Step 1: Add composite hash and v2 join kernels**

Add to `kernels/join.cu`:

```cuda
// Compute composite hash for multi-column keys
extern "C" __global__ void compute_composite_hash(
    const uint8_t* __restrict__ data,
    const uint32_t* __restrict__ col_offsets,
    const uint32_t* __restrict__ col_sizes,
    uint32_t num_key_cols,
    uint32_t num_rows,
    uint32_t row_stride,
    uint64_t* __restrict__ hashes
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint64_t hash = 0xcbf29ce484222325ULL;  // FNV-1a offset basis

    for (uint32_t c = 0; c < num_key_cols; c++) {
        uint32_t offset = col_offsets[c];
        uint32_t size = col_sizes[c];

        for (uint32_t b = 0; b < size; b++) {
            uint8_t byte = data[gid * row_stride + offset + b];
            hash ^= byte;
            hash *= 0x100000001b3ULL;  // FNV-1a prime
        }
    }

    hashes[gid] = hash;
}

// Build hash table with u64 hashes
extern "C" __global__ void hash_join_build_v2(
    const uint64_t* __restrict__ hashes,
    uint32_t num_rows,
    uint32_t* __restrict__ hash_table,  // [hash_table_size]: stores row index or 0xFFFFFFFF
    uint32_t* __restrict__ next_ptrs,   // [num_rows]: linked list next pointers
    uint32_t hash_table_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint64_t hash = hashes[tid];
    uint32_t bucket = (uint32_t)(hash % hash_table_size);

    // Atomic linked list insertion
    uint32_t old = atomicExch(&hash_table[bucket], tid);
    next_ptrs[tid] = old;
}

// Probe with u64 hashes, output matching indices
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
    uint32_t max_output
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_probe) return;

    uint64_t hash = probe_hashes[tid];
    uint32_t bucket = (uint32_t)(hash % hash_table_size);

    uint32_t current = hash_table[bucket];
    while (current != 0xFFFFFFFF) {
        if (build_hashes[current] == hash) {
            uint32_t out_idx = atomicAdd(output_count, 1);
            if (out_idx < max_output) {
                output_left[out_idx] = tid;
                output_right[out_idx] = current;
            }
        }
        current = next_ptrs[current];
    }
}

// Semi-join: mark probe rows that have any match
extern "C" __global__ void hash_join_semi(
    const uint64_t* __restrict__ probe_hashes,
    uint32_t num_probe,
    const uint32_t* __restrict__ hash_table,
    const uint64_t* __restrict__ build_hashes,
    const uint32_t* __restrict__ next_ptrs,
    uint32_t hash_table_size,
    uint8_t* __restrict__ has_match
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_probe) return;

    uint64_t hash = probe_hashes[tid];
    uint32_t bucket = (uint32_t)(hash % hash_table_size);

    uint32_t current = hash_table[bucket];
    while (current != 0xFFFFFFFF) {
        if (build_hashes[current] == hash) {
            has_match[tid] = 1;
            return;
        }
        current = next_ptrs[current];
    }
    has_match[tid] = 0;
}

// Anti-join: mark probe rows that have no match
extern "C" __global__ void hash_join_anti(
    const uint64_t* __restrict__ probe_hashes,
    uint32_t num_probe,
    const uint32_t* __restrict__ hash_table,
    const uint64_t* __restrict__ build_hashes,
    const uint32_t* __restrict__ next_ptrs,
    uint32_t hash_table_size,
    uint8_t* __restrict__ no_match
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_probe) return;

    uint64_t hash = probe_hashes[tid];
    uint32_t bucket = (uint32_t)(hash % hash_table_size);

    uint32_t current = hash_table[bucket];
    while (current != 0xFFFFFFFF) {
        if (build_hashes[current] == hash) {
            no_match[tid] = 0;
            return;
        }
        current = next_ptrs[current];
    }
    no_match[tid] = 1;
}
```

**Step 2: Compile and verify**

Run: `cargo build -p xlog-cuda 2>&1 | head -20`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add kernels/join.cu
git commit -m "feat(xlog-cuda): add multi-column join and semi/anti join kernels"
```

---

## Task 8: Multi-Column Join Rust Wrapper

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Create: `crates/xlog-cuda/tests/join_v2_tests.rs`

**Step 1: Write failing tests**

```rust
// crates/xlog-cuda/tests/join_v2_tests.rs
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager, MemoryBudget};
use xlog_ir::{Schema, JoinType};
use std::sync::Arc;

fn setup_provider() -> Option<CudaKernelProvider> {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return None;
    }
    let device = Arc::new(CudaDevice::new(0).unwrap());
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::default()));
    Some(CudaKernelProvider::new(device, memory).unwrap())
}

#[test]
fn test_hash_join_multi_column() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Left: (a, b, payload)  Right: (x, y, value)
    // Join on (a, b) = (x, y)
    let left_a = vec![1u32, 1, 2];
    let left_b = vec![10u32, 20, 10];
    let left_p = vec![100u32, 200, 300];

    let right_x = vec![1u32, 2, 1];
    let right_y = vec![10u32, 10, 20];
    let right_v = vec![1000u32, 2000, 3000];

    let left_schema = Schema::new(vec![
        xlog_ir::ScalarType::U32,
        xlog_ir::ScalarType::U32,
        xlog_ir::ScalarType::U32,
    ]);
    let right_schema = Schema::new(vec![
        xlog_ir::ScalarType::U32,
        xlog_ir::ScalarType::U32,
        xlog_ir::ScalarType::U32,
    ]);

    let left = provider.create_buffer_from_u32_columns(
        &[&left_a, &left_b, &left_p], left_schema
    ).unwrap();
    let right = provider.create_buffer_from_u32_columns(
        &[&right_x, &right_y, &right_v], right_schema
    ).unwrap();

    let result = provider.hash_join_v2(&left, &right, &[0, 1], &[0, 1]).unwrap();

    // Expected matches:
    // (1, 10, 100) joins (1, 10, 1000) -> (1, 10, 100, 1000)
    // (1, 20, 200) joins (1, 20, 3000) -> (1, 20, 200, 3000)
    // (2, 10, 300) joins (2, 10, 2000) -> (2, 10, 300, 2000)
    assert_eq!(result.num_rows, 3);
}

#[test]
fn test_semi_join() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Left: [1, 2, 3, 4]  Right: [2, 4]
    // Semi-join: left rows that exist in right -> [2, 4]
    let left = vec![1u32, 2, 3, 4];
    let right = vec![2u32, 4];

    let left_buf = provider.create_buffer_from_u32(&left,
        Schema::new(vec![xlog_ir::ScalarType::U32])).unwrap();
    let right_buf = provider.create_buffer_from_u32(&right,
        Schema::new(vec![xlog_ir::ScalarType::U32])).unwrap();

    let result = provider.hash_join_typed(&left_buf, &right_buf, &[0], &[0], JoinType::Semi).unwrap();

    let vals = provider.download_column_u32(&result, 0).unwrap();
    assert_eq!(vals, vec![2, 4]);
}

#[test]
fn test_anti_join() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Left: [1, 2, 3, 4]  Right: [2, 4]
    // Anti-join: left rows that DON'T exist in right -> [1, 3]
    let left = vec![1u32, 2, 3, 4];
    let right = vec![2u32, 4];

    let left_buf = provider.create_buffer_from_u32(&left,
        Schema::new(vec![xlog_ir::ScalarType::U32])).unwrap();
    let right_buf = provider.create_buffer_from_u32(&right,
        Schema::new(vec![xlog_ir::ScalarType::U32])).unwrap();

    let result = provider.hash_join_typed(&left_buf, &right_buf, &[0], &[0], JoinType::Anti).unwrap();

    let vals = provider.download_column_u32(&result, 0).unwrap();
    assert_eq!(vals, vec![1, 3]);
}
```

**Step 2: Run tests to verify failure**

Run: `cargo test -p xlog-cuda --test join_v2_tests 2>&1 | tail -20`
Expected: FAIL

**Step 3: Implement hash_join_v2 and hash_join_typed**

```rust
impl CudaKernelProvider {
    /// Multi-column hash join using composite hashing
    pub fn hash_join_v2(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
    ) -> Result<CudaBuffer> {
        self.hash_join_typed(left, right, left_keys, right_keys, JoinType::Inner)
    }

    /// Hash join with explicit join type
    pub fn hash_join_typed(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        join_type: JoinType,
    ) -> Result<CudaBuffer> {
        match join_type {
            JoinType::Inner => self.hash_join_inner_v2(left, right, left_keys, right_keys),
            JoinType::Semi => self.hash_join_semi_impl(left, right, left_keys, right_keys),
            JoinType::Anti => self.hash_join_anti_impl(left, right, left_keys, right_keys),
            JoinType::LeftOuter => self.hash_join_left_outer_impl(left, right, left_keys, right_keys),
        }
    }

    fn hash_join_semi_impl(...) -> Result<CudaBuffer> {
        // 1. Compute hashes for both sides
        // 2. Build hash table from right
        // 3. Probe with left, get has_match mask
        // 4. filter_by_mask(left, has_match)
    }

    fn hash_join_anti_impl(...) -> Result<CudaBuffer> {
        // 1. Compute hashes for both sides
        // 2. Build hash table from right
        // 3. Probe with left, get no_match mask
        // 4. filter_by_mask(left, no_match)
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p xlog-cuda --test join_v2_tests 2>&1 | tail -20`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/join_v2_tests.rs
git commit -m "feat(xlog-cuda): add multi-column and typed hash join wrappers"
```

---

## Task 9: GPU Set Operations (Union/Diff)

**Files:**
- Create: `kernels/set_ops.cu`
- Modify: `crates/xlog-cuda/build.rs`
- Modify: `crates/xlog-cuda/src/provider/mod.rs`

**Step 1: Write set_ops.cu**

```cuda
// kernels/set_ops.cu
#include <cstdint>

// Concatenate two u32 arrays (device-to-device)
extern "C" __global__ void concat_u32(
    const uint32_t* __restrict__ a,
    uint32_t a_len,
    const uint32_t* __restrict__ b,
    uint32_t b_len,
    uint32_t* __restrict__ output
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = a_len + b_len;
    if (gid >= total) return;

    if (gid < a_len) {
        output[gid] = a[gid];
    } else {
        output[gid] = b[gid - a_len];
    }
}

// Mark elements in sorted array A that are NOT in sorted array B
extern "C" __global__ void sorted_diff_mark(
    const uint32_t* __restrict__ a,
    uint32_t a_len,
    const uint32_t* __restrict__ b,
    uint32_t b_len,
    uint8_t* __restrict__ in_diff  // 1 if a[i] not in b
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= a_len) return;

    uint32_t val = a[gid];

    // Binary search in b
    uint32_t lo = 0, hi = b_len;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (b[mid] < val) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // Check if found
    in_diff[gid] = (lo < b_len && b[lo] == val) ? 0 : 1;
}
```

**Step 2: Update build.rs and compile**

Add `"set_ops"` to kernel list.

**Step 3: Write tests**

```rust
// Add to existing tests or new file
#[test]
fn test_union_gpu() {
    let Some(provider) = setup_provider() else { return; };

    let a = vec![1u32, 2, 3];
    let b = vec![3u32, 4, 5];

    let schema = Schema::new(vec![xlog_ir::ScalarType::U32]);
    let buf_a = provider.create_buffer_from_u32(&a, schema.clone()).unwrap();
    let buf_b = provider.create_buffer_from_u32(&b, schema).unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let vals = provider.download_column_u32(&result, 0).unwrap();

    // Union should deduplicate: [1, 2, 3, 4, 5]
    assert_eq!(vals, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_diff_gpu() {
    let Some(provider) = setup_provider() else { return; };

    let a = vec![1u32, 2, 3, 4];
    let b = vec![2u32, 4];

    let schema = Schema::new(vec![xlog_ir::ScalarType::U32]);
    let buf_a = provider.create_buffer_from_u32(&a, schema.clone()).unwrap();
    let buf_b = provider.create_buffer_from_u32(&b, schema).unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let vals = provider.download_column_u32(&result, 0).unwrap();

    // Diff: a - b = [1, 3]
    assert_eq!(vals, vec![1, 3]);
}
```

**Step 4: Implement union_gpu and diff_gpu**

```rust
impl CudaKernelProvider {
    /// GPU-native union (no host roundtrip)
    pub fn union_gpu(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        // 1. Concatenate on GPU
        // 2. Sort
        // 3. Dedup
    }

    /// GPU-native set difference (no host roundtrip)
    pub fn diff_gpu(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        // 1. Sort both
        // 2. Mark elements in a not in b
        // 3. filter_by_mask
    }
}
```

**Step 5: Run tests and commit**

```bash
git add kernels/set_ops.cu crates/xlog-cuda/build.rs crates/xlog-cuda/src/provider/mod.rs
git commit -m "feat(xlog-cuda): add GPU-native union and diff operations"
```

---

## Task 10: Multi-Aggregation Kernel

**Files:**
- Modify: `kernels/groupby.cu`
- Modify: `crates/xlog-cuda/src/provider/mod.rs`

**Step 1: Enhance groupby.cu with multi-agg support**

Add to `kernels/groupby.cu`:

```cuda
// Detect group boundaries in sorted data
extern "C" __global__ void detect_boundaries(
    const uint32_t* __restrict__ keys,
    uint32_t num_rows,
    uint8_t* __restrict__ boundaries  // 1 if keys[i] != keys[i-1]
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    if (gid == 0) {
        boundaries[gid] = 1;  // First element is always a boundary
    } else {
        boundaries[gid] = (keys[gid] != keys[gid - 1]) ? 1 : 0;
    }
}

// Compute group IDs from boundaries (prefix sum of boundaries)
// This is done by prefix_sum_mask

// Segmented sum reduction
extern "C" __global__ void groupby_sum(
    const uint32_t* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    uint32_t num_groups,
    uint64_t* __restrict__ sums
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint32_t group = group_ids[gid];
    atomicAdd((unsigned long long*)&sums[group], (unsigned long long)values[gid]);
}

// Segmented count
extern "C" __global__ void groupby_count(
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    uint32_t num_groups,
    uint32_t* __restrict__ counts
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint32_t group = group_ids[gid];
    atomicAdd(&counts[group], 1);
}

// Segmented min
extern "C" __global__ void groupby_min(
    const uint32_t* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    uint32_t num_groups,
    uint32_t* __restrict__ mins
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint32_t group = group_ids[gid];
    atomicMin(&mins[group], values[gid]);
}

// Segmented max
extern "C" __global__ void groupby_max(
    const uint32_t* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    uint32_t num_groups,
    uint32_t* __restrict__ maxs
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint32_t group = group_ids[gid];
    atomicMax(&maxs[group], values[gid]);
}

// LogSumExp for numerical stability
extern "C" __global__ void groupby_logsumexp_pass1(
    const double* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    uint32_t num_groups,
    double* __restrict__ max_vals
) {
    // First pass: find max per group for numerical stability
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint32_t group = group_ids[gid];
    // Atomic max for doubles (use integer atomics on bit representation)
    // Simplified: use shared memory reduction
    atomicMax((unsigned long long*)&max_vals[group],
              __double_as_longlong(values[gid]));
}

extern "C" __global__ void groupby_logsumexp_pass2(
    const double* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    const double* __restrict__ max_vals,
    uint32_t num_rows,
    uint32_t num_groups,
    double* __restrict__ sums
) {
    // Second pass: sum exp(x - max)
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint32_t group = group_ids[gid];
    double shifted = values[gid] - max_vals[group];
    // Atomic add for doubles
    atomicAdd(&sums[group], exp(shifted));
}

// Final: result = max + log(sum)
extern "C" __global__ void groupby_logsumexp_final(
    const double* __restrict__ max_vals,
    const double* __restrict__ sums,
    uint32_t num_groups,
    double* __restrict__ output
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_groups) return;

    output[gid] = max_vals[gid] + log(sums[gid]);
}
```

**Step 2: Compile**

Run: `cargo build -p xlog-cuda`

**Step 3: Write tests for multi-agg**

```rust
#[test]
fn test_groupby_multi_agg() {
    let Some(provider) = setup_provider() else { return; };

    // key: [1, 1, 2, 2, 2]
    // val: [10, 20, 5, 15, 25]
    // Expected: group 1: sum=30, count=2, min=10, max=20
    //           group 2: sum=45, count=3, min=5, max=25

    let keys = vec![1u32, 1, 2, 2, 2];
    let vals = vec![10u32, 20, 5, 15, 25];

    let schema = Schema::new(vec![xlog_ir::ScalarType::U32, xlog_ir::ScalarType::U32]);
    let buffer = provider.create_buffer_from_u32_columns(&[&keys, &vals], schema).unwrap();

    let result = provider.groupby_multi_agg(
        &buffer,
        &[0],  // key columns
        &[(1, AggOp::Sum), (1, AggOp::Count), (1, AggOp::Min), (1, AggOp::Max)],
    ).unwrap();

    // Result should have: key, sum, count, min, max
    assert_eq!(result.num_rows, 2);
    // Verify values...
}
```

**Step 4: Implement groupby_multi_agg**

**Step 5: Commit**

```bash
git add kernels/groupby.cu crates/xlog-cuda/src/provider/mod.rs
git commit -m "feat(xlog-cuda): add multi-aggregation groupby with LogSumExp"
```

---

## Task 11: Update Executor to Use New Kernels

**Files:**
- Modify: `crates/xlog-runtime/src/executor.rs`

**Step 1: Update execute_join to use hash_join_typed**

Replace the existing join handling with:

```rust
fn execute_join(&mut self, join: &Join) -> Result<CudaBuffer> {
    let left = self.execute_node(&join.left)?;
    let right = self.execute_node(&join.right)?;

    self.provider.hash_join_typed(
        &left,
        &right,
        &join.left_keys,
        &join.right_keys,
        join.join_type,
    )
}
```

**Step 2: Update execute_filter to use GPU filter**

**Step 3: Update union/diff to use GPU versions**

**Step 4: Update groupby to use multi-agg**

**Step 5: Run all existing tests**

Run: `cargo test -p xlog-runtime 2>&1 | tail -30`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add crates/xlog-runtime/src/executor.rs
git commit -m "refactor(xlog-runtime): update executor to use production-grade kernels"
```

---

## Task 12: Enable Previously Ignored E2E Tests

**Files:**
- Modify: `crates/xlog-runtime/tests/e2e_tests.rs` (or equivalent)

**Step 1: Remove #[ignore] from E2E tests**

Find and remove `#[ignore]` attributes from the 3 ignored tests.

**Step 2: Run E2E tests**

Run: `cargo test -p xlog-runtime --test e2e 2>&1`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add crates/xlog-runtime/tests/
git commit -m "test(xlog-runtime): enable all E2E tests with production kernels"
```

---

## Task 13: Add Comprehensive Type Coverage Tests

**Files:**
- Create: `crates/xlog-cuda/tests/type_coverage_tests.rs`

**Step 1: Write tests for all scalar types**

```rust
#[test]
fn test_filter_i64() { ... }

#[test]
fn test_filter_f64() { ... }

#[test]
fn test_join_i64_keys() { ... }

#[test]
fn test_groupby_f64_values() { ... }

#[test]
fn test_logsumexp_aggregation() { ... }
```

**Step 2: Run type coverage tests**

Run: `cargo test -p xlog-cuda --test type_coverage_tests`
Expected: All PASS

**Step 3: Commit**

```bash
git add crates/xlog-cuda/tests/type_coverage_tests.rs
git commit -m "test(xlog-cuda): add comprehensive type coverage tests"
```

---

## Final Verification

**Run full test suite:**

```bash
cargo test --workspace 2>&1 | tail -50
```

Expected: All tests pass, no ignored tests remaining for core functionality.

**Verify no host roundtrips in hot paths:**

Review `provider.rs` to ensure union, diff, and filter don't copy to host.
