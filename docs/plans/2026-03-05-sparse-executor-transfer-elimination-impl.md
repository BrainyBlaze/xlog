# Sparse Executor + Transfer Elimination Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all three D2H transfers from `compute_ilp_loss_grad_gpu` so the loss/gradient path makes zero device-to-host copies, verified by `host_transfer_stats()`.

**Architecture:** Replace host-side COO assembly, CSR construction, and loss reduction with device-side equivalents. Use existing `scan_u8_mask_device`, `compact_u32_by_mask`, `exclusive_scan_u32_inplace`, and `count_mask` primitives plus four new CUDA kernels (`ilp_coo_fill_from_mask`, `ilp_csr_histogram`, `ilp_reduce_sum_f32/f64`). Over-allocate COO at `num_candidates × num_query` with a memory cap + chunking fallback.

**Tech Stack:** Rust (PyO3, cudarc), CUDA C, Python (PyTorch/DLPack)

**Design doc:** `docs/plans/2026-03-05-transfer-elimination-design.md`

---

### Task 1: `count_mask_device` Rust Wrapper

Add a Rust wrapper on `CudaKernelProvider` for the existing `count_mask` CUDA kernel. This kernel counts nonzero entries in a `u8` mask, writing the result to a device `u32`. The wrapper allocates a 1-element device buffer, zeros it, launches the kernel, and returns the device buffer (no host download).

**Files:**
- Modify: `crates/xlog-cuda/src/provider.rs` (near line 4404, after `scan_u8_mask_device`)
- Test: `crates/xlog-cuda-tests/src/harness/provider.rs` or a new test in the CUDA cert suite

**Step 1: Write the failing test**

Add to the CUDA cert suite (or a new test file `crates/xlog-cuda-tests/tests/count_mask_device.rs`):

```rust
#[test]
fn test_count_mask_device_basic() {
    let provider = test_provider();  // from harness
    let mask_host: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1];
    let d_mask = provider.device().inner()
        .htod_sync_copy(&mask_host).unwrap();
    let d_count = provider.count_mask_device(
        &d_mask.into(), 7
    ).unwrap();
    let mut host_count = vec![0u32; 1];
    provider.device().inner()
        .dtoh_sync_copy_into(&d_count, &mut host_count).unwrap();
    assert_eq!(host_count[0], 4);
}

#[test]
fn test_count_mask_device_all_zeros() {
    let provider = test_provider();
    let mask_host: Vec<u8> = vec![0, 0, 0];
    let d_mask = provider.device().inner()
        .htod_sync_copy(&mask_host).unwrap();
    let d_count = provider.count_mask_device(
        &d_mask.into(), 3
    ).unwrap();
    let mut host_count = vec![0u32; 1];
    provider.device().inner()
        .dtoh_sync_copy_into(&d_count, &mut host_count).unwrap();
    assert_eq!(host_count[0], 0);
}

#[test]
fn test_count_mask_device_empty() {
    let provider = test_provider();
    let d_count = provider.count_mask_device_from_len(0).unwrap();
    // should return a buffer with count = 0
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda-tests count_mask_device --release`
Expected: FAIL — `count_mask_device` method does not exist

**Step 3: Implement `count_mask_device`**

In `crates/xlog-cuda/src/provider.rs`, add after `scan_u8_mask_device`:

```rust
/// Count nonzero entries in a device u8 mask.
/// Returns a 1-element device buffer containing the count.
/// The result stays on device — no D2H transfer.
pub fn count_mask_device(
    &self,
    mask: &TrackedCudaSlice<u8>,
    n: u32,
) -> Result<TrackedCudaSlice<u32>> {
    let mut d_count = self.memory.alloc::<u32>(1)?;
    if n == 0 {
        self.device.inner()
            .htod_sync_copy_into(&[0u32], &mut d_count)
            .map_err(|e| XlogError::Kernel(format!("zero count: {}", e)))?;
        return Ok(d_count);
    }
    // Zero the count
    self.device.inner()
        .htod_sync_copy_into(&[0u32], &mut d_count)
        .map_err(|e| XlogError::Kernel(format!("zero count: {}", e)))?;

    let func = self.device.inner()
        .get_func(SCAN_MODULE, scan_kernels::COUNT_MASK)
        .ok_or_else(|| XlogError::Kernel("count_mask kernel not found".to_string()))?;

    let block_size = 256u32;
    let grid_size = (n + block_size - 1) / block_size;
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (mask, n, &mut d_count),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("count_mask failed: {}", e)))?;
    self.device.synchronize()?;
    Ok(d_count)
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda-tests count_mask_device --release`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider.rs crates/xlog-cuda-tests/
git commit -m "feat(cuda): count_mask_device Rust wrapper for device-side mask count"
```

---

### Task 2: `ilp_coo_fill_from_mask` CUDA Kernel + Rust Wrapper

Write a CUDA kernel that fills COO arrays from a device-side mask + prefix-sum, reading its write offset from a device offset array. This eliminates the mask D2H (Transfer #1) by keeping COO assembly fully on device.

**Kernel signature:**

```cuda
extern "C" __global__ void ilp_coo_fill_from_mask(
    const uint8_t* mask,            // [num_query]
    const uint32_t* prefix_sum,     // [num_query] exclusive prefix-sum of mask
    const uint32_t* fact_indices,   // [num_query] global fact indices for each query row
    uint32_t cidx,                  // candidate index to broadcast
    uint32_t num_query,             // number of query rows
    const uint32_t* d_offsets,      // [num_candidates] write offset from device prefix-sum
    uint32_t* coo_fact,             // output COO fact indices
    uint32_t* coo_cand              // output COO candidate indices
);
```

Thread `tid` checks `mask[tid]`; if set, writes `coo_fact[d_offsets[cidx] + prefix_sum[tid]] = fact_indices[tid]` and `coo_cand[...] = cidx`.

**Files:**
- Modify: `kernels/ilp_credit.cu` (add kernel after `ilp_coo_fill`)
- Modify: `crates/xlog-cuda/src/provider.rs` (add Rust launcher)
- Test: New test in CUDA cert suite or `crates/xlog-cuda-tests/`

**Step 1: Write the failing test**

```rust
#[test]
fn test_ilp_coo_fill_from_mask_basic() {
    let provider = test_provider();
    // mask: [1, 0, 1, 1]  → 3 set bits
    // prefix_sum: [0, 1, 1, 2] (exclusive scan of mask)
    // fact_indices: [10, 20, 30, 40]
    // cidx = 5, d_offsets[5] = 7 (offset into COO arrays)
    //
    // Expected writes at positions 7, 8, 9:
    //   coo_fact[7]=10, coo_fact[8]=30, coo_fact[9]=40
    //   coo_cand[7..10] = [5, 5, 5]
    let mask: Vec<u8> = vec![1, 0, 1, 1];
    let prefix: Vec<u32> = vec![0, 1, 1, 2];
    let facts: Vec<u32> = vec![10, 20, 30, 40];
    let cidx = 5u32;
    // d_offsets has at least cidx+1 entries; d_offsets[5] = 7
    let mut offsets = vec![0u32; 6];
    offsets[5] = 7;

    // Upload to device
    let d_mask = upload(&provider, &mask);
    let d_prefix = upload(&provider, &prefix);
    let d_facts = upload(&provider, &facts);
    let d_offsets = upload(&provider, &offsets);

    // Allocate COO output (large enough)
    let mut d_coo_fact = provider.memory().alloc::<u32>(20).unwrap();
    let mut d_coo_cand = provider.memory().alloc::<u32>(20).unwrap();

    provider.ilp_coo_fill_from_mask_launch(
        &d_mask, &d_prefix, &d_facts,
        cidx, 4, &d_offsets,
        &mut d_coo_fact, &mut d_coo_cand,
    ).unwrap();

    let coo_f = download(&provider, &d_coo_fact);
    let coo_c = download(&provider, &d_coo_cand);
    assert_eq!(coo_f[7], 10);
    assert_eq!(coo_f[8], 30);
    assert_eq!(coo_f[9], 40);
    assert_eq!(coo_c[7], 5);
    assert_eq!(coo_c[8], 5);
    assert_eq!(coo_c[9], 5);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda-tests ilp_coo_fill_from_mask --release`
Expected: FAIL

**Step 3: Implement kernel + Rust wrapper**

Add to `kernels/ilp_credit.cu`:

```cuda
/// Fill COO arrays from device-side mask + prefix-sum.
/// Reads write offset from d_offsets[cidx] on device.
extern "C" __global__ void ilp_coo_fill_from_mask(
    const uint8_t* mask,
    const uint32_t* prefix_sum,
    const uint32_t* fact_indices,
    uint32_t cidx,
    uint32_t num_query,
    const uint32_t* d_offsets,
    uint32_t* coo_fact,
    uint32_t* coo_cand
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_query) return;
    if (mask[tid]) {
        uint32_t offset = d_offsets[cidx];
        uint32_t write_idx = offset + prefix_sum[tid];
        coo_fact[write_idx] = fact_indices[tid];
        coo_cand[write_idx] = cidx;
    }
}
```

Recompile PTX: `python scripts/compile_kernels.py` (or however PTX is generated — check `kernels/Makefile` or build script).

Add Rust launcher in `provider.rs`:

```rust
pub fn ilp_coo_fill_from_mask_launch(
    &self,
    mask: &TrackedCudaSlice<u8>,
    prefix_sum: &TrackedCudaSlice<u32>,
    fact_indices: &TrackedCudaSlice<u32>,
    cidx: u32,
    num_query: u32,
    d_offsets: &TrackedCudaSlice<u32>,
    coo_fact: &mut TrackedCudaSlice<u32>,
    coo_cand: &mut TrackedCudaSlice<u32>,
) -> Result<()> {
    if num_query == 0 { return Ok(()); }
    let func = self.device.inner()
        .get_func(ILP_CREDIT_MODULE, ilp_credit_kernels::ILP_COO_FILL_FROM_MASK)
        .ok_or_else(|| XlogError::Kernel("ilp_coo_fill_from_mask not found".to_string()))?;
    let block_size = 256u32;
    let grid_size = (num_query + block_size - 1) / block_size;
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (mask, prefix_sum, fact_indices, cidx, num_query, d_offsets, coo_fact, coo_cand),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("ilp_coo_fill_from_mask: {}", e)))?;
    self.device.synchronize()?;
    Ok(())
}
```

Add kernel name constant:

```rust
pub const ILP_COO_FILL_FROM_MASK: &str = "ilp_coo_fill_from_mask";
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda-tests ilp_coo_fill_from_mask --release`
Expected: PASS

**Step 5: Commit**

```bash
git add kernels/ilp_credit.cu kernels/ilp_credit.ptx crates/xlog-cuda/src/provider.rs crates/xlog-cuda-tests/
git commit -m "feat(cuda): ilp_coo_fill_from_mask kernel for device-side COO assembly"
```

---

### Task 3: `ilp_csr_histogram` CUDA Kernel + Rust Wrapper

Write a CUDA kernel that builds a histogram of fact indices from sorted COO, producing the count array needed for CSR `row_offsets` via prefix-sum. This eliminates Transfer #2 (sorted COO D2H for host CSR construction).

**Files:**
- Modify: `kernels/ilp_credit.cu`
- Modify: `crates/xlog-cuda/src/provider.rs`
- Test: CUDA cert suite

**Step 1: Write the failing test**

```rust
#[test]
fn test_ilp_csr_histogram_basic() {
    let provider = test_provider();
    // sorted_facts: [0, 0, 1, 3, 3, 3]  (nnz=6, num_facts=5)
    // expected histogram: [2, 1, 0, 3, 0]
    let sorted: Vec<u32> = vec![0, 0, 1, 3, 3, 3];
    let d_sorted = upload(&provider, &sorted);

    let d_hist = provider.ilp_csr_histogram_launch(&d_sorted, 6, 5).unwrap();
    let hist = download_u32(&provider, &d_hist, 5);
    assert_eq!(hist, vec![2, 1, 0, 3, 0]);
}

#[test]
fn test_ilp_csr_histogram_empty() {
    let provider = test_provider();
    let d_sorted = provider.memory().alloc::<u32>(0).unwrap();
    let d_hist = provider.ilp_csr_histogram_launch(&d_sorted, 0, 3).unwrap();
    let hist = download_u32(&provider, &d_hist, 3);
    assert_eq!(hist, vec![0, 0, 0]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda-tests ilp_csr_histogram --release`
Expected: FAIL

**Step 3: Implement kernel + wrapper**

CUDA kernel:

```cuda
/// Histogram of fact indices for CSR row_offsets construction.
/// Each thread atomically increments hist[sorted_facts[tid]].
/// Caller must zero hist before launch.
extern "C" __global__ void ilp_csr_histogram(
    const uint32_t* sorted_facts,  // [nnz]
    uint32_t nnz,
    uint32_t num_facts,            // bounds check
    uint32_t* hist                 // [num_facts] output, must be zeroed
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;
    uint32_t f = sorted_facts[tid];
    if (f < num_facts) {
        atomicAdd(&hist[f], 1);
    }
}
```

Rust wrapper allocates `d_hist[num_facts]`, zeros it via `htod_sync_copy_into`, launches kernel, returns `d_hist`. Then the caller does `exclusive_scan_u32_inplace` to convert histogram → `row_offsets`.

```rust
pub fn ilp_csr_histogram_launch(
    &self,
    sorted_facts: &TrackedCudaSlice<u32>,
    nnz: u32,
    num_facts: u32,
) -> Result<TrackedCudaSlice<u32>> {
    let mut d_hist = self.memory.alloc::<u32>(num_facts as usize)?;
    // Zero the histogram
    let zeros = vec![0u32; num_facts as usize];
    self.device.inner()
        .htod_sync_copy_into(&zeros, &mut d_hist)
        .map_err(|e| XlogError::Kernel(format!("zero hist: {}", e)))?;

    if nnz == 0 {
        return Ok(d_hist);
    }
    let func = self.device.inner()
        .get_func(ILP_CREDIT_MODULE, ilp_credit_kernels::ILP_CSR_HISTOGRAM)
        .ok_or_else(|| XlogError::Kernel("ilp_csr_histogram not found".to_string()))?;
    let block_size = 256u32;
    let grid_size = (nnz + block_size - 1) / block_size;
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (sorted_facts, nnz, num_facts, &mut d_hist),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("ilp_csr_histogram: {}", e)))?;
    self.device.synchronize()?;
    Ok(d_hist)
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda-tests ilp_csr_histogram --release`
Expected: PASS

**Step 5: Commit**

```bash
git add kernels/ilp_credit.cu kernels/ilp_credit.ptx crates/xlog-cuda/src/provider.rs crates/xlog-cuda-tests/
git commit -m "feat(cuda): ilp_csr_histogram kernel for device-side CSR construction"
```

---

### Task 4: `ilp_reduce_sum_f32` / `ilp_reduce_sum_f64` CUDA Kernels + Rust Wrappers

Write GPU reduction kernels that sum a float array to a single scalar on device. This eliminates Transfer #3 (loss_contrib D2H for host summation).

**Files:**
- Modify: `kernels/ilp_credit.cu`
- Modify: `crates/xlog-cuda/src/provider.rs`
- Test: CUDA cert suite

**Step 1: Write the failing test**

```rust
#[test]
fn test_ilp_reduce_sum_f32_basic() {
    let provider = test_provider();
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let d_data = upload_f32(&provider, &data);
    let d_result = provider.ilp_reduce_sum_f32_launch(&d_data, 5).unwrap();
    let mut result = vec![0.0f32; 1];
    provider.device().inner().dtoh_sync_copy_into(&d_result, &mut result).unwrap();
    assert!((result[0] - 15.0).abs() < 1e-6);
}

#[test]
fn test_ilp_reduce_sum_f64_basic() {
    let provider = test_provider();
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let d_data = upload_f64(&provider, &data);
    let d_result = provider.ilp_reduce_sum_f64_launch(&d_data, 5).unwrap();
    let mut result = vec![0.0f64; 1];
    provider.device().inner().dtoh_sync_copy_into(&d_result, &mut result).unwrap();
    assert!((result[0] - 15.0).abs() < 1e-12);
}

#[test]
fn test_ilp_reduce_sum_f32_large() {
    // Test with > 256 elements (multi-block)
    let provider = test_provider();
    let n = 1000;
    let data: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let expected = (n * (n + 1)) as f32 / 2.0;
    let d_data = upload_f32(&provider, &data);
    let d_result = provider.ilp_reduce_sum_f32_launch(&d_data, n as u32).unwrap();
    let mut result = vec![0.0f32; 1];
    provider.device().inner().dtoh_sync_copy_into(&d_result, &mut result).unwrap();
    assert!((result[0] - expected).abs() < 1.0);  // f32 accumulation error
}

#[test]
fn test_ilp_reduce_sum_empty() {
    let provider = test_provider();
    let d_data = provider.memory().alloc::<f32>(0).unwrap();
    let d_result = provider.ilp_reduce_sum_f32_launch(&d_data, 0).unwrap();
    let mut result = vec![0.0f32; 1];
    provider.device().inner().dtoh_sync_copy_into(&d_result, &mut result).unwrap();
    assert_eq!(result[0], 0.0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda-tests ilp_reduce_sum --release`
Expected: FAIL

**Step 3: Implement kernels + wrappers**

CUDA kernel (two-pass: block-level reduction → final block sum):

```cuda
#define REDUCE_BLOCK_SIZE 256

/// Block-level sum reduction (f32). Each block reduces its elements
/// to a single partial sum written to block_sums[blockIdx.x].
extern "C" __global__ void ilp_reduce_sum_f32(
    const float* input,
    uint32_t n,
    float* block_sums    // [num_blocks] or [1] if single block
) {
    __shared__ float sdata[REDUCE_BLOCK_SIZE];
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // Tree reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&block_sums[0], sdata[0]);
    }
}
```

Same pattern for `ilp_reduce_sum_f64` with `double` and `atomicAdd` for double (CUDA supports `atomicAdd` for doubles on sm_60+, which this project targets).

Rust wrapper:

```rust
pub fn ilp_reduce_sum_f32_launch(
    &self,
    input: &TrackedCudaSlice<f32>,
    n: u32,
) -> Result<TrackedCudaSlice<f32>> {
    let mut d_result = self.memory.alloc::<f32>(1)?;
    self.device.inner()
        .htod_sync_copy_into(&[0.0f32], &mut d_result)
        .map_err(|e| XlogError::Kernel(format!("zero result: {}", e)))?;

    if n == 0 {
        return Ok(d_result);
    }

    let func = self.device.inner()
        .get_func(ILP_CREDIT_MODULE, ilp_credit_kernels::ILP_REDUCE_SUM_F32)
        .ok_or_else(|| XlogError::Kernel("ilp_reduce_sum_f32 not found".to_string()))?;
    let block_size = 256u32;
    let grid_size = (n + block_size - 1) / block_size;
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (input, n, &mut d_result),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("ilp_reduce_sum_f32: {}", e)))?;
    self.device.synchronize()?;
    Ok(d_result)
}
```

Same for `ilp_reduce_sum_f64_launch`.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p xlog-cuda-tests ilp_reduce_sum --release`
Expected: PASS

**Step 5: Commit**

```bash
git add kernels/ilp_credit.cu kernels/ilp_credit.ptx crates/xlog-cuda/src/provider.rs crates/xlog-cuda-tests/
git commit -m "feat(cuda): ilp_reduce_sum_f32/f64 kernels for device-side loss reduction"
```

---

### Task 5: `export_loss_grad_device_f32/f64` Helpers

The current `export_loss_grad_f32` takes a host scalar and uploads it. After Task 4, the loss scalar is already on device. Add new export helpers that take a `TrackedCudaSlice<f32>` (1-element) directly.

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (near line 5172, after existing `export_loss_grad_f32`)

**Step 1: Write the helpers**

```rust
fn export_loss_grad_device_f32(
    &self,
    py: Python<'_>,
    d_loss: TrackedCudaSlice<f32>,   // 1-element device buffer
    d_grad: TrackedCudaSlice<f32>,   // [num_cands] device buffer
    num_cands: u32,
) -> PyResult<(PyObject, PyObject)> {
    let schema_f32 = Schema::new(vec![("col_0".to_string(), ScalarType::F32)]);

    let mut d_loss_nrows = self.provider.memory().alloc::<u32>(1)
        .map_err(|e| PyRuntimeError::new_err(format!("alloc: {}", e)))?;
    self.provider.device().inner()
        .htod_sync_copy_into(&[1u32], &mut d_loss_nrows)
        .map_err(|e| PyRuntimeError::new_err(format!("htod: {}", e)))?;

    let loss_buf = CudaBuffer::from_columns(
        vec![d_loss.into_bytes().into()],
        1, d_loss_nrows, schema_f32.clone(),
    );
    // ... rest same as export_loss_grad_f32 but without the htod_sync_copy_into for the loss value
}
```

Pattern the implementation on existing `export_loss_grad_f32` (line 5172-5244), removing the host-scalar upload and using the device buffer directly. Same for f64 variant.

**Step 2: Run existing GPU credit tests to verify nothing breaks**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py -v`
Expected: 13/13 PASS (helpers added but not yet called)

**Step 3: Commit**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "feat(pyxlog): export_loss_grad_device_f32/f64 for zero-D2H loss export"
```

---

### Task 6: Rewrite `compute_ilp_loss_grad_gpu` — Eliminate All Three D2H Transfers

Replace the three D2H sites in `compute_ilp_loss_grad_gpu_impl` (lib.rs:4111-4356) with device-side equivalents using Tasks 1-5.

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:4111-4356` (the `compute_ilp_loss_grad_gpu_impl` method)

**Phase C replacement (Transfer #1 — mask D2H):**

Current code (lib.rs:4158-4189): Per candidate, downloads mask to host, scans for nonzeros, builds host COO vecs.

New code:
1. Pre-compute per-candidate counts: For each candidate, call `count_mask_device(d_mask, num_query)` → write count to `d_counts[cidx]`.
2. Prefix-sum counts: `exclusive_scan_u32_inplace(d_counts, num_candidates)` → `d_offsets`.
3. Over-allocate COO: `upper_bound = num_candidates * num_query`. If `upper_bound * 8 > COO_MEMORY_CAP` (default 16 MB), fall back to chunked processing (see Task 7).
4. Upload `fact_indices` array (the global indices for query rows): one `htod_sync_copy_into` per relation group (H2D, allowed).
5. For each candidate: `scan_u8_mask_device(d_mask)` → `d_prefix`, then `ilp_coo_fill_from_mask_launch(d_mask, d_prefix, d_fact_indices, cidx, num_query, d_offsets, coo_fact, coo_cand)`.

**Phase D replacement (Transfer #2 — sorted COO D2H):**

Current code (lib.rs:4203-4253): Uploads COO, sorts, downloads sorted facts, builds CSR on host.

New code:
1. Radix sort COO on device (same as before, already on device).
2. `ilp_csr_histogram_launch(d_sorted_facts, nnz, num_facts)` → `d_hist`.
3. `exclusive_scan_u32_inplace(d_hist, num_facts + 1)` → `d_row_offsets`.

No D2H. The `row_offsets` stays on device.

**Phase E replacement (Transfer #3 — loss D2H):**

Current code (lib.rs:4299-4306/4334-4341): Downloads `loss_contrib`, sums on host.

New code:
1. `ilp_reduce_sum_f32_launch(d_loss_contrib, num_facts)` → `d_total_loss` (1-element device buffer).
2. Call `export_loss_grad_device_f32(py, d_total_loss, d_grad, num_cands)` instead of `export_loss_grad_f32(py, total_loss, d_grad, num_cands)`.

**Step 1: Rewrite Phase C**

Replace lib.rs:4112-4191 with device-side COO assembly. Key changes:
- Remove `coo_facts: Vec<u32>` and `coo_cands: Vec<u32>` host vectors.
- Add `d_coo_facts` and `d_coo_cands` device allocations at upper_bound.
- Add `d_counts[num_candidates]` and `d_offsets[num_candidates]` device buffers.
- Two-pass: count pass, then fill pass.

**Step 2: Rewrite Phase D**

Replace lib.rs:4203-4265 with device-side CSR. Remove host `sorted_facts` download, host `row_offsets` construction, and host upload.

**Step 3: Rewrite Phase E**

Replace lib.rs:4299-4355 with device-side reduction + device export. Remove `loss_host` download and host summation.

**Step 4: Verify no `dtoh_sync_copy_into` remains**

```bash
# Grep the function body for any remaining D2H
grep -n "dtoh_sync_copy_into" crates/pyxlog/src/lib.rs | grep -v "htod"
```

All `dtoh_sync_copy_into` calls in this function should be gone. `htod_sync_copy_into` calls remain (for uploading `is_positive`, query buffers, zeroing device allocations — all H2D, which is allowed).

**Step 5: Run existing GPU credit tests**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py -v`
Expected: 13/13 PASS (same correctness, zero-D2H path)

**Step 6: Commit**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "feat(ilp): eliminate all D2H transfers from compute_ilp_loss_grad_gpu"
```

---

### Task 7: Memory Cap + Chunking Fallback

Add the memory cap check from the design. If `upper_bound * 8 > COO_MEMORY_CAP`, process candidates in chunks.

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (in `compute_ilp_loss_grad_gpu_impl`)

**Step 1: Write a test that exercises the chunking path**

In `python/tests/test_ilp_credit_gpu.py`:

```python
def test_compute_ilp_loss_grad_gpu_memory_cap():
    """Force chunked COO assembly by setting a tiny memory cap."""
    prog = _compile_reach()
    prog.set_candidate_map([(0, 0, 1)])
    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.7], device=device, dtype=torch.float32)
    positives = [("reach", [1, 3])]
    negatives = [("reach", [1, 4])]

    # Set absurdly small cap to force chunking (1 byte)
    prog.set_coo_memory_cap(1)

    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
        positives, negatives, cand_probs
    )
    loss = torch.from_dlpack(loss_dl)
    grad = torch.from_dlpack(grad_dl)

    # Same result as non-chunked path
    assert loss.item() > 0.0
    assert torch.isfinite(loss).item()
    assert torch.all(torch.isfinite(grad)).item()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py::test_compute_ilp_loss_grad_gpu_memory_cap -v`
Expected: FAIL — `set_coo_memory_cap` doesn't exist

**Step 3: Implement chunking**

Add `coo_memory_cap: u64` field to `CompiledIlpProgram` (default 16 MB = `16 * 1024 * 1024`).

Add PyO3 setter `set_coo_memory_cap(bytes: u64)`.

In Phase C of `compute_ilp_loss_grad_gpu_impl`, check:
```rust
let upper_bound = (num_candidates as u64) * (max_num_query as u64);
let coo_bytes = upper_bound * 8; // 4 bytes each for fact + cand
if coo_bytes > self.coo_memory_cap {
    return self.compute_ilp_loss_grad_gpu_chunked(py, ...);
}
```

The chunked path splits candidates into groups that fit within the cap, processes each group's COO segment, concatenates on device, then proceeds with sort + CSR + forward/backward as normal. Each chunk may download one `u32` scalar (chunk nnz) for next-chunk offset computation — this is acceptable under the design's "chunking trades small D2H for bounded memory" provision.

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py -v`
Expected: 14/14 PASS (13 existing + 1 new)

**Step 5: Commit**

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_ilp_credit_gpu.py
git commit -m "feat(ilp): COO memory cap + chunked fallback for large candidate sets"
```

---

### Task 8: Strict D2H Acceptance Test

Write the acceptance gate test that asserts `host_transfer_stats()` shows zero D2H calls and bytes after `compute_ilp_loss_grad_gpu`.

**Files:**
- Modify: `python/tests/test_ilp_credit_gpu.py`

**Step 1: Write the acceptance test**

```python
def test_zero_dtoh_strict():
    """compute_ilp_loss_grad_gpu must cause zero D2H transfers (strict accounting)."""
    prog = _compile_reach()
    prog.set_candidate_map([(0, 0, 1)])

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.7], device=device, dtype=torch.float32)

    positives = [("reach", [1, 3])]
    negatives = [("reach", [1, 4])]

    # Reset byte-level transfer stats, then call GPU loss/grad path
    prog.reset_host_transfer_stats()
    prog.compute_ilp_loss_grad_gpu(positives, negatives, cand_probs)
    stats = prog.host_transfer_stats()

    assert stats['dtoh_calls'] == 0, (
        f"compute_ilp_loss_grad_gpu caused {stats['dtoh_calls']} D2H calls; expected 0"
    )
    assert stats['dtoh_bytes'] == 0, (
        f"compute_ilp_loss_grad_gpu transferred {stats['dtoh_bytes']} D2H bytes; expected 0"
    )
```

**Step 2: Run test to verify it fails (pre-Task-6)**

This test should fail before Task 6 rewrites the function. After Task 6, it should pass.

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py::test_zero_dtoh_strict -v`
Expected: PASS (if run after Task 6)

**Step 3: Also update the old `test_zero_dtoh_transfers` to use strict accounting**

```python
def test_zero_dtoh_transfers():
    """compute_ilp_loss_grad_gpu must not cause additional D2H column transfers."""
    prog = _compile_reach()
    prog.set_candidate_map([(0, 0, 1)])
    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.7], device=device, dtype=torch.float32)
    positives = [("reach", [1, 3])]
    negatives = [("reach", [1, 4])]

    # Column-level gate (coarse)
    prog.reset_d2h_transfer_count()
    # Byte-level gate (strict)
    prog.reset_host_transfer_stats()

    prog.compute_ilp_loss_grad_gpu(positives, negatives, cand_probs)

    # Coarse gate
    assert prog.d2h_transfer_count() == 0
    # Strict gate
    stats = prog.host_transfer_stats()
    assert stats['dtoh_calls'] == 0
    assert stats['dtoh_bytes'] == 0
```

**Step 4: Run full GPU credit test suite**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add python/tests/test_ilp_credit_gpu.py
git commit -m "test(ilp): strict D2H acceptance gate via host_transfer_stats"
```

---

### Task 9: Full Regression + Report Update

Run the complete test suite to verify no regressions, then update the performance report.

**Step 1: Run Rust workspace tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release`
Expected: PASS

**Step 2: Run CUDA cert suite**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release`
Expected: 206+ PASS

**Step 3: Run Python test suite**

Run: `.venv/bin/python -m pytest python/tests/ -v --timeout=120`
Expected: 119+ PASS (new tests included)

**Step 4: Run ILP 20/20 reliability**

Run: `.venv/bin/python -m pytest python/tests/test_ga_reliability.py::test_ga_reliability_20 -v --timeout=900`
Expected: 20/20 PASS

**Step 5: Update performance report**

Update `docs/reports/2026-03-05-gpu-credit-loss-report.md`:
- Change Go/No-Go to "GO for Phase 2 (true zero-D2H)"
- Update D2H Transfer Accounting section: "Zero D2H transfers confirmed via `host_transfer_stats()`: dtoh_calls=0, dtoh_bytes=0"
- Remove the "Not yet GO" caveat
- Move follow-up tasks to "Completed" section

**Step 6: Commit report + final state**

```bash
git add docs/reports/2026-03-05-gpu-credit-loss-report.md
git commit -m "docs: update GPU credit report — Phase 2 zero-D2H achieved"
```

---

## Execution Order Summary

| Task | What | New Kernels | Eliminates |
|------|------|-------------|------------|
| 1 | `count_mask_device` wrapper | — (existing kernel) | Prerequisite for #2 |
| 2 | `ilp_coo_fill_from_mask` | 1 CUDA kernel | Transfer #1 (mask D2H) |
| 3 | `ilp_csr_histogram` | 1 CUDA kernel | Transfer #2 (COO facts D2H) |
| 4 | `ilp_reduce_sum_f32/f64` | 2 CUDA kernels | Transfer #3 (loss D2H) |
| 5 | `export_loss_grad_device_*` helpers | — | Prerequisite for #6 |
| 6 | Rewrite `compute_ilp_loss_grad_gpu` | — | All three transfers |
| 7 | Memory cap + chunking | — | Memory safety |
| 8 | Strict acceptance test | — | Accounting gate |
| 9 | Full regression + report | — | Validation |
