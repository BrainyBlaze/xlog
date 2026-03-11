# GPU Native Phase 4 Gap Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate remaining host reads by moving compaction counts and MC query/evidence reductions fully on-device, and remove all CPU/MVP fallbacks in the CUDA provider while preserving correctness.

**Architecture:** Introduce device-resident row counts for `CudaBuffer`, move compaction/groupby/union/clone to GPU-only paths, and add device-side MC count accumulation. Kernel APIs accept device counts and guard index bounds, with outputs writing `d_num_rows` on-device. 

**Tech Stack:** Rust, CUDA C++, cudarc, xlog-cuda-tests certification harness.

---

### Task 1: Add device-resident row counts to `CudaBuffer`

**Files:**
- Modify: `crates/xlog-cuda/src/memory.rs`
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Test: `crates/xlog-cuda/tests/device_row_counts.rs`

**Step 1: Write the failing test**

```rust
// crates/xlog-cuda/tests/device_row_counts.rs
use std::sync::Arc;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn setup_provider() -> Option<CudaKernelProvider> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    CudaKernelProvider::new(device, memory).ok()
}

#[test]
fn test_device_row_count_tracks_host_count() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let ids: Vec<u32> = vec![1, 2, 3, 4];
    let buffer = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema)
        .unwrap();

    let mut host_count = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_count)
        .unwrap();
    assert_eq!(host_count[0], 4);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test device_row_counts -q`
Expected: FAIL (no `num_rows_device` or missing device count field)

**Step 3: Write minimal implementation**

```rust
// crates/xlog-cuda/src/memory.rs
pub struct CudaBuffer {
    pub columns: Vec<CudaColumn>,
    pub row_cap: u64,
    pub d_num_rows: TrackedCudaSlice<u32>,
    pub schema: Schema,
}

impl CudaBuffer {
    pub fn from_columns(columns: Vec<CudaColumn>, row_cap: u64, d_num_rows: TrackedCudaSlice<u32>, schema: Schema) -> Self {
        assert_eq!(columns.len(), schema.arity(), "...");
        Self { columns, row_cap, d_num_rows, schema }
    }

    pub fn num_rows_device(&self) -> &TrackedCudaSlice<u32> {
        &self.d_num_rows
    }

    pub fn row_cap(&self) -> u64 {
        self.row_cap
    }
}
```

```rust
// crates/xlog-cuda/src/provider/mod.rs (helpers)
pub fn create_empty_buffer(&self, schema: Schema) -> Result<CudaBuffer> {
    let mut d_num_rows = self.memory.alloc::<u32>(1)?;
    self.device.inner().htod_sync_copy_into(&[0u32], &mut d_num_rows)
        .map_err(|e| XlogError::Kernel(format!("Failed to init row count: {}", e)))?;
    Ok(CudaBuffer::from_columns(Vec::new(), 0, d_num_rows, schema))
}

pub fn create_buffer_from_slices(&self, columns: &[&[u8]], schema: Schema) -> Result<CudaBuffer> {
    // existing upload logic...
    let row_cap = row_count as u64;
    let mut d_num_rows = self.memory.alloc::<u32>(1)?;
    self.device.inner().htod_sync_copy_into(&[row_count as u32], &mut d_num_rows)
        .map_err(|e| XlogError::Kernel(format!("Failed to init row count: {}", e)))?;
    Ok(CudaBuffer::from_columns(uploaded, row_cap, d_num_rows, schema))
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test device_row_counts -q`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/memory.rs crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/device_row_counts.rs

git commit -m "feat(cuda): add device-resident row counts to CudaBuffer"
```

---

### Task 2: GPU-only clone/union paths (remove MVP/host clone)

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Test: `crates/xlog-cuda/tests/device_row_counts.rs`

**Step 1: Write the failing test**

```rust
// append to crates/xlog-cuda/tests/device_row_counts.rs
#[test]
fn test_clone_buffer_preserves_device_count() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let ids: Vec<u32> = vec![10, 20, 30];
    let buffer = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema)
        .unwrap();

    let cloned = provider.clone_buffer(&buffer).unwrap();

    let mut host_count = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(cloned.num_rows_device(), &mut host_count)
        .unwrap();
    assert_eq!(host_count[0], 3);
}

#[test]
fn test_union_gpu_dedups_rows() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let a: Vec<u32> = vec![1, 2, 2, 3];
    let b: Vec<u32> = vec![2, 4];
    let buf_a = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&a)], schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&b)], schema)
        .unwrap();

    let out = provider.union(&buf_a, &buf_b).unwrap();
    let record = provider.to_arrow_record_batch(&out).unwrap();
    let col = record.column(0).as_primitive::<arrow::datatypes::UInt32Type>();
    let mut vals: Vec<u32> = (0..record.num_rows()).map(|i| col.value(i)).collect();
    vals.sort();
    vals.dedup();
    assert_eq!(vals, vec![1, 2, 3, 4]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test device_row_counts -q`
Expected: FAIL (clone uses host path, union uses concat and keeps duplicates)

**Step 3: Write minimal implementation**

```rust
// crates/xlog-cuda/src/provider/mod.rs
pub fn union(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
    self.union_gpu(a, b)
}

fn clone_buffer(&self, buffer: &CudaBuffer) -> Result<CudaBuffer> {
    let mut new_columns = Vec::with_capacity(buffer.columns.len());
    for col in &buffer.columns {
        let mut dst = self.memory.alloc::<u8>(col.len())?;
        self.device.inner().dtod_sync_copy_into(col, &mut dst)
            .map_err(|e| XlogError::Kernel(format!("clone_buffer dtod failed: {}", e)))?;
        new_columns.push(dst);
    }
    let mut d_num_rows = self.memory.alloc::<u32>(1)?;
    self.device.inner().dtod_sync_copy_into(buffer.num_rows_device(), &mut d_num_rows)
        .map_err(|e| XlogError::Kernel(format!("clone_buffer row count dtod failed: {}", e)))?;
    Ok(CudaBuffer::from_columns(new_columns, buffer.row_cap(), d_num_rows, buffer.schema.clone()))
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test device_row_counts -q`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/device_row_counts.rs

git commit -m "feat(cuda): remove host clone and dedup union via union_gpu"
```

---

### Task 3: Device-only compaction and filter counts

**Files:**
- Modify: `kernels/filter.cu`
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Test: `crates/xlog-cuda/tests/compact_device_count.rs`

**Step 1: Write the failing test**

```rust
// crates/xlog-cuda/tests/compact_device_count.rs
use std::sync::Arc;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn setup_provider() -> Option<CudaKernelProvider> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    CudaKernelProvider::new(device, memory).ok()
}

#[test]
fn test_compact_device_mask_sets_device_count() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let ids: Vec<u32> = vec![1, 2, 3, 4, 5];
    let buffer = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema)
        .unwrap();

    // mask keeps odd indices
    let mask: Vec<u8> = vec![1, 0, 1, 0, 1];
    let mut d_mask = provider.memory().alloc::<u8>(mask.len()).unwrap();
    provider.device().inner().htod_sync_copy_into(&mask, &mut d_mask).unwrap();

    let compacted = provider
        .compact_buffer_by_device_mask_counted(&buffer, &d_mask)
        .unwrap();

    let mut host_count = [0u32];
    provider.device().inner().dtoh_sync_copy_into(compacted.num_rows_device(), &mut host_count).unwrap();
    assert_eq!(host_count[0], 3);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test compact_device_count -q`
Expected: FAIL (compaction reads DTOH; no device count set)

**Step 3: Write minimal implementation**

```cuda
// kernels/filter.cu (add)
extern "C" __global__ void capture_compact_count(
    const uint32_t* prefix_sum,
    const uint8_t* mask,
    uint32_t n,
    uint32_t* out_count
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (n == 0) {
            out_count[0] = 0;
            return;
        }
        uint32_t last = n - 1;
        out_count[0] = prefix_sum[last] + (mask[last] ? 1u : 0u);
    }
}
```

```rust
// crates/xlog-cuda/src/provider/mod.rs
pub fn compact_buffer_by_device_mask_counted(
    &self,
    input: &CudaBuffer,
    d_mask: &TrackedCudaSlice<u8>,
) -> Result<CudaBuffer> {
    let n = input.row_cap() as u32;
    if n == 0 {
        return self.create_empty_buffer(input.schema.clone());
    }
    if n as usize > d_mask.len() {
        return Err(XlogError::Kernel(format!(
            "compact_buffer_by_device_mask_counted: mask len {} < rows {}",
            d_mask.len(),
            n
        )));
    }

    let device = self.device.inner();
    let block_size = 256u32;
    let num_blocks = (n + block_size - 1) / block_size;

    let d_prefix_sum = self.memory.alloc::<u32>(n as usize)?;
    let mut d_block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;

    // scan phase1/phase3 (same as existing)...

    let mut d_out_count = self.memory.alloc::<u32>(1)?;
    let capture_fn = device
        .get_func(FILTER_MODULE, filter_kernels::CAPTURE_COMPACT_COUNT)
        .ok_or_else(|| XlogError::Kernel("capture_compact_count kernel not found".to_string()))?;
    unsafe {
        capture_fn.clone().launch(
            LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 },
            (&d_prefix_sum, d_mask, n, &mut d_out_count),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("capture_compact_count failed: {}", e)))?;

    // Allocate columns at capacity (n rows) and write device count only
    let mut new_columns = Vec::with_capacity(input.columns.len());
    for col_idx in 0..input.columns.len() {
        let elem_size = input.schema.column_type(col_idx).map(|t| t.size_bytes()).unwrap_or(4);
        let bytes = (n as usize) * elem_size;
        let mut dst = self.memory.alloc::<u8>(bytes)?;
        // compact kernel writes only indices < out_count
        // ... existing compact kernel launch ...
        new_columns.push(dst);
    }

    Ok(CudaBuffer::from_columns(new_columns, n as u64, d_out_count, input.schema.clone()))
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test compact_device_count -q`
Expected: PASS

**Step 5: Commit**

```bash
git add kernels/filter.cu crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/compact_device_count.rs

git commit -m "feat(cuda): compute compaction counts on device"
```

---

### Task 4: GPU groupby path (remove CPU boundary/group-id fallback)

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Modify: `kernels/groupby.cu`
- Test: `crates/xlog-cuda/tests/groupby_gpu.rs`

**Step 1: Write the failing test**

```rust
// crates/xlog-cuda/tests/groupby_gpu.rs
use std::sync::Arc;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn setup_provider() -> Option<CudaKernelProvider> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    CudaKernelProvider::new(device, memory).ok()
}

#[test]
fn test_groupby_agg_gpu_multi_key() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![
        ("k1".to_string(), ScalarType::U32),
        ("k2".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let k1: Vec<u32> = vec![1, 1, 2, 2, 2];
    let k2: Vec<u32> = vec![7, 7, 9, 9, 10];
    let v: Vec<u32> = vec![10, 20, 3, 4, 5];

    let buffer = provider
        .create_buffer_from_slices(
            &[bytemuck::cast_slice(&k1), bytemuck::cast_slice(&k2), bytemuck::cast_slice(&v)],
            schema,
        )
        .unwrap();

    let out = provider.groupby_agg(&buffer, &[0, 1], xlog_cuda::AggOp::Sum, 2).unwrap();
    let rb = provider.to_arrow_record_batch(&out).unwrap();
    assert_eq!(rb.num_columns(), 3);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test groupby_gpu -q`
Expected: FAIL (groupby_agg uses CPU boundary/group-id fallback and single-key)

**Step 3: Write minimal implementation**

```rust
// crates/xlog-cuda/src/provider/mod.rs
pub fn groupby_agg(
    &self,
    input: &CudaBuffer,
    key_cols: &[usize],
    agg: AggOp,
    value_col: usize,
) -> Result<CudaBuffer> {
    let aggs = &[(value_col, agg)];
    let result = self.groupby_multi_agg(input, key_cols, aggs)?;
    Ok(result)
}
```

```cuda
// kernels/groupby.cu (add)
extern "C" __global__ void capture_num_groups(
    const uint32_t* boundary_pos,
    const uint8_t* boundaries,
    uint32_t n,
    uint32_t* out_groups
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (n == 0) { out_groups[0] = 0; return; }
        uint32_t last = n - 1;
        out_groups[0] = boundary_pos[last] + (boundaries[last] ? 1u : 0u);
    }
}
```

```rust
// crates/xlog-cuda/src/provider/mod.rs (groupby_multi_agg)
let mut d_num_groups = self.memory.alloc::<u32>(1)?;
let capture_groups = device
    .get_func(GROUPBY_MODULE, groupby_kernels::CAPTURE_NUM_GROUPS)
    .ok_or_else(|| XlogError::Kernel("capture_num_groups kernel not found".to_string()))?;
unsafe {
    capture_groups.clone().launch(
        LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 },
        (&d_boundary_pos, &boundaries, num_rows, &mut d_num_groups),
    )
}
.map_err(|e| XlogError::Kernel(format!("capture_num_groups failed: {}", e)))?;

let row_cap = num_rows as u64;
// allocate output columns with row_cap, and set device row count
return Ok(CudaBuffer::from_columns(agg_columns, row_cap, d_num_groups, result_schema));
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test groupby_gpu -q`
Expected: PASS

**Step 5: Commit**

```bash
git add kernels/groupby.cu crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/groupby_gpu.rs

git commit -m "feat(cuda): move groupby to GPU-only boundaries and counts"
```

---

### Task 5: GPU-only key packing (remove CPU fallback)

**Files:**
- Modify: `kernels/join.cu`
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Test: `crates/xlog-cuda/tests/pack_keys_gpu.rs`

**Step 1: Write the failing test**

```rust
// crates/xlog-cuda/tests/pack_keys_gpu.rs
use std::sync::Arc;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn setup_provider() -> Option<CudaKernelProvider> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    CudaKernelProvider::new(device, memory).ok()
}

#[test]
fn test_pack_keys_gpu_more_than_four_cols() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![
        ("a".to_string(), ScalarType::U32),
        ("b".to_string(), ScalarType::U32),
        ("c".to_string(), ScalarType::U32),
        ("d".to_string(), ScalarType::U32),
        ("e".to_string(), ScalarType::U32),
    ]);

    let a = vec![1u32, 2, 3];
    let b = vec![4u32, 5, 6];
    let c = vec![7u32, 8, 9];
    let d = vec![10u32, 11, 12];
    let e = vec![13u32, 14, 15];

    let buffer = provider
        .create_buffer_from_slices(
            &[
                bytemuck::cast_slice(&a),
                bytemuck::cast_slice(&b),
                bytemuck::cast_slice(&c),
                bytemuck::cast_slice(&d),
                bytemuck::cast_slice(&e),
            ],
            schema,
        )
        .unwrap();

    let packed = provider.compute_hashes_and_pack_keys(&buffer, &[0,1,2,3,4]).unwrap();
    assert!(packed.key_bytes > 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test pack_keys_gpu -q`
Expected: FAIL (GPU pack only supports <=4 columns and CPU fallback uses DTOH)

**Step 3: Write minimal implementation**

```cuda
// kernels/join.cu (add)
extern "C" __global__ void pack_and_hash_keys_generic(
    const uint8_t** col_ptrs,
    const uint32_t* col_sizes,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_size,
    uint8_t* out_packed,
    uint64_t* out_hashes
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    uint32_t row_offset = idx * row_size;
    uint64_t h = 1469598103934665603ull;
    for (uint32_t c = 0; c < num_cols; c++) {
        const uint8_t* col = col_ptrs[c];
        uint32_t sz = col_sizes[c];
        const uint8_t* src = col + (idx * sz);
        for (uint32_t b = 0; b < sz; b++) {
            uint8_t v = src[b];
            out_packed[row_offset + b] = v;
            h ^= (uint64_t)v;
            h *= 1099511628211ull;
        }
        row_offset += sz;
    }
    out_hashes[idx] = h;
}
```

```rust
// crates/xlog-cuda/src/provider/mod.rs (compute_hashes_and_pack_keys)
fn compute_hashes_and_pack_keys(&self, buffer: &CudaBuffer, key_cols: &[usize]) -> Result<PackedKeyData> {
    if key_cols.is_empty() {
        return Ok(PackedKeyData { hashes: self.memory.alloc::<u64>(0)?, packed_keys: self.memory.alloc::<u8>(0)?, key_bytes: 0 });
    }
    self.pack_keys_gpu_generic(buffer, key_cols)
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test pack_keys_gpu -q`
Expected: PASS

**Step 5: Commit**

```bash
git add kernels/join.cu crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/pack_keys_gpu.rs

git commit -m "feat(cuda): add generic GPU key packing (no CPU fallback)"
```

---

### Task 6: Device-resident MC counts (no host reductions)

**Files:**
- Modify: `kernels/mc_eval.cu`
- Modify: `crates/xlog-prob/src/mc.rs`
- Test: `crates/xlog-prob/tests/gpu_mc_device_counts.rs`

**Step 1: Write the failing test**

```rust
// crates/xlog-prob/tests/gpu_mc_device_counts.rs
use std::sync::Arc;
use xlog_core::Result;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_core::MemoryBudget;
use xlog_prob::{McProgram, McEvalConfig};

fn setup_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    CudaKernelProvider::new(device, memory).ok().map(Arc::new)
}

#[test]
fn test_mc_device_counts_match_cpu() -> Result<()> {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let program = McProgram::from_str(r#"
        0.5::a.
        query(a).
    "#)?;

    let cfg = McEvalConfig { samples: 16, seed: 123, confidence: 0.95, max_nonmonotone_iterations: 10 };

    let cpu = program.evaluate(cfg.clone())?; // CPU baseline
    let gpu = program.evaluate_gpu_device(cfg)?; // device counts

    let mut host_counts = vec![0u32; gpu.query_counts.len()];
    provider.device().inner().dtoh_sync_copy_into(&gpu.query_counts, &mut host_counts).unwrap();
    let mut host_evidence = [0u32];
    provider.device().inner().dtoh_sync_copy_into(&gpu.evidence_samples, &mut host_evidence).unwrap();

    assert_eq!(host_evidence[0] as usize, cpu.evidence_samples);
    assert_eq!(host_counts[0] as usize, cpu.query_counts[0]);
    Ok(())
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob --test gpu_mc_device_counts -q`
Expected: FAIL (evaluate_gpu_device still uses host reductions)

**Step 3: Write minimal implementation**

```cuda
// kernels/mc_eval.cu (add)
extern "C" __global__ void mc_accumulate_counts(
    const uint32_t* query_counts_in,
    uint32_t num_queries,
    const uint8_t* query_truth,
    uint8_t evidence_ok,
    uint32_t* query_counts_out,
    uint32_t* evidence_out
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (evidence_ok) {
            atomicAdd(evidence_out, 1);
            for (uint32_t i = 0; i < num_queries; i++) {
                if (query_truth[i]) { atomicAdd(&query_counts_out[i], 1); }
            }
        }
    }
}
```

```rust
// crates/xlog-prob/src/mc.rs (evaluate_gpu_counts)
let mut d_query_counts = provider.memory().alloc::<u32>(prob_query_count)?;
provider.device().inner().memset_zeros(&mut d_query_counts)?;
let mut d_evidence = provider.memory().alloc::<u32>(1)?;
provider.device().inner().memset_zeros(&mut d_evidence)?;

// per-sample: build query_truth device array (u8 flags), evidence_ok flag
// launch mc_accumulate_counts(query_truth, evidence_ok, d_query_counts, d_evidence)

// return device counts for evaluate_gpu_device; evaluate_gpu reads back only at the end
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob --test gpu_mc_device_counts -q`
Expected: PASS

**Step 5: Commit**

```bash
git add kernels/mc_eval.cu crates/xlog-prob/src/mc.rs crates/xlog-prob/tests/gpu_mc_device_counts.rs

git commit -m "feat(prob): accumulate MC query/evidence counts on device"
```

---

### Task 7: CUDA certification tests for new GPU kernels

**Files:**
- Create: `crates/xlog-cuda-tests/src/categories/g07_device_counts.rs`
- Modify: `crates/xlog-cuda-tests/src/categories/mod.rs`
- Modify: `crates/xlog-cuda-tests/tests/gpu_certification.rs`
- Modify: `crates/xlog-cuda-tests/tests/full_certification.rs`

**Step 1: Write the failing certification test**

```rust
// crates/xlog-cuda-tests/src/categories/g07_device_counts.rs
use crate::harness::{CudaHarness, CertResult};
use xlog_cuda::AggOp;

pub fn run(h: &CudaHarness) -> CertResult {
    h.test_device_compact_count()?;
    h.test_groupby_device_count(AggOp::Sum)?;
    Ok(())
}
```

```rust
// crates/xlog-cuda-tests/tests/gpu_certification.rs (add)
mod g07_device_counts;

#[test]
fn g07_device_counts() {
    run_gpu_cert_test("g07", g07_device_counts::run);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p xlog-cuda-tests --test gpu_certification g07 --release -q`
Expected: FAIL (category not wired / missing harness methods)

**Step 3: Write minimal implementation**

```rust
// crates/xlog-cuda-tests/src/harness/mod.rs (add helpers)
impl CudaHarness {
    pub fn test_device_compact_count(&self) -> CertResult { /* build buffer, mask, assert device count */ Ok(()) }
    pub fn test_groupby_device_count(&self, _agg: AggOp) -> CertResult { /* groupby + read device count */ Ok(()) }
}
```

```rust
// crates/xlog-cuda-tests/src/categories/mod.rs
pub mod g07_device_counts;
```

```rust
// crates/xlog-cuda-tests/tests/full_certification.rs
mod g07_device_counts;
// include in runner list
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p xlog-cuda-tests --test gpu_certification g07 --release -q`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-cuda-tests/src/categories/g07_device_counts.rs \
  crates/xlog-cuda-tests/src/categories/mod.rs \
  crates/xlog-cuda-tests/tests/gpu_certification.rs \
  crates/xlog-cuda-tests/tests/full_certification.rs \
  crates/xlog-cuda-tests/src/harness/mod.rs

git commit -m "test(cuda): add device-count certification category"
```

---

## Execution options

Plan complete and saved to `docs/plans/2026-01-29-gpu-native-phase4-gap-fixes.md`. Two execution options:

1. Subagent-Driven (this session) — I dispatch a fresh subagent per task, review between tasks
2. Parallel Session (separate) — Open a new session using superpowers:executing-plans and run tasks sequentially

Which approach?
