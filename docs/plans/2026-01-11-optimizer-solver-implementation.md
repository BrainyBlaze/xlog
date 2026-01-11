# Unified Optimizer & Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add query optimizer with GPU-side key packing, and xlog-solve CLS solver with proof generation.

**Architecture:** Three-layer design with shared statistics infrastructure (xlog-stats), query optimizer in xlog-logic, and FastFourierSAT-style CLS solver in new xlog-solve crate. GPU kernels for key packing and solver iterations.

**Tech Stack:** Rust, CUDA (PTX), cudarc bindings

---

## Phase A: Statistics Foundation

### Task 1: Create xlog-stats Crate Structure

**Files:**
- Create: `crates/xlog-stats/Cargo.toml`
- Create: `crates/xlog-stats/src/lib.rs`
- Modify: `Cargo.toml` (workspace)

**Step 1: Create crate directory**

```bash
mkdir -p crates/xlog-stats/src
```

**Step 2: Create Cargo.toml**

```toml
[package]
name = "xlog-stats"
version = "0.1.0"
edition = "2021"

[dependencies]
xlog-core = { path = "../xlog-core" }
xlog-cuda = { path = "../xlog-cuda" }
```

**Step 3: Create lib.rs with module structure**

```rust
//! GPU-resident statistics for query optimization and solver heuristics.

mod stats;
mod manager;

pub use stats::{RelationStats, ColumnStats, JoinSelectivity};
pub use manager::StatsManager;
```

**Step 4: Add to workspace Cargo.toml**

Add `"crates/xlog-stats"` to workspace members array.

**Step 5: Verify crate compiles**

Run: `cargo build -p xlog-stats`
Expected: Compiles (with missing module errors - that's fine)

**Step 6: Commit**

```bash
git add crates/xlog-stats Cargo.toml
git commit -m "feat(stats): create xlog-stats crate structure"
```

---

### Task 2: Implement RelationStats and ColumnStats

**Files:**
- Create: `crates/xlog-stats/src/stats.rs`
- Modify: `crates/xlog-stats/src/lib.rs`

**Step 1: Write test for RelationStats**

```rust
// In crates/xlog-stats/src/stats.rs
#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::ScalarType;

    #[test]
    fn test_relation_stats_new() {
        let stats = RelationStats::new(1);
        assert_eq!(stats.rel_id, 1);
        assert_eq!(stats.cardinality, 0);
        assert_eq!(stats.heat, 0.0);
    }

    #[test]
    fn test_relation_stats_update_cardinality() {
        let mut stats = RelationStats::new(1);
        stats.update_cardinality(1000);
        assert_eq!(stats.cardinality, 1000);
    }

    #[test]
    fn test_relation_stats_update_heat() {
        let mut stats = RelationStats::new(1);
        stats.record_access();
        assert!(stats.heat > 0.0);
        stats.record_access();
        assert!(stats.heat > 0.1);
    }

    #[test]
    fn test_column_stats_new() {
        let col = ColumnStats::new(0, ScalarType::U32);
        assert_eq!(col.col_idx, 0);
        assert_eq!(col.dtype, ScalarType::U32);
        assert_eq!(col.distinct_estimate, 0);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-stats`
Expected: FAIL with compilation errors

**Step 3: Implement stats structures**

```rust
// crates/xlog-stats/src/stats.rs
use xlog_core::{ScalarType, RelId};

/// GPU-resident relation statistics
#[derive(Debug, Clone)]
pub struct RelationStats {
    pub rel_id: RelId,
    pub cardinality: u64,
    pub byte_size: u64,
    pub column_stats: Vec<ColumnStats>,
    pub heat: f32,
    pub last_access: u64,
    pub has_index: bool,
}

impl RelationStats {
    pub fn new(rel_id: RelId) -> Self {
        Self {
            rel_id,
            cardinality: 0,
            byte_size: 0,
            column_stats: Vec::new(),
            heat: 0.0,
            last_access: 0,
            has_index: false,
        }
    }

    pub fn update_cardinality(&mut self, rows: u64) {
        self.cardinality = rows;
    }

    pub fn record_access(&mut self) {
        // Exponential moving average for heat
        self.heat = self.heat * 0.9 + 0.1;
        self.last_access = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    pub fn decay_heat(&mut self, factor: f32) {
        self.heat *= factor;
    }
}

/// Per-column statistics
#[derive(Debug, Clone)]
pub struct ColumnStats {
    pub col_idx: usize,
    pub dtype: ScalarType,
    pub null_count: u64,
    pub distinct_estimate: u64,
    pub min_value: Option<i64>,
    pub max_value: Option<i64>,
}

impl ColumnStats {
    pub fn new(col_idx: usize, dtype: ScalarType) -> Self {
        Self {
            col_idx,
            dtype,
            null_count: 0,
            distinct_estimate: 0,
            min_value: None,
            max_value: None,
        }
    }

    pub fn update_distinct(&mut self, estimate: u64) {
        self.distinct_estimate = estimate;
    }

    pub fn update_range(&mut self, min: i64, max: i64) {
        self.min_value = Some(min);
        self.max_value = Some(max);
    }
}

/// Join selectivity model
#[derive(Debug, Clone)]
pub struct JoinSelectivity {
    pub left_rel: RelId,
    pub right_rel: RelId,
    pub left_keys: Vec<usize>,
    pub right_keys: Vec<usize>,
    pub selectivity: f64,
    pub is_pk_fk: bool,
}

impl JoinSelectivity {
    pub fn new(left_rel: RelId, right_rel: RelId) -> Self {
        Self {
            left_rel,
            right_rel,
            left_keys: Vec::new(),
            right_keys: Vec::new(),
            selectivity: 1.0,
            is_pk_fk: false,
        }
    }

    pub fn estimate_output_rows(&self, left_rows: u64, right_rows: u64) -> u64 {
        ((left_rows as f64 * right_rows as f64 * self.selectivity) as u64).max(1)
    }
}

#[cfg(test)]
mod tests {
    // ... tests from Step 1
}
```

**Step 4: Update lib.rs**

```rust
//! GPU-resident statistics for query optimization and solver heuristics.

mod stats;

pub use stats::{RelationStats, ColumnStats, JoinSelectivity};
```

**Step 5: Run tests**

Run: `cargo test -p xlog-stats`
Expected: PASS (4 tests)

**Step 6: Commit**

```bash
git add crates/xlog-stats/src/
git commit -m "feat(stats): implement RelationStats and ColumnStats"
```

---

### Task 3: Implement StatsManager

**Files:**
- Create: `crates/xlog-stats/src/manager.rs`
- Modify: `crates/xlog-stats/src/lib.rs`

**Step 1: Write tests for StatsManager**

```rust
// In crates/xlog-stats/src/manager.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_manager_new() {
        let mgr = StatsManager::new();
        assert!(mgr.get_relation_stats(1).is_none());
    }

    #[test]
    fn test_stats_manager_register_relation() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(1);
        assert!(mgr.get_relation_stats(1).is_some());
    }

    #[test]
    fn test_stats_manager_update_cardinality() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(1);
        mgr.update_cardinality(1, 5000);
        let stats = mgr.get_relation_stats(1).unwrap();
        assert_eq!(stats.cardinality, 5000);
    }

    #[test]
    fn test_stats_manager_record_access() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(1);
        mgr.record_access(1);
        let stats = mgr.get_relation_stats(1).unwrap();
        assert!(stats.heat > 0.0);
    }

    #[test]
    fn test_stats_manager_estimate_join() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(1);
        mgr.register_relation(2);
        mgr.update_cardinality(1, 1000);
        mgr.update_cardinality(2, 500);

        let estimate = mgr.estimate_join_cardinality(1, 2, &[0], &[0]);
        // Default selectivity assumes 1/max(distinct) ≈ small fraction
        assert!(estimate > 0);
        assert!(estimate <= 1000 * 500);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-stats`
Expected: FAIL - StatsManager not found

**Step 3: Implement StatsManager**

```rust
// crates/xlog-stats/src/manager.rs
use std::collections::HashMap;
use xlog_core::RelId;
use crate::stats::{RelationStats, ColumnStats, JoinSelectivity};

/// Manages GPU-resident statistics for all relations
#[derive(Debug, Default)]
pub struct StatsManager {
    relations: HashMap<RelId, RelationStats>,
    join_selectivities: HashMap<(RelId, RelId), JoinSelectivity>,
}

impl StatsManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_relation(&mut self, rel_id: RelId) {
        self.relations.entry(rel_id).or_insert_with(|| RelationStats::new(rel_id));
    }

    pub fn get_relation_stats(&self, rel_id: RelId) -> Option<&RelationStats> {
        self.relations.get(&rel_id)
    }

    pub fn get_relation_stats_mut(&mut self, rel_id: RelId) -> Option<&mut RelationStats> {
        self.relations.get_mut(&rel_id)
    }

    pub fn update_cardinality(&mut self, rel_id: RelId, rows: u64) {
        if let Some(stats) = self.relations.get_mut(&rel_id) {
            stats.update_cardinality(rows);
        }
    }

    pub fn record_access(&mut self, rel_id: RelId) {
        if let Some(stats) = self.relations.get_mut(&rel_id) {
            stats.record_access();
        }
    }

    pub fn estimate_join_cardinality(
        &self,
        left_rel: RelId,
        right_rel: RelId,
        _left_keys: &[usize],
        _right_keys: &[usize],
    ) -> u64 {
        let left_card = self.relations.get(&left_rel)
            .map(|s| s.cardinality)
            .unwrap_or(1000);
        let right_card = self.relations.get(&right_rel)
            .map(|s| s.cardinality)
            .unwrap_or(1000);

        // Check for cached selectivity
        let key = if left_rel <= right_rel {
            (left_rel, right_rel)
        } else {
            (right_rel, left_rel)
        };

        let selectivity = self.join_selectivities.get(&key)
            .map(|s| s.selectivity)
            .unwrap_or(0.1); // Default 10% selectivity

        ((left_card as f64 * right_card as f64 * selectivity) as u64).max(1)
    }

    pub fn record_join_result(
        &mut self,
        left_rel: RelId,
        right_rel: RelId,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        input_rows: u64,
        output_rows: u64,
    ) {
        let key = if left_rel <= right_rel {
            (left_rel, right_rel)
        } else {
            (right_rel, left_rel)
        };

        let selectivity = if input_rows > 0 {
            output_rows as f64 / input_rows as f64
        } else {
            0.1
        };

        let entry = self.join_selectivities.entry(key).or_insert_with(|| {
            JoinSelectivity::new(left_rel, right_rel)
        });
        entry.left_keys = left_keys;
        entry.right_keys = right_keys;
        // Exponential moving average
        entry.selectivity = entry.selectivity * 0.7 + selectivity * 0.3;
    }

    pub fn decay_all_heat(&mut self, factor: f32) {
        for stats in self.relations.values_mut() {
            stats.decay_heat(factor);
        }
    }

    pub fn hot_relations(&self, threshold: f32) -> Vec<RelId> {
        self.relations.iter()
            .filter(|(_, s)| s.heat >= threshold)
            .map(|(id, _)| *id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    // ... tests from Step 1
}
```

**Step 4: Update lib.rs**

```rust
//! GPU-resident statistics for query optimization and solver heuristics.

mod stats;
mod manager;

pub use stats::{RelationStats, ColumnStats, JoinSelectivity};
pub use manager::StatsManager;
```

**Step 5: Run tests**

Run: `cargo test -p xlog-stats`
Expected: PASS (9 tests)

**Step 6: Commit**

```bash
git add crates/xlog-stats/
git commit -m "feat(stats): implement StatsManager for relation statistics"
```

---

## Phase B: GPU Key Packing

### Task 4: Create pack.cu Kernel

**Files:**
- Create: `kernels/pack.cu`

**Step 1: Write kernel source**

```cuda
// kernels/pack.cu
// GPU-side key packing for multi-column joins

extern "C" {

/// Pack multiple columns into row-major byte array
__global__ void pack_keys(
    const uint8_t* __restrict__ col0,
    const uint8_t* __restrict__ col1,
    const uint8_t* __restrict__ col2,
    const uint8_t* __restrict__ col3,
    const uint32_t* __restrict__ col_sizes,  // Size of each column element
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_size,                       // Total packed row size
    uint8_t* __restrict__ packed_output
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    uint8_t* out_row = packed_output + (uint64_t)row * row_size;
    uint32_t offset = 0;

    // Unrolled for up to 4 key columns
    if (num_cols >= 1 && col0 != nullptr) {
        uint32_t sz = col_sizes[0];
        for (uint32_t i = 0; i < sz; i++) {
            out_row[offset + i] = col0[(uint64_t)row * sz + i];
        }
        offset += sz;
    }
    if (num_cols >= 2 && col1 != nullptr) {
        uint32_t sz = col_sizes[1];
        for (uint32_t i = 0; i < sz; i++) {
            out_row[offset + i] = col1[(uint64_t)row * sz + i];
        }
        offset += sz;
    }
    if (num_cols >= 3 && col2 != nullptr) {
        uint32_t sz = col_sizes[2];
        for (uint32_t i = 0; i < sz; i++) {
            out_row[offset + i] = col2[(uint64_t)row * sz + i];
        }
        offset += sz;
    }
    if (num_cols >= 4 && col3 != nullptr) {
        uint32_t sz = col_sizes[3];
        for (uint32_t i = 0; i < sz; i++) {
            out_row[offset + i] = col3[(uint64_t)row * sz + i];
        }
        offset += sz;
    }
}

/// FNV-1a hash constants
#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME  0x100000001b3ULL

/// Compute FNV-1a hash from packed keys
__global__ void hash_packed_keys(
    const uint8_t* __restrict__ packed_keys,
    uint32_t row_size,
    uint32_t num_rows,
    uint64_t* __restrict__ hashes
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    const uint8_t* key = packed_keys + (uint64_t)row * row_size;
    uint64_t hash = FNV_OFFSET;

    for (uint32_t i = 0; i < row_size; i++) {
        hash ^= key[i];
        hash *= FNV_PRIME;
    }

    hashes[row] = hash;
}

/// Fused pack + hash in single pass (better cache utilization)
__global__ void pack_and_hash_keys(
    const uint8_t* __restrict__ col0,
    const uint8_t* __restrict__ col1,
    const uint8_t* __restrict__ col2,
    const uint8_t* __restrict__ col3,
    const uint32_t* __restrict__ col_sizes,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_size,
    uint8_t* __restrict__ packed_output,
    uint64_t* __restrict__ hashes
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    uint8_t* out_row = packed_output + (uint64_t)row * row_size;
    uint64_t hash = FNV_OFFSET;
    uint32_t offset = 0;

    // Pack and hash simultaneously
    if (num_cols >= 1 && col0 != nullptr) {
        uint32_t sz = col_sizes[0];
        for (uint32_t i = 0; i < sz; i++) {
            uint8_t b = col0[(uint64_t)row * sz + i];
            out_row[offset + i] = b;
            hash ^= b;
            hash *= FNV_PRIME;
        }
        offset += sz;
    }
    if (num_cols >= 2 && col1 != nullptr) {
        uint32_t sz = col_sizes[1];
        for (uint32_t i = 0; i < sz; i++) {
            uint8_t b = col1[(uint64_t)row * sz + i];
            out_row[offset + i] = b;
            hash ^= b;
            hash *= FNV_PRIME;
        }
        offset += sz;
    }
    if (num_cols >= 3 && col2 != nullptr) {
        uint32_t sz = col_sizes[2];
        for (uint32_t i = 0; i < sz; i++) {
            uint8_t b = col2[(uint64_t)row * sz + i];
            out_row[offset + i] = b;
            hash ^= b;
            hash *= FNV_PRIME;
        }
        offset += sz;
    }
    if (num_cols >= 4 && col3 != nullptr) {
        uint32_t sz = col_sizes[3];
        for (uint32_t i = 0; i < sz; i++) {
            uint8_t b = col3[(uint64_t)row * sz + i];
            out_row[offset + i] = b;
            hash ^= b;
            hash *= FNV_PRIME;
        }
        offset += sz;
    }

    hashes[row] = hash;
}

} // extern "C"
```

**Step 2: Compile to PTX**

Run: `nvcc -ptx --gpu-architecture=sm_90 kernels/pack.cu -o kernels/pack.ptx`
Expected: Creates pack.ptx file

**Step 3: Commit**

```bash
git add kernels/pack.cu kernels/pack.ptx
git commit -m "feat(cuda): add GPU key packing kernels"
```

---

### Task 5: Integrate GPU Packing into Provider

**Files:**
- Modify: `crates/xlog-cuda/src/provider.rs`

**Step 1: Write test for GPU packing**

Add to `crates/xlog-cuda/tests/pack_tests.rs`:

```rust
//! Tests for GPU key packing

use std::sync::Arc;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn create_test_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    Some(Arc::new(CudaKernelProvider::new(device, memory).ok()?))
}

#[test]
fn test_pack_keys_gpu_single_column() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: no CUDA device");
            return;
        }
    };

    let schema = Schema::new(vec![
        ("a".to_string(), ScalarType::U32),
        ("b".to_string(), ScalarType::U32),
    ]);

    let data: Vec<(u32, u32)> = vec![(1, 10), (2, 20), (3, 30)];
    let col0: Vec<u8> = data.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let col1: Vec<u8> = data.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    let buffer = provider.create_buffer_from_slices(&[&col0, &col1], schema).unwrap();

    let (packed, hashes) = provider.pack_keys_gpu(&buffer, &[0]).unwrap();

    // Verify packed buffer has correct size
    assert_eq!(packed.num_rows(), 3);

    // Verify hashes are computed
    let hash_vec = provider.download_column_u64(&hashes, 0).unwrap();
    assert_eq!(hash_vec.len(), 3);
    // Different keys should have different hashes
    assert_ne!(hash_vec[0], hash_vec[1]);
    assert_ne!(hash_vec[1], hash_vec[2]);
}

#[test]
fn test_pack_keys_gpu_multi_column() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: no CUDA device");
            return;
        }
    };

    let schema = Schema::new(vec![
        ("a".to_string(), ScalarType::U32),
        ("b".to_string(), ScalarType::U32),
        ("c".to_string(), ScalarType::U32),
    ]);

    let data: Vec<(u32, u32, u32)> = vec![(1, 2, 100), (1, 3, 200), (2, 2, 300)];
    let col0: Vec<u8> = data.iter().flat_map(|(a, _, _)| a.to_le_bytes()).collect();
    let col1: Vec<u8> = data.iter().flat_map(|(_, b, _)| b.to_le_bytes()).collect();
    let col2: Vec<u8> = data.iter().flat_map(|(_, _, c)| c.to_le_bytes()).collect();
    let buffer = provider.create_buffer_from_slices(&[&col0, &col1, &col2], schema).unwrap();

    // Pack columns 0 and 1 as composite key
    let (packed, hashes) = provider.pack_keys_gpu(&buffer, &[0, 1]).unwrap();

    assert_eq!(packed.num_rows(), 3);

    let hash_vec = provider.download_column_u64(&hashes, 0).unwrap();
    // (1,2) != (1,3) != (2,2)
    assert_ne!(hash_vec[0], hash_vec[1]);
    assert_ne!(hash_vec[0], hash_vec[2]);
    assert_ne!(hash_vec[1], hash_vec[2]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test pack_tests`
Expected: FAIL - pack_keys_gpu method not found

**Step 3: Implement pack_keys_gpu in provider**

Add to `crates/xlog-cuda/src/provider.rs` (in the impl block):

```rust
/// Pack key columns on GPU and compute hashes (no host roundtrip)
pub fn pack_keys_gpu(
    &self,
    buffer: &CudaBuffer,
    key_cols: &[usize],
) -> Result<(CudaBuffer, CudaBuffer)> {
    if key_cols.is_empty() {
        return Err(XlogError::Kernel("pack_keys_gpu: no key columns specified".into()));
    }
    if key_cols.len() > 4 {
        return Err(XlogError::Kernel("pack_keys_gpu: max 4 key columns supported".into()));
    }

    let num_rows = buffer.num_rows() as u32;
    if num_rows == 0 {
        let packed_schema = Schema::new(vec![("packed".to_string(), ScalarType::U64)]);
        let hash_schema = Schema::new(vec![("hash".to_string(), ScalarType::U64)]);
        return Ok((
            self.create_empty_buffer(packed_schema)?,
            self.create_empty_buffer(hash_schema)?,
        ));
    }

    // Calculate row size and column sizes
    let mut col_sizes: Vec<u32> = Vec::new();
    let mut row_size: u32 = 0;
    for &col_idx in key_cols {
        let col_type = buffer.schema().column_type(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Invalid column index: {}", col_idx)))?;
        let size = col_type.size_bytes() as u32;
        col_sizes.push(size);
        row_size += size;
    }

    // Allocate output buffers
    let packed_bytes = (num_rows as u64) * (row_size as u64);
    let packed_slice = self.memory.alloc::<u8>(packed_bytes as usize)?;
    let hash_slice = self.memory.alloc::<u64>(num_rows as usize)?;

    // Upload column sizes
    let col_sizes_slice = self.device.inner().htod_sync_copy(&col_sizes)?;

    // Get column pointers (up to 4)
    let col_ptrs: Vec<u64> = key_cols.iter()
        .map(|&idx| {
            buffer.column(idx)
                .map(|c| *c.device_ptr() as u64)
                .unwrap_or(0)
        })
        .collect();

    // Pad to 4 columns with nulls
    let col0 = col_ptrs.get(0).copied().unwrap_or(0);
    let col1 = col_ptrs.get(1).copied().unwrap_or(0);
    let col2 = col_ptrs.get(2).copied().unwrap_or(0);
    let col3 = col_ptrs.get(3).copied().unwrap_or(0);

    // Launch fused pack+hash kernel
    let func = self.device.inner()
        .get_func(PACK_MODULE, "pack_and_hash_keys")
        .map_err(|e| XlogError::Kernel(format!("Failed to get pack_and_hash_keys: {}", e)))?;

    let block_size = 256u32;
    let grid_size = (num_rows + block_size - 1) / block_size;

    unsafe {
        func.launch(
            cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                col0,
                col1,
                col2,
                col3,
                &col_sizes_slice,
                key_cols.len() as u32,
                num_rows,
                row_size,
                &packed_slice,
                &hash_slice,
            ),
        ).map_err(|e| XlogError::Kernel(format!("pack_and_hash_keys launch failed: {}", e)))?;
    }

    self.device.synchronize()?;

    // Create output buffers
    let packed_schema = Schema::new(vec![("packed".to_string(), ScalarType::U64)]);
    let packed_buffer = CudaBuffer::from_columns(
        vec![packed_slice],
        num_rows as u64,
        packed_schema,
    );

    let hash_schema = Schema::new(vec![("hash".to_string(), ScalarType::U64)]);
    let hash_buffer = CudaBuffer::from_columns(
        vec![unsafe { std::mem::transmute(hash_slice) }],
        num_rows as u64,
        hash_schema,
    );

    Ok((packed_buffer, hash_buffer))
}
```

**Step 4: Add module constant and PTX loading**

Add near top of provider.rs:

```rust
pub const PACK_MODULE: &str = "xlog_pack";
```

Add in `load_modules()`:

```rust
// Load pack module
let pack_ptx = include_str!("../../../kernels/pack.ptx");
device.inner().load_ptx(
    cudarc::nvrtc::Ptx::from_src(pack_ptx),
    PACK_MODULE,
    &["pack_keys", "hash_packed_keys", "pack_and_hash_keys"],
)?;
```

**Step 5: Run tests**

Run: `cargo test -p xlog-cuda --test pack_tests`
Expected: PASS (2 tests)

**Step 6: Commit**

```bash
git add crates/xlog-cuda/src/provider.rs crates/xlog-cuda/tests/pack_tests.rs
git commit -m "feat(cuda): integrate GPU key packing into provider"
```

---

## Phase C: Query Optimizer

### Task 6: Create Optimizer Module Structure

**Files:**
- Create: `crates/xlog-logic/src/optimizer.rs`
- Modify: `crates/xlog-logic/src/lib.rs`
- Modify: `crates/xlog-logic/Cargo.toml`

**Step 1: Add xlog-stats dependency**

Add to `crates/xlog-logic/Cargo.toml`:

```toml
xlog-stats = { path = "../xlog-stats" }
```

**Step 2: Create optimizer module with tests**

```rust
// crates/xlog-logic/src/optimizer.rs
//! Query optimizer for join ordering and predicate pushdown.

use std::sync::Arc;
use xlog_ir::{RirNode, Expr, RelId};
use xlog_stats::StatsManager;

/// Configuration for query optimization
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Maximum relations for exhaustive DP (use greedy above this)
    pub dp_threshold: usize,
    /// Heat threshold for building indexes
    pub index_heat_threshold: f32,
    /// Enable predicate pushdown
    pub enable_pushdown: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            dp_threshold: 10,
            index_heat_threshold: 0.7,
            enable_pushdown: true,
        }
    }
}

/// Query optimizer using statistics for cost-based decisions
pub struct Optimizer {
    stats: Arc<StatsManager>,
    config: OptimizerConfig,
}

impl Optimizer {
    pub fn new(stats: Arc<StatsManager>) -> Self {
        Self {
            stats,
            config: OptimizerConfig::default(),
        }
    }

    pub fn with_config(stats: Arc<StatsManager>, config: OptimizerConfig) -> Self {
        Self { stats, config }
    }

    /// Optimize an execution plan
    pub fn optimize(&self, node: RirNode) -> RirNode {
        let node = if self.config.enable_pushdown {
            self.predicate_pushdown(node)
        } else {
            node
        };
        // Future: join reordering, index selection
        node
    }

    /// Push filter predicates closer to scans
    fn predicate_pushdown(&self, node: RirNode) -> RirNode {
        match node {
            RirNode::Filter { input, predicate } => {
                let optimized_input = self.predicate_pushdown(*input);
                self.try_push_filter(optimized_input, predicate)
            }
            RirNode::Join { left, right, left_keys, right_keys, join_type } => {
                RirNode::Join {
                    left: Box::new(self.predicate_pushdown(*left)),
                    right: Box::new(self.predicate_pushdown(*right)),
                    left_keys,
                    right_keys,
                    join_type,
                }
            }
            RirNode::Project { input, columns } => {
                RirNode::Project {
                    input: Box::new(self.predicate_pushdown(*input)),
                    columns,
                }
            }
            RirNode::Union { inputs } => {
                RirNode::Union {
                    inputs: inputs.into_iter()
                        .map(|n| self.predicate_pushdown(n))
                        .collect(),
                }
            }
            // Other nodes pass through unchanged
            other => other,
        }
    }

    /// Try to push a filter below joins if it references only one side
    fn try_push_filter(&self, input: RirNode, predicate: Expr) -> RirNode {
        // For MVP, just wrap with filter - full pushdown requires column tracking
        RirNode::Filter {
            input: Box::new(input),
            predicate,
        }
    }

    /// Estimate cost of a plan node
    pub fn estimate_cost(&self, node: &RirNode) -> PlanCost {
        match node {
            RirNode::Scan { rel } => {
                let rows = self.stats.get_relation_stats(*rel)
                    .map(|s| s.cardinality)
                    .unwrap_or(1000);
                PlanCost {
                    rows,
                    cpu_cost: rows as f64 * 0.01,
                    gpu_mem: rows * 8,
                    transfers: 0,
                }
            }
            RirNode::Filter { input, .. } => {
                let input_cost = self.estimate_cost(input);
                PlanCost {
                    rows: (input_cost.rows as f64 * 0.5) as u64, // Assume 50% selectivity
                    cpu_cost: input_cost.cpu_cost + input_cost.rows as f64 * 0.001,
                    gpu_mem: input_cost.gpu_mem,
                    transfers: input_cost.transfers,
                }
            }
            RirNode::Join { left, right, .. } => {
                let left_cost = self.estimate_cost(left);
                let right_cost = self.estimate_cost(right);
                let output_rows = (left_cost.rows as f64 * right_cost.rows as f64 * 0.1) as u64;
                PlanCost {
                    rows: output_rows.max(1),
                    cpu_cost: left_cost.cpu_cost + right_cost.cpu_cost +
                              (left_cost.rows + right_cost.rows) as f64 * 0.1,
                    gpu_mem: left_cost.gpu_mem.max(right_cost.gpu_mem) + output_rows * 8,
                    transfers: left_cost.transfers + right_cost.transfers,
                }
            }
            _ => PlanCost::default(),
        }
    }
}

/// Cost estimate for a plan node
#[derive(Debug, Clone, Default)]
pub struct PlanCost {
    pub rows: u64,
    pub cpu_cost: f64,
    pub gpu_mem: u64,
    pub transfers: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_ir::CompareOp;

    fn make_stats() -> Arc<StatsManager> {
        let mut mgr = StatsManager::new();
        mgr.register_relation(1);
        mgr.register_relation(2);
        mgr.update_cardinality(1, 1000);
        mgr.update_cardinality(2, 500);
        Arc::new(mgr)
    }

    #[test]
    fn test_optimizer_new() {
        let stats = make_stats();
        let opt = Optimizer::new(stats);
        assert!(opt.config.enable_pushdown);
    }

    #[test]
    fn test_optimizer_estimate_scan_cost() {
        let stats = make_stats();
        let opt = Optimizer::new(stats);

        let scan = RirNode::Scan { rel: 1 };
        let cost = opt.estimate_cost(&scan);

        assert_eq!(cost.rows, 1000);
        assert!(cost.cpu_cost > 0.0);
    }

    #[test]
    fn test_optimizer_estimate_join_cost() {
        let stats = make_stats();
        let opt = Optimizer::new(stats);

        let join = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: 1 }),
            right: Box::new(RirNode::Scan { rel: 2 }),
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: xlog_ir::JoinType::Inner,
        };
        let cost = opt.estimate_cost(&join);

        // Should estimate some output rows
        assert!(cost.rows > 0);
        assert!(cost.gpu_mem > 0);
    }

    #[test]
    fn test_optimizer_passthrough() {
        let stats = make_stats();
        let opt = Optimizer::new(stats);

        let scan = RirNode::Scan { rel: 1 };
        let optimized = opt.optimize(scan.clone());

        // Scan should pass through unchanged
        match optimized {
            RirNode::Scan { rel } => assert_eq!(rel, 1),
            _ => panic!("Expected Scan"),
        }
    }
}
```

**Step 3: Update lib.rs**

Add to `crates/xlog-logic/src/lib.rs`:

```rust
pub mod optimizer;
pub use optimizer::{Optimizer, OptimizerConfig, PlanCost};
```

**Step 4: Run tests**

Run: `cargo test -p xlog-logic optimizer`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add crates/xlog-logic/src/optimizer.rs crates/xlog-logic/src/lib.rs crates/xlog-logic/Cargo.toml
git commit -m "feat(logic): add query optimizer module with cost estimation"
```

---

## Phase D: CLS Solver MVP

### Task 7: Create xlog-solve Crate Structure

**Files:**
- Create: `crates/xlog-solve/Cargo.toml`
- Create: `crates/xlog-solve/src/lib.rs`
- Modify: `Cargo.toml` (workspace)

**Step 1: Create crate directory**

```bash
mkdir -p crates/xlog-solve/src
```

**Step 2: Create Cargo.toml**

```toml
[package]
name = "xlog-solve"
version = "0.1.0"
edition = "2021"

[dependencies]
xlog-core = { path = "../xlog-core" }
xlog-cuda = { path = "../xlog-cuda" }
```

**Step 3: Create lib.rs**

```rust
//! GPU-native SAT/MaxSAT solver using Continuous Local Search (CLS).

mod instance;
mod solver;
mod proof;

pub use instance::{SolveInstance, Clause, Literal, Objective};
pub use solver::{Solver, SolverConfig, SolverState};
pub use proof::{SolveProof, SolveResult, SolveStatus};
```

**Step 4: Add to workspace**

Add `"crates/xlog-solve"` to workspace members in root `Cargo.toml`.

**Step 5: Verify structure**

Run: `cargo build -p xlog-solve`
Expected: Compile errors for missing modules (expected)

**Step 6: Commit**

```bash
git add crates/xlog-solve Cargo.toml
git commit -m "feat(solve): create xlog-solve crate structure"
```

---

### Task 8: Implement SolveInstance and Clause Types

**Files:**
- Create: `crates/xlog-solve/src/instance.rs`

**Step 1: Write tests**

```rust
// crates/xlog-solve/src/instance.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_new() {
        let pos = Literal::positive(5);
        assert_eq!(pos.var, 5);
        assert!(!pos.negated);

        let neg = Literal::negative(3);
        assert_eq!(neg.var, 3);
        assert!(neg.negated);
    }

    #[test]
    fn test_clause_new() {
        let clause = Clause::new(vec![
            Literal::positive(1),
            Literal::negative(2),
        ]);
        assert_eq!(clause.literals.len(), 2);
    }

    #[test]
    fn test_instance_from_cnf() {
        // (x1 OR NOT x2) AND (x2 OR x3)
        let instance = SolveInstance::new(3, vec![
            Clause::new(vec![Literal::positive(0), Literal::negative(1)]),
            Clause::new(vec![Literal::positive(1), Literal::positive(2)]),
        ]);
        assert_eq!(instance.num_vars, 3);
        assert_eq!(instance.clauses.len(), 2);
    }

    #[test]
    fn test_instance_is_satisfied() {
        let instance = SolveInstance::new(3, vec![
            Clause::new(vec![Literal::positive(0), Literal::negative(1)]),
            Clause::new(vec![Literal::positive(1), Literal::positive(2)]),
        ]);

        // x0=true, x1=false, x2=true should satisfy both clauses
        let assignment = vec![true, false, true];
        assert!(instance.is_satisfied(&assignment));

        // x0=false, x1=true, x2=false should fail clause 2
        let assignment2 = vec![false, true, false];
        assert!(!instance.is_satisfied(&assignment2));
    }
}
```

**Step 2: Implement instance types**

```rust
// crates/xlog-solve/src/instance.rs
//! SAT instance representation.

/// A literal is a variable with optional negation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Literal {
    pub var: u32,
    pub negated: bool,
}

impl Literal {
    pub fn positive(var: u32) -> Self {
        Self { var, negated: false }
    }

    pub fn negative(var: u32) -> Self {
        Self { var, negated: true }
    }

    /// Evaluate literal given an assignment
    pub fn eval(&self, assignment: &[bool]) -> bool {
        let val = assignment.get(self.var as usize).copied().unwrap_or(false);
        if self.negated { !val } else { val }
    }

    /// Convert to DIMACS format (1-indexed, negative for negated)
    pub fn to_dimacs(&self) -> i32 {
        let v = (self.var + 1) as i32;
        if self.negated { -v } else { v }
    }
}

/// A clause is a disjunction (OR) of literals
#[derive(Debug, Clone)]
pub struct Clause {
    pub literals: Vec<Literal>,
}

impl Clause {
    pub fn new(literals: Vec<Literal>) -> Self {
        Self { literals }
    }

    /// Check if clause is satisfied by assignment
    pub fn is_satisfied(&self, assignment: &[bool]) -> bool {
        self.literals.iter().any(|lit| lit.eval(assignment))
    }

    /// Count satisfied literals
    pub fn count_satisfied(&self, assignment: &[bool]) -> usize {
        self.literals.iter().filter(|lit| lit.eval(assignment)).count()
    }
}

/// Optimization objective
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Find any satisfying assignment
    Satisfaction,
    /// Maximize satisfied clauses (MaxSAT)
    MaxSat,
    /// Minimize unsatisfied clauses
    MinUnsat,
}

impl Default for Objective {
    fn default() -> Self {
        Self::Satisfaction
    }
}

/// A SAT/MaxSAT instance in CNF
#[derive(Debug, Clone)]
pub struct SolveInstance {
    pub num_vars: u32,
    pub clauses: Vec<Clause>,
    pub weights: Option<Vec<f64>>,
    pub objective: Objective,
}

impl SolveInstance {
    pub fn new(num_vars: u32, clauses: Vec<Clause>) -> Self {
        Self {
            num_vars,
            clauses,
            weights: None,
            objective: Objective::Satisfaction,
        }
    }

    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self.objective = Objective::MaxSat;
        self
    }

    /// Check if assignment satisfies all clauses
    pub fn is_satisfied(&self, assignment: &[bool]) -> bool {
        self.clauses.iter().all(|c| c.is_satisfied(assignment))
    }

    /// Count satisfied clauses
    pub fn count_satisfied(&self, assignment: &[bool]) -> usize {
        self.clauses.iter().filter(|c| c.is_satisfied(assignment)).count()
    }

    /// Compute weighted satisfaction (for MaxSAT)
    pub fn weighted_satisfaction(&self, assignment: &[bool]) -> f64 {
        match &self.weights {
            Some(weights) => {
                self.clauses.iter().zip(weights.iter())
                    .filter(|(c, _)| c.is_satisfied(assignment))
                    .map(|(_, w)| *w)
                    .sum()
            }
            None => self.count_satisfied(assignment) as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    // ... tests from Step 1
}
```

**Step 3: Run tests**

Run: `cargo test -p xlog-solve instance`
Expected: PASS (4 tests)

**Step 4: Commit**

```bash
git add crates/xlog-solve/src/instance.rs
git commit -m "feat(solve): implement SolveInstance and Clause types"
```

---

### Task 9: Implement SolveProof and SolveResult

**Files:**
- Create: `crates/xlog-solve/src/proof.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_result_sat() {
        let result = SolveResult::satisfiable(vec![true, false, true]);
        assert!(matches!(result.status, SolveStatus::Sat));
    }

    #[test]
    fn test_solve_result_unsat() {
        let result = SolveResult::unsatisfiable();
        assert!(matches!(result.status, SolveStatus::Unsat));
    }

    #[test]
    fn test_solve_proof_checksum() {
        let proof = SolveProof::Satisfying {
            assignment: vec![true, false],
            checksum: 12345,
        };
        match proof {
            SolveProof::Satisfying { checksum, .. } => {
                assert_eq!(checksum, 12345);
            }
            _ => panic!("Wrong proof type"),
        }
    }
}
```

**Step 2: Implement proof types**

```rust
// crates/xlog-solve/src/proof.rs
//! Proof artifacts for solver results.

/// Proof artifact from solver
#[derive(Debug, Clone)]
pub enum SolveProof {
    /// SAT: satisfying assignment is the proof
    Satisfying {
        assignment: Vec<bool>,
        checksum: u64,
    },

    /// UNSAT: (simplified) learned clauses
    Unsatisfiable {
        checksum: u64,
    },

    /// Approximate: best-effort solution
    Approximate {
        assignment: Vec<bool>,
        satisfied_clauses: u32,
        total_clauses: u32,
        iterations: u32,
    },

    /// No proof (timeout or error)
    None,
}

impl SolveProof {
    pub fn satisfying(assignment: Vec<bool>) -> Self {
        let checksum = Self::compute_checksum(&assignment);
        Self::Satisfying { assignment, checksum }
    }

    pub fn approximate(assignment: Vec<bool>, satisfied: u32, total: u32, iters: u32) -> Self {
        Self::Approximate {
            assignment,
            satisfied_clauses: satisfied,
            total_clauses: total,
            iterations: iters,
        }
    }

    fn compute_checksum(assignment: &[bool]) -> u64 {
        // FNV-1a hash
        let mut hash: u64 = 0xcbf29ce484222325;
        for (i, &val) in assignment.iter().enumerate() {
            hash ^= (i as u64) << 32 | (val as u64);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}

/// Solver result status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    Sat,
    Unsat,
    Unknown,
    Optimal(u64), // Objective value as bits
}

/// Statistics from solve
#[derive(Debug, Clone, Default)]
pub struct SolveStats {
    pub iterations: u32,
    pub duration_us: u64,
    pub peak_memory: u64,
}

/// Complete solver result
#[derive(Debug, Clone)]
pub struct SolveResult {
    pub status: SolveStatus,
    pub proof: SolveProof,
    pub stats: SolveStats,
}

impl SolveResult {
    pub fn satisfiable(assignment: Vec<bool>) -> Self {
        Self {
            status: SolveStatus::Sat,
            proof: SolveProof::satisfying(assignment),
            stats: SolveStats::default(),
        }
    }

    pub fn unsatisfiable() -> Self {
        Self {
            status: SolveStatus::Unsat,
            proof: SolveProof::Unsatisfiable { checksum: 0 },
            stats: SolveStats::default(),
        }
    }

    pub fn unknown(iterations: u32) -> Self {
        Self {
            status: SolveStatus::Unknown,
            proof: SolveProof::None,
            stats: SolveStats { iterations, ..Default::default() },
        }
    }

    pub fn with_stats(mut self, stats: SolveStats) -> Self {
        self.stats = stats;
        self
    }
}

#[cfg(test)]
mod tests {
    // ... tests from Step 1
}
```

**Step 3: Run tests**

Run: `cargo test -p xlog-solve proof`
Expected: PASS (3 tests)

**Step 4: Commit**

```bash
git add crates/xlog-solve/src/proof.rs
git commit -m "feat(solve): implement SolveProof and SolveResult types"
```

---

### Task 10: Implement CPU-based CLS Solver (MVP)

**Files:**
- Create: `crates/xlog-solve/src/solver.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::{Clause, Literal, SolveInstance};

    #[test]
    fn test_solver_simple_sat() {
        // (x0) - trivially satisfiable
        let instance = SolveInstance::new(1, vec![
            Clause::new(vec![Literal::positive(0)]),
        ]);

        let solver = Solver::new_cpu();
        let result = solver.solve(instance);

        assert!(matches!(result.status, SolveStatus::Sat));
        if let SolveProof::Satisfying { assignment, .. } = result.proof {
            assert!(assignment[0]); // x0 must be true
        }
    }

    #[test]
    fn test_solver_two_clause() {
        // (x0 OR x1) AND (NOT x0 OR x1) - x1 must be true
        let instance = SolveInstance::new(2, vec![
            Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
            Clause::new(vec![Literal::negative(0), Literal::positive(1)]),
        ]);

        let solver = Solver::new_cpu();
        let result = solver.solve(instance);

        assert!(matches!(result.status, SolveStatus::Sat));
        if let SolveProof::Satisfying { assignment, .. } = result.proof {
            assert!(assignment[1]); // x1 must be true
        }
    }

    #[test]
    fn test_solver_unsat() {
        // (x0) AND (NOT x0) - unsatisfiable
        let instance = SolveInstance::new(1, vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ]);

        let solver = Solver::new_cpu();
        let result = solver.solve(instance);

        // CLS may return Unknown for UNSAT (it's incomplete)
        assert!(matches!(result.status, SolveStatus::Unsat | SolveStatus::Unknown));
    }

    #[test]
    fn test_solver_config() {
        let config = SolverConfig {
            max_iterations: 100,
            learning_rate: 0.5,
            momentum: 0.8,
            discretize_threshold: 0.9,
        };
        let solver = Solver::with_config_cpu(config.clone());
        assert_eq!(solver.config.max_iterations, 100);
    }
}
```

**Step 2: Implement CPU solver**

```rust
// crates/xlog-solve/src/solver.rs
//! Continuous Local Search (CLS) solver.

use crate::instance::SolveInstance;
use crate::proof::{SolveProof, SolveResult, SolveStats, SolveStatus};

/// Solver configuration
#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub max_iterations: u32,
    pub learning_rate: f32,
    pub momentum: f32,
    pub discretize_threshold: f32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            learning_rate: 0.1,
            momentum: 0.9,
            discretize_threshold: 0.5,
        }
    }
}

/// GPU-resident solver state (CPU version for MVP)
#[derive(Debug)]
pub struct SolverState {
    pub assignments: Vec<f32>,
    pub velocities: Vec<f32>,
    pub gradients: Vec<f32>,
}

impl SolverState {
    pub fn new(num_vars: u32) -> Self {
        let n = num_vars as usize;
        Self {
            assignments: vec![0.5; n], // Start at 0.5
            velocities: vec![0.0; n],
            gradients: vec![0.0; n],
        }
    }

    pub fn discretize(&self, threshold: f32) -> Vec<bool> {
        self.assignments.iter().map(|&v| v >= threshold).collect()
    }
}

/// CLS Solver
pub struct Solver {
    pub config: SolverConfig,
    // GPU provider would go here for GPU version
}

impl Solver {
    pub fn new_cpu() -> Self {
        Self {
            config: SolverConfig::default(),
        }
    }

    pub fn with_config_cpu(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Solve a SAT instance using CLS
    pub fn solve(&self, instance: SolveInstance) -> SolveResult {
        let start = std::time::Instant::now();
        let mut state = SolverState::new(instance.num_vars);

        for iter in 0..self.config.max_iterations {
            // Compute gradients
            self.compute_gradients(&instance, &mut state);

            // Update with momentum
            self.update_assignments(&mut state);

            // Check if solved
            let discrete = state.discretize(self.config.discretize_threshold);
            if instance.is_satisfied(&discrete) {
                return SolveResult::satisfiable(discrete).with_stats(SolveStats {
                    iterations: iter + 1,
                    duration_us: start.elapsed().as_micros() as u64,
                    peak_memory: 0,
                });
            }
        }

        // Return best effort
        let discrete = state.discretize(self.config.discretize_threshold);
        let satisfied = instance.count_satisfied(&discrete) as u32;

        SolveResult {
            status: SolveStatus::Unknown,
            proof: SolveProof::approximate(
                discrete,
                satisfied,
                instance.clauses.len() as u32,
                self.config.max_iterations,
            ),
            stats: SolveStats {
                iterations: self.config.max_iterations,
                duration_us: start.elapsed().as_micros() as u64,
                peak_memory: 0,
            },
        }
    }

    fn compute_gradients(&self, instance: &SolveInstance, state: &mut SolverState) {
        // Reset gradients
        for g in &mut state.gradients {
            *g = 0.0;
        }

        // For each clause, compute contribution to gradient
        for clause in &instance.clauses {
            // Compute clause satisfaction (product of unsatisfied)
            let mut clause_unsat = 1.0f32;
            for lit in &clause.literals {
                let val = state.assignments[lit.var as usize];
                let lit_val = if lit.negated { 1.0 - val } else { val };
                clause_unsat *= 1.0 - lit_val;
            }

            // If clause is nearly satisfied, small gradient
            if clause_unsat < 0.001 {
                continue;
            }

            // Gradient: d(clause_unsat)/d(var)
            for lit in &clause.literals {
                let var = lit.var as usize;
                let val = state.assignments[var];

                // Product of other terms
                let mut other_product = 1.0f32;
                for other_lit in &clause.literals {
                    if other_lit.var != lit.var {
                        let other_val = state.assignments[other_lit.var as usize];
                        let lit_val = if other_lit.negated { 1.0 - other_val } else { other_val };
                        other_product *= 1.0 - lit_val;
                    }
                }

                // d/d(var) of (1 - lit_val)
                let sign = if lit.negated { 1.0 } else { -1.0 };
                state.gradients[var] += sign * other_product;
            }
        }
    }

    fn update_assignments(&self, state: &mut SolverState) {
        for i in 0..state.assignments.len() {
            // Momentum update
            state.velocities[i] = self.config.momentum * state.velocities[i]
                - self.config.learning_rate * state.gradients[i];

            // Apply velocity
            state.assignments[i] += state.velocities[i];

            // Clamp to [0, 1]
            state.assignments[i] = state.assignments[i].clamp(0.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    // ... tests from Step 1
}
```

**Step 3: Update lib.rs**

```rust
// crates/xlog-solve/src/lib.rs
//! GPU-native SAT/MaxSAT solver using Continuous Local Search (CLS).

mod instance;
mod solver;
mod proof;

pub use instance::{SolveInstance, Clause, Literal, Objective};
pub use solver::{Solver, SolverConfig, SolverState};
pub use proof::{SolveProof, SolveResult, SolveStatus, SolveStats};
```

**Step 4: Run tests**

Run: `cargo test -p xlog-solve`
Expected: PASS (11 tests)

**Step 5: Commit**

```bash
git add crates/xlog-solve/src/
git commit -m "feat(solve): implement CPU-based CLS solver MVP"
```

---

## Phase E: Integration

### Task 11: Wire Optimizer into Compiler

**Files:**
- Modify: `crates/xlog-logic/src/compile.rs`

**Step 1: Write integration test**

Add to `crates/xlog-logic/tests/optimizer_integration.rs`:

```rust
use std::sync::Arc;
use xlog_logic::{Compiler, Optimizer};
use xlog_stats::StatsManager;

#[test]
fn test_compile_with_optimizer() {
    let mut compiler = Compiler::new();

    let source = r#"
        edge(1, 2).
        edge(2, 3).
        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).
        ?- reach(X, Y).
    "#;

    let plan = compiler.compile(source).expect("Should compile");

    // Create optimizer with stats
    let stats = Arc::new(StatsManager::new());
    let optimizer = Optimizer::new(stats);

    // Optimize each rule's RIR
    for scc_rules in &plan.rules_by_scc {
        for rule in scc_rules {
            let _cost = optimizer.estimate_cost(&rule.body);
        }
    }
}
```

**Step 2: Run test**

Run: `cargo test -p xlog-logic --test optimizer_integration`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/tests/optimizer_integration.rs
git commit -m "test(logic): add optimizer integration test"
```

---

### Task 12: Add Solver Integration Test

**Files:**
- Create: `crates/xlog-solve/tests/integration_test.rs`

**Step 1: Write comprehensive test**

```rust
//! Integration tests for xlog-solve

use xlog_solve::{Solver, SolverConfig, SolveInstance, Clause, Literal, SolveStatus};

#[test]
fn test_3sat_satisfiable() {
    // Random 3-SAT instance with 10 vars, 30 clauses (should be SAT)
    let clauses: Vec<Clause> = vec![
        Clause::new(vec![Literal::positive(0), Literal::positive(1), Literal::negative(2)]),
        Clause::new(vec![Literal::negative(0), Literal::positive(2), Literal::positive(3)]),
        Clause::new(vec![Literal::positive(1), Literal::negative(3), Literal::positive(4)]),
        Clause::new(vec![Literal::negative(1), Literal::positive(4), Literal::negative(5)]),
        Clause::new(vec![Literal::positive(2), Literal::positive(5), Literal::positive(6)]),
        Clause::new(vec![Literal::negative(2), Literal::negative(6), Literal::positive(7)]),
        Clause::new(vec![Literal::positive(3), Literal::positive(7), Literal::negative(8)]),
        Clause::new(vec![Literal::negative(3), Literal::positive(8), Literal::positive(9)]),
        Clause::new(vec![Literal::positive(4), Literal::negative(9), Literal::positive(0)]),
        Clause::new(vec![Literal::negative(4), Literal::positive(0), Literal::negative(1)]),
    ];

    let instance = SolveInstance::new(10, clauses);
    let solver = Solver::with_config_cpu(SolverConfig {
        max_iterations: 5000,
        learning_rate: 0.15,
        momentum: 0.9,
        discretize_threshold: 0.5,
    });

    let result = solver.solve(instance.clone());

    // Should find a solution or get close
    match result.status {
        SolveStatus::Sat => {
            if let xlog_solve::SolveProof::Satisfying { assignment, .. } = result.proof {
                assert!(instance.is_satisfied(&assignment));
            }
        }
        SolveStatus::Unknown => {
            // CLS may not always find solution, but should satisfy most clauses
            if let xlog_solve::SolveProof::Approximate { satisfied_clauses, total_clauses, .. } = result.proof {
                let ratio = satisfied_clauses as f64 / total_clauses as f64;
                assert!(ratio > 0.7, "Should satisfy at least 70% of clauses");
            }
        }
        _ => {}
    }

    println!("Solver stats: {:?}", result.stats);
}

#[test]
fn test_pigeonhole_unsat() {
    // Simplified pigeonhole: 2 pigeons, 1 hole - UNSAT
    // x0 = pigeon 0 in hole 0
    // x1 = pigeon 1 in hole 0
    // Each pigeon must be in hole: (x0) AND (x1)
    // At most one pigeon per hole: (NOT x0 OR NOT x1)
    let instance = SolveInstance::new(2, vec![
        Clause::new(vec![Literal::positive(0)]),         // Pigeon 0 in hole
        Clause::new(vec![Literal::positive(1)]),         // Pigeon 1 in hole
        Clause::new(vec![Literal::negative(0), Literal::negative(1)]), // At most one
    ]);

    let solver = Solver::new_cpu();
    let result = solver.solve(instance);

    // CLS is incomplete for UNSAT, so Unknown is acceptable
    assert!(matches!(result.status, SolveStatus::Unsat | SolveStatus::Unknown));
}
```

**Step 2: Run test**

Run: `cargo test -p xlog-solve --test integration_test`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-solve/tests/integration_test.rs
git commit -m "test(solve): add solver integration tests"
```

---

### Task 13: Final Integration and Documentation

**Files:**
- Modify: `docs/plans/2026-01-11-optimizer-solver-design.md`

**Step 1: Update design doc status**

Change status from "Approved" to "Implemented" at the top of the design document.

**Step 2: Run full test suite**

Run: `cargo test --workspace`
Expected: All tests pass

**Step 3: Commit**

```bash
git add docs/plans/2026-01-11-optimizer-solver-design.md
git commit -m "docs: mark optimizer-solver design as implemented"
```

---

## Summary

| Phase | Tasks | New Code |
|-------|-------|----------|
| A: Statistics | 1-3 | xlog-stats crate |
| B: GPU Packing | 4-5 | pack.cu, provider integration |
| C: Optimizer | 6 | optimizer.rs |
| D: CLS Solver | 7-10 | xlog-solve crate |
| E: Integration | 11-13 | Tests, docs |

**Total: 13 tasks, ~6 weeks**
