# P3 Nice-to-Have Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement three enhancement features: CuDF interoperability, Multi-GPU support, and Adaptive indexing for query optimization.

**Architecture:** Each P3 item is independent and can be implemented in any order. CuDF integration adds Arrow-compatible I/O layer. Multi-GPU adds device pooling and distributed operations. Adaptive indexing adds statistics tracking and join strategy selection.

**Tech Stack:** Rust, CUDA C++, cudarc, Arrow (for CuDF), PTX compilation

---

## Overview

| # | Feature | Effort | Value |
|---|---------|--------|-------|
| 1 | CuDF Integration | 2-3 weeks | RAPIDS ecosystem interop |
| 2 | Multi-GPU Support | 4-5 weeks | Scale to larger datasets |
| 3 | Adaptive Indexing | 3-4 weeks | Query optimization |

**Recommended Order:** CuDF → Adaptive Indexing → Multi-GPU (increasing complexity)

---

## Feature 1: CuDF Integration

### Overview

Enable zero-copy data exchange with RAPIDS cuDF GPU DataFrames. This allows xlog to:
- Import data directly from cuDF tables
- Export results to cuDF for further analysis
- Share GPU memory without host roundtrips

### Task 1.1: Add Arrow Dependency and Type Mapping

**Files:**
- Modify: `crates/xlog-cuda/Cargo.toml`
- Modify: `crates/xlog-core/src/types.rs`

**Step 1: Add arrow dependency**

Add to `crates/xlog-cuda/Cargo.toml`:

```toml
[dependencies]
arrow = { version = "53", default-features = false, features = ["ffi"] }
```

**Step 2: Run cargo check**

```bash
cargo check -p xlog-cuda
```

Expected: Compiles with new dependency

**Step 3: Add Arrow type mapping**

Add to `crates/xlog-core/src/types.rs` after `ScalarType` enum:

```rust
impl ScalarType {
    /// Convert to Arrow DataType
    pub fn to_arrow_type(&self) -> arrow::datatypes::DataType {
        use arrow::datatypes::DataType;
        match self {
            ScalarType::Bool => DataType::Boolean,
            ScalarType::U32 => DataType::UInt32,
            ScalarType::I32 => DataType::Int32,
            ScalarType::U64 => DataType::UInt64,
            ScalarType::I64 => DataType::Int64,
            ScalarType::F32 => DataType::Float32,
            ScalarType::F64 => DataType::Float64,
            ScalarType::Symbol => DataType::UInt32, // Symbols are interned u32
        }
    }

    /// Create from Arrow DataType
    pub fn from_arrow_type(dt: &arrow::datatypes::DataType) -> Option<Self> {
        use arrow::datatypes::DataType;
        match dt {
            DataType::Boolean => Some(ScalarType::Bool),
            DataType::UInt32 => Some(ScalarType::U32),
            DataType::Int32 => Some(ScalarType::I32),
            DataType::UInt64 => Some(ScalarType::U64),
            DataType::Int64 => Some(ScalarType::I64),
            DataType::Float32 => Some(ScalarType::F32),
            DataType::Float64 => Some(ScalarType::F64),
            _ => None,
        }
    }
}
```

**Step 4: Run tests**

```bash
cargo test -p xlog-core
```

Expected: All tests pass

**Step 5: Commit**

```bash
git add crates/xlog-cuda/Cargo.toml crates/xlog-core/src/types.rs
git commit -m "feat(xlog-core): add Arrow type mapping for CuDF integration"
```

---

### Task 1.2: Create Arrow Export Functions

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Create: `crates/xlog-cuda/tests/arrow_tests.rs`

**Step 1: Write the failing test**

Create `crates/xlog-cuda/tests/arrow_tests.rs`:

```rust
//! Tests for Arrow/CuDF integration

use std::sync::Arc;
use xlog_core::{MemoryBudget, Schema, ScalarType};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn setup_provider() -> Option<CudaKernelProvider> {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return None;
    }
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    CudaKernelProvider::new(device, memory).ok()
}

#[test]
fn test_export_to_arrow_record_batch() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create a simple buffer with U32 and I64 columns
    let schema = Schema::new(vec![
        ("id".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::I64),
    ]);

    let ids: Vec<u32> = vec![1, 2, 3, 4, 5];
    let values: Vec<i64> = vec![100, 200, 300, 400, 500];

    let buffer = provider.create_buffer_from_slices(
        &[
            bytemuck::cast_slice(&ids),
            bytemuck::cast_slice(&values),
        ],
        schema,
    ).unwrap();

    // Export to Arrow RecordBatch
    let record_batch = provider.to_arrow_record_batch(&buffer).unwrap();

    assert_eq!(record_batch.num_rows(), 5);
    assert_eq!(record_batch.num_columns(), 2);

    // Verify column names
    assert_eq!(record_batch.schema().field(0).name(), "id");
    assert_eq!(record_batch.schema().field(1).name(), "value");
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p xlog-cuda --test arrow_tests -- --nocapture
```

Expected: FAIL with "to_arrow_record_batch not found"

**Step 3: Add create_buffer_from_slices helper**

Add to `provider.rs` after `create_buffer_from_u64_slice`:

```rust
/// Create a buffer from multiple column slices
///
/// # Arguments
/// * `slices` - Byte slices for each column (must match schema order)
/// * `schema` - Schema describing the columns
///
/// # Returns
/// A new CudaBuffer with the data uploaded to GPU
pub fn create_buffer_from_slices(
    &self,
    slices: &[&[u8]],
    schema: Schema,
) -> Result<CudaBuffer> {
    if slices.len() != schema.arity() {
        return Err(XlogError::Kernel(format!(
            "Slice count {} doesn't match schema arity {}",
            slices.len(),
            schema.arity()
        )));
    }

    if slices.is_empty() {
        return self.create_empty_buffer(schema);
    }

    // Calculate row count from first column
    let first_col_size = schema.column_type(0)
        .map(|t| t.size_bytes())
        .unwrap_or(4);
    let num_rows = slices[0].len() / first_col_size;

    let device = self.device.inner();
    let mut columns = Vec::with_capacity(slices.len());

    for (i, slice) in slices.iter().enumerate() {
        let d_col = device
            .htod_sync_copy(slice)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload column {}: {}", i, e)))?;
        columns.push(d_col);
    }

    Ok(CudaBuffer::from_columns(columns, num_rows as u64, schema))
}
```

**Step 4: Add Arrow export function**

Add to `provider.rs`:

```rust
/// Export CudaBuffer to Arrow RecordBatch
///
/// Downloads data from GPU and creates an Arrow RecordBatch.
/// This is NOT zero-copy but provides compatibility with Arrow ecosystem.
///
/// # Arguments
/// * `buffer` - The GPU buffer to export
///
/// # Returns
/// An Arrow RecordBatch with the same data
pub fn to_arrow_record_batch(
    &self,
    buffer: &CudaBuffer,
) -> Result<arrow::record_batch::RecordBatch> {
    use arrow::array::*;
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    let num_rows = buffer.num_rows as usize;

    // Build Arrow schema
    let fields: Vec<Field> = buffer.schema.columns.iter()
        .map(|(name, scalar_type)| {
            Field::new(name, scalar_type.to_arrow_type(), false)
        })
        .collect();
    let arrow_schema = Arc::new(ArrowSchema::new(fields));

    // Download and convert each column
    let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(buffer.arity());

    for (col_idx, (_, scalar_type)) in buffer.schema.columns.iter().enumerate() {
        let col = buffer.column(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

        let mut bytes = vec![0u8; col.len()];
        self.device.inner()
            .dtoh_sync_copy_into(col, &mut bytes)
            .map_err(|e| XlogError::Kernel(format!("Failed to download column: {}", e)))?;

        let array: Arc<dyn Array> = match scalar_type {
            ScalarType::Bool => {
                Arc::new(BooleanArray::from(bytes.iter().map(|&b| b != 0).collect::<Vec<_>>()))
            }
            ScalarType::U32 | ScalarType::Symbol => {
                let values: Vec<u32> = bytes.chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Arc::new(UInt32Array::from(values))
            }
            ScalarType::I32 => {
                let values: Vec<i32> = bytes.chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Arc::new(Int32Array::from(values))
            }
            ScalarType::U64 => {
                let values: Vec<u64> = bytes.chunks_exact(8)
                    .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                Arc::new(UInt64Array::from(values))
            }
            ScalarType::I64 => {
                let values: Vec<i64> = bytes.chunks_exact(8)
                    .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                Arc::new(Int64Array::from(values))
            }
            ScalarType::F32 => {
                let values: Vec<f32> = bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Arc::new(Float32Array::from(values))
            }
            ScalarType::F64 => {
                let values: Vec<f64> = bytes.chunks_exact(8)
                    .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                Arc::new(Float64Array::from(values))
            }
        };

        arrays.push(array);
    }

    arrow::record_batch::RecordBatch::try_new(arrow_schema, arrays)
        .map_err(|e| XlogError::Kernel(format!("Failed to create RecordBatch: {}", e)))
}
```

**Step 5: Add required imports at top of provider.rs**

```rust
use xlog_core::ScalarType; // Add if not present
```

**Step 6: Run test to verify it passes**

```bash
cargo test -p xlog-cuda --test arrow_tests -- --nocapture
```

Expected: PASS

**Step 7: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/arrow_tests.rs
git commit -m "feat(xlog-cuda): add Arrow RecordBatch export for CuDF interop"
```

---

### Task 1.3: Create Arrow Import Functions

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Modify: `crates/xlog-cuda/tests/arrow_tests.rs`

**Step 1: Write the failing test**

Add to `arrow_tests.rs`:

```rust
#[test]
fn test_import_from_arrow_record_batch() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    use arrow::array::*;
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    // Create an Arrow RecordBatch
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("x", DataType::UInt32, false),
        Field::new("y", DataType::Float64, false),
    ]));

    let x_array = Arc::new(UInt32Array::from(vec![10, 20, 30])) as Arc<dyn Array>;
    let y_array = Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5])) as Arc<dyn Array>;

    let record_batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![x_array, y_array],
    ).unwrap();

    // Import into CudaBuffer
    let buffer = provider.from_arrow_record_batch(&record_batch).unwrap();

    assert_eq!(buffer.num_rows(), 3);
    assert_eq!(buffer.arity(), 2);

    // Verify data roundtrips correctly
    let x_values = provider.download_column_u32(&buffer, 0).unwrap();
    let y_values = provider.download_column_f64(&buffer, 1).unwrap();

    assert_eq!(x_values, vec![10, 20, 30]);
    assert!((y_values[0] - 1.5).abs() < 0.001);
    assert!((y_values[1] - 2.5).abs() < 0.001);
    assert!((y_values[2] - 3.5).abs() < 0.001);
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p xlog-cuda --test arrow_tests test_import -- --nocapture
```

Expected: FAIL with "from_arrow_record_batch not found"

**Step 3: Add Arrow import function**

Add to `provider.rs`:

```rust
/// Import Arrow RecordBatch to CudaBuffer
///
/// Uploads Arrow data to GPU memory.
/// This is NOT zero-copy but provides compatibility with Arrow ecosystem.
///
/// # Arguments
/// * `record_batch` - The Arrow RecordBatch to import
///
/// # Returns
/// A new CudaBuffer with the data on GPU
pub fn from_arrow_record_batch(
    &self,
    record_batch: &arrow::record_batch::RecordBatch,
) -> Result<CudaBuffer> {
    use arrow::array::*;
    use arrow::datatypes::DataType;

    let num_rows = record_batch.num_rows() as u64;

    if num_rows == 0 {
        // Build schema from Arrow schema
        let columns: Vec<(String, ScalarType)> = record_batch.schema().fields()
            .iter()
            .filter_map(|f| {
                ScalarType::from_arrow_type(f.data_type())
                    .map(|st| (f.name().clone(), st))
            })
            .collect();
        return self.create_empty_buffer(Schema::new(columns));
    }

    let device = self.device.inner();
    let mut columns = Vec::with_capacity(record_batch.num_columns());
    let mut schema_cols = Vec::with_capacity(record_batch.num_columns());

    for (col_idx, field) in record_batch.schema().fields().iter().enumerate() {
        let array = record_batch.column(col_idx);
        let scalar_type = ScalarType::from_arrow_type(field.data_type())
            .ok_or_else(|| XlogError::Kernel(format!(
                "Unsupported Arrow type: {:?}", field.data_type()
            )))?;

        let bytes: Vec<u8> = match field.data_type() {
            DataType::Boolean => {
                let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                arr.iter().map(|v| if v.unwrap_or(false) { 1u8 } else { 0u8 }).collect()
            }
            DataType::UInt32 => {
                let arr = array.as_any().downcast_ref::<UInt32Array>().unwrap();
                arr.values().iter().flat_map(|v| v.to_le_bytes()).collect()
            }
            DataType::Int32 => {
                let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
                arr.values().iter().flat_map(|v| v.to_le_bytes()).collect()
            }
            DataType::UInt64 => {
                let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
                arr.values().iter().flat_map(|v| v.to_le_bytes()).collect()
            }
            DataType::Int64 => {
                let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
                arr.values().iter().flat_map(|v| v.to_le_bytes()).collect()
            }
            DataType::Float32 => {
                let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
                arr.values().iter().flat_map(|v| v.to_le_bytes()).collect()
            }
            DataType::Float64 => {
                let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
                arr.values().iter().flat_map(|v| v.to_le_bytes()).collect()
            }
            _ => return Err(XlogError::Kernel(format!(
                "Unsupported Arrow type: {:?}", field.data_type()
            ))),
        };

        let d_col = device
            .htod_sync_copy(&bytes)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload column: {}", e)))?;

        columns.push(d_col);
        schema_cols.push((field.name().clone(), scalar_type));
    }

    Ok(CudaBuffer::from_columns(columns, num_rows, Schema::new(schema_cols)))
}
```

**Step 4: Run test to verify it passes**

```bash
cargo test -p xlog-cuda --test arrow_tests -- --nocapture
```

Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/arrow_tests.rs
git commit -m "feat(xlog-cuda): add Arrow RecordBatch import for CuDF interop"
```

---

### Task 1.4: Add Roundtrip Test

**Files:**
- Modify: `crates/xlog-cuda/tests/arrow_tests.rs`

**Step 1: Add comprehensive roundtrip test**

Add to `arrow_tests.rs`:

```rust
#[test]
fn test_arrow_roundtrip_all_types() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create buffer with all supported types
    let schema = Schema::new(vec![
        ("bool_col".to_string(), ScalarType::Bool),
        ("u32_col".to_string(), ScalarType::U32),
        ("i32_col".to_string(), ScalarType::I32),
        ("u64_col".to_string(), ScalarType::U64),
        ("i64_col".to_string(), ScalarType::I64),
        ("f32_col".to_string(), ScalarType::F32),
        ("f64_col".to_string(), ScalarType::F64),
    ]);

    let bool_data: Vec<u8> = vec![1, 0, 1, 0];
    let u32_data: Vec<u8> = [1u32, 2, 3, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
    let i32_data: Vec<u8> = [-1i32, -2, 3, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
    let u64_data: Vec<u8> = [100u64, 200, 300, 400].iter().flat_map(|v| v.to_le_bytes()).collect();
    let i64_data: Vec<u8> = [-100i64, 200, -300, 400].iter().flat_map(|v| v.to_le_bytes()).collect();
    let f32_data: Vec<u8> = [1.5f32, 2.5, 3.5, 4.5].iter().flat_map(|v| v.to_le_bytes()).collect();
    let f64_data: Vec<u8> = [1.5f64, 2.5, 3.5, 4.5].iter().flat_map(|v| v.to_le_bytes()).collect();

    let buffer = provider.create_buffer_from_slices(
        &[&bool_data, &u32_data, &i32_data, &u64_data, &i64_data, &f32_data, &f64_data],
        schema,
    ).unwrap();

    // Export to Arrow
    let record_batch = provider.to_arrow_record_batch(&buffer).unwrap();

    // Import back
    let buffer2 = provider.from_arrow_record_batch(&record_batch).unwrap();

    // Verify roundtrip
    assert_eq!(buffer.num_rows(), buffer2.num_rows());
    assert_eq!(buffer.arity(), buffer2.arity());

    // Check u32 column
    let u32_orig = provider.download_column_u32(&buffer, 1).unwrap();
    let u32_round = provider.download_column_u32(&buffer2, 1).unwrap();
    assert_eq!(u32_orig, u32_round);

    // Check f64 column
    let f64_orig = provider.download_column_f64(&buffer, 6).unwrap();
    let f64_round = provider.download_column_f64(&buffer2, 6).unwrap();
    assert_eq!(f64_orig, f64_round);
}
```

**Step 2: Run all Arrow tests**

```bash
cargo test -p xlog-cuda --test arrow_tests -- --nocapture
```

Expected: All PASS

**Step 3: Commit**

```bash
git add crates/xlog-cuda/tests/arrow_tests.rs
git commit -m "test(xlog-cuda): add Arrow roundtrip test for all types"
```

---

## Feature 2: Multi-GPU Support

### Overview

Enable distribution of operations across multiple GPUs for larger-than-memory datasets. This requires:
- Device pool management
- Data partitioning strategies
- Cross-device synchronization

### Task 2.1: Create Device Pool

**Files:**
- Create: `crates/xlog-cuda/src/device_pool.rs`
- Modify: `crates/xlog-cuda/src/lib.rs`
- Create: `crates/xlog-cuda/tests/multi_gpu_tests.rs`

**Step 1: Write the failing test**

Create `crates/xlog-cuda/tests/multi_gpu_tests.rs`:

```rust
//! Tests for multi-GPU support

use std::sync::Arc;
use xlog_cuda::GpuDevicePool;

#[test]
fn test_device_pool_creation() {
    let device_count = cudarc::driver::CudaDevice::count().unwrap_or(0);
    if device_count == 0 {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let pool = GpuDevicePool::new(device_count as usize).unwrap();

    assert_eq!(pool.device_count(), device_count as usize);
    assert!(pool.get_device(0).is_some());
}

#[test]
fn test_device_pool_round_robin() {
    let device_count = cudarc::driver::CudaDevice::count().unwrap_or(0);
    if device_count < 1 {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let pool = GpuDevicePool::new(device_count as usize).unwrap();

    // Round-robin should cycle through devices
    let d0 = pool.next_device_idx();
    let d1 = pool.next_device_idx();

    if device_count > 1 {
        assert_ne!(d0, d1);
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p xlog-cuda --test multi_gpu_tests -- --nocapture
```

Expected: FAIL with "GpuDevicePool not found"

**Step 3: Create device_pool.rs**

Create `crates/xlog-cuda/src/device_pool.rs`:

```rust
//! Multi-GPU device pool management
//!
//! Provides a pool of CUDA devices for distributing operations.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use xlog_core::{Result, XlogError};

use crate::CudaDevice;

/// Pool of CUDA devices for multi-GPU operations
///
/// Manages multiple devices and provides round-robin scheduling.
pub struct GpuDevicePool {
    /// Available CUDA devices
    devices: Vec<Arc<CudaDevice>>,
    /// Current device index for round-robin scheduling
    current: AtomicUsize,
}

impl GpuDevicePool {
    /// Create a new device pool with the specified number of devices
    ///
    /// # Arguments
    /// * `device_count` - Number of devices to include (0 to device_count-1)
    ///
    /// # Errors
    /// Returns error if any device fails to initialize
    pub fn new(device_count: usize) -> Result<Self> {
        let available = cudarc::driver::CudaDevice::count()
            .map_err(|e| XlogError::Kernel(format!("Failed to count devices: {}", e)))?;

        if device_count > available as usize {
            return Err(XlogError::Kernel(format!(
                "Requested {} devices but only {} available",
                device_count, available
            )));
        }

        let mut devices = Vec::with_capacity(device_count);
        for ordinal in 0..device_count {
            let device = CudaDevice::new(ordinal)
                .map_err(|e| XlogError::Kernel(format!(
                    "Failed to create device {}: {}", ordinal, e
                )))?;
            devices.push(Arc::new(device));
        }

        Ok(Self {
            devices,
            current: AtomicUsize::new(0),
        })
    }

    /// Get the number of devices in the pool
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get a specific device by index
    ///
    /// Returns None if index is out of bounds
    pub fn get_device(&self, idx: usize) -> Option<&Arc<CudaDevice>> {
        self.devices.get(idx)
    }

    /// Get the next device using round-robin scheduling
    ///
    /// Returns the device index selected
    pub fn next_device_idx(&self) -> usize {
        let idx = self.current.fetch_add(1, Ordering::SeqCst);
        idx % self.devices.len()
    }

    /// Get the next device using round-robin scheduling
    pub fn next_device(&self) -> &Arc<CudaDevice> {
        let idx = self.next_device_idx();
        &self.devices[idx]
    }

    /// Synchronize all devices
    ///
    /// Waits for all pending operations on all devices to complete
    pub fn synchronize_all(&self) -> Result<()> {
        for (i, device) in self.devices.iter().enumerate() {
            device.synchronize()
                .map_err(|e| XlogError::Kernel(format!(
                    "Failed to sync device {}: {}", i, e
                )))?;
        }
        Ok(())
    }

    /// Get all devices
    pub fn devices(&self) -> &[Arc<CudaDevice>] {
        &self.devices
    }
}
```

**Step 4: Export from lib.rs**

Add to `crates/xlog-cuda/src/lib.rs`:

```rust
mod device_pool;
pub use device_pool::GpuDevicePool;
```

**Step 5: Run test to verify it passes**

```bash
cargo test -p xlog-cuda --test multi_gpu_tests -- --nocapture
```

Expected: PASS

**Step 6: Commit**

```bash
git add crates/xlog-cuda/src/device_pool.rs crates/xlog-cuda/src/lib.rs crates/xlog-cuda/tests/multi_gpu_tests.rs
git commit -m "feat(xlog-cuda): add GpuDevicePool for multi-GPU support"
```

---

### Task 2.2: Create Multi-GPU Memory Manager

**Files:**
- Create: `crates/xlog-cuda/src/multi_gpu_memory.rs`
- Modify: `crates/xlog-cuda/src/lib.rs`

**Step 1: Write the failing test**

Add to `multi_gpu_tests.rs`:

```rust
use xlog_core::MemoryBudget;
use xlog_cuda::{GpuDevicePool, MultiGpuMemoryManager};

#[test]
fn test_multi_gpu_memory_manager() {
    let device_count = cudarc::driver::CudaDevice::count().unwrap_or(0);
    if device_count == 0 {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let pool = Arc::new(GpuDevicePool::new(device_count as usize).unwrap());
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1GB per device

    let mgr = MultiGpuMemoryManager::new(pool.clone(), budget).unwrap();

    assert_eq!(mgr.device_count(), device_count as usize);

    // Allocate on specific device
    let slice = mgr.alloc_on_device::<u32>(0, 256).unwrap();
    assert_eq!(slice.len(), 256);
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p xlog-cuda --test multi_gpu_tests test_multi_gpu_memory -- --nocapture
```

Expected: FAIL with "MultiGpuMemoryManager not found"

**Step 3: Create multi_gpu_memory.rs**

Create `crates/xlog-cuda/src/multi_gpu_memory.rs`:

```rust
//! Multi-GPU memory management
//!
//! Provides memory allocation across multiple GPU devices.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use xlog_core::{MemoryBudget, Result, XlogError};

use crate::{GpuDevicePool, GpuMemoryManager};

/// Memory manager for multiple GPU devices
///
/// Maintains a separate memory budget per device and provides
/// allocation across the device pool.
pub struct MultiGpuMemoryManager {
    /// Device pool
    pool: Arc<GpuDevicePool>,
    /// Memory managers per device
    managers: Vec<Arc<GpuMemoryManager>>,
}

impl MultiGpuMemoryManager {
    /// Create a new multi-GPU memory manager
    ///
    /// # Arguments
    /// * `pool` - Device pool to manage
    /// * `budget_per_device` - Memory budget for each device
    pub fn new(pool: Arc<GpuDevicePool>, budget_per_device: MemoryBudget) -> Result<Self> {
        let mut managers = Vec::with_capacity(pool.device_count());

        for device in pool.devices() {
            let mgr = GpuMemoryManager::new(device.clone(), budget_per_device.clone());
            managers.push(Arc::new(mgr));
        }

        Ok(Self { pool, managers })
    }

    /// Get the number of devices
    pub fn device_count(&self) -> usize {
        self.pool.device_count()
    }

    /// Allocate memory on a specific device
    ///
    /// # Arguments
    /// * `device_idx` - Device index to allocate on
    /// * `len` - Number of elements to allocate
    pub fn alloc_on_device<T: cudarc::driver::DeviceRepr>(
        &self,
        device_idx: usize,
        len: usize,
    ) -> Result<CudaSlice<T>> {
        let mgr = self.managers.get(device_idx)
            .ok_or_else(|| XlogError::Kernel(format!(
                "Device {} not found", device_idx
            )))?;
        mgr.alloc::<T>(len)
    }

    /// Allocate memory on the next device (round-robin)
    ///
    /// # Returns
    /// Tuple of (device_index, allocated_slice)
    pub fn alloc_next<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<(usize, CudaSlice<T>)> {
        let device_idx = self.pool.next_device_idx();
        let slice = self.alloc_on_device::<T>(device_idx, len)?;
        Ok((device_idx, slice))
    }

    /// Get the memory manager for a specific device
    pub fn get_manager(&self, device_idx: usize) -> Option<&Arc<GpuMemoryManager>> {
        self.managers.get(device_idx)
    }

    /// Get remaining bytes on a specific device
    pub fn remaining_bytes(&self, device_idx: usize) -> u64 {
        self.managers.get(device_idx)
            .map(|m| m.remaining_bytes())
            .unwrap_or(0)
    }

    /// Get the device pool
    pub fn pool(&self) -> &Arc<GpuDevicePool> {
        &self.pool
    }
}
```

**Step 4: Export from lib.rs**

Add to `crates/xlog-cuda/src/lib.rs`:

```rust
mod multi_gpu_memory;
pub use multi_gpu_memory::MultiGpuMemoryManager;
```

**Step 5: Run test to verify it passes**

```bash
cargo test -p xlog-cuda --test multi_gpu_tests -- --nocapture
```

Expected: PASS

**Step 6: Commit**

```bash
git add crates/xlog-cuda/src/multi_gpu_memory.rs crates/xlog-cuda/src/lib.rs crates/xlog-cuda/tests/multi_gpu_tests.rs
git commit -m "feat(xlog-cuda): add MultiGpuMemoryManager for distributed allocation"
```

---

### Task 2.3: Add Distributed Join Skeleton (Documentation)

**Note:** Full distributed join implementation is complex and requires extensive testing. This task documents the architecture for future implementation.

**Files:**
- Create: `docs/architecture/multi-gpu-join.md`

**Step 1: Create architecture document**

Create `docs/architecture/multi-gpu-join.md`:

```markdown
# Multi-GPU Join Architecture

## Overview

Distributed hash join across multiple GPUs using hash-based partitioning.

## Algorithm

### Phase 1: Partition Both Tables

```
For each row in left_table:
    partition_id = hash(key) % num_devices
    send row to device[partition_id]

For each row in right_table:
    partition_id = hash(key) % num_devices
    send row to device[partition_id]
```

### Phase 2: Local Joins

```
For each device d in parallel:
    result[d] = hash_join(left_partition[d], right_partition[d])
```

### Phase 3: Gather Results

```
final_result = concatenate(result[0], result[1], ..., result[n])
```

## Implementation Notes

1. **Partitioning Kernel**: Need GPU kernel to compute partition IDs and scatter
2. **Cross-Device Copy**: Use P2P if available, else copy through host
3. **Load Balancing**: Hash partitioning may be skewed; consider sampling for better distribution
4. **Memory Management**: Each device needs memory for:
   - Input partition
   - Hash table
   - Output buffer

## API Design

```rust
impl MultiGpuKernelProvider {
    pub fn hash_join_distributed(
        &self,
        left: &DistributedBuffer,
        right: &DistributedBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
    ) -> Result<DistributedBuffer>;
}
```

## Future Work

- Implement `DistributedBuffer` type
- Add partitioning kernels
- Implement P2P copy optimization
- Add skew detection and handling
```

**Step 2: Commit**

```bash
mkdir -p docs/architecture
git add docs/architecture/multi-gpu-join.md
git commit -m "docs: add multi-GPU join architecture document"
```

---

## Feature 3: Adaptive Indexing

### Overview

Implement heat-based index selection to optimize join strategies based on query patterns.

### Task 3.1: Add Query Statistics Tracking

**Files:**
- Create: `crates/xlog-runtime/src/statistics.rs`
- Modify: `crates/xlog-runtime/src/lib.rs`
- Create: `crates/xlog-runtime/tests/statistics_tests.rs`

**Step 1: Write the failing test**

Create `crates/xlog-runtime/tests/statistics_tests.rs`:

```rust
//! Tests for query statistics tracking

use xlog_runtime::QueryStatistics;

#[test]
fn test_statistics_tracking() {
    let mut stats = QueryStatistics::new();

    // Record some accesses
    stats.record_scan("users");
    stats.record_scan("users");
    stats.record_scan("orders");
    stats.record_join("users", "orders", 0.1); // 10% selectivity

    assert_eq!(stats.scan_count("users"), 2);
    assert_eq!(stats.scan_count("orders"), 1);
    assert_eq!(stats.scan_count("nonexistent"), 0);

    let join_stats = stats.join_stats("users", "orders").unwrap();
    assert!((join_stats.avg_selectivity - 0.1).abs() < 0.01);
}

#[test]
fn test_heat_calculation() {
    let mut stats = QueryStatistics::new();

    // Hot relation: accessed many times
    for _ in 0..100 {
        stats.record_scan("hot_table");
    }

    // Cold relation: accessed once
    stats.record_scan("cold_table");

    assert!(stats.heat("hot_table") > stats.heat("cold_table"));
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p xlog-runtime --test statistics_tests -- --nocapture
```

Expected: FAIL with "QueryStatistics not found"

**Step 3: Create statistics.rs**

Create `crates/xlog-runtime/src/statistics.rs`:

```rust
//! Query statistics tracking for adaptive optimization
//!
//! Tracks access patterns and selectivity to guide index building decisions.

use std::collections::HashMap;

/// Statistics for a specific join pair
#[derive(Debug, Clone, Default)]
pub struct JoinStats {
    /// Number of times this join was executed
    pub count: u64,
    /// Total selectivity across all executions
    pub total_selectivity: f64,
    /// Average selectivity
    pub avg_selectivity: f64,
}

/// Query statistics tracker
///
/// Records relation access patterns and join selectivities
/// to inform adaptive indexing decisions.
#[derive(Debug, Default)]
pub struct QueryStatistics {
    /// Scan counts per relation
    scan_counts: HashMap<String, u64>,
    /// Join statistics per (left, right) pair
    join_stats: HashMap<(String, String), JoinStats>,
    /// Total operations tracked
    total_ops: u64,
}

impl QueryStatistics {
    /// Create new statistics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a table scan
    pub fn record_scan(&mut self, relation: &str) {
        *self.scan_counts.entry(relation.to_string()).or_insert(0) += 1;
        self.total_ops += 1;
    }

    /// Record a join operation with its selectivity
    ///
    /// # Arguments
    /// * `left` - Left relation name
    /// * `right` - Right relation name
    /// * `selectivity` - Join selectivity (output_rows / (left_rows * right_rows))
    pub fn record_join(&mut self, left: &str, right: &str, selectivity: f64) {
        let key = (left.to_string(), right.to_string());
        let stats = self.join_stats.entry(key).or_default();
        stats.count += 1;
        stats.total_selectivity += selectivity;
        stats.avg_selectivity = stats.total_selectivity / stats.count as f64;
        self.total_ops += 1;
    }

    /// Get scan count for a relation
    pub fn scan_count(&self, relation: &str) -> u64 {
        self.scan_counts.get(relation).copied().unwrap_or(0)
    }

    /// Get join statistics for a pair
    pub fn join_stats(&self, left: &str, right: &str) -> Option<&JoinStats> {
        self.join_stats.get(&(left.to_string(), right.to_string()))
    }

    /// Calculate heat score for a relation
    ///
    /// Heat = scan_count + 2 * (join_count as left) + 2 * (join_count as right)
    /// Higher heat suggests the relation would benefit from indexing
    pub fn heat(&self, relation: &str) -> u64 {
        let scan_heat = self.scan_count(relation);

        let join_heat: u64 = self.join_stats.iter()
            .filter(|((l, r), _)| l == relation || r == relation)
            .map(|(_, stats)| stats.count * 2)
            .sum();

        scan_heat + join_heat
    }

    /// Get all relations sorted by heat (hottest first)
    pub fn relations_by_heat(&self) -> Vec<(String, u64)> {
        let mut relations: Vec<_> = self.scan_counts.keys()
            .map(|r| (r.clone(), self.heat(r)))
            .collect();

        // Also include relations only seen in joins
        for (left, right) in self.join_stats.keys() {
            if !self.scan_counts.contains_key(left) {
                relations.push((left.clone(), self.heat(left)));
            }
            if !self.scan_counts.contains_key(right) {
                relations.push((right.clone(), self.heat(right)));
            }
        }

        relations.sort_by(|a, b| b.1.cmp(&a.1));
        relations.dedup_by(|a, b| a.0 == b.0);
        relations
    }

    /// Clear all statistics
    pub fn clear(&mut self) {
        self.scan_counts.clear();
        self.join_stats.clear();
        self.total_ops = 0;
    }

    /// Get total operations tracked
    pub fn total_ops(&self) -> u64 {
        self.total_ops
    }
}
```

**Step 4: Export from lib.rs**

Add to `crates/xlog-runtime/src/lib.rs`:

```rust
mod statistics;
pub use statistics::{QueryStatistics, JoinStats};
```

**Step 5: Run test to verify it passes**

```bash
cargo test -p xlog-runtime --test statistics_tests -- --nocapture
```

Expected: PASS

**Step 6: Commit**

```bash
git add crates/xlog-runtime/src/statistics.rs crates/xlog-runtime/src/lib.rs crates/xlog-runtime/tests/statistics_tests.rs
git commit -m "feat(xlog-runtime): add QueryStatistics for adaptive indexing"
```

---

### Task 3.2: Add Join Strategy Selection

**Files:**
- Modify: `crates/xlog-runtime/src/executor.rs`
- Modify: `crates/xlog-runtime/tests/statistics_tests.rs`

**Step 1: Write the failing test**

Add to `statistics_tests.rs`:

```rust
use xlog_runtime::JoinStrategy;

#[test]
fn test_join_strategy_selection() {
    let mut stats = QueryStatistics::new();

    // Small table should use nested loop
    let strategy = JoinStrategy::select(100, 10, None, &stats);
    assert_eq!(strategy, JoinStrategy::NestedLoop);

    // Large tables should use hash join
    let strategy = JoinStrategy::select(10000, 10000, None, &stats);
    assert_eq!(strategy, JoinStrategy::Hash);

    // Pre-sorted should use sort-merge
    let strategy = JoinStrategy::select(10000, 10000, Some(true), &stats);
    assert_eq!(strategy, JoinStrategy::SortMerge);
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p xlog-runtime --test statistics_tests test_join_strategy -- --nocapture
```

Expected: FAIL with "JoinStrategy not found"

**Step 3: Add JoinStrategy enum**

Add to `crates/xlog-runtime/src/statistics.rs`:

```rust
/// Join execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinStrategy {
    /// Hash join - build hash table on right, probe with left
    Hash,
    /// Nested loop join - for small right tables
    NestedLoop,
    /// Sort-merge join - for pre-sorted data
    SortMerge,
    /// Index nested loop - use existing index
    IndexNestedLoop,
}

impl JoinStrategy {
    /// Threshold for switching to nested loop (right table size)
    const NESTED_LOOP_THRESHOLD: u64 = 1000;

    /// Select optimal join strategy based on table sizes and data characteristics
    ///
    /// # Arguments
    /// * `left_rows` - Number of rows in left table
    /// * `right_rows` - Number of rows in right table
    /// * `pre_sorted` - Whether both inputs are pre-sorted on join key
    /// * `stats` - Query statistics for historical selectivity
    pub fn select(
        left_rows: u64,
        right_rows: u64,
        pre_sorted: Option<bool>,
        _stats: &QueryStatistics,
    ) -> Self {
        // If data is pre-sorted, sort-merge is efficient
        if pre_sorted == Some(true) {
            return JoinStrategy::SortMerge;
        }

        // For small right tables, nested loop avoids hash table overhead
        if right_rows < Self::NESTED_LOOP_THRESHOLD {
            return JoinStrategy::NestedLoop;
        }

        // Default to hash join
        JoinStrategy::Hash
    }
}
```

**Step 4: Export JoinStrategy**

Update export in `lib.rs`:

```rust
pub use statistics::{QueryStatistics, JoinStats, JoinStrategy};
```

**Step 5: Run test to verify it passes**

```bash
cargo test -p xlog-runtime --test statistics_tests -- --nocapture
```

Expected: PASS

**Step 6: Commit**

```bash
git add crates/xlog-runtime/src/statistics.rs crates/xlog-runtime/src/lib.rs crates/xlog-runtime/tests/statistics_tests.rs
git commit -m "feat(xlog-runtime): add JoinStrategy selection for adaptive optimization"
```

---

### Task 3.3: Integrate Statistics into Executor (Documentation)

**Note:** Full integration requires changes to the execution pipeline. This task documents the integration points.

**Files:**
- Create: `docs/architecture/adaptive-indexing.md`

**Step 1: Create architecture document**

Create `docs/architecture/adaptive-indexing.md`:

```markdown
# Adaptive Indexing Architecture

## Overview

Heat-based index selection (HISA) tracks query patterns and builds indexes
for frequently accessed relations.

## Components

### 1. QueryStatistics
- Tracks scan counts, join selectivities
- Calculates "heat" score per relation
- Location: `xlog-runtime/src/statistics.rs`

### 2. JoinStrategy
- Selects optimal join algorithm
- Options: Hash, NestedLoop, SortMerge, IndexNestedLoop
- Location: `xlog-runtime/src/statistics.rs`

### 3. Executor Integration (TODO)

Update `executor.rs` execute_join():

```rust
fn execute_join(&mut self, ...) -> Result<CudaBuffer> {
    // 1. Record statistics
    self.stats.record_scan(&left_rel);
    self.stats.record_scan(&right_rel);

    // 2. Select strategy
    let strategy = JoinStrategy::select(
        left.num_rows(),
        right.num_rows(),
        self.is_sorted(&left, &left_keys),
        &self.stats,
    );

    // 3. Execute with selected strategy
    match strategy {
        JoinStrategy::Hash => self.provider.hash_join_v2(...),
        JoinStrategy::NestedLoop => self.provider.nested_loop_join(...),
        JoinStrategy::SortMerge => self.provider.sort_merge_join(...),
        JoinStrategy::IndexNestedLoop => self.provider.index_join(...),
    }

    // 4. Record selectivity
    let selectivity = result.num_rows() as f64 /
        (left.num_rows() * right.num_rows()) as f64;
    self.stats.record_join(&left_rel, &right_rel, selectivity);

    Ok(result)
}
```

## Index Building Decisions

When to build an index:
1. Relation heat exceeds threshold
2. Memory budget allows
3. Relation is stable (not being modified)

```rust
fn maybe_build_index(&mut self, relation: &str) {
    let heat = self.stats.heat(relation);
    if heat > INDEX_HEAT_THRESHOLD && self.memory.remaining_bytes() > INDEX_MIN_MEMORY {
        self.build_hash_index(relation);
    }
}
```

## Future Work

1. Implement NestedLoop and SortMerge joins
2. Add index manager with hash index support
3. Integrate statistics into fixpoint loop
4. Add index invalidation on relation updates
```

**Step 2: Commit**

```bash
git add docs/architecture/adaptive-indexing.md
git commit -m "docs: add adaptive indexing architecture document"
```

---

## Final Summary

### Completed Tasks Checklist

**Feature 1: CuDF Integration**
- [ ] Task 1.1: Add Arrow dependency and type mapping
- [ ] Task 1.2: Create Arrow export functions
- [ ] Task 1.3: Create Arrow import functions
- [ ] Task 1.4: Add roundtrip test

**Feature 2: Multi-GPU Support**
- [ ] Task 2.1: Create device pool
- [ ] Task 2.2: Create multi-GPU memory manager
- [ ] Task 2.3: Add distributed join skeleton (docs)

**Feature 3: Adaptive Indexing**
- [ ] Task 3.1: Add query statistics tracking
- [ ] Task 3.2: Add join strategy selection
- [ ] Task 3.3: Integrate statistics into executor (docs)

### Effort Estimates

| Feature | Tasks | Effort |
|---------|-------|--------|
| CuDF Integration | 4 | 2-3 weeks |
| Multi-GPU Support | 3 (foundation) | 1-2 weeks |
| Multi-GPU (full) | +distributed ops | 3-4 weeks |
| Adaptive Indexing | 3 (foundation) | 1-2 weeks |
| Adaptive (full) | +index manager | 2-3 weeks |

**Total P3 Effort:** 8-12 weeks for full implementation

### Dependencies

```
CuDF Integration: arrow crate
Multi-GPU: cudarc P2P (optional)
Adaptive Indexing: None (internal)
```
