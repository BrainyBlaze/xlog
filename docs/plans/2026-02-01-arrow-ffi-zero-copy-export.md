# Arrow FFI Zero-Copy Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a zero-copy Arrow C Data Interface export path for `CudaBuffer` that keeps all column data on device and returns a device-aware FFI handle (export-only; import remains DLPack).

**Architecture:** Introduce a device-FFI wrapper (`ArrowDeviceArray`) and a provider API that builds Arrow FFI schema/arrays backed by GPU buffers via `arrow_buffer::Buffer::from_custom_allocation` with a custom `Allocation` keeping device memory alive. Handle `Bool` by device-side bit packing and `Symbol` by exporting as `UInt32` with explicit schema metadata.

**Tech Stack:** Rust, arrow (ffi), arrow_buffer, cudarc, CUDA kernels (bit-pack), xlog-cuda tests.

---

### Task 1: Add Arrow device FFI types + provider API skeleton

**Files:**
- Create: `crates/xlog-cuda/src/arrow_device.rs`
- Modify: `crates/xlog-cuda/src/lib.rs`
- Modify: `crates/xlog-cuda/src/provider.rs`
- Test: `crates/xlog-cuda/tests/arrow_device_ffi.rs`

**Step 1: Write the failing test**

```rust
// crates/xlog-cuda/tests/arrow_device_ffi.rs
use std::sync::Arc;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
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
fn test_arrow_device_export_no_dtoh() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![
        ("id".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::I64),
    ]);
    let ids: Vec<u32> = vec![1, 2, 3, 4];
    let values: Vec<i64> = vec![10, 20, 30, 40];

    let buffer = provider
        .create_buffer_from_slices(
            &[bytemuck::cast_slice(&ids), bytemuck::cast_slice(&values)],
            schema,
        )
        .unwrap();

    provider.reset_host_transfer_stats();
    let device_rb = provider.to_arrow_device_record_batch(&buffer).unwrap();

    let stats = provider.host_transfer_stats();
    assert_eq!(stats.dtoh_bytes, 0, "device export performed DTOH");

    // sanity: FFI pointers are non-null
    unsafe {
        let ptr = device_rb.as_ptr();
        assert!(!ptr.is_null());
        let arr = (*ptr).array as *mut FFI_ArrowArray;
        let schema = (*ptr).schema as *mut FFI_ArrowSchema;
        assert!(!arr.is_null());
        assert!(!schema.is_null());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test arrow_device_ffi -q`
Expected: FAIL (missing `to_arrow_device_record_batch` and FFI types)

**Step 3: Write minimal implementation**

```rust
// crates/xlog-cuda/src/arrow_device.rs
use std::ffi::c_void;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};

#[repr(C)]
pub struct ArrowDeviceArray {
    pub device_type: i32,
    pub device_id: i32,
    pub array: *mut FFI_ArrowArray,
    pub schema: *mut FFI_ArrowSchema,
    pub release: Option<unsafe extern "C" fn(*mut ArrowDeviceArray)>,
    pub private_data: *mut c_void,
}

pub struct ArrowDeviceArrayOwned {
    ptr: *mut ArrowDeviceArray,
}

impl ArrowDeviceArrayOwned {
    pub fn as_ptr(&self) -> *mut ArrowDeviceArray {
        self.ptr
    }

    pub fn into_raw(self) -> *mut ArrowDeviceArray {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }

    pub unsafe fn from_raw(ptr: *mut ArrowDeviceArray) -> Self {
        Self { ptr }
    }
}

impl Drop for ArrowDeviceArrayOwned {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                if let Some(release) = (*self.ptr).release {
                    release(self.ptr);
                }
            }
        }
    }
}
```

```rust
// crates/xlog-cuda/src/lib.rs
pub mod arrow_device;
pub use arrow_device::{ArrowDeviceArray, ArrowDeviceArrayOwned};
```

```rust
// crates/xlog-cuda/src/provider.rs
pub fn to_arrow_device_record_batch(
    &self,
    _buffer: &CudaBuffer,
) -> Result<crate::arrow_device::ArrowDeviceArrayOwned> {
    Err(XlogError::Kernel("arrow device export not implemented".to_string()))
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test arrow_device_ffi -q`
Expected: FAIL (still not implemented) — this is only the API stub; real behavior comes in later tasks.

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/arrow_device.rs crates/xlog-cuda/src/lib.rs \
  crates/xlog-cuda/src/provider.rs crates/xlog-cuda/tests/arrow_device_ffi.rs

git commit -m "feat(cuda): add Arrow device FFI types and export API"
```

---

### Task 2: Device-backed Arrow buffer allocation (numeric types)

**Files:**
- Modify: `crates/xlog-cuda/src/provider.rs`
- Modify: `crates/xlog-cuda/src/arrow_device.rs`
- Test: `crates/xlog-cuda/tests/arrow_device_ffi.rs`

**Step 1: Write the failing test**

```rust
// append to crates/xlog-cuda/tests/arrow_device_ffi.rs
#[test]
fn test_arrow_device_export_schema_and_buffers() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![
        ("id".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::I64),
    ]);
    let ids: Vec<u32> = vec![1, 2, 3, 4];
    let values: Vec<i64> = vec![10, 20, 30, 40];

    let buffer = provider
        .create_buffer_from_slices(
            &[bytemuck::cast_slice(&ids), bytemuck::cast_slice(&values)],
            schema,
        )
        .unwrap();

    let device_rb = provider.to_arrow_device_record_batch(&buffer).unwrap();
    unsafe {
        let dev_ptr = device_rb.as_ptr();
        let schema_ptr = (*dev_ptr).schema;
        let array_ptr = (*dev_ptr).array;
        assert!(!schema_ptr.is_null());
        assert!(!array_ptr.is_null());

        // Struct array with 2 children
        let arr = &*array_ptr;
        assert_eq!(arr.n_children, 2);
        assert_eq!(arr.length, 4);
        assert!(arr.buffers.is_null() == false);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test arrow_device_ffi -q`
Expected: FAIL (schema/array not populated, not exporting device buffers)

**Step 3: Write minimal implementation**

```rust
// crates/xlog-cuda/src/arrow_device.rs (add helper structs)
use std::sync::Arc;
use arrow_buffer::alloc::Allocation;

pub struct ArrowCudaAllocation {
    _keepalive: Arc<crate::memory::CudaBuffer>,
}

impl Allocation for ArrowCudaAllocation {}
```

```rust
// crates/xlog-cuda/src/provider.rs (inside to_arrow_device_record_batch)
// Build arrow::array::ArrayData per column using Buffer::from_custom_allocation
// Use arrow::ffi::to_ffi to get FFI_ArrowArray + FFI_ArrowSchema
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test arrow_device_ffi -q`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/arrow_device.rs crates/xlog-cuda/src/provider.rs \
  crates/xlog-cuda/tests/arrow_device_ffi.rs

git commit -m "feat(cuda): export numeric columns via Arrow C Data Interface"
```

---

### Task 3: Bool bit-packing kernel + export integration

**Files:**
- Modify: `kernels/pack.cu`
- Modify: `crates/xlog-cuda/src/provider.rs`
- Modify: `crates/xlog-cuda/src/provider.rs` (kernel list)
- Modify: `crates/xlog-cuda/src/provider.rs` (load function)
- Test: `crates/xlog-cuda/tests/arrow_device_ffi.rs`

**Step 1: Write the failing test**

```rust
// append to crates/xlog-cuda/tests/arrow_device_ffi.rs
#[test]
fn test_arrow_device_export_bool_bitpacked() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("flag".to_string(), ScalarType::Bool)]);
    let flags: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0, 1];

    let buffer = provider.create_buffer_from_slices(&[&flags], schema).unwrap();
    let device_rb = provider.to_arrow_device_record_batch(&buffer).unwrap();

    // In tests only: download packed buffer and verify bits (LSB-first)
    let packed = provider.read_arrow_device_bool_packed(&device_rb).unwrap();
    assert_eq!(packed.len(), 2);
    assert_eq!(packed[0] & 0b0000_0001, 1);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test arrow_device_ffi -q`
Expected: FAIL (missing bool pack + test helper)

**Step 3: Write minimal implementation**

```cuda
// kernels/pack.cu (add)
extern "C" __global__ void pack_bools_to_bitmap(
    const uint8_t* __restrict__ input,
    uint32_t num_rows,
    uint8_t* __restrict__ out_bitmap
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t byte_idx = i >> 3;
    if (byte_idx >= ((num_rows + 7) >> 3)) return;

    uint8_t byte = 0;
    uint32_t base = byte_idx << 3;
    #pragma unroll
    for (uint32_t b = 0; b < 8; b++) {
        uint32_t idx = base + b;
        if (idx < num_rows) {
            uint8_t v = input[idx] != 0 ? 1 : 0;
            byte |= (v << b);
        }
    }
    out_bitmap[byte_idx] = byte;
}
```

```rust
// crates/xlog-cuda/src/provider.rs
pub mod pack_kernels {
    pub const PACK_BOOLS_TO_BITMAP: &str = "pack_bools_to_bitmap";
}
```

```rust
// crates/xlog-cuda/src/provider.rs (load_ptx list)
pack_kernels::PACK_BOOLS_TO_BITMAP,
```

```rust
// crates/xlog-cuda/src/provider.rs (to_arrow_device_record_batch)
// For Bool: allocate device bitmap buffer and launch pack_bools_to_bitmap
```

```rust
// crates/xlog-cuda/src/provider.rs (test helper)
#[cfg(test)]
fn read_arrow_device_bool_packed(&self, rb: &ArrowDeviceArrayOwned) -> Result<Vec<u8>> { /* dtoh in tests only */ }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test arrow_device_ffi -q`
Expected: PASS

**Step 5: Commit**

```bash
git add kernels/pack.cu crates/xlog-cuda/src/provider.rs crates/xlog-cuda/tests/arrow_device_ffi.rs

git commit -m "feat(cuda): bit-pack bool columns for Arrow device export"
```

---

### Task 4: Symbol export metadata + safety checks

**Files:**
- Modify: `crates/xlog-cuda/src/provider.rs`
- Test: `crates/xlog-cuda/tests/arrow_device_ffi.rs`

**Step 1: Write the failing test**

```rust
// append to crates/xlog-cuda/tests/arrow_device_ffi.rs
#[test]
fn test_arrow_device_export_symbol_metadata() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("sym".to_string(), ScalarType::Symbol)]);
    let ids: Vec<u32> = vec![1, 2, 3];
    let buffer = provider.create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema).unwrap();

    let device_rb = provider.to_arrow_device_record_batch(&buffer).unwrap();
    let meta = provider.read_arrow_device_schema_metadata(&device_rb).unwrap();
    assert!(meta.contains("xlog.symbol=true"));
    assert!(meta.contains("xlog.symbol_encoding=u32"));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test arrow_device_ffi -q`
Expected: FAIL (missing metadata export)

**Step 3: Write minimal implementation**

```rust
// crates/xlog-cuda/src/provider.rs
// When building Arrow schema, set metadata on symbol fields:
//   xlog.symbol=true
//   xlog.symbol_encoding=u32
// Use arrow_schema::Schema::with_metadata or per-field metadata.

#[cfg(test)]
fn read_arrow_device_schema_metadata(&self, rb: &ArrowDeviceArrayOwned) -> Result<String> { /* test-only dtoh */ }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test arrow_device_ffi -q`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider.rs crates/xlog-cuda/tests/arrow_device_ffi.rs

git commit -m "feat(cuda): export symbol metadata in Arrow device schema"
```

---

### Task 5: Documentation + guardrails

**Files:**
- Modify: `docs/architecture/cudf-interop.md`
- Modify: `docs/architecture/cuda-certification.md` (if export needs certification note)
- Modify: `docs/ROADMAP.md`
- Modify: `CHANGELOG.md`

**Step 1: Write the failing doc check**

```bash
rg -n "Arrow device" docs/architecture/cudf-interop.md
```
Expected: no mention of Arrow C Data Interface export

**Step 2: Update documentation**

Add:
- New API name and export-only contract
- Supported types (U32/U64/I32/I64/F32/F64/Bool bitpacked/Symbol metadata)
- Symbol metadata keys

**Step 3: Verify docs updated**

Run: `rg -n "Arrow C Data Interface" docs/architecture/cudf-interop.md`
Expected: matches new section

**Step 4: Commit**

```bash
git add docs/architecture/cudf-interop.md docs/architecture/cuda-certification.md docs/ROADMAP.md CHANGELOG.md

git commit -m "docs: add Arrow device FFI export support"
```

---

## Execution options

Plan complete and saved to `docs/plans/2026-02-01-arrow-ffi-zero-copy-export.md`. Two execution options:

1. Subagent-Driven (this session) — I dispatch a fresh subagent per task, review between tasks
2. Parallel Session (separate) — Open a new session using superpowers:executing-plans and run tasks sequentially

Which approach?
