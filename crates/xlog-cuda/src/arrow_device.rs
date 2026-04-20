use std::ffi::c_void;
use std::sync::Arc;

use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};

use crate::memory::{CudaBuffer, TrackedCudaSlice};

pub const ARROW_DEVICE_CUDA: i32 = 2;

pub(crate) struct ArrowCudaAllocation {
    _buffer: Arc<CudaBuffer>,
    _extra: Vec<TrackedCudaSlice<u8>>,
}

impl ArrowCudaAllocation {
    pub(crate) fn new(buffer: Arc<CudaBuffer>, extra: Vec<TrackedCudaSlice<u8>>) -> Self {
        Self {
            _buffer: buffer,
            _extra: extra,
        }
    }
}

// SAFETY: ArrowCudaAllocation is used only as a keepalive handle for GPU buffers.
// Dropping CUDA allocations is thread-safe, and no device memory is accessed.
unsafe impl Send for ArrowCudaAllocation {}
unsafe impl Sync for ArrowCudaAllocation {}

#[repr(C)]
pub struct ArrowDeviceArray {
    pub device_type: i32,
    pub device_id: i32,
    pub array: *mut FFI_ArrowArray,
    pub schema: *mut FFI_ArrowSchema,
    pub release: Option<unsafe extern "C" fn(*mut ArrowDeviceArray)>,
    pub private_data: *mut c_void,
}

unsafe extern "C" fn release_arrow_device_array(ptr: *mut ArrowDeviceArray) {
    if ptr.is_null() {
        return;
    }
    let dev = &mut *ptr;
    if !dev.array.is_null() {
        unsafe {
            drop(Box::from_raw(dev.array));
        }
    }
    if !dev.schema.is_null() {
        unsafe {
            drop(Box::from_raw(dev.schema));
        }
    }
    if !dev.private_data.is_null() {
        unsafe {
            drop(Box::from_raw(
                dev.private_data.cast::<ArrowCudaAllocation>(),
            ));
        }
    }
    dev.release = None;
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

    /// Rebuild an owned wrapper from a raw `ArrowDeviceArray` pointer.
    ///
    /// # Safety
    /// `ptr` must be a valid, uniquely owned pointer produced by
    /// `ArrowDeviceArrayOwned::into_raw` or an equivalent allocation that
    /// transfers ownership of the underlying `ArrowDeviceArray`.
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

impl ArrowDeviceArray {
    pub fn new(
        device_type: i32,
        device_id: i32,
        array: *mut FFI_ArrowArray,
        schema: *mut FFI_ArrowSchema,
    ) -> ArrowDeviceArrayOwned {
        let dev = ArrowDeviceArray {
            device_type,
            device_id,
            array,
            schema,
            release: Some(release_arrow_device_array),
            private_data: std::ptr::null_mut(),
        };
        ArrowDeviceArrayOwned {
            ptr: Box::into_raw(Box::new(dev)),
        }
    }
}

/// Keepalive wrapper for imported Arrow device arrays.
///
/// Holds the Arrow ArrayData so the FFI buffers remain alive until all
/// device-backed columns are dropped.
pub struct ArrowDeviceImport {
    _data: arrow::array::ArrayData,
}

impl ArrowDeviceImport {
    pub fn new(data: arrow::array::ArrayData) -> Self {
        Self { _data: data }
    }
}

impl ArrowDeviceArrayOwned {
    /// Take ownership of the underlying FFI array + schema.
    ///
    /// # Safety
    /// The caller must ensure the returned FFI objects are eventually released.
    pub unsafe fn into_ffi_parts(self) -> (i32, i32, FFI_ArrowArray, FFI_ArrowSchema) {
        let ptr = self.into_raw();
        let dev = &mut *ptr;
        let device_type = dev.device_type;
        let device_id = dev.device_id;

        let array_ptr = dev.array;
        let schema_ptr = dev.schema;
        dev.array = std::ptr::null_mut();
        dev.schema = std::ptr::null_mut();

        if let Some(release) = dev.release {
            release(ptr);
        } else {
            drop(Box::from_raw(ptr));
        }

        // Take ownership of the exported FFI structs and free the heap allocations
        // that held them. The returned values will run their own `Drop` (calling
        // the Arrow C-Data release callbacks) when eventually dropped.
        let array = *Box::from_raw(array_ptr);
        let schema = *Box::from_raw(schema_ptr);
        (device_type, device_id, array, schema)
    }
}
