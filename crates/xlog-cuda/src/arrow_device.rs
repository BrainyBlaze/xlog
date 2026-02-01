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
            drop(Box::from_raw(dev.private_data));
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
