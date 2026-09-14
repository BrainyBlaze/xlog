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
        // SAFETY: dev.array is non-null (checked); was originally created via Box::into_raw; we are the sole owner
        unsafe {
            drop(Box::from_raw(dev.array));
        }
    }
    if !dev.schema.is_null() {
        // SAFETY: dev.schema is non-null (checked); was originally created via Box::into_raw; we are the sole owner
        unsafe {
            drop(Box::from_raw(dev.schema));
        }
    }
    if !dev.private_data.is_null() {
        // SAFETY: memory layout is guaranteed by the Arrow C Data Interface specification
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

// SAFETY: both constructors require thread-safe ownership transfer and release.
// The wrapper only reads metadata or invokes the unique release callback; raw
// producer mutation while the wrapper is live violates that import contract.
unsafe impl Send for ArrowDeviceArrayOwned {}
unsafe impl Sync for ArrowDeviceArrayOwned {}

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
    /// All referenced device ranges must be live and producer-ready at handoff.
    /// The producer must coordinate further access, and metadata plus release
    /// must be usable from any host thread for the entire transferred lifetime.
    pub unsafe fn from_raw(ptr: *mut ArrowDeviceArray) -> Self {
        Self { ptr }
    }
}

impl Drop for ArrowDeviceArrayOwned {
    fn drop(&mut self) {
        // SAFETY: ptr is non-null (checked); release was set by the Arrow producer and is valid for the array lifetime
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
    #[expect(
        clippy::new_ret_no_self,
        reason = "the Arrow C device ABI constructor returns its paired FFI schema and array handles"
    )]
    /// Adopt heap-owned Arrow FFI objects containing ready device allocations.
    ///
    /// # Safety
    /// `array` and `schema` must each come from `Box::into_raw`, be uniquely
    /// transferred, and describe live producer-ready device memory. Their
    /// buffers and callbacks must support use and release from any host thread;
    /// further producer accesses must be coordinated with the recipient.
    pub unsafe fn new(
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
    // Release the parsed FFI buffers before their producer's outer keepalive.
    _outer: Option<ArrowDeviceArrayOwned>,
}

impl ArrowDeviceImport {
    pub fn new(data: arrow::array::ArrayData) -> Self {
        Self {
            _data: data,
            _outer: None,
        }
    }

    #[cfg(any(test, feature = "arrow-device-import"))]
    pub(crate) fn from_ffi(data: arrow::array::ArrayData, outer: ArrowDeviceArrayOwned) -> Self {
        Self {
            _data: data,
            _outer: Some(outer),
        }
    }

    #[cfg(feature = "arrow-device-import")]
    pub(crate) fn data(&self) -> &arrow::array::ArrayData {
        &self._data
    }
}

impl ArrowDeviceArrayOwned {
    /// Take ownership of the underlying FFI array + schema.
    ///
    /// # Safety
    /// The returned outer owner must outlive both FFI objects and every parsed
    /// buffer derived from them. All three objects must eventually be released.
    pub unsafe fn into_ffi_parts(self) -> (i32, i32, Self, FFI_ArrowArray, FFI_ArrowSchema) {
        let dev = &mut *self.ptr;
        let device_type = dev.device_type;
        let device_id = dev.device_id;

        // Arrow's move operation leaves empty FFI objects in their original
        // locations. Their containing allocations still belong to the producer;
        // a foreign producer need not have allocated either struct with Box.
        let array = FFI_ArrowArray::from_raw(dev.array);
        let schema = FFI_ArrowSchema::from_raw(dev.schema);
        (device_type, device_id, self, array, schema)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    unsafe extern "C" fn release_outer(ptr: *mut ArrowDeviceArray) {
        let owner = unsafe { Box::from_raw(ptr) };
        // The moved FFI values are released by ArrayData, not by these empty
        // containers. The producer still owns each original container.
        drop(unsafe { Box::from_raw(owner.array) });
        drop(unsafe { Box::from_raw(owner.schema) });
        let drops = unsafe { Box::from_raw(owner.private_data.cast::<Arc<AtomicUsize>>()) };
        drops.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn ffi_import_keeps_outer_producer_alive_until_last_import_owner() {
        let drops = Arc::new(AtomicUsize::new(0));
        let data = arrow::array::ArrayData::new_empty(&arrow::datatypes::DataType::Int32);
        let (array, schema) = arrow::ffi::to_ffi(&data).unwrap();
        let raw = Box::into_raw(Box::new(ArrowDeviceArray {
            device_type: ARROW_DEVICE_CUDA,
            device_id: 0,
            array: Box::into_raw(Box::new(array)),
            schema: Box::into_raw(Box::new(schema)),
            release: Some(release_outer),
            private_data: Box::into_raw(Box::new(Arc::clone(&drops))).cast(),
        }));
        // SAFETY: this zero-length host ownership fixture transfers valid FFI
        // objects and a thread-safe producer release; it performs no GPU work.
        let device = unsafe { ArrowDeviceArrayOwned::from_raw(raw) };
        let (_, _, outer, array, schema) = unsafe { device.into_ffi_parts() };
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        let parsed = unsafe { arrow::ffi::from_ffi(array, &schema) }.unwrap();
        drop(schema);
        let imported = Arc::new(ArrowDeviceImport::from_ffi(parsed, outer));
        let alias = Arc::clone(&imported);
        drop(imported);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(alias);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
