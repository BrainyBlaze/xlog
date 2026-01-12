//! DLPack interop for zero-copy GPU exchange.
//!
//! This module provides a minimal DLPack implementation for exporting XLOG GPU
//! buffers to other ecosystems (e.g., Python cuDF) without device↔host copies.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::DevicePtr;
use xlog_core::{Result, ScalarType, XlogError};

use crate::memory::CudaBuffer;
use crate::provider::CudaKernelProvider;

pub type DLDeviceType = i32;

pub const K_DLCPU: DLDeviceType = 1;
pub const K_DLCUDA: DLDeviceType = 2;

pub type DLDataTypeCode = u8;
pub const K_DLINT: DLDataTypeCode = 0;
pub const K_DLUINT: DLDataTypeCode = 1;
pub const K_DLFLOAT: DLDataTypeCode = 2;
pub const K_DLBOOL: DLDataTypeCode = 6;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDataType {
    pub code: DLDataTypeCode,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
#[derive(Debug)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

pub type DLDeleter = Option<unsafe extern "C" fn(*mut DLManagedTensor)>;

#[repr(C)]
#[derive(Debug)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: DLDeleter,
}

#[allow(dead_code)]
struct DlpackCtx {
    buffer: Arc<CudaBuffer>,
    shape: Box<[i64]>,
}

unsafe extern "C" fn dlpack_deleter(ptr: *mut DLManagedTensor) {
    if ptr.is_null() {
        return;
    }
    let ctx_ptr = unsafe { (*ptr).manager_ctx as *mut DlpackCtx };
    if !ctx_ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ctx_ptr));
        }
    }
    unsafe {
        drop(Box::from_raw(ptr));
    }
}

fn scalar_to_dl_dtype(ty: ScalarType) -> DLDataType {
    match ty {
        ScalarType::U32 | ScalarType::Symbol => DLDataType {
            code: K_DLUINT,
            bits: 32,
            lanes: 1,
        },
        ScalarType::U64 => DLDataType {
            code: K_DLUINT,
            bits: 64,
            lanes: 1,
        },
        ScalarType::I32 => DLDataType {
            code: K_DLINT,
            bits: 32,
            lanes: 1,
        },
        ScalarType::I64 => DLDataType {
            code: K_DLINT,
            bits: 64,
            lanes: 1,
        },
        ScalarType::F32 => DLDataType {
            code: K_DLFLOAT,
            bits: 32,
            lanes: 1,
        },
        ScalarType::F64 => DLDataType {
            code: K_DLFLOAT,
            bits: 64,
            lanes: 1,
        },
        ScalarType::Bool => DLDataType {
            code: K_DLBOOL,
            bits: 8,
            lanes: 1,
        },
    }
}

/// Owned DLPack tensor handle.
///
/// Dropping this value will call the DLPack deleter and free the underlying GPU memory.
pub struct DlpackManagedTensor {
    ptr: *mut DLManagedTensor,
}

impl DlpackManagedTensor {
    pub fn as_ptr(&self) -> *mut DLManagedTensor {
        self.ptr
    }

    pub fn into_raw(self) -> *mut DLManagedTensor {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }
}

impl Drop for DlpackManagedTensor {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                if let Some(deleter) = (*self.ptr).deleter {
                    deleter(self.ptr);
                }
            }
        }
    }
}

/// A table-like wrapper that can export individual columns as DLPack tensors without copies.
///
/// The underlying `CudaBuffer` is reference-counted so multiple DLPack exports can share it.
pub struct DlpackTable {
    buffer: Arc<CudaBuffer>,
    device: DLDevice,
}

impl DlpackTable {
    pub fn column(&self, col_idx: usize) -> Result<DlpackManagedTensor> {
        let dtype = self
            .buffer
            .schema()
            .column_type(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column index {} out of bounds", col_idx)))?;

        let col = self
            .buffer
            .columns
            .get(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

        let device_ptr = *DevicePtr::device_ptr(col) as usize as *mut c_void;

        let mut ctx = Box::new(DlpackCtx {
            buffer: self.buffer.clone(),
            shape: vec![self.buffer.num_rows() as i64].into_boxed_slice(),
        });
        let shape_ptr = ctx.shape.as_mut_ptr();

        let dl_tensor = DLTensor {
            data: device_ptr,
            device: self.device,
            ndim: 1,
            dtype: scalar_to_dl_dtype(dtype),
            shape: shape_ptr,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };

        let managed = Box::new(DLManagedTensor {
            dl_tensor,
            manager_ctx: Box::into_raw(ctx) as *mut c_void,
            deleter: Some(dlpack_deleter),
        });

        Ok(DlpackManagedTensor {
            ptr: Box::into_raw(managed),
        })
    }
}

impl CudaKernelProvider {
    /// Convert a `CudaBuffer` into a DLPack-exportable table without device↔host copies.
    ///
    /// Export each column with `DlpackTable::column(...)`.
    pub fn to_dlpack_table(&self, buffer: CudaBuffer) -> DlpackTable {
        DlpackTable {
            buffer: Arc::new(buffer),
            device: DLDevice {
                device_type: K_DLCUDA,
                device_id: self.device().ordinal() as i32,
            },
        }
    }
}
