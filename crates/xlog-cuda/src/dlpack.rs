//! DLPack interop for zero-copy GPU exchange.
//!
//! This module provides a minimal DLPack implementation for exporting XLOG GPU
//! buffers to other ecosystems (e.g., Python cuDF) without device↔host copies.

use std::ffi::c_void;
use std::sync::Arc;

use xlog_core::{Result, ScalarType, Schema, XlogError};

use crate::memory::{validate_logical_row_count, CudaBuffer, CudaColumn};
use crate::provider::CudaKernelProvider;
use crate::CudaDevice;

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

fn dl_dtype_to_scalar(dtype: DLDataType) -> Result<ScalarType> {
    if dtype.lanes != 1 {
        return Err(XlogError::Kernel(format!(
            "Unsupported DLPack dtype lanes {} (expected 1)",
            dtype.lanes
        )));
    }
    match (dtype.code, dtype.bits) {
        (K_DLUINT, 32) => Ok(ScalarType::U32),
        (K_DLUINT, 64) => Ok(ScalarType::U64),
        (K_DLINT, 32) => Ok(ScalarType::I32),
        (K_DLINT, 64) => Ok(ScalarType::I64),
        (K_DLFLOAT, 32) => Ok(ScalarType::F32),
        (K_DLFLOAT, 64) => Ok(ScalarType::F64),
        // XLOG represents bool as one byte per row today (not bitpacked).
        (K_DLBOOL, 8) => Ok(ScalarType::Bool),
        _ => Err(XlogError::Kernel(format!(
            "Unsupported DLPack dtype code={} bits={} lanes={}",
            dtype.code, dtype.bits, dtype.lanes
        ))),
    }
}

/// Owned DLPack tensor handle.
///
/// Dropping this value will call the DLPack deleter and free the underlying GPU memory.
pub struct DlpackManagedTensor {
    ptr: *mut DLManagedTensor,
}

// SAFETY: DLPack tensors are GPU device pointers with a deleter callback.
// GPU memory is accessible from any CPU thread, and the deleter is a plain
// function pointer with no thread affinity. The pointer is never dereferenced
// concurrently — it is only read during column access and freed on drop.
unsafe impl Send for DlpackManagedTensor {}
unsafe impl Sync for DlpackManagedTensor {}

impl DlpackManagedTensor {
    /// Construct an owned DLPack tensor from a raw pointer.
    ///
    /// # Safety
    /// `ptr` must be a valid `DLManagedTensor*` obtained from a DLPack producer, and ownership
    /// must be transferred to the caller (the returned value will call the DLPack deleter on drop).
    pub unsafe fn from_raw(ptr: *mut DLManagedTensor) -> Self {
        Self { ptr }
    }

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

unsafe fn dlpack_tensor_info(
    provider: &CudaKernelProvider,
    tensor: &DlpackManagedTensor,
) -> Result<(u64, ScalarType, cudarc::driver::sys::CUdeviceptr, usize)> {
    let ptr = tensor.as_ptr();
    if ptr.is_null() {
        return Err(XlogError::Kernel(
            "Null DLManagedTensor pointer".to_string(),
        ));
    }

    let dl = unsafe { &(*ptr).dl_tensor };

    if dl.device.device_type != K_DLCUDA {
        return Err(XlogError::Kernel(format!(
            "Unsupported DLPack device type {} (expected CUDA)",
            dl.device.device_type
        )));
    }
    if dl.device.device_id != provider.device().ordinal() as i32 {
        return Err(XlogError::Kernel(format!(
            "DLPack tensor device_id {} does not match provider device_id {}",
            dl.device.device_id,
            provider.device().ordinal()
        )));
    }

    if dl.ndim != 1 {
        return Err(XlogError::Kernel(format!(
            "Unsupported DLPack ndim {} (expected 1)",
            dl.ndim
        )));
    }
    if dl.shape.is_null() {
        return Err(XlogError::Kernel("DLPack tensor shape is null".to_string()));
    }
    if !dl.strides.is_null() {
        let stride0 = unsafe { *dl.strides };
        if stride0 != 1 {
            return Err(XlogError::Kernel(format!(
                "Non-contiguous DLPack tensor stride {} (expected 1)",
                stride0
            )));
        }
    }

    let shape0 = unsafe { *dl.shape };
    if shape0 < 0 {
        return Err(XlogError::Kernel(format!(
            "Negative DLPack tensor shape {}",
            shape0
        )));
    }
    let num_rows = shape0 as u64;

    let scalar = dl_dtype_to_scalar(dl.dtype)?;
    let elem_size = scalar.size_bytes();
    if dl.byte_offset % (elem_size as u64) != 0 {
        return Err(XlogError::Kernel(format!(
            "DLPack byte_offset {} is not aligned to element size {}",
            dl.byte_offset, elem_size
        )));
    }

    if dl.data.is_null() && num_rows > 0 {
        return Err(XlogError::Kernel(
            "DLPack tensor data pointer is null".to_string(),
        ));
    }

    let base = dl.data as usize;
    let ptr_with_offset = base
        .checked_add(dl.byte_offset as usize)
        .ok_or_else(|| XlogError::Kernel("DLPack data pointer overflow".to_string()))?;

    if ptr_with_offset % elem_size != 0 {
        return Err(XlogError::Kernel(
            "DLPack tensor data is not properly aligned".to_string(),
        ));
    }

    let len_bytes = usize::try_from(num_rows)
        .ok()
        .and_then(|n| n.checked_mul(elem_size))
        .ok_or_else(|| XlogError::Kernel("DLPack tensor length overflow".to_string()))?;

    Ok((num_rows, scalar, ptr_with_offset as u64, len_bytes))
}

fn dlpack_logical_row_count(device: &Arc<CudaDevice>, buffer: &CudaBuffer) -> Result<usize> {
    if let Some(cached_rows) = buffer.cached_row_count() {
        return validate_logical_row_count(buffer.num_rows(), cached_rows as usize);
    }

    let mut host_rows = [0u32];
    device
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
        .map_err(|e| XlogError::Kernel(format!("Failed to read row count: {}", e)))?;
    buffer.set_cached_row_count_if_unset(host_rows[0]);
    validate_logical_row_count(buffer.num_rows(), host_rows[0] as usize)
}

/// A table-like wrapper that can export individual columns as DLPack tensors without copies.
///
/// The underlying `CudaBuffer` is reference-counted so multiple DLPack exports can share it.
pub struct DlpackTable {
    buffer: Arc<CudaBuffer>,
    cuda_device: Arc<CudaDevice>,
    device: DLDevice,
}

impl DlpackTable {
    pub fn column(&self, col_idx: usize) -> Result<DlpackManagedTensor> {
        let logical_rows = dlpack_logical_row_count(&self.cuda_device, &self.buffer)?;
        let dtype =
            self.buffer.schema().column_type(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Column index {} out of bounds", col_idx))
            })?;

        let col = self
            .buffer
            .columns
            .get(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

        let device_ptr = *col.device_ptr() as usize as *mut c_void;

        let mut ctx = Box::new(DlpackCtx {
            buffer: self.buffer.clone(),
            shape: vec![logical_rows as i64].into_boxed_slice(),
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
            cuda_device: Arc::clone(self.device()),
            device: DLDevice {
                device_type: K_DLCUDA,
                device_id: self.device().ordinal() as i32,
            },
        }
    }

    /// Import one DLPack tensor per column as a zero-copy `CudaBuffer`.
    ///
    /// The returned buffer owns the DLPack tensors and will call their deleters on drop.
    pub fn from_dlpack_tensors(&self, tensors: Vec<DlpackManagedTensor>) -> Result<CudaBuffer> {
        if tensors.is_empty() {
            return self.create_empty_buffer(Schema::new(vec![]));
        }

        let mut columns = Vec::with_capacity(tensors.len());
        let mut schema_cols = Vec::with_capacity(tensors.len());
        let mut num_rows: Option<u64> = None;

        for (i, tensor) in tensors.into_iter().enumerate() {
            let (rows, ty, ptr, len_bytes) = unsafe { dlpack_tensor_info(self, &tensor)? };
            if let Some(n) = num_rows {
                if rows != n {
                    return Err(XlogError::Kernel(
                        "DLPack column row counts do not match".to_string(),
                    ));
                }
            } else {
                num_rows = Some(rows);
            }

            schema_cols.push((format!("col_{}", i), ty));
            columns.push(CudaColumn::dlpack(
                ptr,
                len_bytes,
                self.device().inner().stream().clone(),
                tensor,
            ));
        }

        let schema = xlog_core::Schema::new(schema_cols);
        self.buffer_from_columns(columns, num_rows.unwrap_or(0), schema)
    }

    /// Import DLPack column tensors with an explicit schema (type-checked).
    pub fn from_dlpack_tensors_with_schema(
        &self,
        schema: xlog_core::Schema,
        tensors: Vec<DlpackManagedTensor>,
    ) -> Result<CudaBuffer> {
        if schema.arity() != tensors.len() {
            return Err(XlogError::Kernel(format!(
                "Schema arity {} does not match tensor count {}",
                schema.arity(),
                tensors.len()
            )));
        }

        if tensors.is_empty() {
            return self.create_empty_buffer(schema);
        }

        let mut columns = Vec::with_capacity(tensors.len());
        let mut num_rows: Option<u64> = None;

        for (i, tensor) in tensors.into_iter().enumerate() {
            let (rows, ty, ptr, len_bytes) = unsafe { dlpack_tensor_info(self, &tensor)? };
            let expected = schema.column_type(i).ok_or_else(|| {
                XlogError::Kernel(format!("Missing schema type for column {}", i))
            })?;
            if !expected.dlpack_compatible(ty) {
                return Err(XlogError::Kernel(format!(
                    "DLPack column {} dtype {:?} does not match schema {:?}",
                    i, ty, expected
                )));
            }

            if let Some(n) = num_rows {
                if rows != n {
                    return Err(XlogError::Kernel(
                        "DLPack column row counts do not match".to_string(),
                    ));
                }
            } else {
                num_rows = Some(rows);
            }

            columns.push(CudaColumn::dlpack(
                ptr,
                len_bytes,
                self.device().inner().stream().clone(),
                tensor,
            ));
        }

        self.buffer_from_columns(columns, num_rows.unwrap_or(0), schema)
    }
}
