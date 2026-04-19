//! I/O operations: multi-column buffer creation, Arrow interop.

use std::sync::Arc;

use crate::{LaunchAsync, LaunchConfig};
use xlog_core::{Result, ScalarType, Schema, XlogError};

use super::{d4_kernels, pack_kernels, D4_MODULE, PACK_MODULE};
use crate::CudaBuffer;

impl super::CudaKernelProvider {
    // ============== Buffer Helper Methods ==============

    /// Create a CudaBuffer from multiple u32 column slices
    ///
    /// # Arguments
    /// * `columns` - Slice of column data slices (each column as &[u32])
    /// * `schema` - The schema for the buffer
    ///
    /// # Returns
    /// A new CudaBuffer containing all columns
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if upload fails or columns have mismatched lengths
    pub fn create_buffer_from_u32_columns(
        &self,
        columns: &[&[u32]],
        schema: Schema,
    ) -> Result<CudaBuffer> {
        if columns.is_empty() {
            return self.create_empty_buffer(schema);
        }

        let num_rows = columns[0].len();
        for (i, col) in columns.iter().enumerate() {
            if col.len() != num_rows {
                return Err(XlogError::Kernel(format!(
                    "Column {} has {} rows but expected {}",
                    i,
                    col.len(),
                    num_rows
                )));
            }
        }

        let mut cuda_columns = Vec::with_capacity(columns.len());
        for col_data in columns {
            let bytes: Vec<u8> = col_data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut col = self.memory.alloc::<u8>(bytes.len())?;
            self.device
                .inner()
                .htod_sync_copy_into(&bytes, &mut col)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload column: {}", e)))?;
            cuda_columns.push(col.into());
        }

        self.buffer_from_columns(cuda_columns, num_rows as u64, schema)
    }

    /// Create a buffer from multiple column slices (raw bytes)
    ///
    /// This is a generic version that works with any column type by accepting
    /// raw byte slices. Each slice should contain the column data in little-endian
    /// format with the correct size for the column's type.
    ///
    /// # Arguments
    /// * `slices` - Slice of raw byte slices, one per column
    /// * `schema` - The schema for the buffer
    ///
    /// # Returns
    /// A new CudaBuffer containing all columns
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Number of slices doesn't match schema arity
    /// - Upload fails
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

        let first_col_size = schema.column_type(0).map(|t| t.size_bytes()).unwrap_or(4);
        let num_rows = slices[0].len() / first_col_size;

        // Validate that all columns have consistent row counts
        for (i, slice) in slices.iter().enumerate() {
            let col_size = schema.column_type(i).map(|t| t.size_bytes()).unwrap_or(4);
            let col_rows = slice.len() / col_size;
            if col_rows != num_rows {
                return Err(XlogError::Kernel(format!(
                    "Column {} has {} rows but expected {} rows (based on first column)",
                    i, col_rows, num_rows
                )));
            }
            // Also verify the slice length is exactly divisible by the type size
            if slice.len() % col_size != 0 {
                return Err(XlogError::Kernel(format!(
                    "Column {} slice length {} is not divisible by type size {}",
                    i,
                    slice.len(),
                    col_size
                )));
            }
        }

        let mut columns = Vec::with_capacity(slices.len());

        for (i, slice) in slices.iter().enumerate() {
            let mut col = self.memory.alloc::<u8>(slice.len())?;
            self.device
                .inner()
                .htod_sync_copy_into(*slice, &mut col)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload column {}: {}", i, e)))?;
            columns.push(col.into());
        }

        self.buffer_from_columns(columns, num_rows as u64, schema)
    }

    /// Export CudaBuffer to Arrow C Data Interface (device-resident).
    ///
    /// This is a zero-copy export: column buffers remain on device, and the
    /// returned ArrowDeviceArray describes CUDA-resident memory.
    ///
    /// The export requires that the device row count matches the host row_cap
    pub fn to_arrow_device_record_batch(
        &self,
        buffer: CudaBuffer,
    ) -> Result<crate::arrow_device::ArrowDeviceArrayOwned> {
        use arrow::array::ArrayData;
        use arrow::datatypes::{DataType, Field};
        use arrow::ffi::to_ffi;

        use crate::arrow_device::{ArrowDeviceArray, ARROW_DEVICE_CUDA};

        let buffer = Arc::new(buffer);
        let row_cap = buffer.num_rows();
        let num_rows_u32 = u32::try_from(row_cap).map_err(|_| {
            XlogError::Kernel(format!(
                "Arrow device export supports at most {} rows, got {}",
                u32::MAX,
                row_cap
            ))
        })?;

        // GPU-side assertion: device row count must match row_cap (no host reads).
        let assert_fn = self
            .device
            .inner()
            .get_func(D4_MODULE, d4_kernels::D4_ASSERT_U32_EQ)
            .ok_or_else(|| XlogError::Kernel("d4_assert_u32_eq kernel not found".to_string()))?;
        unsafe {
            assert_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (buffer.num_rows_device(), num_rows_u32),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("d4_assert_u32_eq failed: {}", e)))?;
        self.device.synchronize()?;

        let num_rows = usize::try_from(num_rows_u32)
            .map_err(|_| XlogError::Kernel("Arrow device export row count overflow".to_string()))?;

        let mut fields: Vec<Field> = Vec::with_capacity(buffer.arity());
        let mut children: Vec<ArrayData> = Vec::with_capacity(buffer.arity());

        for (col_idx, (name, scalar_type)) in buffer.schema().columns.iter().enumerate() {
            let (field, child) =
                self.build_arrow_device_child(&buffer, col_idx, name, *scalar_type, num_rows)?;
            fields.push(field);
            children.push(child);
        }

        let struct_type = DataType::Struct(fields.into());
        let struct_data = ArrayData::builder(struct_type)
            .len(num_rows)
            .child_data(children)
            .build()
            .map_err(|e| XlogError::Kernel(format!("Arrow device export failed: {}", e)))?;

        let (ffi_array, ffi_schema) =
            to_ffi(&struct_data).map_err(|e| XlogError::Kernel(format!("{}", e)))?;
        let array_ptr = Box::into_raw(Box::new(ffi_array));
        let schema_ptr = Box::into_raw(Box::new(ffi_schema));

        Ok(ArrowDeviceArray::new(
            ARROW_DEVICE_CUDA,
            self.device.ordinal() as i32,
            array_ptr,
            schema_ptr,
        ))
    }

    /// Import Arrow C Data Interface (device-resident) into a CudaBuffer (zero-copy).
    ///
    /// This is an experimental API gated behind the `arrow-device-import` feature.
    #[cfg(feature = "arrow-device-import")]
    pub fn from_arrow_device_record_batch(
        &self,
        device_array: crate::arrow_device::ArrowDeviceArrayOwned,
    ) -> Result<CudaBuffer> {
        use arrow::array::ArrayData;
        use arrow::datatypes::DataType;
        use arrow::ffi::from_ffi;
        use std::sync::Arc;

        use crate::arrow_device::{ArrowDeviceImport, ARROW_DEVICE_CUDA};
        use crate::memory::CudaColumn;

        let (device_type, device_id, ffi_array, ffi_schema) =
            unsafe { device_array.into_ffi_parts() };

        if device_type != ARROW_DEVICE_CUDA {
            return Err(XlogError::Kernel(format!(
                "Arrow device import requires CUDA device type={}, got {}",
                ARROW_DEVICE_CUDA, device_type
            )));
        }
        if device_id != self.device.ordinal() as i32 {
            return Err(XlogError::Kernel(format!(
                "Arrow device import device id mismatch: expected {}, got {}",
                self.device.ordinal(),
                device_id
            )));
        }

        let data: ArrayData = unsafe { from_ffi(ffi_array, &ffi_schema) }
            .map_err(|e| XlogError::Kernel(format!("Arrow device import failed: {}", e)))?;

        let (fields, children) = match data.data_type() {
            DataType::Struct(fields) => (fields.clone(), data.child_data().to_vec()),
            other => {
                return Err(XlogError::Kernel(format!(
                    "Arrow device import expects Struct, got {:?}",
                    other
                )))
            }
        };

        if data.offset() != 0 {
            return Err(XlogError::Kernel(
                "Arrow device import does not support non-zero offsets".to_string(),
            ));
        }
        if data.null_count() > 0 || data.nulls().is_some() {
            return Err(XlogError::Kernel(
                "Arrow device import does not support nulls".to_string(),
            ));
        }

        let num_rows = data.len();
        if fields.len() != children.len() {
            return Err(XlogError::Kernel(
                "Arrow device import field/child length mismatch".to_string(),
            ));
        }

        let keepalive = Arc::new(ArrowDeviceImport::new(data));
        let mut columns = Vec::with_capacity(children.len());
        let mut schema_cols = Vec::with_capacity(children.len());

        for (field, child) in fields.iter().zip(children.iter()) {
            if child.len() != num_rows {
                return Err(XlogError::Kernel(
                    "Arrow device import child length mismatch".to_string(),
                ));
            }
            if child.offset() != 0 {
                return Err(XlogError::Kernel(
                    "Arrow device import does not support child offsets".to_string(),
                ));
            }
            if child.null_count() > 0 || child.nulls().is_some() {
                return Err(XlogError::Kernel(
                    "Arrow device import does not support child nulls".to_string(),
                ));
            }

            let (scalar_type, elem_size) = Self::scalar_type_from_arrow_field(field)?;
            let buffers = child.buffers();
            let buf = buffers.get(0).ok_or_else(|| {
                XlogError::Kernel("Arrow device import missing value buffer".to_string())
            })?;
            let len_bytes = buf.len();
            let expected_bytes = num_rows.checked_mul(elem_size).ok_or_else(|| {
                XlogError::Kernel("Arrow device import size overflow".to_string())
            })?;
            if len_bytes != expected_bytes {
                return Err(XlogError::Kernel(format!(
                    "Arrow device import buffer size mismatch: expected {}, got {}",
                    expected_bytes, len_bytes
                )));
            }

            let ptr = buf.as_ptr();
            if ptr.is_null() && len_bytes > 0 {
                return Err(XlogError::Kernel(
                    "Arrow device import got null buffer pointer".to_string(),
                ));
            }
            let device_ptr = ptr as usize as cudarc::driver::sys::CUdeviceptr;
            columns.push(CudaColumn::arrow_device(
                device_ptr,
                len_bytes,
                self.device().inner().stream().clone(),
                keepalive.clone(),
            ));
            schema_cols.push((field.name().to_string(), scalar_type));
        }

        let schema = Schema::new(schema_cols);
        self.buffer_from_columns(columns, num_rows as u64, schema)
    }

    /// Export CudaBuffer to Arrow RecordBatch
    ///
    /// Downloads data from GPU and converts it to an Arrow RecordBatch for
    /// interoperability with Arrow-based tools like cuDF, Polars, or DuckDB.
    ///
    /// # Arguments
    /// * `buffer` - The CudaBuffer to export
    ///
    /// # Returns
    /// An Arrow RecordBatch containing all columns from the buffer
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Column download fails
    /// - RecordBatch creation fails
    pub fn to_arrow_record_batch(
        &self,
        buffer: &CudaBuffer,
    ) -> Result<arrow::record_batch::RecordBatch> {
        use arrow::array::*;
        use arrow::datatypes::{Field, Schema as ArrowSchema};

        let num_rows = self.device_row_count(buffer)?;

        let fields: Vec<Field> = buffer
            .schema
            .columns
            .iter()
            .map(|(name, scalar_type)| Field::new(name, scalar_type.to_arrow_type(), false))
            .collect();
        let arrow_schema = Arc::new(ArrowSchema::new(fields));

        let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(buffer.arity());

        for (col_idx, (_, scalar_type)) in buffer.schema.columns.iter().enumerate() {
            let col = buffer
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

            // Handle empty buffer case
            if num_rows == 0 {
                let array: Arc<dyn Array> = match scalar_type {
                    ScalarType::Bool => Arc::new(BooleanArray::from(Vec::<bool>::new())),
                    ScalarType::U32 => Arc::new(UInt32Array::from(Vec::<u32>::new())),
                    ScalarType::Symbol => Arc::new(xlog_core::symbol::to_arrow(&[])),
                    ScalarType::I32 => Arc::new(Int32Array::from(Vec::<i32>::new())),
                    ScalarType::U64 => Arc::new(UInt64Array::from(Vec::<u64>::new())),
                    ScalarType::I64 => Arc::new(Int64Array::from(Vec::<i64>::new())),
                    ScalarType::F32 => Arc::new(Float32Array::from(Vec::<f32>::new())),
                    ScalarType::F64 => Arc::new(Float64Array::from(Vec::<f64>::new())),
                };
                arrays.push(array);
                continue;
            }

            let elem_size = scalar_type.size_bytes();
            let num_bytes = num_rows
                .checked_mul(elem_size)
                .ok_or_else(|| XlogError::Kernel("Row byte size overflow".to_string()))?;
            let mut bytes = vec![0u8; num_bytes];
            let col_view = self.column_bytes_view(col, num_bytes)?;
            self.device
                .inner()
                .dtoh_sync_copy_into(&col_view, &mut bytes)
                .map_err(|e| XlogError::Kernel(format!("Failed to download column: {}", e)))?;

            let array: Arc<dyn Array> = match scalar_type {
                ScalarType::Bool => Arc::new(BooleanArray::from(
                    bytes.iter().map(|&b| b != 0).collect::<Vec<_>>(),
                )),
                ScalarType::U32 => {
                    let values: Vec<u32> = bytes
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Arc::new(UInt32Array::from(values))
                }
                ScalarType::Symbol => {
                    let values: Vec<u32> = bytes
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Arc::new(xlog_core::symbol::to_arrow(&values))
                }
                ScalarType::I32 => {
                    let values: Vec<i32> = bytes
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Arc::new(Int32Array::from(values))
                }
                ScalarType::U64 => {
                    let values: Vec<u64> = bytes
                        .chunks_exact(8)
                        .map(|c| {
                            u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect();
                    Arc::new(UInt64Array::from(values))
                }
                ScalarType::I64 => {
                    let values: Vec<i64> = bytes
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect();
                    Arc::new(Int64Array::from(values))
                }
                ScalarType::F32 => {
                    let values: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Arc::new(Float32Array::from(values))
                }
                ScalarType::F64 => {
                    let values: Vec<f64> = bytes
                        .chunks_exact(8)
                        .map(|c| {
                            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect();
                    Arc::new(Float64Array::from(values))
                }
            };

            arrays.push(array);
        }

        arrow::record_batch::RecordBatch::try_new(arrow_schema, arrays)
            .map_err(|e| XlogError::Kernel(format!("Failed to create RecordBatch: {}", e)))
    }

    fn build_arrow_device_child(
        &self,
        buffer: &Arc<CudaBuffer>,
        col_idx: usize,
        name: &str,
        scalar_type: ScalarType,
        num_rows: usize,
    ) -> Result<(arrow::datatypes::Field, arrow::array::ArrayData)> {
        use arrow::array::ArrayData;
        use arrow::buffer::Buffer;
        use arrow::datatypes::{DataType, Field};
        use std::collections::HashMap;
        use std::ptr::NonNull;

        use crate::arrow_device::ArrowCudaAllocation;

        let col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

        let (dtype, metadata) = if scalar_type == ScalarType::Symbol {
            let mut meta = HashMap::new();
            meta.insert("xlog.symbol".to_string(), "true".to_string());
            meta.insert("xlog.symbol_encoding".to_string(), "u32".to_string());
            (DataType::UInt32, Some(meta))
        } else {
            (scalar_type.to_arrow_type(), None)
        };

        let field = match metadata {
            Some(meta) => Field::new(name, dtype.clone(), false).with_metadata(meta),
            None => Field::new(name, dtype.clone(), false),
        };

        let elem_size = match scalar_type {
            ScalarType::Symbol => 4usize,
            _ => scalar_type.size_bytes(),
        };

        let len_bytes = num_rows
            .checked_mul(elem_size)
            .ok_or_else(|| XlogError::Kernel("Arrow device export size overflow".to_string()))?;

        let mut extra = Vec::new();
        let (ptr, len) = match scalar_type {
            ScalarType::Bool => {
                let packed_len = (num_rows + 7) / 8;
                let mut packed = self.memory.alloc::<u8>(packed_len)?;
                let pack_fn = self
                    .device
                    .inner()
                    .get_func(PACK_MODULE, pack_kernels::PACK_BOOLS_TO_BITMAP)
                    .ok_or_else(|| {
                        XlogError::Kernel("pack_bools_to_bitmap kernel not found".to_string())
                    })?;
                let block_size = 256u32;
                let grid_size = (packed_len as u32 + block_size - 1) / block_size;
                unsafe {
                    pack_fn.clone().launch(
                        LaunchConfig {
                            grid_dim: (grid_size, 1, 1),
                            block_dim: (block_size, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (col, num_rows as u32, &mut packed),
                    )
                }
                .map_err(|e| XlogError::Kernel(format!("pack_bools_to_bitmap failed: {}", e)))?;
                self.device.synchronize()?;
                let ptr = *packed.device_ptr() as usize as *mut u8;
                extra.push(packed);
                (ptr, packed_len)
            }
            _ => {
                let ptr = *col.device_ptr() as usize as *mut u8;
                (ptr, len_bytes)
            }
        };

        let alloc = Arc::new(ArrowCudaAllocation::new(Arc::clone(buffer), extra));
        let nn = if len == 0 {
            NonNull::dangling()
        } else {
            NonNull::new(ptr).ok_or_else(|| {
                XlogError::Kernel("Arrow device export got null device pointer".to_string())
            })?
        };
        let buf = unsafe { Buffer::from_custom_allocation(nn, len, alloc) };

        let data = ArrayData::builder(dtype)
            .len(num_rows)
            .add_buffer(buf)
            .build()
            .map_err(|e| XlogError::Kernel(format!("Arrow device export failed: {}", e)))?;

        Ok((field, data))
    }

    #[cfg(feature = "arrow-device-import")]
    fn scalar_type_from_arrow_field(
        field: &arrow::datatypes::Field,
    ) -> Result<(ScalarType, usize)> {
        use arrow::datatypes::DataType;

        // Arrow's Field metadata is a map (possibly empty), not an Option.
        let is_symbol = field
            .metadata()
            .get("xlog.symbol")
            .map(|v| v == "true")
            .unwrap_or(false);

        let scalar = match field.data_type() {
            DataType::Boolean => {
                return Err(XlogError::Kernel(
                    "Arrow device import does not support bit-packed bool yet".to_string(),
                ))
            }
            DataType::UInt32 if is_symbol => ScalarType::Symbol,
            dt => ScalarType::from_arrow_type(dt).ok_or_else(|| {
                XlogError::Kernel(format!("Arrow device import unsupported type {:?}", dt))
            })?,
        };

        let elem_size = match scalar {
            ScalarType::Symbol => 4usize,
            _ => scalar.size_bytes(),
        };

        Ok((scalar, elem_size))
    }

    /// Import Arrow RecordBatch to CudaBuffer
    ///
    /// Uploads Arrow data to GPU memory.
    ///
    /// # Arguments
    /// * `record_batch` - The Arrow RecordBatch to import
    ///
    /// # Returns
    /// A new CudaBuffer with the data on GPU
    ///
    /// # Errors
    /// Returns error if Arrow type is not supported or upload fails
    pub fn from_arrow_record_batch(
        &self,
        record_batch: &arrow::record_batch::RecordBatch,
    ) -> Result<CudaBuffer> {
        use arrow::array::*;
        use arrow::datatypes::DataType;

        let num_rows = record_batch.num_rows() as u64;

        if num_rows == 0 {
            let columns: Vec<(String, ScalarType)> = record_batch
                .schema()
                .fields()
                .iter()
                .filter_map(|f| {
                    ScalarType::from_arrow_type(f.data_type()).map(|st| (f.name().clone(), st))
                })
                .collect();
            return self.create_empty_buffer(Schema::new(columns));
        }

        let device = self.device.inner();
        let mut columns = Vec::with_capacity(record_batch.num_columns());
        let mut schema_cols = Vec::with_capacity(record_batch.num_columns());

        for (col_idx, field) in record_batch.schema().fields().iter().enumerate() {
            let array = record_batch.column(col_idx);
            let scalar_type = ScalarType::from_arrow_type(field.data_type()).ok_or_else(|| {
                XlogError::Kernel(format!("Unsupported Arrow type: {:?}", field.data_type()))
            })?;

            let bytes: Vec<u8> = match field.data_type() {
                DataType::Boolean => {
                    let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                    arr.iter()
                        .map(|v| if v.unwrap_or(false) { 1u8 } else { 0u8 })
                        .collect()
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
                _ => {
                    return Err(XlogError::Kernel(format!(
                        "Unsupported Arrow type: {:?}",
                        field.data_type()
                    )))
                }
            };

            let mut d_col = self.memory.alloc::<u8>(bytes.len())?;
            device
                .htod_sync_copy_into(&bytes, &mut d_col)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload column: {}", e)))?;

            columns.push(d_col.into());
            schema_cols.push((field.name().clone(), scalar_type));
        }

        self.buffer_from_columns(columns, num_rows, Schema::new(schema_cols))
    }

    /// Export a CudaBuffer to an Arrow IPC stream (RecordBatchStream) as bytes.
    ///
    /// This is a convenience wrapper around `to_arrow_record_batch` that enables
    /// interoperability with tools like cuDF via standard Arrow IPC readers.
    ///
    /// Note: This is not zero-copy; data is downloaded from GPU to host memory.
    pub fn to_arrow_ipc_stream(&self, buffer: &CudaBuffer) -> Result<Vec<u8>> {
        use arrow::ipc::writer::StreamWriter;

        let batch = self.to_arrow_record_batch(buffer)?;
        let mut out = Vec::new();
        let mut writer = StreamWriter::try_new(&mut out, &batch.schema())
            .map_err(|e| XlogError::Kernel(format!("Failed to create Arrow IPC writer: {}", e)))?;
        writer
            .write(&batch)
            .map_err(|e| XlogError::Kernel(format!("Failed to write Arrow RecordBatch: {}", e)))?;
        writer
            .finish()
            .map_err(|e| XlogError::Kernel(format!("Failed to finish Arrow IPC stream: {}", e)))?;
        Ok(out)
    }

    /// Import a single-batch Arrow IPC stream (RecordBatchStream) into a CudaBuffer.
    ///
    /// Note: This uploads Arrow data from host to GPU memory.
    pub fn from_arrow_ipc_stream(&self, ipc: &[u8]) -> Result<CudaBuffer> {
        use arrow::ipc::reader::StreamReader;
        use std::io::Cursor;

        let cursor = Cursor::new(ipc);
        let mut reader = StreamReader::try_new(cursor, None)
            .map_err(|e| XlogError::Kernel(format!("Failed to create Arrow IPC reader: {}", e)))?;

        let batches = reader.by_ref();
        let first = batches
            .next()
            .ok_or_else(|| {
                XlogError::Kernel("Arrow IPC stream contained no record batches".to_string())
            })?
            .map_err(|e| XlogError::Kernel(format!("Failed to read Arrow RecordBatch: {}", e)))?;

        if batches.next().is_some() {
            return Err(XlogError::Kernel(
                "Arrow IPC stream contains multiple record batches; this API expects exactly one"
                    .to_string(),
            ));
        }

        self.from_arrow_record_batch(&first)
    }

    /// Write a CudaBuffer to a file as an Arrow IPC stream (RecordBatchStream).
    pub fn write_arrow_ipc_stream_file<P: AsRef<std::path::Path>>(
        &self,
        buffer: &CudaBuffer,
        path: P,
    ) -> Result<()> {
        let bytes = self.to_arrow_ipc_stream(buffer)?;
        std::fs::write(&path, bytes).map_err(|e| {
            XlogError::Kernel(format!(
                "Failed to write Arrow IPC stream to {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;
        Ok(())
    }

    /// Read a CudaBuffer from a file containing an Arrow IPC stream (RecordBatchStream).
    pub fn read_arrow_ipc_stream_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<CudaBuffer> {
        let bytes = std::fs::read(&path).map_err(|e| {
            XlogError::Kernel(format!(
                "Failed to read Arrow IPC stream from {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;
        self.from_arrow_ipc_stream(&bytes)
    }
}
