//! Arithmetic column operations: add, sub, mul, div, mod, abs, min, max, pow, cast, select, combine.

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, ScalarType, Schema, XlogError};

use super::{arith_kernels, ARITH_MODULE};
use crate::memory::TrackedCudaSlice;
use crate::CudaBuffer;

impl super::CudaKernelProvider {
    /// Create a column filled with a constant value
    ///
    /// # Arguments
    /// * `value_bytes` - The raw bytes of the constant value (in little-endian format)
    /// * `col_type` - The ScalarType of the column
    /// * `num_rows` - Number of rows to create
    ///
    /// # Returns
    /// A new single-column CudaBuffer filled with the constant value
    pub fn create_constant_column(
        &self,
        value_bytes: &[u8],
        col_type: ScalarType,
        num_rows: u64,
    ) -> Result<CudaBuffer> {
        if num_rows == 0 {
            let schema = Schema::new(vec![("const".to_string(), col_type)]);
            return self.create_empty_buffer(schema);
        }

        let elem_size = col_type.size_bytes();
        if value_bytes.len() != elem_size {
            return Err(XlogError::Kernel(format!(
                "Value bytes length {} doesn't match type size {}",
                value_bytes.len(),
                elem_size
            )));
        }

        if num_rows > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Constant column supports at most {} rows, got {}",
                u32::MAX,
                num_rows
            )));
        }

        let total_bytes = (num_rows as usize)
            .checked_mul(elem_size)
            .ok_or_else(|| XlogError::Kernel("Constant column size overflow".to_string()))?;

        let mut dst_col = self.memory.alloc::<u8>(total_bytes)?;
        let n = num_rows as u32;

        macro_rules! launch_fill_const {
            ($kernel:expr, $value:expr) => {{
                let func = self
                    .device
                    .inner()
                    .get_func(ARITH_MODULE, $kernel)
                    .ok_or_else(|| XlogError::Kernel("arith fill kernel not found".to_string()))?;
                let config = LaunchConfig::for_num_elems(n);
                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe { func.clone().launch(config, ($value, n, &mut dst_col)) }
                    .map_err(|e| XlogError::Kernel(format!("fill const failed: {}", e)))?;
            }};
        }

        match col_type {
            ScalarType::U32 | ScalarType::Symbol => {
                let value = u32::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_U32, value);
            }
            ScalarType::U64 => {
                let value = u64::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_U64, value);
            }
            ScalarType::I64 => {
                let value = i64::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_I64, value);
            }
            ScalarType::I32 => {
                let value = i32::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_I32, value);
            }
            ScalarType::F64 => {
                let value = f64::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_F64, value);
            }
            ScalarType::F32 => {
                let value = f32::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_F32, value);
            }
            ScalarType::Bool => {
                let value = value_bytes[0];
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_U8, value);
            }
        }

        self.device.synchronize()?;

        let schema = Schema::new(vec![("const".to_string(), col_type)]);
        self.buffer_from_columns(vec![dst_col.into()], num_rows, schema)
    }

    /// Create a constant column sized to `row_cap` while preserving device row count from `d_num_rows_src`.
    pub fn create_constant_column_with_device_count(
        &self,
        value_bytes: &[u8],
        col_type: ScalarType,
        row_cap: u64,
        d_num_rows_src: &TrackedCudaSlice<u32>,
    ) -> Result<CudaBuffer> {
        if row_cap == 0 {
            let schema = Schema::new(vec![("const".to_string(), col_type)]);
            return self.create_empty_buffer(schema);
        }

        let elem_size = col_type.size_bytes();
        if value_bytes.len() != elem_size {
            return Err(XlogError::Kernel(format!(
                "Value bytes length {} doesn't match type size {}",
                value_bytes.len(),
                elem_size
            )));
        }

        if row_cap > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Constant column supports at most {} rows, got {}",
                u32::MAX,
                row_cap
            )));
        }

        let total_bytes = (row_cap as usize)
            .checked_mul(elem_size)
            .ok_or_else(|| XlogError::Kernel("Constant column size overflow".to_string()))?;

        let mut dst_col = self.memory.alloc::<u8>(total_bytes)?;
        let n = row_cap as u32;

        macro_rules! launch_fill_const {
            ($kernel:expr, $value:expr) => {{
                let func = self
                    .device
                    .inner()
                    .get_func(ARITH_MODULE, $kernel)
                    .ok_or_else(|| XlogError::Kernel("arith fill kernel not found".to_string()))?;
                let config = LaunchConfig::for_num_elems(n);
                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe { func.clone().launch(config, ($value, n, &mut dst_col)) }
                    .map_err(|e| XlogError::Kernel(format!("fill const failed: {}", e)))?;
            }};
        }

        match col_type {
            ScalarType::U32 | ScalarType::Symbol => {
                let value = u32::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_U32, value);
            }
            ScalarType::U64 => {
                let value = u64::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_U64, value);
            }
            ScalarType::I64 => {
                let value = i64::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_I64, value);
            }
            ScalarType::I32 => {
                let value = i32::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_I32, value);
            }
            ScalarType::F64 => {
                let value = f64::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_F64, value);
            }
            ScalarType::F32 => {
                let value = f32::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_F32, value);
            }
            ScalarType::Bool => {
                let value = value_bytes[0];
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_U8, value);
            }
        }

        self.device.synchronize()?;

        let schema = Schema::new(vec![("const".to_string(), col_type)]);
        let mut d_num_rows = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .dtod_copy(d_num_rows_src, &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy row count: {}", e)))?;

        Ok(CudaBuffer::from_columns(
            vec![dst_col.into()],
            row_cap,
            d_num_rows,
            schema,
        ))
    }

    /// Element-wise addition of two single-column buffers
    ///
    /// Performs element-wise addition using GPU kernels.
    /// Uses wrapping arithmetic for integer overflow.
    ///
    /// # Arguments
    /// * `a` - First operand buffer (single column)
    /// * `b` - Second operand buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise sum
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn add_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        match a.schema().column_type(0) {
            Some(ScalarType::I64) => {
                self.binary_arith_op_device::<i64>(a, b, 0, arith_kernels::ARITH_BINARY_I64)
            }
            Some(ScalarType::I32) => {
                self.binary_arith_op_device::<i32>(a, b, 0, arith_kernels::ARITH_BINARY_I32)
            }
            Some(ScalarType::U64) => {
                self.binary_arith_op_device::<u64>(a, b, 0, arith_kernels::ARITH_BINARY_U64)
            }
            Some(ScalarType::U32 | ScalarType::Symbol) => {
                self.binary_arith_op_device::<u32>(a, b, 0, arith_kernels::ARITH_BINARY_U32)
            }
            Some(ScalarType::F64) => {
                self.binary_arith_op_device::<f64>(a, b, 0, arith_kernels::ARITH_BINARY_F64)
            }
            Some(ScalarType::F32) => {
                self.binary_arith_op_device::<f32>(a, b, 0, arith_kernels::ARITH_BINARY_F32)
            }
            other => Err(XlogError::Kernel(format!(
                "Arithmetic not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise subtraction of two single-column buffers
    ///
    /// Performs element-wise subtraction using GPU kernels.
    /// Uses wrapping arithmetic for integer overflow.
    ///
    /// # Arguments
    /// * `a` - First operand buffer (single column)
    /// * `b` - Second operand buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise difference
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn sub_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        match a.schema().column_type(0) {
            Some(ScalarType::I64) => {
                self.binary_arith_op_device::<i64>(a, b, 1, arith_kernels::ARITH_BINARY_I64)
            }
            Some(ScalarType::I32) => {
                self.binary_arith_op_device::<i32>(a, b, 1, arith_kernels::ARITH_BINARY_I32)
            }
            Some(ScalarType::U64) => {
                self.binary_arith_op_device::<u64>(a, b, 1, arith_kernels::ARITH_BINARY_U64)
            }
            Some(ScalarType::U32 | ScalarType::Symbol) => {
                self.binary_arith_op_device::<u32>(a, b, 1, arith_kernels::ARITH_BINARY_U32)
            }
            Some(ScalarType::F64) => {
                self.binary_arith_op_device::<f64>(a, b, 1, arith_kernels::ARITH_BINARY_F64)
            }
            Some(ScalarType::F32) => {
                self.binary_arith_op_device::<f32>(a, b, 1, arith_kernels::ARITH_BINARY_F32)
            }
            other => Err(XlogError::Kernel(format!(
                "Arithmetic not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise multiplication of two single-column buffers
    ///
    /// Performs element-wise multiplication using GPU kernels.
    /// Uses wrapping arithmetic for integer overflow.
    ///
    /// # Arguments
    /// * `a` - First operand buffer (single column)
    /// * `b` - Second operand buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise product
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn mul_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        match a.schema().column_type(0) {
            Some(ScalarType::I64) => {
                self.binary_arith_op_device::<i64>(a, b, 2, arith_kernels::ARITH_BINARY_I64)
            }
            Some(ScalarType::I32) => {
                self.binary_arith_op_device::<i32>(a, b, 2, arith_kernels::ARITH_BINARY_I32)
            }
            Some(ScalarType::U64) => {
                self.binary_arith_op_device::<u64>(a, b, 2, arith_kernels::ARITH_BINARY_U64)
            }
            Some(ScalarType::U32 | ScalarType::Symbol) => {
                self.binary_arith_op_device::<u32>(a, b, 2, arith_kernels::ARITH_BINARY_U32)
            }
            Some(ScalarType::F64) => {
                self.binary_arith_op_device::<f64>(a, b, 2, arith_kernels::ARITH_BINARY_F64)
            }
            Some(ScalarType::F32) => {
                self.binary_arith_op_device::<f32>(a, b, 2, arith_kernels::ARITH_BINARY_F32)
            }
            other => Err(XlogError::Kernel(format!(
                "Arithmetic not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise division of two single-column buffers
    ///
    /// Performs element-wise division using GPU kernels.
    /// For signed integers, division by zero returns i64::MAX/i32::MAX.
    /// For unsigned integers, division by zero returns u64::MAX/u32::MAX.
    /// For floats, division by zero produces Inf/NaN as per IEEE 754.
    ///
    /// # Arguments
    /// * `a` - Dividend buffer (single column)
    /// * `b` - Divisor buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise quotient
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn div_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        match a.schema().column_type(0) {
            Some(ScalarType::I64) => {
                self.binary_arith_op_device::<i64>(a, b, 3, arith_kernels::ARITH_BINARY_I64)
            }
            Some(ScalarType::I32) => {
                self.binary_arith_op_device::<i32>(a, b, 3, arith_kernels::ARITH_BINARY_I32)
            }
            Some(ScalarType::U64) => {
                self.binary_arith_op_device::<u64>(a, b, 3, arith_kernels::ARITH_BINARY_U64)
            }
            Some(ScalarType::U32 | ScalarType::Symbol) => {
                self.binary_arith_op_device::<u32>(a, b, 3, arith_kernels::ARITH_BINARY_U32)
            }
            Some(ScalarType::F64) => {
                self.binary_arith_op_device::<f64>(a, b, 3, arith_kernels::ARITH_BINARY_F64)
            }
            Some(ScalarType::F32) => {
                self.binary_arith_op_device::<f32>(a, b, 3, arith_kernels::ARITH_BINARY_F32)
            }
            other => Err(XlogError::Kernel(format!(
                "Arithmetic not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise modulo of two single-column buffers
    ///
    /// Performs element-wise modulo using GPU kernels.
    /// For integers, modulo by zero returns 0.
    /// For floats, modulo by zero produces NaN as per IEEE 754.
    ///
    /// # Arguments
    /// * `a` - Dividend buffer (single column)
    /// * `b` - Divisor buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise remainder
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn mod_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        match a.schema().column_type(0) {
            Some(ScalarType::I64) => {
                self.binary_arith_op_device::<i64>(a, b, 4, arith_kernels::ARITH_BINARY_I64)
            }
            Some(ScalarType::I32) => {
                self.binary_arith_op_device::<i32>(a, b, 4, arith_kernels::ARITH_BINARY_I32)
            }
            Some(ScalarType::U64) => {
                self.binary_arith_op_device::<u64>(a, b, 4, arith_kernels::ARITH_BINARY_U64)
            }
            Some(ScalarType::U32 | ScalarType::Symbol) => {
                self.binary_arith_op_device::<u32>(a, b, 4, arith_kernels::ARITH_BINARY_U32)
            }
            Some(ScalarType::F64) => {
                self.binary_arith_op_device::<f64>(a, b, 4, arith_kernels::ARITH_BINARY_F64)
            }
            Some(ScalarType::F32) => {
                self.binary_arith_op_device::<f32>(a, b, 4, arith_kernels::ARITH_BINARY_F32)
            }
            other => Err(XlogError::Kernel(format!(
                "Arithmetic not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise absolute value of a single-column buffer
    ///
    /// Performs element-wise absolute value using GPU kernels.
    ///
    /// # Arguments
    /// * `a` - Input buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the absolute values
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Buffer is not single-column
    /// - Type is not supported for arithmetic
    pub fn abs_column(&self, a: &CudaBuffer) -> Result<CudaBuffer> {
        if a.arity() != 1 {
            return Err(XlogError::Kernel(
                "Arithmetic requires single-column buffers".into(),
            ));
        }

        if a.num_rows() == 0 {
            return self.create_empty_buffer(a.schema().clone());
        }

        let n: u32 = a.num_rows().try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "abs_column: row count {} exceeds u32::MAX",
                a.num_rows()
            ))
        })?;
        let col = a
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing column 0".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        match a.schema().column_type(0) {
            Some(ScalarType::I64) => {
                let expected_bytes = (n as usize)
                    .checked_mul(std::mem::size_of::<i64>())
                    .ok_or_else(|| XlogError::Kernel("abs_column size overflow".into()))?;
                if col.num_bytes() != expected_bytes {
                    return Err(XlogError::Kernel(format!(
                        "Column 0 has {} bytes but expected {} for {} rows",
                        col.num_bytes(),
                        expected_bytes,
                        a.num_rows()
                    )));
                }
                let mut out = self.memory.alloc::<u8>(expected_bytes)?;
                let func = self
                    .device
                    .inner()
                    .get_func(ARITH_MODULE, arith_kernels::ARITH_ABS_I64)
                    .ok_or_else(|| XlogError::Kernel("arith_abs_i64 not found".into()))?;
                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe { func.clone().launch(config, (col, n, &mut out)) }
                    .map_err(|e| XlogError::Kernel(format!("abs_i64 failed: {}", e)))?;
                self.device.synchronize()?;
                self.buffer_from_columns_with_device_count(
                    vec![out.into()],
                    a.num_rows(),
                    a.schema.clone(),
                    a,
                )
            }
            Some(ScalarType::I32) => {
                let expected_bytes = (n as usize)
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or_else(|| XlogError::Kernel("abs_column size overflow".into()))?;
                if col.num_bytes() != expected_bytes {
                    return Err(XlogError::Kernel(format!(
                        "Column 0 has {} bytes but expected {} for {} rows",
                        col.num_bytes(),
                        expected_bytes,
                        a.num_rows()
                    )));
                }
                let mut out = self.memory.alloc::<u8>(expected_bytes)?;
                let func = self
                    .device
                    .inner()
                    .get_func(ARITH_MODULE, arith_kernels::ARITH_ABS_I32)
                    .ok_or_else(|| XlogError::Kernel("arith_abs_i32 not found".into()))?;
                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe { func.clone().launch(config, (col, n, &mut out)) }
                    .map_err(|e| XlogError::Kernel(format!("abs_i32 failed: {}", e)))?;
                self.device.synchronize()?;
                self.buffer_from_columns_with_device_count(
                    vec![out.into()],
                    a.num_rows(),
                    a.schema.clone(),
                    a,
                )
            }
            Some(ScalarType::F64) => {
                let expected_bytes = (n as usize)
                    .checked_mul(std::mem::size_of::<f64>())
                    .ok_or_else(|| XlogError::Kernel("abs_column size overflow".into()))?;
                if col.num_bytes() != expected_bytes {
                    return Err(XlogError::Kernel(format!(
                        "Column 0 has {} bytes but expected {} for {} rows",
                        col.num_bytes(),
                        expected_bytes,
                        a.num_rows()
                    )));
                }
                let mut out = self.memory.alloc::<u8>(expected_bytes)?;
                let func = self
                    .device
                    .inner()
                    .get_func(ARITH_MODULE, arith_kernels::ARITH_ABS_F64)
                    .ok_or_else(|| XlogError::Kernel("arith_abs_f64 not found".into()))?;
                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe { func.clone().launch(config, (col, n, &mut out)) }
                    .map_err(|e| XlogError::Kernel(format!("abs_f64 failed: {}", e)))?;
                self.device.synchronize()?;
                self.buffer_from_columns_with_device_count(
                    vec![out.into()],
                    a.num_rows(),
                    a.schema.clone(),
                    a,
                )
            }
            Some(ScalarType::F32) => {
                let expected_bytes = (n as usize)
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| XlogError::Kernel("abs_column size overflow".into()))?;
                if col.num_bytes() != expected_bytes {
                    return Err(XlogError::Kernel(format!(
                        "Column 0 has {} bytes but expected {} for {} rows",
                        col.num_bytes(),
                        expected_bytes,
                        a.num_rows()
                    )));
                }
                let mut out = self.memory.alloc::<u8>(expected_bytes)?;
                let func = self
                    .device
                    .inner()
                    .get_func(ARITH_MODULE, arith_kernels::ARITH_ABS_F32)
                    .ok_or_else(|| XlogError::Kernel("arith_abs_f32 not found".into()))?;
                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe { func.clone().launch(config, (col, n, &mut out)) }
                    .map_err(|e| XlogError::Kernel(format!("abs_f32 failed: {}", e)))?;
                self.device.synchronize()?;
                self.buffer_from_columns_with_device_count(
                    vec![out.into()],
                    a.num_rows(),
                    a.schema.clone(),
                    a,
                )
            }
            Some(ScalarType::U32 | ScalarType::U64 | ScalarType::Bool | ScalarType::Symbol) => {
                self.clone_buffer(a)
            }
            other => Err(XlogError::Kernel(format!(
                "Abs not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise minimum of two single-column buffers
    ///
    /// Performs element-wise minimum using GPU kernels.
    ///
    /// # Arguments
    /// * `a` - First operand buffer (single column)
    /// * `b` - Second operand buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise minimums
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn min_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        match a.schema().column_type(0) {
            Some(ScalarType::I64) => {
                self.binary_arith_op_device::<i64>(a, b, 5, arith_kernels::ARITH_BINARY_I64)
            }
            Some(ScalarType::I32) => {
                self.binary_arith_op_device::<i32>(a, b, 5, arith_kernels::ARITH_BINARY_I32)
            }
            Some(ScalarType::U64) => {
                self.binary_arith_op_device::<u64>(a, b, 5, arith_kernels::ARITH_BINARY_U64)
            }
            Some(ScalarType::U32 | ScalarType::Symbol) => {
                self.binary_arith_op_device::<u32>(a, b, 5, arith_kernels::ARITH_BINARY_U32)
            }
            Some(ScalarType::F64) => {
                self.binary_arith_op_device::<f64>(a, b, 5, arith_kernels::ARITH_BINARY_F64)
            }
            Some(ScalarType::F32) => {
                self.binary_arith_op_device::<f32>(a, b, 5, arith_kernels::ARITH_BINARY_F32)
            }
            other => Err(XlogError::Kernel(format!(
                "Arithmetic not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise maximum of two single-column buffers
    ///
    /// Performs element-wise maximum using GPU kernels.
    ///
    /// # Arguments
    /// * `a` - First operand buffer (single column)
    /// * `b` - Second operand buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise maximums
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn max_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        match a.schema().column_type(0) {
            Some(ScalarType::I64) => {
                self.binary_arith_op_device::<i64>(a, b, 6, arith_kernels::ARITH_BINARY_I64)
            }
            Some(ScalarType::I32) => {
                self.binary_arith_op_device::<i32>(a, b, 6, arith_kernels::ARITH_BINARY_I32)
            }
            Some(ScalarType::U64) => {
                self.binary_arith_op_device::<u64>(a, b, 6, arith_kernels::ARITH_BINARY_U64)
            }
            Some(ScalarType::U32 | ScalarType::Symbol) => {
                self.binary_arith_op_device::<u32>(a, b, 6, arith_kernels::ARITH_BINARY_U32)
            }
            Some(ScalarType::F64) => {
                self.binary_arith_op_device::<f64>(a, b, 6, arith_kernels::ARITH_BINARY_F64)
            }
            Some(ScalarType::F32) => {
                self.binary_arith_op_device::<f32>(a, b, 6, arith_kernels::ARITH_BINARY_F32)
            }
            other => Err(XlogError::Kernel(format!(
                "Arithmetic not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise power of two single-column buffers
    ///
    /// Converts both operands to f64, computes x^y on the GPU, and returns f64 result.
    /// This matches the behavior of most database systems where pow() returns a float.
    ///
    /// # Arguments
    /// * `base` - Base values buffer (single column)
    /// * `exp` - Exponent values buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise powers as f64
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn pow_columns(&self, base: &CudaBuffer, exp: &CudaBuffer) -> Result<CudaBuffer> {
        if base.num_rows() != exp.num_rows() {
            return Err(XlogError::Kernel("Row count mismatch".into()));
        }
        if base.arity() != 1 || exp.arity() != 1 {
            return Err(XlogError::Kernel(
                "Arithmetic requires single-column buffers".into(),
            ));
        }

        if base.num_rows() == 0 {
            let schema = Schema::new(vec![("result".to_string(), ScalarType::F64)]);
            return self.create_empty_buffer(schema);
        }

        let n: u32 = base.num_rows().try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "pow_columns: row count {} exceeds u32::MAX",
                base.num_rows()
            ))
        })?;

        let base_f64_buf = if base.schema().column_type(0) == Some(ScalarType::F64) {
            None
        } else {
            Some(self.cast_column(base, ScalarType::F64)?)
        };
        let base_buf = base_f64_buf.as_ref().unwrap_or(base);

        let exp_f64_buf = if exp.schema().column_type(0) == Some(ScalarType::F64) {
            None
        } else {
            Some(self.cast_column(exp, ScalarType::F64)?)
        };
        let exp_buf = exp_f64_buf.as_ref().unwrap_or(exp);

        let base_col = base_buf
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing base column".into()))?;
        let exp_col = exp_buf
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing exp column".into()))?;

        let expected_bytes = (n as usize)
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| XlogError::Kernel("pow_columns size overflow".into()))?;
        if base_col.num_bytes() != expected_bytes || exp_col.num_bytes() != expected_bytes {
            return Err(XlogError::Kernel(format!(
                "pow_columns: expected {} bytes for {} rows",
                expected_bytes,
                base.num_rows()
            )));
        }

        let mut out = self.memory.alloc::<u8>(expected_bytes)?;
        let func = self
            .device
            .inner()
            .get_func(ARITH_MODULE, arith_kernels::ARITH_POW_F64)
            .ok_or_else(|| XlogError::Kernel("arith_pow_f64 not found".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone()
                .launch(config, (base_col, exp_col, n, &mut out))
        }
        .map_err(|e| XlogError::Kernel(format!("pow_f64 failed: {}", e)))?;

        self.device.synchronize()?;

        let schema = Schema::new(vec![("result".to_string(), ScalarType::F64)]);
        self.buffer_from_columns_with_device_count(vec![out.into()], base.num_rows(), schema, base)
    }

    /// Conditional select between two single-column buffers based on a boolean mask.
    ///
    /// For each row: out[i] = mask[i] ? then_vals[i] : else_vals[i]
    ///
    /// # Arguments
    /// * `mask` - Boolean mask buffer (single column, type Bool/u8)
    /// * `then_vals` - Values to select when mask is true
    /// * `else_vals` - Values to select when mask is false
    ///
    /// # Returns
    /// A new CudaBuffer with values selected based on the mask
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Types of then/else values don't match
    pub fn select_columns(
        &self,
        mask: &CudaBuffer,
        then_vals: &CudaBuffer,
        else_vals: &CudaBuffer,
    ) -> Result<CudaBuffer> {
        if mask.num_rows() != then_vals.num_rows() || mask.num_rows() != else_vals.num_rows() {
            return Err(XlogError::Kernel("Row count mismatch in select".into()));
        }
        if mask.arity() != 1 || then_vals.arity() != 1 || else_vals.arity() != 1 {
            return Err(XlogError::Kernel(
                "Select requires single-column buffers".into(),
            ));
        }

        let then_type = then_vals.schema().column_type(0);
        let else_type = else_vals.schema().column_type(0);
        if then_type != else_type {
            return Err(XlogError::Kernel(format!(
                "Type mismatch in select: then={:?}, else={:?}",
                then_type, else_type
            )));
        }

        if mask.num_rows() == 0 {
            let result_type = then_type.unwrap_or(ScalarType::I64);
            let schema = Schema::new(vec![("result".to_string(), result_type)]);
            return self.create_empty_buffer(schema);
        }

        let n: u32 = mask.num_rows().try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "select_columns: row count {} exceeds u32::MAX",
                mask.num_rows()
            ))
        })?;

        let mask_col = mask
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing mask column".into()))?;
        let then_col = then_vals
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing then column".into()))?;
        let else_col = else_vals
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing else column".into()))?;

        let result_type = then_type.unwrap_or(ScalarType::I64);
        let elem_size = result_type.size_bytes();
        let expected_bytes = (n as usize)
            .checked_mul(elem_size)
            .ok_or_else(|| XlogError::Kernel("select_columns size overflow".into()))?;

        let mut out = self.memory.alloc::<u8>(expected_bytes)?;

        let kernel_name = match result_type {
            ScalarType::I64 => arith_kernels::ARITH_SELECT_I64,
            ScalarType::I32 => arith_kernels::ARITH_SELECT_I32,
            ScalarType::U64 => arith_kernels::ARITH_SELECT_U64,
            ScalarType::U32 | ScalarType::Symbol => arith_kernels::ARITH_SELECT_U32,
            ScalarType::F64 => arith_kernels::ARITH_SELECT_F64,
            ScalarType::F32 => arith_kernels::ARITH_SELECT_F32,
            ScalarType::Bool => {
                // Bool is stored as u8, treat as u8 select (use fill + mask trick)
                // For simplicity, cast to u32 and back
                return self.select_columns_bool(mask, then_vals, else_vals);
            }
        };

        let func = self
            .device
            .inner()
            .get_func(ARITH_MODULE, kernel_name)
            .ok_or_else(|| XlogError::Kernel(format!("{} not found", kernel_name)))?;
        let config = LaunchConfig::for_num_elems(n);

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone()
                .launch(config, (mask_col, then_col, else_col, n, &mut out))
        }
        .map_err(|e| XlogError::Kernel(format!("select kernel failed: {}", e)))?;

        self.device.synchronize()?;

        let schema = Schema::new(vec![("result".to_string(), result_type)]);
        self.buffer_from_columns_with_device_count(vec![out.into()], mask.num_rows(), schema, mask)
    }

    /// Helper for select_columns when result type is Bool
    fn select_columns_bool(
        &self,
        mask: &CudaBuffer,
        then_vals: &CudaBuffer,
        else_vals: &CudaBuffer,
    ) -> Result<CudaBuffer> {
        // Cast bool columns to u32, select, then cast back
        let then_u32 = self.cast_column(then_vals, ScalarType::U32)?;
        let else_u32 = self.cast_column(else_vals, ScalarType::U32)?;
        let result_u32 = self.select_columns(mask, &then_u32, &else_u32)?;
        self.cast_column(&result_u32, ScalarType::Bool)
    }

    /// Cast a single-column buffer to a different type
    ///
    /// Casts data on the GPU using the arithmetic cast kernel.
    ///
    /// # Arguments
    /// * `a` - Input buffer (single column)
    /// * `target` - Target scalar type
    ///
    /// # Returns
    /// A new CudaBuffer with the cast values
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Buffer is not single-column
    /// - Source or target type is not supported for casting
    pub fn cast_column(&self, a: &CudaBuffer, target: ScalarType) -> Result<CudaBuffer> {
        if a.arity() != 1 {
            return Err(XlogError::Kernel(
                "Cast requires single-column buffer".into(),
            ));
        }

        let source_type = a
            .schema()
            .column_type(0)
            .ok_or_else(|| XlogError::Kernel("Missing column type".into()))?;

        let schema = Schema::new(vec![("result".to_string(), target)]);

        if a.num_rows() == 0 {
            return self.create_empty_buffer(schema);
        }

        let n: u32 = a.num_rows().try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "cast_column: row count {} exceeds u32::MAX",
                a.num_rows()
            ))
        })?;

        let src_col = a
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing column 0".into()))?;
        let src_bytes = (n as usize)
            .checked_mul(source_type.size_bytes())
            .ok_or_else(|| XlogError::Kernel("cast_column size overflow".into()))?;
        if src_col.num_bytes() != src_bytes {
            return Err(XlogError::Kernel(format!(
                "Column 0 has {} bytes but expected {} for {} rows",
                src_col.num_bytes(),
                src_bytes,
                a.num_rows()
            )));
        }

        let dst_bytes = (n as usize)
            .checked_mul(target.size_bytes())
            .ok_or_else(|| XlogError::Kernel("cast_column size overflow".into()))?;
        let mut out = self.memory.alloc::<u8>(dst_bytes)?;

        let func = self
            .device
            .inner()
            .get_func(ARITH_MODULE, arith_kernels::ARITH_CAST)
            .ok_or_else(|| XlogError::Kernel("arith_cast not found".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                config,
                (
                    src_col,
                    &mut out,
                    n,
                    source_type.to_code(),
                    target.to_code(),
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("cast failed: {}", e)))?;

        self.device.synchronize()?;

        self.buffer_from_columns_with_device_count(vec![out.into()], a.num_rows(), schema, a)
    }

    /// Helper for binary arithmetic operations on device.
    fn binary_arith_op_device<T: DeviceRepr>(
        &self,
        a: &CudaBuffer,
        b: &CudaBuffer,
        op: u8,
        kernel: &str,
    ) -> Result<CudaBuffer> {
        if a.num_rows() != b.num_rows() {
            return Err(XlogError::Kernel("Row count mismatch".into()));
        }
        if a.arity() != 1 || b.arity() != 1 {
            return Err(XlogError::Kernel(
                "Arithmetic requires single-column buffers".into(),
            ));
        }
        if a.schema().column_type(0) != b.schema().column_type(0) {
            return Err(XlogError::Kernel(
                "Arithmetic requires matching column types".into(),
            ));
        }
        if a.num_rows() == 0 {
            return self.create_empty_buffer(a.schema.clone());
        }

        let n: u32 = a.num_rows().try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "arith: row count {} exceeds u32::MAX",
                a.num_rows()
            ))
        })?;

        let expected_bytes = (n as usize)
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| XlogError::Kernel("arith output size overflow".into()))?;

        let col_a = a
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing column 0".into()))?;
        let col_b = b
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing column 0".into()))?;

        if col_a.num_bytes() != expected_bytes || col_b.num_bytes() != expected_bytes {
            return Err(XlogError::Kernel(format!(
                "Arithmetic expects {} bytes per column for {} rows",
                expected_bytes,
                a.num_rows()
            )));
        }

        let mut out = self.memory.alloc::<u8>(expected_bytes)?;
        let func = self
            .device
            .inner()
            .get_func(ARITH_MODULE, kernel)
            .ok_or_else(|| XlogError::Kernel("arith kernel not found".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe { func.clone().launch(config, (col_a, col_b, n, op, &mut out)) }
            .map_err(|e| XlogError::Kernel(format!("arith binary failed: {}", e)))?;

        self.device.synchronize()?;
        self.buffer_from_columns_with_device_count(
            vec![out.into()],
            a.num_rows(),
            a.schema.clone(),
            a,
        )
    }

    /// Combine multiple single-column buffers into a multi-column buffer
    ///
    /// # Arguments
    /// * `columns` - Vector of single-column CudaBuffers to combine
    /// * `types` - Vector of ScalarTypes for each column
    ///
    /// # Returns
    /// A new CudaBuffer with all columns combined
    pub fn combine_columns(
        &self,
        columns: Vec<CudaBuffer>,
        types: Vec<ScalarType>,
    ) -> Result<CudaBuffer> {
        if columns.is_empty() {
            let schema_cols: Vec<(String, ScalarType)> = types
                .iter()
                .enumerate()
                .map(|(i, t)| (format!("col_{}", i), *t))
                .collect();
            let schema = Schema::new(schema_cols);
            return self.create_empty_buffer(schema);
        }

        let row_cap = columns[0].row_cap;

        // Verify all columns have the same row capacity and are single-column
        for (i, col) in columns.iter().enumerate() {
            if col.row_cap != row_cap {
                return Err(XlogError::Kernel(format!(
                    "Column {} has row capacity {}, expected {}",
                    i, col.row_cap, row_cap
                )));
            }
            if col.arity() != 1 {
                return Err(XlogError::Kernel(format!(
                    "Column {} buffer has {} columns, expected 1",
                    i,
                    col.arity()
                )));
            }
        }

        let device = self.device.inner();
        let mut d_num_rows = self.memory.alloc::<u32>(1)?;
        device
            .dtod_copy(columns[0].num_rows_device(), &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy row count: {}", e)))?;

        let mut result_columns = Vec::with_capacity(columns.len());
        for (i, col_buf) in columns.into_iter().enumerate() {
            let src_col = col_buf
                .columns
                .into_iter()
                .next()
                .ok_or_else(|| XlogError::Kernel(format!("Column {} buffer has no data", i)))?;
            result_columns.push(src_col);
        }

        let schema_cols: Vec<(String, ScalarType)> = types
            .iter()
            .enumerate()
            .map(|(i, t)| (format!("col_{}", i), *t))
            .collect();
        let schema = Schema::new(schema_cols);

        Ok(CudaBuffer::from_columns(
            result_columns,
            row_cap,
            d_num_rows,
            schema,
        ))
    }
}
