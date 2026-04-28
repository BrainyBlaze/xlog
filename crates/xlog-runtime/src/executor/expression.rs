//! Expression evaluation methods for the Executor.
//!
//! Production GPU-accelerated filter, predicate mask, arithmetic expression,
//! and mask operation methods.

use cudarc::driver::LaunchConfig;
use xlog_core::{Result, ScalarType, Schema, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{arith_kernels, filter_kernels, ARITH_MODULE, FILTER_MODULE};
use xlog_cuda::{CudaBuffer, LaunchAsync};
use xlog_ir::{CompareOp, ConstValue, Expr, ProjectExpr};

use super::Executor;

impl Executor {
    /// Check if an expression may produce a floating-point result.
    pub(crate) fn expr_may_be_float(expr: &Expr, schema: &Schema) -> bool {
        match expr {
            Expr::Column(col_idx) => matches!(
                schema.column_type(*col_idx),
                Some(ScalarType::F32 | ScalarType::F64)
            ),
            Expr::Const(ConstValue::F32(_) | ConstValue::F64(_)) => true,
            Expr::Cast(_, ScalarType::F32 | ScalarType::F64) => true,
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Mod(l, r)
            | Expr::Min(l, r)
            | Expr::Max(l, r)
            | Expr::Pow(l, r) => {
                Self::expr_may_be_float(l, schema) || Self::expr_may_be_float(r, schema)
            }
            Expr::Abs(inner) | Expr::Cast(inner, _) => Self::expr_may_be_float(inner, schema),
            _ => false,
        }
    }

    /// Execute a Filter node using GPU predicate evaluation.
    pub fn execute_filter(&self, input: &CudaBuffer, predicate: &Expr) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema().clone());
        }

        let mask = self.eval_predicate_mask_gpu(predicate, input)?;
        self.provider.filter_by_device_mask(input, &mask)
    }

    pub(crate) fn eval_predicate_mask_gpu(
        &self,
        expr: &Expr,
        input: &CudaBuffer,
    ) -> Result<TrackedCudaSlice<u8>> {
        if input.num_rows() > u32::MAX as u64 {
            return Err(XlogError::Execution(format!(
                "Predicate evaluation supports at most {} rows, got {}",
                u32::MAX,
                input.num_rows()
            )));
        }
        let n = input.num_rows() as u32;

        match expr {
            Expr::Column(col_idx) => {
                let col_type = input
                    .schema()
                    .column_type(*col_idx)
                    .ok_or_else(|| XlogError::Execution(format!("Column {} not found", col_idx)))?;
                if col_type == ScalarType::Bool {
                    let col_buf = self.wrap_single_column(input, *col_idx)?;
                    let zero = self.provider.create_constant_column_with_device_count(
                        &[0u8],
                        ScalarType::Bool,
                        input.num_rows(),
                        input.num_rows_device(),
                    )?;
                    return self.compare_buffers_mask(&col_buf, &zero, CompareOp::Ne);
                }
                self.mask_filled(n, 1)
            }
            Expr::Const(ConstValue::Bool(b)) => self.mask_filled(n, if *b { 1 } else { 0 }),
            Expr::Const(_) => self.mask_filled(n, 1),
            Expr::Compare { left, op, right } => {
                let use_float = Self::expr_may_be_float(left, input.schema())
                    || Self::expr_may_be_float(right, input.schema());

                let mut left_buf = self.evaluate_arith_expr(left, input)?;
                let mut right_buf = self.evaluate_arith_expr(right, input)?;

                if use_float {
                    left_buf = self.provider.cast_column(&left_buf, ScalarType::F64)?;
                    right_buf = self.provider.cast_column(&right_buf, ScalarType::F64)?;
                }

                self.compare_buffers_mask(&left_buf, &right_buf, *op)
            }
            Expr::And(exprs) => {
                if exprs.is_empty() {
                    return self.mask_filled(n, 1);
                }
                let mut mask = self.eval_predicate_mask_gpu(&exprs[0], input)?;
                for expr in &exprs[1..] {
                    let next = self.eval_predicate_mask_gpu(expr, input)?;
                    mask = self.mask_and(&mask, &next, n)?;
                }
                Ok(mask)
            }
            Expr::Or(exprs) => {
                if exprs.is_empty() {
                    return self.mask_filled(n, 0);
                }
                let mut mask = self.eval_predicate_mask_gpu(&exprs[0], input)?;
                for expr in &exprs[1..] {
                    let next = self.eval_predicate_mask_gpu(expr, input)?;
                    mask = self.mask_or(&mask, &next, n)?;
                }
                Ok(mask)
            }
            Expr::Not(inner) => {
                let mask = self.eval_predicate_mask_gpu(inner, input)?;
                self.mask_not(&mask, n)
            }
            Expr::Add(_, _)
            | Expr::Sub(_, _)
            | Expr::Mul(_, _)
            | Expr::Div(_, _)
            | Expr::Mod(_, _)
            | Expr::Abs(_)
            | Expr::Min(_, _)
            | Expr::Max(_, _)
            | Expr::Pow(_, _)
            | Expr::Cast(_, _)
            | Expr::Conditional { .. } => Err(XlogError::Execution(
                "Arithmetic expression cannot be evaluated as boolean predicate".into(),
            )),
        }
    }

    fn compare_buffers_mask(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        op: CompareOp,
    ) -> Result<TrackedCudaSlice<u8>> {
        if left.arity() != 1 || right.arity() != 1 {
            return Err(XlogError::Execution(
                "Compare requires single-column buffers".into(),
            ));
        }
        if left.num_rows() != right.num_rows() {
            return Err(XlogError::Execution(
                "Compare requires matching row counts".into(),
            ));
        }
        if left.num_rows() > u32::MAX as u64 {
            return Err(XlogError::Execution(format!(
                "Compare supports at most {} rows, got {}",
                u32::MAX,
                left.num_rows()
            )));
        }
        if left.is_empty() {
            return self.provider.memory().alloc::<u8>(0).map_err(|e| {
                XlogError::execution_ctx("compare_buffers_mask", "allocate empty mask", &e)
            });
        }

        let left_type = left
            .schema()
            .column_type(0)
            .ok_or_else(|| XlogError::Execution("Missing left column type".into()))?;
        let right_type = right
            .schema()
            .column_type(0)
            .ok_or_else(|| XlogError::Execution("Missing right column type".into()))?;

        if left_type != right_type {
            return Err(XlogError::Execution(
                "Compare requires matching column types".into(),
            ));
        }

        let kernel = match left_type {
            ScalarType::U32 | ScalarType::Symbol => filter_kernels::FILTER_COMPARE_U32_COL,
            ScalarType::U64 => filter_kernels::FILTER_COMPARE_U64_COL,
            ScalarType::I32 => filter_kernels::FILTER_COMPARE_I32_COL,
            ScalarType::I64 => filter_kernels::FILTER_COMPARE_I64_COL,
            ScalarType::F32 => filter_kernels::FILTER_COMPARE_F32_COL,
            ScalarType::F64 => filter_kernels::FILTER_COMPARE_F64_COL,
            ScalarType::Bool => filter_kernels::FILTER_COMPARE_U8_COL,
        };

        let left_col = left
            .column(0)
            .ok_or_else(|| XlogError::Execution("Missing left column".into()))?;
        let right_col = right
            .column(0)
            .ok_or_else(|| XlogError::Execution("Missing right column".into()))?;

        // Use the host-known capacity as the launch grid; the kernel
        // clamps in-kernel via `num_rows_device` so any rows in
        // `[logical, capacity)` get mask=0 instead of consuming
        // uninitialized column bytes from a join's capacity-sized
        // output. This is the load-bearing fix for the
        // join-output-as-filter-input chain in v0.5.5.
        let row_cap = left.num_rows() as u32;
        let mut d_mask = self.provider.memory().alloc::<u8>(row_cap as usize)?;

        let func = self
            .provider
            .device()
            .inner()
            .get_func(FILTER_MODULE, kernel)
            .ok_or_else(|| XlogError::Execution("filter compare kernel not found".into()))?;
        let config = LaunchConfig::for_num_elems(row_cap);

        // SAFETY: kernel signature matches:
        //   filter_compare_*_col(left, right, row_cap, num_rows_device, op, mask)
        // — device buffers were allocated with sufficient size.
        unsafe {
            func.clone().launch(
                config,
                (
                    left_col,
                    right_col,
                    row_cap,
                    left.num_rows_device(),
                    op as u8,
                    &mut d_mask,
                ),
            )
        }
        .map_err(|e| XlogError::execution_ctx("compare_buffers_mask", "filter compare", &e))?;

        Ok(d_mask)
    }

    fn mask_and(
        &self,
        left: &TrackedCudaSlice<u8>,
        right: &TrackedCudaSlice<u8>,
        n: u32,
    ) -> Result<TrackedCudaSlice<u8>> {
        let mut out = self.provider.memory().alloc::<u8>(n as usize)?;
        if n == 0 {
            return Ok(out);
        }

        let func = self
            .provider
            .device()
            .inner()
            .get_func(FILTER_MODULE, filter_kernels::MASK_AND)
            .ok_or_else(|| XlogError::Execution("mask_and kernel not found".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe { func.clone().launch(config, (left, right, &mut out, n)) }
            .map_err(|e| XlogError::execution_ctx("mask_and", "launch kernel", &e))?;

        Ok(out)
    }

    fn mask_or(
        &self,
        left: &TrackedCudaSlice<u8>,
        right: &TrackedCudaSlice<u8>,
        n: u32,
    ) -> Result<TrackedCudaSlice<u8>> {
        let mut out = self.provider.memory().alloc::<u8>(n as usize)?;
        if n == 0 {
            return Ok(out);
        }

        let func = self
            .provider
            .device()
            .inner()
            .get_func(FILTER_MODULE, filter_kernels::MASK_OR)
            .ok_or_else(|| XlogError::Execution("mask_or kernel not found".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe { func.clone().launch(config, (left, right, &mut out, n)) }
            .map_err(|e| XlogError::execution_ctx("mask_or", "launch kernel", &e))?;

        Ok(out)
    }

    fn mask_not(&self, input: &TrackedCudaSlice<u8>, n: u32) -> Result<TrackedCudaSlice<u8>> {
        let mut out = self.provider.memory().alloc::<u8>(n as usize)?;
        if n == 0 {
            return Ok(out);
        }

        let func = self
            .provider
            .device()
            .inner()
            .get_func(FILTER_MODULE, filter_kernels::MASK_NOT)
            .ok_or_else(|| XlogError::Execution("mask_not kernel not found".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe { func.clone().launch(config, (input, &mut out, n)) }
            .map_err(|e| XlogError::execution_ctx("mask_not", "launch kernel", &e))?;

        Ok(out)
    }

    fn mask_filled(&self, n: u32, value: u8) -> Result<TrackedCudaSlice<u8>> {
        let mut out = self.provider.memory().alloc::<u8>(n as usize)?;
        if n == 0 {
            return Ok(out);
        }

        if value == 0 {
            self.provider
                .device()
                .inner()
                .memset_zeros(&mut out)
                .map_err(|e| XlogError::execution_ctx("mask_filled", "mask memset", &e))?;
            return Ok(out);
        }

        let func = self
            .provider
            .device()
            .inner()
            .get_func(ARITH_MODULE, arith_kernels::ARITH_FILL_CONST_U8)
            .ok_or_else(|| XlogError::Execution("arith fill kernel not found".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe { func.clone().launch(config, (value, n, &mut out)) }
            .map_err(|e| XlogError::execution_ctx("mask_filled", "mask fill", &e))?;

        Ok(out)
    }

    pub(crate) fn wrap_single_column(
        &self,
        buffer: &CudaBuffer,
        col_idx: usize,
    ) -> Result<CudaBuffer> {
        let col_type = buffer
            .schema()
            .column_type(col_idx)
            .ok_or_else(|| XlogError::Execution(format!("Column {} not found", col_idx)))?;
        let schema = Schema::new(vec![("expr".to_string(), col_type)]);

        if buffer.is_empty() {
            return self.create_empty_buffer(schema);
        }

        let num_rows = buffer.num_rows();
        let bytes = (num_rows as usize)
            .checked_mul(col_type.size_bytes())
            .ok_or_else(|| XlogError::Execution("Column size overflow".into()))?;

        let src_col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::Execution(format!("Column {} not found", col_idx)))?;
        let mut dst_col = self.provider.memory().alloc::<u8>(bytes)?;
        if bytes > 0 {
            self.provider
                .device()
                .inner()
                .dtod_copy(src_col, &mut dst_col)
                .map_err(|e| XlogError::execution_ctx("wrap_single_column", "copy column", &e))?;
        }

        let d_num_rows = self.clone_device_row_count(buffer)?;
        self.provider.device().synchronize()?;
        Ok(CudaBuffer::from_columns(
            vec![dst_col.into()],
            num_rows,
            d_num_rows,
            schema,
        ))
    }

    /// Evaluate an arithmetic expression on a buffer, producing a single-column result
    ///
    /// This method recursively evaluates arithmetic expressions (Add, Sub, Mul, Div, etc.)
    /// by delegating to the CUDA kernel provider for GPU-accelerated operations.
    pub(crate) fn evaluate_arith_expr(
        &self,
        expr: &Expr,
        input: &CudaBuffer,
    ) -> Result<CudaBuffer> {
        match expr {
            Expr::Column(idx) => {
                // Extract the column as a single-column buffer without host round-trip
                self.wrap_single_column(input, *idx)
            }
            Expr::Const(val) => {
                // Create a column filled with the constant value
                let (bytes, col_type) = self.const_to_bytes_and_type(val);
                self.provider.create_constant_column_with_device_count(
                    &bytes,
                    col_type,
                    input.num_rows(),
                    input.num_rows_device(),
                )
            }
            Expr::Add(l, r) => {
                let left = self.evaluate_arith_expr(l, input)?;
                let right = self.evaluate_arith_expr(r, input)?;
                self.provider.add_columns(&left, &right)
            }
            Expr::Sub(l, r) => {
                let left = self.evaluate_arith_expr(l, input)?;
                let right = self.evaluate_arith_expr(r, input)?;
                self.provider.sub_columns(&left, &right)
            }
            Expr::Mul(l, r) => {
                let left = self.evaluate_arith_expr(l, input)?;
                let right = self.evaluate_arith_expr(r, input)?;
                self.provider.mul_columns(&left, &right)
            }
            Expr::Div(l, r) => {
                let left = self.evaluate_arith_expr(l, input)?;
                let right = self.evaluate_arith_expr(r, input)?;
                self.provider.div_columns(&left, &right)
            }
            Expr::Mod(l, r) => {
                let left = self.evaluate_arith_expr(l, input)?;
                let right = self.evaluate_arith_expr(r, input)?;
                self.provider.mod_columns(&left, &right)
            }
            Expr::Abs(inner) => {
                let val = self.evaluate_arith_expr(inner, input)?;
                self.provider.abs_column(&val)
            }
            Expr::Min(l, r) => {
                let left = self.evaluate_arith_expr(l, input)?;
                let right = self.evaluate_arith_expr(r, input)?;
                self.provider.min_columns(&left, &right)
            }
            Expr::Max(l, r) => {
                let left = self.evaluate_arith_expr(l, input)?;
                let right = self.evaluate_arith_expr(r, input)?;
                self.provider.max_columns(&left, &right)
            }
            Expr::Pow(base, exp) => {
                let base_buf = self.evaluate_arith_expr(base, input)?;
                let exp_buf = self.evaluate_arith_expr(exp, input)?;
                self.provider.pow_columns(&base_buf, &exp_buf)
            }
            Expr::Cast(inner, target_type) => {
                let val = self.evaluate_arith_expr(inner, input)?;
                self.provider.cast_column(&val, *target_type)
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                // Evaluate condition to get boolean mask
                let mask_slice = self.eval_predicate_mask_gpu(condition, input)?;

                // Convert mask slice to a CudaBuffer for select_columns
                let d_num_rows = self.clone_device_row_count(input)?;
                let mask_buffer = CudaBuffer::from_columns(
                    vec![mask_slice.into()],
                    input.num_rows(),
                    d_num_rows,
                    Schema::new(vec![("mask".to_string(), ScalarType::Bool)]),
                );

                // Evaluate both branches
                let then_buf = self.evaluate_arith_expr(then_expr, input)?;
                let else_buf = self.evaluate_arith_expr(else_expr, input)?;

                // Select based on mask
                self.provider
                    .select_columns(&mask_buffer, &then_buf, &else_buf)
            }
            _ => Err(XlogError::Execution(format!(
                "Unsupported expression in arithmetic evaluation: {:?}",
                expr
            ))),
        }
    }

    /// Convert a ConstValue to raw bytes and ScalarType
    pub(crate) fn const_to_bytes_and_type(&self, val: &ConstValue) -> (Vec<u8>, ScalarType) {
        match val {
            ConstValue::U32(v) => (v.to_le_bytes().to_vec(), ScalarType::U32),
            ConstValue::U64(v) => (v.to_le_bytes().to_vec(), ScalarType::U64),
            ConstValue::I32(v) => (v.to_le_bytes().to_vec(), ScalarType::I32),
            ConstValue::I64(v) => (v.to_le_bytes().to_vec(), ScalarType::I64),
            ConstValue::F32(v) => (v.to_le_bytes().to_vec(), ScalarType::F32),
            ConstValue::F64(v) => (v.to_le_bytes().to_vec(), ScalarType::F64),
            ConstValue::Bool(v) => (vec![if *v { 1u8 } else { 0u8 }], ScalarType::Bool),
            ConstValue::Symbol(s) => (
                xlog_core::symbol::intern(s).to_le_bytes().to_vec(),
                ScalarType::Symbol,
            ),
        }
    }

    /// Execute a Project node
    ///
    /// Selects and reorders columns according to the projection list.
    /// Supports both column pass-through and computed expressions.
    pub(crate) fn execute_project(
        &self,
        input: &CudaBuffer,
        columns: &[ProjectExpr],
    ) -> Result<CudaBuffer> {
        if input.is_empty() {
            // Build projected schema
            let projected_schema = self.project_schema(input.schema(), columns)?;
            return self.create_empty_buffer(projected_schema);
        }

        // Build result columns as single-column CudaBuffers
        let mut result_buffers: Vec<CudaBuffer> = Vec::with_capacity(columns.len());
        let mut result_types: Vec<ScalarType> = Vec::with_capacity(columns.len());

        for proj_expr in columns {
            match proj_expr {
                ProjectExpr::Column(col_idx) => {
                    // Use extract_column to get a single-column buffer
                    let col_buffer = self.provider.extract_column(input, *col_idx)?;
                    let col_type = input
                        .schema()
                        .column_type(*col_idx)
                        .unwrap_or(ScalarType::U64);
                    result_types.push(col_type);
                    result_buffers.push(col_buffer);
                }
                ProjectExpr::Computed(expr, result_type) => {
                    // Evaluate the arithmetic expression to get a single-column buffer
                    let computed_buffer = self.evaluate_arith_expr(expr, input)?;
                    result_types.push(*result_type);
                    result_buffers.push(computed_buffer);
                }
            }
        }

        // Combine all single-column buffers into a multi-column buffer
        self.provider.combine_columns(result_buffers, result_types)
    }

    /// Build a projected schema from ProjectExpr list
    pub(crate) fn project_schema(&self, input: &Schema, columns: &[ProjectExpr]) -> Result<Schema> {
        let mut projected_columns: Vec<(String, ScalarType)> = Vec::with_capacity(columns.len());
        for proj_expr in columns {
            match proj_expr {
                ProjectExpr::Column(col_idx) => {
                    if let Some((name, ty)) = input.columns.get(*col_idx) {
                        projected_columns.push((name.clone(), *ty));
                    } else {
                        return Err(XlogError::Execution(format!(
                            "Column index {} out of bounds",
                            col_idx
                        )));
                    }
                }
                ProjectExpr::Computed(_expr, result_type) => {
                    // Computed columns get a generated name
                    let col_name = format!("computed_{}", projected_columns.len());
                    projected_columns.push((col_name, *result_type));
                }
            }
        }
        Ok(Schema::new(projected_columns))
    }
}
