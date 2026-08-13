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

#[derive(Clone, Copy)]
enum ArithmeticBinaryOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Minimum,
    Maximum,
    Power,
}

#[derive(Clone, Copy)]
enum MaskBinaryOperation {
    And,
    Or,
}

enum ExpressionTask<'a> {
    Arithmetic(&'a Expr),
    Predicate(&'a Expr),
    FinishArithmeticBinary(ArithmeticBinaryOperation),
    FinishAbsoluteValue,
    FinishCast(ScalarType),
    FinishComparison {
        op: CompareOp,
        use_float: bool,
    },
    ContinueMaskFold {
        expressions: &'a [Expr],
        next_index: usize,
        operation: MaskBinaryOperation,
    },
    FinishMaskFold {
        expressions: &'a [Expr],
        next_index: usize,
        operation: MaskBinaryOperation,
    },
    FinishMaskNot,
    PrepareConditional {
        then_expr: &'a Expr,
        else_expr: &'a Expr,
    },
    FinishConditional,
}

enum ExpressionValue {
    Arithmetic(CudaBuffer),
    Predicate(TrackedCudaSlice<u8>),
    SelectionMask(CudaBuffer),
}

impl Executor {
    /// Check if an expression may produce a floating-point result.
    pub(crate) fn expr_may_be_float(expr: &Expr, schema: &Schema) -> bool {
        let mut pending = vec![expr];
        while let Some(current) = pending.pop() {
            match current {
                Expr::Column(col_idx)
                    if matches!(
                        schema.column_type(*col_idx),
                        Some(ScalarType::F32 | ScalarType::F64)
                    ) =>
                {
                    return true;
                }
                Expr::Const(ConstValue::F32(_) | ConstValue::F64(_))
                | Expr::Cast(_, ScalarType::F32 | ScalarType::F64) => return true,
                Expr::Add(left, right)
                | Expr::Sub(left, right)
                | Expr::Mul(left, right)
                | Expr::Div(left, right)
                | Expr::Mod(left, right)
                | Expr::Min(left, right)
                | Expr::Max(left, right)
                | Expr::Pow(left, right) => {
                    pending.push(right);
                    pending.push(left);
                }
                Expr::Abs(inner) | Expr::Cast(inner, _) => pending.push(inner),
                _ => {}
            }
        }
        false
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
        match self.evaluate_expression(expr, input, true)? {
            ExpressionValue::Predicate(mask) => Ok(mask),
            _ => Err(Self::expression_state_error(
                "predicate evaluation produced an arithmetic value",
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

        let num_rows = left.num_rows() as u32;
        let mut d_mask = self.provider.memory().alloc::<u8>(num_rows as usize)?;

        let func = self
            .provider
            .device()
            .inner()
            .get_func(FILTER_MODULE, kernel)
            .ok_or_else(|| XlogError::Execution("filter compare kernel not found".into()))?;
        let config = LaunchConfig::for_num_elems(num_rows);

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                config,
                (left_col, right_col, num_rows, op as u8, &mut d_mask),
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

    /// Evaluate an arithmetic expression on a buffer, producing a single-column result.
    ///
    /// The explicit task stack preserves source-order evaluation without consuming
    /// one native stack frame per nested expression.
    pub(crate) fn evaluate_arith_expr(
        &self,
        expr: &Expr,
        input: &CudaBuffer,
    ) -> Result<CudaBuffer> {
        match self.evaluate_expression(expr, input, false)? {
            ExpressionValue::Arithmetic(buffer) => Ok(buffer),
            _ => Err(Self::expression_state_error(
                "arithmetic evaluation produced a predicate value",
            )),
        }
    }

    fn evaluate_expression(
        &self,
        expression: &Expr,
        input: &CudaBuffer,
        predicate: bool,
    ) -> Result<ExpressionValue> {
        let mut tasks = vec![if predicate {
            ExpressionTask::Predicate(expression)
        } else {
            ExpressionTask::Arithmetic(expression)
        }];
        let mut values = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                ExpressionTask::Arithmetic(expression) => match expression {
                    Expr::Column(index) => {
                        values.push(ExpressionValue::Arithmetic(
                            self.wrap_single_column(input, *index)?,
                        ));
                    }
                    Expr::Const(value) => {
                        let (bytes, column_type) = self.const_to_bytes_and_type(value);
                        values.push(ExpressionValue::Arithmetic(
                            self.provider.create_constant_column_with_device_count(
                                &bytes,
                                column_type,
                                input.num_rows(),
                                input.num_rows_device(),
                            )?,
                        ));
                    }
                    Expr::Add(left, right) => Self::schedule_arithmetic_binary(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticBinaryOperation::Add,
                    ),
                    Expr::Sub(left, right) => Self::schedule_arithmetic_binary(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticBinaryOperation::Subtract,
                    ),
                    Expr::Mul(left, right) => Self::schedule_arithmetic_binary(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticBinaryOperation::Multiply,
                    ),
                    Expr::Div(left, right) => Self::schedule_arithmetic_binary(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticBinaryOperation::Divide,
                    ),
                    Expr::Mod(left, right) => Self::schedule_arithmetic_binary(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticBinaryOperation::Modulo,
                    ),
                    Expr::Min(left, right) => Self::schedule_arithmetic_binary(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticBinaryOperation::Minimum,
                    ),
                    Expr::Max(left, right) => Self::schedule_arithmetic_binary(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticBinaryOperation::Maximum,
                    ),
                    Expr::Pow(left, right) => Self::schedule_arithmetic_binary(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticBinaryOperation::Power,
                    ),
                    Expr::Abs(inner) => {
                        tasks.push(ExpressionTask::FinishAbsoluteValue);
                        tasks.push(ExpressionTask::Arithmetic(inner));
                    }
                    Expr::Cast(inner, target) => {
                        tasks.push(ExpressionTask::FinishCast(*target));
                        tasks.push(ExpressionTask::Arithmetic(inner));
                    }
                    Expr::Conditional {
                        condition,
                        then_expr,
                        else_expr,
                    } => {
                        tasks.push(ExpressionTask::PrepareConditional {
                            then_expr,
                            else_expr,
                        });
                        tasks.push(ExpressionTask::Predicate(condition));
                    }
                    Expr::Compare { .. } | Expr::And(_) | Expr::Or(_) | Expr::Not(_) => {
                        return Err(XlogError::Execution(format!(
                            "Unsupported expression in arithmetic evaluation: {:?}",
                            expression
                        )));
                    }
                },
                ExpressionTask::Predicate(expression) => {
                    if input.num_rows() > u32::MAX as u64 {
                        return Err(XlogError::Execution(format!(
                            "Predicate evaluation supports at most {} rows, got {}",
                            u32::MAX,
                            input.num_rows()
                        )));
                    }
                    let row_count = input.num_rows() as u32;
                    match expression {
                        Expr::Column(column_index) => {
                            let column_type =
                                input.schema().column_type(*column_index).ok_or_else(|| {
                                    XlogError::Execution(format!(
                                        "Column {} not found",
                                        column_index
                                    ))
                                })?;
                            let mask = if column_type == ScalarType::Bool {
                                let column = self.wrap_single_column(input, *column_index)?;
                                let zero = self.provider.create_constant_column_with_device_count(
                                    &[0u8],
                                    ScalarType::Bool,
                                    input.num_rows(),
                                    input.num_rows_device(),
                                )?;
                                self.compare_buffers_mask(&column, &zero, CompareOp::Ne)?
                            } else {
                                self.mask_filled(row_count, 1)?
                            };
                            values.push(ExpressionValue::Predicate(mask));
                        }
                        Expr::Const(ConstValue::Bool(value)) => {
                            values.push(ExpressionValue::Predicate(
                                self.mask_filled(row_count, u8::from(*value))?,
                            ))
                        }
                        Expr::Const(_) => {
                            values.push(ExpressionValue::Predicate(self.mask_filled(row_count, 1)?))
                        }
                        Expr::Compare { left, op, right } => {
                            let use_float = Self::expr_may_be_float(left, input.schema())
                                || Self::expr_may_be_float(right, input.schema());
                            tasks.push(ExpressionTask::FinishComparison { op: *op, use_float });
                            tasks.push(ExpressionTask::Arithmetic(right));
                            tasks.push(ExpressionTask::Arithmetic(left));
                        }
                        Expr::And(expressions) => Self::schedule_mask_fold(
                            &mut tasks,
                            &mut values,
                            expressions,
                            row_count,
                            MaskBinaryOperation::And,
                            self,
                        )?,
                        Expr::Or(expressions) => Self::schedule_mask_fold(
                            &mut tasks,
                            &mut values,
                            expressions,
                            row_count,
                            MaskBinaryOperation::Or,
                            self,
                        )?,
                        Expr::Not(inner) => {
                            tasks.push(ExpressionTask::FinishMaskNot);
                            tasks.push(ExpressionTask::Predicate(inner));
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
                        | Expr::Conditional { .. } => {
                            return Err(XlogError::Execution(
                                "Arithmetic expression cannot be evaluated as boolean predicate"
                                    .into(),
                            ));
                        }
                    }
                }
                ExpressionTask::FinishArithmeticBinary(operation) => {
                    let right = Self::pop_arithmetic_value(&mut values)?;
                    let left = Self::pop_arithmetic_value(&mut values)?;
                    let result = match operation {
                        ArithmeticBinaryOperation::Add => self.provider.add_columns(&left, &right),
                        ArithmeticBinaryOperation::Subtract => {
                            self.provider.sub_columns(&left, &right)
                        }
                        ArithmeticBinaryOperation::Multiply => {
                            self.provider.mul_columns(&left, &right)
                        }
                        ArithmeticBinaryOperation::Divide => {
                            self.provider.div_columns(&left, &right)
                        }
                        ArithmeticBinaryOperation::Modulo => {
                            self.provider.mod_columns(&left, &right)
                        }
                        ArithmeticBinaryOperation::Minimum => {
                            self.provider.min_columns(&left, &right)
                        }
                        ArithmeticBinaryOperation::Maximum => {
                            self.provider.max_columns(&left, &right)
                        }
                        ArithmeticBinaryOperation::Power => {
                            self.provider.pow_columns(&left, &right)
                        }
                    }?;
                    values.push(ExpressionValue::Arithmetic(result));
                }
                ExpressionTask::FinishAbsoluteValue => {
                    let value = Self::pop_arithmetic_value(&mut values)?;
                    values.push(ExpressionValue::Arithmetic(
                        self.provider.abs_column(&value)?,
                    ));
                }
                ExpressionTask::FinishCast(target) => {
                    let value = Self::pop_arithmetic_value(&mut values)?;
                    values.push(ExpressionValue::Arithmetic(
                        self.provider.cast_column(&value, target)?,
                    ));
                }
                ExpressionTask::FinishComparison { op, use_float } => {
                    let mut right = Self::pop_arithmetic_value(&mut values)?;
                    let mut left = Self::pop_arithmetic_value(&mut values)?;
                    if use_float {
                        left = self.provider.cast_column(&left, ScalarType::F64)?;
                        right = self.provider.cast_column(&right, ScalarType::F64)?;
                    }
                    values.push(ExpressionValue::Predicate(
                        self.compare_buffers_mask(&left, &right, op)?,
                    ));
                }
                ExpressionTask::ContinueMaskFold {
                    expressions,
                    next_index,
                    operation,
                } => {
                    if next_index < expressions.len() {
                        tasks.push(ExpressionTask::FinishMaskFold {
                            expressions,
                            next_index: next_index + 1,
                            operation,
                        });
                        tasks.push(ExpressionTask::Predicate(&expressions[next_index]));
                    }
                }
                ExpressionTask::FinishMaskFold {
                    expressions,
                    next_index,
                    operation,
                } => {
                    let right = Self::pop_predicate_value(&mut values)?;
                    let left = Self::pop_predicate_value(&mut values)?;
                    let row_count = input.num_rows() as u32;
                    let combined = match operation {
                        MaskBinaryOperation::And => self.mask_and(&left, &right, row_count),
                        MaskBinaryOperation::Or => self.mask_or(&left, &right, row_count),
                    }?;
                    values.push(ExpressionValue::Predicate(combined));
                    tasks.push(ExpressionTask::ContinueMaskFold {
                        expressions,
                        next_index,
                        operation,
                    });
                }
                ExpressionTask::FinishMaskNot => {
                    let mask = Self::pop_predicate_value(&mut values)?;
                    values.push(ExpressionValue::Predicate(
                        self.mask_not(&mask, input.num_rows() as u32)?,
                    ));
                }
                ExpressionTask::PrepareConditional {
                    then_expr,
                    else_expr,
                } => {
                    let mask = Self::pop_predicate_value(&mut values)?;
                    let device_row_count = self.clone_device_row_count(input)?;
                    values.push(ExpressionValue::SelectionMask(CudaBuffer::from_columns(
                        vec![mask.into()],
                        input.num_rows(),
                        device_row_count,
                        Schema::new(vec![("mask".to_string(), ScalarType::Bool)]),
                    )));
                    tasks.push(ExpressionTask::FinishConditional);
                    tasks.push(ExpressionTask::Arithmetic(else_expr));
                    tasks.push(ExpressionTask::Arithmetic(then_expr));
                }
                ExpressionTask::FinishConditional => {
                    let else_value = Self::pop_arithmetic_value(&mut values)?;
                    let then_value = Self::pop_arithmetic_value(&mut values)?;
                    let mask = match values.pop() {
                        Some(ExpressionValue::SelectionMask(mask)) => mask,
                        _ => {
                            return Err(Self::expression_state_error(
                                "conditional evaluation is missing its selection mask",
                            ));
                        }
                    };
                    values.push(ExpressionValue::Arithmetic(self.provider.select_columns(
                        &mask,
                        &then_value,
                        &else_value,
                    )?));
                }
            }
        }

        if values.len() != 1 {
            return Err(Self::expression_state_error(
                "expression evaluation did not produce exactly one value",
            ));
        }
        values
            .pop()
            .ok_or_else(|| Self::expression_state_error("expression evaluation produced no value"))
    }

    fn schedule_arithmetic_binary<'a>(
        tasks: &mut Vec<ExpressionTask<'a>>,
        left: &'a Expr,
        right: &'a Expr,
        operation: ArithmeticBinaryOperation,
    ) {
        tasks.push(ExpressionTask::FinishArithmeticBinary(operation));
        tasks.push(ExpressionTask::Arithmetic(right));
        tasks.push(ExpressionTask::Arithmetic(left));
    }

    fn schedule_mask_fold<'a>(
        tasks: &mut Vec<ExpressionTask<'a>>,
        values: &mut Vec<ExpressionValue>,
        expressions: &'a [Expr],
        row_count: u32,
        operation: MaskBinaryOperation,
        executor: &Self,
    ) -> Result<()> {
        if expressions.is_empty() {
            let identity = match operation {
                MaskBinaryOperation::And => 1,
                MaskBinaryOperation::Or => 0,
            };
            values.push(ExpressionValue::Predicate(
                executor.mask_filled(row_count, identity)?,
            ));
        } else {
            tasks.push(ExpressionTask::ContinueMaskFold {
                expressions,
                next_index: 1,
                operation,
            });
            tasks.push(ExpressionTask::Predicate(&expressions[0]));
        }
        Ok(())
    }

    fn pop_arithmetic_value(values: &mut Vec<ExpressionValue>) -> Result<CudaBuffer> {
        match values.pop() {
            Some(ExpressionValue::Arithmetic(value)) => Ok(value),
            _ => Err(Self::expression_state_error(
                "arithmetic operation is missing an operand",
            )),
        }
    }

    fn pop_predicate_value(values: &mut Vec<ExpressionValue>) -> Result<TrackedCudaSlice<u8>> {
        match values.pop() {
            Some(ExpressionValue::Predicate(value)) => Ok(value),
            _ => Err(Self::expression_state_error(
                "predicate operation is missing an operand",
            )),
        }
    }

    fn expression_state_error(message: &str) -> XlogError {
        XlogError::Execution(format!("Internal expression evaluator error: {message}"))
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

        if columns.is_empty() {
            // A zero-column projection preserves row existence. Combining an empty
            // list of column buffers would manufacture a zero-row relation and make
            // ground negation treat every matching atom as absent.
            let projected_schema = self.project_schema(input.schema(), columns)?;
            let rows = self.provider.device_row_count(input)?;
            let rows = u32::try_from(rows).map_err(|_| {
                XlogError::Execution(format!(
                    "zero-column projection row count {rows} exceeds the GPU range"
                ))
            })?;
            return self
                .provider
                .create_zero_arity_buffer(projected_schema, rows);
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

        let projected_schema = self.project_schema(input.schema(), columns)?;
        let mut output = self
            .provider
            .combine_columns(result_buffers, result_types)?;
        output.set_schema(projected_schema);
        Ok(output)
    }

    /// Build a projected schema from ProjectExpr list
    pub(crate) fn project_schema(&self, input: &Schema, columns: &[ProjectExpr]) -> Result<Schema> {
        let mut projected_columns: Vec<(String, ScalarType)> = Vec::with_capacity(columns.len());
        let mut projected_sort_labels: Vec<String> = Vec::with_capacity(columns.len());
        for proj_expr in columns {
            match proj_expr {
                ProjectExpr::Column(col_idx) => {
                    if let Some((name, ty)) = input.columns.get(*col_idx) {
                        projected_columns.push((name.clone(), *ty));
                        projected_sort_labels.push(
                            input
                                .column_sort_label(*col_idx)
                                .unwrap_or(name)
                                .to_string(),
                        );
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
                    projected_sort_labels.push(format!("computed_{}", projected_sort_labels.len()));
                }
            }
        }
        Schema::new(projected_columns)
            .with_sort_labels(projected_sort_labels)
            .map_err(XlogError::Execution)
    }
}
