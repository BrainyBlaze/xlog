//! Stack-safe scalar evaluation for normalized arithmetic AST expressions.

use std::collections::HashMap;

use xlog_core::{symbol, Result, ScalarType, XlogError};
use xlog_ir::ConstValue;

use crate::ast::{ArithExpr, CompOp, Term};
use crate::lower::term_to_typed_const_value;

/// A typed scalar accepted by the arithmetic expression evaluator.
#[derive(Debug, Clone, PartialEq)]
pub enum ArithmeticValue {
    /// Signed 32-bit integer.
    I32(i32),
    /// Signed 64-bit integer.
    I64(i64),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// 32-bit floating-point value.
    F32(f32),
    /// 64-bit floating-point value.
    F64(f64),
    /// Boolean value.
    Bool(bool),
    /// Interned symbol identifier.
    Symbol(u32),
    /// String value.
    String(String),
}

impl ArithmeticValue {
    /// Convert an AST term into a scalar value.
    ///
    /// Integer and float terms use the AST literal types (`i64` and `f64`). Callers
    /// with schema information can construct a narrower typed variant directly.
    pub fn from_term(term: &Term) -> Result<Self> {
        match term {
            Term::Integer(value) => Ok(Self::I64(*value)),
            Term::Float(value) => Ok(Self::F64(*value)),
            Term::String(value) => Ok(Self::String(value.clone())),
            Term::Symbol(value) => Ok(Self::Symbol(*value)),
            Term::Variable(name) => Err(XlogError::Compilation(format!(
                "Unbound variable {name} in arithmetic"
            ))),
            Term::Anonymous => Err(XlogError::Compilation(
                "Anonymous variable not allowed in arithmetic".to_string(),
            )),
            Term::List(_)
            | Term::Cons { .. }
            | Term::Compound { .. }
            | Term::PredRef(_)
            | Term::Aggregate(_) => Err(XlogError::Compilation(
                "Arithmetic requires a scalar value".to_string(),
            )),
        }
    }

    /// Convert the evaluated scalar back into an AST term.
    pub fn into_term(self) -> Result<Term> {
        Ok(match self {
            Self::I32(value) => Term::Integer(i64::from(value)),
            Self::I64(value) => Term::Integer(value),
            Self::U32(value) => Term::Integer(i64::from(value)),
            Self::U64(value) => Term::Integer(i64::try_from(value).map_err(|_| {
                XlogError::Compilation("u64 arithmetic result exceeds AST integer range".into())
            })?),
            Self::F32(value) => Term::Float(f64::from(value)),
            Self::F64(value) => Term::Float(value),
            Self::Bool(value) => Term::Integer(i64::from(value)),
            Self::Symbol(value) => Term::Symbol(value),
            Self::String(value) => Term::String(value),
        })
    }

    fn kind(&self) -> ArithmeticValueKind {
        match self {
            Self::I32(_) => ArithmeticValueKind::I32,
            Self::I64(_) => ArithmeticValueKind::I64,
            Self::U32(_) => ArithmeticValueKind::U32,
            Self::U64(_) => ArithmeticValueKind::U64,
            Self::F32(_) => ArithmeticValueKind::F32,
            Self::F64(_) => ArithmeticValueKind::F64,
            Self::Bool(_) => ArithmeticValueKind::Bool,
            Self::Symbol(_) => ArithmeticValueKind::Symbol,
            Self::String(_) => ArithmeticValueKind::String,
        }
    }

    /// Return the runtime scalar type represented by this value.
    ///
    /// Source-only strings have no runtime scalar type until they are interned as
    /// symbols by [`ArithmeticValue::from_typed_term`].
    pub fn scalar_type(&self) -> Option<ScalarType> {
        match self.kind() {
            ArithmeticValueKind::I32 => Some(ScalarType::I32),
            ArithmeticValueKind::I64 => Some(ScalarType::I64),
            ArithmeticValueKind::U32 => Some(ScalarType::U32),
            ArithmeticValueKind::U64 => Some(ScalarType::U64),
            ArithmeticValueKind::F32 => Some(ScalarType::F32),
            ArithmeticValueKind::F64 => Some(ScalarType::F64),
            ArithmeticValueKind::Bool => Some(ScalarType::Bool),
            ArithmeticValueKind::Symbol => Some(ScalarType::Symbol),
            ArithmeticValueKind::String => None,
        }
    }

    /// Convert an AST scalar term using the same declared-type conversion as lowering.
    pub fn from_typed_term(term: &Term, expected: ScalarType) -> Result<Self> {
        let value = term_to_typed_const_value(term, expected)?.ok_or_else(|| {
            XlogError::Compilation("Arithmetic requires a bound scalar value".to_string())
        })?;
        Ok(match value {
            ConstValue::I32(value) => Self::I32(value),
            ConstValue::I64(value) => Self::I64(value),
            ConstValue::U32(value) => Self::U32(value),
            ConstValue::U64(value) => Self::U64(value),
            ConstValue::F32(value) => Self::F32(value),
            ConstValue::F64(value) => Self::F64(value),
            ConstValue::Bool(value) => Self::Bool(value),
            ConstValue::Symbol(value) => Self::Symbol(symbol::intern(&value)),
        })
    }

    fn as_f64(&self) -> Result<f64> {
        match self {
            Self::I32(value) => Ok(f64::from(*value)),
            Self::I64(value) => Ok(*value as f64),
            Self::U32(value) => Ok(f64::from(*value)),
            Self::U64(value) => Ok(*value as f64),
            Self::F32(value) => Ok(f64::from(*value)),
            Self::F64(value) => Ok(*value),
            Self::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
            Self::Symbol(value) => Ok(f64::from(*value)),
            Self::String(_) => Err(numeric_type_error()),
        }
    }

    fn cast(self, target: ScalarType) -> Result<Self> {
        if self.scalar_type() == Some(target) {
            return Ok(self);
        }
        match target {
            ScalarType::I32 => Ok(Self::I32(self.cast_i64()? as i32)),
            ScalarType::I64 => Ok(Self::I64(self.cast_i64()?)),
            ScalarType::U32 => Ok(Self::U32(self.cast_i64()? as u32)),
            ScalarType::U64 => Ok(Self::U64(self.cast_i64()? as u64)),
            ScalarType::F32 => Ok(Self::F32(self.as_f64()? as f32)),
            ScalarType::F64 => Ok(Self::F64(self.as_f64()?)),
            ScalarType::Bool => Ok(Self::Bool(self.cast_i64()? != 0)),
            ScalarType::Symbol => Ok(Self::Symbol(self.cast_i64()? as u32)),
        }
    }

    fn cast_i64(&self) -> Result<i64> {
        match self {
            Self::I32(value) => Ok(i64::from(*value)),
            Self::I64(value) => Ok(*value),
            Self::U32(value) => Ok(i64::from(*value)),
            Self::U64(value) => Ok(*value as i64),
            Self::F32(value) => float_to_i64(f64::from(*value)),
            Self::F64(value) => float_to_i64(*value),
            Self::Bool(value) => Ok(if *value { 1 } else { 0 }),
            Self::Symbol(value) => Ok(i64::from(*value)),
            Self::String(_) => Err(numeric_type_error()),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArithmeticValueKind {
    I32,
    I64,
    U32,
    U64,
    F32,
    F64,
    Bool,
    Symbol,
    String,
}

#[derive(Clone, Copy)]
enum BinaryOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Minimum,
    Maximum,
    Power,
}

enum EvaluationTask<'a> {
    Expression(&'a ArithExpr),
    FinishBinary(BinaryOperation),
    FinishAbsoluteValue,
    FinishCast(ScalarType),
    FinishComparison(CompOp),
    FinishConditional,
}

enum EvaluationValue {
    Arithmetic(ArithmeticValue),
    Predicate(bool),
}

/// Evaluate a normalized arithmetic expression without recursive AST traversal.
///
/// Binary operations require matching numeric types. Integer operations wrap on
/// overflow. Integer division by zero returns the maximum value of the operand type,
/// integer remainder by zero returns zero, and floating-point zero division follows
/// IEEE 754 with canonical positive NaNs. `pow` always returns `f64`. User-defined
/// calls must be expanded before this function is called.
pub fn evaluate_arithmetic_expression(
    expression: &ArithExpr,
    bindings: &HashMap<String, ArithmeticValue>,
) -> Result<ArithmeticValue> {
    let mut tasks = vec![EvaluationTask::Expression(expression)];
    let mut values = Vec::new();

    while let Some(task) = tasks.pop() {
        match task {
            EvaluationTask::Expression(expression) => match expression {
                ArithExpr::Variable(name) => values.push(EvaluationValue::Arithmetic(
                    bindings.get(name).cloned().ok_or_else(|| {
                        XlogError::Compilation(format!("Unbound variable {name} in arithmetic"))
                    })?,
                )),
                ArithExpr::Integer(value) => {
                    values.push(EvaluationValue::Arithmetic(ArithmeticValue::I64(*value)));
                }
                ArithExpr::Float(value) => {
                    values.push(EvaluationValue::Arithmetic(ArithmeticValue::F64(*value)));
                }
                ArithExpr::Add(left, right) => {
                    schedule_binary(&mut tasks, left, right, BinaryOperation::Add)
                }
                ArithExpr::Sub(left, right) => {
                    schedule_binary(&mut tasks, left, right, BinaryOperation::Subtract)
                }
                ArithExpr::Mul(left, right) => {
                    schedule_binary(&mut tasks, left, right, BinaryOperation::Multiply)
                }
                ArithExpr::Div(left, right) => {
                    schedule_binary(&mut tasks, left, right, BinaryOperation::Divide)
                }
                ArithExpr::Mod(left, right) => {
                    schedule_binary(&mut tasks, left, right, BinaryOperation::Modulo)
                }
                ArithExpr::Min(left, right) => {
                    schedule_binary(&mut tasks, left, right, BinaryOperation::Minimum)
                }
                ArithExpr::Max(left, right) => {
                    schedule_binary(&mut tasks, left, right, BinaryOperation::Maximum)
                }
                ArithExpr::Pow(left, right) => {
                    schedule_binary(&mut tasks, left, right, BinaryOperation::Power)
                }
                ArithExpr::Abs(inner) => {
                    tasks.push(EvaluationTask::FinishAbsoluteValue);
                    tasks.push(EvaluationTask::Expression(inner));
                }
                ArithExpr::Cast(inner, target) => {
                    tasks.push(EvaluationTask::FinishCast(*target));
                    tasks.push(EvaluationTask::Expression(inner));
                }
                ArithExpr::Conditional {
                    cond_left,
                    cond_op,
                    cond_right,
                    then_expr,
                    else_expr,
                } => {
                    tasks.push(EvaluationTask::FinishConditional);
                    tasks.push(EvaluationTask::Expression(else_expr));
                    tasks.push(EvaluationTask::Expression(then_expr));
                    tasks.push(EvaluationTask::FinishComparison(*cond_op));
                    tasks.push(EvaluationTask::Expression(cond_right));
                    tasks.push(EvaluationTask::Expression(cond_left));
                }
                ArithExpr::FuncCall { name, .. } => {
                    return Err(XlogError::Compilation(format!(
                        "Function call `{name}` must be expanded before arithmetic evaluation"
                    )));
                }
            },
            EvaluationTask::FinishBinary(operation) => {
                let right = pop_arithmetic(&mut values)?;
                let left = pop_arithmetic(&mut values)?;
                values.push(EvaluationValue::Arithmetic(evaluate_binary(
                    operation, left, right,
                )?));
            }
            EvaluationTask::FinishAbsoluteValue => {
                let value = pop_arithmetic(&mut values)?;
                values.push(EvaluationValue::Arithmetic(evaluate_abs(value)?));
            }
            EvaluationTask::FinishCast(target) => {
                let value = pop_arithmetic(&mut values)?;
                values.push(EvaluationValue::Arithmetic(value.cast(target)?));
            }
            EvaluationTask::FinishComparison(operator) => {
                let right = pop_arithmetic(&mut values)?;
                let left = pop_arithmetic(&mut values)?;
                values.push(EvaluationValue::Predicate(compare_arithmetic_values(
                    &left, operator, &right,
                )?));
            }
            EvaluationTask::FinishConditional => {
                let else_value = pop_arithmetic(&mut values)?;
                let then_value = pop_arithmetic(&mut values)?;
                let condition = pop_predicate(&mut values)?;
                if then_value.kind() != else_value.kind() {
                    return Err(XlogError::Compilation(
                        "Conditional branches require matching scalar types".to_string(),
                    ));
                }
                values.push(EvaluationValue::Arithmetic(if condition {
                    then_value
                } else {
                    else_value
                }));
            }
        }
    }

    let result = pop_arithmetic(&mut values)?;
    if values.is_empty() {
        Ok(result)
    } else {
        Err(evaluation_state_error())
    }
}

fn schedule_binary<'a>(
    tasks: &mut Vec<EvaluationTask<'a>>,
    left: &'a ArithExpr,
    right: &'a ArithExpr,
    operation: BinaryOperation,
) {
    tasks.push(EvaluationTask::FinishBinary(operation));
    tasks.push(EvaluationTask::Expression(right));
    tasks.push(EvaluationTask::Expression(left));
}

fn pop_arithmetic(values: &mut Vec<EvaluationValue>) -> Result<ArithmeticValue> {
    match values.pop() {
        Some(EvaluationValue::Arithmetic(value)) => Ok(value),
        Some(EvaluationValue::Predicate(_)) | None => Err(evaluation_state_error()),
    }
}

fn pop_predicate(values: &mut Vec<EvaluationValue>) -> Result<bool> {
    match values.pop() {
        Some(EvaluationValue::Predicate(value)) => Ok(value),
        Some(EvaluationValue::Arithmetic(_)) | None => Err(evaluation_state_error()),
    }
}

fn evaluate_binary(
    operation: BinaryOperation,
    left: ArithmeticValue,
    right: ArithmeticValue,
) -> Result<ArithmeticValue> {
    if matches!(operation, BinaryOperation::Power) {
        return Ok(ArithmeticValue::F64(normalize_nan_f64(
            left.as_f64()?.powf(right.as_f64()?),
        )));
    }
    if left.kind() != right.kind() {
        return Err(XlogError::Compilation(
            "Arithmetic operation requires matching numeric types".to_string(),
        ));
    }

    macro_rules! integer_operation {
        ($left:expr, $right:expr, $variant:ident, $type:ty) => {{
            let value = match operation {
                BinaryOperation::Add => $left.wrapping_add($right),
                BinaryOperation::Subtract => $left.wrapping_sub($right),
                BinaryOperation::Multiply => $left.wrapping_mul($right),
                BinaryOperation::Divide if $right == 0 => <$type>::MAX,
                BinaryOperation::Divide => $left.wrapping_div($right),
                BinaryOperation::Modulo if $right == 0 => 0,
                BinaryOperation::Modulo => $left.wrapping_rem($right),
                BinaryOperation::Minimum => $left.min($right),
                BinaryOperation::Maximum => $left.max($right),
                BinaryOperation::Power => unreachable!("power handled above"),
            };
            ArithmeticValue::$variant(value)
        }};
    }

    let value = match (left, right) {
        (ArithmeticValue::I32(left), ArithmeticValue::I32(right)) => {
            integer_operation!(left, right, I32, i32)
        }
        (ArithmeticValue::I64(left), ArithmeticValue::I64(right)) => {
            integer_operation!(left, right, I64, i64)
        }
        (ArithmeticValue::U32(left), ArithmeticValue::U32(right)) => {
            integer_operation!(left, right, U32, u32)
        }
        (ArithmeticValue::U64(left), ArithmeticValue::U64(right)) => {
            integer_operation!(left, right, U64, u64)
        }
        (ArithmeticValue::F32(left), ArithmeticValue::F32(right)) => {
            ArithmeticValue::F32(match operation {
                BinaryOperation::Add => left + right,
                BinaryOperation::Subtract => left - right,
                BinaryOperation::Multiply => left * right,
                BinaryOperation::Divide => normalize_nan_f32(left / right),
                BinaryOperation::Modulo => normalize_nan_f32(left % right),
                BinaryOperation::Minimum => {
                    if left < right {
                        left
                    } else {
                        right
                    }
                }
                BinaryOperation::Maximum => {
                    if left > right {
                        left
                    } else {
                        right
                    }
                }
                BinaryOperation::Power => unreachable!("power handled above"),
            })
        }
        (ArithmeticValue::F64(left), ArithmeticValue::F64(right)) => {
            ArithmeticValue::F64(match operation {
                BinaryOperation::Add => left + right,
                BinaryOperation::Subtract => left - right,
                BinaryOperation::Multiply => left * right,
                BinaryOperation::Divide => normalize_nan_f64(left / right),
                BinaryOperation::Modulo => normalize_nan_f64(left % right),
                BinaryOperation::Minimum => {
                    if left < right {
                        left
                    } else {
                        right
                    }
                }
                BinaryOperation::Maximum => {
                    if left > right {
                        left
                    } else {
                        right
                    }
                }
                BinaryOperation::Power => unreachable!("power handled above"),
            })
        }
        _ => return Err(numeric_type_error()),
    };
    Ok(value)
}

fn evaluate_abs(value: ArithmeticValue) -> Result<ArithmeticValue> {
    match value {
        ArithmeticValue::I32(value) => Ok(ArithmeticValue::I32(value.wrapping_abs())),
        ArithmeticValue::I64(value) => Ok(ArithmeticValue::I64(value.wrapping_abs())),
        ArithmeticValue::U32(value) => Ok(ArithmeticValue::U32(value)),
        ArithmeticValue::U64(value) => Ok(ArithmeticValue::U64(value)),
        ArithmeticValue::F32(value) => Ok(ArithmeticValue::F32(value.abs())),
        ArithmeticValue::F64(value) => Ok(ArithmeticValue::F64(value.abs())),
        ArithmeticValue::Bool(_) | ArithmeticValue::Symbol(_) | ArithmeticValue::String(_) => Err(
            XlogError::Compilation("abs() requires numeric input".to_string()),
        ),
    }
}

/// Compare two scalar values using the execution runtime's comparison semantics.
///
/// Equal-width floating-point operands retain their width. When exactly one operand
/// is floating point, or the widths differ, both numeric values are converted to
/// `f64`, matching runtime predicate evaluation. Float equality remains IEEE 754;
/// ordered comparisons use IEEE total ordering.
pub fn compare_arithmetic_values(
    left: &ArithmeticValue,
    op: CompOp,
    right: &ArithmeticValue,
) -> Result<bool> {
    let mixed_float_numeric = left.kind() != right.kind()
        && (is_float(left) || is_float(right))
        && is_numeric(left)
        && is_numeric(right);
    if mixed_float_numeric {
        return Ok(compare_f64(left.as_f64()?, op, right.as_f64()?));
    }
    if left.kind() != right.kind() {
        return Err(XlogError::Compilation(
            "Comparison between differing types is not supported".to_string(),
        ));
    }
    macro_rules! compare_ordered {
        ($left:expr, $right:expr) => {
            match op {
                CompOp::Eq => $left == $right,
                CompOp::Ne => $left != $right,
                CompOp::Lt => $left < $right,
                CompOp::Le => $left <= $right,
                CompOp::Gt => $left > $right,
                CompOp::Ge => $left >= $right,
            }
        };
    }
    Ok(match (left, right) {
        (ArithmeticValue::I32(left), ArithmeticValue::I32(right)) => {
            compare_ordered!(left, right)
        }
        (ArithmeticValue::I64(left), ArithmeticValue::I64(right)) => {
            compare_ordered!(left, right)
        }
        (ArithmeticValue::U32(left), ArithmeticValue::U32(right)) => {
            compare_ordered!(left, right)
        }
        (ArithmeticValue::U64(left), ArithmeticValue::U64(right)) => {
            compare_ordered!(left, right)
        }
        (ArithmeticValue::F32(left), ArithmeticValue::F32(right)) => compare_f32(*left, op, *right),
        (ArithmeticValue::F64(left), ArithmeticValue::F64(right)) => compare_f64(*left, op, *right),
        (ArithmeticValue::Bool(left), ArithmeticValue::Bool(right)) => {
            compare_ordered!(left, right)
        }
        (ArithmeticValue::Symbol(left), ArithmeticValue::Symbol(right)) => {
            compare_ordered!(left, right)
        }
        (ArithmeticValue::String(left), ArithmeticValue::String(right)) => {
            compare_ordered!(left, right)
        }
        _ => unreachable!("scalar types checked above"),
    })
}

fn compare_f32(left: f32, op: CompOp, right: f32) -> bool {
    match op {
        CompOp::Eq => left == right,
        CompOp::Ne => left != right,
        CompOp::Lt => left.total_cmp(&right).is_lt(),
        CompOp::Le => !left.total_cmp(&right).is_gt(),
        CompOp::Gt => left.total_cmp(&right).is_gt(),
        CompOp::Ge => !left.total_cmp(&right).is_lt(),
    }
}

fn compare_f64(left: f64, op: CompOp, right: f64) -> bool {
    match op {
        CompOp::Eq => left == right,
        CompOp::Ne => left != right,
        CompOp::Lt => left.total_cmp(&right).is_lt(),
        CompOp::Le => !left.total_cmp(&right).is_gt(),
        CompOp::Gt => left.total_cmp(&right).is_gt(),
        CompOp::Ge => !left.total_cmp(&right).is_lt(),
    }
}

fn is_float(value: &ArithmeticValue) -> bool {
    matches!(value, ArithmeticValue::F32(_) | ArithmeticValue::F64(_))
}

fn is_numeric(value: &ArithmeticValue) -> bool {
    matches!(
        value,
        ArithmeticValue::I32(_)
            | ArithmeticValue::I64(_)
            | ArithmeticValue::U32(_)
            | ArithmeticValue::U64(_)
            | ArithmeticValue::F32(_)
            | ArithmeticValue::F64(_)
    )
}

fn normalize_nan_f32(value: f32) -> f32 {
    if value.is_nan() {
        f32::from_bits(0x7fc0_0000)
    } else {
        value
    }
}

fn normalize_nan_f64(value: f64) -> f64 {
    if value.is_nan() {
        f64::from_bits(0x7ff8_0000_0000_0000)
    } else {
        value
    }
}

fn float_to_i64(value: f64) -> Result<i64> {
    const I64_MIN_AS_F64: f64 = -9_223_372_036_854_775_808.0;
    const I64_MAX_EXCLUSIVE_AS_F64: f64 = 9_223_372_036_854_775_808.0;
    if !value.is_finite() || !(I64_MIN_AS_F64..I64_MAX_EXCLUSIVE_AS_F64).contains(&value.trunc()) {
        return Err(XlogError::Compilation(
            "Floating-point value cannot be represented by the runtime integer cast".to_string(),
        ));
    }
    Ok(value as i64)
}

fn numeric_type_error() -> XlogError {
    XlogError::Compilation("Arithmetic operation requires matching numeric types".to_string())
}

fn evaluation_state_error() -> XlogError {
    XlogError::Compilation("Invalid arithmetic evaluation state".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate(expression: &ArithExpr) -> Result<ArithmeticValue> {
        evaluate_arithmetic_expression(expression, &HashMap::new())
    }

    #[test]
    fn integer_edge_cases_match_runtime_contract() {
        let zero = ArithExpr::Integer(0);
        let one = ArithExpr::Integer(1);
        assert_eq!(
            evaluate(&ArithExpr::Div(
                Box::new(one.clone()),
                Box::new(zero.clone())
            ))
            .expect("division result"),
            ArithmeticValue::I64(i64::MAX)
        );
        assert_eq!(
            evaluate(&ArithExpr::Mod(Box::new(one), Box::new(zero))).expect("remainder result"),
            ArithmeticValue::I64(0)
        );
        assert_eq!(
            evaluate(&ArithExpr::Add(
                Box::new(ArithExpr::Integer(i64::MAX)),
                Box::new(ArithExpr::Integer(1)),
            ))
            .expect("wrapping sum"),
            ArithmeticValue::I64(i64::MIN)
        );

        let bindings = HashMap::from([
            (
                "wide_unsigned".to_string(),
                ArithmeticValue::U32(2_147_483_648),
            ),
            ("max_signed".to_string(), ArithmeticValue::I32(i32::MAX)),
        ]);
        assert_eq!(
            evaluate_arithmetic_expression(
                &ArithExpr::Add(
                    Box::new(ArithExpr::Variable("wide_unsigned".to_string())),
                    Box::new(ArithExpr::Variable("wide_unsigned".to_string())),
                ),
                &bindings,
            )
            .expect("u32 wrapping sum"),
            ArithmeticValue::U32(0)
        );
        assert_eq!(
            evaluate_arithmetic_expression(
                &ArithExpr::Add(
                    Box::new(ArithExpr::Variable("max_signed".to_string())),
                    Box::new(ArithExpr::Cast(
                        Box::new(ArithExpr::Integer(1)),
                        ScalarType::I32,
                    )),
                ),
                &bindings,
            )
            .expect("i32 wrapping sum"),
            ArithmeticValue::I32(i32::MIN)
        );
    }

    #[test]
    fn casts_power_and_conditionals_preserve_types_and_order() {
        let expression = ArithExpr::Conditional {
            cond_left: Box::new(ArithExpr::Integer(1)),
            cond_op: CompOp::Eq,
            cond_right: Box::new(ArithExpr::Integer(1)),
            then_expr: Box::new(ArithExpr::Cast(
                Box::new(ArithExpr::Pow(
                    Box::new(ArithExpr::Integer(2)),
                    Box::new(ArithExpr::Integer(3)),
                )),
                ScalarType::F32,
            )),
            else_expr: Box::new(ArithExpr::Cast(
                Box::new(ArithExpr::Integer(0)),
                ScalarType::F32,
            )),
        };
        assert_eq!(
            evaluate(&expression).expect("conditional result"),
            ArithmeticValue::F32(8.0)
        );

        let eager_error = ArithExpr::Conditional {
            cond_left: Box::new(ArithExpr::Integer(1)),
            cond_op: CompOp::Eq,
            cond_right: Box::new(ArithExpr::Integer(1)),
            then_expr: Box::new(ArithExpr::Variable("then_missing".to_string())),
            else_expr: Box::new(ArithExpr::Variable("else_missing".to_string())),
        };
        let error = evaluate(&eager_error).expect_err("eager branch error");
        assert!(error.to_string().contains("then_missing"), "{error}");

        let bindings = HashMap::from([
            ("enabled".to_string(), ArithmeticValue::Bool(true)),
            ("label".to_string(), ArithmeticValue::Symbol(7)),
        ]);
        for (name, expected) in [("enabled", 1.0), ("label", 7.0)] {
            let cast = ArithExpr::Cast(
                Box::new(ArithExpr::Variable(name.to_string())),
                ScalarType::F64,
            );
            assert_eq!(
                evaluate_arithmetic_expression(&cast, &bindings).expect("runtime-compatible cast"),
                ArithmeticValue::F64(expected)
            );
        }
    }

    #[test]
    fn strings_and_symbols_remain_distinct_without_panicking() {
        let mut bindings = HashMap::new();
        bindings.insert(
            "string_value".to_string(),
            ArithmeticValue::String("value".to_string()),
        );
        bindings.insert("symbol_value".to_string(), ArithmeticValue::Symbol(7));
        let comparison = ArithExpr::Conditional {
            cond_left: Box::new(ArithExpr::Variable("string_value".to_string())),
            cond_op: CompOp::Eq,
            cond_right: Box::new(ArithExpr::Variable("symbol_value".to_string())),
            then_expr: Box::new(ArithExpr::Integer(1)),
            else_expr: Box::new(ArithExpr::Integer(0)),
        };
        let error = evaluate_arithmetic_expression(&comparison, &bindings)
            .expect_err("string/symbol comparison must fail");
        assert!(error.to_string().contains("differing types"), "{error}");

        let cast = ArithExpr::Cast(
            Box::new(ArithExpr::Variable("string_value".to_string())),
            ScalarType::Symbol,
        );
        assert!(evaluate_arithmetic_expression(&cast, &bindings).is_err());
    }

    #[test]
    fn float_operations_match_cuda_nan_and_selection_semantics() {
        let nan = f64::from_bits(0xfff8_0000_0000_0042);
        let mut bindings = HashMap::new();
        bindings.insert("nan".to_string(), ArithmeticValue::F64(nan));
        bindings.insert("one".to_string(), ArithmeticValue::F64(1.0));
        bindings.insert("positive_zero".to_string(), ArithmeticValue::F64(0.0));
        bindings.insert("negative_zero".to_string(), ArithmeticValue::F64(-0.0));

        for expression in [
            ArithExpr::Div(
                Box::new(ArithExpr::Float(0.0)),
                Box::new(ArithExpr::Float(0.0)),
            ),
            ArithExpr::Mod(
                Box::new(ArithExpr::Float(0.0)),
                Box::new(ArithExpr::Float(0.0)),
            ),
            ArithExpr::Pow(
                Box::new(ArithExpr::Float(-1.0)),
                Box::new(ArithExpr::Float(0.5)),
            ),
        ] {
            let ArithmeticValue::F64(value) =
                evaluate_arithmetic_expression(&expression, &bindings).expect("float result")
            else {
                panic!("expected f64 result");
            };
            assert_eq!(value.to_bits(), 0x7ff8_0000_0000_0000);
        }

        assert_eq!(
            evaluate_arithmetic_expression(
                &ArithExpr::Min(
                    Box::new(ArithExpr::Variable("nan".to_string())),
                    Box::new(ArithExpr::Variable("one".to_string())),
                ),
                &bindings,
            )
            .expect("minimum"),
            ArithmeticValue::F64(1.0)
        );
        let ArithmeticValue::F64(minimum_zero) = evaluate_arithmetic_expression(
            &ArithExpr::Min(
                Box::new(ArithExpr::Variable("negative_zero".to_string())),
                Box::new(ArithExpr::Variable("positive_zero".to_string())),
            ),
            &bindings,
        )
        .expect("minimum zero") else {
            panic!("expected f64 minimum");
        };
        assert_eq!(minimum_zero.to_bits(), 0.0_f64.to_bits());
        let ArithmeticValue::F64(maximum_zero) = evaluate_arithmetic_expression(
            &ArithExpr::Max(
                Box::new(ArithExpr::Variable("positive_zero".to_string())),
                Box::new(ArithExpr::Variable("negative_zero".to_string())),
            ),
            &bindings,
        )
        .expect("maximum zero") else {
            panic!("expected f64 maximum");
        };
        assert_eq!(maximum_zero.to_bits(), (-0.0_f64).to_bits());
    }

    #[test]
    fn float_comparisons_use_runtime_promotion_and_total_ordering() {
        let positive_nan = ArithmeticValue::F64(f64::from_bits(0x7ff8_0000_0000_0000));
        assert!(compare_arithmetic_values(
            &positive_nan,
            CompOp::Gt,
            &ArithmeticValue::F64(f64::INFINITY)
        )
        .expect("NaN ordering"));
        assert!(
            !compare_arithmetic_values(&positive_nan, CompOp::Eq, &positive_nan)
                .expect("NaN equality")
        );
        assert!(
            compare_arithmetic_values(&positive_nan, CompOp::Ne, &positive_nan)
                .expect("NaN inequality")
        );
        assert!(compare_arithmetic_values(
            &ArithmeticValue::F64(-0.0),
            CompOp::Lt,
            &ArithmeticValue::F64(0.0)
        )
        .expect("signed zero ordering"));
        assert!(compare_arithmetic_values(
            &ArithmeticValue::F32(2.5),
            CompOp::Gt,
            &ArithmeticValue::I64(2)
        )
        .expect("mixed numeric comparison"));

        let conditional = ArithExpr::Conditional {
            cond_left: Box::new(ArithExpr::Variable("float".to_string())),
            cond_op: CompOp::Lt,
            cond_right: Box::new(ArithExpr::Variable("integer".to_string())),
            then_expr: Box::new(ArithExpr::Integer(1)),
            else_expr: Box::new(ArithExpr::Integer(0)),
        };
        let bindings = HashMap::from([
            ("float".to_string(), ArithmeticValue::F32(2.5)),
            ("integer".to_string(), ArithmeticValue::I64(2)),
        ]);
        assert_eq!(
            evaluate_arithmetic_expression(&conditional, &bindings).expect("conditional"),
            ArithmeticValue::I64(0)
        );

        let mismatched_branches = ArithExpr::Conditional {
            cond_left: Box::new(ArithExpr::Integer(1)),
            cond_op: CompOp::Eq,
            cond_right: Box::new(ArithExpr::Integer(1)),
            then_expr: Box::new(ArithExpr::Integer(1)),
            else_expr: Box::new(ArithExpr::Float(1.0)),
        };
        assert!(evaluate(&mismatched_branches).is_err());
    }

    #[test]
    fn typed_terms_follow_lowering_widths_and_reject_undefined_float_casts() {
        assert_eq!(
            ArithmeticValue::from_typed_term(&Term::Integer(2_147_483_648), ScalarType::U32)
                .expect("u32 literal"),
            ArithmeticValue::U32(2_147_483_648)
        );
        assert_eq!(
            ArithmeticValue::from_typed_term(&Term::Float(1.5), ScalarType::F32)
                .expect("f32 literal"),
            ArithmeticValue::F32(1.5)
        );
        let invalid_cast = ArithExpr::Cast(Box::new(ArithExpr::Float(f64::NAN)), ScalarType::I64);
        assert!(evaluate(&invalid_cast).is_err());
    }

    #[test]
    fn configured_depth_expression_uses_bounded_native_stack() {
        let mut expression = ArithExpr::Integer(0);
        for _ in 0..1_000 {
            expression = ArithExpr::Add(Box::new(expression), Box::new(ArithExpr::Integer(1)));
        }
        assert_eq!(
            evaluate(&expression).expect("deep expression result"),
            ArithmeticValue::I64(1_000)
        );
    }
}
