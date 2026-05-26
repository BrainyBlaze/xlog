//! Type inference for user-defined functions.
//!
//! Reserved API: type inference is not yet wired into the main compilation pipeline.

use crate::ast::{ArithExpr, FuncDef};
use std::collections::HashMap;
use xlog_core::ScalarType;

/// Type inference context
#[derive(Debug, Default)]
pub(crate) struct TypeContext {
    /// Known variable types
    bindings: HashMap<String, ScalarType>,
}

impl TypeContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a variable to a type
    pub fn bind(&mut self, name: &str, typ: ScalarType) {
        self.bindings.insert(name.to_string(), typ);
    }

    /// Get a variable's type
    pub fn get(&self, name: &str) -> Option<ScalarType> {
        self.bindings.get(name).copied()
    }

    /// Infer type of an expression
    pub fn infer_expr(&self, expr: &ArithExpr) -> Option<ScalarType> {
        match expr {
            ArithExpr::Variable(name) => self.get(name),
            ArithExpr::Integer(_) => Some(ScalarType::I64),
            ArithExpr::Float(_) => Some(ScalarType::F64),
            ArithExpr::Add(l, r)
            | ArithExpr::Sub(l, r)
            | ArithExpr::Mul(l, r)
            | ArithExpr::Div(l, r)
            | ArithExpr::Mod(l, r) => {
                let lt = self.infer_expr(l)?;
                let rt = self.infer_expr(r)?;
                // Numeric promotion: if either is f64, result is f64
                if lt == ScalarType::F64 || rt == ScalarType::F64 {
                    Some(ScalarType::F64)
                } else {
                    Some(lt)
                }
            }
            ArithExpr::Cast(_, t) => Some(*t),
            ArithExpr::Abs(e) => self.infer_expr(e),
            ArithExpr::Min(l, r) | ArithExpr::Max(l, r) | ArithExpr::Pow(l, r) => {
                let lt = self.infer_expr(l)?;
                let rt = self.infer_expr(r)?;
                if lt == ScalarType::F64 || rt == ScalarType::F64 {
                    Some(ScalarType::F64)
                } else {
                    Some(lt)
                }
            }
            ArithExpr::FuncCall { .. } => None, // Need registry lookup
            ArithExpr::Conditional {
                then_expr,
                else_expr,
                ..
            } => {
                // Type is the common type of both branches
                let then_t = self.infer_expr(then_expr)?;
                let else_t = self.infer_expr(else_expr)?;
                if then_t == else_t {
                    Some(then_t)
                } else if then_t == ScalarType::F64 || else_t == ScalarType::F64 {
                    Some(ScalarType::F64)
                } else {
                    Some(then_t)
                }
            }
        }
    }
}

/// Infer parameter types from function definition
pub(crate) fn infer_param_types(func: &FuncDef) -> Vec<Option<ScalarType>> {
    func.params.iter().map(|p| p.typ).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_literal() {
        let ctx = TypeContext::new();
        assert_eq!(
            ctx.infer_expr(&ArithExpr::Integer(5)),
            Some(ScalarType::I64)
        );
        assert_eq!(
            ctx.infer_expr(&ArithExpr::Float(3.25)),
            Some(ScalarType::F64)
        );
    }

    #[test]
    fn test_infer_variable() {
        let mut ctx = TypeContext::new();
        ctx.bind("X", ScalarType::F64);
        assert_eq!(
            ctx.infer_expr(&ArithExpr::Variable("X".into())),
            Some(ScalarType::F64)
        );
    }

    #[test]
    fn test_infer_numeric_promotion() {
        let ctx = TypeContext::new();
        let expr = ArithExpr::Add(
            Box::new(ArithExpr::Integer(5)),
            Box::new(ArithExpr::Float(3.0)),
        );
        assert_eq!(ctx.infer_expr(&expr), Some(ScalarType::F64));
    }

    #[test]
    fn test_infer_unknown_variable() {
        let ctx = TypeContext::new();
        assert_eq!(ctx.infer_expr(&ArithExpr::Variable("Unknown".into())), None);
    }

    #[test]
    fn test_infer_cast() {
        let ctx = TypeContext::new();
        let expr = ArithExpr::Cast(Box::new(ArithExpr::Integer(5)), ScalarType::F64);
        assert_eq!(ctx.infer_expr(&expr), Some(ScalarType::F64));
    }

    #[test]
    fn test_infer_abs() {
        let mut ctx = TypeContext::new();
        ctx.bind("X", ScalarType::I64);
        let expr = ArithExpr::Abs(Box::new(ArithExpr::Variable("X".into())));
        assert_eq!(ctx.infer_expr(&expr), Some(ScalarType::I64));
    }

    #[test]
    fn test_infer_min_max() {
        let ctx = TypeContext::new();
        let expr = ArithExpr::Min(
            Box::new(ArithExpr::Integer(5)),
            Box::new(ArithExpr::Integer(3)),
        );
        assert_eq!(ctx.infer_expr(&expr), Some(ScalarType::I64));

        let expr_float = ArithExpr::Max(
            Box::new(ArithExpr::Integer(5)),
            Box::new(ArithExpr::Float(3.0)),
        );
        assert_eq!(ctx.infer_expr(&expr_float), Some(ScalarType::F64));
    }

    #[test]
    fn test_infer_conditional() {
        use crate::ast::CompOp;

        let mut ctx = TypeContext::new();
        ctx.bind("X", ScalarType::I64);
        let expr = ArithExpr::Conditional {
            cond_left: Box::new(ArithExpr::Variable("X".into())),
            cond_op: CompOp::Lt,
            cond_right: Box::new(ArithExpr::Integer(0)),
            then_expr: Box::new(ArithExpr::Integer(1)),
            else_expr: Box::new(ArithExpr::Integer(2)),
        };
        assert_eq!(ctx.infer_expr(&expr), Some(ScalarType::I64));
    }

    #[test]
    fn test_infer_param_types() {
        use crate::ast::{FuncBody, FuncParam};

        let func = FuncDef {
            name: "test".to_string(),
            params: vec![
                FuncParam {
                    name: "X".to_string(),
                    typ: Some(ScalarType::I64),
                },
                FuncParam {
                    name: "Y".to_string(),
                    typ: None,
                },
            ],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::Integer(1)),
            is_private: false,
        };

        let types = infer_param_types(&func);
        assert_eq!(types.len(), 2);
        assert_eq!(types[0], Some(ScalarType::I64));
        assert_eq!(types[1], None);
    }
}
