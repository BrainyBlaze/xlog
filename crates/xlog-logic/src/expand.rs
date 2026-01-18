//! Inline expansion of user-defined functions.

use crate::ast::{ArithExpr, FuncBody};
use crate::function::{FunctionError, FunctionRegistry};
use std::collections::HashMap;

/// Context for expansion
pub struct ExpansionContext<'a> {
    registry: &'a FunctionRegistry,
    depth: u32,
    max_depth: u32,
}

impl<'a> ExpansionContext<'a> {
    pub fn new(registry: &'a FunctionRegistry, max_depth: u32) -> Self {
        Self {
            registry,
            depth: 0,
            max_depth,
        }
    }

    /// Expand a function call to its body with arguments substituted
    pub fn expand_call(
        &mut self,
        name: &str,
        args: &[ArithExpr],
    ) -> Result<ArithExpr, FunctionError> {
        // Check depth limit
        if self.depth >= self.max_depth {
            return Err(FunctionError::MaxRecursionDepth {
                name: name.to_string(),
                depth: self.max_depth,
            });
        }

        let func = self
            .registry
            .get(name)
            .ok_or_else(|| FunctionError::UndefinedFunction {
                name: name.to_string(),
            })?;

        // Build substitution map
        let mut subst: HashMap<String, ArithExpr> = HashMap::new();
        for (param, arg) in func.params.iter().zip(args.iter()) {
            subst.insert(param.name.clone(), arg.clone());
        }

        // Expand body
        self.depth += 1;
        let result = self.expand_body(&func.body, &subst)?;
        self.depth -= 1;

        Ok(result)
    }

    fn expand_body(
        &mut self,
        body: &FuncBody,
        subst: &HashMap<String, ArithExpr>,
    ) -> Result<ArithExpr, FunctionError> {
        match body {
            FuncBody::Arithmetic(expr) => self.expand_expr(expr, subst),
            FuncBody::Conditional(cond) => {
                // Expand condition parts
                let cond_left = self.expand_expr(&cond.cond_left, subst)?;
                let cond_right = self.expand_expr(&cond.cond_right, subst)?;
                let then_expr = self.expand_body(&cond.then_branch, subst)?;
                let else_expr = self.expand_body(&cond.else_branch, subst)?;

                // Return a conditional ArithExpr (need to represent this)
                // For now, we create a structure that captures the conditional
                // The actual evaluation will happen at runtime
                Ok(ArithExpr::Conditional {
                    cond_left: Box::new(cond_left),
                    cond_op: cond.cond_op,
                    cond_right: Box::new(cond_right),
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
                })
            }
            FuncBody::Predicate { result, body: _ } => {
                // Predicate bodies are expanded differently - they become joins
                // For now, return a placeholder that signals predicate expansion needed
                // The result variable after substitution
                let result_var = subst
                    .get(result)
                    .cloned()
                    .unwrap_or_else(|| ArithExpr::Variable(result.clone()));
                Ok(result_var)
            }
        }
    }

    fn expand_expr(
        &mut self,
        expr: &ArithExpr,
        subst: &HashMap<String, ArithExpr>,
    ) -> Result<ArithExpr, FunctionError> {
        match expr {
            ArithExpr::Variable(name) => {
                Ok(subst.get(name).cloned().unwrap_or_else(|| expr.clone()))
            }
            ArithExpr::Integer(_) | ArithExpr::Float(_) => Ok(expr.clone()),
            ArithExpr::FuncCall { name, args } => {
                // First expand arguments
                let expanded_args: Result<Vec<_>, _> =
                    args.iter().map(|a| self.expand_expr(a, subst)).collect();
                let expanded_args = expanded_args?;

                // Then expand the function call if it's a UDF
                if self.registry.contains(name) {
                    self.expand_call(name, &expanded_args)
                } else {
                    // Built-in function, just return with expanded args
                    Ok(ArithExpr::FuncCall {
                        name: name.clone(),
                        args: expanded_args,
                    })
                }
            }
            ArithExpr::Add(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Add(Box::new(el), Box::new(er)))
            }
            ArithExpr::Sub(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Sub(Box::new(el), Box::new(er)))
            }
            ArithExpr::Mul(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Mul(Box::new(el), Box::new(er)))
            }
            ArithExpr::Div(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Div(Box::new(el), Box::new(er)))
            }
            ArithExpr::Mod(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Mod(Box::new(el), Box::new(er)))
            }
            ArithExpr::Abs(e) => {
                let ee = self.expand_expr(e, subst)?;
                Ok(ArithExpr::Abs(Box::new(ee)))
            }
            ArithExpr::Min(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Min(Box::new(el), Box::new(er)))
            }
            ArithExpr::Max(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Max(Box::new(el), Box::new(er)))
            }
            ArithExpr::Pow(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Pow(Box::new(el), Box::new(er)))
            }
            ArithExpr::Cast(e, t) => {
                let ee = self.expand_expr(e, subst)?;
                Ok(ArithExpr::Cast(Box::new(ee), *t))
            }
            ArithExpr::Conditional {
                cond_left,
                cond_op,
                cond_right,
                then_expr,
                else_expr,
            } => {
                let cl = self.expand_expr(cond_left, subst)?;
                let cr = self.expand_expr(cond_right, subst)?;
                let te = self.expand_expr(then_expr, subst)?;
                let ee = self.expand_expr(else_expr, subst)?;
                Ok(ArithExpr::Conditional {
                    cond_left: Box::new(cl),
                    cond_op: *cond_op,
                    cond_right: Box::new(cr),
                    then_expr: Box::new(te),
                    else_expr: Box::new(ee),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{FuncDef, FuncParam};

    #[test]
    fn test_simple_expansion() {
        let mut reg = FunctionRegistry::new();

        // func double(X) = X + X
        let double = FuncDef {
            name: "double".to_string(),
            params: vec![FuncParam {
                name: "X".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::Add(
                Box::new(ArithExpr::Variable("X".to_string())),
                Box::new(ArithExpr::Variable("X".to_string())),
            )),
            is_private: false,
        };
        reg.register(double).unwrap();

        let mut ctx = ExpansionContext::new(&reg, 100);

        // double(5) should expand to 5 + 5
        let result = ctx.expand_call("double", &[ArithExpr::Integer(5)]).unwrap();

        match result {
            ArithExpr::Add(l, r) => {
                assert!(matches!(*l, ArithExpr::Integer(5)));
                assert!(matches!(*r, ArithExpr::Integer(5)));
            }
            _ => panic!("Expected Add expression"),
        }
    }

    #[test]
    fn test_nested_expansion() {
        let mut reg = FunctionRegistry::new();

        // func double(X) = X + X
        let double = FuncDef {
            name: "double".to_string(),
            params: vec![FuncParam {
                name: "X".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::Add(
                Box::new(ArithExpr::Variable("X".to_string())),
                Box::new(ArithExpr::Variable("X".to_string())),
            )),
            is_private: false,
        };

        // func quadruple(X) = double(double(X))
        let quadruple = FuncDef {
            name: "quadruple".to_string(),
            params: vec![FuncParam {
                name: "X".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::FuncCall {
                name: "double".to_string(),
                args: vec![ArithExpr::FuncCall {
                    name: "double".to_string(),
                    args: vec![ArithExpr::Variable("X".to_string())],
                }],
            }),
            is_private: false,
        };

        reg.register(double).unwrap();
        reg.register(quadruple).unwrap();

        let mut ctx = ExpansionContext::new(&reg, 100);

        // quadruple(2) should expand to (2 + 2) + (2 + 2)
        let result = ctx
            .expand_call("quadruple", &[ArithExpr::Integer(2)])
            .unwrap();

        // Result should be Add(Add(2, 2), Add(2, 2))
        match &result {
            ArithExpr::Add(l, r) => {
                assert!(matches!(l.as_ref(), ArithExpr::Add(_, _)));
                assert!(matches!(r.as_ref(), ArithExpr::Add(_, _)));
            }
            _ => panic!("Expected nested Add expression, got {:?}", result),
        }
    }

    #[test]
    fn test_max_recursion_depth() {
        let mut reg = FunctionRegistry::new();

        // func infinite(X) = infinite(X)
        let infinite = FuncDef {
            name: "infinite".to_string(),
            params: vec![FuncParam {
                name: "X".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::FuncCall {
                name: "infinite".to_string(),
                args: vec![ArithExpr::Variable("X".to_string())],
            }),
            is_private: false,
        };
        reg.register(infinite).unwrap();

        let mut ctx = ExpansionContext::new(&reg, 10);

        let result = ctx.expand_call("infinite", &[ArithExpr::Integer(1)]);
        assert!(matches!(
            result,
            Err(FunctionError::MaxRecursionDepth { .. })
        ));
    }

    #[test]
    fn test_undefined_function() {
        let reg = FunctionRegistry::new();
        let mut ctx = ExpansionContext::new(&reg, 100);

        let result = ctx.expand_call("undefined", &[ArithExpr::Integer(1)]);
        assert!(matches!(
            result,
            Err(FunctionError::UndefinedFunction { .. })
        ));
    }

    #[test]
    fn test_builtin_function_passthrough() {
        let mut reg = FunctionRegistry::new();

        // func abs_x(X) = abs(X)
        let abs_x = FuncDef {
            name: "abs_x".to_string(),
            params: vec![FuncParam {
                name: "X".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::FuncCall {
                name: "abs".to_string(),
                args: vec![ArithExpr::Variable("X".to_string())],
            }),
            is_private: false,
        };
        reg.register(abs_x).unwrap();

        let mut ctx = ExpansionContext::new(&reg, 100);

        let result = ctx.expand_call("abs_x", &[ArithExpr::Integer(-5)]).unwrap();

        // Should preserve abs call with substituted arg
        match result {
            ArithExpr::FuncCall { name, args } => {
                assert_eq!(name, "abs");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], ArithExpr::Integer(-5)));
            }
            _ => panic!("Expected FuncCall for builtin"),
        }
    }

    #[test]
    fn test_variable_substitution() {
        let mut reg = FunctionRegistry::new();

        // func add(X, Y) = X + Y
        let add = FuncDef {
            name: "add".to_string(),
            params: vec![
                FuncParam {
                    name: "X".to_string(),
                    typ: None,
                },
                FuncParam {
                    name: "Y".to_string(),
                    typ: None,
                },
            ],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::Add(
                Box::new(ArithExpr::Variable("X".to_string())),
                Box::new(ArithExpr::Variable("Y".to_string())),
            )),
            is_private: false,
        };
        reg.register(add).unwrap();

        let mut ctx = ExpansionContext::new(&reg, 100);

        // add(3, 7) should expand to 3 + 7
        let result = ctx
            .expand_call("add", &[ArithExpr::Integer(3), ArithExpr::Integer(7)])
            .unwrap();

        match result {
            ArithExpr::Add(l, r) => {
                assert!(matches!(*l, ArithExpr::Integer(3)));
                assert!(matches!(*r, ArithExpr::Integer(7)));
            }
            _ => panic!("Expected Add expression"),
        }
    }

    #[test]
    fn test_expansion_with_variable_args() {
        let mut reg = FunctionRegistry::new();

        // func double(X) = X + X
        let double = FuncDef {
            name: "double".to_string(),
            params: vec![FuncParam {
                name: "X".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::Add(
                Box::new(ArithExpr::Variable("X".to_string())),
                Box::new(ArithExpr::Variable("X".to_string())),
            )),
            is_private: false,
        };
        reg.register(double).unwrap();

        let mut ctx = ExpansionContext::new(&reg, 100);

        // double(Y) should expand to Y + Y
        let result = ctx
            .expand_call("double", &[ArithExpr::Variable("Y".to_string())])
            .unwrap();

        match result {
            ArithExpr::Add(l, r) => {
                assert!(matches!(l.as_ref(), ArithExpr::Variable(n) if n == "Y"));
                assert!(matches!(r.as_ref(), ArithExpr::Variable(n) if n == "Y"));
            }
            _ => panic!("Expected Add expression"),
        }
    }
}
