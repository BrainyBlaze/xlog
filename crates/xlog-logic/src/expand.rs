//! Inline expansion of user-defined functions.

use crate::ast::{ArithExpr, Atom, BodyLiteral, Comparison, FuncBody, FuncDef, IsExpr, Term};
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
            FuncBody::Predicate { result, .. } => {
                // Predicate bodies are expanded via expand_predicate_func at the rule level.
                // When expand_body is called for a predicate body (shouldn't happen directly),
                // return the result variable after substitution as an ArithExpr.
                // The actual expansion to join literals happens via expand_predicate_func.
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

    /// Expand a predicate-based function call to join literals.
    ///
    /// Predicate functions like `func get_parent(X) = P :- parent(X, P).`
    /// expand to body literals that get added to the calling rule.
    ///
    /// Returns the expanded body literals and the result variable name.
    #[allow(dead_code)] // reserved API: predicate-func expansion not yet wired
    pub(crate) fn expand_predicate_func(
        &self,
        func: &FuncDef,
        args: &[ArithExpr],
    ) -> Result<(Vec<BodyLiteral>, String), FunctionError> {
        match &func.body {
            FuncBody::Predicate { result, body } => {
                // Build substitution map from params to args
                let mut subst: HashMap<String, ArithExpr> = HashMap::new();
                for (param, arg) in func.params.iter().zip(args.iter()) {
                    subst.insert(param.name.clone(), arg.clone());
                }

                // Substitute in body literals
                let expanded_body: Vec<BodyLiteral> = body
                    .iter()
                    .map(|lit| self.substitute_literal(lit, &subst))
                    .collect();

                // The result variable becomes the output (substitute if mapped)
                let result_var = self.substitute_var(result, &subst);

                Ok((expanded_body, result_var))
            }
            _ => Err(FunctionError::UndefinedFunction {
                name: func.name.clone(),
            }),
        }
    }

    /// Substitute variables in a body literal using the given substitution map.
    fn substitute_literal(
        &self,
        lit: &BodyLiteral,
        subst: &HashMap<String, ArithExpr>,
    ) -> BodyLiteral {
        match lit {
            BodyLiteral::Positive(atom) => BodyLiteral::Positive(self.substitute_atom(atom, subst)),
            BodyLiteral::Negated(atom) => BodyLiteral::Negated(self.substitute_atom(atom, subst)),
            BodyLiteral::Comparison(cmp) => BodyLiteral::Comparison(Comparison {
                left: self.substitute_term(&cmp.left, subst),
                op: cmp.op,
                right: self.substitute_term(&cmp.right, subst),
            }),
            BodyLiteral::IsExpr(is_expr) => {
                // Substitute in both the target variable and the expression
                let target = self.substitute_var(&is_expr.target, subst);
                // For ArithExpr substitution, we need to substitute variables
                let expr = self.substitute_arith_expr(&is_expr.expr, subst);
                BodyLiteral::IsExpr(IsExpr { target, expr })
            }
        }
    }

    /// Substitute variables in an atom.
    fn substitute_atom(&self, atom: &Atom, subst: &HashMap<String, ArithExpr>) -> Atom {
        Atom {
            predicate: atom.predicate.clone(),
            terms: atom
                .terms
                .iter()
                .map(|t| self.substitute_term(t, subst))
                .collect(),
        }
    }

    /// Substitute a variable in a term.
    fn substitute_term(&self, term: &Term, subst: &HashMap<String, ArithExpr>) -> Term {
        match term {
            Term::Variable(name) => {
                if let Some(replacement) = subst.get(name) {
                    match replacement {
                        ArithExpr::Variable(new_name) => Term::Variable(new_name.clone()),
                        ArithExpr::Integer(n) => Term::Integer(*n),
                        ArithExpr::Float(f) => Term::Float(*f),
                        // For complex expressions, we can't directly substitute into a Term,
                        // so we keep the original variable (this is a limitation)
                        _ => term.clone(),
                    }
                } else {
                    term.clone()
                }
            }
            _ => term.clone(),
        }
    }

    /// Substitute variables in an arithmetic expression.
    fn substitute_arith_expr(
        &self,
        expr: &ArithExpr,
        subst: &HashMap<String, ArithExpr>,
    ) -> ArithExpr {
        match expr {
            ArithExpr::Variable(name) => subst.get(name).cloned().unwrap_or_else(|| expr.clone()),
            ArithExpr::Integer(_) | ArithExpr::Float(_) => expr.clone(),
            ArithExpr::Add(l, r) => ArithExpr::Add(
                Box::new(self.substitute_arith_expr(l, subst)),
                Box::new(self.substitute_arith_expr(r, subst)),
            ),
            ArithExpr::Sub(l, r) => ArithExpr::Sub(
                Box::new(self.substitute_arith_expr(l, subst)),
                Box::new(self.substitute_arith_expr(r, subst)),
            ),
            ArithExpr::Mul(l, r) => ArithExpr::Mul(
                Box::new(self.substitute_arith_expr(l, subst)),
                Box::new(self.substitute_arith_expr(r, subst)),
            ),
            ArithExpr::Div(l, r) => ArithExpr::Div(
                Box::new(self.substitute_arith_expr(l, subst)),
                Box::new(self.substitute_arith_expr(r, subst)),
            ),
            ArithExpr::Mod(l, r) => ArithExpr::Mod(
                Box::new(self.substitute_arith_expr(l, subst)),
                Box::new(self.substitute_arith_expr(r, subst)),
            ),
            ArithExpr::Abs(e) => ArithExpr::Abs(Box::new(self.substitute_arith_expr(e, subst))),
            ArithExpr::Min(l, r) => ArithExpr::Min(
                Box::new(self.substitute_arith_expr(l, subst)),
                Box::new(self.substitute_arith_expr(r, subst)),
            ),
            ArithExpr::Max(l, r) => ArithExpr::Max(
                Box::new(self.substitute_arith_expr(l, subst)),
                Box::new(self.substitute_arith_expr(r, subst)),
            ),
            ArithExpr::Pow(l, r) => ArithExpr::Pow(
                Box::new(self.substitute_arith_expr(l, subst)),
                Box::new(self.substitute_arith_expr(r, subst)),
            ),
            ArithExpr::Cast(e, t) => {
                ArithExpr::Cast(Box::new(self.substitute_arith_expr(e, subst)), *t)
            }
            ArithExpr::FuncCall { name, args } => ArithExpr::FuncCall {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| self.substitute_arith_expr(a, subst))
                    .collect(),
            },
            ArithExpr::Conditional {
                cond_left,
                cond_op,
                cond_right,
                then_expr,
                else_expr,
            } => ArithExpr::Conditional {
                cond_left: Box::new(self.substitute_arith_expr(cond_left, subst)),
                cond_op: *cond_op,
                cond_right: Box::new(self.substitute_arith_expr(cond_right, subst)),
                then_expr: Box::new(self.substitute_arith_expr(then_expr, subst)),
                else_expr: Box::new(self.substitute_arith_expr(else_expr, subst)),
            },
        }
    }

    /// Substitute a variable name using the substitution map.
    fn substitute_var(&self, var: &str, subst: &HashMap<String, ArithExpr>) -> String {
        if let Some(ArithExpr::Variable(new_name)) = subst.get(var) {
            new_name.clone()
        } else {
            var.to_string()
        }
    }

    /// Check if a function has a predicate body.
    #[allow(dead_code)] // reserved API: predicate-func expansion not yet wired
    pub(crate) fn is_predicate_func(&self, name: &str) -> bool {
        self.registry
            .get(name)
            .map(|f| matches!(f.body, FuncBody::Predicate { .. }))
            .unwrap_or(false)
    }

    /// Expand all function calls in an arithmetic expression.
    /// Returns the expanded expression with all UDF calls inlined.
    pub(crate) fn expand_expr_fully(
        &mut self,
        expr: &ArithExpr,
    ) -> Result<ArithExpr, FunctionError> {
        match expr {
            ArithExpr::Variable(_) | ArithExpr::Integer(_) | ArithExpr::Float(_) => {
                Ok(expr.clone())
            }
            ArithExpr::FuncCall { name, args } => {
                // First expand arguments
                let expanded_args: Result<Vec<_>, _> =
                    args.iter().map(|a| self.expand_expr_fully(a)).collect();
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
            ArithExpr::Add(l, r) => Ok(ArithExpr::Add(
                Box::new(self.expand_expr_fully(l)?),
                Box::new(self.expand_expr_fully(r)?),
            )),
            ArithExpr::Sub(l, r) => Ok(ArithExpr::Sub(
                Box::new(self.expand_expr_fully(l)?),
                Box::new(self.expand_expr_fully(r)?),
            )),
            ArithExpr::Mul(l, r) => Ok(ArithExpr::Mul(
                Box::new(self.expand_expr_fully(l)?),
                Box::new(self.expand_expr_fully(r)?),
            )),
            ArithExpr::Div(l, r) => Ok(ArithExpr::Div(
                Box::new(self.expand_expr_fully(l)?),
                Box::new(self.expand_expr_fully(r)?),
            )),
            ArithExpr::Mod(l, r) => Ok(ArithExpr::Mod(
                Box::new(self.expand_expr_fully(l)?),
                Box::new(self.expand_expr_fully(r)?),
            )),
            ArithExpr::Abs(e) => Ok(ArithExpr::Abs(Box::new(self.expand_expr_fully(e)?))),
            ArithExpr::Min(l, r) => Ok(ArithExpr::Min(
                Box::new(self.expand_expr_fully(l)?),
                Box::new(self.expand_expr_fully(r)?),
            )),
            ArithExpr::Max(l, r) => Ok(ArithExpr::Max(
                Box::new(self.expand_expr_fully(l)?),
                Box::new(self.expand_expr_fully(r)?),
            )),
            ArithExpr::Pow(l, r) => Ok(ArithExpr::Pow(
                Box::new(self.expand_expr_fully(l)?),
                Box::new(self.expand_expr_fully(r)?),
            )),
            ArithExpr::Cast(e, t) => Ok(ArithExpr::Cast(Box::new(self.expand_expr_fully(e)?), *t)),
            ArithExpr::Conditional {
                cond_left,
                cond_op,
                cond_right,
                then_expr,
                else_expr,
            } => Ok(ArithExpr::Conditional {
                cond_left: Box::new(self.expand_expr_fully(cond_left)?),
                cond_op: *cond_op,
                cond_right: Box::new(self.expand_expr_fully(cond_right)?),
                then_expr: Box::new(self.expand_expr_fully(then_expr)?),
                else_expr: Box::new(self.expand_expr_fully(else_expr)?),
            }),
        }
    }
}

use crate::ast::{Program, Rule};

/// Expand all user-defined function calls in a program.
/// Returns a new program with all UDF calls replaced by their expanded bodies.
pub fn expand_program_functions(
    program: &Program,
    max_depth: u32,
) -> Result<Program, FunctionError> {
    // Build function registry from program
    let mut registry = FunctionRegistry::new();
    for func in &program.functions {
        registry.register(func.clone())?;
    }

    // If no functions defined, return program unchanged
    if program.functions.is_empty() {
        return Ok(program.clone());
    }

    let mut ctx = ExpansionContext::new(&registry, max_depth);

    // Expand function calls in each rule
    let expanded_rules: Result<Vec<Rule>, FunctionError> = program
        .rules
        .iter()
        .map(|rule| expand_rule_functions(&mut ctx, rule))
        .collect();

    Ok(Program {
        rules: expanded_rules?,
        directives: program.directives.clone(),
        queries: program.queries.clone(),
        predicates: program.predicates.clone(),
        constraints: program.constraints.clone(),
        imports: program.imports.clone(),
        functions: program.functions.clone(),
        domains: program.domains.clone(),
        prob_facts: program.prob_facts.clone(),
        annotated_disjunctions: program.annotated_disjunctions.clone(),
        evidence: program.evidence.clone(),
        prob_queries: program.prob_queries.clone(),
        neural_predicates: program.neural_predicates.clone(),
        learnable_rules: program.learnable_rules.clone(),
    })
}

/// Expand function calls in a single rule.
fn expand_rule_functions(ctx: &mut ExpansionContext, rule: &Rule) -> Result<Rule, FunctionError> {
    let expanded_body: Result<Vec<BodyLiteral>, FunctionError> = rule
        .body
        .iter()
        .map(|lit| expand_literal_functions(ctx, lit))
        .collect();

    Ok(Rule {
        head: rule.head.clone(),
        body: expanded_body?,
    })
}

/// Expand function calls in a body literal.
fn expand_literal_functions(
    ctx: &mut ExpansionContext,
    lit: &BodyLiteral,
) -> Result<BodyLiteral, FunctionError> {
    match lit {
        BodyLiteral::Positive(atom) => Ok(BodyLiteral::Positive(atom.clone())),
        BodyLiteral::Negated(atom) => Ok(BodyLiteral::Negated(atom.clone())),
        BodyLiteral::Comparison(cmp) => Ok(BodyLiteral::Comparison(cmp.clone())),
        BodyLiteral::IsExpr(is_expr) => {
            let expanded_expr = ctx.expand_expr_fully(&is_expr.expr)?;
            Ok(BodyLiteral::IsExpr(IsExpr {
                target: is_expr.target.clone(),
                expr: expanded_expr,
            }))
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

    #[test]
    fn test_predicate_func_expansion() {
        // func get_parent(X) = P :- parent(X, P).
        // get_parent(alice) should expand to: parent(alice, P)

        let func = FuncDef {
            name: "get_parent".to_string(),
            params: vec![FuncParam {
                name: "X".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Predicate {
                result: "P".to_string(),
                body: vec![BodyLiteral::Positive(Atom {
                    predicate: "parent".to_string(),
                    terms: vec![
                        Term::Variable("X".to_string()),
                        Term::Variable("P".to_string()),
                    ],
                })],
            },
            is_private: false,
        };

        let mut reg = FunctionRegistry::new();
        reg.register(func).unwrap();

        let ctx = ExpansionContext::new(&reg, 100);

        // Call get_parent with "alice"
        let args = vec![ArithExpr::Variable("alice".to_string())];
        let func_def = reg.get("get_parent").unwrap();
        let (body, result) = ctx.expand_predicate_func(func_def, &args).unwrap();

        assert_eq!(result, "P");
        assert_eq!(body.len(), 1);

        // Check the expanded literal
        if let BodyLiteral::Positive(atom) = &body[0] {
            assert_eq!(atom.predicate, "parent");
            assert!(matches!(&atom.terms[0], Term::Variable(v) if v == "alice"));
            assert!(matches!(&atom.terms[1], Term::Variable(v) if v == "P"));
        } else {
            panic!("Expected Positive literal");
        }
    }

    #[test]
    fn test_predicate_func_with_constant_arg() {
        // func get_child(P) = C :- parent(C, P).
        // get_child(bob) should expand to: parent(C, bob)

        let func = FuncDef {
            name: "get_child".to_string(),
            params: vec![FuncParam {
                name: "P".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Predicate {
                result: "C".to_string(),
                body: vec![BodyLiteral::Positive(Atom {
                    predicate: "parent".to_string(),
                    terms: vec![
                        Term::Variable("C".to_string()),
                        Term::Variable("P".to_string()),
                    ],
                })],
            },
            is_private: false,
        };

        let mut reg = FunctionRegistry::new();
        reg.register(func).unwrap();

        let ctx = ExpansionContext::new(&reg, 100);

        // Call get_child with integer constant
        let args = vec![ArithExpr::Integer(42)];
        let func_def = reg.get("get_child").unwrap();
        let (body, result) = ctx.expand_predicate_func(func_def, &args).unwrap();

        assert_eq!(result, "C");
        assert_eq!(body.len(), 1);

        // Check the expanded literal has integer substituted
        if let BodyLiteral::Positive(atom) = &body[0] {
            assert_eq!(atom.predicate, "parent");
            assert!(matches!(&atom.terms[0], Term::Variable(v) if v == "C"));
            assert!(matches!(&atom.terms[1], Term::Integer(42)));
        } else {
            panic!("Expected Positive literal");
        }
    }

    #[test]
    fn test_predicate_func_multiple_body_literals() {
        // func get_grandparent(X) = G :- parent(X, P), parent(P, G).
        // get_grandparent(alice) should expand to: parent(alice, P), parent(P, G)

        let func = FuncDef {
            name: "get_grandparent".to_string(),
            params: vec![FuncParam {
                name: "X".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Predicate {
                result: "G".to_string(),
                body: vec![
                    BodyLiteral::Positive(Atom {
                        predicate: "parent".to_string(),
                        terms: vec![
                            Term::Variable("X".to_string()),
                            Term::Variable("P".to_string()),
                        ],
                    }),
                    BodyLiteral::Positive(Atom {
                        predicate: "parent".to_string(),
                        terms: vec![
                            Term::Variable("P".to_string()),
                            Term::Variable("G".to_string()),
                        ],
                    }),
                ],
            },
            is_private: false,
        };

        let mut reg = FunctionRegistry::new();
        reg.register(func).unwrap();

        let ctx = ExpansionContext::new(&reg, 100);

        let args = vec![ArithExpr::Variable("alice".to_string())];
        let func_def = reg.get("get_grandparent").unwrap();
        let (body, result) = ctx.expand_predicate_func(func_def, &args).unwrap();

        assert_eq!(result, "G");
        assert_eq!(body.len(), 2);

        // First literal: parent(alice, P)
        if let BodyLiteral::Positive(atom) = &body[0] {
            assert_eq!(atom.predicate, "parent");
            assert!(matches!(&atom.terms[0], Term::Variable(v) if v == "alice"));
            assert!(matches!(&atom.terms[1], Term::Variable(v) if v == "P"));
        } else {
            panic!("Expected Positive literal for first body");
        }

        // Second literal: parent(P, G)
        if let BodyLiteral::Positive(atom) = &body[1] {
            assert_eq!(atom.predicate, "parent");
            assert!(matches!(&atom.terms[0], Term::Variable(v) if v == "P"));
            assert!(matches!(&atom.terms[1], Term::Variable(v) if v == "G"));
        } else {
            panic!("Expected Positive literal for second body");
        }
    }

    #[test]
    fn test_is_predicate_func() {
        let mut reg = FunctionRegistry::new();

        // Arithmetic function
        let arith_func = FuncDef {
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

        // Predicate function
        let pred_func = FuncDef {
            name: "get_parent".to_string(),
            params: vec![FuncParam {
                name: "X".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Predicate {
                result: "P".to_string(),
                body: vec![BodyLiteral::Positive(Atom {
                    predicate: "parent".to_string(),
                    terms: vec![
                        Term::Variable("X".to_string()),
                        Term::Variable("P".to_string()),
                    ],
                })],
            },
            is_private: false,
        };

        reg.register(arith_func).unwrap();
        reg.register(pred_func).unwrap();

        let ctx = ExpansionContext::new(&reg, 100);

        assert!(!ctx.is_predicate_func("double"));
        assert!(ctx.is_predicate_func("get_parent"));
        assert!(!ctx.is_predicate_func("nonexistent"));
    }
}
