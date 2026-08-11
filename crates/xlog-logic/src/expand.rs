//! Inline expansion of user-defined functions.

use crate::ast::{
    ArithExpr, Atom, BodyLiteral, Comparison, Constraint, FuncBody, FuncDef, IsExpr, Term, Univ,
};
use crate::function::{FunctionError, FunctionRegistry};
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
struct ExpandedExpression {
    generated_literals: Vec<BodyLiteral>,
    expression: ArithExpr,
}

impl ExpandedExpression {
    fn value(expression: ArithExpr) -> Self {
        Self {
            generated_literals: Vec::new(),
            expression,
        }
    }
}

/// Context for inline expansion of user-defined functions.
pub struct ExpansionContext<'a> {
    registry: &'a FunctionRegistry,
    depth: u32,
    max_depth: u32,
    fresh_counter: usize,
}

impl<'a> ExpansionContext<'a> {
    /// Create an expansion context with the given function registry and recursion limit.
    pub fn new(registry: &'a FunctionRegistry, max_depth: u32) -> Self {
        Self {
            registry,
            depth: 0,
            max_depth,
            fresh_counter: 0,
        }
    }

    /// Expand a scalar function call to its body with arguments substituted.
    ///
    /// Calls that produce relational literals require a surrounding rule-like
    /// body and are rejected by this expression-only API.
    pub fn expand_call(
        &mut self,
        name: &str,
        args: &[ArithExpr],
    ) -> Result<ArithExpr, FunctionError> {
        if !self.registry.contains(name) {
            return Err(FunctionError::UndefinedFunction {
                name: name.to_string(),
            });
        }

        let call = ArithExpr::FuncCall {
            name: name.to_string(),
            args: args.to_vec(),
        };
        let mut used_variables = call
            .variables()
            .into_iter()
            .map(ToOwned::to_owned)
            .collect();
        let expanded =
            self.expand_expr_for_rule(&call, &HashMap::new(), &mut used_variables, false)?;
        if expanded.generated_literals.is_empty() {
            Ok(expanded.expression)
        } else {
            Err(FunctionError::PredicateBodyRequiresRuleContext {
                name: name.to_string(),
            })
        }
    }

    fn check_arity(func: &FuncDef, args: &[ArithExpr]) -> Result<(), FunctionError> {
        if func.params.len() == args.len() {
            return Ok(());
        }
        Err(FunctionError::ArityMismatch {
            name: func.name.clone(),
            expected: func.params.len(),
            received: args.len(),
        })
    }

    fn fresh_variable(
        &mut self,
        function_name: &str,
        source_name: &str,
        used_variables: &mut HashSet<String>,
    ) -> String {
        loop {
            let candidate = format!(
                "__XLOG_FUNCTION_{}_{}_{}",
                function_name.to_ascii_uppercase(),
                source_name,
                self.fresh_counter
            );
            self.fresh_counter += 1;
            if used_variables.insert(candidate.clone()) {
                return candidate;
            }
        }
    }

    /// Expand a predicate body into literals for the surrounding rule or constraint.
    fn expand_predicate_func(
        &mut self,
        func: &FuncDef,
        args: &[ArithExpr],
        used_variables: &mut HashSet<String>,
    ) -> Result<ExpandedExpression, FunctionError> {
        let FuncBody::Predicate { result, body } = &func.body else {
            return Err(FunctionError::PredicateBodyRequiresRuleContext {
                name: func.name.clone(),
            });
        };

        let mut subst: HashMap<String, ArithExpr> = func
            .params
            .iter()
            .zip(args)
            .map(|(param, arg)| (param.name.clone(), arg.clone()))
            .collect();
        let parameter_names: HashSet<&str> = func
            .params
            .iter()
            .map(|param| param.name.as_str())
            .collect();
        let mut local_names = Vec::new();
        let mut seen_locals = HashSet::new();

        if !parameter_names.contains(result.as_str()) && seen_locals.insert(result.clone()) {
            local_names.push(result.clone());
        }
        for literal in body {
            for variable in literal.variables() {
                if !parameter_names.contains(variable) && seen_locals.insert(variable.to_string()) {
                    local_names.push(variable.to_string());
                }
            }
        }

        for local in local_names {
            let fresh = self.fresh_variable(&func.name, &local, used_variables);
            subst.insert(local, ArithExpr::Variable(fresh));
        }

        let substituted_body = body
            .iter()
            .map(|literal| self.substitute_literal(&func.name, literal, &subst))
            .collect::<Result<Vec<_>, _>>()?;
        let mut generated_literals = Vec::new();
        for literal in substituted_body {
            generated_literals.extend(self.expand_literal_for_rule(&literal, used_variables)?);
        }

        let expression = subst
            .get(result)
            .cloned()
            .unwrap_or_else(|| ArithExpr::Variable(result.clone()));
        Ok(ExpandedExpression {
            generated_literals,
            expression,
        })
    }

    fn substitute_literal(
        &self,
        function_name: &str,
        lit: &BodyLiteral,
        subst: &HashMap<String, ArithExpr>,
    ) -> Result<BodyLiteral, FunctionError> {
        Ok(match lit {
            BodyLiteral::Positive(atom) => {
                BodyLiteral::Positive(self.substitute_atom(function_name, atom, subst)?)
            }
            BodyLiteral::Negated(atom) => {
                BodyLiteral::Negated(self.substitute_atom(function_name, atom, subst)?)
            }
            BodyLiteral::Epistemic(lit) => BodyLiteral::Epistemic(crate::ast::EpistemicLiteral {
                op: lit.op,
                negated: lit.negated,
                atom: self.substitute_atom(function_name, &lit.atom, subst)?,
            }),
            BodyLiteral::Comparison(cmp) => BodyLiteral::Comparison(Comparison {
                left: self.substitute_term(function_name, &cmp.left, subst)?,
                op: cmp.op,
                right: self.substitute_term(function_name, &cmp.right, subst)?,
            }),
            BodyLiteral::IsExpr(is_expr) => BodyLiteral::IsExpr(IsExpr {
                target: self.substitute_binding_target(function_name, &is_expr.target, subst)?,
                expr: self.substitute_arith_expr(&is_expr.expr, subst),
            }),
            BodyLiteral::Univ(univ) => BodyLiteral::Univ(Univ {
                term: self.substitute_term(function_name, &univ.term, subst)?,
                parts: self.substitute_term(function_name, &univ.parts, subst)?,
            }),
        })
    }

    fn substitute_atom(
        &self,
        function_name: &str,
        atom: &Atom,
        subst: &HashMap<String, ArithExpr>,
    ) -> Result<Atom, FunctionError> {
        Ok(Atom {
            predicate: atom.predicate.clone(),
            terms: atom
                .terms
                .iter()
                .map(|term| self.substitute_term(function_name, term, subst))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn substitute_term(
        &self,
        function_name: &str,
        term: &Term,
        subst: &HashMap<String, ArithExpr>,
    ) -> Result<Term, FunctionError> {
        Ok(match term {
            Term::Variable(name) => match subst.get(name) {
                Some(ArithExpr::Variable(new_name)) => Term::Variable(new_name.clone()),
                Some(ArithExpr::Integer(value)) => Term::Integer(*value),
                Some(ArithExpr::Float(value)) => Term::Float(*value),
                Some(_) => {
                    return Err(FunctionError::UnsupportedPredicateTermArgument {
                        name: function_name.to_string(),
                        parameter: name.clone(),
                    })
                }
                None => term.clone(),
            },
            Term::List(items) => Term::List(
                items
                    .iter()
                    .map(|item| self.substitute_term(function_name, item, subst))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Term::Cons { head, tail } => Term::Cons {
                head: Box::new(self.substitute_term(function_name, head, subst)?),
                tail: Box::new(self.substitute_term(function_name, tail, subst)?),
            },
            Term::Compound { functor, args } => Term::Compound {
                functor: functor.clone(),
                args: args
                    .iter()
                    .map(|arg| self.substitute_term(function_name, arg, subst))
                    .collect::<Result<Vec<_>, _>>()?,
            },
            Term::Aggregate(aggregate) => Term::Aggregate(crate::ast::AggExpr {
                op: aggregate.op,
                variable: self.substitute_binding_target(
                    function_name,
                    &aggregate.variable,
                    subst,
                )?,
            }),
            Term::Anonymous
            | Term::Integer(_)
            | Term::Float(_)
            | Term::String(_)
            | Term::Symbol(_)
            | Term::PredRef(_) => term.clone(),
        })
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

    fn substitute_binding_target(
        &self,
        function_name: &str,
        variable: &str,
        subst: &HashMap<String, ArithExpr>,
    ) -> Result<String, FunctionError> {
        match subst.get(variable) {
            Some(ArithExpr::Variable(new_name)) => Ok(new_name.clone()),
            Some(_) => Err(FunctionError::InvalidPredicateBindingTarget {
                name: function_name.to_string(),
                parameter: variable.to_string(),
            }),
            None => Ok(variable.to_string()),
        }
    }

    fn expand_literal_for_rule(
        &mut self,
        literal: &BodyLiteral,
        used_variables: &mut HashSet<String>,
    ) -> Result<Vec<BodyLiteral>, FunctionError> {
        let BodyLiteral::IsExpr(binding) = literal else {
            return Ok(vec![literal.clone()]);
        };

        let expanded =
            self.expand_expr_for_rule(&binding.expr, &HashMap::new(), used_variables, false)?;
        let mut literals = expanded.generated_literals;
        literals.push(BodyLiteral::IsExpr(IsExpr {
            target: binding.target.clone(),
            expr: expanded.expression,
        }));
        Ok(literals)
    }

    fn expand_expr_for_rule(
        &mut self,
        expression: &ArithExpr,
        subst: &HashMap<String, ArithExpr>,
        used_variables: &mut HashSet<String>,
        in_conditional_branch: bool,
    ) -> Result<ExpandedExpression, FunctionError> {
        match expression {
            ArithExpr::Variable(name) => Ok(ExpandedExpression::value(
                subst
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| expression.clone()),
            )),
            ArithExpr::Integer(_) | ArithExpr::Float(_) => {
                Ok(ExpandedExpression::value(expression.clone()))
            }
            ArithExpr::FuncCall { name, args } => {
                if let Some(func) = self.registry.get(name) {
                    Self::check_arity(func, args)?;
                }

                let mut generated_literals = Vec::new();
                let mut expanded_args = Vec::with_capacity(args.len());
                for argument in args {
                    let expanded = self.expand_expr_for_rule(
                        argument,
                        subst,
                        used_variables,
                        in_conditional_branch,
                    )?;
                    generated_literals.extend(expanded.generated_literals);
                    expanded_args.push(expanded.expression);
                }

                if !self.registry.contains(name) {
                    return Ok(ExpandedExpression {
                        generated_literals,
                        expression: ArithExpr::FuncCall {
                            name: name.clone(),
                            args: expanded_args,
                        },
                    });
                }

                let expanded = self.expand_function_for_rule(
                    name,
                    &expanded_args,
                    used_variables,
                    in_conditional_branch,
                )?;
                generated_literals.extend(expanded.generated_literals);
                Ok(ExpandedExpression {
                    generated_literals,
                    expression: expanded.expression,
                })
            }
            ArithExpr::Add(left, right) => self.expand_binary_for_rule(
                left,
                right,
                subst,
                used_variables,
                in_conditional_branch,
                ArithExpr::Add,
            ),
            ArithExpr::Sub(left, right) => self.expand_binary_for_rule(
                left,
                right,
                subst,
                used_variables,
                in_conditional_branch,
                ArithExpr::Sub,
            ),
            ArithExpr::Mul(left, right) => self.expand_binary_for_rule(
                left,
                right,
                subst,
                used_variables,
                in_conditional_branch,
                ArithExpr::Mul,
            ),
            ArithExpr::Div(left, right) => self.expand_binary_for_rule(
                left,
                right,
                subst,
                used_variables,
                in_conditional_branch,
                ArithExpr::Div,
            ),
            ArithExpr::Mod(left, right) => self.expand_binary_for_rule(
                left,
                right,
                subst,
                used_variables,
                in_conditional_branch,
                ArithExpr::Mod,
            ),
            ArithExpr::Min(left, right) => self.expand_binary_for_rule(
                left,
                right,
                subst,
                used_variables,
                in_conditional_branch,
                ArithExpr::Min,
            ),
            ArithExpr::Max(left, right) => self.expand_binary_for_rule(
                left,
                right,
                subst,
                used_variables,
                in_conditional_branch,
                ArithExpr::Max,
            ),
            ArithExpr::Pow(left, right) => self.expand_binary_for_rule(
                left,
                right,
                subst,
                used_variables,
                in_conditional_branch,
                ArithExpr::Pow,
            ),
            ArithExpr::Abs(inner) => {
                let expanded =
                    self.expand_expr_for_rule(inner, subst, used_variables, in_conditional_branch)?;
                Ok(ExpandedExpression {
                    generated_literals: expanded.generated_literals,
                    expression: ArithExpr::Abs(Box::new(expanded.expression)),
                })
            }
            ArithExpr::Cast(inner, target) => {
                let expanded =
                    self.expand_expr_for_rule(inner, subst, used_variables, in_conditional_branch)?;
                Ok(ExpandedExpression {
                    generated_literals: expanded.generated_literals,
                    expression: ArithExpr::Cast(Box::new(expanded.expression), *target),
                })
            }
            ArithExpr::Conditional {
                cond_left,
                cond_op,
                cond_right,
                then_expr,
                else_expr,
            } => {
                let left = self.expand_expr_for_rule(
                    cond_left,
                    subst,
                    used_variables,
                    in_conditional_branch,
                )?;
                let right = self.expand_expr_for_rule(
                    cond_right,
                    subst,
                    used_variables,
                    in_conditional_branch,
                )?;
                let then_value =
                    self.expand_expr_for_rule(then_expr, subst, used_variables, true)?;
                let else_value =
                    self.expand_expr_for_rule(else_expr, subst, used_variables, true)?;
                let mut generated_literals = left.generated_literals;
                generated_literals.extend(right.generated_literals);
                generated_literals.extend(then_value.generated_literals);
                generated_literals.extend(else_value.generated_literals);
                Ok(ExpandedExpression {
                    generated_literals,
                    expression: ArithExpr::Conditional {
                        cond_left: Box::new(left.expression),
                        cond_op: *cond_op,
                        cond_right: Box::new(right.expression),
                        then_expr: Box::new(then_value.expression),
                        else_expr: Box::new(else_value.expression),
                    },
                })
            }
        }
    }

    fn expand_binary_for_rule(
        &mut self,
        left: &ArithExpr,
        right: &ArithExpr,
        subst: &HashMap<String, ArithExpr>,
        used_variables: &mut HashSet<String>,
        in_conditional_branch: bool,
        constructor: fn(Box<ArithExpr>, Box<ArithExpr>) -> ArithExpr,
    ) -> Result<ExpandedExpression, FunctionError> {
        let left = self.expand_expr_for_rule(left, subst, used_variables, in_conditional_branch)?;
        let right =
            self.expand_expr_for_rule(right, subst, used_variables, in_conditional_branch)?;
        let mut generated_literals = left.generated_literals;
        generated_literals.extend(right.generated_literals);
        Ok(ExpandedExpression {
            generated_literals,
            expression: constructor(Box::new(left.expression), Box::new(right.expression)),
        })
    }

    fn expand_function_for_rule(
        &mut self,
        name: &str,
        args: &[ArithExpr],
        used_variables: &mut HashSet<String>,
        in_conditional_branch: bool,
    ) -> Result<ExpandedExpression, FunctionError> {
        if self.depth >= self.max_depth {
            return Err(FunctionError::MaxRecursionDepth {
                name: name.to_string(),
                depth: self.max_depth,
            });
        }
        let func =
            self.registry
                .get(name)
                .cloned()
                .ok_or_else(|| FunctionError::UndefinedFunction {
                    name: name.to_string(),
                })?;

        if in_conditional_branch && matches!(func.body, FuncBody::Predicate { .. }) {
            return Err(FunctionError::PredicateCallInConditionalBranch {
                name: func.name.clone(),
            });
        }

        let subst: HashMap<String, ArithExpr> = func
            .params
            .iter()
            .zip(args)
            .map(|(param, argument)| (param.name.clone(), argument.clone()))
            .collect();
        self.depth += 1;
        let expanded = self.expand_function_body_for_rule(
            &func,
            &func.body,
            args,
            &subst,
            used_variables,
            in_conditional_branch,
        );
        self.depth -= 1;
        expanded
    }

    fn expand_function_body_for_rule(
        &mut self,
        func: &FuncDef,
        body: &FuncBody,
        args: &[ArithExpr],
        subst: &HashMap<String, ArithExpr>,
        used_variables: &mut HashSet<String>,
        in_conditional_branch: bool,
    ) -> Result<ExpandedExpression, FunctionError> {
        match body {
            FuncBody::Arithmetic(expression) => {
                self.expand_expr_for_rule(expression, subst, used_variables, in_conditional_branch)
            }
            FuncBody::Predicate { .. } => {
                if in_conditional_branch {
                    return Err(FunctionError::PredicateCallInConditionalBranch {
                        name: func.name.clone(),
                    });
                }
                self.expand_predicate_func(func, args, used_variables)
            }
            FuncBody::Conditional(conditional) => {
                let left = self.expand_expr_for_rule(
                    &conditional.cond_left,
                    subst,
                    used_variables,
                    in_conditional_branch,
                )?;
                let right = self.expand_expr_for_rule(
                    &conditional.cond_right,
                    subst,
                    used_variables,
                    in_conditional_branch,
                )?;
                let then_value = self.expand_function_body_for_rule(
                    func,
                    &conditional.then_branch,
                    args,
                    subst,
                    used_variables,
                    true,
                )?;
                let else_value = self.expand_function_body_for_rule(
                    func,
                    &conditional.else_branch,
                    args,
                    subst,
                    used_variables,
                    true,
                )?;
                let mut generated_literals = left.generated_literals;
                generated_literals.extend(right.generated_literals);
                generated_literals.extend(then_value.generated_literals);
                generated_literals.extend(else_value.generated_literals);
                Ok(ExpandedExpression {
                    generated_literals,
                    expression: ArithExpr::Conditional {
                        cond_left: Box::new(left.expression),
                        cond_op: conditional.cond_op,
                        cond_right: Box::new(right.expression),
                        then_expr: Box::new(then_value.expression),
                        else_expr: Box::new(else_value.expression),
                    },
                })
            }
        }
    }

    /// Check if a function has a predicate body.
    #[allow(dead_code)]
    pub(crate) fn is_predicate_func(&self, name: &str) -> bool {
        self.registry
            .get(name)
            .map(|f| matches!(f.body, FuncBody::Predicate { .. }))
            .unwrap_or(false)
    }
}

use crate::ast::{Program, Rule};

/// Expand user-defined function calls in ordinary rules and constraints.
///
/// Scalar calls become arithmetic expressions. Predicate-bodied calls also
/// contribute relational literals immediately before their source binding.
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
    let expanded_constraints: Result<Vec<Constraint>, FunctionError> = program
        .constraints
        .iter()
        .map(|constraint| expand_constraint_functions(&mut ctx, constraint))
        .collect();

    Ok(Program {
        rules: expanded_rules?,
        directives: program.directives.clone(),
        queries: program.queries.clone(),
        predicates: program.predicates.clone(),
        constraints: expanded_constraints?,
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
    let mut used_variables: HashSet<String> = rule
        .head
        .variables()
        .into_iter()
        .chain(rule.body.iter().flat_map(BodyLiteral::variables))
        .map(ToOwned::to_owned)
        .collect();
    let expanded_body = expand_body_functions(ctx, &rule.body, &mut used_variables)?;

    Ok(Rule {
        head: rule.head.clone(),
        body: expanded_body,
    })
}

fn expand_constraint_functions(
    ctx: &mut ExpansionContext,
    constraint: &Constraint,
) -> Result<Constraint, FunctionError> {
    let mut used_variables: HashSet<String> = constraint
        .body
        .iter()
        .flat_map(BodyLiteral::variables)
        .map(ToOwned::to_owned)
        .collect();
    Ok(Constraint {
        body: expand_body_functions(ctx, &constraint.body, &mut used_variables)?,
    })
}

fn expand_body_functions(
    ctx: &mut ExpansionContext,
    body: &[BodyLiteral],
    used_variables: &mut HashSet<String>,
) -> Result<Vec<BodyLiteral>, FunctionError> {
    let mut expanded_body = Vec::new();
    for literal in body {
        expanded_body.extend(ctx.expand_literal_for_rule(literal, used_variables)?);
    }
    Ok(expanded_body)
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

        let mut ctx = ExpansionContext::new(&reg, 100);

        // Call get_parent with "alice"
        let args = vec![ArithExpr::Variable("alice".to_string())];
        let func_def = reg.get("get_parent").unwrap();
        let mut used = HashSet::from(["alice".to_string()]);
        let expanded = ctx
            .expand_predicate_func(func_def, &args, &mut used)
            .unwrap();
        let body = expanded.generated_literals;
        let ArithExpr::Variable(result) = expanded.expression else {
            panic!("Expected variable result")
        };

        assert_ne!(result, "P");
        assert_eq!(body.len(), 1);

        // Check the expanded literal
        if let BodyLiteral::Positive(atom) = &body[0] {
            assert_eq!(atom.predicate, "parent");
            assert!(matches!(&atom.terms[0], Term::Variable(v) if v == "alice"));
            assert!(matches!(&atom.terms[1], Term::Variable(v) if v == &result));
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

        let mut ctx = ExpansionContext::new(&reg, 100);

        // Call get_child with integer constant
        let args = vec![ArithExpr::Integer(42)];
        let func_def = reg.get("get_child").unwrap();
        let mut used = HashSet::new();
        let expanded = ctx
            .expand_predicate_func(func_def, &args, &mut used)
            .unwrap();
        let body = expanded.generated_literals;
        let ArithExpr::Variable(result) = expanded.expression else {
            panic!("Expected variable result")
        };

        assert_ne!(result, "C");
        assert_eq!(body.len(), 1);

        // Check the expanded literal has integer substituted
        if let BodyLiteral::Positive(atom) = &body[0] {
            assert_eq!(atom.predicate, "parent");
            assert!(matches!(&atom.terms[0], Term::Variable(v) if v == &result));
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

        let mut ctx = ExpansionContext::new(&reg, 100);

        let args = vec![ArithExpr::Variable("alice".to_string())];
        let func_def = reg.get("get_grandparent").unwrap();
        let mut used = HashSet::from(["alice".to_string()]);
        let expanded = ctx
            .expand_predicate_func(func_def, &args, &mut used)
            .unwrap();
        let body = expanded.generated_literals;
        let ArithExpr::Variable(result) = expanded.expression else {
            panic!("Expected variable result")
        };

        assert_ne!(result, "G");
        assert_eq!(body.len(), 2);

        // First literal: parent(alice, P)
        if let BodyLiteral::Positive(atom) = &body[0] {
            assert_eq!(atom.predicate, "parent");
            assert!(matches!(&atom.terms[0], Term::Variable(v) if v == "alice"));
            assert!(matches!(&atom.terms[1], Term::Variable(v) if v != "P"));
        } else {
            panic!("Expected Positive literal for first body");
        }

        // Second literal: parent(P, G)
        if let BodyLiteral::Positive(atom) = &body[1] {
            assert_eq!(atom.predicate, "parent");
            assert_eq!(atom.terms[0], body[0].atom().unwrap().terms[1]);
            assert!(matches!(&atom.terms[1], Term::Variable(v) if v == &result));
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
