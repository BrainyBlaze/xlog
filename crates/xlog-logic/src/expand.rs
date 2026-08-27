//! Inline expansion of user-defined functions.

use crate::ast::{
    ArithExpr, Atom, BodyLiteral, Comparison, Constraint, FuncBody, FuncDef, IsExpr, Term, Univ,
};
use crate::function::{is_builtin, FunctionError, FunctionRegistry};
use std::collections::{HashMap, HashSet};

const GENERATED_FUNCTION_VARIABLE_PREFIX: &str = "__XLOG_FUNCTION_";

pub(crate) fn generated_function_variable_name(
    function_name: &str,
    source_name: &str,
    counter: usize,
) -> String {
    format!(
        "{GENERATED_FUNCTION_VARIABLE_PREFIX}{}_{}_{}",
        function_name.to_ascii_uppercase(),
        source_name,
        counter
    )
}

pub(crate) fn generated_function_variable_source<'a>(
    generated_name: &'a str,
    function_name: &str,
) -> Option<&'a str> {
    let prefix = format!(
        "{GENERATED_FUNCTION_VARIABLE_PREFIX}{}_",
        function_name.to_ascii_uppercase()
    );
    let (source_name, counter) = generated_name.strip_prefix(&prefix)?.rsplit_once('_')?;
    (!source_name.is_empty()
        && !counter.is_empty()
        && counter.chars().all(|character| character.is_ascii_digit()))
    .then_some(source_name)
}

#[derive(Debug)]
struct ExpandedExpression {
    generated_literals: Vec<BodyLiteral>,
    expression: ArithExpr,
}

fn inline_trailing_predicate_result_binding(
    mut literals: Vec<BodyLiteral>,
    result: ArithExpr,
) -> (Vec<BodyLiteral>, ArithExpr) {
    let ArithExpr::Variable(result_name) = &result else {
        return (literals, result);
    };
    let Some(BodyLiteral::IsExpr(binding)) = literals.last() else {
        return (literals, result);
    };
    let ArithExpr::Variable(source_name) = &binding.expr else {
        return (literals, result);
    };
    if !predicate_prefix_binds_variable(&literals[..literals.len() - 1], source_name)
        || binding.target != *result_name
        || binding
            .expr
            .variables()
            .into_iter()
            .any(|name| name == result_name)
        || literals[..literals.len() - 1].iter().any(|literal| {
            literal
                .variables()
                .into_iter()
                .any(|name| name == result_name)
        })
    {
        return (literals, result);
    }

    let Some(BodyLiteral::IsExpr(binding)) = literals.pop() else {
        unreachable!("trailing predicate result binding was checked above")
    };
    (literals, binding.expr)
}

fn predicate_prefix_binds_variable(literals: &[BodyLiteral], wanted: &str) -> bool {
    let mut bound = HashSet::new();
    for literal in literals {
        match literal {
            BodyLiteral::Positive(atom) => {
                bound.extend(atom.variables().into_iter().map(ToOwned::to_owned));
            }
            BodyLiteral::IsExpr(binding)
                if binding
                    .expr
                    .variables()
                    .into_iter()
                    .all(|name| bound.contains(name)) =>
            {
                bound.insert(binding.target.clone());
            }
            BodyLiteral::Negated(_)
            | BodyLiteral::Epistemic(_)
            | BodyLiteral::Comparison(_)
            | BodyLiteral::IsExpr(_)
            | BodyLiteral::Univ(_) => {}
        }
    }
    bound.contains(wanted)
}

impl ExpandedExpression {
    fn value(expression: ArithExpr) -> Self {
        Self {
            generated_literals: Vec::new(),
            expression,
        }
    }
}

type BinaryExpressionConstructor = fn(Box<ArithExpr>, Box<ArithExpr>) -> ArithExpr;

enum ExpansionTask {
    Expression {
        expression: ArithExpr,
        subst: HashMap<String, ArithExpr>,
        in_conditional_branch: bool,
    },
    FinishCall {
        name: String,
        argument_count: usize,
        in_conditional_branch: bool,
    },
    EnterFunction {
        name: String,
        args: Vec<ArithExpr>,
        in_conditional_branch: bool,
    },
    FunctionBody {
        function_name: String,
        body: FuncBody,
        subst: HashMap<String, ArithExpr>,
        in_conditional_branch: bool,
    },
    LeaveFunction,
    PrependGenerated {
        literals: Vec<BodyLiteral>,
    },
    FinishBinary {
        constructor: BinaryExpressionConstructor,
    },
    FinishAbs,
    FinishCast(xlog_core::ScalarType),
    FinishConditional(crate::ast::CompOp),
    PredicateLiteral(PreparedPredicateLiteral),
    FinishPredicateBinding {
        target: String,
    },
    FinishPredicateBody {
        literal_count: usize,
        result: ArithExpr,
    },
}

enum PreparedPredicateLiteral {
    Literal(BodyLiteral),
    Binding {
        target: String,
        expression: ArithExpr,
        subst: HashMap<String, ArithExpr>,
    },
}

enum TermSubstitutionTask<'a> {
    Term(&'a Term),
    FinishList(usize),
    FinishCons,
    FinishCompound {
        functor: String,
        argument_count: usize,
    },
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
            let candidate =
                generated_function_variable_name(function_name, source_name, self.fresh_counter);
            self.fresh_counter += 1;
            if used_variables.insert(candidate.clone()) {
                return candidate;
            }
        }
    }

    /// Freshen and substitute a predicate body before the expansion machine visits it.
    fn prepare_predicate_func(
        &mut self,
        function_name: &str,
        result: String,
        body: Vec<BodyLiteral>,
        mut subst: HashMap<String, ArithExpr>,
        used_variables: &mut HashSet<String>,
    ) -> Result<(Vec<PreparedPredicateLiteral>, ArithExpr), FunctionError> {
        let parameter_names: HashSet<String> = subst.keys().cloned().collect();
        let mut local_names = Vec::new();
        let mut seen_locals = HashSet::new();

        if !parameter_names.contains(&result) && seen_locals.insert(result.clone()) {
            local_names.push(result.clone());
        }
        for literal in &body {
            for variable in literal.variables() {
                if !parameter_names.contains(variable) && seen_locals.insert(variable.to_string()) {
                    local_names.push(variable.to_string());
                }
            }
        }

        for local in local_names {
            let fresh = self.fresh_variable(function_name, &local, used_variables);
            subst.insert(local, ArithExpr::Variable(fresh));
        }

        let substituted_body = body
            .into_iter()
            .map(|literal| {
                if let BodyLiteral::IsExpr(binding) = literal {
                    Ok(PreparedPredicateLiteral::Binding {
                        target: self.substitute_binding_target(
                            function_name,
                            &binding.target,
                            &subst,
                        )?,
                        expression: binding.expr,
                        subst: subst.clone(),
                    })
                } else {
                    self.substitute_literal(function_name, &literal, &subst)
                        .map(PreparedPredicateLiteral::Literal)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expression = subst
            .get(&result)
            .cloned()
            .unwrap_or(ArithExpr::Variable(result));
        Ok((substituted_body, expression))
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
            BodyLiteral::IsExpr(_) => unreachable!("predicate bindings are prepared separately"),
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
        let mut tasks = vec![TermSubstitutionTask::Term(term)];
        let mut values = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                TermSubstitutionTask::Term(term) => match term {
                    Term::Variable(name) => values.push(match subst.get(name) {
                        Some(ArithExpr::Variable(new_name)) => Term::Variable(new_name.clone()),
                        Some(ArithExpr::Integer(value)) => Term::Integer(*value),
                        Some(ArithExpr::Float(value)) => Term::Float(*value),
                        Some(_) => {
                            return Err(FunctionError::UnsupportedPredicateTermArgument {
                                name: function_name.to_string(),
                                parameter: name.clone(),
                            });
                        }
                        None => Term::Variable(name.clone()),
                    }),
                    Term::List(items) => {
                        tasks.push(TermSubstitutionTask::FinishList(items.len()));
                        for item in items.iter().rev() {
                            tasks.push(TermSubstitutionTask::Term(item));
                        }
                    }
                    Term::Cons { head, tail } => {
                        tasks.push(TermSubstitutionTask::FinishCons);
                        tasks.push(TermSubstitutionTask::Term(tail));
                        tasks.push(TermSubstitutionTask::Term(head));
                    }
                    Term::Compound { functor, args } => {
                        tasks.push(TermSubstitutionTask::FinishCompound {
                            functor: functor.clone(),
                            argument_count: args.len(),
                        });
                        for argument in args.iter().rev() {
                            tasks.push(TermSubstitutionTask::Term(argument));
                        }
                    }
                    Term::Aggregate(aggregate) => {
                        values.push(Term::Aggregate(crate::ast::AggExpr {
                            op: aggregate.op,
                            variable: self.substitute_binding_target(
                                function_name,
                                &aggregate.variable,
                                subst,
                            )?,
                        }));
                    }
                    Term::Anonymous => values.push(Term::Anonymous),
                    Term::Integer(value) => values.push(Term::Integer(*value)),
                    Term::Float(value) => values.push(Term::Float(*value)),
                    Term::String(value) => values.push(Term::String(value.clone())),
                    Term::Symbol(value) => values.push(Term::Symbol(*value)),
                    Term::PredRef(value) => values.push(Term::PredRef(value.clone())),
                },
                TermSubstitutionTask::FinishList(item_count) => {
                    let start = values
                        .len()
                        .checked_sub(item_count)
                        .expect("list items produce substituted terms");
                    let items = values.split_off(start);
                    values.push(Term::List(items));
                }
                TermSubstitutionTask::FinishCons => {
                    let tail = values.pop().expect("cons tail produces a substituted term");
                    let head = values.pop().expect("cons head produces a substituted term");
                    values.push(Term::Cons {
                        head: Box::new(head),
                        tail: Box::new(tail),
                    });
                }
                TermSubstitutionTask::FinishCompound {
                    functor,
                    argument_count,
                } => {
                    let start = values
                        .len()
                        .checked_sub(argument_count)
                        .expect("compound arguments produce substituted terms");
                    let args = values.split_off(start);
                    values.push(Term::Compound { functor, args });
                }
            }
        }

        let substituted = values.pop().expect("term substitution produces one term");
        debug_assert!(values.is_empty());
        Ok(substituted)
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
        let saved_depth = self.depth;
        let saved_fresh_counter = self.fresh_counter;
        let saved_used_variables = used_variables.clone();
        let result =
            self.run_expansion_machine(expression, subst, used_variables, in_conditional_branch);
        self.depth = saved_depth;
        if result.is_err() {
            self.fresh_counter = saved_fresh_counter;
            *used_variables = saved_used_variables;
        }
        result
    }

    fn run_expansion_machine(
        &mut self,
        expression: &ArithExpr,
        subst: &HashMap<String, ArithExpr>,
        used_variables: &mut HashSet<String>,
        in_conditional_branch: bool,
    ) -> Result<ExpandedExpression, FunctionError> {
        let mut tasks = vec![ExpansionTask::Expression {
            expression: expression.clone(),
            subst: subst.clone(),
            in_conditional_branch,
        }];
        let mut values = Vec::new();
        let mut predicate_literal_values: Vec<Vec<BodyLiteral>> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                ExpansionTask::Expression {
                    expression,
                    subst,
                    in_conditional_branch,
                } => match expression {
                    ArithExpr::Variable(name) => values.push(ExpandedExpression::value(
                        subst
                            .get(&name)
                            .cloned()
                            .unwrap_or(ArithExpr::Variable(name)),
                    )),
                    ArithExpr::Integer(_) | ArithExpr::Float(_) => {
                        values.push(ExpandedExpression::value(expression));
                    }
                    ArithExpr::FuncCall { name, args } => {
                        if let Some(func) = self.registry.get(&name) {
                            Self::check_arity(func, &args)?;
                        }
                        tasks.push(ExpansionTask::FinishCall {
                            name,
                            argument_count: args.len(),
                            in_conditional_branch,
                        });
                        for argument in args.into_iter().rev() {
                            tasks.push(ExpansionTask::Expression {
                                expression: argument,
                                subst: subst.clone(),
                                in_conditional_branch,
                            });
                        }
                    }
                    ArithExpr::Add(left, right) => {
                        Self::schedule_binary(
                            &mut tasks,
                            *left,
                            *right,
                            subst,
                            in_conditional_branch,
                            ArithExpr::Add,
                        );
                    }
                    ArithExpr::Sub(left, right) => {
                        Self::schedule_binary(
                            &mut tasks,
                            *left,
                            *right,
                            subst,
                            in_conditional_branch,
                            ArithExpr::Sub,
                        );
                    }
                    ArithExpr::Mul(left, right) => {
                        Self::schedule_binary(
                            &mut tasks,
                            *left,
                            *right,
                            subst,
                            in_conditional_branch,
                            ArithExpr::Mul,
                        );
                    }
                    ArithExpr::Div(left, right) => {
                        Self::schedule_binary(
                            &mut tasks,
                            *left,
                            *right,
                            subst,
                            in_conditional_branch,
                            ArithExpr::Div,
                        );
                    }
                    ArithExpr::Mod(left, right) => {
                        Self::schedule_binary(
                            &mut tasks,
                            *left,
                            *right,
                            subst,
                            in_conditional_branch,
                            ArithExpr::Mod,
                        );
                    }
                    ArithExpr::Min(left, right) => {
                        Self::schedule_binary(
                            &mut tasks,
                            *left,
                            *right,
                            subst,
                            in_conditional_branch,
                            ArithExpr::Min,
                        );
                    }
                    ArithExpr::Max(left, right) => {
                        Self::schedule_binary(
                            &mut tasks,
                            *left,
                            *right,
                            subst,
                            in_conditional_branch,
                            ArithExpr::Max,
                        );
                    }
                    ArithExpr::Pow(left, right) => {
                        Self::schedule_binary(
                            &mut tasks,
                            *left,
                            *right,
                            subst,
                            in_conditional_branch,
                            ArithExpr::Pow,
                        );
                    }
                    ArithExpr::Abs(inner) => {
                        tasks.push(ExpansionTask::FinishAbs);
                        tasks.push(ExpansionTask::Expression {
                            expression: *inner,
                            subst,
                            in_conditional_branch,
                        });
                    }
                    ArithExpr::Cast(inner, target) => {
                        tasks.push(ExpansionTask::FinishCast(target));
                        tasks.push(ExpansionTask::Expression {
                            expression: *inner,
                            subst,
                            in_conditional_branch,
                        });
                    }
                    ArithExpr::Conditional {
                        cond_left,
                        cond_op,
                        cond_right,
                        then_expr,
                        else_expr,
                    } => {
                        tasks.push(ExpansionTask::FinishConditional(cond_op));
                        tasks.push(ExpansionTask::Expression {
                            expression: *else_expr,
                            subst: subst.clone(),
                            in_conditional_branch: true,
                        });
                        tasks.push(ExpansionTask::Expression {
                            expression: *then_expr,
                            subst: subst.clone(),
                            in_conditional_branch: true,
                        });
                        tasks.push(ExpansionTask::Expression {
                            expression: *cond_right,
                            subst: subst.clone(),
                            in_conditional_branch,
                        });
                        tasks.push(ExpansionTask::Expression {
                            expression: *cond_left,
                            subst,
                            in_conditional_branch,
                        });
                    }
                },
                ExpansionTask::FinishCall {
                    name,
                    argument_count,
                    in_conditional_branch,
                } => {
                    let start = values
                        .len()
                        .checked_sub(argument_count)
                        .expect("function arguments produce expansion values");
                    let argument_values = values.split_off(start);
                    let mut generated_literals = Vec::new();
                    let mut args = Vec::with_capacity(argument_count);
                    for argument in argument_values {
                        generated_literals.extend(argument.generated_literals);
                        args.push(argument.expression);
                    }
                    if self.registry.contains(&name) {
                        tasks.push(ExpansionTask::PrependGenerated {
                            literals: generated_literals,
                        });
                        tasks.push(ExpansionTask::EnterFunction {
                            name,
                            args,
                            in_conditional_branch,
                        });
                    } else if is_builtin(&name) {
                        values.push(ExpandedExpression {
                            generated_literals,
                            expression: ArithExpr::FuncCall { name, args },
                        });
                    } else {
                        return Err(FunctionError::UndefinedFunction { name });
                    }
                }
                ExpansionTask::EnterFunction {
                    name,
                    args,
                    in_conditional_branch,
                } => {
                    let func =
                        self.registry.get(&name).cloned().ok_or_else(|| {
                            FunctionError::UndefinedFunction { name: name.clone() }
                        })?;
                    Self::check_arity(&func, &args)?;
                    if self.depth >= self.max_depth {
                        return Err(FunctionError::MaxRecursionDepth {
                            name,
                            depth: self.max_depth,
                        });
                    }
                    if in_conditional_branch && matches!(func.body, FuncBody::Predicate { .. }) {
                        return Err(FunctionError::PredicateCallInConditionalBranch { name });
                    }

                    self.depth += 1;
                    let subst = func
                        .params
                        .iter()
                        .zip(&args)
                        .map(|(param, argument)| (param.name.clone(), argument.clone()))
                        .collect();
                    tasks.push(ExpansionTask::LeaveFunction);
                    tasks.push(ExpansionTask::FunctionBody {
                        function_name: func.name,
                        body: func.body,
                        subst,
                        in_conditional_branch,
                    });
                }
                ExpansionTask::FunctionBody {
                    function_name,
                    body,
                    subst,
                    in_conditional_branch,
                } => match body {
                    FuncBody::Arithmetic(expression) => {
                        tasks.push(ExpansionTask::Expression {
                            expression,
                            subst,
                            in_conditional_branch,
                        });
                    }
                    FuncBody::Predicate { result, body } => {
                        if in_conditional_branch {
                            return Err(FunctionError::PredicateCallInConditionalBranch {
                                name: function_name,
                            });
                        }
                        let (literals, result) = self.prepare_predicate_func(
                            &function_name,
                            result,
                            body,
                            subst,
                            used_variables,
                        )?;
                        tasks.push(ExpansionTask::FinishPredicateBody {
                            literal_count: literals.len(),
                            result,
                        });
                        for literal in literals.into_iter().rev() {
                            tasks.push(ExpansionTask::PredicateLiteral(literal));
                        }
                    }
                    FuncBody::Conditional(conditional) => {
                        tasks.push(ExpansionTask::FinishConditional(conditional.cond_op));
                        tasks.push(ExpansionTask::FunctionBody {
                            function_name: function_name.clone(),
                            body: *conditional.else_branch,
                            subst: subst.clone(),
                            in_conditional_branch: true,
                        });
                        tasks.push(ExpansionTask::FunctionBody {
                            function_name,
                            body: *conditional.then_branch,
                            subst: subst.clone(),
                            in_conditional_branch: true,
                        });
                        tasks.push(ExpansionTask::Expression {
                            expression: conditional.cond_right,
                            subst: subst.clone(),
                            in_conditional_branch,
                        });
                        tasks.push(ExpansionTask::Expression {
                            expression: conditional.cond_left,
                            subst,
                            in_conditional_branch,
                        });
                    }
                },
                ExpansionTask::LeaveFunction => {
                    debug_assert!(self.depth > 0);
                    self.depth -= 1;
                }
                ExpansionTask::PrependGenerated { mut literals } => {
                    let mut expanded = values
                        .pop()
                        .expect("function call produces an expansion value");
                    literals.append(&mut expanded.generated_literals);
                    expanded.generated_literals = literals;
                    values.push(expanded);
                }
                ExpansionTask::FinishBinary { constructor } => {
                    let right = values.pop().expect("right expression is expanded");
                    let mut left = values.pop().expect("left expression is expanded");
                    left.generated_literals.extend(right.generated_literals);
                    left.expression =
                        constructor(Box::new(left.expression), Box::new(right.expression));
                    values.push(left);
                }
                ExpansionTask::FinishAbs => {
                    let mut expanded = values.pop().expect("absolute-value operand is expanded");
                    expanded.expression = ArithExpr::Abs(Box::new(expanded.expression));
                    values.push(expanded);
                }
                ExpansionTask::FinishCast(target) => {
                    let mut expanded = values.pop().expect("cast operand is expanded");
                    expanded.expression = ArithExpr::Cast(Box::new(expanded.expression), target);
                    values.push(expanded);
                }
                ExpansionTask::FinishConditional(cond_op) => {
                    let else_value = values.pop().expect("else branch is expanded");
                    let then_value = values.pop().expect("then branch is expanded");
                    let right = values.pop().expect("condition right side is expanded");
                    let mut left = values.pop().expect("condition left side is expanded");
                    left.generated_literals.extend(right.generated_literals);
                    left.generated_literals
                        .extend(then_value.generated_literals);
                    left.generated_literals
                        .extend(else_value.generated_literals);
                    left.expression = ArithExpr::Conditional {
                        cond_left: Box::new(left.expression),
                        cond_op,
                        cond_right: Box::new(right.expression),
                        then_expr: Box::new(then_value.expression),
                        else_expr: Box::new(else_value.expression),
                    };
                    values.push(left);
                }
                ExpansionTask::PredicateLiteral(literal) => match literal {
                    PreparedPredicateLiteral::Binding {
                        target,
                        expression,
                        subst,
                    } => {
                        tasks.push(ExpansionTask::FinishPredicateBinding { target });
                        tasks.push(ExpansionTask::Expression {
                            expression,
                            subst,
                            in_conditional_branch: false,
                        });
                    }
                    PreparedPredicateLiteral::Literal(literal) => {
                        predicate_literal_values.push(vec![literal]);
                    }
                },
                ExpansionTask::FinishPredicateBinding { target } => {
                    let mut expanded = values
                        .pop()
                        .expect("predicate-body binding expression is expanded");
                    expanded
                        .generated_literals
                        .push(BodyLiteral::IsExpr(IsExpr {
                            target,
                            expr: expanded.expression,
                        }));
                    predicate_literal_values.push(expanded.generated_literals);
                }
                ExpansionTask::FinishPredicateBody {
                    literal_count,
                    result,
                } => {
                    let start = predicate_literal_values
                        .len()
                        .checked_sub(literal_count)
                        .expect("predicate literals produce expansion values");
                    let generated_literals = predicate_literal_values
                        .split_off(start)
                        .into_iter()
                        .flatten()
                        .collect();
                    let (generated_literals, result) =
                        inline_trailing_predicate_result_binding(generated_literals, result);
                    values.push(ExpandedExpression {
                        generated_literals,
                        expression: result,
                    });
                }
            }
        }

        debug_assert!(predicate_literal_values.is_empty());
        let expanded = values
            .pop()
            .expect("expression expansion produces one value");
        debug_assert!(values.is_empty());
        Ok(expanded)
    }

    fn schedule_binary(
        tasks: &mut Vec<ExpansionTask>,
        left: ArithExpr,
        right: ArithExpr,
        subst: HashMap<String, ArithExpr>,
        in_conditional_branch: bool,
        constructor: BinaryExpressionConstructor,
    ) {
        tasks.push(ExpansionTask::FinishBinary { constructor });
        tasks.push(ExpansionTask::Expression {
            expression: right,
            subst: subst.clone(),
            in_conditional_branch,
        });
        tasks.push(ExpansionTask::Expression {
            expression: left,
            subst,
            in_conditional_branch,
        });
    }

    #[cfg(test)]
    fn is_predicate_func(&self, name: &str) -> bool {
        self.registry
            .get(name)
            .is_some_and(|function| matches!(function.body, FuncBody::Predicate { .. }))
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
    // If no functions defined, return program unchanged
    if program.functions.is_empty() {
        return Ok(program.clone());
    }
    expand_program_functions_impl(program, max_depth)
}

/// [`expand_program_functions`] taking the program by value: without function
/// definitions (the common, fact-heavy case) the program is returned as is,
/// without a clone.
pub fn expand_program_functions_owned(
    program: Program,
    max_depth: u32,
) -> Result<Program, FunctionError> {
    if program.functions.is_empty() {
        return Ok(program);
    }
    expand_program_functions_impl(&program, max_depth)
}

fn expand_program_functions_impl(
    program: &Program,
    max_depth: u32,
) -> Result<Program, FunctionError> {
    let mut registry = FunctionRegistry::new();
    for function in &program.functions {
        registry.register(function.clone())?;
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
        authored_constraint_source_bound: program.authored_constraint_source_bound,
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
        authored_index: constraint.authored_index,
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
        let mut used = HashSet::from(["alice".to_string()]);
        let expanded = ctx
            .expand_expr_for_rule(
                &ArithExpr::FuncCall {
                    name: "get_parent".to_string(),
                    args,
                },
                &HashMap::new(),
                &mut used,
                false,
            )
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
        let mut used = HashSet::new();
        let expanded = ctx
            .expand_expr_for_rule(
                &ArithExpr::FuncCall {
                    name: "get_child".to_string(),
                    args,
                },
                &HashMap::new(),
                &mut used,
                false,
            )
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
        let mut used = HashSet::from(["alice".to_string()]);
        let expanded = ctx
            .expand_expr_for_rule(
                &ArithExpr::FuncCall {
                    name: "get_grandparent".to_string(),
                    args,
                },
                &HashMap::new(),
                &mut used,
                false,
            )
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

    #[test]
    fn generated_variable_names_round_trip_underscored_identifiers() {
        let generated = generated_function_variable_name("get_parent", "Parent_Value", 42);
        assert_eq!(
            generated_function_variable_source(&generated, "get_parent"),
            Some("Parent_Value")
        );
        assert_eq!(generated, "__XLOG_FUNCTION_GET_PARENT_Parent_Value_42");
        assert!(generated_function_variable_source(
            "__XLOG_FUNCTION_GET_PARENT_Parent_Value_not_a_counter",
            "get_parent"
        )
        .is_none());
    }
}
