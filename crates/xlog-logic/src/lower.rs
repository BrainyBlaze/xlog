//! Lowering from AST to IR
//!
//! This module transforms Datalog programs (AST) into the Relational IR (RIR)
//! representation for execution. The lowering process:
//!
//! 1. Infers schemas from facts and predicate declarations
//! 2. Tracks variable positions across atoms for join key computation
//! 3. Builds left-deep join trees for multi-atom rule bodies
//! 4. Handles negation via set difference (Diff) nodes
//! 5. Wraps recursive predicates in Fixpoint nodes
//! 6. Projects to match head variables

use std::collections::{HashMap, HashSet};

use xlog_core::{symbol, AggOp as CoreAggOp, RelId, Result, ScalarType, Schema, XlogError};
use xlog_ir::{
    CompareOp, CompiledRule, ConstValue, ExecutionPlan, Expr, JoinType, PlanBuilder, ProjectExpr,
    RirMeta, RirNode, Scc, Stratum as IrStratum,
};

use crate::ast::{
    AggOp, ArithExpr, Atom, BodyLiteral, CompOp, Comparison, IsExpr, LearnableRule, Program, Rule,
    Term, TypeRef,
};
use crate::stratify::{build_dependency_graph, find_sccs_for_lowering, DepType};

struct JoinPlan<'a> {
    node: RirNode,
    leaf_order: Vec<&'a Atom>,
    leaf_order_idx: Vec<usize>,
    var_pos: HashMap<String, usize>,
    width: usize,
    est_rows: f64,
    total_cost: f64,
}

#[derive(Clone, Copy)]
enum UserFunctionTypeEvidence {
    RequireExpansion,
    Defer,
}

#[derive(Clone, Copy)]
enum ArithmeticTypeOperation {
    Standard,
    Modulo,
    MinMax,
    Power,
}

enum ArithmeticTypeTask<'a> {
    Visit(&'a ArithExpr),
    FinishBinary(ArithmeticTypeOperation),
    FinishAbs,
    FinishCast(ScalarType),
    FinishFunctionArguments(usize),
    FinishConditional,
}

#[derive(Clone, Copy)]
enum ArithmeticExpressionOperation {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Min,
    Max,
    Pow,
}

enum ArithmeticExpressionTask<'a> {
    Visit(&'a ArithExpr),
    FinishBinary(ArithmeticExpressionOperation),
    FinishAbs,
    FinishCast(ScalarType),
    FinishConditional(CompOp),
}

fn resolve_pred_column_type(
    predicate: &str,
    index: usize,
    typ: &TypeRef,
    domains: &HashMap<String, ScalarType>,
) -> Result<ScalarType> {
    match typ {
        TypeRef::Scalar(ty) => Ok(*ty),
        TypeRef::Domain(name) => domains.get(name).copied().ok_or_else(|| {
            XlogError::Compilation(format!(
                "v0.8.5 unknown domain alias '{}' in predicate '{}' column {}",
                name, predicate, index
            ))
        }),
        TypeRef::List(_) | TypeRef::Term | TypeRef::Compound | TypeRef::PredRef => {
            Ok(ScalarType::U64)
        }
    }
}

fn validate_lowerable_terms(program: &Program) -> Result<()> {
    for rule in &program.rules {
        validate_atom_terms(&rule.head, "rule head")?;
        for lit in &rule.body {
            match lit {
                BodyLiteral::Positive(atom) => validate_atom_terms(atom, "positive body atom")?,
                BodyLiteral::Negated(atom) => validate_atom_terms(atom, "negated body atom")?,
                BodyLiteral::Epistemic(_) => {}
                BodyLiteral::Comparison(cmp) => {
                    validate_term_lowerable(&cmp.left, "comparison left operand")?;
                    validate_term_lowerable(&cmp.right, "comparison right operand")?;
                }
                BodyLiteral::IsExpr(_) => {}
                BodyLiteral::Univ(_) => {
                    return Err(XlogError::Compilation(
                        "v0.8.5 meta error: univ literal was not normalized before lowering"
                            .to_string(),
                    ));
                }
            }
        }
    }
    for constraint in &program.constraints {
        for lit in &constraint.body {
            match lit {
                BodyLiteral::Positive(atom) => validate_atom_terms(atom, "constraint body atom")?,
                BodyLiteral::Negated(atom) => {
                    validate_atom_terms(atom, "constraint negated body atom")?
                }
                BodyLiteral::Epistemic(_) => {}
                BodyLiteral::Comparison(cmp) => {
                    validate_term_lowerable(&cmp.left, "constraint comparison left operand")?;
                    validate_term_lowerable(&cmp.right, "constraint comparison right operand")?;
                }
                BodyLiteral::IsExpr(_) => {}
                BodyLiteral::Univ(_) => {
                    return Err(XlogError::Compilation(
                        "v0.8.5 meta error: univ literal was not normalized before lowering"
                            .to_string(),
                    ));
                }
            }
        }
    }
    for query in &program.queries {
        validate_atom_terms(&query.atom, "query atom")?;
    }
    for pf in &program.prob_facts {
        validate_atom_terms(&pf.atom, "probabilistic fact")?;
    }
    for ad in &program.annotated_disjunctions {
        for choice in &ad.choices {
            validate_atom_terms(&choice.atom, "annotated disjunction choice")?;
        }
    }
    for evidence in &program.evidence {
        validate_atom_terms(&evidence.atom, "evidence atom")?;
    }
    for query in &program.prob_queries {
        validate_atom_terms(&query.atom, "probabilistic query")?;
    }
    for neural in &program.neural_predicates {
        validate_atom_terms(&neural.predicate, "neural predicate")?;
    }
    for learnable in &program.learnable_rules {
        validate_atom_terms(&learnable.head, "learnable rule head")?;
        for lit in &learnable.body {
            if let BodyLiteral::Positive(atom) = lit {
                validate_atom_terms(atom, "learnable rule body")?;
            }
        }
    }
    Ok(())
}

fn validate_atom_terms(atom: &Atom, context: &str) -> Result<()> {
    for term in &atom.terms {
        validate_term_lowerable(term, context)?;
    }
    Ok(())
}

fn validate_term_lowerable(term: &Term, context: &str) -> Result<()> {
    match term {
        Term::List(_) => Err(term_not_lowerable_error(context, "list")),
        Term::Cons { .. } => Err(term_not_lowerable_error(context, "cons")),
        Term::Compound { .. } => Err(term_not_lowerable_error(context, "compound")),
        Term::PredRef(_) => Err(term_not_lowerable_error(context, "predref")),
        Term::Variable(_)
        | Term::Anonymous
        | Term::Integer(_)
        | Term::Float(_)
        | Term::String(_)
        | Term::Symbol(_)
        | Term::Aggregate(_) => Ok(()),
    }
}

fn term_not_lowerable_error(context: &str, kind: &str) -> XlogError {
    XlogError::Compilation(format!(
        "term form '{}' in {} is parsed but not lowerable by this execution path",
        kind, context
    ))
}

fn term_kind_for_lowering_error(term: &Term) -> &'static str {
    match term {
        Term::List(_) => "list",
        Term::Cons { .. } => "cons",
        Term::Compound { .. } => "compound",
        Term::PredRef(_) => "predref",
        Term::Variable(_)
        | Term::Anonymous
        | Term::Integer(_)
        | Term::Float(_)
        | Term::String(_)
        | Term::Symbol(_)
        | Term::Aggregate(_) => "term",
    }
}

/// Lowerer transforms AST programs into RIR execution plans.
pub struct Lowerer {
    /// Inferred or declared schemas for each predicate
    schemas: HashMap<String, Schema>,
    /// Stratification result (predicates grouped by strata)
    strata: Vec<Vec<String>>,
    /// Estimated cardinality per predicate (for join ordering)
    est_cardinality: HashMap<String, u64>,
    /// Optional cardinality hints per predicate (e.g., from runtime statistics).
    cardinality_hints: HashMap<String, u64>,
    /// Next available relation ID
    next_rel_id: u32,
    /// Mapping from predicate names to relation IDs
    rel_ids: HashMap<String, RelId>,
    /// SCCs for the program (from stratification)
    sccs: Vec<Scc>,
    /// Maximum active rules for TensorMaskedJoin (default 32)
    max_active_rules: usize,
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}

impl Lowerer {
    /// Create a new lowerer instance
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            strata: Vec::new(),
            est_cardinality: HashMap::new(),
            cardinality_hints: HashMap::new(),
            next_rel_id: 0,
            rel_ids: HashMap::new(),
            sccs: Vec::new(),
            max_active_rules: 32,
        }
    }

    /// Set the maximum active rules for TensorMaskedJoin.
    pub fn set_max_active_rules(&mut self, max: usize) {
        self.max_active_rules = max;
    }

    /// Set the stratification result for ordering
    pub(crate) fn set_strata(&mut self, strata: Vec<Vec<String>>) {
        self.strata = strata;
    }

    /// Set cardinality hints (typically sourced from runtime statistics snapshots).
    ///
    /// These hints are used by lowering-time join ordering when available.
    pub(crate) fn set_cardinality_hints(&mut self, hints: HashMap<String, u64>) {
        self.cardinality_hints = hints;
    }

    /// Get the mapping from predicate names to relation IDs
    pub fn rel_ids(&self) -> &HashMap<String, RelId> {
        &self.rel_ids
    }

    /// Get the inferred schemas for predicates
    pub fn schemas(&self) -> &HashMap<String, Schema> {
        &self.schemas
    }

    pub(crate) fn create_helper_relation(&mut self, schema: Schema) -> (String, RelId) {
        let name = format!("__kclique_helper_{}", self.next_rel_id);
        let rel_id = self.get_or_create_rel_id(&name);
        self.schemas.insert(name.clone(), schema);
        (name, rel_id)
    }

    /// Get or allocate a relation ID for a predicate
    fn get_or_create_rel_id(&mut self, name: &str) -> RelId {
        if let Some(&id) = self.rel_ids.get(name) {
            id
        } else {
            let id = RelId(self.next_rel_id);
            self.next_rel_id += 1;
            self.rel_ids.insert(name.to_string(), id);
            id
        }
    }

    /// Reject rules whose variables draw incompatible column types from
    /// the predicate schemas they touch. The executor requires exact
    /// schema equality when relations meet, so any conflict tolerated
    /// here would surface later as an internal kernel schema error
    /// instead of a source-level diagnostic.
    fn validate_rule_types(&self, program: &Program) -> Result<()> {
        let declared_predicates = program
            .predicates
            .iter()
            .map(|declaration| declaration.name.as_str())
            .collect::<HashSet<_>>();
        for rule in &program.rules {
            let var_types = self.infer_rule_variable_types(rule, |atom, index| {
                self.schemas
                    .get(&atom.predicate)
                    .and_then(|schema| schema.column_type(index))
            })?;
            let Some(head_schema) = self.schemas.get(&rule.head.predicate) else {
                continue;
            };
            for (j, term) in rule.head.terms.iter().enumerate() {
                let Some((_, head_ty)) = head_schema.columns.get(j) else {
                    continue;
                };
                match term {
                    Term::Variable(name) => {
                        let Some((body_ty, source)) = var_types.get(name) else {
                            continue;
                        };
                        if body_ty != head_ty {
                            return Err(XlogError::Compilation(format!(
                                "Type mismatch in rule for '{}': variable {} is {:?} \
                                 (from {}) but {} declares {:?} at position {}",
                                rule.head.predicate,
                                name,
                                body_ty,
                                source,
                                rule.head.predicate,
                                head_ty,
                                j
                            )));
                        }
                    }
                    Term::Anonymous => {}
                    Term::Aggregate(aggregate) => {
                        let aggregate_ty =
                            Self::infer_aggregate_result_type(rule, aggregate, &var_types)?
                                .ok_or_else(|| {
                                    XlogError::UnsafeVariable(aggregate.variable.clone())
                                })?;
                        if aggregate_ty != *head_ty {
                            return Err(XlogError::Compilation(format!(
                                "Type mismatch in rule for '{}': aggregate {:?} over {} produces \
                                 {:?}, but the predicate schema requires {:?} at position {}",
                                rule.head.predicate,
                                aggregate.op,
                                aggregate.variable,
                                aggregate_ty,
                                head_ty,
                                j
                            )));
                        }
                    }
                    Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                        if declared_predicates.contains(rule.head.predicate.as_str()) {
                            term_to_typed_const_value(term, *head_ty).map_err(|error| {
                                XlogError::Compilation(format!(
                                    "Type mismatch in rule for '{}': head term at position {} is \
                                     not compatible with {:?}: {}",
                                    rule.head.predicate, j, head_ty, error
                                ))
                            })?;
                        } else if term.inferred_scalar_type() != *head_ty {
                            return Err(XlogError::Compilation(format!(
                                "Type mismatch in rule for '{}': undeclared head term at position \
                                 {} has inferred type {:?}, but another clause requires {:?}",
                                rule.head.predicate,
                                j,
                                term.inferred_scalar_type(),
                                head_ty
                            )));
                        }
                    }
                    _ if term.inferred_scalar_type() != *head_ty => {
                        return Err(XlogError::Compilation(format!(
                            "Type mismatch in rule for '{}': head term at position {} has type \
                             {:?}, but the predicate schema requires {:?}",
                            rule.head.predicate,
                            j,
                            term.inferred_scalar_type(),
                            head_ty
                        )));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn infer_aggregate_result_type(
        rule: &Rule,
        aggregate: &crate::ast::AggExpr,
        variable_types: &HashMap<String, (ScalarType, String)>,
    ) -> Result<Option<ScalarType>> {
        let Some((input_type, source)) = variable_types.get(&aggregate.variable) else {
            return Ok(aggregate.input_independent_result_type());
        };
        aggregate
            .result_type_for_input(*input_type)
            .map(Some)
            .ok_or_else(|| {
                let required = match aggregate.op {
                    AggOp::Count => "any scalar input",
                    AggOp::Sum | AggOp::Min | AggOp::Max => "U32 or U64 input",
                    AggOp::LogSumExp => "F64 input",
                };
                XlogError::Compilation(format!(
                    "Unsupported aggregate input in rule for '{}': {:?}({}) receives {:?} from \
                     {}, but the execution provider requires {}",
                    rule.head.predicate,
                    aggregate.op,
                    aggregate.variable,
                    input_type,
                    source,
                    required
                ))
            })
    }

    fn record_rule_variable_type(
        rule: &Rule,
        variable_types: &mut HashMap<String, (ScalarType, String)>,
        variable: &str,
        typ: ScalarType,
        source: String,
    ) -> Result<()> {
        match variable_types.get(variable) {
            Some((existing, existing_source)) if *existing != typ => {
                Err(XlogError::Compilation(format!(
                    "Type mismatch in rule for '{}': variable {} is {:?} (from {}) but {:?} \
                     is required by {}",
                    rule.head.predicate, variable, existing, existing_source, typ, source
                )))
            }
            Some(_) => Ok(()),
            None => {
                variable_types.insert(variable.to_string(), (typ, source));
                Ok(())
            }
        }
    }

    /// Collect all type evidence available inside a rule.
    ///
    /// Ordinary body atoms are considered together because lowering joins them
    /// before evaluating arithmetic bindings. Arithmetic bindings are then
    /// processed in source order so chained `is` expressions can propagate their
    /// result types. Unknown inputs defer an arithmetic result until a later
    /// schema-inference iteration; known incompatible evidence is rejected here.
    fn infer_rule_variable_types_with_user_functions<F>(
        &self,
        rule: &Rule,
        mut column_type: F,
        user_functions: UserFunctionTypeEvidence,
    ) -> Result<HashMap<String, (ScalarType, String)>>
    where
        F: FnMut(&Atom, usize) -> Option<ScalarType>,
    {
        let mut variable_types = HashMap::new();

        for literal in &rule.body {
            let atom = match literal {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => atom,
                BodyLiteral::Epistemic(_)
                | BodyLiteral::Comparison(_)
                | BodyLiteral::IsExpr(_)
                | BodyLiteral::Univ(_) => continue,
            };
            for (index, term) in atom.terms.iter().enumerate() {
                let Term::Variable(variable) = term else {
                    continue;
                };
                let Some(typ) = column_type(atom, index) else {
                    continue;
                };
                Self::record_rule_variable_type(
                    rule,
                    &mut variable_types,
                    variable,
                    typ,
                    format!("{} position {}", atom.predicate, index),
                )?;
            }
        }

        for literal in &rule.body {
            let BodyLiteral::IsExpr(is_expr) = literal else {
                continue;
            };
            let result_type = Self::infer_arith_type_from_known_variables(
                &is_expr.expr,
                &|variable| variable_types.get(variable).map(|(typ, _)| *typ),
                user_functions,
            )?;
            if let Some(result_type) = result_type {
                Self::record_rule_variable_type(
                    rule,
                    &mut variable_types,
                    &is_expr.target,
                    result_type,
                    "an arithmetic binding".to_string(),
                )?;
            }
        }

        Ok(variable_types)
    }

    pub(crate) fn infer_rule_variable_types<F>(
        &self,
        rule: &Rule,
        column_type: F,
    ) -> Result<HashMap<String, (ScalarType, String)>>
    where
        F: FnMut(&Atom, usize) -> Option<ScalarType>,
    {
        self.infer_rule_variable_types_with_user_functions(
            rule,
            column_type,
            UserFunctionTypeEvidence::RequireExpansion,
        )
    }

    fn infer_rule_head_column_types_with_user_functions<F>(
        &self,
        rule: &Rule,
        column_type: F,
        user_functions: UserFunctionTypeEvidence,
    ) -> Result<Vec<Option<ScalarType>>>
    where
        F: FnMut(&Atom, usize) -> Option<ScalarType>,
    {
        let variable_types =
            self.infer_rule_variable_types_with_user_functions(rule, column_type, user_functions)?;
        rule.head
            .terms
            .iter()
            .map(|term| match term {
                Term::Variable(name) => Ok(variable_types.get(name).map(|(typ, _)| *typ)),
                Term::Aggregate(aggregate) => {
                    Self::infer_aggregate_result_type(rule, aggregate, &variable_types)
                }
                Term::Anonymous => Ok(None),
                _ => Ok(Some(term.inferred_scalar_type())),
            })
            .collect()
    }

    /// Infer each rule-head column from the same body, arithmetic, and aggregate
    /// evidence used by schema inference during lowering.
    ///
    /// A `None` column has no statically known evidence yet. This path requires
    /// user-defined function calls to have been expanded.
    pub(crate) fn infer_rule_head_column_types<F>(
        &self,
        rule: &Rule,
        column_type: F,
    ) -> Result<Vec<Option<ScalarType>>>
    where
        F: FnMut(&Atom, usize) -> Option<ScalarType>,
    {
        self.infer_rule_head_column_types_with_user_functions(
            rule,
            column_type,
            UserFunctionTypeEvidence::RequireExpansion,
        )
    }

    /// Infer rule-head columns before user-defined functions have been expanded.
    /// Function-call results remain unknown, while independent body, arithmetic,
    /// aggregate, and head-term evidence is still validated and propagated.
    pub(crate) fn infer_rule_head_column_types_before_function_expansion<F>(
        &self,
        rule: &Rule,
        column_type: F,
    ) -> Result<Vec<Option<ScalarType>>>
    where
        F: FnMut(&Atom, usize) -> Option<ScalarType>,
    {
        self.infer_rule_head_column_types_with_user_functions(
            rule,
            column_type,
            UserFunctionTypeEvidence::Defer,
        )
    }

    /// Infer schemas from facts and predicate declarations
    pub(crate) fn infer_schemas(&mut self, program: &Program) -> Result<()> {
        let domains: HashMap<String, ScalarType> = program
            .domains
            .iter()
            .map(|domain| (domain.name.clone(), domain.typ))
            .collect();

        // First, use explicit predicate declarations
        for pred_decl in &program.predicates {
            let declared_columns = pred_decl.schema_columns();
            let columns: Vec<(String, ScalarType)> = declared_columns
                .iter()
                .enumerate()
                .map(|(i, col)| {
                    let name = col.name.clone().unwrap_or_else(|| format!("c{}", i));
                    resolve_pred_column_type(&pred_decl.name, i, &col.typ, &domains)
                        .map(|ty| (name, ty))
                })
                .collect::<Result<Vec<_>>>()?;
            self.schemas
                .insert(pred_decl.name.clone(), Schema::new(columns));
        }

        // Then, infer from facts (if no declaration exists)
        for rule in program.facts() {
            let pred = &rule.head.predicate;
            if !self.schemas.contains_key(pred) {
                let columns: Vec<(String, ScalarType)> = rule
                    .head
                    .terms
                    .iter()
                    .enumerate()
                    .map(|(i, term)| {
                        let ty = term.inferred_scalar_type();
                        (format!("c{}", i), ty)
                    })
                    .collect();
                self.schemas.insert(pred.clone(), Schema::new(columns));
            }
        }

        // Probabilistic facts and annotated-disjunction choices are also
        // ground schema evidence. Register them before the body-only fallback
        // so a rule variable does not default an otherwise typed predicate to
        // U64 merely because its facts live in a probabilistic AST collection.
        for pf in &program.prob_facts {
            let pred = &pf.atom.predicate;
            if self.schemas.contains_key(pred) {
                continue;
            }
            let columns: Vec<(String, ScalarType)> = pf
                .atom
                .terms
                .iter()
                .enumerate()
                .map(|(i, term)| (format!("c{}", i), term.inferred_scalar_type()))
                .collect();
            self.schemas.insert(pred.clone(), Schema::new(columns));
        }

        for ad in &program.annotated_disjunctions {
            for choice in &ad.choices {
                let pred = &choice.atom.predicate;
                if self.schemas.contains_key(pred) {
                    continue;
                }
                let columns: Vec<(String, ScalarType)> = choice
                    .atom
                    .terms
                    .iter()
                    .enumerate()
                    .map(|(i, term)| (format!("c{}", i), term.inferred_scalar_type()))
                    .collect();
                self.schemas.insert(pred.clone(), Schema::new(columns));
            }
        }

        // Infer schemas for extensional predicates that occur only in rule bodies
        // before propagating rule-head types. This lets aggregates and ordinary
        // head variables consume the same body-column types that execution will
        // use, while leaving derived predicates to the fixed-point pass below.
        let derived_predicates = program
            .rules
            .iter()
            .filter(|rule| !rule.body.is_empty())
            .map(|rule| rule.head.predicate.as_str())
            .collect::<HashSet<_>>();
        for rule in &program.rules {
            for lit in &rule.body {
                let atom = match lit {
                    BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => atom,
                    BodyLiteral::Epistemic(_)
                    | BodyLiteral::Comparison(_)
                    | BodyLiteral::IsExpr(_)
                    | BodyLiteral::Univ(_) => continue,
                };
                let pred = &atom.predicate;
                if self.schemas.contains_key(pred) || derived_predicates.contains(pred.as_str()) {
                    continue;
                }
                let columns: Vec<(String, ScalarType)> = atom
                    .terms
                    .iter()
                    .enumerate()
                    .map(|(i, term)| (format!("c{}", i), term.inferred_scalar_type()))
                    .collect();
                let schema = Schema::new(columns)
                    .with_sort_labels(sort_labels_from_terms(&atom.terms))
                    .expect("body sort labels match inferred schema arity");
                self.schemas.insert(pred.clone(), schema);
            }
        }

        // Propagate body-derived rule-head types to a fixed point before
        // defaulting variables that have no type anchor. Columns converge
        // independently so an unresolved sibling cannot hide known evidence.
        let mut inferred_rule_columns: HashMap<String, Vec<Option<ScalarType>>> = HashMap::new();
        for rule in &program.rules {
            if self.schemas.contains_key(&rule.head.predicate) {
                continue;
            }
            inferred_rule_columns
                .entry(rule.head.predicate.clone())
                .or_insert_with(|| vec![None; rule.head.terms.len()]);
        }

        loop {
            let mut changed = false;
            for rule in &program.rules {
                let pred = &rule.head.predicate;
                if self.schemas.contains_key(pred) {
                    continue;
                }

                let Some(current_columns) = inferred_rule_columns.get(pred) else {
                    continue;
                };
                if current_columns.len() != rule.head.terms.len() {
                    continue;
                }

                let resolved_columns = self.infer_rule_head_column_types(rule, |atom, index| {
                    self.schemas
                        .get(&atom.predicate)
                        .and_then(|schema| schema.column_type(index))
                        .or_else(|| {
                            inferred_rule_columns
                                .get(&atom.predicate)
                                .and_then(|columns| columns.get(index))
                                .copied()
                                .flatten()
                        })
                })?;

                let columns = inferred_rule_columns
                    .get_mut(pred)
                    .expect("rule-head inference entry exists");
                for (index, (column, resolved)) in
                    columns.iter_mut().zip(resolved_columns).enumerate()
                {
                    match (*column, resolved) {
                        (None, Some(resolved)) => {
                            *column = Some(resolved);
                            changed = true;
                        }
                        (Some(existing), Some(resolved)) if existing != resolved => {
                            return Err(XlogError::Compilation(format!(
                                "Conflicting inferred schema for predicate '{}': column {} is \
                                 {:?} in one rule and {:?} in another",
                                pred,
                                index + 1,
                                existing,
                                resolved
                            )));
                        }
                        (None, None) | (Some(_), None) | (Some(_), Some(_)) => {}
                    }
                }
            }
            if !changed {
                break;
            }
        }
        for rule in &program.rules {
            let pred = &rule.head.predicate;
            if !self.schemas.contains_key(pred) {
                let inferred_columns = inferred_rule_columns
                    .get(pred)
                    .expect("undeclared rule head has an inference entry");
                let columns = rule
                    .head
                    .terms
                    .iter()
                    .zip(inferred_columns)
                    .enumerate()
                    .map(|(index, (term, inferred))| {
                        (
                            format!("c{index}"),
                            inferred.unwrap_or_else(|| term.inferred_scalar_type()),
                        )
                    })
                    .collect();
                let schema = Schema::new(columns)
                    .with_sort_labels(sort_labels_from_terms(&rule.head.terms))
                    .expect("rule head sort labels match inferred schema arity");
                self.schemas.insert(pred.clone(), schema);
            }
        }

        Ok(())
    }

    /// Infer predicate schemas and validate the type flow within every rule.
    ///
    /// This applies the schema/type contract shared by execution routes without
    /// preprocessing, stratifying, validating unrelated term forms, or building
    /// an execution plan. The inferred schemas remain available through
    /// [`Lowerer::schemas`].
    pub fn infer_and_validate_schemas(&mut self, program: &Program) -> Result<()> {
        self.infer_schemas(program)?;
        self.validate_rule_types(program)
    }

    fn infer_cardinalities(&mut self, program: &Program) {
        self.est_cardinality.clear();

        let mut fact_counts: HashMap<String, u64> = HashMap::new();
        for fact in program.facts() {
            *fact_counts.entry(fact.head.predicate.clone()).or_insert(0) += 1;
        }

        for pred in self.schemas.keys() {
            let est = self
                .cardinality_hints
                .get(pred)
                .copied()
                .or_else(|| fact_counts.get(pred).copied())
                .unwrap_or(1000)
                .max(1);
            self.est_cardinality.insert(pred.clone(), est);
        }
    }

    /// Build SCCs from the dependency graph
    fn build_sccs(&mut self, program: &Program) {
        let graph = build_dependency_graph(program);
        let scc_groups = find_sccs_for_lowering(&graph);

        self.sccs.clear();
        for (id, predicates) in scc_groups.iter().enumerate() {
            // An SCC is recursive if it has more than one predicate
            // or if a single predicate depends on itself positively
            let is_recursive = if predicates.len() > 1 {
                true
            } else {
                let pred = &predicates[0];
                graph
                    .outgoing(pred)
                    .iter()
                    .any(|e| e.to == *pred && e.dep_type == DepType::Positive)
            };

            self.sccs.push(Scc {
                id: id as u32,
                predicates: predicates.clone(),
                is_recursive,
            });
        }
    }

    fn prepare_program_for_lowering(&mut self, program: &Program) -> Result<()> {
        validate_lowerable_terms(program)?;
        self.infer_and_validate_schemas(program)?;
        self.infer_cardinalities(program);

        // Pre-allocate RelIds for declared predicates so schema-only programs
        // can populate relation stores before any facts or executable rules
        // mention those relations. This keeps ILP candidate generation and
        // runtime relation upload aligned with declared schemas.
        for pred_decl in &program.predicates {
            self.get_or_create_rel_id(&pred_decl.name);
        }
        // Facts are grouped and materialized directly into the relation store.
        // They still need stable relation IDs even when no declaration or rule
        // mentions their predicates.
        for fact in program.facts() {
            self.get_or_create_rel_id(&fact.head.predicate);
        }

        Ok(())
    }

    /// Validate every source-level contract enforced while lowering, without
    /// requiring a stratification or constructing an execution plan.
    ///
    /// Epistemic preparation uses this after replacing modal literals with their
    /// validation-only ordinary counterparts. It intentionally exercises the same
    /// schema inference, rule type checks, constant conversion, arithmetic ordering,
    /// negation lowering, and head projection as production lowering.
    pub(crate) fn validate_program_without_plan(&mut self, program: &Program) -> Result<()> {
        self.prepare_program_for_lowering(program)?;

        for rule in program.proper_rules() {
            self.lower_rule(rule)?;
        }

        // Match the relation allocation and validation performed by `lower_program`
        // for learnable rules as well. These rules cannot contain modal literals, but
        // they may share declarations and schemas with the authored program.
        for learnable in &program.learnable_rules {
            self.get_or_create_rel_id(&learnable.head.predicate);
            for lit in &learnable.body {
                if let BodyLiteral::Positive(atom) = lit {
                    self.get_or_create_rel_id(&atom.predicate);
                }
            }
        }
        for learnable in &program.learnable_rules {
            self.lower_learnable_rule(learnable)?;
        }

        Ok(())
    }

    /// Lower an entire program to an execution plan
    pub fn lower_program(&mut self, program: &Program) -> Result<ExecutionPlan> {
        self.prepare_program_for_lowering(program)?;

        // Build SCCs
        self.build_sccs(program);

        // Build execution plan
        let mut builder = PlanBuilder::new();

        // Add SCCs to the builder
        for scc in &self.sccs {
            builder.add_scc(scc.clone());
        }

        // Build strata from our strata field
        for (id, preds) in self.strata.iter().enumerate() {
            // Find which SCCs belong to this stratum
            let scc_ids: Vec<u32> = self
                .sccs
                .iter()
                .filter(|scc| scc.predicates.iter().any(|p| preds.contains(p)))
                .map(|scc| scc.id)
                .collect();

            if !scc_ids.is_empty() {
                builder.add_stratum(IrStratum {
                    id: id as u32,
                    sccs: scc_ids,
                });
            }
        }

        // Lower each rule
        let mut rules_by_pred: HashMap<String, Vec<&Rule>> = HashMap::new();
        for rule in program.proper_rules() {
            rules_by_pred
                .entry(rule.head.predicate.clone())
                .or_default()
                .push(rule);
        }

        // Lower proper rules
        for (pred, rules) in &rules_by_pred {
            let scc_id = self.find_scc_for_predicate(pred);

            for rule in rules {
                let body = self.lower_rule(rule)?;
                let meta = self.create_meta_for_predicate(pred);

                builder.add_rule(
                    scc_id,
                    CompiledRule {
                        head: pred.clone(),
                        body,
                        meta,
                    },
                );
            }
        }

        // Lower learnable rules into tensor-masked joins.
        // Pre-allocate RelIds for ALL learnable predicates (heads + bodies)
        // so every lower_learnable_rule snapshot is complete.
        for learnable in &program.learnable_rules {
            self.get_or_create_rel_id(&learnable.head.predicate);
            for lit in &learnable.body {
                if let BodyLiteral::Positive(atom) = lit {
                    self.get_or_create_rel_id(&atom.predicate);
                }
            }
        }
        for learnable in &program.learnable_rules {
            let head_pred = &learnable.head.predicate;
            let scc_id = self.find_scc_for_predicate(head_pred);
            let body = self.lower_learnable_rule(learnable)?;
            let meta = self.create_meta_for_predicate(head_pred);
            builder.add_rule(
                scc_id,
                CompiledRule {
                    head: head_pred.clone(),
                    body,
                    meta,
                },
            );
        }

        let mut plan = builder.build();
        // Record relation arities for downstream generic multiway shape
        // promoters that size Scan leaves from these values.
        // One pre-pass over the AST covers every predicate the lowerer
        // assigned a RelId: rule heads, positive/negated body atoms,
        // and facts.
        for rule in program.proper_rules() {
            if let Some(&id) = self.rel_ids.get(&rule.head.predicate) {
                plan.rel_arities.insert(id, rule.head.terms.len());
            }
            for lit in &rule.body {
                let atom = match lit {
                    BodyLiteral::Positive(a) | BodyLiteral::Negated(a) => a,
                    _ => continue,
                };
                if let Some(&id) = self.rel_ids.get(&atom.predicate) {
                    plan.rel_arities.insert(id, atom.terms.len());
                }
            }
        }
        for fact in program.facts() {
            if let Some(&id) = self.rel_ids.get(&fact.head.predicate) {
                plan.rel_arities.insert(id, fact.head.terms.len());
            }
        }
        Ok(plan)
    }

    /// Find the SCC ID for a predicate
    fn find_scc_for_predicate(&self, pred: &str) -> u32 {
        self.sccs
            .iter()
            .find(|scc| scc.predicates.contains(&pred.to_string()))
            .map(|scc| scc.id)
            .unwrap_or(0)
    }

    /// Create metadata for a predicate
    fn create_meta_for_predicate(&self, pred: &str) -> RirMeta {
        let schema = self
            .schemas
            .get(pred)
            .cloned()
            .unwrap_or_else(|| Schema::new(vec![]));
        RirMeta::with_schema(schema)
    }

    /// Lower a learnable rule template into a TensorMaskedJoin node.
    /// Validates that the body has exactly two positive atoms.
    /// Sorts rel_index by RelId for deterministic tensor dimension mapping.
    /// Uses get_or_create_rel_id for heads so head-only predicates are handled.
    fn lower_learnable_rule(&mut self, rule: &LearnableRule) -> Result<RirNode> {
        // Validate body shape before indexing fixed body positions.
        if rule.body.len() != 2 {
            return Err(XlogError::Compilation(format!(
                "learnable rule '{}' requires exactly 2 body literals, got {}",
                rule.mask_name,
                rule.body.len()
            )));
        }
        for (idx, lit) in rule.body.iter().enumerate() {
            match lit {
                BodyLiteral::Positive(_) => {}
                _ => {
                    return Err(XlogError::Compilation(format!(
                        "learnable rule '{}' body[{}]: only positive atoms allowed",
                        rule.mask_name, idx
                    )));
                }
            }
        }

        // Sort by RelId for deterministic tensor dimension mapping.
        let mut rel_index: Vec<(RelId, String)> = self
            .rel_ids()
            .iter()
            .map(|(name, id)| (*id, name.clone()))
            .collect();
        rel_index.sort_by_key(|(id, _)| id.0);
        let schema_size = rel_index.len();

        let (left_keys, right_keys) =
            self.extract_template_join_keys(&rule.body[0], &rule.body[1])?;

        let head_rel_name = rule.head.predicate.clone();
        // Allocate lazily because head-only predicates may not have a RelId yet.
        let head_rel_id = self.get_or_create_rel_id(&head_rel_name);

        // Compute head projection: map head variables to join result columns.
        // Join result layout: [left_col_0..left_col_n, right_col_0..right_col_m].
        let left_atom = rule.body[0].atom().unwrap();
        let right_atom = rule.body[1].atom().unwrap();
        let left_arity = left_atom.terms.len();

        // Build variable -> first-occurrence column mapping over joined result
        let mut var_to_col: HashMap<String, usize> = HashMap::new();
        for (i, term) in left_atom.terms.iter().enumerate() {
            if let Some(name) = term.variable_name() {
                var_to_col.entry(name.to_string()).or_insert(i);
            }
        }
        for (i, term) in right_atom.terms.iter().enumerate() {
            if let Some(name) = term.variable_name() {
                var_to_col.entry(name.to_string()).or_insert(left_arity + i);
            }
        }

        let mut head_projection: Vec<usize> = Vec::new();
        for term in &rule.head.terms {
            if let Some(name) = term.variable_name() {
                let col = var_to_col.get(name).ok_or_else(|| {
                    XlogError::Compilation(format!(
                        "Learnable rule head variable '{}' not found in body atoms \
                         ({}, {}). All head variables must appear in the body.",
                        name, left_atom.predicate, right_atom.predicate,
                    ))
                })?;
                head_projection.push(*col);
            } else {
                return Err(XlogError::Compilation(format!(
                    "Learnable rule head must contain only variables, \
                     found constant {:?} in head of '{}'",
                    term, head_rel_name,
                )));
            }
        }

        // Infer schema for head predicate from the learnable rule if not already set.
        // The head's column types come from the projected join columns.
        if !self.schemas.contains_key(&head_rel_name) {
            let columns: Vec<(String, ScalarType)> = head_projection
                .iter()
                .enumerate()
                .map(|(i, &col)| {
                    // Determine the type from left or right atom's schema
                    let ty = if col < left_arity {
                        self.schemas
                            .get(&left_atom.predicate)
                            .and_then(|s| s.column_type(col))
                            .unwrap_or(ScalarType::U32)
                    } else {
                        self.schemas
                            .get(&right_atom.predicate)
                            .and_then(|s| s.column_type(col - left_arity))
                            .unwrap_or(ScalarType::U32)
                    };
                    (format!("c{}", i), ty)
                })
                .collect();
            self.schemas
                .insert(head_rel_name.clone(), Schema::new(columns));
        }

        Ok(RirNode::TensorMaskedJoin {
            mask_name: rule.mask_name.clone(),
            schema_size,
            left_keys,
            right_keys,
            rel_index,
            head_rel_name,
            head_rel_id,
            max_active_rules: self.max_active_rules,
            head_projection,
        })
    }

    /// Extract join keys from two body literals' shared variables.
    /// For `b1(X, Z), b2(Z, Y)`, the shared variable Z gives left_keys=[1], right_keys=[0].
    fn extract_template_join_keys(
        &self,
        left: &BodyLiteral,
        right: &BodyLiteral,
    ) -> Result<(Vec<usize>, Vec<usize>)> {
        let left_atom = left
            .atom()
            .ok_or_else(|| XlogError::Compilation("Learnable body[0] is not an atom".into()))?;
        let right_atom = right
            .atom()
            .ok_or_else(|| XlogError::Compilation("Learnable body[1] is not an atom".into()))?;

        let mut left_keys = Vec::new();
        let mut right_keys = Vec::new();

        for (li, lt) in left_atom.terms.iter().enumerate() {
            if let Some(lname) = lt.variable_name() {
                for (ri, rt) in right_atom.terms.iter().enumerate() {
                    if let Some(rname) = rt.variable_name() {
                        if lname == rname {
                            left_keys.push(li);
                            right_keys.push(ri);
                        }
                    }
                }
            }
        }

        Ok((left_keys, right_keys))
    }

    /// Lower a single rule to an RIR node
    fn lower_rule(&mut self, rule: &Rule) -> Result<RirNode> {
        if let Some(lit) = rule.body.iter().find_map(|lit| match lit {
            BodyLiteral::Epistemic(lit) => Some(lit),
            _ => None,
        }) {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "RIR lowering boundary".to_string(),
                context: format!("{:?} {}({})", lit.op, lit.atom.predicate, lit.atom.arity()),
            });
        }

        // Split body literals.
        let (positive_atoms, negated_atoms, comparisons, is_exprs) =
            Self::split_body_literals(&rule.body);

        // Allocate RelIds for all body predicates in source order so join planning
        // does not influence identifier assignment.
        for lit in &rule.body {
            match lit {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                    self.get_or_create_rel_id(&atom.predicate);
                }
                BodyLiteral::Epistemic(_)
                | BodyLiteral::Comparison(_)
                | BodyLiteral::IsExpr(_)
                | BodyLiteral::Univ(_) => {}
            }
        }

        // Plan positive atoms (join tree shape + leaf order).
        //
        // Rules with no positive atoms are legal for nullary/ground heads in our
        // probabilistic profiles (e.g. `q() :- not p().`). Lower them by seeding
        // the body with a unit relation ({()}) and applying filters/negations.
        let (positive_root, leaf_order) = if positive_atoms.is_empty() {
            (RirNode::Unit, Vec::new())
        } else {
            self.plan_positive_atoms(&positive_atoms)?
        };

        // Build variable environment from the planned leaf order (matches join output layout:
        // left subtree columns then right subtree columns).
        let mut var_env = VariableEnv::new();
        let mut current_col = 0;
        for atom in &leaf_order {
            let schema = self.schemas.get(&atom.predicate);
            for (i, term) in atom.terms.iter().enumerate() {
                if let Term::Variable(name) = term {
                    if name == "_" {
                        continue;
                    }
                    var_env.add_occurrence(name, atom.predicate.clone(), i, current_col + i);
                    // Also record the type for this variable (first occurrence wins)
                    if !var_env.types.contains_key(name) {
                        let typ = schema
                            .and_then(|s| s.column_type(i))
                            .unwrap_or(ScalarType::I64); // Default to I64 for arithmetic
                        var_env.types.insert(name.to_string(), typ);
                    }
                }
            }
            current_col += atom.terms.len();
        }
        var_env.total_cols = current_col;

        // Lower the body starting from the planned positive join root.
        let body_node = self.lower_body_parts(
            positive_root,
            &negated_atoms,
            &comparisons,
            &is_exprs,
            &mut var_env,
        )?;

        if rule.has_aggregation() {
            return self.lower_aggregate_rule(&rule.head, body_node, &var_env);
        }

        // Project to head terms (variables and constants).
        let projection_exprs = self.compute_head_projection(&rule.head, &var_env)?;

        if Self::is_identity_projection(&projection_exprs, var_env.column_count()) {
            Ok(body_node)
        } else {
            Ok(RirNode::Project {
                input: Box::new(body_node),
                columns: projection_exprs,
            })
        }
    }

    fn split_body_literals(
        body: &[BodyLiteral],
    ) -> (Vec<&Atom>, Vec<&Atom>, Vec<&Comparison>, Vec<&IsExpr>) {
        let mut positive_atoms: Vec<&Atom> = Vec::new();
        let mut negated_atoms: Vec<&Atom> = Vec::new();
        let mut comparisons: Vec<&Comparison> = Vec::new();
        let mut is_exprs: Vec<&IsExpr> = Vec::new();

        for lit in body {
            match lit {
                BodyLiteral::Positive(atom) => positive_atoms.push(atom),
                BodyLiteral::Negated(atom) => negated_atoms.push(atom),
                BodyLiteral::Epistemic(_) => {}
                BodyLiteral::Comparison(cmp) => comparisons.push(cmp),
                BodyLiteral::IsExpr(is_expr) => is_exprs.push(is_expr),
                BodyLiteral::Univ(_) => {}
            }
        }

        (positive_atoms, negated_atoms, comparisons, is_exprs)
    }

    fn atom_vars(atom: &Atom) -> std::collections::HashSet<String> {
        atom.terms
            .iter()
            .flat_map(|t| t.variables().into_iter())
            .filter(|name| *name != "_")
            .map(ToOwned::to_owned)
            .collect()
    }

    fn estimate_atom_rows(&self, atom: &Atom) -> f64 {
        let base = self
            .est_cardinality
            .get(&atom.predicate)
            .copied()
            .unwrap_or(1000)
            .max(1) as f64;

        let const_count = atom
            .terms
            .iter()
            .filter(|t| term_to_const_value(t).is_some())
            .count();

        // Equality constants are usually selective; use a conservative default.
        let selectivity = 0.1_f64.powi(const_count as i32);
        (base * selectivity).max(1.0)
    }

    fn build_cartesian_join(
        &self,
        left: RirNode,
        right: RirNode,
        left_width: usize,
        right_width: usize,
    ) -> RirNode {
        // Implement cross join by appending a constant key column to both inputs and joining on it,
        // then projecting away the constant columns.
        let left_const_col =
            ProjectExpr::Computed(Expr::Const(ConstValue::U32(0)), ScalarType::U32);
        let right_const_col =
            ProjectExpr::Computed(Expr::Const(ConstValue::U32(0)), ScalarType::U32);

        let mut left_cols: Vec<ProjectExpr> = (0..left_width).map(ProjectExpr::Column).collect();
        left_cols.push(left_const_col);
        let left_aug = RirNode::Project {
            input: Box::new(left),
            columns: left_cols,
        };

        let mut right_cols: Vec<ProjectExpr> = (0..right_width).map(ProjectExpr::Column).collect();
        right_cols.push(right_const_col);
        let right_aug = RirNode::Project {
            input: Box::new(right),
            columns: right_cols,
        };

        let joined = RirNode::Join {
            left: Box::new(left_aug),
            right: Box::new(right_aug),
            left_keys: vec![left_width],
            right_keys: vec![right_width],
            join_type: JoinType::Inner,
        };

        let mut keep: Vec<ProjectExpr> = Vec::with_capacity(left_width + right_width);
        keep.extend((0..left_width).map(ProjectExpr::Column));
        let right_start = left_width + 1;
        keep.extend((right_start..right_start + right_width).map(ProjectExpr::Column));

        RirNode::Project {
            input: Box::new(joined),
            columns: keep,
        }
    }

    fn make_leaf_plan<'a>(&mut self, atom: &'a Atom, orig_idx: usize) -> Result<JoinPlan<'a>> {
        let rel_id = self.get_or_create_rel_id(&atom.predicate);
        let scan = RirNode::Scan { rel: rel_id };
        let node = self.apply_constant_filters(scan, atom, 0)?;

        let mut var_pos: HashMap<String, usize> = HashMap::new();
        for (i, term) in atom.terms.iter().enumerate() {
            if let Term::Variable(name) = term {
                if name != "_" {
                    var_pos.entry(name.clone()).or_insert(i);
                }
            }
        }

        let est_rows = self.estimate_atom_rows(atom);
        Ok(JoinPlan {
            node,
            leaf_order: vec![atom],
            leaf_order_idx: vec![orig_idx],
            var_pos,
            width: atom.terms.len(),
            est_rows,
            total_cost: est_rows,
        })
    }

    fn join_plans<'a>(&self, left: &JoinPlan<'a>, right: &JoinPlan<'a>) -> JoinPlan<'a> {
        let shared_vars: Vec<&String> = left
            .var_pos
            .keys()
            .filter(|v| right.var_pos.contains_key(*v))
            .collect();

        let node = if shared_vars.is_empty() {
            self.build_cartesian_join(
                left.node.clone(),
                right.node.clone(),
                left.width,
                right.width,
            )
        } else {
            let mut key_pairs: Vec<(usize, usize)> = shared_vars
                .iter()
                .filter_map(|v| {
                    Some((
                        left.var_pos.get(*v).copied()?,
                        right.var_pos.get(*v).copied()?,
                    ))
                })
                .collect();
            key_pairs.sort_unstable();

            let (left_keys, right_keys): (Vec<usize>, Vec<usize>) = key_pairs.into_iter().unzip();

            RirNode::Join {
                left: Box::new(left.node.clone()),
                right: Box::new(right.node.clone()),
                left_keys,
                right_keys,
                join_type: JoinType::Inner,
            }
        };

        let mut leaf_order = left.leaf_order.clone();
        leaf_order.extend(right.leaf_order.iter().copied());

        let mut leaf_order_idx = left.leaf_order_idx.clone();
        leaf_order_idx.extend_from_slice(&right.leaf_order_idx);

        let mut var_pos = left.var_pos.clone();
        for (var, pos) in &right.var_pos {
            var_pos.entry(var.clone()).or_insert(left.width + *pos);
        }

        let shared = shared_vars.len();
        let mut selectivity = if shared == 0 {
            1.0
        } else {
            0.1_f64.powi(shared as i32)
        };
        if shared == 0 {
            // Penalize cartesian joins strongly.
            selectivity *= 1.0e6;
        }

        let output_rows = (left.est_rows * right.est_rows * selectivity).max(1.0);

        // Hash join cost is sensitive to which side is build (right) and probe (left).
        let build_cost = right.est_rows;
        let probe_cost = left.est_rows * 0.5;
        let total_cost = left.total_cost + right.total_cost + build_cost + probe_cost + output_rows;

        JoinPlan {
            node,
            leaf_order,
            leaf_order_idx,
            var_pos,
            width: left.width + right.width,
            est_rows: output_rows,
            total_cost,
        }
    }

    fn plan_positive_atoms_bushy<'a>(
        &mut self,
        atoms: &[&'a Atom],
    ) -> Result<(RirNode, Vec<&'a Atom>)> {
        let n = atoms.len();
        if n == 0 {
            return Err(XlogError::Compilation("Empty rule body".to_string()));
        }
        if n == 1 {
            let plan = self.make_leaf_plan(atoms[0], 0)?;
            return Ok((plan.node, plan.leaf_order));
        }

        let size = 1usize << n;
        let mut best: Vec<Option<JoinPlan<'a>>> = (0..size).map(|_| None).collect();

        for (i, atom) in atoms.iter().enumerate() {
            best[1usize << i] = Some(self.make_leaf_plan(atom, i)?);
        }

        fn lex_lt(a: &[usize], b: &[usize]) -> bool {
            for (ai, bi) in a.iter().zip(b.iter()) {
                if ai != bi {
                    return ai < bi;
                }
            }
            a.len() < b.len()
        }

        for mask in 1..size {
            if mask.count_ones() <= 1 {
                continue;
            }

            let mut best_for_mask: Option<JoinPlan<'a>> = None;

            let mut sub = (mask - 1) & mask;
            while sub > 0 {
                let a = sub;
                let b = mask ^ a;
                if b == 0 {
                    sub = (sub - 1) & mask;
                    continue;
                }

                let (Some(plan_a), Some(plan_b)) = (&best[a], &best[b]) else {
                    sub = (sub - 1) & mask;
                    continue;
                };

                // Consider both orientations: A ⋈ B and B ⋈ A.
                for (left, right) in [(plan_a, plan_b), (plan_b, plan_a)] {
                    let cand = self.join_plans(left, right);
                    let replace = match &best_for_mask {
                        None => true,
                        Some(current) => {
                            if cand.total_cost < current.total_cost {
                                true
                            } else if (cand.total_cost - current.total_cost).abs() < 1e-9 {
                                lex_lt(&cand.leaf_order_idx, &current.leaf_order_idx)
                            } else {
                                false
                            }
                        }
                    };

                    if replace {
                        best_for_mask = Some(cand);
                    }
                }

                sub = (sub - 1) & mask;
            }

            best[mask] = best_for_mask;
        }

        let full_mask = size - 1;
        if let Some(plan) = best[full_mask].take() {
            return Ok((plan.node, plan.leaf_order));
        }

        // Should be unreachable, but fall back to greedy ordering.
        let ordered = self.order_positive_atoms_greedy(atoms);
        let mut dummy_env = VariableEnv::new();
        let node = self.build_join_tree(&ordered, &mut dummy_env)?;
        Ok((node, ordered))
    }

    fn plan_positive_atoms<'a>(&mut self, atoms: &[&'a Atom]) -> Result<(RirNode, Vec<&'a Atom>)> {
        if atoms.len() <= 1 {
            if atoms.is_empty() {
                return Err(XlogError::Compilation("Empty rule body".to_string()));
            }
            let plan = self.make_leaf_plan(atoms[0], 0)?;
            return Ok((plan.node, plan.leaf_order));
        }

        const MAX_BUSHY_DP_ATOMS: usize = 10;
        if atoms.len() <= MAX_BUSHY_DP_ATOMS {
            return self.plan_positive_atoms_bushy(atoms);
        }

        // Greedy bushy join planning for large rules (scales beyond exponential DP).
        self.plan_positive_atoms_bushy_greedy(atoms)
    }

    fn plan_positive_atoms_bushy_greedy<'a>(
        &mut self,
        atoms: &[&'a Atom],
    ) -> Result<(RirNode, Vec<&'a Atom>)> {
        if atoms.is_empty() {
            return Err(XlogError::Compilation("Empty rule body".to_string()));
        }

        fn lex_lt(a: &[usize], b: &[usize]) -> bool {
            for (ai, bi) in a.iter().zip(b.iter()) {
                if ai != bi {
                    return ai < bi;
                }
            }
            a.len() < b.len()
        }

        let mut plans: Vec<JoinPlan<'a>> = Vec::with_capacity(atoms.len());
        for (idx, atom) in atoms.iter().enumerate() {
            plans.push(self.make_leaf_plan(atom, idx)?);
        }

        while plans.len() > 1 {
            let mut best_pair: Option<(usize, usize, JoinPlan<'a>)> = None;

            for i in 0..plans.len() {
                for j in (i + 1)..plans.len() {
                    let a = &plans[i];
                    let b = &plans[j];

                    let cand_ab = self.join_plans(a, b);
                    let cand_ba = self.join_plans(b, a);

                    let cand = if cand_ab.total_cost < cand_ba.total_cost
                        || (cand_ab.total_cost - cand_ba.total_cost).abs() < 1e-9
                            && lex_lt(&cand_ab.leaf_order_idx, &cand_ba.leaf_order_idx)
                    {
                        cand_ab
                    } else {
                        cand_ba
                    };

                    let replace = match &best_pair {
                        None => true,
                        Some((_bi, _bj, best)) => {
                            if cand.total_cost < best.total_cost {
                                true
                            } else if (cand.total_cost - best.total_cost).abs() < 1e-9 {
                                lex_lt(&cand.leaf_order_idx, &best.leaf_order_idx)
                            } else {
                                false
                            }
                        }
                    };

                    if replace {
                        best_pair = Some((i, j, cand));
                    }
                }
            }

            let Some((i, j, joined)) = best_pair else {
                break;
            };

            // Remove joined inputs from the plan list and replace with the join.
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            plans.remove(b);
            plans.remove(a);
            plans.push(joined);
        }

        let plan = plans
            .pop()
            .ok_or_else(|| XlogError::Compilation("Join planning failed".to_string()))?;
        Ok((plan.node, plan.leaf_order))
    }

    fn order_positive_atoms_greedy<'a>(&self, atoms: &[&'a Atom]) -> Vec<&'a Atom> {
        let mut remaining: Vec<(usize, &Atom)> = atoms.iter().copied().enumerate().collect();
        let mut ordered: Vec<&Atom> = Vec::with_capacity(atoms.len());
        let mut bound_vars: HashSet<String> = HashSet::new();

        while !remaining.is_empty() {
            let pick_idx = if ordered.is_empty() {
                remaining
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let (ai, aa) = **a;
                        let (bi, bb) = **b;
                        self.estimate_atom_rows(aa)
                            .partial_cmp(&self.estimate_atom_rows(bb))
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(ai.cmp(&bi))
                    })
                    .map(|(idx, _)| idx)
                    .unwrap()
            } else {
                remaining
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let (ai, aa) = **a;
                        let (bi, bb) = **b;

                        let a_vars = Self::atom_vars(aa);
                        let b_vars = Self::atom_vars(bb);

                        let a_shared = a_vars.intersection(&bound_vars).count();
                        let b_shared = b_vars.intersection(&bound_vars).count();

                        let a_score = if a_shared == 0 {
                            self.estimate_atom_rows(aa) * 1.0e12
                        } else {
                            self.estimate_atom_rows(aa) / a_shared as f64
                        };
                        let b_score = if b_shared == 0 {
                            self.estimate_atom_rows(bb) * 1.0e12
                        } else {
                            self.estimate_atom_rows(bb) / b_shared as f64
                        };

                        a_score
                            .partial_cmp(&b_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(ai.cmp(&bi))
                    })
                    .map(|(idx, _)| idx)
                    .unwrap()
            };

            let (_orig_idx, atom) = remaining.remove(pick_idx);
            ordered.push(atom);
            bound_vars.extend(Self::atom_vars(atom));
        }

        ordered
    }

    fn lower_body_parts(
        &mut self,
        positive_root: RirNode,
        negated_atoms: &[&Atom],
        comparisons: &[&Comparison],
        is_exprs: &[&IsExpr],
        var_env: &mut VariableEnv,
    ) -> Result<RirNode> {
        let mut result = positive_root;

        // Apply comparisons as filters.
        for cmp in comparisons {
            result = self.apply_comparison(result, cmp, var_env)?;
        }

        // Apply is-expressions (must be after atoms that bind the input variables).
        for is_expr in is_exprs {
            result = self.lower_is_expr(is_expr, result, var_env)?;
        }

        // Handle negated atoms via Diff / semi-join.
        for neg_atom in negated_atoms {
            result = self.apply_negation(result, neg_atom, var_env)?;
        }

        Ok(result)
    }

    /// Build a left-deep join tree from positive atoms
    fn build_join_tree(&mut self, atoms: &[&Atom], var_env: &mut VariableEnv) -> Result<RirNode> {
        if atoms.is_empty() {
            return Err(XlogError::Compilation("Empty rule body".to_string()));
        }

        // Start with the first atom as a scan
        let first_atom = atoms[0];
        let rel_id = self.get_or_create_rel_id(&first_atom.predicate);
        let mut result = RirNode::Scan { rel: rel_id };
        let mut result_vars = self.collect_atom_vars(first_atom);
        let mut result_width = first_atom.terms.len();

        // Apply constant filters if any
        result = self.apply_constant_filters(result, first_atom, 0)?;

        // Join with remaining atoms (left-deep)
        for atom in atoms.iter().skip(1) {
            let right_rel_id = self.get_or_create_rel_id(&atom.predicate);
            let right_scan = RirNode::Scan { rel: right_rel_id };

            // Apply constant filters to the right side
            let right_filtered = self.apply_constant_filters(right_scan, atom, 0)?;

            // Compute join keys based on shared variables
            let (left_keys, right_keys) = self.compute_join_keys(&result_vars, atom, result_width);

            if left_keys.is_empty() {
                // Cartesian product (no shared variables)
                result = RirNode::Join {
                    left: Box::new(result),
                    right: Box::new(right_filtered),
                    left_keys: vec![],
                    right_keys: vec![],
                    join_type: JoinType::Inner,
                };
            } else {
                result = RirNode::Join {
                    left: Box::new(result),
                    right: Box::new(right_filtered),
                    left_keys,
                    right_keys,
                    join_type: JoinType::Inner,
                };
            }

            // Update result vars for the next iteration
            for (i, term) in atom.terms.iter().enumerate() {
                if let Term::Variable(name) = term {
                    result_vars.push((name.clone(), result_width + i));
                }
            }
            result_width += atom.terms.len();
        }

        // Update var_env with final positions
        var_env.total_cols = result_width;

        Ok(result)
    }

    /// Collect variable names and their positions within an atom
    fn collect_atom_vars(&self, atom: &Atom) -> Vec<(String, usize)> {
        atom.terms
            .iter()
            .enumerate()
            .filter_map(|(i, term)| {
                if let Term::Variable(name) = term {
                    Some((name.clone(), i))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute join keys between the current result and a new atom
    fn compute_join_keys(
        &self,
        left_vars: &[(String, usize)],
        right_atom: &Atom,
        _left_width: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_keys = Vec::new();
        let mut right_keys = Vec::new();

        for (right_idx, term) in right_atom.terms.iter().enumerate() {
            if let Term::Variable(name) = term {
                // Find if this variable exists in the left side
                for (left_name, left_idx) in left_vars {
                    if left_name == name {
                        left_keys.push(*left_idx);
                        right_keys.push(right_idx);
                        break; // Only use first occurrence for join key
                    }
                }
            }
        }

        (left_keys, right_keys)
    }

    /// Apply constant filters for an atom
    fn apply_constant_filters(
        &self,
        input: RirNode,
        atom: &Atom,
        _base_col: usize,
    ) -> Result<RirNode> {
        let mut filters = Vec::new();
        let mut first_var_col: HashMap<&str, usize> = HashMap::new();
        let schema = self.schemas.get(&atom.predicate).ok_or_else(|| {
            XlogError::Compilation(format!("Missing schema for predicate {}", atom.predicate))
        })?;

        for (i, term) in atom.terms.iter().enumerate() {
            if let Term::Variable(name) = term {
                if name != "_" {
                    if let Some(&first) = first_var_col.get(name.as_str()) {
                        filters.push(Expr::Compare {
                            left: Box::new(Expr::Column(first)),
                            op: CompareOp::Eq,
                            right: Box::new(Expr::Column(i)),
                        });
                    } else {
                        first_var_col.insert(name.as_str(), i);
                    }
                }
            }

            let col_type = schema.column_type(i).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Missing column type for {} column {}",
                    atom.predicate, i
                ))
            })?;
            if let Some(const_val) = term_to_typed_const_value(term, col_type)? {
                filters.push(Expr::Compare {
                    left: Box::new(Expr::Column(i)),
                    op: CompareOp::Eq,
                    right: Box::new(Expr::Const(const_val)),
                });
            }
        }

        if filters.is_empty() {
            Ok(input)
        } else {
            let predicate = if filters.len() == 1 {
                filters.pop().unwrap()
            } else {
                Expr::And(filters)
            };

            Ok(RirNode::Filter {
                input: Box::new(input),
                predicate,
            })
        }
    }

    /// Apply a comparison as a filter
    fn apply_comparison(
        &self,
        input: RirNode,
        cmp: &Comparison,
        var_env: &VariableEnv,
    ) -> Result<RirNode> {
        let (left_expr, right_expr) = match (&cmp.left, &cmp.right) {
            (Term::Variable(name), term) => {
                let col = var_env.get_column(name).ok_or_else(|| {
                    XlogError::Compilation(format!("Variable {} not found in environment", name))
                })?;
                let typ = var_env.get_type(name).ok_or_else(|| {
                    XlogError::Compilation(format!("Missing type for variable {}", name))
                })?;
                if let Some(const_val) = term_to_typed_const_value(term, typ)? {
                    (Expr::Column(col), Expr::Const(const_val))
                } else {
                    (
                        self.term_to_expr(&cmp.left, var_env)?,
                        self.term_to_expr(&cmp.right, var_env)?,
                    )
                }
            }
            (term, Term::Variable(name)) => {
                let col = var_env.get_column(name).ok_or_else(|| {
                    XlogError::Compilation(format!("Variable {} not found in environment", name))
                })?;
                let typ = var_env.get_type(name).ok_or_else(|| {
                    XlogError::Compilation(format!("Missing type for variable {}", name))
                })?;
                if let Some(const_val) = term_to_typed_const_value(term, typ)? {
                    (Expr::Const(const_val), Expr::Column(col))
                } else {
                    (
                        self.term_to_expr(&cmp.left, var_env)?,
                        self.term_to_expr(&cmp.right, var_env)?,
                    )
                }
            }
            _ => (
                self.term_to_expr(&cmp.left, var_env)?,
                self.term_to_expr(&cmp.right, var_env)?,
            ),
        };

        let op = match cmp.op {
            CompOp::Eq => CompareOp::Eq,
            CompOp::Ne => CompareOp::Ne,
            CompOp::Lt => CompareOp::Lt,
            CompOp::Le => CompareOp::Le,
            CompOp::Gt => CompareOp::Gt,
            CompOp::Ge => CompareOp::Ge,
        };

        Ok(RirNode::Filter {
            input: Box::new(input),
            predicate: Expr::Compare {
                left: Box::new(left_expr),
                op,
                right: Box::new(right_expr),
            },
        })
    }

    /// Convert a term to an expression
    fn term_to_expr(&self, term: &Term, var_env: &VariableEnv) -> Result<Expr> {
        match term {
            Term::Variable(name) => {
                if let Some(col) = var_env.get_column(name) {
                    Ok(Expr::Column(col))
                } else {
                    Err(XlogError::Compilation(format!(
                        "Variable {} not found in environment",
                        name
                    )))
                }
            }
            Term::Anonymous => Err(XlogError::Compilation(
                "Anonymous wildcard '_' not allowed in comparisons".to_string(),
            )),
            Term::Integer(i) => Ok(Expr::Const(ConstValue::I64(*i))),
            Term::Float(f) => Ok(Expr::Const(ConstValue::F64(*f))),
            Term::String(s) => Ok(Expr::Const(ConstValue::Symbol(s.clone()))),
            Term::Symbol(id) => Ok(Expr::Const(ConstValue::Symbol(symbol::resolve(*id)))),
            Term::Aggregate(_) => Err(XlogError::Compilation(
                "Aggregates not allowed in comparisons".to_string(),
            )),
            Term::List(_) | Term::Cons { .. } | Term::Compound { .. } | Term::PredRef(_) => Err(
                term_not_lowerable_error("comparison", term_kind_for_lowering_error(term)),
            ),
        }
    }

    /// Apply negation via set difference
    fn apply_negation(
        &mut self,
        input: RirNode,
        neg_atom: &Atom,
        var_env: &VariableEnv,
    ) -> Result<RirNode> {
        let rel_id = self.get_or_create_rel_id(&neg_atom.predicate);
        let neg_scan = RirNode::Scan { rel: rel_id };

        // Apply constant filters to the negated atom
        let neg_filtered = self.apply_constant_filters(neg_scan, neg_atom, 0)?;

        // Find which columns from the input correspond to variables in the negated atom
        let mut input_cols = Vec::new();
        let mut neg_cols = Vec::new();

        for (neg_idx, term) in neg_atom.terms.iter().enumerate() {
            if let Term::Variable(name) = term {
                if let Some(col) = var_env.get_column(name) {
                    input_cols.push(col);
                    neg_cols.push(neg_idx);
                }
            }
        }

        if input_cols.is_empty() {
            // A negated atom with no shared variables is a Boolean existence gate over
            // the entire positive input, not a tuple difference. Give both sides the
            // same synthetic key, anti-join on that key, then remove it. If the
            // negated atom has any matching row, every input row is rejected; if it is
            // empty, the anti-join returns the input unchanged. This also preserves
            // arbitrary input schemas, including the zero-arity unit used by
            // negation-only rules.
            let input_width = var_env.column_count();
            let join_key =
                || ProjectExpr::Computed(Expr::Const(ConstValue::U32(0)), ScalarType::U32);

            let mut keyed_input_columns: Vec<ProjectExpr> =
                (0..input_width).map(ProjectExpr::Column).collect();
            keyed_input_columns.push(join_key());
            let keyed_input = RirNode::Project {
                input: Box::new(input),
                columns: keyed_input_columns,
            };
            let keyed_negation = RirNode::Project {
                input: Box::new(neg_filtered),
                columns: vec![join_key()],
            };
            let gated_input = RirNode::Join {
                left: Box::new(keyed_input),
                right: Box::new(keyed_negation),
                left_keys: vec![input_width],
                right_keys: vec![0],
                join_type: JoinType::Anti,
            };

            Ok(RirNode::Project {
                input: Box::new(gated_input),
                columns: (0..input_width).map(ProjectExpr::Column).collect(),
            })
        } else {
            // Project the negated atom to only the shared variable columns
            let neg_projected = if neg_cols.len() < neg_atom.terms.len() {
                let neg_proj_exprs: Vec<ProjectExpr> =
                    neg_cols.iter().map(|&c| ProjectExpr::Column(c)).collect();
                RirNode::Project {
                    input: Box::new(neg_filtered),
                    columns: neg_proj_exprs,
                }
            } else {
                neg_filtered
            };

            // Project input to matching columns for the diff, then diff
            // Actually, for proper anti-join semantics we need to be careful.
            // The Diff operation subtracts matching tuples.
            // We need to project input to the shared columns, diff, then rejoin.

            // Simpler approach: project input to shared columns, diff with negated,
            // then rejoin with original
            let input_proj_exprs: Vec<ProjectExpr> =
                input_cols.iter().map(|&c| ProjectExpr::Column(c)).collect();
            let input_projected = RirNode::Project {
                input: Box::new(input.clone()),
                columns: input_proj_exprs,
            };

            // The Diff gives us the keys that should be kept
            let kept_keys = RirNode::Diff {
                left: Box::new(input_projected),
                right: Box::new(neg_projected),
            };

            // Join back with original input to get full tuples
            // This effectively filters the input to only rows where the key
            // is not in the negated relation
            Ok(RirNode::Join {
                left: Box::new(input),
                right: Box::new(kept_keys),
                left_keys: input_cols.clone(),
                right_keys: (0..input_cols.len()).collect(),
                join_type: JoinType::Semi,
            })
        }
    }

    fn is_identity_projection(proj: &[ProjectExpr], input_cols: usize) -> bool {
        if proj.len() != input_cols {
            return false;
        }
        proj.iter()
            .enumerate()
            .all(|(i, e)| matches!(e, ProjectExpr::Column(c) if *c == i))
    }

    /// Build a projection list that matches the rule head term order.
    ///
    /// For non-aggregate rules this supports:
    /// - Variables (column passthrough)
    /// - Constants (computed constant columns)
    fn compute_head_projection(
        &self,
        head: &Atom,
        var_env: &VariableEnv,
    ) -> Result<Vec<ProjectExpr>> {
        let mut cols = Vec::with_capacity(head.terms.len());

        for (index, term) in head.terms.iter().enumerate() {
            match term {
                Term::Variable(name) => {
                    let col = var_env
                        .get_column(name)
                        .ok_or_else(|| XlogError::UnsafeVariable(name.clone()))?;
                    cols.push(ProjectExpr::Column(col));
                }
                Term::Anonymous => {
                    return Err(XlogError::Compilation(
                        "Anonymous wildcard '_' not allowed in rule head".to_string(),
                    ));
                }
                Term::Aggregate(_) => {
                    return Err(XlogError::Compilation(
                        "Aggregate term in non-aggregate rule head".to_string(),
                    ));
                }
                Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                    let typ = self
                        .schemas
                        .get(&head.predicate)
                        .and_then(|schema| schema.column_type(index))
                        .ok_or_else(|| {
                            XlogError::Compilation(format!(
                                "Missing schema type for '{}' column {}",
                                head.predicate, index
                            ))
                        })?;
                    let value = term_to_typed_const_value(term, typ)?.ok_or_else(|| {
                        XlogError::Compilation("Expected constant term".to_string())
                    })?;
                    cols.push(ProjectExpr::Computed(Expr::Const(value), typ));
                }
                Term::List(_) | Term::Cons { .. } | Term::Compound { .. } | Term::PredRef(_) => {
                    return Err(term_not_lowerable_error(
                        "rule head projection",
                        term_kind_for_lowering_error(term),
                    ));
                }
            }
        }

        Ok(cols)
    }

    /// Lower an aggregate rule head into `GroupBy` + final projection.
    fn lower_aggregate_rule(
        &mut self,
        head: &Atom,
        body: RirNode,
        var_env: &VariableEnv,
    ) -> Result<RirNode> {
        // Collect unique group keys in head order.
        let mut key_vars: Vec<String> = Vec::new();
        let mut key_var_to_pos: HashMap<String, usize> = HashMap::new();
        let mut key_src_cols: Vec<usize> = Vec::new();

        // Collect unique aggregate specs (op, var) in head order.
        let mut agg_specs: Vec<(AggOp, String)> = Vec::new();
        let mut agg_to_pos: HashMap<(AggOp, String), usize> = HashMap::new();
        let mut value_vars: Vec<String> = Vec::new();
        let mut value_var_to_pos: HashMap<String, usize> = HashMap::new();
        let mut value_src_cols: Vec<usize> = Vec::new();

        for term in &head.terms {
            match term {
                Term::Variable(name) => {
                    if !key_var_to_pos.contains_key(name) {
                        let col = var_env
                            .get_column(name)
                            .ok_or_else(|| XlogError::UnsafeVariable(name.clone()))?;
                        let pos = key_vars.len();
                        key_vars.push(name.clone());
                        key_var_to_pos.insert(name.clone(), pos);
                        key_src_cols.push(col);
                    }
                }
                Term::Aggregate(agg) => {
                    let key = (agg.op, agg.variable.clone());
                    if let std::collections::hash_map::Entry::Vacant(entry) = agg_to_pos.entry(key)
                    {
                        // Ensure the aggregated variable is bound.
                        let col = var_env
                            .get_column(&agg.variable)
                            .ok_or_else(|| XlogError::UnsafeVariable(agg.variable.clone()))?;

                        // Ensure the value variable exists in the groupby input.
                        let value_pos = *value_var_to_pos
                            .entry(agg.variable.clone())
                            .or_insert_with(|| {
                                let p = value_vars.len();
                                value_vars.push(agg.variable.clone());
                                value_src_cols.push(col);
                                p
                            });

                        let agg_pos = agg_specs.len();
                        agg_specs.push((agg.op, agg.variable.clone()));
                        entry.insert(agg_pos);

                        // Keep clippy happy about unused value_pos in insert_with closure.
                        let _ = value_pos;
                    }
                }
                Term::Anonymous => {
                    return Err(XlogError::Compilation(
                        "Anonymous wildcard '_' not allowed in rule head".to_string(),
                    ));
                }
                Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                    // Constants are allowed in the head; they are projected after aggregation.
                }
                Term::List(_) | Term::Cons { .. } | Term::Compound { .. } | Term::PredRef(_) => {
                    return Err(term_not_lowerable_error(
                        "aggregate rule head",
                        term_kind_for_lowering_error(term),
                    ));
                }
            }
        }

        if agg_specs.is_empty() {
            return Err(XlogError::Compilation(
                "Rule marked as aggregate but no aggregate terms found".to_string(),
            ));
        }

        // Build groupby input: [keys..., values...]. For global aggregates (no keys),
        // synthesize a constant key column so GroupBy is well-defined.
        let mut group_input_cols: Vec<ProjectExpr> = Vec::new();
        let mut key_cols: Vec<usize> = Vec::new();

        if key_src_cols.is_empty() {
            group_input_cols.push(ProjectExpr::Computed(
                Expr::Const(ConstValue::U32(0)),
                ScalarType::U32,
            ));
            key_cols.push(0);
        } else {
            for (i, &col) in key_src_cols.iter().enumerate() {
                group_input_cols.push(ProjectExpr::Column(col));
                key_cols.push(i);
            }
        }

        let value_offset = group_input_cols.len();
        for &col in &value_src_cols {
            group_input_cols.push(ProjectExpr::Column(col));
        }

        let group_input = RirNode::Project {
            input: Box::new(body),
            columns: group_input_cols,
        };

        // Build multi-aggregation spec list (value_col indices are in the group_input schema).
        let mut aggs: Vec<(usize, CoreAggOp)> = Vec::with_capacity(agg_specs.len());
        for (op, var) in &agg_specs {
            let value_pos = *value_var_to_pos
                .get(var)
                .ok_or_else(|| XlogError::UnsafeVariable(var.clone()))?;
            let value_col = value_offset + value_pos;
            aggs.push((value_col, convert_agg_op(op)));
        }

        let groupby = RirNode::GroupBy {
            input: Box::new(group_input),
            key_cols,
            aggs,
        };

        // Final projection to match head term order:
        // - variables map to group key columns
        // - aggregates map to groupby output agg columns (after keys)
        // - constants are computed columns
        let key_count = if key_src_cols.is_empty() {
            1
        } else {
            key_vars.len()
        };

        let mut final_proj: Vec<ProjectExpr> = Vec::with_capacity(head.terms.len());
        for (index, term) in head.terms.iter().enumerate() {
            match term {
                Term::Variable(name) => {
                    let idx = if key_src_cols.is_empty() {
                        // Global aggregates have no key vars in the output; binding a variable in the head
                        // is a semantic error because it would be unbound.
                        return Err(XlogError::UnsafeVariable(name.clone()));
                    } else {
                        *key_var_to_pos
                            .get(name)
                            .ok_or_else(|| XlogError::UnsafeVariable(name.clone()))?
                    };
                    final_proj.push(ProjectExpr::Column(idx));
                }
                Term::Aggregate(agg) => {
                    let pos = *agg_to_pos
                        .get(&(agg.op, agg.variable.clone()))
                        .ok_or_else(|| XlogError::UnsafeVariable(agg.variable.clone()))?;
                    final_proj.push(ProjectExpr::Column(key_count + pos));
                }
                Term::Anonymous => {
                    return Err(XlogError::Compilation(
                        "Anonymous wildcard '_' not allowed in rule head".to_string(),
                    ));
                }
                Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                    let typ = self
                        .schemas
                        .get(&head.predicate)
                        .and_then(|schema| schema.column_type(index))
                        .ok_or_else(|| {
                            XlogError::Compilation(format!(
                                "Missing schema type for '{}' column {}",
                                head.predicate, index
                            ))
                        })?;
                    let value = term_to_typed_const_value(term, typ)?.ok_or_else(|| {
                        XlogError::Compilation("Expected constant term".to_string())
                    })?;
                    final_proj.push(ProjectExpr::Computed(Expr::Const(value), typ));
                }
                Term::List(_) | Term::Cons { .. } | Term::Compound { .. } | Term::PredRef(_) => {
                    return Err(term_not_lowerable_error(
                        "aggregate rule projection",
                        term_kind_for_lowering_error(term),
                    ));
                }
            }
        }

        if final_proj.is_empty() {
            return Err(XlogError::Compilation(
                "Aggregate rule produced empty head projection".to_string(),
            ));
        }

        Ok(RirNode::Project {
            input: Box::new(groupby),
            columns: final_proj,
        })
    }

    /// Infer an arithmetic result when every type needed for that result is
    /// currently known. An explicit cast fixes its result type independently of
    /// the operand, which lets schema inference propagate the target type before
    /// all upstream variable schemas have converged.
    fn infer_arith_type_from_known_variables<F>(
        expr: &ArithExpr,
        variable_type: &F,
        user_functions: UserFunctionTypeEvidence,
    ) -> Result<Option<ScalarType>>
    where
        F: Fn(&str) -> Option<ScalarType>,
    {
        let mut tasks = vec![ArithmeticTypeTask::Visit(expr)];
        let mut values: Vec<Option<ScalarType>> = Vec::new();
        while let Some(task) = tasks.pop() {
            match task {
                ArithmeticTypeTask::Visit(expression) => match expression {
                    ArithExpr::Variable(name) => values.push(variable_type(name)),
                    ArithExpr::Integer(_) => values.push(Some(ScalarType::I64)),
                    ArithExpr::Float(_) => values.push(Some(ScalarType::F64)),
                    ArithExpr::Add(left, right)
                    | ArithExpr::Sub(left, right)
                    | ArithExpr::Mul(left, right)
                    | ArithExpr::Div(left, right) => {
                        tasks.push(ArithmeticTypeTask::FinishBinary(
                            ArithmeticTypeOperation::Standard,
                        ));
                        tasks.push(ArithmeticTypeTask::Visit(right));
                        tasks.push(ArithmeticTypeTask::Visit(left));
                    }
                    ArithExpr::Mod(left, right) => {
                        tasks.push(ArithmeticTypeTask::FinishBinary(
                            ArithmeticTypeOperation::Modulo,
                        ));
                        tasks.push(ArithmeticTypeTask::Visit(right));
                        tasks.push(ArithmeticTypeTask::Visit(left));
                    }
                    ArithExpr::Min(left, right) | ArithExpr::Max(left, right) => {
                        tasks.push(ArithmeticTypeTask::FinishBinary(
                            ArithmeticTypeOperation::MinMax,
                        ));
                        tasks.push(ArithmeticTypeTask::Visit(right));
                        tasks.push(ArithmeticTypeTask::Visit(left));
                    }
                    ArithExpr::Pow(left, right) => {
                        tasks.push(ArithmeticTypeTask::FinishBinary(
                            ArithmeticTypeOperation::Power,
                        ));
                        tasks.push(ArithmeticTypeTask::Visit(right));
                        tasks.push(ArithmeticTypeTask::Visit(left));
                    }
                    ArithExpr::Abs(inner) => {
                        tasks.push(ArithmeticTypeTask::FinishAbs);
                        tasks.push(ArithmeticTypeTask::Visit(inner));
                    }
                    ArithExpr::Cast(inner, target) => {
                        tasks.push(ArithmeticTypeTask::FinishCast(*target));
                        tasks.push(ArithmeticTypeTask::Visit(inner));
                    }
                    ArithExpr::FuncCall { name, args } => match user_functions {
                        UserFunctionTypeEvidence::RequireExpansion => {
                            return Err(XlogError::Compilation(format!(
                                "User-defined function '{}' must be inlined before lowering",
                                name
                            )));
                        }
                        UserFunctionTypeEvidence::Defer => {
                            tasks.push(ArithmeticTypeTask::FinishFunctionArguments(args.len()));
                            tasks.extend(args.iter().rev().map(ArithmeticTypeTask::Visit));
                        }
                    },
                    ArithExpr::Conditional {
                        then_expr,
                        else_expr,
                        ..
                    } => {
                        tasks.push(ArithmeticTypeTask::FinishConditional);
                        tasks.push(ArithmeticTypeTask::Visit(else_expr));
                        tasks.push(ArithmeticTypeTask::Visit(then_expr));
                    }
                },
                ArithmeticTypeTask::FinishBinary(operation) => {
                    let right = values.pop().expect("right arithmetic type is inferred");
                    let left = values.pop().expect("left arithmetic type is inferred");
                    values.push(Self::finish_binary_arithmetic_type(operation, left, right)?);
                }
                ArithmeticTypeTask::FinishAbs => {
                    let inferred = values.pop().expect("absolute-value type is inferred");
                    if let Some(typ) = inferred {
                        if !Self::is_numeric_type(&typ) {
                            return Err(XlogError::Compilation(format!(
                                "abs requires numeric type, got {:?}",
                                typ
                            )));
                        }
                    }
                    values.push(inferred);
                }
                ArithmeticTypeTask::FinishCast(target) => {
                    values.pop().expect("cast operand type is inferred");
                    values.push(Some(target));
                }
                ArithmeticTypeTask::FinishFunctionArguments(argument_count) => {
                    let start = values
                        .len()
                        .checked_sub(argument_count)
                        .expect("function argument types are inferred");
                    values.truncate(start);
                    values.push(None);
                }
                ArithmeticTypeTask::FinishConditional => {
                    let else_type = values.pop().expect("else branch type is inferred");
                    let then_type = values.pop().expect("then branch type is inferred");
                    let inferred = match (then_type, else_type) {
                        (Some(then_type), Some(else_type)) => {
                            if then_type != else_type {
                                return Err(XlogError::Compilation(format!(
                                    "Conditional branches have different types: {:?} vs {:?}",
                                    then_type, else_type
                                )));
                            }
                            Some(then_type)
                        }
                        _ => None,
                    };
                    values.push(inferred);
                }
            }
        }
        let inferred = values
            .pop()
            .expect("arithmetic expression produces one inferred type");
        debug_assert!(values.is_empty());
        Ok(inferred)
    }

    fn finish_binary_arithmetic_type(
        operation: ArithmeticTypeOperation,
        left: Option<ScalarType>,
        right: Option<ScalarType>,
    ) -> Result<Option<ScalarType>> {
        let matching_numeric_type = |context: &str| -> Result<Option<ScalarType>> {
            if let (Some(left), Some(right)) = (left, right) {
                if left != right {
                    return Err(XlogError::Compilation(format!(
                        "Type mismatch in {context}: {:?} vs {:?}",
                        left, right
                    )));
                }
            }
            for typ in [left, right].into_iter().flatten() {
                if !Self::is_numeric_type(&typ) {
                    return Err(XlogError::Compilation(format!(
                        "{context} requires numeric type, got {:?}",
                        typ
                    )));
                }
            }
            Ok(match (left, right) {
                (Some(left), Some(_)) => Some(left),
                _ => None,
            })
        };

        match operation {
            ArithmeticTypeOperation::Standard => {
                if let (Some(left), Some(right)) = (left, right) {
                    if left != right {
                        return Err(XlogError::Compilation(format!(
                            "Type mismatch in arithmetic: {:?} vs {:?}. Use cast() for conversion.",
                            left, right
                        )));
                    }
                }
                for typ in [left, right].into_iter().flatten() {
                    if !Self::is_numeric_type(&typ) {
                        return Err(XlogError::Compilation(format!(
                            "Arithmetic requires numeric type, got {:?}",
                            typ
                        )));
                    }
                }
                Ok(match (left, right) {
                    (Some(left), Some(_)) => Some(left),
                    _ => None,
                })
            }
            ArithmeticTypeOperation::Modulo => {
                if let (Some(left), Some(right)) = (left, right) {
                    if left != right {
                        return Err(XlogError::Compilation(format!(
                            "Type mismatch in mod: {:?} vs {:?}",
                            left, right
                        )));
                    }
                }
                for typ in [left, right].into_iter().flatten() {
                    if matches!(typ, ScalarType::F32 | ScalarType::F64) {
                        return Err(XlogError::Compilation(
                            "Modulo (%) not supported for floating point".into(),
                        ));
                    }
                    if !Self::is_numeric_type(&typ) {
                        return Err(XlogError::Compilation(format!(
                            "Modulo (%) requires integer operands, got {:?}",
                            typ
                        )));
                    }
                }
                Ok(match (left, right) {
                    (Some(left), Some(_)) => Some(left),
                    _ => None,
                })
            }
            ArithmeticTypeOperation::MinMax => matching_numeric_type("min/max"),
            ArithmeticTypeOperation::Power => {
                if let Some(invalid) = [left, right]
                    .into_iter()
                    .flatten()
                    .find(|typ| !Self::is_numeric_type(typ))
                {
                    let detail = match (left, right) {
                        (Some(left), Some(right)) => format!("{:?} and {:?}", left, right),
                        _ => format!("{:?}", invalid),
                    };
                    return Err(XlogError::Compilation(format!(
                        "pow requires numeric operands, got {detail}"
                    )));
                }
                Ok((left.is_some() && right.is_some()).then_some(ScalarType::F64))
            }
        }
    }
    /// Infer the result type of an arithmetic expression (strict same-type).
    pub(crate) fn infer_arith_type(
        &self,
        expr: &ArithExpr,
        var_env: &VariableEnv,
    ) -> Result<ScalarType> {
        if let Some(typ) = Self::infer_arith_type_from_known_variables(
            expr,
            &|variable| var_env.get_type(variable),
            UserFunctionTypeEvidence::RequireExpansion,
        )? {
            return Ok(typ);
        }

        let variable = expr
            .variables()
            .into_iter()
            .find(|variable| var_env.get_type(variable).is_none())
            .unwrap_or("<unknown>");
        Err(XlogError::Compilation(format!(
            "Unknown variable {} in arithmetic",
            variable
        )))
    }

    fn is_numeric_type(t: &ScalarType) -> bool {
        matches!(
            t,
            ScalarType::I32
                | ScalarType::I64
                | ScalarType::U32
                | ScalarType::U64
                | ScalarType::F32
                | ScalarType::F64
        )
    }

    /// Convert ArithExpr to IR Expr
    fn arith_to_expr(&self, arith: &ArithExpr, var_env: &VariableEnv) -> Result<Expr> {
        let mut tasks = vec![ArithmeticExpressionTask::Visit(arith)];
        let mut values = Vec::new();
        while let Some(task) = tasks.pop() {
            match task {
                ArithmeticExpressionTask::Visit(expression) => match expression {
                    ArithExpr::Variable(name) => {
                        let column = var_env.get_column(name).ok_or_else(|| {
                            XlogError::Compilation(format!(
                                "Variable {} not bound before use in arithmetic",
                                name
                            ))
                        })?;
                        values.push(Expr::Column(column));
                    }
                    ArithExpr::Integer(value) => {
                        values.push(Expr::Const(ConstValue::I64(*value)));
                    }
                    ArithExpr::Float(value) => {
                        values.push(Expr::Const(ConstValue::F64(*value)));
                    }
                    ArithExpr::Add(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Add,
                    ),
                    ArithExpr::Sub(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Sub,
                    ),
                    ArithExpr::Mul(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Mul,
                    ),
                    ArithExpr::Div(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Div,
                    ),
                    ArithExpr::Mod(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Mod,
                    ),
                    ArithExpr::Min(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Min,
                    ),
                    ArithExpr::Max(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Max,
                    ),
                    ArithExpr::Pow(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Pow,
                    ),
                    ArithExpr::Abs(inner) => {
                        tasks.push(ArithmeticExpressionTask::FinishAbs);
                        tasks.push(ArithmeticExpressionTask::Visit(inner));
                    }
                    ArithExpr::Cast(inner, target) => {
                        tasks.push(ArithmeticExpressionTask::FinishCast(*target));
                        tasks.push(ArithmeticExpressionTask::Visit(inner));
                    }
                    ArithExpr::FuncCall { name, .. } => {
                        return Err(XlogError::Compilation(format!(
                            "User-defined function '{}' must be inlined before lowering",
                            name
                        )));
                    }
                    ArithExpr::Conditional {
                        cond_left,
                        cond_op,
                        cond_right,
                        then_expr,
                        else_expr,
                    } => {
                        tasks.push(ArithmeticExpressionTask::FinishConditional(*cond_op));
                        tasks.push(ArithmeticExpressionTask::Visit(else_expr));
                        tasks.push(ArithmeticExpressionTask::Visit(then_expr));
                        tasks.push(ArithmeticExpressionTask::Visit(cond_right));
                        tasks.push(ArithmeticExpressionTask::Visit(cond_left));
                    }
                },
                ArithmeticExpressionTask::FinishBinary(operation) => {
                    let right = values
                        .pop()
                        .expect("right arithmetic expression is lowered");
                    let left = values.pop().expect("left arithmetic expression is lowered");
                    values.push(match operation {
                        ArithmeticExpressionOperation::Add => {
                            Expr::Add(Box::new(left), Box::new(right))
                        }
                        ArithmeticExpressionOperation::Sub => {
                            Expr::Sub(Box::new(left), Box::new(right))
                        }
                        ArithmeticExpressionOperation::Mul => {
                            Expr::Mul(Box::new(left), Box::new(right))
                        }
                        ArithmeticExpressionOperation::Div => {
                            Expr::Div(Box::new(left), Box::new(right))
                        }
                        ArithmeticExpressionOperation::Mod => {
                            Expr::Mod(Box::new(left), Box::new(right))
                        }
                        ArithmeticExpressionOperation::Min => {
                            Expr::Min(Box::new(left), Box::new(right))
                        }
                        ArithmeticExpressionOperation::Max => {
                            Expr::Max(Box::new(left), Box::new(right))
                        }
                        ArithmeticExpressionOperation::Pow => {
                            Expr::Pow(Box::new(left), Box::new(right))
                        }
                    });
                }
                ArithmeticExpressionTask::FinishAbs => {
                    let inner = values.pop().expect("absolute-value expression is lowered");
                    values.push(Expr::Abs(Box::new(inner)));
                }
                ArithmeticExpressionTask::FinishCast(target) => {
                    let inner = values.pop().expect("cast expression is lowered");
                    values.push(Expr::Cast(Box::new(inner), target));
                }
                ArithmeticExpressionTask::FinishConditional(op) => {
                    let else_expr = values.pop().expect("else expression is lowered");
                    let then_expr = values.pop().expect("then expression is lowered");
                    let right = values.pop().expect("condition right side is lowered");
                    let left = values.pop().expect("condition left side is lowered");
                    let op = match op {
                        CompOp::Eq => CompareOp::Eq,
                        CompOp::Ne => CompareOp::Ne,
                        CompOp::Lt => CompareOp::Lt,
                        CompOp::Le => CompareOp::Le,
                        CompOp::Gt => CompareOp::Gt,
                        CompOp::Ge => CompareOp::Ge,
                    };
                    values.push(Expr::Conditional {
                        condition: Box::new(Expr::Compare {
                            left: Box::new(left),
                            op,
                            right: Box::new(right),
                        }),
                        then_expr: Box::new(then_expr),
                        else_expr: Box::new(else_expr),
                    });
                }
            }
        }
        let expression = values
            .pop()
            .expect("arithmetic expression produces one lowered value");
        debug_assert!(values.is_empty());
        Ok(expression)
    }

    fn schedule_arithmetic_expression<'a>(
        tasks: &mut Vec<ArithmeticExpressionTask<'a>>,
        left: &'a ArithExpr,
        right: &'a ArithExpr,
        operation: ArithmeticExpressionOperation,
    ) {
        tasks.push(ArithmeticExpressionTask::FinishBinary(operation));
        tasks.push(ArithmeticExpressionTask::Visit(right));
        tasks.push(ArithmeticExpressionTask::Visit(left));
    }
    /// Lower an is-expression to a Project node with computed column
    fn lower_is_expr(
        &mut self,
        is_expr: &IsExpr,
        input: RirNode,
        var_env: &mut VariableEnv,
    ) -> Result<RirNode> {
        // 1. Verify target is NOT already bound
        if var_env.contains(&is_expr.target) {
            return Err(XlogError::Compilation(format!(
                "Variable {} already bound; 'is' requires fresh variable",
                is_expr.target
            )));
        }

        // 2. Verify all variables in expression are bound
        for var in is_expr.expr.variables() {
            if !var_env.contains(var) {
                return Err(XlogError::Compilation(format!(
                    "Variable {} used in arithmetic but not bound",
                    var
                )));
            }
        }

        // 3. Infer result type
        let result_type = self.infer_arith_type(&is_expr.expr, var_env)?;

        // 4. Convert expression to IR
        let ir_expr = self.arith_to_expr(&is_expr.expr, var_env)?;

        // 5. Build projection: pass through all existing columns + add computed column
        let num_cols = var_env.column_count();
        let mut proj_exprs: Vec<ProjectExpr> = (0..num_cols).map(ProjectExpr::Column).collect();
        proj_exprs.push(ProjectExpr::Computed(ir_expr, result_type));

        // 6. Bind the new variable
        var_env.bind(&is_expr.target, num_cols, result_type);

        Ok(RirNode::Project {
            input: Box::new(input),
            columns: proj_exprs,
        })
    }
}

/// Track variable occurrences and column positions
pub(crate) struct VariableEnv {
    /// Maps variable name to list of (predicate, position in atom, global column)
    occurrences: HashMap<String, Vec<(String, usize, usize)>>,
    /// Total columns in current result
    total_cols: usize,
    /// Maps variable name to its type (for type inference)
    types: HashMap<String, ScalarType>,
}

impl VariableEnv {
    fn new() -> Self {
        Self {
            occurrences: HashMap::new(),
            total_cols: 0,
            types: HashMap::new(),
        }
    }

    fn add_occurrence(&mut self, var: &str, pred: String, atom_pos: usize, global_col: usize) {
        self.occurrences
            .entry(var.to_string())
            .or_default()
            .push((pred, atom_pos, global_col));
    }

    fn get_column(&self, var: &str) -> Option<usize> {
        self.occurrences
            .get(var)
            .and_then(|occs| occs.first())
            .map(|(_, _, col)| *col)
    }

    /// Bind a variable to a column with a specific type (for type inference)
    fn bind(&mut self, name: &str, column: usize, typ: ScalarType) {
        self.types.insert(name.to_string(), typ);
        // Also add occurrence for column lookup
        self.occurrences
            .entry(name.to_string())
            .or_default()
            .push(("".to_string(), 0, column));
        // Update total_cols to account for the new computed column
        // This is critical for chained is-expressions where each adds a column
        if column >= self.total_cols {
            self.total_cols = column + 1;
        }
    }

    /// Get the type of a bound variable
    fn get_type(&self, name: &str) -> Option<ScalarType> {
        self.types.get(name).copied()
    }

    /// Check if a variable is bound
    fn contains(&self, name: &str) -> bool {
        self.occurrences.contains_key(name)
    }

    /// Get the current column count (for adding new computed columns)
    fn column_count(&self) -> usize {
        self.total_cols
    }
}

fn sort_labels_from_terms(terms: &[Term]) -> Vec<String> {
    terms
        .iter()
        .enumerate()
        .map(|(idx, term)| match term {
            Term::Variable(name) if !name.trim().is_empty() => name.clone(),
            Term::Aggregate(agg) => format!("{:?}_{}", agg.op, agg.variable),
            Term::List(_) => format!("list{}", idx),
            Term::Cons { .. } => format!("cons{}", idx),
            Term::Compound { functor, .. } => functor.clone(),
            Term::PredRef(name) => name.clone(),
            _ => format!("c{}", idx),
        })
        .collect()
}

/// Convert a term to a constant value (if it is a constant)
fn term_to_const_value(term: &Term) -> Option<ConstValue> {
    match term {
        Term::Integer(i) => Some(ConstValue::I64(*i)),
        Term::Float(f) => Some(ConstValue::F64(*f)),
        Term::String(s) => Some(ConstValue::Symbol(s.clone())),
        Term::Symbol(id) => Some(ConstValue::Symbol(symbol::resolve(*id))),
        Term::Variable(_)
        | Term::Anonymous
        | Term::Aggregate(_)
        | Term::List(_)
        | Term::Cons { .. }
        | Term::Compound { .. }
        | Term::PredRef(_) => None,
    }
}

pub(crate) fn term_to_typed_const_value(
    term: &Term,
    expected: ScalarType,
) -> Result<Option<ConstValue>> {
    let const_val = match term {
        Term::Integer(i) => match expected {
            ScalarType::U32 => {
                if *i >= 0 && *i <= u32::MAX as i64 {
                    ConstValue::U32(*i as u32)
                } else {
                    return Err(XlogError::Compilation(format!(
                        "Integer literal {} out of range for {:?}",
                        i, expected
                    )));
                }
            }
            ScalarType::U64 => {
                if *i >= 0 {
                    ConstValue::U64(*i as u64)
                } else {
                    return Err(XlogError::Compilation(format!(
                        "Integer literal {} out of range for {:?}",
                        i, expected
                    )));
                }
            }
            ScalarType::I32 => {
                if *i >= i32::MIN as i64 && *i <= i32::MAX as i64 {
                    ConstValue::I32(*i as i32)
                } else {
                    return Err(XlogError::Compilation(format!(
                        "Integer literal {} out of range for {:?}",
                        i, expected
                    )));
                }
            }
            ScalarType::I64 => ConstValue::I64(*i),
            ScalarType::F32 => {
                let value = *i as f64;
                if value < f32::MIN as f64 || value > f32::MAX as f64 {
                    return Err(XlogError::Compilation(format!(
                        "Integer literal {} out of range for {:?}",
                        i, expected
                    )));
                }
                ConstValue::F32(value as f32)
            }
            ScalarType::F64 => ConstValue::F64(*i as f64),
            ScalarType::Bool => {
                if *i == 0 || *i == 1 {
                    ConstValue::Bool(*i == 1)
                } else {
                    return Err(XlogError::Compilation(format!(
                        "Integer literal {} not valid for {:?}",
                        i, expected
                    )));
                }
            }
            ScalarType::Symbol => {
                return Err(XlogError::Compilation(format!(
                    "Integer literal {} not valid for {:?}",
                    i, expected
                )));
            }
        },
        Term::Float(f) => match expected {
            ScalarType::F32 => {
                if !f.is_finite() {
                    return Err(XlogError::Compilation(format!(
                        "Float literal {} not valid for {:?}",
                        f, expected
                    )));
                }
                if *f < f32::MIN as f64 || *f > f32::MAX as f64 {
                    return Err(XlogError::Compilation(format!(
                        "Float literal {} out of range for {:?}",
                        f, expected
                    )));
                }
                ConstValue::F32(*f as f32)
            }
            ScalarType::F64 => ConstValue::F64(*f),
            ScalarType::U32
            | ScalarType::U64
            | ScalarType::I32
            | ScalarType::I64
            | ScalarType::Bool
            | ScalarType::Symbol => {
                return Err(XlogError::Compilation(format!(
                    "Float literal {} not valid for {:?}",
                    f, expected
                )));
            }
        },
        Term::String(s) => {
            if expected == ScalarType::Symbol {
                ConstValue::Symbol(s.clone())
            } else {
                return Err(XlogError::Compilation(format!(
                    "String literal {} not valid for {:?}",
                    s, expected
                )));
            }
        }
        Term::Symbol(id) => {
            let value = symbol::resolve(*id);
            match expected {
                ScalarType::Symbol => ConstValue::Symbol(value),
                ScalarType::Bool if matches!(value.as_str(), "true" | "false") => {
                    ConstValue::Bool(value == "true")
                }
                _ => {
                    return Err(XlogError::Compilation(format!(
                        "Symbol literal {} not valid for {:?}",
                        value, expected
                    )));
                }
            }
        }
        Term::Variable(_)
        | Term::Anonymous
        | Term::Aggregate(_)
        | Term::List(_)
        | Term::Cons { .. }
        | Term::Compound { .. }
        | Term::PredRef(_) => return Ok(None),
    };

    Ok(Some(const_val))
}

/// Convert AST AggOp to core AggOp
fn convert_agg_op(op: &AggOp) -> CoreAggOp {
    match op {
        AggOp::Count => CoreAggOp::Count,
        AggOp::Sum => CoreAggOp::Sum,
        AggOp::Min => CoreAggOp::Min,
        AggOp::Max => CoreAggOp::Max,
        AggOp::LogSumExp => CoreAggOp::LogSumExp,
    }
}

// Export the find_sccs_for_lowering function from stratify
// We need to add this to the stratify module

#[cfg(test)]
mod arith_type_tests {
    use super::*;
    use crate::ast::ArithExpr;

    #[test]
    fn test_arith_type_inference_same_type() {
        // X + Y where both are i64 should succeed and return i64
        let lowerer = Lowerer::new();
        let mut var_env = VariableEnv::new();
        var_env.bind("X", 0, ScalarType::I64);
        var_env.bind("Y", 1, ScalarType::I64);

        let expr = ArithExpr::Add(
            Box::new(ArithExpr::Variable("X".to_string())),
            Box::new(ArithExpr::Variable("Y".to_string())),
        );
        let result = lowerer.infer_arith_type(&expr, &var_env);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ScalarType::I64);
    }

    #[test]
    fn test_arith_type_inference_mismatch() {
        // X + Y where X is i64 and Y is f64 should fail
        let lowerer = Lowerer::new();
        let mut var_env = VariableEnv::new();
        var_env.bind("X", 0, ScalarType::I64);
        var_env.bind("Y", 1, ScalarType::F64);

        let expr = ArithExpr::Add(
            Box::new(ArithExpr::Variable("X".to_string())),
            Box::new(ArithExpr::Variable("Y".to_string())),
        );
        let result = lowerer.infer_arith_type(&expr, &var_env);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    fn pred_decl(name: &str, types: Vec<ScalarType>) -> PredDecl {
        let type_refs: Vec<TypeRef> = types.into_iter().map(TypeRef::Scalar).collect();
        let columns = type_refs
            .iter()
            .cloned()
            .map(|typ| PredColumn { name: None, typ })
            .collect();
        PredDecl {
            name: name.to_string(),
            types: type_refs,
            columns,
            is_private: false,
        }
    }

    /// Helper to create a simple edge atom
    fn edge_atom(x: &str, y: &str) -> Atom {
        Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Variable(x.to_string()), Term::Variable(y.to_string())],
        }
    }

    /// Helper to create a reach atom
    fn reach_atom(x: &str, y: &str) -> Atom {
        Atom {
            predicate: "reach".to_string(),
            terms: vec![Term::Variable(x.to_string()), Term::Variable(y.to_string())],
        }
    }

    /// Helper to create a node atom
    fn node_atom(x: &str) -> Atom {
        Atom {
            predicate: "node".to_string(),
            terms: vec![Term::Variable(x.to_string())],
        }
    }

    #[test]
    fn test_lowerer_new() {
        let lowerer = Lowerer::new();
        assert!(lowerer.schemas.is_empty());
        assert!(lowerer.strata.is_empty());
        assert_eq!(lowerer.next_rel_id, 0);
    }

    #[test]
    fn test_get_or_create_rel_id() {
        let mut lowerer = Lowerer::new();
        let id1 = lowerer.get_or_create_rel_id("edge");
        let id2 = lowerer.get_or_create_rel_id("reach");
        let id3 = lowerer.get_or_create_rel_id("edge");

        assert_eq!(id1, RelId(0));
        assert_eq!(id2, RelId(1));
        assert_eq!(id3, RelId(0)); // Same as id1
    }

    #[test]
    fn test_infer_schemas_from_facts() {
        let mut program = Program::new();
        program.rules.push(Rule {
            head: Atom {
                predicate: "edge".to_string(),
                terms: vec![Term::Integer(1), Term::Integer(2)],
            },
            body: vec![],
        });

        let mut lowerer = Lowerer::new();
        lowerer.infer_schemas(&program).unwrap();

        assert!(lowerer.schemas.contains_key("edge"));
        let schema = lowerer.schemas.get("edge").unwrap();
        assert_eq!(schema.arity(), 2);
    }

    #[test]
    fn test_infer_schemas_propagates_through_reversed_rule_chains() {
        let mut program = Program::new();
        program.rules.push(Rule {
            head: Atom {
                predicate: "shared".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "intermediate".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            })],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "intermediate".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "source".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            })],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "source".to_string(),
                terms: vec![Term::Symbol(symbol::intern("value"))],
            },
            body: vec![],
        });

        let mut lowerer = Lowerer::new();
        lowerer.infer_schemas(&program).unwrap();

        assert_eq!(
            lowerer
                .schemas
                .get("shared")
                .and_then(|schema| schema.column_type(0)),
            Some(ScalarType::Symbol)
        );
    }

    #[test]
    fn test_lower_simple_rule() {
        // reach(X, Y) :- edge(X, Y).
        let rule = Rule {
            head: reach_atom("X", "Y"),
            body: vec![BodyLiteral::Positive(edge_atom("X", "Y"))],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "edge".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::U32),
                ("c1".to_string(), ScalarType::U32),
            ]),
        );

        let result = lowerer.lower_rule(&rule);
        assert!(result.is_ok());

        let node = result.unwrap();
        // Should be just a scan (no projection needed since columns match)
        assert!(matches!(node, RirNode::Scan { .. }));
    }

    #[test]
    fn test_lower_join_rule() {
        // reach(X, Z) :- reach(X, Y), edge(Y, Z).
        let rule = Rule {
            head: Atom {
                predicate: "reach".to_string(),
                terms: vec![
                    Term::Variable("X".to_string()),
                    Term::Variable("Z".to_string()),
                ],
            },
            body: vec![
                BodyLiteral::Positive(reach_atom("X", "Y")),
                BodyLiteral::Positive(edge_atom("Y", "Z")),
            ],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "reach".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::U32),
                ("c1".to_string(), ScalarType::U32),
            ]),
        );
        lowerer.schemas.insert(
            "edge".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::U32),
                ("c1".to_string(), ScalarType::U32),
            ]),
        );

        let result = lowerer.lower_rule(&rule);
        assert!(result.is_ok());

        let node = result.unwrap();
        // Should be Project(Join(Scan, Scan))
        if let RirNode::Project { input, columns } = node {
            // X from reach (col 0), Z from edge (col 3)
            assert_eq!(
                columns,
                vec![ProjectExpr::Column(0), ProjectExpr::Column(3)]
            );
            assert!(matches!(*input, RirNode::Join { .. }));
            if let RirNode::Join {
                left_keys,
                right_keys,
                ..
            } = *input
            {
                assert_eq!(left_keys, vec![1]); // Y in reach (position 1)
                assert_eq!(right_keys, vec![0]); // Y in edge (position 0)
            }
        } else {
            panic!("Expected Project node");
        }
    }

    #[test]
    fn test_join_order_prefers_smaller_relation() {
        // out(X) :- big(X), small(X).
        let rule = Rule {
            head: Atom {
                predicate: "out".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "big".to_string(),
                    terms: vec![Term::Variable("X".to_string())],
                }),
                BodyLiteral::Positive(Atom {
                    predicate: "small".to_string(),
                    terms: vec![Term::Variable("X".to_string())],
                }),
            ],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "big".to_string(),
            Schema::new(vec![("c0".to_string(), ScalarType::U32)]),
        );
        lowerer.schemas.insert(
            "small".to_string(),
            Schema::new(vec![("c0".to_string(), ScalarType::U32)]),
        );

        // Ensure stable RelIds independent of join order.
        let big_id = lowerer.get_or_create_rel_id("big");
        let small_id = lowerer.get_or_create_rel_id("small");
        assert_eq!(big_id, RelId(0));
        assert_eq!(small_id, RelId(1));

        // Prefer scanning the smaller relation first.
        lowerer.est_cardinality.insert("big".to_string(), 10_000);
        lowerer.est_cardinality.insert("small".to_string(), 10);

        let node = lowerer.lower_rule(&rule).unwrap();
        let join = match node {
            RirNode::Project { input, .. } => *input,
            other => other,
        };

        match join {
            RirNode::Join { left, right, .. } => {
                // Prefer building the hash table on the smaller relation (right/build side).
                assert!(matches!(*left, RirNode::Scan { rel } if rel == big_id));
                assert!(matches!(*right, RirNode::Scan { rel } if rel == small_id));
            }
            other => panic!("Expected Join node, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_negation() {
        // isolated(X) :- node(X), not edge(X, _).
        let rule = Rule {
            head: Atom {
                predicate: "isolated".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            },
            body: vec![
                BodyLiteral::Positive(node_atom("X")),
                BodyLiteral::Negated(Atom {
                    predicate: "edge".to_string(),
                    terms: vec![
                        Term::Variable("X".to_string()),
                        Term::Variable("_".to_string()),
                    ],
                }),
            ],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "node".to_string(),
            Schema::new(vec![("c0".to_string(), ScalarType::U32)]),
        );
        lowerer.schemas.insert(
            "edge".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::U32),
                ("c1".to_string(), ScalarType::U32),
            ]),
        );

        let result = lowerer.lower_rule(&rule);
        assert!(result.is_ok());

        // The result should involve a Diff or semi-join for negation
        let node = result.unwrap();
        // Verify the structure contains the negation handling
        fn contains_diff_or_semi(node: &RirNode) -> bool {
            match node {
                RirNode::Diff { .. } => true,
                RirNode::Join {
                    join_type: JoinType::Semi,
                    ..
                } => true,
                RirNode::Join { left, right, .. } => {
                    contains_diff_or_semi(left) || contains_diff_or_semi(right)
                }
                RirNode::Project { input, .. } => contains_diff_or_semi(input),
                RirNode::Filter { input, .. } => contains_diff_or_semi(input),
                _ => false,
            }
        }
        assert!(contains_diff_or_semi(&node));
    }

    #[test]
    fn test_lower_ground_negation_preserves_the_positive_input_schema() {
        // ok(X) :- x(X), not p(3).
        let rule = Rule {
            head: Atom {
                predicate: "ok".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "x".to_string(),
                    terms: vec![Term::Variable("X".to_string())],
                }),
                BodyLiteral::Negated(Atom {
                    predicate: "p".to_string(),
                    terms: vec![Term::Integer(3)],
                }),
            ],
        };

        let mut lowerer = Lowerer::new();
        for predicate in ["ok", "x", "p"] {
            lowerer.schemas.insert(
                predicate.to_string(),
                Schema::new(vec![("c0".to_string(), ScalarType::U32)]),
            );
        }

        let node = lowerer.lower_rule(&rule).expect("lower ground negation");

        let RirNode::Project { input, columns } = node else {
            panic!("ground negation must project away its internal existence key");
        };
        assert_eq!(columns, vec![ProjectExpr::Column(0)]);
        assert!(matches!(
            *input,
            RirNode::Join {
                left_keys,
                right_keys,
                join_type: JoinType::Anti,
                ..
            } if left_keys == vec![1] && right_keys == vec![0]
        ));
    }

    #[test]
    fn test_lower_comparison() {
        // greater(X, Y) :- pair(X, Y), X > Y.
        let rule = Rule {
            head: Atom {
                predicate: "greater".to_string(),
                terms: vec![
                    Term::Variable("X".to_string()),
                    Term::Variable("Y".to_string()),
                ],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "pair".to_string(),
                    terms: vec![
                        Term::Variable("X".to_string()),
                        Term::Variable("Y".to_string()),
                    ],
                }),
                BodyLiteral::Comparison(Comparison {
                    left: Term::Variable("X".to_string()),
                    op: CompOp::Gt,
                    right: Term::Variable("Y".to_string()),
                }),
            ],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "pair".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::U32),
                ("c1".to_string(), ScalarType::U32),
            ]),
        );

        let result = lowerer.lower_rule(&rule);
        assert!(result.is_ok());

        let node = result.unwrap();
        // Should contain a Filter node
        fn contains_filter(node: &RirNode) -> bool {
            match node {
                RirNode::Filter { .. } => true,
                RirNode::Project { input, .. } => contains_filter(input),
                RirNode::Join { left, right, .. } => {
                    contains_filter(left) || contains_filter(right)
                }
                _ => false,
            }
        }
        assert!(contains_filter(&node));
    }

    #[test]
    fn test_lower_constant_filter() {
        // specific_edge(Y) :- edge(1, Y).
        let rule = Rule {
            head: Atom {
                predicate: "specific_edge".to_string(),
                terms: vec![Term::Variable("Y".to_string())],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "edge".to_string(),
                terms: vec![Term::Integer(1), Term::Variable("Y".to_string())],
            })],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "edge".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::U32),
                ("c1".to_string(), ScalarType::U32),
            ]),
        );

        let result = lowerer.lower_rule(&rule);
        assert!(result.is_ok());

        let node = result.unwrap();
        // Should contain a Filter for the constant 1
        fn has_const_filter(node: &RirNode) -> bool {
            match node {
                RirNode::Filter {
                    predicate: Expr::Compare { right, .. },
                    ..
                } => matches!(**right, Expr::Const(_)),
                RirNode::Project { input, .. } => has_const_filter(input),
                _ => false,
            }
        }
        assert!(has_const_filter(&node));
    }

    #[test]
    fn test_lower_repeated_variable_filter() {
        // self_loop(X) :- edge(X, X).
        let rule = Rule {
            head: Atom {
                predicate: "self_loop".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "edge".to_string(),
                terms: vec![
                    Term::Variable("X".to_string()),
                    Term::Variable("X".to_string()),
                ],
            })],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "edge".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::U32),
                ("c1".to_string(), ScalarType::U32),
            ]),
        );

        let node = lowerer.lower_rule(&rule).expect("lower_rule failed");

        fn has_col_eq_filter(node: &RirNode) -> bool {
            match node {
                RirNode::Filter { predicate, .. } => match predicate {
                    Expr::Compare {
                        left,
                        op: CompareOp::Eq,
                        right,
                    } => {
                        matches!((&**left, &**right), (Expr::Column(0), Expr::Column(1)))
                            || matches!((&**left, &**right), (Expr::Column(1), Expr::Column(0)))
                    }
                    Expr::And(exprs) => exprs.iter().any(|e| match e {
                        Expr::Compare {
                            left,
                            op: CompareOp::Eq,
                            right,
                        } => {
                            matches!((&**left, &**right), (Expr::Column(0), Expr::Column(1)))
                                || matches!((&**left, &**right), (Expr::Column(1), Expr::Column(0)))
                        }
                        _ => false,
                    }),
                    _ => false,
                },
                RirNode::Project { input, .. } => has_col_eq_filter(input),
                _ => false,
            }
        }

        assert!(has_col_eq_filter(&node));
    }

    #[test]
    fn test_lower_program_simple() {
        let mut program = Program::new();

        // edge(1, 2).
        program.rules.push(Rule {
            head: Atom {
                predicate: "edge".to_string(),
                terms: vec![Term::Integer(1), Term::Integer(2)],
            },
            body: vec![],
        });

        // reach(X, Y) :- edge(X, Y).
        program.rules.push(Rule {
            head: reach_atom("X", "Y"),
            body: vec![BodyLiteral::Positive(edge_atom("X", "Y"))],
        });

        let mut lowerer = Lowerer::new();
        lowerer.set_strata(vec![vec!["edge".to_string()], vec!["reach".to_string()]]);

        let result = lowerer.lower_program(&program);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(!plan.sccs.is_empty());
    }

    #[test]
    fn facts_are_metadata_and_not_executable_rules() {
        let mut program = Program::new();
        for value in [1, 2, 2] {
            program.rules.push(Rule {
                head: Atom {
                    predicate: "base".to_string(),
                    terms: vec![Term::Integer(value)],
                },
                body: vec![],
            });
        }

        let mut lowerer = Lowerer::new();
        lowerer.set_strata(vec![vec!["base".to_string()]]);

        let plan = lowerer.lower_program(&program).unwrap();

        assert_eq!(
            plan.rules_by_scc.iter().map(Vec::len).sum::<usize>(),
            0,
            "facts are materialized by the relation loader, not executed as rules"
        );
        assert_eq!(lowerer.schemas.get("base").unwrap().arity(), 1);
        assert_eq!(lowerer.est_cardinality.get("base"), Some(&3));
        assert!(lowerer
            .sccs
            .iter()
            .any(|scc| scc.predicates.iter().any(|predicate| predicate == "base")));

        let base_id = lowerer.rel_ids().get("base").copied().unwrap();
        assert_eq!(plan.rel_arities.get(&base_id), Some(&1));
    }

    #[test]
    fn test_variable_env() {
        let mut env = VariableEnv::new();
        env.add_occurrence("X", "edge".to_string(), 0, 0);
        env.add_occurrence("Y", "edge".to_string(), 1, 1);
        env.add_occurrence("Y", "node".to_string(), 0, 2);

        assert_eq!(env.get_column("X"), Some(0));
        assert_eq!(env.get_column("Y"), Some(1)); // First occurrence
        assert_eq!(env.get_column("Z"), None);
    }

    #[test]
    fn test_inferred_scalar_type() {
        assert_eq!(
            Term::Variable("X".to_string()).inferred_scalar_type(),
            ScalarType::U64
        );
        assert_eq!(Term::Integer(42).inferred_scalar_type(), ScalarType::U32);
        assert_eq!(
            Term::Integer(i64::MAX).inferred_scalar_type(),
            ScalarType::I64
        );
        assert_eq!(Term::Float(3.25).inferred_scalar_type(), ScalarType::F64);
        assert_eq!(
            Term::Symbol(symbol::intern("foo")).inferred_scalar_type(),
            ScalarType::Symbol
        );
    }

    #[test]
    fn test_convert_agg_op() {
        assert_eq!(convert_agg_op(&AggOp::Count), CoreAggOp::Count);
        assert_eq!(convert_agg_op(&AggOp::Sum), CoreAggOp::Sum);
        assert_eq!(convert_agg_op(&AggOp::Min), CoreAggOp::Min);
        assert_eq!(convert_agg_op(&AggOp::Max), CoreAggOp::Max);
        assert_eq!(convert_agg_op(&AggOp::LogSumExp), CoreAggOp::LogSumExp);
    }

    #[test]
    fn test_variable_env_bind_updates_total_cols() {
        // Test that bind() properly updates total_cols for chained is-expressions
        let mut env = VariableEnv::new();
        env.total_cols = 2; // Simulate 2 columns from atoms

        // Bind first computed variable at column 2
        env.bind("A", 2, ScalarType::I64);
        assert_eq!(
            env.column_count(),
            3,
            "total_cols should be 3 after first bind"
        );
        assert_eq!(env.get_column("A"), Some(2));

        // Bind second computed variable at column 3
        env.bind("B", 3, ScalarType::I64);
        assert_eq!(
            env.column_count(),
            4,
            "total_cols should be 4 after second bind"
        );
        assert_eq!(env.get_column("B"), Some(3));
    }

    #[test]
    fn test_lower_chained_is_expressions() {
        // result(A, B) :- input(X, Y), A is X + Y, B is A * 2.
        // This tests that chained is-expressions correctly update column indices
        let rule = Rule {
            head: Atom {
                predicate: "result".to_string(),
                terms: vec![
                    Term::Variable("A".to_string()),
                    Term::Variable("B".to_string()),
                ],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "input".to_string(),
                    terms: vec![
                        Term::Variable("X".to_string()),
                        Term::Variable("Y".to_string()),
                    ],
                }),
                BodyLiteral::IsExpr(IsExpr {
                    target: "A".to_string(),
                    expr: ArithExpr::Add(
                        Box::new(ArithExpr::Variable("X".to_string())),
                        Box::new(ArithExpr::Variable("Y".to_string())),
                    ),
                }),
                BodyLiteral::IsExpr(IsExpr {
                    target: "B".to_string(),
                    expr: ArithExpr::Mul(
                        Box::new(ArithExpr::Variable("A".to_string())),
                        Box::new(ArithExpr::Integer(2)),
                    ),
                }),
            ],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "input".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::I64),
                ("c1".to_string(), ScalarType::I64),
            ]),
        );

        let result = lowerer.lower_rule(&rule);
        assert!(
            result.is_ok(),
            "Lowering chained is-expressions should succeed: {:?}",
            result.err()
        );

        let node = result.unwrap();

        // The structure should be:
        // Project([col 2, col 3]) <-- final projection for A, B
        //   Project([col 0, col 1, col 2, A*2]) <-- second is-expr adds B at col 3
        //     Project([col 0, col 1, X+Y]) <-- first is-expr adds A at col 2
        //       Scan(input)

        // Verify we have nested Project nodes
        fn count_projects(node: &RirNode) -> usize {
            match node {
                RirNode::Project { input, .. } => 1 + count_projects(input),
                _ => 0,
            }
        }

        // We expect 3 Project nodes: 2 for is-expressions + 1 for final head projection
        let project_count = count_projects(&node);
        assert!(
            project_count >= 2,
            "Expected at least 2 Project nodes for chained is-exprs, got {}",
            project_count
        );

        // Verify the final projection references columns 2 and 3 (A and B)
        if let RirNode::Project { columns, .. } = &node {
            assert_eq!(columns.len(), 2, "Head has 2 variables");
            // A should be at column 2, B at column 3
            assert_eq!(columns[0], ProjectExpr::Column(2), "A should be column 2");
            assert_eq!(columns[1], ProjectExpr::Column(3), "B should be column 3");
        } else {
            panic!("Expected top-level Project node");
        }
    }

    #[test]
    fn test_u64_comparison_type_from_pred_decl() {
        // Test that u64 type from pred decl is preserved in comparison lowering
        let mut program = Program::new();

        // pred count_data(symbol, u64).
        program.predicates.push(pred_decl(
            "count_data",
            vec![ScalarType::Symbol, ScalarType::U64],
        ));

        // count_data(alice, 5).
        program.rules.push(Rule {
            head: Atom {
                predicate: "count_data".to_string(),
                terms: vec![
                    Term::Symbol(xlog_core::symbol::intern("alice")),
                    Term::Integer(5),
                ],
            },
            body: vec![],
        });

        // pred big_count(symbol, u64).
        program.predicates.push(pred_decl(
            "big_count",
            vec![ScalarType::Symbol, ScalarType::U64],
        ));

        // big_count(Name, Count) :- count_data(Name, Count), Count >= 3.
        program.rules.push(Rule {
            head: Atom {
                predicate: "big_count".to_string(),
                terms: vec![
                    Term::Variable("Name".to_string()),
                    Term::Variable("Count".to_string()),
                ],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "count_data".to_string(),
                    terms: vec![
                        Term::Variable("Name".to_string()),
                        Term::Variable("Count".to_string()),
                    ],
                }),
                BodyLiteral::Comparison(Comparison {
                    left: Term::Variable("Count".to_string()),
                    op: CompOp::Ge,
                    right: Term::Integer(3),
                }),
            ],
        });

        let mut lowerer = Lowerer::new();
        lowerer.infer_schemas(&program).unwrap();

        // Verify schema has correct types
        let schema = lowerer
            .schemas
            .get("count_data")
            .expect("schema for count_data");
        assert_eq!(
            schema.column_type(0),
            Some(ScalarType::Symbol),
            "First column should be Symbol"
        );
        assert_eq!(
            schema.column_type(1),
            Some(ScalarType::U64),
            "Second column should be U64"
        );

        // Now test lowering the rule with comparison
        lowerer.set_strata(vec![
            vec!["count_data".to_string()],
            vec!["big_count".to_string()],
        ]);
        lowerer.build_sccs(&program);

        let rule = &program.rules[1]; // big_count rule
        let result = lowerer.lower_rule(rule);
        assert!(
            result.is_ok(),
            "Lowering should succeed: {:?}",
            result.err()
        );

        // Check that the filter has the correct constant type
        fn find_compare_const(node: &RirNode) -> Option<&ConstValue> {
            match node {
                RirNode::Filter { predicate, input } => {
                    if let Expr::Compare { right, .. } = predicate {
                        if let Expr::Const(val) = right.as_ref() {
                            return Some(val);
                        }
                    }
                    find_compare_const(input)
                }
                RirNode::Project { input, .. } => find_compare_const(input),
                RirNode::Join { left, right, .. } => {
                    find_compare_const(left).or_else(|| find_compare_const(right))
                }
                _ => None,
            }
        }

        let node = result.unwrap();
        let const_val = find_compare_const(&node);
        assert!(const_val.is_some(), "Should find a constant in comparison");

        // The constant should be U64(3), not I64(3)
        match const_val.unwrap() {
            ConstValue::U64(v) => assert_eq!(*v, 3, "Value should be 3"),
            other => panic!("Expected U64(3), got {:?}", other),
        }
    }

    #[test]
    fn test_u64_comparison_with_aggregation() {
        use crate::ast::AggExpr;

        // Test aggregation + comparison case
        let mut program = Program::new();

        // pred reports_to(symbol, symbol).
        program.predicates.push(pred_decl(
            "reports_to",
            vec![ScalarType::Symbol, ScalarType::Symbol],
        ));

        // reports_to facts
        program.rules.push(Rule {
            head: Atom {
                predicate: "reports_to".to_string(),
                terms: vec![
                    Term::Symbol(xlog_core::symbol::intern("alice")),
                    Term::Symbol(xlog_core::symbol::intern("bob")),
                ],
            },
            body: vec![],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "reports_to".to_string(),
                terms: vec![
                    Term::Symbol(xlog_core::symbol::intern("carol")),
                    Term::Symbol(xlog_core::symbol::intern("bob")),
                ],
            },
            body: vec![],
        });

        // pred direct_count(symbol, u64).
        program.predicates.push(pred_decl(
            "direct_count",
            vec![ScalarType::Symbol, ScalarType::U64],
        ));

        // direct_count(Mgr, count(Emp)) :- reports_to(Emp, Mgr).
        program.rules.push(Rule {
            head: Atom {
                predicate: "direct_count".to_string(),
                terms: vec![
                    Term::Variable("Mgr".to_string()),
                    Term::Aggregate(AggExpr {
                        op: AggOp::Count,
                        variable: "Emp".to_string(),
                    }),
                ],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "reports_to".to_string(),
                terms: vec![
                    Term::Variable("Emp".to_string()),
                    Term::Variable("Mgr".to_string()),
                ],
            })],
        });

        // pred big_manager(symbol, u64).
        program.predicates.push(pred_decl(
            "big_manager",
            vec![ScalarType::Symbol, ScalarType::U64],
        ));

        // big_manager(Mgr, Count) :- direct_count(Mgr, Count), Count >= 2.
        program.rules.push(Rule {
            head: Atom {
                predicate: "big_manager".to_string(),
                terms: vec![
                    Term::Variable("Mgr".to_string()),
                    Term::Variable("Count".to_string()),
                ],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "direct_count".to_string(),
                    terms: vec![
                        Term::Variable("Mgr".to_string()),
                        Term::Variable("Count".to_string()),
                    ],
                }),
                BodyLiteral::Comparison(Comparison {
                    left: Term::Variable("Count".to_string()),
                    op: CompOp::Ge,
                    right: Term::Integer(2),
                }),
            ],
        });

        let mut lowerer = Lowerer::new();
        lowerer.infer_schemas(&program).unwrap();

        // Verify schema has correct types
        let schema = lowerer
            .schemas
            .get("direct_count")
            .expect("schema for direct_count");
        assert_eq!(
            schema.column_type(0),
            Some(ScalarType::Symbol),
            "First column should be Symbol"
        );
        assert_eq!(
            schema.column_type(1),
            Some(ScalarType::U64),
            "Second column should be U64"
        );

        lowerer.set_strata(vec![
            vec!["reports_to".to_string()],
            vec!["direct_count".to_string()],
            vec!["big_manager".to_string()],
        ]);
        lowerer.build_sccs(&program);

        // Lower the big_manager rule (index 3: after 2 facts + aggregation rule)
        let big_manager_rule = &program.rules[3];
        let result = lowerer.lower_rule(big_manager_rule);
        assert!(
            result.is_ok(),
            "Lowering should succeed: {:?}",
            result.err()
        );

        // Check that the filter has the correct constant type
        fn find_compare_const(node: &RirNode) -> Option<&ConstValue> {
            match node {
                RirNode::Filter { predicate, input } => {
                    if let Expr::Compare { right, .. } = predicate {
                        if let Expr::Const(val) = right.as_ref() {
                            return Some(val);
                        }
                    }
                    find_compare_const(input)
                }
                RirNode::Project { input, .. } => find_compare_const(input),
                RirNode::Join { left, right, .. } => {
                    find_compare_const(left).or_else(|| find_compare_const(right))
                }
                _ => None,
            }
        }

        let node = result.unwrap();
        let const_val = find_compare_const(&node);
        assert!(const_val.is_some(), "Should find a constant in comparison");

        // The constant should be U64(2), not I64(2)
        match const_val.unwrap() {
            ConstValue::U64(v) => assert_eq!(*v, 2, "Value should be 2"),
            other => panic!("Expected U64(2), got {:?}", other),
        }
    }

    #[test]
    fn declared_head_constant_is_projected_with_the_schema_type() {
        let program = crate::parse_program(
            r#"
            pred real(f64).
            seed().
            real(1) :- seed().
        "#,
        )
        .expect("parse typed head-constant fixture");
        let mut lowerer = Lowerer::new();
        lowerer
            .infer_schemas(&program)
            .expect("infer declared schemas");
        lowerer
            .validate_rule_types(&program)
            .expect("validate supported literal conversion");

        let node = lowerer
            .lower_rule(&program.rules[1])
            .expect("lower typed head constant");
        let RirNode::Project { columns, .. } = node else {
            panic!("constant rule head should lower to a projection");
        };
        assert_eq!(
            columns,
            vec![ProjectExpr::Computed(
                Expr::Const(ConstValue::F64(1.0)),
                ScalarType::F64,
            )]
        );
    }
}

#[cfg(test)]
mod cross_predicate_type_tests {
    use super::*;

    // Regression for the paper's deliberately ill-typed example: the head
    // declaration constrains bridge column 0 to symbol while the body
    // constrains the same variable to u32 through connected/node. The
    // mismatch must surface as a compilation error naming the predicate
    // and argument positions, never as a kernel schema dump.
    const ILL_TYPED_BRIDGE: &str = r#"
pred node(u32, symbol).
pred connected(u32, u32).
pred bridge(symbol, u32).

node(1, "alice").
connected(1, 2).

bridge(A, B) :- connected(A, B), node(A, _).

?- bridge(W, X).
"#;

    #[test]
    fn cross_predicate_schema_mismatch_is_a_compilation_error() {
        let program = crate::parse_program(ILL_TYPED_BRIDGE).expect("example parses");
        let mut lowerer = Lowerer::new();
        let err = lowerer
            .lower_program(&program)
            .expect_err("ill-typed program must not lower");
        let msg = err.to_string();
        assert!(
            matches!(err, xlog_core::XlogError::Compilation(_)),
            "expected a compilation error, got: {msg}"
        );
        assert!(
            msg.contains("bridge"),
            "must name the head predicate: {msg}"
        );
        assert!(
            msg.contains("connected") || msg.contains("node"),
            "must name the conflicting body predicate: {msg}"
        );
        assert!(msg.contains('A'), "must name the variable: {msg}");
        assert!(msg.contains("position 0"), "must name the position: {msg}");
    }

    #[test]
    fn well_typed_cross_predicate_rule_still_lowers() {
        let source = r#"
pred node(u32, symbol).
pred connected(u32, u32).
pred bridge(u32, u32).

node(1, "alice").
connected(1, 2).

bridge(A, B) :- connected(A, B), node(A, _).

?- bridge(W, X).
"#;
        let program = crate::parse_program(source).expect("example parses");
        let mut lowerer = Lowerer::new();
        lowerer
            .lower_program(&program)
            .expect("well-typed program lowers");
    }

    #[test]
    fn probabilistic_fact_schema_precedes_body_only_variable_defaults() {
        let source = r#"
0.5::possible(0).
observed(0).
accepted(X) :- possible(X), observed(X).
?- accepted(X).
"#;
        let program = crate::parse_program(source).expect("probabilistic join parses");
        let mut lowerer = Lowerer::new();
        lowerer
            .lower_program(&program)
            .expect("probabilistic facts provide body predicate schemas");
        assert_eq!(
            lowerer
                .schemas()
                .get("possible")
                .expect("possible schema")
                .column_type(0),
            Some(ScalarType::U32)
        );
    }

    #[test]
    fn annotated_disjunction_schema_precedes_body_only_variable_defaults() {
        let source = r#"
0.5::possible(0); 0.5::possible(1).
observed(0).
accepted(X) :- possible(X), observed(X).
?- accepted(X).
"#;
        let program = crate::parse_program(source).expect("annotated-disjunction join parses");
        let mut lowerer = Lowerer::new();
        lowerer
            .lower_program(&program)
            .expect("annotated-disjunction choices provide body predicate schemas");
        assert_eq!(
            lowerer
                .schemas()
                .get("possible")
                .expect("possible schema")
                .column_type(0),
            Some(ScalarType::U32)
        );
    }

    #[test]
    fn body_body_schema_conflict_is_a_compilation_error() {
        // The same variable drawing incompatible types from two body atoms
        // must be rejected even when the head is consistent with one side.
        let source = r#"
pred label(symbol).
pred count(u32).
pred out(u32).

label("x").
count(1).

out(A) :- count(A), label(A).

?- out(W).
"#;
        let program = crate::parse_program(source).expect("example parses");
        let mut lowerer = Lowerer::new();
        let err = lowerer
            .lower_program(&program)
            .expect_err("body-body conflict must not lower");
        assert!(
            matches!(err, xlog_core::XlogError::Compilation(_)),
            "expected a compilation error, got: {err}"
        );
    }
}
