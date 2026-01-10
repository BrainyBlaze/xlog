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

use std::collections::HashMap;

use xlog_core::{RelId, Result, Schema, ScalarType, XlogError, AggOp as CoreAggOp};
use xlog_ir::{
    CompiledRule, ExecutionPlan, PlanBuilder, RirMeta, RirNode,
    Scc, Stratum as IrStratum, CompareOp, ConstValue, Expr, JoinType,
};

use crate::ast::{
    AggOp, Atom, BodyLiteral, Comparison, CompOp, Program, Rule, Term,
};
use crate::stratify::{build_dependency_graph, find_sccs_for_lowering, DepType};

/// Lowerer transforms AST programs into RIR execution plans.
pub struct Lowerer {
    /// Inferred or declared schemas for each predicate
    schemas: HashMap<String, Schema>,
    /// Stratification result (predicates grouped by strata)
    strata: Vec<Vec<String>>,
    /// Next available relation ID
    next_rel_id: u32,
    /// Mapping from predicate names to relation IDs
    rel_ids: HashMap<String, RelId>,
    /// SCCs for the program (from stratification)
    sccs: Vec<Scc>,
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
            next_rel_id: 0,
            rel_ids: HashMap::new(),
            sccs: Vec::new(),
        }
    }

    /// Set the stratification result for ordering
    pub fn set_strata(&mut self, strata: Vec<Vec<String>>) {
        self.strata = strata;
    }

    /// Get the mapping from predicate names to relation IDs
    pub fn rel_ids(&self) -> &HashMap<String, RelId> {
        &self.rel_ids
    }

    /// Get the inferred schemas for predicates
    pub fn schemas(&self) -> &HashMap<String, Schema> {
        &self.schemas
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

    /// Infer schemas from facts and predicate declarations
    fn infer_schemas(&mut self, program: &Program) {
        // First, use explicit predicate declarations
        for pred_decl in &program.predicates {
            let columns: Vec<(String, ScalarType)> = pred_decl
                .types
                .iter()
                .enumerate()
                .map(|(i, ty)| (format!("c{}", i), *ty))
                .collect();
            self.schemas.insert(pred_decl.name.clone(), Schema::new(columns));
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
                        let ty = infer_term_type(term);
                        (format!("c{}", i), ty)
                    })
                    .collect();
                self.schemas.insert(pred.clone(), Schema::new(columns));
            }
        }

        // Finally, infer from rule heads if we still don't have a schema
        for rule in &program.rules {
            let pred = &rule.head.predicate;
            if !self.schemas.contains_key(pred) {
                // Use default U64 type for variables
                let columns: Vec<(String, ScalarType)> = rule
                    .head
                    .terms
                    .iter()
                    .enumerate()
                    .map(|(i, term)| {
                        let ty = infer_term_type(term);
                        (format!("c{}", i), ty)
                    })
                    .collect();
                self.schemas.insert(pred.clone(), Schema::new(columns));
            }
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
                graph.outgoing(pred).iter().any(|e| {
                    e.to == *pred && e.dep_type == DepType::Positive
                })
            };

            self.sccs.push(Scc {
                id: id as u32,
                predicates: predicates.clone(),
                is_recursive,
            });
        }
    }

    /// Lower an entire program to an execution plan
    pub fn lower_program(&mut self, program: &Program) -> Result<ExecutionPlan> {
        // Infer schemas
        self.infer_schemas(program);

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
            let scc_ids: Vec<u32> = self.sccs.iter()
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

        // Add facts as scan-only rules
        for fact in program.facts() {
            let pred = &fact.head.predicate;
            let scc_id = self.find_scc_for_predicate(pred);
            let rel_id = self.get_or_create_rel_id(pred);

            let body = RirNode::Scan { rel: rel_id };
            let meta = self.create_meta_for_predicate(pred);

            builder.add_rule(scc_id, CompiledRule {
                head: pred.clone(),
                body,
                meta,
            });
        }

        // Lower proper rules
        for (pred, rules) in &rules_by_pred {
            let scc_id = self.find_scc_for_predicate(pred);

            for rule in rules {
                let body = self.lower_rule(rule)?;
                let meta = self.create_meta_for_predicate(pred);

                builder.add_rule(scc_id, CompiledRule {
                    head: pred.clone(),
                    body,
                    meta,
                });
            }
        }

        Ok(builder.build())
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
        let schema = self.schemas.get(pred).cloned().unwrap_or_else(|| {
            Schema::new(vec![])
        });
        RirMeta::with_schema(schema)
    }

    /// Lower a single rule to an RIR node
    fn lower_rule(&mut self, rule: &Rule) -> Result<RirNode> {
        // Build variable environment: track which variables are at which positions
        let mut var_env = VariableEnv::new();

        // First pass: collect all variable positions from body atoms
        let mut current_col = 0;
        for lit in &rule.body {
            if let Some(atom) = lit.atom() {
                if lit.is_positive() {
                    for (i, term) in atom.terms.iter().enumerate() {
                        if let Term::Variable(name) = term {
                            var_env.add_occurrence(name, atom.predicate.clone(), i, current_col + i);
                        }
                    }
                    current_col += atom.terms.len();
                }
            }
        }

        // Lower the body
        let body_node = self.lower_body(&rule.body, &mut var_env)?;

        // Project to head variables
        let projection_cols = self.compute_projection(&rule.head, &var_env)?;

        if projection_cols.iter().enumerate().all(|(i, &c)| c == i)
            && projection_cols.len() == var_env.total_cols {
            // No projection needed if columns match exactly
            Ok(body_node)
        } else {
            Ok(RirNode::Project {
                input: Box::new(body_node),
                columns: projection_cols,
            })
        }
    }

    /// Lower the body literals to an RIR node
    fn lower_body(&mut self, body: &[BodyLiteral], var_env: &mut VariableEnv) -> Result<RirNode> {
        // Separate positive atoms, negated atoms, and comparisons
        let mut positive_atoms: Vec<&Atom> = Vec::new();
        let mut negated_atoms: Vec<&Atom> = Vec::new();
        let mut comparisons: Vec<&Comparison> = Vec::new();

        for lit in body {
            match lit {
                BodyLiteral::Positive(atom) => positive_atoms.push(atom),
                BodyLiteral::Negated(atom) => negated_atoms.push(atom),
                BodyLiteral::Comparison(cmp) => comparisons.push(cmp),
            }
        }

        // Build join tree from positive atoms
        let mut result = self.build_join_tree(&positive_atoms, var_env)?;

        // Apply comparisons as filters
        for cmp in comparisons {
            result = self.apply_comparison(result, cmp, var_env)?;
        }

        // Handle negated atoms via Diff
        for neg_atom in negated_atoms {
            result = self.apply_negation(result, neg_atom, var_env)?;
        }

        Ok(result)
    }

    /// Build a left-deep join tree from positive atoms
    fn build_join_tree(&mut self, atoms: &[&Atom], var_env: &mut VariableEnv) -> Result<RirNode> {
        if atoms.is_empty() {
            return Err(XlogError::Compilation(
                "Empty rule body".to_string(),
            ));
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
            let (left_keys, right_keys) = self.compute_join_keys(
                &result_vars,
                atom,
                result_width,
            );

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

        for (i, term) in atom.terms.iter().enumerate() {
            if let Some(const_val) = term_to_const_value(term) {
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
        let left_expr = self.term_to_expr(&cmp.left, var_env)?;
        let right_expr = self.term_to_expr(&cmp.right, var_env)?;

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
            Term::Symbol(s) => Ok(Expr::Const(ConstValue::Symbol(s.clone()))),
            Term::Aggregate(_) => Err(XlogError::Compilation(
                "Aggregates not allowed in comparisons".to_string(),
            )),
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
            // No shared variables - this is an existence check
            // If the negated relation is non-empty, result is empty
            // This is a special case we handle with anti-join
            Ok(RirNode::Diff {
                left: Box::new(input),
                right: Box::new(neg_filtered),
            })
        } else {
            // Project the negated atom to only the shared variable columns
            let neg_projected = if neg_cols.len() < neg_atom.terms.len() {
                RirNode::Project {
                    input: Box::new(neg_filtered),
                    columns: neg_cols.clone(),
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
            let input_projected = RirNode::Project {
                input: Box::new(input.clone()),
                columns: input_cols.clone(),
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

    /// Compute the projection columns to match head variables
    fn compute_projection(&self, head: &Atom, var_env: &VariableEnv) -> Result<Vec<usize>> {
        let mut cols = Vec::new();

        for term in &head.terms {
            match term {
                Term::Variable(name) => {
                    if let Some(col) = var_env.get_column(name) {
                        cols.push(col);
                    } else {
                        return Err(XlogError::UnsafeVariable(name.clone()));
                    }
                }
                Term::Anonymous => {
                    // Anonymous wildcard in head is a semantic error
                    return Err(XlogError::Compilation(
                        "Anonymous wildcard '_' not allowed in rule head".to_string(),
                    ));
                }
                Term::Aggregate(agg) => {
                    // For aggregates, we need the column of the aggregated variable
                    if let Some(col) = var_env.get_column(&agg.variable) {
                        cols.push(col);
                    } else {
                        return Err(XlogError::UnsafeVariable(agg.variable.clone()));
                    }
                }
                // Constants in head are handled differently (they're projected out)
                _ => {
                    // For now, skip constants in head
                    // A more complete implementation would validate them
                }
            }
        }

        Ok(cols)
    }
}

/// Track variable occurrences and column positions
struct VariableEnv {
    /// Maps variable name to list of (predicate, position in atom, global column)
    occurrences: HashMap<String, Vec<(String, usize, usize)>>,
    /// Total columns in current result
    total_cols: usize,
}

impl VariableEnv {
    fn new() -> Self {
        Self {
            occurrences: HashMap::new(),
            total_cols: 0,
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
}

/// Infer the type of a term
fn infer_term_type(term: &Term) -> ScalarType {
    match term {
        Term::Variable(_) | Term::Anonymous => ScalarType::U64, // Default for variables
        Term::Integer(i) => {
            if *i >= 0 && *i <= u32::MAX as i64 {
                ScalarType::U32
            } else {
                ScalarType::I64
            }
        }
        Term::Float(_) => ScalarType::F64,
        Term::String(_) | Term::Symbol(_) => ScalarType::Symbol,
        Term::Aggregate(_) => ScalarType::U64, // Aggregates typically produce integers
    }
}

/// Convert a term to a constant value (if it is a constant)
fn term_to_const_value(term: &Term) -> Option<ConstValue> {
    match term {
        Term::Integer(i) => Some(ConstValue::I64(*i)),
        Term::Float(f) => Some(ConstValue::F64(*f)),
        Term::String(s) => Some(ConstValue::Symbol(s.clone())),
        Term::Symbol(s) => Some(ConstValue::Symbol(s.clone())),
        Term::Variable(_) | Term::Anonymous | Term::Aggregate(_) => None,
    }
}

/// Convert AST AggOp to core AggOp
#[allow(dead_code)]
fn convert_agg_op(op: &AggOp) -> CoreAggOp {
    match op {
        AggOp::Count => CoreAggOp::Count,
        AggOp::Sum => CoreAggOp::Sum,
        AggOp::Min => CoreAggOp::Min,
        AggOp::Max => CoreAggOp::Max,
    }
}

// Export the find_sccs_for_lowering function from stratify
// We need to add this to the stratify module

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    /// Helper to create a simple edge atom
    fn edge_atom(x: &str, y: &str) -> Atom {
        Atom {
            predicate: "edge".to_string(),
            terms: vec![
                Term::Variable(x.to_string()),
                Term::Variable(y.to_string()),
            ],
        }
    }

    /// Helper to create a reach atom
    fn reach_atom(x: &str, y: &str) -> Atom {
        Atom {
            predicate: "reach".to_string(),
            terms: vec![
                Term::Variable(x.to_string()),
                Term::Variable(y.to_string()),
            ],
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
        lowerer.infer_schemas(&program);

        assert!(lowerer.schemas.contains_key("edge"));
        let schema = lowerer.schemas.get("edge").unwrap();
        assert_eq!(schema.arity(), 2);
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
            assert_eq!(columns, vec![0, 3]); // X from reach, Z from edge
            assert!(matches!(*input, RirNode::Join { .. }));
            if let RirNode::Join { left_keys, right_keys, .. } = *input {
                assert_eq!(left_keys, vec![1]); // Y in reach (position 1)
                assert_eq!(right_keys, vec![0]); // Y in edge (position 0)
            }
        } else {
            panic!("Expected Project node");
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
                RirNode::Join { join_type: JoinType::Semi, .. } => true,
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
                RirNode::Join { left, right, .. } => contains_filter(left) || contains_filter(right),
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
                RirNode::Filter { predicate, .. } => {
                    if let Expr::Compare { right, .. } = predicate {
                        matches!(**right, Expr::Const(_))
                    } else {
                        false
                    }
                }
                RirNode::Project { input, .. } => has_const_filter(input),
                _ => false,
            }
        }
        assert!(has_const_filter(&node));
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
        lowerer.set_strata(vec![
            vec!["edge".to_string()],
            vec!["reach".to_string()],
        ]);

        let result = lowerer.lower_program(&program);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(!plan.sccs.is_empty());
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
    fn test_infer_term_type() {
        assert_eq!(infer_term_type(&Term::Variable("X".to_string())), ScalarType::U64);
        assert_eq!(infer_term_type(&Term::Integer(42)), ScalarType::U32);
        assert_eq!(infer_term_type(&Term::Integer(i64::MAX)), ScalarType::I64);
        assert_eq!(infer_term_type(&Term::Float(3.14)), ScalarType::F64);
        assert_eq!(infer_term_type(&Term::Symbol("foo".to_string())), ScalarType::Symbol);
    }

    #[test]
    fn test_convert_agg_op() {
        assert_eq!(convert_agg_op(&AggOp::Count), CoreAggOp::Count);
        assert_eq!(convert_agg_op(&AggOp::Sum), CoreAggOp::Sum);
        assert_eq!(convert_agg_op(&AggOp::Min), CoreAggOp::Min);
        assert_eq!(convert_agg_op(&AggOp::Max), CoreAggOp::Max);
    }
}
