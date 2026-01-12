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

use xlog_core::{AggOp as CoreAggOp, RelId, Result, ScalarType, Schema, XlogError};
use xlog_ir::{
    CompareOp, CompiledRule, ConstValue, ExecutionPlan, Expr, JoinType, PlanBuilder, ProjectExpr,
    RirMeta, RirNode, Scc, Stratum as IrStratum,
};

use crate::ast::{
    AggOp, ArithExpr, Atom, BodyLiteral, CompOp, Comparison, IsExpr, Program, Rule, Term,
};
use crate::stratify::{build_dependency_graph, find_sccs_for_lowering, DepType};

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
        }
    }

    /// Set the stratification result for ordering
    pub fn set_strata(&mut self, strata: Vec<Vec<String>>) {
        self.strata = strata;
    }

    /// Set cardinality hints (typically sourced from runtime statistics snapshots).
    ///
    /// These hints are used by lowering-time join ordering when available.
    pub fn set_cardinality_hints(&mut self, hints: HashMap<String, u64>) {
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

    /// Lower an entire program to an execution plan
    pub fn lower_program(&mut self, program: &Program) -> Result<ExecutionPlan> {
        // Infer schemas
        self.infer_schemas(program);
        self.infer_cardinalities(program);

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

        // Add facts as scan-only rules
        for fact in program.facts() {
            let pred = &fact.head.predicate;
            let scc_id = self.find_scc_for_predicate(pred);
            let rel_id = self.get_or_create_rel_id(pred);

            let body = RirNode::Scan { rel: rel_id };
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
        let schema = self
            .schemas
            .get(pred)
            .cloned()
            .unwrap_or_else(|| Schema::new(vec![]));
        RirMeta::with_schema(schema)
    }

    /// Lower a single rule to an RIR node
    fn lower_rule(&mut self, rule: &Rule) -> Result<RirNode> {
        // Split body literals.
        let (positive_atoms, negated_atoms, comparisons, is_exprs) =
            Self::split_body_literals(&rule.body);

        // Order positive atoms for join performance.
        let ordered_atoms = self.order_positive_atoms(&positive_atoms);

        // Build variable environment from the ordered atom layout.
        let mut var_env = VariableEnv::new();
        let mut current_col = 0;
        for atom in &ordered_atoms {
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

        // Lower the body with the chosen join order.
        let body_node = self.lower_body_parts(
            &ordered_atoms,
            &negated_atoms,
            &comparisons,
            &is_exprs,
            &mut var_env,
        )?;

        // Project to head variables
        let projection_cols = self.compute_projection(&rule.head, &var_env)?;

        if projection_cols.iter().enumerate().all(|(i, &c)| c == i)
            && projection_cols.len() == var_env.column_count()
        {
            // No projection needed if columns match exactly
            Ok(body_node)
        } else {
            // Convert to ProjectExpr::Column
            let proj_exprs: Vec<ProjectExpr> = projection_cols
                .into_iter()
                .map(ProjectExpr::Column)
                .collect();
            Ok(RirNode::Project {
                input: Box::new(body_node),
                columns: proj_exprs,
            })
        }
    }

    fn split_body_literals<'a>(
        body: &'a [BodyLiteral],
    ) -> (
        Vec<&'a Atom>,
        Vec<&'a Atom>,
        Vec<&'a Comparison>,
        Vec<&'a IsExpr>,
    ) {
        let mut positive_atoms: Vec<&Atom> = Vec::new();
        let mut negated_atoms: Vec<&Atom> = Vec::new();
        let mut comparisons: Vec<&Comparison> = Vec::new();
        let mut is_exprs: Vec<&IsExpr> = Vec::new();

        for lit in body {
            match lit {
                BodyLiteral::Positive(atom) => positive_atoms.push(atom),
                BodyLiteral::Negated(atom) => negated_atoms.push(atom),
                BodyLiteral::Comparison(cmp) => comparisons.push(cmp),
                BodyLiteral::IsExpr(is_expr) => is_exprs.push(is_expr),
            }
        }

        (positive_atoms, negated_atoms, comparisons, is_exprs)
    }

    fn atom_vars(atom: &Atom) -> std::collections::HashSet<String> {
        atom.terms
            .iter()
            .filter_map(|t| match t {
                Term::Variable(name) if name != "_" => Some(name.clone()),
                _ => None,
            })
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

    fn order_positive_atoms<'a>(&self, atoms: &[&'a Atom]) -> Vec<&'a Atom> {
        if atoms.len() <= 1 {
            return atoms.to_vec();
        }

        // Dynamic programming left-deep ordering for small join bodies; greedy fallback
        // keeps compilation time bounded for very large rules.
        const MAX_DP_ATOMS: usize = 10;
        if atoms.len() <= MAX_DP_ATOMS {
            return self.order_positive_atoms_cost_based(atoms);
        }

        self.order_positive_atoms_greedy(atoms)
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

    fn order_positive_atoms_cost_based<'a>(&self, atoms: &[&'a Atom]) -> Vec<&'a Atom> {
        let n = atoms.len();
        let size = 1usize << n;

        let atom_est: Vec<f64> = atoms.iter().map(|a| self.estimate_atom_rows(a)).collect();
        let atom_vars: Vec<HashSet<String>> = atoms.iter().map(|a| Self::atom_vars(a)).collect();

        let mut union_vars: Vec<HashSet<String>> = vec![HashSet::new(); size];
        for mask in 1..size {
            let bit_idx = (mask as u32).trailing_zeros() as usize;
            let prev = mask & !(1usize << bit_idx);
            let mut vars = union_vars[prev].clone();
            vars.extend(atom_vars[bit_idx].iter().cloned());
            union_vars[mask] = vars;
        }

        let mut best_total_cost: Vec<f64> = vec![f64::INFINITY; size];
        let mut best_card: Vec<f64> = vec![0.0; size];
        let mut best_order: Vec<Vec<usize>> = vec![Vec::new(); size];

        for i in 0..n {
            let mask = 1usize << i;
            best_total_cost[mask] = atom_est[i];
            best_card[mask] = atom_est[i];
            best_order[mask] = vec![i];
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
            let prev_total = best_total_cost[mask];
            if !prev_total.is_finite() {
                continue;
            }
            let prev_card = best_card[mask];
            let bound = &union_vars[mask];

            for next in 0..n {
                if (mask & (1usize << next)) != 0 {
                    continue;
                }

                let shared = bound.intersection(&atom_vars[next]).count();
                let mut selectivity = if shared == 0 {
                    1.0
                } else {
                    0.1_f64.powi(shared as i32)
                };
                if shared == 0 {
                    // Heavily penalize cartesian products in ordering decisions.
                    selectivity *= 1.0e6;
                }

                let next_card = (prev_card * atom_est[next] * selectivity).max(1.0);
                let next_total = prev_total + next_card;
                let next_mask = mask | (1usize << next);

                let should_replace = if next_total < best_total_cost[next_mask] {
                    true
                } else if (next_total - best_total_cost[next_mask]).abs() < 1e-9 {
                    let mut candidate = best_order[mask].clone();
                    candidate.push(next);
                    lex_lt(&candidate, &best_order[next_mask])
                } else {
                    false
                };

                if should_replace {
                    best_total_cost[next_mask] = next_total;
                    best_card[next_mask] = next_card;
                    let mut candidate = best_order[mask].clone();
                    candidate.push(next);
                    best_order[next_mask] = candidate;
                }
            }
        }

        let full_mask = size - 1;
        let order = &best_order[full_mask];
        if order.len() != n {
            return self.order_positive_atoms_greedy(atoms);
        }

        order.iter().map(|&idx| atoms[idx]).collect()
    }

    fn lower_body_parts(
        &mut self,
        positive_atoms: &[&Atom],
        negated_atoms: &[&Atom],
        comparisons: &[&Comparison],
        is_exprs: &[&IsExpr],
        var_env: &mut VariableEnv,
    ) -> Result<RirNode> {
        let mut result = self.build_join_tree(positive_atoms, var_env)?;

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

    /// Infer the result type of an arithmetic expression (strict same-type)
    pub fn infer_arith_type(&self, expr: &ArithExpr, var_env: &VariableEnv) -> Result<ScalarType> {
        match expr {
            ArithExpr::Variable(name) => var_env.get_type(name).ok_or_else(|| {
                XlogError::Compilation(format!("Unknown variable {} in arithmetic", name))
            }),
            ArithExpr::Integer(_) => Ok(ScalarType::I64),
            ArithExpr::Float(_) => Ok(ScalarType::F64),

            ArithExpr::Add(l, r)
            | ArithExpr::Sub(l, r)
            | ArithExpr::Mul(l, r)
            | ArithExpr::Div(l, r) => {
                let lt = self.infer_arith_type(l, var_env)?;
                let rt = self.infer_arith_type(r, var_env)?;

                if lt != rt {
                    return Err(XlogError::Compilation(format!(
                        "Type mismatch in arithmetic: {:?} vs {:?}. Use cast() for conversion.",
                        lt, rt
                    )));
                }

                if !Self::is_numeric_type(&lt) {
                    return Err(XlogError::Compilation(format!(
                        "Arithmetic requires numeric type, got {:?}",
                        lt
                    )));
                }

                Ok(lt)
            }

            ArithExpr::Mod(l, r) => {
                let lt = self.infer_arith_type(l, var_env)?;
                let rt = self.infer_arith_type(r, var_env)?;

                if lt != rt {
                    return Err(XlogError::Compilation(format!(
                        "Type mismatch in mod: {:?} vs {:?}",
                        lt, rt
                    )));
                }

                if matches!(lt, ScalarType::F32 | ScalarType::F64) {
                    return Err(XlogError::Compilation(
                        "Modulo (%) not supported for floating point".into(),
                    ));
                }

                Ok(lt)
            }

            ArithExpr::Abs(inner) => {
                let t = self.infer_arith_type(inner, var_env)?;
                if !Self::is_numeric_type(&t) {
                    return Err(XlogError::Compilation(format!(
                        "abs requires numeric type, got {:?}",
                        t
                    )));
                }
                Ok(t)
            }

            ArithExpr::Min(l, r) | ArithExpr::Max(l, r) => {
                let lt = self.infer_arith_type(l, var_env)?;
                let rt = self.infer_arith_type(r, var_env)?;

                if lt != rt {
                    return Err(XlogError::Compilation(format!(
                        "Type mismatch in min/max: {:?} vs {:?}",
                        lt, rt
                    )));
                }

                if !Self::is_numeric_type(&lt) {
                    return Err(XlogError::Compilation(format!(
                        "min/max requires numeric type, got {:?}",
                        lt
                    )));
                }

                Ok(lt)
            }

            ArithExpr::Pow(base, exp) => {
                let base_t = self.infer_arith_type(base, var_env)?;
                let exp_t = self.infer_arith_type(exp, var_env)?;

                if !Self::is_numeric_type(&base_t) || !Self::is_numeric_type(&exp_t) {
                    return Err(XlogError::Compilation(format!(
                        "pow requires numeric operands, got {:?} and {:?}",
                        base_t, exp_t
                    )));
                }

                // pow always returns f64 (standard math behavior)
                Ok(ScalarType::F64)
            }

            ArithExpr::Cast(_, target) => Ok(*target),
        }
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
        match arith {
            ArithExpr::Variable(name) => {
                let col = var_env.get_column(name).ok_or_else(|| {
                    XlogError::Compilation(format!(
                        "Variable {} not bound before use in arithmetic",
                        name
                    ))
                })?;
                Ok(Expr::Column(col))
            }
            ArithExpr::Integer(i) => Ok(Expr::Const(ConstValue::I64(*i))),
            ArithExpr::Float(f) => Ok(Expr::Const(ConstValue::F64(*f))),

            ArithExpr::Add(l, r) => Ok(Expr::Add(
                Box::new(self.arith_to_expr(l, var_env)?),
                Box::new(self.arith_to_expr(r, var_env)?),
            )),
            ArithExpr::Sub(l, r) => Ok(Expr::Sub(
                Box::new(self.arith_to_expr(l, var_env)?),
                Box::new(self.arith_to_expr(r, var_env)?),
            )),
            ArithExpr::Mul(l, r) => Ok(Expr::Mul(
                Box::new(self.arith_to_expr(l, var_env)?),
                Box::new(self.arith_to_expr(r, var_env)?),
            )),
            ArithExpr::Div(l, r) => Ok(Expr::Div(
                Box::new(self.arith_to_expr(l, var_env)?),
                Box::new(self.arith_to_expr(r, var_env)?),
            )),
            ArithExpr::Mod(l, r) => Ok(Expr::Mod(
                Box::new(self.arith_to_expr(l, var_env)?),
                Box::new(self.arith_to_expr(r, var_env)?),
            )),

            ArithExpr::Abs(e) => Ok(Expr::Abs(Box::new(self.arith_to_expr(e, var_env)?))),
            ArithExpr::Min(l, r) => Ok(Expr::Min(
                Box::new(self.arith_to_expr(l, var_env)?),
                Box::new(self.arith_to_expr(r, var_env)?),
            )),
            ArithExpr::Max(l, r) => Ok(Expr::Max(
                Box::new(self.arith_to_expr(l, var_env)?),
                Box::new(self.arith_to_expr(r, var_env)?),
            )),
            ArithExpr::Pow(l, r) => Ok(Expr::Pow(
                Box::new(self.arith_to_expr(l, var_env)?),
                Box::new(self.arith_to_expr(r, var_env)?),
            )),
            ArithExpr::Cast(e, t) => Ok(Expr::Cast(Box::new(self.arith_to_expr(e, var_env)?), *t)),
        }
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
pub struct VariableEnv {
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
                assert!(matches!(*left, RirNode::Scan { rel } if rel == small_id));
                assert!(matches!(*right, RirNode::Scan { rel } if rel == big_id));
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
        lowerer.set_strata(vec![vec!["edge".to_string()], vec!["reach".to_string()]]);

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
        assert_eq!(
            infer_term_type(&Term::Variable("X".to_string())),
            ScalarType::U64
        );
        assert_eq!(infer_term_type(&Term::Integer(42)), ScalarType::U32);
        assert_eq!(infer_term_type(&Term::Integer(i64::MAX)), ScalarType::I64);
        assert_eq!(infer_term_type(&Term::Float(3.14)), ScalarType::F64);
        assert_eq!(
            infer_term_type(&Term::Symbol("foo".to_string())),
            ScalarType::Symbol
        );
    }

    #[test]
    fn test_convert_agg_op() {
        assert_eq!(convert_agg_op(&AggOp::Count), CoreAggOp::Count);
        assert_eq!(convert_agg_op(&AggOp::Sum), CoreAggOp::Sum);
        assert_eq!(convert_agg_op(&AggOp::Min), CoreAggOp::Min);
        assert_eq!(convert_agg_op(&AggOp::Max), CoreAggOp::Max);
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
}
