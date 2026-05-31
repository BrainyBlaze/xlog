//! Result aggregation, statistics, and the **CPU oracle/debug** evaluation path.
//!
//! The statistics helpers (`binomial_estimate`, `normal_quantile`, …) are used by
//! both the GPU-native and host paths. The host relation-store evaluation
//! (`evaluate_program_inplace`, `evidence_satisfied`, `atom_holds`, …) backs only
//! [`crate::mc::McProgram::evaluate_cpu`] — a seed-matched CPU oracle for
//! validating GPU device counts. It is never part of the GPU-native hot loop and
//! never zero-host/GPU-native release evidence.

#[cfg(feature = "host-io")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "host-io")]
use std::hash::{Hash, Hasher};

#[cfg(feature = "host-io")]
use xlog_core::{Result, XlogError};
#[cfg(feature = "host-io")]
use xlog_logic::ast::{AggExpr, AggOp, Atom, BodyLiteral, Rule, Term};
#[cfg(feature = "host-io")]
use xlog_logic::stratify::{build_dependency_graph, find_sccs_for_lowering};

#[cfg(feature = "host-io")]
use crate::provenance::{
    eval_arith_expr, eval_comparison, unify_atom, value_from_term, GroundAtom, Value,
};

#[cfg(feature = "host-io")]
use super::{EvalStats, McProgram, Relation, SccKind, SccPlan};
#[cfg(feature = "host-io")]
use xlog_logic::ast::Program;

#[cfg(feature = "host-io")]
use crate::aggregates::AggState;
#[cfg(feature = "host-io")]
fn v085_mc_result_term_error(context: &str, kind: &str) -> XlogError {
    XlogError::Compilation(format!(
        "v0.8.5 term form '{}' is parsed but not supported in MC result {} before its G085 implementation node",
        kind, context
    ))
}

#[cfg(feature = "host-io")]
impl McProgram {
    pub(super) fn apply_sample_facts(
        &self,
        store: &mut HashMap<String, Relation>,
        bits: &[u8],
    ) -> Result<()> {
        for pf in &self.prob_facts {
            if bits.get(pf.var_idx).copied().unwrap_or(0) == 0 {
                continue;
            }
            store
                .entry(pf.atom.predicate.clone())
                .or_default()
                .insert_tuple(pf.atom.args.clone());
        }

        for ad in &self.annotated_disjunctions {
            let mut selected: Option<usize> = None;
            for (idx, &var_idx) in ad.decision_vars.iter().enumerate() {
                if bits.get(var_idx).copied().unwrap_or(0) != 0 {
                    selected = Some(idx);
                    break;
                }
            }

            let outcome = match selected {
                Some(i) => Some(i),
                None => {
                    if ad.has_none {
                        None
                    } else {
                        Some(ad.choices.len().saturating_sub(1))
                    }
                }
            };

            if let Some(outcome_idx) = outcome {
                let atom = ad.choices.get(outcome_idx).ok_or_else(|| {
                    XlogError::Compilation(
                        "Annotated disjunction outcome index out of range".to_string(),
                    )
                })?;
                store
                    .entry(atom.predicate.clone())
                    .or_default()
                    .insert_tuple(atom.args.clone());
            }
        }

        Ok(())
    }
}

#[cfg(feature = "host-io")]
pub(super) fn build_scc_plans(program: &Program) -> Result<Vec<SccPlan>> {
    let graph = build_dependency_graph(program);
    let sccs = find_sccs_for_lowering(&graph);

    let mut rules_by_head: HashMap<String, Vec<Rule>> = HashMap::new();
    for rule in program.proper_rules() {
        rules_by_head
            .entry(rule.head.predicate.clone())
            .or_default()
            .push(rule.clone());
    }

    let mut plans: Vec<SccPlan> = Vec::new();
    for scc in sccs {
        let mut scc_rules: Vec<Rule> = Vec::new();
        for pred in &scc {
            if let Some(rules) = rules_by_head.get(pred) {
                scc_rules.extend(rules.iter().cloned());
            }
        }
        if scc_rules.is_empty() {
            continue;
        }

        let predicate_set: HashSet<String> = scc.iter().cloned().collect();

        let nonmonotone = scc_rules.iter().any(|rule| {
            rule.body.iter().any(|lit| match lit {
                BodyLiteral::Negated(atom) => predicate_set.contains(&atom.predicate),
                BodyLiteral::Positive(atom) if rule.has_aggregation() => {
                    predicate_set.contains(&atom.predicate)
                }
                _ => false,
            })
        });

        let kind = if nonmonotone {
            SccKind::NonMonotone
        } else if is_recursive_scc(&scc, &scc_rules) {
            SccKind::MonotoneRecursive
        } else {
            SccKind::MonotoneNonRecursive
        };

        plans.push(SccPlan {
            predicates: scc,
            rules: scc_rules,
            kind,
        });
    }

    Ok(plans)
}

#[cfg(feature = "host-io")]
fn is_recursive_scc(scc: &[String], rules: &[Rule]) -> bool {
    if scc.len() > 1 {
        return true;
    }
    let Some(only) = scc.first() else {
        return false;
    };
    for rule in rules {
        for lit in &rule.body {
            if let BodyLiteral::Positive(atom) = lit {
                if &atom.predicate == only {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(feature = "host-io")]
pub(super) fn evaluate_program_inplace(
    scc_plans: &[SccPlan],
    store: &mut HashMap<String, Relation>,
    max_nonmonotone_iterations: usize,
) -> Result<EvalStats> {
    let mut stats = EvalStats::default();

    for plan in scc_plans {
        match plan.kind {
            SccKind::MonotoneNonRecursive => {
                for rule in &plan.rules {
                    let derived = eval_rule(rule, store, &HashMap::new(), None)?;
                    let rel = store.entry(rule.head.predicate.clone()).or_default();
                    for tuple in derived {
                        rel.insert_tuple(tuple);
                    }
                }
            }
            SccKind::MonotoneRecursive => {
                eval_monotone_recursive_scc(&plan.predicates, &plan.rules, store)?;
            }
            SccKind::NonMonotone => {
                stats.nonmonotone_sccs += 1;
                let (cycle, hit_limit) = eval_nonmonotone_scc(
                    &plan.predicates,
                    &plan.rules,
                    store,
                    max_nonmonotone_iterations,
                )?;
                if cycle {
                    stats.nonmonotone_cycles += 1;
                }
                if hit_limit {
                    stats.nonmonotone_iteration_limit_hits += 1;
                }
            }
        }
    }

    Ok(stats)
}

#[cfg(feature = "host-io")]
fn eval_monotone_recursive_scc(
    scc: &[String],
    rules: &[Rule],
    store: &mut HashMap<String, Relation>,
) -> Result<()> {
    const MAX_ITERS: usize = 1024;

    let scc_set: HashSet<&str> = scc.iter().map(|s| s.as_str()).collect();

    let mut full: HashMap<String, Relation> = HashMap::new();
    for pred in scc {
        let rel = store.get(pred).cloned().unwrap_or_default();
        full.insert(pred.clone(), rel);
    }

    let mut delta: HashMap<String, Relation> = HashMap::new();
    for rule in rules {
        let derived = eval_rule(rule, store, &full, None)?;
        if derived.is_empty() {
            continue;
        }
        let head = rule.head.predicate.clone();
        let delta_rel = delta.entry(head.clone()).or_default();
        let full_rel = full.entry(head).or_default();
        for tuple in derived {
            if full_rel.tuples.insert(tuple.clone()) {
                delta_rel.insert_tuple(tuple);
            }
        }
    }

    let mut reached_fixpoint = false;
    for _ in 0..MAX_ITERS {
        let any_delta = delta.values().any(|r| !r.is_empty());
        if !any_delta {
            reached_fixpoint = true;
            break;
        }

        let full_prev = full.clone();
        let delta_prev = delta.clone();
        delta.clear();

        for rule in rules {
            let body_indices: Vec<usize> = rule
                .body
                .iter()
                .enumerate()
                .filter_map(|(i, lit)| match lit {
                    BodyLiteral::Positive(atom) if scc_set.contains(atom.predicate.as_str()) => {
                        let non_empty = delta_prev
                            .get(&atom.predicate)
                            .map(|r| !r.is_empty())
                            .unwrap_or(false);
                        non_empty.then_some(i)
                    }
                    _ => None,
                })
                .collect();
            if body_indices.is_empty() {
                continue;
            }

            let mut derived_all: HashSet<Vec<Value>> = HashSet::new();
            for idx in body_indices {
                let derived = eval_rule(rule, store, &full_prev, Some((idx, &delta_prev)))?;
                derived_all.extend(derived);
            }
            if derived_all.is_empty() {
                continue;
            }

            let head = rule.head.predicate.clone();
            let delta_rel = delta.entry(head.clone()).or_default();
            let full_rel = full.entry(head).or_default();
            for tuple in derived_all {
                if full_rel.tuples.insert(tuple.clone()) {
                    delta_rel.insert_tuple(tuple);
                }
            }
        }
    }

    if !reached_fixpoint {
        return Err(XlogError::Execution(format!(
            "Monotone SCC fixpoint iteration limit ({}) exceeded for SCC {:?}",
            MAX_ITERS, scc
        )));
    }

    for (pred, rel) in full {
        store.insert(pred, rel);
    }
    Ok(())
}

#[cfg(feature = "host-io")]
fn eval_nonmonotone_scc(
    scc: &[String],
    rules: &[Rule],
    store: &mut HashMap<String, Relation>,
    max_iters: usize,
) -> Result<(bool, bool)> {
    let mut base: HashMap<String, Relation> = HashMap::new();
    for pred in scc {
        base.insert(pred.clone(), store.get(pred).cloned().unwrap_or_default());
    }

    let mut history: Vec<HashMap<String, Relation>> = Vec::new();
    history.push(base.clone());

    let mut seen: HashMap<u64, Vec<usize>> = HashMap::new();
    seen.entry(hash_scc_state(&history[0])).or_default().push(0);

    for _iter in 0..max_iters {
        let current = history.last().expect("history non-empty").clone();

        let mut next: HashMap<String, Relation> = base.clone();

        for rule in rules {
            let derived = eval_rule(rule, store, &current, None)?;
            let rel = next.entry(rule.head.predicate.clone()).or_default();
            for tuple in derived {
                rel.insert_tuple(tuple);
            }
        }

        if next == current {
            for (pred, rel) in next {
                store.insert(pred, rel);
            }
            return Ok((false, false));
        }

        let h = hash_scc_state(&next);
        if let Some(candidates) = seen.get(&h) {
            for &idx in candidates {
                if history.get(idx) == Some(&next) {
                    // Cycle detected: history[idx..] repeats.
                    let cycle_states = &history[idx..];
                    let final_state = intersect_states(cycle_states);
                    for (pred, rel) in final_state {
                        store.insert(pred, rel);
                    }
                    return Ok((true, false));
                }
            }
        }

        let next_index = history.len();
        history.push(next);
        seen.entry(h).or_default().push(next_index);
    }

    // Iteration budget exhausted: fall back to a conservative invariant set.
    let final_state = intersect_states(&history);
    for (pred, rel) in final_state {
        store.insert(pred, rel);
    }
    Ok((false, true))
}

#[cfg(feature = "host-io")]
fn intersect_states(states: &[HashMap<String, Relation>]) -> HashMap<String, Relation> {
    let mut out: HashMap<String, Relation> = HashMap::new();
    let Some(first) = states.first() else {
        return out;
    };

    for (pred, rel0) in first {
        let mut intersection: HashSet<Vec<Value>> = rel0.tuples.clone();
        for state in &states[1..] {
            if let Some(rel) = state.get(pred) {
                intersection.retain(|t| rel.tuples.contains(t));
            } else {
                intersection.clear();
                break;
            }
        }
        out.insert(
            pred.clone(),
            Relation {
                tuples: intersection,
            },
        );
    }

    out
}

#[cfg(feature = "host-io")]
fn hash_scc_state(state: &HashMap<String, Relation>) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();

    let mut preds: Vec<&String> = state.keys().collect();
    preds.sort();
    for pred in preds {
        pred.hash(&mut hasher);
        if let Some(rel) = state.get(pred) {
            let mut tuple_hashes: Vec<u64> = Vec::with_capacity(rel.tuples.len());
            for tuple in &rel.tuples {
                let mut th = DefaultHasher::new();
                Hash::hash(tuple, &mut th);
                tuple_hashes.push(th.finish());
            }
            tuple_hashes.sort_unstable();
            tuple_hashes.hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Evaluate a single rule and produce derived head tuples.
///
/// `full_scc` is a per-SCC snapshot (used for SCC predicates), and `delta_scc` optionally provides
/// a delta relation for a specific body literal index (semi-naive).
#[cfg(feature = "host-io")]
fn eval_rule(
    rule: &Rule,
    global: &HashMap<String, Relation>,
    full_scc: &HashMap<String, Relation>,
    delta_scc: Option<(usize, &HashMap<String, Relation>)>,
) -> Result<Vec<Vec<Value>>> {
    let mut states: Vec<HashMap<String, Value>> = vec![HashMap::new()];

    for (idx, lit) in rule.body.iter().enumerate() {
        let mut next_states: Vec<HashMap<String, Value>> = Vec::new();
        match lit {
            BodyLiteral::Positive(atom) => {
                let rel = select_relation(atom, idx, global, full_scc, delta_scc)?;
                for binding in states {
                    for tuple in &rel.tuples {
                        let mut binding2 = binding.clone();
                        if unify_atom(atom, tuple, &mut binding2)? {
                            next_states.push(binding2);
                        }
                    }
                }
            }
            BodyLiteral::Negated(atom) => {
                let rel = select_relation(atom, idx, global, full_scc, delta_scc)?;
                for binding in states {
                    if negated_atom_holds(atom, rel, &binding)? {
                        next_states.push(binding);
                    }
                }
            }
            BodyLiteral::Epistemic(lit) => {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "host MC evaluation".to_string(),
                    context: format!("{:?} {}({})", lit.op, lit.atom.predicate, lit.atom.arity()),
                });
            }
            BodyLiteral::Comparison(cmp) => {
                for binding in states {
                    if eval_comparison(cmp.op, &cmp.left, &cmp.right, &binding)? {
                        next_states.push(binding);
                    }
                }
            }
            BodyLiteral::IsExpr(is_expr) => {
                for mut binding in states {
                    if binding.contains_key(&is_expr.target) {
                        return Err(XlogError::Compilation(format!(
                            "Is-expression target {} is already bound",
                            is_expr.target
                        )));
                    }
                    let v = eval_arith_expr(&is_expr.expr, &binding)?;
                    binding.insert(is_expr.target.clone(), v);
                    next_states.push(binding);
                }
            }
            BodyLiteral::Univ(_) => {
                return Err(XlogError::Compilation(
                    "v0.8.5 meta error: univ literal was not normalized before MC result materialization"
                        .to_string(),
                ));
            }
        }
        states = next_states;
        if states.is_empty() {
            break;
        }
    }

    if rule.has_aggregation() {
        eval_aggregate_head(&rule.head, states)
    } else {
        let mut out: HashSet<Vec<Value>> = HashSet::new();
        for binding in states {
            let head_tuple = materialize_head_non_aggregate(&rule.head, &binding)?;
            out.insert(head_tuple);
        }
        Ok(out.into_iter().collect())
    }
}

#[cfg(feature = "host-io")]
fn select_relation<'a>(
    atom: &Atom,
    body_index: usize,
    global: &'a HashMap<String, Relation>,
    full_scc: &'a HashMap<String, Relation>,
    delta_scc: Option<(usize, &'a HashMap<String, Relation>)>,
) -> Result<&'a Relation> {
    if let Some((delta_index, delta_map)) = delta_scc {
        if delta_index == body_index {
            return delta_map.get(&atom.predicate).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Missing delta relation for predicate {}",
                    atom.predicate
                ))
            });
        }
    }
    if let Some(rel) = full_scc.get(&atom.predicate) {
        return Ok(rel);
    }
    global
        .get(&atom.predicate)
        .ok_or_else(|| XlogError::Compilation(format!("Unknown predicate {}", atom.predicate)))
}

#[cfg(feature = "host-io")]
fn negated_atom_holds(
    atom: &Atom,
    rel: &Relation,
    binding: &HashMap<String, Value>,
) -> Result<bool> {
    // Safety: all variables in a negated atom must be bound already (otherwise domain is unknown).
    for term in &atom.terms {
        if let Term::Variable(name) = term {
            if !binding.contains_key(name) {
                return Err(XlogError::UnsafeVariable(name.clone()));
            }
        }
    }

    for tuple in &rel.tuples {
        if atom_matches_bound(atom, tuple, binding)? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(feature = "host-io")]
fn atom_matches_bound(
    atom: &Atom,
    tuple: &[Value],
    binding: &HashMap<String, Value>,
) -> Result<bool> {
    if atom.terms.len() != tuple.len() {
        return Err(XlogError::Compilation(format!(
            "Arity mismatch for {}: atom has {}, tuple has {}",
            atom.predicate,
            atom.terms.len(),
            tuple.len()
        )));
    }
    for (term, value) in atom.terms.iter().zip(tuple.iter()) {
        match term {
            Term::Variable(name) => {
                let Some(bound) = binding.get(name) else {
                    return Err(XlogError::UnsafeVariable(name.clone()));
                };
                if bound != value {
                    return Ok(false);
                }
            }
            Term::Anonymous => {}
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                if &value_from_term(term)? != value {
                    return Ok(false);
                }
            }
            Term::Aggregate(AggExpr { .. }) => {
                return Err(XlogError::Compilation(
                    "Aggregate not allowed in body atom".to_string(),
                ));
            }
            Term::List(_) => return Err(v085_mc_result_term_error("matching", "list")),
            Term::Cons { .. } => return Err(v085_mc_result_term_error("matching", "cons")),
            Term::Compound { .. } => {
                return Err(v085_mc_result_term_error("matching", "compound"));
            }
            Term::PredRef(_) => return Err(v085_mc_result_term_error("matching", "predref")),
        }
    }
    Ok(true)
}

#[cfg(feature = "host-io")]
fn materialize_head_non_aggregate(
    head: &Atom,
    binding: &HashMap<String, Value>,
) -> Result<Vec<Value>> {
    let mut out = Vec::with_capacity(head.terms.len());
    for term in &head.terms {
        match term {
            Term::Variable(name) => {
                let v = binding.get(name).ok_or_else(|| {
                    XlogError::Compilation(format!(
                        "Unbound head variable {} in {}",
                        name, head.predicate
                    ))
                })?;
                out.push(v.clone());
            }
            Term::Anonymous => {
                return Err(XlogError::Compilation(format!(
                    "Anonymous wildcard '_' not allowed in rule head (predicate {})",
                    head.predicate
                )));
            }
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                out.push(value_from_term(term)?);
            }
            Term::Aggregate(_) => {
                return Err(XlogError::Compilation(
                    "Aggregate term in non-aggregate rule head".to_string(),
                ));
            }
            Term::List(_) => {
                return Err(v085_mc_result_term_error(
                    "non-aggregate head materialization",
                    "list",
                ));
            }
            Term::Cons { .. } => {
                return Err(v085_mc_result_term_error(
                    "non-aggregate head materialization",
                    "cons",
                ));
            }
            Term::Compound { .. } => {
                return Err(v085_mc_result_term_error(
                    "non-aggregate head materialization",
                    "compound",
                ));
            }
            Term::PredRef(_) => {
                return Err(v085_mc_result_term_error(
                    "non-aggregate head materialization",
                    "predref",
                ));
            }
        }
    }
    Ok(out)
}

#[cfg(feature = "host-io")]
fn eval_aggregate_head(
    head: &Atom,
    states: Vec<HashMap<String, Value>>,
) -> Result<Vec<Vec<Value>>> {
    // Collect unique group keys (variables) in head order.
    let mut key_vars: Vec<String> = Vec::new();
    let mut key_var_to_pos: HashMap<String, usize> = HashMap::new();

    // Collect unique aggregate specs (op, var) in head order.
    let mut agg_specs: Vec<(AggOp, String)> = Vec::new();
    let mut agg_to_pos: HashMap<(AggOp, String), usize> = HashMap::new();

    for term in &head.terms {
        match term {
            Term::Variable(name) => {
                if !key_var_to_pos.contains_key(name) {
                    let pos = key_vars.len();
                    key_vars.push(name.clone());
                    key_var_to_pos.insert(name.clone(), pos);
                }
            }
            Term::Aggregate(agg) => {
                let key = (agg.op, agg.variable.clone());
                if let std::collections::hash_map::Entry::Vacant(entry) =
                    agg_to_pos.entry(key.clone())
                {
                    let pos = agg_specs.len();
                    agg_specs.push(key);
                    entry.insert(pos);
                }
            }
            Term::Anonymous => {
                return Err(XlogError::Compilation(
                    "Anonymous wildcard '_' not allowed in aggregate rule head".to_string(),
                ));
            }
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {}
            Term::List(_) => {
                return Err(v085_mc_result_term_error("aggregate head planning", "list"));
            }
            Term::Cons { .. } => {
                return Err(v085_mc_result_term_error("aggregate head planning", "cons"));
            }
            Term::Compound { .. } => {
                return Err(v085_mc_result_term_error(
                    "aggregate head planning",
                    "compound",
                ));
            }
            Term::PredRef(_) => {
                return Err(v085_mc_result_term_error(
                    "aggregate head planning",
                    "predref",
                ));
            }
        }
    }

    #[derive(Debug)]
    struct GroupState {
        key: Vec<Value>,
        aggs: Vec<AggState>,
    }

    let mut groups: HashMap<Vec<Value>, GroupState> = HashMap::new();

    for binding in states {
        let mut key: Vec<Value> = Vec::with_capacity(key_vars.len());
        for name in &key_vars {
            let v = binding
                .get(name)
                .ok_or_else(|| XlogError::UnsafeVariable(name.clone()))?;
            key.push(v.clone());
        }

        let entry = groups.entry(key.clone()).or_insert_with(|| GroupState {
            key,
            aggs: agg_specs.iter().map(|(op, _)| AggState::new(*op)).collect(),
        });

        for (idx, (op, var)) in agg_specs.iter().enumerate() {
            let v = binding
                .get(var)
                .ok_or_else(|| XlogError::UnsafeVariable(var.clone()))?;
            entry.aggs[idx].update(*op, v)?;
        }
    }

    let mut out: Vec<Vec<Value>> = Vec::with_capacity(groups.len());
    for group in groups.values() {
        let mut tuple: Vec<Value> = Vec::with_capacity(head.terms.len());
        for term in &head.terms {
            match term {
                Term::Variable(name) => {
                    let pos = *key_var_to_pos.get(name).ok_or_else(|| {
                        XlogError::Compilation(format!(
                            "Aggregate head variable {} is not a group key",
                            name
                        ))
                    })?;
                    tuple.push(group.key[pos].clone());
                }
                Term::Aggregate(AggExpr { op, variable }) => {
                    let idx = *agg_to_pos
                        .get(&(*op, variable.clone()))
                        .expect("agg_to_pos missing");
                    tuple.push(group.aggs[idx].finish(*op)?);
                }
                Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                    tuple.push(value_from_term(term)?);
                }
                Term::Anonymous => unreachable!(),
                Term::List(_) => {
                    return Err(v085_mc_result_term_error(
                        "aggregate head materialization",
                        "list",
                    ));
                }
                Term::Cons { .. } => {
                    return Err(v085_mc_result_term_error(
                        "aggregate head materialization",
                        "cons",
                    ));
                }
                Term::Compound { .. } => {
                    return Err(v085_mc_result_term_error(
                        "aggregate head materialization",
                        "compound",
                    ));
                }
                Term::PredRef(_) => {
                    return Err(v085_mc_result_term_error(
                        "aggregate head materialization",
                        "predref",
                    ));
                }
            }
        }
        out.push(tuple);
    }

    Ok(out)
}

#[cfg(feature = "host-io")]
pub(super) fn atom_holds(store: &HashMap<String, Relation>, atom: &GroundAtom) -> bool {
    store
        .get(&atom.predicate)
        .map(|rel| rel.contains(&atom.args))
        .unwrap_or(false)
}

#[cfg(feature = "host-io")]
pub(super) fn evidence_satisfied(
    store: &HashMap<String, Relation>,
    evidence: &[(GroundAtom, bool)],
) -> bool {
    for (atom, value) in evidence {
        let holds = atom_holds(store, atom);
        if holds != *value {
            return false;
        }
    }
    true
}

#[cfg(feature = "host-io")]
pub(super) fn binomial_estimate(k: usize, n: usize, z: f64) -> (f64, f64, f64, f64) {
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n_f = n as f64;
    let p = (k as f64) / n_f;
    let stderr = (p * (1.0 - p) / n_f).sqrt();

    let z2 = z * z;
    let denom = 1.0 + z2 / n_f;
    let center = (p + z2 / (2.0 * n_f)) / denom;
    let half = z * ((p * (1.0 - p) / n_f) + (z2 / (4.0 * n_f * n_f))).sqrt() / denom;

    let mut lo = center - half;
    let mut hi = center + half;
    if lo < 0.0 {
        lo = 0.0;
    }
    if hi > 1.0 {
        hi = 1.0;
    }
    (p, stderr, lo, hi)
}

// Acklam's inverse normal CDF approximation.
#[cfg(feature = "host-io")]
pub(super) fn normal_quantile(p: f64) -> f64 {
    if !(0.0 < p && p < 1.0) || p.is_nan() {
        return f64::NAN;
    }

    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    if p > P_HIGH {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }

    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
}
