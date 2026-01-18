//! Provenance extraction from XLOG programs into PIR.

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

use xlog_core::{symbol, Result, XlogError};
use xlog_logic::ast::{AggExpr, AggOp, ArithExpr, Atom, BodyLiteral, CompOp, Evidence, ProbQuery, Program, Rule, Term};
use xlog_logic::stratify::{build_dependency_graph, find_sccs_for_lowering, stratify};

use crate::pir::{ChoiceVarId, LeafId, PirGraph, PirNodeId};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Value {
    I64(i64),
    F64(u64),
    Symbol(u32),
    String(String),
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Self::I64(v)
    }
}

impl From<u32> for Value {
    fn from(v: u32) -> Self {
        Self::Symbol(v)
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroundAtom {
    pub predicate: String,
    pub args: Vec<Value>,
}

impl GroundAtom {
    fn new(predicate: impl Into<String>, args: Vec<Value>) -> Self {
        Self {
            predicate: predicate.into(),
            args,
        }
    }
}

#[derive(Debug, Clone)]
struct Relation {
    tuples: HashMap<Vec<Value>, PirNodeId>,
}

impl Relation {
    fn new() -> Self {
        Self {
            tuples: HashMap::new(),
        }
    }

    fn get(&self, tuple: &[Value]) -> Option<PirNodeId> {
        self.tuples.get(tuple).copied()
    }

    fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }

    fn insert_or(&mut self, tuple: Vec<Value>, formula: PirNodeId, builder: &mut PirBuilder) {
        let entry = self.tuples.entry(tuple).or_insert_with(|| builder.const_false());
        *entry = builder.or(vec![*entry, formula]);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PirKey {
    Const(bool),
    Lit(LeafId),
    And(Vec<PirNodeId>),
    Or(Vec<PirNodeId>),
    Decision {
        var: ChoiceVarId,
        child_false: PirNodeId,
        child_true: PirNodeId,
    },
}

impl Hash for PirKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            PirKey::Const(b) => {
                0u8.hash(state);
                b.hash(state);
            }
            PirKey::Lit(l) => {
                1u8.hash(state);
                l.hash(state);
            }
            PirKey::And(children) => {
                2u8.hash(state);
                children.hash(state);
            }
            PirKey::Or(children) => {
                3u8.hash(state);
                children.hash(state);
            }
            PirKey::Decision {
                var,
                child_false,
                child_true,
            } => {
                4u8.hash(state);
                var.hash(state);
                child_false.hash(state);
                child_true.hash(state);
            }
        }
    }
}

#[derive(Debug)]
struct PirBuilder {
    pir: PirGraph,
    intern: HashMap<PirKey, PirNodeId>,
    const_true: PirNodeId,
    const_false: PirNodeId,
}

impl PirBuilder {
    fn new() -> Self {
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();
        let const_false = pir.const_false();

        let mut intern = HashMap::new();
        intern.insert(PirKey::Const(true), const_true);
        intern.insert(PirKey::Const(false), const_false);

        Self {
            pir,
            intern,
            const_true,
            const_false,
        }
    }

    fn finish(self) -> PirGraph {
        self.pir
    }

    fn const_true(&self) -> PirNodeId {
        self.const_true
    }

    fn const_false(&self) -> PirNodeId {
        self.const_false
    }

    fn lit(&mut self, leaf: LeafId) -> PirNodeId {
        let key = PirKey::Lit(leaf);
        if let Some(&id) = self.intern.get(&key) {
            return id;
        }
        let id = self.pir.lit(leaf);
        self.intern.insert(key, id);
        id
    }

    fn and(&mut self, mut children: Vec<PirNodeId>) -> PirNodeId {
        children.retain(|&c| c != self.const_true);
        if children.iter().any(|&c| c == self.const_false) {
            return self.const_false;
        }
        if children.is_empty() {
            return self.const_true;
        }
        if children.len() == 1 {
            return children[0];
        }
        children.sort_by_key(|id| id.as_u32());
        children.dedup();
        if children.len() == 1 {
            return children[0];
        }
        let key = PirKey::And(children.clone());
        if let Some(&id) = self.intern.get(&key) {
            return id;
        }
        let id = self.pir.and(children);
        self.intern.insert(key, id);
        id
    }

    fn or(&mut self, mut children: Vec<PirNodeId>) -> PirNodeId {
        children.retain(|&c| c != self.const_false);
        if children.iter().any(|&c| c == self.const_true) {
            return self.const_true;
        }
        if children.is_empty() {
            return self.const_false;
        }
        if children.len() == 1 {
            return children[0];
        }
        children.sort_by_key(|id| id.as_u32());
        children.dedup();
        if children.len() == 1 {
            return children[0];
        }
        let key = PirKey::Or(children.clone());
        if let Some(&id) = self.intern.get(&key) {
            return id;
        }
        let id = self.pir.or(children);
        self.intern.insert(key, id);
        id
    }

    fn decision(&mut self, var: ChoiceVarId, child_false: PirNodeId, child_true: PirNodeId) -> PirNodeId {
        if child_false == child_true {
            return child_true;
        }
        let key = PirKey::Decision {
            var,
            child_false,
            child_true,
        };
        if let Some(&id) = self.intern.get(&key) {
            return id;
        }
        let id = self.pir.decision(var, child_false, child_true);
        self.intern.insert(key, id);
        id
    }

    fn choice_lit(&mut self, var: ChoiceVarId, is_true: bool) -> PirNodeId {
        if is_true {
            self.decision(var, self.const_false(), self.const_true())
        } else {
            self.decision(var, self.const_true(), self.const_false())
        }
    }
}

/// Provenance extraction result: PIR graph plus per-tuple formulas and weight metadata.
#[derive(Debug)]
pub struct Provenance {
    pub pir: PirGraph,
    pub leaf_probs: BTreeMap<LeafId, f64>,
    pub choice_probs: BTreeMap<ChoiceVarId, (f64, f64)>,
    tuple_formulas: HashMap<GroundAtom, PirNodeId>,
    pub queries: Vec<GroundAtom>,
    pub evidence: Vec<(GroundAtom, bool)>,
}

impl Provenance {
    pub fn query_formula(&self, predicate: &str, args: &[Value]) -> Option<PirNodeId> {
        self.tuple_formulas
            .get(&GroundAtom::new(predicate, args.to_vec()))
            .copied()
    }
}

pub fn extract_from_source(source: &str) -> Result<Provenance> {
    let program = xlog_logic::parse_program(source)?;
    extract_from_program(&program)
}

pub fn extract_from_program(program: &Program) -> Result<Provenance> {
    // Stratify first to fail fast on unsupported recursion patterns.
    let _ = stratify(program)?;

    let mut builder = PirBuilder::new();

    let mut leaf_probs: BTreeMap<LeafId, f64> = BTreeMap::new();
    let mut choice_probs: BTreeMap<ChoiceVarId, (f64, f64)> = BTreeMap::new();

    let mut store: HashMap<String, Relation> = HashMap::new();

    // Deterministic facts.
    for fact in program.facts() {
        let key = atom_key_from_ground_atom(&fact.head)?;
        let rel = store.entry(key.predicate.clone()).or_insert_with(Relation::new);
        rel.insert_or(key.args.clone(), builder.const_true(), &mut builder);
    }

    // Probabilistic facts.
    let mut next_leaf: u32 = 0;
    for pf in &program.prob_facts {
        validate_prob(pf.prob, "probabilistic fact")?;
        let key = atom_key_from_ground_atom(&pf.atom)?;
        let leaf = LeafId::new(next_leaf);
        next_leaf = next_leaf.saturating_add(1);
        leaf_probs.insert(leaf, pf.prob);

        let rel = store.entry(key.predicate.clone()).or_insert_with(Relation::new);
        rel.insert_or(key.args.clone(), builder.lit(leaf), &mut builder);
    }

    // Annotated disjunctions: lower to a chain of Bernoulli decisions.
    let mut next_choice: u32 = 0;
    for ad in &program.annotated_disjunctions {
        if ad.choices.is_empty() {
            return Err(XlogError::Compilation(
                "Annotated disjunction must contain at least one choice".to_string(),
            ));
        }
        let (vars, outcome_formulas) =
            compile_annotated_disjunction(ad, &mut next_choice, &mut choice_probs, &mut builder)?;
        let _ = vars;

        for (pf, formula) in ad.choices.iter().zip(outcome_formulas.into_iter()) {
            let key = atom_key_from_ground_atom(&pf.atom)?;
            let rel = store.entry(key.predicate.clone()).or_insert_with(Relation::new);
            rel.insert_or(key.args.clone(), formula, &mut builder);
        }
    }

    // Evaluate rules SCC-by-SCC (semi-naive for recursive SCCs).
    let graph = build_dependency_graph(program);
    for pred in &graph.predicates {
        store.entry(pred.clone()).or_insert_with(Relation::new);
    }
    let sccs = find_sccs_for_lowering(&graph);

    let mut rules_by_head: HashMap<String, Vec<Rule>> = HashMap::new();
    for rule in program.proper_rules() {
        if rule.has_aggregation() {
            return Err(XlogError::Compilation(
                "Provenance extraction does not support aggregation".to_string(),
            ));
        }
        if rule.has_negation() {
            return Err(XlogError::Compilation(
                "Provenance extraction does not support negation".to_string(),
            ));
        }
        rules_by_head
            .entry(rule.head.predicate.clone())
            .or_default()
            .push(rule.clone());
    }

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

        let recursive = is_recursive_scc(&scc, &scc_rules);
        if recursive {
            eval_recursive_scc(&scc, &scc_rules, &mut store, &mut builder)?;
        } else {
            eval_non_recursive_scc(&scc_rules, &mut store, &mut builder)?;
        }
    }

    // Snapshot tuple formulas.
    let mut tuple_formulas: HashMap<GroundAtom, PirNodeId> = HashMap::new();
    for (pred, rel) in &store {
        for (tuple, formula) in &rel.tuples {
            tuple_formulas.insert(GroundAtom::new(pred.clone(), tuple.clone()), *formula);
        }
    }

    let mut queries: Vec<GroundAtom> = Vec::new();
    for ProbQuery { atom } in &program.prob_queries {
        queries.push(atom_key_from_ground_atom(atom)?);
    }

    let mut evidence: Vec<(GroundAtom, bool)> = Vec::new();
    for Evidence { atom, value } in &program.evidence {
        evidence.push((atom_key_from_ground_atom(atom)?, *value));
    }

    Ok(Provenance {
        pir: builder.finish(),
        leaf_probs,
        choice_probs,
        tuple_formulas,
        queries,
        evidence,
    })
}

pub(crate) fn validate_prob(p: f64, what: &str) -> Result<()> {
    if !(0.0..=1.0).contains(&p) || p.is_nan() {
        return Err(XlogError::Compilation(format!(
            "Invalid probability {} for {} (expected 0<=p<=1)",
            p, what
        )));
    }
    Ok(())
}

pub(crate) fn atom_key_from_ground_atom(atom: &Atom) -> Result<GroundAtom> {
    let mut args = Vec::with_capacity(atom.terms.len());
    for term in &atom.terms {
        if !term.is_constant() {
            return Err(XlogError::Compilation(format!(
                "Expected ground atom, found non-constant term in {}",
                atom.predicate
            )));
        }
        args.push(value_from_term(term)?);
    }
    Ok(GroundAtom::new(atom.predicate.clone(), args))
}

pub(crate) fn value_from_term(term: &Term) -> Result<Value> {
    match term {
        Term::Integer(i) => Ok(Value::I64(*i)),
        Term::Float(f) => Ok(Value::F64(f.to_bits())),
        Term::String(s) => Ok(Value::String(s.clone())),
        Term::Symbol(s) => Ok(Value::Symbol(symbol::intern(s))),
        Term::Variable(_) | Term::Anonymous | Term::Aggregate(_) => Err(XlogError::Compilation(
            "Non-constant term cannot be converted to a value".to_string(),
        )),
    }
}

fn compile_annotated_disjunction(
    ad: &xlog_logic::ast::AnnotatedDisjunction,
    next_choice: &mut u32,
    choice_probs: &mut BTreeMap<ChoiceVarId, (f64, f64)>,
    builder: &mut PirBuilder,
) -> Result<(Vec<ChoiceVarId>, Vec<PirNodeId>)> {
    for pf in &ad.choices {
        validate_prob(pf.prob, "annotated disjunction choice")?;
        let _ = atom_key_from_ground_atom(&pf.atom)?;
    }

    let mut probs: Vec<f64> = ad.choices.iter().map(|pf| pf.prob).collect();
    let sum: f64 = probs.iter().copied().sum();
    let eps = 1e-12;
    if sum > 1.0 + eps {
        return Err(XlogError::Compilation(format!(
            "Annotated disjunction probabilities sum to {} (> 1.0)",
            sum
        )));
    }

    let mut has_none = false;
    let none_prob = (1.0 - sum).max(0.0);
    if none_prob > eps {
        probs.push(none_prob);
        has_none = true;
    }

    let m = probs.len();
    if m == 1 {
        return Ok((Vec::new(), vec![builder.const_true()]));
    }

    let mut vars: Vec<ChoiceVarId> = Vec::with_capacity(m.saturating_sub(1));
    let mut remaining = 1.0f64;
    for i in 0..(m - 1) {
        let p_i = probs[i];
        let cond_true = if remaining <= 0.0 { 0.0 } else { p_i / remaining };
        validate_prob(cond_true, "annotated disjunction conditional")?;
        let cond_false = 1.0 - cond_true;
        let var = ChoiceVarId::new(*next_choice);
        *next_choice = next_choice.saturating_add(1);
        vars.push(var);
        choice_probs.insert(var, (cond_true, cond_false));
        remaining -= p_i;
    }

    let mut outcome_formulas: Vec<PirNodeId> = Vec::new();
    for i in 0..ad.choices.len() {
        let mut conds: Vec<PirNodeId> = Vec::new();
        for (j, &var) in vars.iter().enumerate() {
            if j < i {
                conds.push(builder.choice_lit(var, false));
            } else if j == i {
                conds.push(builder.choice_lit(var, true));
                break;
            }
        }
        outcome_formulas.push(builder.and(conds));
    }

    if has_none {
        // None branch consumes the final remaining probability; it produces no fact.
        // We still need the decision variables so probabilities normalize.
    }

    Ok((vars, outcome_formulas))
}

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

fn eval_non_recursive_scc(
    rules: &[Rule],
    store: &mut HashMap<String, Relation>,
    builder: &mut PirBuilder,
) -> Result<()> {
    for rule in rules {
        let derived = eval_rule(rule, store, &HashMap::new(), None, builder)?;
        let rel = store.entry(rule.head.predicate.clone()).or_insert_with(Relation::new);
        for (tuple, formula) in derived {
            rel.insert_or(tuple, formula, builder);
        }
    }
    Ok(())
}

const MAX_PROVENANCE_ITERATIONS: usize = 1024;

fn eval_recursive_scc(
    scc: &[String],
    rules: &[Rule],
    store: &mut HashMap<String, Relation>,
    builder: &mut PirBuilder,
) -> Result<()> {
    let scc_set: std::collections::HashSet<&str> = scc.iter().map(|s| s.as_str()).collect();

    // Snapshot full relations for the SCC.
    let mut full: HashMap<String, Relation> = HashMap::new();
    for pred in scc {
        let rel = store.get(pred).cloned().unwrap_or_else(Relation::new);
        full.insert(pred.clone(), rel);
    }

    // Seed: evaluate all rules once against the current full snapshot.
    let mut delta: HashMap<String, Relation> = HashMap::new();
    for rule in rules {
        let derived = eval_rule(rule, store, &full, None, builder)?;
        if derived.is_empty() {
            continue;
        }
        let head = rule.head.predicate.clone();
        let delta_rel = delta.entry(head.clone()).or_insert_with(Relation::new);
        let full_rel = full.entry(head).or_insert_with(Relation::new);
        for (tuple, proof) in derived {
            let old = full_rel.get(&tuple).unwrap_or(builder.const_false());
            let combined = builder.or(vec![old, proof]);
            if combined != old {
                full_rel.tuples.insert(tuple.clone(), combined);
                delta_rel.insert_or(tuple, proof, builder);
            }
        }
    }

    let mut reached_fixpoint = false;
    for _ in 0..MAX_PROVENANCE_ITERATIONS {
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
                        let pred = &atom.predicate;
                        let non_empty = delta_prev.get(pred).map(|r| !r.is_empty()).unwrap_or(false);
                        non_empty.then_some(i)
                    }
                    _ => None,
                })
                .collect();
            if body_indices.is_empty() {
                continue;
            }

            let mut derived_all: HashMap<Vec<Value>, PirNodeId> = HashMap::new();
            for idx in body_indices {
                let derived = eval_rule(rule, store, &full_prev, Some((idx, &delta_prev)), builder)?;
                for (tuple, proof) in derived {
                    let entry = derived_all.entry(tuple).or_insert_with(|| builder.const_false());
                    *entry = builder.or(vec![*entry, proof]);
                }
            }

            if derived_all.is_empty() {
                continue;
            }

            let head = rule.head.predicate.clone();
            let delta_rel = delta.entry(head.clone()).or_insert_with(Relation::new);
            let full_rel = full.entry(head).or_insert_with(Relation::new);
            for (tuple, proof) in derived_all {
                let old = full_rel.get(&tuple).unwrap_or(builder.const_false());
                let combined = builder.or(vec![old, proof]);
                if combined != old {
                    full_rel.tuples.insert(tuple.clone(), combined);
                    delta_rel.insert_or(tuple, proof, builder);
                }
            }
        }
    }
    if !reached_fixpoint {
        return Err(XlogError::Compilation(format!(
            "Provenance iteration limit ({}) exceeded for SCC {:?}",
            MAX_PROVENANCE_ITERATIONS, scc
        )));
    }

    // Write back SCC relations.
    for (pred, rel) in full {
        store.insert(pred, rel);
    }

    Ok(())
}

/// Evaluate a single rule and produce a map from head tuples to proof formulas.
///
/// `full_scc` is the per-SCC snapshot for recursive predicates; `delta_scc` is optional and
/// provides a delta relation for a specific body literal index.
fn eval_rule(
    rule: &Rule,
    global: &HashMap<String, Relation>,
    full_scc: &HashMap<String, Relation>,
    delta_scc: Option<(usize, &HashMap<String, Relation>)>,
    builder: &mut PirBuilder,
) -> Result<HashMap<Vec<Value>, PirNodeId>> {
    let mut states: Vec<(HashMap<String, Value>, PirNodeId)> = Vec::new();
    states.push((HashMap::new(), builder.const_true()));

    for (idx, lit) in rule.body.iter().enumerate() {
        let mut next_states: Vec<(HashMap<String, Value>, PirNodeId)> = Vec::new();
        match lit {
            BodyLiteral::Positive(atom) => {
                let rel = select_relation(atom, idx, global, full_scc, delta_scc)?;
                for (binding, prov) in states {
                    for (tuple, tuple_prov) in &rel.tuples {
                        let mut binding2 = binding.clone();
                        if unify_atom(atom, tuple, &mut binding2)? {
                            let prov2 = builder.and(vec![prov, *tuple_prov]);
                            next_states.push((binding2, prov2));
                        }
                    }
                }
            }
            BodyLiteral::Comparison(cmp) => {
                for (binding, prov) in states {
                    if eval_comparison(cmp.op, &cmp.left, &cmp.right, &binding)? {
                        next_states.push((binding, prov));
                    }
                }
            }
            BodyLiteral::IsExpr(is_expr) => {
                for (mut binding, prov) in states {
                    if binding.contains_key(&is_expr.target) {
                        return Err(XlogError::Compilation(format!(
                            "Is-expression target {} is already bound",
                            is_expr.target
                        )));
                    }
                    let v = eval_arith_expr(&is_expr.expr, &binding)?;
                    binding.insert(is_expr.target.clone(), v);
                    next_states.push((binding, prov));
                }
            }
            BodyLiteral::Negated(atom) => {
                return Err(XlogError::Compilation(format!(
                    "Negation not supported in provenance extraction (found not {})",
                    atom.predicate
                )));
            }
        }
        states = next_states;
        if states.is_empty() {
            break;
        }
    }

    let mut out: HashMap<Vec<Value>, PirNodeId> = HashMap::new();
    for (binding, prov) in states {
        let head_tuple = materialize_head(&rule.head, &binding)?;
        let entry = out.entry(head_tuple).or_insert_with(|| builder.const_false());
        *entry = builder.or(vec![*entry, prov]);
    }
    Ok(out)
}

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

pub(crate) fn unify_atom(
    atom: &Atom,
    tuple: &[Value],
    binding: &mut HashMap<String, Value>,
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
            Term::Variable(name) => match binding.get(name) {
                Some(existing) => {
                    if existing != value {
                        return Ok(false);
                    }
                }
                None => {
                    binding.insert(name.clone(), value.clone());
                }
            },
            Term::Anonymous => {}
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                if &value_from_term(term)? != value {
                    return Ok(false);
                }
            }
            Term::Aggregate(AggExpr { op: _, variable: _ }) => {
                return Err(XlogError::Compilation(
                    "Aggregation not supported in provenance extraction".to_string(),
                ));
            }
        }
    }
    Ok(true)
}

fn materialize_head(head: &Atom, binding: &HashMap<String, Value>) -> Result<Vec<Value>> {
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
                    "Anonymous variable in head of {} is not supported",
                    head.predicate
                )));
            }
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                out.push(value_from_term(term)?);
            }
            Term::Aggregate(AggExpr { op: AggOp::Count, variable: _ })
            | Term::Aggregate(AggExpr { op: AggOp::Sum, variable: _ })
            | Term::Aggregate(AggExpr { op: AggOp::Min, variable: _ })
            | Term::Aggregate(AggExpr { op: AggOp::Max, variable: _ })
            | Term::Aggregate(AggExpr { op: AggOp::LogSumExp, variable: _ }) => {
                return Err(XlogError::Compilation(
                    "Aggregation not supported in provenance extraction".to_string(),
                ));
            }
        }
    }
    Ok(out)
}

pub(crate) fn eval_comparison(
    op: CompOp,
    left: &Term,
    right: &Term,
    binding: &HashMap<String, Value>,
) -> Result<bool> {
    let l = resolve_term(left, binding)?;
    let r = resolve_term(right, binding)?;
    match (l, r) {
        (Value::I64(a), Value::I64(b)) => Ok(compare_ord(op, a.cmp(&b))),
        (Value::F64(a_bits), Value::F64(b_bits)) => {
            let a = f64::from_bits(a_bits);
            let b = f64::from_bits(b_bits);
            match op {
                CompOp::Eq => Ok(a == b),
                CompOp::Ne => Ok(a != b),
                CompOp::Lt => Ok(a < b),
                CompOp::Le => Ok(a <= b),
                CompOp::Gt => Ok(a > b),
                CompOp::Ge => Ok(a >= b),
            }
        }
        (Value::Symbol(a), Value::Symbol(b)) => Ok(compare_ord(op, a.cmp(&b))),
        (Value::String(a), Value::String(b)) => Ok(compare_ord(op, a.cmp(&b))),
        _ => Err(XlogError::Compilation(
            "Comparison between differing types is not supported".to_string(),
        )),
    }
}

pub(crate) fn compare_ord(op: CompOp, ord: std::cmp::Ordering) -> bool {
    use std::cmp::Ordering;
    match op {
        CompOp::Eq => ord == Ordering::Equal,
        CompOp::Ne => ord != Ordering::Equal,
        CompOp::Lt => ord == Ordering::Less,
        CompOp::Le => ord == Ordering::Less || ord == Ordering::Equal,
        CompOp::Gt => ord == Ordering::Greater,
        CompOp::Ge => ord == Ordering::Greater || ord == Ordering::Equal,
    }
}

pub(crate) fn resolve_term(term: &Term, binding: &HashMap<String, Value>) -> Result<Value> {
    match term {
        Term::Variable(name) => binding.get(name).cloned().ok_or_else(|| {
            XlogError::Compilation(format!("Unbound variable {} in comparison", name))
        }),
        Term::Anonymous => Err(XlogError::Compilation(
            "Anonymous variable not allowed in comparison".to_string(),
        )),
        Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => value_from_term(term),
        Term::Aggregate(_) => Err(XlogError::Compilation(
            "Aggregation not supported in provenance extraction".to_string(),
        )),
    }
}

pub(crate) fn eval_arith_expr(expr: &ArithExpr, binding: &HashMap<String, Value>) -> Result<Value> {
    match expr {
        ArithExpr::Variable(name) => binding.get(name).cloned().ok_or_else(|| {
            XlogError::Compilation(format!("Unbound variable {} in arithmetic", name))
        }),
        ArithExpr::Integer(i) => Ok(Value::I64(*i)),
        ArithExpr::Float(f) => Ok(Value::F64(f.to_bits())),
        ArithExpr::Add(l, r) => eval_bin_op(l, r, binding, |a, b| a + b, |a, b| a + b),
        ArithExpr::Sub(l, r) => eval_bin_op(l, r, binding, |a, b| a - b, |a, b| a - b),
        ArithExpr::Mul(l, r) => eval_bin_op(l, r, binding, |a, b| a * b, |a, b| a * b),
        ArithExpr::Div(l, r) => eval_bin_op(l, r, binding, |a, b| a / b, |a, b| a / b),
        ArithExpr::Mod(l, r) => eval_bin_op(l, r, binding, |a, b| a % b, |a, b| a % b),
        ArithExpr::Abs(e) => match eval_arith_expr(e, binding)? {
            Value::I64(i) => Ok(Value::I64(i.abs())),
            Value::F64(bits) => {
                let f = f64::from_bits(bits).abs();
                Ok(Value::F64(f.to_bits()))
            }
            _ => Err(XlogError::Compilation("abs() requires numeric input".to_string())),
        },
        ArithExpr::Min(l, r) => eval_bin_op(l, r, binding, |a, b| a.min(b), |a, b| a.min(b)),
        ArithExpr::Max(l, r) => eval_bin_op(l, r, binding, |a, b| a.max(b), |a, b| a.max(b)),
        ArithExpr::Pow(l, r) => {
            let a = eval_arith_expr(l, binding)?;
            let b = eval_arith_expr(r, binding)?;
            match (a, b) {
                (Value::I64(a), Value::I64(b)) => Ok(Value::I64(a.pow(u32::try_from(b).map_err(|_| {
                    XlogError::Compilation("pow exponent must fit in u32".to_string())
                })?))),
                (Value::F64(a), Value::F64(b)) => Ok(Value::F64(f64::from_bits(a).powf(f64::from_bits(b)).to_bits())),
                _ => Err(XlogError::Compilation("pow requires numeric inputs of same type".to_string())),
            }
        }
        ArithExpr::Cast(e, _ty) => {
            // For provenance compilation we preserve the numeric value; the runtime has a full
            // type system, but provenance needs only deterministic evaluation.
            eval_arith_expr(e, binding)
        }
    }
}

pub(crate) fn eval_bin_op<FInt, FFloat>(
    l: &ArithExpr,
    r: &ArithExpr,
    binding: &HashMap<String, Value>,
    op_int: FInt,
    op_float: FFloat,
) -> Result<Value>
where
    FInt: FnOnce(i64, i64) -> i64,
    FFloat: FnOnce(f64, f64) -> f64,
{
    let a = eval_arith_expr(l, binding)?;
    let b = eval_arith_expr(r, binding)?;
    match (a, b) {
        (Value::I64(a), Value::I64(b)) => Ok(Value::I64(op_int(a, b))),
        (Value::F64(a), Value::F64(b)) => Ok(Value::F64(op_float(f64::from_bits(a), f64::from_bits(b)).to_bits())),
        _ => Err(XlogError::Compilation(
            "Arithmetic operation requires matching numeric types".to_string(),
        )),
    }
}
