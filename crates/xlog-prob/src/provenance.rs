//! Provenance extraction from XLOG programs into PIR.

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use xlog_core::{symbol, Result, ScalarType, Schema, XlogError};
use xlog_logic::ast::{
    AggExpr, AggOp, ArithExpr, Atom, BodyLiteral, CompOp, Evidence, ProbQuery, Program, Rule, Term,
};
use xlog_logic::stratify::{
    analyze_stratification, build_dependency_graph, find_sccs_for_lowering, stratify,
};
use xlog_logic::{
    compare_arithmetic_values, evaluate_arithmetic_expression, ArithmeticValue, Lowerer,
};

use crate::wfs::{evaluate_wfs_rules, WfsAtom, WfsConfig, WfsLiteral, WfsRule};

use crate::aggregates::{AggState, AggStateKey};
use crate::pir::{ChoiceVarId, LeafId, PirGraph, PirNodeId};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Value {
    I64(i64),
    F64(u64),
    Symbol(u32),
    String(String),
}

type ProvenanceBinding = HashMap<String, Value>;
type ArithmeticBinding = HashMap<String, ArithmeticValue>;
type ProvenanceEvaluationState = (ProvenanceBinding, ArithmeticBinding, PirNodeId);

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

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GroundAtom {
    pub predicate: String,
    pub args: Vec<Value>,
}

impl GroundAtom {
    pub fn new(predicate: impl Into<String>, args: Vec<Value>) -> Self {
        Self {
            predicate: predicate.into(),
            args,
        }
    }
}

/// Metadata for a single Bernoulli decision stage in an annotated disjunction.
#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceSource {
    /// Explicit heads of the annotated disjunction, paired with their declared
    /// (marginal) probabilities. Does not include the synthetic implicit "none"
    /// branch. Shared (`Arc`) across every Bernoulli chain variable of the same
    /// disjunction so that a k-head disjunction pays for one k-length vector,
    /// not O(k) independent clones of it.
    pub choices: Arc<[(GroundAtom, f64)]>,
    /// Position of this ChoiceVarId in the m-1 Bernoulli decision chain.
    pub choice_index: usize,
    /// Enclosing annotated-disjunction identity. `None` in v1.
    pub source_id: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateLiftStatus {
    Fired,
    FallbackExactEnumeration,
    Declined,
}

impl AggregateLiftStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            AggregateLiftStatus::Fired => "fired",
            AggregateLiftStatus::FallbackExactEnumeration => "fallback_exact_enumeration",
            AggregateLiftStatus::Declined => "declined",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregateLiftReport {
    pub predicate: String,
    pub group_key: Vec<Value>,
    pub operator: String,
    pub finite_domain_source: String,
    pub deterministic_rows: usize,
    pub uncertain_rows: usize,
    pub domain_size: usize,
    pub cap: usize,
    pub status: AggregateLiftStatus,
    pub reason: String,
    pub naive_outcomes: u128,
    pub dynamic_programming_states: usize,
}

#[derive(Debug, Clone)]
struct Relation {
    tuples: BTreeMap<Vec<Value>, PirNodeId>,
}

impl Relation {
    fn new() -> Self {
        Self {
            tuples: BTreeMap::new(),
        }
    }

    fn get(&self, tuple: &[Value]) -> Option<PirNodeId> {
        self.tuples.get(tuple).copied()
    }

    fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }

    fn insert_or(&mut self, tuple: Vec<Value>, formula: PirNodeId, builder: &mut PirBuilder) {
        let entry = self
            .tuples
            .entry(tuple)
            .or_insert_with(|| builder.const_false());
        *entry = builder.or(vec![*entry, formula]);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PirKey {
    Const(bool),
    Lit(LeafId),
    NegLit(LeafId),
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
            PirKey::NegLit(l) => {
                5u8.hash(state);
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
    /// Children of interned Or nodes, used to flatten nested ORs and apply
    /// absorption during normalization. Nodes absent from the map are opaque.
    or_children: HashMap<PirNodeId, Vec<PirNodeId>>,
    /// Children of interned And nodes (same role as `or_children`).
    and_children: HashMap<PirNodeId, Vec<PirNodeId>>,
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
            or_children: HashMap::new(),
            and_children: HashMap::new(),
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

    fn neg_lit(&mut self, leaf: LeafId) -> PirNodeId {
        let key = PirKey::NegLit(leaf);
        if let Some(&id) = self.intern.get(&key) {
            return id;
        }
        let id = self.pir.neg_lit(leaf);
        self.intern.insert(key, id);
        id
    }

    fn and(&mut self, children: Vec<PirNodeId>) -> PirNodeId {
        // Flatten nested ANDs (associativity) so recursive-SCC provenance cannot
        // grow syntactically forever while staying semantically fixed.
        let mut flat: Vec<PirNodeId> = Vec::with_capacity(children.len());
        for c in children {
            match self.and_children.get(&c) {
                Some(sub) => flat.extend_from_slice(sub),
                None => flat.push(c),
            }
        }
        let mut children = flat;
        children.retain(|&c| c != self.const_true);
        if children.contains(&self.const_false) {
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
        // Absorption: a ∧ (a ∨ b) = a — drop any Or-child containing another member.
        if children.len() > 1 {
            let members = children.clone();
            children.retain(|c| match self.or_children.get(c) {
                Some(sub) => !sub.iter().any(|s| {
                    s != c
                        && members
                            .binary_search_by_key(&s.as_u32(), |m| m.as_u32())
                            .is_ok()
                }),
                None => true,
            });
        }
        if children.len() == 1 {
            return children[0];
        }
        let key = PirKey::And(children.clone());
        if let Some(&id) = self.intern.get(&key) {
            return id;
        }
        let id = self.pir.and(children.clone());
        self.intern.insert(key, id);
        self.and_children.insert(id, children);
        id
    }

    fn or(&mut self, children: Vec<PirNodeId>) -> PirNodeId {
        // Flatten nested ORs (associativity) — see `and` for rationale.
        let mut flat: Vec<PirNodeId> = Vec::with_capacity(children.len());
        for c in children {
            match self.or_children.get(&c) {
                Some(sub) => flat.extend_from_slice(sub),
                None => flat.push(c),
            }
        }
        let mut children = flat;
        children.retain(|&c| c != self.const_false);
        if children.contains(&self.const_true) {
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
        // Absorption: a ∨ (a ∧ b) = a — drop any And-child containing another member.
        if children.len() > 1 {
            let members = children.clone();
            children.retain(|c| match self.and_children.get(c) {
                Some(sub) => !sub.iter().any(|s| {
                    s != c
                        && members
                            .binary_search_by_key(&s.as_u32(), |m| m.as_u32())
                            .is_ok()
                }),
                None => true,
            });
        }
        if children.len() == 1 {
            return children[0];
        }
        let key = PirKey::Or(children.clone());
        if let Some(&id) = self.intern.get(&key) {
            return id;
        }
        let id = self.pir.or(children.clone());
        self.intern.insert(key, id);
        self.or_children.insert(id, children);
        id
    }

    fn decision(
        &mut self,
        var: ChoiceVarId,
        child_false: PirNodeId,
        child_true: PirNodeId,
    ) -> PirNodeId {
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
    tuple_formulas: BTreeMap<GroundAtom, PirNodeId>,
    pub queries: Vec<GroundAtom>,
    pub evidence: Vec<(GroundAtom, bool)>,
    pub leaf_atoms: BTreeMap<LeafId, GroundAtom>,
    pub choice_sources: BTreeMap<ChoiceVarId, ChoiceSource>,
    pub aggregate_lifting: Vec<AggregateLiftReport>,
    schemas: HashMap<String, Schema>,
}

impl Provenance {
    pub fn query_formula(&self, predicate: &str, args: &[Value]) -> Option<PirNodeId> {
        let atom = self
            .canonical_atom(&GroundAtom::new(predicate, args.to_vec()))
            .ok()?;
        self.tuple_formulas.get(&atom).copied()
    }

    pub(crate) fn canonical_atom(&self, atom: &GroundAtom) -> Result<GroundAtom> {
        let args = canonicalize_public_values(&atom.predicate, &atom.args, &self.schemas)?;
        Ok(GroundAtom::new(atom.predicate.clone(), args))
    }

    pub fn leaf_atom(&self, leaf: LeafId) -> Option<&GroundAtom> {
        self.leaf_atoms.get(&leaf)
    }

    pub fn choice_source(&self, var: ChoiceVarId) -> Option<&ChoiceSource> {
        self.choice_sources.get(&var)
    }

    /// Iterate over canonical semantic tuple keys and their provenance formulas.
    ///
    /// Unlike source-facing query, evidence, leaf, and choice metadata, these keys
    /// describe execution identity. Schema-equivalent quoted and bare symbol
    /// spellings therefore share one key, and derived tuples may have no unique
    /// source spelling.
    pub fn atoms_with_formulas(&self) -> impl Iterator<Item = (&GroundAtom, PirNodeId)> + '_ {
        self.tuple_formulas.iter().map(|(atom, &id)| (atom, id))
    }
}

pub fn extract_from_source(source: &str) -> Result<Provenance> {
    let program = xlog_logic::parse_program(source)?;
    extract_from_program(&program)
}

pub(crate) fn arithmetic_schemas(program: &Program) -> Result<HashMap<String, Schema>> {
    let mut lowerer = Lowerer::new();
    lowerer.infer_and_validate_schemas(program)?;
    let mut schemas = lowerer.schemas().clone();
    for Evidence { atom, .. } in &program.evidence {
        ensure_ground_atom_schema(atom, &mut schemas);
    }
    for ProbQuery { atom } in &program.prob_queries {
        ensure_ground_atom_schema(atom, &mut schemas);
    }
    Ok(schemas)
}

fn ensure_ground_atom_schema(atom: &Atom, schemas: &mut HashMap<String, Schema>) {
    schemas.entry(atom.predicate.clone()).or_insert_with(|| {
        Schema::new(
            atom.terms
                .iter()
                .enumerate()
                .map(|(index, term)| (format!("c{index}"), term.inferred_scalar_type()))
                .collect(),
        )
    });
}

pub(crate) fn canonicalize_probabilistic_program(
    program: &Program,
    schemas: &HashMap<String, Schema>,
) -> Result<Program> {
    let mut program = program.clone();

    for rule in &mut program.rules {
        canonicalize_atom_constants(&mut rule.head, schemas)?;
        canonicalize_body_constants(&mut rule.body, schemas)?;
    }
    for fact in &mut program.prob_facts {
        canonicalize_atom_constants(&mut fact.atom, schemas)?;
    }
    for disjunction in &mut program.annotated_disjunctions {
        for choice in &mut disjunction.choices {
            canonicalize_atom_constants(&mut choice.atom, schemas)?;
        }
    }
    for evidence in &mut program.evidence {
        canonicalize_atom_constants(&mut evidence.atom, schemas)?;
    }
    for query in &mut program.prob_queries {
        canonicalize_atom_constants(&mut query.atom, schemas)?;
    }

    Ok(program)
}

fn canonicalize_body_constants(
    body: &mut [BodyLiteral],
    schemas: &HashMap<String, Schema>,
) -> Result<()> {
    for literal in body {
        match literal {
            BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                canonicalize_atom_constants(atom, schemas)?;
            }
            BodyLiteral::Epistemic(_)
            | BodyLiteral::Comparison(_)
            | BodyLiteral::IsExpr(_)
            | BodyLiteral::Univ(_) => {}
        }
    }
    Ok(())
}

fn canonicalize_atom_constants(atom: &mut Atom, schemas: &HashMap<String, Schema>) -> Result<()> {
    let schema = schemas.get(&atom.predicate).ok_or_else(|| {
        XlogError::Compilation(format!(
            "Probabilistic value canonicalization requires a schema for predicate '{}'",
            atom.predicate
        ))
    })?;
    if schema.arity() != atom.terms.len() {
        return Err(XlogError::Compilation(format!(
            "Predicate '{}' has arity {}, but its inferred schema has arity {}",
            atom.predicate,
            atom.terms.len(),
            schema.arity()
        )));
    }

    for (index, term) in atom.terms.iter_mut().enumerate() {
        if !matches!(
            term,
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_)
        ) {
            continue;
        }
        let scalar_type = schema.column_type(index).ok_or_else(|| {
            XlogError::Compilation(format!(
                "Predicate '{}' has no type for column {}",
                atom.predicate, index
            ))
        })?;
        let value = ArithmeticValue::from_typed_term(term, scalar_type)?;
        *term = term_from_public_value(provenance_value_from_arithmetic(value)?);
    }
    Ok(())
}

pub(crate) fn canonicalize_public_values(
    predicate: &str,
    args: &[Value],
    schemas: &HashMap<String, Schema>,
) -> Result<Vec<Value>> {
    let schema = schemas.get(predicate).ok_or_else(|| {
        XlogError::Compilation(format!(
            "Probabilistic value canonicalization requires a schema for predicate '{predicate}'"
        ))
    })?;
    if schema.arity() != args.len() {
        return Err(XlogError::Compilation(format!(
            "Predicate '{predicate}' has arity {}, but received {} values",
            schema.arity(),
            args.len()
        )));
    }

    args.iter()
        .enumerate()
        .map(|(index, value)| {
            let scalar_type = schema.column_type(index).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Predicate '{predicate}' has no type for column {index}"
                ))
            })?;
            let value = arithmetic_value_from_typed_provenance(value, scalar_type)?;
            provenance_value_from_arithmetic(value)
        })
        .collect()
}

fn term_from_public_value(value: Value) -> Term {
    match value {
        Value::I64(value) => Term::Integer(value),
        Value::F64(value) => Term::Float(f64::from_bits(value)),
        Value::Symbol(value) => Term::Symbol(value),
        Value::String(value) => Term::String(value),
    }
}

pub(crate) fn presentation_atom_from_canonical(
    source: &Atom,
    canonical: &Atom,
    schemas: &HashMap<String, Schema>,
) -> Result<GroundAtom> {
    if source.predicate != canonical.predicate || source.terms.len() != canonical.terms.len() {
        return Err(XlogError::Compilation(
            "Probabilistic source and canonical atoms do not correspond".to_string(),
        ));
    }

    let schema = schemas.get(&canonical.predicate).ok_or_else(|| {
        XlogError::Compilation(format!(
            "Probabilistic presentation requires a schema for predicate '{}'",
            canonical.predicate
        ))
    })?;
    let mut atom = atom_key_from_ground_atom(canonical)?;
    for (index, (value, source_term)) in atom.args.iter_mut().zip(&source.terms).enumerate() {
        if schema.column_type(index) != Some(ScalarType::Symbol)
            || !matches!(value, Value::Symbol(_))
        {
            continue;
        }
        match source_term {
            Term::String(source) => *value = Value::String(source.clone()),
            Term::Symbol(source) => *value = Value::Symbol(*source),
            _ => {}
        }
    }
    Ok(atom)
}

pub fn extract_from_program(program: &Program) -> Result<Provenance> {
    // Stratify first to fail fast on unsupported recursion patterns.
    let _ = stratify(program)?;
    let source_program = program;
    let schemas = arithmetic_schemas(program)?;
    let canonical_program = canonicalize_probabilistic_program(program, &schemas)?;
    let program = &canonical_program;

    let mut builder = PirBuilder::new();

    let mut leaf_probs: BTreeMap<LeafId, f64> = BTreeMap::new();
    let mut choice_probs: BTreeMap<ChoiceVarId, (f64, f64)> = BTreeMap::new();
    let mut leaf_atoms: BTreeMap<LeafId, GroundAtom> = BTreeMap::new();
    let mut choice_sources: BTreeMap<ChoiceVarId, ChoiceSource> = BTreeMap::new();
    let mut aggregate_lifting: Vec<AggregateLiftReport> = Vec::new();

    let mut store: BTreeMap<String, Relation> = BTreeMap::new();

    // Deterministic facts.
    for fact in program.facts() {
        let key = atom_key_from_ground_atom(&fact.head)?;
        let rel = store
            .entry(key.predicate.clone())
            .or_insert_with(Relation::new);
        rel.insert_or(key.args.clone(), builder.const_true(), &mut builder);
    }

    // Probabilistic facts.
    let mut next_leaf: u32 = 0;
    for (pf, source_pf) in program.prob_facts.iter().zip(&source_program.prob_facts) {
        validate_prob(pf.prob, "probabilistic fact")?;
        let key = atom_key_from_ground_atom(&pf.atom)?;
        let leaf = LeafId::new(next_leaf);
        next_leaf = next_leaf.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("probabilistic fact leaf id overflow".to_string())
        })?;
        leaf_probs.insert(leaf, pf.prob);
        leaf_atoms.insert(
            leaf,
            presentation_atom_from_canonical(&source_pf.atom, &pf.atom, &schemas)?,
        );

        let rel = store
            .entry(key.predicate.clone())
            .or_insert_with(Relation::new);
        rel.insert_or(key.args.clone(), builder.lit(leaf), &mut builder);
    }

    // Annotated disjunctions: lower to a chain of Bernoulli decisions.
    let mut next_choice: u32 = 0;
    for (ad, source_ad) in program
        .annotated_disjunctions
        .iter()
        .zip(&source_program.annotated_disjunctions)
    {
        if ad.choices.is_empty() {
            return Err(XlogError::Compilation(
                "Annotated disjunction must contain at least one choice".to_string(),
            ));
        }
        let (vars, outcome_formulas) = compile_annotated_disjunction(
            ad,
            source_ad,
            &schemas,
            &mut next_choice,
            &mut choice_probs,
            &mut choice_sources,
            &mut builder,
        )?;
        let _ = vars;

        for (pf, formula) in ad.choices.iter().zip(outcome_formulas) {
            let key = atom_key_from_ground_atom(&pf.atom)?;
            let rel = store
                .entry(key.predicate.clone())
                .or_insert_with(Relation::new);
            rel.insert_or(key.args.clone(), formula, &mut builder);
        }
    }

    // Evaluate rules SCC-by-SCC (semi-naive for recursive SCCs).
    let graph = build_dependency_graph(program);
    for pred in &graph.predicates {
        store.entry(pred.clone()).or_insert_with(Relation::new);
    }

    // Use analyze_stratification to detect non-monotone SCCs
    let strat_result = analyze_stratification(program);
    let sccs = find_sccs_for_lowering(&graph);

    // Build a set of SCC indices that are non-monotone
    // We need to map the SCCs from find_sccs_for_lowering to analyze_stratification
    // Both use the same SCC algorithm, so indices should match
    let non_monotone_scc_preds: std::collections::HashSet<String> = strat_result
        .sccs
        .iter()
        .enumerate()
        .filter(|(i, _)| strat_result.non_monotone_sccs.contains(i))
        .flat_map(|(_, scc)| scc.iter().cloned())
        .collect();

    let mut rules_by_head: BTreeMap<String, Vec<Rule>> = BTreeMap::new();
    for rule in program.proper_rules() {
        // Note: Negation is now supported via stratified evaluation and negate_provenance()
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

        // Check if any predicate in this SCC is in a non-monotone cycle
        let is_non_monotone = scc.iter().any(|p| non_monotone_scc_preds.contains(p));

        if is_non_monotone {
            // Use WFS for non-monotone SCCs (cycles through negation)
            eval_non_monotone_scc_with_wfs(&scc, &scc_rules, &mut store, &mut builder, &schemas)?;
        } else {
            let recursive = is_recursive_scc(&scc, &scc_rules);
            if recursive {
                eval_recursive_scc(
                    &scc,
                    &scc_rules,
                    &mut store,
                    &mut builder,
                    &mut aggregate_lifting,
                    &schemas,
                )?;
            } else {
                eval_non_recursive_scc(
                    &scc_rules,
                    &mut store,
                    &mut builder,
                    &mut aggregate_lifting,
                    &schemas,
                )?;
            }
        }
    }

    // Snapshot tuple formulas.
    let mut tuple_formulas: BTreeMap<GroundAtom, PirNodeId> = BTreeMap::new();
    for (pred, rel) in &store {
        for (tuple, formula) in &rel.tuples {
            tuple_formulas.insert(GroundAtom::new(pred.clone(), tuple.clone()), *formula);
        }
    }

    let mut queries: Vec<GroundAtom> = Vec::new();
    for (ProbQuery { atom }, ProbQuery { atom: source_atom }) in program
        .prob_queries
        .iter()
        .zip(&source_program.prob_queries)
    {
        queries.push(presentation_atom_from_canonical(
            source_atom,
            atom,
            &schemas,
        )?);
    }

    let mut evidence: Vec<(GroundAtom, bool)> = Vec::new();
    for (
        Evidence { atom, value },
        Evidence {
            atom: source_atom, ..
        },
    ) in program.evidence.iter().zip(&source_program.evidence)
    {
        evidence.push((
            presentation_atom_from_canonical(source_atom, atom, &schemas)?,
            *value,
        ));
    }

    Ok(Provenance {
        pir: builder.finish(),
        leaf_probs,
        choice_probs,
        tuple_formulas,
        queries,
        evidence,
        leaf_atoms,
        choice_sources,
        aggregate_lifting,
        schemas,
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
        Term::Symbol(id) => Ok(Value::Symbol(*id)),
        Term::Variable(_) | Term::Anonymous | Term::Aggregate(_) => Err(XlogError::Compilation(
            "Non-constant term cannot be converted to a value".to_string(),
        )),
        Term::List(_) => Err(unsupported_probabilistic_term_error(
            "value conversion",
            "list",
        )),
        Term::Cons { .. } => Err(unsupported_probabilistic_term_error(
            "value conversion",
            "cons",
        )),
        Term::Compound { .. } => Err(unsupported_probabilistic_term_error(
            "value conversion",
            "compound",
        )),
        Term::PredRef(_) => Err(unsupported_probabilistic_term_error(
            "value conversion",
            "predref",
        )),
    }
}

fn unsupported_probabilistic_term_error(context: &str, kind: &str) -> XlogError {
    XlogError::Compilation(format!(
        "high-level term form '{}' is parsed but not supported in probabilistic provenance {} until a lowering/materialization path exists",
        kind, context
    ))
}

fn compile_annotated_disjunction(
    ad: &xlog_logic::ast::AnnotatedDisjunction,
    source_ad: &xlog_logic::ast::AnnotatedDisjunction,
    schemas: &HashMap<String, Schema>,
    next_choice: &mut u32,
    choice_probs: &mut BTreeMap<ChoiceVarId, (f64, f64)>,
    choice_sources: &mut BTreeMap<ChoiceVarId, ChoiceSource>,
    builder: &mut PirBuilder,
) -> Result<(Vec<ChoiceVarId>, Vec<PirNodeId>)> {
    for pf in &ad.choices {
        validate_prob(pf.prob, "annotated disjunction choice")?;
        let _ = atom_key_from_ground_atom(&pf.atom)?;
    }

    // Built once per disjunction and shared (Arc) across all m-1 chain variables
    // below, instead of being deep-cloned per variable (which would be O(k^2)
    // GroundAtom clones for a k-head disjunction).
    let explicit_choices: Arc<[(GroundAtom, f64)]> = ad
        .choices
        .iter()
        .zip(&source_ad.choices)
        .map(|(pf, source_pf)| {
            presentation_atom_from_canonical(&source_pf.atom, &pf.atom, schemas)
                .map(|atom| (atom, pf.prob))
        })
        .collect::<Result<Vec<_>>>()?
        .into();

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
    for (i, &p_i) in probs.iter().enumerate().take(m - 1) {
        let cond_true = if remaining <= 0.0 {
            0.0
        } else {
            p_i / remaining
        };
        validate_prob(cond_true, "annotated disjunction conditional")?;
        let cond_false = 1.0 - cond_true;
        let var = ChoiceVarId::new(*next_choice);
        *next_choice = (*next_choice).checked_add(1).ok_or_else(|| {
            XlogError::Compilation("annotated disjunction choice id overflow".to_string())
        })?;
        vars.push(var);
        choice_probs.insert(var, (cond_true, cond_false));
        choice_sources.insert(
            var,
            ChoiceSource {
                choices: explicit_choices.clone(),
                choice_index: i,
                source_id: None,
            },
        );
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
    store: &mut BTreeMap<String, Relation>,
    builder: &mut PirBuilder,
    aggregate_lifting: &mut Vec<AggregateLiftReport>,
    schemas: &HashMap<String, Schema>,
) -> Result<()> {
    for rule in rules {
        let derived = eval_rule(
            rule,
            store,
            &BTreeMap::new(),
            None,
            builder,
            aggregate_lifting,
            schemas,
        )?;
        let rel = store
            .entry(rule.head.predicate.clone())
            .or_insert_with(Relation::new);
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
    store: &mut BTreeMap<String, Relation>,
    builder: &mut PirBuilder,
    aggregate_lifting: &mut Vec<AggregateLiftReport>,
    schemas: &HashMap<String, Schema>,
) -> Result<()> {
    let scc_set: std::collections::HashSet<&str> = scc.iter().map(|s| s.as_str()).collect();

    // Snapshot full relations for the SCC.
    let mut full: BTreeMap<String, Relation> = BTreeMap::new();
    for pred in scc {
        let rel = store.get(pred).cloned().unwrap_or_else(Relation::new);
        full.insert(pred.clone(), rel);
    }

    // Seed: evaluate all rules once against the current full snapshot.
    let mut delta: BTreeMap<String, Relation> = BTreeMap::new();
    for rule in rules {
        let derived = eval_rule(
            rule,
            store,
            &full,
            None,
            builder,
            aggregate_lifting,
            schemas,
        )?;
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
                        let non_empty =
                            delta_prev.get(pred).map(|r| !r.is_empty()).unwrap_or(false);
                        non_empty.then_some(i)
                    }
                    _ => None,
                })
                .collect();
            if body_indices.is_empty() {
                continue;
            }

            let mut derived_all: BTreeMap<Vec<Value>, PirNodeId> = BTreeMap::new();
            for idx in body_indices {
                let derived = eval_rule(
                    rule,
                    store,
                    &full_prev,
                    Some((idx, &delta_prev)),
                    builder,
                    aggregate_lifting,
                    schemas,
                )?;
                for (tuple, proof) in derived {
                    let entry = derived_all
                        .entry(tuple)
                        .or_insert_with(|| builder.const_false());
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

/// Evaluate a non-monotone SCC using Well-Founded Semantics.
///
/// This function handles SCCs that have cycles through negation. It:
/// 1. Grounds the rules by enumerating all variable bindings from existing tuples
/// 2. Converts ground rules to WFS rules
/// 3. Calls WFS to compute the well-founded model
/// 4. Stores the results (true atoms with provenance) back
///
/// Undefined atoms (those in a true cycle) get no provenance (probability 0).
fn eval_non_monotone_scc_with_wfs(
    scc: &[String],
    rules: &[Rule],
    store: &mut BTreeMap<String, Relation>,
    builder: &mut PirBuilder,
    schemas: &HashMap<String, Schema>,
) -> Result<()> {
    let scc_set: std::collections::HashSet<&str> = scc.iter().map(|s| s.as_str()).collect();

    // Step 1: Ground all rules in the SCC
    // We enumerate all possible variable bindings by iterating over existing tuples
    let mut wfs_rules: Vec<WfsRule> = Vec::new();

    for rule in rules {
        // Ground this rule against the current store
        let grounded = ground_rule_for_wfs(rule, store, &scc_set, builder, schemas)?;
        wfs_rules.extend(grounded);
    }

    if wfs_rules.is_empty() {
        // No ground rules, nothing to do
        return Ok(());
    }

    // Step 2: Call WFS to compute the well-founded model
    let wfs_result = evaluate_wfs_rules(&wfs_rules, &mut builder.pir, &WfsConfig::default())?;

    // Step 3: Store the results back
    // True atoms get their provenance, false/undefined atoms are not added
    for (wfs_atom, prov) in wfs_result.true_set {
        let args = canonicalize_public_values(&wfs_atom.predicate, &wfs_atom.args, schemas)?;
        let rel = store
            .entry(wfs_atom.predicate.clone())
            .or_insert_with(Relation::new);
        rel.insert_or(args, prov, builder);
    }

    Ok(())
}

/// Ground a rule for WFS evaluation.
///
/// This generates all ground instances of a rule by iterating over existing tuples
/// that match the body literals (excluding SCC predicates which are handled by WFS).
fn ground_rule_for_wfs(
    rule: &Rule,
    store: &BTreeMap<String, Relation>,
    scc_set: &std::collections::HashSet<&str>,
    builder: &mut PirBuilder,
    schemas: &HashMap<String, Schema>,
) -> Result<Vec<WfsRule>> {
    // Start with empty binding
    let mut bindings: Vec<ProvenanceEvaluationState> =
        vec![(HashMap::new(), HashMap::new(), builder.const_true())];

    // Collect body literals that are in the SCC (will become WFS body literals)
    // and non-SCC literals (will be grounded now)
    let mut wfs_body_template: Vec<(usize, bool)> = Vec::new(); // (body_index, is_positive)

    for (idx, lit) in rule.body.iter().enumerate() {
        match lit {
            BodyLiteral::Positive(atom) => {
                if scc_set.contains(atom.predicate.as_str()) {
                    // This will become a WFS body literal
                    wfs_body_template.push((idx, true));
                } else {
                    // Ground now by iterating over existing tuples
                    let rel = store.get(&atom.predicate);
                    let mut next_bindings = Vec::new();

                    for (binding, arithmetic_bindings, prov) in bindings {
                        if let Some(rel) = rel {
                            for (tuple, tuple_prov) in &rel.tuples {
                                let mut new_binding = binding.clone();
                                if unify_atom(atom, tuple, &mut new_binding)? {
                                    let mut new_arithmetic_bindings = arithmetic_bindings.clone();
                                    extend_arithmetic_bindings(
                                        atom,
                                        tuple,
                                        schemas,
                                        &mut new_arithmetic_bindings,
                                    )?;
                                    let new_prov = builder.and(vec![prov, *tuple_prov]);
                                    next_bindings.push((
                                        new_binding,
                                        new_arithmetic_bindings,
                                        new_prov,
                                    ));
                                }
                            }
                        }
                        // If relation doesn't exist, no tuples match
                    }
                    bindings = next_bindings;
                    if bindings.is_empty() {
                        return Ok(Vec::new());
                    }
                }
            }
            BodyLiteral::Negated(atom) => {
                if scc_set.contains(atom.predicate.as_str()) {
                    // This will become a WFS negative body literal
                    wfs_body_template.push((idx, false));
                } else {
                    // Ground now: negation of non-SCC predicate
                    let rel = store.get(&atom.predicate);
                    let mut next_bindings = Vec::new();

                    for (binding, arithmetic_bindings, prov) in bindings {
                        // Check if all variables in the negated atom are bound
                        let all_bound = atom.terms.iter().all(|t| match t {
                            Term::Variable(v) => binding.contains_key(v),
                            _ => true,
                        });

                        if !all_bound {
                            // Skip unsafe negation
                            continue;
                        }

                        if let Some(rel) = rel {
                            // Collect matching tuples
                            let mut matching_provs: Vec<PirNodeId> = Vec::new();
                            for (tuple, tuple_prov) in &rel.tuples {
                                let mut test_binding = binding.clone();
                                if unify_atom(atom, tuple, &mut test_binding)? {
                                    matching_provs.push(*tuple_prov);
                                }
                            }

                            if matching_provs.is_empty() {
                                // No matches - closed world: negation succeeds
                                next_bindings.push((binding, arithmetic_bindings, prov));
                            } else {
                                // Negate the combined provenance
                                let combined = builder.or(matching_provs);
                                let neg_prov = negate_provenance(combined, builder);
                                let new_prov = builder.and(vec![prov, neg_prov]);
                                next_bindings.push((binding, arithmetic_bindings, new_prov));
                            }
                        } else {
                            // Relation doesn't exist - closed world: negation succeeds
                            next_bindings.push((binding, arithmetic_bindings, prov));
                        }
                    }
                    bindings = next_bindings;
                    if bindings.is_empty() {
                        return Ok(Vec::new());
                    }
                }
            }
            BodyLiteral::Epistemic(lit) => {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "probabilistic WFS grounding".to_string(),
                    context: format!("{:?} {}({})", lit.op, lit.atom.predicate, lit.atom.arity()),
                });
            }
            BodyLiteral::Comparison(cmp) => {
                let mut next_bindings = Vec::new();
                for (binding, arithmetic_bindings, prov) in bindings {
                    if eval_comparison_with_arithmetic_bindings(
                        cmp.op,
                        &cmp.left,
                        &cmp.right,
                        &binding,
                        &arithmetic_bindings,
                    )? {
                        next_bindings.push((binding, arithmetic_bindings, prov));
                    }
                }
                bindings = next_bindings;
                if bindings.is_empty() {
                    return Ok(Vec::new());
                }
            }
            BodyLiteral::IsExpr(is_expr) => {
                let mut next_bindings = Vec::new();
                for (mut binding, mut arithmetic_bindings, prov) in bindings {
                    let arithmetic_value =
                        eval_arithmetic_value(&is_expr.expr, &binding, &arithmetic_bindings)?;
                    bind_arithmetic_result(
                        &is_expr.target,
                        arithmetic_value,
                        &mut binding,
                        &mut arithmetic_bindings,
                    )?;
                    next_bindings.push((binding, arithmetic_bindings, prov));
                }
                bindings = next_bindings;
                if bindings.is_empty() {
                    return Ok(Vec::new());
                }
            }
            BodyLiteral::Univ(_) => {
                return Err(XlogError::Compilation(
                    "univ literal was not normalized before provenance extraction".to_string(),
                ));
            }
        }
    }

    // Now create WFS rules for each binding
    let mut result: Vec<WfsRule> = Vec::new();

    for (binding, _, external_prov) in bindings {
        // Build the WFS body from SCC literals
        let mut wfs_body: Vec<WfsLiteral> = Vec::new();

        for &(idx, is_positive) in &wfs_body_template {
            let atom = match &rule.body[idx] {
                BodyLiteral::Positive(a) | BodyLiteral::Negated(a) => a,
                _ => continue,
            };

            // Ground the atom with the current binding
            let mut args: Vec<Value> = Vec::new();
            for term in &atom.terms {
                match term {
                    Term::Variable(name) => {
                        if let Some(v) = binding.get(name) {
                            args.push(v.clone());
                        } else {
                            // Variable not bound - this shouldn't happen for well-formed rules
                            // Skip this ground instance
                            continue;
                        }
                    }
                    _ => {
                        args.push(value_from_term(term)?);
                    }
                }
            }

            let wfs_atom = WfsAtom::new(atom.predicate.clone(), args);
            if is_positive {
                wfs_body.push(WfsLiteral::Positive(wfs_atom));
            } else {
                wfs_body.push(WfsLiteral::Negative(wfs_atom));
            }
        }

        // Build the ground head
        let mut head_args: Vec<Value> = Vec::new();
        for term in &rule.head.terms {
            match term {
                Term::Variable(name) => {
                    if let Some(v) = binding.get(name) {
                        head_args.push(v.clone());
                    } else {
                        // Unbound head variable - skip this instance
                        continue;
                    }
                }
                _ => {
                    head_args.push(value_from_term(term)?);
                }
            }
        }

        let wfs_head = WfsAtom::new(rule.head.predicate.clone(), head_args);
        result.push(WfsRule::new(wfs_head, wfs_body, external_prov));
    }

    Ok(result)
}

/// Negate a provenance formula, pushing negation to leaves (NNF form).
///
/// This implements the logical negation of a provenance formula by applying De Morgan's laws
/// to push negations down to the leaves. At the leaf level:
/// - `Lit { leaf }` becomes `NegLit { leaf }` (negated probabilistic fact)
/// - `NegLit { leaf }` becomes `Lit { leaf }` (double negation elimination)
/// - `Const(true)` becomes `Const(false)` and vice versa
fn negate_provenance(prov: PirNodeId, builder: &mut PirBuilder) -> PirNodeId {
    use crate::pir::PirNode;
    match builder.pir.node(prov).cloned() {
        Some(PirNode::Const(b)) => {
            if b {
                builder.const_false()
            } else {
                builder.const_true()
            }
        }
        Some(PirNode::Lit { leaf }) => builder.neg_lit(leaf),
        Some(PirNode::NegLit { leaf }) => builder.lit(leaf), // Double negation elimination
        Some(PirNode::And { children }) => {
            // De Morgan: not(A and B) = (not A) or (not B)
            let neg_children: Vec<PirNodeId> = children
                .iter()
                .map(|&c| negate_provenance(c, builder))
                .collect();
            builder.or(neg_children)
        }
        Some(PirNode::Or { children }) => {
            // De Morgan: not(A or B) = (not A) and (not B)
            let neg_children: Vec<PirNodeId> = children
                .iter()
                .map(|&c| negate_provenance(c, builder))
                .collect();
            builder.and(neg_children)
        }
        Some(PirNode::Decision {
            var,
            child_false,
            child_true,
        }) => {
            // Negate both branches
            let neg_false = negate_provenance(child_false, builder);
            let neg_true = negate_provenance(child_true, builder);
            builder.decision(var, neg_false, neg_true)
        }
        None => prov,
    }
}

/// Evaluate a single rule and produce a map from head tuples to proof formulas.
///
/// `full_scc` is the per-SCC snapshot for recursive predicates; `delta_scc` is optional and
/// provides a delta relation for a specific body literal index.
fn eval_rule(
    rule: &Rule,
    global: &BTreeMap<String, Relation>,
    full_scc: &BTreeMap<String, Relation>,
    delta_scc: Option<(usize, &BTreeMap<String, Relation>)>,
    builder: &mut PirBuilder,
    aggregate_lifting: &mut Vec<AggregateLiftReport>,
    schemas: &HashMap<String, Schema>,
) -> Result<BTreeMap<Vec<Value>, PirNodeId>> {
    let mut states: Vec<ProvenanceEvaluationState> =
        vec![(HashMap::new(), HashMap::new(), builder.const_true())];

    for (idx, lit) in rule.body.iter().enumerate() {
        let mut next_states = Vec::new();
        match lit {
            BodyLiteral::Positive(atom) => {
                let rel = select_relation(atom, idx, global, full_scc, delta_scc)?;
                for (binding, arithmetic_bindings, prov) in states {
                    for (tuple, tuple_prov) in &rel.tuples {
                        let mut binding2 = binding.clone();
                        if unify_atom(atom, tuple, &mut binding2)? {
                            let mut arithmetic_bindings2 = arithmetic_bindings.clone();
                            extend_arithmetic_bindings(
                                atom,
                                tuple,
                                schemas,
                                &mut arithmetic_bindings2,
                            )?;
                            let prov2 = builder.and(vec![prov, *tuple_prov]);
                            next_states.push((binding2, arithmetic_bindings2, prov2));
                        }
                    }
                }
            }
            BodyLiteral::Comparison(cmp) => {
                for (binding, arithmetic_bindings, prov) in states {
                    if eval_comparison_with_arithmetic_bindings(
                        cmp.op,
                        &cmp.left,
                        &cmp.right,
                        &binding,
                        &arithmetic_bindings,
                    )? {
                        next_states.push((binding, arithmetic_bindings, prov));
                    }
                }
            }
            BodyLiteral::IsExpr(is_expr) => {
                for (mut binding, mut arithmetic_bindings, prov) in states {
                    let arithmetic_value =
                        eval_arithmetic_value(&is_expr.expr, &binding, &arithmetic_bindings)?;
                    bind_arithmetic_result(
                        &is_expr.target,
                        arithmetic_value,
                        &mut binding,
                        &mut arithmetic_bindings,
                    )?;
                    next_states.push((binding, arithmetic_bindings, prov));
                }
            }
            BodyLiteral::Negated(atom) => {
                // Stratified negation: for each binding, check if any matching tuple exists.
                // - If a matching tuple exists with provenance P, the negation has provenance "not P"
                // - If no matching tuple exists, the negation succeeds trivially (closed-world assumption)
                //
                // For negated literals, we only use the global store and full_scc snapshot,
                // never the delta (negation is evaluated against the complete relation).
                let rel = if let Some(r) = full_scc.get(&atom.predicate) {
                    r
                } else if let Some(r) = global.get(&atom.predicate) {
                    r
                } else {
                    // Predicate not found - closed world assumption: all negations succeed
                    for (binding, arithmetic_bindings, prov) in states {
                        // Ensure all variables in the negated atom are bound
                        let all_bound = atom.terms.iter().all(|t| match t {
                            Term::Variable(v) => binding.contains_key(v),
                            _ => true,
                        });
                        if all_bound {
                            next_states.push((binding, arithmetic_bindings, prov));
                        }
                    }
                    states = next_states;
                    if states.is_empty() {
                        break;
                    }
                    continue;
                };

                for (binding, arithmetic_bindings, prov) in states {
                    // First, check if all variables in the negated atom are bound.
                    // Negation requires all variables to be bound (safety condition).
                    let all_bound = atom.terms.iter().all(|t| match t {
                        Term::Variable(v) => binding.contains_key(v),
                        _ => true,
                    });
                    if !all_bound {
                        // Skip this binding - variables must be bound before negation
                        continue;
                    }

                    // Collect matching tuples and their provenances
                    let mut matching_provs: Vec<PirNodeId> = Vec::new();
                    for (tuple, tuple_prov) in &rel.tuples {
                        let mut binding2 = binding.clone();
                        if unify_atom(atom, tuple, &mut binding2)? {
                            // A match was found; we need its negated provenance
                            matching_provs.push(*tuple_prov);
                        }
                    }

                    if matching_provs.is_empty() {
                        // No matching tuples - closed world assumption: negation succeeds trivially
                        next_states.push((binding, arithmetic_bindings, prov));
                    } else {
                        // For negation to succeed, ALL matching tuples must be "absent" (negated).
                        // If tuple can exist via multiple provenances (disjunction), we negate that.
                        // Negation of (proof_a or proof_b or ...) =
                        // (not proof_a) and (not proof_b) and ...
                        let combined_tuple_prov = builder.or(matching_provs);
                        let neg_prov = negate_provenance(combined_tuple_prov, builder);
                        let new_prov = builder.and(vec![prov, neg_prov]);
                        next_states.push((binding, arithmetic_bindings, new_prov));
                    }
                }
            }
            BodyLiteral::Epistemic(lit) => {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "probabilistic provenance evaluation".to_string(),
                    context: format!("{:?} {}({})", lit.op, lit.atom.predicate, lit.atom.arity()),
                });
            }
            BodyLiteral::Univ(_) => {
                return Err(XlogError::Compilation(
                    "univ literal was not normalized before provenance extraction".to_string(),
                ));
            }
        }
        states = next_states;
        if states.is_empty() {
            break;
        }
    }

    let states = states
        .into_iter()
        .map(|(binding, _, provenance)| (binding, provenance))
        .collect::<Vec<_>>();
    let derived = if rule.has_aggregation() {
        eval_aggregate_head_provenance(&rule.head, states, builder, aggregate_lifting)?
    } else {
        let mut out: BTreeMap<Vec<Value>, PirNodeId> = BTreeMap::new();
        for (binding, prov) in states {
            let head_tuple = materialize_head(&rule.head, &binding)?;
            let entry = out
                .entry(head_tuple)
                .or_insert_with(|| builder.const_false());
            *entry = builder.or(vec![*entry, prov]);
        }
        out
    };

    let mut canonical = BTreeMap::new();
    for (tuple, provenance) in derived {
        let tuple = canonicalize_public_values(&rule.head.predicate, &tuple, schemas)?;
        let entry = canonical
            .entry(tuple)
            .or_insert_with(|| builder.const_false());
        *entry = builder.or(vec![*entry, provenance]);
    }
    Ok(canonical)
}

const MAX_EXACT_PROB_AGG_UNCERTAIN_ROWS: usize = 16;
const MAX_EXACT_PROB_COUNT_LIFT_ROWS: usize = 64;

#[derive(Debug, Clone)]
struct AggregateProvRow {
    binding: HashMap<String, Value>,
    prov: PirNodeId,
}

fn eval_aggregate_head_provenance(
    head: &Atom,
    states: Vec<(HashMap<String, Value>, PirNodeId)>,
    builder: &mut PirBuilder,
    aggregate_lifting: &mut Vec<AggregateLiftReport>,
) -> Result<BTreeMap<Vec<Value>, PirNodeId>> {
    let (key_vars, key_var_to_pos, agg_specs, agg_to_pos) = aggregate_head_plan(head)?;

    let mut deduped_states: BTreeMap<Vec<(String, Value)>, AggregateProvRow> = BTreeMap::new();
    for (binding, prov) in states {
        let key = canonical_binding_key(&binding);
        match deduped_states.get_mut(&key) {
            Some(row) => {
                row.prov = builder.or(vec![row.prov, prov]);
            }
            None => {
                deduped_states.insert(key, AggregateProvRow { binding, prov });
            }
        }
    }

    #[derive(Debug)]
    struct GroupRows {
        key: Vec<Value>,
        rows: Vec<AggregateProvRow>,
    }

    let mut groups: BTreeMap<Vec<Value>, GroupRows> = BTreeMap::new();
    for row in deduped_states.into_values() {
        let mut key: Vec<Value> = Vec::with_capacity(key_vars.len());
        for name in &key_vars {
            let v = row
                .binding
                .get(name)
                .ok_or_else(|| XlogError::UnsafeVariable(name.clone()))?;
            key.push(v.clone());
        }
        groups
            .entry(key.clone())
            .or_insert_with(|| GroupRows {
                key,
                rows: Vec::new(),
            })
            .rows
            .push(row);
    }

    let mut out: BTreeMap<Vec<Value>, PirNodeId> = BTreeMap::new();
    let count_only = agg_specs.iter().all(|(op, _)| *op == AggOp::Count);
    for group in groups.into_values() {
        let mut always_rows: Vec<AggregateProvRow> = Vec::new();
        let mut uncertain_rows: Vec<AggregateProvRow> = Vec::new();
        for row in group.rows {
            match pir_const_value(builder, row.prov) {
                Some(true) => always_rows.push(row),
                Some(false) => {}
                None => uncertain_rows.push(row),
            }
        }

        if always_rows.is_empty() && uncertain_rows.is_empty() {
            continue;
        }
        if count_only {
            if uncertain_rows.len() > MAX_EXACT_PROB_COUNT_LIFT_ROWS {
                return Err(XlogError::Compilation(format!(
                    "count aggregate lifting finite domain cap exceeded for predicate {} group {:?}: {} uncertain rows > cap {}; use prob_engine = mc or reduce the finite aggregate domain",
                    head.predicate,
                    group.key,
                    uncertain_rows.len(),
                    MAX_EXACT_PROB_COUNT_LIFT_ROWS
                )));
            }
            validate_count_lift_rows(&agg_specs, &always_rows, &uncertain_rows)?;
            record_aggregate_lift_reports(
                aggregate_lifting,
                head,
                &group.key,
                &agg_specs,
                always_rows.len(),
                uncertain_rows.len(),
                AggregateLiftStatus::Fired,
                "finite count domain lifted with exact cardinality dynamic programming",
                MAX_EXACT_PROB_COUNT_LIFT_ROWS,
                count_lift_dp_states(uncertain_rows.len()),
            );
            let count_formulas = count_lift_formulas(&uncertain_rows, builder);
            for (selected_uncertain_rows, proof) in count_formulas.into_iter().enumerate() {
                if always_rows.is_empty() && selected_uncertain_rows == 0 {
                    continue;
                }
                let count_value = always_rows.len() + selected_uncertain_rows;
                let tuple =
                    materialize_count_lift_tuple(head, &group.key, &key_var_to_pos, count_value)?;
                let entry = out.entry(tuple).or_insert_with(|| builder.const_false());
                *entry = builder.or(vec![*entry, proof]);
            }
            continue;
        }

        if uncertain_rows.len() > MAX_EXACT_PROB_AGG_UNCERTAIN_ROWS {
            return Err(XlogError::Compilation(format!(
                "exact probabilistic aggregate domain cap exceeded for predicate {} group {:?}: {} uncertain rows > cap {}; use prob_engine = mc or reduce the finite aggregate domain",
                head.predicate,
                group.key,
                uncertain_rows.len(),
                MAX_EXACT_PROB_AGG_UNCERTAIN_ROWS
            )));
        }
        let (outcomes, dp_states) =
            factorized_aggregate_outcomes(&agg_specs, &always_rows, &uncertain_rows, builder)?;
        record_aggregate_lift_reports(
            aggregate_lifting,
            head,
            &group.key,
            &agg_specs,
            always_rows.len(),
            uncertain_rows.len(),
            AggregateLiftStatus::Fired,
            "finite outcome domain folded with factorized aggregate-state dynamic programming",
            MAX_EXACT_PROB_AGG_UNCERTAIN_ROWS,
            dp_states,
        );

        for (agg_states, selected_any, proof) in outcomes {
            if always_rows.is_empty() && !selected_any {
                // No deterministic rows and no uncertain row selected: the group
                // is empty in this outcome, so no head tuple materializes.
                continue;
            }

            let tuple = materialize_aggregate_tuple(
                head,
                &group.key,
                &key_var_to_pos,
                &agg_specs,
                &agg_to_pos,
                &agg_states,
            )?;
            let entry = out.entry(tuple).or_insert_with(|| builder.const_false());
            *entry = builder.or(vec![*entry, proof]);
        }
    }

    Ok(out)
}

/// Factorized aggregate-outcome folding for non-count exact aggregates.
///
/// Instead of enumerating all `2^k` present/absent masks over the `k` uncertain
/// rows (one conjunctive PIR formula per mask), fold the rows one at a time
/// through a dynamic program keyed by the aggregate state reached so far.
/// Outcomes that agree on the aggregate state share one PIR sub-DAG, so the
/// emitted PIR is `O(k * #distinct-states)` instead of `O(2^k)` formulas.
///
/// Rows are folded in the same order as the previous mask enumeration
/// (deterministic rows first, then uncertain rows in index order), so every
/// outcome value is bit-identical to the enumerated result and the union of
/// worlds reaching each outcome is unchanged (identical query probabilities).
///
/// Returns the folded outcomes as `(aggregate states, any-uncertain-row-selected,
/// proof formula)` triples plus the total number of DP states visited.
#[allow(clippy::type_complexity)]
fn factorized_aggregate_outcomes(
    agg_specs: &[(AggOp, String)],
    always_rows: &[AggregateProvRow],
    uncertain_rows: &[AggregateProvRow],
    builder: &mut PirBuilder,
) -> Result<(Vec<(Vec<AggState>, bool, PirNodeId)>, usize)> {
    use std::collections::btree_map::Entry;

    fn states_key(states: &[AggState]) -> Vec<AggStateKey> {
        states.iter().map(AggState::dp_key).collect()
    }

    let mut base: Vec<AggState> = agg_specs.iter().map(|(op, _)| AggState::new(*op)).collect();
    for row in always_rows {
        update_aggregate_states(&mut base, agg_specs, row)?;
    }

    let mut dp: BTreeMap<(Vec<AggStateKey>, bool), (Vec<AggState>, PirNodeId)> = BTreeMap::new();
    let true_proof = builder.const_true();
    dp.insert((states_key(&base), false), (base, true_proof));
    let mut dp_states = dp.len();

    for row in uncertain_rows {
        let absent = negate_provenance(row.prov, builder);
        let mut next: BTreeMap<(Vec<AggStateKey>, bool), (Vec<AggState>, PirNodeId)> =
            BTreeMap::new();
        for ((key, selected_any), (states, proof)) in dp {
            let mut present_states = states.clone();
            update_aggregate_states(&mut present_states, agg_specs, row)?;
            let present_key = states_key(&present_states);
            let present_proof = builder.and(vec![proof, row.prov]);
            match next.entry((present_key, true)) {
                Entry::Occupied(mut entry) => {
                    entry.get_mut().1 = builder.or(vec![entry.get().1, present_proof]);
                }
                Entry::Vacant(entry) => {
                    entry.insert((present_states, present_proof));
                }
            }

            let absent_proof = builder.and(vec![proof, absent]);
            match next.entry((key, selected_any)) {
                Entry::Occupied(mut entry) => {
                    entry.get_mut().1 = builder.or(vec![entry.get().1, absent_proof]);
                }
                Entry::Vacant(entry) => {
                    entry.insert((states, absent_proof));
                }
            }
        }
        dp = next;
        dp_states += dp.len();
    }

    let outcomes = dp
        .into_iter()
        .map(|((_, selected_any), (states, proof))| (states, selected_any, proof))
        .collect();
    Ok((outcomes, dp_states))
}

fn validate_count_lift_rows(
    agg_specs: &[(AggOp, String)],
    always_rows: &[AggregateProvRow],
    uncertain_rows: &[AggregateProvRow],
) -> Result<()> {
    for (_, var) in agg_specs {
        for row in always_rows.iter().chain(uncertain_rows.iter()) {
            if !row.binding.contains_key(var) {
                return Err(XlogError::UnsafeVariable(var.clone()));
            }
        }
    }
    Ok(())
}

fn count_lift_formulas(
    uncertain_rows: &[AggregateProvRow],
    builder: &mut PirBuilder,
) -> Vec<PirNodeId> {
    let n = uncertain_rows.len();
    let mut dp = vec![builder.const_false(); n + 1];
    dp[0] = builder.const_true();

    for (idx, row) in uncertain_rows.iter().enumerate() {
        let mut next = vec![builder.const_false(); n + 1];
        let present = row.prov;
        let absent = negate_provenance(row.prov, builder);
        for selected in 0..=idx {
            let absent_case = builder.and(vec![dp[selected], absent]);
            next[selected] = builder.or(vec![next[selected], absent_case]);

            let present_case = builder.and(vec![dp[selected], present]);
            next[selected + 1] = builder.or(vec![next[selected + 1], present_case]);
        }
        dp = next;
    }

    dp
}

fn materialize_count_lift_tuple(
    head: &Atom,
    group_key: &[Value],
    key_var_to_pos: &HashMap<String, usize>,
    count_value: usize,
) -> Result<Vec<Value>> {
    let count_value: i64 = count_value
        .try_into()
        .map_err(|_| XlogError::Compilation("count() overflowed i64".to_string()))?;
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
                tuple.push(group_key[pos].clone());
            }
            Term::Aggregate(AggExpr {
                op: AggOp::Count, ..
            }) => tuple.push(Value::I64(count_value)),
            Term::Aggregate(AggExpr { op, .. }) => {
                return Err(XlogError::Compilation(format!(
                    "Internal aggregate lift state mismatch for {}",
                    agg_op_label(*op)
                )));
            }
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                tuple.push(value_from_term(term)?);
            }
            Term::Anonymous => unreachable!("aggregate head plan rejects anonymous terms"),
            Term::List(_) => {
                return Err(unsupported_probabilistic_term_error(
                    "aggregate head materialization",
                    "list",
                ));
            }
            Term::Cons { .. } => {
                return Err(unsupported_probabilistic_term_error(
                    "aggregate head materialization",
                    "cons",
                ));
            }
            Term::Compound { .. } => {
                return Err(unsupported_probabilistic_term_error(
                    "aggregate head materialization",
                    "compound",
                ));
            }
            Term::PredRef(_) => {
                return Err(unsupported_probabilistic_term_error(
                    "aggregate head materialization",
                    "predref",
                ));
            }
        }
    }
    Ok(tuple)
}

#[allow(clippy::too_many_arguments)]
fn record_aggregate_lift_reports(
    aggregate_lifting: &mut Vec<AggregateLiftReport>,
    head: &Atom,
    group_key: &[Value],
    agg_specs: &[(AggOp, String)],
    deterministic_rows: usize,
    uncertain_rows: usize,
    status: AggregateLiftStatus,
    reason: &str,
    cap: usize,
    dynamic_programming_states: usize,
) {
    for (op, _) in agg_specs {
        aggregate_lifting.push(AggregateLiftReport {
            predicate: head.predicate.clone(),
            group_key: group_key.to_vec(),
            operator: agg_op_label(*op).to_string(),
            finite_domain_source: "grounded body rows".to_string(),
            deterministic_rows,
            uncertain_rows,
            domain_size: deterministic_rows + uncertain_rows,
            cap,
            status,
            reason: reason.to_string(),
            naive_outcomes: naive_outcome_count(uncertain_rows),
            dynamic_programming_states,
        });
    }
}

fn agg_op_label(op: AggOp) -> &'static str {
    match op {
        AggOp::Count => "count",
        AggOp::Sum => "sum",
        AggOp::Min => "min",
        AggOp::Max => "max",
        AggOp::LogSumExp => "logsumexp",
    }
}

fn naive_outcome_count(uncertain_rows: usize) -> u128 {
    if uncertain_rows >= u128::BITS as usize {
        u128::MAX
    } else {
        1u128 << uncertain_rows
    }
}

fn count_lift_dp_states(uncertain_rows: usize) -> usize {
    (uncertain_rows + 1) * (uncertain_rows + 2) / 2
}

type AggregatePlan = (
    Vec<String>,
    HashMap<String, usize>,
    Vec<(AggOp, String)>,
    HashMap<(AggOp, String), usize>,
);

fn aggregate_head_plan(head: &Atom) -> Result<AggregatePlan> {
    let mut key_vars: Vec<String> = Vec::new();
    let mut key_var_to_pos: HashMap<String, usize> = HashMap::new();
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
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {}
            Term::Anonymous => {
                return Err(XlogError::Compilation(format!(
                    "Anonymous variable in aggregate head of {} is not supported",
                    head.predicate
                )));
            }
            Term::List(_) => {
                return Err(unsupported_probabilistic_term_error(
                    "aggregate head planning",
                    "list",
                ));
            }
            Term::Cons { .. } => {
                return Err(unsupported_probabilistic_term_error(
                    "aggregate head planning",
                    "cons",
                ));
            }
            Term::Compound { .. } => {
                return Err(unsupported_probabilistic_term_error(
                    "aggregate head planning",
                    "compound",
                ));
            }
            Term::PredRef(_) => {
                return Err(unsupported_probabilistic_term_error(
                    "aggregate head planning",
                    "predref",
                ));
            }
        }
    }

    Ok((key_vars, key_var_to_pos, agg_specs, agg_to_pos))
}

fn canonical_binding_key(binding: &HashMap<String, Value>) -> Vec<(String, Value)> {
    let mut key: Vec<(String, Value)> = binding
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    key.sort();
    key
}

fn pir_const_value(builder: &PirBuilder, node: PirNodeId) -> Option<bool> {
    match builder.pir.node(node) {
        Some(crate::pir::PirNode::Const(value)) => Some(*value),
        _ => None,
    }
}

fn update_aggregate_states(
    states: &mut [AggState],
    agg_specs: &[(AggOp, String)],
    row: &AggregateProvRow,
) -> Result<()> {
    for (idx, (op, var)) in agg_specs.iter().enumerate() {
        let v = row
            .binding
            .get(var)
            .ok_or_else(|| XlogError::UnsafeVariable(var.clone()))?;
        states[idx].update(*op, v)?;
    }
    Ok(())
}

fn materialize_aggregate_tuple(
    head: &Atom,
    group_key: &[Value],
    key_var_to_pos: &HashMap<String, usize>,
    agg_specs: &[(AggOp, String)],
    agg_to_pos: &HashMap<(AggOp, String), usize>,
    agg_states: &[AggState],
) -> Result<Vec<Value>> {
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
                tuple.push(group_key[pos].clone());
            }
            Term::Aggregate(AggExpr { op, variable }) => {
                let idx = *agg_to_pos
                    .get(&(*op, variable.clone()))
                    .expect("agg_to_pos missing");
                let spec = agg_specs
                    .get(idx)
                    .expect("aggregate state index should have a spec");
                tuple.push(agg_states[idx].finish(spec.0)?);
            }
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                tuple.push(value_from_term(term)?);
            }
            Term::Anonymous => unreachable!("aggregate head plan rejects anonymous terms"),
            Term::List(_) => {
                return Err(unsupported_probabilistic_term_error(
                    "aggregate head materialization",
                    "list",
                ));
            }
            Term::Cons { .. } => {
                return Err(unsupported_probabilistic_term_error(
                    "aggregate head materialization",
                    "cons",
                ));
            }
            Term::Compound { .. } => {
                return Err(unsupported_probabilistic_term_error(
                    "aggregate head materialization",
                    "compound",
                ));
            }
            Term::PredRef(_) => {
                return Err(unsupported_probabilistic_term_error(
                    "aggregate head materialization",
                    "predref",
                ));
            }
        }
    }
    Ok(tuple)
}

fn select_relation<'a>(
    atom: &Atom,
    body_index: usize,
    global: &'a BTreeMap<String, Relation>,
    full_scc: &'a BTreeMap<String, Relation>,
    delta_scc: Option<(usize, &'a BTreeMap<String, Relation>)>,
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
            Term::List(_) => {
                return Err(unsupported_probabilistic_term_error("unification", "list"))
            }
            Term::Cons { .. } => {
                return Err(unsupported_probabilistic_term_error("unification", "cons"))
            }
            Term::Compound { .. } => {
                return Err(unsupported_probabilistic_term_error(
                    "unification",
                    "compound",
                ));
            }
            Term::PredRef(_) => {
                return Err(unsupported_probabilistic_term_error(
                    "unification",
                    "predref",
                ))
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
            Term::Aggregate(AggExpr {
                op: AggOp::Count,
                variable: _,
            })
            | Term::Aggregate(AggExpr {
                op: AggOp::Sum,
                variable: _,
            })
            | Term::Aggregate(AggExpr {
                op: AggOp::Min,
                variable: _,
            })
            | Term::Aggregate(AggExpr {
                op: AggOp::Max,
                variable: _,
            })
            | Term::Aggregate(AggExpr {
                op: AggOp::LogSumExp,
                variable: _,
            }) => {
                return Err(XlogError::Compilation(
                    "Aggregation not supported in provenance extraction".to_string(),
                ));
            }
            Term::List(_) => {
                return Err(unsupported_probabilistic_term_error(
                    "head materialization",
                    "list",
                ));
            }
            Term::Cons { .. } => {
                return Err(unsupported_probabilistic_term_error(
                    "head materialization",
                    "cons",
                ));
            }
            Term::Compound { .. } => {
                return Err(unsupported_probabilistic_term_error(
                    "head materialization",
                    "compound",
                ));
            }
            Term::PredRef(_) => {
                return Err(unsupported_probabilistic_term_error(
                    "head materialization",
                    "predref",
                ));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) fn eval_comparison(
    op: CompOp,
    left: &Term,
    right: &Term,
    binding: &HashMap<String, Value>,
) -> Result<bool> {
    eval_comparison_with_arithmetic_bindings(op, left, right, binding, &HashMap::new())
}

pub(crate) fn eval_comparison_with_arithmetic_bindings(
    op: CompOp,
    left: &Term,
    right: &Term,
    binding: &HashMap<String, Value>,
    arithmetic_bindings: &HashMap<String, ArithmeticValue>,
) -> Result<bool> {
    let bound_left = bound_arithmetic_value(left, binding, arithmetic_bindings);
    let bound_right = bound_arithmetic_value(right, binding, arithmetic_bindings);
    let left_type = bound_left.as_ref().and_then(ArithmeticValue::scalar_type);
    let right_type = bound_right.as_ref().and_then(ArithmeticValue::scalar_type);
    let left = resolve_comparison_arithmetic_term(left, binding, bound_left, right_type)?;
    let right = resolve_comparison_arithmetic_term(right, binding, bound_right, left_type)?;
    compare_arithmetic_values(&left, op, &right)
}

pub(crate) fn resolve_term(term: &Term, binding: &HashMap<String, Value>) -> Result<Value> {
    match term {
        Term::Variable(name) => binding.get(name).cloned().ok_or_else(|| {
            XlogError::Compilation(format!("Unbound variable {} in comparison", name))
        }),
        Term::Anonymous => Err(XlogError::Compilation(
            "Anonymous variable not allowed in comparison".to_string(),
        )),
        Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
            value_from_term(term)
        }
        Term::Aggregate(_) => Err(XlogError::Compilation(
            "Aggregation not supported in provenance extraction".to_string(),
        )),
        Term::List(_) => Err(unsupported_probabilistic_term_error("comparison", "list")),
        Term::Cons { .. } => Err(unsupported_probabilistic_term_error("comparison", "cons")),
        Term::Compound { .. } => Err(unsupported_probabilistic_term_error(
            "comparison",
            "compound",
        )),
        Term::PredRef(_) => Err(unsupported_probabilistic_term_error(
            "comparison",
            "predref",
        )),
    }
}

#[cfg(test)]
pub(crate) fn eval_arith_expr(expr: &ArithExpr, binding: &HashMap<String, Value>) -> Result<Value> {
    let value = eval_arithmetic_value(expr, binding, &HashMap::new())?;
    provenance_value_from_arithmetic(value)
}

pub(crate) fn eval_arithmetic_value(
    expr: &ArithExpr,
    binding: &HashMap<String, Value>,
    arithmetic_bindings: &HashMap<String, ArithmeticValue>,
) -> Result<ArithmeticValue> {
    let bindings = binding
        .iter()
        .map(|(name, value)| (name.clone(), arithmetic_value_from_provenance(value)))
        .chain(
            arithmetic_bindings
                .iter()
                .map(|(name, value)| (name.clone(), value.clone())),
        )
        .collect::<HashMap<_, _>>();
    evaluate_arithmetic_expression(expr, &bindings)
}

pub(crate) fn provenance_value_from_arithmetic(value: ArithmeticValue) -> Result<Value> {
    match value {
        ArithmeticValue::I32(value) => Ok(Value::I64(i64::from(value))),
        ArithmeticValue::I64(value) => Ok(Value::I64(value)),
        ArithmeticValue::U32(value) => Ok(Value::I64(i64::from(value))),
        ArithmeticValue::U64(value) => i64::try_from(value)
            .map(Value::I64)
            .map_err(|_| XlogError::Compilation("u64 arithmetic result exceeds i64".to_string())),
        ArithmeticValue::F32(value) => Ok(Value::F64(f64::from(value).to_bits())),
        ArithmeticValue::F64(value) => Ok(Value::F64(value.to_bits())),
        ArithmeticValue::Bool(value) => Ok(Value::I64(i64::from(value))),
        ArithmeticValue::Symbol(value) => Ok(Value::Symbol(value)),
        ArithmeticValue::String(value) => Ok(Value::String(value)),
    }
}

pub(crate) fn bind_arithmetic_result(
    target: &str,
    value: ArithmeticValue,
    binding: &mut HashMap<String, Value>,
    arithmetic_bindings: &mut HashMap<String, ArithmeticValue>,
) -> Result<()> {
    if binding.contains_key(target) || arithmetic_bindings.contains_key(target) {
        return Err(XlogError::Compilation(format!(
            "Is-expression target {target} is already bound"
        )));
    }
    let provenance_value = provenance_value_from_arithmetic(value.clone())?;
    binding.insert(target.to_string(), provenance_value);
    arithmetic_bindings.insert(target.to_string(), value);
    Ok(())
}

fn bound_arithmetic_value(
    term: &Term,
    binding: &HashMap<String, Value>,
    arithmetic_bindings: &HashMap<String, ArithmeticValue>,
) -> Option<ArithmeticValue> {
    if let Term::Variable(name) = term {
        if let Some(value) = arithmetic_bindings.get(name) {
            return Some(value.clone());
        }
        return binding.get(name).map(arithmetic_value_from_provenance);
    }
    None
}

fn resolve_comparison_arithmetic_term(
    term: &Term,
    binding: &HashMap<String, Value>,
    bound: Option<ArithmeticValue>,
    peer_type: Option<ScalarType>,
) -> Result<ArithmeticValue> {
    if let Some(bound) = bound {
        return Ok(bound);
    }
    if let Some(peer_type) = peer_type {
        return ArithmeticValue::from_typed_term(term, peer_type);
    }
    resolve_term(term, binding).map(|value| arithmetic_value_from_provenance(&value))
}

pub(crate) fn extend_arithmetic_bindings(
    atom: &Atom,
    tuple: &[Value],
    schemas: &HashMap<String, Schema>,
    arithmetic_bindings: &mut HashMap<String, ArithmeticValue>,
) -> Result<()> {
    let schema = schemas.get(&atom.predicate).ok_or_else(|| {
        XlogError::Compilation(format!(
            "Arithmetic evaluation requires a schema for predicate '{}'",
            atom.predicate
        ))
    })?;
    if atom.terms.len() != tuple.len() || schema.arity() != tuple.len() {
        return Err(XlogError::Compilation(format!(
            "Predicate '{}' row arity does not match its arithmetic schema",
            atom.predicate
        )));
    }
    for (index, (term, value)) in atom.terms.iter().zip(tuple).enumerate() {
        let Term::Variable(name) = term else {
            continue;
        };
        let scalar_type = schema.column_type(index).ok_or_else(|| {
            XlogError::Compilation(format!(
                "Arithmetic evaluation requires a type for '{}' column {}",
                atom.predicate,
                index + 1
            ))
        })?;
        let typed_value = arithmetic_value_from_typed_provenance(value, scalar_type)?;
        if let Some(existing) = arithmetic_bindings.get(name) {
            if existing.scalar_type() != typed_value.scalar_type()
                || !compare_arithmetic_values(existing, CompOp::Eq, &typed_value)?
            {
                return Err(XlogError::Compilation(format!(
                    "Arithmetic binding for variable '{name}' has incompatible predicate types"
                )));
            }
        } else {
            arithmetic_bindings.insert(name.clone(), typed_value);
        }
    }
    Ok(())
}

fn arithmetic_value_from_typed_provenance(
    value: &Value,
    scalar_type: ScalarType,
) -> Result<ArithmeticValue> {
    let mismatch = || {
        XlogError::Compilation(format!(
            "Provenance value is incompatible with declared {scalar_type:?} arithmetic type"
        ))
    };
    match (scalar_type, value) {
        (ScalarType::I32, Value::I64(value)) => i32::try_from(*value)
            .map(ArithmeticValue::I32)
            .map_err(|_| mismatch()),
        (ScalarType::I64, Value::I64(value)) => Ok(ArithmeticValue::I64(*value)),
        (ScalarType::U32, Value::I64(value)) => u32::try_from(*value)
            .map(ArithmeticValue::U32)
            .map_err(|_| mismatch()),
        (ScalarType::U64, Value::I64(value)) => u64::try_from(*value)
            .map(ArithmeticValue::U64)
            .map_err(|_| mismatch()),
        (ScalarType::F32, Value::F64(bits)) => {
            Ok(ArithmeticValue::F32(f64::from_bits(*bits) as f32))
        }
        (ScalarType::F64, Value::F64(bits)) => Ok(ArithmeticValue::F64(f64::from_bits(*bits))),
        (ScalarType::Bool, Value::I64(0)) => Ok(ArithmeticValue::Bool(false)),
        (ScalarType::Bool, Value::I64(1)) => Ok(ArithmeticValue::Bool(true)),
        (ScalarType::Symbol, Value::Symbol(value)) => Ok(ArithmeticValue::Symbol(*value)),
        (ScalarType::Symbol, Value::String(value)) => {
            Ok(ArithmeticValue::Symbol(symbol::intern(value)))
        }
        _ => Err(mismatch()),
    }
}

fn arithmetic_value_from_provenance(value: &Value) -> ArithmeticValue {
    match value {
        Value::I64(value) => ArithmeticValue::I64(*value),
        Value::F64(bits) => ArithmeticValue::F64(f64::from_bits(*bits)),
        Value::Symbol(value) => ArithmeticValue::Symbol(*value),
        Value::String(value) => ArithmeticValue::String(value.clone()),
    }
}

#[cfg(test)]
mod arithmetic_evaluation_tests {
    use super::*;
    use xlog_core::ScalarType;

    #[test]
    fn provenance_uses_shared_cast_conditional_and_power_semantics() {
        let expression = ArithExpr::Conditional {
            cond_left: Box::new(ArithExpr::Integer(1)),
            cond_op: CompOp::Eq,
            cond_right: Box::new(ArithExpr::Integer(1)),
            then_expr: Box::new(ArithExpr::Cast(
                Box::new(ArithExpr::Pow(
                    Box::new(ArithExpr::Integer(2)),
                    Box::new(ArithExpr::Integer(3)),
                )),
                ScalarType::F32,
            )),
            else_expr: Box::new(ArithExpr::Cast(
                Box::new(ArithExpr::Integer(0)),
                ScalarType::F32,
            )),
        };
        assert_eq!(
            eval_arith_expr(&expression, &HashMap::new()).expect("provenance value"),
            Value::F64(8.0_f64.to_bits())
        );
    }

    #[test]
    fn provenance_preserves_left_to_right_arithmetic_errors() {
        let expression = ArithExpr::Add(
            Box::new(ArithExpr::Variable("left_missing".to_string())),
            Box::new(ArithExpr::Variable("right_missing".to_string())),
        );
        let error = eval_arith_expr(&expression, &HashMap::new())
            .expect_err("unbound arithmetic must fail");
        assert!(error.to_string().contains("left_missing"), "{error}");

        let error = eval_comparison(
            CompOp::Eq,
            &Term::Variable("left_missing".to_string()),
            &Term::Variable("right_missing".to_string()),
            &HashMap::new(),
        )
        .expect_err("unbound comparison must fail");
        assert!(error.to_string().contains("left_missing"), "{error}");
        assert!(!error.to_string().contains("right_missing"), "{error}");
    }

    #[test]
    fn provenance_comparisons_share_runtime_nan_ordering_with_conditionals() {
        let nan_expression = ArithExpr::Div(
            Box::new(ArithExpr::Float(0.0)),
            Box::new(ArithExpr::Float(0.0)),
        );
        let nan = eval_arith_expr(&nan_expression, &HashMap::new()).expect("canonical NaN");
        let binding = HashMap::from([("X".to_string(), nan)]);
        assert!(eval_comparison(
            CompOp::Gt,
            &Term::Variable("X".to_string()),
            &Term::Float(f64::INFINITY),
            &binding,
        )
        .expect("body comparison"));

        let conditional = ArithExpr::Conditional {
            cond_left: Box::new(ArithExpr::Variable("X".to_string())),
            cond_op: CompOp::Gt,
            cond_right: Box::new(ArithExpr::Float(f64::INFINITY)),
            then_expr: Box::new(ArithExpr::Integer(1)),
            else_expr: Box::new(ArithExpr::Integer(0)),
        };
        assert_eq!(
            eval_arith_expr(&conditional, &binding).expect("conditional comparison"),
            Value::I64(1)
        );
        assert!(!eval_comparison(
            CompOp::Eq,
            &Term::Variable("X".to_string()),
            &Term::Variable("X".to_string()),
            &binding,
        )
        .expect("NaN equality"));
    }

    #[test]
    fn exact_provenance_preserves_declared_and_sequential_arithmetic_widths() {
        let provenance = extract_from_source(
            "pred input(u32).\n\
             pred cast_input(i64).\n\
             pred input_float(f32).\n\
             pred wide_input(u32).\n\
             pred wide_copy(u32).\n\
             pred out_from_input(u32).\n\
             pred out_from_cast(u32).\n\
             pred input_at_least_one(u32).\n\
             pred float_at_least_one(f32).\n\
             0.5::input(1).\n\
             0.5::cast_input(1).\n\
             0.5::input_float(1.1).\n\
             0.5::wide_input(4294967295).\n\
             wide_copy(X) :- wide_input(X).\n\
             out_from_input(Y) :- input(X), Y is X + cast(1, u32).\n\
             out_from_cast(Z) :- cast_input(X), Y is cast(X, u32), Z is Y + cast(1, u32).\n\
             input_at_least_one(X) :- input(X), X >= 1.\n\
             float_at_least_one(X) :- input_float(X), X >= 1.0.\n\
             query(out_from_input(2)).\n\
             query(out_from_cast(2)).\n\
             query(input_at_least_one(1)).\n\
             query(float_at_least_one(1.1)).\n\
             query(wide_copy(4294967295)).\n",
        )
        .expect("extract typed arithmetic provenance");

        for (predicate, expected) in [
            ("out_from_input", 2),
            ("out_from_cast", 2),
            ("input_at_least_one", 1),
        ] {
            assert!(
                provenance
                    .query_formula(predicate, &[Value::I64(expected)])
                    .is_some(),
                "missing derived query formula for {predicate}"
            );
        }
        assert!(
            provenance
                .query_formula("float_at_least_one", &[Value::F64(1.1_f64.to_bits())])
                .is_some(),
            "public lookup must canonicalize an f64 caller value to declared f32"
        );
        assert!(
            provenance
                .query_formula("wide_copy", &[Value::I64(i64::from(u32::MAX))])
                .is_some(),
            "public lookup must retain an in-range u32 boundary"
        );
        assert!(
            provenance
                .query_formula("wide_copy", &[Value::I64(-1)])
                .is_none(),
            "public lookup must reject a value outside the declared u32 range"
        );
    }

    #[test]
    fn provenance_preserves_source_symbol_spelling_in_public_metadata() {
        let provenance = extract_from_source(
            "0.25::gate(\"alpha\").\n\
             0.1::gate(alpha).\n\
             0.4::route(\"beta\"); 0.6::route(gamma).\n\
             evidence(gate(\"alpha\"), true).\n\
             query(gate(\"alpha\")).\n",
        )
        .expect("extract quoted-symbol provenance");

        let quoted_alpha = Value::String("alpha".to_string());
        assert_eq!(
            provenance.queries[0].args.as_slice(),
            std::slice::from_ref(&quoted_alpha)
        );
        assert_eq!(
            provenance.evidence[0].0.args.as_slice(),
            std::slice::from_ref(&quoted_alpha)
        );
        let leaf_values = provenance
            .leaf_atoms
            .values()
            .map(|atom| atom.args[0].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            leaf_values,
            [quoted_alpha.clone(), Value::Symbol(symbol::intern("alpha"))]
        );

        let choices = &provenance
            .choice_sources
            .values()
            .next()
            .expect("annotated-disjunction choice metadata")
            .choices;
        assert_eq!(choices[0].0.args, [Value::String("beta".to_string())]);
        assert_eq!(choices[1].0.args, [Value::Symbol(symbol::intern("gamma"))]);

        let quoted_formula = provenance
            .query_formula("gate", std::slice::from_ref(&quoted_alpha))
            .expect("quoted symbol lookup");
        let bare_formula = provenance
            .query_formula("gate", &[Value::Symbol(symbol::intern("alpha"))])
            .expect("bare symbol lookup");
        assert_eq!(quoted_formula, bare_formula);

        let gate_atoms = provenance
            .atoms_with_formulas()
            .filter(|(atom, _)| atom.predicate == "gate")
            .collect::<Vec<_>>();
        assert_eq!(gate_atoms.len(), 1);
        assert_eq!(
            gate_atoms[0].0.args,
            [Value::Symbol(symbol::intern("alpha"))]
        );
    }

    #[test]
    fn exact_provenance_preserves_stratification_error_precedence() {
        let error = extract_from_source(
            "pred input(f32).\n\
             pred output(f64).\n\
             input(0.1).\n\
             output(X) :- input(X).\n\
             left() :- not right().\n\
             right() :- not left().\n",
        )
        .expect_err("ordinary negation cycle must fail before rule type validation");

        assert!(
            matches!(error, XlogError::StratificationCycle(_)),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn exact_provenance_indexes_runtime_f32_nan_and_infinity() {
        let provenance = extract_from_source(
            "pred seed().\n\
             pred nan_value(f32).\n\
             pred infinite_value(f32).\n\
             0.5::seed().\n\
             nan_value(Y) :- seed(), Y is cast(0.0, f32) / cast(0.0, f32).\n\
             infinite_value(Y) :- seed(), Y is cast(1.0, f32) / cast(0.0, f32).\n",
        )
        .expect("extract non-finite f32 provenance");

        assert!(provenance
            .query_formula("nan_value", &[Value::F64(f64::from(f32::NAN).to_bits())])
            .is_some());
        assert!(provenance
            .query_formula(
                "infinite_value",
                &[Value::F64(f64::from(f32::INFINITY).to_bits())]
            )
            .is_some());
    }

    #[test]
    fn exact_provenance_rejects_arithmetic_results_outside_public_value_range() {
        let error = extract_from_source(
            "pred seed().\n\
             pred wrapped_u64(u64).\n\
             0.5::seed().\n\
             wrapped_u64(Z) :- seed(), Y is cast(0 - 1, u64), Z is Y + cast(1, u64).\n\
             query(wrapped_u64(0)).\n",
        )
        .expect_err("non-representable u64 intermediate must fail before relational use");

        assert!(
            error
                .to_string()
                .contains("u64 arithmetic result exceeds i64"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn wfs_grounding_preserves_declared_arithmetic_widths() {
        extract_from_source(
            "pred input(u32).\n\
             pred left(u32).\n\
             pred right(u32).\n\
             0.5::input(1).\n\
             left(Y) :- input(X), Y is X + cast(1, u32), not right(Y).\n\
             right(Y) :- input(X), Y is X + cast(1, u32), not left(Y).\n\
             query(left(2)).\n",
        )
        .expect("ground typed arithmetic before WFS evaluation");
    }

    #[test]
    fn wfs_grounding_rejects_arithmetic_results_outside_public_value_range() {
        let error = extract_from_source(
            "pred seed().\n\
             pred left().\n\
             pred right(u64).\n\
             0.5::seed().\n\
             left() :- seed(), Y is cast(0 - 1, u64), not right(Y).\n\
             right(Y) :- seed(), Y is cast(0 - 1, u64), not left().\n\
             query(left()).\n",
        )
        .expect_err("WFS grounding must reject non-representable arithmetic bindings");

        assert!(
            error
                .to_string()
                .contains("u64 arithmetic result exceeds i64"),
            "unexpected error: {error}"
        );
    }
}
