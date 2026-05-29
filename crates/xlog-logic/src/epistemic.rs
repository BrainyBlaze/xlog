//! Epistemic mode helpers for compatibility fixtures.

use std::collections::{BTreeMap, BTreeSet};

use xlog_core::{Result, XlogError};
use xlog_ir::{
    EirBodyLiteral, EirEpistemicMode, EirEpistemicOp, EirProgram, EirTerm, EpistemicExecutablePlan,
    EpistemicGpuPlan, EpistemicReductionPlan, EpistemicSolverAssumptionBinding,
    EpistemicSolverServiceContract, EpistemicTupleMembershipBinding, EpistemicWcojReductionStatus,
};
use xlog_stats::StatsSnapshot;

use crate::ast::{
    Atom, BodyLiteral, CompOp, Comparison, Constraint, EpistemicLiteral, EpistemicMode,
    EpistemicOp, Program, Term,
};
use crate::build_eir;
use crate::compile::Compiler;

/// Boolean truth value for bounded epistemic fixture evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruthValue {
    /// The literal is true.
    True,
    /// The literal is false.
    False,
}

impl TruthValue {
    fn from_bool(value: bool) -> Self {
        if value {
            TruthValue::True
        } else {
            TruthValue::False
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum EpistemicTermKey {
    Integer(i64),
    FloatBits(u64),
    String(String),
    Symbol(u32),
    List(Vec<EpistemicTermKey>),
    Cons {
        head: Box<EpistemicTermKey>,
        tail: Box<EpistemicTermKey>,
    },
    Compound {
        functor: String,
        args: Vec<EpistemicTermKey>,
    },
    PredRef(String),
}

impl EpistemicTermKey {
    fn from_term(term: &Term) -> Result<Self> {
        Ok(match term {
            Term::Integer(value) => Self::Integer(*value),
            Term::Float(value) => Self::FloatBits(value.to_bits()),
            Term::String(value) => Self::String(value.clone()),
            Term::Symbol(value) => Self::Symbol(*value),
            Term::List(items) => Self::List(
                items
                    .iter()
                    .map(Self::from_term)
                    .collect::<Result<Vec<_>>>()?,
            ),
            Term::Cons { head, tail } => Self::Cons {
                head: Box::new(Self::from_term(head)?),
                tail: Box::new(Self::from_term(tail)?),
            },
            Term::Compound { functor, args } => Self::Compound {
                functor: functor.clone(),
                args: args
                    .iter()
                    .map(Self::from_term)
                    .collect::<Result<Vec<_>>>()?,
            },
            Term::PredRef(value) => Self::PredRef(value.clone()),
            Term::Variable(_) | Term::Anonymous | Term::Aggregate(_) => {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic tuple key".to_string(),
                    context: "tuple-key epistemic facts require ground terms".to_string(),
                });
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum EpistemicAtomKey {
    Arity {
        predicate: String,
        arity: usize,
    },
    Ground {
        predicate: String,
        terms: Vec<EpistemicTermKey>,
    },
}

impl EpistemicAtomKey {
    fn from_arity(predicate: impl Into<String>, arity: usize) -> Self {
        Self::Arity {
            predicate: predicate.into(),
            arity,
        }
    }

    fn from_terms(predicate: impl Into<String>, terms: &[Term]) -> Result<Self> {
        Ok(Self::Ground {
            predicate: predicate.into(),
            terms: terms
                .iter()
                .map(EpistemicTermKey::from_term)
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn predicate(&self) -> &str {
        match self {
            Self::Arity { predicate, .. } | Self::Ground { predicate, .. } => predicate,
        }
    }

    fn arity(&self) -> usize {
        match self {
            Self::Arity { arity, .. } => *arity,
            Self::Ground { terms, .. } => terms.len(),
        }
    }

    fn matches_atom(&self, atom: &Atom) -> bool {
        if self.predicate() != atom.predicate || self.arity() != atom.arity() {
            return false;
        }
        match self {
            Self::Arity { .. } => true,
            Self::Ground { terms, .. } => atom
                .terms
                .iter()
                .map(EpistemicTermKey::from_term)
                .collect::<Result<Vec<_>>>()
                .is_ok_and(|atom_terms| atom_terms == *terms),
        }
    }

    fn overlaps(&self, other: &Self) -> bool {
        if self.predicate() != other.predicate() || self.arity() != other.arity() {
            return false;
        }
        matches!(self, Self::Arity { .. }) || matches!(other, Self::Arity { .. }) || self == other
    }
}

/// Minimal interpretation used by G91/FAEEL distinction fixtures.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EpistemicInterpretation {
    known: BTreeSet<EpistemicAtomKey>,
    possible: BTreeSet<EpistemicAtomKey>,
    rejected: BTreeSet<EpistemicAtomKey>,
}

impl EpistemicInterpretation {
    /// Create an empty interpretation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a predicate/arity pair as known.
    pub fn with_known(mut self, predicate: impl Into<String>, arity: usize) -> Self {
        self.known
            .insert(EpistemicAtomKey::from_arity(predicate, arity));
        self
    }

    /// Mark a concrete tuple key as known.
    pub fn with_known_terms(
        mut self,
        predicate: impl Into<String>,
        terms: Vec<Term>,
    ) -> Result<Self> {
        self.known
            .insert(EpistemicAtomKey::from_terms(predicate, &terms)?);
        Ok(self)
    }

    /// Mark a predicate/arity pair as possible under G91 compatibility semantics.
    pub fn with_possible(mut self, predicate: impl Into<String>, arity: usize) -> Self {
        self.possible
            .insert(EpistemicAtomKey::from_arity(predicate, arity));
        self
    }

    /// Mark a concrete tuple key as possible under G91 compatibility semantics.
    pub fn with_possible_terms(
        mut self,
        predicate: impl Into<String>,
        terms: Vec<Term>,
    ) -> Result<Self> {
        self.possible
            .insert(EpistemicAtomKey::from_terms(predicate, &terms)?);
        Ok(self)
    }

    /// Mark a predicate/arity pair as rejected by the candidate.
    pub fn with_rejected(mut self, predicate: impl Into<String>, arity: usize) -> Self {
        self.rejected
            .insert(EpistemicAtomKey::from_arity(predicate, arity));
        self
    }

    /// Mark a concrete tuple key as rejected by the candidate.
    pub fn with_rejected_terms(
        mut self,
        predicate: impl Into<String>,
        terms: Vec<Term>,
    ) -> Result<Self> {
        self.rejected
            .insert(EpistemicAtomKey::from_terms(predicate, &terms)?);
        Ok(self)
    }

    fn first_contradiction(&self) -> Option<(String, usize)> {
        self.known
            .iter()
            .find(|key| self.rejected.iter().any(|rejected| key.overlaps(rejected)))
            .map(|key| (key.predicate().to_string(), key.arity()))
    }

    fn contains_known(&self, atom: &Atom) -> bool {
        self.known.iter().any(|key| key.matches_atom(atom))
    }

    fn contains_possible(&self, atom: &Atom) -> bool {
        self.possible.iter().any(|key| key.matches_atom(atom))
    }

    fn contains_rejected(&self, atom: &Atom) -> bool {
        self.rejected.iter().any(|key| key.matches_atom(atom))
    }

    fn epistemic_guess_count(&self) -> usize {
        self.known.len() + self.possible.len() + self.rejected.len()
    }
}

/// One stable model in a bounded epistemic world-view fixture.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EpistemicWorld {
    facts: BTreeSet<EpistemicAtomKey>,
}

impl EpistemicWorld {
    /// Create an empty world.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a predicate/arity fact to this world.
    pub fn with_fact(mut self, predicate: impl Into<String>, arity: usize) -> Self {
        self.facts
            .insert(EpistemicAtomKey::from_arity(predicate, arity));
        self
    }

    /// Add a concrete tuple fact to this world.
    pub fn with_fact_terms(
        mut self,
        predicate: impl Into<String>,
        terms: Vec<Term>,
    ) -> Result<Self> {
        self.facts
            .insert(EpistemicAtomKey::from_terms(predicate, &terms)?);
        Ok(self)
    }

    fn contains(&self, atom: &Atom) -> bool {
        self.facts.iter().any(|fact| fact.matches_atom(atom))
    }
}

/// Non-empty set of accepted stable models used as the epistemic boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicWorldView {
    worlds: Vec<EpistemicWorld>,
}

impl EpistemicWorldView {
    /// Construct a non-empty world view.
    pub fn from_worlds(worlds: Vec<EpistemicWorld>) -> Result<Self> {
        if worlds.is_empty() {
            return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                construct: "world view boundary".to_string(),
                context: "world view requires at least one stable model".to_string(),
            });
        }
        Ok(Self { worlds })
    }

    /// Return the number of worlds in this view.
    pub fn world_count(&self) -> usize {
        self.worlds.len()
    }

    /// Evaluate an epistemic literal over this world view.
    pub fn evaluate(&self, lit: &EpistemicLiteral) -> TruthValue {
        let value = match lit.op {
            EpistemicOp::Know => self.worlds.iter().all(|world| world.contains(&lit.atom)),
            EpistemicOp::Possible => self.worlds.iter().any(|world| world.contains(&lit.atom)),
        };

        TruthValue::from_bool(if lit.negated { !value } else { value })
    }
}

/// Build the production-facing GPU execution contract for an epistemic program.
///
/// This does not launch kernels. It proves that the semantic boundary can be
/// represented as a GPU-native execution plan with explicit hot-path phases,
/// required device buffers, WCOJ planning obligations, and zero CPU fallback
/// counters.
pub fn plan_epistemic_gpu_execution(program: &Program) -> Result<EpistemicGpuPlan> {
    reject_epistemic_constraints(program)?;
    let eir = build_eir(program)?;
    reject_faeel_self_supported_possible(&eir)?;
    let mut epistemic_literals = Vec::new();
    let mut reductions = Vec::new();
    let mut tuple_membership_bindings = Vec::new();
    let mut solver_assumption_bindings = Vec::new();

    for (rule_index, rule) in eir.rules.iter().enumerate() {
        let mut rule_epistemic_literals = Vec::new();
        let mut relational_body_atoms = 0usize;
        let mut has_negated_relational_atom = false;

        for lit in &rule.body {
            match lit {
                EirBodyLiteral::Relational { negated, .. } => {
                    if *negated {
                        has_negated_relational_atom = true;
                    } else {
                        relational_body_atoms += 1;
                    }
                }
                EirBodyLiteral::Epistemic(lit) => {
                    rule_epistemic_literals.push(lit.clone());
                }
                EirBodyLiteral::Constraint | EirBodyLiteral::Binding => {}
            }
        }

        if rule_epistemic_literals.is_empty() {
            continue;
        }

        let reduction_index = reductions.len();
        for lit in rule_epistemic_literals {
            let literal_index = epistemic_literals.len();
            let augmented_head_terms = augmented_eir_head_terms(rule);
            tuple_membership_bindings.push(EpistemicTupleMembershipBinding {
                literal_index,
                reduction_index,
                predicate: lit.atom.predicate.clone(),
                arity: lit.atom.arity,
                key_columns: (0..lit.atom.arity).collect(),
                key_terms: lit.atom.terms.clone(),
                bound_output_columns: bound_output_columns_for_terms(
                    &lit.atom.terms,
                    &augmented_head_terms,
                ),
                op: lit.op,
                negated: lit.negated,
            });
            solver_assumption_bindings.push(EpistemicSolverAssumptionBinding {
                literal_index,
                reduction_index,
                predicate: lit.atom.predicate.clone(),
                arity: lit.atom.arity,
                terms: lit.atom.terms.clone(),
                op: lit.op,
                negated: lit.negated,
            });
            epistemic_literals.push(lit);
        }
        reductions.push(EpistemicReductionPlan {
            rule_index,
            head_predicate: rule.head.predicate.clone(),
            relational_body_atoms,
            wcoj_status: wcoj_status_for_reduction(
                relational_body_atoms,
                has_negated_relational_atom,
            ),
        });
    }

    if epistemic_literals.is_empty() {
        return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
            construct: "epistemic GPU execution plan".to_string(),
            context: "requires at least one epistemic literal".to_string(),
        });
    }

    let final_output_columns = final_output_columns_for_eir(&eir);
    let gpu_plan = EpistemicGpuPlan::new(eir.mode, epistemic_literals, reductions)
        .with_tuple_membership_bindings(tuple_membership_bindings)
        .with_final_output_columns(final_output_columns)
        .with_solver_contract(EpistemicSolverServiceContract::production_default(
            solver_assumption_bindings,
        ));
    gpu_plan.validate_tuple_membership_bindings()?;
    gpu_plan.validate_solver_contract()?;
    Ok(gpu_plan)
}

fn reject_faeel_self_supported_possible(eir: &EirProgram) -> Result<()> {
    if eir.mode != EirEpistemicMode::Faeel {
        return Ok(());
    }

    for (rule_index, rule) in eir.rules.iter().enumerate() {
        for lit in &rule.body {
            let EirBodyLiteral::Epistemic(lit) = lit else {
                continue;
            };
            if lit.atom.predicate == rule.head.predicate && lit.atom.arity == rule.head.arity {
                if has_independent_founded_support(eir, &lit.atom)
                    || has_tuple_level_independent_founded_support(eir, rule, &lit.atom)
                {
                    continue;
                }
                let label = eir_epistemic_literal_label(lit);
                let missing = format_missing_foundation(&lit.atom);
                if lit.atom.arity > 0 {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "FAEEL foundedness guard".to_string(),
                        context: format!(
                            "rule[{rule_index}] has nonzero-arity self-supported {label} {}/{} in default FAEEL mode; no independent founded support proves the tuple key {missing}; accepted GPU lowering requires a tuple-level foundedness proof (a non-circular support rule whose body subsumes this rule's tuple-key domain) or explicit g91 compatibility mode",
                            lit.atom.predicate, lit.atom.arity
                        ),
                    });
                }
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "FAEEL foundedness guard".to_string(),
                    context: format!(
                        "rule[{rule_index}] has self-supported {label} {}/{} in default FAEEL mode; no independent founded support proves {missing}; use explicit g91 compatibility mode or provide a non-circular founded support rule",
                        lit.atom.predicate, lit.atom.arity
                    ),
                });
            }
        }
    }

    reject_faeel_unfounded_modal_cycles(eir)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ModalSupportNode {
    predicate: String,
    arity: usize,
}

impl ModalSupportNode {
    fn from_atom(atom: &xlog_ir::EirAtom) -> Self {
        Self {
            predicate: atom.predicate.clone(),
            arity: atom.arity,
        }
    }
}

#[derive(Debug, Clone)]
struct UnfoundedModalSupportEdge {
    from: ModalSupportNode,
    to: ModalSupportNode,
    rule_index: usize,
    label: &'static str,
}

fn reject_faeel_unfounded_modal_cycles(eir: &EirProgram) -> Result<()> {
    let graph = unfounded_modal_support_graph(eir);
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();

    for node in graph.keys() {
        if visited.contains(node) {
            continue;
        }
        if let Some(edge) = find_unfounded_modal_cycle(node, &graph, &mut visiting, &mut visited) {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "FAEEL foundedness guard".to_string(),
                context: format!(
                    "rule[{}] participates in an unfounded modal support cycle: {}/{} depends on {} {}/{} without independent founded support",
                    edge.rule_index,
                    edge.from.predicate,
                    edge.from.arity,
                    edge.label,
                    edge.to.predicate,
                    edge.to.arity
                ),
            });
        }
    }

    Ok(())
}

fn unfounded_modal_support_graph(
    eir: &EirProgram,
) -> BTreeMap<ModalSupportNode, Vec<UnfoundedModalSupportEdge>> {
    let mut graph: BTreeMap<_, Vec<_>> = BTreeMap::new();

    for (rule_index, rule) in eir.rules.iter().enumerate() {
        let from = ModalSupportNode::from_atom(&rule.head);
        for lit in &rule.body {
            let EirBodyLiteral::Epistemic(lit) = lit else {
                continue;
            };
            if has_independent_founded_support(eir, &lit.atom)
                || has_tuple_level_independent_founded_support(eir, rule, &lit.atom)
            {
                continue;
            }
            graph
                .entry(from.clone())
                .or_default()
                .push(UnfoundedModalSupportEdge {
                    from: from.clone(),
                    to: ModalSupportNode::from_atom(&lit.atom),
                    rule_index,
                    label: eir_epistemic_literal_label(lit),
                });
        }
    }

    graph
}

fn find_unfounded_modal_cycle(
    node: &ModalSupportNode,
    graph: &BTreeMap<ModalSupportNode, Vec<UnfoundedModalSupportEdge>>,
    visiting: &mut BTreeSet<ModalSupportNode>,
    visited: &mut BTreeSet<ModalSupportNode>,
) -> Option<UnfoundedModalSupportEdge> {
    visiting.insert(node.clone());

    if let Some(edges) = graph.get(node) {
        for edge in edges {
            if visiting.contains(&edge.to) {
                return Some(edge.clone());
            }
            if !visited.contains(&edge.to) {
                if let Some(cycle) = find_unfounded_modal_cycle(&edge.to, graph, visiting, visited)
                {
                    return Some(cycle);
                }
            }
        }
    }

    visiting.remove(node);
    visited.insert(node.clone());
    None
}

fn eir_epistemic_literal_label(lit: &xlog_ir::EirEpistemicLiteral) -> &'static str {
    match (lit.negated, lit.op) {
        (false, EirEpistemicOp::Know) => "know",
        (false, EirEpistemicOp::Possible) => "possible",
        (true, EirEpistemicOp::Know) => "not know",
        (true, EirEpistemicOp::Possible) => "not possible",
    }
}

/// Render the predicate/tuple-key whose independent foundation is missing.
///
/// Used to make FAEEL foundedness rejections name the exact tuple key that
/// lacks non-circular support (KPI: precise missing-foundation diagnostic).
fn format_missing_foundation(atom: &xlog_ir::EirAtom) -> String {
    if atom.arity == 0 {
        return format!("{}()", atom.predicate);
    }
    let key = atom
        .terms
        .iter()
        .map(format_eir_term_key)
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}({key})", atom.predicate)
}

fn format_eir_term_key(term: &EirTerm) -> String {
    match term {
        EirTerm::Variable(name) => name.clone(),
        EirTerm::Anonymous => "_".to_string(),
        EirTerm::Integer(value) => value.to_string(),
        EirTerm::String(value) => format!("{value:?}"),
        EirTerm::Symbol(value) => format!("sym#{value}"),
        other => format!("{other:?}"),
    }
}

fn has_independent_founded_support(eir: &EirProgram, atom: &xlog_ir::EirAtom) -> bool {
    if atom.arity > 0 && !atom.terms.iter().all(eir_term_is_ground) {
        return false;
    }

    let mut support_stack = Vec::new();
    has_independent_founded_support_inner(eir, atom, &mut support_stack)
}

fn has_tuple_level_independent_founded_support(
    eir: &EirProgram,
    modal_rule: &xlog_ir::EirRule,
    atom: &xlog_ir::EirAtom,
) -> bool {
    if atom.arity == 0 {
        return false;
    }

    let modal_domain = positive_relational_body_atoms(modal_rule);
    eir.rules.iter().any(|support_rule| {
        if !support_rule_head_matches_modal_atom(support_rule, atom) {
            return false;
        }
        let mut support_stack = vec![(atom.predicate.clone(), atom.arity)];
        if !eir_rule_has_independent_founded_body(eir, support_rule, &mut support_stack) {
            return false;
        }
        let Some(substitution) = head_substitution_to_atom(&support_rule.head, atom) else {
            return false;
        };
        let support_domain = positive_relational_body_atoms(support_rule);
        if support_domain.is_empty() {
            return false;
        }
        let Some(substituted_support_domain) = support_domain
            .iter()
            .map(|atom| substitute_eir_atom(atom, &substitution))
            .collect::<Option<Vec<_>>>()
        else {
            return false;
        };
        substituted_support_domain.iter().all(|support_atom| {
            modal_domain
                .iter()
                .any(|modal_atom| modal_atom == support_atom)
        })
    })
}

fn positive_relational_body_atoms(rule: &xlog_ir::EirRule) -> Vec<xlog_ir::EirAtom> {
    rule.body
        .iter()
        .filter_map(|lit| match lit {
            EirBodyLiteral::Relational {
                negated: false,
                atom,
            } => Some(atom.clone()),
            _ => None,
        })
        .collect()
}

fn support_rule_head_matches_modal_atom(rule: &xlog_ir::EirRule, atom: &xlog_ir::EirAtom) -> bool {
    rule.head.predicate == atom.predicate
        && rule.head.arity == atom.arity
        && head_substitution_to_atom(&rule.head, atom).is_some()
}

fn head_substitution_to_atom(
    head: &xlog_ir::EirAtom,
    atom: &xlog_ir::EirAtom,
) -> Option<BTreeMap<String, EirTerm>> {
    if head.predicate != atom.predicate || head.arity != atom.arity {
        return None;
    }
    let mut substitution = BTreeMap::new();
    for (head_term, atom_term) in head.terms.iter().zip(&atom.terms) {
        match head_term {
            EirTerm::Variable(name) => match substitution.get(name) {
                Some(existing) if existing != atom_term => return None,
                Some(_) => {}
                None => {
                    substitution.insert(name.clone(), atom_term.clone());
                }
            },
            EirTerm::Anonymous => return None,
            other if other == atom_term => {}
            _ => return None,
        }
    }
    Some(substitution)
}

fn substitute_eir_atom(
    atom: &xlog_ir::EirAtom,
    substitution: &BTreeMap<String, EirTerm>,
) -> Option<xlog_ir::EirAtom> {
    let terms = atom
        .terms
        .iter()
        .map(|term| substitute_eir_term(term, substitution))
        .collect::<Option<Vec<_>>>()?;
    Some(xlog_ir::EirAtom {
        predicate: atom.predicate.clone(),
        arity: atom.arity,
        terms,
    })
}

fn substitute_eir_term(
    term: &EirTerm,
    substitution: &BTreeMap<String, EirTerm>,
) -> Option<EirTerm> {
    match term {
        EirTerm::Variable(name) => Some(
            substitution
                .get(name)
                .cloned()
                .unwrap_or_else(|| term.clone()),
        ),
        EirTerm::Anonymous => None,
        EirTerm::List(items) => items
            .iter()
            .map(|item| substitute_eir_term(item, substitution))
            .collect::<Option<Vec<_>>>()
            .map(EirTerm::List),
        EirTerm::Cons { head, tail } => Some(EirTerm::Cons {
            head: Box::new(substitute_eir_term(head, substitution)?),
            tail: Box::new(substitute_eir_term(tail, substitution)?),
        }),
        EirTerm::Compound { functor, args } => Some(EirTerm::Compound {
            functor: functor.clone(),
            args: args
                .iter()
                .map(|arg| substitute_eir_term(arg, substitution))
                .collect::<Option<Vec<_>>>()?,
        }),
        EirTerm::Aggregate { .. } => None,
        EirTerm::Integer(_)
        | EirTerm::FloatBits(_)
        | EirTerm::String(_)
        | EirTerm::Symbol(_)
        | EirTerm::PredRef(_) => Some(term.clone()),
    }
}

fn has_independent_founded_support_inner(
    eir: &EirProgram,
    atom: &xlog_ir::EirAtom,
    support_stack: &mut Vec<(String, usize)>,
) -> bool {
    if atom.arity > 0 && !atom.terms.iter().all(eir_term_is_ground) {
        return false;
    }

    let key = (atom.predicate.clone(), atom.arity);
    if support_stack.iter().any(|ancestor| ancestor == &key) {
        return false;
    }
    support_stack.push(key);

    let supported = eir.rules.iter().any(|rule| {
        let Some(substitution) = head_substitution_to_atom(&rule.head, atom) else {
            return false;
        };
        eir_rule_has_independent_founded_body_with_substitution(
            eir,
            rule,
            &substitution,
            support_stack,
        )
    });

    support_stack.pop();
    supported
}

fn eir_rule_has_independent_founded_body(
    eir: &EirProgram,
    rule: &xlog_ir::EirRule,
    support_stack: &mut Vec<(String, usize)>,
) -> bool {
    eir_rule_has_independent_founded_body_with_substitution(
        eir,
        rule,
        &BTreeMap::new(),
        support_stack,
    )
}

fn eir_rule_has_independent_founded_body_with_substitution(
    eir: &EirProgram,
    rule: &xlog_ir::EirRule,
    substitution: &BTreeMap<String, EirTerm>,
    support_stack: &mut Vec<(String, usize)>,
) -> bool {
    rule.body.iter().all(|lit| match lit {
        EirBodyLiteral::Epistemic(_) => false,
        EirBodyLiteral::Relational { negated: true, .. } => false,
        EirBodyLiteral::Relational {
            negated: false,
            atom,
        } => {
            let Some(atom) = substitute_eir_atom(atom, substitution) else {
                return false;
            };
            let dependency_key = (atom.predicate.clone(), atom.arity);
            if support_stack
                .iter()
                .any(|ancestor| ancestor == &dependency_key)
            {
                return false;
            }
            if !eir
                .rules
                .iter()
                .any(|rule| head_substitution_to_atom(&rule.head, &atom).is_some())
            {
                return true;
            }
            has_independent_founded_support_inner(eir, &atom, support_stack)
        }
        EirBodyLiteral::Constraint | EirBodyLiteral::Binding => true,
    })
}

fn eir_term_is_ground(term: &EirTerm) -> bool {
    match term {
        EirTerm::Variable(_) | EirTerm::Anonymous | EirTerm::Aggregate { .. } => false,
        EirTerm::Integer(_) | EirTerm::FloatBits(_) | EirTerm::String(_) | EirTerm::Symbol(_) => {
            true
        }
        EirTerm::List(items) => items.iter().all(eir_term_is_ground),
        EirTerm::Cons { head, tail } => eir_term_is_ground(head) && eir_term_is_ground(tail),
        EirTerm::Compound { args, .. } => args.iter().all(eir_term_is_ground),
        EirTerm::PredRef(_) => true,
    }
}

/// Compile an epistemic program into its GPU contract and reduced runtime plan.
///
/// This is the first production-lowering boundary for epistemic execution. It
/// removes epistemic literals only after `plan_epistemic_gpu_execution` proves
/// the explicit EIR/GPU semantic contract, then sends the ordinary reduced
/// program through the same compiler, optimizer, helper-splitting, and WCOJ
/// promotion pipeline used by non-epistemic programs.
pub fn compile_epistemic_gpu_execution(program: &Program) -> Result<EpistemicExecutablePlan> {
    compile_epistemic_gpu_execution_with_stats_snapshot(program, None)
}

/// Compile an epistemic program with an optional production statistics snapshot.
///
/// This preserves the W2.x/W38-B planner contract for reduced ordinary bodies:
/// cardinality, selectivity, heat, prefix-degree, sorted-layout, and
/// helper-splitting decisions are owned by the existing production compiler
/// pipeline rather than by an epistemic side planner.
pub fn compile_epistemic_gpu_execution_with_stats_snapshot(
    program: &Program,
    stats_snapshot: Option<&StatsSnapshot>,
) -> Result<EpistemicExecutablePlan> {
    let gpu_plan = plan_epistemic_gpu_execution(program)?;
    require_single_epistemic_output_relation(&gpu_plan)?;
    let reduced_program = reduce_epistemic_program_to_ordinary(program);
    let mut compiler = Compiler::new();
    let reduced_runtime_plan =
        compiler.compile_program_with_stats_snapshot(&reduced_program, stats_snapshot)?;
    let relation_ids = compiler
        .rel_ids()
        .iter()
        .map(|(name, rel)| (name.clone(), *rel))
        .collect();

    Ok(EpistemicExecutablePlan {
        gpu_plan,
        relation_ids,
        reduced_runtime_plan,
    })
}

fn require_single_epistemic_output_relation(gpu_plan: &EpistemicGpuPlan) -> Result<()> {
    let output_relations: BTreeSet<&str> = gpu_plan
        .reductions
        .iter()
        .map(|reduction| reduction.head_predicate.as_str())
        .collect();
    if output_relations.len() > 1 {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "epistemic GPU final output relation".to_string(),
            context: format!(
                "single-plan GPU execution materializes one final output buffer, but reductions \
                 target multiple head predicates {:?}; use split GPU execution for independent \
                 epistemic outputs",
                output_relations
            ),
        });
    }
    Ok(())
}

fn reject_epistemic_constraints(program: &Program) -> Result<()> {
    reject_epistemic_constraints_for_boundary(program, "epistemic GPU constraint", "GPU lowering")
}

fn reject_gpt_epistemic_constraints(program: &Program) -> Result<()> {
    reject_epistemic_constraints_for_boundary(
        program,
        "epistemic GPT constraint",
        "GPT candidate testing",
    )
}

fn reject_epistemic_constraints_for_boundary(
    program: &Program,
    construct: &str,
    boundary: &str,
) -> Result<()> {
    for (constraint_index, constraint) in program.constraints.iter().enumerate() {
        for lit in &constraint.body {
            let BodyLiteral::Epistemic(lit) = lit else {
                continue;
            };
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "constraint[{constraint_index}] contains unsupported {} {}/{}; epistemic integrity constraints must be represented explicitly before {boundary}",
                    epistemic_literal_label(lit),
                    lit.atom.predicate,
                    lit.atom.arity()
                ),
            });
        }
    }
    Ok(())
}

fn epistemic_literal_label(lit: &EpistemicLiteral) -> &'static str {
    match (lit.negated, lit.op) {
        (false, EpistemicOp::Know) => "know",
        (false, EpistemicOp::Possible) => "possible",
        (true, EpistemicOp::Know) => "not know",
        (true, EpistemicOp::Possible) => "not possible",
    }
}

fn bound_output_columns_for_terms(
    key_terms: &[EirTerm],
    output_terms: &[EirTerm],
) -> Vec<Option<usize>> {
    key_terms
        .iter()
        .map(|term| match term {
            EirTerm::Variable(variable) => output_terms.iter().position(
                |head_term| matches!(head_term, EirTerm::Variable(name) if name == variable),
            ),
            _ => None,
        })
        .collect()
}

fn augmented_eir_head_terms(rule: &xlog_ir::EirRule) -> Vec<EirTerm> {
    let mut output_terms = rule.head.terms.clone();
    for lit in &rule.body {
        let EirBodyLiteral::Epistemic(lit) = lit else {
            continue;
        };
        for term in &lit.atom.terms {
            let EirTerm::Variable(variable) = term else {
                continue;
            };
            if !output_terms
                .iter()
                .any(|head_term| matches!(head_term, EirTerm::Variable(name) if name == variable))
            {
                output_terms.push(EirTerm::Variable(variable.clone()));
            }
        }
    }
    output_terms
}

fn final_output_columns_for_eir(eir: &EirProgram) -> Option<Vec<usize>> {
    let mut final_columns = Vec::new();
    let mut needs_projection = false;
    for rule in &eir.rules {
        if !rule
            .body
            .iter()
            .any(|lit| matches!(lit, EirBodyLiteral::Epistemic(_)))
        {
            continue;
        }
        let augmented_len = augmented_eir_head_terms(rule).len();
        if augmented_len > rule.head.terms.len() {
            needs_projection = true;
        }
        if final_columns.is_empty() {
            final_columns = (0..rule.head.terms.len()).collect();
        }
    }
    if needs_projection {
        Some(final_columns)
    } else {
        None
    }
}

/// Return the ordinary runtime program produced after epistemic GPU planning.
///
/// Epistemic literals are removed only for the reduced production runtime
/// dispatch; callers must still plan and certify the explicit epistemic GPU
/// contract before using this reduced program.
pub fn reduce_epistemic_program_to_ordinary(program: &Program) -> Program {
    let mut reduced = program.clone();

    for rule in &mut reduced.rules {
        append_body_local_tuple_key_variables_to_head(rule);
        let was_fact = rule.body.is_empty();
        let had_epistemic_body = rule
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)));
        rule.body
            .retain(|lit| !matches!(lit, BodyLiteral::Epistemic(_)));
        if !was_fact && had_epistemic_body && rule.body.is_empty() {
            rule.body.push(BodyLiteral::Comparison(Comparison {
                left: Term::Integer(1),
                op: CompOp::Eq,
                right: Term::Integer(1),
            }));
        }
    }
    for constraint in &mut reduced.constraints {
        constraint
            .body
            .retain(|lit| !matches!(lit, BodyLiteral::Epistemic(_)));
    }

    reduced
}

fn append_body_local_tuple_key_variables_to_head(rule: &mut crate::ast::Rule) {
    let mut hidden_variables = Vec::new();
    for lit in &rule.body {
        let BodyLiteral::Epistemic(lit) = lit else {
            continue;
        };
        for term in &lit.atom.terms {
            let Term::Variable(variable) = term else {
                continue;
            };
            if variable == "_" {
                continue;
            }
            let already_in_head = rule
                .head
                .terms
                .iter()
                .any(|head_term| matches!(head_term, Term::Variable(name) if name == variable));
            if !already_in_head && !hidden_variables.iter().any(|name| name == variable) {
                hidden_variables.push(variable.clone());
            }
        }
    }
    for variable in hidden_variables {
        rule.head.terms.push(Term::Variable(variable));
    }
}

fn wcoj_status_for_reduction(
    relational_body_atoms: usize,
    has_negated_relational_atom: bool,
) -> EpistemicWcojReductionStatus {
    if relational_body_atoms >= 3 && !has_negated_relational_atom {
        EpistemicWcojReductionStatus::RequiresPlannerEligibility
    } else {
        EpistemicWcojReductionStatus::NotWcojCandidate
    }
}

/// Result of bounded FAEEL candidate evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaeelCandidateResult {
    /// Candidate satisfies the bounded FAEEL fixture semantics.
    Model,
    /// Candidate has no model for a typed reason.
    NoModel(FaeelNoModelReason),
}

/// Typed no-model reason for bounded FAEEL fixtures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaeelNoModelReason {
    /// Candidate uses possible-only support where FAEEL requires founded knowledge.
    UnfoundedPossible {
        /// Predicate name.
        predicate: String,
        /// Predicate arity.
        arity: usize,
    },
    /// Candidate marks the same atom known and rejected.
    Contradiction {
        /// Predicate name.
        predicate: String,
        /// Predicate arity.
        arity: usize,
    },
    /// An epistemic literal is unsatisfied by the candidate.
    UnsatisfiedLiteral {
        /// Predicate name.
        predicate: String,
        /// Predicate arity.
        arity: usize,
    },
}

/// Configuration for bounded Generate-Propagate-Test fixture execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratePropagateTestConfig {
    /// Maximum candidate count accepted by the generate phase.
    pub max_candidates: usize,
}

/// Phase counters emitted by bounded Generate-Propagate-Test execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GeneratePropagateTestTrace {
    /// Number of generated candidates.
    pub generated: usize,
    /// Number of epistemic guesses generated.
    pub guesses: usize,
    /// Number of candidates that survived propagation.
    pub propagated: usize,
    /// Number of candidates pruned during propagation.
    pub pruned: usize,
    /// Number of reduced-program models inspected by the test phase.
    pub reduced_program_models: usize,
    /// Number of candidates tested.
    pub tested: usize,
    /// Number of accepted candidates.
    pub accepted: usize,
    /// Number of accepted world views.
    pub accepted_world_views: usize,
    /// Number of rejected candidates.
    pub rejected: usize,
    /// Rejection reasons observed during propagation and testing.
    pub rejection_reasons: Vec<FaeelNoModelReason>,
}

/// Result of bounded Generate-Propagate-Test fixture execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratePropagateTestOutcome {
    /// Phase counts.
    pub trace: GeneratePropagateTestTrace,
    /// Original indices of accepted candidates.
    pub accepted_candidate_indices: Vec<usize>,
    /// Original indices of rejected candidates in rejection-reason order.
    pub rejected_candidate_indices: Vec<usize>,
}

/// One deterministic dependency component for epistemic splitting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicDependencyComponent {
    /// Sorted predicate names in the component.
    pub predicates: Vec<String>,
    /// Source rule indices owned by the component.
    pub rule_indices: Vec<usize>,
}

/// Deterministic dependency graph used by bounded epistemic splitting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicDependencyGraph {
    /// Sorted components.
    pub components: Vec<EpistemicDependencyComponent>,
}

/// Split plan for independently solvable epistemic components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicSplitPlan {
    /// Components to solve independently.
    pub components: Vec<EpistemicDependencyComponent>,
}

impl EpistemicSplitPlan {
    /// Return the original rule order recovered from all components.
    pub fn recomposed_rule_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self
            .components
            .iter()
            .flat_map(|component| component.rule_indices.iter().copied())
            .collect();
        indices.sort_unstable();
        indices
    }
}

/// One split component lowered through the production epistemic GPU plan path.
#[derive(Debug, Clone)]
pub struct EpistemicSplitExecutableComponent {
    /// Source dependency component covered by this executable subplan.
    pub component: EpistemicDependencyComponent,
    /// GPU contract plus reduced runtime plan for this component.
    pub executable: EpistemicExecutablePlan,
}

/// Executable split plan whose components reuse the normal epistemic GPU lowering.
#[derive(Debug, Clone)]
pub struct EpistemicSplitExecutablePlan {
    /// Original bounded split certificate.
    pub split_plan: EpistemicSplitPlan,
    /// Epistemic components compiled into GPU executable subplans.
    pub components: Vec<EpistemicSplitExecutableComponent>,
}

impl EpistemicSplitExecutablePlan {
    /// Return the original rule order recovered from the split certificate.
    pub fn recomposed_rule_indices(&self) -> Vec<usize> {
        self.split_plan.recomposed_rule_indices()
    }

    /// Return executable components ordered by the first source rule they cover.
    pub fn recomposed_components(&self) -> Vec<&EpistemicSplitExecutableComponent> {
        let mut components: Vec<_> = self.components.iter().collect();
        components.sort_by_key(|component| {
            component
                .component
                .rule_indices
                .iter()
                .copied()
                .min()
                .unwrap_or(usize::MAX)
        });
        components
    }
}

/// Evaluate a single parsed epistemic literal against a bounded interpretation.
pub fn evaluate_epistemic_literal(
    mode: EpistemicMode,
    lit: &EpistemicLiteral,
    interpretation: &EpistemicInterpretation,
) -> TruthValue {
    let value = match lit.op {
        EpistemicOp::Know => interpretation.contains_known(&lit.atom),
        EpistemicOp::Possible => match mode {
            EpistemicMode::G91 => {
                interpretation.contains_known(&lit.atom)
                    || interpretation.contains_possible(&lit.atom)
            }
            EpistemicMode::Faeel => interpretation.contains_known(&lit.atom),
        },
    };

    TruthValue::from_bool(if lit.negated { !value } else { value })
}

/// Evaluate all epistemic literals in a program under bounded FAEEL fixture semantics.
pub fn evaluate_faeel_candidate(
    program: &Program,
    interpretation: &EpistemicInterpretation,
) -> Result<FaeelCandidateResult> {
    evaluate_epistemic_candidate(program, interpretation, EpistemicMode::Faeel)
}

/// Evaluate all epistemic literals in a program under a bounded fixture semantics mode.
pub fn evaluate_epistemic_candidate(
    program: &Program,
    interpretation: &EpistemicInterpretation,
    mode: EpistemicMode,
) -> Result<FaeelCandidateResult> {
    reject_gpt_epistemic_constraints(program)?;
    if let Some((predicate, arity)) = interpretation.first_contradiction() {
        return Ok(FaeelCandidateResult::NoModel(
            FaeelNoModelReason::Contradiction { predicate, arity },
        ));
    }

    for rule in &program.rules {
        for body_lit in &rule.body {
            let BodyLiteral::Epistemic(lit) = body_lit else {
                continue;
            };
            if interpretation.contains_known(&lit.atom)
                && interpretation.contains_rejected(&lit.atom)
            {
                return Ok(FaeelCandidateResult::NoModel(
                    FaeelNoModelReason::Contradiction {
                        predicate: lit.atom.predicate.clone(),
                        arity: lit.atom.arity(),
                    },
                ));
            }
            if mode == EpistemicMode::Faeel
                && lit.op == EpistemicOp::Possible
                && interpretation.contains_possible(&lit.atom)
                && !interpretation.contains_known(&lit.atom)
            {
                return Ok(FaeelCandidateResult::NoModel(
                    FaeelNoModelReason::UnfoundedPossible {
                        predicate: lit.atom.predicate.clone(),
                        arity: lit.atom.arity(),
                    },
                ));
            }
            if evaluate_epistemic_literal(mode, lit, interpretation) == TruthValue::False {
                return Ok(FaeelCandidateResult::NoModel(
                    FaeelNoModelReason::UnsatisfiedLiteral {
                        predicate: lit.atom.predicate.clone(),
                        arity: lit.atom.arity(),
                    },
                ));
            }
        }
    }

    Ok(FaeelCandidateResult::Model)
}

/// Run bounded Generate-Propagate-Test execution over explicit candidates.
pub fn run_generate_propagate_test(
    program: &Program,
    candidates: Vec<EpistemicInterpretation>,
    config: GeneratePropagateTestConfig,
) -> Result<GeneratePropagateTestOutcome> {
    run_generate_propagate_test_with_mode(
        program,
        candidates,
        config,
        program.directives.epistemic_mode_or_default(),
    )
}

/// Run bounded Generate-Propagate-Test execution over explicit candidates and semantics mode.
pub fn run_generate_propagate_test_with_mode(
    program: &Program,
    candidates: Vec<EpistemicInterpretation>,
    config: GeneratePropagateTestConfig,
    mode: EpistemicMode,
) -> Result<GeneratePropagateTestOutcome> {
    reject_gpt_epistemic_constraints(program)?;
    if candidates.len() > config.max_candidates {
        return Err(xlog_core::XlogError::ResourceExhausted {
            context: "epistemic GPT candidate guard".to_string(),
            estimated_bytes: candidates.len() as u64,
            budget_bytes: config.max_candidates as u64,
        });
    }

    let generated = candidates.len();
    let guesses = candidates
        .iter()
        .map(EpistemicInterpretation::epistemic_guess_count)
        .sum();
    let mut propagated_candidates = Vec::new();
    let mut rejection_reasons = Vec::new();
    let mut rejected_candidate_indices = Vec::new();
    for (idx, candidate) in candidates.into_iter().enumerate() {
        if let Some((predicate, arity)) = candidate.first_contradiction() {
            rejection_reasons.push(FaeelNoModelReason::Contradiction { predicate, arity });
            rejected_candidate_indices.push(idx);
        } else {
            propagated_candidates.push((idx, candidate));
        }
    }

    let mut trace = GeneratePropagateTestTrace {
        generated,
        guesses,
        propagated: propagated_candidates.len(),
        pruned: generated.saturating_sub(propagated_candidates.len()),
        reduced_program_models: propagated_candidates.len(),
        rejection_reasons,
        ..GeneratePropagateTestTrace::default()
    };
    let mut accepted_candidate_indices = Vec::new();

    for (idx, candidate) in &propagated_candidates {
        trace.tested += 1;
        match evaluate_epistemic_candidate(program, candidate, mode)? {
            FaeelCandidateResult::Model => {
                trace.accepted += 1;
                trace.accepted_world_views += 1;
                accepted_candidate_indices.push(*idx);
            }
            FaeelCandidateResult::NoModel(reason) => {
                trace.rejected += 1;
                trace.rejection_reasons.push(reason);
                rejected_candidate_indices.push(*idx);
            }
        }
    }

    Ok(GeneratePropagateTestOutcome {
        trace,
        accepted_candidate_indices,
        rejected_candidate_indices,
    })
}

/// Build a deterministic dependency graph for bounded epistemic splitting.
pub fn build_epistemic_dependency_graph(program: &Program) -> Result<EpistemicDependencyGraph> {
    if program.rules.is_empty() {
        return Ok(EpistemicDependencyGraph { components: vec![] });
    }

    let mut parents: Vec<usize> = (0..program.rules.len()).collect();
    let mut rule_predicates = Vec::with_capacity(program.rules.len());
    let mut head_owner: BTreeMap<String, usize> = BTreeMap::new();

    for (idx, rule) in program.rules.iter().enumerate() {
        if let Some(owner) = head_owner.get(&rule.head.predicate).copied() {
            union_components(&mut parents, owner, idx);
        } else {
            head_owner.insert(rule.head.predicate.clone(), idx);
        }
    }

    let mut modal_owner: BTreeMap<EpistemicAtomKey, usize> = BTreeMap::new();
    for (idx, rule) in program.rules.iter().enumerate() {
        let mut predicates = BTreeSet::new();
        predicates.insert(rule.head.predicate.clone());
        for lit in &rule.body {
            if let BodyLiteral::Epistemic(lit) = lit {
                let key =
                    EpistemicAtomKey::from_arity(lit.atom.predicate.clone(), lit.atom.arity());
                if let Some(owner) = modal_owner.get(&key).copied() {
                    union_components(&mut parents, owner, idx);
                } else {
                    modal_owner.insert(key, idx);
                }
            }
            if let Some(atom) = lit.atom() {
                if let Some(owner) = head_owner.get(&atom.predicate).copied() {
                    union_components(&mut parents, owner, idx);
                }
                predicates.insert(atom.predicate.clone());
            }
        }

        rule_predicates.push(predicates);
    }

    let mut constraint_predicates = Vec::with_capacity(program.constraints.len());
    for constraint in &program.constraints {
        let predicates = constraint_predicate_set(constraint);
        let mut owners = predicates
            .iter()
            .filter_map(|predicate| head_owner.get(predicate).copied());
        if let Some(first_owner) = owners.next() {
            for owner in owners {
                union_components(&mut parents, first_owner, owner);
            }
        }
        constraint_predicates.push(predicates);
    }

    let mut grouped: BTreeMap<usize, (BTreeSet<String>, Vec<usize>)> = BTreeMap::new();
    for (idx, predicates) in rule_predicates.into_iter().enumerate() {
        let root = find_component(&mut parents, idx);
        let entry = grouped
            .entry(root)
            .or_insert_with(|| (BTreeSet::new(), vec![]));
        entry.0.extend(predicates);
        entry.1.push(idx);
    }
    for predicates in constraint_predicates {
        let Some(root) = predicates
            .iter()
            .filter_map(|predicate| head_owner.get(predicate).copied())
            .map(|idx| find_component(&mut parents, idx))
            .next()
        else {
            continue;
        };
        grouped
            .entry(root)
            .or_insert_with(|| (BTreeSet::new(), vec![]))
            .0
            .extend(predicates);
    }

    let mut components: Vec<EpistemicDependencyComponent> = grouped
        .into_values()
        .map(|(predicates, mut rule_indices)| {
            rule_indices.sort_unstable();
            EpistemicDependencyComponent {
                predicates: predicates.into_iter().collect(),
                rule_indices,
            }
        })
        .collect();
    components.sort_by(|a, b| a.predicates.cmp(&b.predicates));
    Ok(EpistemicDependencyGraph { components })
}

fn constraint_predicate_set(constraint: &Constraint) -> BTreeSet<String> {
    constraint
        .body
        .iter()
        .filter_map(|lit| lit.atom().map(|atom| atom.predicate.clone()))
        .collect()
}

fn find_component(parents: &mut [usize], idx: usize) -> usize {
    if parents[idx] != idx {
        let root = find_component(parents, parents[idx]);
        parents[idx] = root;
    }
    parents[idx]
}

fn union_components(parents: &mut [usize], left: usize, right: usize) {
    let left_root = find_component(parents, left);
    let right_root = find_component(parents, right);
    if left_root != right_root {
        parents[right_root] = left_root;
    }
}

/// Split an epistemic program into independently solvable bounded components.
pub fn split_epistemic_program(program: &Program) -> Result<EpistemicSplitPlan> {
    for (idx, rule) in program.rules.iter().enumerate() {
        let epistemic_atoms: BTreeSet<String> = rule
            .body
            .iter()
            .filter_map(|lit| match lit {
                BodyLiteral::Epistemic(lit) => {
                    Some(format!("{}/{}", lit.atom.predicate, lit.atom.arity()))
                }
                _ => None,
            })
            .collect();
        if epistemic_atoms.len() > 1 {
            return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic splitting".to_string(),
                context: format!(
                    "rule[{idx}] couples epistemic predicates {:?}",
                    epistemic_atoms
                ),
            });
        }
    }

    Ok(EpistemicSplitPlan {
        components: build_epistemic_dependency_graph(program)?.components,
    })
}

/// Compile valid epistemic split components through the production GPU executable path.
pub fn compile_epistemic_gpu_split_execution(
    program: &Program,
) -> Result<EpistemicSplitExecutablePlan> {
    compile_epistemic_gpu_split_execution_with_stats_snapshot(program, None)
}

/// Compile valid epistemic split components with an optional production stats snapshot.
///
/// Each component subprogram is lowered through
/// [`compile_epistemic_gpu_execution_with_stats_snapshot`], so split execution
/// reuses the same GPU contract, reduced compiler pipeline, WCOJ promotion, and
/// helper-splitting surfaces as unsplit epistemic execution.
pub fn compile_epistemic_gpu_split_execution_with_stats_snapshot(
    program: &Program,
    stats_snapshot: Option<&StatsSnapshot>,
) -> Result<EpistemicSplitExecutablePlan> {
    reject_epistemic_constraints(program)?;
    let split_plan = split_epistemic_program(program)?;
    let mut components = Vec::new();

    for component in &split_plan.components {
        if !component_has_epistemic_rule(program, component) {
            continue;
        }

        let component_program = split_component_program(program, component)?;
        let executable = compile_epistemic_gpu_execution_with_stats_snapshot(
            &component_program,
            stats_snapshot,
        )?;
        components.push(EpistemicSplitExecutableComponent {
            component: component.clone(),
            executable,
        });
    }

    if components.is_empty() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "epistemic GPU split execution".to_string(),
            context: "requires at least one epistemic split component".to_string(),
        });
    }

    Ok(EpistemicSplitExecutablePlan {
        split_plan,
        components,
    })
}

fn component_has_epistemic_rule(
    program: &Program,
    component: &EpistemicDependencyComponent,
) -> bool {
    component
        .rule_indices
        .iter()
        .filter_map(|idx| program.rules.get(*idx))
        .any(|rule| {
            rule.body
                .iter()
                .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
        })
}

fn split_component_program(
    program: &Program,
    component: &EpistemicDependencyComponent,
) -> Result<Program> {
    let mut component_program = program.clone();
    let component_predicates: BTreeSet<&str> =
        component.predicates.iter().map(String::as_str).collect();
    let head_predicates: BTreeSet<&str> = program
        .rules
        .iter()
        .map(|rule| rule.head.predicate.as_str())
        .collect();
    component_program.rules = component
        .rule_indices
        .iter()
        .map(|idx| {
            program.rules.get(*idx).cloned().ok_or_else(|| {
                XlogError::Compilation(format!(
                    "epistemic split component references missing rule[{idx}]"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    component_program.constraints = program
        .constraints
        .iter()
        .filter(|constraint| {
            let predicates = constraint_predicate_set(constraint);
            let has_component_owned_predicate = predicates
                .iter()
                .any(|predicate| head_predicates.contains(predicate.as_str()));
            !has_component_owned_predicate
                || predicates
                    .iter()
                    .all(|predicate| component_predicates.contains(predicate.as_str()))
        })
        .cloned()
        .collect();
    Ok(component_program)
}
