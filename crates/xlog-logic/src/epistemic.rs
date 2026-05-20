//! Epistemic mode helpers for compatibility fixtures.

use std::collections::{BTreeMap, BTreeSet};

use xlog_core::{Result, XlogError};
use xlog_ir::{
    EirBodyLiteral, EirEpistemicMode, EirEpistemicOp, EirProgram, EirTerm, EpistemicExecutablePlan,
    EpistemicGpuPlan, EpistemicReductionPlan, EpistemicTupleMembershipBinding,
    EpistemicWcojReductionStatus,
};
use xlog_stats::StatsSnapshot;

use crate::ast::{BodyLiteral, Constraint, EpistemicLiteral, EpistemicMode, EpistemicOp, Program};
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

/// Minimal interpretation used by G91/FAEEL distinction fixtures.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EpistemicInterpretation {
    known: BTreeSet<(String, usize)>,
    possible: BTreeSet<(String, usize)>,
    rejected: BTreeSet<(String, usize)>,
}

impl EpistemicInterpretation {
    /// Create an empty interpretation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a predicate/arity pair as known.
    pub fn with_known(mut self, predicate: impl Into<String>, arity: usize) -> Self {
        self.known.insert((predicate.into(), arity));
        self
    }

    /// Mark a predicate/arity pair as possible under G91 compatibility semantics.
    pub fn with_possible(mut self, predicate: impl Into<String>, arity: usize) -> Self {
        self.possible.insert((predicate.into(), arity));
        self
    }

    /// Mark a predicate/arity pair as rejected by the candidate.
    pub fn with_rejected(mut self, predicate: impl Into<String>, arity: usize) -> Self {
        self.rejected.insert((predicate.into(), arity));
        self
    }

    fn first_contradiction(&self) -> Option<(String, usize)> {
        self.known
            .iter()
            .find(|key| self.rejected.contains(*key))
            .cloned()
    }
}

/// One stable model in a bounded epistemic world-view fixture.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EpistemicWorld {
    facts: BTreeSet<(String, usize)>,
}

impl EpistemicWorld {
    /// Create an empty world.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a predicate/arity fact to this world.
    pub fn with_fact(mut self, predicate: impl Into<String>, arity: usize) -> Self {
        self.facts.insert((predicate.into(), arity));
        self
    }

    fn contains(&self, predicate: &str, arity: usize) -> bool {
        self.facts.contains(&(predicate.to_string(), arity))
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
        let predicate = lit.atom.predicate.as_str();
        let arity = lit.atom.arity();
        let value = match lit.op {
            EpistemicOp::Know => self
                .worlds
                .iter()
                .all(|world| world.contains(predicate, arity)),
            EpistemicOp::Possible => self
                .worlds
                .iter()
                .any(|world| world.contains(predicate, arity)),
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
            tuple_membership_bindings.push(EpistemicTupleMembershipBinding {
                literal_index,
                reduction_index,
                predicate: lit.atom.predicate.clone(),
                arity: lit.atom.arity,
                key_columns: (0..lit.atom.arity).collect(),
                key_terms: lit.atom.terms.clone(),
                bound_output_columns: bound_output_columns_for_literal(&lit.atom.terms, rule),
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

    Ok(
        EpistemicGpuPlan::new(eir.mode, epistemic_literals, reductions)
            .with_tuple_membership_bindings(tuple_membership_bindings),
    )
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
            if lit.op == EirEpistemicOp::Possible
                && !lit.negated
                && lit.atom.predicate == rule.head.predicate
                && lit.atom.arity == rule.head.arity
            {
                if lit.atom.arity > 0 {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "FAEEL foundedness guard".to_string(),
                        context: format!(
                            "rule[{rule_index}] has nonzero-arity self-supported possible {}/{} in default FAEEL mode; accepted GPU lowering requires tuple-level foundedness proof or explicit g91 compatibility mode",
                            lit.atom.predicate, lit.atom.arity
                        ),
                    });
                }
                if has_independent_founded_support(eir, &rule.head.predicate, rule.head.arity) {
                    continue;
                }
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "FAEEL foundedness guard".to_string(),
                    context: format!(
                        "rule[{rule_index}] has self-supported possible {}/{} in default FAEEL mode; use explicit g91 compatibility mode or provide independent founded support",
                        lit.atom.predicate, lit.atom.arity
                    ),
                });
            }
        }
    }

    Ok(())
}

fn has_independent_founded_support(eir: &EirProgram, predicate: &str, arity: usize) -> bool {
    eir.rules.iter().any(|rule| {
        rule.head.predicate == predicate
            && rule.head.arity == arity
            && rule
                .body
                .iter()
                .all(|lit| !matches!(lit, EirBodyLiteral::Epistemic(_)))
    })
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
    let reduced_program = reduced_ordinary_program(program);
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

fn reject_epistemic_constraints(program: &Program) -> Result<()> {
    for (constraint_index, constraint) in program.constraints.iter().enumerate() {
        for lit in &constraint.body {
            let BodyLiteral::Epistemic(lit) = lit else {
                continue;
            };
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU constraint".to_string(),
                context: format!(
                    "constraint[{constraint_index}] contains unsupported {} {}/{}; epistemic integrity constraints must be represented explicitly before GPU lowering",
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

fn bound_output_columns_for_literal(
    key_terms: &[EirTerm],
    rule: &xlog_ir::EirRule,
) -> Vec<Option<usize>> {
    key_terms
        .iter()
        .map(|term| match term {
            EirTerm::Variable(variable) => rule.head.terms.iter().position(
                |head_term| matches!(head_term, EirTerm::Variable(name) if name == variable),
            ),
            _ => None,
        })
        .collect()
}

fn reduced_ordinary_program(program: &Program) -> Program {
    let mut reduced = program.clone();

    for rule in &mut reduced.rules {
        rule.body
            .retain(|lit| !matches!(lit, BodyLiteral::Epistemic(_)));
    }
    for constraint in &mut reduced.constraints {
        constraint
            .body
            .retain(|lit| !matches!(lit, BodyLiteral::Epistemic(_)));
    }

    reduced
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
}

/// Evaluate a single parsed epistemic literal against a bounded interpretation.
pub fn evaluate_epistemic_literal(
    mode: EpistemicMode,
    lit: &EpistemicLiteral,
    interpretation: &EpistemicInterpretation,
) -> TruthValue {
    let key = (lit.atom.predicate.clone(), lit.atom.arity());
    let value = match lit.op {
        EpistemicOp::Know => interpretation.known.contains(&key),
        EpistemicOp::Possible => match mode {
            EpistemicMode::G91 => {
                interpretation.known.contains(&key) || interpretation.possible.contains(&key)
            }
            EpistemicMode::Faeel => interpretation.known.contains(&key),
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
    for rule in &program.rules {
        for body_lit in &rule.body {
            let BodyLiteral::Epistemic(lit) = body_lit else {
                continue;
            };
            let key = (lit.atom.predicate.clone(), lit.atom.arity());
            if interpretation.known.contains(&key) && interpretation.rejected.contains(&key) {
                return Ok(FaeelCandidateResult::NoModel(
                    FaeelNoModelReason::Contradiction {
                        predicate: key.0,
                        arity: key.1,
                    },
                ));
            }
            if mode == EpistemicMode::Faeel
                && lit.op == EpistemicOp::Possible
                && interpretation.possible.contains(&key)
                && !interpretation.known.contains(&key)
            {
                return Ok(FaeelCandidateResult::NoModel(
                    FaeelNoModelReason::UnfoundedPossible {
                        predicate: key.0,
                        arity: key.1,
                    },
                ));
            }
            if evaluate_epistemic_literal(mode, lit, interpretation) == TruthValue::False {
                return Ok(FaeelCandidateResult::NoModel(
                    FaeelNoModelReason::UnsatisfiedLiteral {
                        predicate: key.0,
                        arity: key.1,
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
    run_generate_propagate_test_with_mode(program, candidates, config, EpistemicMode::Faeel)
}

/// Run bounded Generate-Propagate-Test execution over explicit candidates and semantics mode.
pub fn run_generate_propagate_test_with_mode(
    program: &Program,
    candidates: Vec<EpistemicInterpretation>,
    config: GeneratePropagateTestConfig,
    mode: EpistemicMode,
) -> Result<GeneratePropagateTestOutcome> {
    if candidates.len() > config.max_candidates {
        return Err(xlog_core::XlogError::ResourceExhausted {
            context: "epistemic GPT candidate guard".to_string(),
            estimated_bytes: candidates.len() as u64,
            budget_bytes: config.max_candidates as u64,
        });
    }

    let generated = candidates.len();
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
        guesses: generated,
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

    for (idx, rule) in program.rules.iter().enumerate() {
        let mut predicates = BTreeSet::new();
        predicates.insert(rule.head.predicate.clone());
        for lit in &rule.body {
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
        let epistemic_predicates: BTreeSet<&str> = rule
            .body
            .iter()
            .filter_map(|lit| match lit {
                BodyLiteral::Epistemic(lit) => Some(lit.atom.predicate.as_str()),
                _ => None,
            })
            .collect();
        if epistemic_predicates.len() > 1 {
            return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic splitting".to_string(),
                context: format!(
                    "rule[{idx}] couples epistemic predicates {:?}",
                    epistemic_predicates
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
            !predicates.is_empty()
                && predicates
                    .iter()
                    .all(|predicate| component_predicates.contains(predicate.as_str()))
        })
        .cloned()
        .collect();
    Ok(component_program)
}
