//! Epistemic mode helpers for compatibility fixtures.

use std::collections::BTreeSet;

use xlog_core::Result;

use crate::ast::{BodyLiteral, EpistemicLiteral, EpistemicMode, EpistemicOp, Program};

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
            if lit.op == EpistemicOp::Possible
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
            if evaluate_epistemic_literal(EpistemicMode::Faeel, lit, interpretation)
                == TruthValue::False
            {
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
    for (idx, candidate) in candidates.into_iter().enumerate() {
        if let Some((predicate, arity)) = candidate.first_contradiction() {
            rejection_reasons.push(FaeelNoModelReason::Contradiction { predicate, arity });
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
        match evaluate_faeel_candidate(program, candidate)? {
            FaeelCandidateResult::Model => {
                trace.accepted += 1;
                trace.accepted_world_views += 1;
                accepted_candidate_indices.push(*idx);
            }
            FaeelCandidateResult::NoModel(reason) => {
                trace.rejected += 1;
                trace.rejection_reasons.push(reason);
            }
        }
    }

    Ok(GeneratePropagateTestOutcome {
        trace,
        accepted_candidate_indices,
    })
}

/// Build a deterministic dependency graph for bounded epistemic splitting.
pub fn build_epistemic_dependency_graph(program: &Program) -> Result<EpistemicDependencyGraph> {
    let mut components: Vec<EpistemicDependencyComponent> = Vec::new();

    for (idx, rule) in program.rules.iter().enumerate() {
        let mut predicates = BTreeSet::new();
        predicates.insert(rule.head.predicate.clone());
        for lit in &rule.body {
            if let BodyLiteral::Epistemic(lit) = lit {
                predicates.insert(lit.atom.predicate.clone());
            }
        }

        components.push(EpistemicDependencyComponent {
            predicates: predicates.into_iter().collect(),
            rule_indices: vec![idx],
        });
    }

    components.sort_by(|a, b| a.predicates.cmp(&b.predicates));
    Ok(EpistemicDependencyGraph { components })
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
