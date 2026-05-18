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

    fn has_contradiction(&self) -> bool {
        self.known.iter().any(|key| self.rejected.contains(key))
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
    /// Number of candidates that survived propagation.
    pub propagated: usize,
    /// Number of candidates pruned during propagation.
    pub pruned: usize,
    /// Number of candidates tested.
    pub tested: usize,
    /// Number of accepted candidates.
    pub accepted: usize,
    /// Number of rejected candidates.
    pub rejected: usize,
}

/// Result of bounded Generate-Propagate-Test fixture execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratePropagateTestOutcome {
    /// Phase counts.
    pub trace: GeneratePropagateTestTrace,
    /// Original indices of accepted candidates.
    pub accepted_candidate_indices: Vec<usize>,
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
    let propagated_candidates: Vec<(usize, EpistemicInterpretation)> = candidates
        .into_iter()
        .enumerate()
        .filter(|(_, candidate)| !candidate.has_contradiction())
        .collect();

    let mut trace = GeneratePropagateTestTrace {
        generated,
        propagated: propagated_candidates.len(),
        pruned: generated.saturating_sub(propagated_candidates.len()),
        ..GeneratePropagateTestTrace::default()
    };
    let mut accepted_candidate_indices = Vec::new();

    for (idx, candidate) in &propagated_candidates {
        trace.tested += 1;
        match evaluate_faeel_candidate(program, candidate)? {
            FaeelCandidateResult::Model => {
                trace.accepted += 1;
                accepted_candidate_indices.push(*idx);
            }
            FaeelCandidateResult::NoModel(_) => {
                trace.rejected += 1;
            }
        }
    }

    Ok(GeneratePropagateTestOutcome {
        trace,
        accepted_candidate_indices,
    })
}
