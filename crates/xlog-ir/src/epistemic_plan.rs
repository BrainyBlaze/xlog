//! GPU-native epistemic execution planning contracts.

use crate::eir::{EirEpistemicLiteral, EirEpistemicMode, EirEpistemicOp};
use crate::plan::ExecutionPlan;

/// Generate-Propagate-Test hot-path phase that must execute on GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicGpuHotPathPhase {
    /// Candidate epistemic assumptions are generated on device.
    CandidateGeneration,
    /// Candidate assumptions are propagated into reduced programs on device.
    Propagation,
    /// Reduced-program stable models are checked against world-view guesses on device.
    WorldViewValidation,
    /// Accepted world views and query results are materialized from device buffers.
    ResultMaterialization,
}

/// GPU-resident buffer category required by accepted epistemic execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicGpuBufferKind {
    /// Candidate assumption bitsets.
    CandidateAssumptions,
    /// Accepted and candidate world-view bitsets.
    WorldViews,
    /// Per-model membership checks used by `know` and `possible`.
    ModelMembership,
    /// Structured rejection reasons for failed candidates.
    RejectionReasons,
}

/// WCOJ status for a reduced ordinary program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicWcojReductionStatus {
    /// The reduced body is too small or otherwise not a WCOJ candidate.
    NotWcojCandidate,
    /// The reduced body must be submitted to the production WCOJ planner.
    RequiresPlannerEligibility,
}

/// CPU fallback counters that must remain zero on the accepted hot path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EpistemicCpuFallbackCounters {
    /// CPU candidate enumeration count.
    pub candidate_enumeration: u64,
    /// CPU world-view validation count.
    pub world_view_validation: u64,
    /// CPU SAT/MaxSAT search count.
    pub solver_search: u64,
    /// CPU-only probabilistic recomputation count.
    pub probabilistic_recompute: u64,
}

impl EpistemicCpuFallbackCounters {
    /// Return true when every forbidden CPU fallback counter is zero.
    pub fn is_zero(&self) -> bool {
        self.candidate_enumeration == 0
            && self.world_view_validation == 0
            && self.solver_search == 0
            && self.probabilistic_recompute == 0
    }
}

/// One epistemic rule's reduced ordinary-program planning summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicReductionPlan {
    /// Source-order rule index.
    pub rule_index: usize,
    /// Positive relational body atom count after removing epistemic literals.
    pub relational_body_atoms: usize,
    /// WCOJ planner status for the reduced ordinary body.
    pub wcoj_status: EpistemicWcojReductionStatus,
}

/// Binding from an epistemic literal to reduced stable-model tuple evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicTupleMembershipBinding {
    /// Index of the epistemic literal in `EpistemicGpuPlan::epistemic_literals`.
    pub literal_index: usize,
    /// Index of the reduced rule in `EpistemicGpuPlan::reductions`.
    pub reduction_index: usize,
    /// Predicate whose stable-model tuples must be checked.
    pub predicate: String,
    /// Predicate arity whose stable-model tuples must be checked.
    pub arity: usize,
    /// Epistemic operator whose membership semantics are being checked.
    pub op: EirEpistemicOp,
    /// Whether the epistemic literal is explicitly negated.
    pub negated: bool,
}

/// Production-facing GPU execution contract for an epistemic program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicGpuPlan {
    /// Selected epistemic semantics mode.
    pub mode: EirEpistemicMode,
    /// Epistemic literals preserved from EIR.
    pub epistemic_literals: Vec<EirEpistemicLiteral>,
    /// GPU phases required by the hot path.
    pub required_phases: Vec<EpistemicGpuHotPathPhase>,
    /// GPU buffer classes required by the hot path.
    pub required_buffers: Vec<EpistemicGpuBufferKind>,
    /// Reduced ordinary-program planning summaries.
    pub reductions: Vec<EpistemicReductionPlan>,
    /// Per-literal stable-model tuple membership bindings.
    pub tuple_membership_bindings: Vec<EpistemicTupleMembershipBinding>,
    /// Forbidden CPU fallback counters. Release certification must keep these zero.
    pub cpu_fallbacks: EpistemicCpuFallbackCounters,
}

impl EpistemicGpuPlan {
    /// Create a plan with the standard GPU hot-path phase and buffer requirements.
    pub fn new(
        mode: EirEpistemicMode,
        epistemic_literals: Vec<EirEpistemicLiteral>,
        reductions: Vec<EpistemicReductionPlan>,
    ) -> Self {
        let tuple_membership_bindings = epistemic_literals
            .iter()
            .enumerate()
            .map(|(literal_index, literal)| EpistemicTupleMembershipBinding {
                literal_index,
                reduction_index: literal_index.min(reductions.len().saturating_sub(1)),
                predicate: literal.atom.predicate.clone(),
                arity: literal.atom.arity,
                op: literal.op,
                negated: literal.negated,
            })
            .collect();

        Self {
            mode,
            epistemic_literals,
            required_phases: vec![
                EpistemicGpuHotPathPhase::CandidateGeneration,
                EpistemicGpuHotPathPhase::Propagation,
                EpistemicGpuHotPathPhase::WorldViewValidation,
                EpistemicGpuHotPathPhase::ResultMaterialization,
            ],
            required_buffers: vec![
                EpistemicGpuBufferKind::CandidateAssumptions,
                EpistemicGpuBufferKind::WorldViews,
                EpistemicGpuBufferKind::ModelMembership,
                EpistemicGpuBufferKind::RejectionReasons,
            ],
            reductions,
            tuple_membership_bindings,
            cpu_fallbacks: EpistemicCpuFallbackCounters::default(),
        }
    }

    /// Replace inferred tuple-membership bindings with planner-derived bindings.
    pub fn with_tuple_membership_bindings(
        mut self,
        tuple_membership_bindings: Vec<EpistemicTupleMembershipBinding>,
    ) -> Self {
        self.tuple_membership_bindings = tuple_membership_bindings;
        self
    }

    /// Validate that every epistemic literal has a matching tuple-membership binding.
    pub fn validate_tuple_membership_bindings(&self) -> xlog_core::Result<()> {
        if self.tuple_membership_bindings.len() != self.epistemic_literals.len() {
            return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU tuple membership binding".to_string(),
                context: format!(
                    "expected {} bindings for epistemic literals, found {}",
                    self.epistemic_literals.len(),
                    self.tuple_membership_bindings.len()
                ),
            });
        }

        let mut seen_literals = vec![false; self.epistemic_literals.len()];

        for binding in &self.tuple_membership_bindings {
            if binding.literal_index >= self.epistemic_literals.len() {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU tuple membership binding".to_string(),
                    context: format!(
                        "literal_index {} exceeds literal count {}",
                        binding.literal_index,
                        self.epistemic_literals.len()
                    ),
                });
            }
            if seen_literals[binding.literal_index] {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU tuple membership binding".to_string(),
                    context: format!(
                        "duplicate literal_index {} in tuple-membership bindings",
                        binding.literal_index
                    ),
                });
            }
            seen_literals[binding.literal_index] = true;

            if binding.reduction_index >= self.reductions.len() {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU tuple membership binding".to_string(),
                    context: format!(
                        "reduction_index {} exceeds reduction count {}",
                        binding.reduction_index,
                        self.reductions.len()
                    ),
                });
            }

            let literal = &self.epistemic_literals[binding.literal_index];
            if binding.predicate != literal.atom.predicate
                || binding.arity != literal.atom.arity
                || binding.op != literal.op
                || binding.negated != literal.negated
            {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU tuple membership binding".to_string(),
                    context: format!(
                        "binding for literal_index {} does not match epistemic literal",
                        binding.literal_index
                    ),
                });
            }
        }

        Ok(())
    }
}

/// Production-facing executable plan for accepted epistemic lowering.
#[derive(Debug, Clone)]
pub struct EpistemicExecutablePlan {
    /// GPU semantic contract for the epistemic hot path.
    pub gpu_plan: EpistemicGpuPlan,
    /// Ordinary reduced program compiled through the production runtime pipeline.
    pub reduced_runtime_plan: ExecutionPlan,
}
