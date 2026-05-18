//! GPU-native epistemic execution planning contracts.

use crate::eir::{EirEpistemicLiteral, EirEpistemicMode};
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
            cpu_fallbacks: EpistemicCpuFallbackCounters::default(),
        }
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
