//! Epistemic GPU workspace allocation.

use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_ir::EpistemicGpuPlan;

use super::Executor;

/// Capacity limits for an epistemic GPU workspace allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuWorkspaceCapacities {
    /// Maximum generated epistemic candidates.
    pub max_candidates: usize,
    /// Maximum worlds tracked per candidate.
    pub max_worlds: usize,
    /// Maximum reduced-program models tracked per reduction.
    pub max_models_per_reduction: usize,
}

/// Concrete device-buffer layout for an epistemic GPU workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuWorkspaceLayout {
    /// Candidate-assumption buffer size in bytes.
    pub candidate_assumption_bytes: usize,
    /// World-view buffer size in bytes.
    pub world_view_bytes: usize,
    /// Model-membership buffer size in bytes.
    pub model_membership_bytes: usize,
    /// Rejection-reason slot count.
    pub rejection_reason_slots: usize,
}

impl EpistemicGpuWorkspaceLayout {
    /// Build a workspace layout from an epistemic GPU plan and capacity limits.
    pub fn for_plan(
        plan: &EpistemicGpuPlan,
        capacities: EpistemicGpuWorkspaceCapacities,
    ) -> Result<Self> {
        require_positive(
            capacities.max_candidates,
            "epistemic GPU workspace candidates",
        )?;
        require_positive(capacities.max_worlds, "epistemic GPU workspace worlds")?;
        require_positive(
            capacities.max_models_per_reduction,
            "epistemic GPU workspace models",
        )?;
        require_positive(
            plan.epistemic_literals.len(),
            "epistemic GPU workspace literals",
        )?;
        require_positive(plan.reductions.len(), "epistemic GPU workspace reductions")?;

        let literal_count = plan.epistemic_literals.len();
        let reduction_count = plan.reductions.len();
        let candidate_assumption_bytes = checked_product(capacities.max_candidates, literal_count)?;
        let world_view_bytes = checked_product(capacities.max_candidates, capacities.max_worlds)?;
        let model_membership_bytes = checked_product(
            checked_product(capacities.max_models_per_reduction, reduction_count)?,
            literal_count,
        )?;

        Ok(Self {
            candidate_assumption_bytes,
            world_view_bytes,
            model_membership_bytes,
            rejection_reason_slots: capacities.max_candidates,
        })
    }

    /// Total workspace byte size across every device buffer category.
    pub fn total_bytes(&self) -> usize {
        self.candidate_assumption_bytes
            + self.world_view_bytes
            + self.model_membership_bytes
            + self.rejection_reason_slots * std::mem::size_of::<u32>()
    }
}

/// Device-resident buffers for epistemic Generate-Propagate-Test execution.
pub struct EpistemicGpuWorkspace {
    /// Workspace layout used for allocation.
    pub layout: EpistemicGpuWorkspaceLayout,
    /// Candidate-assumption bitset buffer.
    pub candidate_assumptions: TrackedCudaSlice<u8>,
    /// Candidate and accepted world-view bitset buffer.
    pub world_views: TrackedCudaSlice<u8>,
    /// Per-model membership check buffer.
    pub model_membership: TrackedCudaSlice<u8>,
    /// Structured rejection-reason code buffer.
    pub rejection_reasons: TrackedCudaSlice<u32>,
}

impl Executor {
    /// Allocate GPU-resident buffers required by an epistemic GPU plan.
    pub fn allocate_epistemic_gpu_workspace(
        &self,
        plan: &EpistemicGpuPlan,
        capacities: EpistemicGpuWorkspaceCapacities,
    ) -> Result<EpistemicGpuWorkspace> {
        let layout = EpistemicGpuWorkspaceLayout::for_plan(plan, capacities)?;
        let memory = self.provider.memory();

        Ok(EpistemicGpuWorkspace {
            layout,
            candidate_assumptions: memory.alloc::<u8>(layout.candidate_assumption_bytes)?,
            world_views: memory.alloc::<u8>(layout.world_view_bytes)?,
            model_membership: memory.alloc::<u8>(layout.model_membership_bytes)?,
            rejection_reasons: memory.alloc::<u32>(layout.rejection_reason_slots)?,
        })
    }
}

fn require_positive(value: usize, context: &str) -> Result<()> {
    if value == 0 {
        return Err(XlogError::ResourceExhausted {
            context: context.to_string(),
            estimated_bytes: 0,
            budget_bytes: 1,
        });
    }
    Ok(())
}

fn checked_product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        XlogError::Kernel(format!(
            "epistemic GPU workspace size overflow: {left} * {right}"
        ))
    })
}
