//! Epistemic GPU workspace allocation.

use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_ir::rir::{MultiwayPlan, RirNode};
use xlog_ir::{EpistemicCpuFallbackCounters, EpistemicExecutablePlan, EpistemicGpuPlan};

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

/// Runtime preflight summary for an epistemic executable plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuRuntimePreflight {
    /// GPU workspace layout required by the executable plan.
    pub workspace_layout: EpistemicGpuWorkspaceLayout,
    /// Compiled reduced-runtime rule count.
    pub reduced_runtime_rule_count: usize,
    /// Number of reduced rules carrying a `MultiWayJoin` route.
    pub multiway_reduction_count: usize,
    /// Number of K-clique WCOJ plans reused from the production planner.
    pub kclique_wcoj_plan_count: usize,
    /// Number of structured planned-hash routes.
    pub planned_hash_route_count: usize,
    /// Sorted-layout edge-slot requirements carried by WCOJ plans.
    pub sorted_layout_requirement_count: usize,
    /// Helper-splitting specs carried by WCOJ plans.
    pub helper_split_spec_count: usize,
    /// Forbidden CPU fallback counters copied from the GPU semantic contract.
    pub cpu_fallbacks: EpistemicCpuFallbackCounters,
}

impl EpistemicGpuRuntimePreflight {
    /// Inspect an executable epistemic plan before GPU kernel dispatch.
    pub fn for_executable_plan(
        executable: &EpistemicExecutablePlan,
        capacities: EpistemicGpuWorkspaceCapacities,
    ) -> Result<Self> {
        if !executable.gpu_plan.cpu_fallbacks.is_zero() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU runtime preflight".to_string(),
                context: "nonzero CPU fallback counters".to_string(),
            });
        }

        let workspace_layout =
            EpistemicGpuWorkspaceLayout::for_plan(&executable.gpu_plan, capacities)?;
        let mut routes = RuntimeRouteSummary::default();
        let mut reduced_runtime_rule_count = 0usize;

        for rule in executable
            .reduced_runtime_plan
            .rules_by_scc
            .iter()
            .flatten()
        {
            reduced_runtime_rule_count += 1;
            summarize_runtime_routes(&rule.body, &mut routes);
        }

        Ok(Self {
            workspace_layout,
            reduced_runtime_rule_count,
            multiway_reduction_count: routes.multiway_reduction_count,
            kclique_wcoj_plan_count: routes.kclique_wcoj_plan_count,
            planned_hash_route_count: routes.planned_hash_route_count,
            sorted_layout_requirement_count: routes.sorted_layout_requirement_count,
            helper_split_spec_count: routes.helper_split_spec_count,
            cpu_fallbacks: executable.gpu_plan.cpu_fallbacks,
        })
    }
}

/// Prepared runtime state for epistemic GPU execution.
pub struct EpistemicGpuPreparedExecution {
    /// Static preflight summary.
    pub preflight: EpistemicGpuRuntimePreflight,
    /// Device-resident workspace buffers.
    pub workspace: EpistemicGpuWorkspace,
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

    /// Prepare runtime-owned GPU buffers for an epistemic executable plan.
    pub fn prepare_epistemic_gpu_execution(
        &self,
        executable: &EpistemicExecutablePlan,
        capacities: EpistemicGpuWorkspaceCapacities,
    ) -> Result<EpistemicGpuPreparedExecution> {
        let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(executable, capacities)?;
        let workspace = self.allocate_epistemic_gpu_workspace(&executable.gpu_plan, capacities)?;

        Ok(EpistemicGpuPreparedExecution {
            preflight,
            workspace,
        })
    }
}

#[derive(Default)]
struct RuntimeRouteSummary {
    multiway_reduction_count: usize,
    kclique_wcoj_plan_count: usize,
    planned_hash_route_count: usize,
    sorted_layout_requirement_count: usize,
    helper_split_spec_count: usize,
}

fn summarize_runtime_routes(node: &RirNode, routes: &mut RuntimeRouteSummary) {
    match node {
        RirNode::MultiWayJoin {
            inputs,
            plan,
            fallback,
            ..
        } => {
            routes.multiway_reduction_count += 1;
            match plan {
                Some(MultiwayPlan::WcojWithPlan(order)) => {
                    routes.kclique_wcoj_plan_count += 1;
                    routes.sorted_layout_requirement_count +=
                        order.sorted_layout_requirements.edge_slots.len();
                    routes.helper_split_spec_count += order.helper_split_specs.len();
                }
                Some(MultiwayPlan::PlannedHashRoute { .. }) => {
                    routes.planned_hash_route_count += 1;
                }
                None => {}
            }

            for input in inputs {
                summarize_runtime_routes(input, routes);
            }
            summarize_runtime_routes(fallback, routes);
        }
        RirNode::Filter { input, .. }
        | RirNode::Project { input, .. }
        | RirNode::Distinct { input, .. }
        | RirNode::GroupBy { input, .. } => summarize_runtime_routes(input, routes),
        RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
            summarize_runtime_routes(left, routes);
            summarize_runtime_routes(right, routes);
        }
        RirNode::Union { inputs } => {
            for input in inputs {
                summarize_runtime_routes(input, routes);
            }
        }
        RirNode::Fixpoint {
            base, recursive, ..
        } => {
            summarize_runtime_routes(base, routes);
            summarize_runtime_routes(recursive, routes);
        }
        RirNode::ChainJoin {
            left,
            right,
            fallback,
            ..
        } => {
            summarize_runtime_routes(left, routes);
            summarize_runtime_routes(right, routes);
            summarize_runtime_routes(fallback, routes);
        }
        RirNode::TensorMaskedJoin { .. } | RirNode::Scan { .. } | RirNode::Unit => {}
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
