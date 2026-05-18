//! Epistemic GPU workspace allocation.

use cudarc::driver::LaunchConfig;
use xlog_core::{Result, XlogError};
use xlog_cuda::provider::{epistemic_kernels, HostTransferStats, EPISTEMIC_MODULE};
use xlog_cuda::{memory::TrackedCudaSlice, sys, CudaBuffer, DriverError, LaunchAsync};
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
            checked_product(
                checked_product(
                    capacities.max_candidates,
                    capacities.max_models_per_reduction,
                )?,
                reduction_count,
            )?,
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

/// Trace proving an epistemic GPU workspace was initialized on device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuWorkspaceResetTrace {
    /// Candidate-assumption bytes zeroed on device.
    pub candidate_assumption_bytes: usize,
    /// World-view bytes zeroed on device.
    pub world_view_bytes: usize,
    /// Model-membership bytes zeroed on device.
    pub model_membership_bytes: usize,
    /// Rejection-reason bytes zeroed on device.
    pub rejection_reason_bytes: usize,
    /// Device zeroing operations submitted by the reset path.
    pub device_zero_ops: u32,
    /// Host writes used by the reset path. Accepted GPU execution requires zero.
    pub host_write_ops: u32,
}

/// CUDA-event timing captured around one epistemic GPU kernel launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuKernelTimingTrace {
    /// CUDA event pairs recorded around the launch. Runtime traces require one.
    pub cuda_event_pairs: u32,
    /// CUDA event synchronizations used to make elapsed time observable on host.
    pub timing_sync_ops: u32,
    /// Event-measured stream elapsed time, converted from milliseconds to nanoseconds.
    pub kernel_elapsed_nanos: u64,
}

/// Trace proving candidate assumptions were generated by a GPU kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuCandidateGenerationTrace {
    /// Number of epistemic literals represented per candidate.
    pub literal_count: usize,
    /// Number of candidate rows generated on device.
    pub generated_candidates: usize,
    /// Candidate-assumption bytes written by the kernel.
    pub candidate_assumption_bytes: usize,
    /// Candidate-generation kernel launches.
    pub kernel_launches: u32,
    /// Host writes used by candidate generation. Accepted execution requires zero.
    pub host_write_ops: u32,
    /// CUDA-event timing for the launched kernel.
    pub kernel_timing: EpistemicGpuKernelTimingTrace,
}

/// Trace proving staged candidate buffers were validated by a GPU kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuCandidateValidationTrace {
    /// Number of epistemic literals represented per candidate.
    pub literal_count: usize,
    /// Number of candidate rows validated on device.
    pub validated_candidates: usize,
    /// Candidate-assumption bytes checked by the kernel.
    pub candidate_assumption_bytes_checked: usize,
    /// World-view staging bytes checked by the kernel.
    pub world_view_bytes_checked: usize,
    /// Rejection-reason slots written by the kernel.
    pub rejection_reason_slots_written: usize,
    /// Candidate-validation kernel launches.
    pub kernel_launches: u32,
    /// Host writes used by validation. Accepted GPU execution requires zero.
    pub host_write_ops: u32,
    /// CUDA-event timing for the launched kernel.
    pub kernel_timing: EpistemicGpuKernelTimingTrace,
}

/// Trace proving accepted-candidate materialization staging used a GPU kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuMaterializationTrace {
    /// Number of candidate rows materialized on device.
    pub materialized_candidates: usize,
    /// World-view slots written by the kernel.
    pub world_view_slots_written: usize,
    /// Materialization kernel launches.
    pub kernel_launches: u32,
    /// Host writes used by materialization. Accepted GPU execution requires zero.
    pub host_write_ops: u32,
    /// CUDA-event timing for the launched kernel.
    pub kernel_timing: EpistemicGpuKernelTimingTrace,
}

/// Trace proving final result flags were materialized from device-side output metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuFinalResultMaterializationTrace {
    /// Number of candidate rows materialized on device.
    pub materialized_candidates: usize,
    /// Device output row-count scalars read by the kernel.
    pub output_row_count_device_reads: u32,
    /// World-view result slots written by the kernel.
    pub world_view_slots_written: usize,
    /// Final-result materialization kernel launches.
    pub kernel_launches: u32,
    /// Host writes used by final-result materialization. Accepted execution requires zero.
    pub host_write_ops: u32,
    /// CUDA-event timing for the launched kernel.
    pub kernel_timing: EpistemicGpuKernelTimingTrace,
}

/// Trace proving the epistemic GPU hot path avoided tracked host transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuTransferBudgetTrace {
    /// Number of candidate rows covered by this transfer-budget check.
    pub candidate_count: usize,
    /// Tracked device-to-host bytes observed inside the GPU hot path.
    pub tracked_dtoh_bytes: u64,
    /// Tracked host-to-device bytes observed inside the GPU hot path.
    pub tracked_htod_bytes: u64,
    /// Tracked device-to-host calls observed inside the GPU hot path.
    pub tracked_dtoh_calls: u64,
    /// Tracked host-to-device calls observed inside the GPU hot path.
    pub tracked_htod_calls: u64,
    /// Per-candidate host round trips observed inside the GPU hot path.
    pub per_candidate_host_round_trips: u64,
}

/// Trace proving model-membership staging was performed by a GPU kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuModelMembershipTrace {
    /// Number of epistemic literals represented per candidate/model.
    pub literal_count: usize,
    /// Number of candidate rows checked on device.
    pub candidates_checked: usize,
    /// Number of reduced-program summaries represented in the membership layout.
    pub reduction_count: usize,
    /// Maximum models represented per reduction.
    pub models_per_reduction: usize,
    /// Model-membership bytes written by the kernel.
    pub model_membership_bytes_written: usize,
    /// Device output row-count scalars read by the kernel.
    pub output_row_count_device_reads: u32,
    /// Rejection-reason slots checked by the kernel.
    pub rejection_reason_slots_checked: usize,
    /// Model-membership staging kernel launches.
    pub kernel_launches: u32,
    /// Host writes used by model-membership staging. Accepted execution requires zero.
    pub host_write_ops: u32,
    /// CUDA-event timing for the launched kernel.
    pub kernel_timing: EpistemicGpuKernelTimingTrace,
}

/// Trace proving staged model memberships were validated against world views on GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuWorldViewValidationTrace {
    /// Number of epistemic literals represented per candidate/model.
    pub literal_count: usize,
    /// Number of candidate rows checked on device.
    pub candidates_checked: usize,
    /// Number of reduced-program summaries represented in the membership layout.
    pub reduction_count: usize,
    /// Maximum models represented per reduction.
    pub models_per_reduction: usize,
    /// Model-membership bytes checked by the kernel.
    pub model_membership_bytes_checked: usize,
    /// World-view staging slots checked by the kernel.
    pub world_view_slots_checked: usize,
    /// Rejection-reason slots written by the kernel.
    pub rejection_reason_slots_written: usize,
    /// World-view validation kernel launches.
    pub kernel_launches: u32,
    /// Host writes used by world-view validation. Accepted execution requires zero.
    pub host_write_ops: u32,
    /// CUDA-event timing for the launched kernel.
    pub kernel_timing: EpistemicGpuKernelTimingTrace,
}

/// Trace proving candidate propagation staging was performed by a GPU kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuPropagationTrace {
    /// Number of epistemic literals represented per candidate.
    pub literal_count: usize,
    /// Number of candidate rows propagated on device.
    pub propagated_candidates: usize,
    /// World-view staging bytes written by the kernel.
    pub world_view_bytes_written: usize,
    /// Rejection-reason slots initialized by the kernel.
    pub rejection_reason_slots_written: usize,
    /// Candidate-propagation kernel launches.
    pub kernel_launches: u32,
    /// Host writes used by propagation. Accepted GPU execution requires zero.
    pub host_write_ops: u32,
    /// CUDA-event timing for the launched kernel.
    pub kernel_timing: EpistemicGpuKernelTimingTrace,
}

impl EpistemicGpuKernelTimingTrace {
    /// Empty timing marker used before a runtime launch records CUDA events.
    pub const fn unrecorded() -> Self {
        Self {
            cuda_event_pairs: 0,
            timing_sync_ops: 0,
            kernel_elapsed_nanos: 0,
        }
    }

    /// Convert CUDA's native event elapsed time in milliseconds to a trace.
    pub fn from_cuda_elapsed_ms(elapsed_ms: f32) -> Result<Self> {
        if !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
            return Err(XlogError::Execution(format!(
                "invalid epistemic GPU kernel elapsed time: {elapsed_ms}"
            )));
        }

        Ok(Self {
            cuda_event_pairs: 1,
            timing_sync_ops: 1,
            kernel_elapsed_nanos: (elapsed_ms as f64 * 1_000_000.0).round() as u64,
        })
    }

    /// Whether CUDA-event timing was recorded for this trace.
    pub const fn is_recorded(&self) -> bool {
        self.cuda_event_pairs > 0
    }
}

impl EpistemicGpuCandidateGenerationTrace {
    /// Build a candidate-generation trace for a bounded device launch.
    pub fn for_counts(literal_count: usize, candidate_count: usize) -> Result<Self> {
        require_positive(literal_count, "epistemic GPU candidate literals")?;
        require_positive(candidate_count, "epistemic GPU candidate count")?;
        if literal_count > 31 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU candidate generation".to_string(),
                context: format!("literal count {literal_count} exceeds 31-bit candidate mask"),
            });
        }
        if candidate_count > (1usize << literal_count) {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU candidate count".to_string(),
                estimated_bytes: candidate_count as u64,
                budget_bytes: (1usize << literal_count) as u64,
            });
        }

        Ok(Self {
            literal_count,
            generated_candidates: candidate_count,
            candidate_assumption_bytes: checked_product(literal_count, candidate_count)?,
            kernel_launches: 1,
            host_write_ops: 0,
            kernel_timing: EpistemicGpuKernelTimingTrace::unrecorded(),
        })
    }

    /// Attach CUDA-event timing captured by the runtime launch path.
    pub const fn with_kernel_timing(
        mut self,
        kernel_timing: EpistemicGpuKernelTimingTrace,
    ) -> Self {
        self.kernel_timing = kernel_timing;
        self
    }
}

impl EpistemicGpuCandidateValidationTrace {
    /// Build a validation trace for a bounded device launch.
    pub fn for_counts(literal_count: usize, candidate_count: usize) -> Result<Self> {
        require_positive(literal_count, "epistemic GPU candidate validation literals")?;
        require_positive(
            candidate_count,
            "epistemic GPU candidate validation candidates",
        )?;

        Ok(Self {
            literal_count,
            validated_candidates: candidate_count,
            candidate_assumption_bytes_checked: checked_product(literal_count, candidate_count)?,
            world_view_bytes_checked: candidate_count,
            rejection_reason_slots_written: candidate_count,
            kernel_launches: 1,
            host_write_ops: 0,
            kernel_timing: EpistemicGpuKernelTimingTrace::unrecorded(),
        })
    }

    /// Attach CUDA-event timing captured by the runtime launch path.
    pub const fn with_kernel_timing(
        mut self,
        kernel_timing: EpistemicGpuKernelTimingTrace,
    ) -> Self {
        self.kernel_timing = kernel_timing;
        self
    }
}

impl EpistemicGpuMaterializationTrace {
    /// Build a materialization trace for a bounded device launch.
    pub fn for_count(candidate_count: usize) -> Result<Self> {
        require_positive(candidate_count, "epistemic GPU materialization candidates")?;

        Ok(Self {
            materialized_candidates: candidate_count,
            world_view_slots_written: candidate_count,
            kernel_launches: 1,
            host_write_ops: 0,
            kernel_timing: EpistemicGpuKernelTimingTrace::unrecorded(),
        })
    }

    /// Attach CUDA-event timing captured by the runtime launch path.
    pub const fn with_kernel_timing(
        mut self,
        kernel_timing: EpistemicGpuKernelTimingTrace,
    ) -> Self {
        self.kernel_timing = kernel_timing;
        self
    }
}

impl EpistemicGpuFinalResultMaterializationTrace {
    /// Build a final-result materialization trace for a bounded device launch.
    pub fn for_count(candidate_count: usize) -> Result<Self> {
        require_positive(
            candidate_count,
            "epistemic GPU final-result materialization candidates",
        )?;

        Ok(Self {
            materialized_candidates: candidate_count,
            output_row_count_device_reads: 1,
            world_view_slots_written: candidate_count,
            kernel_launches: 1,
            host_write_ops: 0,
            kernel_timing: EpistemicGpuKernelTimingTrace::unrecorded(),
        })
    }

    /// Attach CUDA-event timing captured by the runtime launch path.
    pub const fn with_kernel_timing(
        mut self,
        kernel_timing: EpistemicGpuKernelTimingTrace,
    ) -> Self {
        self.kernel_timing = kernel_timing;
        self
    }
}

impl EpistemicGpuTransferBudgetTrace {
    /// Build a hot-path transfer trace from provider host-transfer snapshots.
    pub fn from_host_transfer_stats(
        candidate_count: usize,
        before: HostTransferStats,
        after: HostTransferStats,
    ) -> Result<Self> {
        require_positive(candidate_count, "epistemic GPU transfer-budget candidates")?;

        let tracked_dtoh_bytes =
            transfer_counter_delta("dtoh_bytes", before.dtoh_bytes, after.dtoh_bytes)?;
        let tracked_htod_bytes =
            transfer_counter_delta("htod_bytes", before.htod_bytes, after.htod_bytes)?;
        let tracked_dtoh_calls =
            transfer_counter_delta("dtoh_calls", before.dtoh_calls, after.dtoh_calls)?;
        let tracked_htod_calls =
            transfer_counter_delta("htod_calls", before.htod_calls, after.htod_calls)?;

        if tracked_dtoh_bytes != 0
            || tracked_htod_bytes != 0
            || tracked_dtoh_calls != 0
            || tracked_htod_calls != 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU transfer budget".to_string(),
                context: format!(
                    "tracked host transfer in GPU hot path: dtoh_bytes={tracked_dtoh_bytes}, \
                     htod_bytes={tracked_htod_bytes}, dtoh_calls={tracked_dtoh_calls}, \
                     htod_calls={tracked_htod_calls}"
                ),
            });
        }

        Ok(Self {
            candidate_count,
            tracked_dtoh_bytes,
            tracked_htod_bytes,
            tracked_dtoh_calls,
            tracked_htod_calls,
            per_candidate_host_round_trips: 0,
        })
    }
}

fn transfer_counter_delta(name: &str, before: u64, after: u64) -> Result<u64> {
    after
        .checked_sub(before)
        .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
            construct: "epistemic GPU transfer budget".to_string(),
            context: format!(
                "host transfer counter decreased during GPU hot path: {name} before={before}, \
                 after={after}"
            ),
        })
}

impl EpistemicGpuModelMembershipTrace {
    /// Build a model-membership trace for a bounded device launch.
    pub fn for_counts(
        literal_count: usize,
        candidate_count: usize,
        reduction_count: usize,
        models_per_reduction: usize,
    ) -> Result<Self> {
        require_positive(literal_count, "epistemic GPU model-membership literals")?;
        require_positive(candidate_count, "epistemic GPU model-membership candidates")?;
        require_positive(reduction_count, "epistemic GPU model-membership reductions")?;
        require_positive(
            models_per_reduction,
            "epistemic GPU model-membership models",
        )?;
        let model_membership_bytes_written = checked_product(
            checked_product(
                checked_product(candidate_count, reduction_count)?,
                models_per_reduction,
            )?,
            literal_count,
        )?;

        Ok(Self {
            literal_count,
            candidates_checked: candidate_count,
            reduction_count,
            models_per_reduction,
            model_membership_bytes_written,
            output_row_count_device_reads: 1,
            rejection_reason_slots_checked: candidate_count,
            kernel_launches: 1,
            host_write_ops: 0,
            kernel_timing: EpistemicGpuKernelTimingTrace::unrecorded(),
        })
    }

    /// Attach CUDA-event timing captured by the runtime launch path.
    pub const fn with_kernel_timing(
        mut self,
        kernel_timing: EpistemicGpuKernelTimingTrace,
    ) -> Self {
        self.kernel_timing = kernel_timing;
        self
    }
}

impl EpistemicGpuWorldViewValidationTrace {
    /// Build a world-view validation trace for a bounded device launch.
    pub fn for_counts(
        literal_count: usize,
        candidate_count: usize,
        reduction_count: usize,
        models_per_reduction: usize,
    ) -> Result<Self> {
        require_positive(
            literal_count,
            "epistemic GPU world-view validation literals",
        )?;
        require_positive(
            candidate_count,
            "epistemic GPU world-view validation candidates",
        )?;
        require_positive(
            reduction_count,
            "epistemic GPU world-view validation reductions",
        )?;
        require_positive(
            models_per_reduction,
            "epistemic GPU world-view validation models",
        )?;
        let model_membership_bytes_checked = checked_product(
            checked_product(
                checked_product(candidate_count, reduction_count)?,
                models_per_reduction,
            )?,
            literal_count,
        )?;

        Ok(Self {
            literal_count,
            candidates_checked: candidate_count,
            reduction_count,
            models_per_reduction,
            model_membership_bytes_checked,
            world_view_slots_checked: candidate_count,
            rejection_reason_slots_written: candidate_count,
            kernel_launches: 1,
            host_write_ops: 0,
            kernel_timing: EpistemicGpuKernelTimingTrace::unrecorded(),
        })
    }

    /// Attach CUDA-event timing captured by the runtime launch path.
    pub const fn with_kernel_timing(
        mut self,
        kernel_timing: EpistemicGpuKernelTimingTrace,
    ) -> Self {
        self.kernel_timing = kernel_timing;
        self
    }
}

impl EpistemicGpuPropagationTrace {
    /// Build a propagation trace for a bounded device launch.
    pub fn for_counts(literal_count: usize, candidate_count: usize) -> Result<Self> {
        require_positive(literal_count, "epistemic GPU propagation literals")?;
        require_positive(candidate_count, "epistemic GPU propagation candidates")?;

        Ok(Self {
            literal_count,
            propagated_candidates: candidate_count,
            world_view_bytes_written: candidate_count,
            rejection_reason_slots_written: candidate_count,
            kernel_launches: 1,
            host_write_ops: 0,
            kernel_timing: EpistemicGpuKernelTimingTrace::unrecorded(),
        })
    }

    /// Attach CUDA-event timing captured by the runtime launch path.
    pub const fn with_kernel_timing(
        mut self,
        kernel_timing: EpistemicGpuKernelTimingTrace,
    ) -> Self {
        self.kernel_timing = kernel_timing;
        self
    }
}

impl EpistemicGpuWorkspaceResetTrace {
    /// Build the reset trace implied by a workspace layout.
    pub fn for_layout(layout: EpistemicGpuWorkspaceLayout) -> Self {
        Self {
            candidate_assumption_bytes: layout.candidate_assumption_bytes,
            world_view_bytes: layout.world_view_bytes,
            model_membership_bytes: layout.model_membership_bytes,
            rejection_reason_bytes: layout.rejection_reason_slots * std::mem::size_of::<u32>(),
            device_zero_ops: 4,
            host_write_ops: 0,
        }
    }

    /// Total bytes zeroed by the reset path.
    pub fn total_zeroed_bytes(&self) -> usize {
        self.candidate_assumption_bytes
            + self.world_view_bytes
            + self.model_membership_bytes
            + self.rejection_reason_bytes
    }
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
    /// Device-side initialization trace for the workspace buffers.
    pub workspace_reset: EpistemicGpuWorkspaceResetTrace,
}

/// Counter trace captured around a reduced production runtime dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuRuntimeTrace {
    /// Static preflight summary for the executed plan.
    pub preflight: EpistemicGpuRuntimePreflight,
    /// Runtime counters before dispatch.
    pub counters_before: EpistemicGpuRuntimeCounters,
    /// Runtime counters after dispatch.
    pub counters_after: EpistemicGpuRuntimeCounters,
    /// Saturating counter delta for the dispatch window.
    pub counter_delta: EpistemicGpuRuntimeCounters,
    /// WCOJ certification result derived from preflight obligations and deltas.
    pub wcoj_certification: EpistemicGpuRuntimeWcojCertification,
}

impl EpistemicGpuRuntimeTrace {
    /// Build a trace from static preflight data and runtime counter snapshots.
    pub fn from_preflight_and_counters(
        preflight: EpistemicGpuRuntimePreflight,
        counters_before: EpistemicGpuRuntimeCounters,
        counters_after: EpistemicGpuRuntimeCounters,
    ) -> Self {
        let counter_delta = counters_after.saturating_delta_since(counters_before);
        let wcoj_certification = EpistemicGpuRuntimeWcojCertification::for_preflight_and_delta(
            &preflight,
            &counter_delta,
        );

        Self {
            preflight,
            counters_before,
            counters_after,
            counter_delta,
            wcoj_certification,
        }
    }

    /// Fail closed when a WCOJ-required epistemic reduction lacks runtime evidence.
    pub fn require_wcoj_certification(&self) -> Result<()> {
        match self.wcoj_certification {
            EpistemicGpuRuntimeWcojCertification::MissingRequiredWcojDispatch {
                required_kclique_plans,
                observed_wcoj_dispatches,
            } => Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU WCOJ dispatch certification".to_string(),
                context: format!(
                    "required_kclique_plans={required_kclique_plans}, \
                     observed_wcoj_dispatches={observed_wcoj_dispatches}"
                ),
            }),
            EpistemicGpuRuntimeWcojCertification::NotRequired { .. }
            | EpistemicGpuRuntimeWcojCertification::Certified { .. } => Ok(()),
        }
    }
}

/// Runtime counters relevant to epistemic GPU certification.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuRuntimeCounters {
    /// Successful triangle WCOJ dispatches installed by the executor.
    pub wcoj_triangle_dispatch_count: u64,
    /// Successful 4-cycle WCOJ dispatches installed by the executor.
    pub wcoj_4cycle_dispatch_count: u64,
    /// Successful Goal-039 chain dispatches installed by the executor.
    pub w63_chain_dispatch_count: u64,
    /// Successful K=5 clique WCOJ dispatches installed by the executor.
    pub wcoj_clique5_dispatch_count: u64,
    /// Successful K=6 clique WCOJ dispatches installed by the executor.
    pub wcoj_clique6_dispatch_count: u64,
    /// Successful K=7 clique WCOJ dispatches installed by the executor.
    pub wcoj_clique7_dispatch_count: u64,
    /// Successful K=8 clique WCOJ dispatches installed by the executor.
    pub wcoj_clique8_dispatch_count: u64,
    /// Provider-level HG triangle dispatch counter.
    pub provider_wcoj_triangle_hg_dispatch_count: u64,
    /// WCOJ layout-sort invocations observed by the provider.
    pub wcoj_layout_sort_invocation_count: u64,
    /// WCOJ layout fast-path hits observed by the provider.
    pub wcoj_layout_fast_path_hit_count: u64,
    /// K-clique metadata builds observed by the provider.
    pub kclique_metadata_build_count: u64,
}

impl EpistemicGpuRuntimeCounters {
    /// Saturating delta from an earlier snapshot.
    pub fn saturating_delta_since(self, before: Self) -> Self {
        Self {
            wcoj_triangle_dispatch_count: self
                .wcoj_triangle_dispatch_count
                .saturating_sub(before.wcoj_triangle_dispatch_count),
            wcoj_4cycle_dispatch_count: self
                .wcoj_4cycle_dispatch_count
                .saturating_sub(before.wcoj_4cycle_dispatch_count),
            w63_chain_dispatch_count: self
                .w63_chain_dispatch_count
                .saturating_sub(before.w63_chain_dispatch_count),
            wcoj_clique5_dispatch_count: self
                .wcoj_clique5_dispatch_count
                .saturating_sub(before.wcoj_clique5_dispatch_count),
            wcoj_clique6_dispatch_count: self
                .wcoj_clique6_dispatch_count
                .saturating_sub(before.wcoj_clique6_dispatch_count),
            wcoj_clique7_dispatch_count: self
                .wcoj_clique7_dispatch_count
                .saturating_sub(before.wcoj_clique7_dispatch_count),
            wcoj_clique8_dispatch_count: self
                .wcoj_clique8_dispatch_count
                .saturating_sub(before.wcoj_clique8_dispatch_count),
            provider_wcoj_triangle_hg_dispatch_count: self
                .provider_wcoj_triangle_hg_dispatch_count
                .saturating_sub(before.provider_wcoj_triangle_hg_dispatch_count),
            wcoj_layout_sort_invocation_count: self
                .wcoj_layout_sort_invocation_count
                .saturating_sub(before.wcoj_layout_sort_invocation_count),
            wcoj_layout_fast_path_hit_count: self
                .wcoj_layout_fast_path_hit_count
                .saturating_sub(before.wcoj_layout_fast_path_hit_count),
            kclique_metadata_build_count: self
                .kclique_metadata_build_count
                .saturating_sub(before.kclique_metadata_build_count),
        }
    }

    /// Total WCOJ dispatches installed by the executor.
    pub fn wcoj_dispatch_count(&self) -> u64 {
        self.wcoj_triangle_dispatch_count
            + self.wcoj_4cycle_dispatch_count
            + self.wcoj_clique_dispatch_count()
    }

    /// Total K-clique WCOJ dispatches installed by the executor.
    pub fn wcoj_clique_dispatch_count(&self) -> u64 {
        self.wcoj_clique5_dispatch_count
            + self.wcoj_clique6_dispatch_count
            + self.wcoj_clique7_dispatch_count
            + self.wcoj_clique8_dispatch_count
    }
}

/// WCOJ certification status for an epistemic runtime dispatch attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicGpuRuntimeWcojCertification {
    /// The preflight did not require a K-clique WCOJ dispatch.
    NotRequired {
        /// Observed executor-installed WCOJ dispatches.
        observed_wcoj_dispatches: u64,
    },
    /// Runtime counters prove the required WCOJ dispatch happened.
    Certified {
        /// Observed executor-installed WCOJ dispatches.
        observed_wcoj_dispatches: u64,
        /// Observed executor-installed K-clique dispatches.
        observed_kclique_dispatches: u64,
        /// Observed provider WCOJ layout-sort invocations.
        observed_layout_sorts: u64,
        /// Observed provider K-clique metadata builds.
        observed_metadata_builds: u64,
    },
    /// The plan had K-clique WCOJ obligations, but counters did not advance.
    MissingRequiredWcojDispatch {
        /// K-clique WCOJ plans found during preflight.
        required_kclique_plans: usize,
        /// Observed executor-installed WCOJ dispatches.
        observed_wcoj_dispatches: u64,
    },
}

/// Output from executing the reduced production runtime plan for an epistemic program.
pub struct EpistemicGpuExecutionResult {
    /// Prepared workspace and preflight state.
    pub prepared: EpistemicGpuPreparedExecution,
    /// Candidate-generation trace captured before reduced-plan dispatch.
    pub candidate_generation: EpistemicGpuCandidateGenerationTrace,
    /// Candidate-propagation trace captured before reduced-plan dispatch.
    pub propagation: EpistemicGpuPropagationTrace,
    /// Candidate-validation trace captured before reduced-plan dispatch.
    pub candidate_validation: EpistemicGpuCandidateValidationTrace,
    /// Model-membership staging trace captured after reduced-plan dispatch.
    pub model_membership: EpistemicGpuModelMembershipTrace,
    /// World-view validation trace captured after model-membership staging.
    pub world_view_validation: EpistemicGpuWorldViewValidationTrace,
    /// Accepted-candidate materialization trace captured after world-view validation.
    pub materialization: EpistemicGpuMaterializationTrace,
    /// Final result materialization trace captured from reduced output metadata.
    pub final_result_materialization: EpistemicGpuFinalResultMaterializationTrace,
    /// Hot-path host-transfer budget trace for epistemic GPU execution.
    pub transfer_budget: EpistemicGpuTransferBudgetTrace,
    /// Output buffer returned by the reduced production execution plan.
    pub output: CudaBuffer,
    /// Runtime counter trace for the reduced production plan dispatch.
    pub trace: EpistemicGpuRuntimeTrace,
}

impl EpistemicGpuRuntimeWcojCertification {
    /// Compare static preflight obligations with runtime counter deltas.
    pub fn for_preflight_and_delta(
        preflight: &EpistemicGpuRuntimePreflight,
        delta: &EpistemicGpuRuntimeCounters,
    ) -> Self {
        let observed_wcoj_dispatches = delta.wcoj_dispatch_count();
        let observed_kclique_dispatches = delta.wcoj_clique_dispatch_count();

        if preflight.kclique_wcoj_plan_count == 0 {
            return Self::NotRequired {
                observed_wcoj_dispatches,
            };
        }

        if observed_kclique_dispatches < preflight.kclique_wcoj_plan_count as u64 {
            return Self::MissingRequiredWcojDispatch {
                required_kclique_plans: preflight.kclique_wcoj_plan_count,
                observed_wcoj_dispatches,
            };
        }

        Self::Certified {
            observed_wcoj_dispatches,
            observed_kclique_dispatches,
            observed_layout_sorts: delta.wcoj_layout_sort_invocation_count,
            observed_metadata_builds: delta.kclique_metadata_build_count,
        }
    }
}

impl Executor {
    /// Snapshot runtime counters used by epistemic GPU certification.
    pub fn epistemic_gpu_runtime_counters(&self) -> EpistemicGpuRuntimeCounters {
        EpistemicGpuRuntimeCounters {
            wcoj_triangle_dispatch_count: self.wcoj_triangle_dispatch_count(),
            wcoj_4cycle_dispatch_count: self.wcoj_4cycle_dispatch_count(),
            w63_chain_dispatch_count: self.w63_chain_dispatch_count(),
            wcoj_clique5_dispatch_count: self.wcoj_clique5_dispatch_count(),
            wcoj_clique6_dispatch_count: self.wcoj_clique6_dispatch_count(),
            wcoj_clique7_dispatch_count: self.wcoj_clique7_dispatch_count(),
            wcoj_clique8_dispatch_count: self.wcoj_clique8_dispatch_count(),
            provider_wcoj_triangle_hg_dispatch_count: self
                .provider
                .wcoj_triangle_hg_dispatch_count(),
            wcoj_layout_sort_invocation_count: self.provider.wcoj_layout_sort_invocation_count(),
            wcoj_layout_fast_path_hit_count: self.provider.wcoj_layout_fast_path_hit_count(),
            kclique_metadata_build_count: self.provider.kclique_metadata_build_count(),
        }
    }

    fn time_epistemic_gpu_kernel_launch(
        &self,
        operation: &str,
        launch: impl FnOnce() -> std::result::Result<(), DriverError>,
    ) -> Result<EpistemicGpuKernelTimingTrace> {
        let stream = self.provider.device().inner().stream().clone();
        let start = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| XlogError::execution_ctx(operation, "record start timing event", &e))?;
        launch().map_err(|e| XlogError::execution_ctx(operation, "launch kernel", &e))?;
        let end = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| XlogError::execution_ctx(operation, "record end timing event", &e))?;
        let elapsed_ms = start
            .elapsed_ms(&end)
            .map_err(|e| XlogError::execution_ctx(operation, "measure CUDA event elapsed", &e))?;

        EpistemicGpuKernelTimingTrace::from_cuda_elapsed_ms(elapsed_ms)
    }

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

    /// Zero every epistemic workspace buffer on device before hot-path use.
    pub fn reset_epistemic_gpu_workspace(
        &self,
        workspace: &mut EpistemicGpuWorkspace,
    ) -> Result<EpistemicGpuWorkspaceResetTrace> {
        let device = self.provider.device().inner();

        device
            .memset_zeros(&mut workspace.candidate_assumptions)
            .map_err(|e| {
                XlogError::execution_ctx(
                    "epistemic GPU workspace reset",
                    "candidate assumptions memset",
                    &e,
                )
            })?;
        device
            .memset_zeros(&mut workspace.world_views)
            .map_err(|e| {
                XlogError::execution_ctx("epistemic GPU workspace reset", "world views memset", &e)
            })?;
        device
            .memset_zeros(&mut workspace.model_membership)
            .map_err(|e| {
                XlogError::execution_ctx(
                    "epistemic GPU workspace reset",
                    "model membership memset",
                    &e,
                )
            })?;
        device
            .memset_zeros(&mut workspace.rejection_reasons)
            .map_err(|e| {
                XlogError::execution_ctx(
                    "epistemic GPU workspace reset",
                    "rejection reasons memset",
                    &e,
                )
            })?;

        Ok(EpistemicGpuWorkspaceResetTrace::for_layout(
            workspace.layout,
        ))
    }

    /// Generate candidate-assumption bitsets directly into the GPU workspace.
    pub fn generate_epistemic_gpu_candidates(
        &self,
        workspace: &mut EpistemicGpuWorkspace,
        literal_count: usize,
        candidate_count: usize,
    ) -> Result<EpistemicGpuCandidateGenerationTrace> {
        let trace =
            EpistemicGpuCandidateGenerationTrace::for_counts(literal_count, candidate_count)?;
        if trace.candidate_assumption_bytes > workspace.layout.candidate_assumption_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU candidate assumption workspace".to_string(),
                estimated_bytes: trace.candidate_assumption_bytes as u64,
                budget_bytes: workspace.layout.candidate_assumption_bytes as u64,
            });
        }
        if trace.candidate_assumption_bytes > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU candidate generation launch".to_string(),
                estimated_bytes: trace.candidate_assumption_bytes as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let literal_count = literal_count as u32;
        let candidate_count = candidate_count as u32;
        let total = trace.candidate_assumption_bytes as u32;
        let func = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_GENERATE_CANDIDATE_ASSUMPTIONS_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution("epistemic candidate generation kernel not found".to_string())
            })?;
        let config = LaunchConfig::for_num_elems(total);

        let kernel_timing = self.time_epistemic_gpu_kernel_launch(
            "epistemic GPU candidate generation",
            || unsafe {
                // SAFETY: kernel arguments match the PTX signature; the workspace capacity check
                // above proves the output buffer covers literal_count * candidate_count bytes.
                func.clone().launch(
                    config,
                    (
                        literal_count,
                        candidate_count,
                        &mut workspace.candidate_assumptions,
                    ),
                )
            },
        )?;

        Ok(trace.with_kernel_timing(kernel_timing))
    }

    /// Propagate generated candidates into GPU-resident world-view staging buffers.
    pub fn propagate_epistemic_gpu_candidates(
        &self,
        workspace: &mut EpistemicGpuWorkspace,
        literal_count: usize,
        candidate_count: usize,
    ) -> Result<EpistemicGpuPropagationTrace> {
        let trace = EpistemicGpuPropagationTrace::for_counts(literal_count, candidate_count)?;
        let candidate_assumption_bytes = checked_product(literal_count, candidate_count)?;
        if candidate_assumption_bytes > workspace.layout.candidate_assumption_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU propagation candidate workspace".to_string(),
                estimated_bytes: candidate_assumption_bytes as u64,
                budget_bytes: workspace.layout.candidate_assumption_bytes as u64,
            });
        }
        if trace.world_view_bytes_written > workspace.layout.world_view_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU propagation world-view workspace".to_string(),
                estimated_bytes: trace.world_view_bytes_written as u64,
                budget_bytes: workspace.layout.world_view_bytes as u64,
            });
        }
        if trace.rejection_reason_slots_written > workspace.layout.rejection_reason_slots {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU propagation rejection workspace".to_string(),
                estimated_bytes: trace.rejection_reason_slots_written as u64,
                budget_bytes: workspace.layout.rejection_reason_slots as u64,
            });
        }
        if literal_count > u32::MAX as usize || candidate_count > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU propagation launch".to_string(),
                estimated_bytes: literal_count.max(candidate_count) as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let world_stride =
            workspace.layout.world_view_bytes / workspace.layout.rejection_reason_slots;
        if world_stride == 0 || world_stride > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU propagation world stride".to_string(),
                estimated_bytes: world_stride as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let literal_count = literal_count as u32;
        let candidate_count = candidate_count as u32;
        let world_stride = world_stride as u32;
        let func = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_PROPAGATE_CANDIDATES_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution("epistemic candidate propagation kernel not found".to_string())
            })?;
        let config = LaunchConfig::for_num_elems(candidate_count);

        let kernel_timing = self.time_epistemic_gpu_kernel_launch(
            "epistemic GPU candidate propagation",
            || unsafe {
                // SAFETY: kernel arguments match the PTX signature; the capacity checks
                // above prove candidate, world-view, and rejection buffers cover all writes.
                func.clone().launch(
                    config,
                    (
                        literal_count,
                        candidate_count,
                        world_stride,
                        &workspace.candidate_assumptions,
                        &mut workspace.world_views,
                        &mut workspace.rejection_reasons,
                    ),
                )
            },
        )?;

        Ok(trace.with_kernel_timing(kernel_timing))
    }

    /// Validate staged candidate bitsets and world-view activity on device.
    pub fn validate_epistemic_gpu_candidates(
        &self,
        workspace: &mut EpistemicGpuWorkspace,
        literal_count: usize,
        candidate_count: usize,
    ) -> Result<EpistemicGpuCandidateValidationTrace> {
        let trace =
            EpistemicGpuCandidateValidationTrace::for_counts(literal_count, candidate_count)?;
        if trace.candidate_assumption_bytes_checked > workspace.layout.candidate_assumption_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU validation candidate workspace".to_string(),
                estimated_bytes: trace.candidate_assumption_bytes_checked as u64,
                budget_bytes: workspace.layout.candidate_assumption_bytes as u64,
            });
        }
        if trace.world_view_bytes_checked > workspace.layout.world_view_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU validation world-view workspace".to_string(),
                estimated_bytes: trace.world_view_bytes_checked as u64,
                budget_bytes: workspace.layout.world_view_bytes as u64,
            });
        }
        if trace.rejection_reason_slots_written > workspace.layout.rejection_reason_slots {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU validation rejection workspace".to_string(),
                estimated_bytes: trace.rejection_reason_slots_written as u64,
                budget_bytes: workspace.layout.rejection_reason_slots as u64,
            });
        }
        if literal_count > u32::MAX as usize || candidate_count > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU validation launch".to_string(),
                estimated_bytes: literal_count.max(candidate_count) as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let world_stride =
            workspace.layout.world_view_bytes / workspace.layout.rejection_reason_slots;
        if world_stride == 0 || world_stride > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU validation world stride".to_string(),
                estimated_bytes: world_stride as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let literal_count = literal_count as u32;
        let candidate_count = candidate_count as u32;
        let world_stride = world_stride as u32;
        let func = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_VALIDATE_CANDIDATE_BITS_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution("epistemic candidate validation kernel not found".to_string())
            })?;
        let config = LaunchConfig::for_num_elems(candidate_count);

        let kernel_timing = self.time_epistemic_gpu_kernel_launch(
            "epistemic GPU candidate validation",
            || unsafe {
                // SAFETY: kernel arguments match the PTX signature; the capacity checks
                // above prove candidate, world-view, and rejection buffers cover all accesses.
                func.clone().launch(
                    config,
                    (
                        literal_count,
                        candidate_count,
                        world_stride,
                        &workspace.candidate_assumptions,
                        &workspace.world_views,
                        &mut workspace.rejection_reasons,
                    ),
                )
            },
        )?;

        Ok(trace.with_kernel_timing(kernel_timing))
    }

    /// Populate candidate-scoped model-membership staging buffers on device.
    pub fn populate_epistemic_gpu_model_membership(
        &self,
        workspace: &mut EpistemicGpuWorkspace,
        output: &CudaBuffer,
        literal_count: usize,
        candidate_count: usize,
        reduction_count: usize,
        models_per_reduction: usize,
    ) -> Result<EpistemicGpuModelMembershipTrace> {
        let trace = EpistemicGpuModelMembershipTrace::for_counts(
            literal_count,
            candidate_count,
            reduction_count,
            models_per_reduction,
        )?;
        let candidate_assumption_bytes = checked_product(literal_count, candidate_count)?;
        if candidate_assumption_bytes > workspace.layout.candidate_assumption_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU model-membership candidate workspace".to_string(),
                estimated_bytes: candidate_assumption_bytes as u64,
                budget_bytes: workspace.layout.candidate_assumption_bytes as u64,
            });
        }
        if candidate_count > workspace.layout.world_view_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU model-membership world-view workspace".to_string(),
                estimated_bytes: candidate_count as u64,
                budget_bytes: workspace.layout.world_view_bytes as u64,
            });
        }
        if trace.model_membership_bytes_written > workspace.layout.model_membership_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU model-membership workspace".to_string(),
                estimated_bytes: trace.model_membership_bytes_written as u64,
                budget_bytes: workspace.layout.model_membership_bytes as u64,
            });
        }
        if trace.rejection_reason_slots_checked > workspace.layout.rejection_reason_slots {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU model-membership rejection workspace".to_string(),
                estimated_bytes: trace.rejection_reason_slots_checked as u64,
                budget_bytes: workspace.layout.rejection_reason_slots as u64,
            });
        }
        if trace.model_membership_bytes_written > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU model-membership launch".to_string(),
                estimated_bytes: trace.model_membership_bytes_written as u64,
                budget_bytes: u32::MAX as u64,
            });
        }
        if literal_count > u32::MAX as usize
            || candidate_count > u32::MAX as usize
            || reduction_count > u32::MAX as usize
            || models_per_reduction > u32::MAX as usize
        {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU model-membership dimensions".to_string(),
                estimated_bytes: literal_count
                    .max(candidate_count)
                    .max(reduction_count)
                    .max(models_per_reduction) as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let world_stride =
            workspace.layout.world_view_bytes / workspace.layout.rejection_reason_slots;
        if world_stride == 0 || world_stride > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU model-membership world stride".to_string(),
                estimated_bytes: world_stride as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let literal_count = literal_count as u32;
        let candidate_count = candidate_count as u32;
        let reduction_count = reduction_count as u32;
        let models_per_reduction = models_per_reduction as u32;
        let world_stride = world_stride as u32;
        let total = trace.model_membership_bytes_written as u32;
        let func = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution("epistemic model-membership kernel not found".to_string())
            })?;
        let config = LaunchConfig::for_num_elems(total);

        let kernel_timing =
            self.time_epistemic_gpu_kernel_launch("epistemic GPU model membership", || unsafe {
                // SAFETY: kernel arguments match the PTX signature; the capacity checks
                // above prove candidate, world-view, membership, and rejection buffers
                // cover all reads and writes.
                func.clone().launch(
                    config,
                    (
                        literal_count,
                        candidate_count,
                        reduction_count,
                        models_per_reduction,
                        world_stride,
                        output.num_rows_device(),
                        &workspace.candidate_assumptions,
                        &workspace.world_views,
                        &mut workspace.model_membership,
                        &mut workspace.rejection_reasons,
                    ),
                )
            })?;

        Ok(trace.with_kernel_timing(kernel_timing))
    }

    /// Validate staged model memberships against candidate world views on device.
    pub fn validate_epistemic_gpu_world_views(
        &self,
        workspace: &mut EpistemicGpuWorkspace,
        literal_count: usize,
        candidate_count: usize,
        reduction_count: usize,
        models_per_reduction: usize,
    ) -> Result<EpistemicGpuWorldViewValidationTrace> {
        let trace = EpistemicGpuWorldViewValidationTrace::for_counts(
            literal_count,
            candidate_count,
            reduction_count,
            models_per_reduction,
        )?;
        if trace.model_membership_bytes_checked > workspace.layout.model_membership_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU world-view validation membership workspace".to_string(),
                estimated_bytes: trace.model_membership_bytes_checked as u64,
                budget_bytes: workspace.layout.model_membership_bytes as u64,
            });
        }
        if trace.world_view_slots_checked > workspace.layout.world_view_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU world-view validation world-view workspace".to_string(),
                estimated_bytes: trace.world_view_slots_checked as u64,
                budget_bytes: workspace.layout.world_view_bytes as u64,
            });
        }
        if trace.rejection_reason_slots_written > workspace.layout.rejection_reason_slots {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU world-view validation rejection workspace".to_string(),
                estimated_bytes: trace.rejection_reason_slots_written as u64,
                budget_bytes: workspace.layout.rejection_reason_slots as u64,
            });
        }
        if trace.model_membership_bytes_checked > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU world-view validation membership launch".to_string(),
                estimated_bytes: trace.model_membership_bytes_checked as u64,
                budget_bytes: u32::MAX as u64,
            });
        }
        if literal_count > u32::MAX as usize
            || candidate_count > u32::MAX as usize
            || reduction_count > u32::MAX as usize
            || models_per_reduction > u32::MAX as usize
        {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU world-view validation dimensions".to_string(),
                estimated_bytes: literal_count
                    .max(candidate_count)
                    .max(reduction_count)
                    .max(models_per_reduction) as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let world_stride =
            workspace.layout.world_view_bytes / workspace.layout.rejection_reason_slots;
        if world_stride == 0 || world_stride > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU world-view validation world stride".to_string(),
                estimated_bytes: world_stride as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let literal_count = literal_count as u32;
        let candidate_count = candidate_count as u32;
        let reduction_count = reduction_count as u32;
        let models_per_reduction = models_per_reduction as u32;
        let world_stride = world_stride as u32;
        let func = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_VALIDATE_WORLD_VIEWS_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution("epistemic world-view validation kernel not found".to_string())
            })?;
        let config = LaunchConfig::for_num_elems(candidate_count);

        let kernel_timing = self.time_epistemic_gpu_kernel_launch(
            "epistemic GPU world-view validation",
            || unsafe {
                // SAFETY: kernel arguments match the PTX signature; the capacity checks
                // above prove model-membership, world-view, and rejection buffers cover
                // all reads and writes for the candidate range.
                func.clone().launch(
                    config,
                    (
                        literal_count,
                        candidate_count,
                        reduction_count,
                        models_per_reduction,
                        world_stride,
                        &workspace.model_membership,
                        &workspace.world_views,
                        &mut workspace.rejection_reasons,
                    ),
                )
            },
        )?;

        Ok(trace.with_kernel_timing(kernel_timing))
    }

    /// Materialize accepted candidate flags into the GPU world-view buffer.
    pub fn materialize_epistemic_gpu_candidates(
        &self,
        workspace: &mut EpistemicGpuWorkspace,
        candidate_count: usize,
    ) -> Result<EpistemicGpuMaterializationTrace> {
        let trace = EpistemicGpuMaterializationTrace::for_count(candidate_count)?;
        if trace.world_view_slots_written > workspace.layout.world_view_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU materialization world-view workspace".to_string(),
                estimated_bytes: trace.world_view_slots_written as u64,
                budget_bytes: workspace.layout.world_view_bytes as u64,
            });
        }
        if candidate_count > workspace.layout.rejection_reason_slots {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU materialization rejection workspace".to_string(),
                estimated_bytes: candidate_count as u64,
                budget_bytes: workspace.layout.rejection_reason_slots as u64,
            });
        }
        if candidate_count > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU materialization launch".to_string(),
                estimated_bytes: candidate_count as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let world_stride =
            workspace.layout.world_view_bytes / workspace.layout.rejection_reason_slots;
        if world_stride == 0 || world_stride > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU materialization world stride".to_string(),
                estimated_bytes: world_stride as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let candidate_count = candidate_count as u32;
        let world_stride = world_stride as u32;
        let func = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_MATERIALIZE_ACCEPTED_CANDIDATES_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution(
                    "epistemic candidate materialization kernel not found".to_string(),
                )
            })?;
        let config = LaunchConfig::for_num_elems(candidate_count);

        let kernel_timing = self.time_epistemic_gpu_kernel_launch(
            "epistemic GPU candidate materialization",
            || unsafe {
                // SAFETY: kernel arguments match the PTX signature; the capacity checks
                // above prove world-view and rejection buffers cover all accesses.
                func.clone().launch(
                    config,
                    (
                        candidate_count,
                        world_stride,
                        &workspace.rejection_reasons,
                        &mut workspace.world_views,
                    ),
                )
            },
        )?;

        Ok(trace.with_kernel_timing(kernel_timing))
    }

    /// Materialize final result flags from the reduced runtime output row count.
    pub fn materialize_epistemic_gpu_final_results(
        &self,
        workspace: &mut EpistemicGpuWorkspace,
        output: &CudaBuffer,
        candidate_count: usize,
    ) -> Result<EpistemicGpuFinalResultMaterializationTrace> {
        let trace = EpistemicGpuFinalResultMaterializationTrace::for_count(candidate_count)?;
        if trace.world_view_slots_written > workspace.layout.world_view_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final-result world-view workspace".to_string(),
                estimated_bytes: trace.world_view_slots_written as u64,
                budget_bytes: workspace.layout.world_view_bytes as u64,
            });
        }
        if candidate_count > workspace.layout.rejection_reason_slots {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final-result rejection workspace".to_string(),
                estimated_bytes: candidate_count as u64,
                budget_bytes: workspace.layout.rejection_reason_slots as u64,
            });
        }
        if candidate_count > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final-result launch".to_string(),
                estimated_bytes: candidate_count as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let world_stride =
            workspace.layout.world_view_bytes / workspace.layout.rejection_reason_slots;
        if world_stride == 0 || world_stride > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final-result world stride".to_string(),
                estimated_bytes: world_stride as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let candidate_count = candidate_count as u32;
        let world_stride = world_stride as u32;
        let func = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_MATERIALIZE_FINAL_RESULT_FLAGS_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution(
                    "epistemic final-result materialization kernel not found".to_string(),
                )
            })?;
        let config = LaunchConfig::for_num_elems(candidate_count);

        let kernel_timing = self.time_epistemic_gpu_kernel_launch(
            "epistemic GPU final result materialization",
            || unsafe {
                // SAFETY: kernel arguments match the PTX signature; the capacity checks
                // above prove world-view and rejection buffers cover all accesses, and
                // output.num_rows_device() is the runtime-owned device scalar for output
                // row count metadata.
                func.clone().launch(
                    config,
                    (
                        candidate_count,
                        world_stride,
                        output.num_rows_device(),
                        &workspace.rejection_reasons,
                        &mut workspace.world_views,
                    ),
                )
            },
        )?;

        Ok(trace.with_kernel_timing(kernel_timing))
    }

    /// Prepare runtime-owned GPU buffers for an epistemic executable plan.
    pub fn prepare_epistemic_gpu_execution(
        &self,
        executable: &EpistemicExecutablePlan,
        capacities: EpistemicGpuWorkspaceCapacities,
    ) -> Result<EpistemicGpuPreparedExecution> {
        let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(executable, capacities)?;
        let mut workspace =
            self.allocate_epistemic_gpu_workspace(&executable.gpu_plan, capacities)?;
        let workspace_reset = self.reset_epistemic_gpu_workspace(&mut workspace)?;

        Ok(EpistemicGpuPreparedExecution {
            preflight,
            workspace,
            workspace_reset,
        })
    }

    /// Execute the reduced production runtime plan and capture epistemic GPU evidence.
    pub fn execute_epistemic_gpu_execution(
        &mut self,
        executable: &EpistemicExecutablePlan,
        capacities: EpistemicGpuWorkspaceCapacities,
    ) -> Result<EpistemicGpuExecutionResult> {
        let mut prepared = self.prepare_epistemic_gpu_execution(executable, capacities)?;
        let literal_count = executable.gpu_plan.epistemic_literals.len();
        let candidate_count = bounded_candidate_count(literal_count, capacities.max_candidates)?;
        let transfer_budget_start = self.provider.host_transfer_stats();
        let candidate_generation = self.generate_epistemic_gpu_candidates(
            &mut prepared.workspace,
            literal_count,
            candidate_count,
        )?;
        let propagation = self.propagate_epistemic_gpu_candidates(
            &mut prepared.workspace,
            literal_count,
            candidate_count,
        )?;
        let candidate_validation = self.validate_epistemic_gpu_candidates(
            &mut prepared.workspace,
            literal_count,
            candidate_count,
        )?;
        let counters_before = self.epistemic_gpu_runtime_counters();
        let output = self.execute_plan(&executable.reduced_runtime_plan)?;
        let counters_after = self.epistemic_gpu_runtime_counters();
        let trace = EpistemicGpuRuntimeTrace::from_preflight_and_counters(
            prepared.preflight,
            counters_before,
            counters_after,
        );
        trace.require_wcoj_certification()?;
        let model_membership = self.populate_epistemic_gpu_model_membership(
            &mut prepared.workspace,
            &output,
            literal_count,
            candidate_count,
            executable.gpu_plan.reductions.len(),
            capacities.max_models_per_reduction,
        )?;
        let world_view_validation = self.validate_epistemic_gpu_world_views(
            &mut prepared.workspace,
            literal_count,
            candidate_count,
            executable.gpu_plan.reductions.len(),
            capacities.max_models_per_reduction,
        )?;
        let materialization =
            self.materialize_epistemic_gpu_candidates(&mut prepared.workspace, candidate_count)?;
        let final_result_materialization = self.materialize_epistemic_gpu_final_results(
            &mut prepared.workspace,
            &output,
            candidate_count,
        )?;
        let transfer_budget_end = self.provider.host_transfer_stats();
        let transfer_budget = EpistemicGpuTransferBudgetTrace::from_host_transfer_stats(
            candidate_count,
            transfer_budget_start,
            transfer_budget_end,
        )?;

        Ok(EpistemicGpuExecutionResult {
            prepared,
            candidate_generation,
            propagation,
            candidate_validation,
            model_membership,
            world_view_validation,
            materialization,
            final_result_materialization,
            transfer_budget,
            output,
            trace,
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

fn bounded_candidate_count(literal_count: usize, max_candidates: usize) -> Result<usize> {
    require_positive(literal_count, "epistemic GPU execution literals")?;
    require_positive(max_candidates, "epistemic GPU execution candidates")?;
    if literal_count > 31 {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "epistemic GPU execution candidate generation".to_string(),
            context: format!("literal count {literal_count} exceeds 31-bit candidate mask"),
        });
    }
    Ok(max_candidates.min(1usize << literal_count))
}
