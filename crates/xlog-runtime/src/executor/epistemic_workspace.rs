//! Epistemic GPU workspace allocation.

use std::{collections::BTreeSet, ffi::c_void};

use cudarc::driver::LaunchConfig;
use xlog_core::{RelId, Result, ScalarType, XlogError};
use xlog_cuda::provider::{epistemic_kernels, HostTransferStats, EPISTEMIC_MODULE};
use xlog_cuda::{
    memory::TrackedCudaSlice, sys, AsKernelParam, CudaBuffer, CudaColumn, DeviceSlice, DriverError,
    LaunchAsync,
};
use xlog_ir::rir::{MultiwayPlan, RirNode, StreamGroupId};
use xlog_ir::{
    EirEpistemicMode, EirEpistemicOp, EirTerm, EpistemicCpuFallbackCounters,
    EpistemicExecutablePlan, EpistemicGpuPlan,
};

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

/// Trace proving final query tuples were materialized into a device-resident buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuFinalTupleMaterializationTrace {
    /// Number of output columns copied into the final device buffer.
    pub output_column_count: usize,
    /// Row capacity of the final output buffer.
    pub output_row_capacity: usize,
    /// Device tuple bytes covered by the materialization kernels.
    pub tuple_bytes_capacity: usize,
    /// Device output row-count scalars read by the kernels.
    pub output_row_count_device_reads: u32,
    /// Model-membership bytes checked by the kernels before tuple materialization.
    pub model_membership_bytes_checked: usize,
    /// World-view slots checked by the kernels before tuple materialization.
    pub world_view_slots_checked: usize,
    /// Variable-bound tuple row filters applied by the final-row map kernel.
    pub row_filter_count: usize,
    /// Negated variable-bound tuple row filters applied by the final-row map kernel.
    pub negated_row_filter_count: usize,
    /// Device final row-count scalars written by the kernels.
    pub final_row_count_device_writes: u32,
    /// Final tuple materialization kernel launches.
    pub kernel_launches: u32,
    /// Host writes used by final tuple materialization. Accepted execution requires zero.
    pub host_write_ops: u32,
    /// CUDA-event timing for the launched kernel batch.
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

/// Trace accounting for the bounded final-result transfer after the GPU hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuFinalResultTransferTrace {
    /// Logical rows in the final device-resident output buffer.
    pub final_output_rows: usize,
    /// Number of final output columns that a caller may export.
    pub final_output_column_count: usize,
    /// Bytes in one final output row.
    pub final_output_row_width_bytes: usize,
    /// Bounded data-plane payload bytes represented by the final output.
    pub final_output_payload_bytes: u64,
    /// Device row-count metadata reads used for this accounting.
    pub row_count_device_reads: u32,
    /// Data-plane D2H calls issued by accepted execution. Execution returns a device buffer, so this is zero.
    pub tracked_data_plane_dtoh_calls: u64,
    /// Data-plane D2H bytes issued by accepted execution. Execution returns a device buffer, so this is zero.
    pub tracked_data_plane_dtoh_bytes: u64,
}

/// Typed interpretation of nonzero GPU epistemic rejection codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicGpuRejectionReason {
    /// Candidate was rejected because its world-view row was inactive.
    InactiveWorld,
    /// Candidate buffer contained a value outside the valid boolean bit range.
    InvalidCandidateBit,
    /// Candidate did not have a reduced-model tuple source to validate against.
    MissingReducedModel,
    /// Candidate assumptions were not supported by model-membership evidence.
    UnsatisfiedMembership,
}

impl EpistemicGpuRejectionReason {
    /// Return the raw device rejection code used by the CUDA kernels.
    pub const fn code(self) -> u32 {
        match self {
            Self::InactiveWorld => 2,
            Self::InvalidCandidateBit => 3,
            Self::MissingReducedModel => 4,
            Self::UnsatisfiedMembership => 5,
        }
    }

    /// Decode a nonzero device rejection code into a typed reason.
    pub fn from_code(code: u32) -> Result<Self> {
        match code {
            2 => Ok(Self::InactiveWorld),
            3 => Ok(Self::InvalidCandidateBit),
            4 => Ok(Self::MissingReducedModel),
            5 => Ok(Self::UnsatisfiedMembership),
            other => Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU rejection reason".to_string(),
                context: format!("unknown device rejection code {other}"),
            }),
        }
    }
}

/// Device-derived semantic summary for Generate-Propagate-Test execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicGpuSemanticTrace {
    /// Number of candidate rows generated on device.
    pub generated_candidates: usize,
    /// Number of epistemic guesses represented by generated candidate rows.
    pub guesses: usize,
    /// Number of candidate rows propagated on device.
    pub propagated_candidates: usize,
    /// Number of generated candidates not propagated.
    pub pruned_candidates: usize,
    /// Number of candidate rows checked by world-view validation.
    pub tested_candidates: usize,
    /// Number of reduced model slots checked by model-membership/world-view kernels.
    pub reduced_model_slots_checked: usize,
    /// Number of accepted candidates observed in the device rejection buffer.
    pub accepted_candidates: usize,
    /// Candidate indices accepted by the device rejection buffer.
    pub accepted_candidate_indices: Vec<usize>,
    /// Number of accepted world views represented by accepted candidates.
    pub accepted_world_views: usize,
    /// Number of rejected candidates observed in the device rejection buffer.
    pub rejected_candidates: usize,
    /// Candidate indices rejected by the device rejection buffer.
    pub rejected_candidate_indices: Vec<usize>,
    /// Nonzero rejection reason codes copied from the device rejection buffer.
    pub rejection_reasons: Vec<u32>,
    /// Bounded metadata reads from the device rejection buffer after the hot path.
    pub rejection_reason_device_reads: u32,
    /// Bytes read as bounded rejection-reason metadata after the hot path.
    pub rejection_reason_metadata_bytes: u64,
    /// CPU candidate enumerations used by the accepted path.
    pub cpu_candidate_enumerations: u32,
    /// CPU world-view validations used by the accepted path.
    pub cpu_world_view_validations: u32,
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
    /// Device tuple-source row-count scalars read by the kernel.
    pub tuple_source_row_count_device_reads: u32,
    /// Device tuple-key columns read by tuple-source membership kernels.
    pub tuple_source_key_column_device_reads: u32,
    /// Rejection-reason slots checked by the kernel.
    pub rejection_reason_slots_checked: usize,
    /// Source used to populate model-membership bytes.
    pub membership_source: EpistemicGpuModelMembershipSource,
    /// Model-membership staging kernel launches.
    pub kernel_launches: u32,
    /// Host writes used by model-membership staging. Accepted execution requires zero.
    pub host_write_ops: u32,
    /// CUDA-event timing for the launched kernel.
    pub kernel_timing: EpistemicGpuKernelTimingTrace,
}

/// Source of GPU model-membership bytes for epistemic world-view validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicGpuModelMembershipSource {
    /// Current bounded staging only proves the reduced output has rows.
    ReducedOutputRowCountOnly,
    /// Model-membership bytes were populated from reduced stable-model tuple buffers.
    StableModelTupleBuffer,
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

impl EpistemicGpuFinalTupleMaterializationTrace {
    /// Build a final tuple materialization trace for a device-side output buffer.
    pub fn for_counts(
        output_column_count: usize,
        output_row_capacity: usize,
        tuple_bytes_capacity: usize,
        literal_count: usize,
        candidate_count: usize,
        reduction_count: usize,
        models_per_reduction: usize,
    ) -> Result<Self> {
        if output_column_count > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final-tuple output columns".to_string(),
                estimated_bytes: output_column_count as u64,
                budget_bytes: u32::MAX as u64,
            });
        }
        require_positive(literal_count, "epistemic GPU final-tuple literals")?;
        require_positive(candidate_count, "epistemic GPU final-tuple candidates")?;
        require_positive(reduction_count, "epistemic GPU final-tuple reductions")?;
        require_positive(models_per_reduction, "epistemic GPU final-tuple models")?;
        let model_membership_bytes_checked = checked_product(
            checked_product(
                checked_product(candidate_count, reduction_count)?,
                models_per_reduction,
            )?,
            literal_count,
        )?;

        Ok(Self {
            output_column_count,
            output_row_capacity,
            tuple_bytes_capacity,
            output_row_count_device_reads: 1,
            model_membership_bytes_checked,
            world_view_slots_checked: candidate_count,
            row_filter_count: 0,
            negated_row_filter_count: 0,
            final_row_count_device_writes: 1,
            kernel_launches: output_column_count.max(1) as u32,
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

    /// Attach final-row filter metadata captured before launching the row-map kernel.
    pub fn with_row_filter_counts(
        mut self,
        row_filter_count: usize,
        negated_row_filter_count: usize,
    ) -> Result<Self> {
        if negated_row_filter_count > row_filter_count {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final-tuple negated row filters".to_string(),
                estimated_bytes: negated_row_filter_count as u64,
                budget_bytes: row_filter_count as u64,
            });
        }
        self.row_filter_count = row_filter_count;
        self.negated_row_filter_count = negated_row_filter_count;
        Ok(self)
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

impl EpistemicGpuFinalResultTransferTrace {
    /// Account for the final device-resident output after the hot-path budget window closes.
    pub fn from_final_output(
        provider: &xlog_cuda::CudaKernelProvider,
        final_output: &CudaBuffer,
    ) -> Result<Self> {
        let row_count_was_cached = final_output.cached_row_count().is_some();
        let final_output_rows = provider.device_row_count(final_output)?;
        let final_output_column_count = final_output.arity();
        let final_output_row_width_bytes = final_output.schema().row_size_bytes();
        let final_output_payload_bytes =
            checked_product(final_output_rows, final_output_row_width_bytes)? as u64;

        Ok(Self {
            final_output_rows,
            final_output_column_count,
            final_output_row_width_bytes,
            final_output_payload_bytes,
            row_count_device_reads: u32::from(!row_count_was_cached),
            tracked_data_plane_dtoh_calls: 0,
            tracked_data_plane_dtoh_bytes: 0,
        })
    }
}

impl EpistemicGpuSemanticTrace {
    /// Decode nonzero device rejection codes into typed GPU semantic reasons.
    pub fn typed_rejection_reasons(&self) -> Result<Vec<EpistemicGpuRejectionReason>> {
        self.rejection_reasons
            .iter()
            .copied()
            .map(EpistemicGpuRejectionReason::from_code)
            .collect()
    }

    /// Summarize accepted/rejected candidates from the device rejection buffer.
    pub fn from_device_rejection_reasons(
        provider: &xlog_cuda::CudaKernelProvider,
        workspace: &EpistemicGpuWorkspace,
        candidate_generation: &EpistemicGpuCandidateGenerationTrace,
        propagation: &EpistemicGpuPropagationTrace,
        model_membership: &EpistemicGpuModelMembershipTrace,
        world_view_validation: &EpistemicGpuWorldViewValidationTrace,
    ) -> Result<Self> {
        let candidate_count = candidate_generation.generated_candidates;
        require_positive(candidate_count, "epistemic GPU semantic-trace candidates")?;
        if candidate_count > workspace.layout.rejection_reason_slots {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU semantic-trace rejection metadata".to_string(),
                estimated_bytes: candidate_count as u64,
                budget_bytes: workspace.layout.rejection_reason_slots as u64,
            });
        }

        let raw_rejection_reasons = provider
            .dtoh_small_metadata_untracked(&workspace.rejection_reasons, candidate_count)?;
        let mut accepted_candidate_indices = Vec::new();
        let mut rejected_candidate_indices = Vec::new();
        let mut rejection_reasons = Vec::new();
        for (candidate_index, reason) in raw_rejection_reasons.into_iter().enumerate() {
            if reason == 0 {
                accepted_candidate_indices.push(candidate_index);
            } else {
                rejected_candidate_indices.push(candidate_index);
                rejection_reasons.push(reason);
            }
        }
        let accepted_candidates = accepted_candidate_indices.len();
        let rejected_candidates = rejection_reasons.len();
        let reduced_model_slots_checked = checked_product(
            checked_product(
                world_view_validation.candidates_checked,
                model_membership.reduction_count,
            )?,
            model_membership.models_per_reduction,
        )?;
        let rejection_reason_metadata_bytes =
            checked_product(candidate_count, std::mem::size_of::<u32>())? as u64;

        Ok(Self {
            generated_candidates: candidate_count,
            guesses: candidate_count,
            propagated_candidates: propagation.propagated_candidates,
            pruned_candidates: candidate_count.saturating_sub(propagation.propagated_candidates),
            tested_candidates: world_view_validation.candidates_checked,
            reduced_model_slots_checked,
            accepted_candidates,
            accepted_candidate_indices,
            accepted_world_views: accepted_candidates,
            rejected_candidates,
            rejected_candidate_indices,
            rejection_reasons,
            rejection_reason_device_reads: 1,
            rejection_reason_metadata_bytes,
            cpu_candidate_enumerations: 0,
            cpu_world_view_validations: 0,
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
            tuple_source_row_count_device_reads: 0,
            tuple_source_key_column_device_reads: 0,
            rejection_reason_slots_checked: candidate_count,
            membership_source: EpistemicGpuModelMembershipSource::ReducedOutputRowCountOnly,
            kernel_launches: 1,
            host_write_ops: 0,
            kernel_timing: EpistemicGpuKernelTimingTrace::unrecorded(),
        })
    }

    /// Build a model-membership trace backed by reduced stable-model tuple sources.
    pub fn for_stable_model_tuple_sources(
        literal_count: usize,
        candidate_count: usize,
        reduction_count: usize,
        models_per_reduction: usize,
        tuple_source_count: usize,
    ) -> Result<Self> {
        Self::for_stable_model_tuple_sources_with_key_columns(
            literal_count,
            candidate_count,
            reduction_count,
            models_per_reduction,
            tuple_source_count,
            0,
        )
    }

    /// Build a model-membership trace backed by tuple sources and key columns.
    pub fn for_stable_model_tuple_sources_with_key_columns(
        literal_count: usize,
        candidate_count: usize,
        reduction_count: usize,
        models_per_reduction: usize,
        tuple_source_count: usize,
        tuple_source_key_column_count: usize,
    ) -> Result<Self> {
        require_positive(literal_count, "epistemic GPU model-membership literals")?;
        require_positive(candidate_count, "epistemic GPU model-membership candidates")?;
        require_positive(reduction_count, "epistemic GPU model-membership reductions")?;
        require_positive(
            models_per_reduction,
            "epistemic GPU model-membership models",
        )?;
        require_positive(
            tuple_source_count,
            "epistemic GPU model-membership tuple sources",
        )?;
        if tuple_source_count > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU model-membership tuple sources".to_string(),
                estimated_bytes: tuple_source_count as u64,
                budget_bytes: u32::MAX as u64,
            });
        }
        if tuple_source_key_column_count > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU model-membership tuple key columns".to_string(),
                estimated_bytes: tuple_source_key_column_count as u64,
                budget_bytes: u32::MAX as u64,
            });
        }
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
            output_row_count_device_reads: 0,
            tuple_source_row_count_device_reads: tuple_source_count as u32,
            tuple_source_key_column_device_reads: tuple_source_key_column_count as u32,
            rejection_reason_slots_checked: candidate_count,
            membership_source: EpistemicGpuModelMembershipSource::StableModelTupleBuffer,
            kernel_launches: tuple_source_count as u32,
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

    /// Require semantic stable-model tuple membership before accepting execution.
    pub fn require_stable_model_tuple_source(&self) -> Result<()> {
        if self.membership_source != EpistemicGpuModelMembershipSource::StableModelTupleBuffer {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU stable-model membership certification".to_string(),
                context: format!(
                    "model-membership source {:?} is bounded staging only; actual reduced \
                     stable-model tuple membership is required before returning accepted \
                     epistemic execution",
                    self.membership_source
                ),
            });
        }

        Ok(())
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
    /// Selected epistemic semantics mode for the accepted GPU execution.
    pub epistemic_mode: EirEpistemicMode,
    /// GPU workspace layout required by the executable plan.
    pub workspace_layout: EpistemicGpuWorkspaceLayout,
    /// Compiled reduced-runtime rule count.
    pub reduced_runtime_rule_count: usize,
    /// Number of reduced rules carrying a `MultiWayJoin` route.
    pub multiway_reduction_count: usize,
    /// Number of K-clique WCOJ plans reused from the production planner.
    pub kclique_wcoj_plan_count: usize,
    /// Maximum K-clique arity observed across production WCOJ plans.
    pub kclique_wcoj_max_arity: u8,
    /// Live edge-permutation slots carried by production K-clique plans.
    pub kclique_wcoj_edge_permutation_count: usize,
    /// Distinct K-clique stream groups carried by production WCOJ plans.
    pub kclique_stream_group_count: usize,
    /// K-clique WCOJ plans carrying helper-split skew scheduling metadata.
    pub kclique_skew_scheduled_plan_count: usize,
    /// Number of structured planned-hash routes.
    pub planned_hash_route_count: usize,
    /// Sorted-layout edge-slot requirements carried by WCOJ plans.
    pub sorted_layout_requirement_count: usize,
    /// Helper-splitting specs carried by WCOJ plans.
    pub helper_split_spec_count: usize,
    /// Compiler-created helper-split relation rules in the reduced runtime plan.
    pub helper_relation_rule_count: usize,
    /// WCOJ input scans of compiler-created helper-split relations.
    pub helper_relation_scan_count: usize,
    /// Tuple-membership bindings certified for stable-model membership checks.
    pub tuple_membership_binding_count: usize,
    /// Non-negated `know` operators represented by the executable GPU plan.
    pub know_operator_count: usize,
    /// Non-negated `possible` operators represented by the executable GPU plan.
    pub possible_operator_count: usize,
    /// Negated `know` operators represented as `not know`.
    pub not_know_operator_count: usize,
    /// Negated `possible` operators represented as `not possible`.
    pub not_possible_operator_count: usize,
    /// Forbidden CPU fallback counters copied from the GPU semantic contract.
    pub cpu_fallbacks: EpistemicCpuFallbackCounters,
}

impl EpistemicGpuRuntimePreflight {
    /// Whether this accepted execution used G91 compatibility semantics.
    pub fn is_g91_mode(&self) -> bool {
        matches!(self.epistemic_mode, EirEpistemicMode::G91)
    }

    /// Whether this accepted execution used default FAEEL semantics.
    pub fn is_faeel_mode(&self) -> bool {
        matches!(self.epistemic_mode, EirEpistemicMode::Faeel)
    }

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
        executable.gpu_plan.validate_tuple_membership_bindings()?;

        let workspace_layout =
            EpistemicGpuWorkspaceLayout::for_plan(&executable.gpu_plan, capacities)?;
        let mut routes = RuntimeRouteSummary::default();
        let mut reduced_runtime_rule_count = 0usize;
        let helper_relation_ids = helper_relation_ids(executable);
        let mut helper_relation_rule_count = 0usize;
        let mut helper_relation_scan_count = 0usize;

        for rule in executable
            .reduced_runtime_plan
            .rules_by_scc
            .iter()
            .flatten()
        {
            reduced_runtime_rule_count += 1;
            if rule.head.starts_with("__w37_helper_") {
                helper_relation_rule_count += 1;
            }
            helper_relation_scan_count +=
                count_helper_relation_scans(&rule.body, &helper_relation_ids);
            summarize_runtime_routes(&rule.body, &mut routes);
        }

        if routes.helper_split_spec_count > 0
            && (helper_relation_rule_count < routes.helper_split_spec_count
                || helper_relation_scan_count < routes.helper_split_spec_count)
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU helper-split certification".to_string(),
                context: format!(
                    "helper_split_specs={}, helper_relation_rules={}, \
                     helper_relation_scans={}",
                    routes.helper_split_spec_count,
                    helper_relation_rule_count,
                    helper_relation_scan_count
                ),
            });
        }

        let mut know_operator_count = 0usize;
        let mut possible_operator_count = 0usize;
        let mut not_know_operator_count = 0usize;
        let mut not_possible_operator_count = 0usize;
        for literal in &executable.gpu_plan.epistemic_literals {
            match (literal.op, literal.negated) {
                (EirEpistemicOp::Know, false) => know_operator_count += 1,
                (EirEpistemicOp::Possible, false) => possible_operator_count += 1,
                (EirEpistemicOp::Know, true) => not_know_operator_count += 1,
                (EirEpistemicOp::Possible, true) => not_possible_operator_count += 1,
            }
        }

        Ok(Self {
            epistemic_mode: executable.gpu_plan.mode,
            workspace_layout,
            reduced_runtime_rule_count,
            multiway_reduction_count: routes.multiway_reduction_count,
            kclique_wcoj_plan_count: routes.kclique_wcoj_plan_count,
            kclique_wcoj_max_arity: routes.kclique_wcoj_max_arity,
            kclique_wcoj_edge_permutation_count: routes.kclique_wcoj_edge_permutation_count,
            kclique_stream_group_count: routes.kclique_stream_groups.len(),
            kclique_skew_scheduled_plan_count: routes.kclique_skew_scheduled_plan_count,
            planned_hash_route_count: routes.planned_hash_route_count,
            sorted_layout_requirement_count: routes.sorted_layout_requirement_count,
            helper_split_spec_count: routes.helper_split_spec_count,
            helper_relation_rule_count,
            helper_relation_scan_count,
            tuple_membership_binding_count: executable.gpu_plan.tuple_membership_bindings.len(),
            know_operator_count,
            possible_operator_count,
            not_know_operator_count,
            not_possible_operator_count,
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
            EpistemicGpuRuntimeWcojCertification::MissingRequiredWcojLayout {
                required_sorted_layouts,
                observed_layout_events,
            } => Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU WCOJ layout certification".to_string(),
                context: format!(
                    "required_sorted_layouts={required_sorted_layouts}, \
                     observed_layout_events={observed_layout_events}"
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
    /// Provider-observed nanoseconds spent building K-clique metadata.
    pub kclique_metadata_build_nanos: u64,
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
            kclique_metadata_build_nanos: self
                .kclique_metadata_build_nanos
                .saturating_sub(before.kclique_metadata_build_nanos),
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
        /// Edge-permutation slots certified by the dispatched K-clique plans.
        certified_edge_permutation_slots: usize,
        /// Distinct stream groups certified by the dispatched K-clique plans.
        certified_stream_groups: usize,
        /// Helper-split skew-scheduled K-clique plans certified by dispatch.
        certified_skew_scheduled_plans: usize,
        /// Sorted-layout requirements certified by the dispatched K-clique plans.
        certified_sorted_layout_requirements: usize,
        /// Helper-split specs certified by the dispatched K-clique plans.
        certified_helper_split_specs: usize,
        /// Helper relation rules proving production helper-split rewrite happened.
        certified_helper_relation_rules: usize,
        /// Helper relation scans proving WCOJ consumed production helper output.
        certified_helper_relation_scans: usize,
        /// Observed provider WCOJ layout-sort invocations.
        observed_layout_sorts: u64,
        /// Observed provider WCOJ layout fast-path hits.
        observed_layout_fast_path_hits: u64,
        /// Observed provider K-clique metadata builds.
        observed_metadata_builds: u64,
        /// Observed provider time spent building K-clique metadata.
        observed_metadata_build_nanos: u64,
    },
    /// The plan required sorted layouts, but no layout path executed.
    MissingRequiredWcojLayout {
        /// Sorted-layout requirements found during preflight.
        required_sorted_layouts: usize,
        /// Observed layout sort or fast-path events.
        observed_layout_events: u64,
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
    /// Final query tuple materialization trace captured after final-result gating.
    pub final_tuple_materialization: EpistemicGpuFinalTupleMaterializationTrace,
    /// Hot-path host-transfer budget trace for epistemic GPU execution.
    pub transfer_budget: EpistemicGpuTransferBudgetTrace,
    /// Final-result transfer accounting after the GPU hot path.
    pub final_result_transfer: EpistemicGpuFinalResultTransferTrace,
    /// Device-derived semantic summary after world-view validation.
    pub semantic_trace: EpistemicGpuSemanticTrace,
    /// Device-resident final query output buffer.
    pub final_output: CudaBuffer,
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

        let observed_layout_events =
            delta.wcoj_layout_sort_invocation_count + delta.wcoj_layout_fast_path_hit_count;
        if preflight.sorted_layout_requirement_count > 0 && observed_layout_events == 0 {
            return Self::MissingRequiredWcojLayout {
                required_sorted_layouts: preflight.sorted_layout_requirement_count,
                observed_layout_events,
            };
        }

        Self::Certified {
            observed_wcoj_dispatches,
            observed_kclique_dispatches,
            certified_edge_permutation_slots: preflight.kclique_wcoj_edge_permutation_count,
            certified_stream_groups: preflight.kclique_stream_group_count,
            certified_skew_scheduled_plans: preflight.kclique_skew_scheduled_plan_count,
            certified_sorted_layout_requirements: preflight.sorted_layout_requirement_count,
            certified_helper_split_specs: preflight.helper_split_spec_count,
            certified_helper_relation_rules: preflight.helper_relation_rule_count,
            certified_helper_relation_scans: preflight.helper_relation_scan_count,
            observed_layout_sorts: delta.wcoj_layout_sort_invocation_count,
            observed_layout_fast_path_hits: delta.wcoj_layout_fast_path_hit_count,
            observed_metadata_builds: delta.kclique_metadata_build_count,
            observed_metadata_build_nanos: delta.kclique_metadata_build_nanos,
        }
    }
}

enum TupleSourceLaunch<'a> {
    ArityZero {
        literal_index: u32,
        reduction_index: u32,
        negated: u8,
        row_count: &'a TrackedCudaSlice<u32>,
    },
    ArityOne {
        literal_index: u32,
        reduction_index: u32,
        negated: u8,
        row_count: &'a TrackedCudaSlice<u32>,
        key_col0: &'a CudaColumn,
        key_col0_width: u32,
        expected_key_col0_bits: u64,
        expected_key_col0_type_code: u8,
    },
    ArityTwo {
        literal_index: u32,
        reduction_index: u32,
        negated: u8,
        row_count: &'a TrackedCudaSlice<u32>,
        key_col0: &'a CudaColumn,
        key_col0_width: u32,
        expected_key_col0_bits: u64,
        expected_key_col0_type_code: u8,
        key_col1: &'a CudaColumn,
        key_col1_width: u32,
        expected_key_col1_bits: u64,
        expected_key_col1_type_code: u8,
    },
    ArityThree {
        literal_index: u32,
        reduction_index: u32,
        negated: u8,
        row_count: &'a TrackedCudaSlice<u32>,
        key_col0: &'a CudaColumn,
        key_col0_width: u32,
        expected_key_col0_bits: u64,
        expected_key_col0_type_code: u8,
        key_col1: &'a CudaColumn,
        key_col1_width: u32,
        expected_key_col1_bits: u64,
        expected_key_col1_type_code: u8,
        key_col2: &'a CudaColumn,
        key_col2_width: u32,
        expected_key_col2_bits: u64,
        expected_key_col2_type_code: u8,
    },
    ArityN {
        literal_index: u32,
        reduction_index: u32,
        negated: u8,
        row_count: &'a TrackedCudaSlice<u32>,
        bound_value_row_count: &'a TrackedCudaSlice<u32>,
        key_col_count: u32,
        key_col_ptrs: TrackedCudaSlice<u64>,
        key_col_widths: TrackedCudaSlice<u32>,
        expected_key_bits: TrackedCudaSlice<u64>,
        expected_key_type_codes: TrackedCudaSlice<u8>,
        tuple_key_match_modes: TrackedCudaSlice<u8>,
        bound_value_col_ptrs: TrackedCudaSlice<u64>,
        bound_value_col_widths: TrackedCudaSlice<u32>,
        has_bound_value_keys: u8,
    },
}

const TUPLE_KEY_MATCH_MODE_GROUND: u8 = 0;
const TUPLE_KEY_MATCH_MODE_BOUND_OUTPUT: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TupleKeyExpectation {
    bits: u64,
    type_code: u8,
}

impl TupleKeyExpectation {
    fn from_term(term: &EirTerm, column_type: ScalarType) -> Result<Self> {
        let bits = match (term, column_type) {
            (EirTerm::Integer(value), ScalarType::U32) => {
                u32::try_from(*value).map(u64::from).map_err(|_| {
                    tuple_key_expectation_error(format!(
                        "integer {value} is out of range for U32 tuple-key column"
                    ))
                })?
            }
            (EirTerm::Integer(value), ScalarType::I32) => i32::try_from(*value)
                .map(|v| v as u32 as u64)
                .map_err(|_| {
                    tuple_key_expectation_error(format!(
                        "integer {value} is out of range for I32 tuple-key column"
                    ))
                })?,
            (EirTerm::Integer(value), ScalarType::U64) => u64::try_from(*value).map_err(|_| {
                tuple_key_expectation_error(format!(
                    "integer {value} is out of range for U64 tuple-key column"
                ))
            })?,
            (EirTerm::Integer(value), ScalarType::I64) => *value as u64,
            (EirTerm::Integer(value), ScalarType::Bool) => match *value {
                0 => 0,
                1 => 1,
                _ => {
                    return Err(tuple_key_expectation_error(format!(
                        "integer {value} is out of range for Bool tuple-key column"
                    )))
                }
            },
            (EirTerm::Symbol(value), ScalarType::Symbol) => u64::from(*value),
            (EirTerm::String(value), ScalarType::Symbol) => {
                u64::from(xlog_core::symbol::intern(value))
            }
            (EirTerm::FloatBits(bits), ScalarType::F64) => *bits,
            (EirTerm::FloatBits(bits), ScalarType::F32) => {
                (f64::from_bits(*bits) as f32).to_bits() as u64
            }
            (EirTerm::Variable(_), _) => {
                return Err(tuple_key_expectation_error(format!(
                    "term {term:?} cannot be encoded as a ground tuple-key expectation"
                )))
            }
            (EirTerm::Anonymous | EirTerm::Aggregate { .. }, _) => {
                return Err(tuple_key_expectation_error(format!(
                    "term {term:?} cannot be used for GPU tuple-key matching"
                )))
            }
            _ => {
                return Err(tuple_key_expectation_error(format!(
                    "term {term:?} cannot be encoded for {column_type:?} tuple-key column"
                )))
            }
        };

        Ok(Self {
            bits,
            type_code: column_type.to_code(),
        })
    }
}

fn tuple_key_expectation_error(context: String) -> XlogError {
    XlogError::UnsupportedEpistemicConstruct {
        construct: "epistemic GPU tuple-key expectation".to_string(),
        context,
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
            kclique_metadata_build_nanos: self.provider.kclique_metadata_build_nanos(),
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

    /// Populate model-membership bytes from reduced stable-model tuple sources.
    pub fn populate_epistemic_gpu_model_membership_from_tuple_sources(
        &self,
        workspace: &mut EpistemicGpuWorkspace,
        output: &CudaBuffer,
        gpu_plan: &EpistemicGpuPlan,
        candidate_count: usize,
        models_per_reduction: usize,
    ) -> Result<EpistemicGpuModelMembershipTrace> {
        gpu_plan.validate_tuple_membership_bindings()?;

        let literal_count = gpu_plan.epistemic_literals.len();
        let reduction_count = gpu_plan.reductions.len();
        let tuple_source_key_column_count = gpu_plan
            .tuple_membership_bindings
            .iter()
            .try_fold(0usize, |acc, binding| {
                checked_sum(acc, binding.key_columns.len())
            })?;
        let bound_tuple_source_count = gpu_plan
            .tuple_membership_bindings
            .iter()
            .filter(|binding| {
                binding
                    .key_terms
                    .iter()
                    .any(|term| matches!(term, EirTerm::Variable(_)))
            })
            .count();
        let mut trace =
            EpistemicGpuModelMembershipTrace::for_stable_model_tuple_sources_with_key_columns(
                literal_count,
                candidate_count,
                reduction_count,
                models_per_reduction,
                gpu_plan.tuple_membership_bindings.len(),
                tuple_source_key_column_count,
            )?;
        if bound_tuple_source_count > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU bound tuple-key source count".to_string(),
                estimated_bytes: bound_tuple_source_count as u64,
                budget_bytes: u32::MAX as u64,
            });
        }
        trace.output_row_count_device_reads = bound_tuple_source_count as u32;
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

        let per_binding_launch_elems = checked_product(candidate_count, models_per_reduction)?;
        if per_binding_launch_elems > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU model-membership tuple-source launch".to_string(),
                estimated_bytes: per_binding_launch_elems as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let mut tuple_sources = Vec::with_capacity(gpu_plan.tuple_membership_bindings.len());
        for binding in &gpu_plan.tuple_membership_bindings {
            let source_relation =
                self.store()
                    .get(binding.predicate.as_str())
                    .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                        construct: "epistemic GPU stable-model tuple membership".to_string(),
                        context: format!(
                            "missing reduced stable-model tuple source relation {}",
                            binding.predicate
                        ),
                    })?;
            if source_relation.arity() != binding.arity {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU stable-model tuple membership".to_string(),
                    context: format!(
                        "tuple source relation {} arity {} does not match binding arity {}",
                        binding.predicate,
                        source_relation.arity(),
                        binding.arity
                    ),
                });
            }
            let has_bound_value_keys = binding
                .key_terms
                .iter()
                .any(|term| matches!(term, EirTerm::Variable(_)));
            match binding.key_columns.as_slice() {
                [] => tuple_sources.push(TupleSourceLaunch::ArityZero {
                    literal_index: binding.literal_index as u32,
                    reduction_index: binding.reduction_index as u32,
                    negated: binding.negated as u8,
                    row_count: source_relation.num_rows_device(),
                }),
                &[key_col] if !has_bound_value_keys => {
                    let key_col0 = source_relation.column(key_col).ok_or_else(|| {
                        XlogError::UnsupportedEpistemicConstruct {
                            construct: "epistemic GPU stable-model tuple membership".to_string(),
                            context: format!(
                                "tuple source relation {} missing key column {}",
                                binding.predicate, key_col
                            ),
                        }
                    })?;
                    let key_col0_type =
                        source_relation
                            .schema()
                            .column_type(key_col)
                            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                                construct: "epistemic GPU stable-model tuple membership"
                                    .to_string(),
                                context: format!(
                                    "tuple source relation {} missing schema for key column {}",
                                    binding.predicate, key_col
                                ),
                            })?;
                    let key_col0_width = key_col0_type.size_bytes();
                    let key_col0_expectation =
                        TupleKeyExpectation::from_term(&binding.key_terms[0], key_col0_type)?;
                    if key_col0_width > u32::MAX as usize {
                        return Err(XlogError::ResourceExhausted {
                            context: "epistemic GPU tuple-key column width".to_string(),
                            estimated_bytes: key_col0_width as u64,
                            budget_bytes: u32::MAX as u64,
                        });
                    }
                    tuple_sources.push(TupleSourceLaunch::ArityOne {
                        literal_index: binding.literal_index as u32,
                        reduction_index: binding.reduction_index as u32,
                        negated: binding.negated as u8,
                        row_count: source_relation.num_rows_device(),
                        key_col0,
                        key_col0_width: key_col0_width as u32,
                        expected_key_col0_bits: key_col0_expectation.bits,
                        expected_key_col0_type_code: key_col0_expectation.type_code,
                    });
                }
                &[key_col0, key_col1] if !has_bound_value_keys => {
                    let key_col0_ref = source_relation.column(key_col0).ok_or_else(|| {
                        XlogError::UnsupportedEpistemicConstruct {
                            construct: "epistemic GPU stable-model tuple membership".to_string(),
                            context: format!(
                                "tuple source relation {} missing key column {}",
                                binding.predicate, key_col0
                            ),
                        }
                    })?;
                    let key_col1_ref = source_relation.column(key_col1).ok_or_else(|| {
                        XlogError::UnsupportedEpistemicConstruct {
                            construct: "epistemic GPU stable-model tuple membership".to_string(),
                            context: format!(
                                "tuple source relation {} missing key column {}",
                                binding.predicate, key_col1
                            ),
                        }
                    })?;
                    let key_col0_type =
                        source_relation
                            .schema()
                            .column_type(key_col0)
                            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                                construct: "epistemic GPU stable-model tuple membership"
                                    .to_string(),
                                context: format!(
                                    "tuple source relation {} missing schema for key column {}",
                                    binding.predicate, key_col0
                                ),
                            })?;
                    let key_col1_type =
                        source_relation
                            .schema()
                            .column_type(key_col1)
                            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                                construct: "epistemic GPU stable-model tuple membership"
                                    .to_string(),
                                context: format!(
                                    "tuple source relation {} missing schema for key column {}",
                                    binding.predicate, key_col1
                                ),
                            })?;
                    let key_col0_width = key_col0_type.size_bytes();
                    let key_col1_width = key_col1_type.size_bytes();
                    let key_col0_expectation =
                        TupleKeyExpectation::from_term(&binding.key_terms[0], key_col0_type)?;
                    let key_col1_expectation =
                        TupleKeyExpectation::from_term(&binding.key_terms[1], key_col1_type)?;
                    let max_width = key_col0_width.max(key_col1_width);
                    if max_width > u32::MAX as usize {
                        return Err(XlogError::ResourceExhausted {
                            context: "epistemic GPU tuple-key column width".to_string(),
                            estimated_bytes: max_width as u64,
                            budget_bytes: u32::MAX as u64,
                        });
                    }
                    tuple_sources.push(TupleSourceLaunch::ArityTwo {
                        literal_index: binding.literal_index as u32,
                        reduction_index: binding.reduction_index as u32,
                        negated: binding.negated as u8,
                        row_count: source_relation.num_rows_device(),
                        key_col0: key_col0_ref,
                        key_col0_width: key_col0_width as u32,
                        expected_key_col0_bits: key_col0_expectation.bits,
                        expected_key_col0_type_code: key_col0_expectation.type_code,
                        key_col1: key_col1_ref,
                        key_col1_width: key_col1_width as u32,
                        expected_key_col1_bits: key_col1_expectation.bits,
                        expected_key_col1_type_code: key_col1_expectation.type_code,
                    });
                }
                &[key_col0, key_col1, key_col2] if !has_bound_value_keys => {
                    let key_col0_ref = source_relation.column(key_col0).ok_or_else(|| {
                        XlogError::UnsupportedEpistemicConstruct {
                            construct: "epistemic GPU stable-model tuple membership".to_string(),
                            context: format!(
                                "tuple source relation {} missing key column {}",
                                binding.predicate, key_col0
                            ),
                        }
                    })?;
                    let key_col1_ref = source_relation.column(key_col1).ok_or_else(|| {
                        XlogError::UnsupportedEpistemicConstruct {
                            construct: "epistemic GPU stable-model tuple membership".to_string(),
                            context: format!(
                                "tuple source relation {} missing key column {}",
                                binding.predicate, key_col1
                            ),
                        }
                    })?;
                    let key_col2_ref = source_relation.column(key_col2).ok_or_else(|| {
                        XlogError::UnsupportedEpistemicConstruct {
                            construct: "epistemic GPU stable-model tuple membership".to_string(),
                            context: format!(
                                "tuple source relation {} missing key column {}",
                                binding.predicate, key_col2
                            ),
                        }
                    })?;
                    let key_col0_type =
                        source_relation
                            .schema()
                            .column_type(key_col0)
                            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                                construct: "epistemic GPU stable-model tuple membership"
                                    .to_string(),
                                context: format!(
                                    "tuple source relation {} missing schema for key column {}",
                                    binding.predicate, key_col0
                                ),
                            })?;
                    let key_col1_type =
                        source_relation
                            .schema()
                            .column_type(key_col1)
                            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                                construct: "epistemic GPU stable-model tuple membership"
                                    .to_string(),
                                context: format!(
                                    "tuple source relation {} missing schema for key column {}",
                                    binding.predicate, key_col1
                                ),
                            })?;
                    let key_col2_type =
                        source_relation
                            .schema()
                            .column_type(key_col2)
                            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                                construct: "epistemic GPU stable-model tuple membership"
                                    .to_string(),
                                context: format!(
                                    "tuple source relation {} missing schema for key column {}",
                                    binding.predicate, key_col2
                                ),
                            })?;
                    let key_col0_width = key_col0_type.size_bytes();
                    let key_col1_width = key_col1_type.size_bytes();
                    let key_col2_width = key_col2_type.size_bytes();
                    let key_col0_expectation =
                        TupleKeyExpectation::from_term(&binding.key_terms[0], key_col0_type)?;
                    let key_col1_expectation =
                        TupleKeyExpectation::from_term(&binding.key_terms[1], key_col1_type)?;
                    let key_col2_expectation =
                        TupleKeyExpectation::from_term(&binding.key_terms[2], key_col2_type)?;
                    let max_width = key_col0_width.max(key_col1_width).max(key_col2_width);
                    if max_width > u32::MAX as usize {
                        return Err(XlogError::ResourceExhausted {
                            context: "epistemic GPU tuple-key column width".to_string(),
                            estimated_bytes: max_width as u64,
                            budget_bytes: u32::MAX as u64,
                        });
                    }
                    tuple_sources.push(TupleSourceLaunch::ArityThree {
                        literal_index: binding.literal_index as u32,
                        reduction_index: binding.reduction_index as u32,
                        negated: binding.negated as u8,
                        row_count: source_relation.num_rows_device(),
                        key_col0: key_col0_ref,
                        key_col0_width: key_col0_width as u32,
                        expected_key_col0_bits: key_col0_expectation.bits,
                        expected_key_col0_type_code: key_col0_expectation.type_code,
                        key_col1: key_col1_ref,
                        key_col1_width: key_col1_width as u32,
                        expected_key_col1_bits: key_col1_expectation.bits,
                        expected_key_col1_type_code: key_col1_expectation.type_code,
                        key_col2: key_col2_ref,
                        key_col2_width: key_col2_width as u32,
                        expected_key_col2_bits: key_col2_expectation.bits,
                        expected_key_col2_type_code: key_col2_expectation.type_code,
                    });
                }
                key_columns => {
                    if key_columns.len() > u32::MAX as usize {
                        return Err(XlogError::ResourceExhausted {
                            context: "epistemic GPU tuple-key arity".to_string(),
                            estimated_bytes: key_columns.len() as u64,
                            budget_bytes: u32::MAX as u64,
                        });
                    }

                    let mut key_col_ptrs_host = Vec::with_capacity(key_columns.len());
                    let mut key_col_widths_host = Vec::with_capacity(key_columns.len());
                    let mut expected_key_bits_host = Vec::with_capacity(key_columns.len());
                    let mut expected_key_type_codes_host = Vec::with_capacity(key_columns.len());
                    let mut tuple_key_match_modes_host = Vec::with_capacity(key_columns.len());
                    let mut bound_value_col_ptrs_host = Vec::with_capacity(key_columns.len());
                    let mut bound_value_col_widths_host = Vec::with_capacity(key_columns.len());
                    for (term_index, &key_col) in key_columns.iter().enumerate() {
                        let key_col_ref = source_relation.column(key_col).ok_or_else(|| {
                            XlogError::UnsupportedEpistemicConstruct {
                                construct: "epistemic GPU stable-model tuple membership"
                                    .to_string(),
                                context: format!(
                                    "tuple source relation {} missing key column {}",
                                    binding.predicate, key_col
                                ),
                            }
                        })?;
                        let key_col_type = source_relation
                            .schema()
                            .column_type(key_col)
                            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                                construct: "epistemic GPU stable-model tuple membership"
                                    .to_string(),
                                context: format!(
                                    "tuple source relation {} missing schema for key column {}",
                                    binding.predicate, key_col
                                ),
                            })?;
                        let key_col_width = key_col_type.size_bytes();
                        if key_col_width > u32::MAX as usize {
                            return Err(XlogError::ResourceExhausted {
                                context: "epistemic GPU tuple-key column width".to_string(),
                                estimated_bytes: key_col_width as u64,
                                budget_bytes: u32::MAX as u64,
                            });
                        }

                        key_col_ptrs_host.push(*key_col_ref.device_ptr() as u64);
                        key_col_widths_host.push(key_col_width as u32);
                        match &binding.key_terms[term_index] {
                            EirTerm::Variable(variable_name) => {
                                let bound_col_index = binding.bound_output_columns[term_index]
                                    .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                                        construct: "epistemic GPU bound tuple-key matching"
                                            .to_string(),
                                        context: format!(
                                            "tuple key variable {variable_name} has no reduced \
                                             output column binding"
                                        ),
                                    })?;
                                let bound_col =
                                    output.column(bound_col_index).ok_or_else(|| {
                                        XlogError::UnsupportedEpistemicConstruct {
                                            construct: "epistemic GPU bound tuple-key matching"
                                                .to_string(),
                                            context: format!(
                                                "reduced output is missing device column \
                                             {bound_col_index} for variable {variable_name}"
                                            ),
                                        }
                                    })?;
                                let bound_col_type =
                                    output.schema().column_type(bound_col_index).ok_or_else(
                                        || XlogError::UnsupportedEpistemicConstruct {
                                            construct: "epistemic GPU bound tuple-key matching"
                                                .to_string(),
                                            context: format!(
                                                "reduced output is missing schema for variable \
                                             {variable_name}"
                                            ),
                                        },
                                    )?;
                                if bound_col_type != key_col_type {
                                    return Err(XlogError::UnsupportedEpistemicConstruct {
                                        construct: "epistemic GPU bound tuple-key matching"
                                            .to_string(),
                                        context: format!(
                                            "bound variable {variable_name} has output type \
                                             {bound_col_type:?}, but tuple source {} key column \
                                             {} has type {key_col_type:?}",
                                            binding.predicate, key_col
                                        ),
                                    });
                                }
                                let bound_col_width = bound_col_type.size_bytes();
                                if bound_col_width > u32::MAX as usize {
                                    return Err(XlogError::ResourceExhausted {
                                        context: "epistemic GPU bound tuple-key column width"
                                            .to_string(),
                                        estimated_bytes: bound_col_width as u64,
                                        budget_bytes: u32::MAX as u64,
                                    });
                                }

                                expected_key_bits_host.push(0);
                                expected_key_type_codes_host.push(key_col_type.to_code());
                                tuple_key_match_modes_host.push(TUPLE_KEY_MATCH_MODE_BOUND_OUTPUT);
                                bound_value_col_ptrs_host.push(*bound_col.device_ptr() as u64);
                                bound_value_col_widths_host.push(bound_col_width as u32);
                            }
                            term => {
                                let expectation =
                                    TupleKeyExpectation::from_term(term, key_col_type)?;
                                expected_key_bits_host.push(expectation.bits);
                                expected_key_type_codes_host.push(expectation.type_code);
                                tuple_key_match_modes_host.push(TUPLE_KEY_MATCH_MODE_GROUND);
                                bound_value_col_ptrs_host.push(0);
                                bound_value_col_widths_host.push(0);
                            }
                        }
                    }

                    let memory = self.provider.memory();
                    let device = self.provider.device().inner();
                    let mut key_col_ptrs = memory.alloc::<u64>(key_columns.len())?;
                    let mut key_col_widths = memory.alloc::<u32>(key_columns.len())?;
                    let mut expected_key_bits = memory.alloc::<u64>(key_columns.len())?;
                    let mut expected_key_type_codes = memory.alloc::<u8>(key_columns.len())?;
                    let mut tuple_key_match_modes = memory.alloc::<u8>(key_columns.len())?;
                    let mut bound_value_col_ptrs = memory.alloc::<u64>(key_columns.len())?;
                    let mut bound_value_col_widths = memory.alloc::<u32>(key_columns.len())?;
                    device
                        .htod_sync_copy_into(&key_col_ptrs_host, &mut key_col_ptrs)
                        .map_err(|e| {
                            XlogError::execution_ctx(
                                "epistemic GPU tuple-key metadata",
                                "upload key column pointers",
                                &e,
                            )
                        })?;
                    device
                        .htod_sync_copy_into(&key_col_widths_host, &mut key_col_widths)
                        .map_err(|e| {
                            XlogError::execution_ctx(
                                "epistemic GPU tuple-key metadata",
                                "upload key column widths",
                                &e,
                            )
                        })?;
                    device
                        .htod_sync_copy_into(&expected_key_bits_host, &mut expected_key_bits)
                        .map_err(|e| {
                            XlogError::execution_ctx(
                                "epistemic GPU tuple-key metadata",
                                "upload expected key bits",
                                &e,
                            )
                        })?;
                    device
                        .htod_sync_copy_into(
                            &expected_key_type_codes_host,
                            &mut expected_key_type_codes,
                        )
                        .map_err(|e| {
                            XlogError::execution_ctx(
                                "epistemic GPU tuple-key metadata",
                                "upload expected key type codes",
                                &e,
                            )
                        })?;
                    device
                        .htod_sync_copy_into(
                            &tuple_key_match_modes_host,
                            &mut tuple_key_match_modes,
                        )
                        .map_err(|e| {
                            XlogError::execution_ctx(
                                "epistemic GPU tuple-key metadata",
                                "upload tuple key match modes",
                                &e,
                            )
                        })?;
                    device
                        .htod_sync_copy_into(&bound_value_col_ptrs_host, &mut bound_value_col_ptrs)
                        .map_err(|e| {
                            XlogError::execution_ctx(
                                "epistemic GPU tuple-key metadata",
                                "upload bound value column pointers",
                                &e,
                            )
                        })?;
                    device
                        .htod_sync_copy_into(
                            &bound_value_col_widths_host,
                            &mut bound_value_col_widths,
                        )
                        .map_err(|e| {
                            XlogError::execution_ctx(
                                "epistemic GPU tuple-key metadata",
                                "upload bound value column widths",
                                &e,
                            )
                        })?;

                    tuple_sources.push(TupleSourceLaunch::ArityN {
                        literal_index: binding.literal_index as u32,
                        reduction_index: binding.reduction_index as u32,
                        negated: binding.negated as u8,
                        row_count: source_relation.num_rows_device(),
                        bound_value_row_count: output.num_rows_device(),
                        key_col_count: key_columns.len() as u32,
                        key_col_ptrs,
                        key_col_widths,
                        expected_key_bits,
                        expected_key_type_codes,
                        tuple_key_match_modes,
                        bound_value_col_ptrs,
                        bound_value_col_widths,
                        has_bound_value_keys: has_bound_value_keys as u8,
                    });
                }
            }
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
                epistemic_kernels::EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution(
                    "epistemic tuple-source model-membership kernel not found".to_string(),
                )
            })?;
        let func_arity1 = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY1_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution(
                    "epistemic arity-one tuple-source model-membership kernel not found"
                        .to_string(),
                )
            })?;
        let func_arity2 = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY2_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution(
                    "epistemic arity-two tuple-source model-membership kernel not found"
                        .to_string(),
                )
            })?;
        let func_arity3 = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY3_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution(
                    "epistemic arity-three tuple-source model-membership kernel not found"
                        .to_string(),
                )
            })?;
        let func_arity_n = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY_N_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution(
                    "epistemic generic-arity tuple-source model-membership kernel not found"
                        .to_string(),
                )
            })?;
        let config = LaunchConfig::for_num_elems(per_binding_launch_elems as u32);

        let kernel_timing = self.time_epistemic_gpu_kernel_launch(
            "epistemic GPU tuple-source model membership",
            || unsafe {
                for tuple_source in &tuple_sources {
                    match tuple_source {
                        TupleSourceLaunch::ArityZero {
                            literal_index,
                            reduction_index,
                            negated,
                            row_count,
                        } => {
                            // SAFETY: kernel arguments match the PTX signature; the capacity
                            // checks above prove candidate, world-view, membership, rejection,
                            // and tuple-source row-count buffers cover all accesses.
                            let mut params: Vec<*mut c_void> = vec![
                                (&literal_count).as_kernel_param(),
                                (&candidate_count).as_kernel_param(),
                                (&reduction_count).as_kernel_param(),
                                (&models_per_reduction).as_kernel_param(),
                                (&world_stride).as_kernel_param(),
                                literal_index.as_kernel_param(),
                                reduction_index.as_kernel_param(),
                                negated.as_kernel_param(),
                                row_count.as_kernel_param(),
                                (&workspace.candidate_assumptions).as_kernel_param(),
                                (&workspace.world_views).as_kernel_param(),
                                (&mut workspace.model_membership).as_kernel_param(),
                                (&mut workspace.rejection_reasons).as_kernel_param(),
                            ];
                            func.clone().launch(config, &mut params)?;
                        }
                        TupleSourceLaunch::ArityOne {
                            literal_index,
                            reduction_index,
                            negated,
                            row_count,
                            key_col0,
                            key_col0_width,
                            expected_key_col0_bits,
                            expected_key_col0_type_code,
                        } => {
                            // SAFETY: kernel arguments match the PTX signature; capacity checks
                            // above cover workspace buffers, row_count comes from the named source
                            // relation, and key_col0/key_col0_width are schema-validated.
                            let mut params: Vec<*mut c_void> = vec![
                                (&literal_count).as_kernel_param(),
                                (&candidate_count).as_kernel_param(),
                                (&reduction_count).as_kernel_param(),
                                (&models_per_reduction).as_kernel_param(),
                                (&world_stride).as_kernel_param(),
                                literal_index.as_kernel_param(),
                                reduction_index.as_kernel_param(),
                                negated.as_kernel_param(),
                                row_count.as_kernel_param(),
                                key_col0.as_kernel_param(),
                                key_col0_width.as_kernel_param(),
                                expected_key_col0_bits.as_kernel_param(),
                                expected_key_col0_type_code.as_kernel_param(),
                                (&workspace.candidate_assumptions).as_kernel_param(),
                                (&workspace.world_views).as_kernel_param(),
                                (&mut workspace.model_membership).as_kernel_param(),
                                (&mut workspace.rejection_reasons).as_kernel_param(),
                            ];
                            func_arity1.clone().launch(config, &mut params)?;
                        }
                        TupleSourceLaunch::ArityTwo {
                            literal_index,
                            reduction_index,
                            negated,
                            row_count,
                            key_col0,
                            key_col0_width,
                            expected_key_col0_bits,
                            expected_key_col0_type_code,
                            key_col1,
                            key_col1_width,
                            expected_key_col1_bits,
                            expected_key_col1_type_code,
                        } => {
                            // SAFETY: kernel arguments match the PTX signature; capacity checks
                            // above cover workspace buffers, row_count comes from the named source
                            // relation, and both key columns are schema-validated.
                            let mut params: Vec<*mut c_void> = vec![
                                (&literal_count).as_kernel_param(),
                                (&candidate_count).as_kernel_param(),
                                (&reduction_count).as_kernel_param(),
                                (&models_per_reduction).as_kernel_param(),
                                (&world_stride).as_kernel_param(),
                                literal_index.as_kernel_param(),
                                reduction_index.as_kernel_param(),
                                negated.as_kernel_param(),
                                row_count.as_kernel_param(),
                                key_col0.as_kernel_param(),
                                key_col0_width.as_kernel_param(),
                                expected_key_col0_bits.as_kernel_param(),
                                expected_key_col0_type_code.as_kernel_param(),
                                key_col1.as_kernel_param(),
                                key_col1_width.as_kernel_param(),
                                expected_key_col1_bits.as_kernel_param(),
                                expected_key_col1_type_code.as_kernel_param(),
                                (&workspace.candidate_assumptions).as_kernel_param(),
                                (&workspace.world_views).as_kernel_param(),
                                (&mut workspace.model_membership).as_kernel_param(),
                                (&mut workspace.rejection_reasons).as_kernel_param(),
                            ];
                            func_arity2.clone().launch(config, &mut params)?;
                        }
                        TupleSourceLaunch::ArityThree {
                            literal_index,
                            reduction_index,
                            negated,
                            row_count,
                            key_col0,
                            key_col0_width,
                            expected_key_col0_bits,
                            expected_key_col0_type_code,
                            key_col1,
                            key_col1_width,
                            expected_key_col1_bits,
                            expected_key_col1_type_code,
                            key_col2,
                            key_col2_width,
                            expected_key_col2_bits,
                            expected_key_col2_type_code,
                        } => {
                            // SAFETY: kernel arguments match the PTX signature; capacity checks
                            // above cover workspace buffers, row_count comes from the named source
                            // relation, and all key columns are schema-validated.
                            let mut params: Vec<*mut c_void> = vec![
                                (&literal_count).as_kernel_param(),
                                (&candidate_count).as_kernel_param(),
                                (&reduction_count).as_kernel_param(),
                                (&models_per_reduction).as_kernel_param(),
                                (&world_stride).as_kernel_param(),
                                literal_index.as_kernel_param(),
                                reduction_index.as_kernel_param(),
                                negated.as_kernel_param(),
                                row_count.as_kernel_param(),
                                key_col0.as_kernel_param(),
                                key_col0_width.as_kernel_param(),
                                expected_key_col0_bits.as_kernel_param(),
                                expected_key_col0_type_code.as_kernel_param(),
                                key_col1.as_kernel_param(),
                                key_col1_width.as_kernel_param(),
                                expected_key_col1_bits.as_kernel_param(),
                                expected_key_col1_type_code.as_kernel_param(),
                                key_col2.as_kernel_param(),
                                key_col2_width.as_kernel_param(),
                                expected_key_col2_bits.as_kernel_param(),
                                expected_key_col2_type_code.as_kernel_param(),
                                (&workspace.candidate_assumptions).as_kernel_param(),
                                (&workspace.world_views).as_kernel_param(),
                                (&mut workspace.model_membership).as_kernel_param(),
                                (&mut workspace.rejection_reasons).as_kernel_param(),
                            ];
                            func_arity3.clone().launch(config, &mut params)?;
                        }
                        TupleSourceLaunch::ArityN {
                            literal_index,
                            reduction_index,
                            negated,
                            row_count,
                            bound_value_row_count,
                            key_col_count,
                            key_col_ptrs,
                            key_col_widths,
                            expected_key_bits,
                            expected_key_type_codes,
                            tuple_key_match_modes,
                            bound_value_col_ptrs,
                            bound_value_col_widths,
                            has_bound_value_keys,
                        } => {
                            // SAFETY: kernel arguments match the PTX signature; capacity checks
                            // above cover workspace buffers, row_count comes from the named source
                            // relation, and pointer/width/expectation arrays are device-resident
                            // launch metadata for existing relation and reduced-output columns.
                            let mut params: Vec<*mut c_void> = vec![
                                (&literal_count).as_kernel_param(),
                                (&candidate_count).as_kernel_param(),
                                (&reduction_count).as_kernel_param(),
                                (&models_per_reduction).as_kernel_param(),
                                (&world_stride).as_kernel_param(),
                                literal_index.as_kernel_param(),
                                reduction_index.as_kernel_param(),
                                negated.as_kernel_param(),
                                row_count.as_kernel_param(),
                                key_col_ptrs.as_kernel_param(),
                                key_col_widths.as_kernel_param(),
                                expected_key_bits.as_kernel_param(),
                                expected_key_type_codes.as_kernel_param(),
                                tuple_key_match_modes.as_kernel_param(),
                                bound_value_col_ptrs.as_kernel_param(),
                                bound_value_col_widths.as_kernel_param(),
                                bound_value_row_count.as_kernel_param(),
                                key_col_count.as_kernel_param(),
                                has_bound_value_keys.as_kernel_param(),
                                (&workspace.candidate_assumptions).as_kernel_param(),
                                (&workspace.world_views).as_kernel_param(),
                                (&mut workspace.model_membership).as_kernel_param(),
                                (&mut workspace.rejection_reasons).as_kernel_param(),
                            ];
                            func_arity_n.clone().launch(config, &mut params)?;
                        }
                    }
                }
                Ok(())
            },
        )?;

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
                        &workspace.candidate_assumptions,
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

    /// Materialize final query tuples into a device-resident output buffer.
    pub fn materialize_epistemic_gpu_final_tuples(
        &self,
        workspace: &EpistemicGpuWorkspace,
        output: &CudaBuffer,
        gpu_plan: &EpistemicGpuPlan,
        literal_count: usize,
        candidate_count: usize,
        reduction_count: usize,
        models_per_reduction: usize,
    ) -> Result<(CudaBuffer, EpistemicGpuFinalTupleMaterializationTrace)> {
        if candidate_count > workspace.layout.rejection_reason_slots {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final-tuple rejection workspace".to_string(),
                estimated_bytes: candidate_count as u64,
                budget_bytes: workspace.layout.rejection_reason_slots as u64,
            });
        }
        if literal_count > u32::MAX as usize
            || candidate_count > u32::MAX as usize
            || reduction_count > u32::MAX as usize
            || models_per_reduction > u32::MAX as usize
            || output.num_rows() > u32::MAX as u64
        {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final-tuple dimensions".to_string(),
                estimated_bytes: literal_count
                    .max(candidate_count)
                    .max(reduction_count)
                    .max(models_per_reduction)
                    .max(output.num_rows() as usize) as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let mut tuple_bytes_capacity = 0usize;
        let mut source_columns: Vec<(&CudaColumn, u32)> = Vec::with_capacity(output.arity());
        let mut result_columns_raw: Vec<TrackedCudaSlice<u8>> = Vec::with_capacity(output.arity());
        for col_idx in 0..output.arity() {
            let src_col = output.column(col_idx).ok_or_else(|| {
                XlogError::Execution(format!("epistemic final tuple missing column {col_idx}"))
            })?;
            if src_col.len() > u32::MAX as usize {
                return Err(XlogError::ResourceExhausted {
                    context: "epistemic GPU final-tuple column".to_string(),
                    estimated_bytes: src_col.len() as u64,
                    budget_bytes: u32::MAX as u64,
                });
            }
            tuple_bytes_capacity = checked_sum(tuple_bytes_capacity, src_col.len())?;
            source_columns.push((src_col, src_col.len() as u32));
            result_columns_raw.push(self.provider.memory().alloc::<u8>(src_col.len())?);
        }

        let mut final_row_count = self.provider.memory().alloc::<u32>(1)?;
        let output_row_capacity = output.num_rows() as usize;
        let mut row_map = self
            .provider
            .memory()
            .alloc::<u32>(output_row_capacity.max(1))?;
        let row_filter_bindings: Vec<_> = gpu_plan
            .tuple_membership_bindings
            .iter()
            .filter(|binding| binding.bound_output_columns.iter().any(Option::is_some))
            .collect();
        if row_filter_bindings.len() > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final tuple row-filter count".to_string(),
                estimated_bytes: row_filter_bindings.len() as u64,
                budget_bytes: u32::MAX as u64,
            });
        }
        let negated_row_filter_count = row_filter_bindings
            .iter()
            .filter(|binding| binding.negated)
            .count();
        let trace = EpistemicGpuFinalTupleMaterializationTrace::for_counts(
            output.arity(),
            output.num_rows() as usize,
            tuple_bytes_capacity,
            literal_count,
            candidate_count,
            reduction_count,
            models_per_reduction,
        )?
        .with_row_filter_counts(row_filter_bindings.len(), negated_row_filter_count)?;
        if trace.model_membership_bytes_checked > workspace.layout.model_membership_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final-tuple membership workspace".to_string(),
                estimated_bytes: trace.model_membership_bytes_checked as u64,
                budget_bytes: workspace.layout.model_membership_bytes as u64,
            });
        }
        if trace.world_view_slots_checked > workspace.layout.world_view_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final-tuple world-view workspace".to_string(),
                estimated_bytes: trace.world_view_slots_checked as u64,
                budget_bytes: workspace.layout.world_view_bytes as u64,
            });
        }
        if trace.model_membership_bytes_checked > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final-tuple membership launch".to_string(),
                estimated_bytes: trace.model_membership_bytes_checked as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let world_stride =
            workspace.layout.world_view_bytes / workspace.layout.rejection_reason_slots;
        if world_stride == 0 || world_stride > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final-tuple world stride".to_string(),
                estimated_bytes: world_stride as u64,
                budget_bytes: u32::MAX as u64,
            });
        }
        let literal_count = literal_count as u32;
        let candidate_count_u32 = candidate_count as u32;
        let reduction_count = reduction_count as u32;
        let models_per_reduction = models_per_reduction as u32;
        let world_stride = world_stride as u32;
        let output_row_capacity_u32 = output_row_capacity as u32;
        let mut metadata_len = 0usize;
        for binding in &row_filter_bindings {
            metadata_len = checked_sum(metadata_len, binding.key_columns.len())?;
        }
        let metadata_len = metadata_len.max(1);
        let row_filter_metadata_len = row_filter_bindings.len().max(1);
        let memory = self.provider.memory();
        let device = self.provider.device().inner();
        let mut tuple_source_row_count_ptrs = memory.alloc::<u64>(row_filter_metadata_len)?;
        let mut row_filter_negated = memory.alloc::<u8>(row_filter_metadata_len)?;
        let mut row_filter_key_offsets = memory.alloc::<u32>(row_filter_metadata_len)?;
        let mut row_filter_key_counts = memory.alloc::<u32>(row_filter_metadata_len)?;
        let mut key_col_ptrs = memory.alloc::<u64>(metadata_len)?;
        let mut key_col_widths = memory.alloc::<u32>(metadata_len)?;
        let mut expected_key_bits = memory.alloc::<u64>(metadata_len)?;
        let mut expected_key_type_codes = memory.alloc::<u8>(metadata_len)?;
        let mut tuple_key_match_modes = memory.alloc::<u8>(metadata_len)?;
        let mut bound_value_col_ptrs = memory.alloc::<u64>(metadata_len)?;
        let mut bound_value_col_widths = memory.alloc::<u32>(metadata_len)?;
        let row_filter_count = row_filter_bindings.len() as u32;
        let mut tuple_source_row_counts = Vec::with_capacity(row_filter_bindings.len());

        if !row_filter_bindings.is_empty() {
            let mut tuple_source_row_count_ptrs_host =
                Vec::with_capacity(row_filter_bindings.len());
            let mut row_filter_negated_host = Vec::with_capacity(row_filter_bindings.len());
            let mut row_filter_key_offsets_host = Vec::with_capacity(row_filter_bindings.len());
            let mut row_filter_key_counts_host = Vec::with_capacity(row_filter_bindings.len());
            let mut key_col_ptrs_host = Vec::with_capacity(metadata_len);
            let mut key_col_widths_host = Vec::with_capacity(metadata_len);
            let mut expected_key_bits_host = Vec::with_capacity(metadata_len);
            let mut expected_key_type_codes_host = Vec::with_capacity(metadata_len);
            let mut tuple_key_match_modes_host = Vec::with_capacity(metadata_len);
            let mut bound_value_col_ptrs_host = Vec::with_capacity(metadata_len);
            let mut bound_value_col_widths_host = Vec::with_capacity(metadata_len);

            for binding in &row_filter_bindings {
                if binding.key_columns.len() > u32::MAX as usize {
                    return Err(XlogError::ResourceExhausted {
                        context: "epistemic GPU final tuple row-filter key arity".to_string(),
                        estimated_bytes: binding.key_columns.len() as u64,
                        budget_bytes: u32::MAX as u64,
                    });
                }
                if key_col_ptrs_host.len() > u32::MAX as usize {
                    return Err(XlogError::ResourceExhausted {
                        context: "epistemic GPU final tuple row-filter key metadata".to_string(),
                        estimated_bytes: key_col_ptrs_host.len() as u64,
                        budget_bytes: u32::MAX as u64,
                    });
                }
                row_filter_key_offsets_host.push(key_col_ptrs_host.len() as u32);
                row_filter_key_counts_host.push(binding.key_columns.len() as u32);
                row_filter_negated_host.push(binding.negated as u8);

                let source_relation =
                    self.store()
                        .get(binding.predicate.as_str())
                        .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                            construct: "epistemic GPU final tuple row filtering".to_string(),
                            context: format!(
                                "missing tuple source relation {} for final row filter",
                                binding.predicate
                            ),
                        })?;
                let tuple_source_row_count = self.clone_device_row_count(source_relation)?;
                tuple_source_row_count_ptrs_host.push(*tuple_source_row_count.device_ptr() as u64);
                tuple_source_row_counts.push(tuple_source_row_count);

                for (term_index, &key_col) in binding.key_columns.iter().enumerate() {
                    let key_col_ref = source_relation.column(key_col).ok_or_else(|| {
                        XlogError::UnsupportedEpistemicConstruct {
                            construct: "epistemic GPU final tuple row filtering".to_string(),
                            context: format!(
                                "tuple source relation {} missing key column {}",
                                binding.predicate, key_col
                            ),
                        }
                    })?;
                    let key_col_type =
                        source_relation
                            .schema()
                            .column_type(key_col)
                            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                                construct: "epistemic GPU final tuple row filtering".to_string(),
                                context: format!(
                                    "tuple source relation {} missing schema for key column {}",
                                    binding.predicate, key_col
                                ),
                            })?;
                    let key_col_width = key_col_type.size_bytes();
                    if key_col_width > u32::MAX as usize {
                        return Err(XlogError::ResourceExhausted {
                            context: "epistemic GPU final tuple row-filter key width".to_string(),
                            estimated_bytes: key_col_width as u64,
                            budget_bytes: u32::MAX as u64,
                        });
                    }

                    key_col_ptrs_host.push(*key_col_ref.device_ptr() as u64);
                    key_col_widths_host.push(key_col_width as u32);
                    match &binding.key_terms[term_index] {
                        EirTerm::Variable(variable_name) => {
                            let bound_col_index = binding.bound_output_columns[term_index]
                                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                                    construct: "epistemic GPU final tuple row filtering"
                                        .to_string(),
                                    context: format!(
                                        "tuple key variable {variable_name} has no reduced \
                                             output column binding"
                                    ),
                                })?;
                            let bound_col = output.column(bound_col_index).ok_or_else(|| {
                                XlogError::UnsupportedEpistemicConstruct {
                                    construct: "epistemic GPU final tuple row filtering"
                                        .to_string(),
                                    context: format!(
                                        "reduced output missing device column {bound_col_index} \
                                         for variable {variable_name}"
                                    ),
                                }
                            })?;
                            let bound_col_type = output
                                .schema()
                                .column_type(bound_col_index)
                                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                                    construct: "epistemic GPU final tuple row filtering"
                                        .to_string(),
                                    context: format!(
                                        "reduced output missing schema for variable \
                                             {variable_name}"
                                    ),
                                })?;
                            if bound_col_type != key_col_type {
                                return Err(XlogError::UnsupportedEpistemicConstruct {
                                    construct: "epistemic GPU final tuple row filtering"
                                        .to_string(),
                                    context: format!(
                                        "bound variable {variable_name} has output type \
                                         {bound_col_type:?}, but tuple source {} key column {} \
                                         has type {key_col_type:?}",
                                        binding.predicate, key_col
                                    ),
                                });
                            }
                            let bound_col_width = bound_col_type.size_bytes();
                            if bound_col_width > u32::MAX as usize {
                                return Err(XlogError::ResourceExhausted {
                                    context: "epistemic GPU final tuple row-filter bound width"
                                        .to_string(),
                                    estimated_bytes: bound_col_width as u64,
                                    budget_bytes: u32::MAX as u64,
                                });
                            }
                            expected_key_bits_host.push(0);
                            expected_key_type_codes_host.push(key_col_type.to_code());
                            tuple_key_match_modes_host.push(TUPLE_KEY_MATCH_MODE_BOUND_OUTPUT);
                            bound_value_col_ptrs_host.push(*bound_col.device_ptr() as u64);
                            bound_value_col_widths_host.push(bound_col_width as u32);
                        }
                        term => {
                            let expectation = TupleKeyExpectation::from_term(term, key_col_type)?;
                            expected_key_bits_host.push(expectation.bits);
                            expected_key_type_codes_host.push(expectation.type_code);
                            tuple_key_match_modes_host.push(TUPLE_KEY_MATCH_MODE_GROUND);
                            bound_value_col_ptrs_host.push(0);
                            bound_value_col_widths_host.push(0);
                        }
                    }
                }
            }

            let metadata_context = "epistemic GPU final tuple row-filter metadata";
            device
                .htod_sync_copy_into(
                    &tuple_source_row_count_ptrs_host,
                    &mut tuple_source_row_count_ptrs,
                )
                .map_err(|e| {
                    XlogError::execution_ctx(
                        metadata_context,
                        "upload tuple source row-count pointers",
                        &e,
                    )
                })?;
            device
                .htod_sync_copy_into(&row_filter_negated_host, &mut row_filter_negated)
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload row-filter polarity", &e)
                })?;
            device
                .htod_sync_copy_into(&row_filter_key_offsets_host, &mut row_filter_key_offsets)
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload row-filter key offsets", &e)
                })?;
            device
                .htod_sync_copy_into(&row_filter_key_counts_host, &mut row_filter_key_counts)
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload row-filter key counts", &e)
                })?;
            device
                .htod_sync_copy_into(&key_col_ptrs_host, &mut key_col_ptrs)
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload key column pointers", &e)
                })?;
            device
                .htod_sync_copy_into(&key_col_widths_host, &mut key_col_widths)
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload key column widths", &e)
                })?;
            device
                .htod_sync_copy_into(&expected_key_bits_host, &mut expected_key_bits)
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload expected key bits", &e)
                })?;
            device
                .htod_sync_copy_into(&expected_key_type_codes_host, &mut expected_key_type_codes)
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload expected key type codes", &e)
                })?;
            device
                .htod_sync_copy_into(&tuple_key_match_modes_host, &mut tuple_key_match_modes)
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload tuple key match modes", &e)
                })?;
            device
                .htod_sync_copy_into(&bound_value_col_ptrs_host, &mut bound_value_col_ptrs)
                .map_err(|e| {
                    XlogError::execution_ctx(
                        metadata_context,
                        "upload bound value column pointers",
                        &e,
                    )
                })?;
            device
                .htod_sync_copy_into(&bound_value_col_widths_host, &mut bound_value_col_widths)
                .map_err(|e| {
                    XlogError::execution_ctx(
                        metadata_context,
                        "upload bound value column widths",
                        &e,
                    )
                })?;
        } else {
            let metadata_context = "epistemic GPU final tuple row-filter metadata";
            device
                .memset_zeros(&mut tuple_source_row_count_ptrs)
                .map_err(|e| {
                    XlogError::execution_ctx(
                        metadata_context,
                        "tuple source row-count pointer memset",
                        &e,
                    )
                })?;
            device.memset_zeros(&mut row_filter_negated).map_err(|e| {
                XlogError::execution_ctx(metadata_context, "row-filter polarity memset", &e)
            })?;
            device
                .memset_zeros(&mut row_filter_key_offsets)
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "row-filter key offset memset", &e)
                })?;
            device
                .memset_zeros(&mut row_filter_key_counts)
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "row-filter key count memset", &e)
                })?;
            device.memset_zeros(&mut key_col_ptrs).map_err(|e| {
                XlogError::execution_ctx(metadata_context, "key column pointer memset", &e)
            })?;
            device.memset_zeros(&mut key_col_widths).map_err(|e| {
                XlogError::execution_ctx(metadata_context, "key column width memset", &e)
            })?;
            device.memset_zeros(&mut expected_key_bits).map_err(|e| {
                XlogError::execution_ctx(metadata_context, "expected key bits memset", &e)
            })?;
            device
                .memset_zeros(&mut expected_key_type_codes)
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "expected key type code memset", &e)
                })?;
            device
                .memset_zeros(&mut tuple_key_match_modes)
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "tuple key match mode memset", &e)
                })?;
            device
                .memset_zeros(&mut bound_value_col_ptrs)
                .map_err(|e| {
                    XlogError::execution_ctx(
                        metadata_context,
                        "bound value column pointer memset",
                        &e,
                    )
                })?;
            device
                .memset_zeros(&mut bound_value_col_widths)
                .map_err(|e| {
                    XlogError::execution_ctx(
                        metadata_context,
                        "bound value column width memset",
                        &e,
                    )
                })?;
        }

        let row_map_func = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_BUILD_FINAL_TUPLE_ROW_MAP_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution("epistemic final tuple row-map kernel not found".to_string())
            })?;
        let func = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_MATERIALIZE_FINAL_TUPLE_COLUMN_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution(
                    "epistemic final tuple materialization kernel not found".to_string(),
                )
            })?;

        let kernel_timing = self.time_epistemic_gpu_kernel_launch(
            "epistemic GPU final tuple materialization",
            || unsafe {
                self.provider
                    .device()
                    .inner()
                    .memset_zeros(&mut final_row_count)?;
                self.provider.device().inner().memset_zeros(&mut row_map)?;
                let mut row_map_params: Vec<*mut c_void> = vec![
                    (&output_row_capacity_u32).as_kernel_param(),
                    (&literal_count).as_kernel_param(),
                    (&candidate_count_u32).as_kernel_param(),
                    (&reduction_count).as_kernel_param(),
                    (&models_per_reduction).as_kernel_param(),
                    (&world_stride).as_kernel_param(),
                    output.num_rows_device().as_kernel_param(),
                    (&workspace.rejection_reasons).as_kernel_param(),
                    (&workspace.model_membership).as_kernel_param(),
                    (&workspace.world_views).as_kernel_param(),
                    (&tuple_source_row_count_ptrs).as_kernel_param(),
                    (&row_filter_negated).as_kernel_param(),
                    (&row_filter_key_offsets).as_kernel_param(),
                    (&row_filter_key_counts).as_kernel_param(),
                    (&key_col_ptrs).as_kernel_param(),
                    (&key_col_widths).as_kernel_param(),
                    (&expected_key_bits).as_kernel_param(),
                    (&expected_key_type_codes).as_kernel_param(),
                    (&tuple_key_match_modes).as_kernel_param(),
                    (&bound_value_col_ptrs).as_kernel_param(),
                    (&bound_value_col_widths).as_kernel_param(),
                    (&row_filter_count).as_kernel_param(),
                    (&mut row_map).as_kernel_param(),
                    (&mut final_row_count).as_kernel_param(),
                ];
                row_map_func.clone().launch(
                    LaunchConfig::for_num_elems(output_row_capacity_u32.max(1)),
                    &mut row_map_params,
                )?;

                if source_columns.is_empty() {
                    return Ok(());
                }

                for ((src_col, column_byte_len), dst_col) in
                    source_columns.iter().zip(result_columns_raw.iter_mut())
                {
                    // SAFETY: source and destination columns are valid device byte
                    // buffers of identical length, the row-count scalar is
                    // runtime-owned, and membership/world-view buffers were
                    // capacity-checked.
                    let mut params: Vec<*mut c_void> = vec![
                        column_byte_len.as_kernel_param(),
                        (&literal_count).as_kernel_param(),
                        (&candidate_count_u32).as_kernel_param(),
                        (&reduction_count).as_kernel_param(),
                        (&models_per_reduction).as_kernel_param(),
                        (&world_stride).as_kernel_param(),
                        output.num_rows_device().as_kernel_param(),
                        (&workspace.rejection_reasons).as_kernel_param(),
                        (&workspace.model_membership).as_kernel_param(),
                        (&workspace.world_views).as_kernel_param(),
                        (&row_map).as_kernel_param(),
                        (*src_col).as_kernel_param(),
                        dst_col.as_kernel_param(),
                        (&mut final_row_count).as_kernel_param(),
                    ];
                    func.clone().launch(
                        LaunchConfig::for_num_elems((*column_byte_len).max(1)),
                        &mut params,
                    )?;
                }
                Ok(())
            },
        )?;

        let result_columns: Vec<CudaColumn> =
            result_columns_raw.into_iter().map(Into::into).collect();
        let final_output = CudaBuffer::from_columns(
            result_columns,
            output.num_rows(),
            final_row_count,
            output.schema().clone(),
        );

        Ok((final_output, trace.with_kernel_timing(kernel_timing)))
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
        let _reduced_return = self.execute_plan(&executable.reduced_runtime_plan)?;
        let counters_after = self.epistemic_gpu_runtime_counters();
        let trace = EpistemicGpuRuntimeTrace::from_preflight_and_counters(
            prepared.preflight,
            counters_before,
            counters_after,
        );
        trace.require_wcoj_certification()?;
        let output_relation = executable
            .gpu_plan
            .reductions
            .last()
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU reduced output".to_string(),
                context: "executable plan has no epistemic reductions".to_string(),
            })?
            .head_predicate
            .as_str();
        let output = {
            let reduced_output = self.store().get(output_relation).ok_or_else(|| {
                XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU reduced output".to_string(),
                    context: format!(
                        "missing reduced output relation {output_relation} after production \
                         runtime dispatch"
                    ),
                }
            })?;
            self.clone_buffer(reduced_output)?
        };
        let model_membership = self.populate_epistemic_gpu_model_membership_from_tuple_sources(
            &mut prepared.workspace,
            &output,
            &executable.gpu_plan,
            candidate_count,
            capacities.max_models_per_reduction,
        )?;
        model_membership.require_stable_model_tuple_source()?;
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
        let (final_output, final_tuple_materialization) = self
            .materialize_epistemic_gpu_final_tuples(
                &prepared.workspace,
                &output,
                &executable.gpu_plan,
                literal_count,
                candidate_count,
                executable.gpu_plan.reductions.len(),
                capacities.max_models_per_reduction,
            )?;
        let transfer_budget_end = self.provider.host_transfer_stats();
        let transfer_budget = EpistemicGpuTransferBudgetTrace::from_host_transfer_stats(
            candidate_count,
            transfer_budget_start,
            transfer_budget_end,
        )?;
        let final_result_transfer =
            EpistemicGpuFinalResultTransferTrace::from_final_output(&self.provider, &final_output)?;
        let semantic_trace = EpistemicGpuSemanticTrace::from_device_rejection_reasons(
            &self.provider,
            &prepared.workspace,
            &candidate_generation,
            &propagation,
            &model_membership,
            &world_view_validation,
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
            final_tuple_materialization,
            transfer_budget,
            final_result_transfer,
            semantic_trace,
            final_output,
            output,
            trace,
        })
    }

    /// Execute multiple accepted epistemic GPU executable plans in order.
    ///
    /// This is the runtime adapter used by split execution evidence: each
    /// component is still dispatched through [`Self::execute_epistemic_gpu_execution`],
    /// so candidate generation, model-membership, world-view validation,
    /// materialization, transfer-budget, and production runtime counters are
    /// recorded by the existing single-plan path.
    pub fn execute_epistemic_gpu_execution_batch(
        &mut self,
        executables: &[&EpistemicExecutablePlan],
        capacities: EpistemicGpuWorkspaceCapacities,
    ) -> Result<Vec<EpistemicGpuExecutionResult>> {
        if executables.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU batch execution".to_string(),
                context: "batch execution requires at least one executable component".to_string(),
            });
        }

        let mut results = Vec::with_capacity(executables.len());
        for executable in executables {
            results.push(self.execute_epistemic_gpu_execution(executable, capacities)?);
        }
        Ok(results)
    }
}

#[derive(Default)]
struct RuntimeRouteSummary {
    multiway_reduction_count: usize,
    kclique_wcoj_plan_count: usize,
    kclique_wcoj_max_arity: u8,
    kclique_wcoj_edge_permutation_count: usize,
    kclique_stream_groups: BTreeSet<StreamGroupId>,
    kclique_skew_scheduled_plan_count: usize,
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
                    routes.kclique_wcoj_max_arity = routes.kclique_wcoj_max_arity.max(order.k);
                    routes.kclique_wcoj_edge_permutation_count += order
                        .edge_permutation
                        .iter()
                        .take_while(|slot| **slot != u8::MAX)
                        .count();
                    routes.kclique_stream_groups.insert(order.stream_group);
                    if !order.helper_split_specs.is_empty() {
                        routes.kclique_skew_scheduled_plan_count += 1;
                    }
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

fn helper_relation_ids(executable: &EpistemicExecutablePlan) -> BTreeSet<RelId> {
    executable
        .relation_ids
        .iter()
        .filter_map(|(name, rel)| name.starts_with("__w37_helper_").then_some(*rel))
        .collect()
}

fn count_helper_relation_scans(node: &RirNode, helper_relations: &BTreeSet<RelId>) -> usize {
    match node {
        RirNode::Scan { .. } => 0,
        RirNode::MultiWayJoin {
            plan,
            inputs,
            fallback,
            ..
        } => {
            let own_wcoj_inputs = if matches!(plan, Some(MultiwayPlan::WcojWithPlan(_))) {
                inputs
                    .iter()
                    .map(|input| count_helper_relation_leaf_scans(input, helper_relations))
                    .sum()
            } else {
                0
            };
            own_wcoj_inputs
                + inputs
                    .iter()
                    .map(|input| count_helper_relation_scans(input, helper_relations))
                    .sum::<usize>()
                + count_helper_relation_scans(fallback, helper_relations)
        }
        RirNode::Filter { input, .. }
        | RirNode::Project { input, .. }
        | RirNode::Distinct { input, .. }
        | RirNode::GroupBy { input, .. } => count_helper_relation_scans(input, helper_relations),
        RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
            count_helper_relation_scans(left, helper_relations)
                + count_helper_relation_scans(right, helper_relations)
        }
        RirNode::Union { inputs } => inputs
            .iter()
            .map(|input| count_helper_relation_scans(input, helper_relations))
            .sum(),
        RirNode::Fixpoint {
            base, recursive, ..
        } => {
            count_helper_relation_scans(base, helper_relations)
                + count_helper_relation_scans(recursive, helper_relations)
        }
        RirNode::ChainJoin {
            left,
            right,
            fallback,
            ..
        } => {
            count_helper_relation_scans(left, helper_relations)
                + count_helper_relation_scans(right, helper_relations)
                + count_helper_relation_scans(fallback, helper_relations)
        }
        RirNode::TensorMaskedJoin { .. } | RirNode::Unit => 0,
    }
}

fn count_helper_relation_leaf_scans(node: &RirNode, helper_relations: &BTreeSet<RelId>) -> usize {
    match node {
        RirNode::Scan { rel } => usize::from(helper_relations.contains(rel)),
        RirNode::Filter { input, .. }
        | RirNode::Project { input, .. }
        | RirNode::Distinct { input, .. }
        | RirNode::GroupBy { input, .. } => {
            count_helper_relation_leaf_scans(input, helper_relations)
        }
        RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
            count_helper_relation_leaf_scans(left, helper_relations)
                + count_helper_relation_leaf_scans(right, helper_relations)
        }
        RirNode::Union { inputs } => inputs
            .iter()
            .map(|input| count_helper_relation_leaf_scans(input, helper_relations))
            .sum(),
        RirNode::Fixpoint {
            base, recursive, ..
        } => {
            count_helper_relation_leaf_scans(base, helper_relations)
                + count_helper_relation_leaf_scans(recursive, helper_relations)
        }
        RirNode::MultiWayJoin {
            inputs, fallback, ..
        } => {
            inputs
                .iter()
                .map(|input| count_helper_relation_leaf_scans(input, helper_relations))
                .sum::<usize>()
                + count_helper_relation_leaf_scans(fallback, helper_relations)
        }
        RirNode::ChainJoin {
            left,
            right,
            fallback,
            ..
        } => {
            count_helper_relation_leaf_scans(left, helper_relations)
                + count_helper_relation_leaf_scans(right, helper_relations)
                + count_helper_relation_leaf_scans(fallback, helper_relations)
        }
        RirNode::TensorMaskedJoin { .. } | RirNode::Unit => 0,
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

fn checked_sum(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right).ok_or_else(|| {
        XlogError::Kernel(format!(
            "epistemic GPU workspace size overflow: {left} + {right}"
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

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::ScalarType;
    use xlog_ir::EirTerm;

    #[test]
    fn tuple_key_expectation_encodes_ground_integer_for_u32_column() {
        let expectation =
            TupleKeyExpectation::from_term(&EirTerm::Integer(42), ScalarType::U32).unwrap();

        assert_eq!(
            expectation,
            TupleKeyExpectation {
                bits: 42,
                type_code: ScalarType::U32.to_code(),
            }
        );
    }

    #[test]
    fn tuple_key_expectation_encodes_symbol_for_symbol_column() {
        let expectation =
            TupleKeyExpectation::from_term(&EirTerm::Symbol(7), ScalarType::Symbol).unwrap();

        assert_eq!(
            expectation,
            TupleKeyExpectation {
                bits: 7,
                type_code: ScalarType::Symbol.to_code(),
            }
        );
    }

    #[test]
    fn tuple_key_expectation_rejects_variable_as_ground_expectation() {
        let err =
            TupleKeyExpectation::from_term(&EirTerm::Variable("X".to_string()), ScalarType::U32)
                .expect_err("variable tuple keys require bound-output matching");

        match err {
            XlogError::UnsupportedEpistemicConstruct { construct, context } => {
                assert_eq!(construct, "epistemic GPU tuple-key expectation");
                assert!(context.contains("cannot be encoded as a ground tuple-key expectation"));
            }
            other => panic!("expected tuple-key expectation error, got {other:?}"),
        }
    }
}
