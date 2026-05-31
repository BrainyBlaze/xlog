//! Epistemic GPU workspace allocation.

use std::{collections::BTreeSet, ffi::c_void, sync::Arc};

use cudarc::driver::LaunchConfig;
use xlog_core::{RelId, Result, ScalarType, Schema, XlogError};
use xlog_cuda::provider::{
    epistemic_kernels, HostLaunchMetadataTransferStats, HostTransferStats, EPISTEMIC_MODULE,
};
use xlog_cuda::{
    memory::{validate_logical_row_count, TrackedCudaSlice},
    sys, AsKernelParam, CudaBuffer, CudaColumn, DeviceSlice, DriverError, LaunchAsync,
};
use xlog_ir::rir::{MultiwayPlan, PlannedHashReason, RirNode, StreamGroupId};
use xlog_ir::{
    EirEpistemicMode, EirEpistemicOp, EirTerm, EpistemicCpuFallbackCounters,
    EpistemicExecutablePlan, EpistemicGpuBufferKind, EpistemicGpuHotPathPhase, EpistemicGpuPlan,
    EpistemicTupleMembershipBinding, EpistemicWcojReductionStatus,
};

use super::Executor;

const XLOG_CONSTRAINT_RELATION_PREFIX: &str = "__xlog_constraint_";

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
        let world_view_stride = capacities
            .max_worlds
            .max(world_view_bitset_bytes_per_candidate(literal_count)?);
        let world_view_bytes = checked_product(capacities.max_candidates, world_view_stride)?;
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
        self.try_total_bytes()
            .expect("epistemic GPU workspace layout byte total overflowed")
    }

    /// Checked total workspace byte size across every device buffer category.
    pub fn try_total_bytes(&self) -> Result<usize> {
        let rejection_reason_bytes =
            checked_product(self.rejection_reason_slots, std::mem::size_of::<u32>())?;
        checked_sum(
            checked_sum(
                checked_sum(self.candidate_assumption_bytes, self.world_view_bytes)?,
                self.model_membership_bytes,
            )?,
            rejection_reason_bytes,
        )
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
    /// Per-candidate firing integrity-constraint index buffer. Parallel to
    /// `rejection_reasons`, sized `layout.rejection_reason_slots`. Holds the
    /// declaration-order index of the constraint that rejected a candidate, or
    /// the sentinel `u32::MAX` when no integrity constraint rejected it. The
    /// reason code in `rejection_reasons` is left at 6 for constraint
    /// violations; this buffer adds the constraint-specific detail.
    pub constraint_violation_index: TrackedCudaSlice<u32>,
}

impl EpistemicGpuWorkspace {
    /// Require retained device buffers to match the certified workspace layout.
    pub fn require_buffer_lengths_match_layout(&self, construct: &str) -> Result<()> {
        if self.candidate_assumptions.len() != self.layout.candidate_assumption_bytes
            || self.world_views.len() != self.layout.world_view_bytes
            || self.model_membership.len() != self.layout.model_membership_bytes
            || self.rejection_reasons.len() != self.layout.rejection_reason_slots
            || self.constraint_violation_index.len() != self.layout.rejection_reason_slots
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "prepared GPU workspace buffer lengths do not match layout: \
                     candidate_bytes={}/{} world_view_bytes={}/{} model_membership_bytes={}/{} \
                     rejection_reason_slots={}/{} constraint_violation_index_slots={}/{}",
                    self.candidate_assumptions.len(),
                    self.layout.candidate_assumption_bytes,
                    self.world_views.len(),
                    self.layout.world_view_bytes,
                    self.model_membership.len(),
                    self.layout.model_membership_bytes,
                    self.rejection_reasons.len(),
                    self.layout.rejection_reason_slots,
                    self.constraint_violation_index.len(),
                    self.layout.rejection_reason_slots
                ),
            });
        }

        Ok(())
    }
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
    /// Bounded model slots available per reduction during final tuple materialization.
    pub bounded_model_slots_per_reduction: usize,
    /// Output row capacity that can be checked against row-specific model slots.
    pub row_specific_membership_row_capacity: usize,
    /// Output row capacity beyond the bounded model-slot window.
    pub row_filter_row_capacity_outside_model_slot_window: usize,
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

/// Trace proving the epistemic GPU hot path avoided tracked data-plane host transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuTransferBudgetTrace {
    /// Number of candidate rows covered by this transfer-budget check.
    pub candidate_count: usize,
    /// Tracked device-to-host bytes observed inside the GPU hot path.
    pub tracked_dtoh_bytes: u64,
    /// Tracked data-plane host-to-device bytes observed inside the GPU hot path.
    pub tracked_htod_bytes: u64,
    /// Tracked device-to-host calls observed inside the GPU hot path.
    pub tracked_dtoh_calls: u64,
    /// Tracked data-plane host-to-device calls observed inside the GPU hot path.
    pub tracked_htod_calls: u64,
    /// Tracked aggregate host-to-device bytes observed inside the GPU hot path.
    pub tracked_aggregate_htod_bytes: u64,
    /// Tracked aggregate host-to-device calls observed inside the GPU hot path.
    pub tracked_aggregate_htod_calls: u64,
    /// Tracked launch-metadata host-to-device bytes observed inside the GPU hot path.
    pub tracked_launch_metadata_htod_bytes: u64,
    /// Tracked launch-metadata host-to-device calls observed inside the GPU hot path.
    pub tracked_launch_metadata_htod_calls: u64,
    /// Tracked data-plane host-to-device bytes observed inside the GPU hot path.
    pub tracked_data_plane_htod_bytes: u64,
    /// Tracked data-plane host-to-device calls observed inside the GPU hot path.
    pub tracked_data_plane_htod_calls: u64,
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

/// Bounded validation of reduced integrity-constraint relations after GPU execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuConstraintValidationTrace {
    /// Number of compiler-generated `__xlog_constraint_N` relations checked.
    pub checked_constraint_relations: usize,
    /// Number of checked constraint relations that contained violating rows.
    pub violated_constraint_relations: usize,
    /// Constraint row-count reads that had to consult device metadata.
    pub row_count_device_reads: u32,
}

impl EpistemicGpuConstraintValidationTrace {
    /// Require reduced integrity-constraint validation to match preflight obligations.
    pub fn require_matches_preflight(
        &self,
        construct: &str,
        preflight: &EpistemicGpuRuntimePreflight,
    ) -> Result<()> {
        if self.checked_constraint_relations != preflight.reduced_constraint_relation_count
            || self.violated_constraint_relations != 0
            || self.row_count_device_reads as usize > self.checked_constraint_relations
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "constraint validation trace must match reduced runtime preflight, got \
                     checked={} expected_checked={} violations={} row_count_reads={}",
                    self.checked_constraint_relations,
                    preflight.reduced_constraint_relation_count,
                    self.violated_constraint_relations,
                    self.row_count_device_reads
                ),
            });
        }

        Ok(())
    }
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
    /// Accepted world view satisfied an epistemic integrity constraint body.
    WorldViewConstraintViolation,
}

impl EpistemicGpuRejectionReason {
    /// Return the raw device rejection code used by the CUDA kernels.
    pub const fn code(self) -> u32 {
        match self {
            Self::InactiveWorld => 2,
            Self::InvalidCandidateBit => 3,
            Self::MissingReducedModel => 4,
            Self::UnsatisfiedMembership => 5,
            Self::WorldViewConstraintViolation => 6,
        }
    }

    /// Decode a nonzero device rejection code into a typed reason.
    pub fn from_code(code: u32) -> Result<Self> {
        match code {
            2 => Ok(Self::InactiveWorld),
            3 => Ok(Self::InvalidCandidateBit),
            4 => Ok(Self::MissingReducedModel),
            5 => Ok(Self::UnsatisfiedMembership),
            6 => Ok(Self::WorldViewConstraintViolation),
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
    /// Constraint-specific reason per rejected candidate, aligned 1:1 with
    /// `rejected_candidate_indices`. `Some(idx)` when an integrity constraint
    /// (reason code 6) rejected the candidate, where `idx` is the firing
    /// constraint's declaration-order index; `None` for every other rejection
    /// reason. Surfaces EGB-04.K2 constraint-specific rejection detail.
    pub constraint_violation_indices: Vec<Option<u32>>,
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

/// Trace proving epistemic integrity constraints were evaluated against world
/// views on GPU.
///
/// World-view integrity constraints (`:- know unsafe().`) prune accepted
/// candidate world views on device after modal world-view validation. The
/// device kernel never reads accepted worlds back to the host, so accepted
/// execution keeps `host_write_ops` at zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuConstraintWorldViewValidationTrace {
    /// Number of epistemic integrity constraints checked on device.
    pub constraint_count: usize,
    /// Number of constraint-body literal references checked on device.
    pub constraint_literal_refs: usize,
    /// Number of candidate world views checked by the constraint kernel.
    pub candidates_checked: usize,
    /// Rejection-reason slots written by the kernel.
    pub rejection_reason_slots_written: usize,
    /// Constraint world-view validation kernel launches.
    pub kernel_launches: u32,
    /// Host writes used by constraint validation. Accepted execution requires zero.
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
        let elapsed_nanos = ((elapsed_ms as f64) * 1_000_000.0).round();
        if elapsed_nanos >= u64::MAX as f64 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU kernel timing trace".to_string(),
                context: format!(
                    "CUDA elapsed time {elapsed_ms}ms exceeds the u64 nanosecond trace counter"
                ),
            });
        }

        Ok(Self {
            cuda_event_pairs: 1,
            timing_sync_ops: 1,
            kernel_elapsed_nanos: elapsed_nanos as u64,
        })
    }

    /// Whether CUDA-event timing was recorded for this trace.
    pub const fn is_recorded(&self) -> bool {
        self.cuda_event_pairs > 0 && self.timing_sync_ops > 0
    }

    /// Saturating sum used when aggregating multi-kernel or split-batch traces.
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            cuda_event_pairs: self.cuda_event_pairs.saturating_add(other.cuda_event_pairs),
            timing_sync_ops: self.timing_sync_ops.saturating_add(other.timing_sync_ops),
            kernel_elapsed_nanos: self
                .kernel_elapsed_nanos
                .saturating_add(other.kernel_elapsed_nanos),
        }
    }

    /// Checked sum used by accepted certification paths.
    pub fn checked_add(self, other: Self) -> Result<Self> {
        Ok(Self {
            cuda_event_pairs: self
                .cuda_event_pairs
                .checked_add(other.cuda_event_pairs)
                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU kernel timing trace".to_string(),
                    context: format!(
                        "CUDA event-pair counter overflowed while adding {} to {}",
                        other.cuda_event_pairs, self.cuda_event_pairs
                    ),
                })?,
            timing_sync_ops: self
                .timing_sync_ops
                .checked_add(other.timing_sync_ops)
                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU kernel timing trace".to_string(),
                    context: format!(
                        "CUDA timing-sync counter overflowed while adding {} to {}",
                        other.timing_sync_ops, self.timing_sync_ops
                    ),
                })?,
            kernel_elapsed_nanos: self
                .kernel_elapsed_nanos
                .checked_add(other.kernel_elapsed_nanos)
                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU kernel timing trace".to_string(),
                    context: format!(
                        "kernel elapsed-time counter overflowed while adding {} to {}",
                        other.kernel_elapsed_nanos, self.kernel_elapsed_nanos
                    ),
                })?,
        })
    }

    /// Aggregate timing traces from a single execution or split-batch result.
    pub fn sum(traces: impl IntoIterator<Item = Self>) -> Self {
        traces
            .into_iter()
            .fold(Self::unrecorded(), Self::saturating_add)
    }

    /// Checked aggregate timing traces for accepted certification paths.
    pub fn checked_sum(traces: impl IntoIterator<Item = Self>) -> Result<Self> {
        traces
            .into_iter()
            .try_fold(Self::unrecorded(), Self::checked_add)
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

        let candidate_assumption_bytes = checked_product(literal_count, candidate_count)?;
        require_u32_launch_bound(
            candidate_assumption_bytes,
            "epistemic GPU candidate generation launch",
        )?;

        Ok(Self {
            literal_count,
            generated_candidates: candidate_count,
            candidate_assumption_bytes,
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
        require_u32_launch_dimensions(
            &[literal_count, candidate_count],
            "epistemic GPU validation launch",
        )?;
        let candidate_assumption_bytes_checked = checked_product(literal_count, candidate_count)?;

        Ok(Self {
            literal_count,
            validated_candidates: candidate_count,
            candidate_assumption_bytes_checked,
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

    /// Require validation coverage to match the generated candidate workspace.
    pub fn require_matches_candidate_generation(
        &self,
        construct: &str,
        candidate_generation: &EpistemicGpuCandidateGenerationTrace,
    ) -> Result<()> {
        let expected_world_view_bytes = checked_product(
            world_view_bitset_bytes_per_candidate(candidate_generation.literal_count)?,
            candidate_generation.generated_candidates,
        )?;
        if self.literal_count != candidate_generation.literal_count
            || self.validated_candidates != candidate_generation.generated_candidates
            || self.candidate_assumption_bytes_checked
                != candidate_generation.candidate_assumption_bytes
            || self.world_view_bytes_checked != expected_world_view_bytes
            || self.rejection_reason_slots_written != candidate_generation.generated_candidates
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "candidate validation trace does not match generated GPU candidates: \
                     literals={}/{} candidates={}/{} candidate_bytes={}/{} \
                     world_view_bytes={}/{} rejection_slots={}/{}",
                    self.literal_count,
                    candidate_generation.literal_count,
                    self.validated_candidates,
                    candidate_generation.generated_candidates,
                    self.candidate_assumption_bytes_checked,
                    candidate_generation.candidate_assumption_bytes,
                    self.world_view_bytes_checked,
                    expected_world_view_bytes,
                    self.rejection_reason_slots_written,
                    candidate_generation.generated_candidates
                ),
            });
        }

        Ok(())
    }
}

impl EpistemicGpuMaterializationTrace {
    /// Build a materialization trace for a bounded device launch.
    pub fn for_count(candidate_count: usize) -> Result<Self> {
        require_positive(candidate_count, "epistemic GPU materialization candidates")?;
        require_u32_launch_bound(candidate_count, "epistemic GPU materialization launch")?;

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
        require_u32_launch_bound(candidate_count, "epistemic GPU final-result launch")?;

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
        require_u32_launch_bound(output_row_capacity, "epistemic GPU final-tuple output rows")?;
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
        require_u32_launch_bound(
            model_membership_bytes_checked,
            "epistemic GPU final-tuple membership launch",
        )?;
        let output_row_count_device_reads = checked_sum(output_column_count, 1)?;
        let kernel_launches = checked_sum(output_row_count_device_reads, 1)?;
        if kernel_launches > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final-tuple kernel launches".to_string(),
                estimated_bytes: kernel_launches as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        Ok(Self {
            output_column_count,
            output_row_capacity,
            tuple_bytes_capacity,
            output_row_count_device_reads: output_row_count_device_reads as u32,
            model_membership_bytes_checked,
            bounded_model_slots_per_reduction: models_per_reduction,
            row_specific_membership_row_capacity: 0,
            row_filter_row_capacity_outside_model_slot_window: 0,
            world_view_slots_checked: candidate_count,
            row_filter_count: 0,
            negated_row_filter_count: 0,
            final_row_count_device_writes: 1,
            kernel_launches: kernel_launches as u32,
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
        if row_filter_count > 0 {
            self.row_specific_membership_row_capacity = self
                .output_row_capacity
                .min(self.bounded_model_slots_per_reduction);
            self.row_filter_row_capacity_outside_model_slot_window = self
                .output_row_capacity
                .saturating_sub(self.row_specific_membership_row_capacity);
        }
        Ok(self)
    }

    /// Require GPU evidence that row-filtered tuple output fits the validated coverage window.
    pub fn require_row_filter_materialization_evidence(
        &self,
        construct: &str,
        final_output_rows: usize,
    ) -> Result<()> {
        if final_output_rows > self.output_row_capacity {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "final tuple materialization reported {} logical rows for output row \
                     capacity {}",
                    final_output_rows, self.output_row_capacity
                ),
            });
        }
        if self.negated_row_filter_count > self.row_filter_count {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "row-filtered final tuple materialization reported {} negated row filters \
                     for {} total row filters",
                    self.negated_row_filter_count, self.row_filter_count
                ),
            });
        }
        if self.row_filter_count == 0 {
            if self.row_specific_membership_row_capacity != 0
                || self.row_filter_row_capacity_outside_model_slot_window != 0
            {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: construct.to_string(),
                    context: format!(
                        "final tuple materialization without row filters reported row-filter \
                         coverage row_specific_capacity={} fallback_capacity={}",
                        self.row_specific_membership_row_capacity,
                        self.row_filter_row_capacity_outside_model_slot_window
                    ),
                });
            }
            return Ok(());
        }

        let covered_row_capacity = checked_sum(
            self.row_specific_membership_row_capacity,
            self.row_filter_row_capacity_outside_model_slot_window,
        )?;
        if self.output_row_capacity == 0
            || self.row_specific_membership_row_capacity == 0
            || covered_row_capacity != self.output_row_capacity
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "row-filtered final tuple materialization requires GPU row-filter coverage, \
                     got row_filters={} final_output_rows={} output_row_capacity={} \
                     row_specific_capacity={} fallback_capacity={} model_slots_per_reduction={}",
                    self.row_filter_count,
                    final_output_rows,
                    self.output_row_capacity,
                    self.row_specific_membership_row_capacity,
                    self.row_filter_row_capacity_outside_model_slot_window,
                    self.bounded_model_slots_per_reduction
                ),
            });
        }

        let fallback_rows =
            final_output_rows.saturating_sub(self.row_specific_membership_row_capacity);
        if fallback_rows > self.row_filter_row_capacity_outside_model_slot_window {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "row-filtered final tuple materialization has {} logical rows beyond the \
                     row-specific model-slot window but only {} fallback row-filter capacity",
                    fallback_rows, self.row_filter_row_capacity_outside_model_slot_window
                ),
            });
        }
        Ok(())
    }
}

impl EpistemicGpuTransferBudgetTrace {
    /// Build a hot-path transfer trace from provider host-transfer snapshots.
    pub fn from_host_transfer_stats(
        candidate_count: usize,
        before: HostTransferStats,
        after: HostTransferStats,
    ) -> Result<Self> {
        Self::from_host_transfer_stats_with_launch_metadata(
            candidate_count,
            before,
            after,
            HostLaunchMetadataTransferStats::default(),
            HostLaunchMetadataTransferStats::default(),
        )
    }

    /// Build a hot-path transfer trace while distinguishing bounded launch
    /// metadata H2D from data-plane transfers.
    pub fn from_host_transfer_stats_with_launch_metadata(
        candidate_count: usize,
        before: HostTransferStats,
        after: HostTransferStats,
        launch_metadata_before: HostLaunchMetadataTransferStats,
        launch_metadata_after: HostLaunchMetadataTransferStats,
    ) -> Result<Self> {
        require_positive(candidate_count, "epistemic GPU transfer-budget candidates")?;

        let tracked_dtoh_bytes =
            transfer_counter_delta("dtoh_bytes", before.dtoh_bytes, after.dtoh_bytes)?;
        let tracked_data_plane_htod_bytes =
            transfer_counter_delta("htod_bytes", before.htod_bytes, after.htod_bytes)?;
        let tracked_dtoh_calls =
            transfer_counter_delta("dtoh_calls", before.dtoh_calls, after.dtoh_calls)?;
        let tracked_data_plane_htod_calls =
            transfer_counter_delta("htod_calls", before.htod_calls, after.htod_calls)?;
        let tracked_launch_metadata_htod_bytes = transfer_counter_delta(
            "launch_metadata_htod_bytes",
            launch_metadata_before.htod_bytes,
            launch_metadata_after.htod_bytes,
        )?;
        let tracked_launch_metadata_htod_calls = transfer_counter_delta(
            "launch_metadata_htod_calls",
            launch_metadata_before.htod_calls,
            launch_metadata_after.htod_calls,
        )?;
        let tracked_aggregate_htod_bytes = tracked_data_plane_htod_bytes
            .checked_add(tracked_launch_metadata_htod_bytes)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU transfer budget".to_string(),
                context: format!(
                    "aggregate H2D bytes overflowed while adding launch metadata: \
                     data_plane_htod_bytes={tracked_data_plane_htod_bytes}, \
                     launch_metadata_htod_bytes={tracked_launch_metadata_htod_bytes}"
                ),
            })?;
        let tracked_aggregate_htod_calls = tracked_data_plane_htod_calls
            .checked_add(tracked_launch_metadata_htod_calls)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU transfer budget".to_string(),
                context: format!(
                    "aggregate H2D calls overflowed while adding launch metadata: \
                     data_plane_htod_calls={tracked_data_plane_htod_calls}, \
                     launch_metadata_htod_calls={tracked_launch_metadata_htod_calls}"
                ),
            })?;

        if tracked_launch_metadata_htod_bytes != 0 && tracked_launch_metadata_htod_calls == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU transfer budget".to_string(),
                context: format!(
                    "launch metadata H2D bytes require matching H2D calls, got bytes={} calls=0",
                    tracked_launch_metadata_htod_bytes
                ),
            });
        }
        if tracked_launch_metadata_htod_calls != 0 && tracked_launch_metadata_htod_bytes == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU transfer budget".to_string(),
                context: format!(
                    "launch metadata H2D calls require matching payload bytes, got calls={} bytes=0",
                    tracked_launch_metadata_htod_calls
                ),
            });
        }

        if tracked_dtoh_bytes != 0
            || tracked_data_plane_htod_bytes != 0
            || tracked_dtoh_calls != 0
            || tracked_data_plane_htod_calls != 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU transfer budget".to_string(),
                context: format!(
                    "tracked host transfer in GPU hot path: tracked data-plane host transfer: \
                     dtoh_bytes={tracked_dtoh_bytes}, \
                     data_plane_htod_bytes={tracked_data_plane_htod_bytes}, \
                     dtoh_calls={tracked_dtoh_calls}, \
                     data_plane_htod_calls={tracked_data_plane_htod_calls}, \
                     launch_metadata_htod_bytes={tracked_launch_metadata_htod_bytes}, \
                     launch_metadata_htod_calls={tracked_launch_metadata_htod_calls}"
                ),
            });
        }

        Ok(Self {
            candidate_count,
            tracked_dtoh_bytes,
            tracked_htod_bytes: tracked_data_plane_htod_bytes,
            tracked_dtoh_calls,
            tracked_htod_calls: tracked_data_plane_htod_calls,
            tracked_aggregate_htod_bytes,
            tracked_aggregate_htod_calls,
            tracked_launch_metadata_htod_bytes,
            tracked_launch_metadata_htod_calls,
            tracked_data_plane_htod_bytes,
            tracked_data_plane_htod_calls,
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

    /// Require retained final-result transfer accounting to match the final device buffer.
    pub fn require_matches_final_output(
        &self,
        construct: &str,
        final_output: &CudaBuffer,
    ) -> Result<()> {
        let Some(cached_rows) = final_output.cached_row_count() else {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context:
                    "final-result transfer certification requires cached device final row count"
                        .to_string(),
            });
        };
        let logical_rows =
            validate_logical_row_count(final_output.num_rows(), cached_rows as usize).map_err(
                |err| XlogError::UnsupportedEpistemicConstruct {
                    construct: construct.to_string(),
                    context: format!("invalid final-output logical row count: {err}"),
                },
            )?;
        let row_width = final_output.schema().row_size_bytes();
        let payload_bytes = checked_product(logical_rows, row_width)? as u64;
        if self.final_output_rows != logical_rows
            || self.final_output_column_count != final_output.arity()
            || self.final_output_row_width_bytes != row_width
            || self.final_output_payload_bytes != payload_bytes
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "final-result transfer trace does not match final device output: rows={}/{} \
                     columns={}/{} row_width={}/{} payload_bytes={}/{}",
                    self.final_output_rows,
                    logical_rows,
                    self.final_output_column_count,
                    final_output.arity(),
                    self.final_output_row_width_bytes,
                    row_width,
                    self.final_output_payload_bytes,
                    payload_bytes
                ),
            });
        }
        if self.row_count_device_reads > 1 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "final-result transfer reads one device row-count scalar at most, got {}",
                    self.row_count_device_reads
                ),
            });
        }

        Ok(())
    }
}

impl EpistemicGpuSemanticTrace {
    /// Require semantic phase counts to match the retained GPU execution traces.
    pub fn require_matches_execution_traces(
        &self,
        construct: &str,
        candidate_generation: &EpistemicGpuCandidateGenerationTrace,
        propagation: &EpistemicGpuPropagationTrace,
        model_membership: &EpistemicGpuModelMembershipTrace,
        world_view_validation: &EpistemicGpuWorldViewValidationTrace,
    ) -> Result<()> {
        let expected_pruned = self
            .generated_candidates
            .checked_sub(propagation.propagated_candidates)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "semantic trace phase counts cannot propagate more candidates than were \
                     generated: generated={} propagated={}",
                    self.generated_candidates, propagation.propagated_candidates
                ),
            })?;
        let expected_reduced_model_slots = checked_product(
            checked_product(
                world_view_validation.candidates_checked,
                model_membership.reduction_count,
            )?,
            model_membership.models_per_reduction,
        )?;
        let expected_guesses = checked_product(
            candidate_generation.generated_candidates,
            candidate_generation.literal_count,
        )?;
        if self.generated_candidates != candidate_generation.generated_candidates
            || self.guesses != expected_guesses
            || self.propagated_candidates != propagation.propagated_candidates
            || self.pruned_candidates != expected_pruned
            || self.tested_candidates != world_view_validation.candidates_checked
            || self.reduced_model_slots_checked != expected_reduced_model_slots
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "semantic trace phase counts must match retained GPU execution traces, got \
                     generated={} expected_generated={} guesses={} expected_guesses={} \
                     propagated={} expected_propagated={} pruned={} expected_pruned={} \
                     tested={} expected_tested={} reduced_model_slots={} \
                     expected_reduced_model_slots={}",
                    self.generated_candidates,
                    candidate_generation.generated_candidates,
                    self.guesses,
                    expected_guesses,
                    self.propagated_candidates,
                    propagation.propagated_candidates,
                    self.pruned_candidates,
                    expected_pruned,
                    self.tested_candidates,
                    world_view_validation.candidates_checked,
                    self.reduced_model_slots_checked,
                    expected_reduced_model_slots
                ),
            });
        }

        Ok(())
    }

    /// Require bounded rejection-buffer metadata accounting to match generated candidates.
    pub fn require_rejection_metadata_accounting(&self, construct: &str) -> Result<()> {
        let expected_metadata_bytes =
            checked_product(self.generated_candidates, std::mem::size_of::<u32>())? as u64;
        if self.rejection_reason_device_reads != 1
            || self.rejection_reason_metadata_bytes != expected_metadata_bytes
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "semantic trace rejection metadata accounting must match the bounded device \
                     rejection-buffer read, got reads={} bytes={} expected_reads=1 \
                     expected_bytes={}",
                    self.rejection_reason_device_reads,
                    self.rejection_reason_metadata_bytes,
                    expected_metadata_bytes
                ),
            });
        }

        Ok(())
    }

    /// Require accepted/rejected candidate indices to partition generated candidates.
    pub fn require_candidate_index_partition(&self, construct: &str) -> Result<()> {
        let accounted_candidates = self.accepted_candidates.checked_add(self.rejected_candidates).ok_or_else(|| {
            XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "semantic trace candidate index partition accounting overflowed: accepted={} rejected={}",
                    self.accepted_candidates, self.rejected_candidates
                ),
            }
        })?;
        if self.accepted_candidate_indices.len() != self.accepted_candidates
            || self.rejected_candidate_indices.len() != self.rejected_candidates
            || self.accepted_world_views != self.accepted_candidates
            || accounted_candidates != self.generated_candidates
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "semantic trace candidate index partition requires counts and index vectors \
                     to match generated candidates, got generated={} accepted={} \
                     accepted_indices={} accepted_world_views={} rejected={} rejected_indices={}",
                    self.generated_candidates,
                    self.accepted_candidates,
                    self.accepted_candidate_indices.len(),
                    self.accepted_world_views,
                    self.rejected_candidates,
                    self.rejected_candidate_indices.len()
                ),
            });
        }
        if self.rejection_reasons.len() != self.rejected_candidates {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "semantic trace rejection reason count must match rejected candidates, got \
                     reasons={} rejected={}",
                    self.rejection_reasons.len(),
                    self.rejected_candidates
                ),
            });
        }
        self.typed_rejection_reasons()?;

        let mut seen = BTreeSet::new();
        for (kind, indices) in [
            ("accepted", self.accepted_candidate_indices.as_slice()),
            ("rejected", self.rejected_candidate_indices.as_slice()),
        ] {
            for &index in indices {
                if index >= self.generated_candidates {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: construct.to_string(),
                        context: format!(
                            "semantic trace candidate index partition has out-of-range {kind} \
                             index {index} for generated candidate count {}",
                            self.generated_candidates
                        ),
                    });
                }
                if !seen.insert(index) {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: construct.to_string(),
                        context: format!(
                            "semantic trace candidate index partition contains duplicate \
                             candidate index {index}"
                        ),
                    });
                }
            }
        }
        if seen.len() != self.generated_candidates {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "semantic trace candidate index partition covers {} of {} generated \
                     candidates",
                    seen.len(),
                    self.generated_candidates
                ),
            });
        }

        Ok(())
    }

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
        if propagation.literal_count != candidate_generation.literal_count
            || model_membership.literal_count != candidate_generation.literal_count
            || world_view_validation.literal_count != candidate_generation.literal_count
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU semantic trace".to_string(),
                context: format!(
                    "semantic trace requires all GPU stages to agree on literal count, got \
                     generated={} propagated={} membership={} validation={}",
                    candidate_generation.literal_count,
                    propagation.literal_count,
                    model_membership.literal_count,
                    world_view_validation.literal_count
                ),
            });
        }
        let pruned_candidates = candidate_count
            .checked_sub(propagation.propagated_candidates)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU semantic trace".to_string(),
                context: format!(
                    "semantic trace cannot prune more candidates than were generated: \
                     generated={} propagated={}",
                    candidate_count, propagation.propagated_candidates
                ),
            })?;
        if propagation.rejection_reason_slots_written < candidate_count {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU semantic trace".to_string(),
                context: format!(
                    "semantic trace requires rejection metadata for every generated candidate, \
                     got generated={} rejection_slots_initialized={}",
                    candidate_count, propagation.rejection_reason_slots_written
                ),
            });
        }
        if model_membership.candidates_checked != candidate_count
            || world_view_validation.candidates_checked != candidate_count
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU semantic trace".to_string(),
                context: format!(
                    "semantic trace requires GPU validation coverage for every generated \
                     candidate, got generated={} membership_checked={} validation_checked={}",
                    candidate_count,
                    model_membership.candidates_checked,
                    world_view_validation.candidates_checked
                ),
            });
        }
        if model_membership.reduction_count != world_view_validation.reduction_count
            || model_membership.models_per_reduction != world_view_validation.models_per_reduction
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU semantic trace".to_string(),
                context: format!(
                    "semantic trace requires model-membership and world-view validation layouts \
                     to match, got membership_reductions={} validation_reductions={} \
                     membership_models_per_reduction={} validation_models_per_reduction={}",
                    model_membership.reduction_count,
                    world_view_validation.reduction_count,
                    model_membership.models_per_reduction,
                    world_view_validation.models_per_reduction
                ),
            });
        }

        let raw_rejection_reasons = provider
            .dtoh_small_metadata_untracked(&workspace.rejection_reasons, candidate_count)?;
        // Bounded metadata read of the parallel constraint-violation index buffer.
        // Like `rejection_reasons`, this is an untracked post-hot-path metadata
        // read, not a data-plane transfer.
        let raw_constraint_violation_index = provider.dtoh_small_metadata_untracked(
            &workspace.constraint_violation_index,
            candidate_count,
        )?;
        let constraint_violation_code =
            EpistemicGpuRejectionReason::WorldViewConstraintViolation.code();
        let mut accepted_candidate_indices = Vec::new();
        let mut rejected_candidate_indices = Vec::new();
        let mut rejection_reasons = Vec::new();
        let mut constraint_violation_indices: Vec<Option<u32>> = Vec::new();
        for (candidate_index, reason) in raw_rejection_reasons.into_iter().enumerate() {
            if reason == 0 {
                accepted_candidate_indices.push(candidate_index);
            } else {
                EpistemicGpuRejectionReason::from_code(reason)?;
                rejected_candidate_indices.push(candidate_index);
                rejection_reasons.push(reason);
                // Gate the constraint-specific index on the integrity-constraint
                // reason code: the kernel writes `rejection_reasons[c] = 6` and
                // `constraint_violation_index[c] = constraint` together, so the
                // index is trustworthy exactly when the reason is 6. Any other
                // reason -> None, independent of buffer contents (also defends
                // the zero-constraint path where the sentinel is never written).
                let firing = raw_constraint_violation_index
                    .get(candidate_index)
                    .copied()
                    .unwrap_or(u32::MAX);
                if reason == constraint_violation_code && firing != u32::MAX {
                    constraint_violation_indices.push(Some(firing));
                } else {
                    constraint_violation_indices.push(None);
                }
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
            guesses: checked_product(candidate_count, candidate_generation.literal_count)?,
            propagated_candidates: propagation.propagated_candidates,
            pruned_candidates,
            tested_candidates: world_view_validation.candidates_checked,
            reduced_model_slots_checked,
            accepted_candidates,
            accepted_candidate_indices,
            accepted_world_views: accepted_candidates,
            rejected_candidates,
            rejected_candidate_indices,
            rejection_reasons,
            constraint_violation_indices,
            // Counts the bounded metadata read of the rejection-reason code buffer
            // specifically (the certification invariant scopes to that buffer's
            // bytes). The parallel constraint-violation index buffer is a
            // separate bounded metadata read tracked alongside it, not folded
            // into this rejection-reason-specific counter.
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
        require_u32_launch_bound(
            model_membership_bytes_written,
            "epistemic GPU model-membership launch",
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
        require_u32_launch_bound(
            model_membership_bytes_written,
            "epistemic GPU model-membership launch",
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

    /// Require the tuple-key device reads planned for this model-membership trace.
    pub fn require_planned_tuple_key_column_reads(
        &self,
        expected_key_column_reads: usize,
    ) -> Result<()> {
        if self.tuple_source_key_column_device_reads as usize != expected_key_column_reads {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU stable-model membership certification".to_string(),
                context: format!(
                    "model-membership tuple-key device column reads must match the planned \
                     nonzero-arity tuple keys, got reads={} expected={}",
                    self.tuple_source_key_column_device_reads, expected_key_column_reads
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
        require_u32_launch_bound(
            model_membership_bytes_checked,
            "epistemic GPU world-view validation membership launch",
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
        require_u32_launch_dimensions(
            &[literal_count, candidate_count],
            "epistemic GPU propagation launch",
        )?;

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
        Self::try_for_layout(layout)
            .expect("epistemic GPU workspace reset trace byte total overflowed")
    }

    /// Build the reset trace implied by a workspace layout, failing closed on overflow.
    pub fn try_for_layout(layout: EpistemicGpuWorkspaceLayout) -> Result<Self> {
        Ok(Self {
            candidate_assumption_bytes: layout.candidate_assumption_bytes,
            world_view_bytes: layout.world_view_bytes,
            model_membership_bytes: layout.model_membership_bytes,
            rejection_reason_bytes: checked_product(
                layout.rejection_reason_slots,
                std::mem::size_of::<u32>(),
            )?,
            device_zero_ops: 4,
            host_write_ops: 0,
        })
    }

    /// Total bytes zeroed by the reset path.
    pub fn total_zeroed_bytes(&self) -> usize {
        self.try_total_zeroed_bytes()
            .expect("epistemic GPU workspace reset byte total overflowed")
    }

    /// Checked total bytes zeroed by the reset path.
    pub fn try_total_zeroed_bytes(&self) -> Result<usize> {
        checked_sum(
            checked_sum(
                checked_sum(self.candidate_assumption_bytes, self.world_view_bytes)?,
                self.model_membership_bytes,
            )?,
            self.rejection_reason_bytes,
        )
    }

    /// Require the retained reset trace to match the prepared workspace layout.
    pub fn require_matches_layout(
        &self,
        construct: &str,
        layout: EpistemicGpuWorkspaceLayout,
    ) -> Result<()> {
        let expected = Self::try_for_layout(layout)?;
        if *self != expected {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "workspace reset trace does not match prepared GPU workspace layout: \
                     candidate_bytes={}/{} world_view_bytes={}/{} model_membership_bytes={}/{} \
                     rejection_reason_bytes={}/{} device_zero_ops={}/{} host_write_ops={}/{}",
                    self.candidate_assumption_bytes,
                    expected.candidate_assumption_bytes,
                    self.world_view_bytes,
                    expected.world_view_bytes,
                    self.model_membership_bytes,
                    expected.model_membership_bytes,
                    self.rejection_reason_bytes,
                    expected.rejection_reason_bytes,
                    self.device_zero_ops,
                    expected.device_zero_ops,
                    self.host_write_ops,
                    expected.host_write_ops
                ),
            });
        }

        Ok(())
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
    /// Compiler-generated reduced integrity-constraint relations to validate.
    pub reduced_constraint_relation_count: usize,
    /// Reduced rules that the epistemic planner marked as requiring WCOJ eligibility.
    pub wcoj_required_reduction_count: usize,
    /// Number of reduced rules carrying a `MultiWayJoin` route.
    pub multiway_reduction_count: usize,
    /// Number of K-clique WCOJ plans reused from the production planner.
    pub kclique_wcoj_plan_count: usize,
    /// Number of triangle WCOJ routes reused from the production runtime.
    pub wcoj_triangle_route_count: usize,
    /// Number of 4-cycle WCOJ routes reused from the production runtime.
    pub wcoj_4cycle_route_count: usize,
    /// K-clique WCOJ plan counts by arity K=5..8.
    pub kclique_wcoj_plan_count_by_arity: [usize; 4],
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
    /// Planned-hash routes where complete planner costs predicted hash wins.
    pub planned_hash_planner_wins_count: usize,
    /// Planned-hash routes selected because complete WCOJ stats were unavailable.
    pub planned_hash_incomplete_stats_count: usize,
    /// Planned-hash routes carrying finite hash-vs-WCOJ cost evidence.
    pub planned_hash_cost_evidence_count: usize,
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
    /// Solver assumption bindings exported by the semantic plan.
    pub solver_assumption_binding_count: usize,
    /// Solver production capabilities required by the semantic plan.
    pub solver_required_capability_count: usize,
    /// Distinct solver statuses required by the semantic plan.
    pub solver_required_status_count: usize,
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
        executable.gpu_plan.validate_solver_contract()?;
        // A plan may carry MULTIPLE epistemic output heads: a JOINT-SOLVED
        // coalesced multi-head component shares ONE candidate enumeration +
        // world-view validation and materializes each head against the shared
        // accepted world view (see `execute_epistemic_gpu_execution`). Soundness of
        // the coupling is gated in the logic lowering
        // (`classify_cross_component_modal_coupling`); the runtime executes the
        // resulting well-formed plan and is no longer restricted to one head.
        require_epistemic_gpu_kernel_phases(&executable.gpu_plan)?;
        require_epistemic_gpu_buffer_contract(&executable.gpu_plan)?;

        let workspace_layout =
            EpistemicGpuWorkspaceLayout::for_plan(&executable.gpu_plan, capacities)?;
        let mut routes = RuntimeRouteSummary::default();
        let mut reduced_runtime_rule_count = 0usize;
        let mut reduced_constraint_relation_names = Vec::new();
        let wcoj_required_reduction_count = executable
            .gpu_plan
            .reductions
            .iter()
            .filter(|reduction| {
                matches!(
                    reduction.wcoj_status,
                    EpistemicWcojReductionStatus::RequiresPlannerEligibility
                )
            })
            .count();
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
            if rule.head.starts_with(XLOG_CONSTRAINT_RELATION_PREFIX)
                && !reduced_constraint_relation_names
                    .iter()
                    .any(|name| name == &rule.head)
            {
                reduced_constraint_relation_names.push(rule.head.as_str());
            }
            if rule.head.starts_with("__w37_helper_") {
                helper_relation_rule_count += 1;
            }
            helper_relation_scan_count +=
                count_helper_relation_scans(&rule.body, &helper_relation_ids);
            summarize_runtime_routes(&rule.body, &mut routes);
        }

        if wcoj_required_reduction_count > routes.multiway_reduction_count {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU WCOJ route certification".to_string(),
                context: format!(
                    "plan requires {} WCOJ-eligible epistemic reductions, but reduced runtime \
                     plan exposes {} MultiWayJoin routes",
                    wcoj_required_reduction_count, routes.multiway_reduction_count
                ),
            });
        }

        let planned_hash_reason_count = routes
            .planned_hash_planner_wins_count
            .checked_add(routes.planned_hash_incomplete_stats_count)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU planned-hash certification".to_string(),
                context: "planned-hash reason counters overflowed".to_string(),
            })?;
        if planned_hash_reason_count != routes.planned_hash_route_count
            || routes.planned_hash_cost_evidence_count < routes.planned_hash_planner_wins_count
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU planned-hash certification".to_string(),
                context: format!(
                    "planned_hash_routes={}, planner_wins={}, incomplete_stats={}, \
                     finite_cost_evidence={}",
                    routes.planned_hash_route_count,
                    routes.planned_hash_planner_wins_count,
                    routes.planned_hash_incomplete_stats_count,
                    routes.planned_hash_cost_evidence_count
                ),
            });
        }

        if routes.kclique_wcoj_plan_count > 0 && routes.kclique_wcoj_edge_permutation_count == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU K-clique WCOJ certification".to_string(),
                context: format!(
                    "K-clique WCOJ plans require live edge-permutation slots, got \
                     kclique_plans={} edge_permutation_slots=0",
                    routes.kclique_wcoj_plan_count
                ),
            });
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
            reduced_constraint_relation_count: reduced_constraint_relation_names.len(),
            wcoj_required_reduction_count,
            multiway_reduction_count: routes.multiway_reduction_count,
            kclique_wcoj_plan_count: routes.kclique_wcoj_plan_count,
            wcoj_triangle_route_count: routes.wcoj_triangle_route_count,
            wcoj_4cycle_route_count: routes.wcoj_4cycle_route_count,
            kclique_wcoj_plan_count_by_arity: routes.kclique_wcoj_plan_count_by_arity,
            kclique_wcoj_max_arity: routes.kclique_wcoj_max_arity,
            kclique_wcoj_edge_permutation_count: routes.kclique_wcoj_edge_permutation_count,
            kclique_stream_group_count: routes.kclique_stream_groups.len(),
            kclique_skew_scheduled_plan_count: routes.kclique_skew_scheduled_plan_count,
            planned_hash_route_count: routes.planned_hash_route_count,
            planned_hash_planner_wins_count: routes.planned_hash_planner_wins_count,
            planned_hash_incomplete_stats_count: routes.planned_hash_incomplete_stats_count,
            planned_hash_cost_evidence_count: routes.planned_hash_cost_evidence_count,
            sorted_layout_requirement_count: routes.sorted_layout_requirement_count,
            helper_split_spec_count: routes.helper_split_spec_count,
            helper_relation_rule_count,
            helper_relation_scan_count,
            tuple_membership_binding_count: executable.gpu_plan.tuple_membership_bindings.len(),
            solver_assumption_binding_count: executable
                .gpu_plan
                .solver_contract
                .assumption_bindings
                .len(),
            solver_required_capability_count: executable
                .gpu_plan
                .solver_contract
                .distinct_required_capability_count(),
            solver_required_status_count: executable
                .gpu_plan
                .solver_contract
                .distinct_required_status_count(),
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
    /// Planned tuple-membership bindings certified before GPU execution.
    pub tuple_membership_bindings: Vec<EpistemicTupleMembershipBinding>,
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
    /// Checked counter delta for the dispatch window.
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
        Self::try_from_preflight_and_counters(preflight, counters_before, counters_after)
            .expect("runtime counter snapshots must be monotonic")
    }

    /// Build a trace from static preflight data and runtime counter snapshots, failing closed
    /// if runtime proof counters move backwards or overflow while being summarized.
    pub fn try_from_preflight_and_counters(
        preflight: EpistemicGpuRuntimePreflight,
        counters_before: EpistemicGpuRuntimeCounters,
        counters_after: EpistemicGpuRuntimeCounters,
    ) -> Result<Self> {
        let counter_delta = counters_after.checked_delta_since(counters_before)?;
        let wcoj_certification = EpistemicGpuRuntimeWcojCertification::try_for_preflight_and_delta(
            &preflight,
            &counter_delta,
        )?;

        Ok(Self {
            preflight,
            counters_before,
            counters_after,
            counter_delta,
            wcoj_certification,
        })
    }

    /// Fail closed when a WCOJ-required epistemic reduction lacks runtime evidence.
    pub fn require_wcoj_certification(&self) -> Result<()> {
        match self.wcoj_certification {
            EpistemicGpuRuntimeWcojCertification::MissingRequiredWcojDispatch {
                required_multiway_reductions,
                required_kclique_plans,
                observed_wcoj_dispatches,
                observed_kclique_dispatches,
            } => Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU WCOJ dispatch certification".to_string(),
                context: format!(
                    "required_multiway_reductions={required_multiway_reductions}, \
                     required_kclique_plans={required_kclique_plans}, \
                     observed_wcoj_dispatches={observed_wcoj_dispatches}, \
                     observed_kclique_dispatches={observed_kclique_dispatches}"
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
            EpistemicGpuRuntimeWcojCertification::MissingRequiredKcliqueMetadata {
                required_kclique_plans,
                observed_metadata_builds,
                observed_metadata_build_nanos,
            } => Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU K-clique metadata certification".to_string(),
                context: format!(
                    "required_kclique_plans={required_kclique_plans}, \
                     observed_metadata_builds={observed_metadata_builds}, \
                     observed_metadata_build_nanos={observed_metadata_build_nanos}"
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
    /// Recursive Merge-phase K-clique histogram refresh boundaries observed by the executor.
    pub kclique_histogram_refresh_count: u64,
    /// Recursive Merge-phase K-clique histogram refresh accounting time observed by the executor.
    pub kclique_histogram_refresh_nanos: u128,
}

impl EpistemicGpuRuntimeCounters {
    fn checked_counter_delta(counter: &str, after: u64, before: u64) -> Result<u64> {
        after
            .checked_sub(before)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU runtime counter trace".to_string(),
                context: format!(
                    "runtime proof counter {counter} decreased from {before} to {after}"
                ),
            })
    }

    fn checked_counter_delta_u128(counter: &str, after: u128, before: u128) -> Result<u128> {
        after
            .checked_sub(before)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU runtime counter trace".to_string(),
                context: format!(
                    "runtime proof counter {counter} decreased from {before} to {after}"
                ),
            })
    }

    fn checked_counter_sum(counter: &str, values: &[u64]) -> Result<u64> {
        values.iter().try_fold(0u64, |acc, value| {
            acc.checked_add(*value)
                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU runtime counter trace".to_string(),
                    context: format!(
                        "runtime proof counter {counter} overflowed while adding {value} to {acc}"
                    ),
                })
        })
    }

    /// Checked delta from an earlier snapshot.
    pub fn checked_delta_since(self, before: Self) -> Result<Self> {
        Ok(Self {
            wcoj_triangle_dispatch_count: Self::checked_counter_delta(
                "wcoj_triangle_dispatch_count",
                self.wcoj_triangle_dispatch_count,
                before.wcoj_triangle_dispatch_count,
            )?,
            wcoj_4cycle_dispatch_count: Self::checked_counter_delta(
                "wcoj_4cycle_dispatch_count",
                self.wcoj_4cycle_dispatch_count,
                before.wcoj_4cycle_dispatch_count,
            )?,
            w63_chain_dispatch_count: Self::checked_counter_delta(
                "w63_chain_dispatch_count",
                self.w63_chain_dispatch_count,
                before.w63_chain_dispatch_count,
            )?,
            wcoj_clique5_dispatch_count: Self::checked_counter_delta(
                "wcoj_clique5_dispatch_count",
                self.wcoj_clique5_dispatch_count,
                before.wcoj_clique5_dispatch_count,
            )?,
            wcoj_clique6_dispatch_count: Self::checked_counter_delta(
                "wcoj_clique6_dispatch_count",
                self.wcoj_clique6_dispatch_count,
                before.wcoj_clique6_dispatch_count,
            )?,
            wcoj_clique7_dispatch_count: Self::checked_counter_delta(
                "wcoj_clique7_dispatch_count",
                self.wcoj_clique7_dispatch_count,
                before.wcoj_clique7_dispatch_count,
            )?,
            wcoj_clique8_dispatch_count: Self::checked_counter_delta(
                "wcoj_clique8_dispatch_count",
                self.wcoj_clique8_dispatch_count,
                before.wcoj_clique8_dispatch_count,
            )?,
            provider_wcoj_triangle_hg_dispatch_count: Self::checked_counter_delta(
                "provider_wcoj_triangle_hg_dispatch_count",
                self.provider_wcoj_triangle_hg_dispatch_count,
                before.provider_wcoj_triangle_hg_dispatch_count,
            )?,
            wcoj_layout_sort_invocation_count: Self::checked_counter_delta(
                "wcoj_layout_sort_invocation_count",
                self.wcoj_layout_sort_invocation_count,
                before.wcoj_layout_sort_invocation_count,
            )?,
            wcoj_layout_fast_path_hit_count: Self::checked_counter_delta(
                "wcoj_layout_fast_path_hit_count",
                self.wcoj_layout_fast_path_hit_count,
                before.wcoj_layout_fast_path_hit_count,
            )?,
            kclique_metadata_build_count: Self::checked_counter_delta(
                "kclique_metadata_build_count",
                self.kclique_metadata_build_count,
                before.kclique_metadata_build_count,
            )?,
            kclique_metadata_build_nanos: Self::checked_counter_delta(
                "kclique_metadata_build_nanos",
                self.kclique_metadata_build_nanos,
                before.kclique_metadata_build_nanos,
            )?,
            kclique_histogram_refresh_count: Self::checked_counter_delta(
                "kclique_histogram_refresh_count",
                self.kclique_histogram_refresh_count,
                before.kclique_histogram_refresh_count,
            )?,
            kclique_histogram_refresh_nanos: Self::checked_counter_delta_u128(
                "kclique_histogram_refresh_nanos",
                self.kclique_histogram_refresh_nanos,
                before.kclique_histogram_refresh_nanos,
            )?,
        })
    }

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
            kclique_histogram_refresh_count: self
                .kclique_histogram_refresh_count
                .saturating_sub(before.kclique_histogram_refresh_count),
            kclique_histogram_refresh_nanos: self
                .kclique_histogram_refresh_nanos
                .saturating_sub(before.kclique_histogram_refresh_nanos),
        }
    }

    /// Total WCOJ dispatches installed by the executor.
    pub fn wcoj_dispatch_count(&self) -> u64 {
        self.wcoj_triangle_dispatch_count
            .saturating_add(self.wcoj_4cycle_dispatch_count)
            .saturating_add(self.wcoj_clique_dispatch_count())
    }

    /// Checked total WCOJ dispatches installed by the executor.
    pub fn checked_wcoj_dispatch_count(&self) -> Result<u64> {
        Self::checked_counter_sum(
            "wcoj_dispatch_count",
            &[
                self.wcoj_triangle_dispatch_count,
                self.wcoj_4cycle_dispatch_count,
                self.checked_wcoj_clique_dispatch_count()?,
            ],
        )
    }

    /// Total K-clique WCOJ dispatches installed by the executor.
    pub fn wcoj_clique_dispatch_count(&self) -> u64 {
        self.wcoj_clique5_dispatch_count
            .saturating_add(self.wcoj_clique6_dispatch_count)
            .saturating_add(self.wcoj_clique7_dispatch_count)
            .saturating_add(self.wcoj_clique8_dispatch_count)
    }

    /// Checked total K-clique WCOJ dispatches installed by the executor.
    pub fn checked_wcoj_clique_dispatch_count(&self) -> Result<u64> {
        Self::checked_counter_sum(
            "wcoj_clique_dispatch_count",
            &[
                self.wcoj_clique5_dispatch_count,
                self.wcoj_clique6_dispatch_count,
                self.wcoj_clique7_dispatch_count,
                self.wcoj_clique8_dispatch_count,
            ],
        )
    }
}

/// WCOJ certification status for an epistemic runtime dispatch attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicGpuRuntimeWcojCertification {
    /// The preflight did not require a WCOJ dispatch.
    NotRequired {
        /// Observed executor-installed WCOJ dispatches.
        observed_wcoj_dispatches: u64,
        /// Structured planned-hash routes that replaced WCOJ dispatch obligations.
        planned_hash_routes: usize,
        /// Planned-hash routes where complete planner costs predicted hash wins.
        planned_hash_planner_wins: usize,
        /// Planned-hash routes selected because complete WCOJ stats were unavailable.
        planned_hash_incomplete_stats: usize,
        /// Planned-hash routes carrying finite hash-vs-WCOJ cost evidence.
        planned_hash_cost_evidence: usize,
    },
    /// Runtime counters prove the required WCOJ dispatch happened.
    Certified {
        /// Observed executor-installed WCOJ dispatches.
        observed_wcoj_dispatches: u64,
        /// MultiWayJoin reductions certified by the observed WCOJ dispatches.
        certified_multiway_reductions: usize,
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
        /// Observed recursive K-clique histogram refresh boundaries.
        observed_histogram_refreshes: u64,
        /// Observed recursive K-clique histogram refresh accounting time.
        observed_histogram_refresh_nanos: u128,
    },
    /// The plan required sorted layouts, but no layout path executed.
    MissingRequiredWcojLayout {
        /// Sorted-layout requirements found during preflight.
        required_sorted_layouts: usize,
        /// Observed layout sort or fast-path events.
        observed_layout_events: u64,
    },
    /// The plan dispatched a K-clique WCOJ route, but metadata-build counters did not advance.
    MissingRequiredKcliqueMetadata {
        /// K-clique WCOJ plans found during preflight.
        required_kclique_plans: usize,
        /// Observed provider K-clique metadata builds.
        observed_metadata_builds: u64,
        /// Observed provider time spent building K-clique metadata.
        observed_metadata_build_nanos: u64,
    },
    /// The plan had WCOJ obligations, but counters did not advance.
    MissingRequiredWcojDispatch {
        /// MultiWayJoin reductions found during preflight after excluding planned hash routes.
        required_multiway_reductions: usize,
        /// K-clique WCOJ plans found during preflight.
        required_kclique_plans: usize,
        /// Observed executor-installed WCOJ dispatches.
        observed_wcoj_dispatches: u64,
        /// Observed executor-installed K-clique dispatches.
        observed_kclique_dispatches: u64,
    },
}

/// CUDA provider identity that produced an epistemic GPU execution result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicGpuProviderIdentity {
    /// CUDA device ordinal used by the executor.
    pub device_ordinal: usize,
    /// Stable address of the executor's CUDA device wrapper.
    pub device_ptr: usize,
    /// Stable address of the executor's GPU memory manager.
    pub memory_ptr: usize,
}

impl EpistemicGpuProviderIdentity {
    /// Capture the device and memory-manager identity for a CUDA provider.
    pub fn from_provider(provider: &xlog_cuda::CudaKernelProvider) -> Self {
        Self {
            device_ordinal: provider.device().ordinal(),
            device_ptr: Arc::as_ptr(provider.device()) as usize,
            memory_ptr: Arc::as_ptr(provider.memory()) as usize,
        }
    }
}

/// Output from executing the reduced production runtime plan for an epistemic program.
pub struct EpistemicGpuExecutionResult {
    /// CUDA provider identity that owns this result's device-resident buffers.
    pub provider_identity: EpistemicGpuProviderIdentity,
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
    /// World-view integrity-constraint validation trace captured after world-view validation.
    pub constraint_world_view_validation: EpistemicGpuConstraintWorldViewValidationTrace,
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
    /// Reduced integrity-constraint validation after production runtime dispatch.
    pub constraint_validation: EpistemicGpuConstraintValidationTrace,
    /// Device-derived semantic summary after world-view validation.
    pub semantic_trace: EpistemicGpuSemanticTrace,
    /// Tuple-membership bindings that were validated and executed for this result.
    pub tuple_membership_bindings: Vec<EpistemicTupleMembershipBinding>,
    /// Device-resident final query output buffer.
    ///
    /// For a single epistemic output head this is the only materialized relation.
    /// For a JOINT-SOLVED coalesced multi-head component this is the PRIMARY head's
    /// output (the last reduction's head); the remaining coupled heads, each
    /// materialized against the SAME accepted world view, are in
    /// [`Self::additional_head_outputs`].
    pub final_output: CudaBuffer,
    /// Additional coupled-head outputs for a JOINT-SOLVED multi-head component.
    ///
    /// Empty for single-head execution. Each entry is `(head_predicate, buffer)`
    /// for a distinct epistemic output head OTHER than the primary head, filtered
    /// against the shared accepted world view via that head's row-filter bindings.
    pub additional_head_outputs: Vec<(String, CudaBuffer)>,
    /// Device-resident final tuple evidence buffer before public projection.
    pub tuple_evidence_output: Option<CudaBuffer>,
    /// Output buffer returned by the reduced production execution plan.
    pub output: CudaBuffer,
    /// Runtime counter trace for the reduced production plan dispatch.
    pub trace: EpistemicGpuRuntimeTrace,
}

impl EpistemicGpuExecutionResult {
    /// Device-resident output used to derive concrete tuple-membership evidence.
    pub fn tuple_evidence_output(&self) -> &CudaBuffer {
        self.tuple_evidence_output
            .as_ref()
            .unwrap_or(&self.final_output)
    }

    /// Require that the retained runtime trace certifies the prepared execution.
    pub fn require_runtime_dispatch_certification(&self) -> Result<()> {
        if self.trace.preflight != self.prepared.preflight {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU runtime dispatch certification".to_string(),
                context: "runtime trace preflight does not match prepared execution preflight"
                    .to_string(),
            });
        }
        if self.prepared.workspace.layout != self.prepared.preflight.workspace_layout {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU runtime dispatch certification".to_string(),
                context: "prepared GPU workspace layout does not match preflight workspace layout"
                    .to_string(),
            });
        }
        self.prepared
            .workspace
            .require_buffer_lengths_match_layout("epistemic GPU runtime dispatch certification")?;
        if self.tuple_membership_bindings.len()
            != self.prepared.preflight.tuple_membership_binding_count
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU runtime dispatch certification".to_string(),
                context: format!(
                    "runtime tuple-membership bindings do not match prepared preflight, got {} \
                     bindings for preflight count {}",
                    self.tuple_membership_bindings.len(),
                    self.prepared.preflight.tuple_membership_binding_count
                ),
            });
        }
        if self.tuple_membership_bindings != self.prepared.tuple_membership_bindings {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU runtime dispatch certification".to_string(),
                context: "runtime tuple-membership bindings do not match prepared GPU execution"
                    .to_string(),
            });
        }
        self.model_membership
            .require_planned_tuple_key_column_reads(expected_tuple_key_column_reads(
                &self.prepared.tuple_membership_bindings,
            )?)?;
        self.prepared.workspace_reset.require_matches_layout(
            "epistemic GPU runtime dispatch certification",
            self.prepared.preflight.workspace_layout,
        )?;
        self.final_result_transfer.require_matches_final_output(
            "epistemic GPU runtime dispatch certification",
            &self.final_output,
        )?;
        self.constraint_validation.require_matches_preflight(
            "epistemic GPU runtime dispatch certification",
            &self.prepared.preflight,
        )?;
        self.candidate_validation
            .require_matches_candidate_generation(
                "epistemic GPU runtime dispatch certification",
                &self.candidate_generation,
            )?;
        self.semantic_trace.require_matches_execution_traces(
            "epistemic GPU runtime dispatch certification",
            &self.candidate_generation,
            &self.propagation,
            &self.model_membership,
            &self.world_view_validation,
        )?;
        self.semantic_trace.require_rejection_metadata_accounting(
            "epistemic GPU runtime dispatch certification",
        )?;
        self.semantic_trace
            .require_candidate_index_partition("epistemic GPU runtime dispatch certification")?;
        let aggregate_kernel_timing = self.try_aggregate_kernel_timing()?;
        if !aggregate_kernel_timing.is_recorded() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU runtime dispatch certification".to_string(),
                context: "accepted GPU execution did not record CUDA-event timing".to_string(),
            });
        }
        self.trace.require_wcoj_certification()
    }

    /// Aggregate CUDA-event timing from all epistemic GPU hot-path kernels.
    pub fn aggregate_kernel_timing(&self) -> EpistemicGpuKernelTimingTrace {
        self.try_aggregate_kernel_timing()
            .expect("epistemic GPU kernel timing aggregation overflowed")
    }

    /// Checked CUDA-event timing aggregation for certification paths.
    pub fn try_aggregate_kernel_timing(&self) -> Result<EpistemicGpuKernelTimingTrace> {
        let traces = [
            self.candidate_generation.kernel_timing,
            self.propagation.kernel_timing,
            self.candidate_validation.kernel_timing,
            self.model_membership.kernel_timing,
            self.world_view_validation.kernel_timing,
            self.materialization.kernel_timing,
            self.final_result_materialization.kernel_timing,
            self.final_tuple_materialization.kernel_timing,
        ];

        if traces
            .iter()
            .all(EpistemicGpuKernelTimingTrace::is_recorded)
        {
            EpistemicGpuKernelTimingTrace::checked_sum(traces)
        } else {
            Ok(EpistemicGpuKernelTimingTrace::unrecorded())
        }
    }
}

/// Batch-level trace proving split components reused the single-plan GPU path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EpistemicGpuBatchExecutionTrace {
    /// Number of executable components requested by the batch.
    pub component_count: usize,
    /// Number of components executed through `execute_epistemic_gpu_execution`.
    pub gpu_runtime_component_executions: usize,
    /// CPU recomposition steps performed by this batch adapter.
    pub cpu_recomposition_steps: u64,
    /// CPU candidate enumerations observed across component semantic traces.
    pub cpu_candidate_enumerations: u64,
    /// CPU world-view validations observed across component semantic traces.
    pub cpu_world_view_validations: u64,
    /// CPU solver-search fallbacks observed across component preflight traces.
    pub cpu_solver_search_fallbacks: u64,
    /// CPU probability recomputations observed across component preflight traces.
    pub cpu_probability_recomputations: u64,
    /// Hot-path D2H calls tracked across all components.
    pub tracked_dtoh_calls: u64,
    /// Hot-path data-plane H2D calls tracked across all components.
    pub tracked_htod_calls: u64,
    /// Hot-path aggregate H2D calls tracked across all components.
    pub tracked_aggregate_htod_calls: u64,
    /// Hot-path launch-metadata H2D calls tracked across all components.
    pub tracked_launch_metadata_htod_calls: u64,
    /// Hot-path data-plane H2D calls tracked across all components.
    pub tracked_data_plane_htod_calls: u64,
    /// Per-candidate host round trips tracked across all components.
    pub per_candidate_host_round_trips: u64,
    /// Final output rows represented across all component device buffers.
    pub final_output_rows: usize,
    /// Final output payload bytes represented across all component device buffers.
    pub final_output_payload_bytes: u64,
    /// Device row-count metadata reads used for component final-result accounting.
    pub final_result_row_count_device_reads: u32,
    /// Post-hot-path final-result data-plane D2H calls across all components.
    pub final_result_data_plane_dtoh_calls: u64,
    /// Post-hot-path final-result data-plane D2H bytes across all components.
    pub final_result_data_plane_dtoh_bytes: u64,
    /// Reduced integrity-constraint relations checked across all components.
    pub checked_constraint_relations: usize,
    /// Reduced integrity-constraint relations with violating rows across all components.
    pub violated_constraint_relations: usize,
    /// Constraint row-count metadata reads used across all components.
    pub constraint_row_count_device_reads: u32,
    /// Accepted world views observed across component semantic traces.
    pub accepted_world_views: usize,
    /// Rejected candidates observed across component semantic traces.
    pub rejected_candidates: usize,
    /// Non-negated `know` operators observed across component preflight traces.
    pub know_operator_count: usize,
    /// Non-negated `possible` operators observed across component preflight traces.
    pub possible_operator_count: usize,
    /// Negated `know` operators observed as `not know` across component preflight traces.
    pub not_know_operator_count: usize,
    /// Negated `possible` operators observed as `not possible` across component preflight traces.
    pub not_possible_operator_count: usize,
    /// Aggregate CUDA-event timing from all component hot-path kernels.
    pub aggregate_kernel_timing: EpistemicGpuKernelTimingTrace,
}

impl EpistemicGpuBatchExecutionTrace {
    /// Build an aggregate trace from completed component results.
    pub fn from_component_results(results: &[EpistemicGpuExecutionResult]) -> Self {
        Self::try_from_component_results(results)
            .expect("epistemic GPU batch trace aggregation overflowed")
    }

    /// Build an aggregate trace from completed component results and fail closed
    /// if any certification counter overflows.
    pub fn try_from_component_results(results: &[EpistemicGpuExecutionResult]) -> Result<Self> {
        let component_kernel_timings = results
            .iter()
            .map(EpistemicGpuExecutionResult::try_aggregate_kernel_timing)
            .collect::<Result<Vec<_>>>()?;
        let aggregate_kernel_timing = if component_kernel_timings
            .iter()
            .all(EpistemicGpuKernelTimingTrace::is_recorded)
        {
            EpistemicGpuKernelTimingTrace::checked_sum(component_kernel_timings)
        } else {
            Ok(EpistemicGpuKernelTimingTrace::unrecorded())
        };
        let aggregate_kernel_timing = aggregate_kernel_timing?;

        Ok(Self {
            component_count: results.len(),
            gpu_runtime_component_executions: results.len(),
            cpu_recomposition_steps: 0,
            cpu_candidate_enumerations: checked_batch_sum_u64(
                "cpu_candidate_enumerations",
                results
                    .iter()
                    .map(|result| u64::from(result.semantic_trace.cpu_candidate_enumerations)),
            )?,
            cpu_world_view_validations: checked_batch_sum_u64(
                "cpu_world_view_validations",
                results
                    .iter()
                    .map(|result| u64::from(result.semantic_trace.cpu_world_view_validations)),
            )?,
            cpu_solver_search_fallbacks: checked_batch_sum_u64(
                "cpu_solver_search_fallbacks",
                results
                    .iter()
                    .map(|result| result.prepared.preflight.cpu_fallbacks.solver_search),
            )?,
            cpu_probability_recomputations: checked_batch_sum_u64(
                "cpu_probability_recomputations",
                results.iter().map(|result| {
                    result
                        .prepared
                        .preflight
                        .cpu_fallbacks
                        .probabilistic_recompute
                }),
            )?,
            tracked_dtoh_calls: checked_batch_sum_u64(
                "tracked_dtoh_calls",
                results
                    .iter()
                    .map(|result| result.transfer_budget.tracked_dtoh_calls),
            )?,
            tracked_htod_calls: checked_batch_sum_u64(
                "tracked_htod_calls",
                results
                    .iter()
                    .map(|result| result.transfer_budget.tracked_htod_calls),
            )?,
            tracked_aggregate_htod_calls: checked_batch_sum_u64(
                "tracked_aggregate_htod_calls",
                results
                    .iter()
                    .map(|result| result.transfer_budget.tracked_aggregate_htod_calls),
            )?,
            tracked_launch_metadata_htod_calls: checked_batch_sum_u64(
                "tracked_launch_metadata_htod_calls",
                results
                    .iter()
                    .map(|result| result.transfer_budget.tracked_launch_metadata_htod_calls),
            )?,
            tracked_data_plane_htod_calls: checked_batch_sum_u64(
                "tracked_data_plane_htod_calls",
                results
                    .iter()
                    .map(|result| result.transfer_budget.tracked_data_plane_htod_calls),
            )?,
            per_candidate_host_round_trips: checked_batch_sum_u64(
                "per_candidate_host_round_trips",
                results
                    .iter()
                    .map(|result| result.transfer_budget.per_candidate_host_round_trips),
            )?,
            final_output_rows: checked_batch_sum_usize(
                "final_output_rows",
                results
                    .iter()
                    .map(|result| result.final_result_transfer.final_output_rows),
            )?,
            final_output_payload_bytes: checked_batch_sum_u64(
                "final_output_payload_bytes",
                results
                    .iter()
                    .map(|result| result.final_result_transfer.final_output_payload_bytes),
            )?,
            final_result_row_count_device_reads: checked_batch_sum_u32(
                "final_result_row_count_device_reads",
                results
                    .iter()
                    .map(|result| result.final_result_transfer.row_count_device_reads),
            )?,
            final_result_data_plane_dtoh_calls: checked_batch_sum_u64(
                "final_result_data_plane_dtoh_calls",
                results
                    .iter()
                    .map(|result| result.final_result_transfer.tracked_data_plane_dtoh_calls),
            )?,
            final_result_data_plane_dtoh_bytes: checked_batch_sum_u64(
                "final_result_data_plane_dtoh_bytes",
                results
                    .iter()
                    .map(|result| result.final_result_transfer.tracked_data_plane_dtoh_bytes),
            )?,
            checked_constraint_relations: checked_batch_sum_usize(
                "checked_constraint_relations",
                results
                    .iter()
                    .map(|result| result.constraint_validation.checked_constraint_relations),
            )?,
            violated_constraint_relations: checked_batch_sum_usize(
                "violated_constraint_relations",
                results
                    .iter()
                    .map(|result| result.constraint_validation.violated_constraint_relations),
            )?,
            constraint_row_count_device_reads: checked_batch_sum_u32(
                "constraint_row_count_device_reads",
                results
                    .iter()
                    .map(|result| result.constraint_validation.row_count_device_reads),
            )?,
            accepted_world_views: checked_batch_sum_usize(
                "accepted_world_views",
                results
                    .iter()
                    .map(|result| result.semantic_trace.accepted_world_views),
            )?,
            rejected_candidates: checked_batch_sum_usize(
                "rejected_candidates",
                results
                    .iter()
                    .map(|result| result.semantic_trace.rejected_candidates),
            )?,
            know_operator_count: checked_batch_sum_usize(
                "know_operator_count",
                results
                    .iter()
                    .map(|result| result.prepared.preflight.know_operator_count),
            )?,
            possible_operator_count: checked_batch_sum_usize(
                "possible_operator_count",
                results
                    .iter()
                    .map(|result| result.prepared.preflight.possible_operator_count),
            )?,
            not_know_operator_count: checked_batch_sum_usize(
                "not_know_operator_count",
                results
                    .iter()
                    .map(|result| result.prepared.preflight.not_know_operator_count),
            )?,
            not_possible_operator_count: checked_batch_sum_usize(
                "not_possible_operator_count",
                results
                    .iter()
                    .map(|result| result.prepared.preflight.not_possible_operator_count),
            )?,
            aggregate_kernel_timing,
        })
    }
}

fn checked_batch_sum_u64(
    counter: &'static str,
    values: impl IntoIterator<Item = u64>,
) -> Result<u64> {
    values.into_iter().try_fold(0u64, |acc, value| {
        acc.checked_add(value)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU batch execution trace".to_string(),
                context: format!(
                    "batch counter {counter} overflowed while aggregating component traces: \
                     acc={acc} next={value}"
                ),
            })
    })
}

fn checked_batch_sum_u32(
    counter: &'static str,
    values: impl IntoIterator<Item = u32>,
) -> Result<u32> {
    values.into_iter().try_fold(0u32, |acc, value| {
        acc.checked_add(value)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU batch execution trace".to_string(),
                context: format!(
                    "batch counter {counter} overflowed while aggregating component traces: \
                     acc={acc} next={value}"
                ),
            })
    })
}

fn checked_batch_sum_usize(
    counter: &'static str,
    values: impl IntoIterator<Item = usize>,
) -> Result<usize> {
    values.into_iter().try_fold(0usize, |acc, value| {
        acc.checked_add(value)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU batch execution trace".to_string(),
                context: format!(
                    "batch counter {counter} overflowed while aggregating component traces: \
                     acc={acc} next={value}"
                ),
            })
    })
}

/// Results plus aggregate trace from a split/batch epistemic GPU execution.
pub struct EpistemicGpuBatchExecutionResult {
    /// Per-component execution results from the existing single-plan GPU path.
    pub results: Vec<EpistemicGpuExecutionResult>,
    /// Aggregate batch certification trace.
    pub trace: EpistemicGpuBatchExecutionTrace,
}

impl EpistemicGpuBatchExecutionResult {
    /// Require the retained aggregate trace to be derived from the component results.
    pub fn require_trace_matches_components(&self, construct: &str) -> Result<()> {
        if self.results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: "batch evidence requires at least one GPU component".to_string(),
            });
        }
        let expected = EpistemicGpuBatchExecutionTrace::try_from_component_results(&self.results)?;
        if self.trace != expected {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "batch aggregate trace does not match component GPU execution results: \
                     trace_components={}/{} expected_components={}/{} \
                     trace_final_rows={} expected_final_rows={} trace_dtoh_calls={} \
                     expected_dtoh_calls={} trace_data_plane_htod_calls={} \
                     expected_data_plane_htod_calls={} trace_constraint_violations={} \
                     expected_constraint_violations={} trace_accepted_world_views={} \
                     expected_accepted_world_views={}",
                    self.trace.gpu_runtime_component_executions,
                    self.trace.component_count,
                    expected.gpu_runtime_component_executions,
                    expected.component_count,
                    self.trace.final_output_rows,
                    expected.final_output_rows,
                    self.trace.tracked_dtoh_calls,
                    expected.tracked_dtoh_calls,
                    self.trace.tracked_data_plane_htod_calls,
                    expected.tracked_data_plane_htod_calls,
                    self.trace.violated_constraint_relations,
                    expected.violated_constraint_relations,
                    self.trace.accepted_world_views,
                    expected.accepted_world_views
                ),
            });
        }
        if !self.trace.aggregate_kernel_timing.is_recorded() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: "batch GPU execution did not record aggregate CUDA-event timing"
                    .to_string(),
            });
        }
        Ok(())
    }
}

impl EpistemicGpuRuntimeWcojCertification {
    /// Compare static preflight obligations with runtime counter deltas.
    pub fn for_preflight_and_delta(
        preflight: &EpistemicGpuRuntimePreflight,
        delta: &EpistemicGpuRuntimeCounters,
    ) -> Self {
        Self::try_for_preflight_and_delta(preflight, delta)
            .expect("runtime WCOJ certification counters must not overflow")
    }

    /// Compare static preflight obligations with runtime counter deltas, failing closed
    /// if certification counters overflow while being summarized.
    pub fn try_for_preflight_and_delta(
        preflight: &EpistemicGpuRuntimePreflight,
        delta: &EpistemicGpuRuntimeCounters,
    ) -> Result<Self> {
        let observed_wcoj_dispatches = delta.checked_wcoj_dispatch_count()?;
        let observed_kclique_dispatches = delta.checked_wcoj_clique_dispatch_count()?;
        let wcoj_routed_reduction_count = preflight
            .multiway_reduction_count
            .checked_sub(preflight.planned_hash_route_count)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU WCOJ route certification".to_string(),
                context: format!(
                    "planned hash routes exceed observed route obligations: \
                     multiway_reductions={} planned_hash_routes={}",
                    preflight.multiway_reduction_count, preflight.planned_hash_route_count
                ),
            })?;
        let required_multiway_reductions = wcoj_routed_reduction_count;

        if required_multiway_reductions == 0 {
            return Ok(Self::NotRequired {
                observed_wcoj_dispatches,
                planned_hash_routes: preflight.planned_hash_route_count,
                planned_hash_planner_wins: preflight.planned_hash_planner_wins_count,
                planned_hash_incomplete_stats: preflight.planned_hash_incomplete_stats_count,
                planned_hash_cost_evidence: preflight.planned_hash_cost_evidence_count,
            });
        }

        if observed_wcoj_dispatches < required_multiway_reductions as u64
            || observed_kclique_dispatches < preflight.kclique_wcoj_plan_count as u64
            || delta.wcoj_triangle_dispatch_count < preflight.wcoj_triangle_route_count as u64
            || delta.wcoj_4cycle_dispatch_count < preflight.wcoj_4cycle_route_count as u64
            || delta.wcoj_clique5_dispatch_count
                < preflight.kclique_wcoj_plan_count_by_arity[0] as u64
            || delta.wcoj_clique6_dispatch_count
                < preflight.kclique_wcoj_plan_count_by_arity[1] as u64
            || delta.wcoj_clique7_dispatch_count
                < preflight.kclique_wcoj_plan_count_by_arity[2] as u64
            || delta.wcoj_clique8_dispatch_count
                < preflight.kclique_wcoj_plan_count_by_arity[3] as u64
        {
            return Ok(Self::MissingRequiredWcojDispatch {
                required_multiway_reductions,
                required_kclique_plans: preflight.kclique_wcoj_plan_count,
                observed_wcoj_dispatches,
                observed_kclique_dispatches,
            });
        }

        let observed_layout_events = EpistemicGpuRuntimeCounters::checked_counter_sum(
            "wcoj_layout_events",
            &[
                delta.wcoj_layout_sort_invocation_count,
                delta.wcoj_layout_fast_path_hit_count,
            ],
        )?;
        if observed_layout_events < preflight.sorted_layout_requirement_count as u64 {
            return Ok(Self::MissingRequiredWcojLayout {
                required_sorted_layouts: preflight.sorted_layout_requirement_count,
                observed_layout_events,
            });
        }

        if preflight.kclique_wcoj_plan_count > 0
            && (delta.kclique_metadata_build_count < preflight.kclique_wcoj_plan_count as u64
                || delta.kclique_metadata_build_nanos == 0)
        {
            return Ok(Self::MissingRequiredKcliqueMetadata {
                required_kclique_plans: preflight.kclique_wcoj_plan_count,
                observed_metadata_builds: delta.kclique_metadata_build_count,
                observed_metadata_build_nanos: delta.kclique_metadata_build_nanos,
            });
        }

        Ok(Self::Certified {
            observed_wcoj_dispatches,
            certified_multiway_reductions: required_multiway_reductions,
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
            observed_histogram_refreshes: delta.kclique_histogram_refresh_count,
            observed_histogram_refresh_nanos: delta.kclique_histogram_refresh_nanos,
        })
    }
}

#[allow(clippy::large_enum_variant)]
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
/// Anonymous wildcard tuple-key position: matches any stable-model value.
const TUPLE_KEY_MATCH_MODE_WILDCARD: u8 = 2;

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
            (
                EirTerm::Anonymous
                | EirTerm::List(_)
                | EirTerm::Cons { .. }
                | EirTerm::Compound { .. }
                | EirTerm::PredRef(_)
                | EirTerm::Aggregate { .. },
                _,
            ) => {
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
            kclique_histogram_refresh_count: self.kclique_histogram_refresh_count(),
            kclique_histogram_refresh_nanos: self.kclique_histogram_refresh_nanos(),
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
            constraint_violation_index: memory.alloc::<u32>(layout.rejection_reason_slots)?,
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

        EpistemicGpuWorkspaceResetTrace::try_for_layout(workspace.layout)
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

        let literal_count =
            checked_u32_dimension(literal_count, "epistemic GPU candidate generation literals")?;
        let candidate_count = checked_u32_dimension(
            candidate_count,
            "epistemic GPU candidate generation candidates",
        )?;
        let total = checked_u32_dimension(
            trace.candidate_assumption_bytes,
            "epistemic GPU candidate generation launch elements",
        )?;
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
        let mut trace = EpistemicGpuPropagationTrace::for_counts(literal_count, candidate_count)?;
        let candidate_assumption_bytes = checked_product(literal_count, candidate_count)?;
        if candidate_assumption_bytes > workspace.layout.candidate_assumption_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU propagation candidate workspace".to_string(),
                estimated_bytes: candidate_assumption_bytes as u64,
                budget_bytes: workspace.layout.candidate_assumption_bytes as u64,
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
        let world_view_bitset_bytes_per_candidate =
            world_view_bitset_bytes_per_candidate(literal_count)?;
        if world_view_bitset_bytes_per_candidate > world_stride {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU propagation world-view bitset stride".to_string(),
                estimated_bytes: world_view_bitset_bytes_per_candidate as u64,
                budget_bytes: world_stride as u64,
            });
        }
        let world_view_bitset_bytes =
            checked_product(world_view_bitset_bytes_per_candidate, candidate_count)?;
        if world_view_bitset_bytes > workspace.layout.world_view_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU propagation world-view bitsets".to_string(),
                estimated_bytes: world_view_bitset_bytes as u64,
                budget_bytes: workspace.layout.world_view_bytes as u64,
            });
        }
        trace.world_view_bytes_written = checked_product(world_stride, candidate_count)?;
        if trace.world_view_bytes_written > workspace.layout.world_view_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU propagation world-view workspace".to_string(),
                estimated_bytes: trace.world_view_bytes_written as u64,
                budget_bytes: workspace.layout.world_view_bytes as u64,
            });
        }

        let literal_count =
            checked_u32_dimension(literal_count, "epistemic GPU propagation literals")?;
        let candidate_count =
            checked_u32_dimension(candidate_count, "epistemic GPU propagation candidates")?;
        let world_stride =
            checked_u32_dimension(world_stride, "epistemic GPU propagation world stride")?;
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
        let mut trace =
            EpistemicGpuCandidateValidationTrace::for_counts(literal_count, candidate_count)?;
        if trace.candidate_assumption_bytes_checked > workspace.layout.candidate_assumption_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU validation candidate workspace".to_string(),
                estimated_bytes: trace.candidate_assumption_bytes_checked as u64,
                budget_bytes: workspace.layout.candidate_assumption_bytes as u64,
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
        let world_view_bitset_bytes_per_candidate =
            world_view_bitset_bytes_per_candidate(literal_count)?;
        if world_view_bitset_bytes_per_candidate > world_stride {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU validation world-view bitset stride".to_string(),
                estimated_bytes: world_view_bitset_bytes_per_candidate as u64,
                budget_bytes: world_stride as u64,
            });
        }
        let world_view_bitset_bytes =
            checked_product(world_view_bitset_bytes_per_candidate, candidate_count)?;
        if world_view_bitset_bytes > workspace.layout.world_view_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU validation world-view bitsets".to_string(),
                estimated_bytes: world_view_bitset_bytes as u64,
                budget_bytes: workspace.layout.world_view_bytes as u64,
            });
        }
        trace.world_view_bytes_checked = world_view_bitset_bytes;
        if trace.world_view_bytes_checked > workspace.layout.world_view_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU validation world-view workspace".to_string(),
                estimated_bytes: trace.world_view_bytes_checked as u64,
                budget_bytes: workspace.layout.world_view_bytes as u64,
            });
        }

        let literal_count =
            checked_u32_dimension(literal_count, "epistemic GPU validation literals")?;
        let candidate_count =
            checked_u32_dimension(candidate_count, "epistemic GPU validation candidates")?;
        let world_stride =
            checked_u32_dimension(world_stride, "epistemic GPU validation world stride")?;
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
        let mut trace =
            EpistemicGpuModelMembershipTrace::for_stable_model_tuple_sources_with_key_columns(
                literal_count,
                candidate_count,
                reduction_count,
                models_per_reduction,
                gpu_plan.tuple_membership_bindings.len(),
                tuple_source_key_column_count,
            )?;
        trace.output_row_count_device_reads = trace.kernel_launches;
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
            // Anonymous wildcards are value-level matches handled only by the
            // general arm; route any binding carrying a variable or an anonymous
            // term there. The specialized arity arms remain a fast path for
            // all-ground tuple keys.
            let has_value_level_keys = binding
                .key_terms
                .iter()
                .any(|term| matches!(term, EirTerm::Variable(_) | EirTerm::Anonymous));
            match binding.key_columns.as_slice() {
                [] => tuple_sources.push(TupleSourceLaunch::ArityZero {
                    literal_index: binding.literal_index as u32,
                    reduction_index: binding.reduction_index as u32,
                    negated: binding.negated as u8,
                    row_count: source_relation.num_rows_device(),
                }),
                &[key_col] if !has_value_level_keys => {
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
                &[key_col0, key_col1] if !has_value_level_keys => {
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
                &[key_col0, key_col1, key_col2] if !has_value_level_keys => {
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

                        key_col_ptrs_host.push(*key_col_ref.device_ptr());
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
                                bound_value_col_ptrs_host.push(*bound_col.device_ptr());
                                bound_value_col_widths_host.push(bound_col_width as u32);
                            }
                            EirTerm::Anonymous => {
                                // Wildcard: no equality requirement on this
                                // tuple-key column. The device still reads the
                                // column pointer/width, but the kernel matches
                                // every stable-model value in this position.
                                expected_key_bits_host.push(0);
                                expected_key_type_codes_host.push(key_col_type.to_code());
                                tuple_key_match_modes_host.push(TUPLE_KEY_MATCH_MODE_WILDCARD);
                                bound_value_col_ptrs_host.push(0);
                                bound_value_col_widths_host.push(0);
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
                    let mut key_col_ptrs = memory.alloc::<u64>(key_columns.len())?;
                    let mut key_col_widths = memory.alloc::<u32>(key_columns.len())?;
                    let mut expected_key_bits = memory.alloc::<u64>(key_columns.len())?;
                    let mut expected_key_type_codes = memory.alloc::<u8>(key_columns.len())?;
                    let mut tuple_key_match_modes = memory.alloc::<u8>(key_columns.len())?;
                    let mut bound_value_col_ptrs = memory.alloc::<u64>(key_columns.len())?;
                    let mut bound_value_col_widths = memory.alloc::<u32>(key_columns.len())?;
                    self.provider
                        .htod_launch_metadata_sync_copy_into(&key_col_ptrs_host, &mut key_col_ptrs)
                        .map_err(|e| {
                            XlogError::execution_ctx(
                                "epistemic GPU tuple-key metadata",
                                "upload key column pointers",
                                &e,
                            )
                        })?;
                    self.provider
                        .htod_launch_metadata_sync_copy_into(
                            &key_col_widths_host,
                            &mut key_col_widths,
                        )
                        .map_err(|e| {
                            XlogError::execution_ctx(
                                "epistemic GPU tuple-key metadata",
                                "upload key column widths",
                                &e,
                            )
                        })?;
                    self.provider
                        .htod_launch_metadata_sync_copy_into(
                            &expected_key_bits_host,
                            &mut expected_key_bits,
                        )
                        .map_err(|e| {
                            XlogError::execution_ctx(
                                "epistemic GPU tuple-key metadata",
                                "upload expected key bits",
                                &e,
                            )
                        })?;
                    self.provider
                        .htod_launch_metadata_sync_copy_into(
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
                    self.provider
                        .htod_launch_metadata_sync_copy_into(
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
                    self.provider
                        .htod_launch_metadata_sync_copy_into(
                            &bound_value_col_ptrs_host,
                            &mut bound_value_col_ptrs,
                        )
                        .map_err(|e| {
                            XlogError::execution_ctx(
                                "epistemic GPU tuple-key metadata",
                                "upload bound value column pointers",
                                &e,
                            )
                        })?;
                    self.provider
                        .htod_launch_metadata_sync_copy_into(
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

        let mut kernel_timings = Vec::with_capacity(tuple_sources.len());
        for tuple_source in &tuple_sources {
            let kernel_timing = self.time_epistemic_gpu_kernel_launch(
                "epistemic GPU tuple-source model membership",
                || unsafe {
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
                                literal_count.as_kernel_param(),
                                candidate_count.as_kernel_param(),
                                reduction_count.as_kernel_param(),
                                models_per_reduction.as_kernel_param(),
                                world_stride.as_kernel_param(),
                                literal_index.as_kernel_param(),
                                reduction_index.as_kernel_param(),
                                negated.as_kernel_param(),
                                output.num_rows_device().as_kernel_param(),
                                row_count.as_kernel_param(),
                                (&workspace.candidate_assumptions).as_kernel_param(),
                                (&workspace.world_views).as_kernel_param(),
                                (&workspace.model_membership).as_kernel_param(),
                                (&workspace.rejection_reasons).as_kernel_param(),
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
                                literal_count.as_kernel_param(),
                                candidate_count.as_kernel_param(),
                                reduction_count.as_kernel_param(),
                                models_per_reduction.as_kernel_param(),
                                world_stride.as_kernel_param(),
                                literal_index.as_kernel_param(),
                                reduction_index.as_kernel_param(),
                                negated.as_kernel_param(),
                                output.num_rows_device().as_kernel_param(),
                                row_count.as_kernel_param(),
                                key_col0.as_kernel_param(),
                                key_col0_width.as_kernel_param(),
                                expected_key_col0_bits.as_kernel_param(),
                                expected_key_col0_type_code.as_kernel_param(),
                                (&workspace.candidate_assumptions).as_kernel_param(),
                                (&workspace.world_views).as_kernel_param(),
                                (&workspace.model_membership).as_kernel_param(),
                                (&workspace.rejection_reasons).as_kernel_param(),
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
                                literal_count.as_kernel_param(),
                                candidate_count.as_kernel_param(),
                                reduction_count.as_kernel_param(),
                                models_per_reduction.as_kernel_param(),
                                world_stride.as_kernel_param(),
                                literal_index.as_kernel_param(),
                                reduction_index.as_kernel_param(),
                                negated.as_kernel_param(),
                                output.num_rows_device().as_kernel_param(),
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
                                (&workspace.model_membership).as_kernel_param(),
                                (&workspace.rejection_reasons).as_kernel_param(),
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
                                literal_count.as_kernel_param(),
                                candidate_count.as_kernel_param(),
                                reduction_count.as_kernel_param(),
                                models_per_reduction.as_kernel_param(),
                                world_stride.as_kernel_param(),
                                literal_index.as_kernel_param(),
                                reduction_index.as_kernel_param(),
                                negated.as_kernel_param(),
                                output.num_rows_device().as_kernel_param(),
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
                                (&workspace.model_membership).as_kernel_param(),
                                (&workspace.rejection_reasons).as_kernel_param(),
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
                                literal_count.as_kernel_param(),
                                candidate_count.as_kernel_param(),
                                reduction_count.as_kernel_param(),
                                models_per_reduction.as_kernel_param(),
                                world_stride.as_kernel_param(),
                                literal_index.as_kernel_param(),
                                reduction_index.as_kernel_param(),
                                negated.as_kernel_param(),
                                output.num_rows_device().as_kernel_param(),
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
                                (&workspace.model_membership).as_kernel_param(),
                                (&workspace.rejection_reasons).as_kernel_param(),
                            ];
                            func_arity_n.clone().launch(config, &mut params)?;
                        }
                    };
                    Ok(())
                },
            )?;
            kernel_timings.push(kernel_timing);
        }
        let kernel_timing = EpistemicGpuKernelTimingTrace::checked_sum(kernel_timings)?;

        Ok(trace.with_kernel_timing(kernel_timing))
    }

    /// Validate staged model memberships against candidate world views on device.
    pub fn validate_epistemic_gpu_world_views(
        &self,
        workspace: &mut EpistemicGpuWorkspace,
        gpu_plan: &EpistemicGpuPlan,
        candidate_count: usize,
        models_per_reduction: usize,
    ) -> Result<EpistemicGpuWorldViewValidationTrace> {
        gpu_plan.validate_tuple_membership_bindings()?;
        let literal_count = gpu_plan.epistemic_literals.len();
        let reduction_count = gpu_plan.reductions.len();
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

        let mut literal_op_codes_host = vec![0u8; literal_count];
        let mut literal_negated_host = vec![0u8; literal_count];
        let mut literal_bound_to_output_host = vec![0u8; literal_count];
        let mut literal_reduction_indices_host = vec![0u32; literal_count];
        for binding in &gpu_plan.tuple_membership_bindings {
            literal_op_codes_host[binding.literal_index] = epistemic_operator_code(binding.op);
            literal_negated_host[binding.literal_index] = u8::from(binding.negated);
            literal_bound_to_output_host[binding.literal_index] =
                u8::from(binding.bound_output_columns.iter().any(Option::is_some));
            literal_reduction_indices_host[binding.literal_index] = binding.reduction_index as u32;
        }
        let memory = self.provider.memory();
        let mut literal_op_codes = memory.alloc::<u8>(literal_count)?;
        let mut literal_negated = memory.alloc::<u8>(literal_count)?;
        let mut literal_bound_to_output = memory.alloc::<u8>(literal_count)?;
        let mut literal_reduction_indices = memory.alloc::<u32>(literal_count)?;
        self.provider
            .htod_launch_metadata_sync_copy_into(&literal_op_codes_host, &mut literal_op_codes)
            .map_err(|e| {
                XlogError::execution_ctx(
                    "epistemic GPU world-view validation metadata",
                    "upload literal operator codes",
                    &e,
                )
            })?;
        self.provider
            .htod_launch_metadata_sync_copy_into(&literal_negated_host, &mut literal_negated)
            .map_err(|e| {
                XlogError::execution_ctx(
                    "epistemic GPU world-view validation metadata",
                    "upload literal negation flags",
                    &e,
                )
            })?;
        self.provider
            .htod_launch_metadata_sync_copy_into(
                &literal_bound_to_output_host,
                &mut literal_bound_to_output,
            )
            .map_err(|e| {
                XlogError::execution_ctx(
                    "epistemic GPU world-view validation metadata",
                    "upload literal output-binding flags",
                    &e,
                )
            })?;
        self.provider
            .htod_launch_metadata_sync_copy_into(
                &literal_reduction_indices_host,
                &mut literal_reduction_indices,
            )
            .map_err(|e| {
                XlogError::execution_ctx(
                    "epistemic GPU world-view validation metadata",
                    "upload literal reduction indices",
                    &e,
                )
            })?;

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
                let mut params: Vec<*mut c_void> = vec![
                    literal_count.as_kernel_param(),
                    candidate_count.as_kernel_param(),
                    reduction_count.as_kernel_param(),
                    models_per_reduction.as_kernel_param(),
                    world_stride.as_kernel_param(),
                    (&literal_op_codes).as_kernel_param(),
                    (&literal_negated).as_kernel_param(),
                    (&literal_bound_to_output).as_kernel_param(),
                    (&literal_reduction_indices).as_kernel_param(),
                    (&workspace.candidate_assumptions).as_kernel_param(),
                    (&workspace.model_membership).as_kernel_param(),
                    (&workspace.world_views).as_kernel_param(),
                    (&workspace.rejection_reasons).as_kernel_param(),
                ];
                func.clone().launch(config, &mut params)
            },
        )?;

        Ok(trace.with_kernel_timing(kernel_timing))
    }

    /// Prune accepted candidate world views that satisfy an epistemic integrity
    /// constraint body.
    ///
    /// Runs after [`Self::validate_epistemic_gpu_world_views`]: each surviving
    /// candidate's assumption bit equals the negation-folded observed modal
    /// value of its literal, so a constraint body holds in this accepted world
    /// view exactly when every referenced literal's assumption bit is set. Such
    /// candidates are pruned on device with the world-view constraint-violation
    /// rejection code; no accepted world is read back to the host.
    pub fn validate_epistemic_gpu_world_view_constraints(
        &self,
        workspace: &mut EpistemicGpuWorkspace,
        gpu_plan: &EpistemicGpuPlan,
        candidate_count: usize,
    ) -> Result<EpistemicGpuConstraintWorldViewValidationTrace> {
        gpu_plan.validate_constraints()?;
        let literal_count = gpu_plan.epistemic_literals.len();
        let constraint_count = gpu_plan.constraints.len();

        // Initialize the parallel constraint-violation index buffer to the
        // sentinel `u32::MAX` ("not rejected by a constraint") for every
        // candidate, BEFORE the zero-constraint early return below. Zero is a
        // valid constraint index, so the buffer cannot be left zeroed: any
        // candidate rejected by reason codes 1-5 (or accepted) must read back as
        // the sentinel, never a spurious `Some(0)`. The upload rides the
        // launch-metadata channel (like the CSR buffers below), so it adds no
        // tracked data-plane HTOD and keeps `host_write_ops` at zero.
        if candidate_count > workspace.layout.rejection_reason_slots {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU constraint-violation index workspace".to_string(),
                estimated_bytes: candidate_count as u64,
                budget_bytes: workspace.layout.rejection_reason_slots as u64,
            });
        }
        if candidate_count > 0 {
            let sentinel_host = vec![u32::MAX; candidate_count];
            let fill_len = candidate_count;
            let mut sentinel_view = workspace.constraint_violation_index.slice_mut(0..fill_len);
            self.provider
                .htod_launch_metadata_sync_copy_into(&sentinel_host, &mut sentinel_view)
                .map_err(|e| {
                    XlogError::execution_ctx(
                        "epistemic GPU world-view constraint metadata",
                        "initialize constraint-violation index sentinel",
                        &e,
                    )
                })?;
        }

        // Flatten constraint -> literal index references into CSR-style buffers.
        let mut offsets_host = Vec::with_capacity(constraint_count);
        let mut counts_host = Vec::with_capacity(constraint_count);
        let mut indices_host: Vec<u32> = Vec::new();
        for constraint in &gpu_plan.constraints {
            offsets_host.push(indices_host.len() as u32);
            counts_host.push(constraint.literal_indices.len() as u32);
            for &literal_index in &constraint.literal_indices {
                indices_host.push(literal_index as u32);
            }
        }
        let constraint_literal_refs = indices_host.len();

        let trace = EpistemicGpuConstraintWorldViewValidationTrace {
            constraint_count,
            constraint_literal_refs,
            candidates_checked: candidate_count,
            rejection_reason_slots_written: candidate_count,
            kernel_launches: 0,
            host_write_ops: 0,
            kernel_timing: EpistemicGpuKernelTimingTrace::unrecorded(),
        };

        if constraint_count == 0 {
            // No world-view constraints to evaluate; leave the rejection buffer
            // untouched so accepted candidates flow through unchanged.
            return Ok(trace);
        }

        if candidate_count > workspace.layout.rejection_reason_slots {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU world-view constraint rejection workspace".to_string(),
                estimated_bytes: candidate_count as u64,
                budget_bytes: workspace.layout.rejection_reason_slots as u64,
            });
        }
        if candidate_count > u32::MAX as usize
            || literal_count > u32::MAX as usize
            || constraint_count > u32::MAX as usize
            || constraint_literal_refs > u32::MAX as usize
        {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU world-view constraint dimensions".to_string(),
                estimated_bytes: candidate_count
                    .max(literal_count)
                    .max(constraint_count)
                    .max(constraint_literal_refs) as u64,
                budget_bytes: u32::MAX as u64,
            });
        }

        let memory = self.provider.memory();
        let mut constraint_literal_offsets = memory.alloc::<u32>(constraint_count)?;
        let mut constraint_literal_counts = memory.alloc::<u32>(constraint_count)?;
        let mut constraint_literal_indices = memory.alloc::<u32>(constraint_literal_refs.max(1))?;
        self.provider
            .htod_launch_metadata_sync_copy_into(&offsets_host, &mut constraint_literal_offsets)
            .map_err(|e| {
                XlogError::execution_ctx(
                    "epistemic GPU world-view constraint metadata",
                    "upload constraint literal offsets",
                    &e,
                )
            })?;
        self.provider
            .htod_launch_metadata_sync_copy_into(&counts_host, &mut constraint_literal_counts)
            .map_err(|e| {
                XlogError::execution_ctx(
                    "epistemic GPU world-view constraint metadata",
                    "upload constraint literal counts",
                    &e,
                )
            })?;
        if !indices_host.is_empty() {
            self.provider
                .htod_launch_metadata_sync_copy_into(&indices_host, &mut constraint_literal_indices)
                .map_err(|e| {
                    XlogError::execution_ctx(
                        "epistemic GPU world-view constraint metadata",
                        "upload constraint literal indices",
                        &e,
                    )
                })?;
        }

        let literal_count_u32 = literal_count as u32;
        let candidate_count_u32 = candidate_count as u32;
        let constraint_count_u32 = constraint_count as u32;
        let func = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_VALIDATE_CONSTRAINTS_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution(
                    "epistemic world-view constraint validation kernel not found".to_string(),
                )
            })?;
        let config = LaunchConfig::for_num_elems(candidate_count_u32);

        let kernel_timing = self.time_epistemic_gpu_kernel_launch(
            "epistemic GPU world-view constraint validation",
            || unsafe {
                // SAFETY: kernel arguments match the PTX signature; the capacity
                // check above proves the rejection buffer covers every candidate,
                // and CSR offset/count/index buffers are sized to the constraint
                // literal references uploaded above.
                let mut params: Vec<*mut c_void> = vec![
                    literal_count_u32.as_kernel_param(),
                    candidate_count_u32.as_kernel_param(),
                    constraint_count_u32.as_kernel_param(),
                    (&constraint_literal_offsets).as_kernel_param(),
                    (&constraint_literal_counts).as_kernel_param(),
                    (&constraint_literal_indices).as_kernel_param(),
                    (&workspace.candidate_assumptions).as_kernel_param(),
                    (&mut workspace.rejection_reasons).as_kernel_param(),
                    (&mut workspace.constraint_violation_index).as_kernel_param(),
                ];
                func.clone().launch(config, &mut params)
            },
        )?;

        Ok(EpistemicGpuConstraintWorldViewValidationTrace {
            kernel_launches: 1,
            kernel_timing,
            ..trace
        })
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
    #[allow(clippy::too_many_arguments)]
    pub fn materialize_epistemic_gpu_final_tuples(
        &self,
        workspace: &mut EpistemicGpuWorkspace,
        output: &CudaBuffer,
        gpu_plan: &EpistemicGpuPlan,
        literal_count: usize,
        candidate_count: usize,
        reduction_count: usize,
        models_per_reduction: usize,
    ) -> Result<(CudaBuffer, EpistemicGpuFinalTupleMaterializationTrace)> {
        self.materialize_epistemic_gpu_final_tuples_scoped(
            workspace,
            output,
            gpu_plan,
            literal_count,
            candidate_count,
            reduction_count,
            models_per_reduction,
            None,
        )
    }

    /// Materialize final query tuples, optionally scoping the modal row-filter to a
    /// single coalesced head's reductions.
    ///
    /// `head_reduction_filter` is the JOINT-SOLVING multi-output seam: the joint
    /// candidate enumeration + world-view validation runs ONCE over the combined
    /// modal literals (so the accepted world view in `workspace` is shared by every
    /// head), then this method is called once per distinct head with that head's
    /// reduction indices. Only the row-filter bindings whose `reduction_index` is
    /// in the filter drive that head's output filtering; the full joint plan
    /// (`gpu_plan`) is still validated and the joint workspace dimensions are
    /// preserved, so each head is materialized against the SAME accepted world
    /// view. `None` materializes against every binding (the single-head path).
    #[allow(clippy::too_many_arguments)]
    fn materialize_epistemic_gpu_final_tuples_scoped(
        &self,
        workspace: &mut EpistemicGpuWorkspace,
        output: &CudaBuffer,
        gpu_plan: &EpistemicGpuPlan,
        literal_count: usize,
        candidate_count: usize,
        reduction_count: usize,
        models_per_reduction: usize,
        head_reduction_filter: Option<&BTreeSet<usize>>,
    ) -> Result<(CudaBuffer, EpistemicGpuFinalTupleMaterializationTrace)> {
        gpu_plan.validate_tuple_membership_bindings()?;
        if candidate_count > workspace.layout.rejection_reason_slots {
            return Err(XlogError::ResourceExhausted {
                context: "epistemic GPU final-tuple rejection workspace".to_string(),
                estimated_bytes: candidate_count as u64,
                budget_bytes: workspace.layout.rejection_reason_slots as u64,
            });
        }
        let literal_count_u32 =
            checked_u32_dimension(literal_count, "epistemic GPU final-tuple literals")?;
        let candidate_count_u32 =
            checked_u32_dimension(candidate_count, "epistemic GPU final-tuple candidates")?;
        let reduction_count_u32 =
            checked_u32_dimension(reduction_count, "epistemic GPU final-tuple reductions")?;
        let models_per_reduction_u32 = checked_u32_dimension(
            models_per_reduction,
            "epistemic GPU final-tuple models per reduction",
        )?;
        let output_row_capacity =
            usize::try_from(output.num_rows()).map_err(|_| XlogError::ResourceExhausted {
                context: "epistemic GPU final-tuple output rows".to_string(),
                estimated_bytes: output.num_rows(),
                budget_bytes: usize::MAX as u64,
            })?;
        let output_row_capacity_u32 =
            checked_u32_dimension(output_row_capacity, "epistemic GPU final-tuple output rows")?;
        let final_output_columns = final_output_columns_for_materialization(output, gpu_plan)?;
        let mut tuple_bytes_capacity = 0usize;
        let mut source_columns: Vec<(&CudaColumn, u32, u32)> =
            Vec::with_capacity(final_output_columns.len());
        let mut result_columns_raw: Vec<TrackedCudaSlice<u8>> =
            Vec::with_capacity(final_output_columns.len());
        let mut final_schema_columns = Vec::with_capacity(final_output_columns.len());
        let mut final_schema_sort_labels = Vec::with_capacity(final_output_columns.len());
        for &col_idx in &final_output_columns {
            let src_col = output.column(col_idx).ok_or_else(|| {
                XlogError::Execution(format!("epistemic final tuple missing column {col_idx}"))
            })?;
            let (column_name, column_type) = output
                .schema()
                .columns
                .get(col_idx)
                .ok_or_else(|| {
                    XlogError::Execution(format!(
                        "epistemic final tuple missing schema column {col_idx}"
                    ))
                })?
                .clone();
            let column_width = column_type.size_bytes();
            let expected_column_bytes = checked_product(output_row_capacity, column_width)?;
            if src_col.len() < expected_column_bytes {
                return Err(XlogError::ResourceExhausted {
                    context: "epistemic GPU final-tuple column capacity".to_string(),
                    estimated_bytes: expected_column_bytes as u64,
                    budget_bytes: src_col.len() as u64,
                });
            }
            let column_byte_len =
                checked_u32_dimension(src_col.len(), "epistemic GPU final-tuple column")?;
            let column_width =
                checked_u32_dimension(column_width, "epistemic GPU final-tuple column width")?;
            tuple_bytes_capacity = checked_sum(tuple_bytes_capacity, src_col.len())?;
            source_columns.push((src_col, column_byte_len, column_width));
            result_columns_raw.push(self.provider.memory().alloc::<u8>(src_col.len())?);
            final_schema_columns.push((column_name, column_type));
            final_schema_sort_labels.push(
                output
                    .schema()
                    .column_sort_label(col_idx)
                    .unwrap_or("")
                    .to_string(),
            );
        }

        let mut final_row_count = self.provider.memory().alloc::<u32>(1)?;
        let mut row_map = self
            .provider
            .memory()
            .alloc::<u32>(output_row_capacity.max(1))?;
        let row_filter_bindings: Vec<_> = gpu_plan
            .tuple_membership_bindings
            .iter()
            .filter(|binding| binding.bound_output_columns.iter().any(Option::is_some))
            .filter(|binding| {
                head_reduction_filter
                    .map(|reductions| reductions.contains(&binding.reduction_index))
                    .unwrap_or(true)
            })
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
            final_output_columns.len(),
            output_row_capacity,
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
        let world_stride =
            checked_u32_dimension(world_stride, "epistemic GPU final-tuple world stride")?;
        let mut metadata_len = 0usize;
        for binding in &row_filter_bindings {
            metadata_len = checked_sum(metadata_len, binding.key_columns.len())?;
        }
        let metadata_len = metadata_len.max(1);
        let row_filter_metadata_len = row_filter_bindings.len().max(1);
        checked_u32_dimension(
            metadata_len,
            "epistemic GPU final tuple row-filter key metadata",
        )?;
        checked_u32_dimension(
            row_filter_metadata_len,
            "epistemic GPU final tuple row-filter metadata",
        )?;
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
        let row_filter_count = checked_u32_dimension(
            row_filter_bindings.len(),
            "epistemic GPU final tuple row-filter count",
        )?;
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
                let row_filter_key_offset = checked_u32_dimension(
                    key_col_ptrs_host.len(),
                    "epistemic GPU final tuple row-filter key offset",
                )?;
                let row_filter_key_count = checked_u32_dimension(
                    binding.key_columns.len(),
                    "epistemic GPU final tuple row-filter key arity",
                )?;
                row_filter_key_offsets_host.push(row_filter_key_offset);
                row_filter_key_counts_host.push(row_filter_key_count);
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
                tuple_source_row_count_ptrs_host.push(*tuple_source_row_count.device_ptr());
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

                    key_col_ptrs_host.push(*key_col_ref.device_ptr());
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
                            bound_value_col_ptrs_host.push(*bound_col.device_ptr());
                            bound_value_col_widths_host.push(bound_col_width as u32);
                        }
                        EirTerm::Anonymous => {
                            // Wildcard: this tuple-key column imposes no
                            // equality requirement when filtering output rows.
                            expected_key_bits_host.push(0);
                            expected_key_type_codes_host.push(key_col_type.to_code());
                            tuple_key_match_modes_host.push(TUPLE_KEY_MATCH_MODE_WILDCARD);
                            bound_value_col_ptrs_host.push(0);
                            bound_value_col_widths_host.push(0);
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
            self.provider
                .htod_launch_metadata_sync_copy_into(
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
            self.provider
                .htod_launch_metadata_sync_copy_into(
                    &row_filter_negated_host,
                    &mut row_filter_negated,
                )
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload row-filter polarity", &e)
                })?;
            self.provider
                .htod_launch_metadata_sync_copy_into(
                    &row_filter_key_offsets_host,
                    &mut row_filter_key_offsets,
                )
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload row-filter key offsets", &e)
                })?;
            self.provider
                .htod_launch_metadata_sync_copy_into(
                    &row_filter_key_counts_host,
                    &mut row_filter_key_counts,
                )
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload row-filter key counts", &e)
                })?;
            self.provider
                .htod_launch_metadata_sync_copy_into(&key_col_ptrs_host, &mut key_col_ptrs)
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload key column pointers", &e)
                })?;
            self.provider
                .htod_launch_metadata_sync_copy_into(&key_col_widths_host, &mut key_col_widths)
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload key column widths", &e)
                })?;
            self.provider
                .htod_launch_metadata_sync_copy_into(
                    &expected_key_bits_host,
                    &mut expected_key_bits,
                )
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload expected key bits", &e)
                })?;
            self.provider
                .htod_launch_metadata_sync_copy_into(
                    &expected_key_type_codes_host,
                    &mut expected_key_type_codes,
                )
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload expected key type codes", &e)
                })?;
            self.provider
                .htod_launch_metadata_sync_copy_into(
                    &tuple_key_match_modes_host,
                    &mut tuple_key_match_modes,
                )
                .map_err(|e| {
                    XlogError::execution_ctx(metadata_context, "upload tuple key match modes", &e)
                })?;
            self.provider
                .htod_launch_metadata_sync_copy_into(
                    &bound_value_col_ptrs_host,
                    &mut bound_value_col_ptrs,
                )
                .map_err(|e| {
                    XlogError::execution_ctx(
                        metadata_context,
                        "upload bound value column pointers",
                        &e,
                    )
                })?;
            self.provider
                .htod_launch_metadata_sync_copy_into(
                    &bound_value_col_widths_host,
                    &mut bound_value_col_widths,
                )
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

        // Global-gate literal mask: a literal that does not bind any reduced
        // output column (pure-ground, pure-anonymous, or arity-0) is checked by
        // the global membership gate rather than a per-row filter. For those
        // literals the body literal must actually hold in the accepted
        // candidate's world view; per-row (bound-variable) literals are already
        // enforced by the row-filter loop above. The accepted candidate's
        // assumption bit already folds in `know`/`possible` modality and
        // negation (the validation kernel guarantees assumption == observed for
        // accepted candidates), so the gate requires the assumption bit to be
        // set for every global-gate literal.
        // Constraint literals participate in modal world-view evaluation (model
        // membership + assumption-bit pinning) but must NOT gate output rows:
        // their pruning is enforced by the separate world-view constraint kernel,
        // which rejects candidates whose accepted world view satisfies the
        // constraint body. Treating them as required gates would invert the
        // semantics (emit rows only when the forbidden body holds), so exclude
        // them from the output-gating mask.
        let mut is_constraint_literal = vec![false; literal_count.max(1)];
        for constraint in &gpu_plan.constraints {
            for &literal_index in &constraint.literal_indices {
                if literal_index < literal_count {
                    is_constraint_literal[literal_index] = true;
                }
            }
        }
        let mut gate_literal_required_host = vec![0u8; literal_count.max(1)];
        for binding in &gpu_plan.tuple_membership_bindings {
            if !binding.bound_output_columns.iter().any(Option::is_some)
                && binding.literal_index < literal_count
                && !is_constraint_literal[binding.literal_index]
            {
                gate_literal_required_host[binding.literal_index] = 1u8;
            }
        }
        // A rule mixing a per-row (bound-variable) modal literal with a global
        // gate (pure-ground/anonymous/arity-0) literal is materialized soundly:
        // the row-map kernel applies the global-gate `gate_literal_required`
        // mask on BOTH the global membership path and the per-row membership
        // path, so global-gate literals and per-row bound tuple-key gates
        // compose conjunctively. The two gate buffers below are passed to the
        // row-map kernel for both paths.
        let mut gate_literal_required = memory.alloc::<u8>(literal_count.max(1))?;
        self.provider
            .htod_launch_metadata_sync_copy_into(
                &gate_literal_required_host,
                &mut gate_literal_required,
            )
            .map_err(|e| {
                XlogError::execution_ctx(
                    "epistemic GPU final tuple gate metadata",
                    "upload global-gate literal mask",
                    &e,
                )
            })?;

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
        let close_rejections_func = self
            .provider
            .device()
            .inner()
            .get_func(
                EPISTEMIC_MODULE,
                epistemic_kernels::EPISTEMIC_CLOSE_FINAL_TUPLE_REJECTIONS_U8,
            )
            .ok_or_else(|| {
                XlogError::Execution(
                    "epistemic final tuple rejection-close kernel not found".to_string(),
                )
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

        let mut kernel_timings = Vec::with_capacity(checked_sum(source_columns.len(), 2)?);
        let row_map_timing = self.time_epistemic_gpu_kernel_launch(
            "epistemic GPU final tuple row map",
            || unsafe {
                self.provider
                    .device()
                    .inner()
                    .memset_zeros(&mut final_row_count)?;
                self.provider.device().inner().memset_zeros(&mut row_map)?;
                let mut row_map_params: Vec<*mut c_void> = vec![
                    output_row_capacity_u32.as_kernel_param(),
                    literal_count_u32.as_kernel_param(),
                    candidate_count_u32.as_kernel_param(),
                    reduction_count_u32.as_kernel_param(),
                    models_per_reduction_u32.as_kernel_param(),
                    world_stride.as_kernel_param(),
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
                    row_filter_count.as_kernel_param(),
                    (&row_map).as_kernel_param(),
                    (&final_row_count).as_kernel_param(),
                    (&workspace.candidate_assumptions).as_kernel_param(),
                    (&gate_literal_required).as_kernel_param(),
                ];
                row_map_func.clone().launch(
                    LaunchConfig::for_num_elems(output_row_capacity_u32.max(1)),
                    &mut row_map_params,
                )?;
                Ok(())
            },
        )?;
        kernel_timings.push(row_map_timing);

        let close_rejections_timing = self.time_epistemic_gpu_kernel_launch(
            "epistemic GPU final tuple rejection closeout",
            || unsafe {
                let mut close_rejections_params: Vec<*mut c_void> = vec![
                    candidate_count_u32.as_kernel_param(),
                    world_stride.as_kernel_param(),
                    (&final_row_count).as_kernel_param(),
                    (&workspace.rejection_reasons).as_kernel_param(),
                    (&workspace.world_views).as_kernel_param(),
                ];
                close_rejections_func.clone().launch(
                    LaunchConfig::for_num_elems(candidate_count_u32.max(1)),
                    &mut close_rejections_params,
                )?;
                Ok(())
            },
        )?;
        kernel_timings.push(close_rejections_timing);

        for ((src_col, column_byte_len, column_row_width), dst_col) in
            source_columns.iter().zip(result_columns_raw.iter_mut())
        {
            let column_timing = self.time_epistemic_gpu_kernel_launch(
                "epistemic GPU final tuple column materialization",
                || unsafe {
                    // SAFETY: source and destination columns are valid device byte
                    // buffers of identical length, the row-count scalar and schema
                    // row width are runtime-owned, and membership/world-view buffers
                    // were capacity-checked.
                    let mut params: Vec<*mut c_void> = vec![
                        column_byte_len.as_kernel_param(),
                        column_row_width.as_kernel_param(),
                        literal_count_u32.as_kernel_param(),
                        candidate_count_u32.as_kernel_param(),
                        reduction_count_u32.as_kernel_param(),
                        models_per_reduction_u32.as_kernel_param(),
                        world_stride.as_kernel_param(),
                        output.num_rows_device().as_kernel_param(),
                        (&workspace.rejection_reasons).as_kernel_param(),
                        (&workspace.model_membership).as_kernel_param(),
                        (&workspace.world_views).as_kernel_param(),
                        (&row_map).as_kernel_param(),
                        (*src_col).as_kernel_param(),
                        dst_col.as_kernel_param(),
                        (&final_row_count).as_kernel_param(),
                    ];
                    func.clone().launch(
                        LaunchConfig::for_num_elems((*column_byte_len).max(1)),
                        &mut params,
                    )?;
                    Ok(())
                },
            )?;
            kernel_timings.push(column_timing);
        }
        let kernel_timing = EpistemicGpuKernelTimingTrace::checked_sum(kernel_timings)?;

        let result_columns: Vec<CudaColumn> =
            result_columns_raw.into_iter().map(Into::into).collect();
        let final_schema = Schema::new(final_schema_columns)
            .with_sort_labels(final_schema_sort_labels)
            .map_err(|err| XlogError::Execution(format!("epistemic final schema: {err}")))?;
        let final_output = CudaBuffer::from_columns(
            result_columns,
            output.num_rows(),
            final_row_count,
            final_schema,
        );
        let final_output = if gpu_plan.final_output_columns.is_none() {
            final_output
        } else {
            self.provider.dedup_full_row(&final_output)?
        };

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
            tuple_membership_bindings: executable.gpu_plan.tuple_membership_bindings.clone(),
            workspace,
            workspace_reset,
        })
    }

    fn validate_epistemic_gpu_reduced_constraints(
        &self,
        executable: &EpistemicExecutablePlan,
    ) -> Result<EpistemicGpuConstraintValidationTrace> {
        let mut checked_constraint_relations = 0usize;
        let mut violated_constraint_relations = 0usize;
        let mut row_count_device_reads = 0u32;
        let mut violations = Vec::new();

        let mut relation_names = Vec::new();
        for rule in executable
            .reduced_runtime_plan
            .rules_by_scc
            .iter()
            .flatten()
        {
            if rule.head.starts_with(XLOG_CONSTRAINT_RELATION_PREFIX)
                && !relation_names.iter().any(|name| name == &rule.head)
            {
                relation_names.push(rule.head.as_str());
            }
        }

        for relation_name in relation_names {
            checked_constraint_relations += 1;
            let relation = self.store().get(relation_name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "missing reduced constraint relation {relation_name} after production runtime \
                     dispatch"
                ))
            })?;
            let row_count_was_cached = relation.cached_row_count().is_some();
            let rows = self.provider.device_row_count(relation)?;
            row_count_device_reads += u32::from(!row_count_was_cached);
            if rows > 0 {
                violated_constraint_relations += 1;
                violations.push(format!("{relation_name}={rows}"));
            }
        }

        if !violations.is_empty() {
            return Err(XlogError::Execution(format!(
                "epistemic GPU reduced constraint violation: {}",
                violations.join(", ")
            )));
        }

        Ok(EpistemicGpuConstraintValidationTrace {
            checked_constraint_relations,
            violated_constraint_relations,
            row_count_device_reads,
        })
    }

    /// Materialize a stratum's GATED epistemic head output into the relation store
    /// as a base relation, for stratified epistemic execution.
    ///
    /// After a lower stratum computes its modal-gated head extension (the
    /// `final_output`/additional-head buffer), the higher stratum's `know`/
    /// `possible` over that head must read the GATED extension — not the ungated
    /// reduced relation the reduced runtime plan leaves in the store. This OVERWRITES
    /// the store relation under `name` with a device-side clone of the gated buffer,
    /// so the existing EGB-02 tuple-membership filter (which reads the source
    /// relation from the store by predicate name) gates the higher stratum against
    /// the correct extension. No resolve-into-body is performed, so there is no
    /// double-gating against the GPU world-view filter.
    pub fn materialize_epistemic_head_relation(
        &mut self,
        name: &str,
        gated_output: &CudaBuffer,
    ) -> Result<()> {
        let cloned = self.clone_buffer(gated_output)?;
        self.put_relation(name, cloned);
        Ok(())
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
        let launch_metadata_transfer_start = self.provider.host_launch_metadata_transfer_stats();
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
        let trace = EpistemicGpuRuntimeTrace::try_from_preflight_and_counters(
            prepared.preflight,
            counters_before,
            counters_after,
        )?;
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
        let expected_tuple_key_column_reads =
            expected_tuple_key_column_reads(&executable.gpu_plan.tuple_membership_bindings)?;
        model_membership.require_planned_tuple_key_column_reads(expected_tuple_key_column_reads)?;
        let world_view_validation = self.validate_epistemic_gpu_world_views(
            &mut prepared.workspace,
            &executable.gpu_plan,
            candidate_count,
            capacities.max_models_per_reduction,
        )?;
        let constraint_world_view_validation = self.validate_epistemic_gpu_world_view_constraints(
            &mut prepared.workspace,
            &executable.gpu_plan,
            candidate_count,
        )?;
        let materialization =
            self.materialize_epistemic_gpu_candidates(&mut prepared.workspace, candidate_count)?;
        let final_result_materialization = self.materialize_epistemic_gpu_final_results(
            &mut prepared.workspace,
            &output,
            candidate_count,
        )?;
        // Distinct epistemic output heads and the reduction indices that feed each.
        // A single-head plan keeps the unscoped (None) row filter. A JOINT-SOLVED
        // coalesced multi-head plan materializes EACH head against the SAME accepted
        // world view by scoping the modal row-filter to that head's reductions.
        let head_reductions = epistemic_head_reduction_indices(&executable.gpu_plan);
        let is_multi_head = head_reductions.len() > 1;
        let primary_head_filter = if is_multi_head {
            head_reductions.get(output_relation).cloned()
        } else {
            None
        };
        let (final_output, final_tuple_materialization) = self
            .materialize_epistemic_gpu_final_tuples_scoped(
                &mut prepared.workspace,
                &output,
                &executable.gpu_plan,
                literal_count,
                candidate_count,
                executable.gpu_plan.reductions.len(),
                capacities.max_models_per_reduction,
                primary_head_filter.as_ref(),
            )?;
        // Materialize every OTHER coupled head against the shared accepted world
        // view. Each head's reduced relation (already computed jointly by the single
        // reduced-program dispatch above) is the materialization source; only that
        // head's modal row-filter bindings apply.
        let mut additional_head_outputs: Vec<(String, CudaBuffer)> = Vec::new();
        if is_multi_head {
            for (head, reductions) in &head_reductions {
                if head.as_str() == output_relation {
                    continue;
                }
                let head_output = {
                    let reduced_head = self.store().get(head.as_str()).ok_or_else(|| {
                        XlogError::UnsupportedEpistemicConstruct {
                            construct: "epistemic GPU reduced output".to_string(),
                            context: format!(
                                "missing reduced output relation {head} after production runtime \
                                 dispatch for joint multi-head materialization"
                            ),
                        }
                    })?;
                    self.clone_buffer(reduced_head)?
                };
                let (head_final_output, _head_trace) = self
                    .materialize_epistemic_gpu_final_tuples_scoped(
                        &mut prepared.workspace,
                        &head_output,
                        &executable.gpu_plan,
                        literal_count,
                        candidate_count,
                        executable.gpu_plan.reductions.len(),
                        capacities.max_models_per_reduction,
                        Some(reductions),
                    )?;
                additional_head_outputs.push((head.clone(), head_final_output));
            }
        }
        let tuple_evidence_output = if executable.gpu_plan.final_output_columns.is_some() {
            let mut evidence_plan = executable.gpu_plan.clone();
            evidence_plan.final_output_columns = None;
            let (evidence_output, _) = self.materialize_epistemic_gpu_final_tuples(
                &mut prepared.workspace,
                &output,
                &evidence_plan,
                literal_count,
                candidate_count,
                executable.gpu_plan.reductions.len(),
                capacities.max_models_per_reduction,
            )?;
            Some(evidence_output)
        } else {
            None
        };
        let transfer_budget_end = self.provider.host_transfer_stats();
        let launch_metadata_transfer_end = self.provider.host_launch_metadata_transfer_stats();
        let transfer_budget =
            EpistemicGpuTransferBudgetTrace::from_host_transfer_stats_with_launch_metadata(
                candidate_count,
                transfer_budget_start,
                transfer_budget_end,
                launch_metadata_transfer_start,
                launch_metadata_transfer_end,
            )?;
        let final_result_transfer =
            EpistemicGpuFinalResultTransferTrace::from_final_output(&self.provider, &final_output)?;
        final_tuple_materialization.require_row_filter_materialization_evidence(
            "epistemic GPU final tuple materialization",
            final_result_transfer.final_output_rows,
        )?;
        let constraint_validation = self.validate_epistemic_gpu_reduced_constraints(executable)?;
        let semantic_trace = EpistemicGpuSemanticTrace::from_device_rejection_reasons(
            &self.provider,
            &prepared.workspace,
            &candidate_generation,
            &propagation,
            &model_membership,
            &world_view_validation,
        )?;

        Ok(EpistemicGpuExecutionResult {
            provider_identity: EpistemicGpuProviderIdentity::from_provider(&self.provider),
            prepared,
            candidate_generation,
            propagation,
            candidate_validation,
            model_membership,
            world_view_validation,
            constraint_world_view_validation,
            materialization,
            final_result_materialization,
            final_tuple_materialization,
            transfer_budget,
            final_result_transfer,
            constraint_validation,
            semantic_trace,
            tuple_membership_bindings: executable.gpu_plan.tuple_membership_bindings.clone(),
            final_output,
            additional_head_outputs,
            tuple_evidence_output,
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

    /// Execute multiple epistemic GPU executable plans and return an aggregate trace.
    ///
    /// This is used by split-execution certification: every component still
    /// routes through the existing single-plan GPU runtime path, and the batch
    /// trace only aggregates those component traces. It does not perform CPU
    /// recomposition.
    pub fn execute_epistemic_gpu_execution_batch_with_trace(
        &mut self,
        executables: &[&EpistemicExecutablePlan],
        capacities: EpistemicGpuWorkspaceCapacities,
    ) -> Result<EpistemicGpuBatchExecutionResult> {
        let results = self.execute_epistemic_gpu_execution_batch(executables, capacities)?;
        let trace = EpistemicGpuBatchExecutionTrace::try_from_component_results(&results)?;
        Ok(EpistemicGpuBatchExecutionResult { results, trace })
    }
}

#[derive(Default)]
struct RuntimeRouteSummary {
    multiway_reduction_count: usize,
    kclique_wcoj_plan_count: usize,
    wcoj_triangle_route_count: usize,
    wcoj_4cycle_route_count: usize,
    kclique_wcoj_plan_count_by_arity: [usize; 4],
    kclique_wcoj_max_arity: u8,
    kclique_wcoj_edge_permutation_count: usize,
    kclique_stream_groups: BTreeSet<StreamGroupId>,
    kclique_skew_scheduled_plan_count: usize,
    planned_hash_route_count: usize,
    planned_hash_planner_wins_count: usize,
    planned_hash_incomplete_stats_count: usize,
    planned_hash_cost_evidence_count: usize,
    sorted_layout_requirement_count: usize,
    helper_split_spec_count: usize,
}

fn summarize_runtime_routes(node: &RirNode, routes: &mut RuntimeRouteSummary) {
    match node {
        RirNode::MultiWayJoin { inputs, plan, .. } => {
            routes.multiway_reduction_count += 1;
            match plan {
                Some(MultiwayPlan::WcojWithPlan(order)) => {
                    routes.kclique_wcoj_plan_count += 1;
                    if let Some(slot) = usize::from(order.k).checked_sub(5) {
                        if slot < routes.kclique_wcoj_plan_count_by_arity.len() {
                            routes.kclique_wcoj_plan_count_by_arity[slot] += 1;
                        }
                    }
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
                Some(MultiwayPlan::PlannedHashRoute {
                    reason,
                    planner_evidence,
                }) => {
                    routes.planned_hash_route_count += 1;
                    match reason {
                        PlannedHashReason::PlannerPredictsHashWins => {
                            routes.planned_hash_planner_wins_count += 1;
                            if planner_evidence.wcoj_cost.is_finite()
                                && planner_evidence.hash_cost.is_finite()
                                && planner_evidence.hash_cost <= planner_evidence.wcoj_cost
                            {
                                routes.planned_hash_cost_evidence_count += 1;
                            }
                        }
                        PlannedHashReason::IncompleteStatsSafeDefault => {
                            routes.planned_hash_incomplete_stats_count += 1;
                        }
                    }
                }
                None => {
                    if super::wcoj_dispatch::match_multiway_triangle(node).is_some() {
                        routes.wcoj_triangle_route_count += 1;
                    } else if super::wcoj_dispatch::match_multiway_4cycle(node).is_some() {
                        routes.wcoj_4cycle_route_count += 1;
                    }
                }
            }

            for input in inputs {
                summarize_runtime_routes(input, routes);
            }
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
        RirNode::ChainJoin { left, right, .. } => {
            summarize_runtime_routes(left, routes);
            summarize_runtime_routes(right, routes);
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
        RirNode::MultiWayJoin { plan, inputs, .. } => {
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
        RirNode::ChainJoin { left, right, .. } => {
            count_helper_relation_scans(left, helper_relations)
                + count_helper_relation_scans(right, helper_relations)
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
        RirNode::MultiWayJoin { inputs, .. } => inputs
            .iter()
            .map(|input| count_helper_relation_leaf_scans(input, helper_relations))
            .sum(),
        RirNode::ChainJoin { left, right, .. } => {
            count_helper_relation_leaf_scans(left, helper_relations)
                + count_helper_relation_leaf_scans(right, helper_relations)
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

fn checked_u32_dimension(value: usize, context: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| XlogError::ResourceExhausted {
        context: context.to_string(),
        estimated_bytes: value as u64,
        budget_bytes: u32::MAX as u64,
    })
}

/// Map each distinct epistemic output head to the reduction indices feeding it.
///
/// Reduction index = position in `gpu_plan.reductions`, which is exactly the
/// `reduction_index` carried by every tuple-membership binding, so the returned
/// sets scope each head's modal row-filter for joint multi-head materialization.
fn epistemic_head_reduction_indices(
    gpu_plan: &EpistemicGpuPlan,
) -> std::collections::BTreeMap<String, BTreeSet<usize>> {
    let mut heads: std::collections::BTreeMap<String, BTreeSet<usize>> =
        std::collections::BTreeMap::new();
    for (reduction_index, reduction) in gpu_plan.reductions.iter().enumerate() {
        heads
            .entry(reduction.head_predicate.clone())
            .or_default()
            .insert(reduction_index);
    }
    heads
}

fn final_output_columns_for_materialization(
    output: &CudaBuffer,
    gpu_plan: &EpistemicGpuPlan,
) -> Result<Vec<usize>> {
    let Some(final_output_columns) = &gpu_plan.final_output_columns else {
        return Ok((0..output.arity()).collect());
    };

    let mut seen = vec![false; output.arity()];
    for &column in final_output_columns {
        if column >= output.arity() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU final output projection".to_string(),
                context: format!(
                    "final output column {} exceeds reduced output arity {}",
                    column,
                    output.arity()
                ),
            });
        }
        if seen[column] {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU final output projection".to_string(),
                context: format!("duplicate final output column {}", column),
            });
        }
        seen[column] = true;
    }

    Ok(final_output_columns.clone())
}

fn require_u32_launch_bound(value: usize, context: &str) -> Result<()> {
    checked_u32_dimension(value, context).map(|_| ())
}

fn require_u32_launch_dimensions(values: &[usize], context: &str) -> Result<()> {
    let max_value = values.iter().copied().max().unwrap_or(0);
    require_u32_launch_bound(max_value, context)
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

fn require_epistemic_gpu_kernel_phases(gpu_plan: &EpistemicGpuPlan) -> Result<()> {
    let required = [
        EpistemicGpuHotPathPhase::CandidateGeneration,
        EpistemicGpuHotPathPhase::Propagation,
        EpistemicGpuHotPathPhase::CandidateValidation,
        EpistemicGpuHotPathPhase::ModelMembership,
        EpistemicGpuHotPathPhase::WorldViewValidation,
        EpistemicGpuHotPathPhase::ResultMaterialization,
        EpistemicGpuHotPathPhase::FinalResultMaterialization,
        EpistemicGpuHotPathPhase::FinalTupleMaterialization,
    ];

    for phase in required {
        if !gpu_plan.required_kernel_phases.contains(&phase) {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU kernel phase contract".to_string(),
                context: format!(
                    "accepted GPU execution requires kernel phase {:?}, but the plan declared {:?}",
                    phase, gpu_plan.required_kernel_phases
                ),
            });
        }
    }

    Ok(())
}

fn require_epistemic_gpu_buffer_contract(gpu_plan: &EpistemicGpuPlan) -> Result<()> {
    let required = [
        EpistemicGpuBufferKind::CandidateAssumptions,
        EpistemicGpuBufferKind::WorldViews,
        EpistemicGpuBufferKind::ModelMembership,
        EpistemicGpuBufferKind::RejectionReasons,
    ];

    for buffer in required {
        if !gpu_plan.required_buffers.contains(&buffer) {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU buffer contract".to_string(),
                context: format!(
                    "accepted GPU execution requires buffer {:?}, but the plan declared {:?}",
                    buffer, gpu_plan.required_buffers
                ),
            });
        }
    }

    Ok(())
}

fn expected_tuple_key_column_reads(bindings: &[EpistemicTupleMembershipBinding]) -> Result<usize> {
    bindings.iter().try_fold(0usize, |acc, binding| {
        checked_sum(acc, binding.key_columns.len())
    })
}

fn world_view_bitset_bytes_per_candidate(literal_count: usize) -> Result<usize> {
    Ok(checked_sum(literal_count, 7)? / 8)
}

fn epistemic_operator_code(op: EirEpistemicOp) -> u8 {
    match op {
        EirEpistemicOp::Know => 1,
        EirEpistemicOp::Possible => 2,
    }
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
    let required_candidates = 1usize << literal_count;
    if max_candidates < required_candidates {
        return Err(XlogError::ResourceExhausted {
            context: "epistemic GPU execution candidate capacity".to_string(),
            estimated_bytes: required_candidates as u64,
            budget_bytes: max_candidates as u64,
        });
    }
    Ok(required_candidates)
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
