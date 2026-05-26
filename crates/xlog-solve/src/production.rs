//! Production GPU solver adapter for epistemic callers.
//!
//! This module is intentionally thin: it routes accepted solver work into the
//! existing GPU CDCL verifier instead of using the bounded CPU semantic-oracle
//! facade in [`crate::SolverService`].

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::{CudaKernelProvider, DeviceSlice};
use xlog_runtime::{
    EpistemicGpuBatchExecutionResult, EpistemicGpuExecutionResult, EpistemicGpuKernelTimingTrace,
    EpistemicGpuProviderIdentity,
};

use crate::{GpuCdclConfig, GpuCdclSolver, GpuCdclWorkspace, GpuCnf, Objective, SolveInstance};

const PRODUCTION_SOLVER_REQUIRED_CAPABILITY_COUNT: u64 = 5;
const PRODUCTION_SOLVER_REQUIRED_STATUS_COUNT: u64 = 4;
const MAX_WEIGHTED_MAXSAT_FRONTIER_COMPLETION_CANDIDATES: u64 = 64;

macro_rules! checked_solver_trace_counter_inc {
    ($adapter:ident, $field:ident) => {{
        $adapter.trace.$field = GpuSolverProductionAdapter::checked_trace_counter_add(
            $adapter.trace.$field,
            1,
            stringify!($field),
        )?;
    }};
}

macro_rules! checked_solver_report_counter_inc {
    ($report:ident, $field:ident) => {{
        $report.$field = GpuSolverProductionAdapter::checked_report_counter_add(
            $report.$field,
            1,
            stringify!($field),
        )?;
    }};
}

macro_rules! checked_solver_report_counter_add {
    ($report:ident, $field:ident, $delta:expr) => {{
        $report.$field = GpuSolverProductionAdapter::checked_report_counter_add(
            $report.$field,
            $delta,
            stringify!($field),
        )?;
    }};
}

/// Production capability status for solver paths required by v0.9.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSolverProductionCapabilityStatus {
    /// Existing GPU-native production path is available.
    Available,
    /// Required GPU-native production path is not implemented.
    Blocked,
}

/// Capability report for the solver production adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuSolverProductionCapabilities {
    /// Complete SAT/UNSAT execution through the existing GPU CDCL verifier.
    pub gpu_cdcl_sat_unsat: GpuSolverProductionCapabilityStatus,
    /// GPU-native MaxSAT production execution.
    pub gpu_maxsat: GpuSolverProductionCapabilityStatus,
    /// GPU SAT/MaxSAT/status-aware portfolio production execution.
    pub gpu_portfolio_sat_maxsat: GpuSolverProductionCapabilityStatus,
    /// Whether the CPU semantic-oracle solver may satisfy production metrics.
    pub cpu_oracle_solver_allowed: bool,
    /// Blocker reason for GPU-native MaxSAT, or empty when available.
    pub gpu_maxsat_blocker: &'static str,
    /// Blocker reason for GPU SAT/MaxSAT/status-aware portfolio execution.
    pub gpu_portfolio_blocker: &'static str,
}

/// Solver-facing state derived from an accepted GPU epistemic execution result.
///
/// This is the production boundary between the epistemic candidate state machine
/// and solver services. It is built only after the GPU semantic trace, final
/// output, and hot-path counters have passed accepted-path validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuSolverAcceptedCandidateState {
    /// Number of accepted GPU execution records represented by this state.
    pub evidence_records: u64,
    /// Device candidate indices accepted by world-view validation.
    pub accepted_candidate_indices: Vec<usize>,
    /// Number of accepted candidate states entering solver services.
    pub accepted_candidates: u64,
    /// Number of accepted world views represented by accepted candidates.
    pub accepted_world_views: u64,
    /// Logical final-output rows materialized by the accepted GPU execution.
    pub final_output_rows: u64,
    /// Epistemic literal count in the accepted GPU plan.
    pub epistemic_literals: u64,
    /// Tuple-membership bindings consumed by the accepted GPU plan.
    pub tuple_membership_bindings: u64,
    /// Solver assumption bindings exported by the accepted semantic plan.
    pub solver_assumption_bindings: u64,
    /// Solver production capabilities required by the accepted semantic plan.
    pub solver_required_capabilities: u64,
    /// Solver statuses that must cross the accepted semantic boundary distinctly.
    pub solver_required_statuses: u64,
    /// Whether the accepted evidence came from G91 mode.
    pub g91_mode: bool,
    /// Whether the accepted evidence came from FAEEL mode.
    pub faeel_mode: bool,
    /// Whether accepted evidence contains `know` operators.
    pub has_know_operator: bool,
    /// Whether accepted evidence contains `possible` operators.
    pub has_possible_operator: bool,
    /// Whether accepted evidence contains `not possible` operators.
    pub has_not_possible_operator: bool,
    /// Whether accepted evidence contains `not know` operators.
    pub has_not_know_operator: bool,
    /// Tuple-key column reads performed while staging accepted tuple evidence.
    pub tuple_key_column_reads: u64,
    /// Whether accepted evidence includes nonzero-arity tuple keys.
    pub has_nonzero_arity_tuple_keys: bool,
    /// GPU final-tuple row filters used to materialize variable-bound evidence.
    pub final_tuple_row_filters: u64,
    /// Negated GPU final-tuple row filters used to materialize variable-bound evidence.
    pub final_tuple_negated_row_filters: u64,
    /// Final-output row capacity checked against row-specific GPU model slots.
    pub row_specific_membership_row_capacity: u64,
    /// Final-output row capacity checked by fallback GPU row filters outside model slots.
    pub row_filter_fallback_row_capacity: u64,
    /// Reduced integrity-constraint relations checked before entering solver services.
    pub checked_constraint_relations: u64,
    /// Constraint row-count metadata reads used before entering solver services.
    pub constraint_row_count_device_reads: u64,
}

impl GpuSolverAcceptedCandidateState {
    fn from_validated_result(
        result: &EpistemicGpuExecutionResult,
        final_output_rows: usize,
    ) -> Self {
        let preflight = &result.prepared.preflight;
        let tuple_key_column_reads = result.model_membership.tuple_source_key_column_device_reads;
        Self {
            evidence_records: 1,
            accepted_candidate_indices: result.semantic_trace.accepted_candidate_indices.clone(),
            accepted_candidates: result.semantic_trace.accepted_candidates as u64,
            accepted_world_views: result.semantic_trace.accepted_world_views as u64,
            final_output_rows: final_output_rows as u64,
            epistemic_literals: result.candidate_generation.literal_count as u64,
            tuple_membership_bindings: preflight.tuple_membership_binding_count as u64,
            solver_assumption_bindings: preflight.solver_assumption_binding_count as u64,
            solver_required_capabilities: preflight.solver_required_capability_count as u64,
            solver_required_statuses: preflight.solver_required_status_count as u64,
            g91_mode: preflight.is_g91_mode(),
            faeel_mode: preflight.is_faeel_mode(),
            has_know_operator: preflight.know_operator_count > 0,
            has_possible_operator: preflight.possible_operator_count > 0,
            has_not_possible_operator: preflight.not_possible_operator_count > 0,
            has_not_know_operator: preflight.not_know_operator_count > 0,
            tuple_key_column_reads: tuple_key_column_reads as u64,
            has_nonzero_arity_tuple_keys: tuple_key_column_reads > 0,
            final_tuple_row_filters: result.final_tuple_materialization.row_filter_count as u64,
            final_tuple_negated_row_filters: result
                .final_tuple_materialization
                .negated_row_filter_count as u64,
            row_specific_membership_row_capacity: result
                .final_tuple_materialization
                .row_specific_membership_row_capacity
                as u64,
            row_filter_fallback_row_capacity: result
                .final_tuple_materialization
                .row_filter_row_capacity_outside_model_slot_window
                as u64,
            checked_constraint_relations: result.constraint_validation.checked_constraint_relations
                as u64,
            constraint_row_count_device_reads: result.constraint_validation.row_count_device_reads
                as u64,
        }
    }
}

/// Expected GPU CDCL result for one production lifecycle step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSolverProductionExpectation {
    /// The step must be SAT under the currently pushed assumptions.
    Sat,
    /// The step must be UNSAT under the currently pushed assumptions.
    Unsat,
    /// The accepted lifecycle step ended without a determined SAT/UNSAT status.
    Unknown {
        /// Diagnostic reason reported by the GPU-backed lifecycle scheduler.
        reason: &'static str,
    },
    /// The accepted lifecycle step exhausted its GPU-backed budget.
    Timeout {
        /// Nonzero timeout budget observed by the lifecycle scheduler.
        budget_micros: u64,
    },
}

/// One accepted solver lifecycle step backed by existing GPU CDCL inputs.
#[derive(Clone, Copy)]
pub struct GpuSolverProductionLifecycleStep<'a> {
    /// Device-resident CNF for this step, including any assumption clauses.
    pub cnf: &'a GpuCnf,
    /// Device-resident branch limit passed to the GPU CDCL solver.
    pub branch_var_limit: &'a TrackedCudaSlice<u32>,
    /// Expected SAT/UNSAT status for the step.
    pub expectation: GpuSolverProductionExpectation,
}

/// Summary of an accepted solver lifecycle run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuSolverProductionLifecycleReport {
    /// Number of accepted GPU epistemic candidate evidence records consumed.
    pub candidate_evidence_records: u64,
    /// Number of lifecycle steps executed.
    pub steps: u64,
    /// Number of lifecycle steps that reached SAT through GPU CDCL.
    pub sat_steps: u64,
    /// Number of lifecycle steps that reached UNSAT through GPU CDCL.
    pub unsat_steps: u64,
    /// Number of assumption pushes recorded before GPU solves.
    pub assumption_pushes: u64,
    /// Number of assumption retractions recorded after GPU solves.
    pub assumption_retractions: u64,
    /// Number of UNSAT steps that reused the provided GPU CDCL workspace allocation.
    pub workspace_reuses: u64,
    /// Number of lifecycle steps that propagated UNKNOWN without CPU search.
    pub unknown_steps: u64,
    /// Number of lifecycle steps that propagated TIMEOUT without CPU search.
    pub timeout_steps: u64,
}

/// Accepted split/batch GPU epistemic evidence for solver production reuse.
#[derive(Clone, Copy)]
pub struct GpuSolverProductionBatchExecutionEvidence<'a> {
    /// Results plus aggregate trace from the split/batch GPU execution adapter.
    pub batch: &'a EpistemicGpuBatchExecutionResult,
}

/// Summary of a GPU CDCL learned-clause arena publication.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuSolverProductionLearnedClauseArenaReport {
    /// Number of UNSAT solves used to populate the learned-clause/proof arena.
    pub unsat_solves: u64,
    /// Number of learned-clause arenas published from device buffers.
    pub gpu_learned_clause_arena_publications: u64,
    /// Number of learned-count device buffers published with the arena.
    pub gpu_learned_count_buffer_publications: u64,
    /// CPU learned-clause transfers performed by this adapter.
    pub cpu_learned_clause_transfers: u64,
}

/// Summary of a bounded GPU CDCL learned-clause reuse run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuSolverProductionLearnedClauseReuseReport {
    /// Number of accepted GPU epistemic candidate evidence records consumed.
    pub candidate_evidence_records: u64,
    /// Number of accepted candidate solves represented by this bounded reuse run.
    pub candidates: u64,
    /// Number of UNSAT solves executed through the reusable GPU CDCL workspace.
    pub unsat_solves: u64,
    /// Number of learned-clause arenas published from device buffers.
    pub gpu_learned_clause_arena_publications: u64,
    /// Number of learned-clause arenas imported from device buffers.
    pub gpu_learned_clause_imports: u64,
    /// Number of UNSAT solves that reused imported GPU learned clauses.
    pub gpu_learned_clause_reused_solves: u64,
    /// CPU learned-clause transfers performed by this adapter.
    pub cpu_learned_clause_transfers: u64,
}

/// One GPU-CDCL-backed candidate for bounded weighted MaxSAT production solving.
///
/// The candidate CNF should encode the hard clauses plus the soft-clause subset
/// represented by `score`. The adapter certifies each provided candidate through
/// the existing GPU CDCL SAT path; it does not enumerate assignments on CPU.
#[derive(Clone, Copy)]
pub struct GpuSolverProductionMaxSatCandidate<'a> {
    /// Candidate MaxSAT score represented by this satisfiable CNF.
    pub score: u64,
    /// Device-resident CNF for this MaxSAT candidate.
    pub cnf: &'a GpuCnf,
    /// Device-resident branch limit passed to the GPU CDCL solver.
    pub branch_var_limit: &'a TrackedCudaSlice<u32>,
}

/// Expected GPU-CDCL status for a bounded MaxSAT search candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSolverProductionMaxSatSearchStatus {
    /// The candidate should be satisfiable and eligible for optimum scoring.
    Satisfiable,
    /// The candidate should be unsatisfiable and pruned by GPU CDCL.
    Unsatisfiable,
}

/// One GPU-CDCL-backed candidate in a bounded weighted MaxSAT search.
#[derive(Clone, Copy)]
pub struct GpuSolverProductionMaxSatSearchCandidate<'a> {
    /// Candidate MaxSAT score represented when the CNF is satisfiable.
    pub score: u64,
    /// Device-resident CNF for this MaxSAT candidate.
    pub cnf: &'a GpuCnf,
    /// Device-resident branch limit passed to the GPU CDCL solver.
    pub branch_var_limit: &'a TrackedCudaSlice<u32>,
    /// Expected candidate status certified by GPU CDCL.
    pub status: GpuSolverProductionMaxSatSearchStatus,
}

/// One caller-declared weighted soft-clause selection for GPU MaxSAT search encoding.
///
/// The adapter treats `soft_clause_indices` as seed soft clauses for a bounded
/// search candidate, completes any upper-bound boundary candidates implied by
/// UNSAT seeds, uploads each candidate with the existing GPU CNF layout, and
/// certifies `status` through GPU CDCL. It does not enumerate assignments on CPU.
#[derive(Clone, Copy)]
pub struct GpuSolverProductionWeightedMaxSatSelection<'a> {
    /// Indices of weighted soft clauses selected for this bounded candidate.
    pub soft_clause_indices: &'a [usize],
    /// Expected GPU-CDCL status for the encoded selected-clause CNF.
    pub status: GpuSolverProductionMaxSatSearchStatus,
}

struct GpuSolverProductionEncodedMaxSatSearchCandidate {
    score: u64,
    cnf: GpuCnf,
    status: GpuSolverProductionMaxSatSearchStatus,
}

struct GpuSolverProductionCompletedWeightedMaxSatFrontier {
    selections: Vec<GpuSolverProductionOwnedWeightedMaxSatSelection>,
    completion_candidate_count: u64,
}

#[derive(Clone)]
struct GpuSolverProductionOwnedWeightedMaxSatSelection {
    soft_clause_indices: Vec<usize>,
    status: GpuSolverProductionMaxSatSearchStatus,
}

struct GpuSolverProductionUnsatFrontierCertificate {
    indices: Vec<usize>,
    min_weight: u64,
    min_weight_indices: Vec<usize>,
}

/// Summary of one bounded GPU-backed MaxSAT production adapter run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuSolverProductionMaxSatReport {
    /// Number of accepted GPU epistemic candidate evidence records consumed.
    pub candidate_evidence_records: u64,
    /// Best score among GPU-certified satisfiable candidates.
    pub optimum_score: u64,
    /// Number of candidate CNFs checked.
    pub candidates_checked: u64,
    /// Number of GPU-certified satisfiable candidates eligible for scoring.
    pub satisfiable_candidates: u64,
    /// Number of GPU-certified UNSAT candidates pruned from scoring.
    pub unsat_candidates_pruned: u64,
    /// Number of weighted MaxSAT selections encoded into GPU CNF candidates.
    pub gpu_cdcl_candidate_encodes: u64,
    /// Number of candidate solves dispatched through GPU CDCL.
    pub gpu_cdcl_candidate_solves: u64,
    /// Number of encoded weighted frontiers with a certified optimum upper bound.
    pub frontier_upper_bound_certificates: u64,
}

/// Summary of a combined accepted solver lifecycle plus MaxSAT run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuSolverProductionMaxSatLifecycleReport {
    /// Number of accepted GPU epistemic candidate evidence records consumed.
    pub candidate_evidence_records: u64,
    /// Push/solve/retract lifecycle report for the accepted GPU evidence.
    pub lifecycle: GpuSolverProductionLifecycleReport,
    /// Bounded MaxSAT candidate report for the same accepted GPU evidence.
    pub maxsat: GpuSolverProductionMaxSatReport,
}

/// One job in an accepted GPU-backed MaxSAT scheduler batch.
#[derive(Clone, Copy)]
pub enum GpuSolverProductionMaxSatScheduleJob<'a> {
    /// Certify a caller-provided weighted candidate set through GPU CDCL SAT.
    CandidateSet {
        /// Candidate set to certify.
        candidates: &'a [GpuSolverProductionMaxSatCandidate<'a>],
    },
    /// Certify and prune a caller-provided weighted MaxSAT search frontier.
    Search {
        /// Search candidates to certify or prune.
        candidates: &'a [GpuSolverProductionMaxSatSearchCandidate<'a>],
    },
    /// Encode weighted soft-clause selections into GPU CNF candidates before search.
    EncodedSearch {
        /// Weighted MaxSAT instance whose soft clauses define the schedule.
        weighted: &'a SolveInstance,
        /// Device-resident branch limit passed to the GPU CDCL solver.
        branch_var_limit: &'a TrackedCudaSlice<u32>,
        /// Soft-clause selections to encode and certify.
        selections: &'a [GpuSolverProductionWeightedMaxSatSelection<'a>],
    },
    /// A scheduled MaxSAT batch whose GPU-backed budget ended inconclusively.
    Unknown {
        /// Diagnostic reason recorded by the accepted scheduler.
        reason: &'static str,
    },
    /// A scheduled MaxSAT batch whose accepted GPU-backed budget timed out.
    Timeout {
        /// Timeout budget observed by the accepted scheduler.
        budget_micros: u64,
    },
}

/// Summary of a heterogeneous GPU-backed MaxSAT scheduler batch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuSolverProductionMaxSatScheduleReport {
    /// Number of accepted GPU epistemic candidate evidence records consumed.
    pub candidate_evidence_records: u64,
    /// Number of scheduled jobs executed.
    pub jobs: u64,
    /// Number of weighted candidate-set jobs.
    pub candidate_set_jobs: u64,
    /// Number of search-pruning jobs.
    pub search_jobs: u64,
    /// Number of weighted soft-clause encoding plus search jobs.
    pub encoded_search_jobs: u64,
    /// Number of UNKNOWN statuses propagated without CPU search.
    pub unknown_jobs: u64,
    /// Number of TIMEOUT statuses propagated without CPU search.
    pub timeout_jobs: u64,
    /// Best optimum score observed across all GPU-certified scheduled MaxSAT jobs.
    pub optimum_score: u64,
    /// Number of candidate CNFs checked across scheduled MaxSAT jobs.
    pub candidates_checked: u64,
    /// Number of GPU-certified satisfiable candidates eligible for scoring.
    pub satisfiable_candidates: u64,
    /// Number of GPU-certified UNSAT candidates pruned from scoring.
    pub unsat_candidates_pruned: u64,
    /// Number of weighted MaxSAT selections encoded into GPU CNF candidates.
    pub gpu_cdcl_candidate_encodes: u64,
    /// Number of candidate solves dispatched through GPU CDCL.
    pub gpu_cdcl_candidate_solves: u64,
    /// Number of encoded weighted frontiers with a certified optimum upper bound.
    pub frontier_upper_bound_certificates: u64,
}

/// One job in a bounded GPU solver portfolio.
#[derive(Clone, Copy)]
pub enum GpuSolverProductionPortfolioJob<'a> {
    /// A SAT job dispatched through GPU CDCL.
    Sat {
        /// Device-resident CNF for this SAT job.
        cnf: &'a GpuCnf,
        /// Device-resident branch limit passed to the GPU CDCL solver.
        branch_var_limit: &'a TrackedCudaSlice<u32>,
    },
    /// A bounded MaxSAT job dispatched through GPU CDCL candidate checks.
    MaxSat {
        /// Candidate set to certify.
        candidates: &'a [GpuSolverProductionMaxSatCandidate<'a>],
    },
    /// A weighted MaxSAT job encoded into GPU CNF candidates with an optimum
    /// upper-bound certificate.
    EncodedMaxSat {
        /// Weighted MaxSAT instance whose soft clauses define the encoded candidates.
        weighted: &'a SolveInstance,
        /// Device-resident branch limit passed to the GPU CDCL solver.
        branch_var_limit: &'a TrackedCudaSlice<u32>,
        /// Soft-clause selections to encode and certify.
        selections: &'a [GpuSolverProductionWeightedMaxSatSelection<'a>],
    },
    /// A status-aware job whose GPU-backed portfolio budget ended inconclusively.
    Unknown {
        /// Diagnostic reason recorded by the accepted portfolio scheduler.
        reason: &'static str,
    },
    /// A status-aware job whose accepted portfolio budget timed out.
    Timeout {
        /// Timeout budget observed by the accepted portfolio scheduler.
        budget_micros: u64,
    },
}

/// Summary of one bounded GPU SAT/MaxSAT/status-aware portfolio run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuSolverProductionPortfolioReport {
    /// Number of accepted GPU epistemic candidate evidence records consumed.
    pub candidate_evidence_records: u64,
    /// Number of portfolio jobs executed.
    pub jobs: u64,
    /// Number of SAT jobs executed.
    pub sat_jobs: u64,
    /// Number of MaxSAT jobs executed.
    pub maxsat_jobs: u64,
    /// Number of portfolio jobs that propagated UNKNOWN without CPU search.
    pub unknown_jobs: u64,
    /// Number of portfolio jobs that propagated TIMEOUT without CPU search.
    pub timeout_jobs: u64,
    /// Sum of best MaxSAT scores returned by MaxSAT jobs.
    pub maxsat_optimum_scores: u64,
    /// Number of MaxSAT candidate CNFs checked by portfolio jobs.
    pub maxsat_candidates_checked: u64,
    /// Number of satisfiable MaxSAT candidate CNFs scored by portfolio jobs.
    pub maxsat_satisfiable_candidates: u64,
    /// Number of unsatisfiable MaxSAT candidate CNFs pruned by portfolio jobs.
    pub maxsat_unsat_candidates_pruned: u64,
    /// Number of weighted MaxSAT selections encoded into GPU CNF candidates by portfolio
    /// jobs.
    pub maxsat_gpu_cdcl_candidate_encodes: u64,
    /// Number of MaxSAT candidate solves dispatched through GPU CDCL by portfolio jobs.
    pub maxsat_gpu_cdcl_candidate_solves: u64,
    /// Number of encoded weighted frontiers with a certified optimum upper bound.
    pub maxsat_frontier_upper_bound_certificates: u64,
}

/// Return the current production solver capability report.
pub fn production_capabilities() -> GpuSolverProductionCapabilities {
    GpuSolverProductionCapabilities {
        gpu_cdcl_sat_unsat: GpuSolverProductionCapabilityStatus::Available,
        gpu_maxsat: GpuSolverProductionCapabilityStatus::Available,
        gpu_portfolio_sat_maxsat: GpuSolverProductionCapabilityStatus::Available,
        cpu_oracle_solver_allowed: false,
        gpu_maxsat_blocker: "",
        gpu_portfolio_blocker: "",
    }
}

/// Trace counters proving the production adapter stayed on the GPU CDCL path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuSolverProductionTrace {
    /// Number of accepted GPU epistemic candidate evidence records consumed.
    pub accepted_gpu_candidate_evidence_consumed: u64,
    /// Number of accepted GPU candidate states passed into solver services.
    pub accepted_gpu_candidate_state_transitions: u64,
    /// Number of accepted GPU world-view states passed into solver services.
    pub accepted_gpu_world_view_state_transitions: u64,
    /// Logical final-output rows represented by accepted solver evidence.
    pub accepted_gpu_candidate_final_output_rows_consumed: u64,
    /// Number of accepted split/batch GPU epistemic candidate evidence records consumed.
    pub accepted_gpu_batch_candidate_evidence_consumed: u64,
    /// Number of accepted split/batch GPU epistemic component evidence records consumed.
    pub accepted_gpu_batch_candidate_component_evidence_consumed: u64,
    /// Number of accepted G91 GPU epistemic candidate evidence records consumed.
    pub accepted_g91_gpu_candidate_evidence_consumed: u64,
    /// Number of accepted FAEEL GPU epistemic candidate evidence records consumed.
    pub accepted_faeel_gpu_candidate_evidence_consumed: u64,
    /// Number of accepted GPU candidate evidence records containing `know` operators.
    pub accepted_know_gpu_candidate_evidence_consumed: u64,
    /// Number of accepted GPU candidate evidence records containing `possible` operators.
    pub accepted_possible_gpu_candidate_evidence_consumed: u64,
    /// Number of accepted GPU candidate evidence records containing `not possible` operators.
    pub accepted_not_possible_gpu_candidate_evidence_consumed: u64,
    /// Number of accepted GPU candidate evidence records containing `not know` operators.
    pub accepted_not_know_gpu_candidate_evidence_consumed: u64,
    /// Number of accepted GPU candidate evidence records backed by nonzero-arity tuple keys.
    pub accepted_nonzero_arity_gpu_candidate_evidence_consumed: u64,
    /// Aggregate tuple-key column reads consumed from accepted GPU candidate evidence.
    pub accepted_gpu_candidate_tuple_key_column_reads_consumed: u64,
    /// Planner-exported solver assumption bindings consumed from accepted GPU solver evidence.
    pub accepted_solver_assumption_bindings_consumed: u64,
    /// Required solver capabilities consumed from accepted GPU solver evidence.
    pub accepted_solver_required_capabilities_consumed: u64,
    /// Required solver statuses consumed from accepted GPU solver evidence.
    pub accepted_solver_required_statuses_consumed: u64,
    /// GPU final-tuple row filters consumed from accepted GPU solver evidence.
    pub accepted_gpu_final_tuple_row_filters_consumed: u64,
    /// Negated GPU final-tuple row filters consumed from accepted GPU solver evidence.
    pub accepted_gpu_final_tuple_negated_row_filters_consumed: u64,
    /// Row-specific GPU model-slot capacity consumed from accepted GPU solver evidence.
    pub accepted_gpu_row_specific_membership_row_capacity_consumed: u64,
    /// Fallback GPU row-filter capacity consumed outside bounded model-slot windows.
    pub accepted_gpu_row_filter_fallback_row_capacity_consumed: u64,
    /// Reduced integrity-constraint relations checked before accepted solver work.
    pub accepted_gpu_constraint_relations_checked_consumed: u64,
    /// Constraint row-count metadata reads consumed before accepted solver work.
    pub accepted_gpu_constraint_row_count_device_reads_consumed: u64,
    /// GPU solver production/status events that occurred inside accepted epistemic evidence gates.
    pub accepted_gpu_solver_production_path_events: u64,
    /// Number of SAT expectations dispatched through `GpuCdclSolver`.
    pub gpu_cdcl_sat_solves: u64,
    /// Number of UNSAT expectations dispatched through `GpuCdclSolver`.
    pub gpu_cdcl_unsat_solves: u64,
    /// Number of UNSAT expectations dispatched with a reusable GPU workspace.
    pub gpu_cdcl_workspace_unsat_solves: u64,
    /// Number of assumption pushes recorded for accepted lifecycle steps.
    pub gpu_assumption_pushes: u64,
    /// Number of assumption retractions recorded for accepted lifecycle steps.
    pub gpu_assumption_retractions: u64,
    /// Number of lifecycle UNSAT steps that reused the same GPU CDCL workspace.
    pub gpu_lifecycle_workspace_reuses: u64,
    /// Number of lifecycle UNKNOWN statuses propagated without CPU search.
    pub gpu_lifecycle_unknown_status_steps: u64,
    /// Number of lifecycle TIMEOUT statuses propagated without CPU search.
    pub gpu_lifecycle_timeout_status_steps: u64,
    /// Number of device learned-clause arenas published by accepted GPU CDCL solves.
    pub gpu_learned_clause_arena_publications: u64,
    /// Number of device learned-count buffers published with learned-clause arenas.
    pub gpu_learned_count_buffer_publications: u64,
    /// Number of device learned-clause arenas imported into later GPU CDCL solves.
    pub gpu_learned_clause_imports: u64,
    /// Number of GPU CDCL solves that reused imported learned clauses.
    pub gpu_learned_clause_reused_solves: u64,
    /// Number of learned-clause imports rejected because candidate CNFs differ.
    pub gpu_learned_clause_reuse_rejections: u64,
    /// Number of bounded MaxSAT candidate CNFs dispatched through GPU CDCL.
    pub gpu_maxsat_candidate_solves: u64,
    /// Number of MaxSAT candidate solves covered by an encoded frontier upper-bound certificate.
    pub gpu_maxsat_frontier_certified_candidate_solves: u64,
    /// Number of weighted MaxSAT selections encoded into GPU CNF candidates.
    pub gpu_maxsat_candidate_encodes: u64,
    /// Data-plane H2D calls used while uploading encoded MaxSAT CNF candidates.
    pub gpu_maxsat_candidate_cnf_data_plane_htod_calls: u64,
    /// Data-plane H2D bytes used while uploading encoded MaxSAT CNF candidates.
    pub gpu_maxsat_candidate_cnf_data_plane_htod_bytes: u64,
    /// Data-plane D2H calls observed while uploading encoded MaxSAT CNF candidates.
    pub gpu_maxsat_candidate_cnf_data_plane_dtoh_calls: u64,
    /// Data-plane D2H bytes observed while uploading encoded MaxSAT CNF candidates.
    pub gpu_maxsat_candidate_cnf_data_plane_dtoh_bytes: u64,
    /// Launch-metadata H2D calls used while uploading encoded MaxSAT CNF candidates.
    pub gpu_maxsat_candidate_cnf_launch_metadata_htod_calls: u64,
    /// Launch-metadata H2D bytes used while uploading encoded MaxSAT CNF candidates.
    pub gpu_maxsat_candidate_cnf_launch_metadata_htod_bytes: u64,
    /// Number of upper-bound frontier candidates derived before GPU CDCL verification.
    pub gpu_maxsat_frontier_completion_candidate_encodes: u64,
    /// Number of encoded weighted MaxSAT frontiers with a certified optimum upper bound.
    pub gpu_maxsat_frontier_upper_bound_certificates: u64,
    /// Number of heterogeneous MaxSAT scheduler jobs dispatched.
    pub gpu_maxsat_scheduler_jobs: u64,
    /// Number of scheduler candidate-set jobs dispatched.
    pub gpu_maxsat_scheduler_candidate_set_jobs: u64,
    /// Number of scheduler search-pruning jobs dispatched.
    pub gpu_maxsat_scheduler_search_jobs: u64,
    /// Number of scheduler encoded-search jobs dispatched.
    pub gpu_maxsat_scheduler_encoded_search_jobs: u64,
    /// Number of scheduler UNKNOWN statuses propagated without CPU search.
    pub gpu_maxsat_scheduler_unknown_status_jobs: u64,
    /// Number of scheduler TIMEOUT statuses propagated without CPU search.
    pub gpu_maxsat_scheduler_timeout_status_jobs: u64,
    /// Number of bounded MaxSAT search candidates pruned as UNSAT by GPU CDCL.
    pub gpu_maxsat_unsat_candidate_prunes: u64,
    /// Number of bounded MaxSAT optima certified by GPU CDCL candidate solves.
    pub gpu_maxsat_optima: u64,
    /// Number of portfolio jobs dispatched by the production adapter.
    pub gpu_portfolio_jobs: u64,
    /// Number of SAT jobs dispatched through the portfolio adapter.
    pub gpu_portfolio_sat_jobs: u64,
    /// Number of MaxSAT jobs dispatched through the portfolio adapter.
    pub gpu_portfolio_maxsat_jobs: u64,
    /// Number of accepted portfolio UNKNOWN statuses propagated without CPU search.
    pub gpu_portfolio_unknown_status_jobs: u64,
    /// Number of accepted portfolio TIMEOUT statuses propagated without CPU search.
    pub gpu_portfolio_timeout_status_jobs: u64,
    /// CPU exhaustive assignment enumerations performed by this adapter.
    pub cpu_assignment_enumerations: u64,
    /// CPU MaxSAT assignment enumerations performed by this adapter.
    pub cpu_maxsat_enumerations: u64,
    /// CPU learned-clause transfers performed by this adapter.
    pub cpu_learned_clause_transfers: u64,
}

#[derive(Debug, Clone, Copy)]
struct GpuSolverAcceptedPathEventSnapshot {
    production: u64,
    status: u64,
}

impl GpuSolverProductionTrace {
    fn checked_gpu_solver_production_path_events(&self) -> Result<u64> {
        Self::checked_production_event_sum(
            "gpu_solver_production_path_events",
            &[
                self.gpu_cdcl_sat_solves,
                self.gpu_cdcl_unsat_solves,
                self.gpu_cdcl_workspace_unsat_solves,
                self.gpu_learned_clause_arena_publications,
                self.gpu_learned_count_buffer_publications,
                self.gpu_learned_clause_imports,
                self.gpu_learned_clause_reused_solves,
                self.gpu_maxsat_candidate_solves,
                self.gpu_maxsat_candidate_encodes,
                self.gpu_maxsat_scheduler_candidate_set_jobs,
                self.gpu_maxsat_scheduler_search_jobs,
                self.gpu_maxsat_scheduler_encoded_search_jobs,
                self.gpu_maxsat_unsat_candidate_prunes,
                self.gpu_portfolio_sat_jobs,
                self.gpu_portfolio_maxsat_jobs,
            ],
        )
    }

    fn checked_gpu_solver_status_path_events(&self) -> Result<u64> {
        Self::checked_production_event_sum(
            "gpu_solver_status_path_events",
            &[
                self.gpu_lifecycle_unknown_status_steps,
                self.gpu_lifecycle_timeout_status_steps,
                self.gpu_maxsat_scheduler_unknown_status_jobs,
                self.gpu_maxsat_scheduler_timeout_status_jobs,
                self.gpu_portfolio_unknown_status_jobs,
                self.gpu_portfolio_timeout_status_jobs,
            ],
        )
    }

    fn accepted_path_event_snapshot(&self) -> Result<GpuSolverAcceptedPathEventSnapshot> {
        Ok(GpuSolverAcceptedPathEventSnapshot {
            production: self.checked_gpu_solver_production_path_events()?,
            status: self.checked_gpu_solver_status_path_events()?,
        })
    }

    fn checked_production_event_sum(counter: &str, values: &[u64]) -> Result<u64> {
        values.iter().try_fold(0u64, |acc, value| {
            acc.checked_add(*value)
                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production trace accounting".to_string(),
                    context: format!(
                        "GPU solver production counter {counter} overflowed while adding \
                         {value} to {acc}"
                    ),
                })
        })
    }

    fn checked_maxsat_production_metric_events(&self) -> Result<u64> {
        Self::checked_production_event_sum(
            "gpu_solver_maxsat_metric_events",
            &[
                self.gpu_maxsat_candidate_solves,
                self.gpu_maxsat_frontier_certified_candidate_solves,
                self.gpu_maxsat_candidate_encodes,
                self.gpu_maxsat_frontier_upper_bound_certificates,
                self.gpu_maxsat_scheduler_candidate_set_jobs,
                self.gpu_maxsat_scheduler_search_jobs,
                self.gpu_maxsat_scheduler_encoded_search_jobs,
                self.gpu_maxsat_unsat_candidate_prunes,
                self.gpu_maxsat_optima,
                self.gpu_portfolio_maxsat_jobs,
            ],
        )
    }

    fn checked_portfolio_job_kind_events(&self) -> Result<u64> {
        Self::checked_production_event_sum(
            "gpu_solver_portfolio_job_kind_events",
            &[
                self.gpu_portfolio_sat_jobs,
                self.gpu_portfolio_maxsat_jobs,
                self.gpu_portfolio_unknown_status_jobs,
                self.gpu_portfolio_timeout_status_jobs,
            ],
        )
    }

    fn checked_maxsat_scheduler_job_kind_events(&self) -> Result<u64> {
        Self::checked_production_event_sum(
            "gpu_solver_maxsat_scheduler_job_kind_events",
            &[
                self.gpu_maxsat_scheduler_candidate_set_jobs,
                self.gpu_maxsat_scheduler_search_jobs,
                self.gpu_maxsat_scheduler_encoded_search_jobs,
                self.gpu_maxsat_scheduler_unknown_status_jobs,
                self.gpu_maxsat_scheduler_timeout_status_jobs,
            ],
        )
    }

    fn require_maxsat_scheduler_job_accounting(&self) -> Result<()> {
        let job_kind_events = self.checked_maxsat_scheduler_job_kind_events()?;
        if self.gpu_maxsat_scheduler_jobs != job_kind_events {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "MaxSAT scheduler job accounting must match aggregate jobs to job-kind/status counters, \
                     got jobs={} candidate_set={} search={} encoded_search={} unknown={} timeout={}",
                    self.gpu_maxsat_scheduler_jobs,
                    self.gpu_maxsat_scheduler_candidate_set_jobs,
                    self.gpu_maxsat_scheduler_search_jobs,
                    self.gpu_maxsat_scheduler_encoded_search_jobs,
                    self.gpu_maxsat_scheduler_unknown_status_jobs,
                    self.gpu_maxsat_scheduler_timeout_status_jobs
                ),
            });
        }
        Ok(())
    }

    fn require_portfolio_job_accounting(&self) -> Result<()> {
        let job_kind_events = self.checked_portfolio_job_kind_events()?;
        if self.gpu_portfolio_jobs != job_kind_events {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "portfolio job accounting must match aggregate jobs to job-kind counters, \
                     got jobs={} sat={} maxsat={} unknown={} timeout={}",
                    self.gpu_portfolio_jobs,
                    self.gpu_portfolio_sat_jobs,
                    self.gpu_portfolio_maxsat_jobs,
                    self.gpu_portfolio_unknown_status_jobs,
                    self.gpu_portfolio_timeout_status_jobs
                ),
            });
        }
        Ok(())
    }

    fn require_encoded_maxsat_upload_transfer_accounting(&self) -> Result<()> {
        if (self.gpu_maxsat_candidate_cnf_data_plane_htod_bytes != 0
            && self.gpu_maxsat_candidate_cnf_data_plane_htod_calls == 0)
            || (self.gpu_maxsat_candidate_cnf_launch_metadata_htod_bytes != 0
                && self.gpu_maxsat_candidate_cnf_launch_metadata_htod_calls == 0)
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "encoded MaxSAT candidate CNF upload bytes require matching H2D calls, \
                     got data_plane_calls={} data_plane_bytes={} launch_metadata_calls={} \
                     launch_metadata_bytes={}",
                    self.gpu_maxsat_candidate_cnf_data_plane_htod_calls,
                    self.gpu_maxsat_candidate_cnf_data_plane_htod_bytes,
                    self.gpu_maxsat_candidate_cnf_launch_metadata_htod_calls,
                    self.gpu_maxsat_candidate_cnf_launch_metadata_htod_bytes
                ),
            });
        }
        if (self.gpu_maxsat_candidate_cnf_data_plane_htod_calls != 0
            && self.gpu_maxsat_candidate_cnf_data_plane_htod_bytes == 0)
            || (self.gpu_maxsat_candidate_cnf_launch_metadata_htod_calls != 0
                && self.gpu_maxsat_candidate_cnf_launch_metadata_htod_bytes == 0)
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "encoded MaxSAT candidate CNF upload calls require matching H2D bytes, \
                     got data_plane_calls={} data_plane_bytes={} launch_metadata_calls={} \
                     launch_metadata_bytes={}",
                    self.gpu_maxsat_candidate_cnf_data_plane_htod_calls,
                    self.gpu_maxsat_candidate_cnf_data_plane_htod_bytes,
                    self.gpu_maxsat_candidate_cnf_launch_metadata_htod_calls,
                    self.gpu_maxsat_candidate_cnf_launch_metadata_htod_bytes
                ),
            });
        }
        Ok(())
    }

    /// Require that no CPU search counters were used by the production adapter.
    pub fn require_zero_cpu_search(&self) -> Result<()> {
        if self.cpu_assignment_enumerations != 0 || self.cpu_maxsat_enumerations != 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production adapter".to_string(),
                context: format!(
                    "CPU solver search counters must be zero, got assignment={} maxsat={}",
                    self.cpu_assignment_enumerations, self.cpu_maxsat_enumerations
                ),
            });
        }
        if self.cpu_learned_clause_transfers != 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production adapter".to_string(),
                context: format!(
                    "CPU learned-clause transfers must be zero, got {}",
                    self.cpu_learned_clause_transfers
                ),
            });
        }
        Ok(())
    }

    /// Require internally consistent GPU tuple-membership evidence counters.
    pub fn require_accepted_gpu_tuple_membership_trace(&self) -> Result<()> {
        if self.accepted_nonzero_arity_gpu_candidate_evidence_consumed == 0
            && self.accepted_gpu_candidate_tuple_key_column_reads_consumed != 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "accepted tuple-key reads require accepted nonzero-arity GPU evidence, got \
                     nonzero_evidence=0 tuple_key_reads={}",
                    self.accepted_gpu_candidate_tuple_key_column_reads_consumed
                ),
            });
        }
        if self.accepted_nonzero_arity_gpu_candidate_evidence_consumed > 0
            && self.accepted_gpu_candidate_tuple_key_column_reads_consumed == 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "accepted nonzero-arity GPU solver evidence requires tuple-key device column \
                     reads, got nonzero_evidence={} tuple_key_reads=0",
                    self.accepted_nonzero_arity_gpu_candidate_evidence_consumed
                ),
            });
        }
        if self.accepted_gpu_final_tuple_negated_row_filters_consumed
            > self.accepted_gpu_final_tuple_row_filters_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "accepted negated final-tuple row filters cannot exceed total row filters: \
                     negated={} total={}",
                    self.accepted_gpu_final_tuple_negated_row_filters_consumed,
                    self.accepted_gpu_final_tuple_row_filters_consumed
                ),
            });
        }
        if self.accepted_gpu_final_tuple_row_filters_consumed == 0
            && (self.accepted_gpu_row_specific_membership_row_capacity_consumed != 0
                || self.accepted_gpu_row_filter_fallback_row_capacity_consumed != 0)
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "accepted row-specific/fallback tuple capacity requires accepted GPU row \
                     filters, got row_filters=0 row_specific_capacity={} fallback_capacity={}",
                    self.accepted_gpu_row_specific_membership_row_capacity_consumed,
                    self.accepted_gpu_row_filter_fallback_row_capacity_consumed
                ),
            });
        }
        if self.accepted_gpu_final_tuple_row_filters_consumed > 0
            && self.accepted_gpu_row_specific_membership_row_capacity_consumed == 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "accepted GPU final-tuple row filters require row-specific model-slot \
                     capacity, got row_filters={} row_specific_capacity=0",
                    self.accepted_gpu_final_tuple_row_filters_consumed
                ),
            });
        }
        if self.accepted_gpu_constraint_row_count_device_reads_consumed
            > self.accepted_gpu_constraint_relations_checked_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "accepted constraint row-count device reads cannot exceed checked reduced \
                     constraint relations, got reads={} checked={}",
                    self.accepted_gpu_constraint_row_count_device_reads_consumed,
                    self.accepted_gpu_constraint_relations_checked_consumed
                ),
            });
        }
        Ok(())
    }

    /// Require internally consistent accepted GPU solver evidence counters.
    pub fn require_accepted_gpu_candidate_evidence_trace(&self) -> Result<()> {
        let mode_count = self
            .accepted_g91_gpu_candidate_evidence_consumed
            .checked_add(self.accepted_faeel_gpu_candidate_evidence_consumed)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: "accepted GPU solver mode counters overflowed".to_string(),
            })?;
        if self.accepted_gpu_candidate_evidence_consumed != 0
            && mode_count != self.accepted_gpu_candidate_evidence_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "accepted GPU solver evidence must be classified by epistemic mode, got \
                     evidence={} g91={} faeel={}",
                    self.accepted_gpu_candidate_evidence_consumed,
                    self.accepted_g91_gpu_candidate_evidence_consumed,
                    self.accepted_faeel_gpu_candidate_evidence_consumed
                ),
            });
        }
        if self.accepted_know_gpu_candidate_evidence_consumed
            > self.accepted_gpu_candidate_evidence_consumed
            || self.accepted_possible_gpu_candidate_evidence_consumed
                > self.accepted_gpu_candidate_evidence_consumed
            || self.accepted_not_possible_gpu_candidate_evidence_consumed
                > self.accepted_gpu_candidate_evidence_consumed
            || self.accepted_not_know_gpu_candidate_evidence_consumed
                > self.accepted_gpu_candidate_evidence_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "accepted GPU solver operator evidence counters cannot exceed accepted \
                     evidence records, got evidence={} know={} possible={} not_possible={} \
                     not_know={}",
                    self.accepted_gpu_candidate_evidence_consumed,
                    self.accepted_know_gpu_candidate_evidence_consumed,
                    self.accepted_possible_gpu_candidate_evidence_consumed,
                    self.accepted_not_possible_gpu_candidate_evidence_consumed,
                    self.accepted_not_know_gpu_candidate_evidence_consumed
                ),
            });
        }
        if self.accepted_gpu_candidate_evidence_consumed != 0 {
            if self.accepted_gpu_candidate_state_transitions
                != self.accepted_gpu_world_view_state_transitions
            {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production metric gate".to_string(),
                    context: format!(
                        "accepted GPU candidate/world-view state transitions must match, got \
                         candidates={} world_views={}",
                        self.accepted_gpu_candidate_state_transitions,
                        self.accepted_gpu_world_view_state_transitions
                    ),
                });
            }
            if self.accepted_gpu_candidate_state_transitions == 0
                || self.accepted_gpu_world_view_state_transitions == 0
                || self.accepted_gpu_candidate_final_output_rows_consumed == 0
            {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production metric gate".to_string(),
                    context: format!(
                        "accepted GPU solver detailed evidence requires accepted \
                         candidate/world-view states and non-empty final output rows, got \
                         evidence={} candidate_states={} world_view_states={} final_rows={}",
                        self.accepted_gpu_candidate_evidence_consumed,
                        self.accepted_gpu_candidate_state_transitions,
                        self.accepted_gpu_world_view_state_transitions,
                        self.accepted_gpu_candidate_final_output_rows_consumed
                    ),
                });
            }
            if self.accepted_solver_assumption_bindings_consumed
                < self.accepted_gpu_candidate_evidence_consumed
            {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production metric gate".to_string(),
                    context: format!(
                        "accepted GPU solver evidence requires planner-exported assumption \
                         bindings, got evidence={} assumption_bindings={}",
                        self.accepted_gpu_candidate_evidence_consumed,
                        self.accepted_solver_assumption_bindings_consumed
                    ),
                });
            }
            let required_capability_floor = self
                .accepted_gpu_candidate_evidence_consumed
                .checked_mul(PRODUCTION_SOLVER_REQUIRED_CAPABILITY_COUNT)
                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production metric gate".to_string(),
                    context: "accepted GPU solver capability floor overflowed".to_string(),
                })?;
            let required_status_floor = self
                .accepted_gpu_candidate_evidence_consumed
                .checked_mul(PRODUCTION_SOLVER_REQUIRED_STATUS_COUNT)
                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production metric gate".to_string(),
                    context: "accepted GPU solver status floor overflowed".to_string(),
                })?;
            if self.accepted_solver_required_capabilities_consumed < required_capability_floor
                || self.accepted_solver_required_statuses_consumed < required_status_floor
            {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production metric gate".to_string(),
                    context: format!(
                        "accepted GPU solver evidence requires the v0.9 production \
                         capability/status contract, got evidence={} capabilities={} statuses={}",
                        self.accepted_gpu_candidate_evidence_consumed,
                        self.accepted_solver_required_capabilities_consumed,
                        self.accepted_solver_required_statuses_consumed
                    ),
                });
            }
            if self.accepted_gpu_candidate_state_transitions
                < self.accepted_gpu_candidate_evidence_consumed
                || self.accepted_gpu_world_view_state_transitions
                    < self.accepted_gpu_candidate_evidence_consumed
                || self.accepted_gpu_candidate_final_output_rows_consumed
                    < self.accepted_gpu_candidate_evidence_consumed
            {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production metric gate".to_string(),
                    context: format!(
                        "accepted GPU solver state counters must cover each accepted evidence \
                         record, got evidence={} candidate_states={} world_view_states={} \
                         final_rows={}",
                        self.accepted_gpu_candidate_evidence_consumed,
                        self.accepted_gpu_candidate_state_transitions,
                        self.accepted_gpu_world_view_state_transitions,
                        self.accepted_gpu_candidate_final_output_rows_consumed
                    ),
                });
            }
        }
        if self.accepted_nonzero_arity_gpu_candidate_evidence_consumed
            > self.accepted_gpu_candidate_evidence_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "accepted nonzero-arity GPU solver evidence cannot exceed accepted evidence \
                     records, got nonzero={} evidence={}",
                    self.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
                    self.accepted_gpu_candidate_evidence_consumed
                ),
            });
        }
        if self.accepted_gpu_batch_candidate_component_evidence_consumed
            < self.accepted_gpu_batch_candidate_evidence_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "accepted GPU batch component evidence must cover accepted batch evidence, \
                     got batches={} components={}",
                    self.accepted_gpu_batch_candidate_evidence_consumed,
                    self.accepted_gpu_batch_candidate_component_evidence_consumed
                ),
            });
        }
        if self.accepted_gpu_batch_candidate_component_evidence_consumed
            > self.accepted_gpu_candidate_evidence_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "accepted GPU batch component evidence cannot exceed accepted candidate \
                     evidence, got components={} evidence={}",
                    self.accepted_gpu_batch_candidate_component_evidence_consumed,
                    self.accepted_gpu_candidate_evidence_consumed
                ),
            });
        }
        Ok(())
    }

    /// Require that this trace is eligible for v0.9 production solver metrics.
    ///
    /// This is an accepted-path containment gate, not a release-close claim:
    /// the CPU semantic-oracle facade may still exist for fixtures, but it
    /// cannot satisfy production metric evidence.
    pub fn require_production_metric_eligibility(&self) -> Result<()> {
        let capabilities = production_capabilities();
        if capabilities.cpu_oracle_solver_allowed {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: "CPU semantic-oracle solver is not allowed for production metrics"
                    .to_string(),
            });
        }
        if capabilities.gpu_cdcl_sat_unsat != GpuSolverProductionCapabilityStatus::Available {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: "GPU CDCL SAT/UNSAT production capability is not available".to_string(),
            });
        }
        if capabilities.gpu_maxsat != GpuSolverProductionCapabilityStatus::Available {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: capabilities.gpu_maxsat_blocker.to_string(),
            });
        }
        if capabilities.gpu_portfolio_sat_maxsat != GpuSolverProductionCapabilityStatus::Available {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: capabilities.gpu_portfolio_blocker.to_string(),
            });
        }
        if self.accepted_gpu_candidate_evidence_consumed == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: "production solver metrics require accepted GPU candidate evidence"
                    .to_string(),
            });
        }
        let gpu_solver_production_path_events = self.checked_gpu_solver_production_path_events()?;
        let gpu_solver_status_path_events = self.checked_gpu_solver_status_path_events()?;
        let gpu_solver_path_events = Self::checked_production_event_sum(
            "gpu_solver_path_events",
            &[
                gpu_solver_production_path_events,
                gpu_solver_status_path_events,
            ],
        )?;
        if gpu_solver_path_events == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context:
                    "production solver metrics require an existing GPU CDCL/MaxSAT/scheduler/portfolio/status counter"
                        .to_string(),
            });
        }
        if self.accepted_gpu_solver_production_path_events == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: "production solver metrics require GPU solver production/status work inside an accepted epistemic evidence gate"
                    .to_string(),
            });
        }
        if self.accepted_gpu_solver_production_path_events > gpu_solver_path_events {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "accepted GPU solver production/status events cannot exceed total GPU solver production/status events: accepted={} total={}",
                    self.accepted_gpu_solver_production_path_events, gpu_solver_path_events
                ),
            });
        }
        if self.accepted_gpu_solver_production_path_events
            < self.accepted_gpu_candidate_state_transitions
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: format!(
                    "accepted GPU solver production events must cover each accepted candidate \
                     state transition, got accepted_events={} candidate_states={}",
                    self.accepted_gpu_solver_production_path_events,
                    self.accepted_gpu_candidate_state_transitions
                ),
            });
        }
        self.require_maxsat_scheduler_job_accounting()?;
        self.require_portfolio_job_accounting()?;
        let maxsat_metric_events = self.checked_maxsat_production_metric_events()?;
        if maxsat_metric_events != 0 {
            if self.gpu_maxsat_candidate_solves == 0 {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production metric gate".to_string(),
                    context: "MaxSAT production metrics require GPU CDCL candidate solves"
                        .to_string(),
                });
            }
            if self.gpu_maxsat_frontier_certified_candidate_solves
                != self.gpu_maxsat_candidate_solves
            {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production metric gate".to_string(),
                    context: format!(
                        "MaxSAT production metrics require every candidate solve to be covered by \
                         an encoded weighted MaxSAT upper-bound certificate, got certified_solves={} \
                         candidate_solves={}",
                        self.gpu_maxsat_frontier_certified_candidate_solves,
                        self.gpu_maxsat_candidate_solves
                    ),
                });
            }
            if self.gpu_maxsat_candidate_encodes > self.gpu_maxsat_candidate_solves {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production metric gate".to_string(),
                    context: format!(
                        "encoded MaxSAT candidates cannot exceed GPU CDCL candidate solves, \
                         got encodes={} solves={}",
                        self.gpu_maxsat_candidate_encodes, self.gpu_maxsat_candidate_solves
                    ),
                });
            }
            if self.gpu_maxsat_unsat_candidate_prunes > self.gpu_maxsat_candidate_solves {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production metric gate".to_string(),
                    context: format!(
                        "MaxSAT UNSAT candidate prunes cannot exceed GPU CDCL candidate solves, \
                         got prunes={} solves={}",
                        self.gpu_maxsat_unsat_candidate_prunes, self.gpu_maxsat_candidate_solves
                    ),
                });
            }
            if self.gpu_maxsat_optima > self.gpu_maxsat_candidate_solves {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production metric gate".to_string(),
                    context: format!(
                        "MaxSAT optima cannot exceed GPU CDCL candidate solves, got optima={} \
                         solves={}",
                        self.gpu_maxsat_optima, self.gpu_maxsat_candidate_solves
                    ),
                });
            }
            let encoded_maxsat_metric_events = Self::checked_production_event_sum(
                "gpu_solver_encoded_maxsat_metric_events",
                &[
                    self.gpu_maxsat_frontier_certified_candidate_solves,
                    self.gpu_maxsat_candidate_encodes,
                    self.gpu_maxsat_frontier_upper_bound_certificates,
                    self.gpu_maxsat_frontier_completion_candidate_encodes,
                    self.gpu_maxsat_scheduler_encoded_search_jobs,
                ],
            )?;
            let encoded_maxsat_upload_metrics = self.gpu_maxsat_candidate_cnf_data_plane_htod_calls
                != 0
                || self.gpu_maxsat_candidate_cnf_data_plane_htod_bytes != 0
                || self.gpu_maxsat_candidate_cnf_data_plane_dtoh_calls != 0
                || self.gpu_maxsat_candidate_cnf_data_plane_dtoh_bytes != 0
                || self.gpu_maxsat_candidate_cnf_launch_metadata_htod_calls != 0
                || self.gpu_maxsat_candidate_cnf_launch_metadata_htod_bytes != 0;
            if encoded_maxsat_metric_events != 0 || encoded_maxsat_upload_metrics {
                if self.gpu_maxsat_frontier_upper_bound_certificates == 0 {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production metric gate".to_string(),
                        context: "encoded MaxSAT production metrics require a weighted MaxSAT upper-bound certificate"
                            .to_string(),
                    });
                }
                if self.gpu_maxsat_candidate_encodes == 0 {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production metric gate".to_string(),
                        context:
                            "MaxSAT upper-bound certificates require encoded weighted MaxSAT candidates"
                                .to_string(),
                    });
                }
                let max_data_plane_htod_calls = self
                    .gpu_maxsat_candidate_encodes
                    .checked_mul(2)
                    .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production metric gate".to_string(),
                        context: "MaxSAT candidate data-plane H2D call budget overflowed"
                            .to_string(),
                    })?;
                let max_launch_metadata_htod_calls = self
                    .gpu_maxsat_candidate_encodes
                    .checked_mul(3)
                    .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production metric gate".to_string(),
                        context: "MaxSAT candidate launch-metadata H2D call budget overflowed"
                            .to_string(),
                    })?;
                if self.gpu_maxsat_candidate_cnf_data_plane_htod_calls > max_data_plane_htod_calls
                    || self.gpu_maxsat_candidate_cnf_launch_metadata_htod_calls
                        > max_launch_metadata_htod_calls
                {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production metric gate".to_string(),
                        context: format!(
                            "encoded MaxSAT candidate CNF uploads exceeded bounded H2D call budget, \
                             encodes={} data_plane_calls={}/{} launch_metadata_calls={}/{}",
                            self.gpu_maxsat_candidate_encodes,
                            self.gpu_maxsat_candidate_cnf_data_plane_htod_calls,
                            max_data_plane_htod_calls,
                            self.gpu_maxsat_candidate_cnf_launch_metadata_htod_calls,
                            max_launch_metadata_htod_calls
                        ),
                    });
                }
                self.require_encoded_maxsat_upload_transfer_accounting()?;
                if self.gpu_maxsat_candidate_cnf_data_plane_dtoh_calls != 0
                    || self.gpu_maxsat_candidate_cnf_data_plane_dtoh_bytes != 0
                {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production metric gate".to_string(),
                        context: format!(
                            "encoded MaxSAT candidate CNF uploads must not perform data-plane D2H \
                             transfers, got calls={} bytes={}",
                            self.gpu_maxsat_candidate_cnf_data_plane_dtoh_calls,
                            self.gpu_maxsat_candidate_cnf_data_plane_dtoh_bytes
                        ),
                    });
                }
                if self.gpu_maxsat_frontier_completion_candidate_encodes
                    > self.gpu_maxsat_candidate_encodes
                {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production metric gate".to_string(),
                        context: format!(
                            "MaxSAT frontier completion candidates cannot exceed encoded candidates, \
                             got completion_encodes={} total_encodes={}",
                            self.gpu_maxsat_frontier_completion_candidate_encodes,
                            self.gpu_maxsat_candidate_encodes
                        ),
                    });
                }
                if self.gpu_maxsat_frontier_certified_candidate_solves
                    != self.gpu_maxsat_candidate_encodes
                {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production metric gate".to_string(),
                        context: format!(
                            "encoded MaxSAT production metrics require every encoded candidate solve \
                             to be covered by an upper-bound-certified frontier, got certified_solves={} \
                             encoded_candidates={}",
                            self.gpu_maxsat_frontier_certified_candidate_solves,
                            self.gpu_maxsat_candidate_encodes
                        ),
                    });
                }
                if self.gpu_maxsat_frontier_upper_bound_certificates
                    < self.gpu_maxsat_scheduler_encoded_search_jobs
                {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production metric gate".to_string(),
                        context: format!(
                            "encoded MaxSAT scheduler jobs require one upper-bound certificate per job, got certificates={} encoded_jobs={}",
                            self.gpu_maxsat_frontier_upper_bound_certificates,
                            self.gpu_maxsat_scheduler_encoded_search_jobs
                        ),
                    });
                }
            }
            if self.gpu_maxsat_optima == 0 {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production metric gate".to_string(),
                    context: "MaxSAT production metrics require a GPU-certified optimum"
                        .to_string(),
                });
            }
        }
        self.require_accepted_gpu_candidate_evidence_trace()?;
        self.require_accepted_gpu_tuple_membership_trace()?;
        self.require_zero_cpu_search()
    }
}

/// Thin adapter from epistemic solver work to the existing GPU CDCL verifier.
pub struct GpuSolverProductionAdapter {
    provider: Arc<CudaKernelProvider>,
    solver: GpuCdclSolver,
    trace: GpuSolverProductionTrace,
}

impl GpuSolverProductionAdapter {
    /// Create an adapter over the existing GPU CDCL solver implementation.
    pub fn new(provider: Arc<CudaKernelProvider>, config: GpuCdclConfig) -> Self {
        Self {
            solver: GpuCdclSolver::new(Arc::clone(&provider), config),
            provider,
            trace: GpuSolverProductionTrace {
                cpu_assignment_enumerations: 0,
                cpu_maxsat_enumerations: 0,
                cpu_learned_clause_transfers: 0,
                ..GpuSolverProductionTrace::default()
            },
        }
    }

    /// Return the current production-path trace counters.
    pub fn trace(&self) -> GpuSolverProductionTrace {
        self.trace
    }

    fn checked_trace_counter_add(current: u64, delta: u64, counter: &str) -> Result<u64> {
        current
            .checked_add(delta)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production trace accounting".to_string(),
                context: format!(
                    "accepted GPU solver trace counter {counter} overflowed while adding {delta} \
                     to {current}"
                ),
            })
    }

    fn checked_report_counter_add(current: u64, delta: u64, counter: &str) -> Result<u64> {
        current
            .checked_add(delta)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production report accounting".to_string(),
                context: format!(
                    "accepted GPU solver report counter {counter} overflowed while adding {delta} \
                     to {current}"
                ),
            })
    }

    fn checked_report_counter_delta(current: u64, before: u64, counter: &str) -> Result<u64> {
        current
            .checked_sub(before)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production report accounting".to_string(),
                context: format!(
                    "accepted GPU solver report counter {counter} decreased from {before} to \
                     {current}"
                ),
            })
    }

    fn checked_workspace_clause_cap(weighted: &SolveInstance) -> Result<u32> {
        u32::try_from(weighted.clauses.len()).map_err(|_| {
            XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT encoding".to_string(),
                context: format!(
                    "weighted MaxSAT clause count {} exceeds GPU CDCL workspace capacity range",
                    weighted.clauses.len()
                ),
            }
        })
    }

    fn require_adapter_provider_identity(&self, provider: &CudaKernelProvider) -> Result<()> {
        let adapter_identity = EpistemicGpuProviderIdentity::from_provider(&self.provider);
        let evidence_identity = EpistemicGpuProviderIdentity::from_provider(provider);
        if adapter_identity != evidence_identity {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU solver candidate evidence".to_string(),
                context: format!(
                    "solver adapter provider mismatch: adapter device={} evidence device={} \
                     adapter_device_ptr={} evidence_device_ptr={} adapter_memory_ptr={} \
                     evidence_memory_ptr={}",
                    adapter_identity.device_ordinal,
                    evidence_identity.device_ordinal,
                    adapter_identity.device_ptr,
                    evidence_identity.device_ptr,
                    adapter_identity.memory_ptr,
                    evidence_identity.memory_ptr
                ),
            });
        }
        Ok(())
    }

    fn require_accepted_gpu_solver_evidence(
        &self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
    ) -> Result<GpuSolverAcceptedCandidateState> {
        self.require_adapter_provider_identity(provider)?;
        require_accepted_gpu_solver_evidence(provider, result)
    }

    fn require_accepted_gpu_solver_states(
        &self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
    ) -> Result<Vec<GpuSolverAcceptedCandidateState>> {
        self.require_adapter_provider_identity(provider)?;
        require_accepted_gpu_solver_states(provider, results)
    }

    fn require_branch_var_limit_on_adapter_provider(
        &self,
        branch_var_limit: &TrackedCudaSlice<u32>,
        construct: &'static str,
    ) -> Result<()> {
        if branch_var_limit.len() != 1 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "solver lifecycle branch_var_limit must have len=1, got {}",
                    branch_var_limit.len()
                ),
            });
        }
        let expected_memory =
            EpistemicGpuProviderIdentity::from_provider(&self.provider).memory_ptr;
        let actual_memory = branch_var_limit.memory_manager_ptr_value();
        if actual_memory != expected_memory {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "solver lifecycle branch_var_limit belongs to memory manager {actual_memory}, expected {expected_memory}"
                ),
            });
        }
        Ok(())
    }

    fn require_solver_artifact_on_adapter_provider(
        &self,
        cnf: &GpuCnf,
        branch_var_limit: &TrackedCudaSlice<u32>,
        construct: &'static str,
    ) -> Result<()> {
        cnf.require_provider_memory(&self.provider, construct)?;
        self.require_branch_var_limit_on_adapter_provider(branch_var_limit, construct)
    }

    fn require_cnf_on_adapter_provider(&self, cnf: &GpuCnf, construct: &'static str) -> Result<()> {
        cnf.require_provider_memory(&self.provider, construct)
    }

    fn require_workspace_capacity_for_cnf(
        &self,
        workspace: &GpuCdclWorkspace,
        cnf: &GpuCnf,
        construct: &'static str,
    ) -> Result<()> {
        self.solver
            .require_workspace_capacity_for_cnf(workspace, cnf.var_cap, cnf.clause_cap)
            .map_err(|err| XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!("GPU CDCL workspace capacity rejected solver artifact: {err}"),
            })
    }

    fn require_workspace_capacity_for_weighted_maxsat_encoding(
        &self,
        workspace: &GpuCdclWorkspace,
        weighted: &SolveInstance,
        construct: &'static str,
    ) -> Result<()> {
        self.solver
            .require_workspace_capacity_for_cnf(
                workspace,
                weighted.num_vars,
                Self::checked_workspace_clause_cap(weighted)?,
            )
            .map_err(|err| XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!("GPU CDCL workspace capacity rejected MaxSAT encoding: {err}"),
            })
    }

    fn require_assumption_lifecycle_step_artifacts(
        &self,
        workspace: &GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
    ) -> Result<()> {
        for step in steps {
            self.require_solver_artifact_on_adapter_provider(
                step.cnf,
                step.branch_var_limit,
                "GPU solver production lifecycle",
            )?;
            if matches!(step.expectation, GpuSolverProductionExpectation::Unsat) {
                self.require_workspace_capacity_for_cnf(
                    workspace,
                    step.cnf,
                    "GPU solver production lifecycle",
                )?;
            }
        }
        Ok(())
    }

    fn require_weighted_maxsat_candidate_artifacts(
        &self,
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<()> {
        for candidate in candidates {
            self.require_solver_artifact_on_adapter_provider(
                candidate.cnf,
                candidate.branch_var_limit,
                "GPU solver production MaxSAT",
            )?;
        }
        Ok(())
    }

    fn require_weighted_maxsat_candidates_and_artifacts(
        &self,
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<()> {
        Self::require_weighted_maxsat_candidates(candidates)?;
        self.require_weighted_maxsat_candidate_artifacts(candidates)
    }

    fn require_maxsat_lifecycle_artifacts(
        &self,
        workspace: &GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<()> {
        self.require_assumption_lifecycle_step_artifacts(workspace, steps)?;
        self.require_weighted_maxsat_candidate_artifacts(candidates)
    }

    fn require_weighted_maxsat_search_candidate_artifacts(
        &self,
        candidates: &[GpuSolverProductionMaxSatSearchCandidate<'_>],
    ) -> Result<()> {
        for candidate in candidates {
            self.require_solver_artifact_on_adapter_provider(
                candidate.cnf,
                candidate.branch_var_limit,
                "GPU solver production MaxSAT search",
            )?;
        }
        Ok(())
    }

    fn require_weighted_maxsat_search_candidates_and_artifacts(
        &self,
        workspace: &GpuCdclWorkspace,
        candidates: &[GpuSolverProductionMaxSatSearchCandidate<'_>],
    ) -> Result<()> {
        Self::require_weighted_maxsat_search_candidates(candidates)?;
        self.require_weighted_maxsat_search_candidate_artifacts(candidates)?;
        for candidate in candidates {
            if matches!(
                candidate.status,
                GpuSolverProductionMaxSatSearchStatus::Unsatisfiable
            ) {
                self.require_workspace_capacity_for_cnf(
                    workspace,
                    candidate.cnf,
                    "GPU solver production MaxSAT search",
                )?;
            }
        }
        Ok(())
    }

    fn require_learned_clause_publication_artifacts(
        &self,
        workspace: &GpuCdclWorkspace,
        cnf: &GpuCnf,
        branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        self.require_solver_artifact_on_adapter_provider(
            cnf,
            branch_var_limit,
            "GPU solver learned-clause arena",
        )?;
        self.require_workspace_capacity_for_cnf(workspace, cnf, "GPU solver learned-clause arena")
    }

    fn require_learned_clause_reuse_artifacts(
        &self,
        source_cnf: &GpuCnf,
        source_branch_var_limit: &TrackedCudaSlice<u32>,
        target_cnf: &GpuCnf,
        target_branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        self.require_solver_artifact_on_adapter_provider(
            source_cnf,
            source_branch_var_limit,
            "GPU solver learned-clause reuse",
        )?;
        self.require_solver_artifact_on_adapter_provider(
            target_cnf,
            target_branch_var_limit,
            "GPU solver learned-clause reuse",
        )
    }

    fn require_learned_clause_reuse_inputs(
        &mut self,
        workspace: &GpuCdclWorkspace,
        source_cnf: &GpuCnf,
        source_branch_var_limit: &TrackedCudaSlice<u32>,
        target_cnf: &GpuCnf,
        target_branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        self.require_learned_clause_reuse_artifacts(
            source_cnf,
            source_branch_var_limit,
            target_cnf,
            target_branch_var_limit,
        )?;
        if let Err(err) = require_same_gpu_cnf_for_learned_clause_reuse(source_cnf, target_cnf) {
            checked_solver_trace_counter_inc!(self, gpu_learned_clause_reuse_rejections);
            self.trace.require_zero_cpu_search()?;
            return Err(err);
        }
        self.require_workspace_capacity_for_cnf(
            workspace,
            source_cnf,
            "GPU solver learned-clause reuse",
        )?;
        self.require_workspace_capacity_for_cnf(
            workspace,
            target_cnf,
            "GPU solver learned-clause reuse",
        )?;
        Ok(())
    }

    fn require_weighted_maxsat_encoded_search_inputs_and_artifacts(
        &self,
        workspace: &GpuCdclWorkspace,
        weighted: &SolveInstance,
        branch_var_limit: &TrackedCudaSlice<u32>,
        selections: &[GpuSolverProductionWeightedMaxSatSelection<'_>],
    ) -> Result<()> {
        Self::require_weighted_maxsat_encoding_inputs(weighted, selections)?;
        self.require_workspace_capacity_for_weighted_maxsat_encoding(
            workspace,
            weighted,
            "GPU solver production MaxSAT encoding",
        )?;
        self.require_branch_var_limit_on_adapter_provider(
            branch_var_limit,
            "GPU solver production MaxSAT encoding",
        )
    }

    /// Validate accepted GPU epistemic evidence and return the solver-facing candidate state.
    pub fn accepted_candidate_state(
        &self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
    ) -> Result<GpuSolverAcceptedCandidateState> {
        self.require_accepted_gpu_solver_evidence(provider, result)
    }

    fn accepted_solver_results_from_gpu_batch_execution_evidence<'a>(
        &mut self,
        provider: &CudaKernelProvider,
        evidence: GpuSolverProductionBatchExecutionEvidence<'a>,
    ) -> Result<Vec<&'a EpistemicGpuExecutionResult>> {
        self.require_adapter_provider_identity(provider)?;
        require_accepted_gpu_solver_batch_evidence(provider, evidence.batch)
    }

    fn record_accepted_gpu_batch_candidate_evidence(
        &mut self,
        component_count: usize,
    ) -> Result<()> {
        self.trace.accepted_gpu_batch_candidate_evidence_consumed =
            Self::checked_trace_counter_add(
                self.trace.accepted_gpu_batch_candidate_evidence_consumed,
                1,
                "accepted_gpu_batch_candidate_evidence_consumed",
            )?;
        self.trace
            .accepted_gpu_batch_candidate_component_evidence_consumed =
            Self::checked_trace_counter_add(
                self.trace
                    .accepted_gpu_batch_candidate_component_evidence_consumed,
                component_count as u64,
                "accepted_gpu_batch_candidate_component_evidence_consumed",
            )?;
        Ok(())
    }

    fn with_trace_rollback<T>(&mut self, action: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let trace_before = self.trace;
        match action(self) {
            Ok(value) => Ok(value),
            Err(err) => {
                self.trace = trace_before;
                Err(err)
            }
        }
    }

    fn with_trace_rollback_preserving_reuse_rejections<T>(
        &mut self,
        action: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let trace_before = self.trace;
        match action(self) {
            Ok(value) => Ok(value),
            Err(err) => {
                let reuse_rejections_after = self.trace.gpu_learned_clause_reuse_rejections;
                let reuse_rejections_before = trace_before.gpu_learned_clause_reuse_rejections;
                self.trace = trace_before;
                let reuse_rejection_delta = Self::checked_report_counter_delta(
                    reuse_rejections_after,
                    reuse_rejections_before,
                    "gpu_learned_clause_reuse_rejections",
                )?;
                self.trace.gpu_learned_clause_reuse_rejections = Self::checked_trace_counter_add(
                    self.trace.gpu_learned_clause_reuse_rejections,
                    reuse_rejection_delta,
                    "gpu_learned_clause_reuse_rejections",
                )?;
                Err(err)
            }
        }
    }

    /// Allocate a reusable GPU CDCL workspace through the existing solver.
    pub fn new_workspace(&self, max_var_cap: u32, max_clause_cap: u32) -> Result<GpuCdclWorkspace> {
        self.solver.new_workspace(max_var_cap, max_clause_cap)
    }

    fn require_workspace_on_adapter_provider(&self, workspace: &GpuCdclWorkspace) -> Result<()> {
        self.solver.require_workspace_on_provider(workspace)
    }

    fn record_accepted_gpu_candidate_state(
        &mut self,
        state: &GpuSolverAcceptedCandidateState,
    ) -> Result<()> {
        self.trace.accepted_gpu_candidate_evidence_consumed = Self::checked_trace_counter_add(
            self.trace.accepted_gpu_candidate_evidence_consumed,
            state.evidence_records,
            "accepted_gpu_candidate_evidence_consumed",
        )?;
        self.trace.accepted_gpu_candidate_state_transitions = Self::checked_trace_counter_add(
            self.trace.accepted_gpu_candidate_state_transitions,
            state.accepted_candidates,
            "accepted_gpu_candidate_state_transitions",
        )?;
        self.trace.accepted_gpu_world_view_state_transitions = Self::checked_trace_counter_add(
            self.trace.accepted_gpu_world_view_state_transitions,
            state.accepted_world_views,
            "accepted_gpu_world_view_state_transitions",
        )?;
        self.trace.accepted_gpu_candidate_final_output_rows_consumed =
            Self::checked_trace_counter_add(
                self.trace.accepted_gpu_candidate_final_output_rows_consumed,
                state.final_output_rows,
                "accepted_gpu_candidate_final_output_rows_consumed",
            )?;
        if state.g91_mode {
            self.trace.accepted_g91_gpu_candidate_evidence_consumed =
                Self::checked_trace_counter_add(
                    self.trace.accepted_g91_gpu_candidate_evidence_consumed,
                    1,
                    "accepted_g91_gpu_candidate_evidence_consumed",
                )?;
        }
        if state.faeel_mode {
            self.trace.accepted_faeel_gpu_candidate_evidence_consumed =
                Self::checked_trace_counter_add(
                    self.trace.accepted_faeel_gpu_candidate_evidence_consumed,
                    1,
                    "accepted_faeel_gpu_candidate_evidence_consumed",
                )?;
        }
        if state.has_know_operator {
            self.trace.accepted_know_gpu_candidate_evidence_consumed =
                Self::checked_trace_counter_add(
                    self.trace.accepted_know_gpu_candidate_evidence_consumed,
                    1,
                    "accepted_know_gpu_candidate_evidence_consumed",
                )?;
        }
        if state.has_possible_operator {
            self.trace.accepted_possible_gpu_candidate_evidence_consumed =
                Self::checked_trace_counter_add(
                    self.trace.accepted_possible_gpu_candidate_evidence_consumed,
                    1,
                    "accepted_possible_gpu_candidate_evidence_consumed",
                )?;
        }
        if state.has_not_possible_operator {
            self.trace
                .accepted_not_possible_gpu_candidate_evidence_consumed =
                Self::checked_trace_counter_add(
                    self.trace
                        .accepted_not_possible_gpu_candidate_evidence_consumed,
                    1,
                    "accepted_not_possible_gpu_candidate_evidence_consumed",
                )?;
        }
        if state.has_not_know_operator {
            self.trace.accepted_not_know_gpu_candidate_evidence_consumed =
                Self::checked_trace_counter_add(
                    self.trace.accepted_not_know_gpu_candidate_evidence_consumed,
                    1,
                    "accepted_not_know_gpu_candidate_evidence_consumed",
                )?;
        }
        if state.has_nonzero_arity_tuple_keys {
            self.trace
                .accepted_nonzero_arity_gpu_candidate_evidence_consumed =
                Self::checked_trace_counter_add(
                    self.trace
                        .accepted_nonzero_arity_gpu_candidate_evidence_consumed,
                    1,
                    "accepted_nonzero_arity_gpu_candidate_evidence_consumed",
                )?;
        }
        self.trace
            .accepted_gpu_candidate_tuple_key_column_reads_consumed =
            Self::checked_trace_counter_add(
                self.trace
                    .accepted_gpu_candidate_tuple_key_column_reads_consumed,
                state.tuple_key_column_reads,
                "accepted_gpu_candidate_tuple_key_column_reads_consumed",
            )?;
        self.trace.accepted_solver_assumption_bindings_consumed = Self::checked_trace_counter_add(
            self.trace.accepted_solver_assumption_bindings_consumed,
            state.solver_assumption_bindings,
            "accepted_solver_assumption_bindings_consumed",
        )?;
        self.trace.accepted_solver_required_capabilities_consumed =
            Self::checked_trace_counter_add(
                self.trace.accepted_solver_required_capabilities_consumed,
                state.solver_required_capabilities,
                "accepted_solver_required_capabilities_consumed",
            )?;
        self.trace.accepted_solver_required_statuses_consumed = Self::checked_trace_counter_add(
            self.trace.accepted_solver_required_statuses_consumed,
            state.solver_required_statuses,
            "accepted_solver_required_statuses_consumed",
        )?;
        self.trace.accepted_gpu_final_tuple_row_filters_consumed = Self::checked_trace_counter_add(
            self.trace.accepted_gpu_final_tuple_row_filters_consumed,
            state.final_tuple_row_filters,
            "accepted_gpu_final_tuple_row_filters_consumed",
        )?;
        self.trace
            .accepted_gpu_final_tuple_negated_row_filters_consumed =
            Self::checked_trace_counter_add(
                self.trace
                    .accepted_gpu_final_tuple_negated_row_filters_consumed,
                state.final_tuple_negated_row_filters,
                "accepted_gpu_final_tuple_negated_row_filters_consumed",
            )?;
        self.trace
            .accepted_gpu_row_specific_membership_row_capacity_consumed =
            Self::checked_trace_counter_add(
                self.trace
                    .accepted_gpu_row_specific_membership_row_capacity_consumed,
                state.row_specific_membership_row_capacity,
                "accepted_gpu_row_specific_membership_row_capacity_consumed",
            )?;
        self.trace
            .accepted_gpu_row_filter_fallback_row_capacity_consumed =
            Self::checked_trace_counter_add(
                self.trace
                    .accepted_gpu_row_filter_fallback_row_capacity_consumed,
                state.row_filter_fallback_row_capacity,
                "accepted_gpu_row_filter_fallback_row_capacity_consumed",
            )?;
        self.trace
            .accepted_gpu_constraint_relations_checked_consumed = Self::checked_trace_counter_add(
            self.trace
                .accepted_gpu_constraint_relations_checked_consumed,
            state.checked_constraint_relations,
            "accepted_gpu_constraint_relations_checked_consumed",
        )?;
        self.trace
            .accepted_gpu_constraint_row_count_device_reads_consumed =
            Self::checked_trace_counter_add(
                self.trace
                    .accepted_gpu_constraint_row_count_device_reads_consumed,
                state.constraint_row_count_device_reads,
                "accepted_gpu_constraint_row_count_device_reads_consumed",
            )?;
        Ok(())
    }

    fn record_accepted_gpu_solver_production_path_events_since(
        &mut self,
        events_before: GpuSolverAcceptedPathEventSnapshot,
        state: &GpuSolverAcceptedCandidateState,
    ) -> Result<()> {
        let events_after = self.trace.accepted_path_event_snapshot()?;
        let production_delta = events_after
            .production
            .checked_sub(events_before.production)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production trace accounting".to_string(),
                context: format!(
                    "accepted GPU solver production events decreased from {} to {}",
                    events_before.production, events_after.production
                ),
            })?;
        let status_delta = events_after
            .status
            .checked_sub(events_before.status)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production trace accounting".to_string(),
                context: format!(
                    "accepted GPU solver status events decreased from {} to {}",
                    events_before.status, events_after.status
                ),
            })?;
        let accepted_delta = Self::checked_report_counter_add(
            production_delta,
            status_delta,
            "accepted_gpu_solver_path_events",
        )?;
        if accepted_delta < state.accepted_candidates {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production trace accounting".to_string(),
                context: format!(
                    "accepted GPU solver production/status work must cover every accepted \
                     candidate state before evidence is recorded, got production_events={} \
                     status_events={} candidate_states={}",
                    production_delta, status_delta, state.accepted_candidates
                ),
            });
        }
        self.trace.accepted_gpu_solver_production_path_events = Self::checked_trace_counter_add(
            self.trace.accepted_gpu_solver_production_path_events,
            accepted_delta,
            "accepted_gpu_solver_production_path_events",
        )?;
        Ok(())
    }

    /// Solve and enforce SAT entirely on GPU.
    pub fn solve_expect_sat(&mut self, cnf: &GpuCnf) -> Result<TrackedCudaSlice<i8>> {
        self.require_cnf_on_adapter_provider(cnf, "GPU solver production SAT")?;
        let assignment = self.solver.solve_expect_sat(cnf)?;
        checked_solver_trace_counter_inc!(self, gpu_cdcl_sat_solves);
        self.trace.require_zero_cpu_search()?;
        Ok(assignment)
    }

    /// Solve SAT through GPU CDCL after an accepted GPU epistemic execution result.
    pub fn solve_expect_sat_with_gpu_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        cnf: &GpuCnf,
    ) -> Result<TrackedCudaSlice<i8>> {
        self.with_trace_rollback(|this| {
            this.require_cnf_on_adapter_provider(cnf, "GPU solver production SAT")?;
            let state = this.require_accepted_gpu_solver_evidence(provider, result)?;
            let events_before = this.trace.accepted_path_event_snapshot()?;
            let assignment = this.solve_expect_sat(cnf)?;
            this.record_accepted_gpu_solver_production_path_events_since(events_before, &state)?;
            this.record_accepted_gpu_candidate_state(&state)?;
            this.trace.require_zero_cpu_search()?;
            Ok(assignment)
        })
    }

    /// Solve UNSAT through GPU CDCL after an accepted GPU epistemic execution result.
    pub fn solve_expect_unsat_with_gpu_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        cnf: &GpuCnf,
    ) -> Result<()> {
        self.with_trace_rollback(|this| {
            this.require_cnf_on_adapter_provider(cnf, "GPU solver production UNSAT")?;
            let state = this.require_accepted_gpu_solver_evidence(provider, result)?;
            let events_before = this.trace.accepted_path_event_snapshot()?;
            this.solve_expect_unsat(cnf)?;
            this.record_accepted_gpu_solver_production_path_events_since(events_before, &state)?;
            this.record_accepted_gpu_candidate_state(&state)?;
            this.trace.require_zero_cpu_search()
        })
    }

    /// Solve and enforce UNSAT entirely on GPU.
    pub fn solve_expect_unsat(&mut self, cnf: &GpuCnf) -> Result<()> {
        self.require_cnf_on_adapter_provider(cnf, "GPU solver production UNSAT")?;
        self.solver.solve_expect_unsat(cnf)?;
        checked_solver_trace_counter_inc!(self, gpu_cdcl_unsat_solves);
        self.trace.require_zero_cpu_search()
    }

    /// Solve and enforce UNSAT entirely on GPU using a reusable workspace.
    pub fn solve_expect_unsat_with_branch_limit_ws(
        &mut self,
        workspace: &mut GpuCdclWorkspace,
        cnf: &GpuCnf,
        branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        self.require_workspace_on_adapter_provider(workspace)?;
        self.require_solver_artifact_on_adapter_provider(
            cnf,
            branch_var_limit,
            "GPU solver production workspace UNSAT",
        )?;
        self.require_workspace_capacity_for_cnf(
            workspace,
            cnf,
            "GPU solver production workspace UNSAT",
        )?;
        self.solver
            .solve_expect_unsat_with_branch_limit_ws(workspace, cnf, branch_var_limit)?;
        checked_solver_trace_counter_inc!(self, gpu_cdcl_workspace_unsat_solves);
        self.trace.require_zero_cpu_search()
    }

    /// Solve workspace-backed UNSAT through GPU CDCL after accepted GPU epistemic execution.
    pub fn solve_expect_unsat_with_branch_limit_ws_with_gpu_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        workspace: &mut GpuCdclWorkspace,
        cnf: &GpuCnf,
        branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        self.with_trace_rollback(|this| {
            this.require_workspace_on_adapter_provider(workspace)?;
            this.require_solver_artifact_on_adapter_provider(
                cnf,
                branch_var_limit,
                "GPU solver production workspace UNSAT",
            )?;
            this.require_workspace_capacity_for_cnf(
                workspace,
                cnf,
                "GPU solver production workspace UNSAT",
            )?;
            let state = this.require_accepted_gpu_solver_evidence(provider, result)?;
            let events_before = this.trace.accepted_path_event_snapshot()?;
            this.solve_expect_unsat_with_branch_limit_ws(workspace, cnf, branch_var_limit)?;
            this.record_accepted_gpu_solver_production_path_events_since(events_before, &state)?;
            this.record_accepted_gpu_candidate_state(&state)?;
            this.trace.require_zero_cpu_search()
        })
    }

    fn solve_assumption_lifecycle_steps(
        &mut self,
        workspace: &mut GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
    ) -> Result<GpuSolverProductionLifecycleReport> {
        self.with_trace_rollback(|this| {
            this.solve_assumption_lifecycle_steps_impl(workspace, steps)
        })
    }

    fn solve_assumption_lifecycle_steps_impl(
        &mut self,
        workspace: &mut GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
    ) -> Result<GpuSolverProductionLifecycleReport> {
        Self::require_assumption_lifecycle_steps(steps)?;
        self.require_workspace_on_adapter_provider(workspace)?;

        let pushes_before = self.trace.gpu_assumption_pushes;
        let retractions_before = self.trace.gpu_assumption_retractions;
        let workspace_reuses_before = self.trace.gpu_lifecycle_workspace_reuses;
        let unknown_steps_before = self.trace.gpu_lifecycle_unknown_status_steps;
        let timeout_steps_before = self.trace.gpu_lifecycle_timeout_status_steps;
        let mut sat_steps = 0u64;
        let mut unsat_steps = 0u64;

        self.require_assumption_lifecycle_step_artifacts(workspace, steps)?;

        for step in steps {
            checked_solver_trace_counter_inc!(self, gpu_assumption_pushes);
            match step.expectation {
                GpuSolverProductionExpectation::Sat => {
                    self.solver
                        .solve_expect_sat_with_branch_limit(step.cnf, step.branch_var_limit)?;
                    checked_solver_trace_counter_inc!(self, gpu_cdcl_sat_solves);
                    sat_steps =
                        Self::checked_trace_counter_add(sat_steps, 1, "lifecycle_sat_steps")?;
                }
                GpuSolverProductionExpectation::Unsat => {
                    let assign_ptr_before = workspace.assign_device_ptr();
                    self.solve_expect_unsat_with_branch_limit_ws(
                        workspace,
                        step.cnf,
                        step.branch_var_limit,
                    )?;
                    if workspace.assign_device_ptr() == assign_ptr_before {
                        checked_solver_trace_counter_inc!(self, gpu_lifecycle_workspace_reuses);
                    }
                    unsat_steps =
                        Self::checked_trace_counter_add(unsat_steps, 1, "lifecycle_unsat_steps")?;
                }
                GpuSolverProductionExpectation::Unknown { .. } => {
                    checked_solver_trace_counter_inc!(self, gpu_lifecycle_unknown_status_steps);
                }
                GpuSolverProductionExpectation::Timeout { .. } => {
                    checked_solver_trace_counter_inc!(self, gpu_lifecycle_timeout_status_steps);
                }
            };
            checked_solver_trace_counter_inc!(self, gpu_assumption_retractions);
        }

        let assumption_pushes = Self::checked_report_counter_delta(
            self.trace.gpu_assumption_pushes,
            pushes_before,
            "gpu_assumption_pushes",
        )?;
        let assumption_retractions = Self::checked_report_counter_delta(
            self.trace.gpu_assumption_retractions,
            retractions_before,
            "gpu_assumption_retractions",
        )?;
        if assumption_pushes != assumption_retractions {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production lifecycle".to_string(),
                context: format!(
                    "assumption push/retract mismatch: pushes={} retractions={}",
                    assumption_pushes, assumption_retractions
                ),
            });
        }
        let unknown_steps = Self::checked_report_counter_delta(
            self.trace.gpu_lifecycle_unknown_status_steps,
            unknown_steps_before,
            "gpu_lifecycle_unknown_status_steps",
        )?;
        let timeout_steps = Self::checked_report_counter_delta(
            self.trace.gpu_lifecycle_timeout_status_steps,
            timeout_steps_before,
            "gpu_lifecycle_timeout_status_steps",
        )?;
        let accounted_sat_unsat =
            Self::checked_report_counter_add(sat_steps, unsat_steps, "lifecycle_status_steps")?;
        let accounted_known_unknown = Self::checked_report_counter_add(
            accounted_sat_unsat,
            unknown_steps,
            "lifecycle_status_steps",
        )?;
        let accounted_status_steps = Self::checked_report_counter_add(
            accounted_known_unknown,
            timeout_steps,
            "lifecycle_status_steps",
        )?;
        if accounted_status_steps != steps.len() as u64 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production lifecycle".to_string(),
                context: format!(
                    "lifecycle status accounting mismatch: sat={}, unsat={}, unknown={}, \
                     timeout={}, steps={}",
                    sat_steps,
                    unsat_steps,
                    unknown_steps,
                    timeout_steps,
                    steps.len()
                ),
            });
        }

        Ok(GpuSolverProductionLifecycleReport {
            candidate_evidence_records: 0,
            steps: steps.len() as u64,
            sat_steps,
            unsat_steps,
            assumption_pushes,
            assumption_retractions,
            workspace_reuses: Self::checked_report_counter_delta(
                self.trace.gpu_lifecycle_workspace_reuses,
                workspace_reuses_before,
                "gpu_lifecycle_workspace_reuses",
            )?,
            unknown_steps,
            timeout_steps,
        })
    }

    fn require_assumption_lifecycle_steps(
        steps: &[GpuSolverProductionLifecycleStep<'_>],
    ) -> Result<()> {
        if steps.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production lifecycle".to_string(),
                context: "accepted solver lifecycle requires at least one step".to_string(),
            });
        }

        for step in steps {
            match step.expectation {
                GpuSolverProductionExpectation::Unknown { reason } => {
                    if reason.trim().is_empty() {
                        return Err(XlogError::UnsupportedEpistemicConstruct {
                            construct: "GPU solver production lifecycle".to_string(),
                            context: "UNKNOWN lifecycle status requires a diagnostic reason"
                                .to_string(),
                        });
                    }
                }
                GpuSolverProductionExpectation::Timeout { budget_micros } => {
                    if budget_micros == 0 {
                        return Err(XlogError::UnsupportedEpistemicConstruct {
                            construct: "GPU solver production lifecycle".to_string(),
                            context: "TIMEOUT lifecycle status requires a nonzero budget"
                                .to_string(),
                        });
                    }
                }
                GpuSolverProductionExpectation::Sat | GpuSolverProductionExpectation::Unsat => {}
            }
        }
        Ok(())
    }

    /// Execute an accepted push/solve/retract lifecycle through existing GPU CDCL calls.
    pub fn solve_assumption_lifecycle_with_gpu_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        workspace: &mut GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
    ) -> Result<GpuSolverProductionLifecycleReport> {
        self.with_trace_rollback(|this| {
            Self::require_assumption_lifecycle_steps(steps)?;
            this.require_workspace_on_adapter_provider(workspace)?;
            this.require_assumption_lifecycle_step_artifacts(workspace, steps)?;
            let state = this.require_accepted_gpu_solver_evidence(provider, result)?;
            let events_before = this.trace.accepted_path_event_snapshot()?;
            let mut report = this.solve_assumption_lifecycle_steps(workspace, steps)?;
            report.candidate_evidence_records = 1;
            this.record_accepted_gpu_solver_production_path_events_since(events_before, &state)?;
            this.record_accepted_gpu_candidate_state(&state)?;
            this.trace.require_zero_cpu_search()?;
            Ok(report)
        })
    }

    /// Execute accepted push/solve/retract lifecycles for multiple GPU epistemic candidates.
    ///
    /// Each candidate result is validated against the accepted GPU execution boundary, then
    /// the same lifecycle steps are dispatched through the existing GPU CDCL SAT/UNSAT paths.
    pub fn solve_multi_candidate_assumption_lifecycle_with_gpu_execution_results(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        workspace: &mut GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
    ) -> Result<GpuSolverProductionLifecycleReport> {
        self.with_trace_rollback(|this| {
            this.solve_multi_candidate_assumption_lifecycle_with_gpu_execution_results_impl(
                provider, results, workspace, steps,
            )
        })
    }

    fn solve_multi_candidate_assumption_lifecycle_with_gpu_execution_results_impl(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        workspace: &mut GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
    ) -> Result<GpuSolverProductionLifecycleReport> {
        if results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production lifecycle".to_string(),
                context:
                    "multi-candidate solver lifecycle requires at least one accepted GPU result"
                        .to_string(),
            });
        }
        Self::require_assumption_lifecycle_steps(steps)?;
        self.require_workspace_on_adapter_provider(workspace)?;
        self.require_assumption_lifecycle_step_artifacts(workspace, steps)?;

        let states = self.require_accepted_gpu_solver_states(provider, results)?;

        let mut report = GpuSolverProductionLifecycleReport::default();
        for state in &states {
            let events_before = self.trace.accepted_path_event_snapshot()?;
            let step_report = self.solve_assumption_lifecycle_steps(workspace, steps)?;
            checked_solver_report_counter_inc!(report, candidate_evidence_records);
            checked_solver_report_counter_add!(report, steps, step_report.steps);
            checked_solver_report_counter_add!(report, sat_steps, step_report.sat_steps);
            checked_solver_report_counter_add!(report, unsat_steps, step_report.unsat_steps);
            checked_solver_report_counter_add!(
                report,
                assumption_pushes,
                step_report.assumption_pushes
            );
            checked_solver_report_counter_add!(
                report,
                assumption_retractions,
                step_report.assumption_retractions
            );
            checked_solver_report_counter_add!(
                report,
                workspace_reuses,
                step_report.workspace_reuses
            );
            checked_solver_report_counter_add!(report, unknown_steps, step_report.unknown_steps);
            checked_solver_report_counter_add!(report, timeout_steps, step_report.timeout_steps);
            self.record_accepted_gpu_solver_production_path_events_since(events_before, state)?;
            self.record_accepted_gpu_candidate_state(state)?;
        }

        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Execute accepted split/batch push/solve/retract lifecycles through existing GPU CDCL calls.
    ///
    /// The batch evidence must prove every split component ran through the
    /// single-plan GPU runtime path with zero aggregate CPU recomposition,
    /// candidate/world-view fallback, tracked hot-path D2H, and per-candidate
    /// host round trips, plus aggregate CUDA-event timing.
    pub fn solve_assumption_lifecycle_with_gpu_batch_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        evidence: GpuSolverProductionBatchExecutionEvidence<'_>,
        workspace: &mut GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
    ) -> Result<GpuSolverProductionLifecycleReport> {
        self.with_trace_rollback(|this| {
            Self::require_assumption_lifecycle_steps(steps)?;
            this.require_workspace_on_adapter_provider(workspace)?;
            this.require_assumption_lifecycle_step_artifacts(workspace, steps)?;
            let results =
                this.accepted_solver_results_from_gpu_batch_execution_evidence(provider, evidence)?;
            let report = this
                .solve_multi_candidate_assumption_lifecycle_with_gpu_execution_results(
                    provider, &results, workspace, steps,
                )?;
            this.record_accepted_gpu_batch_candidate_evidence(results.len())?;
            this.trace.require_zero_cpu_search()?;
            Ok(report)
        })
    }

    /// Populate and publish the existing GPU CDCL learned-clause/proof arena.
    ///
    /// This records that an accepted epistemic candidate reached the GPU CDCL
    /// learned-clause device buffers. Import/reuse is covered by the bounded
    /// same-device-CNF reuse API below.
    pub fn solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        workspace: &mut GpuCdclWorkspace,
        cnf: &GpuCnf,
        branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<GpuSolverProductionLearnedClauseArenaReport> {
        self.with_trace_rollback(|this| {
            this.solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result_impl(
                provider,
                result,
                workspace,
                cnf,
                branch_var_limit,
            )
        })
    }

    fn solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result_impl(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        workspace: &mut GpuCdclWorkspace,
        cnf: &GpuCnf,
        branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<GpuSolverProductionLearnedClauseArenaReport> {
        self.require_workspace_on_adapter_provider(workspace)?;
        self.require_learned_clause_publication_artifacts(workspace, cnf, branch_var_limit)?;
        let state = self.require_accepted_gpu_solver_evidence(provider, result)?;
        let events_before = self.trace.accepted_path_event_snapshot()?;

        let learned_offsets_ptr = workspace.learned_offsets.device_ptr_value();
        let learned_lits_ptr = workspace.learned_lits.device_ptr_value();
        let proof_offsets_ptr = workspace.proof_offsets.device_ptr_value();
        let proof_data_ptr = workspace.proof_data.device_ptr_value();
        let learned_count_ptr = workspace.out_learned_count.device_ptr_value();

        self.solve_expect_unsat_with_branch_limit_ws(workspace, cnf, branch_var_limit)?;

        if learned_offsets_ptr == 0
            || learned_lits_ptr == 0
            || proof_offsets_ptr == 0
            || proof_data_ptr == 0
            || learned_count_ptr == 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver learned-clause arena".to_string(),
                context: "learned-clause publication requires non-null GPU arena buffers"
                    .to_string(),
            });
        }
        if workspace.learned_offsets.device_ptr_value() != learned_offsets_ptr
            || workspace.learned_lits.device_ptr_value() != learned_lits_ptr
            || workspace.proof_offsets.device_ptr_value() != proof_offsets_ptr
            || workspace.proof_data.device_ptr_value() != proof_data_ptr
            || workspace.out_learned_count.device_ptr_value() != learned_count_ptr
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver learned-clause arena".to_string(),
                context: "learned-clause publication must keep the reusable GPU workspace arena"
                    .to_string(),
            });
        }

        checked_solver_trace_counter_inc!(self, gpu_learned_clause_arena_publications);
        checked_solver_trace_counter_inc!(self, gpu_learned_count_buffer_publications);
        self.record_accepted_gpu_solver_production_path_events_since(events_before, &state)?;
        self.record_accepted_gpu_candidate_state(&state)?;
        self.trace.require_zero_cpu_search()?;

        Ok(GpuSolverProductionLearnedClauseArenaReport {
            unsat_solves: 1,
            gpu_learned_clause_arena_publications: 1,
            gpu_learned_count_buffer_publications: 1,
            cpu_learned_clause_transfers: self.trace.cpu_learned_clause_transfers,
        })
    }

    fn solve_unsat_then_reuse_learned_clauses(
        &mut self,
        workspace: &mut GpuCdclWorkspace,
        source_cnf: &GpuCnf,
        source_branch_var_limit: &TrackedCudaSlice<u32>,
        target_cnf: &GpuCnf,
        target_branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<GpuSolverProductionLearnedClauseReuseReport> {
        self.solve_unsat_then_reuse_learned_clauses_impl(
            workspace,
            source_cnf,
            source_branch_var_limit,
            target_cnf,
            target_branch_var_limit,
        )
    }

    fn solve_unsat_then_reuse_learned_clauses_impl(
        &mut self,
        workspace: &mut GpuCdclWorkspace,
        source_cnf: &GpuCnf,
        source_branch_var_limit: &TrackedCudaSlice<u32>,
        target_cnf: &GpuCnf,
        target_branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<GpuSolverProductionLearnedClauseReuseReport> {
        self.require_workspace_on_adapter_provider(workspace)?;
        self.require_learned_clause_reuse_inputs(
            workspace,
            source_cnf,
            source_branch_var_limit,
            target_cnf,
            target_branch_var_limit,
        )?;

        let learned_offsets_ptr = workspace.learned_offsets.device_ptr_value();
        let learned_lits_ptr = workspace.learned_lits.device_ptr_value();
        let proof_offsets_ptr = workspace.proof_offsets.device_ptr_value();
        let proof_data_ptr = workspace.proof_data.device_ptr_value();
        let learned_count_ptr = workspace.out_learned_count.device_ptr_value();
        if learned_offsets_ptr == 0
            || learned_lits_ptr == 0
            || proof_offsets_ptr == 0
            || proof_data_ptr == 0
            || learned_count_ptr == 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver learned-clause reuse".to_string(),
                context: "learned-clause reuse requires non-null GPU arena buffers".to_string(),
            });
        }

        self.solve_expect_unsat_with_branch_limit_ws(
            workspace,
            source_cnf,
            source_branch_var_limit,
        )?;
        require_stable_learned_clause_arena(
            "publication",
            workspace,
            learned_offsets_ptr,
            learned_lits_ptr,
            proof_offsets_ptr,
            proof_data_ptr,
            learned_count_ptr,
        )?;

        checked_solver_trace_counter_inc!(self, gpu_learned_clause_arena_publications);
        checked_solver_trace_counter_inc!(self, gpu_learned_count_buffer_publications);

        self.solver
            .solve_expect_unsat_with_branch_limit_ws_importing_learned(
                workspace,
                target_cnf,
                target_branch_var_limit,
            )?;
        checked_solver_trace_counter_inc!(self, gpu_cdcl_workspace_unsat_solves);
        require_stable_learned_clause_arena(
            "import",
            workspace,
            learned_offsets_ptr,
            learned_lits_ptr,
            proof_offsets_ptr,
            proof_data_ptr,
            learned_count_ptr,
        )?;

        checked_solver_trace_counter_inc!(self, gpu_learned_clause_imports);
        checked_solver_trace_counter_inc!(self, gpu_learned_clause_reused_solves);
        self.trace.require_zero_cpu_search()?;

        Ok(GpuSolverProductionLearnedClauseReuseReport {
            candidate_evidence_records: 0,
            candidates: 2,
            unsat_solves: 2,
            gpu_learned_clause_arena_publications: 1,
            gpu_learned_clause_imports: 1,
            gpu_learned_clause_reused_solves: 1,
            cpu_learned_clause_transfers: self.trace.cpu_learned_clause_transfers,
        })
    }

    /// Publish learned clauses from one accepted GPU UNSAT solve and import them into another.
    ///
    /// This is deliberately bounded to same-device-CNF reuse. The existing GPU proof trace is
    /// valid for the imported solve only when the base CNF buffers are the same.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        workspace: &mut GpuCdclWorkspace,
        source_cnf: &GpuCnf,
        source_branch_var_limit: &TrackedCudaSlice<u32>,
        target_cnf: &GpuCnf,
        target_branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<GpuSolverProductionLearnedClauseReuseReport> {
        self.with_trace_rollback_preserving_reuse_rejections(|this| {
            this.require_workspace_on_adapter_provider(workspace)?;
            this.require_learned_clause_reuse_inputs(
                workspace,
                source_cnf,
                source_branch_var_limit,
                target_cnf,
                target_branch_var_limit,
            )?;
            let state = this.require_accepted_gpu_solver_evidence(provider, result)?;
            let events_before = this.trace.accepted_path_event_snapshot()?;
            let mut report = this.solve_unsat_then_reuse_learned_clauses(
                workspace,
                source_cnf,
                source_branch_var_limit,
                target_cnf,
                target_branch_var_limit,
            )?;
            report.candidate_evidence_records = 1;
            this.record_accepted_gpu_solver_production_path_events_since(events_before, &state)?;
            this.record_accepted_gpu_candidate_state(&state)?;
            this.trace.require_zero_cpu_search()?;
            Ok(report)
        })
    }

    /// Publish and reuse learned clauses once per accepted GPU epistemic candidate.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        workspace: &mut GpuCdclWorkspace,
        source_cnf: &GpuCnf,
        source_branch_var_limit: &TrackedCudaSlice<u32>,
        target_cnf: &GpuCnf,
        target_branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<GpuSolverProductionLearnedClauseReuseReport> {
        self.with_trace_rollback_preserving_reuse_rejections(|this| {
            this.solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results_impl(
                provider,
                results,
                workspace,
                source_cnf,
                source_branch_var_limit,
                target_cnf,
                target_branch_var_limit,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results_impl(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        workspace: &mut GpuCdclWorkspace,
        source_cnf: &GpuCnf,
        source_branch_var_limit: &TrackedCudaSlice<u32>,
        target_cnf: &GpuCnf,
        target_branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<GpuSolverProductionLearnedClauseReuseReport> {
        if results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver learned-clause reuse".to_string(),
                context:
                    "multi-candidate learned-clause reuse requires at least one accepted GPU result"
                        .to_string(),
            });
        }
        self.require_workspace_on_adapter_provider(workspace)?;
        self.require_learned_clause_reuse_inputs(
            workspace,
            source_cnf,
            source_branch_var_limit,
            target_cnf,
            target_branch_var_limit,
        )?;
        let states = self.require_accepted_gpu_solver_states(provider, results)?;

        let mut report = GpuSolverProductionLearnedClauseReuseReport::default();
        for state in &states {
            let events_before = self.trace.accepted_path_event_snapshot()?;
            let step_report = self.solve_unsat_then_reuse_learned_clauses(
                workspace,
                source_cnf,
                source_branch_var_limit,
                target_cnf,
                target_branch_var_limit,
            )?;
            checked_solver_report_counter_inc!(report, candidate_evidence_records);
            checked_solver_report_counter_add!(report, candidates, step_report.candidates);
            checked_solver_report_counter_add!(report, unsat_solves, step_report.unsat_solves);
            checked_solver_report_counter_add!(
                report,
                gpu_learned_clause_arena_publications,
                step_report.gpu_learned_clause_arena_publications
            );
            checked_solver_report_counter_add!(
                report,
                gpu_learned_clause_imports,
                step_report.gpu_learned_clause_imports
            );
            checked_solver_report_counter_add!(
                report,
                gpu_learned_clause_reused_solves,
                step_report.gpu_learned_clause_reused_solves
            );
            report.cpu_learned_clause_transfers = self.trace.cpu_learned_clause_transfers;
            self.record_accepted_gpu_solver_production_path_events_since(events_before, state)?;
            self.record_accepted_gpu_candidate_state(state)?;
        }

        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Publish and reuse learned clauses once per accepted split/batch GPU component.
    ///
    /// The batch evidence must prove every split component reused the existing
    /// single-plan GPU runtime path before each component is delegated to the
    /// existing multi-candidate learned-clause reuse adapter.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_learned_clause_reuse_with_gpu_batch_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        evidence: GpuSolverProductionBatchExecutionEvidence<'_>,
        workspace: &mut GpuCdclWorkspace,
        source_cnf: &GpuCnf,
        source_branch_var_limit: &TrackedCudaSlice<u32>,
        target_cnf: &GpuCnf,
        target_branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<GpuSolverProductionLearnedClauseReuseReport> {
        self.with_trace_rollback_preserving_reuse_rejections(|this| {
            this.require_workspace_on_adapter_provider(workspace)?;
            this.require_learned_clause_reuse_inputs(
                workspace,
                source_cnf,
                source_branch_var_limit,
                target_cnf,
                target_branch_var_limit,
            )?;
            let results =
                this.accepted_solver_results_from_gpu_batch_execution_evidence(provider, evidence)?;
            let report = this
                .solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results(
                    provider,
                    &results,
                    workspace,
                    source_cnf,
                    source_branch_var_limit,
                    target_cnf,
                    target_branch_var_limit,
                )?;
            this.record_accepted_gpu_batch_candidate_evidence(results.len())?;
            this.trace.require_zero_cpu_search()?;
            Ok(report)
        })
    }

    fn solve_weighted_maxsat_candidates(
        &mut self,
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
        frontier_upper_bound_certificates: u64,
    ) -> Result<GpuSolverProductionMaxSatReport> {
        self.with_trace_rollback(|this| {
            this.solve_weighted_maxsat_candidates_impl(
                candidates,
                frontier_upper_bound_certificates,
            )
        })
    }

    fn solve_weighted_maxsat_candidates_impl(
        &mut self,
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
        frontier_upper_bound_certificates: u64,
    ) -> Result<GpuSolverProductionMaxSatReport> {
        self.require_weighted_maxsat_candidates_and_artifacts(candidates)?;

        let solves_before = self.trace.gpu_maxsat_candidate_solves;
        let mut optimum_score = 0u64;
        for candidate in candidates {
            let _assignment = self
                .solver
                .solve_expect_sat_with_branch_limit(candidate.cnf, candidate.branch_var_limit)?;
            checked_solver_trace_counter_inc!(self, gpu_cdcl_sat_solves);
            checked_solver_trace_counter_inc!(self, gpu_maxsat_candidate_solves);
            optimum_score = optimum_score.max(candidate.score);
        }
        checked_solver_trace_counter_inc!(self, gpu_maxsat_optima);
        self.trace.require_zero_cpu_search()?;
        let gpu_cdcl_candidate_solves = Self::checked_report_counter_delta(
            self.trace.gpu_maxsat_candidate_solves,
            solves_before,
            "gpu_maxsat_candidate_solves",
        )?;
        if frontier_upper_bound_certificates != 0 {
            self.trace.gpu_maxsat_frontier_certified_candidate_solves =
                Self::checked_trace_counter_add(
                    self.trace.gpu_maxsat_frontier_certified_candidate_solves,
                    gpu_cdcl_candidate_solves,
                    "gpu_maxsat_frontier_certified_candidate_solves",
                )?;
        }

        Ok(GpuSolverProductionMaxSatReport {
            candidate_evidence_records: 0,
            optimum_score,
            candidates_checked: candidates.len() as u64,
            satisfiable_candidates: candidates.len() as u64,
            unsat_candidates_pruned: 0,
            gpu_cdcl_candidate_encodes: 0,
            gpu_cdcl_candidate_solves,
            frontier_upper_bound_certificates,
        })
    }

    fn require_weighted_maxsat_candidates(
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<()> {
        if candidates.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT".to_string(),
                context: "bounded MaxSAT adapter requires at least one candidate CNF".to_string(),
            });
        }
        Ok(())
    }

    /// Solve a bounded weighted MaxSAT candidate set after accepted GPU epistemic execution.
    ///
    /// CPU orchestration is limited to launching/checking the provided candidate CNFs and
    /// comparing their declared scores. Each candidate is certified by the existing GPU CDCL
    /// SAT path; this adapter performs no CPU assignment or MaxSAT enumeration.
    pub fn solve_weighted_maxsat_candidates_with_gpu_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        self.with_trace_rollback(|this| {
            this.require_weighted_maxsat_candidates_and_artifacts(candidates)?;
            let state = this.require_accepted_gpu_solver_evidence(provider, result)?;
            let events_before = this.trace.accepted_path_event_snapshot()?;
            let mut report = this.solve_weighted_maxsat_candidates(candidates, 0)?;
            report.candidate_evidence_records = 1;
            this.record_accepted_gpu_solver_production_path_events_since(events_before, &state)?;
            this.record_accepted_gpu_candidate_state(&state)?;
            this.trace.require_zero_cpu_search()?;
            Ok(report)
        })
    }

    fn require_maxsat_lifecycle_inputs(
        steps: &[GpuSolverProductionLifecycleStep<'_>],
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<()> {
        if steps.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT lifecycle".to_string(),
                context: "accepted MaxSAT lifecycle requires at least one lifecycle step"
                    .to_string(),
            });
        }
        if candidates.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT lifecycle".to_string(),
                context: "bounded MaxSAT adapter requires at least one candidate CNF".to_string(),
            });
        }
        Self::require_assumption_lifecycle_steps(steps)?;
        Ok(())
    }

    /// Execute an accepted solver lifecycle, then a bounded MaxSAT candidate set.
    ///
    /// The same accepted GPU epistemic evidence gates both phases. The adapter
    /// records that evidence once, while lifecycle and MaxSAT counters prove the
    /// existing GPU CDCL paths handled all solver work without CPU search.
    pub fn solve_maxsat_lifecycle_with_gpu_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        workspace: &mut GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatLifecycleReport> {
        self.with_trace_rollback(|this| {
            this.solve_maxsat_lifecycle_with_gpu_execution_result_impl(
                provider, result, workspace, steps, candidates,
            )
        })
    }

    fn solve_maxsat_lifecycle_with_gpu_execution_result_impl(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        workspace: &mut GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatLifecycleReport> {
        Self::require_maxsat_lifecycle_inputs(steps, candidates)?;
        self.require_workspace_on_adapter_provider(workspace)?;
        self.require_maxsat_lifecycle_artifacts(workspace, steps, candidates)?;
        let state = self.require_accepted_gpu_solver_evidence(provider, result)?;

        let events_before = self.trace.accepted_path_event_snapshot()?;
        let lifecycle = self.solve_assumption_lifecycle_steps(workspace, steps)?;
        let maxsat = self.solve_weighted_maxsat_candidates(candidates, 0)?;
        self.record_accepted_gpu_solver_production_path_events_since(events_before, &state)?;
        self.record_accepted_gpu_candidate_state(&state)?;
        self.trace.require_zero_cpu_search()?;

        Ok(GpuSolverProductionMaxSatLifecycleReport {
            candidate_evidence_records: 1,
            lifecycle,
            maxsat,
        })
    }

    fn add_maxsat_lifecycle_step_report(
        report: &mut GpuSolverProductionMaxSatLifecycleReport,
        step_report: GpuSolverProductionMaxSatLifecycleReport,
    ) -> Result<()> {
        report.candidate_evidence_records = Self::checked_report_counter_add(
            report.candidate_evidence_records,
            step_report.candidate_evidence_records,
            "candidate_evidence_records",
        )?;
        report.lifecycle.steps = Self::checked_report_counter_add(
            report.lifecycle.steps,
            step_report.lifecycle.steps,
            "lifecycle.steps",
        )?;
        report.lifecycle.sat_steps = Self::checked_report_counter_add(
            report.lifecycle.sat_steps,
            step_report.lifecycle.sat_steps,
            "lifecycle.sat_steps",
        )?;
        report.lifecycle.unsat_steps = Self::checked_report_counter_add(
            report.lifecycle.unsat_steps,
            step_report.lifecycle.unsat_steps,
            "lifecycle.unsat_steps",
        )?;
        report.lifecycle.assumption_pushes = Self::checked_report_counter_add(
            report.lifecycle.assumption_pushes,
            step_report.lifecycle.assumption_pushes,
            "lifecycle.assumption_pushes",
        )?;
        report.lifecycle.assumption_retractions = Self::checked_report_counter_add(
            report.lifecycle.assumption_retractions,
            step_report.lifecycle.assumption_retractions,
            "lifecycle.assumption_retractions",
        )?;
        report.lifecycle.workspace_reuses = Self::checked_report_counter_add(
            report.lifecycle.workspace_reuses,
            step_report.lifecycle.workspace_reuses,
            "lifecycle.workspace_reuses",
        )?;
        report.lifecycle.unknown_steps = Self::checked_report_counter_add(
            report.lifecycle.unknown_steps,
            step_report.lifecycle.unknown_steps,
            "lifecycle.unknown_steps",
        )?;
        report.lifecycle.timeout_steps = Self::checked_report_counter_add(
            report.lifecycle.timeout_steps,
            step_report.lifecycle.timeout_steps,
            "lifecycle.timeout_steps",
        )?;
        report.maxsat.optimum_score = report
            .maxsat
            .optimum_score
            .max(step_report.maxsat.optimum_score);
        report.maxsat.candidates_checked = Self::checked_report_counter_add(
            report.maxsat.candidates_checked,
            step_report.maxsat.candidates_checked,
            "maxsat.candidates_checked",
        )?;
        report.maxsat.satisfiable_candidates = Self::checked_report_counter_add(
            report.maxsat.satisfiable_candidates,
            step_report.maxsat.satisfiable_candidates,
            "maxsat.satisfiable_candidates",
        )?;
        report.maxsat.unsat_candidates_pruned = Self::checked_report_counter_add(
            report.maxsat.unsat_candidates_pruned,
            step_report.maxsat.unsat_candidates_pruned,
            "maxsat.unsat_candidates_pruned",
        )?;
        report.maxsat.gpu_cdcl_candidate_encodes = Self::checked_report_counter_add(
            report.maxsat.gpu_cdcl_candidate_encodes,
            step_report.maxsat.gpu_cdcl_candidate_encodes,
            "maxsat.gpu_cdcl_candidate_encodes",
        )?;
        report.maxsat.gpu_cdcl_candidate_solves = Self::checked_report_counter_add(
            report.maxsat.gpu_cdcl_candidate_solves,
            step_report.maxsat.gpu_cdcl_candidate_solves,
            "maxsat.gpu_cdcl_candidate_solves",
        )?;
        report.maxsat.frontier_upper_bound_certificates = Self::checked_report_counter_add(
            report.maxsat.frontier_upper_bound_certificates,
            step_report.maxsat.frontier_upper_bound_certificates,
            "maxsat.frontier_upper_bound_certificates",
        )?;
        Ok(())
    }

    /// Execute accepted solver lifecycle plus MaxSAT candidate-set work once per evidence record.
    pub fn solve_multi_candidate_maxsat_lifecycle_with_gpu_execution_results(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        workspace: &mut GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatLifecycleReport> {
        self.with_trace_rollback(|this| {
            this.solve_multi_candidate_maxsat_lifecycle_with_gpu_execution_results_impl(
                provider, results, workspace, steps, candidates,
            )
        })
    }

    fn solve_multi_candidate_maxsat_lifecycle_with_gpu_execution_results_impl(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        workspace: &mut GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatLifecycleReport> {
        if results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT lifecycle".to_string(),
                context:
                    "multi-candidate MaxSAT lifecycle requires at least one accepted GPU result"
                        .to_string(),
            });
        }
        Self::require_maxsat_lifecycle_inputs(steps, candidates)?;
        self.require_workspace_on_adapter_provider(workspace)?;
        self.require_maxsat_lifecycle_artifacts(workspace, steps, candidates)?;
        let states = self.require_accepted_gpu_solver_states(provider, results)?;

        let mut report = GpuSolverProductionMaxSatLifecycleReport::default();
        for state in &states {
            let events_before = self.trace.accepted_path_event_snapshot()?;
            let lifecycle = self.solve_assumption_lifecycle_steps(workspace, steps)?;
            let maxsat = self.solve_weighted_maxsat_candidates(candidates, 0)?;
            self.record_accepted_gpu_solver_production_path_events_since(events_before, state)?;
            self.record_accepted_gpu_candidate_state(state)?;
            Self::add_maxsat_lifecycle_step_report(
                &mut report,
                GpuSolverProductionMaxSatLifecycleReport {
                    candidate_evidence_records: 1,
                    lifecycle,
                    maxsat,
                },
            )?;
        }

        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Execute accepted split/batch solver lifecycle plus MaxSAT candidate-set work.
    pub fn solve_maxsat_lifecycle_with_gpu_batch_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        evidence: GpuSolverProductionBatchExecutionEvidence<'_>,
        workspace: &mut GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatLifecycleReport> {
        self.with_trace_rollback(|this| {
            Self::require_maxsat_lifecycle_inputs(steps, candidates)?;
            this.require_workspace_on_adapter_provider(workspace)?;
            this.require_maxsat_lifecycle_artifacts(workspace, steps, candidates)?;
            let results =
                this.accepted_solver_results_from_gpu_batch_execution_evidence(provider, evidence)?;
            let report = this.solve_multi_candidate_maxsat_lifecycle_with_gpu_execution_results(
                provider, &results, workspace, steps, candidates,
            )?;
            this.record_accepted_gpu_batch_candidate_evidence(results.len())?;
            this.trace.require_zero_cpu_search()?;
            Ok(report)
        })
    }

    /// Solve a bounded weighted MaxSAT candidate set once per accepted GPU epistemic candidate.
    pub fn solve_multi_candidate_weighted_maxsat_with_gpu_execution_results(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        self.with_trace_rollback(|this| {
            this.solve_multi_candidate_weighted_maxsat_with_gpu_execution_results_impl(
                provider, results, candidates,
            )
        })
    }

    fn solve_multi_candidate_weighted_maxsat_with_gpu_execution_results_impl(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        if results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT".to_string(),
                context: "multi-candidate MaxSAT requires at least one accepted GPU result"
                    .to_string(),
            });
        }
        self.require_weighted_maxsat_candidates_and_artifacts(candidates)?;
        let states = self.require_accepted_gpu_solver_states(provider, results)?;

        let mut report = GpuSolverProductionMaxSatReport::default();
        for state in &states {
            let events_before = self.trace.accepted_path_event_snapshot()?;
            let step_report = self.solve_weighted_maxsat_candidates(candidates, 0)?;
            checked_solver_report_counter_inc!(report, candidate_evidence_records);
            report.optimum_score = report.optimum_score.max(step_report.optimum_score);
            checked_solver_report_counter_add!(
                report,
                candidates_checked,
                step_report.candidates_checked
            );
            checked_solver_report_counter_add!(
                report,
                satisfiable_candidates,
                step_report.satisfiable_candidates
            );
            checked_solver_report_counter_add!(
                report,
                unsat_candidates_pruned,
                step_report.unsat_candidates_pruned
            );
            checked_solver_report_counter_add!(
                report,
                gpu_cdcl_candidate_encodes,
                step_report.gpu_cdcl_candidate_encodes
            );
            checked_solver_report_counter_add!(
                report,
                gpu_cdcl_candidate_solves,
                step_report.gpu_cdcl_candidate_solves
            );
            checked_solver_report_counter_add!(
                report,
                frontier_upper_bound_certificates,
                step_report.frontier_upper_bound_certificates
            );
            self.record_accepted_gpu_solver_production_path_events_since(events_before, state)?;
            self.record_accepted_gpu_candidate_state(state)?;
        }

        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Solve a bounded weighted MaxSAT candidate set once per accepted split/batch GPU component.
    ///
    /// The batch evidence must prove every split component reused the existing
    /// single-plan GPU runtime path before each component is delegated to the
    /// existing multi-candidate MaxSAT adapter.
    pub fn solve_weighted_maxsat_candidates_with_gpu_batch_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        evidence: GpuSolverProductionBatchExecutionEvidence<'_>,
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        self.with_trace_rollback(|this| {
            this.require_weighted_maxsat_candidates_and_artifacts(candidates)?;
            let results =
                this.accepted_solver_results_from_gpu_batch_execution_evidence(provider, evidence)?;
            let report = this.solve_multi_candidate_weighted_maxsat_with_gpu_execution_results(
                provider, &results, candidates,
            )?;
            this.record_accepted_gpu_batch_candidate_evidence(results.len())?;
            this.trace.require_zero_cpu_search()?;
            Ok(report)
        })
    }

    fn encode_weighted_maxsat_search_candidates(
        &mut self,
        weighted: &SolveInstance,
        selections: &[GpuSolverProductionWeightedMaxSatSelection<'_>],
    ) -> Result<Vec<GpuSolverProductionEncodedMaxSatSearchCandidate>> {
        Self::require_weighted_maxsat_encoding_inputs(weighted, selections)?;

        let weights =
            weighted
                .weights
                .as_ref()
                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production MaxSAT encoding".to_string(),
                    context: "weighted MaxSAT encoding requires explicit soft-clause weights"
                        .to_string(),
                })?;

        let frontier =
            Self::complete_weighted_maxsat_frontier_selections(weighted, weights, selections)?;
        if frontier.completion_candidate_count != 0 {
            self.trace.gpu_maxsat_frontier_completion_candidate_encodes =
                Self::checked_trace_counter_add(
                    self.trace.gpu_maxsat_frontier_completion_candidate_encodes,
                    frontier.completion_candidate_count,
                    "gpu_maxsat_frontier_completion_candidate_encodes",
                )?;
        }
        checked_solver_trace_counter_inc!(self, gpu_maxsat_frontier_upper_bound_certificates);
        let mut encoded = Vec::with_capacity(frontier.selections.len());
        for selection in &frontier.selections {
            encoded.push(self.encode_weighted_maxsat_subset(
                weighted,
                weights,
                &selection.soft_clause_indices,
                selection.status,
            )?);
        }

        Ok(encoded)
    }

    fn encode_weighted_maxsat_subset(
        &mut self,
        weighted: &SolveInstance,
        weights: &[f64],
        soft_clause_indices: &[usize],
        status: GpuSolverProductionMaxSatSearchStatus,
    ) -> Result<GpuSolverProductionEncodedMaxSatSearchCandidate> {
        let mut score = 0u64;
        let mut clauses = Vec::with_capacity(soft_clause_indices.len());
        for &idx in soft_clause_indices {
            let clause = &weighted.clauses[idx];
            let weight = Self::soft_clause_weight_score(idx, weights[idx])?;
            score = score.checked_add(weight).ok_or_else(|| {
                XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production MaxSAT encoding".to_string(),
                    context: format!(
                        "soft-clause selection score overflowed u64 while adding index {}",
                        idx
                    ),
                }
            })?;
            clauses.push(clause.clone());
        }

        let candidate_instance = SolveInstance::new(weighted.num_vars, clauses);
        let data_plane_before = self.provider.host_transfer_stats();
        let launch_metadata_before = self.provider.host_launch_metadata_transfer_stats();
        let cnf = GpuCnf::from_host(&candidate_instance, &self.provider)?;
        let data_plane_after = self.provider.host_transfer_stats();
        let launch_metadata_after = self.provider.host_launch_metadata_transfer_stats();
        self.record_encoded_maxsat_cnf_upload_transfer_delta(
            data_plane_before,
            data_plane_after,
            launch_metadata_before,
            launch_metadata_after,
        )?;
        checked_solver_trace_counter_inc!(self, gpu_maxsat_candidate_encodes);
        Ok(GpuSolverProductionEncodedMaxSatSearchCandidate { score, cnf, status })
    }

    fn record_encoded_maxsat_cnf_upload_transfer_delta(
        &mut self,
        data_plane_before: xlog_cuda::provider::HostTransferStats,
        data_plane_after: xlog_cuda::provider::HostTransferStats,
        launch_metadata_before: xlog_cuda::provider::HostLaunchMetadataTransferStats,
        launch_metadata_after: xlog_cuda::provider::HostLaunchMetadataTransferStats,
    ) -> Result<()> {
        let data_plane_htod_calls = Self::checked_report_counter_delta(
            data_plane_after.htod_calls,
            data_plane_before.htod_calls,
            "gpu_maxsat_candidate_cnf_data_plane_htod_calls",
        )?;
        let data_plane_htod_bytes = Self::checked_report_counter_delta(
            data_plane_after.htod_bytes,
            data_plane_before.htod_bytes,
            "gpu_maxsat_candidate_cnf_data_plane_htod_bytes",
        )?;
        let data_plane_dtoh_calls = Self::checked_report_counter_delta(
            data_plane_after.dtoh_calls,
            data_plane_before.dtoh_calls,
            "gpu_maxsat_candidate_cnf_data_plane_dtoh_calls",
        )?;
        let data_plane_dtoh_bytes = Self::checked_report_counter_delta(
            data_plane_after.dtoh_bytes,
            data_plane_before.dtoh_bytes,
            "gpu_maxsat_candidate_cnf_data_plane_dtoh_bytes",
        )?;
        let launch_metadata_htod_calls = Self::checked_report_counter_delta(
            launch_metadata_after.htod_calls,
            launch_metadata_before.htod_calls,
            "gpu_maxsat_candidate_cnf_launch_metadata_htod_calls",
        )?;
        let launch_metadata_htod_bytes = Self::checked_report_counter_delta(
            launch_metadata_after.htod_bytes,
            launch_metadata_before.htod_bytes,
            "gpu_maxsat_candidate_cnf_launch_metadata_htod_bytes",
        )?;

        self.trace.gpu_maxsat_candidate_cnf_data_plane_htod_calls =
            Self::checked_trace_counter_add(
                self.trace.gpu_maxsat_candidate_cnf_data_plane_htod_calls,
                data_plane_htod_calls,
                "gpu_maxsat_candidate_cnf_data_plane_htod_calls",
            )?;
        self.trace.gpu_maxsat_candidate_cnf_data_plane_htod_bytes =
            Self::checked_trace_counter_add(
                self.trace.gpu_maxsat_candidate_cnf_data_plane_htod_bytes,
                data_plane_htod_bytes,
                "gpu_maxsat_candidate_cnf_data_plane_htod_bytes",
            )?;
        self.trace.gpu_maxsat_candidate_cnf_data_plane_dtoh_calls =
            Self::checked_trace_counter_add(
                self.trace.gpu_maxsat_candidate_cnf_data_plane_dtoh_calls,
                data_plane_dtoh_calls,
                "gpu_maxsat_candidate_cnf_data_plane_dtoh_calls",
            )?;
        self.trace.gpu_maxsat_candidate_cnf_data_plane_dtoh_bytes =
            Self::checked_trace_counter_add(
                self.trace.gpu_maxsat_candidate_cnf_data_plane_dtoh_bytes,
                data_plane_dtoh_bytes,
                "gpu_maxsat_candidate_cnf_data_plane_dtoh_bytes",
            )?;
        self.trace
            .gpu_maxsat_candidate_cnf_launch_metadata_htod_calls = Self::checked_trace_counter_add(
            self.trace
                .gpu_maxsat_candidate_cnf_launch_metadata_htod_calls,
            launch_metadata_htod_calls,
            "gpu_maxsat_candidate_cnf_launch_metadata_htod_calls",
        )?;
        self.trace
            .gpu_maxsat_candidate_cnf_launch_metadata_htod_bytes = Self::checked_trace_counter_add(
            self.trace
                .gpu_maxsat_candidate_cnf_launch_metadata_htod_bytes,
            launch_metadata_htod_bytes,
            "gpu_maxsat_candidate_cnf_launch_metadata_htod_bytes",
        )?;
        Ok(())
    }

    fn solve_weighted_maxsat_search_candidates(
        &mut self,
        workspace: &mut GpuCdclWorkspace,
        candidates: &[GpuSolverProductionMaxSatSearchCandidate<'_>],
        frontier_upper_bound_certificates: u64,
    ) -> Result<GpuSolverProductionMaxSatReport> {
        self.require_workspace_on_adapter_provider(workspace)?;
        self.require_weighted_maxsat_search_candidates_and_artifacts(workspace, candidates)?;

        let solves_before = self.trace.gpu_maxsat_candidate_solves;
        let unsat_prunes_before = self.trace.gpu_maxsat_unsat_candidate_prunes;
        let mut optimum_score = 0u64;
        let mut satisfiable_candidates = 0u64;

        for candidate in candidates {
            match candidate.status {
                GpuSolverProductionMaxSatSearchStatus::Satisfiable => {
                    let _assignment = self.solver.solve_expect_sat_with_branch_limit(
                        candidate.cnf,
                        candidate.branch_var_limit,
                    )?;
                    checked_solver_trace_counter_inc!(self, gpu_cdcl_sat_solves);
                    checked_solver_trace_counter_inc!(self, gpu_maxsat_candidate_solves);
                    satisfiable_candidates = Self::checked_trace_counter_add(
                        satisfiable_candidates,
                        1,
                        "maxsat_satisfiable_candidates",
                    )?;
                    optimum_score = optimum_score.max(candidate.score);
                }
                GpuSolverProductionMaxSatSearchStatus::Unsatisfiable => {
                    self.solve_expect_unsat_with_branch_limit_ws(
                        workspace,
                        candidate.cnf,
                        candidate.branch_var_limit,
                    )?;
                    checked_solver_trace_counter_inc!(self, gpu_maxsat_candidate_solves);
                    checked_solver_trace_counter_inc!(self, gpu_maxsat_unsat_candidate_prunes);
                }
            }
        }

        checked_solver_trace_counter_inc!(self, gpu_maxsat_optima);
        self.trace.require_zero_cpu_search()?;
        let gpu_cdcl_candidate_solves = Self::checked_report_counter_delta(
            self.trace.gpu_maxsat_candidate_solves,
            solves_before,
            "gpu_maxsat_candidate_solves",
        )?;
        if frontier_upper_bound_certificates != 0 {
            self.trace.gpu_maxsat_frontier_certified_candidate_solves =
                Self::checked_trace_counter_add(
                    self.trace.gpu_maxsat_frontier_certified_candidate_solves,
                    gpu_cdcl_candidate_solves,
                    "gpu_maxsat_frontier_certified_candidate_solves",
                )?;
        }

        Ok(GpuSolverProductionMaxSatReport {
            candidate_evidence_records: 0,
            optimum_score,
            candidates_checked: candidates.len() as u64,
            satisfiable_candidates,
            unsat_candidates_pruned: Self::checked_report_counter_delta(
                self.trace.gpu_maxsat_unsat_candidate_prunes,
                unsat_prunes_before,
                "gpu_maxsat_unsat_candidate_prunes",
            )?,
            gpu_cdcl_candidate_encodes: 0,
            gpu_cdcl_candidate_solves,
            frontier_upper_bound_certificates,
        })
    }

    fn require_weighted_maxsat_search_candidates(
        candidates: &[GpuSolverProductionMaxSatSearchCandidate<'_>],
    ) -> Result<()> {
        if candidates.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT search".to_string(),
                context: "bounded MaxSAT search requires at least one candidate CNF".to_string(),
            });
        }
        if !candidates.iter().any(|candidate| {
            matches!(
                candidate.status,
                GpuSolverProductionMaxSatSearchStatus::Satisfiable
            )
        }) {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT search".to_string(),
                context: "bounded MaxSAT search requires at least one satisfiable GPU candidate"
                    .to_string(),
            });
        }
        Ok(())
    }

    fn require_weighted_maxsat_search_selections(
        selections: &[GpuSolverProductionWeightedMaxSatSelection<'_>],
    ) -> Result<()> {
        if selections.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT encoding".to_string(),
                context: "weighted MaxSAT encoding requires at least one selection".to_string(),
            });
        }
        Ok(())
    }

    fn require_weighted_maxsat_encoding_inputs(
        weighted: &SolveInstance,
        selections: &[GpuSolverProductionWeightedMaxSatSelection<'_>],
    ) -> Result<()> {
        Self::require_weighted_maxsat_search_selections(selections)?;

        if weighted.objective != Objective::MaxSat {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT encoding".to_string(),
                context: format!(
                    "weighted MaxSAT encoding requires Objective::MaxSat, got {:?}",
                    weighted.objective
                ),
            });
        }
        if weighted.num_vars == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT encoding".to_string(),
                context: "weighted MaxSAT encoding requires num_vars > 0".to_string(),
            });
        }

        let weights =
            weighted
                .weights
                .as_ref()
                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production MaxSAT encoding".to_string(),
                    context: "weighted MaxSAT encoding requires explicit soft-clause weights"
                        .to_string(),
                })?;
        if weights.len() != weighted.clauses.len() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT encoding".to_string(),
                context: format!(
                    "soft-clause weights length {} does not match clause count {}",
                    weights.len(),
                    weighted.clauses.len()
                ),
            });
        }

        for selection in selections {
            if selection.soft_clause_indices.is_empty() {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production MaxSAT encoding".to_string(),
                    context:
                        "weighted MaxSAT search selections must include at least one soft clause"
                            .to_string(),
                });
            }
            let mut seen_indices = BTreeSet::new();
            for (position, &idx) in selection.soft_clause_indices.iter().enumerate() {
                if !seen_indices.insert(idx) {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production MaxSAT encoding".to_string(),
                        context: format!(
                            "soft-clause selection duplicates index {} at position {}; \
                             weighted MaxSAT candidates must count each soft clause at most once",
                            idx, position
                        ),
                    });
                }
            }
            for &idx in selection.soft_clause_indices {
                let _clause = weighted.clauses.get(idx).ok_or_else(|| {
                    XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production MaxSAT encoding".to_string(),
                        context: format!(
                            "soft-clause selection index {} is out of range for {} clauses",
                            idx,
                            weighted.clauses.len()
                        ),
                    }
                })?;
                let weight =
                    *weights
                        .get(idx)
                        .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                            construct: "GPU solver production MaxSAT encoding".to_string(),
                            context: format!(
                                "soft-clause weight index {} is out of range for {} weights",
                                idx,
                                weights.len()
                            ),
                        })?;
                let _ = Self::soft_clause_weight_score(idx, weight)?;
            }
        }
        Ok(())
    }

    fn complete_weighted_maxsat_frontier_selections(
        weighted: &SolveInstance,
        weights: &[f64],
        selections: &[GpuSolverProductionWeightedMaxSatSelection<'_>],
    ) -> Result<GpuSolverProductionCompletedWeightedMaxSatFrontier> {
        let mut completed = Vec::with_capacity(selections.len());
        let mut completion_candidate_count = 0u64;
        let mut seen = BTreeMap::new();

        for selection in selections {
            let mut indices = selection.soft_clause_indices.to_vec();
            indices.sort_unstable();
            match seen.get(&indices) {
                Some(status) if *status != selection.status => {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production MaxSAT encoding".to_string(),
                        context: format!(
                            "soft-clause selection {:?} has conflicting statuses {:?} and {:?}",
                            indices, status, selection.status
                        ),
                    });
                }
                Some(_) => continue,
                None => {
                    seen.insert(indices.clone(), selection.status);
                    completed.push(GpuSolverProductionOwnedWeightedMaxSatSelection {
                        soft_clause_indices: indices,
                        status: selection.status,
                    });
                }
            }
        }

        let all_clause_indices: Vec<_> = (0..weighted.clauses.len()).collect();
        let certificates = Self::unsat_frontier_certificates(weights, &completed)?;
        let disjoint_frontier = certificates.len() > 1
            && Self::frontier_certificates_are_pairwise_disjoint(&certificates);
        Self::require_weighted_maxsat_frontier_completion_bound(&certificates, disjoint_frontier)?;
        if disjoint_frontier {
            let mut exclusions = Vec::with_capacity(certificates.len());
            Self::complete_disjoint_unsat_frontier_boundaries(
                &certificates,
                0,
                &mut exclusions,
                &all_clause_indices,
                &mut seen,
                &mut completed,
                &mut completion_candidate_count,
            )?;
        } else {
            for certificate in &certificates {
                for &excluded_idx in &certificate.min_weight_indices {
                    let boundary: Vec<_> = all_clause_indices
                        .iter()
                        .copied()
                        .filter(|idx| *idx != excluded_idx)
                        .collect();
                    Self::push_completed_frontier_candidate(
                        boundary,
                        &mut seen,
                        &mut completed,
                        &mut completion_candidate_count,
                    )?;
                }
            }
        }

        Self::require_weighted_maxsat_frontier_upper_bound(weights, &completed)?;
        Ok(GpuSolverProductionCompletedWeightedMaxSatFrontier {
            selections: completed,
            completion_candidate_count,
        })
    }

    fn require_weighted_maxsat_frontier_completion_bound(
        certificates: &[GpuSolverProductionUnsatFrontierCertificate],
        disjoint_frontier: bool,
    ) -> Result<()> {
        let implied_candidates = if disjoint_frontier {
            certificates.iter().try_fold(1u64, |acc, certificate| {
                acc.checked_mul(certificate.min_weight_indices.len() as u64)
                    .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production MaxSAT encoding".to_string(),
                        context: "weighted MaxSAT disjoint frontier completion bound overflowed"
                            .to_string(),
                    })
            })?
        } else {
            certificates.iter().try_fold(0u64, |acc, certificate| {
                acc.checked_add(certificate.min_weight_indices.len() as u64)
                    .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production MaxSAT encoding".to_string(),
                        context: "weighted MaxSAT frontier completion bound overflowed".to_string(),
                    })
            })?
        };

        if implied_candidates > MAX_WEIGHTED_MAXSAT_FRONTIER_COMPLETION_CANDIDATES {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT encoding".to_string(),
                context: format!(
                    "weighted MaxSAT frontier completion would require {} CPU-generated \
                     boundary candidates, exceeding production bound {}; provide explicit \
                     GPU scheduler selections",
                    implied_candidates, MAX_WEIGHTED_MAXSAT_FRONTIER_COMPLETION_CANDIDATES
                ),
            });
        }
        Ok(())
    }

    fn unsat_frontier_certificates(
        weights: &[f64],
        selections: &[GpuSolverProductionOwnedWeightedMaxSatSelection],
    ) -> Result<Vec<GpuSolverProductionUnsatFrontierCertificate>> {
        let mut certificates = Vec::new();
        for selection in selections {
            if selection.status != GpuSolverProductionMaxSatSearchStatus::Unsatisfiable {
                continue;
            }
            let mut indices = selection.soft_clause_indices.clone();
            indices.sort_unstable();
            let mut min_weight = None;
            let mut min_weight_indices = Vec::new();
            for &idx in &indices {
                let weight = Self::soft_clause_weight_score(idx, weights[idx])?;
                match min_weight {
                    None => {
                        min_weight = Some(weight);
                        min_weight_indices.push(idx);
                    }
                    Some(current) if weight < current => {
                        min_weight = Some(weight);
                        min_weight_indices.clear();
                        min_weight_indices.push(idx);
                    }
                    Some(current) if weight == current => min_weight_indices.push(idx),
                    Some(_) => {}
                }
            }
            if let Some(min_weight) = min_weight {
                certificates.push(GpuSolverProductionUnsatFrontierCertificate {
                    indices,
                    min_weight,
                    min_weight_indices,
                });
            }
        }
        Ok(certificates)
    }

    fn frontier_certificates_are_pairwise_disjoint(
        certificates: &[GpuSolverProductionUnsatFrontierCertificate],
    ) -> bool {
        let mut seen = BTreeSet::new();
        for certificate in certificates {
            for &idx in &certificate.indices {
                if !seen.insert(idx) {
                    return false;
                }
            }
        }
        true
    }

    fn complete_disjoint_unsat_frontier_boundaries(
        certificates: &[GpuSolverProductionUnsatFrontierCertificate],
        depth: usize,
        exclusions: &mut Vec<usize>,
        all_clause_indices: &[usize],
        seen: &mut BTreeMap<Vec<usize>, GpuSolverProductionMaxSatSearchStatus>,
        completed: &mut Vec<GpuSolverProductionOwnedWeightedMaxSatSelection>,
        completion_candidate_count: &mut u64,
    ) -> Result<()> {
        if depth == certificates.len() {
            let exclusion_set: BTreeSet<_> = exclusions.iter().copied().collect();
            let boundary: Vec<_> = all_clause_indices
                .iter()
                .copied()
                .filter(|idx| !exclusion_set.contains(idx))
                .collect();
            return Self::push_completed_frontier_candidate(
                boundary,
                seen,
                completed,
                completion_candidate_count,
            );
        }

        for &excluded_idx in &certificates[depth].min_weight_indices {
            exclusions.push(excluded_idx);
            Self::complete_disjoint_unsat_frontier_boundaries(
                certificates,
                depth + 1,
                exclusions,
                all_clause_indices,
                seen,
                completed,
                completion_candidate_count,
            )?;
            exclusions.pop();
        }
        Ok(())
    }

    fn push_completed_frontier_candidate(
        boundary: Vec<usize>,
        seen: &mut BTreeMap<Vec<usize>, GpuSolverProductionMaxSatSearchStatus>,
        completed: &mut Vec<GpuSolverProductionOwnedWeightedMaxSatSelection>,
        completion_candidate_count: &mut u64,
    ) -> Result<()> {
        if boundary.is_empty() || seen.contains_key(&boundary) {
            return Ok(());
        }
        if *completion_candidate_count >= MAX_WEIGHTED_MAXSAT_FRONTIER_COMPLETION_CANDIDATES {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT encoding".to_string(),
                context: format!(
                    "weighted MaxSAT frontier completion exceeded production bound {}; \
                     provide explicit GPU scheduler selections",
                    MAX_WEIGHTED_MAXSAT_FRONTIER_COMPLETION_CANDIDATES
                ),
            });
        }
        seen.insert(
            boundary.clone(),
            GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        );
        *completion_candidate_count =
            completion_candidate_count.checked_add(1).ok_or_else(|| {
                XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production MaxSAT encoding".to_string(),
                    context: "frontier completion candidate count overflowed".to_string(),
                }
            })?;
        completed.push(GpuSolverProductionOwnedWeightedMaxSatSelection {
            soft_clause_indices: boundary,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        });
        Ok(())
    }

    fn require_weighted_maxsat_frontier_upper_bound(
        weights: &[f64],
        selections: &[GpuSolverProductionOwnedWeightedMaxSatSelection],
    ) -> Result<()> {
        let mut total_score = 0u64;
        for (idx, &weight) in weights.iter().enumerate() {
            total_score = total_score
                .checked_add(Self::soft_clause_weight_score(idx, weight)?)
                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production MaxSAT encoding".to_string(),
                    context: format!(
                        "weighted MaxSAT total score overflowed while adding soft clause {}",
                        idx
                    ),
                })?;
        }

        let mut upper_bound = total_score;
        let mut best_satisfiable_score = 0u64;
        let certificates = Self::unsat_frontier_certificates(weights, selections)?;

        for selection in selections {
            let mut indices = selection.soft_clause_indices.to_vec();
            indices.sort_unstable();
            let score = Self::weighted_maxsat_selection_score(weights, &indices)?;
            match selection.status {
                GpuSolverProductionMaxSatSearchStatus::Satisfiable => {
                    best_satisfiable_score = best_satisfiable_score.max(score);
                }
                GpuSolverProductionMaxSatSearchStatus::Unsatisfiable => {}
            }
        }

        if certificates.len() > 1
            && Self::frontier_certificates_are_pairwise_disjoint(&certificates)
        {
            let certified_loss = certificates.iter().try_fold(0u64, |acc, certificate| {
                acc.checked_add(certificate.min_weight).ok_or_else(|| {
                    XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production MaxSAT encoding".to_string(),
                        context: "weighted MaxSAT disjoint frontier loss overflowed".to_string(),
                    }
                })
            })?;
            upper_bound = total_score.checked_sub(certified_loss).ok_or_else(|| {
                XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production MaxSAT encoding".to_string(),
                    context: format!(
                        "weighted MaxSAT disjoint frontier loss {} exceeds total score {}",
                        certified_loss, total_score
                    ),
                }
            })?;
        } else {
            for certificate in &certificates {
                let certificate_bound =
                    total_score
                        .checked_sub(certificate.min_weight)
                        .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                            construct: "GPU solver production MaxSAT encoding".to_string(),
                            context: format!(
                                "weighted MaxSAT UNSAT certificate {:?} has minimum weight {} above total score {}",
                                certificate.indices, certificate.min_weight, total_score
                            ),
                        })?;
                upper_bound = upper_bound.min(certificate_bound);
            }
        }

        if best_satisfiable_score < upper_bound {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT encoding".to_string(),
                context: format!(
                    "weighted MaxSAT frontier is incomplete: best GPU-certified satisfiable score {} is below the certified upper bound {}",
                    best_satisfiable_score, upper_bound
                ),
            });
        }
        Ok(())
    }

    fn weighted_maxsat_selection_score(weights: &[f64], indices: &[usize]) -> Result<u64> {
        let mut score = 0u64;
        for &idx in indices {
            let weight =
                *weights
                    .get(idx)
                    .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production MaxSAT encoding".to_string(),
                        context: format!(
                            "soft-clause weight index {} is out of range for {} weights",
                            idx,
                            weights.len()
                        ),
                    })?;
            score = score
                .checked_add(Self::soft_clause_weight_score(idx, weight)?)
                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU solver production MaxSAT encoding".to_string(),
                    context: format!(
                        "soft-clause selection score overflowed u64 while adding index {}",
                        idx
                    ),
                })?;
        }
        Ok(score)
    }

    fn soft_clause_weight_score(idx: usize, weight: f64) -> Result<u64> {
        if !weight.is_finite() || weight < 0.0 || weight.fract() != 0.0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT encoding".to_string(),
                context: format!(
                    "soft-clause weight at index {} must be a finite nonnegative integer, got {}",
                    idx, weight
                ),
            });
        }
        if weight >= u64::MAX as f64 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT encoding".to_string(),
                context: format!(
                    "soft-clause weight at index {} exceeds u64 score range",
                    idx
                ),
            });
        }
        Ok(weight as u64)
    }

    /// Search a bounded weighted MaxSAT candidate set after accepted GPU epistemic execution.
    ///
    /// Satisfiable candidates are scored through the existing GPU CDCL SAT path.
    /// Unsatisfiable candidates are pruned through the existing workspace-backed
    /// GPU CDCL UNSAT path. The adapter records no CPU assignment or MaxSAT
    /// enumeration.
    pub fn solve_weighted_maxsat_search_with_gpu_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        workspace: &mut GpuCdclWorkspace,
        candidates: &[GpuSolverProductionMaxSatSearchCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        self.with_trace_rollback(|this| {
            this.solve_weighted_maxsat_search_with_gpu_execution_result_impl(
                provider, result, workspace, candidates,
            )
        })
    }

    fn solve_weighted_maxsat_search_with_gpu_execution_result_impl(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        workspace: &mut GpuCdclWorkspace,
        candidates: &[GpuSolverProductionMaxSatSearchCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        self.require_workspace_on_adapter_provider(workspace)?;
        self.require_weighted_maxsat_search_candidates_and_artifacts(workspace, candidates)?;
        let state = self.require_accepted_gpu_solver_evidence(provider, result)?;
        let events_before = self.trace.accepted_path_event_snapshot()?;
        let mut report = self.solve_weighted_maxsat_search_candidates(workspace, candidates, 0)?;
        report.candidate_evidence_records = 1;
        self.record_accepted_gpu_solver_production_path_events_since(events_before, &state)?;
        self.record_accepted_gpu_candidate_state(&state)?;
        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Search a bounded weighted MaxSAT candidate set once per accepted split-batch component.
    pub fn solve_weighted_maxsat_search_with_gpu_batch_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        evidence: GpuSolverProductionBatchExecutionEvidence<'_>,
        workspace: &mut GpuCdclWorkspace,
        candidates: &[GpuSolverProductionMaxSatSearchCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        self.with_trace_rollback(|this| {
            this.require_workspace_on_adapter_provider(workspace)?;
            this.require_weighted_maxsat_search_candidates_and_artifacts(workspace, candidates)?;
            let results =
                this.accepted_solver_results_from_gpu_batch_execution_evidence(provider, evidence)?;
            let report = this
                .solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results(
                    provider, &results, workspace, candidates,
                )?;
            this.record_accepted_gpu_batch_candidate_evidence(results.len())?;
            this.trace.require_zero_cpu_search()?;
            Ok(report)
        })
    }

    /// Search a bounded weighted MaxSAT candidate set once per accepted GPU evidence record.
    pub fn solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        workspace: &mut GpuCdclWorkspace,
        candidates: &[GpuSolverProductionMaxSatSearchCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        self.with_trace_rollback(|this| {
            this.solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results_impl(
                provider, results, workspace, candidates,
            )
        })
    }

    fn solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results_impl(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        workspace: &mut GpuCdclWorkspace,
        candidates: &[GpuSolverProductionMaxSatSearchCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        if results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT search".to_string(),
                context: "multi-candidate MaxSAT search requires at least one accepted GPU result"
                    .to_string(),
            });
        }
        self.require_workspace_on_adapter_provider(workspace)?;
        self.require_weighted_maxsat_search_candidates_and_artifacts(workspace, candidates)?;
        let states = self.require_accepted_gpu_solver_states(provider, results)?;

        let mut report = GpuSolverProductionMaxSatReport::default();
        for state in &states {
            let events_before = self.trace.accepted_path_event_snapshot()?;
            let step_report =
                self.solve_weighted_maxsat_search_candidates(workspace, candidates, 0)?;
            checked_solver_report_counter_inc!(report, candidate_evidence_records);
            report.optimum_score = report.optimum_score.max(step_report.optimum_score);
            checked_solver_report_counter_add!(
                report,
                candidates_checked,
                step_report.candidates_checked
            );
            checked_solver_report_counter_add!(
                report,
                satisfiable_candidates,
                step_report.satisfiable_candidates
            );
            checked_solver_report_counter_add!(
                report,
                unsat_candidates_pruned,
                step_report.unsat_candidates_pruned
            );
            checked_solver_report_counter_add!(
                report,
                gpu_cdcl_candidate_encodes,
                step_report.gpu_cdcl_candidate_encodes
            );
            checked_solver_report_counter_add!(
                report,
                gpu_cdcl_candidate_solves,
                step_report.gpu_cdcl_candidate_solves
            );
            checked_solver_report_counter_add!(
                report,
                frontier_upper_bound_certificates,
                step_report.frontier_upper_bound_certificates
            );
            self.record_accepted_gpu_solver_production_path_events_since(events_before, state)?;
            self.record_accepted_gpu_candidate_state(state)?;
        }

        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Encode weighted soft-clause selections, then search them after accepted GPU evidence.
    ///
    /// Candidate construction is bounded by caller-declared selections. The adapter
    /// builds satisfaction CNFs for those selections, uploads them through the existing
    /// GPU CNF layout, and dispatches SAT/UNSAT certification through GPU CDCL. It
    /// performs no CPU assignment or MaxSAT subset enumeration.
    pub fn solve_weighted_maxsat_encoded_search_with_gpu_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        workspace: &mut GpuCdclWorkspace,
        weighted: &SolveInstance,
        branch_var_limit: &TrackedCudaSlice<u32>,
        selections: &[GpuSolverProductionWeightedMaxSatSelection<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        self.with_trace_rollback(|this| {
            this.solve_weighted_maxsat_encoded_search_with_gpu_execution_result_impl(
                provider,
                result,
                workspace,
                weighted,
                branch_var_limit,
                selections,
            )
        })
    }

    fn solve_weighted_maxsat_encoded_search_with_gpu_execution_result_impl(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        workspace: &mut GpuCdclWorkspace,
        weighted: &SolveInstance,
        branch_var_limit: &TrackedCudaSlice<u32>,
        selections: &[GpuSolverProductionWeightedMaxSatSelection<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        self.require_workspace_on_adapter_provider(workspace)?;
        self.require_weighted_maxsat_encoded_search_inputs_and_artifacts(
            workspace,
            weighted,
            branch_var_limit,
            selections,
        )?;
        let state = self.require_accepted_gpu_solver_evidence(provider, result)?;
        let events_before = self.trace.accepted_path_event_snapshot()?;
        let encodes_before = self.trace.gpu_maxsat_candidate_encodes;
        let certificates_before = self.trace.gpu_maxsat_frontier_upper_bound_certificates;
        let encoded = self.encode_weighted_maxsat_search_candidates(weighted, selections)?;
        let frontier_upper_bound_certificates = Self::checked_report_counter_delta(
            self.trace.gpu_maxsat_frontier_upper_bound_certificates,
            certificates_before,
            "gpu_maxsat_frontier_upper_bound_certificates",
        )?;
        let search_candidates: Vec<_> = encoded
            .iter()
            .map(|candidate| GpuSolverProductionMaxSatSearchCandidate {
                score: candidate.score,
                cnf: &candidate.cnf,
                branch_var_limit,
                status: candidate.status,
            })
            .collect();
        let mut report = self.solve_weighted_maxsat_search_candidates(
            workspace,
            &search_candidates,
            frontier_upper_bound_certificates,
        )?;
        report.candidate_evidence_records = 1;
        report.gpu_cdcl_candidate_encodes = Self::checked_report_counter_delta(
            self.trace.gpu_maxsat_candidate_encodes,
            encodes_before,
            "gpu_cdcl_candidate_encodes",
        )?;
        self.record_accepted_gpu_solver_production_path_events_since(events_before, &state)?;
        self.record_accepted_gpu_candidate_state(&state)?;
        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Encode weighted soft-clause selections, then search once per accepted GPU evidence record.
    ///
    /// This is the multi-candidate scheduler-facing variant of the bounded encoded
    /// MaxSAT search adapter. It validates all accepted GPU epistemic evidence up
    /// front, encodes the caller-declared selections through the existing GPU CNF
    /// layout for each accepted record, and dispatches each candidate through GPU
    /// CDCL SAT/UNSAT certification without CPU assignment or MaxSAT enumeration.
    pub fn solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        workspace: &mut GpuCdclWorkspace,
        weighted: &SolveInstance,
        branch_var_limit: &TrackedCudaSlice<u32>,
        selections: &[GpuSolverProductionWeightedMaxSatSelection<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        self.with_trace_rollback(|this| {
            this.solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results_impl(
                provider,
                results,
                workspace,
                weighted,
                branch_var_limit,
                selections,
            )
        })
    }

    fn solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results_impl(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        workspace: &mut GpuCdclWorkspace,
        weighted: &SolveInstance,
        branch_var_limit: &TrackedCudaSlice<u32>,
        selections: &[GpuSolverProductionWeightedMaxSatSelection<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        if results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT encoding".to_string(),
                context:
                    "multi-candidate weighted MaxSAT encoded search requires at least one accepted GPU result"
                        .to_string(),
            });
        }
        self.require_workspace_on_adapter_provider(workspace)?;
        self.require_weighted_maxsat_encoded_search_inputs_and_artifacts(
            workspace,
            weighted,
            branch_var_limit,
            selections,
        )?;
        let states = self.require_accepted_gpu_solver_states(provider, results)?;

        let mut report = GpuSolverProductionMaxSatReport::default();
        for state in &states {
            let events_before = self.trace.accepted_path_event_snapshot()?;
            let encodes_before = self.trace.gpu_maxsat_candidate_encodes;
            let certificates_before = self.trace.gpu_maxsat_frontier_upper_bound_certificates;
            let encoded = self.encode_weighted_maxsat_search_candidates(weighted, selections)?;
            let frontier_upper_bound_certificates = Self::checked_report_counter_delta(
                self.trace.gpu_maxsat_frontier_upper_bound_certificates,
                certificates_before,
                "gpu_maxsat_frontier_upper_bound_certificates",
            )?;
            let search_candidates: Vec<_> = encoded
                .iter()
                .map(|candidate| GpuSolverProductionMaxSatSearchCandidate {
                    score: candidate.score,
                    cnf: &candidate.cnf,
                    branch_var_limit,
                    status: candidate.status,
                })
                .collect();
            let step_report = self.solve_weighted_maxsat_search_candidates(
                workspace,
                &search_candidates,
                frontier_upper_bound_certificates,
            )?;
            checked_solver_report_counter_inc!(report, candidate_evidence_records);
            report.optimum_score = report.optimum_score.max(step_report.optimum_score);
            checked_solver_report_counter_add!(
                report,
                candidates_checked,
                step_report.candidates_checked
            );
            checked_solver_report_counter_add!(
                report,
                satisfiable_candidates,
                step_report.satisfiable_candidates
            );
            checked_solver_report_counter_add!(
                report,
                unsat_candidates_pruned,
                step_report.unsat_candidates_pruned
            );
            let encoded_delta = Self::checked_report_counter_delta(
                self.trace.gpu_maxsat_candidate_encodes,
                encodes_before,
                "gpu_cdcl_candidate_encodes",
            )?;
            checked_solver_report_counter_add!(report, gpu_cdcl_candidate_encodes, encoded_delta);
            checked_solver_report_counter_add!(
                report,
                gpu_cdcl_candidate_solves,
                step_report.gpu_cdcl_candidate_solves
            );
            checked_solver_report_counter_add!(
                report,
                frontier_upper_bound_certificates,
                step_report.frontier_upper_bound_certificates
            );
            self.record_accepted_gpu_solver_production_path_events_since(events_before, state)?;
            self.record_accepted_gpu_candidate_state(state)?;
        }

        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Encode weighted soft-clause selections, then search once per accepted split-batch component.
    ///
    /// The batch evidence must prove every split component reused the existing
    /// single-plan GPU runtime path before each component is delegated to the
    /// existing multi-candidate weighted MaxSAT encoding adapter.
    pub fn solve_weighted_maxsat_encoded_search_with_gpu_batch_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        evidence: GpuSolverProductionBatchExecutionEvidence<'_>,
        workspace: &mut GpuCdclWorkspace,
        weighted: &SolveInstance,
        branch_var_limit: &TrackedCudaSlice<u32>,
        selections: &[GpuSolverProductionWeightedMaxSatSelection<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        self.with_trace_rollback(|this| {
            this.require_workspace_on_adapter_provider(workspace)?;
            this.require_weighted_maxsat_encoded_search_inputs_and_artifacts(
                workspace,
                weighted,
                branch_var_limit,
                selections,
            )?;
            let results =
                this.accepted_solver_results_from_gpu_batch_execution_evidence(provider, evidence)?;
            let report = this
                .solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results(
                    provider,
                    &results,
                    workspace,
                    weighted,
                    branch_var_limit,
                    selections,
                )?;
            this.record_accepted_gpu_batch_candidate_evidence(results.len())?;
            this.trace.require_zero_cpu_search()?;
            Ok(report)
        })
    }

    fn add_maxsat_schedule_step_report(
        report: &mut GpuSolverProductionMaxSatScheduleReport,
        step_report: GpuSolverProductionMaxSatReport,
    ) -> Result<()> {
        report.optimum_score = report.optimum_score.max(step_report.optimum_score);
        checked_solver_report_counter_add!(
            report,
            candidates_checked,
            step_report.candidates_checked
        );
        checked_solver_report_counter_add!(
            report,
            satisfiable_candidates,
            step_report.satisfiable_candidates
        );
        checked_solver_report_counter_add!(
            report,
            unsat_candidates_pruned,
            step_report.unsat_candidates_pruned
        );
        checked_solver_report_counter_add!(
            report,
            gpu_cdcl_candidate_encodes,
            step_report.gpu_cdcl_candidate_encodes
        );
        checked_solver_report_counter_add!(
            report,
            gpu_cdcl_candidate_solves,
            step_report.gpu_cdcl_candidate_solves
        );
        checked_solver_report_counter_add!(
            report,
            frontier_upper_bound_certificates,
            step_report.frontier_upper_bound_certificates
        );
        Ok(())
    }

    fn solve_maxsat_schedule_jobs(
        &mut self,
        workspace: &mut GpuCdclWorkspace,
        jobs: &[GpuSolverProductionMaxSatScheduleJob<'_>],
    ) -> Result<GpuSolverProductionMaxSatScheduleReport> {
        self.require_workspace_on_adapter_provider(workspace)?;
        Self::require_maxsat_schedule_jobs(jobs)?;
        self.require_maxsat_schedule_job_artifacts(workspace, jobs)?;

        let mut report = GpuSolverProductionMaxSatScheduleReport::default();
        for job in jobs {
            checked_solver_trace_counter_inc!(self, gpu_maxsat_scheduler_jobs);
            checked_solver_report_counter_inc!(report, jobs);

            match job {
                GpuSolverProductionMaxSatScheduleJob::CandidateSet { candidates } => {
                    checked_solver_trace_counter_inc!(
                        self,
                        gpu_maxsat_scheduler_candidate_set_jobs
                    );
                    checked_solver_report_counter_inc!(report, candidate_set_jobs);
                    let step_report = self.solve_weighted_maxsat_candidates(candidates, 0)?;
                    Self::add_maxsat_schedule_step_report(&mut report, step_report)?;
                }
                GpuSolverProductionMaxSatScheduleJob::Search { candidates } => {
                    checked_solver_trace_counter_inc!(self, gpu_maxsat_scheduler_search_jobs);
                    checked_solver_report_counter_inc!(report, search_jobs);
                    let step_report =
                        self.solve_weighted_maxsat_search_candidates(workspace, candidates, 0)?;
                    Self::add_maxsat_schedule_step_report(&mut report, step_report)?;
                }
                GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
                    weighted,
                    branch_var_limit,
                    selections,
                } => {
                    checked_solver_trace_counter_inc!(
                        self,
                        gpu_maxsat_scheduler_encoded_search_jobs
                    );
                    checked_solver_report_counter_inc!(report, encoded_search_jobs);
                    let encodes_before = self.trace.gpu_maxsat_candidate_encodes;
                    let certificates_before =
                        self.trace.gpu_maxsat_frontier_upper_bound_certificates;
                    let encoded =
                        self.encode_weighted_maxsat_search_candidates(weighted, selections)?;
                    let frontier_upper_bound_certificates = Self::checked_report_counter_delta(
                        self.trace.gpu_maxsat_frontier_upper_bound_certificates,
                        certificates_before,
                        "gpu_maxsat_frontier_upper_bound_certificates",
                    )?;
                    let search_candidates: Vec<_> = encoded
                        .iter()
                        .map(|candidate| GpuSolverProductionMaxSatSearchCandidate {
                            score: candidate.score,
                            cnf: &candidate.cnf,
                            branch_var_limit,
                            status: candidate.status,
                        })
                        .collect();
                    let mut step_report = self.solve_weighted_maxsat_search_candidates(
                        workspace,
                        &search_candidates,
                        frontier_upper_bound_certificates,
                    )?;
                    step_report.gpu_cdcl_candidate_encodes = Self::checked_report_counter_delta(
                        self.trace.gpu_maxsat_candidate_encodes,
                        encodes_before,
                        "gpu_cdcl_candidate_encodes",
                    )?;
                    Self::add_maxsat_schedule_step_report(&mut report, step_report)?;
                }
                GpuSolverProductionMaxSatScheduleJob::Unknown { reason } => {
                    if reason.trim().is_empty() {
                        return Err(XlogError::UnsupportedEpistemicConstruct {
                            construct: "GPU solver production MaxSAT scheduler".to_string(),
                            context: "UNKNOWN scheduler status requires a diagnostic reason"
                                .to_string(),
                        });
                    }
                    checked_solver_trace_counter_inc!(
                        self,
                        gpu_maxsat_scheduler_unknown_status_jobs
                    );
                    checked_solver_report_counter_inc!(report, unknown_jobs);
                }
                GpuSolverProductionMaxSatScheduleJob::Timeout { budget_micros } => {
                    if *budget_micros == 0 {
                        return Err(XlogError::UnsupportedEpistemicConstruct {
                            construct: "GPU solver production MaxSAT scheduler".to_string(),
                            context: "TIMEOUT scheduler status requires a nonzero budget"
                                .to_string(),
                        });
                    }
                    checked_solver_trace_counter_inc!(
                        self,
                        gpu_maxsat_scheduler_timeout_status_jobs
                    );
                    checked_solver_report_counter_inc!(report, timeout_jobs);
                }
            }
        }

        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    fn require_maxsat_schedule_jobs(
        jobs: &[GpuSolverProductionMaxSatScheduleJob<'_>],
    ) -> Result<()> {
        if jobs.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT scheduler".to_string(),
                context: "accepted MaxSAT scheduler requires at least one GPU job".to_string(),
            });
        }

        for job in jobs {
            match job {
                GpuSolverProductionMaxSatScheduleJob::CandidateSet { candidates } => {
                    Self::require_weighted_maxsat_candidates(candidates)?;
                }
                GpuSolverProductionMaxSatScheduleJob::Search { candidates } => {
                    Self::require_weighted_maxsat_search_candidates(candidates)?;
                }
                GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
                    weighted,
                    selections,
                    ..
                } => {
                    Self::require_weighted_maxsat_encoding_inputs(weighted, selections)?;
                }
                GpuSolverProductionMaxSatScheduleJob::Unknown { reason } => {
                    if reason.trim().is_empty() {
                        return Err(XlogError::UnsupportedEpistemicConstruct {
                            construct: "GPU solver production MaxSAT scheduler".to_string(),
                            context: "UNKNOWN scheduler status requires a diagnostic reason"
                                .to_string(),
                        });
                    }
                }
                GpuSolverProductionMaxSatScheduleJob::Timeout { budget_micros } => {
                    if *budget_micros == 0 {
                        return Err(XlogError::UnsupportedEpistemicConstruct {
                            construct: "GPU solver production MaxSAT scheduler".to_string(),
                            context: "TIMEOUT scheduler status requires a nonzero budget"
                                .to_string(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn require_maxsat_schedule_job_artifacts(
        &self,
        workspace: &GpuCdclWorkspace,
        jobs: &[GpuSolverProductionMaxSatScheduleJob<'_>],
    ) -> Result<()> {
        for job in jobs {
            match job {
                GpuSolverProductionMaxSatScheduleJob::CandidateSet { candidates } => {
                    self.require_weighted_maxsat_candidate_artifacts(candidates)?;
                }
                GpuSolverProductionMaxSatScheduleJob::Search { candidates } => {
                    self.require_weighted_maxsat_search_candidates_and_artifacts(
                        workspace, candidates,
                    )?;
                }
                GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
                    weighted,
                    branch_var_limit,
                    ..
                } => {
                    self.require_workspace_capacity_for_weighted_maxsat_encoding(
                        workspace,
                        weighted,
                        "GPU solver production MaxSAT scheduler",
                    )?;
                    self.require_branch_var_limit_on_adapter_provider(
                        branch_var_limit,
                        "GPU solver production MaxSAT scheduler",
                    )?;
                }
                GpuSolverProductionMaxSatScheduleJob::Unknown { .. }
                | GpuSolverProductionMaxSatScheduleJob::Timeout { .. } => {}
            }
        }
        Ok(())
    }

    /// Execute a heterogeneous MaxSAT schedule once per accepted GPU evidence record.
    ///
    /// The scheduler is a thin production-path adapter: it validates accepted
    /// epistemic GPU execution up front, then dispatches candidate-set,
    /// search-pruning, and weighted encoded-search jobs through the existing GPU
    /// CNF/CDCL helpers. UNKNOWN and TIMEOUT jobs are status propagation records;
    /// they never fall back to CPU assignment or MaxSAT enumeration.
    pub fn solve_maxsat_schedule_with_gpu_execution_results(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        workspace: &mut GpuCdclWorkspace,
        jobs: &[GpuSolverProductionMaxSatScheduleJob<'_>],
    ) -> Result<GpuSolverProductionMaxSatScheduleReport> {
        self.with_trace_rollback(|this| {
            this.solve_maxsat_schedule_with_gpu_execution_results_impl(
                provider, results, workspace, jobs,
            )
        })
    }

    fn solve_maxsat_schedule_with_gpu_execution_results_impl(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        workspace: &mut GpuCdclWorkspace,
        jobs: &[GpuSolverProductionMaxSatScheduleJob<'_>],
    ) -> Result<GpuSolverProductionMaxSatScheduleReport> {
        if results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT scheduler".to_string(),
                context: "MaxSAT scheduler requires at least one accepted GPU result".to_string(),
            });
        }
        Self::require_maxsat_schedule_jobs(jobs)?;
        self.require_workspace_on_adapter_provider(workspace)?;
        self.require_maxsat_schedule_job_artifacts(workspace, jobs)?;
        let states = self.require_accepted_gpu_solver_states(provider, results)?;

        let mut report = GpuSolverProductionMaxSatScheduleReport::default();
        for state in &states {
            let events_before = self.trace.accepted_path_event_snapshot()?;
            let step_report = self.solve_maxsat_schedule_jobs(workspace, jobs)?;
            checked_solver_report_counter_inc!(report, candidate_evidence_records);
            checked_solver_report_counter_add!(report, jobs, step_report.jobs);
            checked_solver_report_counter_add!(
                report,
                candidate_set_jobs,
                step_report.candidate_set_jobs
            );
            checked_solver_report_counter_add!(report, search_jobs, step_report.search_jobs);
            checked_solver_report_counter_add!(
                report,
                encoded_search_jobs,
                step_report.encoded_search_jobs
            );
            checked_solver_report_counter_add!(report, unknown_jobs, step_report.unknown_jobs);
            checked_solver_report_counter_add!(report, timeout_jobs, step_report.timeout_jobs);
            Self::add_maxsat_schedule_step_report(
                &mut report,
                GpuSolverProductionMaxSatReport {
                    optimum_score: step_report.optimum_score,
                    candidates_checked: step_report.candidates_checked,
                    satisfiable_candidates: step_report.satisfiable_candidates,
                    unsat_candidates_pruned: step_report.unsat_candidates_pruned,
                    gpu_cdcl_candidate_encodes: step_report.gpu_cdcl_candidate_encodes,
                    gpu_cdcl_candidate_solves: step_report.gpu_cdcl_candidate_solves,
                    frontier_upper_bound_certificates: step_report
                        .frontier_upper_bound_certificates,
                    ..GpuSolverProductionMaxSatReport::default()
                },
            )?;
            self.record_accepted_gpu_solver_production_path_events_since(events_before, state)?;
            self.record_accepted_gpu_candidate_state(state)?;
        }

        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Execute a heterogeneous MaxSAT schedule once per accepted split-batch component.
    ///
    /// This preserves the scheduler's existing GPU CNF/CDCL dispatch behavior while
    /// requiring aggregate split-batch evidence with zero CPU recomposition,
    /// fallback, and per-candidate host round trips before any scheduled job runs.
    pub fn solve_maxsat_schedule_with_gpu_batch_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        evidence: GpuSolverProductionBatchExecutionEvidence<'_>,
        workspace: &mut GpuCdclWorkspace,
        jobs: &[GpuSolverProductionMaxSatScheduleJob<'_>],
    ) -> Result<GpuSolverProductionMaxSatScheduleReport> {
        self.with_trace_rollback(|this| {
            Self::require_maxsat_schedule_jobs(jobs)?;
            this.require_workspace_on_adapter_provider(workspace)?;
            this.require_maxsat_schedule_job_artifacts(workspace, jobs)?;
            let results =
                this.accepted_solver_results_from_gpu_batch_execution_evidence(provider, evidence)?;
            let report = this.solve_maxsat_schedule_with_gpu_execution_results(
                provider, &results, workspace, jobs,
            )?;
            this.record_accepted_gpu_batch_candidate_evidence(results.len())?;
            this.trace.require_zero_cpu_search()?;
            Ok(report)
        })
    }

    fn solve_portfolio_jobs(
        &mut self,
        jobs: &[GpuSolverProductionPortfolioJob<'_>],
    ) -> Result<GpuSolverProductionPortfolioReport> {
        self.require_portfolio_jobs_and_artifacts(jobs)?;

        let mut report = GpuSolverProductionPortfolioReport::default();
        for job in jobs {
            checked_solver_trace_counter_inc!(self, gpu_portfolio_jobs);
            checked_solver_report_counter_inc!(report, jobs);

            match job {
                GpuSolverProductionPortfolioJob::Sat {
                    cnf,
                    branch_var_limit,
                } => {
                    let _assignment = self
                        .solver
                        .solve_expect_sat_with_branch_limit(cnf, branch_var_limit)?;
                    checked_solver_trace_counter_inc!(self, gpu_cdcl_sat_solves);
                    checked_solver_trace_counter_inc!(self, gpu_portfolio_sat_jobs);
                    checked_solver_report_counter_inc!(report, sat_jobs);
                }
                GpuSolverProductionPortfolioJob::MaxSat { candidates } => {
                    let maxsat = self.solve_weighted_maxsat_candidates(candidates, 0)?;
                    checked_solver_trace_counter_inc!(self, gpu_portfolio_maxsat_jobs);
                    checked_solver_report_counter_inc!(report, maxsat_jobs);
                    Self::add_portfolio_maxsat_report(&mut report, maxsat)?;
                }
                GpuSolverProductionPortfolioJob::EncodedMaxSat {
                    weighted,
                    branch_var_limit,
                    selections,
                } => {
                    let encodes_before = self.trace.gpu_maxsat_candidate_encodes;
                    let certificates_before =
                        self.trace.gpu_maxsat_frontier_upper_bound_certificates;
                    let encoded =
                        self.encode_weighted_maxsat_search_candidates(weighted, selections)?;
                    let frontier_upper_bound_certificates = Self::checked_report_counter_delta(
                        self.trace.gpu_maxsat_frontier_upper_bound_certificates,
                        certificates_before,
                        "gpu_maxsat_frontier_upper_bound_certificates",
                    )?;
                    let search_candidates: Vec<_> = encoded
                        .iter()
                        .map(|candidate| GpuSolverProductionMaxSatSearchCandidate {
                            score: candidate.score,
                            cnf: &candidate.cnf,
                            branch_var_limit,
                            status: candidate.status,
                        })
                        .collect();
                    let mut workspace = self.new_workspace(
                        weighted.num_vars,
                        Self::checked_workspace_clause_cap(weighted)?,
                    )?;
                    let mut maxsat = self.solve_weighted_maxsat_search_candidates(
                        &mut workspace,
                        &search_candidates,
                        frontier_upper_bound_certificates,
                    )?;
                    maxsat.gpu_cdcl_candidate_encodes = Self::checked_report_counter_delta(
                        self.trace.gpu_maxsat_candidate_encodes,
                        encodes_before,
                        "gpu_cdcl_candidate_encodes",
                    )?;
                    checked_solver_trace_counter_inc!(self, gpu_portfolio_maxsat_jobs);
                    checked_solver_report_counter_inc!(report, maxsat_jobs);
                    Self::add_portfolio_maxsat_report(&mut report, maxsat)?;
                }
                GpuSolverProductionPortfolioJob::Unknown { .. } => {
                    checked_solver_trace_counter_inc!(self, gpu_portfolio_unknown_status_jobs);
                    checked_solver_report_counter_inc!(report, unknown_jobs);
                }
                GpuSolverProductionPortfolioJob::Timeout { .. } => {
                    checked_solver_trace_counter_inc!(self, gpu_portfolio_timeout_status_jobs);
                    checked_solver_report_counter_inc!(report, timeout_jobs);
                }
            }
        }

        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    fn require_portfolio_jobs(jobs: &[GpuSolverProductionPortfolioJob<'_>]) -> Result<()> {
        if jobs.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production portfolio".to_string(),
                context: "accepted solver portfolio requires at least one GPU job".to_string(),
            });
        }

        for job in jobs {
            match job {
                GpuSolverProductionPortfolioJob::Sat { .. } => {}
                GpuSolverProductionPortfolioJob::MaxSat { candidates } => {
                    Self::require_weighted_maxsat_candidates(candidates)?;
                }
                GpuSolverProductionPortfolioJob::EncodedMaxSat {
                    weighted,
                    selections,
                    ..
                } => {
                    Self::require_weighted_maxsat_encoding_inputs(weighted, selections)?;
                    Self::checked_workspace_clause_cap(weighted)?;
                }
                GpuSolverProductionPortfolioJob::Unknown { reason } => {
                    if reason.trim().is_empty() {
                        return Err(XlogError::UnsupportedEpistemicConstruct {
                            construct: "GPU solver production portfolio".to_string(),
                            context: "UNKNOWN portfolio status requires a diagnostic reason"
                                .to_string(),
                        });
                    }
                }
                GpuSolverProductionPortfolioJob::Timeout { budget_micros } => {
                    if *budget_micros == 0 {
                        return Err(XlogError::UnsupportedEpistemicConstruct {
                            construct: "GPU solver production portfolio".to_string(),
                            context: "TIMEOUT portfolio status requires a nonzero budget"
                                .to_string(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn require_portfolio_job_artifacts(
        &self,
        jobs: &[GpuSolverProductionPortfolioJob<'_>],
    ) -> Result<()> {
        for job in jobs {
            match job {
                GpuSolverProductionPortfolioJob::Sat {
                    cnf,
                    branch_var_limit,
                } => {
                    self.require_solver_artifact_on_adapter_provider(
                        cnf,
                        branch_var_limit,
                        "GPU solver production portfolio",
                    )?;
                }
                GpuSolverProductionPortfolioJob::MaxSat { candidates } => {
                    self.require_weighted_maxsat_candidate_artifacts(candidates)?;
                }
                GpuSolverProductionPortfolioJob::EncodedMaxSat {
                    branch_var_limit, ..
                } => {
                    self.require_branch_var_limit_on_adapter_provider(
                        branch_var_limit,
                        "GPU solver production portfolio",
                    )?;
                }
                GpuSolverProductionPortfolioJob::Unknown { .. }
                | GpuSolverProductionPortfolioJob::Timeout { .. } => {}
            }
        }
        Ok(())
    }

    fn require_portfolio_jobs_and_artifacts(
        &self,
        jobs: &[GpuSolverProductionPortfolioJob<'_>],
    ) -> Result<()> {
        Self::require_portfolio_jobs(jobs)?;
        self.require_portfolio_job_artifacts(jobs)
    }

    fn add_portfolio_maxsat_report(
        report: &mut GpuSolverProductionPortfolioReport,
        maxsat: GpuSolverProductionMaxSatReport,
    ) -> Result<()> {
        checked_solver_report_counter_add!(report, maxsat_optimum_scores, maxsat.optimum_score);
        checked_solver_report_counter_add!(
            report,
            maxsat_candidates_checked,
            maxsat.candidates_checked
        );
        checked_solver_report_counter_add!(
            report,
            maxsat_satisfiable_candidates,
            maxsat.satisfiable_candidates
        );
        checked_solver_report_counter_add!(
            report,
            maxsat_unsat_candidates_pruned,
            maxsat.unsat_candidates_pruned
        );
        checked_solver_report_counter_add!(
            report,
            maxsat_gpu_cdcl_candidate_encodes,
            maxsat.gpu_cdcl_candidate_encodes
        );
        checked_solver_report_counter_add!(
            report,
            maxsat_gpu_cdcl_candidate_solves,
            maxsat.gpu_cdcl_candidate_solves
        );
        checked_solver_report_counter_add!(
            report,
            maxsat_frontier_upper_bound_certificates,
            maxsat.frontier_upper_bound_certificates
        );
        Ok(())
    }

    fn add_portfolio_report(
        report: &mut GpuSolverProductionPortfolioReport,
        step_report: GpuSolverProductionPortfolioReport,
    ) -> Result<()> {
        checked_solver_report_counter_add!(report, jobs, step_report.jobs);
        checked_solver_report_counter_add!(report, sat_jobs, step_report.sat_jobs);
        checked_solver_report_counter_add!(report, maxsat_jobs, step_report.maxsat_jobs);
        checked_solver_report_counter_add!(report, unknown_jobs, step_report.unknown_jobs);
        checked_solver_report_counter_add!(report, timeout_jobs, step_report.timeout_jobs);
        checked_solver_report_counter_add!(
            report,
            maxsat_optimum_scores,
            step_report.maxsat_optimum_scores
        );
        checked_solver_report_counter_add!(
            report,
            maxsat_candidates_checked,
            step_report.maxsat_candidates_checked
        );
        checked_solver_report_counter_add!(
            report,
            maxsat_satisfiable_candidates,
            step_report.maxsat_satisfiable_candidates
        );
        checked_solver_report_counter_add!(
            report,
            maxsat_unsat_candidates_pruned,
            step_report.maxsat_unsat_candidates_pruned
        );
        checked_solver_report_counter_add!(
            report,
            maxsat_gpu_cdcl_candidate_encodes,
            step_report.maxsat_gpu_cdcl_candidate_encodes
        );
        checked_solver_report_counter_add!(
            report,
            maxsat_gpu_cdcl_candidate_solves,
            step_report.maxsat_gpu_cdcl_candidate_solves
        );
        checked_solver_report_counter_add!(
            report,
            maxsat_frontier_upper_bound_certificates,
            step_report.maxsat_frontier_upper_bound_certificates
        );
        Ok(())
    }

    /// Execute a bounded SAT/MaxSAT/status-aware portfolio after accepted GPU epistemic execution.
    ///
    /// The portfolio is a production adapter over existing GPU CDCL calls. It records
    /// per-job counters and rejects empty portfolios without falling back to the CPU
    /// semantic-oracle solver.
    pub fn solve_portfolio_with_gpu_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        jobs: &[GpuSolverProductionPortfolioJob<'_>],
    ) -> Result<GpuSolverProductionPortfolioReport> {
        self.with_trace_rollback(|this| {
            this.solve_portfolio_with_gpu_execution_result_impl(provider, result, jobs)
        })
    }

    fn solve_portfolio_with_gpu_execution_result_impl(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        jobs: &[GpuSolverProductionPortfolioJob<'_>],
    ) -> Result<GpuSolverProductionPortfolioReport> {
        self.require_portfolio_jobs_and_artifacts(jobs)?;
        let state = self.require_accepted_gpu_solver_evidence(provider, result)?;

        let events_before = self.trace.accepted_path_event_snapshot()?;
        let mut report = self.solve_portfolio_jobs(jobs)?;
        report.candidate_evidence_records = 1;

        self.record_accepted_gpu_solver_production_path_events_since(events_before, &state)?;
        self.record_accepted_gpu_candidate_state(&state)?;
        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Execute the same bounded portfolio once per accepted GPU epistemic candidate.
    pub fn solve_multi_candidate_portfolio_with_gpu_execution_results(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        jobs: &[GpuSolverProductionPortfolioJob<'_>],
    ) -> Result<GpuSolverProductionPortfolioReport> {
        self.with_trace_rollback(|this| {
            this.solve_multi_candidate_portfolio_with_gpu_execution_results_impl(
                provider, results, jobs,
            )
        })
    }

    fn solve_multi_candidate_portfolio_with_gpu_execution_results_impl(
        &mut self,
        provider: &CudaKernelProvider,
        results: &[&EpistemicGpuExecutionResult],
        jobs: &[GpuSolverProductionPortfolioJob<'_>],
    ) -> Result<GpuSolverProductionPortfolioReport> {
        if results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production portfolio".to_string(),
                context: "multi-candidate portfolio requires at least one accepted GPU result"
                    .to_string(),
            });
        }
        self.require_portfolio_jobs_and_artifacts(jobs)?;
        let states = self.require_accepted_gpu_solver_states(provider, results)?;

        let mut report = GpuSolverProductionPortfolioReport::default();
        for state in &states {
            let events_before = self.trace.accepted_path_event_snapshot()?;
            let step_report = self.solve_portfolio_jobs(jobs)?;
            checked_solver_report_counter_inc!(report, candidate_evidence_records);
            Self::add_portfolio_report(&mut report, step_report)?;
            self.record_accepted_gpu_solver_production_path_events_since(events_before, state)?;
            self.record_accepted_gpu_candidate_state(state)?;
        }

        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Execute a bounded SAT/MaxSAT/status-aware portfolio for accepted split/batch evidence.
    ///
    /// The batch evidence must prove every split component reused the existing
    /// single-plan GPU runtime path before each component is delegated to the
    /// existing multi-candidate portfolio adapter.
    pub fn solve_portfolio_with_gpu_batch_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        evidence: GpuSolverProductionBatchExecutionEvidence<'_>,
        jobs: &[GpuSolverProductionPortfolioJob<'_>],
    ) -> Result<GpuSolverProductionPortfolioReport> {
        self.with_trace_rollback(|this| {
            this.require_portfolio_jobs_and_artifacts(jobs)?;
            let results =
                this.accepted_solver_results_from_gpu_batch_execution_evidence(provider, evidence)?;
            let report = this.solve_multi_candidate_portfolio_with_gpu_execution_results(
                provider, &results, jobs,
            )?;
            this.record_accepted_gpu_batch_candidate_evidence(results.len())?;
            this.trace.require_zero_cpu_search()?;
            Ok(report)
        })
    }
}

fn require_accepted_gpu_solver_states(
    provider: &CudaKernelProvider,
    results: &[&EpistemicGpuExecutionResult],
) -> Result<Vec<GpuSolverAcceptedCandidateState>> {
    results
        .iter()
        .map(|result| require_accepted_gpu_solver_evidence(provider, result))
        .collect()
}

fn require_accepted_gpu_solver_evidence(
    provider: &CudaKernelProvider,
    result: &EpistemicGpuExecutionResult,
) -> Result<GpuSolverAcceptedCandidateState> {
    let provider_identity = EpistemicGpuProviderIdentity::from_provider(provider);
    if result.provider_identity != provider_identity {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: format!(
                "solver evidence provider mismatch: result device={} provider device={} \
                 result_device_ptr={} provider_device_ptr={} result_memory_ptr={} \
                 provider_memory_ptr={}",
                result.provider_identity.device_ordinal,
                provider_identity.device_ordinal,
                result.provider_identity.device_ptr,
                provider_identity.device_ptr,
                result.provider_identity.memory_ptr,
                provider_identity.memory_ptr
            ),
        });
    }
    if !result.prepared.preflight.cpu_fallbacks.is_zero() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: "solver evidence requires zero epistemic CPU fallback counters".to_string(),
        });
    }
    if result.candidate_generation.literal_count == 0
        || result.prepared.preflight.tuple_membership_binding_count == 0
    {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: format!(
                "solver evidence requires at least one GPU-validated epistemic literal and \
                 tuple-membership binding, got literals={} bindings={}",
                result.candidate_generation.literal_count,
                result.prepared.preflight.tuple_membership_binding_count
            ),
        });
    }
    if result.prepared.preflight.solver_assumption_binding_count == 0 {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: "solver evidence requires planner-exported solver assumption bindings"
                .to_string(),
        });
    }
    if result.prepared.preflight.solver_required_capability_count
        < PRODUCTION_SOLVER_REQUIRED_CAPABILITY_COUNT as usize
        || result.prepared.preflight.solver_required_status_count
            < PRODUCTION_SOLVER_REQUIRED_STATUS_COUNT as usize
    {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: format!(
                "solver evidence requires the v0.9 production capability/status contract, got \
                 capabilities={} statuses={}",
                result.prepared.preflight.solver_required_capability_count,
                result.prepared.preflight.solver_required_status_count
            ),
        });
    }
    result.require_runtime_dispatch_certification()?;
    result
        .model_membership
        .require_stable_model_tuple_source()?;
    if result.constraint_validation.violated_constraint_relations != 0 {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: format!(
                "solver evidence requires zero reduced constraint violations, got {} across {} \
                 checked constraint relations",
                result.constraint_validation.violated_constraint_relations,
                result.constraint_validation.checked_constraint_relations
            ),
        });
    }
    if result.constraint_validation.row_count_device_reads as usize
        > result.constraint_validation.checked_constraint_relations
    {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: format!(
                "solver evidence constraint metadata reads cannot exceed checked reduced \
                 constraint relations, got reads={} checked={}",
                result.constraint_validation.row_count_device_reads,
                result.constraint_validation.checked_constraint_relations
            ),
        });
    }
    require_gpu_kernel_trace(
        "candidate generation",
        result.candidate_generation.kernel_launches,
        result.candidate_generation.host_write_ops,
        result.candidate_generation.kernel_timing,
    )?;
    require_gpu_kernel_trace(
        "candidate propagation",
        result.propagation.kernel_launches,
        result.propagation.host_write_ops,
        result.propagation.kernel_timing,
    )?;
    require_gpu_kernel_trace(
        "candidate validation",
        result.candidate_validation.kernel_launches,
        result.candidate_validation.host_write_ops,
        result.candidate_validation.kernel_timing,
    )?;
    require_gpu_kernel_trace(
        "model membership",
        result.model_membership.kernel_launches,
        result.model_membership.host_write_ops,
        result.model_membership.kernel_timing,
    )?;
    require_gpu_kernel_trace(
        "world-view validation",
        result.world_view_validation.kernel_launches,
        result.world_view_validation.host_write_ops,
        result.world_view_validation.kernel_timing,
    )?;
    require_gpu_kernel_trace(
        "accepted-candidate materialization",
        result.materialization.kernel_launches,
        result.materialization.host_write_ops,
        result.materialization.kernel_timing,
    )?;
    require_gpu_kernel_trace(
        "final-result materialization",
        result.final_result_materialization.kernel_launches,
        result.final_result_materialization.host_write_ops,
        result.final_result_materialization.kernel_timing,
    )?;
    require_gpu_kernel_trace(
        "final tuple materialization",
        result.final_tuple_materialization.kernel_launches,
        result.final_tuple_materialization.host_write_ops,
        result.final_tuple_materialization.kernel_timing,
    )?;
    // The runtime has already captured this via read_device_row_count during
    // the bounded final-result transfer; do not re-read it in the solver gate.
    let accepted_rows = result.final_result_transfer.final_output_rows;
    result
        .final_tuple_materialization
        .require_row_filter_materialization_evidence(
            "accepted GPU solver candidate evidence",
            accepted_rows,
        )?;
    if result.transfer_budget.tracked_dtoh_calls != 0
        || result.transfer_budget.tracked_htod_calls != 0
        || result.transfer_budget.tracked_data_plane_htod_calls != 0
        || result.transfer_budget.per_candidate_host_round_trips != 0
    {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: format!(
                "solver evidence requires zero hot-path transfers outside bounded launch \
                 metadata, got dtoh_calls={}, htod_calls={}, data_plane_htod_calls={}, \
                 launch_metadata_htod_calls={}, per_candidate_round_trips={}",
                result.transfer_budget.tracked_dtoh_calls,
                result.transfer_budget.tracked_htod_calls,
                result.transfer_budget.tracked_data_plane_htod_calls,
                result.transfer_budget.tracked_launch_metadata_htod_calls,
                result.transfer_budget.per_candidate_host_round_trips
            ),
        });
    }
    require_accepted_gpu_solver_semantic_trace(result)?;

    if accepted_rows == 0 {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: "solver evidence requires non-empty accepted GPU final output".to_string(),
        });
    }
    if result.semantic_trace.accepted_candidates == 0 {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: "solver evidence requires at least one GPU-accepted candidate".to_string(),
        });
    }

    Ok(GpuSolverAcceptedCandidateState::from_validated_result(
        result,
        accepted_rows,
    ))
}

fn require_accepted_gpu_solver_semantic_trace(result: &EpistemicGpuExecutionResult) -> Result<()> {
    let trace = &result.semantic_trace;
    let accounted_candidates = trace
        .accepted_candidates
        .checked_add(trace.rejected_candidates)
        .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: format!(
                "solver evidence semantic trace candidate accounting overflowed: accepted={} \
                 rejected={}",
                trace.accepted_candidates, trace.rejected_candidates
            ),
        })?;
    if trace.generated_candidates != result.candidate_generation.generated_candidates
        || trace.tested_candidates != result.world_view_validation.candidates_checked
        || trace.accepted_candidates != trace.accepted_candidate_indices.len()
        || trace.rejected_candidates != trace.rejected_candidate_indices.len()
        || trace.accepted_world_views != trace.accepted_candidates
        || accounted_candidates != trace.generated_candidates
        || trace.cpu_candidate_enumerations != 0
        || trace.cpu_world_view_validations != 0
    {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: format!(
                "solver evidence requires a consistent GPU semantic trace with zero CPU \
                 fallbacks, got generated={}, tested={}, expected_generated={}, \
                 expected_tested={}, accepted={} accepted_indices={}, accepted_world_views={}, \
                 rejected={} rejected_indices={}, cpu_candidates={}, cpu_world_views={}",
                trace.generated_candidates,
                trace.tested_candidates,
                result.candidate_generation.generated_candidates,
                result.world_view_validation.candidates_checked,
                trace.accepted_candidates,
                trace.accepted_candidate_indices.len(),
                trace.accepted_world_views,
                trace.rejected_candidates,
                trace.rejected_candidate_indices.len(),
                trace.cpu_candidate_enumerations,
                trace.cpu_world_view_validations
            ),
        });
    }
    Ok(())
}

fn require_accepted_gpu_solver_batch_evidence<'a>(
    provider: &CudaKernelProvider,
    batch: &'a EpistemicGpuBatchExecutionResult,
) -> Result<Vec<&'a EpistemicGpuExecutionResult>> {
    if batch.results.is_empty() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver batch evidence".to_string(),
            context: "solver batch evidence requires at least one accepted GPU component"
                .to_string(),
        });
    }
    let trace = batch.trace;
    if trace.component_count != batch.results.len()
        || trace.gpu_runtime_component_executions != batch.results.len()
        || trace.cpu_recomposition_steps != 0
        || trace.cpu_candidate_enumerations != 0
        || trace.cpu_world_view_validations != 0
        || trace.cpu_solver_search_fallbacks != 0
        || trace.cpu_probability_recomputations != 0
        || trace.tracked_dtoh_calls != 0
        || trace.tracked_htod_calls != 0
        || trace.tracked_data_plane_htod_calls != 0
        || trace.per_candidate_host_round_trips != 0
        || trace.violated_constraint_relations != 0
        || !trace.aggregate_kernel_timing.is_recorded()
    {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver batch evidence".to_string(),
            context: format!(
                "solver batch evidence requires complete GPU component execution and zero \
                 CPU/host fallback counters outside bounded launch metadata plus aggregate \
                 CUDA-event timing, got \
                 components={}/{}, recomposition={}, cpu_candidates={}, cpu_world_views={}, \
                 cpu_solver_search={}, cpu_probability_recompute={}, dtoh_calls={}, \
                 htod_calls={}, data_plane_htod_calls={}, launch_metadata_htod_calls={}, \
                 round_trips={}, constraint_violations={}, aggregate_timing_recorded={}",
                trace.gpu_runtime_component_executions,
                trace.component_count,
                trace.cpu_recomposition_steps,
                trace.cpu_candidate_enumerations,
                trace.cpu_world_view_validations,
                trace.cpu_solver_search_fallbacks,
                trace.cpu_probability_recomputations,
                trace.tracked_dtoh_calls,
                trace.tracked_htod_calls,
                trace.tracked_data_plane_htod_calls,
                trace.tracked_launch_metadata_htod_calls,
                trace.per_candidate_host_round_trips,
                trace.violated_constraint_relations,
                trace.aggregate_kernel_timing.is_recorded()
            ),
        });
    }
    batch.require_trace_matches_components("accepted GPU solver batch evidence")?;

    let mut results = Vec::with_capacity(batch.results.len());
    for result in &batch.results {
        require_accepted_gpu_solver_evidence(provider, result)?;
        results.push(result);
    }
    Ok(results)
}

fn require_same_gpu_cnf_for_learned_clause_reuse(source: &GpuCnf, target: &GpuCnf) -> Result<()> {
    let same_shape = source.var_cap == target.var_cap
        && source.clause_cap == target.clause_cap
        && source.lit_cap == target.lit_cap;
    let same_buffers = source.num_vars.device_ptr_value() == target.num_vars.device_ptr_value()
        && source.num_clauses.device_ptr_value() == target.num_clauses.device_ptr_value()
        && source.num_lits.device_ptr_value() == target.num_lits.device_ptr_value()
        && source.clause_offsets.device_ptr_value() == target.clause_offsets.device_ptr_value()
        && source.literals.device_ptr_value() == target.literals.device_ptr_value();
    if !same_shape || !same_buffers {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "GPU solver learned-clause reuse".to_string(),
            context: "learned-clause import is currently certified only for the same \
                 device-resident CNF; distinct candidate CNFs must not reuse imported clauses"
                .to_string(),
        });
    }
    Ok(())
}

fn require_stable_learned_clause_arena(
    phase: &'static str,
    workspace: &GpuCdclWorkspace,
    learned_offsets_ptr: cudarc::driver::sys::CUdeviceptr,
    learned_lits_ptr: cudarc::driver::sys::CUdeviceptr,
    proof_offsets_ptr: cudarc::driver::sys::CUdeviceptr,
    proof_data_ptr: cudarc::driver::sys::CUdeviceptr,
    learned_count_ptr: cudarc::driver::sys::CUdeviceptr,
) -> Result<()> {
    if workspace.learned_offsets.device_ptr_value() != learned_offsets_ptr
        || workspace.learned_lits.device_ptr_value() != learned_lits_ptr
        || workspace.proof_offsets.device_ptr_value() != proof_offsets_ptr
        || workspace.proof_data.device_ptr_value() != proof_data_ptr
        || workspace.out_learned_count.device_ptr_value() != learned_count_ptr
    {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "GPU solver learned-clause reuse".to_string(),
            context: format!("learned-clause {phase} must keep the reusable GPU workspace arena"),
        });
    }
    Ok(())
}

fn require_gpu_kernel_trace(
    phase: &'static str,
    kernel_launches: u32,
    host_write_ops: u32,
    kernel_timing: EpistemicGpuKernelTimingTrace,
) -> Result<()> {
    if kernel_launches == 0 || host_write_ops != 0 || !kernel_timing.is_recorded() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: format!(
                "solver evidence requires GPU {phase} trace with nonzero launches and \
                 zero host writes plus CUDA-event timing, got launches={kernel_launches}, \
                 host_writes={host_write_ops}, timing_recorded={}",
                kernel_timing.is_recorded()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xlog_core::MemoryBudget;
    use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

    use super::*;
    use crate::{Clause, Literal};

    fn try_provider() -> Option<Arc<CudaKernelProvider>> {
        let device = match CudaDevice::new(0) {
            Ok(device) => Arc::new(device),
            Err(err) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {err}");
                return None;
            }
        };
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        match CudaKernelProvider::new(device, memory) {
            Ok(provider) => Some(Arc::new(provider)),
            Err(err) => {
                eprintln!("Skipping test: failed to create CUDA kernel provider: {err}");
                None
            }
        }
    }

    fn alloc_u32(
        provider: &Arc<CudaKernelProvider>,
        value: u32,
    ) -> xlog_cuda::memory::TrackedCudaSlice<u32> {
        let memory = provider.memory();
        let mut slot = memory.alloc::<u32>(1).expect("alloc u32 scalar");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&[value], &mut slot)
            .expect("upload u32 scalar");
        slot
    }

    #[test]
    fn weighted_maxsat_frontier_completion_fails_closed_before_cpu_expansion() {
        let clauses: Vec<_> = (0..18)
            .map(|idx| {
                let lit = if idx % 2 == 0 {
                    Literal::positive(0)
                } else {
                    Literal::negative(0)
                };
                Clause::new(vec![lit])
            })
            .collect();
        let weighted = SolveInstance::with_weights(1, clauses, vec![1.0; 18]);
        let first_frontier: Vec<_> = (0..9).collect();
        let second_frontier: Vec<_> = (9..18).collect();
        let selections = [
            GpuSolverProductionWeightedMaxSatSelection {
                soft_clause_indices: &first_frontier,
                status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
            },
            GpuSolverProductionWeightedMaxSatSelection {
                soft_clause_indices: &second_frontier,
                status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
            },
        ];

        let result = GpuSolverProductionAdapter::complete_weighted_maxsat_frontier_selections(
            &weighted,
            weighted.weights.as_ref().expect("weighted MaxSAT weights"),
            &selections,
        );
        let Err(err) = result else {
            panic!("frontier completion should reject CPU combinatorial expansion");
        };
        let message = err.to_string();
        assert!(message.contains("frontier completion"));
        assert!(message.contains("explicit GPU scheduler selections"));
    }

    #[test]
    fn encoded_weighted_maxsat_search_runs_real_gpu_sat_unsat_candidates() {
        let Some(provider) = try_provider() else {
            return;
        };

        let weighted = SolveInstance::with_weights(
            1,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::negative(0)]),
            ],
            vec![2.0, 1.0],
        );
        let mut adapter =
            GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
        let mut workspace = adapter
            .new_workspace(weighted.num_vars, weighted.clauses.len() as u32)
            .expect("new MaxSAT workspace");
        let branch_limit = alloc_u32(&provider, weighted.num_vars);
        let contradictory_selection = [0usize, 1usize];
        let selections = [GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &contradictory_selection,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        }];

        let certificates_before = adapter.trace.gpu_maxsat_frontier_upper_bound_certificates;
        let encoded = adapter
            .encode_weighted_maxsat_search_candidates(&weighted, &selections)
            .expect("encode weighted MaxSAT candidates");
        let frontier_upper_bound_certificates =
            GpuSolverProductionAdapter::checked_report_counter_delta(
                adapter.trace.gpu_maxsat_frontier_upper_bound_certificates,
                certificates_before,
                "gpu_maxsat_frontier_upper_bound_certificates",
            )
            .expect("frontier certificate delta");
        let search_candidates: Vec<_> = encoded
            .iter()
            .map(|candidate| GpuSolverProductionMaxSatSearchCandidate {
                score: candidate.score,
                cnf: &candidate.cnf,
                branch_var_limit: &branch_limit,
                status: candidate.status,
            })
            .collect();

        let report = adapter
            .solve_weighted_maxsat_search_candidates(
                &mut workspace,
                &search_candidates,
                frontier_upper_bound_certificates,
            )
            .expect("GPU weighted MaxSAT search");

        assert_eq!(report.candidate_evidence_records, 0);
        assert_eq!(report.optimum_score, 2);
        assert_eq!(report.candidates_checked, 2);
        assert_eq!(report.satisfiable_candidates, 1);
        assert_eq!(report.unsat_candidates_pruned, 1);
        assert_eq!(report.gpu_cdcl_candidate_solves, 2);
        assert_eq!(report.frontier_upper_bound_certificates, 1);

        let trace = adapter.trace();
        assert_eq!(trace.gpu_maxsat_candidate_encodes, 2);
        assert_eq!(trace.gpu_maxsat_frontier_completion_candidate_encodes, 1);
        assert_eq!(trace.gpu_maxsat_frontier_upper_bound_certificates, 1);
        assert_eq!(trace.gpu_maxsat_candidate_solves, 2);
        assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 1);
        assert_eq!(trace.gpu_cdcl_sat_solves, 1);
        assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 1);
        assert_eq!(trace.gpu_maxsat_optima, 1);
        assert_eq!(trace.cpu_assignment_enumerations, 0);
        assert_eq!(trace.cpu_maxsat_enumerations, 0);
        trace
            .require_zero_cpu_search()
            .expect("MaxSAT production search must not use CPU search");
    }

    #[test]
    fn portfolio_jobs_dispatch_real_gpu_sat_and_encoded_maxsat_paths() {
        let Some(provider) = try_provider() else {
            return;
        };

        let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
        let sat_cnf = GpuCnf::from_host(&sat_instance, &provider).expect("SAT GpuCnf upload");
        let weighted = SolveInstance::with_weights(
            1,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::negative(0)]),
            ],
            vec![2.0, 1.0],
        );
        let branch_limit = alloc_u32(&provider, weighted.num_vars);
        let contradictory_selection = [0usize, 1usize];
        let selections = [GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &contradictory_selection,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        }];
        let jobs = [
            GpuSolverProductionPortfolioJob::Sat {
                cnf: &sat_cnf,
                branch_var_limit: &branch_limit,
            },
            GpuSolverProductionPortfolioJob::EncodedMaxSat {
                weighted: &weighted,
                branch_var_limit: &branch_limit,
                selections: &selections,
            },
            GpuSolverProductionPortfolioJob::Unknown {
                reason: "bounded portfolio diagnostic",
            },
            GpuSolverProductionPortfolioJob::Timeout { budget_micros: 1 },
        ];
        let mut adapter =
            GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());

        let report = adapter
            .solve_portfolio_jobs(&jobs)
            .expect("GPU production portfolio jobs");

        assert_eq!(report.candidate_evidence_records, 0);
        assert_eq!(report.jobs, 4);
        assert_eq!(report.sat_jobs, 1);
        assert_eq!(report.maxsat_jobs, 1);
        assert_eq!(report.unknown_jobs, 1);
        assert_eq!(report.timeout_jobs, 1);
        assert_eq!(report.maxsat_optimum_scores, 2);
        assert_eq!(report.maxsat_candidates_checked, 2);
        assert_eq!(report.maxsat_satisfiable_candidates, 1);
        assert_eq!(report.maxsat_unsat_candidates_pruned, 1);
        assert_eq!(report.maxsat_gpu_cdcl_candidate_encodes, 2);
        assert_eq!(report.maxsat_gpu_cdcl_candidate_solves, 2);
        assert_eq!(report.maxsat_frontier_upper_bound_certificates, 1);

        let trace = adapter.trace();
        assert_eq!(trace.gpu_portfolio_jobs, 4);
        assert_eq!(trace.gpu_portfolio_sat_jobs, 1);
        assert_eq!(trace.gpu_portfolio_maxsat_jobs, 1);
        assert_eq!(trace.gpu_portfolio_unknown_status_jobs, 1);
        assert_eq!(trace.gpu_portfolio_timeout_status_jobs, 1);
        assert_eq!(trace.gpu_cdcl_sat_solves, 2);
        assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 1);
        assert_eq!(trace.gpu_maxsat_candidate_encodes, 2);
        assert_eq!(trace.gpu_maxsat_candidate_solves, 2);
        assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 1);
        assert_eq!(trace.gpu_maxsat_optima, 1);
        assert_eq!(trace.cpu_assignment_enumerations, 0);
        assert_eq!(trace.cpu_maxsat_enumerations, 0);
        trace
            .require_zero_cpu_search()
            .expect("portfolio production search must not use CPU search");
    }

    #[test]
    fn learned_clause_reuse_publishes_and_imports_gpu_workspace_arena() {
        let Some(provider) = try_provider() else {
            return;
        };

        let unsat_instance = SolveInstance::new(
            1,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::negative(0)]),
            ],
        );
        let cnf = GpuCnf::from_host(&unsat_instance, &provider).expect("UNSAT GpuCnf upload");
        let branch_limit = alloc_u32(&provider, cnf.var_cap as u32);
        let mut adapter =
            GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
        let mut workspace = adapter
            .new_workspace(cnf.var_cap, cnf.clause_cap)
            .expect("new learned-clause workspace");

        let learned_offsets_ptr = workspace.learned_offsets.device_ptr_value();
        let learned_lits_ptr = workspace.learned_lits.device_ptr_value();
        let proof_offsets_ptr = workspace.proof_offsets.device_ptr_value();
        let proof_data_ptr = workspace.proof_data.device_ptr_value();
        let learned_count_ptr = workspace.out_learned_count.device_ptr_value();

        let report = adapter
            .solve_unsat_then_reuse_learned_clauses(
                &mut workspace,
                &cnf,
                &branch_limit,
                &cnf,
                &branch_limit,
            )
            .expect("GPU learned-clause reuse");

        assert_eq!(report.candidate_evidence_records, 0);
        assert_eq!(report.candidates, 2);
        assert_eq!(report.unsat_solves, 2);
        assert_eq!(report.gpu_learned_clause_arena_publications, 1);
        assert_eq!(report.gpu_learned_clause_imports, 1);
        assert_eq!(report.gpu_learned_clause_reused_solves, 1);
        assert_eq!(report.cpu_learned_clause_transfers, 0);
        assert_eq!(
            workspace.learned_offsets.device_ptr_value(),
            learned_offsets_ptr
        );
        assert_eq!(workspace.learned_lits.device_ptr_value(), learned_lits_ptr);
        assert_eq!(
            workspace.proof_offsets.device_ptr_value(),
            proof_offsets_ptr
        );
        assert_eq!(workspace.proof_data.device_ptr_value(), proof_data_ptr);
        assert_eq!(
            workspace.out_learned_count.device_ptr_value(),
            learned_count_ptr
        );

        let trace = adapter.trace();
        assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
        assert_eq!(trace.gpu_learned_clause_arena_publications, 1);
        assert_eq!(trace.gpu_learned_count_buffer_publications, 1);
        assert_eq!(trace.gpu_learned_clause_imports, 1);
        assert_eq!(trace.gpu_learned_clause_reused_solves, 1);
        assert_eq!(trace.cpu_assignment_enumerations, 0);
        assert_eq!(trace.cpu_maxsat_enumerations, 0);
        assert_eq!(trace.cpu_learned_clause_transfers, 0);
        trace
            .require_zero_cpu_search()
            .expect("learned-clause production reuse must not use CPU search");
    }
}
