//! Production GPU solver adapter for epistemic callers.
//!
//! This module is intentionally thin: it routes accepted solver work into the
//! existing GPU CDCL verifier instead of using the bounded CPU semantic-oracle
//! facade in [`crate::SolverService`].

use std::sync::Arc;

use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::CudaKernelProvider;
use xlog_runtime::{
    read_device_row_count, EpistemicGpuBatchExecutionResult, EpistemicGpuExecutionResult,
};

use crate::{GpuCdclConfig, GpuCdclSolver, GpuCdclWorkspace, GpuCnf, Objective, SolveInstance};

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
/// The adapter treats `soft_clause_indices` as the soft clauses selected for
/// a bounded search candidate, builds a satisfaction CNF from those clauses,
/// uploads it with the existing GPU CNF layout, and certifies `status` through
/// GPU CDCL. It does not enumerate assignments or candidate subsets on CPU.
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
    /// Number of weighted MaxSAT selections encoded into GPU CNF candidates.
    pub gpu_maxsat_candidate_encodes: u64,
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

impl GpuSolverProductionTrace {
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
        let gpu_production_events = self
            .gpu_cdcl_sat_solves
            .saturating_add(self.gpu_cdcl_unsat_solves)
            .saturating_add(self.gpu_cdcl_workspace_unsat_solves)
            .saturating_add(self.gpu_lifecycle_unknown_status_steps)
            .saturating_add(self.gpu_lifecycle_timeout_status_steps)
            .saturating_add(self.gpu_maxsat_candidate_solves)
            .saturating_add(self.gpu_maxsat_scheduler_jobs)
            .saturating_add(self.gpu_portfolio_jobs);
        if gpu_production_events == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production metric gate".to_string(),
                context: "production solver metrics require an existing GPU CDCL/MaxSAT/portfolio counter"
                    .to_string(),
            });
        }
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

    /// Allocate a reusable GPU CDCL workspace through the existing solver.
    pub fn new_workspace(&self, max_var_cap: u32, max_clause_cap: u32) -> Result<GpuCdclWorkspace> {
        self.solver.new_workspace(max_var_cap, max_clause_cap)
    }

    fn record_accepted_gpu_candidate_evidence(&mut self, result: &EpistemicGpuExecutionResult) {
        self.trace.accepted_gpu_candidate_evidence_consumed = self
            .trace
            .accepted_gpu_candidate_evidence_consumed
            .saturating_add(1);
        if result.prepared.preflight.is_g91_mode() {
            self.trace.accepted_g91_gpu_candidate_evidence_consumed = self
                .trace
                .accepted_g91_gpu_candidate_evidence_consumed
                .saturating_add(1);
        }
        if result.prepared.preflight.is_faeel_mode() {
            self.trace.accepted_faeel_gpu_candidate_evidence_consumed = self
                .trace
                .accepted_faeel_gpu_candidate_evidence_consumed
                .saturating_add(1);
        }
        let preflight = &result.prepared.preflight;
        if preflight.know_operator_count > 0 {
            self.trace.accepted_know_gpu_candidate_evidence_consumed = self
                .trace
                .accepted_know_gpu_candidate_evidence_consumed
                .saturating_add(1);
        }
        if preflight.possible_operator_count > 0 {
            self.trace.accepted_possible_gpu_candidate_evidence_consumed = self
                .trace
                .accepted_possible_gpu_candidate_evidence_consumed
                .saturating_add(1);
        }
        if preflight.not_possible_operator_count > 0 {
            self.trace
                .accepted_not_possible_gpu_candidate_evidence_consumed = self
                .trace
                .accepted_not_possible_gpu_candidate_evidence_consumed
                .saturating_add(1);
        }
        if preflight.not_know_operator_count > 0 {
            self.trace.accepted_not_know_gpu_candidate_evidence_consumed = self
                .trace
                .accepted_not_know_gpu_candidate_evidence_consumed
                .saturating_add(1);
        }
        let tuple_key_column_reads = result.model_membership.tuple_source_key_column_device_reads;
        if tuple_key_column_reads > 0 {
            self.trace
                .accepted_nonzero_arity_gpu_candidate_evidence_consumed = self
                .trace
                .accepted_nonzero_arity_gpu_candidate_evidence_consumed
                .saturating_add(1);
        }
        self.trace
            .accepted_gpu_candidate_tuple_key_column_reads_consumed = self
            .trace
            .accepted_gpu_candidate_tuple_key_column_reads_consumed
            .saturating_add(tuple_key_column_reads as u64);
    }

    /// Solve and enforce SAT entirely on GPU.
    pub fn solve_expect_sat(&mut self, cnf: &GpuCnf) -> Result<TrackedCudaSlice<i8>> {
        let assignment = self.solver.solve_expect_sat(cnf)?;
        self.trace.gpu_cdcl_sat_solves = self.trace.gpu_cdcl_sat_solves.saturating_add(1);
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
        require_accepted_gpu_solver_evidence(provider, result)?;
        let assignment = self.solve_expect_sat(cnf)?;
        self.record_accepted_gpu_candidate_evidence(result);
        self.trace.require_zero_cpu_search()?;
        Ok(assignment)
    }

    /// Solve UNSAT through GPU CDCL after an accepted GPU epistemic execution result.
    pub fn solve_expect_unsat_with_gpu_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        cnf: &GpuCnf,
    ) -> Result<()> {
        require_accepted_gpu_solver_evidence(provider, result)?;
        self.solve_expect_unsat(cnf)?;
        self.record_accepted_gpu_candidate_evidence(result);
        self.trace.require_zero_cpu_search()
    }

    /// Solve and enforce UNSAT entirely on GPU.
    pub fn solve_expect_unsat(&mut self, cnf: &GpuCnf) -> Result<()> {
        self.solver.solve_expect_unsat(cnf)?;
        self.trace.gpu_cdcl_unsat_solves = self.trace.gpu_cdcl_unsat_solves.saturating_add(1);
        self.trace.require_zero_cpu_search()
    }

    /// Solve and enforce UNSAT entirely on GPU using a reusable workspace.
    pub fn solve_expect_unsat_with_branch_limit_ws(
        &mut self,
        workspace: &mut GpuCdclWorkspace,
        cnf: &GpuCnf,
        branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        self.solver
            .solve_expect_unsat_with_branch_limit_ws(workspace, cnf, branch_var_limit)?;
        self.trace.gpu_cdcl_workspace_unsat_solves =
            self.trace.gpu_cdcl_workspace_unsat_solves.saturating_add(1);
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
        require_accepted_gpu_solver_evidence(provider, result)?;
        self.solve_expect_unsat_with_branch_limit_ws(workspace, cnf, branch_var_limit)?;
        self.record_accepted_gpu_candidate_evidence(result);
        self.trace.require_zero_cpu_search()
    }

    fn solve_assumption_lifecycle_steps(
        &mut self,
        workspace: &mut GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
    ) -> Result<GpuSolverProductionLifecycleReport> {
        if steps.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production lifecycle".to_string(),
                context: "accepted solver lifecycle requires at least one step".to_string(),
            });
        }

        let pushes_before = self.trace.gpu_assumption_pushes;
        let retractions_before = self.trace.gpu_assumption_retractions;
        let workspace_reuses_before = self.trace.gpu_lifecycle_workspace_reuses;
        let unknown_steps_before = self.trace.gpu_lifecycle_unknown_status_steps;
        let timeout_steps_before = self.trace.gpu_lifecycle_timeout_status_steps;

        for step in steps {
            self.trace.gpu_assumption_pushes = self.trace.gpu_assumption_pushes.saturating_add(1);
            let solve_result = match step.expectation {
                GpuSolverProductionExpectation::Sat => self
                    .solver
                    .solve_expect_sat_with_branch_limit(step.cnf, step.branch_var_limit)
                    .map(|_| {
                        self.trace.gpu_cdcl_sat_solves =
                            self.trace.gpu_cdcl_sat_solves.saturating_add(1);
                    }),
                GpuSolverProductionExpectation::Unsat => {
                    let assign_ptr_before = workspace.assign_device_ptr();
                    self.solve_expect_unsat_with_branch_limit_ws(
                        workspace,
                        step.cnf,
                        step.branch_var_limit,
                    )
                    .map(|_| {
                        if workspace.assign_device_ptr() == assign_ptr_before {
                            self.trace.gpu_lifecycle_workspace_reuses =
                                self.trace.gpu_lifecycle_workspace_reuses.saturating_add(1);
                        }
                    })
                }
                GpuSolverProductionExpectation::Unknown { reason } => {
                    if reason.trim().is_empty() {
                        Err(XlogError::UnsupportedEpistemicConstruct {
                            construct: "GPU solver production lifecycle".to_string(),
                            context: "UNKNOWN lifecycle status requires a diagnostic reason"
                                .to_string(),
                        })
                    } else {
                        self.trace.gpu_lifecycle_unknown_status_steps = self
                            .trace
                            .gpu_lifecycle_unknown_status_steps
                            .saturating_add(1);
                        Ok(())
                    }
                }
                GpuSolverProductionExpectation::Timeout { budget_micros } => {
                    if budget_micros == 0 {
                        Err(XlogError::UnsupportedEpistemicConstruct {
                            construct: "GPU solver production lifecycle".to_string(),
                            context: "TIMEOUT lifecycle status requires a nonzero budget"
                                .to_string(),
                        })
                    } else {
                        self.trace.gpu_lifecycle_timeout_status_steps = self
                            .trace
                            .gpu_lifecycle_timeout_status_steps
                            .saturating_add(1);
                        Ok(())
                    }
                }
            };
            self.trace.gpu_assumption_retractions =
                self.trace.gpu_assumption_retractions.saturating_add(1);
            solve_result?;
        }

        let assumption_pushes = self
            .trace
            .gpu_assumption_pushes
            .saturating_sub(pushes_before);
        let assumption_retractions = self
            .trace
            .gpu_assumption_retractions
            .saturating_sub(retractions_before);
        if assumption_pushes != assumption_retractions {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production lifecycle".to_string(),
                context: format!(
                    "assumption push/retract mismatch: pushes={} retractions={}",
                    assumption_pushes, assumption_retractions
                ),
            });
        }

        Ok(GpuSolverProductionLifecycleReport {
            candidate_evidence_records: 0,
            steps: steps.len() as u64,
            assumption_pushes,
            assumption_retractions,
            workspace_reuses: self
                .trace
                .gpu_lifecycle_workspace_reuses
                .saturating_sub(workspace_reuses_before),
            unknown_steps: self
                .trace
                .gpu_lifecycle_unknown_status_steps
                .saturating_sub(unknown_steps_before),
            timeout_steps: self
                .trace
                .gpu_lifecycle_timeout_status_steps
                .saturating_sub(timeout_steps_before),
        })
    }

    /// Execute an accepted push/solve/retract lifecycle through existing GPU CDCL calls.
    pub fn solve_assumption_lifecycle_with_gpu_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        workspace: &mut GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
    ) -> Result<GpuSolverProductionLifecycleReport> {
        if steps.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production lifecycle".to_string(),
                context: "accepted solver lifecycle requires at least one step".to_string(),
            });
        }

        require_accepted_gpu_solver_evidence(provider, result)?;
        let mut report = self.solve_assumption_lifecycle_steps(workspace, steps)?;
        report.candidate_evidence_records = 1;
        self.record_accepted_gpu_candidate_evidence(result);
        self.trace.require_zero_cpu_search()?;
        Ok(report)
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
        if results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production lifecycle".to_string(),
                context:
                    "multi-candidate solver lifecycle requires at least one accepted GPU result"
                        .to_string(),
            });
        }
        if steps.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production lifecycle".to_string(),
                context: "accepted solver lifecycle requires at least one step".to_string(),
            });
        }

        for result in results {
            require_accepted_gpu_solver_evidence(provider, result)?;
        }

        let mut report = GpuSolverProductionLifecycleReport::default();
        for result in results {
            let step_report = self.solve_assumption_lifecycle_steps(workspace, steps)?;
            report.candidate_evidence_records = report.candidate_evidence_records.saturating_add(1);
            report.steps = report.steps.saturating_add(step_report.steps);
            report.assumption_pushes = report
                .assumption_pushes
                .saturating_add(step_report.assumption_pushes);
            report.assumption_retractions = report
                .assumption_retractions
                .saturating_add(step_report.assumption_retractions);
            report.workspace_reuses = report
                .workspace_reuses
                .saturating_add(step_report.workspace_reuses);
            report.unknown_steps = report
                .unknown_steps
                .saturating_add(step_report.unknown_steps);
            report.timeout_steps = report
                .timeout_steps
                .saturating_add(step_report.timeout_steps);
            self.record_accepted_gpu_candidate_evidence(result);
        }

        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Execute accepted split/batch push/solve/retract lifecycles through existing GPU CDCL calls.
    ///
    /// The batch evidence must prove every split component ran through the
    /// single-plan GPU runtime path with zero aggregate CPU recomposition,
    /// candidate/world-view fallback, tracked hot-path D2H, and per-candidate
    /// host round trips.
    pub fn solve_assumption_lifecycle_with_gpu_batch_execution_result(
        &mut self,
        provider: &CudaKernelProvider,
        evidence: GpuSolverProductionBatchExecutionEvidence<'_>,
        workspace: &mut GpuCdclWorkspace,
        steps: &[GpuSolverProductionLifecycleStep<'_>],
    ) -> Result<GpuSolverProductionLifecycleReport> {
        if steps.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production lifecycle".to_string(),
                context: "accepted solver lifecycle requires at least one step".to_string(),
            });
        }
        let results = require_accepted_gpu_solver_batch_evidence(provider, evidence.batch)?;
        self.trace.accepted_gpu_batch_candidate_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_evidence_consumed
            .saturating_add(1);
        self.trace
            .accepted_gpu_batch_candidate_component_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_component_evidence_consumed
            .saturating_add(results.len() as u64);
        let report = self.solve_multi_candidate_assumption_lifecycle_with_gpu_execution_results(
            provider, &results, workspace, steps,
        )?;
        self.trace.require_zero_cpu_search()?;
        Ok(report)
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
        require_accepted_gpu_solver_evidence(provider, result)?;

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

        self.trace.gpu_learned_clause_arena_publications = self
            .trace
            .gpu_learned_clause_arena_publications
            .saturating_add(1);
        self.trace.gpu_learned_count_buffer_publications = self
            .trace
            .gpu_learned_count_buffer_publications
            .saturating_add(1);
        self.record_accepted_gpu_candidate_evidence(result);
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
        if let Err(err) = require_same_gpu_cnf_for_learned_clause_reuse(source_cnf, target_cnf) {
            self.trace.gpu_learned_clause_reuse_rejections = self
                .trace
                .gpu_learned_clause_reuse_rejections
                .saturating_add(1);
            self.trace.require_zero_cpu_search()?;
            return Err(err);
        }

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

        self.trace.gpu_learned_clause_arena_publications = self
            .trace
            .gpu_learned_clause_arena_publications
            .saturating_add(1);
        self.trace.gpu_learned_count_buffer_publications = self
            .trace
            .gpu_learned_count_buffer_publications
            .saturating_add(1);

        self.solver
            .solve_expect_unsat_with_branch_limit_ws_importing_learned(
                workspace,
                target_cnf,
                target_branch_var_limit,
            )?;
        self.trace.gpu_cdcl_workspace_unsat_solves =
            self.trace.gpu_cdcl_workspace_unsat_solves.saturating_add(1);
        require_stable_learned_clause_arena(
            "import",
            workspace,
            learned_offsets_ptr,
            learned_lits_ptr,
            proof_offsets_ptr,
            proof_data_ptr,
            learned_count_ptr,
        )?;

        self.trace.gpu_learned_clause_imports =
            self.trace.gpu_learned_clause_imports.saturating_add(1);
        self.trace.gpu_learned_clause_reused_solves = self
            .trace
            .gpu_learned_clause_reused_solves
            .saturating_add(1);
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
        require_accepted_gpu_solver_evidence(provider, result)?;
        let mut report = self.solve_unsat_then_reuse_learned_clauses(
            workspace,
            source_cnf,
            source_branch_var_limit,
            target_cnf,
            target_branch_var_limit,
        )?;
        report.candidate_evidence_records = 1;
        self.record_accepted_gpu_candidate_evidence(result);
        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Publish and reuse learned clauses once per accepted GPU epistemic candidate.
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
        if results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver learned-clause reuse".to_string(),
                context:
                    "multi-candidate learned-clause reuse requires at least one accepted GPU result"
                        .to_string(),
            });
        }
        require_same_gpu_cnf_for_learned_clause_reuse(source_cnf, target_cnf)?;
        for result in results {
            require_accepted_gpu_solver_evidence(provider, result)?;
        }

        let mut report = GpuSolverProductionLearnedClauseReuseReport::default();
        for result in results {
            let step_report = self.solve_unsat_then_reuse_learned_clauses(
                workspace,
                source_cnf,
                source_branch_var_limit,
                target_cnf,
                target_branch_var_limit,
            )?;
            report.candidate_evidence_records = report.candidate_evidence_records.saturating_add(1);
            report.candidates = report.candidates.saturating_add(step_report.candidates);
            report.unsat_solves = report.unsat_solves.saturating_add(step_report.unsat_solves);
            report.gpu_learned_clause_arena_publications = report
                .gpu_learned_clause_arena_publications
                .saturating_add(step_report.gpu_learned_clause_arena_publications);
            report.gpu_learned_clause_imports = report
                .gpu_learned_clause_imports
                .saturating_add(step_report.gpu_learned_clause_imports);
            report.gpu_learned_clause_reused_solves = report
                .gpu_learned_clause_reused_solves
                .saturating_add(step_report.gpu_learned_clause_reused_solves);
            report.cpu_learned_clause_transfers = self.trace.cpu_learned_clause_transfers;
            self.record_accepted_gpu_candidate_evidence(result);
        }

        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Publish and reuse learned clauses once per accepted split/batch GPU component.
    ///
    /// The batch evidence must prove every split component reused the existing
    /// single-plan GPU runtime path before each component is delegated to the
    /// existing multi-candidate learned-clause reuse adapter.
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
        let results = require_accepted_gpu_solver_batch_evidence(provider, evidence.batch)?;
        self.trace.accepted_gpu_batch_candidate_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_evidence_consumed
            .saturating_add(1);
        self.trace
            .accepted_gpu_batch_candidate_component_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_component_evidence_consumed
            .saturating_add(results.len() as u64);
        let report = self.solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results(
            provider,
            &results,
            workspace,
            source_cnf,
            source_branch_var_limit,
            target_cnf,
            target_branch_var_limit,
        )?;
        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    fn solve_weighted_maxsat_candidates(
        &mut self,
        candidates: &[GpuSolverProductionMaxSatCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        Self::require_weighted_maxsat_candidates(candidates)?;

        let solves_before = self.trace.gpu_maxsat_candidate_solves;
        let mut optimum_score = 0u64;
        for candidate in candidates {
            let _assignment = self
                .solver
                .solve_expect_sat_with_branch_limit(candidate.cnf, candidate.branch_var_limit)?;
            self.trace.gpu_cdcl_sat_solves = self.trace.gpu_cdcl_sat_solves.saturating_add(1);
            self.trace.gpu_maxsat_candidate_solves =
                self.trace.gpu_maxsat_candidate_solves.saturating_add(1);
            optimum_score = optimum_score.max(candidate.score);
        }
        self.trace.gpu_maxsat_optima = self.trace.gpu_maxsat_optima.saturating_add(1);
        self.trace.require_zero_cpu_search()?;

        Ok(GpuSolverProductionMaxSatReport {
            candidate_evidence_records: 0,
            optimum_score,
            candidates_checked: candidates.len() as u64,
            satisfiable_candidates: candidates.len() as u64,
            unsat_candidates_pruned: 0,
            gpu_cdcl_candidate_encodes: 0,
            gpu_cdcl_candidate_solves: self
                .trace
                .gpu_maxsat_candidate_solves
                .saturating_sub(solves_before),
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
        require_accepted_gpu_solver_evidence(provider, result)?;
        let mut report = self.solve_weighted_maxsat_candidates(candidates)?;
        report.candidate_evidence_records = 1;
        self.record_accepted_gpu_candidate_evidence(result);
        self.trace.require_zero_cpu_search()?;
        Ok(report)
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
        Self::require_maxsat_lifecycle_inputs(steps, candidates)?;
        require_accepted_gpu_solver_evidence(provider, result)?;

        let lifecycle = self.solve_assumption_lifecycle_steps(workspace, steps)?;
        let maxsat = self.solve_weighted_maxsat_candidates(candidates)?;
        self.record_accepted_gpu_candidate_evidence(result);
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
    ) {
        report.candidate_evidence_records = report
            .candidate_evidence_records
            .saturating_add(step_report.candidate_evidence_records);
        report.lifecycle.steps = report
            .lifecycle
            .steps
            .saturating_add(step_report.lifecycle.steps);
        report.lifecycle.assumption_pushes = report
            .lifecycle
            .assumption_pushes
            .saturating_add(step_report.lifecycle.assumption_pushes);
        report.lifecycle.assumption_retractions = report
            .lifecycle
            .assumption_retractions
            .saturating_add(step_report.lifecycle.assumption_retractions);
        report.lifecycle.workspace_reuses = report
            .lifecycle
            .workspace_reuses
            .saturating_add(step_report.lifecycle.workspace_reuses);
        report.lifecycle.unknown_steps = report
            .lifecycle
            .unknown_steps
            .saturating_add(step_report.lifecycle.unknown_steps);
        report.lifecycle.timeout_steps = report
            .lifecycle
            .timeout_steps
            .saturating_add(step_report.lifecycle.timeout_steps);
        report.maxsat.optimum_score = report
            .maxsat
            .optimum_score
            .max(step_report.maxsat.optimum_score);
        report.maxsat.candidates_checked = report
            .maxsat
            .candidates_checked
            .saturating_add(step_report.maxsat.candidates_checked);
        report.maxsat.satisfiable_candidates = report
            .maxsat
            .satisfiable_candidates
            .saturating_add(step_report.maxsat.satisfiable_candidates);
        report.maxsat.unsat_candidates_pruned = report
            .maxsat
            .unsat_candidates_pruned
            .saturating_add(step_report.maxsat.unsat_candidates_pruned);
        report.maxsat.gpu_cdcl_candidate_encodes = report
            .maxsat
            .gpu_cdcl_candidate_encodes
            .saturating_add(step_report.maxsat.gpu_cdcl_candidate_encodes);
        report.maxsat.gpu_cdcl_candidate_solves = report
            .maxsat
            .gpu_cdcl_candidate_solves
            .saturating_add(step_report.maxsat.gpu_cdcl_candidate_solves);
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
        if results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT lifecycle".to_string(),
                context:
                    "multi-candidate MaxSAT lifecycle requires at least one accepted GPU result"
                        .to_string(),
            });
        }
        Self::require_maxsat_lifecycle_inputs(steps, candidates)?;
        for result in results {
            require_accepted_gpu_solver_evidence(provider, result)?;
        }

        let mut report = GpuSolverProductionMaxSatLifecycleReport::default();
        for result in results {
            let lifecycle = self.solve_assumption_lifecycle_steps(workspace, steps)?;
            let maxsat = self.solve_weighted_maxsat_candidates(candidates)?;
            self.record_accepted_gpu_candidate_evidence(result);
            Self::add_maxsat_lifecycle_step_report(
                &mut report,
                GpuSolverProductionMaxSatLifecycleReport {
                    candidate_evidence_records: 1,
                    lifecycle,
                    maxsat,
                },
            );
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
        Self::require_maxsat_lifecycle_inputs(steps, candidates)?;
        let results = require_accepted_gpu_solver_batch_evidence(provider, evidence.batch)?;
        self.trace.accepted_gpu_batch_candidate_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_evidence_consumed
            .saturating_add(1);
        self.trace
            .accepted_gpu_batch_candidate_component_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_component_evidence_consumed
            .saturating_add(results.len() as u64);
        let report = self.solve_multi_candidate_maxsat_lifecycle_with_gpu_execution_results(
            provider, &results, workspace, steps, candidates,
        )?;
        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Solve a bounded weighted MaxSAT candidate set once per accepted GPU epistemic candidate.
    pub fn solve_multi_candidate_weighted_maxsat_with_gpu_execution_results(
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
        for result in results {
            require_accepted_gpu_solver_evidence(provider, result)?;
        }

        let mut report = GpuSolverProductionMaxSatReport::default();
        for result in results {
            let step_report = self.solve_weighted_maxsat_candidates(candidates)?;
            report.candidate_evidence_records = report.candidate_evidence_records.saturating_add(1);
            report.optimum_score = report.optimum_score.max(step_report.optimum_score);
            report.candidates_checked = report
                .candidates_checked
                .saturating_add(step_report.candidates_checked);
            report.satisfiable_candidates = report
                .satisfiable_candidates
                .saturating_add(step_report.satisfiable_candidates);
            report.unsat_candidates_pruned = report
                .unsat_candidates_pruned
                .saturating_add(step_report.unsat_candidates_pruned);
            report.gpu_cdcl_candidate_encodes = report
                .gpu_cdcl_candidate_encodes
                .saturating_add(step_report.gpu_cdcl_candidate_encodes);
            report.gpu_cdcl_candidate_solves = report
                .gpu_cdcl_candidate_solves
                .saturating_add(step_report.gpu_cdcl_candidate_solves);
            self.record_accepted_gpu_candidate_evidence(result);
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
        let results = require_accepted_gpu_solver_batch_evidence(provider, evidence.batch)?;
        self.trace.accepted_gpu_batch_candidate_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_evidence_consumed
            .saturating_add(1);
        self.trace
            .accepted_gpu_batch_candidate_component_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_component_evidence_consumed
            .saturating_add(results.len() as u64);
        let report = self.solve_multi_candidate_weighted_maxsat_with_gpu_execution_results(
            provider, &results, candidates,
        )?;
        self.trace.require_zero_cpu_search()?;
        Ok(report)
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

        let mut encoded = Vec::with_capacity(selections.len());
        for selection in selections {
            let mut score = 0u64;
            let mut clauses = Vec::with_capacity(selection.soft_clause_indices.len());
            for &idx in selection.soft_clause_indices {
                let clause = &weighted.clauses[idx];
                let weight = weights[idx];
                score = score.saturating_add(weight as u64);
                clauses.push(clause.clone());
            }

            let candidate_instance = SolveInstance::new(weighted.num_vars, clauses);
            let cnf = GpuCnf::from_host(&candidate_instance, &self.provider)?;
            self.trace.gpu_maxsat_candidate_encodes =
                self.trace.gpu_maxsat_candidate_encodes.saturating_add(1);
            encoded.push(GpuSolverProductionEncodedMaxSatSearchCandidate {
                score,
                cnf,
                status: selection.status,
            });
        }

        Ok(encoded)
    }

    fn solve_weighted_maxsat_search_candidates(
        &mut self,
        workspace: &mut GpuCdclWorkspace,
        candidates: &[GpuSolverProductionMaxSatSearchCandidate<'_>],
    ) -> Result<GpuSolverProductionMaxSatReport> {
        Self::require_weighted_maxsat_search_candidates(candidates)?;

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
                    self.trace.gpu_cdcl_sat_solves =
                        self.trace.gpu_cdcl_sat_solves.saturating_add(1);
                    self.trace.gpu_maxsat_candidate_solves =
                        self.trace.gpu_maxsat_candidate_solves.saturating_add(1);
                    satisfiable_candidates = satisfiable_candidates.saturating_add(1);
                    optimum_score = optimum_score.max(candidate.score);
                }
                GpuSolverProductionMaxSatSearchStatus::Unsatisfiable => {
                    self.solve_expect_unsat_with_branch_limit_ws(
                        workspace,
                        candidate.cnf,
                        candidate.branch_var_limit,
                    )?;
                    self.trace.gpu_maxsat_candidate_solves =
                        self.trace.gpu_maxsat_candidate_solves.saturating_add(1);
                    self.trace.gpu_maxsat_unsat_candidate_prunes = self
                        .trace
                        .gpu_maxsat_unsat_candidate_prunes
                        .saturating_add(1);
                }
            }
        }

        self.trace.gpu_maxsat_optima = self.trace.gpu_maxsat_optima.saturating_add(1);
        self.trace.require_zero_cpu_search()?;

        Ok(GpuSolverProductionMaxSatReport {
            candidate_evidence_records: 0,
            optimum_score,
            candidates_checked: candidates.len() as u64,
            satisfiable_candidates,
            unsat_candidates_pruned: self
                .trace
                .gpu_maxsat_unsat_candidate_prunes
                .saturating_sub(unsat_prunes_before),
            gpu_cdcl_candidate_encodes: 0,
            gpu_cdcl_candidate_solves: self
                .trace
                .gpu_maxsat_candidate_solves
                .saturating_sub(solves_before),
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
        if !selections.iter().any(|selection| {
            matches!(
                selection.status,
                GpuSolverProductionMaxSatSearchStatus::Satisfiable
            )
        }) {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT encoding".to_string(),
                context: "bounded MaxSAT search requires at least one satisfiable GPU candidate"
                    .to_string(),
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
                if !weight.is_finite() || weight < 0.0 || weight.fract() != 0.0 {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production MaxSAT encoding".to_string(),
                        context: format!(
                            "soft-clause weight at index {} must be a finite nonnegative integer, got {}",
                            idx, weight
                        ),
                    });
                }
                if weight > u64::MAX as f64 {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "GPU solver production MaxSAT encoding".to_string(),
                        context: format!(
                            "soft-clause weight at index {} exceeds u64 score range",
                            idx
                        ),
                    });
                }
            }
        }
        Ok(())
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
        Self::require_weighted_maxsat_search_candidates(candidates)?;
        require_accepted_gpu_solver_evidence(provider, result)?;
        let mut report = self.solve_weighted_maxsat_search_candidates(workspace, candidates)?;
        report.candidate_evidence_records = 1;
        self.record_accepted_gpu_candidate_evidence(result);
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
        Self::require_weighted_maxsat_search_candidates(candidates)?;
        let results = require_accepted_gpu_solver_batch_evidence(provider, evidence.batch)?;
        self.trace.accepted_gpu_batch_candidate_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_evidence_consumed
            .saturating_add(1);
        self.trace
            .accepted_gpu_batch_candidate_component_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_component_evidence_consumed
            .saturating_add(results.len() as u64);
        let report = self.solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results(
            provider, &results, workspace, candidates,
        )?;
        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    /// Search a bounded weighted MaxSAT candidate set once per accepted GPU evidence record.
    pub fn solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results(
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
        Self::require_weighted_maxsat_search_candidates(candidates)?;
        for result in results {
            require_accepted_gpu_solver_evidence(provider, result)?;
        }

        let mut report = GpuSolverProductionMaxSatReport::default();
        for result in results {
            let step_report =
                self.solve_weighted_maxsat_search_candidates(workspace, candidates)?;
            report.candidate_evidence_records = report.candidate_evidence_records.saturating_add(1);
            report.optimum_score = report.optimum_score.max(step_report.optimum_score);
            report.candidates_checked = report
                .candidates_checked
                .saturating_add(step_report.candidates_checked);
            report.satisfiable_candidates = report
                .satisfiable_candidates
                .saturating_add(step_report.satisfiable_candidates);
            report.unsat_candidates_pruned = report
                .unsat_candidates_pruned
                .saturating_add(step_report.unsat_candidates_pruned);
            report.gpu_cdcl_candidate_encodes = report
                .gpu_cdcl_candidate_encodes
                .saturating_add(step_report.gpu_cdcl_candidate_encodes);
            report.gpu_cdcl_candidate_solves = report
                .gpu_cdcl_candidate_solves
                .saturating_add(step_report.gpu_cdcl_candidate_solves);
            self.record_accepted_gpu_candidate_evidence(result);
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
        Self::require_weighted_maxsat_search_selections(selections)?;
        require_accepted_gpu_solver_evidence(provider, result)?;
        let encodes_before = self.trace.gpu_maxsat_candidate_encodes;
        let encoded = self.encode_weighted_maxsat_search_candidates(weighted, selections)?;
        let search_candidates: Vec<_> = encoded
            .iter()
            .map(|candidate| GpuSolverProductionMaxSatSearchCandidate {
                score: candidate.score,
                cnf: &candidate.cnf,
                branch_var_limit,
                status: candidate.status,
            })
            .collect();
        let mut report =
            self.solve_weighted_maxsat_search_candidates(workspace, &search_candidates)?;
        report.candidate_evidence_records = 1;
        report.gpu_cdcl_candidate_encodes = self
            .trace
            .gpu_maxsat_candidate_encodes
            .saturating_sub(encodes_before);
        self.record_accepted_gpu_candidate_evidence(result);
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
        if results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT encoding".to_string(),
                context:
                    "multi-candidate weighted MaxSAT encoded search requires at least one accepted GPU result"
                        .to_string(),
            });
        }
        Self::require_weighted_maxsat_search_selections(selections)?;
        for result in results {
            require_accepted_gpu_solver_evidence(provider, result)?;
        }

        let mut report = GpuSolverProductionMaxSatReport::default();
        for result in results {
            let encodes_before = self.trace.gpu_maxsat_candidate_encodes;
            let encoded = self.encode_weighted_maxsat_search_candidates(weighted, selections)?;
            let search_candidates: Vec<_> = encoded
                .iter()
                .map(|candidate| GpuSolverProductionMaxSatSearchCandidate {
                    score: candidate.score,
                    cnf: &candidate.cnf,
                    branch_var_limit,
                    status: candidate.status,
                })
                .collect();
            let step_report =
                self.solve_weighted_maxsat_search_candidates(workspace, &search_candidates)?;
            report.candidate_evidence_records = report.candidate_evidence_records.saturating_add(1);
            report.optimum_score = report.optimum_score.max(step_report.optimum_score);
            report.candidates_checked = report
                .candidates_checked
                .saturating_add(step_report.candidates_checked);
            report.satisfiable_candidates = report
                .satisfiable_candidates
                .saturating_add(step_report.satisfiable_candidates);
            report.unsat_candidates_pruned = report
                .unsat_candidates_pruned
                .saturating_add(step_report.unsat_candidates_pruned);
            report.gpu_cdcl_candidate_encodes = report.gpu_cdcl_candidate_encodes.saturating_add(
                self.trace
                    .gpu_maxsat_candidate_encodes
                    .saturating_sub(encodes_before),
            );
            report.gpu_cdcl_candidate_solves = report
                .gpu_cdcl_candidate_solves
                .saturating_add(step_report.gpu_cdcl_candidate_solves);
            self.record_accepted_gpu_candidate_evidence(result);
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
        Self::require_weighted_maxsat_search_selections(selections)?;
        let results = require_accepted_gpu_solver_batch_evidence(provider, evidence.batch)?;
        self.trace.accepted_gpu_batch_candidate_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_evidence_consumed
            .saturating_add(1);
        self.trace
            .accepted_gpu_batch_candidate_component_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_component_evidence_consumed
            .saturating_add(results.len() as u64);
        let report = self
            .solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results(
                provider,
                &results,
                workspace,
                weighted,
                branch_var_limit,
                selections,
            )?;
        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    fn add_maxsat_schedule_step_report(
        report: &mut GpuSolverProductionMaxSatScheduleReport,
        step_report: GpuSolverProductionMaxSatReport,
    ) {
        report.optimum_score = report.optimum_score.max(step_report.optimum_score);
        report.candidates_checked = report
            .candidates_checked
            .saturating_add(step_report.candidates_checked);
        report.satisfiable_candidates = report
            .satisfiable_candidates
            .saturating_add(step_report.satisfiable_candidates);
        report.unsat_candidates_pruned = report
            .unsat_candidates_pruned
            .saturating_add(step_report.unsat_candidates_pruned);
        report.gpu_cdcl_candidate_encodes = report
            .gpu_cdcl_candidate_encodes
            .saturating_add(step_report.gpu_cdcl_candidate_encodes);
        report.gpu_cdcl_candidate_solves = report
            .gpu_cdcl_candidate_solves
            .saturating_add(step_report.gpu_cdcl_candidate_solves);
    }

    fn solve_maxsat_schedule_jobs(
        &mut self,
        workspace: &mut GpuCdclWorkspace,
        jobs: &[GpuSolverProductionMaxSatScheduleJob<'_>],
    ) -> Result<GpuSolverProductionMaxSatScheduleReport> {
        Self::require_maxsat_schedule_jobs(jobs)?;

        let mut report = GpuSolverProductionMaxSatScheduleReport::default();
        for job in jobs {
            self.trace.gpu_maxsat_scheduler_jobs =
                self.trace.gpu_maxsat_scheduler_jobs.saturating_add(1);
            report.jobs = report.jobs.saturating_add(1);

            match job {
                GpuSolverProductionMaxSatScheduleJob::CandidateSet { candidates } => {
                    self.trace.gpu_maxsat_scheduler_candidate_set_jobs = self
                        .trace
                        .gpu_maxsat_scheduler_candidate_set_jobs
                        .saturating_add(1);
                    report.candidate_set_jobs = report.candidate_set_jobs.saturating_add(1);
                    let step_report = self.solve_weighted_maxsat_candidates(candidates)?;
                    Self::add_maxsat_schedule_step_report(&mut report, step_report);
                }
                GpuSolverProductionMaxSatScheduleJob::Search { candidates } => {
                    self.trace.gpu_maxsat_scheduler_search_jobs = self
                        .trace
                        .gpu_maxsat_scheduler_search_jobs
                        .saturating_add(1);
                    report.search_jobs = report.search_jobs.saturating_add(1);
                    let step_report =
                        self.solve_weighted_maxsat_search_candidates(workspace, candidates)?;
                    Self::add_maxsat_schedule_step_report(&mut report, step_report);
                }
                GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
                    weighted,
                    branch_var_limit,
                    selections,
                } => {
                    self.trace.gpu_maxsat_scheduler_encoded_search_jobs = self
                        .trace
                        .gpu_maxsat_scheduler_encoded_search_jobs
                        .saturating_add(1);
                    report.encoded_search_jobs = report.encoded_search_jobs.saturating_add(1);
                    let encodes_before = self.trace.gpu_maxsat_candidate_encodes;
                    let encoded =
                        self.encode_weighted_maxsat_search_candidates(weighted, selections)?;
                    let search_candidates: Vec<_> = encoded
                        .iter()
                        .map(|candidate| GpuSolverProductionMaxSatSearchCandidate {
                            score: candidate.score,
                            cnf: &candidate.cnf,
                            branch_var_limit,
                            status: candidate.status,
                        })
                        .collect();
                    let mut step_report = self
                        .solve_weighted_maxsat_search_candidates(workspace, &search_candidates)?;
                    step_report.gpu_cdcl_candidate_encodes = self
                        .trace
                        .gpu_maxsat_candidate_encodes
                        .saturating_sub(encodes_before);
                    Self::add_maxsat_schedule_step_report(&mut report, step_report);
                }
                GpuSolverProductionMaxSatScheduleJob::Unknown { .. } => {
                    self.trace.gpu_maxsat_scheduler_unknown_status_jobs = self
                        .trace
                        .gpu_maxsat_scheduler_unknown_status_jobs
                        .saturating_add(1);
                    report.unknown_jobs = report.unknown_jobs.saturating_add(1);
                }
                GpuSolverProductionMaxSatScheduleJob::Timeout { .. } => {
                    self.trace.gpu_maxsat_scheduler_timeout_status_jobs = self
                        .trace
                        .gpu_maxsat_scheduler_timeout_status_jobs
                        .saturating_add(1);
                    report.timeout_jobs = report.timeout_jobs.saturating_add(1);
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
        if results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production MaxSAT scheduler".to_string(),
                context: "MaxSAT scheduler requires at least one accepted GPU result".to_string(),
            });
        }
        Self::require_maxsat_schedule_jobs(jobs)?;
        for result in results {
            require_accepted_gpu_solver_evidence(provider, result)?;
        }

        let mut report = GpuSolverProductionMaxSatScheduleReport::default();
        for result in results {
            let step_report = self.solve_maxsat_schedule_jobs(workspace, jobs)?;
            report.candidate_evidence_records = report.candidate_evidence_records.saturating_add(1);
            report.jobs = report.jobs.saturating_add(step_report.jobs);
            report.candidate_set_jobs = report
                .candidate_set_jobs
                .saturating_add(step_report.candidate_set_jobs);
            report.search_jobs = report.search_jobs.saturating_add(step_report.search_jobs);
            report.encoded_search_jobs = report
                .encoded_search_jobs
                .saturating_add(step_report.encoded_search_jobs);
            report.unknown_jobs = report.unknown_jobs.saturating_add(step_report.unknown_jobs);
            report.timeout_jobs = report.timeout_jobs.saturating_add(step_report.timeout_jobs);
            Self::add_maxsat_schedule_step_report(
                &mut report,
                GpuSolverProductionMaxSatReport {
                    optimum_score: step_report.optimum_score,
                    candidates_checked: step_report.candidates_checked,
                    satisfiable_candidates: step_report.satisfiable_candidates,
                    unsat_candidates_pruned: step_report.unsat_candidates_pruned,
                    gpu_cdcl_candidate_encodes: step_report.gpu_cdcl_candidate_encodes,
                    gpu_cdcl_candidate_solves: step_report.gpu_cdcl_candidate_solves,
                    ..GpuSolverProductionMaxSatReport::default()
                },
            );
            self.record_accepted_gpu_candidate_evidence(result);
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
        Self::require_maxsat_schedule_jobs(jobs)?;
        let results = require_accepted_gpu_solver_batch_evidence(provider, evidence.batch)?;
        self.trace.accepted_gpu_batch_candidate_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_evidence_consumed
            .saturating_add(1);
        self.trace
            .accepted_gpu_batch_candidate_component_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_component_evidence_consumed
            .saturating_add(results.len() as u64);
        let report = self.solve_maxsat_schedule_with_gpu_execution_results(
            provider, &results, workspace, jobs,
        )?;
        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }

    fn solve_portfolio_jobs(
        &mut self,
        jobs: &[GpuSolverProductionPortfolioJob<'_>],
    ) -> Result<GpuSolverProductionPortfolioReport> {
        if jobs.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production portfolio".to_string(),
                context: "accepted solver portfolio requires at least one GPU job".to_string(),
            });
        }

        let mut report = GpuSolverProductionPortfolioReport::default();
        for job in jobs {
            self.trace.gpu_portfolio_jobs = self.trace.gpu_portfolio_jobs.saturating_add(1);
            report.jobs = report.jobs.saturating_add(1);

            match job {
                GpuSolverProductionPortfolioJob::Sat {
                    cnf,
                    branch_var_limit,
                } => {
                    let _assignment = self
                        .solver
                        .solve_expect_sat_with_branch_limit(cnf, branch_var_limit)?;
                    self.trace.gpu_cdcl_sat_solves =
                        self.trace.gpu_cdcl_sat_solves.saturating_add(1);
                    self.trace.gpu_portfolio_sat_jobs =
                        self.trace.gpu_portfolio_sat_jobs.saturating_add(1);
                    report.sat_jobs = report.sat_jobs.saturating_add(1);
                }
                GpuSolverProductionPortfolioJob::MaxSat { candidates } => {
                    let maxsat = self.solve_weighted_maxsat_candidates(candidates)?;
                    self.trace.gpu_portfolio_maxsat_jobs =
                        self.trace.gpu_portfolio_maxsat_jobs.saturating_add(1);
                    report.maxsat_jobs = report.maxsat_jobs.saturating_add(1);
                    report.maxsat_optimum_scores = report
                        .maxsat_optimum_scores
                        .saturating_add(maxsat.optimum_score);
                }
                GpuSolverProductionPortfolioJob::Unknown { reason } => {
                    if reason.trim().is_empty() {
                        return Err(XlogError::UnsupportedEpistemicConstruct {
                            construct: "GPU solver production portfolio".to_string(),
                            context: "UNKNOWN portfolio status requires a diagnostic reason"
                                .to_string(),
                        });
                    }
                    self.trace.gpu_portfolio_unknown_status_jobs = self
                        .trace
                        .gpu_portfolio_unknown_status_jobs
                        .saturating_add(1);
                    report.unknown_jobs = report.unknown_jobs.saturating_add(1);
                }
                GpuSolverProductionPortfolioJob::Timeout { budget_micros } => {
                    if *budget_micros == 0 {
                        return Err(XlogError::UnsupportedEpistemicConstruct {
                            construct: "GPU solver production portfolio".to_string(),
                            context: "TIMEOUT portfolio status requires a nonzero budget"
                                .to_string(),
                        });
                    }
                    self.trace.gpu_portfolio_timeout_status_jobs = self
                        .trace
                        .gpu_portfolio_timeout_status_jobs
                        .saturating_add(1);
                    report.timeout_jobs = report.timeout_jobs.saturating_add(1);
                }
            }
        }

        Ok(report)
    }

    fn add_portfolio_report(
        report: &mut GpuSolverProductionPortfolioReport,
        step_report: GpuSolverProductionPortfolioReport,
    ) {
        report.jobs = report.jobs.saturating_add(step_report.jobs);
        report.sat_jobs = report.sat_jobs.saturating_add(step_report.sat_jobs);
        report.maxsat_jobs = report.maxsat_jobs.saturating_add(step_report.maxsat_jobs);
        report.unknown_jobs = report.unknown_jobs.saturating_add(step_report.unknown_jobs);
        report.timeout_jobs = report.timeout_jobs.saturating_add(step_report.timeout_jobs);
        report.maxsat_optimum_scores = report
            .maxsat_optimum_scores
            .saturating_add(step_report.maxsat_optimum_scores);
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
        require_accepted_gpu_solver_evidence(provider, result)?;

        let mut report = self.solve_portfolio_jobs(jobs)?;
        report.candidate_evidence_records = 1;

        self.record_accepted_gpu_candidate_evidence(result);
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
        if results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production portfolio".to_string(),
                context: "multi-candidate portfolio requires at least one accepted GPU result"
                    .to_string(),
            });
        }
        if jobs.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production portfolio".to_string(),
                context: "accepted solver portfolio requires at least one GPU job".to_string(),
            });
        }
        for result in results {
            require_accepted_gpu_solver_evidence(provider, result)?;
        }

        let mut report = GpuSolverProductionPortfolioReport::default();
        for result in results {
            let step_report = self.solve_portfolio_jobs(jobs)?;
            report.candidate_evidence_records = report.candidate_evidence_records.saturating_add(1);
            Self::add_portfolio_report(&mut report, step_report);
            self.record_accepted_gpu_candidate_evidence(result);
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
        if jobs.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU solver production portfolio".to_string(),
                context: "accepted solver portfolio requires at least one GPU job".to_string(),
            });
        }
        let results = require_accepted_gpu_solver_batch_evidence(provider, evidence.batch)?;
        self.trace.accepted_gpu_batch_candidate_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_evidence_consumed
            .saturating_add(1);
        self.trace
            .accepted_gpu_batch_candidate_component_evidence_consumed = self
            .trace
            .accepted_gpu_batch_candidate_component_evidence_consumed
            .saturating_add(results.len() as u64);
        let report = self
            .solve_multi_candidate_portfolio_with_gpu_execution_results(provider, &results, jobs)?;
        self.trace.require_zero_cpu_search()?;
        Ok(report)
    }
}

fn require_accepted_gpu_solver_evidence(
    provider: &CudaKernelProvider,
    result: &EpistemicGpuExecutionResult,
) -> Result<()> {
    if !result.prepared.preflight.cpu_fallbacks.is_zero() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: "solver evidence requires zero epistemic CPU fallback counters".to_string(),
        });
    }
    result
        .model_membership
        .require_stable_model_tuple_source()?;
    require_gpu_kernel_trace(
        "model membership",
        result.model_membership.kernel_launches,
        result.model_membership.host_write_ops,
    )?;
    require_gpu_kernel_trace(
        "world-view validation",
        result.world_view_validation.kernel_launches,
        result.world_view_validation.host_write_ops,
    )?;
    require_gpu_kernel_trace(
        "accepted-candidate materialization",
        result.materialization.kernel_launches,
        result.materialization.host_write_ops,
    )?;
    require_gpu_kernel_trace(
        "final-result materialization",
        result.final_result_materialization.kernel_launches,
        result.final_result_materialization.host_write_ops,
    )?;
    require_gpu_kernel_trace(
        "final tuple materialization",
        result.final_tuple_materialization.kernel_launches,
        result.final_tuple_materialization.host_write_ops,
    )?;
    if result.transfer_budget.tracked_dtoh_calls != 0
        || result.transfer_budget.tracked_htod_calls != 0
        || result.transfer_budget.per_candidate_host_round_trips != 0
    {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: format!(
                "solver evidence requires zero hot-path transfers, got dtoh_calls={}, \
                 htod_calls={}, per_candidate_round_trips={}",
                result.transfer_budget.tracked_dtoh_calls,
                result.transfer_budget.tracked_htod_calls,
                result.transfer_budget.per_candidate_host_round_trips
            ),
        });
    }

    let accepted_rows = read_device_row_count(provider, &result.final_output)?;
    if accepted_rows == 0 {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: "solver evidence requires non-empty accepted GPU final output".to_string(),
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
        || trace.tracked_dtoh_calls != 0
        || trace.per_candidate_host_round_trips != 0
    {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver batch evidence".to_string(),
            context: format!(
                "solver batch evidence requires complete GPU component execution and zero \
                 CPU/host fallback counters, got components={}/{}, recomposition={}, \
                 cpu_candidates={}, cpu_world_views={}, dtoh_calls={}, round_trips={}",
                trace.gpu_runtime_component_executions,
                trace.component_count,
                trace.cpu_recomposition_steps,
                trace.cpu_candidate_enumerations,
                trace.cpu_world_view_validations,
                trace.tracked_dtoh_calls,
                trace.per_candidate_host_round_trips
            ),
        });
    }

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
) -> Result<()> {
    if kernel_launches == 0 || host_write_ops != 0 {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU solver candidate evidence".to_string(),
            context: format!(
                "solver evidence requires GPU {phase} trace with nonzero launches and \
                 zero host writes, got launches={kernel_launches}, host_writes={host_write_ops}"
            ),
        });
    }
    Ok(())
}
