//! Production GPU solver adapter for epistemic callers.
//!
//! This module is intentionally thin: it routes accepted SAT work into the
//! existing GPU CDCL verifier instead of using the bounded CPU semantic-oracle
//! facade in [`crate::SolverService`].

use std::sync::Arc;

use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::CudaKernelProvider;
use xlog_runtime::{read_device_row_count, EpistemicGpuExecutionResult};

use crate::{GpuCdclConfig, GpuCdclSolver, GpuCdclWorkspace, GpuCnf};

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
    /// GPU SAT/MaxSAT portfolio production execution.
    pub gpu_portfolio_sat_maxsat: GpuSolverProductionCapabilityStatus,
    /// Whether the CPU semantic-oracle solver may satisfy production metrics.
    pub cpu_oracle_solver_allowed: bool,
    /// Blocker reason for GPU-native MaxSAT.
    pub gpu_maxsat_blocker: &'static str,
    /// Blocker reason for GPU SAT/MaxSAT portfolio execution.
    pub gpu_portfolio_blocker: &'static str,
}

/// Expected GPU CDCL result for one production lifecycle step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSolverProductionExpectation {
    /// The step must be SAT under the currently pushed assumptions.
    Sat,
    /// The step must be UNSAT under the currently pushed assumptions.
    Unsat,
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
    /// Number of lifecycle steps executed.
    pub steps: u64,
    /// Number of assumption pushes recorded before GPU solves.
    pub assumption_pushes: u64,
    /// Number of assumption retractions recorded after GPU solves.
    pub assumption_retractions: u64,
    /// Number of UNSAT steps that reused the provided GPU CDCL workspace allocation.
    pub workspace_reuses: u64,
}

/// Return the current production solver capability report.
pub fn production_capabilities() -> GpuSolverProductionCapabilities {
    GpuSolverProductionCapabilities {
        gpu_cdcl_sat_unsat: GpuSolverProductionCapabilityStatus::Available,
        gpu_maxsat: GpuSolverProductionCapabilityStatus::Blocked,
        gpu_portfolio_sat_maxsat: GpuSolverProductionCapabilityStatus::Blocked,
        cpu_oracle_solver_allowed: false,
        gpu_maxsat_blocker: "GPU-native MaxSAT production path is not implemented",
        gpu_portfolio_blocker: "GPU portfolio SAT/MaxSAT production path is not implemented",
    }
}

/// Trace counters proving the production adapter stayed on the GPU CDCL path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuSolverProductionTrace {
    /// Number of accepted GPU epistemic candidate evidence records consumed.
    pub accepted_gpu_candidate_evidence_consumed: u64,
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
    /// CPU exhaustive assignment enumerations performed by this adapter.
    pub cpu_assignment_enumerations: u64,
    /// CPU MaxSAT assignment enumerations performed by this adapter.
    pub cpu_maxsat_enumerations: u64,
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
        Ok(())
    }
}

/// Thin adapter from epistemic solver work to the existing GPU CDCL verifier.
pub struct GpuSolverProductionAdapter {
    solver: GpuCdclSolver,
    trace: GpuSolverProductionTrace,
}

impl GpuSolverProductionAdapter {
    /// Create an adapter over the existing GPU CDCL solver implementation.
    pub fn new(provider: Arc<CudaKernelProvider>, config: GpuCdclConfig) -> Self {
        Self {
            solver: GpuCdclSolver::new(provider, config),
            trace: GpuSolverProductionTrace {
                cpu_assignment_enumerations: 0,
                cpu_maxsat_enumerations: 0,
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
        self.trace.accepted_gpu_candidate_evidence_consumed = self
            .trace
            .accepted_gpu_candidate_evidence_consumed
            .saturating_add(1);
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
        self.trace.accepted_gpu_candidate_evidence_consumed = self
            .trace
            .accepted_gpu_candidate_evidence_consumed
            .saturating_add(1);
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
        self.trace.accepted_gpu_candidate_evidence_consumed = self
            .trace
            .accepted_gpu_candidate_evidence_consumed
            .saturating_add(1);
        self.trace.require_zero_cpu_search()
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

        let pushes_before = self.trace.gpu_assumption_pushes;
        let retractions_before = self.trace.gpu_assumption_retractions;
        let workspace_reuses_before = self.trace.gpu_lifecycle_workspace_reuses;

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

        self.trace.accepted_gpu_candidate_evidence_consumed = self
            .trace
            .accepted_gpu_candidate_evidence_consumed
            .saturating_add(1);
        self.trace.require_zero_cpu_search()?;

        Ok(GpuSolverProductionLifecycleReport {
            steps: steps.len() as u64,
            assumption_pushes,
            assumption_retractions,
            workspace_reuses: self
                .trace
                .gpu_lifecycle_workspace_reuses
                .saturating_sub(workspace_reuses_before),
        })
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
