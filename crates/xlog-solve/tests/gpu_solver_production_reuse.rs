use std::fs;
use std::path::PathBuf;

use xlog_solve::{production_capabilities, GpuSolverProductionCapabilityStatus};

#[test]
fn production_solver_adapter_reuses_gpu_cdcl_not_cpu_oracle() {
    let lib = include_str!("../src/lib.rs");
    let mut production_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    production_path.push("src");
    production_path.push("production.rs");
    let production = fs::read_to_string(&production_path).unwrap_or_default();

    assert!(lib.contains("GpuSolverProductionAdapter"));
    assert!(lib.contains("GpuSolverProductionTrace"));
    assert!(production.contains("GpuCdclSolver::new"));
    assert!(production.contains("solve_expect_sat(cnf)"));
    assert!(production.contains("solve_expect_sat_with_gpu_execution_result"));
    assert!(production.contains("solve_expect_unsat(cnf)"));
    assert!(production.contains("solve_expect_unsat_with_gpu_execution_result"));
    assert!(production.contains("solve_expect_unsat_with_branch_limit_ws"));
    assert!(
        production.contains("solve_expect_unsat_with_branch_limit_ws_with_gpu_execution_result")
    );
    assert!(production.contains("solve_assumption_lifecycle_with_gpu_execution_result"));
    assert!(production
        .contains("solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result"));
    assert!(production.contains("solve_weighted_maxsat_candidates_with_gpu_execution_result"));
    assert!(production.contains("solve_portfolio_with_gpu_execution_result"));
    assert!(production.contains("GpuSolverProductionLearnedClauseArenaReport"));
    assert!(production.contains("GpuSolverProductionMaxSatCandidate"));
    assert!(production.contains("GpuSolverProductionPortfolioJob"));
    assert!(production.contains("GpuSolverProductionPortfolioJob::Unknown"));
    assert!(production.contains("GpuSolverProductionPortfolioJob::Timeout"));
    assert!(production.contains("GpuSolverProductionLifecycleStep"));
    assert!(production.contains("GpuSolverProductionExpectation"));
    assert!(production.contains("gpu_assumption_pushes"));
    assert!(production.contains("gpu_assumption_retractions"));
    assert!(production.contains("gpu_lifecycle_workspace_reuses"));
    assert!(production.contains("gpu_learned_clause_arena_publications"));
    assert!(production.contains("gpu_learned_count_buffer_publications"));
    assert!(production.contains("gpu_maxsat_candidate_solves"));
    assert!(production.contains("gpu_maxsat_optima"));
    assert!(production.contains("gpu_portfolio_jobs"));
    assert!(production.contains("gpu_portfolio_sat_jobs"));
    assert!(production.contains("gpu_portfolio_maxsat_jobs"));
    assert!(production.contains("gpu_portfolio_unknown_status_jobs"));
    assert!(production.contains("gpu_portfolio_timeout_status_jobs"));
    assert!(production.contains("accepted_gpu_candidate_evidence_consumed"));
    assert!(production.contains("read_device_row_count"));
    assert!(production.contains("require_stable_model_tuple_source"));
    assert!(production.contains("cpu_assignment_enumerations: 0"));
    assert!(production.contains("cpu_maxsat_enumerations: 0"));
    assert!(production.contains("cpu_learned_clause_transfers: 0"));
    assert!(!production.contains("assignment_from_mask"));
    assert!(!production.contains("SolverService::new"));
    assert!(!production.contains("solve_assignments"));
}

#[test]
fn production_solver_capabilities_report_gpu_backed_maxsat_and_portfolio_paths() {
    let lib = include_str!("../src/lib.rs");
    let mut production_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    production_path.push("src");
    production_path.push("production.rs");
    let production = fs::read_to_string(&production_path).unwrap_or_default();

    assert!(lib.contains("GpuSolverProductionCapabilities"));
    assert!(lib.contains("GpuSolverProductionCapabilityStatus"));
    assert!(production.contains("pub fn production_capabilities()"));
    assert!(production.contains("gpu_cdcl_sat_unsat"));
    assert!(production.contains("gpu_maxsat"));
    assert!(production.contains("gpu_portfolio_sat_maxsat"));
    assert!(production.contains("GpuSolverProductionCapabilityStatus::Available"));
    assert!(production.contains("solve_weighted_maxsat_candidates"));
    assert!(production.contains("solve_portfolio_with_gpu_execution_result"));
    assert!(production.contains("cpu_oracle_solver_allowed: false"));

    let capabilities = production_capabilities();
    assert_eq!(
        capabilities.gpu_cdcl_sat_unsat,
        GpuSolverProductionCapabilityStatus::Available
    );
    assert_eq!(
        capabilities.gpu_maxsat,
        GpuSolverProductionCapabilityStatus::Available
    );
    assert_eq!(
        capabilities.gpu_portfolio_sat_maxsat,
        GpuSolverProductionCapabilityStatus::Available
    );
    assert!(!capabilities.cpu_oracle_solver_allowed);
    assert_eq!(capabilities.gpu_maxsat_blocker, "");
    assert_eq!(capabilities.gpu_portfolio_blocker, "");
}
