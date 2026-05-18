use std::fs;
use std::path::PathBuf;

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
    assert!(production.contains("solve_expect_unsat_with_branch_limit_ws"));
    assert!(production.contains("accepted_gpu_candidate_evidence_consumed"));
    assert!(production.contains("read_device_row_count"));
    assert!(production.contains("require_stable_model_tuple_source"));
    assert!(production.contains("cpu_assignment_enumerations: 0"));
    assert!(production.contains("cpu_maxsat_enumerations: 0"));
    assert!(!production.contains("assignment_from_mask"));
    assert!(!production.contains("SolverService::new"));
    assert!(!production.contains("solve_assignments"));
}

#[test]
fn production_solver_capabilities_block_missing_maxsat_and_portfolio_paths() {
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
    assert!(production.contains("GpuSolverProductionCapabilityStatus::Blocked"));
    assert!(production.contains("GPU-native MaxSAT production path is not implemented"));
    assert!(production.contains("GPU portfolio SAT/MaxSAT production path is not implemented"));
    assert!(production.contains("cpu_oracle_solver_allowed: false"));
}
