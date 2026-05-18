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
    assert!(production.contains("solve_expect_unsat(cnf)"));
    assert!(production.contains("solve_expect_unsat_with_branch_limit_ws"));
    assert!(production.contains("cpu_assignment_enumerations: 0"));
    assert!(production.contains("cpu_maxsat_enumerations: 0"));
    assert!(!production.contains("assignment_from_mask"));
    assert!(!production.contains("SolverService::new"));
    assert!(!production.contains("solve_assignments"));
}
