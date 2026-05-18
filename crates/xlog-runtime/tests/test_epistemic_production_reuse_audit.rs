use std::fs;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("xlog-runtime crate should live under crates/")
        .to_path_buf()
}

fn read_workspace_file(relative_path: &str) -> String {
    fs::read_to_string(workspace_root().join(relative_path))
        .unwrap_or_else(|err| panic!("failed to read {relative_path}: {err}"))
}

#[test]
fn production_reuse_audit_reports_no_parallel_epistemic_engines() {
    let audit =
        read_workspace_file("docs/evidence/2026-05-18-v090-production-reuse-audit/README.md");
    assert!(audit.contains("M090_CERT.13"));
    assert!(audit.contains("zero new epistemic-only WCOJ"));
    assert!(audit.contains("EpistemicGpuRuntimePreflight"));
    assert!(audit.contains("GpuSolverProductionAdapter"));
    assert!(audit.contains("EpistemicProbProductionAdapter"));
    assert!(audit.contains("G38 completion audit"));
    assert!(audit.contains("G38-B integration audit"));
    assert!(audit.contains("G39 completion audit"));

    let runtime = read_workspace_file("crates/xlog-runtime/src/executor/epistemic_workspace.rs");
    let logic = read_workspace_file("crates/xlog-logic/src/epistemic.rs");
    let solver = read_workspace_file("crates/xlog-solve/src/production.rs");
    let prob = read_workspace_file("crates/xlog-prob/src/epistemic_production.rs");
    let prob_epistemic = read_workspace_file("crates/xlog-prob/src/epistemic.rs");

    assert!(logic.contains("compile_epistemic_gpu_execution"));
    assert!(logic.contains("compile_program_with_stats_snapshot"));
    assert!(runtime.contains("self.execute_plan(&executable.reduced_runtime_plan)"));
    assert!(runtime.contains("summarize_runtime_routes"));
    assert!(runtime.contains("MultiwayPlan::WcojWithPlan"));
    assert!(runtime.contains("helper_split_spec_count"));
    assert!(runtime.contains("planned_hash_route_count"));
    assert!(runtime.contains("source_relation.column"));
    assert!(runtime.contains("output.column(bound_col_index)"));
    assert!(solver.contains("GpuCdclSolver::new"));
    assert!(solver.contains("solve_expect_sat(cnf)"));
    assert!(solver.contains("solve_expect_sat_with_gpu_execution_result"));
    assert!(solver.contains("solve_expect_unsat_with_gpu_execution_result"));
    assert!(solver.contains("accepted_gpu_candidate_evidence_consumed"));
    assert!(solver.contains("read_device_row_count"));
    assert!(solver.contains("require_stable_model_tuple_source"));
    assert!(solver.contains("cpu_assignment_enumerations: 0"));
    assert!(prob.contains("ExactDdnnfProgram::compile_source_with_gpu"));
    assert!(prob.contains("ExactDdnnfProgram::compile_from_program"));
    assert!(prob.contains("compile_source_with_gpu_execution_result"));
    assert!(prob.contains("evaluate_gpu_with_grads_with_gpu_execution_result"));
    assert!(prob.contains("gpu_exact_gradient_evaluations"));
    assert!(prob.contains("cpu_only_probability_recomputations: 0"));
    assert!(prob_epistemic.contains("from_gpu_execution_result"));
    assert!(prob_epistemic.contains("read_device_row_count"));
    assert!(prob_epistemic.contains("require_stable_model_tuple_source"));

    for (label, source) in [
        ("runtime", runtime.as_str()),
        ("logic", logic.as_str()),
        ("solver", solver.as_str()),
        ("probability", prob.as_str()),
    ] {
        for forbidden in [
            "struct EpistemicWcojPlanner",
            "struct EpistemicRelationStore",
            "struct EpistemicTupleStore",
            "struct EpistemicSolverSearch",
            "struct EpistemicProbabilityEngine",
            "SolverService::new",
            "EpistemicCircuit::compile",
            "conditional_probability_from_logs",
        ] {
            assert!(
                !source.contains(forbidden),
                "{label} accepted path must not introduce or call {forbidden}"
            );
        }
    }
}
