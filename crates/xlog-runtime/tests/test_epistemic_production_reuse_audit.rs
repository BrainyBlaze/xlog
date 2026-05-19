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
    assert!(audit.contains("G38-B closure proposal and integration audit"));
    assert!(audit.contains("G39 completion audit and K7/K8 evidence"));
    assert!(audit.contains("accepted K7 and K8 fixtures"));
    assert!(audit.contains("wcoj_clique8_dispatch_count"));
    assert!(audit.contains("kclique_metadata_build_nanos"));
    assert!(audit.contains("observed metadata-build nanoseconds"));
    assert!(audit.contains("kclique_stream_group_count"));
    assert!(audit.contains("certified_stream_groups"));
    assert!(audit.contains("possible_operator_count"));
    assert!(audit.contains("not_possible_operator_count"));
    assert!(audit.contains("accepted unary, possible, not possible, binary"));
    assert!(audit.contains("row_filter_count"));
    assert!(audit.contains("negated_row_filter_count"));
    assert!(audit.contains("EpistemicGpuSemanticTrace"));
    assert!(audit.contains("kclique_wcoj_max_arity"));

    let runtime = read_workspace_file("crates/xlog-runtime/src/executor/epistemic_workspace.rs");
    let logic = read_workspace_file("crates/xlog-logic/src/epistemic.rs");
    let integration =
        read_workspace_file("crates/xlog-integration/tests/test_epistemic_gpu_wcoj_execution.rs");
    let solver = read_workspace_file("crates/xlog-solve/src/production.rs");
    let prob = read_workspace_file("crates/xlog-prob/src/epistemic_production.rs");
    let prob_epistemic = read_workspace_file("crates/xlog-prob/src/epistemic.rs");

    assert!(logic.contains("compile_epistemic_gpu_execution"));
    assert!(logic.contains("compile_epistemic_gpu_split_execution"));
    assert!(logic.contains("compile_program_with_stats_snapshot"));
    assert!(logic.contains("reject_faeel_self_supported_possible"));
    assert!(logic.contains("FAEEL foundedness guard"));
    assert!(runtime.contains("self.execute_plan(&executable.reduced_runtime_plan)"));
    assert!(runtime.contains("execute_epistemic_gpu_execution_batch"));
    assert!(runtime.contains("self.execute_epistemic_gpu_execution(executable, capacities)"));
    assert!(runtime.contains("summarize_runtime_routes"));
    assert!(runtime.contains("MultiwayPlan::WcojWithPlan"));
    assert!(runtime.contains("kclique_wcoj_max_arity"));
    assert!(runtime.contains("kclique_wcoj_edge_permutation_count"));
    assert!(runtime.contains("kclique_stream_group_count"));
    assert!(runtime.contains("kclique_stream_groups"));
    assert!(runtime.contains("helper_split_spec_count"));
    assert!(runtime.contains("certified_edge_permutation_slots"));
    assert!(runtime.contains("certified_stream_groups"));
    assert!(runtime.contains("certified_sorted_layout_requirements"));
    assert!(runtime.contains("certified_helper_split_specs"));
    assert!(runtime.contains("kclique_metadata_build_nanos"));
    assert!(runtime.contains("observed_metadata_build_nanos"));
    assert!(runtime.contains("know_operator_count"));
    assert!(runtime.contains("possible_operator_count"));
    assert!(runtime.contains("not_know_operator_count"));
    assert!(runtime.contains("not_possible_operator_count"));
    assert!(runtime.contains("row_filter_count"));
    assert!(runtime.contains("negated_row_filter_count"));
    assert!(runtime.contains("planned_hash_route_count"));
    assert!(runtime.contains("EpistemicGpuFinalResultTransferTrace"));
    assert!(runtime.contains("from_final_output(&self.provider, &final_output)"));
    assert!(runtime.contains("EpistemicGpuSemanticTrace"));
    assert!(runtime.contains("from_device_rejection_reasons"));
    assert!(runtime.contains("dtoh_small_metadata_untracked"));
    assert!(runtime.contains("cpu_candidate_enumerations: 0"));
    assert!(runtime.contains("cpu_world_view_validations: 0"));
    assert!(runtime.contains("source_relation.column"));
    assert!(runtime.contains("output.column(bound_col_index)"));
    assert!(integration.contains(
        "accepted_not_possible_nonzero_arity_membership_records_operator_and_polarity_metrics"
    ));
    assert!(integration.contains("not_possible_operator_count"));
    assert!(integration.contains("negated_row_filter_count"));
    assert!(solver.contains("GpuCdclSolver::new"));
    assert!(solver.contains("solve_expect_sat(cnf)"));
    assert!(solver.contains("solve_expect_sat_with_gpu_execution_result"));
    assert!(solver.contains("solve_expect_unsat_with_gpu_execution_result"));
    assert!(solver.contains("solve_expect_unsat_with_branch_limit_ws_with_gpu_execution_result"));
    assert!(solver.contains("solve_assumption_lifecycle_with_gpu_execution_result"));
    assert!(
        solver.contains("solve_multi_candidate_assumption_lifecycle_with_gpu_execution_results")
    );
    assert!(
        solver.contains("solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result")
    );
    assert!(solver.contains("solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result"));
    assert!(
        solver.contains("solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results")
    );
    assert!(solver.contains("solve_weighted_maxsat_candidates_with_gpu_execution_result"));
    assert!(solver.contains("solve_multi_candidate_weighted_maxsat_with_gpu_execution_results"));
    assert!(solver.contains("solve_weighted_maxsat_search_with_gpu_execution_result"));
    assert!(
        solver.contains("solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results")
    );
    assert!(solver.contains("solve_weighted_maxsat_encoded_search_with_gpu_execution_result"));
    assert!(solver.contains(
        "solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results"
    ));
    assert!(solver.contains("encode_weighted_maxsat_search_candidates"));
    assert!(solver.contains("solve_maxsat_schedule_with_gpu_execution_results"));
    assert!(solver.contains("solve_portfolio_with_gpu_execution_result"));
    assert!(solver.contains("GpuSolverProductionLifecycleStep"));
    assert!(solver.contains("GpuSolverProductionExpectation"));
    assert!(solver.contains("GpuSolverProductionExpectation::Unknown"));
    assert!(solver.contains("GpuSolverProductionExpectation::Timeout"));
    assert!(solver.contains("GpuSolverProductionLearnedClauseArenaReport"));
    assert!(solver.contains("GpuSolverProductionLearnedClauseReuseReport"));
    assert!(solver.contains("GpuSolverProductionMaxSatCandidate"));
    assert!(solver.contains("GpuSolverProductionMaxSatSearchCandidate"));
    assert!(solver.contains("GpuSolverProductionMaxSatSearchStatus"));
    assert!(solver.contains("GpuSolverProductionMaxSatScheduleJob"));
    assert!(solver.contains("GpuSolverProductionMaxSatScheduleReport"));
    assert!(solver.contains("GpuSolverProductionWeightedMaxSatSelection"));
    assert!(solver.contains("GpuSolverProductionPortfolioJob"));
    assert!(solver.contains("GpuSolverProductionPortfolioJob::Unknown"));
    assert!(solver.contains("GpuSolverProductionPortfolioJob::Timeout"));
    assert!(solver.contains("candidate_evidence_records"));
    assert!(solver.contains("gpu_lifecycle_unknown_status_steps"));
    assert!(solver.contains("gpu_lifecycle_timeout_status_steps"));
    assert!(solver.contains("gpu_assumption_pushes"));
    assert!(solver.contains("gpu_assumption_retractions"));
    assert!(solver.contains("gpu_lifecycle_workspace_reuses"));
    assert!(solver.contains("gpu_learned_clause_arena_publications"));
    assert!(solver.contains("gpu_learned_count_buffer_publications"));
    assert!(solver.contains("gpu_learned_clause_imports"));
    assert!(solver.contains("gpu_learned_clause_reused_solves"));
    assert!(solver.contains("gpu_learned_clause_reuse_rejections"));
    assert!(solver.contains("gpu_maxsat_candidate_encodes"));
    assert!(solver.contains("gpu_maxsat_candidate_solves"));
    assert!(solver.contains("gpu_maxsat_scheduler_jobs"));
    assert!(solver.contains("gpu_maxsat_scheduler_candidate_set_jobs"));
    assert!(solver.contains("gpu_maxsat_scheduler_search_jobs"));
    assert!(solver.contains("gpu_maxsat_scheduler_encoded_search_jobs"));
    assert!(solver.contains("gpu_maxsat_scheduler_unknown_status_jobs"));
    assert!(solver.contains("gpu_maxsat_scheduler_timeout_status_jobs"));
    assert!(solver.contains("gpu_maxsat_unsat_candidate_prunes"));
    assert!(solver.contains("gpu_maxsat_optima"));
    assert!(solver.contains("gpu_portfolio_jobs"));
    assert!(solver.contains("gpu_portfolio_sat_jobs"));
    assert!(solver.contains("gpu_portfolio_maxsat_jobs"));
    assert!(solver.contains("gpu_portfolio_unknown_status_jobs"));
    assert!(solver.contains("gpu_portfolio_timeout_status_jobs"));
    assert!(solver.contains("accepted_gpu_candidate_evidence_consumed"));
    assert!(solver.contains("require_production_metric_eligibility"));
    assert!(solver.contains("production solver metrics require accepted GPU candidate evidence"));
    assert!(solver.contains("read_device_row_count"));
    assert!(solver.contains("require_stable_model_tuple_source"));
    assert!(solver.contains("GpuCnf::from_host(&candidate_instance, &self.provider)"));
    assert!(solver.contains("cpu_assignment_enumerations: 0"));
    assert!(solver.contains("cpu_learned_clause_transfers: 0"));
    assert!(prob.contains("ExactDdnnfProgram::compile_source_with_gpu"));
    assert!(prob.contains("ExactDdnnfProgram::compile_from_program"));
    assert!(prob.contains("compile_source_with_gpu_execution_result"));
    assert!(prob.contains("compile_program_with_gpu_execution_result"));
    assert!(prob.contains("EpistemicProbGpuExecutionEvidence"));
    assert!(prob.contains("compile_and_evaluate_source_for_gpu_execution_results"));
    assert!(prob.contains("compile_and_evaluate_conditioned_source_for_gpu_execution_results"));
    assert!(prob.contains("compile_and_evaluate_conditioned_program_for_gpu_execution_results"));
    assert!(prob.contains("compile_and_evaluate_source_with_gpu_execution_result"));
    assert!(prob.contains("compile_and_evaluate_conditioned_source_with_gpu_execution_result"));
    assert!(prob.contains("compile_and_evaluate_conditioned_program_with_gpu_execution_result"));
    assert!(prob
        .contains("compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result"));
    assert!(prob
        .contains("compile_and_evaluate_conditioned_source_with_grads_for_gpu_execution_results"));
    assert!(prob
        .contains("compile_and_evaluate_conditioned_program_with_grads_with_gpu_execution_result"));
    assert!(prob
        .contains("compile_and_evaluate_conditioned_program_with_grads_for_gpu_execution_results"));
    assert!(prob.contains("compile_and_evaluate_program_with_gpu_execution_result"));
    assert!(prob.contains("encode_source_pir_cnf_with_gpu_execution_result"));
    assert!(prob.contains("encode_program_pir_cnf_with_gpu_execution_result"));
    assert!(prob.contains("GpuPirGraph::from_host"));
    assert!(prob.contains("encode_cnf_gpu"));
    assert!(prob.contains("gpu_pir_graph_uploads"));
    assert!(prob.contains("gpu_cnf_encodes"));
    assert!(prob.contains("gpu_knowledge_compilation_end_to_end_runs"));
    assert!(prob.contains("gpu_source_knowledge_compilation_end_to_end_runs"));
    assert!(prob.contains("gpu_program_knowledge_compilation_end_to_end_runs"));
    assert!(prob.contains("accepted_evidence_assumptions_consumed"));
    assert!(prob.contains("gpu_conditioned_evidence_facts"));
    assert!(prob.contains("gpu_conditioned_negative_evidence_facts"));
    assert!(prob.contains("record_conditioned_evidence_counts"));
    assert!(prob.contains("condition_source_with_accepted_evidence"));
    assert!(prob.contains("condition_program_with_accepted_evidence"));
    assert!(prob.contains("EpistemicEvidenceTerm"));
    assert!(prob.contains("evidence_term_to_ast_term"));
    assert!(prob.contains("program.evidence.push"));
    assert!(prob.contains("evaluate_with_gpu_execution_result"));
    assert!(prob.contains("evaluate_gpu_with_grads_with_gpu_execution_result"));
    assert!(prob.contains("gpu_exact_query_evaluations"));
    assert!(prob.contains("gpu_exact_gradient_evaluations"));
    assert!(prob.contains("EpistemicProbProductionCapabilities"));
    assert!(prob.contains("fixture_circuit_allowed: false"));
    assert!(prob.contains("require_production_metric_eligibility"));
    assert!(prob.contains("production probability metrics require accepted world-view evidence"));
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
