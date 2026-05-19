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
    assert!(audit.contains("EpistemicGpuBatchExecutionTrace"));
    assert!(audit.contains("GpuSolverProductionAdapter"));
    assert!(audit.contains("GpuSolverProductionBatchExecutionEvidence"));
    assert!(audit.contains("accepted_gpu_batch_candidate_evidence_consumed"));
    assert!(audit.contains("accepted_gpu_batch_candidate_component_evidence_consumed"));
    assert!(audit.contains("solve_weighted_maxsat_encoded_search_with_gpu_batch_execution_result"));
    assert!(audit.contains("solve_maxsat_schedule_with_gpu_batch_execution_result"));
    assert!(audit.contains("EpistemicProbProductionAdapter"));
    assert!(audit.contains("G38 completion audit"));
    assert!(audit.contains("G38-B closure proposal and integration audit"));
    assert!(audit.contains("G39 completion audit and K7/K8 evidence"));
    assert!(audit.contains("accepted K7 and K8 fixtures"));
    assert!(audit.contains("wcoj_clique8_dispatch_count"));
    assert!(audit.contains("kclique_metadata_build_nanos"));
    assert!(audit.contains("observed metadata-build nanoseconds"));
    assert!(audit.contains("layout sort or fast-path"));
    assert!(audit.contains("MissingRequiredWcojLayout"));
    assert!(audit.contains("helper relation rules"));
    assert!(audit.contains("WCOJ helper input scans"));
    assert!(audit.contains("kclique_stream_group_count"));
    assert!(audit.contains("certified_stream_groups"));
    assert!(audit.contains("kclique_skew_scheduled_plan_count"));
    assert!(audit.contains("certified_skew_scheduled_plans"));
    assert!(audit.contains("possible_operator_count"));
    assert!(audit.contains("not_possible_operator_count"));
    assert!(audit.contains("accepted unary, possible, not possible, binary"));
    assert!(audit.contains("row_filter_count"));
    assert!(audit.contains("negated_row_filter_count"));
    assert!(audit.contains("EpistemicGpuSemanticTrace"));
    assert!(audit.contains("kclique_wcoj_max_arity"));

    let runtime = read_workspace_file("crates/xlog-runtime/src/executor/epistemic_workspace.rs");
    let cuda = read_workspace_file("crates/xlog-cuda/kernels/epistemic.cu");
    let logic = read_workspace_file("crates/xlog-logic/src/epistemic.rs");
    let integration =
        read_workspace_file("crates/xlog-integration/tests/test_epistemic_gpu_wcoj_execution.rs");
    let solver = read_workspace_file("crates/xlog-solve/src/production.rs");
    let prob = read_workspace_file("crates/xlog-prob/src/epistemic_production.rs");
    let prob_epistemic = read_workspace_file("crates/xlog-prob/src/epistemic.rs");

    assert!(logic.contains("compile_epistemic_gpu_execution"));
    assert!(logic.contains("compile_epistemic_gpu_split_execution"));
    assert!(logic.contains("compile_program_with_stats_snapshot"));
    assert!(logic.contains("run_generate_propagate_test_with_mode"));
    assert!(logic.contains("rejected_candidate_indices"));
    assert!(logic.contains("reject_faeel_self_supported_possible"));
    assert!(logic.contains("FAEEL foundedness guard"));
    assert!(logic.contains("has_independent_founded_support"));
    assert!(runtime.contains("self.execute_plan(&executable.reduced_runtime_plan)"));
    assert!(runtime.contains("execute_epistemic_gpu_execution_batch"));
    assert!(runtime.contains("execute_epistemic_gpu_execution_batch_with_trace"));
    assert!(runtime.contains("self.execute_epistemic_gpu_execution(executable, capacities)"));
    assert!(runtime.contains("EpistemicGpuBatchExecutionTrace"));
    assert!(runtime.contains("cpu_recomposition_steps: 0"));
    assert!(runtime.contains("summarize_runtime_routes"));
    assert!(runtime.contains("MultiwayPlan::WcojWithPlan"));
    assert!(runtime.contains("kclique_wcoj_max_arity"));
    assert!(runtime.contains("kclique_wcoj_edge_permutation_count"));
    assert!(runtime.contains("kclique_stream_group_count"));
    assert!(runtime.contains("kclique_stream_groups"));
    assert!(runtime.contains("kclique_skew_scheduled_plan_count"));
    assert!(runtime.contains("helper_split_spec_count"));
    assert!(runtime.contains("helper_relation_rule_count"));
    assert!(runtime.contains("helper_relation_scan_count"));
    assert!(runtime.contains("helper_relation_ids"));
    assert!(runtime.contains("count_helper_relation_scans"));
    assert!(runtime.contains("count_helper_relation_leaf_scans"));
    assert!(runtime.contains("epistemic GPU helper-split certification"));
    assert!(runtime.contains("certified_edge_permutation_slots"));
    assert!(runtime.contains("certified_stream_groups"));
    assert!(runtime.contains("certified_skew_scheduled_plans"));
    assert!(runtime.contains("certified_sorted_layout_requirements"));
    assert!(runtime.contains("certified_helper_split_specs"));
    assert!(runtime.contains("certified_helper_relation_rules"));
    assert!(runtime.contains("certified_helper_relation_scans"));
    assert!(runtime.contains("observed_layout_fast_path_hits"));
    assert!(runtime.contains("MissingRequiredWcojLayout"));
    assert!(runtime.contains("required_sorted_layouts"));
    assert!(runtime.contains("wcoj_layout_fast_path_hit_count"));
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
    assert!(runtime.contains("EpistemicGpuRejectionReason"));
    assert!(runtime.contains("typed_rejection_reasons"));
    assert!(runtime.contains("accepted_candidate_indices"));
    assert!(runtime.contains("rejected_candidate_indices"));
    assert!(runtime.contains("dtoh_small_metadata_untracked"));
    assert!(runtime.contains("&workspace.candidate_assumptions"));
    assert!(cuda.contains("complete_membership"));
    assert!(cuda.contains("candidate_assumptions[assumption_base + literal]"));
    assert!(runtime.contains("cpu_candidate_enumerations: 0"));
    assert!(runtime.contains("cpu_world_view_validations: 0"));
    assert!(runtime.contains("source_relation.column"));
    assert!(runtime.contains("output.column(bound_col_index)"));
    assert!(integration.contains(
        "accepted_not_possible_nonzero_arity_membership_records_operator_and_polarity_metrics"
    ));
    assert!(integration
        .contains("world_view_validation_rejects_candidates_missing_one_required_membership"));
    assert!(
        integration.contains("faeel_independently_founded_self_possible_reaches_gpu_runtime_path")
    );
    assert!(integration.contains("g91_self_supported_possible_reaches_gpu_runtime_path"));
    assert!(integration
        .contains("accepted_gpu_execution_semantic_trace_matches_gpt_oracle_rejection_reason"));
    assert!(integration.contains("run_generate_propagate_test"));
    assert!(integration.contains("run_generate_propagate_test_with_mode"));
    assert!(integration.contains("oracle.accepted_candidate_indices"));
    assert!(integration.contains("oracle.rejected_candidate_indices"));
    assert!(integration.contains("accepted_ternary_membership_matches_gpt_oracle_parity"));
    assert!(
        integration.contains("split_gpu_world_view_distinguishes_absent_possible_from_not_known")
    );
    assert!(integration.contains("batch.trace.cpu_recomposition_steps"));
    assert!(integration.contains("batch.trace.per_candidate_host_round_trips"));
    assert!(integration.contains(
        "accepted_gpu_execution_results_gate_batched_negative_conditioned_probabilistic_queries"
    ));
    assert!(integration.contains("accepted_split_batch_gates_solver_lifecycle_path"));
    assert!(
        integration.contains("accepted_gpu_execution_result_gates_solver_maxsat_lifecycle_path")
    );
    assert!(integration.contains(
        "accepted_gpu_execution_result_rejects_empty_maxsat_lifecycle_before_lifecycle_work"
    ));
    assert!(integration.contains("accepted_split_batch_gates_solver_maxsat_lifecycle_path"));
    assert!(integration.contains("accepted_split_batch_gates_solver_portfolio_path"));
    assert!(integration.contains("accepted_split_batch_gates_solver_learned_clause_reuse_path"));
    assert!(integration.contains("accepted_split_batch_gates_solver_maxsat_path"));
    assert!(integration.contains("accepted_split_batch_gates_solver_maxsat_search_pruning"));
    assert!(integration
        .contains("accepted_split_batch_gates_solver_encoded_maxsat_and_scheduler_paths"));
    assert!(integration.contains(
        "accepted_ternary_gpu_execution_result_records_solver_nonzero_arity_evidence_trace"
    ));
    assert!(integration.contains(
        "accepted_split_batch_rejects_invalid_encoded_maxsat_scheduler_before_scheduler_work"
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
    assert!(solver.contains("solve_maxsat_lifecycle_with_gpu_execution_result"));
    assert!(solver.contains("solve_multi_candidate_maxsat_lifecycle_with_gpu_execution_results"));
    assert!(solver.contains("solve_maxsat_lifecycle_with_gpu_batch_execution_result"));
    assert!(solver.contains("require_maxsat_lifecycle_inputs"));
    assert!(solver.contains("GpuSolverProductionBatchExecutionEvidence"));
    assert!(solver.contains("solve_assumption_lifecycle_with_gpu_batch_execution_result"));
    assert!(solver.contains("require_accepted_gpu_solver_batch_evidence"));
    assert!(solver.contains("EpistemicGpuBatchExecutionResult"));
    assert!(
        solver.contains("solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result")
    );
    assert!(solver.contains("solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result"));
    assert!(
        solver.contains("solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results")
    );
    assert!(solver.contains("solve_learned_clause_reuse_with_gpu_batch_execution_result"));
    assert!(solver.contains("solve_weighted_maxsat_candidates_with_gpu_execution_result"));
    assert!(solver.contains("solve_multi_candidate_weighted_maxsat_with_gpu_execution_results"));
    assert!(solver.contains("solve_weighted_maxsat_candidates_with_gpu_batch_execution_result"));
    assert!(solver.contains("solve_weighted_maxsat_search_with_gpu_execution_result"));
    assert!(solver.contains("solve_weighted_maxsat_search_with_gpu_batch_execution_result"));
    assert!(
        solver.contains("solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results")
    );
    assert!(solver.contains("require_weighted_maxsat_search_candidates"));
    assert!(solver.contains("require_weighted_maxsat_candidates"));
    assert!(solver.contains("solve_weighted_maxsat_encoded_search_with_gpu_execution_result"));
    assert!(solver.contains(
        "solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results"
    ));
    assert!(solver.contains("solve_weighted_maxsat_encoded_search_with_gpu_batch_execution_result"));
    assert!(solver.contains("encode_weighted_maxsat_search_candidates"));
    assert!(solver.contains("require_weighted_maxsat_search_selections"));
    assert!(solver.contains("require_weighted_maxsat_encoding_inputs"));
    assert!(solver.contains("require_maxsat_schedule_jobs"));
    assert!(solver.contains("solve_maxsat_schedule_with_gpu_execution_results"));
    assert!(solver.contains("solve_maxsat_schedule_with_gpu_batch_execution_result"));
    assert!(solver.contains("solve_portfolio_with_gpu_execution_result"));
    assert!(solver.contains("solve_portfolio_with_gpu_batch_execution_result"));
    assert!(solver.contains("solve_multi_candidate_portfolio_with_gpu_execution_results"));
    assert!(solver.contains("GpuSolverProductionLifecycleStep"));
    assert!(solver.contains("GpuSolverProductionMaxSatLifecycleReport"));
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
    assert!(solver.contains("accepted_gpu_batch_candidate_evidence_consumed"));
    assert!(solver.contains("accepted_gpu_batch_candidate_component_evidence_consumed"));
    assert!(solver.contains("accepted_g91_gpu_candidate_evidence_consumed"));
    assert!(solver.contains("accepted_faeel_gpu_candidate_evidence_consumed"));
    assert!(solver.contains("accepted_know_gpu_candidate_evidence_consumed"));
    assert!(solver.contains("accepted_possible_gpu_candidate_evidence_consumed"));
    assert!(solver.contains("accepted_not_possible_gpu_candidate_evidence_consumed"));
    assert!(solver.contains("accepted_not_know_gpu_candidate_evidence_consumed"));
    assert!(solver.contains("accepted_nonzero_arity_gpu_candidate_evidence_consumed"));
    assert!(solver.contains("accepted_gpu_candidate_tuple_key_column_reads_consumed"));
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
    assert!(prob.contains("compile_source_for_gpu_execution_results"));
    assert!(prob.contains("compile_source_for_gpu_batch_execution_result"));
    assert!(prob.contains("compile_program_with_gpu_execution_result"));
    assert!(prob.contains("compile_program_for_gpu_execution_results"));
    assert!(prob.contains("compile_program_for_gpu_batch_execution_result"));
    assert!(prob.contains("EpistemicProbGpuExecutionEvidence"));
    assert!(prob.contains("EpistemicProbGpuBatchExecutionEvidence"));
    assert!(prob.contains("compile_and_evaluate_source_for_gpu_execution_results"));
    assert!(prob.contains("compile_and_evaluate_source_for_gpu_batch_execution_result"));
    assert!(prob.contains("compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result"));
    assert!(
        prob.contains("compile_and_evaluate_conditioned_program_for_gpu_batch_execution_result")
    );
    assert!(prob.contains("compile_and_evaluate_program_for_gpu_execution_results"));
    assert!(prob.contains("compile_and_evaluate_program_for_gpu_batch_execution_result"));
    assert!(prob.contains("compile_and_evaluate_conditioned_source_for_gpu_execution_results"));
    assert!(prob.contains("compile_and_evaluate_conditioned_program_for_gpu_execution_results"));
    assert!(prob.contains("compile_and_evaluate_source_with_gpu_execution_result"));
    assert!(prob.contains("compile_and_evaluate_conditioned_source_with_gpu_execution_result"));
    assert!(prob.contains("compile_and_evaluate_conditioned_program_with_gpu_execution_result"));
    assert!(prob
        .contains("compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result"));
    assert!(prob
        .contains("compile_and_evaluate_conditioned_source_with_grads_for_gpu_execution_results"));
    assert!(prob.contains(
        "compile_and_evaluate_conditioned_source_with_grads_for_gpu_batch_execution_result"
    ));
    assert!(prob
        .contains("compile_and_evaluate_conditioned_program_with_grads_with_gpu_execution_result"));
    assert!(prob
        .contains("compile_and_evaluate_conditioned_program_with_grads_for_gpu_execution_results"));
    assert!(prob.contains(
        "compile_and_evaluate_conditioned_program_with_grads_for_gpu_batch_execution_result"
    ));
    assert!(prob.contains("compile_and_evaluate_program_with_gpu_execution_result"));
    assert!(prob.contains("encode_source_pir_cnf_with_gpu_execution_result"));
    assert!(prob.contains("encode_program_pir_cnf_with_gpu_execution_result"));
    assert!(prob.contains("encode_source_pir_cnf_for_gpu_execution_results"));
    assert!(prob.contains("encode_program_pir_cnf_for_gpu_execution_results"));
    assert!(prob.contains("encode_source_pir_cnf_for_gpu_batch_execution_result"));
    assert!(prob.contains("encode_program_pir_cnf_for_gpu_batch_execution_result"));
    assert!(prob.contains("evaluate_for_gpu_execution_results"));
    assert!(prob.contains("evaluate_for_gpu_batch_execution_result"));
    assert!(prob.contains("evaluate_gpu_with_grads_for_gpu_execution_results"));
    assert!(prob.contains("evaluate_gpu_with_grads_for_gpu_batch_execution_result"));
    assert!(prob.contains("GpuPirGraph::from_host"));
    assert!(prob.contains("encode_cnf_gpu"));
    assert!(prob.contains("gpu_pir_graph_uploads"));
    assert!(prob.contains("gpu_cnf_encodes"));
    assert!(prob.contains("gpu_knowledge_compilation_end_to_end_runs"));
    assert!(prob.contains("gpu_source_knowledge_compilation_end_to_end_runs"));
    assert!(prob.contains("gpu_program_knowledge_compilation_end_to_end_runs"));
    assert!(prob.contains("accepted_evidence_assumptions_consumed"));
    assert!(prob.contains("accepted_gpu_batch_evidence_consumed"));
    assert!(prob.contains("accepted_gpu_batch_component_evidence_consumed"));
    assert!(prob.contains("gpu_conditioned_evidence_facts"));
    assert!(prob.contains("gpu_conditioned_nonzero_arity_evidence_facts"));
    assert!(prob.contains("gpu_conditioned_max_evidence_arity"));
    assert!(prob.contains("gpu_conditioned_negative_evidence_facts"));
    assert!(prob.contains("gpu_source_conditioned_nonzero_arity_evidence_facts"));
    assert!(prob.contains("gpu_source_conditioned_max_evidence_arity"));
    assert!(prob.contains("gpu_program_conditioned_nonzero_arity_evidence_facts"));
    assert!(prob.contains("gpu_program_conditioned_max_evidence_arity"));
    assert!(prob.contains("gpu_source_conditioned_know_evidence_facts"));
    assert!(prob.contains("gpu_source_conditioned_possible_evidence_facts"));
    assert!(prob.contains("gpu_source_conditioned_not_known_evidence_facts"));
    assert!(prob.contains("gpu_source_conditioned_not_possible_evidence_facts"));
    assert!(prob.contains("gpu_program_conditioned_know_evidence_facts"));
    assert!(prob.contains("gpu_program_conditioned_possible_evidence_facts"));
    assert!(prob.contains("gpu_program_conditioned_not_known_evidence_facts"));
    assert!(prob.contains("gpu_program_conditioned_not_possible_evidence_facts"));
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
    assert!(
        integration.contains("accepted_split_batch_gates_probabilistic_conditioned_source_path")
    );
    assert!(integration
        .contains("accepted_split_batch_gates_probabilistic_source_and_program_end_to_end_paths"));
    assert!(
        integration.contains("accepted_split_batch_gates_probabilistic_conditioned_program_path")
    );
    assert!(integration
        .contains("accepted_split_batch_gates_probabilistic_conditioned_source_gradients"));
    assert!(integration
        .contains("accepted_split_batch_gates_probabilistic_conditioned_program_gradients"));
    assert!(integration
        .contains("accepted_split_batch_gates_probabilistic_pir_cnf_and_exact_evaluation_paths"));
    assert!(integration
        .contains("accepted_gpu_execution_batches_gate_probabilistic_exact_compile_paths"));
    assert!(
        integration.contains("accepted_ternary_probabilistic_evidence_records_nonzero_arity_trace")
    );

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
