use std::collections::BTreeMap;
use std::sync::Arc;

use xlog_core::{symbol, MemoryBudget, RelId, ScalarType, Schema};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::{
    CompiledRule, EirAtom, EirEpistemicLiteral, EirEpistemicMode, EirEpistemicOp, EirTerm,
    EpistemicExecutablePlan, EpistemicGpuPlan, EpistemicReductionPlan,
    EpistemicWcojReductionStatus, ExecutionPlan, RirMeta, RirNode, Scc, Stratum,
};
use xlog_logic::epistemic::{
    compile_epistemic_gpu_execution, compile_epistemic_gpu_split_execution,
};
use xlog_logic::parse_program;
use xlog_runtime::{EpistemicGpuBatchExecutionResult, EpistemicGpuWorkspaceCapacities, Executor};
use xlog_solve::{
    Clause, GpuCdclConfig, GpuCnf, GpuSolverProductionAdapter,
    GpuSolverProductionBatchExecutionEvidence, GpuSolverProductionExpectation,
    GpuSolverProductionLifecycleStep, GpuSolverProductionMaxSatCandidate,
    GpuSolverProductionMaxSatScheduleJob, GpuSolverProductionMaxSatSearchStatus,
    GpuSolverProductionPortfolioJob, GpuSolverProductionWeightedMaxSatSelection, Literal,
    SolveInstance,
};

fn try_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {e}");
            return None;
        }
    };
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    match CudaKernelProvider::new(device, memory) {
        Ok(p) => Some(Arc::new(p)),
        Err(e) => {
            eprintln!("Skipping test: failed to create CUDA kernel provider: {e}");
            None
        }
    }
}

#[test]
fn public_adapter_consumes_real_runtime_accepted_evidence_before_gpu_sat() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_ground_literal(&provider);

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &provider).expect("SAT GpuCnf upload");
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());

    let assignment = adapter
        .solve_expect_sat_with_gpu_execution_result(&provider, &result, &sat_cnf)
        .expect("accepted runtime evidence should gate a public production SAT solve");
    assert_eq!(assignment.len(), 2);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_candidate_state_transitions, 1);
    assert_eq!(trace.accepted_gpu_world_view_state_transitions, 1);
    assert_eq!(trace.accepted_gpu_candidate_final_output_rows_consumed, 1);
    assert_eq!(trace.accepted_faeel_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        1
    );
    assert_eq!(trace.accepted_solver_assumption_bindings_consumed, 1);
    assert_eq!(trace.accepted_solver_required_capabilities_consumed, 5);
    assert_eq!(trace.accepted_solver_required_statuses_consumed, 4);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 1);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("accepted production path must not use CPU solver search");
    trace
        .require_production_metric_eligibility()
        .expect("accepted runtime evidence plus public GPU SAT solve satisfies metric gate");
}

#[test]
fn public_adapter_rejects_real_runtime_without_accepted_final_output_before_gpu_sat() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_runtime_without_accepted_final_output(&provider);
    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &provider).expect("SAT GpuCnf upload");
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());

    let err = match adapter.solve_expect_sat_with_gpu_execution_result(&provider, &result, &sat_cnf)
    {
        Ok(_) => panic!("runtime evidence with no accepted final rows must not gate GPU SAT"),
        Err(err) => err,
    };

    assert!(format!("{err}").contains("non-empty accepted GPU final output"));
    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 0);
    assert_eq!(trace.gpu_cdcl_sat_solves, 0);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("rejected evidence gate must not fall back to CPU solver search");
    trace
        .require_production_metric_eligibility()
        .expect_err("rejected evidence must not satisfy solver production metrics");
}

#[test]
fn public_adapter_rejects_provider_mismatched_runtime_evidence_before_gpu_sat() {
    let Some(evidence_provider) = try_provider() else {
        return;
    };
    let Some(adapter_provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_ground_literal(&evidence_provider);
    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf =
        GpuCnf::from_host(&sat_instance, &adapter_provider).expect("adapter SAT GpuCnf upload");
    let mut adapter =
        GpuSolverProductionAdapter::new(adapter_provider.clone(), GpuCdclConfig::default());

    let err = match adapter.solve_expect_sat_with_gpu_execution_result(
        &evidence_provider,
        &result,
        &sat_cnf,
    ) {
        Ok(_) => panic!("provider-mismatched runtime evidence must not gate GPU SAT"),
        Err(err) => err,
    };

    assert!(format!("{err}").contains("solver adapter provider mismatch"));
    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 0);
    assert_eq!(trace.gpu_cdcl_sat_solves, 0);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("provider-mismatched evidence rejection must not fall back to CPU search");
    trace
        .require_production_metric_eligibility()
        .expect_err("provider-mismatched evidence must not satisfy solver production metrics");
}

#[test]
fn public_adapter_consumes_real_runtime_accepted_evidence_before_encoded_maxsat() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_ground_literal(&provider);
    let weighted = SolveInstance::with_weights(
        3,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
            Clause::new(vec![Literal::positive(1)]),
            Clause::new(vec![Literal::positive(2)]),
        ],
        vec![4.0, 3.0, 2.0, 1.0],
    );
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(weighted.num_vars, weighted.clauses.len() as u32)
        .expect("new MaxSAT workspace");
    let branch_limit = upload_u32(&provider, weighted.num_vars);
    let contradictory_selection = [0usize, 1usize];
    let selections = [GpuSolverProductionWeightedMaxSatSelection {
        soft_clause_indices: &contradictory_selection,
        status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
    }];

    let report = adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_execution_result(
            &provider,
            &result,
            &mut workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("accepted runtime evidence should gate public encoded MaxSAT search");

    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.optimum_score, 7);
    assert_eq!(report.candidates_checked, 2);
    assert_eq!(report.satisfiable_candidates, 1);
    assert_eq!(report.unsat_candidates_pruned, 1);
    assert_eq!(report.gpu_cdcl_candidate_encodes, 2);
    assert_eq!(report.gpu_cdcl_candidate_solves, 2);
    assert_eq!(report.frontier_upper_bound_certificates, 1);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_candidate_state_transitions, 1);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 7);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 2);
    assert!(trace.gpu_maxsat_candidate_cnf_data_plane_htod_calls > 0);
    assert!(trace.gpu_maxsat_candidate_cnf_data_plane_htod_bytes > 0);
    assert_eq!(trace.gpu_maxsat_candidate_cnf_data_plane_dtoh_calls, 0);
    assert_eq!(trace.gpu_maxsat_candidate_cnf_data_plane_dtoh_bytes, 0);
    assert!(trace.gpu_maxsat_candidate_cnf_launch_metadata_htod_calls > 0);
    assert!(trace.gpu_maxsat_candidate_cnf_launch_metadata_htod_bytes > 0);
    assert_eq!(trace.gpu_maxsat_frontier_completion_candidate_encodes, 1);
    assert_eq!(trace.gpu_maxsat_frontier_upper_bound_certificates, 1);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 1);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(trace.gpu_maxsat_optima, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("accepted MaxSAT path must not use CPU solver search");
    trace
        .require_production_metric_eligibility()
        .expect("accepted runtime evidence plus public GPU MaxSAT search satisfies metric gate");
}

#[test]
fn public_adapter_rejects_mislabeled_encoded_maxsat_status_after_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_ground_literal(&provider);
    let weighted =
        SolveInstance::with_weights(1, vec![Clause::new(vec![Literal::positive(0)])], vec![2.0]);
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(weighted.num_vars, weighted.clauses.len() as u32)
        .expect("new mislabeled MaxSAT workspace");
    let branch_limit = upload_u32(&provider, weighted.num_vars);
    let satisfiable_selection = [0usize];
    let selections = [GpuSolverProductionWeightedMaxSatSelection {
        soft_clause_indices: &satisfiable_selection,
        status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
    }];

    adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_execution_result(
            &provider,
            &result,
            &mut workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect_err("GPU CDCL must reject a satisfiable MaxSAT candidate mislabeled UNSAT");

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 0);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 0);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 0);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 0);
    assert_eq!(trace.gpu_cdcl_sat_solves, 0);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 0);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    trace
        .require_zero_cpu_search()
        .expect("mislabeled MaxSAT rejection must not fall back to CPU search");
    trace
        .require_production_metric_eligibility()
        .expect_err("rejected MaxSAT evidence must not satisfy production metrics");
}

#[test]
fn public_adapter_certifies_independent_encoded_maxsat_frontiers_after_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_ground_literal(&provider);
    let weighted = SolveInstance::with_weights(
        3,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
            Clause::new(vec![Literal::positive(1)]),
            Clause::new(vec![Literal::negative(1)]),
            Clause::new(vec![Literal::positive(2)]),
        ],
        vec![4.0, 3.0, 2.0, 1.0, 5.0],
    );
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(weighted.num_vars, weighted.clauses.len() as u32)
        .expect("new independent-frontier MaxSAT workspace");
    let branch_limit = upload_u32(&provider, weighted.num_vars);
    let x_frontier = [0usize, 1usize];
    let y_frontier = [2usize, 3usize];
    let selections = [
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &x_frontier,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &y_frontier,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
    ];

    let report = adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_execution_result(
            &provider,
            &result,
            &mut workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("accepted runtime evidence should certify independent GPU MaxSAT frontiers");

    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.optimum_score, 11);
    assert_eq!(report.candidates_checked, 3);
    assert_eq!(report.satisfiable_candidates, 1);
    assert_eq!(report.unsat_candidates_pruned, 2);
    assert_eq!(report.gpu_cdcl_candidate_encodes, 3);
    assert_eq!(report.gpu_cdcl_candidate_solves, 3);
    assert_eq!(report.frontier_upper_bound_certificates, 1);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_candidate_state_transitions, 1);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 11);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 3);
    assert!(trace.gpu_maxsat_candidate_cnf_data_plane_htod_calls > 0);
    assert_eq!(trace.gpu_maxsat_candidate_cnf_data_plane_dtoh_calls, 0);
    assert_eq!(trace.gpu_maxsat_frontier_completion_candidate_encodes, 1);
    assert_eq!(trace.gpu_maxsat_frontier_upper_bound_certificates, 1);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 3);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.gpu_maxsat_optima, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    trace
        .require_zero_cpu_search()
        .expect("independent-frontier MaxSAT path must not use CPU search");
    trace
        .require_production_metric_eligibility()
        .expect("independent-frontier GPU MaxSAT satisfies metric gate");
}

#[test]
fn public_adapter_gates_encoded_maxsat_on_symbol_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_symbol_variable_bound_literal(&provider);
    let weighted = SolveInstance::with_weights(
        3,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
            Clause::new(vec![Literal::positive(1)]),
            Clause::new(vec![Literal::positive(2)]),
        ],
        vec![4.0, 3.0, 2.0, 1.0],
    );
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(weighted.num_vars, weighted.clauses.len() as u32)
        .expect("new symbol MaxSAT workspace");
    let branch_limit = upload_u32(&provider, weighted.num_vars);
    let contradictory_selection = [0usize, 1usize];
    let selections = [GpuSolverProductionWeightedMaxSatSelection {
        soft_clause_indices: &contradictory_selection,
        status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
    }];

    let report = adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_execution_result(
            &provider,
            &result,
            &mut workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("symbol accepted runtime evidence should gate encoded MaxSAT search");

    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.optimum_score, 7);
    assert_eq!(report.candidates_checked, 2);
    assert_eq!(report.satisfiable_candidates, 1);
    assert_eq!(report.unsat_candidates_pruned, 1);
    assert_eq!(report.gpu_cdcl_candidate_encodes, 2);
    assert_eq!(report.gpu_cdcl_candidate_solves, 2);
    assert_eq!(report.frontier_upper_bound_certificates, 1);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_candidate_state_transitions, 1);
    assert_eq!(trace.accepted_gpu_world_view_state_transitions, 1);
    assert_eq!(trace.accepted_gpu_candidate_final_output_rows_consumed, 2);
    assert_eq!(trace.accepted_faeel_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        1
    );
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(trace.accepted_solver_assumption_bindings_consumed, 1);
    assert_eq!(trace.accepted_solver_required_capabilities_consumed, 5);
    assert_eq!(trace.accepted_solver_required_statuses_consumed, 4);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 7);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 2);
    assert!(trace.gpu_maxsat_candidate_cnf_data_plane_htod_calls > 0);
    assert!(trace.gpu_maxsat_candidate_cnf_data_plane_htod_bytes > 0);
    assert_eq!(trace.gpu_maxsat_candidate_cnf_data_plane_dtoh_calls, 0);
    assert_eq!(trace.gpu_maxsat_candidate_cnf_data_plane_dtoh_bytes, 0);
    assert!(trace.gpu_maxsat_candidate_cnf_launch_metadata_htod_calls > 0);
    assert!(trace.gpu_maxsat_candidate_cnf_launch_metadata_htod_bytes > 0);
    assert_eq!(trace.gpu_maxsat_frontier_completion_candidate_encodes, 1);
    assert_eq!(trace.gpu_maxsat_frontier_upper_bound_certificates, 1);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 1);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(trace.gpu_maxsat_optima, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("symbol accepted MaxSAT path must not use CPU solver search");
    trace
        .require_production_metric_eligibility()
        .expect("symbol accepted evidence plus GPU MaxSAT search satisfies metric gate");
}

#[test]
fn public_adapter_consumes_real_runtime_accepted_evidence_before_candidate_set_maxsat() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_ground_literal(&provider);
    let lower_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let higher_instance = SolveInstance::new(
        2,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::positive(1)]),
        ],
    );
    let lower_cnf = GpuCnf::from_host(&lower_instance, &provider).expect("lower MaxSAT GpuCnf");
    let higher_cnf = GpuCnf::from_host(&higher_instance, &provider).expect("higher MaxSAT GpuCnf");
    let lower_branch_limit = upload_u32(&provider, lower_instance.num_vars);
    let higher_branch_limit = upload_u32(&provider, higher_instance.num_vars);
    let candidates = [
        GpuSolverProductionMaxSatCandidate {
            score: 2,
            cnf: &lower_cnf,
            branch_var_limit: &lower_branch_limit,
        },
        GpuSolverProductionMaxSatCandidate {
            score: 5,
            cnf: &higher_cnf,
            branch_var_limit: &higher_branch_limit,
        },
    ];
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());

    let report = adapter
        .solve_weighted_maxsat_candidates_with_gpu_execution_result(&provider, &result, &candidates)
        .expect("accepted runtime evidence should gate public GPU MaxSAT candidate set");

    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.optimum_score, 5);
    assert_eq!(report.candidates_checked, 2);
    assert_eq!(report.satisfiable_candidates, 2);
    assert_eq!(report.unsat_candidates_pruned, 0);
    assert_eq!(report.gpu_cdcl_candidate_solves, 2);
    assert_eq!(report.frontier_upper_bound_certificates, 0);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_candidate_state_transitions, 1);
    assert_eq!(trace.accepted_gpu_world_view_state_transitions, 1);
    assert_eq!(trace.accepted_solver_assumption_bindings_consumed, 1);
    assert_eq!(trace.accepted_solver_required_capabilities_consumed, 5);
    assert_eq!(trace.accepted_solver_required_statuses_consumed, 4);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 4);
    assert_eq!(trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(trace.gpu_maxsat_optima, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("accepted candidate-set MaxSAT path must not use CPU search");
    let err = trace
        .require_production_metric_eligibility()
        .expect_err("uncertified candidate-set MaxSAT must not satisfy production metrics");
    assert!(format!("{err}").contains("upper-bound certificate"));
}

#[test]
fn public_adapter_gates_encoded_maxsat_on_parsed_all_operator_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_parsed_all_operator_variable_bound_evidence(&provider);
    let weighted = SolveInstance::with_weights(
        3,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
            Clause::new(vec![Literal::positive(1)]),
            Clause::new(vec![Literal::positive(2)]),
        ],
        vec![4.0, 3.0, 2.0, 1.0],
    );
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(weighted.num_vars, weighted.clauses.len() as u32)
        .expect("new parsed MaxSAT workspace");
    let branch_limit = upload_u32(&provider, weighted.num_vars);
    let contradictory_selection = [0usize, 1usize];
    let selections = [GpuSolverProductionWeightedMaxSatSelection {
        soft_clause_indices: &contradictory_selection,
        status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
    }];

    let report = adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_execution_result(
            &provider,
            &result,
            &mut workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("parsed all-operator runtime evidence should gate encoded GPU MaxSAT search");

    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.optimum_score, 7);
    assert_eq!(report.candidates_checked, 2);
    assert_eq!(report.satisfiable_candidates, 1);
    assert_eq!(report.unsat_candidates_pruned, 1);
    assert_eq!(report.gpu_cdcl_candidate_encodes, 2);
    assert_eq!(report.gpu_cdcl_candidate_solves, 2);
    assert_eq!(report.frontier_upper_bound_certificates, 1);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_possible_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_not_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        4
    );
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 4);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        2
    );
    assert_eq!(trace.accepted_solver_assumption_bindings_consumed, 4);
    assert_eq!(trace.accepted_solver_required_capabilities_consumed, 5);
    assert_eq!(trace.accepted_solver_required_statuses_consumed, 4);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 7);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 2);
    assert!(trace.gpu_maxsat_candidate_cnf_data_plane_htod_calls > 0);
    assert_eq!(trace.gpu_maxsat_candidate_cnf_data_plane_dtoh_calls, 0);
    assert_eq!(trace.gpu_maxsat_frontier_completion_candidate_encodes, 1);
    assert_eq!(trace.gpu_maxsat_frontier_upper_bound_certificates, 1);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 1);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(trace.gpu_maxsat_optima, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("parsed accepted MaxSAT path must not use CPU search");
    trace.require_production_metric_eligibility().expect(
        "parsed all-operator runtime evidence plus encoded GPU MaxSAT satisfies metric gate",
    );
}

#[test]
fn public_adapter_gates_encoded_maxsat_on_g91_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_g91_possible_literal(&provider);
    let weighted = SolveInstance::with_weights(
        3,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
            Clause::new(vec![Literal::positive(1)]),
            Clause::new(vec![Literal::positive(2)]),
        ],
        vec![4.0, 3.0, 2.0, 1.0],
    );
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(weighted.num_vars, weighted.clauses.len() as u32)
        .expect("new G91 MaxSAT workspace");
    let branch_limit = upload_u32(&provider, weighted.num_vars);
    let contradictory_selection = [0usize, 1usize];
    let selections = [GpuSolverProductionWeightedMaxSatSelection {
        soft_clause_indices: &contradictory_selection,
        status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
    }];

    let report = adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_execution_result(
            &provider,
            &result,
            &mut workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("G91 accepted runtime evidence should gate encoded GPU MaxSAT search");

    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.optimum_score, 7);
    assert_eq!(report.candidates_checked, 2);
    assert_eq!(report.satisfiable_candidates, 1);
    assert_eq!(report.unsat_candidates_pruned, 1);
    assert_eq!(report.gpu_cdcl_candidate_encodes, 2);
    assert_eq!(report.gpu_cdcl_candidate_solves, 2);
    assert_eq!(report.frontier_upper_bound_certificates, 1);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_g91_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.accepted_possible_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 0);
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        1
    );
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(trace.accepted_solver_assumption_bindings_consumed, 1);
    assert_eq!(trace.accepted_solver_required_capabilities_consumed, 5);
    assert_eq!(trace.accepted_solver_required_statuses_consumed, 4);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 7);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 2);
    assert!(trace.gpu_maxsat_candidate_cnf_data_plane_htod_calls > 0);
    assert_eq!(trace.gpu_maxsat_candidate_cnf_data_plane_dtoh_calls, 0);
    assert_eq!(trace.gpu_maxsat_frontier_completion_candidate_encodes, 1);
    assert_eq!(trace.gpu_maxsat_frontier_upper_bound_certificates, 1);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 1);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(trace.gpu_maxsat_optima, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("G91 accepted MaxSAT path must not use CPU search");
    trace
        .require_production_metric_eligibility()
        .expect("G91 accepted runtime evidence plus encoded GPU MaxSAT satisfies metric gate");
}

#[test]
fn public_adapter_gates_encoded_maxsat_on_quaternary_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_quaternary_bound_literal(&provider);
    let weighted = SolveInstance::with_weights(
        3,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
            Clause::new(vec![Literal::positive(1)]),
            Clause::new(vec![Literal::positive(2)]),
        ],
        vec![4.0, 3.0, 2.0, 1.0],
    );
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(weighted.num_vars, weighted.clauses.len() as u32)
        .expect("new quaternary MaxSAT workspace");
    let branch_limit = upload_u32(&provider, weighted.num_vars);
    let contradictory_selection = [0usize, 1usize];
    let selections = [GpuSolverProductionWeightedMaxSatSelection {
        soft_clause_indices: &contradictory_selection,
        status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
    }];

    let report = adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_execution_result(
            &provider,
            &result,
            &mut workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("quaternary accepted runtime evidence should gate encoded GPU MaxSAT search");

    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.optimum_score, 7);
    assert_eq!(report.candidates_checked, 2);
    assert_eq!(report.satisfiable_candidates, 1);
    assert_eq!(report.unsat_candidates_pruned, 1);
    assert_eq!(report.gpu_cdcl_candidate_encodes, 2);
    assert_eq!(report.gpu_cdcl_candidate_solves, 2);
    assert_eq!(report.frontier_upper_bound_certificates, 1);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        4
    );
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(trace.accepted_solver_assumption_bindings_consumed, 1);
    assert_eq!(trace.accepted_solver_required_capabilities_consumed, 5);
    assert_eq!(trace.accepted_solver_required_statuses_consumed, 4);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 7);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 2);
    assert!(trace.gpu_maxsat_candidate_cnf_data_plane_htod_calls > 0);
    assert_eq!(trace.gpu_maxsat_candidate_cnf_data_plane_dtoh_calls, 0);
    assert_eq!(trace.gpu_maxsat_frontier_completion_candidate_encodes, 1);
    assert_eq!(trace.gpu_maxsat_frontier_upper_bound_certificates, 1);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 1);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(trace.gpu_maxsat_optima, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("quaternary accepted MaxSAT path must not use CPU search");
    trace.require_production_metric_eligibility().expect(
        "quaternary accepted runtime evidence plus encoded GPU MaxSAT satisfies metric gate",
    );
}

#[test]
fn public_adapter_preserves_status_distinction_after_real_runtime_accepted_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_ground_literal(&provider);
    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &provider).expect("SAT GpuCnf upload");
    let branch_limit = upload_u32(&provider, sat_instance.num_vars);
    let jobs = [
        GpuSolverProductionPortfolioJob::Sat {
            cnf: &sat_cnf,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionPortfolioJob::Unknown {
            reason: "accepted GPU portfolio budget ended inconclusively",
        },
        GpuSolverProductionPortfolioJob::Timeout { budget_micros: 1 },
    ];
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());

    let report = adapter
        .solve_portfolio_with_gpu_execution_result(&provider, &result, &jobs)
        .expect("accepted runtime evidence should gate status-aware GPU portfolio dispatch");

    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.jobs, 3);
    assert_eq!(report.sat_jobs, 1);
    assert_eq!(report.unknown_jobs, 1);
    assert_eq!(report.timeout_jobs, 1);
    assert_eq!(report.maxsat_jobs, 0);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_solver_required_statuses_consumed, 4);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 4);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.gpu_portfolio_jobs, 3);
    assert_eq!(trace.gpu_portfolio_sat_jobs, 1);
    assert_eq!(trace.gpu_portfolio_unknown_status_jobs, 1);
    assert_eq!(trace.gpu_portfolio_timeout_status_jobs, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    trace
        .require_zero_cpu_search()
        .expect("status-aware portfolio path must not use CPU search");
    trace
        .require_production_metric_eligibility()
        .expect("GPU SAT plus UNKNOWN/TIMEOUT status propagation remains eligible");
}

#[test]
fn public_adapter_dispatches_encoded_maxsat_portfolio_after_split_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let batch = execute_accepted_split_batch(&provider);
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
    let branch_limit = upload_u32(&provider, weighted.num_vars);
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
            reason: "accepted split GPU portfolio budget ended inconclusively",
        },
        GpuSolverProductionPortfolioJob::Timeout { budget_micros: 1 },
    ];
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());

    let report = adapter
        .solve_portfolio_with_gpu_batch_execution_result(
            &provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &jobs,
        )
        .expect("accepted split batch evidence should gate status-aware GPU MaxSAT portfolio");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.jobs, 8);
    assert_eq!(report.sat_jobs, 2);
    assert_eq!(report.maxsat_jobs, 2);
    assert_eq!(report.unknown_jobs, 2);
    assert_eq!(report.timeout_jobs, 2);
    assert_eq!(report.maxsat_optimum_scores, 4);
    assert_eq!(report.maxsat_candidates_checked, 4);
    assert_eq!(report.maxsat_satisfiable_candidates, 2);
    assert_eq!(report.maxsat_unsat_candidates_pruned, 2);
    assert_eq!(report.maxsat_gpu_cdcl_candidate_encodes, 4);
    assert_eq!(report.maxsat_gpu_cdcl_candidate_solves, 4);
    assert_eq!(report.maxsat_frontier_upper_bound_certificates, 2);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.accepted_gpu_candidate_state_transitions, 2);
    assert_eq!(trace.accepted_gpu_world_view_state_transitions, 2);
    assert_eq!(trace.accepted_gpu_candidate_final_output_rows_consumed, 2);
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(trace.accepted_solver_assumption_bindings_consumed, 2);
    assert_eq!(trace.accepted_solver_required_capabilities_consumed, 10);
    assert_eq!(trace.accepted_solver_required_statuses_consumed, 8);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 24);
    assert_eq!(trace.gpu_portfolio_jobs, 8);
    assert_eq!(trace.gpu_portfolio_sat_jobs, 2);
    assert_eq!(trace.gpu_portfolio_maxsat_jobs, 2);
    assert_eq!(trace.gpu_portfolio_unknown_status_jobs, 2);
    assert_eq!(trace.gpu_portfolio_timeout_status_jobs, 2);
    assert_eq!(trace.gpu_cdcl_sat_solves, 4);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 4);
    assert!(trace.gpu_maxsat_candidate_cnf_data_plane_htod_calls > 0);
    assert!(trace.gpu_maxsat_candidate_cnf_data_plane_htod_bytes > 0);
    assert_eq!(trace.gpu_maxsat_candidate_cnf_data_plane_dtoh_calls, 0);
    assert_eq!(trace.gpu_maxsat_candidate_cnf_data_plane_dtoh_bytes, 0);
    assert!(trace.gpu_maxsat_candidate_cnf_launch_metadata_htod_calls > 0);
    assert!(trace.gpu_maxsat_candidate_cnf_launch_metadata_htod_bytes > 0);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(trace.gpu_maxsat_frontier_completion_candidate_encodes, 2);
    assert_eq!(trace.gpu_maxsat_frontier_upper_bound_certificates, 2);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(trace.gpu_maxsat_optima, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("accepted portfolio MaxSAT path must not use CPU search");
    trace
        .require_production_metric_eligibility()
        .expect("accepted split batch plus GPU SAT/MaxSAT portfolio satisfies metric gate");
}

#[test]
fn public_adapter_dispatches_encoded_maxsat_scheduler_after_split_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let batch = execute_accepted_split_batch(&provider);
    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![2.0, 1.0],
    );
    let branch_limit = upload_u32(&provider, weighted.num_vars);
    let contradictory_selection = [0usize, 1usize];
    let selections = [GpuSolverProductionWeightedMaxSatSelection {
        soft_clause_indices: &contradictory_selection,
        status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
    }];
    let jobs = [
        GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
            weighted: &weighted,
            branch_var_limit: &branch_limit,
            selections: &selections,
        },
        GpuSolverProductionMaxSatScheduleJob::Unknown {
            reason: "accepted split GPU MaxSAT scheduler budget ended inconclusively",
        },
        GpuSolverProductionMaxSatScheduleJob::Timeout { budget_micros: 1 },
    ];
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(weighted.num_vars, weighted.clauses.len() as u32)
        .expect("new scheduler workspace");

    let report = adapter
        .solve_maxsat_schedule_with_gpu_batch_execution_result(
            &provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &jobs,
        )
        .expect("accepted split batch evidence should gate encoded GPU MaxSAT scheduler");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.jobs, 6);
    assert_eq!(report.candidate_set_jobs, 0);
    assert_eq!(report.search_jobs, 0);
    assert_eq!(report.encoded_search_jobs, 2);
    assert_eq!(report.unknown_jobs, 2);
    assert_eq!(report.timeout_jobs, 2);
    assert_eq!(report.optimum_score, 2);
    assert_eq!(report.candidates_checked, 4);
    assert_eq!(report.satisfiable_candidates, 2);
    assert_eq!(report.unsat_candidates_pruned, 2);
    assert_eq!(report.gpu_cdcl_candidate_encodes, 4);
    assert_eq!(report.gpu_cdcl_candidate_solves, 4);
    assert_eq!(report.frontier_upper_bound_certificates, 2);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.accepted_gpu_candidate_state_transitions, 2);
    assert_eq!(trace.accepted_gpu_world_view_state_transitions, 2);
    assert_eq!(trace.accepted_gpu_candidate_final_output_rows_consumed, 2);
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(trace.accepted_solver_assumption_bindings_consumed, 2);
    assert_eq!(trace.accepted_solver_required_capabilities_consumed, 10);
    assert_eq!(trace.accepted_solver_required_statuses_consumed, 8);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 20);
    assert_eq!(trace.gpu_maxsat_scheduler_jobs, 6);
    assert_eq!(trace.gpu_maxsat_scheduler_candidate_set_jobs, 0);
    assert_eq!(trace.gpu_maxsat_scheduler_search_jobs, 0);
    assert_eq!(trace.gpu_maxsat_scheduler_encoded_search_jobs, 2);
    assert_eq!(trace.gpu_maxsat_scheduler_unknown_status_jobs, 2);
    assert_eq!(trace.gpu_maxsat_scheduler_timeout_status_jobs, 2);
    assert_eq!(trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 4);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(trace.gpu_maxsat_frontier_completion_candidate_encodes, 2);
    assert_eq!(trace.gpu_maxsat_frontier_upper_bound_certificates, 2);
    assert_eq!(trace.gpu_maxsat_frontier_certified_candidate_solves, 4);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(trace.gpu_maxsat_optima, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("accepted MaxSAT scheduler path must not use CPU search");
    trace
        .require_production_metric_eligibility()
        .expect("accepted split batch plus encoded GPU MaxSAT scheduler satisfies metric gate");
}

#[test]
fn public_adapter_runs_assumption_lifecycle_after_real_runtime_accepted_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_ground_literal(&provider);
    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &provider).expect("SAT GpuCnf upload");
    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let unsat_cnf = GpuCnf::from_host(&unsat_instance, &provider).expect("UNSAT GpuCnf upload");
    let sat_branch_limit = upload_u32(&provider, sat_instance.num_vars);
    let unsat_branch_limit = upload_u32(&provider, unsat_instance.num_vars);
    let steps = [
        GpuSolverProductionLifecycleStep {
            cnf: &sat_cnf,
            branch_var_limit: &sat_branch_limit,
            expectation: GpuSolverProductionExpectation::Sat,
        },
        GpuSolverProductionLifecycleStep {
            cnf: &unsat_cnf,
            branch_var_limit: &unsat_branch_limit,
            expectation: GpuSolverProductionExpectation::Unsat,
        },
        GpuSolverProductionLifecycleStep {
            cnf: &sat_cnf,
            branch_var_limit: &sat_branch_limit,
            expectation: GpuSolverProductionExpectation::Unknown {
                reason: "accepted GPU lifecycle budget ended inconclusively",
            },
        },
        GpuSolverProductionLifecycleStep {
            cnf: &sat_cnf,
            branch_var_limit: &sat_branch_limit,
            expectation: GpuSolverProductionExpectation::Timeout { budget_micros: 1 },
        },
    ];
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(unsat_cnf.var_cap, unsat_cnf.clause_cap)
        .expect("new lifecycle workspace");

    let report = adapter
        .solve_assumption_lifecycle_with_gpu_execution_result(
            &provider,
            &result,
            &mut workspace,
            &steps,
        )
        .expect("accepted runtime evidence should gate GPU assumption lifecycle");

    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.steps, 4);
    assert_eq!(report.sat_steps, 1);
    assert_eq!(report.unsat_steps, 1);
    assert_eq!(report.assumption_pushes, 4);
    assert_eq!(report.assumption_retractions, 4);
    assert_eq!(report.workspace_reuses, 1);
    assert_eq!(report.unknown_steps, 1);
    assert_eq!(report.timeout_steps, 1);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_solver_assumption_bindings_consumed, 1);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 4);
    assert_eq!(trace.gpu_assumption_pushes, 4);
    assert_eq!(trace.gpu_assumption_retractions, 4);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 1);
    assert_eq!(trace.gpu_lifecycle_unknown_status_steps, 1);
    assert_eq!(trace.gpu_lifecycle_timeout_status_steps, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("accepted lifecycle path must not use CPU search");
    trace
        .require_production_metric_eligibility()
        .expect("accepted runtime evidence plus GPU lifecycle work satisfies metric gate");
}

#[test]
fn public_adapter_rejects_mislabeled_lifecycle_status_after_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_ground_literal(&provider);
    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let unsat_cnf = GpuCnf::from_host(&unsat_instance, &provider).expect("UNSAT GpuCnf upload");
    let branch_limit = upload_u32(&provider, unsat_instance.num_vars);
    let steps = [GpuSolverProductionLifecycleStep {
        cnf: &unsat_cnf,
        branch_var_limit: &branch_limit,
        expectation: GpuSolverProductionExpectation::Sat,
    }];
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(unsat_cnf.var_cap, unsat_cnf.clause_cap)
        .expect("new mislabeled lifecycle workspace");

    adapter
        .solve_assumption_lifecycle_with_gpu_execution_result(
            &provider,
            &result,
            &mut workspace,
            &steps,
        )
        .expect_err("GPU CDCL must reject an UNSAT lifecycle step mislabeled SAT");

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 0);
    assert_eq!(trace.gpu_assumption_pushes, 0);
    assert_eq!(trace.gpu_assumption_retractions, 0);
    assert_eq!(trace.gpu_cdcl_sat_solves, 0);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 0);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 0);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    trace
        .require_zero_cpu_search()
        .expect("mislabeled lifecycle rejection must not fall back to CPU search");
    trace
        .require_production_metric_eligibility()
        .expect_err("rejected lifecycle evidence must not satisfy production metrics");
}

#[test]
fn public_adapter_reuses_learned_clause_arena_after_real_runtime_accepted_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_ground_literal(&provider);
    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let unsat_cnf = GpuCnf::from_host(&unsat_instance, &provider).expect("UNSAT GpuCnf upload");
    let branch_limit = upload_u32(&provider, unsat_instance.num_vars);
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(unsat_cnf.var_cap, unsat_cnf.clause_cap)
        .expect("new learned-clause workspace");

    let report = adapter
        .solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result(
            &provider,
            &result,
            &mut workspace,
            &unsat_cnf,
            &branch_limit,
            &unsat_cnf,
            &branch_limit,
        )
        .expect("accepted runtime evidence should gate GPU learned-clause reuse");

    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.candidates, 2);
    assert_eq!(report.unsat_solves, 2);
    assert_eq!(report.gpu_learned_clause_arena_publications, 1);
    assert_eq!(report.gpu_learned_clause_imports, 1);
    assert_eq!(report.gpu_learned_clause_reused_solves, 1);
    assert_eq!(report.cpu_learned_clause_transfers, 0);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_solver_assumption_bindings_consumed, 1);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 6);
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
        .expect("accepted learned-clause path must not use CPU search");
    trace
        .require_production_metric_eligibility()
        .expect("accepted runtime evidence plus learned-clause reuse satisfies metric gate");
}

#[test]
fn public_adapter_runs_batch_assumption_lifecycle_after_split_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let batch = execute_accepted_split_batch(&provider);
    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &provider).expect("SAT GpuCnf upload");
    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let unsat_cnf = GpuCnf::from_host(&unsat_instance, &provider).expect("UNSAT GpuCnf upload");
    let sat_branch_limit = upload_u32(&provider, sat_instance.num_vars);
    let unsat_branch_limit = upload_u32(&provider, unsat_instance.num_vars);
    let steps = [
        GpuSolverProductionLifecycleStep {
            cnf: &sat_cnf,
            branch_var_limit: &sat_branch_limit,
            expectation: GpuSolverProductionExpectation::Sat,
        },
        GpuSolverProductionLifecycleStep {
            cnf: &unsat_cnf,
            branch_var_limit: &unsat_branch_limit,
            expectation: GpuSolverProductionExpectation::Unsat,
        },
    ];
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(unsat_cnf.var_cap, unsat_cnf.clause_cap)
        .expect("new batch lifecycle workspace");

    let report = adapter
        .solve_assumption_lifecycle_with_gpu_batch_execution_result(
            &provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &steps,
        )
        .expect("accepted split batch evidence should gate GPU assumption lifecycle");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.steps, 4);
    assert_eq!(report.sat_steps, 2);
    assert_eq!(report.unsat_steps, 2);
    assert_eq!(report.assumption_pushes, 4);
    assert_eq!(report.assumption_retractions, 4);
    assert_eq!(report.workspace_reuses, 2);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.accepted_gpu_candidate_state_transitions, 2);
    assert_eq!(trace.accepted_gpu_world_view_state_transitions, 2);
    assert_eq!(trace.accepted_gpu_candidate_final_output_rows_consumed, 2);
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(trace.accepted_solver_assumption_bindings_consumed, 2);
    assert_eq!(trace.accepted_solver_required_capabilities_consumed, 10);
    assert_eq!(trace.accepted_solver_required_statuses_consumed, 8);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 4);
    assert_eq!(trace.gpu_assumption_pushes, 4);
    assert_eq!(trace.gpu_assumption_retractions, 4);
    assert_eq!(trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("accepted batch lifecycle path must not use CPU search");
    trace
        .require_production_metric_eligibility()
        .expect("accepted split batch plus GPU lifecycle work satisfies metric gate");
}

#[test]
fn public_adapter_preserves_batch_lifecycle_statuses_after_split_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let batch = execute_accepted_split_batch(&provider);
    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &provider).expect("SAT GpuCnf upload");
    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let unsat_cnf = GpuCnf::from_host(&unsat_instance, &provider).expect("UNSAT GpuCnf upload");
    let sat_branch_limit = upload_u32(&provider, sat_instance.num_vars);
    let unsat_branch_limit = upload_u32(&provider, unsat_instance.num_vars);
    let steps = [
        GpuSolverProductionLifecycleStep {
            cnf: &sat_cnf,
            branch_var_limit: &sat_branch_limit,
            expectation: GpuSolverProductionExpectation::Sat,
        },
        GpuSolverProductionLifecycleStep {
            cnf: &unsat_cnf,
            branch_var_limit: &unsat_branch_limit,
            expectation: GpuSolverProductionExpectation::Unsat,
        },
        GpuSolverProductionLifecycleStep {
            cnf: &sat_cnf,
            branch_var_limit: &sat_branch_limit,
            expectation: GpuSolverProductionExpectation::Unknown {
                reason: "accepted split GPU lifecycle budget ended inconclusively",
            },
        },
        GpuSolverProductionLifecycleStep {
            cnf: &sat_cnf,
            branch_var_limit: &sat_branch_limit,
            expectation: GpuSolverProductionExpectation::Timeout { budget_micros: 1 },
        },
    ];
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(unsat_cnf.var_cap, unsat_cnf.clause_cap)
        .expect("new batch status lifecycle workspace");

    let report = adapter
        .solve_assumption_lifecycle_with_gpu_batch_execution_result(
            &provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &steps,
        )
        .expect("accepted split batch evidence should preserve lifecycle status distinctions");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.steps, 8);
    assert_eq!(report.sat_steps, 2);
    assert_eq!(report.unsat_steps, 2);
    assert_eq!(report.assumption_pushes, 8);
    assert_eq!(report.assumption_retractions, 8);
    assert_eq!(report.workspace_reuses, 2);
    assert_eq!(report.unknown_steps, 2);
    assert_eq!(report.timeout_steps, 2);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.accepted_gpu_candidate_state_transitions, 2);
    assert_eq!(trace.accepted_gpu_world_view_state_transitions, 2);
    assert_eq!(trace.accepted_gpu_candidate_final_output_rows_consumed, 2);
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(trace.accepted_solver_assumption_bindings_consumed, 2);
    assert_eq!(trace.accepted_solver_required_capabilities_consumed, 10);
    assert_eq!(trace.accepted_solver_required_statuses_consumed, 8);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 8);
    assert_eq!(trace.gpu_assumption_pushes, 8);
    assert_eq!(trace.gpu_assumption_retractions, 8);
    assert_eq!(trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 2);
    assert_eq!(trace.gpu_lifecycle_unknown_status_steps, 2);
    assert_eq!(trace.gpu_lifecycle_timeout_status_steps, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("accepted batch lifecycle status path must not use CPU search");
    trace
        .require_production_metric_eligibility()
        .expect("accepted split batch plus GPU lifecycle statuses satisfies metric gate");
}

#[test]
fn public_adapter_runs_batch_encoded_maxsat_after_split_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let batch = execute_accepted_split_batch(&provider);
    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![2.0, 1.0],
    );
    let contradictory_selection = [0usize, 1usize];
    let selections = [GpuSolverProductionWeightedMaxSatSelection {
        soft_clause_indices: &contradictory_selection,
        status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
    }];
    let branch_limit = upload_u32(&provider, weighted.num_vars);
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(weighted.num_vars, weighted.clauses.len() as u32)
        .expect("new batch MaxSAT workspace");

    let report = adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_batch_execution_result(
            &provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("accepted split batch evidence should gate encoded GPU MaxSAT");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.optimum_score, 2);
    assert_eq!(report.candidates_checked, 4);
    assert_eq!(report.satisfiable_candidates, 2);
    assert_eq!(report.unsat_candidates_pruned, 2);
    assert_eq!(report.gpu_cdcl_candidate_encodes, 4);
    assert_eq!(report.gpu_cdcl_candidate_solves, 4);
    assert_eq!(report.frontier_upper_bound_certificates, 2);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.accepted_gpu_candidate_state_transitions, 2);
    assert_eq!(trace.accepted_gpu_world_view_state_transitions, 2);
    assert_eq!(trace.accepted_gpu_candidate_final_output_rows_consumed, 2);
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(trace.accepted_solver_assumption_bindings_consumed, 2);
    assert_eq!(trace.accepted_solver_required_capabilities_consumed, 10);
    assert_eq!(trace.accepted_solver_required_statuses_consumed, 8);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 14);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 4);
    assert_eq!(trace.gpu_maxsat_frontier_completion_candidate_encodes, 2);
    assert_eq!(trace.gpu_maxsat_frontier_upper_bound_certificates, 2);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.gpu_maxsat_optima, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("accepted batch encoded MaxSAT path must not use CPU search");
    trace
        .require_production_metric_eligibility()
        .expect("accepted split batch plus encoded GPU MaxSAT satisfies metric gate");
}

#[test]
fn public_adapter_runs_batch_encoded_maxsat_after_modal_hidden_body_local_split_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let batch = execute_modal_hidden_body_local_split_batch(&provider);
    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![2.0, 1.0],
    );
    let contradictory_selection = [0usize, 1usize];
    let selections = [GpuSolverProductionWeightedMaxSatSelection {
        soft_clause_indices: &contradictory_selection,
        status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
    }];
    let branch_limit = upload_u32(&provider, weighted.num_vars);
    let mut adapter = GpuSolverProductionAdapter::new(provider.clone(), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(weighted.num_vars, weighted.clauses.len() as u32)
        .expect("new modal body-local batch MaxSAT workspace");

    let report = adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_batch_execution_result(
            &provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("modal hidden body-local split batch should gate encoded GPU MaxSAT");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.optimum_score, 2);
    assert_eq!(report.candidates_checked, 4);
    assert_eq!(report.satisfiable_candidates, 2);
    assert_eq!(report.unsat_candidates_pruned, 2);
    assert_eq!(report.gpu_cdcl_candidate_encodes, 4);
    assert_eq!(report.gpu_cdcl_candidate_solves, 4);
    assert_eq!(report.frontier_upper_bound_certificates, 2);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.accepted_gpu_candidate_state_transitions, 2);
    assert_eq!(trace.accepted_gpu_world_view_state_transitions, 2);
    assert_eq!(trace.accepted_gpu_candidate_final_output_rows_consumed, 4);
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(trace.accepted_possible_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 2);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        1
    );
    assert_eq!(trace.accepted_solver_assumption_bindings_consumed, 2);
    assert_eq!(trace.accepted_solver_required_capabilities_consumed, 10);
    assert_eq!(trace.accepted_solver_required_statuses_consumed, 8);
    assert_eq!(trace.accepted_gpu_solver_production_path_events, 14);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 4);
    assert_eq!(trace.gpu_maxsat_frontier_completion_candidate_encodes, 2);
    assert_eq!(trace.gpu_maxsat_frontier_upper_bound_certificates, 2);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.gpu_maxsat_optima, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    trace
        .require_zero_cpu_search()
        .expect("modal body-local batch encoded MaxSAT path must not use CPU search");
    trace
        .require_production_metric_eligibility()
        .expect("modal body-local split batch plus encoded GPU MaxSAT satisfies metric gate");
}

fn execute_accepted_ground_literal(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let mut executor = Executor::new(provider.clone());
    executor.register_relation(RelId(1), "base");
    executor.put_relation("base", upload_unary_u32(provider, &[7], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &accepted_ground_literal_executable(),
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 4,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("real runtime accepted GPU epistemic execution");

    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(result.prepared.preflight.solver_assumption_binding_count, 1);
    assert_eq!(
        result.prepared.preflight.solver_required_capability_count,
        5
    );
    assert_eq!(result.prepared.preflight.solver_required_status_count, 4);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    result
}

fn execute_runtime_without_accepted_final_output(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        pred seed(u32).
        pred gate(u32).
        pred out(u32).

        out(X) :- seed(X), know gate(X).
        "#,
    )
    .expect("parse empty-output solver evidence program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile empty-output epistemic GPU plan");
    let mut executor = Executor::new(provider.clone());

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation("seed", upload_unary_u32(provider, &[7], "x"));
    executor.put_relation("gate", upload_unary_u32(provider, &[8], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("real runtime should produce GPU semantic evidence with no final rows");

    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.solver_assumption_binding_count, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.semantic_trace.accepted_candidates, 0);
    assert_eq!(result.semantic_trace.rejected_candidates, 2);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.final_result_transfer.final_output_rows, 0);
    result
        .require_runtime_dispatch_certification()
        .expect("empty-output evidence should still retain GPU runtime certification");
    result
}

fn execute_parsed_all_operator_variable_bound_evidence(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        pred seed(u32).
        pred known_gate(u32).
        pred possible_gate(u32).
        pred not_known_gate(u32).
        pred not_possible_gate(u32).
        pred out(u32).

        out(X) :- seed(X), know known_gate(X), possible possible_gate(X),
                  not know not_known_gate(X), not possible not_possible_gate(X).
        "#,
    )
    .expect("parse all-operator epistemic solver evidence program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile parsed epistemic GPU plan");
    let mut executor = Executor::new(provider.clone());

    for (name, rows) in [
        ("seed", &[7][..]),
        ("known_gate", &[7][..]),
        ("possible_gate", &[7][..]),
        ("not_known_gate", &[8][..]),
        ("not_possible_gate", &[9][..]),
    ] {
        let rel = *executable
            .relation_ids
            .get(name)
            .unwrap_or_else(|| panic!("compiled plan should expose relation id for {name}"));
        executor.register_relation(rel, name);
        executor.put_relation(name, upload_unary_u32(provider, rows, "x"));
    }

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 16,
                max_worlds: 4,
                max_models_per_reduction: 1,
            },
        )
        .expect("parsed all-operator program should execute on GPU before solver handoff");

    assert_eq!(result.prepared.preflight.reduced_runtime_rule_count, 1);
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.solver_assumption_binding_count, 4);
    assert_eq!(
        result.prepared.preflight.solver_required_capability_count,
        5
    );
    assert_eq!(result.prepared.preflight.solver_required_status_count, 4);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 4);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        2
    );
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![15]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidates, 15);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);

    let values = provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download parsed all-operator solver evidence values");
    assert_eq!(values, vec![7]);

    result
        .require_runtime_dispatch_certification()
        .expect("parsed solver evidence should retain GPU runtime certification");
    result
}

fn execute_accepted_symbol_variable_bound_literal(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        pred seed(symbol).
        pred gate(symbol).
        pred out(symbol).

        out(X) :- seed(X), know gate(X).
        "#,
    )
    .expect("parse symbol accepted solver evidence program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile symbol epistemic GPU plan");
    let mut executor = Executor::new(provider.clone());

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    let alpha = symbol::intern("solver-alpha");
    let beta = symbol::intern("solver-beta");
    let gamma = symbol::intern("solver-gamma");
    executor.put_relation(
        "seed",
        upload_unary_typed_u32(provider, &[alpha, beta, gamma], "x", ScalarType::Symbol),
    );
    executor.put_relation(
        "gate",
        upload_unary_typed_u32(provider, &[alpha, gamma], "x", ScalarType::Symbol),
    );

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("symbol program should execute on GPU before solver MaxSAT handoff");

    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.solver_assumption_binding_count, 1);
    assert_eq!(
        result.prepared.preflight.solver_required_capability_count,
        5
    );
    assert_eq!(result.prepared.preflight.solver_required_status_count, 4);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_output.schema().column_type(0),
        Some(ScalarType::Symbol)
    );
    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidates, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let mut rows = provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final symbol solver output values");
    rows.sort_unstable();
    assert_eq!(rows, vec![alpha, gamma]);
    result
        .require_runtime_dispatch_certification()
        .expect("symbol solver evidence should retain GPU runtime certification");
    result
}

fn execute_accepted_g91_possible_literal(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        #pragma epistemic_mode = g91
        pred seed(u32).
        pred p(u32).

        p(X) :- seed(X), possible p(X).
        "#,
    )
    .expect("parse G91 accepted solver evidence program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile G91 epistemic GPU plan");
    let mut executor = Executor::new(provider.clone());

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation("seed", upload_unary_u32(provider, &[7], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("G91 possible program should execute on GPU before solver handoff");

    assert_eq!(
        result.prepared.preflight.epistemic_mode,
        EirEpistemicMode::G91
    );
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.solver_assumption_binding_count, 1);
    assert_eq!(
        result.prepared.preflight.solver_required_capability_count,
        5
    );
    assert_eq!(result.prepared.preflight.solver_required_status_count, 4);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    result
        .require_runtime_dispatch_certification()
        .expect("G91 solver evidence should retain GPU runtime certification");
    result
}

fn execute_accepted_quaternary_bound_literal(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        pred seed(u32, u32, u32, u32).
        pred gate(u32, u32, u32, u32).
        pred out(u32, u32, u32, u32).

        out(W, X, Y, Z) :- seed(W, X, Y, Z), know gate(W, X, Y, Z).
        "#,
    )
    .expect("parse quaternary accepted solver evidence program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile quaternary epistemic GPU plan");
    let mut executor = Executor::new(provider.clone());

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "seed",
        upload_quaternary_u32(
            provider,
            &[(1, 2, 3, 4), (1, 2, 3, 5), (2, 3, 5, 8)],
            "w",
            "x",
            "y",
            "z",
        ),
    );
    executor.put_relation(
        "gate",
        upload_quaternary_u32(provider, &[(1, 2, 3, 4), (2, 3, 5, 8)], "w", "x", "y", "z"),
    );

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("quaternary program should execute on GPU before solver handoff");

    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.solver_assumption_binding_count, 1);
    assert_eq!(
        result.prepared.preflight.solver_required_capability_count,
        5
    );
    assert_eq!(result.prepared.preflight.solver_required_status_count, 4);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result
            .final_tuple_materialization
            .row_specific_membership_row_capacity,
        1
    );
    assert_eq!(
        result
            .final_tuple_materialization
            .row_filter_row_capacity_outside_model_slot_window,
        2
    );
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let ws = provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final quaternary solver w values");
    let xs = provider
        .download_column::<u32>(&result.final_output, 1)
        .expect("download final quaternary solver x values");
    let ys = provider
        .download_column::<u32>(&result.final_output, 2)
        .expect("download final quaternary solver y values");
    let zs = provider
        .download_column::<u32>(&result.final_output, 3)
        .expect("download final quaternary solver z values");
    let mut rows: Vec<_> = ws
        .into_iter()
        .zip(xs)
        .zip(ys)
        .zip(zs)
        .map(|(((w, x), y), z)| (w, x, y, z))
        .collect();
    rows.sort_unstable();
    assert_eq!(rows, vec![(1, 2, 3, 4), (2, 3, 5, 8)]);
    result
        .require_runtime_dispatch_certification()
        .expect("quaternary solver evidence should retain GPU runtime certification");
    result
}

fn execute_accepted_split_batch(
    provider: &Arc<CudaKernelProvider>,
) -> EpistemicGpuBatchExecutionResult {
    let mut executor = Executor::new(provider.clone());
    executor.register_relation(RelId(11), "left_base");
    executor.register_relation(RelId(12), "right_base");
    executor.put_relation("left_base", upload_unary_u32(provider, &[7], "x"));
    executor.put_relation("right_base", upload_unary_u32(provider, &[9], "x"));

    let left = accepted_ground_literal_component_executable("left_base", "left_out", RelId(11), 7);
    let right =
        accepted_ground_literal_component_executable("right_base", "right_out", RelId(12), 9);
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &[&left, &right],
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 4,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("real split components should execute through GPU runtime batch adapter");

    assert_eq!(batch.results.len(), 2);
    batch
        .require_trace_matches_components("solver accepted split batch")
        .expect("batch trace must match real component results");
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 2);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.cpu_solver_search_fallbacks, 0);
    assert_eq!(batch.trace.cpu_probability_recomputations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert_eq!(batch.trace.final_output_rows, 2);
    assert_eq!(batch.trace.accepted_world_views, 2);
    batch
}

fn execute_modal_hidden_body_local_split_batch(
    provider: &Arc<CudaKernelProvider>,
) -> EpistemicGpuBatchExecutionResult {
    let program = parse_program(
        r#"
        pred left_edge(u32, u32).
        pred left_maybe(u32).
        pred left_out(u32).
        pred right_edge(u32, u32).
        pred right_maybe_blocked(u32).
        pred right_out(u32).

        left_out(Y) :- left_edge(X, Y), possible left_maybe(X).
        right_out(Y) :- right_edge(X, Y), not possible right_maybe_blocked(X).
        "#,
    )
    .expect("parse modal split body-local solver evidence program");
    let split = compile_epistemic_gpu_split_execution(&program)
        .expect("compile modal split body-local solver evidence program");
    let mut executor = Executor::new(provider.clone());

    for component in split.recomposed_components() {
        for (name, rel) in &component.executable.relation_ids {
            executor.register_relation(*rel, name);
        }
    }
    executor.put_relation(
        "left_edge",
        upload_binary_u32(provider, &[(1, 10), (2, 20), (3, 30), (4, 10)], "x", "y"),
    );
    executor.put_relation("left_maybe", upload_unary_u32(provider, &[1, 3, 4], "x"));
    executor.put_relation(
        "right_edge",
        upload_binary_u32(provider, &[(5, 40), (6, 50), (7, 60), (8, 40)], "x", "y"),
    );
    executor.put_relation("right_maybe_blocked", upload_unary_u32(provider, &[6], "x"));

    let recomposed_components = split.recomposed_components();
    let executables: Vec<_> = recomposed_components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("modal split body-local solver components should execute through GPU batch");

    assert_eq!(batch.results.len(), 2);
    batch
        .require_trace_matches_components("solver modal hidden body-local split batch")
        .expect("modal hidden body-local split batch trace must match real component results");
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 2);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.cpu_solver_search_fallbacks, 0);
    assert_eq!(batch.trace.cpu_probability_recomputations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert_eq!(batch.trace.final_output_rows, 4);
    assert_eq!(batch.trace.accepted_world_views, 2);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.results[0].final_output.arity(), 1);
    assert_eq!(batch.results[0].output.arity(), 2);
    assert_eq!(batch.results[1].final_output.arity(), 1);
    assert_eq!(batch.results[1].output.arity(), 2);
    assert_eq!(
        batch.results[0]
            .final_tuple_materialization
            .row_filter_count,
        1
    );
    assert_eq!(
        batch.results[0]
            .final_tuple_materialization
            .negated_row_filter_count,
        0
    );
    assert_eq!(
        batch.results[1]
            .final_tuple_materialization
            .row_filter_count,
        1
    );
    assert_eq!(
        batch.results[1]
            .final_tuple_materialization
            .negated_row_filter_count,
        1
    );
    batch
}

fn accepted_ground_literal_executable() -> EpistemicExecutablePlan {
    let gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![EirEpistemicLiteral {
            op: EirEpistemicOp::Know,
            negated: false,
            atom: EirAtom {
                predicate: "base".to_string(),
                arity: 1,
                terms: vec![EirTerm::Integer(7)],
            },
        }],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "out".to_string(),
            relational_body_atoms: 1,
            wcoj_status: EpistemicWcojReductionStatus::NotWcojCandidate,
        }],
    );

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: BTreeMap::new(),
        reduced_runtime_plan: scan_base_into_out_plan(),
    }
}

fn accepted_ground_literal_component_executable(
    predicate: &str,
    output_predicate: &str,
    rel: RelId,
    value: i64,
) -> EpistemicExecutablePlan {
    let gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![EirEpistemicLiteral {
            op: EirEpistemicOp::Know,
            negated: false,
            atom: EirAtom {
                predicate: predicate.to_string(),
                arity: 1,
                terms: vec![EirTerm::Integer(value)],
            },
        }],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: output_predicate.to_string(),
            relational_body_atoms: 1,
            wcoj_status: EpistemicWcojReductionStatus::NotWcojCandidate,
        }],
    );

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: BTreeMap::new(),
        reduced_runtime_plan: scan_relation_into_output_plan(rel, output_predicate),
    }
}

fn scan_base_into_out_plan() -> ExecutionPlan {
    let mut plan = ExecutionPlan::new(vec![Scc {
        id: 0,
        predicates: vec!["out".to_string()],
        is_recursive: false,
    }])
    .with_strata(vec![Stratum {
        id: 0,
        sccs: vec![0],
    }]);
    plan.rules_by_scc = vec![vec![CompiledRule {
        head: "out".to_string(),
        body: RirNode::Scan { rel: RelId(1) },
        meta: RirMeta::with_schema(Schema::new(vec![("x".to_string(), ScalarType::U32)])),
    }]];
    plan
}

fn scan_relation_into_output_plan(rel: RelId, output_predicate: &str) -> ExecutionPlan {
    let mut plan = ExecutionPlan::new(vec![Scc {
        id: 0,
        predicates: vec![output_predicate.to_string()],
        is_recursive: false,
    }])
    .with_strata(vec![Stratum {
        id: 0,
        sccs: vec![0],
    }]);
    plan.rules_by_scc = vec![vec![CompiledRule {
        head: output_predicate.to_string(),
        body: RirNode::Scan { rel },
        meta: RirMeta::with_schema(Schema::new(vec![("x".to_string(), ScalarType::U32)])),
    }]];
    plan
}

fn upload_unary_u32(provider: &Arc<CudaKernelProvider>, rows: &[u32], name: &str) -> CudaBuffer {
    upload_unary_typed_u32(provider, rows, name, ScalarType::U32)
}

fn upload_unary_typed_u32(
    provider: &Arc<CudaKernelProvider>,
    rows: &[u32],
    name: &str,
    column_type: ScalarType,
) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = std::mem::size_of_val(rows);
    let memory = provider.memory();
    let mut col = memory.alloc::<u8>(bytes_per_col).expect("alloc unary col");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let bytes: Vec<u8> = rows.iter().flat_map(|value| value.to_le_bytes()).collect();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&bytes, &mut col)
        .expect("upload unary col");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("upload unary row count");

    CudaBuffer::from_columns_with_host_count(
        vec![col.into()],
        u64::from(n),
        d_num_rows,
        Schema::new(vec![(name.to_string(), column_type)]),
        n,
    )
}

fn upload_binary_u32(
    provider: &Arc<CudaKernelProvider>,
    rows: &[(u32, u32)],
    name_a: &str,
    name_b: &str,
) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = rows.len() * std::mem::size_of::<u32>();
    let memory = provider.memory();
    let mut col0 = memory
        .alloc::<u8>(bytes_per_col)
        .expect("alloc binary col0");
    let mut col1 = memory
        .alloc::<u8>(bytes_per_col)
        .expect("alloc binary col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let bytes0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let bytes1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&bytes0, &mut col0)
        .expect("upload binary col0");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&bytes1, &mut col1)
        .expect("upload binary col1");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("upload binary row count");

    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        u64::from(n),
        d_num_rows,
        Schema::new(vec![
            (name_a.to_string(), ScalarType::U32),
            (name_b.to_string(), ScalarType::U32),
        ]),
        n,
    )
}

fn upload_quaternary_u32(
    provider: &Arc<CudaKernelProvider>,
    rows: &[(u32, u32, u32, u32)],
    name_a: &str,
    name_b: &str,
    name_c: &str,
    name_d: &str,
) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = rows.len() * std::mem::size_of::<u32>();
    let memory = provider.memory();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut col2 = memory.alloc::<u8>(bytes_per_col).expect("alloc col2");
    let mut col3 = memory.alloc::<u8>(bytes_per_col).expect("alloc col3");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let bytes0: Vec<u8> = rows
        .iter()
        .flat_map(|(a, _, _, _)| a.to_le_bytes())
        .collect();
    let bytes1: Vec<u8> = rows
        .iter()
        .flat_map(|(_, b, _, _)| b.to_le_bytes())
        .collect();
    let bytes2: Vec<u8> = rows
        .iter()
        .flat_map(|(_, _, c, _)| c.to_le_bytes())
        .collect();
    let bytes3: Vec<u8> = rows
        .iter()
        .flat_map(|(_, _, _, d)| d.to_le_bytes())
        .collect();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&bytes0, &mut col0)
        .expect("upload col0");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&bytes1, &mut col1)
        .expect("upload col1");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&bytes2, &mut col2)
        .expect("upload col2");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&bytes3, &mut col3)
        .expect("upload col3");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("upload quaternary row count");

    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into(), col2.into(), col3.into()],
        u64::from(n),
        d_num_rows,
        Schema::new(vec![
            (name_a.to_string(), ScalarType::U32),
            (name_b.to_string(), ScalarType::U32),
            (name_c.to_string(), ScalarType::U32),
            (name_d.to_string(), ScalarType::U32),
        ]),
        n,
    )
}

fn upload_u32(
    provider: &Arc<CudaKernelProvider>,
    value: u32,
) -> xlog_cuda::memory::TrackedCudaSlice<u32> {
    let mut slot = provider.memory().alloc::<u32>(1).expect("alloc u32 scalar");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[value], &mut slot)
        .expect("upload u32 scalar");
    slot
}
