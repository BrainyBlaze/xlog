use xlog_solve::{
    production_capabilities, GpuSolverProductionCapabilityStatus, GpuSolverProductionTrace,
};

#[test]
fn production_solver_capabilities_are_gpu_backed_and_cpu_oracle_is_not_metric_eligible() {
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

#[test]
fn production_solver_metric_gate_requires_gpu_work_inside_accepted_evidence_gate() {
    let missing_accepted_work = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        gpu_cdcl_sat_solves: 1,
        ..GpuSolverProductionTrace::default()
    };
    let err = missing_accepted_work
        .require_production_metric_eligibility()
        .expect_err("GPU work outside the accepted evidence gate must not satisfy metrics");
    assert!(format!("{err}").contains("inside an accepted epistemic evidence gate"));

    let impossible_accepted_work = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        accepted_gpu_solver_production_path_events: 2,
        gpu_cdcl_sat_solves: 1,
        ..GpuSolverProductionTrace::default()
    };
    let err = impossible_accepted_work
        .require_production_metric_eligibility()
        .expect_err("accepted GPU work cannot exceed total production work");
    assert!(format!("{err}").contains("cannot exceed total GPU solver production/status events"));
}

#[test]
fn production_solver_metric_gate_requires_consistent_batch_component_accounting() {
    let uncovered_batch = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 2,
        accepted_gpu_candidate_state_transitions: 2,
        accepted_gpu_world_view_state_transitions: 2,
        accepted_gpu_candidate_final_output_rows_consumed: 2,
        accepted_gpu_batch_candidate_evidence_consumed: 2,
        accepted_gpu_batch_candidate_component_evidence_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 2,
        accepted_solver_assumption_bindings_consumed: 2,
        accepted_solver_required_capabilities_consumed: 10,
        accepted_solver_required_statuses_consumed: 8,
        accepted_gpu_solver_production_path_events: 2,
        gpu_cdcl_sat_solves: 2,
        ..GpuSolverProductionTrace::default()
    };
    let err = uncovered_batch
        .require_production_metric_eligibility()
        .expect_err("batch evidence must account for every component");
    assert!(format!("{err}").contains("component evidence must cover accepted batch evidence"));

    let impossible_components = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_gpu_batch_candidate_evidence_consumed: 1,
        accepted_gpu_batch_candidate_component_evidence_consumed: 2,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        accepted_gpu_solver_production_path_events: 1,
        gpu_cdcl_sat_solves: 1,
        ..GpuSolverProductionTrace::default()
    };
    let err = impossible_components
        .require_production_metric_eligibility()
        .expect_err("batch components cannot exceed accepted candidate evidence");
    assert!(format!("{err}").contains("component evidence cannot exceed accepted candidate"));
}

#[test]
fn production_solver_metric_gate_rejects_impossible_operator_accounting() {
    let all_operator_evidence = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_know_gpu_candidate_evidence_consumed: 1,
        accepted_possible_gpu_candidate_evidence_consumed: 1,
        accepted_not_possible_gpu_candidate_evidence_consumed: 1,
        accepted_not_know_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        accepted_gpu_solver_production_path_events: 1,
        gpu_cdcl_sat_solves: 1,
        ..GpuSolverProductionTrace::default()
    };
    assert!(all_operator_evidence
        .require_production_metric_eligibility()
        .is_ok());

    let impossible_know_count = GpuSolverProductionTrace {
        accepted_know_gpu_candidate_evidence_consumed: 2,
        ..all_operator_evidence
    };
    let err = impossible_know_count
        .require_production_metric_eligibility()
        .expect_err("operator counters cannot exceed accepted evidence records");
    assert!(format!("{err}").contains("operator evidence counters cannot exceed"));

    let impossible_possible_count = GpuSolverProductionTrace {
        accepted_possible_gpu_candidate_evidence_consumed: 2,
        ..all_operator_evidence
    };
    let err = impossible_possible_count
        .require_production_metric_eligibility()
        .expect_err("possible operator counters cannot exceed accepted evidence records");
    assert!(format!("{err}").contains("operator evidence counters cannot exceed"));

    let impossible_not_possible_count = GpuSolverProductionTrace {
        accepted_not_possible_gpu_candidate_evidence_consumed: 2,
        ..all_operator_evidence
    };
    let err = impossible_not_possible_count
        .require_production_metric_eligibility()
        .expect_err("not-possible operator counters cannot exceed accepted evidence records");
    assert!(format!("{err}").contains("operator evidence counters cannot exceed"));

    let impossible_not_know_count = GpuSolverProductionTrace {
        accepted_not_know_gpu_candidate_evidence_consumed: 2,
        ..all_operator_evidence
    };
    let err = impossible_not_know_count
        .require_production_metric_eligibility()
        .expect_err("not-know operator counters cannot exceed accepted evidence records");
    assert!(format!("{err}").contains("operator evidence counters cannot exceed"));
}

#[test]
fn production_solver_metric_gate_rejects_cpu_oracle_only_traces() {
    let empty = GpuSolverProductionTrace::default();
    let err = empty
        .require_production_metric_eligibility()
        .expect_err("empty solver trace must not satisfy production metrics");
    assert!(format!("{err}").contains("accepted GPU candidate evidence"));

    let eligible = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        accepted_gpu_solver_production_path_events: 1,
        gpu_cdcl_sat_solves: 1,
        ..GpuSolverProductionTrace::default()
    };
    assert!(eligible.require_production_metric_eligibility().is_ok());

    let encoded_without_optimum = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        accepted_gpu_solver_production_path_events: 2,
        gpu_maxsat_candidate_encodes: 1,
        gpu_maxsat_candidate_solves: 1,
        gpu_maxsat_frontier_certified_candidate_solves: 1,
        gpu_maxsat_frontier_upper_bound_certificates: 1,
        ..GpuSolverProductionTrace::default()
    };
    let err = encoded_without_optimum
        .require_production_metric_eligibility()
        .expect_err("encoded MaxSAT metrics require a GPU-certified optimum");
    assert!(format!("{err}").contains("GPU-certified optimum"));

    let encoded_maxsat = GpuSolverProductionTrace {
        gpu_maxsat_optima: 1,
        ..encoded_without_optimum
    };
    assert!(encoded_maxsat
        .require_production_metric_eligibility()
        .is_ok());

    let candidate_maxsat = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        accepted_gpu_solver_production_path_events: 1,
        gpu_maxsat_candidate_solves: 1,
        gpu_maxsat_optima: 1,
        ..GpuSolverProductionTrace::default()
    };
    let err = candidate_maxsat
        .require_production_metric_eligibility()
        .expect_err("uncertified MaxSAT candidates must not satisfy production metrics");
    assert!(format!("{err}").contains("upper-bound certificate"));

    let encoded_without_certificate = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        accepted_gpu_solver_production_path_events: 2,
        gpu_maxsat_candidate_encodes: 1,
        gpu_maxsat_candidate_solves: 1,
        gpu_maxsat_frontier_certified_candidate_solves: 1,
        ..GpuSolverProductionTrace::default()
    };
    let err = encoded_without_certificate
        .require_production_metric_eligibility()
        .expect_err("encoded MaxSAT metrics must still require a certificate");
    assert!(format!("{err}").contains("upper-bound certificate"));

    let status_only = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        accepted_gpu_solver_production_path_events: 2,
        gpu_lifecycle_unknown_status_steps: 1,
        gpu_lifecycle_timeout_status_steps: 1,
        ..GpuSolverProductionTrace::default()
    };
    assert!(status_only.require_production_metric_eligibility().is_ok());

    let batch_eligible = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 2,
        accepted_gpu_candidate_state_transitions: 2,
        accepted_gpu_world_view_state_transitions: 2,
        accepted_gpu_candidate_final_output_rows_consumed: 2,
        accepted_gpu_batch_candidate_evidence_consumed: 1,
        accepted_gpu_batch_candidate_component_evidence_consumed: 2,
        accepted_g91_gpu_candidate_evidence_consumed: 2,
        accepted_solver_assumption_bindings_consumed: 2,
        accepted_solver_required_capabilities_consumed: 10,
        accepted_solver_required_statuses_consumed: 8,
        accepted_gpu_solver_production_path_events: 2,
        gpu_cdcl_sat_solves: 2,
        ..GpuSolverProductionTrace::default()
    };
    assert!(batch_eligible
        .require_production_metric_eligibility()
        .is_ok());

    let cpu_fallback = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        accepted_gpu_solver_production_path_events: 1,
        gpu_cdcl_sat_solves: 1,
        cpu_assignment_enumerations: 1,
        ..GpuSolverProductionTrace::default()
    };
    let err = cpu_fallback
        .require_production_metric_eligibility()
        .expect_err("CPU search trace must not satisfy production metrics");
    assert!(format!("{err}").contains("CPU solver search counters must be zero"));

    let cpu_maxsat_fallback = GpuSolverProductionTrace {
        cpu_assignment_enumerations: 0,
        cpu_maxsat_enumerations: 1,
        ..cpu_fallback
    };
    let err = cpu_maxsat_fallback
        .require_production_metric_eligibility()
        .expect_err("CPU MaxSAT search trace must not satisfy production metrics");
    assert!(format!("{err}").contains("CPU solver search counters must be zero"));

    let cpu_learned_clause_transfer = GpuSolverProductionTrace {
        cpu_assignment_enumerations: 0,
        cpu_learned_clause_transfers: 1,
        ..cpu_fallback
    };
    let err = cpu_learned_clause_transfer
        .require_production_metric_eligibility()
        .expect_err("CPU learned-clause transfer trace must not satisfy production metrics");
    assert!(format!("{err}").contains("CPU learned-clause transfers must be zero"));
}

#[test]
fn production_solver_metric_gate_rejects_impossible_maxsat_accounting() {
    let eligible_maxsat = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        accepted_gpu_solver_production_path_events: 1,
        gpu_maxsat_candidate_encodes: 1,
        gpu_maxsat_candidate_solves: 1,
        gpu_maxsat_frontier_certified_candidate_solves: 1,
        gpu_maxsat_frontier_upper_bound_certificates: 1,
        gpu_maxsat_optima: 1,
        ..GpuSolverProductionTrace::default()
    };
    assert!(eligible_maxsat
        .require_production_metric_eligibility()
        .is_ok());

    let prunes_exceed_solves = GpuSolverProductionTrace {
        gpu_maxsat_unsat_candidate_prunes: 2,
        ..eligible_maxsat
    };
    let err = prunes_exceed_solves
        .require_production_metric_eligibility()
        .expect_err("MaxSAT prunes cannot exceed solved candidates");
    assert!(format!("{err}").contains("UNSAT candidate prunes cannot exceed"));

    let optima_exceed_solves = GpuSolverProductionTrace {
        gpu_maxsat_optima: 2,
        ..eligible_maxsat
    };
    let err = optima_exceed_solves
        .require_production_metric_eligibility()
        .expect_err("MaxSAT optima cannot exceed solved candidates");
    assert!(format!("{err}").contains("MaxSAT optima cannot exceed"));

    let encodes_exceed_solves = GpuSolverProductionTrace {
        accepted_gpu_solver_production_path_events: 2,
        gpu_maxsat_candidate_encodes: 2,
        gpu_maxsat_frontier_certified_candidate_solves: 1,
        gpu_maxsat_frontier_upper_bound_certificates: 1,
        ..eligible_maxsat
    };
    let err = encodes_exceed_solves
        .require_production_metric_eligibility()
        .expect_err("encoded MaxSAT candidates cannot exceed solved candidates");
    assert!(format!("{err}").contains("encoded MaxSAT candidates cannot exceed"));

    let data_plane_bytes_without_calls = GpuSolverProductionTrace {
        gpu_maxsat_candidate_cnf_data_plane_htod_bytes: 64,
        ..eligible_maxsat
    };
    let err = data_plane_bytes_without_calls
        .require_production_metric_eligibility()
        .expect_err("encoded MaxSAT data-plane H2D bytes require matching H2D calls");
    assert!(format!("{err}").contains("CNF upload bytes require matching H2D calls"));

    let launch_metadata_bytes_without_calls = GpuSolverProductionTrace {
        gpu_maxsat_candidate_cnf_launch_metadata_htod_bytes: 64,
        ..eligible_maxsat
    };
    let err = launch_metadata_bytes_without_calls
        .require_production_metric_eligibility()
        .expect_err("encoded MaxSAT launch-metadata H2D bytes require matching H2D calls");
    assert!(format!("{err}").contains("CNF upload bytes require matching H2D calls"));

    let data_plane_calls_without_bytes = GpuSolverProductionTrace {
        gpu_maxsat_candidate_cnf_data_plane_htod_calls: 1,
        ..eligible_maxsat
    };
    let err = data_plane_calls_without_bytes
        .require_production_metric_eligibility()
        .expect_err("encoded MaxSAT data-plane H2D calls require matching H2D bytes");
    assert!(format!("{err}").contains("CNF upload calls require matching H2D bytes"));

    let launch_metadata_calls_without_bytes = GpuSolverProductionTrace {
        gpu_maxsat_candidate_cnf_launch_metadata_htod_calls: 1,
        ..eligible_maxsat
    };
    let err = launch_metadata_calls_without_bytes
        .require_production_metric_eligibility()
        .expect_err("encoded MaxSAT launch-metadata H2D calls require matching H2D bytes");
    assert!(format!("{err}").contains("CNF upload calls require matching H2D bytes"));
}

#[test]
fn production_solver_metric_gate_requires_consistent_maxsat_scheduler_job_accounting() {
    let aggregate_without_kind = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        accepted_gpu_solver_production_path_events: 1,
        gpu_cdcl_sat_solves: 1,
        gpu_maxsat_scheduler_jobs: 1,
        ..GpuSolverProductionTrace::default()
    };
    let err = aggregate_without_kind
        .require_production_metric_eligibility()
        .expect_err("MaxSAT scheduler jobs must be classified by job kind or status");
    assert!(format!("{err}").contains("MaxSAT scheduler job accounting must match"));

    let kind_without_aggregate = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        accepted_gpu_solver_production_path_events: 1,
        gpu_maxsat_scheduler_unknown_status_jobs: 1,
        ..GpuSolverProductionTrace::default()
    };
    let err = kind_without_aggregate
        .require_production_metric_eligibility()
        .expect_err("MaxSAT scheduler job kinds require aggregate scheduler accounting");
    assert!(format!("{err}").contains("MaxSAT scheduler job accounting must match"));
}

#[test]
fn production_solver_metric_gate_requires_consistent_portfolio_job_accounting() {
    let aggregate_without_kind = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        accepted_gpu_solver_production_path_events: 1,
        gpu_cdcl_sat_solves: 1,
        gpu_portfolio_jobs: 1,
        ..GpuSolverProductionTrace::default()
    };
    let err = aggregate_without_kind
        .require_production_metric_eligibility()
        .expect_err("portfolio jobs must be classified by job kind");
    assert!(format!("{err}").contains("portfolio job accounting must match"));

    let kind_without_aggregate = GpuSolverProductionTrace {
        accepted_gpu_candidate_evidence_consumed: 1,
        accepted_gpu_candidate_state_transitions: 1,
        accepted_gpu_world_view_state_transitions: 1,
        accepted_gpu_candidate_final_output_rows_consumed: 1,
        accepted_g91_gpu_candidate_evidence_consumed: 1,
        accepted_solver_assumption_bindings_consumed: 1,
        accepted_solver_required_capabilities_consumed: 5,
        accepted_solver_required_statuses_consumed: 4,
        accepted_gpu_solver_production_path_events: 2,
        gpu_cdcl_sat_solves: 1,
        gpu_portfolio_sat_jobs: 1,
        ..GpuSolverProductionTrace::default()
    };
    let err = kind_without_aggregate
        .require_production_metric_eligibility()
        .expect_err("portfolio job kinds require aggregate portfolio accounting");
    assert!(format!("{err}").contains("portfolio job accounting must match"));
}
