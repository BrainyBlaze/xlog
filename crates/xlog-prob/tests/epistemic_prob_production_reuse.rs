use std::fs;
use std::path::PathBuf;

use xlog_prob::epistemic_production::{
    production_capabilities, EpistemicProbProductionCapabilityStatus, EpistemicProbProductionTrace,
};

#[test]
fn production_prob_adapter_reuses_gpu_exact_path_not_fixture_circuit() {
    let lib = include_str!("../src/lib.rs");
    let mut production_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    production_path.push("src");
    production_path.push("epistemic_production.rs");
    let production = fs::read_to_string(&production_path).unwrap_or_default();

    assert!(lib.contains("epistemic_production"));
    assert!(production.contains("EpistemicProbProductionAdapter"));
    assert!(production.contains("EpistemicProbProductionTrace"));
    assert!(production.contains("compile_source_with_gpu_execution_result"));
    assert!(production.contains("compile_program_with_gpu_execution_result"));
    assert!(production.contains("compile_and_evaluate_source_with_gpu_execution_result"));
    assert!(production.contains("EpistemicProbGpuExecutionEvidence"));
    assert!(production.contains("EpistemicProbGpuBatchExecutionEvidence"));
    assert!(production.contains("compile_and_evaluate_source_for_gpu_execution_results"));
    assert!(production.contains("compile_and_evaluate_source_for_gpu_batch_execution_result"));
    assert!(production
        .contains("compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result"));
    assert!(production
        .contains("compile_and_evaluate_conditioned_program_for_gpu_batch_execution_result"));
    assert!(production.contains("compile_and_evaluate_program_for_gpu_execution_results"));
    assert!(production.contains("compile_and_evaluate_program_for_gpu_batch_execution_result"));
    assert!(
        production.contains("compile_and_evaluate_conditioned_source_for_gpu_execution_results")
    );
    assert!(
        production.contains("compile_and_evaluate_conditioned_program_for_gpu_execution_results")
    );
    assert!(
        production.contains("compile_and_evaluate_conditioned_source_with_gpu_execution_result")
    );
    assert!(
        production.contains("compile_and_evaluate_conditioned_program_with_gpu_execution_result")
    );
    assert!(production
        .contains("compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result"));
    assert!(production
        .contains("compile_and_evaluate_conditioned_source_with_grads_for_gpu_execution_results"));
    assert!(production.contains(
        "compile_and_evaluate_conditioned_source_with_grads_for_gpu_batch_execution_result"
    ));
    assert!(production
        .contains("compile_and_evaluate_conditioned_program_with_grads_with_gpu_execution_result"));
    assert!(production
        .contains("compile_and_evaluate_conditioned_program_with_grads_for_gpu_execution_results"));
    assert!(production.contains(
        "compile_and_evaluate_conditioned_program_with_grads_for_gpu_batch_execution_result"
    ));
    assert!(production.contains("compile_and_evaluate_program_with_gpu_execution_result"));
    assert!(production.contains("encode_source_pir_cnf_with_gpu_execution_result"));
    assert!(production.contains("encode_program_pir_cnf_with_gpu_execution_result"));
    assert!(production.contains("encode_source_pir_cnf_for_gpu_execution_results"));
    assert!(production.contains("encode_program_pir_cnf_for_gpu_execution_results"));
    assert!(production.contains("evaluate_with_gpu_execution_result"));
    assert!(production.contains("evaluate_gpu_with_grads_with_gpu_execution_result"));
    assert!(production.contains("evaluate_for_gpu_execution_results"));
    assert!(production.contains("evaluate_gpu_with_grads_for_gpu_execution_results"));
    assert!(production.contains("apply_accepted_world_view_to_circuit_with_gpu_execution_result"));
    assert!(production.contains("accepted_incremental_circuit_updates"));
    assert!(production.contains("from_gpu_execution_result"));
    assert!(production.contains("ExactDdnnfProgram::compile_source_with_gpu"));
    assert!(production.contains("ExactDdnnfProgram::compile_from_program"));
    assert!(production.contains("GpuPirGraph::from_host"));
    assert!(production.contains("encode_cnf_gpu"));
    assert!(production.contains("evaluate_gpu_with_grads"));
    assert!(production.contains("gpu_pir_graph_uploads"));
    assert!(production.contains("gpu_source_pir_graph_uploads"));
    assert!(production.contains("gpu_program_pir_graph_uploads"));
    assert!(production.contains("gpu_cnf_encodes"));
    assert!(production.contains("gpu_source_cnf_encodes"));
    assert!(production.contains("gpu_program_cnf_encodes"));
    assert!(production.contains("gpu_knowledge_compilation_end_to_end_runs"));
    assert!(production.contains("gpu_source_knowledge_compilation_end_to_end_runs"));
    assert!(production.contains("gpu_program_knowledge_compilation_end_to_end_runs"));
    assert!(production.contains("accepted_evidence_assumptions_consumed"));
    assert!(production.contains("accepted_gpu_batch_evidence_consumed"));
    assert!(production.contains("accepted_gpu_batch_component_evidence_consumed"));
    assert!(production.contains("gpu_conditioned_evidence_facts"));
    assert!(production.contains("gpu_conditioned_negative_evidence_facts"));
    assert!(production.contains("gpu_source_conditioned_evidence_facts"));
    assert!(production.contains("gpu_program_conditioned_evidence_facts"));
    assert!(production.contains("gpu_source_conditioned_negative_evidence_facts"));
    assert!(production.contains("gpu_program_conditioned_negative_evidence_facts"));
    assert!(production.contains("gpu_source_conditioned_gradient_evaluations"));
    assert!(production.contains("gpu_program_conditioned_gradient_evaluations"));
    assert!(production.contains("gpu_conditioned_know_evidence_facts"));
    assert!(production.contains("gpu_conditioned_possible_evidence_facts"));
    assert!(production.contains("gpu_conditioned_not_known_evidence_facts"));
    assert!(production.contains("gpu_conditioned_not_possible_evidence_facts"));
    assert!(production.contains("gpu_source_conditioned_know_evidence_facts"));
    assert!(production.contains("gpu_source_conditioned_possible_evidence_facts"));
    assert!(production.contains("gpu_source_conditioned_not_known_evidence_facts"));
    assert!(production.contains("gpu_source_conditioned_not_possible_evidence_facts"));
    assert!(production.contains("gpu_program_conditioned_know_evidence_facts"));
    assert!(production.contains("gpu_program_conditioned_possible_evidence_facts"));
    assert!(production.contains("gpu_program_conditioned_not_known_evidence_facts"));
    assert!(production.contains("gpu_program_conditioned_not_possible_evidence_facts"));
    assert!(production.contains("record_conditioned_evidence_counts"));
    assert!(production.contains("condition_source_with_accepted_evidence"));
    assert!(production.contains("condition_program_with_accepted_evidence"));
    assert!(production.contains("EpistemicEvidenceTerm"));
    assert!(production.contains("evidence_term_to_ast_term"));
    assert!(production.contains("program.evidence.push"));
    assert!(production.contains("gpu_exact_query_evaluations"));
    assert!(production.contains("gpu_source_exact_query_evaluations"));
    assert!(production.contains("gpu_program_exact_query_evaluations"));
    assert!(production.contains("gpu_exact_gradient_evaluations"));
    assert!(production.contains("cpu_only_probability_recomputations: 0"));
    assert!(production.contains("fixture_circuit_evaluations: 0"));
    assert!(!production.contains("EpistemicCircuit::compile"));
    assert!(!production.contains("conditional_probability_from_logs"));
    assert!(!production.contains("query_probability"));
}

#[test]
fn production_prob_capabilities_disallow_fixture_circuit_metrics() {
    let capabilities = production_capabilities();

    assert_eq!(
        capabilities.gpu_exact_provenance,
        EpistemicProbProductionCapabilityStatus::Available
    );
    assert_eq!(
        capabilities.gpu_pir_cnf,
        EpistemicProbProductionCapabilityStatus::Available
    );
    assert_eq!(
        capabilities.gpu_knowledge_compilation,
        EpistemicProbProductionCapabilityStatus::Available
    );
    assert_eq!(
        capabilities.gpu_exact_query_and_gradient,
        EpistemicProbProductionCapabilityStatus::Available
    );
    assert!(!capabilities.fixture_circuit_allowed);
    assert_eq!(capabilities.gpu_knowledge_compilation_blocker, "");
}

#[test]
fn production_prob_batch_paths_use_single_gpu_batch_gate() {
    let mut production_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    production_path.push("src");
    production_path.push("epistemic_production.rs");
    let production = fs::read_to_string(&production_path).unwrap_or_default();

    let manual_batch_guard_count = production
        .matches("batch_trace.component_count != evidence.batch.results.len()")
        .count();
    assert_eq!(
        manual_batch_guard_count, 1,
        "probabilistic split-batch paths must share the central accepted GPU batch gate"
    );
}

#[test]
fn production_prob_metric_gate_rejects_fixture_only_traces() {
    let empty = EpistemicProbProductionTrace::default();
    let err = empty
        .require_production_metric_eligibility()
        .expect_err("empty probability trace must not satisfy production metrics");
    assert!(format!("{err}").contains("accepted world-view evidence"));

    let eligible = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        gpu_pir_graph_uploads: 1,
        gpu_cnf_encodes: 1,
        ..EpistemicProbProductionTrace::default()
    };
    assert!(eligible.require_production_metric_eligibility().is_ok());

    let conditioned_only = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        gpu_conditioned_evidence_facts: 1,
        gpu_conditioned_negative_evidence_facts: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = conditioned_only
        .require_production_metric_eligibility()
        .expect_err("conditioned evidence facts alone must not satisfy production metrics");
    assert!(format!("{err}")
        .contains("existing GPU exact/provenance/PIR/CNF/knowledge-compilation counter"));

    let conditioned_negative = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        gpu_conditioned_evidence_facts: 1,
        gpu_conditioned_negative_evidence_facts: 1,
        gpu_source_knowledge_compilation_end_to_end_runs: 1,
        ..EpistemicProbProductionTrace::default()
    };
    assert!(conditioned_negative
        .require_production_metric_eligibility()
        .is_ok());

    let incremental_fixture_only = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_incremental_circuit_updates: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = incremental_fixture_only
        .require_production_metric_eligibility()
        .expect_err(
            "incremental fixture circuit updates alone must not satisfy production metrics",
        );
    assert!(format!("{err}")
        .contains("existing GPU exact/provenance/PIR/CNF/knowledge-compilation counter"));

    let fixture = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        gpu_pir_graph_uploads: 1,
        gpu_cnf_encodes: 1,
        fixture_circuit_evaluations: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = fixture
        .require_production_metric_eligibility()
        .expect_err("fixture circuit trace must not satisfy production metrics");
    assert!(format!("{err}").contains("CPU probabilistic fallback counters must be zero"));
}
