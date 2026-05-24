use xlog_prob::epistemic_production::{
    production_capabilities, EpistemicProbProductionCapabilityStatus, EpistemicProbProductionTrace,
};

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
fn production_prob_metric_gate_requires_gpu_work_inside_accepted_evidence_gate() {
    let outside_gate = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        gpu_pir_graph_uploads: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = outside_gate
        .require_production_metric_eligibility()
        .expect_err("GPU probability work outside accepted evidence gate is not eligible");
    assert!(format!("{err}").contains("inside an accepted world-view evidence gate"));

    let over_counted = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        gpu_pir_graph_uploads: 1,
        accepted_gpu_production_path_events: 2,
        ..EpistemicProbProductionTrace::default()
    };
    let err = over_counted
        .require_production_metric_eligibility()
        .expect_err("accepted GPU production events cannot exceed total GPU events");
    assert!(format!("{err}").contains("cannot exceed total GPU production events"));

    let under_counted = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 2,
        accepted_faeel_world_view_evidence_consumed: 2,
        gpu_pir_graph_uploads: 2,
        accepted_gpu_production_path_events: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = under_counted
        .require_production_metric_eligibility()
        .expect_err("every accepted world-view evidence record needs covered GPU production work");
    assert!(format!("{err}").contains("must cover each accepted"));
}

#[test]
fn production_prob_metric_gate_requires_mode_and_batch_component_accounting() {
    let unclassified_mode = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        gpu_pir_graph_uploads: 1,
        accepted_gpu_production_path_events: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = unclassified_mode
        .require_production_metric_eligibility()
        .expect_err("accepted GPU evidence must be classified by epistemic mode");
    assert!(format!("{err}").contains("classified by epistemic mode"));

    let missing_batch_components = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        accepted_gpu_batch_evidence_consumed: 2,
        accepted_gpu_batch_component_evidence_consumed: 1,
        gpu_pir_graph_uploads: 1,
        accepted_gpu_production_path_events: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = missing_batch_components
        .require_production_metric_eligibility()
        .expect_err("accepted batch evidence must have component evidence");
    assert!(format!("{err}").contains("must cover accepted batch evidence"));

    let excess_batch_components = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        accepted_gpu_batch_evidence_consumed: 1,
        accepted_gpu_batch_component_evidence_consumed: 2,
        gpu_pir_graph_uploads: 1,
        accepted_gpu_production_path_events: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = excess_batch_components
        .require_production_metric_eligibility()
        .expect_err("batch components cannot exceed accepted world-view evidence");
    assert!(format!("{err}").contains("cannot exceed accepted world-view evidence"));
}

#[test]
fn production_prob_metric_gate_requires_classified_gpu_path_accounting() {
    let aggregate_without_path = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        gpu_exact_query_evaluations: 1,
        accepted_gpu_production_path_events: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = aggregate_without_path
        .require_production_metric_eligibility()
        .expect_err("GPU exact query metrics must be classified by source/program path");
    assert!(format!("{err}").contains("GPU production path accounting must match"));

    let path_without_aggregate = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        gpu_source_exact_query_evaluations: 1,
        accepted_gpu_production_path_events: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = path_without_aggregate
        .require_production_metric_eligibility()
        .expect_err("source/program GPU exact query metrics require aggregate accounting");
    assert!(format!("{err}").contains("GPU production path accounting must match"));
}

#[test]
fn production_prob_metric_gate_requires_nonzero_tuple_membership_evidence() {
    let missing_tuple_reads = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        accepted_gpu_nonzero_arity_evidence_assumptions_consumed: 1,
        accepted_gpu_max_evidence_arity_consumed: 2,
        gpu_pir_graph_uploads: 1,
        accepted_gpu_production_path_events: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = missing_tuple_reads
        .require_production_metric_eligibility()
        .expect_err("nonzero-arity GPU evidence requires tuple-key device reads");
    assert!(format!("{err}").contains("tuple-key device column reads"));

    let tuple_reads_without_nonzero = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        accepted_gpu_tuple_key_column_reads_consumed: 2,
        gpu_pir_graph_uploads: 1,
        gpu_source_pir_graph_uploads: 1,
        gpu_cnf_encodes: 1,
        gpu_source_cnf_encodes: 1,
        accepted_gpu_production_path_events: 4,
        ..EpistemicProbProductionTrace::default()
    };
    let err = tuple_reads_without_nonzero
        .require_production_metric_eligibility()
        .expect_err("tuple-key reads require accepted nonzero-arity GPU evidence");
    assert!(format!("{err}").contains("tuple-key reads require accepted nonzero-arity"));

    let nonzero_without_max_arity = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        accepted_gpu_nonzero_arity_evidence_assumptions_consumed: 1,
        accepted_gpu_tuple_key_column_reads_consumed: 2,
        accepted_gpu_final_tuple_row_filters_consumed: 1,
        accepted_gpu_row_specific_membership_row_capacity_consumed: 1,
        gpu_pir_graph_uploads: 1,
        gpu_source_pir_graph_uploads: 1,
        gpu_cnf_encodes: 1,
        gpu_source_cnf_encodes: 1,
        accepted_gpu_production_path_events: 4,
        ..EpistemicProbProductionTrace::default()
    };
    let err = nonzero_without_max_arity
        .require_production_metric_eligibility()
        .expect_err("accepted nonzero-arity GPU evidence requires max arity");
    assert!(format!("{err}").contains("requires accepted max evidence arity"));

    let negated_filters_without_total = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        accepted_gpu_nonzero_arity_evidence_assumptions_consumed: 1,
        accepted_gpu_max_evidence_arity_consumed: 2,
        accepted_gpu_tuple_key_column_reads_consumed: 2,
        accepted_gpu_final_tuple_negated_row_filters_consumed: 1,
        gpu_pir_graph_uploads: 1,
        accepted_gpu_production_path_events: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = negated_filters_without_total
        .require_production_metric_eligibility()
        .expect_err("negated GPU row filters cannot exceed total GPU row filters");
    assert!(format!("{err}").contains("negated final-tuple row filters cannot exceed"));

    let eligible = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        accepted_gpu_nonzero_arity_evidence_assumptions_consumed: 1,
        accepted_gpu_max_evidence_arity_consumed: 2,
        accepted_gpu_tuple_key_column_reads_consumed: 2,
        accepted_gpu_final_tuple_row_filters_consumed: 1,
        accepted_gpu_row_specific_membership_row_capacity_consumed: 1,
        gpu_pir_graph_uploads: 1,
        gpu_source_pir_graph_uploads: 1,
        gpu_cnf_encodes: 1,
        gpu_source_cnf_encodes: 1,
        accepted_gpu_production_path_events: 4,
        ..EpistemicProbProductionTrace::default()
    };
    assert!(eligible.require_production_metric_eligibility().is_ok());
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
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        gpu_pir_graph_uploads: 1,
        gpu_source_pir_graph_uploads: 1,
        gpu_cnf_encodes: 1,
        gpu_source_cnf_encodes: 1,
        accepted_gpu_production_path_events: 4,
        ..EpistemicProbProductionTrace::default()
    };
    assert!(eligible.require_production_metric_eligibility().is_ok());

    let missing_assumptions = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        gpu_pir_graph_uploads: 1,
        accepted_gpu_production_path_events: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = missing_assumptions
        .require_production_metric_eligibility()
        .expect_err("accepted GPU world-view evidence requires accepted assumptions");
    assert!(format!("{err}").contains("at least one accepted epistemic assumption"));

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
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        accepted_conditioned_world_view_evidence_consumed: 1,
        accepted_source_conditioned_world_view_evidence_consumed: 1,
        gpu_conditioned_evidence_facts: 1,
        gpu_conditioned_negative_evidence_facts: 1,
        gpu_conditioned_not_known_evidence_facts: 1,
        gpu_source_conditioned_evidence_facts: 1,
        gpu_source_conditioned_negative_evidence_facts: 1,
        gpu_source_conditioned_not_known_evidence_facts: 1,
        gpu_knowledge_compilation_end_to_end_runs: 1,
        gpu_source_knowledge_compilation_end_to_end_runs: 1,
        accepted_gpu_production_path_events: 2,
        ..EpistemicProbProductionTrace::default()
    };
    assert!(conditioned_negative
        .require_production_metric_eligibility()
        .is_ok());

    let under_counted_conditioned_facts = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 2,
        accepted_faeel_world_view_evidence_consumed: 2,
        accepted_evidence_assumptions_consumed: 2,
        accepted_conditioned_world_view_evidence_consumed: 2,
        accepted_source_conditioned_world_view_evidence_consumed: 2,
        gpu_conditioned_evidence_facts: 1,
        gpu_conditioned_know_evidence_facts: 1,
        gpu_source_conditioned_evidence_facts: 1,
        gpu_source_conditioned_know_evidence_facts: 1,
        gpu_source_knowledge_compilation_end_to_end_runs: 2,
        accepted_gpu_production_path_events: 2,
        ..EpistemicProbProductionTrace::default()
    };
    let err = under_counted_conditioned_facts
        .require_conditioned_evidence_metric_eligibility()
        .expect_err("conditioned evidence facts must cover conditioned evidence records");
    assert!(format!("{err}").contains("facts must cover each accepted conditioned"));

    let facts_without_conditioned_boundary = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        gpu_conditioned_evidence_facts: 1,
        gpu_conditioned_know_evidence_facts: 1,
        gpu_source_conditioned_evidence_facts: 1,
        gpu_source_conditioned_know_evidence_facts: 1,
        gpu_knowledge_compilation_end_to_end_runs: 1,
        gpu_source_knowledge_compilation_end_to_end_runs: 1,
        accepted_gpu_production_path_events: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = facts_without_conditioned_boundary
        .require_production_metric_eligibility()
        .expect_err("conditioned evidence facts require an accepted conditioned boundary");
    assert!(format!("{err}").contains("require accepted conditioned world-view evidence"));

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
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        gpu_pir_graph_uploads: 1,
        gpu_source_pir_graph_uploads: 1,
        gpu_cnf_encodes: 1,
        gpu_source_cnf_encodes: 1,
        accepted_gpu_production_path_events: 4,
        fixture_circuit_evaluations: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = fixture
        .require_production_metric_eligibility()
        .expect_err("fixture circuit trace must not satisfy production metrics");
    assert!(format!("{err}").contains("CPU probabilistic fallback counters must be zero"));

    let cpu_recompute = EpistemicProbProductionTrace {
        cpu_only_probability_recomputations: 1,
        ..eligible
    };
    let err = cpu_recompute
        .require_production_metric_eligibility()
        .expect_err("CPU-only probability recompute trace must not satisfy production metrics");
    assert!(format!("{err}").contains("CPU probabilistic fallback counters must be zero"));
}

#[test]
fn production_prob_metric_gate_rejects_unpaired_pir_cnf_accounting() {
    let pir_without_cnf = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        gpu_pir_graph_uploads: 1,
        accepted_gpu_production_path_events: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = pir_without_cnf
        .require_production_metric_eligibility()
        .expect_err("accepted PIR uploads must be paired with GPU CNF encoding");
    assert!(format!("{err}").contains("PIR/CNF production accounting must match"));

    let cnf_without_pir = EpistemicProbProductionTrace {
        accepted_world_view_evidence_consumed: 1,
        accepted_faeel_world_view_evidence_consumed: 1,
        accepted_evidence_assumptions_consumed: 1,
        gpu_cnf_encodes: 1,
        accepted_gpu_production_path_events: 1,
        ..EpistemicProbProductionTrace::default()
    };
    let err = cnf_without_pir
        .require_production_metric_eligibility()
        .expect_err("accepted CNF encodes must be paired with GPU PIR upload");
    assert!(format!("{err}").contains("PIR/CNF production accounting must match"));
}
