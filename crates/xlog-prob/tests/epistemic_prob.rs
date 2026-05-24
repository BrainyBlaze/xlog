use xlog_logic::epistemic::{EpistemicWorld, EpistemicWorldView};
use xlog_prob::epistemic::{
    conditional_probability_from_logs, AcceptedWorldViewEvidence, CircuitUpdateMode,
    CompilerAdapterKind, CompilerAdapterSupport, CompilerInputFormat, CompilerOutputFormat,
    EpistemicAssumption, EpistemicCircuit, EpistemicEvidenceTerm, EpistemicProbabilisticRole,
    KnowledgeCompilerAdapter, EPISTEMIC_PROBABILITY_TOLERANCE,
};

#[test]
fn epistemic_choices_are_compiled_as_probabilistic_evidence() {
    let assumption = EpistemicAssumption::known("rain", 0, true);
    let mut circuit = EpistemicCircuit::compile(
        0.25,
        vec![(assumption.clone(), 0.75)],
        KnowledgeCompilerAdapter::gpu_d4(),
    )
    .unwrap();

    assert_eq!(
        circuit.semantic_contract().epistemic_role,
        EpistemicProbabilisticRole::EvidenceConditioning
    );
    assert_eq!(assumption.evidence_literal(), "know:rain/0=true");
    assert!(circuit.compiler_evidence_literals().is_empty());

    circuit.apply_assumption(assumption).unwrap();
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec!["know:rain/0=true".to_string()]
    );
    assert!(circuit.query_probability().within_tolerance(0.75));
}

#[test]
fn incremental_assumption_update_reuses_circuit_when_adapter_supports_it() {
    let assumption = EpistemicAssumption::known("rain", 0, true);
    let mut circuit = EpistemicCircuit::compile(
        0.25,
        vec![(assumption.clone(), 0.75)],
        KnowledgeCompilerAdapter::gpu_d4(),
    )
    .unwrap();
    let original_fingerprint = circuit.circuit_fingerprint();

    let update = circuit.apply_assumption(assumption).unwrap();

    assert_eq!(update.mode, CircuitUpdateMode::IncrementalEvidence);
    assert_eq!(update.compile_count, 1);
    assert_eq!(update.circuit_fingerprint, original_fingerprint);
    assert_eq!(circuit.incremental_update_count(), 1);
}

#[test]
fn changed_assumption_replaces_active_evidence_without_rebuilding_circuit() {
    let rain_true = EpistemicAssumption::known("rain", 0, true);
    let rain_false = EpistemicAssumption::known("rain", 0, false);
    let true_world_view =
        EpistemicWorldView::from_worlds(vec![EpistemicWorld::new().with_fact("rain", 0)]).unwrap();
    let false_world_view = EpistemicWorldView::from_worlds(vec![EpistemicWorld::new()]).unwrap();
    let mut circuit = EpistemicCircuit::compile(
        0.25,
        vec![(rain_true.clone(), 0.75), (rain_false.clone(), 0.10)],
        KnowledgeCompilerAdapter::gpu_d4(),
    )
    .unwrap();
    let original_fingerprint = circuit.circuit_fingerprint();

    let first_update = circuit
        .apply_accepted_world_view(
            AcceptedWorldViewEvidence::new(&true_world_view, vec![rain_true]).unwrap(),
        )
        .unwrap();
    assert_eq!(first_update.mode, CircuitUpdateMode::IncrementalEvidence);
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec!["know:rain/0=true"]
    );
    assert!(circuit.query_probability().within_tolerance(0.75));

    let changed_update = circuit
        .apply_accepted_world_view(
            AcceptedWorldViewEvidence::new(&false_world_view, vec![rain_false.clone()]).unwrap(),
        )
        .unwrap();

    assert_eq!(changed_update.mode, CircuitUpdateMode::IncrementalEvidence);
    assert_eq!(changed_update.compile_count, 1);
    assert_eq!(changed_update.circuit_fingerprint, original_fingerprint);
    assert_eq!(circuit.incremental_update_count(), 2);
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec!["know:rain/0=false"]
    );
    assert!(circuit.query_probability().within_tolerance(0.10));

    let unchanged_update = circuit
        .apply_accepted_world_view(
            AcceptedWorldViewEvidence::new(&false_world_view, vec![rain_false]).unwrap(),
        )
        .unwrap();
    assert_eq!(unchanged_update.mode, CircuitUpdateMode::Unchanged);
    assert_eq!(circuit.incremental_update_count(), 2);
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec!["know:rain/0=false"]
    );
}

#[test]
fn full_rebuild_fingerprint_distinguishes_nonzero_arity_evidence_terms() {
    let gate_7 =
        EpistemicAssumption::known_tuple("gate", vec![EpistemicEvidenceTerm::integer(7)], true);
    let gate_9 =
        EpistemicAssumption::known_tuple("gate", vec![EpistemicEvidenceTerm::integer(9)], true);
    let conditioned = vec![(gate_7.clone(), 0.7), (gate_9.clone(), 0.9)];
    let mut circuit_7 = EpistemicCircuit::compile(
        0.25,
        conditioned.clone(),
        KnowledgeCompilerAdapter::external_c2d(),
    )
    .unwrap();
    let mut circuit_9 =
        EpistemicCircuit::compile(0.25, conditioned, KnowledgeCompilerAdapter::external_c2d())
            .unwrap();

    let update_7 = circuit_7.apply_assumption(gate_7).unwrap();
    let update_9 = circuit_9.apply_assumption(gate_9).unwrap();

    assert_eq!(update_7.mode, CircuitUpdateMode::FullRebuild);
    assert_eq!(update_9.mode, CircuitUpdateMode::FullRebuild);
    assert_eq!(update_7.compile_count, 2);
    assert_eq!(update_9.compile_count, 2);
    assert_ne!(update_7.circuit_fingerprint, update_9.circuit_fingerprint);
    assert!(circuit_7.query_probability().within_tolerance(0.7));
    assert!(circuit_9.query_probability().within_tolerance(0.9));
}

#[test]
fn external_ddnnf_text_compiler_adapter_is_explicitly_represented() {
    let adapter = KnowledgeCompilerAdapter::external_ddnnf_text("d4-compatible-ddnnf");

    assert_eq!(adapter.kind, CompilerAdapterKind::ExternalDdnnfText);
    assert_eq!(adapter.support, CompilerAdapterSupport::DesignOnly);
    assert_eq!(adapter.input_format, CompilerInputFormat::DimacsCnf);
    assert_eq!(
        adapter.output_format,
        CompilerOutputFormat::DecisionDnnfText
    );
    assert!(!adapter.supports_incremental_evidence());
}

#[test]
fn c2d_and_minic2d_compiler_adapters_are_explicitly_represented() {
    let c2d = KnowledgeCompilerAdapter::external_c2d();
    let minic2d = KnowledgeCompilerAdapter::external_mini_c2d();

    assert_eq!(c2d.name, "c2d");
    assert_eq!(c2d.kind, CompilerAdapterKind::ExternalC2d);
    assert_eq!(c2d.support, CompilerAdapterSupport::DesignOnly);
    assert_eq!(c2d.input_format, CompilerInputFormat::DimacsCnf);
    assert_eq!(c2d.output_format, CompilerOutputFormat::DecisionDnnfText);
    assert!(!c2d.supports_incremental_evidence());

    assert_eq!(minic2d.name, "miniC2D");
    assert_eq!(minic2d.kind, CompilerAdapterKind::ExternalMiniC2d);
    assert_eq!(minic2d.support, CompilerAdapterSupport::DesignOnly);
    assert_eq!(minic2d.input_format, CompilerInputFormat::DimacsCnf);
    assert_eq!(
        minic2d.output_format,
        CompilerOutputFormat::DecisionDnnfText
    );
    assert!(!minic2d.supports_incremental_evidence());
}

#[test]
fn log_space_conditional_probability_is_tolerance_bounded() {
    let probability = conditional_probability_from_logs(
        0.21f64.ln(),
        0.3f64.ln(),
        EPISTEMIC_PROBABILITY_TOLERANCE,
    )
    .unwrap();

    assert!(probability.within_tolerance(0.7));

    let clipped = conditional_probability_from_logs(
        (1.0f64 + EPISTEMIC_PROBABILITY_TOLERANCE / 2.0).ln(),
        1.0f64.ln(),
        EPISTEMIC_PROBABILITY_TOLERANCE,
    )
    .unwrap();

    assert_eq!(clipped.probability, 1.0);
}

#[test]
fn evidence_conditioning_consumes_accepted_world_view() {
    let assumption = EpistemicAssumption::known("rain", 0, true);
    let world_view = EpistemicWorldView::from_worlds(vec![
        EpistemicWorld::new().with_fact("rain", 0),
        EpistemicWorld::new().with_fact("rain", 0),
    ])
    .unwrap();
    let evidence = AcceptedWorldViewEvidence::new(&world_view, vec![assumption.clone()]).unwrap();
    assert_eq!(evidence.world_count(), 2);

    let mut circuit = EpistemicCircuit::compile(
        0.25,
        vec![(assumption, 0.75)],
        KnowledgeCompilerAdapter::gpu_d4(),
    )
    .unwrap();

    let update = circuit.apply_accepted_world_view(evidence).unwrap();

    assert_eq!(update.mode, CircuitUpdateMode::IncrementalEvidence);
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec!["know:rain/0=true"]
    );
    assert!(circuit.query_probability().within_tolerance(0.75));
}

#[test]
fn accepted_world_view_evidence_rejects_unvalidated_assumption() {
    let world_view =
        EpistemicWorldView::from_worlds(vec![EpistemicWorld::new().with_fact("rain", 0)]).unwrap();

    let err = AcceptedWorldViewEvidence::new(
        &world_view,
        vec![EpistemicAssumption::known("sun", 0, true)],
    )
    .expect_err(
        "assumption absent from the accepted world view must not become probability evidence",
    );

    assert!(format!("{err}").contains("not accepted by world view"));
}
