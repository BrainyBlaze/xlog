use xlog_prob::epistemic::{
    conditional_probability_from_logs, CircuitUpdateMode, CompilerAdapterKind,
    CompilerAdapterSupport, CompilerInputFormat, CompilerOutputFormat, EpistemicAssumption,
    EpistemicCircuit, EpistemicProbabilisticRole, KnowledgeCompilerAdapter,
    EPISTEMIC_PROBABILITY_TOLERANCE,
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
