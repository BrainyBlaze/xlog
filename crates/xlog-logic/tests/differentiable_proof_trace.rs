use xlog_logic::{DifferentiableProofTraceMap, ProofTraceSpec};

#[test]
fn proof_path_exports_stable_ids_and_clause_gradients() {
    let mut traces = DifferentiableProofTraceMap::new();
    let proof_id = traces.insert(ProofTraceSpec {
        answer_key: "root(case_1, primary_root)".to_string(),
        clause_id: "clause_primary".to_string(),
        support_atoms: vec![
            "neural_root(case_1, primary_root)".to_string(),
            "candidate(case_1, primary_root)".to_string(),
        ],
        initial_weight: 0.0,
    });
    let repeated_id = traces.insert(ProofTraceSpec {
        answer_key: "root(case_1, primary_root)".to_string(),
        clause_id: "clause_primary".to_string(),
        support_atoms: vec![
            "neural_root(case_1, primary_root)".to_string(),
            "candidate(case_1, primary_root)".to_string(),
        ],
        initial_weight: 0.0,
    });

    assert_eq!(proof_id, repeated_id);

    let loss = traces
        .accumulate_binary_logistic_gradients(&[("root(case_1, primary_root)".to_string(), 1.0)]);
    assert!(loss > 0.0);

    let trace = traces.trace(proof_id).expect("trace should be exported");
    assert_eq!(trace.clause_id, "clause_primary");
    assert_eq!(trace.support_atoms.len(), 2);
    assert!(trace.gradient.abs() > 0.0);

    let weight_before = trace.weight;
    traces.apply_gradients(0.5);
    let weight_after = traces.trace(proof_id).unwrap().weight;
    assert_ne!(weight_after, weight_before);
}
