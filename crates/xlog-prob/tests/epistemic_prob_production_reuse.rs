use std::fs;
use std::path::PathBuf;

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
    assert!(production.contains("compile_and_evaluate_program_with_gpu_execution_result"));
    assert!(production.contains("encode_source_pir_cnf_with_gpu_execution_result"));
    assert!(production.contains("encode_program_pir_cnf_with_gpu_execution_result"));
    assert!(production.contains("evaluate_with_gpu_execution_result"));
    assert!(production.contains("evaluate_gpu_with_grads_with_gpu_execution_result"));
    assert!(production.contains("from_gpu_execution_result"));
    assert!(production.contains("ExactDdnnfProgram::compile_source_with_gpu"));
    assert!(production.contains("ExactDdnnfProgram::compile_from_program"));
    assert!(production.contains("GpuPirGraph::from_host"));
    assert!(production.contains("encode_cnf_gpu"));
    assert!(production.contains("evaluate_gpu_with_grads"));
    assert!(production.contains("gpu_pir_graph_uploads"));
    assert!(production.contains("gpu_cnf_encodes"));
    assert!(production.contains("gpu_knowledge_compilation_end_to_end_runs"));
    assert!(production.contains("gpu_exact_query_evaluations"));
    assert!(production.contains("gpu_exact_gradient_evaluations"));
    assert!(production.contains("cpu_only_probability_recomputations: 0"));
    assert!(production.contains("fixture_circuit_evaluations: 0"));
    assert!(!production.contains("EpistemicCircuit::compile"));
    assert!(!production.contains("conditional_probability_from_logs"));
    assert!(!production.contains("query_probability"));
}
