use std::collections::BTreeMap;
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

#[cfg(feature = "host-io")]
use xlog_core::symbol;
use xlog_core::{MemoryBudget, RelId, ScalarType, Schema};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::{
    CompiledRule, EirAtom, EirEpistemicLiteral, EirEpistemicMode, EirEpistemicOp, EirTerm,
    EpistemicExecutablePlan, EpistemicGpuPlan, EpistemicReductionPlan,
    EpistemicWcojReductionStatus, ExecutionPlan, RirMeta, RirNode, Scc, Stratum,
};
#[cfg(feature = "host-io")]
use xlog_logic::epistemic::{
    compile_epistemic_gpu_execution, compile_epistemic_gpu_split_execution,
};
#[cfg(feature = "host-io")]
use xlog_logic::parse_program;
#[cfg(feature = "host-io")]
use xlog_prob::epistemic::{
    CircuitUpdateMode, EpistemicAssumption, EpistemicCircuit, EpistemicEvidenceTerm,
    KnowledgeCompilerAdapter,
};
#[cfg(feature = "host-io")]
use xlog_prob::epistemic_production::EpistemicProbGpuBatchExecutionEvidence;
use xlog_prob::epistemic_production::EpistemicProbProductionAdapter;
use xlog_prob::exact::GpuConfig;
#[cfg(feature = "host-io")]
use xlog_prob::provenance::Value;
#[cfg(feature = "host-io")]
use xlog_runtime::EpistemicGpuBatchExecutionResult;
use xlog_runtime::{EpistemicGpuWorkspaceCapacities, Executor};

struct LockedCudaProvider {
    _guard: MutexGuard<'static, ()>,
    provider: Arc<CudaKernelProvider>,
}

impl Deref for LockedCudaProvider {
    type Target = Arc<CudaKernelProvider>;

    fn deref(&self) -> &Self::Target {
        &self.provider
    }
}

fn gpu_exact_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn try_provider() -> Option<LockedCudaProvider> {
    let guard = gpu_exact_test_lock();
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
        Ok(p) => Some(LockedCudaProvider {
            _guard: guard,
            provider: Arc::new(p),
        }),
        Err(e) => {
            eprintln!("Skipping test: failed to create CUDA kernel provider: {e}");
            None
        }
    }
}

#[test]
fn public_prob_adapter_consumes_real_runtime_accepted_evidence_for_gpu_pir_cnf() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_ground_literal(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let source = r#"
0.6::base(7).
query(base(7)).
"#;

    let pir_cnf = adapter
        .encode_source_pir_cnf_with_gpu_execution_result(source, &provider, &result, Vec::new())
        .expect("accepted runtime evidence should gate GPU PIR/CNF production encoding");

    assert!(pir_cnf.pir_nodes > 0);
    assert!(pir_cnf.root_count > 0);
    assert!(pir_cnf.cnf_var_cap > 0);
    assert!(pir_cnf.cnf_clause_cap > 0);
    assert!(pir_cnf.cnf_lit_cap > 0);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        1
    );
    assert_eq!(trace.accepted_gpu_max_evidence_arity_consumed, 1);
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 1);
    assert_eq!(trace.gpu_pir_graph_uploads, 1);
    assert_eq!(trace.gpu_source_pir_graph_uploads, 1);
    assert_eq!(trace.gpu_cnf_encodes, 1);
    assert_eq!(trace.gpu_source_cnf_encodes, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 4);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("accepted probability path must not use CPU recomputation");
    trace
        .require_production_metric_eligibility()
        .expect("accepted runtime evidence plus GPU PIR/CNF reuse satisfies prob metric gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_rejects_real_runtime_without_accepted_final_output_before_gpu_pir_cnf() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_runtime_without_accepted_final_output(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let source = r#"
0.6::base(7).
query(base(7)).
"#;

    let err = match adapter.encode_source_pir_cnf_with_gpu_execution_result(
        source,
        &provider,
        &result,
        Vec::new(),
    ) {
        Ok(_) => {
            panic!(
                "runtime evidence with no accepted final rows must not gate GPU PIR/CNF encoding"
            )
        }
        Err(err) => err,
    };

    assert!(format!("{err}").contains("non-empty accepted GPU final output"));
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 0);
    assert_eq!(trace.accepted_gpu_production_path_events, 0);
    assert_eq!(trace.gpu_pir_graph_uploads, 0);
    assert_eq!(trace.gpu_cnf_encodes, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("rejected evidence gate must not use probability recomputation");
    trace
        .require_production_metric_eligibility()
        .expect_err("rejected evidence must not satisfy probability production metrics");
}

#[test]
fn public_prob_adapter_consumes_variable_bound_runtime_evidence_for_gpu_pir_cnf() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_variable_bound_literal(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let source = r#"
0.6::base(7).
query(base(7)).
"#;

    let pir_cnf = adapter
        .encode_source_pir_cnf_with_gpu_execution_result(source, &provider, &result, Vec::new())
        .expect("variable-bound accepted runtime evidence should gate GPU PIR/CNF encoding");

    assert!(pir_cnf.pir_nodes > 0);
    assert!(pir_cnf.root_count > 0);
    assert!(pir_cnf.cnf_var_cap > 0);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        1
    );
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 1);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_row_specific_membership_row_capacity_consumed,
        1
    );
    assert_eq!(
        trace.accepted_gpu_row_filter_fallback_row_capacity_consumed,
        0
    );
    assert_eq!(trace.gpu_pir_graph_uploads, 1);
    assert_eq!(trace.gpu_cnf_encodes, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 4);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("variable-bound accepted probability path must not use CPU recomputation");
    trace
        .require_production_metric_eligibility()
        .expect("variable-bound accepted runtime evidence plus GPU PIR/CNF reuse is eligible");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_consumes_parsed_program_runtime_evidence_for_gpu_pir_cnf() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_variable_bound_literal(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.6::base(7).
query(base(7)).
"#;
    let program = parse_program(src).expect("parse PIR/CNF probability program");

    let pir_cnf = adapter
        .encode_program_pir_cnf_with_gpu_execution_result(&program, &provider, &result, Vec::new())
        .expect("parsed program accepted runtime evidence should gate GPU PIR/CNF encoding");

    assert!(pir_cnf.pir_nodes > 0);
    assert!(pir_cnf.root_count > 0);
    assert!(pir_cnf.cnf_var_cap > 0);
    assert!(pir_cnf.cnf_clause_cap > 0);
    assert!(pir_cnf.cnf_lit_cap > 0);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        1
    );
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 1);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_row_specific_membership_row_capacity_consumed,
        1
    );
    assert_eq!(trace.gpu_pir_graph_uploads, 1);
    assert_eq!(trace.gpu_source_pir_graph_uploads, 0);
    assert_eq!(trace.gpu_program_pir_graph_uploads, 1);
    assert_eq!(trace.gpu_cnf_encodes, 1);
    assert_eq!(trace.gpu_source_cnf_encodes, 0);
    assert_eq!(trace.gpu_program_cnf_encodes, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 4);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("parsed-program PIR/CNF path must not use CPU recomputation");
    trace
        .require_production_metric_eligibility()
        .expect("parsed-program accepted runtime evidence plus GPU PIR/CNF reuse is eligible");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_gpu_exact_query_on_real_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_ground_literal(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.6::base(7).
query(base(7)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("accepted runtime evidence should condition GPU exact evaluation");

    assert!((prob_of(&exact, "base", &[Value::I64(7)]) - 1.0).abs() < 1e-9);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        1
    );
    assert_eq!(trace.accepted_gpu_max_evidence_arity_consumed, 1);
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 1);
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        1
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_source_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 5);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("conditioned GPU exact evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_gpu_exact_query_on_binary_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_binary_bound_literal(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.2::gate(1, 2).
0.4::gate(1, 3).
0.6::gate(2, 4).
query(gate(1, 2)).
query(gate(1, 3)).
query(gate(2, 4)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("binary accepted runtime evidence should condition GPU exact evaluation");

    assert!((prob_of(&exact, "gate", &[Value::I64(1), Value::I64(2)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "gate", &[Value::I64(1), Value::I64(3)]) - 0.4).abs() < 1e-9);
    assert!((prob_of(&exact, "gate", &[Value::I64(2), Value::I64(4)]) - 1.0).abs() < 1e-9);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_max_evidence_arity_consumed, 2);
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 2);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_row_specific_membership_row_capacity_consumed,
        1
    );
    assert_eq!(
        trace.accepted_gpu_row_filter_fallback_row_capacity_consumed,
        2
    );
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        1
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 2);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 2);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_source_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 5);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("binary conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("binary conditioned GPU exact evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_binary_negative_gpu_evidence_on_exact_path() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_parsed_binary_operator_evidence(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.2::known_gate(1, 2).
0.4::possible_gate(1, 2).
0.3::missing_known_gate(1, 2).
0.6::missing_possible_gate(1, 2).
query(known_gate(1, 2)).
query(possible_gate(1, 2)).
query(missing_known_gate(1, 2)).
query(missing_possible_gate(1, 2)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("binary negative accepted runtime evidence should condition GPU exact evaluation");

    assert!((prob_of(&exact, "known_gate", &[Value::I64(1), Value::I64(2)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "possible_gate", &[Value::I64(1), Value::I64(2)]) - 1.0).abs() < 1e-9);
    assert!(
        (prob_of(
            &exact,
            "missing_known_gate",
            &[Value::I64(1), Value::I64(2)]
        ) - 0.0)
            .abs()
            < 1e-9
    );
    assert!(
        (prob_of(
            &exact,
            "missing_possible_gate",
            &[Value::I64(1), Value::I64(2)]
        ) - 0.0)
            .abs()
            < 1e-9
    );

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        4
    );
    assert_eq!(trace.accepted_gpu_max_evidence_arity_consumed, 2);
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 8);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 4);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        2
    );
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        1
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 4);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 2);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 2);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_known_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_source_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 5);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("binary negative conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("binary negative evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_gpu_exact_query_on_quaternary_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_quaternary_bound_literal(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.2::gate(1, 2, 3, 4).
0.4::gate(1, 2, 3, 5).
0.6::gate(2, 3, 5, 8).
query(gate(1, 2, 3, 4)).
query(gate(1, 2, 3, 5)).
query(gate(2, 3, 5, 8)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("quaternary accepted runtime evidence should condition GPU exact evaluation");

    assert!(
        (prob_of(
            &exact,
            "gate",
            &[Value::I64(1), Value::I64(2), Value::I64(3), Value::I64(4)]
        ) - 1.0)
            .abs()
            < 1e-9
    );
    assert!(
        (prob_of(
            &exact,
            "gate",
            &[Value::I64(1), Value::I64(2), Value::I64(3), Value::I64(5)]
        ) - 0.4)
            .abs()
            < 1e-9
    );
    assert!(
        (prob_of(
            &exact,
            "gate",
            &[Value::I64(2), Value::I64(3), Value::I64(5), Value::I64(8)]
        ) - 1.0)
            .abs()
            < 1e-9
    );

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_max_evidence_arity_consumed, 4);
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 4);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_row_specific_membership_row_capacity_consumed,
        3
    );
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        1
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_source_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 5);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("quaternary conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("quaternary conditioned GPU exact evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_gpu_exact_query_on_symbol_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let alpha = symbol::intern("alpha");
    let beta = symbol::intern("beta");
    let gamma = symbol::intern("gamma");
    let result = execute_accepted_symbol_variable_bound_literal(&provider, alpha, beta, gamma);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.2::gate(alpha).
0.4::gate(beta).
0.6::gate(gamma).
query(gate(alpha)).
query(gate(beta)).
query(gate(gamma)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("symbol accepted runtime evidence should condition GPU exact evaluation");

    assert!((prob_of(&exact, "gate", &[Value::Symbol(alpha)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "gate", &[Value::Symbol(beta)]) - 0.4).abs() < 1e-9);
    assert!((prob_of(&exact, "gate", &[Value::Symbol(gamma)]) - 1.0).abs() < 1e-9);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_max_evidence_arity_consumed, 1);
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 1);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        1
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_source_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 5);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("symbol conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("symbol conditioned GPU exact evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_quoted_symbol_query_on_symbol_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let alpha = symbol::intern("alpha");
    let beta = symbol::intern("beta");
    let gamma = symbol::intern("gamma");
    let result = execute_accepted_symbol_variable_bound_literal(&provider, alpha, beta, gamma);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.2::gate("alpha").
0.4::gate("beta").
0.6::gate("gamma").
query(gate("alpha")).
query(gate("beta")).
query(gate("gamma")).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("quoted symbol-valued accepted evidence should condition GPU exact evaluation");

    assert!((prob_of(&exact, "gate", &[Value::String("alpha".to_string())]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "gate", &[Value::String("beta".to_string())]) - 0.4).abs() < 1e-9);
    assert!((prob_of(&exact, "gate", &[Value::String("gamma".to_string())]) - 1.0).abs() < 1e-9);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("quoted symbol conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect(
            "quoted symbol conditioned GPU exact evidence should satisfy the stricter prob gate",
        );
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_gpu_exact_query_on_g91_runtime_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_g91_possible_literal(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.3::p(7).
query(p(7)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("G91 accepted runtime evidence should condition GPU exact evaluation");

    assert!((prob_of(&exact, "p", &[Value::I64(7)]) - 1.0).abs() < 1e-9);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_g91_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 0);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        1
    );
    assert_eq!(trace.accepted_gpu_max_evidence_arity_consumed, 1);
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 1);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        1
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 0);
    assert_eq!(trace.gpu_source_conditioned_know_evidence_facts, 0);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_source_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 5);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("G91 conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("G91 conditioned GPU exact evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_applies_g91_possible_gpu_evidence_incrementally() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_g91_possible_literal(&provider);
    let accepted_possible =
        EpistemicAssumption::possible_tuple("p", vec![EpistemicEvidenceTerm::integer(7)], true);
    let mut circuit = EpistemicCircuit::compile(
        0.25,
        vec![(accepted_possible.clone(), 0.73)],
        KnowledgeCompilerAdapter::gpu_d4(),
    )
    .expect("compile incremental G91 epistemic circuit");
    let original_fingerprint = circuit.circuit_fingerprint();
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());

    let update = adapter
        .apply_accepted_world_view_to_circuit_with_gpu_execution_result(
            &mut circuit,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("G91 accepted possible GPU evidence should update incremental circuit");

    assert_eq!(update.mode, CircuitUpdateMode::IncrementalEvidence);
    assert_eq!(update.compile_count, 1);
    assert_eq!(update.circuit_fingerprint, original_fingerprint);
    assert_eq!(circuit.incremental_update_count(), 1);
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec![accepted_possible.evidence_literal()]
    );
    assert!(circuit.query_probability().within_tolerance(0.73));

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_g91_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 0);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        1
    );
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 1);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(trace.accepted_incremental_circuit_updates, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("G91 incremental probability evidence must not use CPU recomputation");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_applies_gpu_evidence_incrementally_before_gpu_exact_path() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_accepted_variable_bound_literal(&provider);
    let accepted_base =
        EpistemicAssumption::known_tuple("base", vec![EpistemicEvidenceTerm::integer(7)], true);
    let mut circuit = EpistemicCircuit::compile(
        0.25,
        vec![(accepted_base.clone(), 0.85)],
        KnowledgeCompilerAdapter::gpu_d4(),
    )
    .expect("compile incremental epistemic circuit");
    let original_fingerprint = circuit.circuit_fingerprint();
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());

    let update = adapter
        .apply_accepted_world_view_to_circuit_with_gpu_execution_result(
            &mut circuit,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("accepted GPU runtime evidence should update incremental circuit");

    assert_eq!(update.mode, CircuitUpdateMode::IncrementalEvidence);
    assert_eq!(update.compile_count, 1);
    assert_eq!(update.circuit_fingerprint, original_fingerprint);
    assert_eq!(circuit.incremental_update_count(), 1);
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec![accepted_base.evidence_literal()]
    );
    assert!(circuit.query_probability().within_tolerance(0.85));

    let src = r#"
0.6::base(7).
query(base(7)).
"#;
    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("accepted GPU runtime evidence should still gate GPU exact evaluation");

    assert!((prob_of(&exact, "base", &[Value::I64(7)]) - 1.0).abs() < 1e-9);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 2);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 2);
    assert_eq!(trace.accepted_incremental_circuit_updates, 1);
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        1
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_source_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 5);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("incremental plus exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("incremental evidence plus GPU exact path should satisfy conditioned prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_applies_changed_gpu_evidence_incrementally_without_rebuild() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result_7 = execute_accepted_variable_bound_literal_with_base(&provider, 7);
    let result_9 = execute_accepted_variable_bound_literal_with_base(&provider, 9);
    let accepted_base_7 =
        EpistemicAssumption::known_tuple("base", vec![EpistemicEvidenceTerm::integer(7)], true);
    let accepted_base_9 =
        EpistemicAssumption::known_tuple("base", vec![EpistemicEvidenceTerm::integer(9)], true);
    let mut circuit = EpistemicCircuit::compile(
        0.25,
        vec![
            (accepted_base_7.clone(), 0.85),
            (accepted_base_9.clone(), 0.95),
        ],
        KnowledgeCompilerAdapter::gpu_d4(),
    )
    .expect("compile incremental epistemic circuit");
    let original_fingerprint = circuit.circuit_fingerprint();
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());

    let update_7 = adapter
        .apply_accepted_world_view_to_circuit_with_gpu_execution_result(
            &mut circuit,
            &provider,
            &result_7,
            Vec::new(),
        )
        .expect("first accepted GPU runtime evidence should update incremental circuit");

    assert_eq!(update_7.mode, CircuitUpdateMode::IncrementalEvidence);
    assert_eq!(update_7.compile_count, 1);
    assert_eq!(update_7.circuit_fingerprint, original_fingerprint);
    assert_eq!(circuit.incremental_update_count(), 1);
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec![accepted_base_7.evidence_literal()]
    );
    assert!(circuit.query_probability().within_tolerance(0.85));

    let update_9 = adapter
        .apply_accepted_world_view_to_circuit_with_gpu_execution_result(
            &mut circuit,
            &provider,
            &result_9,
            Vec::new(),
        )
        .expect("changed accepted GPU runtime evidence should update incremental circuit");

    assert_eq!(update_9.mode, CircuitUpdateMode::IncrementalEvidence);
    assert_eq!(update_9.compile_count, 1);
    assert_eq!(update_9.circuit_fingerprint, original_fingerprint);
    assert_eq!(circuit.incremental_update_count(), 2);
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec![
            accepted_base_7.evidence_literal(),
            accepted_base_9.evidence_literal(),
        ]
    );
    assert!(circuit.query_probability().within_tolerance(0.85));

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 2);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 2);
    assert_eq!(trace.accepted_incremental_circuit_updates, 2);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("changed incremental GPU evidence must not use CPU recomputation");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_applies_negated_gpu_operator_evidence_incrementally() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_parsed_all_operator_variable_bound_evidence(&provider);
    let accepted_known = EpistemicAssumption::known_tuple(
        "known_gate",
        vec![EpistemicEvidenceTerm::integer(7)],
        true,
    );
    let accepted_not_known = EpistemicAssumption::known_tuple(
        "not_known_gate",
        vec![EpistemicEvidenceTerm::integer(7)],
        false,
    );
    let accepted_not_possible = EpistemicAssumption::possible_tuple(
        "not_possible_gate",
        vec![EpistemicEvidenceTerm::integer(7)],
        false,
    );
    let accepted_possible = EpistemicAssumption::possible_tuple(
        "possible_gate",
        vec![EpistemicEvidenceTerm::integer(7)],
        true,
    );
    let mut circuit = EpistemicCircuit::compile(
        0.25,
        vec![
            (accepted_known.clone(), 0.61),
            (accepted_not_known.clone(), 0.37),
            (accepted_not_possible.clone(), 0.23),
            (accepted_possible.clone(), 0.71),
        ],
        KnowledgeCompilerAdapter::gpu_d4(),
    )
    .expect("compile mixed-operator incremental epistemic circuit");
    let original_fingerprint = circuit.circuit_fingerprint();
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());

    let update = adapter
        .apply_accepted_world_view_to_circuit_with_gpu_execution_result(
            &mut circuit,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("mixed positive and negated GPU evidence should update incremental circuit");

    assert_eq!(update.mode, CircuitUpdateMode::IncrementalEvidence);
    assert_eq!(update.compile_count, 1);
    assert_eq!(update.circuit_fingerprint, original_fingerprint);
    assert_eq!(circuit.incremental_update_count(), 4);
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec![
            accepted_known.evidence_literal(),
            accepted_not_known.evidence_literal(),
            accepted_not_possible.evidence_literal(),
            accepted_possible.evidence_literal(),
        ]
    );
    assert!(circuit.query_probability().within_tolerance(0.61));

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        4
    );
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 4);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 4);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        2
    );
    assert_eq!(trace.accepted_incremental_circuit_updates, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("mixed-operator incremental GPU evidence must not use CPU recomputation");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_applies_external_c2d_gpu_evidence_with_full_rebuild() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_parsed_all_operator_variable_bound_evidence(&provider);
    let accepted_known = EpistemicAssumption::known_tuple(
        "known_gate",
        vec![EpistemicEvidenceTerm::integer(7)],
        true,
    );
    let accepted_not_known = EpistemicAssumption::known_tuple(
        "not_known_gate",
        vec![EpistemicEvidenceTerm::integer(7)],
        false,
    );
    let accepted_not_possible = EpistemicAssumption::possible_tuple(
        "not_possible_gate",
        vec![EpistemicEvidenceTerm::integer(7)],
        false,
    );
    let accepted_possible = EpistemicAssumption::possible_tuple(
        "possible_gate",
        vec![EpistemicEvidenceTerm::integer(7)],
        true,
    );
    let mut circuit = EpistemicCircuit::compile(
        0.25,
        vec![
            (accepted_known.clone(), 0.61),
            (accepted_not_known.clone(), 0.37),
            (accepted_not_possible.clone(), 0.23),
            (accepted_possible.clone(), 0.71),
        ],
        KnowledgeCompilerAdapter::external_c2d(),
    )
    .expect("compile external c2d epistemic circuit");
    let original_fingerprint = circuit.circuit_fingerprint();
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());

    let update = adapter
        .apply_accepted_world_view_to_circuit_with_gpu_execution_result(
            &mut circuit,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("accepted GPU evidence should update external c2d circuit");

    assert_eq!(update.mode, CircuitUpdateMode::FullRebuild);
    assert_eq!(update.compile_count, 5);
    assert_ne!(update.circuit_fingerprint, original_fingerprint);
    assert_eq!(circuit.incremental_update_count(), 0);
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec![
            accepted_known.evidence_literal(),
            accepted_not_known.evidence_literal(),
            accepted_not_possible.evidence_literal(),
            accepted_possible.evidence_literal(),
        ]
    );
    assert!(circuit.query_probability().within_tolerance(0.61));

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        4
    );
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 4);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 4);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        2
    );
    assert_eq!(trace.accepted_incremental_circuit_updates, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("external c2d accepted GPU evidence must not use CPU recomputation");
    trace
        .require_production_metric_eligibility()
        .expect_err("external c2d circuit updates alone must not satisfy production metrics");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_applies_split_batch_gpu_evidence_to_incremental_circuit() {
    let Some(provider) = try_provider() else {
        return;
    };

    let batch = execute_accepted_split_batch(&provider);
    let empty_assumptions: &[xlog_prob::epistemic::EpistemicAssumption] = &[];
    let assumptions_by_component = [empty_assumptions, empty_assumptions];
    let accepted_left = EpistemicAssumption::known_tuple(
        "left_base",
        vec![EpistemicEvidenceTerm::integer(7)],
        true,
    );
    let accepted_right = EpistemicAssumption::known_tuple(
        "right_base",
        vec![EpistemicEvidenceTerm::integer(9)],
        true,
    );
    let mut circuit = EpistemicCircuit::compile(
        0.25,
        vec![
            (accepted_left.clone(), 0.61),
            (accepted_right.clone(), 0.83),
        ],
        KnowledgeCompilerAdapter::gpu_d4(),
    )
    .expect("compile split-batch incremental epistemic circuit");
    let original_fingerprint = circuit.circuit_fingerprint();
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());

    let updates = adapter
        .apply_accepted_world_views_to_circuit_for_gpu_batch_execution_result(
            &mut circuit,
            &provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumptions_by_component,
            },
        )
        .expect("split-batch GPU evidence should update incremental circuit");

    assert_eq!(updates.len(), 2);
    assert_eq!(updates[0].mode, CircuitUpdateMode::IncrementalEvidence);
    assert_eq!(updates[1].mode, CircuitUpdateMode::IncrementalEvidence);
    assert_eq!(updates[0].compile_count, 1);
    assert_eq!(updates[1].compile_count, 1);
    assert_eq!(updates[0].circuit_fingerprint, original_fingerprint);
    assert_eq!(updates[1].circuit_fingerprint, original_fingerprint);
    assert_eq!(circuit.incremental_update_count(), 2);
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec![
            accepted_left.evidence_literal(),
            accepted_right.evidence_literal(),
        ]
    );
    assert!(circuit.query_probability().within_tolerance(0.61));

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_max_evidence_arity_consumed, 1);
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 2);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 0);
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 2);
    assert_eq!(trace.accepted_incremental_circuit_updates, 2);
    assert_eq!(trace.accepted_gpu_production_path_events, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("split-batch incremental GPU evidence must not use CPU recomputation");
    trace
        .require_production_metric_eligibility()
        .expect_err("incremental circuit updates alone must not satisfy production metrics");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_applies_hidden_body_local_batch_evidence_to_incremental_circuit() {
    let Some(provider) = try_provider() else {
        return;
    };

    let batch = execute_hidden_body_local_split_batch(&provider);
    let empty_assumptions: &[xlog_prob::epistemic::EpistemicAssumption] = &[];
    let assumptions_by_component = [empty_assumptions, empty_assumptions];
    let left_1 = EpistemicAssumption::known_tuple(
        "left_gate",
        vec![EpistemicEvidenceTerm::integer(1)],
        true,
    );
    let left_3 = EpistemicAssumption::known_tuple(
        "left_gate",
        vec![EpistemicEvidenceTerm::integer(3)],
        true,
    );
    let left_4 = EpistemicAssumption::known_tuple(
        "left_gate",
        vec![EpistemicEvidenceTerm::integer(4)],
        true,
    );
    let right_5 = EpistemicAssumption::known_tuple(
        "right_blocked",
        vec![EpistemicEvidenceTerm::integer(5)],
        false,
    );
    let right_7 = EpistemicAssumption::known_tuple(
        "right_blocked",
        vec![EpistemicEvidenceTerm::integer(7)],
        false,
    );
    let right_8 = EpistemicAssumption::known_tuple(
        "right_blocked",
        vec![EpistemicEvidenceTerm::integer(8)],
        false,
    );
    let mut circuit = EpistemicCircuit::compile(
        0.25,
        vec![
            (left_1.clone(), 0.61),
            (left_3.clone(), 0.62),
            (left_4.clone(), 0.63),
            (right_5.clone(), 0.41),
            (right_7.clone(), 0.42),
            (right_8.clone(), 0.43),
        ],
        KnowledgeCompilerAdapter::gpu_d4(),
    )
    .expect("compile hidden split-batch incremental epistemic circuit");
    let original_fingerprint = circuit.circuit_fingerprint();
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());

    let updates = adapter
        .apply_accepted_world_views_to_circuit_for_gpu_batch_execution_result(
            &mut circuit,
            &provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumptions_by_component,
            },
        )
        .expect("hidden split-batch GPU evidence should update incremental circuit");

    assert_eq!(updates.len(), 2);
    assert_eq!(updates[0].mode, CircuitUpdateMode::IncrementalEvidence);
    assert_eq!(updates[1].mode, CircuitUpdateMode::IncrementalEvidence);
    assert_eq!(updates[0].compile_count, 1);
    assert_eq!(updates[1].compile_count, 1);
    assert_eq!(updates[0].circuit_fingerprint, original_fingerprint);
    assert_eq!(updates[1].circuit_fingerprint, original_fingerprint);
    assert_eq!(circuit.incremental_update_count(), 6);
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec![
            left_1.evidence_literal(),
            left_3.evidence_literal(),
            left_4.evidence_literal(),
            right_5.evidence_literal(),
            right_7.evidence_literal(),
            right_8.evidence_literal(),
        ]
    );
    assert!(circuit.query_probability().within_tolerance(0.61));

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 6);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        6
    );
    assert_eq!(trace.accepted_gpu_max_evidence_arity_consumed, 1);
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 2);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 2);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        1
    );
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 2);
    assert_eq!(trace.accepted_incremental_circuit_updates, 2);
    assert_eq!(trace.accepted_gpu_production_path_events, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("hidden split-batch incremental GPU evidence must not use CPU recomputation");
    trace
        .require_production_metric_eligibility()
        .expect_err("hidden incremental circuit updates alone must not satisfy production metrics");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_rejects_split_batch_with_mismatched_component_assumption_groups() {
    let Some(provider) = try_provider() else {
        return;
    };

    let batch = execute_accepted_split_batch(&provider);
    let empty_assumptions: &[xlog_prob::epistemic::EpistemicAssumption] = &[];
    let assumptions_by_component = [empty_assumptions];
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.6::left_base(7).
0.4::right_base(9).
query(left_base(7)).
query(right_base(9)).
"#;

    let err = match adapter.compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result(
        src,
        &provider,
        EpistemicProbGpuBatchExecutionEvidence {
            batch: &batch,
            assumptions_by_component: &assumptions_by_component,
        },
    ) {
        Ok(_) => {
            panic!("mismatched split-batch assumption groups must not gate GPU exact evaluation")
        }
        Err(err) => err,
    };

    let err = format!("{err}");
    assert!(err.contains("assumption group count 1"));
    assert!(err.contains("GPU batch component count 2"));
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 0);
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 0);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 0);
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 0);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 0);
    assert_eq!(trace.gpu_exact_source_compiles, 0);
    assert_eq!(trace.gpu_exact_query_evaluations, 0);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.accepted_gpu_production_path_events, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("rejected split-batch evidence must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect_err("rejected split-batch evidence must not satisfy conditioned prob metrics");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_gpu_exact_query_on_parsed_operator_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_parsed_all_operator_variable_bound_evidence(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.6::known_gate(7).
0.4::possible_gate(7).
0.3::not_known_gate(7).
0.2::not_possible_gate(7).
query(known_gate(7)).
query(possible_gate(7)).
query(not_known_gate(7)).
query(not_possible_gate(7)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("parsed all-operator runtime evidence should condition GPU exact evaluation");

    assert!((prob_of(&exact, "known_gate", &[Value::I64(7)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "possible_gate", &[Value::I64(7)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "not_known_gate", &[Value::I64(7)]) - 0.0).abs() < 1e-9);
    assert!((prob_of(&exact, "not_possible_gate", &[Value::I64(7)]) - 0.0).abs() < 1e-9);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        4
    );
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 4);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 4);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        2
    );
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        1
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 4);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_known_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_source_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 5);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("parsed all-operator conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("parsed all-operator evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_hidden_body_local_tuple_keys_from_gpu_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_parsed_body_local_tuple_key_evidence(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.2::gate(1).
0.3::gate(3).
0.4::gate(4).
0.5::gate(10).
0.6::gate(30).
query(gate(1)).
query(gate(3)).
query(gate(4)).
query(gate(10)).
query(gate(30)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("body-local tuple-key runtime evidence should condition GPU exact evaluation");

    assert!((prob_of(&exact, "gate", &[Value::I64(1)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "gate", &[Value::I64(3)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "gate", &[Value::I64(4)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "gate", &[Value::I64(10)]) - 0.5).abs() < 1e-9);
    assert!((prob_of(&exact, "gate", &[Value::I64(30)]) - 0.6).abs() < 1e-9);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        3
    );
    assert_eq!(trace.accepted_gpu_max_evidence_arity_consumed, 1);
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 1);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        0
    );
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        1
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 3);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 3);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 0);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 5);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("body-local conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("body-local evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_possible_hidden_body_local_tuple_keys_from_gpu_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_parsed_possible_body_local_tuple_key_evidence(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.2::maybe_gate(1).
0.3::maybe_gate(3).
0.4::maybe_gate(4).
0.5::maybe_gate(10).
0.6::maybe_gate(30).
query(maybe_gate(1)).
query(maybe_gate(3)).
query(maybe_gate(4)).
query(maybe_gate(10)).
query(maybe_gate(30)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect(
            "body-local possible tuple-key runtime evidence should condition GPU exact evaluation",
        );

    assert!((prob_of(&exact, "maybe_gate", &[Value::I64(1)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "maybe_gate", &[Value::I64(3)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "maybe_gate", &[Value::I64(4)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "maybe_gate", &[Value::I64(10)]) - 0.5).abs() < 1e-9);
    assert!((prob_of(&exact, "maybe_gate", &[Value::I64(30)]) - 0.6).abs() < 1e-9);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        3
    );
    assert_eq!(trace.accepted_gpu_max_evidence_arity_consumed, 1);
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 1);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        0
    );
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        1
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 3);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 3);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 0);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 5);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("body-local possible conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("body-local possible evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_zero_arity_output_hidden_body_local_tuple_keys_from_gpu_evidence()
{
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_parsed_zero_arity_body_local_tuple_key_evidence(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.2::gate(1).
0.3::gate(3).
0.4::gate(4).
0.5::gate(10).
query(gate(1)).
query(gate(3)).
query(gate(4)).
query(gate(10)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect(
            "zero-arity body-local tuple-key runtime evidence should condition GPU exact evaluation",
        );

    assert!((prob_of(&exact, "gate", &[Value::I64(1)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "gate", &[Value::I64(3)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "gate", &[Value::I64(4)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact, "gate", &[Value::I64(10)]) - 0.5).abs() < 1e-9);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        3
    );
    assert_eq!(trace.accepted_gpu_max_evidence_arity_consumed, 1);
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 1);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        0
    );
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        1
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 3);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 3);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 0);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 5);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("zero-arity body-local conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("zero-arity body-local evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_negated_hidden_body_local_tuple_keys_from_gpu_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_parsed_negated_body_local_tuple_key_evidence(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.2::blocked(5).
0.3::blocked(7).
0.4::blocked(8).
0.5::blocked(40).
0.6::blocked(60).
query(blocked(5)).
query(blocked(7)).
query(blocked(8)).
query(blocked(40)).
query(blocked(60)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect(
            "negated body-local tuple-key runtime evidence should condition GPU exact evaluation",
        );

    assert!((prob_of(&exact, "blocked", &[Value::I64(5)]) - 0.0).abs() < 1e-9);
    assert!((prob_of(&exact, "blocked", &[Value::I64(7)]) - 0.0).abs() < 1e-9);
    assert!((prob_of(&exact, "blocked", &[Value::I64(8)]) - 0.0).abs() < 1e-9);
    assert!((prob_of(&exact, "blocked", &[Value::I64(40)]) - 0.5).abs() < 1e-9);
    assert!((prob_of(&exact, "blocked", &[Value::I64(60)]) - 0.6).abs() < 1e-9);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        3
    );
    assert_eq!(trace.accepted_gpu_max_evidence_arity_consumed, 1);
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 1);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        1
    );
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        1
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 3);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 3);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_not_known_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 3);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 5);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("negated body-local conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("negated body-local evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_not_possible_hidden_body_local_tuple_keys_from_gpu_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_parsed_not_possible_body_local_tuple_key_evidence(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.2::maybe_blocked(5).
0.3::maybe_blocked(7).
0.4::maybe_blocked(8).
0.5::maybe_blocked(40).
0.6::maybe_blocked(60).
query(maybe_blocked(5)).
query(maybe_blocked(7)).
query(maybe_blocked(8)).
query(maybe_blocked(40)).
query(maybe_blocked(60)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect(
            "body-local not-possible tuple-key runtime evidence should condition GPU exact evaluation",
        );

    assert!((prob_of(&exact, "maybe_blocked", &[Value::I64(5)]) - 0.0).abs() < 1e-9);
    assert!((prob_of(&exact, "maybe_blocked", &[Value::I64(7)]) - 0.0).abs() < 1e-9);
    assert!((prob_of(&exact, "maybe_blocked", &[Value::I64(8)]) - 0.0).abs() < 1e-9);
    assert!((prob_of(&exact, "maybe_blocked", &[Value::I64(40)]) - 0.5).abs() < 1e-9);
    assert!((prob_of(&exact, "maybe_blocked", &[Value::I64(60)]) - 0.6).abs() < 1e-9);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        3
    );
    assert_eq!(trace.accepted_gpu_max_evidence_arity_consumed, 1);
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 1);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        1
    );
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        1
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 3);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 3);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_not_possible_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 3);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 5);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("body-local not-possible conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("body-local not-possible evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_gpu_exact_gradient_on_parsed_operator_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    let result = execute_parsed_all_operator_variable_bound_evidence(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.6::known_gate(7).
0.4::possible_gate(7).
0.3::not_known_gate(7).
0.2::not_possible_gate(7).
0.5::rain().
dry_when_known() :- known_gate(7), not rain().
query(dry_when_known()).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result(
            src,
            &provider,
            &result,
            Vec::new(),
        )
        .expect("parsed all-operator runtime evidence should condition GPU exact gradients");

    let dry = grad_of(&exact, "dry_when_known", &[]);
    assert!((dry.prob - 0.5).abs() < 1e-9);
    assert_eq!(dry.grad_true.len(), dry.grad_false.len());
    assert!(
        dry.grad_true
            .iter()
            .zip(dry.grad_false.iter())
            .any(|(grad_true, grad_false)| {
                (*grad_true + 0.5).abs() < 1e-9 && (*grad_false - 0.5).abs() < 1e-9
            }),
        "expected a rain gradient pair with true=-0.5 and false=0.5, got true={:?} false={:?}",
        dry.grad_true,
        dry.grad_false
    );

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        4
    );
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 4);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 4);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        2
    );
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        1
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 4);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 1);
    assert_eq!(trace.gpu_source_exact_gradient_evaluations, 1);
    assert_eq!(trace.gpu_source_conditioned_gradient_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.accepted_gpu_production_path_events, 6);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("parsed all-operator conditioned gradient path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("parsed all-operator gradient evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_gpu_exact_queries_for_split_runtime_batch() {
    let Some(provider) = try_provider() else {
        return;
    };

    let batch = execute_accepted_split_batch(&provider);
    let empty_assumptions: &[xlog_prob::epistemic::EpistemicAssumption] = &[];
    let assumptions_by_component = [empty_assumptions, empty_assumptions];
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.6::left_base(7).
0.4::right_base(9).
query(left_base(7)).
query(right_base(9)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result(
            src,
            &provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumptions_by_component,
            },
        )
        .expect("accepted split batch evidence should condition GPU exact evaluation");

    assert_eq!(exact.len(), 2);
    assert!((prob_of(&exact[0], "left_base", &[Value::I64(7)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact[0], "right_base", &[Value::I64(9)]) - 0.4).abs() < 1e-9);
    assert!((prob_of(&exact[1], "left_base", &[Value::I64(7)]) - 0.6).abs() < 1e-9);
    assert!((prob_of(&exact[1], "right_base", &[Value::I64(9)]) - 1.0).abs() < 1e-9);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 2);
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 2);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        2
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_source_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.accepted_gpu_production_path_events, 10);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("batch conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("split batch conditioned GPU exact evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_hidden_body_local_tuple_keys_for_split_runtime_batch() {
    let Some(provider) = try_provider() else {
        return;
    };

    let batch = execute_hidden_body_local_split_batch(&provider);
    let empty_assumptions: &[xlog_prob::epistemic::EpistemicAssumption] = &[];
    let assumptions_by_component = [empty_assumptions, empty_assumptions];
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.2::left_gate(1).
0.3::left_gate(3).
0.4::left_gate(4).
0.5::left_gate(10).
0.6::left_gate(30).
0.25::right_blocked(5).
0.35::right_blocked(7).
0.45::right_blocked(8).
0.55::right_blocked(40).
0.65::right_blocked(60).
query(left_gate(1)).
query(left_gate(3)).
query(left_gate(4)).
query(left_gate(10)).
query(left_gate(30)).
query(right_blocked(5)).
query(right_blocked(7)).
query(right_blocked(8)).
query(right_blocked(40)).
query(right_blocked(60)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result(
            src,
            &provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumptions_by_component,
            },
        )
        .expect("hidden body-local split batch evidence should condition GPU exact evaluation");

    assert_eq!(exact.len(), 2);
    assert!((prob_of(&exact[0], "left_gate", &[Value::I64(1)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact[0], "left_gate", &[Value::I64(3)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact[0], "left_gate", &[Value::I64(4)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact[0], "left_gate", &[Value::I64(10)]) - 0.5).abs() < 1e-9);
    assert!((prob_of(&exact[0], "left_gate", &[Value::I64(30)]) - 0.6).abs() < 1e-9);
    assert!((prob_of(&exact[0], "right_blocked", &[Value::I64(5)]) - 0.25).abs() < 1e-9);

    assert!((prob_of(&exact[1], "left_gate", &[Value::I64(1)]) - 0.2).abs() < 1e-9);
    assert!((prob_of(&exact[1], "right_blocked", &[Value::I64(5)]) - 0.0).abs() < 1e-9);
    assert!((prob_of(&exact[1], "right_blocked", &[Value::I64(7)]) - 0.0).abs() < 1e-9);
    assert!((prob_of(&exact[1], "right_blocked", &[Value::I64(8)]) - 0.0).abs() < 1e-9);
    assert!((prob_of(&exact[1], "right_blocked", &[Value::I64(40)]) - 0.55).abs() < 1e-9);
    assert!((prob_of(&exact[1], "right_blocked", &[Value::I64(60)]) - 0.65).abs() < 1e-9);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 6);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        6
    );
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 2);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 2);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        1
    );
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 2);
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 2);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        2
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 6);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 6);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 6);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 6);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_not_known_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 3);
    assert_eq!(trace.gpu_exact_source_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_source_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.accepted_gpu_production_path_events, 10);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("hidden body-local batch conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("hidden body-local batch evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_modal_hidden_body_local_tuple_keys_for_split_runtime_batch() {
    let Some(provider) = try_provider() else {
        return;
    };

    let batch = execute_modal_hidden_body_local_split_batch(&provider);
    let empty_assumptions: &[xlog_prob::epistemic::EpistemicAssumption] = &[];
    let assumptions_by_component = [empty_assumptions, empty_assumptions];
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.2::left_maybe(1).
0.3::left_maybe(3).
0.4::left_maybe(4).
0.5::left_maybe(10).
0.6::left_maybe(30).
0.25::right_maybe_blocked(5).
0.35::right_maybe_blocked(7).
0.45::right_maybe_blocked(8).
0.55::right_maybe_blocked(40).
0.65::right_maybe_blocked(60).
query(left_maybe(1)).
query(left_maybe(3)).
query(left_maybe(4)).
query(left_maybe(10)).
query(left_maybe(30)).
query(right_maybe_blocked(5)).
query(right_maybe_blocked(7)).
query(right_maybe_blocked(8)).
query(right_maybe_blocked(40)).
query(right_maybe_blocked(60)).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result(
            src,
            &provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumptions_by_component,
            },
        )
        .expect(
            "modal hidden body-local split batch evidence should condition GPU exact evaluation",
        );

    assert_eq!(exact.len(), 2);
    assert!((prob_of(&exact[0], "left_maybe", &[Value::I64(1)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact[0], "left_maybe", &[Value::I64(3)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact[0], "left_maybe", &[Value::I64(4)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact[0], "left_maybe", &[Value::I64(10)]) - 0.5).abs() < 1e-9);
    assert!((prob_of(&exact[0], "left_maybe", &[Value::I64(30)]) - 0.6).abs() < 1e-9);
    assert!((prob_of(&exact[0], "right_maybe_blocked", &[Value::I64(5)]) - 0.25).abs() < 1e-9);

    assert!((prob_of(&exact[1], "left_maybe", &[Value::I64(1)]) - 0.2).abs() < 1e-9);
    assert!((prob_of(&exact[1], "right_maybe_blocked", &[Value::I64(5)]) - 0.0).abs() < 1e-9);
    assert!((prob_of(&exact[1], "right_maybe_blocked", &[Value::I64(7)]) - 0.0).abs() < 1e-9);
    assert!((prob_of(&exact[1], "right_maybe_blocked", &[Value::I64(8)]) - 0.0).abs() < 1e-9);
    assert!((prob_of(&exact[1], "right_maybe_blocked", &[Value::I64(40)]) - 0.55).abs() < 1e-9);
    assert!((prob_of(&exact[1], "right_maybe_blocked", &[Value::I64(60)]) - 0.65).abs() < 1e-9);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 6);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        6
    );
    assert_eq!(trace.accepted_gpu_tuple_key_column_reads_consumed, 2);
    assert_eq!(trace.accepted_gpu_final_tuple_row_filters_consumed, 2);
    assert_eq!(
        trace.accepted_gpu_final_tuple_negated_row_filters_consumed,
        1
    );
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 2);
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 2);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        2
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 6);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 6);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 6);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 6);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_not_possible_evidence_facts, 3);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 3);
    assert_eq!(trace.gpu_exact_source_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_source_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.accepted_gpu_production_path_events, 10);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("modal body-local batch conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect("modal body-local batch evidence should satisfy the stricter prob gate");
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_parsed_program_for_split_runtime_batch() {
    let Some(provider) = try_provider() else {
        return;
    };

    let batch = execute_accepted_split_batch(&provider);
    let empty_assumptions: &[xlog_prob::epistemic::EpistemicAssumption] = &[];
    let assumptions_by_component = [empty_assumptions, empty_assumptions];
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.6::left_base(7).
0.4::right_base(9).
query(left_base(7)).
query(right_base(9)).
"#;
    let program = parse_program(src).expect("parse probability program");

    let exact = adapter
        .compile_and_evaluate_conditioned_program_for_gpu_batch_execution_result(
            &program,
            &provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumptions_by_component,
            },
        )
        .expect(
            "accepted split batch evidence should condition parsed-program GPU exact evaluation",
        );

    assert_eq!(exact.len(), 2);
    assert!((prob_of(&exact[0], "left_base", &[Value::I64(7)]) - 1.0).abs() < 1e-9);
    assert!((prob_of(&exact[0], "right_base", &[Value::I64(9)]) - 0.4).abs() < 1e-9);
    assert!((prob_of(&exact[1], "left_base", &[Value::I64(7)]) - 0.6).abs() < 1e-9);
    assert!((prob_of(&exact[1], "right_base", &[Value::I64(9)]) - 1.0).abs() < 1e-9);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 2);
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 2);
    assert_eq!(
        trace.accepted_program_conditioned_world_view_evidence_consumed,
        2
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_program_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(
        trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        2
    );
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_program_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_program_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_program_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_program_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.accepted_gpu_production_path_events, 10);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("parsed-program batch conditioned exact path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect(
            "split batch conditioned parsed-program GPU exact evidence should satisfy the stricter prob gate",
        );
}

#[test]
#[cfg(feature = "host-io")]
fn public_prob_adapter_conditions_gpu_exact_gradients_for_split_runtime_batch() {
    let Some(provider) = try_provider() else {
        return;
    };

    let batch = execute_accepted_split_batch(&provider);
    let empty_assumptions: &[xlog_prob::epistemic::EpistemicAssumption] = &[];
    let assumptions_by_component = [empty_assumptions, empty_assumptions];
    let mut adapter = EpistemicProbProductionAdapter::new(GpuConfig::default());
    let src = r#"
0.6::left_base(7).
0.4::right_base(9).
0.5::rain().
dry_left() :- left_base(7), not rain().
dry_right() :- right_base(9), not rain().
query(dry_left()).
query(dry_right()).
"#;

    let exact = adapter
        .compile_and_evaluate_conditioned_source_with_grads_for_gpu_batch_execution_result(
            src,
            &provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumptions_by_component,
            },
        )
        .expect("accepted split batch evidence should condition GPU exact gradients");

    assert_eq!(exact.len(), 2);
    let left_component_dry_left = grad_of(&exact[0], "dry_left", &[]);
    let left_component_dry_right = grad_of(&exact[0], "dry_right", &[]);
    let right_component_dry_left = grad_of(&exact[1], "dry_left", &[]);
    let right_component_dry_right = grad_of(&exact[1], "dry_right", &[]);
    assert!((left_component_dry_left.prob - 0.5).abs() < 1e-9);
    assert!((left_component_dry_right.prob - 0.2).abs() < 1e-9);
    assert!((right_component_dry_left.prob - 0.3).abs() < 1e-9);
    assert!((right_component_dry_right.prob - 0.5).abs() < 1e-9);
    assert!(
        left_component_dry_left
            .grad_true
            .iter()
            .zip(left_component_dry_left.grad_false.iter())
            .any(|(grad_true, grad_false)| {
                (*grad_true + 0.5).abs() < 1e-9 && (*grad_false - 0.5).abs() < 1e-9
            }),
        "expected left component dry_left to expose rain gradient true=-0.5 false=0.5, got true={:?} false={:?}",
        left_component_dry_left.grad_true,
        left_component_dry_left.grad_false
    );
    assert!(
        right_component_dry_right
            .grad_true
            .iter()
            .zip(right_component_dry_right.grad_false.iter())
            .any(|(grad_true, grad_false)| {
                (*grad_true + 0.5).abs() < 1e-9 && (*grad_false - 0.5).abs() < 1e-9
            }),
        "expected right component dry_right to expose rain gradient true=-0.5 false=0.5, got true={:?} false={:?}",
        right_component_dry_right.grad_true,
        right_component_dry_right.grad_false
    );

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(
        trace.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 2);
    assert_eq!(trace.accepted_conditioned_world_view_evidence_consumed, 2);
    assert_eq!(
        trace.accepted_source_conditioned_world_view_evidence_consumed,
        2
    );
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 2);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(trace.gpu_source_exact_gradient_evaluations, 2);
    assert_eq!(trace.gpu_source_conditioned_gradient_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.accepted_gpu_production_path_events, 12);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    trace
        .require_zero_cpu_recompute()
        .expect("batch conditioned gradient path must not use CPU recomputation");
    trace
        .require_conditioned_evidence_metric_eligibility()
        .expect(
            "split batch conditioned GPU gradient evidence should satisfy the stricter prob gate",
        );
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
    assert_eq!(
        result.prepared.preflight.epistemic_mode,
        EirEpistemicMode::Faeel
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    result
}

#[cfg(feature = "host-io")]
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
    .expect("parse empty-output probability evidence program");
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

fn execute_accepted_variable_bound_literal(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    execute_accepted_variable_bound_literal_with_base(provider, 7)
}

fn execute_accepted_variable_bound_literal_with_base(
    provider: &Arc<CudaKernelProvider>,
    value: u32,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let mut executor = Executor::new(provider.clone());
    executor.register_relation(RelId(1), "base");
    executor.put_relation("base", upload_unary_u32(provider, &[value], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &accepted_variable_bound_literal_executable(),
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 4,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("real runtime accepted variable-bound GPU epistemic execution");

    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result
            .final_tuple_materialization
            .row_specific_membership_row_capacity,
        1
    );
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    result
}

#[cfg(feature = "host-io")]
fn execute_accepted_binary_bound_literal(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        pred seed(u32, u32).
        pred gate(u32, u32).
        pred out(u32, u32).

        out(X, Y) :- seed(X, Y), know gate(X, Y).
        "#,
    )
    .expect("parse binary accepted probability evidence program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile binary epistemic GPU plan");
    let mut executor = Executor::new(provider.clone());

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "seed",
        upload_binary_u32(provider, &[(1, 2), (1, 3), (2, 4)], "x", "y"),
    );
    executor.put_relation(
        "gate",
        upload_binary_u32(provider, &[(1, 2), (2, 4)], "x", "y"),
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
        .expect("real runtime accepted binary-bound GPU epistemic execution");

    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        2
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
    result
}

#[cfg(feature = "host-io")]
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
    .expect("parse quaternary accepted probability evidence program");
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
                max_models_per_reduction: 3,
            },
        )
        .expect("real runtime accepted quaternary-bound GPU epistemic execution");

    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result
            .final_tuple_materialization
            .row_specific_membership_row_capacity,
        3
    );
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    result
}

#[cfg(feature = "host-io")]
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
    .expect("parse G91 accepted probability evidence program");
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
        .expect("real runtime accepted G91 possible GPU epistemic execution");

    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(
        result.prepared.preflight.epistemic_mode,
        EirEpistemicMode::G91
    );
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
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
    result
}

#[cfg(feature = "host-io")]
fn execute_accepted_symbol_variable_bound_literal(
    provider: &Arc<CudaKernelProvider>,
    alpha: u32,
    beta: u32,
    gamma: u32,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        pred seed(symbol).
        pred gate(symbol).
        pred out(symbol).

        out(X) :- seed(X), know gate(X).
        "#,
    )
    .expect("parse symbol accepted probability evidence program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile symbol epistemic GPU plan");
    let mut executor = Executor::new(provider.clone());

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
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
        .expect("real runtime accepted symbol-bound GPU epistemic execution");

    assert_eq!(
        result.final_output.schema().column_type(0),
        Some(ScalarType::Symbol)
    );
    assert_eq!(result.final_result_transfer.final_output_rows, 2);
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
    result
}

#[cfg(feature = "host-io")]
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
    .expect("parse all-operator epistemic probability evidence program");
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
        .expect("parsed all-operator runtime evidence should execute through GPU");

    assert_eq!(result.final_result_transfer.final_output_rows, 1);
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
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    result
}

#[cfg(feature = "host-io")]
fn execute_parsed_body_local_tuple_key_evidence(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        pred edge(u32, u32).
        pred gate(u32).
        pred out(u32).

        out(Y) :- edge(X, Y), know gate(X).
        "#,
    )
    .expect("parse body-local epistemic probability evidence program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile body-local epistemic GPU plan");
    let mut executor = Executor::new(provider.clone());

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "edge",
        upload_binary_u32(provider, &[(1, 10), (2, 20), (3, 30), (4, 10)], "x", "y"),
    );
    executor.put_relation("gate", upload_unary_u32(provider, &[1, 3, 4], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("body-local runtime evidence should execute through GPU");

    assert_eq!(result.output.arity(), 2);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        0
    );
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    result
}

#[cfg(feature = "host-io")]
fn execute_parsed_possible_body_local_tuple_key_evidence(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        pred edge(u32, u32).
        pred maybe_gate(u32).
        pred out(u32).

        out(Y) :- edge(X, Y), possible maybe_gate(X).
        "#,
    )
    .expect("parse body-local possible epistemic probability evidence program");
    let executable = compile_epistemic_gpu_execution(&program)
        .expect("compile body-local possible epistemic GPU plan");
    let mut executor = Executor::new(provider.clone());

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "edge",
        upload_binary_u32(provider, &[(1, 10), (2, 20), (3, 30), (4, 10)], "x", "y"),
    );
    executor.put_relation("maybe_gate", upload_unary_u32(provider, &[1, 3, 4], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("body-local possible runtime evidence should execute through GPU");

    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.output.arity(), 2);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        0
    );
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    result
}

#[cfg(feature = "host-io")]
fn execute_parsed_zero_arity_body_local_tuple_key_evidence(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        pred edge(u32).
        pred gate(u32).
        pred out().

        out() :- edge(X), know gate(X).
        "#,
    )
    .expect("parse zero-arity body-local epistemic probability evidence program");
    let executable = compile_epistemic_gpu_execution(&program)
        .expect("compile zero-arity body-local epistemic GPU plan");
    let mut executor = Executor::new(provider.clone());

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation("edge", upload_unary_u32(provider, &[1, 3, 4], "x"));
    executor.put_relation("gate", upload_unary_u32(provider, &[1, 3, 4], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("zero-arity body-local runtime evidence should execute through GPU");

    assert_eq!(result.output.arity(), 1);
    assert_eq!(result.final_output.arity(), 0);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(result.final_result_transfer.final_output_payload_bytes, 0);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        0
    );
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    result
}

#[cfg(feature = "host-io")]
fn execute_parsed_negated_body_local_tuple_key_evidence(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        pred edge(u32, u32).
        pred blocked(u32).
        pred out(u32).

        out(Y) :- edge(X, Y), not know blocked(X).
        "#,
    )
    .expect("parse negated body-local epistemic probability evidence program");
    let executable = compile_epistemic_gpu_execution(&program)
        .expect("compile negated body-local epistemic GPU plan");
    let mut executor = Executor::new(provider.clone());

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "edge",
        upload_binary_u32(provider, &[(5, 40), (6, 50), (7, 60), (8, 40)], "x", "y"),
    );
    executor.put_relation("blocked", upload_unary_u32(provider, &[6], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("negated body-local runtime evidence should execute through GPU");

    assert_eq!(result.output.arity(), 2);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    result
}

#[cfg(feature = "host-io")]
fn execute_parsed_not_possible_body_local_tuple_key_evidence(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        pred edge(u32, u32).
        pred maybe_blocked(u32).
        pred out(u32).

        out(Y) :- edge(X, Y), not possible maybe_blocked(X).
        "#,
    )
    .expect("parse not-possible body-local epistemic probability evidence program");
    let executable = compile_epistemic_gpu_execution(&program)
        .expect("compile not-possible body-local epistemic GPU plan");
    let mut executor = Executor::new(provider.clone());

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "edge",
        upload_binary_u32(provider, &[(5, 40), (6, 50), (7, 60), (8, 40)], "x", "y"),
    );
    executor.put_relation("maybe_blocked", upload_unary_u32(provider, &[6], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("not-possible body-local runtime evidence should execute through GPU");

    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(result.output.arity(), 2);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    result
}

#[cfg(feature = "host-io")]
fn execute_parsed_binary_operator_evidence(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        pred seed(u32, u32).
        pred known_gate(u32, u32).
        pred possible_gate(u32, u32).
        pred missing_known_gate(u32, u32).
        pred missing_possible_gate(u32, u32).
        pred out(u32, u32).

        out(X, Y) :- seed(X, Y), know known_gate(X, Y), possible possible_gate(X, Y),
                     not know missing_known_gate(X, Y),
                     not possible missing_possible_gate(X, Y).
        "#,
    )
    .expect("parse binary all-operator epistemic probability evidence program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile binary epistemic GPU plan");
    let mut executor = Executor::new(provider.clone());

    for (name, rows) in [
        ("seed", &[(1, 2)][..]),
        ("known_gate", &[(1, 2)][..]),
        ("possible_gate", &[(1, 2)][..]),
        ("missing_known_gate", &[(1, 3)][..]),
        ("missing_possible_gate", &[(2, 2)][..]),
    ] {
        let rel = *executable
            .relation_ids
            .get(name)
            .unwrap_or_else(|| panic!("compiled plan should expose relation id for {name}"));
        executor.register_relation(rel, name);
        executor.put_relation(name, upload_binary_u32(provider, rows, "x", "y"));
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
        .expect("binary all-operator runtime evidence should execute through GPU");

    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        8
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 4);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        2
    );
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![15]);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    result
}

#[cfg(feature = "host-io")]
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
        .require_trace_matches_components("prob accepted split batch")
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

#[cfg(feature = "host-io")]
fn execute_hidden_body_local_split_batch(
    provider: &Arc<CudaKernelProvider>,
) -> EpistemicGpuBatchExecutionResult {
    let program = parse_program(
        r#"
        pred left_edge(u32, u32).
        pred left_gate(u32).
        pred left_out(u32).
        pred right_edge(u32, u32).
        pred right_blocked(u32).
        pred right_out(u32).

        left_out(Y) :- left_edge(X, Y), know left_gate(X).
        right_out(Y) :- right_edge(X, Y), not know right_blocked(X).
        "#,
    )
    .expect("parse split body-local probability evidence program");
    let split = compile_epistemic_gpu_split_execution(&program)
        .expect("compile split body-local probability evidence program");
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
    executor.put_relation("left_gate", upload_unary_u32(provider, &[1, 3, 4], "x"));
    executor.put_relation(
        "right_edge",
        upload_binary_u32(provider, &[(5, 40), (6, 50), (7, 60), (8, 40)], "x", "y"),
    );
    executor.put_relation("right_blocked", upload_unary_u32(provider, &[6], "x"));

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
        .expect("split body-local probability components should execute through GPU batch");

    assert_eq!(batch.results.len(), 2);
    batch
        .require_trace_matches_components("prob hidden body-local split batch")
        .expect("hidden body-local split batch trace must match real component results");
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
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.results[0].final_output.arity(), 1);
    assert_eq!(batch.results[0].output.arity(), 2);
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
    assert_eq!(batch.results[1].final_output.arity(), 1);
    assert_eq!(batch.results[1].output.arity(), 2);
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

#[cfg(feature = "host-io")]
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
    .expect("parse modal split body-local probability evidence program");
    let split = compile_epistemic_gpu_split_execution(&program)
        .expect("compile modal split body-local probability evidence program");
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
        .expect("modal split body-local probability components should execute through GPU batch");

    assert_eq!(batch.results.len(), 2);
    batch
        .require_trace_matches_components("prob modal hidden body-local split batch")
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
    assert_eq!(batch.results[1].final_output.arity(), 1);
    assert_eq!(batch.results[1].output.arity(), 2);
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
            public_head_arity: 1,
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

fn accepted_variable_bound_literal_executable() -> EpistemicExecutablePlan {
    let mut gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![EirEpistemicLiteral {
            op: EirEpistemicOp::Know,
            negated: false,
            atom: EirAtom {
                predicate: "base".to_string(),
                arity: 1,
                terms: vec![EirTerm::Variable("X".to_string())],
            },
        }],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "out".to_string(),
            public_head_arity: 1,
            relational_body_atoms: 1,
            wcoj_status: EpistemicWcojReductionStatus::NotWcojCandidate,
        }],
    );
    gpu_plan.tuple_membership_bindings[0].bound_output_columns[0] = Some(0);

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: BTreeMap::new(),
        reduced_runtime_plan: scan_base_into_out_plan(),
    }
}

#[cfg(feature = "host-io")]
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
            public_head_arity: 1,
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

#[cfg(feature = "host-io")]
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

#[cfg(feature = "host-io")]
fn prob_of(result: &xlog_prob::exact::ExactResult, predicate: &str, args: &[Value]) -> f64 {
    result
        .query_probs
        .iter()
        .find(|query| query.atom.predicate == predicate && query.atom.args == args)
        .unwrap_or_else(|| panic!("missing query result for {predicate} with args {args:?}"))
        .prob
}

#[cfg(feature = "host-io")]
fn grad_of<'a>(
    result: &'a xlog_prob::exact::ExactResultWithGrads,
    predicate: &str,
    args: &[Value],
) -> &'a xlog_prob::exact::QueryGradients {
    result
        .query_grads
        .iter()
        .find(|query| query.atom.predicate == predicate && query.atom.args == args)
        .unwrap_or_else(|| panic!("missing query gradients for {predicate} with args {args:?}"))
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

#[cfg(feature = "host-io")]
fn upload_binary_u32(
    provider: &Arc<CudaKernelProvider>,
    rows: &[(u32, u32)],
    name_a: &str,
    name_b: &str,
) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = rows.len() * std::mem::size_of::<u32>();
    let memory = provider.memory();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let bytes0: Vec<u8> = rows
        .iter()
        .flat_map(|(left, _)| left.to_le_bytes())
        .collect();
    let bytes1: Vec<u8> = rows
        .iter()
        .flat_map(|(_, right)| right.to_le_bytes())
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

#[cfg(feature = "host-io")]
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
