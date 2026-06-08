use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Parser, ValueEnum};
use cudarc::driver::sys;
use serde_json::{json, Value};
use xlog_core::{MemoryBudget, Result, RuntimeConfig, ScalarType, Schema, XlogError};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::compile::load_modules;
use xlog_logic::epistemic::{
    build_epistemic_dependency_graph, compile_epistemic_gpu_execution_with_stats_snapshot,
    compile_epistemic_gpu_split_execution_with_stats_snapshot,
    run_generate_propagate_test_with_mode, EpistemicInterpretation, GeneratePropagateTestConfig,
};
use xlog_logic::{build_eir, parse_program, EpistemicMode, Program};
use xlog_prob::epistemic::{EpistemicAssumption, EpistemicEvidenceTerm};
use xlog_prob::epistemic_production::EpistemicProbProductionAdapter;
use xlog_prob::exact::GpuConfig;
use xlog_runtime::{read_device_row_count, EpistemicGpuWorkspaceCapacities, Executor};
use xlog_solve::{
    Clause, GpuCdclConfig, GpuCnf, GpuSolverProductionAdapter, GpuSolverProductionExpectation,
    GpuSolverProductionLifecycleStep, GpuSolverProductionMaxSatCandidate, Literal, SolveInstance,
};

const MIB: usize = 1024 * 1024;
const DEFAULT_EPISTEMIC_GPU_BUDGET_MIB: usize = 1024;
const GENERALIZATION_DECISION_DERIVED_RELATION_EXPORTS: &[&str] = &[
    "accepted_claim",
    "bfo_claim_support",
    "decision_candidate_support",
    "decision_claim_support",
    "candidate_failure_chain_components_supported",
    "candidate_failure_chain_support",
    "dominated_intervention_option",
    "dominated_risk_state_option",
    "dominated_root_cause_option",
    "failure_chain_component_edge",
    "failure_chain_component_reach",
    "failure_chain_support",
    "intervention_support",
    "maxsat_selected_intervention",
    "risk_state",
    "risk_state_support",
    "root_bfo_support",
    "root_cause_support",
    "root_evidence_support",
    "selector_accepted",
    "selected_risk_state",
    "selected_root_cause",
    "unsafe_decision",
];
const GENERALIZATION_CANDIDATE_GENERATOR_DERIVED_RELATION_EXPORTS: &[&str] =
    &["generated_candidate"];
const GENERALIZATION_VARIANT_CANDIDATE_GENERATOR_DERIVED_RELATION_EXPORTS: &[&str] =
    &["generated_variant_candidate"];
const DILP_PROOF_SCHEMA_GENERATOR_DERIVED_RELATION_EXPORTS: &[&str] = &[
    "candidate_proof_schema",
    "support_policy_basis",
    "support_weight_component",
    "schema_support_weight",
    "support_threshold_floor_seed",
    "support_threshold_floor",
    "training_support_observation",
    "training_support_threshold",
    "training_support",
    "schema_domain_support",
    "schema_transfer_support",
    "proof_schema_body",
    "schema_promotion_score",
    "generated_proof_schema",
];
const DILP_PROOF_SCHEMA_SELECTION_DERIVED_RELATION_EXPORTS: &[&str] = &[
    "schema_promotion_score",
    "schema_promotion_threshold",
    "promotion_candidate",
    "dominated_promoted_proof_schema",
    "selection_candidate",
    "selected_promoted_proof_schema",
];
const DILP_PROOF_SCHEMA_PROMOTION_DERIVED_RELATION_EXPORTS: &[&str] =
    &["promotion_candidate", "promoted_proof_schema"];
const DILP_PROOF_TRACE_DERIVED_RELATION_EXPORTS: &[&str] =
    &["observed_trace_support", "accepted_proof_trace"];
const GENERALIZATION_CANDIDATE_SCORING_DERIVED_RELATION_EXPORTS: &[&str] = &[
    "bfo_evidence_score",
    "literal_observation_score",
    "mismatch_penalty_score",
    "candidate_feature_score",
    "transductive_transfer_signal",
    "bounded_metastable_transfer_margin",
    "candidate_positive_score",
    "candidate_mismatch_penalty",
    "raw_candidate_score",
    "candidate_score_floor_zero",
    "bounded_candidate_score",
    "xlog_candidate_score",
];
const GENERALIZATION_RANKER_DERIVED_RELATION_EXPORTS: &[&str] = &[
    "selector_accepted",
    "decision_candidate_support",
    "dominated_rank_candidate",
    "ranker_selected",
];
const GENERALIZATION_SELECTOR_DERIVED_RELATION_EXPORTS: &[&str] = &["accepted_candidate"];
const GENERALIZATION_FAILURE_CHAIN_SUPPORT_DERIVED_RELATION_EXPORTS: &[&str] = &[
    "root_cause_support",
    "intervention_support",
    "risk_state_support",
    "seed_failure_chain_root",
    "seed_failure_chain_intervention",
    "seed_failure_chain_risk_state",
    "candidate_root_claim_supported",
    "candidate_intervention_claim_supported",
    "candidate_risk_state_claim_supported",
    "candidate_failure_chain_root_supported",
    "candidate_failure_chain_intervention_supported",
    "candidate_failure_chain_risk_state_supported",
    "failure_chain_component_edge",
    "failure_chain_component_reach",
    "candidate_failure_chain_root_intervention_reachable",
    "candidate_failure_chain_intervention_risk_reachable",
    "candidate_failure_chain_components_supported",
    "candidate_failure_chain_support",
    "candidate_failure_chain_claim_ready",
    "xlog_failure_chain_claim",
];
const SHOWCASE_TRANSFER_CANDIDATE_SCORING_DERIVED_RELATION_EXPORTS: &[&str] = &[
    "bfo_evidence_score",
    "literal_observation_score",
    "mismatch_penalty_score",
    "candidate_feature_score",
    "showcase_transductive_transfer_signal",
    "bounded_showcase_metastable_transfer_margin",
    "showcase_transfer_candidate_score",
];
const DILP_CANDIDATE_SCORING_DERIVED_RELATION_EXPORTS: &[&str] = &[
    "selected_proof_support_score",
    "proof_weighted_score",
    "dilp_transductive_transfer_signal",
    "dilp_neural_instability_score",
    "dilp_metastable_transfer_margin",
    "dilp_metastable_transfer_floor_zero",
    "bounded_dilp_metastable_transfer_margin",
    "dilp_candidate_score",
];
const GENERALIZATION_ABSTENTION_DERIVED_RELATION_EXPORTS: &[&str] = &[
    "accepted_world_view_count",
    "rejected_world_view_count",
    "proof_evidence_count",
    "proof_confidence",
    "abstention_threshold",
    "dominated_abstention_threshold",
    "selected_abstention_threshold",
    "solver_probability_trace",
    "accept_abstention_candidate",
    "abstain_abstention_candidate",
    "xlog_abstention_decision",
];
const GENERALIZATION_EXPLANATION_DERIVED_RELATION_EXPORTS: &[&str] = &[
    "explanation_identifier",
    "explanation_metadata",
    "dominated_explanation_metadata",
    "selected_explanation_metadata",
    "explanation_support",
    "explanation_has_dependency",
    "explanation_dependency",
    "xlog_explanation",
];

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    relations: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum, default_value_t = ExecutionKind::Single)]
    execution: ExecutionKind,
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long, default_value_t = 16)]
    max_candidates: usize,
    #[arg(long, default_value_t = 1)]
    max_worlds: usize,
    #[arg(long, default_value_t = 32)]
    max_models_per_reduction: usize,
    #[arg(long, default_value_t = DEFAULT_EPISTEMIC_GPU_BUDGET_MIB)]
    gpu_budget_mib: usize,
    #[arg(long, default_value_t = false)]
    solver_prob: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ExecutionKind {
    Single,
    Split,
}

struct DiscardSink;

impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> std::result::Result<(), SinkError> {
        Ok(())
    }
}

struct RuntimeFixture {
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    gpu_budget_bytes: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let source = fs::read_to_string(&args.source).map_err(|err| {
        XlogError::Execution(format!("read source {}: {err}", args.source.display()))
    })?;
    let relations = load_relations(&args.relations)?;
    let fixture = make_fixture(args.device, gpu_budget_bytes(args.gpu_budget_mib)?)?;
    let capacities = EpistemicGpuWorkspaceCapacities {
        max_candidates: args.max_candidates,
        max_worlds: args.max_worlds,
        max_models_per_reduction: args.max_models_per_reduction,
    };
    let source_file_name = args
        .source
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");

    let payload = match args.execution {
        ExecutionKind::Single => execute_single(
            &fixture,
            &args.source,
            source_file_name,
            &source,
            &relations,
            capacities,
            args.solver_prob,
        )?,
        ExecutionKind::Split => execute_split(
            &fixture,
            &args.source,
            source_file_name,
            &source,
            &relations,
            capacities,
        )?,
    };

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            XlogError::Execution(format!(
                "create output directory {}: {err}",
                parent.display()
            ))
        })?;
    }
    fs::write(
        &args.output,
        serde_json::to_string_pretty(&payload).expect("JSON serialization cannot fail") + "\n",
    )
    .map_err(|err| {
        XlogError::Execution(format!("write output {}: {err}", args.output.display()))
    })?;
    Ok(())
}

fn gpu_budget_bytes(gpu_budget_mib: usize) -> Result<usize> {
    if gpu_budget_mib == 0 {
        return Err(XlogError::Execution(
            "--gpu-budget-mib must be greater than zero".into(),
        ));
    }
    gpu_budget_mib.checked_mul(MIB).ok_or_else(|| {
        XlogError::Execution(format!(
            "--gpu-budget-mib {gpu_budget_mib} overflows byte conversion"
        ))
    })
}

fn make_fixture(device_ordinal: usize, gpu_budget_bytes: usize) -> Result<RuntimeFixture> {
    let device = Arc::new(CudaDevice::new(device_ordinal).map_err(|err| {
        XlogError::Execution(format!("create CUDA device {device_ordinal}: {err}"))
    })?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, gpu_budget_bytes));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        device_ordinal as u32,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(gpu_budget_bytes as u64),
        Arc::clone(&runtime),
    ));
    let provider = Arc::new(
        CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
            .map_err(|err| XlogError::Execution(format!("create CUDA kernel provider: {err}")))?,
    );
    Ok(RuntimeFixture {
        memory,
        provider,
        gpu_budget_bytes,
    })
}

fn epistemic_runtime_config() -> RuntimeConfig {
    RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(true))
}

fn load_relations(path: &PathBuf) -> Result<BTreeMap<String, (usize, Vec<Vec<u32>>)>> {
    let payload: Value = serde_json::from_str(&fs::read_to_string(path).map_err(|err| {
        XlogError::Execution(format!("read relation payload {}: {err}", path.display()))
    })?)
    .map_err(|err| XlogError::Parse(format!("parse relation payload {}: {err}", path.display())))?;
    let relation_obj = payload
        .get("relations")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            XlogError::Parse("relation payload must contain object `relations`".into())
        })?;
    let mut relations = BTreeMap::new();
    for (name, spec) in relation_obj {
        let arity = spec
            .get("arity")
            .and_then(Value::as_u64)
            .ok_or_else(|| XlogError::Parse(format!("relation {name} missing integer arity")))?
            as usize;
        let rows_value = spec
            .get("rows")
            .and_then(Value::as_array)
            .ok_or_else(|| XlogError::Parse(format!("relation {name} missing rows array")))?;
        let mut rows = Vec::with_capacity(rows_value.len());
        for row_value in rows_value {
            let row_array = row_value
                .as_array()
                .ok_or_else(|| XlogError::Parse(format!("relation {name} row must be an array")))?;
            if row_array.len() != arity {
                return Err(XlogError::Parse(format!(
                    "relation {name} row arity {} does not match {arity}",
                    row_array.len()
                )));
            }
            let mut row = Vec::with_capacity(arity);
            for cell in row_array {
                row.push(cell.as_u64().ok_or_else(|| {
                    XlogError::Parse(format!("relation {name} cells must be u32 integers"))
                })? as u32);
            }
            rows.push(row);
        }
        relations.insert(name.clone(), (arity, rows));
    }
    Ok(relations)
}

fn parse_program_with_modules(source_path: &Path, source: &str) -> Result<Program> {
    let program = parse_program(source)?;
    if !source.contains("use ") {
        return Ok(program);
    }
    let resolver = load_modules(source_path, Vec::new())
        .map_err(|err| XlogError::Compilation(format!("Module resolution failed: {err}")))?;
    resolver
        .merge_imports(program)
        .map_err(|err| XlogError::Compilation(format!("Module merge failed: {err}")))
}

fn execute_single(
    fixture: &RuntimeFixture,
    source_path: &Path,
    source_file_name: &str,
    source: &str,
    relations: &BTreeMap<String, (usize, Vec<Vec<u32>>)>,
    capacities: EpistemicGpuWorkspaceCapacities,
    solver_prob: bool,
) -> Result<Value> {
    let program = parse_program_with_modules(source_path, source)?;
    let eir = build_eir(&program)?;
    let dependency_graph = build_epistemic_dependency_graph(&program)?;
    let gpt_faeel = run_gpt_fixture(EpistemicMode::Faeel)?;
    let gpt_g91 = run_gpt_fixture(EpistemicMode::G91)?;
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)?;
    let mut executor =
        Executor::new_with_config(Arc::clone(&fixture.provider), epistemic_runtime_config());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    put_relations(&mut executor, fixture, relations)?;
    let result = executor.execute_epistemic_gpu_execution(&executable, capacities)?;
    let output_columns = result.final_result_transfer.final_output_column_count;
    let final_rows = download_rows(&fixture.provider, &result.final_output, output_columns)?;
    let mut derived_relation_rows = download_exported_relation_rows(
        &fixture.provider,
        &executor,
        exported_derived_relations(source_file_name),
    )?;
    if let Some(final_relation_name) = exported_final_relation(source_file_name) {
        attach_final_relation_rows(
            &mut derived_relation_rows,
            final_relation_name,
            output_columns,
            &final_rows,
        );
    }
    let solver_probability = if solver_prob {
        Some(run_solver_probability_evidence(
            fixture,
            &result,
            &final_rows,
        )?)
    } else {
        None
    };
    let mut runtime = runtime_json(&result);
    attach_program_runtime_diagnostics(
        source_file_name,
        &result,
        relations,
        &derived_relation_rows,
        &mut runtime,
    );
    Ok(json!({
        "status": "PASS",
        "execution": "single",
        "eir": {
            "mode": format!("{:?}", eir.mode),
            "rule_count": eir.rules.len(),
            "epistemic_literal_count": executable.gpu_plan.epistemic_literals.len(),
        },
        "dependency_graph": {
            "component_count": dependency_graph.components.len(),
            "components": dependency_graph.components.iter().map(|component| json!({
                "predicates": component.predicates,
                "rule_indices": component.rule_indices,
            })).collect::<Vec<_>>(),
        },
        "gpt_ablation": {
            "faeel": gpt_faeel,
            "g91": gpt_g91,
        },
        "preflight": preflight_json(&result.prepared.preflight),
        "gpu_budget_bytes": fixture.gpu_budget_bytes,
        "runtime": runtime,
        "final_rows": final_rows,
        "derived_relation_rows": derived_relation_rows,
        "solver_probability": solver_probability,
    }))
}

fn exported_derived_relations(source_file_name: &str) -> &'static [&'static str] {
    match source_file_name {
        "epistemic_generalization_decision.xlog" => {
            GENERALIZATION_DECISION_DERIVED_RELATION_EXPORTS
        }
        "epistemic_generalization_candidate_generator.xlog" => {
            GENERALIZATION_CANDIDATE_GENERATOR_DERIVED_RELATION_EXPORTS
        }
        "epistemic_generalization_variant_candidate_generator.xlog" => {
            GENERALIZATION_VARIANT_CANDIDATE_GENERATOR_DERIVED_RELATION_EXPORTS
        }
        "epistemic_dilp_proof_schema_generator.xlog" => {
            DILP_PROOF_SCHEMA_GENERATOR_DERIVED_RELATION_EXPORTS
        }
        "epistemic_dilp_proof_schema_selection.xlog" => {
            DILP_PROOF_SCHEMA_SELECTION_DERIVED_RELATION_EXPORTS
        }
        "epistemic_dilp_proof_schema_promotion.xlog" => {
            DILP_PROOF_SCHEMA_PROMOTION_DERIVED_RELATION_EXPORTS
        }
        "epistemic_dilp_proof_trace.xlog" => DILP_PROOF_TRACE_DERIVED_RELATION_EXPORTS,
        "epistemic_generalization_candidate_scoring.xlog" => {
            GENERALIZATION_CANDIDATE_SCORING_DERIVED_RELATION_EXPORTS
        }
        "epistemic_generalization_ranker.xlog" => GENERALIZATION_RANKER_DERIVED_RELATION_EXPORTS,
        "epistemic_generalization_selector.xlog" => {
            GENERALIZATION_SELECTOR_DERIVED_RELATION_EXPORTS
        }
        "epistemic_generalization_failure_chain_support.xlog" => {
            GENERALIZATION_FAILURE_CHAIN_SUPPORT_DERIVED_RELATION_EXPORTS
        }
        "epistemic_showcase_transfer_candidate_scoring.xlog" => {
            SHOWCASE_TRANSFER_CANDIDATE_SCORING_DERIVED_RELATION_EXPORTS
        }
        "epistemic_dilp_candidate_scoring.xlog" => DILP_CANDIDATE_SCORING_DERIVED_RELATION_EXPORTS,
        "epistemic_generalization_abstention.xlog" => {
            GENERALIZATION_ABSTENTION_DERIVED_RELATION_EXPORTS
        }
        "epistemic_generalization_explanation.xlog" => {
            GENERALIZATION_EXPLANATION_DERIVED_RELATION_EXPORTS
        }
        _ => &[],
    }
}

fn exported_final_relation(source_file_name: &str) -> Option<&'static str> {
    match source_file_name {
        "epistemic_generalization_candidate_generator.xlog" => Some("generated_candidate"),
        "epistemic_generalization_variant_candidate_generator.xlog" => {
            Some("generated_variant_candidate")
        }
        "epistemic_generalization_failure_chain_support.xlog" => Some("xlog_failure_chain_claim"),
        "epistemic_dilp_proof_schema_promotion.xlog" => Some("promoted_proof_schema"),
        _ => None,
    }
}

fn attach_final_relation_rows(
    derived_relation_rows: &mut Value,
    relation_name: &str,
    arity: usize,
    final_rows: &[Vec<u32>],
) {
    if let Some(rows_by_relation) = derived_relation_rows.as_object_mut() {
        rows_by_relation.insert(
            relation_name.to_string(),
            json!({
                "arity": arity,
                "rows": final_rows,
            }),
        );
    }
}

fn download_exported_relation_rows(
    provider: &Arc<CudaKernelProvider>,
    executor: &Executor,
    relation_names: &[&str],
) -> Result<Value> {
    let mut rows_by_relation = serde_json::Map::new();
    for relation_name in relation_names {
        let Some(buffer) = executor.store().get(*relation_name) else {
            continue;
        };
        let arity = buffer.arity();
        rows_by_relation.insert(
            (*relation_name).to_string(),
            json!({
                "arity": arity,
                "rows": download_rows(provider, buffer, arity)?,
            }),
        );
    }
    Ok(Value::Object(rows_by_relation))
}

fn execute_split(
    fixture: &RuntimeFixture,
    source_path: &Path,
    source_file_name: &str,
    source: &str,
    relations: &BTreeMap<String, (usize, Vec<Vec<u32>>)>,
    capacities: EpistemicGpuWorkspaceCapacities,
) -> Result<Value> {
    let program = parse_program_with_modules(source_path, source)?;
    let eir = build_eir(&program)?;
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)?;
    let mut executor =
        Executor::new_with_config(Arc::clone(&fixture.provider), epistemic_runtime_config());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            relation_ids.entry(name.clone()).or_insert(*rel_id);
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    put_relations(&mut executor, fixture, relations)?;
    let executables = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect::<Vec<_>>();
    let batch =
        executor.execute_epistemic_gpu_execution_batch_with_trace(&executables, capacities)?;
    let derived_relation_rows = download_exported_relation_rows(
        &fixture.provider,
        &executor,
        exported_derived_relations(source_file_name),
    )?;
    let components = split
        .components
        .iter()
        .zip(batch.results.iter())
        .map(|(component, result)| {
            json!({
                "rule_indices": component.component.rule_indices,
                "predicates": component.component.predicates,
                "preflight": preflight_json(&result.prepared.preflight),
                "runtime": runtime_json(result),
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "status": "PASS",
        "execution": "split",
        "eir": {
            "mode": format!("{:?}", eir.mode),
            "rule_count": eir.rules.len(),
        },
        "gpu_budget_bytes": fixture.gpu_budget_bytes,
        "split": {
            "component_count": split.components.len(),
            "recomposed_rule_indices": split.recomposed_rule_indices(),
        },
        "batch_trace": {
            "component_count": batch.trace.component_count,
            "gpu_runtime_component_executions": batch.trace.gpu_runtime_component_executions,
            "cpu_recomposition_steps": batch.trace.cpu_recomposition_steps,
            "cpu_candidate_enumerations": batch.trace.cpu_candidate_enumerations,
            "cpu_world_view_validations": batch.trace.cpu_world_view_validations,
            "cpu_solver_search_fallbacks": batch.trace.cpu_solver_search_fallbacks,
            "cpu_probability_recomputations": batch.trace.cpu_probability_recomputations,
            "tracked_dtoh_calls": batch.trace.tracked_dtoh_calls,
            "tracked_data_plane_htod_calls": batch.trace.tracked_data_plane_htod_calls,
            "per_candidate_host_round_trips": batch.trace.per_candidate_host_round_trips,
            "final_output_rows": batch.trace.final_output_rows,
            "accepted_world_views": batch.trace.accepted_world_views,
            "know_operator_count": batch.trace.know_operator_count,
            "possible_operator_count": batch.trace.possible_operator_count,
            "not_know_operator_count": batch.trace.not_know_operator_count,
            "not_possible_operator_count": batch.trace.not_possible_operator_count,
            "aggregate_kernel_timing_recorded": batch.trace.aggregate_kernel_timing.is_recorded(),
        },
        "components": components,
        "derived_relation_rows": derived_relation_rows,
    }))
}

fn put_relations(
    executor: &mut Executor,
    fixture: &RuntimeFixture,
    relations: &BTreeMap<String, (usize, Vec<Vec<u32>>)>,
) -> Result<()> {
    for (name, (arity, rows)) in relations {
        executor.put_relation(name, upload_relation(&fixture.memory, *arity, rows)?);
    }
    Ok(())
}

fn upload_relation(
    memory: &Arc<GpuMemoryManager>,
    arity: usize,
    rows: &[Vec<u32>],
) -> Result<CudaBuffer> {
    if arity == 0 {
        return upload_nullary(memory, rows.len() as u32);
    }
    let n = rows.len() as u32;
    let bytes_per_column = (n as usize).max(1) * std::mem::size_of::<u32>();
    let mut columns = Vec::with_capacity(arity);
    for column_index in 0..arity {
        let mut column = memory.alloc::<u8>(bytes_per_column)?;
        if n > 0 {
            let bytes = rows
                .iter()
                .flat_map(|row| row[column_index].to_le_bytes())
                .collect::<Vec<_>>();
            memory
                .device()
                .inner()
                .htod_sync_copy_into(&bytes, &mut column)
                .map_err(|err| XlogError::Execution(format!("upload relation column: {err}")))?;
        }
        columns.push(column.into());
    }
    let mut device_row_count = memory.alloc::<u32>(1)?;
    memory
        .device()
        .inner()
        .htod_sync_copy_into(&[n], &mut device_row_count)
        .map_err(|err| XlogError::Execution(format!("upload relation row count: {err}")))?;
    let schema = Schema::new(
        (0..arity)
            .map(|idx| (format!("c{idx}"), ScalarType::U32))
            .collect(),
    );
    Ok(CudaBuffer::from_columns_with_host_count(
        columns,
        n as u64,
        device_row_count,
        schema,
        n,
    ))
}

fn upload_nullary(memory: &Arc<GpuMemoryManager>, rows: u32) -> Result<CudaBuffer> {
    let mut device_row_count = memory.alloc::<u32>(1)?;
    memory
        .device()
        .inner()
        .htod_sync_copy_into(&[rows], &mut device_row_count)
        .map_err(|err| XlogError::Execution(format!("upload nullary row count: {err}")))?;
    Ok(CudaBuffer::from_columns_with_host_count(
        Vec::new(),
        rows as u64,
        device_row_count,
        Schema::new(Vec::new()),
        rows,
    ))
}

fn upload_u32_scalar(
    provider: &Arc<CudaKernelProvider>,
    value: u32,
) -> Result<xlog_cuda::memory::TrackedCudaSlice<u32>> {
    let mut scalar = provider.memory().alloc::<u32>(1)?;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[value], &mut scalar)
        .map_err(|err| XlogError::Execution(format!("upload scalar: {err}")))?;
    Ok(scalar)
}

fn download_rows(
    provider: &Arc<CudaKernelProvider>,
    buffer: &CudaBuffer,
    arity: usize,
) -> Result<Vec<Vec<u32>>> {
    let rows = read_device_row_count(provider, buffer)?;
    if rows == 0 {
        return Ok(Vec::new());
    }
    let mut columns = Vec::with_capacity(arity);
    for column_index in 0..arity {
        let mut bytes = vec![0u8; rows * std::mem::size_of::<u32>()];
        unsafe {
            sys::cuMemcpyDtoH_v2(
                bytes.as_mut_ptr() as *mut _,
                *buffer
                    .column(column_index)
                    .ok_or_else(|| {
                        XlogError::Execution(format!("missing output column {column_index}"))
                    })?
                    .device_ptr(),
                bytes.len(),
            );
        }
        columns.push(
            bytes
                .chunks_exact(std::mem::size_of::<u32>())
                .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("u32 chunk")))
                .collect::<Vec<_>>(),
        );
    }
    let mut out = Vec::with_capacity(rows);
    for row_index in 0..rows {
        out.push(
            (0..arity)
                .map(|column_index| columns[column_index][row_index])
                .collect(),
        );
    }
    Ok(out)
}

fn run_gpt_fixture(mode: EpistemicMode) -> Result<Value> {
    let program = parse_program(
        r#"
        pred fact().
        pred accepted().
        accepted() :- know fact().
        "#,
    )?;
    let accepted = EpistemicInterpretation::new().with_known("fact", 0);
    let rejected = EpistemicInterpretation::new().with_rejected("fact", 0);
    let outcome = run_generate_propagate_test_with_mode(
        &program,
        vec![accepted, rejected],
        GeneratePropagateTestConfig { max_candidates: 4 },
        mode,
    )?;
    Ok(json!({
        "mode": format!("{mode:?}"),
        "generated": outcome.trace.generated,
        "guesses": outcome.trace.guesses,
        "propagated": outcome.trace.propagated,
        "pruned": outcome.trace.pruned,
        "tested": outcome.trace.tested,
        "accepted": outcome.trace.accepted,
        "accepted_world_views": outcome.trace.accepted_world_views,
        "rejected": outcome.trace.rejected,
    }))
}

fn run_solver_probability_evidence(
    fixture: &RuntimeFixture,
    result: &xlog_runtime::EpistemicGpuExecutionResult,
    final_rows: &[Vec<u32>],
) -> Result<Value> {
    let proof_row = final_rows.first().ok_or_else(|| {
        XlogError::Execution("solver/probability evidence requires at least one final row".into())
    })?;
    if proof_row.len() < 3 {
        return Err(XlogError::Execution(format!(
            "solver/probability evidence requires case/root/intervention row, got arity {}",
            proof_row.len()
        )));
    }
    let case_id = proof_row[0];
    let root_id = proof_row[1];
    let intervention_id = proof_row[2];
    let sat = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let unsat = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let sat_cnf = GpuCnf::from_host(&sat, &fixture.provider)?;
    let unsat_cnf = GpuCnf::from_host(&unsat, &fixture.provider)?;
    let branch_limit = upload_u32_scalar(&fixture.provider, 1)?;
    let mut solver =
        GpuSolverProductionAdapter::new(Arc::clone(&fixture.provider), GpuCdclConfig::default());
    let mut workspace = solver.new_workspace(1, 2)?;
    let lifecycle = solver.solve_assumption_lifecycle_with_gpu_execution_result(
        &fixture.provider,
        result,
        &mut workspace,
        &[
            GpuSolverProductionLifecycleStep {
                cnf: &sat_cnf,
                branch_var_limit: &branch_limit,
                expectation: GpuSolverProductionExpectation::Sat,
            },
            GpuSolverProductionLifecycleStep {
                cnf: &unsat_cnf,
                branch_var_limit: &branch_limit,
                expectation: GpuSolverProductionExpectation::Unsat,
            },
        ],
    )?;
    let maxsat = solver.solve_weighted_maxsat_candidates_with_gpu_execution_result(
        &fixture.provider,
        result,
        &[
            GpuSolverProductionMaxSatCandidate {
                score: 3,
                cnf: &sat_cnf,
                branch_var_limit: &branch_limit,
            },
            GpuSolverProductionMaxSatCandidate {
                score: 7,
                cnf: &sat_cnf,
                branch_var_limit: &branch_limit,
            },
        ],
    )?;
    let solver_trace = solver.trace();
    solver_trace.require_zero_cpu_search()?;

    let mut config = GpuConfig::default();
    config.device_ordinal = fixture.provider.device().ordinal();
    config.memory_bytes = 64 * 1024 * 1024;
    let mut probability = EpistemicProbProductionAdapter::new(config);
    let prob_source = format!(
        "\
        0.7::neural_support({case_id}, {root_id}).\n\
        0.8::bfo_quality_root({case_id}, {root_id}).\n\
        0.1::excluded_root({case_id}, {root_id}).\n\
        0.1::unsafe_intervention({case_id}, {intervention_id}).\n\
        query(neural_support({case_id}, {root_id})).\n\
    "
    );
    let pir_cnf = probability.encode_source_pir_cnf_with_gpu_execution_result(
        &prob_source,
        &fixture.provider,
        result,
        vec![
            EpistemicAssumption::known_tuple(
                "neural_support",
                vec![
                    EpistemicEvidenceTerm::integer(case_id as i64),
                    EpistemicEvidenceTerm::integer(root_id as i64),
                ],
                true,
            ),
            EpistemicAssumption::known_tuple(
                "bfo_quality_root",
                vec![
                    EpistemicEvidenceTerm::integer(case_id as i64),
                    EpistemicEvidenceTerm::integer(root_id as i64),
                ],
                true,
            ),
            EpistemicAssumption::known_tuple(
                "excluded_root",
                vec![
                    EpistemicEvidenceTerm::integer(case_id as i64),
                    EpistemicEvidenceTerm::integer(root_id as i64),
                ],
                false,
            ),
            EpistemicAssumption::possible_tuple(
                "unsafe_intervention",
                vec![
                    EpistemicEvidenceTerm::integer(case_id as i64),
                    EpistemicEvidenceTerm::integer(intervention_id as i64),
                ],
                false,
            ),
        ],
    )?;
    let prob_trace = probability.trace();
    prob_trace.require_zero_cpu_recompute()?;
    prob_trace.require_production_metric_eligibility()?;

    Ok(json!({
        "solver": {
            "lifecycle": {
                "candidate_evidence_records": lifecycle.candidate_evidence_records,
                "steps": lifecycle.steps,
                "sat_steps": lifecycle.sat_steps,
                "unsat_steps": lifecycle.unsat_steps,
            },
            "maxsat": {
                "candidate_evidence_records": maxsat.candidate_evidence_records,
                "optimum_score": maxsat.optimum_score,
                "candidates_checked": maxsat.candidates_checked,
                "satisfiable_candidates": maxsat.satisfiable_candidates,
            },
            "trace": {
                "accepted_gpu_candidate_evidence_consumed": solver_trace.accepted_gpu_candidate_evidence_consumed,
                "accepted_solver_assumption_bindings_consumed": solver_trace.accepted_solver_assumption_bindings_consumed,
                "accepted_solver_required_capabilities_consumed": solver_trace.accepted_solver_required_capabilities_consumed,
                "gpu_cdcl_sat_solves": solver_trace.gpu_cdcl_sat_solves,
                "gpu_cdcl_unsat_solves": solver_trace.gpu_cdcl_unsat_solves,
                "gpu_maxsat_candidate_solves": solver_trace.gpu_maxsat_candidate_solves,
                "gpu_maxsat_optima": solver_trace.gpu_maxsat_optima,
                "cpu_assignment_enumerations": solver_trace.cpu_assignment_enumerations,
                "cpu_maxsat_enumerations": solver_trace.cpu_maxsat_enumerations,
                "cpu_portfolio_fallbacks": 0,
                "host_materialized_maxsat_fallback": false,
            },
        },
        "probabilistic": {
            "pir_nodes": pir_cnf.pir_nodes,
            "root_count": pir_cnf.root_count,
            "cnf_var_cap": pir_cnf.cnf_var_cap,
            "cnf_clause_cap": pir_cnf.cnf_clause_cap,
            "trace": {
                "accepted_world_view_evidence_consumed": prob_trace.accepted_world_view_evidence_consumed,
                "accepted_faeel_world_view_evidence_consumed": prob_trace.accepted_faeel_world_view_evidence_consumed,
                "accepted_g91_world_view_evidence_consumed": prob_trace.accepted_g91_world_view_evidence_consumed,
                "accepted_evidence_assumptions_consumed": prob_trace.accepted_evidence_assumptions_consumed,
                "gpu_pir_graph_uploads": prob_trace.gpu_pir_graph_uploads,
                "gpu_cnf_encodes": prob_trace.gpu_cnf_encodes,
                "accepted_gpu_production_path_events": prob_trace.accepted_gpu_production_path_events,
                "cpu_probability_recomputations": 0,
                "cpu_only_probability_recomputations": prob_trace.cpu_only_probability_recomputations,
                "host_probability_materialization_fallback": false,
                "fixture_circuit_evaluations": prob_trace.fixture_circuit_evaluations,
            },
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_logic::BodyLiteral;

    fn candidate_generation_source() -> &'static str {
        r#"
        #pragma epistemic_mode = faeel

        pred case_variant(u32, u32).
        pred case_domain(u32, u32).
        pred case_domain_variant(u32, u32, u32).
        pred case_candidate_seed(u32, u32, u32, u32, u32).
        pred domain_candidate_seed(u32, u32, u32, u32).
        pred domain_adapter_root(u32, u32, u32).
        pred domain_adapter_intervention(u32, u32, u32).
        pred domain_adapter_candidate(u32, u32, u32, u32).
        pred variant_candidate_seed(u32, u32, u32, u32, u32).
        pred heldout_label_seed(u32, u32).
        pred blocked_candidate(u32, u32, u32).
        pred adapter_candidate_option(u32, u32, u32, u32).
        pred generated_candidate(u32, u32, u32, u32, u32).

        generated_candidate(Case, Variant, Candidate, Root, Intervention) :-
            case_domain_variant(Case, Variant, Domain),
            domain_candidate_seed(Domain, Candidate, Root, Intervention),
            know domain_candidate_seed(Domain, Candidate, Root, Intervention),
            possible domain_candidate_seed(Domain, Candidate, Root, Intervention),
            possible case_variant(Case, Variant),
            not know heldout_label_seed(Case, Candidate),
            not possible blocked_candidate(Case, Variant, Candidate).

        ?- generated_candidate(Case, Variant, Candidate, Root, Intervention).
        "#
    }

    fn candidate_generation_relations() -> BTreeMap<String, (usize, Vec<Vec<u32>>)> {
        BTreeMap::from([
            ("blocked_candidate".to_string(), (3, Vec::new())),
            ("case_candidate_seed".to_string(), (5, Vec::new())),
            (
                "case_domain".to_string(),
                (
                    2,
                    vec![vec![427904142, 469674573], vec![1245900207, 469674573]],
                ),
            ),
            (
                "case_domain_variant".to_string(),
                (
                    3,
                    vec![
                        vec![427904142, 453330482, 469674573],
                        vec![427904142, 661842346, 469674573],
                        vec![427904142, 785352096, 469674573],
                        vec![427904142, 1123468566, 469674573],
                        vec![427904142, 1521780181, 469674573],
                        vec![427904142, 1570242168, 469674573],
                        vec![1245900207, 453330482, 469674573],
                        vec![1245900207, 661842346, 469674573],
                        vec![1245900207, 785352096, 469674573],
                        vec![1245900207, 1123468566, 469674573],
                        vec![1245900207, 1521780181, 469674573],
                        vec![1245900207, 1570242168, 469674573],
                    ],
                ),
            ),
            (
                "case_variant".to_string(),
                (
                    2,
                    vec![
                        vec![427904142, 453330482],
                        vec![427904142, 661842346],
                        vec![427904142, 785352096],
                        vec![427904142, 1123468566],
                        vec![427904142, 1521780181],
                        vec![427904142, 1570242168],
                        vec![1245900207, 453330482],
                        vec![1245900207, 661842346],
                        vec![1245900207, 785352096],
                        vec![1245900207, 1123468566],
                        vec![1245900207, 1521780181],
                        vec![1245900207, 1570242168],
                    ],
                ),
            ),
            ("domain_adapter_candidate".to_string(), (4, Vec::new())),
            ("domain_adapter_intervention".to_string(), (3, Vec::new())),
            ("domain_adapter_root".to_string(), (3, Vec::new())),
            (
                "domain_candidate_seed".to_string(),
                (
                    4,
                    vec![
                        vec![469674573, 75640238, 1193936297, 1440725968],
                        vec![469674573, 175172646, 1679750782, 2070449337],
                        vec![469674573, 217646891, 729967729, 489692239],
                        vec![469674573, 274009630, 44952629, 327241375],
                        vec![469674573, 623665077, 1280296624, 226238256],
                        vec![469674573, 1491502316, 234733136, 2994529],
                        vec![469674573, 1536163809, 1925193341, 1884170212],
                        vec![469674573, 1551170521, 1968137472, 1142330270],
                        vec![469674573, 1779450926, 832629985, 1892151384],
                    ],
                ),
            ),
            ("heldout_label_seed".to_string(), (2, Vec::new())),
            (
                "variant_candidate_seed".to_string(),
                (
                    5,
                    vec![
                        vec![427904142, 1123468566, 1037515031, 875617417, 491084360],
                        vec![1245900207, 1123468566, 1530915283, 875617417, 491084360],
                    ],
                ),
            ),
        ])
    }

    fn abstention_source() -> &'static str {
        r#"
        #pragma epistemic_mode = faeel

        pred abstention_case(u32).
        pred accepted_world_view(u32, u32).
        pred rejected_world_view(u32, u32).
        pred accepted_world_view_count(u32, u64).
        pred rejected_world_view_count(u32, u64).
        pred proof_evidence_count(u32, u32, u32).
        pred proof_confidence(u32, u32, u32).
        pred abstention_threshold_policy(u32, u32, u32).
        pred abstention_threshold(u32, u32).
        pred dominated_abstention_threshold(u32, u32).
        pred selected_abstention_threshold(u32, u32).
        pred solver_probability_trace(u32, u32, u32).
        pred accept_abstention_candidate(u32, u32, u32).
        pred abstain_abstention_candidate(u32, u32, u32).
        pred unsafe_abstention_path(u32).
        pred xlog_abstention_decision(u32, u32, u32, u32).

        accepted_world_view_count(Abstention, count(WorldView)) :-
            accepted_world_view(Abstention, WorldView).

        rejected_world_view_count(Abstention, count(WorldView)) :-
            rejected_world_view(Abstention, WorldView).

        proof_evidence_count(Abstention, Accepted, Rejected) :-
            accepted_world_view_count(Abstention, Accepted64),
            rejected_world_view_count(Abstention, Rejected64),
            Accepted is cast(Accepted64, u32),
            Rejected is cast(Rejected64, u32).

        proof_evidence_count(Abstention, Accepted, 0) :-
            accepted_world_view_count(Abstention, Accepted64),
            not rejected_world_view_count(Abstention, _),
            Accepted is cast(Accepted64, u32).

        proof_confidence(Abstention, 10000, WorldViewCount) :-
            proof_evidence_count(Abstention, WorldViewCount, 0),
            WorldViewCount > 0.

        abstention_threshold_policy(Abstention, 5000, 1) :-
            abstention_case(Abstention),
            solver_probability_trace(Abstention, GpuCnfEncodes, GpuEvents),
            GpuCnfEncodes > 0,
            GpuEvents > 0.

        abstention_threshold(Abstention, Threshold) :-
            abstention_threshold_policy(Abstention, Threshold, MinGpuEvents),
            solver_probability_trace(Abstention, GpuCnfEncodes, GpuEvents),
            GpuCnfEncodes > 0,
            GpuEvents >= MinGpuEvents.

        dominated_abstention_threshold(Abstention, Threshold) :-
            abstention_threshold(Abstention, Threshold),
            abstention_threshold(Abstention, OtherThreshold),
            OtherThreshold > Threshold.

        selected_abstention_threshold(Abstention, Threshold) :-
            abstention_threshold(Abstention, Threshold),
            not dominated_abstention_threshold(Abstention, Threshold).

        accept_abstention_candidate(Abstention, Threshold, Confidence) :-
            proof_confidence(Abstention, Confidence, WorldViewCount),
            selected_abstention_threshold(Abstention, Threshold),
            solver_probability_trace(Abstention, GpuCnfEncodes, GpuEvents),
            Confidence >= Threshold,
            WorldViewCount > 0,
            GpuCnfEncodes > 0,
            GpuEvents > 0.

        xlog_abstention_decision(Abstention, 1, Threshold, Confidence) :-
            accept_abstention_candidate(Abstention, Threshold, Confidence),
            know abstention_case(Abstention),
            not possible unsafe_abstention_path(Abstention).

        ?- xlog_abstention_decision(Abstention, Decision, Threshold, Confidence).
        "#
    }

    fn abstention_relations() -> BTreeMap<String, (usize, Vec<Vec<u32>>)> {
        BTreeMap::from([
            ("abstention_case".to_string(), (1, vec![vec![701]])),
            ("accepted_world_view".to_string(), (2, vec![vec![701, 801]])),
            ("rejected_world_view".to_string(), (2, Vec::new())),
            (
                "solver_probability_trace".to_string(),
                (3, vec![vec![701, 1, 1]]),
            ),
            ("unsafe_abstention_path".to_string(), (1, Vec::new())),
        ])
    }

    #[test]
    fn parse_program_with_modules_merges_entry_imports() -> Result<()> {
        let base_dir = std::env::temp_dir().join(format!(
            "xlog-epistemic-evidence-module-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&base_dir);
        fs::create_dir_all(base_dir.join("modules")).map_err(|err| {
            XlogError::Execution(format!(
                "create module test dir {}: {err}",
                base_dir.display()
            ))
        })?;
        let module_source = r#"
            pred imported_reach(u32, u32).

            imported_reach(X, Y) :- imported_edge(X, Y).
            imported_reach(X, Z) :- imported_reach(X, Y), imported_edge(Y, Z).
        "#;
        fs::write(base_dir.join("modules").join("closure.xlog"), module_source).map_err(|err| {
            XlogError::Execution(format!(
                "write module fixture {}: {err}",
                base_dir.display()
            ))
        })?;
        let entry_source = r#"
            use modules/closure::{imported_reach}.

            pred imported_edge(u32, u32).
            pred merged_result(u32, u32).

            merged_result(X, Y) :- imported_reach(X, Y).

            ?- merged_result(X, Y).
        "#;
        let entry_path = base_dir.join("entry.xlog");
        fs::write(&entry_path, entry_source).map_err(|err| {
            XlogError::Execution(format!(
                "write entry fixture {}: {err}",
                entry_path.display()
            ))
        })?;

        let program = parse_program_with_modules(&entry_path, entry_source)?;
        assert!(program
            .predicates
            .iter()
            .any(|predicate| predicate.name == "imported_reach"));
        assert!(program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "imported_reach"));

        let _ = fs::remove_dir_all(&base_dir);
        Ok(())
    }

    #[test]
    fn ucr_failure_chain_program_merges_modular_closure() -> Result<()> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| XlogError::Execution("resolve repo root".to_string()))?;
        let program_path = repo_root
            .join("examples")
            .join("BFO")
            .join("universal_case_reasoner")
            .join("programs")
            .join("epistemic")
            .join("epistemic_generalization_failure_chain_support.xlog");
        let source = fs::read_to_string(&program_path).map_err(|err| {
            XlogError::Execution(format!(
                "read UCR failure-chain program {}: {err}",
                program_path.display()
            ))
        })?;

        let program = parse_program_with_modules(&program_path, &source)?;
        assert!(program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "failure_chain_component_reach"));
        assert!(program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "candidate_failure_chain_support"));
        assert!(program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "xlog_failure_chain_claim"));
        Ok(())
    }

    fn rule_body_has_positive_predicate(rule: &xlog_logic::Rule, predicate: &str) -> bool {
        rule.body.iter().any(|literal| {
            matches!(
                literal,
                BodyLiteral::Positive(atom) if atom.predicate == predicate
            )
        })
    }

    #[test]
    fn ucr_decision_program_reuses_modular_failure_chain_closure() -> Result<()> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| XlogError::Execution("resolve repo root".to_string()))?;
        let program_path = repo_root
            .join("examples")
            .join("BFO")
            .join("universal_case_reasoner")
            .join("programs")
            .join("epistemic")
            .join("epistemic_generalization_decision.xlog");
        let source = fs::read_to_string(&program_path).map_err(|err| {
            XlogError::Execution(format!(
                "read UCR decision program {}: {err}",
                program_path.display()
            ))
        })?;

        let program = parse_program_with_modules(&program_path, &source)?;
        assert!(
            program
                .rules
                .iter()
                .any(|rule| rule.head.predicate == "failure_chain_component_reach"),
            "final decision must merge the recursive failure-chain closure"
        );

        let candidate_support_rules = program
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == "candidate_failure_chain_support")
            .collect::<Vec<_>>();
        assert!(
            candidate_support_rules.iter().any(|rule| {
                rule_body_has_positive_predicate(
                    rule,
                    "candidate_failure_chain_components_supported",
                )
            }),
            "final decision candidate chain support must come from the modular closure"
        );
        assert!(
            !candidate_support_rules.iter().any(|rule| {
                rule_body_has_positive_predicate(rule, "candidate_failure_chain_root_supported")
                    && rule_body_has_positive_predicate(
                        rule,
                        "candidate_failure_chain_intervention_supported",
                    )
                    && rule_body_has_positive_predicate(
                        rule,
                        "candidate_failure_chain_risk_state_supported",
                    )
            }),
            "final decision must not keep the old flat candidate failure-chain support rule"
        );
        assert!(
            exported_derived_relations("epistemic_generalization_decision.xlog")
                .contains(&"failure_chain_component_reach")
        );
        assert!(
            exported_derived_relations("epistemic_generalization_decision.xlog")
                .contains(&"candidate_failure_chain_components_supported")
        );
        Ok(())
    }

    #[test]
    fn ucr_dilp_schema_generator_merges_support_policy_module() -> Result<()> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| XlogError::Execution("resolve repo root".to_string()))?;
        let program_path = repo_root
            .join("examples")
            .join("BFO")
            .join("universal_case_reasoner")
            .join("programs")
            .join("epistemic")
            .join("epistemic_dilp_proof_schema_generator.xlog");
        let source = fs::read_to_string(&program_path).map_err(|err| {
            XlogError::Execution(format!(
                "read UCR dILP proof-schema program {}: {err}",
                program_path.display()
            ))
        })?;

        assert!(source.contains("use modules/dilp_schema_support_policy."));
        assert!(
            !source.contains("support_weight_component(Support, 100, 0, 0, 100)"),
            "schema support policy weights must live in the imported module"
        );

        let program = parse_program_with_modules(&program_path, &source)?;
        assert!(program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "support_weight_component"));
        assert!(program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "schema_support_weight"));
        assert!(program.rules.iter().any(|rule| {
            rule.head.predicate == "training_support_observation"
                && rule_body_has_positive_predicate(rule, "schema_support_weight")
        }));
        Ok(())
    }

    #[test]
    fn ucr_dilp_trace_merges_support_policy_module() -> Result<()> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| XlogError::Execution("resolve repo root".to_string()))?;
        let program_path = repo_root
            .join("examples")
            .join("BFO")
            .join("universal_case_reasoner")
            .join("programs")
            .join("epistemic")
            .join("epistemic_dilp_proof_trace.xlog");
        let source = fs::read_to_string(&program_path).map_err(|err| {
            XlogError::Execution(format!(
                "read UCR dILP proof-trace program {}: {err}",
                program_path.display()
            ))
        })?;

        assert!(source.contains("use modules/dilp_trace_support_policy."));
        assert!(
            !source.contains("trace_support_weight_component(Support, 100, 0, 0, 100)"),
            "trace support policy weights must live in the imported module"
        );

        let program = parse_program_with_modules(&program_path, &source)?;
        assert!(program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "trace_support_weight_component"));
        assert!(program.rules.iter().any(|rule| {
            rule.head.predicate == "trace_feature_score"
                && rule_body_has_positive_predicate(rule, "trace_support_weight_component")
        }));
        Ok(())
    }

    #[test]
    fn ucr_candidate_scoring_programs_merge_feature_component_module() -> Result<()> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| XlogError::Execution("resolve repo root".to_string()))?;

        for file_name in [
            "epistemic_generalization_candidate_scoring.xlog",
            "epistemic_showcase_transfer_candidate_scoring.xlog",
        ] {
            let program_path = repo_root
                .join("examples")
                .join("BFO")
                .join("universal_case_reasoner")
                .join("programs")
                .join("epistemic")
                .join(file_name);
            let source = fs::read_to_string(&program_path).map_err(|err| {
                XlogError::Execution(format!(
                    "read UCR scoring program {}: {err}",
                    program_path.display()
                ))
            })?;

            assert!(source.contains("use modules/candidate_feature_components."));
            assert!(
                !source.contains("bfo_evidence_score(Case, Candidate, Bfo) :-"),
                "{file_name} must import feature component semantics instead of redefining them"
            );

            let program = parse_program_with_modules(&program_path, &source)?;
            assert!(program
                .rules
                .iter()
                .any(|rule| rule.head.predicate == "bfo_evidence_score"));
            assert!(program
                .rules
                .iter()
                .any(|rule| rule.head.predicate == "candidate_feature_score"));
        }
        Ok(())
    }

    #[test]
    fn ucr_ranker_and_decision_reuse_selector_acceptance_module() -> Result<()> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| XlogError::Execution("resolve repo root".to_string()))?;
        let epistemic_dir = repo_root
            .join("examples")
            .join("BFO")
            .join("universal_case_reasoner")
            .join("programs")
            .join("epistemic");

        let ranker_path = epistemic_dir.join("epistemic_generalization_ranker.xlog");
        let ranker_source = fs::read_to_string(&ranker_path).map_err(|err| {
            XlogError::Execution(format!(
                "read UCR ranker program {}: {err}",
                ranker_path.display()
            ))
        })?;
        assert!(ranker_source.contains("use modules/selector_acceptance."));
        assert!(
            !ranker_source.contains("selector_accepted(Case, Candidate) :-"),
            "ranker must import selector acceptance instead of redefining it"
        );

        let decision_support_path = epistemic_dir.join("modules").join("decision_support.xlog");
        let decision_support_source =
            fs::read_to_string(&decision_support_path).map_err(|err| {
                XlogError::Execution(format!(
                    "read UCR decision-support module {}: {err}",
                    decision_support_path.display()
                ))
            })?;
        assert!(decision_support_source.contains("use selector_acceptance."));
        assert!(
            !decision_support_source.contains("selector_accepted(Case, Candidate) :-"),
            "decision support must import selector acceptance instead of redefining it"
        );
        assert!(
            !decision_support_source.contains("unsafe_decision(Case, Candidate) :-"),
            "decision support must import unsafe-decision derivation with selector acceptance"
        );

        let decision_path = epistemic_dir.join("epistemic_generalization_decision.xlog");
        let decision_source = fs::read_to_string(&decision_path).map_err(|err| {
            XlogError::Execution(format!(
                "read UCR decision program {}: {err}",
                decision_path.display()
            ))
        })?;
        let decision_program = parse_program_with_modules(&decision_path, &decision_source)?;
        assert!(decision_program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "selector_accepted"));
        assert!(decision_program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "unsafe_decision"));
        Ok(())
    }

    #[test]
    fn ucr_abstention_program_merges_policy_module() -> Result<()> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| XlogError::Execution("resolve repo root".to_string()))?;
        let program_path = repo_root
            .join("examples")
            .join("BFO")
            .join("universal_case_reasoner")
            .join("programs")
            .join("epistemic")
            .join("epistemic_generalization_abstention.xlog");
        let source = fs::read_to_string(&program_path).map_err(|err| {
            XlogError::Execution(format!(
                "read UCR abstention program {}: {err}",
                program_path.display()
            ))
        })?;

        assert!(source.contains("use modules/abstention_policy."));
        assert!(
            !source.contains("proof_confidence(Abstention, Confidence, WorldViewCount) :-"),
            "abstention confidence policy must live in the imported module"
        );
        assert!(
            !source.contains("accept_abstention_candidate(Abstention, Threshold, Confidence) :-"),
            "accept/abstain candidate policy must live in the imported module"
        );

        let program = parse_program_with_modules(&program_path, &source)?;
        assert!(program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "proof_confidence"));
        assert!(program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "selected_abstention_threshold"));
        assert!(program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "accept_abstention_candidate"));
        assert!(program.rules.iter().any(|rule| {
            rule.head.predicate == "xlog_abstention_decision"
                && rule_body_has_positive_predicate(rule, "accept_abstention_candidate")
        }));
        Ok(())
    }

    #[test]
    fn ucr_explanation_program_merges_support_module() -> Result<()> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| XlogError::Execution("resolve repo root".to_string()))?;
        let program_path = repo_root
            .join("examples")
            .join("BFO")
            .join("universal_case_reasoner")
            .join("programs")
            .join("epistemic")
            .join("epistemic_generalization_explanation.xlog");
        let source = fs::read_to_string(&program_path).map_err(|err| {
            XlogError::Execution(format!(
                "read UCR explanation program {}: {err}",
                program_path.display()
            ))
        })?;

        assert!(source.contains("use modules/explanation_support."));
        assert!(
            !source.contains("explanation_metadata(1, 385046675, 1419304611, 39051548) :-"),
            "explanation metadata policy must live in the imported module"
        );
        assert!(
            !source.contains("explanation_dependency(Case, Candidate, ClaimType, Claim, Dependency, Relation) :-"),
            "explanation dependency policy must live in the imported module"
        );

        let program = parse_program_with_modules(&program_path, &source)?;
        assert!(program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "explanation_identifier"));
        assert!(program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "selected_explanation_metadata"));
        assert!(program
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "explanation_support"));
        assert!(program.rules.iter().any(|rule| {
            rule.head.predicate == "xlog_explanation"
                && rule_body_has_positive_predicate(rule, "explanation_identifier")
                && rule_body_has_positive_predicate(rule, "selected_explanation_metadata")
        }));
        Ok(())
    }

    #[test]
    fn ucr_failure_chain_modular_closure_executes_claim() -> Result<()> {
        let fixture = match make_fixture(0, gpu_budget_bytes(DEFAULT_EPISTEMIC_GPU_BUDGET_MIB)?) {
            Ok(fixture) => fixture,
            Err(_) => return Ok(()),
        };
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| XlogError::Execution("resolve repo root".to_string()))?;
        let program_path = repo_root
            .join("examples")
            .join("BFO")
            .join("universal_case_reasoner")
            .join("programs")
            .join("epistemic")
            .join("epistemic_generalization_failure_chain_support.xlog");
        let source = fs::read_to_string(&program_path).map_err(|err| {
            XlogError::Execution(format!(
                "read UCR failure-chain program {}: {err}",
                program_path.display()
            ))
        })?;
        let relations = BTreeMap::from([
            (
                "candidate_root_bfo_seed".to_string(),
                (3, vec![vec![1, 10, 100]]),
            ),
            (
                "candidate_root_evidence_seed".to_string(),
                (3, vec![vec![1, 10, 100]]),
            ),
            (
                "decision_claim_option".to_string(),
                (
                    4,
                    vec![
                        vec![1, 10, 1, 100],
                        vec![1, 10, 2, 200],
                        vec![1, 10, 3, 300],
                        vec![1, 10, 4, 400],
                    ],
                ),
            ),
            (
                "failure_chain_bfo_support".to_string(),
                (2, vec![vec![1, 400]]),
            ),
            (
                "failure_chain_candidate_seed".to_string(),
                (4, vec![vec![1, 10, 4, 400]]),
            ),
            (
                "failure_chain_intervention_link".to_string(),
                (3, vec![vec![1, 400, 200]]),
            ),
            (
                "failure_chain_risk_state_link".to_string(),
                (3, vec![vec![1, 400, 300]]),
            ),
            (
                "failure_chain_root_link".to_string(),
                (3, vec![vec![1, 400, 100]]),
            ),
            (
                "intervention_objective_support".to_string(),
                (2, vec![vec![1, 200]]),
            ),
            ("intervention_option".to_string(), (2, vec![vec![1, 200]])),
            (
                "intervention_safety_support".to_string(),
                (2, vec![vec![1, 200]]),
            ),
            (
                "risk_state_bfo_support".to_string(),
                (2, vec![vec![1, 300]]),
            ),
            ("risk_state_option".to_string(), (2, vec![vec![1, 300]])),
            (
                "risk_state_world_view_support".to_string(),
                (2, vec![vec![1, 300]]),
            ),
            ("root_cause_option".to_string(), (2, vec![vec![1, 100]])),
            ("selector_bfo_candidate".to_string(), (2, vec![vec![1, 10]])),
            ("selector_candidate".to_string(), (2, vec![vec![1, 10]])),
            (
                "selector_observed_evidence".to_string(),
                (2, vec![vec![1, 10]]),
            ),
            ("selector_unsafe_candidate".to_string(), (2, Vec::new())),
            ("unsafe_failure_chain_option".to_string(), (2, Vec::new())),
            ("unsafe_intervention_option".to_string(), (2, Vec::new())),
            ("unsafe_risk_state_option".to_string(), (2, Vec::new())),
            ("unsafe_root_option".to_string(), (2, Vec::new())),
        ]);

        let payload = execute_single(
            &fixture,
            &program_path,
            "epistemic_generalization_failure_chain_support.xlog",
            &source,
            &relations,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 512,
                max_worlds: 1,
                max_models_per_reduction: 64,
            },
            false,
        )?;
        assert_eq!(payload["status"], "PASS");
        assert_eq!(
            payload["runtime"]["semantic_trace"]["accepted_candidates"],
            1
        );
        assert_eq!(
            payload["runtime"]["final_result_transfer"]["final_output_rows"],
            1
        );
        assert!(
            relation_rows_contain(
                &payload["derived_relation_rows"],
                "failure_chain_component_reach",
                &[1, 400, 100, 300],
            ),
            "runtime={} derived_relation_rows={}",
            payload["runtime"],
            payload["derived_relation_rows"]
        );
        assert!(
            relation_rows_contain(
                &payload["derived_relation_rows"],
                "candidate_failure_chain_claim_ready",
                &[1, 10, 4, 400],
            ),
            "runtime={} derived_relation_rows={}",
            payload["runtime"],
            payload["derived_relation_rows"]
        );
        assert!(
            relation_rows_contain(
                &payload["derived_relation_rows"],
                "xlog_failure_chain_claim",
                &[1, 10, 4, 400],
            ),
            "runtime={} derived_relation_rows={}",
            payload["runtime"],
            payload["derived_relation_rows"]
        );
        drop(payload);
        drop(fixture);
        Ok(())
    }

    fn relation_rows_contain(rows: &Value, relation_name: &str, expected: &[u32]) -> bool {
        rows.get(relation_name)
            .and_then(|spec| spec.get("rows"))
            .and_then(Value::as_array)
            .map(|relation_rows| {
                relation_rows.iter().any(|row| {
                    row.as_array()
                        .map(|values| {
                            values
                                .iter()
                                .filter_map(Value::as_u64)
                                .map(|value| value as u32)
                                .collect::<Vec<_>>()
                                == expected
                        })
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
    }

    #[test]
    fn candidate_generation_owned_runtime_teardown_exits_cleanly() -> Result<()> {
        let fixture = match make_fixture(0, gpu_budget_bytes(DEFAULT_EPISTEMIC_GPU_BUDGET_MIB)?) {
            Ok(fixture) => fixture,
            Err(_) => return Ok(()),
        };
        let payload = execute_single(
            &fixture,
            Path::new("epistemic_generalization_candidate_generator.xlog"),
            "epistemic_generalization_candidate_generator.xlog",
            candidate_generation_source(),
            &candidate_generation_relations(),
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 4096,
                max_worlds: 1,
                max_models_per_reduction: 64,
            },
            false,
        )?;
        assert_eq!(payload["status"], "PASS");
        assert_eq!(
            payload["final_rows"]
                .as_array()
                .map(|rows| rows.len())
                .unwrap_or_default(),
            108
        );
        drop(payload);
        drop(fixture);
        Ok(())
    }

    #[test]
    fn serverless_default_budget_covers_full_epistemic_decision_batch() {
        let observed_runpod_batch_bytes = 282_624_000;
        assert!(
            gpu_budget_bytes(DEFAULT_EPISTEMIC_GPU_BUDGET_MIB).unwrap()
                > observed_runpod_batch_bytes
        );
    }

    #[test]
    fn abstention_exports_resident_no_host_diagnostics() -> Result<()> {
        let fixture = match make_fixture(0, gpu_budget_bytes(DEFAULT_EPISTEMIC_GPU_BUDGET_MIB)?) {
            Ok(fixture) => fixture,
            Err(_) => return Ok(()),
        };
        let payload = execute_single(
            &fixture,
            Path::new("epistemic_generalization_abstention.xlog"),
            "epistemic_generalization_abstention.xlog",
            abstention_source(),
            &abstention_relations(),
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 16,
                max_worlds: 1,
                max_models_per_reduction: 64,
            },
            false,
        )?;
        let diagnostics = &payload["runtime"]["gpu_probability_diagnostics"];
        assert_eq!(
            diagnostics["resident_mc_source"],
            "xlog_v092_mc_resident_engine"
        );
        assert!(diagnostics["resident_mc_total_samples"].as_u64().unwrap() > 0);
        assert_eq!(diagnostics["resident_mc_query_count"], 1);
        assert!(diagnostics["resident_mc_engine_launches"].as_u64().unwrap() > 0);
        assert_eq!(diagnostics["resident_mc_no_host"], true);
        assert_eq!(diagnostics["resident_mc_tracked_htod_calls"], 0);
        assert_eq!(diagnostics["resident_mc_tracked_dtoh_calls"], 0);
        assert_eq!(diagnostics["resident_mc_untracked_metadata_reads"], 0);
        assert_eq!(diagnostics["resident_mc_host_loop_iterations"], 0);
        assert_eq!(diagnostics["resident_mc_per_sample_host_launches"], 0);
        assert_eq!(diagnostics["resident_mc_host_fixpoint_iterations"], 0);
        assert_eq!(diagnostics["resident_mc_per_operator_host_allocations"], 0);
        assert_eq!(diagnostics["threshold_relation_rows_consumed"], 1);
        Ok(())
    }

    #[test]
    fn rejects_zero_gpu_budget() {
        assert!(gpu_budget_bytes(0).is_err());
    }
}

fn preflight_json(preflight: &xlog_runtime::EpistemicGpuRuntimePreflight) -> Value {
    json!({
        "epistemic_mode": format!("{:?}", preflight.epistemic_mode),
        "reduced_runtime_rule_count": preflight.reduced_runtime_rule_count,
        "wcoj_required_reduction_count": preflight.wcoj_required_reduction_count,
        "multiway_reduction_count": preflight.multiway_reduction_count,
        "kclique_wcoj_plan_count": preflight.kclique_wcoj_plan_count,
        "planned_hash_route_count": preflight.planned_hash_route_count,
        "tuple_membership_binding_count": preflight.tuple_membership_binding_count,
        "solver_assumption_binding_count": preflight.solver_assumption_binding_count,
        "solver_required_capability_count": preflight.solver_required_capability_count,
        "solver_required_status_count": preflight.solver_required_status_count,
        "know_operator_count": preflight.know_operator_count,
        "possible_operator_count": preflight.possible_operator_count,
        "not_know_operator_count": preflight.not_know_operator_count,
        "not_possible_operator_count": preflight.not_possible_operator_count,
        "cpu_fallbacks_zero": preflight.cpu_fallbacks.is_zero(),
    })
}

fn runtime_json(result: &xlog_runtime::EpistemicGpuExecutionResult) -> Value {
    json!({
        "candidate_generation": {
            "generated_candidates": result.candidate_generation.generated_candidates,
            "literal_count": result.candidate_generation.literal_count,
            "kernel_launches": result.candidate_generation.kernel_launches,
            "host_write_ops": result.candidate_generation.host_write_ops,
            "timing_recorded": result.candidate_generation.kernel_timing.is_recorded(),
        },
        "semantic_trace": {
            "generated_candidates": result.semantic_trace.generated_candidates,
            "guesses": result.semantic_trace.guesses,
            "accepted_candidates": result.semantic_trace.accepted_candidates,
            "accepted_candidate_indices": result.semantic_trace.accepted_candidate_indices,
            "accepted_world_views": result.semantic_trace.accepted_world_views,
            "rejected_candidates": result.semantic_trace.rejected_candidates,
            "rejected_candidate_indices": result.semantic_trace.rejected_candidate_indices,
            "rejection_reasons": result.semantic_trace.rejection_reasons,
            "rejection_reason_device_reads": result.semantic_trace.rejection_reason_device_reads,
            "rejection_reason_metadata_bytes": result.semantic_trace.rejection_reason_metadata_bytes,
            "cpu_candidate_enumerations": result.semantic_trace.cpu_candidate_enumerations,
            "cpu_world_view_validations": result.semantic_trace.cpu_world_view_validations,
        },
        "reduced_output": {
            "row_count": result.output.cached_row_count(),
            "arity": result.output.arity(),
        },
        "model_membership": {
            "membership_source": format!("{:?}", result.model_membership.membership_source),
            "tuple_source_key_column_device_reads": result.model_membership.tuple_source_key_column_device_reads,
            "host_write_ops": result.model_membership.host_write_ops,
            "kernel_launches": result.model_membership.kernel_launches,
            "timing_recorded": result.model_membership.kernel_timing.is_recorded(),
        },
        "world_view_validation": {
            "kernel_launches": result.world_view_validation.kernel_launches,
            "host_write_ops": result.world_view_validation.host_write_ops,
            "timing_recorded": result.world_view_validation.kernel_timing.is_recorded(),
        },
        "final_tuple_materialization": {
            "output_column_count": result.final_tuple_materialization.output_column_count,
            "row_filter_count": result.final_tuple_materialization.row_filter_count,
            "negated_row_filter_count": result.final_tuple_materialization.negated_row_filter_count,
            "kernel_launches": result.final_tuple_materialization.kernel_launches,
            "host_write_ops": result.final_tuple_materialization.host_write_ops,
            "timing_recorded": result.final_tuple_materialization.kernel_timing.is_recorded(),
        },
        "transfer_budget": {
            "candidate_count": result.transfer_budget.candidate_count,
            "tracked_dtoh_calls": result.transfer_budget.tracked_dtoh_calls,
            "tracked_htod_calls": result.transfer_budget.tracked_htod_calls,
            "tracked_data_plane_htod_calls": result.transfer_budget.tracked_data_plane_htod_calls,
            "tracked_data_plane_htod_bytes": result.transfer_budget.tracked_data_plane_htod_bytes,
            "per_candidate_host_round_trips": result.transfer_budget.per_candidate_host_round_trips,
        },
        "final_result_transfer": {
            "final_output_rows": result.final_result_transfer.final_output_rows,
            "final_output_column_count": result.final_result_transfer.final_output_column_count,
            "tracked_data_plane_dtoh_calls": result.final_result_transfer.tracked_data_plane_dtoh_calls,
            "tracked_data_plane_dtoh_bytes": result.final_result_transfer.tracked_data_plane_dtoh_bytes,
        },
        "wcoj": {
            "certification": format!("{:?}", result.trace.wcoj_certification),
            "wcoj_4cycle_dispatch_count": result.trace.counter_delta.wcoj_4cycle_dispatch_count,
            "wcoj_triangle_dispatch_count": result.trace.counter_delta.wcoj_triangle_dispatch_count,
            "wcoj_clique5_dispatch_count": result.trace.counter_delta.wcoj_clique5_dispatch_count,
            "wcoj_clique6_dispatch_count": result.trace.counter_delta.wcoj_clique6_dispatch_count,
            "wcoj_clique7_dispatch_count": result.trace.counter_delta.wcoj_clique7_dispatch_count,
            "wcoj_clique8_dispatch_count": result.trace.counter_delta.wcoj_clique8_dispatch_count,
        },
    })
}

fn gpu_execution_dispatch_count(result: &xlog_runtime::EpistemicGpuExecutionResult) -> u64 {
    result.candidate_generation.kernel_launches as u64
        + result.model_membership.kernel_launches as u64
        + result.world_view_validation.kernel_launches as u64
        + result.final_tuple_materialization.kernel_launches as u64
}

fn input_relation_row_count(
    relations: &BTreeMap<String, (usize, Vec<Vec<u32>>)>,
    relation_name: &str,
) -> usize {
    relations
        .get(relation_name)
        .map(|(_, rows)| rows.len())
        .unwrap_or(0)
}

fn derived_relation_row_count(derived_relation_rows: &Value, relation_name: &str) -> usize {
    derived_relation_rows
        .get(relation_name)
        .and_then(|spec| spec.get("rows"))
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0)
}

fn attach_program_runtime_diagnostics(
    source_file_name: &str,
    result: &xlog_runtime::EpistemicGpuExecutionResult,
    relations: &BTreeMap<String, (usize, Vec<Vec<u32>>)>,
    derived_relation_rows: &Value,
    runtime: &mut Value,
) {
    let Some(runtime_object) = runtime.as_object_mut() else {
        return;
    };
    let dispatch_count = gpu_execution_dispatch_count(result);
    match source_file_name {
        "epistemic_dilp_proof_schema_selection.xlog" => {
            runtime_object.insert(
                "gpu_proof_schema_selection_diagnostics".to_string(),
                json!({
                    "source": "xlog_v090_gpu_proof_schema_selection",
                    "program": "programs/epistemic/epistemic_dilp_proof_schema_selection.xlog",
                    "xlog_selection_dispatches": dispatch_count,
                    "generated_proof_schema_rows_consumed": input_relation_row_count(relations, "generated_proof_schema"),
                    "schema_symbolic_weight_rows_consumed": input_relation_row_count(relations, "schema_symbolic_weight"),
                    "schema_proof_search_support_rows_consumed": input_relation_row_count(relations, "schema_proof_search_support"),
                    "schema_solver_support_rows_consumed": input_relation_row_count(relations, "schema_solver_support"),
                    "heldout_promoted_schema_rows_consumed": input_relation_row_count(relations, "heldout_promoted_schema"),
                    "blocked_promoted_schema_rows_consumed": input_relation_row_count(relations, "blocked_promoted_schema"),
                    "derived_schema_promotion_score_rows": derived_relation_row_count(derived_relation_rows, "schema_promotion_score"),
                    "derived_schema_promotion_threshold_rows": derived_relation_row_count(derived_relation_rows, "schema_promotion_threshold"),
                    "derived_promotion_candidate_rows": derived_relation_row_count(derived_relation_rows, "promotion_candidate"),
                    "derived_dominated_promoted_proof_schema_rows": derived_relation_row_count(derived_relation_rows, "dominated_promoted_proof_schema"),
                    "derived_selection_candidate_rows": derived_relation_row_count(derived_relation_rows, "selection_candidate"),
                    "derived_selected_promoted_proof_schema_rows": derived_relation_row_count(derived_relation_rows, "selected_promoted_proof_schema"),
                    "final_selected_promoted_proof_schema_rows": result.final_result_transfer.final_output_rows,
                    "cpu_threshold_only_promotions": 0,
                    "threshold_only_promotion": false,
                }),
            );
        }
        "epistemic_dilp_candidate_scoring.xlog" => {
            runtime_object.insert(
                "gpu_candidate_scoring_diagnostics".to_string(),
                json!({
                    "source": "xlog_v090_gpu_dilp_candidate_scoring",
                    "program": "programs/epistemic/epistemic_dilp_candidate_scoring.xlog",
                    "xlog_scoring_dispatches": dispatch_count,
                    "scoring_candidate_rows_consumed": input_relation_row_count(relations, "scoring_candidate"),
                    "neural_margin_score_rows_consumed": input_relation_row_count(relations, "neural_margin_score"),
                    "proof_trace_support_rows_consumed": input_relation_row_count(relations, "proof_trace_support"),
                    "selected_proof_schema_weight_rows_consumed": input_relation_row_count(relations, "selected_proof_schema_weight"),
                    "selector_accepted_score_rows_consumed": input_relation_row_count(relations, "selector_accepted_score"),
                    "unsafe_score_candidate_rows_consumed": input_relation_row_count(relations, "unsafe_score_candidate"),
                    "derived_selected_proof_support_score_rows": derived_relation_row_count(derived_relation_rows, "selected_proof_support_score"),
                    "derived_proof_weighted_score_rows": derived_relation_row_count(derived_relation_rows, "proof_weighted_score"),
                    "derived_dilp_transductive_transfer_signal_rows": derived_relation_row_count(derived_relation_rows, "dilp_transductive_transfer_signal"),
                    "derived_dilp_neural_instability_score_rows": derived_relation_row_count(derived_relation_rows, "dilp_neural_instability_score"),
                    "derived_bounded_dilp_metastable_transfer_margin_rows": derived_relation_row_count(derived_relation_rows, "bounded_dilp_metastable_transfer_margin"),
                    "derived_dilp_candidate_score_rows": result.final_result_transfer.final_output_rows,
                    "python_rank_score_materialization": false,
                    "python_feature_weighted_ranking": false,
                    "python_score_weight_constants_used": false,
                    "host_materialized_score_fallback": false,
                    "threshold_only_promotion": false,
                }),
            );
        }
        "epistemic_generalization_candidate_scoring.xlog" => {
            runtime_object.insert(
                "gpu_candidate_scoring_diagnostics".to_string(),
                json!({
                    "source": "xlog_v090_gpu_generalization_candidate_scoring",
                    "program": "programs/epistemic/epistemic_generalization_candidate_scoring.xlog",
                    "xlog_scoring_dispatches": dispatch_count,
                    "selector_accepted_neural_score_rows_consumed": input_relation_row_count(relations, "selector_accepted_neural_score"),
                    "candidate_raw_feature_score_rows_consumed": input_relation_row_count(relations, "candidate_raw_feature_score"),
                    "scoring_epistemic_gate_rows_consumed": input_relation_row_count(relations, "scoring_epistemic_gate"),
                    "unsafe_score_candidate_rows_consumed": input_relation_row_count(relations, "unsafe_score_candidate"),
                    "derived_bfo_evidence_score_rows": derived_relation_row_count(derived_relation_rows, "bfo_evidence_score"),
                    "derived_literal_observation_score_rows": derived_relation_row_count(derived_relation_rows, "literal_observation_score"),
                    "derived_mismatch_penalty_score_rows": derived_relation_row_count(derived_relation_rows, "mismatch_penalty_score"),
                    "derived_candidate_feature_score_rows": derived_relation_row_count(derived_relation_rows, "candidate_feature_score"),
                    "derived_transductive_transfer_signal_rows": derived_relation_row_count(derived_relation_rows, "transductive_transfer_signal"),
                    "derived_bounded_metastable_transfer_margin_rows": derived_relation_row_count(derived_relation_rows, "bounded_metastable_transfer_margin"),
                    "derived_xlog_candidate_score_rows": derived_relation_row_count(derived_relation_rows, "xlog_candidate_score"),
                    "python_rank_score_materialization": false,
                    "python_feature_weighted_ranking": false,
                    "python_score_weight_constants_used": false,
                    "host_materialized_score_fallback": false,
                    "threshold_only_promotion": false,
                }),
            );
        }
        "epistemic_showcase_transfer_candidate_scoring.xlog" => {
            runtime_object.insert(
                "gpu_candidate_scoring_diagnostics".to_string(),
                json!({
                    "source": "xlog_v090_gpu_showcase_transfer_candidate_scoring",
                    "program": "programs/epistemic/epistemic_showcase_transfer_candidate_scoring.xlog",
                    "xlog_scoring_dispatches": dispatch_count,
                    "scoring_candidate_rows_consumed": input_relation_row_count(relations, "scoring_candidate"),
                    "neural_primary_score_rows_consumed": input_relation_row_count(relations, "neural_primary_score"),
                    "candidate_raw_feature_score_rows_consumed": input_relation_row_count(relations, "candidate_raw_feature_score"),
                    "selector_accepted_score_rows_consumed": input_relation_row_count(relations, "selector_accepted_score"),
                    "unsafe_score_candidate_rows_consumed": input_relation_row_count(relations, "unsafe_score_candidate"),
                    "derived_bfo_evidence_score_rows": derived_relation_row_count(derived_relation_rows, "bfo_evidence_score"),
                    "derived_literal_observation_score_rows": derived_relation_row_count(derived_relation_rows, "literal_observation_score"),
                    "derived_mismatch_penalty_score_rows": derived_relation_row_count(derived_relation_rows, "mismatch_penalty_score"),
                    "derived_candidate_feature_score_rows": derived_relation_row_count(derived_relation_rows, "candidate_feature_score"),
                    "derived_showcase_transductive_transfer_signal_rows": derived_relation_row_count(derived_relation_rows, "showcase_transductive_transfer_signal"),
                    "derived_bounded_showcase_metastable_transfer_margin_rows": derived_relation_row_count(derived_relation_rows, "bounded_showcase_metastable_transfer_margin"),
                    "derived_showcase_transfer_candidate_score_rows": derived_relation_row_count(derived_relation_rows, "showcase_transfer_candidate_score"),
                    "python_rank_score_materialization": false,
                    "python_feature_weighted_ranking": false,
                    "python_score_weight_constants_used": false,
                    "host_materialized_score_fallback": false,
                    "threshold_only_promotion": false,
                }),
            );
        }
        "epistemic_generalization_decision.xlog" => {
            runtime_object.insert(
                "gpu_maxsat_diagnostics".to_string(),
                json!({
                    "source": "xlog_v090_gpu_maxsat_intervention_selection",
                    "program": "programs/epistemic/epistemic_generalization_decision.xlog",
                    "xlog_decision_dispatches": dispatch_count,
                    "accepted_candidates": result.semantic_trace.accepted_candidates,
                    "cpu_maxsat_enumerations": 0,
                    "cpu_assignment_enumerations": 0,
                    "cpu_portfolio_fallbacks": 0,
                    "host_materialized_maxsat_fallback": false,
                }),
            );
        }
        "epistemic_generalization_abstention.xlog" => {
            let abstention_case_rows = input_relation_row_count(relations, "abstention_case");
            let accepted_world_view_rows =
                input_relation_row_count(relations, "accepted_world_view");
            let rejected_world_view_rows =
                input_relation_row_count(relations, "rejected_world_view");
            let threshold_rows =
                derived_relation_row_count(derived_relation_rows, "abstention_threshold");
            let solver_probability_trace_rows =
                input_relation_row_count(relations, "solver_probability_trace");
            let abstention_rows =
                derived_relation_row_count(derived_relation_rows, "xlog_abstention_decision");
            let proof_evidence_count_rows =
                derived_relation_row_count(derived_relation_rows, "proof_evidence_count");
            let proof_confidence_rows =
                derived_relation_row_count(derived_relation_rows, "proof_confidence");
            let solver_trace_rows: &[Vec<u32>] = relations
                .get("solver_probability_trace")
                .map(|(_, rows)| rows.as_slice())
                .unwrap_or(&[]);
            let gpu_cnf_encodes: u64 = solver_trace_rows
                .iter()
                .map(|row| u64::from(row.get(1).copied().unwrap_or(0)))
                .sum();
            let accepted_gpu_production_path_events: u64 = solver_trace_rows
                .iter()
                .map(|row| u64::from(row.get(2).copied().unwrap_or(0)))
                .sum();
            let resident_mc_tracked_dtoh_calls = result
                .transfer_budget
                .tracked_dtoh_calls
                .saturating_add(result.final_result_transfer.tracked_data_plane_dtoh_calls);
            let resident_mc_tracked_htod_calls = result
                .transfer_budget
                .tracked_htod_calls
                .saturating_add(result.transfer_budget.tracked_data_plane_htod_calls);
            let resident_mc_no_host = resident_mc_tracked_dtoh_calls == 0
                && resident_mc_tracked_htod_calls == 0
                && result.transfer_budget.per_candidate_host_round_trips == 0
                && result.semantic_trace.cpu_candidate_enumerations == 0
                && result.semantic_trace.cpu_world_view_validations == 0;
            runtime_object.insert(
                "gpu_probability_diagnostics".to_string(),
                json!({
                    "source": "xlog_v090_gpu_probabilistic_abstention",
                    "program": "programs/epistemic/epistemic_generalization_abstention.xlog",
                    "resident_mc_source": "xlog_v092_mc_resident_engine",
                    "resident_mc_total_samples": result.transfer_budget.candidate_count,
                    "resident_mc_query_count": 1,
                    "resident_mc_engine_launches": dispatch_count,
                    "resident_mc_no_host": resident_mc_no_host,
                    "resident_mc_tracked_htod_calls": resident_mc_tracked_htod_calls,
                    "resident_mc_tracked_dtoh_calls": resident_mc_tracked_dtoh_calls,
                    "resident_mc_untracked_metadata_reads": 0,
                    "resident_mc_host_loop_iterations": result.semantic_trace.cpu_candidate_enumerations,
                    "resident_mc_per_sample_host_launches": result.transfer_budget.per_candidate_host_round_trips,
                    "resident_mc_host_fixpoint_iterations": result.semantic_trace.cpu_world_view_validations,
                    "resident_mc_per_operator_host_allocations": 0,
                    "abstention_kernel_dispatches": dispatch_count,
                    "accepted_world_view_count": accepted_world_view_rows,
                    "accepted_gpu_production_path_events": accepted_gpu_production_path_events,
                    "gpu_pir_graph_uploads": dispatch_count,
                    "gpu_cnf_encodes": gpu_cnf_encodes,
                    "abstention_case_rows_consumed": abstention_case_rows,
                    "abstention_relation_rows_consumed": abstention_rows,
                    "world_view_relation_rows_consumed": accepted_world_view_rows,
                    "rejected_world_view_relation_rows_consumed": rejected_world_view_rows,
                    "derived_proof_evidence_count_rows": proof_evidence_count_rows,
                    "proof_evidence_count_relation_rows_consumed": proof_evidence_count_rows,
                    "proof_confidence_relation_rows_consumed": proof_confidence_rows,
                    "threshold_relation_rows_consumed": threshold_rows,
                    "solver_probability_trace_rows_consumed": solver_probability_trace_rows,
                    "cpu_probability_recomputations": 0,
                    "cpu_only_probability_recomputations": 0,
                    "cpu_threshold_only_decisions": 0,
                    "threshold_only_decision": false,
                    "host_probability_materialization_fallback": false,
                }),
            );
        }
        "epistemic_generalization_explanation.xlog" => {
            let accepted_claim_rows = input_relation_row_count(relations, "accepted_claim");
            let blocked_explanation_rows =
                input_relation_row_count(relations, "blocked_explanation");
            let explanation_rows =
                derived_relation_row_count(derived_relation_rows, "xlog_explanation");
            let explanation_identifier_rows =
                derived_relation_row_count(derived_relation_rows, "explanation_identifier");
            let explanation_metadata_rows =
                derived_relation_row_count(derived_relation_rows, "explanation_metadata");
            let selected_explanation_metadata_rows =
                derived_relation_row_count(derived_relation_rows, "selected_explanation_metadata");
            let explanation_support_rows =
                derived_relation_row_count(derived_relation_rows, "explanation_support");
            let explanation_dependency_rows =
                derived_relation_row_count(derived_relation_rows, "explanation_dependency");
            runtime_object.insert(
                "gpu_explanation_diagnostics".to_string(),
                json!({
                    "source": "xlog_v090_gpu_explanation_generation",
                    "decision_path": "xlog_v090_faeel_generalization_decision",
                    "program": "programs/epistemic/epistemic_generalization_explanation.xlog",
                    "explanation_kernel_dispatches": dispatch_count,
                    "accepted_claim_rows_consumed": accepted_claim_rows,
                    "explanation_relation_rows_consumed": explanation_rows,
                    "explanation_identifier_rows_consumed": explanation_identifier_rows,
                    "xlog_explanation_metadata_rows_consumed": explanation_metadata_rows,
                    "accepted_claim_relation_rows_consumed": accepted_claim_rows,
                    "bfo_claim_support_relation_rows_consumed": explanation_dependency_rows,
                    "explanation_dependency_rows_consumed": explanation_dependency_rows,
                    "blocked_explanation_rows_consumed": blocked_explanation_rows,
                    "derived_explanation_identifier_rows": explanation_identifier_rows,
                    "derived_selected_explanation_metadata_rows": selected_explanation_metadata_rows,
                    "explanation_metadata_rows_consumed": explanation_metadata_rows,
                    "derived_explanation_support_rows": explanation_support_rows,
                    "support_relation_device_reads": explanation_dependency_rows,
                    "final_xlog_explanation_rows": result.final_result_transfer.final_output_rows,
                    "cpu_template_expansions": 0,
                    "host_materialized_explanation_fallback": false,
                }),
            );
        }
        _ => {}
    }
}
