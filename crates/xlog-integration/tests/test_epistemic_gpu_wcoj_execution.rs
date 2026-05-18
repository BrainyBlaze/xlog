use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RelId, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::epistemic::compile_epistemic_gpu_execution_with_stats_snapshot;
use xlog_logic::{parse_program, Compiler};
use xlog_prob::epistemic::EpistemicAssumption;
use xlog_prob::epistemic_production::EpistemicProbProductionAdapter;
use xlog_prob::exact::GpuConfig;
use xlog_runtime::{
    read_device_row_count, EpistemicGpuModelMembershipSource, EpistemicGpuRuntimeWcojCertification,
    EpistemicGpuWorkspaceCapacities, Executor,
};
use xlog_stats::{
    ColumnStats, JoinSelectivity, KeyHeatStats, PrefixDegreeStats, RelationStats, StatsSnapshot,
};

struct DiscardSink;

impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct RuntimeBackedFixture {
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
}

fn make_runtime_backed_fixture() -> Option<RuntimeBackedFixture> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let async_r: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(AsyncCudaResource::new(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
    ));
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_r,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 64 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(64 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);

    Some(RuntimeBackedFixture { memory, provider })
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_column = (n as usize).max(1) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_column).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_column).expect("alloc col1");
    let mut device_row_count = memory.alloc::<u32>(1).expect("alloc row count");
    let dev = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let c1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        dev.htod_sync_copy_into(&c0, &mut col0).unwrap();
        dev.htod_sync_copy_into(&c1, &mut col1).unwrap();
    }
    dev.htod_sync_copy_into(&[n], &mut device_row_count)
        .unwrap();
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        device_row_count,
        Schema::new(vec![
            ("c0".to_string(), ScalarType::U32),
            ("c1".to_string(), ScalarType::U32),
        ]),
        n,
    )
}

fn upload_unary_u32(memory: &Arc<GpuMemoryManager>, rows: &[u32]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes = (n as usize).max(1) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes).expect("alloc unary col0");
    let mut device_row_count = memory.alloc::<u32>(1).expect("alloc unary row count");
    let dev = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows.iter().flat_map(|value| value.to_le_bytes()).collect();
        dev.htod_sync_copy_into(&c0, &mut col0).unwrap();
    }
    dev.htod_sync_copy_into(&[n], &mut device_row_count)
        .unwrap();
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into()],
        n as u64,
        device_row_count,
        Schema::new(vec![("c0".to_string(), ScalarType::U32)]),
        n,
    )
}

fn upload_nullary(memory: &Arc<GpuMemoryManager>, rows: u32) -> CudaBuffer {
    let mut device_row_count = memory.alloc::<u32>(1).expect("alloc nullary row count");
    memory
        .device()
        .inner()
        .htod_sync_copy_into(&[rows], &mut device_row_count)
        .unwrap();
    CudaBuffer::from_columns_with_host_count(
        vec![],
        rows as u64,
        device_row_count,
        Schema::new(vec![]),
        rows,
    )
}

fn download_unary_u32(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<u32> {
    let rows = read_device_row_count(provider, buffer).expect("device row count");
    if rows == 0 {
        return Vec::new();
    }
    let mut bytes = vec![0u8; rows * std::mem::size_of::<u32>()];
    unsafe {
        sys::cuMemcpyDtoH_v2(
            bytes.as_mut_ptr() as *mut _,
            *buffer.column(0).expect("unary column").device_ptr(),
            bytes.len(),
        );
    }
    bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn accepted_epistemic_k5_execution_certifies_production_wcoj_dispatch() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(EPISTEMIC_K5_SRC).expect("parse epistemic K5");
    let rel_ids = rel_ids_for_reduced_k5();
    let stats = k5_stats(&rel_ids);
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, Some(&stats))
        .expect("compile epistemic executable K5");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in k_clique_inputs(5) {
        executor.put_relation(&name, upload_binary_u32(&fix.memory, &rows));
    }
    executor.put_relation("gate", upload_nullary(&fix.memory, 1));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 1,
            },
        )
        .expect("execute accepted epistemic K5");

    assert_eq!(result.prepared.preflight.kclique_wcoj_plan_count, 1);
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert!(matches!(
        result.trace.wcoj_certification,
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches: 1..,
            observed_kclique_dispatches: 1..,
            ..
        }
    ));
    assert!(
        result.trace.counter_delta.wcoj_clique5_dispatch_count >= 1,
        "accepted epistemic K5 must dispatch through production K5 WCOJ"
    );
    assert_eq!(
        read_device_row_count(&fix.provider, &result.final_output).expect("final row count"),
        1,
        "accepted epistemic K5 final output must materialize the production WCOJ row"
    );
}

#[test]
fn accepted_nonzero_arity_membership_filters_final_rows_by_bound_tuple_key() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
    )
    .expect("parse nonzero-arity epistemic fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile nonzero-arity epistemic executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute nonzero-arity epistemic fixture");

    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![1],
        "final output must keep only reduced rows whose bound tuple key appears in the stable model"
    );
}

#[test]
fn accepted_gpu_execution_result_gates_probabilistic_exact_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
    )
    .expect("parse nonzero-arity epistemic fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile nonzero-arity epistemic executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute accepted epistemic fixture");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let _exact = adapter
        .compile_source_with_gpu_execution_result(
            r#"
            0.5::rain().
            query(rain()).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known("edge", 1, true)],
        )
        .expect("accepted GPU runtime evidence must gate probabilistic exact path");

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

const EPISTEMIC_K5_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32). pred e04(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32).
    pred e34(u32, u32).
    pred gate().
    pred clique5(u32, u32, u32, u32, u32).
    gate().
    clique5(V0, V1, V2, V3, V4) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4),
        e12(V1, V2), e13(V1, V3), e14(V1, V4),
        e23(V2, V3), e24(V2, V4),
        e34(V3, V4),
        know gate().
"#;

const REDUCED_K5_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32). pred e04(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32).
    pred e34(u32, u32).
    pred gate().
    pred clique5(u32, u32, u32, u32, u32).
    gate().
    clique5(V0, V1, V2, V3, V4) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4),
        e12(V1, V2), e13(V1, V3), e14(V1, V4),
        e23(V2, V3), e24(V2, V4),
        e34(V3, V4).
"#;

const K5_EDGES: [(&str, usize, usize); 10] = [
    ("e01", 0, 1),
    ("e02", 0, 2),
    ("e03", 0, 3),
    ("e04", 0, 4),
    ("e12", 1, 2),
    ("e13", 1, 3),
    ("e14", 1, 4),
    ("e23", 2, 3),
    ("e24", 2, 4),
    ("e34", 3, 4),
];

fn k_clique_inputs(k: usize) -> BTreeMap<String, Vec<(u32, u32)>> {
    let mut inputs = BTreeMap::new();
    for i in 0u32..(k as u32) {
        for j in (i + 1)..(k as u32) {
            inputs.insert(format!("e{}{}", i, j), vec![(i + 1, j + 1)]);
        }
    }
    inputs
}

fn rel_ids_for_reduced_k5() -> BTreeMap<String, RelId> {
    let mut compiler = Compiler::new();
    let _ = compiler
        .compile(REDUCED_K5_SRC)
        .expect("compile reduced K5");
    compiler
        .rel_ids()
        .iter()
        .map(|(name, rel)| (name.clone(), *rel))
        .collect()
}

fn k5_stats(rel_ids: &BTreeMap<String, RelId>) -> StatsSnapshot {
    let mut snapshot = StatsSnapshot::default();
    for (name, left, right) in K5_EDGES {
        let rel = *rel_ids.get(name).expect("edge rel id");
        snapshot.rel_names.push((rel, name.to_string()));
        let mut stats = RelationStats::new(rel);
        stats.update_cardinality(10_000);
        for (col_idx, variable) in [(0usize, left), (1usize, right)] {
            let mut col = ColumnStats::new(col_idx, ScalarType::U32);
            col.update_distinct(10_000);
            stats.add_column(col);
            stats.add_prefix_degree(PrefixDegreeStats::new(col_idx, 1.0, 1.25));
            let heat = if variable == 3 { 5.0 } else { 0.25 };
            stats.add_key_heat(KeyHeatStats::new(col_idx, heat, heat));
        }
        snapshot.relations.push(stats);
    }

    for (left_idx, (left_name, left_i, left_j)) in K5_EDGES.iter().enumerate() {
        let left_rel = *rel_ids.get(*left_name).expect("left rel id");
        for (right_name, right_i, right_j) in K5_EDGES.iter().skip(left_idx + 1) {
            if left_i == right_i || left_i == right_j || left_j == right_i || left_j == right_j {
                let right_rel = *rel_ids.get(*right_name).expect("right rel id");
                let mut sel = JoinSelectivity::new(left_rel, right_rel);
                sel.set_keys(vec![0], vec![0]);
                sel.set_selectivity(0.001);
                snapshot.join_selectivities.push(sel);
            }
        }
    }

    snapshot
}
