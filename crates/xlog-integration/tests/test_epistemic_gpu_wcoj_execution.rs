use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RelId, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::epistemic::compile_epistemic_gpu_execution_with_stats_snapshot;
use xlog_logic::{parse_program, Compiler};
use xlog_prob::epistemic::EpistemicAssumption;
use xlog_prob::epistemic_production::{
    EpistemicProbGpuExecutionEvidence, EpistemicProbProductionAdapter,
};
use xlog_prob::exact::GpuConfig;
use xlog_runtime::{
    read_device_row_count, EpistemicGpuModelMembershipSource, EpistemicGpuRuntimePreflight,
    EpistemicGpuRuntimeWcojCertification, EpistemicGpuWorkspaceCapacities, Executor,
};
use xlog_solve::{
    Clause, GpuCdclConfig, GpuCnf, GpuSolverProductionAdapter, GpuSolverProductionExpectation,
    GpuSolverProductionLifecycleStep, GpuSolverProductionMaxSatCandidate,
    GpuSolverProductionPortfolioJob, Literal, SolveInstance,
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

fn upload_u32_scalar(provider: &Arc<CudaKernelProvider>, value: u32) -> TrackedCudaSlice<u32> {
    let mut slot = provider.memory().alloc::<u32>(1).expect("alloc u32 scalar");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[value], &mut slot)
        .expect("upload u32 scalar");
    slot
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

fn download_binary_u32(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<(u32, u32)> {
    let rows = read_device_row_count(provider, buffer).expect("device row count");
    if rows == 0 {
        return Vec::new();
    }
    let mut col0 = vec![0u8; rows * std::mem::size_of::<u32>()];
    let mut col1 = vec![0u8; rows * std::mem::size_of::<u32>()];
    unsafe {
        sys::cuMemcpyDtoH_v2(
            col0.as_mut_ptr() as *mut _,
            *buffer.column(0).expect("binary column 0").device_ptr(),
            col0.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col1.as_mut_ptr() as *mut _,
            *buffer.column(1).expect("binary column 1").device_ptr(),
            col1.len(),
        );
    }
    col0.chunks_exact(std::mem::size_of::<u32>())
        .zip(col1.chunks_exact(std::mem::size_of::<u32>()))
        .map(|(a, b)| {
            (
                u32::from_le_bytes(a.try_into().unwrap()),
                u32::from_le_bytes(b.try_into().unwrap()),
            )
        })
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
    assert_eq!(result.prepared.preflight.sorted_layout_requirement_count, 1);
    assert_eq!(result.prepared.preflight.helper_split_spec_count, 1);
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
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(result.final_result_transfer.final_output_column_count, 5);
    assert_eq!(
        result.final_result_transfer.final_output_payload_bytes,
        5 * std::mem::size_of::<u32>() as u64
    );
    assert_eq!(result.final_result_transfer.row_count_device_reads, 1);
    assert_eq!(
        result.final_result_transfer.tracked_data_plane_dtoh_calls,
        0
    );
    assert_eq!(
        result.final_result_transfer.tracked_data_plane_dtoh_bytes,
        0
    );
    assert_eq!(
        read_device_row_count(&fix.provider, &result.final_output).expect("final row count"),
        1,
        "accepted epistemic K5 final output must materialize the production WCOJ row"
    );
}

#[test]
fn epistemic_k7_k8_reductions_reuse_g39_kclique_planner_preflight_surface() {
    for k in [7u8, 8u8] {
        let source = epistemic_kclique_source(k, true);
        let program = parse_program(&source).expect("parse epistemic K-clique");
        let rel_ids = rel_ids_for_reduced_kclique(k);
        let stats = kclique_stats(&rel_ids, k, Some((k - 2, 5.0)));
        let executable =
            compile_epistemic_gpu_execution_with_stats_snapshot(&program, Some(&stats))
                .expect("compile epistemic executable K-clique");

        let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 1,
            },
        )
        .expect("preflight epistemic K-clique");

        let expected_edges = usize::from(k) * usize::from(k - 1) / 2;
        assert_eq!(preflight.kclique_wcoj_plan_count, 1);
        assert_eq!(preflight.kclique_wcoj_max_arity, k);
        assert_eq!(
            preflight.kclique_wcoj_edge_permutation_count,
            expected_edges
        );
        assert_eq!(preflight.planned_hash_route_count, 0);
        assert!(
            preflight.sorted_layout_requirement_count >= 1,
            "K{k} must carry production sorted-layout requirements"
        );
        assert!(preflight.cpu_fallbacks.is_zero());
    }
}

#[test]
fn accepted_epistemic_k7_execution_certifies_production_wcoj_dispatch() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let source = epistemic_kclique_source(7, true);
    let program = parse_program(&source).expect("parse epistemic K7");
    let rel_ids = rel_ids_for_reduced_kclique(7);
    let stats = kclique_stats(&rel_ids, 7, Some((5, 5.0)));
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, Some(&stats))
        .expect("compile epistemic executable K7");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in k_clique_inputs(7) {
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
        .expect("execute accepted epistemic K7");

    assert_eq!(result.prepared.preflight.kclique_wcoj_plan_count, 1);
    assert_eq!(result.prepared.preflight.kclique_wcoj_max_arity, 7);
    assert_eq!(
        result
            .prepared
            .preflight
            .kclique_wcoj_edge_permutation_count,
        21
    );
    assert!(
        result.prepared.preflight.sorted_layout_requirement_count >= 1,
        "accepted K7 must carry production sorted-layout requirements"
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
        result.trace.counter_delta.wcoj_clique7_dispatch_count >= 1,
        "accepted epistemic K7 must dispatch through production K7 WCOJ"
    );
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(result.final_result_transfer.final_output_column_count, 7);
    assert_eq!(
        result.final_result_transfer.final_output_payload_bytes,
        7 * std::mem::size_of::<u32>() as u64
    );
    assert_eq!(
        read_device_row_count(&fix.provider, &result.final_output).expect("final row count"),
        1,
        "accepted epistemic K7 final output must materialize the production WCOJ row"
    );
}

#[test]
fn accepted_epistemic_k8_execution_certifies_production_wcoj_dispatch() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let source = epistemic_kclique_source(8, true);
    let program = parse_program(&source).expect("parse epistemic K8");
    let rel_ids = rel_ids_for_reduced_kclique(8);
    let stats = kclique_stats(&rel_ids, 8, Some((6, 5.0)));
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, Some(&stats))
        .expect("compile epistemic executable K8");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in k_clique_inputs(8) {
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
        .expect("execute accepted epistemic K8");

    assert_eq!(result.prepared.preflight.kclique_wcoj_plan_count, 1);
    assert_eq!(result.prepared.preflight.kclique_wcoj_max_arity, 8);
    assert_eq!(
        result
            .prepared
            .preflight
            .kclique_wcoj_edge_permutation_count,
        28
    );
    assert!(
        result.prepared.preflight.sorted_layout_requirement_count >= 1,
        "accepted K8 must carry production sorted-layout requirements"
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
        result.trace.counter_delta.wcoj_clique8_dispatch_count >= 1,
        "accepted epistemic K8 must dispatch through production K8 WCOJ"
    );
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(result.final_result_transfer.final_output_column_count, 8);
    assert_eq!(
        result.final_result_transfer.final_output_payload_bytes,
        8 * std::mem::size_of::<u32>() as u64
    );
    assert_eq!(
        read_device_row_count(&fix.provider, &result.final_output).expect("final row count"),
        1,
        "accepted epistemic K8 final output must materialize the production WCOJ row"
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
fn accepted_gpu_execution_records_device_semantic_trace_counts() {
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
    .expect("parse semantic-trace epistemic fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile semantic-trace epistemic executable");

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
        .expect("execute semantic-trace epistemic fixture");

    assert_eq!(result.semantic_trace.generated_candidates, 2);
    assert_eq!(result.semantic_trace.propagated_candidates, 2);
    assert_eq!(result.semantic_trace.tested_candidates, 2);
    assert_eq!(result.semantic_trace.reduced_model_slots_checked, 4);
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidates, 1);
    assert_eq!(result.semantic_trace.rejection_reasons, vec![5]);
    assert_eq!(result.semantic_trace.rejection_reason_device_reads, 1);
    assert_eq!(
        result.semantic_trace.rejection_reason_metadata_bytes,
        2 * std::mem::size_of::<u32>() as u64
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
}

#[test]
fn accepted_not_know_nonzero_arity_membership_filters_final_rows_by_absent_bound_tuple_key() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), not know edge(X).
        "#,
    )
    .expect("parse negated nonzero-arity epistemic fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile negated nonzero-arity epistemic executable");

    assert!(executable.gpu_plan.tuple_membership_bindings[0].negated);

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute negated nonzero-arity epistemic fixture");

    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![2],
        "not know must keep only reduced rows whose bound tuple key is absent from the stable model"
    );
}

#[test]
fn accepted_binary_membership_filters_final_rows_by_bound_tuple_key() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred accepted(u32, u32).
        accepted(X, Y) :- pair(X, Y), know edge(X, Y).
        "#,
    )
    .expect("parse binary epistemic fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile binary epistemic executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("pair", upload_binary_u32(&fix.memory, &[(1, 2), (2, 3)]));
    executor.put_relation("edge", upload_binary_u32(&fix.memory, &[(1, 2)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute binary epistemic fixture");

    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(
        download_binary_u32(&fix.provider, &result.final_output),
        vec![(1, 2)],
        "final output must keep only reduced rows whose binary tuple key appears in the stable model"
    );
}

#[test]
fn accepted_multiple_memberships_filter_final_rows_by_all_bound_tuple_keys() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X), know color(X).
        "#,
    )
    .expect("parse multi-membership epistemic fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile multi-membership epistemic executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 2]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2, 3]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute multi-membership epistemic fixture");

    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 2);
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(
        result
            .final_tuple_materialization
            .model_membership_bytes_checked,
        result.model_membership.model_membership_bytes_written
    );
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![2],
        "final output must keep only rows accepted by all bound tuple-key memberships"
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

#[test]
fn accepted_gpu_execution_result_gates_probabilistic_program_compile_path() {
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

    let prob_program = parse_program(
        r#"
        0.5::rain().
        query(rain()).
        "#,
    )
    .expect("parse probabilistic program");
    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let _exact = adapter
        .compile_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known("edge", 1, true)],
        )
        .expect("accepted GPU runtime evidence must gate probabilistic program compile path");

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.gpu_exact_program_compiles, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_result_gates_probabilistic_query_evaluation_path() {
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
    let exact = adapter
        .compile_source_with_gpu_execution_result(
            r#"
            0.5::rain().
            query(rain()).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known("edge", 1, true)],
        )
        .expect("accepted GPU runtime evidence must gate probabilistic exact compile");
    let evaluated = adapter
        .evaluate_with_gpu_execution_result(
            &exact,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known("edge", 1, true)],
        )
        .expect("accepted GPU runtime evidence must gate probabilistic query evaluation");

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!((evaluated.query_probs[0].prob - 0.5).abs() < 1.0e-6);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_result_gates_probabilistic_end_to_end_knowledge_compilation_path() {
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
    let evaluated = adapter
        .compile_and_evaluate_source_with_gpu_execution_result(
            r#"
            0.5::rain().
            query(rain()).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known("edge", 1, true)],
        )
        .expect("accepted GPU runtime evidence must gate end-to-end knowledge compilation");

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!((evaluated.query_probs[0].prob - 0.5).abs() < 1.0e-6);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_results_gate_batched_probabilistic_knowledge_compilation_path() {
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

    let make_result = |edge_rows: &[u32]| {
        let mut executor =
            Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
        for (name, rel_id) in &executable.relation_ids {
            executor.register_relation(*rel_id, name);
        }
        executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2]));
        executor.put_relation("edge", upload_unary_u32(&fix.memory, edge_rows));

        executor
            .execute_epistemic_gpu_execution(
                &executable,
                EpistemicGpuWorkspaceCapacities {
                    max_candidates: 2,
                    max_worlds: 1,
                    max_models_per_reduction: 2,
                },
            )
            .expect("execute accepted epistemic fixture")
    };
    let result_a = make_result(&[1]);
    let result_b = make_result(&[2]);
    let assumptions_a = [EpistemicAssumption::known("edge", 1, true)];
    let assumptions_b = [EpistemicAssumption::known("edge", 1, true)];

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_source_for_gpu_execution_results(
            r#"
            0.25::rain().
            query(rain()).
            "#,
            &fix.provider,
            &[
                EpistemicProbGpuExecutionEvidence {
                    result: &result_a,
                    assumptions: &assumptions_a,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &result_b,
                    assumptions: &assumptions_b,
                },
            ],
        )
        .expect("accepted GPU runtime evidence must gate batched knowledge compilation");

    assert_eq!(evaluated.len(), 2);
    for result in &evaluated {
        assert_eq!(result.query_probs.len(), 1);
        assert!((result.query_probs[0].prob - 0.25).abs() < 1.0e-6);
    }
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_result_conditions_zero_arity_probabilistic_evidence() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred gate().
        pred accepted(u32).
        gate().
        accepted(X) :- node(X), know gate().
        "#,
    )
    .expect("parse zero-arity epistemic fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile zero-arity epistemic executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1]));
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
        .expect("execute zero-arity accepted epistemic fixture");
    assert_eq!(
        read_device_row_count(&fix.provider, &result.final_output).unwrap(),
        1
    );

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
            0.7::gate().
            query(gate()).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known("gate", 0, true)],
        )
        .expect("accepted GPU runtime evidence must condition probabilistic exact evidence");

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!(
        (evaluated.query_probs[0].prob - 1.0).abs() < 1.0e-6,
        "accepted zero-arity know gate evidence must condition query probability to true"
    );
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_result_gates_probabilistic_program_end_to_end_path() {
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

    let prob_program = parse_program(
        r#"
        0.5::rain().
        query(rain()).
        "#,
    )
    .expect("parse probabilistic program");
    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known("edge", 1, true)],
        )
        .expect("accepted GPU runtime evidence must gate parsed-program knowledge compilation");

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!((evaluated.query_probs[0].prob - 0.5).abs() < 1.0e-6);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.gpu_exact_program_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_result_gates_probabilistic_pir_cnf_path() {
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
    let pir_cnf = adapter
        .encode_source_pir_cnf_with_gpu_execution_result(
            r#"
            0.5::rain().
            query(rain()).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known("edge", 1, true)],
        )
        .expect("accepted GPU runtime evidence must gate probabilistic PIR/CNF path");

    assert!(pir_cnf.pir_nodes > 0);
    assert!(pir_cnf.root_count > 0);
    assert!(pir_cnf.cnf_var_cap > 0);
    assert!(pir_cnf.cnf_clause_cap > 0);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.gpu_pir_graph_uploads, 1);
    assert_eq!(trace.gpu_cnf_encodes, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_result_gates_probabilistic_gradient_evaluation_path() {
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
    let exact = adapter
        .compile_source_with_gpu_execution_result(
            r#"
            0.5::rain().
            dry() :- not rain().
            query(dry()).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known("edge", 1, true)],
        )
        .expect("accepted GPU runtime evidence must gate probabilistic exact compile");
    let grads = adapter
        .evaluate_gpu_with_grads_with_gpu_execution_result(
            &exact,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known("edge", 1, true)],
        )
        .expect("accepted GPU runtime evidence must gate probabilistic gradient evaluation");

    assert_eq!(grads.query_grads.len(), 1);
    assert!((grads.query_grads[0].prob - 0.5).abs() < 1.0e-6);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_result_gates_solver_cdcl_sat_path() {
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

    let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let cnf = GpuCnf::from_host(&instance, &fix.provider).expect("upload SAT CNF");
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let _assignment = adapter
        .solve_expect_sat_with_gpu_execution_result(&fix.provider, &result, &cnf)
        .expect("accepted GPU runtime evidence must gate solver CDCL SAT path");

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_result_gates_solver_cdcl_unsat_path() {
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

    let instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let cnf = GpuCnf::from_host(&instance, &fix.provider).expect("upload UNSAT CNF");
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    adapter
        .solve_expect_unsat_with_gpu_execution_result(&fix.provider, &result, &cnf)
        .expect("accepted GPU runtime evidence must gate solver CDCL UNSAT path");

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_cdcl_unsat_solves, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_result_gates_solver_workspace_unsat_path() {
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

    let instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let cnf = GpuCnf::from_host(&instance, &fix.provider).expect("upload UNSAT CNF");
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(cnf.var_cap, cnf.clause_cap)
        .expect("new workspace");
    let assign_ptr_before = workspace.assign_device_ptr();
    let branch_limit = upload_u32_scalar(&fix.provider, 1);

    adapter
        .solve_expect_unsat_with_branch_limit_ws_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut workspace,
            &cnf,
            &branch_limit,
        )
        .expect("accepted GPU runtime evidence must gate workspace UNSAT path");

    assert_eq!(
        workspace.assign_device_ptr(),
        assign_ptr_before,
        "accepted workspace UNSAT gate must reuse the allocated GPU CDCL workspace"
    );
    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_result_gates_solver_assumption_lifecycle_path() {
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

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let unsat_cnf = GpuCnf::from_host(&unsat_instance, &fix.provider).expect("upload UNSAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(unsat_cnf.var_cap, unsat_cnf.clause_cap)
        .expect("new workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let report = adapter
        .solve_assumption_lifecycle_with_gpu_execution_result(
            &fix.provider,
            &result,
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
        )
        .expect("accepted GPU runtime evidence must gate solver lifecycle path");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.steps, 2);
    assert_eq!(report.assumption_pushes, 2);
    assert_eq!(report.assumption_retractions, 2);
    assert_eq!(report.workspace_reuses, 1);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_assumption_pushes, 2);
    assert_eq!(trace.gpu_assumption_retractions, 2);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 1);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_result_gates_status_aware_solver_lifecycle_path() {
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

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let unsat_cnf = GpuCnf::from_host(&unsat_instance, &fix.provider).expect("upload UNSAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(unsat_cnf.var_cap, unsat_cnf.clause_cap)
        .expect("new workspace");

    let report = adapter
        .solve_assumption_lifecycle_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut workspace,
            &[
                GpuSolverProductionLifecycleStep {
                    cnf: &sat_cnf,
                    branch_var_limit: &branch_limit,
                    expectation: GpuSolverProductionExpectation::Sat,
                },
                GpuSolverProductionLifecycleStep {
                    cnf: &sat_cnf,
                    branch_var_limit: &branch_limit,
                    expectation: GpuSolverProductionExpectation::Unknown {
                        reason: "bounded branch budget exhausted before a determined status",
                    },
                },
                GpuSolverProductionLifecycleStep {
                    cnf: &sat_cnf,
                    branch_var_limit: &branch_limit,
                    expectation: GpuSolverProductionExpectation::Timeout { budget_micros: 10 },
                },
                GpuSolverProductionLifecycleStep {
                    cnf: &unsat_cnf,
                    branch_var_limit: &branch_limit,
                    expectation: GpuSolverProductionExpectation::Unsat,
                },
            ],
        )
        .expect("accepted GPU runtime evidence must gate status-aware solver lifecycle path");

    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.steps, 4);
    assert_eq!(report.assumption_pushes, 4);
    assert_eq!(report.assumption_retractions, 4);
    assert_eq!(report.workspace_reuses, 1);
    assert_eq!(report.unknown_steps, 1);
    assert_eq!(report.timeout_steps, 1);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_assumption_pushes, 4);
    assert_eq!(trace.gpu_assumption_retractions, 4);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(trace.gpu_lifecycle_unknown_status_steps, 1);
    assert_eq!(trace.gpu_lifecycle_timeout_status_steps, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_results_gate_multi_candidate_solver_lifecycle_path() {
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

    let make_result = |edge_rows: &[u32]| {
        let mut executor =
            Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
        for (name, rel_id) in &executable.relation_ids {
            executor.register_relation(*rel_id, name);
        }
        executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2]));
        executor.put_relation("edge", upload_unary_u32(&fix.memory, edge_rows));

        executor
            .execute_epistemic_gpu_execution(
                &executable,
                EpistemicGpuWorkspaceCapacities {
                    max_candidates: 2,
                    max_worlds: 1,
                    max_models_per_reduction: 2,
                },
            )
            .expect("execute accepted epistemic fixture")
    };
    let result_a = make_result(&[1]);
    let result_b = make_result(&[2]);

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let unsat_cnf = GpuCnf::from_host(&unsat_instance, &fix.provider).expect("upload UNSAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(unsat_cnf.var_cap, unsat_cnf.clause_cap)
        .expect("new workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let report = adapter
        .solve_multi_candidate_assumption_lifecycle_with_gpu_execution_results(
            &fix.provider,
            &[&result_a, &result_b],
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
        )
        .expect("accepted GPU runtime evidence must gate multi-candidate solver lifecycle path");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.steps, 4);
    assert_eq!(report.assumption_pushes, 4);
    assert_eq!(report.assumption_retractions, 4);
    assert_eq!(report.workspace_reuses, 2);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.gpu_assumption_pushes, 4);
    assert_eq!(trace.gpu_assumption_retractions, 4);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 2);
    assert_eq!(trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_result_gates_solver_learned_clause_arena_publication() {
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

    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let unsat_cnf = GpuCnf::from_host(&unsat_instance, &fix.provider).expect("upload UNSAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(unsat_cnf.var_cap, unsat_cnf.clause_cap)
        .expect("new workspace");

    let report = adapter
        .solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut workspace,
            &unsat_cnf,
            &branch_limit,
        )
        .expect("accepted GPU runtime evidence must gate learned-clause arena publication");

    assert_eq!(report.unsat_solves, 1);
    assert_eq!(report.gpu_learned_clause_arena_publications, 1);
    assert_eq!(report.gpu_learned_count_buffer_publications, 1);
    assert_eq!(report.cpu_learned_clause_transfers, 0);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(trace.gpu_learned_clause_arena_publications, 1);
    assert_eq!(trace.gpu_learned_count_buffer_publications, 1);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_result_gates_solver_same_cnf_learned_clause_reuse() {
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

    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let unsat_cnf = GpuCnf::from_host(&unsat_instance, &fix.provider).expect("upload UNSAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(unsat_cnf.var_cap, unsat_cnf.clause_cap)
        .expect("new workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let report = adapter
        .solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut workspace,
            &unsat_cnf,
            &branch_limit,
            &unsat_cnf,
            &branch_limit,
        )
        .expect("accepted GPU runtime evidence must gate learned-clause reuse");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.candidates, 2);
    assert_eq!(report.unsat_solves, 2);
    assert_eq!(report.gpu_learned_clause_arena_publications, 1);
    assert_eq!(report.gpu_learned_clause_imports, 1);
    assert_eq!(report.gpu_learned_clause_reused_solves, 1);
    assert_eq!(report.cpu_learned_clause_transfers, 0);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.gpu_learned_clause_arena_publications, 1);
    assert_eq!(trace.gpu_learned_clause_imports, 1);
    assert_eq!(trace.gpu_learned_clause_reused_solves, 1);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_results_gate_multi_candidate_learned_clause_reuse() {
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

    let make_result = |edge_rows: &[u32]| {
        let mut executor =
            Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
        for (name, rel_id) in &executable.relation_ids {
            executor.register_relation(*rel_id, name);
        }
        executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2]));
        executor.put_relation("edge", upload_unary_u32(&fix.memory, edge_rows));

        executor
            .execute_epistemic_gpu_execution(
                &executable,
                EpistemicGpuWorkspaceCapacities {
                    max_candidates: 2,
                    max_worlds: 1,
                    max_models_per_reduction: 2,
                },
            )
            .expect("execute accepted epistemic fixture")
    };
    let result_a = make_result(&[1]);
    let result_b = make_result(&[2]);

    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let unsat_cnf = GpuCnf::from_host(&unsat_instance, &fix.provider).expect("upload UNSAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(unsat_cnf.var_cap, unsat_cnf.clause_cap)
        .expect("new workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let report = adapter
        .solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results(
            &fix.provider,
            &[&result_a, &result_b],
            &mut workspace,
            &unsat_cnf,
            &branch_limit,
            &unsat_cnf,
            &branch_limit,
        )
        .expect("accepted GPU runtime evidence must gate multi-candidate learned-clause reuse");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.candidates, 4);
    assert_eq!(report.unsat_solves, 4);
    assert_eq!(report.gpu_learned_clause_arena_publications, 2);
    assert_eq!(report.gpu_learned_clause_imports, 2);
    assert_eq!(report.gpu_learned_clause_reused_solves, 2);
    assert_eq!(report.cpu_learned_clause_transfers, 0);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 4);
    assert_eq!(trace.gpu_learned_clause_arena_publications, 2);
    assert_eq!(trace.gpu_learned_clause_imports, 2);
    assert_eq!(trace.gpu_learned_clause_reused_solves, 2);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_result_rejects_distinct_cnf_learned_clause_reuse() {
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

    let source_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let target_instance = SolveInstance::new(
        2,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let source_cnf =
        GpuCnf::from_host(&source_instance, &fix.provider).expect("upload source UNSAT CNF");
    let target_cnf =
        GpuCnf::from_host(&target_instance, &fix.provider).expect("upload target UNSAT CNF");
    let source_branch_limit = upload_u32_scalar(&fix.provider, 1);
    let target_branch_limit = upload_u32_scalar(&fix.provider, 2);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(target_cnf.var_cap, target_cnf.clause_cap)
        .expect("new workspace");

    let err = adapter
        .solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut workspace,
            &source_cnf,
            &source_branch_limit,
            &target_cnf,
            &target_branch_limit,
        )
        .expect_err("distinct candidate CNFs must reject learned-clause import");

    assert!(format!("{err}").contains("distinct candidate CNFs"));

    let trace = adapter.trace();
    assert_eq!(trace.gpu_learned_clause_reuse_rejections, 1);
    assert_eq!(trace.gpu_learned_clause_arena_publications, 0);
    assert_eq!(trace.gpu_learned_clause_imports, 0);
    assert_eq!(trace.gpu_learned_clause_reused_solves, 0);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_result_gates_solver_maxsat_and_portfolio_paths() {
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

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let maxsat_candidate =
        GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload MaxSAT candidate CNF");
    let portfolio_sat =
        GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload portfolio SAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());

    let maxsat = adapter
        .solve_weighted_maxsat_candidates_with_gpu_execution_result(
            &fix.provider,
            &result,
            &[GpuSolverProductionMaxSatCandidate {
                score: 5,
                cnf: &maxsat_candidate,
                branch_var_limit: &branch_limit,
            }],
        )
        .expect("accepted GPU runtime evidence must gate MaxSAT through GPU CDCL");
    assert_eq!(maxsat.optimum_score, 5);
    assert_eq!(maxsat.gpu_cdcl_candidate_solves, 1);

    let portfolio = adapter
        .solve_portfolio_with_gpu_execution_result(
            &fix.provider,
            &result,
            &[
                GpuSolverProductionPortfolioJob::Sat {
                    cnf: &portfolio_sat,
                    branch_var_limit: &branch_limit,
                },
                GpuSolverProductionPortfolioJob::MaxSat {
                    candidates: &[GpuSolverProductionMaxSatCandidate {
                        score: 5,
                        cnf: &maxsat_candidate,
                        branch_var_limit: &branch_limit,
                    }],
                },
                GpuSolverProductionPortfolioJob::Unknown {
                    reason: "bounded branch budget exhausted before a determined status",
                },
                GpuSolverProductionPortfolioJob::Timeout { budget_micros: 10 },
            ],
        )
        .expect("accepted GPU runtime evidence must gate status-aware portfolio path");
    assert_eq!(portfolio.jobs, 4);
    assert_eq!(portfolio.sat_jobs, 1);
    assert_eq!(portfolio.maxsat_jobs, 1);
    assert_eq!(portfolio.unknown_jobs, 1);
    assert_eq!(portfolio.timeout_jobs, 1);
    assert_eq!(portfolio.maxsat_optimum_scores, 5);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(trace.gpu_maxsat_optima, 2);
    assert_eq!(trace.gpu_portfolio_jobs, 4);
    assert_eq!(trace.gpu_portfolio_sat_jobs, 1);
    assert_eq!(trace.gpu_portfolio_maxsat_jobs, 1);
    assert_eq!(trace.gpu_portfolio_unknown_status_jobs, 1);
    assert_eq!(trace.gpu_portfolio_timeout_status_jobs, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
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

fn rel_ids_for_reduced_kclique(k: u8) -> BTreeMap<String, RelId> {
    let mut compiler = Compiler::new();
    let source = epistemic_kclique_source(k, false);
    let _ = compiler
        .compile(&source)
        .expect("compile reduced K-clique source");
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

fn kclique_stats(
    rel_ids: &BTreeMap<String, RelId>,
    k: u8,
    hot: Option<(u8, f64)>,
) -> StatsSnapshot {
    let mut snapshot = StatsSnapshot::default();
    let mut edges = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            let name = format!("e{i}{j}");
            let rel = *rel_ids.get(&name).expect("edge rel id");
            snapshot.rel_names.push((rel, name));
            edges.push((rel, i, j));

            let mut stats = RelationStats::new(rel);
            stats.update_cardinality(10_000);
            for (col_idx, variable) in [(0usize, i), (1usize, j)] {
                let mut col = ColumnStats::new(col_idx, ScalarType::U32);
                col.update_distinct(10_000);
                stats.add_column(col);
                stats.add_prefix_degree(PrefixDegreeStats::new(col_idx, 1.0, 1.25));
                let heat = match hot {
                    Some((hot_var, hot_heat)) if hot_var == variable => hot_heat,
                    _ => 0.25,
                };
                stats.add_key_heat(KeyHeatStats::new(col_idx, heat, heat));
            }
            snapshot.relations.push(stats);
        }
    }

    for (left_idx, (left_rel, left_i, left_j)) in edges.iter().enumerate() {
        for (right_rel, right_i, right_j) in edges.iter().skip(left_idx + 1) {
            if left_i == right_i || left_i == right_j || left_j == right_i || left_j == right_j {
                let mut sel = JoinSelectivity::new(*left_rel, *right_rel);
                sel.set_keys(vec![0], vec![0]);
                sel.set_selectivity(0.001);
                snapshot.join_selectivities.push(sel);
            }
        }
    }

    snapshot
}

fn epistemic_kclique_source(k: u8, include_epistemic: bool) -> String {
    let mut source = String::new();
    for i in 0..k {
        for j in (i + 1)..k {
            source.push_str(&format!("pred e{i}{j}(u32, u32).\n"));
        }
    }
    source.push_str("pred gate().\n");
    source.push_str(&format!("pred clique{k}("));
    for idx in 0..k {
        if idx > 0 {
            source.push_str(", ");
        }
        source.push_str("u32");
    }
    source.push_str(").\n");
    source.push_str("gate().\n");
    source.push_str(&format!("clique{k}("));
    for idx in 0..k {
        if idx > 0 {
            source.push_str(", ");
        }
        source.push_str(&format!("V{idx}"));
    }
    source.push_str(") :-\n");

    let mut atoms = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            atoms.push(format!("e{i}{j}(V{i}, V{j})"));
        }
    }
    if include_epistemic {
        atoms.push("know gate()".to_string());
    }
    source.push_str("    ");
    source.push_str(&atoms.join(", "));
    source.push_str(".\n");
    source
}
