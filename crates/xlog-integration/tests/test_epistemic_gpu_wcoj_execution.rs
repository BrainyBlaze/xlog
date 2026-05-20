use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RelId, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::{EirEpistemicMode, EpistemicCpuFallbackCounters};
use xlog_logic::epistemic::{
    compile_epistemic_gpu_execution_with_stats_snapshot,
    compile_epistemic_gpu_split_execution_with_stats_snapshot, run_generate_propagate_test,
    run_generate_propagate_test_with_mode, EpistemicInterpretation, EpistemicSplitExecutablePlan,
    FaeelNoModelReason, GeneratePropagateTestConfig,
};
use xlog_logic::{parse_program, Compiler, EpistemicMode};
use xlog_prob::epistemic::{
    CircuitUpdateMode, EpistemicAssumption, EpistemicCircuit, EpistemicEvidenceTerm,
    KnowledgeCompilerAdapter,
};
use xlog_prob::epistemic_production::{
    EpistemicProbGpuBatchExecutionEvidence, EpistemicProbGpuExecutionEvidence,
    EpistemicProbProductionAdapter,
};
use xlog_prob::exact::GpuConfig;
use xlog_runtime::{
    read_device_row_count, EpistemicGpuBatchExecutionResult, EpistemicGpuBatchExecutionTrace,
    EpistemicGpuExecutionResult, EpistemicGpuModelMembershipSource, EpistemicGpuRejectionReason,
    EpistemicGpuRuntimePreflight, EpistemicGpuRuntimeWcojCertification,
    EpistemicGpuWorkspaceCapacities, Executor,
};
use xlog_solve::{
    Clause, GpuCdclConfig, GpuCnf, GpuSolverProductionAdapter,
    GpuSolverProductionBatchExecutionEvidence, GpuSolverProductionExpectation,
    GpuSolverProductionLifecycleStep, GpuSolverProductionMaxSatCandidate,
    GpuSolverProductionMaxSatScheduleJob, GpuSolverProductionMaxSatSearchCandidate,
    GpuSolverProductionMaxSatSearchStatus, GpuSolverProductionPortfolioJob,
    GpuSolverProductionWeightedMaxSatSelection, Literal, SolveInstance,
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

fn upload_ternary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_column = (n as usize).max(1) * std::mem::size_of::<u32>();
    let mut col0 = memory
        .alloc::<u8>(bytes_per_column)
        .expect("alloc ternary col0");
    let mut col1 = memory
        .alloc::<u8>(bytes_per_column)
        .expect("alloc ternary col1");
    let mut col2 = memory
        .alloc::<u8>(bytes_per_column)
        .expect("alloc ternary col2");
    let mut device_row_count = memory.alloc::<u32>(1).expect("alloc ternary row count");
    let dev = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows.iter().flat_map(|(a, _, _)| a.to_le_bytes()).collect();
        let c1: Vec<u8> = rows.iter().flat_map(|(_, b, _)| b.to_le_bytes()).collect();
        let c2: Vec<u8> = rows.iter().flat_map(|(_, _, c)| c.to_le_bytes()).collect();
        dev.htod_sync_copy_into(&c0, &mut col0).unwrap();
        dev.htod_sync_copy_into(&c1, &mut col1).unwrap();
        dev.htod_sync_copy_into(&c2, &mut col2).unwrap();
    }
    dev.htod_sync_copy_into(&[n], &mut device_row_count)
        .unwrap();
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into(), col2.into()],
        n as u64,
        device_row_count,
        Schema::new(vec![
            ("c0".to_string(), ScalarType::U32),
            ("c1".to_string(), ScalarType::U32),
            ("c2".to_string(), ScalarType::U32),
        ]),
        n,
    )
}

fn upload_quaternary_u32(
    memory: &Arc<GpuMemoryManager>,
    rows: &[(u32, u32, u32, u32)],
) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_column = (n as usize).max(1) * std::mem::size_of::<u32>();
    let mut col0 = memory
        .alloc::<u8>(bytes_per_column)
        .expect("alloc quaternary col0");
    let mut col1 = memory
        .alloc::<u8>(bytes_per_column)
        .expect("alloc quaternary col1");
    let mut col2 = memory
        .alloc::<u8>(bytes_per_column)
        .expect("alloc quaternary col2");
    let mut col3 = memory
        .alloc::<u8>(bytes_per_column)
        .expect("alloc quaternary col3");
    let mut device_row_count = memory.alloc::<u32>(1).expect("alloc quaternary row count");
    let dev = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows
            .iter()
            .flat_map(|(a, _, _, _)| a.to_le_bytes())
            .collect();
        let c1: Vec<u8> = rows
            .iter()
            .flat_map(|(_, b, _, _)| b.to_le_bytes())
            .collect();
        let c2: Vec<u8> = rows
            .iter()
            .flat_map(|(_, _, c, _)| c.to_le_bytes())
            .collect();
        let c3: Vec<u8> = rows
            .iter()
            .flat_map(|(_, _, _, d)| d.to_le_bytes())
            .collect();
        dev.htod_sync_copy_into(&c0, &mut col0).unwrap();
        dev.htod_sync_copy_into(&c1, &mut col1).unwrap();
        dev.htod_sync_copy_into(&c2, &mut col2).unwrap();
        dev.htod_sync_copy_into(&c3, &mut col3).unwrap();
    }
    dev.htod_sync_copy_into(&[n], &mut device_row_count)
        .unwrap();
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into(), col2.into(), col3.into()],
        n as u64,
        device_row_count,
        Schema::new(vec![
            ("c0".to_string(), ScalarType::U32),
            ("c1".to_string(), ScalarType::U32),
            ("c2".to_string(), ScalarType::U32),
            ("c3".to_string(), ScalarType::U32),
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

fn download_ternary_u32(
    provider: &CudaKernelProvider,
    buffer: &CudaBuffer,
) -> Vec<(u32, u32, u32)> {
    let rows = read_device_row_count(provider, buffer).expect("device row count");
    if rows == 0 {
        return Vec::new();
    }
    let mut col0 = vec![0u8; rows * std::mem::size_of::<u32>()];
    let mut col1 = vec![0u8; rows * std::mem::size_of::<u32>()];
    let mut col2 = vec![0u8; rows * std::mem::size_of::<u32>()];
    unsafe {
        sys::cuMemcpyDtoH_v2(
            col0.as_mut_ptr() as *mut _,
            *buffer.column(0).expect("ternary column 0").device_ptr(),
            col0.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col1.as_mut_ptr() as *mut _,
            *buffer.column(1).expect("ternary column 1").device_ptr(),
            col1.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col2.as_mut_ptr() as *mut _,
            *buffer.column(2).expect("ternary column 2").device_ptr(),
            col2.len(),
        );
    }
    col0.chunks_exact(std::mem::size_of::<u32>())
        .zip(col1.chunks_exact(std::mem::size_of::<u32>()))
        .zip(col2.chunks_exact(std::mem::size_of::<u32>()))
        .map(|((a, b), c)| {
            (
                u32::from_le_bytes(a.try_into().unwrap()),
                u32::from_le_bytes(b.try_into().unwrap()),
                u32::from_le_bytes(c.try_into().unwrap()),
            )
        })
        .collect()
}

fn download_quaternary_u32(
    provider: &CudaKernelProvider,
    buffer: &CudaBuffer,
) -> Vec<(u32, u32, u32, u32)> {
    let rows = read_device_row_count(provider, buffer).expect("device row count");
    if rows == 0 {
        return Vec::new();
    }
    let mut col0 = vec![0u8; rows * std::mem::size_of::<u32>()];
    let mut col1 = vec![0u8; rows * std::mem::size_of::<u32>()];
    let mut col2 = vec![0u8; rows * std::mem::size_of::<u32>()];
    let mut col3 = vec![0u8; rows * std::mem::size_of::<u32>()];
    unsafe {
        sys::cuMemcpyDtoH_v2(
            col0.as_mut_ptr() as *mut _,
            *buffer.column(0).expect("quaternary column 0").device_ptr(),
            col0.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col1.as_mut_ptr() as *mut _,
            *buffer.column(1).expect("quaternary column 1").device_ptr(),
            col1.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col2.as_mut_ptr() as *mut _,
            *buffer.column(2).expect("quaternary column 2").device_ptr(),
            col2.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col3.as_mut_ptr() as *mut _,
            *buffer.column(3).expect("quaternary column 3").device_ptr(),
            col3.len(),
        );
    }
    col0.chunks_exact(std::mem::size_of::<u32>())
        .zip(col1.chunks_exact(std::mem::size_of::<u32>()))
        .zip(col2.chunks_exact(std::mem::size_of::<u32>()))
        .zip(col3.chunks_exact(std::mem::size_of::<u32>()))
        .map(|(((a, b), c), d)| {
            (
                u32::from_le_bytes(a.try_into().unwrap()),
                u32::from_le_bytes(b.try_into().unwrap()),
                u32::from_le_bytes(c.try_into().unwrap()),
                u32::from_le_bytes(d.try_into().unwrap()),
            )
        })
        .collect()
}

fn execute_unary_edge_epistemic_fixture(
    fix: &RuntimeBackedFixture,
    source: &str,
    node_rows: &[u32],
    edge_rows: &[u32],
) -> EpistemicGpuExecutionResult {
    let program = parse_program(source).expect("parse unary edge epistemic fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile unary edge epistemic executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, node_rows));
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
        .expect("execute unary edge epistemic fixture")
}

fn execute_binary_edge_epistemic_fixture(
    fix: &RuntimeBackedFixture,
    source: &str,
    pair_rows: &[(u32, u32)],
    edge_rows: &[(u32, u32)],
) -> EpistemicGpuExecutionResult {
    let program = parse_program(source).expect("parse binary edge epistemic fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile binary edge epistemic executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("pair", upload_binary_u32(&fix.memory, pair_rows));
    executor.put_relation("edge", upload_binary_u32(&fix.memory, edge_rows));

    executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: pair_rows.len().max(1),
            },
        )
        .expect("execute binary edge epistemic fixture")
}

fn execute_split_all_binary_operator_batch(
    fix: &RuntimeBackedFixture,
) -> (
    EpistemicSplitExecutablePlan,
    EpistemicGpuBatchExecutionResult,
) {
    let program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred alt(u32, u32).
        pred blocked(u32, u32).
        pred seen(u32, u32).
        pred known_edge(u32, u32).
        pred possible_alt(u32, u32).
        pred clear_pair(u32, u32).
        pred unknown_pair(u32, u32).
        known_edge(X, Y) :- pair(X, Y), know edge(X, Y).
        possible_alt(X, Y) :- pair(X, Y), possible alt(X, Y).
        clear_pair(X, Y) :- pair(X, Y), not possible blocked(X, Y).
        unknown_pair(X, Y) :- pair(X, Y), not know seen(X, Y).
        "#,
    )
    .expect("parse split all binary operator fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split all binary operator components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split all-operator components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "pair",
        upload_binary_u32(&fix.memory, &[(1, 2), (2, 3), (3, 4)]),
    );
    executor.put_relation("edge", upload_binary_u32(&fix.memory, &[(1, 2), (3, 4)]));
    executor.put_relation("alt", upload_binary_u32(&fix.memory, &[(2, 3)]));
    executor.put_relation("blocked", upload_binary_u32(&fix.memory, &[(3, 4)]));
    executor.put_relation("seen", upload_binary_u32(&fix.memory, &[(1, 2)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split all binary operator components through GPU batch path");

    (split, batch)
}

fn execute_split_quaternary_all_operator_batch(
    fix: &RuntimeBackedFixture,
) -> (
    EpistemicSplitExecutablePlan,
    EpistemicGpuBatchExecutionResult,
) {
    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred edge4(u32, u32, u32, u32).
        pred alt4(u32, u32, u32, u32).
        pred blocked_fact4(u32, u32, u32, u32).
        pred hidden_fact4(u32, u32, u32, u32).
        pred known4(u32, u32, u32, u32).
        pred possible4(u32, u32, u32, u32).
        pred clear4(u32, u32, u32, u32).
        pred unknown4(u32, u32, u32, u32).
        known4(A, B, C, D) :- tuple4(A, B, C, D), know edge4(A, B, C, D).
        possible4(A, B, C, D) :- tuple4(A, B, C, D), possible alt4(A, B, C, D).
        clear4(A, B, C, D) :- tuple4(A, B, C, D), not possible blocked_fact4(A, B, C, D).
        unknown4(A, B, C, D) :- tuple4(A, B, C, D), not know hidden_fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary all-operator fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split quaternary all-operator components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split quaternary all-operator components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(
            &fix.memory,
            &[(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6), (9, 9, 9, 9)],
        ),
    );
    executor.put_relation("edge4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));
    executor.put_relation("alt4", upload_quaternary_u32(&fix.memory, &[(3, 4, 5, 6)]));
    executor.put_relation(
        "blocked_fact4",
        upload_quaternary_u32(&fix.memory, &[(9, 9, 9, 9)]),
    );
    executor.put_relation(
        "hidden_fact4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4)]),
    );

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split quaternary all-operator components through GPU batch path");

    (split, batch)
}

fn execute_all_operator_mixed_membership_fixture(
    fix: &RuntimeBackedFixture,
) -> EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred alt(u32).
        pred hidden(u32).
        pred blocked(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X), possible alt(X), not know hidden(X), not possible blocked(X).
        "#,
    )
    .expect("parse all-operator mixed-membership epistemic fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile all-operator mixed-membership epistemic executable");
    assert_eq!(executable.gpu_plan.epistemic_literals.len(), 4);
    assert_eq!(executable.gpu_plan.tuple_membership_bindings.len(), 4);
    assert_eq!(
        executable
            .gpu_plan
            .tuple_membership_bindings
            .iter()
            .filter(|binding| binding.negated)
            .count(),
        2,
        "not-know and not-possible memberships must compile as negated tuple-key filters"
    );

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3, 4]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 2]));
    executor.put_relation("alt", upload_unary_u32(&fix.memory, &[2, 3]));
    executor.put_relation("hidden", upload_unary_u32(&fix.memory, &[1, 4]));
    executor.put_relation("blocked", upload_unary_u32(&fix.memory, &[3, 4]));

    executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 16,
                max_worlds: 1,
                max_models_per_reduction: 4,
            },
        )
        .expect("execute all-operator mixed-membership epistemic fixture")
}

fn all_operator_mixed_membership_assumptions() -> Vec<EpistemicAssumption> {
    vec![
        EpistemicAssumption::known_tuple("edge", vec![EpistemicEvidenceTerm::integer(2)], true),
        EpistemicAssumption::possible_tuple("alt", vec![EpistemicEvidenceTerm::integer(2)], true),
        EpistemicAssumption::known_tuple("hidden", vec![EpistemicEvidenceTerm::integer(2)], false),
        EpistemicAssumption::possible_tuple(
            "blocked",
            vec![EpistemicEvidenceTerm::integer(2)],
            false,
        ),
    ]
}

#[test]
fn accepted_epistemic_v070_4cycle_execution_certifies_production_wcoj_dispatch() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred e1(u32, u32).
        pred e2(u32, u32).
        pred e3(u32, u32).
        pred e4(u32, u32).
        pred gate().
        pred cycle4(u32, u32, u32, u32).
        gate().
        cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W), know gate().
        "#,
    )
    .expect("parse epistemic v0.7 4-cycle fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile epistemic v0.7 4-cycle executable");

    let mut executor = Executor::new_with_config(
        Arc::clone(&fix.provider),
        RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(true)),
    );
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("e1", upload_binary_u32(&fix.memory, &[(1, 2)]));
    executor.put_relation("e2", upload_binary_u32(&fix.memory, &[(2, 3)]));
    executor.put_relation("e3", upload_binary_u32(&fix.memory, &[(3, 4)]));
    executor.put_relation("e4", upload_binary_u32(&fix.memory, &[(4, 1)]));
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
        .expect("execute accepted epistemic v0.7 4-cycle");

    assert_eq!(result.prepared.preflight.multiway_reduction_count, 1);
    assert_eq!(result.prepared.preflight.kclique_wcoj_plan_count, 0);
    assert!(matches!(
        result.trace.wcoj_certification,
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches: 1..,
            certified_multiway_reductions: 1,
            observed_kclique_dispatches: 0,
            certified_edge_permutation_slots: 0,
            ..
        }
    ));
    assert!(
        result.trace.counter_delta.wcoj_4cycle_dispatch_count >= 1,
        "accepted epistemic 4-cycle must dispatch through the v0.7.0 production 4-cycle WCOJ path"
    );
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(result.final_result_transfer.final_output_column_count, 4);
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(1, 2, 3, 4)],
        "accepted epistemic 4-cycle final output must materialize the production WCOJ row"
    );
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
    assert_eq!(result.prepared.preflight.kclique_stream_group_count, 1);
    assert_eq!(
        result.prepared.preflight.kclique_skew_scheduled_plan_count, 1,
        "accepted K5 must carry G38-B buried-skew helper scheduling"
    );
    assert_eq!(result.prepared.preflight.sorted_layout_requirement_count, 1);
    assert_eq!(result.prepared.preflight.helper_split_spec_count, 1);
    assert_eq!(result.prepared.preflight.helper_relation_rule_count, 1);
    assert_eq!(result.prepared.preflight.helper_relation_scan_count, 1);
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert!(matches!(
        result.trace.wcoj_certification,
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches: 1..,
            observed_kclique_dispatches: 1..,
            certified_edge_permutation_slots: 10,
            certified_stream_groups: 1,
            certified_skew_scheduled_plans: 1,
            certified_sorted_layout_requirements: 1,
            certified_helper_split_specs: 1,
            certified_helper_relation_rules: 1,
            certified_helper_relation_scans: 1,
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
fn accepted_epistemic_k6_execution_certifies_g38b_helper_histogram_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let source = epistemic_kclique_source(6, true);
    let program = parse_program(&source).expect("parse epistemic K6");
    let rel_ids = rel_ids_for_reduced_kclique(6);
    let stats = kclique_stats(&rel_ids, 6, Some((4, 5.0)));
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, Some(&stats))
        .expect("compile epistemic executable K6");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in k_clique_inputs(6) {
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
        .expect("execute accepted epistemic K6");

    assert_eq!(result.prepared.preflight.kclique_wcoj_plan_count, 1);
    assert_eq!(result.prepared.preflight.kclique_wcoj_max_arity, 6);
    assert_eq!(
        result
            .prepared
            .preflight
            .kclique_wcoj_edge_permutation_count,
        15
    );
    assert_eq!(result.prepared.preflight.kclique_stream_group_count, 1);
    assert_eq!(
        result.prepared.preflight.kclique_skew_scheduled_plan_count, 1,
        "accepted K6 must carry G38-B buried-skew helper scheduling"
    );
    assert!(
        result.prepared.preflight.sorted_layout_requirement_count >= 1,
        "accepted K6 must carry production sorted-layout requirements"
    );
    assert!(
        result.prepared.preflight.helper_split_spec_count >= 1,
        "accepted K6 must carry G38-B helper-split specs for buried skew"
    );
    assert!(
        result.prepared.preflight.helper_relation_rule_count >= 1,
        "accepted K6 must include production helper relation rules"
    );
    assert!(
        result.prepared.preflight.helper_relation_scan_count >= 1,
        "accepted K6 WCOJ plan must scan production helper relation output"
    );
    assert!(matches!(
        result.trace.wcoj_certification,
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches: 1..,
            observed_kclique_dispatches: 1..,
            certified_edge_permutation_slots: 15,
            certified_stream_groups: 1..,
            certified_skew_scheduled_plans: 1..,
            certified_sorted_layout_requirements: 1..,
            certified_helper_split_specs: 1..,
            certified_helper_relation_rules: 1..,
            certified_helper_relation_scans: 1..,
            observed_metadata_builds: 1..,
            observed_metadata_build_nanos: 1..,
            ..
        }
    ));
    assert!(
        result.trace.counter_delta.wcoj_clique6_dispatch_count >= 1,
        "accepted epistemic K6 must dispatch through production K6 WCOJ"
    );
    assert!(
        result.trace.counter_delta.kclique_metadata_build_count >= 1,
        "accepted epistemic K6 must build production K-clique histogram metadata"
    );
    assert!(
        result.trace.counter_delta.kclique_metadata_build_nanos >= 1,
        "accepted epistemic K6 must time production K-clique histogram metadata builds"
    );
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(result.final_result_transfer.final_output_column_count, 6);
    assert_eq!(
        result.final_result_transfer.final_output_payload_bytes,
        6 * std::mem::size_of::<u32>() as u64
    );
    assert_eq!(
        read_device_row_count(&fix.provider, &result.final_output).expect("final row count"),
        1,
        "accepted epistemic K6 final output must materialize the production WCOJ row"
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
        assert_eq!(preflight.kclique_stream_group_count, 1);
        assert_eq!(preflight.planned_hash_route_count, 0);
        assert!(
            preflight.sorted_layout_requirement_count >= 1,
            "K{k} must carry production sorted-layout requirements"
        );
        if preflight.helper_split_spec_count > 0 {
            assert!(
                preflight.helper_relation_rule_count >= preflight.helper_split_spec_count,
                "K{k} helper-split specs must be backed by production helper relation rules"
            );
            assert!(
                preflight.helper_relation_scan_count >= preflight.helper_split_spec_count,
                "K{k} helper-split specs must be consumed by production helper relation scans"
            );
        }
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
    assert_eq!(result.prepared.preflight.kclique_stream_group_count, 1);
    assert!(
        result.prepared.preflight.sorted_layout_requirement_count >= 1,
        "accepted K7 must carry production sorted-layout requirements"
    );
    assert!(matches!(
        result.trace.wcoj_certification,
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches: 1..,
            observed_kclique_dispatches: 1..,
            certified_stream_groups: 1..,
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
    assert_eq!(result.prepared.preflight.kclique_stream_group_count, 1);
    assert!(
        result.prepared.preflight.sorted_layout_requirement_count >= 1,
        "accepted K8 must carry production sorted-layout requirements"
    );
    assert!(matches!(
        result.trace.wcoj_certification,
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches: 1..,
            observed_kclique_dispatches: 1..,
            certified_stream_groups: 1..,
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
fn accepted_split_components_execute_gpu_runtime_and_match_component_oracles() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let edge_oracle_program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred a(u32).
        a(X) :- node(X), know edge(X).
        "#,
    )
    .expect("parse split edge oracle");
    let color_oracle_program = parse_program(
        r#"
        pred node(u32).
        pred color(u32).
        pred b(u32).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split color oracle");
    let split_oracles = vec![
        run_generate_propagate_test(
            &edge_oracle_program,
            vec![
                EpistemicInterpretation::new(),
                EpistemicInterpretation::new().with_known("edge", 1),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split edge oracle"),
        run_generate_propagate_test(
            &color_oracle_program,
            vec![
                EpistemicInterpretation::new(),
                EpistemicInterpretation::new().with_known("color", 1),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split color oracle"),
    ];
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    assert_eq!(split.components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 2);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert_eq!(
        batch.trace.accepted_world_views,
        split_oracles
            .iter()
            .map(|oracle| oracle.trace.accepted_world_views)
            .sum::<usize>()
    );
    assert_eq!(
        batch.trace.rejected_candidates,
        split_oracles
            .iter()
            .map(|oracle| oracle.trace.rejected)
            .sum::<usize>()
    );

    let results = &batch.results;
    assert_eq!(results.len(), 2);
    assert_eq!(
        download_unary_u32(&fix.provider, &results[0].final_output),
        vec![1, 3]
    );
    assert_eq!(
        download_unary_u32(&fix.provider, &results[1].final_output),
        vec![2]
    );
    for (result, oracle) in results.iter().zip(split_oracles.iter()) {
        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            result.semantic_trace.generated_candidates,
            oracle.trace.generated
        );
        assert_eq!(
            result.semantic_trace.propagated_candidates,
            oracle.trace.propagated
        );
        assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
        assert_eq!(
            result.semantic_trace.accepted_candidates,
            oracle.trace.accepted
        );
        assert_eq!(
            result.semantic_trace.accepted_world_views,
            oracle.trace.accepted_world_views
        );
        assert_eq!(
            result.semantic_trace.rejected_candidates,
            oracle.trace.rejected
        );
        assert_eq!(
            result.semantic_trace.accepted_candidate_indices,
            oracle.accepted_candidate_indices
        );
        assert_eq!(
            result.semantic_trace.rejected_candidate_indices,
            oracle.rejected_candidate_indices
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    }
}

#[test]
fn accepted_split_binary_operator_components_match_gpt_oracles() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred blocked(u32, u32).
        pred possible_edge(u32, u32).
        pred clear_pair(u32, u32).
        possible_edge(X, Y) :- pair(X, Y), possible edge(X, Y).
        clear_pair(X, Y) :- pair(X, Y), not possible blocked(X, Y).
        "#,
    )
    .expect("parse split binary operator fixture");
    let possible_oracle_program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred possible_edge(u32, u32).
        possible_edge(X, Y) :- pair(X, Y), possible edge(X, Y).
        "#,
    )
    .expect("parse split binary possible oracle");
    let not_possible_oracle_program = parse_program(
        r#"
        pred pair(u32, u32).
        pred blocked(u32, u32).
        pred clear_pair(u32, u32).
        clear_pair(X, Y) :- pair(X, Y), not possible blocked(X, Y).
        "#,
    )
    .expect("parse split binary not-possible oracle");
    let split_oracles = vec![
        run_generate_propagate_test(
            &possible_oracle_program,
            vec![
                EpistemicInterpretation::new(),
                EpistemicInterpretation::new().with_known("edge", 2),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split binary possible oracle"),
        run_generate_propagate_test(
            &not_possible_oracle_program,
            vec![
                EpistemicInterpretation::new().with_known("blocked", 2),
                EpistemicInterpretation::new(),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split binary not-possible oracle"),
    ];
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split binary operator components");

    assert_eq!(split.components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);
    let possible_component_idx = split
        .components
        .iter()
        .position(|component| component.component.rule_indices == vec![0])
        .expect("possible component must retain source rule index 0");
    let not_possible_component_idx = split
        .components
        .iter()
        .position(|component| component.component.rule_indices == vec![1])
        .expect("not-possible component must retain source rule index 1");
    assert_ne!(possible_component_idx, not_possible_component_idx);

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split binary operator components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "pair",
        upload_binary_u32(&fix.memory, &[(1, 2), (2, 3), (3, 4)]),
    );
    executor.put_relation("edge", upload_binary_u32(&fix.memory, &[(1, 2), (3, 4)]));
    executor.put_relation("blocked", upload_binary_u32(&fix.memory, &[(3, 4)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split binary operator components through GPU batch path");

    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 2);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert_eq!(batch.trace.know_operator_count, 0);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 0);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(
        batch.trace.accepted_world_views,
        split_oracles
            .iter()
            .map(|oracle| oracle.trace.accepted_world_views)
            .sum::<usize>()
    );
    assert_eq!(
        batch.trace.rejected_candidates,
        split_oracles
            .iter()
            .map(|oracle| oracle.trace.rejected)
            .sum::<usize>()
    );

    let results = &batch.results;
    assert_eq!(results.len(), 2);
    assert_eq!(
        download_binary_u32(&fix.provider, &results[possible_component_idx].final_output),
        vec![(1, 2), (3, 4)]
    );
    assert_eq!(
        download_binary_u32(
            &fix.provider,
            &results[not_possible_component_idx].final_output
        ),
        vec![(1, 2), (2, 3)]
    );
    assert_eq!(
        results[possible_component_idx]
            .final_tuple_materialization
            .row_filter_count,
        1
    );
    assert_eq!(
        results[possible_component_idx]
            .final_tuple_materialization
            .negated_row_filter_count,
        0
    );
    assert_eq!(
        results[not_possible_component_idx]
            .final_tuple_materialization
            .row_filter_count,
        1
    );
    assert_eq!(
        results[not_possible_component_idx]
            .final_tuple_materialization
            .negated_row_filter_count,
        1
    );
    for (component, result) in split.components.iter().zip(results.iter()) {
        let oracle = match component.component.rule_indices.as_slice() {
            [0] => &split_oracles[0],
            [1] => &split_oracles[1],
            other => panic!("unexpected split binary operator rule indices: {other:?}"),
        };
        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            result.semantic_trace.generated_candidates,
            oracle.trace.generated
        );
        assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
        assert_eq!(
            result.semantic_trace.propagated_candidates,
            oracle.trace.propagated
        );
        assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
        assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
        assert_eq!(
            result.semantic_trace.accepted_candidates,
            oracle.trace.accepted
        );
        assert_eq!(
            result.semantic_trace.accepted_world_views,
            oracle.trace.accepted_world_views
        );
        assert_eq!(
            result.semantic_trace.rejected_candidates,
            oracle.trace.rejected
        );
        assert_eq!(
            result.semantic_trace.accepted_candidate_indices,
            oracle.accepted_candidate_indices
        );
        assert_eq!(
            result.semantic_trace.rejected_candidate_indices,
            oracle.rejected_candidate_indices
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    }
}

#[test]
fn accepted_split_all_binary_operators_match_gpt_oracles() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred alt(u32, u32).
        pred blocked(u32, u32).
        pred seen(u32, u32).
        pred known_edge(u32, u32).
        pred possible_alt(u32, u32).
        pred clear_pair(u32, u32).
        pred unknown_pair(u32, u32).
        known_edge(X, Y) :- pair(X, Y), know edge(X, Y).
        possible_alt(X, Y) :- pair(X, Y), possible alt(X, Y).
        clear_pair(X, Y) :- pair(X, Y), not possible blocked(X, Y).
        unknown_pair(X, Y) :- pair(X, Y), not know seen(X, Y).
        "#,
    )
    .expect("parse split all binary operators fixture");
    let know_oracle_program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred known_edge(u32, u32).
        known_edge(X, Y) :- pair(X, Y), know edge(X, Y).
        "#,
    )
    .expect("parse split binary know oracle");
    let possible_oracle_program = parse_program(
        r#"
        pred pair(u32, u32).
        pred alt(u32, u32).
        pred possible_alt(u32, u32).
        possible_alt(X, Y) :- pair(X, Y), possible alt(X, Y).
        "#,
    )
    .expect("parse split binary possible oracle");
    let not_possible_oracle_program = parse_program(
        r#"
        pred pair(u32, u32).
        pred blocked(u32, u32).
        pred clear_pair(u32, u32).
        clear_pair(X, Y) :- pair(X, Y), not possible blocked(X, Y).
        "#,
    )
    .expect("parse split binary not-possible oracle");
    let not_know_oracle_program = parse_program(
        r#"
        pred pair(u32, u32).
        pred seen(u32, u32).
        pred unknown_pair(u32, u32).
        unknown_pair(X, Y) :- pair(X, Y), not know seen(X, Y).
        "#,
    )
    .expect("parse split binary not-know oracle");
    let split_oracles = vec![
        run_generate_propagate_test(
            &know_oracle_program,
            vec![
                EpistemicInterpretation::new(),
                EpistemicInterpretation::new().with_known("edge", 2),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split binary know oracle"),
        run_generate_propagate_test(
            &possible_oracle_program,
            vec![
                EpistemicInterpretation::new(),
                EpistemicInterpretation::new().with_known("alt", 2),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split binary possible oracle"),
        run_generate_propagate_test(
            &not_possible_oracle_program,
            vec![
                EpistemicInterpretation::new().with_known("blocked", 2),
                EpistemicInterpretation::new(),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split binary not-possible oracle"),
        run_generate_propagate_test(
            &not_know_oracle_program,
            vec![
                EpistemicInterpretation::new().with_known("seen", 2),
                EpistemicInterpretation::new(),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split binary not-know oracle"),
    ];
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split all binary operator components");

    assert_eq!(split.components.len(), 4);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1, 2, 3]);
    for rule_idx in 0..4 {
        assert!(
            split
                .components
                .iter()
                .any(|component| component.component.rule_indices == vec![rule_idx]),
            "split all-operator fixture must retain source rule index {rule_idx}"
        );
    }

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split all-operator components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "pair",
        upload_binary_u32(&fix.memory, &[(1, 2), (2, 3), (3, 4)]),
    );
    executor.put_relation("edge", upload_binary_u32(&fix.memory, &[(1, 2), (3, 4)]));
    executor.put_relation("alt", upload_binary_u32(&fix.memory, &[(2, 3)]));
    executor.put_relation("blocked", upload_binary_u32(&fix.memory, &[(3, 4)]));
    executor.put_relation("seen", upload_binary_u32(&fix.memory, &[(1, 2)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split all binary operator components through GPU batch path");

    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 4);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());
    assert_eq!(batch.trace.aggregate_kernel_timing.cuda_event_pairs, 32);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(
        batch.trace.accepted_world_views,
        split_oracles
            .iter()
            .map(|oracle| oracle.trace.accepted_world_views)
            .sum::<usize>()
    );
    assert_eq!(
        batch.trace.rejected_candidates,
        split_oracles
            .iter()
            .map(|oracle| oracle.trace.rejected)
            .sum::<usize>()
    );

    let results = &batch.results;
    assert_eq!(results.len(), 4);
    for (component, result) in split.components.iter().zip(results.iter()) {
        let rule_idx = *component
            .component
            .rule_indices
            .first()
            .expect("split component must carry one source rule index");
        let oracle = &split_oracles[rule_idx];
        let (expected_rows, expected_negated_filters) = match rule_idx {
            0 => (vec![(1, 2), (3, 4)], 0),
            1 => (vec![(2, 3)], 0),
            2 => (vec![(1, 2), (2, 3)], 1),
            3 => (vec![(2, 3), (3, 4)], 1),
            other => panic!("unexpected split all-operator rule index: {other}"),
        };

        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            download_binary_u32(&fix.provider, &result.final_output),
            expected_rows
        );
        assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
        assert_eq!(
            result.final_tuple_materialization.negated_row_filter_count,
            expected_negated_filters
        );
        assert_eq!(
            result.semantic_trace.generated_candidates,
            oracle.trace.generated
        );
        assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
        assert_eq!(
            result.semantic_trace.propagated_candidates,
            oracle.trace.propagated
        );
        assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
        assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
        assert_eq!(
            result.semantic_trace.accepted_candidates,
            oracle.trace.accepted
        );
        assert_eq!(
            result.semantic_trace.accepted_world_views,
            oracle.trace.accepted_world_views
        );
        assert_eq!(
            result.semantic_trace.rejected_candidates,
            oracle.trace.rejected
        );
        assert_eq!(
            result.semantic_trace.accepted_candidate_indices,
            oracle.accepted_candidate_indices
        );
        assert_eq!(
            result.semantic_trace.rejected_candidate_indices,
            oracle.rejected_candidate_indices
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    }
}

#[test]
fn accepted_split_quaternary_all_operators_match_gpt_oracles() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred edge4(u32, u32, u32, u32).
        pred alt4(u32, u32, u32, u32).
        pred blocked_fact4(u32, u32, u32, u32).
        pred hidden_fact4(u32, u32, u32, u32).
        pred known4(u32, u32, u32, u32).
        pred possible4(u32, u32, u32, u32).
        pred clear4(u32, u32, u32, u32).
        pred unknown4(u32, u32, u32, u32).
        known4(A, B, C, D) :- tuple4(A, B, C, D), know edge4(A, B, C, D).
        possible4(A, B, C, D) :- tuple4(A, B, C, D), possible alt4(A, B, C, D).
        clear4(A, B, C, D) :- tuple4(A, B, C, D), not possible blocked_fact4(A, B, C, D).
        unknown4(A, B, C, D) :- tuple4(A, B, C, D), not know hidden_fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary all-operator fixture");
    let know_oracle_program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred edge4(u32, u32, u32, u32).
        pred known4(u32, u32, u32, u32).
        known4(A, B, C, D) :- tuple4(A, B, C, D), know edge4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary know oracle");
    let possible_oracle_program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred alt4(u32, u32, u32, u32).
        pred possible4(u32, u32, u32, u32).
        possible4(A, B, C, D) :- tuple4(A, B, C, D), possible alt4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary possible oracle");
    let not_possible_oracle_program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred blocked_fact4(u32, u32, u32, u32).
        pred clear4(u32, u32, u32, u32).
        clear4(A, B, C, D) :- tuple4(A, B, C, D), not possible blocked_fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary not-possible oracle");
    let not_know_oracle_program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred hidden_fact4(u32, u32, u32, u32).
        pred unknown4(u32, u32, u32, u32).
        unknown4(A, B, C, D) :- tuple4(A, B, C, D), not know hidden_fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary not-know oracle");
    let split_oracles = vec![
        run_generate_propagate_test(
            &know_oracle_program,
            vec![
                EpistemicInterpretation::new(),
                EpistemicInterpretation::new().with_known("edge4", 4),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split quaternary know oracle"),
        run_generate_propagate_test(
            &possible_oracle_program,
            vec![
                EpistemicInterpretation::new(),
                EpistemicInterpretation::new().with_known("alt4", 4),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split quaternary possible oracle"),
        run_generate_propagate_test(
            &not_possible_oracle_program,
            vec![
                EpistemicInterpretation::new().with_known("blocked_fact4", 4),
                EpistemicInterpretation::new(),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split quaternary not-possible oracle"),
        run_generate_propagate_test(
            &not_know_oracle_program,
            vec![
                EpistemicInterpretation::new().with_known("hidden_fact4", 4),
                EpistemicInterpretation::new(),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split quaternary not-know oracle"),
    ];
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split quaternary all-operator components");

    assert_eq!(split.components.len(), 4);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1, 2, 3]);
    for rule_idx in 0..4 {
        assert!(
            split
                .components
                .iter()
                .any(|component| component.component.rule_indices == vec![rule_idx]),
            "split quaternary all-operator fixture must retain source rule index {rule_idx}"
        );
    }

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split quaternary all-operator components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(
            &fix.memory,
            &[(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6), (9, 9, 9, 9)],
        ),
    );
    executor.put_relation("edge4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));
    executor.put_relation("alt4", upload_quaternary_u32(&fix.memory, &[(3, 4, 5, 6)]));
    executor.put_relation(
        "blocked_fact4",
        upload_quaternary_u32(&fix.memory, &[(9, 9, 9, 9)]),
    );
    executor.put_relation(
        "hidden_fact4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4)]),
    );

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split quaternary all-operator components through GPU batch path");

    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 4);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());
    assert_eq!(batch.trace.aggregate_kernel_timing.cuda_event_pairs, 32);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(
        batch.trace.accepted_world_views,
        split_oracles
            .iter()
            .map(|oracle| oracle.trace.accepted_world_views)
            .sum::<usize>()
    );
    assert_eq!(
        batch.trace.rejected_candidates,
        split_oracles
            .iter()
            .map(|oracle| oracle.trace.rejected)
            .sum::<usize>()
    );

    let results = &batch.results;
    assert_eq!(results.len(), 4);
    for (component, result) in split.components.iter().zip(results.iter()) {
        let rule_idx = *component
            .component
            .rule_indices
            .first()
            .expect("split component must carry one source rule index");
        let oracle = &split_oracles[rule_idx];
        let (expected_rows, expected_negated_filters) = match rule_idx {
            0 => (vec![(2, 3, 4, 5)], 0),
            1 => (vec![(3, 4, 5, 6)], 0),
            2 => (vec![(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)], 1),
            3 => (vec![(2, 3, 4, 5), (3, 4, 5, 6), (9, 9, 9, 9)], 1),
            other => panic!("unexpected split quaternary all-operator rule index: {other}"),
        };

        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            4
        );
        assert_eq!(
            download_quaternary_u32(&fix.provider, &result.final_output),
            expected_rows
        );
        assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
        assert_eq!(
            result.final_tuple_materialization.negated_row_filter_count,
            expected_negated_filters
        );
        assert_eq!(
            result.semantic_trace.generated_candidates,
            oracle.trace.generated
        );
        assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
        assert_eq!(
            result.semantic_trace.propagated_candidates,
            oracle.trace.propagated
        );
        assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
        assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
        assert_eq!(
            result.semantic_trace.accepted_candidates,
            oracle.trace.accepted
        );
        assert_eq!(
            result.semantic_trace.accepted_world_views,
            oracle.trace.accepted_world_views
        );
        assert_eq!(
            result.semantic_trace.rejected_candidates,
            oracle.trace.rejected
        );
        assert_eq!(
            result.semantic_trace.accepted_candidate_indices,
            oracle.accepted_candidate_indices
        );
        assert_eq!(
            result.semantic_trace.rejected_candidate_indices,
            oracle.rejected_candidate_indices
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    }
}

#[test]
fn accepted_split_quaternary_not_possible_batch_matches_gpt_oracles() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred known4(u32, u32, u32, u32).
        pred clear4(u32, u32, u32, u32).
        known4(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        clear4(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary GPT oracle fixture");
    let know_oracle_program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred known4(u32, u32, u32, u32).
        known4(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary know oracle");
    let not_possible_oracle_program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred clear4(u32, u32, u32, u32).
        clear4(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary not-possible oracle");
    let split_oracles = vec![
        run_generate_propagate_test(
            &know_oracle_program,
            vec![
                EpistemicInterpretation::new(),
                EpistemicInterpretation::new().with_known("fact4", 4),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split quaternary know oracle"),
        run_generate_propagate_test(
            &not_possible_oracle_program,
            vec![
                EpistemicInterpretation::new().with_known("fact4", 4),
                EpistemicInterpretation::new(),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split quaternary not-possible oracle"),
    ];
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split quaternary GPT oracle components");

    assert_eq!(split.components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);
    for rule_idx in 0..2 {
        assert!(
            split
                .components
                .iter()
                .any(|component| component.component.rule_indices == vec![rule_idx]),
            "split quaternary fixture must retain source rule index {rule_idx}"
        );
    }

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split quaternary components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split quaternary components through GPU batch path");

    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 2);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());
    assert_eq!(batch.trace.aggregate_kernel_timing.cuda_event_pairs, 16);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 0);
    assert_eq!(batch.trace.not_know_operator_count, 0);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(
        batch.trace.accepted_world_views,
        split_oracles
            .iter()
            .map(|oracle| oracle.trace.accepted_world_views)
            .sum::<usize>()
    );
    assert_eq!(
        batch.trace.rejected_candidates,
        split_oracles
            .iter()
            .map(|oracle| oracle.trace.rejected)
            .sum::<usize>()
    );

    let results = &batch.results;
    assert_eq!(results.len(), 2);
    for (component, result) in split.components.iter().zip(results.iter()) {
        let rule_idx = *component
            .component
            .rule_indices
            .first()
            .expect("split component must carry one source rule index");
        let oracle = &split_oracles[rule_idx];
        let (expected_rows, expected_negated_filters) = match rule_idx {
            0 => (vec![(2, 3, 4, 5)], 0),
            1 => (vec![(1, 2, 3, 4), (9, 9, 9, 9)], 1),
            other => panic!("unexpected split quaternary rule index: {other}"),
        };

        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            4
        );
        assert_eq!(
            download_quaternary_u32(&fix.provider, &result.final_output),
            expected_rows
        );
        assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
        assert_eq!(
            result.final_tuple_materialization.negated_row_filter_count,
            expected_negated_filters
        );
        assert_eq!(
            result.semantic_trace.generated_candidates,
            oracle.trace.generated
        );
        assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
        assert_eq!(
            result.semantic_trace.propagated_candidates,
            oracle.trace.propagated
        );
        assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
        assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
        assert_eq!(
            result.semantic_trace.accepted_candidates,
            oracle.trace.accepted
        );
        assert_eq!(
            result.semantic_trace.accepted_world_views,
            oracle.trace.accepted_world_views
        );
        assert_eq!(
            result.semantic_trace.rejected_candidates,
            oracle.trace.rejected
        );
        assert_eq!(
            result.semantic_trace.accepted_candidate_indices,
            oracle.accepted_candidate_indices
        );
        assert_eq!(
            result.semantic_trace.rejected_candidate_indices,
            oracle.rejected_candidate_indices
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    }
}

#[test]
fn accepted_split_quaternary_possible_and_not_know_batch_matches_gpt_oracles() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred possible4(u32, u32, u32, u32).
        pred unknown4(u32, u32, u32, u32).
        possible4(A, B, C, D) :- tuple4(A, B, C, D), possible fact4(A, B, C, D).
        unknown4(A, B, C, D) :- tuple4(A, B, C, D), not know fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary possible/not-know GPT oracle fixture");
    let possible_oracle_program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred possible4(u32, u32, u32, u32).
        possible4(A, B, C, D) :- tuple4(A, B, C, D), possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary possible oracle");
    let not_know_oracle_program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred unknown4(u32, u32, u32, u32).
        unknown4(A, B, C, D) :- tuple4(A, B, C, D), not know fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary not-know oracle");
    let split_oracles = vec![
        run_generate_propagate_test(
            &possible_oracle_program,
            vec![
                EpistemicInterpretation::new(),
                EpistemicInterpretation::new().with_known("fact4", 4),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split quaternary possible oracle"),
        run_generate_propagate_test(
            &not_know_oracle_program,
            vec![
                EpistemicInterpretation::new().with_known("fact4", 4),
                EpistemicInterpretation::new(),
            ],
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect("run split quaternary not-know oracle"),
    ];
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split quaternary possible/not-know GPT oracle components");

    assert_eq!(split.components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);
    for rule_idx in 0..2 {
        assert!(
            split
                .components
                .iter()
                .any(|component| component.component.rule_indices == vec![rule_idx]),
            "split quaternary possible/not-know fixture must retain source rule index {rule_idx}"
        );
    }

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split quaternary possible/not-know components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split quaternary possible/not-know components through GPU batch path");

    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 2);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());
    assert_eq!(batch.trace.aggregate_kernel_timing.cuda_event_pairs, 16);
    assert_eq!(batch.trace.know_operator_count, 0);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 0);
    assert_eq!(
        batch.trace.accepted_world_views,
        split_oracles
            .iter()
            .map(|oracle| oracle.trace.accepted_world_views)
            .sum::<usize>()
    );
    assert_eq!(
        batch.trace.rejected_candidates,
        split_oracles
            .iter()
            .map(|oracle| oracle.trace.rejected)
            .sum::<usize>()
    );

    let results = &batch.results;
    assert_eq!(results.len(), 2);
    for (component, result) in split.components.iter().zip(results.iter()) {
        let rule_idx = *component
            .component
            .rule_indices
            .first()
            .expect("split component must carry one source rule index");
        let oracle = &split_oracles[rule_idx];
        let (expected_rows, expected_negated_filters) = match rule_idx {
            0 => (vec![(2, 3, 4, 5)], 0),
            1 => (vec![(1, 2, 3, 4), (9, 9, 9, 9)], 1),
            other => panic!("unexpected split quaternary possible/not-know rule index: {other}"),
        };

        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            4
        );
        assert_eq!(
            download_quaternary_u32(&fix.provider, &result.final_output),
            expected_rows
        );
        assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
        assert_eq!(
            result.final_tuple_materialization.negated_row_filter_count,
            expected_negated_filters
        );
        assert_eq!(
            result.semantic_trace.generated_candidates,
            oracle.trace.generated
        );
        assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
        assert_eq!(
            result.semantic_trace.propagated_candidates,
            oracle.trace.propagated
        );
        assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
        assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
        assert_eq!(
            result.semantic_trace.accepted_candidates,
            oracle.trace.accepted
        );
        assert_eq!(
            result.semantic_trace.accepted_world_views,
            oracle.trace.accepted_world_views
        );
        assert_eq!(
            result.semantic_trace.rejected_candidates,
            oracle.trace.rejected
        );
        assert_eq!(
            result.semantic_trace.accepted_candidate_indices,
            oracle.accepted_candidate_indices
        );
        assert_eq!(
            result.semantic_trace.rejected_candidate_indices,
            oracle.rejected_candidate_indices
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    }
}

#[test]
fn accepted_split_batch_solver_gate_rejects_unrecorded_aggregate_kernel_timing() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (_split, mut batch) = execute_split_all_binary_operator_batch(&fix);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());
    batch.trace.aggregate_kernel_timing = Default::default();

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(sat_cnf.var_cap, sat_cnf.clause_cap)
        .expect("new workspace");

    let err = match adapter.solve_assumption_lifecycle_with_gpu_batch_execution_result(
        &fix.provider,
        GpuSolverProductionBatchExecutionEvidence { batch: &batch },
        &mut workspace,
        &[GpuSolverProductionLifecycleStep {
            cnf: &sat_cnf,
            branch_var_limit: &branch_limit,
            expectation: GpuSolverProductionExpectation::Sat,
        }],
    ) {
        Ok(_) => panic!("solver batch evidence without aggregate timing must reject"),
        Err(err) => err,
    };
    assert!(format!("{err}").contains("aggregate CUDA-event timing"));
}

#[test]
fn accepted_single_solver_gate_rejects_unrecorded_candidate_generation_timing() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let mut result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
        &[1, 2],
        &[1],
    );
    assert!(result.candidate_generation.kernel_timing.is_recorded());
    result.candidate_generation.kernel_timing = Default::default();

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(sat_cnf.var_cap, sat_cnf.clause_cap)
        .expect("new workspace");

    let err = match adapter.solve_assumption_lifecycle_with_gpu_execution_result(
        &fix.provider,
        &result,
        &mut workspace,
        &[GpuSolverProductionLifecycleStep {
            cnf: &sat_cnf,
            branch_var_limit: &branch_limit,
            expectation: GpuSolverProductionExpectation::Sat,
        }],
    ) {
        Ok(_) => panic!("solver evidence without candidate-generation timing must reject"),
        Err(err) => err,
    };
    assert!(format!("{err}").contains("CUDA-event timing"));
}

#[test]
fn accepted_single_prob_gate_rejects_unrecorded_candidate_generation_timing() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let mut result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
        &[1, 2],
        &[1],
    );
    assert!(result.candidate_generation.kernel_timing.is_recorded());
    result.candidate_generation.kernel_timing = Default::default();

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let err = match adapter.compile_and_evaluate_conditioned_source_with_gpu_execution_result(
        r#"
            0.7::edge(1).
            query(edge(1)).
            "#,
        &fix.provider,
        &result,
        vec![EpistemicAssumption::known_tuple(
            "edge",
            vec![EpistemicEvidenceTerm::integer(1)],
            true,
        )],
    ) {
        Ok(_) => panic!("probabilistic evidence without candidate-generation timing must reject"),
        Err(err) => err,
    };
    assert!(format!("{err}").contains("CUDA-event timing"));
}

#[test]
fn accepted_split_batch_prob_gate_rejects_unrecorded_aggregate_kernel_timing() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (split, mut batch) = execute_split_all_binary_operator_batch(&fix);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());
    batch.trace.aggregate_kernel_timing = Default::default();

    let assumption_groups_owned: Vec<Vec<EpistemicAssumption>> = split
        .components
        .iter()
        .map(
            |component| match component.component.rule_indices.as_slice() {
                [0] => vec![EpistemicAssumption::known_tuple(
                    "edge",
                    vec![
                        EpistemicEvidenceTerm::integer(1),
                        EpistemicEvidenceTerm::integer(2),
                    ],
                    true,
                )],
                [1] => vec![EpistemicAssumption::possible_tuple(
                    "alt",
                    vec![
                        EpistemicEvidenceTerm::integer(2),
                        EpistemicEvidenceTerm::integer(3),
                    ],
                    true,
                )],
                [2] => vec![EpistemicAssumption::possible_tuple(
                    "blocked",
                    vec![
                        EpistemicEvidenceTerm::integer(1),
                        EpistemicEvidenceTerm::integer(2),
                    ],
                    false,
                )],
                [3] => vec![EpistemicAssumption::known_tuple(
                    "seen",
                    vec![
                        EpistemicEvidenceTerm::integer(2),
                        EpistemicEvidenceTerm::integer(3),
                    ],
                    false,
                )],
                other => panic!("unexpected split all-operator rule indices: {other:?}"),
            },
        )
        .collect();
    let assumption_groups: Vec<&[EpistemicAssumption]> =
        assumption_groups_owned.iter().map(Vec::as_slice).collect();

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let err = match adapter.compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result(
        r#"
            0.4::edge(1, 2).
            0.5::alt(2, 3).
            0.7::blocked(1, 2).
            0.8::seen(2, 3).
            query(edge(1, 2)).
            query(alt(2, 3)).
            query(blocked(1, 2)).
            query(seen(2, 3)).
            "#,
        &fix.provider,
        EpistemicProbGpuBatchExecutionEvidence {
            batch: &batch,
            assumptions_by_component: &assumption_groups,
        },
    ) {
        Ok(_) => panic!("probabilistic batch evidence without aggregate timing must reject"),
        Err(err) => err,
    };
    assert!(format!("{err}").contains("aggregate CUDA-event timing"));
}

#[test]
fn aggregate_timing_requires_every_component_phase_to_be_recorded() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (_split, mut batch) = execute_split_all_binary_operator_batch(&fix);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());
    assert!(batch.results[0]
        .candidate_generation
        .kernel_timing
        .is_recorded());

    batch.results[0].candidate_generation.kernel_timing = Default::default();
    batch.trace = EpistemicGpuBatchExecutionTrace::from_component_results(&batch.results);

    assert!(
        !batch.results[0].aggregate_kernel_timing().is_recorded(),
        "single-result aggregate timing must fail closed when any hot-path phase lacks CUDA events"
    );
    assert!(
        !batch.trace.aggregate_kernel_timing.is_recorded(),
        "batch aggregate timing must fail closed when any component phase lacks CUDA events"
    );
}

#[test]
fn accepted_split_quaternary_all_operator_batch_records_component_kernel_timing() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (split, batch) = execute_split_quaternary_all_operator_batch(&fix);

    assert_eq!(split.components.len(), 4);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1, 2, 3]);
    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 4);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());
    assert_eq!(batch.trace.aggregate_kernel_timing.cuda_event_pairs, 32);
    assert_eq!(batch.trace.aggregate_kernel_timing.timing_sync_ops, 32);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);

    let assert_phase =
        |phase: &str,
         kernel_launches: u32,
         host_write_ops: u32,
         timing: xlog_runtime::EpistemicGpuKernelTimingTrace| {
            assert!(kernel_launches > 0, "{phase} must launch GPU kernels");
            assert_eq!(host_write_ops, 0, "{phase} must avoid host writes");
            assert!(
                timing.is_recorded(),
                "{phase} must record CUDA-event timing"
            );
            assert_eq!(timing.cuda_event_pairs, 1, "{phase} event pair count");
            assert_eq!(timing.timing_sync_ops, 1, "{phase} timing sync count");
        };

    for (idx, component) in split.components.iter().enumerate() {
        let result = &batch.results[idx];
        let (expected_rows, expected_negated_filters) =
            match component.component.rule_indices.as_slice() {
                [0] => (vec![(2, 3, 4, 5)], 0),
                [1] => (vec![(3, 4, 5, 6)], 0),
                [2] => (vec![(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)], 1),
                [3] => (vec![(2, 3, 4, 5), (3, 4, 5, 6), (9, 9, 9, 9)], 1),
                other => {
                    panic!("unexpected split quaternary timing rule indices: {other:?}")
                }
            };

        assert_eq!(
            download_quaternary_u32(&fix.provider, &result.final_output),
            expected_rows
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            result.model_membership.tuple_source_row_count_device_reads,
            1
        );
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            4
        );
        assert!(result.model_membership.model_membership_bytes_written > 0);
        assert!(result.world_view_validation.model_membership_bytes_checked > 0);
        assert!(
            result
                .final_tuple_materialization
                .model_membership_bytes_checked
                > 0
        );
        assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
        assert_eq!(
            result.final_tuple_materialization.negated_row_filter_count,
            expected_negated_filters
        );

        assert_phase(
            "candidate generation",
            result.candidate_generation.kernel_launches,
            result.candidate_generation.host_write_ops,
            result.candidate_generation.kernel_timing,
        );
        assert_phase(
            "candidate propagation",
            result.propagation.kernel_launches,
            result.propagation.host_write_ops,
            result.propagation.kernel_timing,
        );
        assert_phase(
            "candidate validation",
            result.candidate_validation.kernel_launches,
            result.candidate_validation.host_write_ops,
            result.candidate_validation.kernel_timing,
        );
        assert_phase(
            "model membership",
            result.model_membership.kernel_launches,
            result.model_membership.host_write_ops,
            result.model_membership.kernel_timing,
        );
        assert_phase(
            "world-view validation",
            result.world_view_validation.kernel_launches,
            result.world_view_validation.host_write_ops,
            result.world_view_validation.kernel_timing,
        );
        assert_phase(
            "accepted materialization",
            result.materialization.kernel_launches,
            result.materialization.host_write_ops,
            result.materialization.kernel_timing,
        );
        assert_phase(
            "final result materialization",
            result.final_result_materialization.kernel_launches,
            result.final_result_materialization.host_write_ops,
            result.final_result_materialization.kernel_timing,
        );
        assert_phase(
            "final tuple materialization",
            result.final_tuple_materialization.kernel_launches,
            result.final_tuple_materialization.host_write_ops,
            result.final_tuple_materialization.kernel_timing,
        );

        let total_kernel_launches = result.candidate_generation.kernel_launches
            + result.propagation.kernel_launches
            + result.candidate_validation.kernel_launches
            + result.model_membership.kernel_launches
            + result.world_view_validation.kernel_launches
            + result.materialization.kernel_launches
            + result.final_result_materialization.kernel_launches
            + result.final_tuple_materialization.kernel_launches;
        assert_eq!(total_kernel_launches, 11);
        assert_eq!(result.aggregate_kernel_timing().cuda_event_pairs, 8);
        assert_eq!(result.aggregate_kernel_timing().timing_sync_ops, 8);
    }
}

#[test]
fn accepted_split_quaternary_all_operator_batch_records_device_workspace_buffers() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (split, batch) = execute_split_quaternary_all_operator_batch(&fix);

    assert_eq!(split.components.len(), 4);
    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 4);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);

    for result in &batch.results {
        let layout = result.prepared.preflight.workspace_layout;
        let workspace = &result.prepared.workspace;
        let reset = result.prepared.workspace_reset;

        assert_eq!(workspace.layout, layout);
        assert_eq!(
            workspace.candidate_assumptions.len(),
            layout.candidate_assumption_bytes
        );
        assert_eq!(workspace.world_views.len(), layout.world_view_bytes);
        assert_eq!(
            workspace.model_membership.len(),
            layout.model_membership_bytes
        );
        assert_eq!(
            workspace.rejection_reasons.len(),
            layout.rejection_reason_slots
        );

        let workspace_ptrs = BTreeSet::from([
            *workspace.candidate_assumptions.device_ptr(),
            *workspace.world_views.device_ptr(),
            *workspace.model_membership.device_ptr(),
            *workspace.rejection_reasons.device_ptr(),
        ]);
        assert_eq!(workspace_ptrs.len(), 4);
        assert!(workspace_ptrs.iter().all(|ptr| *ptr != 0));

        assert_eq!(
            reset.candidate_assumption_bytes,
            layout.candidate_assumption_bytes
        );
        assert_eq!(reset.world_view_bytes, layout.world_view_bytes);
        assert_eq!(reset.model_membership_bytes, layout.model_membership_bytes);
        assert_eq!(
            reset.rejection_reason_bytes,
            layout.rejection_reason_slots * std::mem::size_of::<u32>()
        );
        assert_eq!(reset.device_zero_ops, 4);
        assert_eq!(reset.host_write_ops, 0);
        assert_eq!(reset.total_zeroed_bytes(), layout.total_bytes());

        assert_eq!(
            layout.candidate_assumption_bytes,
            result.candidate_generation.candidate_assumption_bytes
        );
        assert_eq!(
            layout.world_view_bytes,
            result.propagation.world_view_bytes_written
        );
        assert_eq!(
            layout.model_membership_bytes,
            result.model_membership.model_membership_bytes_written
        );
        assert_eq!(
            layout.rejection_reason_slots,
            result.propagation.rejection_reason_slots_written
        );
        assert_eq!(
            layout.rejection_reason_slots,
            result.semantic_trace.generated_candidates
        );
        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
    }
}

#[test]
fn accepted_split_quaternary_all_operator_batch_rejects_cpu_fallback_counters() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (split, mut batch) = execute_split_quaternary_all_operator_batch(&fix);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);

    batch.trace.cpu_candidate_enumerations = 1;
    batch.trace.cpu_world_view_validations = 1;

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut solver_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = solver_adapter
        .new_workspace(sat_cnf.var_cap, sat_cnf.clause_cap)
        .expect("new fallback rejection workspace");

    let solver_err = match solver_adapter
        .solve_assumption_lifecycle_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &[GpuSolverProductionLifecycleStep {
                cnf: &sat_cnf,
                branch_var_limit: &branch_limit,
                expectation: GpuSolverProductionExpectation::Sat,
            }],
        ) {
        Ok(_) => panic!("solver batch evidence with CPU fallback counters must reject"),
        Err(err) => err,
    };
    let solver_err = format!("{solver_err}");
    assert!(solver_err.contains("CPU/host fallback counters"));
    assert!(solver_err.contains("cpu_candidates=1"));
    assert!(solver_err.contains("cpu_world_views=1"));

    let terms = |a, b, c, d| {
        vec![
            EpistemicEvidenceTerm::integer(a),
            EpistemicEvidenceTerm::integer(b),
            EpistemicEvidenceTerm::integer(c),
            EpistemicEvidenceTerm::integer(d),
        ]
    };
    let assumption_groups_owned: Vec<Vec<EpistemicAssumption>> = split
        .components
        .iter()
        .map(
            |component| match component.component.rule_indices.as_slice() {
                [0] => vec![EpistemicAssumption::known_tuple(
                    "edge4",
                    terms(2, 3, 4, 5),
                    true,
                )],
                [1] => vec![EpistemicAssumption::possible_tuple(
                    "alt4",
                    terms(3, 4, 5, 6),
                    true,
                )],
                [2] => vec![EpistemicAssumption::possible_tuple(
                    "blocked_fact4",
                    terms(1, 2, 3, 4),
                    false,
                )],
                [3] => vec![EpistemicAssumption::known_tuple(
                    "hidden_fact4",
                    terms(2, 3, 4, 5),
                    false,
                )],
                other => panic!("unexpected split quaternary fallback rule indices: {other:?}"),
            },
        )
        .collect();
    let assumption_groups: Vec<&[EpistemicAssumption]> =
        assumption_groups_owned.iter().map(Vec::as_slice).collect();

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut prob_adapter = EpistemicProbProductionAdapter::new(config);
    let prob_err = match prob_adapter
        .compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result(
            r#"
        0.8::edge4(2, 3, 4, 5).
        0.7::alt4(3, 4, 5, 6).
        0.6::blocked_fact4(1, 2, 3, 4).
        0.5::hidden_fact4(2, 3, 4, 5).
        query(edge4(2, 3, 4, 5)).
        query(alt4(3, 4, 5, 6)).
        query(blocked_fact4(1, 2, 3, 4)).
        query(hidden_fact4(2, 3, 4, 5)).
        "#,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        ) {
        Ok(_) => panic!("probabilistic batch evidence with CPU fallback counters must reject"),
        Err(err) => err,
    };
    let prob_err = format!("{prob_err}");
    assert!(prob_err.contains("CPU/host fallback counters"));
    assert!(prob_err.contains("cpu_candidates=1"));
    assert!(prob_err.contains("cpu_world_views=1"));
}

#[test]
fn accepted_gpu_execution_result_rejects_cpu_fallback_counters() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let mut result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
        &[1, 2],
        &[1],
    );
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![1]
    );
    assert!(result.prepared.preflight.cpu_fallbacks.is_zero());
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    result.prepared.preflight.cpu_fallbacks = EpistemicCpuFallbackCounters {
        candidate_enumeration: 1,
        world_view_validation: 1,
        solver_search: 1,
        probabilistic_recompute: 1,
    };

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut solver_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = solver_adapter
        .new_workspace(sat_cnf.var_cap, sat_cnf.clause_cap)
        .expect("new single-result fallback rejection workspace");

    let solver_err = match solver_adapter.solve_assumption_lifecycle_with_gpu_execution_result(
        &fix.provider,
        &result,
        &mut workspace,
        &[GpuSolverProductionLifecycleStep {
            cnf: &sat_cnf,
            branch_var_limit: &branch_limit,
            expectation: GpuSolverProductionExpectation::Sat,
        }],
    ) {
        Ok(_) => panic!("solver evidence with single-result CPU fallbacks must reject"),
        Err(err) => err,
    };
    assert!(format!("{solver_err}").contains("zero epistemic CPU fallback counters"));
    let solver_trace = solver_adapter.trace();
    assert_eq!(solver_trace.accepted_gpu_candidate_evidence_consumed, 0);
    assert_eq!(solver_trace.gpu_assumption_pushes, 0);
    assert_eq!(solver_trace.cpu_assignment_enumerations, 0);

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut prob_adapter = EpistemicProbProductionAdapter::new(config);
    let prob_err = match prob_adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
        0.8::edge(1).
        query(edge(1)).
        "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known_tuple(
                "edge",
                vec![EpistemicEvidenceTerm::integer(1)],
                true,
            )],
        ) {
        Ok(_) => panic!("probability evidence with single-result CPU fallbacks must reject"),
        Err(err) => err,
    };
    assert!(format!("{prob_err}").contains("zero epistemic CPU fallback counters"));
    let prob_trace = prob_adapter.trace();
    assert_eq!(prob_trace.accepted_world_view_evidence_consumed, 0);
    assert_eq!(prob_trace.gpu_conditioned_evidence_facts, 0);
    assert_eq!(prob_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(prob_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_result_rejects_hot_path_host_transfers() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let mut result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
        &[1, 2],
        &[1],
    );
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![1]
    );
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    assert_eq!(result.transfer_budget.tracked_htod_calls, 0);
    assert_eq!(result.transfer_budget.per_candidate_host_round_trips, 0);

    result.transfer_budget.tracked_dtoh_calls = 1;
    result.transfer_budget.tracked_htod_calls = 1;
    result.transfer_budget.per_candidate_host_round_trips = 1;

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut solver_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = solver_adapter
        .new_workspace(sat_cnf.var_cap, sat_cnf.clause_cap)
        .expect("new single-result transfer rejection workspace");

    let solver_err = match solver_adapter.solve_assumption_lifecycle_with_gpu_execution_result(
        &fix.provider,
        &result,
        &mut workspace,
        &[GpuSolverProductionLifecycleStep {
            cnf: &sat_cnf,
            branch_var_limit: &branch_limit,
            expectation: GpuSolverProductionExpectation::Sat,
        }],
    ) {
        Ok(_) => panic!("solver evidence with single-result hot transfers must reject"),
        Err(err) => err,
    };
    let solver_err = format!("{solver_err}");
    assert!(solver_err.contains("zero hot-path transfers"));
    assert!(solver_err.contains("dtoh_calls=1"));
    assert!(solver_err.contains("htod_calls=1"));
    assert!(solver_err.contains("per_candidate_round_trips=1"));
    let solver_trace = solver_adapter.trace();
    assert_eq!(solver_trace.accepted_gpu_candidate_evidence_consumed, 0);
    assert_eq!(solver_trace.gpu_assumption_pushes, 0);
    assert_eq!(solver_trace.cpu_assignment_enumerations, 0);

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut prob_adapter = EpistemicProbProductionAdapter::new(config);
    let prob_err = match prob_adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
        0.8::edge(1).
        query(edge(1)).
        "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known_tuple(
                "edge",
                vec![EpistemicEvidenceTerm::integer(1)],
                true,
            )],
        ) {
        Ok(_) => panic!("probability evidence with single-result hot transfers must reject"),
        Err(err) => err,
    };
    let prob_err = format!("{prob_err}");
    assert!(prob_err.contains("zero hot-path transfers"));
    assert!(prob_err.contains("dtoh_calls=1"));
    assert!(prob_err.contains("htod_calls=1"));
    assert!(prob_err.contains("per_candidate_round_trips=1"));
    let prob_trace = prob_adapter.trace();
    assert_eq!(prob_trace.accepted_world_view_evidence_consumed, 0);
    assert_eq!(prob_trace.gpu_conditioned_evidence_facts, 0);
    assert_eq!(prob_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(prob_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_quaternary_all_operator_batch_rejects_row_count_only_membership() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (split, mut batch) = execute_split_quaternary_all_operator_batch(&fix);
    assert_eq!(batch.results.len(), 4);
    assert!(batch
        .results
        .iter()
        .all(|result| result.model_membership.membership_source
            == EpistemicGpuModelMembershipSource::StableModelTupleBuffer));

    batch.results[0].model_membership.membership_source =
        EpistemicGpuModelMembershipSource::ReducedOutputRowCountOnly;

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut solver_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = solver_adapter
        .new_workspace(sat_cnf.var_cap, sat_cnf.clause_cap)
        .expect("new row-count rejection workspace");

    let solver_err = match solver_adapter
        .solve_assumption_lifecycle_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &[GpuSolverProductionLifecycleStep {
                cnf: &sat_cnf,
                branch_var_limit: &branch_limit,
                expectation: GpuSolverProductionExpectation::Sat,
            }],
        ) {
        Ok(_) => panic!("solver batch evidence with row-count-only membership must reject"),
        Err(err) => err,
    };
    let solver_err = format!("{solver_err}");
    assert!(solver_err.contains("stable-model membership certification"));
    assert!(solver_err.contains("ReducedOutputRowCountOnly"));

    let terms = |a, b, c, d| {
        vec![
            EpistemicEvidenceTerm::integer(a),
            EpistemicEvidenceTerm::integer(b),
            EpistemicEvidenceTerm::integer(c),
            EpistemicEvidenceTerm::integer(d),
        ]
    };
    let assumption_groups_owned: Vec<Vec<EpistemicAssumption>> = split
        .components
        .iter()
        .map(
            |component| match component.component.rule_indices.as_slice() {
                [0] => vec![EpistemicAssumption::known_tuple(
                    "edge4",
                    terms(2, 3, 4, 5),
                    true,
                )],
                [1] => vec![EpistemicAssumption::possible_tuple(
                    "alt4",
                    terms(3, 4, 5, 6),
                    true,
                )],
                [2] => vec![EpistemicAssumption::possible_tuple(
                    "blocked_fact4",
                    terms(1, 2, 3, 4),
                    false,
                )],
                [3] => vec![EpistemicAssumption::known_tuple(
                    "hidden_fact4",
                    terms(2, 3, 4, 5),
                    false,
                )],
                other => panic!("unexpected split quaternary row-count rule indices: {other:?}"),
            },
        )
        .collect();
    let assumption_groups: Vec<&[EpistemicAssumption]> =
        assumption_groups_owned.iter().map(Vec::as_slice).collect();

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut prob_adapter = EpistemicProbProductionAdapter::new(config);
    let prob_err = match prob_adapter
        .compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result(
            r#"
        0.8::edge4(2, 3, 4, 5).
        0.7::alt4(3, 4, 5, 6).
        0.6::blocked_fact4(1, 2, 3, 4).
        0.5::hidden_fact4(2, 3, 4, 5).
        query(edge4(2, 3, 4, 5)).
        query(alt4(3, 4, 5, 6)).
        query(blocked_fact4(1, 2, 3, 4)).
        query(hidden_fact4(2, 3, 4, 5)).
        "#,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        ) {
        Ok(_) => panic!("probabilistic batch evidence with row-count-only membership must reject"),
        Err(err) => err,
    };
    let prob_err = format!("{prob_err}");
    assert!(prob_err.contains("stable-model membership certification"));
    assert!(prob_err.contains("ReducedOutputRowCountOnly"));
}

#[test]
fn accepted_split_quaternary_all_operator_batch_rejects_hot_path_host_transfers() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (_split, mut batch) = execute_split_quaternary_all_operator_batch(&fix);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);

    batch.trace.tracked_dtoh_calls = 1;
    batch.trace.per_candidate_host_round_trips = 1;

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut solver_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = solver_adapter
        .new_workspace(sat_cnf.var_cap, sat_cnf.clause_cap)
        .expect("new host-transfer rejection workspace");

    let solver_err = match solver_adapter
        .solve_assumption_lifecycle_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &[GpuSolverProductionLifecycleStep {
                cnf: &sat_cnf,
                branch_var_limit: &branch_limit,
                expectation: GpuSolverProductionExpectation::Sat,
            }],
        ) {
        Ok(_) => panic!("solver batch evidence with hot-path host transfers must reject"),
        Err(err) => err,
    };
    let solver_err = format!("{solver_err}");
    assert!(solver_err.contains("CPU/host fallback counters"));
    assert!(solver_err.contains("dtoh_calls=1"));
    assert!(solver_err.contains("round_trips=1"));

    let empty_groups_owned: Vec<Vec<EpistemicAssumption>> = vec![Vec::new(); batch.results.len()];
    let empty_groups: Vec<&[EpistemicAssumption]> =
        empty_groups_owned.iter().map(Vec::as_slice).collect();

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut prob_adapter = EpistemicProbProductionAdapter::new(config);
    let prob_err = match prob_adapter
        .compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result(
            r#"
        0.8::edge4(2, 3, 4, 5).
        query(edge4(2, 3, 4, 5)).
        "#,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &empty_groups,
            },
        ) {
        Ok(_) => panic!("probabilistic batch evidence with hot-path host transfers must reject"),
        Err(err) => err,
    };
    let prob_err = format!("{prob_err}");
    assert!(prob_err.contains("CPU/host fallback counters"));
    assert!(prob_err.contains("dtoh_calls=1"));
    assert!(prob_err.contains("round_trips=1"));
}

#[test]
fn split_multi_membership_modal_coupling_rejects_gpu_batching() {
    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred alt(u32).
        pred blocked(u32).
        pred both_known(u32).
        pred safe_alt(u32).
        both_known(X) :- node(X), know edge(X), know color(X).
        safe_alt(X) :- node(X), possible alt(X), not possible blocked(X).
        "#,
    )
    .expect("parse split multi-membership fixture");
    let err = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect_err("multi-membership modal coupling must reject unsafe split batching");

    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic splitting");
            assert!(context.contains("rule[0] couples epistemic predicates"));
            assert!(context.contains("\"edge\""));
            assert!(context.contains("\"color\""));
        }
        other => panic!("expected typed split rejection, got {other:?}"),
    }
}

#[test]
fn accepted_split_all_binary_operator_batch_conditions_probabilistic_evidence() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred alt(u32, u32).
        pred blocked(u32, u32).
        pred seen(u32, u32).
        pred known_edge(u32, u32).
        pred possible_alt(u32, u32).
        pred clear_pair(u32, u32).
        pred unknown_pair(u32, u32).
        known_edge(X, Y) :- pair(X, Y), know edge(X, Y).
        possible_alt(X, Y) :- pair(X, Y), possible alt(X, Y).
        clear_pair(X, Y) :- pair(X, Y), not possible blocked(X, Y).
        unknown_pair(X, Y) :- pair(X, Y), not know seen(X, Y).
        "#,
    )
    .expect("parse split all binary probabilistic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split all binary operator components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split all-operator components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "pair",
        upload_binary_u32(&fix.memory, &[(1, 2), (2, 3), (3, 4)]),
    );
    executor.put_relation("edge", upload_binary_u32(&fix.memory, &[(1, 2), (3, 4)]));
    executor.put_relation("alt", upload_binary_u32(&fix.memory, &[(2, 3)]));
    executor.put_relation("blocked", upload_binary_u32(&fix.memory, &[(3, 4)]));
    executor.put_relation("seen", upload_binary_u32(&fix.memory, &[(1, 2)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split all binary operator components through GPU batch path");

    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);

    let assumption_groups_owned: Vec<Vec<EpistemicAssumption>> = split
        .components
        .iter()
        .map(
            |component| match component.component.rule_indices.as_slice() {
                [0] => vec![EpistemicAssumption::known_tuple(
                    "edge",
                    vec![
                        EpistemicEvidenceTerm::integer(1),
                        EpistemicEvidenceTerm::integer(2),
                    ],
                    true,
                )],
                [1] => vec![EpistemicAssumption::possible_tuple(
                    "alt",
                    vec![
                        EpistemicEvidenceTerm::integer(2),
                        EpistemicEvidenceTerm::integer(3),
                    ],
                    true,
                )],
                [2] => vec![EpistemicAssumption::possible_tuple(
                    "blocked",
                    vec![
                        EpistemicEvidenceTerm::integer(1),
                        EpistemicEvidenceTerm::integer(2),
                    ],
                    false,
                )],
                [3] => vec![EpistemicAssumption::known_tuple(
                    "seen",
                    vec![
                        EpistemicEvidenceTerm::integer(2),
                        EpistemicEvidenceTerm::integer(3),
                    ],
                    false,
                )],
                other => panic!("unexpected split all-operator rule indices: {other:?}"),
            },
        )
        .collect();
    let assumption_groups: Vec<&[EpistemicAssumption]> =
        assumption_groups_owned.iter().map(Vec::as_slice).collect();

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result(
            r#"
            0.4::edge(1, 2).
            0.5::alt(2, 3).
            0.7::blocked(1, 2).
            0.8::seen(2, 3).
            query(edge(1, 2)).
            query(alt(2, 3)).
            query(blocked(1, 2)).
            query(seen(2, 3)).
            "#,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split all-operator batch must condition probabilistic exact evidence");

    assert_eq!(evaluated.len(), 4);
    for (component, evaluated_result) in split.components.iter().zip(evaluated.iter()) {
        let expected = match component.component.rule_indices.as_slice() {
            [0] => [1.0, 0.5, 0.7, 0.8],
            [1] => [0.4, 1.0, 0.7, 0.8],
            [2] => [0.4, 0.5, 0.0, 0.8],
            [3] => [0.4, 0.5, 0.7, 0.0],
            other => panic!("unexpected split all-operator rule indices: {other:?}"),
        };
        assert_eq!(evaluated_result.query_probs.len(), expected.len());
        for (actual, expected) in evaluated_result.query_probs.iter().zip(expected) {
            assert!(
                (actual.prob - expected).abs() < 1.0e-6,
                "conditioned split all-operator probability mismatch: actual={} expected={}",
                actual.prob,
                expected
            );
        }
    }

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 4);
    assert_eq!(trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_known_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 4);
    assert_eq!(trace.gpu_exact_query_evaluations, 4);
    assert_eq!(trace.gpu_source_exact_query_evaluations, 4);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_all_binary_operator_batch_updates_incremental_probability_circuit() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (split, batch) = execute_split_all_binary_operator_batch(&fix);
    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 4);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    let assumption_groups_owned: Vec<Vec<EpistemicAssumption>> = split
        .components
        .iter()
        .map(
            |component| match component.component.rule_indices.as_slice() {
                [0] => vec![EpistemicAssumption::known_tuple(
                    "edge",
                    vec![
                        EpistemicEvidenceTerm::integer(1),
                        EpistemicEvidenceTerm::integer(2),
                    ],
                    true,
                )],
                [1] => vec![EpistemicAssumption::possible_tuple(
                    "alt",
                    vec![
                        EpistemicEvidenceTerm::integer(2),
                        EpistemicEvidenceTerm::integer(3),
                    ],
                    true,
                )],
                [2] => vec![EpistemicAssumption::possible_tuple(
                    "blocked",
                    vec![
                        EpistemicEvidenceTerm::integer(1),
                        EpistemicEvidenceTerm::integer(2),
                    ],
                    false,
                )],
                [3] => vec![EpistemicAssumption::known_tuple(
                    "seen",
                    vec![
                        EpistemicEvidenceTerm::integer(2),
                        EpistemicEvidenceTerm::integer(3),
                    ],
                    false,
                )],
                other => panic!("unexpected split all-operator rule indices: {other:?}"),
            },
        )
        .collect();
    let assumption_groups: Vec<&[EpistemicAssumption]> =
        assumption_groups_owned.iter().map(Vec::as_slice).collect();
    let conditioned = assumption_groups_owned
        .iter()
        .flat_map(|group| group.iter().cloned())
        .map(|assumption| (assumption, 0.9))
        .collect();
    let mut circuit =
        EpistemicCircuit::compile(0.15, conditioned, KnowledgeCompilerAdapter::gpu_d4())
            .expect("compile split-batch incremental probability circuit");
    let original_fingerprint = circuit.circuit_fingerprint();

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let updates = adapter
        .apply_accepted_world_views_to_circuit_for_gpu_batch_execution_result(
            &mut circuit,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split GPU evidence must update incremental circuit evidence");

    assert_eq!(updates.len(), 4);
    for update in &updates {
        assert_eq!(update.mode, CircuitUpdateMode::IncrementalEvidence);
        assert_eq!(update.compile_count, 1);
        assert_eq!(update.circuit_fingerprint, original_fingerprint);
    }
    assert_eq!(circuit.incremental_update_count(), 4);
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec![
            "know:edge/2(1,2)=true".to_string(),
            "know:seen/2(2,3)=false".to_string(),
            "possible:alt/2(2,3)=true".to_string(),
            "possible:blocked/2(1,2)=false".to_string(),
        ]
    );

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 4);
    assert_eq!(trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(trace.accepted_incremental_circuit_updates, 4);
    assert_eq!(trace.gpu_exact_source_compiles, 0);
    assert_eq!(trace.gpu_exact_program_compiles, 0);
    assert_eq!(trace.gpu_exact_query_evaluations, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    let metric_err = trace.require_production_metric_eligibility().expect_err(
        "split incremental fixture circuit update alone must not satisfy production metrics",
    );
    assert!(format!("{metric_err}")
        .contains("existing GPU exact/provenance/PIR/CNF/knowledge-compilation counter"));
}

#[test]
fn accepted_split_quaternary_all_operator_batch_conditions_probabilistic_evidence() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (split, batch) = execute_split_quaternary_all_operator_batch(&fix);

    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 4);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    let assumption_groups_owned: Vec<Vec<EpistemicAssumption>> = split
        .components
        .iter()
        .enumerate()
        .map(
            |(idx, component)| match component.component.rule_indices.as_slice() {
                [0] => {
                    assert_eq!(
                        download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                        vec![(2, 3, 4, 5)]
                    );
                    assert_eq!(batch.results[idx].prepared.preflight.know_operator_count, 1);
                    assert_eq!(
                        batch.results[idx]
                            .final_tuple_materialization
                            .negated_row_filter_count,
                        0
                    );
                    vec![EpistemicAssumption::known_tuple(
                        "edge4",
                        vec![
                            EpistemicEvidenceTerm::integer(2),
                            EpistemicEvidenceTerm::integer(3),
                            EpistemicEvidenceTerm::integer(4),
                            EpistemicEvidenceTerm::integer(5),
                        ],
                        true,
                    )]
                }
                [1] => {
                    assert_eq!(
                        download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                        vec![(3, 4, 5, 6)]
                    );
                    assert_eq!(
                        batch.results[idx]
                            .prepared
                            .preflight
                            .possible_operator_count,
                        1
                    );
                    assert_eq!(
                        batch.results[idx]
                            .final_tuple_materialization
                            .negated_row_filter_count,
                        0
                    );
                    vec![EpistemicAssumption::possible_tuple(
                        "alt4",
                        vec![
                            EpistemicEvidenceTerm::integer(3),
                            EpistemicEvidenceTerm::integer(4),
                            EpistemicEvidenceTerm::integer(5),
                            EpistemicEvidenceTerm::integer(6),
                        ],
                        true,
                    )]
                }
                [2] => {
                    assert_eq!(
                        download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                        vec![(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)]
                    );
                    assert_eq!(
                        batch.results[idx]
                            .prepared
                            .preflight
                            .not_possible_operator_count,
                        1
                    );
                    assert_eq!(
                        batch.results[idx]
                            .final_tuple_materialization
                            .negated_row_filter_count,
                        1
                    );
                    vec![EpistemicAssumption::possible_tuple(
                        "blocked_fact4",
                        vec![
                            EpistemicEvidenceTerm::integer(1),
                            EpistemicEvidenceTerm::integer(2),
                            EpistemicEvidenceTerm::integer(3),
                            EpistemicEvidenceTerm::integer(4),
                        ],
                        false,
                    )]
                }
                [3] => {
                    assert_eq!(
                        download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                        vec![(2, 3, 4, 5), (3, 4, 5, 6), (9, 9, 9, 9)]
                    );
                    assert_eq!(
                        batch.results[idx]
                            .prepared
                            .preflight
                            .not_know_operator_count,
                        1
                    );
                    assert_eq!(
                        batch.results[idx]
                            .final_tuple_materialization
                            .negated_row_filter_count,
                        1
                    );
                    vec![EpistemicAssumption::known_tuple(
                        "hidden_fact4",
                        vec![
                            EpistemicEvidenceTerm::integer(2),
                            EpistemicEvidenceTerm::integer(3),
                            EpistemicEvidenceTerm::integer(4),
                            EpistemicEvidenceTerm::integer(5),
                        ],
                        false,
                    )]
                }
                other => panic!("unexpected split quaternary all-operator indices: {other:?}"),
            },
        )
        .collect();
    let assumption_groups: Vec<&[EpistemicAssumption]> =
        assumption_groups_owned.iter().map(Vec::as_slice).collect();

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result(
            r#"
            0.8::edge4(2, 3, 4, 5).
            0.7::alt4(3, 4, 5, 6).
            0.6::blocked_fact4(1, 2, 3, 4).
            0.5::hidden_fact4(2, 3, 4, 5).
            query(edge4(2, 3, 4, 5)).
            query(alt4(3, 4, 5, 6)).
            query(blocked_fact4(1, 2, 3, 4)).
            query(hidden_fact4(2, 3, 4, 5)).
            "#,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect(
            "accepted split quaternary all-operator batch must condition source exact evidence",
        );

    assert_eq!(evaluated.len(), 4);
    for (component, evaluated_result) in split.components.iter().zip(evaluated.iter()) {
        let expected = match component.component.rule_indices.as_slice() {
            [0] => [1.0, 0.7, 0.6, 0.5],
            [1] => [0.8, 1.0, 0.6, 0.5],
            [2] => [0.8, 0.7, 0.0, 0.5],
            [3] => [0.8, 0.7, 0.6, 0.0],
            other => panic!("unexpected split quaternary all-operator indices: {other:?}"),
        };
        assert_eq!(evaluated_result.query_probs.len(), expected.len());
        for (actual, expected) in evaluated_result.query_probs.iter().zip(expected) {
            assert!(
                (actual.prob - expected).abs() < 1.0e-6,
                "conditioned split quaternary all-operator probability mismatch: actual={} expected={}",
                actual.prob,
                expected
            );
        }
    }

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 4);
    assert_eq!(trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 4);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 4);
    assert_eq!(trace.gpu_program_conditioned_evidence_facts, 0);
    assert_eq!(
        trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        0
    );
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_program_conditioned_max_evidence_arity, 0);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_negative_evidence_facts, 2);
    assert_eq!(trace.gpu_program_conditioned_negative_evidence_facts, 0);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_known_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_not_known_evidence_facts, 1);
    assert_eq!(trace.gpu_program_conditioned_know_evidence_facts, 0);
    assert_eq!(trace.gpu_program_conditioned_possible_evidence_facts, 0);
    assert_eq!(trace.gpu_program_conditioned_not_possible_evidence_facts, 0);
    assert_eq!(trace.gpu_program_conditioned_not_known_evidence_facts, 0);
    assert_eq!(trace.gpu_exact_source_compiles, 4);
    assert_eq!(trace.gpu_exact_program_compiles, 0);
    assert_eq!(trace.gpu_exact_query_evaluations, 4);
    assert_eq!(trace.gpu_source_exact_query_evaluations, 4);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 4);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 4);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_quaternary_all_operator_batch_gates_probabilistic_program_and_gradient_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (split, batch) = execute_split_quaternary_all_operator_batch(&fix);

    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 4);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    let terms = |a, b, c, d| {
        vec![
            EpistemicEvidenceTerm::integer(a),
            EpistemicEvidenceTerm::integer(b),
            EpistemicEvidenceTerm::integer(c),
            EpistemicEvidenceTerm::integer(d),
        ]
    };
    let assumption_groups_owned: Vec<Vec<EpistemicAssumption>> = split
        .components
        .iter()
        .map(
            |component| match component.component.rule_indices.as_slice() {
                [0] => vec![EpistemicAssumption::known_tuple(
                    "edge4",
                    terms(2, 3, 4, 5),
                    true,
                )],
                [1] => vec![EpistemicAssumption::possible_tuple(
                    "alt4",
                    terms(3, 4, 5, 6),
                    true,
                )],
                [2] => vec![EpistemicAssumption::possible_tuple(
                    "blocked_fact4",
                    terms(1, 2, 3, 4),
                    false,
                )],
                [3] => vec![EpistemicAssumption::known_tuple(
                    "hidden_fact4",
                    terms(2, 3, 4, 5),
                    false,
                )],
                other => panic!("unexpected split quaternary all-operator indices: {other:?}"),
            },
        )
        .collect();
    let assumption_groups: Vec<&[EpistemicAssumption]> =
        assumption_groups_owned.iter().map(Vec::as_slice).collect();

    let probabilistic_source = r#"
        0.8::edge4(2, 3, 4, 5).
        0.7::alt4(3, 4, 5, 6).
        0.6::blocked_fact4(1, 2, 3, 4).
        0.5::hidden_fact4(2, 3, 4, 5).
        query(edge4(2, 3, 4, 5)).
        query(alt4(3, 4, 5, 6)).
        query(blocked_fact4(1, 2, 3, 4)).
        query(hidden_fact4(2, 3, 4, 5)).
        "#;
    let parsed_probabilistic_program =
        parse_program(probabilistic_source).expect("parse split quaternary all-operator program");

    let assert_expected = |rule_indices: &[usize], probs: &[f64]| {
        let expected = match rule_indices {
            [0] => [1.0, 0.7, 0.6, 0.5],
            [1] => [0.8, 1.0, 0.6, 0.5],
            [2] => [0.8, 0.7, 0.0, 0.5],
            [3] => [0.8, 0.7, 0.6, 0.0],
            other => panic!("unexpected split quaternary all-operator indices: {other:?}"),
        };
        assert_eq!(probs.len(), expected.len());
        for (actual, expected) in probs.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "conditioned split quaternary all-operator probability mismatch: actual={actual} expected={expected}",
            );
        }
    };

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;

    let mut program_adapter = EpistemicProbProductionAdapter::new(config);
    let program_evaluated = program_adapter
        .compile_and_evaluate_conditioned_program_for_gpu_batch_execution_result(
            &parsed_probabilistic_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect(
            "accepted split quaternary all-operator batch must condition parsed-program exact evidence",
        );
    assert_eq!(program_evaluated.len(), 4);
    for (component, evaluated_result) in split.components.iter().zip(program_evaluated.iter()) {
        let probs: Vec<f64> = evaluated_result
            .query_probs
            .iter()
            .map(|query| query.prob)
            .collect();
        assert_expected(&component.component.rule_indices, &probs);
    }

    let program_trace = program_adapter.trace();
    assert_eq!(program_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        program_trace.accepted_gpu_batch_component_evidence_consumed,
        4
    );
    assert_eq!(program_trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(program_trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(program_trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(
        program_trace.gpu_conditioned_nonzero_arity_evidence_facts,
        4
    );
    assert_eq!(program_trace.gpu_conditioned_max_evidence_arity, 4);
    assert_eq!(program_trace.gpu_conditioned_negative_evidence_facts, 2);
    assert_eq!(program_trace.gpu_conditioned_know_evidence_facts, 1);
    assert_eq!(program_trace.gpu_conditioned_possible_evidence_facts, 1);
    assert_eq!(program_trace.gpu_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(program_trace.gpu_conditioned_not_known_evidence_facts, 1);
    assert_eq!(program_trace.gpu_source_conditioned_evidence_facts, 0);
    assert_eq!(program_trace.gpu_program_conditioned_evidence_facts, 4);
    assert_eq!(
        program_trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        4
    );
    assert_eq!(program_trace.gpu_program_conditioned_max_evidence_arity, 4);
    assert_eq!(
        program_trace.gpu_program_conditioned_negative_evidence_facts,
        2
    );
    assert_eq!(program_trace.gpu_program_conditioned_know_evidence_facts, 1);
    assert_eq!(
        program_trace.gpu_program_conditioned_possible_evidence_facts,
        1
    );
    assert_eq!(
        program_trace.gpu_program_conditioned_not_possible_evidence_facts,
        1
    );
    assert_eq!(
        program_trace.gpu_program_conditioned_not_known_evidence_facts,
        1
    );
    assert_eq!(program_trace.gpu_exact_source_compiles, 0);
    assert_eq!(program_trace.gpu_exact_program_compiles, 4);
    assert_eq!(program_trace.gpu_exact_query_evaluations, 4);
    assert_eq!(program_trace.gpu_source_exact_query_evaluations, 0);
    assert_eq!(program_trace.gpu_program_exact_query_evaluations, 4);
    assert_eq!(program_trace.gpu_knowledge_compilation_end_to_end_runs, 4);
    assert_eq!(
        program_trace.gpu_program_knowledge_compilation_end_to_end_runs,
        4
    );
    assert_eq!(program_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(program_trace.fixture_circuit_evaluations, 0);

    let mut source_gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let source_gradients = source_gradient_adapter
        .compile_and_evaluate_conditioned_source_with_grads_for_gpu_batch_execution_result(
            probabilistic_source,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split quaternary all-operator batch must condition source gradients");
    assert_eq!(source_gradients.len(), 4);
    for (component, gradient_result) in split.components.iter().zip(source_gradients.iter()) {
        let probs: Vec<f64> = gradient_result
            .query_grads
            .iter()
            .map(|query| query.prob)
            .collect();
        assert_expected(&component.component.rule_indices, &probs);
    }

    let source_gradient_trace = source_gradient_adapter.trace();
    assert_eq!(
        source_gradient_trace.accepted_gpu_batch_evidence_consumed,
        1
    );
    assert_eq!(
        source_gradient_trace.accepted_gpu_batch_component_evidence_consumed,
        4
    );
    assert_eq!(
        source_gradient_trace.accepted_world_view_evidence_consumed,
        4
    );
    assert_eq!(
        source_gradient_trace.accepted_evidence_assumptions_consumed,
        4
    );
    assert_eq!(source_gradient_trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(
        source_gradient_trace.gpu_conditioned_nonzero_arity_evidence_facts,
        4
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_evidence_facts,
        4
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_nonzero_arity_evidence_facts,
        4
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_max_evidence_arity,
        4
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_negative_evidence_facts,
        2
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_know_evidence_facts,
        1
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_possible_evidence_facts,
        1
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_not_possible_evidence_facts,
        1
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_not_known_evidence_facts,
        1
    );
    assert_eq!(
        source_gradient_trace.gpu_program_conditioned_evidence_facts,
        0
    );
    assert_eq!(source_gradient_trace.gpu_exact_source_compiles, 4);
    assert_eq!(source_gradient_trace.gpu_exact_program_compiles, 0);
    assert_eq!(source_gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(source_gradient_trace.gpu_exact_gradient_evaluations, 4);
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_gradient_evaluations,
        4
    );
    assert_eq!(
        source_gradient_trace.gpu_knowledge_compilation_end_to_end_runs,
        4
    );
    assert_eq!(
        source_gradient_trace.gpu_source_knowledge_compilation_end_to_end_runs,
        4
    );
    assert_eq!(source_gradient_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(source_gradient_trace.fixture_circuit_evaluations, 0);

    let mut program_gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let program_gradients = program_gradient_adapter
        .compile_and_evaluate_conditioned_program_with_grads_for_gpu_batch_execution_result(
            &parsed_probabilistic_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect(
            "accepted split quaternary all-operator batch must condition parsed-program gradients",
        );
    assert_eq!(program_gradients.len(), 4);
    for (component, gradient_result) in split.components.iter().zip(program_gradients.iter()) {
        let probs: Vec<f64> = gradient_result
            .query_grads
            .iter()
            .map(|query| query.prob)
            .collect();
        assert_expected(&component.component.rule_indices, &probs);
    }

    let program_gradient_trace = program_gradient_adapter.trace();
    assert_eq!(
        program_gradient_trace.accepted_gpu_batch_evidence_consumed,
        1
    );
    assert_eq!(
        program_gradient_trace.accepted_gpu_batch_component_evidence_consumed,
        4
    );
    assert_eq!(
        program_gradient_trace.accepted_world_view_evidence_consumed,
        4
    );
    assert_eq!(
        program_gradient_trace.accepted_evidence_assumptions_consumed,
        4
    );
    assert_eq!(program_gradient_trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(
        program_gradient_trace.gpu_source_conditioned_evidence_facts,
        0
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_evidence_facts,
        4
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        4
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_max_evidence_arity,
        4
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_negative_evidence_facts,
        2
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_know_evidence_facts,
        1
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_possible_evidence_facts,
        1
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_not_possible_evidence_facts,
        1
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_not_known_evidence_facts,
        1
    );
    assert_eq!(program_gradient_trace.gpu_exact_source_compiles, 0);
    assert_eq!(program_gradient_trace.gpu_exact_program_compiles, 4);
    assert_eq!(program_gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(program_gradient_trace.gpu_exact_gradient_evaluations, 4);
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_gradient_evaluations,
        4
    );
    assert_eq!(
        program_gradient_trace.gpu_knowledge_compilation_end_to_end_runs,
        4
    );
    assert_eq!(
        program_gradient_trace.gpu_program_knowledge_compilation_end_to_end_runs,
        4
    );
    assert_eq!(
        program_gradient_trace.cpu_only_probability_recomputations,
        0
    );
    assert_eq!(program_gradient_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_quaternary_all_operator_batch_gates_probabilistic_pir_cnf_and_exact_evaluation_paths(
) {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (split, batch) = execute_split_quaternary_all_operator_batch(&fix);

    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 4);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    let terms = |a, b, c, d| {
        vec![
            EpistemicEvidenceTerm::integer(a),
            EpistemicEvidenceTerm::integer(b),
            EpistemicEvidenceTerm::integer(c),
            EpistemicEvidenceTerm::integer(d),
        ]
    };
    let assumption_groups_owned: Vec<Vec<EpistemicAssumption>> = split
        .components
        .iter()
        .map(
            |component| match component.component.rule_indices.as_slice() {
                [0] => vec![EpistemicAssumption::known_tuple(
                    "edge4",
                    terms(2, 3, 4, 5),
                    true,
                )],
                [1] => vec![EpistemicAssumption::possible_tuple(
                    "alt4",
                    terms(3, 4, 5, 6),
                    true,
                )],
                [2] => vec![EpistemicAssumption::possible_tuple(
                    "blocked_fact4",
                    terms(1, 2, 3, 4),
                    false,
                )],
                [3] => vec![EpistemicAssumption::known_tuple(
                    "hidden_fact4",
                    terms(2, 3, 4, 5),
                    false,
                )],
                other => panic!("unexpected split quaternary all-operator indices: {other:?}"),
            },
        )
        .collect();
    let assumption_groups: Vec<&[EpistemicAssumption]> =
        assumption_groups_owned.iter().map(Vec::as_slice).collect();

    let probabilistic_source = r#"
        0.8::edge4(2, 3, 4, 5).
        0.7::alt4(3, 4, 5, 6).
        0.6::blocked_fact4(1, 2, 3, 4).
        0.5::hidden_fact4(2, 3, 4, 5).
        query(edge4(2, 3, 4, 5)).
        query(alt4(3, 4, 5, 6)).
        query(blocked_fact4(1, 2, 3, 4)).
        query(hidden_fact4(2, 3, 4, 5)).
        "#;
    let parsed_probabilistic_program = parse_program(probabilistic_source)
        .expect("parse split quaternary all-operator PIR/CNF source");

    let assert_unconditioned = |probs: &[f64]| {
        let expected = [0.8, 0.7, 0.6, 0.5];
        assert_eq!(probs.len(), expected.len());
        for (actual, expected) in probs.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "unconditioned split quaternary all-operator probability mismatch: actual={actual} expected={expected}",
            );
        }
    };

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;

    let mut source_pir_adapter = EpistemicProbProductionAdapter::new(config);
    let source_pir_cnfs = source_pir_adapter
        .encode_source_pir_cnf_for_gpu_batch_execution_result(
            probabilistic_source,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split quaternary all-operator batch must gate source PIR/CNF encoding");
    assert_eq!(source_pir_cnfs.len(), 4);
    for pir_cnf in &source_pir_cnfs {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let source_pir_trace = source_pir_adapter.trace();
    assert_eq!(source_pir_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        source_pir_trace.accepted_gpu_batch_component_evidence_consumed,
        4
    );
    assert_eq!(source_pir_trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(source_pir_trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(source_pir_trace.gpu_pir_graph_uploads, 4);
    assert_eq!(source_pir_trace.gpu_source_pir_graph_uploads, 4);
    assert_eq!(source_pir_trace.gpu_program_pir_graph_uploads, 0);
    assert_eq!(source_pir_trace.gpu_cnf_encodes, 4);
    assert_eq!(source_pir_trace.gpu_source_cnf_encodes, 4);
    assert_eq!(source_pir_trace.gpu_program_cnf_encodes, 0);
    assert_eq!(source_pir_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(source_pir_trace.fixture_circuit_evaluations, 0);

    let mut program_pir_adapter = EpistemicProbProductionAdapter::new(config);
    let program_pir_cnfs = program_pir_adapter
        .encode_program_pir_cnf_for_gpu_batch_execution_result(
            &parsed_probabilistic_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect(
            "accepted split quaternary all-operator batch must gate parsed-program PIR/CNF encoding",
        );
    assert_eq!(program_pir_cnfs.len(), 4);
    for pir_cnf in &program_pir_cnfs {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let program_pir_trace = program_pir_adapter.trace();
    assert_eq!(program_pir_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        program_pir_trace.accepted_gpu_batch_component_evidence_consumed,
        4
    );
    assert_eq!(program_pir_trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(program_pir_trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(program_pir_trace.gpu_pir_graph_uploads, 4);
    assert_eq!(program_pir_trace.gpu_source_pir_graph_uploads, 0);
    assert_eq!(program_pir_trace.gpu_program_pir_graph_uploads, 4);
    assert_eq!(program_pir_trace.gpu_cnf_encodes, 4);
    assert_eq!(program_pir_trace.gpu_source_cnf_encodes, 0);
    assert_eq!(program_pir_trace.gpu_program_cnf_encodes, 4);
    assert_eq!(program_pir_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(program_pir_trace.fixture_circuit_evaluations, 0);

    let mut query_adapter = EpistemicProbProductionAdapter::new(config);
    let exact = query_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &batch.results[0],
            assumption_groups_owned[0].clone(),
        )
        .expect("accepted split quaternary all-operator component must gate exact compile");
    let evaluated = query_adapter
        .evaluate_for_gpu_batch_execution_result(
            &exact,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split quaternary all-operator batch must gate exact query evaluation");
    assert_eq!(evaluated.len(), 4);
    for result in &evaluated {
        let probs: Vec<f64> = result.query_probs.iter().map(|query| query.prob).collect();
        assert_unconditioned(&probs);
    }
    let query_trace = query_adapter.trace();
    assert_eq!(query_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        query_trace.accepted_gpu_batch_component_evidence_consumed,
        4
    );
    assert_eq!(query_trace.accepted_world_view_evidence_consumed, 5);
    assert_eq!(query_trace.accepted_evidence_assumptions_consumed, 5);
    assert_eq!(query_trace.gpu_exact_source_compiles, 1);
    assert_eq!(query_trace.gpu_exact_query_evaluations, 4);
    assert_eq!(query_trace.gpu_exact_gradient_evaluations, 0);
    assert_eq!(query_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(query_trace.fixture_circuit_evaluations, 0);

    let mut gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let exact = gradient_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &batch.results[0],
            assumption_groups_owned[0].clone(),
        )
        .expect("accepted split quaternary all-operator component must gate exact compile");
    let gradients = gradient_adapter
        .evaluate_gpu_with_grads_for_gpu_batch_execution_result(
            &exact,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split quaternary all-operator batch must gate exact gradient evaluation");
    assert_eq!(gradients.len(), 4);
    for result in &gradients {
        let probs: Vec<f64> = result.query_grads.iter().map(|query| query.prob).collect();
        assert_unconditioned(&probs);
    }
    let gradient_trace = gradient_adapter.trace();
    assert_eq!(gradient_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        gradient_trace.accepted_gpu_batch_component_evidence_consumed,
        4
    );
    assert_eq!(gradient_trace.accepted_world_view_evidence_consumed, 5);
    assert_eq!(gradient_trace.accepted_evidence_assumptions_consumed, 5);
    assert_eq!(gradient_trace.gpu_exact_source_compiles, 1);
    assert_eq!(gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(gradient_trace.gpu_exact_gradient_evaluations, 4);
    assert_eq!(gradient_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(gradient_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_quaternary_not_possible_batch_conditions_parsed_program_probabilistic_evidence() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred known4(u32, u32, u32, u32).
        pred clear4(u32, u32, u32, u32).
        known4(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        clear4(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary not-possible probabilistic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split quaternary not-possible components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split quaternary components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split quaternary not-possible batch through GPU runtime path");

    assert_eq!(batch.results.len(), 2);
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);

    let assumption_groups_owned: Vec<Vec<EpistemicAssumption>> = split
        .components
        .iter()
        .enumerate()
        .map(
            |(idx, component)| match component.component.rule_indices.as_slice() {
                [0] => {
                    assert_eq!(
                        download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                        vec![(2, 3, 4, 5)]
                    );
                    vec![EpistemicAssumption::known_tuple(
                        "fact4",
                        vec![
                            EpistemicEvidenceTerm::integer(2),
                            EpistemicEvidenceTerm::integer(3),
                            EpistemicEvidenceTerm::integer(4),
                            EpistemicEvidenceTerm::integer(5),
                        ],
                        true,
                    )]
                }
                [1] => {
                    assert_eq!(
                        download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                        vec![(1, 2, 3, 4), (9, 9, 9, 9)]
                    );
                    assert_eq!(
                        batch.results[idx]
                            .prepared
                            .preflight
                            .not_possible_operator_count,
                        1
                    );
                    assert_eq!(
                        batch.results[idx]
                            .final_tuple_materialization
                            .negated_row_filter_count,
                        1
                    );
                    vec![EpistemicAssumption::possible_tuple(
                        "fact4",
                        vec![
                            EpistemicEvidenceTerm::integer(1),
                            EpistemicEvidenceTerm::integer(2),
                            EpistemicEvidenceTerm::integer(3),
                            EpistemicEvidenceTerm::integer(4),
                        ],
                        false,
                    )]
                }
                other => panic!("unexpected split quaternary rule indices: {other:?}"),
            },
        )
        .collect();
    let assumption_groups: Vec<&[EpistemicAssumption]> =
        assumption_groups_owned.iter().map(Vec::as_slice).collect();
    let parsed_program = parse_program(
        r#"
        0.8::fact4(1, 2, 3, 4).
        0.6::fact4(2, 3, 4, 5).
        query(fact4(1, 2, 3, 4)).
        query(fact4(2, 3, 4, 5)).
        "#,
    )
    .expect("parse split quaternary probabilistic program");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_program_for_gpu_batch_execution_result(
            &parsed_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split quaternary GPU evidence must condition parsed exact evidence");

    assert_eq!(evaluated.len(), 2);
    for (component, evaluated_result) in split.components.iter().zip(evaluated.iter()) {
        let expected = match component.component.rule_indices.as_slice() {
            [0] => [0.8, 1.0],
            [1] => [0.0, 0.6],
            other => panic!("unexpected split quaternary rule indices: {other:?}"),
        };
        assert_eq!(evaluated_result.query_probs.len(), expected.len());
        for (actual, expected) in evaluated_result.query_probs.iter().zip(expected) {
            assert!(
                (actual.prob - expected).abs() < 1.0e-6,
                "conditioned split quaternary probability mismatch: actual={} expected={}",
                actual.prob,
                expected
            );
        }
    }

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 2);
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 0);
    assert_eq!(
        trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        2
    );
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 0);
    assert_eq!(trace.gpu_program_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_program_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_program_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_quaternary_possible_and_not_know_batch_gates_solver_and_probabilistic_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred possible4(u32, u32, u32, u32).
        pred unknown4(u32, u32, u32, u32).
        possible4(A, B, C, D) :- tuple4(A, B, C, D), possible fact4(A, B, C, D).
        unknown4(A, B, C, D) :- tuple4(A, B, C, D), not know fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary possible/not-know production reuse fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split quaternary possible/not-know components");

    assert_eq!(split.components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split quaternary possible/not-know components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split quaternary possible/not-know components through GPU batch path");

    assert_eq!(batch.results.len(), 2);
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 2);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());
    assert_eq!(batch.trace.know_operator_count, 0);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 0);

    let assumption_groups_owned: Vec<Vec<EpistemicAssumption>> = split
        .components
        .iter()
        .enumerate()
        .map(
            |(idx, component)| match component.component.rule_indices.as_slice() {
                [0] => {
                    assert_eq!(
                        download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                        vec![(2, 3, 4, 5)]
                    );
                    assert_eq!(
                        batch.results[idx]
                            .prepared
                            .preflight
                            .possible_operator_count,
                        1
                    );
                    assert_eq!(
                        batch.results[idx]
                            .final_tuple_materialization
                            .negated_row_filter_count,
                        0
                    );
                    assert_eq!(
                        batch.results[idx]
                            .model_membership
                            .tuple_source_key_column_device_reads,
                        4
                    );
                    assert_eq!(
                        batch.results[idx].model_membership.membership_source,
                        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                    );
                    vec![EpistemicAssumption::possible_tuple(
                        "fact4",
                        vec![
                            EpistemicEvidenceTerm::integer(2),
                            EpistemicEvidenceTerm::integer(3),
                            EpistemicEvidenceTerm::integer(4),
                            EpistemicEvidenceTerm::integer(5),
                        ],
                        true,
                    )]
                }
                [1] => {
                    assert_eq!(
                        download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                        vec![(1, 2, 3, 4), (9, 9, 9, 9)]
                    );
                    assert_eq!(
                        batch.results[idx]
                            .prepared
                            .preflight
                            .not_know_operator_count,
                        1
                    );
                    assert_eq!(
                        batch.results[idx]
                            .final_tuple_materialization
                            .negated_row_filter_count,
                        1
                    );
                    assert_eq!(
                        batch.results[idx]
                            .model_membership
                            .tuple_source_key_column_device_reads,
                        4
                    );
                    assert_eq!(
                        batch.results[idx].model_membership.membership_source,
                        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                    );
                    vec![EpistemicAssumption::known_tuple(
                        "fact4",
                        vec![
                            EpistemicEvidenceTerm::integer(1),
                            EpistemicEvidenceTerm::integer(2),
                            EpistemicEvidenceTerm::integer(3),
                            EpistemicEvidenceTerm::integer(4),
                        ],
                        false,
                    )]
                }
                other => panic!("unexpected split quaternary possible/not-know indices: {other:?}"),
            },
        )
        .collect();
    let assumption_groups: Vec<&[EpistemicAssumption]> =
        assumption_groups_owned.iter().map(Vec::as_slice).collect();

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
    let mut solver =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = solver
        .new_workspace(unsat_cnf.var_cap, unsat_cnf.clause_cap)
        .expect("new split quaternary possible/not-know solver workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let report = solver
        .solve_assumption_lifecycle_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
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
        .expect(
            "accepted split quaternary possible/not-know GPU evidence must gate solver lifecycle",
        );

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.steps, 4);
    assert_eq!(report.assumption_pushes, 4);
    assert_eq!(report.assumption_retractions, 4);
    assert_eq!(report.workspace_reuses, 2);

    let solver_trace = solver.trace();
    assert_eq!(
        solver_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        solver_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(solver_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        solver_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        solver_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        solver_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        solver_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        solver_trace.accepted_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        solver_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(solver_trace.gpu_assumption_pushes, 4);
    assert_eq!(solver_trace.gpu_assumption_retractions, 4);
    assert_eq!(solver_trace.gpu_lifecycle_workspace_reuses, 2);
    assert_eq!(solver_trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(solver_trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(solver_trace.cpu_assignment_enumerations, 0);
    assert_eq!(solver_trace.cpu_maxsat_enumerations, 0);

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut prob = EpistemicProbProductionAdapter::new(config);
    let evaluated = prob
        .compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result(
            r#"
            0.8::fact4(2, 3, 4, 5).
            0.6::fact4(1, 2, 3, 4).
            query(fact4(2, 3, 4, 5)).
            query(fact4(1, 2, 3, 4)).
            "#,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect(
            "accepted split quaternary possible/not-know GPU evidence must condition source exact evidence",
        );

    assert_eq!(evaluated.len(), 2);
    for (component, evaluated_result) in split.components.iter().zip(evaluated.iter()) {
        let expected = match component.component.rule_indices.as_slice() {
            [0] => [1.0, 0.6],
            [1] => [0.8, 0.0],
            other => panic!("unexpected split quaternary possible/not-know indices: {other:?}"),
        };
        assert_eq!(evaluated_result.query_probs.len(), expected.len());
        for (actual, expected) in evaluated_result.query_probs.iter().zip(expected) {
            assert!(
                (actual.prob - expected).abs() < 1.0e-6,
                "conditioned split quaternary possible/not-know probability mismatch: actual={} expected={}",
                actual.prob,
                expected
            );
        }
    }

    let prob_trace = prob.trace();
    assert_eq!(prob_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(prob_trace.accepted_gpu_batch_component_evidence_consumed, 2);
    assert_eq!(prob_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(prob_trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(prob_trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(prob_trace.gpu_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(
        prob_trace.gpu_source_conditioned_nonzero_arity_evidence_facts,
        2
    );
    assert_eq!(
        prob_trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        0
    );
    assert_eq!(prob_trace.gpu_conditioned_max_evidence_arity, 4);
    assert_eq!(prob_trace.gpu_source_conditioned_max_evidence_arity, 4);
    assert_eq!(prob_trace.gpu_program_conditioned_max_evidence_arity, 0);
    assert_eq!(prob_trace.gpu_conditioned_negative_evidence_facts, 1);
    assert_eq!(prob_trace.gpu_conditioned_possible_evidence_facts, 1);
    assert_eq!(prob_trace.gpu_conditioned_not_known_evidence_facts, 1);
    assert_eq!(prob_trace.gpu_conditioned_know_evidence_facts, 0);
    assert_eq!(prob_trace.gpu_conditioned_not_possible_evidence_facts, 0);
    assert_eq!(prob_trace.gpu_source_conditioned_possible_evidence_facts, 1);
    assert_eq!(
        prob_trace.gpu_source_conditioned_not_known_evidence_facts,
        1
    );
    assert_eq!(prob_trace.gpu_exact_source_compiles, 2);
    assert_eq!(prob_trace.gpu_exact_program_compiles, 0);
    assert_eq!(prob_trace.gpu_exact_query_evaluations, 2);
    assert_eq!(prob_trace.gpu_source_exact_query_evaluations, 2);
    assert_eq!(prob_trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(
        prob_trace.gpu_source_knowledge_compilation_end_to_end_runs,
        2
    );
    assert_eq!(
        prob_trace.gpu_program_knowledge_compilation_end_to_end_runs,
        0
    );
    assert_eq!(prob_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(prob_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_quaternary_possible_and_not_know_batch_gates_probabilistic_gradient_pir_cnf_and_exact_evaluation_paths(
) {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred possible4(u32, u32, u32, u32).
        pred unknown4(u32, u32, u32, u32).
        possible4(A, B, C, D) :- tuple4(A, B, C, D), possible fact4(A, B, C, D).
        unknown4(A, B, C, D) :- tuple4(A, B, C, D), not know fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary possible/not-know probabilistic deep-reuse fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split quaternary possible/not-know probabilistic deep-reuse components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split quaternary possible/not-know probability components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split quaternary possible/not-know probability batch through GPU path");

    assert_eq!(batch.results.len(), 2);
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.know_operator_count, 0);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 0);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    let assumption_groups_owned: Vec<Vec<EpistemicAssumption>> = split
        .components
        .iter()
        .enumerate()
        .map(
            |(idx, component)| match component.component.rule_indices.as_slice() {
                [0] => {
                    assert_eq!(
                        download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                        vec![(2, 3, 4, 5)]
                    );
                    assert_eq!(
                        batch.results[idx]
                            .prepared
                            .preflight
                            .possible_operator_count,
                        1
                    );
                    assert_eq!(
                        batch.results[idx]
                            .model_membership
                            .tuple_source_key_column_device_reads,
                        4
                    );
                    assert_eq!(
                        batch.results[idx].model_membership.membership_source,
                        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                    );
                    vec![EpistemicAssumption::possible_tuple(
                        "fact4",
                        vec![
                            EpistemicEvidenceTerm::integer(2),
                            EpistemicEvidenceTerm::integer(3),
                            EpistemicEvidenceTerm::integer(4),
                            EpistemicEvidenceTerm::integer(5),
                        ],
                        true,
                    )]
                }
                [1] => {
                    assert_eq!(
                        download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                        vec![(1, 2, 3, 4), (9, 9, 9, 9)]
                    );
                    assert_eq!(
                        batch.results[idx]
                            .prepared
                            .preflight
                            .not_know_operator_count,
                        1
                    );
                    assert_eq!(
                        batch.results[idx]
                            .final_tuple_materialization
                            .negated_row_filter_count,
                        1
                    );
                    assert_eq!(
                        batch.results[idx]
                            .model_membership
                            .tuple_source_key_column_device_reads,
                        4
                    );
                    assert_eq!(
                        batch.results[idx].model_membership.membership_source,
                        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                    );
                    vec![EpistemicAssumption::known_tuple(
                        "fact4",
                        vec![
                            EpistemicEvidenceTerm::integer(1),
                            EpistemicEvidenceTerm::integer(2),
                            EpistemicEvidenceTerm::integer(3),
                            EpistemicEvidenceTerm::integer(4),
                        ],
                        false,
                    )]
                }
                other => {
                    panic!("unexpected split quaternary possible/not-know rule indices: {other:?}")
                }
            },
        )
        .collect();
    let assumption_groups: Vec<&[EpistemicAssumption]> =
        assumption_groups_owned.iter().map(Vec::as_slice).collect();

    let probabilistic_source = r#"
        0.8::fact4(2, 3, 4, 5).
        0.6::fact4(1, 2, 3, 4).
        query(fact4(2, 3, 4, 5)).
        query(fact4(1, 2, 3, 4)).
        "#;
    let parsed_probabilistic_program = parse_program(probabilistic_source)
        .expect("parse possible/not-know quaternary probabilistic program");
    let assert_conditioned = |rule_indices: &[usize], probs: &[f64]| {
        let expected = match rule_indices {
            [0] => [1.0, 0.6],
            [1] => [0.8, 0.0],
            other => {
                panic!("unexpected split quaternary possible/not-know rule indices: {other:?}")
            }
        };
        assert_eq!(probs.len(), expected.len());
        for (actual, expected) in probs.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "conditioned split quaternary possible/not-know probability mismatch: actual={actual} expected={expected}",
            );
        }
    };
    let assert_unconditioned = |probs: &[f64]| {
        let expected = [0.8, 0.6];
        assert_eq!(probs.len(), expected.len());
        for (actual, expected) in probs.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "unconditioned split quaternary possible/not-know probability mismatch: actual={actual} expected={expected}",
            );
        }
    };

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;

    let mut source_gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let source_gradients = source_gradient_adapter
        .compile_and_evaluate_conditioned_source_with_grads_for_gpu_batch_execution_result(
            probabilistic_source,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect(
            "accepted split quaternary possible/not-know batch must condition source gradients",
        );
    assert_eq!(source_gradients.len(), 2);
    for (component, gradient_result) in split.components.iter().zip(source_gradients.iter()) {
        let probs: Vec<f64> = gradient_result
            .query_grads
            .iter()
            .map(|query| query.prob)
            .collect();
        assert_conditioned(&component.component.rule_indices, &probs);
    }
    let source_gradient_trace = source_gradient_adapter.trace();
    assert_eq!(
        source_gradient_trace.accepted_gpu_batch_evidence_consumed,
        1
    );
    assert_eq!(
        source_gradient_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(
        source_gradient_trace.accepted_world_view_evidence_consumed,
        2
    );
    assert_eq!(
        source_gradient_trace.accepted_evidence_assumptions_consumed,
        2
    );
    assert_eq!(source_gradient_trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_evidence_facts,
        2
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_nonzero_arity_evidence_facts,
        2
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_max_evidence_arity,
        4
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_negative_evidence_facts,
        1
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_possible_evidence_facts,
        1
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_not_known_evidence_facts,
        1
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_know_evidence_facts,
        0
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_not_possible_evidence_facts,
        0
    );
    assert_eq!(source_gradient_trace.gpu_exact_source_compiles, 2);
    assert_eq!(source_gradient_trace.gpu_exact_program_compiles, 0);
    assert_eq!(source_gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(source_gradient_trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_gradient_evaluations,
        2
    );
    assert_eq!(
        source_gradient_trace.gpu_knowledge_compilation_end_to_end_runs,
        2
    );
    assert_eq!(
        source_gradient_trace.gpu_source_knowledge_compilation_end_to_end_runs,
        2
    );
    assert_eq!(source_gradient_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(source_gradient_trace.fixture_circuit_evaluations, 0);

    let mut program_gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let program_gradients = program_gradient_adapter
        .compile_and_evaluate_conditioned_program_with_grads_for_gpu_batch_execution_result(
            &parsed_probabilistic_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect(
            "accepted split quaternary possible/not-know batch must condition parsed-program gradients",
        );
    assert_eq!(program_gradients.len(), 2);
    for (component, gradient_result) in split.components.iter().zip(program_gradients.iter()) {
        let probs: Vec<f64> = gradient_result
            .query_grads
            .iter()
            .map(|query| query.prob)
            .collect();
        assert_conditioned(&component.component.rule_indices, &probs);
    }
    let program_gradient_trace = program_gradient_adapter.trace();
    assert_eq!(
        program_gradient_trace.accepted_gpu_batch_evidence_consumed,
        1
    );
    assert_eq!(
        program_gradient_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(
        program_gradient_trace.accepted_world_view_evidence_consumed,
        2
    );
    assert_eq!(
        program_gradient_trace.accepted_evidence_assumptions_consumed,
        2
    );
    assert_eq!(program_gradient_trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_evidence_facts,
        2
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        2
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_max_evidence_arity,
        4
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_negative_evidence_facts,
        1
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_possible_evidence_facts,
        1
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_not_known_evidence_facts,
        1
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_know_evidence_facts,
        0
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_not_possible_evidence_facts,
        0
    );
    assert_eq!(program_gradient_trace.gpu_exact_source_compiles, 0);
    assert_eq!(program_gradient_trace.gpu_exact_program_compiles, 2);
    assert_eq!(program_gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(program_gradient_trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_gradient_evaluations,
        2
    );
    assert_eq!(
        program_gradient_trace.gpu_knowledge_compilation_end_to_end_runs,
        2
    );
    assert_eq!(
        program_gradient_trace.gpu_program_knowledge_compilation_end_to_end_runs,
        2
    );
    assert_eq!(
        program_gradient_trace.cpu_only_probability_recomputations,
        0
    );
    assert_eq!(program_gradient_trace.fixture_circuit_evaluations, 0);

    let mut source_pir_adapter = EpistemicProbProductionAdapter::new(config);
    let source_pir_cnfs = source_pir_adapter
        .encode_source_pir_cnf_for_gpu_batch_execution_result(
            probabilistic_source,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect(
            "accepted split quaternary possible/not-know batch must gate source PIR/CNF encoding",
        );
    assert_eq!(source_pir_cnfs.len(), 2);
    for pir_cnf in &source_pir_cnfs {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let source_pir_trace = source_pir_adapter.trace();
    assert_eq!(source_pir_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        source_pir_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(source_pir_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(source_pir_trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(source_pir_trace.gpu_pir_graph_uploads, 2);
    assert_eq!(source_pir_trace.gpu_source_pir_graph_uploads, 2);
    assert_eq!(source_pir_trace.gpu_program_pir_graph_uploads, 0);
    assert_eq!(source_pir_trace.gpu_cnf_encodes, 2);
    assert_eq!(source_pir_trace.gpu_source_cnf_encodes, 2);
    assert_eq!(source_pir_trace.gpu_program_cnf_encodes, 0);
    assert_eq!(source_pir_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(source_pir_trace.fixture_circuit_evaluations, 0);

    let mut program_pir_adapter = EpistemicProbProductionAdapter::new(config);
    let program_pir_cnfs = program_pir_adapter
        .encode_program_pir_cnf_for_gpu_batch_execution_result(
            &parsed_probabilistic_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect(
            "accepted split quaternary possible/not-know batch must gate parsed-program PIR/CNF encoding",
        );
    assert_eq!(program_pir_cnfs.len(), 2);
    for pir_cnf in &program_pir_cnfs {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let program_pir_trace = program_pir_adapter.trace();
    assert_eq!(program_pir_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        program_pir_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(program_pir_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(program_pir_trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(program_pir_trace.gpu_pir_graph_uploads, 2);
    assert_eq!(program_pir_trace.gpu_source_pir_graph_uploads, 0);
    assert_eq!(program_pir_trace.gpu_program_pir_graph_uploads, 2);
    assert_eq!(program_pir_trace.gpu_cnf_encodes, 2);
    assert_eq!(program_pir_trace.gpu_source_cnf_encodes, 0);
    assert_eq!(program_pir_trace.gpu_program_cnf_encodes, 2);
    assert_eq!(program_pir_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(program_pir_trace.fixture_circuit_evaluations, 0);

    let mut query_adapter = EpistemicProbProductionAdapter::new(config);
    let exact = query_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &batch.results[0],
            assumption_groups_owned[0].clone(),
        )
        .expect("accepted split quaternary possible/not-know component must gate exact compile");
    let evaluated = query_adapter
        .evaluate_for_gpu_batch_execution_result(
            &exact,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect(
            "accepted split quaternary possible/not-know batch must gate exact query evaluation",
        );
    assert_eq!(evaluated.len(), 2);
    for result in &evaluated {
        let probs: Vec<f64> = result.query_probs.iter().map(|query| query.prob).collect();
        assert_unconditioned(&probs);
    }
    let query_trace = query_adapter.trace();
    assert_eq!(query_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        query_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(query_trace.accepted_world_view_evidence_consumed, 3);
    assert_eq!(query_trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(query_trace.gpu_exact_source_compiles, 1);
    assert_eq!(query_trace.gpu_exact_query_evaluations, 2);
    assert_eq!(query_trace.gpu_exact_gradient_evaluations, 0);
    assert_eq!(query_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(query_trace.fixture_circuit_evaluations, 0);

    let mut gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let exact = gradient_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &batch.results[0],
            assumption_groups_owned[0].clone(),
        )
        .expect("accepted split quaternary possible/not-know component must gate exact compile");
    let gradients = gradient_adapter
        .evaluate_gpu_with_grads_for_gpu_batch_execution_result(
            &exact,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect(
            "accepted split quaternary possible/not-know batch must gate exact gradient evaluation",
        );
    assert_eq!(gradients.len(), 2);
    for result in &gradients {
        let probs: Vec<f64> = result.query_grads.iter().map(|query| query.prob).collect();
        assert_unconditioned(&probs);
    }
    let gradient_trace = gradient_adapter.trace();
    assert_eq!(gradient_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        gradient_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(gradient_trace.accepted_world_view_evidence_consumed, 3);
    assert_eq!(gradient_trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(gradient_trace.gpu_exact_source_compiles, 1);
    assert_eq!(gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(gradient_trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(gradient_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(gradient_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_quaternary_not_possible_batch_gates_probabilistic_gradient_pir_cnf_and_exact_evaluation_paths(
) {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred known4(u32, u32, u32, u32).
        pred clear4(u32, u32, u32, u32).
        known4(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        clear4(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary probabilistic gradient/PIR fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split quaternary probabilistic gradient/PIR components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split quaternary probability components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split quaternary probability batch through GPU runtime path");

    assert_eq!(batch.results.len(), 2);
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);

    let assumption_groups_owned: Vec<Vec<EpistemicAssumption>> = split
        .components
        .iter()
        .enumerate()
        .map(
            |(idx, component)| match component.component.rule_indices.as_slice() {
                [0] => {
                    assert_eq!(
                        download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                        vec![(2, 3, 4, 5)]
                    );
                    assert_eq!(
                        batch.results[idx]
                            .model_membership
                            .tuple_source_key_column_device_reads,
                        4
                    );
                    vec![EpistemicAssumption::known_tuple(
                        "fact4",
                        vec![
                            EpistemicEvidenceTerm::integer(2),
                            EpistemicEvidenceTerm::integer(3),
                            EpistemicEvidenceTerm::integer(4),
                            EpistemicEvidenceTerm::integer(5),
                        ],
                        true,
                    )]
                }
                [1] => {
                    assert_eq!(
                        download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                        vec![(1, 2, 3, 4), (9, 9, 9, 9)]
                    );
                    assert_eq!(
                        batch.results[idx]
                            .prepared
                            .preflight
                            .not_possible_operator_count,
                        1
                    );
                    assert_eq!(
                        batch.results[idx]
                            .final_tuple_materialization
                            .negated_row_filter_count,
                        1
                    );
                    assert_eq!(
                        batch.results[idx]
                            .model_membership
                            .tuple_source_key_column_device_reads,
                        4
                    );
                    vec![EpistemicAssumption::possible_tuple(
                        "fact4",
                        vec![
                            EpistemicEvidenceTerm::integer(1),
                            EpistemicEvidenceTerm::integer(2),
                            EpistemicEvidenceTerm::integer(3),
                            EpistemicEvidenceTerm::integer(4),
                        ],
                        false,
                    )]
                }
                other => panic!("unexpected split quaternary rule indices: {other:?}"),
            },
        )
        .collect();
    let assumption_groups: Vec<&[EpistemicAssumption]> =
        assumption_groups_owned.iter().map(Vec::as_slice).collect();

    let probabilistic_source = r#"
        0.8::fact4(1, 2, 3, 4).
        0.6::fact4(2, 3, 4, 5).
        query(fact4(1, 2, 3, 4)).
        query(fact4(2, 3, 4, 5)).
        "#;
    let parsed_probabilistic_program =
        parse_program(probabilistic_source).expect("parse quaternary probabilistic program");
    let assert_conditioned = |rule_indices: &[usize], probs: &[f64]| {
        let expected = match rule_indices {
            [0] => [0.8, 1.0],
            [1] => [0.0, 0.6],
            other => panic!("unexpected split quaternary rule indices: {other:?}"),
        };
        assert_eq!(probs.len(), expected.len());
        for (actual, expected) in probs.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "conditioned split quaternary probability mismatch: actual={actual} expected={expected}",
            );
        }
    };
    let assert_unconditioned = |probs: &[f64]| {
        let expected = [0.8, 0.6];
        assert_eq!(probs.len(), expected.len());
        for (actual, expected) in probs.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "unconditioned split quaternary probability mismatch: actual={actual} expected={expected}",
            );
        }
    };

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;

    let mut source_gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let source_gradients = source_gradient_adapter
        .compile_and_evaluate_conditioned_source_with_grads_for_gpu_batch_execution_result(
            probabilistic_source,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split quaternary batch must condition source gradients");
    assert_eq!(source_gradients.len(), 2);
    for (component, gradient_result) in split.components.iter().zip(source_gradients.iter()) {
        let probs: Vec<f64> = gradient_result
            .query_grads
            .iter()
            .map(|query| query.prob)
            .collect();
        assert_conditioned(&component.component.rule_indices, &probs);
    }
    let source_gradient_trace = source_gradient_adapter.trace();
    assert_eq!(
        source_gradient_trace.accepted_gpu_batch_evidence_consumed,
        1
    );
    assert_eq!(
        source_gradient_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(
        source_gradient_trace.accepted_world_view_evidence_consumed,
        2
    );
    assert_eq!(
        source_gradient_trace.accepted_evidence_assumptions_consumed,
        2
    );
    assert_eq!(source_gradient_trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_evidence_facts,
        2
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_nonzero_arity_evidence_facts,
        2
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_max_evidence_arity,
        4
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_negative_evidence_facts,
        1
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_know_evidence_facts,
        1
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_not_possible_evidence_facts,
        1
    );
    assert_eq!(source_gradient_trace.gpu_exact_source_compiles, 2);
    assert_eq!(source_gradient_trace.gpu_exact_program_compiles, 0);
    assert_eq!(source_gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(source_gradient_trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_gradient_evaluations,
        2
    );
    assert_eq!(
        source_gradient_trace.gpu_knowledge_compilation_end_to_end_runs,
        2
    );
    assert_eq!(
        source_gradient_trace.gpu_source_knowledge_compilation_end_to_end_runs,
        2
    );
    assert_eq!(source_gradient_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(source_gradient_trace.fixture_circuit_evaluations, 0);

    let mut program_gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let program_gradients = program_gradient_adapter
        .compile_and_evaluate_conditioned_program_with_grads_for_gpu_batch_execution_result(
            &parsed_probabilistic_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split quaternary batch must condition parsed-program gradients");
    assert_eq!(program_gradients.len(), 2);
    for (component, gradient_result) in split.components.iter().zip(program_gradients.iter()) {
        let probs: Vec<f64> = gradient_result
            .query_grads
            .iter()
            .map(|query| query.prob)
            .collect();
        assert_conditioned(&component.component.rule_indices, &probs);
    }
    let program_gradient_trace = program_gradient_adapter.trace();
    assert_eq!(
        program_gradient_trace.accepted_gpu_batch_evidence_consumed,
        1
    );
    assert_eq!(
        program_gradient_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(
        program_gradient_trace.accepted_world_view_evidence_consumed,
        2
    );
    assert_eq!(
        program_gradient_trace.accepted_evidence_assumptions_consumed,
        2
    );
    assert_eq!(program_gradient_trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_evidence_facts,
        2
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        2
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_max_evidence_arity,
        4
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_negative_evidence_facts,
        1
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_know_evidence_facts,
        1
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_not_possible_evidence_facts,
        1
    );
    assert_eq!(program_gradient_trace.gpu_exact_source_compiles, 0);
    assert_eq!(program_gradient_trace.gpu_exact_program_compiles, 2);
    assert_eq!(program_gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(program_gradient_trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_gradient_evaluations,
        2
    );
    assert_eq!(
        program_gradient_trace.gpu_knowledge_compilation_end_to_end_runs,
        2
    );
    assert_eq!(
        program_gradient_trace.gpu_program_knowledge_compilation_end_to_end_runs,
        2
    );
    assert_eq!(
        program_gradient_trace.cpu_only_probability_recomputations,
        0
    );
    assert_eq!(program_gradient_trace.fixture_circuit_evaluations, 0);

    let mut source_pir_adapter = EpistemicProbProductionAdapter::new(config);
    let source_pir_cnfs = source_pir_adapter
        .encode_source_pir_cnf_for_gpu_batch_execution_result(
            probabilistic_source,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split quaternary batch must gate source PIR/CNF encoding");
    assert_eq!(source_pir_cnfs.len(), 2);
    for pir_cnf in &source_pir_cnfs {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let source_pir_trace = source_pir_adapter.trace();
    assert_eq!(source_pir_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        source_pir_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(source_pir_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(source_pir_trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(source_pir_trace.gpu_pir_graph_uploads, 2);
    assert_eq!(source_pir_trace.gpu_source_pir_graph_uploads, 2);
    assert_eq!(source_pir_trace.gpu_program_pir_graph_uploads, 0);
    assert_eq!(source_pir_trace.gpu_cnf_encodes, 2);
    assert_eq!(source_pir_trace.gpu_source_cnf_encodes, 2);
    assert_eq!(source_pir_trace.gpu_program_cnf_encodes, 0);
    assert_eq!(source_pir_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(source_pir_trace.fixture_circuit_evaluations, 0);

    let mut program_pir_adapter = EpistemicProbProductionAdapter::new(config);
    let program_pir_cnfs = program_pir_adapter
        .encode_program_pir_cnf_for_gpu_batch_execution_result(
            &parsed_probabilistic_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split quaternary batch must gate parsed-program PIR/CNF encoding");
    assert_eq!(program_pir_cnfs.len(), 2);
    for pir_cnf in &program_pir_cnfs {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let program_pir_trace = program_pir_adapter.trace();
    assert_eq!(program_pir_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        program_pir_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(program_pir_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(program_pir_trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(program_pir_trace.gpu_pir_graph_uploads, 2);
    assert_eq!(program_pir_trace.gpu_source_pir_graph_uploads, 0);
    assert_eq!(program_pir_trace.gpu_program_pir_graph_uploads, 2);
    assert_eq!(program_pir_trace.gpu_cnf_encodes, 2);
    assert_eq!(program_pir_trace.gpu_source_cnf_encodes, 0);
    assert_eq!(program_pir_trace.gpu_program_cnf_encodes, 2);
    assert_eq!(program_pir_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(program_pir_trace.fixture_circuit_evaluations, 0);

    let mut query_adapter = EpistemicProbProductionAdapter::new(config);
    let exact = query_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &batch.results[0],
            assumption_groups_owned[0].clone(),
        )
        .expect("accepted split quaternary component must gate probabilistic exact compile");
    let evaluated = query_adapter
        .evaluate_for_gpu_batch_execution_result(
            &exact,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split quaternary batch must gate exact query evaluation");
    assert_eq!(evaluated.len(), 2);
    for result in &evaluated {
        let probs: Vec<f64> = result.query_probs.iter().map(|query| query.prob).collect();
        assert_unconditioned(&probs);
    }
    let query_trace = query_adapter.trace();
    assert_eq!(query_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        query_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(query_trace.accepted_world_view_evidence_consumed, 3);
    assert_eq!(query_trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(query_trace.gpu_exact_source_compiles, 1);
    assert_eq!(query_trace.gpu_exact_query_evaluations, 2);
    assert_eq!(query_trace.gpu_exact_gradient_evaluations, 0);
    assert_eq!(query_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(query_trace.fixture_circuit_evaluations, 0);

    let mut gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let exact = gradient_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &batch.results[0],
            assumption_groups_owned[0].clone(),
        )
        .expect("accepted split quaternary component must gate probabilistic exact compile");
    let gradients = gradient_adapter
        .evaluate_gpu_with_grads_for_gpu_batch_execution_result(
            &exact,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split quaternary batch must gate exact gradient evaluation");
    assert_eq!(gradients.len(), 2);
    for result in &gradients {
        let probs: Vec<f64> = result.query_grads.iter().map(|query| query.prob).collect();
        assert_unconditioned(&probs);
    }
    let gradient_trace = gradient_adapter.trace();
    assert_eq!(gradient_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        gradient_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(gradient_trace.accepted_world_view_evidence_consumed, 3);
    assert_eq!(gradient_trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(gradient_trace.gpu_exact_source_compiles, 1);
    assert_eq!(gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(gradient_trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(gradient_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(gradient_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_all_binary_operator_batch_gates_probabilistic_program_and_gradient_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred alt(u32, u32).
        pred blocked(u32, u32).
        pred seen(u32, u32).
        pred known_edge(u32, u32).
        pred possible_alt(u32, u32).
        pred clear_pair(u32, u32).
        pred unknown_pair(u32, u32).
        known_edge(X, Y) :- pair(X, Y), know edge(X, Y).
        possible_alt(X, Y) :- pair(X, Y), possible alt(X, Y).
        clear_pair(X, Y) :- pair(X, Y), not possible blocked(X, Y).
        unknown_pair(X, Y) :- pair(X, Y), not know seen(X, Y).
        "#,
    )
    .expect("parse split all binary probabilistic program/gradient fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split all binary operator components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split all-operator components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "pair",
        upload_binary_u32(&fix.memory, &[(1, 2), (2, 3), (3, 4)]),
    );
    executor.put_relation("edge", upload_binary_u32(&fix.memory, &[(1, 2), (3, 4)]));
    executor.put_relation("alt", upload_binary_u32(&fix.memory, &[(2, 3)]));
    executor.put_relation("blocked", upload_binary_u32(&fix.memory, &[(3, 4)]));
    executor.put_relation("seen", upload_binary_u32(&fix.memory, &[(1, 2)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split all binary operator components through GPU batch path");

    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);

    let assumption_groups_owned: Vec<Vec<EpistemicAssumption>> = split
        .components
        .iter()
        .map(
            |component| match component.component.rule_indices.as_slice() {
                [0] => vec![EpistemicAssumption::known_tuple(
                    "edge",
                    vec![
                        EpistemicEvidenceTerm::integer(1),
                        EpistemicEvidenceTerm::integer(2),
                    ],
                    true,
                )],
                [1] => vec![EpistemicAssumption::possible_tuple(
                    "alt",
                    vec![
                        EpistemicEvidenceTerm::integer(2),
                        EpistemicEvidenceTerm::integer(3),
                    ],
                    true,
                )],
                [2] => vec![EpistemicAssumption::possible_tuple(
                    "blocked",
                    vec![
                        EpistemicEvidenceTerm::integer(1),
                        EpistemicEvidenceTerm::integer(2),
                    ],
                    false,
                )],
                [3] => vec![EpistemicAssumption::known_tuple(
                    "seen",
                    vec![
                        EpistemicEvidenceTerm::integer(2),
                        EpistemicEvidenceTerm::integer(3),
                    ],
                    false,
                )],
                other => panic!("unexpected split all-operator rule indices: {other:?}"),
            },
        )
        .collect();
    let assumption_groups: Vec<&[EpistemicAssumption]> =
        assumption_groups_owned.iter().map(Vec::as_slice).collect();

    let probabilistic_source = r#"
        0.4::edge(1, 2).
        0.5::alt(2, 3).
        0.7::blocked(1, 2).
        0.8::seen(2, 3).
        query(edge(1, 2)).
        query(alt(2, 3)).
        query(blocked(1, 2)).
        query(seen(2, 3)).
        "#;
    let parsed_probabilistic_program =
        parse_program(probabilistic_source).expect("parse all-binary probabilistic program");

    let assert_expected = |rule_indices: &[usize], probs: &[f64]| {
        let expected = match rule_indices {
            [0] => [1.0, 0.5, 0.7, 0.8],
            [1] => [0.4, 1.0, 0.7, 0.8],
            [2] => [0.4, 0.5, 0.0, 0.8],
            [3] => [0.4, 0.5, 0.7, 0.0],
            other => panic!("unexpected split all-operator rule indices: {other:?}"),
        };
        assert_eq!(probs.len(), expected.len());
        for (actual, expected) in probs.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "conditioned split all-operator probability mismatch: actual={actual} expected={expected}",
            );
        }
    };

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;

    let mut program_adapter = EpistemicProbProductionAdapter::new(config);
    let program_evaluated = program_adapter
        .compile_and_evaluate_conditioned_program_for_gpu_batch_execution_result(
            &parsed_probabilistic_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split all-operator batch must condition parsed-program exact evidence");
    assert_eq!(program_evaluated.len(), 4);
    for (component, evaluated_result) in split.components.iter().zip(program_evaluated.iter()) {
        let probs: Vec<f64> = evaluated_result
            .query_probs
            .iter()
            .map(|query| query.prob)
            .collect();
        assert_expected(&component.component.rule_indices, &probs);
    }
    let program_trace = program_adapter.trace();
    assert_eq!(program_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        program_trace.accepted_gpu_batch_component_evidence_consumed,
        4
    );
    assert_eq!(program_trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(program_trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(program_trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(
        program_trace.gpu_conditioned_nonzero_arity_evidence_facts,
        4
    );
    assert_eq!(program_trace.gpu_conditioned_max_evidence_arity, 2);
    assert_eq!(program_trace.gpu_conditioned_negative_evidence_facts, 2);
    assert_eq!(program_trace.gpu_conditioned_know_evidence_facts, 1);
    assert_eq!(program_trace.gpu_conditioned_possible_evidence_facts, 1);
    assert_eq!(program_trace.gpu_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(program_trace.gpu_conditioned_not_known_evidence_facts, 1);
    assert_eq!(program_trace.gpu_program_conditioned_evidence_facts, 4);
    assert_eq!(
        program_trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        4
    );
    assert_eq!(program_trace.gpu_program_conditioned_max_evidence_arity, 2);
    assert_eq!(
        program_trace.gpu_program_conditioned_negative_evidence_facts,
        2
    );
    assert_eq!(program_trace.gpu_program_conditioned_know_evidence_facts, 1);
    assert_eq!(
        program_trace.gpu_program_conditioned_possible_evidence_facts,
        1
    );
    assert_eq!(
        program_trace.gpu_program_conditioned_not_possible_evidence_facts,
        1
    );
    assert_eq!(
        program_trace.gpu_program_conditioned_not_known_evidence_facts,
        1
    );
    assert_eq!(program_trace.gpu_exact_source_compiles, 0);
    assert_eq!(program_trace.gpu_exact_program_compiles, 4);
    assert_eq!(program_trace.gpu_exact_query_evaluations, 4);
    assert_eq!(program_trace.gpu_source_exact_query_evaluations, 0);
    assert_eq!(program_trace.gpu_program_exact_query_evaluations, 4);
    assert_eq!(program_trace.gpu_knowledge_compilation_end_to_end_runs, 4);
    assert_eq!(
        program_trace.gpu_program_knowledge_compilation_end_to_end_runs,
        4
    );
    assert_eq!(program_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(program_trace.fixture_circuit_evaluations, 0);

    let mut source_gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let source_gradients = source_gradient_adapter
        .compile_and_evaluate_conditioned_source_with_grads_for_gpu_batch_execution_result(
            probabilistic_source,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split all-operator batch must condition source gradients");
    assert_eq!(source_gradients.len(), 4);
    for (component, gradient_result) in split.components.iter().zip(source_gradients.iter()) {
        let probs: Vec<f64> = gradient_result
            .query_grads
            .iter()
            .map(|query| query.prob)
            .collect();
        assert_expected(&component.component.rule_indices, &probs);
    }
    let source_gradient_trace = source_gradient_adapter.trace();
    assert_eq!(
        source_gradient_trace.accepted_gpu_batch_evidence_consumed,
        1
    );
    assert_eq!(
        source_gradient_trace.accepted_gpu_batch_component_evidence_consumed,
        4
    );
    assert_eq!(
        source_gradient_trace.accepted_world_view_evidence_consumed,
        4
    );
    assert_eq!(
        source_gradient_trace.accepted_evidence_assumptions_consumed,
        4
    );
    assert_eq!(source_gradient_trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_evidence_facts,
        4
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_nonzero_arity_evidence_facts,
        4
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_max_evidence_arity,
        2
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_negative_evidence_facts,
        2
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_know_evidence_facts,
        1
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_possible_evidence_facts,
        1
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_not_possible_evidence_facts,
        1
    );
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_not_known_evidence_facts,
        1
    );
    assert_eq!(source_gradient_trace.gpu_exact_source_compiles, 4);
    assert_eq!(source_gradient_trace.gpu_exact_program_compiles, 0);
    assert_eq!(source_gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(source_gradient_trace.gpu_exact_gradient_evaluations, 4);
    assert_eq!(
        source_gradient_trace.gpu_source_conditioned_gradient_evaluations,
        4
    );
    assert_eq!(
        source_gradient_trace.gpu_knowledge_compilation_end_to_end_runs,
        4
    );
    assert_eq!(
        source_gradient_trace.gpu_source_knowledge_compilation_end_to_end_runs,
        4
    );
    assert_eq!(source_gradient_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(source_gradient_trace.fixture_circuit_evaluations, 0);

    let mut program_gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let program_gradients = program_gradient_adapter
        .compile_and_evaluate_conditioned_program_with_grads_for_gpu_batch_execution_result(
            &parsed_probabilistic_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split all-operator batch must condition parsed-program gradients");
    assert_eq!(program_gradients.len(), 4);
    for (component, gradient_result) in split.components.iter().zip(program_gradients.iter()) {
        let probs: Vec<f64> = gradient_result
            .query_grads
            .iter()
            .map(|query| query.prob)
            .collect();
        assert_expected(&component.component.rule_indices, &probs);
    }
    let program_gradient_trace = program_gradient_adapter.trace();
    assert_eq!(
        program_gradient_trace.accepted_gpu_batch_evidence_consumed,
        1
    );
    assert_eq!(
        program_gradient_trace.accepted_gpu_batch_component_evidence_consumed,
        4
    );
    assert_eq!(
        program_gradient_trace.accepted_world_view_evidence_consumed,
        4
    );
    assert_eq!(
        program_gradient_trace.accepted_evidence_assumptions_consumed,
        4
    );
    assert_eq!(program_gradient_trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_evidence_facts,
        4
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        4
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_max_evidence_arity,
        2
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_negative_evidence_facts,
        2
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_know_evidence_facts,
        1
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_possible_evidence_facts,
        1
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_not_possible_evidence_facts,
        1
    );
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_not_known_evidence_facts,
        1
    );
    assert_eq!(program_gradient_trace.gpu_exact_source_compiles, 0);
    assert_eq!(program_gradient_trace.gpu_exact_program_compiles, 4);
    assert_eq!(program_gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(program_gradient_trace.gpu_exact_gradient_evaluations, 4);
    assert_eq!(
        program_gradient_trace.gpu_program_conditioned_gradient_evaluations,
        4
    );
    assert_eq!(
        program_gradient_trace.gpu_knowledge_compilation_end_to_end_runs,
        4
    );
    assert_eq!(
        program_gradient_trace.gpu_program_knowledge_compilation_end_to_end_runs,
        4
    );
    assert_eq!(
        program_gradient_trace.cpu_only_probability_recomputations,
        0
    );
    assert_eq!(program_gradient_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_all_binary_operator_batch_gates_probabilistic_pir_cnf_and_exact_evaluation_paths()
{
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred alt(u32, u32).
        pred blocked(u32, u32).
        pred seen(u32, u32).
        pred known_edge(u32, u32).
        pred possible_alt(u32, u32).
        pred clear_pair(u32, u32).
        pred unknown_pair(u32, u32).
        known_edge(X, Y) :- pair(X, Y), know edge(X, Y).
        possible_alt(X, Y) :- pair(X, Y), possible alt(X, Y).
        clear_pair(X, Y) :- pair(X, Y), not possible blocked(X, Y).
        unknown_pair(X, Y) :- pair(X, Y), not know seen(X, Y).
        "#,
    )
    .expect("parse split all binary probabilistic PIR/CNF fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split all binary operator components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split all-operator components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "pair",
        upload_binary_u32(&fix.memory, &[(1, 2), (2, 3), (3, 4)]),
    );
    executor.put_relation("edge", upload_binary_u32(&fix.memory, &[(1, 2), (3, 4)]));
    executor.put_relation("alt", upload_binary_u32(&fix.memory, &[(2, 3)]));
    executor.put_relation("blocked", upload_binary_u32(&fix.memory, &[(3, 4)]));
    executor.put_relation("seen", upload_binary_u32(&fix.memory, &[(1, 2)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split all binary operator components through GPU batch path");

    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);

    let assumption_groups_owned: Vec<Vec<EpistemicAssumption>> = split
        .components
        .iter()
        .map(
            |component| match component.component.rule_indices.as_slice() {
                [0] => vec![EpistemicAssumption::known_tuple(
                    "edge",
                    vec![
                        EpistemicEvidenceTerm::integer(1),
                        EpistemicEvidenceTerm::integer(2),
                    ],
                    true,
                )],
                [1] => vec![EpistemicAssumption::possible_tuple(
                    "alt",
                    vec![
                        EpistemicEvidenceTerm::integer(2),
                        EpistemicEvidenceTerm::integer(3),
                    ],
                    true,
                )],
                [2] => vec![EpistemicAssumption::possible_tuple(
                    "blocked",
                    vec![
                        EpistemicEvidenceTerm::integer(1),
                        EpistemicEvidenceTerm::integer(2),
                    ],
                    false,
                )],
                [3] => vec![EpistemicAssumption::known_tuple(
                    "seen",
                    vec![
                        EpistemicEvidenceTerm::integer(2),
                        EpistemicEvidenceTerm::integer(3),
                    ],
                    false,
                )],
                other => panic!("unexpected split all-operator rule indices: {other:?}"),
            },
        )
        .collect();
    let assumption_groups: Vec<&[EpistemicAssumption]> =
        assumption_groups_owned.iter().map(Vec::as_slice).collect();

    let probabilistic_source = r#"
        0.4::edge(1, 2).
        0.5::alt(2, 3).
        0.7::blocked(1, 2).
        0.8::seen(2, 3).
        query(edge(1, 2)).
        query(alt(2, 3)).
        query(blocked(1, 2)).
        query(seen(2, 3)).
        "#;
    let parsed_probabilistic_program =
        parse_program(probabilistic_source).expect("parse all-binary probabilistic PIR/CNF source");

    let assert_unconditioned = |probs: &[f64]| {
        let expected = [0.4, 0.5, 0.7, 0.8];
        assert_eq!(probs.len(), expected.len());
        for (actual, expected) in probs.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "unconditioned split all-operator probability mismatch: actual={actual} expected={expected}",
            );
        }
    };

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;

    let mut source_pir_adapter = EpistemicProbProductionAdapter::new(config);
    let source_pir_cnfs = source_pir_adapter
        .encode_source_pir_cnf_for_gpu_batch_execution_result(
            probabilistic_source,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split all-operator batch must gate source PIR/CNF encoding");
    assert_eq!(source_pir_cnfs.len(), 4);
    for pir_cnf in &source_pir_cnfs {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let source_pir_trace = source_pir_adapter.trace();
    assert_eq!(source_pir_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        source_pir_trace.accepted_gpu_batch_component_evidence_consumed,
        4
    );
    assert_eq!(source_pir_trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(source_pir_trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(source_pir_trace.gpu_pir_graph_uploads, 4);
    assert_eq!(source_pir_trace.gpu_source_pir_graph_uploads, 4);
    assert_eq!(source_pir_trace.gpu_program_pir_graph_uploads, 0);
    assert_eq!(source_pir_trace.gpu_cnf_encodes, 4);
    assert_eq!(source_pir_trace.gpu_source_cnf_encodes, 4);
    assert_eq!(source_pir_trace.gpu_program_cnf_encodes, 0);
    assert_eq!(source_pir_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(source_pir_trace.fixture_circuit_evaluations, 0);

    let mut program_pir_adapter = EpistemicProbProductionAdapter::new(config);
    let program_pir_cnfs = program_pir_adapter
        .encode_program_pir_cnf_for_gpu_batch_execution_result(
            &parsed_probabilistic_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split all-operator batch must gate parsed-program PIR/CNF encoding");
    assert_eq!(program_pir_cnfs.len(), 4);
    for pir_cnf in &program_pir_cnfs {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let program_pir_trace = program_pir_adapter.trace();
    assert_eq!(program_pir_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        program_pir_trace.accepted_gpu_batch_component_evidence_consumed,
        4
    );
    assert_eq!(program_pir_trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(program_pir_trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(program_pir_trace.gpu_pir_graph_uploads, 4);
    assert_eq!(program_pir_trace.gpu_source_pir_graph_uploads, 0);
    assert_eq!(program_pir_trace.gpu_program_pir_graph_uploads, 4);
    assert_eq!(program_pir_trace.gpu_cnf_encodes, 4);
    assert_eq!(program_pir_trace.gpu_source_cnf_encodes, 0);
    assert_eq!(program_pir_trace.gpu_program_cnf_encodes, 4);
    assert_eq!(program_pir_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(program_pir_trace.fixture_circuit_evaluations, 0);

    let mut query_adapter = EpistemicProbProductionAdapter::new(config);
    let exact = query_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &batch.results[0],
            assumption_groups_owned[0].clone(),
        )
        .expect("accepted split all-operator component must gate probabilistic exact compile");
    let evaluated = query_adapter
        .evaluate_for_gpu_batch_execution_result(
            &exact,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split all-operator batch must gate exact query evaluation");
    assert_eq!(evaluated.len(), 4);
    for result in &evaluated {
        let probs: Vec<f64> = result.query_probs.iter().map(|query| query.prob).collect();
        assert_unconditioned(&probs);
    }
    let query_trace = query_adapter.trace();
    assert_eq!(query_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        query_trace.accepted_gpu_batch_component_evidence_consumed,
        4
    );
    assert_eq!(query_trace.accepted_world_view_evidence_consumed, 5);
    assert_eq!(query_trace.accepted_evidence_assumptions_consumed, 5);
    assert_eq!(query_trace.gpu_exact_source_compiles, 1);
    assert_eq!(query_trace.gpu_exact_query_evaluations, 4);
    assert_eq!(query_trace.gpu_exact_gradient_evaluations, 0);
    assert_eq!(query_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(query_trace.fixture_circuit_evaluations, 0);

    let mut gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let exact = gradient_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &batch.results[0],
            assumption_groups_owned[0].clone(),
        )
        .expect("accepted split all-operator component must gate probabilistic exact compile");
    let gradients = gradient_adapter
        .evaluate_gpu_with_grads_for_gpu_batch_execution_result(
            &exact,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split all-operator batch must gate exact gradient evaluation");
    assert_eq!(gradients.len(), 4);
    for result in &gradients {
        let probs: Vec<f64> = result.query_grads.iter().map(|query| query.prob).collect();
        assert_unconditioned(&probs);
    }
    let gradient_trace = gradient_adapter.trace();
    assert_eq!(gradient_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        gradient_trace.accepted_gpu_batch_component_evidence_consumed,
        4
    );
    assert_eq!(gradient_trace.accepted_world_view_evidence_consumed, 5);
    assert_eq!(gradient_trace.accepted_evidence_assumptions_consumed, 5);
    assert_eq!(gradient_trace.gpu_exact_source_compiles, 1);
    assert_eq!(gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(gradient_trace.gpu_exact_gradient_evaluations, 4);
    assert_eq!(gradient_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(gradient_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_all_binary_operator_batch_gates_solver_lifecycle_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred alt(u32, u32).
        pred blocked(u32, u32).
        pred seen(u32, u32).
        pred known_edge(u32, u32).
        pred possible_alt(u32, u32).
        pred clear_pair(u32, u32).
        pred unknown_pair(u32, u32).
        known_edge(X, Y) :- pair(X, Y), know edge(X, Y).
        possible_alt(X, Y) :- pair(X, Y), possible alt(X, Y).
        clear_pair(X, Y) :- pair(X, Y), not possible blocked(X, Y).
        unknown_pair(X, Y) :- pair(X, Y), not know seen(X, Y).
        "#,
    )
    .expect("parse split all binary solver fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split all binary operator components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split all-operator components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "pair",
        upload_binary_u32(&fix.memory, &[(1, 2), (2, 3), (3, 4)]),
    );
    executor.put_relation("edge", upload_binary_u32(&fix.memory, &[(1, 2), (3, 4)]));
    executor.put_relation("alt", upload_binary_u32(&fix.memory, &[(2, 3)]));
    executor.put_relation("blocked", upload_binary_u32(&fix.memory, &[(3, 4)]));
    executor.put_relation("seen", upload_binary_u32(&fix.memory, &[(1, 2)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split all binary operator components through GPU batch path");

    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);

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
        .solve_assumption_lifecycle_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
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
        .expect("accepted all-operator split GPU batch evidence must gate solver lifecycle path");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(report.candidate_evidence_records, 4);
    assert_eq!(report.steps, 8);
    assert_eq!(report.assumption_pushes, 8);
    assert_eq!(report.assumption_retractions, 8);
    assert_eq!(report.workspace_reuses, 4);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        4
    );
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 4);
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_possible_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(trace.accepted_not_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_assumption_pushes, 8);
    assert_eq!(trace.gpu_assumption_retractions, 8);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 4);
    assert_eq!(trace.gpu_cdcl_sat_solves, 4);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 4);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_quaternary_not_possible_batch_gates_solver_lifecycle_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred known4(u32, u32, u32, u32).
        pred clear4(u32, u32, u32, u32).
        known4(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        clear4(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary solver fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split quaternary solver components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split quaternary solver components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split quaternary solver components through GPU batch path");

    assert_eq!(batch.results.len(), 2);
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);

    for (idx, component) in split.components.iter().enumerate() {
        match component.component.rule_indices.as_slice() {
            [0] => {
                assert_eq!(
                    download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                    vec![(2, 3, 4, 5)]
                );
                assert_eq!(
                    batch.results[idx]
                        .model_membership
                        .tuple_source_key_column_device_reads,
                    4
                );
                assert_eq!(
                    batch.results[idx].model_membership.membership_source,
                    EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                );
            }
            [1] => {
                assert_eq!(
                    download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                    vec![(1, 2, 3, 4), (9, 9, 9, 9)]
                );
                assert_eq!(
                    batch.results[idx]
                        .prepared
                        .preflight
                        .not_possible_operator_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .final_tuple_materialization
                        .negated_row_filter_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .model_membership
                        .tuple_source_key_column_device_reads,
                    4
                );
                assert_eq!(
                    batch.results[idx].model_membership.membership_source,
                    EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                );
            }
            other => panic!("unexpected split quaternary solver rule indices: {other:?}"),
        }
    }

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
        .expect("new split quaternary solver workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let report = adapter
        .solve_assumption_lifecycle_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
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
        .expect("accepted split quaternary GPU batch evidence must gate solver lifecycle path");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.steps, 4);
    assert_eq!(report.assumption_pushes, 4);
    assert_eq!(report.assumption_retractions, 4);
    assert_eq!(report.workspace_reuses, 2);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_possible_gpu_candidate_evidence_consumed, 0);
    assert_eq!(
        trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(trace.accepted_not_know_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.gpu_assumption_pushes, 4);
    assert_eq!(trace.gpu_assumption_retractions, 4);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 2);
    assert_eq!(trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_quaternary_all_operator_batch_gates_solver_lifecycle_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred edge4(u32, u32, u32, u32).
        pred alt4(u32, u32, u32, u32).
        pred blocked_fact4(u32, u32, u32, u32).
        pred hidden_fact4(u32, u32, u32, u32).
        pred known4(u32, u32, u32, u32).
        pred possible4(u32, u32, u32, u32).
        pred clear4(u32, u32, u32, u32).
        pred unknown4(u32, u32, u32, u32).
        known4(A, B, C, D) :- tuple4(A, B, C, D), know edge4(A, B, C, D).
        possible4(A, B, C, D) :- tuple4(A, B, C, D), possible alt4(A, B, C, D).
        clear4(A, B, C, D) :- tuple4(A, B, C, D), not possible blocked_fact4(A, B, C, D).
        unknown4(A, B, C, D) :- tuple4(A, B, C, D), not know hidden_fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary all-operator solver fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split quaternary all-operator solver components");

    assert_eq!(split.components.len(), 4);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1, 2, 3]);

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split quaternary all-operator solver components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(
            &fix.memory,
            &[(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6), (9, 9, 9, 9)],
        ),
    );
    executor.put_relation("edge4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));
    executor.put_relation("alt4", upload_quaternary_u32(&fix.memory, &[(3, 4, 5, 6)]));
    executor.put_relation(
        "blocked_fact4",
        upload_quaternary_u32(&fix.memory, &[(9, 9, 9, 9)]),
    );
    executor.put_relation(
        "hidden_fact4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4)]),
    );

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split quaternary all-operator solver components through GPU batch path");

    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 4);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());
    assert_eq!(batch.trace.aggregate_kernel_timing.cuda_event_pairs, 32);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);

    for (idx, component) in split.components.iter().enumerate() {
        let (expected_rows, expected_negated_filters, expected_counts) =
            match component.component.rule_indices.as_slice() {
                [0] => (vec![(2, 3, 4, 5)], 0, (1usize, 0usize, 0usize, 0usize)),
                [1] => (vec![(3, 4, 5, 6)], 0, (0usize, 1usize, 0usize, 0usize)),
                [2] => (
                    vec![(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)],
                    1,
                    (0usize, 0usize, 0usize, 1usize),
                ),
                [3] => (
                    vec![(2, 3, 4, 5), (3, 4, 5, 6), (9, 9, 9, 9)],
                    1,
                    (0usize, 0usize, 1usize, 0usize),
                ),
                other => panic!("unexpected split quaternary all-operator rule indices: {other:?}"),
            };
        assert_eq!(
            download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
            expected_rows
        );
        assert_eq!(
            batch.results[idx]
                .final_tuple_materialization
                .row_filter_count,
            1
        );
        assert_eq!(
            batch.results[idx]
                .final_tuple_materialization
                .negated_row_filter_count,
            expected_negated_filters
        );
        assert_eq!(
            batch.results[idx]
                .model_membership
                .tuple_source_key_column_device_reads,
            4
        );
        assert_eq!(
            batch.results[idx].model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            batch.results[idx].prepared.preflight.know_operator_count,
            expected_counts.0
        );
        assert_eq!(
            batch.results[idx]
                .prepared
                .preflight
                .possible_operator_count,
            expected_counts.1
        );
        assert_eq!(
            batch.results[idx]
                .prepared
                .preflight
                .not_know_operator_count,
            expected_counts.2
        );
        assert_eq!(
            batch.results[idx]
                .prepared
                .preflight
                .not_possible_operator_count,
            expected_counts.3
        );
    }

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
        .expect("new split quaternary all-operator solver workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let report = adapter
        .solve_assumption_lifecycle_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
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
        .expect(
            "accepted split quaternary all-operator GPU batch evidence must gate solver lifecycle path",
        );

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(report.candidate_evidence_records, 4);
    assert_eq!(report.steps, 8);
    assert_eq!(report.assumption_pushes, 8);
    assert_eq!(report.assumption_retractions, 8);
    assert_eq!(report.workspace_reuses, 4);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        4
    );
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 4);
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        4
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        16
    );
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_possible_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(trace.accepted_not_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_assumption_pushes, 8);
    assert_eq!(trace.gpu_assumption_retractions, 8);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 4);
    assert_eq!(trace.gpu_cdcl_sat_solves, 4);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 4);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_quaternary_all_operator_batch_gates_solver_reuse_and_maxsat_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (split, batch) = execute_split_quaternary_all_operator_batch(&fix);

    assert_eq!(split.components.len(), 4);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1, 2, 3]);
    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 4);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());
    assert_eq!(batch.trace.aggregate_kernel_timing.cuda_event_pairs, 32);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);

    for (idx, component) in split.components.iter().enumerate() {
        let (expected_rows, expected_negated_filters) =
            match component.component.rule_indices.as_slice() {
                [0] => (vec![(2, 3, 4, 5)], 0),
                [1] => (vec![(3, 4, 5, 6)], 0),
                [2] => (vec![(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)], 1),
                [3] => (vec![(2, 3, 4, 5), (3, 4, 5, 6), (9, 9, 9, 9)], 1),
                other => {
                    panic!("unexpected split quaternary all-operator reuse rule indices: {other:?}")
                }
            };
        assert_eq!(
            download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
            expected_rows
        );
        assert_eq!(
            batch.results[idx]
                .final_tuple_materialization
                .negated_row_filter_count,
            expected_negated_filters
        );
        assert_eq!(
            batch.results[idx]
                .model_membership
                .tuple_source_key_column_device_reads,
            4
        );
        assert_eq!(
            batch.results[idx].model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
    }

    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let source_cnf = GpuCnf::from_host(&unsat_instance, &fix.provider).expect("upload UNSAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut reuse_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = reuse_adapter
        .new_workspace(source_cnf.var_cap, source_cnf.clause_cap)
        .expect("new all-operator learned-clause reuse workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let reuse_report = reuse_adapter
        .solve_learned_clause_reuse_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &source_cnf,
            &branch_limit,
            &source_cnf,
            &branch_limit,
        )
        .expect("accepted all-operator split GPU batch evidence must gate learned-clause reuse");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(reuse_report.candidate_evidence_records, 4);
    assert_eq!(reuse_report.candidates, 8);
    assert_eq!(reuse_report.unsat_solves, 8);
    assert_eq!(reuse_report.gpu_learned_clause_arena_publications, 4);
    assert_eq!(reuse_report.gpu_learned_clause_imports, 4);
    assert_eq!(reuse_report.gpu_learned_clause_reused_solves, 4);
    assert_eq!(reuse_report.cpu_learned_clause_transfers, 0);

    let reuse_trace = reuse_adapter.trace();
    assert_eq!(
        reuse_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        reuse_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        4
    );
    assert_eq!(reuse_trace.accepted_gpu_candidate_evidence_consumed, 4);
    assert_eq!(
        reuse_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        4
    );
    assert_eq!(
        reuse_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        16
    );
    assert_eq!(reuse_trace.accepted_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        reuse_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        reuse_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        reuse_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(reuse_trace.gpu_cdcl_workspace_unsat_solves, 8);
    assert_eq!(reuse_trace.gpu_learned_clause_arena_publications, 4);
    assert_eq!(reuse_trace.gpu_learned_count_buffer_publications, 4);
    assert_eq!(reuse_trace.gpu_learned_clause_imports, 4);
    assert_eq!(reuse_trace.gpu_learned_clause_reused_solves, 4);
    assert_eq!(reuse_trace.cpu_learned_clause_transfers, 0);
    assert_eq!(reuse_trace.cpu_assignment_enumerations, 0);
    assert_eq!(reuse_trace.cpu_maxsat_enumerations, 0);

    let candidate_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let candidate_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let gpu_candidate_low =
        GpuCnf::from_host(&candidate_low, &fix.provider).expect("upload low-score MaxSAT CNF");
    let gpu_candidate_high =
        GpuCnf::from_host(&candidate_high, &fix.provider).expect("upload high-score MaxSAT CNF");
    let mut maxsat_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());

    let maxsat_report = maxsat_adapter
        .solve_weighted_maxsat_candidates_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &[
                GpuSolverProductionMaxSatCandidate {
                    score: 3,
                    cnf: &gpu_candidate_low,
                    branch_var_limit: &branch_limit,
                },
                GpuSolverProductionMaxSatCandidate {
                    score: 7,
                    cnf: &gpu_candidate_high,
                    branch_var_limit: &branch_limit,
                },
            ],
        )
        .expect("accepted all-operator split GPU batch evidence must gate MaxSAT through GPU CDCL");

    assert_eq!(maxsat_report.candidate_evidence_records, 4);
    assert_eq!(maxsat_report.optimum_score, 7);
    assert_eq!(maxsat_report.candidates_checked, 8);
    assert_eq!(maxsat_report.satisfiable_candidates, 8);
    assert_eq!(maxsat_report.unsat_candidates_pruned, 0);
    assert_eq!(maxsat_report.gpu_cdcl_candidate_encodes, 0);
    assert_eq!(maxsat_report.gpu_cdcl_candidate_solves, 8);

    let maxsat_trace = maxsat_adapter.trace();
    assert_eq!(
        maxsat_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        maxsat_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        4
    );
    assert_eq!(maxsat_trace.accepted_gpu_candidate_evidence_consumed, 4);
    assert_eq!(
        maxsat_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        4
    );
    assert_eq!(
        maxsat_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        16
    );
    assert_eq!(
        maxsat_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        maxsat_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        maxsat_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        maxsat_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(maxsat_trace.gpu_cdcl_sat_solves, 8);
    assert_eq!(maxsat_trace.gpu_maxsat_candidate_solves, 8);
    assert_eq!(maxsat_trace.gpu_maxsat_optima, 4);
    assert_eq!(maxsat_trace.cpu_assignment_enumerations, 0);
    assert_eq!(maxsat_trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_quaternary_all_operator_batch_gates_solver_search_scheduler_and_portfolio_paths()
{
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (_split, batch) = execute_split_quaternary_all_operator_batch(&fix);

    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    let sat_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let unsat = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let gpu_sat_low = GpuCnf::from_host(&sat_low, &fix.provider).expect("upload SAT low");
    let gpu_sat_high = GpuCnf::from_host(&sat_high, &fix.provider).expect("upload SAT high");
    let gpu_unsat = GpuCnf::from_host(&unsat, &fix.provider).expect("upload UNSAT candidate");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);

    let mut search_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut search_workspace = search_adapter
        .new_workspace(gpu_unsat.var_cap, gpu_unsat.clause_cap)
        .expect("new all-operator MaxSAT search workspace");
    let search_report = search_adapter
        .solve_weighted_maxsat_search_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut search_workspace,
            &[
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 7,
                    cnf: &gpu_sat_low,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                },
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 11,
                    cnf: &gpu_unsat,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                },
            ],
        )
        .expect("accepted all-operator split GPU batch evidence must gate MaxSAT search");

    assert_eq!(search_report.candidate_evidence_records, 4);
    assert_eq!(search_report.optimum_score, 7);
    assert_eq!(search_report.candidates_checked, 8);
    assert_eq!(search_report.satisfiable_candidates, 4);
    assert_eq!(search_report.unsat_candidates_pruned, 4);
    assert_eq!(search_report.gpu_cdcl_candidate_encodes, 0);
    assert_eq!(search_report.gpu_cdcl_candidate_solves, 8);

    let search_trace = search_adapter.trace();
    assert_eq!(
        search_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        4
    );
    assert_eq!(search_trace.accepted_gpu_candidate_evidence_consumed, 4);
    assert_eq!(
        search_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        4
    );
    assert_eq!(
        search_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        16
    );
    assert_eq!(
        search_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(search_trace.gpu_cdcl_sat_solves, 4);
    assert_eq!(search_trace.gpu_cdcl_workspace_unsat_solves, 4);
    assert_eq!(search_trace.gpu_maxsat_candidate_solves, 8);
    assert_eq!(search_trace.gpu_maxsat_unsat_candidate_prunes, 4);
    assert_eq!(search_trace.gpu_maxsat_optima, 4);
    assert_eq!(search_trace.cpu_assignment_enumerations, 0);
    assert_eq!(search_trace.cpu_maxsat_enumerations, 0);

    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![7.0, 9.0],
    );
    let both_soft_clauses = [0usize, 1usize];
    let sat_soft_clause = [0usize];
    let selections = [
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &both_soft_clauses,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &sat_soft_clause,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];

    let mut encoded_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut encoded_workspace = encoded_adapter
        .new_workspace(1, 2)
        .expect("new all-operator encoded MaxSAT workspace");
    let encoded_report = encoded_adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut encoded_workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("accepted all-operator split GPU batch evidence must gate encoded MaxSAT");

    assert_eq!(encoded_report.candidate_evidence_records, 4);
    assert_eq!(encoded_report.optimum_score, 7);
    assert_eq!(encoded_report.candidates_checked, 8);
    assert_eq!(encoded_report.satisfiable_candidates, 4);
    assert_eq!(encoded_report.unsat_candidates_pruned, 4);
    assert_eq!(encoded_report.gpu_cdcl_candidate_encodes, 8);
    assert_eq!(encoded_report.gpu_cdcl_candidate_solves, 8);

    let encoded_trace = encoded_adapter.trace();
    assert_eq!(
        encoded_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        4
    );
    assert_eq!(encoded_trace.accepted_gpu_candidate_evidence_consumed, 4);
    assert_eq!(
        encoded_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        4
    );
    assert_eq!(
        encoded_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        16
    );
    assert_eq!(
        encoded_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(encoded_trace.gpu_maxsat_candidate_encodes, 8);
    assert_eq!(encoded_trace.gpu_maxsat_candidate_solves, 8);
    assert_eq!(encoded_trace.gpu_maxsat_unsat_candidate_prunes, 4);
    assert_eq!(encoded_trace.gpu_maxsat_optima, 4);
    assert_eq!(encoded_trace.cpu_assignment_enumerations, 0);
    assert_eq!(encoded_trace.cpu_maxsat_enumerations, 0);

    let candidate_set = [
        GpuSolverProductionMaxSatCandidate {
            score: 3,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionMaxSatCandidate {
            score: 5,
            cnf: &gpu_sat_high,
            branch_var_limit: &branch_limit,
        },
    ];
    let search_candidates = [
        GpuSolverProductionMaxSatSearchCandidate {
            score: 9,
            cnf: &gpu_unsat,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionMaxSatSearchCandidate {
            score: 7,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];
    let jobs = [
        GpuSolverProductionMaxSatScheduleJob::CandidateSet {
            candidates: &candidate_set,
        },
        GpuSolverProductionMaxSatScheduleJob::Search {
            candidates: &search_candidates,
        },
        GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
            weighted: &weighted,
            branch_var_limit: &branch_limit,
            selections: &selections,
        },
        GpuSolverProductionMaxSatScheduleJob::Unknown {
            reason: "all-operator scheduler branch budget exhausted",
        },
        GpuSolverProductionMaxSatScheduleJob::Timeout { budget_micros: 25 },
    ];
    let mut scheduler_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut scheduler_workspace = scheduler_adapter
        .new_workspace(1, 2)
        .expect("new all-operator generalized MaxSAT scheduler workspace");
    let schedule_report = scheduler_adapter
        .solve_maxsat_schedule_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut scheduler_workspace,
            &jobs,
        )
        .expect("accepted all-operator split GPU batch evidence must gate MaxSAT scheduler");

    assert_eq!(schedule_report.candidate_evidence_records, 4);
    assert_eq!(schedule_report.jobs, 20);
    assert_eq!(schedule_report.candidate_set_jobs, 4);
    assert_eq!(schedule_report.search_jobs, 4);
    assert_eq!(schedule_report.encoded_search_jobs, 4);
    assert_eq!(schedule_report.unknown_jobs, 4);
    assert_eq!(schedule_report.timeout_jobs, 4);
    assert_eq!(schedule_report.optimum_score, 7);
    assert_eq!(schedule_report.candidates_checked, 24);
    assert_eq!(schedule_report.satisfiable_candidates, 16);
    assert_eq!(schedule_report.unsat_candidates_pruned, 8);
    assert_eq!(schedule_report.gpu_cdcl_candidate_encodes, 8);
    assert_eq!(schedule_report.gpu_cdcl_candidate_solves, 24);

    let scheduler_trace = scheduler_adapter.trace();
    assert_eq!(
        scheduler_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        4
    );
    assert_eq!(scheduler_trace.accepted_gpu_candidate_evidence_consumed, 4);
    assert_eq!(
        scheduler_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        4
    );
    assert_eq!(
        scheduler_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        16
    );
    assert_eq!(
        scheduler_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_jobs, 20);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_candidate_set_jobs, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_search_jobs, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_encoded_search_jobs, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_unknown_status_jobs, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_timeout_status_jobs, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_encodes, 8);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_solves, 24);
    assert_eq!(scheduler_trace.gpu_maxsat_unsat_candidate_prunes, 8);
    assert_eq!(scheduler_trace.gpu_maxsat_optima, 12);
    assert_eq!(scheduler_trace.cpu_assignment_enumerations, 0);
    assert_eq!(scheduler_trace.cpu_maxsat_enumerations, 0);

    let portfolio_sat =
        GpuCnf::from_host(&sat_low, &fix.provider).expect("upload portfolio SAT CNF");
    let portfolio_maxsat_candidates = [GpuSolverProductionMaxSatCandidate {
        score: 5,
        cnf: &gpu_sat_high,
        branch_var_limit: &branch_limit,
    }];
    let portfolio_jobs = [
        GpuSolverProductionPortfolioJob::Sat {
            cnf: &portfolio_sat,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionPortfolioJob::MaxSat {
            candidates: &portfolio_maxsat_candidates,
        },
        GpuSolverProductionPortfolioJob::Unknown {
            reason: "all-operator portfolio branch budget exhausted",
        },
        GpuSolverProductionPortfolioJob::Timeout { budget_micros: 10 },
    ];
    let mut portfolio_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let portfolio_report = portfolio_adapter
        .solve_portfolio_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &portfolio_jobs,
        )
        .expect("accepted all-operator split GPU batch evidence must gate portfolio path");

    assert_eq!(portfolio_report.candidate_evidence_records, 4);
    assert_eq!(portfolio_report.jobs, 16);
    assert_eq!(portfolio_report.sat_jobs, 4);
    assert_eq!(portfolio_report.maxsat_jobs, 4);
    assert_eq!(portfolio_report.unknown_jobs, 4);
    assert_eq!(portfolio_report.timeout_jobs, 4);
    assert_eq!(portfolio_report.maxsat_optimum_scores, 20);

    let portfolio_trace = portfolio_adapter.trace();
    assert_eq!(
        portfolio_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        portfolio_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        4
    );
    assert_eq!(portfolio_trace.accepted_gpu_candidate_evidence_consumed, 4);
    assert_eq!(
        portfolio_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        4
    );
    assert_eq!(
        portfolio_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        16
    );
    assert_eq!(
        portfolio_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        portfolio_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        portfolio_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        portfolio_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(portfolio_trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(portfolio_trace.gpu_maxsat_optima, 4);
    assert_eq!(portfolio_trace.gpu_portfolio_jobs, 16);
    assert_eq!(portfolio_trace.gpu_portfolio_sat_jobs, 4);
    assert_eq!(portfolio_trace.gpu_portfolio_maxsat_jobs, 4);
    assert_eq!(portfolio_trace.gpu_portfolio_unknown_status_jobs, 4);
    assert_eq!(portfolio_trace.gpu_portfolio_timeout_status_jobs, 4);
    assert_eq!(portfolio_trace.cpu_assignment_enumerations, 0);
    assert_eq!(portfolio_trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_quaternary_not_possible_batch_gates_solver_reuse_and_maxsat_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred known4(u32, u32, u32, u32).
        pred clear4(u32, u32, u32, u32).
        known4(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        clear4(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary solver reuse fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split quaternary solver reuse components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split quaternary solver reuse components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split quaternary solver reuse components through GPU batch path");

    assert_eq!(batch.results.len(), 2);
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);

    for (idx, component) in split.components.iter().enumerate() {
        match component.component.rule_indices.as_slice() {
            [0] => {
                assert_eq!(
                    download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                    vec![(2, 3, 4, 5)]
                );
                assert_eq!(
                    batch.results[idx]
                        .model_membership
                        .tuple_source_key_column_device_reads,
                    4
                );
                assert_eq!(
                    batch.results[idx].model_membership.membership_source,
                    EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                );
            }
            [1] => {
                assert_eq!(
                    download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                    vec![(1, 2, 3, 4), (9, 9, 9, 9)]
                );
                assert_eq!(
                    batch.results[idx]
                        .prepared
                        .preflight
                        .not_possible_operator_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .final_tuple_materialization
                        .negated_row_filter_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .model_membership
                        .tuple_source_key_column_device_reads,
                    4
                );
                assert_eq!(
                    batch.results[idx].model_membership.membership_source,
                    EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                );
            }
            other => panic!("unexpected split quaternary solver reuse rule indices: {other:?}"),
        }
    }

    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let source_cnf = GpuCnf::from_host(&unsat_instance, &fix.provider).expect("upload UNSAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut reuse_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = reuse_adapter
        .new_workspace(source_cnf.var_cap, source_cnf.clause_cap)
        .expect("new quaternary learned-clause reuse workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let reuse_report = reuse_adapter
        .solve_learned_clause_reuse_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &source_cnf,
            &branch_limit,
            &source_cnf,
            &branch_limit,
        )
        .expect("accepted quaternary split GPU batch evidence must gate learned-clause reuse");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(reuse_report.candidate_evidence_records, 2);
    assert_eq!(reuse_report.candidates, 4);
    assert_eq!(reuse_report.unsat_solves, 4);
    assert_eq!(reuse_report.gpu_learned_clause_arena_publications, 2);
    assert_eq!(reuse_report.gpu_learned_clause_imports, 2);
    assert_eq!(reuse_report.gpu_learned_clause_reused_solves, 2);
    assert_eq!(reuse_report.cpu_learned_clause_transfers, 0);

    let reuse_trace = reuse_adapter.trace();
    assert_eq!(
        reuse_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        reuse_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(reuse_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        reuse_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        reuse_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(reuse_trace.accepted_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        reuse_trace.accepted_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        reuse_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        reuse_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(reuse_trace.gpu_cdcl_workspace_unsat_solves, 4);
    assert_eq!(reuse_trace.gpu_learned_clause_arena_publications, 2);
    assert_eq!(reuse_trace.gpu_learned_count_buffer_publications, 2);
    assert_eq!(reuse_trace.gpu_learned_clause_imports, 2);
    assert_eq!(reuse_trace.gpu_learned_clause_reused_solves, 2);
    assert_eq!(reuse_trace.cpu_learned_clause_transfers, 0);
    assert_eq!(reuse_trace.cpu_assignment_enumerations, 0);
    assert_eq!(reuse_trace.cpu_maxsat_enumerations, 0);

    let candidate_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let candidate_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let gpu_candidate_low =
        GpuCnf::from_host(&candidate_low, &fix.provider).expect("upload low-score MaxSAT CNF");
    let gpu_candidate_high =
        GpuCnf::from_host(&candidate_high, &fix.provider).expect("upload high-score MaxSAT CNF");
    let mut maxsat_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());

    let maxsat_report = maxsat_adapter
        .solve_weighted_maxsat_candidates_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &[
                GpuSolverProductionMaxSatCandidate {
                    score: 3,
                    cnf: &gpu_candidate_low,
                    branch_var_limit: &branch_limit,
                },
                GpuSolverProductionMaxSatCandidate {
                    score: 7,
                    cnf: &gpu_candidate_high,
                    branch_var_limit: &branch_limit,
                },
            ],
        )
        .expect("accepted quaternary split GPU batch evidence must gate MaxSAT through GPU CDCL");

    assert_eq!(maxsat_report.candidate_evidence_records, 2);
    assert_eq!(maxsat_report.optimum_score, 7);
    assert_eq!(maxsat_report.candidates_checked, 4);
    assert_eq!(maxsat_report.satisfiable_candidates, 4);
    assert_eq!(maxsat_report.unsat_candidates_pruned, 0);
    assert_eq!(maxsat_report.gpu_cdcl_candidate_encodes, 0);
    assert_eq!(maxsat_report.gpu_cdcl_candidate_solves, 4);

    let maxsat_trace = maxsat_adapter.trace();
    assert_eq!(
        maxsat_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        maxsat_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(maxsat_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        maxsat_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        maxsat_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        maxsat_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        maxsat_trace.accepted_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        maxsat_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        maxsat_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(maxsat_trace.gpu_cdcl_sat_solves, 4);
    assert_eq!(maxsat_trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(maxsat_trace.gpu_maxsat_optima, 2);
    assert_eq!(maxsat_trace.cpu_assignment_enumerations, 0);
    assert_eq!(maxsat_trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_quaternary_not_possible_batch_gates_solver_search_scheduler_and_portfolio_paths()
{
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred known4(u32, u32, u32, u32).
        pred clear4(u32, u32, u32, u32).
        known4(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        clear4(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary not-possible solver search fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split quaternary not-possible solver search components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split quaternary not-possible solver search components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split quaternary not-possible solver search through GPU batch path");

    assert_eq!(batch.results.len(), 2);
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 0);
    assert_eq!(batch.trace.not_know_operator_count, 0);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    for (idx, component) in split.components.iter().enumerate() {
        match component.component.rule_indices.as_slice() {
            [0] => {
                assert_eq!(
                    download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                    vec![(2, 3, 4, 5)]
                );
                assert_eq!(batch.results[idx].prepared.preflight.know_operator_count, 1);
                assert_eq!(
                    batch.results[idx]
                        .model_membership
                        .tuple_source_key_column_device_reads,
                    4
                );
                assert_eq!(
                    batch.results[idx].model_membership.membership_source,
                    EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                );
            }
            [1] => {
                assert_eq!(
                    download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                    vec![(1, 2, 3, 4), (9, 9, 9, 9)]
                );
                assert_eq!(
                    batch.results[idx]
                        .prepared
                        .preflight
                        .not_possible_operator_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .final_tuple_materialization
                        .negated_row_filter_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .model_membership
                        .tuple_source_key_column_device_reads,
                    4
                );
                assert_eq!(
                    batch.results[idx].model_membership.membership_source,
                    EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                );
            }
            other => {
                panic!("unexpected split quaternary not-possible solver search rule indices: {other:?}")
            }
        }
    }

    let sat_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let unsat = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let gpu_sat_low = GpuCnf::from_host(&sat_low, &fix.provider).expect("upload SAT low");
    let gpu_sat_high = GpuCnf::from_host(&sat_high, &fix.provider).expect("upload SAT high");
    let gpu_unsat = GpuCnf::from_host(&unsat, &fix.provider).expect("upload UNSAT candidate");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);

    let mut search_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut search_workspace = search_adapter
        .new_workspace(gpu_unsat.var_cap, gpu_unsat.clause_cap)
        .expect("new not-possible MaxSAT search workspace");
    let search_report = search_adapter
        .solve_weighted_maxsat_search_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut search_workspace,
            &[
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 7,
                    cnf: &gpu_sat_low,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                },
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 11,
                    cnf: &gpu_unsat,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                },
            ],
        )
        .expect("accepted not-possible split GPU batch evidence must gate MaxSAT search");

    assert_eq!(search_report.candidate_evidence_records, 2);
    assert_eq!(search_report.optimum_score, 7);
    assert_eq!(search_report.candidates_checked, 4);
    assert_eq!(search_report.satisfiable_candidates, 2);
    assert_eq!(search_report.unsat_candidates_pruned, 2);
    assert_eq!(search_report.gpu_cdcl_candidate_solves, 4);

    let search_trace = search_adapter.trace();
    assert_eq!(
        search_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(search_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        search_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        search_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        search_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        search_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(search_trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(search_trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(search_trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(search_trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(search_trace.gpu_maxsat_optima, 2);
    assert_eq!(search_trace.cpu_assignment_enumerations, 0);
    assert_eq!(search_trace.cpu_maxsat_enumerations, 0);

    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![7.0, 9.0],
    );
    let both_soft_clauses = [0usize, 1usize];
    let sat_soft_clause = [0usize];
    let selections = [
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &both_soft_clauses,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &sat_soft_clause,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];

    let mut encoded_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut encoded_workspace = encoded_adapter
        .new_workspace(1, 2)
        .expect("new not-possible encoded MaxSAT workspace");
    let encoded_report = encoded_adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut encoded_workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("accepted not-possible split GPU batch evidence must gate encoded MaxSAT");

    assert_eq!(encoded_report.candidate_evidence_records, 2);
    assert_eq!(encoded_report.optimum_score, 7);
    assert_eq!(encoded_report.candidates_checked, 4);
    assert_eq!(encoded_report.satisfiable_candidates, 2);
    assert_eq!(encoded_report.unsat_candidates_pruned, 2);
    assert_eq!(encoded_report.gpu_cdcl_candidate_encodes, 4);
    assert_eq!(encoded_report.gpu_cdcl_candidate_solves, 4);

    let encoded_trace = encoded_adapter.trace();
    assert_eq!(
        encoded_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(encoded_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        encoded_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        encoded_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        encoded_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(encoded_trace.gpu_maxsat_candidate_encodes, 4);
    assert_eq!(encoded_trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(encoded_trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(encoded_trace.gpu_maxsat_optima, 2);
    assert_eq!(encoded_trace.cpu_assignment_enumerations, 0);
    assert_eq!(encoded_trace.cpu_maxsat_enumerations, 0);

    let candidate_set = [
        GpuSolverProductionMaxSatCandidate {
            score: 3,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionMaxSatCandidate {
            score: 5,
            cnf: &gpu_sat_high,
            branch_var_limit: &branch_limit,
        },
    ];
    let search_candidates = [
        GpuSolverProductionMaxSatSearchCandidate {
            score: 9,
            cnf: &gpu_unsat,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionMaxSatSearchCandidate {
            score: 7,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];
    let jobs = [
        GpuSolverProductionMaxSatScheduleJob::CandidateSet {
            candidates: &candidate_set,
        },
        GpuSolverProductionMaxSatScheduleJob::Search {
            candidates: &search_candidates,
        },
        GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
            weighted: &weighted,
            branch_var_limit: &branch_limit,
            selections: &selections,
        },
        GpuSolverProductionMaxSatScheduleJob::Unknown {
            reason: "not-possible scheduler branch budget exhausted",
        },
        GpuSolverProductionMaxSatScheduleJob::Timeout { budget_micros: 25 },
    ];
    let mut scheduler_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut scheduler_workspace = scheduler_adapter
        .new_workspace(1, 2)
        .expect("new not-possible generalized MaxSAT scheduler workspace");
    let schedule_report = scheduler_adapter
        .solve_maxsat_schedule_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut scheduler_workspace,
            &jobs,
        )
        .expect("accepted not-possible split GPU batch evidence must gate MaxSAT scheduler");

    assert_eq!(schedule_report.candidate_evidence_records, 2);
    assert_eq!(schedule_report.jobs, 10);
    assert_eq!(schedule_report.candidate_set_jobs, 2);
    assert_eq!(schedule_report.search_jobs, 2);
    assert_eq!(schedule_report.encoded_search_jobs, 2);
    assert_eq!(schedule_report.unknown_jobs, 2);
    assert_eq!(schedule_report.timeout_jobs, 2);
    assert_eq!(schedule_report.optimum_score, 7);
    assert_eq!(schedule_report.candidates_checked, 12);
    assert_eq!(schedule_report.satisfiable_candidates, 8);
    assert_eq!(schedule_report.unsat_candidates_pruned, 4);
    assert_eq!(schedule_report.gpu_cdcl_candidate_encodes, 4);
    assert_eq!(schedule_report.gpu_cdcl_candidate_solves, 12);

    let scheduler_trace = scheduler_adapter.trace();
    assert_eq!(
        scheduler_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(scheduler_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        scheduler_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        scheduler_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        scheduler_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_jobs, 10);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_candidate_set_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_search_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_encoded_search_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_unknown_status_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_timeout_status_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_encodes, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_solves, 12);
    assert_eq!(scheduler_trace.gpu_maxsat_unsat_candidate_prunes, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_optima, 6);
    assert_eq!(scheduler_trace.cpu_assignment_enumerations, 0);
    assert_eq!(scheduler_trace.cpu_maxsat_enumerations, 0);

    let portfolio_sat =
        GpuCnf::from_host(&sat_low, &fix.provider).expect("upload portfolio SAT CNF");
    let portfolio_maxsat_candidates = [GpuSolverProductionMaxSatCandidate {
        score: 5,
        cnf: &gpu_sat_high,
        branch_var_limit: &branch_limit,
    }];
    let portfolio_jobs = [
        GpuSolverProductionPortfolioJob::Sat {
            cnf: &portfolio_sat,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionPortfolioJob::MaxSat {
            candidates: &portfolio_maxsat_candidates,
        },
        GpuSolverProductionPortfolioJob::Unknown {
            reason: "not-possible portfolio branch budget exhausted",
        },
        GpuSolverProductionPortfolioJob::Timeout { budget_micros: 10 },
    ];
    let mut portfolio_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let portfolio_report = portfolio_adapter
        .solve_portfolio_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &portfolio_jobs,
        )
        .expect("accepted not-possible split GPU batch evidence must gate portfolio path");

    assert_eq!(portfolio_report.candidate_evidence_records, 2);
    assert_eq!(portfolio_report.jobs, 8);
    assert_eq!(portfolio_report.sat_jobs, 2);
    assert_eq!(portfolio_report.maxsat_jobs, 2);
    assert_eq!(portfolio_report.unknown_jobs, 2);
    assert_eq!(portfolio_report.timeout_jobs, 2);
    assert_eq!(portfolio_report.maxsat_optimum_scores, 10);

    let portfolio_trace = portfolio_adapter.trace();
    assert_eq!(
        portfolio_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        portfolio_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(portfolio_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        portfolio_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        portfolio_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        portfolio_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        portfolio_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(portfolio_trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(portfolio_trace.gpu_maxsat_optima, 2);
    assert_eq!(portfolio_trace.gpu_portfolio_jobs, 8);
    assert_eq!(portfolio_trace.gpu_portfolio_sat_jobs, 2);
    assert_eq!(portfolio_trace.gpu_portfolio_maxsat_jobs, 2);
    assert_eq!(portfolio_trace.gpu_portfolio_unknown_status_jobs, 2);
    assert_eq!(portfolio_trace.gpu_portfolio_timeout_status_jobs, 2);
    assert_eq!(portfolio_trace.cpu_assignment_enumerations, 0);
    assert_eq!(portfolio_trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_quaternary_possible_and_not_know_batch_gates_solver_reuse_and_maxsat_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred possible4(u32, u32, u32, u32).
        pred unknown4(u32, u32, u32, u32).
        possible4(A, B, C, D) :- tuple4(A, B, C, D), possible fact4(A, B, C, D).
        unknown4(A, B, C, D) :- tuple4(A, B, C, D), not know fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary possible/not-know solver reuse fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split quaternary possible/not-know solver reuse components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split quaternary possible/not-know solver reuse components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split quaternary possible/not-know solver reuse through GPU batch path");

    assert_eq!(batch.results.len(), 2);
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.know_operator_count, 0);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 0);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    for (idx, component) in split.components.iter().enumerate() {
        match component.component.rule_indices.as_slice() {
            [0] => {
                assert_eq!(
                    download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                    vec![(2, 3, 4, 5)]
                );
                assert_eq!(
                    batch.results[idx]
                        .prepared
                        .preflight
                        .possible_operator_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .final_tuple_materialization
                        .negated_row_filter_count,
                    0
                );
                assert_eq!(
                    batch.results[idx]
                        .model_membership
                        .tuple_source_key_column_device_reads,
                    4
                );
                assert_eq!(
                    batch.results[idx].model_membership.membership_source,
                    EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                );
            }
            [1] => {
                assert_eq!(
                    download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                    vec![(1, 2, 3, 4), (9, 9, 9, 9)]
                );
                assert_eq!(
                    batch.results[idx]
                        .prepared
                        .preflight
                        .not_know_operator_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .final_tuple_materialization
                        .negated_row_filter_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .model_membership
                        .tuple_source_key_column_device_reads,
                    4
                );
                assert_eq!(
                    batch.results[idx].model_membership.membership_source,
                    EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                );
            }
            other => {
                panic!("unexpected split quaternary possible/not-know solver reuse rule indices: {other:?}")
            }
        }
    }

    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let source_cnf = GpuCnf::from_host(&unsat_instance, &fix.provider).expect("upload UNSAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut reuse_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = reuse_adapter
        .new_workspace(source_cnf.var_cap, source_cnf.clause_cap)
        .expect("new possible/not-know learned-clause reuse workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let reuse_report = reuse_adapter
        .solve_learned_clause_reuse_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &source_cnf,
            &branch_limit,
            &source_cnf,
            &branch_limit,
        )
        .expect(
            "accepted possible/not-know split GPU batch evidence must gate learned-clause reuse",
        );

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(reuse_report.candidate_evidence_records, 2);
    assert_eq!(reuse_report.candidates, 4);
    assert_eq!(reuse_report.unsat_solves, 4);
    assert_eq!(reuse_report.gpu_learned_clause_arena_publications, 2);
    assert_eq!(reuse_report.gpu_learned_clause_imports, 2);
    assert_eq!(reuse_report.gpu_learned_clause_reused_solves, 2);
    assert_eq!(reuse_report.cpu_learned_clause_transfers, 0);

    let reuse_trace = reuse_adapter.trace();
    assert_eq!(
        reuse_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        reuse_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(reuse_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        reuse_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        reuse_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(reuse_trace.accepted_know_gpu_candidate_evidence_consumed, 0);
    assert_eq!(
        reuse_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        reuse_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        reuse_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(reuse_trace.gpu_cdcl_workspace_unsat_solves, 4);
    assert_eq!(reuse_trace.gpu_learned_clause_arena_publications, 2);
    assert_eq!(reuse_trace.gpu_learned_count_buffer_publications, 2);
    assert_eq!(reuse_trace.gpu_learned_clause_imports, 2);
    assert_eq!(reuse_trace.gpu_learned_clause_reused_solves, 2);
    assert_eq!(reuse_trace.cpu_learned_clause_transfers, 0);
    assert_eq!(reuse_trace.cpu_assignment_enumerations, 0);
    assert_eq!(reuse_trace.cpu_maxsat_enumerations, 0);

    let candidate_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let candidate_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let gpu_candidate_low =
        GpuCnf::from_host(&candidate_low, &fix.provider).expect("upload low-score MaxSAT CNF");
    let gpu_candidate_high =
        GpuCnf::from_host(&candidate_high, &fix.provider).expect("upload high-score MaxSAT CNF");
    let mut maxsat_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());

    let maxsat_report = maxsat_adapter
        .solve_weighted_maxsat_candidates_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &[
                GpuSolverProductionMaxSatCandidate {
                    score: 3,
                    cnf: &gpu_candidate_low,
                    branch_var_limit: &branch_limit,
                },
                GpuSolverProductionMaxSatCandidate {
                    score: 7,
                    cnf: &gpu_candidate_high,
                    branch_var_limit: &branch_limit,
                },
            ],
        )
        .expect(
            "accepted possible/not-know split GPU batch evidence must gate MaxSAT through GPU CDCL",
        );

    assert_eq!(maxsat_report.candidate_evidence_records, 2);
    assert_eq!(maxsat_report.optimum_score, 7);
    assert_eq!(maxsat_report.candidates_checked, 4);
    assert_eq!(maxsat_report.satisfiable_candidates, 4);
    assert_eq!(maxsat_report.unsat_candidates_pruned, 0);
    assert_eq!(maxsat_report.gpu_cdcl_candidate_encodes, 0);
    assert_eq!(maxsat_report.gpu_cdcl_candidate_solves, 4);

    let maxsat_trace = maxsat_adapter.trace();
    assert_eq!(
        maxsat_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        maxsat_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(maxsat_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        maxsat_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        maxsat_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        maxsat_trace.accepted_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        maxsat_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        maxsat_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        maxsat_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(maxsat_trace.gpu_cdcl_sat_solves, 4);
    assert_eq!(maxsat_trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(maxsat_trace.gpu_maxsat_optima, 2);
    assert_eq!(maxsat_trace.cpu_assignment_enumerations, 0);
    assert_eq!(maxsat_trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_quaternary_possible_and_not_know_batch_gates_solver_search_scheduler_and_portfolio_paths(
) {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred possible4(u32, u32, u32, u32).
        pred unknown4(u32, u32, u32, u32).
        possible4(A, B, C, D) :- tuple4(A, B, C, D), possible fact4(A, B, C, D).
        unknown4(A, B, C, D) :- tuple4(A, B, C, D), not know fact4(A, B, C, D).
        "#,
    )
    .expect("parse split quaternary possible/not-know solver search fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split quaternary possible/not-know solver search components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split quaternary possible/not-know solver search components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split quaternary possible/not-know solver search through GPU batch path");

    assert_eq!(batch.results.len(), 2);
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.know_operator_count, 0);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 0);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    for (idx, component) in split.components.iter().enumerate() {
        match component.component.rule_indices.as_slice() {
            [0] => {
                assert_eq!(
                    download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                    vec![(2, 3, 4, 5)]
                );
                assert_eq!(
                    batch.results[idx]
                        .prepared
                        .preflight
                        .possible_operator_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .model_membership
                        .tuple_source_key_column_device_reads,
                    4
                );
                assert_eq!(
                    batch.results[idx].model_membership.membership_source,
                    EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                );
            }
            [1] => {
                assert_eq!(
                    download_quaternary_u32(&fix.provider, &batch.results[idx].final_output),
                    vec![(1, 2, 3, 4), (9, 9, 9, 9)]
                );
                assert_eq!(
                    batch.results[idx]
                        .prepared
                        .preflight
                        .not_know_operator_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .final_tuple_materialization
                        .negated_row_filter_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .model_membership
                        .tuple_source_key_column_device_reads,
                    4
                );
                assert_eq!(
                    batch.results[idx].model_membership.membership_source,
                    EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                );
            }
            other => {
                panic!("unexpected split quaternary possible/not-know solver search rule indices: {other:?}")
            }
        }
    }

    let sat_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let unsat = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let gpu_sat_low = GpuCnf::from_host(&sat_low, &fix.provider).expect("upload SAT low");
    let gpu_sat_high = GpuCnf::from_host(&sat_high, &fix.provider).expect("upload SAT high");
    let gpu_unsat = GpuCnf::from_host(&unsat, &fix.provider).expect("upload UNSAT candidate");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);

    let mut search_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut search_workspace = search_adapter
        .new_workspace(gpu_unsat.var_cap, gpu_unsat.clause_cap)
        .expect("new possible/not-know MaxSAT search workspace");
    let search_report = search_adapter
        .solve_weighted_maxsat_search_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut search_workspace,
            &[
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 7,
                    cnf: &gpu_sat_low,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                },
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 11,
                    cnf: &gpu_unsat,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                },
            ],
        )
        .expect("accepted possible/not-know split GPU batch evidence must gate MaxSAT search");

    assert_eq!(search_report.candidate_evidence_records, 2);
    assert_eq!(search_report.optimum_score, 7);
    assert_eq!(search_report.candidates_checked, 4);
    assert_eq!(search_report.satisfiable_candidates, 2);
    assert_eq!(search_report.unsat_candidates_pruned, 2);
    assert_eq!(search_report.gpu_cdcl_candidate_encodes, 0);
    assert_eq!(search_report.gpu_cdcl_candidate_solves, 4);

    let search_trace = search_adapter.trace();
    assert_eq!(
        search_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(search_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        search_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        search_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        search_trace.accepted_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        search_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        search_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(search_trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(search_trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(search_trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(search_trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(search_trace.gpu_maxsat_optima, 2);
    assert_eq!(search_trace.cpu_assignment_enumerations, 0);
    assert_eq!(search_trace.cpu_maxsat_enumerations, 0);

    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![7.0, 9.0],
    );
    let both_soft_clauses = [0usize, 1usize];
    let sat_soft_clause = [0usize];
    let selections = [
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &both_soft_clauses,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &sat_soft_clause,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];

    let mut encoded_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut encoded_workspace = encoded_adapter
        .new_workspace(1, 2)
        .expect("new possible/not-know encoded MaxSAT workspace");
    let encoded_report = encoded_adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut encoded_workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("accepted possible/not-know split GPU batch evidence must gate encoded MaxSAT");

    assert_eq!(encoded_report.candidate_evidence_records, 2);
    assert_eq!(encoded_report.optimum_score, 7);
    assert_eq!(encoded_report.candidates_checked, 4);
    assert_eq!(encoded_report.satisfiable_candidates, 2);
    assert_eq!(encoded_report.unsat_candidates_pruned, 2);
    assert_eq!(encoded_report.gpu_cdcl_candidate_encodes, 4);
    assert_eq!(encoded_report.gpu_cdcl_candidate_solves, 4);

    let encoded_trace = encoded_adapter.trace();
    assert_eq!(
        encoded_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(encoded_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        encoded_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        encoded_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        encoded_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(encoded_trace.gpu_maxsat_candidate_encodes, 4);
    assert_eq!(encoded_trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(encoded_trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(encoded_trace.gpu_maxsat_optima, 2);
    assert_eq!(encoded_trace.cpu_assignment_enumerations, 0);
    assert_eq!(encoded_trace.cpu_maxsat_enumerations, 0);

    let candidate_set = [
        GpuSolverProductionMaxSatCandidate {
            score: 3,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionMaxSatCandidate {
            score: 5,
            cnf: &gpu_sat_high,
            branch_var_limit: &branch_limit,
        },
    ];
    let search_candidates = [
        GpuSolverProductionMaxSatSearchCandidate {
            score: 9,
            cnf: &gpu_unsat,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionMaxSatSearchCandidate {
            score: 7,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];
    let jobs = [
        GpuSolverProductionMaxSatScheduleJob::CandidateSet {
            candidates: &candidate_set,
        },
        GpuSolverProductionMaxSatScheduleJob::Search {
            candidates: &search_candidates,
        },
        GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
            weighted: &weighted,
            branch_var_limit: &branch_limit,
            selections: &selections,
        },
        GpuSolverProductionMaxSatScheduleJob::Unknown {
            reason: "possible/not-know scheduler branch budget exhausted",
        },
        GpuSolverProductionMaxSatScheduleJob::Timeout { budget_micros: 25 },
    ];
    let mut scheduler_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut scheduler_workspace = scheduler_adapter
        .new_workspace(1, 2)
        .expect("new possible/not-know generalized MaxSAT scheduler workspace");
    let schedule_report = scheduler_adapter
        .solve_maxsat_schedule_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut scheduler_workspace,
            &jobs,
        )
        .expect("accepted possible/not-know split GPU batch evidence must gate MaxSAT scheduler");

    assert_eq!(schedule_report.candidate_evidence_records, 2);
    assert_eq!(schedule_report.jobs, 10);
    assert_eq!(schedule_report.candidate_set_jobs, 2);
    assert_eq!(schedule_report.search_jobs, 2);
    assert_eq!(schedule_report.encoded_search_jobs, 2);
    assert_eq!(schedule_report.unknown_jobs, 2);
    assert_eq!(schedule_report.timeout_jobs, 2);
    assert_eq!(schedule_report.optimum_score, 7);
    assert_eq!(schedule_report.candidates_checked, 12);
    assert_eq!(schedule_report.satisfiable_candidates, 8);
    assert_eq!(schedule_report.unsat_candidates_pruned, 4);
    assert_eq!(schedule_report.gpu_cdcl_candidate_encodes, 4);
    assert_eq!(schedule_report.gpu_cdcl_candidate_solves, 12);

    let scheduler_trace = scheduler_adapter.trace();
    assert_eq!(
        scheduler_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(scheduler_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        scheduler_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        scheduler_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        scheduler_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_jobs, 10);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_candidate_set_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_search_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_encoded_search_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_unknown_status_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_timeout_status_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_encodes, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_solves, 12);
    assert_eq!(scheduler_trace.gpu_maxsat_unsat_candidate_prunes, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_optima, 6);
    assert_eq!(scheduler_trace.cpu_assignment_enumerations, 0);
    assert_eq!(scheduler_trace.cpu_maxsat_enumerations, 0);

    let portfolio_sat =
        GpuCnf::from_host(&sat_low, &fix.provider).expect("upload portfolio SAT CNF");
    let portfolio_maxsat_candidates = [GpuSolverProductionMaxSatCandidate {
        score: 5,
        cnf: &gpu_sat_high,
        branch_var_limit: &branch_limit,
    }];
    let portfolio_jobs = [
        GpuSolverProductionPortfolioJob::Sat {
            cnf: &portfolio_sat,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionPortfolioJob::MaxSat {
            candidates: &portfolio_maxsat_candidates,
        },
        GpuSolverProductionPortfolioJob::Unknown {
            reason: "possible/not-know portfolio branch budget exhausted",
        },
        GpuSolverProductionPortfolioJob::Timeout { budget_micros: 10 },
    ];
    let mut portfolio_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let portfolio_report = portfolio_adapter
        .solve_portfolio_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &portfolio_jobs,
        )
        .expect("accepted possible/not-know split GPU batch evidence must gate portfolio path");

    assert_eq!(portfolio_report.candidate_evidence_records, 2);
    assert_eq!(portfolio_report.jobs, 8);
    assert_eq!(portfolio_report.sat_jobs, 2);
    assert_eq!(portfolio_report.maxsat_jobs, 2);
    assert_eq!(portfolio_report.unknown_jobs, 2);
    assert_eq!(portfolio_report.timeout_jobs, 2);
    assert_eq!(portfolio_report.maxsat_optimum_scores, 10);

    let portfolio_trace = portfolio_adapter.trace();
    assert_eq!(
        portfolio_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        portfolio_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(portfolio_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        portfolio_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        portfolio_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        portfolio_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        portfolio_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(portfolio_trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(portfolio_trace.gpu_maxsat_optima, 2);
    assert_eq!(portfolio_trace.gpu_portfolio_jobs, 8);
    assert_eq!(portfolio_trace.gpu_portfolio_sat_jobs, 2);
    assert_eq!(portfolio_trace.gpu_portfolio_maxsat_jobs, 2);
    assert_eq!(portfolio_trace.gpu_portfolio_unknown_status_jobs, 2);
    assert_eq!(portfolio_trace.gpu_portfolio_timeout_status_jobs, 2);
    assert_eq!(portfolio_trace.cpu_assignment_enumerations, 0);
    assert_eq!(portfolio_trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_all_binary_operator_batch_gates_solver_reuse_and_maxsat_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred alt(u32, u32).
        pred blocked(u32, u32).
        pred seen(u32, u32).
        pred known_edge(u32, u32).
        pred possible_alt(u32, u32).
        pred clear_pair(u32, u32).
        pred unknown_pair(u32, u32).
        known_edge(X, Y) :- pair(X, Y), know edge(X, Y).
        possible_alt(X, Y) :- pair(X, Y), possible alt(X, Y).
        clear_pair(X, Y) :- pair(X, Y), not possible blocked(X, Y).
        unknown_pair(X, Y) :- pair(X, Y), not know seen(X, Y).
        "#,
    )
    .expect("parse split all binary solver reuse fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split all binary operator components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split all-operator components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "pair",
        upload_binary_u32(&fix.memory, &[(1, 2), (2, 3), (3, 4)]),
    );
    executor.put_relation("edge", upload_binary_u32(&fix.memory, &[(1, 2), (3, 4)]));
    executor.put_relation("alt", upload_binary_u32(&fix.memory, &[(2, 3)]));
    executor.put_relation("blocked", upload_binary_u32(&fix.memory, &[(3, 4)]));
    executor.put_relation("seen", upload_binary_u32(&fix.memory, &[(1, 2)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split all binary operator components through GPU batch path");

    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);

    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let source_cnf = GpuCnf::from_host(&unsat_instance, &fix.provider).expect("upload UNSAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut reuse_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = reuse_adapter
        .new_workspace(source_cnf.var_cap, source_cnf.clause_cap)
        .expect("new learned-clause reuse workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let reuse_report = reuse_adapter
        .solve_learned_clause_reuse_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &source_cnf,
            &branch_limit,
            &source_cnf,
            &branch_limit,
        )
        .expect("accepted all-operator split GPU batch evidence must gate learned-clause reuse");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(reuse_report.candidate_evidence_records, 4);
    assert_eq!(reuse_report.candidates, 8);
    assert_eq!(reuse_report.unsat_solves, 8);
    assert_eq!(reuse_report.gpu_learned_clause_arena_publications, 4);
    assert_eq!(reuse_report.gpu_learned_clause_imports, 4);
    assert_eq!(reuse_report.gpu_learned_clause_reused_solves, 4);
    assert_eq!(reuse_report.cpu_learned_clause_transfers, 0);

    let reuse_trace = reuse_adapter.trace();
    assert_eq!(
        reuse_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        reuse_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        4
    );
    assert_eq!(reuse_trace.accepted_gpu_candidate_evidence_consumed, 4);
    assert_eq!(reuse_trace.accepted_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        reuse_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        reuse_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        reuse_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(reuse_trace.gpu_cdcl_workspace_unsat_solves, 8);
    assert_eq!(reuse_trace.gpu_learned_clause_arena_publications, 4);
    assert_eq!(reuse_trace.gpu_learned_count_buffer_publications, 4);
    assert_eq!(reuse_trace.gpu_learned_clause_imports, 4);
    assert_eq!(reuse_trace.gpu_learned_clause_reused_solves, 4);
    assert_eq!(reuse_trace.cpu_learned_clause_transfers, 0);
    assert_eq!(reuse_trace.cpu_assignment_enumerations, 0);
    assert_eq!(reuse_trace.cpu_maxsat_enumerations, 0);

    let candidate_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let candidate_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let gpu_candidate_low =
        GpuCnf::from_host(&candidate_low, &fix.provider).expect("upload low-score MaxSAT CNF");
    let gpu_candidate_high =
        GpuCnf::from_host(&candidate_high, &fix.provider).expect("upload high-score MaxSAT CNF");
    let mut maxsat_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());

    let maxsat_report = maxsat_adapter
        .solve_weighted_maxsat_candidates_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &[
                GpuSolverProductionMaxSatCandidate {
                    score: 3,
                    cnf: &gpu_candidate_low,
                    branch_var_limit: &branch_limit,
                },
                GpuSolverProductionMaxSatCandidate {
                    score: 7,
                    cnf: &gpu_candidate_high,
                    branch_var_limit: &branch_limit,
                },
            ],
        )
        .expect("accepted all-operator split GPU batch evidence must gate MaxSAT through GPU CDCL");

    assert_eq!(maxsat_report.candidate_evidence_records, 4);
    assert_eq!(maxsat_report.optimum_score, 7);
    assert_eq!(maxsat_report.candidates_checked, 8);
    assert_eq!(maxsat_report.satisfiable_candidates, 8);
    assert_eq!(maxsat_report.unsat_candidates_pruned, 0);
    assert_eq!(maxsat_report.gpu_cdcl_candidate_encodes, 0);
    assert_eq!(maxsat_report.gpu_cdcl_candidate_solves, 8);

    let maxsat_trace = maxsat_adapter.trace();
    assert_eq!(
        maxsat_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        maxsat_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        4
    );
    assert_eq!(maxsat_trace.accepted_gpu_candidate_evidence_consumed, 4);
    assert_eq!(
        maxsat_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        maxsat_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        maxsat_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        maxsat_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(maxsat_trace.gpu_cdcl_sat_solves, 8);
    assert_eq!(maxsat_trace.gpu_maxsat_candidate_solves, 8);
    assert_eq!(maxsat_trace.gpu_maxsat_optima, 4);
    assert_eq!(maxsat_trace.cpu_assignment_enumerations, 0);
    assert_eq!(maxsat_trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_all_binary_operator_batch_gates_solver_search_scheduler_and_portfolio_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred alt(u32, u32).
        pred blocked(u32, u32).
        pred seen(u32, u32).
        pred known_edge(u32, u32).
        pred possible_alt(u32, u32).
        pred clear_pair(u32, u32).
        pred unknown_pair(u32, u32).
        known_edge(X, Y) :- pair(X, Y), know edge(X, Y).
        possible_alt(X, Y) :- pair(X, Y), possible alt(X, Y).
        clear_pair(X, Y) :- pair(X, Y), not possible blocked(X, Y).
        unknown_pair(X, Y) :- pair(X, Y), not know seen(X, Y).
        "#,
    )
    .expect("parse split all binary solver search fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split all binary solver search components");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split all-binary solver search components must preserve shared relation ids"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "pair",
        upload_binary_u32(&fix.memory, &[(1, 2), (2, 3), (3, 4)]),
    );
    executor.put_relation("edge", upload_binary_u32(&fix.memory, &[(1, 2), (3, 4)]));
    executor.put_relation("alt", upload_binary_u32(&fix.memory, &[(2, 3)]));
    executor.put_relation("blocked", upload_binary_u32(&fix.memory, &[(3, 4)]));
    executor.put_relation("seen", upload_binary_u32(&fix.memory, &[(1, 2)]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute split all-binary solver search through GPU batch path");

    assert_eq!(batch.results.len(), 4);
    assert_eq!(batch.trace.component_count, 4);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    for (idx, component) in split.components.iter().enumerate() {
        match component.component.rule_indices.as_slice() {
            [0] => {
                assert_eq!(
                    download_binary_u32(&fix.provider, &batch.results[idx].final_output),
                    vec![(1, 2), (3, 4)]
                );
                assert_eq!(batch.results[idx].prepared.preflight.know_operator_count, 1);
                assert_eq!(
                    batch.results[idx]
                        .model_membership
                        .tuple_source_key_column_device_reads,
                    2
                );
                assert_eq!(
                    batch.results[idx].model_membership.membership_source,
                    EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                );
            }
            [1] => {
                assert_eq!(
                    download_binary_u32(&fix.provider, &batch.results[idx].final_output),
                    vec![(2, 3)]
                );
                assert_eq!(
                    batch.results[idx]
                        .prepared
                        .preflight
                        .possible_operator_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .model_membership
                        .tuple_source_key_column_device_reads,
                    2
                );
                assert_eq!(
                    batch.results[idx].model_membership.membership_source,
                    EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                );
            }
            [2] => {
                assert_eq!(
                    download_binary_u32(&fix.provider, &batch.results[idx].final_output),
                    vec![(1, 2), (2, 3)]
                );
                assert_eq!(
                    batch.results[idx]
                        .prepared
                        .preflight
                        .not_possible_operator_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .final_tuple_materialization
                        .negated_row_filter_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .model_membership
                        .tuple_source_key_column_device_reads,
                    2
                );
                assert_eq!(
                    batch.results[idx].model_membership.membership_source,
                    EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                );
            }
            [3] => {
                assert_eq!(
                    download_binary_u32(&fix.provider, &batch.results[idx].final_output),
                    vec![(2, 3), (3, 4)]
                );
                assert_eq!(
                    batch.results[idx]
                        .prepared
                        .preflight
                        .not_know_operator_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .final_tuple_materialization
                        .negated_row_filter_count,
                    1
                );
                assert_eq!(
                    batch.results[idx]
                        .model_membership
                        .tuple_source_key_column_device_reads,
                    2
                );
                assert_eq!(
                    batch.results[idx].model_membership.membership_source,
                    EpistemicGpuModelMembershipSource::StableModelTupleBuffer
                );
            }
            other => {
                panic!("unexpected split all-binary solver search rule indices: {other:?}")
            }
        }
    }

    let sat_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let unsat = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let gpu_sat_low = GpuCnf::from_host(&sat_low, &fix.provider).expect("upload SAT low");
    let gpu_sat_high = GpuCnf::from_host(&sat_high, &fix.provider).expect("upload SAT high");
    let gpu_unsat = GpuCnf::from_host(&unsat, &fix.provider).expect("upload UNSAT candidate");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);

    let mut search_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut search_workspace = search_adapter
        .new_workspace(gpu_unsat.var_cap, gpu_unsat.clause_cap)
        .expect("new all-binary MaxSAT search workspace");
    let search_report = search_adapter
        .solve_weighted_maxsat_search_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut search_workspace,
            &[
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 7,
                    cnf: &gpu_sat_low,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                },
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 11,
                    cnf: &gpu_unsat,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                },
            ],
        )
        .expect("accepted all-binary split GPU batch evidence must gate MaxSAT search");

    assert_eq!(search_report.candidate_evidence_records, 4);
    assert_eq!(search_report.optimum_score, 7);
    assert_eq!(search_report.candidates_checked, 8);
    assert_eq!(search_report.satisfiable_candidates, 4);
    assert_eq!(search_report.unsat_candidates_pruned, 4);
    assert_eq!(search_report.gpu_cdcl_candidate_solves, 8);

    let search_trace = search_adapter.trace();
    assert_eq!(
        search_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        4
    );
    assert_eq!(search_trace.accepted_gpu_candidate_evidence_consumed, 4);
    assert_eq!(
        search_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        4
    );
    assert_eq!(
        search_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        search_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(search_trace.gpu_cdcl_sat_solves, 4);
    assert_eq!(search_trace.gpu_cdcl_workspace_unsat_solves, 4);
    assert_eq!(search_trace.gpu_maxsat_candidate_solves, 8);
    assert_eq!(search_trace.gpu_maxsat_unsat_candidate_prunes, 4);
    assert_eq!(search_trace.gpu_maxsat_optima, 4);
    assert_eq!(search_trace.cpu_assignment_enumerations, 0);
    assert_eq!(search_trace.cpu_maxsat_enumerations, 0);

    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![7.0, 9.0],
    );
    let both_soft_clauses = [0usize, 1usize];
    let sat_soft_clause = [0usize];
    let selections = [
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &both_soft_clauses,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &sat_soft_clause,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];

    let mut encoded_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut encoded_workspace = encoded_adapter
        .new_workspace(1, 2)
        .expect("new all-binary encoded MaxSAT workspace");
    let encoded_report = encoded_adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut encoded_workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("accepted all-binary split GPU batch evidence must gate encoded MaxSAT");

    assert_eq!(encoded_report.candidate_evidence_records, 4);
    assert_eq!(encoded_report.optimum_score, 7);
    assert_eq!(encoded_report.candidates_checked, 8);
    assert_eq!(encoded_report.satisfiable_candidates, 4);
    assert_eq!(encoded_report.unsat_candidates_pruned, 4);
    assert_eq!(encoded_report.gpu_cdcl_candidate_encodes, 8);
    assert_eq!(encoded_report.gpu_cdcl_candidate_solves, 8);

    let encoded_trace = encoded_adapter.trace();
    assert_eq!(
        encoded_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        4
    );
    assert_eq!(encoded_trace.accepted_gpu_candidate_evidence_consumed, 4);
    assert_eq!(
        encoded_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        4
    );
    assert_eq!(
        encoded_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        encoded_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(encoded_trace.gpu_maxsat_candidate_encodes, 8);
    assert_eq!(encoded_trace.gpu_maxsat_candidate_solves, 8);
    assert_eq!(encoded_trace.gpu_maxsat_unsat_candidate_prunes, 4);
    assert_eq!(encoded_trace.gpu_maxsat_optima, 4);
    assert_eq!(encoded_trace.cpu_assignment_enumerations, 0);
    assert_eq!(encoded_trace.cpu_maxsat_enumerations, 0);

    let candidate_set = [
        GpuSolverProductionMaxSatCandidate {
            score: 3,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionMaxSatCandidate {
            score: 5,
            cnf: &gpu_sat_high,
            branch_var_limit: &branch_limit,
        },
    ];
    let search_candidates = [
        GpuSolverProductionMaxSatSearchCandidate {
            score: 9,
            cnf: &gpu_unsat,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionMaxSatSearchCandidate {
            score: 7,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];
    let jobs = [
        GpuSolverProductionMaxSatScheduleJob::CandidateSet {
            candidates: &candidate_set,
        },
        GpuSolverProductionMaxSatScheduleJob::Search {
            candidates: &search_candidates,
        },
        GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
            weighted: &weighted,
            branch_var_limit: &branch_limit,
            selections: &selections,
        },
        GpuSolverProductionMaxSatScheduleJob::Unknown {
            reason: "all-binary scheduler branch budget exhausted",
        },
        GpuSolverProductionMaxSatScheduleJob::Timeout { budget_micros: 25 },
    ];
    let mut scheduler_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut scheduler_workspace = scheduler_adapter
        .new_workspace(1, 2)
        .expect("new all-binary generalized MaxSAT scheduler workspace");
    let schedule_report = scheduler_adapter
        .solve_maxsat_schedule_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut scheduler_workspace,
            &jobs,
        )
        .expect("accepted all-binary split GPU batch evidence must gate MaxSAT scheduler");

    assert_eq!(schedule_report.candidate_evidence_records, 4);
    assert_eq!(schedule_report.jobs, 20);
    assert_eq!(schedule_report.candidate_set_jobs, 4);
    assert_eq!(schedule_report.search_jobs, 4);
    assert_eq!(schedule_report.encoded_search_jobs, 4);
    assert_eq!(schedule_report.unknown_jobs, 4);
    assert_eq!(schedule_report.timeout_jobs, 4);
    assert_eq!(schedule_report.optimum_score, 7);
    assert_eq!(schedule_report.candidates_checked, 24);
    assert_eq!(schedule_report.satisfiable_candidates, 16);
    assert_eq!(schedule_report.unsat_candidates_pruned, 8);
    assert_eq!(schedule_report.gpu_cdcl_candidate_encodes, 8);
    assert_eq!(schedule_report.gpu_cdcl_candidate_solves, 24);

    let scheduler_trace = scheduler_adapter.trace();
    assert_eq!(
        scheduler_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        4
    );
    assert_eq!(scheduler_trace.accepted_gpu_candidate_evidence_consumed, 4);
    assert_eq!(
        scheduler_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        4
    );
    assert_eq!(
        scheduler_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        scheduler_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_jobs, 20);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_candidate_set_jobs, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_search_jobs, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_encoded_search_jobs, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_unknown_status_jobs, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_timeout_status_jobs, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_encodes, 8);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_solves, 24);
    assert_eq!(scheduler_trace.gpu_maxsat_unsat_candidate_prunes, 8);
    assert_eq!(scheduler_trace.gpu_maxsat_optima, 12);
    assert_eq!(scheduler_trace.cpu_assignment_enumerations, 0);
    assert_eq!(scheduler_trace.cpu_maxsat_enumerations, 0);

    let portfolio_sat =
        GpuCnf::from_host(&sat_low, &fix.provider).expect("upload portfolio SAT CNF");
    let portfolio_maxsat_candidates = [GpuSolverProductionMaxSatCandidate {
        score: 5,
        cnf: &gpu_sat_high,
        branch_var_limit: &branch_limit,
    }];
    let portfolio_jobs = [
        GpuSolverProductionPortfolioJob::Sat {
            cnf: &portfolio_sat,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionPortfolioJob::MaxSat {
            candidates: &portfolio_maxsat_candidates,
        },
        GpuSolverProductionPortfolioJob::Unknown {
            reason: "all-binary portfolio branch budget exhausted",
        },
        GpuSolverProductionPortfolioJob::Timeout { budget_micros: 10 },
    ];
    let mut portfolio_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let portfolio_report = portfolio_adapter
        .solve_portfolio_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &portfolio_jobs,
        )
        .expect("accepted all-binary split GPU batch evidence must gate portfolio path");

    assert_eq!(portfolio_report.candidate_evidence_records, 4);
    assert_eq!(portfolio_report.jobs, 16);
    assert_eq!(portfolio_report.sat_jobs, 4);
    assert_eq!(portfolio_report.maxsat_jobs, 4);
    assert_eq!(portfolio_report.unknown_jobs, 4);
    assert_eq!(portfolio_report.timeout_jobs, 4);
    assert_eq!(portfolio_report.maxsat_optimum_scores, 20);

    let portfolio_trace = portfolio_adapter.trace();
    assert_eq!(
        portfolio_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        portfolio_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        4
    );
    assert_eq!(portfolio_trace.accepted_gpu_candidate_evidence_consumed, 4);
    assert_eq!(
        portfolio_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        4
    );
    assert_eq!(
        portfolio_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        portfolio_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        portfolio_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        portfolio_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        portfolio_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(portfolio_trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(portfolio_trace.gpu_maxsat_optima, 4);
    assert_eq!(portfolio_trace.gpu_portfolio_jobs, 16);
    assert_eq!(portfolio_trace.gpu_portfolio_sat_jobs, 4);
    assert_eq!(portfolio_trace.gpu_portfolio_maxsat_jobs, 4);
    assert_eq!(portfolio_trace.gpu_portfolio_unknown_status_jobs, 4);
    assert_eq!(portfolio_trace.gpu_portfolio_timeout_status_jobs, 4);
    assert_eq!(portfolio_trace.cpu_assignment_enumerations, 0);
    assert_eq!(portfolio_trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_batch_gates_probabilistic_source_and_program_end_to_end_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

    let empty_assumptions: [EpistemicAssumption; 0] = [];
    let assumption_groups: [&[EpistemicAssumption]; 2] = [&empty_assumptions, &empty_assumptions];
    let probabilistic_source = r#"
        0.3::edge(1).
        0.7::color(2).
        query(edge(1)).
        query(color(2)).
        "#;
    let parsed_probabilistic_program = parse_program(probabilistic_source)
        .expect("parse split-batch probabilistic source/program fixture");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut source_adapter = EpistemicProbProductionAdapter::new(config);
    let source_evaluated = source_adapter
        .compile_and_evaluate_source_for_gpu_batch_execution_result(
            probabilistic_source,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split GPU batch evidence must gate source exact queries");

    assert_eq!(source_evaluated.len(), 2);
    for result in &source_evaluated {
        assert_eq!(result.query_probs.len(), 2);
        assert!((result.query_probs[0].prob - 0.3).abs() < 1.0e-6);
        assert!((result.query_probs[1].prob - 0.7).abs() < 1.0e-6);
    }

    let source_trace = source_adapter.trace();
    assert_eq!(source_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        source_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(source_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(source_trace.accepted_evidence_assumptions_consumed, 0);
    assert_eq!(source_trace.gpu_exact_source_compiles, 2);
    assert_eq!(source_trace.gpu_exact_program_compiles, 0);
    assert_eq!(source_trace.gpu_exact_query_evaluations, 2);
    assert_eq!(source_trace.gpu_source_exact_query_evaluations, 2);
    assert_eq!(source_trace.gpu_program_exact_query_evaluations, 0);
    assert_eq!(source_trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(
        source_trace.gpu_source_knowledge_compilation_end_to_end_runs,
        2
    );
    assert_eq!(
        source_trace.gpu_program_knowledge_compilation_end_to_end_runs,
        0
    );
    assert_eq!(source_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(source_trace.fixture_circuit_evaluations, 0);

    let mut program_adapter = EpistemicProbProductionAdapter::new(config);
    let program_evaluated = program_adapter
        .compile_and_evaluate_program_for_gpu_batch_execution_result(
            &parsed_probabilistic_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split GPU batch evidence must gate parsed-program exact queries");

    assert_eq!(program_evaluated.len(), 2);
    for result in &program_evaluated {
        assert_eq!(result.query_probs.len(), 2);
        assert!((result.query_probs[0].prob - 0.3).abs() < 1.0e-6);
        assert!((result.query_probs[1].prob - 0.7).abs() < 1.0e-6);
    }

    let program_trace = program_adapter.trace();
    assert_eq!(program_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        program_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(program_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(program_trace.accepted_evidence_assumptions_consumed, 0);
    assert_eq!(program_trace.gpu_exact_source_compiles, 0);
    assert_eq!(program_trace.gpu_exact_program_compiles, 2);
    assert_eq!(program_trace.gpu_exact_query_evaluations, 2);
    assert_eq!(program_trace.gpu_source_exact_query_evaluations, 0);
    assert_eq!(program_trace.gpu_program_exact_query_evaluations, 2);
    assert_eq!(program_trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(
        program_trace.gpu_source_knowledge_compilation_end_to_end_runs,
        0
    );
    assert_eq!(
        program_trace.gpu_program_knowledge_compilation_end_to_end_runs,
        2
    );
    assert_eq!(program_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(program_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_batch_gates_probabilistic_conditioned_source_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

    let edge_assumptions = [EpistemicAssumption::known_tuple(
        "edge",
        vec![EpistemicEvidenceTerm::integer(1)],
        true,
    )];
    let color_assumptions = [EpistemicAssumption::known_tuple(
        "color",
        vec![EpistemicEvidenceTerm::integer(2)],
        true,
    )];
    let assumption_groups: [&[EpistemicAssumption]; 2] = [&edge_assumptions, &color_assumptions];

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result(
            r#"
            0.3::edge(1).
            0.7::color(2).
            query(edge(1)).
            query(color(2)).
            "#,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split GPU batch evidence must gate conditioned exact queries");

    assert_eq!(evaluated.len(), 2);
    assert_eq!(evaluated[0].query_probs.len(), 2);
    assert!((evaluated[0].query_probs[0].prob - 1.0).abs() < 1.0e-6);
    assert!((evaluated[0].query_probs[1].prob - 0.7).abs() < 1.0e-6);
    assert_eq!(evaluated[1].query_probs.len(), 2);
    assert!((evaluated[1].query_probs[0].prob - 0.3).abs() < 1.0e-6);
    assert!((evaluated[1].query_probs[1].prob - 1.0).abs() < 1.0e-6);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 2);
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_batch_gates_probabilistic_conditioned_program_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

    let edge_assumptions = [EpistemicAssumption::known_tuple(
        "edge",
        vec![EpistemicEvidenceTerm::integer(1)],
        true,
    )];
    let color_assumptions = [EpistemicAssumption::known_tuple(
        "color",
        vec![EpistemicEvidenceTerm::integer(2)],
        true,
    )];
    let assumption_groups: [&[EpistemicAssumption]; 2] = [&edge_assumptions, &color_assumptions];
    let parsed_program = parse_program(
        r#"
        0.3::edge(1).
        0.7::color(2).
        query(edge(1)).
        query(color(2)).
        "#,
    )
    .expect("parse split-batch conditioned probabilistic program");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_program_for_gpu_batch_execution_result(
            &parsed_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split GPU batch evidence must gate conditioned parsed-program queries");

    assert_eq!(evaluated.len(), 2);
    assert_eq!(evaluated[0].query_probs.len(), 2);
    assert!((evaluated[0].query_probs[0].prob - 1.0).abs() < 1.0e-6);
    assert!((evaluated[0].query_probs[1].prob - 0.7).abs() < 1.0e-6);
    assert_eq!(evaluated[1].query_probs.len(), 2);
    assert!((evaluated[1].query_probs[0].prob - 0.3).abs() < 1.0e-6);
    assert!((evaluated[1].query_probs[1].prob - 1.0).abs() < 1.0e-6);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 2);
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_program_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_batch_gates_probabilistic_conditioned_source_gradients() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

    let edge_assumptions = [EpistemicAssumption::known_tuple(
        "edge",
        vec![EpistemicEvidenceTerm::integer(1)],
        true,
    )];
    let color_assumptions = [EpistemicAssumption::known_tuple(
        "color",
        vec![EpistemicEvidenceTerm::integer(2)],
        true,
    )];
    let assumption_groups: [&[EpistemicAssumption]; 2] = [&edge_assumptions, &color_assumptions];

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_with_grads_for_gpu_batch_execution_result(
            r#"
            0.3::edge(1).
            0.7::color(2).
            query(edge(1)).
            query(color(2)).
            "#,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split GPU batch evidence must gate conditioned source gradients");

    assert_eq!(evaluated.len(), 2);
    assert_eq!(evaluated[0].query_grads.len(), 2);
    assert!((evaluated[0].query_grads[0].prob - 1.0).abs() < 1.0e-6);
    assert!((evaluated[0].query_grads[1].prob - 0.7).abs() < 1.0e-6);
    assert_eq!(evaluated[1].query_grads.len(), 2);
    assert!((evaluated[1].query_grads[0].prob - 0.3).abs() < 1.0e-6);
    assert!((evaluated[1].query_grads[1].prob - 1.0).abs() < 1.0e-6);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 2);
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 0);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(trace.gpu_source_conditioned_gradient_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_batch_gates_probabilistic_conditioned_program_gradients() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

    let edge_assumptions = [EpistemicAssumption::known_tuple(
        "edge",
        vec![EpistemicEvidenceTerm::integer(1)],
        true,
    )];
    let color_assumptions = [EpistemicAssumption::known_tuple(
        "color",
        vec![EpistemicEvidenceTerm::integer(2)],
        true,
    )];
    let assumption_groups: [&[EpistemicAssumption]; 2] = [&edge_assumptions, &color_assumptions];
    let parsed_program = parse_program(
        r#"
        0.3::edge(1).
        0.7::color(2).
        query(edge(1)).
        query(color(2)).
        "#,
    )
    .expect("parse split-batch conditioned probabilistic program");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_program_with_grads_for_gpu_batch_execution_result(
            &parsed_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split GPU batch evidence must gate conditioned parsed-program gradients");

    assert_eq!(evaluated.len(), 2);
    assert_eq!(evaluated[0].query_grads.len(), 2);
    assert!((evaluated[0].query_grads[0].prob - 1.0).abs() < 1.0e-6);
    assert!((evaluated[0].query_grads[1].prob - 0.7).abs() < 1.0e-6);
    assert_eq!(evaluated[1].query_grads.len(), 2);
    assert!((evaluated[1].query_grads[0].prob - 0.3).abs() < 1.0e-6);
    assert!((evaluated[1].query_grads[1].prob - 1.0).abs() < 1.0e-6);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(trace.accepted_gpu_batch_component_evidence_consumed, 2);
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_program_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 0);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(trace.gpu_program_conditioned_gradient_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_batch_gates_solver_lifecycle_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

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
        .solve_assumption_lifecycle_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
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
        .expect("accepted split GPU batch evidence must gate solver lifecycle path");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.steps, 4);
    assert_eq!(report.assumption_pushes, 4);
    assert_eq!(report.assumption_retractions, 4);
    assert_eq!(report.workspace_reuses, 2);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
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
fn accepted_split_batch_gates_solver_maxsat_lifecycle_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

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
    let maxsat_candidates = [
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
    ];
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(unsat_cnf.var_cap, unsat_cnf.clause_cap)
        .expect("new MaxSAT lifecycle workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let report = adapter
        .solve_maxsat_lifecycle_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
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
            &maxsat_candidates,
        )
        .expect("accepted split GPU batch evidence must gate MaxSAT lifecycle path");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.lifecycle.steps, 4);
    assert_eq!(report.lifecycle.assumption_pushes, 4);
    assert_eq!(report.lifecycle.assumption_retractions, 4);
    assert_eq!(report.lifecycle.workspace_reuses, 2);
    assert_eq!(report.maxsat.optimum_score, 7);
    assert_eq!(report.maxsat.candidates_checked, 4);
    assert_eq!(report.maxsat.satisfiable_candidates, 4);
    assert_eq!(report.maxsat.gpu_cdcl_candidate_solves, 4);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.gpu_assumption_pushes, 4);
    assert_eq!(trace.gpu_assumption_retractions, 4);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 2);
    assert_eq!(trace.gpu_cdcl_sat_solves, 6);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(trace.gpu_maxsat_optima, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_batch_gates_solver_portfolio_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let maxsat_candidate =
        GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload MaxSAT candidate CNF");
    let portfolio_sat =
        GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload portfolio SAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let maxsat_candidates = [GpuSolverProductionMaxSatCandidate {
        score: 5,
        cnf: &maxsat_candidate,
        branch_var_limit: &branch_limit,
    }];
    let portfolio_jobs = [
        GpuSolverProductionPortfolioJob::Sat {
            cnf: &portfolio_sat,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionPortfolioJob::MaxSat {
            candidates: &maxsat_candidates,
        },
        GpuSolverProductionPortfolioJob::Unknown {
            reason: "bounded branch budget exhausted before a determined status",
        },
        GpuSolverProductionPortfolioJob::Timeout { budget_micros: 10 },
    ];
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());

    let report = adapter
        .solve_portfolio_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &portfolio_jobs,
        )
        .expect("accepted split GPU batch evidence must gate status-aware portfolio path");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.jobs, 8);
    assert_eq!(report.sat_jobs, 2);
    assert_eq!(report.maxsat_jobs, 2);
    assert_eq!(report.unknown_jobs, 2);
    assert_eq!(report.timeout_jobs, 2);
    assert_eq!(report.maxsat_optimum_scores, 10);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(trace.gpu_maxsat_optima, 2);
    assert_eq!(trace.gpu_portfolio_jobs, 8);
    assert_eq!(trace.gpu_portfolio_sat_jobs, 2);
    assert_eq!(trace.gpu_portfolio_maxsat_jobs, 2);
    assert_eq!(trace.gpu_portfolio_unknown_status_jobs, 2);
    assert_eq!(trace.gpu_portfolio_timeout_status_jobs, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_batch_gates_solver_learned_clause_reuse_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let source_cnf = GpuCnf::from_host(&unsat_instance, &fix.provider).expect("upload UNSAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(source_cnf.var_cap, source_cnf.clause_cap)
        .expect("new workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let report = adapter
        .solve_learned_clause_reuse_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &source_cnf,
            &branch_limit,
            &source_cnf,
            &branch_limit,
        )
        .expect("accepted split GPU batch evidence must gate learned-clause reuse path");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.candidates, 4);
    assert_eq!(report.unsat_solves, 4);
    assert_eq!(report.gpu_learned_clause_arena_publications, 2);
    assert_eq!(report.gpu_learned_clause_imports, 2);
    assert_eq!(report.gpu_learned_clause_reused_solves, 2);
    assert_eq!(report.cpu_learned_clause_transfers, 0);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 4);
    assert_eq!(trace.gpu_learned_clause_arena_publications, 2);
    assert_eq!(trace.gpu_learned_count_buffer_publications, 2);
    assert_eq!(trace.gpu_learned_clause_imports, 2);
    assert_eq!(trace.gpu_learned_clause_reused_solves, 2);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_batch_gates_solver_maxsat_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

    let candidate_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let candidate_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let gpu_candidate_low =
        GpuCnf::from_host(&candidate_low, &fix.provider).expect("upload low-score MaxSAT CNF");
    let gpu_candidate_high =
        GpuCnf::from_host(&candidate_high, &fix.provider).expect("upload high-score MaxSAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());

    let report = adapter
        .solve_weighted_maxsat_candidates_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &[
                GpuSolverProductionMaxSatCandidate {
                    score: 3,
                    cnf: &gpu_candidate_low,
                    branch_var_limit: &branch_limit,
                },
                GpuSolverProductionMaxSatCandidate {
                    score: 7,
                    cnf: &gpu_candidate_high,
                    branch_var_limit: &branch_limit,
                },
            ],
        )
        .expect("accepted split GPU batch evidence must gate MaxSAT through GPU CDCL");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.optimum_score, 7);
    assert_eq!(report.candidates_checked, 4);
    assert_eq!(report.satisfiable_candidates, 4);
    assert_eq!(report.unsat_candidates_pruned, 0);
    assert_eq!(report.gpu_cdcl_candidate_encodes, 0);
    assert_eq!(report.gpu_cdcl_candidate_solves, 4);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.gpu_cdcl_sat_solves, 4);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(trace.gpu_maxsat_optima, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_batch_gates_solver_maxsat_search_pruning() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let sat_cnf =
        GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT MaxSAT search CNF");
    let unsat_cnf =
        GpuCnf::from_host(&unsat_instance, &fix.provider).expect("upload UNSAT MaxSAT search CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(unsat_cnf.var_cap, unsat_cnf.clause_cap)
        .expect("new split-batch MaxSAT search workspace");

    let report = adapter
        .solve_weighted_maxsat_search_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &[
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 7,
                    cnf: &sat_cnf,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                },
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 11,
                    cnf: &unsat_cnf,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                },
            ],
        )
        .expect("accepted split GPU batch evidence must gate MaxSAT search pruning");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.optimum_score, 7);
    assert_eq!(report.candidates_checked, 4);
    assert_eq!(report.satisfiable_candidates, 2);
    assert_eq!(report.unsat_candidates_pruned, 2);
    assert_eq!(report.gpu_cdcl_candidate_encodes, 0);
    assert_eq!(report.gpu_cdcl_candidate_solves, 4);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(trace.gpu_maxsat_optima, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_batch_gates_solver_encoded_maxsat_and_scheduler_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

    let sat_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let unsat = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let gpu_sat_low = GpuCnf::from_host(&sat_low, &fix.provider).expect("upload SAT low");
    let gpu_sat_high = GpuCnf::from_host(&sat_high, &fix.provider).expect("upload SAT high");
    let gpu_unsat = GpuCnf::from_host(&unsat, &fix.provider).expect("upload UNSAT candidate");
    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![7.0, 9.0],
    );
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let both_soft_clauses = [0usize, 1usize];
    let sat_soft_clause = [0usize];

    let mut encoded_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut encoded_workspace = encoded_adapter
        .new_workspace(1, 2)
        .expect("new split-batch encoded MaxSAT workspace");
    let encoded_report = encoded_adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut encoded_workspace,
            &weighted,
            &branch_limit,
            &[
                GpuSolverProductionWeightedMaxSatSelection {
                    soft_clause_indices: &both_soft_clauses,
                    status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                },
                GpuSolverProductionWeightedMaxSatSelection {
                    soft_clause_indices: &sat_soft_clause,
                    status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                },
            ],
        )
        .expect("accepted split GPU batch evidence must gate weighted MaxSAT encoding");

    assert_eq!(encoded_report.candidate_evidence_records, 2);
    assert_eq!(encoded_report.optimum_score, 7);
    assert_eq!(encoded_report.candidates_checked, 4);
    assert_eq!(encoded_report.satisfiable_candidates, 2);
    assert_eq!(encoded_report.unsat_candidates_pruned, 2);
    assert_eq!(encoded_report.gpu_cdcl_candidate_encodes, 4);
    assert_eq!(encoded_report.gpu_cdcl_candidate_solves, 4);

    let encoded_trace = encoded_adapter.trace();
    assert_eq!(
        encoded_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(encoded_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(encoded_trace.gpu_maxsat_candidate_encodes, 4);
    assert_eq!(encoded_trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(encoded_trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(encoded_trace.gpu_maxsat_optima, 2);
    assert_eq!(encoded_trace.cpu_assignment_enumerations, 0);
    assert_eq!(encoded_trace.cpu_maxsat_enumerations, 0);

    let candidate_set = [
        GpuSolverProductionMaxSatCandidate {
            score: 3,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionMaxSatCandidate {
            score: 5,
            cnf: &gpu_sat_high,
            branch_var_limit: &branch_limit,
        },
    ];
    let search_candidates = [
        GpuSolverProductionMaxSatSearchCandidate {
            score: 9,
            cnf: &gpu_unsat,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionMaxSatSearchCandidate {
            score: 7,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];
    let selections = [
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &both_soft_clauses,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &sat_soft_clause,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];
    let jobs = [
        GpuSolverProductionMaxSatScheduleJob::CandidateSet {
            candidates: &candidate_set,
        },
        GpuSolverProductionMaxSatScheduleJob::Search {
            candidates: &search_candidates,
        },
        GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
            weighted: &weighted,
            branch_var_limit: &branch_limit,
            selections: &selections,
        },
        GpuSolverProductionMaxSatScheduleJob::Unknown {
            reason: "bounded scheduler branch budget exhausted",
        },
        GpuSolverProductionMaxSatScheduleJob::Timeout { budget_micros: 25 },
    ];
    let mut scheduler_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut scheduler_workspace = scheduler_adapter
        .new_workspace(1, 2)
        .expect("new split-batch generalized MaxSAT scheduler workspace");
    let schedule_report = scheduler_adapter
        .solve_maxsat_schedule_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut scheduler_workspace,
            &jobs,
        )
        .expect("accepted split GPU batch evidence must gate generalized MaxSAT scheduler");

    assert_eq!(schedule_report.candidate_evidence_records, 2);
    assert_eq!(schedule_report.jobs, 10);
    assert_eq!(schedule_report.candidate_set_jobs, 2);
    assert_eq!(schedule_report.search_jobs, 2);
    assert_eq!(schedule_report.encoded_search_jobs, 2);
    assert_eq!(schedule_report.unknown_jobs, 2);
    assert_eq!(schedule_report.timeout_jobs, 2);
    assert_eq!(schedule_report.optimum_score, 7);
    assert_eq!(schedule_report.candidates_checked, 12);
    assert_eq!(schedule_report.satisfiable_candidates, 8);
    assert_eq!(schedule_report.unsat_candidates_pruned, 4);
    assert_eq!(schedule_report.gpu_cdcl_candidate_encodes, 4);
    assert_eq!(schedule_report.gpu_cdcl_candidate_solves, 12);

    let scheduler_trace = scheduler_adapter.trace();
    assert_eq!(
        scheduler_trace.accepted_gpu_batch_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        2
    );
    assert_eq!(scheduler_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_jobs, 10);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_candidate_set_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_search_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_encoded_search_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_unknown_status_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_timeout_status_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_encodes, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_solves, 12);
    assert_eq!(scheduler_trace.gpu_maxsat_unsat_candidate_prunes, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_optima, 6);
    assert_eq!(scheduler_trace.cpu_assignment_enumerations, 0);
    assert_eq!(scheduler_trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_split_batch_rejects_invalid_encoded_maxsat_scheduler_before_scheduler_work() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![7.0, 9.0],
    );
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let both_soft_clauses = [0usize, 1usize];
    let selections = [GpuSolverProductionWeightedMaxSatSelection {
        soft_clause_indices: &both_soft_clauses,
        status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
    }];
    let jobs = [GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
        weighted: &weighted,
        branch_var_limit: &branch_limit,
        selections: &selections,
    }];

    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(1, 2)
        .expect("new split-batch generalized MaxSAT scheduler workspace");
    let err = adapter
        .solve_maxsat_schedule_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut workspace,
            &jobs,
        )
        .expect_err("all-UNSAT encoded scheduler jobs must fail before scheduler work");
    assert!(format!("{err}")
        .contains("bounded MaxSAT search requires at least one satisfiable GPU candidate"));

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 0);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        0
    );
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.gpu_maxsat_scheduler_jobs, 0);
    assert_eq!(trace.gpu_maxsat_scheduler_encoded_search_jobs, 0);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 0);
    assert_eq!(trace.gpu_cdcl_sat_solves, 0);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 0);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 0);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 0);
    assert_eq!(trace.gpu_maxsat_optima, 0);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);

    let empty_soft_clauses: [usize; 0] = [];
    let empty_selections = [GpuSolverProductionWeightedMaxSatSelection {
        soft_clause_indices: &empty_soft_clauses,
        status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
    }];
    let empty_jobs = [GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
        weighted: &weighted,
        branch_var_limit: &branch_limit,
        selections: &empty_selections,
    }];
    let mut empty_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut empty_workspace = empty_adapter
        .new_workspace(1, 2)
        .expect("new empty-selection generalized MaxSAT scheduler workspace");
    let err = empty_adapter
        .solve_maxsat_schedule_with_gpu_batch_execution_result(
            &fix.provider,
            GpuSolverProductionBatchExecutionEvidence { batch: &batch },
            &mut empty_workspace,
            &empty_jobs,
        )
        .expect_err("empty encoded scheduler selections must fail before scheduler work");
    assert!(format!("{err}")
        .contains("weighted MaxSAT search selections must include at least one soft clause"));

    let trace = empty_adapter.trace();
    assert_eq!(trace.accepted_gpu_batch_candidate_evidence_consumed, 0);
    assert_eq!(
        trace.accepted_gpu_batch_candidate_component_evidence_consumed,
        0
    );
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.gpu_maxsat_scheduler_jobs, 0);
    assert_eq!(trace.gpu_maxsat_scheduler_encoded_search_jobs, 0);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 0);
    assert_eq!(trace.gpu_cdcl_sat_solves, 0);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 0);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 0);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 0);
    assert_eq!(trace.gpu_maxsat_optima, 0);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn split_gpu_world_view_distinguishes_absent_possible_from_not_known() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred possible_edge(u32).
        pred not_known_edge(u32).
        possible_edge(X) :- node(X), possible edge(X).
        not_known_edge(X) :- node(X), not know edge(X).
        "#,
    )
    .expect("parse possible-vs-not-known split fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile possible-vs-not-known split fixture");

    assert_eq!(split.components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute possible-vs-not-known split components");
    assert_eq!(batch.trace.know_operator_count, 0);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 0);

    let mut outputs_by_head = BTreeMap::new();
    for (component, result) in split.components.iter().zip(batch.results.iter()) {
        let head = component.executable.gpu_plan.reductions[0]
            .head_predicate
            .clone();
        outputs_by_head.insert(
            head,
            download_unary_u32(&fix.provider, &result.final_output),
        );
        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    }

    assert_eq!(
        outputs_by_head
            .get("possible_edge")
            .cloned()
            .unwrap_or_default(),
        Vec::<u32>::new(),
        "possible edge(X) must reject rows when the stable-model tuple source is absent"
    );
    assert_eq!(
        outputs_by_head
            .get("not_known_edge")
            .cloned()
            .unwrap_or_default(),
        vec![1, 2, 3],
        "not know edge(X) must accept rows when the stable-model tuple source is absent"
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
fn accepted_gpu_execution_semantic_trace_matches_gpt_oracle_rejection_reason() {
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
    .expect("parse GPT oracle parity fixture");
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            EpistemicInterpretation::new(),
            EpistemicInterpretation::new().with_known("edge", 1),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
    )
    .expect("run bounded GPT oracle");
    assert_eq!(
        oracle.trace.rejection_reasons,
        vec![FaeelNoModelReason::UnsatisfiedLiteral {
            predicate: "edge".to_string(),
            arity: 1,
        }]
    );

    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile GPT oracle parity executable");

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
        .expect("execute GPT oracle parity fixture");

    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
    assert_eq!(
        result
            .semantic_trace
            .typed_rejection_reasons()
            .expect("decode GPU rejection reasons"),
        vec![EpistemicGpuRejectionReason::UnsatisfiedMembership]
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
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            // Candidate index 0 represents the negated literal as false.
            EpistemicInterpretation::new().with_known("edge", 1),
            // Candidate index 1 represents the negated literal as true.
            EpistemicInterpretation::new(),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
    )
    .expect("run not-know GPT oracle");
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

    assert_eq!(result.prepared.preflight.know_operator_count, 0);
    assert_eq!(result.prepared.preflight.possible_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
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
fn accepted_possible_nonzero_arity_membership_records_operator_metrics() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), possible edge(X).
        "#,
    )
    .expect("parse possible nonzero-arity epistemic fixture");
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            EpistemicInterpretation::new(),
            EpistemicInterpretation::new().with_known("edge", 1),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
    )
    .expect("run possible GPT oracle");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile possible nonzero-arity epistemic executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[2, 3]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute possible nonzero-arity epistemic fixture");

    assert_eq!(result.prepared.preflight.know_operator_count, 0);
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![2, 3],
        "possible must keep only reduced rows whose bound tuple key appears in the stable model"
    );
}

#[test]
fn accepted_not_possible_nonzero_arity_membership_records_operator_and_polarity_metrics() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), not possible edge(X).
        "#,
    )
    .expect("parse not-possible nonzero-arity epistemic fixture");
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            // Candidate index 0 represents the negated literal as false.
            EpistemicInterpretation::new().with_known("edge", 1),
            // Candidate index 1 represents the negated literal as true.
            EpistemicInterpretation::new(),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
    )
    .expect("run not-possible GPT oracle");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile not-possible nonzero-arity epistemic executable");

    assert!(executable.gpu_plan.tuple_membership_bindings[0].negated);

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[2]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute not-possible nonzero-arity epistemic fixture");

    assert_eq!(result.prepared.preflight.know_operator_count, 0);
    assert_eq!(result.prepared.preflight.possible_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![1, 3],
        "not possible must keep only reduced rows whose bound tuple key is absent from every stable-model tuple source"
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
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            EpistemicInterpretation::new(),
            EpistemicInterpretation::new().with_known("edge", 2),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
    )
    .expect("run binary GPT oracle");
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

    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
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
fn accepted_ternary_membership_matches_gpt_oracle_parity() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred triple(u32, u32, u32).
        pred fact3(u32, u32, u32).
        pred accepted(u32, u32, u32).
        accepted(A, B, C) :- triple(A, B, C), know fact3(A, B, C).
        "#,
    )
    .expect("parse ternary epistemic fixture");
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            EpistemicInterpretation::new(),
            EpistemicInterpretation::new().with_known("fact3", 3),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
    )
    .expect("run ternary GPT oracle");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile ternary epistemic executable");
    assert_eq!(executable.gpu_plan.tuple_membership_bindings.len(), 1);
    assert_eq!(executable.gpu_plan.tuple_membership_bindings[0].arity, 3);
    assert_eq!(
        executable.gpu_plan.tuple_membership_bindings[0].bound_output_columns,
        vec![Some(0), Some(1), Some(2)]
    );

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "triple",
        upload_ternary_u32(&fix.memory, &[(1, 2, 3), (2, 3, 5), (8, 13, 21)]),
    );
    executor.put_relation("fact3", upload_ternary_u32(&fix.memory, &[(2, 3, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute ternary epistemic fixture");

    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    assert_eq!(
        download_ternary_u32(&fix.provider, &result.final_output),
        vec![(2, 3, 5)],
        "final output must keep only rows whose arity-3 tuple key appears in the stable model"
    );
}

#[test]
fn accepted_quaternary_membership_matches_gpt_oracle_parity() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary epistemic fixture");
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            EpistemicInterpretation::new(),
            EpistemicInterpretation::new().with_known("fact4", 4),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
    )
    .expect("run quaternary GPT oracle");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary epistemic executable");
    assert_eq!(executable.gpu_plan.tuple_membership_bindings.len(), 1);
    assert_eq!(executable.gpu_plan.tuple_membership_bindings[0].arity, 4);
    assert_eq!(
        executable.gpu_plan.tuple_membership_bindings[0].bound_output_columns,
        vec![Some(0), Some(1), Some(2), Some(3)]
    );

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary epistemic fixture");

    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(2, 3, 4, 5)],
        "final output must keep only rows whose arity-4 tuple key appears in the stable model"
    );
}

#[test]
fn accepted_quaternary_not_possible_membership_matches_gpt_oracle_parity() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary not-possible epistemic fixture");
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            // Candidate index 0 represents the negated literal as false.
            EpistemicInterpretation::new().with_known("fact4", 4),
            // Candidate index 1 represents the negated literal as true.
            EpistemicInterpretation::new(),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
    )
    .expect("run quaternary not-possible GPT oracle");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary not-possible epistemic executable");
    assert_eq!(executable.gpu_plan.tuple_membership_bindings.len(), 1);
    assert_eq!(executable.gpu_plan.tuple_membership_bindings[0].arity, 4);
    assert_eq!(
        executable.gpu_plan.tuple_membership_bindings[0].bound_output_columns,
        vec![Some(0), Some(1), Some(2), Some(3)]
    );
    assert!(executable.gpu_plan.tuple_membership_bindings[0].negated);

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary not-possible epistemic fixture");

    assert_eq!(result.prepared.preflight.know_operator_count, 0);
    assert_eq!(result.prepared.preflight.possible_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(1, 2, 3, 4), (9, 9, 9, 9)],
        "quaternary not possible must keep only tuple keys absent from the stable model"
    );
}

#[test]
fn accepted_quaternary_possible_and_not_know_memberships_match_gpt_oracle_parity() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let run_case = |label: &str,
                    program_source: &str,
                    oracle_candidates: Vec<EpistemicInterpretation>,
                    expected_rows: Vec<(u32, u32, u32, u32)>,
                    expected_possible: usize,
                    expected_not_know: usize,
                    expected_negated_filters: usize| {
        let program = parse_program(program_source).expect(label);
        let oracle = run_generate_propagate_test(
            &program,
            oracle_candidates,
            GeneratePropagateTestConfig { max_candidates: 2 },
        )
        .expect(label);
        let executable =
            compile_epistemic_gpu_execution_with_stats_snapshot(&program, None).expect(label);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings.len(), 1);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings[0].arity, 4);
        assert_eq!(
            executable.gpu_plan.tuple_membership_bindings[0].bound_output_columns,
            vec![Some(0), Some(1), Some(2), Some(3)]
        );
        assert_eq!(
            executable.gpu_plan.tuple_membership_bindings[0].negated,
            expected_negated_filters == 1
        );

        let mut executor =
            Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
        for (name, rel_id) in &executable.relation_ids {
            executor.register_relation(*rel_id, name);
        }
        executor.put_relation(
            "tuple4",
            upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
        );
        executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

        let result = executor
            .execute_epistemic_gpu_execution(
                &executable,
                EpistemicGpuWorkspaceCapacities {
                    max_candidates: 2,
                    max_worlds: 1,
                    max_models_per_reduction: 3,
                },
            )
            .expect(label);

        assert_eq!(result.prepared.preflight.know_operator_count, 0);
        assert_eq!(
            result.prepared.preflight.possible_operator_count,
            expected_possible
        );
        assert_eq!(
            result.prepared.preflight.not_know_operator_count,
            expected_not_know
        );
        assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
        assert_eq!(
            result.semantic_trace.generated_candidates,
            oracle.trace.generated
        );
        assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
        assert_eq!(
            result.semantic_trace.propagated_candidates,
            oracle.trace.propagated
        );
        assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
        assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
        assert_eq!(
            result.semantic_trace.accepted_candidates,
            oracle.trace.accepted
        );
        assert_eq!(
            result.semantic_trace.accepted_world_views,
            oracle.trace.accepted_world_views
        );
        assert_eq!(
            result.semantic_trace.rejected_candidates,
            oracle.trace.rejected
        );
        assert_eq!(
            result.semantic_trace.accepted_candidate_indices,
            oracle.accepted_candidate_indices
        );
        assert_eq!(
            result.semantic_trace.rejected_candidate_indices,
            oracle.rejected_candidate_indices
        );
        assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
        assert_eq!(
            result.final_tuple_materialization.negated_row_filter_count,
            expected_negated_filters
        );
        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            4
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
        assert_eq!(
            download_quaternary_u32(&fix.provider, &result.final_output),
            expected_rows,
            "{label} must filter by the arity-4 tuple key on device"
        );
    };

    run_case(
        "quaternary possible GPT oracle parity",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), possible fact4(A, B, C, D).
        "#,
        vec![
            EpistemicInterpretation::new(),
            EpistemicInterpretation::new().with_known("fact4", 4),
        ],
        vec![(2, 3, 4, 5)],
        1,
        0,
        0,
    );
    run_case(
        "quaternary not-know GPT oracle parity",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not know fact4(A, B, C, D).
        "#,
        vec![
            EpistemicInterpretation::new().with_known("fact4", 4),
            EpistemicInterpretation::new(),
        ],
        vec![(1, 2, 3, 4), (9, 9, 9, 9)],
        0,
        1,
        1,
    );
}

#[test]
fn accepted_binary_possible_membership_matches_gpt_oracle_parity() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred accepted(u32, u32).
        accepted(X, Y) :- pair(X, Y), possible edge(X, Y).
        "#,
    )
    .expect("parse binary possible epistemic fixture");
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            EpistemicInterpretation::new(),
            EpistemicInterpretation::new().with_known("edge", 2),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
    )
    .expect("run binary possible GPT oracle");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile binary possible epistemic executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "pair",
        upload_binary_u32(&fix.memory, &[(1, 2), (2, 3), (3, 4)]),
    );
    executor.put_relation("edge", upload_binary_u32(&fix.memory, &[(1, 2), (3, 4)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute binary possible epistemic fixture");

    assert_eq!(result.prepared.preflight.know_operator_count, 0);
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(
        download_binary_u32(&fix.provider, &result.final_output),
        vec![(1, 2), (3, 4)],
        "binary possible must keep only bound tuple keys present in the stable model"
    );
}

#[test]
fn accepted_binary_not_possible_membership_matches_gpt_oracle_parity() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred accepted(u32, u32).
        accepted(X, Y) :- pair(X, Y), not possible edge(X, Y).
        "#,
    )
    .expect("parse binary not-possible epistemic fixture");
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            // Candidate index 0 represents the negated literal as false.
            EpistemicInterpretation::new().with_known("edge", 2),
            // Candidate index 1 represents the negated literal as true.
            EpistemicInterpretation::new(),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
    )
    .expect("run binary not-possible GPT oracle");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile binary not-possible epistemic executable");
    assert!(executable.gpu_plan.tuple_membership_bindings[0].negated);

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "pair",
        upload_binary_u32(&fix.memory, &[(1, 2), (2, 3), (3, 4)]),
    );
    executor.put_relation("edge", upload_binary_u32(&fix.memory, &[(2, 3)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute binary not-possible epistemic fixture");

    assert_eq!(result.prepared.preflight.know_operator_count, 0);
    assert_eq!(result.prepared.preflight.possible_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(
        download_binary_u32(&fix.provider, &result.final_output),
        vec![(1, 2), (3, 4)],
        "binary not possible must keep only tuple keys absent from the stable model"
    );
}

#[test]
fn accepted_binary_not_know_membership_matches_gpt_oracle_parity() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred accepted(u32, u32).
        accepted(X, Y) :- pair(X, Y), not know edge(X, Y).
        "#,
    )
    .expect("parse binary not-know epistemic fixture");
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            // Candidate index 0 represents the negated literal as false.
            EpistemicInterpretation::new().with_known("edge", 2),
            // Candidate index 1 represents the negated literal as true.
            EpistemicInterpretation::new(),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
    )
    .expect("run binary not-know GPT oracle");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile binary not-know epistemic executable");
    assert!(executable.gpu_plan.tuple_membership_bindings[0].negated);

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "pair",
        upload_binary_u32(&fix.memory, &[(1, 2), (2, 3), (3, 4)]),
    );
    executor.put_relation("edge", upload_binary_u32(&fix.memory, &[(2, 3)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute binary not-know epistemic fixture");

    assert_eq!(result.prepared.preflight.know_operator_count, 0);
    assert_eq!(result.prepared.preflight.possible_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    assert_eq!(
        download_binary_u32(&fix.provider, &result.final_output),
        vec![(1, 2), (3, 4)],
        "binary not know must keep only tuple keys absent from the stable model"
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
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            EpistemicInterpretation::new(),
            EpistemicInterpretation::new().with_known("edge", 1),
            EpistemicInterpretation::new().with_known("color", 1),
            EpistemicInterpretation::new()
                .with_known("edge", 1)
                .with_known("color", 1),
        ],
        GeneratePropagateTestConfig { max_candidates: 4 },
    )
    .expect("run multi-membership GPT oracle");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile multi-membership epistemic executable");
    assert_eq!(executable.gpu_plan.epistemic_literals.len(), 2);

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
                max_candidates: 4,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute multi-membership epistemic fixture");

    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 2);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views, 1,
        "only the candidate whose assumptions are supported by every required membership can pass"
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
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
fn accepted_mixed_memberships_match_gpt_oracle_parity() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred alt(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X), possible alt(X).
        "#,
    )
    .expect("parse mixed-membership epistemic fixture");
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            EpistemicInterpretation::new(),
            EpistemicInterpretation::new().with_known("edge", 1),
            EpistemicInterpretation::new().with_known("alt", 1),
            EpistemicInterpretation::new()
                .with_known("edge", 1)
                .with_known("alt", 1),
        ],
        GeneratePropagateTestConfig { max_candidates: 4 },
    )
    .expect("run mixed-membership GPT oracle");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile mixed-membership epistemic executable");
    assert_eq!(executable.gpu_plan.epistemic_literals.len(), 2);

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 2]));
    executor.put_relation("alt", upload_unary_u32(&fix.memory, &[2, 3]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 4,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute mixed-membership epistemic fixture");

    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 2);
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
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
    assert_eq!(result.final_tuple_materialization.row_filter_count, 2);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        0
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![2],
        "mixed know/possible membership must keep only rows accepted by both bound tuple keys"
    );
}

#[test]
fn accepted_negated_mixed_memberships_match_gpt_oracle_parity() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred blocked(u32).
        pred accepted(u32).
        accepted(X) :- node(X), not know edge(X), not possible blocked(X).
        "#,
    )
    .expect("parse negated mixed-membership epistemic fixture");
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            // Candidate index 0 represents both negated literals as false.
            EpistemicInterpretation::new()
                .with_known("edge", 1)
                .with_known("blocked", 1),
            // Candidate index 1 represents not-know false and not-possible true.
            EpistemicInterpretation::new().with_known("edge", 1),
            // Candidate index 2 represents not-know true and not-possible false.
            EpistemicInterpretation::new().with_known("blocked", 1),
            // Candidate index 3 represents both negated literals as true.
            EpistemicInterpretation::new(),
        ],
        GeneratePropagateTestConfig { max_candidates: 4 },
    )
    .expect("run negated mixed-membership GPT oracle");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile negated mixed-membership epistemic executable");
    assert_eq!(executable.gpu_plan.epistemic_literals.len(), 2);
    assert!(
        executable
            .gpu_plan
            .tuple_membership_bindings
            .iter()
            .all(|binding| binding.negated),
        "both mixed negated memberships must compile as negated tuple-key filters"
    );

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1]));
    executor.put_relation("blocked", upload_unary_u32(&fix.memory, &[3]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 4,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute negated mixed-membership epistemic fixture");

    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 2);
    assert_eq!(result.prepared.preflight.know_operator_count, 0);
    assert_eq!(result.prepared.preflight.possible_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
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
    assert_eq!(result.final_tuple_materialization.row_filter_count, 2);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        2
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![2],
        "negated mixed membership must keep only rows absent from both bound tuple-key sources"
    );
}

#[test]
fn accepted_all_operator_mixed_memberships_match_gpt_oracle_parity() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred alt(u32).
        pred hidden(u32).
        pred blocked(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X), possible alt(X), not know hidden(X), not possible blocked(X).
        "#,
    )
    .expect("parse all-operator mixed-membership epistemic fixture");
    let oracle_candidates = (0..16)
        .map(|bits| {
            let mut interpretation = EpistemicInterpretation::new();
            if bits & 0b0001 != 0 {
                interpretation = interpretation.with_known("edge", 1);
            }
            if bits & 0b0010 != 0 {
                interpretation = interpretation.with_known("alt", 1);
            }
            if bits & 0b0100 == 0 {
                interpretation = interpretation.with_known("hidden", 1);
            }
            if bits & 0b1000 == 0 {
                interpretation = interpretation.with_known("blocked", 1);
            }
            interpretation
        })
        .collect();
    let oracle = run_generate_propagate_test(
        &program,
        oracle_candidates,
        GeneratePropagateTestConfig { max_candidates: 16 },
    )
    .expect("run all-operator mixed-membership GPT oracle");
    let result = execute_all_operator_mixed_membership_fixture(&fix);

    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 4);
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(result.semantic_trace.guesses, oracle.trace.guesses);
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.pruned_candidates, oracle.trace.pruned);
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
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
    assert_eq!(result.final_tuple_materialization.row_filter_count, 4);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        2
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![2],
        "all-operator mixed membership must conjunct positive and negated tuple-key filters"
    );
}

#[test]
fn world_view_validation_rejects_candidates_missing_one_required_membership() {
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
    .expect("parse multi-membership rejection fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile multi-membership rejection executable");
    assert_eq!(executable.gpu_plan.epistemic_literals.len(), 2);

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 4,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute multi-membership rejection fixture");

    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 2);
    assert_eq!(
        result.semantic_trace.accepted_world_views, 0,
        "world-view validation must reject candidates missing any required epistemic membership"
    );
    assert_eq!(result.semantic_trace.rejected_candidates, 4);
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        Vec::<u32>::new(),
        "final output must remain empty when no candidate satisfies the full world-view boundary"
    );
}

#[test]
fn rejected_gpu_execution_result_cannot_gate_solver_or_probability() {
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
    .expect("parse rejected world-view fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile rejected world-view executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 4,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute rejected world-view fixture");

    assert_eq!(result.semantic_trace.accepted_world_views, 0);
    assert_eq!(
        read_device_row_count(&fix.provider, &result.final_output).expect("final row count"),
        0
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert!(result.aggregate_kernel_timing().is_recorded());
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut solver_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = solver_adapter
        .new_workspace(sat_cnf.var_cap, sat_cnf.clause_cap)
        .expect("new rejected world-view workspace");

    let solver_err = match solver_adapter.solve_assumption_lifecycle_with_gpu_execution_result(
        &fix.provider,
        &result,
        &mut workspace,
        &[GpuSolverProductionLifecycleStep {
            cnf: &sat_cnf,
            branch_var_limit: &branch_limit,
            expectation: GpuSolverProductionExpectation::Sat,
        }],
    ) {
        Ok(_) => panic!("solver evidence must reject an execution with no accepted world view"),
        Err(err) => err,
    };
    assert!(
        format!("{solver_err}").contains("non-empty accepted GPU final output"),
        "unexpected solver rejection: {solver_err}"
    );

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut prob_adapter = EpistemicProbProductionAdapter::new(config);
    let prob_err = match prob_adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
        0.8::edge(1).
        query(edge(1)).
        "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known("edge", 1, true)],
        ) {
        Ok(_) => {
            panic!("probability evidence must reject an execution with no accepted world view")
        }
        Err(err) => err,
    };
    assert!(
        format!("{prob_err}").contains("non-empty accepted GPU final output"),
        "unexpected probability rejection: {prob_err}"
    );
}

#[test]
fn g91_self_supported_possible_reaches_gpu_runtime_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        #pragma epistemic_mode = g91
        pred p().
        p() :- possible p().
        "#,
    )
    .expect("parse G91 self-supported possible fixture");
    let oracle = run_generate_propagate_test_with_mode(
        &program,
        vec![
            EpistemicInterpretation::new(),
            EpistemicInterpretation::new().with_possible("p", 0),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
        EpistemicMode::G91,
    )
    .expect("run G91 compatibility oracle");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile G91 self-supported possible executable");

    assert_eq!(executable.gpu_plan.mode, EirEpistemicMode::G91);

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    // The reduced G91 program is an empty-body nullary fact; seed it through
    // the existing relation buffer path used by production fact loading.
    executor.put_relation("p", upload_nullary(&fix.memory, 1));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 1,
            },
        )
        .expect("execute G91 self-supported possible fixture");

    assert_eq!(
        result.prepared.preflight.epistemic_mode,
        EirEpistemicMode::G91
    );
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    assert_eq!(
        read_device_row_count(&fix.provider, &result.final_output).expect("final row count"),
        1,
        "explicit G91 compatibility should accept and materialize self-supported p()"
    );
}

#[test]
fn faeel_independently_founded_self_possible_reaches_gpu_runtime_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred seed().
        pred p().
        p() :- seed().
        p() :- possible p().
        "#,
    )
    .expect("parse independently founded FAEEL fixture");
    let oracle = run_generate_propagate_test(
        &program,
        vec![
            EpistemicInterpretation::new(),
            EpistemicInterpretation::new().with_known("p", 0),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
    )
    .expect("run independently founded FAEEL oracle");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile independently founded FAEEL executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("seed", upload_nullary(&fix.memory, 1));
    executor.put_relation("p", upload_nullary(&fix.memory, 0));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 1,
            },
        )
        .expect("execute independently founded FAEEL fixture");

    assert_eq!(
        result.prepared.preflight.epistemic_mode,
        EirEpistemicMode::Faeel
    );
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    assert_eq!(
        result.semantic_trace.generated_candidates,
        oracle.trace.generated
    );
    assert_eq!(
        result.semantic_trace.propagated_candidates,
        oracle.trace.propagated
    );
    assert_eq!(result.semantic_trace.tested_candidates, oracle.trace.tested);
    assert_eq!(
        result.semantic_trace.accepted_candidates,
        oracle.trace.accepted
    );
    assert_eq!(
        result.semantic_trace.accepted_world_views,
        oracle.trace.accepted_world_views
    );
    assert_eq!(
        result.semantic_trace.rejected_candidates,
        oracle.trace.rejected
    );
    assert_eq!(
        result.semantic_trace.accepted_candidate_indices,
        oracle.accepted_candidate_indices
    );
    assert_eq!(
        result.semantic_trace.rejected_candidate_indices,
        oracle.rejected_candidate_indices
    );
    assert_eq!(
        read_device_row_count(&fix.provider, &result.final_output).expect("final row count"),
        1,
        "independently founded self-possible FAEEL fixture should materialize p()"
    );
}

#[test]
fn accepted_g91_and_faeel_modes_gate_probabilistic_production_trace() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let g91_program = parse_program(
        r#"
        #pragma epistemic_mode = g91
        pred p().
        p() :- possible p().
        "#,
    )
    .expect("parse G91 self-supported possible fixture");
    let g91_executable = compile_epistemic_gpu_execution_with_stats_snapshot(&g91_program, None)
        .expect("compile G91 self-supported possible executable");
    let mut g91_executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &g91_executable.relation_ids {
        g91_executor.register_relation(*rel_id, name);
    }
    g91_executor.put_relation("p", upload_nullary(&fix.memory, 1));
    let g91_result = g91_executor
        .execute_epistemic_gpu_execution(
            &g91_executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 1,
            },
        )
        .expect("execute G91 self-supported possible fixture");
    assert_eq!(
        g91_result.prepared.preflight.epistemic_mode,
        EirEpistemicMode::G91
    );

    let faeel_program = parse_program(
        r#"
        pred seed().
        pred p().
        p() :- seed().
        p() :- possible p().
        "#,
    )
    .expect("parse independently founded FAEEL fixture");
    let faeel_executable =
        compile_epistemic_gpu_execution_with_stats_snapshot(&faeel_program, None)
            .expect("compile independently founded FAEEL executable");
    let mut faeel_executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &faeel_executable.relation_ids {
        faeel_executor.register_relation(*rel_id, name);
    }
    faeel_executor.put_relation("seed", upload_nullary(&fix.memory, 1));
    faeel_executor.put_relation("p", upload_nullary(&fix.memory, 0));
    let faeel_result = faeel_executor
        .execute_epistemic_gpu_execution(
            &faeel_executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 1,
            },
        )
        .expect("execute independently founded FAEEL fixture");
    assert_eq!(
        faeel_result.prepared.preflight.epistemic_mode,
        EirEpistemicMode::Faeel
    );

    let g91_assumptions = [EpistemicAssumption::possible("p", 0, true)];
    let faeel_assumptions = [EpistemicAssumption::known("p", 0, true)];

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_for_gpu_execution_results(
            r#"
            0.5::p().
            query(p()).
            "#,
            &fix.provider,
            &[
                EpistemicProbGpuExecutionEvidence {
                    result: &g91_result,
                    assumptions: &g91_assumptions,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &faeel_result,
                    assumptions: &faeel_assumptions,
                },
            ],
        )
        .expect("accepted G91 and FAEEL GPU evidence must gate conditioned exact path");

    assert_eq!(evaluated.len(), 2);
    assert_eq!(evaluated[0].query_probs.len(), 1);
    assert!((evaluated[0].query_probs[0].prob - 1.0).abs() < 1.0e-6);
    assert_eq!(evaluated[1].query_probs.len(), 1);
    assert!((evaluated[1].query_probs[0].prob - 1.0).abs() < 1.0e-6);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_g91_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 0);
    assert_eq!(trace.gpu_exact_source_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_g91_and_faeel_modes_gate_solver_production_trace() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let g91_program = parse_program(
        r#"
        #pragma epistemic_mode = g91
        pred p().
        p() :- possible p().
        "#,
    )
    .expect("parse G91 self-supported possible fixture");
    let g91_executable = compile_epistemic_gpu_execution_with_stats_snapshot(&g91_program, None)
        .expect("compile G91 self-supported possible executable");
    let mut g91_executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &g91_executable.relation_ids {
        g91_executor.register_relation(*rel_id, name);
    }
    g91_executor.put_relation("p", upload_nullary(&fix.memory, 1));
    let g91_result = g91_executor
        .execute_epistemic_gpu_execution(
            &g91_executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 1,
            },
        )
        .expect("execute G91 self-supported possible fixture");
    assert_eq!(
        g91_result.prepared.preflight.epistemic_mode,
        EirEpistemicMode::G91
    );

    let faeel_program = parse_program(
        r#"
        pred seed().
        pred p().
        p() :- seed().
        p() :- possible p().
        "#,
    )
    .expect("parse independently founded FAEEL fixture");
    let faeel_executable =
        compile_epistemic_gpu_execution_with_stats_snapshot(&faeel_program, None)
            .expect("compile independently founded FAEEL executable");
    let mut faeel_executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &faeel_executable.relation_ids {
        faeel_executor.register_relation(*rel_id, name);
    }
    faeel_executor.put_relation("seed", upload_nullary(&fix.memory, 1));
    faeel_executor.put_relation("p", upload_nullary(&fix.memory, 0));
    let faeel_result = faeel_executor
        .execute_epistemic_gpu_execution(
            &faeel_executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 1,
            },
        )
        .expect("execute independently founded FAEEL fixture");
    assert_eq!(
        faeel_result.prepared.preflight.epistemic_mode,
        EirEpistemicMode::Faeel
    );

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
        .solve_multi_candidate_assumption_lifecycle_with_gpu_execution_results(
            &fix.provider,
            &[&g91_result, &faeel_result],
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
        .expect("accepted G91 and FAEEL GPU evidence must gate solver lifecycle path");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.steps, 4);
    assert_eq!(report.assumption_pushes, 4);
    assert_eq!(report.assumption_retractions, 4);
    assert_eq!(report.workspace_reuses, 2);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.accepted_g91_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_faeel_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_assumption_pushes, 4);
    assert_eq!(trace.gpu_assumption_retractions, 4);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 2);
    assert_eq!(trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
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
fn accepted_gpu_execution_batches_gate_probabilistic_exact_compile_paths() {
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
    let assumptions_b = [EpistemicAssumption::known("edge", 2, true)];
    let probabilistic_source = r#"
        0.5::rain().
        query(rain()).
        "#;
    let prob_program = parse_program(probabilistic_source).expect("parse probabilistic program");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;

    let mut source_adapter = EpistemicProbProductionAdapter::new(config);
    let source_exacts = source_adapter
        .compile_source_for_gpu_execution_results(
            probabilistic_source,
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
        .expect("accepted GPU runtime evidence must gate batched source exact compile");
    assert_eq!(source_exacts.len(), 2);
    let source_trace = source_adapter.trace();
    assert_eq!(source_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(source_trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(source_trace.gpu_exact_source_compiles, 2);
    assert_eq!(source_trace.gpu_exact_program_compiles, 0);
    assert_eq!(source_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(source_trace.fixture_circuit_evaluations, 0);

    let mut program_adapter = EpistemicProbProductionAdapter::new(config);
    let program_exacts = program_adapter
        .compile_program_for_gpu_execution_results(
            &prob_program,
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
        .expect("accepted GPU runtime evidence must gate batched parsed-program exact compile");
    assert_eq!(program_exacts.len(), 2);
    let program_trace = program_adapter.trace();
    assert_eq!(program_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(program_trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(program_trace.gpu_exact_source_compiles, 0);
    assert_eq!(program_trace.gpu_exact_program_compiles, 2);
    assert_eq!(program_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(program_trace.fixture_circuit_evaluations, 0);

    let mut batch_executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        batch_executor.register_relation(*rel_id, name);
    }
    batch_executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2]));
    batch_executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1]));
    let executables = [&executable, &executable];
    let batch = batch_executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute accepted epistemic batch fixture");
    let batch_assumptions = [EpistemicAssumption::known("edge", 1, true)];
    let assumption_groups: [&[EpistemicAssumption]; 2] = [&batch_assumptions, &batch_assumptions];

    let mut source_batch_adapter = EpistemicProbProductionAdapter::new(config);
    let source_batch_exacts = source_batch_adapter
        .compile_source_for_gpu_batch_execution_result(
            probabilistic_source,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted GPU batch evidence must gate source exact compile");
    assert_eq!(source_batch_exacts.len(), 2);
    let source_batch_trace = source_batch_adapter.trace();
    assert_eq!(source_batch_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        source_batch_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(source_batch_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(source_batch_trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(source_batch_trace.gpu_exact_source_compiles, 2);
    assert_eq!(source_batch_trace.gpu_exact_program_compiles, 0);
    assert_eq!(source_batch_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(source_batch_trace.fixture_circuit_evaluations, 0);

    let mut program_batch_adapter = EpistemicProbProductionAdapter::new(config);
    let program_batch_exacts = program_batch_adapter
        .compile_program_for_gpu_batch_execution_result(
            &prob_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted GPU batch evidence must gate parsed-program exact compile");
    assert_eq!(program_batch_exacts.len(), 2);
    let program_batch_trace = program_batch_adapter.trace();
    assert_eq!(program_batch_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        program_batch_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(program_batch_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(
        program_batch_trace.accepted_evidence_assumptions_consumed,
        2
    );
    assert_eq!(program_batch_trace.gpu_exact_source_compiles, 0);
    assert_eq!(program_batch_trace.gpu_exact_program_compiles, 2);
    assert_eq!(program_batch_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(program_batch_trace.fixture_circuit_evaluations, 0);
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
fn accepted_gpu_execution_results_gate_batched_probabilistic_query_evaluations() {
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
    let assumptions_b = [EpistemicAssumption::known("edge", 2, true)];

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
            &result_a,
            vec![EpistemicAssumption::known("edge", 1, true)],
        )
        .expect("accepted GPU runtime evidence must gate probabilistic exact compile");
    let evaluated = adapter
        .evaluate_for_gpu_execution_results(
            &exact,
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
        .expect("accepted GPU runtime evidence must gate batched query evaluation");

    assert_eq!(evaluated.len(), 2);
    for result in &evaluated {
        assert_eq!(result.query_probs.len(), 1);
        assert!((result.query_probs[0].prob - 0.5).abs() < 1.0e-6);
    }
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 3);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 0);
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
fn accepted_gpu_execution_results_gate_batched_probabilistic_program_knowledge_compilation_path() {
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
    let prob_program = parse_program(
        r#"
        0.25::rain().
        query(rain()).
        "#,
    )
    .expect("parse probabilistic program");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_program_for_gpu_execution_results(
            &prob_program,
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
        .expect(
            "accepted GPU runtime evidence must gate batched parsed-program knowledge compilation",
        );

    assert_eq!(evaluated.len(), 2);
    for result in &evaluated {
        assert_eq!(result.query_probs.len(), 1);
        assert!((result.query_probs[0].prob - 0.25).abs() < 1.0e-6);
    }
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 0);
    assert_eq!(trace.gpu_exact_program_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 2);
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
fn accepted_gpu_execution_result_conditions_nonzero_arity_probabilistic_evidence() {
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
        .expect("execute nonzero-arity accepted epistemic fixture");
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
            0.4::edge(1).
            query(edge(1)).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known_tuple(
                "edge",
                vec![EpistemicEvidenceTerm::integer(1)],
                true,
            )],
        )
        .expect("accepted GPU runtime evidence must condition nonzero tuple evidence");

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!(
        (evaluated.query_probs[0].prob - 1.0).abs() < 1.0e-6,
        "accepted nonzero-arity know edge(1) evidence must condition query probability to true"
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
fn accepted_gpu_execution_result_updates_incremental_probability_circuit() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
        &[1, 2],
        &[1],
    );
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![1]
    );

    let assumption =
        EpistemicAssumption::known_tuple("edge", vec![EpistemicEvidenceTerm::integer(1)], true);
    let mut circuit = EpistemicCircuit::compile(
        0.15,
        vec![(assumption.clone(), 0.9)],
        KnowledgeCompilerAdapter::gpu_d4(),
    )
    .expect("compile bounded incremental probability circuit");
    let original_fingerprint = circuit.circuit_fingerprint();

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let update = adapter
        .apply_accepted_world_view_to_circuit_with_gpu_execution_result(
            &mut circuit,
            &fix.provider,
            &result,
            vec![assumption],
        )
        .expect("accepted GPU evidence must update incremental circuit evidence");

    assert_eq!(update.mode, CircuitUpdateMode::IncrementalEvidence);
    assert_eq!(update.compile_count, 1);
    assert_eq!(update.circuit_fingerprint, original_fingerprint);
    assert_eq!(circuit.incremental_update_count(), 1);
    assert_eq!(
        circuit.compiler_evidence_literals(),
        vec!["know:edge/1(1)=true".to_string()]
    );
    assert!(circuit.query_probability().within_tolerance(0.9));

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.accepted_incremental_circuit_updates, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 0);
    assert_eq!(trace.gpu_exact_program_compiles, 0);
    assert_eq!(trace.gpu_exact_query_evaluations, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
    let metric_err = trace
        .require_production_metric_eligibility()
        .expect_err("incremental fixture circuit update alone must not satisfy production metrics");
    assert!(format!("{metric_err}")
        .contains("existing GPU exact/provenance/PIR/CNF/knowledge-compilation counter"));
}

#[test]
fn accepted_ternary_probabilistic_evidence_records_nonzero_arity_trace() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred triple(u32, u32, u32).
        pred fact3(u32, u32, u32).
        pred accepted(u32, u32, u32).
        accepted(A, B, C) :- triple(A, B, C), know fact3(A, B, C).
        "#,
    )
    .expect("parse ternary probabilistic evidence fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile ternary probabilistic evidence executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "triple",
        upload_ternary_u32(&fix.memory, &[(1, 2, 3), (2, 3, 5), (8, 13, 21)]),
    );
    executor.put_relation("fact3", upload_ternary_u32(&fix.memory, &[(2, 3, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute ternary probabilistic evidence fixture");
    assert_eq!(
        download_ternary_u32(&fix.provider, &result.final_output),
        vec![(2, 3, 5)]
    );

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
            0.8::fact3(2, 3, 5).
            query(fact3(2, 3, 5)).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known_tuple(
                "fact3",
                vec![
                    EpistemicEvidenceTerm::integer(2),
                    EpistemicEvidenceTerm::integer(3),
                    EpistemicEvidenceTerm::integer(5),
                ],
                true,
            )],
        )
        .expect("accepted ternary GPU evidence must condition exact evidence");

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!(
        (evaluated.query_probs[0].prob - 1.0).abs() < 1.0e-6,
        "accepted ternary know fact3(2, 3, 5) evidence must condition query probability to true"
    );
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(
        trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        0
    );
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 3);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 3);
    assert_eq!(trace.gpu_program_conditioned_max_evidence_arity, 0);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_quaternary_source_probabilistic_evidence_records_nonzero_arity_trace() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary source probabilistic evidence fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary source probabilistic evidence executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary source probabilistic evidence fixture");
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(2, 3, 4, 5)]
    );

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
            0.8::fact4(2, 3, 4, 5).
            query(fact4(2, 3, 4, 5)).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known_tuple(
                "fact4",
                vec![
                    EpistemicEvidenceTerm::integer(2),
                    EpistemicEvidenceTerm::integer(3),
                    EpistemicEvidenceTerm::integer(4),
                    EpistemicEvidenceTerm::integer(5),
                ],
                true,
            )],
        )
        .expect("accepted quaternary GPU evidence must condition source exact evidence");

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!(
        (evaluated.query_probs[0].prob - 1.0).abs() < 1.0e-6,
        "accepted quaternary know fact4(2, 3, 4, 5) evidence must condition source query probability to true"
    );
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(
        trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        0
    );
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_program_conditioned_max_evidence_arity, 0);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_program_compiles, 0);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_quaternary_parsed_program_probabilistic_evidence_records_nonzero_arity_trace() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary parsed-program probabilistic evidence fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary parsed-program probabilistic evidence executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary parsed-program probabilistic evidence fixture");
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(2, 3, 4, 5)]
    );

    let prob_program = parse_program(
        r#"
        0.8::fact4(2, 3, 4, 5).
        query(fact4(2, 3, 4, 5)).
        "#,
    )
    .expect("parse quaternary probabilistic program");
    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known_tuple(
                "fact4",
                vec![
                    EpistemicEvidenceTerm::integer(2),
                    EpistemicEvidenceTerm::integer(3),
                    EpistemicEvidenceTerm::integer(4),
                    EpistemicEvidenceTerm::integer(5),
                ],
                true,
            )],
        )
        .expect("accepted quaternary GPU evidence must condition parsed exact evidence");

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!(
        (evaluated.query_probs[0].prob - 1.0).abs() < 1.0e-6,
        "accepted quaternary know fact4(2, 3, 4, 5) evidence must condition parsed query probability to true"
    );
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 0);
    assert_eq!(
        trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        1
    );
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 0);
    assert_eq!(trace.gpu_program_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_exact_program_compiles, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 0);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_quaternary_gpu_execution_result_conditions_source_and_program_probabilistic_gradients()
{
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary conditioned-gradient probabilistic evidence fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary conditioned-gradient probabilistic evidence executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary conditioned-gradient probabilistic evidence fixture");
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(2, 3, 4, 5)]
    );
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let probabilistic_source = r#"
        0.8::fact4(2, 3, 4, 5).
        query(fact4(2, 3, 4, 5)).
        "#;
    let prob_program = parse_program(probabilistic_source)
        .expect("parse quaternary gradient probabilistic program");
    let assumptions = vec![EpistemicAssumption::known_tuple(
        "fact4",
        vec![
            EpistemicEvidenceTerm::integer(2),
            EpistemicEvidenceTerm::integer(3),
            EpistemicEvidenceTerm::integer(4),
            EpistemicEvidenceTerm::integer(5),
        ],
        true,
    )];

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let source_grads = adapter
        .compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &result,
            assumptions.clone(),
        )
        .expect("accepted quaternary know evidence must condition source gradients");
    let program_grads = adapter
        .compile_and_evaluate_conditioned_program_with_grads_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            assumptions,
        )
        .expect("accepted quaternary know evidence must condition parsed-program gradients");

    assert_eq!(source_grads.query_grads.len(), 1);
    assert_eq!(program_grads.query_grads.len(), 1);
    assert!((source_grads.query_grads[0].prob - 1.0).abs() < 1.0e-6);
    assert!((program_grads.query_grads[0].prob - 1.0).abs() < 1.0e-6);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 0);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_source_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_program_conditioned_evidence_facts, 1);
    assert_eq!(
        trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        1
    );
    assert_eq!(trace.gpu_program_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_program_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_program_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 0);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(trace.gpu_source_conditioned_gradient_evaluations, 1);
    assert_eq!(trace.gpu_program_conditioned_gradient_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_quaternary_gpu_execution_result_gates_source_and_program_pir_cnf_and_exact_evaluation_paths(
) {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary deep probabilistic evidence fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary deep probabilistic evidence executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary deep probabilistic evidence fixture");
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(2, 3, 4, 5)]
    );
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        0
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);

    let assumptions = [EpistemicAssumption::known_tuple(
        "fact4",
        vec![
            EpistemicEvidenceTerm::integer(2),
            EpistemicEvidenceTerm::integer(3),
            EpistemicEvidenceTerm::integer(4),
            EpistemicEvidenceTerm::integer(5),
        ],
        true,
    )];
    let probabilistic_source = r#"
        0.8::fact4(2, 3, 4, 5).
        query(fact4(2, 3, 4, 5)).
        "#;
    let prob_program =
        parse_program(probabilistic_source).expect("parse quaternary deep probabilistic program");

    let assert_single_probability = |actual: f64| {
        assert!(
            (actual - 0.8).abs() < 1.0e-6,
            "already-compiled quaternary know evaluation must preserve unconditioned probability: actual={actual}"
        );
    };

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;

    let mut pir_adapter = EpistemicProbProductionAdapter::new(config);
    let source_pir_cnf = pir_adapter
        .encode_source_pir_cnf_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary know evidence must gate source PIR/CNF");
    let program_pir_cnf = pir_adapter
        .encode_program_pir_cnf_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary know evidence must gate parsed-program PIR/CNF");
    for pir_cnf in [&source_pir_cnf, &program_pir_cnf] {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let pir_trace = pir_adapter.trace();
    assert_eq!(pir_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(pir_trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(pir_trace.gpu_pir_graph_uploads, 2);
    assert_eq!(pir_trace.gpu_source_pir_graph_uploads, 1);
    assert_eq!(pir_trace.gpu_program_pir_graph_uploads, 1);
    assert_eq!(pir_trace.gpu_cnf_encodes, 2);
    assert_eq!(pir_trace.gpu_source_cnf_encodes, 1);
    assert_eq!(pir_trace.gpu_program_cnf_encodes, 1);
    assert_eq!(pir_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(pir_trace.fixture_circuit_evaluations, 0);

    let mut query_adapter = EpistemicProbProductionAdapter::new(config);
    let source_exact = query_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary know evidence must gate source exact compile");
    let program_exact = query_adapter
        .compile_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary know evidence must gate parsed-program exact compile");
    let source_eval = query_adapter
        .evaluate_with_gpu_execution_result(
            &source_exact,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary know evidence must gate source exact query");
    let program_eval = query_adapter
        .evaluate_with_gpu_execution_result(
            &program_exact,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary know evidence must gate parsed-program exact query");
    assert_eq!(source_eval.query_probs.len(), 1);
    assert_eq!(program_eval.query_probs.len(), 1);
    assert_single_probability(source_eval.query_probs[0].prob);
    assert_single_probability(program_eval.query_probs[0].prob);
    let query_trace = query_adapter.trace();
    assert_eq!(query_trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(query_trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(query_trace.gpu_exact_source_compiles, 1);
    assert_eq!(query_trace.gpu_exact_program_compiles, 1);
    assert_eq!(query_trace.gpu_exact_query_evaluations, 2);
    assert_eq!(query_trace.gpu_exact_gradient_evaluations, 0);
    assert_eq!(query_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(query_trace.fixture_circuit_evaluations, 0);

    let mut gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let source_exact = gradient_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary know evidence must gate source exact compile");
    let program_exact = gradient_adapter
        .compile_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary know evidence must gate parsed-program exact compile");
    let source_grad = gradient_adapter
        .evaluate_gpu_with_grads_with_gpu_execution_result(
            &source_exact,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary know evidence must gate source gradients");
    let program_grad = gradient_adapter
        .evaluate_gpu_with_grads_with_gpu_execution_result(
            &program_exact,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary know evidence must gate parsed-program gradients");
    assert_eq!(source_grad.query_grads.len(), 1);
    assert_eq!(program_grad.query_grads.len(), 1);
    assert_single_probability(source_grad.query_grads[0].prob);
    assert_single_probability(program_grad.query_grads[0].prob);
    let gradient_trace = gradient_adapter.trace();
    assert_eq!(gradient_trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(gradient_trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(gradient_trace.gpu_exact_source_compiles, 1);
    assert_eq!(gradient_trace.gpu_exact_program_compiles, 1);
    assert_eq!(gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(gradient_trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(gradient_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(gradient_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_quaternary_not_possible_probabilistic_evidence_records_negative_nonzero_arity_trace() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary not-possible probabilistic evidence fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary not-possible probabilistic evidence executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary not-possible probabilistic evidence fixture");
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(1, 2, 3, 4), (9, 9, 9, 9)]
    );
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
            0.8::fact4(1, 2, 3, 4).
            query(fact4(1, 2, 3, 4)).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::possible_tuple(
                "fact4",
                vec![
                    EpistemicEvidenceTerm::integer(1),
                    EpistemicEvidenceTerm::integer(2),
                    EpistemicEvidenceTerm::integer(3),
                    EpistemicEvidenceTerm::integer(4),
                ],
                false,
            )],
        )
        .expect(
            "accepted quaternary not-possible GPU evidence must condition source exact evidence",
        );

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!(
        evaluated.query_probs[0].prob.abs() < 1.0e-6,
        "accepted quaternary not possible fact4(1, 2, 3, 4) evidence must condition source query probability to false"
    );
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(
        trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        0
    );
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_program_conditioned_max_evidence_arity, 0);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 0);
    assert_eq!(trace.gpu_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_program_compiles, 0);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_quaternary_not_possible_parsed_program_probabilistic_evidence_records_negative_nonzero_arity_trace(
) {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary not-possible parsed-program probabilistic evidence fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary not-possible parsed-program probabilistic evidence executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary not-possible parsed-program probabilistic evidence fixture");
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(1, 2, 3, 4), (9, 9, 9, 9)]
    );
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );

    let prob_program = parse_program(
        r#"
        0.8::fact4(1, 2, 3, 4).
        query(fact4(1, 2, 3, 4)).
        "#,
    )
    .expect("parse quaternary not-possible probabilistic program");
    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::possible_tuple(
                "fact4",
                vec![
                    EpistemicEvidenceTerm::integer(1),
                    EpistemicEvidenceTerm::integer(2),
                    EpistemicEvidenceTerm::integer(3),
                    EpistemicEvidenceTerm::integer(4),
                ],
                false,
            )],
        )
        .expect(
            "accepted quaternary not-possible GPU evidence must condition parsed exact evidence",
        );

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!(
        evaluated.query_probs[0].prob.abs() < 1.0e-6,
        "accepted quaternary not possible fact4(1, 2, 3, 4) evidence must condition parsed query probability to false"
    );
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 0);
    assert_eq!(
        trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        1
    );
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 0);
    assert_eq!(trace.gpu_program_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 0);
    assert_eq!(trace.gpu_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_program_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_program_compiles, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 0);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_quaternary_not_possible_conditions_source_and_program_probabilistic_gradients() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary not-possible conditioned-gradient fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary not-possible conditioned-gradient executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary not-possible conditioned-gradient fixture");
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(1, 2, 3, 4), (9, 9, 9, 9)]
    );
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let probabilistic_source = r#"
        0.8::fact4(1, 2, 3, 4).
        query(fact4(1, 2, 3, 4)).
        "#;
    let prob_program = parse_program(probabilistic_source)
        .expect("parse quaternary not-possible gradient probabilistic program");
    let assumptions = vec![EpistemicAssumption::possible_tuple(
        "fact4",
        vec![
            EpistemicEvidenceTerm::integer(1),
            EpistemicEvidenceTerm::integer(2),
            EpistemicEvidenceTerm::integer(3),
            EpistemicEvidenceTerm::integer(4),
        ],
        false,
    )];

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let source_grads = adapter
        .compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &result,
            assumptions.clone(),
        )
        .expect("accepted quaternary not-possible evidence must condition source gradients");
    let program_grads = adapter
        .compile_and_evaluate_conditioned_program_with_grads_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            assumptions,
        )
        .expect(
            "accepted quaternary not-possible evidence must condition parsed-program gradients",
        );

    assert_eq!(source_grads.query_grads.len(), 1);
    assert_eq!(program_grads.query_grads.len(), 1);
    assert!(source_grads.query_grads[0].prob.abs() < 1.0e-6);
    assert!(program_grads.query_grads[0].prob.abs() < 1.0e-6);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 0);
    assert_eq!(trace.gpu_conditioned_not_possible_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_source_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_program_conditioned_evidence_facts, 1);
    assert_eq!(
        trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        1
    );
    assert_eq!(trace.gpu_program_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_program_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_program_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_program_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 0);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(trace.gpu_source_conditioned_gradient_evaluations, 1);
    assert_eq!(trace.gpu_program_conditioned_gradient_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_quaternary_not_possible_gates_source_and_program_pir_cnf_and_exact_evaluation_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary not-possible deep probabilistic evidence fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary not-possible deep probabilistic evidence executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary not-possible deep probabilistic evidence fixture");
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(1, 2, 3, 4), (9, 9, 9, 9)]
    );
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);

    let assumptions = [EpistemicAssumption::possible_tuple(
        "fact4",
        vec![
            EpistemicEvidenceTerm::integer(1),
            EpistemicEvidenceTerm::integer(2),
            EpistemicEvidenceTerm::integer(3),
            EpistemicEvidenceTerm::integer(4),
        ],
        false,
    )];
    let probabilistic_source = r#"
        0.8::fact4(1, 2, 3, 4).
        query(fact4(1, 2, 3, 4)).
        "#;
    let prob_program = parse_program(probabilistic_source)
        .expect("parse quaternary not-possible deep probabilistic program");

    let assert_single_probability = |actual: f64| {
        assert!(
            (actual - 0.8).abs() < 1.0e-6,
            "already-compiled quaternary not-possible evaluation must preserve unconditioned probability: actual={actual}"
        );
    };

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;

    let mut pir_adapter = EpistemicProbProductionAdapter::new(config);
    let source_pir_cnf = pir_adapter
        .encode_source_pir_cnf_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary not-possible evidence must gate source PIR/CNF");
    let program_pir_cnf = pir_adapter
        .encode_program_pir_cnf_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary not-possible evidence must gate parsed-program PIR/CNF");
    for pir_cnf in [&source_pir_cnf, &program_pir_cnf] {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let pir_trace = pir_adapter.trace();
    assert_eq!(pir_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(pir_trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(pir_trace.gpu_pir_graph_uploads, 2);
    assert_eq!(pir_trace.gpu_source_pir_graph_uploads, 1);
    assert_eq!(pir_trace.gpu_program_pir_graph_uploads, 1);
    assert_eq!(pir_trace.gpu_cnf_encodes, 2);
    assert_eq!(pir_trace.gpu_source_cnf_encodes, 1);
    assert_eq!(pir_trace.gpu_program_cnf_encodes, 1);
    assert_eq!(pir_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(pir_trace.fixture_circuit_evaluations, 0);

    let mut query_adapter = EpistemicProbProductionAdapter::new(config);
    let source_exact = query_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary not-possible evidence must gate source exact compile");
    let program_exact = query_adapter
        .compile_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary not-possible evidence must gate parsed-program exact compile");
    let source_eval = query_adapter
        .evaluate_with_gpu_execution_result(
            &source_exact,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary not-possible evidence must gate source exact query");
    let program_eval = query_adapter
        .evaluate_with_gpu_execution_result(
            &program_exact,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary not-possible evidence must gate parsed-program exact query");
    assert_eq!(source_eval.query_probs.len(), 1);
    assert_eq!(program_eval.query_probs.len(), 1);
    assert_single_probability(source_eval.query_probs[0].prob);
    assert_single_probability(program_eval.query_probs[0].prob);
    let query_trace = query_adapter.trace();
    assert_eq!(query_trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(query_trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(query_trace.gpu_exact_source_compiles, 1);
    assert_eq!(query_trace.gpu_exact_program_compiles, 1);
    assert_eq!(query_trace.gpu_exact_query_evaluations, 2);
    assert_eq!(query_trace.gpu_exact_gradient_evaluations, 0);
    assert_eq!(query_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(query_trace.fixture_circuit_evaluations, 0);

    let mut gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let source_exact = gradient_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary not-possible evidence must gate source exact compile");
    let program_exact = gradient_adapter
        .compile_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary not-possible evidence must gate parsed-program exact compile");
    let source_grad = gradient_adapter
        .evaluate_gpu_with_grads_with_gpu_execution_result(
            &source_exact,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary not-possible evidence must gate source gradients");
    let program_grad = gradient_adapter
        .evaluate_gpu_with_grads_with_gpu_execution_result(
            &program_exact,
            &fix.provider,
            &result,
            assumptions.to_vec(),
        )
        .expect("accepted quaternary not-possible evidence must gate parsed-program gradients");
    assert_eq!(source_grad.query_grads.len(), 1);
    assert_eq!(program_grad.query_grads.len(), 1);
    assert_single_probability(source_grad.query_grads[0].prob);
    assert_single_probability(program_grad.query_grads[0].prob);
    let gradient_trace = gradient_adapter.trace();
    assert_eq!(gradient_trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(gradient_trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(gradient_trace.gpu_exact_source_compiles, 1);
    assert_eq!(gradient_trace.gpu_exact_program_compiles, 1);
    assert_eq!(gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(gradient_trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(gradient_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(gradient_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_quaternary_possible_and_not_know_results_gate_solver_and_probabilistic_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let run_case = |label: &str,
                    program_source: &str,
                    expected_rows: Vec<(u32, u32, u32, u32)>,
                    expected_possible: usize,
                    expected_not_know: usize,
                    expected_negated_filters: usize|
     -> EpistemicGpuExecutionResult {
        let program = parse_program(program_source).expect(label);
        let executable =
            compile_epistemic_gpu_execution_with_stats_snapshot(&program, None).expect(label);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings.len(), 1);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings[0].arity, 4);
        assert_eq!(
            executable.gpu_plan.tuple_membership_bindings[0].bound_output_columns,
            vec![Some(0), Some(1), Some(2), Some(3)]
        );
        assert_eq!(
            executable.gpu_plan.tuple_membership_bindings[0].negated,
            expected_negated_filters == 1
        );

        let mut executor =
            Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
        for (name, rel_id) in &executable.relation_ids {
            executor.register_relation(*rel_id, name);
        }
        executor.put_relation(
            "tuple4",
            upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
        );
        executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

        let result = executor
            .execute_epistemic_gpu_execution(
                &executable,
                EpistemicGpuWorkspaceCapacities {
                    max_candidates: 2,
                    max_worlds: 1,
                    max_models_per_reduction: 3,
                },
            )
            .expect(label);

        assert_eq!(result.prepared.preflight.know_operator_count, 0);
        assert_eq!(
            result.prepared.preflight.possible_operator_count,
            expected_possible
        );
        assert_eq!(
            result.prepared.preflight.not_know_operator_count,
            expected_not_know
        );
        assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
        assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
        assert_eq!(
            result.final_tuple_materialization.negated_row_filter_count,
            expected_negated_filters
        );
        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            4
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
        assert_eq!(
            download_quaternary_u32(&fix.provider, &result.final_output),
            expected_rows,
            "{label} must filter by the accepted arity-4 tuple key evidence"
        );
        result
    };

    let possible_result = run_case(
        "quaternary possible production reuse",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), possible fact4(A, B, C, D).
        "#,
        vec![(2, 3, 4, 5)],
        1,
        0,
        0,
    );
    let not_know_result = run_case(
        "quaternary not-know production reuse",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not know fact4(A, B, C, D).
        "#,
        vec![(1, 2, 3, 4), (9, 9, 9, 9)],
        0,
        1,
        1,
    );

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let mut solver =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let possible_assignment = solver
        .solve_expect_sat_with_gpu_execution_result(&fix.provider, &possible_result, &sat_cnf)
        .expect("accepted quaternary possible GPU evidence must gate solver SAT path");
    assert_ne!(*possible_assignment.device_ptr(), 0);
    let not_know_assignment = solver
        .solve_expect_sat_with_gpu_execution_result(&fix.provider, &not_know_result, &sat_cnf)
        .expect("accepted quaternary not-know GPU evidence must gate solver SAT path");
    assert_ne!(*not_know_assignment.device_ptr(), 0);

    let solver_trace = solver.trace();
    assert_eq!(solver_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        solver_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        solver_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        solver_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        solver_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        solver_trace.accepted_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        solver_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(solver_trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(solver_trace.cpu_assignment_enumerations, 0);
    assert_eq!(solver_trace.cpu_maxsat_enumerations, 0);

    let possible_assumptions = [EpistemicAssumption::possible_tuple(
        "fact4",
        vec![
            EpistemicEvidenceTerm::integer(2),
            EpistemicEvidenceTerm::integer(3),
            EpistemicEvidenceTerm::integer(4),
            EpistemicEvidenceTerm::integer(5),
        ],
        true,
    )];
    let not_know_assumptions = [EpistemicAssumption::known_tuple(
        "fact4",
        vec![
            EpistemicEvidenceTerm::integer(1),
            EpistemicEvidenceTerm::integer(2),
            EpistemicEvidenceTerm::integer(3),
            EpistemicEvidenceTerm::integer(4),
        ],
        false,
    )];

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut prob = EpistemicProbProductionAdapter::new(config);
    let evaluated = prob
        .compile_and_evaluate_conditioned_source_for_gpu_execution_results(
            r#"
            0.8::fact4(2, 3, 4, 5).
            0.6::fact4(1, 2, 3, 4).
            query(fact4(2, 3, 4, 5)).
            query(fact4(1, 2, 3, 4)).
            "#,
            &fix.provider,
            &[
                EpistemicProbGpuExecutionEvidence {
                    result: &possible_result,
                    assumptions: &possible_assumptions,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &not_know_result,
                    assumptions: &not_know_assumptions,
                },
            ],
        )
        .expect("accepted quaternary possible/not-know GPU evidence must gate probability path");

    assert_eq!(evaluated.len(), 2);
    assert_eq!(evaluated[0].query_probs.len(), 2);
    assert_eq!(evaluated[1].query_probs.len(), 2);
    assert!(
        (evaluated[0].query_probs[0].prob - 1.0).abs() < 1.0e-6,
        "accepted possible fact4(2, 3, 4, 5) evidence must condition probability to true"
    );
    assert!(
        evaluated[1].query_probs[1].prob.abs() < 1.0e-6,
        "accepted not-know fact4(1, 2, 3, 4) evidence must condition probability to false"
    );

    let prob_trace = prob.trace();
    assert_eq!(prob_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(prob_trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(prob_trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(prob_trace.gpu_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(
        prob_trace.gpu_source_conditioned_nonzero_arity_evidence_facts,
        2
    );
    assert_eq!(
        prob_trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        0
    );
    assert_eq!(prob_trace.gpu_conditioned_max_evidence_arity, 4);
    assert_eq!(prob_trace.gpu_source_conditioned_max_evidence_arity, 4);
    assert_eq!(prob_trace.gpu_program_conditioned_max_evidence_arity, 0);
    assert_eq!(prob_trace.gpu_conditioned_negative_evidence_facts, 1);
    assert_eq!(prob_trace.gpu_conditioned_possible_evidence_facts, 1);
    assert_eq!(prob_trace.gpu_conditioned_not_known_evidence_facts, 1);
    assert_eq!(prob_trace.gpu_conditioned_know_evidence_facts, 0);
    assert_eq!(prob_trace.gpu_conditioned_not_possible_evidence_facts, 0);
    assert_eq!(prob_trace.gpu_source_conditioned_possible_evidence_facts, 1);
    assert_eq!(
        prob_trace.gpu_source_conditioned_not_known_evidence_facts,
        1
    );
    assert_eq!(prob_trace.gpu_exact_source_compiles, 2);
    assert_eq!(prob_trace.gpu_exact_program_compiles, 0);
    assert_eq!(prob_trace.gpu_exact_query_evaluations, 2);
    assert_eq!(prob_trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(
        prob_trace.gpu_source_knowledge_compilation_end_to_end_runs,
        2
    );
    assert_eq!(
        prob_trace.gpu_program_knowledge_compilation_end_to_end_runs,
        0
    );
    assert_eq!(prob_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(prob_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_quaternary_possible_and_not_know_results_gate_solver_reuse_maxsat_and_portfolio_paths()
{
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let run_case = |label: &str,
                    program_source: &str,
                    expected_rows: Vec<(u32, u32, u32, u32)>,
                    expected_possible: usize,
                    expected_not_know: usize,
                    expected_negated_filters: usize|
     -> EpistemicGpuExecutionResult {
        let program = parse_program(program_source).expect(label);
        let executable =
            compile_epistemic_gpu_execution_with_stats_snapshot(&program, None).expect(label);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings.len(), 1);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings[0].arity, 4);
        assert_eq!(
            executable.gpu_plan.tuple_membership_bindings[0].bound_output_columns,
            vec![Some(0), Some(1), Some(2), Some(3)]
        );
        assert_eq!(
            executable.gpu_plan.tuple_membership_bindings[0].negated,
            expected_negated_filters == 1
        );

        let mut executor =
            Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
        for (name, rel_id) in &executable.relation_ids {
            executor.register_relation(*rel_id, name);
        }
        executor.put_relation(
            "tuple4",
            upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
        );
        executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

        let result = executor
            .execute_epistemic_gpu_execution(
                &executable,
                EpistemicGpuWorkspaceCapacities {
                    max_candidates: 2,
                    max_worlds: 1,
                    max_models_per_reduction: 3,
                },
            )
            .expect(label);

        assert_eq!(result.prepared.preflight.know_operator_count, 0);
        assert_eq!(
            result.prepared.preflight.possible_operator_count,
            expected_possible
        );
        assert_eq!(
            result.prepared.preflight.not_know_operator_count,
            expected_not_know
        );
        assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
        assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
        assert_eq!(
            result.final_tuple_materialization.negated_row_filter_count,
            expected_negated_filters
        );
        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            4
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
        assert_eq!(
            download_quaternary_u32(&fix.provider, &result.final_output),
            expected_rows,
            "{label} must filter by the accepted arity-4 tuple key evidence"
        );
        result
    };

    let possible_result = run_case(
        "quaternary possible solver reuse",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), possible fact4(A, B, C, D).
        "#,
        vec![(2, 3, 4, 5)],
        1,
        0,
        0,
    );
    let not_know_result = run_case(
        "quaternary not-know solver reuse",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not know fact4(A, B, C, D).
        "#,
        vec![(1, 2, 3, 4), (9, 9, 9, 9)],
        0,
        1,
        1,
    );
    let accepted_results = [&possible_result, &not_know_result];

    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let source_cnf = GpuCnf::from_host(&unsat_instance, &fix.provider).expect("upload UNSAT CNF");
    let candidate_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let candidate_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let gpu_candidate_low =
        GpuCnf::from_host(&candidate_low, &fix.provider).expect("upload low-score MaxSAT CNF");
    let gpu_candidate_high =
        GpuCnf::from_host(&candidate_high, &fix.provider).expect("upload high-score MaxSAT CNF");
    let portfolio_sat =
        GpuCnf::from_host(&candidate_low, &fix.provider).expect("upload portfolio SAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(source_cnf.var_cap, source_cnf.clause_cap)
        .expect("new possible/not-know learned-clause reuse workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let learned = adapter
        .solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results(
            &fix.provider,
            &accepted_results,
            &mut workspace,
            &source_cnf,
            &branch_limit,
            &source_cnf,
            &branch_limit,
        )
        .expect("accepted possible/not-know GPU evidence must gate learned-clause reuse");
    let maxsat = adapter
        .solve_multi_candidate_weighted_maxsat_with_gpu_execution_results(
            &fix.provider,
            &accepted_results,
            &[
                GpuSolverProductionMaxSatCandidate {
                    score: 3,
                    cnf: &gpu_candidate_low,
                    branch_var_limit: &branch_limit,
                },
                GpuSolverProductionMaxSatCandidate {
                    score: 7,
                    cnf: &gpu_candidate_high,
                    branch_var_limit: &branch_limit,
                },
            ],
        )
        .expect("accepted possible/not-know GPU evidence must gate multi-candidate MaxSAT");
    let portfolio_candidates = [GpuSolverProductionMaxSatCandidate {
        score: 7,
        cnf: &gpu_candidate_high,
        branch_var_limit: &branch_limit,
    }];
    let portfolio = adapter
        .solve_multi_candidate_portfolio_with_gpu_execution_results(
            &fix.provider,
            &accepted_results,
            &[
                GpuSolverProductionPortfolioJob::Sat {
                    cnf: &portfolio_sat,
                    branch_var_limit: &branch_limit,
                },
                GpuSolverProductionPortfolioJob::MaxSat {
                    candidates: &portfolio_candidates,
                },
                GpuSolverProductionPortfolioJob::Unknown {
                    reason: "quaternary possible/not-know branch budget exhausted",
                },
                GpuSolverProductionPortfolioJob::Timeout { budget_micros: 10 },
            ],
        )
        .expect("accepted possible/not-know GPU evidence must gate multi-candidate portfolio path");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(learned.candidate_evidence_records, 2);
    assert_eq!(learned.candidates, 4);
    assert_eq!(learned.unsat_solves, 4);
    assert_eq!(learned.gpu_learned_clause_arena_publications, 2);
    assert_eq!(learned.gpu_learned_clause_imports, 2);
    assert_eq!(learned.gpu_learned_clause_reused_solves, 2);
    assert_eq!(learned.cpu_learned_clause_transfers, 0);
    assert_eq!(maxsat.candidate_evidence_records, 2);
    assert_eq!(maxsat.optimum_score, 7);
    assert_eq!(maxsat.candidates_checked, 4);
    assert_eq!(maxsat.gpu_cdcl_candidate_solves, 4);
    assert_eq!(portfolio.candidate_evidence_records, 2);
    assert_eq!(portfolio.jobs, 8);
    assert_eq!(portfolio.sat_jobs, 2);
    assert_eq!(portfolio.maxsat_jobs, 2);
    assert_eq!(portfolio.unknown_jobs, 2);
    assert_eq!(portfolio.timeout_jobs, 2);
    assert_eq!(portfolio.maxsat_optimum_scores, 14);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 6);
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        6
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        24
    );
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.accepted_possible_gpu_candidate_evidence_consumed, 3);
    assert_eq!(
        trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(trace.accepted_not_know_gpu_candidate_evidence_consumed, 3);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 4);
    assert_eq!(trace.gpu_learned_clause_arena_publications, 2);
    assert_eq!(trace.gpu_learned_clause_imports, 2);
    assert_eq!(trace.gpu_learned_clause_reused_solves, 2);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 6);
    assert_eq!(trace.gpu_maxsat_optima, 4);
    assert_eq!(trace.gpu_portfolio_jobs, 8);
    assert_eq!(trace.gpu_portfolio_sat_jobs, 2);
    assert_eq!(trace.gpu_portfolio_maxsat_jobs, 2);
    assert_eq!(trace.gpu_portfolio_unknown_status_jobs, 2);
    assert_eq!(trace.gpu_portfolio_timeout_status_jobs, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_quaternary_possible_and_not_know_results_gate_solver_search_and_scheduler_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let run_case = |label: &str,
                    program_source: &str,
                    expected_rows: Vec<(u32, u32, u32, u32)>,
                    expected_possible: usize,
                    expected_not_know: usize,
                    expected_negated_filters: usize|
     -> EpistemicGpuExecutionResult {
        let program = parse_program(program_source).expect(label);
        let executable =
            compile_epistemic_gpu_execution_with_stats_snapshot(&program, None).expect(label);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings.len(), 1);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings[0].arity, 4);
        assert_eq!(
            executable.gpu_plan.tuple_membership_bindings[0].bound_output_columns,
            vec![Some(0), Some(1), Some(2), Some(3)]
        );
        assert_eq!(
            executable.gpu_plan.tuple_membership_bindings[0].negated,
            expected_negated_filters == 1
        );

        let mut executor =
            Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
        for (name, rel_id) in &executable.relation_ids {
            executor.register_relation(*rel_id, name);
        }
        executor.put_relation(
            "tuple4",
            upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
        );
        executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

        let result = executor
            .execute_epistemic_gpu_execution(
                &executable,
                EpistemicGpuWorkspaceCapacities {
                    max_candidates: 2,
                    max_worlds: 1,
                    max_models_per_reduction: 3,
                },
            )
            .expect(label);

        assert_eq!(result.prepared.preflight.know_operator_count, 0);
        assert_eq!(
            result.prepared.preflight.possible_operator_count,
            expected_possible
        );
        assert_eq!(
            result.prepared.preflight.not_know_operator_count,
            expected_not_know
        );
        assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
        assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
        assert_eq!(
            result.final_tuple_materialization.negated_row_filter_count,
            expected_negated_filters
        );
        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            4
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
        assert_eq!(
            download_quaternary_u32(&fix.provider, &result.final_output),
            expected_rows,
            "{label} must filter by the accepted arity-4 tuple key evidence"
        );
        result
    };

    let possible_result = run_case(
        "quaternary possible solver search",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), possible fact4(A, B, C, D).
        "#,
        vec![(2, 3, 4, 5)],
        1,
        0,
        0,
    );
    let not_know_result = run_case(
        "quaternary not-know solver search",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not know fact4(A, B, C, D).
        "#,
        vec![(1, 2, 3, 4), (9, 9, 9, 9)],
        0,
        1,
        1,
    );
    let accepted_results = [&possible_result, &not_know_result];

    let sat_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let unsat = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let gpu_sat_low = GpuCnf::from_host(&sat_low, &fix.provider).expect("upload SAT low");
    let gpu_sat_high = GpuCnf::from_host(&sat_high, &fix.provider).expect("upload SAT high");
    let gpu_unsat = GpuCnf::from_host(&unsat, &fix.provider).expect("upload UNSAT candidate");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);

    let mut search_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut search_workspace = search_adapter
        .new_workspace(gpu_unsat.var_cap, gpu_unsat.clause_cap)
        .expect("new possible/not-know MaxSAT search workspace");
    let search_report = search_adapter
        .solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results(
            &fix.provider,
            &accepted_results,
            &mut search_workspace,
            &[
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 9,
                    cnf: &gpu_unsat,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                },
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 7,
                    cnf: &gpu_sat_low,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                },
            ],
        )
        .expect("accepted possible/not-know GPU evidence must gate MaxSAT search");

    assert_eq!(search_report.candidate_evidence_records, 2);
    assert_eq!(search_report.optimum_score, 7);
    assert_eq!(search_report.candidates_checked, 4);
    assert_eq!(search_report.satisfiable_candidates, 2);
    assert_eq!(search_report.unsat_candidates_pruned, 2);
    assert_eq!(search_report.gpu_cdcl_candidate_solves, 4);

    let search_trace = search_adapter.trace();
    assert_eq!(search_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        search_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        search_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        search_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(search_trace.gpu_cdcl_sat_solves, 2);
    assert_eq!(search_trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(search_trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(search_trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(search_trace.gpu_maxsat_optima, 2);
    assert_eq!(search_trace.cpu_assignment_enumerations, 0);
    assert_eq!(search_trace.cpu_maxsat_enumerations, 0);

    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![7.0, 9.0],
    );
    let both_soft_clauses = [0usize, 1usize];
    let sat_soft_clause = [0usize];
    let selections = [
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &both_soft_clauses,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &sat_soft_clause,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];

    let mut encoded_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut encoded_workspace = encoded_adapter
        .new_workspace(1, 2)
        .expect("new possible/not-know encoded MaxSAT workspace");
    let encoded_report = encoded_adapter
        .solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results(
            &fix.provider,
            &accepted_results,
            &mut encoded_workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("accepted possible/not-know GPU evidence must gate encoded MaxSAT");

    assert_eq!(encoded_report.candidate_evidence_records, 2);
    assert_eq!(encoded_report.optimum_score, 7);
    assert_eq!(encoded_report.candidates_checked, 4);
    assert_eq!(encoded_report.satisfiable_candidates, 2);
    assert_eq!(encoded_report.unsat_candidates_pruned, 2);
    assert_eq!(encoded_report.gpu_cdcl_candidate_encodes, 4);
    assert_eq!(encoded_report.gpu_cdcl_candidate_solves, 4);

    let encoded_trace = encoded_adapter.trace();
    assert_eq!(encoded_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        encoded_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        encoded_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        encoded_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(encoded_trace.gpu_maxsat_candidate_encodes, 4);
    assert_eq!(encoded_trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(encoded_trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(encoded_trace.gpu_maxsat_optima, 2);
    assert_eq!(encoded_trace.cpu_assignment_enumerations, 0);
    assert_eq!(encoded_trace.cpu_maxsat_enumerations, 0);

    let candidate_set = [
        GpuSolverProductionMaxSatCandidate {
            score: 3,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionMaxSatCandidate {
            score: 5,
            cnf: &gpu_sat_high,
            branch_var_limit: &branch_limit,
        },
    ];
    let search_candidates = [
        GpuSolverProductionMaxSatSearchCandidate {
            score: 9,
            cnf: &gpu_unsat,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionMaxSatSearchCandidate {
            score: 7,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];
    let jobs = [
        GpuSolverProductionMaxSatScheduleJob::CandidateSet {
            candidates: &candidate_set,
        },
        GpuSolverProductionMaxSatScheduleJob::Search {
            candidates: &search_candidates,
        },
        GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
            weighted: &weighted,
            branch_var_limit: &branch_limit,
            selections: &selections,
        },
        GpuSolverProductionMaxSatScheduleJob::Unknown {
            reason: "quaternary possible/not-know scheduler branch budget exhausted",
        },
        GpuSolverProductionMaxSatScheduleJob::Timeout { budget_micros: 25 },
    ];
    let mut scheduler_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut scheduler_workspace = scheduler_adapter
        .new_workspace(1, 2)
        .expect("new possible/not-know generalized MaxSAT scheduler workspace");
    let schedule_report = scheduler_adapter
        .solve_maxsat_schedule_with_gpu_execution_results(
            &fix.provider,
            &accepted_results,
            &mut scheduler_workspace,
            &jobs,
        )
        .expect("accepted possible/not-know GPU evidence must gate MaxSAT scheduler");

    assert_eq!(schedule_report.candidate_evidence_records, 2);
    assert_eq!(schedule_report.jobs, 10);
    assert_eq!(schedule_report.candidate_set_jobs, 2);
    assert_eq!(schedule_report.search_jobs, 2);
    assert_eq!(schedule_report.encoded_search_jobs, 2);
    assert_eq!(schedule_report.unknown_jobs, 2);
    assert_eq!(schedule_report.timeout_jobs, 2);
    assert_eq!(schedule_report.optimum_score, 7);
    assert_eq!(schedule_report.candidates_checked, 12);
    assert_eq!(schedule_report.satisfiable_candidates, 8);
    assert_eq!(schedule_report.unsat_candidates_pruned, 4);
    assert_eq!(schedule_report.gpu_cdcl_candidate_encodes, 4);
    assert_eq!(schedule_report.gpu_cdcl_candidate_solves, 12);

    let scheduler_trace = scheduler_adapter.trace();
    assert_eq!(scheduler_trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        scheduler_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(
        scheduler_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        8
    );
    assert_eq!(
        scheduler_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_jobs, 10);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_candidate_set_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_search_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_encoded_search_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_unknown_status_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_timeout_status_jobs, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_encodes, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_solves, 12);
    assert_eq!(scheduler_trace.gpu_maxsat_unsat_candidate_prunes, 4);
    assert_eq!(scheduler_trace.gpu_maxsat_optima, 6);
    assert_eq!(scheduler_trace.cpu_assignment_enumerations, 0);
    assert_eq!(scheduler_trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_quaternary_possible_and_not_know_results_gate_source_conditioned_probabilistic_gradients(
) {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let run_case = |label: &str,
                    program_source: &str,
                    expected_rows: Vec<(u32, u32, u32, u32)>,
                    expected_possible: usize,
                    expected_not_know: usize,
                    expected_negated_filters: usize|
     -> EpistemicGpuExecutionResult {
        let program = parse_program(program_source).expect(label);
        let executable =
            compile_epistemic_gpu_execution_with_stats_snapshot(&program, None).expect(label);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings.len(), 1);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings[0].arity, 4);

        let mut executor =
            Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
        for (name, rel_id) in &executable.relation_ids {
            executor.register_relation(*rel_id, name);
        }
        executor.put_relation(
            "tuple4",
            upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
        );
        executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

        let result = executor
            .execute_epistemic_gpu_execution(
                &executable,
                EpistemicGpuWorkspaceCapacities {
                    max_candidates: 2,
                    max_worlds: 1,
                    max_models_per_reduction: 3,
                },
            )
            .expect(label);
        assert_eq!(
            result.prepared.preflight.possible_operator_count,
            expected_possible
        );
        assert_eq!(
            result.prepared.preflight.not_know_operator_count,
            expected_not_know
        );
        assert_eq!(
            result.final_tuple_materialization.negated_row_filter_count,
            expected_negated_filters
        );
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            4
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(
            download_quaternary_u32(&fix.provider, &result.final_output),
            expected_rows,
            "{label} must filter by accepted arity-4 tuple evidence"
        );
        result
    };

    let possible_result = run_case(
        "quaternary possible source conditioned-gradient reuse",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), possible fact4(A, B, C, D).
        "#,
        vec![(2, 3, 4, 5)],
        1,
        0,
        0,
    );
    let not_know_result = run_case(
        "quaternary not-know source conditioned-gradient reuse",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not know fact4(A, B, C, D).
        "#,
        vec![(1, 2, 3, 4), (9, 9, 9, 9)],
        0,
        1,
        1,
    );

    let possible_assumptions = [EpistemicAssumption::possible_tuple(
        "fact4",
        vec![
            EpistemicEvidenceTerm::integer(2),
            EpistemicEvidenceTerm::integer(3),
            EpistemicEvidenceTerm::integer(4),
            EpistemicEvidenceTerm::integer(5),
        ],
        true,
    )];
    let not_know_assumptions = [EpistemicAssumption::known_tuple(
        "fact4",
        vec![
            EpistemicEvidenceTerm::integer(1),
            EpistemicEvidenceTerm::integer(2),
            EpistemicEvidenceTerm::integer(3),
            EpistemicEvidenceTerm::integer(4),
        ],
        false,
    )];
    let probabilistic_source = r#"
        0.8::fact4(2, 3, 4, 5).
        0.6::fact4(1, 2, 3, 4).
        query(fact4(2, 3, 4, 5)).
        query(fact4(1, 2, 3, 4)).
        "#;
    let assert_probs = |actual: &[f64], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "quaternary possible/not-know source conditioned-gradient mismatch: actual={actual} expected={expected}",
            );
        }
    };

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let gradients = adapter
        .compile_and_evaluate_conditioned_source_with_grads_for_gpu_execution_results(
            probabilistic_source,
            &fix.provider,
            &[
                EpistemicProbGpuExecutionEvidence {
                    result: &possible_result,
                    assumptions: &possible_assumptions,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &not_know_result,
                    assumptions: &not_know_assumptions,
                },
            ],
        )
        .expect("accepted quaternary possible/not-know evidence must condition source gradients");
    assert_eq!(gradients.len(), 2);
    let possible_grad_probs: Vec<f64> = gradients[0]
        .query_grads
        .iter()
        .map(|query| query.prob)
        .collect();
    assert_probs(&possible_grad_probs, &[1.0, 0.6]);
    let not_know_grad_probs: Vec<f64> = gradients[1]
        .query_grads
        .iter()
        .map(|query| query.prob)
        .collect();
    assert_probs(&not_know_grad_probs, &[0.8, 0.0]);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 2);
    assert_eq!(
        trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        0
    );
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_program_conditioned_max_evidence_arity, 0);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_known_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_not_known_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 2);
    assert_eq!(trace.gpu_exact_program_compiles, 0);
    assert_eq!(trace.gpu_exact_query_evaluations, 0);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(trace.gpu_source_conditioned_gradient_evaluations, 2);
    assert_eq!(trace.gpu_program_conditioned_gradient_evaluations, 0);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_quaternary_possible_and_not_know_results_gate_source_pir_cnf_and_exact_evaluation_paths(
) {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let run_case = |label: &str,
                    program_source: &str,
                    expected_rows: Vec<(u32, u32, u32, u32)>,
                    expected_possible: usize,
                    expected_not_know: usize,
                    expected_negated_filters: usize|
     -> EpistemicGpuExecutionResult {
        let program = parse_program(program_source).expect(label);
        let executable =
            compile_epistemic_gpu_execution_with_stats_snapshot(&program, None).expect(label);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings.len(), 1);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings[0].arity, 4);
        assert_eq!(
            executable.gpu_plan.tuple_membership_bindings[0].bound_output_columns,
            vec![Some(0), Some(1), Some(2), Some(3)]
        );
        assert_eq!(
            executable.gpu_plan.tuple_membership_bindings[0].negated,
            expected_negated_filters == 1
        );

        let mut executor =
            Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
        for (name, rel_id) in &executable.relation_ids {
            executor.register_relation(*rel_id, name);
        }
        executor.put_relation(
            "tuple4",
            upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
        );
        executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

        let result = executor
            .execute_epistemic_gpu_execution(
                &executable,
                EpistemicGpuWorkspaceCapacities {
                    max_candidates: 2,
                    max_worlds: 1,
                    max_models_per_reduction: 3,
                },
            )
            .expect(label);

        assert_eq!(
            result.prepared.preflight.possible_operator_count,
            expected_possible
        );
        assert_eq!(
            result.prepared.preflight.not_know_operator_count,
            expected_not_know
        );
        assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
        assert_eq!(
            result.final_tuple_materialization.negated_row_filter_count,
            expected_negated_filters
        );
        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            4
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
        assert_eq!(
            download_quaternary_u32(&fix.provider, &result.final_output),
            expected_rows,
            "{label} must filter by the accepted arity-4 tuple key evidence"
        );
        result
    };

    let possible_result = run_case(
        "quaternary possible source PIR/CNF reuse",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), possible fact4(A, B, C, D).
        "#,
        vec![(2, 3, 4, 5)],
        1,
        0,
        0,
    );
    let not_know_result = run_case(
        "quaternary not-know source PIR/CNF reuse",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not know fact4(A, B, C, D).
        "#,
        vec![(1, 2, 3, 4), (9, 9, 9, 9)],
        0,
        1,
        1,
    );

    let possible_assumptions = [EpistemicAssumption::possible_tuple(
        "fact4",
        vec![
            EpistemicEvidenceTerm::integer(2),
            EpistemicEvidenceTerm::integer(3),
            EpistemicEvidenceTerm::integer(4),
            EpistemicEvidenceTerm::integer(5),
        ],
        true,
    )];
    let not_know_assumptions = [EpistemicAssumption::known_tuple(
        "fact4",
        vec![
            EpistemicEvidenceTerm::integer(1),
            EpistemicEvidenceTerm::integer(2),
            EpistemicEvidenceTerm::integer(3),
            EpistemicEvidenceTerm::integer(4),
        ],
        false,
    )];
    let probabilistic_source = r#"
        0.8::fact4(2, 3, 4, 5).
        0.6::fact4(1, 2, 3, 4).
        query(fact4(2, 3, 4, 5)).
        query(fact4(1, 2, 3, 4)).
        "#;

    let assert_probs = |actual: &[f64], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "quaternary possible/not-know source probability mismatch: actual={actual} expected={expected}",
            );
        }
    };

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;

    let mut pir_adapter = EpistemicProbProductionAdapter::new(config);
    let pir_cnfs = pir_adapter
        .encode_source_pir_cnf_for_gpu_execution_results(
            probabilistic_source,
            &fix.provider,
            &[
                EpistemicProbGpuExecutionEvidence {
                    result: &possible_result,
                    assumptions: &possible_assumptions,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &not_know_result,
                    assumptions: &not_know_assumptions,
                },
            ],
        )
        .expect("accepted quaternary possible/not-know evidence must gate source PIR/CNF");
    assert_eq!(pir_cnfs.len(), 2);
    for pir_cnf in &pir_cnfs {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let pir_trace = pir_adapter.trace();
    assert_eq!(pir_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(pir_trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(pir_trace.gpu_pir_graph_uploads, 2);
    assert_eq!(pir_trace.gpu_source_pir_graph_uploads, 2);
    assert_eq!(pir_trace.gpu_program_pir_graph_uploads, 0);
    assert_eq!(pir_trace.gpu_cnf_encodes, 2);
    assert_eq!(pir_trace.gpu_source_cnf_encodes, 2);
    assert_eq!(pir_trace.gpu_program_cnf_encodes, 0);
    assert_eq!(pir_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(pir_trace.fixture_circuit_evaluations, 0);

    let mut query_adapter = EpistemicProbProductionAdapter::new(config);
    let exact = query_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &possible_result,
            possible_assumptions.to_vec(),
        )
        .expect("accepted quaternary possible evidence must gate source exact compile");
    let evaluated = query_adapter
        .evaluate_for_gpu_execution_results(
            &exact,
            &fix.provider,
            &[
                EpistemicProbGpuExecutionEvidence {
                    result: &possible_result,
                    assumptions: &possible_assumptions,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &not_know_result,
                    assumptions: &not_know_assumptions,
                },
            ],
        )
        .expect("accepted quaternary possible/not-know evidence must gate source exact queries");
    assert_eq!(evaluated.len(), 2);
    for result in &evaluated {
        let query_probs: Vec<f64> = result.query_probs.iter().map(|query| query.prob).collect();
        assert_probs(&query_probs, &[0.8, 0.6]);
    }
    let query_trace = query_adapter.trace();
    assert_eq!(query_trace.accepted_world_view_evidence_consumed, 3);
    assert_eq!(query_trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(query_trace.gpu_exact_source_compiles, 1);
    assert_eq!(query_trace.gpu_exact_program_compiles, 0);
    assert_eq!(query_trace.gpu_exact_query_evaluations, 2);
    assert_eq!(query_trace.gpu_exact_gradient_evaluations, 0);
    assert_eq!(query_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(query_trace.fixture_circuit_evaluations, 0);

    let mut gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let exact = gradient_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &possible_result,
            possible_assumptions.to_vec(),
        )
        .expect("accepted quaternary possible evidence must gate source exact compile");
    let gradients = gradient_adapter
        .evaluate_gpu_with_grads_for_gpu_execution_results(
            &exact,
            &fix.provider,
            &[
                EpistemicProbGpuExecutionEvidence {
                    result: &possible_result,
                    assumptions: &possible_assumptions,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &not_know_result,
                    assumptions: &not_know_assumptions,
                },
            ],
        )
        .expect("accepted quaternary possible/not-know evidence must gate source gradients");
    assert_eq!(gradients.len(), 2);
    for result in &gradients {
        let grad_probs: Vec<f64> = result.query_grads.iter().map(|query| query.prob).collect();
        assert_probs(&grad_probs, &[0.8, 0.6]);
    }
    let gradient_trace = gradient_adapter.trace();
    assert_eq!(gradient_trace.accepted_world_view_evidence_consumed, 3);
    assert_eq!(gradient_trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(gradient_trace.gpu_exact_source_compiles, 1);
    assert_eq!(gradient_trace.gpu_exact_program_compiles, 0);
    assert_eq!(gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(gradient_trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(gradient_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(gradient_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_quaternary_possible_and_not_know_results_gate_parsed_program_probabilistic_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let run_case = |label: &str,
                    program_source: &str,
                    expected_rows: Vec<(u32, u32, u32, u32)>,
                    expected_possible: usize,
                    expected_not_know: usize,
                    expected_negated_filters: usize|
     -> EpistemicGpuExecutionResult {
        let program = parse_program(program_source).expect(label);
        let executable =
            compile_epistemic_gpu_execution_with_stats_snapshot(&program, None).expect(label);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings.len(), 1);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings[0].arity, 4);

        let mut executor =
            Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
        for (name, rel_id) in &executable.relation_ids {
            executor.register_relation(*rel_id, name);
        }
        executor.put_relation(
            "tuple4",
            upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
        );
        executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

        let result = executor
            .execute_epistemic_gpu_execution(
                &executable,
                EpistemicGpuWorkspaceCapacities {
                    max_candidates: 2,
                    max_worlds: 1,
                    max_models_per_reduction: 3,
                },
            )
            .expect(label);

        assert_eq!(
            result.prepared.preflight.possible_operator_count,
            expected_possible
        );
        assert_eq!(
            result.prepared.preflight.not_know_operator_count,
            expected_not_know
        );
        assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
        assert_eq!(
            result.final_tuple_materialization.negated_row_filter_count,
            expected_negated_filters
        );
        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            4
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(
            download_quaternary_u32(&fix.provider, &result.final_output),
            expected_rows,
            "{label} must filter by arity-4 tuple-key evidence"
        );
        result
    };

    let possible_result = run_case(
        "quaternary possible parsed-program probability reuse",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), possible fact4(A, B, C, D).
        "#,
        vec![(2, 3, 4, 5)],
        1,
        0,
        0,
    );
    let not_know_result = run_case(
        "quaternary not-know parsed-program probability reuse",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not know fact4(A, B, C, D).
        "#,
        vec![(1, 2, 3, 4), (9, 9, 9, 9)],
        0,
        1,
        1,
    );

    let possible_assumptions = [EpistemicAssumption::possible_tuple(
        "fact4",
        vec![
            EpistemicEvidenceTerm::integer(2),
            EpistemicEvidenceTerm::integer(3),
            EpistemicEvidenceTerm::integer(4),
            EpistemicEvidenceTerm::integer(5),
        ],
        true,
    )];
    let not_know_assumptions = [EpistemicAssumption::known_tuple(
        "fact4",
        vec![
            EpistemicEvidenceTerm::integer(1),
            EpistemicEvidenceTerm::integer(2),
            EpistemicEvidenceTerm::integer(3),
            EpistemicEvidenceTerm::integer(4),
        ],
        false,
    )];
    let prob_program = parse_program(
        r#"
        0.8::fact4(2, 3, 4, 5).
        0.6::fact4(1, 2, 3, 4).
        query(fact4(2, 3, 4, 5)).
        query(fact4(1, 2, 3, 4)).
        "#,
    )
    .expect("parse quaternary possible/not-know parsed probabilistic program");

    let assert_probs = |actual: &[f64], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "quaternary possible/not-know parsed-program probability mismatch: actual={actual} expected={expected}",
            );
        }
    };

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_program_for_gpu_execution_results(
            &prob_program,
            &fix.provider,
            &[
                EpistemicProbGpuExecutionEvidence {
                    result: &possible_result,
                    assumptions: &possible_assumptions,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &not_know_result,
                    assumptions: &not_know_assumptions,
                },
            ],
        )
        .expect(
            "accepted quaternary possible/not-know evidence must condition parsed-program queries",
        );
    let gradients = adapter
        .compile_and_evaluate_conditioned_program_with_grads_for_gpu_execution_results(
            &prob_program,
            &fix.provider,
            &[
                EpistemicProbGpuExecutionEvidence {
                    result: &possible_result,
                    assumptions: &possible_assumptions,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &not_know_result,
                    assumptions: &not_know_assumptions,
                },
            ],
        )
        .expect("accepted quaternary possible/not-know evidence must condition parsed-program gradients");

    assert_eq!(evaluated.len(), 2);
    assert_eq!(gradients.len(), 2);
    let possible_probs: Vec<f64> = evaluated[0]
        .query_probs
        .iter()
        .map(|query| query.prob)
        .collect();
    assert_probs(&possible_probs, &[1.0, 0.6]);
    let not_know_probs: Vec<f64> = evaluated[1]
        .query_probs
        .iter()
        .map(|query| query.prob)
        .collect();
    assert_probs(&not_know_probs, &[0.8, 0.0]);
    let possible_grad_probs: Vec<f64> = gradients[0]
        .query_grads
        .iter()
        .map(|query| query.prob)
        .collect();
    assert_probs(&possible_grad_probs, &[1.0, 0.6]);
    let not_know_grad_probs: Vec<f64> = gradients[1]
        .query_grads
        .iter()
        .map(|query| query.prob)
        .collect();
    assert_probs(&not_know_grad_probs, &[0.8, 0.0]);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 4);
    assert_eq!(
        trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        4
    );
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 0);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_program_conditioned_max_evidence_arity, 4);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 0);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_not_known_evidence_facts, 2);
    assert_eq!(trace.gpu_program_conditioned_possible_evidence_facts, 2);
    assert_eq!(trace.gpu_program_conditioned_not_known_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 0);
    assert_eq!(trace.gpu_exact_program_compiles, 4);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_program_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(trace.gpu_program_conditioned_gradient_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 4);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 4);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_quaternary_possible_and_not_know_results_gate_parsed_program_pir_cnf_and_exact_evaluation_paths(
) {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let run_case = |label: &str,
                    program_source: &str,
                    expected_rows: Vec<(u32, u32, u32, u32)>,
                    expected_possible: usize,
                    expected_not_know: usize,
                    expected_negated_filters: usize|
     -> EpistemicGpuExecutionResult {
        let program = parse_program(program_source).expect(label);
        let executable =
            compile_epistemic_gpu_execution_with_stats_snapshot(&program, None).expect(label);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings.len(), 1);
        assert_eq!(executable.gpu_plan.tuple_membership_bindings[0].arity, 4);

        let mut executor =
            Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
        for (name, rel_id) in &executable.relation_ids {
            executor.register_relation(*rel_id, name);
        }
        executor.put_relation(
            "tuple4",
            upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
        );
        executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

        let result = executor
            .execute_epistemic_gpu_execution(
                &executable,
                EpistemicGpuWorkspaceCapacities {
                    max_candidates: 2,
                    max_worlds: 1,
                    max_models_per_reduction: 3,
                },
            )
            .expect(label);

        assert_eq!(
            result.prepared.preflight.possible_operator_count,
            expected_possible
        );
        assert_eq!(
            result.prepared.preflight.not_know_operator_count,
            expected_not_know
        );
        assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
        assert_eq!(
            result.final_tuple_materialization.negated_row_filter_count,
            expected_negated_filters
        );
        assert_eq!(
            result.model_membership.membership_source,
            EpistemicGpuModelMembershipSource::StableModelTupleBuffer
        );
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            4
        );
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert_eq!(
            download_quaternary_u32(&fix.provider, &result.final_output),
            expected_rows,
            "{label} must filter by arity-4 tuple-key evidence"
        );
        result
    };

    let possible_result = run_case(
        "quaternary possible parsed-program PIR/CNF reuse",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), possible fact4(A, B, C, D).
        "#,
        vec![(2, 3, 4, 5)],
        1,
        0,
        0,
    );
    let not_know_result = run_case(
        "quaternary not-know parsed-program PIR/CNF reuse",
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not know fact4(A, B, C, D).
        "#,
        vec![(1, 2, 3, 4), (9, 9, 9, 9)],
        0,
        1,
        1,
    );

    let possible_assumptions = [EpistemicAssumption::possible_tuple(
        "fact4",
        vec![
            EpistemicEvidenceTerm::integer(2),
            EpistemicEvidenceTerm::integer(3),
            EpistemicEvidenceTerm::integer(4),
            EpistemicEvidenceTerm::integer(5),
        ],
        true,
    )];
    let not_know_assumptions = [EpistemicAssumption::known_tuple(
        "fact4",
        vec![
            EpistemicEvidenceTerm::integer(1),
            EpistemicEvidenceTerm::integer(2),
            EpistemicEvidenceTerm::integer(3),
            EpistemicEvidenceTerm::integer(4),
        ],
        false,
    )];
    let prob_program = parse_program(
        r#"
        0.8::fact4(2, 3, 4, 5).
        0.6::fact4(1, 2, 3, 4).
        query(fact4(2, 3, 4, 5)).
        query(fact4(1, 2, 3, 4)).
        "#,
    )
    .expect("parse quaternary possible/not-know parsed probabilistic program");

    let assert_probs = |actual: &[f64], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "quaternary possible/not-know parsed-program probability mismatch: actual={actual} expected={expected}",
            );
        }
    };

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;

    let mut pir_adapter = EpistemicProbProductionAdapter::new(config);
    let pir_cnfs = pir_adapter
        .encode_program_pir_cnf_for_gpu_execution_results(
            &prob_program,
            &fix.provider,
            &[
                EpistemicProbGpuExecutionEvidence {
                    result: &possible_result,
                    assumptions: &possible_assumptions,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &not_know_result,
                    assumptions: &not_know_assumptions,
                },
            ],
        )
        .expect("accepted quaternary possible/not-know evidence must gate parsed-program PIR/CNF");
    assert_eq!(pir_cnfs.len(), 2);
    for pir_cnf in &pir_cnfs {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let pir_trace = pir_adapter.trace();
    assert_eq!(pir_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(pir_trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(pir_trace.gpu_pir_graph_uploads, 2);
    assert_eq!(pir_trace.gpu_program_pir_graph_uploads, 2);
    assert_eq!(pir_trace.gpu_source_pir_graph_uploads, 0);
    assert_eq!(pir_trace.gpu_cnf_encodes, 2);
    assert_eq!(pir_trace.gpu_program_cnf_encodes, 2);
    assert_eq!(pir_trace.gpu_source_cnf_encodes, 0);
    assert_eq!(pir_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(pir_trace.fixture_circuit_evaluations, 0);

    let mut query_adapter = EpistemicProbProductionAdapter::new(config);
    let exact = query_adapter
        .compile_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &possible_result,
            possible_assumptions.to_vec(),
        )
        .expect("accepted quaternary possible evidence must gate parsed-program exact compile");
    let evaluated = query_adapter
        .evaluate_for_gpu_execution_results(
            &exact,
            &fix.provider,
            &[
                EpistemicProbGpuExecutionEvidence {
                    result: &possible_result,
                    assumptions: &possible_assumptions,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &not_know_result,
                    assumptions: &not_know_assumptions,
                },
            ],
        )
        .expect(
            "accepted quaternary possible/not-know evidence must gate parsed-program exact queries",
        );
    assert_eq!(evaluated.len(), 2);
    for result in &evaluated {
        let query_probs: Vec<f64> = result.query_probs.iter().map(|query| query.prob).collect();
        assert_probs(&query_probs, &[0.8, 0.6]);
    }
    let query_trace = query_adapter.trace();
    assert_eq!(query_trace.accepted_world_view_evidence_consumed, 3);
    assert_eq!(query_trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(query_trace.gpu_exact_source_compiles, 0);
    assert_eq!(query_trace.gpu_exact_program_compiles, 1);
    assert_eq!(query_trace.gpu_exact_query_evaluations, 2);
    assert_eq!(query_trace.gpu_exact_gradient_evaluations, 0);
    assert_eq!(query_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(query_trace.fixture_circuit_evaluations, 0);

    let mut gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let exact = gradient_adapter
        .compile_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &possible_result,
            possible_assumptions.to_vec(),
        )
        .expect("accepted quaternary possible evidence must gate parsed-program exact compile");
    let gradients = gradient_adapter
        .evaluate_gpu_with_grads_for_gpu_execution_results(
            &exact,
            &fix.provider,
            &[
                EpistemicProbGpuExecutionEvidence {
                    result: &possible_result,
                    assumptions: &possible_assumptions,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &not_know_result,
                    assumptions: &not_know_assumptions,
                },
            ],
        )
        .expect(
            "accepted quaternary possible/not-know evidence must gate parsed-program gradients",
        );
    assert_eq!(gradients.len(), 2);
    for result in &gradients {
        let grad_probs: Vec<f64> = result.query_grads.iter().map(|query| query.prob).collect();
        assert_probs(&grad_probs, &[0.8, 0.6]);
    }
    let gradient_trace = gradient_adapter.trace();
    assert_eq!(gradient_trace.accepted_world_view_evidence_consumed, 3);
    assert_eq!(gradient_trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(gradient_trace.gpu_exact_source_compiles, 0);
    assert_eq!(gradient_trace.gpu_exact_program_compiles, 1);
    assert_eq!(gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(gradient_trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(gradient_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(gradient_trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_result_conditions_negative_nonzero_arity_probabilistic_evidence() {
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
    .expect("parse negative nonzero-arity epistemic fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile negative nonzero-arity epistemic executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[2]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute negative nonzero-arity accepted epistemic fixture");
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![1]
    );

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
            0.7::edge(1).
            query(edge(1)).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known_tuple(
                "edge",
                vec![EpistemicEvidenceTerm::integer(1)],
                false,
            )],
        )
        .expect(
            "accepted GPU runtime evidence must condition negative probabilistic exact evidence",
        );

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!(
        evaluated.query_probs[0].prob.abs() < 1.0e-6,
        "accepted not know edge(1) evidence must condition query probability to false"
    );
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_possible_operator_conditions_probabilistic_evidence() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), possible edge(X).
        "#,
        &[1, 2],
        &[1],
    );
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![1]
    );
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
            0.4::edge(1).
            query(edge(1)).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::possible_tuple(
                "edge",
                vec![EpistemicEvidenceTerm::integer(1)],
                true,
            )],
        )
        .expect("accepted possible GPU evidence must condition probabilistic exact evidence");

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!(
        (evaluated.query_probs[0].prob - 1.0).abs() < 1.0e-6,
        "accepted possible edge(1) evidence must condition query probability to true"
    );
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 0);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_not_possible_operator_conditions_negative_probabilistic_evidence() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), not possible edge(X).
        "#,
        &[1, 2],
        &[2],
    );
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![1]
    );
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
            0.7::edge(1).
            query(edge(1)).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::possible_tuple(
                "edge",
                vec![EpistemicEvidenceTerm::integer(1)],
                false,
            )],
        )
        .expect("accepted not-possible GPU evidence must condition negative exact evidence");

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!(
        evaluated.query_probs[0].prob.abs() < 1.0e-6,
        "accepted not possible edge(1) evidence must condition query probability to false"
    );
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_binary_not_know_operator_conditions_negative_probabilistic_evidence() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_binary_edge_epistemic_fixture(
        &fix,
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred accepted(u32, u32).
        accepted(X, Y) :- pair(X, Y), not know edge(X, Y).
        "#,
        &[(1, 2), (2, 3), (3, 4)],
        &[(2, 3)],
    );
    assert_eq!(
        download_binary_u32(&fix.provider, &result.final_output),
        vec![(1, 2), (3, 4)]
    );
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
            0.7::edge(1, 2).
            query(edge(1, 2)).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known_tuple(
                "edge",
                vec![
                    EpistemicEvidenceTerm::integer(1),
                    EpistemicEvidenceTerm::integer(2),
                ],
                false,
            )],
        )
        .expect("accepted binary not-know GPU evidence must condition negative exact evidence");

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!(
        evaluated.query_probs[0].prob.abs() < 1.0e-6,
        "accepted binary not know edge(1, 2) evidence must condition query probability to false"
    );
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_binary_possible_operator_conditions_probabilistic_evidence() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_binary_edge_epistemic_fixture(
        &fix,
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred accepted(u32, u32).
        accepted(X, Y) :- pair(X, Y), possible edge(X, Y).
        "#,
        &[(1, 2), (2, 3), (3, 4)],
        &[(1, 2), (3, 4)],
    );
    assert_eq!(
        download_binary_u32(&fix.provider, &result.final_output),
        vec![(1, 2), (3, 4)]
    );
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
            0.4::edge(1, 2).
            query(edge(1, 2)).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::possible_tuple(
                "edge",
                vec![
                    EpistemicEvidenceTerm::integer(1),
                    EpistemicEvidenceTerm::integer(2),
                ],
                true,
            )],
        )
        .expect("accepted binary possible GPU evidence must condition exact evidence");

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!(
        (evaluated.query_probs[0].prob - 1.0).abs() < 1.0e-6,
        "accepted binary possible edge(1, 2) evidence must condition query probability to true"
    );
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 0);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_binary_not_possible_operator_conditions_negative_probabilistic_evidence() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_binary_edge_epistemic_fixture(
        &fix,
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred accepted(u32, u32).
        accepted(X, Y) :- pair(X, Y), not possible edge(X, Y).
        "#,
        &[(1, 2), (2, 3), (3, 4)],
        &[(2, 3)],
    );
    assert_eq!(
        download_binary_u32(&fix.provider, &result.final_output),
        vec![(1, 2), (3, 4)]
    );
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
            0.7::edge(1, 2).
            query(edge(1, 2)).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::possible_tuple(
                "edge",
                vec![
                    EpistemicEvidenceTerm::integer(1),
                    EpistemicEvidenceTerm::integer(2),
                ],
                false,
            )],
        )
        .expect("accepted binary not-possible GPU evidence must condition negative exact evidence");

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!(
        evaluated.query_probs[0].prob.abs() < 1.0e-6,
        "accepted binary not possible edge(1, 2) evidence must condition query probability to false"
    );
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_operator_conditions_record_probabilistic_operator_trace_counters() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let know_result = execute_binary_edge_epistemic_fixture(
        &fix,
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred accepted(u32, u32).
        accepted(X, Y) :- pair(X, Y), know edge(X, Y).
        "#,
        &[(1, 2), (2, 3)],
        &[(1, 2)],
    );
    let possible_result = execute_binary_edge_epistemic_fixture(
        &fix,
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred accepted(u32, u32).
        accepted(X, Y) :- pair(X, Y), possible edge(X, Y).
        "#,
        &[(1, 2), (2, 3)],
        &[(1, 2)],
    );
    let not_know_result = execute_binary_edge_epistemic_fixture(
        &fix,
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred accepted(u32, u32).
        accepted(X, Y) :- pair(X, Y), not know edge(X, Y).
        "#,
        &[(1, 2), (2, 3)],
        &[(2, 3)],
    );
    let not_possible_result = execute_binary_edge_epistemic_fixture(
        &fix,
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred accepted(u32, u32).
        accepted(X, Y) :- pair(X, Y), not possible edge(X, Y).
        "#,
        &[(1, 2), (2, 3)],
        &[(2, 3)],
    );

    assert_eq!(know_result.prepared.preflight.know_operator_count, 1);
    assert_eq!(
        possible_result.prepared.preflight.possible_operator_count,
        1
    );
    assert_eq!(
        not_know_result.prepared.preflight.not_know_operator_count,
        1
    );
    assert_eq!(
        not_possible_result
            .prepared
            .preflight
            .not_possible_operator_count,
        1
    );

    let know_assumptions = [EpistemicAssumption::known_tuple(
        "edge",
        vec![
            EpistemicEvidenceTerm::integer(1),
            EpistemicEvidenceTerm::integer(2),
        ],
        true,
    )];
    let possible_assumptions = [EpistemicAssumption::possible_tuple(
        "edge",
        vec![
            EpistemicEvidenceTerm::integer(1),
            EpistemicEvidenceTerm::integer(2),
        ],
        true,
    )];
    let not_know_assumptions = [EpistemicAssumption::known_tuple(
        "edge",
        vec![
            EpistemicEvidenceTerm::integer(1),
            EpistemicEvidenceTerm::integer(2),
        ],
        false,
    )];
    let not_possible_assumptions = [EpistemicAssumption::possible_tuple(
        "edge",
        vec![
            EpistemicEvidenceTerm::integer(1),
            EpistemicEvidenceTerm::integer(2),
        ],
        false,
    )];

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_for_gpu_execution_results(
            r#"
            0.4::edge(1, 2).
            query(edge(1, 2)).
            "#,
            &fix.provider,
            &[
                EpistemicProbGpuExecutionEvidence {
                    result: &know_result,
                    assumptions: &know_assumptions,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &possible_result,
                    assumptions: &possible_assumptions,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &not_know_result,
                    assumptions: &not_know_assumptions,
                },
                EpistemicProbGpuExecutionEvidence {
                    result: &not_possible_result,
                    assumptions: &not_possible_assumptions,
                },
            ],
        )
        .expect("accepted operator GPU evidence must record exact-path operator counters");

    assert_eq!(evaluated.len(), 4);
    assert!((evaluated[0].query_probs[0].prob - 1.0).abs() < 1.0e-6);
    assert!((evaluated[1].query_probs[0].prob - 1.0).abs() < 1.0e-6);
    assert!(evaluated[2].query_probs[0].prob.abs() < 1.0e-6);
    assert!(evaluated[3].query_probs[0].prob.abs() < 1.0e-6);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 4);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_known_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 4);
    assert_eq!(trace.gpu_exact_query_evaluations, 4);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_all_operator_mixed_membership_conditions_probabilistic_evidence() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_all_operator_mixed_membership_fixture(&fix);
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![2]
    );
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 4);
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(result.final_tuple_materialization.row_filter_count, 4);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        2
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
            0.3::edge(2).
            0.4::alt(2).
            0.5::hidden(2).
            0.6::blocked(2).
            query(edge(2)).
            query(alt(2)).
            query(hidden(2)).
            query(blocked(2)).
            "#,
            &fix.provider,
            &result,
            all_operator_mixed_membership_assumptions(),
        )
        .expect("same-rule all-operator GPU evidence must condition probabilistic exact evidence");

    assert_eq!(evaluated.query_probs.len(), 4);
    assert!((evaluated.query_probs[0].prob - 1.0).abs() < 1.0e-6);
    assert!((evaluated.query_probs[1].prob - 1.0).abs() < 1.0e-6);
    assert!(evaluated.query_probs[2].prob.abs() < 1.0e-6);
    assert!(evaluated.query_probs[3].prob.abs() < 1.0e-6);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 4);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_known_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 4);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_source_conditioned_negative_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_not_known_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_all_operator_mixed_membership_gates_probabilistic_program_gradient_and_pir_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_all_operator_mixed_membership_fixture(&fix);
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![2]
    );

    let prob_program = parse_program(
        r#"
        0.3::edge(2).
        0.4::alt(2).
        0.5::hidden(2).
        0.6::blocked(2).
        query(edge(2)).
        query(alt(2)).
        query(hidden(2)).
        query(blocked(2)).
        "#,
    )
    .expect("parse all-operator probabilistic program");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            all_operator_mixed_membership_assumptions(),
        )
        .expect("same-rule all-operator evidence must condition parsed-program exact evidence");
    let grads = adapter
        .compile_and_evaluate_conditioned_program_with_grads_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            all_operator_mixed_membership_assumptions(),
        )
        .expect("same-rule all-operator evidence must condition parsed-program gradients");
    let pir_cnf = adapter
        .encode_program_pir_cnf_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            all_operator_mixed_membership_assumptions(),
        )
        .expect("same-rule all-operator evidence must gate parsed-program PIR/CNF");

    assert_eq!(evaluated.query_probs.len(), 4);
    assert!((evaluated.query_probs[0].prob - 1.0).abs() < 1.0e-6);
    assert!((evaluated.query_probs[1].prob - 1.0).abs() < 1.0e-6);
    assert!(evaluated.query_probs[2].prob.abs() < 1.0e-6);
    assert!(evaluated.query_probs[3].prob.abs() < 1.0e-6);
    assert_eq!(grads.query_grads.len(), 4);
    assert!((grads.query_grads[0].prob - 1.0).abs() < 1.0e-6);
    assert!((grads.query_grads[1].prob - 1.0).abs() < 1.0e-6);
    assert!(grads.query_grads[2].prob.abs() < 1.0e-6);
    assert!(grads.query_grads[3].prob.abs() < 1.0e-6);
    assert!(pir_cnf.pir_nodes > 0);
    assert!(pir_cnf.root_count > 0);
    assert!(pir_cnf.cnf_var_cap > 0);
    assert!(pir_cnf.cnf_clause_cap > 0);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 3);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 12);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 8);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 8);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_not_known_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_not_possible_evidence_facts, 2);
    assert_eq!(trace.gpu_program_conditioned_evidence_facts, 8);
    assert_eq!(
        trace.gpu_program_conditioned_nonzero_arity_evidence_facts,
        8
    );
    assert_eq!(trace.gpu_program_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_program_conditioned_negative_evidence_facts, 4);
    assert_eq!(trace.gpu_program_conditioned_know_evidence_facts, 2);
    assert_eq!(trace.gpu_program_conditioned_possible_evidence_facts, 2);
    assert_eq!(trace.gpu_program_conditioned_not_known_evidence_facts, 2);
    assert_eq!(trace.gpu_program_conditioned_not_possible_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_program_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_program_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 1);
    assert_eq!(trace.gpu_program_conditioned_gradient_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_pir_graph_uploads, 1);
    assert_eq!(trace.gpu_program_pir_graph_uploads, 1);
    assert_eq!(trace.gpu_cnf_encodes, 1);
    assert_eq!(trace.gpu_program_cnf_encodes, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_all_operator_mixed_membership_gates_probabilistic_source_pir_and_exact_evaluation_paths(
) {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_all_operator_mixed_membership_fixture(&fix);
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![2]
    );
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 4);
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let probabilistic_source = r#"
        0.3::edge(2).
        0.4::alt(2).
        0.5::hidden(2).
        0.6::blocked(2).
        query(edge(2)).
        query(alt(2)).
        query(hidden(2)).
        query(blocked(2)).
        "#;
    let assert_conditioned = |probs: &[f64]| {
        let expected = [1.0, 1.0, 0.0, 0.0];
        assert_eq!(probs.len(), expected.len());
        for (actual, expected) in probs.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "conditioned all-operator probability mismatch: actual={actual} expected={expected}",
            );
        }
    };
    let assert_unconditioned = |probs: &[f64]| {
        let expected = [0.3, 0.4, 0.5, 0.6];
        assert_eq!(probs.len(), expected.len());
        for (actual, expected) in probs.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "unconditioned all-operator probability mismatch: actual={actual} expected={expected}",
            );
        }
    };

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let conditioned_grads = adapter
        .compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &result,
            all_operator_mixed_membership_assumptions(),
        )
        .expect("same-rule all-operator evidence must condition source gradients");
    let pir_cnf = adapter
        .encode_source_pir_cnf_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &result,
            all_operator_mixed_membership_assumptions(),
        )
        .expect("same-rule all-operator evidence must gate source PIR/CNF");
    let exact = adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &result,
            all_operator_mixed_membership_assumptions(),
        )
        .expect("same-rule all-operator evidence must gate source exact compile");
    let evaluated = adapter
        .evaluate_with_gpu_execution_result(
            &exact,
            &fix.provider,
            &result,
            all_operator_mixed_membership_assumptions(),
        )
        .expect("same-rule all-operator evidence must gate source exact query evaluation");
    let evaluated_grads = adapter
        .evaluate_gpu_with_grads_with_gpu_execution_result(
            &exact,
            &fix.provider,
            &result,
            all_operator_mixed_membership_assumptions(),
        )
        .expect("same-rule all-operator evidence must gate source exact gradient evaluation");

    let conditioned_probs: Vec<f64> = conditioned_grads
        .query_grads
        .iter()
        .map(|query| query.prob)
        .collect();
    assert_conditioned(&conditioned_probs);
    assert!(pir_cnf.pir_nodes > 0);
    assert!(pir_cnf.root_count > 0);
    assert!(pir_cnf.cnf_var_cap > 0);
    assert!(pir_cnf.cnf_clause_cap > 0);
    let query_probs: Vec<f64> = evaluated
        .query_probs
        .iter()
        .map(|query| query.prob)
        .collect();
    assert_unconditioned(&query_probs);
    let gradient_probs: Vec<f64> = evaluated_grads
        .query_grads
        .iter()
        .map(|query| query.prob)
        .collect();
    assert_unconditioned(&gradient_probs);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 5);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 20);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_nonzero_arity_evidence_facts, 4);
    assert_eq!(trace.gpu_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_known_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 4);
    assert_eq!(trace.gpu_source_conditioned_nonzero_arity_evidence_facts, 4);
    assert_eq!(trace.gpu_source_conditioned_max_evidence_arity, 1);
    assert_eq!(trace.gpu_source_conditioned_negative_evidence_facts, 2);
    assert_eq!(trace.gpu_source_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_not_known_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_not_possible_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(trace.gpu_source_conditioned_gradient_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_pir_graph_uploads, 1);
    assert_eq!(trace.gpu_source_pir_graph_uploads, 1);
    assert_eq!(trace.gpu_program_pir_graph_uploads, 0);
    assert_eq!(trace.gpu_cnf_encodes, 1);
    assert_eq!(trace.gpu_source_cnf_encodes, 1);
    assert_eq!(trace.gpu_program_cnf_encodes, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_all_operator_mixed_membership_gates_probabilistic_program_exact_evaluation_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_all_operator_mixed_membership_fixture(&fix);
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![2]
    );
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 4);
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let prob_program = parse_program(
        r#"
        0.3::edge(2).
        0.4::alt(2).
        0.5::hidden(2).
        0.6::blocked(2).
        query(edge(2)).
        query(alt(2)).
        query(hidden(2)).
        query(blocked(2)).
        "#,
    )
    .expect("parse all-operator probabilistic program");
    let assert_unconditioned = |probs: &[f64]| {
        let expected = [0.3, 0.4, 0.5, 0.6];
        assert_eq!(probs.len(), expected.len());
        for (actual, expected) in probs.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "unconditioned all-operator program probability mismatch: actual={actual} expected={expected}",
            );
        }
    };

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let exact = adapter
        .compile_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            all_operator_mixed_membership_assumptions(),
        )
        .expect("same-rule all-operator evidence must gate parsed-program exact compile");
    let evaluated = adapter
        .evaluate_with_gpu_execution_result(
            &exact,
            &fix.provider,
            &result,
            all_operator_mixed_membership_assumptions(),
        )
        .expect("same-rule all-operator evidence must gate parsed-program exact query evaluation");
    let evaluated_grads = adapter
        .evaluate_gpu_with_grads_with_gpu_execution_result(
            &exact,
            &fix.provider,
            &result,
            all_operator_mixed_membership_assumptions(),
        )
        .expect(
            "same-rule all-operator evidence must gate parsed-program exact gradient evaluation",
        );

    let query_probs: Vec<f64> = evaluated
        .query_probs
        .iter()
        .map(|query| query.prob)
        .collect();
    assert_unconditioned(&query_probs);
    let gradient_probs: Vec<f64> = evaluated_grads
        .query_grads
        .iter()
        .map(|query| query.prob)
        .collect();
    assert_unconditioned(&gradient_probs);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 3);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 12);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 0);
    assert_eq!(trace.gpu_exact_source_compiles, 0);
    assert_eq!(trace.gpu_exact_program_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn conditioned_probabilistic_evidence_records_source_and_program_trace_counters() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let source_result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
        &[1],
        &[1],
    );
    let program_result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), not know edge(X).
        "#,
        &[1],
        &[],
    );

    assert_eq!(
        download_unary_u32(&fix.provider, &source_result.final_output),
        vec![1]
    );
    assert_eq!(
        download_unary_u32(&fix.provider, &program_result.final_output),
        vec![1]
    );

    let parsed_program = parse_program(
        r#"
        0.6::edge(1).
        query(edge(1)).
        "#,
    )
    .expect("parse conditioned program trace fixture");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let source_eval = adapter
        .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
            r#"
            0.6::edge(1).
            query(edge(1)).
            "#,
            &fix.provider,
            &source_result,
            vec![EpistemicAssumption::known_tuple(
                "edge",
                vec![EpistemicEvidenceTerm::integer(1)],
                true,
            )],
        )
        .expect("accepted source evidence must condition exact source path");
    let program_eval = adapter
        .compile_and_evaluate_conditioned_program_with_gpu_execution_result(
            &parsed_program,
            &fix.provider,
            &program_result,
            vec![EpistemicAssumption::known_tuple(
                "edge",
                vec![EpistemicEvidenceTerm::integer(1)],
                false,
            )],
        )
        .expect("accepted program evidence must condition exact parsed-program path");

    assert!((source_eval.query_probs[0].prob - 1.0).abs() < 1.0e-6);
    assert!(program_eval.query_probs[0].prob.abs() < 1.0e-6);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_program_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_negative_evidence_facts, 0);
    assert_eq!(trace.gpu_program_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_know_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_possible_evidence_facts, 0);
    assert_eq!(trace.gpu_source_conditioned_not_known_evidence_facts, 0);
    assert_eq!(trace.gpu_source_conditioned_not_possible_evidence_facts, 0);
    assert_eq!(trace.gpu_program_conditioned_know_evidence_facts, 0);
    assert_eq!(trace.gpu_program_conditioned_possible_evidence_facts, 0);
    assert_eq!(trace.gpu_program_conditioned_not_known_evidence_facts, 1);
    assert_eq!(trace.gpu_program_conditioned_not_possible_evidence_facts, 0);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_program_compiles, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn conditioned_probabilistic_gradients_record_source_and_program_trace_counters() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let source_result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
        &[1],
        &[1],
    );
    let program_result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), not know edge(X).
        "#,
        &[1],
        &[],
    );

    let parsed_program = parse_program(
        r#"
        0.6::edge(1).
        query(edge(1)).
        "#,
    )
    .expect("parse conditioned gradient trace fixture");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let source_grads = adapter
        .compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result(
            r#"
            0.6::edge(1).
            query(edge(1)).
            "#,
            &fix.provider,
            &source_result,
            vec![EpistemicAssumption::known_tuple(
                "edge",
                vec![EpistemicEvidenceTerm::integer(1)],
                true,
            )],
        )
        .expect("accepted source evidence must condition exact source gradient path");
    let program_grads = adapter
        .compile_and_evaluate_conditioned_program_with_grads_with_gpu_execution_result(
            &parsed_program,
            &fix.provider,
            &program_result,
            vec![EpistemicAssumption::known_tuple(
                "edge",
                vec![EpistemicEvidenceTerm::integer(1)],
                false,
            )],
        )
        .expect("accepted program evidence must condition exact parsed-program gradient path");

    assert!((source_grads.query_grads[0].prob - 1.0).abs() < 1.0e-6);
    assert!(program_grads.query_grads[0].prob.abs() < 1.0e-6);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_program_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_source_conditioned_negative_evidence_facts, 0);
    assert_eq!(trace.gpu_program_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_program_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 0);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(trace.gpu_source_conditioned_gradient_evaluations, 1);
    assert_eq!(trace.gpu_program_conditioned_gradient_evaluations, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_result_conditions_parsed_program_probabilistic_evidence() {
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
        .expect("execute nonzero-arity accepted epistemic fixture");

    let prob_program = parse_program(
        r#"
        0.4::edge(1).
        query(edge(1)).
        "#,
    )
    .expect("parse probabilistic program");
    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known_tuple(
                "edge",
                vec![EpistemicEvidenceTerm::integer(1)],
                true,
            )],
        )
        .expect("accepted GPU runtime evidence must condition parsed probabilistic program");

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!((evaluated.query_probs[0].prob - 1.0).abs() < 1.0e-6);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_program_compiles, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 0);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_result_conditions_negative_parsed_program_probabilistic_evidence() {
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
    .expect("parse negative nonzero-arity epistemic fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile negative nonzero-arity epistemic executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[2]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute negative nonzero-arity accepted epistemic fixture");
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![1]
    );

    let prob_program = parse_program(
        r#"
        0.4::edge(1).
        query(edge(1)).
        "#,
    )
    .expect("parse probabilistic program");
    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known_tuple(
                "edge",
                vec![EpistemicEvidenceTerm::integer(1)],
                false,
            )],
        )
        .expect(
            "accepted GPU runtime evidence must condition parsed negative probabilistic evidence",
        );

    assert_eq!(evaluated.query_probs.len(), 1);
    assert!(
        evaluated.query_probs[0].prob.abs() < 1.0e-6,
        "accepted not know edge(1) evidence must condition parsed query probability to false"
    );
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_program_compiles, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 0);
    assert_eq!(trace.gpu_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_results_gate_batched_conditioned_probabilistic_queries() {
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
    let assumptions_a = [EpistemicAssumption::known_tuple(
        "edge",
        vec![EpistemicEvidenceTerm::integer(1)],
        true,
    )];
    let assumptions_b = [EpistemicAssumption::known_tuple(
        "edge",
        vec![EpistemicEvidenceTerm::integer(2)],
        true,
    )];

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_for_gpu_execution_results(
            r#"
            0.2::edge(1).
            0.8::edge(2).
            query(edge(1)).
            query(edge(2)).
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
        .expect("accepted GPU runtime evidence must gate batched conditioned queries");

    assert_eq!(evaluated.len(), 2);
    assert_eq!(evaluated[0].query_probs.len(), 2);
    assert!((evaluated[0].query_probs[0].prob - 1.0).abs() < 1.0e-6);
    assert!((evaluated[0].query_probs[1].prob - 0.8).abs() < 1.0e-6);
    assert_eq!(evaluated[1].query_probs.len(), 2);
    assert!((evaluated[1].query_probs[0].prob - 0.2).abs() < 1.0e-6);
    assert!((evaluated[1].query_probs[1].prob - 1.0).abs() < 1.0e-6);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 2);
    assert_eq!(trace.gpu_exact_program_compiles, 0);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_results_gate_batched_negative_conditioned_probabilistic_queries() {
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
    .expect("parse negative nonzero-arity epistemic fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile negative nonzero-arity epistemic executable");

    let make_result = |node_rows: &[u32]| {
        let mut executor =
            Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
        for (name, rel_id) in &executable.relation_ids {
            executor.register_relation(*rel_id, name);
        }
        executor.put_relation("node", upload_unary_u32(&fix.memory, node_rows));
        executor.put_relation("edge", upload_unary_u32(&fix.memory, &[]));

        executor
            .execute_epistemic_gpu_execution(
                &executable,
                EpistemicGpuWorkspaceCapacities {
                    max_candidates: 2,
                    max_worlds: 1,
                    max_models_per_reduction: 2,
                },
            )
            .expect("execute negative accepted epistemic fixture")
    };
    let result_a = make_result(&[1]);
    let result_b = make_result(&[2]);
    let assumptions_a = [EpistemicAssumption::known_tuple(
        "edge",
        vec![EpistemicEvidenceTerm::integer(1)],
        false,
    )];
    let assumptions_b = [EpistemicAssumption::known_tuple(
        "edge",
        vec![EpistemicEvidenceTerm::integer(2)],
        false,
    )];

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_source_for_gpu_execution_results(
            r#"
            0.2::edge(1).
            0.8::edge(2).
            query(edge(1)).
            query(edge(2)).
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
        .expect("accepted GPU runtime evidence must gate batched negative conditions");

    assert_eq!(evaluated.len(), 2);
    assert_eq!(evaluated[0].query_probs.len(), 2);
    assert!(evaluated[0].query_probs[0].prob.abs() < 1.0e-6);
    assert!((evaluated[0].query_probs[1].prob - 0.8).abs() < 1.0e-6);
    assert_eq!(evaluated[1].query_probs.len(), 2);
    assert!((evaluated[1].query_probs[0].prob - 0.2).abs() < 1.0e-6);
    assert!(evaluated[1].query_probs[1].prob.abs() < 1.0e-6);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_conditioned_negative_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 2);
    assert_eq!(trace.gpu_exact_program_compiles, 0);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_results_gate_batched_conditioned_parsed_program_queries() {
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
    let assumptions_a = [EpistemicAssumption::known_tuple(
        "edge",
        vec![EpistemicEvidenceTerm::integer(1)],
        true,
    )];
    let assumptions_b = [EpistemicAssumption::known_tuple(
        "edge",
        vec![EpistemicEvidenceTerm::integer(2)],
        true,
    )];
    let prob_program = parse_program(
        r#"
        0.2::edge(1).
        0.8::edge(2).
        query(edge(1)).
        query(edge(2)).
        "#,
    )
    .expect("parse probabilistic program");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_program_for_gpu_execution_results(
            &prob_program,
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
        .expect("accepted GPU runtime evidence must gate batched parsed-program conditions");

    assert_eq!(evaluated.len(), 2);
    assert_eq!(evaluated[0].query_probs.len(), 2);
    assert!((evaluated[0].query_probs[0].prob - 1.0).abs() < 1.0e-6);
    assert!((evaluated[0].query_probs[1].prob - 0.8).abs() < 1.0e-6);
    assert_eq!(evaluated[1].query_probs.len(), 2);
    assert!((evaluated[1].query_probs[0].prob - 0.2).abs() < 1.0e-6);
    assert!((evaluated[1].query_probs[1].prob - 1.0).abs() < 1.0e-6);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 0);
    assert_eq!(trace.gpu_exact_program_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_result_conditions_probabilistic_gradient_evidence() {
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
        .expect("execute nonzero-arity accepted epistemic fixture");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let grads = adapter
        .compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result(
            r#"
            0.2::edge(1).
            0.8::edge(2).
            query(edge(1)).
            query(edge(2)).
            "#,
            &fix.provider,
            &result,
            vec![EpistemicAssumption::known_tuple(
                "edge",
                vec![EpistemicEvidenceTerm::integer(1)],
                true,
            )],
        )
        .expect("accepted GPU runtime evidence must condition probabilistic gradient evaluation");

    assert_eq!(grads.query_grads.len(), 2);
    assert!((grads.query_grads[0].prob - 1.0).abs() < 1.0e-6);
    assert!((grads.query_grads[1].prob - 0.8).abs() < 1.0e-6);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 1);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 1);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 1);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_program_compiles, 0);
    assert_eq!(trace.gpu_exact_query_evaluations, 0);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_results_gate_batched_conditioned_parsed_program_gradients() {
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
    let assumptions_a = [EpistemicAssumption::known_tuple(
        "edge",
        vec![EpistemicEvidenceTerm::integer(1)],
        true,
    )];
    let assumptions_b = [EpistemicAssumption::known_tuple(
        "edge",
        vec![EpistemicEvidenceTerm::integer(2)],
        true,
    )];
    let prob_program = parse_program(
        r#"
        0.2::edge(1).
        0.8::edge(2).
        query(edge(1)).
        query(edge(2)).
        "#,
    )
    .expect("parse probabilistic program");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let evaluated = adapter
        .compile_and_evaluate_conditioned_program_with_grads_for_gpu_execution_results(
            &prob_program,
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
        .expect("accepted GPU runtime evidence must gate batched conditioned gradients");

    assert_eq!(evaluated.len(), 2);
    assert_eq!(evaluated[0].query_grads.len(), 2);
    assert!((evaluated[0].query_grads[0].prob - 1.0).abs() < 1.0e-6);
    assert!((evaluated[0].query_grads[1].prob - 0.8).abs() < 1.0e-6);
    assert_eq!(evaluated[1].query_grads.len(), 2);
    assert!((evaluated[1].query_grads[0].prob - 0.2).abs() < 1.0e-6);
    assert!((evaluated[1].query_grads[1].prob - 1.0).abs() < 1.0e-6);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_conditioned_evidence_facts, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 0);
    assert_eq!(trace.gpu_exact_program_compiles, 2);
    assert_eq!(trace.gpu_exact_query_evaluations, 0);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 0);
    assert_eq!(trace.gpu_program_knowledge_compilation_end_to_end_runs, 2);
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
fn probabilistic_end_to_end_records_source_and_program_query_trace_counters() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let source_result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
        &[1],
        &[1],
    );
    let program_result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), not know edge(X).
        "#,
        &[1],
        &[],
    );
    let prob_program = parse_program(
        r#"
        0.5::rain().
        query(rain()).
        "#,
    )
    .expect("parse probabilistic end-to-end program");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let source_evaluated = adapter
        .compile_and_evaluate_source_with_gpu_execution_result(
            r#"
            0.5::rain().
            query(rain()).
            "#,
            &fix.provider,
            &source_result,
            vec![EpistemicAssumption::known("edge", 1, true)],
        )
        .expect("accepted source evidence must gate source exact query path");
    let program_evaluated = adapter
        .compile_and_evaluate_program_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &program_result,
            vec![EpistemicAssumption::known("edge", 1, false)],
        )
        .expect("accepted program evidence must gate parsed-program exact query path");

    assert_eq!(source_evaluated.query_probs.len(), 1);
    assert!((source_evaluated.query_probs[0].prob - 0.5).abs() < 1.0e-6);
    assert_eq!(program_evaluated.query_probs.len(), 1);
    assert!((program_evaluated.query_probs[0].prob - 0.5).abs() < 1.0e-6);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_program_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 2);
    assert_eq!(trace.gpu_source_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_program_exact_query_evaluations, 1);
    assert_eq!(trace.gpu_knowledge_compilation_end_to_end_runs, 2);
    assert_eq!(trace.gpu_source_knowledge_compilation_end_to_end_runs, 1);
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
fn probabilistic_pir_cnf_records_source_and_program_trace_counters() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let source_result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
        &[1],
        &[1],
    );
    let program_result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), not know edge(X).
        "#,
        &[1],
        &[],
    );
    let prob_program = parse_program(
        r#"
        0.5::rain().
        query(rain()).
        "#,
    )
    .expect("parse probabilistic PIR/CNF program");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let source_pir_cnf = adapter
        .encode_source_pir_cnf_with_gpu_execution_result(
            r#"
            0.5::rain().
            query(rain()).
            "#,
            &fix.provider,
            &source_result,
            vec![EpistemicAssumption::known("edge", 1, true)],
        )
        .expect("accepted source evidence must gate source PIR/CNF path");
    let program_pir_cnf = adapter
        .encode_program_pir_cnf_with_gpu_execution_result(
            &prob_program,
            &fix.provider,
            &program_result,
            vec![EpistemicAssumption::known("edge", 1, false)],
        )
        .expect("accepted program evidence must gate parsed-program PIR/CNF path");

    assert!(source_pir_cnf.pir_nodes > 0);
    assert!(source_pir_cnf.root_count > 0);
    assert!(program_pir_cnf.pir_nodes > 0);
    assert!(program_pir_cnf.root_count > 0);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_pir_graph_uploads, 2);
    assert_eq!(trace.gpu_cnf_encodes, 2);
    assert_eq!(trace.gpu_source_pir_graph_uploads, 1);
    assert_eq!(trace.gpu_program_pir_graph_uploads, 1);
    assert_eq!(trace.gpu_source_cnf_encodes, 1);
    assert_eq!(trace.gpu_program_cnf_encodes, 1);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_results_gate_batched_probabilistic_source_pir_cnf_path() {
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
    let assumptions_b = [EpistemicAssumption::known("edge", 2, true)];

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let pir_cnfs = adapter
        .encode_source_pir_cnf_for_gpu_execution_results(
            r#"
            0.5::rain().
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
        .expect("accepted GPU runtime evidence must gate batched source PIR/CNF path");

    assert_eq!(pir_cnfs.len(), 2);
    for pir_cnf in &pir_cnfs {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_pir_graph_uploads, 2);
    assert_eq!(trace.gpu_cnf_encodes, 2);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_gpu_execution_results_gate_batched_probabilistic_program_pir_cnf_path() {
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
    let assumptions_b = [EpistemicAssumption::known("edge", 2, true)];
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
    let pir_cnfs = adapter
        .encode_program_pir_cnf_for_gpu_execution_results(
            &prob_program,
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
        .expect("accepted GPU runtime evidence must gate batched parsed-program PIR/CNF path");

    assert_eq!(pir_cnfs.len(), 2);
    for pir_cnf in &pir_cnfs {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 2);
    assert_eq!(trace.gpu_pir_graph_uploads, 2);
    assert_eq!(trace.gpu_cnf_encodes, 2);
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
fn accepted_gpu_execution_results_gate_batched_probabilistic_gradient_evaluations() {
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
    let assumptions_b = [EpistemicAssumption::known("edge", 2, true)];

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
            &result_a,
            vec![EpistemicAssumption::known("edge", 1, true)],
        )
        .expect("accepted GPU runtime evidence must gate probabilistic exact compile");
    let evaluated = adapter
        .evaluate_gpu_with_grads_for_gpu_execution_results(
            &exact,
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
        .expect("accepted GPU runtime evidence must gate batched gradient evaluation");

    assert_eq!(evaluated.len(), 2);
    for result in &evaluated {
        assert_eq!(result.query_grads.len(), 1);
        assert!((result.query_grads[0].prob - 0.5).abs() < 1.0e-6);
    }
    let trace = adapter.trace();
    assert_eq!(trace.accepted_world_view_evidence_consumed, 3);
    assert_eq!(trace.accepted_evidence_assumptions_consumed, 3);
    assert_eq!(trace.gpu_exact_source_compiles, 1);
    assert_eq!(trace.gpu_exact_query_evaluations, 0);
    assert_eq!(trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(trace.cpu_only_probability_recomputations, 0);
    assert_eq!(trace.fixture_circuit_evaluations, 0);
}

#[test]
fn accepted_split_batch_gates_probabilistic_pir_cnf_and_exact_evaluation_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse split epistemic fixture");
    let split = compile_epistemic_gpu_split_execution_with_stats_snapshot(&program, None)
        .expect("compile split components through GPU executable path");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let mut relation_ids = BTreeMap::new();
    for component in &split.components {
        for (name, rel_id) in &component.executable.relation_ids {
            if let Some(previous) = relation_ids.insert(name.clone(), *rel_id) {
                assert_eq!(
                    previous, *rel_id,
                    "split components must preserve relation ids for shared declarations"
                );
            }
        }
    }
    for (name, rel_id) in &relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("node", upload_unary_u32(&fix.memory, &[1, 2, 3]));
    executor.put_relation("edge", upload_unary_u32(&fix.memory, &[1, 3]));
    executor.put_relation("color", upload_unary_u32(&fix.memory, &[2]));

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 2,
            },
        )
        .expect("execute split GPU components through runtime batch path");

    let empty_assumptions: [EpistemicAssumption; 0] = [];
    let assumption_groups: [&[EpistemicAssumption]; 2] = [&empty_assumptions, &empty_assumptions];
    let probabilistic_source = r#"
        0.3::edge(1).
        0.7::color(2).
        query(edge(1)).
        query(color(2)).
        "#;
    let parsed_probabilistic_program = parse_program(probabilistic_source)
        .expect("parse split-batch probabilistic source/program fixture");

    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 64 * 1024 * 1024;

    let mut source_pir_adapter = EpistemicProbProductionAdapter::new(config);
    let source_pir_cnfs = source_pir_adapter
        .encode_source_pir_cnf_for_gpu_batch_execution_result(
            probabilistic_source,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split GPU batch evidence must gate source PIR/CNF encoding");
    assert_eq!(source_pir_cnfs.len(), 2);
    for pir_cnf in &source_pir_cnfs {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let source_pir_trace = source_pir_adapter.trace();
    assert_eq!(source_pir_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        source_pir_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(source_pir_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(source_pir_trace.gpu_pir_graph_uploads, 2);
    assert_eq!(source_pir_trace.gpu_source_pir_graph_uploads, 2);
    assert_eq!(source_pir_trace.gpu_program_pir_graph_uploads, 0);
    assert_eq!(source_pir_trace.gpu_cnf_encodes, 2);
    assert_eq!(source_pir_trace.gpu_source_cnf_encodes, 2);
    assert_eq!(source_pir_trace.gpu_program_cnf_encodes, 0);
    assert_eq!(source_pir_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(source_pir_trace.fixture_circuit_evaluations, 0);

    let mut program_pir_adapter = EpistemicProbProductionAdapter::new(config);
    let program_pir_cnfs = program_pir_adapter
        .encode_program_pir_cnf_for_gpu_batch_execution_result(
            &parsed_probabilistic_program,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split GPU batch evidence must gate parsed-program PIR/CNF encoding");
    assert_eq!(program_pir_cnfs.len(), 2);
    for pir_cnf in &program_pir_cnfs {
        assert!(pir_cnf.pir_nodes > 0);
        assert!(pir_cnf.root_count > 0);
        assert!(pir_cnf.cnf_var_cap > 0);
        assert!(pir_cnf.cnf_clause_cap > 0);
    }
    let program_pir_trace = program_pir_adapter.trace();
    assert_eq!(program_pir_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        program_pir_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(program_pir_trace.accepted_world_view_evidence_consumed, 2);
    assert_eq!(program_pir_trace.gpu_pir_graph_uploads, 2);
    assert_eq!(program_pir_trace.gpu_source_pir_graph_uploads, 0);
    assert_eq!(program_pir_trace.gpu_program_pir_graph_uploads, 2);
    assert_eq!(program_pir_trace.gpu_cnf_encodes, 2);
    assert_eq!(program_pir_trace.gpu_source_cnf_encodes, 0);
    assert_eq!(program_pir_trace.gpu_program_cnf_encodes, 2);
    assert_eq!(program_pir_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(program_pir_trace.fixture_circuit_evaluations, 0);

    let mut query_adapter = EpistemicProbProductionAdapter::new(config);
    let exact = query_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &batch.results[0],
            vec![],
        )
        .expect("accepted split component must gate probabilistic exact compile");
    let evaluated = query_adapter
        .evaluate_for_gpu_batch_execution_result(
            &exact,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split GPU batch evidence must gate exact query evaluation");
    assert_eq!(evaluated.len(), 2);
    for result in &evaluated {
        assert_eq!(result.query_probs.len(), 2);
        assert!((result.query_probs[0].prob - 0.3).abs() < 1.0e-6);
        assert!((result.query_probs[1].prob - 0.7).abs() < 1.0e-6);
    }
    let query_trace = query_adapter.trace();
    assert_eq!(query_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        query_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(query_trace.accepted_world_view_evidence_consumed, 3);
    assert_eq!(query_trace.gpu_exact_source_compiles, 1);
    assert_eq!(query_trace.gpu_exact_query_evaluations, 2);
    assert_eq!(query_trace.gpu_exact_gradient_evaluations, 0);
    assert_eq!(query_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(query_trace.fixture_circuit_evaluations, 0);

    let mut gradient_adapter = EpistemicProbProductionAdapter::new(config);
    let exact = gradient_adapter
        .compile_source_with_gpu_execution_result(
            probabilistic_source,
            &fix.provider,
            &batch.results[0],
            vec![],
        )
        .expect("accepted split component must gate probabilistic exact compile");
    let gradients = gradient_adapter
        .evaluate_gpu_with_grads_for_gpu_batch_execution_result(
            &exact,
            &fix.provider,
            EpistemicProbGpuBatchExecutionEvidence {
                batch: &batch,
                assumptions_by_component: &assumption_groups,
            },
        )
        .expect("accepted split GPU batch evidence must gate exact gradient evaluation");
    assert_eq!(gradients.len(), 2);
    for result in &gradients {
        assert_eq!(result.query_grads.len(), 2);
        assert!((result.query_grads[0].prob - 0.3).abs() < 1.0e-6);
        assert!((result.query_grads[1].prob - 0.7).abs() < 1.0e-6);
    }
    let gradient_trace = gradient_adapter.trace();
    assert_eq!(gradient_trace.accepted_gpu_batch_evidence_consumed, 1);
    assert_eq!(
        gradient_trace.accepted_gpu_batch_component_evidence_consumed,
        2
    );
    assert_eq!(gradient_trace.accepted_world_view_evidence_consumed, 3);
    assert_eq!(gradient_trace.gpu_exact_source_compiles, 1);
    assert_eq!(gradient_trace.gpu_exact_query_evaluations, 0);
    assert_eq!(gradient_trace.gpu_exact_gradient_evaluations, 2);
    assert_eq!(gradient_trace.cpu_only_probability_recomputations, 0);
    assert_eq!(gradient_trace.fixture_circuit_evaluations, 0);
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
fn accepted_gpu_execution_result_gates_solver_maxsat_lifecycle_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
        &[1, 2],
        &[1],
    );

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
        .expect("new MaxSAT lifecycle workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let report = adapter
        .solve_maxsat_lifecycle_with_gpu_execution_result(
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
        )
        .expect("accepted GPU runtime evidence must gate MaxSAT lifecycle path");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.lifecycle.candidate_evidence_records, 0);
    assert_eq!(report.lifecycle.steps, 2);
    assert_eq!(report.lifecycle.assumption_pushes, 2);
    assert_eq!(report.lifecycle.assumption_retractions, 2);
    assert_eq!(report.lifecycle.workspace_reuses, 1);
    assert_eq!(report.maxsat.candidate_evidence_records, 0);
    assert_eq!(report.maxsat.optimum_score, 7);
    assert_eq!(report.maxsat.candidates_checked, 2);
    assert_eq!(report.maxsat.satisfiable_candidates, 2);
    assert_eq!(report.maxsat.gpu_cdcl_candidate_solves, 2);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_assumption_pushes, 2);
    assert_eq!(trace.gpu_assumption_retractions, 2);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 1);
    assert_eq!(trace.gpu_cdcl_sat_solves, 3);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(trace.gpu_maxsat_optima, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_result_rejects_empty_maxsat_lifecycle_before_lifecycle_work() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
        &[1, 2],
        &[1],
    );

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
        .expect("new MaxSAT lifecycle workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let err = adapter
        .solve_maxsat_lifecycle_with_gpu_execution_result(
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
            &[],
        )
        .expect_err("empty MaxSAT lifecycle candidates must fail before lifecycle work");
    assert!(format!("{err}").contains("bounded MaxSAT adapter requires at least one candidate CNF"));

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.gpu_assumption_pushes, 0);
    assert_eq!(trace.gpu_assumption_retractions, 0);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 0);
    assert_eq!(trace.gpu_cdcl_sat_solves, 0);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 0);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 0);
    assert_eq!(trace.gpu_maxsat_optima, 0);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_operator_gpu_execution_results_gate_solver_lifecycle_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let possible_result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), possible edge(X).
        "#,
        &[1, 2],
        &[1],
    );
    let not_possible_result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), not possible edge(X).
        "#,
        &[1, 2],
        &[2],
    );
    let binary_possible_result = execute_binary_edge_epistemic_fixture(
        &fix,
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred accepted(u32, u32).
        accepted(X, Y) :- pair(X, Y), possible edge(X, Y).
        "#,
        &[(1, 2), (2, 3), (3, 4)],
        &[(1, 2), (3, 4)],
    );
    let binary_not_possible_result = execute_binary_edge_epistemic_fixture(
        &fix,
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred accepted(u32, u32).
        accepted(X, Y) :- pair(X, Y), not possible edge(X, Y).
        "#,
        &[(1, 2), (2, 3), (3, 4)],
        &[(2, 3)],
    );
    let not_know_result = execute_binary_edge_epistemic_fixture(
        &fix,
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred accepted(u32, u32).
        accepted(X, Y) :- pair(X, Y), not know edge(X, Y).
        "#,
        &[(1, 2), (2, 3), (3, 4)],
        &[(2, 3)],
    );
    assert_eq!(
        download_unary_u32(&fix.provider, &possible_result.final_output),
        vec![1]
    );
    assert_eq!(
        download_unary_u32(&fix.provider, &not_possible_result.final_output),
        vec![1]
    );
    assert_eq!(
        download_binary_u32(&fix.provider, &binary_possible_result.final_output),
        vec![(1, 2), (3, 4)]
    );
    assert_eq!(
        download_binary_u32(&fix.provider, &binary_not_possible_result.final_output),
        vec![(1, 2), (3, 4)]
    );
    assert_eq!(
        download_binary_u32(&fix.provider, &not_know_result.final_output),
        vec![(1, 2), (3, 4)]
    );
    assert_eq!(
        possible_result.prepared.preflight.possible_operator_count,
        1
    );
    assert_eq!(
        not_possible_result
            .prepared
            .preflight
            .not_possible_operator_count,
        1
    );
    assert_eq!(
        binary_possible_result
            .prepared
            .preflight
            .possible_operator_count,
        1
    );
    assert_eq!(
        binary_not_possible_result
            .prepared
            .preflight
            .not_possible_operator_count,
        1
    );
    assert_eq!(
        binary_not_possible_result
            .final_tuple_materialization
            .negated_row_filter_count,
        1
    );
    assert_eq!(
        not_know_result.prepared.preflight.not_know_operator_count,
        1
    );
    assert_eq!(
        not_know_result
            .final_tuple_materialization
            .negated_row_filter_count,
        1
    );

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
            &[
                &possible_result,
                &not_possible_result,
                &binary_possible_result,
                &binary_not_possible_result,
                &not_know_result,
            ],
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
        .expect("accepted operator GPU evidence must gate solver lifecycle path");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(report.candidate_evidence_records, 5);
    assert_eq!(report.steps, 10);
    assert_eq!(report.assumption_pushes, 10);
    assert_eq!(report.assumption_retractions, 10);
    assert_eq!(report.workspace_reuses, 5);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 5);
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.accepted_possible_gpu_candidate_evidence_consumed, 2);
    assert_eq!(
        trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        2
    );
    assert_eq!(trace.accepted_not_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_assumption_pushes, 10);
    assert_eq!(trace.gpu_assumption_retractions, 10);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 5);
    assert_eq!(trace.gpu_cdcl_sat_solves, 5);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 5);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_all_operator_mixed_membership_gates_solver_lifecycle_path() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_all_operator_mixed_membership_fixture(&fix);
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![2]
    );
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 4);
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(result.final_tuple_materialization.row_filter_count, 4);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        2
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

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
        .expect("same-rule all-operator GPU evidence must gate solver lifecycle path");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.steps, 2);
    assert_eq!(report.assumption_pushes, 2);
    assert_eq!(report.assumption_retractions, 2);
    assert_eq!(report.workspace_reuses, 1);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.accepted_possible_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(trace.accepted_not_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        4
    );
    assert_eq!(trace.gpu_assumption_pushes, 2);
    assert_eq!(trace.gpu_assumption_retractions, 2);
    assert_eq!(trace.gpu_lifecycle_workspace_reuses, 1);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_all_operator_mixed_membership_gates_solver_reuse_maxsat_and_portfolio_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_all_operator_mixed_membership_fixture(&fix);
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![2]
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );

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

    let learned = adapter
        .solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut workspace,
            &unsat_cnf,
            &branch_limit,
            &unsat_cnf,
            &branch_limit,
        )
        .expect("same-rule all-operator GPU evidence must gate learned-clause reuse");
    let maxsat = adapter
        .solve_weighted_maxsat_candidates_with_gpu_execution_result(
            &fix.provider,
            &result,
            &[GpuSolverProductionMaxSatCandidate {
                score: 11,
                cnf: &sat_cnf,
                branch_var_limit: &branch_limit,
            }],
        )
        .expect("same-rule all-operator GPU evidence must gate MaxSAT through GPU CDCL");
    let portfolio = adapter
        .solve_portfolio_with_gpu_execution_result(
            &fix.provider,
            &result,
            &[
                GpuSolverProductionPortfolioJob::Sat {
                    cnf: &sat_cnf,
                    branch_var_limit: &branch_limit,
                },
                GpuSolverProductionPortfolioJob::MaxSat {
                    candidates: &[GpuSolverProductionMaxSatCandidate {
                        score: 11,
                        cnf: &sat_cnf,
                        branch_var_limit: &branch_limit,
                    }],
                },
                GpuSolverProductionPortfolioJob::Unknown {
                    reason: "bounded branch budget exhausted before a determined status",
                },
                GpuSolverProductionPortfolioJob::Timeout { budget_micros: 10 },
            ],
        )
        .expect("same-rule all-operator GPU evidence must gate status-aware portfolio path");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(learned.candidate_evidence_records, 1);
    assert_eq!(learned.candidates, 2);
    assert_eq!(learned.unsat_solves, 2);
    assert_eq!(learned.gpu_learned_clause_arena_publications, 1);
    assert_eq!(learned.gpu_learned_clause_imports, 1);
    assert_eq!(learned.gpu_learned_clause_reused_solves, 1);
    assert_eq!(learned.cpu_learned_clause_transfers, 0);
    assert_eq!(maxsat.candidate_evidence_records, 1);
    assert_eq!(maxsat.optimum_score, 11);
    assert_eq!(maxsat.gpu_cdcl_candidate_solves, 1);
    assert_eq!(portfolio.jobs, 4);
    assert_eq!(portfolio.sat_jobs, 1);
    assert_eq!(portfolio.maxsat_jobs, 1);
    assert_eq!(portfolio.unknown_jobs, 1);
    assert_eq!(portfolio.timeout_jobs, 1);
    assert_eq!(portfolio.maxsat_optimum_scores, 11);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 3);
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 3);
    assert_eq!(trace.accepted_possible_gpu_candidate_evidence_consumed, 3);
    assert_eq!(
        trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        3
    );
    assert_eq!(trace.accepted_not_know_gpu_candidate_evidence_consumed, 3);
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        3
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        12
    );
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.gpu_learned_clause_arena_publications, 1);
    assert_eq!(trace.gpu_learned_clause_imports, 1);
    assert_eq!(trace.gpu_learned_clause_reused_solves, 1);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
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

#[test]
fn accepted_all_operator_mixed_membership_gates_solver_search_and_scheduler_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_all_operator_mixed_membership_fixture(&fix);
    assert_eq!(
        download_unary_u32(&fix.provider, &result.final_output),
        vec![2]
    );
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let sat_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let unsat = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let gpu_sat_low = GpuCnf::from_host(&sat_low, &fix.provider).expect("upload SAT low");
    let gpu_sat_high = GpuCnf::from_host(&sat_high, &fix.provider).expect("upload SAT high");
    let gpu_unsat = GpuCnf::from_host(&unsat, &fix.provider).expect("upload UNSAT candidate");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);

    let mut search_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut search_workspace = search_adapter
        .new_workspace(gpu_unsat.var_cap, gpu_unsat.clause_cap)
        .expect("new all-operator MaxSAT search workspace");
    let search_report = search_adapter
        .solve_weighted_maxsat_search_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut search_workspace,
            &[
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 7,
                    cnf: &gpu_sat_low,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                },
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 11,
                    cnf: &gpu_unsat,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                },
            ],
        )
        .expect("same-rule all-operator GPU evidence must gate MaxSAT search");

    assert_eq!(search_report.candidate_evidence_records, 1);
    assert_eq!(search_report.optimum_score, 7);
    assert_eq!(search_report.candidates_checked, 2);
    assert_eq!(search_report.satisfiable_candidates, 1);
    assert_eq!(search_report.unsat_candidates_pruned, 1);
    assert_eq!(search_report.gpu_cdcl_candidate_solves, 2);

    let search_trace = search_adapter.trace();
    assert_eq!(search_trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        search_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        4
    );
    assert_eq!(search_trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(search_trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(search_trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(search_trace.gpu_maxsat_unsat_candidate_prunes, 1);
    assert_eq!(search_trace.gpu_maxsat_optima, 1);
    assert_eq!(search_trace.cpu_assignment_enumerations, 0);
    assert_eq!(search_trace.cpu_maxsat_enumerations, 0);

    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![7.0, 9.0],
    );
    let both_soft_clauses = [0usize, 1usize];
    let sat_soft_clause = [0usize];
    let selections = [
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &both_soft_clauses,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &sat_soft_clause,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];

    let mut encoded_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut encoded_workspace = encoded_adapter
        .new_workspace(1, 2)
        .expect("new all-operator encoded MaxSAT workspace");
    let encoded_report = encoded_adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut encoded_workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("same-rule all-operator GPU evidence must gate encoded MaxSAT");

    assert_eq!(encoded_report.candidate_evidence_records, 1);
    assert_eq!(encoded_report.optimum_score, 7);
    assert_eq!(encoded_report.candidates_checked, 2);
    assert_eq!(encoded_report.satisfiable_candidates, 1);
    assert_eq!(encoded_report.unsat_candidates_pruned, 1);
    assert_eq!(encoded_report.gpu_cdcl_candidate_encodes, 2);
    assert_eq!(encoded_report.gpu_cdcl_candidate_solves, 2);

    let encoded_trace = encoded_adapter.trace();
    assert_eq!(encoded_trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        encoded_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        4
    );
    assert_eq!(encoded_trace.gpu_maxsat_candidate_encodes, 2);
    assert_eq!(encoded_trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(encoded_trace.gpu_maxsat_unsat_candidate_prunes, 1);
    assert_eq!(encoded_trace.gpu_maxsat_optima, 1);
    assert_eq!(encoded_trace.cpu_assignment_enumerations, 0);
    assert_eq!(encoded_trace.cpu_maxsat_enumerations, 0);

    let candidate_set = [
        GpuSolverProductionMaxSatCandidate {
            score: 3,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionMaxSatCandidate {
            score: 5,
            cnf: &gpu_sat_high,
            branch_var_limit: &branch_limit,
        },
    ];
    let search_candidates = [
        GpuSolverProductionMaxSatSearchCandidate {
            score: 9,
            cnf: &gpu_unsat,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionMaxSatSearchCandidate {
            score: 7,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];
    let jobs = [
        GpuSolverProductionMaxSatScheduleJob::CandidateSet {
            candidates: &candidate_set,
        },
        GpuSolverProductionMaxSatScheduleJob::Search {
            candidates: &search_candidates,
        },
        GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
            weighted: &weighted,
            branch_var_limit: &branch_limit,
            selections: &selections,
        },
        GpuSolverProductionMaxSatScheduleJob::Unknown {
            reason: "all-operator scheduler branch budget exhausted",
        },
        GpuSolverProductionMaxSatScheduleJob::Timeout { budget_micros: 25 },
    ];
    let mut scheduler_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut scheduler_workspace = scheduler_adapter
        .new_workspace(1, 2)
        .expect("new all-operator generalized MaxSAT scheduler workspace");
    let schedule_report = scheduler_adapter
        .solve_maxsat_schedule_with_gpu_execution_results(
            &fix.provider,
            &[&result],
            &mut scheduler_workspace,
            &jobs,
        )
        .expect("same-rule all-operator GPU evidence must gate MaxSAT scheduler");

    assert_eq!(schedule_report.candidate_evidence_records, 1);
    assert_eq!(schedule_report.jobs, 5);
    assert_eq!(schedule_report.candidate_set_jobs, 1);
    assert_eq!(schedule_report.search_jobs, 1);
    assert_eq!(schedule_report.encoded_search_jobs, 1);
    assert_eq!(schedule_report.unknown_jobs, 1);
    assert_eq!(schedule_report.timeout_jobs, 1);
    assert_eq!(schedule_report.optimum_score, 7);
    assert_eq!(schedule_report.candidates_checked, 6);
    assert_eq!(schedule_report.satisfiable_candidates, 4);
    assert_eq!(schedule_report.unsat_candidates_pruned, 2);
    assert_eq!(schedule_report.gpu_cdcl_candidate_encodes, 2);
    assert_eq!(schedule_report.gpu_cdcl_candidate_solves, 6);

    let scheduler_trace = scheduler_adapter.trace();
    assert_eq!(scheduler_trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        scheduler_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        4
    );
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_jobs, 5);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_candidate_set_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_search_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_encoded_search_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_unknown_status_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_timeout_status_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_encodes, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_solves, 6);
    assert_eq!(scheduler_trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_optima, 3);
    assert_eq!(scheduler_trace.cpu_assignment_enumerations, 0);
    assert_eq!(scheduler_trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_ternary_gpu_execution_result_records_solver_nonzero_arity_evidence_trace() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred triple(u32, u32, u32).
        pred fact3(u32, u32, u32).
        pred accepted(u32, u32, u32).
        accepted(A, B, C) :- triple(A, B, C), know fact3(A, B, C).
        "#,
    )
    .expect("parse ternary solver evidence fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile ternary solver evidence executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "triple",
        upload_ternary_u32(&fix.memory, &[(1, 2, 3), (2, 3, 5), (8, 13, 21)]),
    );
    executor.put_relation("fact3", upload_ternary_u32(&fix.memory, &[(2, 3, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute ternary solver evidence fixture");
    assert_eq!(
        download_ternary_u32(&fix.provider, &result.final_output),
        vec![(2, 3, 5)]
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        3
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let assignment = adapter
        .solve_expect_sat_with_gpu_execution_result(&fix.provider, &result, &sat_cnf)
        .expect("accepted ternary GPU evidence must gate solver SAT path");
    assert_ne!(*assignment.device_ptr(), 0);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        3
    );
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_quaternary_gpu_execution_result_records_solver_nonzero_arity_evidence_trace() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary solver evidence fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary solver evidence executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary solver evidence fixture");
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(2, 3, 4, 5)]
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let assignment = adapter
        .solve_expect_sat_with_gpu_execution_result(&fix.provider, &result, &sat_cnf)
        .expect("accepted quaternary GPU evidence must gate solver SAT path");
    assert_ne!(*assignment.device_ptr(), 0);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        4
    );
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_quaternary_gpu_execution_result_gates_solver_reuse_maxsat_and_portfolio_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary solver reuse fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary solver reuse executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary solver reuse fixture");
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(2, 3, 4, 5)]
    );
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 0);
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        0
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);

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
        .expect("new quaternary solver reuse workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let learned = adapter
        .solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut workspace,
            &unsat_cnf,
            &branch_limit,
            &unsat_cnf,
            &branch_limit,
        )
        .expect("accepted quaternary GPU evidence must gate learned-clause reuse");
    let maxsat = adapter
        .solve_weighted_maxsat_candidates_with_gpu_execution_result(
            &fix.provider,
            &result,
            &[GpuSolverProductionMaxSatCandidate {
                score: 13,
                cnf: &sat_cnf,
                branch_var_limit: &branch_limit,
            }],
        )
        .expect("accepted quaternary GPU evidence must gate MaxSAT");
    let portfolio = adapter
        .solve_portfolio_with_gpu_execution_result(
            &fix.provider,
            &result,
            &[
                GpuSolverProductionPortfolioJob::Sat {
                    cnf: &sat_cnf,
                    branch_var_limit: &branch_limit,
                },
                GpuSolverProductionPortfolioJob::MaxSat {
                    candidates: &[GpuSolverProductionMaxSatCandidate {
                        score: 13,
                        cnf: &sat_cnf,
                        branch_var_limit: &branch_limit,
                    }],
                },
                GpuSolverProductionPortfolioJob::Unknown {
                    reason: "quaternary know branch budget exhausted",
                },
                GpuSolverProductionPortfolioJob::Timeout { budget_micros: 10 },
            ],
        )
        .expect("accepted quaternary GPU evidence must gate portfolio path");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(learned.candidate_evidence_records, 1);
    assert_eq!(learned.candidates, 2);
    assert_eq!(learned.unsat_solves, 2);
    assert_eq!(learned.gpu_learned_clause_arena_publications, 1);
    assert_eq!(learned.gpu_learned_clause_imports, 1);
    assert_eq!(learned.gpu_learned_clause_reused_solves, 1);
    assert_eq!(learned.cpu_learned_clause_transfers, 0);
    assert_eq!(maxsat.candidate_evidence_records, 1);
    assert_eq!(maxsat.optimum_score, 13);
    assert_eq!(maxsat.gpu_cdcl_candidate_solves, 1);
    assert_eq!(portfolio.jobs, 4);
    assert_eq!(portfolio.sat_jobs, 1);
    assert_eq!(portfolio.maxsat_jobs, 1);
    assert_eq!(portfolio.unknown_jobs, 1);
    assert_eq!(portfolio.timeout_jobs, 1);
    assert_eq!(portfolio.maxsat_optimum_scores, 13);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 3);
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        3
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        12
    );
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 3);
    assert_eq!(trace.accepted_possible_gpu_candidate_evidence_consumed, 0);
    assert_eq!(
        trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(trace.accepted_not_know_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.gpu_learned_clause_arena_publications, 1);
    assert_eq!(trace.gpu_learned_clause_imports, 1);
    assert_eq!(trace.gpu_learned_clause_reused_solves, 1);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
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

#[test]
fn accepted_quaternary_gpu_execution_result_gates_solver_search_and_scheduler_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary solver search fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary solver search executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary solver search fixture");
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(2, 3, 4, 5)]
    );
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 0);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 0);
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        0
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);

    let sat_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let unsat = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let gpu_sat_low = GpuCnf::from_host(&sat_low, &fix.provider).expect("upload SAT low");
    let gpu_sat_high = GpuCnf::from_host(&sat_high, &fix.provider).expect("upload SAT high");
    let gpu_unsat = GpuCnf::from_host(&unsat, &fix.provider).expect("upload UNSAT candidate");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);

    let mut search_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut search_workspace = search_adapter
        .new_workspace(gpu_unsat.var_cap, gpu_unsat.clause_cap)
        .expect("new quaternary MaxSAT search workspace");
    let search_report = search_adapter
        .solve_weighted_maxsat_search_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut search_workspace,
            &[
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 7,
                    cnf: &gpu_sat_low,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                },
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 11,
                    cnf: &gpu_unsat,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                },
            ],
        )
        .expect("accepted quaternary GPU evidence must gate MaxSAT search");

    assert_eq!(search_report.candidate_evidence_records, 1);
    assert_eq!(search_report.optimum_score, 7);
    assert_eq!(search_report.candidates_checked, 2);
    assert_eq!(search_report.satisfiable_candidates, 1);
    assert_eq!(search_report.unsat_candidates_pruned, 1);
    assert_eq!(search_report.gpu_cdcl_candidate_solves, 2);

    let search_trace = search_adapter.trace();
    assert_eq!(search_trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        search_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        4
    );
    assert_eq!(
        search_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        search_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        search_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(search_trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(search_trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(search_trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(search_trace.gpu_maxsat_unsat_candidate_prunes, 1);
    assert_eq!(search_trace.gpu_maxsat_optima, 1);
    assert_eq!(search_trace.cpu_assignment_enumerations, 0);
    assert_eq!(search_trace.cpu_maxsat_enumerations, 0);

    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![7.0, 9.0],
    );
    let both_soft_clauses = [0usize, 1usize];
    let sat_soft_clause = [0usize];
    let selections = [
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &both_soft_clauses,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &sat_soft_clause,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];

    let mut encoded_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut encoded_workspace = encoded_adapter
        .new_workspace(1, 2)
        .expect("new quaternary encoded MaxSAT workspace");
    let encoded_report = encoded_adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut encoded_workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("accepted quaternary GPU evidence must gate encoded MaxSAT");

    assert_eq!(encoded_report.candidate_evidence_records, 1);
    assert_eq!(encoded_report.optimum_score, 7);
    assert_eq!(encoded_report.candidates_checked, 2);
    assert_eq!(encoded_report.satisfiable_candidates, 1);
    assert_eq!(encoded_report.unsat_candidates_pruned, 1);
    assert_eq!(encoded_report.gpu_cdcl_candidate_encodes, 2);
    assert_eq!(encoded_report.gpu_cdcl_candidate_solves, 2);

    let encoded_trace = encoded_adapter.trace();
    assert_eq!(encoded_trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        encoded_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        4
    );
    assert_eq!(
        encoded_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        encoded_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        encoded_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(encoded_trace.gpu_maxsat_candidate_encodes, 2);
    assert_eq!(encoded_trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(encoded_trace.gpu_maxsat_unsat_candidate_prunes, 1);
    assert_eq!(encoded_trace.gpu_maxsat_optima, 1);
    assert_eq!(encoded_trace.cpu_assignment_enumerations, 0);
    assert_eq!(encoded_trace.cpu_maxsat_enumerations, 0);

    let candidate_set = [
        GpuSolverProductionMaxSatCandidate {
            score: 3,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionMaxSatCandidate {
            score: 5,
            cnf: &gpu_sat_high,
            branch_var_limit: &branch_limit,
        },
    ];
    let search_candidates = [
        GpuSolverProductionMaxSatSearchCandidate {
            score: 9,
            cnf: &gpu_unsat,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionMaxSatSearchCandidate {
            score: 7,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];
    let jobs = [
        GpuSolverProductionMaxSatScheduleJob::CandidateSet {
            candidates: &candidate_set,
        },
        GpuSolverProductionMaxSatScheduleJob::Search {
            candidates: &search_candidates,
        },
        GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
            weighted: &weighted,
            branch_var_limit: &branch_limit,
            selections: &selections,
        },
        GpuSolverProductionMaxSatScheduleJob::Unknown {
            reason: "quaternary know scheduler branch budget exhausted",
        },
        GpuSolverProductionMaxSatScheduleJob::Timeout { budget_micros: 25 },
    ];
    let mut scheduler_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut scheduler_workspace = scheduler_adapter
        .new_workspace(1, 2)
        .expect("new quaternary generalized MaxSAT scheduler workspace");
    let schedule_report = scheduler_adapter
        .solve_maxsat_schedule_with_gpu_execution_results(
            &fix.provider,
            &[&result],
            &mut scheduler_workspace,
            &jobs,
        )
        .expect("accepted quaternary GPU evidence must gate MaxSAT scheduler");

    assert_eq!(schedule_report.candidate_evidence_records, 1);
    assert_eq!(schedule_report.jobs, 5);
    assert_eq!(schedule_report.candidate_set_jobs, 1);
    assert_eq!(schedule_report.search_jobs, 1);
    assert_eq!(schedule_report.encoded_search_jobs, 1);
    assert_eq!(schedule_report.unknown_jobs, 1);
    assert_eq!(schedule_report.timeout_jobs, 1);
    assert_eq!(schedule_report.optimum_score, 7);
    assert_eq!(schedule_report.candidates_checked, 6);
    assert_eq!(schedule_report.satisfiable_candidates, 4);
    assert_eq!(schedule_report.unsat_candidates_pruned, 2);
    assert_eq!(schedule_report.gpu_cdcl_candidate_encodes, 2);
    assert_eq!(schedule_report.gpu_cdcl_candidate_solves, 6);

    let scheduler_trace = scheduler_adapter.trace();
    assert_eq!(scheduler_trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        scheduler_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        4
    );
    assert_eq!(
        scheduler_trace.accepted_know_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        scheduler_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        scheduler_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_jobs, 5);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_candidate_set_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_search_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_encoded_search_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_unknown_status_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_timeout_status_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_encodes, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_solves, 6);
    assert_eq!(scheduler_trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_optima, 3);
    assert_eq!(scheduler_trace.cpu_assignment_enumerations, 0);
    assert_eq!(scheduler_trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_quaternary_not_possible_solver_nonzero_arity_evidence_trace() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary not-possible solver evidence fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary not-possible solver evidence executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary not-possible solver evidence fixture");
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(1, 2, 3, 4), (9, 9, 9, 9)]
    );
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload SAT CNF");
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let assignment = adapter
        .solve_expect_sat_with_gpu_execution_result(&fix.provider, &result, &sat_cnf)
        .expect("accepted quaternary not-possible GPU evidence must gate solver SAT path");
    assert_ne!(*assignment.device_ptr(), 0);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        4
    );
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.accepted_possible_gpu_candidate_evidence_consumed, 0);
    assert_eq!(
        trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(trace.accepted_not_know_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_quaternary_not_possible_gates_solver_reuse_maxsat_and_portfolio_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary not-possible solver reuse fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary not-possible solver reuse executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary not-possible solver reuse fixture");
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(1, 2, 3, 4), (9, 9, 9, 9)]
    );
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

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
        .expect("new quaternary not-possible reuse workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let learned = adapter
        .solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut workspace,
            &unsat_cnf,
            &branch_limit,
            &unsat_cnf,
            &branch_limit,
        )
        .expect("accepted quaternary not-possible GPU evidence must gate learned-clause reuse");
    let maxsat = adapter
        .solve_weighted_maxsat_candidates_with_gpu_execution_result(
            &fix.provider,
            &result,
            &[GpuSolverProductionMaxSatCandidate {
                score: 13,
                cnf: &sat_cnf,
                branch_var_limit: &branch_limit,
            }],
        )
        .expect("accepted quaternary not-possible GPU evidence must gate MaxSAT");
    let portfolio = adapter
        .solve_portfolio_with_gpu_execution_result(
            &fix.provider,
            &result,
            &[
                GpuSolverProductionPortfolioJob::Sat {
                    cnf: &sat_cnf,
                    branch_var_limit: &branch_limit,
                },
                GpuSolverProductionPortfolioJob::MaxSat {
                    candidates: &[GpuSolverProductionMaxSatCandidate {
                        score: 13,
                        cnf: &sat_cnf,
                        branch_var_limit: &branch_limit,
                    }],
                },
                GpuSolverProductionPortfolioJob::Unknown {
                    reason: "quaternary not-possible branch budget exhausted",
                },
                GpuSolverProductionPortfolioJob::Timeout { budget_micros: 10 },
            ],
        )
        .expect("accepted quaternary not-possible GPU evidence must gate portfolio path");

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    assert_eq!(learned.candidate_evidence_records, 1);
    assert_eq!(learned.candidates, 2);
    assert_eq!(learned.unsat_solves, 2);
    assert_eq!(learned.gpu_learned_clause_arena_publications, 1);
    assert_eq!(learned.gpu_learned_clause_imports, 1);
    assert_eq!(learned.gpu_learned_clause_reused_solves, 1);
    assert_eq!(learned.cpu_learned_clause_transfers, 0);
    assert_eq!(maxsat.candidate_evidence_records, 1);
    assert_eq!(maxsat.optimum_score, 13);
    assert_eq!(maxsat.gpu_cdcl_candidate_solves, 1);
    assert_eq!(portfolio.jobs, 4);
    assert_eq!(portfolio.sat_jobs, 1);
    assert_eq!(portfolio.maxsat_jobs, 1);
    assert_eq!(portfolio.unknown_jobs, 1);
    assert_eq!(portfolio.timeout_jobs, 1);
    assert_eq!(portfolio.maxsat_optimum_scores, 13);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 3);
    assert_eq!(
        trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        3
    );
    assert_eq!(
        trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        12
    );
    assert_eq!(trace.accepted_know_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.accepted_possible_gpu_candidate_evidence_consumed, 0);
    assert_eq!(
        trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        3
    );
    assert_eq!(trace.accepted_not_know_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 2);
    assert_eq!(trace.gpu_learned_clause_arena_publications, 1);
    assert_eq!(trace.gpu_learned_clause_imports, 1);
    assert_eq!(trace.gpu_learned_clause_reused_solves, 1);
    assert_eq!(trace.cpu_learned_clause_transfers, 0);
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

#[test]
fn accepted_quaternary_not_possible_gates_solver_search_and_scheduler_paths() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let program = parse_program(
        r#"
        pred tuple4(u32, u32, u32, u32).
        pred fact4(u32, u32, u32, u32).
        pred accepted(u32, u32, u32, u32).
        accepted(A, B, C, D) :- tuple4(A, B, C, D), not possible fact4(A, B, C, D).
        "#,
    )
    .expect("parse quaternary not-possible solver search fixture");
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, None)
        .expect("compile quaternary not-possible solver search executable");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &executable.relation_ids {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "tuple4",
        upload_quaternary_u32(&fix.memory, &[(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]),
    );
    executor.put_relation("fact4", upload_quaternary_u32(&fix.memory, &[(2, 3, 4, 5)]));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 1,
                max_models_per_reduction: 3,
            },
        )
        .expect("execute quaternary not-possible solver search fixture");
    assert_eq!(
        download_quaternary_u32(&fix.provider, &result.final_output),
        vec![(1, 2, 3, 4), (9, 9, 9, 9)]
    );
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let sat_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let unsat = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let gpu_sat_low = GpuCnf::from_host(&sat_low, &fix.provider).expect("upload SAT low");
    let gpu_sat_high = GpuCnf::from_host(&sat_high, &fix.provider).expect("upload SAT high");
    let gpu_unsat = GpuCnf::from_host(&unsat, &fix.provider).expect("upload UNSAT candidate");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);

    let mut search_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut search_workspace = search_adapter
        .new_workspace(gpu_unsat.var_cap, gpu_unsat.clause_cap)
        .expect("new quaternary not-possible MaxSAT search workspace");
    let search_report = search_adapter
        .solve_weighted_maxsat_search_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut search_workspace,
            &[
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 7,
                    cnf: &gpu_sat_low,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                },
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 11,
                    cnf: &gpu_unsat,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                },
            ],
        )
        .expect("accepted quaternary not-possible GPU evidence must gate MaxSAT search");

    assert_eq!(search_report.candidate_evidence_records, 1);
    assert_eq!(search_report.optimum_score, 7);
    assert_eq!(search_report.candidates_checked, 2);
    assert_eq!(search_report.satisfiable_candidates, 1);
    assert_eq!(search_report.unsat_candidates_pruned, 1);
    assert_eq!(search_report.gpu_cdcl_candidate_solves, 2);

    let search_trace = search_adapter.trace();
    assert_eq!(search_trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        search_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        4
    );
    assert_eq!(
        search_trace.accepted_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        search_trace.accepted_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        search_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        search_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(search_trace.gpu_cdcl_sat_solves, 1);
    assert_eq!(search_trace.gpu_cdcl_workspace_unsat_solves, 1);
    assert_eq!(search_trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(search_trace.gpu_maxsat_unsat_candidate_prunes, 1);
    assert_eq!(search_trace.gpu_maxsat_optima, 1);
    assert_eq!(search_trace.cpu_assignment_enumerations, 0);
    assert_eq!(search_trace.cpu_maxsat_enumerations, 0);

    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![7.0, 9.0],
    );
    let both_soft_clauses = [0usize, 1usize];
    let sat_soft_clause = [0usize];
    let selections = [
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &both_soft_clauses,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionWeightedMaxSatSelection {
            soft_clause_indices: &sat_soft_clause,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];

    let mut encoded_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut encoded_workspace = encoded_adapter
        .new_workspace(1, 2)
        .expect("new quaternary not-possible encoded MaxSAT workspace");
    let encoded_report = encoded_adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut encoded_workspace,
            &weighted,
            &branch_limit,
            &selections,
        )
        .expect("accepted quaternary not-possible GPU evidence must gate encoded MaxSAT");

    assert_eq!(encoded_report.candidate_evidence_records, 1);
    assert_eq!(encoded_report.optimum_score, 7);
    assert_eq!(encoded_report.candidates_checked, 2);
    assert_eq!(encoded_report.satisfiable_candidates, 1);
    assert_eq!(encoded_report.unsat_candidates_pruned, 1);
    assert_eq!(encoded_report.gpu_cdcl_candidate_encodes, 2);
    assert_eq!(encoded_report.gpu_cdcl_candidate_solves, 2);

    let encoded_trace = encoded_adapter.trace();
    assert_eq!(encoded_trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        encoded_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        4
    );
    assert_eq!(
        encoded_trace.accepted_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        encoded_trace.accepted_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        encoded_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        encoded_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(encoded_trace.gpu_maxsat_candidate_encodes, 2);
    assert_eq!(encoded_trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(encoded_trace.gpu_maxsat_unsat_candidate_prunes, 1);
    assert_eq!(encoded_trace.gpu_maxsat_optima, 1);
    assert_eq!(encoded_trace.cpu_assignment_enumerations, 0);
    assert_eq!(encoded_trace.cpu_maxsat_enumerations, 0);

    let candidate_set = [
        GpuSolverProductionMaxSatCandidate {
            score: 3,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionMaxSatCandidate {
            score: 5,
            cnf: &gpu_sat_high,
            branch_var_limit: &branch_limit,
        },
    ];
    let search_candidates = [
        GpuSolverProductionMaxSatSearchCandidate {
            score: 9,
            cnf: &gpu_unsat,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
        },
        GpuSolverProductionMaxSatSearchCandidate {
            score: 7,
            cnf: &gpu_sat_low,
            branch_var_limit: &branch_limit,
            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
        },
    ];
    let jobs = [
        GpuSolverProductionMaxSatScheduleJob::CandidateSet {
            candidates: &candidate_set,
        },
        GpuSolverProductionMaxSatScheduleJob::Search {
            candidates: &search_candidates,
        },
        GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
            weighted: &weighted,
            branch_var_limit: &branch_limit,
            selections: &selections,
        },
        GpuSolverProductionMaxSatScheduleJob::Unknown {
            reason: "quaternary not-possible scheduler branch budget exhausted",
        },
        GpuSolverProductionMaxSatScheduleJob::Timeout { budget_micros: 25 },
    ];
    let mut scheduler_adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut scheduler_workspace = scheduler_adapter
        .new_workspace(1, 2)
        .expect("new quaternary not-possible generalized MaxSAT scheduler workspace");
    let schedule_report = scheduler_adapter
        .solve_maxsat_schedule_with_gpu_execution_results(
            &fix.provider,
            &[&result],
            &mut scheduler_workspace,
            &jobs,
        )
        .expect("accepted quaternary not-possible GPU evidence must gate MaxSAT scheduler");

    assert_eq!(schedule_report.candidate_evidence_records, 1);
    assert_eq!(schedule_report.jobs, 5);
    assert_eq!(schedule_report.candidate_set_jobs, 1);
    assert_eq!(schedule_report.search_jobs, 1);
    assert_eq!(schedule_report.encoded_search_jobs, 1);
    assert_eq!(schedule_report.unknown_jobs, 1);
    assert_eq!(schedule_report.timeout_jobs, 1);
    assert_eq!(schedule_report.optimum_score, 7);
    assert_eq!(schedule_report.candidates_checked, 6);
    assert_eq!(schedule_report.satisfiable_candidates, 4);
    assert_eq!(schedule_report.unsat_candidates_pruned, 2);
    assert_eq!(schedule_report.gpu_cdcl_candidate_encodes, 2);
    assert_eq!(schedule_report.gpu_cdcl_candidate_solves, 6);

    let scheduler_trace = scheduler_adapter.trace();
    assert_eq!(scheduler_trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(
        scheduler_trace.accepted_nonzero_arity_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_gpu_candidate_tuple_key_column_reads_consumed,
        4
    );
    assert_eq!(
        scheduler_trace.accepted_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        scheduler_trace.accepted_possible_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(
        scheduler_trace.accepted_not_possible_gpu_candidate_evidence_consumed,
        1
    );
    assert_eq!(
        scheduler_trace.accepted_not_know_gpu_candidate_evidence_consumed,
        0
    );
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_jobs, 5);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_candidate_set_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_search_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_encoded_search_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_unknown_status_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_scheduler_timeout_status_jobs, 1);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_encodes, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_candidate_solves, 6);
    assert_eq!(scheduler_trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(scheduler_trace.gpu_maxsat_optima, 3);
    assert_eq!(scheduler_trace.cpu_assignment_enumerations, 0);
    assert_eq!(scheduler_trace.cpu_maxsat_enumerations, 0);
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
    assert_eq!(maxsat.candidate_evidence_records, 1);
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

#[test]
fn accepted_gpu_execution_results_gate_multi_candidate_portfolio_path() {
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
    let maxsat_candidate =
        GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload MaxSAT candidate CNF");
    let portfolio_sat =
        GpuCnf::from_host(&sat_instance, &fix.provider).expect("upload portfolio SAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let maxsat_candidates = [GpuSolverProductionMaxSatCandidate {
        score: 5,
        cnf: &maxsat_candidate,
        branch_var_limit: &branch_limit,
    }];
    let portfolio_jobs = [
        GpuSolverProductionPortfolioJob::Sat {
            cnf: &portfolio_sat,
            branch_var_limit: &branch_limit,
        },
        GpuSolverProductionPortfolioJob::MaxSat {
            candidates: &maxsat_candidates,
        },
        GpuSolverProductionPortfolioJob::Unknown {
            reason: "bounded branch budget exhausted before a determined status",
        },
        GpuSolverProductionPortfolioJob::Timeout { budget_micros: 10 },
    ];
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());

    let report = adapter
        .solve_multi_candidate_portfolio_with_gpu_execution_results(
            &fix.provider,
            &[&result_a, &result_b],
            &portfolio_jobs,
        )
        .expect("accepted GPU runtime evidence must gate two-record portfolio path");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.jobs, 8);
    assert_eq!(report.sat_jobs, 2);
    assert_eq!(report.maxsat_jobs, 2);
    assert_eq!(report.unknown_jobs, 2);
    assert_eq!(report.timeout_jobs, 2);
    assert_eq!(report.maxsat_optimum_scores, 10);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(trace.gpu_maxsat_optima, 2);
    assert_eq!(trace.gpu_portfolio_jobs, 8);
    assert_eq!(trace.gpu_portfolio_sat_jobs, 2);
    assert_eq!(trace.gpu_portfolio_maxsat_jobs, 2);
    assert_eq!(trace.gpu_portfolio_unknown_status_jobs, 2);
    assert_eq!(trace.gpu_portfolio_timeout_status_jobs, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_results_gate_multi_candidate_maxsat_path() {
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

    let candidate_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let candidate_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let gpu_candidate_low =
        GpuCnf::from_host(&candidate_low, &fix.provider).expect("upload low-score MaxSAT CNF");
    let gpu_candidate_high =
        GpuCnf::from_host(&candidate_high, &fix.provider).expect("upload high-score MaxSAT CNF");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());

    let report = adapter
        .solve_multi_candidate_weighted_maxsat_with_gpu_execution_results(
            &fix.provider,
            &[&result_a, &result_b],
            &[
                GpuSolverProductionMaxSatCandidate {
                    score: 3,
                    cnf: &gpu_candidate_low,
                    branch_var_limit: &branch_limit,
                },
                GpuSolverProductionMaxSatCandidate {
                    score: 7,
                    cnf: &gpu_candidate_high,
                    branch_var_limit: &branch_limit,
                },
            ],
        )
        .expect("accepted GPU runtime evidence must gate multi-candidate MaxSAT through GPU CDCL");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.optimum_score, 7);
    assert_eq!(report.candidates_checked, 4);
    assert_eq!(report.gpu_cdcl_candidate_solves, 4);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(trace.gpu_maxsat_optima, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_result_prunes_unsat_maxsat_search_candidates() {
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

    let unsat_candidate = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let sat_candidate = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let gpu_unsat_candidate =
        GpuCnf::from_host(&unsat_candidate, &fix.provider).expect("upload UNSAT candidate");
    let gpu_sat_candidate =
        GpuCnf::from_host(&sat_candidate, &fix.provider).expect("upload SAT candidate");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(gpu_unsat_candidate.var_cap, gpu_unsat_candidate.clause_cap)
        .expect("new MaxSAT search workspace");

    let report = adapter
        .solve_weighted_maxsat_search_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut workspace,
            &[
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 9,
                    cnf: &gpu_unsat_candidate,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                },
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 7,
                    cnf: &gpu_sat_candidate,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                },
            ],
        )
        .expect("accepted GPU evidence must gate MaxSAT search pruning through GPU CDCL");

    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.optimum_score, 7);
    assert_eq!(report.candidates_checked, 2);
    assert_eq!(report.satisfiable_candidates, 1);
    assert_eq!(report.unsat_candidates_pruned, 1);
    assert_eq!(report.gpu_cdcl_candidate_solves, 2);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 1);
    assert_eq!(trace.gpu_maxsat_optima, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_result_rejects_all_unsat_maxsat_search_before_solver_work() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
        &[1, 2],
        &[1],
    );

    let unsat_candidate = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let gpu_unsat_candidate =
        GpuCnf::from_host(&unsat_candidate, &fix.provider).expect("upload UNSAT candidate");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(gpu_unsat_candidate.var_cap, gpu_unsat_candidate.clause_cap)
        .expect("new MaxSAT search workspace");
    let assign_ptr_before = workspace.assign_device_ptr();

    let err = adapter
        .solve_weighted_maxsat_search_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut workspace,
            &[GpuSolverProductionMaxSatSearchCandidate {
                score: 9,
                cnf: &gpu_unsat_candidate,
                branch_var_limit: &branch_limit,
                status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
            }],
        )
        .expect_err("all-UNSAT MaxSAT search candidates must fail before solver work");
    assert!(format!("{err}")
        .contains("bounded MaxSAT search requires at least one satisfiable GPU candidate"));

    assert_eq!(workspace.assign_device_ptr(), assign_ptr_before);
    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.gpu_cdcl_sat_solves, 0);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 0);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 0);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 0);
    assert_eq!(trace.gpu_maxsat_optima, 0);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_results_gate_multi_candidate_maxsat_search_pruning() {
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

    let unsat_candidate = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let sat_candidate = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let gpu_unsat_candidate =
        GpuCnf::from_host(&unsat_candidate, &fix.provider).expect("upload UNSAT candidate");
    let gpu_sat_candidate =
        GpuCnf::from_host(&sat_candidate, &fix.provider).expect("upload SAT candidate");
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(gpu_unsat_candidate.var_cap, gpu_unsat_candidate.clause_cap)
        .expect("new multi-result MaxSAT search workspace");

    let report = adapter
        .solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results(
            &fix.provider,
            &[&result_a, &result_b],
            &mut workspace,
            &[
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 9,
                    cnf: &gpu_unsat_candidate,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                },
                GpuSolverProductionMaxSatSearchCandidate {
                    score: 7,
                    cnf: &gpu_sat_candidate,
                    branch_var_limit: &branch_limit,
                    status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                },
            ],
        )
        .expect("accepted GPU evidence must gate multi-result MaxSAT search pruning");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.optimum_score, 7);
    assert_eq!(report.candidates_checked, 4);
    assert_eq!(report.satisfiable_candidates, 2);
    assert_eq!(report.unsat_candidates_pruned, 2);
    assert_eq!(report.gpu_cdcl_candidate_solves, 4);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(trace.gpu_maxsat_optima, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_result_encodes_weighted_maxsat_search_candidates() {
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

    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![7.0, 9.0],
    );
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(1, 2)
        .expect("new encoded MaxSAT search workspace");
    let unsat_selection = [0usize, 1usize];
    let sat_selection = [0usize];

    let report = adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut workspace,
            &weighted,
            &branch_limit,
            &[
                GpuSolverProductionWeightedMaxSatSelection {
                    soft_clause_indices: &unsat_selection,
                    status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                },
                GpuSolverProductionWeightedMaxSatSelection {
                    soft_clause_indices: &sat_selection,
                    status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                },
            ],
        )
        .expect("accepted GPU evidence must gate weighted MaxSAT encoding through GPU CDCL");

    assert_eq!(report.candidate_evidence_records, 1);
    assert_eq!(report.optimum_score, 7);
    assert_eq!(report.candidates_checked, 2);
    assert_eq!(report.satisfiable_candidates, 1);
    assert_eq!(report.unsat_candidates_pruned, 1);
    assert_eq!(report.gpu_cdcl_candidate_encodes, 2);
    assert_eq!(report.gpu_cdcl_candidate_solves, 2);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 1);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 2);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 2);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 1);
    assert_eq!(trace.gpu_maxsat_optima, 1);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_result_rejects_all_unsat_encoded_maxsat_before_encoding_work() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let result = execute_unary_edge_epistemic_fixture(
        &fix,
        r#"
        pred node(u32).
        pred edge(u32).
        pred accepted(u32).
        accepted(X) :- node(X), know edge(X).
        "#,
        &[1, 2],
        &[1],
    );

    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![7.0, 9.0],
    );
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(1, 2)
        .expect("new encoded MaxSAT search workspace");
    let unsat_selection = [0usize, 1usize];

    let err = adapter
        .solve_weighted_maxsat_encoded_search_with_gpu_execution_result(
            &fix.provider,
            &result,
            &mut workspace,
            &weighted,
            &branch_limit,
            &[GpuSolverProductionWeightedMaxSatSelection {
                soft_clause_indices: &unsat_selection,
                status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
            }],
        )
        .expect_err("all-UNSAT encoded MaxSAT selections must fail before encoding work");
    assert!(format!("{err}")
        .contains("bounded MaxSAT search requires at least one satisfiable GPU candidate"));

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 0);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 0);
    assert_eq!(trace.gpu_cdcl_sat_solves, 0);
    assert_eq!(trace.gpu_cdcl_workspace_unsat_solves, 0);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 0);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 0);
    assert_eq!(trace.gpu_maxsat_optima, 0);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_results_gate_multi_candidate_weighted_maxsat_encoded_search() {
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

    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![7.0, 9.0],
    );
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(1, 2)
        .expect("new multi-result encoded MaxSAT search workspace");
    let unsat_selection = [0usize, 1usize];
    let sat_selection = [0usize];

    let report = adapter
        .solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results(
            &fix.provider,
            &[&result_a, &result_b],
            &mut workspace,
            &weighted,
            &branch_limit,
            &[
                GpuSolverProductionWeightedMaxSatSelection {
                    soft_clause_indices: &unsat_selection,
                    status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                },
                GpuSolverProductionWeightedMaxSatSelection {
                    soft_clause_indices: &sat_selection,
                    status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                },
            ],
        )
        .expect("accepted GPU evidence must gate multi-result weighted MaxSAT encoding");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.optimum_score, 7);
    assert_eq!(report.candidates_checked, 4);
    assert_eq!(report.satisfiable_candidates, 2);
    assert_eq!(report.unsat_candidates_pruned, 2);
    assert_eq!(report.gpu_cdcl_candidate_encodes, 4);
    assert_eq!(report.gpu_cdcl_candidate_solves, 4);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 4);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 4);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 2);
    assert_eq!(trace.gpu_maxsat_optima, 2);
    assert_eq!(trace.cpu_assignment_enumerations, 0);
    assert_eq!(trace.cpu_maxsat_enumerations, 0);
}

#[test]
fn accepted_gpu_execution_results_gate_generalized_maxsat_scheduler() {
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

    let sat_low = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_high = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
    let unsat = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let gpu_sat_low = GpuCnf::from_host(&sat_low, &fix.provider).expect("upload SAT low");
    let gpu_sat_high = GpuCnf::from_host(&sat_high, &fix.provider).expect("upload SAT high");
    let gpu_unsat = GpuCnf::from_host(&unsat, &fix.provider).expect("upload UNSAT candidate");
    let weighted = SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![7.0, 9.0],
    );
    let branch_limit = upload_u32_scalar(&fix.provider, 1);
    let mut adapter =
        GpuSolverProductionAdapter::new(Arc::clone(&fix.provider), GpuCdclConfig::default());
    let mut workspace = adapter
        .new_workspace(1, 2)
        .expect("new generalized MaxSAT scheduler workspace");
    let both_soft_clauses = [0usize, 1usize];
    let sat_soft_clause = [0usize];

    let report = adapter
        .solve_maxsat_schedule_with_gpu_execution_results(
            &fix.provider,
            &[&result_a, &result_b],
            &mut workspace,
            &[
                GpuSolverProductionMaxSatScheduleJob::CandidateSet {
                    candidates: &[
                        GpuSolverProductionMaxSatCandidate {
                            score: 3,
                            cnf: &gpu_sat_low,
                            branch_var_limit: &branch_limit,
                        },
                        GpuSolverProductionMaxSatCandidate {
                            score: 5,
                            cnf: &gpu_sat_high,
                            branch_var_limit: &branch_limit,
                        },
                    ],
                },
                GpuSolverProductionMaxSatScheduleJob::Search {
                    candidates: &[
                        GpuSolverProductionMaxSatSearchCandidate {
                            score: 9,
                            cnf: &gpu_unsat,
                            branch_var_limit: &branch_limit,
                            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                        },
                        GpuSolverProductionMaxSatSearchCandidate {
                            score: 7,
                            cnf: &gpu_sat_low,
                            branch_var_limit: &branch_limit,
                            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                        },
                    ],
                },
                GpuSolverProductionMaxSatScheduleJob::EncodedSearch {
                    weighted: &weighted,
                    branch_var_limit: &branch_limit,
                    selections: &[
                        GpuSolverProductionWeightedMaxSatSelection {
                            soft_clause_indices: &both_soft_clauses,
                            status: GpuSolverProductionMaxSatSearchStatus::Unsatisfiable,
                        },
                        GpuSolverProductionWeightedMaxSatSelection {
                            soft_clause_indices: &sat_soft_clause,
                            status: GpuSolverProductionMaxSatSearchStatus::Satisfiable,
                        },
                    ],
                },
                GpuSolverProductionMaxSatScheduleJob::Unknown {
                    reason: "bounded scheduler branch budget exhausted",
                },
                GpuSolverProductionMaxSatScheduleJob::Timeout { budget_micros: 25 },
            ],
        )
        .expect("accepted GPU evidence must gate generalized MaxSAT scheduler");

    assert_eq!(report.candidate_evidence_records, 2);
    assert_eq!(report.jobs, 10);
    assert_eq!(report.candidate_set_jobs, 2);
    assert_eq!(report.search_jobs, 2);
    assert_eq!(report.encoded_search_jobs, 2);
    assert_eq!(report.unknown_jobs, 2);
    assert_eq!(report.timeout_jobs, 2);
    assert_eq!(report.optimum_score, 7);
    assert_eq!(report.candidates_checked, 12);
    assert_eq!(report.satisfiable_candidates, 8);
    assert_eq!(report.unsat_candidates_pruned, 4);
    assert_eq!(report.gpu_cdcl_candidate_encodes, 4);
    assert_eq!(report.gpu_cdcl_candidate_solves, 12);

    let trace = adapter.trace();
    assert_eq!(trace.accepted_gpu_candidate_evidence_consumed, 2);
    assert_eq!(trace.gpu_maxsat_scheduler_jobs, 10);
    assert_eq!(trace.gpu_maxsat_scheduler_candidate_set_jobs, 2);
    assert_eq!(trace.gpu_maxsat_scheduler_search_jobs, 2);
    assert_eq!(trace.gpu_maxsat_scheduler_encoded_search_jobs, 2);
    assert_eq!(trace.gpu_maxsat_scheduler_unknown_status_jobs, 2);
    assert_eq!(trace.gpu_maxsat_scheduler_timeout_status_jobs, 2);
    assert_eq!(trace.gpu_maxsat_candidate_encodes, 4);
    assert_eq!(trace.gpu_maxsat_candidate_solves, 12);
    assert_eq!(trace.gpu_maxsat_unsat_candidate_prunes, 4);
    assert_eq!(trace.gpu_maxsat_optima, 6);
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
