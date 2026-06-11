//! D1 aggregate-fused WCOJ — end-to-end executor wiring.
//!
//! A count-aggregate head over a triangle body must compile (promoter
//! descends the aggregate wrapper), dispatch the fused group-by-root count
//! kernel (counter == 1), and produce the same rows as the
//! materialize+groupby path (kill switch forces the unfused path; rows must
//! be identical). All phases run inside ONE test because the kill switch is
//! a process-global env var.

use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::{CudaBuffer, CudaColumn};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct Fixture {
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
}

fn make_fixture() -> Option<Fixture> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        runtime,
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(Fixture { memory, provider })
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let mut col0 = memory
        .alloc::<u8>(rows.len() * std::mem::size_of::<u32>())
        .expect("alloc col0");
    let mut col1 = memory
        .alloc::<u8>(rows.len() * std::mem::size_of::<u32>())
        .expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let col0_bytes: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    let device = memory.device().inner();
    device
        .htod_sync_copy_into(&col0_bytes, &mut col0)
        .expect("htod col0");
    device
        .htod_sync_copy_into(&col1_bytes, &mut col1)
        .expect("htod col1");
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![
        ("col0".to_string(), ScalarType::U32),
        ("col1".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

fn buffer_rows(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> usize {
    if let Some(n) = buffer.cached_row_count() {
        return n as usize;
    }
    let mut host = [0u32; 1];
    memory
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host)
        .expect("dtoh num_rows");
    host[0] as usize
}

fn download_column_bytes(
    memory: &Arc<GpuMemoryManager>,
    buffer: &CudaBuffer,
    col: usize,
    elem_size: usize,
) -> Vec<u8> {
    let n = buffer_rows(memory, buffer);
    let mut bytes = vec![0u8; n * elem_size];
    if n == 0 {
        return bytes;
    }
    let CudaColumn::Owned(c) = buffer.column(col).expect("column") else {
        panic!("column must be owned");
    };
    unsafe {
        let res = sys::cuMemcpyDtoH_v2(bytes.as_mut_ptr() as *mut _, *c.device_ptr(), bytes.len());
        assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS, "dtoh column copy");
    }
    bytes
}

fn download_group_counts(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u32, u64)> {
    assert_eq!(buffer.arity(), 2, "expected (X, count) output");
    let keys: Vec<u32> = download_column_bytes(memory, buffer, 0, 4)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let counts: Vec<u64> = download_column_bytes(memory, buffer, 1, 8)
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut out: Vec<(u32, u64)> = keys.into_iter().zip(counts).collect();
    out.sort();
    out
}

const SOURCE: &str = "deg(X, count(Z)) :- e1(X, Y), e2(Y, Z), e3(X, Z).";

fn run_program(fix: &Fixture, inputs: &BTreeMap<&str, Vec<(u32, u32)>>) -> (Vec<(u32, u64)>, u64) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(SOURCE).expect("compile aggregate rule");
    let mut executor = Executor::new(Arc::clone(&fix.provider));
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        executor.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    executor.execute_plan(&plan).expect("execute plan");
    let deg = executor.store().get("deg").expect("deg relation");
    let rows = download_group_counts(&fix.memory, deg);
    (rows, executor.wcoj_groupby_fusion_dispatch_count())
}

#[test]
fn groupby_fusion_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // K4 on {1..4} plus a disjoint triangle {5,6,7}; per-X completion
    // counts: X=1 -> 3, X=2 -> 1, X=5 -> 1.
    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    inputs.insert(
        "e1",
        vec![
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (5, 6),
            (5, 7),
            (6, 7),
        ],
    );
    inputs.insert("e2", vec![(2, 3), (2, 4), (3, 4), (6, 7)]);
    inputs.insert("e3", vec![(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)]);
    let expected = vec![(1u32, 3u64), (2, 1), (5, 1)];

    // Phase 1: fused path fires and is correct.
    let (fused_rows, fused_count) = run_program(&fix, &inputs);
    assert_eq!(fused_rows, expected, "fused path row set");
    assert_eq!(
        fused_count, 1,
        "fused group-by-root count dispatch must fire exactly once"
    );

    // Phase 2: kill switch forces the materialize+groupby path; rows must
    // be identical and the counter must stay 0.
    // SAFETY: single-threaded phase of this test; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (unfused_rows, unfused_count) = run_program(&fix, &inputs);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    assert_eq!(unfused_rows, expected, "kill-switch path row set");
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
}

#[test]
fn groupby_fusion_declines_non_triangle_shape() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // 2-atom chain body: not a triangle; fusion must decline silently and
    // the standard path must produce the correct counts.
    let source = "deg(X, count(Z)) :- e1(X, Y), e2(Y, Z).";
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile chain aggregate");
    let mut executor = Executor::new(Arc::clone(&fix.provider));
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "e1",
        upload_binary_u32(&fix.memory, &[(1, 10), (1, 11), (2, 10)]),
    );
    executor.put_relation(
        "e2",
        upload_binary_u32(&fix.memory, &[(10, 100), (10, 101), (11, 100)]),
    );
    executor.execute_plan(&plan).expect("execute plan");
    let deg = executor.store().get("deg").expect("deg relation");
    let rows = download_group_counts(&fix.memory, deg);
    assert_eq!(rows, vec![(1u32, 3u64), (2, 2)], "chain aggregate rows");
    assert_eq!(
        executor.wcoj_groupby_fusion_dispatch_count(),
        0,
        "non-triangle shape must not consume the fusion counter"
    );
}
