// crates/xlog-integration/tests/test_wcoj_4cycle_dispatch_stream_reuse.rs
//! v0.6.5 slice 2 — confirms the cached `Executor::wcoj_dispatch_stream`
//! is reused across triangle and 4-cycle dispatch within the same
//! Executor.
//!
//! The slice 2 stream rename made the launch stream shape-agnostic.
//! Acquiring per-shape would silently drain the StreamPool (cap 16,
//! grow-only) on long-lived runtimes with both shapes active. This
//! test compiles a program with both a triangle rule AND a 4-cycle
//! rule, runs it through one Executor under both force gates, and
//! confirms both dispatches succeed without exhausting the pool.

use std::collections::BTreeMap;
use std::sync::Arc;

use xlog_core::{MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)]
struct RuntimeBackedFixture {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
}

fn make_runtime_backed_fixture() -> Option<RuntimeBackedFixture> {
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
    Some(RuntimeBackedFixture {
        device,
        runtime,
        memory,
        provider,
        pool,
    })
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let dev = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows.iter().flat_map(|&(a, _)| a.to_le_bytes()).collect();
        let c1: Vec<u8> = rows.iter().flat_map(|&(_, b)| b.to_le_bytes()).collect();
        dev.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
        dev.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
    }
    dev.htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod n");
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

const COMBINED_SOURCE: &str = "
tri(X, Y, Z) :- t1(X, Y), t2(Y, Z), t3(X, Z).
cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).
";

fn fixture() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    let edges: Vec<(u32, u32)> = vec![(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)];
    m.insert("t1", edges.clone());
    m.insert("t2", edges.clone());
    m.insert("t3", edges.clone());
    m.insert("e1", edges.clone());
    m.insert("e2", edges.clone());
    m.insert("e3", edges.clone());
    m.insert("e4", edges);
    m
}

#[test]
fn triangle_and_4cycle_share_dispatch_stream() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = fixture();
    let config = RuntimeConfig::default()
        .with_wcoj_triangle_dispatch(Some(true))
        .with_wcoj_4cycle_dispatch(Some(true));
    let mut compiler = Compiler::new();
    let plan = compiler.compile(COMBINED_SOURCE).expect("compile");
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in &inputs {
        let buf = upload_binary_u32(&fix.memory, rows);
        executor.put_relation(name, buf);
    }
    executor.execute_plan(&plan).expect("execute_plan");

    // Both dispatches fire — the cached wcoj_dispatch_stream is
    // shared, not duplicated. If acquisition were per-shape, the
    // second dispatch would either drain the pool or return Ok(None)
    // silently; either way one of these counters would stay 0.
    assert_eq!(
        executor.wcoj_triangle_dispatch_count(),
        1,
        "triangle dispatch must fire under shared stream"
    );
    assert_eq!(
        executor.wcoj_4cycle_dispatch_count(),
        1,
        "4-cycle dispatch must fire under shared stream"
    );
}

#[test]
fn repeated_dispatch_does_not_drain_stream_pool() {
    // Run the same compiled plan many times in one Executor;
    // the shared launch stream is acquired once and reused. A
    // per-acquisition pattern would silently fall back after 16
    // iterations (default StreamPool cap), making the dispatch
    // counter stop incrementing.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = fixture();
    let config = RuntimeConfig::default()
        .with_wcoj_triangle_dispatch(Some(true))
        .with_wcoj_4cycle_dispatch(Some(true));
    let mut compiler = Compiler::new();
    let plan = compiler.compile(COMBINED_SOURCE).expect("compile");
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in &inputs {
        let buf = upload_binary_u32(&fix.memory, rows);
        executor.put_relation(name, buf);
    }

    // 20 iterations > StreamPool default cap of 16. If the launch
    // stream were acquired per-iteration, dispatches past iter 16
    // would silently fail.
    for _ in 0..20 {
        executor.execute_plan(&plan).expect("execute_plan");
    }
    // Each iteration installs both heads (overwriting via union),
    // so each iteration's triangle attempt and 4-cycle attempt
    // increment their counters once.
    assert_eq!(
        executor.wcoj_triangle_dispatch_count(),
        20,
        "20 iterations must produce 20 triangle dispatches; counter staying < 20 would mean the \
         stream pool was drained"
    );
    assert_eq!(
        executor.wcoj_4cycle_dispatch_count(),
        20,
        "20 iterations must produce 20 4-cycle dispatches"
    );
}
