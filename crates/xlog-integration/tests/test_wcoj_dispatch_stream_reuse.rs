// crates/xlog-integration/tests/test_wcoj_dispatch_stream_reuse.rs
//! Regression: the executor's WCOJ triangle dispatch hook must
//! reuse a single launch stream across all invocations on a
//! given executor instance.
//!
//! Without the cached stream (an earlier shape of the dispatch
//! hook), each invocation called
//! `runtime.stream_pool().acquire()`. The
//! [`xlog_cuda::device_runtime::StreamPool`] is grow-only with
//! a hard cap (default 16, see `DEFAULT_MAX_STREAMS`); after
//! 16 successful acquisitions, every subsequent acquire returns
//! a `Capacity` error which the dispatch hook silently swallowed.
//! Long-lived runtimes (benchmarks, soak tests, programs with
//! >16 matching triangle rules) would route the rest of their
//! work through the binary-join fallback while the dispatch
//! counter stopped incrementing — invalidating the gate-on
//! certification path AND any benchmark numbers built on it.
//!
//! This test exercises >16 matching triangle dispatches on a
//! single `Executor` and asserts the dispatch counter keeps
//! incrementing one-per-execute_plan call, not stalling at 16.

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
struct Fix {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
}

fn make_fix() -> Option<Fix> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    // Use the DEFAULT pool (cap = 16). The whole point is to
    // verify the executor doesn't drain it with one stream
    // per dispatch.
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let limit_bytes: usize = 256 * 1024 * 1024;
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, limit_bytes));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(limit_bytes as u64),
        Arc::clone(&runtime),
    ));
    let provider = Arc::new(
        CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?,
    );
    Some(Fix {
        device,
        runtime,
        memory,
        provider,
        pool,
    })
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bpc = (n as usize) * 4;
    let mut col0 = memory.alloc::<u8>(bpc).expect("alloc c0");
    let mut col1 = memory.alloc::<u8>(bpc).expect("alloc c1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc nr");
    let device = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let c1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
        device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod nr");
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

const SOURCE: &str = "tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).";

#[test]
fn dispatch_counter_grows_past_stream_pool_cap() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Default pool cap is 16. We run 20 execute_plan calls on a
    // SINGLE executor, each dispatching the same triangle rule
    // through the WCOJ hook. The cached-stream contract requires
    // counter == 20 at the end. Without the cache, counter would
    // saturate at 16 because acquire() returns Capacity for
    // every call after the 16th and the hook silently falls back.
    const ITERATIONS: u64 = 20;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(SOURCE).expect("compile");
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true));
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }

    // Tiny fixture so each iteration is fast. Same shape that
    // the wiring tests already cert.
    let e1: Vec<(u32, u32)> = vec![(1, 2), (1, 3), (2, 3)];
    let e2: Vec<(u32, u32)> = vec![(2, 3), (3, 4)];
    let e3: Vec<(u32, u32)> = vec![(1, 3), (1, 4), (2, 4)];

    for i in 0..ITERATIONS {
        // Each iteration uploads a fresh copy of the inputs
        // (put_relation overwrites the prior buffer) so the
        // executor sees a non-degenerate dispatch each pass.
        executor.put_relation("e1", upload_binary_u32(&fix.memory, &e1));
        executor.put_relation("e2", upload_binary_u32(&fix.memory, &e2));
        executor.put_relation("e3", upload_binary_u32(&fix.memory, &e3));
        executor.execute_plan(&plan).expect("execute_plan");
        let counter = executor.wcoj_triangle_dispatch_count();
        assert_eq!(
            counter,
            i + 1,
            "after {} execute_plan calls dispatch counter must be {} (stream-pool cap is 16; \
             saturation here would mean the executor is acquiring per-invocation rather than \
             reusing the cached stream)",
            i + 1,
            i + 1,
        );
    }

    // Final lock — past the pool cap, counter still grows 1:1.
    assert!(
        executor.wcoj_triangle_dispatch_count() > 16,
        "regression: counter saturated at {} after {} iterations (cap is 16)",
        executor.wcoj_triangle_dispatch_count(),
        ITERATIONS
    );

    // Sanity: the pool itself acquired exactly one non-default
    // stream. If the executor were acquiring per-invocation,
    // this would be either 16 (capped) or some other > 1 value.
    assert_eq!(
        fix.pool.non_default_len(),
        1,
        "executor must acquire exactly one non-default stream across {ITERATIONS} dispatches; \
         got {}",
        fix.pool.non_default_len()
    );
}

#[test]
fn _silence_unused_imports() {
    let _: BTreeMap<&str, ()> = BTreeMap::new();
}
