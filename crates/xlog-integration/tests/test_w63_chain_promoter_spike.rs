//! Goal-039 G_W63_CHAIN spike cert.
//!
//! The spike reuses `RirNode::MultiWayJoin` as the chain-shape wrapper
//! instead of adding the production `ChainJoin` enum variant. This cert
//! proves the end-to-end path still has the required fallback identity:
//! default-on chain dispatch and env-disabled fallback produce the same
//! rows, while the dispatch counter distinguishes the paths.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use cudarc::driver::sys;
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
    let device = memory.device().inner();
    if n > 0 {
        let col0_bytes: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        device
            .htod_sync_copy_into(&col0_bytes, &mut col0)
            .expect("htod col0");
        device
            .htod_sync_copy_into(&col1_bytes, &mut col1)
            .expect("htod col1");
    }
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

fn download_pairs(buf: &CudaBuffer) -> Vec<(u32, u32)> {
    let n = match buf.cached_row_count() {
        Some(c) => c as usize,
        None => {
            let mut count_host = [0u32; 1];
            unsafe {
                sys::cuMemcpyDtoH_v2(
                    count_host.as_mut_ptr() as *mut _,
                    *buf.num_rows_device().device_ptr(),
                    std::mem::size_of::<u32>(),
                );
            }
            count_host[0] as usize
        }
    };
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(buf.arity(), 2);
    let mut col0_bytes = vec![0u8; n * 4];
    let mut col1_bytes = vec![0u8; n * 4];
    unsafe {
        sys::cuMemcpyDtoH_v2(
            col0_bytes.as_mut_ptr() as *mut _,
            *buf.column(0).unwrap().device_ptr(),
            col0_bytes.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col1_bytes.as_mut_ptr() as *mut _,
            *buf.column(1).unwrap().device_ptr(),
            col1_bytes.len(),
        );
    }
    let mut out: Vec<(u32, u32)> = (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(col0_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col1_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            )
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

const CHAIN_SOURCE: &str = r#"
    pred a(u32, u32).
    pred b(u32, u32).
    pred out(u32, u32).
    out(X, Y) :- a(X, Z), b(Z, Y).
"#;

fn chain_fixture() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m = BTreeMap::new();
    m.insert("a", (0..128u32).map(|i| (10_000 + i, i)).collect());
    m.insert("b", (0..128u32).map(|i| (i, 20_000 + i)).collect());
    m
}

fn chain_fixture_n(n: u32) -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m = BTreeMap::new();
    m.insert("a", (0..n).map(|i| (10_000_000 + i, i)).collect());
    m.insert("b", (0..n).map(|i| (i, 20_000_000 + i)).collect());
    m
}

fn run_chain(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> Executor {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(CHAIN_SOURCE).expect("compile chain");
    let mut executor = Executor::new_with_config(provider, RuntimeConfig::default());
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        executor.put_relation(name, upload_binary_u32(memory, rows));
    }
    executor.execute_plan(&plan).expect("execute chain");
    executor
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[test]
fn chain_dispatch_default_on_matches_env_disabled_fallback() {
    let _guard = env_lock().lock().expect("W63 env lock poisoned");
    let old = std::env::var("XLOG_WCOJ_W63_CHAIN_ENABLE").ok();
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = chain_fixture();

    unsafe {
        std::env::set_var("XLOG_WCOJ_W63_CHAIN_ENABLE", "0");
    }
    let fallback = run_chain(Arc::clone(&fix.provider), &fix.memory, &inputs);
    let fallback_rows: BTreeSet<(u32, u32)> = download_pairs(
        fallback
            .store()
            .get("out")
            .expect("fallback out relation must exist"),
    )
    .into_iter()
    .collect();
    assert_eq!(fallback.w63_chain_dispatch_count(), 0);

    unsafe {
        std::env::remove_var("XLOG_WCOJ_W63_CHAIN_ENABLE");
    }
    let dispatched = run_chain(Arc::clone(&fix.provider), &fix.memory, &inputs);
    let dispatched_rows: BTreeSet<(u32, u32)> = download_pairs(
        dispatched
            .store()
            .get("out")
            .expect("dispatched out relation must exist"),
    )
    .into_iter()
    .collect();

    unsafe {
        match old {
            Some(v) => std::env::set_var("XLOG_WCOJ_W63_CHAIN_ENABLE", v),
            None => std::env::remove_var("XLOG_WCOJ_W63_CHAIN_ENABLE"),
        }
    }

    assert_eq!(dispatched.w63_chain_dispatch_count(), 1);
    assert_eq!(dispatched_rows.len(), 128);
    assert_eq!(dispatched_rows, fallback_rows);
}

fn timed_chain_runs(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    iterations: u32,
) -> (Duration, u64) {
    let start = Instant::now();
    let mut dispatches = 0;
    for _ in 0..iterations {
        let executor = run_chain(Arc::clone(&provider), memory, inputs);
        dispatches += executor.w63_chain_dispatch_count();
    }
    (start.elapsed(), dispatches)
}

#[test]
#[ignore = "performance smoke; run manually for W63 spike timing evidence"]
fn chain_dispatch_timing_smoke_sorted_threshold_cell() {
    let _guard = env_lock().lock().expect("W63 env lock poisoned");
    let old = std::env::var("XLOG_WCOJ_W63_CHAIN_ENABLE").ok();
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = chain_fixture_n(2_000);
    let iterations = 20;

    unsafe {
        std::env::set_var("XLOG_WCOJ_W63_CHAIN_ENABLE", "0");
    }
    let (fallback_elapsed, fallback_dispatches) =
        timed_chain_runs(Arc::clone(&fix.provider), &fix.memory, &inputs, iterations);

    unsafe {
        std::env::remove_var("XLOG_WCOJ_W63_CHAIN_ENABLE");
    }
    let (chain_elapsed, chain_dispatches) =
        timed_chain_runs(Arc::clone(&fix.provider), &fix.memory, &inputs, iterations);

    unsafe {
        match old {
            Some(v) => std::env::set_var("XLOG_WCOJ_W63_CHAIN_ENABLE", v),
            None => std::env::remove_var("XLOG_WCOJ_W63_CHAIN_ENABLE"),
        }
    }

    let ratio = fallback_elapsed.as_secs_f64() / chain_elapsed.as_secs_f64();
    eprintln!(
        "W63_CHAIN_TIMING sorted_threshold n=2000 iterations={} fallback_ms={:.3} chain_ms={:.3} ratio={:.6} fallback_dispatches={} chain_dispatches={}",
        iterations,
        fallback_elapsed.as_secs_f64() * 1000.0,
        chain_elapsed.as_secs_f64() * 1000.0,
        ratio,
        fallback_dispatches,
        chain_dispatches
    );
    assert_eq!(fallback_dispatches, 0);
    assert_eq!(chain_dispatches, iterations as u64);
}
