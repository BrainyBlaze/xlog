// crates/xlog-integration/tests/test_wcoj_4cycle_adaptive_dispatch.rs
//! Adaptive opt-in dispatch for 4-cycle.
//!
//! Locks cardinality-backed adaptive behavior:
//!   * Seeded large relation cards → WCOJ dispatches.
//!   * Missing runtime stats → binary fallback.
//!   * Default config (no overrides, no env): adaptive is OFF →
//!     no dispatch (4-cycle adaptive contract; contrasts with triangle).

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
        let c0: Vec<u8> = rows.iter().flat_map(|&(a, _)| a.to_le_bytes()).collect();
        let c1: Vec<u8> = rows.iter().flat_map(|&(_, b)| b.to_le_bytes()).collect();
        device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
        device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
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

const FOUR_CYCLE_SOURCE: &str = "cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).";

/// Super-hub fixture: a single hub vertex (1) dominates the edge
/// list. Classifier should detect heavy concentration on at least
/// one column and produce a score well above 0.10.
fn superhub_fixture() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    let mut edges = Vec::new();
    // Hub at vertex 1: many edges share vertex 1 — strong skew.
    for v in 2..=300 {
        edges.push((1u32, v));
        edges.push((v, 1));
    }
    // A handful of non-hub edges so 4-cycles still exist (binary
    // fallback path needs to produce some output rows for the
    // executor to install; size doesn't affect the dispatch count
    // assertion but keeps the test from edge-casing).
    edges.push((2, 3));
    edges.push((3, 4));
    edges.push((4, 2));
    edges.sort();
    edges.dedup();
    m.insert("e1", edges.clone());
    m.insert("e2", edges.clone());
    m.insert("e3", edges.clone());
    m.insert("e4", edges);
    m
}

/// Uniform fixture: edges spread evenly across vertex pairs (no
/// hub). Classifier score stays below threshold. Kept small so the
/// binary-join fallback path stays within memory budget.
fn uniform_fixture() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    let mut edges = Vec::new();
    // 20x20 grid of edges, no hub. ~380 edges spread across 20
    // vertices — uniform per-vertex degree.
    for a in 1..=20u32 {
        for b in 1..=20u32 {
            if a != b {
                edges.push((a, b));
            }
        }
    }
    edges.sort();
    edges.dedup();
    m.insert("e1", edges.clone());
    m.insert("e2", edges.clone());
    m.insert("e3", edges.clone());
    m.insert("e4", edges);
    m
}

fn run_program(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    config: RuntimeConfig,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> u64 {
    run_program_with_cards(provider, memory, config, inputs, &BTreeMap::new())
}

fn run_program_with_cards(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    config: RuntimeConfig,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    seeded_cards: &BTreeMap<&str, u64>,
) -> u64 {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(FOUR_CYCLE_SOURCE).expect("compile");
    let mut executor = Executor::new_with_config(provider, config);
    let rel_ids = compiler.rel_ids().clone();
    for (name, rel_id) in &rel_ids {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        let buf = upload_binary_u32(memory, rows);
        executor.put_relation(name, buf);
    }
    for (name, card) in seeded_cards {
        if let Some(rel_id) = rel_ids.get(*name) {
            executor.stats_mut().register_relation(*rel_id);
            executor.stats_mut().update_cardinality(*rel_id, *card);
        }
    }
    executor.execute_plan(&plan).expect("execute_plan");
    executor.wcoj_4cycle_dispatch_count()
}

#[test]
fn adaptive_dispatches_on_superhub_fixture_with_seeded_cards() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let seeded_cards = BTreeMap::from([
        ("e1", 100_000u64),
        ("e2", 100_000u64),
        ("e3", 100_000u64),
        ("e4", 100_000u64),
    ]);
    let config = RuntimeConfig::default().with_wcoj_4cycle_dispatch_adaptive(Some(true));
    let counter = run_program_with_cards(
        Arc::clone(&fix.provider),
        &fix.memory,
        config,
        &superhub_fixture(),
        &seeded_cards,
    );
    assert_eq!(
        counter, 1,
        "adaptive opt-in with seeded large cards must engage the WCOJ kernel; \
         got dispatch counter {counter}"
    );
}

#[test]
fn adaptive_falls_back_on_uniform_fixture() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let config = RuntimeConfig::default().with_wcoj_4cycle_dispatch_adaptive(Some(true));
    let counter = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        config,
        &uniform_fixture(),
    );
    assert_eq!(
        counter, 0,
        "adaptive opt-in without seeded runtime stats must fall back; \
         got dispatch counter {counter}"
    );
}

#[test]
fn adaptive_default_off_does_not_dispatch_on_superhub() {
    // No overrides, no env. The 4-cycle adaptive dispatch contract
    // defaults OFF (opt-in). The same super-hub fixture that
    // would dispatch under adaptive=Some(true) must NOT dispatch
    // under default config.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let config = RuntimeConfig::default();
    let counter = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        config,
        &superhub_fixture(),
    );
    assert_eq!(
        counter, 0,
        "default config must not dispatch (4-cycle adaptive is opt-in)"
    );
}

#[test]
fn force_gate_dispatches_regardless_of_adaptive() {
    // Force gate must bypass adaptive/cardinality decisions even
    // when adaptive would decline.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let config = RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(true));
    let counter = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        config,
        &uniform_fixture(),
    );
    assert_eq!(counter, 1, "force gate must bypass adaptive cost model");
}
