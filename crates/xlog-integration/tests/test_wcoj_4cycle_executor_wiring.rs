// crates/xlog-integration/tests/test_wcoj_4cycle_executor_wiring.rs
//! v0.6.5 slice 2 — executor-level WCOJ 4-cycle dispatch wiring.
//!
//! End-to-end coverage for the 4-cycle dispatch path through
//! Compiler + Executor:
//!
//!   * Force-gate dispatches AND the row set matches the
//!     binary-join reference (gate-off baseline).
//!   * Gate-off path produces the same row set with counter == 0.
//!   * Kill switch beats force.
//!   * Adaptive opt-in defaults OFF (slice 2 contract).
//!
//! Mirrors `test_wcoj_executor_wiring.rs` (triangle) for the
//! 4-cycle shape.

use std::collections::BTreeMap;
use std::sync::Arc;

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

fn download_quads(buf: &CudaBuffer) -> Vec<(u32, u32, u32, u32)> {
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
    assert_eq!(buf.arity(), 4);
    let mut bytes = [
        vec![0u8; n * 4],
        vec![0u8; n * 4],
        vec![0u8; n * 4],
        vec![0u8; n * 4],
    ];
    for i in 0..4 {
        unsafe {
            sys::cuMemcpyDtoH_v2(
                bytes[i].as_mut_ptr() as *mut _,
                *buf.column(i).unwrap().device_ptr(),
                bytes[i].len(),
            );
        }
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push((
            u32::from_le_bytes(bytes[0][i * 4..i * 4 + 4].try_into().unwrap()),
            u32::from_le_bytes(bytes[1][i * 4..i * 4 + 4].try_into().unwrap()),
            u32::from_le_bytes(bytes[2][i * 4..i * 4 + 4].try_into().unwrap()),
            u32::from_le_bytes(bytes[3][i * 4..i * 4 + 4].try_into().unwrap()),
        ));
    }
    out.sort();
    out.dedup();
    out
}

const FOUR_CYCLE_SOURCE: &str = "cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).";

/// Dedicated 4-cycle fixture — distinct from the triangle K_4
/// fixture. Vertices {1, 2, 3, 4, 5, 6} with two embedded 4-cycles:
/// {1→2→3→4→1} and {1→5→6→4→1}. e1, e2, e3, e4 each carry the same
/// edge set (all directed edges among the cycles).
fn fourcycle_fixture() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    let edges = vec![
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
        // second cycle through extra vertices
        (1, 5),
        (5, 6),
        (6, 4),
    ];
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
) -> (Executor, u64, u64) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(FOUR_CYCLE_SOURCE).expect("compile");
    let mut executor = Executor::new_with_config(provider, config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        let buf = upload_binary_u32(memory, rows);
        executor.put_relation(name, buf);
    }
    executor.execute_plan(&plan).expect("execute_plan");
    let tri_counter = executor.wcoj_triangle_dispatch_count();
    let four_counter = executor.wcoj_4cycle_dispatch_count();
    (executor, tri_counter, four_counter)
}

#[test]
fn wiring_gate_off_does_not_dispatch_and_produces_row_set() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = fourcycle_fixture();
    let config = RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(false));
    let (executor, tri, four) =
        run_program(Arc::clone(&fix.provider), &fix.memory, config, &inputs);
    assert_eq!(
        tri, 0,
        "triangle dispatch must not fire on a 4-cycle program"
    );
    assert_eq!(four, 0, "gate=Some(false) must not dispatch 4-cycle");
    let rows = download_quads(executor.store().get("cycle4").expect("cycle4 present"));
    assert!(
        !rows.is_empty(),
        "binary-join executor produced an empty 4-cycle result on K_4-like fixture"
    );
}

#[test]
fn wiring_gate_on_dispatches_and_matches_binary_join_output() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = fourcycle_fixture();

    // Reference: gate off (binary-join chain).
    let config_off = RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(false));
    let (exec_off, _, four_off) =
        run_program(Arc::clone(&fix.provider), &fix.memory, config_off, &inputs);
    assert_eq!(four_off, 0);
    let reference_rows = download_quads(exec_off.store().get("cycle4").expect("cycle4"));

    // Force gate on.
    let config_on = RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(true));
    let (exec_on, tri_on, four_on) =
        run_program(Arc::clone(&fix.provider), &fix.memory, config_on, &inputs);
    assert_eq!(
        tri_on, 0,
        "triangle counter must stay 0 on a 4-cycle program"
    );
    assert_eq!(
        four_on, 1,
        "gate=Some(true) on the 4-cycle rule must dispatch exactly once; \
         got 4-cycle counter {four_on}"
    );
    let dispatch_rows = download_quads(exec_on.store().get("cycle4").expect("cycle4"));
    assert_eq!(
        dispatch_rows, reference_rows,
        "WCOJ 4-cycle dispatch must produce the same row set as the binary-join reference"
    );
}

#[test]
fn wiring_kill_switch_beats_force() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = fourcycle_fixture();
    let config = RuntimeConfig::default()
        .with_wcoj_4cycle_dispatch(Some(true))
        .with_wcoj_4cycle_dispatch_disabled(Some(true));
    let (executor, _, four) = run_program(Arc::clone(&fix.provider), &fix.memory, config, &inputs);
    assert_eq!(four, 0, "kill switch must override force-on");
    // Result still computed via binary-join fallback.
    let rows = download_quads(executor.store().get("cycle4").expect("cycle4"));
    assert!(!rows.is_empty());
}

#[test]
fn wiring_adaptive_optin_default_off_does_not_dispatch() {
    // Slice 2 contract: 4-cycle adaptive defaults OFF (opt-in).
    // With NO explicit force gate and NO adaptive opt-in, no
    // dispatch fires. Triangle would default-on the adaptive
    // here; 4-cycle does not.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = fourcycle_fixture();
    let config = RuntimeConfig::default(); // no overrides
    let (_executor, _, four) = run_program(Arc::clone(&fix.provider), &fix.memory, config, &inputs);
    assert_eq!(
        four, 0,
        "adaptive defaults OFF for 4-cycle (slice 2 contract); no dispatch on default config"
    );
}

// -----------------------------------------------------------------
// Symbol parity — Symbol shares u32's 4-byte physical layout, so
// the kernel + matcher accept Symbol 4-cycle inputs unchanged.
// -----------------------------------------------------------------

/// Symbol-typed sibling of `upload_binary_u32`. Same on-device
/// bytes; only the schema differs.
fn upload_binary_symbol(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
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
        ("col0".to_string(), ScalarType::Symbol),
        ("col1".to_string(), ScalarType::Symbol),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

#[test]
fn wiring_gate_on_symbol_4cycle_dispatches_and_preserves_schema() {
    // Same 4-cycle topology + bits as the U32 wiring test, but the
    // input buffers carry Symbol-typed schemas. The classifier and
    // kernel read the same 4-byte bits unchanged; the output
    // buffer's schema preserves Symbol per column (no silent
    // widening to U32).
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = fourcycle_fixture();

    let mut compiler = Compiler::new();
    let plan = compiler.compile(FOUR_CYCLE_SOURCE).expect("compile");
    let config = RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(true));
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in &inputs {
        let buf = upload_binary_symbol(&fix.memory, rows);
        executor.put_relation(name, buf);
    }
    executor.execute_plan(&plan).expect("execute_plan");
    let counter = executor.wcoj_4cycle_dispatch_count();
    assert_eq!(
        counter, 1,
        "gate=Some(true) on Symbol-typed 4-cycle inputs must dispatch exactly once; \
         got counter {counter}"
    );

    // Schema preservation: the kernel built its output schema from
    // the inputs' per-column types, so Symbol-input → Symbol-output.
    let buf = executor.store().get("cycle4").expect("cycle4 present");
    assert_eq!(buf.schema.column_type(0), Some(ScalarType::Symbol));
    assert_eq!(buf.schema.column_type(1), Some(ScalarType::Symbol));
    assert_eq!(buf.schema.column_type(2), Some(ScalarType::Symbol));
    assert_eq!(buf.schema.column_type(3), Some(ScalarType::Symbol));

    // Row set: the kernel's bit-equality joins produce the same
    // quads as the U32 path on the same bit-pattern fixture.
    let rows = download_quads(buf);
    assert!(
        !rows.is_empty(),
        "Symbol 4-cycle must produce non-empty rows"
    );
}
