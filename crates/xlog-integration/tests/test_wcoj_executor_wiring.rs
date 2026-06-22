// crates/xlog-integration/tests/test_wcoj_executor_wiring.rs
//! Executor-level WCOJ triangle dispatch wiring.
//!
//! Locks the contract for the hook installed in
//! [`xlog_runtime::Executor::execute_plan`]'s non-recursive
//! per-rule branch. The hook engages when ALL of the following
//! hold:
//!
//!   * Dispatch is not hard-disabled
//!     (`wcoj_triangle_dispatch_disabled` /
//!     `XLOG_DISABLE_WCOJ_TRIANGLE` not set), AND
//!   * Either force-on (`wcoj_triangle_dispatch=Some(true)` /
//!     `XLOG_USE_WCOJ_TRIANGLE_U32=1`) is set, OR the adaptive
//!     classifier is on (default-on after adaptive dispatch was introduced, opt-out via
//!     `wcoj_triangle_dispatch_adaptive=Some(false)`).
//!   * The rule's RIR matches the canonical triangle shape:
//!     `Project([0, 1, 3]) → Join → Join → Scan, Scan, Scan`
//!     with the join keys produced by lowering
//!     `tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z)`.
//!   * All three input buffers are 2-column u32 / Symbol /
//!     u64 with widths agreeing across slots.
//!   * A runtime-backed `GpuMemoryManager` is available (the
//!     recorded WCOJ primitives require it).
//!
//! These wiring tests use explicit `with_wcoj_triangle_dispatch(Some(...))`
//! rather than relying on the default to avoid coupling to the
//! adaptive classifier's score on small fixtures. Default-on
//! adaptive behavior is locked separately in
//! `test_wcoj_adaptive_default_on.rs`.
//!
//! Test surface (all run end-to-end via Compiler + Executor):
//!   1. gate=Some(false) on the triangle rule produces the
//!      reference output AND `wcoj_triangle_dispatch_count() == 0`
//!      (locks "no-surprise opt-in").
//!   2. gate=Some(true) on the triangle rule produces the same
//!      reference output (binary-join answer) AND
//!      `wcoj_triangle_dispatch_count() == 1` (locks "WCOJ path
//!      actually fires under the gate, with row-set agreement").
//!   3. gate=Some(true) on a non-matching shape (2-atom path
//!      rule) falls back silently: `wcoj_triangle_dispatch_count()
//!      == 0` and output equals the binary-join reference.
//!   4. gate=Some(true) on a recursive rule (transitive closure)
//!      stays in the existing recursive path: counter == 0,
//!      output correct.
//!   5. gate=Some(true) on a triangle whose inputs are NOT
//!      runtime-backed (legacy `GpuMemoryManager`) falls back:
//!      counter == 0, output correct via the binary path.

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

// ---------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)] // device/runtime kept alive via Arc clones for cross-stream lifetimes
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

#[allow(dead_code)]
struct LegacyFixture {
    device: Arc<CudaDevice>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
}

fn make_legacy_fixture() -> Option<LegacyFixture> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let memory = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(64 * 1024 * 1024),
    ));
    let provider =
        Arc::new(CudaKernelProvider::new(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(LegacyFixture {
        device,
        memory,
        provider,
    })
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let col0_host: Vec<u32> = rows.iter().map(|(a, _)| *a).collect();
    let col1_host: Vec<u32> = rows.iter().map(|(_, b)| *b).collect();
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !col0_host.is_empty() {
        let col0_bytes: Vec<u8> = col0_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let col1_bytes: Vec<u8> = col1_host.iter().flat_map(|v| v.to_le_bytes()).collect();
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

/// Read a 3-column u32 result buffer to a sorted, deduped
/// `Vec<(u32, u32, u32)>`. Uses cached_row_count when available
/// (set by the executor / WCOJ kernel) and falls back to a 4-byte
/// device-to-host read of `d_num_rows` for compact-in-place outputs.
fn download_triples(buf: &CudaBuffer) -> Vec<(u32, u32, u32)> {
    let n = match buf.cached_row_count() {
        Some(c) => c as usize,
        None => {
            let mut count_host = [0u32; 1];
            unsafe {
                let res = sys::cuMemcpyDtoH_v2(
                    count_host.as_mut_ptr() as *mut _,
                    *buf.num_rows_device().device_ptr(),
                    std::mem::size_of::<u32>(),
                );
                assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
            }
            count_host[0] as usize
        }
    };
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(buf.arity(), 3);
    let mut col0_bytes = vec![0u8; n * 4];
    let mut col1_bytes = vec![0u8; n * 4];
    let mut col2_bytes = vec![0u8; n * 4];
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
        sys::cuMemcpyDtoH_v2(
            col2_bytes.as_mut_ptr() as *mut _,
            *buf.column(2).unwrap().device_ptr(),
            col2_bytes.len(),
        );
    }
    let mut out: Vec<(u32, u32, u32)> = (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(col0_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col1_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col2_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            )
        })
        .collect();
    out.sort();
    out.dedup();
    out
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

/// Run a `.xlog` source program through Compiler + Executor and
/// return the final relation store keyed by predicate name. The
/// executor counter snapshot is also returned so tests can
/// distinguish "WCOJ fired" from "binary path fired with same
/// answer."
fn run_program(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    config: RuntimeConfig,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> (Executor, u64) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile");
    let mut executor = Executor::new_with_config(provider, config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        let buf = upload_binary_u32(memory, rows);
        executor.put_relation(name, buf);
    }
    let _ = executor.execute_plan(&plan).expect("execute_plan");
    let counter = executor.wcoj_triangle_dispatch_count();
    (executor, counter)
}

// ---------------------------------------------------------------
// Triangle fixture (shared across multiple tests)
// ---------------------------------------------------------------

const TRIANGLE_SOURCE: &str = "tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).";

fn triangle_fixture() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    // K_4 + small triangle, intentionally unsorted+duplicated so the
    // executor's binary-join chain and the WCOJ dispatch both have to
    // do real work.
    m.insert(
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
    m.insert("e2", vec![(2, 3), (2, 4), (3, 4), (6, 7)]);
    m.insert("e3", vec![(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)]);
    m
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[test]
fn wiring_gate_off_does_not_dispatch_and_matches_reference() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = triangle_fixture();
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false));
    let (executor, counter) = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        config,
        TRIANGLE_SOURCE,
        &inputs,
    );
    assert_eq!(
        counter, 0,
        "gate=Some(false) must not dispatch; got counter {counter}"
    );
    // Sanity: result has rows. Locked by the gate-on test below.
    let buf = executor.store().get("tri").expect("tri present");
    let rows = download_triples(buf);
    assert!(
        !rows.is_empty(),
        "binary-join executor produced an empty triangle result"
    );
}

#[test]
fn wiring_gate_on_dispatches_and_matches_binary_join_output() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = triangle_fixture();

    // ----- reference: gate off (existing binary-join chain) -----
    let config_off = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false));
    let (exec_off, counter_off) = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        config_off,
        TRIANGLE_SOURCE,
        &inputs,
    );
    assert_eq!(counter_off, 0);
    assert_eq!(fix.provider.wcoj_triangle_hg_dispatch_count(), 0);
    let reference_rows = download_triples(exec_off.store().get("tri").expect("tri"));

    // ----- gate on: WCOJ dispatch must fire AND match the reference -----
    let hg_before = fix.provider.wcoj_triangle_hg_dispatch_count();
    let config_on = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true));
    let (exec_on, counter_on) = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        config_on,
        TRIANGLE_SOURCE,
        &inputs,
    );
    assert_eq!(
        counter_on, 1,
        "gate=Some(true) on the triangle rule must dispatch exactly once; \
         got counter {counter_on} (likely the RIR matcher silently fell back, \
         leaving the binary path to produce the answer)"
    );
    assert_eq!(
        fix.provider.wcoj_triangle_hg_dispatch_count(),
        hg_before + 1,
        "gate=Some(true) must route the u32 triangle through the HG block-slice provider entry"
    );
    let dispatch_rows = download_triples(exec_on.store().get("tri").expect("tri"));
    assert_eq!(
        dispatch_rows, reference_rows,
        "WCOJ dispatch output must equal the binary-join reference row-for-row"
    );
}

#[test]
fn wiring_gate_on_two_atom_rule_falls_back_silently() {
    // 2-atom path rule does not match the triangle RIR shape.
    // Hook returns Ok(None); binary-join chain produces the
    // result. Counter stays at 0.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let source = "path(X, Z) :- e1(X, Y), e2(Y, Z).";
    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    inputs.insert("e1", vec![(1, 2), (2, 3), (3, 4)]);
    inputs.insert("e2", vec![(2, 5), (3, 6), (4, 7)]);
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true));
    let (executor, counter) = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        config,
        source,
        &inputs,
    );
    assert_eq!(
        counter, 0,
        "2-atom rule must NOT dispatch; got counter {counter}"
    );
    let path_buf = executor.store().get("path").expect("path present");
    let rows = download_pairs(path_buf);
    let mut expected = vec![(1u32, 5u32), (2, 6), (3, 7)];
    expected.sort();
    assert_eq!(rows, expected);
}

// Note on recursive-rule fallback:
//
// The recursive-rule case is locked structurally rather than
// behaviorally. In `executor::recursive::execute_stratum_impl`,
// the WCOJ dispatch hook (`try_dispatch_wcoj_triangle`) is
// called inside the `else` branch of `if is_recursive { ... }
// else { ... }` — recursive SCCs route through
// `execute_recursive_scc`, never touching the hook. A
// behavioral test would require constructing a working TC-style
// program through the existing Compiler+Executor pipeline; the
// runtime-supplied edge-buffer schemas conflict with the
// compiler's inferred recursive-relation schema (independent
// pre-existing infrastructure friction). The two-atom path
// fallback test above already exercises the silent-fallback
// contract on a non-matching shape; the structural defense
// covers the recursive case.

#[test]
fn wiring_gate_on_legacy_manager_falls_back_silently() {
    // Legacy GpuMemoryManager (no runtime) → recorded WCOJ
    // primitives can't run; the dispatch hook detects this and
    // returns Ok(None). Binary-join chain produces the answer.
    let Some(fix) = make_legacy_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = triangle_fixture();
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true));
    let (executor, counter) = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        config,
        TRIANGLE_SOURCE,
        &inputs,
    );
    assert_eq!(
        counter, 0,
        "legacy manager must trigger silent fallback; got counter {counter}"
    );
    let buf = executor.store().get("tri").expect("tri present");
    let rows = download_triples(buf);
    assert!(
        !rows.is_empty(),
        "binary-join chain must produce some triangles"
    );
}

// ---------------------------------------------------------------
// Symbol support — Symbol shares u32's 4-byte physical layout, so
// the kernel + RIR matcher accept Symbol triangles unchanged.
// ---------------------------------------------------------------

/// Symbol-typed sibling of `upload_binary_u32`. Same on-device
/// bytes; only the schema differs.
fn upload_binary_symbol(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let col0_host: Vec<u32> = rows.iter().map(|(a, _)| *a).collect();
    let col1_host: Vec<u32> = rows.iter().map(|(_, b)| *b).collect();
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !col0_host.is_empty() {
        let col0_bytes: Vec<u8> = col0_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let col1_bytes: Vec<u8> = col1_host.iter().flat_map(|v| v.to_le_bytes()).collect();
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
fn wiring_gate_on_symbol_triangle_dispatches_and_preserves_schema() {
    // Same triangle topology + bits as the U32 wiring test, but
    // the input buffers (registered via `put_relation`) carry
    // Symbol-typed schemas. The executor's RIR matcher accepts
    // Symbol via the widened `is_two_col_u32`, the WCOJ kernel
    // reads the same 4-byte bits unchanged, and the output
    // buffer's schema preserves Symbol per column (no silent
    // widening to U32).
    //
    // We don't compare against a gate-off reference here — the
    // existing binary-join chain may apply schema policies
    // (Union, dedup) that are unrelated to this path. The check
    // is narrower: gate-on dispatches, output schema is correct,
    // row count is the expected 5.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = triangle_fixture();

    let mut compiler = Compiler::new();
    let plan = compiler.compile(TRIANGLE_SOURCE).expect("compile");
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true));
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in &inputs {
        let buf = upload_binary_symbol(&fix.memory, rows);
        executor.put_relation(name, buf);
    }
    let _ = executor.execute_plan(&plan).expect("execute_plan");
    let counter = executor.wcoj_triangle_dispatch_count();
    assert_eq!(
        counter, 1,
        "gate=Some(true) on Symbol-typed triangle inputs must dispatch exactly once; \
         got counter {counter}"
    );

    // Schema preservation: the kernel built its output schema
    // from the inputs' per-column types, so Symbol-input → Symbol-output.
    let tri_buf = executor.store().get("tri").expect("tri present");
    assert_eq!(tri_buf.schema.column_type(0), Some(ScalarType::Symbol));
    assert_eq!(tri_buf.schema.column_type(1), Some(ScalarType::Symbol));
    assert_eq!(tri_buf.schema.column_type(2), Some(ScalarType::Symbol));

    // Row count: the kernel's bit-equality joins produce the
    // same 5 triangles as the U32 path on the same bit-pattern
    // fixture.
    let rows = download_triples(tri_buf);
    assert_eq!(
        rows.len(),
        5,
        "expected 5 triangles on this fixture; got {}",
        rows.len()
    );
}

// ---------------------------------------------------------------
// U64 + mixed-width executor wiring.
// ---------------------------------------------------------------

/// U64 sibling of `upload_binary_u32` / `upload_binary_symbol`.
fn upload_binary_u64(memory: &Arc<GpuMemoryManager>, rows: &[(u64, u64)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let col0_host: Vec<u64> = rows.iter().map(|(a, _)| *a).collect();
    let col1_host: Vec<u64> = rows.iter().map(|(_, b)| *b).collect();
    let bytes_per_col = (n as usize) * std::mem::size_of::<u64>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !col0_host.is_empty() {
        let c0: Vec<u8> = col0_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let c1: Vec<u8> = col1_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
        device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![
        ("col0".to_string(), ScalarType::U64),
        ("col1".to_string(), ScalarType::U64),
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
fn wiring_gate_on_u64_triangle_dispatches_and_preserves_schema() {
    // U64-typed triangle inputs registered via `put_relation`.
    // The executor's RIR matcher accepts U64 via the widened
    // width admission, the WCOJ U64 entry runs, and the output
    // buffer's schema preserves U64 per column.
    //
    // Fixture is the multi-triangle topology shifted into hi-half
    // u64 space — a buggy width-truncating dispatch (e.g. routing
    // U64 inputs through the U32 entry) would visibly fail.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let big = (u32::MAX as u64) + 1;
    let mut inputs: BTreeMap<&str, Vec<(u64, u64)>> = BTreeMap::new();
    inputs.insert(
        "e1",
        vec![
            (big + 1, big + 2),
            (big + 1, big + 3),
            (big + 1, big + 4),
            (big + 2, big + 3),
            (big + 2, big + 4),
            (big + 3, big + 4),
            (big + 5, big + 6),
            (big + 5, big + 7),
            (big + 6, big + 7),
        ],
    );
    inputs.insert(
        "e2",
        vec![
            (big + 2, big + 3),
            (big + 2, big + 4),
            (big + 3, big + 4),
            (big + 6, big + 7),
        ],
    );
    inputs.insert(
        "e3",
        vec![
            (big + 1, big + 3),
            (big + 1, big + 4),
            (big + 2, big + 4),
            (big + 3, big + 4),
            (big + 5, big + 7),
        ],
    );

    let mut compiler = Compiler::new();
    let plan = compiler.compile(TRIANGLE_SOURCE).expect("compile");
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true));
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in &inputs {
        let buf = upload_binary_u64(&fix.memory, rows);
        executor.put_relation(name, buf);
    }
    let _ = executor.execute_plan(&plan).expect("execute_plan");
    let counter = executor.wcoj_triangle_dispatch_count();
    assert_eq!(
        counter, 1,
        "gate=Some(true) on U64-typed triangle inputs must dispatch exactly once; \
         got counter {counter}"
    );

    // Schema preservation: output columns must remain U64.
    let tri_buf = executor.store().get("tri").expect("tri present");
    assert_eq!(tri_buf.schema.column_type(0), Some(ScalarType::U64));
    assert_eq!(tri_buf.schema.column_type(1), Some(ScalarType::U64));
    assert_eq!(tri_buf.schema.column_type(2), Some(ScalarType::U64));

    // Same 5 triangles as the U32 path on the same fixture shape.
    assert_eq!(tri_buf.num_rows() as usize, 5);
}

#[test]
fn wiring_gate_on_mixed_u32_u64_triangle_falls_back_silently() {
    // Mixed-width triangle: e1 + e3 are U32, e2 is U64. The
    // RIR-level matcher must reject the dispatch (counter stays
    // 0) so the binary-join chain handles the rule. Locks that
    // a future schema-admission shortcut does not run bit-
    // equality joins across U32 and U64 buffers.
    //
    // We don't assert on the binary-join chain's output here —
    // the planner upstream would normally reject this fixture
    // via `analyze_typed`, and even if it didn't, the binary
    // executor's behavior on type-mixed inputs is out of scope.
    // The narrow lock is that the dispatch counter == 0.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let mut compiler = Compiler::new();
    let plan = compiler.compile(TRIANGLE_SOURCE).expect("compile");
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true));
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("e1", upload_binary_u32(&fix.memory, &[(1, 2)]));
    executor.put_relation("e2", upload_binary_u64(&fix.memory, &[(2, 3)]));
    executor.put_relation("e3", upload_binary_u32(&fix.memory, &[(1, 3)]));
    let _ = executor.execute_plan(&plan);
    let counter = executor.wcoj_triangle_dispatch_count();
    assert_eq!(
        counter, 0,
        "mixed-width triangle (U32 + U64 slots) must not dispatch the WCOJ path; \
         got counter {counter}"
    );
}
