// crates/xlog-integration/tests/test_wcoj_4cycle_rir_shape_cert.rs
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

//! RIR-shape certification for the 4-cycle WCOJ
//! dispatch.
//!
//! Locks the v1 dispatch policy across syntactic variants of a
//! 4-cycle rule. The matcher in
//! `xlog_runtime::executor::wcoj_dispatch::match_multiway_4cycle`
//! and the promoter in `xlog_logic::promote::try_promote_4cycle`
//! together accept exactly the canonical bushy RIR shape that the
//! lowerer produces for
//! `cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W)`.
//! Equivalent rules whose RIR shape differs (rotated head order,
//! injected filter, fewer head columns) are intentionally **out of
//! v1 dispatch scope** and silently fall back to the embedded
//! binary-join `fallback`.
//!
//! Each test runs the same fixture through Compiler + Executor
//! twice (gate off vs. gate on) and asserts:
//!   * Gate-off counter is always 0.
//!   * Gate-on counter equals `expected_dispatched` (1 if matcher
//!     accepts; 0 if rejected).
//!   * Result row sets between the two paths are identical.

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
struct RuntimeFixture {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
}

fn make_runtime_fixture() -> Option<RuntimeFixture> {
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
    Some(RuntimeFixture {
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

fn download_n_cols(buf: &CudaBuffer, arity: usize) -> Vec<Vec<u32>> {
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
        return vec![Vec::new(); arity];
    }
    let mut cols = vec![Vec::with_capacity(n); arity];
    for c in 0..arity {
        let mut bytes = vec![0u8; n * 4];
        unsafe {
            sys::cuMemcpyDtoH_v2(
                bytes.as_mut_ptr() as *mut _,
                *buf.column(c).unwrap().device_ptr(),
                bytes.len(),
            );
        }
        for i in 0..n {
            cols[c].push(u32::from_le_bytes(
                bytes[i * 4..i * 4 + 4].try_into().unwrap(),
            ));
        }
    }
    cols
}

fn rows_sorted_dedup(cols: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let n = cols[0].len();
    let mut rows: Vec<Vec<u32>> = (0..n)
        .map(|i| cols.iter().map(|c| c[i]).collect())
        .collect();
    rows.sort();
    rows.dedup();
    rows
}

fn fourcycle_fixture() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    let edges: Vec<(u32, u32)> = vec![(1, 2), (2, 3), (3, 4), (4, 1), (1, 5), (5, 6), (6, 4)];
    m.insert("e1", edges.clone());
    m.insert("e2", edges.clone());
    m.insert("e3", edges.clone());
    m.insert("e4", edges);
    m
}

fn run_program(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    gate: bool,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> (Executor, u64) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile");
    let config = RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(gate));
    let mut executor = Executor::new_with_config(provider, config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        let buf = upload_binary_u32(memory, rows);
        executor.put_relation(name, buf);
    }
    executor.execute_plan(&plan).expect("execute_plan");
    let counter = executor.wcoj_4cycle_dispatch_count();
    (executor, counter)
}

fn assert_dispatch_policy(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    head_predicate: &str,
    head_arity: usize,
    expected_dispatched: u64,
    case_label: &str,
) {
    let (exec_off, counter_off) = run_program(Arc::clone(&provider), memory, false, source, inputs);
    assert_eq!(counter_off, 0, "[{case_label}] gate-off must not dispatch");
    let buf_off = exec_off
        .store()
        .get(head_predicate)
        .expect("head predicate present (gate off)");
    let cols_off = download_n_cols(buf_off, head_arity);
    let rows_off = rows_sorted_dedup(&cols_off);

    let (exec_on, counter_on) = run_program(Arc::clone(&provider), memory, true, source, inputs);
    assert_eq!(
        counter_on, expected_dispatched,
        "[{case_label}] gate-on counter: expected {expected_dispatched}, got {counter_on}"
    );
    let buf_on = exec_on
        .store()
        .get(head_predicate)
        .expect("head predicate present (gate on)");
    let cols_on = download_n_cols(buf_on, head_arity);
    let rows_on = rows_sorted_dedup(&cols_on);
    assert_eq!(
        rows_off, rows_on,
        "[{case_label}] gate-on row set must equal gate-off reference"
    );
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[test]
fn rir_cert_canonical_4cycle_dispatches() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    assert_dispatch_policy(
        Arc::clone(&fix.provider),
        &fix.memory,
        "cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).",
        &fourcycle_fixture(),
        "cycle4",
        4,
        1,
        "canonical",
    );
}

#[test]
fn rir_cert_head_var_order_rotated_falls_back() {
    // Head order rotated: cycle4'(X, Y, Z, W) instead of (W, X, Y, Z).
    // The lowerer emits a different ProjectExpr column order, so the
    // matcher's strict output_columns check rejects.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    assert_dispatch_policy(
        Arc::clone(&fix.provider),
        &fix.memory,
        "rotated(X, Y, Z, W) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).",
        &fourcycle_fixture(),
        "rotated",
        4,
        0,
        "head_rotated",
    );
}

#[test]
fn rir_cert_3arity_head_falls_back() {
    // Head omits Z. Output projection has only 3 columns; matcher
    // rejects (canonical requires exactly 4).
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    assert_dispatch_policy(
        Arc::clone(&fix.provider),
        &fix.memory,
        "three(W, X, Y) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).",
        &fourcycle_fixture(),
        "three",
        3,
        0,
        "3arity_head",
    );
}

#[test]
fn rir_cert_with_comparison_filter_falls_back() {
    // Adds W < 100 — the lowerer inserts a Filter between Project
    // and the outer Join, which the strict matcher rejects.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    assert_dispatch_policy(
        Arc::clone(&fix.provider),
        &fix.memory,
        "filtered(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W), W != X.",
        &fourcycle_fixture(),
        "filtered",
        4,
        0,
        "comparison_filter",
    );
}
