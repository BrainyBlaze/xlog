// crates/xlog-integration/tests/test_wcoj_rir_shape_cert.rs
//! v0.6.2 RIR-shape certification for the executor's env-gated
//! WCOJ triangle dispatch.
//!
//! Locks the v1 dispatch policy across syntactic variants of a
//! triangle rule. As of `the executor's matcher
//! `xlog_runtime::executor::wcoj_dispatch::match_multiway_triangle`
//! consumes a `RirNode::MultiWayJoin` produced by
//! `xlog_logic::promote::promote_multiway` after the optimizer
//! pass. The matcher is intentionally narrow — it admits exactly
//! the canonical (X, Y, Z) emit order over Scan inputs whose
//! variable-class layout is `[[A,B],[B,C],[A,C]]`. Equivalent
//! rules whose post-optimizer RIR shape doesn't promote (different
//! join keys, different Project columns, an injected Filter, fewer
//! head columns) are intentionally **out of v1 dispatch scope** and
//! silently fall back to the embedded binary-join `fallback`
//! subtree, which produces the same row set.
//!
//! Each test below runs the same fixture through Compiler +
//! Executor twice (gate off vs. gate on) and asserts:
//!
//!   * On gate-off: counter == 0 (sanity).
//!   * On gate-on: counter is 0 or 1 per the locked policy.
//!   * Result row set matches the gate-off reference (correctness
//!     under both paths).
//!
//! The full policy table is documented in the slice commit. Each
//! test below carries the rationale for its specific decision.

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
// Fixture helpers (mirror test_wcoj_executor_wiring.rs conventions)
// ---------------------------------------------------------------

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

fn download_logical_count(buf: &CudaBuffer) -> usize {
    if let Some(c) = buf.cached_row_count() {
        return c as usize;
    }
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

fn download_n_cols(buf: &CudaBuffer, ncols: usize) -> Vec<Vec<u32>> {
    let n = download_logical_count(buf);
    let mut cols: Vec<Vec<u32>> = vec![Vec::with_capacity(n); ncols];
    if n == 0 {
        return cols;
    }
    assert_eq!(buf.arity(), ncols);
    for (c, col) in cols.iter_mut().enumerate().take(ncols) {
        let mut bytes = vec![0u8; n * 4];
        unsafe {
            sys::cuMemcpyDtoH_v2(
                bytes.as_mut_ptr() as *mut _,
                *buf.column(c).unwrap().device_ptr(),
                bytes.len(),
            );
        }
        for i in 0..n {
            col.push(u32::from_le_bytes(
                bytes[i * 4..i * 4 + 4].try_into().unwrap(),
            ));
        }
    }
    cols
}

/// Sort rows lex (across all columns) and dedup. Used to compare
/// outputs whose row order might differ between paths.
fn rows_sorted_dedup(cols: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let n = cols.first().map(|c| c.len()).unwrap_or(0);
    let mut rows: Vec<Vec<u32>> = (0..n)
        .map(|i| cols.iter().map(|c| c[i]).collect())
        .collect();
    rows.sort();
    rows.dedup();
    rows
}

/// Run the source program through Compiler+Executor with the
/// given gate setting. Returns (executor, dispatch_counter).
fn run_program(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    gate: bool,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> (Executor, u64) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile");
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(gate));
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

/// Build the K_4 + small-disjoint-triangle fixture used across
/// the cert tests. Three logically-distinct edge predicates with
/// the same content as the canonical triangle test fixture.
fn three_relation_fixture() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
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

/// Cross-check helper: run the same `source` + `inputs` with
/// gate off (reference) and gate on (under test), assert:
///   * gate-off counter is always 0.
///   * gate-on counter equals `expected_dispatched` (1 if the
///     RIR matches and dispatch fires; 0 if the matcher rejects
///     and the binary path takes over).
///   * The output row set for `head_predicate` is identical
///     between the two paths (correctness under both).
///
/// `head_arity` selects how many output columns to download.
#[allow(clippy::too_many_arguments)]
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
        "[{case_label}] gate-on dispatch count: expected {expected_dispatched}, got {counter_on}"
    );
    let buf_on = exec_on
        .store()
        .get(head_predicate)
        .expect("head predicate present (gate on)");
    let cols_on = download_n_cols(buf_on, head_arity);
    let rows_on = rows_sorted_dedup(&cols_on);

    assert_eq!(
        rows_on, rows_off,
        "[{case_label}] gate-on output must equal gate-off reference row-for-row"
    );
}

// ---------------------------------------------------------------
// Variant tests
// ---------------------------------------------------------------

#[test]
fn rir_cert_canonical_triangle_dispatches() {
    // Sanity: the v1-matched canonical shape still dispatches and
    // produces the correct answer. Locks the existing behavior so
    // future matcher changes can't silently regress this case.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = three_relation_fixture();
    assert_dispatch_policy(
        Arc::clone(&fix.provider),
        &fix.memory,
        "tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).",
        &inputs,
        "tri",
        3,
        1,
        "canonical",
    );
}

#[test]
fn rir_cert_single_relation_triangle_dispatches() {
    // Single-relation triangle: all three atoms over the same
    // edge predicate. The lowered RIR is structurally identical
    // to the canonical (just with all three Scan rel IDs equal).
    // The executor matcher passes; the dispatch happens with the
    // same buffer in all three slots — `wcoj_layout_u32_recorded`
    // produces three independent sorted+deduped layouts of the
    // same underlying data, no aliasing issues.
    //
    // This is a common natural shape for graph queries (e.g.
    // counting triangles in an undirected graph encoded by a
    // single edge relation), so dispatching here is the
    // higher-leverage choice.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    inputs.insert(
        "e",
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
    assert_dispatch_policy(
        Arc::clone(&fix.provider),
        &fix.memory,
        "tri(X, Y, Z) :- e(X, Y), e(Y, Z), e(X, Z).",
        &inputs,
        "tri",
        3,
        1,
        "single-relation",
    );
}

#[test]
fn rir_cert_body_order_rotated_falls_back() {
    // Body atom order rotated: `e3(X, Z), e1(X, Y), e2(Y, Z)`.
    // The lowerer compiles this with a different join graph
    // (X-join inner, then [Y, Z] outer) — RIR keys differ from
    // canonical. v1 matcher rejects; binary-join chain produces
    // the answer.
    //
    // Decision rationale: a body rotation produces a structurally
    // different RIR; v1's slot identification (e_xy / e_yz /
    // e_xz) depends on the canonical key positions. Generalizing
    // would require enumerating all six rotations and reordering
    // kernel inputs accordingly — separate slice's worth of work.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = three_relation_fixture();
    assert_dispatch_policy(
        Arc::clone(&fix.provider),
        &fix.memory,
        "tri(X, Y, Z) :- e3(X, Z), e1(X, Y), e2(Y, Z).",
        &inputs,
        "tri",
        3,
        0,
        "body-rotated",
    );
}

#[test]
fn rir_cert_head_var_order_rotated_falls_back() {
    // Head var order rotated: `tri(Y, X, Z) :- e1(X, Y), ...`.
    // The body RIR is unchanged from canonical, but the Project
    // permutes the output columns: `[Column(1), Column(0),
    // Column(3)]` instead of `[0, 1, 3]`. v1 matcher rejects on
    // the column check; binary path produces the answer.
    //
    // Decision rationale: the WCOJ kernel emits (X, Y, Z) in
    // fixed order matching the original head positions. To
    // dispatch a permuted head, the hook would need to reorder
    // the output columns post-kernel. v1 keeps the kernel
    // contract narrow; column reordering is a follow-up slice.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = three_relation_fixture();
    assert_dispatch_policy(
        Arc::clone(&fix.provider),
        &fix.memory,
        "permuted(Y, X, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).",
        &inputs,
        "permuted",
        3,
        0,
        "head-permuted",
    );
}

#[test]
fn rir_cert_reversed_atom_falls_back() {
    // One atom's args are NOT in head-position order:
    // `e1(Y, X)` instead of `e1(X, Y)`. The lowerer joins on
    // the reversed column (X-join via [0]=[0] in the inner Join,
    // not [1]=[0]), producing a different RIR. v1 matcher
    // rejects; binary path produces the answer.
    //
    // Decision rationale: a reversed atom's column semantics
    // don't align with the WCOJ kernel's slot expectations
    // (e_xy.col0=X, e_xy.col1=Y). Supporting reversed inputs
    // means either (a) reordering the input buffer columns
    // before the layout pass or (b) extending the kernel to
    // accept reversed inputs. Both are out of v1 scope.
    //
    // Note: e1(Y, X) and e1(X, Y) are NOT semantically equivalent
    // unless e1 is symmetric, so the row sets here may differ
    // from the canonical case. Both gate-off and gate-on paths
    // see the same logical rule, so they produce the same
    // (possibly empty or different-from-canonical) result, which
    // the assertion still locks.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = three_relation_fixture();
    assert_dispatch_policy(
        Arc::clone(&fix.provider),
        &fix.memory,
        "tri(X, Y, Z) :- e1(Y, X), e2(Y, Z), e3(X, Z).",
        &inputs,
        "tri",
        3,
        0,
        "reversed-atom",
    );
}

#[test]
fn rir_cert_with_comparison_filter_falls_back() {
    // Adding a body comparison `X < Y` injects a `Filter` node
    // between the inner Join and the outer Join in the RIR. v1
    // matcher requires Project→Join→Join; the extra Filter
    // breaks the pattern match. Binary path produces the answer.
    //
    // Decision rationale: the WCOJ kernel is set-semantic on
    // intersections. Comparison filters change intermediate row
    // sets in ways that don't compose with the kernel's
    // assumptions. Supporting them means either (a) lowering
    // the filter into the WCOJ kernel itself or (b) running a
    // separate filter pass on the kernel output. Both are
    // separate slices; v1 falls back.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = three_relation_fixture();
    assert_dispatch_policy(
        Arc::clone(&fix.provider),
        &fix.memory,
        "tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z), X < Y.",
        &inputs,
        "tri",
        3,
        0,
        "comparison-filter",
    );
}

#[test]
fn rir_cert_2arity_head_falls_back() {
    // Head has only 2 vars: `two(X, Z) :- e1(X, Y), ...`. The
    // body's Y is projected away. Project columns are `[0, 3]`
    // (2 columns). v1 matcher requires exactly 3 Project
    // columns — falls back.
    //
    // Decision rationale: the WCOJ kernel always emits 3
    // columns. Supporting a 2-arity head means projecting one
    // column away after the kernel runs. v1's hook does not
    // post-process kernel output; falls back to keep the kernel
    // contract narrow.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = three_relation_fixture();
    assert_dispatch_policy(
        Arc::clone(&fix.provider),
        &fix.memory,
        "two(X, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).",
        &inputs,
        "two",
        2,
        0,
        "2-arity-head",
    );
}
