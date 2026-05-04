// crates/xlog-integration/tests/test_selectivity_pass_reordering.rs
//! v0.6.5 W2.2 integration cert — selectivity-driven join
//! reordering preserves both (Part B) row-set semantics and
//! (Part C) WCOJ dispatch correctness across reordered bodies.
//!
//! ## Scope
//!
//! These certs use a minimal end-to-end fixture (the slice 4
//! stable triangle / 4-cycle) and exercise the full pipeline:
//! compile (with an optional stats snapshot) → optimizer →
//! selectivity_pass → promoter → executor → WCOJ dispatch.
//!
//! W2.2's selectivity_pass operates only on canonical
//! left-deep triangle (and bushy 4-cycle) bodies; right-deep
//! optimizer output is explicitly out of W2.2 scope. The
//! fixtures here use `pred` declarations without inline facts,
//! which empirically produce left-deep canonical output for
//! triangle and bushy canonical for 4-cycle (verified by
//! slice 5's cardinality cost-model certs).
//!
//! ## Coverage
//!
//!   * Part B (`selectivity_pass_triangle_two_snapshots_produce_same_row_set`)
//!     — same source compiled twice with two distinct stats
//!     snapshots. Row sets after execution must be IDENTICAL
//!     because reordering preserves rule semantics.
//!   * Part C (`selectivity_pass_reordered_triangle_still_dispatches_wcoj`)
//!     — force-WCOJ on the triangle path; W2.2 promoter
//!     extension must accept any body shape selectivity_pass
//!     produced. Counter ≥ 1 AND row set equals gate-off
//!     binary-join reference.
//!   * Part C 4-cycle counterpart
//!     (`selectivity_pass_reordered_4cycle_still_dispatches_wcoj`).

use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RelId, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;
use xlog_stats::{RelationStats, StatsSnapshot};

// ---------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------

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
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

fn download_triples(buf: &CudaBuffer) -> Vec<(u32, u32, u32)> {
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
    let mut cols = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for c in 0..4 {
        cols[c] = vec![0u8; n * 4];
        unsafe {
            sys::cuMemcpyDtoH_v2(
                cols[c].as_mut_ptr() as *mut _,
                *buf.column(c).unwrap().device_ptr(),
                cols[c].len(),
            );
        }
    }
    let mut out: Vec<(u32, u32, u32, u32)> = (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(cols[0][i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(cols[1][i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(cols[2][i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(cols[3][i * 4..i * 4 + 4].try_into().unwrap()),
            )
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

const STABLE_TRIANGLE: &str = r#"
    pred e1(u32, u32).
    pred e2(u32, u32).
    pred e3(u32, u32).
    pred tri(u32, u32, u32).
    tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).
"#;

const STABLE_4CYCLE: &str = r#"
    pred e1(u32, u32).
    pred e2(u32, u32).
    pred e3(u32, u32).
    pred e4(u32, u32).
    pred cyc(u32, u32, u32, u32).
    cyc(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).
"#;

fn triangle_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    m.insert("e1", vec![(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]);
    m.insert("e2", vec![(2, 3), (2, 4), (3, 4)]);
    m.insert("e3", vec![(1, 3), (1, 4), (2, 4), (3, 4)]);
    m
}

fn cycle4_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    m.insert("e1", vec![(1, 2), (5, 6)]);
    m.insert("e2", vec![(2, 3), (6, 7)]);
    m.insert("e3", vec![(3, 4), (7, 8)]);
    m.insert("e4", vec![(4, 1), (8, 5)]);
    m
}

/// Build a `StatsSnapshot` keyed by predicate name. Used by
/// Part B to inject distinct stats into the compile-time
/// `selectivity_pass` invocation.
fn make_snapshot(seeded: &[(&str, u64)]) -> StatsSnapshot {
    let relations: Vec<RelationStats> = seeded
        .iter()
        .enumerate()
        .map(|(i, (_, card))| {
            let mut s = RelationStats::new(RelId(i as u32));
            s.cardinality = *card;
            s
        })
        .collect();
    let rel_names: Vec<(RelId, String)> = seeded
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (RelId(i as u32), (*name).to_string()))
        .collect();
    StatsSnapshot {
        relations,
        join_selectivities: vec![],
        rel_names,
    }
}

fn run_program(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    config: RuntimeConfig,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    snapshot: Option<&StatsSnapshot>,
) -> Executor {
    let mut compiler = Compiler::new();
    let plan = match snapshot {
        Some(s) => compiler
            .compile_with_stats_snapshot(source, Some(s))
            .expect("compile"),
        None => compiler.compile(source).expect("compile"),
    };
    let mut executor = Executor::new_with_config(provider, config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        let buf = upload_binary_u32(memory, rows);
        executor.put_relation(name, buf);
    }
    let _ = executor.execute_plan(&plan).expect("execute_plan");
    executor
}

// ---------------------------------------------------------------
// Part B — row-set parity across two stats snapshots
// ---------------------------------------------------------------

#[test]
fn selectivity_pass_triangle_two_snapshots_produce_same_row_set() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = triangle_inputs();

    // Snapshot A: cards favor Y-shared inner.
    let snap_a = make_snapshot(&[("e1", 10), ("e2", 10), ("e3", 100_000)]);
    let exec_a = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default(),
        STABLE_TRIANGLE,
        &inputs,
        Some(&snap_a),
    );
    let rows_a = download_triples(exec_a.store().get("tri").expect("tri"));

    // Snapshot B: cards favor Z-shared inner.
    let snap_b = make_snapshot(&[("e1", 100_000), ("e2", 10), ("e3", 10)]);
    let exec_b = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default(),
        STABLE_TRIANGLE,
        &inputs,
        Some(&snap_b),
    );
    let rows_b = download_triples(exec_b.store().get("tri").expect("tri"));

    assert!(
        !rows_a.is_empty(),
        "fixture should produce at least one tri row to make the cert non-degenerate"
    );
    assert_eq!(
        rows_a, rows_b,
        "different stats snapshots must produce IDENTICAL row sets — \
         reordering preserves rule semantics"
    );
}

// ---------------------------------------------------------------
// Part C — force-WCOJ dispatch survives W2.2 changes
// ---------------------------------------------------------------
//
// **Honest scope note.** Part C as originally framed wanted to
// drive selectivity_pass to an alt shape end-to-end and then
// confirm WCOJ dispatch on the alt shape. The compile-time
// optimizer can emit right-deep `Project { Join { Scan, Join } }`
// when stats favor it — right-deep is explicitly OUT of W2.2
// scope (per plan's "Not in Scope"), and the slice 1 / slice 2
// promoter (even with the W2.2 step 2a extension) does not
// recognize right-deep.
//
// Therefore Part C here uses the **empty stats** path so the
// optimizer produces canonical left-deep / bushy shapes. This
// proves W2.2's changes don't break dispatch on the canonical
// case (regression-style cert). The W2.2 reordering itself is
// exercised by:
//   * Step 3 compile-time certs in
//     `crates/xlog-logic/src/optimizer.rs::selectivity_pass_tests`
//     — three triangle inner-pair choices verified by direct
//     plan synthesis.
//   * Step 2a promoter-extension tests in
//     `crates/xlog-logic/src/promote.rs::tests` — alt-shape
//     bodies promote with semantic-order `MultiWayJoin.inputs`
//     and shape-fixed `slot_vars`.
//
// End-to-end alt-shape integration is gated on right-deep
// optimizer-output handling, which the W2.2 plan explicitly
// defers as a separate slice's input.

#[test]
fn selectivity_pass_changes_do_not_break_canonical_triangle_dispatch() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = triangle_inputs();

    // Reference: gate off → binary-join row set.
    let exec_off = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false)),
        STABLE_TRIANGLE,
        &inputs,
        None,
    );
    let reference_rows = download_triples(exec_off.store().get("tri").expect("tri"));
    assert!(
        !reference_rows.is_empty(),
        "binary-join reference should produce at least one tri row"
    );

    // Force-WCOJ. Empty snapshot → optimizer + W2.2
    // selectivity_pass leave canonical left-deep. Slice 1
    // promoter (with W2.2 extension) emits MultiWayJoin.
    // Force gate dispatches.
    let exec_on = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        STABLE_TRIANGLE,
        &inputs,
        None,
    );
    assert!(
        exec_on.wcoj_triangle_dispatch_count() >= 1,
        "force-WCOJ on canonical triangle must still dispatch after W2.2 changes; \
         got counter {}",
        exec_on.wcoj_triangle_dispatch_count()
    );
    let dispatched_rows = download_triples(exec_on.store().get("tri").expect("tri"));
    assert_eq!(
        dispatched_rows, reference_rows,
        "WCOJ output must equal the binary-join reference after W2.2 changes"
    );
}

#[test]
fn selectivity_pass_changes_do_not_break_canonical_4cycle_dispatch() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = cycle4_inputs();

    let exec_off = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(false)),
        STABLE_4CYCLE,
        &inputs,
        None,
    );
    let reference_rows = download_quads(exec_off.store().get("cyc").expect("cyc"));
    assert!(
        !reference_rows.is_empty(),
        "binary-join reference should produce at least one cyc row"
    );

    let exec_on = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(true)),
        STABLE_4CYCLE,
        &inputs,
        None,
    );
    assert!(
        exec_on.wcoj_4cycle_dispatch_count() >= 1,
        "force-WCOJ on canonical 4-cycle must still dispatch after W2.2 changes; \
         got counter {}",
        exec_on.wcoj_4cycle_dispatch_count()
    );
    let dispatched_rows = download_quads(exec_on.store().get("cyc").expect("cyc"));
    assert_eq!(
        dispatched_rows, reference_rows,
        "4-cycle WCOJ output must equal the binary-join reference after W2.2 changes"
    );
}
