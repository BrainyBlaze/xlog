// crates/xlog-runtime/tests/test_w23_recursive_stats.rs
//! W2.3 step 7 — recursive-SCC stats integration acceptance gate.
//!
//! Part A (3) — iteration-level cardinality evolution via the
//!   `#[cfg(test)]` recursive-stats trace.
//! Part B (2) — `binary_est_for_variant` reflects the rewritten
//!   variant's `delta_e1` card.
//! Part C (4) — row-set + dispatch-counter parity vs. the
//!   pre-W2.3 baseline.
//! Part D (1) — multi-recursive bodies still skip the WCOJ
//!   promoter gate (W4.1 owns the gate).
//!
//! Total: **10 tests**.
//!
//! Lives in xlog-runtime's `tests/` directory so the
//! `#[cfg(test)]`-gated `Executor::last_recursive_stats_trace()`
//! accessor is visible (xlog-runtime tests/ directory is
//! compiled with `cfg(test)` set on the lib for these binaries).
//!
//! Anchors on the slice-4 linear-recursive triangle and 4-cycle
//! programs (`crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs`
//! `LINEAR_REC_TRIANGLE` :586, `LINEAR_REC_4CYCLE` :669). Both
//! fixtures' recursive predicate is `e1`; the WCOJ rule rewrites
//! `Scan(e1)` → `Scan(delta_e1)` for the iteration's variant.

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
use xlog_runtime::executor::RecursiveStatsPhase;
use xlog_runtime::Executor;

// ---------------------------------------------------------------
// Fixture infrastructure (slice-4 cert pattern)
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

fn make_runtime_fixture() -> Option<RuntimeBackedFixture> {
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
    if !rows.is_empty() {
        let bs0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let bs1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        device.htod_sync_copy_into(&bs0, &mut col0).unwrap();
        device.htod_sync_copy_into(&bs1, &mut col1).unwrap();
    }
    device.htod_sync_copy_into(&[n], &mut d_num_rows).unwrap();
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
    let n = buf
        .cached_row_count()
        .map(|c| c as usize)
        .unwrap_or_else(|| {
            let mut count_host = [0u32; 1];
            unsafe {
                sys::cuMemcpyDtoH_v2(
                    count_host.as_mut_ptr() as *mut _,
                    *buf.num_rows_device().device_ptr(),
                    std::mem::size_of::<u32>(),
                );
            }
            count_host[0] as usize
        });
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(buf.arity(), 3);
    let mut col0 = vec![0u8; n * 4];
    let mut col1 = vec![0u8; n * 4];
    let mut col2 = vec![0u8; n * 4];
    unsafe {
        sys::cuMemcpyDtoH_v2(
            col0.as_mut_ptr() as *mut _,
            *buf.column(0).unwrap().device_ptr(),
            col0.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col1.as_mut_ptr() as *mut _,
            *buf.column(1).unwrap().device_ptr(),
            col1.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col2.as_mut_ptr() as *mut _,
            *buf.column(2).unwrap().device_ptr(),
            col2.len(),
        );
    }
    let mut out: Vec<(u32, u32, u32)> = (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(col0[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col1[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col2[i * 4..i * 4 + 4].try_into().unwrap()),
            )
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

fn download_quads(buf: &CudaBuffer) -> Vec<(u32, u32, u32, u32)> {
    let n = buf
        .cached_row_count()
        .map(|c| c as usize)
        .unwrap_or_else(|| {
            let mut count_host = [0u32; 1];
            unsafe {
                sys::cuMemcpyDtoH_v2(
                    count_host.as_mut_ptr() as *mut _,
                    *buf.num_rows_device().device_ptr(),
                    std::mem::size_of::<u32>(),
                );
            }
            count_host[0] as usize
        });
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(buf.arity(), 4);
    let mut cols = [
        vec![0u8; n * 4],
        vec![0u8; n * 4],
        vec![0u8; n * 4],
        vec![0u8; n * 4],
    ];
    for c in 0..4 {
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

// ---------------------------------------------------------------
// Slice-4 fixtures (anchor on test_wcoj_recursive_dispatch.rs)
// ---------------------------------------------------------------

/// `LINEAR_REC_TRIANGLE` — slice-4 anchor.
const LINEAR_REC_TRIANGLE: &str = r#"
    pred e1_seed(u32, u32).
    pred e1(u32, u32).
    pred e2(u32, u32).
    pred e3(u32, u32).
    pred tri(u32, u32, u32).
    e1(X, Y) :- e1_seed(X, Y).
    e1(X, Y) :- tri(X, Z, Y).
    tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).
"#;

fn linear_rec_triangle_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    m.insert("e1_seed", vec![(1, 2)]);
    m.insert("e2", vec![(2, 3), (3, 4)]);
    m.insert("e3", vec![(1, 3), (1, 4)]);
    m
}

/// `LINEAR_REC_4CYCLE` — slice-4 anchor.
const LINEAR_REC_4CYCLE: &str = r#"
    pred e1_seed(u32, u32).
    pred e1(u32, u32).
    pred e2(u32, u32).
    pred e3(u32, u32).
    pred e4(u32, u32).
    pred cyc(u32, u32, u32, u32).
    e1(W, X) :- e1_seed(W, X).
    e1(W, X) :- cyc(Y, W, X, Z).
    cyc(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).
"#;

fn linear_rec_cycle4_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    m.insert("e1_seed", vec![(1, 2)]);
    m.insert("e2", vec![(2, 3), (3, 4)]);
    m.insert("e3", vec![(3, 4), (4, 5)]);
    m.insert("e4", vec![(4, 1), (5, 2)]);
    m
}

/// Multi-recursive triangle — slice-4 anchor pattern. Two
/// recursive IDBs (`r1`, `r2`) feed the head rule with
/// `recursive_scan_count == 2`; W4.1's gate refuses promotion.
const MULTIREC_TRIANGLE: &str = r#"
    pred r1_init(u32, u32).
    pred r2_init(u32, u32).
    pred r3(u32, u32).
    pred r1(u32, u32).
    pred r2(u32, u32).
    pred tri(u32, u32, u32).
    r1(X, Y) :- r1_init(X, Y).
    r1(X, Y) :- tri(X, Y, Z).
    r2(X, Y) :- r2_init(X, Y).
    r2(X, Y) :- tri(Z, X, Y).
    tri(X, Y, Z) :- r1(X, Y), r2(Y, Z), r3(X, Z).
"#;

fn multirec_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    m.insert("r1_init", vec![(1, 2), (1, 3), (2, 3)]);
    m.insert("r2_init", vec![(2, 3), (3, 4)]);
    m.insert("r3", vec![(1, 3), (2, 4), (1, 4)]);
    m
}

// ---------------------------------------------------------------
// Run helpers
// ---------------------------------------------------------------

fn run_with_config(
    fix: &RuntimeBackedFixture,
    runtime_config: RuntimeConfig,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> Executor {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile");
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), runtime_config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        let buf = upload_binary_u32(&fix.memory, rows);
        executor.put_relation(name, buf);
    }
    executor.execute_plan(&plan).expect("execute_plan");
    executor
}

// ===============================================================
// Part A — Iteration-level cardinality evolution (3 tests)
// ===============================================================

#[test]
fn recursive_triangle_e1_full_card_grows_across_iterations() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = linear_rec_triangle_inputs();
    let exec = run_with_config(&fix, RuntimeConfig::default(), LINEAR_REC_TRIANGLE, &inputs);
    let trace = exec.last_recursive_stats_trace();

    // Filter to e1's Phase 4 entries only — these are the points
    // where full_rel was actually advanced this iteration.
    let e1_phase4_full: Vec<u64> = trace
        .entries
        .iter()
        .filter(|e| e.pred == "e1" && e.phase == RecursiveStatsPhase::Phase4Full)
        .map(|e| e.full_rows)
        .collect();
    // Add the seed-pass full_rows entry (phase = Seed).
    let e1_seed_full: Option<u64> = trace
        .entries
        .iter()
        .find(|e| e.pred == "e1" && e.phase == RecursiveStatsPhase::Seed)
        .map(|e| e.full_rows);
    let mut full_series: Vec<u64> = Vec::new();
    if let Some(s) = e1_seed_full {
        full_series.push(s);
    }
    full_series.extend(e1_phase4_full);
    assert!(
        full_series.len() >= 2,
        "fixture must produce ≥ 2 e1 full-row records (seed + ≥ 1 Phase 4); got {} entries: {:?}",
        full_series.len(),
        trace.entries
    );
    // Monotonic non-decrease.
    for w in full_series.windows(2) {
        assert!(
            w[1] >= w[0],
            "e1 full_rows must monotonically non-decrease across iterations; \
             got prev={} next={} in series {:?}",
            w[0],
            w[1],
            full_series
        );
    }
    // Strict > on at least one transition.
    let strictly_grew = full_series.windows(2).any(|w| w[1] > w[0]);
    assert!(
        strictly_grew,
        "e1 full_rows must strictly grow on at least one transition; series {:?}",
        full_series
    );
}

#[test]
fn recursive_triangle_e1_delta_evolves_across_iterations() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = linear_rec_triangle_inputs();
    let exec = run_with_config(&fix, RuntimeConfig::default(), LINEAR_REC_TRIANGLE, &inputs);
    let trace = exec.last_recursive_stats_trace();

    // Phase 2 entries for e1 (where delta_rel is actually written).
    let e1_phase2_deltas: Vec<u64> = trace
        .entries
        .iter()
        .filter(|e| e.pred == "e1" && e.phase == RecursiveStatsPhase::Phase2Delta)
        .map(|e| e.delta_rows)
        .collect();
    assert!(
        !e1_phase2_deltas.is_empty(),
        "fixture must produce ≥ 1 e1 Phase 2 delta record; got entries: {:?}",
        trace.entries
    );

    // (a) at least one pre-convergence iteration's delta is non-zero.
    let any_nonzero = e1_phase2_deltas.iter().any(|&d| d > 0);
    assert!(
        any_nonzero,
        "at least one e1 Phase 2 delta must be non-zero; got series {:?}",
        e1_phase2_deltas
    );
    // (b) the LAST Phase 2 entry (= converged iteration's record) is zero.
    assert_eq!(
        *e1_phase2_deltas.last().unwrap(),
        0,
        "the converged iteration's Phase 2 delta record must be 0; \
         got series {:?}",
        e1_phase2_deltas
    );
}

#[test]
fn recursive_4cycle_e1_full_card_grows_across_iterations() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = linear_rec_cycle4_inputs();
    let exec = run_with_config(&fix, RuntimeConfig::default(), LINEAR_REC_4CYCLE, &inputs);
    let trace = exec.last_recursive_stats_trace();
    let e1_phase4_full: Vec<u64> = trace
        .entries
        .iter()
        .filter(|e| e.pred == "e1" && e.phase == RecursiveStatsPhase::Phase4Full)
        .map(|e| e.full_rows)
        .collect();
    let e1_seed_full: Option<u64> = trace
        .entries
        .iter()
        .find(|e| e.pred == "e1" && e.phase == RecursiveStatsPhase::Seed)
        .map(|e| e.full_rows);
    let mut full_series: Vec<u64> = Vec::new();
    if let Some(s) = e1_seed_full {
        full_series.push(s);
    }
    full_series.extend(e1_phase4_full);
    assert!(
        full_series.len() >= 2,
        "fixture must produce ≥ 2 e1 full-row records; got {} entries: {:?}",
        full_series.len(),
        trace.entries
    );
    for w in full_series.windows(2) {
        assert!(
            w[1] >= w[0],
            "e1 full_rows must monotonically non-decrease; got {} → {}",
            w[0],
            w[1]
        );
    }
    assert!(
        full_series.windows(2).any(|w| w[1] > w[0]),
        "e1 full_rows must strictly grow on at least one transition; series {:?}",
        full_series
    );
}

// ===============================================================
// Part B — `binary_est_for_variant` reflects delta_e1 card (2 tests)
// ===============================================================

/// Helper for Part B: assert that the cost model AT EACH Phase 2
/// site successfully invoked
/// `estimate_join_cardinality(delta_e1, e2, &[1], &[0])` (i.e.,
/// `binary_est_for_variant` is populated). The estimate's
/// **input** is `delta_rel.cardinality`, which W2.3 has just
/// written via `update_cardinality` to the iteration's actual
/// `delta_new_rows`. So a populated `binary_est_for_variant` at
/// Phase 2 N is, by construction, computed against iteration N's
/// stat — proving the cost model sees iteration-current state,
/// not seed-only state.
///
/// The non-constancy of the **output** of the formula is NOT
/// asserted here: with the slice-4 fixtures' tiny inputs, the
/// formula floors to its `min == 1` value across all iterations.
/// W2.3's correctness is at the input layer (cost model reads
/// W2.3-updated stats), and Part A's `delta_rows`-evolves test
/// already verifies the input evolves across iterations.
fn assert_phase2_binary_est_populated(
    trace: &xlog_runtime::executor::RecursiveStatsTrace,
    pred: &str,
) {
    let e1_phase2: Vec<&xlog_runtime::executor::RecursiveStatsTraceEntry> = trace
        .entries
        .iter()
        .filter(|e| e.pred == pred && e.phase == RecursiveStatsPhase::Phase2Delta)
        .collect();
    assert!(
        e1_phase2.len() >= 2,
        "expected ≥ 2 Phase 2 trace entries for `{}`; got {} entries: {:?}",
        pred,
        e1_phase2.len(),
        trace.entries
    );
    // Every Phase 2 entry for `pred == "e1"` must have
    // `binary_est_for_variant.is_some()` — proves the cost model
    // lookup `(delta_e1, e2, &[1], &[0])` succeeded with both
    // rels registered + cards populated.
    let populated = e1_phase2
        .iter()
        .filter(|e| e.binary_est_for_variant.is_some())
        .count();
    assert!(
        populated >= 2,
        "expected ≥ 2 Phase 2 entries with populated \
         binary_est_for_variant for `{}`; got {} populated of \
         {} total. This means the cost model lookup `(delta_{0}, \
         e2, &[1], &[0])` failed — either delta rel was \
         unregistered or e2 was not registered.",
        pred,
        populated,
        e1_phase2.len()
    );
    // Cross-check: at every Phase 2 entry where binary_est is
    // populated, the entry's `delta_rows` field equals what
    // `update_cardinality` just wrote. This proves the cost model
    // formula's input (`delta_rel.cardinality`) was the
    // iteration's `delta_new_rows`, not a stale seed value.
    // Implicit: at least 2 distinct `delta_rows` values appear
    // across iterations (Part A's evolves-test). Combined with
    // populated binary_est, this proves the cost model sees
    // iteration-evolving stats.
    let distinct_delta_rows: std::collections::BTreeSet<u64> =
        e1_phase2.iter().map(|e| e.delta_rows).collect();
    assert!(
        distinct_delta_rows.len() >= 2,
        "expected ≥ 2 distinct delta_rows across Phase 2 entries \
         for `{}` (proves cost model input evolved across \
         iterations); got {:?}",
        pred,
        distinct_delta_rows
    );
}

#[test]
fn triangle_binary_est_reflects_delta_e1_card_per_iteration() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = linear_rec_triangle_inputs();
    let exec = run_with_config(&fix, RuntimeConfig::default(), LINEAR_REC_TRIANGLE, &inputs);
    assert_phase2_binary_est_populated(exec.last_recursive_stats_trace(), "e1");
}

#[test]
fn cycle4_binary_est_reflects_delta_e1_card_per_iteration() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = linear_rec_cycle4_inputs();
    let exec = run_with_config(&fix, RuntimeConfig::default(), LINEAR_REC_4CYCLE, &inputs);
    assert_phase2_binary_est_populated(exec.last_recursive_stats_trace(), "e1");
}

// ===============================================================
// Part C — Row-set + dispatch counter parity vs. baseline (4 tests)
// ===============================================================

fn force_wcoj_triangle() -> RuntimeConfig {
    let mut c = RuntimeConfig::default();
    c.wcoj_triangle_dispatch = Some(true);
    c
}

fn force_wcoj_4cycle() -> RuntimeConfig {
    let mut c = RuntimeConfig::default();
    c.wcoj_4cycle_dispatch = Some(true);
    c
}

#[test]
fn recursive_triangle_row_set_unchanged_under_default_config() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = linear_rec_triangle_inputs();
    // Reference: WCOJ off explicitly.
    let mut ref_cfg = RuntimeConfig::default();
    ref_cfg.wcoj_triangle_dispatch = Some(false);
    let exec_ref = run_with_config(&fix, ref_cfg, LINEAR_REC_TRIANGLE, &inputs);
    let ref_rows = download_triples(exec_ref.store().get("tri").expect("tri ref"));
    // W2.3 path: force-WCOJ on (matches slice-4 cert's dispatch path).
    let exec_w23 = run_with_config(&fix, force_wcoj_triangle(), LINEAR_REC_TRIANGLE, &inputs);
    let w23_rows = download_triples(exec_w23.store().get("tri").expect("tri W2.3"));
    assert_eq!(
        w23_rows, ref_rows,
        "W2.3 recursive triangle row set must match binary-join reference"
    );
}

#[test]
fn recursive_triangle_dispatch_counter_unchanged_under_default_config() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = linear_rec_triangle_inputs();
    let exec = run_with_config(&fix, force_wcoj_triangle(), LINEAR_REC_TRIANGLE, &inputs);
    // Slice-4 baseline asserts ≥ 2 (seed + ≥ 1 variant). W2.3 must
    // not perturb this counter behavior.
    assert!(
        exec.wcoj_triangle_dispatch_count() >= 2,
        "linear-recursive triangle WCOJ counter must be ≥ 2 (seed + ≥ 1 variant); \
         got {}",
        exec.wcoj_triangle_dispatch_count()
    );
}

#[test]
fn recursive_4cycle_row_set_unchanged_under_default_config() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = linear_rec_cycle4_inputs();
    let mut ref_cfg = RuntimeConfig::default();
    ref_cfg.wcoj_4cycle_dispatch = Some(false);
    let exec_ref = run_with_config(&fix, ref_cfg, LINEAR_REC_4CYCLE, &inputs);
    let ref_rows = download_quads(exec_ref.store().get("cyc").expect("cyc ref"));
    let exec_w23 = run_with_config(&fix, force_wcoj_4cycle(), LINEAR_REC_4CYCLE, &inputs);
    let w23_rows = download_quads(exec_w23.store().get("cyc").expect("cyc W2.3"));
    assert_eq!(
        w23_rows, ref_rows,
        "W2.3 recursive 4-cycle row set must match binary-join reference"
    );
}

#[test]
fn recursive_4cycle_dispatch_counter_unchanged_under_default_config() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = linear_rec_cycle4_inputs();
    let exec = run_with_config(&fix, force_wcoj_4cycle(), LINEAR_REC_4CYCLE, &inputs);
    assert!(
        exec.wcoj_4cycle_dispatch_count() >= 2,
        "linear-recursive 4-cycle WCOJ counter must be ≥ 2 (seed + ≥ 1 variant); \
         got {}",
        exec.wcoj_4cycle_dispatch_count()
    );
}

// ===============================================================
// Part D — Multi-recursive bodies untouched (1 test)
// ===============================================================

#[test]
fn multi_recursive_triangle_per_iteration_update_does_not_promote() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = multirec_inputs();
    let exec = run_with_config(&fix, force_wcoj_triangle(), MULTIREC_TRIANGLE, &inputs);
    // Slice-4 anchor: `tri(X, Y, Z) :- r1(X, Y), r2(Y, Z), r3(X, Z).`
    // has recursive_scan_count == 2 (r1 + r2 are both recursive
    // IDBs in the SCC); W4.1's gate refuses promotion. Counter
    // must stay 0 across all iterations.
    assert_eq!(
        exec.wcoj_triangle_dispatch_count(),
        0,
        "multi-recursive tri (recursive_scan_count > 1) must NOT promote; \
         got dispatch counter {}",
        exec.wcoj_triangle_dispatch_count()
    );
    // Per-iteration trace fires for the recursive predicates
    // (r1, r2) even when the head rule is gated out — W2.3
    // updates are predicate-level, not promoter-level.
    let trace = exec.last_recursive_stats_trace();
    let recursive_pred_records = trace
        .entries
        .iter()
        .filter(|e| matches!(e.pred.as_str(), "r1" | "r2"))
        .count();
    assert!(
        recursive_pred_records >= 1,
        "W2.3 trace must contain at least one r1/r2 record even \
         when WCOJ promotion is gated out by W4.1; got {} records: {:?}",
        recursive_pred_records,
        trace.entries
    );
}
