//! D3 Phase B — factorized recursive delta: end-to-end executor wiring.
//!
//! Plan: `docs/plans/2026-06-12-d3-phase-b-plan.md`. Qualifying
//! recursive variants (ChainJoin over two Scans, one on the delta)
//! must dispatch the factorized novel-set pipeline
//! (`factorized_delta_dispatch_count >= 1`) and produce the same row
//! set as the legacy path (kill switch `XLOG_DISABLE_FACTORIZED_DELTA`
//! forces it; both phases run inside ONE test per fixture because the
//! switch is a process-global env var, guarded by ENV_LOCK).
//!
//! Decline coverage (counter stays 0, results stay correct): u64
//! schemas, 3-atom recursive bodies (no ChainJoin), domain over the
//! default cap. The long-chain fixture exercises the per-iteration
//! work floor (mixed factorized/legacy iterations inside one fixpoint).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::{CudaBuffer, CudaColumn};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct Fixture {
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
}

fn make_fixture_with_budget(budget_bytes: u64) -> Option<Fixture> {
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
        Box::new(GlobalDeviceBudget::new(logging, budget_bytes as usize));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(budget_bytes),
        runtime,
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(Fixture { memory, provider })
}

fn make_fixture() -> Option<Fixture> {
    make_fixture_with_budget(512 * 1024 * 1024)
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let mut col0 = memory
        .alloc::<u8>(rows.len() * std::mem::size_of::<u32>())
        .expect("alloc col0");
    let mut col1 = memory
        .alloc::<u8>(rows.len() * std::mem::size_of::<u32>())
        .expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let col0_bytes: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    let device = memory.device().inner();
    device
        .htod_sync_copy_into(&col0_bytes, &mut col0)
        .expect("htod col0");
    device
        .htod_sync_copy_into(&col1_bytes, &mut col1)
        .expect("htod col1");
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

fn upload_binary_u64(memory: &Arc<GpuMemoryManager>, rows: &[(u64, u64)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let mut col0 = memory
        .alloc::<u8>(rows.len() * std::mem::size_of::<u64>())
        .expect("alloc col0");
    let mut col1 = memory
        .alloc::<u8>(rows.len() * std::mem::size_of::<u64>())
        .expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let col0_bytes: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    let device = memory.device().inner();
    device
        .htod_sync_copy_into(&col0_bytes, &mut col0)
        .expect("htod col0");
    device
        .htod_sync_copy_into(&col1_bytes, &mut col1)
        .expect("htod col1");
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

fn buffer_rows(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> usize {
    if let Some(n) = buffer.cached_row_count() {
        return n as usize;
    }
    let mut host = [0u32; 1];
    memory
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host)
        .expect("dtoh num_rows");
    host[0] as usize
}

fn download_column_bytes(buffer: &CudaBuffer, col: usize, elem: usize, n: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; n * elem];
    if n == 0 {
        return bytes;
    }
    let CudaColumn::Owned(c) = buffer.column(col).expect("column") else {
        panic!("column must be owned");
    };
    unsafe {
        let res = sys::cuMemcpyDtoH_v2(bytes.as_mut_ptr() as *mut _, *c.device_ptr(), bytes.len());
        assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS, "dtoh column copy");
    }
    bytes
}

fn download_row_set_u32(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u32, u32)> {
    let n = buffer_rows(memory, buffer);
    let c0 = download_column_bytes(buffer, 0, 4, n);
    let c1 = download_column_bytes(buffer, 1, 4, n);
    let mut rows: Vec<(u32, u32)> = (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(c0[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(c1[i * 4..i * 4 + 4].try_into().unwrap()),
            )
        })
        .collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

fn download_row_set_u64(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u64, u64)> {
    let n = buffer_rows(memory, buffer);
    let c0 = download_column_bytes(buffer, 0, 8, n);
    let c1 = download_column_bytes(buffer, 1, 8, n);
    let mut rows: Vec<(u64, u64)> = (0..n)
        .map(|i| {
            (
                u64::from_le_bytes(c0[i * 8..i * 8 + 8].try_into().unwrap()),
                u64::from_le_bytes(c1[i * 8..i * 8 + 8].try_into().unwrap()),
            )
        })
        .collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// Compile + execute `source` with `edge` as the single EDB input;
/// return the executor for counter/store inspection.
fn run_program(fix: &Fixture, source: &str, edge: CudaBuffer) -> Executor {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile program");
    let mut executor = Executor::new(Arc::clone(&fix.provider));
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("edge", edge);
    executor.execute_plan(&plan).expect("execute plan");
    executor
}

/// CPU oracle: transitive closure (right-linear semantics — same set
/// for all TC formulations).
fn oracle_tc(edge: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut succ: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for &(a, b) in edge {
        succ.entry(a).or_default().insert(b);
    }
    let mut r: BTreeSet<(u32, u32)> = edge.iter().copied().collect();
    loop {
        let mut grew = false;
        let snapshot: Vec<(u32, u32)> = r.iter().copied().collect();
        for &(x, y) in &snapshot {
            if let Some(zs) = succ.get(&y) {
                for &z in zs {
                    grew |= r.insert((x, z));
                }
            }
        }
        if !grew {
            break;
        }
    }
    r.into_iter().collect()
}

/// The kill switch is a process-global env var: every test that
/// toggles it (or asserts the dispatch counter) takes this lock.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const KILL_SWITCH: &str = "XLOG_DISABLE_FACTORIZED_DELTA";

/// Irregular digraph (cycles, diamond, hub, dead end, self-loop).
fn irregular_edges() -> Vec<(u32, u32)> {
    vec![
        (0, 1),
        (1, 2),
        (2, 0),
        (3, 4),
        (3, 5),
        (4, 6),
        (5, 6),
        (6, 7),
        (8, 0),
        (8, 3),
        (8, 7),
        (8, 9),
        (9, 9),
        (11, 1),
    ]
}

/// Run `source` twice — dispatch ON then kill-switched — and assert
/// row-set parity plus the expected counter behavior.
fn assert_fires_with_parity(source: &str, edges: &[(u32, u32)], expect: Option<&[(u32, u32)]>) {
    let _guard = env_lock();
    std::env::remove_var(KILL_SWITCH);
    let Some(fix) = make_fixture() else {
        eprintln!("skipping: no CUDA device");
        return;
    };

    let executor = run_program(&fix, source, upload_binary_u32(&fix.memory, edges));
    let on_rows = download_row_set_u32(&fix.memory, executor.store().get("q").expect("q"));
    let fired = executor.factorized_delta_dispatch_count();
    assert!(
        fired >= 1,
        "factorized delta must dispatch at least once (got {fired})"
    );

    std::env::set_var(KILL_SWITCH, "1");
    let legacy = run_program(&fix, source, upload_binary_u32(&fix.memory, edges));
    let off_rows = download_row_set_u32(&fix.memory, legacy.store().get("q").expect("q"));
    assert_eq!(
        legacy.factorized_delta_dispatch_count(),
        0,
        "kill switch must force the legacy path"
    );
    std::env::remove_var(KILL_SWITCH);

    assert_eq!(on_rows, off_rows, "dispatch ON/OFF row sets must match");
    if let Some(expected) = expect {
        assert_eq!(on_rows, expected.to_vec(), "row set must match the oracle");
    }
}

#[test]
fn right_linear_tc_fires_with_kill_switch_parity() {
    let edges = irregular_edges();
    let expected = oracle_tc(&edges);
    assert_fires_with_parity(
        "pred edge(u32, u32).\n\
         pred q(u32, u32).\n\
         q(X, Y) :- edge(X, Y).\n\
         q(X, Z) :- q(X, Y), edge(Y, Z).",
        &edges,
        Some(&expected),
    );
}

#[test]
fn left_linear_tc_fires_with_kill_switch_parity() {
    let edges = irregular_edges();
    let expected = oracle_tc(&edges);
    assert_fires_with_parity(
        "pred edge(u32, u32).\n\
         pred q(u32, u32).\n\
         q(X, Y) :- edge(X, Y).\n\
         q(X, Z) :- edge(X, Y), q(Y, Z).",
        &edges,
        Some(&expected),
    );
}

#[test]
fn nonlinear_self_join_tc_fires_with_kill_switch_parity() {
    let edges = irregular_edges();
    let expected = oracle_tc(&edges);
    assert_fires_with_parity(
        "pred edge(u32, u32).\n\
         pred q(u32, u32).\n\
         q(X, Y) :- edge(X, Y).\n\
         q(X, Z) :- q(X, Y), q(Y, Z).",
        &edges,
        Some(&expected),
    );
}

#[test]
fn swapped_head_fires_with_kill_switch_parity() {
    // Head reverses the (carry, value) order — the dispatcher must
    // place columns per the head projection. Not a TC, so parity vs
    // the kill-switched legacy run is the oracle.
    let edges = irregular_edges();
    assert_fires_with_parity(
        "pred edge(u32, u32).\n\
         pred q(u32, u32).\n\
         q(Y, X) :- edge(X, Y).\n\
         q(Z, X) :- q(Y, X), edge(Y, Z).",
        &edges,
        None,
    );
}

/// Long path graph: late iterations have tiny deltas, so the
/// per-iteration work floor must bail to the legacy path while early
/// iterations may dispatch — the mixed fixpoint must stay exact.
#[test]
fn long_chain_mixed_iterations_stay_exact() {
    let edges: Vec<(u32, u32)> = (0..200u32).map(|i| (i, i + 1)).collect();
    let expected = oracle_tc(&edges);
    assert_fires_with_parity(
        "pred edge(u32, u32).\n\
         pred q(u32, u32).\n\
         q(X, Y) :- edge(X, Y).\n\
         q(X, Z) :- q(X, Y), edge(Y, Z).",
        &edges,
        Some(&expected),
    );
}

#[test]
fn u64_tc_declines_silently() {
    let _guard = env_lock();
    std::env::remove_var(KILL_SWITCH);
    let Some(fix) = make_fixture() else {
        eprintln!("skipping u64 decline: no CUDA device");
        return;
    };
    const HI: u64 = 1u64 << 32;
    let edges: Vec<(u64, u64)> = vec![(HI, HI + 1), (HI + 1, HI + 2), (HI + 2, HI)];
    let source = "pred edge(u64, u64).\n\
                  pred q(u64, u64).\n\
                  q(X, Y) :- edge(X, Y).\n\
                  q(X, Z) :- q(X, Y), edge(Y, Z).";
    let executor = run_program(&fix, source, upload_binary_u64(&fix.memory, &edges));
    assert_eq!(
        executor.factorized_delta_dispatch_count(),
        0,
        "u64 width must decline the factorized path"
    );
    let rows = download_row_set_u64(&fix.memory, executor.store().get("q").expect("q"));
    // 3-cycle closure: all 9 ordered pairs.
    assert_eq!(
        rows.len(),
        9,
        "u64 TC must still be exact via the legacy path"
    );
}

#[test]
fn three_atom_recursive_body_declines_silently() {
    let _guard = env_lock();
    std::env::remove_var(KILL_SWITCH);
    let Some(fix) = make_fixture() else {
        eprintln!("skipping 3-atom decline: no CUDA device");
        return;
    };
    // Two static hops per step — no ChainJoin, so the factorized path
    // must not fire (the Free Join or binary walker handles it).
    let edges = irregular_edges();
    let source = "pred edge(u32, u32).\n\
                  pred q(u32, u32).\n\
                  q(X, Y) :- edge(X, Y).\n\
                  q(X, W) :- q(X, Y), edge(Y, Z), edge(Z, W).";
    let executor = run_program(&fix, source, upload_binary_u32(&fix.memory, &edges));
    assert_eq!(
        executor.factorized_delta_dispatch_count(),
        0,
        "3-atom recursive bodies must decline the factorized path"
    );
    assert!(!download_row_set_u32(&fix.memory, executor.store().get("q").expect("q")).is_empty());
}

const MAX_TABLE_BYTES: &str = "XLOG_FACTORIZED_DELTA_MAX_TABLE_BYTES";

#[test]
fn sparse_table_over_budget_declines_to_legacy() {
    let _guard = env_lock();
    std::env::remove_var(KILL_SWITCH);
    // Pin the sparse table ceiling to a tiny value so the route's
    // conservative table cannot fit: the dispatcher must decline to the
    // legacy path (counter 0) and still produce the exact closure.
    // Domain ~2^15 > the dense cap, so dense never applies — this
    // isolates the sparse budget-decline boundary.
    std::env::set_var(MAX_TABLE_BYTES, "256");
    let Some(fix) = make_fixture() else {
        std::env::remove_var(MAX_TABLE_BYTES);
        eprintln!("skipping sparse budget decline: no CUDA device");
        return;
    };
    let edges = large_id_tc_edges();
    let source = "pred edge(u32, u32).\n\
                  pred q(u32, u32).\n\
                  q(X, Y) :- edge(X, Y).\n\
                  q(X, Z) :- q(X, Y), edge(Y, Z).";
    let executor = run_program(&fix, source, upload_binary_u32(&fix.memory, &edges));
    let declined = executor.factorized_delta_dispatch_count();
    let on_rows = download_row_set_u32(&fix.memory, executor.store().get("q").expect("q"));
    std::env::remove_var(MAX_TABLE_BYTES);

    assert_eq!(
        declined, 0,
        "a sparse table over the byte ceiling must decline to the legacy path"
    );
    let expected = oracle_tc(&edges);
    assert_eq!(
        on_rows, expected,
        "decline path must still match the oracle closure"
    );
    assert!(
        !on_rows.is_empty(),
        "fixture must produce a non-empty closure"
    );
}

// ---------------------------------------------------------------------------
// Phase B production-dispatch bench guard (#[ignore], RunPod only).
//
// Distinct from the S3 spike-loop gate: this drives the PRODUCTION
// executor on the TC program with the factorized dispatch ON vs the
// kill switch ON (legacy hash-join -> diff), measuring peak bytes and
// wall-clock. Two fixtures:
//   * dense block-cycle (factorized must win — same physics as S3);
//   * sparse long path chain (the per-iteration work floor must bail,
//     so ON must NOT regress vs OFF beyond 1.2x).

use std::time::Instant;

fn block_cycle_edges(k: u32, b: u32) -> Vec<(u32, u32)> {
    let mut edges = Vec::with_capacity((k * b * b) as usize);
    for i in 0..k {
        let src = i * b;
        let dst = ((i + 1) % k) * b;
        for u in 0..b {
            for v in 0..b {
                edges.push((src + u, dst + v));
            }
        }
    }
    edges
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

const TC_SOURCE: &str = "pred edge(u32, u32).\n\
                         pred q(u32, u32).\n\
                         q(X, Y) :- edge(X, Y).\n\
                         q(X, Z) :- q(X, Y), edge(Y, Z).";

/// One engine run on `edges`, returning (wall_ms, peak_bytes,
/// dispatch_count, row_count). Fresh fixture each call so the peak is
/// attributable to this run alone.
fn engine_run(
    edges: &[(u32, u32)],
    budget_bytes: u64,
    factorized_on: bool,
) -> (f64, u64, u64, usize) {
    let fix = make_fixture_with_budget(budget_bytes).expect("CUDA fixture");
    if factorized_on {
        std::env::remove_var(KILL_SWITCH);
    } else {
        std::env::set_var(KILL_SWITCH, "1");
    }
    let edge_buf = upload_binary_u32(&fix.memory, edges);
    fix.memory.reset_peak();
    let t0 = Instant::now();
    let executor = run_program(&fix, TC_SOURCE, edge_buf);
    let dt = t0.elapsed().as_secs_f64() * 1000.0;
    let peak = fix.memory.peak_bytes();
    let q = executor.store().get("q").expect("q");
    let rows = buffer_rows(&fix.memory, q);
    (dt, peak, executor.factorized_delta_dispatch_count(), rows)
}

/// A/B bench: legacy (kill switch) vs factorized-default. Variants are
/// INTERLEAVED per rep (legacy, then factorized, repeat) so any
/// monotonic drift — thermal throttling, fragmentation — lands on both
/// arms equally instead of mapping onto the A/B axis. Both arms share
/// `budget_bytes`; the dense fixture needs the legacy path's full peak
/// headroom or its OFF run OOMs before timing.
fn bench_guard(name: &str, edges: &[(u32, u32)], budget_bytes: u64, expect_dispatch: bool) {
    let _guard = env_lock();
    if make_fixture_with_budget(budget_bytes).is_none() {
        eprintln!("skipping {name}: no CUDA device");
        return;
    }
    const REPS: usize = 3;
    let mut off_ms = Vec::new();
    let mut off_peak = Vec::new();
    let mut on_ms = Vec::new();
    let mut on_peak = Vec::new();
    let mut on_dispatch = 0u64;
    let mut off_rows = 0usize;
    let mut on_rows = 0usize;

    // One warm-up of each arm (discarded) to page in PTX / JIT and
    // reach steady GPU clocks before timed reps.
    let _ = engine_run(edges, budget_bytes, false);
    let _ = engine_run(edges, budget_bytes, true);

    for rep in 0..REPS {
        let (off_dt, off_pk, _, off_r) = engine_run(edges, budget_bytes, false);
        let (on_dt, on_pk, disp, on_r) = engine_run(edges, budget_bytes, true);
        off_ms.push(off_dt);
        off_peak.push(off_pk as f64);
        on_ms.push(on_dt);
        on_peak.push(on_pk as f64);
        on_dispatch = disp;
        off_rows = off_r;
        on_rows = on_r;
        eprintln!(
            "S4 {name} rep {rep}: legacy {off_dt:.1} ms / {:.1} MiB ; \
             factorized {on_dt:.1} ms / {:.1} MiB (dispatch={disp})",
            off_pk as f64 / (1024.0 * 1024.0),
            on_pk as f64 / (1024.0 * 1024.0),
        );
    }
    std::env::remove_var(KILL_SWITCH);

    assert_eq!(on_rows, off_rows, "{name}: ON/OFF row counts must match");
    let om = median(&mut off_ms);
    let opk = median(&mut off_peak);
    let nm = median(&mut on_ms);
    let npk = median(&mut on_peak);
    eprintln!(
        "S4 bench {name}: |E|={} rows={on_rows} dispatch_on={on_dispatch} | \
         legacy {om:.1} ms / {:.1} MiB ; factorized {nm:.1} ms / {:.1} MiB | \
         peak {:.2}x  wall-clock {:.3}x",
        edges.len(),
        opk / (1024.0 * 1024.0),
        npk / (1024.0 * 1024.0),
        opk / npk.max(1.0),
        nm / om.max(1.0),
    );
    if expect_dispatch {
        assert!(on_dispatch >= 1, "{name}: factorized path must fire");
        // The dense path must WIN (it is the whole point of D3).
        assert!(
            npk * 5.0 <= opk,
            "{name}: factorized must cut peak >=5x (peak {:.1} vs {:.1} MiB)",
            npk / (1024.0 * 1024.0),
            opk / (1024.0 * 1024.0)
        );
        assert!(
            nm <= om * 1.2,
            "{name}: factorized must not regress wall-clock (got {nm:.1} vs {om:.1} ms)"
        );
    } else {
        assert_eq!(on_dispatch, 0, "{name}: work floor must bail (no dispatch)");
        // No-regression bar for the sparse path the engine must NOT
        // route factorized.
        assert!(
            nm <= om * 1.2,
            "{name}: factorized-default must not regress sparse wall-clock beyond 1.2x \
             (factorized {nm:.1} ms vs legacy {om:.1} ms)"
        );
    }
}

/// Dense block-cycle — factorized dispatch must fire and win. The
/// legacy arm peaks ~2.3 GiB (S3 evidence), so the shared budget must
/// clear that or the OFF run OOMs before timing.
#[test]
#[ignore = "S4 bench guard — run on RunPod, never locally"]
fn s4_bench_dense_block_cycle() {
    bench_guard(
        "dense",
        &block_cycle_edges(4, 256),
        10 * 1024 * 1024 * 1024,
        true,
    );
}

/// Sparse long path chain (1500 nodes) — late iterations have tiny
/// deltas; the work floor must bail so ON never regresses vs OFF.
#[test]
#[ignore = "S4 bench guard — run on RunPod, never locally"]
fn s4_bench_sparse_long_chain() {
    let edges: Vec<(u32, u32)> = (0..1500u32).map(|i| (i, i + 1)).collect();
    bench_guard("sparse-chain", &edges, 512 * 1024 * 1024, false);
}

// ---------------------------------------------------------------------------
// D3 sparse-domain route (Phase: production integration). When the
// domain exceeds the dense cap (default 2^14), the dispatcher routes
// the factorized delta through the hash-set path instead of declining.

/// TC over a large-id graph: ids are the irregular fixture shifted into
/// a band starting at 2^15, so the domain exceeds the dense cap and the
/// factorized delta can only fire via the sparse route.
fn large_id_tc_edges() -> Vec<(u32, u32)> {
    const BASE: u32 = 1 << 15;
    irregular_edges()
        .into_iter()
        .map(|(a, b)| (BASE + a, BASE + b))
        .collect()
}

#[test]
fn large_id_tc_fires_via_sparse_route_with_kill_switch_parity() {
    let edges = large_id_tc_edges();
    let expected = oracle_tc(&edges);
    // Domain ~2^15 > dense cap 2^14 ⇒ any factorized dispatch is the
    // sparse route. assert_fires_with_parity checks counter>=1 ON,
    // counter==0 kill-switched, and row-set parity + oracle.
    assert_fires_with_parity(
        "pred edge(u32, u32).\n\
         pred q(u32, u32).\n\
         q(X, Y) :- edge(X, Y).\n\
         q(X, Z) :- q(X, Y), edge(Y, Z).",
        &edges,
        Some(&expected),
    );
}

/// Large-domain block-cycle: the dense block-cycle remapped by a stride
/// so the id domain spreads past 2^16 (dense bitvector infeasible),
/// preserving the per-iteration witness blowup. Routes sparse.
fn sparse_blowup_edges(k: u32, b: u32, stride: u32) -> Vec<(u32, u32)> {
    block_cycle_edges(k, b)
        .into_iter()
        .map(|(a, c)| (a * stride, c * stride))
        .collect()
}

/// Full-fixpoint S4-equivalent bench for the sparse route: production
/// executor ON vs kill-switched, interleaved per rep (the S4
/// methodology). Dense bitvector is infeasible at this domain, so the
/// comparison is sparse-hash-set vs legacy hash-join → diff.
#[test]
#[ignore = "S4-sparse bench guard — run on RunPod, never locally"]
fn s4_bench_sparse_domain_blowup() {
    // block-cycle k=4 b=128 (=512 nodes, |TC|=262144) remapped by
    // stride 4096 ⇒ ids up to ~2.09M, domain ≫ 2^16. Witness blowup
    // preserved (b duplicate witnesses per novel pair).
    let edges = sparse_blowup_edges(4, 128, 4096);
    let n = 4u32 * 128;
    let expected_rows = (n as usize) * (n as usize);
    let _guard = env_lock();
    if make_fixture_with_budget(12 << 30).is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let budget = 12u64 << 30;
    eprintln!(
        "S4-sparse fixture: nodes={n} |E|={} stride=4096 expected|TC|={expected_rows}",
        edges.len()
    );

    const REPS: usize = 3;
    let mut off_ms = Vec::new();
    let mut off_peak = Vec::new();
    let mut on_ms = Vec::new();
    let mut on_peak = Vec::new();
    let mut on_dispatch = 0u64;
    let mut on_rows = 0usize;
    let mut off_rows = 0usize;

    let _ = engine_run(&edges, budget, false);
    let _ = engine_run(&edges, budget, true);
    for rep in 0..REPS {
        let (o_dt, o_pk, _, o_r) = engine_run(&edges, budget, false);
        let (n_dt, n_pk, disp, n_r) = engine_run(&edges, budget, true);
        off_ms.push(o_dt);
        off_peak.push(o_pk as f64);
        on_ms.push(n_dt);
        on_peak.push(n_pk as f64);
        on_dispatch = disp;
        off_rows = o_r;
        on_rows = n_r;
        eprintln!(
            "S4-sparse rep {rep}: legacy {o_dt:.1} ms / {:.1} MiB ; factorized {n_dt:.1} ms / {:.1} MiB (dispatch={disp})",
            o_pk as f64 / (1024.0 * 1024.0),
            n_pk as f64 / (1024.0 * 1024.0),
        );
    }
    std::env::remove_var(KILL_SWITCH);

    assert_eq!(on_rows, expected_rows, "sparse-route TC row count");
    assert_eq!(off_rows, expected_rows, "legacy TC row count");
    assert!(
        on_dispatch >= 1,
        "sparse route must fire (domain > dense cap)"
    );

    let om = median(&mut off_ms);
    let opk = median(&mut off_peak);
    let nm = median(&mut on_ms);
    let npk = median(&mut on_peak);
    eprintln!(
        "S4-sparse bench: rows={on_rows} dispatch={on_dispatch} | legacy {om:.1} ms / {:.1} MiB ; \
         factorized {nm:.1} ms / {:.1} MiB | peak {:.2}x  wall-clock {:.3}x  (gate: peak<1.0x at wall<=1.2x)",
        opk / (1024.0 * 1024.0),
        npk / (1024.0 * 1024.0),
        opk / npk.max(1.0),
        nm / om.max(1.0),
    );
    assert!(npk < opk, "sparse route must cut peak vs legacy");
    assert!(
        nm <= om * 1.2,
        "sparse route must not regress wall-clock beyond 1.2x"
    );
}
