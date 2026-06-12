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

fn make_fixture() -> Option<Fixture> {
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
        Box::new(GlobalDeviceBudget::new(logging, 512 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(512 * 1024 * 1024),
        runtime,
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(Fixture { memory, provider })
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
    ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
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
    assert_eq!(rows.len(), 9, "u64 TC must still be exact via the legacy path");
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
    assert!(
        !download_row_set_u32(&fix.memory, executor.store().get("q").expect("q")).is_empty()
    );
}

#[test]
fn domain_over_cap_declines_silently() {
    let _guard = env_lock();
    std::env::remove_var(KILL_SWITCH);
    let Some(fix) = make_fixture() else {
        eprintln!("skipping domain decline: no CUDA device");
        return;
    };
    // Ids above the default 2^14 dispatch cap (but far below u32) —
    // the dense-domain gate must decline for the whole fixpoint.
    const BASE: u32 = 1 << 15;
    let edges: Vec<(u32, u32)> = vec![(BASE, BASE + 1), (BASE + 1, BASE + 2), (BASE + 2, BASE)];
    let source = "pred edge(u32, u32).\n\
                  pred q(u32, u32).\n\
                  q(X, Y) :- edge(X, Y).\n\
                  q(X, Z) :- q(X, Y), edge(Y, Z).";
    let executor = run_program(&fix, source, upload_binary_u32(&fix.memory, &edges));
    assert_eq!(
        executor.factorized_delta_dispatch_count(),
        0,
        "domain over the cap must decline the factorized path"
    );
    let rows = download_row_set_u32(&fix.memory, executor.store().get("q").expect("q"));
    assert_eq!(rows.len(), 9, "closure must still be exact via the legacy path");
}
