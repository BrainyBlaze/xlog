//! D2 Free Join — end-to-end executor wiring.
//!
//! A general >=3-atom inner-join body with no dedicated kernel must
//! compile (general multiway promoter emits `MultiWayJoin` with
//! `plan: Some(MultiwayPlan::FreeJoin)`), dispatch the Free Join
//! frontier engine
//! (`free_join_dispatch_count() == 1`), and produce the same row set
//! as the embedded binary fallback (kill switch
//! `XLOG_DISABLE_FREE_JOIN=1` forces the fallback; deduped sorted
//! rows must be identical). Fused/kill-switch phases run inside ONE
//! test because the kill switch is a process-global env var.
//!
//! Decline coverage: dedicated shapes (triangle) never route through
//! the generic engine, and non-prefix bodies (a bound variable behind
//! an unbound column — flat sorted tries consume columns physically
//! left-to-right) decline silently to the fallback with the counter
//! at 0 and correct rows.

use std::collections::BTreeMap;
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

fn download_column_u32(
    memory: &Arc<GpuMemoryManager>,
    buffer: &CudaBuffer,
    col: usize,
) -> Vec<u32> {
    let n = buffer_rows(memory, buffer);
    let mut bytes = vec![0u8; n * 4];
    if n == 0 {
        return Vec::new();
    }
    let CudaColumn::Owned(c) = buffer.column(col).expect("column") else {
        panic!("column must be owned");
    };
    unsafe {
        let res = sys::cuMemcpyDtoH_v2(bytes.as_mut_ptr() as *mut _, *c.device_ptr(), bytes.len());
        assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS, "dtoh column copy");
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Download all u32 columns of `buffer` and return the deduped sorted
/// row set (Datalog set semantics — derivation multiplicity is not
/// part of the contract and may differ between engines).
fn download_row_set(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<Vec<u32>> {
    let cols: Vec<Vec<u32>> = (0..buffer.arity())
        .map(|c| download_column_u32(memory, buffer, c))
        .collect();
    let n = cols.first().map_or(0, Vec::len);
    let mut rows: Vec<Vec<u32>> = (0..n).map(|r| cols.iter().map(|c| c[r]).collect()).collect();
    rows.sort();
    rows.dedup();
    rows
}

/// Compile + execute `source` over `inputs`, download the `q` relation,
/// and return its deduped sorted row set plus the Free Join dispatch
/// counter.
fn run_program(
    fix: &Fixture,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> (Vec<Vec<u32>>, u64) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile rule");
    let mut executor = Executor::new(Arc::clone(&fix.provider));
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        executor.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    executor.execute_plan(&plan).expect("execute plan");
    let q = executor.store().get("q").expect("q relation");
    let rows = download_row_set(&fix.memory, q);
    (rows, executor.free_join_dispatch_count())
}

/// The kill switch is a process-global env var: every test that toggles
/// it (or asserts the dispatch counter fired) takes this lock so a
/// concurrent kill-switch phase cannot leak into another test's phase.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// 4-atom chain: no dedicated kernel, every shared variable is a
/// leading prefix of its later atom — Free Join must fire.
const CHAIN_SOURCE: &str = "q(A, B) :- r(A, X), s(X, Y), t(Y, Z), u(Z, B).";

fn chain_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    inputs.insert("r", vec![(1, 10), (2, 20), (3, 30)]);
    inputs.insert("s", vec![(10, 100), (10, 101), (20, 200)]);
    inputs.insert("t", vec![(100, 1000), (101, 1000), (200, 2000), (200, 2001)]);
    inputs.insert("u", vec![(1000, 7), (2000, 8), (2001, 8), (1000, 9)]);
    inputs
}

/// A=1: X=10 -> Y in {100,101} -> Z=1000 -> B in {7,9}.
/// A=2: X=20 -> Y=200 -> Z in {2000,2001} -> B=8.
/// A=3: X=30 -> no s row.
fn chain_expected() -> Vec<Vec<u32>> {
    vec![vec![1, 7], vec![1, 9], vec![2, 8]]
}

#[test]
fn free_join_fires_on_4atom_chain_with_kill_switch_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = env_lock();
    let inputs = chain_inputs();
    let expected = chain_expected();

    let (fused, fused_count) = run_program(&fix, CHAIN_SOURCE, &inputs);
    assert_eq!(fused, expected, "Free Join path row set");
    assert_eq!(fused_count, 1, "Free Join dispatch must fire exactly once");

    // SAFETY: single-threaded phase of this test; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_FREE_JOIN", "1");
    }
    let (unfused, unfused_count) = run_program(&fix, CHAIN_SOURCE, &inputs);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_FREE_JOIN");
    }
    assert_eq!(unfused, expected, "kill-switch fallback row set");
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
}

/// Dedicated-shape carve-out: a canonical triangle routes through the
/// dedicated triangle dispatchers (promoted by `try_promote_triangle`
/// long before the general promoter runs), so the Free Join counter
/// must stay at 0 while the rows are still correct.
#[test]
fn free_join_declines_dedicated_triangle_shape() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = env_lock();
    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    // Triangles: (1,2,3), (1,2,4).
    inputs.insert("e1", vec![(1, 2), (1, 3)]);
    inputs.insert("e2", vec![(2, 3), (2, 4)]);
    inputs.insert("e3", vec![(1, 3), (1, 4)]);

    let source = "q(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).";
    let (rows, fj_count) = run_program(&fix, source, &inputs);
    assert_eq!(
        rows,
        vec![vec![1, 2, 3], vec![1, 2, 4]],
        "triangle row set"
    );
    assert_eq!(fj_count, 0, "dedicated triangle must not route through Free Join");
}

/// Non-prefix decline: X is shared between r and s but sits at column 1
/// of s BEHIND the unbound Y — no atom order makes the bound variables
/// a leading prefix, so the dispatcher declines and the embedded binary
/// fallback produces the rows. Counter must stay at 0.
#[test]
fn free_join_declines_non_prefix_body_with_correct_fallback() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = env_lock();
    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    inputs.insert("r", vec![(1, 10), (2, 20)]);
    inputs.insert("s", vec![(100, 10), (200, 20), (201, 20)]);
    inputs.insert("t", vec![(100, 5), (200, 6), (201, 7)]);

    // A=1: X=10 -> Y=100 -> B=5. A=2: X=20 -> Y in {200,201} -> B in {6,7}.
    let source = "q(A, B) :- r(A, X), s(Y, X), t(Y, B).";
    let (rows, fj_count) = run_program(&fix, source, &inputs);
    assert_eq!(
        rows,
        vec![vec![1, 5], vec![2, 6], vec![2, 7]],
        "non-prefix fallback row set"
    );
    assert_eq!(fj_count, 0, "non-prefix body must decline Free Join");
}

/// Recursive-SCC integration: a linear-recursive 3-atom chain body
/// (`reach(X,B) :- e1(X,Y), e2(Y,Z), reach(Z,B)`) has no dedicated
/// kernel, so the general promoter emits a generic MultiWayJoin and the
/// recursive engine's `execute_wcoj_or_fallback_node` hook dispatches
/// Free Join on the seeding pass AND on each semi-naive delta-rewritten
/// variant. Kill-switch run must produce the identical fixpoint with the
/// counter at 0.
#[test]
fn free_join_fires_inside_recursive_scc_with_kill_switch_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = env_lock();

    let source = "pred e1(u32, u32).\n\
                  pred e2(u32, u32).\n\
                  pred seed(u32, u32).\n\
                  pred reach(u32, u32).\n\
                  reach(X, B) :- e1(X, Y), e2(Y, Z), reach(Z, B).\n\
                  reach(X, B) :- seed(X, B).";
    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    inputs.insert("seed", vec![(1, 100), (2, 200)]);
    inputs.insert("e1", vec![(10, 20), (11, 21), (12, 22)]);
    inputs.insert("e2", vec![(20, 1), (21, 2), (22, 10)]);

    // Fixpoint: seed rows; iter1 (10,100) via z=1, (11,200) via z=2;
    // iter2 (12,100) via z=10; iter3 empty.
    let expected: Vec<Vec<u32>> = vec![
        vec![1, 100],
        vec![2, 200],
        vec![10, 100],
        vec![11, 200],
        vec![12, 100],
    ];

    let run = |fix: &Fixture| -> (Vec<Vec<u32>>, u64) {
        let mut compiler = Compiler::new();
        let plan = compiler.compile(source).expect("compile recursive rule");
        let mut executor = Executor::new(Arc::clone(&fix.provider));
        for (name, rel_id) in compiler.rel_ids() {
            executor.register_relation(*rel_id, name);
        }
        for (name, rows) in &inputs {
            executor.put_relation(name, upload_binary_u32(&fix.memory, rows));
        }
        executor.execute_plan(&plan).expect("execute plan");
        let reach = executor.store().get("reach").expect("reach relation");
        (
            download_row_set(&fix.memory, reach),
            executor.free_join_dispatch_count(),
        )
    };

    let (fused, fused_count) = run(&fix);
    assert_eq!(fused, expected, "recursive Free Join fixpoint row set");
    // Seeding pass + at least one delta-variant iteration must have
    // dispatched; the exact count depends on iteration scheduling, so
    // pin only the lower bound.
    assert!(
        fused_count >= 2,
        "free join must fire on seeding and delta variants, got {fused_count}"
    );

    // SAFETY: single-threaded phase of this test; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_FREE_JOIN", "1");
    }
    let (unfused, unfused_count) = run(&fix);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_FREE_JOIN");
    }
    assert_eq!(unfused, expected, "kill-switch recursive fixpoint row set");
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
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

fn download_column_u64(
    memory: &Arc<GpuMemoryManager>,
    buffer: &CudaBuffer,
    col: usize,
) -> Vec<u64> {
    let n = buffer_rows(memory, buffer);
    let mut bytes = vec![0u8; n * 8];
    if n == 0 {
        return Vec::new();
    }
    let CudaColumn::Owned(c) = buffer.column(col).expect("column") else {
        panic!("column must be owned");
    };
    unsafe {
        let res = sys::cuMemcpyDtoH_v2(bytes.as_mut_ptr() as *mut _, *c.device_ptr(), bytes.len());
        assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS, "dtoh column copy");
    }
    bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn download_row_set_u64(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<Vec<u64>> {
    let cols: Vec<Vec<u64>> = (0..buffer.arity())
        .map(|c| download_column_u64(memory, buffer, c))
        .collect();
    let n = cols.first().map_or(0, Vec::len);
    let mut rows: Vec<Vec<u64>> = (0..n).map(|r| cols.iter().map(|c| c[r]).collect()).collect();
    rows.sort();
    rows.dedup();
    rows
}

/// u64 width-class end-to-end: a 4-atom chain over `u64` predicates
/// must route through `free_join_execute_u64_recorded` (counter == 1)
/// with kill-switch parity. Values above 2^32 prove true 64-bit key
/// handling through the whole executor path.
#[test]
fn free_join_fires_on_u64_chain_with_kill_switch_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = env_lock();

    const HI: u64 = 1u64 << 32;
    let source = "pred r(u64, u64).\n\
                  pred s(u64, u64).\n\
                  pred t(u64, u64).\n\
                  pred u(u64, u64).\n\
                  pred q(u64, u64).\n\
                  q(A, B) :- r(A, X), s(X, Y), t(Y, Z), u(Z, B).";
    let mut inputs: BTreeMap<&str, Vec<(u64, u64)>> = BTreeMap::new();
    // (1, 5+HI) must NOT join the truncation decoy (5, ...) — only
    // the true u64 key 5+HI.
    inputs.insert("r", vec![(1, 5 + HI), (2, 6)]);
    inputs.insert("s", vec![(5 + HI, 10 + 2 * HI), (5, 666), (6, 20)]);
    inputs.insert("t", vec![(10 + 2 * HI, 30), (10, 667), (20, 40 + 3 * HI)]);
    inputs.insert("u", vec![(30, 7 + 4 * HI), (40 + 3 * HI, 8), (40, 668)]);

    let expected: Vec<Vec<u64>> = vec![vec![1, 7 + 4 * HI], vec![2, 8]];

    let run = |fix: &Fixture| -> (Vec<Vec<u64>>, u64) {
        let mut compiler = Compiler::new();
        let plan = compiler.compile(source).expect("compile u64 chain");
        let mut executor = Executor::new(Arc::clone(&fix.provider));
        for (name, rel_id) in compiler.rel_ids() {
            executor.register_relation(*rel_id, name);
        }
        for (name, rows) in &inputs {
            executor.put_relation(name, upload_binary_u64(&fix.memory, rows));
        }
        executor.execute_plan(&plan).expect("execute plan");
        let q = executor.store().get("q").expect("q relation");
        (
            download_row_set_u64(&fix.memory, q),
            executor.free_join_dispatch_count(),
        )
    };

    let (fused, fused_count) = run(&fix);
    assert_eq!(fused, expected, "u64 Free Join path row set");
    assert_eq!(fused_count, 1, "u64 Free Join dispatch must fire exactly once");

    // SAFETY: single-threaded phase of this test; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_FREE_JOIN", "1");
    }
    let (unfused, unfused_count) = run(&fix);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_FREE_JOIN");
    }
    assert_eq!(unfused, expected, "u64 kill-switch fallback row set");
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
}

/// D2 §2.4 — fused factorized count-by-root over a general body: a
/// count aggregate over the 4-atom chain must dispatch the Free Join
/// count engine (both the fused-groupby counter AND the Free Join
/// counter fire), with row parity against BOTH kill switches
/// (`XLOG_DISABLE_WCOJ_GROUPBY_FUSION` disables the fused hook;
/// `XLOG_DISABLE_FREE_JOIN` disables the Free Join route).
#[test]
fn free_join_fused_count_fires_with_kill_switch_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = env_lock();

    let source = "deg(A, count(B)) :- r(A, X), s(X, Y), t(Y, Z), u(Z, B).";
    let inputs = chain_inputs();
    // Full bindings per A: A=1 -> (10,100,1000,7),(10,100,1000,9),
    // (10,101,1000,7),(10,101,1000,9) = 4; A=2 -> (20,200,2000,8),
    // (20,200,2001,8) = 2.
    let expected: Vec<(u32, u64)> = vec![(1, 4), (2, 2)];

    let run = |fix: &Fixture| -> (Vec<(u32, u64)>, u64, u64) {
        let mut compiler = Compiler::new();
        let plan = compiler.compile(source).expect("compile count rule");
        let mut executor = Executor::new(Arc::clone(&fix.provider));
        for (name, rel_id) in compiler.rel_ids() {
            executor.register_relation(*rel_id, name);
        }
        for (name, rows) in &inputs {
            executor.put_relation(name, upload_binary_u32(&fix.memory, rows));
        }
        executor.execute_plan(&plan).expect("execute plan");
        let deg = executor.store().get("deg").expect("deg relation");
        let keys = download_column_u32(&fix.memory, deg, 0);
        let counts = download_column_u64(&fix.memory, deg, 1);
        let mut rows: Vec<(u32, u64)> = keys.into_iter().zip(counts).collect();
        rows.sort();
        rows.dedup();
        (
            rows,
            executor.wcoj_groupby_fusion_dispatch_count(),
            executor.free_join_dispatch_count(),
        )
    };

    let (fused, fusion_count, fj_count) = run(&fix);
    assert_eq!(fused, expected, "fused factorized count row set");
    assert_eq!(fusion_count, 1, "fused groupby dispatch must fire once");
    assert_eq!(fj_count, 1, "free join count dispatch must fire once");

    for kill in ["XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "XLOG_DISABLE_FREE_JOIN"] {
        // SAFETY: single-threaded phase of this test; restored below.
        unsafe {
            std::env::set_var(kill, "1");
        }
        let (unfused, fusion_count, fj_count) = run(&fix);
        unsafe {
            std::env::remove_var(kill);
        }
        assert_eq!(unfused, expected, "kill-switch ({kill}) row set");
        assert_eq!(fusion_count, 0, "{kill} must keep the fusion counter at 0");
        assert_eq!(fj_count, 0, "{kill} must keep the free join counter at 0");
    }
}
