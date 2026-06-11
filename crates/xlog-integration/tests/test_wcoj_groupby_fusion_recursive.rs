//! D1 aggregate-fused WCOJ over recursive-stratum inputs.
//!
//! The legal fragment: a NON-recursive aggregate rule in a later stratum
//! whose triangle body reads predicates computed by an earlier recursive
//! stratum (aggregates inside recursive rules are stratification-rejected
//! at compile time and out of scope by language contract). The fused
//! group-by-root kernel must (a) fire (counter == 1), (b) match the
//! kill-switched materialize+groupby row set, and (c) match a
//! host-computed oracle over the transitive closure. The recursive
//! stratum's final merged relation comes from union_gpu + dedup, whose
//! output is lex-sorted + deduped — the layout contract the fused
//! provider entry's binary-search work plan requires; these tests lock
//! that end-to-end behavior.
//!
//! Fused/kill-switch phases run inside ONE test because the kill switch
//! is a process-global env var.

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

fn download_column_bytes(
    memory: &Arc<GpuMemoryManager>,
    buffer: &CudaBuffer,
    col: usize,
    elem_size: usize,
) -> Vec<u8> {
    let n = buffer_rows(memory, buffer);
    let mut bytes = vec![0u8; n * elem_size];
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

fn download_group_counts(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u32, u64)> {
    assert_eq!(buffer.arity(), 2, "expected (X, count) output");
    let keys: Vec<u32> = download_column_bytes(memory, buffer, 0, 4)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let counts: Vec<u64> = download_column_bytes(memory, buffer, 1, 8)
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut out: Vec<(u32, u64)> = keys.into_iter().zip(counts).collect();
    out.sort();
    out
}

/// Host oracle: transitive closure of `e` (the recursive stratum's
/// semantics for `tc(X,Y) :- e(X,Y). tc(X,Y) :- tc(X,Z), e(Z,Y).`).
fn host_transitive_closure(e: &[(u32, u32)]) -> BTreeSet<(u32, u32)> {
    let mut tc: BTreeSet<(u32, u32)> = e.iter().copied().collect();
    loop {
        let mut new: Vec<(u32, u32)> = Vec::new();
        for &(x, z) in &tc {
            for &(z2, y) in e {
                if z == z2 && !tc.contains(&(x, y)) {
                    new.push((x, y));
                }
            }
        }
        if new.is_empty() {
            break;
        }
        tc.extend(new);
    }
    tc
}

/// Host oracle: `deg(X, count(Z)) :- ab(X,Y), bc(Y,Z), ac(X,Z).`
fn host_group_counts(
    ab: &BTreeSet<(u32, u32)>,
    bc: &BTreeSet<(u32, u32)>,
    ac: &BTreeSet<(u32, u32)>,
) -> Vec<(u32, u64)> {
    let mut counts: BTreeMap<u32, u64> = BTreeMap::new();
    for &(x, y) in ab {
        for &(y2, z) in bc {
            if y == y2 && ac.contains(&(x, z)) {
                *counts.entry(x).or_insert(0) += 1;
            }
        }
    }
    counts.into_iter().collect()
}

/// Host oracle: `deg(X, sum(Z)) :- ab(X,Y), bc(Y,Z), ac(X,Z).`
fn host_group_sums(
    ab: &BTreeSet<(u32, u32)>,
    bc: &BTreeSet<(u32, u32)>,
    ac: &BTreeSet<(u32, u32)>,
) -> Vec<(u32, u64)> {
    let mut sums: BTreeMap<u32, u64> = BTreeMap::new();
    for &(x, y) in ab {
        for &(y2, z) in bc {
            if y == y2 && ac.contains(&(x, z)) {
                *sums.entry(x).or_insert(0) += z as u64;
            }
        }
    }
    sums.into_iter().collect()
}

/// Compile and execute a multi-stratum program, returning the `deg`
/// rows, the fusion dispatch counter, and the error-decline counter.
fn run_program(
    fix: &Fixture,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> (Vec<(u32, u64)>, u64, u64) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile program");
    let mut executor = Executor::new(Arc::clone(&fix.provider));
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        executor.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    executor.execute_plan(&plan).expect("execute plan");
    let deg = executor.store().get("deg").expect("deg relation");
    let rows = download_group_counts(&fix.memory, deg);
    (
        rows,
        executor.wcoj_groupby_fusion_dispatch_count(),
        executor.wcoj_error_decline_count(),
    )
}

/// The kill switch is a process-global env var: every test that toggles
/// it (or asserts the fused counter fired) takes this lock so a
/// concurrent kill-switch phase cannot leak into another test's fused
/// phase.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Fused phase + kill-switch phase + oracle check for one program.
fn assert_recursive_fusion_parity(
    fix: &Fixture,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    expected: &[(u32, u64)],
) {
    let _guard = env_lock();

    // Phase 1: fused path fires once, no error declines, oracle rows.
    let (fused, fused_count, fused_errors) = run_program(fix, source, inputs);
    assert_eq!(fused, expected, "fused path row set: {source}");
    assert_eq!(
        fused_count, 1,
        "fused group-by-root dispatch must fire exactly once: {source}"
    );
    assert_eq!(
        fused_errors, 0,
        "fused path must not decline on pipeline errors: {source}"
    );

    // Phase 2: kill switch forces the materialize+groupby path; rows
    // must be identical and the counter must stay 0.
    // SAFETY: serialized by ENV_LOCK; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (unfused, unfused_count, _) = run_program(fix, source, inputs);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    assert_eq!(unfused, expected, "kill-switch path row set: {source}");
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
}

/// Base edges: a 4-node path {1->2->3->4} plus a disjoint 3-node path
/// {5->6->7}. tc = transitive closure = K4-as-DAG on {1..4} plus the
/// {5,6,7} triangle — multi-iteration fixpoint (3 closure hops), so the
/// final `tc` buffer is the product of repeated union_gpu delta merges.
fn base_edges() -> Vec<(u32, u32)> {
    vec![(1, 2), (2, 3), (3, 4), (5, 6), (6, 7)]
}

/// Closing edges for the mixed-body case (EDB `q`).
fn closing_edges() -> Vec<(u32, u32)> {
    vec![(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)]
}

#[test]
fn groupby_fusion_fires_over_recursive_stratum_inputs() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Aggregate stratum reads the recursive stratum's tc twice plus an
    // EDB closing edge q. Explicit u32 `pred` declarations keep the
    // recursive seed's declared schema aligned with the uploaded EDB
    // width (untyped variables default to u64).
    let source = "pred e(u32, u32).\n\
                  pred q(u32, u32).\n\
                  pred tc(u32, u32).\n\
                  tc(X, Y) :- e(X, Y).\n\
                  tc(X, Y) :- tc(X, Z), e(Z, Y).\n\
                  deg(X, count(Z)) :- tc(X, Y), tc(Y, Z), q(X, Z).";
    let e = base_edges();
    let q = closing_edges();
    let tc = host_transitive_closure(&e);
    let q_set: BTreeSet<(u32, u32)> = q.iter().copied().collect();
    let expected = host_group_counts(&tc, &tc, &q_set);
    // Guard against a degenerate fixture: the oracle must be non-empty.
    assert_eq!(expected, vec![(1, 3), (2, 1), (5, 1)], "oracle sanity");

    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    inputs.insert("e", e);
    inputs.insert("q", q);
    assert_recursive_fusion_parity(&fix, source, &inputs, &expected);
}

#[test]
fn groupby_fusion_fires_over_recursive_self_join_body() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Edge case: every body atom is the SAME recursive predicate — the
    // promoter's inside-aggregate descent sees a body whose scans all
    // point at the recursive stratum's RelId.
    let source = "pred e(u32, u32).\n\
                  pred tc(u32, u32).\n\
                  tc(X, Y) :- e(X, Y).\n\
                  tc(X, Y) :- tc(X, Z), e(Z, Y).\n\
                  deg(X, count(Z)) :- tc(X, Y), tc(Y, Z), tc(X, Z).";
    let e = base_edges();
    let tc = host_transitive_closure(&e);
    let expected = host_group_counts(&tc, &tc, &tc);
    assert_eq!(expected, vec![(1, 3), (2, 1), (5, 1)], "oracle sanity");

    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    inputs.insert("e", e);
    assert_recursive_fusion_parity(&fix, source, &inputs, &expected);
}

#[test]
fn groupby_fusion_sum_fires_over_recursive_stratum_inputs() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Value-reading aggregate (sum) over recursive-stratum inputs: the
    // widened _agg dispatch must read Z during traversal of the
    // recursive stratum's merged buffers.
    let source = "pred e(u32, u32).\n\
                  pred q(u32, u32).\n\
                  pred tc(u32, u32).\n\
                  tc(X, Y) :- e(X, Y).\n\
                  tc(X, Y) :- tc(X, Z), e(Z, Y).\n\
                  deg(X, sum(Z)) :- tc(X, Y), tc(Y, Z), q(X, Z).";
    let e = base_edges();
    let q = closing_edges();
    let tc = host_transitive_closure(&e);
    let q_set: BTreeSet<(u32, u32)> = q.iter().copied().collect();
    let expected = host_group_sums(&tc, &tc, &q_set);
    assert_eq!(expected, vec![(1, 11), (2, 4), (5, 7)], "oracle sanity");

    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    inputs.insert("e", e);
    inputs.insert("q", q);
    assert_recursive_fusion_parity(&fix, source, &inputs, &expected);
}
