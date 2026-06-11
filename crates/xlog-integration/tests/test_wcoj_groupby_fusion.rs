//! D1 aggregate-fused WCOJ — end-to-end executor wiring.
//!
//! A count/sum/min/max-aggregate head over a triangle body must compile
//! (promoter descends the aggregate wrapper), dispatch the fused
//! group-by-root kernel (counter == 1), and produce the same rows as the
//! materialize+groupby path (kill switch forces the unfused path; rows must
//! be identical). Fused/kill-switch phases run inside ONE test because the
//! kill switch is a process-global env var.

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

const SOURCE: &str = "deg(X, count(Z)) :- e1(X, Y), e2(Y, Z), e3(X, Z).";

fn run_program(fix: &Fixture, inputs: &BTreeMap<&str, Vec<(u32, u32)>>) -> (Vec<(u32, u64)>, u64) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(SOURCE).expect("compile aggregate rule");
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
    (rows, executor.wcoj_groupby_fusion_dispatch_count())
}

/// Run an aggregate program over the given inputs, download the `agg`
/// relation with `download`, and return the rows plus the fusion dispatch
/// counter.
fn run_agg_program<T>(
    fix: &Fixture,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    download: impl Fn(&Arc<GpuMemoryManager>, &CudaBuffer) -> Vec<T>,
) -> (Vec<T>, u64) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile aggregate rule");
    let mut executor = Executor::new(Arc::clone(&fix.provider));
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        executor.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    executor.execute_plan(&plan).expect("execute plan");
    let agg = executor.store().get("agg").expect("agg relation");
    let rows = download(&fix.memory, agg);
    (rows, executor.wcoj_groupby_fusion_dispatch_count())
}

fn download_groups_u32(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u32, u32)> {
    assert_eq!(buffer.arity(), 2, "expected (X, agg) output");
    let keys: Vec<u32> = download_column_bytes(memory, buffer, 0, 4)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let aggs: Vec<u32> = download_column_bytes(memory, buffer, 1, 4)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut out: Vec<(u32, u32)> = keys.into_iter().zip(aggs).collect();
    out.sort();
    out
}

/// The kill switch is a process-global env var: every test that toggles it
/// (or asserts the fused counter fired) takes this lock so a concurrent
/// kill-switch phase cannot leak into another test's fused phase.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Shared triangle fixture: K4 on {1..4} plus a disjoint triangle {5,6,7}.
/// Completions: X=1 -> (Y,Z) in {(2,3),(2,4),(3,4)}; X=2 -> (3,4);
/// X=5 -> (6,7).
fn triangle_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    inputs.insert(
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
    inputs.insert("e2", vec![(2, 3), (2, 4), (3, 4), (6, 7)]);
    inputs.insert("e3", vec![(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)]);
    inputs
}

/// Fused-vs-kill-switch phases for one u64-valued aggregate source (sum).
fn assert_fusion_parity_u64(fix: &Fixture, source: &str, expected: &[(u32, u64)]) {
    let _guard = env_lock();
    let inputs = triangle_inputs();
    let (fused, fused_count) = run_agg_program(fix, source, &inputs, download_group_counts);
    assert_eq!(fused, expected, "fused path row set: {source}");
    assert_eq!(fused_count, 1, "fused dispatch must fire once: {source}");

    // SAFETY: single-threaded phase of this test; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (unfused, unfused_count) = run_agg_program(fix, source, &inputs, download_group_counts);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    assert_eq!(unfused, expected, "kill-switch path row set: {source}");
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
}

/// Fused-vs-kill-switch phases for one u32-valued aggregate source (min/max).
fn assert_fusion_parity_u32(fix: &Fixture, source: &str, expected: &[(u32, u32)]) {
    let _guard = env_lock();
    let inputs = triangle_inputs();
    let (fused, fused_count) = run_agg_program(fix, source, &inputs, download_groups_u32);
    assert_eq!(fused, expected, "fused path row set: {source}");
    assert_eq!(fused_count, 1, "fused dispatch must fire once: {source}");

    // SAFETY: single-threaded phase of this test; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (unfused, unfused_count) = run_agg_program(fix, source, &inputs, download_groups_u32);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    assert_eq!(unfused, expected, "kill-switch path row set: {source}");
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
}

#[test]
fn groupby_fusion_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let _guard = env_lock();
    // K4 on {1..4} plus a disjoint triangle {5,6,7}; per-X completion
    // counts: X=1 -> 3, X=2 -> 1, X=5 -> 1.
    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    inputs.insert(
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
    inputs.insert("e2", vec![(2, 3), (2, 4), (3, 4), (6, 7)]);
    inputs.insert("e3", vec![(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)]);
    let expected = vec![(1u32, 3u64), (2, 1), (5, 1)];

    // Phase 1: fused path fires and is correct.
    let (fused_rows, fused_count) = run_program(&fix, &inputs);
    assert_eq!(fused_rows, expected, "fused path row set");
    assert_eq!(
        fused_count, 1,
        "fused group-by-root count dispatch must fire exactly once"
    );

    // Phase 2: kill switch forces the materialize+groupby path; rows must
    // be identical and the counter must stay 0.
    // SAFETY: single-threaded phase of this test; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (unfused_rows, unfused_count) = run_program(&fix, &inputs);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    assert_eq!(unfused_rows, expected, "kill-switch path row set");
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
}

#[test]
fn groupby_fusion_sum_z_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Per-X completions (see triangle_inputs): sums of Z.
    // X=1: 3+4+4 = 11; X=2: 4; X=5: 7.
    assert_fusion_parity_u64(
        &fix,
        "agg(X, sum(Z)) :- e1(X, Y), e2(Y, Z), e3(X, Z).",
        &[(1, 11), (2, 4), (5, 7)],
    );
}

#[test]
fn groupby_fusion_sum_y_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Sums of Y per completion. X=1: 2+2+3 = 7; X=2: 3; X=5: 6.
    assert_fusion_parity_u64(
        &fix,
        "agg(X, sum(Y)) :- e1(X, Y), e2(Y, Z), e3(X, Z).",
        &[(1, 7), (2, 3), (5, 6)],
    );
}

#[test]
fn groupby_fusion_min_z_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Min of Z. X=1: min(3,4,4)=3; X=2: 4; X=5: 7.
    assert_fusion_parity_u32(
        &fix,
        "agg(X, min(Z)) :- e1(X, Y), e2(Y, Z), e3(X, Z).",
        &[(1, 3), (2, 4), (5, 7)],
    );
}

#[test]
fn groupby_fusion_max_z_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Max of Z. X=1: max(3,4,4)=4; X=2: 4; X=5: 7.
    assert_fusion_parity_u32(
        &fix,
        "agg(X, max(Z)) :- e1(X, Y), e2(Y, Z), e3(X, Z).",
        &[(1, 4), (2, 4), (5, 7)],
    );
}

#[test]
fn groupby_fusion_max_y_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Max of Y. X=1: max(2,2,3)=3; X=2: 3; X=5: 6.
    assert_fusion_parity_u32(
        &fix,
        "agg(X, max(Y)) :- e1(X, Y), e2(Y, Z), e3(X, Z).",
        &[(1, 3), (2, 3), (5, 6)],
    );
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

fn download_groups_u64_u64(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u64, u64)> {
    assert_eq!(buffer.arity(), 2, "expected (X, count) output");
    let keys: Vec<u64> = download_column_bytes(memory, buffer, 0, 8)
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let counts: Vec<u64> = download_column_bytes(memory, buffer, 1, 8)
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut out: Vec<(u64, u64)> = keys.into_iter().zip(counts).collect();
    out.sort();
    out
}

/// Run the count program over U64 relations and return rows + counter.
fn run_count_program_u64(
    fix: &Fixture,
    inputs: &BTreeMap<&str, Vec<(u64, u64)>>,
) -> (Vec<(u64, u64)>, u64) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(SOURCE).expect("compile aggregate rule");
    let mut executor = Executor::new(Arc::clone(&fix.provider));
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        executor.put_relation(name, upload_binary_u64(&fix.memory, rows));
    }
    executor.execute_plan(&plan).expect("execute plan");
    let deg = executor.store().get("deg").expect("deg relation");
    let rows = download_groups_u64_u64(&fix.memory, deg);
    (rows, executor.wcoj_groupby_fusion_dispatch_count())
}

#[test]
fn groupby_fusion_count_u64_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let _guard = env_lock();

    // Same K4 + disjoint triangle shape with keys above u32::MAX, so a
    // width-truncating dispatch would visibly fail.
    const B: u64 = 1 << 33;
    let map = |rows: &[(u32, u32)]| -> Vec<(u64, u64)> {
        rows.iter()
            .map(|&(a, b)| (B + a as u64, B + b as u64))
            .collect()
    };
    let mut inputs: BTreeMap<&str, Vec<(u64, u64)>> = BTreeMap::new();
    inputs.insert(
        "e1",
        map(&[
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (5, 6),
            (5, 7),
            (6, 7),
        ]),
    );
    inputs.insert("e2", map(&[(2, 3), (2, 4), (3, 4), (6, 7)]));
    inputs.insert("e3", map(&[(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)]));
    let expected = vec![(B + 1, 3u64), (B + 2, 1), (B + 5, 1)];

    // Phase 1: fused u64 path fires and is correct.
    let (fused_rows, fused_count) = run_count_program_u64(&fix, &inputs);
    assert_eq!(fused_rows, expected, "fused u64 path row set");
    assert_eq!(
        fused_count, 1,
        "fused u64 group-by-root count dispatch must fire exactly once"
    );

    // Phase 2: kill switch forces the materialize+groupby path.
    // SAFETY: serialized by ENV_LOCK; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (unfused_rows, unfused_count) = run_count_program_u64(&fix, &inputs);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    assert_eq!(unfused_rows, expected, "kill-switch u64 path row set");
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
}

/// Run an aggregate program over U64 relations and return rows + counter.
fn run_agg_program_u64(
    fix: &Fixture,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u64, u64)>>,
) -> (Vec<(u64, u64)>, u64) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile aggregate rule");
    let mut executor = Executor::new(Arc::clone(&fix.provider));
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        executor.put_relation(name, upload_binary_u64(&fix.memory, rows));
    }
    executor.execute_plan(&plan).expect("execute plan");
    let agg = executor.store().get("agg").expect("agg relation");
    let rows = download_groups_u64_u64(&fix.memory, agg);
    (rows, executor.wcoj_groupby_fusion_dispatch_count())
}

/// U64 triangle fixture: the K4 + disjoint triangle shape shifted above
/// 2^33 so width truncation visibly fails.
fn triangle_inputs_u64() -> (BTreeMap<&'static str, Vec<(u64, u64)>>, u64) {
    const B: u64 = 1 << 33;
    let map = |rows: &[(u32, u32)]| -> Vec<(u64, u64)> {
        rows.iter()
            .map(|&(a, b)| (B + a as u64, B + b as u64))
            .collect()
    };
    let mut inputs: BTreeMap<&str, Vec<(u64, u64)>> = BTreeMap::new();
    inputs.insert(
        "e1",
        map(&[
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (5, 6),
            (5, 7),
            (6, 7),
        ]),
    );
    inputs.insert("e2", map(&[(2, 3), (2, 4), (3, 4), (6, 7)]));
    inputs.insert("e3", map(&[(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)]));
    (inputs, B)
}

/// Fused-vs-kill-switch phases for one u64-relation aggregate source.
fn assert_fusion_parity_u64_keys(fix: &Fixture, source: &str, expected: &[(u64, u64)]) {
    let _guard = env_lock();
    let (inputs, _) = triangle_inputs_u64();
    let (fused, fused_count) = run_agg_program_u64(fix, source, &inputs);
    assert_eq!(fused, expected, "fused u64 path row set: {source}");
    assert_eq!(fused_count, 1, "fused dispatch must fire once: {source}");

    // SAFETY: serialized by ENV_LOCK; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (unfused, unfused_count) = run_agg_program_u64(fix, source, &inputs);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    assert_eq!(unfused, expected, "kill-switch u64 path row set: {source}");
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
}

#[test]
fn groupby_fusion_sum_z_u64_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    const B: u64 = 1 << 33;
    // Sums of Z per root (B-shifted): X=1: 3B+11; X=2: B+4; X=5: B+7.
    assert_fusion_parity_u64_keys(
        &fix,
        "agg(X, sum(Z)) :- e1(X, Y), e2(Y, Z), e3(X, Z).",
        &[(B + 1, 3 * B + 11), (B + 2, B + 4), (B + 5, B + 7)],
    );
}

#[test]
fn groupby_fusion_min_z_u64_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    const B: u64 = 1 << 33;
    // Min of Z. X=1: B+3; X=2: B+4; X=5: B+7.
    assert_fusion_parity_u64_keys(
        &fix,
        "agg(X, min(Z)) :- e1(X, Y), e2(Y, Z), e3(X, Z).",
        &[(B + 1, B + 3), (B + 2, B + 4), (B + 5, B + 7)],
    );
}

#[test]
fn groupby_fusion_max_y_u64_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    const B: u64 = 1 << 33;
    // Max of Y. X=1: B+3; X=2: B+3; X=5: B+6.
    assert_fusion_parity_u64_keys(
        &fix,
        "agg(X, max(Y)) :- e1(X, Y), e2(Y, Z), e3(X, Z).",
        &[(B + 1, B + 3), (B + 2, B + 3), (B + 5, B + 6)],
    );
}

/// Upload a binary relation with Symbol-typed columns (u32-physical).
fn upload_binary_symbol(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
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

/// Compile + execute an aggregate program over Symbol relations; returns
/// the execute_plan Result alongside the executor so callers can assert
/// either success-with-rows or matching rejection.
fn run_symbol_program(
    fix: &Fixture,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> (xlog_core::Result<()>, Executor) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile aggregate rule");
    let mut executor = Executor::new(Arc::clone(&fix.provider));
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        executor.put_relation(name, upload_binary_symbol(&fix.memory, rows));
    }
    let result = executor.execute_plan(&plan).map(|_| ());
    (result, executor)
}

/// Symbol-typed relations are u32-physical: the fused count path must fire
/// and match the kill-switch rows (count never reads the value column, so
/// Symbol keys/values are admissible).
#[test]
fn groupby_fusion_count_symbol_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let _guard = env_lock();
    let inputs = triangle_inputs();
    let expected = vec![(1u32, 3u64), (2, 1), (5, 1)];

    let (result, executor) = run_symbol_program(&fix, SOURCE, &inputs);
    result.expect("fused symbol count plan must execute");
    let deg = executor.store().get("deg").expect("deg relation");
    assert_eq!(
        deg.schema().column_type(0),
        Some(ScalarType::Symbol),
        "fused output must preserve the Symbol key type"
    );
    let fused_rows = download_group_counts(&fix.memory, deg);
    assert_eq!(fused_rows, expected, "fused symbol count row set");
    assert_eq!(
        executor.wcoj_groupby_fusion_dispatch_count(),
        1,
        "fused count over Symbol relations must fire exactly once"
    );

    // SAFETY: serialized by ENV_LOCK; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (result, executor) = run_symbol_program(&fix, SOURCE, &inputs);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    result.expect("kill-switch symbol count plan must execute");
    let deg = executor.store().get("deg").expect("deg relation");
    let unfused_rows = download_group_counts(&fix.memory, deg);
    assert_eq!(unfused_rows, expected, "kill-switch symbol count row set");
    assert_eq!(
        executor.wcoj_groupby_fusion_dispatch_count(),
        0,
        "kill switch must keep the counter at 0"
    );
}

/// Min/max/sum over Symbol VALUES is semantically questionable (symbol ids
/// carry no arithmetic order), and the unfused groupby rejects it with an
/// error. The fused path must DECLINE (counter == 0) so the query fails
/// through the same unfused rejection — never silently aggregating ids.
#[test]
fn groupby_fusion_symbol_valued_min_declines_and_unfused_rejects() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let _guard = env_lock();
    let source = "agg(X, min(Z)) :- e1(X, Y), e2(Y, Z), e3(X, Z).";
    let inputs = triangle_inputs();

    // Fusion enabled: the hook declines (Symbol value column), the unfused
    // groupby rejects, and the query errors.
    let (result, executor) = run_symbol_program(&fix, source, &inputs);
    let err = result.expect_err("min over Symbol values must be rejected");
    assert!(
        format!("{err}").contains("values"),
        "rejection must come from the groupby value-type gate, got: {err}"
    );
    assert_eq!(
        executor.wcoj_groupby_fusion_dispatch_count(),
        0,
        "fused path must decline Symbol-valued min, not dispatch it"
    );

    // Kill switch: identical rejection through the same unfused gate.
    // SAFETY: serialized by ENV_LOCK; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (result_unfused, _) = run_symbol_program(&fix, source, &inputs);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    let err_unfused =
        result_unfused.expect_err("kill-switch min over Symbol values must be rejected");
    assert_eq!(
        format!("{err}"),
        format!("{err_unfused}"),
        "fused-declined and kill-switch runs must reject identically"
    );
}

/// Shared 4-cycle fixture. Completions per root W:
/// W=1 -> (X,Y,Z) in {(10,20,30),(10,21,30),(11,20,30)};
/// W=2 -> {(10,20,30),(10,21,30)}. W=3 has e1/e2/e3 paths but no closing
/// e4 edge back to itself.
fn cycle4_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    inputs.insert("e1", vec![(1, 10), (1, 11), (2, 10), (3, 12)]);
    inputs.insert("e2", vec![(10, 20), (10, 21), (11, 20), (12, 22)]);
    inputs.insert("e3", vec![(20, 30), (21, 30), (22, 31)]);
    inputs.insert("e4", vec![(30, 1), (30, 2), (31, 9)]);
    inputs
}

#[test]
fn groupby_fusion_4cycle_count_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let _guard = env_lock();
    let source = "agg(W, count(Z)) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).";
    let inputs = cycle4_inputs();
    let expected = vec![(1u32, 3u64), (2, 2)];

    // Phase 1: fused 4-cycle count path fires and is correct.
    let (fused_rows, fused_count) =
        run_agg_program(&fix, source, &inputs, download_group_counts);
    assert_eq!(fused_rows, expected, "fused 4-cycle count row set");
    assert_eq!(
        fused_count, 1,
        "fused 4-cycle group-by-root count dispatch must fire exactly once"
    );

    // Phase 2: kill switch forces the materialize+groupby path.
    // SAFETY: serialized by ENV_LOCK; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (unfused_rows, unfused_count) =
        run_agg_program(&fix, source, &inputs, download_group_counts);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    assert_eq!(unfused_rows, expected, "kill-switch 4-cycle count row set");
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
}

/// S1d fused-vs-kill-switch phases for one u64-valued 4-cycle aggregate
/// source (sum).
fn assert_4cycle_fusion_parity_u64(fix: &Fixture, source: &str, expected: &[(u32, u64)]) {
    let _guard = env_lock();
    let inputs = cycle4_inputs();
    let (fused, fused_count) = run_agg_program(fix, source, &inputs, download_group_counts);
    assert_eq!(fused, expected, "fused path row set: {source}");
    assert_eq!(fused_count, 1, "fused dispatch must fire once: {source}");

    // SAFETY: serialized by ENV_LOCK; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (unfused, unfused_count) = run_agg_program(fix, source, &inputs, download_group_counts);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    assert_eq!(unfused, expected, "kill-switch path row set: {source}");
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
}

/// S1d fused-vs-kill-switch phases for one u32-valued 4-cycle aggregate
/// source (min/max).
fn assert_4cycle_fusion_parity_u32(fix: &Fixture, source: &str, expected: &[(u32, u32)]) {
    let _guard = env_lock();
    let inputs = cycle4_inputs();
    let (fused, fused_count) = run_agg_program(fix, source, &inputs, download_groups_u32);
    assert_eq!(fused, expected, "fused path row set: {source}");
    assert_eq!(fused_count, 1, "fused dispatch must fire once: {source}");

    // SAFETY: serialized by ENV_LOCK; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (unfused, unfused_count) = run_agg_program(fix, source, &inputs, download_groups_u32);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    assert_eq!(unfused, expected, "kill-switch path row set: {source}");
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
}

#[test]
fn groupby_fusion_4cycle_sum_z_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // W=1: 30+30+30 = 90; W=2: 30+30 = 60.
    assert_4cycle_fusion_parity_u64(
        &fix,
        "agg(W, sum(Z)) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).",
        &[(1u32, 90u64), (2, 60)],
    );
}

#[test]
fn groupby_fusion_4cycle_sum_x_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // W=1: 10+10+11 = 31; W=2: 10+10 = 20.
    assert_4cycle_fusion_parity_u64(
        &fix,
        "agg(W, sum(X)) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).",
        &[(1u32, 31u64), (2, 20)],
    );
}

#[test]
fn groupby_fusion_4cycle_min_x_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // W=1: min(10, 10, 11) = 10; W=2: min(10, 10) = 10.
    assert_4cycle_fusion_parity_u32(
        &fix,
        "agg(W, min(X)) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).",
        &[(1u32, 10u32), (2, 10)],
    );
}

#[test]
fn groupby_fusion_4cycle_max_y_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // W=1: max(20, 21, 20) = 21; W=2: max(20, 21) = 21.
    assert_4cycle_fusion_parity_u32(
        &fix,
        "agg(W, max(Y)) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).",
        &[(1u32, 21u32), (2, 21)],
    );
}

/// S1d Symbol lock, 4-cycle sibling of
/// `groupby_fusion_symbol_valued_min_declines_and_unfused_rejects`: min
/// over Symbol VALUES on a 4-cycle body must DECLINE fused (counter == 0)
/// and fail through the same unfused value-type rejection in fused-enabled
/// and kill-switch runs alike.
#[test]
fn groupby_fusion_4cycle_symbol_valued_min_declines_and_unfused_rejects() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let _guard = env_lock();
    let source = "agg(W, min(Z)) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).";
    let inputs = cycle4_inputs();

    let (result, executor) = run_symbol_program(&fix, source, &inputs);
    let err = result.expect_err("4-cycle min over Symbol values must be rejected");
    assert!(
        format!("{err}").contains("values"),
        "rejection must come from the groupby value-type gate, got: {err}"
    );
    assert_eq!(
        executor.wcoj_groupby_fusion_dispatch_count(),
        0,
        "fused path must decline Symbol-valued 4-cycle min, not dispatch it"
    );

    // Kill switch: identical rejection through the same unfused gate.
    // SAFETY: serialized by ENV_LOCK; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (result_unfused, _) = run_symbol_program(&fix, source, &inputs);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    let err_unfused =
        result_unfused.expect_err("kill-switch 4-cycle min over Symbol values must be rejected");
    assert_eq!(
        format!("{err}"),
        format!("{err_unfused}"),
        "fused-declined and kill-switch runs must reject identically"
    );
}

/// S1d slice 2 — u64-key 4-cycle count fusion: same 4-cycle fixture
/// shifted above 2^33 so width truncation visibly fails.
#[test]
fn groupby_fusion_4cycle_count_u64_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let _guard = env_lock();
    let source = "agg(W, count(Z)) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).";

    const B: u64 = 1 << 33;
    let mut inputs: BTreeMap<&str, Vec<(u64, u64)>> = BTreeMap::new();
    for (name, rows) in cycle4_inputs() {
        inputs.insert(
            name,
            rows.iter()
                .map(|&(a, b)| (B + a as u64, B + b as u64))
                .collect(),
        );
    }
    let expected = vec![(B + 1, 3u64), (B + 2, 2)];

    // Phase 1: fused u64 4-cycle path fires and is correct.
    let (fused_rows, fused_count) = run_agg_program_u64(&fix, source, &inputs);
    assert_eq!(fused_rows, expected, "fused u64 4-cycle count row set");
    assert_eq!(
        fused_count, 1,
        "fused u64 4-cycle group-by-root count dispatch must fire exactly once"
    );

    // Phase 2: kill switch forces the materialize+groupby path.
    // SAFETY: serialized by ENV_LOCK; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (unfused_rows, unfused_count) = run_agg_program_u64(&fix, source, &inputs);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    assert_eq!(
        unfused_rows, expected,
        "kill-switch u64 4-cycle count row set"
    );
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
}

#[test]
fn groupby_fusion_sum_declines_non_triangle_shape() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // 2-atom chain body: not a triangle; sum fusion must decline silently
    // and the standard path must produce the correct sums.
    let source = "agg(X, sum(Z)) :- e1(X, Y), e2(Y, Z).";
    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    inputs.insert("e1", vec![(1, 10), (1, 11), (2, 10)]);
    inputs.insert("e2", vec![(10, 100), (10, 101), (11, 100)]);
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile chain aggregate");
    let mut executor = Executor::new(Arc::clone(&fix.provider));
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in &inputs {
        executor.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    executor.execute_plan(&plan).expect("execute plan");
    let agg = executor.store().get("agg").expect("agg relation");
    let rows = download_group_counts(&fix.memory, agg);
    // X=1: 100+101+100 = 301; X=2: 100+101 = 201.
    assert_eq!(rows, vec![(1u32, 301u64), (2, 201)], "chain sum rows");
    assert_eq!(
        executor.wcoj_groupby_fusion_dispatch_count(),
        0,
        "non-triangle shape must not consume the fusion counter"
    );
}

#[test]
fn groupby_fusion_declines_non_triangle_shape() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // 2-atom chain body: not a triangle; fusion must decline silently and
    // the standard path must produce the correct counts.
    let source = "deg(X, count(Z)) :- e1(X, Y), e2(Y, Z).";
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile chain aggregate");
    let mut executor = Executor::new(Arc::clone(&fix.provider));
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "e1",
        upload_binary_u32(&fix.memory, &[(1, 10), (1, 11), (2, 10)]),
    );
    executor.put_relation(
        "e2",
        upload_binary_u32(&fix.memory, &[(10, 100), (10, 101), (11, 100)]),
    );
    executor.execute_plan(&plan).expect("execute plan");
    let deg = executor.store().get("deg").expect("deg relation");
    let rows = download_group_counts(&fix.memory, deg);
    assert_eq!(rows, vec![(1u32, 3u64), (2, 2)], "chain aggregate rows");
    assert_eq!(
        executor.wcoj_groupby_fusion_dispatch_count(),
        0,
        "non-triangle shape must not consume the fusion counter"
    );
}
