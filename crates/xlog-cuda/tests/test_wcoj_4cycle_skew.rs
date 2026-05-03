// crates/xlog-cuda/tests/test_wcoj_4cycle_skew.rs
//! v0.6.5 slice 2 — 4-cycle skew classifier provider tests.
//!
//! Exercises `wcoj_4cycle_skew_score_u32` and `_u64` directly with
//! per-axis-skewed fixtures. Each test concentrates rows on the
//! lookup-key column of one specific join axis and asserts the
//! classifier reports a high score; uniform fixtures should score
//! low.
//!
//! Together with `test_wcoj_4cycle_adaptive_dispatch.rs` (which
//! exercises the dispatch-decision integration), this pins both
//! ends of the classifier contract:
//!
//!   * Each of the four lookup-key columns (e1.col0, e2.col0,
//!     e3.col0, e4.col0) is independently skew-detected.
//!   * Skew on ANY ONE axis is sufficient to clear the threshold —
//!     the reduction is `max(score_per_axis)`.

use std::sync::Arc;

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

const SKEW_THRESHOLD: f64 = 0.10;

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
    provider: CudaKernelProvider,
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
        CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?;
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

/// Build a 2-column relation where `col0` is heavily concentrated
/// on a single value (skew_target appears ≥ 90% of the time) and
/// `col1` is uniformly distributed. This isolates skew on col0.
fn col0_skewed_fixture(n: usize, skew_target: u32) -> Vec<(u32, u32)> {
    let mut rows = Vec::with_capacity(n);
    let skewed = (n * 9) / 10; // 90% on the hub
    for i in 0..skewed {
        rows.push((skew_target, 1000 + i as u32));
    }
    for i in skewed..n {
        rows.push((10000 + i as u32, 20000 + i as u32));
    }
    rows.sort();
    rows.dedup();
    rows
}

/// Uniform 2-column relation: no axis is concentrated. col0 ranges
/// over 1..=N, col1 over 1..=N with no repeated col0 values.
fn uniform_fixture(n: usize) -> Vec<(u32, u32)> {
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        rows.push(((i + 1) as u32, ((i * 7) % n + 1) as u32));
    }
    rows.sort();
    rows.dedup();
    rows
}

fn make_skewed_axis_fixtures(skewed_axis: usize, n: usize) -> [Vec<(u32, u32)>; 4] {
    let mut sets = [
        uniform_fixture(n),
        uniform_fixture(n),
        uniform_fixture(n),
        uniform_fixture(n),
    ];
    sets[skewed_axis] = col0_skewed_fixture(n, 1);
    sets
}

fn skew_score(fix: &RuntimeFixture, sets: &[Vec<(u32, u32)>; 4]) -> f64 {
    let buf_e1 = upload_binary_u32(&fix.memory, &sets[0]);
    let buf_e2 = upload_binary_u32(&fix.memory, &sets[1]);
    let buf_e3 = upload_binary_u32(&fix.memory, &sets[2]);
    let buf_e4 = upload_binary_u32(&fix.memory, &sets[3]);
    let launch_stream = fix.pool.acquire().expect("acquire launch_stream");
    fix.provider
        .wcoj_4cycle_skew_score_u32(&buf_e1, &buf_e2, &buf_e3, &buf_e4, launch_stream)
        .expect("skew_score must succeed")
        .expect("skew_score returns Some on valid inputs")
}

#[test]
fn skew_detected_on_e1_col0() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let sets = make_skewed_axis_fixtures(0, 200);
    let s = skew_score(&fix, &sets);
    assert!(
        s >= SKEW_THRESHOLD,
        "e1.col0 skew must clear threshold; got {}",
        s
    );
}

#[test]
fn skew_detected_on_e2_col0() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let sets = make_skewed_axis_fixtures(1, 200);
    let s = skew_score(&fix, &sets);
    assert!(
        s >= SKEW_THRESHOLD,
        "e2.col0 skew must clear threshold; got {}",
        s
    );
}

#[test]
fn skew_detected_on_e3_col0() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let sets = make_skewed_axis_fixtures(2, 200);
    let s = skew_score(&fix, &sets);
    assert!(
        s >= SKEW_THRESHOLD,
        "e3.col0 skew must clear threshold; got {}",
        s
    );
}

#[test]
fn skew_detected_on_e4_col0() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let sets = make_skewed_axis_fixtures(3, 200);
    let s = skew_score(&fix, &sets);
    assert!(
        s >= SKEW_THRESHOLD,
        "e4.col0 skew must clear threshold; got {}",
        s
    );
}

#[test]
fn uniform_inputs_score_below_threshold() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let sets = [
        uniform_fixture(200),
        uniform_fixture(200),
        uniform_fixture(200),
        uniform_fixture(200),
    ];
    let s = skew_score(&fix, &sets);
    assert!(
        s < SKEW_THRESHOLD,
        "uniform inputs must score below threshold; got {}",
        s
    );
}
