// crates/xlog-cuda/tests/test_mt_sort_hj_alloc_ordering.rs
//! Multi-threaded regression for the cross-thread cert-mode flake
//! exposed by `XLOG_USE_RECORDED_OPS=1 XLOG_USE_DEVICE_RUNTIME=1
//! cargo test -p xlog-integration --test real_world_tests --
//! --test-threads=8` — observed at ~98% pass / ~4% per-iter
//! fail rate with `+SORT+HJ` as the minimum env reproducer.
//!
//! # Hypothesis under test
//!
//! `LaunchRecorder::record_block_use` records a `cu_event` on
//! `use_stream` for **dealloc-time** safety (alloc_stream waits
//! for the recorded event before queueing `cuMemFreeAsync`).
//! It does NOT add the symmetric wait — `use_stream` does NOT
//! wait on the block's `alloc_stream` event before the queued
//! kernel begins. Recorded-path runtime allocations are routed
//! through `StreamId::DEFAULT` (the cudarc default stream of
//! each thread's `CudaDevice`); recorded operators launch on a
//! non-default `recorded_op_stream`. Single-threaded runs hide
//! this because pack/sort paths route through
//! `htod_sync_copy_into` whose host-blocking semantics
//! accidentally serialize the alloc-stream allocation before
//! the launch-stream kernel. Multi-threaded runs lose that
//! accidental serialization once threads contend on
//! `cuMemAllocAsync`.
//!
//! # Test shape
//!
//! `mt_sort_hj_recorded_no_prewarm` — N threads, each builds
//! its own runtime stack (`AsyncCudaResource → LoggingResource →
//! GlobalDeviceBudget`), provider, non-default launch_stream,
//! and runs `sort_recorded(friend, &[1]) →
//! hash_join_v2_recorded(sorted, friend, &[1], &[0], Inner)`
//! repeatedly. Result is read once via `download_column::<u32>`
//! at the END for verification; no intermediate D2H sits
//! between sort and join (an intermediate D2H would
//! accidentally synchronize and mask the bug).
//!
//! `mt_sort_hj_recorded_with_prewarm` — same shape, but every
//! thread first runs one full sort+HJ serially (in the calling
//! thread, before spawning) so kernel modules are already
//! loaded into the CUDA primary context. If failure
//! disappears only with prewarm, cudarc's lazy module-load is
//! implicated; otherwise the failure points at stream/memory
//! ordering — exactly the alloc/use stream-wait gap.

use std::sync::Arc;

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager, JoinType};

const TEST_BUDGET_BYTES: usize = 512 * 1024 * 1024;
const N_THREADS: usize = 8;
const ITERS_PER_THREAD: usize = 128;
const ROUNDS_PER_ITER: usize = 3;

/// Synthetic friend-of-friend workload. Bidirectional cliques
/// + cross-cluster bridges so the self-join produces dense
/// matches on multiple key values (worst case for
/// alloc-pressure on the build table side of the join).
fn make_friends() -> Vec<(u32, u32)> {
    let mut edges = Vec::new();
    const CLUSTERS: u32 = 8;
    const PER_CLUSTER: u32 = 16;
    for c in 0..CLUSTERS {
        let base = c * PER_CLUSTER + 1;
        for i in 0..PER_CLUSTER {
            for j in 0..PER_CLUSTER {
                if i != j {
                    edges.push((base + i, base + j));
                }
            }
        }
    }
    for c in 0..CLUSTERS {
        let from = c * PER_CLUSTER + 1;
        let to = ((c + 1) % CLUSTERS) * PER_CLUSTER + 1;
        edges.push((from, to));
    }
    edges
}

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

/// Builds the per-thread runtime + provider stack identical in
/// shape to what `xlog-integration::real_world_tests` uses
/// under `XLOG_USE_DEVICE_RUNTIME=1`.
fn build_provider() -> Option<(
    Arc<CudaKernelProvider>,
    Arc<XlogDeviceRuntime>,
    Arc<StreamPool>,
)> {
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
        Box::new(GlobalDeviceBudget::new(logging, TEST_BUDGET_BYTES));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(TEST_BUDGET_BYTES as u64),
        Arc::clone(&runtime),
    ));
    let provider = Arc::new(
        CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?,
    );
    Some((provider, runtime, pool))
}

fn build_friend_buffer(provider: &CudaKernelProvider, edges: &[(u32, u32)]) -> CudaBuffer {
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    let col0: Vec<u8> = edges.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let col1: Vec<u8> = edges.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    provider
        .create_buffer_from_slices(&[&col0, &col1], schema)
        .expect("create friend buffer")
}

/// Compute the expected `friend ⨝ friend on (col1 == col0)`
/// multiset host-side. Result rows are 4-tuples
/// `(X, Y, Y, Z)` because the join concatenates the two
/// schemas without projection — the recorded inner join
/// returns the full 4-column row.
fn expected_self_join(edges: &[(u32, u32)]) -> std::collections::HashMap<(u32, u32, u32, u32), usize>
{
    let mut out: std::collections::HashMap<(u32, u32, u32, u32), usize> = Default::default();
    for &(x, y) in edges {
        for &(yp, z) in edges {
            if y == yp {
                *out.entry((x, y, yp, z)).or_insert(0) += 1;
            }
        }
    }
    out
}

/// One iteration of the failing chain on the calling thread.
/// `launch_stream` is acquired once per thread by the caller
/// and reused across iterations — same shape as production
/// `recorded_op_stream_or_init`. Each iteration does
/// `ROUNDS_PER_ITER` chained sort+HJ pairs to amplify the
/// transient-buffer pressure that the integration tests'
/// recursive eval applies.
fn run_one_iter(
    provider: &CudaKernelProvider,
    runtime: &XlogDeviceRuntime,
    pool: &StreamPool,
    launch_stream: StreamId,
    edges: &[(u32, u32)],
    expected: &std::collections::HashMap<(u32, u32, u32, u32), usize>,
) -> Result<(), String> {
    let friend = build_friend_buffer(provider, edges);

    for round in 0..ROUNDS_PER_ITER {
        // Sort by col1 (the join key on the build side).
        let sorted = provider
            .sort_recorded(&friend, &[1], launch_stream)
            .map_err(|e| format!("round {} sort_recorded: {}", round, e))?;

        // Self-join: `sorted` (left) ⨝ `friend` (right) on
        // `sorted.col1 == friend.col0`.
        //
        // CRITICAL: no intermediate D2H here. `sorted` is
        // handed directly to the join. An intermediate
        // `download_column` / row-count read would
        // host-synchronize and accidentally serialize alloc
        // → launch, masking the bug.
        let result = provider
            .hash_join_v2_recorded(
                &sorted,
                &friend,
                &[1],
                &[0],
                JoinType::Inner,
                None,
                launch_stream,
            )
            .map_err(|e| format!("round {} hash_join_v2_recorded: {}", round, e))?;

        // Final D2H read after BOTH ops. Sync the
        // launch_stream first so all queued work has
        // completed.
        let cu_stream = pool
            .resolve(launch_stream)
            .ok_or_else(|| "resolve launch_stream".to_string())?;
        cu_stream
            .synchronize()
            .map_err(|e| format!("round {} sync launch_stream: {}", round, e))?;

        let n = result.num_rows() as usize;
        let c0 = provider
            .download_column::<u32>(&result, 0)
            .map_err(|e| format!("round {} dtoh c0: {}", round, e))?;
        let c1 = provider
            .download_column::<u32>(&result, 1)
            .map_err(|e| format!("round {} dtoh c1: {}", round, e))?;
        let c2 = provider
            .download_column::<u32>(&result, 2)
            .map_err(|e| format!("round {} dtoh c2: {}", round, e))?;
        let c3 = provider
            .download_column::<u32>(&result, 3)
            .map_err(|e| format!("round {} dtoh c3: {}", round, e))?;
        if c0.len() != n || c1.len() != n || c2.len() != n || c3.len() != n {
            return Err(format!(
                "round {} dtoh length mismatch n={} ({}, {}, {}, {})",
                round,
                n,
                c0.len(),
                c1.len(),
                c2.len(),
                c3.len()
            ));
        }

        let mut observed: std::collections::HashMap<(u32, u32, u32, u32), usize> =
            Default::default();
        for i in 0..n {
            *observed.entry((c0[i], c1[i], c2[i], c3[i])).or_insert(0) += 1;
        }
        if &observed != expected {
            return Err(format!(
                "round {} result-set mismatch: expected {} rows / {} distinct, \
                 observed {} rows / {} distinct",
                round,
                expected.values().sum::<usize>(),
                expected.len(),
                n,
                observed.len(),
            ));
        }

        drop(sorted);
        drop(result);
    }

    drop(friend);
    runtime.reap_pending().map_err(|e| format!("reap: {}", e))?;
    Ok(())
}

fn run_threaded(prewarm: bool) -> (usize, Vec<String>) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let edges = make_friends();
    let expected = expected_self_join(&edges);
    if prewarm {
        let Some((provider, runtime, pool)) = build_provider() else {
            eprintln!("Skipping: no CUDA device for prewarm");
            return (0, vec!["skip".to_string()]);
        };
        let launch_stream = pool.acquire().expect("acquire prewarm launch_stream");
        // One full sort+HJ on the calling thread to load every
        // kernel module into the CUDA primary context.
        run_one_iter(&provider, &runtime, &pool, launch_stream, &edges, &expected)
            .expect("prewarm iter must pass");
    }

    let pass = Arc::new(AtomicUsize::new(0));
    let fails = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::with_capacity(N_THREADS);
    for tid in 0..N_THREADS {
        let pass = Arc::clone(&pass);
        let fails = Arc::clone(&fails);
        let edges_t = edges.clone();
        let expected_t = expected.clone();
        handles.push(std::thread::spawn(move || {
            let Some((provider, runtime, pool)) = build_provider() else {
                eprintln!("Skipping thread {}: no CUDA device", tid);
                return;
            };
            let launch_stream = pool.acquire().expect("acquire per-thread launch_stream");
            for it in 0..ITERS_PER_THREAD {
                match run_one_iter(
                    &provider,
                    &runtime,
                    &pool,
                    launch_stream,
                    &edges_t,
                    &expected_t,
                ) {
                    Ok(_) => {
                        pass.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(msg) => {
                        let mut f = fails.lock().unwrap();
                        if f.len() < 5 {
                            f.push(format!("[t{} it{}] {}", tid, it, msg));
                        }
                    }
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("thread join");
    }
    (
        pass.load(Ordering::Relaxed),
        Arc::try_unwrap(fails)
            .map(|m| m.into_inner().unwrap())
            .unwrap_or_default(),
    )
}

#[test]
fn mt_sort_hj_recorded_no_prewarm() {
    if CudaDevice::new(0).is_err() {
        eprintln!("Skipping: no CUDA device");
        return;
    }
    let total = N_THREADS * ITERS_PER_THREAD;
    let (pass, fails) = run_threaded(false);
    eprintln!(
        "[mt_sort_hj no_prewarm] pass={}/{}  fails={:?}",
        pass, total, fails
    );
    assert!(
        pass == total,
        "{}/{} sort+HJ iterations failed across {} threads (no prewarm). \
         Sample failures: {:?}",
        total - pass,
        total,
        N_THREADS,
        fails,
    );
}

#[test]
fn mt_sort_hj_recorded_with_prewarm() {
    if CudaDevice::new(0).is_err() {
        eprintln!("Skipping: no CUDA device");
        return;
    }
    let total = N_THREADS * ITERS_PER_THREAD;
    let (pass, fails) = run_threaded(true);
    eprintln!(
        "[mt_sort_hj with_prewarm] pass={}/{}  fails={:?}",
        pass, total, fails
    );
    assert!(
        pass == total,
        "{}/{} sort+HJ iterations failed across {} threads (with prewarm). \
         Sample failures: {:?}",
        total - pass,
        total,
        N_THREADS,
        fails,
    );
}
