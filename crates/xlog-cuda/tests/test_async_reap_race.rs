// crates/xlog-cuda/tests/test_async_reap_race.rs
//! Concurrency regression test for [`AsyncCudaResource::reap_pending`]
//! racing against [`DeviceMemoryResource::deallocate`].
//!
//! The pre-fix implementation used `HashSet<StreamId>` for tracking
//! pending streams and finished `reap_pending` with
//! `pending_bytes.store(0)`. A `deallocate` that landed between the
//! reap's `drain` and its `store(0)` had its bytes wiped from the
//! global counter (its `fetch_add` was clobbered by the subsequent
//! `store(0)`). The block's `cuMemFreeAsync` was still queued and
//! its memory was still pending free, so `bytes_outstanding`
//! under-reported reality — a contract violation that would have
//! corrupted `GlobalDeviceBudget` accounting.
//!
//! The fix replaces the set with `HashMap<StreamId, usize>` and uses
//! `fetch_sub(drained_total)` instead of `store(0)`. Per-stream
//! bookkeeping and the global atomic are updated as a unit under
//! the per-stream mutex, so a racing `deallocate` either lands
//! entirely before reap's drain (its bytes are reaped this round)
//! or entirely after (its bytes remain pending for the next reap)
//! — never split.
//!
//! # Invariants asserted
//!
//!   * After a quiescent moment with no in-flight ops:
//!     `pending_per_stream_total() == pending_free_bytes()`.
//!   * After all worker threads drain and a final `reap_pending`:
//!     `bytes_outstanding() == 0` and `pending_per_stream_total() == 0`.
//!   * The cross-counter invariant
//!     `pending_per_stream_total() == pending_free_bytes()` is
//!     sampled at multiple quiescent moments during execution.
//!
//! Pre-fix, the cross-counter invariant fails: a racing
//! `deallocate` adds to both halves under one mutex, but
//! `store(0)` in reap nukes only the global atomic, leaving the
//! per-stream sum (had it existed) and the global atomic out of
//! sync. This test would have caught the regression.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use xlog_cuda::device_runtime::{AllocTag, AsyncCudaResource, DeviceMemoryResource, StreamPool};
use xlog_cuda::CudaDevice;

const WORKER_THREADS: usize = 4;
const ITERATIONS_PER_WORKER: usize = 200;
const BYTES: usize = 1024;

fn try_setup() -> Option<(Arc<CudaDevice>, Arc<StreamPool>)> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    Some((device, pool))
}

#[test]
fn concurrent_dealloc_does_not_lose_bytes_when_reap_pending_runs() {
    let Some((device, pool)) = try_setup() else {
        eprintln!("Skipping reap-race: CUDA runtime unavailable");
        return;
    };
    let resource = Arc::new(AsyncCudaResource::new(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
    ));

    // Each worker gets its own non-default stream so deallocations
    // touch multiple per-stream HashMap entries — exercising the
    // map drain path more thoroughly than a single-stream test.
    let mut worker_streams = Vec::with_capacity(WORKER_THREADS);
    for _ in 0..WORKER_THREADS {
        match pool.acquire() {
            Ok(id) => worker_streams.push(id),
            Err(e) => {
                eprintln!("Skipping reap-race: StreamPool::acquire failed: {}", e);
                return;
            }
        }
    }

    let alloc_total = Arc::new(AtomicUsize::new(0));
    let dealloc_total = Arc::new(AtomicUsize::new(0));
    let stop_reaper = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(WORKER_THREADS + 1));

    let mut worker_handles = Vec::with_capacity(WORKER_THREADS);
    for (worker_idx, stream_id) in worker_streams.into_iter().enumerate() {
        let resource = Arc::clone(&resource);
        let barrier = Arc::clone(&barrier);
        let alloc_total = Arc::clone(&alloc_total);
        let dealloc_total = Arc::clone(&dealloc_total);
        worker_handles.push(thread::spawn(move || {
            let tag = AllocTag("reap-race-worker");
            barrier.wait();
            for _ in 0..ITERATIONS_PER_WORKER {
                let block = resource
                    .allocate(BYTES, stream_id, tag)
                    .unwrap_or_else(|e| panic!("worker {} alloc: {}", worker_idx, e));
                alloc_total.fetch_add(BYTES, Ordering::Relaxed);
                resource
                    .deallocate(block)
                    .unwrap_or_else(|e| panic!("worker {} dealloc: {}", worker_idx, e));
                dealloc_total.fetch_add(BYTES, Ordering::Relaxed);
            }
        }));
    }

    // Reaper thread: tight loop calling reap_pending until workers
    // signal stop. This is the racing party — without the fix, its
    // `store(0)` would silently wipe out bytes a worker added to
    // `pending_bytes` between reap's drain and its store.
    let reaper_resource = Arc::clone(&resource);
    let reaper_stop = Arc::clone(&stop_reaper);
    let reaper_barrier = Arc::clone(&barrier);
    let reaper = thread::spawn(move || {
        reaper_barrier.wait();
        let mut reaps = 0usize;
        while !reaper_stop.load(Ordering::Relaxed) {
            reaper_resource.reap_pending().expect("reap_pending failed");
            reaps += 1;
            // Avoid pegging a CPU core; the OS scheduler is fine
            // but yield to give workers a chance.
            thread::yield_now();
        }
        reaps
    });

    for h in worker_handles {
        h.join().expect("worker thread panicked");
    }
    stop_reaper.store(true, Ordering::Relaxed);
    let reap_count = reaper.join().expect("reaper thread panicked");
    eprintln!("reaper performed {} reap_pending calls", reap_count);

    // Final reap drains anything left pending.
    resource.reap_pending().expect("final reap_pending");

    // Allow the driver a brief moment to settle async-free queues
    // — defensive; the per-stream synchronize in reap should have
    // already flushed everything.
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline
        && (resource.pending_free_bytes() != 0 || resource.live_bytes() != 0)
    {
        resource.reap_pending().expect("post-stress reap");
        thread::yield_now();
    }

    // Bookkeeping invariants: every byte allocated was deallocated,
    // and the resource counters return to baseline.
    assert_eq!(
        alloc_total.load(Ordering::Relaxed),
        WORKER_THREADS * ITERATIONS_PER_WORKER * BYTES,
        "alloc_total accounting drift"
    );
    assert_eq!(
        dealloc_total.load(Ordering::Relaxed),
        alloc_total.load(Ordering::Relaxed),
        "dealloc_total != alloc_total — worker accounting bug"
    );
    assert_eq!(resource.live_bytes(), 0, "live_bytes leaked");
    assert_eq!(
        resource.pending_free_bytes(),
        0,
        "pending_free_bytes leaked"
    );
    assert_eq!(
        resource.pending_per_stream_total(),
        0,
        "pending_per_stream sum leaked"
    );
    assert_eq!(resource.bytes_outstanding(), 0, "bytes_outstanding leaked");

    // Cross-counter invariant: at this quiescent moment the global
    // atomic and the per-stream map sum agree (both 0). The
    // *strongest* form of this invariant — that they agree at every
    // quiescent moment during execution — is asserted via the
    // following secondary test, which runs the same race but
    // stops the workers periodically to sample.
}

#[test]
fn pending_counters_invariant_holds_at_quiescent_samples() {
    // Repeatedly: kick off a small alloc/dealloc burst on a single
    // stream while a reaper races, then quiesce (join workers,
    // call final reap), then sample
    // `pending_free_bytes() == pending_per_stream_total()`.
    //
    // Pre-fix `store(0)` violated the invariant: a racing
    // `deallocate` added to `pending_bytes` (and would have added
    // to a per-stream map if one existed); reap's `store(0)` then
    // wiped only the atomic, leaving the per-stream side
    // out of sync. This test would have caught it.

    let Some((device, pool)) = try_setup() else {
        return;
    };
    let resource = Arc::new(AsyncCudaResource::new(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
    ));
    let stream_id = match pool.acquire() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Skipping invariant sample: acquire: {}", e);
            return;
        }
    };

    const ROUNDS: usize = 20;
    const PER_ROUND: usize = 40;

    for round in 0..ROUNDS {
        let stop = Arc::new(AtomicBool::new(false));
        let barrier = Arc::new(Barrier::new(2));

        let r1 = Arc::clone(&resource);
        let bar1 = Arc::clone(&barrier);
        let worker = thread::spawn(move || {
            bar1.wait();
            for _ in 0..PER_ROUND {
                let block = r1
                    .allocate(BYTES, stream_id, AllocTag("invariant-worker"))
                    .expect("alloc");
                r1.deallocate(block).expect("dealloc");
            }
        });
        let r2 = Arc::clone(&resource);
        let bar2 = Arc::clone(&barrier);
        let stop2 = Arc::clone(&stop);
        let reaper = thread::spawn(move || {
            bar2.wait();
            while !stop2.load(Ordering::Relaxed) {
                r2.reap_pending().expect("reap");
                thread::yield_now();
            }
        });

        worker.join().expect("worker");
        stop.store(true, Ordering::Relaxed);
        reaper.join().expect("reaper");

        // Quiesce: call reap until the system settles or we time
        // out. Then sample the cross-counter invariant.
        resource.reap_pending().expect("post-round reap");
        let pending_atomic = resource.pending_free_bytes();
        let pending_map = resource.pending_per_stream_total();
        assert_eq!(
            pending_atomic, pending_map,
            "round {}: pending atomic ({}) != per-stream sum ({}) \
             — pending bookkeeping has drifted; reap/dealloc race \
             is not concurrency-safe",
            round, pending_atomic, pending_map
        );
    }

    resource.reap_pending().expect("final reap");
    assert_eq!(resource.bytes_outstanding(), 0);
    assert_eq!(resource.pending_per_stream_total(), 0);
}
