// crates/xlog-cuda/tests/test_runtime_a2_stream_lifetime.rs
//! Acceptance gate **A2** for the v0.6 device-runtime allocator.
//!
//! Encodes the **stream-ordered alloc/free/reuse contract** that the
//! async backend (`AsyncCudaResource`) is required to honor:
//!
//!   * `allocate(.., S, ..)` is ordered on stream S.
//!   * `deallocate(block)` is ordered on `block.alloc_stream` (same S).
//!   * A subsequent `allocate(.., S, ..)` may reuse the underlying
//!     byte address, but only **after** the queued free completes on
//!     S. Reuse is safe with respect to writes queued on S between
//!     the original allocate and the deallocate, because the driver's
//!     stream-ordered memory allocator semantics make those writes
//!     happen-before the free, and the free happen-before the new
//!     allocation, on stream S.
//!
//! Test shape (single non-default stream, no host sync between
//! phases until the final readback):
//!
//!   1. Allocate block A on stream S.
//!   2. Queue async HtoD write of `pattern_a` to A on S.
//!   3. Deallocate A — queues `cuMemFreeAsync` on S, ordered after
//!      the write in phase 2.
//!   4. Allocate block B on S — may receive the same byte address as
//!      A. The new alloc is ordered after the queued free on S.
//!   5. Queue async HtoD write of `pattern_b` to B on S.
//!   6. `cuStreamSynchronize(S)` — only host wait in the test.
//!   7. Read back B via synchronous `cuMemcpyDtoH_v2`. Must equal
//!      `pattern_b` byte-for-byte.
//!
//! Failure modes this catches:
//!
//!   * Backend treats `deallocate` as a synchronous `cuMemFree` while
//!     work is still queued on S — driver may error, or the address
//!     reuse races the unfinished phase-2 write so the readback in
//!     phase 7 contains pattern_a bytes.
//!   * Backend ignores the caller-supplied `StreamId` and routes the
//!     async alloc/free onto the default stream — the queued writes
//!     and the alloc/free no longer share an ordering, and the
//!     readback can mix patterns.
//!
//! What this test does **not** prove:
//!
//!   * Cross-stream reuse safety. A block allocated on S1 and used
//!     by a kernel on S2 still requires explicit event/sync from the
//!     caller before deallocation. That contract is documented but
//!     not yet exercised here — separate test once we have a kernel
//!     to launch on the second stream.
//!   * Behavior when the CUDA context lacks async-alloc support
//!     (`has_async_alloc == false`). cudarc falls back to synchronous
//!     `cuMemAlloc`/`cuMemFree` in that case, and this test will
//!     still pass by serialization rather than by stream ordering —
//!     A2 is necessary but not sufficient for production async-alloc
//!     correctness on hosts without async-alloc support. M1
//!     (Compute-Sanitizer) is the manual gate that catches that
//!     class of regression.

use std::sync::Arc;

use cudarc::driver::sys;
use xlog_cuda::device_runtime::{
    AllocTag, AsyncCudaResource, DeviceMemoryResource, StreamId, StreamPool,
};
use xlog_cuda::CudaDevice;

const BYTES: usize = 4096;
const REUSE_ITERATIONS: usize = 32;

/// Setup helper. Returns `None` if no CUDA device is available so the
/// test skips cleanly on CPU-only CI.
fn try_setup() -> Option<(Arc<CudaDevice>, Arc<StreamPool>)> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    Some((device, pool))
}

/// Async HtoD copy on a specific stream. Wraps the raw cudarc sys call
/// so the test reads as ordering intent, not memcpy plumbing.
unsafe fn htod_async(stream: sys::CUstream, dst: u64, src: &[u8]) {
    let res = sys::cuMemcpyHtoDAsync_v2(dst, src.as_ptr() as *const _, src.len(), stream);
    assert_eq!(
        res,
        sys::cudaError_enum::CUDA_SUCCESS,
        "cuMemcpyHtoDAsync_v2 returned {:?}",
        res
    );
}

/// Synchronous DtoH copy. Used only for the final readback in
/// step 7 — by then the stream has been synchronized so a sync copy
/// is safe.
unsafe fn dtoh_sync(dst: &mut [u8], src: u64) {
    let res = sys::cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut _, src, dst.len());
    assert_eq!(
        res,
        sys::cudaError_enum::CUDA_SUCCESS,
        "cuMemcpyDtoH_v2 returned {:?}",
        res
    );
}

#[test]
fn a2_alloc_write_free_realloc_no_host_sync_between_phases() {
    let Some((device, pool)) = try_setup() else {
        eprintln!("Skipping A2: CUDA runtime unavailable");
        return;
    };
    let resource = AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool));

    // Acquire a real non-default stream. If the pool falls back to
    // DEFAULT (e.g., fork failed on this host) the test cannot prove
    // stream-ordered reuse — bail rather than assert a falsehood.
    let stream_id = match pool.acquire() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Skipping A2: StreamPool::acquire failed: {}", e);
            return;
        }
    };
    assert_ne!(stream_id, StreamId::DEFAULT);
    let stream = pool
        .resolve(stream_id)
        .expect("acquired StreamId must resolve");
    let cu_stream = stream.cu_stream();

    // Phase 1: allocate A on the non-default stream.
    let block_a = resource
        .allocate(BYTES, stream_id, AllocTag("a2-A"))
        .expect("alloc A");
    assert_eq!(block_a.alloc_stream, stream_id);
    assert_eq!(block_a.bytes, BYTES);
    let bytes_after_a = resource.bytes_outstanding();
    assert_eq!(bytes_after_a, BYTES);

    // Phase 2: queue an async write of pattern_a. No host sync.
    let pattern_a = vec![0xAAu8; BYTES];
    unsafe {
        htod_async(cu_stream, block_a.ptr, &pattern_a);
    }

    // Phase 3: deallocate A. Drop runs cuMemFreeAsync on the same
    // stream — ordered after the queued write. The freed bytes
    // remain counted in `bytes_outstanding` until the queued free
    // drains (i.e., until we either sync the stream or call
    // `reap_pending`); confirming this is the trait contract for
    // async backends.
    resource.deallocate(block_a).expect("dealloc A");
    assert_eq!(
        resource.bytes_outstanding(),
        BYTES,
        "queued cuMemFreeAsync must remain counted as pending until reaped"
    );

    // Phase 4: allocate B on the same stream. The driver may return
    // the same byte address as A; if so, reuse must be ordered
    // strictly after the queued free.
    let block_b = resource
        .allocate(BYTES, stream_id, AllocTag("a2-B"))
        .expect("alloc B");
    assert_eq!(block_b.alloc_stream, stream_id);
    assert_eq!(block_b.bytes, BYTES);

    // Phase 5: queue async write of pattern_b. Still no host sync.
    let pattern_b = vec![0xBBu8; BYTES];
    unsafe {
        htod_async(cu_stream, block_b.ptr, &pattern_b);
    }

    // Phase 6: only now synchronize the stream.
    stream.synchronize().expect("stream sync");

    // Phase 7: read back B. If the backend honored stream ordering,
    // pattern_b is the last writer on B's address and the readback
    // matches byte-for-byte. If A's queued write raced reuse, we'd
    // see 0xAA bytes mixed in.
    let mut readback = vec![0u8; BYTES];
    unsafe {
        dtoh_sync(&mut readback, block_b.ptr);
    }
    assert_eq!(
        readback, pattern_b,
        "stream-ordered reuse violated: block B contains stale bytes from A's queued write"
    );

    resource.deallocate(block_b).expect("dealloc B");
    // Final reap drains both A's and B's queued frees (their stream
    // has already been synced once for the readback; reap re-syncs
    // and clears the pending counter).
    resource.reap_pending().expect("reap pending");
    assert_eq!(resource.bytes_outstanding(), 0);
}

#[test]
fn a2_repeated_alloc_free_realloc_on_same_stream_stays_stream_ordered() {
    // Tighter version of the above: alternate alloc/queue-write/free
    // many times on a single stream without intervening host sync.
    // Each iteration's pattern is unique; the final readback after a
    // single sync at the end must match the last-iteration pattern.
    let Some((device, pool)) = try_setup() else {
        return;
    };
    let resource = AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool));

    let stream_id = match pool.acquire() {
        Ok(id) => id,
        Err(e) => {
            eprintln!(
                "Skipping A2 reuse stress: StreamPool::acquire failed: {}",
                e
            );
            return;
        }
    };
    assert_ne!(stream_id, StreamId::DEFAULT);
    let stream = pool
        .resolve(stream_id)
        .expect("acquired StreamId must resolve");
    let cu_stream = stream.cu_stream();

    // Carry the live block across iterations so the read happens
    // after the loop but before the final dealloc — this proves the
    // last write survived all the prior alloc/free churn.
    let mut current = resource
        .allocate(BYTES, stream_id, AllocTag("a2-stress-init"))
        .expect("initial alloc");

    let mut last_pattern = vec![0u8; BYTES];

    for iter in 0..REUSE_ITERATIONS {
        let stamp: u8 = ((iter as u32) & 0xFF) as u8;
        let pattern: Vec<u8> = (0..BYTES)
            .map(|i| stamp.wrapping_add((i & 0xFF) as u8))
            .collect();
        unsafe {
            htod_async(cu_stream, current.ptr, &pattern);
        }

        // On the last iteration, keep `current` alive for readback.
        if iter == REUSE_ITERATIONS - 1 {
            last_pattern = pattern;
            break;
        }

        // Otherwise: free current, alloc next on the same stream.
        // Both operations are stream-ordered on `stream_id`.
        resource.deallocate(current).expect("dealloc mid-loop");
        current = resource
            .allocate(BYTES, stream_id, AllocTag("a2-stress-iter"))
            .expect("alloc mid-loop");
    }

    stream.synchronize().expect("stream sync");

    let mut readback = vec![0u8; BYTES];
    unsafe {
        dtoh_sync(&mut readback, current.ptr);
    }
    assert_eq!(
        readback, last_pattern,
        "stream-ordered reuse violated under repeated alloc/free on the same stream"
    );

    resource.deallocate(current).expect("dealloc final");
    resource.reap_pending().expect("reap pending");
    assert_eq!(resource.bytes_outstanding(), 0);
}
