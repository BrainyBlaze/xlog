// crates/xlog-cuda/tests/test_runtime_cross_stream_use_after_free.rs
//! Failing-by-design reproducer for the v0.6 stream/lifetime-safety
//! bug class identified by A4 (see
//! `docs/plans/2026-04-28-v0.6-stream-lifetime-prerequisite.md`).
//!
//! Bug class
//! ---------
//! `AsyncCudaResource::allocate(bytes, alloc_stream, ...)` returns a
//! block whose `cuMemFreeAsync` will be queued on `alloc_stream`
//! when the block is dropped. The CUDA stream-ordered memory
//! allocator (`cuMemAllocAsync` / `cuMemFreeAsync`) tracks
//! free-then-reuse ordering ONLY against the stream the free is
//! queued on, plus any explicit `cuStreamWaitEvent` dependencies
//! the caller registered. **Work submitted on any other stream
//! that touches this memory is invisible to the pool.**
//!
//! Production code (xlog-cuda's `GpuMemoryManager::with_runtime` +
//! the prototype's raw `Vec<*mut c_void>` launches) violates this
//! contract: it launches kernels on streams other than the slice's
//! `alloc_stream`, then drops the slice, then re-allocates. The
//! mempool can reuse the address while the cross-stream work is
//! still in flight, silently corrupting the new allocation. A4
//! showed this as `mask len > row cap`, arithmetic value-mixing,
//! missing Datalog derivations, and SIGSEGV under shared-runtime
//! parallel use.
//!
//! What this test does
//! -------------------
//!   1. Build a `StreamPool` and acquire two non-default streams,
//!      `s_alloc` and `s_use`.
//!   2. `AsyncCudaResource::allocate(N, s_alloc, ...)` produces
//!      block A. The slice's queued `cuMemFreeAsync` will run on
//!      `s_alloc`.
//!   3. Synchronously copy a pattern `P_a` into block A — proves
//!      the address is real and the resource handed it back
//!      cleanly.
//!   4. Queue an *async* HtoD copy of pattern `P_use` to block A's
//!      bytes ON `s_use`. **No host sync. `s_use` is a different
//!      stream from `s_alloc`.** This represents a kernel
//!      launched on the executor's default stream while the
//!      slice is bound to a different alloc stream.
//!   5. Drop / `deallocate` block A. `cuMemFreeAsync` queues on
//!      `s_alloc`. From the pool's perspective the memory is now
//!      reusable on `s_alloc` (and on streams that wait for
//!      `s_alloc`'s free event). `s_use`'s queued copy is
//!      invisible to the pool — the pool does not know that work
//!      is still pending against this address.
//!   6. Immediately allocate block B of the same size on
//!      `s_alloc`. The pool is likely to return the same byte
//!      address as A.
//!   7. Synchronously copy a different pattern `P_b` into B on
//!      `s_alloc`. (Sync HtoD goes through legacy default stream
//!      with explicit sync — its semantics here are
//!      "cuMemcpyHtoD then return"; on the host side that
//!      happens-before any subsequent host code.)
//!   8. **Synchronize `s_use`.** This is the point where the
//!      pending cross-stream copy from step 4 finally lands. If
//!      B reused A's address, the copy lands *into B*.
//!   9. Read B back and compare against `P_b`. If `s_use`'s late
//!      copy clobbered B, the readback contains `P_use` (or a
//!      mix). That's the bug.
//!
//! On a sound stream/lifetime-safety layer, the runtime would
//! have recorded an event for `s_use`'s pending work and queued
//! `cuStreamWaitEvent(s_alloc, last_use_event)` before
//! `cuMemFreeAsync`, so the pool would not have returned the
//! same address to step 6 until `s_use`'s work completed.
//! Currently no such event exists.
//!
//! This test must FAIL on `main` HEAD (commit 2fde633f) to lock
//! the bug class. After the safety layer lands, it must PASS.
//! Skipped cleanly if no CUDA device is available.

use std::sync::Arc;

use cudarc::driver::sys;
use xlog_cuda::device_runtime::{
    AllocTag, AsyncCudaResource, DeviceMemoryResource, StreamId, StreamPool,
};
use xlog_cuda::CudaDevice;

const BYTES: usize = 4096;

/// Async HtoD on a specific raw stream handle.
#[allow(dead_code)]
unsafe fn htod_async(stream: sys::CUstream, dst: u64, src: &[u8]) {
    let res = sys::cuMemcpyHtoDAsync_v2(dst, src.as_ptr() as *const _, src.len(), stream);
    assert_eq!(
        res,
        sys::cudaError_enum::CUDA_SUCCESS,
        "cuMemcpyHtoDAsync_v2: {:?}",
        res
    );
}

/// Async memset on a specific raw stream handle. Unlike async
/// memcpy from a non-pinned host buffer (which the driver may
/// stage synchronously), `cuMemsetD8Async` is genuinely
/// stream-asynchronous — it queues on the stream and returns
/// immediately. Used here to ensure the cross-stream "use" of
/// the allocation is actually pending when we drop.
unsafe fn memset_async(stream: sys::CUstream, dst: u64, value: u8, len: usize) {
    let res = sys::cuMemsetD8Async(dst, value, len, stream);
    assert_eq!(
        res,
        sys::cudaError_enum::CUDA_SUCCESS,
        "cuMemsetD8Async: {:?}",
        res
    );
}

/// Synchronous HtoD (uses cudarc default-stream semantics).
unsafe fn htod_sync(dst: u64, src: &[u8]) {
    let res = sys::cuMemcpyHtoD_v2(dst, src.as_ptr() as *const _, src.len());
    assert_eq!(
        res,
        sys::cudaError_enum::CUDA_SUCCESS,
        "cuMemcpyHtoD_v2: {:?}",
        res
    );
}

/// Synchronous DtoH (called only after both involved streams have
/// been synchronized).
unsafe fn dtoh_sync(dst: &mut [u8], src: u64) {
    let res = sys::cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut _, src, dst.len());
    assert_eq!(
        res,
        sys::cudaError_enum::CUDA_SUCCESS,
        "cuMemcpyDtoH_v2: {:?}",
        res
    );
}

/// Marked `#[ignore]` because this test is RED on the current
/// branch and on `main` HEAD by design — it pins the bug class
/// the v0.6 stream/lifetime-safety PR is built to fix. Opt in via
/// `cargo test -- --ignored runtime_backed_drop_after_cross_stream_use`.
/// The `#[ignore]` attribute MUST be removed in the same commit
/// that lands the fix and turns this test green.
#[test]
#[ignore = "RED reproducer for v0.6 stream/lifetime-safety; remove ignore when the safety layer lands"]
fn runtime_backed_drop_after_cross_stream_use_corrupts_reuse() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let resource = AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool));

    // Two distinct non-default streams. If the pool can't fork at
    // least two streams, skip — the bug class needs the
    // alloc-stream / use-stream split.
    let s_alloc_id = match pool.acquire() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Skipping: StreamPool::acquire failed: {}", e);
            return;
        }
    };
    let s_use_id = match pool.acquire() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Skipping: StreamPool::acquire (second) failed: {}", e);
            return;
        }
    };
    assert_ne!(s_alloc_id, s_use_id);
    assert_ne!(s_alloc_id, StreamId::DEFAULT);
    assert_ne!(s_use_id, StreamId::DEFAULT);

    let s_alloc = pool.resolve(s_alloc_id).expect("s_alloc resolves");
    let s_use = pool.resolve(s_use_id).expect("s_use resolves");
    let s_use_handle = s_use.cu_stream();

    // Repeat the alloc/cross-stream-use/drop/realloc cycle many
    // times. The bug is racy under a single trial; under a tight
    // loop the cuMemAllocAsync pool returns the same address with
    // high probability.
    const ITERATIONS: usize = 64;
    const PATTERN_USE: u8 = 0xCD;

    let mut corrupt_count = 0usize;
    let mut reuse_observed = 0usize;
    let _ = htod_sync; // legacy default-stream sync drains all streams; intentionally unused below

    for _ in 0..ITERATIONS {
        // Step 2: allocate block A on s_alloc.
        let block_a = resource
            .allocate(BYTES, s_alloc_id, AllocTag("rep-A"))
            .expect("alloc A on s_alloc");
        let ptr_a = block_a.ptr;

        // Step 3: initialize A's contents to zeros via an
        // s_alloc-bound memset. This stays on s_alloc — does NOT
        // synchronize across streams — so we don't accidentally
        // drain s_use's later queued work.
        unsafe { memset_async(s_alloc.cu_stream(), ptr_a, 0x00, BYTES) };

        // Step 4: cross-stream queued memset of `PATTERN_USE` on
        // `s_use`. cuMemsetD8Async is genuinely stream-async (no
        // host-buffer staging), so the write is queued, not
        // immediate. `s_use` is NOT `s_alloc`.
        unsafe { memset_async(s_use_handle, ptr_a, PATTERN_USE, BYTES) };

        // Step 5: drop block A — cuMemFreeAsync queues on s_alloc.
        // s_use's pending memset to ptr_a is invisible to the pool.
        resource.deallocate(block_a).expect("dealloc A");

        // Step 6: allocate block B on s_alloc; very likely reuses
        // the address. Per CUDA mempool semantics this is
        // "stream-ordered safe" only with respect to s_alloc's
        // queued free — NOT with respect to s_use's queued
        // memset, which the pool was never told about.
        let block_b = resource
            .allocate(BYTES, s_alloc_id, AllocTag("rep-B"))
            .expect("alloc B on s_alloc");
        let ptr_b = block_b.ptr;
        let same_address = ptr_b == ptr_a;
        if same_address {
            reuse_observed += 1;
        }

        // Step 7: drain s_use FIRST. Its queued memset of
        // PATTERN_USE to ptr_a finally lands. If ptr_b == ptr_a,
        // it lands into B's bytes.
        s_use.synchronize().expect("sync s_use");

        // Step 8: drain s_alloc (the cuMemFreeAsync of A and any
        // deferred bookkeeping). After this, B is "ready for use"
        // by xlog's contract — alloc on s_alloc, all pending work
        // on s_alloc drained.
        s_alloc.synchronize().expect("sync s_alloc");

        // Step 9: read B and check whether it contains the
        // PATTERN_USE leaked from s_use's late write. If so,
        // that's the cross-stream lifetime bug.
        let mut readback = vec![0u8; BYTES];
        unsafe { dtoh_sync(&mut readback, ptr_b) };

        // Count an iteration as corrupt if B's first byte is
        // PATTERN_USE — extremely unlikely by chance from a fresh
        // mempool allocation, deterministic if the bug reuse path
        // fired.
        if same_address && readback[0] == PATTERN_USE && readback[BYTES - 1] == PATTERN_USE {
            corrupt_count += 1;
        }

        resource.deallocate(block_b).expect("dealloc B");
        resource.reap_pending().expect("reap");
    }

    eprintln!(
        "iterations={} reuse_observed={} corrupt={}",
        ITERATIONS, reuse_observed, corrupt_count
    );

    // The test demands BOTH conditions: the pool must reuse
    // addresses (otherwise the bug surface isn't exercised) AND
    // corruption must be observable. If reuse never occurs the
    // host's mempool is unusually conservative and we cannot
    // certify the safety layer with this reproducer — surface
    // that as a panic so a follow-up reproducer can sharpen the
    // test.
    assert!(
        reuse_observed > 0,
        "address reuse never observed across {} iterations; \
         the test cannot exercise the cross-stream lifetime gap. \
         Adjust BYTES/ITERATIONS or stream selection.",
        ITERATIONS
    );
    assert_eq!(
        corrupt_count, 0,
        "cross-stream lifetime bug observed: {}/{} iterations had \
         block B's contents clobbered by s_use's pending write to \
         block A's address (reuse_observed={}). Stream-ordered \
         allocator did not protect against use-after-free queued \
         on a different stream than alloc_stream. See \
         docs/plans/2026-04-28-v0.6-stream-lifetime-prerequisite.md.",
        corrupt_count, ITERATIONS, reuse_observed,
    );
}
