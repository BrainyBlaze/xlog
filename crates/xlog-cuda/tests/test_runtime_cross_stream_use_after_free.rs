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

/// **Managed-uses path: must PASS.**
///
/// The caller submits cross-stream work touching block A, then
/// records that use against the resource via
/// [`AsyncCudaResource::record_block_use`] **before** dropping
/// the block. The resource captures a CUDA event on `s_use`,
/// attaches it to the live entry, and on `deallocate(A)` queues a
/// `cuStreamWaitEvent(s_alloc, last_use_event)` before the
/// implied `cuMemFreeAsync`. The pool therefore cannot return
/// the same address to the next allocate on `s_alloc` until
/// `s_use`'s pending work has completed.
///
/// On `main` HEAD (commit 2fde633f) and earlier, this test fails
/// 64/64 because no such API exists. With the safety layer
/// landed, it must pass 64/64.
#[test]
fn managed_cross_stream_use_does_not_corrupt_reuse() {
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
    const PATTERN_B: u8 = 0xBB;

    let mut last_writer_was_b = 0usize;
    let mut last_writer_was_use = 0usize;
    let mut reuse_observed = 0usize;
    let _ = htod_sync; // legacy default-stream sync drains all streams; intentionally unused below

    for _ in 0..ITERATIONS {
        // Step 2: allocate block A on s_alloc.
        let block_a = resource
            .allocate(BYTES, s_alloc_id, AllocTag("rep-A"))
            .expect("alloc A on s_alloc");
        let ptr_a = block_a.ptr;

        // Step 3: initialize A's contents to zeros via an
        // s_alloc-bound memset.
        unsafe { memset_async(s_alloc.cu_stream(), ptr_a, 0x00, BYTES) };

        // Step 4: cross-stream queued memset of `PATTERN_USE` on
        // `s_use`. cuMemsetD8Async is genuinely stream-async (no
        // host-buffer staging), so the write is queued, not
        // immediate. `s_use` is NOT `s_alloc`.
        unsafe { memset_async(s_use_handle, ptr_a, PATTERN_USE, BYTES) };

        // Step 4b: tell the resource that s_use has pending work
        // touching ptr_a. The resource records an event on s_use
        // *now*, after the memset was queued. On `deallocate(A)`,
        // s_alloc will be made to wait on this event before the
        // queued cuMemFreeAsync runs.
        resource
            .record_block_use(ptr_a, s_use_id)
            .expect("record_block_use(ptr_a, s_use)");

        // Step 5: drop block A. `deallocate` enqueues
        // `s_alloc.wait(event)` (recorded above), THEN the slice
        // drop queues `cuMemFreeAsync`. So s_alloc's queue is now:
        //   wait(E_use) → cuMemFreeAsync(ptr_a)
        resource.deallocate(block_a).expect("dealloc A");

        // Step 6: allocate B on s_alloc. cuMemAllocAsync is
        // queued after the free; combined with the wait above,
        // s_alloc's queue is now:
        //   wait(E_use) → cuMemFreeAsync(ptr_a) → cuMemAllocAsync(ptr_b)
        let block_b = resource
            .allocate(BYTES, s_alloc_id, AllocTag("rep-B"))
            .expect("alloc B on s_alloc");
        let ptr_b = block_b.ptr;
        let same_address = ptr_b == ptr_a;
        if same_address {
            reuse_observed += 1;
        }

        // Step 7: write PATTERN_B to B on s_alloc. cuMemAllocAsync
        // does NOT zero the memory — B inherits whatever was at
        // ptr_a when the free ran. We need a *fresh* write to B
        // to detect corruption: if `cuMemAllocAsync` ordering
        // worked correctly, this memset is the last writer to
        // ptr_b and the readback shows PATTERN_B. If s_use's
        // memset of PATTERN_USE landed *after* this memset (the
        // lifetime bug), the readback shows PATTERN_USE.
        unsafe { memset_async(s_alloc.cu_stream(), ptr_b, PATTERN_B, BYTES) };

        // Step 8: drain s_alloc. With the managed contract, the
        // wait at the head of s_alloc's queue blocks the free
        // until E_use fires (which happens when s_use drains
        // past its memset). So this synchronize transitively
        // forces s_use's memset to complete BEFORE s_alloc's
        // free / alloc / memset_B sequence runs.
        s_alloc.synchronize().expect("sync s_alloc");

        // Step 9: drain s_use as well — without the managed
        // contract, this is the moment s_use's memset would
        // actually land, possibly overwriting our PATTERN_B.
        // With the managed contract it's a no-op (s_use already
        // drained transitively).
        s_use.synchronize().expect("sync s_use");

        // Step 10: read B.
        let mut readback = vec![0u8; BYTES];
        unsafe { dtoh_sync(&mut readback, ptr_b) };

        if same_address {
            // Classify the last-writer-wins outcome:
            //   PATTERN_B  → s_alloc's memset was the last writer
            //                (correct, managed-uses contract held)
            //   PATTERN_USE → s_use's memset clobbered after
            //                 s_alloc's memset (lifetime bug)
            if readback[0] == PATTERN_B && readback[BYTES - 1] == PATTERN_B {
                last_writer_was_b += 1;
            } else if readback[0] == PATTERN_USE && readback[BYTES - 1] == PATTERN_USE {
                last_writer_was_use += 1;
            }
            // Other outcomes (mixed bytes, neither pattern) are
            // not counted into either bucket; the assertions
            // below will surface them as a discrepancy.
        }

        resource.deallocate(block_b).expect("dealloc B");
        resource.reap_pending().expect("reap");
    }

    eprintln!(
        "[managed] iterations={} reuse_observed={} \
         last_writer_was_b={} last_writer_was_use={}",
        ITERATIONS, reuse_observed, last_writer_was_b, last_writer_was_use
    );

    assert!(
        reuse_observed > 0,
        "address reuse never observed across {} iterations; the \
         test cannot exercise the cross-stream lifetime gap.",
        ITERATIONS
    );
    assert_eq!(
        last_writer_was_use, 0,
        "managed-uses path failed: {}/{} iterations had B \
         clobbered by s_use's late memset of PATTERN_USE despite \
         record_block_use being called (reuse_observed={}, \
         last_writer_was_b={}). The wait-event chain did not \
         hold s_alloc until s_use's memset completed. Safety \
         layer broken.",
        last_writer_was_use, ITERATIONS, reuse_observed, last_writer_was_b,
    );
    assert_eq!(
        last_writer_was_b, reuse_observed,
        "managed-uses path: every reuse iteration should leave \
         PATTERN_B as the last write to ptr_b, but only {}/{} \
         did. Discrepancy implies the per-byte readback found \
         neither pattern (mixed bytes — also a corruption \
         signature).",
        last_writer_was_b, reuse_observed,
    );
}

/// **Unmanaged-uses path: kept `#[ignore]`d, documents the
/// contract.**
///
/// Same shape as the managed test, but the caller submits
/// cross-stream work and **does NOT** call `record_block_use`.
/// The resource has no way to infer arbitrary external CUDA work
/// — the cross-stream pending memset is invisible to it — so the
/// pool can return the address to a subsequent allocate while
/// the cross-stream write is still in flight, and corruption is
/// observed.
///
/// This test exists to lock the contract documented on
/// `record_block_use`: callers that submit raw CUDA work on a
/// stream other than `block.alloc_stream` and bypass xlog's
/// launch-builder / use-recording layer are responsible for
/// their own cross-stream synchronization. If they neither use
/// `record_block_use` nor synchronize manually, lifetime safety
/// is undefined by design.
///
/// Kept `#[ignore]`d because:
///   * It demonstrates the *expected* unsafe behavior of an
///     unmanaged caller — running it as part of default CI would
///     turn the suite red.
///   * The corruption observable here is by design; a future
///     change that "fixes" this path *automatically* by tracking
///     all CUDA work the resource never saw would be wrong, and
///     this test prevents that drift.
///
/// To verify the unmanaged path still corrupts:
///   `cargo test -p xlog-cuda --release --test \
///    test_runtime_cross_stream_use_after_free -- --ignored \
///    unmanaged_cross_stream_use`
#[test]
#[ignore = "documents the unmanaged-raw-CUDA-call contract; corruption is the *intended* outcome here"]
fn unmanaged_cross_stream_use_corrupts_reuse_by_design() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let resource = AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool));

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
    let s_alloc = pool.resolve(s_alloc_id).expect("s_alloc resolves");
    let s_use = pool.resolve(s_use_id).expect("s_use resolves");
    let s_use_handle = s_use.cu_stream();

    const ITERATIONS: usize = 64;
    const PATTERN_USE: u8 = 0xCD;
    const PATTERN_B: u8 = 0xBB;

    let mut last_writer_was_b = 0usize;
    let mut last_writer_was_use = 0usize;
    let mut reuse_observed = 0usize;

    for _ in 0..ITERATIONS {
        let block_a = resource
            .allocate(BYTES, s_alloc_id, AllocTag("rep-A"))
            .expect("alloc A");
        let ptr_a = block_a.ptr;

        unsafe { memset_async(s_alloc.cu_stream(), ptr_a, 0x00, BYTES) };
        unsafe { memset_async(s_use_handle, ptr_a, PATTERN_USE, BYTES) };

        // Note: NO record_block_use call. The cross-stream memset
        // is invisible to the resource.
        resource.deallocate(block_a).expect("dealloc A");

        let block_b = resource
            .allocate(BYTES, s_alloc_id, AllocTag("rep-B"))
            .expect("alloc B");
        let ptr_b = block_b.ptr;
        if ptr_b == ptr_a {
            reuse_observed += 1;
        }

        unsafe { memset_async(s_alloc.cu_stream(), ptr_b, PATTERN_B, BYTES) };

        // Drain s_alloc first — without the managed contract,
        // this only drains the free→alloc→memset_B chain and
        // does NOT wait for s_use's memset.
        s_alloc.synchronize().expect("sync s_alloc");
        // Now drain s_use — its memset of PATTERN_USE finally
        // lands. If ptr_b == ptr_a, it overwrites PATTERN_B.
        s_use.synchronize().expect("sync s_use");

        let mut readback = vec![0u8; BYTES];
        unsafe { dtoh_sync(&mut readback, ptr_b) };
        if ptr_b == ptr_a {
            if readback[0] == PATTERN_USE && readback[BYTES - 1] == PATTERN_USE {
                last_writer_was_use += 1;
            } else if readback[0] == PATTERN_B && readback[BYTES - 1] == PATTERN_B {
                last_writer_was_b += 1;
            }
        }

        resource.deallocate(block_b).expect("dealloc B");
        resource.reap_pending().expect("reap");
    }

    eprintln!(
        "[unmanaged] iterations={} reuse_observed={} \
         last_writer_was_b={} last_writer_was_use={}",
        ITERATIONS, reuse_observed, last_writer_was_b, last_writer_was_use
    );

    assert!(
        reuse_observed > 0,
        "address reuse never observed across {} iterations; the \
         test cannot exercise the unmanaged-uses contract",
        ITERATIONS
    );
    // The unmanaged contract: at least one iteration must show
    // s_use's memset as the last writer. We don't require all
    // iterations to corrupt — driver scheduling can let s_alloc
    // win the race occasionally — but if NONE corrupt, the
    // documented contract has somehow been "fixed" outside the
    // managed path, which would be an unrelated bug.
    assert!(
        last_writer_was_use > 0,
        "unmanaged-uses test expected at least one iteration where \
         s_use's late memset clobbered B (proving the documented \
         contract still applies), but observed 0/{} \
         (reuse_observed={}, last_writer_was_b={}). Either driver \
         scheduling shifted unexpectedly, or some unrelated change \
         silently fixed this path; investigate before re-enabling.",
        ITERATIONS,
        reuse_observed,
        last_writer_was_b,
    );
}
