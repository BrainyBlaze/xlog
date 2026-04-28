// crates/xlog-cuda/tests/test_runtime_a2_via_runtime.rs
//! A2 variant exercised through the [`XlogDeviceRuntime`] facade
//! rather than against an `AsyncCudaResource` directly.
//!
//! The original `test_runtime_a2_stream_lifetime.rs` constructs an
//! `AsyncCudaResource` and calls its trait methods directly. That
//! proves the resource itself honors the stream-ordered contract,
//! but it bypasses the runtime — so it would not catch a regression
//! introduced in `XlogDeviceRuntime::allocate`/`deallocate`/
//! `reap_pending` that mishandled stream-ordering across the
//! mutex-protected `Box<dyn DeviceMemoryResource>` boundary (e.g.,
//! holding the resource lock too long, dropping it across a sync,
//! or losing the pending-bytes invariant).
//!
//! This test composes a runtime via
//! [`XlogDeviceRuntime::with_resource`] using `AsyncCudaResource`
//! as the active backend, then re-runs the A2 contract through the
//! runtime's public API. It is **not** the singleton — that path
//! still uses the cudarc default (non-pooled) backend.
//!
//! Same shape as A2:
//!
//!   1. `runtime.allocate` block A on a non-default stream.
//!   2. Async HtoD pattern_a (no host sync).
//!   3. `runtime.deallocate(A)` — queues cuMemFreeAsync. Bytes
//!      remain counted in `runtime.bytes_outstanding()` until
//!      `reap_pending`.
//!   4. `runtime.allocate` block B on the same stream — may reuse
//!      A's address.
//!   5. Async HtoD pattern_b.
//!   6. `cuStreamSynchronize`.
//!   7. Read back B; must equal pattern_b.
//!   8. `runtime.deallocate(B)`, `runtime.reap_pending()`.
//!   9. `runtime.bytes_outstanding() == 0`.
//!
//! Skips when CUDA is unavailable or the pool can't fork a
//! non-default stream.

use std::sync::Arc;

use cudarc::driver::sys;
use xlog_cuda::device_runtime::{
    AllocTag, AsyncCudaResource, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::CudaDevice;

const BYTES: usize = 4096;

unsafe fn htod_async(stream: sys::CUstream, dst: u64, src: &[u8]) {
    let res = sys::cuMemcpyHtoDAsync_v2(dst, src.as_ptr() as *const _, src.len(), stream);
    assert_eq!(
        res,
        sys::cudaError_enum::CUDA_SUCCESS,
        "cuMemcpyHtoDAsync_v2 returned {:?}",
        res
    );
}

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
fn a2_stream_ordered_alloc_free_reuse_through_runtime_facade() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping A2-via-runtime: CUDA runtime unavailable");
        return;
    };
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let resource = Box::new(AsyncCudaResource::new(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
    ));
    let runtime =
        XlogDeviceRuntime::with_resource(Arc::clone(&device), 0, Arc::clone(&pool), resource);

    let stream_id = match pool.acquire() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Skipping A2-via-runtime: StreamPool::acquire failed: {}", e);
            return;
        }
    };
    assert_ne!(stream_id, StreamId::DEFAULT);
    let stream = pool
        .resolve(stream_id)
        .expect("acquired StreamId must resolve");
    let cu_stream = stream.cu_stream();

    // Phase 1: allocate A through the runtime.
    let block_a = runtime
        .allocate(BYTES, stream_id, AllocTag("a2-runtime-A"))
        .expect("runtime.allocate A");
    assert_eq!(block_a.alloc_stream, stream_id);
    assert_eq!(block_a.bytes, BYTES);
    assert_eq!(runtime.bytes_outstanding(), BYTES);

    // Phase 2: queue async write of pattern_a. No host sync.
    let pattern_a = vec![0xCDu8; BYTES];
    unsafe {
        htod_async(cu_stream, block_a.ptr, &pattern_a);
    }

    // Phase 3: deallocate A through the runtime. The queued
    // cuMemFreeAsync must remain counted as pending until reap.
    runtime.deallocate(block_a).expect("runtime.deallocate A");
    assert_eq!(
        runtime.bytes_outstanding(),
        BYTES,
        "runtime.bytes_outstanding must report pending free bytes after async deallocate"
    );

    // Phase 4: allocate B on the same stream through the runtime.
    let block_b = runtime
        .allocate(BYTES, stream_id, AllocTag("a2-runtime-B"))
        .expect("runtime.allocate B");
    assert_eq!(block_b.alloc_stream, stream_id);

    // Phase 5: queue async write of pattern_b.
    let pattern_b = vec![0xEFu8; BYTES];
    unsafe {
        htod_async(cu_stream, block_b.ptr, &pattern_b);
    }

    // Phase 6: synchronize the stream once.
    stream.synchronize().expect("stream sync");

    // Phase 7: read back B. If the runtime facade or the underlying
    // resource broke stream ordering, the readback would mix
    // pattern_a bytes into B.
    let mut readback = vec![0u8; BYTES];
    unsafe {
        dtoh_sync(&mut readback, block_b.ptr);
    }
    assert_eq!(
        readback, pattern_b,
        "stream-ordered reuse violated through runtime facade: \
         B contains stale bytes from A's queued write"
    );

    // Phase 8: deallocate B + final reap.
    runtime.deallocate(block_b).expect("runtime.deallocate B");
    runtime.reap_pending().expect("runtime.reap_pending");

    // Phase 9: counters return to zero.
    assert_eq!(runtime.bytes_outstanding(), 0);
}
