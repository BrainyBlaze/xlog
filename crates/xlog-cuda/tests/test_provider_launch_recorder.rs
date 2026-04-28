// crates/xlog-cuda/tests/test_provider_launch_recorder.rs
//! Provider-level test for the first-slice migrated launch path
//! (`CudaKernelProvider::memset_recorded`) — proves a real xlog
//! launch records the buffer use automatically through the
//! runtime and survives drop+reuse without cross-stream
//! corruption.
//!
//! End-to-end stack exercised:
//!   GpuMemoryManager::with_runtime
//!     → CudaKernelProvider::with_runtime
//!       → XlogDeviceRuntime::with_resource(GlobalDeviceBudget(
//!           LoggingResource(AsyncCudaResource)))
//!
//! Test shape (mirrors the bug class from
//! `test_runtime_cross_stream_use_after_free.rs` but at the
//! provider level):
//!
//!   1. Provider built with a runtime-backed manager. Acquire a
//!      non-default `launch_stream` from the runtime's pool.
//!   2. Allocate `input` via the manager. Its `alloc_stream` is
//!      `StreamId::DEFAULT` (the manager's current routing).
//!   3. Call `provider.memset_recorded(&mut input, 0xCD, launch_stream)`.
//!      Internally this queues `cuMemsetD8Async(input, 0xCD)` on
//!      `launch_stream` and records the use against the runtime.
//!   4. Drop `input` immediately (no host sync). The runtime's
//!      `deallocate` waits for the recorded event before
//!      queueing `cuMemFreeAsync` on alloc_stream.
//!   5. Allocate `next` of the same size — the pool may reuse
//!      `input`'s address. The new allocate is ordered after the
//!      free (which is ordered after the recorded launch).
//!   6. Synchronously memset `next` to PATTERN_NEW on the
//!      default stream.
//!   7. Drain `launch_stream` (no-op semantically; the wait
//!      already chained it) and the default stream.
//!   8. Read `next`. Bug class would have left 0xCD; safety
//!      layer must have preserved PATTERN_NEW.
//!
//! Without the recorder's call to `runtime.record_block_use`,
//! step 4's deallocate would queue `cuMemFreeAsync` without a
//! cross-stream wait, the pool could return the same address to
//! step 5 immediately, and step 7's `launch_stream` drain would
//! land 0xCD on top of `next` (bug). With the recorder, the
//! free is correctly ordered and PATTERN_NEW survives.

use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, XlogError};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

const BYTES: usize = 4096;
const PATTERN_LAUNCH: u8 = 0xCD;
const PATTERN_NEW: u8 = 0xBB;

/// Local discard sink so the runtime stack matches the
/// production composition without retaining records.
struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

unsafe fn dtoh_sync(dst: &mut [u8], src: u64) {
    let res = sys::cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut _, src, dst.len());
    assert_eq!(
        res,
        sys::cudaError_enum::CUDA_SUCCESS,
        "cuMemcpyDtoH_v2: {:?}",
        res
    );
}
unsafe fn memset_sync_default(dst: u64, value: u8, len: usize) {
    // cuMemsetD8 (no Async) is synchronous w.r.t. host AND
    // synchronizes against the legacy default stream.
    let res = sys::cuMemsetD8_v2(dst, value, len);
    assert_eq!(
        res,
        sys::cudaError_enum::CUDA_SUCCESS,
        "cuMemsetD8: {:?}",
        res
    );
}

#[test]
fn provider_memset_recorded_survives_drop_and_reuse() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));

    // Production stack: AsyncCudaResource (cross-stream
    // tracking) → LoggingResource → GlobalDeviceBudget.
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
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");

    let launch_stream = pool.acquire().expect("acquire non-default launch stream");
    assert_ne!(launch_stream, StreamId::DEFAULT);
    let launch_stream_handle = pool.resolve(launch_stream).expect("launch_stream resolves");
    let default_stream = device.inner().stream();

    const ITERATIONS: usize = 64;
    let mut last_writer_was_new = 0usize;
    let mut last_writer_was_launch = 0usize;
    let mut reuse_observed = 0usize;

    for _ in 0..ITERATIONS {
        // Step 2: allocate input (alloc_stream = DEFAULT).
        let mut input = memory.alloc::<u8>(BYTES).expect("alloc input");
        let in_ptr = input.device_ptr_value();
        assert!(input.runtime_block().is_some());

        // Step 3: real provider launch. memset_recorded queues
        // cuMemsetD8Async(input, 0xCD) on launch_stream AND
        // records the use against the runtime. This is the
        // path being certified by this test.
        provider
            .memset_recorded(&mut input, PATTERN_LAUNCH, launch_stream)
            .expect("memset_recorded must succeed against runtime-backed provider");

        // Step 4: drop input (no host sync). runtime.deallocate
        // queues cuStreamWaitEvent(alloc_stream, recorded_event)
        // BEFORE cuMemFreeAsync.
        drop(input);

        // Step 5: allocate next on default — same alloc_stream
        // as input. Pool likely reuses in_ptr.
        let next = memory.alloc::<u8>(BYTES).expect("alloc next");
        let next_ptr = next.device_ptr_value();
        if next_ptr == in_ptr {
            reuse_observed += 1;
        }

        // Step 6: synchronously memset `next` to PATTERN_NEW
        // via cuMemsetD8 on the legacy default stream. With
        // legacy-default-stream semantics this synchronizes
        // against ALL prior queued work on every stream — but
        // critically, the runtime's wait-event in deallocate
        // already ordered the launch_stream memset before the
        // free before this alloc. So when this synchronous
        // memset runs, PATTERN_LAUNCH has long since been
        // overwritten by the free's own bookkeeping (or never
        // landed on next_ptr if the address was different).
        unsafe { memset_sync_default(next_ptr, PATTERN_NEW, BYTES) };

        // Step 7: explicit synchronizes for clarity.
        launch_stream_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Step 8: read next.
        let mut readback = vec![0u8; BYTES];
        unsafe { dtoh_sync(&mut readback, next_ptr) };

        if next_ptr == in_ptr {
            if readback[0] == PATTERN_NEW && readback[BYTES - 1] == PATTERN_NEW {
                last_writer_was_new += 1;
            } else if readback[0] == PATTERN_LAUNCH && readback[BYTES - 1] == PATTERN_LAUNCH {
                last_writer_was_launch += 1;
            }
        }

        drop(next);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[recorder] iterations={} reuse_observed={} \
         last_writer_was_new={} last_writer_was_launch={}",
        ITERATIONS, reuse_observed, last_writer_was_new, last_writer_was_launch
    );

    assert!(
        reuse_observed > 0,
        "address reuse never observed across {} iterations; the test \
         cannot exercise the cross-stream lifetime safety path",
        ITERATIONS
    );
    assert_eq!(
        last_writer_was_launch, 0,
        "launch recorder failed: {}/{} iterations had `next` clobbered \
         by the launch_stream memset of PATTERN_LAUNCH (reuse_observed={}, \
         last_writer_was_new={}). The recorder's record_block_use call \
         must have either not run or not propagated the wait through \
         the deallocate path.",
        last_writer_was_launch, ITERATIONS, reuse_observed, last_writer_was_new,
    );
}

/// Negative test: the same launch path against a manager built
/// via legacy `GpuMemoryManager::new` (no runtime attached).
/// `memset_recorded` must fail loudly with `XlogError::Kernel`,
/// NOT silently fall back. Locks the contract that this
/// migrated path requires runtime-backed allocation.
#[test]
fn provider_memset_recorded_rejects_legacy_manager() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let memory = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(1024 * 1024),
    ));
    let provider =
        CudaKernelProvider::new(Arc::clone(&device), Arc::clone(&memory)).expect("legacy provider");

    let mut buf = memory.alloc::<u8>(64).expect("alloc legacy");
    assert!(buf.runtime_block().is_none());

    let err = provider.memset_recorded(&mut buf, 0xAA, StreamId::DEFAULT);
    match err {
        Err(XlogError::Kernel(msg)) => {
            assert!(
                msg.contains("requires") || msg.contains("with_runtime"),
                "expected helpful Kernel error message, got {:?}",
                msg
            );
        }
        other => panic!(
            "memset_recorded must reject legacy manager with XlogError::Kernel, got {:?}",
            other
        ),
    }
}

/// Negative test: provider built around `DirectCudaResource`
/// (the trait default that intentionally rejects
/// `record_block_use`). The launch's memset queues fine, but
/// the recorder's commit must surface `StreamMisuse` wrapped in
/// `XlogError::Kernel` rather than masking the fault.
#[test]
fn provider_memset_recorded_surfaces_stream_misuse_from_direct_resource() {
    use xlog_cuda::device_runtime::DirectCudaResource;

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let direct: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        direct,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider =
        CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).expect("p");

    let mut buf = memory.alloc::<u8>(64).expect("alloc");
    let err = provider.memset_recorded(&mut buf, 0xAA, StreamId::DEFAULT);
    match err {
        Err(XlogError::Kernel(msg)) => {
            assert!(
                msg.contains("commit failed") || msg.contains("unsupported"),
                "expected commit-failed StreamMisuse-derived error, got {:?}",
                msg
            );
        }
        other => panic!(
            "memset_recorded must surface StreamMisuse from DirectCudaResource as \
             XlogError::Kernel, got {:?}",
            other
        ),
    }
}
