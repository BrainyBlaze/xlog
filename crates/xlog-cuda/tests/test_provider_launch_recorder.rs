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
    // cuMemsetD8 (no Async) is synchronous from the host's
    // perspective: it returns only after the operation
    // completes. It runs on the legacy default stream.
    //
    // Important: legacy-default-stream implicit synchronization
    // does NOT extend to non-blocking streams. Per NVIDIA's
    // "Default Stream" docs, only blocking streams synchronize
    // implicitly with the legacy default; streams created via
    // `cudaStreamNonBlocking` (which is what
    // `StreamPool::fork` produces) are explicitly excluded.
    // The runtime's cross-stream wait-event chain set up at
    // `record_block_use` -> `deallocate` is what actually
    // orders the launch_stream memset before any subsequent
    // alloc on alloc_stream — the legacy-default sync below
    // does not by itself protect against the bug class.
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
        // via cuMemsetD8 on the legacy default stream. The
        // legacy default does NOT implicitly synchronize with
        // the non-blocking streams produced by
        // `StreamPool::fork`; per NVIDIA's default-stream docs
        // only blocking streams synchronize implicitly. The
        // safety here comes from the runtime's wait-event
        // chain: at `drop(input)`, the runtime queued
        // `cuStreamWaitEvent(alloc_stream, recorded_event)`
        // BEFORE `cuMemFreeAsync`, so the alloc-stream's free
        // could not run until the launch_stream memset
        // completed. The subsequent allocate on alloc_stream
        // is therefore correctly ordered after the launch's
        // memset. Without that wait-event, this synchronous
        // default-stream memset would race the still-queued
        // launch_stream memset and PATTERN_LAUNCH could clobber
        // PATTERN_NEW under reuse.
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
/// `record_block_use`). With the strict-mode + preflight
/// pattern the launch's memset is **never enqueued** —
/// preflight surfaces `StreamMisuse` BEFORE the CUDA call. The
/// error message identifies preflight rather than commit.
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
                msg.contains("preflight failed") && msg.contains("does not support cross-stream"),
                "expected preflight-failed StreamMisuse-derived error, got {:?}",
                msg
            );
        }
        other => panic!(
            "memset_recorded must surface StreamMisuse from DirectCudaResource as \
             XlogError::Kernel at preflight, got {:?}",
            other
        ),
    }
}

/// Column-level variant: prove `provider.memset_column_recorded`
/// records use through the `CudaColumn::Owned` runtime block
/// automatically. Same drop+reuse safety check as the slice
/// version.
#[test]
fn provider_memset_column_recorded_survives_drop_and_reuse() {
    use xlog_cuda::CudaColumn;

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
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
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire");
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const ITERATIONS: usize = 32;
    let mut last_writer_was_new = 0usize;
    let mut last_writer_was_launch = 0usize;
    let mut reuse_observed = 0usize;

    for _ in 0..ITERATIONS {
        let slice = memory.alloc::<u8>(BYTES).expect("alloc");
        let in_ptr = slice.device_ptr_value();
        let mut col = CudaColumn::owned(slice);
        provider
            .memset_column_recorded(&mut col, PATTERN_LAUNCH, launch_stream)
            .expect("memset_column_recorded");
        drop(col);

        let next = memory.alloc::<u8>(BYTES).expect("alloc next");
        let next_ptr = next.device_ptr_value();
        if next_ptr == in_ptr {
            reuse_observed += 1;
        }
        unsafe {
            let res = cudarc::driver::sys::cuMemsetD8_v2(next_ptr, PATTERN_NEW, BYTES);
            assert_eq!(res, cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS);
        }
        launch_handle.synchronize().expect("sync");
        default_stream.synchronize().expect("sync");

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
        "[column] iters={} reuse={} new={} launch={}",
        ITERATIONS, reuse_observed, last_writer_was_new, last_writer_was_launch
    );
    assert!(reuse_observed > 0);
    assert_eq!(last_writer_was_launch, 0);
}

/// Column-level negative test: external memory (we synthesize a
/// DLPack column without an actual tensor — sufficient because
/// the recorder's strict mode rejects on `is_external()`
/// without dereferencing). Strict mode must reject at preflight
/// before the memset is queued.
///
/// We only construct a fake DLPack column if the public
/// `CudaColumn::dlpack` constructor is reachable. As a
/// stand-in, we allocate a fresh raw device pointer via
/// `cuMemAlloc` and synthesize a `DlpackManagedTensor` around
/// it. If the test fixture cannot construct a DLPack column on
/// this host, the test skips cleanly.
///
/// Actually: constructing a DlpackManagedTensor without a real
/// producer is non-trivial. Skip this case for now — it would
/// require wiring up cudarc's DLPack support against a fake
/// tensor. The launch.rs unit test
/// `strict_rejects_legacy_at_preflight` already covers the
/// strict-mode preflight rejection logic; the column-level
/// equivalent (Dlpack) just exercises the same `is_external()`
/// branch. If a real DLPack workflow needs explicit coverage,
/// add it once the DLPack producer side has a test fixture.
#[test]
#[ignore = "requires DLPack producer fixture; covered by launch.rs unit test for the underlying logic"]
fn provider_memset_column_recorded_rejects_dlpack_column() {
    // Intentional placeholder. See doc above.
}
