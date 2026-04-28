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

/// Column-level negative test: TRUE EXTERNAL DLPack memory
/// (no `source_slice`) is rejected by the strict launch
/// recorder at preflight, before any CUDA work is queued, with
/// the "external (DLPack / ArrowDevice) memory" message.
///
/// Synthesizes a `CudaColumn::dlpack` over a null
/// `DlpackManagedTensor`. The tensor is never dereferenced —
/// the recorder only inspects `is_external()` on the column
/// — and the null pointer is drop-safe (the
/// `DlpackManagedTensor::Drop` impl checks for null before
/// invoking the deleter).
#[test]
fn provider_memset_column_recorded_rejects_external_dlpack_column() {
    use xlog_cuda::{CudaColumn, DlpackManagedTensor};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
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
    let launch_stream = pool.acquire().expect("acquire launch_stream");

    // SAFETY: null-pointer tensor is drop-safe (DlpackManagedTensor's
    // Drop impl null-checks before invoking the deleter). The recorder
    // never derefs the tensor.
    let tensor = unsafe { DlpackManagedTensor::from_raw(std::ptr::null_mut()) };
    let mut col = CudaColumn::dlpack(0, 0, device.inner().stream().clone(), tensor);
    assert!(col.is_external());

    let err = provider.memset_column_recorded(&mut col, 0xAA, launch_stream);
    match err {
        Err(XlogError::Kernel(msg)) => {
            assert!(
                msg.contains("preflight failed") && msg.contains("external"),
                "expected preflight-failed external-memory error, got {:?}",
                msg
            );
        }
        other => panic!(
            "memset_column_recorded must reject external DLPack at preflight, got {:?}",
            other
        ),
    }
}

/// Column-level positive test: an XLOG-OWNED DLPack column —
/// constructed via `CudaColumn::dlpack_xlog_owned` over a
/// runtime-backed slice — is recorded successfully by the
/// strict launch recorder. `is_external` reports false and
/// `runtime_block()` resolves to the source slice's
/// `DeviceBlock`, so the recorder treats the column the same
/// way it treats `CudaColumn::Owned`.
///
/// This locks the slice's intent: zero-copy DLPack export
/// where xlog retains ownership preserves runtime identity
/// and remains safe under the strict-recorder discipline.
/// True external DLPack producers continue to be rejected
/// (covered by the sibling test above).
#[test]
fn provider_memset_column_recorded_accepts_xlog_owned_dlpack_column() {
    use xlog_cuda::{CudaColumn, DlpackManagedTensor};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
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
    let launch_stream = pool.acquire().expect("acquire launch_stream");
    let launch_handle = pool.resolve(launch_stream).expect("resolve");

    let slice = memory.alloc::<u8>(BYTES).expect("alloc runtime-backed");
    assert!(slice.runtime_block().is_some());
    let tensor = unsafe { DlpackManagedTensor::from_raw(std::ptr::null_mut()) };
    let mut col =
        CudaColumn::dlpack_xlog_owned(Arc::new(slice), device.inner().stream().clone(), tensor);
    assert!(!col.is_external());
    assert!(col.runtime_block().is_some());

    provider
        .memset_column_recorded(&mut col, PATTERN_LAUNCH, launch_stream)
        .expect("memset_column_recorded must accept xlog-owned DLPack column");

    // Sync to make readback well-defined; we just verify that
    // strict-recorder dispatch succeeded end-to-end.
    launch_handle.synchronize().expect("sync launch");
    drop(col);
    runtime.reap_pending().expect("reap");
}

/// Filter-class slice. Proves that the migrated
/// `compare_const_mask_recorded` correctly threads the column
/// READ through the runtime: the kernel runs on a non-default
/// `launch_stream`, but the column was allocated on the default
/// alloc-stream. Dropping `input` BEFORE the kernel completes
/// must NOT free the column out from under the launch-stream
/// read.
///
/// Bug class: without `record_block_use`, `drop(input)` queues
/// `cuMemFreeAsync(alloc_stream)` with no wait on the
/// launch_stream's recorded event. The pool reuses the address
/// to satisfy the next allocation, which is then trampled with a
/// known pattern on the default stream. The launch_stream
/// kernel — still pending — reads the trampled bytes and
/// produces a corrupted mask. With the recorder, the free is
/// gated on the launch_stream event and the kernel reads the
/// original column data.
///
/// Mask check: column is filled with `(i % 5) as u32`; we
/// compare-equal against `TARGET=2`, so the expected mask is
/// `1` at every `i where i%5==2` and `0` elsewhere. Any
/// deviation is a corruption signal.
#[test]
fn provider_compare_const_mask_recorded_survives_drop_and_reuse() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CompareOp, CudaBuffer};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
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

    let launch_stream = pool.acquire().expect("acquire launch_stream");
    assert_ne!(launch_stream, StreamId::DEFAULT);
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const ROWS: usize = 1024;
    const TARGET: u32 = 2;
    const ITERATIONS: usize = 32;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![("v".to_string(), ScalarType::U32)]);

    let mut reuse_observed = 0usize;
    let mut bad_mask = 0usize;

    for iter in 0..ITERATIONS {
        // Build column data: [0,1,2,3,4,0,1,2,3,4,...] mod 5.
        let mut col_data = Vec::with_capacity(ROWS * 4);
        for i in 0..ROWS {
            col_data.extend_from_slice(&((i as u32) % 5).to_le_bytes());
        }
        let mut col_bytes = memory.alloc::<u8>(ROWS * 4).expect("alloc col");
        device
            .inner()
            .htod_sync_copy_into(&col_data, &mut col_bytes)
            .expect("htod col");
        let col_ptr = col_bytes.device_ptr_value();

        let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
        device
            .inner()
            .htod_sync_copy_into(&[ROWS as u32], &mut d_num_rows)
            .expect("htod rows");

        let input = CudaBuffer::from_columns(
            vec![col_bytes.into()],
            ROWS as u64,
            d_num_rows,
            schema.clone(),
        );

        // Migrated launch: kernel queues on launch_stream, read
        // recorded against runtime via LaunchRecorder strict.
        let d_mask = provider
            .compare_const_mask_recorded::<u32>(&input, 0, TARGET, CompareOp::Eq, launch_stream)
            .expect("compare_const_mask_recorded");

        // Drop input WITHOUT host sync. runtime.deallocate must
        // queue cuStreamWaitEvent(alloc_stream, recorded_event)
        // BEFORE cuMemFreeAsync, gating the free on the kernel's
        // read.
        drop(input);

        // Try to reuse the column slot.
        let next = memory.alloc::<u8>(ROWS * 4).expect("alloc next");
        let next_ptr = next.device_ptr_value();
        let reused = next_ptr == col_ptr;
        if reused {
            reuse_observed += 1;
        }

        // Trample the slot on default stream. With the
        // wait-event chain, this is correctly ordered AFTER the
        // kernel's read; without it, the kernel would race this
        // memset.
        unsafe { memset_sync_default(next_ptr, TRAMPLE, ROWS * 4) };

        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Verify mask integrity: any byte that doesn't match
        // (i%5 == TARGET) is a corruption signal.
        let mut readback = vec![0u8; ROWS];
        unsafe { dtoh_sync(&mut readback, *d_mask.device_ptr()) };
        for (i, &b) in readback.iter().enumerate() {
            let expected = if (i as u32) % 5 == TARGET { 1 } else { 0 };
            if b != expected {
                bad_mask += 1;
                if iter == 0 {
                    eprintln!(
                        "[compare_recorded] iter=0 row={} got={} expected={} reused={}",
                        i, b, expected, reused
                    );
                }
                break;
            }
        }

        drop(d_mask);
        drop(next);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[compare_recorded] iterations={} reuse_observed={} bad_mask={}",
        ITERATIONS, reuse_observed, bad_mask
    );
    assert!(
        reuse_observed > 0,
        "address reuse never observed across {} iterations; cannot exercise \
         the cross-stream lifetime safety path",
        ITERATIONS
    );
    assert_eq!(
        bad_mask, 0,
        "compare_const_mask_recorded produced a corrupted mask in {}/{} iterations \
         (reuse_observed={}). The recorder's read on input.column(0) failed to \
         propagate a wait-event through the deallocate path.",
        bad_mask, ITERATIONS, reuse_observed,
    );
}

/// Negative test: filter-class migrated path against legacy
/// (no-runtime) manager. Must reject before any allocation
/// happens.
#[test]
fn provider_compare_const_mask_recorded_rejects_legacy_manager() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CompareOp, CudaBuffer};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let memory = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(1024 * 1024),
    ));
    let provider =
        CudaKernelProvider::new(Arc::clone(&device), Arc::clone(&memory)).expect("legacy provider");

    let mut col_bytes = memory.alloc::<u8>(16).expect("alloc col");
    let payload = [1u32, 2, 3, 4];
    let bytes: Vec<u8> = payload.iter().flat_map(|v| v.to_le_bytes()).collect();
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut col_bytes)
        .expect("htod col");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut d_num_rows)
        .expect("htod rows");
    let input = CudaBuffer::from_columns(
        vec![col_bytes.into()],
        4,
        d_num_rows,
        Schema::new(vec![("v".to_string(), ScalarType::U32)]),
    );

    let err = provider.compare_const_mask_recorded::<u32>(
        &input,
        0,
        2u32,
        CompareOp::Eq,
        StreamId::DEFAULT,
    );
    match err {
        Err(XlogError::Kernel(msg)) => {
            assert!(
                msg.contains("requires") || msg.contains("with_runtime"),
                "expected helpful Kernel error message, got {:?}",
                msg
            );
        }
        Err(other) => panic!(
            "compare_const_mask_recorded must reject legacy manager with XlogError::Kernel, got {:?}",
            other
        ),
        Ok(_) => panic!(
            "compare_const_mask_recorded must reject legacy manager — unexpectedly returned Ok"
        ),
    }
}

/// Filter-class slice 2: column-column compare. Same drop+reuse
/// shape as `compare_const_mask_recorded`, but exercises BOTH
/// column reads being recorded before preflight.
///
/// The bug class is the same: without `record_block_use`, the
/// alloc-stream `cuMemFreeAsync` would not wait on the
/// launch_stream kernel that's still reading either column,
/// and a subsequent reuse + trample on the default stream
/// would race the kernel's read. Mask integrity proves the
/// wait-event chain held for cross-stream READS of TWO
/// distinct buffers.
///
/// Mask check: left column is `(i % 5)`, right column is
/// `(i % 4)`. We compare-equal; expected mask is `1` exactly
/// where `i%5 == i%4`. Any deviation from the expected
/// pattern is a corruption signal.
#[test]
fn provider_compare_columns_mask_recorded_survives_drop_and_reuse() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CompareOp, CudaBuffer};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
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

    let launch_stream = pool.acquire().expect("acquire launch_stream");
    assert_ne!(launch_stream, StreamId::DEFAULT);
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const ROWS: usize = 1024;
    const ITERATIONS: usize = 32;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("l".to_string(), ScalarType::U32),
        ("r".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_mask = 0usize;

    for iter in 0..ITERATIONS {
        let mut left_data = Vec::with_capacity(ROWS * 4);
        let mut right_data = Vec::with_capacity(ROWS * 4);
        for i in 0..ROWS {
            left_data.extend_from_slice(&((i as u32) % 5).to_le_bytes());
            right_data.extend_from_slice(&((i as u32) % 4).to_le_bytes());
        }

        let mut left_bytes = memory.alloc::<u8>(ROWS * 4).expect("alloc left");
        device
            .inner()
            .htod_sync_copy_into(&left_data, &mut left_bytes)
            .expect("htod left");
        let left_ptr = left_bytes.device_ptr_value();

        let mut right_bytes = memory.alloc::<u8>(ROWS * 4).expect("alloc right");
        device
            .inner()
            .htod_sync_copy_into(&right_data, &mut right_bytes)
            .expect("htod right");
        let right_ptr = right_bytes.device_ptr_value();

        let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
        device
            .inner()
            .htod_sync_copy_into(&[ROWS as u32], &mut d_num_rows)
            .expect("htod rows");

        let input = CudaBuffer::from_columns(
            vec![left_bytes.into(), right_bytes.into()],
            ROWS as u64,
            d_num_rows,
            schema.clone(),
        );

        let d_mask = provider
            .compare_columns_mask_recorded::<u32>(&input, 0, 1, CompareOp::Eq, launch_stream)
            .expect("compare_columns_mask_recorded");

        // Drop input WITHOUT host sync — the wait-event chain
        // must order the alloc_stream free of BOTH columns
        // after the launch_stream kernel's reads.
        drop(input);

        // Reuse + trample one of the slots (left). With the
        // chain, this is correctly ordered after the kernel's
        // read; without it, the kernel would race the trample.
        let next_left = memory.alloc::<u8>(ROWS * 4).expect("alloc next_left");
        let next_left_ptr = next_left.device_ptr_value();
        let reused_left = next_left_ptr == left_ptr;

        let next_right = memory.alloc::<u8>(ROWS * 4).expect("alloc next_right");
        let next_right_ptr = next_right.device_ptr_value();
        let reused_right = next_right_ptr == right_ptr;
        if reused_left || reused_right {
            reuse_observed += 1;
        }

        unsafe {
            memset_sync_default(next_left_ptr, TRAMPLE, ROWS * 4);
            memset_sync_default(next_right_ptr, TRAMPLE, ROWS * 4);
        }

        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        let mut readback = vec![0u8; ROWS];
        unsafe { dtoh_sync(&mut readback, *d_mask.device_ptr()) };
        for (i, &b) in readback.iter().enumerate() {
            let expected = if (i as u32) % 5 == (i as u32) % 4 {
                1
            } else {
                0
            };
            if b != expected {
                bad_mask += 1;
                if iter == 0 {
                    eprintln!(
                        "[compare_cols_recorded] iter=0 row={} got={} expected={} \
                         reused_left={} reused_right={}",
                        i, b, expected, reused_left, reused_right
                    );
                }
                break;
            }
        }

        drop(d_mask);
        drop(next_left);
        drop(next_right);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[compare_cols_recorded] iterations={} reuse_observed={} bad_mask={}",
        ITERATIONS, reuse_observed, bad_mask
    );
    assert!(
        reuse_observed > 0,
        "address reuse never observed across {} iterations; cannot exercise \
         the cross-stream lifetime safety path",
        ITERATIONS
    );
    assert_eq!(
        bad_mask, 0,
        "compare_columns_mask_recorded produced a corrupted mask in {}/{} iterations \
         (reuse_observed={}). The recorder's reads on input.column(left/right) failed \
         to propagate a wait-event through the deallocate path.",
        bad_mask, ITERATIONS, reuse_observed,
    );
}

/// Negative test: column-column migrated path against
/// no-runtime manager. Must reject before any allocation
/// happens.
#[test]
fn provider_compare_columns_mask_recorded_rejects_legacy_manager() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CompareOp, CudaBuffer};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let memory = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(1024 * 1024),
    ));
    let provider =
        CudaKernelProvider::new(Arc::clone(&device), Arc::clone(&memory)).expect("legacy provider");

    let mut left_bytes = memory.alloc::<u8>(16).expect("alloc left");
    let mut right_bytes = memory.alloc::<u8>(16).expect("alloc right");
    let payload = [1u32, 2, 3, 4];
    let bytes: Vec<u8> = payload.iter().flat_map(|v| v.to_le_bytes()).collect();
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut left_bytes)
        .expect("htod left");
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut right_bytes)
        .expect("htod right");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut d_num_rows)
        .expect("htod rows");
    let input = CudaBuffer::from_columns(
        vec![left_bytes.into(), right_bytes.into()],
        4,
        d_num_rows,
        Schema::new(vec![
            ("l".to_string(), ScalarType::U32),
            ("r".to_string(), ScalarType::U32),
        ]),
    );

    let err = provider.compare_columns_mask_recorded::<u32>(
        &input,
        0,
        1,
        CompareOp::Eq,
        StreamId::DEFAULT,
    );
    match err {
        Err(XlogError::Kernel(msg)) => {
            assert!(
                msg.contains("requires") || msg.contains("with_runtime"),
                "expected helpful Kernel error message, got {:?}",
                msg
            );
        }
        Err(other) => panic!(
            "compare_columns_mask_recorded must reject legacy manager with XlogError::Kernel, got {:?}",
            other
        ),
        Ok(_) => panic!(
            "compare_columns_mask_recorded must reject legacy manager — unexpectedly returned Ok"
        ),
    }
}

/// Filter-class slice 3: COMPACT path. The compact pipeline is
/// a multi-kernel chain — `mask_clamp_rows` →
/// `multiblock_scan_phase1` (and inplace+phase3 when
/// `num_blocks > 1`) → `capture_compact_count` →
/// `cu_stream.synchronize()` → host scalar read → per-column
/// `compact_bytes_by_mask`. Every kernel runs on the same
/// explicit `launch_stream` via `launch_on_stream`, and the
/// host scalar read is explicitly ordered against the
/// launch_stream rather than relying on default-stream
/// implicit sync (which non-blocking pool streams do NOT get).
///
/// The bug class extends to ALL caller-provided buffers: input
/// columns AND `d_mask`. Without the recorder, dropping either
/// one without host sync would let the alloc-stream
/// `cuMemFreeAsync` fire while the launch_stream chain is
/// still reading them, and a subsequent reuse + trample on
/// the default stream would corrupt the kernel's reads. With
/// the recorder, both are recorded as reads BEFORE preflight
/// and the wait-event chain at deallocate gates the frees.
///
/// Output check: input col [10, 20, 30, 40, 50] with mask
/// [1, 0, 1, 0, 1] must compact to [10, 30, 50] (with the
/// remaining 2 of 5 row_cap slots untouched / unspecified).
#[test]
fn provider_compact_buffer_by_device_mask_counted_recorded_survives_drop_and_reuse() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::CudaBuffer;

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
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

    let launch_stream = pool.acquire().expect("acquire launch_stream");
    assert_ne!(launch_stream, StreamId::DEFAULT);
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const ROWS: usize = 1024;
    const ITERATIONS: usize = 32;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![("v".to_string(), ScalarType::U32)]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        // Build column [0, 1, 2, ..., ROWS-1] in u32 on alloc_stream.
        let mut col_data = Vec::with_capacity(ROWS * 4);
        for i in 0..ROWS {
            col_data.extend_from_slice(&(i as u32).to_le_bytes());
        }
        let mut col_bytes = memory.alloc::<u8>(ROWS * 4).expect("alloc col");
        device
            .inner()
            .htod_sync_copy_into(&col_data, &mut col_bytes)
            .expect("htod col");
        let col_ptr = col_bytes.device_ptr_value();

        // Mask: keep every other row → [1,0,1,0,...].
        let mut mask_data = vec![0u8; ROWS];
        for i in (0..ROWS).step_by(2) {
            mask_data[i] = 1;
        }
        let mut d_mask = memory.alloc::<u8>(ROWS).expect("alloc mask");
        device
            .inner()
            .htod_sync_copy_into(&mask_data, &mut d_mask)
            .expect("htod mask");
        let mask_ptr = d_mask.device_ptr_value();

        let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
        device
            .inner()
            .htod_sync_copy_into(&[ROWS as u32], &mut d_num_rows)
            .expect("htod rows");

        let input = CudaBuffer::from_columns(
            vec![col_bytes.into()],
            ROWS as u64,
            d_num_rows,
            schema.clone(),
        );

        // Migrated compact: queues the entire chain on
        // launch_stream, returns AFTER the host scalar read
        // sync but BEFORE compact_bytes_by_mask completes.
        let output_buf = provider
            .compact_buffer_by_device_mask_counted_recorded(&input, &d_mask, launch_stream)
            .expect("compact_buffer_by_device_mask_counted_recorded");

        // Drop input AND d_mask without host sync. The
        // recorder's commit must have ordered the alloc-stream
        // frees of both AFTER the launch_stream chain.
        drop(input);
        drop(d_mask);

        // Try to reuse the freed slots — pool may serve
        // either the column or mask address (or both).
        let next_a = memory.alloc::<u8>(ROWS * 4).expect("alloc next_a");
        let next_a_ptr = next_a.device_ptr_value();
        let next_b = memory.alloc::<u8>(ROWS).expect("alloc next_b");
        let next_b_ptr = next_b.device_ptr_value();
        let reused = next_a_ptr == col_ptr
            || next_a_ptr == mask_ptr
            || next_b_ptr == col_ptr
            || next_b_ptr == mask_ptr;
        if reused {
            reuse_observed += 1;
        }

        // Trample whatever was reused (and the other slot
        // too — harmless on a fresh alloc) on default stream.
        unsafe {
            memset_sync_default(next_a_ptr, TRAMPLE, ROWS * 4);
            memset_sync_default(next_b_ptr, TRAMPLE, ROWS);
        }

        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Read the compacted output column. Expected:
        // [0, 2, 4, ..., ROWS-2] in the first ROWS/2 slots.
        // Mask preserved every even index, so output[k] should
        // equal 2*k as u32 for k in 0..ROWS/2.
        let out_col = output_buf.column(0).expect("output column 0 must exist");
        let mut readback = vec![0u8; (ROWS / 2) * 4];
        unsafe { dtoh_sync(&mut readback, *out_col.device_ptr()) };
        let mut local_bad = 0usize;
        for k in 0..(ROWS / 2) {
            let bytes = &readback[k * 4..k * 4 + 4];
            let v = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let expected = (k * 2) as u32;
            if v != expected {
                local_bad += 1;
                if iter == 0 && local_bad <= 4 {
                    eprintln!(
                        "[compact_recorded] iter=0 out[{}]={} expected={} reused={}",
                        k, v, expected, reused
                    );
                }
            }
        }
        if local_bad > 0 {
            bad_output += 1;
        }

        drop(output_buf);
        drop(next_a);
        drop(next_b);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[compact_recorded] iterations={} reuse_observed={} bad_output={}",
        ITERATIONS, reuse_observed, bad_output
    );
    assert!(
        reuse_observed > 0,
        "address reuse never observed across {} iterations; cannot exercise \
         the cross-stream lifetime safety path",
        ITERATIONS
    );
    assert_eq!(
        bad_output, 0,
        "compact_buffer_by_device_mask_counted_recorded produced corrupted output \
         in {}/{} iterations (reuse_observed={}). The recorder's reads on input.column(0) \
         and/or d_mask failed to propagate a wait-event through the deallocate path, \
         OR the host scalar read was not properly ordered against the launch_stream.",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Negative test: compact migrated path against legacy
/// (no-runtime) manager. Must reject before any allocation.
#[test]
fn provider_compact_buffer_by_device_mask_counted_recorded_rejects_legacy_manager() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::CudaBuffer;

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let memory = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(1024 * 1024),
    ));
    let provider =
        CudaKernelProvider::new(Arc::clone(&device), Arc::clone(&memory)).expect("legacy provider");

    let mut col_bytes = memory.alloc::<u8>(16).expect("alloc col");
    let payload = [10u32, 20, 30, 40];
    let bytes: Vec<u8> = payload.iter().flat_map(|v| v.to_le_bytes()).collect();
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut col_bytes)
        .expect("htod col");
    let mut d_mask = memory.alloc::<u8>(4).expect("alloc mask");
    device
        .inner()
        .htod_sync_copy_into(&[1u8, 0, 1, 0], &mut d_mask)
        .expect("htod mask");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut d_num_rows)
        .expect("htod rows");
    let input = CudaBuffer::from_columns(
        vec![col_bytes.into()],
        4,
        d_num_rows,
        Schema::new(vec![("v".to_string(), ScalarType::U32)]),
    );

    let err =
        provider.compact_buffer_by_device_mask_counted_recorded(&input, &d_mask, StreamId::DEFAULT);
    match err {
        Err(XlogError::Kernel(msg)) => {
            assert!(
                msg.contains("requires") || msg.contains("with_runtime"),
                "expected helpful Kernel error message, got {:?}",
                msg
            );
        }
        Err(other) => panic!(
            "compact_buffer_by_device_mask_counted_recorded must reject legacy manager \
             with XlogError::Kernel, got {:?}",
            other
        ),
        Ok(_) => panic!(
            "compact_buffer_by_device_mask_counted_recorded must reject legacy manager — \
             unexpectedly returned Ok"
        ),
    }
}

/// End-to-end filter slice: the first COMPOSED migrated path.
///
/// `filter_recorded::<u32>` chains
/// `compare_const_mask_recorded` then
/// `compact_buffer_by_device_mask_counted_recorded` on a
/// single `launch_stream`. Each primitive builds its own
/// recorder; the runtime's `record_block_use` APPENDS every
/// recorded event to the live entry's
/// `last_use_events: Vec<CudaEvent>`, and `deallocate` waits
/// on EVERY event before `cuMemFreeAsync`. So the compare's
/// commit and the compact's later commit each push their own
/// event for `input.column[0]` (and shared buffers), and the
/// deallocate gates the free behind both — chaining the
/// cross-stream lifetime safety end-to-end.
///
/// Bug class: caller drops `input` after `filter_recorded`
/// returns, without host sync. Per-column compact_bytes_by_mask
/// kernels are still in flight on launch_stream — the
/// function only synchronized launch_stream BEFORE the host
/// scalar read in the middle of the chain; the per-column
/// compact kernels enqueue AFTER that sync. Without proper
/// recording, the alloc-stream `cuMemFreeAsync` would race
/// the still-pending compact reads of `input.column[i]`, and
/// a reuse + trample on the default stream would corrupt the
/// kernels' reads → output column would have wrong contents.
///
/// Mask predicate: `i % 5 == PREDICATE_KEY`. Input column 0
/// holds `i % 5`, column 1 holds `i * 100`. Expected output
/// rows are `{i : i%5 == PREDICATE_KEY}`.
#[test]
fn provider_filter_recorded_survives_drop_and_reuse() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CompareOp, CudaBuffer};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
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

    let launch_stream = pool.acquire().expect("acquire launch_stream");
    assert_ne!(launch_stream, StreamId::DEFAULT);
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const ROWS: usize = 1024;
    const PREDICATE_KEY: u32 = 2;
    const ITERATIONS: usize = 32;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        // Column 0 = i%5 (predicate), column 1 = i*100 (payload).
        let mut k_data = Vec::with_capacity(ROWS * 4);
        let mut v_data = Vec::with_capacity(ROWS * 4);
        for i in 0..ROWS {
            k_data.extend_from_slice(&((i as u32) % 5).to_le_bytes());
            v_data.extend_from_slice(&((i as u32) * 100).to_le_bytes());
        }
        let mut k_bytes = memory.alloc::<u8>(ROWS * 4).expect("alloc k");
        device
            .inner()
            .htod_sync_copy_into(&k_data, &mut k_bytes)
            .expect("htod k");
        let k_ptr = k_bytes.device_ptr_value();

        let mut v_bytes = memory.alloc::<u8>(ROWS * 4).expect("alloc v");
        device
            .inner()
            .htod_sync_copy_into(&v_data, &mut v_bytes)
            .expect("htod v");
        let v_ptr = v_bytes.device_ptr_value();

        let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
        device
            .inner()
            .htod_sync_copy_into(&[ROWS as u32], &mut d_num_rows)
            .expect("htod rows");

        let input = CudaBuffer::from_columns(
            vec![k_bytes.into(), v_bytes.into()],
            ROWS as u64,
            d_num_rows,
            schema.clone(),
        );

        let output_buf = provider
            .filter_recorded::<u32>(&input, 0, PREDICATE_KEY, CompareOp::Eq, launch_stream)
            .expect("filter_recorded");

        // Drop input WITHOUT host sync. Per-column
        // compact_bytes_by_mask kernels are still pending on
        // launch_stream.
        drop(input);

        // Reuse + trample. Pool may serve k or v slot.
        let next_a = memory.alloc::<u8>(ROWS * 4).expect("alloc next_a");
        let next_a_ptr = next_a.device_ptr_value();
        let next_b = memory.alloc::<u8>(ROWS * 4).expect("alloc next_b");
        let next_b_ptr = next_b.device_ptr_value();
        let reused = next_a_ptr == k_ptr
            || next_a_ptr == v_ptr
            || next_b_ptr == k_ptr
            || next_b_ptr == v_ptr;
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            memset_sync_default(next_a_ptr, TRAMPLE, ROWS * 4);
            memset_sync_default(next_b_ptr, TRAMPLE, ROWS * 4);
        }

        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Verify: predicate keeps i where i%5 == PREDICATE_KEY.
        // Output rows = {2, 7, 12, ..., 1022} → 205 rows.
        let expected_count: usize = (0..ROWS)
            .filter(|i| (*i as u32) % 5 == PREDICATE_KEY)
            .count();
        let k_out = output_buf.column(0).expect("out col 0");
        let v_out = output_buf.column(1).expect("out col 1");

        let mut k_back = vec![0u8; expected_count * 4];
        let mut v_back = vec![0u8; expected_count * 4];
        unsafe {
            dtoh_sync(&mut k_back, *k_out.device_ptr());
            dtoh_sync(&mut v_back, *v_out.device_ptr());
        }

        let mut local_bad = 0usize;
        let mut kept_idx = 0usize;
        for i in 0..ROWS {
            if (i as u32) % 5 != PREDICATE_KEY {
                continue;
            }
            let kb = &k_back[kept_idx * 4..kept_idx * 4 + 4];
            let vb = &v_back[kept_idx * 4..kept_idx * 4 + 4];
            let kv = u32::from_le_bytes([kb[0], kb[1], kb[2], kb[3]]);
            let vv = u32::from_le_bytes([vb[0], vb[1], vb[2], vb[3]]);
            let k_expected = (i as u32) % 5;
            let v_expected = (i as u32) * 100;
            if kv != k_expected || vv != v_expected {
                local_bad += 1;
                if iter == 0 && local_bad <= 4 {
                    eprintln!(
                        "[filter_recorded] iter=0 row={} kept={} k={}/{} v={}/{} reused={}",
                        i, kept_idx, kv, k_expected, vv, v_expected, reused
                    );
                }
            }
            kept_idx += 1;
        }
        if local_bad > 0 {
            bad_output += 1;
        }

        drop(output_buf);
        drop(next_a);
        drop(next_b);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[filter_recorded] iterations={} reuse_observed={} bad_output={}",
        ITERATIONS, reuse_observed, bad_output
    );
    assert!(
        reuse_observed > 0,
        "address reuse never observed across {} iterations; cannot exercise \
         the cross-stream lifetime safety path",
        ITERATIONS
    );
    assert_eq!(
        bad_output, 0,
        "filter_recorded produced corrupted output in {}/{} iterations \
         (reuse_observed={}). Either the recorder chain (compare → compact) \
         failed to retain every recorded input event until deallocate, OR an \
         in-flight compact_bytes_by_mask read of an input column was clobbered \
         by the alloc-stream reuse + trample.",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Negative test: composed migrated path against no-runtime
/// manager. The first underlying primitive
/// (`compare_const_mask_recorded`) must reject before any
/// allocation / kernel — same loud-failure contract.
#[test]
fn provider_filter_recorded_rejects_legacy_manager() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CompareOp, CudaBuffer};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let memory = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(1024 * 1024),
    ));
    let provider =
        CudaKernelProvider::new(Arc::clone(&device), Arc::clone(&memory)).expect("legacy provider");

    let mut col_bytes = memory.alloc::<u8>(16).expect("alloc col");
    let payload = [1u32, 2, 3, 4];
    let bytes: Vec<u8> = payload.iter().flat_map(|v| v.to_le_bytes()).collect();
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut col_bytes)
        .expect("htod col");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut d_num_rows)
        .expect("htod rows");
    let input = CudaBuffer::from_columns(
        vec![col_bytes.into()],
        4,
        d_num_rows,
        Schema::new(vec![("v".to_string(), ScalarType::U32)]),
    );

    let err = provider.filter_recorded::<u32>(&input, 0, 2u32, CompareOp::Eq, StreamId::DEFAULT);
    match err {
        Err(XlogError::Kernel(msg)) => {
            assert!(
                msg.contains("requires") || msg.contains("with_runtime"),
                "expected helpful Kernel error message, got {:?}",
                msg
            );
        }
        Err(other) => panic!(
            "filter_recorded must reject legacy manager with XlogError::Kernel, got {:?}",
            other
        ),
        Ok(_) => {
            panic!("filter_recorded must reject legacy manager — unexpectedly returned Ok")
        }
    }
}

/// End-to-end COLUMN-COLUMN filter slice. Closes the filter
/// predicate matrix: `filter_recorded` covers `col <op> const`,
/// this covers `col[left] <op> col[right]`.
///
/// Composes `compare_columns_mask_recorded` (which records
/// BOTH input columns + `input.num_rows_device()` as reads)
/// with `compact_buffer_by_device_mask_counted_recorded`
/// (which records every input column + `d_mask` +
/// `input.num_rows_device()` as reads). Every shared input
/// buffer accumulates two events on `launch_stream` (one per
/// primitive's commit); deallocate waits on every event in
/// `last_use_events: Vec<CudaEvent>`, so dropping `input`
/// after the function returns is gated on BOTH operations'
/// completion.
///
/// Predicate: `col[0] == col[1]`. Column 0 = `i % 5`,
/// column 1 = `i % 4`. Expected output keeps rows where
/// `i%5 == i%4`. Column 2 (payload `i*100`) confirms full
/// row data integrity post-compact.
#[test]
fn provider_filter_columns_recorded_survives_drop_and_reuse() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CompareOp, CudaBuffer};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
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

    let launch_stream = pool.acquire().expect("acquire launch_stream");
    assert_ne!(launch_stream, StreamId::DEFAULT);
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const ROWS: usize = 1024;
    const ITERATIONS: usize = 32;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("l".to_string(), ScalarType::U32),
        ("r".to_string(), ScalarType::U32),
        ("p".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        let mut l_data = Vec::with_capacity(ROWS * 4);
        let mut r_data = Vec::with_capacity(ROWS * 4);
        let mut p_data = Vec::with_capacity(ROWS * 4);
        for i in 0..ROWS {
            l_data.extend_from_slice(&((i as u32) % 5).to_le_bytes());
            r_data.extend_from_slice(&((i as u32) % 4).to_le_bytes());
            p_data.extend_from_slice(&((i as u32) * 100).to_le_bytes());
        }
        let mut l_bytes = memory.alloc::<u8>(ROWS * 4).expect("alloc l");
        device
            .inner()
            .htod_sync_copy_into(&l_data, &mut l_bytes)
            .expect("htod l");
        let l_ptr = l_bytes.device_ptr_value();

        let mut r_bytes = memory.alloc::<u8>(ROWS * 4).expect("alloc r");
        device
            .inner()
            .htod_sync_copy_into(&r_data, &mut r_bytes)
            .expect("htod r");
        let r_ptr = r_bytes.device_ptr_value();

        let mut p_bytes = memory.alloc::<u8>(ROWS * 4).expect("alloc p");
        device
            .inner()
            .htod_sync_copy_into(&p_data, &mut p_bytes)
            .expect("htod p");
        let p_ptr = p_bytes.device_ptr_value();

        let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
        device
            .inner()
            .htod_sync_copy_into(&[ROWS as u32], &mut d_num_rows)
            .expect("htod rows");

        let input = CudaBuffer::from_columns(
            vec![l_bytes.into(), r_bytes.into(), p_bytes.into()],
            ROWS as u64,
            d_num_rows,
            schema.clone(),
        );

        let output_buf = provider
            .filter_columns_recorded::<u32>(&input, 0, 1, CompareOp::Eq, launch_stream)
            .expect("filter_columns_recorded");

        // Drop input WITHOUT host sync. Per-column
        // compact_bytes_by_mask kernels for ALL THREE columns
        // are still pending on launch_stream.
        drop(input);

        // Reuse + trample three slots.
        let next_a = memory.alloc::<u8>(ROWS * 4).expect("alloc next_a");
        let next_a_ptr = next_a.device_ptr_value();
        let next_b = memory.alloc::<u8>(ROWS * 4).expect("alloc next_b");
        let next_b_ptr = next_b.device_ptr_value();
        let next_c = memory.alloc::<u8>(ROWS * 4).expect("alloc next_c");
        let next_c_ptr = next_c.device_ptr_value();
        let reused = [next_a_ptr, next_b_ptr, next_c_ptr]
            .iter()
            .any(|p| *p == l_ptr || *p == r_ptr || *p == p_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            memset_sync_default(next_a_ptr, TRAMPLE, ROWS * 4);
            memset_sync_default(next_b_ptr, TRAMPLE, ROWS * 4);
            memset_sync_default(next_c_ptr, TRAMPLE, ROWS * 4);
        }

        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Verify: predicate keeps i where i%5 == i%4.
        let kept: Vec<usize> = (0..ROWS)
            .filter(|i| (*i as u32) % 5 == (*i as u32) % 4)
            .collect();
        let expected_count = kept.len();
        let l_out = output_buf.column(0).expect("out col 0");
        let r_out = output_buf.column(1).expect("out col 1");
        let p_out = output_buf.column(2).expect("out col 2");

        let mut l_back = vec![0u8; expected_count * 4];
        let mut r_back = vec![0u8; expected_count * 4];
        let mut p_back = vec![0u8; expected_count * 4];
        unsafe {
            dtoh_sync(&mut l_back, *l_out.device_ptr());
            dtoh_sync(&mut r_back, *r_out.device_ptr());
            dtoh_sync(&mut p_back, *p_out.device_ptr());
        }

        let mut local_bad = 0usize;
        for (k, &i) in kept.iter().enumerate() {
            let lb = &l_back[k * 4..k * 4 + 4];
            let rb = &r_back[k * 4..k * 4 + 4];
            let pb = &p_back[k * 4..k * 4 + 4];
            let lv = u32::from_le_bytes([lb[0], lb[1], lb[2], lb[3]]);
            let rv = u32::from_le_bytes([rb[0], rb[1], rb[2], rb[3]]);
            let pv = u32::from_le_bytes([pb[0], pb[1], pb[2], pb[3]]);
            let l_e = (i as u32) % 5;
            let r_e = (i as u32) % 4;
            let p_e = (i as u32) * 100;
            if lv != l_e || rv != r_e || pv != p_e {
                local_bad += 1;
                if iter == 0 && local_bad <= 4 {
                    eprintln!(
                        "[filter_cols_recorded] iter=0 row={} kept={} l={}/{} r={}/{} p={}/{} reused={}",
                        i, k, lv, l_e, rv, r_e, pv, p_e, reused
                    );
                }
            }
        }
        if local_bad > 0 {
            bad_output += 1;
        }

        drop(output_buf);
        drop(next_a);
        drop(next_b);
        drop(next_c);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[filter_cols_recorded] iterations={} reuse_observed={} bad_output={}",
        ITERATIONS, reuse_observed, bad_output
    );
    assert!(
        reuse_observed > 0,
        "address reuse never observed across {} iterations; cannot exercise \
         the cross-stream lifetime safety path",
        ITERATIONS
    );
    assert_eq!(
        bad_output, 0,
        "filter_columns_recorded produced corrupted output in {}/{} iterations \
         (reuse_observed={}). Either the recorder chain (compare-cols → compact) \
         failed to retain events on every shared input column, OR an in-flight \
         compact_bytes_by_mask read of an input column was clobbered by the \
         alloc-stream reuse + trample.",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Negative test: column-column composed migrated path
/// against no-runtime manager. The first underlying primitive
/// (`compare_columns_mask_recorded`) must reject before any
/// allocation / kernel launch.
#[test]
fn provider_filter_columns_recorded_rejects_legacy_manager() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CompareOp, CudaBuffer};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let memory = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(1024 * 1024),
    ));
    let provider =
        CudaKernelProvider::new(Arc::clone(&device), Arc::clone(&memory)).expect("legacy provider");

    let mut l_bytes = memory.alloc::<u8>(16).expect("alloc l");
    let mut r_bytes = memory.alloc::<u8>(16).expect("alloc r");
    let payload = [1u32, 2, 3, 4];
    let bytes: Vec<u8> = payload.iter().flat_map(|v| v.to_le_bytes()).collect();
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut l_bytes)
        .expect("htod l");
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut r_bytes)
        .expect("htod r");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut d_num_rows)
        .expect("htod rows");
    let input = CudaBuffer::from_columns(
        vec![l_bytes.into(), r_bytes.into()],
        4,
        d_num_rows,
        Schema::new(vec![
            ("l".to_string(), ScalarType::U32),
            ("r".to_string(), ScalarType::U32),
        ]),
    );

    let err =
        provider.filter_columns_recorded::<u32>(&input, 0, 1, CompareOp::Eq, StreamId::DEFAULT);
    match err {
        Err(XlogError::Kernel(msg)) => {
            assert!(
                msg.contains("requires") || msg.contains("with_runtime"),
                "expected helpful Kernel error message, got {:?}",
                msg
            );
        }
        Err(other) => panic!(
            "filter_columns_recorded must reject legacy manager with XlogError::Kernel, \
             got {:?}",
            other
        ),
        Ok(_) => {
            panic!("filter_columns_recorded must reject legacy manager — unexpectedly returned Ok")
        }
    }
}

/// Fused-recorded slice: the migrated fused
/// `compare+scan+compact` fast path. `filter_recorded::<u32>`
/// is already covered by
/// `provider_filter_recorded_survives_drop_and_reuse` and now
/// dispatches through `filter_fused_scan_recorded`. This test
/// adds the f64 case explicitly — different scalar type, same
/// drop+reuse contract.
///
/// The fused path is a single launch that produces
/// `(d_mask, d_prefix_sum, d_block_sums)` together; the rest
/// of the chain (`multiblock_scan_phase3`,
/// `capture_compact_count`, `cu_stream.synchronize()`,
/// per-column `compact_bytes_by_mask`) matches the non-fused
/// recorded compact tail.
///
/// Predicate: `column[0] == TARGET` (f64 equality, exact bit
/// match for the integral payload values used here). Input
/// column is `(i % 5) as f64`; expected kept rows = i where
/// `i%5 == TARGET as usize`.
#[test]
fn provider_filter_fused_scan_recorded_f64_survives_drop_and_reuse() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CompareOp, CudaBuffer};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
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

    let launch_stream = pool.acquire().expect("acquire launch_stream");
    assert_ne!(launch_stream, StreamId::DEFAULT);
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const ROWS: usize = 1024;
    const TARGET: f64 = 2.0;
    const ITERATIONS: usize = 32;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::F64),
        ("v".to_string(), ScalarType::F64),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        let mut k_data = Vec::with_capacity(ROWS * 8);
        let mut v_data = Vec::with_capacity(ROWS * 8);
        for i in 0..ROWS {
            k_data.extend_from_slice(&((i as f64) % 5.0).to_le_bytes());
            v_data.extend_from_slice(&((i as f64) * 100.0).to_le_bytes());
        }
        let mut k_bytes = memory.alloc::<u8>(ROWS * 8).expect("alloc k");
        device
            .inner()
            .htod_sync_copy_into(&k_data, &mut k_bytes)
            .expect("htod k");
        let k_ptr = k_bytes.device_ptr_value();

        let mut v_bytes = memory.alloc::<u8>(ROWS * 8).expect("alloc v");
        device
            .inner()
            .htod_sync_copy_into(&v_data, &mut v_bytes)
            .expect("htod v");
        let v_ptr = v_bytes.device_ptr_value();

        let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
        device
            .inner()
            .htod_sync_copy_into(&[ROWS as u32], &mut d_num_rows)
            .expect("htod rows");

        let input = CudaBuffer::from_columns(
            vec![k_bytes.into(), v_bytes.into()],
            ROWS as u64,
            d_num_rows,
            schema.clone(),
        );

        let output_buf = provider
            .filter_fused_scan_recorded::<f64>(&input, 0, TARGET, CompareOp::Eq, launch_stream)
            .expect("filter_fused_scan_recorded::<f64>");

        // Drop input WITHOUT host sync. Per-column
        // compact_bytes_by_mask kernels are still pending on
        // launch_stream.
        drop(input);

        let next_a = memory.alloc::<u8>(ROWS * 8).expect("alloc next_a");
        let next_a_ptr = next_a.device_ptr_value();
        let next_b = memory.alloc::<u8>(ROWS * 8).expect("alloc next_b");
        let next_b_ptr = next_b.device_ptr_value();
        let reused = next_a_ptr == k_ptr
            || next_a_ptr == v_ptr
            || next_b_ptr == k_ptr
            || next_b_ptr == v_ptr;
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            memset_sync_default(next_a_ptr, TRAMPLE, ROWS * 8);
            memset_sync_default(next_b_ptr, TRAMPLE, ROWS * 8);
        }

        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        let target_idx = TARGET as usize;
        let kept: Vec<usize> = (0..ROWS).filter(|i| (*i % 5) == target_idx).collect();
        let expected_count = kept.len();
        let k_out = output_buf.column(0).expect("out col 0");
        let v_out = output_buf.column(1).expect("out col 1");

        let mut k_back = vec![0u8; expected_count * 8];
        let mut v_back = vec![0u8; expected_count * 8];
        unsafe {
            dtoh_sync(&mut k_back, *k_out.device_ptr());
            dtoh_sync(&mut v_back, *v_out.device_ptr());
        }

        let mut local_bad = 0usize;
        for (k, &i) in kept.iter().enumerate() {
            let kb = &k_back[k * 8..k * 8 + 8];
            let vb = &v_back[k * 8..k * 8 + 8];
            let kv = f64::from_le_bytes([kb[0], kb[1], kb[2], kb[3], kb[4], kb[5], kb[6], kb[7]]);
            let vv = f64::from_le_bytes([vb[0], vb[1], vb[2], vb[3], vb[4], vb[5], vb[6], vb[7]]);
            let k_expected = (i as f64) % 5.0;
            let v_expected = (i as f64) * 100.0;
            if kv != k_expected || vv != v_expected {
                local_bad += 1;
                if iter == 0 && local_bad <= 4 {
                    eprintln!(
                        "[fused_recorded_f64] iter=0 row={} kept={} k={}/{} v={}/{} reused={}",
                        i, k, kv, k_expected, vv, v_expected, reused
                    );
                }
            }
        }
        if local_bad > 0 {
            bad_output += 1;
        }

        drop(output_buf);
        drop(next_a);
        drop(next_b);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[fused_recorded_f64] iterations={} reuse_observed={} bad_output={}",
        ITERATIONS, reuse_observed, bad_output
    );
    assert!(
        reuse_observed > 0,
        "address reuse never observed across {} iterations; cannot exercise \
         the cross-stream lifetime safety path",
        ITERATIONS
    );
    assert_eq!(
        bad_output, 0,
        "filter_fused_scan_recorded::<f64> produced corrupted output in {}/{} iterations \
         (reuse_observed={}). Either the fused phase1 launch's reads of input.column[0] \
         were not properly recorded, or one of the chain steps (scan/phase3/capture/sync/\
         compact) raced an alloc-stream reuse + trample.",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Negative test: fused-recorded path against a no-runtime
/// manager. Must reject before any allocation / kernel.
#[test]
fn provider_filter_fused_scan_recorded_rejects_legacy_manager() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CompareOp, CudaBuffer};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let memory = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(1024 * 1024),
    ));
    let provider =
        CudaKernelProvider::new(Arc::clone(&device), Arc::clone(&memory)).expect("legacy provider");

    let mut col_bytes = memory.alloc::<u8>(16).expect("alloc col");
    let payload = [1u32, 2, 3, 4];
    let bytes: Vec<u8> = payload.iter().flat_map(|v| v.to_le_bytes()).collect();
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut col_bytes)
        .expect("htod col");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut d_num_rows)
        .expect("htod rows");
    let input = CudaBuffer::from_columns(
        vec![col_bytes.into()],
        4,
        d_num_rows,
        Schema::new(vec![("v".to_string(), ScalarType::U32)]),
    );

    let err = provider.filter_fused_scan_recorded::<u32>(
        &input,
        0,
        2u32,
        CompareOp::Eq,
        StreamId::DEFAULT,
    );
    match err {
        Err(XlogError::Kernel(msg)) => {
            assert!(
                msg.contains("requires") || msg.contains("with_runtime"),
                "expected helpful Kernel error message, got {:?}",
                msg
            );
        }
        Err(other) => panic!(
            "filter_fused_scan_recorded must reject legacy manager with XlogError::Kernel, \
             got {:?}",
            other
        ),
        Ok(_) => panic!(
            "filter_fused_scan_recorded must reject legacy manager — unexpectedly returned Ok"
        ),
    }
}
