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

/// Slice #5 sort: drop+reuse test for `sort_recorded` with
/// u32 keys. The whole sort chain (init_indices → 8-pass radix
/// sort → multi-column gather) runs on the explicit
/// `launch_stream`. After the call returns, the gather
/// kernels for every input column are still in flight on
/// launch_stream — the test drops the input WITHOUT host sync,
/// reuses+tramples the column slot on the default stream, and
/// asserts the sorted output is bit-correct.
///
/// Input: column 0 is a reverse-sequence `[ROWS-1, ROWS-2,
/// ..., 0]` of u32; column 1 is a payload identifying each
/// row. Sorting by column 0 ascending must produce
/// `[(0, payload_for(0)), (1, payload_for(1)), ...]`. Any
/// mismatch implies the gather kernel read trampled bytes
/// instead of the original column.
#[test]
fn provider_sort_recorded_survives_drop_and_reuse() {
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
    const ITERATIONS: usize = 16;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        // Reverse-sequence keys, payload = row index.
        let mut k_data = Vec::with_capacity(ROWS * 4);
        let mut v_data = Vec::with_capacity(ROWS * 4);
        for i in 0..ROWS {
            k_data.extend_from_slice(&((ROWS - 1 - i) as u32).to_le_bytes());
            v_data.extend_from_slice(&((ROWS - 1 - i) as u32).to_le_bytes());
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

        let sorted = provider
            .sort_recorded(&input, &[0], launch_stream)
            .expect("sort_recorded");

        // Drop input WITHOUT host sync.
        drop(input);

        // Reuse + trample the original column slots.
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

        // Verify sorted output: column 0 ascending [0, 1, ...,
        // ROWS-1] and column 1 mirrors column 0 (payload was
        // ROWS-1-i which is the same value on the same row).
        let k_out = sorted.column(0).expect("out k");
        let v_out = sorted.column(1).expect("out v");
        let mut k_back = vec![0u8; ROWS * 4];
        let mut v_back = vec![0u8; ROWS * 4];
        unsafe {
            dtoh_sync(&mut k_back, *k_out.device_ptr());
            dtoh_sync(&mut v_back, *v_out.device_ptr());
        }
        let mut local_bad = 0usize;
        for i in 0..ROWS {
            let kb = &k_back[i * 4..i * 4 + 4];
            let vb = &v_back[i * 4..i * 4 + 4];
            let kv = u32::from_le_bytes([kb[0], kb[1], kb[2], kb[3]]);
            let vv = u32::from_le_bytes([vb[0], vb[1], vb[2], vb[3]]);
            let expected = i as u32;
            if kv != expected || vv != expected {
                local_bad += 1;
                if iter == 0 && local_bad <= 4 {
                    eprintln!(
                        "[sort_recorded] iter=0 row={} k={}/{} v={}/{} reused={}",
                        i, kv, expected, vv, expected, reused
                    );
                }
            }
        }
        if local_bad > 0 {
            bad_output += 1;
        }

        drop(sorted);
        drop(next_a);
        drop(next_b);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[sort_recorded] iterations={} reuse_observed={} bad_output={}",
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
        "sort_recorded produced corrupted output in {}/{} iterations \
         (reuse_observed={}). Either the LSD radix passes' reads of input.column[k] \
         or the final apply_permutation_bytes gather raced an alloc-stream reuse \
         + trample.",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Slice #5 dedup: drop+reuse test for `dedup_full_row_recorded`.
/// Composes `sort_recorded` (typed sort on launch_stream) →
/// on-stream `mark_unique_full_row_bytewise` → recorded
/// compact tail. After the call returns, multiple kernel
/// chains are still in flight (gather, mark, scan, capture,
/// per-column compact). Drop input WITHOUT host sync, reuse +
/// trample, assert the deduped output is correct.
///
/// Input: 1024 rows where each `(k, v)` tuple appears
/// duplicated four times in scrambled order (i.e. 256 unique
/// tuples × 4). Expected output: 256 unique sorted
/// `(k, v)` rows.
#[test]
fn provider_dedup_full_row_recorded_survives_drop_and_reuse() {
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
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const UNIQUE: usize = 256;
    const REPS: usize = 4;
    const ROWS: usize = UNIQUE * REPS;
    // dedup_full_row_recorded allocates many intermediates
    // (sort scratch + mark_unique buffers + compact tail), so
    // freed input slots end up deeper in the pool free-list.
    // 128 iterations + a per-iter drain (allocating 4 extra
    // probe slots per iter) gives the pool enough pressure to
    // observe reuse reliably.
    const ITERATIONS: usize = 128;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        // Build duplicated tuples in scrambled order: row i has
        // unique tuple (i / REPS, i / REPS * 7) where the
        // duplication count is REPS. Rotation per iteration
        // randomizes the order.
        let mut k_data = Vec::with_capacity(ROWS * 4);
        let mut v_data = Vec::with_capacity(ROWS * 4);
        for i in 0..ROWS {
            let rot = (i + iter * 17) % ROWS;
            let u = (rot / REPS) as u32;
            k_data.extend_from_slice(&u.to_le_bytes());
            v_data.extend_from_slice(&(u.wrapping_mul(7)).to_le_bytes());
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

        let deduped = provider
            .dedup_full_row_recorded(&input, launch_stream)
            .expect("dedup_full_row_recorded");

        drop(input);

        // Reuse + trample. Allocate four probe slots per iter
        // to drain the pool's free-list and increase the
        // chance of catching a reuse of the freed input slots.
        let next_a = memory.alloc::<u8>(ROWS * 4).expect("alloc next_a");
        let next_a_ptr = next_a.device_ptr_value();
        let next_b = memory.alloc::<u8>(ROWS * 4).expect("alloc next_b");
        let next_b_ptr = next_b.device_ptr_value();
        let next_c = memory.alloc::<u8>(ROWS * 4).expect("alloc next_c");
        let next_c_ptr = next_c.device_ptr_value();
        let next_d = memory.alloc::<u8>(ROWS * 4).expect("alloc next_d");
        let next_d_ptr = next_d.device_ptr_value();
        let probe_ptrs = [next_a_ptr, next_b_ptr, next_c_ptr, next_d_ptr];
        let reused = probe_ptrs.iter().any(|p| *p == k_ptr || *p == v_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            for &p in &probe_ptrs {
                memset_sync_default(p, TRAMPLE, ROWS * 4);
            }
        }
        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Verify: UNIQUE rows kept, sorted ascending by k. Row
        // i should be (i, i*7).
        let mut host_rows = [0u32];
        device
            .inner()
            .dtoh_sync_copy_into(deduped.num_rows_device(), &mut host_rows)
            .expect("dtoh dedup count");
        let actual_rows = host_rows[0] as usize;
        if actual_rows != UNIQUE {
            bad_output += 1;
            if iter == 0 {
                eprintln!(
                    "[dedup_full_row_recorded] iter=0 actual_rows={} expected={} reused={}",
                    actual_rows, UNIQUE, reused
                );
            }
            drop(deduped);
            drop(next_a);
            drop(next_b);
            drop(next_c);
            drop(next_d);
            runtime.reap_pending().expect("reap");
            continue;
        }

        let k_out = deduped.column(0).expect("out k");
        let v_out = deduped.column(1).expect("out v");
        let mut k_back = vec![0u8; UNIQUE * 4];
        let mut v_back = vec![0u8; UNIQUE * 4];
        unsafe {
            dtoh_sync(&mut k_back, *k_out.device_ptr());
            dtoh_sync(&mut v_back, *v_out.device_ptr());
        }
        let mut local_bad = 0usize;
        for i in 0..UNIQUE {
            let kb = &k_back[i * 4..i * 4 + 4];
            let vb = &v_back[i * 4..i * 4 + 4];
            let kv = u32::from_le_bytes([kb[0], kb[1], kb[2], kb[3]]);
            let vv = u32::from_le_bytes([vb[0], vb[1], vb[2], vb[3]]);
            let k_e = i as u32;
            let v_e = k_e.wrapping_mul(7);
            if kv != k_e || vv != v_e {
                local_bad += 1;
                if iter == 0 && local_bad <= 4 {
                    eprintln!(
                        "[dedup_full_row_recorded] iter=0 row={} k={}/{} v={}/{} reused={}",
                        i, kv, k_e, vv, v_e, reused
                    );
                }
            }
        }
        if local_bad > 0 {
            bad_output += 1;
        }

        drop(deduped);
        drop(next_a);
        drop(next_b);
        drop(next_c);
        drop(next_d);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[dedup_full_row_recorded] iterations={} reuse_observed={} bad_output={}",
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
        "dedup_full_row_recorded produced corrupted output in {}/{} iterations \
         (reuse_observed={}). Either sort_recorded, mark_unique_full_row_bytewise, \
         or the recorded compact tail raced an alloc-stream reuse + trample.",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Slice #6 GroupBy: drop+reuse for the recorded multi-agg
/// chain. Composes `sort_recorded` → `pack_keys_gpu_on_stream`
/// → boundary detect → multi-block scan → capture_num_groups
/// → group-id derivation → per-aggregation kernels (Count,
/// Sum, Min, Max) → key gather/unpack — every kernel on the
/// caller-supplied `launch_stream`. Dropping `input` after
/// the call returns must NOT race the still-pending agg /
/// gather / unpack kernels.
///
/// Input: 1024 rows where row `i` has `(key = i % 64,
/// value = i)`. Expected per-key results:
///   * count = 16
///   * sum   = 16*k + 64*(0+1+...+15) = 16*k + 7680
///   * min   = k
///   * max   = k + 15*64 = k + 960
#[test]
fn provider_groupby_multi_agg_recorded_survives_drop_and_reuse() {
    use xlog_core::{AggOp, ScalarType, Schema};
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
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const KEYS: usize = 64;
    const REPS: usize = 16;
    const ROWS: usize = KEYS * REPS;
    // GroupBy allocates many fresh buffers per call (sort
    // scratch + pack + boundaries + scan + group ids + agg
    // outputs + unpacked keys). 256 iterations × 8 probe
    // slots gives enough pool pressure to observe reuse of
    // the freed input column slots.
    const ITERATIONS: usize = 256;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        let mut k_data = Vec::with_capacity(ROWS * 4);
        let mut v_data = Vec::with_capacity(ROWS * 4);
        for i in 0..ROWS {
            let k = (i % KEYS) as u32;
            let v = i as u32;
            k_data.extend_from_slice(&k.to_le_bytes());
            v_data.extend_from_slice(&v.to_le_bytes());
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

        let result = provider
            .groupby_multi_agg_recorded(
                &input,
                &[0],
                &[
                    (1, AggOp::Count),
                    (1, AggOp::Sum),
                    (1, AggOp::Min),
                    (1, AggOp::Max),
                ],
                launch_stream,
            )
            .expect("groupby_multi_agg_recorded");

        // Drop input WITHOUT host sync. The agg / gather /
        // unpack kernels are still in flight on launch_stream.
        drop(input);

        // Reuse + trample. Allocate eight probe slots per
        // iter to drain the pool free-list aggressively.
        let mut probes: Vec<_> = (0..8)
            .map(|_| memory.alloc::<u8>(ROWS * 4).expect("alloc probe"))
            .collect();
        let probe_ptrs: Vec<u64> = probes.iter().map(|p| p.device_ptr_value()).collect();
        let reused = probe_ptrs.iter().any(|p| *p == k_ptr || *p == v_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            for &p in &probe_ptrs {
                memset_sync_default(p, TRAMPLE, ROWS * 4);
            }
        }
        let _ = &mut probes;
        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Verify: 64 groups, each with count=16, sum=16k+7680,
        // min=k, max=k+960.
        let mut host_rows = [0u32];
        device
            .inner()
            .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
            .expect("dtoh group count");
        if host_rows[0] as usize != KEYS {
            bad_output += 1;
            if iter == 0 {
                eprintln!(
                    "[groupby_recorded] iter=0 group_count={} expected={} reused={}",
                    host_rows[0], KEYS, reused
                );
            }
            drop(result);
            drop(probes);
            runtime.reap_pending().expect("reap");
            continue;
        }
        // Read back: column 0 = key (u32), col 1 = count (u64),
        // col 2 = sum (u64), col 3 = min (u32), col 4 = max (u32).
        let k_out = result.column(0).expect("out k");
        let count_out = result.column(1).expect("out count");
        let sum_out = result.column(2).expect("out sum");
        let min_out = result.column(3).expect("out min");
        let max_out = result.column(4).expect("out max");
        let mut k_back = vec![0u8; KEYS * 4];
        let mut count_back = vec![0u8; KEYS * 8];
        let mut sum_back = vec![0u8; KEYS * 8];
        let mut min_back = vec![0u8; KEYS * 4];
        let mut max_back = vec![0u8; KEYS * 4];
        unsafe {
            dtoh_sync(&mut k_back, *k_out.device_ptr());
            dtoh_sync(&mut count_back, *count_out.device_ptr());
            dtoh_sync(&mut sum_back, *sum_out.device_ptr());
            dtoh_sync(&mut min_back, *min_out.device_ptr());
            dtoh_sync(&mut max_back, *max_out.device_ptr());
        }
        let mut local_bad = 0usize;
        for grp in 0..KEYS {
            let kb = &k_back[grp * 4..grp * 4 + 4];
            let key = u32::from_le_bytes([kb[0], kb[1], kb[2], kb[3]]);
            let k_e = grp as u32;
            let cb = &count_back[grp * 8..grp * 8 + 8];
            let count =
                u64::from_le_bytes([cb[0], cb[1], cb[2], cb[3], cb[4], cb[5], cb[6], cb[7]]);
            let sb = &sum_back[grp * 8..grp * 8 + 8];
            let sum = u64::from_le_bytes([sb[0], sb[1], sb[2], sb[3], sb[4], sb[5], sb[6], sb[7]]);
            let mb = &min_back[grp * 4..grp * 4 + 4];
            let min_v = u32::from_le_bytes([mb[0], mb[1], mb[2], mb[3]]);
            let xb = &max_back[grp * 4..grp * 4 + 4];
            let max_v = u32::from_le_bytes([xb[0], xb[1], xb[2], xb[3]]);
            let count_e: u64 = REPS as u64;
            let sum_e: u64 = (REPS as u64) * (k_e as u64) + 64 * (0..REPS as u64).sum::<u64>();
            let min_e = k_e;
            let max_e = k_e + ((REPS - 1) as u32) * (KEYS as u32);
            if key != k_e || count != count_e || sum != sum_e || min_v != min_e || max_v != max_e {
                local_bad += 1;
                if iter == 0 && local_bad <= 4 {
                    eprintln!(
                        "[groupby_recorded] iter=0 grp={} k={}/{} count={}/{} sum={}/{} \
                         min={}/{} max={}/{} reused={}",
                        grp,
                        key,
                        k_e,
                        count,
                        count_e,
                        sum,
                        sum_e,
                        min_v,
                        min_e,
                        max_v,
                        max_e,
                        reused
                    );
                }
            }
        }
        if local_bad > 0 {
            bad_output += 1;
        }

        drop(result);
        drop(probes);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[groupby_recorded] iterations={} reuse_observed={} bad_output={}",
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
        "groupby_multi_agg_recorded produced corrupted output in {}/{} iterations \
         (reuse_observed={}). One of the chain steps (sort, pack, boundary, scan, \
         capture, group-id derivation, per-agg kernel, gather, unpack) raced an \
         alloc-stream reuse + trample.",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Negative test: recorded GroupBy against a no-runtime manager.
#[test]
fn provider_groupby_multi_agg_recorded_rejects_legacy_manager() {
    use xlog_core::{AggOp, ScalarType, Schema};
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
    let mut k = memory.alloc::<u8>(16).expect("alloc k");
    let mut v = memory.alloc::<u8>(16).expect("alloc v");
    let kv = [0u32, 1, 0, 1];
    let kv_b: Vec<u8> = kv.iter().flat_map(|x| x.to_le_bytes()).collect();
    device
        .inner()
        .htod_sync_copy_into(&kv_b, &mut k)
        .expect("htod k");
    device
        .inner()
        .htod_sync_copy_into(&kv_b, &mut v)
        .expect("htod v");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut d_num_rows)
        .expect("htod rows");
    let input = CudaBuffer::from_columns(
        vec![k.into(), v.into()],
        4,
        d_num_rows,
        Schema::new(vec![
            ("k".to_string(), ScalarType::U32),
            ("v".to_string(), ScalarType::U32),
        ]),
    );
    let err =
        provider.groupby_multi_agg_recorded(&input, &[0], &[(1, AggOp::Count)], StreamId::DEFAULT);
    match err {
        Err(XlogError::Kernel(msg)) => assert!(
            msg.contains("requires") || msg.contains("with_runtime"),
            "expected helpful Kernel error, got {:?}",
            msg
        ),
        Err(other) => panic!(
            "groupby_multi_agg_recorded must reject legacy manager with XlogError::Kernel, \
             got {:?}",
            other
        ),
        Ok(_) => panic!(
            "groupby_multi_agg_recorded must reject legacy manager — unexpectedly returned Ok"
        ),
    }
}

/// Slice #7A: drop+reuse for the recorded inner hash join.
/// Composes `pack_keys_gpu_on_stream` (slice #6) →
/// `build_hash_table_v2_on_stream` (this slice) → probe count
/// pass → `cu_stream.synchronize()` → host scalar count read
/// → probe materialize pass → sync + count read → per-side
/// `gather_buffer_by_indices_on_stream` (this slice). Every
/// kernel runs on the explicit `launch_stream`; host scalar
/// reads are explicitly ordered against it.
///
/// Predicate: equi-join on column 0. Left has rows
/// `(k, l_value)` with `k = i % LKEYS`. Right has rows
/// `(k, r_value)` with `k = i % RKEYS`. `LKEYS=64`,
/// `RKEYS=32` so every left key matches multiple right rows
/// → cross product per matching key.
#[test]
fn provider_hash_join_inner_v2_recorded_survives_drop_and_reuse() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CudaBuffer, JoinType};

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
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire launch_stream");
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const LROWS: usize = 256;
    const RROWS: usize = 256;
    const LKEYS: u32 = 64;
    const RKEYS: u32 = 32;
    // Hash join allocates ~20+ fresh buffers per call (pack
    // ×6, hash table ×4, probe outputs ×3, gather outputs
    // ×4-per-side). 512 iterations × 16 probe slots is the
    // pressure needed to push the pool's free-list past all
    // those slots and reuse the freed input columns.
    const ITERATIONS: usize = 512;
    const PROBE_SLOTS: usize = 16;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        // Build left: row i has (k = i % LKEYS, v = i + 100_000).
        let mut lk = Vec::with_capacity(LROWS * 4);
        let mut lv = Vec::with_capacity(LROWS * 4);
        for i in 0..LROWS {
            let k = (i as u32) % LKEYS;
            let v = (i as u32) + 100_000;
            lk.extend_from_slice(&k.to_le_bytes());
            lv.extend_from_slice(&v.to_le_bytes());
        }
        let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
        let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
        device
            .inner()
            .htod_sync_copy_into(&lk, &mut lk_b)
            .expect("htod lk");
        device
            .inner()
            .htod_sync_copy_into(&lv, &mut lv_b)
            .expect("htod lv");
        let lk_ptr = lk_b.device_ptr_value();
        let lv_ptr = lv_b.device_ptr_value();
        let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
        device
            .inner()
            .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
            .expect("htod l_rows");
        let left = CudaBuffer::from_columns(
            vec![lk_b.into(), lv_b.into()],
            LROWS as u64,
            l_rows,
            schema.clone(),
        );

        // Build right: row j has (k = j % RKEYS, v = j + 200_000).
        let mut rk = Vec::with_capacity(RROWS * 4);
        let mut rv = Vec::with_capacity(RROWS * 4);
        for j in 0..RROWS {
            let k = (j as u32) % RKEYS;
            let v = (j as u32) + 200_000;
            rk.extend_from_slice(&k.to_le_bytes());
            rv.extend_from_slice(&v.to_le_bytes());
        }
        let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
        let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
        device
            .inner()
            .htod_sync_copy_into(&rk, &mut rk_b)
            .expect("htod rk");
        device
            .inner()
            .htod_sync_copy_into(&rv, &mut rv_b)
            .expect("htod rv");
        let rk_ptr = rk_b.device_ptr_value();
        let rv_ptr = rv_b.device_ptr_value();
        let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
        device
            .inner()
            .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
            .expect("htod r_rows");
        let right = CudaBuffer::from_columns(
            vec![rk_b.into(), rv_b.into()],
            RROWS as u64,
            r_rows,
            schema.clone(),
        );

        // Recorded inner join.
        let result = provider
            .hash_join_v2_recorded(
                &left,
                &right,
                &[0],
                &[0],
                JoinType::Inner,
                None,
                launch_stream,
            )
            .expect("hash_join_v2_recorded");

        // Drop both inputs WITHOUT host sync.
        drop(left);
        drop(right);

        // Reuse + trample.
        let mut probes: Vec<_> = (0..PROBE_SLOTS)
            .map(|_| memory.alloc::<u8>(LROWS * 4).expect("alloc probe"))
            .collect();
        let probe_ptrs: Vec<u64> = probes.iter().map(|p| p.device_ptr_value()).collect();
        let reused = probe_ptrs
            .iter()
            .any(|p| *p == lk_ptr || *p == lv_ptr || *p == rk_ptr || *p == rv_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            for &p in &probe_ptrs {
                memset_sync_default(p, TRAMPLE, LROWS * 4);
            }
        }
        let _ = &mut probes;

        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Verify result. Inner join semantics:
        //   * For every (i, j) where i % LKEYS == j % RKEYS:
        //     output row = (k, l_value, k, r_value).
        //   * RKEYS divides LKEYS (32 | 64), so each left
        //     row matches LROWS/LKEYS * RKEYS/RKEYS_match
        //     ... easier: build expected set host-side.
        let mut expected: Vec<(u32, u32, u32)> = Vec::new(); // (k, lv, rv)
        for i in 0..LROWS {
            let lk_e = (i as u32) % LKEYS;
            let lv_e = (i as u32) + 100_000;
            for j in 0..RROWS {
                let rk_e = (j as u32) % RKEYS;
                if lk_e == rk_e {
                    let rv_e = (j as u32) + 200_000;
                    expected.push((lk_e, lv_e, rv_e));
                }
            }
        }
        let expected_count = expected.len();
        let mut host_rows = [0u32];
        device
            .inner()
            .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
            .expect("dtoh result count");
        if host_rows[0] as usize != expected_count {
            bad_output += 1;
            if iter == 0 {
                eprintln!(
                    "[hash_join_inner_recorded] iter=0 actual={} expected={} reused={}",
                    host_rows[0], expected_count, reused
                );
            }
            drop(result);
            drop(probes);
            runtime.reap_pending().expect("reap");
            continue;
        }

        // Read back all four output columns.
        let mut col0 = vec![0u8; expected_count * 4];
        let mut col1 = vec![0u8; expected_count * 4];
        let mut col2 = vec![0u8; expected_count * 4];
        let mut col3 = vec![0u8; expected_count * 4];
        unsafe {
            dtoh_sync(&mut col0, *result.column(0).expect("col0").device_ptr());
            dtoh_sync(&mut col1, *result.column(1).expect("col1").device_ptr());
            dtoh_sync(&mut col2, *result.column(2).expect("col2").device_ptr());
            dtoh_sync(&mut col3, *result.column(3).expect("col3").device_ptr());
        }

        // Build a set from observed rows + a set from expected.
        let mut observed: std::collections::HashSet<(u32, u32, u32, u32)> =
            std::collections::HashSet::new();
        for i in 0..expected_count {
            let c0 = u32::from_le_bytes([
                col0[i * 4],
                col0[i * 4 + 1],
                col0[i * 4 + 2],
                col0[i * 4 + 3],
            ]);
            let c1 = u32::from_le_bytes([
                col1[i * 4],
                col1[i * 4 + 1],
                col1[i * 4 + 2],
                col1[i * 4 + 3],
            ]);
            let c2 = u32::from_le_bytes([
                col2[i * 4],
                col2[i * 4 + 1],
                col2[i * 4 + 2],
                col2[i * 4 + 3],
            ]);
            let c3 = u32::from_le_bytes([
                col3[i * 4],
                col3[i * 4 + 1],
                col3[i * 4 + 2],
                col3[i * 4 + 3],
            ]);
            observed.insert((c0, c1, c2, c3));
        }
        let expected_set: std::collections::HashSet<(u32, u32, u32, u32)> = expected
            .iter()
            .map(|(k, lv, rv)| (*k, *lv, *k, *rv))
            .collect();
        if observed != expected_set {
            bad_output += 1;
            if iter == 0 {
                let missing = expected_set.difference(&observed).count();
                let extra = observed.difference(&expected_set).count();
                eprintln!(
                    "[hash_join_inner_recorded] iter=0 set mismatch missing={} extra={} reused={}",
                    missing, extra, reused
                );
            }
        }

        drop(result);
        drop(probes);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[hash_join_inner_recorded] iterations={} reuse_observed={} bad_output={}",
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
        "hash_join_inner_v2_recorded produced corrupted output in {}/{} iterations \
         (reuse_observed={}). One of the chain steps (pack, hash table, probe, \
         gather) raced an alloc-stream reuse + trample.",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Negative test: recorded inner hash join against no-runtime
/// manager. Must reject before any allocation / kernel.
#[test]
fn provider_hash_join_v2_recorded_rejects_legacy_manager() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CudaBuffer, JoinType};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let memory = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(1024 * 1024),
    ));
    let provider =
        CudaKernelProvider::new(Arc::clone(&device), Arc::clone(&memory)).expect("legacy provider");
    let mut k = memory.alloc::<u8>(16).expect("alloc k");
    let mut v = memory.alloc::<u8>(16).expect("alloc v");
    let kv = [0u32, 1, 0, 1];
    let kvb: Vec<u8> = kv.iter().flat_map(|x| x.to_le_bytes()).collect();
    device
        .inner()
        .htod_sync_copy_into(&kvb, &mut k)
        .expect("htod k");
    device
        .inner()
        .htod_sync_copy_into(&kvb, &mut v)
        .expect("htod v");
    let mut rows = memory.alloc::<u32>(1).expect("alloc rows");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut rows)
        .expect("htod rows");
    let lhs = CudaBuffer::from_columns(
        vec![k.into(), v.into()],
        4,
        rows,
        Schema::new(vec![
            ("k".to_string(), ScalarType::U32),
            ("v".to_string(), ScalarType::U32),
        ]),
    );
    let mut k2 = memory.alloc::<u8>(16).expect("alloc k2");
    let mut v2 = memory.alloc::<u8>(16).expect("alloc v2");
    device
        .inner()
        .htod_sync_copy_into(&kvb, &mut k2)
        .expect("htod k2");
    device
        .inner()
        .htod_sync_copy_into(&kvb, &mut v2)
        .expect("htod v2");
    let mut rows2 = memory.alloc::<u32>(1).expect("alloc rows2");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut rows2)
        .expect("htod rows2");
    let rhs = CudaBuffer::from_columns(
        vec![k2.into(), v2.into()],
        4,
        rows2,
        Schema::new(vec![
            ("k".to_string(), ScalarType::U32),
            ("v".to_string(), ScalarType::U32),
        ]),
    );
    let err = provider.hash_join_v2_recorded(
        &lhs,
        &rhs,
        &[0],
        &[0],
        JoinType::Inner,
        None,
        StreamId::DEFAULT,
    );
    match err {
        Err(XlogError::Kernel(msg)) => assert!(
            msg.contains("requires") || msg.contains("with_runtime"),
            "expected helpful Kernel error, got {:?}",
            msg
        ),
        Err(other) => panic!(
            "hash_join_v2_recorded must reject legacy manager with XlogError::Kernel, got {:?}",
            other
        ),
        Ok(_) => {
            panic!("hash_join_v2_recorded must reject legacy manager — unexpectedly returned Ok")
        }
    }
}

/// Slice-boundary lock: `hash_join_v2_recorded` now accepts
/// Inner / Semi / Anti / LeftOuter (slices #7A / #7B / #7C).
/// The remaining deferred surface is the indexed variant
/// (slice #7D), which goes through `hash_join_v2_with_index`,
/// not through `hash_join_v2_recorded`. Asserts every
/// currently-supported join type returns Ok against a
/// minimal runtime-backed setup.
#[test]
fn provider_hash_join_v2_recorded_accepts_all_join_types() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CudaBuffer, JoinType};

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
        Box::new(GlobalDeviceBudget::new(logging, 4 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(4 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire launch_stream");

    let mut k = memory.alloc::<u8>(16).expect("alloc k");
    let kv = [0u32, 1, 0, 1];
    let kvb: Vec<u8> = kv.iter().flat_map(|x| x.to_le_bytes()).collect();
    device
        .inner()
        .htod_sync_copy_into(&kvb, &mut k)
        .expect("htod k");
    let mut rows = memory.alloc::<u32>(1).expect("alloc rows");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut rows)
        .expect("htod rows");
    let lhs = CudaBuffer::from_columns(
        vec![k.into()],
        4,
        rows,
        Schema::new(vec![("k".to_string(), ScalarType::U32)]),
    );
    let mut k2 = memory.alloc::<u8>(16).expect("alloc k2");
    device
        .inner()
        .htod_sync_copy_into(&kvb, &mut k2)
        .expect("htod k2");
    let mut rows2 = memory.alloc::<u32>(1).expect("alloc rows2");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut rows2)
        .expect("htod rows2");
    let rhs = CudaBuffer::from_columns(
        vec![k2.into()],
        4,
        rows2,
        Schema::new(vec![("k".to_string(), ScalarType::U32)]),
    );

    for jt in [
        JoinType::Inner,
        JoinType::Semi,
        JoinType::Anti,
        JoinType::LeftOuter,
    ] {
        let r = provider.hash_join_v2_recorded(&lhs, &rhs, &[0], &[0], jt, None, launch_stream);
        assert!(
            r.is_ok(),
            "hash_join_v2_recorded must accept {:?}, got error",
            jt
        );
    }
}

/// Slice #7B: drop+reuse for the recorded Semi hash join.
/// Semi keeps left rows whose key has a match in right.
/// Composes pack_keys_on_stream ×2 → build_hash_table_on_stream
/// → HASH_JOIN_SEMI kernel → recorded compact tail. Drops both
/// inputs after the call returns.
///
/// Predicate: left rows have keys 0..LKEYS, right rows have
/// keys 0..RKEYS where RKEYS=LKEYS/2. Expected: only left rows
/// whose key is in [0, RKEYS) survive — half of left.
#[test]
fn provider_hash_join_semi_v2_recorded_survives_drop_and_reuse() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CudaBuffer, JoinType};

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
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire launch_stream");
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const LROWS: usize = 256;
    const RROWS: usize = 256;
    const LKEYS: u32 = 64;
    const RKEYS: u32 = 32;
    // Pool pressure tuned to the same shape as inner-join +
    // recorded compact tail combined.
    const ITERATIONS: usize = 512;
    const PROBE_SLOTS: usize = 16;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        // Left: row i has (k = i % LKEYS, v = i + 100_000).
        let mut lk = Vec::with_capacity(LROWS * 4);
        let mut lv = Vec::with_capacity(LROWS * 4);
        for i in 0..LROWS {
            lk.extend_from_slice(&((i as u32) % LKEYS).to_le_bytes());
            lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
        }
        let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
        let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
        device
            .inner()
            .htod_sync_copy_into(&lk, &mut lk_b)
            .expect("htod lk");
        device
            .inner()
            .htod_sync_copy_into(&lv, &mut lv_b)
            .expect("htod lv");
        let lk_ptr = lk_b.device_ptr_value();
        let lv_ptr = lv_b.device_ptr_value();
        let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
        device
            .inner()
            .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
            .expect("htod l_rows");
        let left = CudaBuffer::from_columns(
            vec![lk_b.into(), lv_b.into()],
            LROWS as u64,
            l_rows,
            schema.clone(),
        );

        // Right: row j has (k = j % RKEYS, v = j + 200_000).
        let mut rk = Vec::with_capacity(RROWS * 4);
        let mut rv = Vec::with_capacity(RROWS * 4);
        for j in 0..RROWS {
            rk.extend_from_slice(&((j as u32) % RKEYS).to_le_bytes());
            rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
        }
        let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
        let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
        device
            .inner()
            .htod_sync_copy_into(&rk, &mut rk_b)
            .expect("htod rk");
        device
            .inner()
            .htod_sync_copy_into(&rv, &mut rv_b)
            .expect("htod rv");
        let rk_ptr = rk_b.device_ptr_value();
        let rv_ptr = rv_b.device_ptr_value();
        let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
        device
            .inner()
            .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
            .expect("htod r_rows");
        let right = CudaBuffer::from_columns(
            vec![rk_b.into(), rv_b.into()],
            RROWS as u64,
            r_rows,
            schema.clone(),
        );

        let result = provider
            .hash_join_v2_recorded(
                &left,
                &right,
                &[0],
                &[0],
                JoinType::Semi,
                None,
                launch_stream,
            )
            .expect("hash_join_v2_recorded::<Semi>");

        // Drop both inputs WITHOUT host sync.
        drop(left);
        drop(right);

        // Reuse + trample.
        let mut probes: Vec<_> = (0..PROBE_SLOTS)
            .map(|_| memory.alloc::<u8>(LROWS * 4).expect("alloc probe"))
            .collect();
        let probe_ptrs: Vec<u64> = probes.iter().map(|p| p.device_ptr_value()).collect();
        let reused = probe_ptrs
            .iter()
            .any(|p| *p == lk_ptr || *p == lv_ptr || *p == rk_ptr || *p == rv_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            for &p in &probe_ptrs {
                memset_sync_default(p, TRAMPLE, LROWS * 4);
            }
        }
        let _ = &mut probes;

        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Verify: kept rows are exactly i where i % LKEYS < RKEYS.
        let expected: Vec<(u32, u32)> = (0..LROWS)
            .filter_map(|i| {
                let k = (i as u32) % LKEYS;
                if k < RKEYS {
                    Some((k, (i as u32) + 100_000))
                } else {
                    None
                }
            })
            .collect();
        let expected_count = expected.len();
        let mut host_rows = [0u32];
        device
            .inner()
            .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
            .expect("dtoh result count");
        if host_rows[0] as usize != expected_count {
            bad_output += 1;
            if iter == 0 {
                eprintln!(
                    "[hash_join_semi_recorded] iter=0 actual={} expected={} reused={}",
                    host_rows[0], expected_count, reused
                );
            }
            drop(result);
            drop(probes);
            runtime.reap_pending().expect("reap");
            continue;
        }
        let mut k_back = vec![0u8; expected_count * 4];
        let mut v_back = vec![0u8; expected_count * 4];
        unsafe {
            dtoh_sync(&mut k_back, *result.column(0).expect("col0").device_ptr());
            dtoh_sync(&mut v_back, *result.column(1).expect("col1").device_ptr());
        }
        // Compare as sets (compact preserves order, so we
        // could compare positionally; sets are stricter and
        // catch any reorder regression).
        let observed: std::collections::HashSet<(u32, u32)> = (0..expected_count)
            .map(|i| {
                let k = u32::from_le_bytes([
                    k_back[i * 4],
                    k_back[i * 4 + 1],
                    k_back[i * 4 + 2],
                    k_back[i * 4 + 3],
                ]);
                let v = u32::from_le_bytes([
                    v_back[i * 4],
                    v_back[i * 4 + 1],
                    v_back[i * 4 + 2],
                    v_back[i * 4 + 3],
                ]);
                (k, v)
            })
            .collect();
        let expected_set: std::collections::HashSet<(u32, u32)> =
            expected.iter().copied().collect();
        if observed != expected_set {
            bad_output += 1;
            if iter == 0 {
                eprintln!(
                    "[hash_join_semi_recorded] iter=0 set mismatch reused={}",
                    reused
                );
            }
        }

        drop(result);
        drop(probes);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[hash_join_semi_recorded] iterations={} reuse_observed={} bad_output={}",
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
        "hash_join_v2_recorded::<Semi> produced corrupted output in {}/{} iterations \
         (reuse_observed={}). Pack / hash table / semi probe / compact tail raced \
         an alloc-stream reuse + trample.",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Slice #7B: drop+reuse for the recorded Anti hash join.
/// Anti keeps left rows whose key has NO match in right —
/// the complement of Semi. Same chain shape as Semi; the
/// kernel selection swaps `HASH_JOIN_SEMI` for `HASH_JOIN_ANTI`
/// inside `hash_join_semi_or_anti_v2_recorded`. Expected:
/// only left rows with key ≥ RKEYS survive.
#[test]
fn provider_hash_join_anti_v2_recorded_survives_drop_and_reuse() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CudaBuffer, JoinType};

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
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire launch_stream");
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const LROWS: usize = 256;
    const RROWS: usize = 256;
    const LKEYS: u32 = 64;
    const RKEYS: u32 = 32;
    const ITERATIONS: usize = 512;
    const PROBE_SLOTS: usize = 16;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        let mut lk = Vec::with_capacity(LROWS * 4);
        let mut lv = Vec::with_capacity(LROWS * 4);
        for i in 0..LROWS {
            lk.extend_from_slice(&((i as u32) % LKEYS).to_le_bytes());
            lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
        }
        let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
        let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
        device
            .inner()
            .htod_sync_copy_into(&lk, &mut lk_b)
            .expect("htod lk");
        device
            .inner()
            .htod_sync_copy_into(&lv, &mut lv_b)
            .expect("htod lv");
        let lk_ptr = lk_b.device_ptr_value();
        let lv_ptr = lv_b.device_ptr_value();
        let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
        device
            .inner()
            .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
            .expect("htod l_rows");
        let left = CudaBuffer::from_columns(
            vec![lk_b.into(), lv_b.into()],
            LROWS as u64,
            l_rows,
            schema.clone(),
        );

        let mut rk = Vec::with_capacity(RROWS * 4);
        let mut rv = Vec::with_capacity(RROWS * 4);
        for j in 0..RROWS {
            rk.extend_from_slice(&((j as u32) % RKEYS).to_le_bytes());
            rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
        }
        let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
        let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
        device
            .inner()
            .htod_sync_copy_into(&rk, &mut rk_b)
            .expect("htod rk");
        device
            .inner()
            .htod_sync_copy_into(&rv, &mut rv_b)
            .expect("htod rv");
        let rk_ptr = rk_b.device_ptr_value();
        let rv_ptr = rv_b.device_ptr_value();
        let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
        device
            .inner()
            .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
            .expect("htod r_rows");
        let right = CudaBuffer::from_columns(
            vec![rk_b.into(), rv_b.into()],
            RROWS as u64,
            r_rows,
            schema.clone(),
        );

        let result = provider
            .hash_join_v2_recorded(
                &left,
                &right,
                &[0],
                &[0],
                JoinType::Anti,
                None,
                launch_stream,
            )
            .expect("hash_join_v2_recorded::<Anti>");

        drop(left);
        drop(right);

        let mut probes: Vec<_> = (0..PROBE_SLOTS)
            .map(|_| memory.alloc::<u8>(LROWS * 4).expect("alloc probe"))
            .collect();
        let probe_ptrs: Vec<u64> = probes.iter().map(|p| p.device_ptr_value()).collect();
        let reused = probe_ptrs
            .iter()
            .any(|p| *p == lk_ptr || *p == lv_ptr || *p == rk_ptr || *p == rv_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            for &p in &probe_ptrs {
                memset_sync_default(p, TRAMPLE, LROWS * 4);
            }
        }
        let _ = &mut probes;

        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Verify: kept rows are i where i%LKEYS >= RKEYS.
        let expected: Vec<(u32, u32)> = (0..LROWS)
            .filter_map(|i| {
                let k = (i as u32) % LKEYS;
                if k >= RKEYS {
                    Some((k, (i as u32) + 100_000))
                } else {
                    None
                }
            })
            .collect();
        let expected_count = expected.len();
        let mut host_rows = [0u32];
        device
            .inner()
            .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
            .expect("dtoh result count");
        if host_rows[0] as usize != expected_count {
            bad_output += 1;
            if iter == 0 {
                eprintln!(
                    "[hash_join_anti_recorded] iter=0 actual={} expected={} reused={}",
                    host_rows[0], expected_count, reused
                );
            }
            drop(result);
            drop(probes);
            runtime.reap_pending().expect("reap");
            continue;
        }
        let mut k_back = vec![0u8; expected_count * 4];
        let mut v_back = vec![0u8; expected_count * 4];
        unsafe {
            dtoh_sync(&mut k_back, *result.column(0).expect("col0").device_ptr());
            dtoh_sync(&mut v_back, *result.column(1).expect("col1").device_ptr());
        }
        let observed: std::collections::HashSet<(u32, u32)> = (0..expected_count)
            .map(|i| {
                let k = u32::from_le_bytes([
                    k_back[i * 4],
                    k_back[i * 4 + 1],
                    k_back[i * 4 + 2],
                    k_back[i * 4 + 3],
                ]);
                let v = u32::from_le_bytes([
                    v_back[i * 4],
                    v_back[i * 4 + 1],
                    v_back[i * 4 + 2],
                    v_back[i * 4 + 3],
                ]);
                (k, v)
            })
            .collect();
        let expected_set: std::collections::HashSet<(u32, u32)> =
            expected.iter().copied().collect();
        if observed != expected_set {
            bad_output += 1;
            if iter == 0 {
                eprintln!(
                    "[hash_join_anti_recorded] iter=0 set mismatch reused={}",
                    reused
                );
            }
        }

        drop(result);
        drop(probes);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[hash_join_anti_recorded] iterations={} reuse_observed={} bad_output={}",
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
        "hash_join_v2_recorded::<Anti> produced corrupted output in {}/{} iterations \
         (reuse_observed={}). Pack / hash table / anti probe / compact tail raced \
         an alloc-stream reuse + trample.",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Slice #7C: drop+reuse for the recorded LeftOuter hash
/// join — partial-match shape (some left rows match, some
/// don't). Composes pack ×2 → table → SEMI mask → PROBE
/// count + materialize → mask_not → recorded compact tail
/// → 2× gather → per-column dtod-async concat with zero
/// fills for the right side. Drops both inputs WITHOUT host
/// sync.
///
/// Predicate: left has keys 0..LKEYS, right has keys 0..RKEYS
/// where RKEYS=LKEYS/2. Matched rows: i % LKEYS < RKEYS.
/// Expected output: matched cross-product (in the inner
/// region) + unmatched left rows with right columns
/// zero-filled.
#[test]
fn provider_hash_join_left_outer_v2_recorded_partial_match_survives_drop_and_reuse() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CudaBuffer, JoinType};

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
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire launch_stream");
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const LROWS: usize = 256;
    const RROWS: usize = 256;
    const LKEYS: u32 = 64;
    const RKEYS: u32 = 32;
    const ITERATIONS: usize = 512;
    const PROBE_SLOTS: usize = 16;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        let mut lk = Vec::with_capacity(LROWS * 4);
        let mut lv = Vec::with_capacity(LROWS * 4);
        for i in 0..LROWS {
            lk.extend_from_slice(&((i as u32) % LKEYS).to_le_bytes());
            lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
        }
        let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
        let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
        device
            .inner()
            .htod_sync_copy_into(&lk, &mut lk_b)
            .expect("htod lk");
        device
            .inner()
            .htod_sync_copy_into(&lv, &mut lv_b)
            .expect("htod lv");
        let lk_ptr = lk_b.device_ptr_value();
        let lv_ptr = lv_b.device_ptr_value();
        let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
        device
            .inner()
            .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
            .expect("htod l_rows");
        let left = CudaBuffer::from_columns(
            vec![lk_b.into(), lv_b.into()],
            LROWS as u64,
            l_rows,
            schema.clone(),
        );

        let mut rk = Vec::with_capacity(RROWS * 4);
        let mut rv = Vec::with_capacity(RROWS * 4);
        for j in 0..RROWS {
            rk.extend_from_slice(&((j as u32) % RKEYS).to_le_bytes());
            rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
        }
        let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
        let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
        device
            .inner()
            .htod_sync_copy_into(&rk, &mut rk_b)
            .expect("htod rk");
        device
            .inner()
            .htod_sync_copy_into(&rv, &mut rv_b)
            .expect("htod rv");
        let rk_ptr = rk_b.device_ptr_value();
        let rv_ptr = rv_b.device_ptr_value();
        let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
        device
            .inner()
            .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
            .expect("htod r_rows");
        let right = CudaBuffer::from_columns(
            vec![rk_b.into(), rv_b.into()],
            RROWS as u64,
            r_rows,
            schema.clone(),
        );

        let result = provider
            .hash_join_v2_recorded(
                &left,
                &right,
                &[0],
                &[0],
                JoinType::LeftOuter,
                None,
                launch_stream,
            )
            .expect("hash_join_v2_recorded::<LeftOuter>");

        drop(left);
        drop(right);

        let mut probes: Vec<_> = (0..PROBE_SLOTS)
            .map(|_| memory.alloc::<u8>(LROWS * 4).expect("alloc probe"))
            .collect();
        let probe_ptrs: Vec<u64> = probes.iter().map(|p| p.device_ptr_value()).collect();
        let reused = probe_ptrs
            .iter()
            .any(|p| *p == lk_ptr || *p == lv_ptr || *p == rk_ptr || *p == rv_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            for &p in &probe_ptrs {
                memset_sync_default(p, TRAMPLE, LROWS * 4);
            }
        }
        let _ = &mut probes;

        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Build expected as a multiset of (k, lv, k_or_0, rv_or_0):
        // - matched rows: for each (i, j) with i%LKEYS == j%RKEYS,
        //   row = (i%LKEYS, i+100_000, i%LKEYS, j+200_000).
        // - unmatched rows: for each i with i%LKEYS >= RKEYS,
        //   row = (i%LKEYS, i+100_000, 0, 0).
        let mut expected: std::collections::HashMap<(u32, u32, u32, u32), usize> =
            std::collections::HashMap::new();
        for i in 0..LROWS {
            let lk_e = (i as u32) % LKEYS;
            let lv_e = (i as u32) + 100_000;
            if lk_e < RKEYS {
                for j in 0..RROWS {
                    let rk_e = (j as u32) % RKEYS;
                    if lk_e == rk_e {
                        let rv_e = (j as u32) + 200_000;
                        *expected.entry((lk_e, lv_e, lk_e, rv_e)).or_insert(0) += 1;
                    }
                }
            } else {
                *expected.entry((lk_e, lv_e, 0, 0)).or_insert(0) += 1;
            }
        }
        let expected_count: usize = expected.values().sum();

        let mut host_rows = [0u32];
        device
            .inner()
            .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
            .expect("dtoh result count");
        if host_rows[0] as usize != expected_count {
            bad_output += 1;
            if iter == 0 {
                eprintln!(
                    "[left_outer_recorded] iter=0 actual={} expected={} reused={}",
                    host_rows[0], expected_count, reused
                );
            }
            drop(result);
            drop(probes);
            runtime.reap_pending().expect("reap");
            continue;
        }
        let mut col0 = vec![0u8; expected_count * 4];
        let mut col1 = vec![0u8; expected_count * 4];
        let mut col2 = vec![0u8; expected_count * 4];
        let mut col3 = vec![0u8; expected_count * 4];
        unsafe {
            dtoh_sync(&mut col0, *result.column(0).expect("c0").device_ptr());
            dtoh_sync(&mut col1, *result.column(1).expect("c1").device_ptr());
            dtoh_sync(&mut col2, *result.column(2).expect("c2").device_ptr());
            dtoh_sync(&mut col3, *result.column(3).expect("c3").device_ptr());
        }
        let mut observed: std::collections::HashMap<(u32, u32, u32, u32), usize> =
            std::collections::HashMap::new();
        for i in 0..expected_count {
            let r0 = u32::from_le_bytes([
                col0[i * 4],
                col0[i * 4 + 1],
                col0[i * 4 + 2],
                col0[i * 4 + 3],
            ]);
            let r1 = u32::from_le_bytes([
                col1[i * 4],
                col1[i * 4 + 1],
                col1[i * 4 + 2],
                col1[i * 4 + 3],
            ]);
            let r2 = u32::from_le_bytes([
                col2[i * 4],
                col2[i * 4 + 1],
                col2[i * 4 + 2],
                col2[i * 4 + 3],
            ]);
            let r3 = u32::from_le_bytes([
                col3[i * 4],
                col3[i * 4 + 1],
                col3[i * 4 + 2],
                col3[i * 4 + 3],
            ]);
            *observed.entry((r0, r1, r2, r3)).or_insert(0) += 1;
        }
        if observed != expected {
            bad_output += 1;
            if iter == 0 {
                eprintln!(
                    "[left_outer_recorded] iter=0 multiset mismatch reused={}",
                    reused
                );
            }
        }

        drop(result);
        drop(probes);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[left_outer_recorded] iterations={} reuse_observed={} bad_output={}",
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
        "hash_join_v2_recorded::<LeftOuter> produced corrupted output in {}/{} iterations \
         (reuse_observed={}). Pack / hash table / SEMI / PROBE / mask_not / compact / \
         gather / per-column dtod-concat raced an alloc-stream reuse + trample.",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Slice #7C: all-unmatched LeftOuter — disjoint key spaces.
/// inner_count = 0; every left row should appear with right
/// columns zero-filled. Exercises the inner_count == 0 path
/// (per-right-column zero-fill only, no inner copy).
#[test]
fn provider_hash_join_left_outer_v2_recorded_all_unmatched_survives_drop_and_reuse() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CudaBuffer, JoinType};

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
    let default_stream = device.inner().stream();

    const LROWS: usize = 128;
    const RROWS: usize = 128;
    const ITERATIONS: usize = 512;
    const PROBE_SLOTS: usize = 16;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for _iter in 0..ITERATIONS {
        // Left keys 0..LROWS; right keys 1_000_000..1_000_000+RROWS.
        // Disjoint → every left row is unmatched.
        let mut lk = Vec::with_capacity(LROWS * 4);
        let mut lv = Vec::with_capacity(LROWS * 4);
        for i in 0..LROWS {
            lk.extend_from_slice(&(i as u32).to_le_bytes());
            lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
        }
        let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
        let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
        device
            .inner()
            .htod_sync_copy_into(&lk, &mut lk_b)
            .expect("htod lk");
        device
            .inner()
            .htod_sync_copy_into(&lv, &mut lv_b)
            .expect("htod lv");
        let lk_ptr = lk_b.device_ptr_value();
        let lv_ptr = lv_b.device_ptr_value();
        let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
        device
            .inner()
            .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
            .expect("htod l_rows");
        let left = CudaBuffer::from_columns(
            vec![lk_b.into(), lv_b.into()],
            LROWS as u64,
            l_rows,
            schema.clone(),
        );

        let mut rk = Vec::with_capacity(RROWS * 4);
        let mut rv = Vec::with_capacity(RROWS * 4);
        for j in 0..RROWS {
            rk.extend_from_slice(&((j as u32) + 1_000_000).to_le_bytes());
            rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
        }
        let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
        let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
        device
            .inner()
            .htod_sync_copy_into(&rk, &mut rk_b)
            .expect("htod rk");
        device
            .inner()
            .htod_sync_copy_into(&rv, &mut rv_b)
            .expect("htod rv");
        let rk_ptr = rk_b.device_ptr_value();
        let rv_ptr = rv_b.device_ptr_value();
        let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
        device
            .inner()
            .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
            .expect("htod r_rows");
        let right = CudaBuffer::from_columns(
            vec![rk_b.into(), rv_b.into()],
            RROWS as u64,
            r_rows,
            schema.clone(),
        );

        let result = provider
            .hash_join_v2_recorded(
                &left,
                &right,
                &[0],
                &[0],
                JoinType::LeftOuter,
                None,
                launch_stream,
            )
            .expect("hash_join_v2_recorded::<LeftOuter> all-unmatched");

        drop(left);
        drop(right);

        let mut probes: Vec<_> = (0..PROBE_SLOTS)
            .map(|_| memory.alloc::<u8>(LROWS * 4).expect("alloc probe"))
            .collect();
        let probe_ptrs: Vec<u64> = probes.iter().map(|p| p.device_ptr_value()).collect();
        let reused = probe_ptrs
            .iter()
            .any(|p| *p == lk_ptr || *p == lv_ptr || *p == rk_ptr || *p == rv_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            for &p in &probe_ptrs {
                memset_sync_default(p, TRAMPLE, LROWS * 4);
            }
        }
        let _ = &mut probes;

        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Verify: result has LROWS rows, all with original
        // (k, v) on the left and (0, 0) on the right.
        let mut host_rows = [0u32];
        device
            .inner()
            .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
            .expect("dtoh result count");
        if host_rows[0] as usize != LROWS {
            bad_output += 1;
            drop(result);
            drop(probes);
            runtime.reap_pending().expect("reap");
            continue;
        }
        let mut c0 = vec![0u8; LROWS * 4];
        let mut c1 = vec![0u8; LROWS * 4];
        let mut c2 = vec![0u8; LROWS * 4];
        let mut c3 = vec![0u8; LROWS * 4];
        unsafe {
            dtoh_sync(&mut c0, *result.column(0).expect("c0").device_ptr());
            dtoh_sync(&mut c1, *result.column(1).expect("c1").device_ptr());
            dtoh_sync(&mut c2, *result.column(2).expect("c2").device_ptr());
            dtoh_sync(&mut c3, *result.column(3).expect("c3").device_ptr());
        }
        let observed: std::collections::HashSet<(u32, u32, u32, u32)> = (0..LROWS)
            .map(|i| {
                let r0 =
                    u32::from_le_bytes([c0[i * 4], c0[i * 4 + 1], c0[i * 4 + 2], c0[i * 4 + 3]]);
                let r1 =
                    u32::from_le_bytes([c1[i * 4], c1[i * 4 + 1], c1[i * 4 + 2], c1[i * 4 + 3]]);
                let r2 =
                    u32::from_le_bytes([c2[i * 4], c2[i * 4 + 1], c2[i * 4 + 2], c2[i * 4 + 3]]);
                let r3 =
                    u32::from_le_bytes([c3[i * 4], c3[i * 4 + 1], c3[i * 4 + 2], c3[i * 4 + 3]]);
                (r0, r1, r2, r3)
            })
            .collect();
        let expected: std::collections::HashSet<(u32, u32, u32, u32)> = (0..LROWS)
            .map(|i| (i as u32, (i as u32) + 100_000, 0u32, 0u32))
            .collect();
        if observed != expected {
            bad_output += 1;
        }

        drop(result);
        drop(probes);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[left_outer_recorded all-unmatched] iters={} reuse_observed={} bad_output={}",
        ITERATIONS, reuse_observed, bad_output
    );
    assert!(reuse_observed > 0, "no reuse observed");
    assert_eq!(
        bad_output, 0,
        "all-unmatched LeftOuter produced wrong output in {}/{} iterations",
        bad_output, ITERATIONS,
    );
}

/// Slice #7C: empty-right LeftOuter. Falls back to the
/// legacy `left_outer_with_nulls` path; no launch_stream
/// work is queued, so dropping inputs after the call is safe
/// because the legacy path syncs before returning. This test
/// confirms that path is reachable from
/// `hash_join_v2_recorded(LeftOuter)` and produces the
/// expected (left columns + right zeros) shape.
#[test]
fn provider_hash_join_left_outer_v2_recorded_empty_right() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CudaBuffer, JoinType};

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
        Box::new(GlobalDeviceBudget::new(logging, 4 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(4 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire launch_stream");

    const LROWS: usize = 8;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut lk = Vec::with_capacity(LROWS * 4);
    let mut lv = Vec::with_capacity(LROWS * 4);
    for i in 0..LROWS {
        lk.extend_from_slice(&(i as u32).to_le_bytes());
        lv.extend_from_slice(&((i as u32) * 11).to_le_bytes());
    }
    let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
    let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
    device
        .inner()
        .htod_sync_copy_into(&lk, &mut lk_b)
        .expect("htod lk");
    device
        .inner()
        .htod_sync_copy_into(&lv, &mut lv_b)
        .expect("htod lv");
    let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
    device
        .inner()
        .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
        .expect("htod l_rows");
    let left = CudaBuffer::from_columns(
        vec![lk_b.into(), lv_b.into()],
        LROWS as u64,
        l_rows,
        schema.clone(),
    );

    // Empty right: zero-row buffer with the same schema.
    let rk_b = memory.alloc::<u8>(0).expect("alloc empty rk");
    let rv_b = memory.alloc::<u8>(0).expect("alloc empty rv");
    let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
    device
        .inner()
        .htod_sync_copy_into(&[0u32], &mut r_rows)
        .expect("htod r_rows");
    let right = CudaBuffer::from_columns(vec![rk_b.into(), rv_b.into()], 0, r_rows, schema.clone());

    let result = provider
        .hash_join_v2_recorded(
            &left,
            &right,
            &[0],
            &[0],
            JoinType::LeftOuter,
            None,
            launch_stream,
        )
        .expect("hash_join_v2_recorded::<LeftOuter> empty-right");

    let mut host_rows = [0u32];
    device
        .inner()
        .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
        .expect("dtoh result count");
    assert_eq!(host_rows[0] as usize, LROWS);

    let mut c0 = vec![0u8; LROWS * 4];
    let mut c1 = vec![0u8; LROWS * 4];
    let mut c2 = vec![0u8; LROWS * 4];
    let mut c3 = vec![0u8; LROWS * 4];
    unsafe {
        dtoh_sync(&mut c0, *result.column(0).expect("c0").device_ptr());
        dtoh_sync(&mut c1, *result.column(1).expect("c1").device_ptr());
        dtoh_sync(&mut c2, *result.column(2).expect("c2").device_ptr());
        dtoh_sync(&mut c3, *result.column(3).expect("c3").device_ptr());
    }
    for i in 0..LROWS {
        let r0 = u32::from_le_bytes([c0[i * 4], c0[i * 4 + 1], c0[i * 4 + 2], c0[i * 4 + 3]]);
        let r1 = u32::from_le_bytes([c1[i * 4], c1[i * 4 + 1], c1[i * 4 + 2], c1[i * 4 + 3]]);
        let r2 = u32::from_le_bytes([c2[i * 4], c2[i * 4 + 1], c2[i * 4 + 2], c2[i * 4 + 3]]);
        let r3 = u32::from_le_bytes([c3[i * 4], c3[i * 4 + 1], c3[i * 4 + 2], c3[i * 4 + 3]]);
        assert_eq!(r0, i as u32);
        assert_eq!(r1, (i as u32) * 11);
        assert_eq!(r2, 0);
        assert_eq!(r3, 0);
    }
}

/// Slice #7D: drop+reuse for indexed Inner.
/// `hash_join_v2_with_index_recorded` reuses the cached
/// `JoinIndexV2` (built via the legacy `build_join_index_v2`)
/// for the build side and packs only `left` on launch_stream.
/// After the call, drop both `input` buffers AND the `index`
/// — the recorder must keep all three alive until the
/// launch_stream chain completes.
#[test]
fn provider_hash_join_v2_with_index_recorded_inner_survives_drop_and_reuse() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CudaBuffer, JoinType};

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
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire launch_stream");
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const LROWS: usize = 256;
    const RROWS: usize = 256;
    const LKEYS: u32 = 64;
    const RKEYS: u32 = 32;
    const ITERATIONS: usize = 512;
    const PROBE_SLOTS: usize = 16;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        let mut lk = Vec::with_capacity(LROWS * 4);
        let mut lv = Vec::with_capacity(LROWS * 4);
        for i in 0..LROWS {
            lk.extend_from_slice(&((i as u32) % LKEYS).to_le_bytes());
            lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
        }
        let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
        let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
        device
            .inner()
            .htod_sync_copy_into(&lk, &mut lk_b)
            .expect("htod lk");
        device
            .inner()
            .htod_sync_copy_into(&lv, &mut lv_b)
            .expect("htod lv");
        let lk_ptr = lk_b.device_ptr_value();
        let lv_ptr = lv_b.device_ptr_value();
        let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
        device
            .inner()
            .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
            .expect("htod l_rows");
        let left = CudaBuffer::from_columns(
            vec![lk_b.into(), lv_b.into()],
            LROWS as u64,
            l_rows,
            schema.clone(),
        );

        let mut rk = Vec::with_capacity(RROWS * 4);
        let mut rv = Vec::with_capacity(RROWS * 4);
        for j in 0..RROWS {
            rk.extend_from_slice(&((j as u32) % RKEYS).to_le_bytes());
            rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
        }
        let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
        let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
        device
            .inner()
            .htod_sync_copy_into(&rk, &mut rk_b)
            .expect("htod rk");
        device
            .inner()
            .htod_sync_copy_into(&rv, &mut rv_b)
            .expect("htod rv");
        let rk_ptr = rk_b.device_ptr_value();
        let rv_ptr = rv_b.device_ptr_value();
        let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
        device
            .inner()
            .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
            .expect("htod r_rows");
        let right = CudaBuffer::from_columns(
            vec![rk_b.into(), rv_b.into()],
            RROWS as u64,
            r_rows,
            schema.clone(),
        );

        // Build the cached index (legacy default-stream).
        let index = provider
            .build_join_index_v2(&right, &[0])
            .expect("build_join_index_v2");

        let result = provider
            .hash_join_v2_with_index_recorded(
                &left,
                &right,
                &[0],
                &[0],
                JoinType::Inner,
                &index,
                None,
                launch_stream,
            )
            .expect("hash_join_v2_with_index_recorded::<Inner>");

        // Drop ALL three: input buffers + cached index. The
        // index holds packed_keys + table buffers that the
        // launch_stream chain just read; the recorder must
        // keep them alive until the chain completes.
        drop(left);
        drop(right);
        drop(index);

        let mut probes: Vec<_> = (0..PROBE_SLOTS)
            .map(|_| memory.alloc::<u8>(LROWS * 4).expect("alloc probe"))
            .collect();
        let probe_ptrs: Vec<u64> = probes.iter().map(|p| p.device_ptr_value()).collect();
        let reused = probe_ptrs
            .iter()
            .any(|p| *p == lk_ptr || *p == lv_ptr || *p == rk_ptr || *p == rv_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            for &p in &probe_ptrs {
                memset_sync_default(p, TRAMPLE, LROWS * 4);
            }
        }
        let _ = &mut probes;
        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Verify: same set as non-indexed inner.
        let mut expected: Vec<(u32, u32, u32, u32)> = Vec::new();
        for i in 0..LROWS {
            let lk_e = (i as u32) % LKEYS;
            let lv_e = (i as u32) + 100_000;
            for j in 0..RROWS {
                if (j as u32) % RKEYS == lk_e {
                    let rv_e = (j as u32) + 200_000;
                    expected.push((lk_e, lv_e, lk_e, rv_e));
                }
            }
        }
        let expected_count = expected.len();
        let mut host_rows = [0u32];
        device
            .inner()
            .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
            .expect("dtoh result count");
        if host_rows[0] as usize != expected_count {
            bad_output += 1;
            if iter == 0 {
                eprintln!(
                    "[indexed_inner] iter=0 actual={} expected={} reused={}",
                    host_rows[0], expected_count, reused
                );
            }
            drop(result);
            drop(probes);
            runtime.reap_pending().expect("reap");
            continue;
        }
        let mut c0 = vec![0u8; expected_count * 4];
        let mut c1 = vec![0u8; expected_count * 4];
        let mut c2 = vec![0u8; expected_count * 4];
        let mut c3 = vec![0u8; expected_count * 4];
        unsafe {
            dtoh_sync(&mut c0, *result.column(0).expect("c0").device_ptr());
            dtoh_sync(&mut c1, *result.column(1).expect("c1").device_ptr());
            dtoh_sync(&mut c2, *result.column(2).expect("c2").device_ptr());
            dtoh_sync(&mut c3, *result.column(3).expect("c3").device_ptr());
        }
        let observed: std::collections::HashSet<(u32, u32, u32, u32)> = (0..expected_count)
            .map(|i| {
                let r0 =
                    u32::from_le_bytes([c0[i * 4], c0[i * 4 + 1], c0[i * 4 + 2], c0[i * 4 + 3]]);
                let r1 =
                    u32::from_le_bytes([c1[i * 4], c1[i * 4 + 1], c1[i * 4 + 2], c1[i * 4 + 3]]);
                let r2 =
                    u32::from_le_bytes([c2[i * 4], c2[i * 4 + 1], c2[i * 4 + 2], c2[i * 4 + 3]]);
                let r3 =
                    u32::from_le_bytes([c3[i * 4], c3[i * 4 + 1], c3[i * 4 + 2], c3[i * 4 + 3]]);
                (r0, r1, r2, r3)
            })
            .collect();
        let expected_set: std::collections::HashSet<(u32, u32, u32, u32)> =
            expected.iter().copied().collect();
        if observed != expected_set {
            bad_output += 1;
            if iter == 0 {
                eprintln!("[indexed_inner] iter=0 set mismatch reused={}", reused);
            }
        }

        drop(result);
        drop(probes);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[indexed_inner] iterations={} reuse_observed={} bad_output={}",
        ITERATIONS, reuse_observed, bad_output
    );
    assert!(reuse_observed > 0, "no reuse observed");
    assert_eq!(
        bad_output, 0,
        "hash_join_v2_with_index_recorded::<Inner> produced corrupted output in {}/{} iterations \
         (reuse_observed={}). The index's packed_keys / table buffers were not properly \
         recorded against the launch_stream chain.",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Slice #7D: drop+reuse for indexed Anti.
#[test]
fn provider_hash_join_v2_with_index_recorded_anti_survives_drop_and_reuse() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CudaBuffer, JoinType};

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
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire launch_stream");
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const LROWS: usize = 256;
    const RROWS: usize = 256;
    const LKEYS: u32 = 64;
    const RKEYS: u32 = 32;
    const ITERATIONS: usize = 512;
    const PROBE_SLOTS: usize = 16;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for _iter in 0..ITERATIONS {
        let mut lk = Vec::with_capacity(LROWS * 4);
        let mut lv = Vec::with_capacity(LROWS * 4);
        for i in 0..LROWS {
            lk.extend_from_slice(&((i as u32) % LKEYS).to_le_bytes());
            lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
        }
        let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
        let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
        device
            .inner()
            .htod_sync_copy_into(&lk, &mut lk_b)
            .expect("htod lk");
        device
            .inner()
            .htod_sync_copy_into(&lv, &mut lv_b)
            .expect("htod lv");
        let lk_ptr = lk_b.device_ptr_value();
        let lv_ptr = lv_b.device_ptr_value();
        let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
        device
            .inner()
            .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
            .expect("htod l_rows");
        let left = CudaBuffer::from_columns(
            vec![lk_b.into(), lv_b.into()],
            LROWS as u64,
            l_rows,
            schema.clone(),
        );

        let mut rk = Vec::with_capacity(RROWS * 4);
        let mut rv = Vec::with_capacity(RROWS * 4);
        for j in 0..RROWS {
            rk.extend_from_slice(&((j as u32) % RKEYS).to_le_bytes());
            rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
        }
        let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
        let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
        device
            .inner()
            .htod_sync_copy_into(&rk, &mut rk_b)
            .expect("htod rk");
        device
            .inner()
            .htod_sync_copy_into(&rv, &mut rv_b)
            .expect("htod rv");
        let rk_ptr = rk_b.device_ptr_value();
        let rv_ptr = rv_b.device_ptr_value();
        let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
        device
            .inner()
            .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
            .expect("htod r_rows");
        let right = CudaBuffer::from_columns(
            vec![rk_b.into(), rv_b.into()],
            RROWS as u64,
            r_rows,
            schema.clone(),
        );
        let index = provider
            .build_join_index_v2(&right, &[0])
            .expect("build_join_index_v2");

        let result = provider
            .hash_join_v2_with_index_recorded(
                &left,
                &right,
                &[0],
                &[0],
                JoinType::Anti,
                &index,
                None,
                launch_stream,
            )
            .expect("indexed Anti");

        drop(left);
        drop(right);
        drop(index);

        let mut probes: Vec<_> = (0..PROBE_SLOTS)
            .map(|_| memory.alloc::<u8>(LROWS * 4).expect("alloc probe"))
            .collect();
        let probe_ptrs: Vec<u64> = probes.iter().map(|p| p.device_ptr_value()).collect();
        let reused = probe_ptrs
            .iter()
            .any(|p| *p == lk_ptr || *p == lv_ptr || *p == rk_ptr || *p == rv_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            for &p in &probe_ptrs {
                memset_sync_default(p, TRAMPLE, LROWS * 4);
            }
        }
        let _ = &mut probes;

        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        let expected: Vec<(u32, u32)> = (0..LROWS)
            .filter_map(|i| {
                let k = (i as u32) % LKEYS;
                if k >= RKEYS {
                    Some((k, (i as u32) + 100_000))
                } else {
                    None
                }
            })
            .collect();
        let expected_count = expected.len();
        let mut host_rows = [0u32];
        device
            .inner()
            .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
            .expect("dtoh result count");
        if host_rows[0] as usize != expected_count {
            bad_output += 1;
            drop(result);
            drop(probes);
            runtime.reap_pending().expect("reap");
            continue;
        }
        let mut k_back = vec![0u8; expected_count * 4];
        let mut v_back = vec![0u8; expected_count * 4];
        unsafe {
            dtoh_sync(&mut k_back, *result.column(0).expect("c0").device_ptr());
            dtoh_sync(&mut v_back, *result.column(1).expect("c1").device_ptr());
        }
        let observed: std::collections::HashSet<(u32, u32)> = (0..expected_count)
            .map(|i| {
                let k = u32::from_le_bytes([
                    k_back[i * 4],
                    k_back[i * 4 + 1],
                    k_back[i * 4 + 2],
                    k_back[i * 4 + 3],
                ]);
                let v = u32::from_le_bytes([
                    v_back[i * 4],
                    v_back[i * 4 + 1],
                    v_back[i * 4 + 2],
                    v_back[i * 4 + 3],
                ]);
                (k, v)
            })
            .collect();
        let expected_set: std::collections::HashSet<(u32, u32)> =
            expected.iter().copied().collect();
        if observed != expected_set {
            bad_output += 1;
        }

        drop(result);
        drop(probes);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[indexed_anti] iterations={} reuse_observed={} bad_output={}",
        ITERATIONS, reuse_observed, bad_output
    );
    assert!(reuse_observed > 0, "no reuse observed");
    assert_eq!(
        bad_output, 0,
        "hash_join_v2_with_index_recorded::<Anti> produced corrupted output \
         in {}/{} iterations (reuse_observed={}).",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Slice #7D: drop+reuse for indexed LeftOuter (partial-match).
#[test]
fn provider_hash_join_v2_with_index_recorded_left_outer_survives_drop_and_reuse() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CudaBuffer, JoinType};

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
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire launch_stream");
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const LROWS: usize = 256;
    const RROWS: usize = 256;
    const LKEYS: u32 = 64;
    const RKEYS: u32 = 32;
    // Indexed LeftOuter has the largest fresh-allocation
    // count of any recorded path (probe count+materialize +
    // gather ×2 + mask_not + compact tail + per-column
    // dtod-async concat), and adds the index buffers to the
    // pool. Need higher pressure than other indexed paths.
    const ITERATIONS: usize = 1024;
    const PROBE_SLOTS: usize = 24;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        let mut lk = Vec::with_capacity(LROWS * 4);
        let mut lv = Vec::with_capacity(LROWS * 4);
        for i in 0..LROWS {
            lk.extend_from_slice(&((i as u32) % LKEYS).to_le_bytes());
            lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
        }
        let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
        let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
        device
            .inner()
            .htod_sync_copy_into(&lk, &mut lk_b)
            .expect("htod lk");
        device
            .inner()
            .htod_sync_copy_into(&lv, &mut lv_b)
            .expect("htod lv");
        let lk_ptr = lk_b.device_ptr_value();
        let lv_ptr = lv_b.device_ptr_value();
        let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
        device
            .inner()
            .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
            .expect("htod l_rows");
        let left = CudaBuffer::from_columns(
            vec![lk_b.into(), lv_b.into()],
            LROWS as u64,
            l_rows,
            schema.clone(),
        );

        let mut rk = Vec::with_capacity(RROWS * 4);
        let mut rv = Vec::with_capacity(RROWS * 4);
        for j in 0..RROWS {
            rk.extend_from_slice(&((j as u32) % RKEYS).to_le_bytes());
            rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
        }
        let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
        let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
        device
            .inner()
            .htod_sync_copy_into(&rk, &mut rk_b)
            .expect("htod rk");
        device
            .inner()
            .htod_sync_copy_into(&rv, &mut rv_b)
            .expect("htod rv");
        let rk_ptr = rk_b.device_ptr_value();
        let rv_ptr = rv_b.device_ptr_value();
        let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
        device
            .inner()
            .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
            .expect("htod r_rows");
        let right = CudaBuffer::from_columns(
            vec![rk_b.into(), rv_b.into()],
            RROWS as u64,
            r_rows,
            schema.clone(),
        );
        let index = provider
            .build_join_index_v2(&right, &[0])
            .expect("build_join_index_v2");

        let result = provider
            .hash_join_v2_with_index_recorded(
                &left,
                &right,
                &[0],
                &[0],
                JoinType::LeftOuter,
                &index,
                None,
                launch_stream,
            )
            .expect("indexed LeftOuter");

        drop(left);
        drop(right);
        drop(index);

        let mut probes: Vec<_> = (0..PROBE_SLOTS)
            .map(|_| memory.alloc::<u8>(LROWS * 4).expect("alloc probe"))
            .collect();
        let probe_ptrs: Vec<u64> = probes.iter().map(|p| p.device_ptr_value()).collect();
        let reused = probe_ptrs
            .iter()
            .any(|p| *p == lk_ptr || *p == lv_ptr || *p == rk_ptr || *p == rv_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            for &p in &probe_ptrs {
                memset_sync_default(p, TRAMPLE, LROWS * 4);
            }
        }
        let _ = &mut probes;
        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Same expected multiset as non-indexed LeftOuter.
        let mut expected: std::collections::HashMap<(u32, u32, u32, u32), usize> =
            std::collections::HashMap::new();
        for i in 0..LROWS {
            let lk_e = (i as u32) % LKEYS;
            let lv_e = (i as u32) + 100_000;
            if lk_e < RKEYS {
                for j in 0..RROWS {
                    if (j as u32) % RKEYS == lk_e {
                        let rv_e = (j as u32) + 200_000;
                        *expected.entry((lk_e, lv_e, lk_e, rv_e)).or_insert(0) += 1;
                    }
                }
            } else {
                *expected.entry((lk_e, lv_e, 0, 0)).or_insert(0) += 1;
            }
        }
        let expected_count: usize = expected.values().sum();
        let mut host_rows = [0u32];
        device
            .inner()
            .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
            .expect("dtoh result count");
        if host_rows[0] as usize != expected_count {
            bad_output += 1;
            if iter == 0 {
                eprintln!(
                    "[indexed_left_outer] iter=0 actual={} expected={} reused={}",
                    host_rows[0], expected_count, reused
                );
            }
            drop(result);
            drop(probes);
            runtime.reap_pending().expect("reap");
            continue;
        }
        let mut c0 = vec![0u8; expected_count * 4];
        let mut c1 = vec![0u8; expected_count * 4];
        let mut c2 = vec![0u8; expected_count * 4];
        let mut c3 = vec![0u8; expected_count * 4];
        unsafe {
            dtoh_sync(&mut c0, *result.column(0).expect("c0").device_ptr());
            dtoh_sync(&mut c1, *result.column(1).expect("c1").device_ptr());
            dtoh_sync(&mut c2, *result.column(2).expect("c2").device_ptr());
            dtoh_sync(&mut c3, *result.column(3).expect("c3").device_ptr());
        }
        let mut observed: std::collections::HashMap<(u32, u32, u32, u32), usize> =
            std::collections::HashMap::new();
        for i in 0..expected_count {
            let r0 = u32::from_le_bytes([c0[i * 4], c0[i * 4 + 1], c0[i * 4 + 2], c0[i * 4 + 3]]);
            let r1 = u32::from_le_bytes([c1[i * 4], c1[i * 4 + 1], c1[i * 4 + 2], c1[i * 4 + 3]]);
            let r2 = u32::from_le_bytes([c2[i * 4], c2[i * 4 + 1], c2[i * 4 + 2], c2[i * 4 + 3]]);
            let r3 = u32::from_le_bytes([c3[i * 4], c3[i * 4 + 1], c3[i * 4 + 2], c3[i * 4 + 3]]);
            *observed.entry((r0, r1, r2, r3)).or_insert(0) += 1;
        }
        if observed != expected {
            bad_output += 1;
        }

        drop(result);
        drop(probes);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[indexed_left_outer] iterations={} reuse_observed={} bad_output={}",
        ITERATIONS, reuse_observed, bad_output
    );
    assert!(reuse_observed > 0, "no reuse observed");
    assert_eq!(
        bad_output, 0,
        "hash_join_v2_with_index_recorded::<LeftOuter> produced corrupted output \
         in {}/{} iterations (reuse_observed={}).",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Negative test: indexed recorded path against a no-runtime
/// manager. Must reject before any allocation / kernel.
#[test]
fn provider_hash_join_v2_with_index_recorded_rejects_legacy_manager() {
    use xlog_core::{ScalarType, Schema};
    use xlog_cuda::{CudaBuffer, JoinType};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let memory = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(1024 * 1024),
    ));
    let provider =
        CudaKernelProvider::new(Arc::clone(&device), Arc::clone(&memory)).expect("legacy provider");

    let mut lk = memory.alloc::<u8>(16).expect("alloc lk");
    let kvb: Vec<u8> = [0u32, 1, 2, 3]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    device
        .inner()
        .htod_sync_copy_into(&kvb, &mut lk)
        .expect("htod lk");
    let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut l_rows)
        .expect("htod l_rows");
    let left = CudaBuffer::from_columns(
        vec![lk.into()],
        4,
        l_rows,
        Schema::new(vec![("k".to_string(), ScalarType::U32)]),
    );
    let mut rk = memory.alloc::<u8>(16).expect("alloc rk");
    device
        .inner()
        .htod_sync_copy_into(&kvb, &mut rk)
        .expect("htod rk");
    let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut r_rows)
        .expect("htod r_rows");
    let right = CudaBuffer::from_columns(
        vec![rk.into()],
        4,
        r_rows,
        Schema::new(vec![("k".to_string(), ScalarType::U32)]),
    );
    let index = provider
        .build_join_index_v2(&right, &[0])
        .expect("build_join_index_v2");

    let err = provider.hash_join_v2_with_index_recorded(
        &left,
        &right,
        &[0],
        &[0],
        JoinType::Inner,
        &index,
        None,
        StreamId::DEFAULT,
    );
    match err {
        Err(XlogError::Kernel(msg)) => assert!(
            msg.contains("requires") || msg.contains("with_runtime"),
            "expected helpful Kernel error, got {:?}",
            msg
        ),
        Err(other) => panic!(
            "hash_join_v2_with_index_recorded must reject legacy manager with Kernel error, got {:?}",
            other
        ),
        Ok(_) => panic!(
            "hash_join_v2_with_index_recorded must reject legacy manager — unexpectedly returned Ok"
        ),
    }
}

/// Negative test: recorded sort against a no-runtime manager.
#[test]
fn provider_sort_recorded_rejects_legacy_manager() {
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
    let mut col = memory.alloc::<u8>(16).expect("alloc col");
    let payload = [3u32, 1, 2, 0];
    let bytes: Vec<u8> = payload.iter().flat_map(|v| v.to_le_bytes()).collect();
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut col)
        .expect("htod col");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut d_num_rows)
        .expect("htod rows");
    let input = CudaBuffer::from_columns(
        vec![col.into()],
        4,
        d_num_rows,
        Schema::new(vec![("v".to_string(), ScalarType::U32)]),
    );
    let err = provider.sort_recorded(&input, &[0], StreamId::DEFAULT);
    match err {
        Err(XlogError::Kernel(msg)) => assert!(
            msg.contains("requires") || msg.contains("with_runtime"),
            "expected helpful Kernel error, got {:?}",
            msg
        ),
        Err(other) => panic!(
            "sort_recorded must reject legacy manager with XlogError::Kernel, got {:?}",
            other
        ),
        Ok(_) => panic!("sort_recorded must reject legacy manager — unexpectedly returned Ok"),
    }
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

/// Regression for recorded compaction with `row_cap > logical_count`.
///
/// Recorded hash-join Semi/Anti produces a mask sized to the
/// left input's logical row count, while recursive buffers can
/// retain a larger row capacity. The compact path must expand
/// that short logical-domain mask through `mask_clamp_rows` and
/// then use the clamped row-capacity-domain mask for BOTH
/// `capture_compact_count` and `compact_bytes_by_mask`.
#[test]
fn provider_compact_recorded_short_mask_ignores_capacity_slack() {
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
    let launch_handle = pool.resolve(launch_stream).expect("resolve");

    const ROW_CAP: usize = 8;
    const LOGICAL: usize = 3;
    let payload = [10u32, 20, 30, 200, 201, 202, 203, 204];
    let mut col_bytes = memory.alloc::<u8>(ROW_CAP * 4).expect("alloc col");
    let bytes: Vec<u8> = payload.iter().flat_map(|v| v.to_le_bytes()).collect();
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut col_bytes)
        .expect("htod col");

    // Logical-domain mask: keep rows 0 and 2. There are no
    // mask bytes for capacity slack rows [3, ROW_CAP).
    let mut d_mask = memory.alloc::<u8>(LOGICAL).expect("alloc short mask");
    device
        .inner()
        .htod_sync_copy_into(&[1u8, 0, 1], &mut d_mask)
        .expect("htod mask");

    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rowcount");
    device
        .inner()
        .htod_sync_copy_into(&[LOGICAL as u32], &mut d_num_rows)
        .expect("htod rows");
    let input = CudaBuffer::from_columns(
        vec![col_bytes.into()],
        ROW_CAP as u64,
        d_num_rows,
        Schema::new(vec![("v".to_string(), ScalarType::U32)]),
    );

    let output = provider
        .compact_buffer_by_device_mask_counted_recorded(&input, &d_mask, launch_stream)
        .expect("compact recorded short mask");
    launch_handle.synchronize().expect("sync launch");

    let mut host_rows = [0u32];
    device
        .inner()
        .dtoh_sync_copy_into(output.num_rows_device(), &mut host_rows)
        .expect("dtoh row count");
    assert_eq!(host_rows[0], 2);

    let out_col = output.column(0).expect("output col");
    let mut readback = vec![0u8; 2 * 4];
    unsafe { dtoh_sync(&mut readback, *out_col.device_ptr()) };
    let observed: Vec<u32> = readback
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(observed, vec![10, 30]);
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

/// Result-correctness regression for the count-scan-materialize
/// (CSM) Inner hash join (binary-join retake sub-slice #1).
///
/// Asserts the deterministic-ordering CSM path produces the
/// same result-set as the existing recorded inner path. The
/// CSM kernel writes `output[per_probe_offsets[tid] + local]`
/// directly with no atomic on `output_count`, so output
/// ordering is a deterministic function of (probe-row index,
/// per-row match discovery order). This test compares the
/// result-set as a HashSet (set equality) against expected
/// rows; the deterministic-ordering invariant is exercised
/// indirectly by the drop+reuse test which depends on
/// race-free output across many iterations.
#[test]
fn provider_hash_join_inner_csm_v2_recorded_result_set_matches() {
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
    let launch_handle = pool.resolve(launch_stream).expect("resolve");

    const LROWS: usize = 64;
    const RROWS: usize = 64;
    const LKEYS: u32 = 16;
    const RKEYS: u32 = 8;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    // Left/right with overlapping key spaces.
    let mut lk = Vec::with_capacity(LROWS * 4);
    let mut lv = Vec::with_capacity(LROWS * 4);
    for i in 0..LROWS {
        lk.extend_from_slice(&((i as u32) % LKEYS).to_le_bytes());
        lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
    }
    let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
    let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
    device
        .inner()
        .htod_sync_copy_into(&lk, &mut lk_b)
        .expect("htod lk");
    device
        .inner()
        .htod_sync_copy_into(&lv, &mut lv_b)
        .expect("htod lv");
    let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
    device
        .inner()
        .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
        .expect("htod l_rows");
    let left = CudaBuffer::from_columns(
        vec![lk_b.into(), lv_b.into()],
        LROWS as u64,
        l_rows,
        schema.clone(),
    );

    let mut rk = Vec::with_capacity(RROWS * 4);
    let mut rv = Vec::with_capacity(RROWS * 4);
    for j in 0..RROWS {
        rk.extend_from_slice(&((j as u32) % RKEYS).to_le_bytes());
        rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
    }
    let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
    let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
    device
        .inner()
        .htod_sync_copy_into(&rk, &mut rk_b)
        .expect("htod rk");
    device
        .inner()
        .htod_sync_copy_into(&rv, &mut rv_b)
        .expect("htod rv");
    let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
    device
        .inner()
        .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
        .expect("htod r_rows");
    let right = CudaBuffer::from_columns(
        vec![rk_b.into(), rv_b.into()],
        RROWS as u64,
        r_rows,
        schema.clone(),
    );

    let result = provider
        .hash_join_inner_v2_count_scan_materialize_recorded(
            &left,
            &right,
            &[0],
            &[0],
            None,
            launch_stream,
        )
        .expect("CSM inner");
    launch_handle.synchronize().expect("sync launch");

    // Build expected set host-side: every (i,j) where
    // i % LKEYS == j % RKEYS.
    let mut expected: std::collections::HashSet<(u32, u32, u32, u32)> = Default::default();
    for i in 0..LROWS {
        let lk_e = (i as u32) % LKEYS;
        let lv_e = (i as u32) + 100_000;
        for j in 0..RROWS {
            if (j as u32) % RKEYS == lk_e {
                let rv_e = (j as u32) + 200_000;
                expected.insert((lk_e, lv_e, lk_e, rv_e));
            }
        }
    }
    let mut host_rows = [0u32];
    device
        .inner()
        .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
        .expect("dtoh count");
    assert_eq!(host_rows[0] as usize, expected.len(), "row count mismatch");

    let n = expected.len();
    let mut c0 = vec![0u8; n * 4];
    let mut c1 = vec![0u8; n * 4];
    let mut c2 = vec![0u8; n * 4];
    let mut c3 = vec![0u8; n * 4];
    unsafe {
        dtoh_sync(&mut c0, *result.column(0).expect("c0").device_ptr());
        dtoh_sync(&mut c1, *result.column(1).expect("c1").device_ptr());
        dtoh_sync(&mut c2, *result.column(2).expect("c2").device_ptr());
        dtoh_sync(&mut c3, *result.column(3).expect("c3").device_ptr());
    }
    let observed: std::collections::HashSet<(u32, u32, u32, u32)> = (0..n)
        .map(|i| {
            let r0 = u32::from_le_bytes([c0[i * 4], c0[i * 4 + 1], c0[i * 4 + 2], c0[i * 4 + 3]]);
            let r1 = u32::from_le_bytes([c1[i * 4], c1[i * 4 + 1], c1[i * 4 + 2], c1[i * 4 + 3]]);
            let r2 = u32::from_le_bytes([c2[i * 4], c2[i * 4 + 1], c2[i * 4 + 2], c2[i * 4 + 3]]);
            let r3 = u32::from_le_bytes([c3[i * 4], c3[i * 4 + 1], c3[i * 4 + 2], c3[i * 4 + 3]]);
            (r0, r1, r2, r3)
        })
        .collect();
    assert_eq!(observed, expected, "result set mismatch");
}

/// Drop+reuse for the CSM Inner path. Same shape as
/// `provider_hash_join_inner_v2_recorded_survives_drop_and_reuse`:
/// after the CSM call returns, drops both inputs WITHOUT host
/// sync, reuses + tramples slots, asserts result-set integrity.
#[test]
fn provider_hash_join_inner_csm_v2_recorded_survives_drop_and_reuse() {
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
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire launch_stream");
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const LROWS: usize = 256;
    const RROWS: usize = 256;
    const LKEYS: u32 = 64;
    const RKEYS: u32 = 32;
    const ITERATIONS: usize = 512;
    const PROBE_SLOTS: usize = 16;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        let mut lk = Vec::with_capacity(LROWS * 4);
        let mut lv = Vec::with_capacity(LROWS * 4);
        for i in 0..LROWS {
            lk.extend_from_slice(&((i as u32) % LKEYS).to_le_bytes());
            lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
        }
        let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
        let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
        device
            .inner()
            .htod_sync_copy_into(&lk, &mut lk_b)
            .expect("htod lk");
        device
            .inner()
            .htod_sync_copy_into(&lv, &mut lv_b)
            .expect("htod lv");
        let lk_ptr = lk_b.device_ptr_value();
        let lv_ptr = lv_b.device_ptr_value();
        let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
        device
            .inner()
            .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
            .expect("htod l_rows");
        let left = CudaBuffer::from_columns(
            vec![lk_b.into(), lv_b.into()],
            LROWS as u64,
            l_rows,
            schema.clone(),
        );

        let mut rk = Vec::with_capacity(RROWS * 4);
        let mut rv = Vec::with_capacity(RROWS * 4);
        for j in 0..RROWS {
            rk.extend_from_slice(&((j as u32) % RKEYS).to_le_bytes());
            rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
        }
        let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
        let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
        device
            .inner()
            .htod_sync_copy_into(&rk, &mut rk_b)
            .expect("htod rk");
        device
            .inner()
            .htod_sync_copy_into(&rv, &mut rv_b)
            .expect("htod rv");
        let rk_ptr = rk_b.device_ptr_value();
        let rv_ptr = rv_b.device_ptr_value();
        let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
        device
            .inner()
            .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
            .expect("htod r_rows");
        let right = CudaBuffer::from_columns(
            vec![rk_b.into(), rv_b.into()],
            RROWS as u64,
            r_rows,
            schema.clone(),
        );

        let result = provider
            .hash_join_inner_v2_count_scan_materialize_recorded(
                &left,
                &right,
                &[0],
                &[0],
                None,
                launch_stream,
            )
            .expect("CSM inner");

        drop(left);
        drop(right);

        let mut probes: Vec<_> = (0..PROBE_SLOTS)
            .map(|_| memory.alloc::<u8>(LROWS * 4).expect("alloc probe"))
            .collect();
        let probe_ptrs: Vec<u64> = probes.iter().map(|p| p.device_ptr_value()).collect();
        let reused = probe_ptrs
            .iter()
            .any(|p| *p == lk_ptr || *p == lv_ptr || *p == rk_ptr || *p == rv_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            for &p in &probe_ptrs {
                memset_sync_default(p, TRAMPLE, LROWS * 4);
            }
        }
        let _ = &mut probes;
        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        let mut expected: Vec<(u32, u32, u32, u32)> = Vec::new();
        for i in 0..LROWS {
            let lk_e = (i as u32) % LKEYS;
            let lv_e = (i as u32) + 100_000;
            for j in 0..RROWS {
                if (j as u32) % RKEYS == lk_e {
                    let rv_e = (j as u32) + 200_000;
                    expected.push((lk_e, lv_e, lk_e, rv_e));
                }
            }
        }
        let expected_set: std::collections::HashSet<(u32, u32, u32, u32)> =
            expected.iter().copied().collect();
        let expected_count = expected.len();
        let mut host_rows = [0u32];
        device
            .inner()
            .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
            .expect("dtoh result count");
        if host_rows[0] as usize != expected_count {
            bad_output += 1;
            if iter == 0 {
                eprintln!(
                    "[csm_inner] iter=0 actual={} expected={} reused={}",
                    host_rows[0], expected_count, reused
                );
            }
            drop(result);
            drop(probes);
            runtime.reap_pending().expect("reap");
            continue;
        }
        let mut c0 = vec![0u8; expected_count * 4];
        let mut c1 = vec![0u8; expected_count * 4];
        let mut c2 = vec![0u8; expected_count * 4];
        let mut c3 = vec![0u8; expected_count * 4];
        unsafe {
            dtoh_sync(&mut c0, *result.column(0).expect("c0").device_ptr());
            dtoh_sync(&mut c1, *result.column(1).expect("c1").device_ptr());
            dtoh_sync(&mut c2, *result.column(2).expect("c2").device_ptr());
            dtoh_sync(&mut c3, *result.column(3).expect("c3").device_ptr());
        }
        let observed: std::collections::HashSet<(u32, u32, u32, u32)> = (0..expected_count)
            .map(|i| {
                let r0 =
                    u32::from_le_bytes([c0[i * 4], c0[i * 4 + 1], c0[i * 4 + 2], c0[i * 4 + 3]]);
                let r1 =
                    u32::from_le_bytes([c1[i * 4], c1[i * 4 + 1], c1[i * 4 + 2], c1[i * 4 + 3]]);
                let r2 =
                    u32::from_le_bytes([c2[i * 4], c2[i * 4 + 1], c2[i * 4 + 2], c2[i * 4 + 3]]);
                let r3 =
                    u32::from_le_bytes([c3[i * 4], c3[i * 4 + 1], c3[i * 4 + 2], c3[i * 4 + 3]]);
                (r0, r1, r2, r3)
            })
            .collect();
        if observed != expected_set {
            bad_output += 1;
            if iter == 0 {
                eprintln!("[csm_inner] iter=0 set mismatch reused={}", reused);
            }
        }

        drop(result);
        drop(probes);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[csm_inner] iterations={} reuse_observed={} bad_output={}",
        ITERATIONS, reuse_observed, bad_output
    );
    assert!(reuse_observed > 0, "no reuse observed");
    assert_eq!(
        bad_output, 0,
        "CSM inner produced corrupted output in {}/{} iterations \
         (reuse_observed={}). One of the chain steps (count, scan, total, \
         materialize, gather) raced an alloc-stream reuse + trample.",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Negative test: CSM Inner against a no-runtime manager.
#[test]
fn provider_hash_join_inner_csm_v2_recorded_rejects_legacy_manager() {
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

    let mut lk = memory.alloc::<u8>(16).expect("alloc lk");
    let mut rk = memory.alloc::<u8>(16).expect("alloc rk");
    let payload = [0u32, 1, 2, 3];
    let bytes: Vec<u8> = payload.iter().flat_map(|x| x.to_le_bytes()).collect();
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut lk)
        .expect("htod lk");
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut rk)
        .expect("htod rk");
    let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut l_rows)
        .expect("htod l_rows");
    let left = CudaBuffer::from_columns(
        vec![lk.into()],
        4,
        l_rows,
        Schema::new(vec![("k".to_string(), ScalarType::U32)]),
    );
    let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut r_rows)
        .expect("htod r_rows");
    let right = CudaBuffer::from_columns(
        vec![rk.into()],
        4,
        r_rows,
        Schema::new(vec![("k".to_string(), ScalarType::U32)]),
    );

    let err = provider.hash_join_inner_v2_count_scan_materialize_recorded(
        &left,
        &right,
        &[0],
        &[0],
        None,
        StreamId::DEFAULT,
    );
    match err {
        Err(XlogError::Kernel(msg)) => assert!(
            msg.contains("requires") || msg.contains("with_runtime"),
            "expected helpful Kernel error, got {:?}",
            msg
        ),
        Err(other) => panic!(
            "CSM inner must reject legacy manager with Kernel error, got {:?}",
            other
        ),
        Ok(_) => panic!("CSM inner must reject legacy manager — unexpectedly returned Ok"),
    }
}

/// Result-set correctness for indexed-Inner CSM
/// (binary-join retake sub-slice 2). Mirrors the non-indexed
/// CSM correctness test but uses a cached `JoinIndexV2` for
/// the build side.
#[test]
fn provider_hash_join_inner_csm_v2_with_index_recorded_result_set_matches() {
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
    let launch_handle = pool.resolve(launch_stream).expect("resolve");

    const LROWS: usize = 64;
    const RROWS: usize = 64;
    const LKEYS: u32 = 16;
    const RKEYS: u32 = 8;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut lk = Vec::with_capacity(LROWS * 4);
    let mut lv = Vec::with_capacity(LROWS * 4);
    for i in 0..LROWS {
        lk.extend_from_slice(&((i as u32) % LKEYS).to_le_bytes());
        lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
    }
    let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
    let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
    device
        .inner()
        .htod_sync_copy_into(&lk, &mut lk_b)
        .expect("htod lk");
    device
        .inner()
        .htod_sync_copy_into(&lv, &mut lv_b)
        .expect("htod lv");
    let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
    device
        .inner()
        .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
        .expect("htod l_rows");
    let left = CudaBuffer::from_columns(
        vec![lk_b.into(), lv_b.into()],
        LROWS as u64,
        l_rows,
        schema.clone(),
    );

    let mut rk = Vec::with_capacity(RROWS * 4);
    let mut rv = Vec::with_capacity(RROWS * 4);
    for j in 0..RROWS {
        rk.extend_from_slice(&((j as u32) % RKEYS).to_le_bytes());
        rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
    }
    let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
    let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
    device
        .inner()
        .htod_sync_copy_into(&rk, &mut rk_b)
        .expect("htod rk");
    device
        .inner()
        .htod_sync_copy_into(&rv, &mut rv_b)
        .expect("htod rv");
    let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
    device
        .inner()
        .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
        .expect("htod r_rows");
    let right = CudaBuffer::from_columns(
        vec![rk_b.into(), rv_b.into()],
        RROWS as u64,
        r_rows,
        schema.clone(),
    );

    let index = provider
        .build_join_index_v2(&right, &[0])
        .expect("build_join_index_v2");

    let result = provider
        .hash_join_inner_v2_with_index_count_scan_materialize_recorded(
            &left,
            &right,
            &[0],
            &[0],
            &index,
            None,
            launch_stream,
        )
        .expect("indexed CSM inner");
    launch_handle.synchronize().expect("sync launch");

    let mut expected: std::collections::HashSet<(u32, u32, u32, u32)> = Default::default();
    for i in 0..LROWS {
        let lk_e = (i as u32) % LKEYS;
        let lv_e = (i as u32) + 100_000;
        for j in 0..RROWS {
            if (j as u32) % RKEYS == lk_e {
                let rv_e = (j as u32) + 200_000;
                expected.insert((lk_e, lv_e, lk_e, rv_e));
            }
        }
    }
    let mut host_rows = [0u32];
    device
        .inner()
        .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
        .expect("dtoh count");
    assert_eq!(host_rows[0] as usize, expected.len(), "row count mismatch");

    let n = expected.len();
    let mut c0 = vec![0u8; n * 4];
    let mut c1 = vec![0u8; n * 4];
    let mut c2 = vec![0u8; n * 4];
    let mut c3 = vec![0u8; n * 4];
    unsafe {
        dtoh_sync(&mut c0, *result.column(0).expect("c0").device_ptr());
        dtoh_sync(&mut c1, *result.column(1).expect("c1").device_ptr());
        dtoh_sync(&mut c2, *result.column(2).expect("c2").device_ptr());
        dtoh_sync(&mut c3, *result.column(3).expect("c3").device_ptr());
    }
    let observed: std::collections::HashSet<(u32, u32, u32, u32)> = (0..n)
        .map(|i| {
            let r0 = u32::from_le_bytes([c0[i * 4], c0[i * 4 + 1], c0[i * 4 + 2], c0[i * 4 + 3]]);
            let r1 = u32::from_le_bytes([c1[i * 4], c1[i * 4 + 1], c1[i * 4 + 2], c1[i * 4 + 3]]);
            let r2 = u32::from_le_bytes([c2[i * 4], c2[i * 4 + 1], c2[i * 4 + 2], c2[i * 4 + 3]]);
            let r3 = u32::from_le_bytes([c3[i * 4], c3[i * 4 + 1], c3[i * 4 + 2], c3[i * 4 + 3]]);
            (r0, r1, r2, r3)
        })
        .collect();
    assert_eq!(observed, expected, "result set mismatch");
}

/// Drop+reuse for indexed-Inner CSM. Drops `left`, `right`,
/// AND `index` after the call without host sync — the index
/// holds packed_keys + 4 table buckets that the chain reads
/// on launch_stream; the recorder must keep them all alive
/// until the chain completes.
#[test]
fn provider_hash_join_inner_csm_v2_with_index_recorded_survives_drop_and_reuse() {
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
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire launch_stream");
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const LROWS: usize = 256;
    const RROWS: usize = 256;
    const LKEYS: u32 = 64;
    const RKEYS: u32 = 32;
    const ITERATIONS: usize = 512;
    const PROBE_SLOTS: usize = 16;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for iter in 0..ITERATIONS {
        let mut lk = Vec::with_capacity(LROWS * 4);
        let mut lv = Vec::with_capacity(LROWS * 4);
        for i in 0..LROWS {
            lk.extend_from_slice(&((i as u32) % LKEYS).to_le_bytes());
            lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
        }
        let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
        let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
        device
            .inner()
            .htod_sync_copy_into(&lk, &mut lk_b)
            .expect("htod lk");
        device
            .inner()
            .htod_sync_copy_into(&lv, &mut lv_b)
            .expect("htod lv");
        let lk_ptr = lk_b.device_ptr_value();
        let lv_ptr = lv_b.device_ptr_value();
        let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
        device
            .inner()
            .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
            .expect("htod l_rows");
        let left = CudaBuffer::from_columns(
            vec![lk_b.into(), lv_b.into()],
            LROWS as u64,
            l_rows,
            schema.clone(),
        );

        let mut rk = Vec::with_capacity(RROWS * 4);
        let mut rv = Vec::with_capacity(RROWS * 4);
        for j in 0..RROWS {
            rk.extend_from_slice(&((j as u32) % RKEYS).to_le_bytes());
            rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
        }
        let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
        let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
        device
            .inner()
            .htod_sync_copy_into(&rk, &mut rk_b)
            .expect("htod rk");
        device
            .inner()
            .htod_sync_copy_into(&rv, &mut rv_b)
            .expect("htod rv");
        let rk_ptr = rk_b.device_ptr_value();
        let rv_ptr = rv_b.device_ptr_value();
        let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
        device
            .inner()
            .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
            .expect("htod r_rows");
        let right = CudaBuffer::from_columns(
            vec![rk_b.into(), rv_b.into()],
            RROWS as u64,
            r_rows,
            schema.clone(),
        );

        let index = provider
            .build_join_index_v2(&right, &[0])
            .expect("build_join_index_v2");

        let result = provider
            .hash_join_inner_v2_with_index_count_scan_materialize_recorded(
                &left,
                &right,
                &[0],
                &[0],
                &index,
                None,
                launch_stream,
            )
            .expect("indexed CSM inner");

        drop(left);
        drop(right);
        drop(index);

        let mut probes: Vec<_> = (0..PROBE_SLOTS)
            .map(|_| memory.alloc::<u8>(LROWS * 4).expect("alloc probe"))
            .collect();
        let probe_ptrs: Vec<u64> = probes.iter().map(|p| p.device_ptr_value()).collect();
        let reused = probe_ptrs
            .iter()
            .any(|p| *p == lk_ptr || *p == lv_ptr || *p == rk_ptr || *p == rv_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            for &p in &probe_ptrs {
                memset_sync_default(p, TRAMPLE, LROWS * 4);
            }
        }
        let _ = &mut probes;
        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        let mut expected: Vec<(u32, u32, u32, u32)> = Vec::new();
        for i in 0..LROWS {
            let lk_e = (i as u32) % LKEYS;
            let lv_e = (i as u32) + 100_000;
            for j in 0..RROWS {
                if (j as u32) % RKEYS == lk_e {
                    let rv_e = (j as u32) + 200_000;
                    expected.push((lk_e, lv_e, lk_e, rv_e));
                }
            }
        }
        let expected_set: std::collections::HashSet<(u32, u32, u32, u32)> =
            expected.iter().copied().collect();
        let expected_count = expected.len();
        let mut host_rows = [0u32];
        device
            .inner()
            .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
            .expect("dtoh result count");
        if host_rows[0] as usize != expected_count {
            bad_output += 1;
            if iter == 0 {
                eprintln!(
                    "[indexed_csm_inner] iter=0 actual={} expected={} reused={}",
                    host_rows[0], expected_count, reused
                );
            }
            drop(result);
            drop(probes);
            runtime.reap_pending().expect("reap");
            continue;
        }
        let mut c0 = vec![0u8; expected_count * 4];
        let mut c1 = vec![0u8; expected_count * 4];
        let mut c2 = vec![0u8; expected_count * 4];
        let mut c3 = vec![0u8; expected_count * 4];
        unsafe {
            dtoh_sync(&mut c0, *result.column(0).expect("c0").device_ptr());
            dtoh_sync(&mut c1, *result.column(1).expect("c1").device_ptr());
            dtoh_sync(&mut c2, *result.column(2).expect("c2").device_ptr());
            dtoh_sync(&mut c3, *result.column(3).expect("c3").device_ptr());
        }
        let observed: std::collections::HashSet<(u32, u32, u32, u32)> = (0..expected_count)
            .map(|i| {
                let r0 =
                    u32::from_le_bytes([c0[i * 4], c0[i * 4 + 1], c0[i * 4 + 2], c0[i * 4 + 3]]);
                let r1 =
                    u32::from_le_bytes([c1[i * 4], c1[i * 4 + 1], c1[i * 4 + 2], c1[i * 4 + 3]]);
                let r2 =
                    u32::from_le_bytes([c2[i * 4], c2[i * 4 + 1], c2[i * 4 + 2], c2[i * 4 + 3]]);
                let r3 =
                    u32::from_le_bytes([c3[i * 4], c3[i * 4 + 1], c3[i * 4 + 2], c3[i * 4 + 3]]);
                (r0, r1, r2, r3)
            })
            .collect();
        if observed != expected_set {
            bad_output += 1;
            if iter == 0 {
                eprintln!("[indexed_csm_inner] iter=0 set mismatch reused={}", reused);
            }
        }

        drop(result);
        drop(probes);
        runtime.reap_pending().expect("reap");
    }

    eprintln!(
        "[indexed_csm_inner] iterations={} reuse_observed={} bad_output={}",
        ITERATIONS, reuse_observed, bad_output
    );
    assert!(reuse_observed > 0, "no reuse observed");
    assert_eq!(
        bad_output, 0,
        "indexed CSM inner produced corrupted output in {}/{} iterations \
         (reuse_observed={}). The chain (count, scan, total, materialize, \
         gather) raced an alloc-stream reuse + trample, OR the index \
         buffers' lifetime was not properly recorded.",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Negative test: indexed CSM Inner against a no-runtime
/// manager.
#[test]
fn provider_hash_join_inner_csm_v2_with_index_recorded_rejects_legacy_manager() {
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

    let mut lk = memory.alloc::<u8>(16).expect("alloc lk");
    let mut rk = memory.alloc::<u8>(16).expect("alloc rk");
    let payload = [0u32, 1, 2, 3];
    let bytes: Vec<u8> = payload.iter().flat_map(|x| x.to_le_bytes()).collect();
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut lk)
        .expect("htod lk");
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut rk)
        .expect("htod rk");
    let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut l_rows)
        .expect("htod l_rows");
    let left = CudaBuffer::from_columns(
        vec![lk.into()],
        4,
        l_rows,
        Schema::new(vec![("k".to_string(), ScalarType::U32)]),
    );
    let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut r_rows)
        .expect("htod r_rows");
    let right = CudaBuffer::from_columns(
        vec![rk.into()],
        4,
        r_rows,
        Schema::new(vec![("k".to_string(), ScalarType::U32)]),
    );
    let index = provider
        .build_join_index_v2(&right, &[0])
        .expect("build_join_index_v2");

    let err = provider.hash_join_inner_v2_with_index_count_scan_materialize_recorded(
        &left,
        &right,
        &[0],
        &[0],
        &index,
        None,
        StreamId::DEFAULT,
    );
    match err {
        Err(XlogError::Kernel(msg)) => assert!(
            msg.contains("requires") || msg.contains("with_runtime"),
            "expected helpful Kernel error, got {:?}",
            msg
        ),
        Err(other) => panic!(
            "indexed CSM inner must reject legacy manager with Kernel error, got {:?}",
            other
        ),
        Ok(_) => panic!("indexed CSM inner must reject legacy manager — unexpectedly returned Ok"),
    }
}

// ───────────────────────────────────────────────────────────
// Sub-slice 3: non-indexed LeftOuter CSM
// (`hash_join_left_outer_v2_count_scan_materialize_recorded`)
//
// Same gate matrix as sub-slices 1 & 2: result-set
// correctness, partial-match drop+reuse, all-unmatched
// drop+reuse, empty-right fallback, legacy-manager
// rejection. Calls the new method directly (no env
// dispatch wiring in this slice).
// ───────────────────────────────────────────────────────────

/// Result-set correctness for non-indexed LeftOuter CSM.
/// Builds left/right with overlapping key spaces so both the
/// matched branch (lk < RKEYS) AND the unmatched-left branch
/// (lk >= RKEYS) are exercised. Compares observed multiset
/// to host-computed expected multiset.
#[test]
fn provider_hash_join_left_outer_csm_v2_recorded_result_set_matches() {
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
    let launch_handle = pool.resolve(launch_stream).expect("resolve");

    const LROWS: usize = 64;
    const RROWS: usize = 64;
    const LKEYS: u32 = 16;
    const RKEYS: u32 = 8;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut lk = Vec::with_capacity(LROWS * 4);
    let mut lv = Vec::with_capacity(LROWS * 4);
    for i in 0..LROWS {
        lk.extend_from_slice(&((i as u32) % LKEYS).to_le_bytes());
        lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
    }
    let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
    let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
    device
        .inner()
        .htod_sync_copy_into(&lk, &mut lk_b)
        .expect("htod lk");
    device
        .inner()
        .htod_sync_copy_into(&lv, &mut lv_b)
        .expect("htod lv");
    let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
    device
        .inner()
        .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
        .expect("htod l_rows");
    let left = CudaBuffer::from_columns(
        vec![lk_b.into(), lv_b.into()],
        LROWS as u64,
        l_rows,
        schema.clone(),
    );

    let mut rk = Vec::with_capacity(RROWS * 4);
    let mut rv = Vec::with_capacity(RROWS * 4);
    for j in 0..RROWS {
        rk.extend_from_slice(&((j as u32) % RKEYS).to_le_bytes());
        rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
    }
    let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
    let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
    device
        .inner()
        .htod_sync_copy_into(&rk, &mut rk_b)
        .expect("htod rk");
    device
        .inner()
        .htod_sync_copy_into(&rv, &mut rv_b)
        .expect("htod rv");
    let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
    device
        .inner()
        .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
        .expect("htod r_rows");
    let right = CudaBuffer::from_columns(
        vec![rk_b.into(), rv_b.into()],
        RROWS as u64,
        r_rows,
        schema.clone(),
    );

    let result = provider
        .hash_join_left_outer_v2_count_scan_materialize_recorded(
            &left,
            &right,
            &[0],
            &[0],
            None,
            launch_stream,
        )
        .expect("CSM left_outer");
    launch_handle.synchronize().expect("sync launch");

    // Build expected multiset host-side:
    //   matched: (lk, lv, lk, rv) for every (i,j) where
    //     i % LKEYS == j % RKEYS (only when lk < RKEYS).
    //   unmatched: (lk, lv, 0, 0) for every i where
    //     i % LKEYS >= RKEYS.
    let mut expected: std::collections::HashMap<(u32, u32, u32, u32), usize> =
        std::collections::HashMap::new();
    for i in 0..LROWS {
        let lk_e = (i as u32) % LKEYS;
        let lv_e = (i as u32) + 100_000;
        if lk_e < RKEYS {
            for j in 0..RROWS {
                if (j as u32) % RKEYS == lk_e {
                    let rv_e = (j as u32) + 200_000;
                    *expected.entry((lk_e, lv_e, lk_e, rv_e)).or_insert(0) += 1;
                }
            }
        } else {
            *expected.entry((lk_e, lv_e, 0, 0)).or_insert(0) += 1;
        }
    }
    let expected_count: usize = expected.values().sum();

    let mut host_rows = [0u32];
    device
        .inner()
        .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
        .expect("dtoh count");
    assert_eq!(
        host_rows[0] as usize, expected_count,
        "row count mismatch (matched + unmatched)"
    );

    let n = expected_count;
    let mut c0 = vec![0u8; n * 4];
    let mut c1 = vec![0u8; n * 4];
    let mut c2 = vec![0u8; n * 4];
    let mut c3 = vec![0u8; n * 4];
    unsafe {
        dtoh_sync(&mut c0, *result.column(0).expect("c0").device_ptr());
        dtoh_sync(&mut c1, *result.column(1).expect("c1").device_ptr());
        dtoh_sync(&mut c2, *result.column(2).expect("c2").device_ptr());
        dtoh_sync(&mut c3, *result.column(3).expect("c3").device_ptr());
    }
    let mut observed: std::collections::HashMap<(u32, u32, u32, u32), usize> =
        std::collections::HashMap::new();
    for i in 0..n {
        let r0 = u32::from_le_bytes([c0[i * 4], c0[i * 4 + 1], c0[i * 4 + 2], c0[i * 4 + 3]]);
        let r1 = u32::from_le_bytes([c1[i * 4], c1[i * 4 + 1], c1[i * 4 + 2], c1[i * 4 + 3]]);
        let r2 = u32::from_le_bytes([c2[i * 4], c2[i * 4 + 1], c2[i * 4 + 2], c2[i * 4 + 3]]);
        let r3 = u32::from_le_bytes([c3[i * 4], c3[i * 4 + 1], c3[i * 4 + 2], c3[i * 4 + 3]]);
        *observed.entry((r0, r1, r2, r3)).or_insert(0) += 1;
    }
    assert_eq!(observed, expected, "result multiset mismatch");
}

/// Drop+reuse stress with PARTIAL-match graphs.
/// Same shape as
/// `provider_hash_join_left_outer_v2_recorded_partial_match_survives_drop_and_reuse`
/// but routed through the CSM path. Verifies the recorded
/// chain keeps both the matched and unmatched outputs alive
/// after the inputs are dropped + their addresses reused +
/// trampled.
#[test]
fn provider_hash_join_left_outer_csm_v2_recorded_partial_match_survives_drop_and_reuse() {
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
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire launch_stream");
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    // Match the existing non-CSM LeftOuter partial-match test
    // pressure so address reuse is observable on the same
    // mempool sizes.
    const LROWS: usize = 256;
    const RROWS: usize = 256;
    const LKEYS: u32 = 64;
    const RKEYS: u32 = 32;
    const ITERATIONS: usize = 512;
    const PROBE_SLOTS: usize = 16;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for _ in 0..ITERATIONS {
        let mut lk = Vec::with_capacity(LROWS * 4);
        let mut lv = Vec::with_capacity(LROWS * 4);
        for i in 0..LROWS {
            lk.extend_from_slice(&((i as u32) % LKEYS).to_le_bytes());
            lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
        }
        let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
        let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
        device
            .inner()
            .htod_sync_copy_into(&lk, &mut lk_b)
            .expect("htod lk");
        device
            .inner()
            .htod_sync_copy_into(&lv, &mut lv_b)
            .expect("htod lv");
        let lk_ptr = lk_b.device_ptr_value();
        let lv_ptr = lv_b.device_ptr_value();
        let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
        device
            .inner()
            .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
            .expect("htod l_rows");
        let left = CudaBuffer::from_columns(
            vec![lk_b.into(), lv_b.into()],
            LROWS as u64,
            l_rows,
            schema.clone(),
        );

        let mut rk = Vec::with_capacity(RROWS * 4);
        let mut rv = Vec::with_capacity(RROWS * 4);
        for j in 0..RROWS {
            rk.extend_from_slice(&((j as u32) % RKEYS).to_le_bytes());
            rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
        }
        let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
        let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
        device
            .inner()
            .htod_sync_copy_into(&rk, &mut rk_b)
            .expect("htod rk");
        device
            .inner()
            .htod_sync_copy_into(&rv, &mut rv_b)
            .expect("htod rv");
        let rk_ptr = rk_b.device_ptr_value();
        let rv_ptr = rv_b.device_ptr_value();
        let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
        device
            .inner()
            .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
            .expect("htod r_rows");
        let right = CudaBuffer::from_columns(
            vec![rk_b.into(), rv_b.into()],
            RROWS as u64,
            r_rows,
            schema.clone(),
        );

        let result = provider
            .hash_join_left_outer_v2_count_scan_materialize_recorded(
                &left,
                &right,
                &[0],
                &[0],
                None,
                launch_stream,
            )
            .expect("CSM left_outer");

        // Drop both inputs WITHOUT host sync.
        drop(left);
        drop(right);

        // Reuse + trample.
        let mut probes: Vec<_> = (0..PROBE_SLOTS)
            .map(|_| memory.alloc::<u8>(LROWS * 4).expect("alloc probe"))
            .collect();
        let probe_ptrs: Vec<u64> = probes.iter().map(|p| p.device_ptr_value()).collect();
        let reused = probe_ptrs
            .iter()
            .any(|p| *p == lk_ptr || *p == lv_ptr || *p == rk_ptr || *p == rv_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            for &p in &probe_ptrs {
                memset_sync_default(p, TRAMPLE, LROWS * 4);
            }
        }
        let _ = &mut probes;

        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // Verify result. Expected: matched (i,j) where
        // i % LKEYS == j % RKEYS for lk < RKEYS, plus
        // (lk, lv, 0, 0) for lk >= RKEYS.
        let mut expected: std::collections::HashMap<(u32, u32, u32, u32), usize> =
            std::collections::HashMap::new();
        for i in 0..LROWS {
            let lk_e = (i as u32) % LKEYS;
            let lv_e = (i as u32) + 100_000;
            if lk_e < RKEYS {
                for j in 0..RROWS {
                    if (j as u32) % RKEYS == lk_e {
                        let rv_e = (j as u32) + 200_000;
                        *expected.entry((lk_e, lv_e, lk_e, rv_e)).or_insert(0) += 1;
                    }
                }
            } else {
                *expected.entry((lk_e, lv_e, 0, 0)).or_insert(0) += 1;
            }
        }
        let expected_count: usize = expected.values().sum();

        let mut host_rows = [0u32];
        device
            .inner()
            .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
            .expect("dtoh count");
        if host_rows[0] as usize != expected_count {
            bad_output += 1;
            drop(result);
            drop(probes);
            runtime.reap_pending().expect("reap");
            continue;
        }

        let n = expected_count;
        let mut c0 = vec![0u8; n * 4];
        let mut c1 = vec![0u8; n * 4];
        let mut c2 = vec![0u8; n * 4];
        let mut c3 = vec![0u8; n * 4];
        unsafe {
            dtoh_sync(&mut c0, *result.column(0).expect("c0").device_ptr());
            dtoh_sync(&mut c1, *result.column(1).expect("c1").device_ptr());
            dtoh_sync(&mut c2, *result.column(2).expect("c2").device_ptr());
            dtoh_sync(&mut c3, *result.column(3).expect("c3").device_ptr());
        }
        let mut observed: std::collections::HashMap<(u32, u32, u32, u32), usize> =
            std::collections::HashMap::new();
        for i in 0..n {
            let r0 = u32::from_le_bytes([c0[i * 4], c0[i * 4 + 1], c0[i * 4 + 2], c0[i * 4 + 3]]);
            let r1 = u32::from_le_bytes([c1[i * 4], c1[i * 4 + 1], c1[i * 4 + 2], c1[i * 4 + 3]]);
            let r2 = u32::from_le_bytes([c2[i * 4], c2[i * 4 + 1], c2[i * 4 + 2], c2[i * 4 + 3]]);
            let r3 = u32::from_le_bytes([c3[i * 4], c3[i * 4 + 1], c3[i * 4 + 2], c3[i * 4 + 3]]);
            *observed.entry((r0, r1, r2, r3)).or_insert(0) += 1;
        }
        if observed != expected {
            bad_output += 1;
        }

        drop(result);
        drop(probes);
        runtime.reap_pending().expect("reap");
    }

    assert!(
        reuse_observed > 0,
        "address reuse never observed across {} iterations; cannot exercise \
         the cross-stream lifetime safety path",
        ITERATIONS
    );
    assert_eq!(
        bad_output, 0,
        "CSM left_outer produced corrupted output in {}/{} iterations \
         (reuse_observed={})",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Drop+reuse stress with ALL-UNMATCHED graphs.
/// Disjoint key spaces so every left row goes through the
/// unmatched-left branch (no matched output, only the
/// unmatched + zero-right portion of the result). Verifies
/// the recorded chain keeps the result alive through
/// reuse+trample even when the inner-matched branch is
/// empty.
#[test]
fn provider_hash_join_left_outer_csm_v2_recorded_all_unmatched_survives_drop_and_reuse() {
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
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire launch_stream");
    let launch_handle = pool.resolve(launch_stream).expect("resolve");
    let default_stream = device.inner().stream();

    const LROWS: usize = 256;
    const RROWS: usize = 128;
    // Match the existing non-CSM LeftOuter all-unmatched test
    // pressure so address reuse is observable on the same
    // mempool sizes.
    const ITERATIONS: usize = 512;
    const PROBE_SLOTS: usize = 16;
    const TRAMPLE: u8 = 0xEE;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let mut reuse_observed = 0usize;
    let mut bad_output = 0usize;

    for _ in 0..ITERATIONS {
        // Disjoint key spaces: left keys in [0, 100), right
        // keys in [1000, 1064). Every left row is unmatched.
        let mut lk = Vec::with_capacity(LROWS * 4);
        let mut lv = Vec::with_capacity(LROWS * 4);
        for i in 0..LROWS {
            lk.extend_from_slice(&((i as u32) % 100).to_le_bytes());
            lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
        }
        let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
        let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
        device
            .inner()
            .htod_sync_copy_into(&lk, &mut lk_b)
            .expect("htod lk");
        device
            .inner()
            .htod_sync_copy_into(&lv, &mut lv_b)
            .expect("htod lv");
        let lk_ptr = lk_b.device_ptr_value();
        let lv_ptr = lv_b.device_ptr_value();
        let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
        device
            .inner()
            .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
            .expect("htod l_rows");
        let left = CudaBuffer::from_columns(
            vec![lk_b.into(), lv_b.into()],
            LROWS as u64,
            l_rows,
            schema.clone(),
        );

        let mut rk = Vec::with_capacity(RROWS * 4);
        let mut rv = Vec::with_capacity(RROWS * 4);
        for j in 0..RROWS {
            rk.extend_from_slice(&(1000u32 + (j as u32)).to_le_bytes());
            rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
        }
        let mut rk_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
        let mut rv_b = memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
        device
            .inner()
            .htod_sync_copy_into(&rk, &mut rk_b)
            .expect("htod rk");
        device
            .inner()
            .htod_sync_copy_into(&rv, &mut rv_b)
            .expect("htod rv");
        let rk_ptr = rk_b.device_ptr_value();
        let rv_ptr = rv_b.device_ptr_value();
        let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
        device
            .inner()
            .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
            .expect("htod r_rows");
        let right = CudaBuffer::from_columns(
            vec![rk_b.into(), rv_b.into()],
            RROWS as u64,
            r_rows,
            schema.clone(),
        );

        let result = provider
            .hash_join_left_outer_v2_count_scan_materialize_recorded(
                &left,
                &right,
                &[0],
                &[0],
                None,
                launch_stream,
            )
            .expect("CSM left_outer all-unmatched");

        drop(left);
        drop(right);

        let mut probes: Vec<_> = (0..PROBE_SLOTS)
            .map(|_| memory.alloc::<u8>(LROWS * 4).expect("alloc probe"))
            .collect();
        let probe_ptrs: Vec<u64> = probes.iter().map(|p| p.device_ptr_value()).collect();
        let reused = probe_ptrs
            .iter()
            .any(|p| *p == lk_ptr || *p == lv_ptr || *p == rk_ptr || *p == rv_ptr);
        if reused {
            reuse_observed += 1;
        }
        unsafe {
            for &p in &probe_ptrs {
                memset_sync_default(p, TRAMPLE, LROWS * 4);
            }
        }
        let _ = &mut probes;

        launch_handle.synchronize().expect("sync launch");
        default_stream.synchronize().expect("sync default");

        // All-unmatched: result has exactly LROWS rows of
        // (lk, lv, 0, 0).
        let mut host_rows = [0u32];
        device
            .inner()
            .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
            .expect("dtoh count");
        if host_rows[0] as usize != LROWS {
            bad_output += 1;
            drop(result);
            drop(probes);
            runtime.reap_pending().expect("reap");
            continue;
        }

        let n = LROWS;
        let mut c0 = vec![0u8; n * 4];
        let mut c1 = vec![0u8; n * 4];
        let mut c2 = vec![0u8; n * 4];
        let mut c3 = vec![0u8; n * 4];
        unsafe {
            dtoh_sync(&mut c0, *result.column(0).expect("c0").device_ptr());
            dtoh_sync(&mut c1, *result.column(1).expect("c1").device_ptr());
            dtoh_sync(&mut c2, *result.column(2).expect("c2").device_ptr());
            dtoh_sync(&mut c3, *result.column(3).expect("c3").device_ptr());
        }
        let mut expected: std::collections::HashMap<(u32, u32, u32, u32), usize> =
            std::collections::HashMap::new();
        for i in 0..LROWS {
            let lk_e = (i as u32) % 100;
            let lv_e = (i as u32) + 100_000;
            *expected.entry((lk_e, lv_e, 0, 0)).or_insert(0) += 1;
        }
        let mut observed: std::collections::HashMap<(u32, u32, u32, u32), usize> =
            std::collections::HashMap::new();
        for i in 0..n {
            let r0 = u32::from_le_bytes([c0[i * 4], c0[i * 4 + 1], c0[i * 4 + 2], c0[i * 4 + 3]]);
            let r1 = u32::from_le_bytes([c1[i * 4], c1[i * 4 + 1], c1[i * 4 + 2], c1[i * 4 + 3]]);
            let r2 = u32::from_le_bytes([c2[i * 4], c2[i * 4 + 1], c2[i * 4 + 2], c2[i * 4 + 3]]);
            let r3 = u32::from_le_bytes([c3[i * 4], c3[i * 4 + 1], c3[i * 4 + 2], c3[i * 4 + 3]]);
            *observed.entry((r0, r1, r2, r3)).or_insert(0) += 1;
        }
        if observed != expected {
            bad_output += 1;
        }

        drop(result);
        drop(probes);
        runtime.reap_pending().expect("reap");
    }

    assert!(
        reuse_observed > 0,
        "address reuse never observed across {} iterations",
        ITERATIONS
    );
    assert_eq!(
        bad_output, 0,
        "CSM left_outer (all-unmatched) produced corrupted output in {}/{} iterations \
         (reuse_observed={})",
        bad_output, ITERATIONS, reuse_observed,
    );
}

/// Empty-right input: every left row should appear in the
/// output with zero-filled right columns.
#[test]
fn provider_hash_join_left_outer_csm_v2_recorded_empty_right() {
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
        Box::new(GlobalDeviceBudget::new(logging, 16 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(16 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    let launch_stream = pool.acquire().expect("acquire launch_stream");

    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    const LROWS: usize = 32;
    let mut lk = Vec::with_capacity(LROWS * 4);
    let mut lv = Vec::with_capacity(LROWS * 4);
    for i in 0..LROWS {
        lk.extend_from_slice(&(i as u32).to_le_bytes());
        lv.extend_from_slice(&((i as u32) + 9_000).to_le_bytes());
    }
    let mut lk_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
    let mut lv_b = memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
    device
        .inner()
        .htod_sync_copy_into(&lk, &mut lk_b)
        .expect("htod lk");
    device
        .inner()
        .htod_sync_copy_into(&lv, &mut lv_b)
        .expect("htod lv");
    let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
    device
        .inner()
        .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
        .expect("htod l_rows");
    let left = CudaBuffer::from_columns(
        vec![lk_b.into(), lv_b.into()],
        LROWS as u64,
        l_rows,
        schema.clone(),
    );

    // Empty right.
    let right = provider
        .create_empty_buffer(schema.clone())
        .expect("empty right");

    let result = provider
        .hash_join_left_outer_v2_count_scan_materialize_recorded(
            &left,
            &right,
            &[0],
            &[0],
            None,
            launch_stream,
        )
        .expect("CSM left_outer empty-right");

    let mut host_rows = [0u32];
    device
        .inner()
        .dtoh_sync_copy_into(result.num_rows_device(), &mut host_rows)
        .expect("dtoh count");
    assert_eq!(
        host_rows[0] as usize, LROWS,
        "empty-right LeftOuter must produce one row per left row"
    );

    let n = LROWS;
    let mut c0 = vec![0u8; n * 4];
    let mut c1 = vec![0u8; n * 4];
    let mut c2 = vec![0u8; n * 4];
    let mut c3 = vec![0u8; n * 4];
    unsafe {
        dtoh_sync(&mut c0, *result.column(0).expect("c0").device_ptr());
        dtoh_sync(&mut c1, *result.column(1).expect("c1").device_ptr());
        dtoh_sync(&mut c2, *result.column(2).expect("c2").device_ptr());
        dtoh_sync(&mut c3, *result.column(3).expect("c3").device_ptr());
    }
    let mut expected: std::collections::HashMap<(u32, u32, u32, u32), usize> =
        std::collections::HashMap::new();
    for i in 0..LROWS {
        *expected
            .entry(((i as u32), (i as u32) + 9_000, 0, 0))
            .or_insert(0) += 1;
    }
    let mut observed: std::collections::HashMap<(u32, u32, u32, u32), usize> =
        std::collections::HashMap::new();
    for i in 0..n {
        let r0 = u32::from_le_bytes([c0[i * 4], c0[i * 4 + 1], c0[i * 4 + 2], c0[i * 4 + 3]]);
        let r1 = u32::from_le_bytes([c1[i * 4], c1[i * 4 + 1], c1[i * 4 + 2], c1[i * 4 + 3]]);
        let r2 = u32::from_le_bytes([c2[i * 4], c2[i * 4 + 1], c2[i * 4 + 2], c2[i * 4 + 3]]);
        let r3 = u32::from_le_bytes([c3[i * 4], c3[i * 4 + 1], c3[i * 4 + 2], c3[i * 4 + 3]]);
        *observed.entry((r0, r1, r2, r3)).or_insert(0) += 1;
    }
    assert_eq!(observed, expected, "empty-right LeftOuter result mismatch");
}

/// Legacy-manager rejection: CSM LeftOuter must surface a
/// helpful Kernel error when the manager has no runtime.
#[test]
fn provider_hash_join_left_outer_csm_v2_recorded_rejects_legacy_manager() {
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

    let mut lk = memory.alloc::<u8>(16).expect("alloc lk");
    let mut rk = memory.alloc::<u8>(16).expect("alloc rk");
    let payload = [0u32, 1, 2, 3];
    let bytes: Vec<u8> = payload.iter().flat_map(|x| x.to_le_bytes()).collect();
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut lk)
        .expect("htod lk");
    device
        .inner()
        .htod_sync_copy_into(&bytes, &mut rk)
        .expect("htod rk");
    let mut l_rows = memory.alloc::<u32>(1).expect("alloc l_rows");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut l_rows)
        .expect("htod l_rows");
    let left = CudaBuffer::from_columns(
        vec![lk.into()],
        4,
        l_rows,
        Schema::new(vec![("k".to_string(), ScalarType::U32)]),
    );
    let mut r_rows = memory.alloc::<u32>(1).expect("alloc r_rows");
    device
        .inner()
        .htod_sync_copy_into(&[4u32], &mut r_rows)
        .expect("htod r_rows");
    let right = CudaBuffer::from_columns(
        vec![rk.into()],
        4,
        r_rows,
        Schema::new(vec![("k".to_string(), ScalarType::U32)]),
    );

    let err = provider.hash_join_left_outer_v2_count_scan_materialize_recorded(
        &left,
        &right,
        &[0],
        &[0],
        None,
        StreamId::DEFAULT,
    );
    match err {
        Err(XlogError::Kernel(msg)) => assert!(
            msg.contains("requires") || msg.contains("with_runtime"),
            "expected helpful Kernel error, got {:?}",
            msg
        ),
        Err(other) => panic!(
            "CSM left_outer must reject legacy manager with Kernel error, got {:?}",
            other
        ),
        Ok(_) => panic!("CSM left_outer must reject legacy manager — unexpectedly returned Ok"),
    }
}
