// crates/xlog-cuda/tests/test_provider_runtime_routing.rs
//! First-slice migration test: prove `GpuMemoryManager::alloc_raw`
//! routes through an [`XlogDeviceRuntime`] composed via
//! [`XlogDeviceRuntime::with_resource`], stacking
//! [`GlobalDeviceBudget`] over [`LoggingResource`] over
//! [`AsyncCudaResource`].
//!
//! Scope is intentionally narrow:
//!   * Only `alloc_raw` flows through the runtime.
//!   * The pre-existing `alloc::<T>` typed-slice path is out of
//!     scope for this slice — it stays on the cudarc-default
//!     allocator and existing tests cover it.
//!   * `CudaKernelProvider`'s public surface is unchanged. This
//!     test composes a `GpuMemoryManager` via `with_runtime`
//!     directly; the provider construction in production code
//!     still uses `GpuMemoryManager::new` and is unaffected.
//!
//! What this test asserts:
//!   1. A successful `alloc_raw` produces a record in the logging
//!      sink, increases the runtime's `bytes_outstanding`, and
//!      raises both the local manager counter and the budget's
//!      reserved bytes.
//!   2. Dropping the returned [`RuntimeAllocBlock`] releases both
//!      the manager counter (immediately) and, after a
//!      `runtime.reap_pending()`, the runtime's reserved bytes
//!      (correctly held while async-free is queued).
//!   3. An over-limit `alloc_raw` returns
//!      `XlogError::ResourceExhausted` originating from the budget,
//!      with no leak to the local counter and no log record for
//!      the failed inner allocation.

use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::device_runtime::{
    AllocTag, AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, InMemorySink, LogAction,
    LogResult, LoggingResource, LoggingSink, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::{CudaDevice, GpuMemoryManager};

const RUNTIME_LIMIT: usize = 32 * 1024;
const LOCAL_BUDGET: u64 = 64 * 1024;

fn build_stack() -> Option<(
    Arc<GpuMemoryManager>,
    Arc<XlogDeviceRuntime>,
    Arc<InMemorySink>,
)> {
    let device = CudaDevice::new(0).ok().map(Arc::new)?;
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let sink: Arc<InMemorySink> = Arc::new(InMemorySink::new());

    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        sink.clone() as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, RUNTIME_LIMIT));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));

    let mut local_budget = MemoryBudget::default();
    local_budget.device_bytes = LOCAL_BUDGET;
    let manager = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        local_budget,
        Arc::clone(&runtime),
    ));
    Some((manager, runtime, sink))
}

#[test]
fn alloc_raw_routes_through_runtime_budget_and_logging() {
    let Some((manager, runtime, sink)) = build_stack() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    assert!(manager.runtime().is_some());

    let block = manager
        .alloc_raw(4096, AllocTag("provider-rt-A"))
        .expect("alloc_raw under budget");
    assert_eq!(block.bytes(), 4096);
    assert!(block.ptr() != 0, "ptr must be non-null");
    assert_eq!(manager.allocated_bytes(), 4096);
    assert_eq!(runtime.bytes_outstanding(), 4096);

    let recs = sink.snapshot();
    assert_eq!(recs.len(), 1, "expected exactly one record, got {:?}", recs);
    assert_eq!(recs[0].action, LogAction::Allocate);
    assert_eq!(recs[0].result, LogResult::Ok);
    assert_eq!(recs[0].bytes, Some(4096));
    assert_eq!(recs[0].tag, Some(AllocTag("provider-rt-A")));

    // Drop the block: manager counter releases immediately, runtime
    // counter holds bytes pending until reap_pending drains.
    drop(block);
    assert_eq!(manager.allocated_bytes(), 0);
    assert_eq!(
        runtime.bytes_outstanding(),
        4096,
        "async inner: runtime holds bytes until reap"
    );

    runtime.reap_pending().expect("reap");
    assert_eq!(runtime.bytes_outstanding(), 0);

    let recs = sink.snapshot();
    assert_eq!(
        recs.len(),
        3,
        "expected alloc + dealloc + reap records, got {:?}",
        recs
    );
    assert_eq!(recs[1].action, LogAction::Deallocate);
    assert_eq!(recs[1].result, LogResult::Ok);
    assert_eq!(recs[2].action, LogAction::ReapPending);
    assert_eq!(recs[2].result, LogResult::Ok);
}

#[test]
fn alloc_raw_rejected_by_runtime_budget_does_not_leak_local_counter() {
    let Some((manager, runtime, sink)) = build_stack() else {
        return;
    };

    // RUNTIME_LIMIT is 32 KiB; LOCAL_BUDGET is 64 KiB. Asking for
    // 40 KiB hits the runtime budget first.
    let req = RUNTIME_LIMIT + 8 * 1024;
    let err = manager.alloc_raw(req, AllocTag("provider-rt-too-big"));
    assert!(
        err.is_err(),
        "alloc_raw must reject over runtime budget, got {:?}",
        err.as_ref().map(|b| b.bytes())
    );
    // Match the error variant: ResourceExhausted should be raised
    // by the runtime-budget rejection path.
    match err {
        Err(xlog_core::XlogError::ResourceExhausted { .. }) => {}
        other => panic!("expected ResourceExhausted, got {:?}", other),
    }

    // Local counter must not be left in a partially-reserved state.
    assert_eq!(manager.allocated_bytes(), 0);
    assert_eq!(runtime.bytes_outstanding(), 0);

    // The logger sits *inside* the budget in this stack, so the
    // OutOfBudget rejection at the budget layer short-circuits
    // before reaching the logger.
    assert_eq!(
        sink.len(),
        0,
        "OutOfBudget at the budget layer must not produce a log record"
    );
}

#[test]
fn alloc_raw_rejected_by_local_budget_does_not_call_runtime() {
    // Local budget is the smaller one for this test.
    let device = match CudaDevice::new(0).ok().map(Arc::new) {
        Some(d) => d,
        None => return,
    };
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let sink: Arc<InMemorySink> = Arc::new(InMemorySink::new());

    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        sink.clone() as Arc<dyn LoggingSink>,
    ));
    // Runtime budget set very large — local budget is the binding
    // constraint here.
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));

    let mut local = MemoryBudget::default();
    local.device_bytes = 4096;
    let manager = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        local,
        Arc::clone(&runtime),
    ));

    let err = manager.alloc_raw(8192, AllocTag::UNTAGGED);
    match err {
        Err(xlog_core::XlogError::ResourceExhausted { .. }) => {}
        other => panic!("expected local-budget ResourceExhausted, got {:?}", other),
    }
    assert_eq!(manager.allocated_bytes(), 0);
    assert_eq!(runtime.bytes_outstanding(), 0);
    assert_eq!(
        sink.len(),
        0,
        "local-budget rejection must short-circuit before the runtime"
    );
}

#[test]
fn alloc_u8_via_runtime_records_in_sink_and_releases_on_drop() {
    // Typed-slice path through the runtime: alloc::<u8>(len) on a
    // manager built via with_runtime must produce a TrackedCudaSlice<u8>
    // whose underlying allocation is owned by the runtime.
    // Allocate, observe sink + counters, drop, reap, observe release.
    let Some((manager, runtime, sink)) = build_stack() else {
        return;
    };

    let len = 1024usize;
    let slice = manager.alloc::<u8>(len).expect("alloc<u8> via runtime");
    assert_eq!(slice.len(), len);
    // Both counters reflect the allocation.
    assert_eq!(manager.allocated_bytes(), len as u64);
    assert_eq!(runtime.bytes_outstanding(), len);

    let recs = sink.snapshot();
    assert_eq!(recs.len(), 1, "expected 1 alloc record, got {:?}", recs);
    assert_eq!(recs[0].action, LogAction::Allocate);
    assert_eq!(recs[0].result, LogResult::Ok);
    assert_eq!(recs[0].bytes, Some(len));

    // Drop frees through the runtime (Backing::Runtime branch).
    drop(slice);
    assert_eq!(manager.allocated_bytes(), 0);
    assert_eq!(
        runtime.bytes_outstanding(),
        len,
        "async backend: runtime holds bytes pending until reap"
    );

    runtime.reap_pending().expect("reap");
    assert_eq!(runtime.bytes_outstanding(), 0);

    let recs = sink.snapshot();
    assert!(
        recs.iter().any(|r| r.action == LogAction::Deallocate),
        "expected a Deallocate record after drop, got {:?}",
        recs
    );
}

#[test]
fn alloc_non_byte_type_via_runtime_routes_correctly() {
    // Non-byte (4-byte) element type: alloc::<u32>(len). Verifies
    // that bytes accounting uses len * size_of::<T>() and that the
    // typed view via upgrade_device_ptr::<u32> behaves correctly.
    let Some((manager, runtime, sink)) = build_stack() else {
        return;
    };

    let len = 256usize;
    let bytes = len * std::mem::size_of::<u32>();
    let slice = manager.alloc::<u32>(len).expect("alloc<u32> via runtime");
    assert_eq!(slice.len(), len);
    assert_eq!(manager.allocated_bytes(), bytes as u64);
    assert_eq!(runtime.bytes_outstanding(), bytes);

    let recs = sink.snapshot();
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].bytes, Some(bytes));
    assert_eq!(recs[0].result, LogResult::Ok);

    drop(slice);
    runtime.reap_pending().expect("reap");
    assert_eq!(manager.allocated_bytes(), 0);
    assert_eq!(runtime.bytes_outstanding(), 0);
}

#[test]
fn into_bytes_preserves_runtime_backing() {
    // A runtime-routed alloc::<u32>(N) converted into a u8 view via
    // into_bytes must remain runtime-routed: drop should free
    // through the runtime, not through cudarc, and counters must
    // return to zero after reap.
    let Some((manager, runtime, _sink)) = build_stack() else {
        return;
    };

    let len = 128usize;
    let bytes = len * std::mem::size_of::<u32>();
    let typed = manager.alloc::<u32>(len).expect("alloc<u32> via runtime");
    assert_eq!(runtime.bytes_outstanding(), bytes);

    let as_bytes = typed.into_bytes();
    // Bytes accounting unchanged: into_bytes is a reinterpretation,
    // not a new allocation.
    assert_eq!(manager.allocated_bytes(), bytes as u64);
    assert_eq!(runtime.bytes_outstanding(), bytes);

    drop(as_bytes);
    runtime.reap_pending().expect("reap");
    assert_eq!(manager.allocated_bytes(), 0);
    assert_eq!(
        runtime.bytes_outstanding(),
        0,
        "into_bytes must preserve Backing::Runtime so the runtime frees on drop"
    );
}

#[test]
fn legacy_alloc_path_unchanged_when_no_runtime_attached() {
    // Manager constructed via legacy `new` must not touch any
    // runtime: build a runtime as a side channel and verify its
    // sink stays empty and its bytes_outstanding stays at zero
    // throughout an alloc<T>/drop cycle on the legacy manager.
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };

    // Side-channel runtime — same device, never attached to the
    // legacy manager.
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let side_sink: Arc<InMemorySink> = Arc::new(InMemorySink::new());
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        side_sink.clone() as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 1024 * 1024));
    let side_runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));

    // Legacy manager: NO runtime attached. MemoryBudget::default()
    // would give device_bytes = 0; set an explicit budget so the
    // legacy path's TOCTOU-safe reservation actually succeeds.
    let mut legacy_budget = MemoryBudget::default();
    legacy_budget.device_bytes = 64 * 1024;
    let manager = Arc::new(GpuMemoryManager::new(Arc::clone(&device), legacy_budget));
    assert!(manager.runtime().is_none());

    // Allocate, mutate counter, drop. Side-channel runtime must
    // be entirely undisturbed.
    let len = 512usize;
    let bytes = (len * std::mem::size_of::<u32>()) as u64;
    let slice = manager
        .alloc::<u32>(len)
        .expect("legacy alloc::<u32> still works");
    assert_eq!(slice.len(), len);
    assert_eq!(manager.allocated_bytes(), bytes);
    assert_eq!(
        side_runtime.bytes_outstanding(),
        0,
        "side-channel runtime must not see legacy allocations"
    );
    assert_eq!(
        side_sink.len(),
        0,
        "side-channel sink must remain empty for legacy allocations"
    );

    drop(slice);
    // Legacy drop frees through cudarc and decrements the local
    // counter; runtime side channel still untouched.
    assert_eq!(manager.allocated_bytes(), 0);
    assert_eq!(side_runtime.bytes_outstanding(), 0);
    assert_eq!(side_sink.len(), 0);
}

#[test]
fn alloc_raw_without_runtime_returns_kernel_error() {
    // Manager constructed via legacy `new` — no runtime attached.
    // alloc_raw must surface a clear error rather than silently
    // falling back.
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let manager = Arc::new(GpuMemoryManager::new(device, MemoryBudget::default()));
    assert!(manager.runtime().is_none());

    let err = manager.alloc_raw(64, AllocTag::UNTAGGED);
    assert!(
        matches!(err, Err(xlog_core::XlogError::Kernel(_))),
        "expected XlogError::Kernel, got {:?}",
        err.as_ref().map(|b| b.bytes())
    );
}
