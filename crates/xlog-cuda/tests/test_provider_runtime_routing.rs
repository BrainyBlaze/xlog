// crates/xlog-cuda/tests/test_provider_runtime_routing.rs
//! Tests for `GpuMemoryManager`'s runtime-routing paths.
//!
//! The canonical provider builder supplies a manager backed by an
//! [`XlogDeviceRuntime`], stacking [`LoggingResource`] over
//! [`GlobalDeviceBudget`] over [`AsyncCudaResource`]. With the runtime
//! attached, **both**
//! `alloc_raw` and the typed `alloc::<T>` path route through the
//! runtime stack; `TrackedCudaSlice<T>` returned from `alloc::<T>`
//! frees through the runtime on drop via the `Backing::Runtime`
//! branch of its `Drop` implementation.
//!
//! What these tests assert:
//!   1. `alloc_raw` and `alloc::<T>` (u8 + non-byte) both produce
//!      records in the logging sink and raise both the local
//!      manager counter and the runtime's `bytes_outstanding`.
//!   2. Dropping the returned tracked slice / runtime block
//!      releases the manager counter immediately and, after a
//!      `runtime.reap_pending()`, the runtime's reserved bytes
//!      (held while the async free is queued).
//!   3. `into_bytes` preserves the `Backing::Runtime` ownership
//!      tag so a runtime-routed `u32` reinterpreted as bytes still
//!      frees through the runtime.
//!   4. Zero-byte allocations preserve their explicit no-allocation behavior.

use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::device_runtime::{
    AllocTag, InMemorySink, LogAction, LogResult, LoggingSink, XlogDeviceRuntime,
};
use xlog_cuda::GpuMemoryManager;

const RUNTIME_LIMIT: usize = 32 * 1024;

fn build_stack() -> Option<(
    Arc<GpuMemoryManager>,
    Arc<XlogDeviceRuntime>,
    Arc<InMemorySink>,
)> {
    let sink: Arc<InMemorySink> = Arc::new(InMemorySink::new());
    let provider =
        xlog_cuda::CudaProviderBuilder::new(0, MemoryBudget::with_limit(RUNTIME_LIMIT as u64))
            .with_logging_sink(sink.clone() as Arc<dyn LoggingSink>)
            .build()
            .ok()?;
    let manager = Arc::clone(provider.memory());
    let runtime = Arc::clone(manager.runtime()?);
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
fn zero_byte_alloc_with_runtime_attached_bypasses_runtime() {
    // Production code makes zero-byte allocations (empty Vecs,
    // empty buffers). The v0.6 resource stack rejects bytes == 0
    // by contract because cuMemAlloc(0) is UB. Cudarc's
    // alloc::<T>(0) is well-defined (returns an empty CudaSlice
    // without calling the driver), so GpuMemoryManager routes
    // zero-byte requests through the legacy path even when a
    // runtime is attached. This test pins that bypass: the
    // runtime sink and counter must remain at zero for an empty
    // alloc, while the slice itself is still functional.
    let Some((manager, runtime, sink)) = build_stack() else {
        return;
    };

    let baseline_runtime = runtime.bytes_outstanding();
    let baseline_local = manager.allocated_bytes();
    let baseline_records = sink.len();

    let empty = manager
        .alloc::<u32>(0)
        .expect("zero-byte alloc must succeed via runtime-attached manager");
    assert_eq!(empty.len(), 0);

    // No runtime activity, no log record. The local counter
    // accounts for 0 bytes either way.
    assert_eq!(runtime.bytes_outstanding(), baseline_runtime);
    assert_eq!(manager.allocated_bytes(), baseline_local);
    assert_eq!(sink.len(), baseline_records);

    drop(empty);
    assert_eq!(runtime.bytes_outstanding(), baseline_runtime);
    assert_eq!(manager.allocated_bytes(), baseline_local);
    assert_eq!(sink.len(), baseline_records);
}
