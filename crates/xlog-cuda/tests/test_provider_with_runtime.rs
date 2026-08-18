// crates/xlog-cuda/tests/test_provider_with_runtime.rs
//! Provider-level test: when a `CudaKernelProvider` is constructed
//! via [`CudaKernelProvider::with_runtime`] using a
//! [`GpuMemoryManager`] built via
//! [`GpuMemoryManager::with_runtime`], the provider's normal
//! allocation entry points (`memory.alloc::<T>`) flow through the
//! v0.6 runtime stack: `GlobalDeviceBudget` accepts/rejects,
//! `LoggingResource` records, `AsyncCudaResource` queues async
//! frees that the runtime drains via `reap_pending`.
//!
//! Scope:
//!   * Drives a real provider call (`create_buffer_from_slice`)
//!     that internally goes through `self.memory.alloc::<u8>(..)`.
//!   * Asserts the sink saw the alloc record with the expected byte
//!     count and the runtime's `bytes_outstanding` reflects it.
//!   * Drops the buffer; manager counter releases immediately;
//!     runtime `reap_pending` drains; final state is zero.
//!   * Verifies `with_runtime` rejects a manager built via legacy
//!     `new` so the opt-in path can't be misused.
//!
//! Out of scope: A3 parallel stress, join prototype rebase,
//! flipping `CudaKernelProvider::new` default. The legacy
//! production constructor remains the default.

use std::sync::Arc;

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, DirectCudaResource, GlobalDeviceBudget, InMemorySink,
    LogAction, LogResult, LoggingResource, LoggingSink, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::{dedup_kernels, CudaDevice, CudaKernelProvider, GpuMemoryManager, DEDUP_MODULE};

const RUNTIME_LIMIT: usize = 256 * 1024;
const LOCAL_BUDGET: u64 = 1024 * 1024;

type RuntimeProviderFixture = (
    CudaKernelProvider,
    Arc<GpuMemoryManager>,
    Arc<XlogDeviceRuntime>,
    Arc<InMemorySink>,
);

fn async_runtime_for(
    device: &Arc<CudaDevice>,
    device_ordinal: u32,
    limit: usize,
) -> Arc<XlogDeviceRuntime> {
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(device)));
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(device), device_ordinal, Arc::clone(&pool)),
    );
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(async_resource, limit));
    Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(device),
        device_ordinal,
        pool,
        budget,
    ))
}

fn build_runtime_provider() -> Option<RuntimeProviderFixture> {
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
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        local_budget,
        Arc::clone(&runtime),
    ));

    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime must succeed when manager has runtime attached");

    Some((provider, memory, runtime, sink))
}

#[test]
fn provider_create_buffer_from_slice_routes_through_runtime() {
    let Some((provider, memory, runtime, sink)) = build_runtime_provider() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Sanity: the manager really has a runtime, and the provider's
    // memory accessor returns the same instance.
    assert!(memory.runtime().is_some());
    assert!(Arc::ptr_eq(provider.memory(), &memory));

    let baseline_runtime = runtime.bytes_outstanding();
    let baseline_local = memory.allocated_bytes();
    let baseline_records = sink.len();

    // Real provider entry: this calls self.memory.alloc::<u8>(bytes.len())
    // internally for the column buffer, plus at least one
    // alloc::<u32>(1) for the device-resident row-count scalar
    // (see provider/transfer.rs::create_buffer_from_slice and
    // upload_device_row_count). The exact number of internal
    // allocations is an implementation detail; the test pins the
    // *invariants* the runtime stack must satisfy:
    //   - Local manager and runtime counters agree (both mirror
    //     the same set of allocations through the same paths).
    //   - The sink saw at least one Allocate record reflecting
    //     the column byte count.
    //   - Drop + reap return both counters to baseline and produce
    //     matching Deallocate records.
    let col0: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let column_bytes = col0.len() * std::mem::size_of::<u32>();
    let schema = Schema::new(vec![("col0".to_string(), ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_slice::<u32>(&col0, schema)
        .expect("create_buffer_from_slice succeeds under a generous budget");

    // Both counters reflect the same total post-call. The runtime
    // and the manager mirror each other for runtime-routed
    // allocations.
    let post_local = memory.allocated_bytes();
    let post_runtime = runtime.bytes_outstanding();
    assert!(
        post_local > baseline_local,
        "expected manager counter to increase, got {} -> {}",
        baseline_local,
        post_local
    );
    assert_eq!(
        post_runtime - baseline_runtime,
        (post_local - baseline_local) as usize,
        "runtime counter delta must match manager counter delta"
    );
    assert!(
        post_local - baseline_local >= column_bytes as u64,
        "post-call bytes ({}) must include at least the column ({} bytes)",
        post_local - baseline_local,
        column_bytes
    );

    // Sink saw new allocations, all Ok, and at least one whose
    // byte count matches the column size.
    let recs_after_alloc = sink.snapshot();
    let new_records: Vec<_> = recs_after_alloc.iter().skip(baseline_records).collect();
    assert!(
        !new_records.is_empty(),
        "expected at least one new sink record from the provider allocation"
    );
    assert!(
        new_records.iter().all(|r| r.result == LogResult::Ok),
        "every new alloc record must be Ok, got {:?}",
        new_records
    );
    assert!(
        new_records
            .iter()
            .any(|r| r.action == LogAction::Allocate && r.bytes == Some(column_bytes)),
        "expected an Allocate record for the column byte size {}, got {:?}",
        column_bytes,
        new_records
    );

    // Drop the buffer: TrackedCudaSlice<u8>/<u32> Backing::Runtime
    // branch fires for every column. Manager counter releases
    // immediately; runtime holds bytes pending until reap_pending.
    drop(buffer);
    assert_eq!(memory.allocated_bytes(), baseline_local);
    assert_eq!(
        runtime.bytes_outstanding(),
        post_runtime,
        "async backend: runtime holds the freed bytes pending until reap"
    );

    runtime.reap_pending().expect("reap");
    assert_eq!(runtime.bytes_outstanding(), baseline_runtime);

    // After drop + reap the sink should contain a Deallocate
    // record matching the column byte count (and one or more for
    // any auxiliary allocations the provider made — we don't pin
    // those individually).
    let recs_final = sink.snapshot();
    assert!(
        recs_final
            .iter()
            .any(|r| r.action == LogAction::Deallocate && r.bytes == Some(column_bytes)),
        "expected a Deallocate record matching {} bytes, got {:?}",
        column_bytes,
        recs_final
    );
}

#[test]
fn provider_with_runtime_rejects_manager_without_runtime() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    // Legacy manager — no runtime attached.
    let mut budget = MemoryBudget::default();
    budget.device_bytes = 1024 * 1024;
    let memory = Arc::new(GpuMemoryManager::new(Arc::clone(&device), budget));
    assert!(memory.runtime().is_none());

    let err = CudaKernelProvider::with_runtime(device, memory);
    match err {
        Err(xlog_core::XlogError::Kernel(msg)) => {
            assert!(
                msg.contains("requires a GpuMemoryManager built via")
                    || msg.contains("with_runtime"),
                "unexpected error message: {}",
                msg
            );
        }
        other => panic!(
            "expected XlogError::Kernel from with_runtime on a manager without runtime, got {:?}",
            other.is_err()
        ),
    }
}

#[test]
fn runtime_overlay_shares_budget_accounting_in_both_allocation_orders() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let parent = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(1024),
    ));
    let runtime = async_runtime_for(&device, 0, 1024);
    let overlay = parent
        .with_runtime_overlay(Arc::clone(&runtime))
        .expect("same-device async runtime must produce an overlay");

    assert!(Arc::ptr_eq(parent.device(), overlay.device()));
    assert!(Arc::ptr_eq(
        overlay.runtime().expect("overlay runtime"),
        &runtime
    ));
    assert_eq!(parent.budget_limit_bytes(), 1024);
    assert_eq!(overlay.budget_limit_bytes(), 1024);

    let parent_first = parent.alloc::<u8>(128).expect("parent allocation");
    let overlay_second = overlay.alloc::<u8>(256).expect("overlay allocation");
    for manager in [&parent, &overlay] {
        assert_eq!(manager.allocated_bytes(), 384);
        assert_eq!(manager.remaining_bytes(), 640);
        assert_eq!(manager.peak_bytes(), 384);
        assert_eq!(manager.alloc_count(), 2);
    }
    assert!(
        overlay.check_budget(641).is_err(),
        "the overlay must include parent allocations in admission checks"
    );

    drop(parent_first);
    assert_eq!(parent.allocated_bytes(), 256);
    assert_eq!(overlay.allocated_bytes(), 256);
    drop(overlay_second);
    runtime.reap_pending().expect("reap first allocation order");
    assert_eq!(parent.allocated_bytes(), 0);
    assert_eq!(overlay.remaining_bytes(), 1024);

    parent.reset_peak();
    parent.reset_alloc_count();
    let overlay_first = overlay.alloc::<u8>(192).expect("overlay allocation");
    let parent_second = parent.alloc::<u8>(64).expect("parent allocation");
    for manager in [&parent, &overlay] {
        assert_eq!(manager.allocated_bytes(), 256);
        assert_eq!(manager.remaining_bytes(), 768);
        assert_eq!(manager.peak_bytes(), 256);
        assert_eq!(manager.alloc_count(), 2);
    }

    drop(overlay_first);
    assert_eq!(parent.allocated_bytes(), 64);
    assert_eq!(overlay.allocated_bytes(), 64);
    drop(parent_second);
    runtime
        .reap_pending()
        .expect("reap second allocation order");
    assert_eq!(parent.allocated_bytes(), 0);
    assert_eq!(overlay.remaining_bytes(), 1024);
    assert_eq!(parent.peak_bytes(), 256);
    assert_eq!(overlay.alloc_count(), 2);
}

#[test]
fn runtime_overlay_rejects_identity_ordinal_and_tracking_mismatches_without_mutation() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let parent = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(1024),
    ));
    let before = (
        parent.allocated_bytes(),
        parent.remaining_bytes(),
        parent.peak_bytes(),
        parent.alloc_count(),
    );

    let Some(other_device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let other_runtime = async_runtime_for(&other_device, 0, 1024);
    assert!(parent.with_runtime_overlay(other_runtime).is_err());

    let wrong_ordinal_runtime = async_runtime_for(&device, 1, 1024);
    assert!(parent.with_runtime_overlay(wrong_ordinal_runtime).is_err());

    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let direct: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
    let direct_runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        pool,
        direct,
    ));
    assert!(!direct_runtime.supports_block_use_tracking());
    assert!(parent.with_runtime_overlay(direct_runtime).is_err());

    assert_eq!(
        (
            parent.allocated_bytes(),
            parent.remaining_bytes(),
            parent.peak_bytes(),
            parent.alloc_count(),
        ),
        before,
        "validation failures must not mutate the parent ledger"
    );
    assert!(parent.runtime().is_none());
}

#[test]
fn runtime_memory_view_preserves_loaded_module_handles_and_runs_kernels() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        return;
    };
    let parent_memory = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(16 * 1024 * 1024),
    ));
    let original = CudaKernelProvider::new(Arc::clone(&device), Arc::clone(&parent_memory))
        .expect("load provider modules once");
    let handle_before = format!(
        "{:?}",
        device
            .inner()
            .get_func(DEDUP_MODULE, dedup_kernels::MARK_DUPLICATES)
            .expect("dedup handle before view")
    );
    assert!(original
        .with_runtime_memory_view(Arc::clone(&parent_memory))
        .is_err());

    let wrong_ordinal_runtime = async_runtime_for(&device, 1, 16 * 1024 * 1024);
    let wrong_ordinal_memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(16 * 1024 * 1024),
        wrong_ordinal_runtime,
    ));
    assert!(original
        .with_runtime_memory_view(wrong_ordinal_memory)
        .is_err());

    let other_device = Arc::new(CudaDevice::new(0).expect("second CUDA context"));
    let other_runtime = async_runtime_for(&other_device, 0, 16 * 1024 * 1024);
    let other_memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&other_device),
        MemoryBudget::with_limit(16 * 1024 * 1024),
        other_runtime,
    ));
    assert!(original.with_runtime_memory_view(other_memory).is_err());

    let runtime = async_runtime_for(&device, 0, 16 * 1024 * 1024);
    let overlay = parent_memory
        .with_runtime_overlay(runtime)
        .expect("same-device overlay");
    let view = original
        .with_runtime_memory_view(Arc::clone(&overlay))
        .expect("no-reload runtime memory view");

    assert!(Arc::ptr_eq(original.device(), view.device()));
    assert!(Arc::ptr_eq(view.memory(), &overlay));
    let handle_after = format!(
        "{:?}",
        device
            .inner()
            .get_func(DEDUP_MODULE, dedup_kernels::MARK_DUPLICATES)
            .expect("dedup handle after view")
    );
    assert_eq!(
        handle_after, handle_before,
        "creating the view must not replace the already-loaded CUDA module"
    );

    let schema = Schema::new(vec![("value".to_string(), ScalarType::U32)]);
    let input = [3u32, 1, 1, 2];
    for provider in [&original, &view] {
        let buffer = provider
            .create_buffer_from_slice(&input, schema.clone())
            .expect("upload input");
        let deduped = provider.dedup_full_row(&buffer).expect("run dedup kernel");
        let mut values = provider
            .download_column::<u32>(&deduped, 0)
            .expect("download deduplicated values");
        values.sort_unstable();
        assert_eq!(values, vec![1, 2, 3]);
    }
}
