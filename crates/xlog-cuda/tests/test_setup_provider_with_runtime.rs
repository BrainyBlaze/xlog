// crates/xlog-cuda/tests/test_setup_provider_with_runtime.rs
//! Higher-level construction-path test for the v0.6 opt-in fixture.
//!
//! `tests/common/mod.rs::setup_provider` is the canonical higher-
//! level construction path used across xlog-cuda's integration
//! tests: it builds a `CudaKernelProvider` for "I need a working
//! provider" callers without exposing the device, memory manager,
//! or kernel-loading details. The opt-in v0.6 sibling
//! `setup_provider_with_runtime` produces the same provider shape
//! but composes the runtime stack
//! `GlobalDeviceBudget(LoggingResource(AsyncCudaResource))` and
//! constructs the provider via the `with_runtime` constructor.
//!
//! This test exercises the new fixture end-to-end: build the
//! provider via the runtime fixture, run a real provider operation
//! (`create_buffer_from_slice`), then assert the runtime stack
//! observed the allocation / deallocation through the same
//! manager `Arc` that the provider holds. Drop + reap returns the
//! runtime to baseline.
//!
//! Out of scope: A3 parallel stress, join prototype rebase, any
//! migration of external `setup_provider` call sites. The legacy
//! fixture remains the default for tests that do not need to
//! observe runtime routing.

mod common;

use common::setup_provider_with_runtime;
use xlog_core::{ScalarType, Schema};
use xlog_cuda::device_runtime::{LogAction, LogResult};

#[test]
fn setup_provider_with_runtime_routes_real_provider_alloc() {
    let Some(handles) = setup_provider_with_runtime() else {
        return;
    };

    // Sanity: the manager the fixture returned is the same Arc the
    // provider holds, and it has a runtime attached.
    assert!(handles.memory.runtime().is_some());
    assert!(std::sync::Arc::ptr_eq(
        handles.provider.memory(),
        &handles.memory
    ));

    let baseline_runtime = handles.runtime.bytes_outstanding();
    let baseline_local = handles.memory.allocated_bytes();
    let baseline_records = handles.sink.len();

    // Drive a real provider operation that internally calls
    // memory.alloc::<u8>(..) (and a small alloc::<u32>(1) for the
    // device row count). The exact allocation count is an
    // implementation detail; the routing invariant is what we pin.
    let col0: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let column_bytes = col0.len() * std::mem::size_of::<u32>();
    let schema = Schema::new(vec![("col0".to_string(), ScalarType::U32)]);
    let buffer = handles
        .provider
        .create_buffer_from_slice::<u32>(&col0, schema)
        .expect("create_buffer_from_slice succeeds under the fixture's generous budget");

    // Local + runtime counters mirror the same set of allocations.
    let post_local = handles.memory.allocated_bytes();
    let post_runtime = handles.runtime.bytes_outstanding();
    assert!(post_local > baseline_local);
    assert_eq!(
        post_runtime - baseline_runtime,
        (post_local - baseline_local) as usize,
        "runtime delta must match manager delta when routing through with_runtime"
    );
    assert!(
        post_local - baseline_local >= column_bytes as u64,
        "post-call total ({}) must include at least the column ({} bytes)",
        post_local - baseline_local,
        column_bytes,
    );

    // Sink saw new Allocate records, all Ok, and at least one for
    // the column byte size.
    let recs = handles.sink.snapshot();
    let new_records: Vec<_> = recs.iter().skip(baseline_records).collect();
    assert!(!new_records.is_empty());
    assert!(new_records.iter().all(|r| r.result == LogResult::Ok));
    assert!(
        new_records
            .iter()
            .any(|r| r.action == LogAction::Allocate && r.bytes == Some(column_bytes)),
        "expected an Allocate record for {} bytes, got {:?}",
        column_bytes,
        new_records
    );

    // Drop the buffer: TrackedCudaSlice<u8>/<u32> with
    // Backing::Runtime queues async frees through the runtime; the
    // manager counter releases immediately, the runtime holds
    // pending until reap.
    drop(buffer);
    assert_eq!(handles.memory.allocated_bytes(), baseline_local);
    assert_eq!(
        handles.runtime.bytes_outstanding(),
        post_runtime,
        "async backend: runtime holds bytes pending until reap"
    );

    handles.runtime.reap_pending().expect("reap");
    assert_eq!(handles.runtime.bytes_outstanding(), baseline_runtime);

    let recs_final = handles.sink.snapshot();
    assert!(
        recs_final
            .iter()
            .any(|r| r.action == LogAction::Deallocate && r.bytes == Some(column_bytes)),
        "expected a Deallocate record for {} bytes after drop + reap",
        column_bytes
    );
}
