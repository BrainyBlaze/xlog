// crates/xlog-cuda/tests/test_budget_via_runtime.rs
//! Integration test for [`GlobalDeviceBudget`] composed with
//! [`LoggingResource`] and [`AsyncCudaResource`] through
//! [`XlogDeviceRuntime::with_resource`].
//!
//! This is the production-recommended stack:
//!
//!   LoggingResource(InMemorySink)
//!     -> GlobalDeviceBudget
//!         -> AsyncCudaResource
//!
//! It exercises:
//!
//!   * Successful alloc/dealloc/reap through the full stack.
//!   * Budget enforcement reports exact current, requested, remaining, and
//!     limit values without calling the underlying allocator.
//!   * Async pending-free behavior end-to-end: dealloc keeps the
//!     budget reserved until reap_pending drains.
//!   * The outer logger records admitted operations and typed `OutOfBudget`
//!     rejections exactly once.
//!
//! Skips when CUDA is unavailable.

use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::device_runtime::{
    AllocTag, InMemorySink, LogAction, LogResult, LoggingSink, ResourceError, StreamId, StreamPool,
    XlogDeviceRuntime,
};
use xlog_cuda::CudaProviderBuilder;

const LIMIT: usize = 16 * 1024;

fn build_runtime() -> Option<(Arc<XlogDeviceRuntime>, Arc<InMemorySink>, Arc<StreamPool>)> {
    let sink: Arc<InMemorySink> = Arc::new(InMemorySink::new());
    let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(LIMIT as u64))
        .with_logging_sink(sink.clone() as Arc<dyn LoggingSink>)
        .build()
        .ok()?;
    let runtime = Arc::clone(provider.memory().runtime()?);
    let pool = Arc::clone(runtime.stream_pool());
    Some((runtime, sink, pool))
}

#[test]
fn budget_logging_async_stack_full_lifecycle() {
    let Some((runtime, sink, _pool)) = build_runtime() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Alloc 4 KiB → reserved 4 KiB, remaining 12 KiB.
    let block = runtime
        .allocate(4096, StreamId::DEFAULT, AllocTag("budget-rt-A"))
        .expect("alloc within budget");
    assert_eq!(runtime.bytes_outstanding(), 4096);

    runtime.deallocate(block).expect("dealloc");
    // Async inner: dealloc queues cuMemFreeAsync; budget held.
    assert_eq!(
        runtime.bytes_outstanding(),
        4096,
        "budget+async: bytes_outstanding still reports pending until reap"
    );

    runtime.reap_pending().expect("reap");
    assert_eq!(runtime.bytes_outstanding(), 0);

    let recs = sink.snapshot();
    assert_eq!(recs.len(), 3, "expected 3 records, got {:?}", recs);
    assert_eq!(recs[0].action, LogAction::Allocate);
    assert_eq!(recs[0].result, LogResult::Ok);
    assert_eq!(recs[0].bytes, Some(4096));
    assert_eq!(recs[1].action, LogAction::Deallocate);
    assert_eq!(recs[1].result, LogResult::Ok);
    assert_eq!(recs[2].action, LogAction::ReapPending);
    assert_eq!(recs[2].result, LogResult::Ok);
}

#[test]
fn budget_rejects_over_limit_before_allocator_and_logs_typed_result() {
    let Some((runtime, sink, _pool)) = build_runtime() else {
        return;
    };

    let err = runtime.allocate(LIMIT + 1, StreamId::DEFAULT, AllocTag("budget-rt-too-big"));
    assert!(
        matches!(
            err,
            Err(ResourceError::OutOfBudget {
                requested,
                current: 0,
                remaining: LIMIT,
                limit: LIMIT,
            }) if requested == LIMIT + 1
        ),
        "expected OutOfBudget {{LIMIT+1, LIMIT}}, got {:?}",
        err
    );

    let records = sink.snapshot();
    assert_eq!(records.len(), 1, "the rejected request must be logged once");
    let record = &records[0];
    assert_eq!(record.action, LogAction::Allocate);
    assert_eq!(record.bytes, Some(LIMIT + 1));
    assert_eq!(record.tag, Some(AllocTag("budget-rt-too-big")));
    assert!(record.ptr.is_none());
    assert!(record.generation.is_none());
    assert!(matches!(
        record.result,
        LogResult::Err {
            kind: "OutOfBudget",
            ..
        }
    ));
    assert_eq!(runtime.bytes_outstanding(), 0);
}

#[test]
fn budget_rejection_after_partial_use_reports_correct_remaining() {
    let Some((runtime, _sink, _pool)) = build_runtime() else {
        return;
    };

    let block = runtime
        .allocate(LIMIT - 4096, StreamId::DEFAULT, AllocTag("budget-rt-fill"))
        .expect("alloc fills most of budget");
    // Remaining = 4096; ask for 8192.
    let err = runtime.allocate(8192, StreamId::DEFAULT, AllocTag("budget-rt-overflow"));
    assert!(
        matches!(
            err,
            Err(ResourceError::OutOfBudget {
                requested: 8192,
                current,
                remaining: 4096,
                limit: LIMIT,
            }) if current == LIMIT - 4096
        ),
        "expected OutOfBudget {{8192, 4096}}, got {:?}",
        err
    );

    runtime.deallocate(block).expect("dealloc");
    runtime.reap_pending().expect("reap");
    assert_eq!(runtime.bytes_outstanding(), 0);
}
