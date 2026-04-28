// crates/xlog-cuda/tests/test_budget_via_runtime.rs
//! Integration test for [`GlobalDeviceBudget`] composed with
//! [`LoggingResource`] and [`AsyncCudaResource`] through
//! [`XlogDeviceRuntime::with_resource`].
//!
//! This is the production-recommended stack:
//!
//!   GlobalDeviceBudget
//!     -> LoggingResource(InMemorySink)
//!         -> AsyncCudaResource
//!
//! It exercises:
//!
//!   * Successful alloc/dealloc/reap through the full stack.
//!   * Budget enforcement at the outermost decorator: an over-limit
//!     allocation returns `OutOfBudget { requested, remaining }`
//!     without ever calling into the logger or the underlying
//!     resource.
//!   * Async pending-free behavior end-to-end: dealloc keeps the
//!     budget reserved until reap_pending drains.
//!   * Logging records reflect the budget's view: an `OutOfBudget`
//!     rejection at the top of the stack does not propagate down
//!     to the inner, so the logger does not see those calls. (In
//!     this stack ordering, the logger sits *inside* the budget so
//!     it only records work that actually reached it.)
//!
//! Skips when CUDA is unavailable.

use std::sync::Arc;

use xlog_cuda::device_runtime::{
    AllocTag, AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, InMemorySink, LogAction,
    LogResult, LoggingResource, LoggingSink, ResourceError, StreamId, StreamPool,
    XlogDeviceRuntime,
};
use xlog_cuda::CudaDevice;

const LIMIT: usize = 16 * 1024;

fn build_runtime() -> Option<(XlogDeviceRuntime, Arc<InMemorySink>, Arc<StreamPool>)> {
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
        Box::new(GlobalDeviceBudget::new(logging, LIMIT));
    let runtime =
        XlogDeviceRuntime::with_resource(Arc::clone(&device), 0, Arc::clone(&pool), budget);
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
fn budget_rejects_over_limit_without_calling_inner() {
    let Some((runtime, sink, _pool)) = build_runtime() else {
        return;
    };

    let err = runtime.allocate(LIMIT + 1, StreamId::DEFAULT, AllocTag("budget-rt-too-big"));
    assert!(
        matches!(
            err,
            Err(ResourceError::OutOfBudget {
                requested,
                remaining: LIMIT
            }) if requested == LIMIT + 1
        ),
        "expected OutOfBudget {{LIMIT+1, LIMIT}}, got {:?}",
        err
    );

    // The logger sits *inside* the budget, so an OutOfBudget
    // rejection at the top short-circuits before reaching the
    // logger — there must be zero records.
    assert_eq!(
        sink.len(),
        0,
        "OutOfBudget at the top of the stack must not reach the inner logger"
    );
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
                remaining: 4096
            })
        ),
        "expected OutOfBudget {{8192, 4096}}, got {:?}",
        err
    );

    runtime.deallocate(block).expect("dealloc");
    runtime.reap_pending().expect("reap");
    assert_eq!(runtime.bytes_outstanding(), 0);
}
