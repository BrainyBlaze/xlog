// crates/xlog-cuda/tests/test_logging_via_runtime.rs
//! Integration test for [`LoggingResource`] composed under
//! [`XlogDeviceRuntime::with_resource`].
//!
//! Builds a runtime whose active resource is
//! `LoggingResource<AsyncCudaResource>`, drives a small sequence of
//! alloc / dealloc / reap_pending calls through the runtime facade,
//! and asserts the in-memory sink captured the expected sequence of
//! records with the right field shape.
//!
//! The point of this test is to prove the decorator survives the
//! composition through `with_resource` (i.e., the runtime's
//! `Box<dyn DeviceMemoryResource + Send + Sync>` mutex does not
//! reorder, eat, or duplicate emitted records) and that the emitted
//! records reflect what the *runtime* did, not what the underlying
//! `AsyncCudaResource` would have logged on its own.
//!
//! Skips when no CUDA device is available or `StreamPool::acquire`
//! cannot fork a non-default stream.

use std::sync::Arc;

use xlog_cuda::device_runtime::{
    AllocTag, AsyncCudaResource, DeviceMemoryResource, InMemorySink, LogAction, LogResult,
    LoggingResource, LoggingSink, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::CudaDevice;

#[test]
fn logging_resource_composed_through_runtime_records_full_lifecycle() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));

    let stream_id = match pool.acquire() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Skipping: StreamPool::acquire failed: {}", e);
            return;
        }
    };
    assert_ne!(stream_id, StreamId::DEFAULT);

    let sink: Arc<InMemorySink> = Arc::new(InMemorySink::new());
    let inner: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(AsyncCudaResource::new(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
    ));
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        inner,
        sink.clone() as Arc<dyn LoggingSink>,
    ));

    let runtime =
        XlogDeviceRuntime::with_resource(Arc::clone(&device), 0, Arc::clone(&pool), logging);

    // Sequence: alloc(A) → alloc(B) → dealloc(A) → dealloc(B) → reap.
    let block_a = runtime
        .allocate(2048, stream_id, AllocTag("logging-rt-A"))
        .expect("alloc A");
    let block_b = runtime
        .allocate(1024, StreamId::DEFAULT, AllocTag("logging-rt-B"))
        .expect("alloc B");

    let a_ptr = block_a.ptr;
    let a_gen = block_a.generation;
    let b_ptr = block_b.ptr;
    let b_gen = block_b.generation;

    runtime.deallocate(block_a).expect("dealloc A");
    runtime.deallocate(block_b).expect("dealloc B");
    runtime.reap_pending().expect("reap");

    // Final allocator state is correct (defense-in-depth — the
    // primary purpose of this test is the records below).
    assert_eq!(runtime.bytes_outstanding(), 0);

    let recs = sink.snapshot();
    assert_eq!(
        recs.len(),
        5,
        "expected 5 records (2 alloc + 2 dealloc + 1 reap), got {}: {:?}",
        recs.len(),
        recs
    );

    // Strict ordering by emission counter.
    let mut last = 0u64;
    for rec in &recs {
        assert!(rec.order_counter > last);
        last = rec.order_counter;
    }

    // Record 0: alloc A on non-default stream.
    assert_eq!(recs[0].action, LogAction::Allocate);
    assert_eq!(recs[0].result, LogResult::Ok);
    assert_eq!(recs[0].stream_id, Some(stream_id));
    assert_eq!(recs[0].bytes, Some(2048));
    assert_eq!(recs[0].ptr, Some(a_ptr));
    assert_eq!(recs[0].generation, Some(a_gen));
    assert_eq!(recs[0].tag, Some(AllocTag("logging-rt-A")));

    // Record 1: alloc B on default stream.
    assert_eq!(recs[1].action, LogAction::Allocate);
    assert_eq!(recs[1].result, LogResult::Ok);
    assert_eq!(recs[1].stream_id, Some(StreamId::DEFAULT));
    assert_eq!(recs[1].bytes, Some(1024));
    assert_eq!(recs[1].ptr, Some(b_ptr));
    assert_eq!(recs[1].generation, Some(b_gen));

    // Record 2: dealloc A.
    assert_eq!(recs[2].action, LogAction::Deallocate);
    assert_eq!(recs[2].result, LogResult::Ok);
    assert_eq!(recs[2].ptr, Some(a_ptr));
    assert_eq!(recs[2].generation, Some(a_gen));
    assert_eq!(recs[2].stream_id, Some(stream_id));

    // Record 3: dealloc B.
    assert_eq!(recs[3].action, LogAction::Deallocate);
    assert_eq!(recs[3].result, LogResult::Ok);
    assert_eq!(recs[3].ptr, Some(b_ptr));
    assert_eq!(recs[3].generation, Some(b_gen));
    assert_eq!(recs[3].stream_id, Some(StreamId::DEFAULT));

    // Record 4: reap_pending.
    assert_eq!(recs[4].action, LogAction::ReapPending);
    assert_eq!(recs[4].result, LogResult::Ok);
    assert!(recs[4].stream_id.is_none());
    assert!(recs[4].ptr.is_none());
    assert!(recs[4].bytes.is_none());

    // device_ordinal is propagated on every record.
    for rec in &recs {
        assert_eq!(rec.device_ordinal, 0);
    }
}
