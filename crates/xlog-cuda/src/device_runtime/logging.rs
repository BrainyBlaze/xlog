//! [`LoggingResource`] — telemetry decorator for any
//! [`DeviceMemoryResource`].
//!
//! Wraps an inner resource and records every `allocate`,
//! `deallocate`, and `reap_pending` call (success and failure) into
//! a [`LoggingSink`]. The decorator is fully transparent to allocator
//! semantics: it forwards every method, never alters the inner
//! result, and never panics or returns an error caused by the sink
//! itself. Logging failure is recorded into a per-resource diagnostic
//! counter (visible via [`LoggingResource::dropped_records`]) and
//! otherwise silenced — losing a log line must not corrupt
//! allocation correctness.
//!
//! Sink design
//! -----------
//! [`LoggingSink`] is a tiny trait with a single `emit` method that
//! returns `Result<(), SinkError>`. The default in-memory sink
//! ([`InMemorySink`]) buffers records in a `Mutex<Vec<LogRecord>>`
//! and is the test workhorse — it lets unit tests assert the exact
//! sequence of records without filesystem dependencies. A future
//! `CsvFileSink` or `RingBufferSink` slots in without touching the
//! decorator.
//!
//! Record contents
//! ---------------
//! Every emitted [`LogRecord`] carries:
//!   * `action` — Allocate / Deallocate / ReapPending
//!   * `device_ordinal`
//!   * `stream_id` — present for Allocate / Deallocate (the block's
//!     alloc_stream); absent for ReapPending which spans streams.
//!   * `ptr` / `bytes` / `tag` / `generation` — present when the
//!     operation has a corresponding [`DeviceBlock`] reference.
//!   * `thread_id` — `std::thread::current().id()` rendered as u64
//!     for portability.
//!   * `order_counter` — monotonic per-process u64. Strictly
//!     increasing across all sinks, all threads, all resources.
//!   * `timestamp_nanos` — `SystemTime::now()` since UNIX epoch in
//!     nanoseconds. Wall-clock; not monotonic. Use `order_counter`
//!     for ordering, `timestamp_nanos` for human-readable spans.
//!   * `result` — Ok or short error tag string.
//!
//! Decorator does NOT capture the inner resource's pre-call state
//! (e.g., bytes_outstanding before allocate) — that would require
//! locking the inner resource an extra time and is not part of the
//! v0.6 telemetry contract. Callers that need pre/post diffs read
//! `bytes_outstanding` themselves.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use super::resource::{
    AllocTag, DeviceBlock, DeviceMemoryResource, Generation, ResourceError, ResourceResult,
    StreamId,
};

/// Action recorded in a [`LogRecord`]. Distinct variants for the
/// three inner methods so consumers can filter without parsing the
/// result message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LogAction {
    Allocate,
    Deallocate,
    ReapPending,
}

impl fmt::Display for LogAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogAction::Allocate => f.write_str("allocate"),
            LogAction::Deallocate => f.write_str("deallocate"),
            LogAction::ReapPending => f.write_str("reap_pending"),
        }
    }
}

/// Result status as recorded in a log entry. Successful operations
/// are recorded as `Ok`; failed ones carry the error variant tag
/// plus an optional short message. We keep the message bounded so a
/// runaway driver-error string cannot blow up the sink buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LogResult {
    Ok,
    Err { kind: &'static str, message: String },
}

impl LogResult {
    /// Convert a [`ResourceResult`] outcome into a `LogResult`,
    /// truncating overlong messages to keep records bounded.
    pub fn from_result<T>(r: &ResourceResult<T>) -> Self {
        match r {
            Ok(_) => LogResult::Ok,
            Err(e) => LogResult::Err {
                kind: classify_error(e),
                message: truncate(format!("{}", e), 256),
            },
        }
    }
}

fn classify_error(e: &ResourceError) -> &'static str {
    match e {
        ResourceError::OutOfBudget { .. } => "OutOfBudget",
        ResourceError::Driver(_) => "Driver",
        ResourceError::StreamMisuse(_) => "StreamMisuse",
        ResourceError::UseAfterFree { .. } => "UseAfterFree",
        ResourceError::OutOfBounds { .. } => "OutOfBounds",
    }
}

fn truncate(mut s: String, cap: usize) -> String {
    if s.len() > cap {
        s.truncate(cap);
        s.push_str("…");
    }
    s
}

/// A single allocation-log entry. Values are owned/`Copy` so the
/// record is `Clone + Send + Sync` and trivially serializable.
#[derive(Clone, Debug)]
pub struct LogRecord {
    pub action: LogAction,
    pub device_ordinal: u32,
    pub stream_id: Option<StreamId>,
    pub ptr: Option<u64>,
    pub bytes: Option<usize>,
    pub tag: Option<AllocTag>,
    pub generation: Option<Generation>,
    pub thread_id: u64,
    pub order_counter: u64,
    pub timestamp_nanos: u128,
    pub result: LogResult,
}

/// Process-wide monotonic counter used for `LogRecord::order_counter`.
/// Cross-resource, cross-thread strict order — useful for
/// reconstructing the exact emission sequence in tests with multiple
/// resources composed.
static ORDER_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_order_counter() -> u64 {
    ORDER_COUNTER.fetch_add(1, Ordering::Relaxed)
}

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

fn current_thread_id_u64() -> u64 {
    // `ThreadId` does not expose its inner u64 on stable. Hash the
    // Debug representation instead — stable for the lifetime of the
    // thread, distinct across threads, and fits in u64.
    let s = format!("{:?}", std::thread::current().id());
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Sink-side error. Returned by [`LoggingSink::emit`] when the sink
/// cannot accept a record (full ring buffer, IO error, etc.). The
/// decorator catches this and increments
/// [`LoggingResource::dropped_records`]; it does **not** propagate
/// the error to the allocator caller.
#[derive(Debug)]
pub enum SinkError {
    /// The sink is closed or full and refused the record.
    Refused(String),
    /// IO-backed sinks may surface a wrapped IO error here.
    Io(String),
}

impl fmt::Display for SinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SinkError::Refused(m) => write!(f, "log sink refused record: {}", m),
            SinkError::Io(m) => write!(f, "log sink io error: {}", m),
        }
    }
}

impl std::error::Error for SinkError {}

/// Trait for log destinations. Implementations are required to be
/// thread-safe (`Send + Sync`); the decorator may emit from any
/// thread that calls into the resource.
pub trait LoggingSink: Send + Sync {
    /// Accept a record. Implementations must not panic on full /
    /// closed / IO-error conditions — return [`SinkError`] instead.
    fn emit(&self, record: LogRecord) -> Result<(), SinkError>;
}

/// In-memory sink for tests and lightweight use. Records are
/// appended to a `Vec` under a `Mutex`. `snapshot` returns a clone
/// so consumers can inspect without holding the mutex.
pub struct InMemorySink {
    records: Mutex<Vec<LogRecord>>,
}

impl InMemorySink {
    pub fn new() -> Self {
        Self {
            records: Mutex::new(Vec::new()),
        }
    }

    /// Clone the current record buffer.
    pub fn snapshot(&self) -> Vec<LogRecord> {
        self.records.lock().expect("InMemorySink poisoned").clone()
    }

    /// Drop all buffered records. Useful for tests that want a
    /// clean slate between phases.
    pub fn clear(&self) {
        self.records.lock().expect("InMemorySink poisoned").clear();
    }

    /// Number of records currently buffered.
    pub fn len(&self) -> usize {
        self.records.lock().expect("InMemorySink poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for InMemorySink {
    fn default() -> Self {
        Self::new()
    }
}

impl LoggingSink for InMemorySink {
    fn emit(&self, record: LogRecord) -> Result<(), SinkError> {
        self.records
            .lock()
            .expect("InMemorySink poisoned")
            .push(record);
        Ok(())
    }
}

/// Telemetry decorator for [`DeviceMemoryResource`].
pub struct LoggingResource {
    inner: Box<dyn DeviceMemoryResource + Send + Sync>,
    sink: std::sync::Arc<dyn LoggingSink>,
    /// Count of records the sink refused or errored on. Surfaced as
    /// a diagnostic for callers that want to detect telemetry loss
    /// without halting the data-plane.
    dropped_records: AtomicU64,
}

impl LoggingResource {
    /// Wrap `inner` with `sink`. Records are emitted on every public
    /// call; sink failures are counted in `dropped_records`.
    pub fn new(
        inner: Box<dyn DeviceMemoryResource + Send + Sync>,
        sink: std::sync::Arc<dyn LoggingSink>,
    ) -> Self {
        Self {
            inner,
            sink,
            dropped_records: AtomicU64::new(0),
        }
    }

    /// Total records the sink has refused. A nonzero value means
    /// telemetry was lost; allocator semantics are unaffected.
    pub fn dropped_records(&self) -> u64 {
        self.dropped_records.load(Ordering::Relaxed)
    }

    fn emit(&self, record: LogRecord) {
        if self.sink.emit(record).is_err() {
            self.dropped_records.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl DeviceMemoryResource for LoggingResource {
    fn allocate(
        &self,
        bytes: usize,
        stream: StreamId,
        tag: AllocTag,
    ) -> ResourceResult<DeviceBlock> {
        let result = self.inner.allocate(bytes, stream, tag);
        let (ptr, gen, recorded_bytes) = match &result {
            Ok(b) => (Some(b.ptr), Some(b.generation), Some(b.bytes)),
            Err(_) => (None, None, Some(bytes)),
        };
        self.emit(LogRecord {
            action: LogAction::Allocate,
            device_ordinal: self.inner.device_ordinal(),
            stream_id: Some(stream),
            ptr,
            bytes: recorded_bytes,
            tag: Some(tag),
            generation: gen,
            thread_id: current_thread_id_u64(),
            order_counter: next_order_counter(),
            timestamp_nanos: now_nanos(),
            result: LogResult::from_result(&result),
        });
        result
    }

    fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()> {
        // Capture identifying fields before the move into inner —
        // on success the block is consumed and we still need them
        // for the log record.
        let ptr = block.ptr;
        let bytes = block.bytes;
        let tag = block.tag;
        let gen = block.generation;
        let stream = block.alloc_stream;
        let dev = block.device_ordinal;

        let result = self.inner.deallocate(block);
        self.emit(LogRecord {
            action: LogAction::Deallocate,
            device_ordinal: dev,
            stream_id: Some(stream),
            ptr: Some(ptr),
            bytes: Some(bytes),
            tag: Some(tag),
            generation: Some(gen),
            thread_id: current_thread_id_u64(),
            order_counter: next_order_counter(),
            timestamp_nanos: now_nanos(),
            result: LogResult::from_result(&result),
        });
        result
    }

    fn device_ordinal(&self) -> u32 {
        self.inner.device_ordinal()
    }

    fn bytes_outstanding(&self) -> usize {
        self.inner.bytes_outstanding()
    }

    fn reap_pending(&self) -> ResourceResult<()> {
        let result = self.inner.reap_pending();
        self.emit(LogRecord {
            action: LogAction::ReapPending,
            device_ordinal: self.inner.device_ordinal(),
            stream_id: None,
            ptr: None,
            bytes: None,
            tag: None,
            generation: None,
            thread_id: current_thread_id_u64(),
            order_counter: next_order_counter(),
            timestamp_nanos: now_nanos(),
            result: LogResult::from_result(&result),
        });
        result
    }
}

#[cfg(test)]
mod tests {
    use super::super::direct::DirectCudaResource;
    use super::super::resource::BlockState;
    use super::*;
    use std::sync::Arc;

    use crate::CudaDevice;

    fn try_device() -> Option<Arc<CudaDevice>> {
        CudaDevice::new(0).ok().map(Arc::new)
    }

    #[test]
    fn pass_through_alloc_dealloc_emits_two_ok_records() {
        let Some(device) = try_device() else {
            return;
        };
        let inner = Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let sink = Arc::new(InMemorySink::new());
        let r = LoggingResource::new(inner, sink.clone());

        let block = r
            .allocate(1024, StreamId::DEFAULT, AllocTag("logging-test"))
            .expect("alloc");
        assert_eq!(block.bytes, 1024);
        assert_eq!(block.state, BlockState::Live);
        r.deallocate(block).expect("dealloc");

        let recs = sink.snapshot();
        assert_eq!(recs.len(), 2, "expected 2 records, got {:?}", recs);
        assert_eq!(recs[0].action, LogAction::Allocate);
        assert_eq!(recs[0].result, LogResult::Ok);
        assert_eq!(recs[0].bytes, Some(1024));
        assert_eq!(recs[0].stream_id, Some(StreamId::DEFAULT));
        assert!(recs[0].ptr.is_some());

        assert_eq!(recs[1].action, LogAction::Deallocate);
        assert_eq!(recs[1].result, LogResult::Ok);
        assert_eq!(recs[1].ptr, recs[0].ptr);
        assert_eq!(recs[1].generation, recs[0].generation);
    }

    #[test]
    fn order_counter_strictly_increases_across_records() {
        let Some(device) = try_device() else {
            return;
        };
        let inner = Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let sink = Arc::new(InMemorySink::new());
        let r = LoggingResource::new(inner, sink.clone());

        for _ in 0..4 {
            let b = r
                .allocate(64, StreamId::DEFAULT, AllocTag::UNTAGGED)
                .expect("alloc");
            r.deallocate(b).expect("dealloc");
        }
        r.reap_pending().expect("reap");

        let recs = sink.snapshot();
        assert_eq!(recs.len(), 9); // 4*alloc + 4*dealloc + 1 reap
        let mut last = 0u64;
        for rec in &recs {
            assert!(
                rec.order_counter > last,
                "order_counter must strictly increase: prev={}, now={}",
                last,
                rec.order_counter
            );
            last = rec.order_counter;
        }
    }

    #[test]
    fn failed_alloc_records_error_result() {
        let Some(device) = try_device() else {
            return;
        };
        let inner = Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let sink = Arc::new(InMemorySink::new());
        let r = LoggingResource::new(inner, sink.clone());

        // Zero-byte alloc fails per resource contract.
        let _ = r.allocate(0, StreamId::DEFAULT, AllocTag::UNTAGGED);
        let recs = sink.snapshot();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].action, LogAction::Allocate);
        assert!(matches!(recs[0].result, LogResult::Err { kind, .. } if kind == "Driver"));
        // Failed allocs still record the requested byte count.
        assert_eq!(recs[0].bytes, Some(0));
        assert!(recs[0].ptr.is_none());
        assert!(recs[0].generation.is_none());
    }

    #[test]
    fn failed_dealloc_records_error_result() {
        let Some(device) = try_device() else {
            return;
        };
        let inner = Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let sink = Arc::new(InMemorySink::new());
        let r = LoggingResource::new(inner, sink.clone());

        let bogus = DeviceBlock {
            ptr: 0xdead_beef,
            device_ordinal: 0,
            alloc_stream: StreamId::DEFAULT,
            bytes: 16,
            align: 1,
            tag: AllocTag::UNTAGGED,
            generation: Generation::next(),
            state: BlockState::Live,
        };
        let res = r.deallocate(bogus);
        assert!(res.is_err());
        let recs = sink.snapshot();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].action, LogAction::Deallocate);
        assert!(matches!(recs[0].result, LogResult::Err { kind, .. } if kind == "UseAfterFree"));
        assert_eq!(recs[0].ptr, Some(0xdead_beef));
    }

    #[test]
    fn reap_pending_emits_record() {
        let Some(device) = try_device() else {
            return;
        };
        let inner = Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let sink = Arc::new(InMemorySink::new());
        let r = LoggingResource::new(inner, sink.clone());

        r.reap_pending().expect("reap");
        let recs = sink.snapshot();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].action, LogAction::ReapPending);
        assert_eq!(recs[0].result, LogResult::Ok);
        assert!(recs[0].stream_id.is_none());
        assert!(recs[0].ptr.is_none());
    }

    #[test]
    fn sink_failure_increments_dropped_records_but_does_not_break_alloc() {
        // Custom sink that always refuses. Verifies LoggingResource
        // never propagates sink errors and counts losses.
        struct RefuseAllSink;
        impl LoggingSink for RefuseAllSink {
            fn emit(&self, _r: LogRecord) -> Result<(), SinkError> {
                Err(SinkError::Refused("test sink refuses all".into()))
            }
        }

        let Some(device) = try_device() else {
            return;
        };
        let inner = Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let sink = Arc::new(RefuseAllSink);
        let r = LoggingResource::new(inner, sink);

        // Allocator semantics are still correct.
        let block = r
            .allocate(128, StreamId::DEFAULT, AllocTag("refuse-test"))
            .expect("alloc must succeed even when sink refuses");
        r.deallocate(block).expect("dealloc must succeed too");
        // Two refused records (alloc + dealloc).
        assert_eq!(r.dropped_records(), 2);
    }

    #[test]
    fn forwards_bytes_outstanding_and_device_ordinal() {
        let Some(device) = try_device() else {
            return;
        };
        let inner = Box::new(DirectCudaResource::new(Arc::clone(&device), 3));
        let sink = Arc::new(InMemorySink::new());
        let r = LoggingResource::new(inner, sink);

        assert_eq!(r.device_ordinal(), 3);
        assert_eq!(r.bytes_outstanding(), 0);
    }
}
