//! Core [`DeviceMemoryResource`] trait and supporting types.
//!
//! Mirrors RMM's `device_memory_resource` shape so a future optional
//! RMM backend can satisfy the same trait without requiring callers to
//! change. Stream-ordered: every alloc/dealloc names a stream; cross-
//! stream reuse requires explicit event-based synchronization.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Identifier for a CUDA stream owned by the runtime's stream pool.
/// Wraps the raw cudarc stream handle the resource will use for
/// `cuMemAllocAsync` / `cuMemFreeAsync` ordering. Construction goes
/// through the runtime; do not fabricate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct StreamId(pub u32);

impl StreamId {
    /// The "default" pool stream for tests and synchronous codepaths
    /// that have no other stream context. Production callers should
    /// always carry a real stream from the executor / kernel launch
    /// site.
    pub const DEFAULT: StreamId = StreamId(0);
}

/// Caller-supplied tag for allocation log lines. Short-lived strings
/// are interned by the logging resource; long-lived borrows are not
/// retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AllocTag(pub &'static str);

impl AllocTag {
    pub const UNTAGGED: AllocTag = AllocTag("untagged");
}

/// Monotonic counter for distinguishing reuse of the same byte address
/// across drop / reallocate cycles. Logging and debug-guard resources
/// use this to detect use-after-free.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Generation(pub u64);

static GENERATION_COUNTER: AtomicU64 = AtomicU64::new(1);

impl Generation {
    /// Allocate a fresh, monotonically increasing generation number.
    /// Concurrent calls return distinct values.
    pub fn next() -> Generation {
        Generation(GENERATION_COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// State of an outstanding [`DeviceBlock`] from the runtime's
/// perspective. Adaptors flip blocks between these states; bug-detection
/// resources reject operations on blocks in an unexpected state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockState {
    /// Returned from `allocate`; safe to read/write on `alloc_stream`
    /// or after a synchronization to another stream.
    Live,
    /// Returned from `deallocate` but still pending kernel completion
    /// on its owning stream. Reuse must wait for stream sync.
    Retired,
    /// Held by `DebugGuardResource` for delayed reuse / canary
    /// validation. Not reissued until the quarantine window passes.
    Quarantined,
    /// Memory has been physically freed. Any further use is a bug.
    Freed,
}

/// One outstanding device-memory allocation. Owned by the caller until
/// returned to its originating resource via
/// [`DeviceMemoryResource::deallocate`].
///
/// Carries the metadata required for stream-ordered correctness and
/// post-mortem debugging: the resource that owns the block, the device
/// ordinal, the stream the allocation is bound to, byte size, alignment,
/// caller tag, generation number, and current state.
#[derive(Debug)]
pub struct DeviceBlock {
    /// Raw device pointer (opaque to safe Rust callers).
    pub ptr: u64,
    /// CUDA ordinal of the device this block lives on.
    pub device_ordinal: u32,
    /// Allocation stream. Reads/writes on a different stream require
    /// explicit synchronization (event wait or device sync).
    pub alloc_stream: StreamId,
    /// Size in bytes. May exceed the caller-requested size when the
    /// resource rounds up for alignment or pool granularity.
    pub bytes: usize,
    /// Alignment in bytes (always ≥ caller request).
    pub align: usize,
    /// Caller-supplied tag, surfaced in allocation logs.
    pub tag: AllocTag,
    /// Monotonic generation. Reused addresses get fresh generations.
    pub generation: Generation,
    /// Current state. Adaptors transition this; tests assert on it.
    pub state: BlockState,
}

/// Errors returned by resource implementations. Distinct variants for
/// the cases stress tests need to pin (out-of-budget vs CUDA driver
/// failure vs use-after-free etc.).
#[derive(Debug)]
pub enum ResourceError {
    /// The requested allocation would exceed the resource's budget.
    /// Carries the requested bytes and the remaining budget so tests
    /// can assert deterministic refusal.
    OutOfBudget { requested: usize, remaining: usize },
    /// CUDA driver returned an error. Carries the wrapped message.
    Driver(String),
    /// A stream-ordered contract was violated (e.g. dealloc on a
    /// stream that does not match the alloc stream without an
    /// intervening sync).
    StreamMisuse(String),
    /// A debug-guard or logging adaptor detected a use-after-free or
    /// double-free. Hard error in debug builds; surfaced upward.
    UseAfterFree { generation: Generation },
    /// A debug-guard adaptor detected an out-of-bounds write past a
    /// canary boundary.
    OutOfBounds { generation: Generation },
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBudget {
                requested,
                remaining,
            } => write!(
                f,
                "out of budget: requested {} bytes, remaining {} bytes",
                requested, remaining
            ),
            Self::Driver(msg) => write!(f, "CUDA driver error: {}", msg),
            Self::StreamMisuse(msg) => write!(f, "stream-ordered contract violated: {}", msg),
            Self::UseAfterFree { generation } => {
                write!(f, "use-after-free on generation {:?}", generation)
            }
            Self::OutOfBounds { generation } => {
                write!(f, "out-of-bounds write on generation {:?}", generation)
            }
        }
    }
}

impl std::error::Error for ResourceError {}

pub type ResourceResult<T> = std::result::Result<T, ResourceError>;

/// Stream-ordered device memory resource. Implementations:
///   * [`crate::device_runtime::direct::DirectCudaResource`] —
///     cudarc default (non-pooled) backend; **candidate** for the
///     sanitizer/cert role, **unproven** until the M1 manual gate
///     runs on a Compute-Sanitizer-supported host.
///   * [`crate::device_runtime::async_resource::AsyncCudaResource`] —
///     stream-ordered cuMemAllocAsync/cuMemFreeAsync backend;
///     production default when the context supports async-alloc.
///   * [`crate::device_runtime::logging::LoggingResource`] —
///     telemetry decorator over any inner resource.
///   * [`crate::device_runtime::budget::GlobalDeviceBudget`] —
///     per-runtime byte-limit decorator over any inner resource.
///   * `PoolResource` — performance tier; v0.7+ (not implemented).
///   * `DebugGuardResource` — canary/poison/quarantine; v0.7+
///     (not implemented).
///
/// Implementations must be thread-safe. The runtime composes resources
/// via decoration (each resource wraps an inner `Box<dyn
/// DeviceMemoryResource + Send + Sync>`).
pub trait DeviceMemoryResource: Send + Sync {
    /// Allocate `bytes` bytes on the resource's device, ordered on
    /// `stream`. The returned block is in [`BlockState::Live`].
    fn allocate(
        &self,
        bytes: usize,
        stream: StreamId,
        tag: AllocTag,
    ) -> ResourceResult<DeviceBlock>;

    /// Return `block` to the resource. After this call the block's
    /// state is [`BlockState::Retired`] (or [`BlockState::Quarantined`]
    /// for debug-guard resources). Reuse of the underlying memory is
    /// resource-specific but must respect the stream-ordered contract.
    ///
    /// `block.alloc_stream` is authoritative for ordering. If the
    /// caller has touched the memory on a different stream, they must
    /// have synchronized before calling `deallocate`.
    fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()>;

    /// CUDA device ordinal this resource serves. Resources are pinned
    /// to a single device.
    fn device_ordinal(&self) -> u32;

    /// Bytes currently outstanding (live + retired-but-not-yet-freed).
    /// Used by tests and by the global budget adaptor.
    fn bytes_outstanding(&self) -> usize;

    /// Drain any retired-but-not-yet-freed bytes whose underlying
    /// CUDA work has completed. For synchronous backends this is a
    /// no-op. For stream-ordered async backends this synchronizes
    /// the streams that have queued `cuMemFreeAsync` calls and
    /// re-counts `bytes_outstanding` accordingly.
    ///
    /// Callers that need an accurate budget reading after a burst
    /// of asynchronous deallocations should call this before
    /// reading `bytes_outstanding`. Calling on a synchronous backend
    /// is harmless and free.
    fn reap_pending(&self) -> ResourceResult<()> {
        Ok(())
    }

    /// Record that work has been (or is being) submitted on
    /// `use_stream` that touches `block`'s bytes. Resources that
    /// participate in cross-stream lifetime tracking (notably the
    /// stream-ordered async backend) MUST attach a CUDA event from
    /// `use_stream` to the block; on `deallocate(block)`, the
    /// block's `alloc_stream` will wait on every recorded event
    /// before queueing the underlying free.
    ///
    /// **The default implementation returns
    /// [`ResourceError::StreamMisuse`].** This is intentional: a
    /// silent no-op default would let a launch builder call
    /// `record_block_use` against a resource that does not
    /// actually track cross-stream uses (e.g.,
    /// [`crate::device_runtime::direct::DirectCudaResource`]),
    /// observe `Ok(())`, queue a kernel on a different stream,
    /// then drop the block — and quietly hit the cross-stream
    /// use-after-free that this API exists to prevent. False
    /// safety is worse than no safety. Resources that cannot
    /// track cross-stream uses MUST inherit this default;
    /// callers (notably the future xlog launch builder) MUST
    /// surface the error rather than masking it.
    ///
    /// Override status today:
    ///   * [`crate::device_runtime::async_resource::AsyncCudaResource`]
    ///     overrides with real event tracking.
    ///   * [`crate::device_runtime::logging::LoggingResource`] and
    ///     [`crate::device_runtime::budget::GlobalDeviceBudget`]
    ///     forward to their inner resource (so the underlying
    ///     backend's behavior surfaces unchanged).
    ///   * [`crate::device_runtime::direct::DirectCudaResource`]
    ///     does NOT override — it correctly returns
    ///     `StreamMisuse` and forces callers to either route
    ///     allocations through `AsyncCudaResource` or take
    ///     responsibility for cross-stream synchronization
    ///     themselves.
    ///
    /// # Errors
    ///   * [`ResourceError::StreamMisuse`] from the default impl
    ///     when the resource cannot track cross-stream uses.
    ///   * [`ResourceError::UseAfterFree`] if `block` is not the
    ///     block currently live at `block.ptr` (caller likely
    ///     handed back a stale [`DeviceBlock`] whose generation
    ///     no longer matches the live entry).
    ///   * [`ResourceError::StreamMisuse`] if `use_stream` does
    ///     not resolve in the resource's stream pool.
    ///   * [`ResourceError::Driver`] for CUDA driver / event
    ///     creation failures.
    ///
    /// Callers that bypass this API and submit cross-stream work
    /// directly (raw `cuMemcpyDtoHAsync`, raw `Vec<*mut c_void>`
    /// kernel launches that the launch builder did not see, etc.)
    /// are responsible for their own cross-stream synchronization.
    /// The resource cannot infer arbitrary external CUDA work.
    fn record_block_use(&self, block: &DeviceBlock, use_stream: StreamId) -> ResourceResult<()> {
        let _ = (block, use_stream);
        Err(ResourceError::StreamMisuse(
            "record_block_use unsupported by this resource (the active backend \
             does not track cross-stream uses; route allocations through a \
             stream-ordered backend such as AsyncCudaResource, or take \
             responsibility for cross-stream synchronization explicitly)"
                .to_string(),
        ))
    }

    /// Whether this resource (and any inner resources it
    /// composes) actually tracks cross-stream uses via
    /// `record_block_use`. Used by the launch recorder's
    /// preflight to fail BEFORE queueing CUDA work, rather than
    /// after. The default returns `false` to match the trait's
    /// default `record_block_use` behavior; resources that
    /// override `record_block_use` to track events MUST override
    /// this to return `true`. Decorators forward to inner.
    fn supports_block_use_tracking(&self) -> bool {
        false
    }
}
