//! Stream-ordered device memory runtime, RMM-inspired.
//!
//! v0.6 architecture work. Replaces the per-`CudaKernelProvider`
//! `GpuMemoryManager` model (which cannot enforce a real per-device
//! budget across parallel tests, Python users, or multiple executors
//! on a single physical GPU) with a per-CUDA-ordinal singleton
//! [`XlogDeviceRuntime`] composed of swappable
//! [`DeviceMemoryResource`] adaptors:
//!
//! ```text
//! XlogDeviceRuntime per CUDA ordinal
//!   -> StreamPool of non-blocking streams
//!   -> GlobalDeviceBudget per physical GPU
//!   -> Logging / Debug adaptor (optional)
//!   -> AsyncCudaResource (production) | DirectCudaResource (sanitizer/cert)
//! ```
//!
//! Required resources:
//!   * [`DirectCudaResource`] — cudarc default (non-pooled) allocation
//!     backend (`CudaDeviceInner::alloc::<u8>` / drop, which on
//!     async-alloc hosts forwards to `cuMemAllocAsync`). Candidate
//!     for the sanitizer/cert role because there is no `xlog`-level
//!     pool suballocation hiding out-of-bounds access from Compute
//!     Sanitizer; the sanitizer-visibility property itself is
//!     **unproven** until the M1 acceptance gate runs on a
//!     Compute-Sanitizer-supported host. A genuine raw-driver
//!     `cuMemAlloc`/`cuMemFree` backend is a separate future commit.
//!   * `AsyncCudaResource` — `cuMemAllocAsync`/`cuMemFreeAsync`
//!     bound to a caller-supplied stream via the stream pool;
//!     production default when supported.
//!   * `PoolResource` — performance tier, not part of this PR; gated
//!     behind correctness certification of the direct/async backends.
//!   * `DebugGuardResource` — optional canary/poison/quarantine layer.
//!   * `LoggingResource` — CSV allocation log: thread, time, action,
//!     ptr, bytes, stream, device, tag, query id.
//!
//! Stream-ordered contract: every alloc / dealloc names a stream;
//! reuse across streams requires explicit event/sync. No reliance on
//! the CUDA legacy null/default stream. Mirrors RMM's stream-ordered
//! rule — see https://github.com/rapidsai/RMM .
//!
//! v0.5.5 closed at PRs #49 / #50 / #52 (metadata-read state for
//! binary-join output counts). The fully GPU-resident binary-join
//! materialization rebase is gated on this allocator landing first.

pub mod async_resource;
pub mod direct;
pub mod logging;
pub mod resource;
pub mod runtime;
pub mod stream_pool;

pub use async_resource::AsyncCudaResource;
pub use direct::DirectCudaResource;
pub use logging::{
    InMemorySink, LogAction, LogRecord, LogResult, LoggingResource, LoggingSink, SinkError,
};
pub use resource::{
    AllocTag, BlockState, DeviceBlock, DeviceMemoryResource, Generation, ResourceError,
    ResourceResult, StreamId,
};
pub use runtime::{XlogDeviceRuntime, MAX_DEVICE_ORDINALS};
pub use stream_pool::{StreamPool, StreamPoolError, DEFAULT_MAX_STREAMS};
