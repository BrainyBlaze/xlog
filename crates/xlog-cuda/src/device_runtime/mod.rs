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
//!   * [`DirectCudaResource`] — `cuMemAlloc`/`cuMemFree`. Default for
//!     sanitizer / debug / certification: pool suballocation hides
//!     out-of-bounds access from Compute Sanitizer, so the
//!     correctness-gating path goes through the direct backend.
//!   * `AsyncCudaResource` — `cuMemAllocAsync`/`cuMemFreeAsync` via the
//!     stream pool; production default when supported.
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

pub mod resource;

pub use resource::{
    AllocTag, DeviceBlock, DeviceMemoryResource, ResourceError, ResourceResult, StreamId,
};
