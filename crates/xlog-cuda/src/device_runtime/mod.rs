//! Stream-ordered device memory runtime.
//!
//! [`crate::CudaProviderBuilder`] constructs one [`XlogDeviceRuntime`] and one
//! [`crate::GpuMemoryManager`] that share the provider's exact CUDA device.
//! The runtime owns the stream pool and composes the production resource stack:
//!
//! ```text
//! XlogDeviceRuntime
//!   -> StreamPool of non-blocking streams
//!   -> LoggingResource (optional)
//!   -> GlobalDeviceBudget
//!   -> AsyncCudaResource
//! ```
//!
//! [`GlobalDeviceBudget`] is the sole byte-admission authority. Optional
//! [`LoggingResource`] wraps it so successful operations and typed admission
//! failures are both observable exactly once. [`DirectCudaResource`] remains
//! available for crate tests that need a synchronous allocator; canonical
//! providers always use [`AsyncCudaResource`].
//!
//! Stream-ordered contract: every alloc / dealloc names a stream;
//! reuse across streams requires explicit event/sync. No reliance on
//! the CUDA null/default stream.

pub mod async_resource;
pub mod budget;
pub mod direct;
pub mod logging;
pub mod resource;
pub mod runtime;
pub mod stream_pool;

pub use async_resource::AsyncCudaResource;
pub use budget::GlobalDeviceBudget;
pub use direct::DirectCudaResource;
pub use logging::{
    InMemorySink, LogAction, LogRecord, LogResult, LoggingResource, LoggingSink, NullSink,
    SinkError,
};
pub use resource::{
    Access, AllocTag, BlockId, BlockState, DeviceBlock, DeviceMemoryResource, Generation,
    ResourceBudgetSnapshot, ResourceError, ResourceResult, StreamId,
};
pub(crate) use runtime::RuntimeMemoryReservation;
pub use runtime::{
    ConditionalGraphStats, EventLifecycleStats, ResidentCompletionEvent,
    ResidentGraphHandleLifecycleStats, XlogDeviceRuntime,
};
pub use stream_pool::{StreamPool, StreamPoolError, DEFAULT_MAX_STREAMS};
