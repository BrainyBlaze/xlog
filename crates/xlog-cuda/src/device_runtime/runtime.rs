//! [`XlogDeviceRuntime`] — per-CUDA-ordinal singleton hosting the
//! device-runtime allocator stack.
//!
//! Replaces the per-`CudaKernelProvider` `GpuMemoryManager` model with
//! a single live runtime per physical GPU. All `CudaKernelProvider`s
//! on a given ordinal share the same runtime once the migration
//! commit lands; until then this type is constructed and used by
//! tests only.
//!
//! Singleton lifetime: leaked-Box, so the returned `&'static` borrows
//! are valid for the process. No teardown on drop — appropriate for a
//! GPU device runtime that should outlive any single executor.
//!
//! # Initialization race semantics
//!
//! Earlier revisions used `OnceLock::get_or_init(|| leaked_box)`
//! after building the runtime outside the lock. That pattern leaked
//! the loser's runtime (and its CUDA context handle) when two
//! threads raced on the first access for an ordinal.
//!
//! This module now uses an explicit per-ordinal `Mutex` plus
//! `OnceLock`: callers fast-path on `OnceLock::get()`, and on a miss
//! take the per-ordinal mutex, double-check the `OnceLock`, and only
//! the winner inside the mutex builds and stores the runtime. The
//! mutex is held only across the build, so subsequent reads are still
//! lock-free.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

use xlog_core::{Result, XlogError};

use super::direct::DirectCudaResource;
use super::resource::{
    Access, AllocTag, BlockId, DeviceBlock, DeviceMemoryResource, ResourceResult, StreamId,
};
use super::stream_pool::StreamPool;
use crate::CudaDevice;

/// Maximum CUDA ordinal supported by the singleton table. CUDA itself
/// caps at 16 visible devices in typical configurations; raise here
/// only when a multi-GPU node demands it.
pub const MAX_DEVICE_ORDINALS: usize = 16;

/// Per-ordinal singleton table. Each slot is initialized at most once
/// via `OnceLock`, gated by [`INIT_LOCKS`] so failed initialization
/// does not leak partial state.
static RUNTIMES: [OnceLock<&'static XlogDeviceRuntime>; MAX_DEVICE_ORDINALS] =
    [const { OnceLock::new() }; MAX_DEVICE_ORDINALS];

/// Per-ordinal initialization mutex. Only the holder may build and
/// store a runtime in [`RUNTIMES`]. Held across the device-open and
/// resource-construction calls so concurrent first callers do not
/// race-leak loser runtimes.
static INIT_LOCKS: [Mutex<()>; MAX_DEVICE_ORDINALS] =
    [const { Mutex::new(()) }; MAX_DEVICE_ORDINALS];

/// Per-CUDA-ordinal device-runtime singleton.
///
/// Owns the device handle, stream pool, and resource stack. Allocate
/// / deallocate calls forward to the resource. The resource is fixed
/// at construction (currently always [`DirectCudaResource`]); a
/// future commit will swap in [`AsyncCudaResource`] as the default
/// while keeping the direct backend reachable for sanitizer mode.
pub struct XlogDeviceRuntime {
    device_ordinal: u32,
    device: Arc<CudaDevice>,
    stream_pool: Arc<StreamPool>,
    resource: Mutex<Box<dyn DeviceMemoryResource + Send + Sync>>,
}

impl XlogDeviceRuntime {
    /// Compose an owned runtime around a caller-supplied resource
    /// stack. **Not** a singleton — the returned value is *not*
    /// stored in [`RUNTIMES`] and does not interact with `try_get`.
    ///
    /// Intended uses:
    ///   * Tests that need to drive a specific backend (e.g.,
    ///     `AsyncCudaResource`) through the same facade production
    ///     code uses, instead of constructing the resource directly.
    ///   * Future decorator stacks (`LoggingResource`,
    ///     `GlobalDeviceBudget`, `DebugGuardResource`) that wrap the
    ///     base resource before installation.
    ///
    /// The `device` and `stream_pool` arguments must be consistent
    /// with `device_ordinal` (the pool must be bound to the same
    /// device handle, and the device must be the one the resource
    /// allocates against). The constructor does not verify this —
    /// callers that compose mismatched parts get undefined
    /// runtime-level behavior, but the per-resource device-ordinal
    /// check on `deallocate` will still surface obvious mistakes as
    /// `ResourceError::Driver`.
    ///
    /// The singleton path remains [`Self::try_get`], which today
    /// always installs the cudarc default (non-pooled) backend
    /// ([`DirectCudaResource`]). Swapping the singleton's default
    /// resource is a separate later change gated on
    /// `GlobalDeviceBudget` and `LoggingResource` landing.
    pub fn with_resource(
        device: Arc<CudaDevice>,
        device_ordinal: u32,
        stream_pool: Arc<StreamPool>,
        resource: Box<dyn DeviceMemoryResource + Send + Sync>,
    ) -> Self {
        Self {
            device_ordinal,
            device,
            stream_pool,
            resource: Mutex::new(resource),
        }
    }

    /// Get the singleton for `ordinal`, initializing it on first
    /// access. Subsequent calls return the same `&'static`.
    ///
    /// Errors:
    ///   * `XlogError::Kernel` if `ordinal >= MAX_DEVICE_ORDINALS`.
    ///   * `XlogError::Kernel` if the CUDA device cannot be opened.
    ///
    /// Concurrency: at most one thread builds the runtime for a
    /// given ordinal. Other concurrent first callers block on the
    /// per-ordinal init mutex until the winner publishes via
    /// `OnceLock::set`, after which they observe the published
    /// runtime via the inside-mutex double-check or the lock-free
    /// fast path on subsequent calls.
    pub fn try_get(ordinal: u32) -> Result<&'static XlogDeviceRuntime> {
        let idx = ordinal as usize;
        if idx >= MAX_DEVICE_ORDINALS {
            return Err(XlogError::Kernel(format!(
                "XlogDeviceRuntime: ordinal {} exceeds MAX_DEVICE_ORDINALS={}",
                ordinal, MAX_DEVICE_ORDINALS
            )));
        }
        // Fast path: another thread already initialized this slot.
        if let Some(rt) = RUNTIMES[idx].get() {
            return Ok(*rt);
        }

        // Slow path: take the per-ordinal init mutex. Only one
        // thread per ordinal builds the runtime; the rest wait here
        // and observe the published value on the double-check below.
        let _guard = INIT_LOCKS[idx]
            .lock()
            .expect("XlogDeviceRuntime init mutex poisoned");

        // Double-check inside the lock: a previous holder may have
        // initialized while we were waiting for the mutex.
        if let Some(rt) = RUNTIMES[idx].get() {
            return Ok(*rt);
        }

        // We are the first writer for this ordinal. Build the
        // runtime; if any step fails, return the error and leave
        // RUNTIMES[idx] uninitialized so the next caller can retry.
        let device = Arc::new(CudaDevice::new(ordinal as usize).map_err(|e| {
            XlogError::Kernel(format!(
                "XlogDeviceRuntime: failed to open device {}: {}",
                ordinal, e
            ))
        })?);
        let stream_pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let resource: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(DirectCudaResource::new(Arc::clone(&device), ordinal));
        let runtime = Box::new(XlogDeviceRuntime {
            device_ordinal: ordinal,
            device,
            stream_pool,
            resource: Mutex::new(resource),
        });
        let leaked: &'static XlogDeviceRuntime = Box::leak(runtime);

        // We hold INIT_LOCKS[idx] and confirmed RUNTIMES[idx] is
        // empty under that lock, so this `set` cannot fail. Fall
        // through to a hard panic if it does — it indicates a
        // process-internal bug we cannot recover from.
        RUNTIMES[idx]
            .set(leaked)
            .map_err(|_| ())
            .expect("XlogDeviceRuntime: OnceLock::set raced under INIT_LOCKS — bug");
        Ok(leaked)
    }

    /// CUDA ordinal this runtime serves.
    pub fn device_ordinal(&self) -> u32 {
        self.device_ordinal
    }

    /// Borrow the device handle.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Borrow the stream pool.
    pub fn stream_pool(&self) -> &Arc<StreamPool> {
        &self.stream_pool
    }

    /// Allocate via the underlying resource. Stream-ordered: the
    /// returned [`DeviceBlock`] is bound to `stream`.
    pub fn allocate(
        &self,
        bytes: usize,
        stream: StreamId,
        tag: AllocTag,
    ) -> ResourceResult<DeviceBlock> {
        self.resource
            .lock()
            .expect("device-runtime resource poisoned")
            .allocate(bytes, stream, tag)
    }

    /// Deallocate via the underlying resource.
    pub fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()> {
        self.resource
            .lock()
            .expect("device-runtime resource poisoned")
            .deallocate(block)
    }

    /// Sum of bytes currently outstanding on this device, as reported
    /// by the underlying resource. Used by the global-budget adaptor
    /// (later commit) and the parallel-stress acceptance test.
    pub fn bytes_outstanding(&self) -> usize {
        self.resource
            .lock()
            .expect("device-runtime resource poisoned")
            .bytes_outstanding()
    }

    /// Drain pending async frees on the underlying resource. No-op
    /// for synchronous backends. Callers that need an accurate
    /// `bytes_outstanding` reading after a burst of asynchronous
    /// deallocations should call this first.
    pub fn reap_pending(&self) -> ResourceResult<()> {
        self.resource
            .lock()
            .expect("device-runtime resource poisoned")
            .reap_pending()
    }

    /// Record that work has been (or is being) submitted on
    /// `use_stream` that touches `block`. Forwards to the
    /// underlying resource stack
    /// (`GlobalDeviceBudget` → `LoggingResource` → `AsyncCudaResource`),
    /// where the stream-ordered backend attaches a CUDA event so
    /// `block.alloc_stream` waits on it before the queued
    /// `cuMemFreeAsync` runs. This is the production-reachable
    /// hook the future xlog launch builder will call for
    /// `read` / `write` / `read_write` buffer args; until that
    /// lands, callers that submit raw CUDA work on a stream
    /// other than `block.alloc_stream` should call this directly.
    /// See [`DeviceMemoryResource::record_block_use`] for the
    /// underlying contract.
    pub fn record_block_use(
        &self,
        block: &DeviceBlock,
        use_stream: StreamId,
    ) -> ResourceResult<()> {
        self.resource
            .lock()
            .expect("device-runtime resource poisoned")
            .record_block_use(block, use_stream)
    }

    /// Whether the active resource stack tracks cross-stream
    /// uses (i.e., supports `record_block_use`). The launch
    /// recorder's preflight checks this BEFORE queuing CUDA
    /// work, so a misconfigured runtime fails loudly at the
    /// boundary rather than after the launch is in flight.
    pub fn supports_block_use_tracking(&self) -> bool {
        self.resource
            .lock()
            .expect("device-runtime resource poisoned")
            .supports_block_use_tracking()
    }

    /// Pre-launch hook: queue cross-stream waits required for
    /// `use_stream` to safely access `block` with `access`
    /// semantics. MUST be called BEFORE the GPU work is enqueued
    /// on `use_stream`. Forwards to the resource stack; see
    /// [`DeviceMemoryResource::prepare_block_use`] for the
    /// underlying contract.
    pub fn prepare_block_use(
        &self,
        block: BlockId,
        use_stream: StreamId,
        access: Access,
    ) -> ResourceResult<()> {
        self.resource
            .lock()
            .expect("device-runtime resource poisoned")
            .prepare_block_use(block, use_stream, access)
    }

    /// Post-launch hook: record an event on `use_stream`
    /// capturing the work just enqueued and update `block`'s
    /// dependency state. MUST be called AFTER the launch /
    /// copy is queued. Forwards to the resource stack; see
    /// [`DeviceMemoryResource::finish_block_use`] for the
    /// underlying contract.
    pub fn finish_block_use(
        &self,
        block: BlockId,
        use_stream: StreamId,
        access: Access,
    ) -> ResourceResult<()> {
        self.resource
            .lock()
            .expect("device-runtime resource poisoned")
            .finish_block_use(block, use_stream, access)
    }

    /// Convenience for helper-internal scratch allocations that
    /// will be immediately written / read on `use_stream`.
    ///
    /// Looks up the [`BlockId`] from the slice's runtime block
    /// and calls [`Self::prepare_block_use`] with `access`. Use
    /// this directly after `GpuMemoryManager::alloc` when the
    /// buffer's first cross-stream consumer is the same operator
    /// (e.g., a hash-table bucket array memset on `launch_stream`
    /// against a buffer freshly allocated on the manager's
    /// default stream).
    ///
    /// Returns `Err(ResourceError::StreamMisuse)` if `slice` is
    /// not runtime-backed — strict callers should ensure their
    /// memory manager carries a runtime.
    pub fn prepare_first_use<T: cudarc::driver::DeviceRepr>(
        &self,
        slice: &crate::memory::TrackedCudaSlice<T>,
        use_stream: StreamId,
        access: Access,
    ) -> ResourceResult<()> {
        let block = slice.runtime_block().ok_or_else(|| {
            super::resource::ResourceError::StreamMisuse(
                "prepare_first_use: slice is not runtime-backed (the helper's \
                 GpuMemoryManager must be built via with_runtime)"
                    .to_string(),
            )
        })?;
        self.prepare_block_use(BlockId::from_block(block), use_stream, access)
    }

    /// Convenience for helper-internal scratch finish: looks up
    /// the [`BlockId`] from the slice and forwards to
    /// [`Self::finish_block_use`].
    pub fn finish_first_use<T: cudarc::driver::DeviceRepr>(
        &self,
        slice: &crate::memory::TrackedCudaSlice<T>,
        use_stream: StreamId,
        access: Access,
    ) -> ResourceResult<()> {
        let block = slice.runtime_block().ok_or_else(|| {
            super::resource::ResourceError::StreamMisuse(
                "finish_first_use: slice is not runtime-backed".to_string(),
            )
        })?;
        self.finish_block_use(BlockId::from_block(block), use_stream, access)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_runtime() -> Option<&'static XlogDeviceRuntime> {
        XlogDeviceRuntime::try_get(0).ok()
    }

    #[test]
    fn try_get_returns_same_singleton() {
        let Some(a) = try_runtime() else {
            return;
        };
        let b = XlogDeviceRuntime::try_get(0).expect("re-get");
        assert!(std::ptr::eq(a, b), "singleton must be stable for ordinal 0");
        assert_eq!(a.device_ordinal(), 0);
    }

    #[test]
    fn allocate_then_deallocate_via_runtime() {
        let Some(rt) = try_runtime() else {
            return;
        };
        let before = rt.bytes_outstanding();
        let block = rt
            .allocate(2048, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("alloc");
        assert_eq!(block.bytes, 2048);
        assert_eq!(rt.bytes_outstanding(), before + 2048);
        rt.deallocate(block).expect("dealloc");
        rt.reap_pending().expect("reap pending");
        assert_eq!(rt.bytes_outstanding(), before);
    }

    #[test]
    fn try_get_rejects_out_of_range_ordinal() {
        let err = XlogDeviceRuntime::try_get(MAX_DEVICE_ORDINALS as u32);
        assert!(err.is_err());
    }

    #[test]
    fn with_resource_composes_owned_runtime_outside_singleton() {
        use super::super::async_resource::AsyncCudaResource;

        let Some(rt) = try_runtime() else {
            return;
        };
        let device = Arc::clone(rt.device());
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let resource = Box::new(AsyncCudaResource::new(
            Arc::clone(&device),
            0,
            Arc::clone(&pool),
        ));

        let owned = XlogDeviceRuntime::with_resource(device, 0, pool, resource);
        assert_eq!(owned.device_ordinal(), 0);

        let block = owned
            .allocate(1024, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("alloc through composed runtime");
        assert_eq!(block.bytes, 1024);
        assert_eq!(owned.bytes_outstanding(), 1024);
        owned.deallocate(block).expect("dealloc");
        owned.reap_pending().expect("reap");
        assert_eq!(owned.bytes_outstanding(), 0);

        // Composed runtime is not stored in the singleton table:
        // the singleton for ordinal 0 is whatever `try_get` returns,
        // which must be a different memory address.
        let singleton = XlogDeviceRuntime::try_get(0).expect("singleton");
        assert!(
            !std::ptr::eq(&owned, singleton),
            "with_resource must not aliase the singleton slot"
        );
    }

    /// `try_get` installs `DirectCudaResource` by default. The
    /// runtime's `record_block_use` must therefore return
    /// `StreamMisuse` (the trait's default) rather than silently
    /// claiming success — anything else would let a launch
    /// builder running against the singleton observe `Ok(())`
    /// while no event is actually recorded, reproducing the
    /// cross-stream use-after-free this whole layer exists to
    /// prevent. See the trait-level doc on
    /// `DeviceMemoryResource::record_block_use`.
    #[test]
    fn try_get_runtime_record_block_use_rejected_with_stream_misuse() {
        let Some(rt) = try_runtime() else {
            return;
        };
        let block = rt
            .allocate(64, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("alloc through runtime");
        let err = rt.record_block_use(&block, StreamId::DEFAULT);
        match err {
            Err(super::super::resource::ResourceError::StreamMisuse(msg)) => {
                assert!(
                    msg.contains("unsupported"),
                    "expected 'unsupported' in StreamMisuse message, got {:?}",
                    msg
                );
            }
            other => panic!(
                "XlogDeviceRuntime::try_get default (DirectCudaResource) must \
                 reject record_block_use with StreamMisuse; got {:?}",
                other
            ),
        }
        rt.deallocate(block).expect("dealloc still works");
    }
}
