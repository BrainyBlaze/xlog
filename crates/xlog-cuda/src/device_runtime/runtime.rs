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
use super::resource::{AllocTag, DeviceBlock, DeviceMemoryResource, ResourceResult, StreamId};
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
}
