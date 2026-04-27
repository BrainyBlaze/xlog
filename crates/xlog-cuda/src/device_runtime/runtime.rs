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
/// via `OnceLock`. Slots are leaked Boxes — the runtime lives for the
/// process's lifetime, mirroring how the underlying CUDA context
/// behaves.
static RUNTIMES: [OnceLock<&'static XlogDeviceRuntime>; MAX_DEVICE_ORDINALS] = [
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
];

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
    /// Get the singleton for `ordinal`, initializing it on first
    /// access. Subsequent calls return the same `&'static`.
    ///
    /// Errors:
    ///   * `XlogError::Kernel` if `ordinal >= MAX_DEVICE_ORDINALS`.
    ///   * `XlogError::Kernel` if the CUDA device cannot be opened.
    pub fn try_get(ordinal: u32) -> Result<&'static XlogDeviceRuntime> {
        let idx = ordinal as usize;
        if idx >= MAX_DEVICE_ORDINALS {
            return Err(XlogError::Kernel(format!(
                "XlogDeviceRuntime: ordinal {} exceeds MAX_DEVICE_ORDINALS={}",
                ordinal, MAX_DEVICE_ORDINALS
            )));
        }
        if let Some(rt) = RUNTIMES[idx].get() {
            return Ok(*rt);
        }
        // First access for this ordinal: build the runtime and store
        // it. `get_or_init` serializes initialization, but the device
        // open and resource construction may fail. We use a
        // build-then-store pattern: try to build, and only call
        // `get_or_init` with the leaked reference. If the build fails,
        // surface the error so the caller can retry on a different
        // ordinal.
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

        // Race: another thread may have initialized this slot between
        // our `get` check and here. `get_or_init` returns the winner;
        // if our box wasn't installed the leaked allocation becomes
        // dead, but the device handle was the only resource cost and
        // it survives via `Arc::clone` in our box's path.
        let stored = RUNTIMES[idx].get_or_init(|| leaked);
        Ok(*stored)
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
        assert_eq!(rt.bytes_outstanding(), before);
    }

    #[test]
    fn try_get_rejects_out_of_range_ordinal() {
        let err = XlogDeviceRuntime::try_get(MAX_DEVICE_ORDINALS as u32);
        assert!(err.is_err());
    }
}
