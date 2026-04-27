//! [`DirectCudaResource`] — synchronous `cuMemAlloc` / `cuMemFree`
//! backend.
//!
//! Default for sanitizer / debug / certification mode. Each
//! [`DeviceMemoryResource::allocate`] call performs a fresh
//! `cuMemAlloc` (no pooling, no suballocation), and each
//! [`DeviceMemoryResource::deallocate`] call performs a `cuMemFree` —
//! so out-of-bounds access patterns are visible to Compute Sanitizer
//! at byte granularity, which is the load-bearing reason this backend
//! exists separately from the future async/pool tiers.
//!
//! Stream-ordered semantics on a synchronous backend are degenerate:
//! `cuMemAlloc`/`cuMemFree` are device-wide and not stream-ordered, so
//! reuse across streams is always safe (assuming the caller has
//! synchronized before deallocating). The backend still records the
//! `alloc_stream` for downstream resources and tests; it does not
//! enforce stream-ordering itself because the underlying API has none.
//! That enforcement is `AsyncCudaResource`'s job.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::CudaSlice;

use super::resource::{
    AllocTag, BlockState, DeviceBlock, DeviceMemoryResource, Generation, ResourceError,
    ResourceResult, StreamId,
};
use crate::CudaDevice;

/// Synchronous `cuMemAlloc` / `cuMemFree` adaptor. Holds the
/// underlying `CudaSlice<u8>` allocations alive in an internal map so
/// the runtime returns opaque [`DeviceBlock`]s to callers; on
/// deallocate the slice is dropped, which invokes `cuMemFree`.
///
/// Concurrency: `Send + Sync`. The internal map is protected by a
/// `Mutex`. Allocate and deallocate are short-running map operations
/// plus the underlying CUDA call.
pub struct DirectCudaResource {
    device: Arc<CudaDevice>,
    device_ordinal: u32,
    /// Live + retired-but-not-yet-freed allocations, keyed by raw
    /// device pointer. Holding the slice keeps `cuMemFree` from
    /// running until we explicitly drop it on deallocate.
    live: Mutex<HashMap<u64, CudaSlice<u8>>>,
    /// Sum of bytes outstanding (live + retired). Updated together
    /// with the map under the `live` mutex.
    bytes_outstanding: AtomicUsize,
}

impl DirectCudaResource {
    /// Construct a resource bound to `device`. `device_ordinal` is the
    /// CUDA ordinal for logging / multi-device disambiguation.
    pub fn new(device: Arc<CudaDevice>, device_ordinal: u32) -> Self {
        Self {
            device,
            device_ordinal,
            live: Mutex::new(HashMap::new()),
            bytes_outstanding: AtomicUsize::new(0),
        }
    }

    /// Borrow the device handle. Tests and downstream resources use
    /// this to launch kernels against the same device this resource
    /// allocates on.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

impl DeviceMemoryResource for DirectCudaResource {
    fn allocate(
        &self,
        bytes: usize,
        stream: StreamId,
        tag: AllocTag,
    ) -> ResourceResult<DeviceBlock> {
        if bytes == 0 {
            // Zero-byte allocations are not legal in CUDA; surface as
            // a contract error rather than calling cuMemAlloc(0).
            return Err(ResourceError::Driver(
                "DirectCudaResource: zero-byte allocation not supported".to_string(),
            ));
        }

        // SAFETY: the device handle is valid for the lifetime of `self`,
        // and `bytes > 0` is checked above. cudarc's `alloc::<u8>(bytes)`
        // forwards to `cuMemAlloc(bytes)`. Failure is propagated as
        // `ResourceError::Driver`.
        let slice = unsafe {
            self.device
                .inner()
                .alloc::<u8>(bytes)
                .map_err(|e| ResourceError::Driver(format!("cuMemAlloc({}): {}", bytes, e)))?
        };

        // Extract the raw device pointer. The "sync" handle returned by
        // `device_ptr` is intentionally leaked — the slice's lifetime is
        // managed by the map, not the sync.
        let (raw_ptr, sync) =
            <CudaSlice<u8> as cudarc::driver::DevicePtr<u8>>::device_ptr(&slice, slice.stream());
        std::mem::forget(sync);
        let ptr = raw_ptr;

        {
            let mut live = self.live.lock().expect("live map poisoned");
            // Pointer collisions on a single CUDA driver are not
            // possible while the prior allocation is still live;
            // surface as a hard error if it ever happens.
            if live.insert(ptr, slice).is_some() {
                return Err(ResourceError::Driver(format!(
                    "DirectCudaResource: pointer collision on alloc ({:#x})",
                    ptr
                )));
            }
        }
        self.bytes_outstanding.fetch_add(bytes, Ordering::Relaxed);

        Ok(DeviceBlock {
            ptr,
            device_ordinal: self.device_ordinal,
            alloc_stream: stream,
            bytes,
            align: std::mem::align_of::<u8>(),
            tag,
            generation: Generation::next(),
            state: BlockState::Live,
        })
    }

    fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()> {
        if block.device_ordinal != self.device_ordinal {
            return Err(ResourceError::Driver(format!(
                "DirectCudaResource: deallocate on wrong device (block ord {} vs resource ord {})",
                block.device_ordinal, self.device_ordinal
            )));
        }

        let removed = {
            let mut live = self.live.lock().expect("live map poisoned");
            live.remove(&block.ptr)
        };
        let slice = removed.ok_or_else(|| ResourceError::UseAfterFree {
            generation: block.generation,
        })?;

        self.bytes_outstanding
            .fetch_sub(block.bytes, Ordering::Relaxed);

        // Dropping the CudaSlice<u8> calls `cuMemFree`. cuMemFree is
        // device-wide and not stream-ordered; if the caller has work
        // queued on a stream that touches this memory they were
        // responsible for synchronizing before calling deallocate.
        drop(slice);
        Ok(())
    }

    fn device_ordinal(&self) -> u32 {
        self.device_ordinal
    }

    fn bytes_outstanding(&self) -> usize {
        self.bytes_outstanding.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_device() -> Option<Arc<CudaDevice>> {
        CudaDevice::new(0).ok().map(Arc::new)
    }

    #[test]
    fn allocate_then_deallocate_round_trips() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no CUDA device");
            return;
        };
        let r = DirectCudaResource::new(device, 0);
        assert_eq!(r.bytes_outstanding(), 0);

        let block = r
            .allocate(4096, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("alloc");
        assert_eq!(block.bytes, 4096);
        assert_eq!(block.state, BlockState::Live);
        assert_eq!(r.bytes_outstanding(), 4096);

        r.deallocate(block).expect("dealloc");
        assert_eq!(r.bytes_outstanding(), 0);
    }

    #[test]
    fn zero_byte_allocate_rejects() {
        let Some(device) = try_device() else {
            return;
        };
        let r = DirectCudaResource::new(device, 0);
        let err = r.allocate(0, StreamId::DEFAULT, AllocTag::UNTAGGED);
        assert!(matches!(err, Err(ResourceError::Driver(_))));
        assert_eq!(r.bytes_outstanding(), 0);
    }

    #[test]
    fn deallocate_unknown_block_returns_use_after_free() {
        let Some(device) = try_device() else {
            return;
        };
        let r = DirectCudaResource::new(device, 0);
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
        assert!(matches!(
            r.deallocate(bogus),
            Err(ResourceError::UseAfterFree { .. })
        ));
    }
}
