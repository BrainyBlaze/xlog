//! [`AsyncCudaResource`] — stream-ordered allocation backed by
//! cudarc's `CudaStream::alloc` (which forwards to `cuMemAllocAsync`
//! when the context supports it).
//!
//! Each [`DeviceMemoryResource::allocate`] call resolves the
//! caller-supplied [`StreamId`] to a live `cudarc::driver::CudaStream`
//! via the [`StreamPool`], allocates against that stream, and stores
//! the resulting `CudaSlice<u8>` in the resource's live map. Drop on
//! deallocate invokes `cuMemFreeAsync` (when supported) on the same
//! stream the allocation was bound to.
//!
//! This backend is the production candidate. It is **not** the
//! sanitizer/cert backend — pool/async behavior can hide byte-level
//! out-of-bounds patterns from Compute Sanitizer; the cert role
//! belongs to [`DirectCudaResource`] (subject to M1 confirmation on a
//! supported host).
//!
//! Stream-ordering contract enforced here:
//!   * `allocate(.., stream, ..)` is ordered on the resolved
//!     `CudaStream`. The returned `DeviceBlock` carries the same
//!     `alloc_stream`.
//!   * `deallocate(block)` releases the underlying memory ordered on
//!     the block's `alloc_stream`. Callers must have synchronized any
//!     work on a different stream before deallocation.
//!   * Reuse of the underlying byte address by a future `allocate` is
//!     ordered after the previous deallocate by the CUDA driver's
//!     stream-ordered memory allocator semantics. A2 will encode this
//!     as a regression test.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::CudaSlice;

use super::resource::{
    AllocTag, BlockState, DeviceBlock, DeviceMemoryResource, Generation, ResourceError,
    ResourceResult, StreamId,
};
use super::stream_pool::StreamPool;
use crate::CudaDevice;

/// Stream-ordered cudarc-backed allocator.
pub struct AsyncCudaResource {
    device: Arc<CudaDevice>,
    device_ordinal: u32,
    stream_pool: Arc<StreamPool>,
    /// Live + retired allocations keyed by raw device pointer. The
    /// stored `CudaSlice<u8>` carries its own stream binding, which is
    /// dropped on deallocate to invoke `cuMemFreeAsync` (when
    /// supported).
    live: Mutex<HashMap<u64, CudaSlice<u8>>>,
    bytes_outstanding: AtomicUsize,
}

impl AsyncCudaResource {
    /// Construct a resource bound to `device` using `stream_pool` for
    /// stream resolution. `device_ordinal` is the CUDA ordinal for
    /// logging / multi-device disambiguation.
    pub fn new(device: Arc<CudaDevice>, device_ordinal: u32, stream_pool: Arc<StreamPool>) -> Self {
        Self {
            device,
            device_ordinal,
            stream_pool,
            live: Mutex::new(HashMap::new()),
            bytes_outstanding: AtomicUsize::new(0),
        }
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn stream_pool(&self) -> &Arc<StreamPool> {
        &self.stream_pool
    }
}

impl DeviceMemoryResource for AsyncCudaResource {
    fn allocate(
        &self,
        bytes: usize,
        stream: StreamId,
        tag: AllocTag,
    ) -> ResourceResult<DeviceBlock> {
        if bytes == 0 {
            return Err(ResourceError::Driver(
                "AsyncCudaResource: zero-byte allocation not supported".to_string(),
            ));
        }
        let cu_stream = self.stream_pool.resolve(stream).ok_or_else(|| {
            ResourceError::StreamMisuse(format!(
                "AsyncCudaResource: unknown StreamId({})",
                stream.0
            ))
        })?;

        // SAFETY: bytes > 0 verified above. cudarc's
        // `CudaStream::alloc::<u8>(len)` forwards to `cuMemAllocAsync`
        // when the context has async-alloc enabled (CUDA 11.2+);
        // otherwise it falls back to synchronous alloc internally.
        // Failures are surfaced as `ResourceError::Driver`.
        let slice = unsafe {
            cu_stream
                .alloc::<u8>(bytes)
                .map_err(|e| ResourceError::Driver(format!("cuMemAllocAsync({}): {}", bytes, e)))?
        };

        // Extract the raw device pointer for the DeviceBlock surface.
        // The "sync" handle returned by `device_ptr` is intentionally
        // leaked — the slice's lifetime is managed by our live map,
        // not by the sync token.
        let (raw_ptr, sync) =
            <CudaSlice<u8> as cudarc::driver::DevicePtr<u8>>::device_ptr(&slice, slice.stream());
        std::mem::forget(sync);
        let ptr = raw_ptr;

        {
            let mut live = self
                .live
                .lock()
                .expect("AsyncCudaResource live map poisoned");
            if live.insert(ptr, slice).is_some() {
                return Err(ResourceError::Driver(format!(
                    "AsyncCudaResource: pointer collision on alloc ({:#x})",
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
                "AsyncCudaResource: deallocate on wrong device (block ord {} vs resource ord {})",
                block.device_ordinal, self.device_ordinal
            )));
        }
        let removed = {
            let mut live = self
                .live
                .lock()
                .expect("AsyncCudaResource live map poisoned");
            live.remove(&block.ptr)
        };
        let slice = removed.ok_or_else(|| ResourceError::UseAfterFree {
            generation: block.generation,
        })?;

        self.bytes_outstanding
            .fetch_sub(block.bytes, Ordering::Relaxed);

        // Dropping the CudaSlice<u8> invokes cuMemFreeAsync on its
        // bound stream when async-alloc is enabled, otherwise falls
        // back to synchronous cuMemFree. Either way the deallocation
        // is ordered on the slice's stream, which matches the
        // DeviceBlock's `alloc_stream`.
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

    fn try_setup() -> Option<(Arc<CudaDevice>, Arc<StreamPool>)> {
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        Some((device, pool))
    }

    #[test]
    fn allocate_then_deallocate_round_trips_on_default_stream() {
        let Some((device, pool)) = try_setup() else {
            return;
        };
        let r = AsyncCudaResource::new(device, 0, pool);
        let block = r
            .allocate(2048, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("alloc");
        assert_eq!(block.bytes, 2048);
        assert_eq!(block.alloc_stream, StreamId::DEFAULT);
        assert_eq!(r.bytes_outstanding(), 2048);
        r.deallocate(block).expect("dealloc");
        assert_eq!(r.bytes_outstanding(), 0);
    }

    #[test]
    fn allocate_on_acquired_non_default_stream() {
        let Some((device, pool)) = try_setup() else {
            return;
        };
        let r = AsyncCudaResource::new(device, 0, Arc::clone(&pool));
        let stream = pool.acquire();
        let block = r
            .allocate(1024, stream, AllocTag("async-test"))
            .expect("alloc on non-default stream");
        assert_eq!(block.alloc_stream, stream);
        r.deallocate(block).expect("dealloc");
        assert_eq!(r.bytes_outstanding(), 0);
    }

    #[test]
    fn allocate_unknown_stream_id_rejected() {
        let Some((device, pool)) = try_setup() else {
            return;
        };
        let r = AsyncCudaResource::new(device, 0, pool);
        let err = r.allocate(64, StreamId(99), AllocTag::UNTAGGED);
        assert!(matches!(err, Err(ResourceError::StreamMisuse(_))));
    }

    #[test]
    fn deallocate_unknown_block_returns_use_after_free() {
        let Some((device, pool)) = try_setup() else {
            return;
        };
        let r = AsyncCudaResource::new(device, 0, pool);
        let bogus = DeviceBlock {
            ptr: 0xfeed_face,
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
