//! [`DirectCudaResource`] — cudarc default (non-pooled) allocation
//! backend.
//!
//! Each [`DeviceMemoryResource::allocate`] call goes through cudarc's
//! `CudaDeviceInner::alloc::<u8>(bytes)`. cudarc itself routes that
//! through `CudaStream::alloc` against the device's default stream,
//! which forwards to **`cuMemAllocAsync` on contexts that support
//! async-alloc** and falls back to a synchronous path otherwise.
//! There is no `xlog`-level pooling or suballocation in this layer —
//! every `allocate` is one cudarc call, every `deallocate` drops the
//! resulting `CudaSlice<u8>` (which in turn invokes `cuMemFreeAsync`
//! or the synchronous fallback that cudarc selected).
//!
//! Earlier revisions described this backend as "raw `cuMemAlloc` /
//! `cuMemFree`". That was wrong. A genuine raw-driver direct backend
//! (bypassing cudarc entirely) is a separate work item; until that
//! exists, this backend is the **non-pooled default** — not a synchronous
//! `cuMemAlloc`/`cuMemFree` adaptor — and it does not by itself
//! guarantee that pool suballocation is absent from the underlying
//! call path on a given host.
//!
//! **Sanitizer status: unproven.** The intent of having a non-pooled
//! backend is that pool *suballocation* hides byte-level
//! out-of-bounds access from Compute Sanitizer. The cudarc default
//! path forwards to `cuMemAllocAsync`, which on async-alloc hosts is
//! a stream-ordered allocator; whether that is sufficiently
//! sanitizer-visible is exactly what the **M1 acceptance gate**
//! (manual, Compute-Sanitizer-supported host) is supposed to
//! confirm. Do not describe this backend as "sanitizer-certified"
//! until M1 has produced a captured negative-test pass; until M1
//! lands, treat the sanitizer role as "candidate, not certified".
//!
//! Stream-ordered semantics: the backend records the caller-supplied
//! `alloc_stream` on the returned [`DeviceBlock`] but does **not**
//! attempt to bind the underlying cudarc allocation to that stream —
//! cudarc allocates against the device's default stream regardless.
//! Stream-ordered allocation/free that honors a caller-supplied
//! [`StreamId`] is `AsyncCudaResource`'s responsibility (separate
//! commit).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::CudaSlice;

use super::resource::{
    AllocTag, BlockState, DeviceBlock, DeviceMemoryResource, Generation, ResourceError,
    ResourceResult, StreamId,
};
use crate::CudaDevice;

/// cudarc default (non-pooled) allocation adaptor. Holds the
/// underlying `CudaSlice<u8>` allocations alive in an internal map so
/// the runtime returns opaque [`DeviceBlock`]s to callers; on
/// deallocate the slice is dropped, which invokes whichever cudarc
/// free path matches the alloc path (`cuMemFreeAsync` on async-alloc
/// hosts, the synchronous fallback otherwise).
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

        // SAFETY: the device handle is valid for the lifetime of
        // `self`, and `bytes > 0` is checked above. cudarc's
        // `CudaDeviceInner::alloc::<u8>(bytes)` forwards to
        // `cuMemAllocAsync` (against the device's default stream)
        // when the context supports async-alloc, otherwise to the
        // synchronous fallback. Failure is propagated as
        // `ResourceError::Driver`.
        let slice = unsafe {
            self.device.inner().alloc::<u8>(bytes).map_err(|e| {
                ResourceError::Driver(format!("cudarc alloc::<u8>({}): {}", bytes, e))
            })?
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
            // The CUDA driver does not return the same byte address
            // for two simultaneously live allocations. If our map
            // already has this pointer, it indicates a bookkeeping
            // bug or driver behavior we want to surface loudly.
            // Use `contains_key` then `insert` so a (theoretical)
            // collision returns `Err` without mutating the map —
            // a `live.insert(ptr, slice).is_some()` pattern would
            // replace the existing entry, drop the old slice (which
            // calls cuMemFree on memory we still believe we own),
            // and leave the new slice resident in `live` while we
            // return Err. Avoid that here.
            if live.contains_key(&ptr) {
                return Err(ResourceError::Driver(format!(
                    "DirectCudaResource: pointer collision on alloc ({:#x})",
                    ptr
                )));
            }
            live.insert(ptr, slice);
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
        let slice = removed.ok_or(ResourceError::UseAfterFree {
            generation: block.generation,
        })?;

        self.bytes_outstanding
            .fetch_sub(block.bytes, Ordering::Relaxed);

        // Dropping the `CudaSlice<u8>` invokes whichever cudarc free
        // path matches the alloc path: `cuMemFreeAsync` on the
        // device's default stream when the context supports
        // async-alloc, the synchronous fallback otherwise. Either
        // way the caller-supplied `block.alloc_stream` is **not**
        // honored here — only `AsyncCudaResource` does that. If the
        // caller has work queued on a non-default stream that
        // touches this memory they were responsible for
        // synchronizing before calling deallocate.
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

    /// Locks the contract that DirectCudaResource does NOT
    /// silently accept `record_block_use`. If a caller (e.g. the
    /// future xlog launch builder) calls record_block_use against
    /// a runtime built around DirectCudaResource, the call must
    /// fail loudly with StreamMisuse — not return Ok and quietly
    /// fail to track anything. False safety here would let
    /// downstream code queue cross-stream kernels and drop
    /// blocks while the cross-stream use was never recorded,
    /// reproducing exactly the use-after-free this whole layer
    /// exists to prevent.
    ///
    /// Implementation note: DirectCudaResource inherits the
    /// trait's default `record_block_use` impl (which returns
    /// `StreamMisuse`). It does NOT override. If a future change
    /// adds a real override, it must make the override
    /// genuinely track cross-stream uses (similar to
    /// AsyncCudaResource's implementation) — anything else
    /// regresses this contract.
    #[test]
    fn record_block_use_rejected_with_stream_misuse() {
        let Some(device) = try_device() else {
            return;
        };
        let r = DirectCudaResource::new(device, 0);
        let block = r
            .allocate(64, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("alloc");
        let err = r.record_block_use(&block, StreamId::DEFAULT);
        match err {
            Err(ResourceError::StreamMisuse(msg)) => {
                assert!(
                    msg.contains("unsupported"),
                    "expected 'unsupported' in StreamMisuse message, got {:?}",
                    msg
                );
            }
            other => panic!(
                "DirectCudaResource::record_block_use must return StreamMisuse \
                 to surface unsupported cross-stream tracking; got {:?}",
                other
            ),
        }
        // The block stays live — a failed record_block_use must
        // NOT have removed the entry or dropped the slice.
        assert_eq!(r.bytes_outstanding(), 64);
        r.deallocate(block).expect("dealloc still works");
    }
}
