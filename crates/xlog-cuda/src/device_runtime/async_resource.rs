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
//! # Stream-ordering contract enforced here
//!   * `allocate(.., stream, ..)` is ordered on the resolved
//!     `CudaStream`. The returned `DeviceBlock` carries the same
//!     `alloc_stream`.
//!   * `deallocate(block)` releases the underlying memory ordered on
//!     the block's `alloc_stream`. Callers must have synchronized any
//!     work on a different stream before deallocation.
//!   * Reuse of the underlying byte address by a future `allocate` is
//!     ordered after the previous deallocate by the CUDA driver's
//!     stream-ordered memory allocator semantics. A2 encodes this
//!     as a regression test.
//!
//! # `bytes_outstanding` and pending-free accounting
//!
//! The trait contract is "live + retired-but-not-yet-freed". A queued
//! `cuMemFreeAsync` is "retired-but-not-yet-freed" until the host
//! synchronizes the stream the free was queued on. We therefore keep
//! two atomic counters:
//!
//!   * `live_bytes` — bytes for blocks currently in the live map.
//!   * `pending_bytes` — bytes for blocks whose `CudaSlice` has been
//!     dropped (so a `cuMemFreeAsync` is queued on the alloc stream)
//!     but whose stream has not yet been synchronized by us.
//!
//! `bytes_outstanding()` returns `live_bytes + pending_bytes`.
//!
//! `reap_pending()` synchronizes each unique stream that has queued
//! frees we haven't drained, then atomically zeros `pending_bytes`
//! and clears the per-stream tracking. Callers (the future
//! `GlobalDeviceBudget`, A2's final assertions) call this before
//! treating `bytes_outstanding()` as authoritative.

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
    /// Live allocations keyed by raw device pointer. Removed on
    /// deallocate; the slice is then dropped, queueing
    /// `cuMemFreeAsync` on its bound stream.
    live: Mutex<HashMap<u64, CudaSlice<u8>>>,
    /// Bytes for blocks currently in `live`. Always accurate.
    live_bytes: AtomicUsize,
    /// Bytes for blocks dropped (queued for cuMemFreeAsync) but
    /// whose owning stream has not yet been synchronized by us.
    /// Equal to the sum of values in `pending_per_stream`. Both are
    /// updated under the `pending_per_stream` mutex so a concurrent
    /// `reap_pending` cannot wipe out bytes that a racing
    /// `deallocate` queued after reap drained the per-stream map.
    pending_bytes: AtomicUsize,
    /// Per-stream pending-free byte totals. Used by `reap_pending`
    /// to (a) compute the total to subtract from `pending_bytes`
    /// after stream synchronization, and (b) preserve any bytes
    /// added by a `deallocate` that races with reap — those bytes
    /// remain in this map and in `pending_bytes`, ready for the
    /// next reap.
    pending_per_stream: Mutex<HashMap<StreamId, usize>>,
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
            live_bytes: AtomicUsize::new(0),
            pending_bytes: AtomicUsize::new(0),
            pending_per_stream: Mutex::new(HashMap::new()),
        }
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn stream_pool(&self) -> &Arc<StreamPool> {
        &self.stream_pool
    }

    /// Bytes currently held by live blocks (excludes pending frees).
    /// Test/diagnostic accessor — production code should use
    /// `bytes_outstanding`.
    pub fn live_bytes(&self) -> usize {
        self.live_bytes.load(Ordering::Relaxed)
    }

    /// Bytes queued for `cuMemFreeAsync` whose stream has not yet
    /// been synchronized by us. Test/diagnostic accessor.
    pub fn pending_free_bytes(&self) -> usize {
        self.pending_bytes.load(Ordering::Relaxed)
    }

    /// Sum of per-stream pending byte tallies. Test/diagnostic
    /// accessor used to assert the invariant
    /// `pending_free_bytes() == pending_per_stream_total()`. The
    /// invariant must hold at any quiescent moment; if it fails
    /// the bookkeeping under the `pending_per_stream` mutex has
    /// drifted from the global atomic — see `deallocate` and
    /// `reap_pending`, which update both as a unit.
    pub fn pending_per_stream_total(&self) -> usize {
        let map = self
            .pending_per_stream
            .lock()
            .expect("AsyncCudaResource pending_per_stream poisoned");
        map.values().copied().sum()
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
        self.live_bytes.fetch_add(bytes, Ordering::Relaxed);

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

        // Move the bytes from "live" to "pending free": the slice
        // drop below queues `cuMemFreeAsync` on `block.alloc_stream`,
        // but the driver may not actually free until that stream
        // drains. The trait contract requires us to keep counting
        // these bytes until `reap_pending` confirms completion.
        //
        // The pending bookkeeping is updated as a unit under the
        // `pending_per_stream` mutex: per-stream tally first, then
        // the global atomic. `reap_pending` reads (drain, sync,
        // subtract) symmetrically under the same mutex around the
        // drain so it can only subtract the exact total it drained.
        // A `deallocate` that races with reap therefore lands either
        // entirely before reap's drain (its bytes are reaped this
        // round) or entirely after (its bytes stay pending for the
        // next reap) — never split.
        self.live_bytes.fetch_sub(block.bytes, Ordering::Relaxed);
        {
            let mut per_stream = self
                .pending_per_stream
                .lock()
                .expect("AsyncCudaResource pending_per_stream poisoned");
            *per_stream.entry(block.alloc_stream).or_insert(0) += block.bytes;
            self.pending_bytes.fetch_add(block.bytes, Ordering::Relaxed);
        }

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
        self.live_bytes.load(Ordering::Relaxed) + self.pending_bytes.load(Ordering::Relaxed)
    }

    fn reap_pending(&self) -> ResourceResult<()> {
        // Drain the per-stream map atomically. Anything added by a
        // racing `deallocate` after this point lands in a fresh
        // entry and waits for the next reap.
        //
        // Critically, we do NOT touch `pending_bytes` here — only
        // after the streams have synchronized do we subtract the
        // exact total we drained. A `deallocate` that races between
        // our drain and our subtract has already added to
        // `pending_bytes` under the same mutex (see `deallocate`),
        // and that addition is preserved because we use
        // `fetch_sub(drained_total)` rather than `store(0)`.
        let drained: HashMap<StreamId, usize> = {
            let mut per_stream = self
                .pending_per_stream
                .lock()
                .expect("AsyncCudaResource pending_per_stream poisoned");
            std::mem::take(&mut *per_stream)
        };
        if drained.is_empty() {
            return Ok(());
        }

        let mut drained_total: usize = 0;
        for (stream_id, bytes) in &drained {
            drained_total = drained_total.saturating_add(*bytes);
            // The pool may have rotated entries (it currently does
            // not — streams stay alive for the runtime's lifetime —
            // but be defensive). If the id is unresolved we still
            // count the pending bytes as drained: there is no
            // stream we can sync on, so the only consistent
            // accounting is to clear and let the caller surface a
            // fresh alloc against a known stream.
            if let Some(stream) = self.stream_pool.resolve(*stream_id) {
                stream.synchronize().map_err(|e| {
                    ResourceError::Driver(format!(
                        "AsyncCudaResource::reap_pending: stream sync failed: {}",
                        e
                    ))
                })?;
            }
        }

        // Once the streams have synchronized, the queued
        // `cuMemFreeAsync` calls for the bytes we drained have
        // completed by definition. Subtract exactly that total. Any
        // bytes added by a racing `deallocate` between our drain and
        // this line remain accounted for in `pending_bytes` and in
        // a fresh `pending_per_stream` entry for the next reap.
        self.pending_bytes
            .fetch_sub(drained_total, Ordering::Relaxed);
        Ok(())
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
        assert_eq!(r.live_bytes(), 2048);
        assert_eq!(r.pending_free_bytes(), 0);

        r.deallocate(block).expect("dealloc");
        // Pending after dealloc — cuMemFreeAsync is queued, not drained.
        assert_eq!(r.live_bytes(), 0);
        assert_eq!(r.pending_free_bytes(), 2048);
        assert_eq!(r.bytes_outstanding(), 2048);

        r.reap_pending().expect("reap pending");
        assert_eq!(r.bytes_outstanding(), 0);
        assert_eq!(r.pending_free_bytes(), 0);
    }

    #[test]
    fn allocate_on_acquired_non_default_stream() {
        let Some((device, pool)) = try_setup() else {
            return;
        };
        let r = AsyncCudaResource::new(device, 0, Arc::clone(&pool));
        let stream = pool.acquire().expect("acquire non-default stream");
        let block = r
            .allocate(1024, stream, AllocTag("async-test"))
            .expect("alloc on non-default stream");
        assert_eq!(block.alloc_stream, stream);
        r.deallocate(block).expect("dealloc");
        // Still counted as outstanding until reap.
        assert_eq!(r.bytes_outstanding(), 1024);
        r.reap_pending().expect("reap pending");
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

    #[test]
    fn reap_with_no_pending_is_noop() {
        let Some((device, pool)) = try_setup() else {
            return;
        };
        let r = AsyncCudaResource::new(device, 0, pool);
        r.reap_pending().expect("reap on empty");
        assert_eq!(r.bytes_outstanding(), 0);
    }
}
