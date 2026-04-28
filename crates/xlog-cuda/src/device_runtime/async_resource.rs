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
//! `reap_pending()` drains the per-stream pending map under the
//! per-stream mutex, synchronizes each drained stream, and then
//! subtracts only the **synchronized** total from `pending_bytes`
//! via `fetch_sub` — it does **not** zero the counter. A
//! `deallocate` that races between reap's drain and its `fetch_sub`
//! re-populates both the per-stream map and the global atomic
//! together (under the same mutex), so its bytes either land
//! entirely before the drain (reaped this round) or entirely after
//! (kept for the next reap), never split.
//!
//! On the first stream-sync failure, the failing entry and every
//! remaining un-iterated drained entry are **restored** into
//! `pending_per_stream` so a subsequent reap can retry them. Only
//! the bytes for streams that successfully synchronized are
//! decremented from `pending_bytes`. Without this recovery, a
//! transient driver error mid-reap would lose track of pending
//! bytes forever — the drained map would be gone, `pending_bytes`
//! would still count them, but no stream id would be queued for
//! a future reap. Production callers (`GlobalDeviceBudget`, A2's
//! final assertions) thus see consistent
//! `bytes_outstanding()` even on transient sync failures.

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
            // Use `contains_key` then `insert` so a (theoretical)
            // pointer collision returns `Err` without mutating the
            // map. The `live.insert(ptr, slice).is_some()` pattern
            // would replace the existing entry, drop the old slice
            // (queueing cuMemFreeAsync on memory we still believe
            // we own), and leave the new slice resident while we
            // return Err — `live_bytes` would also not be updated.
            // Avoid that here.
            if live.contains_key(&ptr) {
                return Err(ResourceError::Driver(format!(
                    "AsyncCudaResource: pointer collision on alloc ({:#x})",
                    ptr
                )));
            }
            live.insert(ptr, slice);
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
        self.reap_pending_with(|stream_id| match self.stream_pool.resolve(stream_id) {
            Some(stream) => stream.synchronize().map_err(|e| {
                ResourceError::Driver(format!(
                    "AsyncCudaResource::reap_pending: stream sync failed: {}",
                    e
                ))
            }),
            // Pool returned no handle for this id. The pool currently
            // never rotates entries, so this is a defensive branch.
            // If the id is unresolved there is no stream we can
            // synchronize on; treat the bytes as definitely freed —
            // the only consistent accounting is to release them and
            // let the caller surface any subsequent error against a
            // known stream.
            None => Ok(()),
        })
    }
}

impl AsyncCudaResource {
    /// Drain pending per-stream entries and synchronize each
    /// drained stream via `sync_stream`, releasing only the bytes
    /// for streams that the closure successfully synchronized.
    ///
    /// On the first synchronization failure, the failing entry and
    /// **every remaining un-iterated drained entry** are restored
    /// into `pending_per_stream` so a subsequent reap can retry
    /// them, and `pending_bytes` is decremented only by the
    /// already-synchronized total. The closure's error is then
    /// returned to the caller. Without this recovery, a transient
    /// driver error mid-reap would lose track of pending bytes
    /// forever (drained map is gone, `pending_bytes` still counts
    /// them, but no stream is queued for a future reap).
    ///
    /// Production callers go through [`reap_pending`]
    /// (the trait method), which passes a closure that resolves
    /// the [`StreamId`] against [`StreamPool`] and calls
    /// `CudaStream::synchronize`. This helper exists so unit tests
    /// can inject controlled sync failures without touching the
    /// CUDA driver.
    pub(crate) fn reap_pending_with<F>(&self, mut sync_stream: F) -> ResourceResult<()>
    where
        F: FnMut(StreamId) -> ResourceResult<()>,
    {
        // Drain the per-stream map atomically. Anything added by a
        // racing `deallocate` after this point lands in a fresh
        // entry and waits for the next reap.
        //
        // Critically, we do NOT touch `pending_bytes` here — only
        // after a stream has synchronized do we subtract its bytes.
        // A `deallocate` that races between our drain and our
        // subtract has already added to `pending_bytes` under the
        // same mutex (see `deallocate`), and that addition is
        // preserved because we `fetch_sub` the synchronized total
        // rather than `store(0)`.
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

        let mut synced_total: usize = 0;
        let mut failure: Option<ResourceError> = None;
        let mut unsynced: Vec<(StreamId, usize)> = Vec::new();
        let mut iter = drained.into_iter();
        while let Some((stream_id, bytes)) = iter.next() {
            match sync_stream(stream_id) {
                Ok(()) => {
                    synced_total = synced_total.saturating_add(bytes);
                }
                Err(e) => {
                    // Restore the failing entry and every remaining
                    // drained entry so they can be retried by a
                    // future reap.
                    unsynced.push((stream_id, bytes));
                    unsynced.extend(iter.by_ref());
                    failure = Some(e);
                    break;
                }
            }
        }

        if !unsynced.is_empty() {
            let mut per_stream = self
                .pending_per_stream
                .lock()
                .expect("AsyncCudaResource pending_per_stream poisoned");
            for (stream_id, bytes) in unsynced {
                *per_stream.entry(stream_id).or_insert(0) += bytes;
            }
        }

        if synced_total > 0 {
            self.pending_bytes
                .fetch_sub(synced_total, Ordering::Relaxed);
        }

        match failure {
            Some(e) => Err(e),
            None => Ok(()),
        }
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

    /// Test-only helper: install pending state directly so we can
    /// exercise `reap_pending_with` without going through real
    /// CUDA streams. Bypasses the normal `allocate`/`deallocate`
    /// path; intended exclusively for the failure-recovery test.
    fn install_pending(r: &AsyncCudaResource, entries: &[(StreamId, usize)]) {
        let mut per_stream = r
            .pending_per_stream
            .lock()
            .expect("AsyncCudaResource pending_per_stream poisoned");
        let mut total: usize = 0;
        for (id, bytes) in entries {
            *per_stream.entry(*id).or_insert(0) += *bytes;
            total = total.saturating_add(*bytes);
        }
        drop(per_stream);
        r.pending_bytes.fetch_add(total, Ordering::Relaxed);
    }

    #[test]
    fn reap_pending_recovers_unsynced_streams_when_sync_fails() {
        // No CUDA needed for the recovery semantics — we use the
        // real AsyncCudaResource (constructor needs a device only)
        // and inject sync failures via `reap_pending_with`.
        let Some((device, pool)) = try_setup() else {
            return;
        };
        let r = AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool));

        // Install two pending entries: the test will fail sync for
        // StreamId(2). Bytes total 3072.
        install_pending(&r, &[(StreamId(1), 1024), (StreamId(2), 2048)]);
        assert_eq!(r.pending_free_bytes(), 3072);
        assert_eq!(r.pending_per_stream_total(), 3072);

        // Track which streams the closure successfully synchronized.
        // HashMap iteration order is unspecified, so an
        // order-independent assertion uses this set: the test must
        // hold for any iteration order.
        let synced = std::sync::Mutex::new(Vec::<StreamId>::new());
        let result = r.reap_pending_with(|stream_id| {
            if stream_id == StreamId(2) {
                Err(ResourceError::Driver(
                    "simulated sync failure on StreamId(2)".into(),
                ))
            } else {
                synced.lock().unwrap().push(stream_id);
                Ok(())
            }
        });

        assert!(matches!(result, Err(ResourceError::Driver(_))));

        let synced = synced.into_inner().unwrap();
        // Iteration order [1,2]: 1 syncs ok, 2 fails → synced=[1],
        //   synced_total=1024, pending_bytes=2048, map=[(2,2048)].
        // Iteration order [2,1]: 2 fails first, break aborts → synced=[],
        //   synced_total=0, pending_bytes=3072, map=[(1,1024),(2,2048)].
        // Both must satisfy: pending == 3072 - synced_bytes.
        let synced_bytes: usize = if synced.contains(&StreamId(1)) {
            1024
        } else {
            0
        };
        let expected_pending = 3072 - synced_bytes;
        assert_eq!(
            r.pending_free_bytes(),
            expected_pending,
            "synced={:?}; pending_bytes must reflect only un-synced bytes",
            synced
        );
        assert_eq!(
            r.pending_per_stream_total(),
            expected_pending,
            "synced={:?}; pending_per_stream_total must equal pending_free_bytes \
             (cross-counter invariant)",
            synced
        );

        // A second reap with a closure that succeeds for everything
        // must drain the rest cleanly — proves the restored entries
        // are retried, not lost.
        r.reap_pending_with(|_| Ok(())).expect("retry reap");
        assert_eq!(r.pending_free_bytes(), 0);
        assert_eq!(r.pending_per_stream_total(), 0);
    }

    #[test]
    fn reap_pending_drains_normally_when_sync_always_succeeds() {
        // Sanity: closure-based variant of the success path. Proves
        // the new factoring hasn't regressed the happy case.
        let Some((device, pool)) = try_setup() else {
            return;
        };
        let r = AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool));

        install_pending(&r, &[(StreamId(1), 256), (StreamId(2), 512)]);
        r.reap_pending_with(|_| Ok(())).expect("reap");
        assert_eq!(r.pending_free_bytes(), 0);
        assert_eq!(r.pending_per_stream_total(), 0);
    }
}
