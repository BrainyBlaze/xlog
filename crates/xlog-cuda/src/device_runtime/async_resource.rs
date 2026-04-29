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

use cudarc::driver::{CudaEvent, CudaSlice};

use super::resource::{
    Access, AllocTag, BlockId, BlockState, DeviceBlock, DeviceMemoryResource, Generation,
    ResourceError, ResourceResult, StreamId,
};
use super::stream_pool::StreamPool;
use crate::CudaDevice;

/// One live allocation tracked by [`AsyncCudaResource`]. Carries
/// the cudarc-owned `CudaSlice<u8>` (whose drop queues the
/// underlying `cuMemFreeAsync`) plus access-aware dependency
/// state and the allocation's [`Generation`].
///
/// # Dependency state
///
/// The block's outstanding dependencies are tracked in two
/// distinct sets so future operations can wait on the minimal
/// correct fence:
///
///   * `last_write` — the most recent write event recorded on
///     the block, paired with the stream that recorded it. A
///     subsequent read on a different stream must wait on this
///     event; a subsequent write on a different stream must wait
///     on this event AND every entry in `outstanding_reads`.
///   * `outstanding_reads` — every read event recorded since the
///     current `last_write` was installed (or since allocation,
///     if no write has occurred yet), each paired with its
///     recording stream. A subsequent write on a different
///     stream must wait on each entry here. Cleared at finish
///     time when a new write event replaces `last_write`: the
///     writer's prepare-time waits already subsumed every prior
///     reader's dependency, so any future operation that waits
///     on the new `last_write` transitively observes those
///     reads' completion.
///
/// On `deallocate`, the alloc stream waits on `last_write` (if
/// any) AND every entry in `outstanding_reads` before the queued
/// `cuMemFreeAsync` runs.
///
/// # ABA / generation guard
///
/// The `generation` field guards against address recycling:
/// every API that mutates the entry validates
/// `block.generation == entry.generation` before touching it.
/// Mismatch returns [`ResourceError::UseAfterFree`] and the
/// entry is unchanged.
struct LiveEntry {
    slice: CudaSlice<u8>,
    generation: Generation,
    /// Most recent write event on this block, OR the
    /// allocation-ready event if no write has happened yet.
    /// Future reads/writes on a different stream wait on this
    /// event. Replaced by `finish_block_use` for
    /// `Access::Write` / `Access::ReadWrite`. The
    /// allocation-ready seed exists because cuMemAllocAsync
    /// orders the allocation only on `alloc_stream` — a
    /// cross-stream consumer that submits a kernel before
    /// allocation completes would read pool-recycled garbage.
    last_write: Option<(StreamId, CudaEvent)>,
    /// Read events recorded since `last_write` was installed
    /// (or since allocation). Future writes on a different
    /// stream wait on each entry. Cleared by `finish_block_use`
    /// when a write replaces `last_write`.
    outstanding_reads: Vec<(StreamId, CudaEvent)>,
}

/// Stream-ordered cudarc-backed allocator.
pub struct AsyncCudaResource {
    device: Arc<CudaDevice>,
    device_ordinal: u32,
    stream_pool: Arc<StreamPool>,
    /// Live allocations keyed by raw device pointer. Each entry
    /// holds the cudarc slice and any recorded last-use events
    /// from cross-stream consumers. Removed on deallocate; the
    /// slice is then dropped, queueing `cuMemFreeAsync` on its
    /// bound stream — *after* the stream has been told to wait on
    /// every recorded event.
    live: Mutex<HashMap<u64, LiveEntry>>,
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

    /// Number of recorded outstanding-read events plus a
    /// last_write event (0 or 1) currently attached to the live
    /// block at `ptr`. Test/diagnostic accessor — used by
    /// reproducers to confirm `finish_block_use` actually
    /// attached events before deallocate consumed them. Returns
    /// `None` if `ptr` is not currently in the live map.
    pub fn pending_use_event_count(&self, ptr: u64) -> Option<usize> {
        let live = self
            .live
            .lock()
            .expect("AsyncCudaResource live map poisoned");
        live.get(&ptr).map(|e| {
            e.outstanding_reads.len() + if e.last_write.is_some() { 1 } else { 0 }
        })
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

        // Record an "allocation-ready" event on the alloc stream
        // immediately after the cuMemAllocAsync call. Cross-
        // stream consumers MUST wait on this event before
        // touching the bytes, otherwise the launch (on a
        // different stream) may begin before the allocation
        // completes and read pre-init / pool-recycled garbage.
        // We store it in `last_write` so the access-aware
        // prepare path's existing read-waits-on-last_write and
        // write-waits-on-last_write rules cover it for free.
        // Same-stream consumers skip the wait (already ordered).
        let alloc_event = cu_stream.record_event(None).map_err(|e| {
            ResourceError::Driver(format!(
                "AsyncCudaResource::allocate: record allocation-ready event failed: {}",
                e
            ))
        })?;

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
            // Generation must match between the LiveEntry and the
            // returned DeviceBlock so record_block_use and
            // deallocate can ABA-validate by (ptr, generation).
            let generation = Generation::next();
            live.insert(
                ptr,
                LiveEntry {
                    slice,
                    generation,
                    last_write: Some((stream, alloc_event)),
                    outstanding_reads: Vec::new(),
                },
            );
            self.live_bytes.fetch_add(bytes, Ordering::Relaxed);
            Ok(DeviceBlock {
                ptr,
                device_ordinal: self.device_ordinal,
                alloc_stream: stream,
                bytes,
                align: std::mem::align_of::<u8>(),
                tag,
                generation,
                state: BlockState::Live,
            })
        }
    }

    fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()> {
        if block.device_ordinal != self.device_ordinal {
            return Err(ResourceError::Driver(format!(
                "AsyncCudaResource: deallocate on wrong device (block ord {} vs resource ord {})",
                block.device_ordinal, self.device_ordinal
            )));
        }
        // Resolve the alloc stream FIRST. If resolution fails the
        // live entry stays in place and accounting is unchanged —
        // the caller can retry. Removing the entry first then
        // erroring would queue `cuMemFreeAsync` on a stream the
        // caller did not expect (via the slice drop on the error
        // return path) AND leave accounting drift behind.
        let alloc_stream = self
            .stream_pool
            .resolve(block.alloc_stream)
            .ok_or_else(|| {
                ResourceError::StreamMisuse(format!(
                    "AsyncCudaResource::deallocate: alloc_stream StreamId({}) does not resolve",
                    block.alloc_stream.0
                ))
            })?;

        // Take the live-map lock and validate (ptr, generation)
        // before removing. The generation guard closes the ABA
        // window: if the address was freed and reused, the older
        // block's deallocate must NOT tear down the new live
        // entry. Mismatch -> UseAfterFree, no mutation.
        //
        // While the entry is still in the map, queue waits on
        // alloc_stream for: the block's last_write (if any) and
        // every outstanding_read. cudarc's `wait` records the
        // dependency synchronously; if any wait call fails, the
        // events stay owned by the entry, the entry stays in the
        // map, and accounting is untouched — caller can retry.
        //
        // Same-stream waits are skipped — events recorded on
        // `block.alloc_stream` are already ordered before
        // anything else queued there, so requesting a wait would
        // just be busywork. Cross-stream events are the ones
        // that fence the queued cuMemFreeAsync against in-flight
        // consumers.
        //
        // Only after every wait succeeds do we remove the entry,
        // taking ownership of the slice and events, and exit the
        // lock. From that point removal is committed and the
        // slice drop below queues cuMemFreeAsync correctly
        // ordered after every wait we just submitted.
        let (slice, last_write, outstanding_reads) = {
            let mut live = self
                .live
                .lock()
                .expect("AsyncCudaResource live map poisoned");
            match live.get(&block.ptr) {
                Some(entry) if entry.generation == block.generation => {
                    if let Some((write_stream, event)) = &entry.last_write {
                        if *write_stream != block.alloc_stream {
                            alloc_stream.wait(event).map_err(|e| {
                                ResourceError::Driver(format!(
                                    "AsyncCudaResource::deallocate: cuStreamWaitEvent on \
                                     last_write failed: {}",
                                    e
                                ))
                            })?;
                        }
                    }
                    for (read_stream, event) in &entry.outstanding_reads {
                        if *read_stream != block.alloc_stream {
                            alloc_stream.wait(event).map_err(|e| {
                                ResourceError::Driver(format!(
                                    "AsyncCudaResource::deallocate: cuStreamWaitEvent on \
                                     outstanding read failed: {}",
                                    e
                                ))
                            })?;
                        }
                    }
                    let LiveEntry {
                        slice,
                        last_write,
                        outstanding_reads,
                        ..
                    } = live
                        .remove(&block.ptr)
                        .expect("present under lock per get above");
                    (slice, last_write, outstanding_reads)
                }
                Some(_) | None => {
                    return Err(ResourceError::UseAfterFree {
                        generation: block.generation,
                    });
                }
            }
        };

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
        // DeviceBlock's `alloc_stream` — and now also waits for
        // every recorded cross-stream use event we just queued
        // above.
        drop(slice);
        // Drop the events explicitly after the slice drop has
        // queued the free. The event handles can be released as
        // soon as the wait calls return — cudarc's `wait` records
        // the dependency in the stream and does not retain the
        // event.
        drop(last_write);
        drop(outstanding_reads);
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

    fn supports_block_use_tracking(&self) -> bool {
        true
    }

    fn record_block_use(&self, block: &DeviceBlock, use_stream: StreamId) -> ResourceResult<()> {
        // Backward-compatibility shim. Pre-migration callers used
        // `record_block_use` for "this stream did SOMETHING with
        // this block; please wait on me before freeing." That
        // semantics maps to `finish_block_use(.., Access::Read)`:
        // the event is recorded on `use_stream` and appended to
        // outstanding_reads so deallocate waits on it. New
        // callers MUST call `prepare_block_use` BEFORE the launch
        // and `finish_block_use` after; this shim does NOT queue
        // the pre-launch wait so it is unsafe for use-after-write
        // / use-after-prior-read scenarios.
        self.finish_block_use(BlockId::from_block(block), use_stream, Access::Read)
    }

    fn prepare_block_use(
        &self,
        block: BlockId,
        use_stream: StreamId,
        access: Access,
    ) -> ResourceResult<()> {
        if block.device_ordinal != self.device_ordinal {
            return Err(ResourceError::Driver(format!(
                "AsyncCudaResource::prepare_block_use: block device {} != resource device {}",
                block.device_ordinal, self.device_ordinal
            )));
        }
        let use_cu_stream = self.stream_pool.resolve(use_stream).ok_or_else(|| {
            ResourceError::StreamMisuse(format!(
                "AsyncCudaResource::prepare_block_use: unknown StreamId({})",
                use_stream.0
            ))
        })?;

        // Validate (ptr, generation) and queue cross-stream
        // waits while holding the live-map lock. The waits are
        // cuStreamWaitEvent calls which record a dependency in
        // the use stream and return — they don't block, so the
        // lock is held only briefly. Same-stream events are
        // skipped (already ordered).
        let live = self
            .live
            .lock()
            .expect("AsyncCudaResource live map poisoned");
        let entry = match live.get(&block.ptr) {
            Some(entry) if entry.generation == block.generation => entry,
            Some(_) | None => {
                return Err(ResourceError::UseAfterFree {
                    generation: block.generation,
                });
            }
        };
        if access.reads() || access.writes() {
            // Reader: wait on prior write.
            // Writer / RW: wait on prior write AND every prior reader.
            if let Some((write_stream, event)) = &entry.last_write {
                if *write_stream != use_stream {
                    use_cu_stream.wait(event).map_err(|e| {
                        ResourceError::Driver(format!(
                            "AsyncCudaResource::prepare_block_use: wait on last_write failed: {}",
                            e
                        ))
                    })?;
                }
            }
        }
        if access.writes() {
            for (read_stream, event) in &entry.outstanding_reads {
                if *read_stream != use_stream {
                    use_cu_stream.wait(event).map_err(|e| {
                        ResourceError::Driver(format!(
                            "AsyncCudaResource::prepare_block_use: wait on outstanding read \
                             failed: {}",
                            e
                        ))
                    })?;
                }
            }
        }
        Ok(())
    }

    fn finish_block_use(
        &self,
        block: BlockId,
        use_stream: StreamId,
        access: Access,
    ) -> ResourceResult<()> {
        if block.device_ordinal != self.device_ordinal {
            return Err(ResourceError::Driver(format!(
                "AsyncCudaResource::finish_block_use: block device {} != resource device {}",
                block.device_ordinal, self.device_ordinal
            )));
        }
        let use_cu_stream = self.stream_pool.resolve(use_stream).ok_or_else(|| {
            ResourceError::StreamMisuse(format!(
                "AsyncCudaResource::finish_block_use: unknown StreamId({})",
                use_stream.0
            ))
        })?;
        // Validate (ptr, generation) BEFORE recording the event
        // on `use_stream`. This avoids creating an event that we
        // would have to immediately destroy on the ABA failure
        // path.
        {
            let live = self
                .live
                .lock()
                .expect("AsyncCudaResource live map poisoned");
            match live.get(&block.ptr) {
                Some(entry) if entry.generation == block.generation => {}
                Some(_) | None => {
                    return Err(ResourceError::UseAfterFree {
                        generation: block.generation,
                    });
                }
            }
        }
        // Record the event on the use stream OUTSIDE the live-map
        // lock — event creation/record can block on the CUDA
        // driver and we don't want to hold the live-map lock
        // across that. Re-validate generation after acquiring the
        // lock so a racing dealloc that already removed the entry
        // doesn't see a phantom event attached to a stale block.
        let event = use_cu_stream.record_event(None).map_err(|e| {
            ResourceError::Driver(format!(
                "AsyncCudaResource::finish_block_use: event record failed: {}",
                e
            ))
        })?;
        let mut live = self
            .live
            .lock()
            .expect("AsyncCudaResource live map poisoned");
        match live.get_mut(&block.ptr) {
            Some(entry) if entry.generation == block.generation => {
                if access.writes() {
                    // Writer: the prepare phase queued waits on
                    // every prior reader and on last_write, so
                    // any future op that observes the new
                    // last_write transitively observes those
                    // dependencies. Drop the prior state.
                    entry.last_write = Some((use_stream, event));
                    entry.outstanding_reads.clear();
                } else {
                    debug_assert!(access.reads());
                    entry.outstanding_reads.push((use_stream, event));
                }
                Ok(())
            }
            Some(_) | None => {
                // Event drops here, releasing the CUDA event.
                // cudarc's wait was never queued so no stream
                // dependency leaks.
                drop(event);
                Err(ResourceError::UseAfterFree {
                    generation: block.generation,
                })
            }
        }
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
