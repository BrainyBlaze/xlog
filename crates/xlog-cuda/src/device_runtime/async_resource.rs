//! Stream-bound allocation through the shared raw storage owner.
//!
//! Producer-ready events are created before malloc and recorded before block
//! publication. Failed initialization retains actual storage and its byte charge.
//! Deallocation moves real owners into the existing pending queue. Cold reaping
//! proves physical free before releasing accounting; pool IDs alone are not
//! completion evidence. Device/sanitizer qualification remains a separate gate.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::memory::RawDeviceAllocation;

#[cfg(test)]
use super::resource::AllocTag;
use super::resource::{
    Access, AllocationAccounting, AllocationRequest, BlockId, BlockState, DeviceAccessDependencies,
    DeviceBlock, DeviceMemoryResource, Generation, ResourceBudgetSnapshot, ResourceError,
    ResourceResult, StreamId,
};
use super::stream_pool::StreamPool;
use crate::CudaDevice;

/// A live raw allocation, its generation, and the authoritative dependency
/// owner shared by storage aliases. Read and write frontiers retain every
/// recorded execution stream until its ordering is proved. Deallocation moves
/// this owner to the pending queue; physical release waits those frontiers.
/// Generation checks reject stale identities before mutating the live entry.
struct LiveEntry {
    slice: Arc<RawDeviceAllocation>,
    generation: Generation,
    alloc_stream: StreamId,
    /// The same event history used by storage aliases. Its initial writer is
    /// the allocation-ready event: cuMemAllocAsync orders allocation only on
    /// the allocation stream. Exact CUDA stream/context identity, rather than
    /// this resource's pool-local index, determines which waits can be skipped.
    dependencies: Arc<DeviceAccessDependencies>,
}

/// Stream-bound allocator with explicitly owned deferred reclamation.
pub struct AsyncCudaResource {
    device: Arc<CudaDevice>,
    device_ordinal: u32,
    stream_pool: Arc<StreamPool>,
    /// Live raw owners keyed by device pointer. Deallocation transfers the
    /// complete owner and dependency history into `pending_per_stream`.
    live: Mutex<HashMap<u64, LiveEntry>>,
    /// Bytes for blocks currently in `live`. Always accurate.
    live_bytes: AtomicUsize,
    /// Includes allocations retained after failed initialization, before a
    /// public block exists. The raw owner releases this counter only after free.
    outstanding_bytes: Arc<AllocationAccounting>,
    /// Bytes transferred to pending reclamation but not yet proven freed.
    /// Charged to each exact allocation ticket, independently of queue or
    /// backend lifetime. Retained stream handles alone do not count as bytes.
    pending_bytes: Arc<AtomicUsize>,
    /// Actual owners awaiting physical reclamation, grouped by stream. A reap
    /// removes only its own batch; concurrently queued owners remain here until
    /// the next reap. Unfinished owners are restored; their tickets settle
    /// physical bytes independently of any later handle-cleanup failure.
    pending_per_stream: Mutex<HashMap<StreamId, Vec<Option<Arc<RawDeviceAllocation>>>>>,
}

impl Drop for AsyncCudaResource {
    fn drop(&mut self) {
        // Backend destruction must not bypass the access history merely because
        // no DeviceStorage wrapper remains (for example, a bare recorded block).
        // Move real native allocations and their context/stream owners out before
        // scheduling cold cleanup; no live-map or admission mutex is held there.
        let live = std::mem::take(
            self.live
                .get_mut()
                .unwrap_or_else(|error| error.into_inner()),
        );
        let pending = std::mem::take(
            self.pending_per_stream
                .get_mut()
                .unwrap_or_else(|error| error.into_inner()),
        );
        // The last raw owner performs its own dependency wait and retryable
        // physical release. A second batch-level wait here would strand every
        // allocation if it failed before handing them to that canonical owner.
        // Outstanding leases keep their exact raw allocation alive independently.
        drop(live);
        drop(pending);
    }
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
            outstanding_bytes: Arc::default(),
            pending_bytes: Arc::new(AtomicUsize::new(0)),
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
    /// invariant holds after successful queue handoff at quiescent moments.
    /// After a failed handoff, the canonical cold queue can own charged bytes
    /// outside this map. Physically freed owners retained for handle cleanup
    /// are excluded from both physical-byte counts.
    pub fn pending_per_stream_total(&self) -> usize {
        let map = self
            .pending_per_stream
            .lock()
            .expect("AsyncCudaResource pending_per_stream poisoned");
        map.values()
            .flatten()
            .flatten()
            .filter(|owner| !owner.reclamation().was_released())
            .map(|owner| owner.len())
            .sum()
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
        live.get(&ptr)
            .map(|entry| entry.dependencies.pending_event_count())
    }
}

impl DeviceMemoryResource for AsyncCudaResource {
    fn materialize(&self, request: AllocationRequest) -> ResourceResult<DeviceBlock> {
        let AllocationRequest {
            bytes,
            stream,
            tag,
            reclamation,
            ..
        } = request;
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
        reclamation.attach_resource(Arc::clone(&self.outstanding_bytes), bytes)?;
        let allocation = crate::memory::RawDeviceAllocation::allocate(
            Arc::clone(&cu_stream),
            bytes,
            None,
            Arc::clone(&reclamation),
        )?;
        let ptr = allocation.ptr();
        let dependencies = allocation.dependencies();
        let mut live = self
            .live
            .lock()
            .expect("AsyncCudaResource live map poisoned");
        if live.contains_key(&ptr) {
            return Err(
                ResourceError::Driver(format!("allocation pointer collision: {ptr:#x}"))
                    .retaining(bytes, reclamation),
            );
        }
        let generation = Generation::next();
        live.insert(
            ptr,
            LiveEntry {
                slice: allocation,
                generation,
                alloc_stream: stream,
                dependencies,
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
        let _alloc_stream = self
            .stream_pool
            .resolve(block.alloc_stream)
            .ok_or_else(|| {
                ResourceError::StreamMisuse(format!(
                    "AsyncCudaResource::deallocate: alloc_stream StreamId({}) does not resolve",
                    block.alloc_stream.0
                ))
            })?;

        // Logical detach validates the complete identity but submits no free
        // or dependency waits. Existing storage leases remain usable. The
        // unique physical owner later reserves and waits its exact range.
        let (slice, dependencies) = {
            let mut live = self
                .live
                .lock()
                .expect("AsyncCudaResource live map poisoned");
            match live.get(&block.ptr) {
                Some(entry) => {
                    BlockId::from_block(&block).validate_allocation(
                        block.bytes,
                        BlockId {
                            ptr: block.ptr,
                            generation: entry.generation,
                            alloc_stream: entry.alloc_stream,
                            device_ordinal: self.device_ordinal,
                        },
                        entry.slice.len(),
                    )?;
                    entry
                        .slice
                        .reclamation()
                        .attach_pending(Arc::clone(&self.pending_bytes), block.bytes)?;
                    let LiveEntry {
                        slice,
                        dependencies,
                        ..
                    } = live
                        .remove(&block.ptr)
                        .expect("present under lock per get above");
                    self.live_bytes.fetch_sub(block.bytes, Ordering::Relaxed);
                    (slice, dependencies)
                }
                None => {
                    return Err(ResourceError::UseAfterFree {
                        generation: block.generation,
                    });
                }
            }
        };

        // Keep the real owner in the existing pending-free queue until cold
        // reclamation proves free. A byte tally alone cannot retain CUDA events.
        let mut pending = self
            .pending_per_stream
            .lock()
            .expect("pending allocation queue poisoned");
        pending
            .entry(block.alloc_stream)
            .or_default()
            .push(Some(slice));
        drop(pending);
        drop(dependencies);
        Ok(())
    }

    fn device_ordinal(&self) -> u32 {
        self.device_ordinal
    }

    fn bytes_outstanding(&self) -> usize {
        self.outstanding_bytes.snapshot().0
    }

    fn allocation_accounting(&self) -> Arc<AllocationAccounting> {
        Arc::clone(&self.outstanding_bytes)
    }

    fn budget_snapshot(&self) -> Option<ResourceBudgetSnapshot> {
        None
    }

    fn reap_pending(&self) -> ResourceResult<()> {
        // Each pending owner authenticates and waits its own context/event
        // history. Pool-local default-stream IDs are not completion evidence.
        self.reap_pending_with(|_| Ok(()))
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

    fn access_dependencies(
        &self,
        block: BlockId,
        bytes: usize,
    ) -> ResourceResult<Option<Arc<DeviceAccessDependencies>>> {
        let live = self
            .live
            .lock()
            .expect("AsyncCudaResource live map poisoned");
        let entry = live.get(&block.ptr).ok_or(ResourceError::UseAfterFree {
            generation: block.generation,
        })?;
        block.validate_allocation(
            bytes,
            BlockId {
                ptr: block.ptr,
                generation: entry.generation,
                alloc_stream: entry.alloc_stream,
                device_ordinal: self.device_ordinal,
            },
            entry.slice.len(),
        )?;
        Ok(Some(Arc::clone(&entry.dependencies)))
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
        entry.dependencies.prepare(&use_cu_stream, access)
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
        // Keep validation and publication atomic with backend deallocation,
        // including direct resource callers without a runtime reservation.
        let live = self
            .live
            .lock()
            .expect("AsyncCudaResource live map poisoned");
        let entry = match live.get(&block.ptr) {
            Some(entry) if entry.generation == block.generation => entry,
            _ => {
                return Err(ResourceError::UseAfterFree {
                    generation: block.generation,
                });
            }
        };
        // Frontier retirement replaces only the same execution stream, whose
        // new event retains that stream/context; it cannot drop their last owner.
        entry.dependencies.record_completion(use_cu_stream, access)
    }
}

impl AsyncCudaResource {
    /// Reap drained actual owners, skipping shared leases. Each unique raw
    /// owner proves its own dependency and physical-free completion. The
    /// per-stream hook is a no-op in production and lets tests inject failure
    /// before a stream's owners are visited.
    ///
    /// Errors and unwind restore every unfinished owner to the same pending
    /// queue. Only proven physical releases decrement `pending_bytes`.
    pub(crate) fn reap_pending_with<F>(&self, mut sync_stream: F) -> ResourceResult<()>
    where
        F: FnMut(StreamId) -> ResourceResult<()>,
    {
        // Drain the per-stream map atomically. Anything added by a
        // racing `deallocate` after this point lands in a fresh
        // entry and waits for the next reap.
        //
        // Each allocation ticket subtracts only its own pending charge after
        // physical release. A concurrent detach or a retained handle owner
        // cannot erase another allocation's bytes.
        let mut pending = crate::memory::PendingReclamationBatch::take(
            &self.pending_per_stream,
            |queue, pending| {
                for (id, owners) in pending.iter_mut() {
                    owners.retain(Option::is_some);
                    if !owners.is_empty() {
                        queue.entry(*id).or_default().append(owners);
                    }
                }
                pending.clear();
            },
        );
        let mut failure = None;
        for (stream_id, allocations) in pending.iter_mut() {
            if let Err(error) = sync_stream(*stream_id) {
                failure = Some(error);
                break;
            }
            if let Err(error) = crate::memory::reap_raw_allocations(allocations) {
                failure = Some(error);
                break;
            }
        }
        // The batch guard restores unfinished owners on return and unwind.
        match failure {
            Some(error) => Err(error),
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
        for (id, bytes) in entries {
            let block = r
                .allocate(*bytes, *id, AllocTag::UNTAGGED)
                .expect("allocate pending owner");
            r.deallocate(block).expect("retire pending owner");
        }
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

        let first = pool.acquire().expect("first allocation stream");
        let second = pool.acquire().expect("second allocation stream");
        assert_eq!((first, second), (StreamId(1), StreamId(2)));

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
