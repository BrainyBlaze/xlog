//! Core [`DeviceMemoryResource`] trait and supporting types.
//!
//! Mirrors RMM's `device_memory_resource` shape so a future optional
//! RMM backend can satisfy the same trait without requiring callers to
//! change. Stream-ordered: every alloc/dealloc names a stream; cross-
//! stream reuse requires explicit event-based synchronization.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaEvent, CudaStream};

/// Identifier for a CUDA stream owned by the runtime's stream pool.
/// Wraps the raw cudarc stream handle the resource will use for
/// `cuMemAllocAsync` / `cuMemFreeAsync` ordering. Construction goes
/// through the runtime; do not fabricate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct StreamId(pub u32);

impl StreamId {
    /// The "default" pool stream for tests and synchronous codepaths
    /// that have no other stream context. Production callers should
    /// always carry a real stream from the executor / kernel launch
    /// site.
    pub const DEFAULT: StreamId = StreamId(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StorageDropProbe {
        drops: Arc<std::sync::atomic::AtomicUsize>,
        released: Arc<std::sync::atomic::AtomicBool>,
        registry: std::sync::Weak<Mutex<BlockUseRegistry>>,
    }

    impl MemoryStorageOwner for StorageDropProbe {
        fn dependencies(&self) -> ResourceResult<Arc<DeviceAccessDependencies>> {
            panic!("host ownership test must not initialize CUDA dependencies")
        }
    }

    impl Drop for StorageDropProbe {
        fn drop(&mut self) {
            if let Some(registry) = self.registry.upgrade() {
                assert!(
                    registry.try_lock().is_ok(),
                    "last storage owner dropped under registry lock"
                );
            }
            self.drops.fetch_add(1, Ordering::SeqCst);
            self.released.store(true, Ordering::Release);
        }
    }

    struct HostOperationOwners {
        registry: Arc<Mutex<BlockUseRegistry>>,
        _retained: Vec<RetainedStorageUse>,
        completion_ready: std::sync::atomic::AtomicBool,
    }

    impl HostOperationOwners {
        fn cancel(&self, groups: &[MemoryUseGroup]) -> ResourceResult<()> {
            assert_eq!(groups.len(), 1);
            self.registry.lock().unwrap().release_memory_uses(groups[0])
        }
    }

    impl crate::launch::RecorderCleanup<MemoryUseGroup> for HostOperationOwners {
        fn synchronize_retired(&self) -> ResourceResult<()> {
            if self.completion_ready.load(Ordering::Acquire) {
                Ok(())
            } else {
                Err(ResourceError::Driver(
                    "host operation completion remains unknown".into(),
                ))
            }
        }

        fn cancel_retired(&self, group: MemoryUseGroup) -> ResourceResult<()> {
            self.cancel(&[group])
        }
    }

    fn admitted_owner_operation() -> (
        Arc<Mutex<BlockUseRegistry>>,
        Arc<std::sync::atomic::AtomicUsize>,
        crate::launch::RecorderTransaction<HostOperationOwners, MemoryUseGroup>,
    ) {
        let registry = Arc::new(Mutex::new(BlockUseRegistry::default()));
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut external = Vec::new();
        // Distinct allocation/import wrappers over overlapping address ranges
        // must both be retained, independent of any wrapper/generation IDs.
        for range in [
            span(0x1000, 64, Access::ReadWrite),
            span(0x1020, 16, Access::ReadWrite),
        ] {
            let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let storage: Arc<dyn MemoryStorageOwner> = Arc::new(StorageDropProbe {
                drops: Arc::clone(&drops),
                released: Arc::clone(&released),
                registry: Arc::downgrade(&registry),
            });
            registry
                .lock()
                .unwrap()
                .register_storage(7, range, Arc::downgrade(&storage), released);
            external.push(storage);
        }
        let uses = [span(0x1020, 8, Access::Write)];
        let mut retained = Vec::new();
        let group = {
            let mut registry = registry.lock().unwrap();
            registry
                .retain_storage_uses(7, &uses, &mut retained)
                .unwrap();
            registry.reserve_memory_uses(7, &uses).unwrap()
        };
        let operation = crate::launch::RecorderTransaction::from_admitted(
            Arc::new(HostOperationOwners {
                registry: Arc::clone(&registry),
                _retained: retained,
                completion_ready: std::sync::atomic::AtomicBool::new(false),
            }),
            Box::new([group]),
        );
        drop(external);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        (registry, drops, operation)
    }

    #[test]
    fn completed_operation_releases_all_alias_owners_once_outside_registry() {
        let (registry, drops, mut operation) = admitted_owner_operation();
        operation
            .enqueue_operation_with(
                || Ok(()),
                |_| Ok(()),
                |_| Ok::<(), ResourceError>(()),
                |_| Ok(()),
                HostOperationOwners::cancel,
            )
            .unwrap();
        operation
            .commit_with(
                |owner, group| owner.cancel(&[group]),
                |_| Ok(()),
                HostOperationOwners::cancel,
            )
            .unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        drop(operation);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        registry
            .lock()
            .unwrap()
            .reserve_memory_uses(7, &[span(0x1020, 8, Access::ReadWrite)])
            .unwrap();
    }

    #[test]
    fn unknown_operation_retains_alias_owners_and_blocks_later_conflicting_access() {
        let _serial = crate::cuda_graph::capture_lifecycle_test_guard();
        let (registry, drops, mut operation) = admitted_owner_operation();
        let owner = Arc::downgrade(operation.prepared_owner().unwrap());
        let result = operation.enqueue_operation_with(
            || Ok(()),
            |_| Ok(()),
            |_| Err::<(), _>(ResourceError::Driver("partial enqueue".into())),
            |_| Err(ResourceError::Driver("completion unknown".into())),
            HostOperationOwners::cancel,
        );
        assert!(result.is_err());
        drop(operation);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(registry
            .lock()
            .unwrap()
            .reserve_memory_uses(7, &[span(0x1020, 8, Access::Read)])
            .is_err());
        // Unknown completion retains this same capsule in cold retirement,
        // just as it retains native owners, without cancelling the reservation.
        crate::cuda_graph::reap_capture_retirements();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        owner
            .upgrade()
            .unwrap()
            .completion_ready
            .store(true, Ordering::Release);
        std::thread::spawn(crate::cuda_graph::reap_capture_retirements)
            .join()
            .unwrap();
        assert!(owner.upgrade().is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        let group = registry
            .lock()
            .unwrap()
            .reserve_memory_uses(7, &[span(0x1020, 8, Access::Read)])
            .unwrap();
        registry.lock().unwrap().release_memory_uses(group).unwrap();
    }

    #[test]
    fn unknown_dependency_preparation_retains_owners_before_the_launch_callback() {
        let _serial = crate::cuda_graph::capture_lifecycle_test_guard();
        let (registry, drops, mut operation) = admitted_owner_operation();
        let owner = Arc::downgrade(operation.prepared_owner().unwrap());
        let result = operation.enqueue_operation_with(
            || Ok(()),
            |_| Err(ResourceError::Driver("dependency wait failed".into())),
            |_| -> ResourceResult<()> { panic!("launch must not run after failed dependencies") },
            |_| Err(ResourceError::Driver("completion unknown".into())),
            HostOperationOwners::cancel,
        );
        assert!(matches!(
            result,
            Err(crate::launch::LaunchEnqueueError::PreparationAndCleanup { .. })
        ));
        drop(operation);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(registry
            .lock()
            .unwrap()
            .reserve_memory_uses(7, &[span(0x1020, 8, Access::Write)])
            .is_err());
        owner
            .upgrade()
            .unwrap()
            .completion_ready
            .store(true, Ordering::Release);
        crate::cuda_graph::reap_capture_retirements();
        assert!(owner.upgrade().is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn allocation_dependency_lookup_validates_the_full_allocation_identity() {
        let live = read(0x1000, 3).block;
        live.validate_allocation(64, live, 64).unwrap();
        assert!(live.validate_allocation(63, live, 64).is_err());
        assert!(live.validate_allocation(65, live, 64).is_err());
        for altered in [
            BlockId {
                ptr: 0x2000,
                ..live
            },
            BlockId {
                generation: Generation(2),
                ..live
            },
            BlockId {
                alloc_stream: StreamId(4),
                ..live
            },
            BlockId {
                device_ordinal: 1,
                ..live
            },
        ] {
            assert!(altered.validate_allocation(64, live, 64).is_err());
        }
    }

    #[test]
    fn completed_reclamation_pruning_preserves_unrelated_operation_identity() {
        let mut registry = BlockUseRegistry::default();
        let reclamation = crate::memory::AllocationReclamation::default();
        let block = DeviceBlock {
            ptr: 0x1000,
            bytes: 64,
            align: 8,
            device_ordinal: 0,
            alloc_stream: StreamId::DEFAULT,
            tag: AllocTag::UNTAGGED,
            generation: Generation(1),
            state: BlockState::Live,
        };
        let free = registry
            .reserve_reclamation(
                7,
                MemoryUse::new(block.ptr, block.bytes, Access::ReadWrite).unwrap(),
                reclamation.release_proof(),
            )
            .unwrap();
        let ordinary = registry
            .reserve_memory_uses(7, &[span(0x2000, 64, Access::Write)])
            .unwrap();
        reclamation.complete().unwrap();
        let replacement = registry
            .reserve_memory_uses(7, &[span(0x1000, 64, Access::Write)])
            .unwrap();
        assert!(!registry.prepared_memory.contains_key(&free));
        assert!(registry.prepared_memory.contains_key(&ordinary));
        assert!(registry.prepared_memory.contains_key(&replacement));
        assert_eq!(registry.prepared_memory.len(), 2);
        registry.release_memory_uses(ordinary).unwrap();
        assert!(registry.release_memory_uses(ordinary).is_err());
        assert!(registry
            .reserve_memory_uses(7, &[span(0x1000, 64, Access::Read)])
            .is_err());
        registry.release_memory_uses(replacement).unwrap();
    }

    fn read(ptr: u64, generation: u64) -> BlockUse {
        BlockUse {
            block: BlockId {
                ptr,
                generation: Generation(generation),
                alloc_stream: StreamId(3),
                device_ordinal: 0,
            },
            bytes: 64,
            access: Access::Read,
        }
    }

    #[test]
    fn pending_alias_blocks_allocation_retirement_until_exact_group_release() {
        let mut registry = BlockUseRegistry::default();
        let allocation = span(0x1000, 64, Access::ReadWrite);
        let first = registry
            .reserve_memory_uses(7, &[span(0x1010, 8, Access::Read)])
            .unwrap();
        let second = registry
            .reserve_memory_uses(7, &[span(0x1020, 8, Access::Read)])
            .unwrap();
        assert!(registry.reserve_memory_uses(7, &[allocation]).is_err());
        let other_context = registry.reserve_memory_uses(8, &[allocation]).unwrap();
        registry.release_memory_uses(other_context).unwrap();
        registry.release_memory_uses(first).unwrap();
        assert!(registry.release_memory_uses(first).is_err());
        assert!(registry.reserve_memory_uses(7, &[allocation]).is_err());
        registry.release_memory_uses(second).unwrap();
        let released = registry.reserve_memory_uses(7, &[allocation]).unwrap();
        registry.release_memory_uses(released).unwrap();
    }

    fn span(start: u64, bytes: usize, access: Access) -> MemoryUse {
        MemoryUse::new(start, bytes, access).expect("valid test span")
    }

    #[test]
    fn memory_access_groups_compare_storage_ranges_not_alias_identity() {
        let mut registry = BlockUseRegistry::default();
        let read = span(0x1000, 64, Access::Read);
        let first = registry.reserve_memory_uses(7, &[read]).unwrap();
        let alias = span(0x1020, 64, Access::Read);
        let second = registry.reserve_memory_uses(7, &[alias]).unwrap();
        assert!(registry
            .reserve_memory_uses(7, &[span(0x1030, 8, Access::Write)])
            .is_err());
        registry.release_memory_uses(first).unwrap();
        assert!(registry
            .reserve_memory_uses(7, &[span(0x1040, 8, Access::Write)])
            .is_err());
        registry.release_memory_uses(second).unwrap();
        let writer = registry
            .reserve_memory_uses(7, &[span(0x1030, 8, Access::Write)])
            .unwrap();
        registry.release_memory_uses(writer).unwrap();
    }

    #[test]
    fn peer_copy_reserves_both_contexts_atomically() {
        let mut registry = BlockUseRegistry::default();
        let source = span(0x1000, 16, Access::Read);
        let destination = span(0x1000, 16, Access::Write);
        let occupied = registry.reserve_memory_uses(8, &[destination]).unwrap();
        assert!(registry
            .reserve_context_memory_uses(&[(7, source), (8, destination)])
            .is_err());
        let source_writer = registry.reserve_memory_uses(7, &[destination]).unwrap();
        registry.release_memory_uses(source_writer).unwrap();
        registry.release_memory_uses(occupied).unwrap();
        let peer = registry
            .reserve_context_memory_uses(&[(7, source), (8, destination)])
            .unwrap();
        assert!(registry.reserve_memory_uses(7, &[destination]).is_err());
        assert!(registry.reserve_memory_uses(8, &[source]).is_err());
        let tail = registry
            .reserve_memory_uses(8, &[span(0x1010, 16, Access::Write)])
            .unwrap();
        registry.release_memory_uses(tail).unwrap();
        registry.release_memory_uses(peer).unwrap();
        registry.reserve_memory_uses(7, &[destination]).unwrap();
        registry.reserve_memory_uses(8, &[source]).unwrap();
    }

    #[test]
    fn copy_ranges_reject_overlap_only_in_the_same_context() {
        let source = span(0x1000, 16, Access::Read);
        let overlapping = span(0x1008, 16, Access::Write);
        assert!(source.validate_copy_to(7, overlapping, 7).is_err());
        assert!(source.validate_copy_to(7, source, 7).is_err());
        source.validate_copy_to(7, overlapping, 8).unwrap();
        source
            .validate_copy_to(7, span(0x1010, 16, Access::Write), 7)
            .unwrap();
        let empty = span(0x1000, 0, Access::Write);
        empty.validate_copy_to(7, empty, 7).unwrap();
    }

    #[test]
    fn memory_access_groups_preserve_context_and_disjoint_ranges() {
        let mut registry = BlockUseRegistry::default();
        let writer = span(0x1000, 64, Access::Write);
        let first = registry.reserve_memory_uses(7, &[writer]).unwrap();
        let other_context = registry.reserve_memory_uses(8, &[writer]).unwrap();
        let adjacent = registry
            .reserve_memory_uses(7, &[span(0x1040, 8, Access::Write)])
            .unwrap();
        let empty = registry
            .reserve_memory_uses(7, &[span(0x1010, 0, Access::Write)])
            .unwrap();
        for group in [first, other_context, adjacent, empty] {
            registry.release_memory_uses(group).unwrap();
        }
    }

    #[test]
    fn memory_access_group_admission_and_release_are_atomic() {
        let mut registry = BlockUseRegistry::default();
        let first = registry
            .reserve_memory_uses(7, &[span(0x1000, 64, Access::Write)])
            .unwrap();
        assert!(registry
            .reserve_memory_uses(
                7,
                &[
                    span(0x2000, 64, Access::Write),
                    span(0x1030, 64, Access::Read),
                ],
            )
            .is_err());
        // A failure in the last member must not retain the earlier member.
        let second = registry
            .reserve_memory_uses(7, &[span(0x2000, 64, Access::Write)])
            .unwrap();
        registry.release_memory_uses(second).unwrap();
        assert!(registry.release_memory_uses(second).is_err());
        assert!(registry
            .reserve_memory_uses(7, &[span(0x1000, 64, Access::Read)])
            .is_err());
        registry.release_memory_uses(first).unwrap();
        assert!(MemoryUse::new(u64::MAX, 1, Access::Read).is_err());
    }

    #[test]
    fn memory_access_group_preserves_per_range_access_with_internal_aliases() {
        let mut registry = BlockUseRegistry::default();
        let group = registry
            .reserve_memory_uses(
                7,
                &[
                    span(0x1000, 64, Access::Read),
                    span(0x1020, 16, Access::Write),
                    span(0x2000, 64, Access::Read),
                ],
            )
            .unwrap();
        let reader = registry
            .reserve_memory_uses(7, &[span(0x1000, 16, Access::Read)])
            .unwrap();
        assert!(registry
            .reserve_memory_uses(7, &[span(0x1020, 16, Access::Read)])
            .is_err());
        // Do not turn the gap between two arguments into an exclusive span.
        let gap = registry
            .reserve_memory_uses(7, &[span(0x1800, 64, Access::Write)])
            .unwrap();
        for token in [group, reader, gap] {
            registry.release_memory_uses(token).unwrap();
        }
    }

    #[test]
    fn access_dependencies_keep_published_writes_visible_to_another_stream() {
        let mut dependencies = AccessDependencies::after_write((7, 1), "allocated");
        let retired = dependencies.publish((7, 1), "written", Access::Write);
        drop(retired);
        let mut waited = Vec::new();
        dependencies
            .prepare(&(7, 2), Access::Read, |event| {
                waited.push(*event);
                Ok(())
            })
            .unwrap();
        assert_eq!(waited, ["written"]);
    }

    #[test]
    fn access_dependencies_preserve_other_streams_until_order_is_proved() {
        let mut dependencies = AccessDependencies::after_write((7, 1), "written");
        assert!(dependencies.publish((7, 2), "read", Access::Read).is_none());
        let mut waited = Vec::new();
        dependencies
            .prepare(&(7, 3), Access::Write, |event| {
                waited.push(*event);
                Ok(())
            })
            .unwrap();
        assert_eq!(waited, ["written", "read"]);
        assert!(dependencies
            .publish((7, 3), "rewritten", Access::Write)
            .is_none());
        assert_eq!(dependencies.outstanding_reads, [((7, 2), "read")]);
        let mut waited = Vec::new();
        dependencies
            .prepare(&(7, 4), Access::Read, |event| {
                waited.push(*event);
                Ok(())
            })
            .unwrap();
        assert_eq!(waited, ["written", "rewritten"]);
    }

    #[test]
    fn access_dependencies_skip_only_the_exact_same_stream() {
        let dependencies = AccessDependencies::after_write((7, 1), "written");
        dependencies
            .prepare(&(7, 1), Access::Read, |_| {
                panic!("same-stream order needs no wait")
            })
            .unwrap();
        let mut waits = 0;
        dependencies
            .prepare(&(8, 1), Access::Read, |_| {
                waits += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(
            waits, 1,
            "a pool-local stream index is not a stream identity"
        );
    }

    #[test]
    fn access_dependencies_wait_failure_keeps_all_published_events() {
        let mut dependencies = AccessDependencies::after_write((7, 1), "written");
        dependencies.publish((7, 2), "read", Access::Read);
        assert!(dependencies
            .prepare(&(7, 3), Access::Write, |_| {
                Err(ResourceError::Driver("wait failure".into()))
            })
            .is_err());
        assert_eq!(dependencies.outstanding_writes, [((7, 1), "written")]);
        assert_eq!(dependencies.outstanding_reads, [((7, 2), "read")]);
    }

    #[test]
    fn disjoint_operations_do_not_erase_a_concurrently_published_dependency() {
        let mut registry = BlockUseRegistry::default();
        let reader = registry
            .reserve_memory_uses(7, &[span(0x1000, 16, Access::Read)])
            .unwrap();
        let writer = registry
            .reserve_memory_uses(7, &[span(0x1010, 16, Access::Write)])
            .unwrap();
        let mut dependencies = AccessDependencies::after_write(0, "allocated");
        // Both operations prepare before either completion is published. The
        // writer's event therefore does not imply completion of this reader.
        dependencies.prepare(&1, Access::Read, |_| Ok(())).unwrap();
        dependencies.prepare(&2, Access::Write, |_| Ok(())).unwrap();
        dependencies.publish(1, "left read", Access::Read);
        dependencies.publish(2, "right write", Access::Write);
        registry.release_memory_uses(reader).unwrap();
        registry.release_memory_uses(writer).unwrap();
        let mut waited = Vec::new();
        dependencies
            .prepare(&3, Access::Write, |event| {
                waited.push(*event);
                Ok(())
            })
            .unwrap();
        assert!(waited.contains(&"left read"));
        assert!(waited.contains(&"right write"));
    }

    #[test]
    fn disjoint_writers_keep_both_completion_frontiers() {
        let mut dependencies = AccessDependencies::after_write(0, "allocated");
        dependencies.prepare(&1, Access::Write, |_| Ok(())).unwrap();
        dependencies.prepare(&2, Access::Write, |_| Ok(())).unwrap();
        dependencies.publish(1, "left write", Access::Write);
        dependencies.publish(2, "right write", Access::Write);
        let mut waited = Vec::new();
        dependencies
            .prepare(&3, Access::Read, |event| {
                waited.push(*event);
                Ok(())
            })
            .unwrap();
        assert!(waited.contains(&"left write"));
        assert!(waited.contains(&"right write"));
    }
}

/// Caller-supplied tag for allocation log lines. Short-lived strings
/// are interned by the logging resource; long-lived borrows are not
/// retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AllocTag(pub &'static str);

impl AllocTag {
    pub const UNTAGGED: AllocTag = AllocTag("untagged");
}

/// One allocation admission carried unchanged through every resource decorator.
/// The terminal backend retains its reclamation ticket with the actual storage.
pub struct AllocationRequest {
    pub bytes: usize,
    pub stream: StreamId,
    pub tag: AllocTag,
    /// Bytes promised by the runtime but not yet physically allocated.
    pub reservation_pressure_bytes: usize,
    pub(crate) reclamation: Arc<crate::memory::AllocationReclamation>,
}

impl AllocationRequest {
    pub fn new(bytes: usize, stream: StreamId, tag: AllocTag) -> Self {
        Self {
            bytes,
            stream,
            tag,
            reservation_pressure_bytes: 0,
            reclamation: Arc::default(),
        }
    }

    /// The same lifecycle ticket used by admission and the physical owner.
    pub fn reclamation(&self) -> Arc<crate::memory::AllocationReclamation> {
        Arc::clone(&self.reclamation)
    }
}

#[derive(Debug, Default)]
struct AllocationAccountingState {
    outstanding: usize,
    total_reclaimed: u128,
}

/// Exact physical accounting for one terminal resource and all its decorators.
/// Storage owners publish acquisition and proven release through their tickets.
/// This ledger owns no allocations and has no separate reclamation queue.
#[derive(Debug, Default)]
pub struct AllocationAccounting {
    state: Mutex<AllocationAccountingState>,
}

impl AllocationAccounting {
    /// Atomically snapshot outstanding bytes and cumulative proven releases.
    pub fn snapshot(&self) -> (usize, u128) {
        let state = self.state.lock().expect("allocation accounting poisoned");
        (state.outstanding, state.total_reclaimed)
    }

    /// Publish one acquisition through its lifecycle ticket.
    /// Arithmetic failure leaves the complete ledger unchanged.
    pub(crate) fn acquire(&self, bytes: usize) -> ResourceResult<()> {
        let mut state = self.state.lock().expect("allocation accounting poisoned");
        let outstanding = state.outstanding.checked_add(bytes).ok_or_else(|| {
            ResourceError::Driver("resource allocation accounting overflow".into())
        })?;
        state.outstanding = outstanding;
        Ok(())
    }

    /// Publish a proven physical release through its lifecycle ticket.
    /// Outstanding bytes and the cumulative release total change atomically.
    pub(crate) fn release(&self, bytes: usize) -> ResourceResult<()> {
        let mut state = self.state.lock().expect("allocation accounting poisoned");
        let outstanding = state.outstanding.checked_sub(bytes).ok_or_else(|| {
            ResourceError::Driver("resource release exceeds outstanding bytes".into())
        })?;
        let total_reclaimed = state
            .total_reclaimed
            .checked_add(bytes as u128)
            .ok_or_else(|| {
                ResourceError::Driver("resource reclamation accounting overflow".into())
            })?;
        state.outstanding = outstanding;
        state.total_reclaimed = total_reclaimed;
        Ok(())
    }
}

/// Monotonic counter for distinguishing reuse of the same byte address
/// across drop / reallocate cycles. Logging and debug-guard resources
/// use this to detect use-after-free.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Generation(pub u64);

static GENERATION_COUNTER: AtomicU64 = AtomicU64::new(1);

impl Generation {
    /// Allocate a fresh, monotonically increasing generation number.
    /// Concurrent calls return distinct values.
    pub fn next() -> Generation {
        Generation(GENERATION_COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Access kind for a single block use. Drives the cross-stream
/// dependency edges the resource queues during
/// [`DeviceMemoryResource::prepare_block_use`] and the events it
/// records during [`DeviceMemoryResource::finish_block_use`].
///
///   * [`Access::Read`] — the work consumes the block's bytes.
///     Must wait on any prior write on a different stream. The
///     resulting event advances that stream's read frontier so future writers
///     (and the eventual deallocate) can wait on it.
///   * [`Access::Write`] — the work overwrites the block's bytes
///     unconditionally. Must wait on the block's prior write AND
///     all outstanding reads on different streams. The resulting
///     event advances that stream's write frontier. Other streams' concurrent
///     uses remain visible until their completion is independently ordered.
///   * [`Access::ReadWrite`] — both. Same wait set as `Write`,
///     and the resulting event likewise advances its write frontier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

impl Access {
    /// Whether work of this access kind reads the block's bytes.
    pub fn reads(self) -> bool {
        matches!(self, Access::Read | Access::ReadWrite)
    }

    /// Whether work of this access kind writes the block's bytes.
    pub fn writes(self) -> bool {
        matches!(self, Access::Write | Access::ReadWrite)
    }
}

/// Access-aware event history shared by the allocator and every storage alias.
/// Disjoint subranges may prepare concurrently, so a writer's publication does
/// not subsume another stream's later publication. Only the same ordered stream
/// frontier is replaced. Record and publication must be serialized by the owner.
struct AccessDependencies<K, E> {
    outstanding_writes: Vec<(K, E)>,
    outstanding_reads: Vec<(K, E)>,
}

impl<K: PartialEq, E> AccessDependencies<K, E> {
    fn after_write(stream: K, event: E) -> Self {
        Self {
            outstanding_writes: vec![(stream, event)],
            outstanding_reads: Vec::new(),
        }
    }

    fn prepare(
        &self,
        stream: &K,
        access: Access,
        mut wait: impl FnMut(&E) -> ResourceResult<()>,
    ) -> ResourceResult<()> {
        for (writer, event) in &self.outstanding_writes {
            if writer != stream {
                wait(event)?;
            }
        }
        if access.writes() {
            for (reader, event) in &self.outstanding_reads {
                if reader != stream {
                    wait(event)?;
                }
            }
        }
        Ok(())
    }

    /// Return retired events to the caller so their destruction occurs after
    /// releasing the enclosing state mutex.
    fn publish(&mut self, stream: K, event: E, access: Access) -> Option<Self> {
        let mut retired = Self {
            outstanding_writes: Vec::new(),
            outstanding_reads: Vec::new(),
        };
        if let Some(index) = self
            .outstanding_reads
            .iter()
            .position(|(key, _)| *key == stream)
        {
            retired
                .outstanding_reads
                .push(self.outstanding_reads.swap_remove(index));
        }
        if access.writes() {
            if let Some(index) = self
                .outstanding_writes
                .iter()
                .position(|(key, _)| *key == stream)
            {
                retired
                    .outstanding_writes
                    .push(self.outstanding_writes.swap_remove(index));
            }
            self.outstanding_writes.push((stream, event));
        } else {
            self.outstanding_reads.push((stream, event));
        }
        if retired.outstanding_reads.is_empty() && retired.outstanding_writes.is_empty() {
            None
        } else {
            Some(retired)
        }
    }
}

struct RetainedAccessEvent {
    event: CudaEvent,
    // Keep the exact stream/context alive until the last dependency is retired.
    _stream: Arc<CudaStream>,
}

/// Shared, access-aware CUDA dependencies for an allocation. Resource wrappers
/// forward this same owner; they must not reconstruct its event history.
#[doc(hidden)]
pub struct DeviceAccessDependencies {
    _allocation_context: Arc<CudaContext>,
    state: Mutex<AccessDependencies<u64, Arc<RetainedAccessEvent>>>,
    pub(crate) reclamation: Arc<crate::memory::AllocationReclamation>,
    allocation: std::sync::OnceLock<std::sync::Weak<crate::memory::RawDeviceAllocation>>,
}

impl DeviceAccessDependencies {
    pub(crate) fn bind_allocation(&self, allocation: &Arc<crate::memory::RawDeviceAllocation>) {
        assert!(self.allocation.set(Arc::downgrade(allocation)).is_ok());
    }

    /// Upgrade the actual allocation, not a runtime or dependency proxy. The
    /// caller validates the block identity before this atomic owner acquisition.
    pub(crate) fn retain_allocation(
        &self,
    ) -> ResourceResult<Arc<crate::memory::RawDeviceAllocation>> {
        self.allocation
            .get()
            .and_then(std::sync::Weak::upgrade)
            .ok_or_else(|| {
                ResourceError::StreamMisuse(
                    "device allocation has entered physical retirement".into(),
                )
            })
    }

    pub(crate) fn mark_allocation_released(&self) -> ResourceResult<()> {
        self.reclamation.complete()
    }

    pub(crate) fn allocation_was_released(&self) -> bool {
        self.reclamation.was_released()
    }
    /// Cold final-owner retirement waits on recorded events, not on a raw
    /// per-thread stream handle that may name a different producer thread.
    pub(crate) fn synchronize(&self) -> ResourceResult<()> {
        let fences = {
            let state = self.state.lock().map_err(|_| {
                ResourceError::Driver(
                    "cannot prove completion from poisoned device dependency state".into(),
                )
            })?;
            state
                .outstanding_writes
                .iter()
                .chain(&state.outstanding_reads)
                .map(|(_, fence)| Arc::clone(fence))
                .collect::<Vec<_>>()
        };
        for fence in fences {
            fence.event.synchronize()?;
        }
        Ok(())
    }

    /// A foreign producer has completed its handoff before the unsafe import
    /// boundary. XLOG accesses after that handoff populate this same history.
    pub(crate) fn after_ready(context: Arc<CudaContext>) -> Self {
        Self {
            reclamation: Arc::default(),
            allocation: std::sync::OnceLock::new(),
            _allocation_context: context,
            state: Mutex::new(AccessDependencies {
                outstanding_writes: Vec::new(),
                outstanding_reads: Vec::new(),
            }),
        }
    }
    pub(crate) fn pending_event_count(&self) -> usize {
        let state = self
            .state
            .lock()
            .expect("device access dependencies poisoned");
        state.outstanding_reads.len() + state.outstanding_writes.len()
    }

    pub(crate) fn after_event(
        stream: Arc<CudaStream>,
        execution_id: u64,
        event: CudaEvent,
        reclamation: Arc<crate::memory::AllocationReclamation>,
    ) -> Self {
        Self {
            reclamation,
            allocation: std::sync::OnceLock::new(),
            _allocation_context: Arc::clone(stream.context()),
            state: Mutex::new(AccessDependencies::after_write(
                execution_id,
                Arc::new(RetainedAccessEvent {
                    event,
                    _stream: stream,
                }),
            )),
        }
    }

    pub(crate) fn prepare(&self, stream: &Arc<CudaStream>, access: Access) -> ResourceResult<()> {
        let execution_id = crate::cuda_graph::stream_execution_id(stream)?;
        let state = self
            .state
            .lock()
            .expect("device access dependencies poisoned");
        state.prepare(&execution_id, access, |fence| {
            stream.wait(&fence.event).map_err(|error| {
                ResourceError::Driver(format!("device access dependency wait failed: {error}"))
            })
        })
    }

    pub(crate) fn record_completion(
        &self,
        stream: Arc<CudaStream>,
        access: Access,
    ) -> ResourceResult<()> {
        // Peer copies publish the destination stream's completion as a source
        // read dependency. Events belong to the recording context, which need
        // not be the allocation context; cross-context event waits are valid.
        let execution_id = crate::cuda_graph::stream_execution_id(&stream)?;
        let event = stream.context().new_event(None)?;
        let mut state = self
            .state
            .lock()
            .expect("device access dependencies poisoned");
        // Serialize the actual record with frontier replacement. Otherwise a
        // delayed host publisher could replace a newer same-stream event with
        // its older event, losing work already submitted by another host thread.
        if let Err(error) = event.record(&stream) {
            drop(state);
            return Err(ResourceError::Driver(format!(
                "device access completion event record failed: {error}"
            )));
        }
        let retired = state.publish(
            execution_id,
            Arc::new(RetainedAccessEvent {
                event,
                _stream: stream,
            }),
            access,
        );
        drop(state);
        drop(retired);
        Ok(())
    }

    /// Publish one completion event to every allocation frontier touched by
    /// an admitted operation.
    pub(crate) fn record_operation_completion(
        dependencies: &[(Arc<Self>, Access)],
        stream: Arc<CudaStream>,
    ) -> ResourceResult<()> {
        if dependencies.is_empty() {
            return Ok(());
        }
        let execution_id = crate::cuda_graph::stream_execution_id(&stream)?;
        let event = stream.context().new_event(None)?;
        let mut ordered = dependencies.to_vec();
        ordered.sort_unstable_by_key(|(dependency, _)| Arc::as_ptr(dependency) as usize);
        let mut combined = Vec::<(Arc<Self>, Access)>::with_capacity(ordered.len());
        for (dependency, access) in ordered {
            if let Some((existing, existing_access)) = combined.last_mut() {
                if Arc::ptr_eq(existing, &dependency) {
                    *existing_access = if existing_access.writes() || access.writes() {
                        Access::ReadWrite
                    } else {
                        Access::Read
                    };
                    continue;
                }
            }
            combined.push((dependency, access));
        }
        // Lock every frontier in address order before recording. This keeps
        // publication atomic without allowing two multi-allocation operations
        // to acquire the same frontier set in opposite orders.
        let mut states = Vec::with_capacity(combined.len());
        for (dependency, access) in &combined {
            states.push((
                dependency
                    .state
                    .lock()
                    .expect("device access dependencies poisoned"),
                *access,
            ));
        }
        if let Err(error) = event.record(&stream) {
            drop(states);
            return Err(ResourceError::Driver(format!(
                "device access completion event record failed: {error}"
            )));
        }
        // One stream event proves completion of the whole admitted operation;
        // each allocation frontier retains the same event and stream owner.
        let event = Arc::new(RetainedAccessEvent {
            event,
            _stream: stream,
        });
        let retired = states
            .iter_mut()
            .filter_map(|(state, access)| state.publish(execution_id, Arc::clone(&event), *access))
            .collect::<Vec<_>>();
        drop(states);
        drop(retired);
        Ok(())
    }
}

/// Compact identity of a [`DeviceBlock`] suitable for snapshotting
/// into structures whose lifetime should not be tied to the source
/// slice's borrow. The fields needed to validate `(ptr, generation)`
/// against the resource's live map and to resolve `alloc_stream` for
/// cross-stream waits / dealloc ordering.
///
/// Created via [`BlockId::from_block`]. Pure data; no resource
/// handle, no `Drop`. Cheap to copy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockId {
    pub ptr: u64,
    pub generation: Generation,
    pub alloc_stream: StreamId,
    pub device_ordinal: u32,
}

impl BlockId {
    pub(crate) fn validate_allocation(
        self,
        bytes: usize,
        actual: Self,
        actual_bytes: usize,
    ) -> ResourceResult<()> {
        if self != actual || bytes != actual_bytes {
            return Err(ResourceError::UseAfterFree {
                generation: self.generation,
            });
        }
        Ok(())
    }

    /// Snapshot a [`DeviceBlock`]'s identity. The returned id is
    /// independent of the original block's borrow lifetime; the
    /// runtime's generation guard catches stale ids whose backing
    /// allocation has been recycled.
    pub fn from_block(block: &DeviceBlock) -> Self {
        Self {
            ptr: block.ptr,
            generation: block.generation,
            alloc_stream: block.alloc_stream,
            device_ordinal: block.device_ordinal,
        }
    }
}

/// State of an outstanding [`DeviceBlock`] from the runtime's
/// perspective. Adaptors flip blocks between these states; bug-detection
/// resources reject operations on blocks in an unexpected state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockState {
    /// Returned from `allocate`; safe to read/write on `alloc_stream`
    /// or after a synchronization to another stream.
    Live,
    /// Returned from `deallocate` but still pending kernel completion
    /// on its owning stream. Reuse must wait for stream sync.
    Retired,
    /// Held by `DebugGuardResource` for delayed reuse / canary
    /// validation. Not reissued until the quarantine window passes.
    Quarantined,
    /// Memory has been physically freed. Any further use is a bug.
    Freed,
}

/// One outstanding device-memory allocation. Owned by the caller until
/// returned to its originating resource via
/// [`DeviceMemoryResource::deallocate`].
///
/// Carries the metadata required for stream-ordered correctness and
/// post-mortem debugging: the resource that owns the block, the device
/// ordinal, the stream the allocation is bound to, byte size, alignment,
/// caller tag, generation number, and current state.
#[derive(Debug)]
pub struct DeviceBlock {
    /// Raw device pointer (opaque to safe Rust callers).
    pub ptr: u64,
    /// CUDA ordinal of the device this block lives on.
    pub device_ordinal: u32,
    /// Allocation stream. Reads/writes on a different stream require
    /// explicit synchronization (event wait or device sync).
    pub alloc_stream: StreamId,
    /// Size in bytes. May exceed the caller-requested size when the
    /// resource rounds up for alignment or pool granularity.
    pub bytes: usize,
    /// Alignment in bytes (always ≥ caller request).
    pub align: usize,
    /// Caller-supplied tag, surfaced in allocation logs.
    pub tag: AllocTag,
    /// Monotonic generation. Reused addresses get fresh generations.
    pub generation: Generation,
    /// Current state. Adaptors transition this; tests assert on it.
    pub state: BlockState,
}

/// Errors returned by resource implementations. Distinct variants for
/// the cases stress tests need to pin (out-of-budget vs CUDA driver
/// failure vs use-after-free etc.).
#[derive(Debug)]
pub enum ResourceError {
    /// Allocation succeeded but a later step failed. Its actual owner retains
    /// unresolved cleanup independently of this diagnostic. The same ticket
    /// reports later reclamation; discarding the error does not abandon storage.
    AllocationRetained(RetainedAllocation),
    /// The requested allocation would exceed the resource's budget.
    /// Carries the exact accounting state captured at the rejecting
    /// decision point so callers can report cumulative pressure.
    OutOfBudget {
        requested: usize,
        current: usize,
        remaining: usize,
        limit: usize,
    },
    /// CUDA driver returned an error. Carries the wrapped message.
    Driver(String),
    /// A stream-ordered contract was violated (e.g. dealloc on a
    /// stream that does not match the alloc stream without an
    /// intervening sync).
    StreamMisuse(String),
    /// A debug-guard or logging adaptor detected a use-after-free or
    /// double-free. Hard error in debug builds; surfaced upward.
    UseAfterFree { generation: Generation },
    /// A debug-guard adaptor detected an out-of-bounds write past a
    /// canary boundary.
    OutOfBounds { generation: Generation },
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllocationRetained(owner) => write!(f, "allocation of {} bytes failed after acquisition: {}", owner.bytes, owner.cause),
            Self::OutOfBudget {
                requested,
                current,
                remaining,
                limit,
            } => write!(
                f,
                "out of budget: current {} bytes, requested {} bytes, required {} bytes, limit {} bytes, remaining {} bytes",
                current,
                requested,
                *current as u128 + *requested as u128,
                limit,
                remaining,
            ),
            Self::Driver(msg) => write!(f, "CUDA driver error: {}", msg),
            Self::StreamMisuse(msg) => write!(f, "stream-ordered contract violated: {}", msg),
            Self::UseAfterFree { generation } => {
                write!(f, "use-after-free on generation {:?}", generation)
            }
            Self::OutOfBounds { generation } => {
                write!(f, "out-of-bounds write on generation {:?}", generation)
            }
        }
    }
}

impl std::error::Error for ResourceError {}

/// Diagnostic sharing the allocation's lifecycle ticket. The actual storage
/// owner remains governed by its cold reaper, independently of this error.
pub struct RetainedAllocation {
    bytes: usize,
    cause: Box<ResourceError>,
    reclamation: Arc<crate::memory::AllocationReclamation>,
}

impl fmt::Debug for RetainedAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetainedAllocation")
            .field("bytes", &self.bytes)
            .field("cause", &self.cause)
            .field("released", &self.reclamation.was_released())
            .finish_non_exhaustive()
    }
}

impl ResourceError {
    pub(crate) fn retaining(
        self,
        bytes: usize,
        reclamation: Arc<crate::memory::AllocationReclamation>,
    ) -> Self {
        Self::AllocationRetained(RetainedAllocation {
            bytes,
            cause: Box::new(self),
            reclamation,
        })
    }

    #[cfg(test)]
    pub(crate) fn retained_allocation_bytes(&self) -> usize {
        match self {
            Self::AllocationRetained(owner) => owner.bytes,
            _ => 0,
        }
    }
}

pub type ResourceResult<T> = std::result::Result<T, ResourceError>;

impl From<cudarc::driver::DriverError> for ResourceError {
    fn from(error: cudarc::driver::DriverError) -> Self {
        Self::Driver(error.to_string())
    }
}

/// One immutable block identity and its strongest access for a launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BlockUse {
    pub(crate) block: BlockId,
    /// Complete allocation extent captured from the actual block owner.
    pub(crate) bytes: usize,
    pub(crate) access: Access,
}

/// A checked half-open device byte range and the operation's access to it.
/// Allocation generations and import wrapper identities do not distinguish
/// overlapping storage in one CUDA context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MemoryUse {
    start: u64,
    end: u64,
    access: Access,
}

impl MemoryUse {
    pub(crate) fn access(self) -> Access {
        self.access
    }

    pub(crate) fn new(start: u64, bytes: usize, access: Access) -> ResourceResult<Self> {
        let bytes = u64::try_from(bytes)
            .map_err(|_| ResourceError::Driver("device byte range is not representable".into()))?;
        let end = start.checked_add(bytes).ok_or_else(|| {
            ResourceError::Driver("device byte range overflows its address".into())
        })?;
        Ok(Self { start, end, access })
    }

    pub(crate) fn covers(self, other: Self) -> bool {
        self.start <= other.start
            && self.end >= other.end
            && (!other.access.reads() || self.access.reads())
            && (!other.access.writes() || self.access.writes())
    }

    pub(crate) fn validate_copy_to(
        self,
        source_context: usize,
        destination: Self,
        destination_context: usize,
    ) -> ResourceResult<()> {
        if self.end - self.start != destination.end - destination.start {
            return Err(ResourceError::StreamMisuse(
                "device copy source and destination extents differ".into(),
            ));
        }
        if source_context == destination_context && self.overlaps(destination) {
            return Err(ResourceError::StreamMisuse(
                "device copy source and destination overlap".into(),
            ));
        }
        Ok(())
    }

    fn overlaps(self, other: Self) -> bool {
        self.start < self.end
            && other.start < other.end
            && self.start < other.end
            && other.start < self.end
    }

    fn conflicts(self, other: Self) -> bool {
        self.overlaps(other) && (self.access.writes() || other.access.writes())
    }
}

/// Implemented by concrete native/runtime/import storage owners. An operation
/// retains this owner, not just a pointer, allocation identity, or budget lease.
pub(crate) trait MemoryStorageOwner: Send + Sync {
    fn dependencies(&self) -> ResourceResult<Arc<DeviceAccessDependencies>>;
}

pub(crate) struct RetainedStorageUse {
    pub(crate) owner: Arc<dyn MemoryStorageOwner>,
    pub(crate) access: Access,
}

struct RegisteredStorage {
    range: MemoryUse,
    owner: std::sync::Weak<dyn MemoryStorageOwner>,
    // Set only after cold retirement has finished. A failed Weak upgrade while
    // this is false means retirement is running or uncertain, not empty history.
    released: Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Default)]
struct RegisteredStorageIndex {
    // The maximum registered span bounds the only start keys that can overlap
    // a queried range, while the ordered map excludes later allocations.
    by_start: BTreeMap<u64, Vec<RegisteredStorage>>,
    max_span: u64,
}

/// Exact reservation identity. Releasing one operation cannot release a
/// different operation with the same ranges, context, or stream.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct MemoryUseGroup(u64);

struct PreparedMemoryUses {
    uses: Box<[(usize, MemoryUse)]>,
    release_proof: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl PreparedMemoryUses {
    fn physically_released(&self) -> bool {
        self.release_proof
            .as_ref()
            .is_some_and(|proof| proof.load(Ordering::Acquire))
    }
}

/// Pending ordinary block uses and storage-range reservations. Range
/// reservations belong to an operation, not to an imported wrapper. The
/// operation's owner capsule retains the actual storage and CUDA context.
#[derive(Default)]
pub(crate) struct BlockUseRegistry {
    prepared_memory: HashMap<MemoryUseGroup, PreparedMemoryUses>,
    next_memory_group: u64,
    storage: HashMap<usize, RegisteredStorageIndex>,
}

/// One admission registry for safe XLOG operations, including independently
/// constructed runtimes sharing the same CUDA context. It does not own those
/// runtimes or their allocations; live operations retain their concrete owners.
pub(crate) fn block_use_registry() -> &'static Mutex<BlockUseRegistry> {
    static REGISTRY: OnceLock<Mutex<BlockUseRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BlockUseRegistry::default()))
}

impl BlockUseRegistry {
    pub(crate) fn register_storage(
        &mut self,
        context: usize,
        range: MemoryUse,
        owner: std::sync::Weak<dyn MemoryStorageOwner>,
        released: Arc<std::sync::atomic::AtomicBool>,
    ) {
        let index = self.storage.entry(context).or_default();
        index.max_span = index.max_span.max(range.end.saturating_sub(range.start));
        let entries = index.by_start.entry(range.start).or_default();
        entries.retain(|entry| !entry.released.load(std::sync::atomic::Ordering::Acquire));
        entries.push(RegisteredStorage {
            range,
            owner,
            released,
        });
    }

    /// Append strong owner snapshots to caller-owned storage. On error this
    /// deliberately leaves earlier snapshots in the caller's vector, so the
    /// caller can release the registry mutex before dropping the last owner.
    pub(crate) fn retain_storage_uses(
        &mut self,
        context: usize,
        uses: &[MemoryUse],
        retained: &mut Vec<RetainedStorageUse>,
    ) -> ResourceResult<()> {
        let Some(index) = self.storage.get_mut(&context) else {
            return Ok(());
        };
        for use_ in uses {
            let first_possible_start = use_.start.saturating_sub(index.max_span);
            for entries in index
                .by_start
                .range_mut(first_possible_start..use_.end)
                .map(|(_, entries)| entries)
            {
                entries.retain(|entry| !entry.released.load(std::sync::atomic::Ordering::Acquire));
                for entry in entries {
                    if !use_.overlaps(entry.range) {
                        continue;
                    }
                    let owner = entry.owner.upgrade().ok_or_else(|| {
                        ResourceError::StreamMisuse(
                            "overlapping device storage is retiring without proven release".into(),
                        )
                    })?;
                    if let Some(existing) = retained
                        .iter_mut()
                        .find(|existing| Arc::ptr_eq(&existing.owner, &owner))
                    {
                        existing.access = if existing.access.writes() || use_.access.writes() {
                            Access::ReadWrite
                        } else {
                            Access::Read
                        };
                    } else {
                        retained.push(RetainedStorageUse {
                            owner,
                            access: use_.access,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Admit a complete operation atomically. Aliases within the operation are
    /// deliberately compared only against other operations: kernels may read
    /// and write the same storage. Keeping exact spans also preserves read-only
    /// subranges and gaps instead of making a bounding interval exclusive.
    #[cfg(test)]
    pub(crate) fn reserve_memory_uses(
        &mut self,
        context: usize,
        uses: &[MemoryUse],
    ) -> ResourceResult<MemoryUseGroup> {
        self.reserve_context_memory_uses(
            &uses.iter().map(|use_| (context, *use_)).collect::<Vec<_>>(),
        )
    }

    /// A peer copy reserves source and destination contexts in the same atomic
    /// group. No partial source reservation remains if the destination conflicts.
    pub(crate) fn reserve_context_memory_uses(
        &mut self,
        uses: &[(usize, MemoryUse)],
    ) -> ResourceResult<MemoryUseGroup> {
        self.prepared_memory
            .retain(|_, pending| !pending.physically_released());
        for pending in self.prepared_memory.values() {
            if pending.uses.iter().any(|(prior_context, prior)| {
                uses.iter().any(|(next_context, next)| {
                    prior_context == next_context && prior.conflicts(*next)
                })
            }) {
                return Err(ResourceError::StreamMisuse(
                    "device storage overlaps a conflicting pending operation".into(),
                ));
            }
        }
        let next = self.next_memory_group.checked_add(1).ok_or_else(|| {
            ResourceError::Driver("device memory-use reservation identity exhausted".into())
        })?;
        let group = MemoryUseGroup(next);
        let uses = uses.to_vec().into_boxed_slice();
        self.prepared_memory.insert(
            group,
            PreparedMemoryUses {
                uses,
                release_proof: None,
            },
        );
        self.next_memory_group = next;
        Ok(group)
    }

    /// The freeing operation stays exclusive until the actual raw owner proves
    /// release, including when another reaper completes it after an error.
    pub(crate) fn reserve_reclamation(
        &mut self,
        context: usize,
        memory: MemoryUse,
        release_proof: Arc<std::sync::atomic::AtomicBool>,
    ) -> ResourceResult<MemoryUseGroup> {
        let group = self.reserve_context_memory_uses(&[(context, memory)])?;
        self.prepared_memory
            .get_mut(&group)
            .expect("new reservation present")
            .release_proof = Some(release_proof);
        Ok(group)
    }

    /// Reserve before checking the source proofs: a concurrent late reaper can
    /// publish release without this mutex. Its old free group must either still
    /// conflict, or be pruned before we reject the now-released source owner.
    pub(crate) fn reserve_owned_memory_uses(
        &mut self,
        uses: &[(usize, MemoryUse)],
        source_proofs: &[Arc<std::sync::atomic::AtomicBool>],
    ) -> ResourceResult<MemoryUseGroup> {
        let group = self.reserve_context_memory_uses(uses)?;
        if source_proofs
            .iter()
            .any(|proof| proof.load(Ordering::Acquire))
        {
            self.release_memory_uses(group)?;
            return Err(ResourceError::StreamMisuse(
                "device memory source owner has already been physically released".into(),
            ));
        }
        Ok(group)
    }

    /// Cancel before enqueue, or release after completion has been proved.
    /// Event publication alone is not completion; the enclosing transaction
    /// must retain this reservation if enqueue or synchronization is uncertain.
    pub(crate) fn release_memory_uses(&mut self, group: MemoryUseGroup) -> ResourceResult<()> {
        self.prepared_memory
            .remove(&group)
            .map(|_| ())
            .ok_or_else(|| {
                ResourceError::StreamMisuse("memory-use release has no matching reservation".into())
            })
    }
}

/// Exact byte-accounting snapshot exposed by a reservable resource decorator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceBudgetSnapshot {
    pub limit: usize,
    pub reserved: usize,
}

impl ResourceBudgetSnapshot {
    pub fn remaining(self) -> usize {
        self.limit.saturating_sub(self.reserved)
    }
}

/// Physical retirement starts only after the last shared allocation lease has
/// gone. The exact reservation survives an error or unwind and is retired by
/// the same release proof as the allocation and its accounting.
pub(crate) fn with_reclamation_admission(
    registry: &Mutex<BlockUseRegistry>,
    context: usize,
    memory: MemoryUse,
    released: Arc<std::sync::atomic::AtomicBool>,
    admission: &mut Option<MemoryUseGroup>,
    reclaim: impl FnOnce() -> ResourceResult<()>,
) -> ResourceResult<()> {
    if admission.is_none() {
        *admission = Some(
            registry
                .lock()
                .expect("device block-use registry poisoned")
                .reserve_reclamation(context, memory, released)?,
        );
    }
    // No driver work, owner destruction, or callback runs under the registry.
    reclaim()
}

/// Stream-ordered device memory resource. Implementations:
///   * [`crate::device_runtime::direct::DirectCudaResource`] —
///     cudarc default (non-pooled) backend; **candidate** for the
///     sanitizer/cert role, **unproven** until the manual Compute Sanitizer
///     acceptance gate runs on a supported host.
///   * [`crate::device_runtime::async_resource::AsyncCudaResource`] —
///     stream-ordered cuMemAllocAsync/cuMemFreeAsync backend;
///     production default when the context supports async-alloc.
///   * [`crate::device_runtime::logging::LoggingResource`] —
///     telemetry decorator over any inner resource.
///   * [`crate::device_runtime::budget::GlobalDeviceBudget`] —
///     per-runtime byte-limit decorator over any inner resource.
///
/// Implementations must be thread-safe. The runtime composes resources
/// via decoration (each resource wraps an inner `Box<dyn
/// DeviceMemoryResource + Send + Sync>`).
pub trait DeviceMemoryResource: Send + Sync {
    /// Materialize one request. Decorators must forward the same request.
    ///
    /// The terminal backend binds its ledger with `attach_resource` before
    /// malloc, retains the ticket with its storage owner, and calls `acquired`
    /// immediately after arming the actual pointer. Failed initialization must
    /// still reach physical reclamation. Only proven physical release permits
    /// `unsafe { ticket.confirm_physical_release() }`; an error or unwind alone
    /// never proves storage was released.
    ///
    /// Custom implementations must implement this lifecycle-bearing entry point;
    /// ordinary caller-facing allocation signatures remain conveniences below.
    fn materialize(&self, request: AllocationRequest) -> ResourceResult<DeviceBlock>;

    /// Exact terminal-backend accounting domain. Decorators return the identical
    /// Arc. Later allocations must pass through the domain's installed budgets.
    fn allocation_accounting(&self) -> Arc<AllocationAccounting>;

    /// Allocate `bytes` bytes on the resource's device, ordered on
    /// `stream`. The returned block is in [`BlockState::Live`].
    fn allocate(
        &self,
        bytes: usize,
        stream: StreamId,
        tag: AllocTag,
    ) -> ResourceResult<DeviceBlock> {
        self.materialize(AllocationRequest::new(bytes, stream, tag))
    }

    /// Allocate while accounting for bytes promised by the runtime but not yet
    /// materialized through this resource stack.
    ///
    /// Budget decorators must include the reservation pressure in their
    /// admission decision. Telemetry decorators must forward this call so they
    /// observe both successful allocations and admission failures. All layers
    /// forward the same lifecycle-bearing request to `materialize`.
    fn allocate_with_reservation_pressure(
        &self,
        bytes: usize,
        reservation_pressure_bytes: usize,
        stream: StreamId,
        tag: AllocTag,
    ) -> ResourceResult<DeviceBlock> {
        let mut request = AllocationRequest::new(bytes, stream, tag);
        request.reservation_pressure_bytes = reservation_pressure_bytes;
        self.materialize(request)
    }

    /// Return `block` to the resource. After this call the block's
    /// state is [`BlockState::Retired`] (or [`BlockState::Quarantined`]
    /// for debug-guard resources). Reuse of the underlying memory is
    /// resource-specific but must respect the stream-ordered contract.
    ///
    /// `block.alloc_stream` is authoritative for ordering. If the
    /// caller has touched the memory on a different stream, they must
    /// have synchronized before calling `deallocate`.
    fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()>;

    /// CUDA device ordinal this resource serves. Resources are pinned
    /// to a single device.
    fn device_ordinal(&self) -> u32;

    /// Bytes currently outstanding (live + retired-but-not-yet-freed).
    /// Used by tests and by the global budget adaptor.
    fn bytes_outstanding(&self) -> usize;

    /// Return the outer resource stack's reservable budget, if one exists.
    /// Base allocators have no declarative admission limit and return `None`;
    /// decorators must forward this query unless they enforce their own limit.
    fn budget_snapshot(&self) -> Option<ResourceBudgetSnapshot> {
        None
    }

    /// Drain any retired-but-not-yet-freed bytes whose underlying
    /// CUDA work has completed. For synchronous backends this is a
    /// no-op. For stream-ordered async backends this synchronizes
    /// the streams that have queued `cuMemFreeAsync` calls and
    /// re-counts `bytes_outstanding` accordingly.
    ///
    /// Callers that need an accurate budget reading after a burst
    /// of asynchronous deallocations should call this before
    /// reading `bytes_outstanding`. Calling on a synchronous backend
    /// is harmless and free.
    fn reap_pending(&self) -> ResourceResult<()> {
        Ok(())
    }

    /// Record that work has been (or is being) submitted on
    /// `use_stream` that touches `block`'s bytes. Resources that
    /// participate in cross-stream lifetime tracking (notably the
    /// stream-ordered async backend) MUST attach a CUDA event from
    /// `use_stream` to the block; on `deallocate(block)`, the
    /// block's `alloc_stream` will wait on every recorded event
    /// before queueing the underlying free.
    ///
    /// **The default implementation returns
    /// [`ResourceError::StreamMisuse`].** This is intentional: a
    /// silent no-op default would let a launch builder call
    /// `record_block_use` against a resource that does not
    /// actually track cross-stream uses (e.g.,
    /// [`crate::device_runtime::direct::DirectCudaResource`]),
    /// observe `Ok(())`, queue a kernel on a different stream,
    /// then drop the block — and quietly hit the cross-stream
    /// use-after-free that this API exists to prevent. False
    /// safety is worse than no safety. Resources that cannot
    /// track cross-stream uses MUST inherit this default;
    /// callers (notably the future xlog launch builder) MUST
    /// surface the error rather than masking it.
    ///
    /// Override status today:
    ///   * [`crate::device_runtime::async_resource::AsyncCudaResource`]
    ///     overrides with real event tracking.
    ///   * [`crate::device_runtime::logging::LoggingResource`] and
    ///     [`crate::device_runtime::budget::GlobalDeviceBudget`]
    ///     forward to their inner resource (so the underlying
    ///     backend's behavior surfaces unchanged).
    ///   * [`crate::device_runtime::direct::DirectCudaResource`]
    ///     does NOT override — it correctly returns
    ///     `StreamMisuse` and forces callers to either route
    ///     allocations through `AsyncCudaResource` or take
    ///     responsibility for cross-stream synchronization
    ///     themselves.
    ///
    /// # Errors
    ///   * [`ResourceError::StreamMisuse`] from the default impl
    ///     when the resource cannot track cross-stream uses.
    ///   * [`ResourceError::UseAfterFree`] if `block` is not the
    ///     block currently live at `block.ptr` (caller likely
    ///     handed back a stale [`DeviceBlock`] whose generation
    ///     no longer matches the live entry).
    ///   * [`ResourceError::StreamMisuse`] if `use_stream` does
    ///     not resolve in the resource's stream pool.
    ///   * [`ResourceError::Driver`] for CUDA driver / event
    ///     creation failures.
    ///
    /// Callers that bypass this API and submit cross-stream work
    /// directly (raw `cuMemcpyDtoHAsync`, raw `Vec<*mut c_void>`
    /// kernel launches that the launch builder did not see, etc.)
    /// are responsible for their own cross-stream synchronization.
    /// The resource cannot infer arbitrary external CUDA work.
    fn record_block_use(&self, block: &DeviceBlock, use_stream: StreamId) -> ResourceResult<()> {
        let _ = (block, use_stream);
        Err(ResourceError::StreamMisuse(
            "record_block_use unsupported by this resource (the active backend \
             does not track cross-stream uses; route allocations through a \
             stream-ordered backend such as AsyncCudaResource, or take \
             responsibility for cross-stream synchronization explicitly)"
                .to_string(),
        ))
    }

    /// Whether this resource (and any inner resources it
    /// composes) actually tracks cross-stream uses via
    /// `record_block_use`. Used by the launch recorder's
    /// preflight to fail BEFORE queueing CUDA work, rather than
    /// after. The default returns `false` to match the trait's
    /// default `record_block_use` behavior; resources that
    /// override `record_block_use` to track events MUST override
    /// this to return `true`. Decorators forward to inner.
    fn supports_block_use_tracking(&self) -> bool {
        false
    }

    /// Share the allocation's existing access history with its actual storage
    /// owner. Decorators forward the same Arc; reconstructing event history
    /// would lose writes already published through this resource.
    ///
    /// A non-tracking allocator has no history to share. A tracking allocator
    /// must implement this method instead of claiming that its history is empty.
    #[doc(hidden)]
    fn access_dependencies(
        &self,
        _block: BlockId,
        _bytes: usize,
    ) -> ResourceResult<Option<Arc<DeviceAccessDependencies>>> {
        if self.supports_block_use_tracking() {
            Err(ResourceError::StreamMisuse(
                "tracking allocator does not expose its allocation dependency owner".into(),
            ))
        } else {
            Ok(None)
        }
    }

    /// Pre-launch / pre-copy hook: queue any cross-stream waits
    /// required for `use_stream` to safely access `block` with
    /// `access` semantics. MUST be called BEFORE the GPU work is
    /// enqueued on `use_stream`.
    ///
    /// On [`Access::Read`] the resource must queue waits on every
    /// outstanding write from a different stream. On [`Access::Write`] /
    /// [`Access::ReadWrite`] the resource must additionally queue
    /// waits on every outstanding read recorded on a different
    /// stream — the writer must observe completion of every prior
    /// reader. Same-stream events are skipped (CUDA stream order
    /// already covers them).
    ///
    /// **The default implementation returns
    /// [`ResourceError::StreamMisuse`].** Same rationale as
    /// `record_block_use`: a silent no-op default would let
    /// callers paired against a non-tracking backend believe the
    /// dependency edge was queued. Decorators forward; tracking
    /// backends override.
    ///
    /// # Errors
    ///   * [`ResourceError::StreamMisuse`] from the default impl
    ///     when the resource cannot track cross-stream uses.
    ///   * [`ResourceError::UseAfterFree`] if `block` is not the
    ///     id currently live at `block.ptr`.
    ///   * [`ResourceError::Driver`] for CUDA driver / event-wait
    ///     failures.
    fn prepare_block_use(
        &self,
        block: BlockId,
        use_stream: StreamId,
        access: Access,
    ) -> ResourceResult<()> {
        let _ = (block, use_stream, access);
        Err(ResourceError::StreamMisuse(
            "prepare_block_use unsupported by this resource (the active backend \
             does not track cross-stream uses; route allocations through \
             AsyncCudaResource or take responsibility for cross-stream \
             synchronization explicitly)"
                .to_string(),
        ))
    }

    /// Post-launch / post-copy hook: record an event on
    /// `use_stream` capturing the work just enqueued and update
    /// `block`'s dependency state.
    ///
    /// Advance the recording stream's read or write frontier. Publication must
    /// preserve other streams' concurrent completions: disjoint ranges of one
    /// allocation can prepare before either operation publishes. Only events
    /// whose ordering is proved may be retired. Same-stream record and frontier
    /// replacement must be serialized so an older host publisher cannot erase
    /// a newer completion.
    ///
    /// **The default implementation returns
    /// [`ResourceError::StreamMisuse`].** Same rationale as
    /// `record_block_use`. Decorators forward; tracking backends
    /// override.
    fn finish_block_use(
        &self,
        block: BlockId,
        use_stream: StreamId,
        access: Access,
    ) -> ResourceResult<()> {
        let _ = (block, use_stream, access);
        Err(ResourceError::StreamMisuse(
            "finish_block_use unsupported by this resource (the active backend \
             does not track cross-stream uses; route allocations through \
             AsyncCudaResource or take responsibility for cross-stream \
             synchronization explicitly)"
                .to_string(),
        ))
    }
}
