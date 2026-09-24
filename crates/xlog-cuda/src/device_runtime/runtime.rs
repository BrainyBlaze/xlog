//! [`XlogDeviceRuntime`] hosts one provider-owned CUDA device, stream
//! pool, and decorated memory-resource stack.
//!
//! The canonical provider builder constructs the complete ownership graph and
//! shares its exact handles with the memory manager. Allocation state remains
//! provider-owned. The common admission registry compares pending storage
//! ranges across runtimes without owning their allocator state.

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use cudarc::driver::{CudaEvent, CudaStream};
use xlog_core::{Result, XlogError};

use super::resource::{
    AllocTag, AllocationRequest, BlockId, DeviceBlock, DeviceMemoryResource, ResourceError,
    ResourceResult, StreamId,
};
use super::stream_pool::StreamPool;
use crate::CudaDevice;

/// Execution counters for the device-controlled conditional-graph route.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConditionalGraphStats {
    /// Successfully enqueued parent-graph launches.
    pub launches: u64,
    /// Terminal event synchronizations performed by the host.
    pub terminal_synchronizations: u64,
    /// Host-side fixpoint iterations (required to remain zero).
    pub host_iterations: u64,
    /// Allocations performed after a graph launch (required to remain zero).
    pub host_allocations: u64,
    /// Device status-writer kernels included in launches.
    pub device_status_writer_launches: u64,
    /// Terminal statuses written directly by the host (required to remain zero).
    pub host_status_injections: u64,
}

/// Lifetime counters for CUDA events owned by resident graph launches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EventLifecycleStats {
    /// Events that currently own a real CUDA event handle.
    pub live_events: u64,
    /// Events successfully created and recorded.
    pub created_events: u64,
    /// Event handles destroyed after completion.
    pub destroyed_events: u64,
    /// In-flight drops that had to wait for completion.
    pub drop_waits: u64,
}

/// Lifetime counters for resident CUDA graph and executable handles.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentGraphHandleLifecycleStats {
    /// Parent graph handles currently retained by a prepared or in-flight run.
    pub live_graphs: u64,
    /// Instantiated graph executable handles currently retained.
    pub live_graph_execs: u64,
    /// Parent graph handles successfully created.
    pub created_graphs: u64,
    /// Parent graph handles destroyed.
    pub destroyed_graphs: u64,
    /// Graph executable handles successfully instantiated.
    pub created_graph_execs: u64,
    /// Graph executable handles destroyed.
    pub destroyed_graph_execs: u64,
}

#[derive(Default)]
struct ResidentRuntimeTelemetry {
    launches: AtomicU64,
    terminal_synchronizations: AtomicU64,
    host_iterations: AtomicU64,
    host_allocations: AtomicU64,
    device_status_writer_launches: AtomicU64,
    host_status_injections: AtomicU64,
    live_events: AtomicU64,
    created_events: AtomicU64,
    destroyed_events: AtomicU64,
    drop_waits: AtomicU64,
    live_graphs: AtomicU64,
    live_graph_execs: AtomicU64,
    created_graphs: AtomicU64,
    destroyed_graphs: AtomicU64,
    created_graph_execs: AtomicU64,
    destroyed_graph_execs: AtomicU64,
}

/// RAII proof that one live graph and executable pair is retained.
///
/// Construct this only after both CUDA handles have been created successfully,
/// and retain it beside the owning graph object so its counters follow the
/// actual handle lifetime.
pub(crate) struct ResidentGraphHandleLease {
    telemetry: Arc<ResidentRuntimeTelemetry>,
}

impl Drop for ResidentGraphHandleLease {
    fn drop(&mut self) {
        self.telemetry
            .live_graph_execs
            .fetch_sub(1, Ordering::AcqRel);
        self.telemetry
            .destroyed_graph_execs
            .fetch_add(1, Ordering::Relaxed);
        self.telemetry.live_graphs.fetch_sub(1, Ordering::AcqRel);
        self.telemetry
            .destroyed_graphs
            .fetch_add(1, Ordering::Relaxed);
    }
}

/// Completion event whose accounting is tied to a real cudarc event handle.
pub struct ResidentCompletionEvent {
    event: Option<CudaEvent>,
    stream: Arc<CudaStream>,
    telemetry: Arc<ResidentRuntimeTelemetry>,
    synchronized: bool,
}

impl ResidentCompletionEvent {
    /// Wait for the single terminal event. Repeated calls are no-ops.
    pub fn synchronize(&mut self) -> Result<()> {
        if self.synchronized {
            return Ok(());
        }
        let _capture_exclusion = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| {
                XlogError::Kernel(format!(
                    "resident completion wait requires an uncaptured stream: {error}"
                ))
            })?;
        self.event
            .as_ref()
            .expect("resident completion event missing before drop")
            .synchronize()
            .map_err(|error| {
                XlogError::Kernel(format!(
                    "resident conditional graph terminal event synchronization failed: {error}"
                ))
            })?;
        self.synchronized = true;
        self.telemetry
            .terminal_synchronizations
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for ResidentCompletionEvent {
    fn drop(&mut self) {
        if let Some(event) = self.event.take() {
            retire_resident_completion(
                (event, Arc::clone(&self.stream)),
                Arc::clone(&self.telemetry),
                self.synchronized,
                |(event, _)| event.synchronize().is_ok(),
                || eprintln!("CUDA completion remains unknown; retaining event and stream owners"),
            );
        }
    }
}

fn retire_resident_completion<T: Send + 'static>(
    owners: T,
    telemetry: Arc<ResidentRuntimeTelemetry>,
    synchronized: bool,
    wait: impl FnOnce(&T) -> bool + Send + 'static,
    diagnose: impl FnOnce() + Send + 'static,
) {
    let mut owners = ManuallyDrop::new(owners);
    let mut telemetry = ManuallyDrop::new(telemetry);
    crate::cuda_graph::retire_after_stream_captures(move || {
        // An unsuccessful or unwinding wait is not permission to release the
        // event, exact stream/context, or their accounting owner.
        if !synchronized {
            telemetry.drop_waits.fetch_add(1, Ordering::Relaxed);
            if !wait(&owners) {
                diagnose();
                return;
            }
        }
        // SAFETY: a successful terminal wait proves the event's completion.
        // This is the only release; unknown completion retains both owners.
        unsafe { ManuallyDrop::drop(&mut owners) };
        telemetry.live_events.fetch_sub(1, Ordering::AcqRel);
        telemetry.destroyed_events.fetch_add(1, Ordering::Relaxed);
        // SAFETY: accounting has completed and this owner was not released above.
        unsafe { ManuallyDrop::drop(&mut telemetry) };
    });
}

/// Provider-owned CUDA device runtime.
///
/// Owns the device handle, stream pool, and resource stack. Allocation and
/// deallocation calls forward through the canonical provider-built stack:
/// optional logging over one global byte budget and the asynchronous CUDA
/// allocator. Tests may inject a different stack through the crate-private
/// constructor.
pub struct XlogDeviceRuntime {
    device_ordinal: u32,
    device: Arc<CudaDevice>,
    stream_pool: Arc<StreamPool>,

    resource: Arc<dyn DeviceMemoryResource + Send + Sync>,
    /// Complete-request bytes promised but not yet materialized through the
    /// resource stack. This mutex serializes complete budget snapshots and
    /// materialization. Reclamation only lowers the independent physical ledger
    /// and never takes this lock, including reentry from failed initialization.
    reservation_bytes: Mutex<usize>,
    resident_telemetry: Arc<ResidentRuntimeTelemetry>,
}

/// One complete byte claim against a runtime resource stack's global budget.
pub(crate) struct RuntimeMemoryReservation {
    runtime: Arc<XlogDeviceRuntime>,
    total_bytes: usize,
    remaining_bytes: usize,
}

impl RuntimeMemoryReservation {
    pub(crate) fn materialize(
        &mut self,
        mut request: AllocationRequest,
    ) -> ResourceResult<DeviceBlock> {
        let bytes = request.bytes;
        if bytes > self.remaining_bytes {
            return Err(ResourceError::OutOfBudget {
                requested: bytes,
                current: self.total_bytes - self.remaining_bytes,
                remaining: self.remaining_bytes,
                limit: self.total_bytes,
            });
        }

        let mut reserved = self
            .runtime
            .reservation_bytes
            .lock()
            .expect("device-runtime reservation accounting poisoned");
        *reserved = reserved.checked_sub(bytes).ok_or_else(|| {
            ResourceError::Driver("device-runtime reservation accounting underflow".to_string())
        })?;
        self.remaining_bytes -= bytes;

        request.reservation_pressure_bytes = *reserved;
        let attempt = RuntimeReservationAttempt {
            reserved: &mut reserved,
            remaining: &mut self.remaining_bytes,
            bytes,
            reclamation: request.reclamation(),
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.runtime.resource.materialize(request)
        }));
        // Restore only an unacquired promise, then unlock before resuming a
        // panic. An initializer panic must not poison future cold reclamation.
        drop(attempt);
        drop(reserved);
        match outcome {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }
}

struct RuntimeReservationAttempt<'a> {
    reserved: &'a mut usize,
    remaining: &'a mut usize,
    bytes: usize,
    reclamation: Arc<crate::memory::AllocationReclamation>,
}

impl Drop for RuntimeReservationAttempt<'_> {
    fn drop(&mut self) {
        if !self.reclamation.was_acquired() {
            *self.reserved = self
                .reserved
                .checked_add(self.bytes)
                .expect("runtime promise rollback remains representable");
            *self.remaining = self
                .remaining
                .checked_add(self.bytes)
                .expect("runtime token rollback remains representable");
        }
    }
}

impl Drop for RuntimeMemoryReservation {
    fn drop(&mut self) {
        let mut reserved = self
            .runtime
            .reservation_bytes
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *reserved = reserved
            .checked_sub(self.remaining_bytes)
            .expect("device-runtime reservation accounting underflow");
        self.remaining_bytes = 0;
    }
}

impl XlogDeviceRuntime {
    /// Compose a runtime from already validated provider-builder parts.
    ///
    /// Kept crate-private for allocator fault injection and the canonical
    /// builder. Production callers cannot assemble mismatched resource stacks.
    pub(crate) fn with_resource(
        device: Arc<CudaDevice>,
        device_ordinal: u32,
        stream_pool: Arc<StreamPool>,
        resource: Box<dyn DeviceMemoryResource + Send + Sync>,
    ) -> Self {
        Self {
            device_ordinal,
            device,
            stream_pool,

            resource: Arc::from(resource),
            reservation_bytes: Mutex::new(0),
            resident_telemetry: Arc::new(ResidentRuntimeTelemetry::default()),
        }
    }

    /// Atomically promise `bytes` against the complete resource-stack budget.
    /// The stack must expose a finite reservable budget; otherwise complete
    /// multi-allocation admission cannot be guaranteed and is refused.
    pub(crate) fn reserve_memory(
        self: &Arc<Self>,
        bytes: usize,
    ) -> ResourceResult<RuntimeMemoryReservation> {
        let mut reserved = self
            .reservation_bytes
            .lock()
            .expect("device-runtime reservation accounting poisoned");
        let snapshot = self.resource.budget_snapshot().ok_or_else(|| {
            ResourceError::Driver(
                "device-runtime resource stack has no reservable global budget".to_string(),
            )
        })?;
        let current = snapshot.reserved.checked_add(*reserved).ok_or_else(|| {
            ResourceError::Driver("device-runtime reservation accounting overflow".to_string())
        })?;
        let remaining = snapshot.limit.saturating_sub(current);
        if bytes > remaining {
            return Err(ResourceError::OutOfBudget {
                requested: bytes,
                current,
                remaining,
                limit: snapshot.limit,
            });
        }
        *reserved = reserved.checked_add(bytes).ok_or_else(|| {
            ResourceError::Driver("device-runtime reservation accounting overflow".to_string())
        })?;
        Ok(RuntimeMemoryReservation {
            runtime: Arc::clone(self),
            total_bytes: bytes,
            remaining_bytes: bytes,
        })
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

    /// Snapshot conditional-graph execution counters.
    pub fn conditional_graph_stats(&self) -> ConditionalGraphStats {
        let telemetry = &self.resident_telemetry;
        ConditionalGraphStats {
            launches: telemetry.launches.load(Ordering::Relaxed),
            terminal_synchronizations: telemetry.terminal_synchronizations.load(Ordering::Relaxed),
            host_iterations: telemetry.host_iterations.load(Ordering::Relaxed),
            host_allocations: telemetry.host_allocations.load(Ordering::Relaxed),
            device_status_writer_launches: telemetry
                .device_status_writer_launches
                .load(Ordering::Relaxed),
            host_status_injections: telemetry.host_status_injections.load(Ordering::Relaxed),
        }
    }

    /// Reset per-execution conditional-graph counters.
    ///
    /// Handle and event lifetime counters are intentionally cumulative and are
    /// not reset because callers compare snapshots around an execution.
    pub fn reset_conditional_graph_stats(&self) {
        let telemetry = &self.resident_telemetry;
        telemetry.launches.store(0, Ordering::Relaxed);
        telemetry
            .terminal_synchronizations
            .store(0, Ordering::Relaxed);
        telemetry.host_iterations.store(0, Ordering::Relaxed);
        telemetry.host_allocations.store(0, Ordering::Relaxed);
        telemetry
            .device_status_writer_launches
            .store(0, Ordering::Relaxed);
        telemetry.host_status_injections.store(0, Ordering::Relaxed);
    }

    /// Snapshot resident completion-event lifetime counters.
    pub fn event_lifecycle_stats(&self) -> EventLifecycleStats {
        let telemetry = &self.resident_telemetry;
        EventLifecycleStats {
            live_events: telemetry.live_events.load(Ordering::Acquire),
            created_events: telemetry.created_events.load(Ordering::Relaxed),
            destroyed_events: telemetry.destroyed_events.load(Ordering::Relaxed),
            drop_waits: telemetry.drop_waits.load(Ordering::Relaxed),
        }
    }

    /// Snapshot resident graph-handle lifetime counters.
    pub fn resident_graph_handle_lifecycle_stats(&self) -> ResidentGraphHandleLifecycleStats {
        let telemetry = &self.resident_telemetry;
        ResidentGraphHandleLifecycleStats {
            live_graphs: telemetry.live_graphs.load(Ordering::Acquire),
            live_graph_execs: telemetry.live_graph_execs.load(Ordering::Acquire),
            created_graphs: telemetry.created_graphs.load(Ordering::Relaxed),
            destroyed_graphs: telemetry.destroyed_graphs.load(Ordering::Relaxed),
            created_graph_execs: telemetry.created_graph_execs.load(Ordering::Relaxed),
            destroyed_graph_execs: telemetry.destroyed_graph_execs.load(Ordering::Relaxed),
        }
    }

    /// Tie lifecycle accounting to a successfully created graph/exec pair.
    pub(crate) fn resident_graph_handle_lease(&self) -> ResidentGraphHandleLease {
        let telemetry = Arc::clone(&self.resident_telemetry);
        telemetry.live_graphs.fetch_add(1, Ordering::AcqRel);
        telemetry.created_graphs.fetch_add(1, Ordering::Relaxed);
        telemetry.live_graph_execs.fetch_add(1, Ordering::AcqRel);
        telemetry
            .created_graph_execs
            .fetch_add(1, Ordering::Relaxed);
        ResidentGraphHandleLease { telemetry }
    }

    /// Record that one prepared parent graph was successfully enqueued.
    #[doc(hidden)]
    pub fn record_conditional_graph_launch(&self, has_device_status_writer: bool) {
        self.resident_telemetry
            .launches
            .fetch_add(1, Ordering::Relaxed);
        if has_device_status_writer {
            self.resident_telemetry
                .device_status_writer_launches
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a real completion event immediately after a graph launch.
    #[doc(hidden)]
    pub fn record_resident_completion_event(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<ResidentCompletionEvent> {
        let _capture_exclusion =
            crate::cuda_graph::reserve_uncaptured_stream(stream).map_err(|error| {
                XlogError::Kernel(format!(
                    "resident completion record requires an uncaptured stream: {error}"
                ))
            })?;
        let event = stream.record_event(None).map_err(|error| {
            XlogError::Kernel(format!(
                "resident conditional graph completion event record failed: {error}"
            ))
        })?;
        let telemetry = Arc::clone(&self.resident_telemetry);
        telemetry.live_events.fetch_add(1, Ordering::AcqRel);
        telemetry.created_events.fetch_add(1, Ordering::Relaxed);
        Ok(ResidentCompletionEvent {
            event: Some(event),
            stream: Arc::clone(stream),
            telemetry,
            synchronized: false,
        })
    }

    /// Allocate via the underlying resource. Stream-ordered: the
    /// returned [`DeviceBlock`] is bound to `stream`.
    pub fn allocate(
        &self,
        bytes: usize,
        stream: StreamId,
        tag: AllocTag,
    ) -> ResourceResult<DeviceBlock> {
        self.materialize(AllocationRequest::new(bytes, stream, tag))
    }

    pub(crate) fn materialize(
        &self,
        mut request: AllocationRequest,
    ) -> ResourceResult<DeviceBlock> {
        let reservation_pressure_bytes = self
            .reservation_bytes
            .lock()
            .expect("device-runtime reservation accounting poisoned");
        request.reservation_pressure_bytes = *reservation_pressure_bytes;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.resource.materialize(request)
        }));
        drop(reservation_pressure_bytes);
        match outcome {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    /// Deallocate via the underlying resource.
    pub fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()> {
        // This is logical detachment, not proof of physical free. The backend
        // validates the exact live identity and transfers its actual Arc into
        // pending ownership. Only the last shared owner may reserve a free.
        // No promises change here. Reclamation can run from failed allocation
        // cleanup while materialization holds the promise lock on this thread.
        self.resource.deallocate(block)
    }

    /// Sum of bytes currently outstanding on this device, as reported
    /// by the underlying resource.
    pub fn bytes_outstanding(&self) -> usize {
        self.resource.bytes_outstanding()
    }

    pub(crate) fn retirement_resource(&self) -> Arc<dyn DeviceMemoryResource + Send + Sync> {
        Arc::clone(&self.resource)
    }

    /// Drain pending async frees on the underlying resource. No-op
    /// for synchronous backends. Callers that need an accurate
    /// `bytes_outstanding` reading after a burst of asynchronous
    /// deallocations should call this first.
    pub fn reap_pending(&self) -> ResourceResult<()> {
        // Last-owner retirement can invoke accounting callbacks. Drain outside
        // this runtime's budget lock and outside every allocator-map lock.
        crate::cuda_graph::reap_capture_retirements();
        // The backend owns its queue synchronization; physical settlement is
        // atomic in its ledger and needs no promise serialization.
        self.resource.reap_pending()
    }

    /// Record that work has been (or is being) submitted on
    /// `use_stream` that touches `block`. Forwards to the
    /// underlying resource stack
    /// (`LoggingResource` → `GlobalDeviceBudget` → `AsyncCudaResource`),
    /// where the stream-ordered backend attaches a CUDA event so
    /// `block.alloc_stream` waits on it before the queued
    /// `cuMemFreeAsync` runs. This is the production-reachable
    /// hook used by provider-recorded launches for `read`, `write`, and
    /// `read_write` buffer arguments. Callers that submit raw CUDA work on a
    /// stream other than `block.alloc_stream` must call this directly.
    /// See [`DeviceMemoryResource::record_block_use`] for the
    /// underlying contract.
    pub fn record_block_use(
        &self,
        block: &DeviceBlock,
        use_stream: StreamId,
    ) -> ResourceResult<()> {
        self.resource.record_block_use(block, use_stream)
    }

    /// Whether the active resource stack tracks cross-stream
    /// uses (i.e., supports `record_block_use`). The launch
    /// recorder's preflight checks this BEFORE queuing CUDA
    /// work, so a misconfigured runtime fails loudly at the
    /// boundary rather than after the launch is in flight.
    pub fn supports_block_use_tracking(&self) -> bool {
        self.resource.supports_block_use_tracking()
    }

    pub(crate) fn allocation_dependencies(
        &self,
        block: BlockId,
        bytes: usize,
    ) -> ResourceResult<Option<Arc<super::resource::DeviceAccessDependencies>>> {
        self.resource.access_dependencies(block, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_reservation_unwind_restores_only_unacquired_promises() {
        for acquired in [false, true] {
            for already_released in [false, true] {
                if already_released && !acquired {
                    continue;
                }
                let ticket = Arc::new(crate::memory::AllocationReclamation::default());
                let mut reserved = 64;
                let mut remaining = 64;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _attempt = RuntimeReservationAttempt {
                        reserved: &mut reserved,
                        remaining: &mut remaining,
                        bytes: 64,
                        reclamation: Arc::clone(&ticket),
                    };
                    if acquired {
                        ticket.acquired().unwrap();
                    }
                    if already_released {
                        ticket.complete().unwrap();
                    }
                    panic!("materialization interrupted");
                }));
                assert!(result.is_err());
                assert_eq!(reserved, if acquired { 64 } else { 128 });
                assert_eq!(remaining, if acquired { 64 } else { 128 });
            }
        }
    }
    use crate::device_runtime::resource::{
        with_reclamation_admission, BlockUseRegistry, MemoryUse,
    };
    use crate::device_runtime::{Access, BlockState, Generation};
    use std::cell::Cell;

    #[test]
    fn accepted_pending_reclamation_does_not_release_range() {
        let registry = Mutex::new(BlockUseRegistry::default());
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
        let mut admission = None;
        let reclamation = crate::memory::AllocationReclamation::default();
        with_reclamation_admission(
            &registry,
            7,
            MemoryUse::new(block.ptr, block.bytes, Access::ReadWrite).unwrap(),
            reclamation.release_proof(),
            &mut admission,
            || Ok(()),
        )
        .unwrap();
        assert!(admission.is_some());
        assert!(registry
            .lock()
            .unwrap()
            .reserve_memory_uses(
                7,
                &[MemoryUse::new(block.ptr, block.bytes, Access::ReadWrite).unwrap()]
            )
            .is_err());
    }

    #[test]
    fn physical_admission_conflict_does_not_strand_an_exact_retry() {
        let registry = Mutex::new(BlockUseRegistry::default());
        let reclamation = crate::memory::AllocationReclamation::default();
        let memory = MemoryUse::new(0x1000, 64, Access::ReadWrite).unwrap();
        let use_group = registry
            .lock()
            .unwrap()
            .reserve_memory_uses(7, &[memory])
            .unwrap();
        let mut admission = None;
        assert!(with_reclamation_admission(
            &registry,
            7,
            memory,
            reclamation.release_proof(),
            &mut admission,
            || panic!("conflicting admission must precede any physical release"),
        )
        .is_err());
        assert!(admission.is_none());
        registry
            .lock()
            .unwrap()
            .release_memory_uses(use_group)
            .unwrap();
        with_reclamation_admission(
            &registry,
            7,
            memory,
            reclamation.release_proof(),
            &mut admission,
            || {
                reclamation.complete().unwrap();
                // Another admission may already prune the completed free group.
                let mut registry = registry.try_lock().unwrap();
                let next = registry
                    .reserve_memory_uses(
                        7,
                        &[
                            super::super::resource::MemoryUse::new(0x1000, 64, Access::Write)
                                .unwrap(),
                        ],
                    )
                    .unwrap();
                registry.release_memory_uses(next).unwrap();
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn physical_reclamation_keeps_range_reserved_without_holding_registry_mutex() {
        let registry = Mutex::new(BlockUseRegistry::default());
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
        let memory = MemoryUse::new(block.ptr, block.bytes, Access::ReadWrite).unwrap();
        let reclamation = crate::memory::AllocationReclamation::default();
        let mut admission = None;
        with_reclamation_admission(
            &registry,
            7,
            MemoryUse::new(block.ptr, block.bytes, Access::ReadWrite).unwrap(),
            reclamation.release_proof(),
            &mut admission,
            || {
                let mut guard = registry
                    .try_lock()
                    .expect("driver wait must not hold registry mutex");
                assert!(guard.reserve_memory_uses(7, &[memory]).is_err());
                reclamation.complete()
            },
        )
        .unwrap();
        let mut registry = registry.lock().unwrap();
        let reused = registry.reserve_memory_uses(7, &[memory]).unwrap();
        registry.release_memory_uses(reused).unwrap();
    }

    #[test]
    fn physical_reclamation_is_not_forwarded_while_a_block_use_is_pending() {
        let mut registry = BlockUseRegistry::default();
        let context = 7;
        let block_id = BlockId {
            ptr: 0x1000,
            generation: Generation(1),
            alloc_stream: StreamId::DEFAULT,
            device_ordinal: 0,
        };
        registry
            .reserve_memory_uses(
                context,
                &[super::super::resource::MemoryUse::new(block_id.ptr, 64, Access::Read).unwrap()],
            )
            .expect("memory-use reservation");

        let forwarded = Cell::new(false);
        let block = DeviceBlock {
            ptr: block_id.ptr,
            device_ordinal: block_id.device_ordinal,
            alloc_stream: block_id.alloc_stream,
            bytes: 64,
            align: 8,
            tag: AllocTag::UNTAGGED,
            generation: block_id.generation,
            state: BlockState::Live,
        };
        assert!(with_reclamation_admission(
            &Mutex::new(registry),
            context,
            MemoryUse::new(block.ptr, block.bytes, Access::ReadWrite).unwrap(),
            crate::memory::AllocationReclamation::default().release_proof(),
            &mut None,
            || {
                forwarded.set(true);
                Ok(())
            }
        )
        .is_err());
        assert!(
            !forwarded.get(),
            "pending-use rejection must happen before resource deallocation"
        );
    }

    fn try_runtime() -> Option<XlogDeviceRuntime> {
        use super::super::async_resource::AsyncCudaResource;

        match CudaDevice::new(0) {
            Ok(device) => {
                let device = Arc::new(device);
                let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
                let resource = Box::new(AsyncCudaResource::new(
                    Arc::clone(&device),
                    0,
                    Arc::clone(&pool),
                ));
                Some(XlogDeviceRuntime::with_resource(device, 0, pool, resource))
            }
            Err(error) => {
                if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
                    panic!("XLOG_REQUIRE_CUDA=1 but CUDA is unavailable: {error}");
                }
                eprintln!("Skipping device-runtime test: CUDA unavailable: {error}");
                None
            }
        }
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
    fn with_resource_composes_owned_runtime() {
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
    }

    #[test]
    fn resident_completion_retirement_retains_unknown_event_and_stream() {
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        use std::sync::atomic::AtomicUsize;

        struct Owner(Arc<AtomicUsize>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        for completion in [false, true] {
            let drops = Arc::new(AtomicUsize::new(0));
            let event = Arc::new(Owner(Arc::clone(&drops)));
            let stream = Arc::new(Owner(Arc::clone(&drops)));
            let event_observer = Arc::downgrade(&event);
            let stream_observer = Arc::downgrade(&stream);
            let telemetry = Arc::new(super::ResidentRuntimeTelemetry::default());
            telemetry.live_events.store(1, Ordering::Relaxed);
            super::retire_resident_completion(
                (event, stream),
                Arc::clone(&telemetry),
                false,
                move |_| completion,
                || {},
            );
            assert_eq!(drops.load(Ordering::SeqCst), if completion { 2 } else { 0 });
            assert_eq!(event_observer.upgrade().is_some(), !completion);
            assert_eq!(stream_observer.upgrade().is_some(), !completion);
            assert_eq!(
                telemetry.live_events.load(Ordering::Relaxed),
                u64::from(!completion)
            );
            assert_eq!(
                telemetry.destroyed_events.load(Ordering::Relaxed),
                u64::from(completion)
            );
            assert_eq!(telemetry.drop_waits.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn resident_completion_retirement_preserves_owners_when_diagnostics_unwind() {
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        let owner = Arc::new(());
        let observer = Arc::downgrade(&owner);
        let telemetry = Arc::new(super::ResidentRuntimeTelemetry::default());
        telemetry.live_events.store(1, Ordering::Relaxed);
        let observed_telemetry = Arc::clone(&telemetry);
        let outcome = std::panic::catch_unwind(move || {
            super::retire_resident_completion(
                owner,
                telemetry,
                false,
                |_| false,
                || panic!("controlled completion diagnostic failure"),
            );
        });
        assert!(outcome.is_err());
        assert!(observer.upgrade().is_some());
        assert_eq!(observed_telemetry.live_events.load(Ordering::Relaxed), 1);
        assert_eq!(
            observed_telemetry.destroyed_events.load(Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn resident_completion_retirement_does_not_repeat_a_proven_wait() {
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        let owner = Arc::new(());
        let observer = Arc::downgrade(&owner);
        let telemetry = Arc::new(super::ResidentRuntimeTelemetry::default());
        telemetry.live_events.store(1, Ordering::Relaxed);
        super::retire_resident_completion(
            owner,
            Arc::clone(&telemetry),
            true,
            |_| panic!("already completed event must not wait again"),
            || panic!("already completed event must not report unknown completion"),
        );
        assert!(observer.upgrade().is_none());
        assert_eq!(telemetry.drop_waits.load(Ordering::Relaxed), 0);
        assert_eq!(telemetry.live_events.load(Ordering::Relaxed), 0);
        assert_eq!(telemetry.destroyed_events.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn resident_completion_event_accounts_a_real_recorded_event() {
        let Some(runtime) = try_runtime() else {
            return;
        };
        let stream = runtime
            .stream_pool()
            .resolve(StreamId::DEFAULT)
            .expect("default stream");
        let before = runtime.event_lifecycle_stats();
        let mut completion = runtime
            .record_resident_completion_event(&stream)
            .expect("record completion event");
        let live = runtime.event_lifecycle_stats();
        assert_eq!(live.live_events, before.live_events + 1);
        assert_eq!(live.created_events, before.created_events + 1);
        completion
            .synchronize()
            .expect("synchronize completion event");
        drop(completion);
        let after = runtime.event_lifecycle_stats();
        assert_eq!(after.live_events, before.live_events);
        assert_eq!(after.destroyed_events, before.destroyed_events + 1);
        assert_eq!(after.drop_waits, before.drop_waits);
    }

    #[test]
    fn resident_graph_handle_lease_balances_one_owner_slot() {
        let Some(runtime) = try_runtime() else {
            return;
        };
        let before = runtime.resident_graph_handle_lifecycle_stats();
        let lease = runtime.resident_graph_handle_lease();
        let live = runtime.resident_graph_handle_lifecycle_stats();
        assert_eq!(live.live_graphs, before.live_graphs + 1);
        assert_eq!(live.live_graph_execs, before.live_graph_execs + 1);
        drop(lease);
        let after = runtime.resident_graph_handle_lifecycle_stats();
        assert_eq!(after.live_graphs, before.live_graphs);
        assert_eq!(after.live_graph_execs, before.live_graph_execs);
        assert_eq!(
            after.created_graphs - before.created_graphs,
            after.destroyed_graphs - before.destroyed_graphs
        );
        assert_eq!(
            after.created_graph_execs - before.created_graph_execs,
            after.destroyed_graph_execs - before.destroyed_graph_execs
        );
    }
}
