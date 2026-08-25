//! [`XlogDeviceRuntime`] hosts one provider-owned CUDA device, stream
//! pool, and decorated memory-resource stack.
//!
//! The canonical provider builder constructs the complete ownership graph and
//! shares its exact handles with the memory manager. There is no process-global
//! runtime registry: dropping a provider releases its runtime normally, and
//! independently built providers never share allocator state accidentally.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use cudarc::driver::{CudaEvent, CudaStream};
use xlog_core::{Result, XlogError};

use super::resource::{
    Access, AllocTag, BlockId, DeviceBlock, DeviceMemoryResource, ResourceError, ResourceResult,
    StreamId,
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
    telemetry: Arc<ResidentRuntimeTelemetry>,
    synchronized: bool,
}

impl ResidentCompletionEvent {
    /// Wait for the single terminal event. Repeated calls are no-ops.
    pub fn synchronize(&mut self) -> Result<()> {
        if self.synchronized {
            return Ok(());
        }
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
        if !self.synchronized {
            self.telemetry.drop_waits.fetch_add(1, Ordering::Relaxed);
            if let Some(event) = &self.event {
                // Buffer and module lifetimes cannot end while the graph is in
                // flight. Drop cannot return an error, so this is best effort.
                let _ = event.synchronize();
            }
        }
        if self.event.take().is_some() {
            self.telemetry.live_events.fetch_sub(1, Ordering::AcqRel);
            self.telemetry
                .destroyed_events
                .fetch_add(1, Ordering::Relaxed);
        }
    }
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
    resource: Mutex<Box<dyn DeviceMemoryResource + Send + Sync>>,
    /// Complete-request bytes promised but not yet materialized through the
    /// resource stack. Always inspected while `resource` is locked when an
    /// allocation or new reservation competes for budget.
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
    pub(crate) fn allocate(
        &mut self,
        bytes: usize,
        stream: StreamId,
        tag: AllocTag,
    ) -> ResourceResult<DeviceBlock> {
        if bytes > self.remaining_bytes {
            return Err(ResourceError::OutOfBudget {
                requested: bytes,
                current: self.total_bytes - self.remaining_bytes,
                remaining: self.remaining_bytes,
                limit: self.total_bytes,
            });
        }

        let resource = self
            .runtime
            .resource
            .lock()
            .expect("device-runtime resource poisoned");
        let mut reserved = self
            .runtime
            .reservation_bytes
            .lock()
            .expect("device-runtime reservation accounting poisoned");
        *reserved = reserved.checked_sub(bytes).ok_or_else(|| {
            ResourceError::Driver("device-runtime reservation accounting underflow".to_string())
        })?;
        self.remaining_bytes -= bytes;

        match resource.allocate(bytes, stream, tag) {
            Ok(block) => Ok(block),
            Err(error) => {
                *reserved = reserved.checked_add(bytes).ok_or_else(|| {
                    ResourceError::Driver(
                        "device-runtime reservation rollback overflow".to_string(),
                    )
                })?;
                self.remaining_bytes =
                    self.remaining_bytes.checked_add(bytes).ok_or_else(|| {
                        ResourceError::Driver("device-runtime token rollback overflow".to_string())
                    })?;
                Err(error)
            }
        }
    }
}

impl Drop for RuntimeMemoryReservation {
    fn drop(&mut self) {
        let mut reserved = self
            .runtime
            .reservation_bytes
            .lock()
            .expect("device-runtime reservation accounting poisoned");
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
            resource: Mutex::new(resource),
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
        let resource = self
            .resource
            .lock()
            .expect("device-runtime resource poisoned");
        let snapshot = resource.budget_snapshot().ok_or_else(|| {
            ResourceError::Driver(
                "device-runtime resource stack has no reservable global budget".to_string(),
            )
        })?;
        let mut reserved = self
            .reservation_bytes
            .lock()
            .expect("device-runtime reservation accounting poisoned");
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
        stream: &CudaStream,
    ) -> Result<ResidentCompletionEvent> {
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
        let resource = self
            .resource
            .lock()
            .expect("device-runtime resource poisoned");
        let reservation_pressure_bytes = *self
            .reservation_bytes
            .lock()
            .expect("device-runtime reservation accounting poisoned");
        resource.allocate_with_reservation_pressure(bytes, reservation_pressure_bytes, stream, tag)
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

    /// Drain pending async frees on the underlying resource. No-op
    /// for synchronous backends. Callers that need an accurate
    /// `bytes_outstanding` reading after a burst of asynchronous
    /// deallocations should call this first.
    pub fn reap_pending(&self) -> ResourceResult<()> {
        self.resource
            .lock()
            .expect("device-runtime resource poisoned")
            .reap_pending()
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
        self.resource
            .lock()
            .expect("device-runtime resource poisoned")
            .record_block_use(block, use_stream)
    }

    /// Whether the active resource stack tracks cross-stream
    /// uses (i.e., supports `record_block_use`). The launch
    /// recorder's preflight checks this BEFORE queuing CUDA
    /// work, so a misconfigured runtime fails loudly at the
    /// boundary rather than after the launch is in flight.
    pub fn supports_block_use_tracking(&self) -> bool {
        self.resource
            .lock()
            .expect("device-runtime resource poisoned")
            .supports_block_use_tracking()
    }

    /// Pre-launch hook: queue cross-stream waits required for
    /// `use_stream` to safely access `block` with `access`
    /// semantics. MUST be called BEFORE the GPU work is enqueued
    /// on `use_stream`. Forwards to the resource stack; see
    /// [`DeviceMemoryResource::prepare_block_use`] for the
    /// underlying contract.
    pub fn prepare_block_use(
        &self,
        block: BlockId,
        use_stream: StreamId,
        access: Access,
    ) -> ResourceResult<()> {
        self.resource
            .lock()
            .expect("device-runtime resource poisoned")
            .prepare_block_use(block, use_stream, access)
    }

    /// Post-launch hook: record an event on `use_stream`
    /// capturing the work just enqueued and update `block`'s
    /// dependency state. MUST be called AFTER the launch /
    /// copy is queued. Forwards to the resource stack; see
    /// [`DeviceMemoryResource::finish_block_use`] for the
    /// underlying contract.
    pub fn finish_block_use(
        &self,
        block: BlockId,
        use_stream: StreamId,
        access: Access,
    ) -> ResourceResult<()> {
        self.resource
            .lock()
            .expect("device-runtime resource poisoned")
            .finish_block_use(block, use_stream, access)
    }

    /// Convenience for helper-internal scratch allocations that
    /// will be immediately written / read on `use_stream`.
    ///
    /// Looks up the [`BlockId`] from the slice's runtime block
    /// and calls [`Self::prepare_block_use`] with `access`. Use
    /// this directly after `GpuMemoryManager::alloc` when the
    /// buffer's first cross-stream consumer is the same operator
    /// (e.g., a hash-table bucket array memset on `launch_stream`
    /// against a buffer freshly allocated on the manager's
    /// default stream).
    ///
    /// Returns `Err(ResourceError::StreamMisuse)` if `slice` is
    /// not runtime-backed — strict callers should ensure their
    /// memory manager carries a runtime.
    pub fn prepare_first_use<T: cudarc::driver::DeviceRepr>(
        &self,
        slice: &crate::memory::TrackedCudaSlice<T>,
        use_stream: StreamId,
        access: Access,
    ) -> ResourceResult<()> {
        let block = slice.runtime_block().ok_or_else(|| {
            super::resource::ResourceError::StreamMisuse(
                "prepare_first_use: slice is not runtime-backed (the helper's \
                 GpuMemoryManager must be built via with_runtime)"
                    .to_string(),
            )
        })?;
        self.prepare_block_use(BlockId::from_block(block), use_stream, access)
    }

    /// Convenience for helper-internal scratch finish: looks up
    /// the [`BlockId`] from the slice and forwards to
    /// [`Self::finish_block_use`].
    pub fn finish_first_use<T: cudarc::driver::DeviceRepr>(
        &self,
        slice: &crate::memory::TrackedCudaSlice<T>,
        use_stream: StreamId,
        access: Access,
    ) -> ResourceResult<()> {
        let block = slice.runtime_block().ok_or_else(|| {
            super::resource::ResourceError::StreamMisuse(
                "finish_first_use: slice is not runtime-backed".to_string(),
            )
        })?;
        self.finish_block_use(BlockId::from_block(block), use_stream, access)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
