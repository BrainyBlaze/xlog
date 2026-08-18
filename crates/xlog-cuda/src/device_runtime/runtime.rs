//! [`XlogDeviceRuntime`] — per-CUDA-ordinal singleton hosting the
//! device-runtime allocator stack.
//!
//! Replaces the per-`CudaKernelProvider` `GpuMemoryManager` model with
//! a single live runtime per physical GPU. All `CudaKernelProvider`s
//! on a given ordinal share the same runtime once the migration
//! commit lands; until then this type is constructed and used by
//! tests only.
//!
//! Singleton lifetime: leaked-Box, so the returned `&'static` borrows
//! are valid for the process. No teardown on drop — appropriate for a
//! GPU device runtime that should outlive any single executor.
//!
//! # Initialization race semantics
//!
//! Earlier revisions used `OnceLock::get_or_init(|| leaked_box)`
//! after building the runtime outside the lock. That pattern leaked
//! the loser's runtime (and its CUDA context handle) when two
//! threads raced on the first access for an ordinal.
//!
//! This module now uses an explicit per-ordinal `Mutex` plus
//! `OnceLock`: callers fast-path on `OnceLock::get()`, and on a miss
//! take the per-ordinal mutex, double-check the `OnceLock`, and only
//! the winner inside the mutex builds and stores the runtime. The
//! mutex is held only across the build, so subsequent reads are still
//! lock-free.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

use cudarc::driver::{CudaEvent, CudaStream};
use xlog_core::{Result, XlogError};

use super::direct::DirectCudaResource;
use super::resource::{
    Access, AllocTag, BlockId, DeviceBlock, DeviceMemoryResource, ResourceError, ResourceResult,
    StreamId,
};
use super::stream_pool::StreamPool;
use crate::CudaDevice;

/// Maximum CUDA ordinal supported by the singleton table. CUDA itself
/// caps at 16 visible devices in typical configurations; raise here
/// only when a multi-GPU node demands it.
pub const MAX_DEVICE_ORDINALS: usize = 16;

/// Per-ordinal singleton table. Each slot is initialized at most once
/// via `OnceLock`, gated by [`INIT_LOCKS`] so failed initialization
/// does not leak partial state.
static RUNTIMES: [OnceLock<&'static XlogDeviceRuntime>; MAX_DEVICE_ORDINALS] =
    [const { OnceLock::new() }; MAX_DEVICE_ORDINALS];

/// Per-ordinal initialization mutex. Only the holder may build and
/// store a runtime in [`RUNTIMES`]. Held across the device-open and
/// resource-construction calls so concurrent first callers do not
/// race-leak loser runtimes.
static INIT_LOCKS: [Mutex<()>; MAX_DEVICE_ORDINALS] =
    [const { Mutex::new(()) }; MAX_DEVICE_ORDINALS];

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

/// Per-CUDA-ordinal device-runtime singleton.
///
/// Owns the device handle, stream pool, and resource stack. Allocate
/// / deallocate calls forward to the resource. The resource is fixed
/// at construction (currently always [`DirectCudaResource`]); a
/// future commit will swap in [`AsyncCudaResource`] as the default
/// while keeping the direct backend reachable for sanitizer mode.
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
    /// Compose an owned runtime around a caller-supplied resource
    /// stack. **Not** a singleton — the returned value is *not*
    /// stored in [`RUNTIMES`] and does not interact with `try_get`.
    ///
    /// Intended uses:
    ///   * Tests that need to drive a specific backend (e.g.,
    ///     `AsyncCudaResource`) through the same facade production
    ///     code uses, instead of constructing the resource directly.
    ///   * Future decorator stacks (`LoggingResource`,
    ///     `GlobalDeviceBudget`, `DebugGuardResource`) that wrap the
    ///     base resource before installation.
    ///
    /// The `device` and `stream_pool` arguments must be consistent
    /// with `device_ordinal` (the pool must be bound to the same
    /// device handle, and the device must be the one the resource
    /// allocates against). The constructor does not verify this —
    /// callers that compose mismatched parts get undefined
    /// runtime-level behavior, but the per-resource device-ordinal
    /// check on `deallocate` will still surface obvious mistakes as
    /// `ResourceError::Driver`.
    ///
    /// The singleton path remains [`Self::try_get`], which today
    /// always installs the cudarc default (non-pooled) backend
    /// ([`DirectCudaResource`]). Swapping the singleton's default
    /// resource is a separate later change gated on
    /// `GlobalDeviceBudget` and `LoggingResource` landing.
    pub fn with_resource(
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

    /// Get the singleton for `ordinal`, initializing it on first
    /// access. Subsequent calls return the same `&'static`.
    ///
    /// Errors:
    ///   * `XlogError::Kernel` if `ordinal >= MAX_DEVICE_ORDINALS`.
    ///   * `XlogError::Kernel` if the CUDA device cannot be opened.
    ///
    /// Concurrency: at most one thread builds the runtime for a
    /// given ordinal. Other concurrent first callers block on the
    /// per-ordinal init mutex until the winner publishes via
    /// `OnceLock::set`, after which they observe the published
    /// runtime via the inside-mutex double-check or the lock-free
    /// fast path on subsequent calls.
    pub fn try_get(ordinal: u32) -> Result<&'static XlogDeviceRuntime> {
        let idx = ordinal as usize;
        if idx >= MAX_DEVICE_ORDINALS {
            return Err(XlogError::Kernel(format!(
                "XlogDeviceRuntime: ordinal {} exceeds MAX_DEVICE_ORDINALS={}",
                ordinal, MAX_DEVICE_ORDINALS
            )));
        }
        // Fast path: another thread already initialized this slot.
        if let Some(rt) = RUNTIMES[idx].get() {
            return Ok(*rt);
        }

        // Slow path: take the per-ordinal init mutex. Only one
        // thread per ordinal builds the runtime; the rest wait here
        // and observe the published value on the double-check below.
        let _guard = INIT_LOCKS[idx]
            .lock()
            .expect("XlogDeviceRuntime init mutex poisoned");

        // Double-check inside the lock: a previous holder may have
        // initialized while we were waiting for the mutex.
        if let Some(rt) = RUNTIMES[idx].get() {
            return Ok(*rt);
        }

        // We are the first writer for this ordinal. Build the
        // runtime; if any step fails, return the error and leave
        // RUNTIMES[idx] uninitialized so the next caller can retry.
        let device = Arc::new(CudaDevice::new(ordinal as usize).map_err(|e| {
            XlogError::Kernel(format!(
                "XlogDeviceRuntime: failed to open device {}: {}",
                ordinal, e
            ))
        })?);
        let stream_pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let resource: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(DirectCudaResource::new(Arc::clone(&device), ordinal));
        let runtime = Box::new(XlogDeviceRuntime {
            device_ordinal: ordinal,
            device,
            stream_pool,
            resource: Mutex::new(resource),
            reservation_bytes: Mutex::new(0),
            resident_telemetry: Arc::new(ResidentRuntimeTelemetry::default()),
        });
        let leaked: &'static XlogDeviceRuntime = Box::leak(runtime);

        // We hold INIT_LOCKS[idx] and confirmed RUNTIMES[idx] is
        // empty under that lock, so this `set` cannot fail. Fall
        // through to a hard panic if it does — it indicates a
        // process-internal bug we cannot recover from.
        RUNTIMES[idx]
            .set(leaked)
            .map_err(|_| ())
            .expect("XlogDeviceRuntime: OnceLock::set raced under INIT_LOCKS — bug");
        Ok(leaked)
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
        if let Some(snapshot) = resource.budget_snapshot() {
            let reserved = self
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
        }
        resource.allocate(bytes, stream, tag)
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
    /// (`GlobalDeviceBudget` → `LoggingResource` → `AsyncCudaResource`),
    /// where the stream-ordered backend attaches a CUDA event so
    /// `block.alloc_stream` waits on it before the queued
    /// `cuMemFreeAsync` runs. This is the production-reachable
    /// hook the future xlog launch builder will call for
    /// `read` / `write` / `read_write` buffer args; until that
    /// lands, callers that submit raw CUDA work on a stream
    /// other than `block.alloc_stream` should call this directly.
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

    fn try_runtime() -> Option<&'static XlogDeviceRuntime> {
        match XlogDeviceRuntime::try_get(0) {
            Ok(runtime) => Some(runtime),
            Err(error) => {
                if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
                    panic!(
                        "XLOG_REQUIRE_CUDA=1 but CUDA is unavailable \
                         (XlogDeviceRuntime::try_get): {error}"
                    );
                }
                eprintln!(
                    "Skipping device-runtime test: CUDA unavailable \
                     (XlogDeviceRuntime::try_get): {error}"
                );
                None
            }
        }
    }

    #[test]
    fn try_get_returns_same_singleton() {
        let Some(a) = try_runtime() else {
            return;
        };
        let b = XlogDeviceRuntime::try_get(0).expect("re-get");
        assert!(std::ptr::eq(a, b), "singleton must be stable for ordinal 0");
        assert_eq!(a.device_ordinal(), 0);
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
    fn try_get_rejects_out_of_range_ordinal() {
        let err = XlogDeviceRuntime::try_get(MAX_DEVICE_ORDINALS as u32);
        assert!(err.is_err());
    }

    #[test]
    fn with_resource_composes_owned_runtime_outside_singleton() {
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

        // Composed runtime is not stored in the singleton table:
        // the singleton for ordinal 0 is whatever `try_get` returns,
        // which must be a different memory address.
        let singleton = XlogDeviceRuntime::try_get(0).expect("singleton");
        assert!(
            !std::ptr::eq(&owned, singleton),
            "with_resource must not aliase the singleton slot"
        );
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

    /// `try_get` installs `DirectCudaResource` by default. The
    /// runtime's `record_block_use` must therefore return
    /// `StreamMisuse` (the trait's default) rather than silently
    /// claiming success — anything else would let a launch
    /// builder running against the singleton observe `Ok(())`
    /// while no event is actually recorded, reproducing the
    /// cross-stream use-after-free this whole layer exists to
    /// prevent. See the trait-level doc on
    /// `DeviceMemoryResource::record_block_use`.
    #[test]
    fn try_get_runtime_record_block_use_rejected_with_stream_misuse() {
        let Some(rt) = try_runtime() else {
            return;
        };
        let block = rt
            .allocate(64, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("alloc through runtime");
        let err = rt.record_block_use(&block, StreamId::DEFAULT);
        match err {
            Err(super::super::resource::ResourceError::StreamMisuse(msg)) => {
                assert!(
                    msg.contains("unsupported"),
                    "expected 'unsupported' in StreamMisuse message, got {:?}",
                    msg
                );
            }
            other => panic!(
                "XlogDeviceRuntime::try_get default (DirectCudaResource) must \
                 reject record_block_use with StreamMisuse; got {:?}",
                other
            ),
        }
        rt.deallocate(block).expect("dealloc still works");
    }
}
