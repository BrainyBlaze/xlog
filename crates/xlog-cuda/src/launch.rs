//! Launch / use recorder for runtime-backed buffers.
//!
//! Closes the production-side of the cross-stream lifetime gap
//! identified by A4. Code that enqueues kernels or copies on a
//! `launch_stream` other than the buffer's `alloc_stream` MUST
//! tell the runtime about the use so the runtime can wait for
//! the launch to complete before its `cuMemFreeAsync` runs.
//! Without this, the CUDA mempool is free to reuse the address
//! while the cross-stream work is still in flight.
//!
//! # Modes
//!
//! Two construction modes:
//!
//!   * [`LaunchRecorder::new_permissive`] — silently skips
//!     buffers that have no runtime-side identity (legacy
//!     cudarc-backed `TrackedCudaSlice`, external `Dlpack` /
//!     `ArrowDevice` columns). Intended for low-level helpers
//!     during the migration window where mixed legacy/runtime
//!     calls are unavoidable. **Not safe for production
//!     migrated paths** — silent skips are silent gaps.
//!
//!   * [`LaunchRecorder::new_strict`] — rejects any buffer that
//!     cannot be tracked. Intended for production migrated
//!     launch paths: any buffer the recorder cannot attach an
//!     event to is a structural problem the caller must fix
//!     (route the allocation through the runtime, or refuse
//!     external memory in this code path).
//!
//! # Preflight + commit
//!
//! Production callers split the recorder into TWO phases around
//! the actual CUDA call:
//!
//!   1. Build the recorder, register buffers via `read` /
//!      `write` / `read_write` / `read_column` / etc.
//!   2. Call [`LaunchRecorder::preflight`] BEFORE enqueueing the
//!      CUDA work. Preflight verifies the active resource
//!      supports cross-stream tracking AND (in strict mode)
//!      that every recorded buffer has a runtime block. On
//!      failure no CUDA work has been queued yet.
//!   3. Enqueue the CUDA call on `launch_stream`.
//!   4. Call [`LaunchRecorder::commit`] AFTER the launch is
//!      enqueued. Commit calls `record_block_use` on each
//!      tracked block — the runtime records its event on
//!      `launch_stream` at this point, and that event will
//!      fire when the queued work completes.
//!
//! Without preflight, a recorder against `DirectCudaResource`
//! would discover the resource is unsupported only AFTER the
//! launch has been queued, leaving in-flight work without
//! cross-stream protection.
//!
//! # External memory (DLPack, Arrow device)
//!
//! Strict mode rejects [`crate::memory::CudaColumn::Dlpack`]
//! and [`crate::memory::CudaColumn::ArrowDevice`] columns
//! outright. External memory has no xlog-side runtime identity
//! — the `record_block_use` API cannot attach an event to a
//! buffer the runtime did not allocate. Callers that need to
//! consume external columns must either:
//!   * use a permissive recorder (and accept that no
//!     cross-stream safety applies to those buffers), or
//!   * synchronize externally (e.g., wait on the producing
//!     framework's stream / event before queueing xlog work).
//!
//! Permissive mode skips external columns silently, matching
//! the legacy-buffer policy.

use crate::device_runtime::{
    DeviceBlock, ResourceError, ResourceResult, StreamId, XlogDeviceRuntime,
};

/// Recorder construction mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecorderMode {
    /// Silently skip untracked buffers. Acceptable for low-level
    /// helpers during the migration window; **not** safe for
    /// production migrated paths.
    Permissive,
    /// Reject untracked buffers. Production migrated paths use
    /// this so silent skips become loud failures.
    Strict,
}

/// Records buffer uses for a single launch / copy on
/// `launch_stream`. Drop without `commit` is a programmer error;
/// the recorder logs (debug builds only) and never panics.
pub struct LaunchRecorder<'b> {
    launch_stream: StreamId,
    mode: RecorderMode,
    /// Recorded uses, by reference. Held as `&DeviceBlock`
    /// rather than copied so the caller's borrow of the source
    /// buffer enforces that the buffer is alive at
    /// recorder-construction time.
    uses: Vec<RecordedUse<'b>>,
    /// First strict-mode rejection encountered while recording.
    /// Surfaced from `preflight`; the recorder's record methods
    /// return `&mut Self` so callers can chain naturally.
    strict_reject: Option<ResourceError>,
    committed: bool,
}

struct RecordedUse<'b> {
    block: &'b DeviceBlock,
    /// Site label (e.g., `"read"`, `"write"`,
    /// `"read_column"`) for diagnostics. Not used at runtime
    /// beyond error messages.
    #[allow(dead_code)]
    label: &'static str,
}

impl<'b> LaunchRecorder<'b> {
    /// Permissive recorder: silently skips untracked buffers.
    pub fn new_permissive(launch_stream: StreamId) -> Self {
        Self::new(launch_stream, RecorderMode::Permissive)
    }

    /// Strict recorder: rejects any untracked buffer.
    /// Production migrated launch paths use this.
    pub fn new_strict(launch_stream: StreamId) -> Self {
        Self::new(launch_stream, RecorderMode::Strict)
    }

    fn new(launch_stream: StreamId, mode: RecorderMode) -> Self {
        Self {
            launch_stream,
            mode,
            uses: Vec::new(),
            strict_reject: None,
            committed: false,
        }
    }

    /// Configured launch stream.
    pub fn launch_stream(&self) -> StreamId {
        self.launch_stream
    }

    /// Configured mode.
    pub fn mode(&self) -> RecorderMode {
        self.mode
    }

    fn note(
        &mut self,
        label: &'static str,
        block: Option<&'b DeviceBlock>,
        external: bool,
    ) -> &mut Self {
        if let Some(b) = block {
            self.uses.push(RecordedUse { block: b, label });
            return self;
        }
        if self.mode == RecorderMode::Strict && self.strict_reject.is_none() {
            let why = if external {
                "external (DLPack / ArrowDevice) memory has no runtime identity; \
                 strict launch recorders cannot attach a cross-stream use to it. \
                 Use a permissive recorder OR coordinate the cross-stream \
                 synchronization explicitly outside xlog"
            } else {
                "buffer is legacy cudarc-backed (no runtime block); strict launch \
                 recorders require the allocation to be routed through \
                 GpuMemoryManager::with_runtime so a DeviceBlock is available"
            };
            self.strict_reject = Some(ResourceError::StreamMisuse(format!(
                "LaunchRecorder::{}: untracked buffer rejected — {}",
                label, why
            )));
        }
        self
    }

    /// Record a runtime-backed [`crate::memory::TrackedCudaSlice`]
    /// the launch will read.
    pub fn read<T: cudarc::driver::DeviceRepr>(
        &mut self,
        slice: &'b crate::memory::TrackedCudaSlice<T>,
    ) -> &mut Self {
        self.note("read", slice.runtime_block(), false)
    }

    /// Record a runtime-backed slice the launch will write.
    pub fn write<T: cudarc::driver::DeviceRepr>(
        &mut self,
        slice: &'b crate::memory::TrackedCudaSlice<T>,
    ) -> &mut Self {
        self.note("write", slice.runtime_block(), false)
    }

    /// Record a runtime-backed slice the launch will both read
    /// and write.
    pub fn read_write<T: cudarc::driver::DeviceRepr>(
        &mut self,
        slice: &'b mut crate::memory::TrackedCudaSlice<T>,
    ) -> &mut Self {
        self.note("read_write", slice.runtime_block(), false)
    }

    /// Record a [`crate::memory::CudaColumn`] the launch will
    /// read. Owned columns surface their runtime block; external
    /// (`Dlpack` / `ArrowDevice`) columns are rejected in strict
    /// mode and silently skipped in permissive mode.
    pub fn read_column(&mut self, col: &'b crate::memory::CudaColumn) -> &mut Self {
        self.note("read_column", col.runtime_block(), col.is_external())
    }

    /// Record a [`crate::memory::CudaColumn`] the launch will
    /// write.
    pub fn write_column(&mut self, col: &'b crate::memory::CudaColumn) -> &mut Self {
        self.note("write_column", col.runtime_block(), col.is_external())
    }

    /// Record a [`crate::provider::RawCudaView`]-style view that
    /// borrows a region of a runtime-backed allocation. The
    /// view must carry its source block via `runtime_block()`;
    /// strict mode rejects views built from legacy / external
    /// paths.
    ///
    /// Public API placeholder for the upcoming filter-class
    /// migration; no production caller exists yet.
    #[allow(dead_code)]
    pub(crate) fn read_view_runtime(&mut self, block: Option<&'b DeviceBlock>) -> &mut Self {
        self.note("read_view", block, false)
    }

    /// Number of recorded runtime-backed uses. Diagnostic.
    pub fn recorded_count(&self) -> usize {
        self.uses.len()
    }

    /// Preflight: validate the recorder is ready to commit
    /// against `runtime`. Call BEFORE enqueueing the CUDA
    /// launch / copy. On failure no CUDA work has been queued
    /// yet — production migrated paths must fail here rather
    /// than after enqueue.
    ///
    /// Verifies (in order):
    ///   * No strict-mode rejection during recording
    ///     (untracked / external buffer in strict mode).
    ///   * The active resource stack supports cross-stream
    ///     tracking (`runtime.supports_block_use_tracking()`)
    ///     OR the recorder has zero tracked uses (no events to
    ///     record).
    ///
    /// Returns `Ok(())` on success.
    pub fn preflight(&self, runtime: &XlogDeviceRuntime) -> ResourceResult<()> {
        if let Some(err) = &self.strict_reject {
            return Err(ResourceError::StreamMisuse(format!("{}", err)));
        }
        if !self.uses.is_empty() && !runtime.supports_block_use_tracking() {
            return Err(ResourceError::StreamMisuse(
                "LaunchRecorder::preflight: active resource does not support \
                 cross-stream use tracking. Build the runtime around \
                 AsyncCudaResource (or a decorator stack over it) for \
                 stream-lifetime-safe launches"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Commit the recorded uses to the runtime. MUST be called
    /// AFTER preflight succeeded AND the CUDA launch has been
    /// enqueued on `launch_stream`. Returns the first error from
    /// [`XlogDeviceRuntime::record_block_use`].
    pub fn commit(mut self, runtime: &XlogDeviceRuntime) -> ResourceResult<()> {
        // Re-check the strict reject — preflight may not have
        // been called.
        if let Some(err) = self.strict_reject.take() {
            return Err(err);
        }
        for used in &self.uses {
            runtime.record_block_use(used.block, self.launch_stream)?;
        }
        self.committed = true;
        Ok(())
    }
}

impl Drop for LaunchRecorder<'_> {
    fn drop(&mut self) {
        if !self.committed && !self.uses.is_empty() {
            #[cfg(debug_assertions)]
            eprintln!(
                "[xlog_cuda::launch] LaunchRecorder dropped without commit: \
                 {} uses on launch_stream={} (mode={:?}) were NOT recorded; \
                 cross-stream lifetime safety lost for this launch",
                self.uses.len(),
                self.launch_stream.0,
                self.mode,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_runtime::{
        AsyncCudaResource, DeviceMemoryResource, DirectCudaResource, StreamPool,
    };
    use crate::CudaDevice;
    use std::sync::Arc;
    use xlog_core::MemoryBudget;

    fn try_async_runtime() -> Option<(Arc<CudaDevice>, Arc<XlogDeviceRuntime>, StreamId)> {
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
            AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
        );
        let runtime = Arc::new(XlogDeviceRuntime::with_resource(
            Arc::clone(&device),
            0,
            Arc::clone(&pool),
            async_resource,
        ));
        let launch_stream = pool.acquire().ok()?;
        Some((device, runtime, launch_stream))
    }

    fn try_direct_runtime() -> Option<(Arc<CudaDevice>, Arc<XlogDeviceRuntime>, StreamId)> {
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let direct: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let runtime = Arc::new(XlogDeviceRuntime::with_resource(
            Arc::clone(&device),
            0,
            Arc::clone(&pool),
            direct,
        ));
        Some((device, runtime, StreamId::DEFAULT))
    }

    #[test]
    fn empty_commit_is_ok_in_both_modes() {
        let Some((_d, rt, ls)) = try_async_runtime() else {
            return;
        };
        LaunchRecorder::new_permissive(ls)
            .commit(&rt)
            .expect("permissive empty");
        LaunchRecorder::new_strict(ls)
            .commit(&rt)
            .expect("strict empty");
    }

    #[test]
    fn permissive_skips_legacy_silently() {
        let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
            return;
        };
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
            AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
        );
        let runtime = Arc::new(XlogDeviceRuntime::with_resource(
            Arc::clone(&device),
            0,
            Arc::clone(&pool),
            async_resource,
        ));
        let launch_stream = pool.acquire().expect("acquire");

        // Legacy manager — no runtime — produces None block.
        let manager = Arc::new(crate::GpuMemoryManager::new(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
        ));
        let legacy = manager.alloc::<u8>(64).expect("legacy alloc");
        assert!(legacy.runtime_block().is_none());

        let mut rec = LaunchRecorder::new_permissive(launch_stream);
        rec.read(&legacy);
        assert_eq!(rec.recorded_count(), 0);
        rec.preflight(&runtime).expect("permissive preflight");
        rec.commit(&runtime).expect("permissive commit");
    }

    #[test]
    fn strict_rejects_legacy_at_preflight() {
        let Some((device, runtime, launch_stream)) = try_async_runtime() else {
            return;
        };
        let manager = Arc::new(crate::GpuMemoryManager::new(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
        ));
        let legacy = manager.alloc::<u8>(64).expect("legacy alloc");

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(&legacy);
        let err = rec.preflight(&runtime);
        match err {
            Err(ResourceError::StreamMisuse(msg)) => {
                assert!(msg.contains("untracked buffer rejected"), "msg: {}", msg);
            }
            other => panic!(
                "strict mode must reject untracked buffer at preflight; got {:?}",
                other
            ),
        }
    }

    #[test]
    fn preflight_rejects_direct_runtime_before_enqueue() {
        let Some((device, runtime, launch_stream)) = try_direct_runtime() else {
            return;
        };
        // Need a runtime-backed slice to record (so the
        // strict_reject path doesn't preempt the
        // supports-tracking check). Use the manager built
        // around the Direct-backed runtime — alloc returns a
        // DeviceBlock; the supports-tracking failure surfaces
        // at preflight.
        let manager = Arc::new(crate::GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
            Arc::clone(&runtime),
        ));
        let buf = manager.alloc::<u8>(64).expect("alloc");
        assert!(buf.runtime_block().is_some());

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(&buf);
        let err = rec.preflight(&runtime);
        match err {
            Err(ResourceError::StreamMisuse(msg)) => {
                assert!(
                    msg.contains("does not support cross-stream use tracking"),
                    "msg: {}",
                    msg
                );
            }
            other => panic!(
                "preflight must reject Direct-backed runtime before enqueue; got {:?}",
                other
            ),
        }
    }

    #[test]
    fn preflight_then_commit_async_runtime() {
        let Some((device, runtime, launch_stream)) = try_async_runtime() else {
            return;
        };
        let manager = Arc::new(crate::GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
            Arc::clone(&runtime),
        ));
        let buf = manager.alloc::<u8>(64).expect("alloc");

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(&buf);
        rec.preflight(&runtime).expect("preflight ok");
        // (in production: enqueue CUDA launch here)
        rec.commit(&runtime).expect("commit ok");
    }

    #[test]
    fn read_column_owned_runtime_backed() {
        use crate::memory::CudaColumn;
        let Some((device, runtime, launch_stream)) = try_async_runtime() else {
            return;
        };
        let manager = Arc::new(crate::GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
            Arc::clone(&runtime),
        ));
        let slice = manager.alloc::<u8>(64).expect("alloc");
        let col = CudaColumn::owned(slice);
        assert!(col.runtime_block().is_some());

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read_column(&col);
        assert_eq!(rec.recorded_count(), 1);
        rec.preflight(&runtime).expect("preflight");
        rec.commit(&runtime).expect("commit");
    }
}
