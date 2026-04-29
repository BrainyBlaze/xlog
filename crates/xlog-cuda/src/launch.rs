//! Launch / use recorder for runtime-backed buffers.
//!
//! Closes the production-side of the cross-stream lifetime gap
//! identified by A4 *and* the use-after-prior-write hazard
//! discovered by the multi-threaded sort+hash-join regression.
//! Code that enqueues kernels or copies on a `launch_stream`
//! other than the buffer's `alloc_stream` MUST tell the runtime
//! about the use BEFORE the launch (so prior cross-stream waits
//! can be queued ahead of the work) AND AFTER the launch (so a
//! use-event is recorded for future readers / writers and for
//! the eventual deallocate).
//!
//! Without preflight, the CUDA mempool is free to reuse the
//! address while the cross-stream work is still in flight, AND
//! prior writes / reads on a different stream remain
//! invisible to the new work — kernels read torn state.
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
//!   1. Build the recorder, register every buffer the launch
//!      will touch via `read` / `write` / `read_write` /
//!      `read_column` / `write_column` *before* enqueueing any
//!      CUDA work. Fresh output buffers go through the same
//!      `write` / `write_column` API — there is no separate
//!      post-launch path. The recorder snapshots the block id
//!      at record time and immediately drops the slice borrow,
//!      so callers can take `&mut` afterwards.
//!   2. Call [`LaunchRecorder::preflight`] BEFORE enqueueing
//!      any CUDA work. Preflight verifies the active resource
//!      supports cross-stream tracking and (in strict mode)
//!      that every recorded buffer has a runtime block, then
//!      queues the cross-stream waits required by each
//!      recorded access kind via
//!      [`crate::device_runtime::XlogDeviceRuntime::prepare_block_use`].
//!      On failure no CUDA work has been queued yet.
//!   3. Enqueue the CUDA call on `launch_stream`.
//!   4. Call [`LaunchRecorder::commit`] AFTER the launch is
//!      enqueued. Commit calls `finish_block_use` on each
//!      tracked block — the runtime records its event on
//!      `launch_stream` at this point, and that event becomes
//!      part of the block's dependency state for future
//!      readers / writers and the eventual deallocate.
//!
//! # Why preflight queues waits, not just validates
//!
//! Earlier revisions only validated the resource stack at
//! preflight and queued waits implicitly via deallocate.
//! That protected free-after-use but NOT use-after-prior-write
//! across streams: if sort writes column A on stream X and
//! join reads column A on stream Y, the join's read kernel
//! could observe sort's pre-write contents because no event
//! fenced X→Y. This recorder closes that gap by queuing
//! `cuStreamWaitEvent` calls in preflight, before the join
//! kernel is enqueued on Y, against sort's recorded write
//! event on X.
//!
//! # External memory (DLPack, Arrow device)
//!
//! Strict mode rejects [`crate::memory::CudaColumn::Dlpack`]
//! and [`crate::memory::CudaColumn::ArrowDevice`] columns
//! outright. External memory has no xlog-side runtime identity
//! — the prepare/finish APIs cannot attach events to a buffer
//! the runtime did not allocate. Callers that need to consume
//! external columns must either:
//!   * use a permissive recorder (and accept that no
//!     cross-stream safety applies to those buffers), or
//!   * synchronize externally (e.g., wait on the producing
//!     framework's stream / event before queueing xlog work).
//!
//! Permissive mode skips external columns silently, matching
//! the legacy-buffer policy.

use std::collections::HashMap;

use crate::device_runtime::{
    Access, BlockId, DeviceBlock, Generation, ResourceError, ResourceResult, StreamId,
    XlogDeviceRuntime,
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
///
/// # Lifetime model
///
/// The recorder snapshots each registered block's identity
/// ([`BlockId`]) at record time and immediately drops the source
/// slice borrow. The recorder type itself carries no lifetime
/// parameter, so callers can interleave `rec.read(&buf)` calls
/// with later `&mut buf` kernel-param borrows freely. The
/// runtime's generation guard catches misuse where the snapshot
/// outlives the underlying allocation.
///
/// # Required call order for non-empty recorders
///
/// `preflight(&runtime)` MUST be called and return `Ok(())`
/// BEFORE any CUDA work is enqueued, AND BEFORE `commit`.
/// Preflight queues the cross-stream waits each recorded access
/// kind requires (read waits on prior writes; write waits on
/// prior writes AND prior reads), so the launch sees a
/// well-fenced view of every input. Commit then records the
/// new event on `launch_stream` so future ops can wait on it.
///
/// Empty recorders (no `read`/`write`/... calls) are a no-op
/// and bypass the preflight requirement: there are no waits
/// to queue, no events to record.
pub struct LaunchRecorder {
    launch_stream: StreamId,
    mode: RecorderMode,
    /// Recorded uses, snapshotted from source blocks at record
    /// time. The recorder holds no slice borrows after the
    /// record call returns — `&mut` kernel params are free.
    uses: Vec<RecordedUse>,
    /// First strict-mode rejection encountered while recording.
    /// Surfaced from `preflight`; the recorder's record methods
    /// return `&mut Self` so callers can chain naturally.
    strict_reject: Option<ResourceError>,
    /// `true` after a successful `preflight(&runtime)` returns
    /// `Ok(())`. `commit` rejects non-empty recorders that
    /// were not preflighted.
    preflighted: bool,
    committed: bool,
}

#[derive(Clone, Copy)]
struct RecordedUse {
    block: BlockId,
    access: Access,
    /// Site label (e.g., `"read"`, `"write"`, `"read_column"`)
    /// for diagnostics. Not used at runtime beyond error
    /// messages.
    #[allow(dead_code)]
    label: &'static str,
}

impl LaunchRecorder {
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
            preflighted: false,
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

    /// Snapshot a block reference into a recorded use. Reject
    /// post-preflight additions so the validity check at
    /// preflight time stays the source of truth.
    fn note(
        &mut self,
        label: &'static str,
        block: Option<&DeviceBlock>,
        access: Access,
        external: bool,
    ) -> &mut Self {
        if self.preflighted && self.strict_reject.is_none() {
            self.strict_reject = Some(ResourceError::StreamMisuse(format!(
                "LaunchRecorder::{}: recorded after preflight — once preflight \
                 succeeds, the set of uses is frozen so commit-time discoveries \
                 cannot leave unprotected work in flight. Record this use BEFORE \
                 preflight (the recorder is lifetime-free; snapshots release the \
                 source borrow immediately, so kernel-param &mut borrows still \
                 work)",
                label,
            )));
            return self;
        }
        if let Some(b) = block {
            self.uses.push(RecordedUse {
                block: BlockId::from_block(b),
                access,
                label,
            });
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
        slice: &crate::memory::TrackedCudaSlice<T>,
    ) -> &mut Self {
        self.note("read", slice.runtime_block(), Access::Read, false)
    }

    /// Record a runtime-backed slice the launch will write.
    /// Use this for both pre-existing buffers being overwritten
    /// AND for fresh runtime-backed allocations whose lifetime
    /// began in the same operator. The recorder snapshots block
    /// identity at record time and drops the borrow, so kernel
    /// `&mut slice` borrows after preflight are unaffected.
    pub fn write<T: cudarc::driver::DeviceRepr>(
        &mut self,
        slice: &crate::memory::TrackedCudaSlice<T>,
    ) -> &mut Self {
        self.note("write", slice.runtime_block(), Access::Write, false)
    }

    /// Record a runtime-backed slice the launch will both read
    /// and write.
    pub fn read_write<T: cudarc::driver::DeviceRepr>(
        &mut self,
        slice: &crate::memory::TrackedCudaSlice<T>,
    ) -> &mut Self {
        self.note(
            "read_write",
            slice.runtime_block(),
            Access::ReadWrite,
            false,
        )
    }

    /// Record a [`crate::memory::CudaColumn`] the launch will
    /// read. Owned columns surface their runtime block; external
    /// (`Dlpack` / `ArrowDevice`) columns are rejected in strict
    /// mode and silently skipped in permissive mode.
    pub fn read_column(&mut self, col: &crate::memory::CudaColumn) -> &mut Self {
        self.note(
            "read_column",
            col.runtime_block(),
            Access::Read,
            col.is_external(),
        )
    }

    /// Record a [`crate::memory::CudaColumn`] the launch will
    /// write.
    pub fn write_column(&mut self, col: &crate::memory::CudaColumn) -> &mut Self {
        self.note(
            "write_column",
            col.runtime_block(),
            Access::Write,
            col.is_external(),
        )
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
    pub(crate) fn read_view_runtime(&mut self, block: Option<&DeviceBlock>) -> &mut Self {
        self.note("read_view", block, Access::Read, false)
    }

    /// Number of recorded runtime-backed uses. Diagnostic.
    pub fn recorded_count(&self) -> usize {
        self.uses.len()
    }

    /// Preflight: validate the recorder is ready to commit
    /// against `runtime` AND queue every cross-stream wait the
    /// recorded access kinds require. **Stateful** — sets a flag
    /// that `commit` checks. MUST be called BEFORE enqueueing
    /// the CUDA launch / copy. On failure no CUDA work has been
    /// queued yet, the flag remains unset, and the caller can
    /// either fix the recorder or abandon the launch.
    ///
    /// Verifies (in order):
    ///   * No strict-mode rejection accumulated during recording
    ///     (untracked / external buffer in strict mode, or
    ///     post-preflight `note` attempt).
    ///   * The active resource stack supports cross-stream
    ///     tracking (`runtime.supports_block_use_tracking()`)
    ///     OR the recorder has zero tracked uses (no events to
    ///     record).
    ///
    /// Then for each recorded use, calls
    /// [`XlogDeviceRuntime::prepare_block_use`] which queues
    /// `cuStreamWaitEvent` calls on `launch_stream` for any
    /// prior write (read access) or any prior write + prior
    /// reads (write / read-write access) on a different stream.
    /// Same-stream events are skipped — already ordered.
    ///
    /// Repeated registrations of the same block in the same
    /// recorder are deduplicated to a single prepare call (the
    /// strongest access kind wins): `read` + `write` of the
    /// same block becomes one `Access::ReadWrite` prepare.
    pub fn preflight(&mut self, runtime: &XlogDeviceRuntime) -> ResourceResult<()> {
        if let Some(err) = &self.strict_reject {
            // Surface the captured strict-mode rejection
            // verbatim. Do NOT mark preflighted.
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

        let deduped = dedup_uses(&self.uses);
        for use_ in &deduped {
            runtime.prepare_block_use(use_.block, self.launch_stream, use_.access)?;
        }

        self.preflighted = true;
        Ok(())
    }

    /// Commit the recorded uses to the runtime. MUST be called
    /// AFTER preflight succeeded AND the CUDA launch has been
    /// enqueued on `launch_stream`.
    ///
    /// **Non-empty recorders that were not preflighted are
    /// rejected** with `StreamMisuse`. This closes the footgun
    /// where a caller could enqueue CUDA work, then call
    /// commit, then discover at commit-time that the active
    /// resource is unsupported — leaving unprotected work in
    /// flight. Production migrated launch paths must therefore
    /// always preflight BEFORE the CUDA call.
    ///
    /// Empty recorders (no recorded uses) bypass the check:
    /// nothing to record, no events to fire, no contract to
    /// honor.
    ///
    /// For each recorded use, calls
    /// [`XlogDeviceRuntime::finish_block_use`] which records an
    /// event on `launch_stream` and folds it into the block's
    /// dependency state (writers replace `last_write` and clear
    /// `outstanding_reads`; readers append to
    /// `outstanding_reads`). Repeated registrations of the same
    /// block are deduplicated identically to preflight.
    pub fn commit(mut self, runtime: &XlogDeviceRuntime) -> ResourceResult<()> {
        // Re-check any strict reject that may have accumulated
        // — preflight may not have been called, or may not have
        // surfaced this particular path. (Same string as
        // preflight would produce.)
        if let Some(err) = self.strict_reject.take() {
            return Err(err);
        }
        if !self.uses.is_empty() && !self.preflighted {
            return Err(ResourceError::StreamMisuse(
                "LaunchRecorder::commit: non-empty recorder reached commit without \
                 a successful preflight. The caller MUST call preflight(&runtime) \
                 BEFORE enqueueing CUDA work; otherwise commit-time failures leave \
                 unprotected work in flight. See the preflight + commit contract \
                 in the LaunchRecorder doc"
                    .to_string(),
            ));
        }

        let deduped = dedup_uses(&self.uses);
        for use_ in &deduped {
            runtime.finish_block_use(use_.block, self.launch_stream, use_.access)?;
        }
        self.committed = true;
        Ok(())
    }
}

/// Collapse multiple registrations of the same block into one
/// use with the strongest access.
///
/// The dedup key is `(ptr, generation, device_ordinal)` — NOT
/// `ptr` alone. ABA reuse inside a single recorder is rare but
/// possible (record use of buffer X, drop X, allocate a new
/// block reusing X's address, record THAT) and a ptr-only key
/// would incorrectly merge those two distinct uses. Keying by
/// the full identity tuple lets the prepare/finish path see
/// each generation's events independently; the runtime's
/// generation guard then catches any stale id at the resource
/// boundary.
///
/// Access combine: Read+Write → ReadWrite; otherwise the
/// strongest of the two operands wins.
fn dedup_uses(uses: &[RecordedUse]) -> Vec<RecordedUse> {
    let mut by_id: HashMap<(u64, Generation, u32), usize> = HashMap::with_capacity(uses.len());
    let mut deduped: Vec<RecordedUse> = Vec::with_capacity(uses.len());
    for use_ in uses {
        let key = (
            use_.block.ptr,
            use_.block.generation,
            use_.block.device_ordinal,
        );
        match by_id.get(&key) {
            Some(&idx) => {
                deduped[idx].access = combine_access(deduped[idx].access, use_.access);
            }
            None => {
                by_id.insert(key, deduped.len());
                deduped.push(*use_);
            }
        }
    }
    deduped
}

/// Strongest-access lattice: ReadWrite >= Write/Read; Write+Read = ReadWrite.
fn combine_access(a: Access, b: Access) -> Access {
    match (a, b) {
        (Access::ReadWrite, _) | (_, Access::ReadWrite) => Access::ReadWrite,
        (Access::Read, Access::Write) | (Access::Write, Access::Read) => Access::ReadWrite,
        (Access::Read, Access::Read) => Access::Read,
        (Access::Write, Access::Write) => Access::Write,
    }
}

impl Drop for LaunchRecorder {
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
    fn commit_rejects_un_preflighted_strict_recorder() {
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
        let err = rec.commit(&runtime);
        match err {
            Err(ResourceError::StreamMisuse(msg)) => {
                assert!(
                    msg.contains("without a successful preflight"),
                    "msg: {}",
                    msg
                );
            }
            other => panic!(
                "non-empty un-preflighted commit must return StreamMisuse, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn empty_recorder_commit_without_preflight_is_ok() {
        let Some((_d, rt, ls)) = try_async_runtime() else {
            return;
        };
        LaunchRecorder::new_strict(ls)
            .commit(&rt)
            .expect("empty strict commit without preflight");
    }

    #[test]
    fn note_after_preflight_via_standard_method_is_rejected() {
        let Some((device, runtime, launch_stream)) = try_async_runtime() else {
            return;
        };
        let manager = Arc::new(crate::GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
            Arc::clone(&runtime),
        ));
        let buf_a = manager.alloc::<u8>(64).expect("alloc a");
        let buf_b = manager.alloc::<u8>(64).expect("alloc b");

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(&buf_a);
        rec.preflight(&runtime).expect("preflight ok");
        rec.read(&buf_b);
        let err = rec.commit(&runtime);
        match err {
            Err(ResourceError::StreamMisuse(msg)) => {
                assert!(msg.contains("recorded after preflight"), "msg: {}", msg);
            }
            other => panic!(
                "post-preflight standard-method record must be rejected; got {:?}",
                other
            ),
        }
    }

    /// Pre-launch fresh-write path: fresh outputs are recorded
    /// BEFORE preflight via the regular `write` API. Snapshot
    /// drops the source borrow, so kernel `&mut` borrows after
    /// preflight remain valid.
    #[test]
    fn pre_preflight_fresh_write_is_accepted() {
        let Some((device, runtime, launch_stream)) = try_async_runtime() else {
            return;
        };
        let manager = Arc::new(crate::GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
            Arc::clone(&runtime),
        ));
        let buf_a = manager.alloc::<u8>(64).expect("alloc a");
        let mut buf_fresh = manager.alloc::<u8>(64).expect("alloc fresh");

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(&buf_a);
        rec.write(&buf_fresh);
        rec.preflight(&runtime).expect("preflight ok");
        // Borrows are released; kernel-style &mut works here.
        let _kernel_param = &mut buf_fresh;
        rec.commit(&runtime).expect("commit ok");
    }

    /// Read+write of the same block in a single recorder
    /// dedupes to a single ReadWrite prepare/finish call.
    #[test]
    fn read_then_write_same_block_dedupes_to_read_write() {
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
        rec.write(&buf);
        rec.preflight(&runtime).expect("preflight");
        rec.commit(&runtime).expect("commit");
    }

    /// Locks the dedup key: `(ptr, generation, device_ordinal)`,
    /// not `ptr` alone. Two `RecordedUse`s sharing a ptr but
    /// differing in generation MUST be treated as distinct
    /// entries — otherwise an ABA reuse inside a single recorder
    /// would silently collapse an event for the new allocation
    /// onto the old block's prepare/finish chain.
    #[test]
    fn dedup_keys_on_full_block_id_not_ptr_alone() {
        // Construct two RecordedUses with the same ptr but
        // distinct generations — directly drive `dedup_uses` so
        // the test is deterministic and does not require ABA to
        // actually occur on real CUDA.
        let block_a = BlockId {
            ptr: 0xdead_beef,
            generation: Generation(1),
            alloc_stream: StreamId::DEFAULT,
            device_ordinal: 0,
        };
        let block_b = BlockId {
            ptr: 0xdead_beef,
            generation: Generation(2),
            alloc_stream: StreamId::DEFAULT,
            device_ordinal: 0,
        };
        let uses = vec![
            RecordedUse {
                block: block_a,
                access: Access::Read,
                label: "read",
            },
            RecordedUse {
                block: block_b,
                access: Access::Write,
                label: "write",
            },
        ];
        let deduped = dedup_uses(&uses);
        assert_eq!(deduped.len(), 2, "ABA generations must NOT collapse");
        assert_eq!(deduped[0].block.generation, Generation(1));
        assert_eq!(deduped[0].access, Access::Read);
        assert_eq!(deduped[1].block.generation, Generation(2));
        assert_eq!(deduped[1].access, Access::Write);

        // Same ptr + same generation + duplicate access must
        // collapse into one entry with combined access.
        let same_id = vec![
            RecordedUse {
                block: block_a,
                access: Access::Read,
                label: "read",
            },
            RecordedUse {
                block: block_a,
                access: Access::Write,
                label: "write",
            },
        ];
        let collapsed = dedup_uses(&same_id);
        assert_eq!(collapsed.len(), 1);
        assert_eq!(collapsed[0].access, Access::ReadWrite);
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
