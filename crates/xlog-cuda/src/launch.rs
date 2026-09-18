//! Owner-bearing access recorder for one CUDA launch.
//!
//! The recorder retains real allocations/import tokens without retaining Rust
//! borrows, so callers can subsequently build mutable kernel parameters.
//! Exact CUDA context and byte spans participate in the same atomic admission
//! as safe device copies, including independently imported overlapping views.
//! Runtime-only block identities are validated against the owning allocator.
//!
//! `enqueue` admits the whole manifest, arms cleanup, queues dependency waits,
//! and invokes one synchronous enqueue closure on the resolved stream. Commit
//! publishes every shared dependency before releasing the reservation. Failure
//! or unwind synchronizes possible work; unknown completion retains the actual
//! owners and the whole reservation.
//!
//! Both modes retain owner-bearing sources. Strict mode is additionally required
//! by sealed resident execution domains. Neither mode permits arbitrary writes
//! by an external framework: unsafe import must establish the producer handoff,
//! and subsequent foreign access still requires coordination with that producer.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use cudarc::driver::CudaStream;

use crate::device_runtime::resource::MemoryUseGroup;
use crate::device_runtime::{
    Access, BlockId, BlockUse, Generation, ResourceError, ResourceResult, StreamId,
    XlogDeviceRuntime,
};
use crate::memory::{
    admit_memory_access, DeviceMemoryAccess, DeviceRead, DeviceWrite, MemoryOperationOwner,
};
use xlog_core::XlogError;

/// Recorder construction mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecorderMode {
    /// Unbound execution domain; all registered owners participate in admission.
    Permissive,
    /// Eligible for a sealed resident execution domain.
    Strict,
}

/// Failure while preparing, enqueueing, or cleaning up a recorded operation.
#[non_exhaustive]
#[derive(Debug)]
pub enum LaunchEnqueueError<E> {
    /// The recorder could not prepare the operation for enqueue.
    Preparation(ResourceError),
    /// Preparation was rejected and phase-aware cleanup also failed.
    PreparationAndCleanup {
        preparation: ResourceError,
        cleanup: ResourceError,
    },
    /// The caller's operation returned its typed error after preparation.
    Operation(E),
    /// The operation failed and phase-aware cleanup also failed.
    OperationAndCleanup {
        operation: E,
        cleanup: ResourceError,
    },
}

impl<E: fmt::Display> fmt::Display for LaunchEnqueueError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preparation(error) => write!(formatter, "launch preparation failed: {error}"),
            Self::PreparationAndCleanup {
                preparation,
                cleanup,
            } => write!(
                formatter,
                "launch preparation failed: {preparation}; cleanup also failed: {cleanup}"
            ),
            Self::Operation(error) => write!(formatter, "launch operation failed: {error}"),
            Self::OperationAndCleanup { operation, cleanup } => write!(
                formatter,
                "launch operation failed: {operation}; cleanup also failed: {cleanup}"
            ),
        }
    }
}

impl<E> std::error::Error for LaunchEnqueueError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Preparation(error) => Some(error),
            Self::PreparationAndCleanup { preparation, .. } => Some(preparation),
            Self::Operation(error) => Some(error),
            Self::OperationAndCleanup { operation, .. } => Some(operation),
        }
    }
}

impl LaunchEnqueueError<XlogError> {
    pub(crate) fn into_xlog_error(self) -> XlogError {
        match self {
            Self::Operation(error) => error,
            other => XlogError::Kernel(other.to_string()),
        }
    }
}

fn preparation_with_cleanup<E>(
    preparation: ResourceError,
    cleanup: ResourceResult<()>,
) -> LaunchEnqueueError<E> {
    match cleanup {
        Ok(()) => LaunchEnqueueError::Preparation(preparation),
        Err(cleanup) => LaunchEnqueueError::PreparationAndCleanup {
            preparation,
            cleanup,
        },
    }
}

fn operation_with_cleanup<E>(operation: E, cleanup: ResourceResult<()>) -> LaunchEnqueueError<E> {
    match cleanup {
        Ok(()) => LaunchEnqueueError::Operation(operation),
        Err(cleanup) => LaunchEnqueueError::OperationAndCleanup { operation, cleanup },
    }
}

/// Records buffer uses for a single launch or copy on `launch_stream`.
/// Consume it with [`Self::enqueue`], then consume the returned
/// [`EnqueuedLaunch`] with `commit` or `abort`. Dropping a prepared
/// transaction automatically aborts it and never panics.
///
/// # Lifetime model
///
/// Registration clones the actual backing owner, not the device bytes, and
/// immediately releases the source borrow. Callers can subsequently borrow
/// mutable kernel parameters. Internal bare block identities are validated
/// against the retained runtime's allocator during atomic admission.
///
/// # Required public call order for non-empty recorders
///
/// [`Self::enqueue`] MUST own the recorder and the closure MUST
/// enqueue every operation synchronously on its supplied stream.
/// Preparation queues the cross-stream waits each recorded access
/// kind requires (read waits on prior writes; write waits on prior
/// writes and prior reads), so the operation sees a well-fenced view
/// of every input. [`EnqueuedLaunch::commit`] then records the new
/// event on `launch_stream` so future operations can wait on it.
///
/// During managed stream capture, the same registration transfers passive
/// allocation owners and access ranges to the graph without recording events.
/// Each replay admits its complete manifest before submitting device work.
/// Nested kernels and graphs use the supplied [`CudaEnqueue`] through their
/// `launch_in` methods, without preparing another recorder inside the callback.
///
/// Empty recorders (no `read`/`write`/... calls) are a no-op
/// and bypass runtime preparation: there are no waits
/// to queue, no events to record.
pub struct LaunchRecorder {
    launch_stream: StreamId,
    mode: RecorderMode,
    /// Recorded uses, snapshotted from source blocks at record
    /// time. The recorder holds no slice borrows after the
    /// record call returns — `&mut` kernel params are free.
    uses: Vec<BlockUse>,
    accesses: Vec<DeviceMemoryAccess>,
    /// First strict-mode rejection encountered while recording.
    /// Surfaced from `preflight`; the recorder's record methods
    /// return `&mut Self` so callers can chain naturally.
    strict_reject: Option<ResourceError>,
    transaction: RecorderTransaction<MemoryOperationOwner, MemoryUseGroup>,
    // Release host capture admission even when storage enters quarantine.
    submission: Option<crate::cuda_graph::StreamSubmissionPhase>,
    bound_runtime: Option<Arc<XlogDeviceRuntime>>,
    bound_domain: Option<Arc<()>>,
}

/// Borrowed authority for one already-admitted enqueue. Nested kernel and graph
/// submissions use this proof rather than re-entering capture or memory admission.
/// It cannot escape the callback or be reconstructed from a CUDA stream handle.
pub struct CudaEnqueue<'a> {
    owner: &'a Arc<MemoryOperationOwner>,
    submission: &'a crate::cuda_graph::StreamSubmissionPin<'a>,
    execution_id: u64,
}

impl CudaEnqueue<'_> {
    pub(crate) fn from_admission<'a>(
        owner: &'a Arc<MemoryOperationOwner>,
        submission: &'a crate::cuda_graph::StreamSubmissionPin<'a>,
    ) -> ResourceResult<CudaEnqueue<'a>> {
        Ok(CudaEnqueue {
            owner,
            submission,
            execution_id: submission.validate_execution(owner.stream())?,
        })
    }
    pub fn stream(&self) -> &Arc<CudaStream> {
        self.owner.stream()
    }

    pub(crate) fn manifest(&self) -> &Arc<crate::memory::MemoryAccessManifest> {
        self.owner.manifest()
    }

    pub(crate) fn completion(&self) -> Arc<crate::memory::OperationCompletion> {
        Arc::clone(self.owner.completion())
    }

    pub(crate) fn submit<T>(
        &self,
        operation: impl FnOnce(
            Option<&crate::cuda_graph::CaptureOwners>,
        ) -> Result<T, cudarc::driver::DriverError>,
    ) -> Result<T, cudarc::driver::DriverError> {
        if self.submission.validate_execution(self.stream())? != self.execution_id {
            return Err(cudarc::driver::DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        self.submission.with_submission(operation)
    }
}

/// Enqueued operation that must be completed or cancelled exactly once.
#[must_use = "an enqueued launch must be committed or aborted"]
pub struct EnqueuedLaunch {
    recorder: LaunchRecorder,
}

impl EnqueuedLaunch {
    /// Publish the recorded uses after the operation was enqueued.
    pub fn commit(self) -> ResourceResult<()> {
        self.recorder.commit()
    }

    /// Synchronize possible work and cancel the prepared reservations.
    pub fn abort(self) -> ResourceResult<()> {
        self.recorder.abort()
    }

    pub(crate) fn commit_bound(self, domain: &Arc<()>) -> ResourceResult<()> {
        self.recorder.commit_bound(domain)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EnqueuePhase {
    NotStarted,
    MayHaveEnqueued,
}

/// Cleanup remains attached to the admitted owner after its caller has gone.
pub(crate) trait RecorderCleanup<U>: Send + Sync + 'static {
    /// Prove this closed admission complete without reinterpreting a foreign
    /// thread's implicit stream. Called only by the cold retirement queue.
    fn synchronize_retired(&self) -> ResourceResult<()>;

    /// Release one exact reservation. Error or unwind must not consume it;
    /// successful cancellation is recorded before attempting the next one.
    fn cancel_retired(&self, use_: U) -> ResourceResult<()>;
}

pub(crate) struct PreparedRecorderTransaction<R, U: Copy + 'static = BlockUse> {
    owner: Arc<R>,
    pending: Box<[U]>,
    enqueue_phase: EnqueuePhase,
}

#[must_use = "an admitted transaction must be committed or aborted"]
pub(crate) enum RecorderTransaction<R, U: Copy + 'static = BlockUse> {
    Recording,
    Prepared(PreparedRecorderTransaction<R, U>),
    Terminal,
}

struct PreparedCleanupGuard<R: RecorderCleanup<U>, U: Copy + Send + 'static = BlockUse> {
    owner: Option<Arc<R>>,
    pending: Option<Box<[U]>>,
    remaining_start: usize,
    needs_completion: bool,
    armed: bool,
}

impl<R: RecorderCleanup<U>, U: Copy + Send + 'static> PreparedCleanupGuard<R, U> {
    fn new(prepared: PreparedRecorderTransaction<R, U>) -> Self {
        Self {
            owner: Some(prepared.owner),
            pending: Some(prepared.pending),
            remaining_start: 0,
            needs_completion: prepared.enqueue_phase == EnqueuePhase::MayHaveEnqueued,
            armed: true,
        }
    }

    fn owner(&self) -> &R {
        self.owner.as_deref().expect("prepared owner is present")
    }

    fn pending(&self) -> &[U] {
        self.pending.as_deref().expect("prepared uses are present")
    }

    fn set_remaining_start(&mut self, remaining_start: usize) {
        self.remaining_start = remaining_start;
    }

    fn disarm(&mut self) {
        self.armed = false;
    }

    fn quarantine(mut self) {
        let (owner, pending) = self.take_pending();
        self.armed = false;
        quarantine_owner_and_pending(owner, pending, self.remaining_start, self.needs_completion);
    }

    fn take_pending(&mut self) -> (Arc<R>, Box<[U]>) {
        let owner = self.owner.take().expect("prepared owner is present");
        let pending = self.pending.take().expect("prepared uses are present");
        // Transfer the original capsule, without allocating while unwinding.
        (owner, pending)
    }
}

impl<R: RecorderCleanup<U>, U: Copy + Send + 'static> Drop for PreparedCleanupGuard<R, U> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let (owner, pending) = self.take_pending();
        quarantine_owner_and_pending(owner, pending, self.remaining_start, self.needs_completion);
    }
}

impl<R: RecorderCleanup<U>, U: Copy + Send + 'static> RecorderTransaction<R, U> {
    /// Take the exact owner and reservations immediately after atomic admission.
    /// The caller must install this in its cleanup owner before fallible work.
    pub(crate) fn from_admitted(owner: Arc<R>, pending: Box<[U]>) -> Self {
        Self::Prepared(PreparedRecorderTransaction {
            owner,
            pending,
            enqueue_phase: EnqueuePhase::NotStarted,
        })
    }

    fn recording() -> Self {
        Self::Recording
    }

    fn is_recording(&self) -> bool {
        matches!(self, Self::Recording)
    }

    pub(crate) fn is_prepared(&self) -> bool {
        matches!(self, Self::Prepared(_))
    }

    pub(crate) fn prepared_owner(&self) -> Option<&Arc<R>> {
        match self {
            Self::Prepared(prepared) => Some(&prepared.owner),
            _ => None,
        }
    }

    #[cfg(test)]
    fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal)
    }

    fn mark_terminal(&mut self) {
        *self = Self::Terminal;
    }

    fn mark_may_have_enqueued(&mut self) -> ResourceResult<()> {
        let Self::Prepared(prepared) = self else {
            return Err(ResourceError::StreamMisuse(
                "launch enqueue requires successful preflight".to_string(),
            ));
        };
        match prepared.enqueue_phase {
            EnqueuePhase::NotStarted => {
                prepared.enqueue_phase = EnqueuePhase::MayHaveEnqueued;
                Ok(())
            }
            EnqueuePhase::MayHaveEnqueued => Err(ResourceError::StreamMisuse(
                "launch enqueue is one-shot".to_string(),
            )),
        }
    }

    fn prepare_enqueue_with<Prepare>(&mut self, prepare: Prepare) -> ResourceResult<()>
    where
        Prepare: FnOnce(&R) -> ResourceResult<()>,
    {
        self.mark_may_have_enqueued()?;
        // Dependency waits can themselves enqueue work. The owner and every
        // reservation must already be armed when a wait fails or unwinds.
        let Self::Prepared(prepared) = self else {
            unreachable!("successful enqueue marking preserves the prepared owner")
        };
        prepare(&prepared.owner)
    }

    pub(crate) fn enqueue_operation_with<E, Admission, Admit, Prepare, Operation, Sync, Cancel>(
        &mut self,
        admit: Admit,
        prepare: Prepare,
        operation: Operation,
        sync: Sync,
        cancel: Cancel,
    ) -> Result<(), LaunchEnqueueError<E>>
    where
        Admit: FnOnce() -> ResourceResult<Admission>,
        Prepare: FnOnce(&R) -> ResourceResult<()>,
        Operation: FnOnce(&Admission) -> Result<(), E>,
        Sync: FnOnce(&R) -> ResourceResult<()>,
        Cancel: FnOnce(&R, &[U]) -> ResourceResult<()>,
    {
        // Validate and pin the exact submission phase before any dependency
        // wait can touch the stream. A stale phase has enqueued nothing and
        // must cancel without synchronizing a potentially different capture.
        let admission = match admit() {
            Ok(admission) => admission,
            Err(preparation) => {
                return Err(preparation_with_cleanup(
                    preparation,
                    self.abort_with(sync, cancel),
                ));
            }
        };
        if let Err(preparation) = self.prepare_enqueue_with(prepare) {
            drop(admission);
            return Err(preparation_with_cleanup(
                preparation,
                self.abort_with(sync, cancel),
            ));
        }
        let result = operation(&admission);
        // Capture close may now proceed. Never hold its executing pin over
        // cleanup synchronization; unwinding also drops this local guard.
        drop(admission);
        match result {
            Ok(()) => Ok(()),
            Err(operation) => Err(operation_with_cleanup(
                operation,
                self.abort_with(sync, cancel),
            )),
        }
    }

    pub(crate) fn abort_with<Sync, Cancel>(
        &mut self,
        sync: Sync,
        cancel: Cancel,
    ) -> ResourceResult<()>
    where
        Sync: FnOnce(&R) -> ResourceResult<()>,
        Cancel: FnOnce(&R, &[U]) -> ResourceResult<()>,
    {
        let transaction = std::mem::replace(self, Self::Terminal);
        let Self::Prepared(prepared) = transaction else {
            return Err(ResourceError::StreamMisuse(
                "LaunchRecorder::abort requires successful preflight".to_string(),
            ));
        };
        let enqueue_phase = prepared.enqueue_phase;
        let mut guard = PreparedCleanupGuard::new(prepared);

        if enqueue_phase == EnqueuePhase::MayHaveEnqueued {
            if let Err(sync_error) = sync(guard.owner()) {
                guard.quarantine();
                return Err(ResourceError::Driver(format!(
                    "LaunchRecorder::abort could not synchronize possible launch work; \
                     runtime and reservations were quarantined: {sync_error}"
                )));
            }
            guard.needs_completion = false;
        }
        if guard.pending().is_empty() {
            guard.disarm();
            return Ok(());
        }
        if let Err(cancel_error) = cancel(guard.owner(), guard.pending()) {
            guard.quarantine();
            return Err(ResourceError::Driver(format!(
                "LaunchRecorder::abort could not cancel prepared reservations; \
                 runtime and reservations were quarantined: {cancel_error}"
            )));
        }
        guard.disarm();
        Ok(())
    }

    pub(crate) fn commit_with<Finish, Sync, Cancel>(
        &mut self,
        mut finish: Finish,
        sync: Sync,
        cancel: Cancel,
    ) -> ResourceResult<()>
    where
        Finish: FnMut(&R, U) -> ResourceResult<()>,
        Sync: FnOnce(&R) -> ResourceResult<()>,
        Cancel: FnOnce(&R, &[U]) -> ResourceResult<()>,
    {
        let transaction = std::mem::replace(self, Self::Terminal);
        let Self::Prepared(prepared) = transaction else {
            return Err(ResourceError::StreamMisuse(
                "LaunchRecorder::commit requires successful preflight".to_string(),
            ));
        };
        let enqueue_phase = prepared.enqueue_phase;
        let mut guard = PreparedCleanupGuard::new(prepared);

        if enqueue_phase == EnqueuePhase::NotStarted && !guard.pending().is_empty() {
            let error = ResourceError::StreamMisuse(
                "LaunchRecorder::commit requires an executed enqueue callback".to_string(),
            );
            if let Err(cancel_error) = cancel(guard.owner(), guard.pending()) {
                guard.quarantine();
                return Err(cleanup_failure(
                    "commit before enqueue boundary",
                    &error,
                    &cancel_error,
                ));
            }
            guard.disarm();
            return Err(error);
        }

        let pending_len = guard.pending().len();
        for index in 0..pending_len {
            let use_ = guard.pending()[index];
            guard.set_remaining_start(index);
            if let Err(finish_error) = finish(guard.owner(), use_) {
                if let Err(sync_error) = sync(guard.owner()) {
                    guard.quarantine();
                    return Err(cleanup_failure(
                        "commit synchronization",
                        &finish_error,
                        &sync_error,
                    ));
                }
                guard.needs_completion = false;
                if let Err(cancel_error) = cancel(guard.owner(), &guard.pending()[index..]) {
                    guard.quarantine();
                    return Err(cleanup_failure(
                        "commit tail cancellation",
                        &finish_error,
                        &cancel_error,
                    ));
                }
                guard.disarm();
                return Err(finish_error);
            }
            guard.set_remaining_start(index + 1);
        }
        guard.disarm();
        Ok(())
    }
}

fn cleanup_failure(stage: &str, primary: &ResourceError, cleanup: &ResourceError) -> ResourceError {
    ResourceError::Driver(format!(
        "LaunchRecorder {stage} failed after {primary}; runtime and remaining \
         reservations were quarantined: {cleanup}"
    ))
}

/// Preserve the exact owner and unfinished reservations until cold cleanup can
/// prove completion. The authoritative queue retains this same capsule on error
/// or unwind; no launch, successful cancellation, or completed wait is repeated.
fn quarantine_owner_and_pending<R: RecorderCleanup<U>, U: Copy + Send + 'static>(
    owner: Arc<R>,
    pending: Box<[U]>,
    mut remaining_start: usize,
    mut needs_completion: bool,
) {
    let mut retained = Some(owner);
    // Enqueue may eagerly drain. Its unwind restores the callable owner to the
    // queue, but must not replace the caller's original error or outer panic.
    let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::cuda_graph::retry_retirement_after_stream_captures(move || {
            let Some(owner) = retained.as_ref() else {
                return true;
            };
            if needs_completion {
                if owner.synchronize_retired().is_err() {
                    return false;
                }
                needs_completion = false;
            }
            while let Some(use_) = pending.get(remaining_start) {
                if owner.cancel_retired(*use_).is_err() {
                    return false;
                }
                remaining_start += 1;
            }
            drop(retained.take());
            true
        });
    }));
    if let Err(payload) = attempted {
        // An arbitrary secondary panic payload can itself panic in Drop. Its
        // diagnostic value cannot justify replacing the primary failure; the
        // actual resource capsule is already restored to the retirement queue.
        std::mem::forget(payload);
    }
}

impl LaunchRecorder {
    /// Recorder for an unbound execution domain. Owner-bearing inputs are retained.
    pub fn new_permissive(launch_stream: StreamId) -> Self {
        Self::new(launch_stream, RecorderMode::Permissive)
    }

    /// Recorder eligible for a sealed resident execution domain.
    pub fn new_strict(launch_stream: StreamId) -> Self {
        Self::new(launch_stream, RecorderMode::Strict)
    }

    pub(crate) fn new_strict_bound(
        launch_stream: StreamId,
        runtime: Arc<XlogDeviceRuntime>,
        domain: Arc<()>,
    ) -> Self {
        let mut recorder = Self::new(launch_stream, RecorderMode::Strict);
        recorder.bound_runtime = Some(runtime);
        recorder.bound_domain = Some(domain);
        recorder
    }

    fn new(launch_stream: StreamId, mode: RecorderMode) -> Self {
        Self {
            launch_stream,
            mode,
            uses: Vec::new(),
            accesses: Vec::new(),
            strict_reject: None,
            transaction: RecorderTransaction::recording(),
            submission: None,
            bound_runtime: None,
            bound_domain: None,
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

    pub(crate) fn require_bound_domain<'a>(
        &'a mut self,
        runtime: &Arc<XlogDeviceRuntime>,
        domain: &Arc<()>,
        launch_stream: StreamId,
    ) -> &'a mut Self {
        let matches = self.mode == RecorderMode::Strict
            && self.launch_stream == launch_stream
            && same_arc_identity(self.bound_runtime.as_ref(), runtime)
            && same_arc_identity(self.bound_domain.as_ref(), domain);
        if !matches && self.strict_reject.is_none() {
            self.strict_reject = Some(ResourceError::StreamMisuse(
                "LaunchRecorder: recorder is not bound to the resident execution domain"
                    .to_string(),
            ));
        }
        self
    }

    fn note_identity(
        &mut self,
        label: &'static str,
        block: Option<(BlockId, usize)>,
        access: Access,
    ) -> &mut Self {
        if self.strict_reject.is_some() {
            return self;
        }
        if !self.transaction.is_recording() {
            self.strict_reject = Some(ResourceError::StreamMisuse(format!(
                "LaunchRecorder::{label}: recorded after preflight"
            )));
            return self;
        }
        if let Some((block, bytes)) = block {
            self.uses.push(BlockUse {
                block,
                bytes,
                access,
            });
        } else {
            self.strict_reject = Some(ResourceError::StreamMisuse(format!(
                "LaunchRecorder::{label}: pointer-table entry has neither an allocation identity nor a storage owner"
            )));
        }
        self
    }

    fn note_memory(
        &mut self,
        label: &'static str,
        access: ResourceResult<DeviceMemoryAccess>,
    ) -> &mut Self {
        if self.strict_reject.is_some() {
            return self;
        }
        if !self.transaction.is_recording() {
            self.strict_reject = Some(ResourceError::StreamMisuse(format!(
                "LaunchRecorder::{label}: recorded after preflight"
            )));
            return self;
        }
        match access {
            Ok(access) => self.accesses.push(access),
            Err(error) => self.strict_reject = Some(error),
        }
        self
    }

    /// Retain the actual allocation/import owner and record its exact read span.
    pub fn read<T: cudarc::driver::DeviceRepr>(
        &mut self,
        slice: &(impl DeviceRead<T> + ?Sized),
    ) -> &mut Self {
        self.note_memory("read", slice.device_view().access(Access::Read))
    }

    /// Record a read through an already-validated runtime block.
    ///
    /// Crate-internal owner capsules use this when a device pointer table
    /// retains immutable host-side block identities instead of the typed
    /// slices that originally supplied the pointees.
    #[cfg(test)]
    pub(crate) fn read_device_block(
        &mut self,
        block: &crate::device_runtime::DeviceBlock,
    ) -> &mut Self {
        self.note_identity(
            "read_device_block",
            Some((BlockId::from_block(block), block.bytes)),
            Access::Read,
        )
    }

    /// Record a read through a prevalidated immutable block-identity snapshot.
    #[cfg(test)]
    pub(crate) fn read_block_identity(&mut self, block: (BlockId, usize)) -> &mut Self {
        self.note_identity("read_block_identity", Some(block), Access::Read)
    }

    pub(crate) fn read_optional_block_identity(
        &mut self,
        block: Option<(BlockId, usize)>,
    ) -> &mut Self {
        self.note_identity("read_optional_block_identity", block, Access::Read)
    }

    /// Retain the actual owner and record the exact writable span, including
    /// fresh output allocations and independently imported aliases.
    pub fn write<T: cudarc::driver::DeviceRepr>(
        &mut self,
        slice: &(impl DeviceWrite<T> + ?Sized),
    ) -> &mut Self {
        self.note_memory("write", slice.device_view().access(Access::Write))
    }

    /// Retain an owner-bearing span used for both reading and writing.
    pub fn read_write<T: cudarc::driver::DeviceRepr>(
        &mut self,
        slice: &(impl DeviceWrite<T> + ?Sized),
    ) -> &mut Self {
        self.note_memory("read_write", slice.device_view().access(Access::ReadWrite))
    }

    /// Record a column read and retain its allocation or external import token.
    pub fn read_column(&mut self, col: &crate::memory::CudaColumn) -> &mut Self {
        self.read(col)
    }

    /// Record a [`crate::memory::CudaColumn`] the launch will
    /// write.
    pub fn write_column(&mut self, col: &crate::memory::CudaColumn) -> &mut Self {
        self.write(col)
    }

    /// Number of recorded registrations before deduplication. Diagnostic.
    pub fn recorded_count(&self) -> usize {
        self.uses.len() + self.accesses.len()
    }

    /// Atomically admit all recorded spans and retain their actual owners.
    /// This stage queues no driver work. The enqueue boundary installs required
    /// dependency waits after arming the transaction for failure and unwind.
    /// Repeated calls are rejected without discarding an existing reservation.
    pub(crate) fn preflight(&mut self, runtime: &Arc<XlogDeviceRuntime>) -> ResourceResult<()> {
        if !self.transaction.is_recording() {
            return Err(ResourceError::StreamMisuse(
                "LaunchRecorder::preflight is one-shot".into(),
            ));
        }
        self.transaction.mark_terminal();
        if let Some(bound_runtime) = &self.bound_runtime {
            if !Arc::ptr_eq(bound_runtime, runtime) {
                return Err(ResourceError::StreamMisuse(
                    "LaunchRecorder::preflight: bound recorder received a foreign runtime".into(),
                ));
            }
        }
        if let Some(error) = self.strict_reject.take() {
            return Err(error);
        }
        let stream = runtime
            .stream_pool()
            .resolve(self.launch_stream)
            .ok_or_else(|| {
                ResourceError::StreamMisuse(
                    "LaunchRecorder::preflight: launch stream is not owned by the runtime".into(),
                )
            })?;
        if self
            .accesses
            .iter()
            .any(|access| access.context() != stream.context().cu_ctx() as usize)
        {
            return Err(ResourceError::StreamMisuse(
                "LaunchRecorder::preflight: allocation and launch CUDA contexts differ".into(),
            ));
        }
        let submission = crate::cuda_graph::acquire_stream_submission_phase(&stream)?;
        self.transaction = admit_memory_access(
            stream,
            self.accesses.clone(),
            Some(Arc::clone(runtime)),
            &dedup_uses(&self.uses),
            submission.execution_id(),
        )?;
        self.submission = Some(submission);
        Ok(())
    }

    /// Prepare the recorded uses and synchronously invoke one enqueue operation.
    ///
    /// # Safety
    ///
    /// `operation` must enqueue all work synchronously before it returns and use
    /// only the supplied enqueue capability. Every XLOG-owned allocation it
    /// touches must be registered on this recorder with its exact access mode.
    /// Nested operations must reuse that capability rather than admitting
    /// another recorder or submitting independently on its raw stream.
    pub unsafe fn enqueue<E, F>(
        mut self,
        runtime: &Arc<XlogDeviceRuntime>,
        operation: F,
    ) -> Result<EnqueuedLaunch, LaunchEnqueueError<E>>
    where
        F: FnOnce(&CudaEnqueue<'_>) -> Result<(), E>,
    {
        if self.bound_runtime.is_some() || self.bound_domain.is_some() {
            let preparation = ResourceError::StreamMisuse(
                "LaunchRecorder::enqueue does not accept an execution-domain-bound recorder"
                    .to_string(),
            );
            return Err(self.reject_enqueue(preparation));
        }
        if !self.transaction.is_recording() {
            let preparation = ResourceError::StreamMisuse(
                "LaunchRecorder::enqueue requires a recorder in the recording phase".to_string(),
            );
            return Err(self.reject_enqueue(preparation));
        }

        let Some(stream) = runtime.stream_pool().resolve(self.launch_stream) else {
            let preparation = ResourceError::StreamMisuse(format!(
                "LaunchRecorder::enqueue: stream {:?} does not resolve through the supplied runtime",
                self.launch_stream
            ));
            return Err(self.reject_enqueue(preparation));
        };

        self.preflight(runtime)
            .map_err(LaunchEnqueueError::Preparation)?;

        // SAFETY: this helper carries the same caller obligations as `enqueue`.
        unsafe { self.enqueue_prepared_with(stream.as_ref(), operation) }
    }

    /// Consume a recorder whose runtime, domain marker, and stream were sealed together.
    ///
    /// # Safety
    ///
    /// `operation` must satisfy the same synchronous enqueue, stream identity,
    /// allocation registration, and enclosing-transaction obligations as
    /// [`Self::enqueue`].
    pub(crate) unsafe fn enqueue_bound<E, F>(
        mut self,
        runtime: &Arc<XlogDeviceRuntime>,
        domain: &Arc<()>,
        launch_stream: StreamId,
        stream: &CudaStream,
        operation: F,
    ) -> Result<EnqueuedLaunch, LaunchEnqueueError<E>>
    where
        F: FnOnce(&CudaEnqueue<'_>) -> Result<(), E>,
    {
        self.require_bound_domain(runtime, domain, launch_stream);
        if !self.transaction.is_recording() {
            let preparation = ResourceError::StreamMisuse(
                "LaunchRecorder::enqueue_bound requires a recorder in the recording phase"
                    .to_string(),
            );
            return Err(self.reject_enqueue(preparation));
        }
        self.preflight(runtime)
            .map_err(LaunchEnqueueError::Preparation)?;

        // SAFETY: this helper carries the same caller obligations as `enqueue_bound`.
        unsafe { self.enqueue_prepared_with(stream, operation) }
    }

    /// Enqueue using the exact admission and stream selected by preflight.
    ///
    /// # Safety
    /// `stream` must be the stream admitted by this recorder, and `operation`
    /// must satisfy the synchronous enqueue and access obligations of `enqueue`.
    pub(crate) unsafe fn enqueue_prepared_with<E, F>(
        self,
        stream: &CudaStream,
        operation: F,
    ) -> Result<EnqueuedLaunch, LaunchEnqueueError<E>>
    where
        F: FnOnce(&CudaEnqueue<'_>) -> Result<(), E>,
    {
        let mut enqueued = EnqueuedLaunch { recorder: self };
        let submission = enqueued
            .recorder
            .submission
            .as_ref()
            .expect("launch phase admitted");
        let owner = Arc::clone(
            enqueued
                .recorder
                .transaction
                .prepared_owner()
                .expect("memory admitted"),
        );
        if !std::ptr::eq(stream, owner.stream().as_ref()) {
            return Err(enqueued
                .recorder
                .reject_enqueue(ResourceError::StreamMisuse(
                    "enqueue used a different admitted stream".into(),
                )));
        }
        let result = enqueued.recorder.transaction.enqueue_operation_with(
            || {
                let pin = submission.pin()?;
                let execution_id = pin.validate_execution(owner.stream())?;
                owner.bind_submission(&pin)?;
                Ok((pin, execution_id))
            },
            MemoryOperationOwner::prepare,
            |(pin, execution_id)| {
                pin.with_serialized_submission(|| {
                    operation(&CudaEnqueue {
                        owner: &owner,
                        submission: pin,
                        execution_id: *execution_id,
                    })
                })
            },
            MemoryOperationOwner::synchronize,
            MemoryOperationOwner::cancel,
        );
        match result {
            Ok(()) => Ok(enqueued),
            Err(error) => Err(error),
        }
    }

    fn reject_enqueue<E>(&mut self, preparation: ResourceError) -> LaunchEnqueueError<E> {
        if !self.transaction.is_prepared() {
            self.transaction.mark_terminal();
            return LaunchEnqueueError::Preparation(preparation);
        }

        preparation_with_cleanup(preparation, self.abort_transaction())
    }

    /// Cancel this launch transaction using the runtime saved by preflight.
    pub(crate) fn abort(mut self) -> ResourceResult<()> {
        self.abort_transaction()
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
    /// Records completion in every retained shared dependency owner before
    /// releasing the one admission group. Only the recording stream's frontier
    /// is replaced; other streams' concurrent operations remain visible.
    pub(crate) fn commit(mut self) -> ResourceResult<()> {
        if self.bound_runtime.is_some() || self.bound_domain.is_some() {
            let error = ResourceError::StreamMisuse(
                "LaunchRecorder::commit: domain-bound recorder requires commit_bound".to_string(),
            );
            return Err(self.fail_with_abort(error));
        }
        self.commit_inner()
    }

    pub(crate) fn commit_bound(mut self, domain: &Arc<()>) -> ResourceResult<()> {
        if !same_arc_identity(self.bound_domain.as_ref(), domain) {
            let error = ResourceError::StreamMisuse(
                "LaunchRecorder::commit_bound: foreign or unbound execution domain".to_string(),
            );
            return Err(self.fail_with_abort(error));
        }
        self.commit_inner()
    }

    fn commit_inner(&mut self) -> ResourceResult<()> {
        if let Some(err) = self.strict_reject.take() {
            return Err(self.fail_with_abort(err));
        }
        if self.recorded_count() == 0 && self.transaction.is_recording() {
            self.transaction.mark_terminal();
            return Ok(());
        }
        if !self.transaction.is_prepared() {
            self.transaction.mark_terminal();
            return Err(ResourceError::StreamMisuse(
                "LaunchRecorder::commit: non-empty recorder reached commit without \
                 a successful preflight. The caller MUST call preflight(&runtime) \
                 BEFORE enqueueing CUDA work; otherwise commit-time failures leave \
                 unprotected work in flight. See the preflight + commit contract \
                 in the LaunchRecorder doc"
                    .to_string(),
            ));
        }

        self.transaction.commit_with(
            MemoryOperationOwner::finish,
            MemoryOperationOwner::synchronize,
            MemoryOperationOwner::cancel,
        )
    }

    fn abort_transaction(&mut self) -> ResourceResult<()> {
        self.transaction.abort_with(
            MemoryOperationOwner::synchronize,
            MemoryOperationOwner::cancel,
        )
    }

    fn fail_with_abort(&mut self, primary: ResourceError) -> ResourceError {
        if !self.transaction.is_prepared() {
            self.transaction.mark_terminal();
            return primary;
        }
        match self.abort_transaction() {
            Ok(()) => primary,
            Err(cleanup) => cleanup_failure("error cleanup", &primary, &cleanup),
        }
    }
}

/// Collapse multiple registrations of the same block into one
/// use with the strongest access.
///
/// The dedup key is the complete [`BlockId`] identity
/// `(ptr, generation, alloc_stream, device_ordinal)` — NOT
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
fn dedup_uses(uses: &[BlockUse]) -> Vec<BlockUse> {
    let mut by_id: HashMap<(u64, Generation, StreamId, u32, usize), usize> =
        HashMap::with_capacity(uses.len());
    let mut deduped: Vec<BlockUse> = Vec::with_capacity(uses.len());
    for use_ in uses {
        let key = (
            use_.block.ptr,
            use_.block.generation,
            use_.block.alloc_stream,
            use_.block.device_ordinal,
            use_.bytes,
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
pub(crate) fn combine_access(a: Access, b: Access) -> Access {
    match (a, b) {
        (Access::ReadWrite, _) | (_, Access::ReadWrite) => Access::ReadWrite,
        (Access::Read, Access::Write) | (Access::Write, Access::Read) => Access::ReadWrite,
        (Access::Read, Access::Read) => Access::Read,
        (Access::Write, Access::Write) => Access::Write,
    }
}

fn same_arc_identity<T>(bound: Option<&Arc<T>>, expected: &Arc<T>) -> bool {
    bound.is_some_and(|bound| Arc::ptr_eq(bound, expected))
}

fn catch_drop_cleanup<F>(cleanup: F) -> std::thread::Result<ResourceResult<()>>
where
    F: FnOnce() -> ResourceResult<()>,
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(cleanup))
}

#[cfg(debug_assertions)]
fn best_effort_drop_diagnostic(arguments: fmt::Arguments<'_>) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        use std::io::Write;

        let mut stderr = std::io::stderr().lock();
        let _ = stderr.write_fmt(arguments);
        let _ = stderr.write_all(b"\n");
    }));
}

impl Drop for LaunchRecorder {
    fn drop(&mut self) {
        if self.transaction.is_prepared() {
            match catch_drop_cleanup(|| self.abort_transaction()) {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    #[cfg(debug_assertions)]
                    best_effort_drop_diagnostic(format_args!(
                        "[xlog_cuda::launch] LaunchRecorder drop cleanup failed; \
                         runtime was quarantined: {error}"
                    ));
                }
                Err(_) => {
                    #[cfg(debug_assertions)]
                    best_effort_drop_diagnostic(format_args!(
                        "[xlog_cuda::launch] LaunchRecorder drop cleanup panicked; \
                         runtime was quarantined"
                    ));
                }
            }
        } else if self.transaction.is_recording() && self.recorded_count() != 0 {
            #[cfg(debug_assertions)]
            best_effort_drop_diagnostic(format_args!(
                "[xlog_cuda::launch] LaunchRecorder dropped without enqueue: \
                 {} uses on launch_stream={} (mode={:?}) were not prepared",
                self.recorded_count(),
                self.launch_stream.0,
                self.mode,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_runtime::{
        AsyncCudaResource, DeviceBlock, DeviceMemoryResource, DirectCudaResource, StreamPool,
    };
    use crate::CudaDevice;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use xlog_core::MemoryBudget;

    #[derive(Default)]
    struct TestTransactionOwner {
        events: Arc<Mutex<Vec<&'static str>>>,
        retirement_ready: Arc<AtomicBool>,
        retired_uses: Arc<Mutex<Vec<BlockUse>>>,
        retirement_panics: AtomicBool,
        retirement_payload_panics: AtomicBool,
        cancel_second_fails: AtomicBool,
    }

    impl RecorderCleanup<BlockUse> for TestTransactionOwner {
        fn synchronize_retired(&self) -> ResourceResult<()> {
            if self.retirement_payload_panics.swap(false, Ordering::AcqRel) {
                struct PanickingPayload;
                impl Drop for PanickingPayload {
                    fn drop(&mut self) {
                        panic!("secondary payload destructor panic");
                    }
                }
                std::panic::panic_any(PanickingPayload);
            }
            assert!(
                !self.retirement_panics.swap(false, Ordering::AcqRel),
                "retirement panic"
            );
            if !self.retirement_ready.load(Ordering::Acquire) {
                return Err(ResourceError::Driver("completion remains unknown".into()));
            }
            self.events.lock().unwrap().push("retire_sync");
            Ok(())
        }

        fn cancel_retired(&self, use_: BlockUse) -> ResourceResult<()> {
            if !self.retirement_ready.load(Ordering::Acquire) {
                return Err(ResourceError::Driver(
                    "cancellation remains unavailable".into(),
                ));
            }
            if use_.block.ptr == 0x2000 && self.cancel_second_fails.load(Ordering::Acquire) {
                return Err(ResourceError::Driver(
                    "second cancellation unavailable".into(),
                ));
            }
            self.retired_uses.lock().unwrap().push(use_);
            Ok(())
        }
    }

    #[test]
    fn failed_abort_retires_exact_owner_after_origin_thread_exits() {
        let _serial = crate::cuda_graph::capture_lifecycle_test_guard();
        let owner = Arc::new(TestTransactionOwner::default());
        let witness = Arc::downgrade(&owner);
        let ready = Arc::clone(&owner.retirement_ready);
        let retired = Arc::clone(&owner.retired_uses);
        std::thread::spawn(move || {
            let mut transaction = RecorderTransaction::from_admitted(
                owner,
                Box::new([test_use(0x1000), test_use(0x2000)]),
            );
            transaction.mark_may_have_enqueued().unwrap();
            let error = transaction
                .abort_with(
                    |_| Err(ResourceError::Driver("original wait failed".into())),
                    |_, _| panic!("unknown completion must not cancel"),
                )
                .unwrap_err();
            assert!(error.to_string().contains("original wait failed"));
        })
        .join()
        .unwrap();
        crate::cuda_graph::reap_capture_retirements();
        assert!(witness.upgrade().is_some());
        assert!(retired.lock().unwrap().is_empty());

        ready.store(true, Ordering::Release);
        crate::cuda_graph::reap_capture_retirements();
        assert!(
            witness.upgrade().is_none(),
            "completed owner was permanently abandoned"
        );
        assert_eq!(
            *retired.lock().unwrap(),
            [test_use(0x1000), test_use(0x2000)]
        );
        crate::cuda_graph::reap_capture_retirements();
        assert_eq!(retired.lock().unwrap().len(), 2);
    }

    #[test]
    fn failed_commit_retires_only_unfinished_tail_without_repeating_completed_wait() {
        let _serial = crate::cuda_graph::capture_lifecycle_test_guard();
        let owner = Arc::new(TestTransactionOwner::default());
        let mut transaction = RecorderTransaction::from_admitted(
            Arc::clone(&owner),
            Box::new([test_use(0x1000), test_use(0x2000), test_use(0x3000)]),
        );
        transaction.mark_may_have_enqueued().unwrap();
        let result = transaction.commit_with(
            |owner, use_| {
                if use_.block.ptr == 0x2000 {
                    return Err(ResourceError::Driver("finish failed".into()));
                }
                owner.retired_uses.lock().unwrap().push(use_);
                Ok(())
            },
            |_| Ok(()),
            |_, _| Err(ResourceError::Driver("cancel failed".into())),
        );
        assert!(result.is_err());
        owner.retirement_ready.store(true, Ordering::Release);
        owner.cancel_second_fails.store(true, Ordering::Release);
        crate::cuda_graph::reap_capture_retirements();
        assert_eq!(*owner.retired_uses.lock().unwrap(), [test_use(0x1000)]);
        owner.cancel_second_fails.store(false, Ordering::Release);
        crate::cuda_graph::reap_capture_retirements();
        crate::cuda_graph::reap_capture_retirements();
        assert!(
            owner.events.lock().unwrap().is_empty(),
            "successful wait was repeated"
        );
        assert_eq!(
            *owner.retired_uses.lock().unwrap(),
            [test_use(0x1000), test_use(0x2000), test_use(0x3000)]
        );
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn failed_abort_preserves_original_panic_when_eager_retirement_also_panics() {
        let _serial = crate::cuda_graph::capture_lifecycle_test_guard();
        let owner = Arc::new(TestTransactionOwner::default());
        owner.retirement_panics.store(true, Ordering::Release);
        let witness = Arc::downgrade(&owner);
        let ready = Arc::clone(&owner.retirement_ready);
        let mut transaction =
            RecorderTransaction::from_admitted(owner, Box::new([test_use(0x1000)]));
        transaction.mark_may_have_enqueued().unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = transaction.abort_with(|_| panic!("original wait panic"), |_, _| Ok(()));
        }))
        .unwrap_err();
        assert_eq!(panic.downcast_ref::<&str>(), Some(&"original wait panic"));
        assert!(witness.upgrade().is_some());
        ready.store(true, Ordering::Release);
        crate::cuda_graph::reap_capture_retirements();
        assert!(witness.upgrade().is_none());
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn failed_abort_contains_a_secondary_panicking_payload_destructor() {
        let _serial = crate::cuda_graph::capture_lifecycle_test_guard();
        let owner = Arc::new(TestTransactionOwner::default());
        owner
            .retirement_payload_panics
            .store(true, Ordering::Release);
        let witness = Arc::downgrade(&owner);
        let ready = Arc::clone(&owner.retirement_ready);
        let mut transaction =
            RecorderTransaction::from_admitted(owner, Box::new([test_use(0x1000)]));
        transaction.mark_may_have_enqueued().unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            transaction.abort_with(
                |_| Err(ResourceError::Driver("original wait error".into())),
                |_, _| panic!("unknown completion must not cancel"),
            )
        }));
        // Clean the queued owner even on the failing side of this regression.
        ready.store(true, Ordering::Release);
        crate::cuda_graph::reap_capture_retirements();
        assert!(witness.upgrade().is_none());
        assert!(
            result.is_ok(),
            "secondary panic payload escaped retirement containment"
        );
        assert!(result
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("original wait error"));
    }

    struct TestEnqueuedTransaction {
        transaction: RecorderTransaction<TestTransactionOwner>,
    }

    impl Drop for TestEnqueuedTransaction {
        fn drop(&mut self) {
            let _ = self.transaction.abort_with(
                |owner| {
                    owner.events.lock().expect("events").push("sync");
                    Ok(())
                },
                |owner, _| {
                    owner.events.lock().expect("events").push("cancel");
                    Ok(())
                },
            );
        }
    }

    fn test_use(ptr: u64) -> BlockUse {
        BlockUse {
            block: BlockId {
                ptr,
                generation: Generation(1),
                alloc_stream: StreamId(3),
                device_ordinal: 0,
            },
            bytes: 64,
            access: Access::Read,
        }
    }

    #[test]
    fn submission_admission_rejection_cancels_without_touching_the_stream() {
        for cancellation_fails in [false, true] {
            let owner = Arc::new(TestTransactionOwner::default());
            let mut transaction = RecorderTransaction::from_admitted(
                Arc::clone(&owner),
                Box::new([test_use(0x1000)]),
            );
            let result = transaction.enqueue_operation_with(
                || Err::<(), _>(ResourceError::StreamMisuse("capture target closed".into())),
                |_| panic!("rejected admission must not enqueue dependency waits"),
                |_| -> ResourceResult<()> { panic!("rejected admission must not enqueue work") },
                |_| panic!("rejected admission must not synchronize a different capture"),
                |owner, pending| {
                    assert_eq!(pending, &[test_use(0x1000)]);
                    owner.events.lock().unwrap().push("cancel");
                    if cancellation_fails {
                        Err(ResourceError::Driver("cancellation failed".into()))
                    } else {
                        Ok(())
                    }
                },
            );
            assert!(matches!(
                (&result, cancellation_fails),
                (Err(LaunchEnqueueError::Preparation(_)), false)
                    | (Err(LaunchEnqueueError::PreparationAndCleanup { .. }), true)
            ));
            assert_eq!(*owner.events.lock().unwrap(), ["cancel"]);
            assert!(transaction.is_terminal());
        }
    }

    #[test]
    fn submission_admission_guard_covers_prepare_and_operation_but_not_cleanup() {
        struct AdmissionGuard(Arc<AtomicBool>);
        impl Drop for AdmissionGuard {
            fn drop(&mut self) {
                assert!(self.0.swap(false, Ordering::SeqCst));
            }
        }

        for outcome in [
            "success",
            "prepare error",
            "operation error",
            "prepare panic",
            "operation panic",
        ] {
            let active = Arc::new(AtomicBool::new(false));
            let owner = Arc::new(TestTransactionOwner::default());
            let mut transaction = RecorderTransaction::from_admitted(
                Arc::clone(&owner),
                Box::new([test_use(0x1000)]),
            );
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                transaction.enqueue_operation_with(
                    || {
                        assert!(!active.swap(true, Ordering::SeqCst));
                        Ok(AdmissionGuard(Arc::clone(&active)))
                    },
                    |_| {
                        assert!(active.load(Ordering::SeqCst));
                        match outcome {
                            "prepare error" => Err(ResourceError::Driver("prepare failed".into())),
                            "prepare panic" => panic!("prepare unwound"),
                            _ => Ok(()),
                        }
                    },
                    |_| {
                        assert!(active.load(Ordering::SeqCst));
                        match outcome {
                            "operation error" => Err("operation failed"),
                            "operation panic" => panic!("operation unwound"),
                            _ => Ok(()),
                        }
                    },
                    |_| {
                        assert!(!active.load(Ordering::SeqCst));
                        Ok(())
                    },
                    |_, _| {
                        assert!(!active.load(Ordering::SeqCst));
                        Ok(())
                    },
                )
            }));
            assert!(!active.load(Ordering::SeqCst));
            match outcome {
                "success" => assert!(matches!(result, Ok(Ok(())))),
                "prepare error" => assert!(matches!(
                    result,
                    Ok(Err(LaunchEnqueueError::Preparation(_)))
                )),
                "operation error" => assert!(matches!(
                    result,
                    Ok(Err(LaunchEnqueueError::Operation("operation failed")))
                )),
                _ => assert!(result.is_err()),
            }
            if transaction.is_prepared() {
                transaction
                    .abort_with(
                        |_| {
                            assert!(!active.load(Ordering::SeqCst));
                            Ok(())
                        },
                        |_, _| Ok(()),
                    )
                    .unwrap();
            }
        }
    }

    #[test]
    fn bound_enqueue_success_marks_nonempty_transaction_before_operation() {
        let owner = Arc::new(TestTransactionOwner::default());
        let mut transaction =
            RecorderTransaction::from_admitted(Arc::clone(&owner), Box::new([test_use(0x1000)]));

        let result: Result<(), LaunchEnqueueError<&'static str>> = transaction
            .enqueue_operation_with(
                || Ok(()),
                |owner| {
                    owner.events.lock().expect("events").push("prepare");
                    Ok(())
                },
                |_| {
                    owner.events.lock().expect("events").push("operation");
                    Ok(())
                },
                |_| Ok(()),
                |_, _| Ok(()),
            );
        result.expect("enqueue operation");
        transaction
            .commit_with(
                |owner, _| {
                    owner.events.lock().expect("events").push("finish");
                    Ok(())
                },
                |_| Ok(()),
                |_, _| Ok(()),
            )
            .expect("commit");

        assert_eq!(
            *owner.events.lock().expect("events"),
            ["prepare", "operation", "finish"]
        );
    }

    #[test]
    fn bound_enqueue_operation_error_synchronizes_and_cancels() {
        let owner = Arc::new(TestTransactionOwner::default());
        let mut transaction =
            RecorderTransaction::from_admitted(Arc::clone(&owner), Box::new([test_use(0x1000)]));

        let error = transaction
            .enqueue_operation_with(
                || Ok(()),
                |_| Ok(()),
                |_| {
                    owner.events.lock().expect("events").push("operation");
                    Err("injected operation failure")
                },
                |owner| {
                    owner.events.lock().expect("events").push("sync");
                    Ok(())
                },
                |owner, _| {
                    owner.events.lock().expect("events").push("cancel");
                    Ok(())
                },
            )
            .expect_err("operation must fail");

        assert!(matches!(
            error,
            LaunchEnqueueError::Operation("injected operation failure")
        ));
        assert_eq!(
            *owner.events.lock().expect("events"),
            ["operation", "sync", "cancel"]
        );
        assert!(transaction.is_terminal());
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn bound_enqueue_operation_panic_drops_through_sync_and_cancel() {
        let owner = Arc::new(TestTransactionOwner::default());
        let events = Arc::clone(&owner.events);
        let transaction =
            RecorderTransaction::from_admitted(Arc::clone(&owner), Box::new([test_use(0x1000)]));
        drop(owner);
        let operation_events = Arc::clone(&events);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let mut enqueued = TestEnqueuedTransaction { transaction };
            let _result: Result<(), LaunchEnqueueError<&'static str>> =
                enqueued.transaction.enqueue_operation_with(
                    || Ok(()),
                    |_| Ok(()),
                    |_| {
                        operation_events.lock().expect("events").push("operation");
                        panic!("injected operation panic");
                    },
                    |_| Ok(()),
                    |_, _| Ok(()),
                );
        }));

        assert!(panic.is_err());
        assert_eq!(
            *events.lock().expect("events"),
            ["operation", "sync", "cancel"]
        );
    }

    #[cfg(panic = "unwind")]
    struct PanicCleanupDuringUnwind {
        transaction: RecorderTransaction<TestTransactionOwner>,
        cleanup_panic_caught: Arc<AtomicBool>,
    }

    #[cfg(panic = "unwind")]
    impl Drop for PanicCleanupDuringUnwind {
        fn drop(&mut self) {
            let cleanup = catch_drop_cleanup(|| {
                self.transaction
                    .abort_with(|_| Ok(()), |_, _| panic!("injected cleanup panic"))
            });
            self.cleanup_panic_caught
                .store(cleanup.is_err(), Ordering::SeqCst);
        }
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn drop_cleanup_panic_does_not_replace_an_outer_unwind() {
        let owner = Arc::new(TestTransactionOwner::default());
        let witness = Arc::downgrade(&owner);
        let transaction =
            RecorderTransaction::from_admitted(Arc::clone(&owner), Box::new([test_use(0x1000)]));
        drop(owner);

        let cleanup_panic_caught = Arc::new(AtomicBool::new(false));
        let outer = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let cleanup_panic_caught = Arc::clone(&cleanup_panic_caught);
            move || {
                let _cleanup = PanicCleanupDuringUnwind {
                    transaction,
                    cleanup_panic_caught,
                };
                panic!("outer operation panic");
            }
        }));

        let payload = outer.expect_err("outer operation must still unwind");
        assert_eq!(
            payload.downcast_ref::<&'static str>(),
            Some(&"outer operation panic")
        );
        assert!(cleanup_panic_caught.load(Ordering::SeqCst));
        assert!(
            witness.upgrade().is_some(),
            "cleanup panic must quarantine the exact transaction owner"
        );
    }

    #[test]
    fn dependency_failure_cancels_the_complete_admitted_group() {
        let owner = Arc::new(TestTransactionOwner::default());
        let uses = Box::new([test_use(0x1000), test_use(0x2000)]);
        let mut transaction = RecorderTransaction::from_admitted(Arc::clone(&owner), uses);
        let result = transaction.enqueue_operation_with(
            || Ok(()),
            |_| Err(ResourceError::Driver("injected dependency failure".into())),
            |_| -> ResourceResult<()> { panic!("kernel must not run") },
            |owner| {
                owner.events.lock().unwrap().push("sync");
                Ok(())
            },
            |owner, pending| {
                assert_eq!(pending, &[test_use(0x1000), test_use(0x2000)]);
                owner.events.lock().unwrap().push("cancel");
                Ok(())
            },
        );
        assert!(matches!(result, Err(LaunchEnqueueError::Preparation(_))));
        assert_eq!(*owner.events.lock().unwrap(), ["sync", "cancel"]);
        assert!(transaction.is_terminal());
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn dependency_preparation_panic_keeps_the_armed_owner_for_cleanup() {
        let owner = Arc::new(TestTransactionOwner::default());
        let witness = Arc::downgrade(&owner);
        let mut transaction = RecorderTransaction::from_admitted(
            owner,
            Box::new([test_use(0x1000), test_use(0x2000)]),
        );
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<(), LaunchEnqueueError<()>> = transaction.enqueue_operation_with(
                || Ok(()),
                |_| panic!("injected dependency preparation panic"),
                |_| panic!("kernel must not run"),
                |_| Ok(()),
                |_, _| Ok(()),
            );
        }));
        assert!(panic.is_err());
        assert!(witness.upgrade().is_some());
        assert!(transaction.is_prepared());
        assert!(transaction
            .abort_with(
                |_| Err(ResourceError::Driver("completion unknown".into())),
                |_, _| panic!("unknown completion must not cancel"),
            )
            .is_err());
        assert!(witness.upgrade().is_some());
        assert!(transaction.is_terminal());
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn abort_callback_panic_quarantines_the_prepared_owner() {
        let owner = Arc::new(TestTransactionOwner::default());
        let witness = Arc::downgrade(&owner);
        let mut transaction =
            RecorderTransaction::from_admitted(Arc::clone(&owner), Box::new([test_use(0x1000)]));
        drop(owner);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            transaction
                .abort_with(|_| Ok(()), |_, _| panic!("injected cancellation panic"))
                .expect("abort should panic before returning");
        }));

        assert!(panic.is_err());
        assert!(
            witness.upgrade().is_some(),
            "abort callback panic must retain the exact owner Arc"
        );
        assert!(transaction.is_terminal());
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn abort_sync_panic_quarantines_the_enqueued_owner() {
        let owner = Arc::new(TestTransactionOwner::default());
        let witness = Arc::downgrade(&owner);
        let mut transaction =
            RecorderTransaction::from_admitted(Arc::clone(&owner), Box::new([test_use(0x1000)]));
        transaction
            .mark_may_have_enqueued()
            .expect("enqueue boundary");
        drop(owner);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            transaction
                .abort_with(|_| panic!("injected synchronization panic"), |_, _| Ok(()))
                .expect("abort should panic before returning");
        }));

        assert!(panic.is_err());
        assert!(
            witness.upgrade().is_some(),
            "abort synchronization panic must retain the exact owner Arc"
        );
        assert!(transaction.is_terminal());
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn commit_callback_panic_quarantines_the_unfinished_owner() {
        let owner = Arc::new(TestTransactionOwner::default());
        let witness = Arc::downgrade(&owner);
        let mut transaction =
            RecorderTransaction::from_admitted(Arc::clone(&owner), Box::new([test_use(0x1000)]));
        transaction
            .mark_may_have_enqueued()
            .expect("enqueue boundary");
        drop(owner);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            transaction
                .commit_with(
                    |_, _| panic!("injected finish panic"),
                    |_| Ok(()),
                    |_, _| Ok(()),
                )
                .expect("commit should panic before returning");
        }));

        assert!(panic.is_err());
        assert!(
            witness.upgrade().is_some(),
            "commit callback panic must retain the exact owner Arc"
        );
        assert!(transaction.is_terminal());
    }

    #[test]
    fn enqueue_cleanup_errors_preserve_both_failures() {
        let preparation_error: LaunchEnqueueError<()> = preparation_with_cleanup(
            ResourceError::Driver("injected preparation failure".into()),
            Err(ResourceError::Driver(
                "injected preparation cleanup failure".into(),
            )),
        );
        match preparation_error {
            LaunchEnqueueError::PreparationAndCleanup {
                preparation,
                cleanup,
            } => {
                assert!(preparation.to_string().contains("preparation failure"));
                assert!(cleanup.to_string().contains("preparation cleanup failure"));
            }
            _ => panic!("preparation cleanup failure lost its structure"),
        }

        let cleanup = ResourceError::Driver("injected operation cleanup failure".to_string());
        let operation_error = operation_with_cleanup("typed operation failure", Err(cleanup));
        match operation_error {
            LaunchEnqueueError::OperationAndCleanup { operation, cleanup } => {
                assert_eq!(operation, "typed operation failure");
                assert!(cleanup.to_string().contains("operation cleanup failure"));
            }
            _ => panic!("operation cleanup failure lost its structure"),
        }
    }

    #[test]
    fn dependency_cleanup_failure_quarantines_exact_owner_and_whole_group() {
        let owner = Arc::new(TestTransactionOwner::default());
        let witness = Arc::downgrade(&owner);
        let mut transaction = RecorderTransaction::from_admitted(
            owner,
            Box::new([test_use(0x1000), test_use(0x2000)]),
        );
        let result = transaction.enqueue_operation_with(
            || Ok(()),
            |_| Err(ResourceError::Driver("dependency failure".into())),
            |_| -> ResourceResult<()> { panic!("kernel must not run") },
            |_| Ok(()),
            |_, pending| {
                assert_eq!(pending, &[test_use(0x1000), test_use(0x2000)]);
                Err(ResourceError::Driver("cancellation failure".into()))
            },
        );
        let error = result.unwrap_err().to_string();
        assert!(error.contains("dependency failure"), "{error}");
        assert!(error.contains("cancellation failure"), "{error}");
        assert!(witness.upgrade().is_some());
        assert!(transaction.is_terminal());
    }

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
        let Some((_d, _rt, ls)) = try_async_runtime() else {
            return;
        };
        LaunchRecorder::new_permissive(ls)
            .commit()
            .expect("permissive empty");
        LaunchRecorder::new_strict(ls)
            .commit()
            .expect("strict empty");
    }

    #[test]
    fn permissive_records_native_storage_without_a_runtime_block() {
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

        // A native allocation remains owned even without a runtime block.
        let manager = Arc::new(crate::GpuMemoryManager::new(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
        ));
        let native = manager.alloc::<u8>(64).expect("native alloc");
        assert!(native.runtime_block().is_none());

        let mut rec = LaunchRecorder::new_permissive(launch_stream);
        rec.read(&native);
        assert_eq!(rec.recorded_count(), 1);
        rec.preflight(&runtime).expect("permissive preflight");
        let stream = runtime.stream_pool().resolve(launch_stream).unwrap();
        // SAFETY: the callback enqueues no additional work.
        let rec = unsafe { rec.enqueue_prepared_with(&stream, |_| Ok::<_, ResourceError>(())) }
            .expect("prepare dependencies");
        rec.commit().expect("permissive commit");
    }

    #[test]
    fn strict_retains_native_storage_after_the_callers_handle_is_dropped() {
        let Some((device, runtime, launch_stream)) = try_async_runtime() else {
            return;
        };
        let manager = Arc::new(crate::GpuMemoryManager::new(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
        ));
        let native = manager.alloc::<u8>(64).expect("native alloc");

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(&native);
        rec.preflight(&runtime).expect("strict native preflight");
        drop(native);
        assert_eq!(manager.allocated_bytes(), 64);
        rec.abort().expect("cancel before enqueue");
        assert_eq!(manager.allocated_bytes(), 0);
    }

    #[test]
    fn preflight_accepts_owned_storage_from_a_direct_runtime() {
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
        rec.preflight(&runtime).expect("direct storage preflight");
        let stream = runtime.stream_pool().resolve(launch_stream).unwrap();
        // SAFETY: the callback enqueues no additional work.
        let rec = unsafe { rec.enqueue_prepared_with(&stream, |_| Ok::<_, ResourceError>(())) }
            .expect("prepare dependencies");
        rec.commit().expect("publish storage completion");
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
        let stream = runtime.stream_pool().resolve(launch_stream).unwrap();
        // SAFETY: the callback enqueues no additional work.
        let rec = unsafe { rec.enqueue_prepared_with(&stream, |_| Ok::<_, ResourceError>(())) }
            .expect("enqueue boundary");
        rec.commit().expect("commit ok");
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
        let err = rec.commit();
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
        let Some((_d, _rt, ls)) = try_async_runtime() else {
            return;
        };
        LaunchRecorder::new_strict(ls)
            .commit()
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
        let err = rec.commit();
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
        let stream = runtime.stream_pool().resolve(launch_stream).unwrap();
        // SAFETY: the callback enqueues no additional work.
        let rec = unsafe { rec.enqueue_prepared_with(&stream, |_| Ok::<_, ResourceError>(())) }
            .expect("enqueue boundary");
        rec.commit().expect("commit ok");
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
        let stream = runtime.stream_pool().resolve(launch_stream).unwrap();
        // SAFETY: the callback enqueues no additional work.
        let rec = unsafe { rec.enqueue_prepared_with(&stream, |_| Ok::<_, ResourceError>(())) }
            .expect("enqueue boundary");
        rec.commit().expect("commit");
    }

    /// Locks the dedup key: `(ptr, generation, device_ordinal)`,
    /// not `ptr` alone. Two `BlockUse`s sharing a ptr but
    /// differing in generation MUST be treated as distinct
    /// entries — otherwise an ABA reuse inside a single recorder
    /// would silently collapse an event for the new allocation
    /// onto the old block's prepare/finish chain.
    #[test]
    fn dedup_keys_on_full_block_id_not_ptr_alone() {
        // Construct two block uses with the same ptr but
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
            BlockUse {
                block: block_a,
                bytes: 64,
                access: Access::Read,
            },
            BlockUse {
                block: block_b,
                bytes: 64,
                access: Access::Write,
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
            BlockUse {
                block: block_a,
                bytes: 64,
                access: Access::Read,
            },
            BlockUse {
                block: block_a,
                bytes: 64,
                access: Access::Write,
            },
        ];
        let collapsed = dedup_uses(&same_id);
        assert_eq!(collapsed.len(), 1);
        assert_eq!(collapsed[0].access, Access::ReadWrite);
    }

    #[test]
    fn dedup_distinguishes_allocation_stream_in_full_block_identity() {
        let block_a = BlockId {
            ptr: 0xdead_beef,
            generation: Generation(1),
            alloc_stream: StreamId(7),
            device_ordinal: 0,
        };
        let block_b = BlockId {
            alloc_stream: StreamId(11),
            ..block_a
        };
        let uses = vec![
            BlockUse {
                block: block_a,
                bytes: 64,
                access: Access::Read,
            },
            BlockUse {
                block: block_b,
                bytes: 64,
                access: Access::Write,
            },
        ];

        let deduped = dedup_uses(&uses);
        assert_eq!(deduped.len(), 2, "allocation streams are part of BlockId");
        assert_eq!(deduped[0].block.alloc_stream, StreamId(7));
        assert_eq!(deduped[1].block.alloc_stream, StreamId(11));
    }

    #[test]
    fn read_device_block_snapshots_complete_identity_as_read() {
        let block = DeviceBlock {
            ptr: 0x1234,
            device_ordinal: 2,
            alloc_stream: StreamId(5),
            bytes: 64,
            align: 16,
            tag: crate::device_runtime::AllocTag::UNTAGGED,
            generation: Generation(9),
            state: crate::device_runtime::BlockState::Live,
        };
        let expected = BlockId::from_block(&block);
        let mut recorder = LaunchRecorder::new_strict(StreamId(8));

        recorder.read_device_block(&block);

        assert_eq!(recorder.uses.len(), 1);
        assert_eq!(recorder.uses[0].block, expected);
        assert_eq!(recorder.uses[0].bytes, block.bytes);
        assert_eq!(recorder.uses[0].access, Access::Read);
        recorder.transaction.mark_terminal();
    }

    #[test]
    fn read_block_identity_records_prevalidated_receipt_pointee() {
        let identity = BlockId {
            ptr: 0x9876,
            generation: Generation(12),
            alloc_stream: StreamId(4),
            device_ordinal: 3,
        };
        let mut recorder = LaunchRecorder::new_strict(StreamId(6));

        recorder.read_block_identity((identity, 128));

        assert_eq!(recorder.uses.len(), 1);
        assert_eq!(recorder.uses[0].block, identity);
        assert_eq!(recorder.uses[0].bytes, 128);
        assert_eq!(recorder.uses[0].access, Access::Read);
        recorder.transaction.mark_terminal();
    }

    #[test]
    fn bound_identity_uses_arc_ownership_not_value_equality() {
        let owner = Arc::new(());
        let same_owner = Arc::clone(&owner);
        let equal_value_foreign_owner = Arc::new(());

        assert!(same_arc_identity(Some(&owner), &same_owner));
        assert!(!same_arc_identity(Some(&owner), &equal_value_foreign_owner));
        assert!(!same_arc_identity::<()>(None, &owner));
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
        let stream = runtime.stream_pool().resolve(launch_stream).unwrap();
        // SAFETY: the callback enqueues no additional work.
        let rec = unsafe { rec.enqueue_prepared_with(&stream, |_| Ok::<_, ResourceError>(())) }
            .expect("enqueue boundary");
        rec.commit().expect("commit");
    }
}
