//! Model expenditure inside an original admitted cold operation. This owner has
//! no evaluation, backward-tape, content-admission or publication authority.

use super::*;
use crate::semantic_work::ModelWorkPlan;

/// Original signed record custody. Bounded callers retain the loaded byte owner;
/// ordinary unbound callers retain their existing immutable record allocation.
pub enum SemanticColdModelWorkAdmission {
    Ordinary(Arc<[u8]>),
    Loaded(SemanticReplayLoadedMaterial),
}

impl From<Arc<[u8]>> for SemanticColdModelWorkAdmission {
    fn from(bytes: Arc<[u8]>) -> Self {
        Self::Ordinary(bytes)
    }
}

impl From<SemanticReplayLoadedMaterial> for SemanticColdModelWorkAdmission {
    fn from(bytes: SemanticReplayLoadedMaterial) -> Self {
        Self::Loaded(bytes)
    }
}

impl AsRef<[u8]> for SemanticColdModelWorkAdmission {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Ordinary(bytes) => bytes,
            Self::Loaded(bytes) => bytes.as_bytes(),
        }
    }
}

struct RetainedColdAdmission {
    material: SemanticColdModelWorkAdmission,
    _reservation: Option<SemanticReplayMaterialReservation>,
}

impl std::fmt::Debug for RetainedColdAdmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedColdAdmission")
            .field("bytes", &self.material.as_ref().len())
            .finish_non_exhaustive()
    }
}

impl RetainedColdAdmission {
    fn same_material(&self, other: &SemanticColdModelWorkAdmission) -> bool {
        match (&self.material, other) {
            (
                SemanticColdModelWorkAdmission::Ordinary(left),
                SemanticColdModelWorkAdmission::Ordinary(right),
            ) => Arc::ptr_eq(left, right),
            (
                SemanticColdModelWorkAdmission::Loaded(left),
                SemanticColdModelWorkAdmission::Loaded(right),
            ) => {
                Arc::ptr_eq(&left.bytes.owner, &right.bytes.owner)
                    && left.bytes.range == right.bytes.range
            }
            _ => false,
        }
    }

    fn matches_instruction(&self, instruction: &Arc<[u8]>) -> bool {
        matches!(&self.material, SemanticColdModelWorkAdmission::Ordinary(original) if Arc::ptr_eq(original, instruction))
    }

    fn instruction(&self) -> Result<&Arc<[u8]>, SemanticTransitionError> {
        match &self.material {
            SemanticColdModelWorkAdmission::Ordinary(instruction) => Ok(instruction),
            SemanticColdModelWorkAdmission::Loaded(_) => {
                Err(SemanticTransitionError::ObservationMismatch)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct SemanticColdModelWork {
    issuer: Arc<()>,
    invocation: Arc<()>,
    token: u64,
    operation_ordinal: u64,
    admission: Arc<RetainedColdAdmission>,
    final_cursor: Arc<std::sync::OnceLock<CompletedColdReportCursor>>,
}

/// The actual enclosing native phase selects its producer roster. Neither an
/// external model callback nor the caller's region geometry selects this law.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticColdModelWorkPurpose {
    ControlRetirement,
    Delivery,
    IntermediateRestore,
    TerminalRefusal,
    SourceEvaluation,
    PrivateEvaluation,
    PrivateCheckpoint,
    PrivatePrefix,
    PrivateModelPreparation,
    PrivateSuccessorPreparation,
    PrivateRetirement,
    PrivateAdoption,
    PrivateExecution,
    PrivateRestore,
    PreparedActorRefresh,
}

/// A single registration region inside the original cold report. Regions
/// append to the same event roster and device slots; they never reset or reopen
/// the enclosing report and grant no numerical evaluation authority.
#[derive(Clone, Debug)]
pub struct SemanticColdModelWorkRegion {
    work: SemanticColdModelWork,
    index: usize,
}

impl SemanticColdModelWorkRegion {
    /// Original position in the enclosing native report's registered roster.
    pub fn ordinal(&self) -> usize {
        self.index
    }
}

/// An original plan occurrence, issued before the corresponding model effect.
#[derive(Clone)]
pub struct SemanticColdModelWorkOperation {
    work: SemanticColdModelWork,
    index: usize,
    issuance: Arc<()>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OperationState {
    Prepared,
    Entered,
    Completing,
    Complete,
    NotEntered,
    Unknown,
}

struct ColdOperationAttempt {
    issuance: Arc<()>,
    region: Option<usize>,
    event: ModelWorkEvent,
    slot: usize,
    state: OperationState,
}

impl SemanticColdModelWork {
    pub fn same_invocation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.issuer, &other.issuer)
            && Arc::ptr_eq(&self.invocation, &other.invocation)
            && self.token == other.token
    }

    /// Immutable proof authored only by the original known native completion.
    /// It survives release without acquiring another Session or reader lease.
    pub fn completed_report_cursor(
        &self,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<Option<SemanticColdModelWorkReportCursor>, SemanticTransitionError> {
        if region.is_some_and(|region| !region.work.same_invocation(self)) {
            return Err(publication_input_error(
                "completed cold cursor changed its original region",
            ));
        }
        let Some(proof) = self.final_cursor.get() else {
            return Ok(None);
        };
        let index = match region {
            Some(region) if proof.segmented => region.index,
            None if !proof.segmented => 0,
            _ => {
                return Err(publication_input_error(
                    "completed cold cursor requires its original region",
                ));
            }
        };
        proof.regions.get(index).cloned().map(Some).ok_or_else(|| {
            publication_input_error("completed cold cursor changed its original region")
        })
    }

    /// Checked bytes for this exact native work-buffer ABI, not measured usage.
    pub fn allocation_bytes(capacity: usize) -> Result<usize, SemanticTransitionError> {
        if capacity == 0 {
            return Err(publication_input_error(
                "cold model work requires a positive original event capacity",
            ));
        }
        capacity
            .checked_mul(size_of::<ModelWorkEvent>() + 3 * size_of::<u64>())
            .and_then(|bytes| bytes.checked_add((15 + 11) * size_of::<u64>()))
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }
}

/// The original operation's native tally carried into its actual child graph.
/// Construction is private; neither a raw pointer nor another report can mint
/// this custody. Allocation and stream identity remain those of the issuer.
#[derive(Clone)]
pub struct SemanticColdNativeWork {
    provider: Arc<CudaKernelProvider>,
    domain: ResidentExecutionDomain,
    work: DeviceMemoryView<u64>,
    custody: Arc<()>,
    allowance: Arc<Mutex<native_work_bound::ColdNativeAllowance>>,
}

impl SemanticColdNativeWork {
    pub fn same_custody(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.custody, &other.custody)
            && Arc::ptr_eq(&self.allowance, &other.allowance)
            && Arc::ptr_eq(&self.provider, &other.provider)
            && self.domain.stream_id() == other.domain.stream_id()
    }

    pub(crate) fn graph_custody(
        self,
        provider: &Arc<CudaKernelProvider>,
        domain: &ResidentExecutionDomain,
    ) -> Result<
        (
            DeviceMemoryView<u64>,
            Arc<()>,
            Arc<Mutex<native_work_bound::ColdNativeAllowance>>,
        ),
        SemanticTransitionError,
    > {
        if !Arc::ptr_eq(provider, &self.provider) || domain.stream_id() != self.domain.stream_id() {
            return Err(publication_input_error(
                "cold child construction changed its original allocation owner or stream",
            ));
        }
        Ok((self.work, self.custody, self.allowance))
    }
}

/// One admitted native restore before its original model callback begins.
/// This receipt retains the original tally; it grants no numerical entry.
pub struct SemanticColdResidentRestorationPrelude {
    original: SemanticColdModelWork,
    restoration: std::sync::Weak<resident_model_memory::ResidentModelRestorationOwner>,
    native_work: SemanticColdNativeWork,
    _reservation: SemanticReplayMaterialReservation,
}

impl SemanticColdResidentRestorationPrelude {
    pub fn require_restoration(
        &self,
        restoration: &SemanticPreparedResidentModelRestoration,
    ) -> Result<(), SemanticTransitionError> {
        if !self.restoration.ptr_eq(&Arc::downgrade(&restoration.owner)) {
            return Err(publication_input_error(
                "native restoration prelude changed its original restoration issuer",
            ));
        }
        restoration.require_report(&self.original, 2)
    }

    pub fn shared_native_work(&self) -> &SemanticColdNativeWork {
        &self.native_work
    }

    pub(super) fn allowance(&self) -> &Arc<Mutex<native_work_bound::ColdNativeAllowance>> {
        &self.native_work.allowance
    }

    pub(super) fn require_callback_owner(
        &self,
        callback: &SemanticTransitionSession,
    ) -> Result<(), SemanticTransitionError> {
        if !Arc::ptr_eq(&self.native_work.provider, &callback.provider)
            || self.native_work.domain.stream_id() != callback.domain.stream_id()
            || callback
                .replay_material_budget
                .as_ref()
                .is_none_or(|budget| !self._reservation.budget().same_pool(budget))
            || callback
                .graph
                .cold_work_allowance()
                .as_ref()
                .is_none_or(|allowance| !Arc::ptr_eq(allowance, self.allowance()))
        {
            return Err(publication_input_error(
                "native restoration prelude changed its original callback domain or allowance",
            ));
        }
        callback
            .graph
            .require_cold_work_owner(&self.native_work.work, true)
            .map_err(SemanticTransitionError::Semantic)
    }
}

/// Actual model and native components, not the complete operation's
/// native work or physical peak. Other cold producers remain independent.
#[derive(Clone, Copy, Debug)]
pub struct SemanticColdModelWorkResult {
    pub model_work: u64,
    pub operation_count: u64,
    pub work_bound: u64,
    pub model_calls: u64,
    pub native_work: u64,
    pub native_events: [u64; 9],
}

impl SemanticColdModelWorkResult {
    fn include(&mut self, other: Self) -> Result<(), SemanticTransitionError> {
        let add = |left: u64, right: u64| {
            left.checked_add(right)
                .ok_or(SemanticTransitionError::GenerationExhausted)
        };
        self.model_work = add(self.model_work, other.model_work)?;
        self.operation_count = add(self.operation_count, other.operation_count)?;
        self.work_bound = add(self.work_bound, other.work_bound)?;
        self.model_calls = add(self.model_calls, other.model_calls)?;
        self.native_work = add(self.native_work, other.native_work)?;
        for (total, actual) in self.native_events.iter_mut().zip(other.native_events) {
            *total = add(*total, actual)?;
        }
        Ok(())
    }
}

pub(super) struct CompletedModelContextColdResult {
    instruction: Arc<[u8]>,
    result: SemanticColdModelWorkResult,
}

impl CompletedModelContextColdResult {
    /// The continuation owner has prechecked this same instruction and original
    /// aggregate before handing its freshly issued claim to this final binding.
    pub(super) fn bind(self, original: &mut actor_refresh_program::InstructionAdmission) {
        assert_eq!(original.instruction.as_ref(), self.instruction.as_ref());
        assert!(original.cold_result.is_none());
        original.cold_result = Some(self.result);
    }
}

/// Metadata of the exact original report, not model expenditure or GPU completion.
#[derive(Clone, Debug)]
pub struct SemanticColdModelWorkReportCursor {
    pub region_begin: usize,
    pub region_end: usize,
    pub declared_cursor: usize,
    pub recorded_slots: usize,
    pub skipped_declared_spans: Vec<(usize, usize)>,
    pub skipped_region_ordinals: Vec<usize>,
    pub recording_closed: bool,
    pub report_completed: bool,
}

#[derive(Debug)]
struct CompletedColdReportCursor {
    segmented: bool,
    regions: Vec<SemanticColdModelWorkReportCursor>,
}

/// Authored by the enclosing native operation after its known outcome, never by
/// a callback error or a Python declaration of partial work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticColdModelWorkDisposition {
    Complete,
    KnownRefusal,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RecordingState {
    Waiting,
    Recording,
    Closed,
    Failed,
    Submitting,
    Submitted,
    Completed,
    NotEntered,
}

#[derive(Clone, Copy)]
enum ColdWorkReader<'a> {
    Published(&'a SemanticPublishedLease),
    Admitted(
        &'a SemanticSegmentInstructionAdmission,
        &'a SemanticPublishedLease,
    ),
    AdmittedCancellation(
        &'a SemanticSegmentInstructionAdmission,
        &'a SemanticPublishedLease,
        &'a SemanticPreparedSegmentNonSubmission,
    ),
    AdmittedRetirement(&'a SemanticSegmentInstructionAdmission),
    Prepared(&'a SemanticPreparedStep, &'a SemanticPublishedLease),
    FrozenPrepared(
        &'a SemanticModelContextContinuation,
        &'a SemanticPreparedStep,
    ),
    ModelContextActivation(
        &'a SemanticModelContextContinuation,
        &'a SemanticTransitionSession,
        &'a SemanticPublishedLease,
    ),
    ModelContextMaterial(
        &'a SemanticModelContextContinuation,
        &'a SemanticTransitionSession,
        &'a SemanticPublishedLease,
    ),
}

#[derive(Clone, Copy)]
enum InstructionColdRole {
    Preparation,
    Retirement,
    Prepared,
}

enum ColdWorkAdmission<'a> {
    Instruction(&'a SemanticSegmentInstructionAdmission, InstructionColdRole),
    ModelContext(
        &'a SemanticModelContextContinuation,
        ModelContextColdKind,
        Option<SemanticColdModelWorkResult>,
    ),
}

pub(super) struct ModelContextPrefillContentBinding {
    context: SemanticModelContextContinuation,
    scope: Arc<()>,
    occurrence: usize,
    kind: InitialPrefillCaptureKind,
    verify: bool,
    activating: bool,
    restored: Option<SemanticRestoredModelContextPrefill>,
    allowance: Arc<Mutex<native_work_bound::ColdNativeAllowance>>,
}

impl ModelContextPrefillContentBinding {
    pub(super) fn bind(
        &self,
        domain: &ResidentExecutionDomain,
        tensors: &TensorContentBuffers,
    ) -> Result<(), SemanticTransitionError> {
        let mut original = tensors
            .original_cold
            .lock()
            .map_err(|_| publication_input_error("imported prefill content custody is poisoned"))?;
        if let Some(batch) = original.as_ref() {
            let same_allowance = batch
                .allowance
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
                .is_some_and(|owner| Arc::ptr_eq(&owner, &self.allowance));
            if !same_allowance && (!self.activating || batch.cursor != batch.commands.len()) {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            if same_allowance && batch.context_prefill_occurrence == Some(self.occurrence) {
                if batch.verify != self.verify {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                return Ok(());
            }
            if batch.context_prefill_occurrence.is_none() || batch.cursor != batch.commands.len() {
                return Err(publication_input_error(
                    "context prefill cannot replace an unresolved original content occurrence",
                ));
            }
        }
        let count = tensors
            .tensors
            .len()
            .checked_add(usize::from(matches!(
                tensors.seals,
                TensorContentSeals::Model(_)
            )))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut batch = if let Some(proof) = &self.restored {
            proof.take_consumer_batch(self.occurrence, self.kind, tensors)?
        } else {
            native_work_bound::OriginalContentBatch::new(domain, self.verify, count)?
        };
        if let Some(proof) = &self.restored {
            if !self.verify {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            proof.require_consumer_content_slot(self.occurrence, self.kind, tensors)?;
        } else {
            self.allowance
                .lock()
                .map_err(|_| publication_input_error("imported prefill allowance is poisoned"))?
                .claim_model_context_prefill_content(
                    &self.context,
                    &self.scope,
                    self.occurrence,
                    tensors,
                    self.verify,
                )?;
        }
        batch.allowance = Some(Arc::downgrade(&self.allowance));
        batch.context_prefill_occurrence = Some(self.occurrence);
        *original = Some(batch);
        Ok(())
    }

    pub(super) fn complete(
        &self,
        tensors: &TensorContentBuffers,
    ) -> Result<(), SemanticTransitionError> {
        if let Some(proof) = &self.restored {
            proof.retain_completed_consumer_batch(self.occurrence, tensors)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelContextColdKind {
    Preparation,
    Activation,
    Checkpoint,
    DormantRestore,
}

/// Borrowed from SOURCE's actual counted report at its original reader.
/// This grants material transport/restoration, never a TARGET model callback.
pub(super) struct ModelContextMaterialWork {
    owner: std::sync::Weak<model_context_continuation::ModelContextContinuationOwner>,
    pub(super) work: SemanticColdModelWork,
    source_parent: SemanticPublishedIdentity,
    native_work: SemanticColdNativeWork,
    dma_ceiling: [u64; 9],
    callback_region: Option<SemanticColdModelWorkRegion>,
    _callback_reservation: Option<SemanticReplayMaterialReservation>,
    restoring: bool,
    admitted: bool,
    pending: bool,
    result: Option<SemanticColdModelWorkResult>,
    prior: Option<SemanticColdModelWorkResult>,
}

struct ModelContextColdWork {
    owner: std::sync::Weak<model_context_continuation::ModelContextContinuationOwner>,
    kind: ModelContextColdKind,
    budget: [u64; 3],
    prior: Option<SemanticColdModelWorkResult>,
    native_ceiling: [u64; 9],
}

struct EvaluationCleanup {
    region: usize,
    original: SemanticModelEvaluation,
    cancellation: Option<SemanticCancelledModelEvaluation>,
}

struct ColdReportCompletion {
    metadata: Option<crate::device::RetainedDeviceWrite<ModelWorkEvent>>,
    metadata_counted: bool,
    metadata_recorded: bool,
    metadata_completed: bool,
    command: OriginalNativeCommand,
    read: PublicationRead<u64>,
    poisoned: bool,
}

pub(super) struct ColdModelWorkStorage {
    invocation: Arc<()>,
    final_cursor: Arc<std::sync::OnceLock<CompletedColdReportCursor>>,
    aliases: Arc<()>,
    purpose: SemanticColdModelWorkPurpose,
    prepared_scope: Option<Arc<()>>,
    model_context: Option<ModelContextColdWork>,
    operation_ordinal: u64,
    admission: Arc<RetainedColdAdmission>,
    allowance: Arc<Mutex<native_work_bound::ColdNativeAllowance>>,
    source_geometry: Option<Arc<prepared_source_content::OriginalSourceGeometryRoster>>,
    capacity_limit: Option<usize>,
    work: Option<PreparedModelWork>,
    native_work: TrackedCudaSlice<u64>,
    initialization_reset: Option<OriginalNativeCommand>,
    initialization_write: Option<crate::device::RetainedDeviceWrite<u64>>,
    initialization_write_completed: bool,
    initialization_poisoned: bool,
    native_initialized: bool,
    native_initialization_recorded: bool,
    initialized: bool,
    report: TrackedCudaSlice<u64>,
    completion: Option<Arc<Mutex<ColdReportCompletion>>>,
    state: RecordingState,
    regions: Vec<RecordingState>,
    plan: Option<ModelWorkPlan>,
    evaluation_cleanup: Option<EvaluationCleanup>,
    disposition: Option<SemanticColdModelWorkDisposition>,
    result: Option<SemanticColdModelWorkResult>,
    streams: Option<Vec<u64>>,
    operations: Vec<ColdOperationAttempt>,
    pending_operation: Option<usize>,
    pending_operation_end: usize,
    stopped_before_entry: bool,
}

/// Fresh authority borrowed from the original recording report. It cannot
/// outlive the owner borrow or become a retained permission for a later call.
pub struct SemanticColdNativeWorkActivationPermit<'a> {
    owner: &'a SemanticTransitionSession,
    storage: &'a ColdModelWorkStorage,
    region: Option<usize>,
}

/// Comparison custody of an original activation, not a reusable permission.
#[derive(Clone)]
pub struct SemanticColdNativeWorkActivationIdentity {
    invocation: Arc<()>,
    admission: Arc<RetainedColdAdmission>,
    operation_ordinal: u64,
    region: Option<usize>,
    allowance: Arc<Mutex<native_work_bound::ColdNativeAllowance>>,
    work: DeviceMemoryView<u64>,
    callback_issuer: Arc<()>,
    callback_token: u64,
    callback_identity: SemanticPublishedIdentity,
}

impl SemanticColdNativeWorkActivationIdentity {
    pub fn same_context(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.invocation, &other.invocation)
            && Arc::ptr_eq(&self.admission, &other.admission)
            && self.operation_ordinal == other.operation_ordinal
            && self.region == other.region
            && Arc::ptr_eq(&self.allowance, &other.allowance)
            && self.work.device_ptr() == other.work.device_ptr()
            && self.work.len() == other.work.len()
            && Arc::ptr_eq(&self.callback_issuer, &other.callback_issuer)
            && self.callback_token == other.callback_token
            && self.callback_identity == other.callback_identity
    }

    pub(super) fn work(&self) -> &DeviceMemoryView<u64> {
        &self.work
    }

    pub(super) fn allowance(&self) -> &Arc<Mutex<native_work_bound::ColdNativeAllowance>> {
        &self.allowance
    }

    pub(super) fn region_ordinal(&self) -> usize {
        self.region.unwrap_or(0)
    }
}

impl SemanticColdNativeWorkActivationPermit<'_> {
    pub fn identity_for_callback(
        &self,
        callback: &SemanticTransitionSession,
        parent: &SemanticPublishedLease,
    ) -> Result<SemanticColdNativeWorkActivationIdentity, SemanticTransitionError> {
        callback.require_original_cold_content_reader(parent)?;
        if !Arc::ptr_eq(&self.owner.provider, &callback.provider)
            || self.owner.domain.stream_id() != callback.domain.stream_id()
            || callback
                .graph
                .cold_work_allowance()
                .as_ref()
                .is_none_or(|allowance| !Arc::ptr_eq(allowance, &self.storage.allowance))
        {
            return Err(publication_input_error(
                "cold activation changed its original callback domain or allowance",
            ));
        }
        callback
            .graph
            .require_cold_work_owner(
                &self.storage.native_work.view(),
                !Arc::ptr_eq(&self.owner.publication_issuer, &callback.publication_issuer),
            )
            .map_err(SemanticTransitionError::Semantic)?;
        Ok(SemanticColdNativeWorkActivationIdentity {
            invocation: Arc::clone(&self.storage.invocation),
            admission: Arc::clone(&self.storage.admission),
            operation_ordinal: self.storage.operation_ordinal,
            region: self.region,
            allowance: Arc::clone(&self.storage.allowance),
            work: self.storage.native_work.view(),
            callback_issuer: Arc::clone(&callback.publication_issuer),
            callback_token: parent.token,
            callback_identity: parent.identity,
        })
    }
}

impl ColdModelWorkStorage {
    fn work(&self) -> Result<&PreparedModelWork, SemanticTransitionError> {
        self.work.as_ref().ok_or_else(|| {
            publication_input_error("cold model slots require the original admitted nonempty plan")
        })
    }

    fn work_mut(&mut self) -> Result<&mut PreparedModelWork, SemanticTransitionError> {
        self.work.as_mut().ok_or_else(|| {
            publication_input_error("cold model slots require the original admitted nonempty plan")
        })
    }

    fn recorded_events(&self) -> &[ModelWorkEvent] {
        self.work
            .as_ref()
            .map_or(&[], |work| work.recording.events())
    }

    fn report_cursor(
        &self,
        region: Option<usize>,
        completed: bool,
    ) -> Result<SemanticColdModelWorkReportCursor, SemanticTransitionError> {
        let storage = self;
        if !completed
            && !matches!(
                storage.state,
                RecordingState::Waiting
                    | RecordingState::Recording
                    | RecordingState::Closed
                    | RecordingState::Completed
            )
            || storage.pending_operation.is_some()
            || (!completed && storage.stopped_before_entry)
            || storage.operations.iter().any(|attempt| {
                if completed {
                    matches!(
                        attempt.state,
                        OperationState::Entered
                            | OperationState::Completing
                            | OperationState::Unknown
                    )
                } else {
                    attempt.state != OperationState::Complete
                }
            })
            || region.is_some_and(|index| storage.regions.get(index).is_none())
            || (region.is_none() && !storage.regions.is_empty())
        {
            return Err(publication_input_error(
                "cold cursor requires its known original report without unresolved or stopped work",
            ));
        }
        let plan = storage.plan()?;
        plan.require_recorded_trace(storage.recorded_events())
            .map_err(publication_input_error)?;
        let (region_begin, region_end, declared_cursor, recorded_slots, skipped_declared_spans) =
            plan.cursor_projection(region)
                .map_err(publication_input_error)?;
        let recording_closed = completed
            || matches!(
                storage.state,
                RecordingState::Closed | RecordingState::Completed
            );
        if !completed
            && recording_closed
            && (!plan.complete()
                || storage.regions.iter().any(|state| {
                    !matches!(state, RecordingState::Closed | RecordingState::NotEntered)
                }))
            || (!completed
                && storage.state == RecordingState::Completed
                && storage.result.is_none())
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(SemanticColdModelWorkReportCursor {
            region_begin,
            region_end,
            declared_cursor,
            recorded_slots,
            skipped_declared_spans,
            skipped_region_ordinals: storage
                .regions
                .iter()
                .enumerate()
                .filter_map(|(index, state)| {
                    (*state == RecordingState::NotEntered).then_some(index)
                })
                .collect(),
            recording_closed,
            report_completed: completed || storage.state == RecordingState::Completed,
        })
    }

    fn plan(&self) -> Result<&ModelWorkPlan, SemanticTransitionError> {
        self.plan.as_ref().ok_or_else(|| {
            publication_input_error("cold recording requires its admitted operation plan")
        })
    }

    fn plan_mut(&mut self) -> Result<&mut ModelWorkPlan, SemanticTransitionError> {
        self.plan.as_mut().ok_or_else(|| {
            publication_input_error("cold recording requires its admitted operation plan")
        })
    }
}

impl SemanticTransitionSession {
    /// Stage one material-only operation at SOURCE's actual post-retirement
    /// reader. Its TARGET borrower never gains instruction or model rights.
    pub(super) fn prepare_model_context_material_work(
        &mut self,
        target: &mut Self,
        context: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        restoring: bool,
        dma_ceiling: [u64; 9],
        native_ceiling: [u64; 9],
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.require_model_context_cold_pair(context, target)?;
        self.checked_original_consumer_step(parent)?;
        if !Arc::ptr_eq(&self.publication_issuer, &context.inner.source_issuer)
            || !Arc::ptr_eq(&target.publication_issuer, &context.inner.target_issuer)
            || !parent.active
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if let Some(original) = target
            .model_context_material_work
            .as_ref()
            .filter(|original| original.result.is_none())
        {
            if !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
                || original.source_parent != parent.identity
                || original.restoring != restoring
                || original.dma_ceiling != dma_ceiling
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let handle = original.work.clone();
            let storage = self.original_cold_model_work(&handle)?;
            if storage
                .model_context
                .as_ref()
                .is_none_or(|original| original.native_ceiling != native_ceiling)
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            return Ok(handle);
        }
        self.ensure_quiescent()?;
        self.checked_reader(parent)?;
        let program = context
            .inner
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        self.require_original_program_context_checkpoint(&program, parent)?;
        drop(program);
        self.require_completed_cold_model_work()?;
        if target.graph.borrowed_cold_work().is_some() {
            return Err(publication_input_error(
                "context material cannot replace another original borrowed report",
            ));
        }
        let prior = target
            .model_context_material_work
            .as_ref()
            .map(|original| {
                if !original.owner.ptr_eq(&Arc::downgrade(&context.inner)) {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                let mut result = original
                    .result
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if let Some(previous) = original.prior {
                    result.include(previous)?;
                }
                Ok(result)
            })
            .transpose()?;
        let kind = if restoring {
            ModelContextColdKind::DormantRestore
        } else {
            ModelContextColdKind::Checkpoint
        };
        let purpose = if restoring {
            SemanticColdModelWorkPurpose::PrivateRestore
        } else {
            SemanticColdModelWorkPurpose::PrivateCheckpoint
        };
        let handle = self.install_cold_model_work(
            parent.token,
            None,
            parent.model_context_header()?.proposal,
            context.inner.original_program_instruction()?.into(),
            purpose,
            Some(Arc::clone(&context.inner.target_issuer)),
            Some(ColdWorkAdmission::ModelContext(context, kind, prior)),
        )?;
        let native_work = self.share_cold_native_work(&handle)?;
        let callback_reservation = if restoring {
            Some(
                self.replay_material_budget
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?
                    .reserve_auxiliary(
                        (size_of::<SemanticColdModelWorkRegion>()
                            + size_of::<RecordingState>()
                            + 2 * size_of::<Vec<usize>>()) as u64,
                    )?,
            )
        } else {
            None
        };
        let callback_region = if restoring {
            self.prepare_cold_model_work_regions(&handle, 1)?.pop()
        } else {
            None
        };
        target
            .graph
            .begin_borrowed_cold_work(
                native_work.work.clone(),
                Arc::clone(&native_work.custody),
                Arc::clone(&native_work.allowance),
            )
            .map_err(SemanticTransitionError::Semantic)?;
        target.model_context_material_work = Some(ModelContextMaterialWork {
            owner: Arc::downgrade(&context.inner),
            work: handle.clone(),
            source_parent: parent.identity,
            native_work,
            dma_ceiling,
            callback_region,
            _callback_reservation: callback_reservation,
            restoring,
            admitted: false,
            pending: false,
            result: None,
            prior,
        });
        self.freeze_model_context_cold_native_work(&handle, native_ceiling)?;
        self.original_cold_model_work(&handle)?
            .allowance
            .lock()
            .map_err(|_| publication_input_error("context material allowance is poisoned"))?
            .freeze_model_context_material_dma(dma_ceiling)?;
        Ok(handle)
    }

    fn stage_model_context_material_work(
        &self,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
        restoring: bool,
    ) -> Result<&ModelContextMaterialWork, SemanticTransitionError> {
        let original = self
            .model_context_material_work
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&self.publication_issuer, &context.inner.target_issuer)
            || !Arc::ptr_eq(&work.issuer, &context.inner.source_issuer)
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !Arc::ptr_eq(owner, &context.inner))
            || !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
            || !original.work.same_invocation(work)
            || original.restoring != restoring
            || original.result.is_some()
            || !Arc::ptr_eq(
                self.provider.memory(),
                original.native_work.provider.memory(),
            )
            || self.domain.stream_id() != original.native_work.domain.stream_id()
            || self
                .graph
                .cold_work_allowance()
                .as_ref()
                .is_none_or(|allowance| !Arc::ptr_eq(allowance, &original.native_work.allowance))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.graph
            .require_cold_work_owner(&original.native_work.work, true)
            .map_err(SemanticTransitionError::Semantic)?;
        Ok(original)
    }

    fn require_model_context_material_work(
        &self,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
        restoring: bool,
    ) -> Result<&ModelContextMaterialWork, SemanticTransitionError> {
        let original = self.stage_model_context_material_work(context, work, restoring)?;
        if !original.admitted {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(original)
    }

    pub(super) fn stage_model_context_material_work_allowance(
        &self,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
        restoring: bool,
    ) -> Result<Arc<Mutex<native_work_bound::ColdNativeAllowance>>, SemanticTransitionError> {
        Ok(Arc::clone(
            &self
                .stage_model_context_material_work(context, work, restoring)?
                .native_work
                .allowance,
        ))
    }

    /// Admit and initialize only after all original material owners are retained.
    pub(super) fn initialize_model_context_material_work(
        &mut self,
        target: &mut Self,
        context: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.require_model_context_cold_pair(context, target)?;
        self.checked_original_consumer_step(parent)?;
        let original = target.stage_model_context_material_work(
            context,
            work,
            target
                .model_context_material_work
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .restoring,
        )?;
        if original.source_parent != parent.identity
            || !Arc::ptr_eq(&self.publication_issuer, &context.inner.source_issuer)
            || !Arc::ptr_eq(&target.publication_issuer, &context.inner.target_issuer)
            || !parent.active
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.original_cold_model_work(work)?;
        let outcome = (|| {
            if self.original_cold_model_work(work)?.plan.is_some() {
                self.resume_cold_model_work_plan(work)?;
            } else {
                if target
                    .model_context_material_work
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?
                    .restoring
                {
                    return Err(publication_input_error(
                        "restoration requires its complete original factory plan before admission",
                    ));
                }
                self.admit_cold_model_work_plan_with_sources(work, &[], &[], None, &[], &[])?;
            }
            if self.original_cold_model_work(work)?.state == RecordingState::Waiting {
                self.begin_cold_model_work(work)?;
            }
            Ok::<_, SemanticTransitionError>(())
        })();
        target
            .model_context_material_work
            .as_mut()
            .expect("original material report")
            .pending = self.original_cold_completion_pending(work)?;
        outcome?;
        target
            .model_context_material_work
            .as_mut()
            .expect("original material report")
            .admitted = true;
        Ok(())
    }

    pub fn model_context_material_restore_callback_region(
        &self,
        target: &Self,
        context: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
    ) -> Result<SemanticColdModelWorkRegion, SemanticTransitionError> {
        self.require_model_context_cold_pair(context, target)?;
        self.checked_original_consumer_step(parent)?;
        let original = target.stage_model_context_material_work(context, work, true)?;
        if original.source_parent != parent.identity
            || self.original_cold_model_work(work)?.regions.len() != 1
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        original
            .callback_region
            .clone()
            .ok_or(SemanticTransitionError::NotBound)
    }

    pub fn require_model_context_material_restore_callback_work(
        &self,
        target: &Self,
        context: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
        region: &SemanticColdModelWorkRegion,
    ) -> Result<(), SemanticTransitionError> {
        self.require_model_context_restore_source_callback_work(
            context, parent, work, region, false,
        )?;
        let original =
            self.model_context_material_restore_callback_region(target, context, parent, work)?;
        let storage = self.original_cold_model_work(work)?;
        if original.index != region.index
            || !original.work.same_invocation(&region.work)
            || region.index != 0
            || storage.regions.len() != 1
            || !matches!(
                storage.state,
                RecordingState::Waiting | RecordingState::Recording
            )
            || !matches!(
                storage.regions[0],
                RecordingState::Waiting | RecordingState::Recording
            )
            || storage
                .model_context
                .as_ref()
                .is_none_or(|original| original.kind != ModelContextColdKind::DormantRestore)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let allowance = target.stage_model_context_material_work_allowance(context, work, true)?;
        if !Arc::ptr_eq(&allowance, &storage.allowance) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub fn require_model_context_material_restore_counter_read(
        &self,
        target: &Self,
        context: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
        region: &SemanticColdModelWorkRegion,
    ) -> Result<(), SemanticTransitionError> {
        self.require_model_context_cold_pair(context, target)?;
        self.require_model_context_restore_source_callback_work(
            context, parent, work, region, true,
        )?;
        let original = target
            .model_context_material_work
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let storage = self.original_cold_model_work(work)?;
        if !Arc::ptr_eq(&target.publication_issuer, &context.inner.target_issuer)
            || !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
            || !original.work.same_invocation(work)
            || original.source_parent != parent.identity
            || !original.restoring
            || (storage.regions[0] == RecordingState::Closed && !original.admitted)
            || original.callback_region.as_ref().is_none_or(|actual| {
                actual.index != region.index || !actual.work.same_invocation(&region.work)
            })
            || !Arc::ptr_eq(&original.native_work.allowance, &storage.allowance)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    fn require_model_context_restore_source_callback_work(
        &self,
        context: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
        region: &SemanticColdModelWorkRegion,
        counter_read: bool,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_original_consumer_step(parent)?;
        let storage = self.original_cold_model_work(work)?;
        if !Arc::ptr_eq(&self.publication_issuer, &context.inner.source_issuer)
            || !Arc::ptr_eq(&parent.issuer, &self.publication_issuer)
            || !parent.active
            || work.token != parent.token
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !Arc::ptr_eq(owner, &context.inner))
            || storage.model_context.as_ref().is_none_or(|original| {
                original.kind != ModelContextColdKind::DormantRestore
                    || !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
            })
            || storage
                .prepared_scope
                .as_ref()
                .is_none_or(|scope| !Arc::ptr_eq(scope, &context.inner.target_issuer))
            || storage.purpose != SemanticColdModelWorkPurpose::PrivateRestore
            || region.index != 0
            || !region.work.same_invocation(work)
            || storage.regions.len() != 1
            || (if counter_read {
                if storage.regions[0] == RecordingState::Closed {
                    !matches!(
                        storage.state,
                        RecordingState::Recording | RecordingState::Closed
                    )
                } else {
                    !matches!(
                        storage.state,
                        RecordingState::Waiting | RecordingState::Recording
                    ) || !matches!(
                        storage.regions[0],
                        RecordingState::Waiting | RecordingState::Recording
                    )
                }
            } else {
                !matches!(
                    storage.state,
                    RecordingState::Waiting | RecordingState::Recording
                ) || !matches!(
                    storage.regions[0],
                    RecordingState::Waiting | RecordingState::Recording
                )
            })
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub(super) fn model_context_material_completion_pending(
        &self,
        context: &SemanticModelContextContinuation,
    ) -> Result<bool, SemanticTransitionError> {
        let Some(original) = &self.model_context_material_work else {
            return Ok(false);
        };
        if !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
            || !Arc::ptr_eq(&self.publication_issuer, &context.inner.target_issuer)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(original.pending)
    }

    fn original_cold_completion_pending(
        &self,
        work: &SemanticColdModelWork,
    ) -> Result<bool, SemanticTransitionError> {
        let storage = self.original_cold_model_work(work)?;
        if (!storage.initialization_write_completed
            && storage
                .initialization_write
                .as_ref()
                .is_some_and(|write| write.completion_pending()))
            || storage
                .initialization_reset
                .as_ref()
                .is_some_and(OriginalNativeCommand::completion_pending)
        {
            return Ok(true);
        }
        let Some(completion) = &storage.completion else {
            return Ok(false);
        };
        let completion = completion
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        Ok((!completion.metadata_completed
            && completion
                .metadata
                .as_ref()
                .is_some_and(|write| write.completion_pending()))
            || completion.command.completion_pending()
            || completion.read.completion_pending())
    }

    pub(super) fn require_dormant_model_context_checkpoint_work(
        &self,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
    ) -> Result<Arc<Mutex<native_work_bound::ColdNativeAllowance>>, SemanticTransitionError> {
        Ok(Arc::clone(
            &self
                .require_model_context_material_work(context, work, false)?
                .native_work
                .allowance,
        ))
    }

    pub(super) fn require_dormant_model_context_restore_work(
        &self,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
    ) -> Result<Arc<Mutex<native_work_bound::ColdNativeAllowance>>, SemanticTransitionError> {
        Ok(Arc::clone(
            &self
                .require_model_context_material_work(context, work, true)?
                .native_work
                .allowance,
        ))
    }

    pub(super) fn dormant_model_context_restore_may_submit(
        &self,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
        allowance: &Arc<Mutex<native_work_bound::ColdNativeAllowance>>,
    ) -> bool {
        self.require_model_context_material_work(context, work, true)
            .is_ok_and(|original| Arc::ptr_eq(&original.native_work.allowance, allowance))
    }

    pub(super) fn finish_model_context_material_work(
        &mut self,
        target: &mut Self,
        context: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.check_cold_work_completion_reader(
            ColdWorkReader::ModelContextMaterial(context, target, parent),
            work,
        )?;
        if let Some(result) = target
            .model_context_material_work
            .as_ref()
            .and_then(|original| original.result)
        {
            return Ok(result);
        }
        if target.graph.borrowed_cold_work().is_some() {
            let borrowed = target
                .model_context_material_work
                .as_ref()
                .expect("original material report");
            target
                .graph
                .complete_cold_work(&borrowed.native_work.work)
                .map_err(SemanticTransitionError::Semantic)?;
        }
        if self.original_cold_model_work(work)?.state == RecordingState::Recording {
            self.close_cold_model_work(work)?;
        }
        let outcome = self.finish_cold_model_work_for_reader(
            ColdWorkReader::ModelContextMaterial(context, target, parent),
            work,
            &[],
            SemanticColdModelWorkDisposition::Complete,
        );
        target
            .model_context_material_work
            .as_mut()
            .expect("original material report")
            .pending = self.original_cold_completion_pending(work)?;
        let result = outcome?;
        target
            .model_context_material_work
            .as_mut()
            .expect("original material report")
            .result = Some(result);
        Ok(result)
    }

    pub(super) fn attach_model_context_prefill_work(
        &mut self,
        context: &SemanticModelContextContinuation,
        source: &Self,
        parent: &SemanticPublishedLease,
        admission: &SemanticSegmentInstructionAdmission,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        context.inner.require_imported_target(self)?;
        source.require_original_admitted_cold_model_work(admission, parent, work)?;
        let original = source.cold_model_work(work)?;
        if self
            .model_context_continuation
            .as_ref()
            .is_none_or(|owner| !Arc::ptr_eq(owner, &context.inner))
            || source
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !Arc::ptr_eq(owner, &context.inner))
            || Arc::ptr_eq(&self.publication_issuer, &source.publication_issuer)
            || !Arc::ptr_eq(self.provider.memory(), source.provider.memory())
            || self.domain.stream_id() != source.domain.stream_id()
            || !Arc::ptr_eq(
                self.domain.execution_stream(),
                source.domain.execution_stream(),
            )
            || !Arc::ptr_eq(
                self.domain.execution_stream().context(),
                source.domain.execution_stream().context(),
            )
            || self.domain.execution_stream().context().cu_ctx()
                != source.domain.execution_stream().context().cu_ctx()
            || original.purpose != SemanticColdModelWorkPurpose::PrivatePrefix
            || !matches!(
                original.state,
                RecordingState::Waiting | RecordingState::Recording
            )
            || self
                .replay_material_budget
                .as_ref()
                .zip(source.replay_material_budget.as_ref())
                .is_none_or(|(target, source)| !target.same_pool(source))
        {
            return Err(publication_input_error(
                "imported prefill changed its original SOURCE preparation report or material pool",
            ));
        }
        let shared = source.share_cold_native_work(work)?;
        if self.graph.borrowed_cold_work().is_some() {
            self.graph
                .require_cold_work_owner(&shared.work, true)
                .map_err(SemanticTransitionError::Semantic)?;
            if self
                .graph
                .cold_work_allowance()
                .as_ref()
                .is_none_or(|allowance| !Arc::ptr_eq(allowance, &shared.allowance))
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            return Ok(());
        }
        // The opaque owner authenticates this exact context pair. Keep generic
        // provider-identity admission strict for every other cold child.
        self.graph
            .begin_borrowed_cold_work(shared.work, shared.custody, shared.allowance)
            .map_err(SemanticTransitionError::Semantic)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the original instruction reader and imported producer roster have distinct custody"
    )]
    pub(super) fn freeze_model_context_prefill_roster(
        &self,
        context: &SemanticModelContextContinuation,
        target: &Self,
        parent: &SemanticPublishedLease,
        admission: &SemanticSegmentInstructionAdmission,
        work: &SemanticColdModelWork,
        declarations: &[native_work_bound::ModelContextPrefillContentGeometry],
        prefill_token_count: usize,
    ) -> Result<(), SemanticTransitionError> {
        context.inner.require_imported_target(target)?;
        self.require_original_admitted_cold_model_work(admission, parent, work)?;
        let original = self.cold_model_work(work)?;
        if self
            .model_context_continuation
            .as_ref()
            .is_none_or(|owner| !Arc::ptr_eq(owner, &context.inner))
            || target
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !Arc::ptr_eq(owner, &context.inner))
            || original.purpose != SemanticColdModelWorkPurpose::PrivatePrefix
            || original.state != RecordingState::Waiting
            || original.plan.is_some()
            || original.initialization_write.is_some()
            || original.initialization_reset.is_some()
            || original.prepared_scope.is_some()
        {
            return Err(publication_input_error(
                "imported prefill geometry must precede the original SOURCE cold declaration and effects",
            ));
        }
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if target
            .replay_material_budget
            .as_ref()
            .is_none_or(|other| !budget.same_pool(other))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let roster = native_work_bound::FrozenModelContextPrefillRoster::new(
            context,
            budget,
            declarations,
            prefill_token_count,
        )?;
        let bytes = prefill_token_count
            .checked_mul(size_of::<u64>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .filter(|bytes| *bytes != 0)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let dma = [1, 0, 0, 0, 0, 0, 0, 0, bytes];
        let input_guard = [
            1,
            u64::try_from(prefill_token_count)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        let first = self.admitted_segment_first_program_ordinal(admission)?;
        let budget = self
            .actor_refresh_program
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .steps
            .get(usize::try_from(first).map_err(|_| SemanticTransitionError::GenerationExhausted)?)
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .budget;
        original
            .allowance
            .lock()
            .map_err(|_| publication_input_error("imported prefill allowance is poisoned"))?
            .freeze_model_context_prefill(roster, dma, input_guard, budget)
    }

    pub(super) fn prepare_model_context_prefill_content_binding(
        &self,
        context: &SemanticModelContextContinuation,
        initial: &SemanticInitialPrefillLease,
        occurrence: usize,
        kind: InitialPrefillCaptureKind,
        verify: bool,
    ) -> Result<ModelContextPrefillContentBinding, SemanticTransitionError> {
        if !Arc::ptr_eq(&initial.issuer, &self.publication_issuer) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let dormant = self.initial_prefill.is_none() && self.retained_initial_prefill.is_none();
        let restored = if dormant {
            Some(
                context
                    .inner
                    .restored_prefill_content_proof(self, &initial.scope)?,
            )
        } else {
            None
        };
        let activating = if dormant {
            false
        } else {
            let activating = context
                .inner
                .require_initial_content_scope(self, &initial.scope)?;
            let scope = self
                .initial_prefill
                .as_ref()
                .map(|stage| &stage.scope)
                .or_else(|| {
                    self.retained_initial_prefill
                        .as_ref()
                        .map(|stage| &stage.scope)
                })
                .ok_or(SemanticTransitionError::NotBound)?;
            if !Arc::ptr_eq(scope, &initial.scope) {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            activating
        };
        let allowance = self.graph.cold_work_allowance().ok_or_else(|| {
            publication_input_error("imported prefill lost its original shared SOURCE allowance")
        })?;
        let work = self.graph.active_cold_work().ok_or_else(|| {
            publication_input_error("imported prefill lost its original shared SOURCE tally")
        })?;
        self.graph
            .require_cold_work_owner(&work, !activating)
            .map_err(SemanticTransitionError::Semantic)?;
        if dormant {
            let original = self
                .model_context_material_work
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let actual =
                self.require_dormant_model_context_restore_work(context, &original.work)?;
            if !Arc::ptr_eq(&allowance, &actual) {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        }
        Ok(ModelContextPrefillContentBinding {
            context: context.clone(),
            scope: Arc::clone(&initial.scope),
            occurrence,
            kind,
            verify,
            activating,
            restored,
            allowance,
        })
    }

    pub(super) fn claim_model_context_prefill_input_guard(
        &self,
        context: &SemanticModelContextContinuation,
        initial: &SemanticInitialPrefillLease,
    ) -> Result<(), SemanticTransitionError> {
        context
            .inner
            .require_initial_prefill_preparation(self, &initial.scope)?;
        let stage = self
            .initial_prefill
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&initial.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&initial.scope, &stage.scope)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let work = self
            .graph
            .borrowed_cold_work()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        self.graph
            .require_cold_work_owner(&work, true)
            .map_err(SemanticTransitionError::Semantic)?;
        let allowance = self
            .graph
            .cold_work_allowance()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let mut allowance = allowance
            .lock()
            .map_err(|_| publication_input_error("context prefill allowance is poisoned"))?;
        allowance.claim_model_context_prefill_input_guard(
            context,
            &initial.scope,
            stage.layout.prefill_tokens.len(),
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the original SOURCE report and actual TARGET upload roster have distinct custody"
    )]
    pub(super) fn admit_model_context_publication_dma(
        &self,
        context: &SemanticModelContextContinuation,
        target: &Self,
        parent: &SemanticPublishedLease,
        admission: &SemanticSegmentInstructionAdmission,
        work: &SemanticColdModelWork,
        preparer: &Arc<state_restoration::OriginalPublicationPreparation>,
        memory: &[crate::device::RetainedDeviceMemoryCommand],
        uploads: &[state_restoration::PublicationUploadPlan],
        staging_bytes: usize,
        writes: &[state_restoration::OriginalDeviceWrite],
    ) -> Result<Arc<Mutex<native_work_bound::ColdNativeAllowance>>, SemanticTransitionError> {
        context.inner.require_parent_preparation(target)?;
        self.require_model_context_cold_pair(context, target)?;
        self.require_original_admitted_cold_model_work(admission, parent, work)?;
        preparer.require_issuer(target)?;
        if target
            .publication_preparation
            .as_ref()
            .is_none_or(|original| !Arc::ptr_eq(original, preparer))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let storage = self.original_cold_model_work(work)?;
        if storage.state != RecordingState::Recording || !storage.initialized {
            return Err(publication_input_error(
                "imported publication uploads require the original initialized SOURCE recording report",
            ));
        }
        let mut dma = [0; 9];
        for command in memory {
            native_work_bound::add_native_work_ceiling(
                &mut dma,
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    u64::try_from(command.destination().len())
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                ],
            )?;
        }
        for upload in uploads {
            let bytes = upload.source.len();
            if bytes == 0 || bytes != upload.destination.len() || staging_bytes == 0 {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let commands = bytes
                .checked_add(staging_bytes - 1)
                .ok_or(SemanticTransitionError::GenerationExhausted)?
                / staging_bytes;
            native_work_bound::add_native_work_ceiling(
                &mut dma,
                [
                    u64::try_from(commands)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    u64::try_from(bytes)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                ],
            )?;
        }
        for write in writes {
            native_work_bound::add_native_work_ceiling(
                &mut dma,
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    u64::try_from(write.dma_bytes())
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                ],
            )?;
        }
        let first = self.admitted_segment_first_program_ordinal(admission)?;
        let budget = self
            .actor_refresh_program
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .steps
            .get(usize::try_from(first).map_err(|_| SemanticTransitionError::GenerationExhausted)?)
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .budget;
        storage
            .allowance
            .lock()
            .map_err(|_| publication_input_error("context publication allowance is poisoned"))?
            .admit_model_context_publication_dma(
                context,
                preparer,
                dma,
                storage.plan()?.quantities(),
                budget,
            )?;
        Ok(Arc::clone(&storage.allowance))
    }

    /// Bind only writer metadata to an original unopened cold region.
    /// This does not admit a numerical plan or enter a model operation.
    pub fn require_cold_checkpoint_writer_preparation(
        &self,
        work: &SemanticColdModelWork,
        region_ordinal: usize,
    ) -> Result<(), SemanticTransitionError> {
        let original = self.original_cold_model_work(work)?;
        if region_ordinal >= original.regions.len().max(1)
            || !matches!(
                original.state,
                RecordingState::Waiting | RecordingState::Recording
            )
            || original.plan.is_some()
            || original
                .regions
                .iter()
                .any(|state| *state != RecordingState::Waiting)
            || original.pending_operation.is_some()
            || original.operations.iter().any(|operation| {
                matches!(
                    operation.state,
                    OperationState::Entered | OperationState::Completing | OperationState::Unknown
                )
            })
            || original.stopped_before_entry
            || original.completion.is_some()
            || original.result.is_some()
            || original.initialization_poisoned
            || self.poisoned
        {
            return Err(publication_input_error(
                "checkpoint writer preparation requires its same unopened cold report and region",
            ));
        }
        Ok(())
    }

    /// Admit only the native prelude of the original restore region. The
    /// numerical region stays waiting until its actual model callback begins.
    pub fn prepare_cold_resident_restoration(
        &mut self,
        handle: &SemanticColdModelWork,
        restoration: &SemanticPreparedResidentModelRestoration,
    ) -> Result<SemanticColdResidentRestorationPrelude, SemanticTransitionError> {
        let storage = self.original_cold_model_work(handle)?;
        if self.is_poisoned()
            || self.pending
            || self.pending_replay_delivery.is_some()
            || self.pending_training_materialization.is_some()
            || storage.purpose != SemanticColdModelWorkPurpose::IntermediateRestore
            || !storage.initialized
            || storage.state != RecordingState::Recording
            || storage.completion.is_some()
            || storage.stopped_before_entry
            || storage.pending_operation.is_some()
            || storage.regions.as_slice()
                != [
                    RecordingState::Closed,
                    RecordingState::Closed,
                    RecordingState::Waiting,
                    RecordingState::Waiting,
                ]
            || storage
                .operations
                .iter()
                .any(|attempt| attempt.state != OperationState::Complete)
        {
            return Err(publication_input_error(
                "native restoration prelude requires its original admitted report after known snapshot and retirement completion",
            ));
        }
        storage
            .plan()?
            .require_region_start(2)
            .map_err(publication_input_error)?;
        restoration.require_report(handle, 2)?;
        let reservation_bytes = size_of::<SemanticColdResidentRestorationPrelude>()
            .checked_add(2 * size_of::<usize>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = self
            .replay_material_budget
            .as_ref()
            .ok_or_else(|| {
                publication_input_error(
                    "native restoration prelude lost its original replay material pool",
                )
            })?
            .reserve_auxiliary(reservation_bytes)?;
        storage
            .allowance
            .lock()
            .map_err(|_| publication_input_error("original native restore allowance is poisoned"))?
            .claim_resident_restoration(restoration)?;
        Ok(SemanticColdResidentRestorationPrelude {
            original: handle.clone(),
            restoration: Arc::downgrade(&restoration.owner),
            native_work: SemanticColdNativeWork {
                provider: Arc::clone(&self.provider),
                domain: self.domain.clone(),
                work: storage.native_work.view(),
                custody: Arc::clone(&storage.aliases),
                allowance: Arc::clone(&storage.allowance),
            },
            _reservation: reservation,
        })
    }

    /// Check actual recording authority without acquiring, entering or
    /// resetting a report. The borrow remains live through a child activation.
    pub fn cold_model_work_activation_permit<'a>(
        &'a self,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<SemanticColdNativeWorkActivationPermit<'a>, SemanticTransitionError> {
        let storage = self.original_cold_model_work(handle)?;
        if storage
            .model_context
            .as_ref()
            .is_some_and(|context| context.kind == ModelContextColdKind::Activation)
            || !storage.initialized
            || storage.state != RecordingState::Recording
            || storage.plan.is_none()
            || storage.stopped_before_entry
            || storage.completion.is_some()
            || storage.operations.iter().any(|attempt| {
                matches!(
                    attempt.state,
                    OperationState::Unknown
                        | OperationState::Completing
                        | OperationState::NotEntered
                )
            })
        {
            return Err(publication_input_error(
                "cold activation requires its original admitted recording report without unresolved model effects",
            ));
        }
        match region {
            Some(region) => {
                if !region.work.same_invocation(handle)
                    || storage.regions.get(region.index) != Some(&RecordingState::Recording)
                    || storage.regions[..region.index].iter().any(|state| {
                        !matches!(state, RecordingState::Closed | RecordingState::NotEntered)
                    })
                {
                    return Err(publication_input_error(
                        "cold activation changed its original recording region or order",
                    ));
                }
            }
            None if !storage.regions.is_empty() => {
                return Err(publication_input_error(
                    "segmented cold activation requires its original recording region",
                ));
            }
            None => {}
        }
        Ok(SemanticColdNativeWorkActivationPermit {
            owner: self,
            storage,
            region: region.map(|region| region.index),
        })
    }

    /// Same-owner activation checks authority immediately before its mutable
    /// entry. The returned identity only matches original pending custody.
    pub fn require_cold_model_work_activation(
        &self,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
        parent: &SemanticPublishedLease,
    ) -> Result<SemanticColdNativeWorkActivationIdentity, SemanticTransitionError> {
        self.cold_model_work_activation_permit(handle, region)?
            .identity_for_callback(self, parent)
    }

    pub(super) fn cold_native_allowance(
        &self,
        token: u64,
    ) -> Result<Option<Arc<Mutex<native_work_bound::ColdNativeAllowance>>>, SemanticTransitionError>
    {
        let step = self.steps.get(&token).ok_or_else(|| {
            publication_input_error("cold content lost its original acquired reader")
        })?;
        let allowance = self.graph.cold_work_allowance();
        if let Some(storage) = &step.cold_model_work {
            if matches!(
                storage.state,
                RecordingState::Waiting | RecordingState::Recording | RecordingState::Closed
            ) && allowance
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, &storage.allowance))
            {
                return Err(publication_input_error(
                    "cold content changed its original tally allowance owner",
                ));
            }
        }
        Ok(allowance)
    }

    pub(super) fn cold_native_work(
        &self,
        token: u64,
    ) -> Result<Option<DeviceMemoryView<u64>>, SemanticTransitionError> {
        // Captured steps already own their execution tally. Keep a borrowed
        // cold tally out of capture, but retain it for final-use guards and
        // resource retirement after the actual completed handoff.
        if self
            .steps
            .get(&token)
            .is_some_and(|step| step.prepared.is_some())
            && self
                .prepared_segment
                .as_ref()
                .is_some_and(|build| build.capturing)
        {
            return Ok(None);
        }
        let Some(storage) = self
            .steps
            .get(&token)
            .and_then(|step| step.cold_model_work.as_ref())
        else {
            return Ok(self.graph.active_cold_work());
        };
        match storage.state {
            RecordingState::Waiting | RecordingState::Recording | RecordingState::Closed => {
                Ok(Some(storage.native_work.view()))
            }
            // A completed local report does not hide the next original
            // operation's borrowed tally. Late verification and retirement of
            // this reader must charge their actual commands to that live owner.
            RecordingState::Completed => Ok(self.graph.active_cold_work()),
            _ => Err(publication_input_error(
                "cold native producers cannot run after failure or report submission",
            )),
        }
    }

    /// Mint only from the retained original invocation, before child admission.
    /// The guard counts as a live alias until that actual child stream joins.
    pub fn share_cold_native_work(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<SemanticColdNativeWork, SemanticTransitionError> {
        let storage = self.cold_model_work(handle)?;
        if !matches!(
            storage.state,
            RecordingState::Waiting | RecordingState::Recording
        ) {
            return Err(publication_input_error(
                "cold child construction requires its original open operation",
            ));
        }
        Ok(SemanticColdNativeWork {
            provider: Arc::clone(&self.provider),
            domain: self.domain.clone(),
            work: storage.native_work.view(),
            custody: Arc::clone(&storage.aliases),
            allowance: Arc::clone(&storage.allowance),
        })
    }

    /// Completion removes only a borrowed tally, after the actual reader's
    /// entire consumer roster joins. Unknown completion keeps the same guard.
    pub fn complete_shared_cold_native_work(
        &mut self,
        lease: &SemanticPublishedLease,
        streams: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        self.checked_original_consumer_step(lease)?;
        if !lease.active {
            return Err(publication_input_error(
                "shared cold completion requires its original acquired reader",
            ));
        }
        let work = self.graph.borrowed_cold_work().ok_or_else(|| {
            publication_input_error("child completion lost its original borrowed tally")
        })?;
        self.complete_step_consumers(lease, streams)?;
        self.graph
            .complete_cold_work(&work)
            .map_err(SemanticTransitionError::Semantic)
    }

    /// Attach the original phase's open tally to an already restored reader,
    /// before its next actual cold preparation. No new counter is allocated.
    pub fn attach_shared_cold_native_work(
        &mut self,
        lease: &SemanticPublishedLease,
        work: SemanticColdNativeWork,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_reader(lease)?;
        self.attach_cold_native_work(work)
    }

    pub fn attach_shared_admitted_cold_native_work(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        work: SemanticColdNativeWork,
    ) -> Result<(), SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        if !matches!(
            self.instruction_admission(admission)?.state,
            actor_refresh_program::InstructionState::Released
                | actor_refresh_program::InstructionState::BuildEntered
                | actor_refresh_program::InstructionState::Bound
        ) {
            return Err(publication_input_error(
                "shared prepared work requires its original handed-off instruction parent",
            ));
        }
        self.attach_cold_native_work(work)
    }

    pub(super) fn attach_cold_native_work(
        &mut self,
        work: SemanticColdNativeWork,
    ) -> Result<(), SemanticTransitionError> {
        let (view, custody, allowance) = work.graph_custody(&self.provider, &self.domain)?;
        self.graph
            .begin_borrowed_cold_work(view, custody, allowance)
            .map_err(SemanticTransitionError::Semantic)
    }

    /// Remove visibility of this exact borrowed tally before its issuer joins
    /// and reads the report. Detachment is not completion or resource release;
    /// the original work allocation and unfinished report remain retained.
    pub fn detach_shared_cold_native_work(
        &mut self,
        work: &SemanticColdNativeWork,
    ) -> Result<(), SemanticTransitionError> {
        if !Arc::ptr_eq(&self.provider, &work.provider)
            || self.domain.stream_id() != work.domain.stream_id()
        {
            return Err(publication_input_error(
                "cold detachment changed its original allocation owner or stream",
            ));
        }
        self.graph
            .complete_cold_work(&work.work)
            .map_err(SemanticTransitionError::Semantic)
    }

    /// Keep the same open operation alive at its already attached surviving
    /// reader before the private issuer is completely retired. Move the actual
    /// storage and region states: no allocation, copy, reset or new invocation.
    #[cfg(feature = "semantic-policy")]
    pub fn handoff_cold_model_work(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        survivor: &mut SemanticTransitionSession,
        survivor_lease: &SemanticPublishedLease,
        streams: &[u64],
    ) -> Result<(SemanticColdModelWork, Vec<SemanticColdModelWorkRegion>), SemanticTransitionError>
    {
        self.checked_reader(lease)?;
        survivor.checked_reader(survivor_lease)?;
        survivor.require_completed_cold_model_work()?;
        let original = self.cold_model_work(handle)?;
        if lease.token != handle.token
            || !Arc::ptr_eq(&self.provider, &survivor.provider)
            || self.domain.stream_id() != survivor.domain.stream_id()
            || Arc::ptr_eq(&self.publication_issuer, &survivor.publication_issuer)
            || original.state != RecordingState::Recording
            || original.regions.is_empty()
            || original
                .regions
                .iter()
                .any(|state| !matches!(state, RecordingState::Waiting | RecordingState::Closed))
            || survivor
                .checked_step(survivor_lease)?
                .cold_model_work
                .as_ref()
                .is_some_and(|prior| {
                    !Arc::ptr_eq(&prior.admission, &original.admission)
                        || prior.operation_ordinal > original.operation_ordinal
                        || prior.capacity_limit != original.capacity_limit
                })
        {
            return Err(publication_input_error(
                "cold ownership handoff requires the same known open operation and attached survivor",
            ));
        }
        // All actual consumers join before changing graph visibility. A failed
        // join leaves the same issuer/report retained, never a second transfer.
        self.quiesce_published_reader(lease, streams)?;
        survivor.quiesce_published_reader(survivor_lease, streams)?;
        let original = self.cold_model_work(handle)?;
        let view = original.native_work.view();
        let custody = Arc::clone(&original.aliases);
        let allowance = Arc::clone(&original.allowance);
        let regions = original.regions.len();
        self.graph
            .require_cold_work_owner(&view, false)
            .map_err(SemanticTransitionError::Semantic)?;
        survivor
            .graph
            .require_cold_work_owner(&view, true)
            .map_err(SemanticTransitionError::Semantic)?;
        // Validation and joins above precede these ownership-only changes.
        // The old storage remains live until the graph ownership is transferred.
        self.graph
            .complete_cold_work(&view)
            .map_err(SemanticTransitionError::Semantic)?;
        survivor
            .graph
            .complete_cold_work(&view)
            .map_err(SemanticTransitionError::Semantic)?;
        survivor
            .graph
            .begin_cold_work(view.clone(), Arc::clone(&allowance))
            .map_err(SemanticTransitionError::Semantic)?;
        self.graph
            .begin_borrowed_cold_work(view, custody, allowance)
            .map_err(SemanticTransitionError::Semantic)?;
        let storage = self
            .steps
            .get_mut(&lease.token)
            .expect("checked original step owner")
            .cold_model_work
            .take()
            .expect("checked original cold report");
        let transferred = SemanticColdModelWork {
            issuer: Arc::clone(&survivor.publication_issuer),
            invocation: Arc::clone(&storage.invocation),
            token: survivor_lease.token,
            operation_ordinal: storage.operation_ordinal,
            admission: Arc::clone(&storage.admission),
            final_cursor: Arc::clone(&storage.final_cursor),
        };
        survivor
            .steps
            .get_mut(&survivor_lease.token)
            .expect("checked surviving reader")
            .cold_model_work = Some(storage);
        Ok((
            transferred.clone(),
            (0..regions)
                .map(|index| SemanticColdModelWorkRegion {
                    work: transferred.clone(),
                    index,
                })
                .collect(),
        ))
    }

    /// The actual model recorder must have closed before a prepared graph can
    /// take over numerical execution. Waiting or failed registration is not a
    /// CPU-only or zero-work observation.
    pub fn require_closed_cold_model_work(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        if self.cold_model_work(handle)?.state != RecordingState::Closed {
            return Err(publication_input_error(
                "prepared execution requires its original closed cold preparation recorder",
            ));
        }
        Ok(())
    }

    pub(super) fn require_completed_cold_model_work(&self) -> Result<(), SemanticTransitionError> {
        if self.steps.values().any(|step| {
            step.cold_model_work.as_ref().is_some_and(|storage| {
                storage.state != RecordingState::Completed
                    || Arc::strong_count(&storage.aliases) != 1
            })
        }) {
            return Err(publication_input_error(
                "state-changing work or release requires completed cold model work without scratch aliases",
            ));
        }
        Ok(())
    }

    /// Called only after the original signed admission and physical interval
    /// have begun. Retention precedes every reset or external model callback.
    pub fn prepare_cold_model_work(
        &mut self,
        lease: &SemanticPublishedLease,
        capacity: usize,
        operation_ordinal: u64,
        admission: impl Into<SemanticColdModelWorkAdmission>,
        purpose: SemanticColdModelWorkPurpose,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.prepare_cold_model_work_inner(
            lease,
            Some(capacity),
            operation_ordinal,
            admission.into(),
            purpose,
            None,
        )
    }

    pub fn prepare_cold_model_work_deferred(
        &mut self,
        lease: &SemanticPublishedLease,
        operation_ordinal: u64,
        admission: impl Into<SemanticColdModelWorkAdmission>,
        purpose: SemanticColdModelWorkPurpose,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.prepare_cold_model_work_inner(
            lease,
            None,
            operation_ordinal,
            admission.into(),
            purpose,
            None,
        )
    }

    pub fn prepare_admitted_cold_model_work(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        let original = self.instruction_admission(admission)?;
        if original.state != actor_refresh_program::InstructionState::Admitted {
            return Err(publication_input_error(
                "instruction preparation must retain its single original cold attempt",
            ));
        }
        if let Some(handle) = original.cold_work.clone() {
            self.continue_cold_model_work_initialization(&handle)?;
            return Ok(handle);
        }
        let instruction = Arc::clone(&original.instruction);
        let ordinal = self.admitted_segment_first_program_ordinal(admission)?;
        self.prepare_cold_model_work_inner(
            parent,
            None,
            ordinal,
            instruction.into(),
            SemanticColdModelWorkPurpose::PrivatePrefix,
            Some((admission, InstructionColdRole::Preparation)),
        )
    }

    pub fn require_admitted_cold_model_work(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.require_original_admitted_cold_model_work(admission, parent, work)?;
        self.ensure_quiescent()
    }

    /// Authenticate only the retained original callback, including while its
    /// own initializer awaits completion. This grants no recording authority.
    pub fn require_original_admitted_cold_model_work(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        let original = self.instruction_admission(admission)?;
        if original.state != actor_refresh_program::InstructionState::Admitted
            || original
                .cold_work
                .as_ref()
                .is_none_or(|retained| !retained.same_invocation(work))
        {
            return Err(publication_input_error(
                "cold callback does not own this original instruction invocation",
            ));
        }
        self.check_cold_work_completion_reader(ColdWorkReader::Published(parent), work)?;
        Ok(())
    }

    pub fn bind_shared_admitted_cold_model_work(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        source: &Self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        let storage = source.cold_model_work(handle)?;
        let allowance = self.graph.cold_work_allowance().ok_or_else(|| {
            publication_input_error("instruction preparation lost its original shared tally")
        })?;
        let original = self.instruction_admission_mut(admission)?;
        if original.state != actor_refresh_program::InstructionState::Admitted
            || !Arc::ptr_eq(&allowance, &storage.allowance)
            || !matches!(
                storage.state,
                RecordingState::Waiting | RecordingState::Recording
            )
            || original
                .cold_work
                .as_ref()
                .is_some_and(|work| !work.same_invocation(handle))
        {
            return Err(publication_input_error(
                "shared preparation changed its original native invocation or allowance",
            ));
        }
        original.cold_work = Some(handle.clone());
        Ok(())
    }

    /// Borrow the one original instruction preparation report while issuing
    /// structural Update ports. No report is closed, replaced or fabricated.
    pub fn require_admitted_structural_cold_model_work(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        step: &SemanticPreparedStep,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.require_admitted_cold_model_work(admission, parent, work)?;
        self.require_admitted_structural_cold_model_work_owner(
            admission,
            parent,
            step,
            self.original_cold_model_work(work)?,
        )
    }

    pub fn require_original_admitted_structural_cold_model_work(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        step: &SemanticPreparedStep,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.require_original_admitted_cold_model_work(admission, parent, work)?;
        self.require_admitted_structural_cold_model_work_owner(
            admission,
            parent,
            step,
            self.original_cold_model_work(work)?,
        )
    }

    pub fn require_original_prepared_cold_model_work(
        &self,
        step: &SemanticPreparedStep,
        parent: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.check_cold_work_completion_reader(ColdWorkReader::Prepared(step, parent), work)?;
        Ok(())
    }

    fn require_admitted_structural_cold_model_work_owner(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        step: &SemanticPreparedStep,
        original: &ColdModelWorkStorage,
    ) -> Result<(), SemanticTransitionError> {
        self.require_original_prepared_structure()?;
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        build.check_retained(step, &self.publication_issuer)?;
        if build
            .program_admission
            .as_ref()
            .is_none_or(|original| !original.same_handle(admission))
            || build.requested_kind(step, &self.publication_issuer)?
                != SemanticTransitionKind::Update
            || !parent.active
        {
            return Err(publication_input_error(
                "structural cold callback requires its exact admitted Update and original active source",
            ));
        }
        if original.purpose != SemanticColdModelWorkPurpose::PrivatePrefix
            || !matches!(
                original.state,
                RecordingState::Waiting | RecordingState::Recording
            )
            || original.result.is_some()
            || original.prepared_scope.is_some()
        {
            return Err(publication_input_error(
                "structural callback must borrow its same still-open original prefix recorder",
            ));
        }
        Ok(())
    }

    /// Enter numerical initialization only after the same complete structural
    /// roster and original Prefix operation plan are frozen.
    pub fn require_admitted_prepared_cold_initialization(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        step: &SemanticPreparedStep,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.require_admitted_structural_cold_model_work(admission, parent, step, work)?;
        self.require_frozen_prepared_cold_initialization(self.original_cold_model_work(work)?)
    }

    /// Authenticate initialization against the already bound original shared
    /// Prefix owner; the target does not create or replace its report.
    pub fn require_shared_admitted_prepared_cold_initialization(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        step: &SemanticPreparedStep,
        source: &Self,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        self.ensure_quiescent()?;
        self.checked_original_consumer_step(parent)?;
        let original = self.instruction_admission(admission)?;
        let storage = source.cold_model_work(work)?;
        let allowance = self.graph.cold_work_allowance().ok_or_else(|| {
            publication_input_error("cold initialization lost its original shared tally")
        })?;
        if original.state != actor_refresh_program::InstructionState::Admitted
            || original
                .cold_work
                .as_ref()
                .is_none_or(|retained| !retained.same_invocation(work))
            || !Arc::ptr_eq(&allowance, &storage.allowance)
        {
            return Err(publication_input_error(
                "shared cold initialization changed its original instruction invocation or allowance",
            ));
        }
        self.require_admitted_structural_cold_model_work_owner(admission, parent, step, storage)?;
        self.require_frozen_prepared_cold_initialization(storage)
    }

    fn require_frozen_prepared_cold_initialization(
        &self,
        original: &ColdModelWorkStorage,
    ) -> Result<(), SemanticTransitionError> {
        let build = self
            .prepared_segment
            .as_ref()
            .expect("original structural scope");
        if !build.structural_frozen || original.state != RecordingState::Recording {
            return Err(publication_input_error(
                "cold initialization requires its original active recorder and complete frozen structural roster",
            ));
        }
        original.plan()?;
        if original.pending_operation.is_some() || original.stopped_before_entry {
            return Err(publication_input_error(
                "cold initialization requires its original active prefix without an unresolved attempt",
            ));
        }
        Ok(())
    }

    pub fn adopt_shared_admitted_cold_model_work_result(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        source: &Self,
        handle: &SemanticColdModelWork,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        let storage = source.cold_model_work(handle)?;
        let result = storage.result.ok_or_else(|| {
            publication_input_error("retain the original unfinished shared preparation report")
        })?;
        let original = self.instruction_admission_mut(admission)?;
        if original.state != actor_refresh_program::InstructionState::Released
            || storage.state != RecordingState::Completed
            || original
                .cold_work
                .as_ref()
                .is_none_or(|work| !work.same_invocation(handle))
        {
            return Err(publication_input_error(
                "shared preparation report changed its original handed-off invocation",
            ));
        }
        original.cold_result = Some(result);
        Ok(result)
    }

    pub fn finish_admitted_cold_model_work(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
        streams: &[u64],
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        let result = self.finish_cold_model_work_for_reader(
            ColdWorkReader::Admitted(admission, parent),
            work,
            streams,
            SemanticColdModelWorkDisposition::Complete,
        )?;
        self.instruction_admission_mut(admission)?.cold_result = Some(result);
        Ok(result)
    }

    /// Close only the known actual prefix of the original cancelled build.
    /// Its live parent remains admitted until genuine source retirement.
    pub fn finish_cancelled_admitted_cold_model_work(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        proof: &SemanticPreparedSegmentNonSubmission,
        work: &SemanticColdModelWork,
        streams: &[u64],
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        let reader = ColdWorkReader::AdmittedCancellation(admission, parent, proof);
        self.check_cold_work_completion_reader(reader, work)?;
        let storage = self.original_cold_model_work(work)?;
        match storage.state {
            RecordingState::Recording => {
                let regions = storage.regions.len();
                self.cancel_unentered_cold_model_work_regions(work, regions)?;
                self.close_cold_model_work(work)?;
            }
            RecordingState::Waiting => {
                // Metadata preparation may complete before any recorder entry.
                // Do not invent a begin marker or an executed model span.
                self.ensure_quiescent()?;
                let storage = self.original_cold_model_work(work)?;
                if storage.pending_operation.is_some()
                    || !storage.recorded_events().is_empty()
                    || storage.regions.iter().any(|state| {
                        !matches!(state, RecordingState::Waiting | RecordingState::NotEntered)
                    })
                {
                    return Err(publication_input_error(
                        "cancelled preparation cannot close entered or unknown original work",
                    ));
                }
                let storage = self
                    .steps
                    .get_mut(&work.token)
                    .expect("checked original reader")
                    .cold_model_work
                    .as_mut()
                    .expect("checked original work");
                for (index, state) in storage.regions.iter_mut().enumerate() {
                    if *state == RecordingState::Waiting {
                        if let Some(plan) = &mut storage.plan {
                            plan.skip_unentered_region(index)
                                .map_err(publication_input_error)?;
                        }
                        storage
                            .allowance
                            .lock()
                            .map_err(|_| {
                                publication_input_error(
                                    "original resident read allowance is poisoned",
                                )
                            })?
                            .skip_unentered_resident_reads(index)?;
                        *state = RecordingState::NotEntered;
                    }
                }
                if let Some(work) = &mut storage.work {
                    work.uncaptured_recording = false;
                }
                storage.state = RecordingState::Closed;
            }
            RecordingState::Closed
            | RecordingState::Submitting
            | RecordingState::Submitted
            | RecordingState::Completed => (),
            _ => {
                return Err(publication_input_error(
                    "cancelled preparation retains its failed or unresolved original recorder",
                ));
            }
        }
        let result = self.finish_cold_model_work_for_reader(
            reader,
            work,
            streams,
            SemanticColdModelWorkDisposition::KnownRefusal,
        )?;
        self.instruction_admission_mut(admission)?.cold_result = Some(result);
        Ok(result)
    }

    pub fn admitted_segment_cold_result(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<Option<SemanticColdModelWorkResult>, SemanticTransitionError> {
        let original = self.instruction_admission(admission)?;
        let Some(mut total) = original.cold_result else {
            return Ok(None);
        };
        for (_, report) in &original.prepared_cold_results {
            let Some(report) = report else {
                return Ok(None);
            };
            total.include(*report)?;
        }
        Ok(Some(total))
    }

    pub fn prepare_admitted_retirement_cold_model_work(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.prepare_admitted_retirement_cold_model_work_inner(admission, None)
    }

    pub fn prepare_cancelled_admitted_retirement_cold_model_work(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        proof: &SemanticPreparedSegmentNonSubmission,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.prepare_admitted_retirement_cold_model_work_inner(admission, Some(proof))
    }

    fn prepare_admitted_retirement_cold_model_work_inner(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        cancellation: Option<&SemanticPreparedSegmentNonSubmission>,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        let original = self.instruction_admission(admission)?;
        if let Some(handle) = original.retirement_work.clone() {
            self.require_admitted_retirement_disposition(admission, cancellation)?;
            self.continue_cold_model_work_initialization(&handle)?;
            return Ok(handle);
        }
        self.ensure_quiescent()?;
        self.require_completed_cold_model_work()?;
        let original = self.instruction_admission(admission)?;
        let instruction = Arc::clone(&original.instruction);
        let build = self.prepared_segment.as_ref().ok_or_else(|| {
            publication_input_error("instruction retirement lost its original prepared owner")
        })?;
        if build
            .program_admission
            .as_ref()
            .is_none_or(|original| !original.same_handle(admission))
        {
            return Err(publication_input_error(
                "cold retirement requires the same original admitted segment",
            ));
        }
        if let Some(proof) = cancellation {
            build.require_non_submission(proof)?;
        } else if !build.completed {
            return Err(publication_input_error(
                "ordinary cold retirement requires actual segment completion",
            ));
        }
        let token = *build.tokens.first().ok_or_else(|| {
            publication_input_error("instruction retirement lost its original storage token")
        })?;
        let scope = Arc::clone(&build.scope);
        let ordinal = self.admitted_segment_first_program_ordinal(admission)?;
        // Freeze the native disposition before installing or initializing the
        // original report. An uncertain initialization cannot change its mode.
        self.instruction_admission_mut(admission)?
            .retirement_cancellation = cancellation.cloned();
        self.install_cold_model_work(
            token,
            None,
            ordinal,
            instruction.into(),
            SemanticColdModelWorkPurpose::PrivateRetirement,
            Some(scope),
            Some(ColdWorkAdmission::Instruction(
                admission,
                InstructionColdRole::Retirement,
            )),
        )
    }

    pub fn require_admitted_retirement_cold_model_work(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.check_cold_work_reader(ColdWorkReader::AdmittedRetirement(admission), work)?;
        Ok(())
    }

    pub fn require_original_admitted_retirement_cold_model_work(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.check_cold_work_completion_reader(
            ColdWorkReader::AdmittedRetirement(admission),
            work,
        )?;
        Ok(())
    }

    pub fn bind_shared_admitted_retirement_cold_model_work(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        source: &Self,
        handle: &SemanticColdModelWork,
        cancellation: Option<&SemanticPreparedSegmentNonSubmission>,
    ) -> Result<(), SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        let build = self.prepared_segment.as_ref().ok_or_else(|| {
            publication_input_error("shared retirement lost its original prepared owner")
        })?;
        if build
            .program_admission
            .as_ref()
            .is_none_or(|original| !original.same_handle(admission))
        {
            return Err(publication_input_error(
                "shared retirement changed its original admitted segment",
            ));
        }
        if let Some(proof) = cancellation {
            build.require_non_submission(proof)?;
        } else if !build.completed {
            return Err(publication_input_error(
                "shared retirement requires actual original segment completion",
            ));
        }
        let storage = source.cold_model_work(handle)?;
        let allowance = self.graph.cold_work_allowance().ok_or_else(|| {
            publication_input_error("instruction retirement lost its original shared tally")
        })?;
        let original = self.instruction_admission_mut(admission)?;
        let same_cancellation = match (&original.retirement_cancellation, cancellation) {
            (None, None) => true,
            (Some(stored), Some(proof)) => stored.matches(&proof.steps),
            _ => false,
        };
        let original_state = matches!(
            original.state,
            actor_refresh_program::InstructionState::Bound
                | actor_refresh_program::InstructionState::Completed
        ) || (cancellation.is_some()
            && matches!(
                original.state,
                actor_refresh_program::InstructionState::Admitted
                    | actor_refresh_program::InstructionState::Released
                    | actor_refresh_program::InstructionState::BuildEntered
            ));
        if !original_state
            || storage.purpose != SemanticColdModelWorkPurpose::PrivateRetirement
            || !Arc::ptr_eq(&allowance, &storage.allowance)
            || !matches!(
                storage.state,
                RecordingState::Waiting | RecordingState::Recording
            )
            || original
                .retirement_work
                .as_ref()
                .is_some_and(|work| !work.same_invocation(handle) || !same_cancellation)
        {
            return Err(publication_input_error(
                "shared retirement changed its original invocation, allowance or disposition",
            ));
        }
        original.retirement_work = Some(handle.clone());
        original.retirement_cancellation = cancellation.cloned();
        Ok(())
    }

    pub fn adopt_shared_admitted_retirement_cold_model_work_result(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        source: &Self,
        handle: &SemanticColdModelWork,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        let original = self.instruction_admission(admission)?;
        if original
            .retirement_work
            .as_ref()
            .is_none_or(|work| !work.same_invocation(handle))
            || !Arc::ptr_eq(&handle.issuer, &source.publication_issuer)
        {
            return Err(publication_input_error(
                "shared retirement report changed its original invocation",
            ));
        }
        if let Some(result) = original.retirement_result {
            return Ok(result);
        }
        // Only the canonical resource-retirement path removes the prepared
        // owner, after graph destruction and every joined step/alias release.
        // Admission custody also proves full retirement when its parent is gone.
        if self.prepared_segment.is_some()
            || self.captured.is_some()
            || self.steps.contains_key(&parent.token)
            || !matches!(
                (original.state, original.retirement_cancellation.is_some()),
                (actor_refresh_program::InstructionState::Completed, false)
                    | (actor_refresh_program::InstructionState::Bound, true)
                    | (actor_refresh_program::InstructionState::Admitted, true)
                    | (actor_refresh_program::InstructionState::Released, true)
                    | (actor_refresh_program::InstructionState::BuildEntered, true)
            )
        {
            return Err(publication_input_error(
                "shared retirement report requires fully retired original native storage",
            ));
        }
        let storage = source.cold_model_work(handle)?;
        let result = storage.result.ok_or_else(|| {
            publication_input_error("retain the original unfinished shared retirement report")
        })?;
        if storage.purpose != SemanticColdModelWorkPurpose::PrivateRetirement
            || storage.state != RecordingState::Completed
            || storage.disposition != Some(SemanticColdModelWorkDisposition::Complete)
        {
            return Err(publication_input_error(
                "shared retirement requires its original completed report",
            ));
        }
        self.instruction_admission_mut(admission)?.retirement_result = Some(result);
        Ok(result)
    }

    pub fn admitted_retirement_cold_model_work_buffer(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        work: &SemanticColdModelWork,
        stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.cold_model_work_buffer_for_reader(
            ColdWorkReader::AdmittedRetirement(admission),
            work,
            stream,
            None,
        )
    }

    pub fn finish_admitted_retirement_cold_model_work(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        work: &SemanticColdModelWork,
        streams: &[u64],
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.finish_admitted_retirement_cold_model_work_inner(admission, None, work, streams)
    }

    pub fn finish_cancelled_admitted_retirement_cold_model_work(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        proof: &SemanticPreparedSegmentNonSubmission,
        work: &SemanticColdModelWork,
        streams: &[u64],
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.finish_admitted_retirement_cold_model_work_inner(admission, Some(proof), work, streams)
    }

    fn require_admitted_retirement_disposition(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        cancellation: Option<&SemanticPreparedSegmentNonSubmission>,
    ) -> Result<SemanticColdModelWorkDisposition, SemanticTransitionError> {
        let original = self.instruction_admission(admission)?;
        match (&original.retirement_cancellation, cancellation) {
            (None, None) => Ok(SemanticColdModelWorkDisposition::Complete),
            (Some(stored), Some(proof)) if stored.matches(&proof.steps) => {
                self.prepared_segment
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?
                    .require_non_submission(proof)?;
                Ok(SemanticColdModelWorkDisposition::KnownRefusal)
            }
            _ => Err(publication_input_error(
                "cold retirement changed its original native disposition or cancellation proof",
            )),
        }
    }

    fn finish_admitted_retirement_cold_model_work_inner(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        cancellation: Option<&SemanticPreparedSegmentNonSubmission>,
        work: &SemanticColdModelWork,
        streams: &[u64],
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        let disposition = self.require_admitted_retirement_disposition(admission, cancellation)?;
        let result = self.finish_cold_model_work_for_reader(
            ColdWorkReader::AdmittedRetirement(admission),
            work,
            streams,
            disposition,
        )?;
        self.instruction_admission_mut(admission)?.retirement_result = Some(result);
        Ok(result)
    }

    fn prepare_cold_model_work_inner(
        &mut self,
        lease: &SemanticPublishedLease,
        capacity_limit: Option<usize>,
        operation_ordinal: u64,
        admission: SemanticColdModelWorkAdmission,
        purpose: SemanticColdModelWorkPurpose,
        instruction: Option<(&SemanticSegmentInstructionAdmission, InstructionColdRole)>,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.require_completed_cold_model_work()?;
        self.checked_reader(lease)?;
        let step = self.checked_step(lease)?;
        // A later actual callback is a new invocation, never a reopened
        // recording. The previous report is complete and its scratch aliases
        // are gone before replacement; the enclosing operation retains its
        // returned immutable result and physical interval independently.
        if step.cold_model_work.as_ref().is_some_and(|previous| {
            !previous.admission.same_material(&admission)
                || previous.capacity_limit != capacity_limit
                || previous.operation_ordinal > operation_ordinal
        }) {
            return Err(publication_input_error(
                "a later cold callback cannot change its original admission, capacity or operation order",
            ));
        }
        if capacity_limit == Some(0)
            || admission.as_ref().is_empty()
            || self.prepared_segment.is_some()
            || step.prepared.is_some()
        {
            return Err(publication_input_error(
                "cold model work requires its sole admitted operation and positive original capacity",
            ));
        }
        self.retire_completed_evaluation_storage(lease)?;
        self.install_cold_model_work(
            lease.token,
            capacity_limit,
            operation_ordinal,
            admission,
            purpose,
            None,
            instruction.map(|(admission, role)| ColdWorkAdmission::Instruction(admission, role)),
        )
    }

    /// Retain the original prepared callback's report without acquiring a reader.
    pub fn prepare_model_context_cold_work(
        &mut self,
        context: &SemanticModelContextContinuation,
        step: &SemanticPreparedStep,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        context
            .inner
            .require_frozen_preparation(self, &step.scope)?;
        if let Some(previous) = self
            .steps
            .get(&step.token)
            .and_then(|owner| owner.cold_model_work.as_ref())
        {
            let original = previous
                .model_context
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if original.kind != ModelContextColdKind::Preparation
                || !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let ceiling = original.native_ceiling;
            return self.prepare_frozen_context_cold_model_work(context, step, ceiling);
        }
        let roster = native_work_bound::FrozenModelContextPreparation::new(self, step)?;
        let ceiling = roster.ceiling;
        let work = self.prepare_frozen_context_cold_model_work(context, step, ceiling)?;
        self.original_cold_model_work(&work)?
            .allowance
            .lock()
            .map_err(|_| publication_input_error("prepared context allowance is poisoned"))?
            .freeze_model_context_preparation(roster)?;
        Ok(work)
    }

    pub(super) fn claim_model_context_prepared_model_roster(
        &self,
        context: &SemanticModelContextContinuation,
        step: &SemanticPreparedStep,
        work: &SemanticColdModelWork,
        bank: usize,
        bytes: usize,
    ) -> Result<Arc<Mutex<native_work_bound::ColdNativeAllowance>>, SemanticTransitionError> {
        self.require_frozen_context_cold_model_work(context, step, work)?;
        let storage = self.original_cold_model_work(work)?;
        if !matches!(
            storage.state,
            RecordingState::Waiting | RecordingState::Recording
        ) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        storage
            .allowance
            .lock()
            .map_err(|_| publication_input_error("prepared context allowance is poisoned"))?
            .claim_model_context_prepared_model_roster(bank, bytes)?;
        Ok(Arc::clone(&storage.allowance))
    }

    pub fn resolve_model_context_prepared_model_rosters(
        &mut self,
        context: &SemanticModelContextContinuation,
        step: &SemanticPreparedStep,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.require_frozen_context_cold_model_work(context, step, work)?;
        let scope = Arc::clone(&step.scope);
        let tokens = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .tokens
            .clone();
        for token in tokens {
            let issued = SemanticPreparedStep {
                issuer: Arc::clone(&self.publication_issuer),
                scope: Arc::clone(&scope),
                token,
            };
            for content in &self.steps[&token].content {
                let TensorContentSeals::Model(seals) = &content.seals else {
                    continue;
                };
                let Some(prepared) = &seals.prepared else {
                    continue;
                };
                let mut original = prepared.roster_write.lock().map_err(|_| {
                    publication_input_error("prepared model roster custody is poisoned")
                })?;
                let Some(original) = original.as_mut() else {
                    continue;
                };
                if original.allowance.is_none() {
                    // The plan derives this exact branch and byte extent before
                    // admission. Its one claim survives late DMA completion.
                    original.allowance = Some(self.claim_model_context_prepared_model_roster(
                        context,
                        &issued,
                        work,
                        prepared.bank,
                        original.write.dma_bytes(),
                    )?);
                }
                original.write.resolve_original(
                    &self.domain,
                    &self.stream,
                    &self.provider,
                    &mut self.poisoned,
                    true,
                    original.allowance.as_ref(),
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn prepare_frozen_context_cold_model_work(
        &mut self,
        context: &SemanticModelContextContinuation,
        step: &SemanticPreparedStep,
        native_ceiling: [u64; 9],
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        context
            .inner
            .require_frozen_preparation(self, &step.scope)?;
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        build.check_retained(step, &self.publication_issuer)?;
        let address = build
            .program_steps
            .get(&step.token)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if build.program_admission.is_some()
            || address.kind != Some(SemanticTransitionKind::Recompute)
            || address.budget != context.inner.cold_budget()?
        {
            return Err(publication_input_error(
                "frozen context cold work requires its original future instruction without live admission",
            ));
        }
        let ordinal = address.ordinal;
        if let Some(previous) = self
            .steps
            .get(&step.token)
            .and_then(|owner| owner.cold_model_work.as_ref())
        {
            let original = previous
                .model_context
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
                || original.kind != ModelContextColdKind::Preparation
                || original.native_ceiling != native_ceiling
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let handle = self.model_context_cold_handle(step.token)?;
            self.continue_cold_model_work_initialization(&handle)?;
            return Ok(handle);
        }
        self.ensure_quiescent()?;
        self.require_prepared_structure_mutable()?;
        self.require_completed_cold_model_work()?;
        self.checked_prepared_step(step, false)?;
        let handle = self.install_cold_model_work(
            step.token,
            None,
            ordinal,
            context.inner.original_program_instruction()?.into(),
            SemanticColdModelWorkPurpose::PreparedActorRefresh,
            Some(Arc::clone(&step.scope)),
            Some(ColdWorkAdmission::ModelContext(
                context,
                ModelContextColdKind::Preparation,
                None,
            )),
        )?;
        self.freeze_model_context_cold_native_work(&handle, native_ceiling)?;
        Ok(handle)
    }

    pub fn require_frozen_context_cold_model_work(
        &self,
        context: &SemanticModelContextContinuation,
        step: &SemanticPreparedStep,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.check_cold_work_reader(ColdWorkReader::FrozenPrepared(context, step), work)?;
        Ok(())
    }

    pub fn finish_frozen_context_cold_model_work(
        &mut self,
        context: &SemanticModelContextContinuation,
        step: &SemanticPreparedStep,
        work: &SemanticColdModelWork,
        streams: &[u64],
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.finish_cold_model_work_for_reader(
            ColdWorkReader::FrozenPrepared(context, step),
            work,
            streams,
            SemanticColdModelWorkDisposition::Complete,
        )
    }

    pub fn resolve_frozen_context_cold_model_work(
        &mut self,
        context: &SemanticModelContextContinuation,
        step: &SemanticPreparedStep,
        work: &SemanticColdModelWork,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.resolve_cold_model_work_for_reader(ColdWorkReader::FrozenPrepared(context, step), work)
    }

    pub fn frozen_context_cold_model_work_buffer_port(
        &self,
        context: &SemanticModelContextContinuation,
        step: &SemanticPreparedStep,
        work: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<SemanticModelWorkBufferPort, SemanticTransitionError> {
        self.issue_cold_model_work_buffer_port(
            ColdWorkReader::FrozenPrepared(context, step),
            work,
            stream,
            region,
        )
    }

    pub fn export_frozen_context_cold_model_work_buffer_port(
        &mut self,
        context: &SemanticModelContextContinuation,
        step: &SemanticPreparedStep,
        port: &SemanticModelWorkBufferPort,
        work: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<SemanticModelWorkBufferExport, SemanticTransitionError> {
        self.export_cold_model_work_buffer_port_for_reader(
            ColdWorkReader::FrozenPrepared(context, step),
            port,
            work,
            region,
        )
    }

    pub fn frozen_context_cold_model_work_buffer_port_finalized_extent(
        &self,
        context: &SemanticModelContextContinuation,
        step: &SemanticPreparedStep,
        port: &SemanticModelWorkBufferPort,
        work: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<usize, SemanticTransitionError> {
        self.cold_model_work_buffer_port_finalized_extent_for_reader(
            ColdWorkReader::FrozenPrepared(context, step),
            port,
            work,
            region,
        )
    }

    pub fn frozen_context_cold_model_work_buffer(
        &mut self,
        context: &SemanticModelContextContinuation,
        step: &SemanticPreparedStep,
        work: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.cold_model_work_buffer_for_reader(
            ColdWorkReader::FrozenPrepared(context, step),
            work,
            stream,
            region,
        )
    }

    pub fn frozen_context_cold_model_work_buffer_layout(
        &self,
        context: &SemanticModelContextContinuation,
        step: &SemanticPreparedStep,
        work: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<(DeviceAllocationProvenance, SemanticTensorLayout), SemanticTransitionError> {
        let (_, allocation, layout) = self.cold_model_work_buffer_metadata(
            ColdWorkReader::FrozenPrepared(context, step),
            work,
            stream,
            region,
        )?;
        Ok((allocation, layout))
    }

    /// Retain the native seal and original activation callback's full plan in
    /// one report. Registration precedes admission and every native effect.
    pub(super) fn prepare_model_context_activation_work(
        &mut self,
        source: &SemanticTransitionSession,
        current: &SemanticPublishedLease,
        context: &SemanticModelContextContinuation,
        native_ceiling: [u64; 9],
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let scope = Arc::clone(&build.scope);
        let token = *build
            .tokens
            .first()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let address = build
            .program_steps
            .get(&token)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let ordinal = address.ordinal;
        context
            .inner
            .require_activation_report(source, current, self, &scope)?;
        let previous = self
            .steps
            .get(&token)
            .and_then(|owner| owner.cold_model_work.as_ref())
            .ok_or_else(|| {
                publication_input_error(
                    "context activation lost its original completed preparation report",
                )
            })?;
        let original = previous
            .model_context
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if !original.owner.ptr_eq(&Arc::downgrade(&context.inner)) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if original.kind == ModelContextColdKind::Activation {
            if original.native_ceiling != native_ceiling {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            if !matches!(
                previous.state,
                RecordingState::Waiting
                    | RecordingState::Recording
                    | RecordingState::Closed
                    | RecordingState::Submitting
                    | RecordingState::Submitted
                    | RecordingState::Completed
            ) {
                return Err(publication_input_error(
                    "context activation retains its failed or undeclared original report",
                ));
            }
            let handle = self.model_context_cold_handle(token)?;
            return Ok(handle);
        }
        let prior = previous.result.ok_or_else(|| {
            publication_input_error(
                "context activation requires known original cold preparation completion",
            )
        })?;
        let mut prior = prior;
        if let Some(material) = self.model_context_material_work.as_ref() {
            if !material.owner.ptr_eq(&Arc::downgrade(&context.inner)) {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            prior.include(material.result.ok_or_else(|| {
                publication_input_error(
                    "context activation requires its known original material completion",
                )
            })?)?;
            if let Some(previous) = material.prior {
                prior.include(previous)?;
            }
        }
        self.ensure_quiescent()?;
        self.require_completed_cold_model_work()?;
        let handle = self.install_cold_model_work(
            token,
            None,
            ordinal,
            context.inner.original_program_instruction()?.into(),
            SemanticColdModelWorkPurpose::PrivateModelPreparation,
            Some(scope),
            Some(ColdWorkAdmission::ModelContext(
                context,
                ModelContextColdKind::Activation,
                Some(prior),
            )),
        )?;
        self.freeze_model_context_cold_native_work(&handle, native_ceiling)?;
        let dma_bytes =
            size_of::<model_context_continuation::PreparedModelContextContinuationDevice>()
                .checked_add(size_of::<PublicationHeader>())
                .and_then(|bytes| {
                    bytes.checked_add(
                        3 * size_of::<PublicationLease>()
                            + size_of::<PublicationHeader>()
                            + 32 * size_of::<SemanticTextSlot>(),
                    )
                })
                .and_then(|bytes| {
                    bytes.checked_add(
                        self.publication.as_ref()?.bank_templates[0]
                            .len()
                            .checked_mul(size_of::<PublicationRange>())?,
                    )
                })
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        self.original_cold_model_work(&handle)?
            .allowance
            .lock()
            .map_err(|_| publication_input_error("context activation allowance is poisoned"))?
            .freeze_model_context_activation_dma(
                native_ceiling,
                [
                    8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    u64::try_from(dma_bytes)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                ],
            )?;
        self.prepare_cold_model_work_regions(&handle, 1)?;
        // This canonical enclosing scope grants registration only while its
        // one region is waiting. Every MODEL producer still requires its
        // complete admitted plan, initialized slots and region begin.
        self.begin_cold_model_work(&handle)?;
        Ok(handle)
    }

    /// The sole MODEL span uses the existing region protocol; native alias,
    /// publication and completion remain in the enclosing original report.
    pub fn model_context_activation_model_region(
        &self,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
    ) -> Result<SemanticColdModelWorkRegion, SemanticTransitionError> {
        let storage = self.original_cold_model_work(work)?;
        let original = storage
            .model_context
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if original.kind != ModelContextColdKind::Activation
            || !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
            || storage.regions.len() != 1
            || storage.state != RecordingState::Recording
        {
            return Err(publication_input_error(
                "activation MODEL span requires its original sole admitted cold region",
            ));
        }
        Ok(SemanticColdModelWorkRegion {
            work: work.clone(),
            index: 0,
        })
    }

    pub(super) fn model_context_activation_handle(
        &self,
        context: &SemanticModelContextContinuation,
    ) -> Result<Option<SemanticColdModelWork>, SemanticTransitionError> {
        let token = *self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .tokens
            .first()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let Some(storage) = self
            .steps
            .get(&token)
            .and_then(|step| step.cold_model_work.as_ref())
        else {
            return Ok(None);
        };
        if storage.model_context.as_ref().is_some_and(|original| {
            original.kind == ModelContextColdKind::Activation
                && original.owner.ptr_eq(&Arc::downgrade(&context.inner))
        }) {
            Ok(Some(self.model_context_cold_handle(token)?))
        } else {
            Ok(None)
        }
    }

    pub fn begin_model_context_activation_work(
        &mut self,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.model_context_activation_native_work(context, work)?;
        self.original_cold_model_work(work)?
            .allowance
            .lock()
            .map_err(|_| publication_input_error("activation content allowance is poisoned"))?
            .require_model_context_activation_prefill()?;
        self.continue_cold_model_work_initialization(work)?;
        if self.original_cold_model_work(work)?.state == RecordingState::Waiting {
            self.begin_cold_model_work(work)?;
        }
        if self.original_cold_model_work(work)?.state != RecordingState::Recording
            || !self.original_cold_model_work(work)?.initialized
        {
            return Err(publication_input_error(
                "activation requires the admitted full original callback plan",
            ));
        }
        Ok(())
    }

    pub fn freeze_model_context_activation_prefill(
        &self,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
        declarations: &[SemanticModelContextPrefillContentGeometry],
    ) -> Result<(), SemanticTransitionError> {
        self.model_context_activation_native_work(context, work)?;
        let storage = self.original_cold_model_work(work)?;
        if storage.state != RecordingState::Recording || storage.plan.is_some() {
            return Err(publication_input_error(
                "activation content is structural metadata before the one original admission",
            ));
        }
        if declarations.is_empty() || declarations.iter().any(|declaration| !declaration.verify) {
            return Err(publication_input_error(
                "activation must describe actual retained prefill verifications, not a new capture or empty substitute",
            ));
        }
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let roster = native_work_bound::FrozenModelContextPrefillRoster::new(
            context,
            budget,
            declarations,
            0,
        )?;
        storage
            .allowance
            .lock()
            .map_err(|_| publication_input_error("activation content allowance is poisoned"))?
            .freeze_model_context_activation_prefill(roster)
    }

    pub(super) fn finish_model_context_activation_work(
        &mut self,
        source: &SemanticTransitionSession,
        current: &SemanticPublishedLease,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
        streams: &[u64],
        disposition: SemanticColdModelWorkDisposition,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        if self.original_cold_model_work(work)?.state == RecordingState::Recording {
            if disposition == SemanticColdModelWorkDisposition::KnownRefusal {
                self.cancel_unentered_cold_model_work_regions(work, 1)?;
            }
            self.close_cold_model_work(work)?;
        }
        self.finish_cold_model_work_for_reader(
            ColdWorkReader::ModelContextActivation(context, source, current),
            work,
            streams,
            disposition,
        )
    }

    pub(super) fn model_context_activation_native_work(
        &self,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
    ) -> Result<
        (
            DeviceMemoryView<u64>,
            Arc<Mutex<native_work_bound::ColdNativeAllowance>>,
        ),
        SemanticTransitionError,
    > {
        let storage = self.original_cold_model_work(work)?;
        let original = storage
            .model_context
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if original.kind != ModelContextColdKind::Activation
            || !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !Arc::ptr_eq(owner, &context.inner))
            || !matches!(
                storage.state,
                RecordingState::Waiting | RecordingState::Recording
            )
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.graph
            .require_cold_work_owner(&storage.native_work.view(), false)
            .map_err(SemanticTransitionError::Semantic)?;
        Ok((storage.native_work.view(), Arc::clone(&storage.allowance)))
    }

    pub(super) fn claim_model_context_activation_work(
        &self,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        let (_, allowance) = self.model_context_activation_native_work(context, work)?;
        let storage = self.original_cold_model_work(work)?;
        if storage.state != RecordingState::Recording || !storage.initialized {
            return Err(publication_input_error(
                "native activation requires its original complete admitted callback plan",
            ));
        }
        let ceiling = self
            .original_cold_model_work(work)?
            .model_context
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .native_ceiling;
        let mut allowance = allowance
            .lock()
            .map_err(|_| publication_input_error("context activation allowance is poisoned"))?;
        allowance.claim_model_context_activation(ceiling)
    }

    /// Only confirmed original submissions awaiting their own completion are
    /// pending; preparation or driver entry alone is not submission proof.
    pub(super) fn model_context_activation_completion_pending(
        &self,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
    ) -> Result<bool, SemanticTransitionError> {
        let storage = self.original_cold_model_work(work)?;
        let original = storage
            .model_context
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if original.kind != ModelContextColdKind::Activation
            || !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !Arc::ptr_eq(owner, &context.inner))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if !storage.initialization_write_completed
            && storage
                .initialization_write
                .as_ref()
                .is_some_and(|write| write.completion_pending())
        {
            return Ok(true);
        }
        if storage
            .initialization_reset
            .as_ref()
            .is_some_and(OriginalNativeCommand::completion_pending)
        {
            return Ok(true);
        }
        let Some(completion) = &storage.completion else {
            return Ok(false);
        };
        let completion = completion.lock().map_err(|_| {
            publication_input_error("context activation completion owner is poisoned")
        })?;
        Ok((completion
            .metadata
            .as_ref()
            .is_some_and(|write| write.completion_pending())
            && !completion.metadata_completed)
            || completion.command.completion_pending()
            || completion.read.completion_pending())
    }

    pub(super) fn attach_model_context_activation_work(
        &mut self,
        current: &SemanticPublishedLease,
        context: &SemanticModelContextContinuation,
        target: &Self,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_reader(current)?;
        let storage = target.original_cold_model_work(work)?;
        let scope = storage
            .prepared_scope
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        context
            .inner
            .require_activation_report(self, current, target, scope)?;
        target.model_context_activation_native_work(context, work)?;
        let storage = target.original_cold_model_work(work)?;
        if storage.state != RecordingState::Recording || !storage.initialized {
            return Err(publication_input_error(
                "alias ordering cannot precede full activation admission",
            ));
        }
        self.require_model_context_cold_pair(context, target)?;
        let shared = target.share_cold_native_work(work)?;
        if self.graph.borrowed_cold_work().is_some() {
            self.graph
                .require_cold_work_owner(&shared.work, true)
                .map_err(SemanticTransitionError::Semantic)?;
            if self
                .graph
                .cold_work_allowance()
                .as_ref()
                .is_none_or(|owner| !Arc::ptr_eq(owner, &shared.allowance))
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            return Ok(());
        }
        self.graph
            .begin_borrowed_cold_work(shared.work, shared.custody, shared.allowance)
            .map_err(SemanticTransitionError::Semantic)
    }

    pub(super) fn detach_model_context_activation_work(
        &mut self,
        current: &SemanticPublishedLease,
        context: &SemanticModelContextContinuation,
        target: &Self,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.require_model_context_cold_pair(context, target)?;
        context.inner.require_activation_report(
            self,
            current,
            target,
            target
                .original_cold_model_work(work)?
                .prepared_scope
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
        )?;
        let storage = target.original_cold_model_work(work)?;
        let original = storage
            .model_context
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if original.kind != ModelContextColdKind::Activation
            || !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.graph
            .complete_cold_work(&storage.native_work.view())
            .map_err(SemanticTransitionError::Semantic)
    }

    fn require_model_context_cold_pair(
        &self,
        context: &SemanticModelContextContinuation,
        other: &Self,
    ) -> Result<(), SemanticTransitionError> {
        if self
            .model_context_continuation
            .as_ref()
            .is_none_or(|owner| !Arc::ptr_eq(owner, &context.inner))
            || other
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !Arc::ptr_eq(owner, &context.inner))
            || Arc::ptr_eq(&self.publication_issuer, &other.publication_issuer)
            || !Arc::ptr_eq(self.provider.memory(), other.provider.memory())
            || self.domain.stream_id() != other.domain.stream_id()
            || !Arc::ptr_eq(
                self.domain.execution_stream(),
                other.domain.execution_stream(),
            )
            || !Arc::ptr_eq(
                self.domain.execution_stream().context(),
                other.domain.execution_stream().context(),
            )
            || self.domain.execution_stream().context().cu_ctx()
                != other.domain.execution_stream().context().cu_ctx()
        {
            return Err(publication_input_error(
                "context cold work changed its original allocator, stream or context pair",
            ));
        }
        Ok(())
    }

    pub(super) fn resolve_model_context_activation_work(
        &mut self,
        source: &SemanticTransitionSession,
        current: &SemanticPublishedLease,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.resolve_cold_model_work_for_reader(
            ColdWorkReader::ModelContextActivation(context, source, current),
            work,
        )
    }

    pub(super) fn completed_model_context_cold_result(
        &self,
        context: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
    ) -> Result<CompletedModelContextColdResult, SemanticTransitionError> {
        let storage = self.original_cold_model_work(work)?;
        let original = storage
            .model_context
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if storage.state != RecordingState::Completed
            || original.kind != ModelContextColdKind::Activation
            || !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
        {
            return Err(publication_input_error(
                "context report binding requires its same completed original native activation",
            ));
        }
        let mut result = original
            .prior
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        result.include(
            storage
                .result
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
        )?;
        Ok(CompletedModelContextColdResult {
            instruction: Arc::clone(storage.admission.instruction()?),
            result,
        })
    }

    pub(super) fn model_context_cold_handle(
        &self,
        token: u64,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        let storage = self
            .steps
            .get(&token)
            .and_then(|owner| owner.cold_model_work.as_ref())
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        Ok(SemanticColdModelWork {
            issuer: Arc::clone(&self.publication_issuer),
            invocation: Arc::clone(&storage.invocation),
            token,
            operation_ordinal: storage.operation_ordinal,
            admission: Arc::clone(&storage.admission),
            final_cursor: Arc::clone(&storage.final_cursor),
        })
    }

    fn freeze_model_context_cold_native_work(
        &mut self,
        work: &SemanticColdModelWork,
        ceiling: [u64; 9],
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.original_cold_model_work(work)?;
        storage
            .allowance
            .lock()
            .map_err(|_| publication_input_error("model context native allowance is poisoned"))?
            .freeze_model_context_native_work(ceiling)?;
        self.steps
            .get_mut(&work.token)
            .expect("original context storage")
            .cold_model_work
            .as_mut()
            .expect("original context report")
            .model_context
            .as_mut()
            .expect("original context authority")
            .native_ceiling = ceiling;
        Ok(())
    }

    pub fn prepare_prepared_cold_model_work(
        &mut self,
        step: &SemanticPreparedStep,
        parent: &SemanticPublishedLease,
        capacity: usize,
        operation_ordinal: u64,
        admission: impl Into<SemanticColdModelWorkAdmission>,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.prepare_prepared_cold_model_work_inner(
            step,
            parent,
            Some(capacity),
            operation_ordinal,
            admission.into(),
            None,
        )
    }

    fn prepare_prepared_cold_model_work_inner(
        &mut self,
        step: &SemanticPreparedStep,
        parent: &SemanticPublishedLease,
        capacity_limit: Option<usize>,
        operation_ordinal: u64,
        admission: SemanticColdModelWorkAdmission,
        original_instruction: Option<&SemanticSegmentInstructionAdmission>,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.require_prepared_replay_parent(step, parent)?;
        if let Some(original) = original_instruction {
            self.require_segment_instruction_admission(original, parent)?;
            if !matches!(&admission, SemanticColdModelWorkAdmission::Ordinary(bytes)
                if Arc::ptr_eq(bytes, &self.instruction_admission(original)?.instruction))
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        }
        self.require_prepared_structure_mutable()?;
        self.require_completed_cold_model_work()?;
        let owner = self.checked_prepared_step(step, false)?;
        if capacity_limit == Some(0)
            || admission.as_ref().is_empty()
            || owner.cold_model_work.is_some()
            || owner.evaluation.is_some()
        {
            return Err(publication_input_error(
                "prepared cold work requires one original admitted callback before execution",
            ));
        }
        self.install_cold_model_work(
            step.token,
            capacity_limit,
            operation_ordinal,
            admission,
            SemanticColdModelWorkPurpose::PreparedActorRefresh,
            Some(Arc::clone(&step.scope)),
            original_instruction.map(|original| {
                ColdWorkAdmission::Instruction(original, InstructionColdRole::Prepared)
            }),
        )
    }

    pub fn prepare_admitted_prepared_cold_model_work(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        step: &SemanticPreparedStep,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        let build = self.prepared_segment.as_ref().ok_or_else(|| {
            publication_input_error("prepared cold work lost its original segment owner")
        })?;
        if build
            .program_admission
            .as_ref()
            .is_none_or(|original| !original.same_handle(admission))
            || build.requested_kind(step, &self.publication_issuer)?
                != SemanticTransitionKind::Update
        {
            return Err(publication_input_error(
                "prepared cold work changed its admitted Update",
            ));
        }
        let position = build
            .tokens
            .iter()
            .position(|token| *token == step.token)
            .ok_or_else(|| {
                publication_input_error("prepared cold work lost its original step position")
            })?;
        let ordinal = self
            .admitted_segment_first_program_ordinal(admission)?
            .checked_add(
                u64::try_from(position)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            )
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let instruction = Arc::clone(&self.instruction_admission(admission)?.instruction);
        let original = self.instruction_admission(admission)?;
        if let Some((_, result)) = original
            .prepared_cold_results
            .iter()
            .find(|(address, _)| *address == ordinal)
        {
            let owner = self.steps.get(&step.token).ok_or_else(|| {
                publication_input_error("retain the original prepared cold allocation attempt")
            })?;
            let Some(storage) = owner.cold_model_work.as_ref() else {
                if result.is_some() {
                    return Err(publication_input_error(
                        "completed prepared cold work lost its original storage",
                    ));
                }
                // Canonical installation retains storage before its first
                // reset or transfer. Resume only this unentered allocation
                // prefix, preserving the original ordinal and report claim.
                return self.prepare_prepared_cold_model_work_inner(
                    step,
                    parent,
                    None,
                    ordinal,
                    instruction.into(),
                    Some(admission),
                );
            };
            if storage.operation_ordinal != ordinal
                || !storage.admission.matches_instruction(&instruction)
            {
                return Err(publication_input_error(
                    "retain the original prepared cold allocation attempt",
                ));
            }
            let handle = SemanticColdModelWork {
                issuer: Arc::clone(&self.publication_issuer),
                invocation: Arc::clone(&storage.invocation),
                token: step.token,
                operation_ordinal: ordinal,
                admission: Arc::clone(&storage.admission),
                final_cursor: Arc::clone(&storage.final_cursor),
            };
            self.continue_cold_model_work_initialization(&handle)?;
            return Ok(handle);
        }
        let original = self.instruction_admission_mut(admission)?;
        original.prepared_cold_results.push((ordinal, None));
        self.prepare_prepared_cold_model_work_inner(
            step,
            parent,
            None,
            ordinal,
            instruction.into(),
            Some(admission),
        )
    }

    pub fn finish_admitted_prepared_cold_model_work(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        step: &SemanticPreparedStep,
        work: &SemanticColdModelWork,
        streams: &[u64],
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        let build = self.prepared_segment.as_ref().ok_or_else(|| {
            publication_input_error("prepared cold completion lost its original segment")
        })?;
        if build
            .program_admission
            .as_ref()
            .is_none_or(|original| !original.same_handle(admission))
        {
            return Err(publication_input_error(
                "prepared cold completion changed its instruction",
            ));
        }
        let position = build
            .tokens
            .iter()
            .position(|token| *token == step.token)
            .ok_or_else(|| {
                publication_input_error("prepared cold completion lost its original position")
            })?;
        let ordinal = self
            .admitted_segment_first_program_ordinal(admission)?
            .checked_add(
                u64::try_from(position)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            )
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let original = self.instruction_admission(admission)?;
        if work.operation_ordinal != ordinal
            || !work.admission.matches_instruction(&original.instruction)
            || !original
                .prepared_cold_results
                .iter()
                .any(|(address, _)| *address == ordinal)
        {
            return Err(publication_input_error(
                "prepared cold completion has no original attempt",
            ));
        }
        let result = self.finish_prepared_cold_model_work(
            step,
            parent,
            work,
            streams,
            SemanticColdModelWorkDisposition::Complete,
        )?;
        let original = self.instruction_admission_mut(admission)?;
        original
            .prepared_cold_results
            .iter_mut()
            .find(|(address, _)| *address == ordinal)
            .expect("retained original prepared report")
            .1 = Some(result);
        Ok(result)
    }

    fn install_cold_model_work(
        &mut self,
        token: u64,
        capacity_limit: Option<usize>,
        operation_ordinal: u64,
        admission: SemanticColdModelWorkAdmission,
        purpose: SemanticColdModelWorkPurpose,
        prepared_scope: Option<Arc<()>>,
        instruction: Option<ColdWorkAdmission<'_>>,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        let model_context = match &instruction {
            Some(ColdWorkAdmission::ModelContext(context, kind, prior)) => {
                let original = context.inner.original_program_instruction()?;
                if self
                    .model_context_continuation
                    .as_ref()
                    .is_none_or(|owner| !Arc::ptr_eq(owner, &context.inner))
                    || !matches!(&admission, SemanticColdModelWorkAdmission::Ordinary(bytes)
                        if Arc::ptr_eq(bytes, &original))
                    || prepared_scope.is_none()
                {
                    return Err(publication_input_error(
                        "model context cold work changed its original instruction or native owner",
                    ));
                }
                Some(ModelContextColdWork {
                    owner: Arc::downgrade(&context.inner),
                    kind: *kind,
                    budget: context.inner.cold_budget()?,
                    prior: *prior,
                    native_ceiling: [0; 9],
                })
            }
            _ => None,
        };
        if let SemanticColdModelWorkAdmission::Loaded(bytes) = &admission {
            bytes.require_budget(
                self.replay_material_budget
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?,
            )?;
        } else if self.replay_material_budget.is_some() && instruction.is_none() {
            return Err(publication_input_error(
                "bounded cold work requires its original loaded signed record",
            ));
        }
        let admission_reservation = self
            .replay_material_budget
            .as_ref()
            .map(|budget| {
                budget.reserve_auxiliary(
                    u64::try_from(size_of::<RetainedColdAdmission>() + 2 * size_of::<usize>())
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                )
            })
            .transpose()?;
        let admission = Arc::new(RetainedColdAdmission {
            material: admission,
            _reservation: admission_reservation,
        });
        let bytes = (15 + 11) * size_of::<u64>();
        let mut reservation = self
            .provider
            .memory()
            .reserve_bytes(bytes as u64)
            .map_err(|error| runtime_error("cold model work reservation", error))?;
        let native_work = reservation
            .alloc(11)
            .map_err(|error| runtime_error("cold semantic work allocation", error))?;
        let report = reservation
            .alloc(15)
            .map_err(|error| runtime_error("cold model work report allocation", error))?;
        let invocation = Arc::new(());
        let final_cursor = Arc::new(std::sync::OnceLock::new());
        let handle = SemanticColdModelWork {
            issuer: Arc::clone(&self.publication_issuer),
            invocation: Arc::clone(&invocation),
            token,
            operation_ordinal,
            admission: Arc::clone(&admission),
            final_cursor: Arc::clone(&final_cursor),
        };
        // Keep the completed storage until all new allocation succeeds. The
        // new invocation is retained before reset or any model callback; an
        // allocation failure grants no rights to replay the prior callback.
        self.steps
            .get_mut(&token)
            .expect("checked original reader")
            .cold_model_work = Some(ColdModelWorkStorage {
            invocation,
            final_cursor,
            aliases: Arc::new(()),
            purpose,
            prepared_scope,
            model_context,
            operation_ordinal,
            admission,
            allowance: Arc::new(Mutex::new(native_work_bound::ColdNativeAllowance::new(
                purpose,
            ))),
            source_geometry: None,
            capacity_limit,
            work: None,
            native_work,
            initialization_reset: None,
            initialization_write: None,
            initialization_write_completed: false,
            initialization_poisoned: false,
            native_initialized: false,
            native_initialization_recorded: false,
            initialized: false,
            report,
            completion: None,
            state: RecordingState::Waiting,
            regions: Vec::new(),
            plan: None,
            evaluation_cleanup: None,
            disposition: None,
            result: None,
            streams: None,
            operations: Vec::new(),
            pending_operation: None,
            pending_operation_end: 0,
            stopped_before_entry: false,
        });
        if let Some(ColdWorkAdmission::Instruction(instruction, role)) = instruction {
            // Admission owns this exact invocation before the first reset or
            // transfer. An uncertain initialization never grants another attempt.
            let original = self.instruction_admission_mut(instruction)?;
            match role {
                InstructionColdRole::Preparation => original.cold_work = Some(handle.clone()),
                InstructionColdRole::Retirement => original.retirement_work = Some(handle.clone()),
                InstructionColdRole::Prepared => (),
            }
        }
        let storage = self.steps[&token]
            .cold_model_work
            .as_ref()
            .expect("retained original cold report");
        // This registers custody only. No event, reset, transfer or model slot
        // exists until the original complete declaration seals its exact E.
        self.graph
            .begin_cold_work(storage.native_work.view(), Arc::clone(&storage.allowance))
            .map_err(SemanticTransitionError::Semantic)?;
        Ok(handle)
    }

    fn continue_cold_model_work_initialization(
        &mut self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.initialize_original_cold_model_work(handle, None)
    }

    fn initialize_original_cold_model_work(
        &mut self,
        handle: &SemanticColdModelWork,
        terminal_reader: Option<ColdWorkReader<'_>>,
    ) -> Result<(), SemanticTransitionError> {
        if let Some(reader) = terminal_reader {
            self.check_cold_work_completion_reader(reader, handle)?;
            let original = self.original_cold_model_work(handle)?;
            if original.plan.is_some()
                || original.state != RecordingState::Closed
                || original.pending_operation.is_some()
                || !original.operations.is_empty()
                || !original.recorded_events().is_empty()
                || original.regions.iter().any(|state| {
                    !matches!(state, RecordingState::Waiting | RecordingState::NotEntered)
                })
            {
                return Err(publication_input_error(
                    "terminal cold initialization requires its authenticated unentered original report",
                ));
            }
        }
        // Authenticate the exact retained allocation without a healthy-session
        // precondition: joining its entered prefix grants no unrelated work.
        let another_initializer = self.steps.iter().any(|(token, step)| {
            *token != handle.token
                && step.cold_model_work.as_ref().is_some_and(|storage| {
                    (storage.initialization_write.is_some() && !storage.native_initialized)
                        || (storage.initialization_reset.is_some() && !storage.initialized)
                })
        });
        let storage = self
            .steps
            .get_mut(&handle.token)
            .and_then(|step| step.cold_model_work.as_mut())
            .ok_or_else(|| {
                publication_input_error("cold initialization lost its original owner")
            })?;
        if !Arc::ptr_eq(&handle.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&handle.invocation, &storage.invocation)
            || !Arc::ptr_eq(&handle.admission, &storage.admission)
            || handle.operation_ordinal != storage.operation_ordinal
        {
            return Err(publication_input_error(
                "cold initialization belongs to another original invocation",
            ));
        }
        if storage.initialized {
            return Ok(());
        }
        let capacity = match &storage.plan {
            Some(plan) => usize::try_from(plan.quantities()[1])
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            None if terminal_reader.is_some() => 0,
            None => return Ok(()),
        };
        let historical_poison = self.poisoned || self.graph.ensure_not_poisoned().is_err();
        if historical_poison || another_initializer {
            // A different owner or abort may have poisoned the Session while
            // this initializer was unresolved. Join only its entered work;
            // never submit an unentered suffix or grant callback entry.
            if let Some(command) = &mut storage.initialization_reset {
                command.resolve_entered(&mut storage.initialization_poisoned)?;
            }
            if let Some(write) = storage
                .initialization_write
                .as_mut()
                .filter(|write| write.entered())
            {
                write.resolve().map_err(|error| {
                    storage.initialization_poisoned = true;
                    runtime_error("cold semantic work initialization", error)
                })?;
                storage.initialization_write_completed = true;
            }
            return Err(if historical_poison {
                SemanticTransitionError::Poisoned
            } else {
                SemanticTransitionError::OverlappingLaunch
            });
        }
        if capacity > 0 && storage.work.is_none() {
            let bytes = capacity
                .checked_mul(size_of::<ModelWorkEvent>() + 3 * size_of::<u64>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let mut reservation = self
                .provider
                .memory()
                .reserve_bytes(bytes)
                .map_err(|error| runtime_error("admitted cold model work reservation", error))?;
            let mut work = PreparedModelWork::allocate(&self.provider, &mut reservation, capacity)?;
            work.uncaptured_recording = storage.state == RecordingState::Recording;
            storage.work = Some(work);
        }
        if storage.initialization_reset.is_none() && capacity > 0 {
            storage.initialization_reset = Some(OriginalNativeCommand::new(&self.domain)?);
        }
        if !storage.native_initialized {
            storage
                .allowance
                .lock()
                .map_err(|_| publication_input_error("cold native allowance was poisoned"))?
                .claim_native_tally_initialization()?;
            if storage.initialization_write.is_none() {
                storage.initialization_write = Some(
                    crate::device::RetainedDeviceWrite::new(
                        self.domain.execution_stream(),
                        &[0u64; 11],
                        storage.native_work.view(),
                    )
                    .map_err(|error| runtime_error("cold semantic work staging", error))?,
                );
            }
            let write = storage
                .initialization_write
                .as_mut()
                .expect("retained original tally initialization");
            let submission = if write.entered() {
                Ok(())
            } else {
                write.enqueue(self.domain.execution_stream())
            };
            if write.submitted() {
                storage
                    .allowance
                    .lock()
                    .map_err(|_| publication_input_error("cold native allowance was poisoned"))?
                    .record_submitted_dma(
                        11 * size_of::<u64>(),
                        &mut storage.native_initialization_recorded,
                    )?;
            }
            submission.map_err(|error| {
                storage.initialization_poisoned |= write.entered();
                runtime_error("cold semantic work initialization", error)
            })?;
            write.resolve().map_err(|error| {
                storage.initialization_poisoned |= write.entered();
                runtime_error("cold semantic work initialization", error)
            })?;
            storage.initialization_write_completed = true;
            storage.native_initialized = true;
        }
        if let Some(work) = &mut storage.work {
            work.reset_slots_with_original(
                &self.domain,
                &mut storage.initialization_poisoned,
                0,
                capacity,
                Some(&storage.native_work.view()),
                storage.initialization_reset.as_mut(),
            )?;
        }
        storage.initialized = true;
        if storage.state == RecordingState::Recording {
            storage
                .allowance
                .lock()
                .map_err(|_| publication_input_error("context prefill allowance is poisoned"))?
                .set_model_context_prefill_recording(true);
        }
        Ok(())
    }

    pub(super) fn has_uninitialized_cold_model_work(&self) -> bool {
        self.steps.values().any(|step| {
            step.cold_model_work.as_ref().is_some_and(|storage| {
                (storage.initialization_write.is_some() && !storage.native_initialized)
                    || (storage.initialization_reset.is_some() && !storage.initialized)
            })
        })
    }

    pub(super) fn has_pending_cold_model_work_completion(&self, except_token: Option<u64>) -> bool {
        self.steps.iter().any(|(token, step)| {
            except_token != Some(*token)
                && step
                    .cold_model_work
                    .as_ref()
                    .is_some_and(|storage| storage.completion.is_some() && storage.result.is_none())
        })
    }

    fn cold_model_work(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<&ColdModelWorkStorage, SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.original_cold_model_work(handle)
    }

    // Metadata authentication alone permits the original completion owner to
    // join its pending consumer edge. It grants no new model or report effect.
    fn original_cold_model_work(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<&ColdModelWorkStorage, SemanticTransitionError> {
        let work = self
            .steps
            .get(&handle.token)
            .and_then(|step| step.cold_model_work.as_ref())
            .ok_or_else(|| {
                publication_input_error("cold model work lost its original operation")
            })?;
        if !Arc::ptr_eq(&handle.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&handle.invocation, &work.invocation)
            || !Arc::ptr_eq(&handle.admission, &work.admission)
            || !Arc::ptr_eq(&handle.final_cursor, &work.final_cursor)
            || handle.operation_ordinal != work.operation_ordinal
        {
            return Err(publication_input_error(
                "cold model work belongs to another Session, admission or operation",
            ));
        }
        Ok(work)
    }

    pub(super) fn original_source_geometry_allowance(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<Arc<Mutex<ColdNativeAllowance>>, SemanticTransitionError> {
        Ok(Arc::clone(
            &self.original_cold_model_work(handle)?.allowance,
        ))
    }

    pub(super) fn original_source_geometry_roster(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<
        Option<&Arc<prepared_source_content::OriginalSourceGeometryRoster>>,
        SemanticTransitionError,
    > {
        Ok(self
            .original_cold_model_work(handle)?
            .source_geometry
            .as_ref())
    }

    pub(super) fn register_original_source_geometry_roster(
        &mut self,
        handle: &SemanticColdModelWork,
        roster: Arc<prepared_source_content::OriginalSourceGeometryRoster>,
    ) -> Result<(), SemanticTransitionError> {
        let original = self.original_cold_model_work(handle)?;
        if original.source_geometry.is_some()
            || original.purpose != SemanticColdModelWorkPurpose::PrivatePrefix
            || original.native_initialized
            || original.initialization_write.is_some()
            || original.initialization_reset.is_some()
            || original.state != RecordingState::Waiting
            || original.plan.is_some()
            || original.completion.is_some()
            || original.stopped_before_entry
        {
            return Err(publication_input_error(
                "original child geometry must be retained before its first native effect",
            ));
        }
        self.steps
            .get_mut(&handle.token)
            .expect("authenticated original report")
            .cold_model_work
            .as_mut()
            .expect("authenticated original report")
            .source_geometry = Some(roster);
        Ok(())
    }

    fn check_cold_work_reader(
        &self,
        reader: ColdWorkReader<'_>,
        handle: &SemanticColdModelWork,
    ) -> Result<u64, SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.check_cold_work_completion_reader(reader, handle)
    }

    fn check_cold_work_completion_reader(
        &self,
        reader: ColdWorkReader<'_>,
        handle: &SemanticColdModelWork,
    ) -> Result<u64, SemanticTransitionError> {
        let storage = self.original_cold_model_work(handle)?;
        let token = match reader {
            ColdWorkReader::Published(lease) => {
                self.checked_original_consumer_step(lease)?;
                if !lease.active || storage.prepared_scope.is_some() {
                    return Err(publication_input_error(
                        "prepared cold work cannot substitute a published reader",
                    ));
                }
                lease.token
            }
            ColdWorkReader::Admitted(admission, parent) => {
                self.require_segment_instruction_admission(admission, parent)?;
                let original = self.instruction_admission(admission)?;
                if original.state != actor_refresh_program::InstructionState::Released
                    || original
                        .cold_work
                        .as_ref()
                        .is_none_or(|work| !work.same_invocation(handle))
                    || storage.prepared_scope.is_some()
                {
                    return Err(publication_input_error(
                        "cold completion requires its same handed-off instruction and original invocation",
                    ));
                }
                self.checked_original_consumer_step(parent)?;
                parent.token
            }
            ColdWorkReader::AdmittedCancellation(admission, parent, proof) => {
                self.require_segment_instruction_admission(admission, parent)?;
                let original = self.instruction_admission(admission)?;
                let build = self.prepared_segment.as_ref().ok_or_else(|| {
                    publication_input_error(
                        "cancelled preparation lost its original prepared owner",
                    )
                })?;
                build.require_non_submission(proof)?;
                if original.state != actor_refresh_program::InstructionState::Admitted
                    || !parent.active
                    || build
                        .program_admission
                        .as_ref()
                        .is_none_or(|retained| !retained.same_handle(admission))
                    || original
                        .cold_work
                        .as_ref()
                        .is_none_or(|work| !work.same_invocation(handle))
                    || storage.purpose != SemanticColdModelWorkPurpose::PrivatePrefix
                    || !storage.admission.matches_instruction(&original.instruction)
                    || storage.prepared_scope.is_some()
                {
                    return Err(publication_input_error(
                        "cancelled preparation changed its original admission, parent or prefix invocation",
                    ));
                }
                self.checked_original_consumer_step(parent)?;
                parent.token
            }
            ColdWorkReader::AdmittedRetirement(admission) => {
                let original = self.instruction_admission(admission)?;
                let build = self.prepared_segment.as_ref().ok_or_else(|| {
                    publication_input_error(
                        "instruction retirement lost its original prepared owner",
                    )
                })?;
                if let Some(proof) = &original.retirement_cancellation {
                    build.require_non_submission(proof)?;
                } else if !build.completed {
                    return Err(publication_input_error(
                        "cold retirement requires actual original segment completion",
                    ));
                }
                if build
                    .program_admission
                    .as_ref()
                    .is_none_or(|original| !original.same_handle(admission))
                    || original
                        .retirement_work
                        .as_ref()
                        .is_none_or(|work| !work.same_invocation(handle))
                    || storage
                        .prepared_scope
                        .as_ref()
                        .is_none_or(|scope| !Arc::ptr_eq(scope, &build.scope))
                    || !build.tokens.contains(&handle.token)
                {
                    return Err(publication_input_error(
                        "cold retirement changed its same completed segment or invocation",
                    ));
                }
                handle.token
            }
            ColdWorkReader::Prepared(step, parent) => {
                self.require_original_prepared_replay_parent(step, parent)?;
                if storage
                    .prepared_scope
                    .as_ref()
                    .is_none_or(|scope| !Arc::ptr_eq(scope, &step.scope))
                {
                    return Err(publication_input_error(
                        "prepared cold work changed its original construction scope",
                    ));
                }
                step.token
            }
            ColdWorkReader::FrozenPrepared(context, step) => {
                let original = storage
                    .model_context
                    .as_ref()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if original.kind == ModelContextColdKind::Activation {
                    context.inner.require_activation_scope(self, &step.scope)?;
                } else if original.kind == ModelContextColdKind::Preparation {
                    context
                        .inner
                        .require_frozen_preparation(self, &step.scope)?;
                } else {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                if !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
                    || storage
                        .prepared_scope
                        .as_ref()
                        .is_none_or(|scope| !Arc::ptr_eq(scope, &step.scope))
                {
                    return Err(publication_input_error(
                        "frozen cold work changed its original continuation or prepared scope",
                    ));
                }
                step.token
            }
            ColdWorkReader::ModelContextActivation(context, source, current) => {
                let scope = storage
                    .prepared_scope
                    .as_ref()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                context
                    .inner
                    .require_activation_report(source, current, self, scope)?;
                let original = storage
                    .model_context
                    .as_ref()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if original.kind != ModelContextColdKind::Activation
                    || !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
                {
                    return Err(publication_input_error(
                        "context activation changed its original bounded seal and callback report",
                    ));
                }
                handle.token
            }
            ColdWorkReader::ModelContextMaterial(context, target, parent) => {
                self.require_model_context_cold_pair(context, target)?;
                self.checked_original_consumer_step(parent)?;
                let original = storage
                    .model_context
                    .as_ref()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let borrowed = target
                    .model_context_material_work
                    .as_ref()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if !original.owner.ptr_eq(&Arc::downgrade(&context.inner))
                    || !matches!(
                        original.kind,
                        ModelContextColdKind::Checkpoint | ModelContextColdKind::DormantRestore
                    )
                    || !borrowed.owner.ptr_eq(&Arc::downgrade(&context.inner))
                    || !borrowed.work.same_invocation(handle)
                    || borrowed.source_parent != parent.identity
                    || !parent.active
                    || parent.token != handle.token
                    || self.publication.as_ref().is_none_or(|storage| {
                        !Arc::ptr_eq(storage, &context.inner.source_publication)
                    })
                    || !Arc::ptr_eq(&self.publication_issuer, &context.inner.source_issuer)
                    || !Arc::ptr_eq(&target.publication_issuer, &context.inner.target_issuer)
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                parent.token
            }
        };
        if token != handle.token {
            return Err(publication_input_error(
                "cold work changed its original native storage owner",
            ));
        }
        Ok(token)
    }

    pub fn cold_model_work_stream(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<u64, SemanticTransitionError> {
        self.cold_model_work(handle)?;
        Ok(self.stream.cu_stream() as u64)
    }

    /// Whether this original report has retained its sole complete plan.
    pub fn cold_model_work_plan_is_admitted(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<bool, SemanticTransitionError> {
        Ok(self.original_cold_model_work(handle)?.plan.is_some())
    }

    /// Structural future reads must be issued before this original report's
    /// sole declaration or any region operation has entered.
    pub fn require_cold_model_work_resident_read_preparation(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.original_cold_model_work(handle)?;
        if storage.plan.is_some()
            || storage.regions.len() != 4
            || storage
                .regions
                .iter()
                .any(|state| *state != RecordingState::Waiting)
            || !storage.operations.is_empty()
            || storage.pending_operation.is_some()
            || storage.stopped_before_entry
            || storage.initialization_reset.is_some()
            || storage.initialization_write.is_some()
            || storage.completion.is_some()
            || storage.result.is_some()
        {
            return Err(publication_input_error(
                "future resident reads require the original undeclared four-region report",
            ));
        }
        Ok(())
    }

    /// Fixed registration-region roster of this exact original report.
    pub fn cold_model_work_region_count(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<usize, SemanticTransitionError> {
        Ok(self.cold_model_work(handle)?.regions.len())
    }

    /// Validate every immutable original region without entering any callback.
    pub fn require_cold_model_work_report_regions(
        &self,
        handle: &SemanticColdModelWork,
        regions: &[SemanticColdModelWorkRegion],
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.cold_model_work(handle)?;
        if storage.regions.len() != regions.len()
            || regions.iter().enumerate().any(|(index, region)| {
                region.index != index || !region.work.same_invocation(handle)
            })
        {
            return Err(publication_input_error(
                "cold metadata changed its complete original region roster",
            ));
        }
        Ok(())
    }

    /// Authenticate the genuine later callback parent against this report's
    /// existing attached tally. This does not attach, reset or enqueue anything.
    pub fn require_cold_model_work_callback_parent(
        &self,
        handle: &SemanticColdModelWork,
        callback: &SemanticTransitionSession,
        parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.cold_model_work(handle)?;
        callback.checked_reader(parent)?;
        if !Arc::ptr_eq(&self.provider, &callback.provider)
            || self.domain.stream_id() != callback.domain.stream_id()
            || callback
                .graph
                .cold_work_allowance()
                .as_ref()
                .is_none_or(|allowance| !Arc::ptr_eq(allowance, &storage.allowance))
        {
            return Err(publication_input_error(
                "cold callback parent changed its original report domain or allowance",
            ));
        }
        callback
            .graph
            .require_cold_work_owner(
                &storage.native_work.view(),
                !Arc::ptr_eq(&self.publication_issuer, &callback.publication_issuer),
            )
            .map_err(SemanticTransitionError::Semantic)
    }

    /// A transferred handle must identify the same privately minted invocation,
    /// admission, operation and region. Its issuer/token may change only because
    /// the original native handoff moved that storage to this surviving reader.
    pub fn require_cold_model_work_handoff(
        &self,
        original: &SemanticColdModelWork,
        transferred: &SemanticColdModelWork,
        original_region: Option<&SemanticColdModelWorkRegion>,
        transferred_region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.cold_model_work(transferred)?;
        if !Arc::ptr_eq(&original.invocation, &storage.invocation)
            || !Arc::ptr_eq(&original.admission, &storage.admission)
            || !Arc::ptr_eq(&original.final_cursor, &storage.final_cursor)
            || original.operation_ordinal != storage.operation_ordinal
            || storage.state != RecordingState::Recording
            || match (original_region, transferred_region) {
                (Some(old), Some(new)) => {
                    old.index != new.index
                        || !old.work.same_invocation(original)
                        || !new.work.same_invocation(transferred)
                        || storage.regions.get(new.index).is_none()
                }
                (None, None) => !storage.regions.is_empty(),
                _ => true,
            }
        {
            return Err(publication_input_error(
                "cold callback handoff changed its original report or region identity",
            ));
        }
        Ok(())
    }

    /// Exact native declaration/physical-slot mapping, including proven skips.
    /// Unknown attempts and stopped prefixes cannot authorize continuation.
    pub fn cold_model_work_report_cursor(
        &self,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<SemanticColdModelWorkReportCursor, SemanticTransitionError> {
        let storage = self.cold_model_work(handle)?;
        if region.is_some_and(|region| !region.work.same_invocation(handle)) {
            return Err(publication_input_error(
                "cold cursor changed its original region owner",
            ));
        }
        storage.report_cursor(region.map(|region| region.index), false)
    }

    /// Admit the complete ordered model recipe before its first actual recorder
    /// begins. An enclosing region report may already be open without any model
    /// event. Returned quantities are (work bound, event count, call upper).
    pub fn admit_cold_model_work_plan(
        &mut self,
        handle: &SemanticColdModelWork,
        operations: &[(ModelWorkKind, Vec<u64>)],
        region_ends: &[usize],
        evaluation_content: Option<SemanticColdEvaluationContent>,
    ) -> Result<[u64; 3], SemanticTransitionError> {
        self.admit_cold_model_work_plan_in_region(
            handle,
            None,
            operations,
            region_ends,
            evaluation_content,
            &[],
            &[],
            true,
        )
    }

    pub fn admit_cold_model_work_region_plan(
        &mut self,
        region: &SemanticColdModelWorkRegion,
        operations: &[(ModelWorkKind, Vec<u64>)],
        region_ends: &[usize],
        evaluation_content: Option<SemanticColdEvaluationContent>,
    ) -> Result<[u64; 3], SemanticTransitionError> {
        self.admit_cold_model_work_plan_in_region(
            &region.work,
            Some(region),
            operations,
            region_ends,
            evaluation_content,
            &[],
            &[],
            true,
        )
    }

    pub fn admit_cold_model_work_plan_with_sources(
        &mut self,
        handle: &SemanticColdModelWork,
        operations: &[(ModelWorkKind, Vec<u64>)],
        region_ends: &[usize],
        evaluation_content: Option<SemanticColdEvaluationContent>,
        sources: &[SemanticPreparedSourceContent],
        resident_reads: &[(usize, SemanticResidentModelRead)],
    ) -> Result<[u64; 3], SemanticTransitionError> {
        self.admit_cold_model_work_plan_in_region(
            handle,
            None,
            operations,
            region_ends,
            evaluation_content,
            sources,
            resident_reads,
            true,
        )
    }

    /// Retain and price the canonical complete recipe before any counter or
    /// recorder initialization. Original owners must outlive later admission.
    pub fn declare_cold_model_work_plan_with_sources(
        &mut self,
        handle: &SemanticColdModelWork,
        operations: &[(ModelWorkKind, Vec<u64>)],
        region_ends: &[usize],
        evaluation_content: Option<SemanticColdEvaluationContent>,
        sources: &[SemanticPreparedSourceContent],
        resident_reads: &[(usize, SemanticResidentModelRead)],
    ) -> Result<[u64; 3], SemanticTransitionError> {
        self.admit_cold_model_work_plan_in_region(
            handle,
            None,
            operations,
            region_ends,
            evaluation_content,
            sources,
            resident_reads,
            false,
        )
    }

    pub fn admit_cold_model_work_region_plan_with_sources(
        &mut self,
        region: &SemanticColdModelWorkRegion,
        operations: &[(ModelWorkKind, Vec<u64>)],
        region_ends: &[usize],
        evaluation_content: Option<SemanticColdEvaluationContent>,
        sources: &[SemanticPreparedSourceContent],
        resident_reads: &[(usize, SemanticResidentModelRead)],
    ) -> Result<[u64; 3], SemanticTransitionError> {
        self.admit_cold_model_work_plan_in_region(
            &region.work,
            Some(region),
            operations,
            region_ends,
            evaluation_content,
            sources,
            resident_reads,
            true,
        )
    }

    pub fn declare_cold_model_work_region_plan_with_sources(
        &mut self,
        region: &SemanticColdModelWorkRegion,
        operations: &[(ModelWorkKind, Vec<u64>)],
        region_ends: &[usize],
        evaluation_content: Option<SemanticColdEvaluationContent>,
        sources: &[SemanticPreparedSourceContent],
        resident_reads: &[(usize, SemanticResidentModelRead)],
    ) -> Result<[u64; 3], SemanticTransitionError> {
        self.admit_cold_model_work_plan_in_region(
            &region.work,
            Some(region),
            operations,
            region_ends,
            evaluation_content,
            sources,
            resident_reads,
            false,
        )
    }

    fn admit_cold_model_work_plan_in_region(
        &mut self,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
        operations: &[(ModelWorkKind, Vec<u64>)],
        region_ends: &[usize],
        evaluation_content: Option<SemanticColdEvaluationContent>,
        sources: &[SemanticPreparedSourceContent],
        resident_reads: &[(usize, SemanticResidentModelRead)],
        initialize: bool,
    ) -> Result<[u64; 3], SemanticTransitionError> {
        let mut retained_plan = false;
        let result = (|| {
            if let Some(region) = region {
                self.require_cold_model_work_region_metadata(
                    region,
                    RecordingState::Waiting,
                    !initialize,
                )?;
            } else {
                let storage = self.cold_model_work(handle)?;
                if storage.state != RecordingState::Waiting || !storage.regions.is_empty() {
                    return Err(publication_input_error(
                        "cold plan requires its original waiting recorder",
                    ));
                }
            }
            let storage = self.cold_model_work(handle)?;
            if storage.plan.is_some()
                || !storage.recorded_events().is_empty()
                || storage.regions.iter().any(|state| {
                    !matches!(state, RecordingState::Waiting | RecordingState::NotEntered)
                })
            {
                return Err(publication_input_error(
                    "the original cold operation plan is admitted once",
                ));
            }
            let capacity = operations.len();
            if storage.model_context.as_ref().is_some_and(|context| {
                matches!(
                    context.kind,
                    ModelContextColdKind::Checkpoint | ModelContextColdKind::DormantRestore
                )
            }) && (!operations.is_empty() || !resident_reads.is_empty() || !sources.is_empty())
            {
                return Err(publication_input_error(
                    "context material work does not admit numerical producers",
                ));
            }
            if storage.capacity_limit.is_some_and(|limit| capacity > limit) {
                return Err(publication_input_error(
                    "cold plan exceeds its original explicit event limit",
                ));
            }
            let mut plan =
                ModelWorkPlan::new(capacity, operations, region_ends, storage.regions.len())
                    .map_err(publication_input_error)?;
            for (index, state) in storage.regions.iter().enumerate() {
                if *state == RecordingState::NotEntered {
                    plan.skip_unentered_region(index)
                        .map_err(publication_input_error)?;
                }
            }
            let quantities = plan.quantities();
            let allowance = Arc::clone(&storage.allowance);
            if resident_reads.iter().any(|(ordinal, _)| {
                if storage.regions.is_empty() {
                    *ordinal != 0
                } else {
                    *ordinal >= storage.regions.len()
                }
            }) {
                return Err(publication_input_error(
                    "resident read declaration changed its original report region roster",
                ));
            }
            let resident_content =
                native_work_bound::FrozenColdResidentReads::new(handle, resident_reads)?;
            let source_content = if storage.purpose == SemanticColdModelWorkPurpose::PrivatePrefix {
                let target_roster = if let Some(build) =
                    self.prepared_segment.as_ref().filter(|build| {
                        build
                            .program_admission
                            .as_ref()
                            .and_then(|admission| self.instruction_admission(admission).ok())
                            .is_some_and(|original| {
                                storage.admission.matches_instruction(&original.instruction)
                            })
                    }) {
                    Some(build.tokens.iter().zip(&build.transitions)
                        .filter(|(_, kind)| **kind == SemanticTransitionKind::Update)
                        .map(|(&token, _)| {
                            let view = self.steps[&token].prepared.as_ref().and_then(|prepared| prepared.training_view.as_ref())
                                .ok_or_else(|| publication_input_error("SOURCE union requires its original target training views before admission"))?;
                            Ok((token, view.edit_group_members().to_vec()))
                        }).collect::<Result<Vec<_>, SemanticTransitionError>>()?)
                } else {
                    None
                };
                if !sources.is_empty() && target_roster.is_none() {
                    return Err(publication_input_error(
                        "SOURCE union lost its original target instruction scope",
                    ));
                }
                Some(native_work_bound::FrozenColdSourceContent::new(
                    &allowance,
                    storage.admission.instruction()?,
                    sources,
                    target_roster.as_deref(),
                )?)
            } else if !sources.is_empty() {
                return Err(publication_input_error(
                    "SOURCE preparation belongs only to its original Prefix admission",
                ));
            } else {
                None
            };
            let mut allowance_guard = allowance.lock().map_err(|_| {
                publication_input_error("original cold native allowance is poisoned")
            })?;
            if allowance_guard.is_evaluation() != evaluation_content.is_some() {
                return Err(publication_input_error(
                    "the native evaluation purpose requires its original output and objective content",
                ));
            }
            if let Some(content) = evaluation_content {
                if !Arc::ptr_eq(&allowance, &content.allowance) {
                    return Err(publication_input_error(
                        "evaluation content belongs to another original cold report",
                    ));
                }
                allowance_guard.freeze_content(content.content)?;
            }
            if let Some(source_content) = source_content {
                allowance_guard.freeze_source_content(source_content)?;
            }
            let scratch =
                native_work_bound::model_scratch_native_work_ceiling(operations.len(), true)?;
            if storage.model_context.is_some()
                || (storage.purpose == SemanticColdModelWorkPurpose::PrivatePrefix
                    && allowance_guard.is_model_context_report())
            {
                allowance_guard.freeze_model_context_report_dma(operations.len())?;
            }
            allowance_guard.freeze_model_scratch(scratch)?;
            allowance_guard.freeze_resident_reads(resident_content)?;
            for (ordinal, state) in storage.regions.iter().enumerate() {
                if *state == RecordingState::NotEntered {
                    allowance_guard.skip_unentered_resident_reads(ordinal)?;
                }
            }
            // Validate the whole native-derived union before any original
            // scratch initialization or numerical/control recorder can enter.
            let native =
                native_work_bound::native_work_ceiling_units(allowance_guard.content_ceiling()?)?;
            allowance_guard.require_model_context_source_budget(quantities)?;
            if let Some(context) = &storage.model_context {
                let prior = context.prior;
                let prior_work = prior.map_or(Ok(0), |prior| {
                    prior
                        .model_work
                        .checked_add(prior.native_work)
                        .ok_or(SemanticTransitionError::GenerationExhausted)
                })?;
                let combined = prior_work
                    .checked_add(quantities[0])
                    .and_then(|work| work.checked_add(native))
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                let calls = quantities[2]
                    .checked_add(prior.map_or(0, |prior| prior.model_calls))
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                if combined > context.budget[0] || calls > context.budget[2] {
                    return Err(publication_input_error(
                        "model context cold preparation exceeds its original instruction budget",
                    ));
                }
            }
            drop(allowance_guard);
            self.steps
                .get_mut(&handle.token)
                .expect("checked reader")
                .cold_model_work
                .as_mut()
                .expect("checked cold work")
                .plan = Some(plan);
            retained_plan = true;
            if initialize {
                self.continue_cold_model_work_initialization(handle)?;
            }
            Ok(quantities)
        })();
        if result.is_err() && !retained_plan {
            self.fail_cold_model_work(handle);
        }
        result
    }

    /// Read the retained complete report plan before admitting dependent work.
    /// Native-authorized skipped regions never change these upper quantities.
    pub fn cold_model_work_plan_quantities(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<[u64; 3], SemanticTransitionError> {
        Ok(self.cold_model_work(handle)?.plan()?.quantities())
    }

    /// The same sealed extent, without issuing physical allocation provenance.
    pub fn cold_model_work_buffer_plan_layout(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<Option<SemanticTensorLayout>, SemanticTransitionError> {
        let storage = self.original_cold_model_work(handle)?;
        storage
            .plan
            .as_ref()
            .map(|plan| {
                canonical_tensor_layout(SemanticTensorLayout {
                    role: 0,
                    index: 0,
                    scalar_type: 3,
                    element_bytes: size_of::<u64>() as u64,
                    rank: 2,
                    dimensions: [plan.quantities()[1], 3, 0, 0],
                    logical_axis: u64::MAX,
                    strides_bytes: [0; 4],
                })
            })
            .transpose()
    }

    /// Join/materialize only the already retained original declaration.
    /// This cannot admit a replacement plan or repeat a model callback.
    pub fn resume_cold_model_work_plan(
        &mut self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.original_cold_model_work(handle)?;
        if storage.plan.is_none()
            || !matches!(
                storage.state,
                RecordingState::Waiting | RecordingState::Recording
            )
        {
            return Err(publication_input_error(
                "cold materialization requires its same retained original plan",
            ));
        }
        self.continue_cold_model_work_initialization(handle)
    }

    /// Original finite content subtotal only, not the whole native lifecycle.
    pub fn cold_model_work_content_ceiling(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<[u64; 9], SemanticTransitionError> {
        self.cold_model_work(handle)?
            .allowance
            .lock()
            .map_err(|_| publication_input_error("original cold native allowance is poisoned"))?
            .content_ceiling()
    }

    pub fn cold_model_work_buffer(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.cold_model_work_buffer_for_region(lease, handle, stream, None)
    }

    pub fn model_context_material_restore_buffer_port(
        &self,
        target: &Self,
        context: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
        stream: u64,
        region: &SemanticColdModelWorkRegion,
    ) -> Result<SemanticModelWorkBufferPort, SemanticTransitionError> {
        self.require_model_context_material_restore_callback_work(
            target, context, parent, work, region,
        )?;
        self.issue_cold_model_work_buffer_port(
            ColdWorkReader::ModelContextMaterial(context, target, parent),
            work,
            stream,
            Some(region),
        )
    }

    pub fn export_model_context_material_restore_buffer_port(
        &mut self,
        target: &Self,
        context: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        port: &SemanticModelWorkBufferPort,
        work: &SemanticColdModelWork,
        region: &SemanticColdModelWorkRegion,
    ) -> Result<SemanticModelWorkBufferExport, SemanticTransitionError> {
        self.require_model_context_material_restore_counter_read(
            target, context, parent, work, region,
        )?;
        self.export_cold_model_work_buffer_port_for_reader(
            ColdWorkReader::ModelContextMaterial(context, target, parent),
            port,
            work,
            Some(region),
        )
    }

    pub fn model_context_material_restore_buffer_port_finalized_extent(
        &self,
        target: &Self,
        context: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        port: &SemanticModelWorkBufferPort,
        work: &SemanticColdModelWork,
        region: &SemanticColdModelWorkRegion,
    ) -> Result<usize, SemanticTransitionError> {
        self.require_model_context_material_restore_counter_read(
            target, context, parent, work, region,
        )?;
        self.cold_model_work_buffer_port_finalized_extent_for_reader(
            ColdWorkReader::ModelContextMaterial(context, target, parent),
            port,
            work,
            Some(region),
        )
    }

    pub fn model_context_material_restore_buffer(
        &mut self,
        target: &Self,
        context: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
        stream: u64,
        region: &SemanticColdModelWorkRegion,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.require_model_context_material_restore_counter_read(
            target, context, parent, work, region,
        )?;
        self.cold_model_work_buffer_for_reader(
            ColdWorkReader::ModelContextMaterial(context, target, parent),
            work,
            stream,
            Some(region),
        )
    }

    pub fn cold_model_work_buffer_port(
        &self,
        lease: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<SemanticModelWorkBufferPort, SemanticTransitionError> {
        self.issue_cold_model_work_buffer_port(
            ColdWorkReader::Published(lease),
            handle,
            stream,
            region,
        )
    }

    pub fn prepared_cold_model_work_buffer_port(
        &self,
        step: &SemanticPreparedStep,
        parent: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<SemanticModelWorkBufferPort, SemanticTransitionError> {
        self.issue_cold_model_work_buffer_port(
            ColdWorkReader::Prepared(step, parent),
            handle,
            stream,
            region,
        )
    }

    pub fn admitted_retirement_cold_model_work_buffer_port(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        handle: &SemanticColdModelWork,
        stream: u64,
    ) -> Result<SemanticModelWorkBufferPort, SemanticTransitionError> {
        self.issue_cold_model_work_buffer_port(
            ColdWorkReader::AdmittedRetirement(admission),
            handle,
            stream,
            None,
        )
    }

    fn issue_cold_model_work_buffer_port(
        &self,
        reader: ColdWorkReader<'_>,
        handle: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<SemanticModelWorkBufferPort, SemanticTransitionError> {
        self.cold_model_work_buffer_scope(reader, handle, stream, region)?;
        Ok(SemanticModelWorkBufferPort::for_cold(
            handle.clone(),
            region.cloned(),
            self.provider.device().ordinal(),
            stream,
        ))
    }

    pub fn export_cold_model_work_buffer_port(
        &mut self,
        lease: &SemanticPublishedLease,
        port: &SemanticModelWorkBufferPort,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<SemanticModelWorkBufferExport, SemanticTransitionError> {
        self.export_cold_model_work_buffer_port_for_reader(
            ColdWorkReader::Published(lease),
            port,
            handle,
            region,
        )
    }

    pub fn export_prepared_cold_model_work_buffer_port(
        &mut self,
        step: &SemanticPreparedStep,
        parent: &SemanticPublishedLease,
        port: &SemanticModelWorkBufferPort,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<SemanticModelWorkBufferExport, SemanticTransitionError> {
        self.export_cold_model_work_buffer_port_for_reader(
            ColdWorkReader::Prepared(step, parent),
            port,
            handle,
            region,
        )
    }

    pub fn export_admitted_retirement_cold_model_work_buffer_port(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        port: &SemanticModelWorkBufferPort,
        handle: &SemanticColdModelWork,
    ) -> Result<SemanticModelWorkBufferExport, SemanticTransitionError> {
        self.export_cold_model_work_buffer_port_for_reader(
            ColdWorkReader::AdmittedRetirement(admission),
            port,
            handle,
            None,
        )
    }

    fn export_cold_model_work_buffer_port_for_reader(
        &mut self,
        reader: ColdWorkReader<'_>,
        port: &SemanticModelWorkBufferPort,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<SemanticModelWorkBufferExport, SemanticTransitionError> {
        let (stream, device) =
            self.require_cold_model_work_buffer_port(reader, port, handle, region)?;
        if !(matches!(reader, ColdWorkReader::ModelContextMaterial(_, _, _))
            && self.original_cold_model_work(handle)?.state == RecordingState::Closed)
        {
            self.resume_cold_model_work_plan(handle)?;
        }
        let (_, allocation, layout) =
            self.cold_model_work_buffer_metadata(reader, handle, stream, region)?;
        let tensor = self.cold_model_work_buffer_for_reader(reader, handle, stream, region)?;
        SemanticModelWorkBufferExport::new(allocation, layout, tensor, device, stream)
    }

    fn require_cold_model_work_buffer_port(
        &self,
        reader: ColdWorkReader<'_>,
        port: &SemanticModelWorkBufferPort,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<(u64, usize), SemanticTransitionError> {
        let (issued, issued_region, stream, device) = port.cold_binding()?;
        if device != self.provider.device().ordinal() || stream != self.stream.cu_stream() as u64 {
            return Err(publication_input_error(
                "cold counter port changed its original device or stream",
            ));
        }
        if !issued.same_invocation(handle) {
            self.require_cold_model_work_handoff(issued, handle, issued_region, region)?;
        } else if match (issued_region, region) {
            (Some(issued), Some(actual)) => {
                issued.index != actual.index || !actual.work.same_invocation(handle)
            }
            (None, None) => false,
            _ => true,
        } {
            return Err(publication_input_error(
                "cold counter export changed its original region binding",
            ));
        }
        self.check_cold_work_completion_reader(reader, handle)?;
        Ok((stream, device))
    }

    /// Exact admitted extent, including zero, without constructing an export.
    pub fn cold_model_work_buffer_port_finalized_extent(
        &self,
        lease: &SemanticPublishedLease,
        port: &SemanticModelWorkBufferPort,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<usize, SemanticTransitionError> {
        self.cold_model_work_buffer_port_finalized_extent_for_reader(
            ColdWorkReader::Published(lease),
            port,
            handle,
            region,
        )
    }

    pub fn prepared_cold_model_work_buffer_port_finalized_extent(
        &self,
        step: &SemanticPreparedStep,
        parent: &SemanticPublishedLease,
        port: &SemanticModelWorkBufferPort,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<usize, SemanticTransitionError> {
        self.cold_model_work_buffer_port_finalized_extent_for_reader(
            ColdWorkReader::Prepared(step, parent),
            port,
            handle,
            region,
        )
    }

    pub fn admitted_retirement_cold_model_work_buffer_port_finalized_extent(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        port: &SemanticModelWorkBufferPort,
        handle: &SemanticColdModelWork,
    ) -> Result<usize, SemanticTransitionError> {
        self.cold_model_work_buffer_port_finalized_extent_for_reader(
            ColdWorkReader::AdmittedRetirement(admission),
            port,
            handle,
            None,
        )
    }

    fn cold_model_work_buffer_port_finalized_extent_for_reader(
        &self,
        reader: ColdWorkReader<'_>,
        port: &SemanticModelWorkBufferPort,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<usize, SemanticTransitionError> {
        let (stream, _) = self.require_cold_model_work_buffer_port(reader, port, handle, region)?;
        self.cold_model_work_buffer_scope(reader, handle, stream, region)?;
        let storage = self.cold_model_work(handle)?;
        if !storage.initialized {
            return Err(publication_input_error(
                "cold finalized extent requires its original known initialization",
            ));
        }
        usize::try_from(storage.plan()?.quantities()[1])
            .map_err(|_| SemanticTransitionError::GenerationExhausted)
    }

    /// Project original scratch geometry without exporting or admitting a plan.
    pub fn cold_model_work_buffer_layout(
        &self,
        lease: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<(DeviceAllocationProvenance, SemanticTensorLayout), SemanticTransitionError> {
        let (_, allocation, layout) = self.cold_model_work_buffer_metadata(
            ColdWorkReader::Published(lease),
            handle,
            stream,
            region,
        )?;
        Ok((allocation, layout))
    }

    pub fn prepared_cold_model_work_buffer_layout(
        &self,
        step: &SemanticPreparedStep,
        parent: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<(DeviceAllocationProvenance, SemanticTensorLayout), SemanticTransitionError> {
        let (_, allocation, layout) = self.cold_model_work_buffer_metadata(
            ColdWorkReader::Prepared(step, parent),
            handle,
            stream,
            region,
        )?;
        Ok((allocation, layout))
    }

    pub fn admitted_retirement_cold_model_work_buffer_layout(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        handle: &SemanticColdModelWork,
        stream: u64,
    ) -> Result<(DeviceAllocationProvenance, SemanticTensorLayout), SemanticTransitionError> {
        let (_, allocation, layout) = self.cold_model_work_buffer_metadata(
            ColdWorkReader::AdmittedRetirement(admission),
            handle,
            stream,
            None,
        )?;
        Ok((allocation, layout))
    }

    pub fn cold_model_work_region_buffer(
        &mut self,
        lease: &SemanticPublishedLease,
        region: &SemanticColdModelWorkRegion,
        stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.cold_model_work_buffer_for_region(lease, &region.work, stream, Some(region))
    }

    fn cold_model_work_buffer_for_region(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.cold_model_work_buffer_for_reader(
            ColdWorkReader::Published(lease),
            handle,
            stream,
            region,
        )
    }

    pub fn prepared_cold_model_work_buffer(
        &mut self,
        step: &SemanticPreparedStep,
        parent: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.cold_model_work_buffer_for_reader(
            ColdWorkReader::Prepared(step, parent),
            handle,
            stream,
            region,
        )
    }

    fn cold_model_work_buffer_metadata(
        &self,
        reader: ColdWorkReader<'_>,
        handle: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<(u64, DeviceAllocationProvenance, SemanticTensorLayout), SemanticTransitionError>
    {
        let token = self.cold_model_work_buffer_scope(reader, handle, stream, region)?;
        let storage = self.cold_model_work(handle)?;
        if !storage.initialized {
            return Err(publication_input_error(
                "cold physical scratch requires its original completed initialization",
            ));
        }
        let words = storage.work()?.actual.len();
        if words == 0 || words % 3 != 0 {
            return Err(publication_input_error(
                "original cold work buffer has invalid slot geometry",
            ));
        }
        let layout = self
            .cold_model_work_buffer_plan_layout(handle)?
            .ok_or_else(|| {
                publication_input_error("cold physical scratch lost its sealed original extent")
            })?;
        if layout.dimensions[0] != (words / 3) as u64 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let allocation = storage
            .work()?
            .actual
            .view()
            .allocation_provenance()
            .ok_or_else(|| {
                publication_input_error("original cold work buffer lost its native allocation")
            })?;
        Ok((token, allocation, layout))
    }

    fn cold_model_work_buffer_scope(
        &self,
        reader: ColdWorkReader<'_>,
        handle: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<u64, SemanticTransitionError> {
        let token = self.check_cold_work_reader(reader, handle)?;
        let storage = self.cold_model_work(handle)?;
        let available = if let Some(region) = region {
            if !region.work.same_invocation(handle) {
                return Err(publication_input_error(
                    "cold model scratch region belongs to another original invocation",
                ));
            }
            if matches!(reader, ColdWorkReader::ModelContextMaterial(_, _, _))
                && storage
                    .model_context
                    .as_ref()
                    .is_some_and(|original| original.kind == ModelContextColdKind::DormantRestore)
            {
                let state = *storage
                    .regions
                    .get(region.index)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if state == RecordingState::Closed {
                    matches!(
                        storage.state,
                        RecordingState::Recording | RecordingState::Closed
                    )
                } else {
                    self.require_cold_model_work_region_metadata(region, state, true)?;
                    matches!(state, RecordingState::Waiting | RecordingState::Recording)
                }
            } else {
                self.require_cold_model_work_region(region, RecordingState::Waiting)?;
                storage.state == RecordingState::Recording
            }
        } else {
            storage.regions.is_empty() && storage.state == RecordingState::Waiting
        };
        if stream != self.stream.cu_stream() as u64 || !available {
            return Err(publication_input_error(
                "cold model scratch requires its original reader and stream before recording",
            ));
        }
        Ok(token)
    }

    fn cold_model_work_buffer_for_reader(
        &mut self,
        reader: ColdWorkReader<'_>,
        handle: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        let (token, _, layout) =
            self.cold_model_work_buffer_metadata(reader, handle, stream, region)?;
        let storage = self.cold_model_work(handle)?;
        storage.plan()?;
        // SAFETY: reset initialized all three words of every original slot.
        let view = unsafe { storage.work()?.actual.view().cast::<u8>() }
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        // This is the Session's own stream, already joined by cold completion,
        // not an additional external consumer omitted from the frozen roster.
        let scratch = Arc::clone(&storage.aliases);
        let guard = Arc::clone(&self.steps[&token].aliases);
        self.export_owned_view(
            view,
            layout.dimensions[..2].iter().map(|&n| n as i64).collect(),
            layout.strides_bytes[..2]
                .iter()
                .map(|&n| (n / layout.element_bytes) as i64)
                .collect(),
            (1, 64),
            guard,
            stream,
            Some(scratch),
        )
    }

    /// Freeze the finite region roster before the original report begins.
    pub fn prepare_cold_model_work_regions(
        &mut self,
        handle: &SemanticColdModelWork,
        count: usize,
    ) -> Result<Vec<SemanticColdModelWorkRegion>, SemanticTransitionError> {
        let storage = self.cold_model_work(handle)?;
        if count == 0
            || storage.state != RecordingState::Waiting
            || !storage.regions.is_empty()
            || storage.plan.is_some()
        {
            return Err(publication_input_error(
                "cold registration regions require their sole original waiting report",
            ));
        }
        self.steps
            .get_mut(&handle.token)
            .expect("checked original reader")
            .cold_model_work
            .as_mut()
            .expect("checked original work")
            .regions = vec![RecordingState::Waiting; count];
        Ok((0..count)
            .map(|index| SemanticColdModelWorkRegion {
                work: handle.clone(),
                index,
            })
            .collect())
    }

    fn require_cold_model_work_region(
        &self,
        region: &SemanticColdModelWorkRegion,
        expected: RecordingState,
    ) -> Result<(), SemanticTransitionError> {
        self.require_cold_model_work_region_metadata(region, expected, false)
    }

    fn require_cold_model_work_region_metadata(
        &self,
        region: &SemanticColdModelWorkRegion,
        expected: RecordingState,
        declaration: bool,
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.cold_model_work(&region.work)?;
        if !(storage.state == RecordingState::Recording
            || (declaration && storage.state == RecordingState::Waiting))
            || storage.regions.get(region.index) != Some(&expected)
            || storage.regions[..region.index]
                .iter()
                .any(|state| !matches!(state, RecordingState::Closed | RecordingState::NotEntered))
        {
            return Err(publication_input_error(
                "cold registration region changed its original report, order or one-shot state",
            ));
        }
        Ok(())
    }

    pub fn require_closed_cold_model_work_region(
        &self,
        region: &SemanticColdModelWorkRegion,
    ) -> Result<(), SemanticTransitionError> {
        self.require_cold_model_work_region(region, RecordingState::Closed)
    }

    pub fn begin_cold_model_work_region(
        &mut self,
        region: &SemanticColdModelWorkRegion,
    ) -> Result<(), SemanticTransitionError> {
        if self.cold_model_work(&region.work)?.stopped_before_entry {
            return Err(publication_input_error(
                "known non-entry closed its original recording prefix",
            ));
        }
        self.require_cold_model_work_region(region, RecordingState::Waiting)?;
        self.cold_model_work(&region.work)?
            .plan()?
            .require_region_start(region.index)
            .map_err(publication_input_error)?;
        self.steps
            .get_mut(&region.work.token)
            .expect("checked original reader")
            .cold_model_work
            .as_mut()
            .expect("checked original work")
            .regions[region.index] = RecordingState::Recording;
        Ok(())
    }

    /// Retain the original result owner before its evaluation can be launched.
    /// The region remains part of the one original report and admitted plan.
    pub fn bind_cold_model_work_evaluation_cleanup(
        &mut self,
        region: &SemanticColdModelWorkRegion,
        original: &SemanticModelEvaluation,
        evaluation_owner: Option<&SemanticTransitionSession>,
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.cold_model_work(&region.work)?;
        if storage.state != RecordingState::Recording
            || storage.regions.get(region.index) != Some(&RecordingState::Waiting)
            || storage.evaluation_cleanup.is_some()
        {
            return Err(publication_input_error(
                "evaluation cleanup requires its original waiting region and sole result owner",
            ));
        }
        let work = storage.native_work.view();
        match evaluation_owner {
            Some(owner) => owner.require_evaluation_cold_work_origin(original, None, &work)?,
            None => {
                self.require_evaluation_cold_work_origin(original, Some(region.work.token), &work)?
            }
        }
        let storage = self
            .steps
            .get_mut(&region.work.token)
            .expect("checked original reader")
            .cold_model_work
            .as_mut()
            .expect("checked original work");
        storage.evaluation_cleanup = Some(EvaluationCleanup {
            region: region.index,
            original: original.clone(),
            cancellation: None,
        });
        Ok(())
    }

    /// Only the original native non-submission proof can make its result copies
    /// unreachable. Native cleanup still enters and closes the existing region.
    pub fn cancel_cold_model_work_evaluation_cleanup(
        &mut self,
        region: &SemanticColdModelWorkRegion,
        cancelled: &SemanticCancelledModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        self.require_cold_model_work_region(region, RecordingState::Waiting)?;
        let storage = self.cold_model_work(&region.work)?;
        let cleanup = storage.evaluation_cleanup.as_ref().ok_or_else(|| {
            publication_input_error("cancelled cleanup lost its original evaluation result owner")
        })?;
        if cleanup.region != region.index
            || cleanup.cancellation.is_some()
            || !cancelled.belongs_to_original(&cleanup.original)
        {
            return Err(publication_input_error(
                "cleanup cancellation changed its original native evaluation proof",
            ));
        }
        storage
            .plan()?
            .require_evaluation_result_copy_region(region.index)
            .map_err(publication_input_error)?;
        self.steps
            .get_mut(&region.work.token)
            .expect("checked original reader")
            .cold_model_work
            .as_mut()
            .expect("checked original work")
            .evaluation_cleanup
            .as_mut()
            .expect("checked original evaluation")
            .cancellation = Some(cancelled.clone());
        Ok(())
    }

    pub fn close_cold_model_work_region(
        &mut self,
        region: &SemanticColdModelWorkRegion,
    ) -> Result<(), SemanticTransitionError> {
        self.require_cold_model_work_region(region, RecordingState::Recording)?;
        if self
            .cold_model_work(&region.work)?
            .pending_operation
            .is_some()
        {
            return Err(publication_input_error(
                "an unresolved original operation keeps its region open",
            ));
        }
        let cancelled = self
            .cold_model_work(&region.work)?
            .evaluation_cleanup
            .as_ref()
            .is_some_and(|cleanup| {
                cleanup.region == region.index && cleanup.cancellation.is_some()
            });
        let storage = self
            .steps
            .get_mut(&region.work.token)
            .expect("checked original reader")
            .cold_model_work
            .as_mut()
            .expect("checked original work");
        if cancelled {
            storage
                .plan()?
                .require_evaluation_result_copy_region(region.index)
                .map_err(publication_input_error)?;
            storage
                .plan_mut()?
                .skip_unentered_region(region.index)
                .map_err(publication_input_error)?;
        } else if !(storage.stopped_before_entry
            && storage.operations.last().is_some_and(|attempt| {
                attempt.region == Some(region.index) && attempt.state == OperationState::NotEntered
            }))
        {
            storage
                .plan()?
                .require_region_end(region.index)
                .map_err(publication_input_error)?;
        }
        storage.regions[region.index] = RecordingState::Closed;
        Ok(())
    }

    pub fn begin_cold_model_work(
        &mut self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.cold_model_work(handle)?;
        if storage.state != RecordingState::Waiting {
            return Err(publication_input_error(
                "original cold recorder begins exactly once",
            ));
        }
        // Enclosing report visibility may precede its sole declaration, but
        // no region/operation or numerical authority is granted before it.
        if storage.regions.is_empty() {
            storage.plan()?;
        }
        if storage.plan.is_some() && !storage.initialized {
            return Err(publication_input_error(
                "cold recording requires its original completed plan initialization",
            ));
        }
        let storage = self
            .steps
            .get_mut(&handle.token)
            .expect("checked reader")
            .cold_model_work
            .as_mut()
            .expect("checked cold work");
        storage.state = RecordingState::Recording;
        storage
            .allowance
            .lock()
            .map_err(|_| publication_input_error("context prefill allowance is poisoned"))?
            .set_model_context_prefill_recording(storage.initialized && storage.plan.is_some());
        if let Some(work) = &mut storage.work {
            work.uncaptured_recording = true;
        }
        Ok(())
    }

    /// The phase calls this only after known numerical completion, native
    /// cancellation or complete private retirement has joined consumers. The
    /// original report and physical interval remain open; entered, failed or
    /// unknown regions stay unchanged, including at the report's full extent.
    pub fn cancel_unentered_cold_model_work_regions(
        &mut self,
        handle: &SemanticColdModelWork,
        before: usize,
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.cold_model_work(handle)?;
        if storage.state != RecordingState::Recording
            || storage.pending_operation.is_some()
            || before > storage.regions.len()
            || storage.regions[..before].iter().any(|state| {
                !matches!(
                    state,
                    RecordingState::Waiting | RecordingState::Closed | RecordingState::NotEntered
                )
            })
        {
            return Err(publication_input_error(
                "known cancellation cannot close an entered, failed or unknown cold region",
            ));
        }
        let storage = self
            .steps
            .get_mut(&handle.token)
            .expect("checked original reader")
            .cold_model_work
            .as_mut()
            .expect("checked original work");
        for (index, state) in storage.regions[..before].iter_mut().enumerate() {
            if *state == RecordingState::Waiting {
                if let Some(plan) = &mut storage.plan {
                    plan.skip_unentered_region(index)
                        .map_err(publication_input_error)?;
                }
                storage
                    .allowance
                    .lock()
                    .map_err(|_| {
                        publication_input_error("original resident read allowance is poisoned")
                    })?
                    .skip_unentered_resident_reads(index)?;
                *state = RecordingState::NotEntered;
            }
        }
        Ok(())
    }

    pub fn close_cold_model_work(
        &mut self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.cold_model_work(handle)?;
        if storage.state != RecordingState::Recording
            || storage.pending_operation.is_some()
            || storage
                .regions
                .iter()
                .any(|state| !matches!(state, RecordingState::Closed | RecordingState::NotEntered))
        {
            return Err(publication_input_error(
                "cold recorder closes its sole original recording",
            ));
        }
        let storage = self
            .steps
            .get_mut(&handle.token)
            .expect("checked reader")
            .cold_model_work
            .as_mut()
            .expect("checked cold work");
        if let Some(work) = &mut storage.work {
            work.uncaptured_recording = false;
        }
        storage.state = RecordingState::Closed;
        storage
            .allowance
            .lock()
            .map_err(|_| publication_input_error("context prefill allowance is poisoned"))?
            .set_model_context_prefill_recording(false);
        Ok(())
    }

    /// Failure never clears allocations, resets capacity or opens another recorder.
    pub fn fail_cold_model_work(&mut self, handle: &SemanticColdModelWork) {
        if let Some(storage) = self
            .steps
            .get_mut(&handle.token)
            .and_then(|step| step.cold_model_work.as_mut())
            .filter(|storage| Arc::ptr_eq(&storage.invocation, &handle.invocation))
        {
            storage.state = RecordingState::Failed;
            if let Ok(mut allowance) = storage.allowance.lock() {
                allowance.set_model_context_prefill_recording(false);
            }
            if let Some(work) = &mut storage.work {
                work.uncaptured_recording = false;
            }
        }
    }

    /// Authenticate the next immutable occurrence without registering expenditure.
    pub fn prepare_cold_model_work_operation(
        &mut self,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
        kind: ModelWorkKind,
        dimensions: &[u64],
        device_produced: bool,
    ) -> Result<SemanticColdModelWorkOperation, SemanticTransitionError> {
        let mut operations = self.prepare_cold_model_work_operations(
            handle,
            region,
            &[(kind, dimensions.to_vec(), device_produced)],
        )?;
        Ok(operations.remove(0))
    }

    /// Reserve one original helper or nested operator's exact plan prefix.
    /// Each issued occurrence still needs its own entry and actual completion.
    pub fn prepare_cold_model_work_operations(
        &mut self,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
        operations: &[(ModelWorkKind, Vec<u64>, bool)],
    ) -> Result<Vec<SemanticColdModelWorkOperation>, SemanticTransitionError> {
        if let Some(region) = region {
            if !region.work.same_invocation(handle) {
                return Err(publication_input_error(
                    "operation changed its original region owner",
                ));
            }
            self.require_cold_model_work_region(region, RecordingState::Recording)?;
        }
        let storage = self.cold_model_work(handle)?;
        if storage.state != RecordingState::Recording
            || storage.pending_operation.is_some()
            || storage.stopped_before_entry
            || (region.is_none() && !storage.regions.is_empty())
            || storage.evaluation_cleanup.as_ref().is_some_and(|cleanup| {
                region.is_some_and(|region| region.index == cleanup.region)
                    && cleanup.cancellation.is_some()
            })
        {
            return Err(publication_input_error(
                "operation requires its original active recorder without an unresolved attempt",
            ));
        }
        let first_slot = storage
            .work()?
            .next_slot()
            .map_err(publication_input_error)?;
        let capacity = storage.work()?.actual.len() / 3;
        if operations.is_empty()
            || first_slot
                .checked_add(operations.len())
                .is_none_or(|end| end > capacity)
            || storage
                .operations
                .len()
                .checked_add(operations.len())
                .is_none_or(|end| end > capacity)
        {
            return Err(publication_input_error(
                "operation exceeds its original finite event capacity",
            ));
        }
        let mut attempts = Vec::new();
        let mut issued = Vec::new();
        attempts
            .try_reserve_exact(operations.len())
            .map_err(|error| {
                runtime_error("original operation group custody reservation", error)
            })?;
        issued
            .try_reserve_exact(operations.len())
            .map_err(|error| runtime_error("original operation group handle reservation", error))?;
        for (offset, (kind, dimensions, device_produced)) in operations.iter().enumerate() {
            storage
                .plan()?
                .require_next_at(offset, *kind, dimensions, region.map(|region| region.index))
                .map_err(publication_input_error)?;
            let slot = first_slot + offset;
            let event = if *device_produced {
                let address = storage
                    .work()?
                    .actual
                    .device_ptr_value()
                    .checked_add((slot * 3 * size_of::<u64>()) as u64)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                ModelWorkEvent::device_operation(*kind, dimensions, address)
            } else {
                ModelWorkEvent::operation(*kind, dimensions)
            }
            .map_err(publication_input_error)?;
            let issuance = Arc::new(());
            issued.push(SemanticColdModelWorkOperation {
                work: handle.clone(),
                index: storage.operations.len() + offset,
                issuance: Arc::clone(&issuance),
            });
            attempts.push(ColdOperationAttempt {
                issuance,
                region: region.map(|region| region.index),
                event,
                slot,
                state: OperationState::Prepared,
            });
        }
        let storage = self
            .steps
            .get_mut(&handle.token)
            .expect("checked original step")
            .cold_model_work
            .as_mut()
            .expect("checked original recorder");
        storage
            .operations
            .try_reserve(operations.len())
            .map_err(|error| runtime_error("original operation custody reservation", error))?;
        let index = storage.operations.len();
        storage.operations.extend(attempts);
        storage.pending_operation = Some(index);
        storage.pending_operation_end = storage.operations.len();
        // Instrumentation is prepared before the real effect. Never reset this
        // slot after the actual producer may have written its reached usage.
        if operations
            .iter()
            .any(|(_, _, device_produced)| *device_produced)
        {
            let native_work = storage.native_work.view();
            if let Err(error) = storage.work_mut()?.reset_slots(
                &self.domain,
                &mut self.poisoned,
                first_slot,
                operations.len(),
                Some(&native_work),
            ) {
                for attempt in &mut storage.operations[index..] {
                    attempt.state = OperationState::Unknown;
                }
                return Err(error);
            }
        }
        Ok(issued)
    }

    fn original_cold_operation(
        &self,
        operation: &SemanticColdModelWorkOperation,
    ) -> Result<&ColdOperationAttempt, SemanticTransitionError> {
        self.cold_model_work(&operation.work)?
            .operations
            .get(operation.index)
            .filter(|attempt| Arc::ptr_eq(&attempt.issuance, &operation.issuance))
            .ok_or_else(|| {
                publication_input_error("operation is not its original native issued attempt")
            })
    }

    pub fn cold_model_work_operation_slot(
        &self,
        operation: &SemanticColdModelWorkOperation,
    ) -> Result<usize, SemanticTransitionError> {
        Ok(self.original_cold_operation(operation)?.slot)
    }

    /// Cross the effect boundary once; this is not an actual-work receipt.
    pub fn enter_cold_model_work_operation(
        &mut self,
        operation: &SemanticColdModelWorkOperation,
    ) -> Result<usize, SemanticTransitionError> {
        let attempt = self.original_cold_operation(operation)?;
        let storage = self.cold_model_work(&operation.work)?;
        if attempt.state != OperationState::Prepared
            || storage.state != RecordingState::Recording
            || storage.pending_operation.is_none_or(|start| {
                operation.index < start
                    || operation.index >= storage.pending_operation_end
                    || storage.operations[start..storage.pending_operation_end]
                        .iter()
                        .any(|attempt| {
                            matches!(
                                attempt.state,
                                OperationState::Unknown | OperationState::Completing
                            )
                        })
            })
            || attempt.region.is_some_and(|region| {
                storage.regions.get(region) != Some(&RecordingState::Recording)
            })
        {
            return Err(publication_input_error(
                "original operation entry is single-use",
            ));
        }
        let slot = attempt.slot;
        self.steps
            .get_mut(&operation.work.token)
            .expect("checked original step")
            .cold_model_work
            .as_mut()
            .expect("checked original recorder")
            .operations[operation.index]
            .state = OperationState::Entered;
        Ok(slot)
    }

    /// Register only the original successfully returned effect, once.
    pub fn complete_cold_model_work_operation(
        &mut self,
        operation: &SemanticColdModelWorkOperation,
    ) -> Result<usize, SemanticTransitionError> {
        let attempt = self.original_cold_operation(operation)?;
        if attempt.state == OperationState::Complete {
            return Ok(attempt.slot);
        }
        let storage = self.cold_model_work(&operation.work)?;
        if attempt.state != OperationState::Entered
            || storage.state != RecordingState::Recording
            || storage.pending_operation != Some(operation.index)
            || storage.operations[operation.index..storage.pending_operation_end]
                .iter()
                .any(|attempt| {
                    matches!(
                        attempt.state,
                        OperationState::Unknown | OperationState::Completing
                    )
                })
            || attempt.region.is_some_and(|region| {
                storage.regions.get(region) != Some(&RecordingState::Recording)
            })
            || storage
                .work()?
                .next_slot()
                .map_err(publication_input_error)?
                != attempt.slot
        {
            return Err(publication_input_error(
                "completion requires its same entered original operation",
            ));
        }
        let event = attempt.event;
        let slot = attempt.slot;
        let storage = self
            .steps
            .get_mut(&operation.work.token)
            .expect("checked original step")
            .cold_model_work
            .as_mut()
            .expect("checked original recorder");
        storage.operations[operation.index].state = OperationState::Completing;
        // The event and slot were validated before entry. This sole registrar
        // does not launch the model, reset its slot, or repeat its callback.
        let recorded = storage
            .work_mut()?
            .record_event(event)
            .map_err(publication_input_error)?;
        debug_assert_eq!(recorded, slot);
        storage.plan_mut()?.consume();
        storage.operations[operation.index].state = OperationState::Complete;
        let next = operation.index + 1;
        storage.pending_operation = (next < storage.pending_operation_end).then_some(next);
        Ok(slot)
    }

    pub fn retain_unknown_cold_model_work_operation(
        &mut self,
        operation: &SemanticColdModelWorkOperation,
    ) -> Result<(), SemanticTransitionError> {
        let attempt = self.original_cold_operation(operation)?;
        if !matches!(
            attempt.state,
            OperationState::Entered | OperationState::Completing | OperationState::Unknown
        ) {
            return Err(publication_input_error(
                "unknown outcome requires its original entered attempt",
            ));
        }
        self.steps
            .get_mut(&operation.work.token)
            .expect("checked original step")
            .cold_model_work
            .as_mut()
            .expect("checked original recorder")
            .operations[operation.index]
            .state = OperationState::Unknown;
        Ok(())
    }

    /// Proven non-entry closes the original prefix, never reopens this event.
    pub fn cancel_unentered_cold_model_work_operation(
        &mut self,
        operation: &SemanticColdModelWorkOperation,
    ) -> Result<(), SemanticTransitionError> {
        let attempt = self.original_cold_operation(operation)?;
        let storage = self.cold_model_work(&operation.work)?;
        if attempt.state != OperationState::Prepared
            || storage.pending_operation.is_none_or(|start| {
                operation.index < start
                    || operation.index >= storage.pending_operation_end
                    || storage.operations[start..storage.pending_operation_end]
                        .iter()
                        .any(|attempt| attempt.state != OperationState::Prepared)
            })
        {
            return Err(publication_input_error(
                "non-entry cannot revoke an entered original operation",
            ));
        }
        let storage = self
            .steps
            .get_mut(&operation.work.token)
            .expect("checked original step")
            .cold_model_work
            .as_mut()
            .expect("checked original recorder");
        let start = storage
            .pending_operation
            .expect("checked pending original group");
        for attempt in &mut storage.operations[start..storage.pending_operation_end] {
            attempt.state = OperationState::NotEntered;
        }
        storage.pending_operation = None;
        storage.stopped_before_entry = true;
        Ok(())
    }

    pub fn record_cold_model_work(
        &mut self,
        handle: &SemanticColdModelWork,
        kind: ModelWorkKind,
        dimensions: &[u64],
        device_produced: bool,
    ) -> Result<usize, SemanticTransitionError> {
        self.record_cold_model_work_in_region(handle, None, kind, dimensions, device_produced)
    }

    pub fn record_cold_model_work_region(
        &mut self,
        region: &SemanticColdModelWorkRegion,
        kind: ModelWorkKind,
        dimensions: &[u64],
        device_produced: bool,
    ) -> Result<usize, SemanticTransitionError> {
        self.record_cold_model_work_in_region(
            &region.work,
            Some(region),
            kind,
            dimensions,
            device_produced,
        )
    }

    fn record_cold_model_work_in_region(
        &mut self,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
        kind: ModelWorkKind,
        dimensions: &[u64],
        device_produced: bool,
    ) -> Result<usize, SemanticTransitionError> {
        let result = (|| {
            if self.cold_model_work(handle)?.pending_operation.is_some()
                || self.cold_model_work(handle)?.stopped_before_entry
            {
                return Err(publication_input_error(
                    "direct registration cannot bypass an original operation attempt",
                ));
            }
            if let Some(region) = region {
                self.require_cold_model_work_region(region, RecordingState::Recording)?;
            } else if !self.cold_model_work(handle)?.regions.is_empty() {
                return Err(publication_input_error(
                    "segmented cold registration requires its actual original region",
                ));
            }
            if self.cold_model_work(handle)?.state != RecordingState::Recording {
                return Err(publication_input_error(
                    "cold model producer requires its active original recorder",
                ));
            }
            if self
                .cold_model_work(handle)?
                .evaluation_cleanup
                .as_ref()
                .is_some_and(|cleanup| {
                    region.is_some_and(|region| region.index == cleanup.region)
                        && cleanup.cancellation.is_some()
                })
            {
                return Err(publication_input_error(
                    "cancelled evaluation cannot execute its unsubmitted result copies",
                ));
            }
            self.cold_model_work(handle)?
                .plan()?
                .require_next(kind, dimensions, region.map(|region| region.index))
                .map_err(publication_input_error)?;
            let storage = self
                .steps
                .get_mut(&handle.token)
                .expect("checked reader")
                .cold_model_work
                .as_mut()
                .expect("checked cold work");
            let native_work = storage.native_work.view();
            let slot = storage.work_mut()?.record_operation(
                kind,
                dimensions,
                device_produced,
                Some(&native_work),
                &self.domain,
                &mut self.poisoned,
            )?;
            storage.plan_mut()?.consume();
            Ok(slot)
        })();
        if result.is_err() {
            self.fail_cold_model_work(handle);
        }
        result
    }

    pub fn record_cold_model_invocation(
        &mut self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.record_cold_model_invocation_in_region(handle, None)
    }

    pub fn record_cold_model_region_invocation(
        &mut self,
        region: &SemanticColdModelWorkRegion,
    ) -> Result<(), SemanticTransitionError> {
        self.record_cold_model_invocation_in_region(&region.work, Some(region))
    }

    fn record_cold_model_invocation_in_region(
        &mut self,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<(), SemanticTransitionError> {
        let result = (|| {
            if self.cold_model_work(handle)?.pending_operation.is_some()
                || self.cold_model_work(handle)?.stopped_before_entry
            {
                return Err(publication_input_error(
                    "model invocation cannot bypass an original operation attempt",
                ));
            }
            if let Some(region) = region {
                self.require_cold_model_work_region(region, RecordingState::Recording)?;
            } else if !self.cold_model_work(handle)?.regions.is_empty() {
                return Err(publication_input_error(
                    "segmented cold invocation requires its actual original region",
                ));
            }
            if self.cold_model_work(handle)?.state != RecordingState::Recording {
                return Err(publication_input_error(
                    "cold model call requires its active original recorder",
                ));
            }
            self.cold_model_work(handle)?
                .plan()?
                .require_next(
                    ModelWorkKind::ModelInvocation,
                    &[],
                    region.map(|region| region.index),
                )
                .map_err(publication_input_error)?;
            let storage = self
                .steps
                .get_mut(&handle.token)
                .expect("checked reader")
                .cold_model_work
                .as_mut()
                .expect("checked cold work");
            let native_work = storage.native_work.view();
            storage.work_mut()?.record_invocation(
                &self.domain,
                &mut self.poisoned,
                Some(&native_work),
            )?;
            storage.plan_mut()?.consume();
            Ok(())
        })();
        if result.is_err() {
            self.fail_cold_model_work(handle);
        }
        result
    }

    /// Native completion, after the callback and all original consumer joins.
    /// Model and reached semantic commands are returned together. Other native S
    /// and physical M remain the original operation's mandatory producers.
    pub fn finish_cold_model_work(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        streams: &[u64],
        disposition: SemanticColdModelWorkDisposition,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.finish_cold_model_work_for_reader(
            ColdWorkReader::Published(lease),
            handle,
            streams,
            disposition,
        )
    }

    pub fn finish_prepared_cold_model_work(
        &mut self,
        step: &SemanticPreparedStep,
        parent: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        streams: &[u64],
        disposition: SemanticColdModelWorkDisposition,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.finish_cold_model_work_for_reader(
            ColdWorkReader::Prepared(step, parent),
            handle,
            streams,
            disposition,
        )
    }

    fn finish_cold_model_work_for_reader(
        &mut self,
        reader: ColdWorkReader<'_>,
        handle: &SemanticColdModelWork,
        streams: &[u64],
        disposition: SemanticColdModelWorkDisposition,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        let token = self.check_cold_work_completion_reader(reader, handle)?;
        let storage = self.original_cold_model_work(handle)?;
        if storage.pending_operation.is_some()
            || (storage.stopped_before_entry
                && disposition != SemanticColdModelWorkDisposition::KnownRefusal)
        {
            return Err(publication_input_error(
                "cold completion cannot resolve or discard an original operation attempt",
            ));
        }
        if let Some(plan) = &storage.plan {
            plan.require_recorded_trace(storage.recorded_events())
                .map_err(publication_input_error)?;
            if disposition == SemanticColdModelWorkDisposition::Complete && !plan.complete() {
                return Err(publication_input_error(
                    "successful cold completion requires every non-skipped original plan span",
                ));
            }
        } else if disposition != SemanticColdModelWorkDisposition::KnownRefusal
            || !storage.recorded_events().is_empty()
            || storage
                .regions
                .iter()
                .any(|state| !matches!(state, RecordingState::Waiting | RecordingState::NotEntered))
        {
            return Err(publication_input_error(
                "cold completion without an admitted plan requires known refusal before every model callback",
            ));
        }
        if disposition == SemanticColdModelWorkDisposition::Complete {
            storage
                .allowance
                .lock()
                .map_err(|_| {
                    publication_input_error("original resident read allowance is poisoned")
                })?
                .require_resident_reads_complete()?;
        }
        if matches!(
            storage.state,
            RecordingState::Submitting | RecordingState::Submitted | RecordingState::Completed
        ) {
            if storage.streams.as_deref() != Some(streams)
                || storage.disposition != Some(disposition)
            {
                return Err(publication_input_error(
                    "cold completion changed its original submitted consumer roster or disposition",
                ));
            }
            return self.resolve_cold_model_work_for_reader(reader, handle);
        }
        if self.original_cold_model_work(handle)?.state != RecordingState::Closed
            || Arc::strong_count(&self.original_cold_model_work(handle)?.aliases) != 1
        {
            return Err(publication_input_error(
                "cold completion requires its original closed recorder without scratch or child aliases",
            ));
        }
        if self.original_cold_model_work(handle)?.plan.is_none() {
            self.initialize_original_cold_model_work(handle, Some(reader))?;
        } else {
            self.continue_cold_model_work_initialization(handle)?;
        }
        match reader {
            ColdWorkReader::Published(lease) => self.quiesce_published_reader(lease, streams)?,
            ColdWorkReader::Prepared(_, _)
            | ColdWorkReader::FrozenPrepared(_, _)
            | ColdWorkReader::ModelContextActivation(_, _, _)
            | ColdWorkReader::ModelContextMaterial(_, _, _)
            | ColdWorkReader::Admitted(_, _)
            | ColdWorkReader::AdmittedCancellation(_, _, _)
            | ColdWorkReader::AdmittedRetirement(_) => {
                let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
                    .map_err(|error| {
                        runtime_error("prepared cold completion stream admission", error)
                    })?;
                // Join this original cold callback without retiring the live
                // prepared model/content aliases needed by its future graph.
                self.complete_step_consumers_by_token(token, streams)?;
            }
        }
        // Joining only this retained consumer owner must not authorize a fresh
        // report submission while another operation remains pending or poisoned.
        self.ensure_quiescent()?;
        let storage = self.original_cold_model_work(handle)?;
        let metadata = if storage.recorded_events().is_empty() {
            None
        } else {
            Some(
                crate::device::RetainedDeviceWrite::new(
                    self.domain.execution_stream(),
                    storage.recorded_events(),
                    storage
                        .work()?
                        .device
                        .view()
                        .slice(..storage.recorded_events().len()),
                )
                .map_err(|error| runtime_error("cold model work metadata staging", error))?,
            )
        };
        let completion = ColdReportCompletion {
            metadata,
            metadata_counted: false,
            metadata_recorded: false,
            metadata_completed: false,
            command: OriginalNativeCommand::new(&self.domain)?,
            read: self.stage_publication_read(storage.report.view())?,
            poisoned: false,
        };
        let storage = self
            .steps
            .get_mut(&handle.token)
            .expect("checked reader")
            .cold_model_work
            .as_mut()
            .expect("checked cold work");
        if let Some(work) = &mut storage.work {
            work.recording
                .freeze_cold()
                .map_err(publication_input_error)?;
        }
        storage.state = RecordingState::Submitting;
        storage.streams = Some(streams.to_vec());
        storage.disposition = Some(disposition);
        storage.completion = Some(Arc::new(Mutex::new(completion)));
        self.resolve_cold_model_work_for_reader(reader, handle)
    }

    /// Read only the original already submitted report; never replay a callback,
    /// work marker, reset, metadata upload or result kernel on continuation.
    pub fn resolve_cold_model_work(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.resolve_cold_model_work_for_reader(ColdWorkReader::Published(lease), handle)
    }

    pub fn resolve_prepared_cold_model_work(
        &mut self,
        step: &SemanticPreparedStep,
        parent: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.resolve_cold_model_work_for_reader(ColdWorkReader::Prepared(step, parent), handle)
    }

    fn resolve_cold_model_work_for_reader(
        &mut self,
        reader: ColdWorkReader<'_>,
        handle: &SemanticColdModelWork,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        let token = self.check_cold_work_completion_reader(reader, handle)?;
        let storage = self.original_cold_model_work(handle)?;
        if !matches!(
            storage.state,
            RecordingState::Submitting | RecordingState::Submitted | RecordingState::Completed
        ) {
            return Err(publication_input_error(
                "cold resolution requires its original submitted report",
            ));
        }
        if let Some(result) = storage.result {
            return Ok(result);
        }
        // This owner is installed only after the original consumers joined.
        // Do not create another edge owner while resolving its report prefix.
        let mut admission_error = self.ensure_quiescent_except_cold_report(Some(token)).err();
        let original = Arc::clone(storage.completion.as_ref().ok_or_else(|| {
            publication_input_error("cold resolution lost its original report completion owner")
        })?);
        let mut original = original.lock().map_err(|_| {
            publication_input_error("original cold report completion owner is poisoned")
        })?;
        let completion = &mut *original;
        let storage = self
            .steps
            .get_mut(&handle.token)
            .expect("completed reader")
            .cold_model_work
            .as_mut()
            .expect("completed cold work");
        if let Some(metadata) = &mut completion.metadata {
            if !metadata.entered() {
                if let Some(error) = admission_error.take() {
                    return Err(error);
                }
                if !completion.metadata_counted {
                    self.provider
                        .admit_launch_metadata_htod(std::mem::size_of_val(
                            storage.recorded_events(),
                        ));
                    completion.metadata_counted = true;
                }
                let submitted = metadata
                    .enqueue(self.domain.execution_stream())
                    .map_err(|error| {
                        completion.poisoned |= metadata.entered();
                        runtime_error("cold model work metadata upload", error)
                    });
                let context_report = storage
                    .allowance
                    .lock()
                    .map_err(|_| publication_input_error("context report allowance is poisoned"))?
                    .is_model_context_report();
                if context_report && metadata.submitted() && !completion.metadata_recorded {
                    storage
                        .allowance
                        .lock()
                        .map_err(|_| {
                            publication_input_error("context report metadata allowance is poisoned")
                        })?
                        .record_submitted_dma(
                            std::mem::size_of_val(storage.recorded_events()),
                            &mut completion.metadata_recorded,
                        )?;
                }
                submitted?;
            }
            metadata.resolve().map_err(|error| {
                completion.poisoned |= metadata.entered();
                runtime_error("cold model work metadata completion", error)
            })?;
            completion.metadata_completed = true;
        }
        if !completion
            .command
            .resolve_entered(&mut completion.poisoned)?
        {
            if let Some(error) = admission_error.take() {
                return Err(error);
            }
            let kernel = self
                .provider
                .device()
                .inner()
                .get_func(
                    "xlog_semantic_transition",
                    "semantic_cold_model_work_result",
                )
                .ok_or_else(|| {
                    runtime_error("kernel lookup", "cold model work result unavailable")
                })?;
            let input = if storage.recorded_events().is_empty() {
                ModelWorkInput::default()
            } else {
                storage.work()?.descriptor()
            };
            let mut recorder = self.domain.new_strict_recorder();
            if let Some(work) = &storage.work {
                work.record_reads(&mut recorder);
            }
            recorder.read(&storage.native_work);
            recorder.write(&storage.report);
            let arguments = (
                input.events,
                input.count,
                input.bound,
                storage.native_work.device_ptr_value(),
                storage.report.device_ptr_value(),
            );
            completion.command.run(
                &self.domain,
                &mut completion.poisoned,
                recorder,
                |enqueue, entered, submitted| {
                    let (events, count, bound, native, report) = arguments;
                    let events = events.into_kernel_param_storage();
                    let count = count.into_kernel_param_storage();
                    let bound = bound.into_kernel_param_storage();
                    let native = native.into_kernel_param_storage();
                    let report = report.into_kernel_param_storage();
                    let mut parameters = [
                        events.as_kernel_param(),
                        count.as_kernel_param(),
                        bound.as_kernel_param(),
                        native.as_kernel_param(),
                        report.as_kernel_param(),
                    ];
                    // SAFETY: all inputs and the report remain in this exact
                    // original invocation through driver entry and completion.
                    unsafe {
                        kernel.launch_raw_in_original(
                            enqueue,
                            LaunchConfig {
                                grid_dim: (1, 1, 1),
                                block_dim: (1, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            &mut parameters,
                            false,
                            entered,
                            submitted,
                        )
                    }
                    .map_err(|error| XlogError::Kernel(error.to_string()))
                },
            )?;
        }
        storage.state = RecordingState::Submitted;
        if !completion.read.read.entered() {
            if let Some(error) = admission_error.take() {
                return Err(error);
            }
        }
        let words = self
            .resolve_publication_read_with_poison(&mut completion.read, &mut completion.poisoned)?;
        let storage = self
            .steps
            .get_mut(&handle.token)
            .expect("completed reader")
            .cold_model_work
            .as_mut()
            .expect("completed cold work");
        if words.len() != 15
            || words[0] != 0
            || words[2] != storage.recorded_events().len() as u64
            || Some(words[3])
                != storage
                    .work
                    .as_ref()
                    .map_or(Some(0), |work| work.recording.frozen_bound())
            || words[1] > words[3]
            || words[4]
                != storage
                    .recorded_events()
                    .iter()
                    .filter(|event| event.kind == ModelWorkKind::ModelInvocation as u64)
                    .count() as u64
            || words[6..15]
                .iter()
                .try_fold(0u64, |sum, units| sum.checked_add(*units))
                != Some(words[5])
            || words[1].checked_add(words[5]).is_none()
        {
            storage.state = RecordingState::Failed;
            return Err(publication_input_error(
                "cold model work report is incomplete; retain its original operation",
            ));
        }
        let mut result = SemanticColdModelWorkResult {
            model_work: words[1],
            operation_count: words[2],
            work_bound: words[3],
            model_calls: words[4],
            native_work: words[5],
            native_events: words[6..15]
                .try_into()
                .expect("checked original native event extent"),
        };
        storage
            .allowance
            .lock()
            .map_err(|_| publication_input_error("original cold allowance is poisoned"))?
            .merge_submitted_dma(&mut result)?;
        if let Some(context) = &storage.model_context {
            let prior = context.prior;
            let actual = result
                .model_work
                .checked_add(result.native_work)
                .and_then(|work| {
                    prior.map_or(Some(work), |prior| {
                        work.checked_add(prior.model_work)?
                            .checked_add(prior.native_work)
                    })
                })
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let calls = result
                .model_calls
                .checked_add(prior.map_or(0, |prior| prior.model_calls))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            if actual > context.budget[0] || calls > context.budget[2] {
                storage.state = RecordingState::Failed;
                return Err(publication_input_error(
                    "model context cold result exceeds its original instruction budget",
                ));
            }
        }
        let final_cursor = if storage.plan.is_some() {
            let mut proof = Vec::new();
            let count = storage.regions.len().max(1);
            proof
                .try_reserve_exact(count)
                .map_err(|error| runtime_error("cold completion cursor storage", error))?;
            for index in 0..count {
                proof.push(storage.report_cursor(
                    if storage.regions.is_empty() {
                        None
                    } else {
                        Some(index)
                    },
                    true,
                )?);
            }
            Some(CompletedColdReportCursor {
                segmented: !storage.regions.is_empty(),
                regions: proof,
            })
        } else {
            None
        };
        self.graph
            .complete_cold_work(&storage.native_work.view())
            .map_err(SemanticTransitionError::Semantic)?;
        storage.result = Some(result);
        storage.state = RecordingState::Completed;
        if let Some(proof) = final_cursor {
            storage
                .final_cursor
                .set(proof)
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        }
        completion.read.read.retire_completed_values();
        Ok(result)
    }
}
