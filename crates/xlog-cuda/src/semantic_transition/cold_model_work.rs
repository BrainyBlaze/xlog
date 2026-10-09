//! Model expenditure inside an original admitted cold operation. This owner has
//! no evaluation, backward-tape, content-admission or publication authority.

use super::*;
use crate::semantic_work::ModelWorkPlan;

#[derive(Clone, Debug)]
pub struct SemanticColdModelWork {
    issuer: Arc<()>,
    invocation: Arc<()>,
    token: u64,
    operation_ordinal: u64,
    admission: Arc<[u8]>,
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
    AdmittedRetirement(&'a SemanticSegmentInstructionAdmission),
    Prepared(&'a SemanticPreparedStep, &'a SemanticPublishedLease),
}

#[derive(Clone, Copy)]
enum InstructionColdRole {
    Preparation,
    Retirement,
}

struct EvaluationCleanup {
    region: usize,
    original: SemanticModelEvaluation,
    cancellation: Option<SemanticCancelledModelEvaluation>,
}

struct ColdReportCompletion {
    metadata: Option<crate::device::RetainedDeviceWrite<ModelWorkEvent>>,
    metadata_counted: bool,
    command: OriginalNativeCommand,
    read: PublicationRead<u64>,
    poisoned: bool,
}

pub(super) struct ColdModelWorkStorage {
    invocation: Arc<()>,
    aliases: Arc<()>,
    prepared_scope: Option<Arc<()>>,
    operation_ordinal: u64,
    admission: Arc<[u8]>,
    allowance: Arc<Mutex<native_work_bound::ColdNativeAllowance>>,
    work: PreparedModelWork,
    native_work: TrackedCudaSlice<u64>,
    initialization_reset: OriginalNativeCommand,
    initialization_write: crate::device::RetainedDeviceWrite<u64>,
    initialization_poisoned: bool,
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

impl ColdModelWorkStorage {
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
            return Ok(self.graph.borrowed_cold_work());
        };
        match storage.state {
            RecordingState::Waiting | RecordingState::Recording | RecordingState::Closed => {
                Ok(Some(storage.native_work.view()))
            }
            // A completed local report does not hide the next original
            // operation's borrowed tally. Late verification and retirement of
            // this reader must charge their actual commands to that live owner.
            RecordingState::Completed => Ok(self.graph.borrowed_cold_work()),
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

    fn attach_cold_native_work(
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
                        || prior.work.actual.len() != original.work.actual.len()
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
            .begin_cold_work(view.clone())
            .map_err(SemanticTransitionError::Semantic)?;
        self.graph
            .begin_borrowed_cold_work(view, custody)
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
        admission: Arc<[u8]>,
        purpose: SemanticColdModelWorkPurpose,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.prepare_cold_model_work_inner(
            lease,
            capacity,
            operation_ordinal,
            admission,
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
        let capacity = self
            .admitted_segment_cold_capacity(admission)?
            .model_work_capacity;
        let ordinal = self.admitted_segment_first_program_ordinal(admission)?;
        self.prepare_cold_model_work_inner(
            parent,
            capacity,
            ordinal,
            instruction,
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
        self.check_cold_work_reader(ColdWorkReader::Published(parent), work)?;
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
            let add = |left: u64, right: u64| {
                left.checked_add(right)
                    .ok_or(SemanticTransitionError::GenerationExhausted)
            };
            total.model_work = add(total.model_work, report.model_work)?;
            total.operation_count = add(total.operation_count, report.operation_count)?;
            total.work_bound = add(total.work_bound, report.work_bound)?;
            total.model_calls = add(total.model_calls, report.model_calls)?;
            total.native_work = add(total.native_work, report.native_work)?;
            for (total, actual) in total.native_events.iter_mut().zip(report.native_events) {
                *total = add(*total, actual)?;
            }
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
        let capacity = self
            .admitted_segment_cold_capacity(admission)?
            .model_work_capacity;
        let ordinal = self.admitted_segment_first_program_ordinal(admission)?;
        // Freeze the native disposition before installing or initializing the
        // original report. An uncertain initialization cannot change its mode.
        self.instruction_admission_mut(admission)?.retirement_cancellation = cancellation.cloned();
        self.install_cold_model_work(
            token,
            capacity,
            ordinal,
            instruction,
            SemanticColdModelWorkPurpose::PrivateRetirement,
            Some(scope),
            Some((admission, InstructionColdRole::Retirement)),
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
                self.prepared_segment.as_ref()
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
        capacity: usize,
        operation_ordinal: u64,
        admission: Arc<[u8]>,
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
            !Arc::ptr_eq(&previous.admission, &admission)
                || previous.work.actual.len() / 3 != capacity
                || previous.operation_ordinal > operation_ordinal
        }) {
            return Err(publication_input_error(
                "a later cold callback cannot change its original admission, capacity or operation order",
            ));
        }
        if capacity == 0
            || admission.is_empty()
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
            capacity,
            operation_ordinal,
            admission,
            purpose,
            None,
            instruction,
        )
    }

    /// Retain the original prepared callback's report without acquiring a reader.
    pub fn prepare_prepared_cold_model_work(
        &mut self,
        step: &SemanticPreparedStep,
        parent: &SemanticPublishedLease,
        capacity: usize,
        operation_ordinal: u64,
        admission: Arc<[u8]>,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.require_prepared_replay_parent(step, parent)?;
        self.require_completed_cold_model_work()?;
        let owner = self.checked_prepared_step(step, false)?;
        if capacity == 0
            || admission.is_empty()
            || owner.cold_model_work.is_some()
            || owner.evaluation.is_some()
            || capacity
                > self
                    .prepared_segment
                    .as_ref()
                    .expect("checked prepared scope")
                    .cold_capacity
                    .model_work_capacity
        {
            return Err(publication_input_error(
                "prepared cold work requires one original admitted callback before execution",
            ));
        }
        self.install_cold_model_work(
            step.token,
            capacity,
            operation_ordinal,
            admission,
            SemanticColdModelWorkPurpose::PreparedActorRefresh,
            Some(Arc::clone(&step.scope)),
            None,
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
        let capacity = self
            .admitted_segment_cold_capacity(admission)?
            .model_work_capacity;
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
                return self
                    .prepare_prepared_cold_model_work(step, parent, capacity, ordinal, instruction);
            };
            if storage.operation_ordinal != ordinal
                || !Arc::ptr_eq(&storage.admission, &instruction)
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
            };
            self.continue_cold_model_work_initialization(&handle)?;
            return Ok(handle);
        }
        let original = self.instruction_admission_mut(admission)?;
        original.prepared_cold_results.push((ordinal, None));
        self.prepare_prepared_cold_model_work(step, parent, capacity, ordinal, instruction)
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
            || !Arc::ptr_eq(&work.admission, &original.instruction)
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
        capacity: usize,
        operation_ordinal: u64,
        admission: Arc<[u8]>,
        purpose: SemanticColdModelWorkPurpose,
        prepared_scope: Option<Arc<()>>,
        instruction: Option<(&SemanticSegmentInstructionAdmission, InstructionColdRole)>,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        let bytes = SemanticColdModelWork::allocation_bytes(capacity)?;
        let mut reservation = self
            .provider
            .memory()
            .reserve_bytes(bytes as u64)
            .map_err(|error| runtime_error("cold model work reservation", error))?;
        let work = PreparedModelWork::allocate(&self.provider, &mut reservation, capacity)?;
        let native_work = reservation
            .alloc(11)
            .map_err(|error| runtime_error("cold semantic work allocation", error))?;
        let report = reservation
            .alloc(15)
            .map_err(|error| runtime_error("cold model work report allocation", error))?;
        let initialization_reset = OriginalNativeCommand::new(&self.domain)?;
        let initialization_write = crate::device::RetainedDeviceWrite::new(
            self.domain.execution_stream(),
            &[0u64; 11],
            native_work.view(),
        )
        .map_err(|error| runtime_error("cold semantic work staging", error))?;
        let invocation = Arc::new(());
        let handle = SemanticColdModelWork {
            issuer: Arc::clone(&self.publication_issuer),
            invocation: Arc::clone(&invocation),
            token,
            operation_ordinal,
            admission: Arc::clone(&admission),
        };
        // Keep the completed storage until all new allocation succeeds. The
        // new invocation is retained before reset or any model callback; an
        // allocation failure grants no rights to replay the prior callback.
        self.steps
            .get_mut(&token)
            .expect("checked original reader")
            .cold_model_work = Some(ColdModelWorkStorage {
            invocation,
            aliases: Arc::new(()),
            prepared_scope,
            operation_ordinal,
            admission,
            allowance: Arc::new(Mutex::new(native_work_bound::ColdNativeAllowance::new(
                purpose,
            ))),
            work,
            native_work,
            initialization_reset,
            initialization_write,
            initialization_poisoned: false,
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
        if let Some((instruction, role)) = instruction {
            // Admission owns this exact invocation before the first reset or
            // transfer. An uncertain initialization never grants another attempt.
            let original = self.instruction_admission_mut(instruction)?;
            match role {
                InstructionColdRole::Preparation => original.cold_work = Some(handle.clone()),
                InstructionColdRole::Retirement => original.retirement_work = Some(handle.clone()),
            }
        }
        self.continue_cold_model_work_initialization(&handle)?;
        Ok(handle)
    }

    fn continue_cold_model_work_initialization(
        &mut self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        // Authenticate the exact retained allocation without a healthy-session
        // precondition: joining its entered prefix grants no unrelated work.
        let another_initializer = self.steps.iter().any(|(token, step)| {
            *token != handle.token
                && step
                    .cold_model_work
                    .as_ref()
                    .is_some_and(|storage| !storage.initialized)
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
        let historical_poison = self.poisoned || self.graph.ensure_not_poisoned().is_err();
        if historical_poison || another_initializer {
            // A different owner or abort may have poisoned the Session while
            // this initializer was unresolved. Join only its entered work;
            // never submit an unentered suffix or grant callback entry.
            storage
                .initialization_reset
                .resolve_entered(&mut storage.initialization_poisoned)?;
            if storage.initialization_write.entered() {
                storage.initialization_write.resolve().map_err(|error| {
                    storage.initialization_poisoned = true;
                    runtime_error("cold semantic work initialization", error)
                })?;
            }
            return Err(if historical_poison {
                SemanticTransitionError::Poisoned
            } else {
                SemanticTransitionError::OverlappingLaunch
            });
        }
        storage.work.reset_slots_with_original(
            &self.domain,
            &mut storage.initialization_poisoned,
            0,
            storage.work.actual.len() / 3,
            Some(&mut storage.initialization_reset),
        )?;
        if !storage.initialization_write.entered() {
            if let Err(error) = storage
                .initialization_write
                .enqueue(self.domain.execution_stream())
            {
                storage.initialization_poisoned |= storage.initialization_write.entered();
                return Err(runtime_error("cold semantic work initialization", error));
            }
        }
        storage.initialization_write.resolve().map_err(|error| {
            storage.initialization_poisoned |= storage.initialization_write.entered();
            runtime_error("cold semantic work initialization", error)
        })?;
        self.graph
            .begin_cold_work(storage.native_work.view(), Arc::clone(&storage.allowance))
            .map_err(SemanticTransitionError::Semantic)?;
        storage.initialized = true;
        Ok(())
    }

    pub(super) fn has_uninitialized_cold_model_work(&self) -> bool {
        self.steps.values().any(|step| {
            step.cold_model_work
                .as_ref()
                .is_some_and(|storage| !storage.initialized)
        })
    }

    pub(super) fn has_pending_cold_model_work_completion(&self, except_token: Option<u64>) -> bool {
        self.steps.iter().any(|(token, step)| {
            except_token != Some(*token)
                && step.cold_model_work.as_ref().is_some_and(|storage| {
                    storage.completion.is_some() && storage.result.is_none()
                })
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
            || handle.operation_ordinal != work.operation_ordinal
        {
            return Err(publication_input_error(
                "cold model work belongs to another Session, admission or operation",
            ));
        }
        Ok(work)
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
        Ok(self.cold_model_work(handle)?.plan.is_some())
    }

    /// Fixed registration-region roster of this exact original report.
    pub fn cold_model_work_region_count(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<usize, SemanticTransitionError> {
        Ok(self.cold_model_work(handle)?.regions.len())
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
        )
    }

    fn admit_cold_model_work_plan_in_region(
        &mut self,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
        operations: &[(ModelWorkKind, Vec<u64>)],
        region_ends: &[usize],
        evaluation_content: Option<SemanticColdEvaluationContent>,
    ) -> Result<[u64; 3], SemanticTransitionError> {
        let result = (|| {
            if let Some(region) = region {
                self.require_cold_model_work_region(region, RecordingState::Waiting)?;
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
                || !storage.work.recording.events().is_empty()
                || storage.regions.iter().any(|state| {
                    !matches!(state, RecordingState::Waiting | RecordingState::NotEntered)
                })
            {
                return Err(publication_input_error(
                    "the original cold operation plan is admitted once",
                ));
            }
            let capacity = storage.work.actual.len() / 3;
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
            let mut allowance_guard = allowance.lock().map_err(|_| {
                publication_input_error("original cold native allowance is poisoned")
            })?;
            if allowance_guard.is_evaluation() != evaluation_content.is_some() {
                return Err(publication_input_error("the native evaluation purpose requires its original output and objective content"));
            }
            if let Some(content) = evaluation_content {
                if !Arc::ptr_eq(&allowance, &content.allowance) {
                    return Err(publication_input_error(
                        "evaluation content belongs to another original cold report",
                    ));
                }
                allowance_guard.freeze_content(content.content)?;
            }
            drop(allowance_guard);
            self.steps
                .get_mut(&handle.token)
                .expect("checked reader")
                .cold_model_work
                .as_mut()
                .expect("checked cold work")
                .plan = Some(plan);
            Ok(quantities)
        })();
        if result.is_err() {
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

    fn cold_model_work_buffer_for_reader(
        &mut self,
        reader: ColdWorkReader<'_>,
        handle: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        let token = self.check_cold_work_reader(reader, handle)?;
        let storage = self.cold_model_work(handle)?;
        let available = if let Some(region) = region {
            self.require_cold_model_work_region(region, RecordingState::Waiting)?;
            storage.state == RecordingState::Recording
        } else {
            storage.regions.is_empty() && storage.state == RecordingState::Waiting
        };
        if stream != self.stream.cu_stream() as u64 || !available {
            return Err(publication_input_error(
                "cold model scratch requires its original reader and stream before recording",
            ));
        }
        storage.plan()?;
        // SAFETY: reset initialized all three words of every original slot.
        let view = unsafe { storage.work.actual.view().cast::<u8>() }
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let capacity = i64::try_from(storage.work.actual.len() / 3)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        // This is the Session's own stream, already joined by cold completion,
        // not an additional external consumer omitted from the frozen roster.
        let scratch = Arc::clone(&storage.aliases);
        let guard = Arc::clone(&self.steps[&token].aliases);
        self.export_owned_view(
            view,
            vec![capacity, 3],
            vec![3, 1],
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
        let storage = self.cold_model_work(&region.work)?;
        if storage.state != RecordingState::Recording
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
        if storage.regions.is_empty() {
            storage.plan()?;
        }
        let storage = self
            .steps
            .get_mut(&handle.token)
            .expect("checked reader")
            .cold_model_work
            .as_mut()
            .expect("checked cold work");
        storage.state = RecordingState::Recording;
        storage.work.uncaptured_recording = true;
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
        storage.work.uncaptured_recording = false;
        storage.state = RecordingState::Closed;
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
            storage.work.uncaptured_recording = false;
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
        let first_slot = storage.work.next_slot().map_err(publication_input_error)?;
        if operations.is_empty()
            || first_slot
                .checked_add(operations.len())
                .is_none_or(|end| end > storage.work.actual.len() / 3)
            || storage
                .operations
                .len()
                .checked_add(operations.len())
                .is_none_or(|end| end > storage.work.actual.len() / 3)
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
                    .work
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
            if let Err(error) = storage.work.reset_slots(
                &self.domain,
                &mut self.poisoned,
                first_slot,
                operations.len(),
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
            || storage.work.next_slot().map_err(publication_input_error)? != attempt.slot
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
            .work
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
            let slot = storage.work.record_operation(
                kind,
                dimensions,
                device_produced,
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
            storage
                .work
                .record_invocation(&self.domain, &mut self.poisoned)?;
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
            plan.require_recorded_trace(storage.work.recording.events())
                .map_err(publication_input_error)?;
            if disposition == SemanticColdModelWorkDisposition::Complete && !plan.complete() {
                return Err(publication_input_error(
                    "successful cold completion requires every non-skipped original plan span",
                ));
            }
        } else if disposition != SemanticColdModelWorkDisposition::KnownRefusal
            || !storage.work.recording.events().is_empty()
            || storage
                .regions
                .iter()
                .any(|state| !matches!(state, RecordingState::Waiting | RecordingState::NotEntered))
        {
            return Err(publication_input_error(
                "cold completion without an admitted plan requires known refusal before every model callback",
            ));
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
        match reader {
            ColdWorkReader::Published(lease) => self.quiesce_published_reader(lease, streams)?,
            ColdWorkReader::Prepared(_, _)
            | ColdWorkReader::Admitted(_, _)
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
        let metadata = if storage.work.recording.events().is_empty() {
            None
        } else {
            Some(
                crate::device::RetainedDeviceWrite::new(
                    self.domain.execution_stream(),
                    storage.work.recording.events(),
                    storage
                        .work
                        .device
                        .view()
                        .slice(..storage.work.recording.events().len()),
                )
                .map_err(|error| runtime_error("cold model work metadata staging", error))?,
            )
        };
        let completion = ColdReportCompletion {
            metadata,
            metadata_counted: false,
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
        storage
            .work
            .recording
            .freeze_cold()
            .map_err(publication_input_error)?;
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
                    self.provider.admit_launch_metadata_htod(
                        std::mem::size_of_val(storage.work.recording.events()),
                    );
                    completion.metadata_counted = true;
                }
                metadata
                    .enqueue(self.domain.execution_stream())
                    .map_err(|error| {
                        completion.poisoned |= metadata.entered();
                        runtime_error("cold model work metadata upload", error)
                    })?;
            }
            metadata.resolve().map_err(|error| {
                completion.poisoned |= metadata.entered();
                runtime_error("cold model work metadata completion", error)
            })?;
        }
        if !completion.command.resolve_entered(&mut completion.poisoned)? {
            if let Some(error) = admission_error.take() {
                return Err(error);
            }
            let kernel = self
                .provider
                .device()
                .inner()
                .get_func("xlog_semantic_transition", "semantic_cold_model_work_result")
                .ok_or_else(|| runtime_error("kernel lookup", "cold model work result unavailable"))?;
            let input = if storage.work.recording.events().is_empty() {
                ModelWorkInput::default()
            } else {
                storage.work.descriptor()
            };
            let mut recorder = self.domain.new_strict_recorder();
            storage.work.record_reads(&mut recorder);
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
            || words[2] != storage.work.recording.events().len() as u64
            || Some(words[3]) != storage.work.recording.frozen_bound()
            || words[1] > words[3]
            || words[4]
                != storage
                    .work
                    .recording
                    .events()
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
        self.graph
            .complete_cold_work(&storage.native_work.view())
            .map_err(SemanticTransitionError::Semantic)?;
        storage.result = Some(result);
        storage.state = RecordingState::Completed;
        completion.read.read.retire_completed_values();
        Ok(result)
    }
}
