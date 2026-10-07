//! Model expenditure inside an original admitted cold operation. This owner has
//! no evaluation, backward-tape, content-admission or publication authority.

use super::*;

#[derive(Clone, Debug)]
pub struct SemanticColdModelWork {
    issuer: Arc<()>,
    invocation: Arc<()>,
    token: u64,
    operation_ordinal: u64,
    admission: Arc<[u8]>,
}

impl SemanticColdModelWork {
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum RecordingState {
    Waiting,
    Recording,
    Closed,
    Failed,
    Submitting,
    Submitted,
    Completed,
}

pub(super) struct ColdModelWorkStorage {
    invocation: Arc<()>,
    operation_ordinal: u64,
    admission: Arc<[u8]>,
    work: PreparedModelWork,
    native_work: TrackedCudaSlice<u64>,
    report: TrackedCudaSlice<u64>,
    state: RecordingState,
    result: Option<SemanticColdModelWorkResult>,
    streams: Option<Vec<u64>>,
}

impl SemanticTransitionSession {
    pub(super) fn cold_native_work(
        &self,
        token: u64,
    ) -> Result<Option<DeviceMemoryView<u64>>, SemanticTransitionError> {
        let Some(storage) = self
            .steps
            .get(&token)
            .and_then(|step| step.cold_model_work.as_ref())
        else {
            return Ok(None);
        };
        match storage.state {
            RecordingState::Waiting | RecordingState::Recording | RecordingState::Closed => {
                Ok(Some(storage.native_work.view()))
            }
            RecordingState::Completed => Ok(None),
            _ => Err(publication_input_error(
                "cold native producers cannot run after failure or report submission",
            )),
        }
    }

    pub(super) fn require_completed_cold_model_work(&self) -> Result<(), SemanticTransitionError> {
        if self.steps.values().any(|step| {
            step.cold_model_work
                .as_ref()
                .is_some_and(|storage| storage.state != RecordingState::Completed)
        }) {
            return Err(publication_input_error(
                "state-changing work or release cannot interrupt an original cold model operation",
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
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.require_completed_cold_model_work()?;
        self.checked_reader(lease)?;
        let step = self.checked_step(lease)?;
        if capacity == 0
            || admission.is_empty()
            || self.prepared_segment.is_some()
            || step.prepared.is_some()
            || step.evaluation.is_some()
            || step.cold_model_work.is_some()
        {
            return Err(publication_input_error(
                "cold model work requires its sole admitted operation and positive original capacity",
            ));
        }
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
        let invocation = Arc::new(());
        let handle = SemanticColdModelWork {
            issuer: Arc::clone(&self.publication_issuer),
            invocation: Arc::clone(&invocation),
            token: lease.token,
            operation_ordinal,
            admission: Arc::clone(&admission),
        };
        self.steps
            .get_mut(&lease.token)
            .expect("checked original reader")
            .cold_model_work = Some(ColdModelWorkStorage {
            invocation,
            operation_ordinal,
            admission,
            work,
            native_work,
            report,
            state: RecordingState::Waiting,
            result: None,
            streams: None,
        });
        self.steps[&lease.token]
            .cold_model_work
            .as_ref()
            .expect("retained cold work")
            .work
            .reset_slots(&self.domain, &mut self.poisoned, 0, capacity)?;
        let storage = self
            .steps
            .get_mut(&lease.token)
            .expect("retained reader")
            .cold_model_work
            .as_mut()
            .expect("retained cold work");
        // Initialize the actual tally once, before the original semantic commands.
        // This fixed instrumentation metadata is not a fabricated work occurrence.
        self.provider
            .htod_launch_metadata_sync_copy_into(&[0u64; 11], &mut storage.native_work)
            .map_err(|error| runtime_error("cold semantic work initialization", error))?;
        self.graph
            .begin_cold_work(storage.native_work.view())
            .map_err(SemanticTransitionError::Semantic)?;
        Ok(handle)
    }

    fn cold_model_work(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<&ColdModelWorkStorage, SemanticTransitionError> {
        self.ensure_quiescent()?;
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

    pub fn cold_model_work_stream(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<u64, SemanticTransitionError> {
        self.cold_model_work(handle)?;
        Ok(self.stream.cu_stream() as u64)
    }

    pub fn cold_model_work_buffer(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.checked_reader(lease)?;
        let storage = self.cold_model_work(handle)?;
        if lease.token != handle.token
            || stream != self.stream.cu_stream() as u64
            || storage.state != RecordingState::Waiting
        {
            return Err(publication_input_error(
                "cold model scratch requires its original reader and stream before recording",
            ));
        }
        // SAFETY: reset initialized all three words of every original slot.
        let view = unsafe { storage.work.actual.view().cast::<u8>() }
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let capacity = i64::try_from(storage.work.actual.len() / 3)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        // This is the Session's own stream, already joined by cold completion,
        // not an additional external consumer omitted from the frozen roster.
        let guard = Arc::clone(&self.checked_step(lease)?.aliases);
        self.export_owned_view(view, vec![capacity, 3], vec![3, 1], (1, 64), guard, stream)
    }

    pub fn begin_cold_model_work(
        &mut self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        if self.cold_model_work(handle)?.state != RecordingState::Waiting {
            return Err(publication_input_error(
                "original cold recorder begins exactly once",
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
        storage.work.uncaptured_recording = true;
        Ok(())
    }

    pub fn close_cold_model_work(
        &mut self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        if self.cold_model_work(handle)?.state != RecordingState::Recording {
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

    pub fn record_cold_model_work(
        &mut self,
        handle: &SemanticColdModelWork,
        kind: ModelWorkKind,
        dimensions: &[u64],
        device_produced: bool,
    ) -> Result<usize, SemanticTransitionError> {
        let result = (|| {
            if self.cold_model_work(handle)?.state != RecordingState::Recording {
                return Err(publication_input_error(
                    "cold model producer requires its active original recorder",
                ));
            }
            let work = &mut self
                .steps
                .get_mut(&handle.token)
                .expect("checked reader")
                .cold_model_work
                .as_mut()
                .expect("checked cold work")
                .work;
            work.record_operation(
                kind,
                dimensions,
                device_produced,
                &self.domain,
                &mut self.poisoned,
            )
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
        let result = (|| {
            if self.cold_model_work(handle)?.state != RecordingState::Recording {
                return Err(publication_input_error(
                    "cold model call requires its active original recorder",
                ));
            }
            self.steps
                .get_mut(&handle.token)
                .expect("checked reader")
                .cold_model_work
                .as_mut()
                .expect("checked cold work")
                .work
                .record_invocation(&self.domain, &mut self.poisoned)
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
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.checked_reader(lease)?;
        if lease.token != handle.token
            || self.cold_model_work(handle)?.state != RecordingState::Closed
        {
            return Err(publication_input_error(
                "cold completion requires its original explicitly closed recorder",
            ));
        }
        self.quiesce_published_reader(lease, streams)?;
        let kernel = self
            .provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_cold_model_work_result",
            )
            .ok_or_else(|| runtime_error("kernel lookup", "cold model work result unavailable"))?;
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
        let work = &storage.work;
        let input = if work.recording.events().is_empty() {
            ModelWorkInput::default()
        } else {
            let mut destination = work.device.view().slice(..work.recording.events().len());
            self.provider
                .htod_launch_metadata_sync_copy_into(work.recording.events(), &mut destination)
                .map_err(|error| runtime_error("cold model work metadata upload", error))?;
            work.descriptor()
        };
        let mut recorder = self.domain.new_strict_recorder();
        work.record_reads(&mut recorder);
        recorder.read(&storage.native_work);
        recorder.write(&storage.report);
        let arguments = (
            input.events,
            input.count,
            input.bound,
            storage.native_work.device_ptr_value(),
            storage.report.device_ptr_value(),
        );
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            // SAFETY: the retained private inputs and fifteen-word report belong to
            // this one operation; no model alias can modify event metadata.
            unsafe {
                kernel.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })?;
        self.steps
            .get_mut(&handle.token)
            .expect("submitted reader")
            .cold_model_work
            .as_mut()
            .expect("submitted cold work")
            .state = RecordingState::Submitted;
        self.resolve_cold_model_work(lease, handle)
    }

    /// Read only the original already submitted report; never replay a callback,
    /// work marker, reset, metadata upload or result kernel on continuation.
    pub fn resolve_cold_model_work(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.checked_reader(lease)?;
        let storage = self.cold_model_work(handle)?;
        if lease.token != handle.token
            || !matches!(
                storage.state,
                RecordingState::Submitted | RecordingState::Completed
            )
        {
            return Err(publication_input_error(
                "cold resolution requires its original submitted report",
            ));
        }
        if let Some(result) = storage.result {
            return Ok(result);
        }
        let streams = storage
            .streams
            .as_ref()
            .expect("submitted original streams")
            .clone();
        self.complete_step_consumers(lease, &streams)?;
        let words = self.publication_read(self.cold_model_work(handle)?.report.view())?;
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
        let result = SemanticColdModelWorkResult {
            model_work: words[1],
            operation_count: words[2],
            work_bound: words[3],
            model_calls: words[4],
            native_work: words[5],
            native_events: words[6..15]
                .try_into()
                .expect("checked original native event extent"),
        };
        self.graph
            .complete_cold_work(&storage.native_work.view())
            .map_err(SemanticTransitionError::Semantic)?;
        storage.result = Some(result);
        storage.state = RecordingState::Completed;
        Ok(result)
    }
}
