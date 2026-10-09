//! Read-only model invocations over one original native-selected cohort.

use super::*;

/// Lexical recording of this invocation, never an executable application plan.
pub struct SemanticModelEvaluationCapture {
    invocation: Arc<()>,
    stream: Arc<CudaStream>,
}

pub struct SemanticCapturedModelEvaluation {
    invocation: Arc<()>,
    graph: crate::cuda_graph::CapturedCudaGraph,
}

impl SemanticModelEvaluationCapture {
    pub fn record<F>(
        self,
        resources: Vec<Arc<dyn Send + Sync>>,
        record: F,
    ) -> Result<SemanticCapturedModelEvaluation, XlogError>
    where
        F: FnOnce() -> Result<(), XlogError>,
    {
        let graph = crate::cuda_graph::CapturedCudaGraph::capture_on_stream_retaining(
            &self.stream,
            resources,
            record,
        )?;
        Ok(SemanticCapturedModelEvaluation {
            invocation: self.invocation,
            graph,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EvaluationExecution {
    Prepared,
    Recording,
    CaptureFinalizing,
    Captured,
    LaunchEntered,
    Launched,
    Completed,
}

/// Original selection and its immutable device content, independent of the
/// source Runtime's lifetime. Final evaluations borrow this same allocation.
pub struct SemanticEvaluationCohort {
    issuance: Arc<()>,
    selected: SemanticSelectedTrainingView,
    content: TensorContentBuffers,
    domain: Identity256,
    task: SemanticTaskContentIdentity,
    device: usize,
    provider: Arc<CudaKernelProvider>,
}

impl SemanticEvaluationCohort {
    pub(super) fn content_tensors(&self) -> &[PreparedSemanticTensor] {
        &self.content.tensors
    }

    pub(super) fn content_native_work_ceiling(
        &self,
        verify: bool,
    ) -> Result<[u64; 9], SemanticTransitionError> {
        self.content.native_work_ceiling(verify)
    }
}

/// Metadata-only carrier from the actual original evaluation Session to its
/// enclosing cold report, including retained real numerical allocation owners.
pub struct SemanticColdEvaluationContent {
    pub(super) allowance: Arc<Mutex<native_work_bound::ColdNativeAllowance>>,
    pub(super) content: native_work_bound::FrozenColdEvaluationContent,
}

/// Session-issued invocation handle, not an Update or a publication capability.
#[derive(Clone)]
pub struct SemanticModelEvaluation {
    issuer: Arc<()>,
    invocation: Arc<()>,
    token: u64,
    cohort: Arc<SemanticEvaluationCohort>,
}

impl SemanticModelEvaluation {
    pub fn cohort(&self) -> Arc<SemanticEvaluationCohort> {
        Arc::clone(&self.cohort)
    }
}

/// Cold observation of the original FP32 outputs and canonical work tally.
/// Memory is retained allocation capacity, not an allocator-wide time sample.
#[derive(Clone, Copy, Debug)]
pub struct SemanticModelEvaluationResult {
    pub status: u64,
    pub loss_bits: [u32; 6],
    pub model_work: u64,
    pub operation_count: u64,
    pub work_bound: u64,
    pub retained_allocation_bytes: u64,
    pub model_calls: u64,
}

/// Original completed result with CPU-only issuance and parent custody. No
/// selected cohort, model, reader or CUDA allocation survives through this type.
pub struct SemanticCompletedModelEvaluation {
    issuer: Arc<()>,
    invocation: Arc<()>,
    cohort_issuance: Arc<()>,
    parent: SemanticPublishedIdentity,
    result: SemanticModelEvaluationResult,
}

impl SemanticCompletedModelEvaluation {
    pub fn result(&self) -> SemanticModelEvaluationResult {
        self.result
    }

    pub fn same_invocation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.invocation, &other.invocation)
    }

    pub fn belongs_to(&self, session: &SemanticTransitionSession) -> bool {
        Arc::ptr_eq(&self.issuer, &session.publication_issuer)
    }

    /// Same original selection, not another cohort with equal serialized data.
    /// The CPU token retains none of the cohort's device storage or provider.
    pub fn belongs_to_cohort(&self, cohort: &SemanticEvaluationCohort) -> bool {
        Arc::ptr_eq(&self.cohort_issuance, &cohort.issuance)
    }

    pub fn parent(&self) -> SemanticPublishedIdentity {
        self.parent
    }
}

/// Actual expenditure of the original known-cancelled invocation, not a model
/// observation or a successful evaluation receipt. No objective is synthesized.
#[derive(Clone)]
pub struct SemanticCancelledModelEvaluation {
    issuer: Arc<()>,
    invocation: Arc<()>,
    parent: SemanticPublishedIdentity,
    work: u64,
    calls: u64,
}

impl SemanticCancelledModelEvaluation {
    pub fn belongs_to(
        &self,
        session: &SemanticTransitionSession,
        original: &SemanticModelEvaluation,
    ) -> bool {
        Arc::ptr_eq(&self.issuer, &session.publication_issuer) && self.belongs_to_original(original)
    }

    pub(super) fn belongs_to_original(&self, original: &SemanticModelEvaluation) -> bool {
        Arc::ptr_eq(&self.issuer, &original.issuer)
            && Arc::ptr_eq(&self.invocation, &original.invocation)
    }

    pub fn parent(&self) -> SemanticPublishedIdentity {
        self.parent
    }
    pub fn model_work(&self) -> u64 {
        self.work
    }
    pub fn model_calls(&self) -> u64 {
        self.calls
    }
}

pub(super) struct EvaluationPreparation {
    identity: SemanticPublishedIdentity,
    event_capacity: usize,
    requested_cohort: Option<Arc<SemanticEvaluationCohort>>,
    source_observation: Option<SemanticPublishedStateObservation>,
    source_material: Option<Identity256>,
    cohort: Option<Arc<SemanticEvaluationCohort>>,
    selection: Option<(DeviceMemoryView<u8>, [OriginalNativeCommand; 2])>,
    selected: bool,
    sealed: bool,
    handle: Option<SemanticModelEvaluation>,
    guarded: bool,
    started: bool,
    complete: bool,
    reset: OriginalNativeCommand,
}

pub(super) struct EvaluationStorage {
    invocation: Arc<()>,
    work_aliases: Arc<()>,
    cohort: Arc<SemanticEvaluationCohort>,
    work: PreparedModelWork,
    report: TrackedCudaSlice<u64>,
    source_material: Option<Identity256>,
    cold_work: Option<DeviceMemoryView<u64>>,
    stream: u64,
    submitted: bool,
    report_submitted: bool,
    consumer_streams: Option<Vec<u64>>,
    result: Option<SemanticModelEvaluationResult>,
    execution: EvaluationExecution,
    graph: Option<Arc<crate::cuda_graph::CapturedCudaGraph>>,
    output_witness: Option<SemanticTensorContentWitness>,
    graph_retirement: Option<crate::cuda_graph::CudaGraphRetirement>,
    capture_native_work: [u64; 9],
    frozen_native_work: Option<[u64; 9]>,
    launch_marker_read: Option<PublicationRead<u64>>,
    launch_markers: Option<Vec<u64>>,
    report_read: Option<PublicationRead<u64>>,
    report_words: Option<[u64; 12]>,
    source_observation: Option<SemanticPublishedStateObservation>,
    resolved_source_material: Option<Identity256>,
}

impl SemanticTransitionSession {
    pub fn prepare_cold_evaluation_content(
        &self,
        lease: &SemanticPublishedLease,
        inputs: [Vec<SemanticTensorInput>; 2],
    ) -> Result<SemanticColdEvaluationContent, SemanticTransitionError> {
        self.checked_reader(lease)?;
        let evaluation = self.steps[&lease.token]
            .evaluation
            .as_ref()
            .ok_or_else(|| {
                publication_input_error(
                    "cold evaluation content requires its original issued evaluation",
                )
            })?;
        if evaluation.execution != EvaluationExecution::Prepared || evaluation.submitted {
            return Err(publication_input_error(
                "cold evaluation content must precede original numerical submission",
            ));
        }
        let allowance = self.cold_native_allowance(lease.token)?.ok_or_else(|| {
            publication_input_error("evaluation content lost its original cold allowance")
        })?;
        let work = self.cold_native_work(lease.token)?.ok_or_else(|| {
            publication_input_error("evaluation content lost its original cold tally")
        })?;
        if evaluation.cold_work.as_ref().is_none_or(|original| {
            original.device_ptr() != work.device_ptr() || original.len() != work.len()
        }) {
            return Err(publication_input_error(
                "evaluation content changed its original report allocation",
            ));
        }
        let storage = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let contract = lease
            .directory
            .iter()
            .find(|range| range.role == SemanticStateRole::ModelContract as u64 && range.index == 0)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let mut model_ceiling =
            native_work_bound::content_digest_ceiling(contract.length_bytes, 21, false)?;
        for range in lease
            .directory
            .iter()
            .filter(|range| matches!(range.role, 18..=20))
        {
            let mut layout = *storage
                .layouts
                .get(&(range.role, range.index))
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if layout.logical_axis != u64::MAX {
                let axis = usize::try_from(layout.logical_axis)
                    .ok()
                    .filter(|axis| *axis < layout.rank as usize)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                layout.dimensions[axis] = range
                    .logical_end
                    .checked_sub(range.logical_begin)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
            }
            native_work_bound::add_native_work_ceiling(
                &mut model_ceiling,
                native_work_bound::tensor_layout_native_work_ceiling(
                    &layout,
                    range.logical_begin,
                    range.logical_end,
                    usize::try_from(range.length_bytes)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                    true,
                )?,
            )?;
        }
        let [output, objective] = inputs;
        let output = prepare_semantic_tensors(&self.provider, output)?;
        let objective = prepare_semantic_tensors(&self.provider, objective)?;
        Ok(SemanticColdEvaluationContent {
            allowance,
            content: native_work_bound::FrozenColdEvaluationContent::new(
                Arc::clone(&evaluation.cohort),
                output,
                objective,
                model_ceiling,
            )?,
        })
    }

    /// Original immutable port metadata, without selecting, exporting, or
    /// reading device extents. Numerical owners can reserve their empty outputs
    /// before the original cold operation plan is admitted.
    pub fn evaluation_source_geometry(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<Vec<(Vec<i64>, Vec<i64>, (u8, u8))>, SemanticTransitionError> {
        self.checked_reader(lease)?;
        let arena = self
            .training_views
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        SemanticTrainingViewPort::ALL
            .into_iter()
            .map(|port| arena.port_layout(port))
            .collect()
    }

    /// Select only for the source observation. Supplying its issued cohort
    /// reuses all seventeen ports, never final-model RNG or a caller ordinal.
    pub fn prepare_model_evaluation(
        &mut self,
        lease: &SemanticPublishedLease,
        cohort: Option<Arc<SemanticEvaluationCohort>>,
        event_capacity: usize,
    ) -> Result<SemanticModelEvaluation, SemanticTransitionError> {
        if self
            .steps
            .get(&lease.token)
            .is_some_and(|step| step.evaluation_preparation.is_some())
        {
            let step = self.require_original_cold_content_reader(lease)?;
            let original = step
                .evaluation_preparation
                .as_ref()
                .expect("retained preparation")
                .lock()
                .map_err(|_| publication_input_error("evaluation preparation owner is poisoned"))?;
            let same_cohort = match (&original.requested_cohort, &cohort) {
                (None, None) => true,
                (Some(original), Some(requested)) => Arc::ptr_eq(original, requested),
                _ => false,
            };
            if original.identity != lease.identity
                || original.event_capacity != event_capacity
                || !same_cohort
            {
                return Err(publication_input_error(
                    "evaluation preparation changed its original parent, cohort or capacity",
                ));
            }
        } else {
            self.checked_reader(lease)?;
            if self.steps[&lease.token]
                .evaluation
                .as_ref()
                .is_some_and(|original| {
                    original
                        .result
                        .is_none_or(|result| !matches!(result.status, 0 | 1))
                        || Arc::strong_count(&original.work_aliases) != 1
                })
                || self.prepared_segment.is_some()
            {
                return Err(publication_input_error(
                "evaluation requires known completion and release of the previous invocation's scratch aliases outside a prepared segment",
                ));
            }
            let arena = self
                .training_views
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let task = self
                .task
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .0
                .content_identity();
            if cohort.as_ref().is_some_and(|original| {
                original.domain != arena.training_domain_identity()
                    || original.task != task
                    || original.device != self.provider.device().ordinal()
            }) {
                return Err(publication_input_error("evaluation cohort differs from the original admitted task, domain or CUDA device"));
            }
            let reset = OriginalNativeCommand::new(&self.domain)?;
            let borrowed = cohort.is_some();
            self.steps
                .get_mut(&lease.token)
                .expect("checked acquired step")
                .evaluation_preparation = Some(Arc::new(Mutex::new(EvaluationPreparation {
                identity: lease.identity,
                event_capacity,
                requested_cohort: cohort.clone(),
                source_observation: None,
                source_material: None,
                cohort,
                selection: None,
                selected: borrowed,
                sealed: borrowed,
                handle: None,
                guarded: false,
                started: false,
                complete: false,
                reset,
            })));
        }
        let owner = Arc::clone(
            self.steps
                .get(&lease.token)
                .expect("retained acquired step")
                .evaluation_preparation
                .as_ref()
                .expect("retained preparation"),
        );
        let mut original = owner
            .lock()
            .map_err(|_| publication_input_error("evaluation preparation owner is poisoned"))?;
        // Issue the original numerical owners without entering selection,
        // exports, seals or reset. The cold plan must retain output geometry
        // and admit its complete producers before the first consumer below.
        self.resume_model_evaluation_preparation(lease, &mut original, false)
    }

    /// Identity-only evidence for resuming the original preparation. It does
    /// not admit a new invocation or clear historical Session poison.
    pub fn pending_model_evaluation_preparation_identity(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<Option<SemanticPublishedIdentity>, SemanticTransitionError> {
        self.require_original_cold_content_reader(lease)?
            .evaluation_preparation
            .as_ref()
            .map(|original| {
                original
                    .lock()
                    .map(|original| original.identity)
                    .map_err(|_| {
                        publication_input_error("evaluation preparation owner is poisoned")
                    })
            })
            .transpose()
    }

    fn resume_model_evaluation_preparation(
        &mut self,
        lease: &SemanticPublishedLease,
        original: &mut EvaluationPreparation,
        submit: bool,
    ) -> Result<SemanticModelEvaluation, SemanticTransitionError> {
        if original.complete {
            let handle = original
                .handle
                .as_ref()
                .expect("completed original evaluation");
            self.original_evaluation(handle)?;
            return Ok(handle.clone());
        }
        let event_capacity = original.event_capacity;
        let arena = self
            .training_views
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let domain = arena.training_domain_identity();
        let task = self
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .0
            .content_identity();
        let device = self.provider.device().ordinal();
        if submit {
            if let Some(allowance) = self.cold_native_allowance(lease.token)? {
                allowance
                    .lock()
                    .map_err(|_| {
                        publication_input_error("original cold native allowance is poisoned")
                    })?
                    .content_ceiling()?;
            }
            original.started = true;
        }
        let cold_work = self.cold_native_work(lease.token)?;
        let allowance = self.cold_native_allowance(lease.token)?;
        if original.cohort.is_none() {
            let (arena, selected_view) = self.training_view_selection_basis(lease)?;
            let commands = [
                OriginalNativeCommand::new(&self.domain)?,
                OriginalNativeCommand::new(&self.domain)?,
            ];
            let selected =
                arena.prepare_selection(self.training_origins.as_ref().map(Arc::clone))?;
            let cells = Arc::new(allocate_publication(&self.provider, 17 * 4)?);
            let mut tensors = Vec::with_capacity(17);
            let mut digests = Vec::with_capacity(17);
            for (index, port) in SemanticTrainingViewPort::ALL.into_iter().enumerate() {
                let (view, shape, strides, dtype) = selected.port(port)?;
                let mut layout = SemanticTensorLayout {
                    role: 0,
                    index: index as u64,
                    element_bytes: u64::from(dtype.1 / 8),
                    scalar_type: match dtype {
                        (1, 64) => 3,
                        (0, 64) => 7,
                        (2, 32) => 6,
                        _ => return Err(SemanticTransitionError::ObservationMismatch),
                    },
                    rank: shape.len() as u64,
                    logical_axis: u64::MAX,
                    dimensions: [0; 4],
                    strides_bytes: [0; 4],
                };
                for (axis, (extent, stride)) in shape.iter().zip(&strides).enumerate() {
                    layout.dimensions[axis] = *extent as u64;
                    layout.strides_bytes[axis] = *stride as u64 * layout.element_bytes;
                }
                tensors.push(PreparedSemanticTensor {
                    layout,
                    logical_begin: 0,
                    logical_end: 0,
                    data: *view.device_ptr(),
                    native_allocation: view.allocation_provenance(),
                    source: Some(view),
                    _empty_owner: None,
                });
                digests.push(CapturedTensorDigest::Tensor {
                    cells: Arc::clone(&cells),
                    offset: index * 4,
                    producer_sealed: false,
                });
            }
            let content = TensorContentBuffers {
                tensors,
                seals: TensorContentSeals::Captured(digests),
                verification_inputs: Vec::new(),
                original_cold: Mutex::new(None),
            };
            original.selection = Some((selected_view, commands));
            original.cohort = Some(Arc::new(SemanticEvaluationCohort {
                issuance: Arc::new(()),
                selected,
                content,
                domain,
                task,
                device,
                provider: Arc::clone(&self.provider),
            }));
        }
        let cohort = Arc::clone(original.cohort.as_ref().expect("retained original cohort"));
        if submit && original.source_material.is_none() {
            if original.source_observation.is_none() {
                original.source_observation =
                    Some(self.prepare_published_state_observation(lease, false)?);
            }
            let source_bytes = self.resolve_published_state_observation(
                lease,
                original
                    .source_observation
                    .as_mut()
                    .expect("retained source observation"),
            )?;
            PublicationMaterial::decode(&source_bytes)?.require_current_model_caches()?;
            original.source_material = Some(Identity256::from_bytes(
                Sha256::digest(&source_bytes).into(),
            ));
            self.steps
                .get_mut(&lease.token)
                .expect("retained original reader")
                .evaluation
                .as_mut()
                .expect("retained original evaluation")
                .source_material = original.source_material;
        }
        if submit && !original.selected {
            let (selected_view, commands) = original
                .selection
                .as_mut()
                .expect("retained selection commands");
            cohort.selected.enqueue_original(
                selected_view.clone(),
                lease.header.training_cursor,
                lease.header.training_rng,
                cold_work.as_ref(),
                commands,
                &mut self.poisoned,
            )?;
            original.selected = true;
        }
        if submit && !original.sealed {
            let seal = self
                .provider
                .device()
                .inner()
                .get_func(
                    "xlog_semantic_transition",
                    "semantic_tensor_content_witness",
                )
                .ok_or_else(|| {
                    runtime_error("kernel lookup", "tensor content witness unavailable")
                })?;
            cohort.content.enqueue_with_custody(
                &self.domain,
                &mut self.poisoned,
                &seal,
                false,
                None,
                false,
                cold_work.as_ref(),
                false,
                allowance.as_ref(),
            )?;
            cohort.content.complete_original_cold_content()?;
            original.sealed = true;
        }
        if original.handle.is_none() {
            let bytes = event_capacity
                .checked_mul(size_of::<ModelWorkEvent>() + 3 * size_of::<u64>())
                .and_then(|bytes| bytes.checked_add(12 * size_of::<u64>()))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let mut reservation = self
                .provider
                .memory()
                .reserve_bytes(bytes as u64)
                .map_err(|error| runtime_error("evaluation work reservation", error))?;
            let work =
                PreparedModelWork::allocate(&self.provider, &mut reservation, event_capacity)?;
            let report = reservation
                .alloc(12)
                .map_err(|error| runtime_error("evaluation report allocation", error))?;
            let handle = SemanticModelEvaluation {
                issuer: Arc::clone(&self.publication_issuer),
                invocation: Arc::new(()),
                token: lease.token,
                cohort: Arc::clone(&cohort),
            };
            self.steps
                .get_mut(&lease.token)
                .expect("checked acquired step")
                .evaluation = Some(EvaluationStorage {
                invocation: Arc::clone(&handle.invocation),
                work_aliases: Arc::new(()),
                cohort,
                work,
                report,
                source_material: None,
                cold_work,
                stream: self.stream.cu_stream() as u64,
                submitted: false,
                report_submitted: false,
                consumer_streams: None,
                result: None,
                execution: EvaluationExecution::Prepared,
                graph: None,
                output_witness: None,
                graph_retirement: None,
                capture_native_work: [0; 9],
                frozen_native_work: None,
                launch_marker_read: None,
                launch_markers: None,
                report_read: None,
                report_words: None,
                source_observation: None,
                resolved_source_material: None,
            });
            original.handle = Some(handle);
        }
        let handle = original
            .handle
            .as_ref()
            .expect("retained original evaluation");
        self.original_evaluation(handle)?;
        if !submit {
            return Ok(handle.clone());
        }
        if !original.guarded {
            self.enqueue_evaluation_cohort(Arc::clone(&handle.cohort), handle.token)?;
            original.guarded = true;
        }
        let work = &self.steps[&lease.token]
            .evaluation
            .as_ref()
            .expect("issued evaluation")
            .work;
        work.reset_slots_original(
            &self.domain,
            &mut self.poisoned,
            0,
            event_capacity,
            &mut original.reset,
        )?;
        original.complete = true;
        Ok(handle.clone())
    }

    pub fn model_evaluation_preparation_is_original(
        &self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
    ) -> Result<bool, SemanticTransitionError> {
        self.checked_original_evaluation_reader(lease, handle)?;
        Ok(self.steps[&lease.token].evaluation_preparation.is_some())
    }

    pub fn model_evaluation_preparation_pending(&self, handle: &SemanticModelEvaluation) -> bool {
        self.original_evaluation(handle).is_ok()
            && self
                .steps
                .get(&handle.token)
                .and_then(|step| step.evaluation_preparation.as_ref())
                .is_some_and(|owner| {
                    owner
                        .lock()
                        .map_or(true, |original| original.started && !original.complete)
                })
    }

    fn resume_original_evaluation_preparation(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_original_evaluation_reader(lease, handle)?;
        let Some(owner) = self.steps[&lease.token].evaluation_preparation.as_ref() else {
            return Ok(());
        };
        let owner = Arc::clone(owner);
        let mut original = owner
            .lock()
            .map_err(|_| publication_input_error("evaluation preparation owner is poisoned"))?;
        let resumed = self.resume_model_evaluation_preparation(lease, &mut original, true)?;
        if !Arc::ptr_eq(&resumed.invocation, &handle.invocation) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub(super) fn require_evaluation_cold_work_origin(
        &self,
        original: &SemanticModelEvaluation,
        reader_token: Option<u64>,
        expected_work: &DeviceMemoryView<u64>,
    ) -> Result<(), SemanticTransitionError> {
        let evaluation = self.evaluation(original)?;
        let actual = evaluation.cold_work.as_ref().ok_or_else(|| {
            publication_input_error("evaluation has no original cold operation allocation")
        })?;
        let actual_owner = actual
            .allocation_provenance()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let expected_owner = expected_work
            .allocation_provenance()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if reader_token.is_some_and(|token| token != original.token)
            || evaluation.execution != EvaluationExecution::Prepared
            || !actual_owner.same_allocation(&expected_owner)
            || actual.device_ptr() != expected_work.device_ptr()
            || actual.len() != expected_work.len()
        {
            return Err(publication_input_error(
                "evaluation cleanup changed its original reader or cold operation allocation",
            ));
        }
        Ok(())
    }

    fn evaluation(
        &self,
        handle: &SemanticModelEvaluation,
    ) -> Result<&EvaluationStorage, SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.original_evaluation(handle)
    }

    /// Identity-only lookup for joining an already retained original operation.
    /// It grants no new submission and never clears Session poison.
    fn original_evaluation(
        &self,
        handle: &SemanticModelEvaluation,
    ) -> Result<&EvaluationStorage, SemanticTransitionError> {
        let evaluation = self
            .steps
            .get(&handle.token)
            .and_then(|step| step.evaluation.as_ref())
            .ok_or_else(|| {
                publication_input_error("evaluation has no retained original invocation")
            })?;
        if !Arc::ptr_eq(&handle.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&handle.invocation, &evaluation.invocation)
            || !Arc::ptr_eq(&handle.cohort, &evaluation.cohort)
        {
            return Err(publication_input_error(
                "evaluation belongs to another Session, invocation or original cohort",
            ));
        }
        Ok(evaluation)
    }

    fn checked_original_evaluation_reader(
        &self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
    ) -> Result<&EvaluationStorage, SemanticTransitionError> {
        if self.pending {
            return Err(SemanticTransitionError::OverlappingLaunch);
        }
        if !lease.active
            || !Arc::ptr_eq(&lease.issuer, &self.publication_issuer)
            || lease.token != handle.token
            || !self.readers.contains_key(&lease.token)
        {
            return Err(publication_input_error(
                "evaluation resolution lost its original live reader",
            ));
        }
        self.original_evaluation(handle)
    }

    fn guard_evaluation_cohort(
        &mut self,
        handle: &SemanticModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        let cohort = Arc::clone(&self.evaluation(handle)?.cohort);
        self.enqueue_evaluation_cohort(cohort, handle.token)
    }

    fn enqueue_evaluation_cohort(
        &mut self,
        cohort: Arc<SemanticEvaluationCohort>,
        token: u64,
    ) -> Result<(), SemanticTransitionError> {
        let cold_work = self.cold_native_work(token)?;
        let allowance = self.cold_native_allowance(token)?;
        let seal = self
            .provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_tensor_content_witness",
            )
            .ok_or_else(|| runtime_error("kernel lookup", "tensor content witness unavailable"))?;
        cohort.content.enqueue_with_custody(
            &self.domain,
            &mut self.poisoned,
            &seal,
            true,
            None,
            false,
            cold_work.as_ref(),
            false,
            allowance.as_ref(),
        )?;
        cohort.content.complete_original_cold_content()
    }

    pub fn evaluation_stream(
        &self,
        handle: &SemanticModelEvaluation,
    ) -> Result<u64, SemanticTransitionError> {
        Ok(self.evaluation(handle)?.stream)
    }

    pub fn evaluation_training_view_port(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
        port: SemanticTrainingViewPort,
        stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.resume_original_evaluation_preparation(lease, handle)?;
        self.checked_evaluation_parent(lease, handle)?;
        if self.original_evaluation(handle)?.execution != EvaluationExecution::Prepared {
            return Err(publication_input_error(
                "evaluation training ports export only before the original capture",
            ));
        }
        let (view, shape, strides, dtype) = self
            .original_evaluation(handle)?
            .cohort
            .selected
            .port(port)?;
        dlpack_consumer_stream(stream)?;
        let guard = Arc::clone(&self.require_original_cold_content_reader(lease)?.aliases);
        self.steps
            .get_mut(&lease.token)
            .expect("original evaluation reader")
            .consumer_streams
            .insert(stream);
        self.export_owned_view(view, shape, strides, dtype, guard, stream, None)
    }

    fn checked_evaluation_parent(
        &self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        let evaluation = if self
            .steps
            .get(&lease.token)
            .is_some_and(|step| step.evaluation_preparation.is_some())
        {
            self.checked_original_evaluation_reader(lease, handle)?
        } else {
            self.checked_reader(lease)?;
            self.evaluation(handle)?
        };
        if lease.token != handle.token || evaluation.submitted || evaluation.result.is_some() {
            return Err(publication_input_error(
                "evaluation is submitted, closed or belongs to another acquired parent",
            ));
        }
        Ok(())
    }

    pub fn evaluation_model_work_buffer(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
        stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.resume_original_evaluation_preparation(lease, handle)?;
        self.checked_evaluation_parent(lease, handle)?;
        let evaluation = self.original_evaluation(handle)?;
        if stream != evaluation.stream
            || evaluation.execution != EvaluationExecution::Prepared
            || evaluation.submitted
        {
            return Err(publication_input_error(
                "evaluation work scratch requires its cold original stream",
            ));
        }
        // SAFETY: the entire padding-free bank was initialized by the cold reset.
        let view = unsafe { evaluation.work.actual.view().cast::<u8>() }
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let capacity = (evaluation.work.actual.len() / 3) as i64;
        let scratch = Arc::clone(&evaluation.work_aliases);
        let reader = Arc::clone(&self.require_original_cold_content_reader(lease)?.aliases);
        self.export_owned_view(
            view,
            vec![capacity, 3],
            vec![3, 1],
            (1, 64),
            reader,
            stream,
            Some(scratch),
        )
    }

    pub fn begin_model_evaluation_capture(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
        witness: &SemanticTensorContentWitness,
        streams: &[u64],
    ) -> Result<SemanticModelEvaluationCapture, SemanticTransitionError> {
        self.resume_original_evaluation_preparation(lease, handle)?;
        self.checked_evaluation_parent(lease, handle)?;
        self.checked_content_witness(lease, witness)?;
        let streams = self
            .step_consumer_streams(lease.token, streams)?
            .into_iter()
            .collect::<Vec<_>>();
        let evaluation = self.evaluation(handle)?;
        if evaluation.execution != EvaluationExecution::Prepared {
            return Err(publication_input_error(
                "evaluation captures its original numerical body exactly once",
            ));
        }
        self.guard_evaluation_cohort(handle)?;
        // All original input handoffs precede driver capture. No callback may
        // substitute another stream after this roster is retained.
        self.complete_step_consumers(lease, &streams)?;
        let evaluation = self
            .steps
            .get_mut(&handle.token)
            .expect("checked invocation")
            .evaluation
            .as_mut()
            .expect("checked evaluation");
        evaluation
            .work
            .begin_capture(0, 0)
            .map_err(publication_input_error)?;
        evaluation.execution = EvaluationExecution::Recording;
        evaluation.output_witness = Some(SemanticTensorContentWitness {
            issuer: Arc::clone(&witness.issuer),
            reader_token: witness.reader_token,
            index: witness.index,
            _witness: Arc::clone(&witness._witness),
        });
        evaluation.consumer_streams = Some(streams);
        self.steps
            .get_mut(&handle.token)
            .expect("checked invocation")
            .evaluation_preparation = None;
        Ok(SemanticModelEvaluationCapture {
            invocation: Arc::clone(&handle.invocation),
            stream: Arc::clone(&self.stream),
        })
    }

    pub(super) fn evaluation_content_capture_active(&self, token: u64) -> bool {
        self.steps
            .get(&token)
            .and_then(|step| step.evaluation.as_ref())
            .is_some_and(|evaluation| evaluation.execution == EvaluationExecution::Recording)
    }

    /// The same content producer as ordinary publication witnesses, with the
    /// original capture's already ordered stream and allocation owners.
    pub(super) fn record_evaluation_content_witness(
        &mut self,
        witness: &SemanticTensorContentWitness,
        consumer_stream: u64,
        verify: bool,
    ) -> Result<(), SemanticTransitionError> {
        let original = self.steps.get(&witness.reader_token).ok_or_else(|| {
            publication_input_error("evaluation content lost its original reader")
        })?;
        if !Arc::ptr_eq(&witness.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&witness._witness, &original.content_witnesses)
            || witness.index >= original.content.len()
        {
            return Err(publication_input_error(
                "evaluation content changed its original witness issuer, reader or index",
            ));
        }
        let evaluation = self
            .steps
            .get(&witness.reader_token)
            .and_then(|step| step.evaluation.as_ref())
            .filter(|evaluation| evaluation.execution == EvaluationExecution::Recording)
            .ok_or_else(|| {
                publication_input_error(
                    "content recording requires its original evaluation capture",
                )
            })?;
        if consumer_stream != evaluation.stream {
            return Err(publication_input_error(
                "evaluation capture cannot change its original numerical stream",
            ));
        }
        let native =
            self.steps[&witness.reader_token].content[witness.index].native_work_ceiling(verify)?;
        let evaluation = self
            .steps
            .get_mut(&witness.reader_token)
            .expect("checked original reader")
            .evaluation
            .as_mut()
            .expect("checked original recording");
        super::native_work_bound::add_native_work_ceiling(
            &mut evaluation.capture_native_work,
            native,
        )?;
        let execute = self
            .provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_tensor_content_witness",
            )
            .ok_or_else(|| runtime_error("kernel lookup", "tensor content witness unavailable"))?;
        let cold_work = self.cold_native_work(witness.reader_token)?;
        let allowance = self.cold_native_allowance(witness.reader_token)?;
        self.steps[&witness.reader_token].content[witness.index].enqueue_with_custody(
            &self.domain,
            &mut self.poisoned,
            &execute,
            verify,
            None,
            false,
            cold_work.as_ref(),
            true,
            allowance.as_ref(),
        )
    }

    pub fn record_model_evaluation_output(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
        witness: &SemanticTensorContentWitness,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_evaluation_parent(lease, handle)?;
        self.checked_content_witness(lease, witness)?;
        let evaluation = self.evaluation(handle)?;
        if evaluation
            .output_witness
            .as_ref()
            .map(|original| original.index)
            != Some(witness.index)
        {
            return Err(publication_input_error(
                "evaluation changed its original captured output witness",
            ));
        }
        self.record_evaluation_content_witness(witness, evaluation.stream, false)
    }

    pub fn finish_model_evaluation_capture(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
        captured: SemanticCapturedModelEvaluation,
    ) -> Result<[u64; 3], SemanticTransitionError> {
        self.checked_evaluation_parent(lease, handle)?;
        if !Arc::ptr_eq(&captured.invocation, &handle.invocation) {
            return Err(publication_input_error(
                "evaluation graph belongs to another original invocation",
            ));
        }
        let evaluation = self
            .steps
            .get_mut(&handle.token)
            .expect("checked invocation")
            .evaluation
            .as_mut()
            .expect("checked evaluation");
        if evaluation.execution != EvaluationExecution::Recording || evaluation.graph.is_some() {
            return Err(publication_input_error(
                "evaluation cannot replace or repeat its original capture",
            ));
        }
        // Retain the graph before any fallible descriptor finalization.
        evaluation.graph = Some(Arc::new(captured.graph));
        evaluation.execution = EvaluationExecution::CaptureFinalizing;
        evaluation
            .work
            .finish_capture(0)
            .map_err(publication_input_error)?;
        let work = &evaluation.work;
        let mut destination = work.device.view().slice(..work.recording.events().len());
        self.provider
            .htod_launch_metadata_sync_copy_into(work.recording.events(), &mut destination)
            .map_err(|error| runtime_error("evaluation work metadata upload", error))?;
        evaluation.frozen_native_work = Some(evaluation.capture_native_work);
        evaluation.execution = EvaluationExecution::Captured;
        self.model_evaluation_capture_quantities(handle)
    }

    pub fn model_evaluation_capture_quantities(
        &self,
        handle: &SemanticModelEvaluation,
    ) -> Result<[u64; 3], SemanticTransitionError> {
        let evaluation = self.evaluation(handle)?;
        let bound = evaluation.work.recording.frozen_bound().ok_or_else(|| {
            publication_input_error("evaluation requires its original frozen capture")
        })?;
        let events = evaluation.work.recording.events();
        Ok([
            bound,
            events.len() as u64,
            events
                .iter()
                .filter(|event| event.kind == ModelWorkKind::ModelInvocation as u64)
                .count() as u64,
        ])
    }

    /// Native producer ceiling of this same frozen graph, not its cold tail.
    pub fn model_evaluation_capture_native_work_bound(
        &self,
        handle: &SemanticModelEvaluation,
    ) -> Result<u64, SemanticTransitionError> {
        let frozen = self.evaluation(handle)?.frozen_native_work.ok_or_else(|| {
            publication_input_error("evaluation requires its original frozen native capture")
        })?;
        super::native_work_bound::native_work_ceiling_units(frozen)
    }

    pub fn launch_model_evaluation(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_evaluation_parent(lease, handle)?;
        let evaluation = self.evaluation(handle)?;
        if evaluation.execution != EvaluationExecution::Captured {
            return Err(publication_input_error(
                "unknown evaluation graph entry cannot replay or cancel its original submission",
            ));
        }
        let graph = Arc::clone(evaluation.graph.as_ref().ok_or_else(|| {
            publication_input_error("evaluation lost its original captured graph")
        })?);
        self.steps
            .get_mut(&handle.token)
            .expect("checked invocation")
            .evaluation
            .as_mut()
            .expect("checked evaluation")
            .execution = EvaluationExecution::LaunchEntered;
        graph
            .launch(&self.stream)
            .map_err(|error| runtime_error("evaluation graph submission", error))?;
        self.steps
            .get_mut(&handle.token)
            .expect("checked invocation")
            .evaluation
            .as_mut()
            .expect("checked evaluation")
            .execution = EvaluationExecution::Launched;
        self.resolve_model_evaluation_launch(lease, handle)
    }

    pub fn resolve_model_evaluation_launch(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        let evaluation = self.checked_original_evaluation_reader(lease, handle)?;
        if evaluation.submitted || evaluation.result.is_some() {
            return Err(publication_input_error(
                "evaluation launch resolution cannot reopen its submitted result",
            ));
        }
        if evaluation.execution == EvaluationExecution::Completed {
            return Ok(());
        }
        if !matches!(
            evaluation.execution,
            EvaluationExecution::LaunchEntered | EvaluationExecution::Launched
        ) {
            return Err(publication_input_error(
                "evaluation launch resolution requires its same entered submission",
            ));
        }
        let uncertain = evaluation.execution == EvaluationExecution::LaunchEntered;
        let streams = evaluation
            .consumer_streams
            .as_ref()
            .ok_or_else(|| {
                publication_input_error("evaluation lost its original capture consumer roster")
            })?
            .clone();
        if !self.is_poisoned() {
            self.complete_step_consumers(lease, &streams)?;
        } else if self
            .original_evaluation(handle)?
            .launch_marker_read
            .is_none()
        {
            return Err(SemanticTransitionError::Poisoned);
        }
        if uncertain {
            // A successful join is not proof that an errored graph submission
            // entered. Only this original capture's real invocation markers
            // can prove execution; zero markers remain unknown, never replay.
            let (actual, invocations) = {
                let evaluation = self.original_evaluation(handle)?;
                (
                    evaluation.work.actual.view(),
                    evaluation
                        .work
                        .recording
                        .events()
                        .iter()
                        .enumerate()
                        .filter_map(|(slot, event)| {
                            (event.kind == ModelWorkKind::ModelInvocation as u64).then_some(slot)
                        })
                        .collect::<Vec<_>>(),
                )
            };
            if invocations.is_empty() {
                return Err(publication_input_error(
                    "evaluation launch resolution lost its original model invocation roster",
                ));
            }
            if self.original_evaluation(handle)?.launch_markers.is_none() {
                if self
                    .original_evaluation(handle)?
                    .launch_marker_read
                    .is_none()
                {
                    let read = self.prepare_publication_read(actual)?;
                    self.steps
                        .get_mut(&handle.token)
                        .expect("checked invocation")
                        .evaluation
                        .as_mut()
                        .expect("checked evaluation")
                        .launch_marker_read = Some(read);
                }
                let mut read = self
                    .steps
                    .get_mut(&handle.token)
                    .expect("checked invocation")
                    .evaluation
                    .as_mut()
                    .expect("checked evaluation")
                    .launch_marker_read
                    .take()
                    .expect("retained original marker copy");
                let result = if !read.read.entered() {
                    self.submit_publication_read(&mut read)
                        .and_then(|()| self.resolve_publication_read(&mut read))
                } else {
                    self.resolve_publication_read(&mut read)
                };
                self.steps
                    .get_mut(&handle.token)
                    .expect("checked invocation")
                    .evaluation
                    .as_mut()
                    .expect("checked evaluation")
                    .launch_marker_read = Some(read);
                let words = result?;
                self.steps
                    .get_mut(&handle.token)
                    .expect("checked invocation")
                    .evaluation
                    .as_mut()
                    .expect("checked evaluation")
                    .launch_markers = Some(words);
            }
            let actual = self
                .original_evaluation(handle)?
                .launch_markers
                .as_ref()
                .expect("original marker observation");
            for slot in invocations {
                if actual.get(slot * 3..slot * 3 + 3) != Some(&[1, 0, 1][..]) {
                    return Err(publication_input_error("unknown evaluation graph submission lacks its original completed invocation markers"));
                }
            }
        }
        self.steps
            .get_mut(&handle.token)
            .expect("checked invocation")
            .evaluation
            .as_mut()
            .expect("checked evaluation")
            .execution = EvaluationExecution::Completed;
        Ok(())
    }

    pub fn model_evaluation_launch_pending(&self, handle: &SemanticModelEvaluation) -> bool {
        !self.is_poisoned()
            && self.evaluation(handle).is_ok_and(|evaluation| {
                matches!(
                    evaluation.execution,
                    EvaluationExecution::LaunchEntered | EvaluationExecution::Launched
                )
            })
    }

    pub fn require_model_evaluation_launch_completed(
        &self,
        handle: &SemanticModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        if self.evaluation(handle)?.execution != EvaluationExecution::Completed {
            return Err(publication_input_error(
                "evaluation output projection requires its original known-completed graph launch",
            ));
        }
        Ok(())
    }

    fn retire_model_evaluation_graph(
        &mut self,
        handle: &SemanticModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        let evaluation = self
            .steps
            .get_mut(&handle.token)
            .expect("checked invocation")
            .evaluation
            .as_mut()
            .expect("checked evaluation");
        if let Some(graph) = evaluation.graph.take() {
            evaluation.graph_retirement = Some(graph.retirement());
            drop(graph);
        }
        crate::cuda_graph::reap_capture_retirements();
        if let Some(retirement) = &evaluation.graph_retirement {
            retirement
                .require_completed()
                .map_err(|error| runtime_error("evaluation graph retirement", error))?;
        }
        Ok(())
    }

    pub fn record_evaluation_model_work(
        &mut self,
        handle: &SemanticModelEvaluation,
        kind: ModelWorkKind,
        dimensions: &[u64],
        device_produced: bool,
    ) -> Result<usize, SemanticTransitionError> {
        if self.evaluation(handle)?.execution != EvaluationExecution::Recording {
            return Err(publication_input_error(
                "model work requires its original active evaluation capture",
            ));
        }
        let evaluation = self
            .steps
            .get_mut(&handle.token)
            .expect("checked invocation")
            .evaluation
            .as_mut()
            .expect("checked evaluation");
        let work = &mut evaluation.work;
        work.record_operation(
            kind,
            dimensions,
            device_produced,
            &self.domain,
            &mut self.poisoned,
        )
    }

    /// Same original work registrar and stream as the actual read-only model.
    pub fn record_evaluation_model_invocation(
        &mut self,
        handle: &SemanticModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        if self.evaluation(handle)?.execution != EvaluationExecution::Recording {
            return Err(publication_input_error(
                "model invocation requires its original active evaluation capture",
            ));
        }
        let result = self
            .steps
            .get_mut(&handle.token)
            .expect("checked invocation")
            .evaluation
            .as_mut()
            .expect("checked evaluation")
            .work
            .record_invocation(&self.domain, &mut self.poisoned);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// Freeze the same registrar, validate its genuine output witness, then
    /// observe only after stream completion and exact full-publication recheck.
    /// A failed completion retains every buffer in the Session and quarantines it.
    pub fn finish_model_evaluation(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
        witness: &SemanticTensorContentWitness,
        streams: &[u64],
    ) -> Result<SemanticCompletedModelEvaluation, SemanticTransitionError> {
        self.checked_evaluation_parent(lease, handle)?;
        let streams = self
            .step_consumer_streams(lease.token, streams)?
            .into_iter()
            .collect::<Vec<_>>();
        self.checked_content_witness(lease, witness)?;
        let original = self.evaluation(handle)?;
        if original
            .output_witness
            .as_ref()
            .map(|original| original.index)
            != Some(witness.index)
            || original.consumer_streams.as_ref() != Some(&streams)
            || original.submitted
            || original.work.recording.frozen_bound().is_none()
            || original.execution != EvaluationExecution::Completed
        {
            return Err(publication_input_error("evaluation finish requires the same frozen capture, original output and consumer roster"));
        }
        let output = &self.steps[&lease.token].content[witness.index];
        if output.tensors.len() != 3 {
            return Err(publication_input_error("evaluation output witness requires six FP32 losses, one Bool admissibility and the original UInt8 slab"));
        }
        let losses = output.tensors[0].clone();
        let admissible = output.tensors[1].clone();
        let slab = &output.tensors[2];
        let vector = |tensor: &PreparedSemanticTensor, scalar, bytes, extent| {
            tensor.layout.role == 0
                && tensor.layout.scalar_type == scalar
                && tensor.layout.element_bytes == bytes
                && tensor.layout.rank == 1
                && tensor.layout.dimensions == [extent, 0, 0, 0]
                && tensor.layout.strides_bytes == [bytes, 0, 0, 0]
                && tensor.layout.logical_axis == u64::MAX
                && tensor.logical_begin == 0
                && tensor.logical_end == 0
        };
        if losses.layout.index != 0
            || admissible.layout.index != 1
            || slab.layout.index != 2
            || !vector(&losses, 6, 4, 6)
            || !vector(&admissible, 8, 1, 1)
            || !vector(slab, 1, 1, slab.layout.dimensions[0])
            || slab.layout.dimensions[0] == 0
        {
            return Err(publication_input_error(
                "evaluation outputs changed their original complete typed geometry",
            ));
        }
        let external_bytes = accounted_tensor_allocations(&[&losses, &admissible], slab)?;
        let slab_bytes = tensor_layout_bytes(&slab.layout)?;
        for allocation in &self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .allocations
        {
            let Some(allocation) = allocation.live_slice() else {
                continue;
            };
            if step_input_overlap(
                slab.data,
                slab_bytes,
                allocation.device_ptr_value(),
                allocation.len(),
            )? {
                return Err(publication_input_error(
                    "evaluation slab aliases original publication storage",
                ));
            }
        }
        let stream = self.evaluation(handle)?.stream;
        self.enqueue_tensor_content(lease, witness, stream, true)?;
        self.guard_evaluation_cohort(handle)?;
        let cohort_provider = &self.evaluation(handle)?.cohort.provider;
        let retained_source_bytes = if Arc::ptr_eq(cohort_provider, &self.provider) {
            0
        } else {
            cohort_provider.memory().allocated_bytes()
        };
        let peak = self
            .provider
            .memory()
            .allocated_bytes()
            .checked_add(retained_source_bytes)
            .and_then(|bytes| bytes.checked_add(external_bytes))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let kernel = self
            .provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_model_evaluation_result",
            )
            .ok_or_else(|| runtime_error("kernel lookup", "model evaluation result unavailable"))?;
        let evaluation = self
            .steps
            .get_mut(&handle.token)
            .expect("checked invocation")
            .evaluation
            .as_mut()
            .expect("checked evaluation");
        if evaluation.execution != EvaluationExecution::Completed || evaluation.submitted {
            return Err(publication_input_error(
                "evaluation result requires its original launched capture",
            ));
        }
        let work = &evaluation.work;
        let input = work.descriptor();
        let selected = evaluation.cohort.selected.selection();
        let mut recorder = self.domain.new_strict_recorder();
        work.record_reads(&mut recorder);
        recorder.read(&selected);
        recorder.read(
            losses
                .source
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
        );
        recorder.read(
            admissible
                .source
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
        );
        recorder.write(&evaluation.report);
        evaluation.submitted = true;
        evaluation.consumer_streams = Some(streams);
        let arguments = (
            input.events,
            input.count,
            input.bound,
            *selected.device_ptr(),
            losses.data,
            admissible.data,
            peak,
            evaluation.report.device_ptr_value(),
        );
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            // SAFETY: all typed inputs and the complete native result are retained
            // by this invocation; the kernel writes exactly twelve U64 words.
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
        // Entry is irreversible even when enqueue fails before its operation
        // closure. Only a successful consuming commit permits report resolution.
        self.steps
            .get_mut(&handle.token)
            .expect("submitted invocation")
            .evaluation
            .as_mut()
            .expect("submitted evaluation")
            .report_submitted = true;
        self.resolve_model_evaluation(lease, handle)
    }

    /// Complete only the already submitted original report. No recording,
    /// output witness, stream roster or result kernel can be submitted again.
    /// Driver failure remains terminal quarantine, never a reset or replay.
    pub fn resolve_model_evaluation(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
    ) -> Result<SemanticCompletedModelEvaluation, SemanticTransitionError> {
        let evaluation = self.checked_original_evaluation_reader(lease, handle)?;
        if lease.token != handle.token || !evaluation.report_submitted {
            return Err(publication_input_error(
                "evaluation resolution requires its original submitted invocation",
            ));
        }
        let parent = lease.identity;
        let result = if let Some(result) = self.original_evaluation(handle)?.result {
            result
        } else {
            let streams = self
                .original_evaluation(handle)?
                .consumer_streams
                .as_ref()
                .ok_or_else(|| {
                    publication_input_error("evaluation lost its original consumer streams")
                })?
                .clone();
            if !self.is_poisoned() {
                self.complete_step_consumers(lease, &streams)?;
            } else if self.original_evaluation(handle)?.report_read.is_none() {
                return Err(SemanticTransitionError::Poisoned);
            }
            let report = self.steps[&handle.token]
                .evaluation
                .as_ref()
                .expect("retained evaluation")
                .report
                .view();
            if self.original_evaluation(handle)?.report_read.is_none() {
                // Both observations must retain their entire original suffix
                // before the first DMA can leave completion unknown.
                let read = self.prepare_publication_read(report)?;
                let observation = self.prepare_published_state_observation(lease, false)?;
                let evaluation = self
                    .steps
                    .get_mut(&handle.token)
                    .expect("retained invocation")
                    .evaluation
                    .as_mut()
                    .expect("retained evaluation");
                evaluation.report_read = Some(read);
                evaluation.source_observation = Some(observation);
            }
            if self.original_evaluation(handle)?.report_words.is_none() {
                let mut read = self
                    .steps
                    .get_mut(&handle.token)
                    .expect("retained invocation")
                    .evaluation
                    .as_mut()
                    .expect("retained evaluation")
                    .report_read
                    .take()
                    .expect("retained original report copy");
                let result = if !read.read.entered() {
                    self.submit_publication_read(&mut read)
                        .and_then(|()| self.resolve_publication_read(&mut read))
                } else {
                    self.resolve_publication_read(&mut read)
                };
                self.steps
                    .get_mut(&handle.token)
                    .expect("retained invocation")
                    .evaluation
                    .as_mut()
                    .expect("retained evaluation")
                    .report_read = Some(read);
                let words: [u64; 12] = result?
                    .try_into()
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
                self.steps
                    .get_mut(&handle.token)
                    .expect("retained invocation")
                    .evaluation
                    .as_mut()
                    .expect("retained evaluation")
                    .report_words = Some(words);
            }
            if self
                .original_evaluation(handle)?
                .resolved_source_material
                .is_none()
            {
                let mut observation = self
                    .steps
                    .get_mut(&handle.token)
                    .expect("retained invocation")
                    .evaluation
                    .as_mut()
                    .expect("retained evaluation")
                    .source_observation
                    .take()
                    .expect("retained original source observation");
                let result = self.resolve_published_state_observation(lease, &mut observation);
                self.steps
                    .get_mut(&handle.token)
                    .expect("retained invocation")
                    .evaluation
                    .as_mut()
                    .expect("retained evaluation")
                    .source_observation = Some(observation);
                let material = Identity256::from_bytes(Sha256::digest(result?).into());
                let evaluation = self
                    .steps
                    .get_mut(&handle.token)
                    .expect("retained invocation")
                    .evaluation
                    .as_mut()
                    .expect("retained evaluation");
                evaluation.resolved_source_material = Some(material);
                evaluation.source_observation = None;
            }
            let evaluation = self
                .steps
                .get_mut(&handle.token)
                .expect("retained invocation")
                .evaluation
                .as_mut()
                .expect("retained evaluation");
            if evaluation.source_material.is_none()
                || evaluation.resolved_source_material != evaluation.source_material
            {
                self.poisoned = true;
                return Err(publication_input_error(
                    "read-only evaluation changed the full original publication",
                ));
            }
            let words = evaluation
                .report_words
                .expect("original completed report observation");
            let result = SemanticModelEvaluationResult {
                status: words[0],
                loss_bits: std::array::from_fn(|index| words[index + 1] as u32),
                model_work: words[7],
                operation_count: words[8],
                work_bound: words[9],
                retained_allocation_bytes: words[10],
                model_calls: words[11],
            };
            evaluation.result = Some(result);
            result
        };
        if !matches!(result.status, 0 | 1) {
            return Err(publication_input_error(
                "incomplete evaluation expenditure retains its original invocation; no completed numerical result is available",
            ));
        }
        self.retire_model_evaluation_graph(handle)?;
        Ok(SemanticCompletedModelEvaluation {
            issuer: Arc::clone(&handle.issuer),
            invocation: Arc::clone(&handle.invocation),
            cohort_issuance: Arc::clone(&handle.cohort.issuance),
            parent,
            result,
        })
    }

    /// A late cold failure may resolve this same submission. A poisoned
    /// Session or a known incomplete work report can never use this handoff.
    pub fn model_evaluation_completion_pending(&self, handle: &SemanticModelEvaluation) -> bool {
        !self.is_poisoned()
            && self.evaluation(handle).is_ok_and(|evaluation| {
                evaluation.report_submitted
                    && evaluation.consumer_streams.is_some()
                    && evaluation
                        .result
                        .is_none_or(|result| matches!(result.status, 0 | 1))
            })
    }

    /// Cancellation joins actual consumers before allowing the original parent
    /// to be reused. Unknown completion never clears the retained invocation.
    pub fn cancel_model_evaluation(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
        streams: &[u64],
    ) -> Result<SemanticCancelledModelEvaluation, SemanticTransitionError> {
        self.checked_evaluation_parent(lease, handle)?;
        let parent = self.published_identity(lease)?;
        let streams = self
            .step_consumer_streams(lease.token, streams)?
            .into_iter()
            .collect::<Vec<_>>();
        let metadata_only = self.steps[&handle.token]
            .evaluation_preparation
            .as_ref()
            .map(|owner| {
                owner.lock().map(|original| !original.started).map_err(|_| {
                    publication_input_error("original evaluation preparation is poisoned")
                })
            })
            .transpose()?
            .unwrap_or(false);
        if metadata_only {
            // This original invocation has issued allocations only. There is
            // no selected content or model graph to verify, and no numerical
            // result is synthesized. Join allocation/consumer lifetime before
            // releasing this exact original non-submission owner.
            self.complete_step_consumers(lease, &streams)?;
            let step = self
                .steps
                .get_mut(&handle.token)
                .expect("original evaluation reader");
            step.evaluation = None;
            step.evaluation_preparation = None;
            return Ok(SemanticCancelledModelEvaluation {
                issuer: Arc::clone(&handle.issuer),
                invocation: Arc::clone(&handle.invocation),
                parent,
                work: 0,
                calls: 0,
            });
        }
        if self.model_evaluation_preparation_pending(handle) {
            return Err(publication_input_error(
                "original evaluation preparation must resolve before cancellation",
            ));
        }
        let evaluation = self.evaluation(handle)?;
        if evaluation.submitted
            || matches!(
                evaluation.execution,
                EvaluationExecution::LaunchEntered
                    | EvaluationExecution::Launched
                    | EvaluationExecution::Completed
            )
        {
            return Err(publication_input_error(
                "entered evaluation launch must resolve its original submission, never cancel",
            ));
        }
        // A successful EndCapture (or an independently verified inactive
        // stream after failed capture), not a callback exception, proves that
        // none of this recording's model producers were submitted.
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("evaluation non-submission admission", error))?;
        self.complete_step_consumers(lease, &streams)?;
        self.guard_evaluation_cohort(handle)?;
        let material =
            Identity256::from_bytes(Sha256::digest(self.published_state_material(lease)?).into());
        if Some(material) != self.evaluation(handle)?.source_material {
            self.poisoned = true;
            return Err(publication_input_error(
                "cancelled evaluation changed the original publication",
            ));
        }
        drop(_ordinary);
        self.retire_model_evaluation_graph(handle)?;
        self.steps
            .get_mut(&handle.token)
            .expect("completed invocation")
            .evaluation = None;
        self.steps
            .get_mut(&handle.token)
            .expect("completed invocation")
            .evaluation_preparation = None;
        Ok(SemanticCancelledModelEvaluation {
            issuer: Arc::clone(&handle.issuer),
            invocation: Arc::clone(&handle.invocation),
            parent,
            // This is a native non-submission proof, not an executed zero-cost
            // observation. Preparation/guard expense stays on the cold owner.
            work: 0,
            calls: 0,
        })
    }

    pub(super) fn require_closed_evaluations(&self) -> Result<(), SemanticTransitionError> {
        if self.steps.values().any(|step| {
            (step.evaluation_preparation.is_some() && step.evaluation.is_none())
                || step.evaluation.as_ref().is_some_and(|evaluation| {
                    evaluation
                        .result
                        .is_none_or(|result| !matches!(result.status, 0 | 1))
                })
        }) {
            return Err(publication_input_error(
                "state-changing work or release cannot interrupt an original read-only evaluation",
            ));
        }
        self.require_completed_cold_model_work()
    }

    /// A later cold operation may release only the original completed temporary
    /// evaluation storage after its actual scratch aliases are gone. Immutable
    /// receipts and the held cohort remain independently owned by the phase.
    pub(super) fn retire_completed_evaluation_storage(
        &mut self,
        lease: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_reader(lease)?;
        if let Some(evaluation) = self.steps[&lease.token].evaluation.as_ref() {
            if evaluation
                .result
                .is_none_or(|result| !matches!(result.status, 0 | 1))
                || Arc::strong_count(&evaluation.work_aliases) != 1
            {
                return Err(publication_input_error(
                    "later cold work cannot retire an unknown evaluation or its live scratch aliases",
                ));
            }
            self.steps
                .get_mut(&lease.token)
                .expect("checked original reader")
                .evaluation = None;
        }
        Ok(())
    }
}
