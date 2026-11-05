//! Native read-only ownership at the existing model content/work boundary.

use super::*;
use xlog_cuda::{
    SemanticCancelledEvaluationCapture, SemanticCancelledModelEvaluation,
    SemanticCompletedModelEvaluation, SemanticEvaluationCohort, SemanticModelEvaluation,
    SemanticModelEvaluationResult,
};

pyo3::create_exception!(
    pyxlog._native,
    SemanticModelEvaluationPending,
    PyRuntimeError,
    "An original evaluation or its cold producer awaits completion; retain its owners and never resubmit."
);

/// Known original non-submission, independent of the temporary capture owners.
#[pyclass(
    name = "SemanticCancelledModelEvaluation",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticCancelledModelEvaluation {
    inner: SemanticCancelledModelEvaluation,
}

#[pymethods]
impl PySemanticCancelledModelEvaluation {
    /// Native recorder entry is not a successful evaluation or Python completion.
    #[getter]
    fn capture_disposition(&self) -> &'static str {
        match self.inner.capture_disposition() {
            SemanticCancelledEvaluationCapture::NotEntered => "not-entered",
            SemanticCancelledEvaluationCapture::EnteredAborted => "entered-aborted",
            SemanticCancelledEvaluationCapture::RecordedAndCancelled => "recorded-and-cancelled",
        }
    }

    /// Authenticate against the actual original invocation, never counts or flags.
    fn belongs_to(&self, py: Python<'_>, original: &PySemanticModelEvaluation) -> PyResult<bool> {
        let parent = original.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        Ok(self.inner.belongs_to(&owner, &original.inner)
            && self.inner.parent()
                == owner
                    .published_identity(&*parent.lease()?)
                    .map_err(xlog_err)?)
    }
}

/// Immutable completed observation, deliberately independent of the temporary
/// evaluation's original model, selected cohort, reader and CUDA allocations.
#[pyclass(
    name = "SemanticCompletedModelEvaluation",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticCompletedModelEvaluation {
    inner: SemanticCompletedModelEvaluation,
    binding: (Option<String>, Vec<u8>),
}

#[pymethods]
impl PySemanticCompletedModelEvaluation {
    #[getter]
    fn status(&self) -> u64 {
        self.inner.result().status
    }

    #[getter]
    fn loss_bits(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.inner.result().loss_bits)?.unbind())
    }

    #[getter]
    fn model_work(&self) -> u64 {
        self.inner.result().model_work
    }

    #[getter]
    fn model_calls(&self) -> u64 {
        self.inner.result().model_calls
    }

    #[getter]
    fn operation_count(&self) -> u64 {
        self.inner.result().operation_count
    }

    #[getter]
    fn work_bound(&self) -> u64 {
        self.inner.result().work_bound
    }

    #[getter]
    fn retained_allocation_bytes(&self) -> u64 {
        self.inner.result().retained_allocation_bytes
    }
}

impl PySemanticCompletedModelEvaluation {
    pub(super) fn original_receipt(&self) -> &SemanticCompletedModelEvaluation {
        &self.inner
    }

    pub(super) fn require_original_observation(
        &self,
        py: Python<'_>,
        parent: &PySemanticPublishedParent,
        cohort: &PySemanticEvaluationCohort,
    ) -> PyResult<SemanticModelEvaluationResult> {
        let session = parent.session.borrow(py);
        let owner = session.owner()?;
        if !self.inner.belongs_to(&owner)
            || !self.inner.belongs_to_cohort(&cohort.inner)
            || self.inner.parent()
                != owner
                    .published_identity(&*parent.lease()?)
                    .map_err(xlog_err)?
            || self.binding != parent.content_binding_with_owner(py, &owner)?
        {
            return Err(invalid(
                "evaluation changed its original completed native observation, parent or cohort",
            ));
        }
        Ok(self.inner.result())
    }
}

/// Retains only the original selected device roster and its native seals.
/// It does not retain a source Runtime, grant or mutable model generation.
#[pyclass(name = "SemanticEvaluationCohort", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticEvaluationCohort {
    pub(super) inner: Arc<SemanticEvaluationCohort>,
}

/// Single original forward/objective recording; no backward or publication.
#[pyclass(name = "SemanticModelEvaluation", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticModelEvaluation {
    pub(super) parent: Py<PySemanticPublishedParent>,
    pub(super) inner: SemanticModelEvaluation,
    cohort: Py<PySemanticEvaluationCohort>,
    pub(super) binding: (Option<String>, Vec<u8>),
    closed: AtomicBool,
    output: Mutex<Option<Py<PySemanticTensorOwner>>>,
    completed: Mutex<Option<Py<PySemanticCompletedModelEvaluation>>>,
    cancelled: Mutex<Option<SemanticCancelledModelEvaluation>>,
    cancel_entered: AtomicBool,
    capture_owners: Mutex<Option<Arc<EvaluationCaptureOwners>>>,
}

struct EvaluationCaptureOwners {
    _enqueue: Py<PyAny>,
    _memory_scope: Py<PyAny>,
    output: Py<PySemanticTensorOwner>,
}

pub(super) struct PendingEvaluationPreparation {
    parent: Py<PySemanticPublishedParent>,
    task_use: Py<PySemanticTransitionTaskUse>,
    requested_cohort: Option<Py<PySemanticEvaluationCohort>>,
    capacity: Option<usize>,
    binding: (Option<String>, Vec<u8>),
    inner: Option<SemanticModelEvaluation>,
    cohort: Option<Py<PySemanticEvaluationCohort>>,
    result: Option<Py<PySemanticModelEvaluation>>,
    phase_entered: bool,
    operation_bound: bool,
}

#[pymethods]
impl PySemanticModelEvaluation {
    #[pyo3(signature = (*, vocabulary, scalar_type=5, block_positions=64))]
    fn prepare_model_forward_stream(
        slf: Py<Self>,
        py: Python<'_>,
        vocabulary: &Bound<'_, PyAny>,
        scalar_type: u64,
        block_positions: u64,
    ) -> PyResult<super::model_forward_stream::PySemanticModelForwardStream> {
        let vocabulary = ColdValue::read(vocabulary, &mut 128, 0)?.unsigned()?;
        let evaluation = slf.borrow(py);
        let parent = evaluation.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let mut owner = session.owner()?;
        evaluation.check(py, &owner)?;
        let inner = owner
            .prepare_evaluation_model_forward_stream(
                &*parent.lease()?,
                &evaluation.inner,
                vocabulary,
                scalar_type,
                block_positions,
            )
            .map_err(xlog_err)?;
        Ok(
            super::model_forward_stream::PySemanticModelForwardStream::from_evaluation(
                inner,
                slf.clone_ref(py),
            ),
        )
    }
}

impl PySemanticModelEvaluation {
    pub(in crate::semantic_transition) fn issue_model_work_buffer_port(
        &self,
        py: Python<'_>,
        stream: u64,
    ) -> PyResult<xlog_cuda::SemanticModelWorkBufferPort> {
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        self.check(py, &owner)?;
        let result = owner
            .evaluation_model_work_buffer_port(&*parent.lease()?, &self.inner, stream)
            .map_err(xlog_err);
        result
    }

    pub(in crate::semantic_transition) fn finalized_model_work_extent(
        &self,
        py: Python<'_>,
        port: &xlog_cuda::SemanticModelWorkBufferPort,
    ) -> PyResult<usize> {
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        self.check(py, &owner)?;
        let result = owner
            .evaluation_model_work_buffer_port_finalized_extent(
                &*parent.lease()?,
                port,
                &self.inner,
            )
            .map_err(xlog_err);
        result
    }

    pub(in crate::semantic_transition) fn export_model_work_buffer_port(
        &self,
        py: Python<'_>,
        port: &xlog_cuda::SemanticModelWorkBufferPort,
    ) -> PyResult<(
        xlog_cuda::SemanticModelWorkBufferExport,
        Py<PySemanticTransitionSession>,
    )> {
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let mut owner = session.owner()?;
        self.check(py, &owner)?;
        let result = owner
            .export_evaluation_model_work_buffer_port(&*parent.lease()?, port, &self.inner)
            .map_err(xlog_err)?;
        Ok((result, parent.session.clone_ref(py)))
    }

    fn phase_cold_boundary(
        &self,
        py: Python<'_>,
        from: learning_phase::phase_evaluation::EvaluationColdStage,
        to: learning_phase::phase_evaluation::EvaluationColdStage,
    ) -> PyResult<()> {
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        if let Some(instruction) = super::evaluation_instruction::current(&session)? {
            instruction.cold_boundary(py, self, from, to)?;
        } else if let Some(original) = private_execution_owner(py, &session)? {
            original
                .borrow(py)
                .evaluation_cold_boundary(py, self, from, to)?;
        }
        Ok(())
    }

    fn completion_error(&self, py: Python<'_>, original: PyErr) -> PyErr {
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        let Ok(owner) = session.owner() else {
            return original;
        };
        if !owner.model_evaluation_completion_pending(&self.inner) {
            return original;
        }
        let pending = SemanticModelEvaluationPending::new_err(
            "original evaluation was submitted; retain its Runtime and resolve_completion without another forward or finish",
        );
        pending.set_cause(py, Some(original));
        pending
    }

    fn launch_error(&self, py: Python<'_>, original: PyErr) -> PyErr {
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        if session
            .owner()
            .is_ok_and(|owner| owner.model_evaluation_launch_pending(&self.inner))
        {
            let pending = SemanticModelEvaluationPending::new_err(
                "original evaluation graph entered submission; retain its owners and resolve_launch without another capture or launch",
            );
            pending.set_cause(py, Some(original));
            pending
        } else {
            original
        }
    }

    fn publish_observation(
        &self,
        py: Python<'_>,
        inner: SemanticCompletedModelEvaluation,
    ) -> PyResult<Py<PySemanticCompletedModelEvaluation>> {
        if !inner.belongs_to_cohort(self.cohort.borrow(py).inner.as_ref()) {
            return Err(invalid(
                "completed evaluation changed its original native cohort",
            ));
        }
        // Allocate and retain the immutable CPU receipt before reopening the
        // original phase. A late failure resolves this same observation, not
        // another recording or another Python result object.
        let observation = Py::new(
            py,
            PySemanticCompletedModelEvaluation {
                inner,
                binding: self.binding.clone(),
            },
        )?;
        *self
            .completed
            .lock()
            .map_err(|_| invalid("evaluation completed observation owner is poisoned"))? =
            Some(observation.clone_ref(py));
        // Native report resolution has joined and retired the graph. Break the
        // callback/lifetime cycle only after its original consumers are known.
        self.capture_owners
            .lock()
            .map_err(|_| invalid("evaluation capture owner is poisoned"))?
            .take();
        self.restore_phase(py)?;
        Ok(observation)
    }

    pub(super) fn check(&self, py: Python<'_>, owner: &SemanticTransitionSession) -> PyResult<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(invalid("read-only evaluation is already closed"));
        }
        let parent = self.parent.borrow(py);
        if owner
            .model_evaluation_preparation_is_original(&*parent.lease()?, &self.inner)
            .map_err(xlog_err)?
        {
            let issued = parent.task_use.borrow(py);
            issued.issuance.require_current()?;
            let state = issued.state()?;
            if owner.task_evaluation_epoch() != issued.task_epoch
                || owner
                    .task_evaluation_identity()
                    .is_none_or(|identity| identity.as_bytes() != &issued.task_identity)
                || !matches!(state.phase, TaskUsePhase::Evaluating(_))
                || state.content_handoff_binding()? != self.binding
            {
                return Err(invalid(
                    "original evaluation preparation changed its task or authority binding",
                ));
            }
            return Ok(());
        }
        if !matches!(
            parent.task_use.borrow(py).state()?.phase,
            TaskUsePhase::Evaluating(_)
        ) || parent.content_binding_with_owner(py, owner)? != self.binding
        {
            return Err(invalid(
                "evaluation changed its original task, operation or authority snapshot",
            ));
        }
        Ok(())
    }

    fn restore_phase(&self, py: Python<'_>) -> PyResult<()> {
        use learning_phase::phase_evaluation::EvaluationColdStage;
        // Only a retained genuine completed receipt admits cleanup. Cancellation
        // restores the ordinary phase but cannot synthesize that receipt.
        if self
            .completed
            .lock()
            .map_err(|_| invalid("evaluation completed observation owner is poisoned"))?
            .is_some()
        {
            self.phase_cold_boundary(
                py,
                EvaluationColdStage::AwaitingCompletion,
                EvaluationColdStage::Cleanup,
            )?;
        }
        let parent = self.parent.borrow(py);
        let issued = parent.task_use.borrow(py);
        let mut state = issued.state()?;
        let phase = std::mem::replace(&mut state.phase, TaskUsePhase::Refused);
        let TaskUsePhase::Evaluating(original) = phase else {
            return Err(invalid(
                "evaluation lost its original phase before completion",
            ));
        };
        state.phase = *original;
        self.closed.store(true, Ordering::Release);
        let session = parent.session.borrow(py);
        let retained_instruction = super::evaluation_instruction::current(&session)?.is_some();
        session
            .recording
            .store(retained_instruction, Ordering::Release);
        Ok(())
    }

    fn export(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
        f: impl FnOnce(
            &mut SemanticTransitionSession,
            &SemanticPublishedLease,
            &SemanticModelEvaluation,
            u64,
        ) -> Result<DlpackManagedTensor, xlog_cuda::SemanticTransitionError>,
    ) -> PyResult<Py<PyAny>> {
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        let mut owner = session.owner()?;
        self.check(py, &owner)?;
        let tensor = match f(&mut owner, &*parent.lease()?, &self.inner, stream) {
            Ok(tensor) => tensor,
            Err(error) if owner.model_evaluation_preparation_pending(&self.inner) => {
                let pending = SemanticModelEvaluationPending::new_err(
                    "original evaluation preparation is pending; retain this evaluation and resume the same buffer or view call");
                pending.set_cause(py, Some(xlog_err(error)));
                return Err(pending);
            }
            Err(error) => return Err(xlog_err(error)),
        };
        let tensor = retain_export_owner(tensor, self.parent.clone_ref(py), session.owner_thread)?;
        crate::dlpack_capsule_from_tensor(py, tensor)
    }
}

impl Drop for PySemanticModelEvaluation {
    fn drop(&mut self) {
        if self.closed.load(Ordering::Acquire) {
            return;
        }
        // Dropping a capsule/owner is not proof of completion. The Session's
        // original step keeps the work/output allocations under quarantine.
        let _ = Python::try_attach(|py| {
            let parent = self.parent.borrow(py);
            let session = parent.session.borrow(py);
            if let Some(owner) = session
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_mut()
            {
                owner.abort();
            }
            if let Ok(mut state) = parent.task_use.borrow(py).state() {
                state.phase = TaskUsePhase::Refused;
            }
            session.recording.store(false, Ordering::Release);
        });
    }
}

#[pymethods]
impl PySemanticModelEvaluation {
    /// Admit the complete original capture recipe before its work-slot export.
    fn admit_plan(
        &self,
        py: Python<'_>,
        operations: &Bound<'_, PyAny>,
    ) -> PyResult<(u64, u64, u64)> {
        if !operations.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "evaluation capture plan requires an exact immutable tuple",
            ));
        }
        let mut recipe = Vec::new();
        let operations = operations.cast::<PyTuple>()?;
        recipe
            .try_reserve_exact(operations.len())
            .map_err(|_| invalid("evaluation capture plan reservation failed"))?;
        for operation in operations.iter() {
            if !operation.is_exact_instance_of::<PyTuple>()
                || operation.cast::<PyTuple>()?.len() != 2
            {
                return Err(invalid(
                    "evaluation capture plan requires exact kind and geometry pairs",
                ));
            }
            let operation = operation.cast::<PyTuple>()?;
            let kind = operation.get_item(0)?;
            let dimensions = operation.get_item(1)?;
            if !dimensions.is_exact_instance_of::<PyTuple>() {
                return Err(invalid(
                    "evaluation capture geometry requires an exact immutable tuple",
                ));
            }
            let (kind, dimensions, rank) = parse_model_work(&kind, &dimensions)?;
            recipe.push((kind, dimensions[..rank].to_vec()));
        }
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let mut owner = session.owner()?;
        self.check(py, &owner)?;
        let quantities = owner
            .admit_model_evaluation_plan(&*parent.lease()?, &self.inner, &recipe)
            .map_err(xlog_err)?;
        Ok((quantities[0], quantities[1], quantities[2]))
    }

    #[getter]
    fn cohort(&self, py: Python<'_>) -> Py<PySemanticEvaluationCohort> {
        self.cohort.clone_ref(py)
    }

    #[getter]
    fn consumer_stream(&self, py: Python<'_>) -> PyResult<u64> {
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        let owner = session.owner()?;
        self.check(py, &owner)?;
        owner.evaluation_stream(&self.inner).map_err(xlog_err)
    }

    /// Allocation-backed cohort geometry without resuming preparation.
    #[pyo3(signature = (*, consumer_stream))]
    fn training_view_layouts(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        self.check(py, &owner)?;
        let original = owner
            .evaluation_training_view_layouts(&*parent.lease()?, &self.inner, stream)
            .map_err(xlog_err)?;
        let mut rows = Vec::with_capacity(original.len());
        for (provenance, shape, strides, dtype) in original {
            rows.push(
                (
                    Py::new(
                        py,
                        PyNativeTensorAllocation {
                            provenance,
                            policy_layout: None,
                        },
                    )?,
                    PyTuple::new(py, shape)?,
                    PyTuple::new(py, strides)?,
                    dtype,
                )
                    .into_pyobject(py)?
                    .unbind(),
            );
        }
        Ok(PyTuple::new(py, rows)?.unbind())
    }

    /// Original capture counter ownership before its complete plan is frozen.
    #[pyo3(signature = (*, consumer_stream))]
    fn model_work_buffer_port(
        slf: Py<Self>,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<super::model_work_port::PySemanticModelWorkBufferPort>> {
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let port = slf.borrow(py).issue_model_work_buffer_port(py, stream)?;
        Py::new(
            py,
            super::model_work_port::PySemanticModelWorkBufferPort::new_evaluation(slf, port),
        )
    }

    /// Original numerical scratch geometry, not the cold-report scratch.
    #[pyo3(signature = (*, consumer_stream))]
    fn model_work_buffer_layout(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        self.check(py, &owner)?;
        let (provenance, shape, strides, dtype) = owner
            .evaluation_model_work_buffer_layout(&*parent.lease()?, &self.inner, stream)
            .map_err(xlog_err)?;
        Ok((
            Py::new(
                py,
                PyNativeTensorAllocation {
                    provenance,
                    policy_layout: None,
                },
            )?,
            PyTuple::new(py, shape)?,
            PyTuple::new(py, strides)?,
            dtype,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// All seventeen original ports, unchanged for source and final models.
    #[pyo3(signature = (*, consumer_stream))]
    fn training_view(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let mut ports = Vec::with_capacity(17);
        for port in SemanticTrainingViewPort::ALL {
            ports.push(
                self.export(py, consumer_stream, |owner, parent, handle, stream| {
                    owner.evaluation_training_view_port(parent, handle, port, stream)
                })?,
            );
        }
        Ok(PyTuple::new(py, ports)?.unbind())
    }

    /// Same native U64[capacity,3] scratch and registrar as prepared execution.
    #[pyo3(signature = (*, consumer_stream))]
    fn model_work_buffer(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.export(
            py,
            consumer_stream,
            SemanticTransitionSession::evaluation_model_work_buffer,
        )
    }

    /// Capture the one original forward/objective body, never launch it.
    #[pyo3(signature = (enqueue, *, memory_scope, output_owner, consumer_streams))]
    fn capture(
        &self,
        py: Python<'_>,
        enqueue: Py<PyAny>,
        memory_scope: Py<PyAny>,
        output_owner: Py<PySemanticTensorOwner>,
        consumer_streams: &Bound<'_, PyAny>,
    ) -> PyResult<(u64, u64, u64)> {
        let _operation =
            TensorProducerOperation::begin(&self.parent.borrow(py).session.borrow(py))?;
        let streams = ColdValue::read(consumer_streams, &mut 4096, 0)?;
        let streams = streams
            .sequence()?
            .iter()
            .map(ColdValue::unsigned)
            .collect::<PyResult<Vec<_>>>()?;
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        self.check(py, &*session.owner()?)?;
        let tensor_owner = output_owner.borrow(py);
        if tensor_owner.session.as_ptr() != parent.session.as_ptr()
            || !matches!(&tensor_owner.parent, ContentStepOwner::Published(original)
                if original.as_ptr() == self.parent.as_ptr())
        {
            return Err(invalid(
                "evaluation capture requires its exact original published output owner",
            ));
        }
        let resources = Arc::new(EvaluationCaptureOwners {
            _enqueue: enqueue.clone_ref(py),
            _memory_scope: memory_scope.clone_ref(py),
            output: output_owner.clone_ref(py),
        });
        {
            let mut retained = self
                .capture_owners
                .lock()
                .map_err(|_| invalid("evaluation capture owner is poisoned"))?;
            if retained.is_some() {
                return Err(invalid(
                    "evaluation cannot replace or repeat its original capture callback",
                ));
            }
            *retained = Some(Arc::clone(&resources));
        }
        use learning_phase::phase_evaluation::EvaluationColdStage;
        self.phase_cold_boundary(
            py,
            EvaluationColdStage::Preparation,
            EvaluationColdStage::OutputProjection,
        )?;
        let check = || {
            session.require_creator()?;
            self.check(py, &*session.owner()?)
        };
        let mut memory = PreparedMemoryScope::new(py, memory_scope.bind(py), &check)?;
        recording_callback(&check, || {
            memory_scope.bind(py).getattr("__enter__")?.call0()?;
            memory.entered = true;
            Ok(())
        })?;
        let recorded = (|| {
            let capture = {
                let mut owner = session.owner()?;
                self.check(py, &owner)?;
                owner
                    .begin_model_evaluation_capture(
                        &*parent.lease()?,
                        &self.inner,
                        &tensor_owner.inner,
                        &streams,
                    )
                    .map_err(xlog_err)?
            };
            let callback_error = std::cell::RefCell::new(None);
            let external: Arc<dyn Send + Sync> = resources;
            let captured = capture.record(vec![external], || {
                let result = (|| {
                    recording_callback(&check, || {
                        if !enqueue.bind(py).call0()?.is_none() {
                            return Err(invalid("evaluation enqueue must retain its original outputs and return None"));
                        }
                        Ok(())
                    })?;
                    session.owner()?.record_model_evaluation_output(
                        &*parent.lease()?, &self.inner, &tensor_owner.inner,
                    ).map_err(xlog_err)
                })();
                if let Err(error) = result {
                    *callback_error.borrow_mut() = Some(error);
                    return Err(xlog_core::error::XlogError::Kernel(
                        "original evaluation capture callback failed".into(),
                    ));
                }
                Ok(())
            }).map_err(|error| callback_error.into_inner().unwrap_or_else(|| xlog_err(error)))?;
            let quantities = session
                .owner()?
                .finish_model_evaluation_capture(&*parent.lease()?, &self.inner, captured)
                .map_err(xlog_err)?;
            Ok((quantities[0], quantities[1], quantities[2]))
        })();
        let cleanup = memory.finish(recorded.as_ref().err());
        finish_with_cleanup(py, recorded, cleanup)
    }

    /// Admit the retained original capture, enqueue it once and join it.
    fn launch(&self, py: Python<'_>) -> PyResult<()> {
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        let (quantities, native_work) = {
            let owner = session.owner()?;
            self.check(py, &owner)?;
            (
                owner
                    .model_evaluation_capture_quantities(&self.inner)
                    .map_err(xlog_err)?,
                owner
                    .model_evaluation_capture_native_work_bound(&self.inner)
                    .map_err(xlog_err)?,
            )
        };
        let phase = private_execution_owner(py, &session)?.ok_or_else(|| {
            invalid("evaluation launch requires its original signed phase admission")
        })?;
        phase
            .borrow(py)
            .admit_evaluation_capture(py, self, quantities, native_work)?;
        let result = {
            let mut owner = session.owner()?;
            owner
                .launch_model_evaluation(&*parent.lease()?, &self.inner)
                .map_err(xlog_err)
        };
        result.map_err(|error| self.launch_error(py, error))
    }

    /// Join only the same entered graph; no new launch or recording occurs.
    fn resolve_launch(&self, py: Python<'_>) -> PyResult<()> {
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let result = {
            let mut owner = session.owner()?;
            self.check(py, &owner)?;
            owner
                .resolve_model_evaluation_launch(&*parent.lease()?, &self.inner)
                .map_err(xlog_err)
        };
        result.map_err(|error| self.launch_error(py, error))
    }

    /// Record on the original invocation stream at the real model forward site.
    fn record_model_invocation(&self, py: Python<'_>) -> PyResult<()> {
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        let mut owner = session.owner()?;
        self.check(py, &owner)?;
        owner
            .record_evaluation_model_invocation(&self.inner)
            .map_err(xlog_err)
    }

    fn record_model_work(
        &self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        dimensions: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let (kind, values, rank) = parse_model_work(kind, dimensions)?;
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        let mut owner = session.owner()?;
        self.check(py, &owner)?;
        owner
            .record_evaluation_model_work(&self.inner, kind, &values[..rank], false)
            .map(|_| ())
            .map_err(xlog_err)
    }

    fn record_model_device_work(
        &self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        upper_dimensions: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let (kind, values, rank) = parse_model_work(kind, upper_dimensions)?;
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        let mut owner = session.owner()?;
        self.check(py, &owner)?;
        owner
            .record_evaluation_model_work(&self.inner, kind, &values[..rank], true)
            .map_err(xlog_err)
    }

    /// Output ownership belongs to the original PublishedParent: three tensor
    /// rows in order, FP32[6], Bool[1], UInt8[slab_bytes]. No Python loss, work
    /// or memory total is accepted. This is cold completion, outside residency.
    #[pyo3(signature = (output_owner, *, consumer_streams))]
    fn finish(
        &self,
        py: Python<'_>,
        output_owner: Py<PySemanticTensorOwner>,
        consumer_streams: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PySemanticCompletedModelEvaluation>> {
        let streams = ColdValue::read(consumer_streams, &mut 4096, 0)?;
        let streams = streams
            .sequence()?
            .iter()
            .map(ColdValue::unsigned)
            .collect::<PyResult<Vec<_>>>()?;
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        let tensor_owner = output_owner.borrow(py);
        if tensor_owner.session.as_ptr() != parent.session.as_ptr()
            || !matches!(&tensor_owner.parent, ContentStepOwner::Published(original) if original.as_ptr() == self.parent.as_ptr())
        {
            return Err(invalid(
                "evaluation outputs require the exact original published-parent content owner",
            ));
        }
        {
            let owner = session.owner()?;
            self.check(py, &owner)?;
            owner
                .require_model_evaluation_launch_completed(&self.inner)
                .map_err(xlog_err)?;
        }
        let mut retained = self
            .output
            .lock()
            .map_err(|_| invalid("evaluation output owner is poisoned"))?;
        if retained.is_some() {
            return Err(invalid("evaluation output was already submitted"));
        }
        if self
            .capture_owners
            .lock()
            .map_err(|_| invalid("evaluation capture owner is poisoned"))?
            .as_ref()
            .is_none_or(|original| original.output.as_ptr() != output_owner.as_ptr())
        {
            return Err(invalid(
                "evaluation finish cannot substitute its captured output owner",
            ));
        }
        *retained = Some(output_owner.clone_ref(py));
        drop(retained);
        if let Some(instruction) = super::evaluation_instruction::current(&session)? {
            instruction.retain_consumer_streams(py, self, &streams)?;
        }
        use learning_phase::phase_evaluation::EvaluationColdStage;
        self.phase_cold_boundary(
            py,
            EvaluationColdStage::OutputProjection,
            EvaluationColdStage::AwaitingCompletion,
        )?;
        let mut owner = session.owner()?;
        self.check(py, &owner)?;
        let result = owner
            .finish_model_evaluation(
                &*parent.lease()?,
                &self.inner,
                &tensor_owner.inner,
                &streams,
            )
            .map_err(xlog_err);
        drop(owner);
        result
            .and_then(|inner| self.publish_observation(py, inner))
            .map_err(|original| self.completion_error(py, original))
    }

    /// Resolve only this original submitted result. No new output, stream,
    /// forward, recording or result kernel is accepted or executed here.
    fn resolve_completion(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<PySemanticCompletedModelEvaluation>> {
        let result = (|| {
            let parent = self.parent.borrow(py);
            let session = parent.session.borrow(py);
            session.require_creator()?;
            let observed = self
                .completed
                .lock()
                .map_err(|_| invalid("evaluation completed observation owner is poisoned"))?
                .as_ref()
                .map(|original| original.clone_ref(py));
            if let Some(observed) = observed {
                if !self.closed.load(Ordering::Acquire) {
                    let owner = session.owner()?;
                    self.check(py, &owner)?;
                    drop(owner);
                    self.restore_phase(py)?;
                }
                return Ok(observed);
            }
            let mut owner = session.owner()?;
            self.check(py, &owner)?;
            let inner = owner
                .resolve_model_evaluation(&*parent.lease()?, &self.inner)
                .map_err(xlog_err)?;
            drop(owner);
            self.publish_observation(py, inner)
        })();
        result.map_err(|original| self.completion_error(py, original))
    }

    /// Rejoin consumers and revalidate full source before restoring its phase.
    /// Failure/unknown completion leaves this owner and Session quarantined.
    #[pyo3(signature = (*, consumer_streams))]
    fn cancel(&self, py: Python<'_>, consumer_streams: &Bound<'_, PyAny>) -> PyResult<()> {
        let streams = ColdValue::read(consumer_streams, &mut 4096, 0)?;
        let streams = streams
            .sequence()?
            .iter()
            .map(ColdValue::unsigned)
            .collect::<PyResult<Vec<_>>>()?;
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        let mut owner = session.owner()?;
        self.check(py, &owner)?;
        if owner.model_evaluation_preparation_pending(&self.inner) {
            return Err(SemanticModelEvaluationPending::new_err(
                "original evaluation preparation must resolve through the same buffer or view before cancellation"));
        }
        if self
            .output
            .lock()
            .map_err(|_| invalid("evaluation output owner is poisoned"))?
            .is_some()
        {
            return Err(invalid(
                "submitted evaluation must resolve its original result, not cancel",
            ));
        }
        self.cancel_entered
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                invalid("unknown evaluation cancellation cannot repeat its original invocation")
            })?;
        if let Some(instruction) = super::evaluation_instruction::current(&session)? {
            instruction.retain_consumer_streams(py, self, &streams)?;
        }
        let cancelled = owner
            .cancel_model_evaluation(&*parent.lease()?, &self.inner, &streams)
            .map_err(xlog_err)?;
        *self
            .cancelled
            .lock()
            .map_err(|_| invalid("evaluation cancellation custody is poisoned"))? =
            Some(cancelled.clone());
        drop(owner);
        self.capture_owners
            .lock()
            .map_err(|_| invalid("evaluation capture owner is poisoned"))?
            .take();
        if let Some(instruction) = super::evaluation_instruction::current(&session)? {
            instruction.cancel_evaluation(py, self, &cancelled)?;
        }
        if let Some(original) = private_execution_owner(py, &session)? {
            original.borrow(py).cancel_phase_evaluation(
                py,
                self,
                &cancelled,
                &self.inner,
                &parent,
            )?;
        }
        self.restore_phase(py)
    }

    /// Project only the retained proof of a known original native cancellation.
    fn cancellation_proof(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<PySemanticCancelledModelEvaluation>> {
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        let retained = self
            .cancelled
            .lock()
            .map_err(|_| invalid("evaluation cancellation custody is poisoned"))?;
        let original = retained
            .as_ref()
            .ok_or_else(|| invalid("evaluation has no known original cancellation proof"))?;
        if !original.belongs_to(&owner, &self.inner)
            || original.parent()
                != owner
                    .published_identity(&*parent.lease()?)
                    .map_err(xlog_err)?
        {
            return Err(invalid(
                "evaluation cancellation changed its original invocation or parent",
            ));
        }
        Py::new(
            py,
            PySemanticCancelledModelEvaluation {
                inner: original.clone(),
            },
        )
    }
}

#[pymethods]
impl PySemanticTransitionController {
    /// Issue a genuine read-only owner on the acquired parent. Subsequent
    /// final observations pass the source's cohort, even across full restore.
    #[pyo3(signature = (task_use, parent, *, model_work_capacity=None, cohort=None))]
    fn prepare_model_evaluation(
        &self,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
        parent: Py<PySemanticPublishedParent>,
        model_work_capacity: Option<&Bound<'_, PyAny>>,
        cohort: Option<Py<PySemanticEvaluationCohort>>,
    ) -> PyResult<Py<PySemanticModelEvaluation>> {
        let capacity = model_work_capacity
            .map(|capacity| {
                usize::try_from(ColdValue::read(capacity, &mut 128, 0)?.unsigned()?).map_err(|_| {
                    invalid("evaluation model work capacity exceeds native address space")
                })
            })
            .transpose()?;
        let acquired = parent.borrow(py);
        let session = self.session.borrow(py);
        session.require_creator()?;
        let issued = task_use.borrow(py);
        self.require_read_issued(&issued)?;
        let continuing = session
            .pending_evaluation_preparation
            .lock()
            .map_err(|_| invalid("evaluation preparation owner is poisoned"))?
            .is_some();
        let public_instruction = super::evaluation_instruction::current(&session)?;
        if let Some(instruction) = &public_instruction {
            instruction.require_preparation(
                py,
                self,
                &issued,
                &acquired,
                cohort.as_ref(),
                continuing,
            )?;
        }
        if !continuing {
            let public_use = issued.state()?.require_public_use();
            if let Err(public_error) = public_use {
                #[cfg(not(feature = "semantic-policy"))]
                return Err(public_error);
                #[cfg(feature = "semantic-policy")]
                {
                    let pending = session
                        .learning_transition
                        .lock()
                        .map_err(|_| invalid("learning-phase retention mutex is poisoned"))?
                        .as_ref()
                        .map(|pending| pending.clone_ref(py));
                    let pending = pending.ok_or(public_error)?;
                    pending
                        .borrow(py)
                        .require_phase_evaluation(py, self, &issued, &acquired)?;
                }
            } else if session.learning_preparing.load(Ordering::Acquire) {
                #[cfg(feature = "semantic-policy")]
                {
                    let pending = private_execution_owner(py, &session)?.ok_or_else(|| {
                        invalid("private evaluation lost its original phase owner")
                    })?;
                    pending
                        .borrow(py)
                        .require_phase_evaluation(py, self, &issued, &acquired)?;
                }
                #[cfg(not(feature = "semantic-policy"))]
                return Err(invalid("private evaluation requires semantic-policy"));
            }
        }
        if acquired.session.as_ptr() != self.session.as_ptr() {
            return Err(invalid("evaluation parent belongs to another Session"));
        }
        acquired.require_task(py, &issued)?;
        let mut owner = session.owner()?;
        let lease = acquired.lease()?;
        let mut pending = session
            .pending_evaluation_preparation
            .lock()
            .map_err(|_| invalid("evaluation preparation owner is poisoned"))?;
        let continuing = pending.is_some();
        if let Some(original) = pending.as_ref() {
            let same_cohort = match (&original.requested_cohort, &cohort) {
                (None, None) => true,
                (Some(original), Some(requested)) => original.as_ptr() == requested.as_ptr(),
                _ => false,
            };
            if original.parent.as_ptr() != parent.as_ptr()
                || original.task_use.as_ptr() != task_use.as_ptr()
                || original.capacity != capacity
                || !same_cohort
                || issued.state()?.content_handoff_binding()? != original.binding
            {
                return Err(invalid("evaluation preparation continuation changed its original parent, task authority, cohort or capacity"));
            }
            if owner.task_evaluation_epoch() != issued.task_epoch
                || owner
                    .task_evaluation_identity()
                    .is_none_or(|identity| identity.as_bytes() != &issued.task_identity)
            {
                return Err(invalid(
                    "evaluation preparation changed its original native task binding",
                ));
            }
            if owner
                .pending_model_evaluation_preparation_identity(&lease)
                .map_err(xlog_err)?
                .is_none()
            {
                return Err(invalid(
                    "evaluation preparation lost its original native owner",
                ));
            }
        }
        if session.importing.load(Ordering::Acquire) || session.retiring.load(Ordering::Acquire) {
            return Err(invalid(
                "read-only evaluation cannot overlap import, recording or retirement",
            ));
        }
        if !continuing {
            let binding = acquired.content_binding_with_owner(py, &owner)?;
            if session.recording.swap(true, Ordering::AcqRel) && public_instruction.is_none() {
                return Err(invalid(
                    "read-only evaluation cannot overlap an original recording",
                ));
            }
            *pending = Some(PendingEvaluationPreparation {
                parent: parent.clone_ref(py),
                task_use: task_use.clone_ref(py),
                requested_cohort: cohort.as_ref().map(|value| value.clone_ref(py)),
                capacity,
                binding,
                inner: None,
                cohort: None,
                result: None,
                phase_entered: false,
                operation_bound: false,
            });
        }
        let original = pending.as_mut().expect("retained preparation inputs");
        if original.inner.is_none() {
            let requested_cohort = original
                .requested_cohort
                .as_ref()
                .map(|value| Arc::clone(&value.borrow(py).inner));
            let prepared = if let Some(capacity) = capacity {
                owner.prepare_model_evaluation(&lease, requested_cohort, capacity)
            } else {
                owner.prepare_model_evaluation_deferred(&lease, requested_cohort)
            };
            match prepared {
                Ok(inner) => original.inner = Some(inner),
                Err(error) => {
                    let error = xlog_err(error);
                    if owner
                        .pending_model_evaluation_preparation_identity(&lease)
                        .is_ok_and(|value| value.is_some())
                    {
                        let pending = SemanticModelEvaluationPending::new_err("original evaluation preparation awaits completion; call prepare_model_evaluation again with the same parent, task, cohort and capacity");
                        pending.set_cause(py, Some(error));
                        return Err(pending);
                    }
                    pending.take();
                    session
                        .recording
                        .store(public_instruction.is_some(), Ordering::Release);
                    return Err(error);
                }
            }
        }
        if original.cohort.is_none() {
            original.cohort = Some(match &original.requested_cohort {
                Some(cohort) => cohort.clone_ref(py),
                None => Py::new(
                    py,
                    PySemanticEvaluationCohort {
                        inner: original
                            .inner
                            .as_ref()
                            .expect("retained native preparation")
                            .cohort(),
                    },
                )?,
            });
        }
        if original.result.is_none() {
            original.result = Some(Py::new(
                py,
                PySemanticModelEvaluation {
                    parent: parent.clone_ref(py),
                    inner: original
                        .inner
                        .as_ref()
                        .expect("retained native preparation")
                        .clone(),
                    cohort: original
                        .cohort
                        .as_ref()
                        .expect("retained cohort")
                        .clone_ref(py),
                    binding: original.binding.clone(),
                    closed: AtomicBool::new(false),
                    output: Mutex::new(None),
                    completed: Mutex::new(None),
                    cancelled: Mutex::new(None),
                    cancel_entered: AtomicBool::new(false),
                    capture_owners: Mutex::new(None),
                },
            )?);
        }
        if !original.phase_entered {
            let mut state = issued.state()?;
            state.phase = TaskUsePhase::Evaluating(Box::new(state.phase.clone()));
            original.phase_entered = true;
        }
        // The phase binder acquires the same Session owner; release this guard
        // before it and retain the issued object across a binder failure.
        drop(owner);
        if !original.operation_bound {
            if let Some(instruction) = &public_instruction {
                let result = original.result.as_ref().expect("retained evaluation");
                instruction.bind_evaluation(py, result, &result.borrow(py).inner)?;
            } else if let Some(phase) = private_execution_owner(py, &session)? {
                let result = original.result.as_ref().expect("retained evaluation");
                phase
                    .borrow(py)
                    .bind_phase_evaluation(py, result, &result.borrow(py).inner)?;
            }
            original.operation_bound = true;
        }
        let result = original
            .result
            .as_ref()
            .expect("retained evaluation")
            .clone_ref(py);
        // The returned owner now carries the original parent and native
        // invocation. Do not keep a Session -> parent -> Session cycle after
        // the complete handoff; the recording flag rejects a second prepare.
        pending.take();
        Ok(result)
    }
}
