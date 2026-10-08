//! Native read-only ownership at the existing model content/work boundary.

use super::*;
use xlog_cuda::{
    SemanticCancelledModelEvaluation, SemanticCompletedModelEvaluation, SemanticEvaluationCohort,
    SemanticModelEvaluation, SemanticModelEvaluationResult,
};

pyo3::create_exception!(
    pyxlog._native,
    SemanticModelEvaluationPending,
    PyRuntimeError,
    "An original submitted evaluation awaits cold completion; retain its owners and never resubmit."
);

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
    pub(super) fn require_phase_observation(
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
            return Err(invalid("phase evaluation changed its original completed native observation, parent or cohort"));
        }
        Ok(self.inner.result())
    }
}

/// Retains only the original selected device roster and its native seals.
/// It does not retain a source Runtime, grant or mutable model generation.
#[pyclass(name = "SemanticEvaluationCohort", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticEvaluationCohort {
    inner: Arc<SemanticEvaluationCohort>,
}

/// Single original forward/objective recording; no backward or publication.
#[pyclass(name = "SemanticModelEvaluation", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticModelEvaluation {
    parent: Py<PySemanticPublishedParent>,
    inner: SemanticModelEvaluation,
    cohort: Py<PySemanticEvaluationCohort>,
    binding: (Option<String>, Vec<u8>),
    closed: AtomicBool,
    output: Mutex<Option<Py<PySemanticTensorContentWitness>>>,
    completed: Mutex<Option<Py<PySemanticCompletedModelEvaluation>>>,
    cancelled: Mutex<Option<SemanticCancelledModelEvaluation>>,
    cancel_entered: AtomicBool,
}

impl PySemanticModelEvaluation {
    fn phase_cold_boundary(
        &self,
        py: Python<'_>,
        from: learning_phase::phase_evaluation::EvaluationColdStage,
        to: learning_phase::phase_evaluation::EvaluationColdStage,
    ) -> PyResult<()> {
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        if let Some(original) = private_execution_owner(py, &session)? {
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
        self.restore_phase(py)?;
        Ok(observation)
    }

    fn check(&self, py: Python<'_>, owner: &SemanticTransitionSession) -> PyResult<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(invalid("read-only evaluation is already closed"));
        }
        let parent = self.parent.borrow(py);
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
        parent
            .session
            .borrow(py)
            .recording
            .store(false, Ordering::Release);
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
        let tensor = f(&mut owner, &*parent.lease()?, &self.inner, stream).map_err(xlog_err)?;
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

    fn begin(&self, py: Python<'_>) -> PyResult<()> {
        use learning_phase::phase_evaluation::EvaluationColdStage;
        self.phase_cold_boundary(
            py,
            EvaluationColdStage::Preparation,
            EvaluationColdStage::OutputProjection,
        )?;
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        let mut owner = session.owner()?;
        self.check(py, &owner)?;
        let lease = parent.lease()?;
        owner
            .begin_model_evaluation(&lease, &self.inner)
            .map_err(xlog_err)
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

    /// Output witness belongs to the original PublishedParent: three tensor
    /// rows in order, FP32[6], Bool[1], UInt8[slab_bytes]. No Python loss, work
    /// or memory total is accepted. This is cold completion, outside residency.
    #[pyo3(signature = (output_witness, *, consumer_streams))]
    fn finish(
        &self,
        py: Python<'_>,
        output_witness: Py<PySemanticTensorContentWitness>,
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
        let witness = output_witness.borrow(py);
        if witness.session.as_ptr() != parent.session.as_ptr()
            || !matches!(&witness.parent, ContentStepOwner::Published(original) if original.as_ptr() == self.parent.as_ptr())
        {
            return Err(invalid(
                "evaluation outputs require the exact original published-parent content owner",
            ));
        }
        let mut retained = self
            .output
            .lock()
            .map_err(|_| invalid("evaluation output owner is poisoned"))?;
        if retained.is_some() {
            return Err(invalid("evaluation output was already submitted"));
        }
        *retained = Some(output_witness.clone_ref(py));
        drop(retained);
        use learning_phase::phase_evaluation::EvaluationColdStage;
        self.phase_cold_boundary(
            py,
            EvaluationColdStage::OutputProjection,
            EvaluationColdStage::AwaitingCompletion,
        )?;
        let mut owner = session.owner()?;
        self.check(py, &owner)?;
        let result = owner
            .finish_model_evaluation(&*parent.lease()?, &self.inner, &witness.inner, &streams)
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
        let cancelled = owner
            .cancel_model_evaluation(&*parent.lease()?, &self.inner, &streams)
            .map_err(xlog_err)?;
        *self
            .cancelled
            .lock()
            .map_err(|_| invalid("evaluation cancellation custody is poisoned"))? =
            Some(cancelled.clone());
        drop(owner);
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
}

#[pymethods]
impl PySemanticTransitionController {
    /// Issue a genuine read-only owner on the acquired parent. Subsequent
    /// final observations pass the source's cohort, even across full restore.
    #[pyo3(signature = (task_use, parent, *, model_work_capacity, cohort=None))]
    fn prepare_model_evaluation(
        &self,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
        parent: Py<PySemanticPublishedParent>,
        model_work_capacity: &Bound<'_, PyAny>,
        cohort: Option<Py<PySemanticEvaluationCohort>>,
    ) -> PyResult<Py<PySemanticModelEvaluation>> {
        let capacity =
            usize::try_from(ColdValue::read(model_work_capacity, &mut 128, 0)?.unsigned()?)
                .map_err(|_| {
                    invalid("evaluation model work capacity exceeds native address space")
                })?;
        let acquired = parent.borrow(py);
        let session = self.session.borrow(py);
        session.require_creator()?;
        let issued = task_use.borrow(py);
        self.require_read_issued(&issued)?;
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
                let pending = private_execution_owner(py, &session)?
                    .ok_or_else(|| invalid("private evaluation lost its original phase owner"))?;
                pending
                    .borrow(py)
                    .require_phase_evaluation(py, self, &issued, &acquired)?;
            }
            #[cfg(not(feature = "semantic-policy"))]
            return Err(invalid("private evaluation requires semantic-policy"));
        }
        if acquired.session.as_ptr() != self.session.as_ptr() {
            return Err(invalid("evaluation parent belongs to another Session"));
        }
        acquired.require_task(py, &issued)?;
        let mut owner = session.owner()?;
        let binding = acquired.content_binding_with_owner(py, &owner)?;
        let lease = acquired.lease()?;
        if session.importing.load(Ordering::Acquire)
            || session.retiring.load(Ordering::Acquire)
            || session.recording.swap(true, Ordering::AcqRel)
        {
            return Err(invalid(
                "read-only evaluation cannot overlap import, recording or retirement",
            ));
        }
        let inner = owner.prepare_model_evaluation(
            &lease,
            cohort
                .as_ref()
                .map(|value| Arc::clone(&value.borrow(py).inner)),
            capacity,
        );
        let inner = match inner {
            Ok(inner) => inner,
            Err(error) => {
                session.recording.store(false, Ordering::Release);
                return Err(xlog_err(error));
            }
        };
        let issued_cohort = (|| {
            let cohort = match cohort {
                Some(original) => original,
                None => Py::new(
                    py,
                    PySemanticEvaluationCohort {
                        inner: inner.cohort(),
                    },
                )?,
            };
            let mut state = issued.state()?;
            state.phase = TaskUsePhase::Evaluating(Box::new(state.phase.clone()));
            Ok::<_, PyErr>(cohort)
        })();
        let cohort = match issued_cohort {
            Ok(cohort) => cohort,
            Err(error) => {
                // Native preparation may already have queued content guards.
                // A Python allocation/state failure is not their completion.
                owner.abort();
                if let Ok(mut state) = issued.state() {
                    state.phase = TaskUsePhase::Refused;
                }
                session.recording.store(false, Ordering::Release);
                return Err(error);
            }
        };
        drop(owner);
        let original = Py::new(
            py,
            PySemanticModelEvaluation {
                parent: parent.clone_ref(py),
                inner,
                cohort,
                binding,
                closed: AtomicBool::new(false),
                output: Mutex::new(None),
                completed: Mutex::new(None),
                cancelled: Mutex::new(None),
                cancel_entered: AtomicBool::new(false),
            },
        )?;
        if let Some(pending) = private_execution_owner(py, &session)? {
            pending.borrow(py).bind_phase_evaluation(py, &original)?;
        }
        Ok(original)
    }
}
