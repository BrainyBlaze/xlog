//! Native read-only ownership at the existing model content/work boundary.

use super::*;
use xlog_cuda::{SemanticEvaluationCohort, SemanticModelEvaluation};

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
}

impl PySemanticModelEvaluation {
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
    ) -> PyResult<Py<PyDict>> {
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
        let mut owner = session.owner()?;
        self.check(py, &owner)?;
        let result = owner
            .finish_model_evaluation(&*parent.lease()?, &self.inner, &witness.inner, &streams)
            .map_err(xlog_err)?;
        drop(owner);
        // Native finish returns only a complete expenditure certificate, including
        // a known numerical refusal. An incomplete invocation errors above and
        // retains this original phase/output/owner without reopening admission.
        if matches!(result.status, 0 | 1) {
            self.restore_phase(py)?;
        }
        let value = PyDict::new(py);
        value.set_item("status", result.status)?;
        value.set_item("loss_bits", PyTuple::new(py, result.loss_bits)?)?;
        value.set_item("model_work", result.model_work)?;
        value.set_item("model_calls", result.model_calls)?;
        value.set_item("operation_count", result.operation_count)?;
        value.set_item("work_bound", result.work_bound)?;
        value.set_item(
            "retained_allocation_bytes",
            result.retained_allocation_bytes,
        )?;
        Ok(value.unbind())
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
        owner
            .cancel_model_evaluation(&*parent.lease()?, &self.inner, &streams)
            .map_err(xlog_err)?;
        drop(owner);
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
        self.require_issued(&issued)?;
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
        Py::new(
            py,
            PySemanticModelEvaluation {
                parent: parent.clone_ref(py),
                inner,
                cohort,
                binding,
                closed: AtomicBool::new(false),
                output: Mutex::new(None),
            },
        )
    }
}
