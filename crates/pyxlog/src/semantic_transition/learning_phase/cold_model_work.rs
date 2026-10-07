//! Projection of the native model-work registrar during the original cold
//! callback. It grants none of the prepared/evaluation numerical authorities.

use super::*;
use xlog_cuda::SemanticColdModelWork;

#[pyclass(name = "SemanticColdModelWork", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticColdModelWork {
    pub(super) parent: Py<PySemanticPublishedParent>,
    pub(super) inner: SemanticColdModelWork,
    pub(super) active: AtomicBool,
}

impl PySemanticColdModelWork {
    pub(super) fn check(&self, py: Python<'_>) -> PyResult<()> {
        let parent = self.parent.borrow(py);
        let source = parent.session.borrow(py);
        source.require_creator()?;
        if !self.active.load(Ordering::Acquire)
            || !source.learning_preparing.load(Ordering::Acquire)
            || !matches!(
                &parent.task_use.borrow(py).state()?.phase,
                TaskUsePhase::CheckpointReading { original, .. }
                    if matches!(original.as_ref(), TaskUsePhase::ArenaPreparing(_))
            )
        {
            return Err(invalid(
                "cold model work is accessible only inside its original admitted callback",
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl PySemanticColdModelWork {
    #[getter]
    fn consumer_stream(&self, py: Python<'_>) -> PyResult<u64> {
        self.check(py)?;
        self.parent
            .borrow(py)
            .session
            .borrow(py)
            .owner()?
            .cold_model_work_stream(&self.inner)
            .map_err(xlog_err)
    }

    /// Native U64[capacity,3] actual-work slots on the original CUDA stream.
    #[pyo3(signature = (*, consumer_stream))]
    fn model_work_buffer(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.check(py)?;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let parent = self.parent.borrow(py);
        let source = parent.session.borrow(py);
        let tensor = source
            .owner()?
            .cold_model_work_buffer(&*parent.lease()?, &self.inner, stream)
            .map_err(xlog_err)?;
        let tensor = retain_export_owner(tensor, self.parent.clone_ref(py), source.owner_thread)?;
        crate::dlpack_capsule_from_tensor(py, tensor)
    }

    /// Attach the existing recorder once, even for a CPU/meta-only callback.
    fn begin(&self, py: Python<'_>) -> PyResult<()> {
        self.check(py)?;
        self.parent
            .borrow(py)
            .session
            .borrow(py)
            .owner()?
            .begin_cold_model_work(&self.inner)
            .map_err(xlog_err)
    }

    /// Close registration only. Callback return does not certify CUDA or
    /// allocator completion; the enclosing native operation joins those later.
    fn end(&self, py: Python<'_>) -> PyResult<()> {
        self.check(py)?;
        self.parent
            .borrow(py)
            .session
            .borrow(py)
            .owner()?
            .close_cold_model_work(&self.inner)
            .map_err(xlog_err)
    }

    fn fail(&self, py: Python<'_>) -> PyResult<()> {
        self.check(py)?;
        self.parent
            .borrow(py)
            .session
            .borrow(py)
            .owner()?
            .fail_cold_model_work(&self.inner);
        Ok(())
    }

    fn record_model_invocation(&self, py: Python<'_>) -> PyResult<()> {
        self.check(py)?;
        self.parent
            .borrow(py)
            .session
            .borrow(py)
            .owner()?
            .record_cold_model_invocation(&self.inner)
            .map_err(xlog_err)
    }

    fn record_model_work(
        &self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        dimensions: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.record(py, kind, dimensions, false).map(|_| ())
    }

    fn record_model_device_work(
        &self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        upper_dimensions: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        self.record(py, kind, upper_dimensions, true)
    }
}

impl PySemanticColdModelWork {
    fn record(
        &self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        dimensions: &Bound<'_, PyAny>,
        device_produced: bool,
    ) -> PyResult<usize> {
        self.check(py)?;
        let parent = self.parent.borrow(py);
        let source = parent.session.borrow(py);
        let mut owner = source.owner()?;
        let result = (|| {
            let (kind, values, rank) = parse_model_work(kind, dimensions)?;
            owner
                .record_cold_model_work(&self.inner, kind, &values[..rank], device_produced)
                .map_err(xlog_err)
        })();
        if result.is_err() {
            owner.fail_cold_model_work(&self.inner);
        }
        result
    }
}

/// Visibility is bounded by the actual native callback, including exceptions.
pub(super) struct ColdCallbackScope<'a>(pub(super) &'a AtomicBool);

impl Drop for ColdCallbackScope<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn retained_work(
    py: Python<'_>,
    source: &PySemanticTransitionSession,
) -> PyResult<Py<PySemanticColdModelWork>> {
    source.require_creator()?;
    let pending = source
        .learning_transition
        .lock()
        .map_err(|_| invalid("learning-phase retention mutex is poisoned"))?
        .as_ref()
        .map(|pending| pending.clone_ref(py))
        .ok_or_else(|| invalid("there is no original admitted cold operation"))?;
    let pending = pending.borrow(py);
    let work = pending
        .source_preparation()?
        .as_ref()
        .and_then(|source| source.cold_model_work.as_ref())
        .map(|work| work.clone_ref(py))
        .ok_or_else(|| invalid("the original cold model registrar has not been issued"))?;
    work.borrow(py).check(py)?;
    Ok(work)
}

#[pymethods]
impl PySemanticTransitionTaskUse {
    /// The original owner is available inside the callback, before prepare
    /// returns. Another task or a stale retained handle cannot borrow its rights.
    fn active_cold_model_work(&self, py: Python<'_>) -> PyResult<Py<PySemanticColdModelWork>> {
        let work = retained_work(py, &self.session.borrow(py))?;
        if !std::ptr::eq(
            &*work.borrow(py).parent.borrow(py).task_use.borrow(py),
            self,
        ) {
            return Err(invalid(
                "cold model work belongs to another original task owner",
            ));
        }
        Ok(work)
    }
}

#[pymethods]
impl PySemanticTransitionController {
    fn active_cold_model_work(&self, py: Python<'_>) -> PyResult<Py<PySemanticColdModelWork>> {
        let work = retained_work(py, &self.session.borrow(py))?;
        self.require_issued(&work.borrow(py).parent.borrow(py).task_use.borrow(py))?;
        Ok(work)
    }
}
