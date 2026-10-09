//! Projection of the native model-work registrar during the original cold
//! callback. It grants none of the prepared/evaluation numerical authorities.

use super::*;
use xlog_cuda::{SemanticColdModelWork, SemanticColdModelWorkRegion};

fn cold_region_ends(value: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if !value.is_exact_instance_of::<PyTuple>() {
        return Err(invalid("cold region ends require an exact immutable tuple"));
    }
    value
        .cast::<PyTuple>()?
        .iter()
        .map(|end| {
            usize::try_from(ColdValue::read(&end, &mut 128, 0)?.unsigned()?)
                .map_err(|_| invalid("cold region end exceeds the native event extent"))
        })
        .collect()
}

#[pyclass(name = "SemanticColdModelWork", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticColdModelWork {
    pub(super) parent: Py<PySemanticPublishedParent>,
    // The work may precede child allocation. Its original allocation/report
    // reader stays held while the actual child owns the model callback.
    pub(super) reader: Py<PySemanticPublishedParent>,
    pub(super) inner: SemanticColdModelWork,
    pub(super) region: Option<SemanticColdModelWorkRegion>,
    pub(super) active: AtomicBool,
}

impl PySemanticColdModelWork {
    pub(super) fn check(&self, py: Python<'_>) -> PyResult<()> {
        let parent = self.parent.borrow(py);
        let source = parent.session.borrow(py);
        source.require_creator()?;
        if !self.active.load(Ordering::Acquire) {
            return Err(invalid(
                "cold model work is accessible only inside its original admitted callback",
            ));
        }
        #[cfg(feature = "semantic-policy")]
        if require_admitted_cold_callback(py, &source, self)? {
            return Ok(());
        }
        #[cfg(feature = "semantic-policy")]
        if prepared_cold_callback_step(py, &source, self)?.is_some() {
            return Ok(());
        }
        #[cfg(feature = "semantic-policy")]
        if admitted_retirement_cold_callback(py, &source, self)?.is_some() {
            return Ok(());
        }
        if !source.learning_preparing.load(Ordering::Acquire) {
            return Err(invalid(
                "cold model work is accessible only inside its original admitted callback",
            ));
        }
        if matches!(&parent.task_use.borrow(py).state()?.phase,
            TaskUsePhase::CheckpointReading { original, .. } if matches!(original.as_ref(), TaskUsePhase::ArenaPreparing(_)))
        {
            return Ok(());
        }
        #[cfg(feature = "semantic-policy")]
        {
            let pending = source
                .learning_transition
                .lock()
                .map_err(|_| invalid("private cold callback lost its original phase mutex"))?
                .as_ref()
                .map(|pending| pending.clone_ref(py))
                .ok_or_else(|| {
                    invalid("private cold callback lost its original native phase owner")
                })?;
            if self.region.is_some() {
                if pending
                    .borrow(py)
                    .require_terminal_refusal_cold_callback(py, self)?
                {
                    return Ok(());
                }
                return pending
                    .borrow(py)
                    .require_evaluation_cold_callback(py, self);
            }
            return pending
                .borrow(py)
                .require_private_numeric_cold_callback(py, self);
        }
        #[cfg(not(feature = "semantic-policy"))]
        Err(invalid(
            "private numerical cold work requires semantic-policy",
        ))
    }
}

#[pyclass(
    name = "SemanticColdModelWorkOperation",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticColdModelWorkOperation {
    work: Py<PySemanticColdModelWork>,
    inner: xlog_cuda::SemanticColdModelWorkOperation,
}

#[pymethods]
impl PySemanticColdModelWorkOperation {
    /// The original pre-initialized device slot, not actual expenditure.
    #[getter]
    fn slot(&self, py: Python<'_>) -> PyResult<usize> {
        let work = self.work.borrow(py);
        work.check(py)?;
        let reader = work.reader.borrow(py);
        let session = reader.session.borrow(py);
        let result = session
            .owner()?
            .cold_model_work_operation_slot(&self.inner)
            .map_err(xlog_err);
        result
    }

    fn enter(&self, py: Python<'_>) -> PyResult<usize> {
        let work = self.work.borrow(py);
        work.check(py)?;
        let reader = work.reader.borrow(py);
        let session = reader.session.borrow(py);
        let result = session
            .owner()?
            .enter_cold_model_work_operation(&self.inner)
            .map_err(xlog_err);
        result
    }

    fn complete(&self, py: Python<'_>) -> PyResult<usize> {
        let work = self.work.borrow(py);
        work.check(py)?;
        let reader = work.reader.borrow(py);
        let session = reader.session.borrow(py);
        let result = session
            .owner()?
            .complete_cold_model_work_operation(&self.inner)
            .map_err(xlog_err);
        result
    }

    fn unknown(&self, py: Python<'_>) -> PyResult<()> {
        let work = self.work.borrow(py);
        work.check(py)?;
        let reader = work.reader.borrow(py);
        let session = reader.session.borrow(py);
        let result = session
            .owner()?
            .retain_unknown_cold_model_work_operation(&self.inner)
            .map_err(xlog_err);
        result
    }

    fn cancel_before_entry(&self, py: Python<'_>) -> PyResult<()> {
        let work = self.work.borrow(py);
        work.check(py)?;
        let reader = work.reader.borrow(py);
        let session = reader.session.borrow(py);
        let result = session
            .owner()?
            .cancel_unentered_cold_model_work_operation(&self.inner)
            .map_err(xlog_err);
        result
    }
}

#[pymethods]
impl PySemanticColdModelWork {
    #[pyo3(signature = (kind, dimensions, *, device_produced = false))]
    fn prepare_operation(
        slf: Py<Self>,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        dimensions: &Bound<'_, PyAny>,
        device_produced: bool,
    ) -> PyResult<Py<PySemanticColdModelWorkOperation>> {
        let work = slf.borrow(py);
        work.check(py)?;
        if !dimensions.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "operation geometry requires its exact immutable tuple",
            ));
        }
        let (kind, values, rank) = parse_model_work(kind, dimensions)?;
        let reader = work.reader.borrow(py);
        let session = reader.session.borrow(py);
        let inner = session
            .owner()?
            .prepare_cold_model_work_operation(
                &work.inner,
                work.region.as_ref(),
                kind,
                &values[..rank],
                device_produced,
            )
            .map_err(xlog_err)?;
        Py::new(
            py,
            PySemanticColdModelWorkOperation {
                work: slf.clone_ref(py),
                inner,
            },
        )
    }

    fn prepare_operations(
        slf: Py<Self>,
        py: Python<'_>,
        operations: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let work = slf.borrow(py);
        work.check(py)?;
        if !operations.is_exact_instance_of::<PyTuple>() {
            return Err(invalid("model operations require an exact immutable tuple"));
        }
        let mut parsed = Vec::new();
        for operation in operations.cast::<PyTuple>()?.iter() {
            if !operation.is_exact_instance_of::<PyTuple>() {
                return Err(invalid(
                    "model operation requires its exact geometry and production triple",
                ));
            }
            let operation = operation.cast::<PyTuple>()?;
            if operation.len() != 3
                || !operation.get_item(1)?.is_exact_instance_of::<PyTuple>()
                || !operation.get_item(2)?.is_exact_instance_of::<PyBool>()
            {
                return Err(invalid(
                    "model operation requires kind, immutable dimensions and exact bool",
                ));
            }
            let (kind, values, rank) =
                parse_model_work(&operation.get_item(0)?, &operation.get_item(1)?)?;
            parsed.push((
                kind,
                values[..rank].to_vec(),
                operation.get_item(2)?.extract::<bool>()?,
            ));
        }
        let reader = work.reader.borrow(py);
        let session = reader.session.borrow(py);
        let handles = session
            .owner()?
            .prepare_cold_model_work_operations(&work.inner, work.region.as_ref(), &parsed)
            .map_err(xlog_err)?;
        let handles = handles
            .into_iter()
            .map(|inner| {
                Py::new(
                    py,
                    PySemanticColdModelWorkOperation {
                        work: slf.clone_ref(py),
                        inner,
                    },
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, handles)?.unbind())
    }

    /// Original native plan state, not a Python callback invocation count.
    #[getter]
    fn plan_admitted(&self, py: Python<'_>) -> PyResult<bool> {
        self.check(py)?;
        self.reader
            .borrow(py)
            .session
            .borrow(py)
            .owner()?
            .cold_model_work_plan_is_admitted(&self.inner)
            .map_err(xlog_err)
    }

    /// Fixed native report geometry, not a callback invocation counter.
    #[getter]
    fn region_count(&self, py: Python<'_>) -> PyResult<usize> {
        self.check(py)?;
        self.reader
            .borrow(py)
            .session
            .borrow(py)
            .owner()?
            .cold_model_work_region_count(&self.inner)
            .map_err(xlog_err)
    }

    /// Seventeen original source-port layouts; no device export or selection.
    fn evaluation_source_geometry(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<(Vec<i64>, Vec<i64>, (u8, u8))>> {
        self.check(py)?;
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        let result = session
            .owner()?
            .evaluation_source_geometry(&*parent.lease()?)
            .map_err(xlog_err);
        result
    }

    /// Retain the complete physical operation recipe before original recording.
    /// Native returns (work bound, event count, call upper), not a Python estimate.
    #[pyo3(
        signature = (operations, *, region_ends = Vec::new(), evaluation_content = None),
        text_signature = "($self, operations, *, region_ends=(), evaluation_content=None)"
    )]
    fn admit_plan(
        &self,
        py: Python<'_>,
        operations: &Bound<'_, PyAny>,
        #[pyo3(from_py_with = cold_region_ends)] region_ends: Vec<usize>,
        evaluation_content: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(u64, u64, u64)> {
        self.check(py)?;
        // Invoke original tensor producers before taking either native Session
        // mutex. The actual evaluation owner guards every producer callback.
        let (phase, content) = if let Some(value) = evaluation_content {
            let parent = self.parent.borrow(py);
            let session = parent.session.borrow(py);
            let phase = private_execution_owner(py, &session)?.ok_or_else(|| {
                invalid("evaluation content requires its actual original phase owner")
            })?;
            let evaluation = phase
                .borrow(py)
                .evaluation_cold_content_owner(py, &self.inner)?;
            let content = evaluation.borrow(py).prepare_cold_content(py, value)?;
            (Some(phase), Some(content))
        } else {
            (None, None)
        };
        let reader = self.reader.borrow(py);
        let session = reader.session.borrow(py);
        let mut owner = session.owner()?;
        let result = (|| {
            if !operations.is_exact_instance_of::<PyTuple>() {
                return Err(invalid(
                    "cold operation plan requires an exact immutable tuple",
                ));
            }
            let operations = operations.cast::<PyTuple>()?;
            let mut recipe = Vec::new();
            recipe
                .try_reserve_exact(operations.len())
                .map_err(|_| invalid("cold operation plan reservation failed"))?;
            for operation in operations.iter() {
                if !operation.is_exact_instance_of::<PyTuple>()
                    || operation.cast::<PyTuple>()?.len() != 2
                {
                    return Err(invalid(
                        "cold operation plan requires exact kind and geometry pairs",
                    ));
                }
                let operation = operation.cast::<PyTuple>()?;
                let kind = operation.get_item(0)?;
                let dimensions = operation.get_item(1)?;
                if !dimensions.is_exact_instance_of::<PyTuple>() {
                    return Err(invalid(
                        "cold operation geometry requires an exact immutable tuple",
                    ));
                }
                let (kind, values, rank) = parse_model_work(&kind, &dimensions)?;
                recipe.push((kind, values[..rank].to_vec()));
            }
            let quantities = if let Some(region) = &self.region {
                owner.admit_cold_model_work_region_plan(region, &recipe, &region_ends, content)
            } else {
                owner.admit_cold_model_work_plan(&self.inner, &recipe, &region_ends, content)
            }
            .map_err(xlog_err)?;
            Ok((quantities[0], quantities[1], quantities[2]))
        })();
        if result.is_err() {
            owner.fail_cold_model_work(&self.inner);
        }
        drop(owner);
        let quantities = result?;
        if let Some(phase) = phase {
            if let Err(error) = phase
                .borrow(py)
                .admit_evaluation_cold_content(py, &self.inner)
            {
                session.owner()?.fail_cold_model_work(&self.inner);
                return Err(error);
            }
        }
        Ok(quantities)
    }

    #[getter]
    fn consumer_stream(&self, py: Python<'_>) -> PyResult<u64> {
        self.check(py)?;
        self.reader
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
        let parent = self.reader.borrow(py);
        let source = parent.session.borrow(py);
        #[cfg(feature = "semantic-policy")]
        let prepared = prepared_cold_callback_step(py, &source, self)?;
        #[cfg(feature = "semantic-policy")]
        let retirement = admitted_retirement_cold_callback(py, &source, self)?;
        let mut owner = source.owner()?;
        #[cfg(feature = "semantic-policy")]
        let tensor = if let Some(instruction) = retirement {
            owner.admitted_retirement_cold_model_work_buffer(
                &instruction.inner,
                &self.inner,
                stream,
            )
        } else if let Some(step) = prepared {
            owner.prepared_cold_model_work_buffer(
                &step.borrow(py).inner,
                &*parent.lease()?,
                &self.inner,
                stream,
                self.region.as_ref(),
            )
        } else if let Some(region) = &self.region {
            owner.cold_model_work_region_buffer(&*parent.lease()?, region, stream)
        } else {
            owner.cold_model_work_buffer(&*parent.lease()?, &self.inner, stream)
        }
        .map_err(xlog_err)?;
        #[cfg(not(feature = "semantic-policy"))]
        let tensor = if let Some(region) = &self.region {
            owner.cold_model_work_region_buffer(&*parent.lease()?, region, stream)
        } else {
            owner.cold_model_work_buffer(&*parent.lease()?, &self.inner, stream)
        }
        .map_err(xlog_err)?;
        let tensor = retain_export_owner(tensor, self.parent.clone_ref(py), source.owner_thread)?;
        crate::dlpack_capsule_from_tensor(py, tensor)
    }

    /// Attach the existing recorder once, even for a CPU/meta-only callback.
    fn begin(&self, py: Python<'_>) -> PyResult<()> {
        self.check(py)?;
        let reader = self.reader.borrow(py);
        let session = reader.session.borrow(py);
        let mut owner = session.owner()?;
        if let Some(region) = &self.region {
            owner.begin_cold_model_work_region(region)
        } else {
            owner.begin_cold_model_work(&self.inner)
        }
        .map_err(xlog_err)
    }

    /// Close registration only. Callback return does not certify CUDA or
    /// allocator completion; the enclosing native operation joins those later.
    fn end(&self, py: Python<'_>) -> PyResult<()> {
        self.check(py)?;
        let reader = self.reader.borrow(py);
        let session = reader.session.borrow(py);
        let mut owner = session.owner()?;
        if let Some(region) = &self.region {
            owner.close_cold_model_work_region(region)
        } else {
            owner.close_cold_model_work(&self.inner)
        }
        .map_err(xlog_err)
    }

    fn fail(&self, py: Python<'_>) -> PyResult<()> {
        self.check(py)?;
        self.reader
            .borrow(py)
            .session
            .borrow(py)
            .owner()?
            .fail_cold_model_work(&self.inner);
        Ok(())
    }

    fn record_model_invocation(&self, py: Python<'_>) -> PyResult<()> {
        self.check(py)?;
        let reader = self.reader.borrow(py);
        let session = reader.session.borrow(py);
        let mut owner = session.owner()?;
        if let Some(region) = &self.region {
            owner.record_cold_model_region_invocation(region)
        } else {
            owner.record_cold_model_invocation(&self.inner)
        }
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
        let parent = self.reader.borrow(py);
        let source = parent.session.borrow(py);
        let mut owner = source.owner()?;
        let result = (|| {
            let (kind, values, rank) = parse_model_work(kind, dimensions)?;
            if let Some(region) = &self.region {
                owner.record_cold_model_work_region(region, kind, &values[..rank], device_produced)
            } else {
                owner.record_cold_model_work(&self.inner, kind, &values[..rank], device_produced)
            }
            .map_err(xlog_err)
        })();
        if result.is_err() {
            owner.fail_cold_model_work(&self.inner);
        }
        result
    }
}

/// Visibility belongs to the actual callback's Session, not the source phase's
/// first registrar. Private restored Sessions use the same callback boundary.
pub(in crate::semantic_transition) struct ColdCallbackScope<'a> {
    active: &'a AtomicBool,
    session: &'a PySemanticTransitionSession,
}

impl<'a> ColdCallbackScope<'a> {
    pub(in crate::semantic_transition) fn enter(
        py: Python<'_>,
        session: &'a PySemanticTransitionSession,
        work: &'a PySemanticColdModelWork,
        retained: Py<PySemanticColdModelWork>,
    ) -> PyResult<Self> {
        session.require_creator()?;
        if !std::ptr::eq(&*retained.borrow(py), work)
            || !std::ptr::eq(&*work.parent.borrow(py).session.borrow(py), session)
        {
            return Err(invalid(
                "cold callback requires its actual original Session and registrar",
            ));
        }
        let mut current = session
            .active_cold_model_work
            .lock()
            .map_err(|_| invalid("active cold callback custody mutex is poisoned"))?;
        if current.is_some() {
            return Err(invalid("another cold callback is already active"));
        }
        work.active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid("the original cold registrar is already active"))?;
        *current = Some(retained);
        drop(current);
        let scope = Self {
            active: &work.active,
            session,
        };
        work.check(py)?;
        Ok(scope)
    }
}

impl Drop for ColdCallbackScope<'_> {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
        let retained = self
            .session
            .active_cold_model_work
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        // Dropping visibility cannot release or complete the native work. Its
        // original operation retains the registrar and asynchronous storage.
        drop(retained);
    }
}

fn retained_work(
    py: Python<'_>,
    session: &PySemanticTransitionSession,
) -> PyResult<Py<PySemanticColdModelWork>> {
    session.require_creator()?;
    let work = session
        .active_cold_model_work
        .lock()
        .map_err(|_| invalid("active cold callback custody mutex is poisoned"))?
        .as_ref()
        .map(|work| work.clone_ref(py))
        .ok_or_else(|| invalid("there is no original admitted cold callback in this Session"))?;
    work.borrow(py).check(py)?;
    Ok(work)
}

#[pymethods]
impl PySemanticTransitionTaskUse {
    /// The original owner is available inside the callback, before prepare
    /// returns. Another task or a stale retained handle cannot borrow its rights.
    fn active_cold_model_work(&self, py: Python<'_>) -> PyResult<Py<PySemanticColdModelWork>> {
        #[cfg(feature = "semantic-policy")]
        {
            let source = self.session.borrow(py);
            let pending = source
                .learning_transition
                .lock()
                .map_err(|_| invalid("private cold callback lost its original phase mutex"))?
                .as_ref()
                .map(|pending| pending.clone_ref(py));
            if let Some(pending) = pending {
                if let Some(work) = pending.borrow(py).evaluation_cold_work(py, self)? {
                    return Ok(work);
                }
                if pending.borrow(py).private_adoption_due()? {
                    return pending.borrow(py).private_adoption_work(py, &source, self);
                }
            }
        }
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
        let task = work.borrow(py).parent.borrow(py).task_use.clone_ref(py);
        self.require_issued(&task.borrow(py))?;
        let original = task.borrow(py).active_cold_model_work(py);
        original
    }
}
