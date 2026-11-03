//! Cold observation of the actual selected published task, after resident execution.

use super::*;

pyo3::create_exception!(
    pyxlog._native,
    SemanticSelectedTaskFactsPending,
    PyRuntimeError,
    "The original selected task observation retains unfinished reads; resolve the same owner."
);

struct SelectedTaskFactsState {
    native: xlog_cuda::SemanticSelectedTaskFactsObservation,
    completed: Option<Py<PySemanticCompletedTaskFacts>>,
}

#[pyclass(
    name = "SemanticSelectedTaskFactsObservation",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticSelectedTaskFactsObservation {
    parent: Py<PySemanticPublishedParent>,
    state: Mutex<SelectedTaskFactsState>,
    active: AtomicBool,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

#[pymethods]
impl PySemanticPublishedParent {
    /// Prepare one read of this exact publication's selected task receipts.
    /// Keep the returned owner through known completion, including Pending.
    fn prepare_selected_task_facts(
        slf: Py<Self>,
        py: Python<'_>,
    ) -> PyResult<Py<PySemanticSelectedTaskFactsObservation>> {
        let parent = slf.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let task = parent.task_use.borrow(py);
        let budget = task.checkpoint.material_sources.native_budget()?;
        let metadata = material_bytes::reserve_python_material::<
            PySemanticSelectedTaskFactsObservation,
        >(py, &budget)?;
        let native = {
            let owner = session.owner()?;
            task.require_current(&owner)?;
            parent.require_task(py, &task)?;
            owner
                .prepare_selected_task_facts_observation(&*parent.lease()?)
                .map_err(xlog_err)?
        };
        Py::new(
            py,
            PySemanticSelectedTaskFactsObservation {
                parent: slf.clone_ref(py),
                state: Mutex::new(SelectedTaskFactsState {
                    native,
                    completed: None,
                }),
                active: AtomicBool::new(false),
                _metadata: metadata,
            },
        )
    }
}

#[pymethods]
impl PySemanticSelectedTaskFactsObservation {
    /// Join the same original reads and return immutable actual selected facts.
    fn resolve_completion(
        slf: Py<Self>,
        py: Python<'_>,
    ) -> PyResult<Py<PySemanticCompletedTaskFacts>> {
        let observation = slf.borrow(py);
        observation
            .active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid("selected task observation completion cannot be nested"))?;
        let _active = PreparedRetirementGuard(&observation.active);
        let mut state = observation
            .state
            .try_lock()
            .map_err(|_| invalid("selected task observation custody is busy or poisoned"))?;
        if let Some(completed) = &state.completed {
            return Ok(completed.clone_ref(py));
        }
        let parent = observation.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let task = parent.task_use.borrow(py);
        let (result, pending) = {
            let mut owner = session.owner()?;
            task.require_original_identity(&owner)?;
            parent.require_task(py, &task)?;
            let result = owner.resolve_selected_task_facts(&*parent.lease()?, &mut state.native);
            let pending = result.is_err() && owner.selected_task_facts_pending(&state.native);
            (result, pending)
        };
        let result = match result {
            Ok(result) => result,
            Err(cause) if pending => {
                drop(state);
                let error = SemanticSelectedTaskFactsPending::new_err(
                    "the original selected task observation has not completed",
                );
                error.value(py).setattr("observation", slf.clone_ref(py))?;
                error.value(py).setattr("owner", slf.clone_ref(py))?;
                error.set_cause(py, Some(xlog_err(cause)));
                return Err(error);
            }
            Err(error) => return Err(xlog_err(error)),
        };
        if result.task_identity.as_bytes() != &task.task_identity {
            return Err(invalid(
                "selected task observation belongs to another original task",
            ));
        }
        {
            let owner = session.owner()?;
            if owner
                .published_identity(&*parent.lease()?)
                .map_err(xlog_err)?
                != result.parent
            {
                return Err(invalid(
                    "selected task observation belongs to another publication",
                ));
            }
        }
        let count = result
            .facts
            .truth
            .len()
            .checked_add(result.facts.attained.len())
            .and_then(|count| count.checked_add(result.facts.correct.as_ref().map_or(0, Vec::len)))
            .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
            .ok_or_else(|| invalid("selected task fact projection extent overflow"))?;
        let budget = task.checkpoint.material_sources.native_budget()?;
        let count = count
            .checked_add(material_bytes::python_instance_extent::<
                PySemanticCompletedTaskFacts,
            >(py)?)
            .and_then(|bytes| {
                bytes.checked_add(std::mem::size_of::<PySemanticCompletedTaskFacts>())
            })
            .ok_or_else(|| invalid("selected task fact owner extent overflow"))?;
        let metadata = budget
            .reserve_auxiliary(
                u64::try_from(count)
                    .map_err(|_| invalid("selected task fact projection exceeds u64"))?,
            )
            .map_err(xlog_err)?;
        let completed = Py::new(
            py,
            PySemanticCompletedTaskFacts {
                slot: result.selected_slot,
                inner: result.facts.clone(),
                _metadata: Some(metadata),
                _selected_owner: Some(result),
            },
        )?;
        state.completed = Some(completed.clone_ref(py));
        Ok(completed)
    }
}

#[pymethods]
impl PySemanticCompletedTaskFacts {
    /// Authenticate this cold result against its original selected publication.
    fn belongs_to(&self, py: Python<'_>, parent: &PySemanticPublishedParent) -> PyResult<bool> {
        let Some(selected) = &self._selected_owner else {
            return Ok(false);
        };
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        let task = parent.task_use.borrow(py);
        parent.require_task(py, &task)?;
        Ok(selected.task_identity.as_bytes() == &task.task_identity
            && owner
                .published_identity(&*parent.lease()?)
                .map_err(xlog_err)?
                == selected.parent)
    }
}
