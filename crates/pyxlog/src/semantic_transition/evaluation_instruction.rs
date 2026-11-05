//! Public read-only Evaluation at the original native Program/cold boundary.
use super::model_evaluation::{
    PySemanticCompletedModelEvaluation, PySemanticEvaluationCohort, PySemanticModelEvaluation,
    SemanticModelEvaluationPending,
};
use super::*;
use learning_phase::cold_model_work::PySemanticColdModelWork;
use learning_phase::phase_evaluation::{EvaluationColdStage, EvaluationColdVisibility};
use xlog_cuda::{
    SemanticCancelledModelEvaluation, SemanticColdModelWork, SemanticColdModelWorkDisposition,
    SemanticColdModelWorkRegion, SemanticColdModelWorkResult,
    SemanticEvaluationInstructionAdmission, SemanticModelEvaluation,
};

pub(super) struct EvaluationInstructionCustody {
    controller: Py<PySemanticTransitionController>,
    session: Py<PySemanticTransitionSession>,
    task: Py<PySemanticTransitionTaskUse>,
    parent: Py<PySemanticPublishedParent>,
    native: SemanticEvaluationInstructionAdmission,
    ordinal: u64,
    observer: learning_phase::ResourceObserver,
    state: Mutex<EvaluationInstructionState>,
}

struct EvaluationInstructionState {
    callback: Option<Py<PyAny>>,
    callback_entered: bool,
    callback_pending: bool,
    callback_result: Option<Py<PyAny>>,
    callback_error: Option<PyErr>,
    running: bool,
    observer_begin_entered: bool,
    work: Option<SemanticColdModelWork>,
    native_regions: Vec<SemanticColdModelWorkRegion>,
    regions: Vec<Py<PySemanticColdModelWork>>,
    work_started: bool,
    stage: EvaluationColdStage,
    cohort: Option<Py<PySemanticEvaluationCohort>>,
    evaluation: Option<Py<PySemanticModelEvaluation>>,
    native_evaluation: Option<SemanticModelEvaluation>,
    cancelled: Option<SemanticCancelledModelEvaluation>,
    streams: Option<Vec<u64>>,
    evaluation_retired: bool,
    work_closed: bool,
    cold_result: Option<SemanticColdModelWorkResult>,
    observer_finish_entered: bool,
    peak: Option<u64>,
    observation: Option<Py<PyDict>>,
    observer_release_entered: bool,
    completed: bool,
}

impl Default for EvaluationInstructionState {
    fn default() -> Self {
        Self {
            callback: None,
            callback_entered: false,
            callback_pending: false,
            callback_result: None,
            callback_error: None,
            running: false,
            observer_begin_entered: false,
            work: None,
            native_regions: Vec::new(),
            regions: Vec::new(),
            work_started: false,
            stage: EvaluationColdStage::Preparation,
            cohort: None,
            evaluation: None,
            native_evaluation: None,
            cancelled: None,
            streams: None,
            evaluation_retired: false,
            work_closed: false,
            cold_result: None,
            observer_finish_entered: false,
            peak: None,
            observation: None,
            observer_release_entered: false,
            completed: false,
        }
    }
}

#[pyclass(
    name = "SemanticEvaluationInstructionAdmission",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticEvaluationInstructionAdmission {
    inner: Arc<EvaluationInstructionCustody>,
}

pub(super) fn current(
    session: &PySemanticTransitionSession,
) -> PyResult<Option<Arc<EvaluationInstructionCustody>>> {
    Ok(session
        .evaluation_instruction
        .lock()
        .map_err(|_| invalid("original Evaluation instruction custody is poisoned"))?
        .as_ref()
        .map(Arc::clone))
}

fn native_error(py: Python<'_>, error: xlog_cuda::SemanticTransitionError) -> PyErr {
    let pending = error.native_completion_pending();
    let cause = xlog_err(error);
    if !pending {
        return cause;
    }
    let result = SemanticModelEvaluationPending::new_err(
        "the original Evaluation cold producer awaits completion; retain its admission and callback");
    result.set_cause(py, Some(cause));
    result
}

impl EvaluationInstructionCustody {
    fn state(&self) -> PyResult<std::sync::MutexGuard<'_, EvaluationInstructionState>> {
        self.state
            .lock()
            .map_err(|_| invalid("original Evaluation state is poisoned"))
    }

    fn require_owner(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        if current(&session)?
            .as_ref()
            .is_none_or(|owner| !std::ptr::eq(self, owner.as_ref()))
        {
            return Err(invalid(
                "Evaluation requires its original retained instruction owner",
            ));
        }
        self.task.borrow(py).issuance.require_current()?;
        Ok(())
    }

    pub(super) fn require_preparation(
        &self,
        py: Python<'_>,
        controller: &PySemanticTransitionController,
        task: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
        cohort: Option<&Py<PySemanticEvaluationCohort>>,
        continuing: bool,
    ) -> PyResult<()> {
        self.require_owner(py)?;
        let state = self.state()?;
        if !state.running
            || state.callback_result.is_some()
            || state.stage != EvaluationColdStage::Preparation
            || !continuing && state.evaluation.is_some()
            || !std::ptr::eq(controller, &*self.controller.borrow(py))
            || !std::ptr::eq(task, &*self.task.borrow(py))
            || !std::ptr::eq(parent, &*self.parent.borrow(py))
            || cohort.is_none_or(|cohort| {
                state
                    .cohort
                    .as_ref()
                    .is_none_or(|original| original.as_ptr() != cohort.as_ptr())
            })
        {
            return Err(invalid("Evaluation preparation changed its original callback, task, parent or frozen cohort"));
        }
        Ok(())
    }

    pub(super) fn bind_evaluation(
        &self,
        py: Python<'_>,
        evaluation: &Py<PySemanticModelEvaluation>,
        native: &SemanticModelEvaluation,
    ) -> PyResult<()> {
        self.require_owner(py)?;
        {
            let state = self.state()?;
            if !state.running
                || state.stage != EvaluationColdStage::Preparation
                || state
                    .evaluation
                    .as_ref()
                    .is_some_and(|original| original.as_ptr() != evaluation.as_ptr())
            {
                return Err(invalid("Evaluation changed its original numerical owner"));
            }
        }
        self.session
            .borrow(py)
            .owner()?
            .bind_evaluation_instruction(&self.native, &*self.parent.borrow(py).lease()?, native)
            .map_err(xlog_err)?;
        let mut state = self.state()?;
        state.native_evaluation = Some(native.clone());
        state.evaluation = Some(evaluation.clone_ref(py));
        Ok(())
    }

    fn require_evaluation(
        &self,
        py: Python<'_>,
        evaluation: &PySemanticModelEvaluation,
    ) -> PyResult<()> {
        self.require_owner(py)?;
        let state = self.state()?;
        if !state.running
            || state
                .evaluation
                .as_ref()
                .is_none_or(|original| !std::ptr::eq(&*original.borrow(py), evaluation))
        {
            return Err(invalid(
                "Evaluation changed its original numerical callback",
            ));
        }
        Ok(())
    }

    pub(super) fn retain_consumer_streams(
        &self,
        py: Python<'_>,
        evaluation: &PySemanticModelEvaluation,
        streams: &[u64],
    ) -> PyResult<()> {
        self.require_evaluation(py, evaluation)?;
        let mut state = self.state()?;
        if state
            .streams
            .as_ref()
            .is_some_and(|original| original.as_slice() != streams)
        {
            return Err(invalid(
                "Evaluation changed its original completion consumer roster",
            ));
        }
        state.streams = Some(streams.to_vec());
        Ok(())
    }

    pub(super) fn cancel_evaluation(
        &self,
        py: Python<'_>,
        evaluation: &PySemanticModelEvaluation,
        cancelled: &SemanticCancelledModelEvaluation,
    ) -> PyResult<()> {
        self.require_evaluation(py, evaluation)?;
        let mut state = self.state()?;
        if state.cancelled.is_some() {
            return Err(invalid(
                "Evaluation cancellation cannot replace its original proof",
            ));
        }
        state.cancelled = Some(cancelled.clone());
        state.stage = EvaluationColdStage::Cleanup;
        drop(state);
        drop(EvaluationColdVisibility {
            session: &self.session.borrow(py),
            py,
        });
        Ok(())
    }

    pub(super) fn cold_boundary(
        &self,
        py: Python<'_>,
        evaluation: &PySemanticModelEvaluation,
        from: EvaluationColdStage,
        to: EvaluationColdStage,
    ) -> PyResult<()> {
        self.require_evaluation(py, evaluation)?;
        let mut state = self.state()?;
        if from == EvaluationColdStage::AwaitingCompletion
            && to == EvaluationColdStage::Cleanup
            && state.stage == to
        {
            return Ok(());
        }
        if state.stage != from {
            return Err(invalid(
                "Evaluation cannot repeat or reorder its original cold regions",
            ));
        }
        let index = match (from, to) {
            (EvaluationColdStage::Preparation, EvaluationColdStage::OutputProjection) => Some(0),
            (EvaluationColdStage::OutputProjection, EvaluationColdStage::AwaitingCompletion) => {
                Some(1)
            }
            (EvaluationColdStage::AwaitingCompletion, EvaluationColdStage::Cleanup) => None,
            _ => return Err(invalid("Evaluation has no such numerical cold boundary")),
        };
        if let Some(index) = index {
            let region = state
                .regions
                .get(index)
                .ok_or_else(|| invalid("Evaluation lost its original cold region"))?
                .borrow(py);
            self.session
                .borrow(py)
                .owner()?
                .require_closed_cold_model_work_region(
                    region
                        .custody(py)?
                        .region
                        .as_ref()
                        .ok_or_else(|| invalid("Evaluation lost its original report region"))?,
                )
                .map_err(xlog_err)?;
        }
        state.stage = to;
        drop(state);
        drop(EvaluationColdVisibility {
            session: &self.session.borrow(py),
            py,
        });
        Ok(())
    }

    pub(super) fn current_work(
        &self,
        py: Python<'_>,
        task: &PySemanticTransitionTaskUse,
        activate: bool,
    ) -> PyResult<Py<PySemanticColdModelWork>> {
        self.require_owner(py)?;
        if !std::ptr::eq(task, &*self.task.borrow(py)) {
            return Err(invalid(
                "Evaluation cold work belongs to another original TaskUse",
            ));
        }
        let state = self.state()?;
        if !state.running || state.callback_result.is_some() {
            return Err(invalid(
                "Evaluation cold work requires its original active callback",
            ));
        }
        let index = match state.stage {
            EvaluationColdStage::Preparation => 0,
            EvaluationColdStage::OutputProjection => 1,
            EvaluationColdStage::Cleanup => 2,
            _ => {
                return Err(invalid(
                    "the original pending Evaluation has no active cold region",
                ))
            }
        };
        let work = state
            .regions
            .get(index)
            .ok_or_else(|| invalid("Evaluation lost its original cold registrar"))?
            .clone_ref(py);
        drop(state);
        if activate {
            let session = self.session.borrow(py);
            let mut visible = session
                .active_cold_model_work
                .lock()
                .map_err(|_| invalid("Evaluation cold visibility is poisoned"))?;
            if visible
                .as_ref()
                .is_some_and(|original| original.as_ptr() != work.as_ptr())
            {
                return Err(invalid(
                    "Evaluation cannot replace another active cold callback",
                ));
            }
            work.borrow(py).active.store(true, Ordering::Release);
            *visible = Some(work.clone_ref(py));
        }
        Ok(work)
    }

    pub(super) fn require_cold_callback(
        &self,
        py: Python<'_>,
        work: &PySemanticColdModelWork,
    ) -> PyResult<()> {
        let original = self.current_work(py, &self.task.borrow(py), false)?;
        if !std::ptr::eq(&*original.borrow(py), work) || !work.active.load(Ordering::Acquire) {
            return Err(invalid(
                "Evaluation cold work changed its original registrar",
            ));
        }
        let custody = work.custody(py)?;
        self.session
            .borrow(py)
            .owner()?
            .require_evaluation_instruction_cold_work(
                &self.native,
                &*self.parent.borrow(py).lease()?,
                &custody.inner,
            )
            .map_err(xlog_err)
    }

    fn initialize(&self, py: Python<'_>) -> PyResult<()> {
        self.require_owner(py)?;
        {
            let mut state = self.state()?;
            if !state.observer_begin_entered {
                state.observer_begin_entered = true;
                drop(state);
                self.observer.begin(py, self.ordinal)?;
            }
        }
        if self.state()?.work.is_none() {
            let work = self
                .session
                .borrow(py)
                .owner()?
                .prepare_evaluation_instruction_work(
                    &self.native,
                    &*self.parent.borrow(py).lease()?,
                )
                .map_err(|error| native_error(py, error))?;
            self.state()?.work = Some(work);
        }
        let work = self
            .state()?
            .work
            .clone()
            .expect("retained original Evaluation cold work");
        if self.state()?.native_regions.is_empty() {
            let regions = self
                .session
                .borrow(py)
                .owner()?
                .prepare_cold_model_work_regions(&work, 3)
                .map_err(xlog_err)?;
            self.state()?.native_regions = regions;
        }
        loop {
            let state = self.state()?;
            let index = state.regions.len();
            let Some(region) = state.native_regions.get(index).cloned() else {
                break;
            };
            drop(state);
            let original = Py::new(
                py,
                PySemanticColdModelWork::new(
                    Some(self.parent.clone_ref(py)),
                    self.parent.clone_ref(py),
                    work.clone(),
                    Some(region),
                    false,
                ),
            )?;
            self.state()?.regions.push(original);
        }
        if !self.state()?.work_started {
            self.session
                .borrow(py)
                .owner()?
                .begin_cold_model_work(&work)
                .map_err(xlog_err)?;
            self.state()?.work_started = true;
        }
        Ok(())
    }

    fn finish(&self, py: Python<'_>) -> PyResult<()> {
        if self.state()?.completed {
            return Ok(());
        }
        let (evaluation, completed, cancelled, retired, cohort) = {
            let state = self.state()?;
            if state.stage != EvaluationColdStage::Cleanup {
                return Err(invalid(
                    "Evaluation has not reached its original cleanup region",
                ));
            }
            (
                state
                    .native_evaluation
                    .clone()
                    .ok_or_else(|| invalid("Evaluation lost its original numerical invocation"))?,
                state
                    .callback_result
                    .as_ref()
                    .map(|result| result.clone_ref(py)),
                state.cancelled.clone(),
                state.evaluation_retired,
                state.cohort.as_ref().map(|cohort| cohort.clone_ref(py)),
            )
        };
        if cancelled.is_none() {
            let result = completed
                .as_ref()
                .ok_or_else(|| invalid("Evaluation lacks its original completed carrier"))?;
            let receipt = result
                .bind(py)
                .extract::<PyRef<'_, PySemanticCompletedModelEvaluation>>()?;
            if !receipt.original_receipt().belongs_to_original(&evaluation) {
                return Err(invalid(
                    "Evaluation returned another original completed carrier",
                ));
            }
            let cohort = cohort
                .as_ref()
                .ok_or_else(|| invalid("Evaluation lost its original frozen cohort"))?;
            receipt.require_original_observation(
                py,
                &self.parent.borrow(py),
                &cohort.borrow(py),
            )?;
        }
        // Drop only authenticated completed/cancelled numerical ownership,
        // outside the state mutex and before retiring its scratch aliases.
        let retained = self.state()?.evaluation.take();
        drop(retained);
        let session = self.session.borrow(py);
        if !retired {
            let mut owner = session.owner()?;
            let parent = self.parent.borrow(py);
            let lease = parent.lease()?;
            if let Some(cancelled) = &cancelled {
                owner
                    .retire_evaluation_instruction_cancellation(
                        &self.native,
                        &lease,
                        &evaluation,
                        cancelled,
                    )
                    .map_err(|error| native_error(py, error))?;
            } else {
                let result = completed
                    .as_ref()
                    .ok_or_else(|| invalid("Evaluation lacks its original completed carrier"))?;
                let receipt = result
                    .bind(py)
                    .extract::<PyRef<'_, PySemanticCompletedModelEvaluation>>()?;
                owner
                    .retire_evaluation_instruction_observation(
                        &self.native,
                        &lease,
                        &evaluation,
                        receipt.original_receipt(),
                    )
                    .map_err(|error| native_error(py, error))?;
            }
            self.state()?.evaluation_retired = true;
        }
        let work = self
            .state()?
            .work
            .clone()
            .expect("retained original cold work");
        if !self.state()?.work_closed {
            let mut owner = session.owner()?;
            if cancelled.is_some() {
                owner
                    .cancel_unentered_cold_model_work_regions(&work, 3)
                    .map_err(xlog_err)?;
            } else {
                for region in self.state()?.native_regions.iter() {
                    owner
                        .require_closed_cold_model_work_region(region)
                        .map_err(xlog_err)?;
                }
            }
            owner.close_cold_model_work(&work).map_err(xlog_err)?;
            self.state()?.work_closed = true;
        }
        if self.state()?.cold_result.is_none() {
            let streams =
                self.state()?.streams.clone().ok_or_else(|| {
                    invalid("Evaluation lost its actual completion consumer roster")
                })?;
            let result = session
                .owner()?
                .finish_evaluation_instruction_work(
                    &self.native,
                    &*self.parent.borrow(py).lease()?,
                    &work,
                    &streams,
                    if cancelled.is_some() {
                        SemanticColdModelWorkDisposition::KnownRefusal
                    } else {
                        SemanticColdModelWorkDisposition::Complete
                    },
                )
                .map_err(|error| native_error(py, error))?;
            self.state()?.cold_result = Some(result);
        }
        {
            let mut state = self.state()?;
            if !state.observer_finish_entered {
                state.observer_finish_entered = true;
                drop(state);
                self.observer.finish(py)?;
            }
        }
        if self.state()?.peak.is_none() {
            let peak = self.observer.backing_peak(py)?;
            self.state()?.peak = Some(peak);
        }
        if self.state()?.observation.is_none() {
            let peak = self.state()?.peak.expect("actual backing observation");
            let mut owner = session.owner()?;
            owner
                .complete_evaluation_instruction(
                    &self.native,
                    &*self.parent.borrow(py).lease()?,
                    peak,
                )
                .map_err(xlog_err)?;
            let observation = owner
                .evaluation_instruction_observation(&self.native)
                .map_err(xlog_err)?
                .ok_or_else(|| invalid("Evaluation has no original retired observation"))?;
            drop(owner);
            let observation = native_observation_report(py, observation)?;
            self.observer
                .observation_arguments(py, observation.bind(py))?;
            self.state()?.observation = Some(observation);
        }
        {
            let mut state = self.state()?;
            if state.observer_release_entered {
                return Err(invalid(
                    "unknown Evaluation observer release retains its original custody",
                ));
            }
            state.observer_release_entered = true;
        }
        self.observer.release(py)?;
        let (cohort, evaluation) = {
            let mut state = self.state()?;
            (state.cohort.take(), state.native_evaluation.take())
        };
        drop(cohort);
        drop(evaluation);
        let mut current = session
            .evaluation_instruction
            .lock()
            .map_err(|_| invalid("Evaluation instruction retention is poisoned"))?;
        if current
            .as_ref()
            .is_none_or(|owner| !std::ptr::eq(self, owner.as_ref()))
        {
            return Err(invalid(
                "Evaluation completion changed its original instruction owner",
            ));
        }
        let released = current.take();
        drop(current);
        drop(released);
        session.recording.store(false, Ordering::Release);
        self.state()?.completed = true;
        Ok(())
    }
}

#[pymethods]
impl PySemanticEvaluationInstructionAdmission {
    #[getter]
    fn cohort(&self, py: Python<'_>) -> PyResult<Py<PySemanticEvaluationCohort>> {
        self.inner.require_owner(py)?;
        {
            let state = self.inner.state()?;
            if !state.running || state.stage != EvaluationColdStage::Preparation {
                return Err(invalid(
                    "Evaluation cohort requires its original preparation callback",
                ));
            }
            if let Some(cohort) = &state.cohort {
                return Ok(cohort.clone_ref(py));
            }
        }
        let work = self
            .inner
            .state()?
            .work
            .clone()
            .ok_or_else(|| invalid("Evaluation lost its original cold work"))?;
        let cohort = self
            .inner
            .session
            .borrow(py)
            .owner()?
            .prepare_evaluation_instruction_cohort(
                &self.inner.native,
                &*self.inner.parent.borrow(py).lease()?,
                &work,
            )
            .map_err(|error| native_error(py, error))?;
        let cohort = Py::new(py, PySemanticEvaluationCohort { inner: cohort })?;
        self.inner.state()?.cohort = Some(cohort.clone_ref(py));
        Ok(cohort)
    }

    #[getter]
    fn resources_retired(&self, py: Python<'_>) -> PyResult<bool> {
        self.inner.session.borrow(py).require_creator()?;
        Ok(self.inner.state()?.completed)
    }

    #[getter]
    fn observation(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.inner.session.borrow(py).require_creator()?;
        let state = self.inner.state()?;
        Ok(if state.completed {
            state.observation.as_ref().map(|value| value.clone_ref(py))
        } else {
            None
        })
    }

    #[getter]
    fn cold_work_result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.session.borrow(py).require_creator()?;
        let Some(result) = self.inner.state()?.cold_result else {
            return Ok(py.None());
        };
        Ok((
            result.model_work,
            result.operation_count,
            result.work_bound,
            result.model_calls,
            result.native_work,
            PyTuple::new(py, result.native_events)?,
        )
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }

    fn execute(&self, py: Python<'_>, callback: Py<PyAny>) -> PyResult<Py<PyAny>> {
        self.inner.session.borrow(py).require_creator()?;
        if !callback.bind(py).is_callable() {
            return Err(invalid(
                "Evaluation requires its original numerical callback",
            ));
        }
        {
            let mut state = self.inner.state()?;
            if state
                .callback
                .as_ref()
                .is_some_and(|original| original.as_ptr() != callback.as_ptr())
            {
                return Err(invalid("Evaluation cannot replace its original callback"));
            }
            if state.completed {
                return match &state.callback_error {
                    Some(error) => Err(error.clone_ref(py)),
                    None => Ok(state
                        .callback_result
                        .as_ref()
                        .expect("completed carrier")
                        .clone_ref(py)),
                };
            }
            if state.callback_error.is_some()
                && !state.callback_pending
                && state.callback_result.is_none()
                && state.cancelled.is_none()
            {
                return Err(state
                    .callback_error
                    .as_ref()
                    .expect("retained original failure")
                    .clone_ref(py));
            }
            if state.running
                || state.callback_entered
                    && !state.callback_pending
                    && state.callback_result.is_none()
                    && state.cancelled.is_none()
            {
                return Err(state
                    .callback_error
                    .as_ref()
                    .map(|error| error.clone_ref(py))
                    .unwrap_or_else(|| {
                        invalid("retain the original unresolved Evaluation callback")
                    }));
            }
            if state.callback.is_none() {
                state.callback = Some(callback.clone_ref(py));
            }
        }
        if let Err(error) = self.inner.initialize(py) {
            if !error.is_instance_of::<SemanticModelEvaluationPending>(py) {
                self.inner.state()?.callback_error = Some(error.clone_ref(py));
            }
            return Err(error);
        }
        let call_needed = {
            let state = self.inner.state()?;
            state.callback_result.is_none() && state.cancelled.is_none()
        };
        if call_needed {
            {
                let mut state = self.inner.state()?;
                state.callback_entered = true;
                state.callback_pending = false;
                state.running = true;
            }
            let returned = {
                let session = self.inner.session.borrow(py);
                let _visibility = EvaluationColdVisibility {
                    session: &session,
                    py,
                };
                callback.bind(py).call0().map(Bound::unbind)
            };
            let mut state = self.inner.state()?;
            state.running = false;
            match returned {
                Ok(result) => state.callback_result = Some(result),
                Err(error) => {
                    state.callback_pending =
                        error.is_instance_of::<SemanticModelEvaluationPending>(py);
                    if state.callback_pending {
                        return Err(error);
                    }
                    state.callback_error = Some(error.clone_ref(py));
                    if state.cancelled.is_none() {
                        return Err(error);
                    }
                }
            }
        }
        self.inner.finish(py)?;
        let state = self.inner.state()?;
        match &state.callback_error {
            Some(error) => Err(error.clone_ref(py)),
            None => Ok(state
                .callback_result
                .as_ref()
                .expect("original completed carrier")
                .clone_ref(py)),
        }
    }
}

#[pymethods]
impl PySemanticTransitionController {
    #[pyo3(signature = (task_use, *, parent, instruction_bytes, program_ordinal, resource_observer))]
    fn admit_evaluation_instruction(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
        parent: Py<PySemanticPublishedParent>,
        instruction_bytes: &Bound<'_, PyAny>,
        program_ordinal: &Bound<'_, PyAny>,
        resource_observer: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PySemanticEvaluationInstructionAdmission>> {
        let controlled = slf.borrow();
        let session = controlled.session.borrow(py);
        session.require_creator()?;
        let issued = task_use.borrow(py);
        controlled.require_read_issued(&issued)?;
        issued.state()?.require_public_use()?;
        parent.borrow(py).require_task(py, &issued)?;
        if parent.borrow(py).session.as_ptr() != controlled.session.as_ptr()
            || session.importing.load(Ordering::Acquire)
            || session.recording.load(Ordering::Acquire)
            || session.retiring.load(Ordering::Acquire)
            || session.learning_preparing.load(Ordering::Acquire)
            || session
                .segment_instruction
                .lock()
                .map_err(|_| invalid("original instruction retention is poisoned"))?
                .is_some()
            || session
                .prepared_segment
                .lock()
                .map_err(|_| invalid("prepared segment retention is poisoned"))?
                .is_some()
        {
            return Err(invalid(
                "Evaluation requires its original quiescent public Session",
            ));
        }
        let mut retained = session
            .evaluation_instruction
            .lock()
            .map_err(|_| invalid("Evaluation instruction retention is poisoned"))?;
        if retained.is_some() {
            return Err(invalid(
                "the original Evaluation instruction has not retired",
            ));
        }
        let ordinal = ColdValue::read(program_ordinal, &mut 128, 0)?.unsigned()?;
        let bytes = exact_replay_bytes(instruction_bytes)?;
        let device = session
            .owner()?
            .resource_device_identity()
            .map_err(xlog_err)?;
        let observer = learning_phase::ResourceObserver::capture(
            resource_observer.cast::<pyo3::types::PyCapsule>()?,
            device,
        )?;
        let native = session
            .owner()?
            .admit_evaluation_instruction(&*parent.borrow(py).lease()?, ordinal, bytes)
            .map_err(xlog_err)?;
        let custody = Arc::new(EvaluationInstructionCustody {
            controller: slf.clone().unbind(),
            session: controlled.session.clone_ref(py),
            task: task_use.clone_ref(py),
            parent,
            native,
            ordinal,
            observer,
            state: Mutex::new(EvaluationInstructionState::default()),
        });
        *retained = Some(Arc::clone(&custody));
        session.recording.store(true, Ordering::Release);
        drop(retained);
        Py::new(
            py,
            PySemanticEvaluationInstructionAdmission { inner: custody },
        )
    }
}
