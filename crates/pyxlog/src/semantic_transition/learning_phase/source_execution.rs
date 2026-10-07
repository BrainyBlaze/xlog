//! The original source observations, before any private trajectory is restored.

use super::super::model_evaluation::{
    PySemanticCompletedModelEvaluation, PySemanticEvaluationCohort, SemanticModelEvaluationPending,
};
use super::*;
use xlog_cuda::{SemanticColdModelWork, SemanticColdModelWorkResult};

pub(super) struct SourceEvaluation {
    entries: Py<PyTuple>,
    material: Vec<u8>,
    instruction: Vec<u8>,
    ordinal: u64,
    step: u64,
    budget: [u64; 3],
    started: bool,
    ready: bool,
    evaluation_admitted: bool,
    callback_entered: bool,
    callback_pending: bool,
    callback_result: Option<Py<PyAny>>,
    callback_error: Option<PyErr>,
    work: Option<SemanticColdModelWork>,
    work_closed: bool,
    work_finish_entered: bool,
    work_result: Option<SemanticColdModelWorkResult>,
    observer_finish_entered: bool,
    record_entered: bool,
    recorded: bool,
}

impl PySemanticLearningPhaseTransition {
    pub(in crate::semantic_transition) fn require_source_feedback_projection(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        parent: &Py<PySemanticPublishedParent>,
    ) -> PyResult<()> {
        if !std::ptr::eq(session, &*self.source.borrow(py))
            || parent.as_ptr() != self.parent.as_ptr()
            || !session.learning_preparing.load(Ordering::Acquire)
        {
            return Err(invalid(
                "source feedback projection changed its original Session or held parent",
            ));
        }
        self.records()?.require_preparation_admission()?;
        feedback_materials(
            self.preparation_inputs
                .feedback_interventions
                .bind(py)
                .as_any(),
        )?;
        let cold_callback = session
            .active_cold_model_work
            .lock()
            .map_err(|_| invalid("active cold callback owner mutex is poisoned"))?
            .is_some();
        if !cold_callback && !self.source_evaluation_active.load(Ordering::Acquire) {
            return Err(invalid("held source feedback is readable only inside its original serialization or evaluation callback"));
        }
        if self.source_evaluation_active.load(Ordering::Acquire) {
            let retained = self.source_evaluations()?;
            let current = retained
                .last()
                .ok_or_else(|| invalid("source feedback lost its original evaluation entry"))?;
            Self::require_source_evaluation_entry(py, current)?;
            if !current.ready || !current.evaluation_admitted || current.callback_result.is_some() {
                return Err(invalid(
                    "source feedback lost its actual admitted source evaluation",
                ));
            }
        }
        Ok(())
    }

    fn source_evaluations(&self) -> PyResult<MutexGuard<'_, Vec<SourceEvaluation>>> {
        self.source_evaluations
            .lock()
            .map_err(|_| invalid("original source evaluation custody mutex is poisoned"))
    }

    /// This is not general public admission. Only the one retained original
    /// source callback can issue its read-only evaluation, once, on that parent.
    pub(in crate::semantic_transition) fn require_source_evaluation(
        &self,
        py: Python<'_>,
        controller: &PySemanticTransitionController,
        task: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
    ) -> PyResult<()> {
        if !self.source_evaluation_active.load(Ordering::Acquire)
            || !std::ptr::eq(controller, &*self.source_controller.borrow(py))
            || !std::ptr::eq(task, &*self.task_use.borrow(py))
            || !std::ptr::eq(parent, &*self.parent.borrow(py))
            || !matches!(task.state()?.phase, TaskUsePhase::ArenaPreparing(_))
        {
            return Err(invalid("private source evaluation requires its original active callback and exact source owners"));
        }
        self.records()?.require_preparation_admission()?;
        self.preparation_inputs
            .require_program(py, &self.scientific_owner)?;
        let mut retained = self.source_evaluations()?;
        let current = retained
            .last_mut()
            .ok_or_else(|| invalid("source evaluation lost its original scheduled invocation"))?;
        Self::require_source_evaluation_entry(py, current)?;
        if !current.started
            || !current.callback_entered
            || current.callback_result.is_some()
            || current.evaluation_admitted
            || current.record_entered
        {
            return Err(invalid(
                "the original source evaluation cannot be replaced or admitted twice",
            ));
        }
        current.evaluation_admitted = true;
        Ok(())
    }

    fn require_source_evaluation_entry(py: Python<'_>, current: &SourceEvaluation) -> PyResult<()> {
        let (material, instruction) = Self::singleton_lifecycle_material(current.entries.bind(py))?;
        if material != current.material || instruction != current.instruction {
            return Err(invalid(
                "source evaluation changed its original frozen entry",
            ));
        }
        Ok(())
    }

    pub(super) fn record_source_preparation(&self, py: Python<'_>, peak: u64) -> PyResult<()> {
        let (instruction, result, recorded, entered) = {
            let retained = self.source_preparation()?;
            let source = retained.as_ref().expect("retained original preparation");
            (
                source.instruction.clone(),
                source.model_work_result.ok_or_else(|| {
                    invalid("source preparation lacks its actual joined expenditure")
                })?,
                source.recorded,
                source.record_entered,
            )
        };
        if recorded {
            return Ok(());
        }
        if entered {
            return Err(invalid(
                "unknown source history append cannot repeat its original record callback",
            ));
        }
        let manifest = SemanticCheckpointManifest::decode(&self.source_checkpoint)?;
        let native = SemanticTransitionSession::state_material_input_projection(&manifest.native)
            .map_err(xlog_err)?;
        let work = result
            .model_work
            .checked_add(result.native_work)
            .ok_or_else(|| invalid("source preparation work overflowed"))?;
        let arguments = PyDict::new(py);
        arguments.set_item("instruction_bytes", PyBytes::new(py, &instruction))?;
        arguments.set_item("operation_ordinal", 0)?;
        arguments.set_item("source_parent", self.parent.borrow(py).identity(py)?)?;
        arguments.set_item(
            "model_binding",
            (
                native.model_generation,
                PyBytes::new(py, native.model_geometry_digest.as_bytes()),
                PyBytes::new(py, native.model_numerical_digest.as_bytes()),
            ),
        )?;
        arguments.set_item(
            "source_checkpoint",
            PyBytes::new(py, &self.source_checkpoint),
        )?;
        arguments.set_item("resource_usage", (work, peak, result.model_calls))?;
        let callback = self
            .scientific_owner
            .bind(py)
            .getattr("record_preparation")?;
        self.source_preparation()?
            .as_mut()
            .expect("retained original preparation")
            .record_entered = true;
        callback.call((), Some(&arguments))?;
        self.source_preparation()?
            .as_mut()
            .expect("retained original preparation")
            .recorded = true;
        self.preparation_inputs.resource_observer.release(py)
    }

    /// Scientific projects its complete original schedule. Native retains each
    /// tuple before admission; the original numerical callback owns its decoder
    /// and, when pending, the same suspended evaluation frame.
    pub(super) fn execute_source_evaluations(&self, py: Python<'_>) -> PyResult<()> {
        loop {
            let unfinished = self
                .source_evaluations()?
                .last()
                .is_some_and(|evaluation| !evaluation.recorded);
            if !unfinished {
                self.preparation_inputs.resource_observer.release(py)?;
                self.preparation_inputs
                    .require_program(py, &self.scientific_owner)?;
                let next = self.scientific_owner.bind(py).getattr("next_operation")?;
                if !next.is_exact_instance_of::<PyTuple>() {
                    return Err(invalid(
                        "source execution lost the original complete scheduled group",
                    ));
                }
                let entries = next.cast::<PyTuple>()?;
                let first = entries.get_item(0)?;
                if !first.is_exact_instance_of::<PyDict>() {
                    return Err(invalid(
                        "source execution requires an original exact scheduled entry",
                    ));
                }
                let first = first.cast::<PyDict>()?;
                let field = |name| -> PyResult<ColdValue> {
                    ColdValue::read(
                        &first
                            .get_item(name)?
                            .ok_or_else(|| invalid("source evaluation lost a scheduled field"))?,
                        &mut (16 * 1024 * 1024),
                        0,
                    )
                };
                if field("branch")?.text()? != "source" {
                    return Ok(());
                }
                let (material, instruction) = Self::singleton_lifecycle_material(entries)?;
                if field("operation")?.text()? != "evaluation" {
                    return Err(invalid("source model may execute only its original read-only evaluations after preparation"));
                }
                let ordinal = field("operation_ordinal")?.unsigned()?;
                let expected = u64::try_from(self.source_evaluations()?.len())
                    .ok()
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| invalid("source evaluation ordinal overflowed"))?;
                if ordinal != expected {
                    return Err(invalid(
                        "source evaluation changed its original contiguous schedule position",
                    ));
                }
                let step = field("step_ordinal")?.unsigned()?;
                let budget = field("budget")?;
                let budget = budget.fields(3)?;
                let budget = [
                    budget[0].unsigned()?,
                    budget[1].unsigned()?,
                    budget[2].unsigned()?,
                ];
                self.source_evaluations()?.push(SourceEvaluation {
                    entries: entries.clone().unbind(),
                    material,
                    instruction,
                    ordinal,
                    step,
                    budget,
                    started: false,
                    ready: false,
                    evaluation_admitted: false,
                    callback_entered: false,
                    callback_pending: false,
                    callback_result: None,
                    callback_error: None,
                    work: None,
                    work_closed: false,
                    work_finish_entered: false,
                    work_result: None,
                    observer_finish_entered: false,
                    record_entered: false,
                    recorded: false,
                });
            }
            self.execute_source_evaluation(py)?;
        }
    }

    fn execute_source_evaluation(&self, py: Python<'_>) -> PyResult<()> {
        let source = self.source.borrow(py);
        let task = self.task_use.borrow(py);
        let parent = self.parent.borrow(py);
        let manifest = SemanticCheckpointManifest::decode(&self.source_checkpoint)?;
        let (_, saved_snapshot, _, _) = TaskCheckpointSeed::decode(&manifest.task)?;
        let inputs = &self.preparation_inputs;
        inputs.require_execution_inputs(py)?;
        inputs.require_program(py, &self.scientific_owner)?;
        self.records()?.require_preparation_admission()?;
        let fresh = self.refresh_snapshot.bind(py).call0()?;
        let authority =
            AuthoritySnapshot::parse(&ColdValue::read(&fresh, &mut (16 * 1024 * 1024), 0)?)?;
        authority.newer_than(&task.state()?.snapshot)?;
        check_learning_grant(&task, &self.grant_reference, &authority)?;
        let (entries, ordinal, started, result, callback_pending, callback_entered) = {
            let retained = self.source_evaluations()?;
            let current = retained.last().expect("retained original evaluation");
            Self::require_source_evaluation_entry(py, current)?;
            if let Some(error) = &current.callback_error {
                return Err(error.clone_ref(py));
            }
            (
                current.entries.clone_ref(py),
                current.ordinal,
                current.started,
                current
                    .callback_result
                    .as_ref()
                    .map(|result| result.clone_ref(py)),
                current.callback_pending,
                current.callback_entered,
            )
        };
        let streams = inputs.consumer_streams.python_value(py)?;
        let streams = checkpoint_consumer_streams(streams.bind(py), &mut (16 * 1024 * 1024))?;
        if !started {
            // Retain entry before begin: a failed begin or allocation is not
            // proof of nonentry, and may not start a replacement interval.
            self.source_evaluations()?
                .last_mut()
                .expect("retained evaluation")
                .started = true;
            inputs.resource_observer.begin(py, ordinal)?;
            let admission = self.records()?.confirmed_admission()?;
            let work = source
                .owner()?
                .prepare_cold_model_work(
                    &*parent.lease()?,
                    inputs.cold_model_work_capacity,
                    ordinal,
                    admission,
                )
                .map_err(xlog_err)?;
            self.source_evaluations()?
                .last_mut()
                .expect("retained evaluation")
                .work = Some(work.clone());
            source
                .owner()?
                .begin_cold_model_work(&work)
                .map_err(xlog_err)?;
            verify_original_source(py, &task, &parent, &manifest, &saved_snapshot)?;
            task.state()?.snapshot = authority;
            self.source_evaluations()?
                .last_mut()
                .expect("retained evaluation")
                .ready = true;
        }
        if !self
            .source_evaluations()?
            .last()
            .expect("retained evaluation")
            .ready
        {
            return Err(invalid("unknown source evaluation admission retains its original interval and native owners; it cannot enter or repeat numerical execution"));
        }
        let result = if let Some(result) = result {
            result
        } else {
            if callback_entered && !callback_pending {
                return Err(invalid(
                    "unknown source evaluation cannot repeat its original callback",
                ));
            }
            // Only the original pending numerical exception admits continuation
            // into its retained frame. Other exceptions are terminal custody.
            let arguments = (
                self.source_controller.clone_ref(py),
                self.task_use.clone_ref(py),
                self.parent.clone_ref(py),
                inputs.source_model.clone_ref(py),
                entries,
            );
            self.source_evaluations()?
                .last_mut()
                .expect("retained evaluation")
                .callback_entered = true;
            let _active = PhaseOperation::begin(&self.source_evaluation_active)?;
            let returned = inputs.execute_phase_instruction.bind(py).call1(arguments);
            let mut retained = self.source_evaluations()?;
            let current = retained.last_mut().expect("retained evaluation");
            match returned {
                Ok(returned) => {
                    current.callback_pending = false;
                    current.callback_result = Some(returned.clone().unbind());
                    returned.unbind()
                }
                Err(error) => {
                    current.callback_pending =
                        error.is_instance_of::<SemanticModelEvaluationPending>(py);
                    if !current.callback_pending {
                        current.callback_error = Some(error.clone_ref(py));
                    }
                    return Err(error);
                }
            }
        };
        let result = result.bind(py).cast::<PyTuple>()?;
        if !result.is_exact_instance_of::<PyTuple>()
            || result.len() != 3
            || result.get_item(0)?.as_ptr() != self.parent.as_ptr()
            || result.get_item(1)?.as_ptr() != inputs.source_model.as_ptr()
        {
            return Err(invalid(
                "source evaluation changed its actual original parent/model handoff",
            ));
        }
        let rows = result.get_item(2)?;
        if !rows.is_exact_instance_of::<PyTuple>() || rows.cast::<PyTuple>()?.len() != 1 {
            return Err(invalid(
                "source evaluation lost its original singleton observation",
            ));
        }
        let row = rows.cast::<PyTuple>()?.get_item(0)?;
        let row = row.cast::<PyTuple>()?;
        if !row.is_exact_instance_of::<PyTuple>() || row.len() != 3 {
            return Err(invalid(
                "source evaluation changed its actual observation/receipt/cohort triple",
            ));
        }
        let observation = row.get_item(0)?;
        if !observation.is_exact_instance_of::<PyDict>() {
            return Err(invalid(
                "source evaluation requires its actual numerical observation",
            ));
        }
        let observation = observation.cast::<PyDict>()?;
        let receipt = row.get_item(1)?;
        let receipt = receipt.extract::<PyRef<'_, PySemanticCompletedModelEvaluation>>()?;
        let cohort = row.get_item(2)?;
        let cohort = cohort.extract::<PyRef<'_, PySemanticEvaluationCohort>>()?;
        let measured = receipt.require_phase_observation(py, &parent, &cohort)?;
        let (work, closed, finish_entered, work_result) = {
            let retained = self.source_evaluations()?;
            let current = retained.last().expect("retained evaluation");
            (
                current
                    .work
                    .clone()
                    .ok_or_else(|| invalid("evaluation lost its native expense owner"))?,
                current.work_closed,
                current.work_finish_entered,
                current.work_result,
            )
        };
        if !closed {
            verify_original_source(py, &task, &parent, &manifest, &saved_snapshot)?;
            source
                .owner()?
                .close_cold_model_work(&work)
                .map_err(xlog_err)?;
            self.source_evaluations()?
                .last_mut()
                .expect("retained evaluation")
                .work_closed = true;
        }
        let native = match work_result {
            Some(result) => result,
            None => {
                self.source_evaluations()?
                    .last_mut()
                    .expect("retained evaluation")
                    .work_finish_entered = true;
                let result = if finish_entered {
                    source
                        .owner()?
                        .resolve_cold_model_work(&*parent.lease()?, &work)
                } else {
                    source
                        .owner()?
                        .finish_cold_model_work(&*parent.lease()?, &work, &streams)
                }
                .map_err(xlog_err)?;
                self.source_evaluations()?
                    .last_mut()
                    .expect("retained evaluation")
                    .work_result = Some(result);
                result
            }
        };
        let finish_needed = {
            let mut retained = self.source_evaluations()?;
            let current = retained.last_mut().expect("retained evaluation");
            let needed = !current.observer_finish_entered;
            current.observer_finish_entered = true;
            needed
        };
        if finish_needed {
            inputs.resource_observer.finish(py)?;
        }
        let peak = inputs.resource_observer.physical_peak(py)?;
        let work = measured
            .model_work
            .checked_add(native.native_work)
            .and_then(|work| work.checked_add(native.model_work))
            .ok_or_else(|| invalid("source evaluation expense overflowed"))?;
        let calls = measured
            .model_calls
            .checked_add(native.model_calls)
            .ok_or_else(|| invalid("source evaluation model calls overflowed"))?;
        let (instruction, step, budget, entered) = {
            let retained = self.source_evaluations()?;
            let current = retained.last().expect("retained evaluation");
            (
                current.instruction.clone(),
                current.step,
                current.budget,
                current.record_entered,
            )
        };
        if entered {
            return Err(invalid(
                "unknown evaluation history append cannot repeat its original record callback",
            ));
        }
        let field = |name| {
            observation
                .get_item(name)?
                .ok_or_else(|| invalid("evaluation lost an original numerical result field"))
        };
        let arguments = PyDict::new(py);
        arguments.set_item("instruction_bytes", PyBytes::new(py, &instruction))?;
        arguments.set_item("operation_ordinal", ordinal)?;
        arguments.set_item("branch", "source")?;
        arguments.set_item("step_ordinal", step)?;
        for name in [
            "completed_model_binding",
            "objective",
            "cohort_bytes",
            "refusal",
        ] {
            arguments.set_item(name, field(name)?)?;
        }
        arguments.set_item("resource_usage", (work, peak, calls))?;
        let callback = self
            .scientific_owner
            .bind(py)
            .getattr("record_evaluation")?;
        self.source_evaluations()?
            .last_mut()
            .expect("retained evaluation")
            .record_entered = true;
        callback.call((), Some(&arguments))?;
        self.source_evaluations()?
            .last_mut()
            .expect("retained evaluation")
            .recorded = true;
        inputs.resource_observer.release(py)?;
        if work > budget[0] || peak > budget[1] || calls > budget[2] {
            return Err(invalid("source evaluation exceeded its original operation budget; its actual observation is retained"));
        }
        Ok(())
    }
}
