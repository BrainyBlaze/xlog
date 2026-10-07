//! Original read-only phase observations over held source or private model owners.

use super::super::model_evaluation::{
    PySemanticCompletedModelEvaluation, PySemanticEvaluationCohort, PySemanticModelEvaluation,
    SemanticModelEvaluationPending,
};
use super::*;
use xlog_cuda::{
    SemanticCancelledModelEvaluation, SemanticColdModelWork, SemanticColdModelWorkResult,
    SemanticColdNativeWork, SemanticModelEvaluation,
};

pub(super) struct EvaluationOwners {
    pub(super) controller: Py<PySemanticTransitionController>,
    pub(super) task: Py<PySemanticTransitionTaskUse>,
    pub(super) parent: Py<PySemanticPublishedParent>,
    pub(super) model: Py<PyAny>,
}

pub(super) struct PhaseEvaluation {
    owners: EvaluationOwners,
    branch: &'static str,
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
    regions: Vec<Py<PySemanticColdModelWork>>,
    cold_stage: EvaluationColdStage,
    native_evaluation: Option<Py<PySemanticModelEvaluation>>,
    cancelled: Option<SemanticCancelledModelEvaluation>,
    cancelled_expense: Option<(u64, u64, u64)>,
    custody: Option<SemanticColdNativeWork>,
    work_closed: bool,
    work_result: Option<SemanticColdModelWorkResult>,
    observer_finish_entered: bool,
    record_entered: bool,
    recorded: bool,
    physical_peak: Option<u64>,
    released: bool,
    budget_exceeded: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::semantic_transition) enum EvaluationColdStage {
    Preparation,
    OutputProjection,
    AwaitingCompletion,
    Cleanup,
}

struct EvaluationColdVisibility<'a> {
    session: &'a PySemanticTransitionSession,
    py: Python<'a>,
}

impl Drop for EvaluationColdVisibility<'_> {
    fn drop(&mut self) {
        let original = self
            .session
            .active_cold_model_work
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(original) = original {
            original
                .borrow(self.py)
                .active
                .store(false, Ordering::Release);
        }
    }
}

impl PySemanticLearningPhaseTransition {
    pub(super) fn drop_completed_evaluation_owners(&self, branch: &'static str) -> PyResult<()> {
        let mut retained = self.phase_evaluations()?;
        if retained
            .iter()
            .filter(|entry| entry.branch == branch)
            .any(|entry| !entry.recorded || !entry.released || entry.budget_exceeded)
        {
            return Err(invalid(
                "model retirement cannot discard an unfinished evaluation",
            ));
        }
        let mut original = Vec::new();
        let mut index = 0;
        while index < retained.len() {
            if retained[index].branch == branch {
                original.push(retained.remove(index));
            } else {
                index += 1;
            }
        }
        drop(retained);
        drop(original);
        Ok(())
    }

    fn private_evaluation_completion(&self, branch: &'static str) -> PyResult<&AtomicBool> {
        match branch {
            "control" => Ok(&self.control_evaluations_done),
            "real" => Ok(&self.real_evaluations_done),
            _ => Err(invalid(
                "private evaluations require an original private branch",
            )),
        }
    }

    pub(super) fn private_checkpoint_input(
        &self,
        py: Python<'_>,
        branch: &'static str,
    ) -> PyResult<(EvaluationOwners, u64)> {
        if !self
            .private_evaluation_completion(branch)?
            .load(Ordering::Acquire)
        {
            return Err(invalid(
                "private checkpoint precedes its complete original evaluations",
            ));
        }
        let (owners, mut ordinal) = self.private_group_successor_input(py, branch)?;
        if let Some(evaluation) = self
            .phase_evaluations()?
            .last()
            .filter(|entry| entry.branch == branch)
        {
            if !evaluation.recorded || !evaluation.released || evaluation.budget_exceeded {
                return Err(invalid(
                    "private checkpoint precedes known evaluation accounting and release",
                ));
            }
            ordinal = evaluation
                .ordinal
                .checked_add(1)
                .ok_or_else(|| invalid("private checkpoint position overflowed"))?;
        }
        Ok((owners, ordinal))
    }

    pub(in crate::semantic_transition) fn cancel_phase_evaluation(
        &self,
        py: Python<'_>,
        original: &PySemanticModelEvaluation,
        cancelled: &SemanticCancelledModelEvaluation,
        native: &SemanticModelEvaluation,
        parent: &PySemanticPublishedParent,
    ) -> PyResult<()> {
        let mut retained = self.phase_evaluations()?;
        let current = retained
            .last_mut()
            .ok_or_else(|| invalid("cancelled evaluation lost its original phase owner"))?;
        Self::require_evaluation_entry(py, current)?;
        if !self.phase_evaluation_active.load(Ordering::Acquire)
            || current
                .native_evaluation
                .as_ref()
                .is_none_or(|owner| !std::ptr::eq(&*owner.borrow(py), original))
            || !std::ptr::eq(&*current.owners.parent.borrow(py), parent)
            || current.cancelled.is_some()
        {
            return Err(invalid(
                "evaluation cancellation changed its original invocation or parent",
            ));
        }
        let session = parent.session.borrow(py);
        let owner = session.owner()?;
        if !cancelled.belongs_to(&owner, native)
            || cancelled.parent()
                != owner
                    .published_identity(&*parent.lease()?)
                    .map_err(xlog_err)?
        {
            return Err(invalid(
                "cancelled evaluation lost its original native expenditure proof",
            ));
        }
        drop(owner);
        // Retain the actual cancel proof before attempting any further state
        // transition. Failed/entered cold registration remains quarantined.
        current.cancelled = Some(cancelled.clone());
        self.source
            .borrow(py)
            .owner()?
            .cancel_unentered_cold_model_work_regions(
                current
                    .work
                    .as_ref()
                    .ok_or_else(|| invalid("cancelled evaluation lost its original cold report"))?,
                2,
            )
            .map_err(xlog_err)?;
        current.cold_stage = EvaluationColdStage::Cleanup;
        Self::clear_evaluation_cold_visibility(py, &session);
        Ok(())
    }

    pub(in crate::semantic_transition) fn bind_phase_evaluation(
        &self,
        py: Python<'_>,
        original: &Py<PySemanticModelEvaluation>,
    ) -> PyResult<()> {
        let mut retained = self.phase_evaluations()?;
        let current = retained
            .last_mut()
            .ok_or_else(|| invalid("phase evaluation lost its original admission"))?;
        Self::require_evaluation_entry(py, current)?;
        if !self.phase_evaluation_active.load(Ordering::Acquire)
            || !current.evaluation_admitted
            || current.native_evaluation.is_some()
        {
            return Err(invalid(
                "phase evaluation cannot replace its original native invocation",
            ));
        }
        current.native_evaluation = Some(original.clone_ref(py));
        Ok(())
    }

    pub(in crate::semantic_transition) fn evaluation_cold_boundary(
        &self,
        py: Python<'_>,
        original: &PySemanticModelEvaluation,
        from: EvaluationColdStage,
        to: EvaluationColdStage,
    ) -> PyResult<()> {
        let mut retained = self.phase_evaluations()?;
        let current = retained
            .last_mut()
            .ok_or_else(|| invalid("phase evaluation lost its original cold lifecycle"))?;
        Self::require_evaluation_entry(py, current)?;
        if !self.phase_evaluation_active.load(Ordering::Acquire)
            || current
                .native_evaluation
                .as_ref()
                .is_none_or(|owner| !std::ptr::eq(&*owner.borrow(py), original))
        {
            return Err(invalid(
                "phase evaluation changed its actual native cold boundary",
            ));
        }
        // Known completion may restore the same already retained receipt after
        // a late phase-visibility failure. It cannot mint another cleanup region.
        if from == EvaluationColdStage::AwaitingCompletion
            && to == EvaluationColdStage::Cleanup
            && current.cold_stage == to
        {
            return Ok(());
        }
        if current.cold_stage != from {
            return Err(invalid(
                "phase evaluation cannot repeat or reorder its cold regions",
            ));
        }
        let index = match from {
            EvaluationColdStage::Preparation => Some(0),
            EvaluationColdStage::OutputProjection => Some(1),
            EvaluationColdStage::AwaitingCompletion => None,
            EvaluationColdStage::Cleanup => {
                return Err(invalid("evaluation cleanup has no new numerical boundary"))
            }
        };
        if let Some(index) = index {
            let region = current
                .regions
                .get(index)
                .ok_or_else(|| invalid("phase evaluation lost its original region proof"))?;
            let region = region.borrow(py);
            self.source
                .borrow(py)
                .owner()?
                .require_closed_cold_model_work_region(
                    region.region.as_ref().expect("original evaluation region"),
                )
                .map_err(xlog_err)?;
        }
        current.cold_stage = to;
        Self::clear_evaluation_cold_visibility(
            py,
            &current.owners.parent.borrow(py).session.borrow(py),
        );
        Ok(())
    }

    fn clear_evaluation_cold_visibility(py: Python<'_>, session: &PySemanticTransitionSession) {
        let original = session
            .active_cold_model_work
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(original) = original {
            original.borrow(py).active.store(false, Ordering::Release);
        }
    }

    pub(in crate::semantic_transition) fn require_evaluation_cold_callback(
        &self,
        py: Python<'_>,
        work: &PySemanticColdModelWork,
    ) -> PyResult<()> {
        let retained = self.phase_evaluations()?;
        let current = retained
            .last()
            .ok_or_else(|| invalid("evaluation cold callback lost its original operation"))?;
        Self::require_evaluation_entry(py, current)?;
        let index = match current.cold_stage {
            EvaluationColdStage::Preparation => 0,
            EvaluationColdStage::OutputProjection => 1,
            EvaluationColdStage::Cleanup => 2,
            EvaluationColdStage::AwaitingCompletion => {
                return Err(invalid(
                    "pending evaluation cannot open cleanup or repeat output projection",
                ))
            }
        };
        if !self.phase_evaluation_active.load(Ordering::Acquire)
            || current.callback_result.is_some()
            || current
                .regions
                .get(index)
                .is_none_or(|original| !std::ptr::eq(&*original.borrow(py), work))
            || current.owners.parent.as_ptr() != work.parent.as_ptr()
            || self.parent.as_ptr() != work.reader.as_ptr()
        {
            return Err(invalid(
                "evaluation cold callback changed its actual region, TaskUse or parent",
            ));
        }
        Ok(())
    }

    pub(in crate::semantic_transition) fn evaluation_cold_work(
        &self,
        py: Python<'_>,
        task: &PySemanticTransitionTaskUse,
    ) -> PyResult<Option<Py<PySemanticColdModelWork>>> {
        if !self.phase_evaluation_active.load(Ordering::Acquire) {
            return Ok(None);
        }
        let retained = self.phase_evaluations()?;
        let current = retained
            .last()
            .ok_or_else(|| invalid("evaluation cold callback lost its original operation"))?;
        if !std::ptr::eq(&*current.owners.task.borrow(py), task) {
            return Err(invalid(
                "evaluation cold callback changed its original issuing TaskUse",
            ));
        }
        let index = match current.cold_stage {
            EvaluationColdStage::Preparation => 0,
            EvaluationColdStage::OutputProjection => 1,
            EvaluationColdStage::Cleanup => 2,
            EvaluationColdStage::AwaitingCompletion => return Err(invalid(
                "pending evaluation retains its original closed regions; cleanup is not admitted",
            )),
        };
        let original = current
            .regions
            .get(index)
            .ok_or_else(|| invalid("evaluation cold callback lost its actual region owner"))?
            .clone_ref(py);
        drop(retained);
        self.require_evaluation_cold_callback(py, &original.borrow(py))?;
        let session = task.session.borrow(py);
        let mut visible = session
            .active_cold_model_work
            .lock()
            .map_err(|_| invalid("evaluation cold visibility mutex is poisoned"))?;
        if visible
            .as_ref()
            .is_some_and(|previous| previous.as_ptr() != original.as_ptr())
        {
            return Err(invalid(
                "evaluation cold region cannot replace another active callback",
            ));
        }
        original.borrow(py).active.store(true, Ordering::Release);
        *visible = Some(original.clone_ref(py));
        Ok(Some(original))
    }
    pub(super) fn source_evaluation_count(&self) -> PyResult<usize> {
        Ok(self
            .phase_evaluations()?
            .iter()
            .filter(|evaluation| evaluation.branch == "source")
            .count())
    }

    pub(in crate::semantic_transition) fn private_evaluation_feedback_projection(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        parent: &Py<PySemanticPublishedParent>,
    ) -> PyResult<Option<Py<PyTuple>>> {
        if !self.phase_evaluation_active.load(Ordering::Acquire) {
            return Ok(None);
        }
        let retained = self.phase_evaluations()?;
        let current = retained
            .last()
            .ok_or_else(|| invalid("phase feedback lost its original evaluation"))?;
        if current.branch == "source" {
            return Ok(None);
        }
        Self::require_evaluation_entry(py, current)?;
        if !current.ready
            || !current.evaluation_admitted
            || current.callback_result.is_some()
            || parent.as_ptr() != current.owners.parent.as_ptr()
            || !std::ptr::eq(
                session,
                &*current.owners.parent.borrow(py).session.borrow(py),
            )
        {
            return Err(invalid(
                "phase feedback changed its actual private evaluation or acquired model owner",
            ));
        }
        let materials = feedback_materials(
            self.preparation_inputs
                .feedback_interventions
                .bind(py)
                .as_any(),
        )?;
        let real = current.branch == "real";
        Ok(Some(
            (materials[usize::from(!real)].clone(), real)
                .into_pyobject(py)?
                .unbind(),
        ))
    }

    /// Project each original private observation only after the selected group
    /// and its physical interval are fully consumed. No new model is restored.
    pub(super) fn execute_private_evaluations(
        &self,
        py: Python<'_>,
        branch: &'static str,
    ) -> PyResult<()> {
        let completion = self.private_evaluation_completion(branch)?;
        if completion.load(Ordering::Acquire) {
            return Ok(());
        }
        loop {
            let unfinished = self
                .phase_evaluations()?
                .last()
                .is_some_and(|evaluation| evaluation.branch == branch && !evaluation.recorded);
            if unfinished {
                self.execute_evaluation(py)?;
                continue;
            }
            if self
                .phase_evaluations()?
                .last()
                .is_some_and(|evaluation| evaluation.branch == branch)
            {
                self.release_evaluation_record(py)?;
            }
            let (owners, mut ordinal) = self.private_group_successor_input(py, branch)?;
            if let Some(previous) = self
                .phase_evaluations()?
                .last()
                .filter(|evaluation| evaluation.branch == branch)
            {
                ordinal = previous
                    .ordinal
                    .checked_add(1)
                    .ok_or_else(|| invalid("private evaluation position overflowed"))?;
            }
            self.preparation_inputs
                .require_program(py, &self.scientific_owner)?;
            let next = self.scientific_owner.bind(py).getattr("next_operation")?;
            if !next.is_exact_instance_of::<PyTuple>() || next.cast::<PyTuple>()?.is_empty() {
                return Err(invalid(
                    "private evaluation lost its complete original scheduled group",
                ));
            }
            let entries = next.cast::<PyTuple>()?;
            let first = entries.get_item(0)?;
            if !first.is_exact_instance_of::<PyDict>() {
                return Err(invalid("private evaluation lost its original exact entry"));
            }
            let first = first.cast::<PyDict>()?;
            let operation = first
                .get_item("operation")?
                .ok_or_else(|| invalid("private evaluation lost its operation"))?;
            let scheduled_branch = first
                .get_item("branch")?
                .ok_or_else(|| invalid("private evaluation lost its branch"))?;
            if ColdValue::read(&scheduled_branch, &mut 128, 0)?.text()? != branch {
                return Err(invalid(
                    "private evaluation changed its original scheduled branch",
                ));
            }
            let operation = ColdValue::read(&operation, &mut 128, 0)?;
            let operation = operation.text()?;
            if operation != "evaluation" && operation != "checkpoint" {
                return Err(invalid(&format!(
                    "private trajectory requires its next original {operation} group before final evaluations"
                )));
            }
            let (material, instruction) = Self::singleton_lifecycle_material(entries)?;
            let fields = ColdValue::from_canonical_bytes(&material)?;
            let fields = fields.fields(6)?;
            if fields[2].unsigned()? != ordinal || fields[3].unsigned()? != 0 {
                return Err(invalid(
                    "private evaluation changed its contiguous original position",
                ));
            }
            if operation == "checkpoint" {
                if fields[5] != ColdValue::None {
                    return Err(invalid(
                        "private checkpoint acquired an evaluation comparison",
                    ));
                }
                completion.store(true, Ordering::Release);
                return Ok(());
            }
            if fields[5] == ColdValue::None {
                return Err(invalid("private evaluation lost its original comparison"));
            }
            let budget = fields[4].fields(3)?;
            let budget = [
                budget[0].unsigned()?,
                budget[1].unsigned()?,
                budget[2].unsigned()?,
            ];
            self.phase_evaluations()?.push(PhaseEvaluation {
                owners,
                branch,
                entries: entries.clone().unbind(),
                material,
                instruction,
                ordinal,
                step: 0,
                budget,
                started: false,
                ready: false,
                evaluation_admitted: false,
                callback_entered: false,
                callback_pending: false,
                callback_result: None,
                callback_error: None,
                work: None,
                regions: Vec::new(),
                cold_stage: EvaluationColdStage::Preparation,
                native_evaluation: None,
                cancelled: None,
                cancelled_expense: None,
                custody: None,
                work_closed: false,
                work_result: None,
                observer_finish_entered: false,
                record_entered: false,
                recorded: false,
                physical_peak: None,
                released: false,
                budget_exceeded: false,
            });
        }
    }

    pub(in crate::semantic_transition) fn is_original_source(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
    ) -> bool {
        std::ptr::eq(session, &*self.source.borrow(py))
    }
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
        if !cold_callback && !self.phase_evaluation_active.load(Ordering::Acquire) {
            return Err(invalid("held source feedback is readable only inside its original serialization or evaluation callback"));
        }
        if self.phase_evaluation_active.load(Ordering::Acquire) {
            let retained = self.phase_evaluations()?;
            let current = retained
                .last()
                .ok_or_else(|| invalid("source feedback lost its original evaluation entry"))?;
            Self::require_evaluation_entry(py, current)?;
            if current.branch != "source"
                || !current.ready
                || !current.evaluation_admitted
                || current.callback_result.is_some()
            {
                return Err(invalid(
                    "source feedback lost its actual admitted source evaluation",
                ));
            }
        }
        Ok(())
    }

    fn phase_evaluations(&self) -> PyResult<MutexGuard<'_, Vec<PhaseEvaluation>>> {
        self.phase_evaluations
            .lock()
            .map_err(|_| invalid("original source evaluation custody mutex is poisoned"))
    }

    /// This is not general public admission. Only the one retained original
    /// retained callback can issue its read-only evaluation, once, on that parent.
    pub(in crate::semantic_transition) fn require_phase_evaluation(
        &self,
        py: Python<'_>,
        controller: &PySemanticTransitionController,
        task: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
    ) -> PyResult<()> {
        if !self.phase_evaluation_active.load(Ordering::Acquire) {
            return Err(invalid(
                "phase evaluation requires its original active callback",
            ));
        }
        self.records()?.require_preparation_admission()?;
        self.preparation_inputs
            .require_program(py, &self.scientific_owner)?;
        let mut retained = self.phase_evaluations()?;
        let current = retained
            .last_mut()
            .ok_or_else(|| invalid("source evaluation lost its original scheduled invocation"))?;
        if !std::ptr::eq(controller, &*current.owners.controller.borrow(py))
            || !std::ptr::eq(task, &*current.owners.task.borrow(py))
            || !std::ptr::eq(parent, &*current.owners.parent.borrow(py))
            || (current.branch == "source"
                && !matches!(task.state()?.phase, TaskUsePhase::ArenaPreparing(_)))
        {
            return Err(invalid(
                "phase evaluation changed its original controller, task or acquired parent",
            ));
        }
        Self::require_evaluation_entry(py, current)?;
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

    fn require_evaluation_entry(py: Python<'_>, current: &PhaseEvaluation) -> PyResult<()> {
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
                .phase_evaluations()?
                .last()
                .is_some_and(|evaluation| !evaluation.recorded);
            if !unfinished {
                if !self.phase_evaluations()?.is_empty() {
                    self.release_evaluation_record(py)?;
                }
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
                let expected = u64::try_from(self.phase_evaluations()?.len())
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
                self.phase_evaluations()?.push(PhaseEvaluation {
                    owners: EvaluationOwners {
                        controller: self.source_controller.clone_ref(py),
                        task: self.task_use.clone_ref(py),
                        parent: self.parent.clone_ref(py),
                        model: self.model_owner(py, PhaseModelOwner::Source)?,
                    },
                    branch: "source",
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
                    regions: Vec::new(),
                    cold_stage: EvaluationColdStage::Preparation,
                    native_evaluation: None,
                    cancelled: None,
                    cancelled_expense: None,
                    custody: None,
                    work_closed: false,
                    work_result: None,
                    observer_finish_entered: false,
                    record_entered: false,
                    recorded: false,
                    physical_peak: None,
                    released: false,
                    budget_exceeded: false,
                });
            }
            self.execute_evaluation(py)?;
        }
    }

    fn execute_evaluation(&self, py: Python<'_>) -> PyResult<()> {
        let terminal = self
            .phase_evaluations()?
            .last()
            .expect("retained evaluation")
            .callback_error
            .as_ref()
            .map(|error| error.clone_ref(py));
        if let Some(error) = terminal {
            return self.finish_cancelled_evaluation(py, error);
        }
        let (controller, task_owner, parent_owner, model, branch) = {
            let retained = self.phase_evaluations()?;
            let current = retained.last().expect("retained original evaluation");
            (
                current.owners.controller.clone_ref(py),
                current.owners.task.clone_ref(py),
                current.owners.parent.clone_ref(py),
                current.owners.model.clone_ref(py),
                current.branch,
            )
        };
        let source = self.source.borrow(py);
        let task = task_owner.borrow(py);
        let parent = parent_owner.borrow(py);
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
            let retained = self.phase_evaluations()?;
            let current = retained.last().expect("retained original evaluation");
            Self::require_evaluation_entry(py, current)?;
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
        if !started {
            // Retain entry before begin: a failed begin or allocation is not
            // proof of nonentry, and may not start a replacement interval.
            self.phase_evaluations()?
                .last_mut()
                .expect("retained evaluation")
                .started = true;
            inputs.resource_observer.begin(py, ordinal)?;
            let admission = self.records()?.confirmed_admission()?;
            let work = source
                .owner()?
                .prepare_cold_model_work(
                    &*self.parent.borrow(py).lease()?,
                    inputs.cold_model_work_capacity,
                    ordinal,
                    admission,
                )
                .map_err(xlog_err)?;
            self.phase_evaluations()?
                .last_mut()
                .expect("retained evaluation")
                .work = Some(work.clone());
            let regions = source
                .owner()?
                .prepare_cold_model_work_regions(&work, 3)
                .map_err(xlog_err)?;
            for region in regions {
                let original = Py::new(
                    py,
                    PySemanticColdModelWork {
                        parent: parent_owner.clone_ref(py),
                        reader: self.parent.clone_ref(py),
                        inner: work.clone(),
                        region: Some(region),
                        active: AtomicBool::new(false),
                    },
                )?;
                self.phase_evaluations()?
                    .last_mut()
                    .expect("retained evaluation")
                    .regions
                    .push(original);
            }
            source
                .owner()?
                .begin_cold_model_work(&work)
                .map_err(xlog_err)?;
            if branch == "source" {
                verify_phase_checkpoint(py, &task, &parent, &manifest, &saved_snapshot)?;
            } else {
                let custody = source
                    .owner()?
                    .share_cold_native_work(&work)
                    .map_err(xlog_err)?;
                self.phase_evaluations()?
                    .last_mut()
                    .expect("retained evaluation")
                    .custody = Some(custody.clone());
                parent
                    .session
                    .borrow(py)
                    .owner()?
                    .attach_shared_cold_native_work(&*parent.lease()?, custody)
                    .map_err(xlog_err)?;
            }
            task.state()?.snapshot = authority;
            self.phase_evaluations()?
                .last_mut()
                .expect("retained evaluation")
                .ready = true;
        }
        if !self
            .phase_evaluations()?
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
                controller.clone_ref(py),
                task_owner.clone_ref(py),
                parent_owner.clone_ref(py),
                model.clone_ref(py),
                entries,
            );
            self.phase_evaluations()?
                .last_mut()
                .expect("retained evaluation")
                .callback_entered = true;
            let session = parent.session.borrow(py);
            let _source_active = if branch == "source" {
                Some(PhaseOperation::begin(&self.phase_evaluation_active)?)
            } else {
                None
            };
            let _private_scope = if branch == "source" {
                None
            } else {
                Some(super::private_execution::PrivateTaskCallbackScope::enter(
                    &self.phase_evaluation_active,
                    &task,
                    &session,
                )?)
            };
            let _cold_visibility = EvaluationColdVisibility {
                session: &session,
                py,
            };
            let callback = self.model_owner(py, PhaseModelOwner::Execute)?;
            let returned = callback.bind(py).call1(arguments);
            let mut retained = self.phase_evaluations()?;
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
                    let cancelled = !current.callback_pending && current.cancelled.is_some();
                    drop(retained);
                    if cancelled {
                        return self.finish_cancelled_evaluation(py, error);
                    }
                    return Err(error);
                }
            }
        };
        let result = result.bind(py).cast::<PyTuple>()?;
        if !result.is_exact_instance_of::<PyTuple>()
            || result.len() != 3
            || result.get_item(0)?.as_ptr() != parent_owner.as_ptr()
            || result.get_item(1)?.as_ptr() != model.as_ptr()
        {
            return Err(invalid(
                "phase evaluation changed its actual original parent/model handoff",
            ));
        }
        let rows = result.get_item(2)?;
        if !rows.is_exact_instance_of::<PyTuple>() || rows.cast::<PyTuple>()?.len() != 1 {
            return Err(invalid(
                "phase evaluation lost its original singleton observation",
            ));
        }
        let row = rows.cast::<PyTuple>()?.get_item(0)?;
        let row = row.cast::<PyTuple>()?;
        if !row.is_exact_instance_of::<PyTuple>() || row.len() != 3 {
            return Err(invalid(
                "phase evaluation changed its actual observation/receipt/cohort triple",
            ));
        }
        let observation = row.get_item(0)?;
        if !observation.is_exact_instance_of::<PyDict>() {
            return Err(invalid(
                "phase evaluation requires its actual numerical observation",
            ));
        }
        let observation = observation.cast::<PyDict>()?;
        let receipt = row.get_item(1)?;
        let receipt = receipt.extract::<PyRef<'_, PySemanticCompletedModelEvaluation>>()?;
        let cohort = row.get_item(2)?;
        let cohort = cohort.extract::<PyRef<'_, PySemanticEvaluationCohort>>()?;
        let measured = receipt.require_phase_observation(py, &parent, &cohort)?;
        if branch == "source" {
            verify_phase_checkpoint(py, &task, &parent, &manifest, &saved_snapshot)?;
        }
        let (native, peak) = self.finish_evaluation_expense(py)?;
        let work = measured
            .model_work
            .checked_add(native.native_work)
            .and_then(|work| work.checked_add(native.model_work))
            .ok_or_else(|| invalid("phase evaluation expense overflowed"))?;
        let calls = measured
            .model_calls
            .checked_add(native.model_calls)
            .ok_or_else(|| invalid("phase evaluation model calls overflowed"))?;
        let (instruction, step, budget, entered) = {
            let retained = self.phase_evaluations()?;
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
        arguments.set_item("branch", branch)?;
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
        self.phase_evaluations()?
            .last_mut()
            .expect("retained evaluation")
            .record_entered = true;
        self.phase_evaluations()?
            .last_mut()
            .expect("retained evaluation")
            .budget_exceeded = work > budget[0] || peak > budget[1] || calls > budget[2];
        callback.call((), Some(&arguments))?;
        self.phase_evaluations()?
            .last_mut()
            .expect("retained evaluation")
            .recorded = true;
        self.release_evaluation_record(py)
    }

    fn release_evaluation_record(&self, py: Python<'_>) -> PyResult<()> {
        let (recorded, released, exceeded) = {
            let retained = self.phase_evaluations()?;
            let current = retained.last().expect("retained evaluation");
            (current.recorded, current.released, current.budget_exceeded)
        };
        if !recorded {
            return Err(invalid(
                "evaluation release precedes its original known history record",
            ));
        }
        if !released {
            self.preparation_inputs.resource_observer.release(py)?;
            self.phase_evaluations()?
                .last_mut()
                .expect("retained evaluation")
                .released = true;
        }
        if exceeded {
            return Err(invalid("phase evaluation exceeded its original operation budget; its actual observation is retained"));
        }
        Ok(())
    }

    /// Both known observation and known cancellation consume this same original
    /// cold report and physical interval. Neither can replay the model callback.
    fn finish_evaluation_expense(
        &self,
        py: Python<'_>,
    ) -> PyResult<(SemanticColdModelWorkResult, u64)> {
        let (work, closed, work_result, parent, custody, cached_peak) = {
            let retained = self.phase_evaluations()?;
            let current = retained.last().expect("retained evaluation");
            if current.cold_stage != EvaluationColdStage::Cleanup
                || (current.callback_result.is_none() && current.cancelled.is_none())
            {
                return Err(invalid("evaluation expense lacks its original known numerical completion or cancellation"));
            }
            (
                current
                    .work
                    .clone()
                    .ok_or_else(|| invalid("evaluation lost its native expense owner"))?,
                current.work_closed,
                current.work_result,
                current.owners.parent.clone_ref(py),
                current.custody.clone(),
                current.physical_peak,
            )
        };
        let source = self.source.borrow(py);
        if !closed {
            source
                .owner()?
                .close_cold_model_work(&work)
                .map_err(xlog_err)?;
            self.phase_evaluations()?
                .last_mut()
                .expect("retained evaluation")
                .work_closed = true;
        }
        let native = match work_result {
            Some(result) => result,
            None => {
                if let Some(custody) = custody {
                    parent
                        .borrow(py)
                        .session
                        .borrow(py)
                        .owner()?
                        .detach_shared_cold_native_work(&custody)
                        .map_err(xlog_err)?;
                    self.phase_evaluations()?
                        .last_mut()
                        .expect("retained evaluation")
                        .custody = None;
                    drop(custody);
                }
                let streams = self.preparation_inputs.consumer_streams.python_value(py)?;
                let streams =
                    checkpoint_consumer_streams(streams.bind(py), &mut (16 * 1024 * 1024))?;
                // Native Closed/Submitted custody, not an attempted finish flag,
                // chooses the original submit or read-only resolution.
                let result = source
                    .owner()?
                    .finish_cold_model_work(&*self.parent.borrow(py).lease()?, &work, &streams)
                    .map_err(xlog_err)?;
                self.phase_evaluations()?
                    .last_mut()
                    .expect("retained evaluation")
                    .work_result = Some(result);
                result
            }
        };
        let finish_needed = {
            let mut retained = self.phase_evaluations()?;
            let current = retained.last_mut().expect("retained evaluation");
            let needed = !current.observer_finish_entered;
            current.observer_finish_entered = true;
            needed
        };
        if finish_needed {
            self.preparation_inputs.resource_observer.finish(py)?;
        }
        let peak = match cached_peak {
            Some(peak) => peak,
            None => {
                let peak = self
                    .preparation_inputs
                    .resource_observer
                    .physical_peak(py)?;
                self.phase_evaluations()?
                    .last_mut()
                    .expect("retained evaluation")
                    .physical_peak = Some(peak);
                peak
            }
        };
        Ok((native, peak))
    }

    fn finish_cancelled_evaluation(&self, py: Python<'_>, original: PyErr) -> PyResult<()> {
        let (cancelled, cached, released) = {
            let retained = self.phase_evaluations()?;
            let current = retained.last().expect("retained evaluation");
            (
                current.cancelled.clone(),
                current.cancelled_expense,
                current.released,
            )
        };
        let Some(cancelled) = cancelled else {
            return Err(original);
        };
        let result = (|| {
            if cached.is_none() {
                let (cold, peak) = self.finish_evaluation_expense(py)?;
                let work = cancelled
                    .model_work()
                    .checked_add(cold.model_work)
                    .and_then(|work| work.checked_add(cold.native_work))
                    .ok_or_else(|| invalid("cancelled evaluation expenditure overflowed"))?;
                let calls = cancelled
                    .model_calls()
                    .checked_add(cold.model_calls)
                    .ok_or_else(|| invalid("cancelled evaluation calls overflowed"))?;
                let mut retained = self.phase_evaluations()?;
                let current = retained.last_mut().expect("retained evaluation");
                current.cancelled_expense = Some((work, peak, calls));
                current.budget_exceeded = work > current.budget[0]
                    || peak > current.budget[1]
                    || calls > current.budget[2];
            }
            if !released {
                self.preparation_inputs.resource_observer.release(py)?;
                self.phase_evaluations()?
                    .last_mut()
                    .expect("retained evaluation")
                    .released = true;
            }
            Ok::<(), PyErr>(())
        })();
        if let Err(incomplete) = result {
            incomplete.set_cause(py, Some(original));
            return Err(incomplete);
        }
        // Cancellation is terminal, never a record_evaluation observation or
        // permission to advance the original scientific schedule.
        Err(original)
    }
}
