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
    owners: Option<EvaluationOwners>,
    branch: &'static str,
    entries: Option<Py<PyTuple>>,
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
    refusal: Option<CancelledEvaluationRefusal>,
}

struct CancelledEvaluationRefusal {
    cause: &'static str,
    prefix: Arc<[u8]>,
    reason: ColdValue,
    private_save_entered: bool,
    private_save_admitted: bool,
    private_save_verified: bool,
    private_checkpoint: Option<Arc<[u8]>>,
    retirement_entered: bool,
    retirement_returned: bool,
    private_owners_dropped: bool,
    session_release_entered: bool,
    private_released: bool,
    cleanup_entered: bool,
    cleanup_returned: bool,
    cleanup_completed: bool,
    serializer_entered: bool,
    serialized_model: Option<Py<PyAny>>,
    source_verified: bool,
    history: Option<Arc<[u8]>>,
    payload: Option<Arc<[u8]>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::semantic_transition) enum EvaluationColdStage {
    Preparation,
    OutputProjection,
    AwaitingCompletion,
    Cleanup,
    CancelledExecutionCleanup,
    CancelledPrivateCheckpoint,
    CancelledPrivateRetirement,
    SourceVerification,
}

impl PhaseEvaluation {
    fn owners(&self) -> PyResult<&EvaluationOwners> {
        self.owners
            .as_ref()
            .ok_or_else(|| invalid("cancelled evaluation private owners are already retired"))
    }

    fn entries(&self) -> PyResult<&Py<PyTuple>> {
        self.entries
            .as_ref()
            .ok_or_else(|| invalid("cancelled evaluation numerical entry has been released"))
    }

    fn cold_region_index(&self) -> PyResult<usize> {
        match self.cold_stage {
            EvaluationColdStage::Preparation => Ok(0),
            EvaluationColdStage::OutputProjection => Ok(1),
            EvaluationColdStage::Cleanup => Ok(2),
            EvaluationColdStage::CancelledPrivateCheckpoint => Ok(3),
            EvaluationColdStage::CancelledExecutionCleanup => {
                Ok(if self.branch == "source" { 3 } else { 4 })
            }
            EvaluationColdStage::CancelledPrivateRetirement => Ok(5),
            EvaluationColdStage::SourceVerification => {
                Ok(if self.branch == "source" { 4 } else { 0 })
            }
            EvaluationColdStage::AwaitingCompletion => Err(invalid(
                "pending evaluation has no active cold callback region",
            )),
        }
    }
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
    pub(super) fn cancelled_evaluation_refusal_retained(&self) -> PyResult<bool> {
        Ok(self
            .phase_evaluations()?
            .last()
            .is_some_and(|current| current.refusal.is_some()))
    }

    fn capture_cancelled_evaluation_refusal(&self, py: Python<'_>) -> PyResult<()> {
        if self.cancelled_evaluation_refusal_retained()? {
            return Ok(());
        }
        let branch = self
            .phase_evaluations()?
            .last()
            .ok_or_else(|| invalid("cancellation lost its original evaluation"))?
            .branch;
        if !matches!(branch, "source" | "control" | "real")
            || branch == "source"
                && (self.candidate_entered.load(Ordering::Acquire)
                    || self.private_restore()?.is_some())
            || branch != "source" && self.private_restore()?.is_none()
            || self.acceptance()?.is_some()
            || self
                .candidate
                .lock()
                .map_err(|_| invalid("phase candidate custody is poisoned"))?
                .is_some()
            || self.records()?.preparation_outcome_known
        {
            return Err(invalid("evaluation cancellation contradicts its original restoration or entered scientific acceptance"));
        }
        self.preparation_inputs
            .require_program(py, &self.scientific_owner)?;
        let history = self.scientific_owner.bind(py).getattr("history_bytes")?;
        if !history.is_exact_instance_of::<PyBytes>() {
            return Err(invalid(
                "Source cancellation lost its original actual prefix history",
            ));
        }
        let prefix: Arc<[u8]> = Arc::from(history.cast::<PyBytes>()?.as_bytes());
        let mut retained = self.phase_evaluations()?;
        let current = retained
            .last_mut()
            .ok_or_else(|| invalid("Source cancellation lost its original evaluation"))?;
        Self::require_evaluation_entry(py, current)?;
        if current.callback_pending
            || current.callback_result.is_some()
            || current.cold_stage != EvaluationColdStage::Cleanup
            || current.work_closed
            || current.observer_finish_entered
            || current.record_entered
            || current.released
            || branch == "source"
                && (current.owners()?.parent.as_ptr() != self.parent.as_ptr()
                    || current.owners()?.task.as_ptr() != self.task_use.as_ptr())
        {
            return Err(invalid(
                "evaluation cancellation lacks known original cleanup before its measured end",
            ));
        }
        let cancelled = current
            .cancelled
            .as_ref()
            .ok_or_else(|| invalid("Source cancellation lacks its original native proof"))?;
        let region = current
            .regions
            .get(2)
            .ok_or_else(|| invalid("Source cancellation lost its actual cleanup region"))?
            .borrow(py);
        self.source
            .borrow(py)
            .owner()?
            .require_closed_cold_model_work_region(
                region.region.as_ref().expect("original cleanup region"),
            )
            .map_err(xlog_err)?;
        let cause = if branch == "source" {
            cold_restore::SOURCE_EVALUATION_CANCELLED
        } else {
            cold_restore::PRIVATE_EVALUATION_CANCELLED
        };
        current.refusal = Some(CancelledEvaluationRefusal {
            cause,
            prefix,
            reason: cold_restore::cancelled_evaluation_reason(cancelled, cause),
            private_save_entered: false,
            private_save_admitted: false,
            private_save_verified: branch == "source",
            private_checkpoint: None,
            retirement_entered: false,
            retirement_returned: false,
            private_owners_dropped: false,
            session_release_entered: false,
            private_released: branch == "source",
            cleanup_entered: false,
            cleanup_returned: false,
            cleanup_completed: false,
            serializer_entered: false,
            serialized_model: None,
            source_verified: false,
            history: None,
            payload: None,
        });
        Ok(())
    }

    pub(super) fn finish_cancelled_phase_evaluation(&self, py: Python<'_>) -> PyResult<()> {
        if matches!(*self.status_lock()?, Completion::Refused) {
            return Ok(());
        }
        let branch = self
            .phase_evaluations()?
            .last()
            .expect("retained cancellation")
            .branch;
        if branch != "source" {
            self.save_cancelled_private_checkpoint(py)?;
        }
        self.release_cancelled_evaluation_execution(py)?;
        // Only CPU proof is retained. Drop original native cohorts and failed
        // Python numerical frames outside the mutex while the tally is recording.
        let (prior, native, error) = {
            let mut retained = self.phase_evaluations()?;
            let count = retained.len();
            if retained
                .iter()
                .take(count.saturating_sub(1))
                .any(|previous| {
                    !previous.recorded || !previous.released || previous.budget_exceeded
                })
            {
                return Err(invalid(
                    "Source cancellation cannot discard an unfinished original evaluation prefix",
                ));
            }
            let current = retained
                .last_mut()
                .ok_or_else(|| invalid("terminal Source lost its evaluation"))?;
            if current.refusal.is_none() {
                return Err(invalid(
                    "terminal Source lacks its original cancellation frame",
                ));
            }
            let native = current.native_evaluation.take();
            let error = current.callback_error.take();
            let prior: Vec<_> = retained.drain(..count - 1).collect();
            (prior, native, error)
        };
        drop(prior);
        drop(native);
        drop(error);
        if branch != "source" {
            self.retire_cancelled_private_evaluation(py)?;
        }
        let verified = self
            .phase_evaluations()?
            .last()
            .expect("retained cancellation")
            .refusal
            .as_ref()
            .expect("original refusal")
            .source_verified;
        if !verified {
            self.verify_cancelled_source(py)?;
        }
        let (cancelled, expense) = {
            let retained = self.phase_evaluations()?;
            let current = retained.last().expect("retained cancellation");
            (
                current
                    .cancelled
                    .clone()
                    .expect("original native cancellation"),
                current.cancelled_expense,
            )
        };
        let usage = if let Some(usage) = expense {
            [usage.0, usage.1, usage.2]
        } else {
            let (cold, peak) = self.finish_evaluation_expense(py)?;
            let work = cancelled
                .model_work()
                .checked_add(cold.model_work)
                .and_then(|work| work.checked_add(cold.native_work))
                .ok_or_else(|| invalid("cancelled evaluation work overflowed"))?;
            let calls = cancelled
                .model_calls()
                .checked_add(cold.model_calls)
                .ok_or_else(|| invalid("cancelled evaluation calls overflowed"))?;
            self.phase_evaluations()?
                .last_mut()
                .expect("retained cancellation")
                .cancelled_expense = Some((work, peak, calls));
            [work, peak, calls]
        };
        self.record_cancelled_evaluation(py, usage)?;
        self.resume_terminal_source(py)
    }

    pub(super) fn require_cancelled_evaluation_checkpoint(
        &self,
        py: Python<'_>,
        controller: &PySemanticTransitionController,
        task: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
    ) -> PyResult<()> {
        let mut retained = self.phase_evaluations()?;
        let current = retained
            .last_mut()
            .ok_or_else(|| invalid("cancelled snapshot lost its original evaluation"))?;
        Self::require_evaluation_entry(py, current)?;
        let same = std::ptr::eq(controller, &*current.owners()?.controller.borrow(py))
            && std::ptr::eq(task, &*current.owners()?.task.borrow(py))
            && std::ptr::eq(parent, &*current.owners()?.parent.borrow(py));
        let refusal = current
            .refusal
            .as_mut()
            .ok_or_else(|| invalid("cancelled snapshot lacks original refusal custody"))?;
        if !self.phase_evaluation_active.load(Ordering::Acquire)
            || current.branch == "source"
            || current.cold_stage != EvaluationColdStage::CancelledPrivateCheckpoint
            || current.cancelled.is_none()
            || !same
            || !refusal.private_save_entered
            || refusal.private_save_admitted
            || refusal.private_checkpoint.is_some()
            || !matches!(task.state()?.phase, TaskUsePhase::ArenaPreparing(_))
        {
            return Err(invalid(
                "cancelled snapshot changed or repeated its original full private save",
            ));
        }
        refusal.private_save_admitted = true;
        Ok(())
    }

    fn save_cancelled_private_checkpoint(&self, py: Python<'_>) -> PyResult<()> {
        let saved = self
            .phase_evaluations()?
            .last()
            .expect("retained cancellation")
            .refusal
            .as_ref()
            .expect("original cancellation")
            .private_checkpoint
            .is_some();
        if saved {
            return self.verify_cancelled_private_checkpoint(py);
        }
        let (owners, region) = {
            let mut retained = self.phase_evaluations()?;
            let current = retained.last_mut().expect("retained cancellation");
            let refusal = current.refusal.as_ref().expect("original cancellation");
            if refusal.private_save_entered {
                return Err(invalid(
                    "unknown cancelled full private save cannot repeat its original serializer",
                ));
            }
            Self::require_evaluation_entry(py, current)?;
            let original = current.owners()?;
            let owners = EvaluationOwners {
                controller: original.controller.clone_ref(py),
                task: original.task.clone_ref(py),
                parent: original.parent.clone_ref(py),
                model: original.model.clone_ref(py),
            };
            current.cold_stage = EvaluationColdStage::CancelledPrivateCheckpoint;
            (owners, current.regions[3].clone_ref(py))
        };
        let serializer = self.model_owner(py, PhaseModelOwner::SerializeCandidate)?;
        let selected = owners.model.clone_ref(py);
        let session = owners.parent.borrow(py).session.clone_ref(py);
        let original_region = region.clone_ref(py);
        let snapshot_model =
            pyo3::types::PyCFunction::new_closure(py, None, None, move |args, kwargs| {
                let py = args.py();
                if !args.is_empty() || kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
                    return Err(invalid(
                        "cancelled serializer requires its original zero-argument call",
                    ));
                }
                let session = session.borrow(py);
                let work = original_region.borrow(py);
                let _scope =
                    ColdCallbackScope::enter(py, &session, &work, original_region.clone_ref(py))?;
                serializer
                    .bind(py)
                    .call1((selected.clone_ref(py),))
                    .map(Bound::unbind)
            })?;
        let fresh = self.refresh_snapshot.bind(py).call0()?;
        let authority =
            AuthoritySnapshot::parse(&ColdValue::read(&fresh, &mut (16 * 1024 * 1024), 0)?)?;
        {
            let issued = owners.task.borrow(py);
            authority.newer_than(&issued.state()?.snapshot)?;
            check_learning_grant(&issued, &self.grant_reference, &authority)?;
        }
        let streams = self.preparation_inputs.consumer_streams.python_value(py)?;
        self.phase_evaluations()?
            .last_mut()
            .expect("retained cancellation")
            .refusal
            .as_mut()
            .expect("original cancellation")
            .private_save_entered = true;
        let saved = {
            let _active = PhaseOperation::begin(&self.phase_evaluation_active)?;
            owners.controller.borrow(py).save_checkpoint(
                py,
                &owners.task.borrow(py),
                &owners.parent.borrow(py),
                streams.bind(py),
                &fresh,
                snapshot_model.as_any(),
            )?
        };
        // Preserve the exact successful canonical return before verification.
        let checkpoint: Arc<[u8]> = Arc::from(saved.bind(py).as_bytes());
        self.phase_evaluations()?
            .last_mut()
            .expect("retained cancellation")
            .refusal
            .as_mut()
            .expect("original cancellation")
            .private_checkpoint = Some(Arc::clone(&checkpoint));
        drop(snapshot_model);
        self.verify_cancelled_private_checkpoint(py)
    }

    fn verify_cancelled_private_checkpoint(&self, py: Python<'_>) -> PyResult<()> {
        let (checkpoint, parent, task, region, proof) = {
            let retained = self.phase_evaluations()?;
            let current = retained.last().expect("retained cancellation");
            let refusal = current.refusal.as_ref().expect("original cancellation");
            if refusal.private_save_verified {
                return Ok(());
            }
            (
                Arc::clone(refusal.private_checkpoint.as_ref().ok_or_else(|| {
                    invalid("cancelled save lacks its original canonical return")
                })?),
                current.owners()?.parent.clone_ref(py),
                current.owners()?.task.clone_ref(py),
                current.regions[3].clone_ref(py),
                current
                    .cancelled
                    .as_ref()
                    .expect("original native cancellation")
                    .parent(),
            )
        };
        self.source
            .borrow(py)
            .owner()?
            .require_closed_cold_model_work_region(
                region
                    .borrow(py)
                    .region
                    .as_ref()
                    .expect("original full private save region"),
            )
            .map_err(xlog_err)?;
        let manifest = SemanticCheckpointManifest::decode(&checkpoint)?;
        let (_, authority, _, _) = TaskCheckpointSeed::decode(&manifest.task)?;
        verify_phase_checkpoint(
            py,
            &task.borrow(py),
            &parent.borrow(py),
            &manifest,
            &authority,
        )?;
        if SemanticTransitionSession::state_material_projection(&manifest.native)
            .map_err(xlog_err)?
            .publication
            != proof
        {
            return Err(invalid(
                "cancelled full private checkpoint changed its original native cancel parent",
            ));
        }
        self.phase_evaluations()?
            .last_mut()
            .expect("retained cancellation")
            .refusal
            .as_mut()
            .expect("original cancellation")
            .private_save_verified = true;
        Ok(())
    }

    fn retire_cancelled_private_evaluation(&self, py: Python<'_>) -> PyResult<()> {
        let (parent, task, session, region, returned, entered, dropped) = {
            let mut retained = self.phase_evaluations()?;
            let current = retained.last_mut().expect("retained cancellation");
            let refusal = current.refusal.as_ref().expect("original cancellation");
            if refusal.private_released {
                return Ok(());
            }
            if !refusal.cleanup_completed || !refusal.private_save_verified {
                return Err(invalid(
                    "private retirement precedes known full save and failed execution cleanup",
                ));
            }
            // The full retirement registrar remains held until known Session release.
            let region = current.regions[5].clone_ref(py);
            let parent = region.borrow(py).parent.clone_ref(py);
            let task = parent.borrow(py).task_use.clone_ref(py);
            let session = parent.borrow(py).session.clone_ref(py);
            let state = (
                refusal.retirement_returned,
                refusal.retirement_entered,
                refusal.private_owners_dropped,
            );
            current.cold_stage = EvaluationColdStage::CancelledPrivateRetirement;
            (parent, task, session, region, state.0, state.1, state.2)
        };
        if !returned {
            if entered {
                return Err(invalid(
                    "unknown private cancellation retirement cannot repeat its original callback",
                ));
            }
            let model = self
                .phase_evaluations()?
                .last()
                .expect("retained cancellation")
                .owners()?
                .model
                .clone_ref(py);
            let fresh = self.refresh_snapshot.bind(py).call0()?;
            let authority =
                AuthoritySnapshot::parse(&ColdValue::read(&fresh, &mut (16 * 1024 * 1024), 0)?)?;
            {
                let issued = task.borrow(py);
                authority.newer_than(&issued.state()?.snapshot)?;
                check_learning_grant(&issued, &self.grant_reference, &authority)?;
            }
            let callback = self.model_owner(py, PhaseModelOwner::RetirePrivate)?;
            self.phase_evaluations()?
                .last_mut()
                .expect("retained cancellation")
                .refusal
                .as_mut()
                .expect("original cancellation")
                .retirement_entered = true;
            {
                let _active = PhaseOperation::begin(&self.phase_evaluation_active)?;
                let session = session.borrow(py);
                let original_parent = parent.borrow(py);
                let issued = task.borrow(py);
                let _reads = ImportReadScope::checkpoint(&session, &issued, &original_parent, py)?;
                let work = region.borrow(py);
                let _scope = ColdCallbackScope::enter(py, &session, &work, region.clone_ref(py))?;
                if !callback.bind(py).call1((model,))?.is_none() {
                    return Err(invalid(
                        "cancelled private retirement requires its original known None completion",
                    ));
                }
            }
            self.phase_evaluations()?
                .last_mut()
                .expect("retained cancellation")
                .refusal
                .as_mut()
                .expect("original cancellation")
                .retirement_returned = true;
        }
        self.source
            .borrow(py)
            .owner()?
            .require_closed_cold_model_work_region(
                region
                    .borrow(py)
                    .region
                    .as_ref()
                    .expect("original private retirement region"),
            )
            .map_err(xlog_err)?;
        session
            .borrow(py)
            .owner()?
            .require_retired_publication(&*parent.borrow(py).lease()?)
            .map_err(xlog_err)?;
        if !dropped {
            let custody = self
                .phase_evaluations()?
                .last()
                .expect("retained cancellation")
                .custody
                .clone();
            if let Some(custody) = custody {
                session
                    .borrow(py)
                    .owner()?
                    .detach_shared_cold_native_work(&custody)
                    .map_err(xlog_err)?;
                self.phase_evaluations()?
                    .last_mut()
                    .expect("retained cancellation")
                    .custody = None;
            }
            let branch = self
                .phase_evaluations()?
                .last()
                .expect("retained cancellation")
                .branch;
            self.drop_completed_private_checkpoints(branch)?;
            self.drop_completed_private_execution_owners(branch)?;
            self.drop_completed_private_restore_owners(branch)?;
            let construction = self.private_restore()?.take();
            let (owners, entries) = {
                let mut retained = self.phase_evaluations()?;
                let current = retained.last_mut().expect("retained cancellation");
                (current.owners.take(), current.entries.take())
            };
            drop(construction);
            drop(owners);
            drop(entries);
            self.phase_evaluations()?
                .last_mut()
                .expect("retained cancellation")
                .refusal
                .as_mut()
                .expect("original cancellation")
                .private_owners_dropped = true;
        }
        let entered = self
            .phase_evaluations()?
            .last()
            .expect("retained cancellation")
            .refusal
            .as_ref()
            .expect("original cancellation")
            .session_release_entered;
        if entered {
            return Err(invalid(
                "unknown cancelled private Session release retains its original custody",
            ));
        }
        self.phase_evaluations()?
            .last_mut()
            .expect("retained cancellation")
            .refusal
            .as_mut()
            .expect("original cancellation")
            .session_release_entered = true;
        session
            .borrow(py)
            .release_retired_publication(py, &parent.borrow(py))?;
        task.borrow(py).state()?.phase = TaskUsePhase::Refused;
        session
            .borrow(py)
            .learning_preparing
            .store(false, Ordering::Release);
        let original = session
            .borrow(py)
            .learning_transition
            .lock()
            .map_err(|_| invalid("cancelled private Session lost its original phase custody"))?
            .take();
        drop(original);
        // All private callback registrars hold the selected parent. Only the
        // original Source-verification registrar remains for the same report.
        let regions: Vec<_> = self
            .phase_evaluations()?
            .last_mut()
            .expect("retained cancellation")
            .regions
            .drain(..6)
            .collect();
        drop(regions);
        drop(region);
        drop(parent);
        drop(task);
        drop(session);
        self.phase_evaluations()?
            .last_mut()
            .expect("retained cancellation")
            .refusal
            .as_mut()
            .expect("original cancellation")
            .private_released = true;
        Ok(())
    }

    fn release_cancelled_evaluation_execution(&self, py: Python<'_>) -> PyResult<()> {
        let (entered, returned, region) = {
            let mut retained = self.phase_evaluations()?;
            let current = retained
                .last_mut()
                .ok_or_else(|| invalid("cancelled execution lost its original evaluation"))?;
            let refusal = current
                .refusal
                .as_ref()
                .ok_or_else(|| invalid("cancelled execution lacks its original refusal custody"))?;
            if refusal.cleanup_completed {
                return Ok(());
            }
            Self::require_evaluation_entry(py, current)?;
            if current.cancelled.is_none()
                || !refusal.private_save_verified
                || current.callback_pending
                || current.callback_result.is_some()
                || current.work_closed
                || current.observer_finish_entered
                || current.record_entered
                || current.released
            {
                return Err(invalid("cancelled execution release requires known original native cancellation before its measured end"));
            }
            current.cold_stage = EvaluationColdStage::CancelledExecutionCleanup;
            (
                refusal.cleanup_entered,
                refusal.cleanup_returned,
                current.regions[if current.branch == "source" { 3 } else { 4 }].clone_ref(py),
            )
        };
        let parent = region.borrow(py).parent.clone_ref(py);
        let session_owner = parent.borrow(py).session.clone_ref(py);
        let session = session_owner.borrow(py);
        session.require_creator()?;
        if !returned {
            if entered {
                return Err(invalid(
                    "unknown cancelled execution release cannot repeat its original callback",
                ));
            }
            let _active = PhaseOperation::begin(&self.phase_evaluation_active)?;
            let _visibility = EvaluationColdVisibility {
                session: &session,
                py,
            };
            let callback = self.model_owner(py, PhaseModelOwner::ReleaseCancelledExecution)?;
            self.phase_evaluations()?
                .last_mut()
                .expect("retained cancellation")
                .refusal
                .as_mut()
                .expect("original cancellation")
                .cleanup_entered = true;
            let work = region.borrow(py);
            let _callback = ColdCallbackScope::enter(py, &session, &work, region.clone_ref(py))?;
            let result = callback.bind(py).call0()?;
            if !result.is_none() {
                return Err(invalid(
                    "cancelled execution release requires the original known None completion",
                ));
            }
            self.phase_evaluations()?
                .last_mut()
                .expect("retained cancellation")
                .refusal
                .as_mut()
                .expect("original cancellation")
                .cleanup_returned = true;
        }
        self.source
            .borrow(py)
            .owner()?
            .require_closed_cold_model_work_region(
                region
                    .borrow(py)
                    .region
                    .as_ref()
                    .expect("original cancelled execution cleanup region"),
            )
            .map_err(xlog_err)?;
        self.phase_evaluations()?
            .last_mut()
            .expect("retained cancellation")
            .refusal
            .as_mut()
            .expect("original cancellation")
            .cleanup_completed = true;
        Ok(())
    }

    fn verify_cancelled_source(&self, py: Python<'_>) -> PyResult<()> {
        let source = self.source.borrow(py);
        source.require_creator()?;
        let task = self.task_use.borrow(py);
        if !matches!(task.state()?.phase, TaskUsePhase::ArenaPreparing(_)) {
            return Err(invalid(
                "cancelled evaluation lost its held original Source",
            ));
        }
        let refreshed = self.refresh_snapshot.bind(py).call0()?;
        let snapshot =
            AuthoritySnapshot::parse(&ColdValue::read(&refreshed, &mut (16 * 1024 * 1024), 0)?)?;
        snapshot.newer_than(&task.state()?.snapshot)?;
        check_learning_grant(&task, &self.grant_reference, &snapshot)?;
        let manifest = SemanticCheckpointManifest::decode(&self.source_checkpoint)?;
        let (seed, _, _, _) = TaskCheckpointSeed::decode(&manifest.task)?;
        let live_expense = task.checkpoint.proposal_expense()?;
        let private = self
            .phase_evaluations()?
            .last()
            .expect("retained cancellation")
            .refusal
            .as_ref()
            .expect("original cancellation")
            .private_checkpoint
            .as_ref()
            .map(Arc::clone);
        let expected_expense = if let Some(private) = private {
            let private_manifest = SemanticCheckpointManifest::decode(&private)?;
            let (private_seed, _, _, _) = TaskCheckpointSeed::decode(&private_manifest.task)?;
            private_seed.proposal_expense()?
        } else {
            seed.proposal_expense()?
        };
        if expected_expense != live_expense
            || seed.proposal_expense()?.capacity != live_expense.capacity
            || seed.proposal_expense()?.spent > live_expense.spent
        {
            return Err(invalid(
                "Source cancellation changed its original cumulative Proposal expense",
            ));
        }
        let parent = self.parent.borrow(py);
        verify_phase_native(py, &task, &parent, &manifest.native, true)?;
        let (entered, result, region) = {
            let mut retained = self.phase_evaluations()?;
            let current = retained.last_mut().expect("retained cancellation");
            current.cold_stage = EvaluationColdStage::SourceVerification;
            let refusal = current.refusal.as_ref().expect("original cancellation");
            if !refusal.cleanup_completed || !refusal.private_released {
                return Err(invalid(
                    "Source verification precedes known original execution-frame release",
                ));
            }
            (
                refusal.serializer_entered,
                refusal
                    .serialized_model
                    .as_ref()
                    .map(|result| result.clone_ref(py)),
                current.regions[if current.branch == "source" { 4 } else { 0 }].clone_ref(py),
            )
        };
        let model = match result {
            Some(result) => result,
            None => {
                if entered {
                    return Err(invalid(
                        "unknown terminal Source serializer cannot repeat its original callback",
                    ));
                }
                let _active = PhaseOperation::begin(&self.phase_evaluation_active)?;
                let _visibility = EvaluationColdVisibility {
                    session: &source,
                    py,
                };
                let _reads = ImportReadScope::checkpoint(&source, &task, &parent, py)?;
                let callback = self.model_owner(py, PhaseModelOwner::SerializeSource)?;
                self.phase_evaluations()?
                    .last_mut()
                    .expect("retained cancellation")
                    .refusal
                    .as_mut()
                    .expect("original cancellation")
                    .serializer_entered = true;
                let work = region.borrow(py);
                let _callback = ColdCallbackScope::enter(py, &source, &work, region.clone_ref(py))?;
                let result = callback.bind(py).call0()?.unbind();
                self.phase_evaluations()?
                    .last_mut()
                    .expect("retained cancellation")
                    .refusal
                    .as_mut()
                    .expect("original cancellation")
                    .serialized_model = Some(result.clone_ref(py));
                result
            }
        };
        source
            .owner()?
            .require_closed_cold_model_work_region(
                region
                    .borrow(py)
                    .region
                    .as_ref()
                    .expect("original Source verification region"),
            )
            .map_err(xlog_err)?;
        require_model_bytes(model.bind(py), &manifest.model)?;
        verify_phase_native(py, &task, &parent, &manifest.native, true)?;
        task.state()?.snapshot = snapshot;
        self.phase_evaluations()?
            .last_mut()
            .expect("retained cancellation")
            .refusal
            .as_mut()
            .expect("original cancellation")
            .source_verified = true;
        Ok(())
    }

    fn record_cancelled_evaluation(&self, py: Python<'_>, usage: [u64; 3]) -> PyResult<()> {
        let (entered, recorded, instruction, ordinal, branch, material, prefix, reason) = {
            let retained = self.phase_evaluations()?;
            let current = retained.last().expect("retained cancellation");
            let refusal = current.refusal.as_ref().expect("original cancellation");
            (
                current.record_entered,
                current.recorded,
                current.instruction.clone(),
                current.ordinal,
                current.branch,
                current.material.clone(),
                Arc::clone(&refusal.prefix),
                refusal.reason.clone(),
            )
        };
        if !recorded {
            if entered {
                return Err(invalid(
                    "unknown Source refusal history append cannot repeat its callback",
                ));
            }
            let arguments = PyDict::new(py);
            arguments.set_item("instruction_bytes", PyBytes::new(py, &instruction))?;
            arguments.set_item("operation_ordinal", ordinal)?;
            arguments.set_item("branch", branch)?;
            arguments.set_item("resource_usage", (usage[0], usage[1], usage[2]))?;
            arguments.set_item("refusal", reason.python_value(py)?)?;
            let callback = self.scientific_owner.bind(py).getattr("record_refusal")?;
            self.phase_evaluations()?
                .last_mut()
                .expect("retained cancellation")
                .record_entered = true;
            callback.call((), Some(&arguments))?;
            let mut retained = self.phase_evaluations()?;
            let current = retained.last_mut().expect("retained cancellation");
            current.recorded = true;
            current.budget_exceeded = usage
                .into_iter()
                .zip(current.budget)
                .any(|(actual, limit)| actual > limit);
        }
        let history = self
            .phase_evaluations()?
            .last()
            .expect("retained cancellation")
            .refusal
            .as_ref()
            .expect("original cancellation")
            .history
            .as_ref()
            .map(Arc::clone);
        if history.is_none() {
            let value = self.scientific_owner.bind(py).getattr("history_bytes")?;
            if !value.is_exact_instance_of::<PyBytes>() {
                return Err(invalid(
                    "Source cancellation lost its actual terminal history",
                ));
            }
            let history: Arc<[u8]> = Arc::from(value.cast::<PyBytes>()?.as_bytes());
            cold_restore::require_terminal_history(
                &prefix,
                &history,
                &instruction,
                ordinal,
                branch,
                &reason,
                usage,
            )?;
            self.phase_evaluations()?
                .last_mut()
                .expect("retained cancellation")
                .refusal
                .as_mut()
                .expect("original cancellation")
                .history = Some(history);
        }
        if !self.records()?.refusal_known {
            if self.records()?.attempt.is_some() {
                self.resolve_terminal_record(py)?;
            } else {
                let payload = self
                    .phase_evaluations()?
                    .last()
                    .expect("retained cancellation")
                    .refusal
                    .as_ref()
                    .expect("original cancellation")
                    .payload
                    .as_ref()
                    .map(Arc::clone);
                let payload = match payload {
                    Some(payload) => payload,
                    None => {
                        let retained = self.phase_evaluations()?;
                        let refusal = retained
                            .last()
                            .expect("retained cancellation")
                            .refusal
                            .as_ref()
                            .expect("original cancellation");
                        let history =
                            Arc::clone(refusal.history.as_ref().expect("known terminal history"));
                        let cause = refusal.cause;
                        let private = refusal.private_checkpoint.as_ref().map(Arc::clone);
                        drop(retained);
                        let snapshot = self.task_use.borrow(py).state()?.snapshot.canonical.clone();
                        let reason = reason.canonical_bytes();
                        let usage: Vec<u8> = usage.into_iter().flat_map(u64::to_le_bytes).collect();
                        let private_material = match (branch, private.as_deref()) {
                            ("source", None) => &[][..],
                            ("control" | "real", Some(original)) => original,
                            _ => return Err(invalid("cancelled evaluation lost its original full private custody or Source-only absence")),
                        };
                        // Acceptance never entered. Only Source cancellation
                        // additionally has proven-absent private checkpoint custody.
                        let payload = phase_record::refusal_payload([
                            cause.as_bytes(),
                            self.source_checkpoint.as_slice(),
                            private_material,
                            prefix.as_ref(),
                            &[],
                            history.as_ref(),
                            material.as_slice(),
                            reason.as_slice(),
                            snapshot.as_slice(),
                            usage.as_slice(),
                        ])?;
                        self.records()?.check_payload_length(payload.len())?;
                        self.phase_evaluations()?
                            .last_mut()
                            .expect("retained cancellation")
                            .refusal
                            .as_mut()
                            .expect("original cancellation")
                            .payload = Some(Arc::clone(&payload));
                        payload
                    }
                };
                self.append_phase_record(py, RecordKind::TerminalRefusal, &payload)?;
            }
        }
        let released = self
            .phase_evaluations()?
            .last()
            .expect("retained cancellation")
            .released;
        if !released {
            self.preparation_inputs.resource_observer.release(py)?;
            self.phase_evaluations()?
                .last_mut()
                .expect("retained cancellation")
                .released = true;
        }
        Ok(())
    }

    pub(super) fn cancelled_evaluation_closure(
        &self,
        py: Python<'_>,
    ) -> PyResult<Arc<cold_restore::VerifiedClosure>> {
        let (phase_id, admission, record, issuer, limits) = {
            let records = self.records()?;
            if !records.refusal_known {
                return Err(invalid(
                    "Source cancellation custody precedes exact terminal readback",
                ));
            }
            (
                records.phase_id,
                records.confirmed_admission()?,
                Arc::clone(
                    records
                        .refusal_record
                        .as_ref()
                        .expect("known terminal record"),
                ),
                records.issuer(),
                records.limits,
            )
        };
        let retained = self.phase_evaluations()?;
        let current = retained.last().expect("retained cancellation");
        let refusal = current.refusal.as_ref().expect("original cancellation");
        if !current.released || !refusal.source_verified {
            return Err(invalid("Source cancellation custody precedes known physical release and Source verification"));
        }
        drop(retained);
        let issuer = PyBytes::new(py, &issuer);
        let records = PyTuple::new(
            py,
            [PyBytes::new(py, &admission), PyBytes::new(py, &record)],
        )?;
        let limits = (limits.record_bytes, limits.total_bytes, limits.records).into_pyobject(py)?;
        let fence = (
            PyBytes::new(py, &phase_id),
            PyBytes::new(py, &Sha256::digest(&self.source_checkpoint)),
            py.None(),
            self.destination.as_str(),
            phase_name(self.recipe.borrow(py).inner.source),
            phase_name(self.preparation_inputs.final_phase),
        )
            .into_pyobject(py)?;
        let closure = cold_restore::VerifiedClosure::read(
            &self.source_checkpoint,
            issuer.as_any(),
            records.as_any(),
            limits.as_any(),
            fence.as_any(),
            None,
        )?;
        let expense = closure
            .refusal
            .as_ref()
            .expect("verified native cancellation")
            .expense;
        if expense != self.task_use.borrow(py).checkpoint.proposal_expense()? {
            return Err(invalid(
                "terminal Source custody changed its original cumulative Proposal expense",
            ));
        }
        Ok(Arc::new(closure))
    }

    pub(super) fn cancelled_evaluation_outcome(
        &self,
        py: Python<'_>,
        record: &[u8],
    ) -> PyResult<Py<PyTuple>> {
        let retained = self.phase_evaluations()?;
        let refusal = retained
            .last()
            .expect("retained cancellation")
            .refusal
            .as_ref()
            .expect("original cancellation");
        Ok((
            refusal.cause,
            py.None(),
            PyBytes::new(
                py,
                refusal.history.as_ref().expect("known terminal history"),
            ),
            PyBytes::new(py, record),
        )
            .into_pyobject(py)?
            .unbind())
    }

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
            || !std::ptr::eq(&*current.owners()?.parent.borrow(py), parent)
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
            EvaluationColdStage::Cleanup
            | EvaluationColdStage::CancelledPrivateCheckpoint
            | EvaluationColdStage::CancelledPrivateRetirement => {
                return Err(invalid("evaluation cleanup has no new numerical boundary"))
            }
            EvaluationColdStage::CancelledExecutionCleanup => {
                return Err(invalid(
                    "cancelled execution cleanup has no numerical successor",
                ))
            }
            EvaluationColdStage::SourceVerification => {
                return Err(invalid(
                    "terminal Source verification has no numerical successor",
                ))
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
            &current.owners()?.parent.borrow(py).session.borrow(py),
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
        let terminal_source = current.cold_stage == EvaluationColdStage::SourceVerification;
        if !terminal_source || current.branch == "source" {
            Self::require_evaluation_entry(py, current)?;
        } else if current
            .refusal
            .as_ref()
            .is_none_or(|refusal| !refusal.private_released || refusal.private_checkpoint.is_none())
        {
            return Err(invalid(
                "Source verification precedes complete original private release",
            ));
        }
        let index = current.cold_region_index()?;
        if matches!(
            current.cold_stage,
            EvaluationColdStage::CancelledExecutionCleanup
                | EvaluationColdStage::CancelledPrivateCheckpoint
                | EvaluationColdStage::CancelledPrivateRetirement
                | EvaluationColdStage::SourceVerification
        ) && (current.refusal.is_none() || current.cancelled.is_none())
        {
            return Err(invalid(
                "terminal evaluation callback lacks known original cancellation custody",
            ));
        }
        if !self.phase_evaluation_active.load(Ordering::Acquire)
            || current.callback_result.is_some()
            || current
                .regions
                .get(index)
                .is_none_or(|original| !std::ptr::eq(&*original.borrow(py), work))
            || if terminal_source {
                self.parent.as_ptr() != work.parent.as_ptr()
            } else {
                current.owners()?.parent.as_ptr() != work.parent.as_ptr()
            }
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
        let original_task = if current.cold_stage == EvaluationColdStage::SourceVerification {
            &self.task_use
        } else {
            &current.owners()?.task
        };
        if !std::ptr::eq(&*original_task.borrow(py), task) {
            return Err(invalid(
                "evaluation cold callback changed its original issuing TaskUse",
            ));
        }
        let index = current.cold_region_index()?;
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
        if current.cold_stage == EvaluationColdStage::SourceVerification {
            if !self.is_original_source(py, session) || parent.as_ptr() != self.parent.as_ptr() {
                return Err(invalid(
                    "terminal feedback projection requires its held original Source",
                ));
            }
            // The caller next checks Source's original cancellation scope.
            // Its retired private numerical entry cannot supply this projection.
            return Ok(None);
        }
        Self::require_evaluation_entry(py, current)?;
        if !current.ready
            || !current.evaluation_admitted
            || current.callback_result.is_some()
            || parent.as_ptr() != current.owners()?.parent.as_ptr()
            || !std::ptr::eq(
                session,
                &*current.owners()?.parent.borrow(py).session.borrow(py),
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
                if self.cancelled_evaluation_refusal_retained()? {
                    return Ok(());
                }
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
                owners: Some(owners),
                branch,
                entries: Some(entries.clone().unbind()),
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
                refusal: None,
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
            if current.cold_stage == EvaluationColdStage::SourceVerification {
                if current
                    .refusal
                    .as_ref()
                    .is_none_or(|refusal| !refusal.cleanup_completed || !refusal.private_released)
                    || current.cancelled.is_none()
                    || !cold_callback
                {
                    return Err(invalid(
                        "Source feedback precedes known original cancellation cleanup",
                    ));
                }
                return Ok(());
            }
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
        if !std::ptr::eq(controller, &*current.owners()?.controller.borrow(py))
            || !std::ptr::eq(task, &*current.owners()?.task.borrow(py))
            || !std::ptr::eq(parent, &*current.owners()?.parent.borrow(py))
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
        let (material, instruction) =
            Self::singleton_lifecycle_material(current.entries()?.bind(py))?;
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
                    owners: Some(EvaluationOwners {
                        controller: self.source_controller.clone_ref(py),
                        task: self.task_use.clone_ref(py),
                        parent: self.parent.clone_ref(py),
                        model: self.model_owner(py, PhaseModelOwner::Source)?,
                    }),
                    branch: "source",
                    entries: Some(entries.clone().unbind()),
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
                    refusal: None,
                });
            }
            self.execute_evaluation(py)?;
            if self.cancelled_evaluation_refusal_retained()? {
                return Ok(());
            }
        }
    }

    fn execute_evaluation(&self, py: Python<'_>) -> PyResult<()> {
        if self.cancelled_evaluation_refusal_retained()? {
            return self.finish_cancelled_phase_evaluation(py);
        }
        match self.execute_evaluation_operation(py) {
            Ok(()) => Ok(()),
            Err(error) => self.finish_cancelled_evaluation(py, error),
        }
    }

    // Return through this boundary before terminal cleanup: the numerical stack
    // itself holds the private model, selected parent and original entry tuple.
    fn execute_evaluation_operation(&self, py: Python<'_>) -> PyResult<()> {
        let terminal = self
            .phase_evaluations()?
            .last()
            .expect("retained evaluation")
            .callback_error
            .as_ref()
            .map(|error| error.clone_ref(py));
        if let Some(error) = terminal {
            return Err(error);
        }
        let (controller, task_owner, parent_owner, model, branch) = {
            let retained = self.phase_evaluations()?;
            let current = retained.last().expect("retained original evaluation");
            (
                current.owners()?.controller.clone_ref(py),
                current.owners()?.task.clone_ref(py),
                current.owners()?.parent.clone_ref(py),
                current.owners()?.model.clone_ref(py),
                current.branch,
            )
        };
        let source = self.source.borrow(py);
        let task = task_owner.borrow(py);
        let parent = parent_owner.borrow(py);
        let manifest = SemanticCheckpointManifest::decode(&self.source_checkpoint)?;
        let (_, saved_snapshot, _, _) = TaskCheckpointSeed::decode(&manifest.task)?;
        let inputs = &self.preparation_inputs;
        inputs.require_execution_inputs(
            py,
            &self.model_owner(py, PhaseModelOwner::Source)?,
            &self.model_owner(py, PhaseModelOwner::Execute)?,
        )?;
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
                current.entries()?.clone_ref(py),
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
                .prepare_cold_model_work_regions(&work, if branch == "source" { 5 } else { 7 })
                .map_err(xlog_err)?;
            for (index, region) in regions.into_iter().enumerate() {
                let original = Py::new(
                    py,
                    PySemanticColdModelWork {
                        parent: if branch != "source" && index == 6 {
                            self.parent.clone_ref(py)
                        } else {
                            parent_owner.clone_ref(py)
                        },
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
                        // The failed numerical callback has returned. Close its
                        // visibility before entering the distinct original
                        // execution-frame cleanup and Source serializer scopes.
                        drop(_cold_visibility);
                        drop(_private_scope);
                        drop(_source_active);
                        drop(callback);
                        return Err(error);
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
            if !matches!(
                current.cold_stage,
                EvaluationColdStage::Cleanup | EvaluationColdStage::SourceVerification
            ) || (current.callback_result.is_none() && current.cancelled.is_none())
            {
                return Err(invalid("evaluation expense lacks its original known numerical completion or cancellation"));
            }
            if current
                .refusal
                .as_ref()
                .is_some_and(|refusal| !refusal.cleanup_completed || !refusal.source_verified)
            {
                return Err(invalid(
                    "cancelled evaluation expense precedes known cleanup and Source verification",
                ));
            }
            (
                current
                    .work
                    .clone()
                    .ok_or_else(|| invalid("evaluation lost its native expense owner"))?,
                current.work_closed,
                current.work_result,
                current
                    .owners
                    .as_ref()
                    .map(|owners| owners.parent.clone_ref(py)),
                current.custody.clone(),
                current.physical_peak,
            )
        };
        let source = self.source.borrow(py);
        if !closed {
            let unused_terminal_regions = {
                let retained = self.phase_evaluations()?;
                let current = retained.last().expect("retained evaluation");
                current.refusal.is_none().then_some(current.regions.len())
            };
            if let Some(end) = unused_terminal_regions {
                source
                    .owner()?
                    .cancel_unentered_cold_model_work_regions(&work, end)
                    .map_err(xlog_err)?;
            }
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
                        .as_ref()
                        .ok_or_else(|| {
                            invalid("evaluation expense lost its original attached private parent")
                        })?
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
        let cancelled = self
            .phase_evaluations()?
            .last()
            .is_some_and(|current| current.cancelled.is_some());
        if cancelled {
            self.capture_cancelled_evaluation_refusal(py)?;
            // The native proof, not this exception or its traceback, carries the
            // known disposition. Release failed numerical frames before sealing
            // the same original physical interval.
            drop(original);
            return self.finish_cancelled_phase_evaluation(py);
        }
        Err(original)
    }
}
