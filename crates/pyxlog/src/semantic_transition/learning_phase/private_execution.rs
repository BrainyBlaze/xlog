//! Original private segment admission, reports and suspended callback custody.

use super::*;
use xlog_cuda::{SemanticColdModelWorkResult, SemanticColdNativeWork};

#[derive(Clone, Copy, PartialEq, Eq)]
enum PrivateColdRole {
    Prefix,
    ModelPreparation,
    SuccessorPreparation,
    Retirement,
    Adoption,
}

pub(super) struct PrivateExecutionGroup {
    previous: Option<Box<PrivateExecutionGroup>>,
    branch: &'static str,
    entries: Py<PyTuple>,
    materials: Vec<Vec<u8>>,
    instruction: Vec<u8>,
    kinds: Vec<SemanticTransitionKind>,
    ordinal: u64,
    budgets: Vec<[u64; 3]>,
    restored: Py<PySemanticTransitionRestoredCheckpoint>,
    started: bool,
    ready: bool,
    preparation_work: Option<Py<PySemanticColdModelWork>>,
    preparation_custody: Option<SemanticColdNativeWork>,
    preparation_report: Option<SemanticColdModelWorkResult>,
    preparation_role: PrivateColdRole,
    preparation_reports: Vec<SemanticColdModelWorkResult>,
    preparation_detached: bool,
    build_entered: bool,
    build_scope: Option<Arc<()>>,
    native_steps: Vec<SemanticPreparedStep>,
    selected_parent: Option<Py<PySemanticPublishedParent>>,
    native_retired: bool,
    retirement_work: Option<Py<PySemanticColdModelWork>>,
    retirement_custody: Option<SemanticColdNativeWork>,
    retirement_report: Option<SemanticColdModelWorkResult>,
    retirement_ready: bool,
    retirement_entered: bool,
    adoption_work: Option<Py<PySemanticColdModelWork>>,
    adoption_custody: Option<SemanticColdNativeWork>,
    adoption_report: Option<SemanticColdModelWorkResult>,
    callback_entered: bool,
    callback_pending: bool,
    callback_result: Option<Py<PyAny>>,
    callback_error: Option<PyErr>,
    observer_finish_entered: bool,
    physical_peak: Option<u64>,
    record_entered: Option<usize>,
    records_completed: usize,
    records_released: bool,
    budget_exceeded: bool,
}

/// Only the retained numerical callback sees the original inner native phase.
/// Suspension restores its actual Recording/Prepared/CompletedPending state
/// inside ArenaPreparing; it neither clears privacy nor invents nonentry.
pub(super) struct PrivateTaskCallbackScope<'a> {
    active: &'a AtomicBool,
    task: &'a PySemanticTransitionTaskUse,
    session: &'a PySemanticTransitionSession,
}

impl<'a> PrivateTaskCallbackScope<'a> {
    pub(super) fn enter(
        active: &'a AtomicBool,
        task: &'a PySemanticTransitionTaskUse,
        session: &'a PySemanticTransitionSession,
    ) -> PyResult<Self> {
        session.require_creator()?;
        if !session.learning_preparing.load(Ordering::Acquire) {
            return Err(invalid(
                "private execution lost its original preparing Session",
            ));
        }
        let mut state = task.state()?;
        if !matches!(state.phase, TaskUsePhase::ArenaPreparing(_)) {
            return Err(invalid(
                "private execution requires its retained private task state",
            ));
        }
        active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid("private numerical callback cannot be nested"))?;
        state.finish_arena_preparation()?;
        Ok(Self {
            active,
            task,
            session,
        })
    }
}

impl Drop for PrivateTaskCallbackScope<'_> {
    fn drop(&mut self) {
        let work = self
            .session
            .active_cold_model_work
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(work) = work {
            Python::attach(|py| work.borrow(py).active.store(false, Ordering::Release));
        }
        let mut state = self
            .task
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !matches!(state.phase, TaskUsePhase::ArenaPreparing(_)) {
            state.phase = TaskUsePhase::ArenaPreparing(Box::new(state.phase.clone()));
        }
        self.active.store(false, Ordering::Release);
    }
}

impl PySemanticLearningPhaseTransition {
    pub(super) fn private_segment_terminal_input(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<super::phase_evaluation::SegmentTerminalInput>> {
        let retained = self.private_group()?;
        let Some(group) = retained.as_ref().filter(|group| group.budget_exceeded) else {
            return Ok(None);
        };
        if !group.native_retired
            || group.kinds.is_empty()
            || group.preparation_report.is_none()
            || group.retirement_report.is_none()
            || group.adoption_report.is_none()
            || !group.observer_finish_entered
            || group.callback_pending
            || group.callback_error.is_some()
            || group.record_entered.is_some()
            || group.records_completed != group.kinds.len()
            || !group.records_released
            || group.physical_peak.is_none()
        {
            return Err(invalid(
                "terminal cleanup requires the original complete group and known physical release",
            ));
        }
        let returned = group
            .callback_result
            .as_ref()
            .ok_or_else(|| invalid("terminal cleanup lost its original selected model handoff"))?
            .bind(py)
            .cast::<PyTuple>()?;
        let parent = group
            .selected_parent
            .as_ref()
            .ok_or_else(|| invalid("terminal cleanup lost its original selected native parent"))?;
        if returned.len() != 3
            || returned.get_item(0)?.as_ptr() != parent.as_ptr()
            || returned.get_item(1)?.is_none()
        {
            return Err(invalid(
                "terminal cleanup cannot replace its original completed handoff",
            ));
        }
        let restored = group.restored.borrow(py);
        let original_parent = parent.borrow(py);
        let identity = original_parent
            .session
            .borrow(py)
            .owner()?
            .published_identity(&*original_parent.lease()?)
            .map_err(xlog_err)?;
        Ok(Some(super::phase_evaluation::SegmentTerminalInput {
            owners: super::phase_evaluation::EvaluationOwners {
                controller: restored.controller.clone_ref(py),
                task: restored.task_use.clone_ref(py),
                parent: parent.clone_ref(py),
                model: returned.get_item(1)?.unbind(),
            },
            branch: group.branch,
            instruction: group.instruction.clone(),
            completed: super::phase_evaluation::CompletedSegmentTerminal {
                group_ordinal: group.ordinal,
                count: u64::try_from(group.kinds.len())
                    .map_err(|_| invalid("terminal original group extent overflowed"))?,
                parent: identity,
            },
        }))
    }

    pub(super) fn drop_completed_private_execution_owners(
        &self,
        branch: &'static str,
    ) -> PyResult<()> {
        let original = {
            let mut retained = self.private_group()?;
            let mut cursor = retained.as_ref();
            while let Some(group) = cursor {
                if group.branch != branch
                    || !group.native_retired
                    || group.records_completed != group.kinds.len()
                    || !group.records_released
                    || group.callback_error.is_some()
                    || group.budget_exceeded
                        && !self.completed_segment_terminal_matches(
                            branch,
                            group.ordinal,
                            group.kinds.len() as u64,
                        )?
                {
                    return Err(invalid(
                        "model retirement cannot discard unfinished private execution owners",
                    ));
                }
                cursor = group.previous.as_deref();
            }
            retained.take()
        };
        drop(original);
        Ok(())
    }

    pub(super) fn private_group_successor_input(
        &self,
        py: Python<'_>,
        branch: &'static str,
    ) -> PyResult<(super::phase_evaluation::EvaluationOwners, u64)> {
        let retained = self.private_group()?;
        let group = retained
            .as_ref()
            .ok_or_else(|| invalid("private evaluation lost its original numerical group"))?;
        if group.branch != branch {
            return Err(invalid(
                "private evaluation changed its original branch handoff",
            ));
        }
        if group.records_completed != group.kinds.len()
            || !group.records_released
            || group.budget_exceeded
        {
            return Err(invalid("private evaluation precedes its complete original numerical receipts or physical release"));
        }
        let returned = group
            .callback_result
            .as_ref()
            .ok_or_else(|| invalid("private evaluation lost its selected parent/model handoff"))?;
        let returned = returned.bind(py).cast::<PyTuple>()?;
        let restored = group.restored.borrow(py);
        let parent = group
            .selected_parent
            .as_ref()
            .expect("known selected parent");
        if returned.get_item(0)?.as_ptr() != parent.as_ptr() {
            return Err(invalid(
                "private evaluation changed its selected native parent",
            ));
        }
        Ok((
            super::phase_evaluation::EvaluationOwners {
                controller: restored.controller.clone_ref(py),
                task: restored.task_use.clone_ref(py),
                parent: parent.clone_ref(py),
                model: returned.get_item(1)?.unbind(),
            },
            group
                .ordinal
                .checked_add(group.kinds.len() as u64)
                .ok_or_else(|| invalid("private evaluation position overflowed"))?,
        ))
    }

    pub(super) fn private_current_restore_input(
        &self,
        py: Python<'_>,
        branch: &'static str,
    ) -> PyResult<(super::phase_evaluation::EvaluationOwners, u64)> {
        if self.private_group()?.is_some() {
            return self.private_group_successor_input(py, branch);
        }
        // A restore can be the first private operation or immediately follow
        // another restore. Use that actual known handoff, never the source copy.
        let (restored, ordinal) = self.private_execution_input(py, branch)?;
        let restored = restored.borrow(py);
        Ok((
            super::phase_evaluation::EvaluationOwners {
                controller: restored.controller.clone_ref(py),
                task: restored.task_use.clone_ref(py),
                parent: restored.parent.clone_ref(py),
                model: restored.model.clone_ref(py),
            },
            ordinal,
        ))
    }

    fn private_group(&self) -> PyResult<MutexGuard<'_, Option<PrivateExecutionGroup>>> {
        self.private_execution
            .lock()
            .map_err(|_| invalid("original private execution custody mutex is poisoned"))
    }

    fn require_private_group_entries(
        py: Python<'_>,
        group: &PrivateExecutionGroup,
    ) -> PyResult<()> {
        let entries = group.entries.bind(py);
        if entries.len() != group.materials.len() {
            return Err(invalid(
                "private execution changed its original complete group",
            ));
        }
        for (entry, original) in entries.iter().zip(&group.materials) {
            let singleton = PyTuple::new(py, [entry])?;
            let (material, instruction) = Self::singleton_lifecycle_material(&singleton)?;
            if &material != original || instruction != group.instruction {
                return Err(invalid(
                    "private execution changed its original frozen entries",
                ));
            }
        }
        Ok(())
    }

    fn require_private_group_owner(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
    ) -> PyResult<()> {
        if !self.private_execution_active.load(Ordering::Acquire) {
            return Err(invalid(
                "private operation belongs only to its original active callback",
            ));
        }
        self.records()?.require_preparation_admission()?;
        self.preparation_inputs
            .require_program(py, &self.scientific_owner)?;
        let retained = self.private_group()?;
        let group = retained
            .as_ref()
            .ok_or_else(|| invalid("private callback lost its original group"))?;
        Self::require_private_group_entries(py, group)?;
        let restored = group.restored.borrow(py);
        if !group.ready
            || !group.callback_entered
            || group.callback_result.is_some()
            || !std::ptr::eq(session, &*restored.session.borrow(py))
            || !std::ptr::eq(task, &*restored.task_use.borrow(py))
        {
            return Err(invalid(
                "private callback changed its original group, Session or task",
            ));
        }
        Ok(())
    }

    pub(in crate::semantic_transition) fn require_private_group_build(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
        transitions: &[SemanticTransitionKind],
    ) -> PyResult<()> {
        self.require_private_group_owner(py, session, task)?;
        let mut retained = self.private_group()?;
        let group = retained.as_mut().expect("retained original group");
        if group.build_entered || transitions != group.kinds {
            return Err(invalid(
                "private build must consume its complete original roster exactly once",
            ));
        }
        let work = group
            .preparation_work
            .as_ref()
            .expect("retained preparation work")
            .borrow(py);
        self.source
            .borrow(py)
            .owner()?
            .require_closed_cold_model_work(&work.inner)
            .map_err(xlog_err)?;
        // This precedes nominal reservation, allocation and producer callbacks.
        group.build_entered = true;
        Ok(())
    }

    pub(in crate::semantic_transition) fn retain_private_build_scope(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
        scope: &Arc<()>,
        steps: &[SemanticPreparedStep],
    ) -> PyResult<()> {
        self.require_private_group_owner(py, session, task)?;
        let mut retained = self.private_group()?;
        let group = retained.as_mut().expect("retained original group");
        if !group.build_entered || group.build_scope.is_some() || steps.len() != group.kinds.len() {
            return Err(invalid(
                "private build cannot replace its original recording scope",
            ));
        }
        group.build_scope = Some(Arc::clone(scope));
        group.native_steps = steps.to_vec();
        Ok(())
    }

    fn finish_private_cold_report(
        &self,
        py: Python<'_>,
        role: PrivateColdRole,
    ) -> PyResult<SemanticColdModelWorkResult> {
        let (restored, work, custody, report) = {
            let retained = self.private_group()?;
            let group = retained.as_ref().expect("retained original group");
            let (work, custody, report) = match role {
                PrivateColdRole::Adoption => (
                    &group.adoption_work,
                    &group.adoption_custody,
                    group.adoption_report,
                ),
                PrivateColdRole::Retirement => (
                    &group.retirement_work,
                    &group.retirement_custody,
                    group.retirement_report,
                ),
                original if original == group.preparation_role => (
                    &group.preparation_work,
                    &group.preparation_custody,
                    group.preparation_report,
                ),
                _ => {
                    return Err(invalid(
                        "private cold report changed its original callback position",
                    ))
                }
            };
            (
                group.restored.clone_ref(py),
                work.as_ref().map(|work| work.clone_ref(py)),
                custody.as_ref().cloned(),
                report,
            )
        };
        if let Some(report) = report {
            return Ok(report);
        }
        let work =
            work.ok_or_else(|| invalid("private operation lacks its actual cold model recorder"))?;
        if let Some(custody) = custody {
            restored
                .borrow(py)
                .session
                .borrow(py)
                .owner()?
                .detach_shared_cold_native_work(&custody)
                .map_err(xlog_err)?;
            let mut retained = self.private_group()?;
            let group = retained.as_mut().expect("retained original group");
            match role {
                PrivateColdRole::Adoption => group.adoption_custody = None,
                PrivateColdRole::Retirement => group.retirement_custody = None,
                _ => group.preparation_custody = None,
            }
            drop(retained);
            drop(custody);
        }
        let streams = checkpoint_consumer_streams(
            self.preparation_inputs
                .consumer_streams
                .python_value(py)?
                .bind(py),
            &mut (16 * 1024 * 1024),
        )?;
        let report = self
            .source
            .borrow(py)
            .owner()?
            .finish_cold_model_work(
                &*self.parent.borrow(py).lease()?,
                &work.borrow(py).inner,
                &streams,
            )
            .map_err(xlog_err)?;
        work.borrow(py).active.store(false, Ordering::Release);
        let child = restored.borrow(py);
        let session = child.session.borrow(py);
        let mut active = session
            .active_cold_model_work
            .lock()
            .map_err(|_| invalid("private cold callback custody mutex is poisoned"))?;
        if active
            .as_ref()
            .is_some_and(|original| original.as_ptr() == work.as_ptr())
        {
            active.take();
        }
        drop(active);
        let mut retained = self.private_group()?;
        let group = retained.as_mut().expect("retained original group");
        match role {
            PrivateColdRole::Adoption => group.adoption_report = Some(report),
            PrivateColdRole::Retirement => group.retirement_report = Some(report),
            _ => group.preparation_report = Some(report),
        }
        Ok(report)
    }

    fn issue_private_cold_callback(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        role: PrivateColdRole,
        parent: &Py<PySemanticPublishedParent>,
        ordinal: u64,
    ) -> PyResult<Py<PySemanticColdModelWork>> {
        let inner = self
            .source
            .borrow(py)
            .owner()?
            .prepare_cold_model_work(
                &*self.parent.borrow(py).lease()?,
                self.preparation_inputs.cold_model_work_capacity,
                ordinal,
                self.records()?.confirmed_admission()?,
            )
            .map_err(xlog_err)?;
        let custody = self
            .source
            .borrow(py)
            .owner()?
            .share_cold_native_work(&inner)
            .map_err(xlog_err)?;
        let work = Py::new(
            py,
            PySemanticColdModelWork {
                parent: parent.clone_ref(py),
                reader: self.parent.clone_ref(py),
                inner,
                region: None,
                active: AtomicBool::new(true),
            },
        )?;
        {
            let mut retained = self.private_group()?;
            let group = retained.as_mut().expect("retained original group");
            match role {
                PrivateColdRole::Adoption => {
                    group.adoption_work = Some(work.clone_ref(py));
                    group.adoption_custody = Some(custody.clone());
                }
                PrivateColdRole::Retirement => {
                    group.retirement_work = Some(work.clone_ref(py));
                    group.retirement_custody = Some(custody.clone());
                }
                _ => {
                    if let Some(report) = group.preparation_report.take() {
                        group.preparation_reports.push(report);
                    }
                    group.preparation_role = role;
                    group.preparation_work = Some(work.clone_ref(py));
                    group.preparation_custody = Some(custody.clone());
                    group.preparation_detached = false;
                }
            }
        }
        session
            .owner()?
            .attach_shared_cold_native_work(&*parent.borrow(py).lease()?, custody)
            .map_err(xlog_err)?;
        if role == PrivateColdRole::Retirement {
            self.private_group()?
                .as_mut()
                .expect("retained original group")
                .retirement_ready = true;
        }
        *session
            .active_cold_model_work
            .lock()
            .map_err(|_| invalid("private cold callback custody mutex is poisoned"))? =
            Some(work.clone_ref(py));
        Ok(work)
    }

    pub(in crate::semantic_transition) fn begin_private_model_preparation(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
        successors: bool,
    ) -> PyResult<()> {
        self.require_private_group_owner(py, session, task)?;
        let (parent, ordinal, previous) = {
            let retained = self.private_group()?;
            let group = retained.as_ref().expect("retained original group");
            let previous = if successors {
                PrivateColdRole::ModelPreparation
            } else {
                PrivateColdRole::Prefix
            };
            if group.build_scope.is_none() || group.preparation_role != previous {
                return Err(invalid(
                    "private preparation callback changed its original build position",
                ));
            }
            let parent = group.restored.borrow(py).parent.clone_ref(py);
            (parent, group.ordinal, previous)
        };
        self.finish_private_cold_report(py, previous)?;
        self.issue_private_cold_callback(
            py,
            session,
            if successors {
                PrivateColdRole::SuccessorPreparation
            } else {
                PrivateColdRole::ModelPreparation
            },
            &parent,
            ordinal,
        )
        .map(|_| ())
    }

    pub(in crate::semantic_transition) fn private_capture_boundary(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
        end: bool,
    ) -> PyResult<()> {
        self.require_private_group_owner(py, session, task)?;
        let mut retained = self.private_group()?;
        let group = retained.as_mut().expect("retained original group");
        if group.preparation_role != PrivateColdRole::SuccessorPreparation
            || group.preparation_detached != end
        {
            return Err(invalid(
                "private capture changed its original cold preparation boundary",
            ));
        }
        let work = group
            .preparation_work
            .as_ref()
            .expect("retained successor recorder")
            .borrow(py);
        self.source
            .borrow(py)
            .owner()?
            .require_closed_cold_model_work(&work.inner)
            .map_err(xlog_err)?;
        let custody = group
            .preparation_custody
            .as_ref()
            .expect("retained original tally");
        if end {
            session
                .owner()?
                .attach_shared_cold_native_work(
                    &*group.restored.borrow(py).parent.borrow(py).lease()?,
                    custody.clone(),
                )
                .map_err(xlog_err)?;
        } else {
            session
                .owner()?
                .detach_shared_cold_native_work(custody)
                .map_err(xlog_err)?;
        }
        group.preparation_detached = !end;
        Ok(())
    }

    pub(in crate::semantic_transition) fn bind_private_step_capture(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
        step: &PySemanticPreparedStep,
        capture_stream: u64,
        end: bool,
    ) -> PyResult<()> {
        self.require_private_group_owner(py, session, task)?;
        let (ordinal, index) = {
            let retained = self.private_group()?;
            let group = retained.as_ref().expect("retained original group");
            if group
                .build_scope
                .as_ref()
                .is_none_or(|scope| !Arc::ptr_eq(scope, &step.scope))
            {
                return Err(invalid(
                    "physical capture changed its original native step scope",
                ));
            }
            let prepared = session
                .prepared_segment
                .lock()
                .map_err(|_| invalid("prepared segment mutex is poisoned"))?;
            let prepared = prepared
                .as_ref()
                .ok_or_else(|| invalid("physical capture lost its original prepared roster"))?;
            let index = prepared
                .steps
                .iter()
                .position(|original| std::ptr::eq(&*original.borrow(py), step))
                .ok_or_else(|| invalid("physical capture changed its actual original step"))?;
            (
                group
                    .ordinal
                    .checked_add(index as u64)
                    .ok_or_else(|| invalid("physical step position overflowed"))?,
                index as u64,
            )
        };
        self.preparation_inputs.resource_observer.bind_step_capture(
            py,
            ordinal,
            index,
            capture_stream,
            end,
        )
    }

    pub(in crate::semantic_transition) fn finish_private_group_preparation(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
        scope: &Arc<()>,
    ) -> PyResult<()> {
        self.require_private_group_owner(py, session, task)?;
        {
            let retained = self.private_group()?;
            let group = retained.as_ref().expect("retained original group");
            if group
                .build_scope
                .as_ref()
                .is_none_or(|original| !Arc::ptr_eq(original, scope))
            {
                return Err(invalid(
                    "private submission changed its original prepared scope",
                ));
            }
        }
        self.finish_private_cold_report(py, PrivateColdRole::SuccessorPreparation)
            .map(|_| ())
    }

    pub(in crate::semantic_transition) fn begin_private_segment_retirement(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
    ) -> PyResult<()> {
        self.require_private_group_owner(py, session, task)?;
        let mut retained = self.private_group()?;
        let group = retained.as_mut().expect("retained original group");
        if group.retirement_work.is_none()
            || !group.retirement_ready
            || group.retirement_entered
            || group.native_retired
        {
            return Err(invalid(
                "private retirement requires its original final-use cold owner",
            ));
        }
        group.retirement_entered = true;
        Ok(())
    }

    pub(in crate::semantic_transition) fn retain_private_selected_parent(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
        parent: &Py<PySemanticPublishedParent>,
    ) -> PyResult<()> {
        self.require_private_group_owner(py, session, task)?;
        parent.borrow(py).require_task(py, task)?;
        let mut retained = self.private_group()?;
        let group = retained.as_mut().expect("retained original group");
        if group.preparation_report.is_none()
            || group
                .selected_parent
                .as_ref()
                .is_some_and(|original| original.as_ptr() != parent.as_ptr())
        {
            return Err(invalid(
                "private completion changed its original selected parent",
            ));
        }
        group.selected_parent = Some(parent.clone_ref(py));
        let ordinal = group
            .ordinal
            .checked_add(group.kinds.len() as u64 - 1)
            .ok_or_else(|| invalid("private retirement position overflowed"))?;
        if group.retirement_work.is_some() {
            return if group.retirement_ready {
                Ok(())
            } else {
                Err(invalid(
                    "private final-use cold admission has unknown completion",
                ))
            };
        }
        drop(retained);
        self.issue_private_cold_callback(py, session, PrivateColdRole::Retirement, parent, ordinal)
            .map(|_| ())
    }

    pub(in crate::semantic_transition) fn retain_private_segment_retirement(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
    ) -> PyResult<()> {
        self.require_private_group_owner(py, session, task)?;
        let mut retained = self.private_group()?;
        let group = retained.as_mut().expect("retained original group");
        if group.selected_parent.is_none() || !group.retirement_entered || group.native_retired {
            return Err(invalid(
                "private adoption requires one known original segment retirement",
            ));
        }
        group.native_retired = true;
        Ok(())
    }

    pub(in crate::semantic_transition) fn private_adoption_work(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
    ) -> PyResult<Py<PySemanticColdModelWork>> {
        self.require_private_group_owner(py, session, task)?;
        let (parent, ordinal) = {
            let retained = self.private_group()?;
            let group = retained.as_ref().expect("retained original group");
            if !group.native_retired
                || group.preparation_report.is_none()
                || group.adoption_work.is_some()
            {
                return Err(invalid(
                    "cold adoption cannot replace preparation or repeat its original recorder",
                ));
            }
            (
                group
                    .selected_parent
                    .as_ref()
                    .expect("known selected parent")
                    .clone_ref(py),
                group
                    .ordinal
                    .checked_add(group.kinds.len() as u64 - 1)
                    .ok_or_else(|| invalid("private adoption position overflowed"))?,
            )
        };
        self.finish_private_cold_report(py, PrivateColdRole::Retirement)?;
        self.issue_private_cold_callback(py, session, PrivateColdRole::Adoption, &parent, ordinal)
    }

    pub(in crate::semantic_transition) fn private_adoption_due(&self) -> PyResult<bool> {
        Ok(self.private_execution_active.load(Ordering::Acquire)
            && self
                .private_group()?
                .as_ref()
                .is_some_and(|group| group.native_retired && group.adoption_work.is_none()))
    }

    pub(in crate::semantic_transition) fn private_numeric_active(&self) -> bool {
        self.private_execution_active.load(Ordering::Acquire)
    }

    pub(in crate::semantic_transition) fn require_private_numeric_cold_callback(
        &self,
        py: Python<'_>,
        work: &PySemanticColdModelWork,
    ) -> PyResult<()> {
        let parent = work.parent.borrow(py);
        self.require_private_group_owner(
            py,
            &parent.session.borrow(py),
            &parent.task_use.borrow(py),
        )?;
        let retained = self.private_group()?;
        let group = retained.as_ref().expect("retained original group");
        if ![
            &group.preparation_work,
            &group.retirement_work,
            &group.adoption_work,
        ]
        .iter()
        .any(|original| {
            original
                .as_ref()
                .is_some_and(|original| std::ptr::eq(&*original.borrow(py), work))
        }) {
            return Err(invalid(
                "private cold callback changed its original operation-owned registrar",
            ));
        }
        Ok(())
    }

    pub(in crate::semantic_transition) fn private_numeric_feedback_projection(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        parent: &ContentStepOwner,
    ) -> PyResult<Py<PyTuple>> {
        let retained = self.private_group()?;
        let group = retained
            .as_ref()
            .ok_or_else(|| invalid("private feedback lost its original group"))?;
        let restored = group.restored.clone_ref(py);
        match parent {
            ContentStepOwner::Published(parent) => {
                let initial = &restored.borrow(py).parent;
                if parent.as_ptr() != initial.as_ptr()
                    && group
                        .selected_parent
                        .as_ref()
                        .is_none_or(|actual| actual.as_ptr() != parent.as_ptr())
                {
                    return Err(invalid(
                        "private feedback changed its actual numerical parent",
                    ));
                }
            }
            ContentStepOwner::Prepared(step) => {
                if group
                    .build_scope
                    .as_ref()
                    .is_none_or(|scope| !Arc::ptr_eq(scope, &step.borrow(py).scope))
                {
                    return Err(invalid(
                        "private feedback changed its original prepared scope",
                    ));
                }
            }
        }
        drop(retained);
        self.require_private_group_owner(py, session, &restored.borrow(py).task_use.borrow(py))?;
        let materials = feedback_materials(
            self.preparation_inputs
                .feedback_interventions
                .bind(py)
                .as_any(),
        )?;
        let retained = self.private_group()?;
        let branch = retained.as_ref().expect("original private group").branch;
        let (material, enabled) = match branch {
            "real" => (materials[0].clone(), true),
            "control" => (materials[1].clone(), false),
            _ => {
                return Err(invalid(
                    "private numerical feedback lost its original branch",
                ))
            }
        };
        Ok((material, enabled).into_pyobject(py)?.unbind())
    }

    pub(super) fn execute_private_numerical_sequence(
        &self,
        py: Python<'_>,
        pending: &Py<Self>,
        branch: &'static str,
    ) -> PyResult<()> {
        if self.readonly_terminal_refusal_retained()? {
            return self.finish_readonly_terminal_refusal(py);
        }
        // Return out of the numerical stack before full private retirement:
        // local handoff tuples must not keep the original model/Session alive.
        match self.execute_private_numerical_operations(py, pending, branch) {
            Ok(()) => Ok(()),
            Err(original) => {
                if self.private_segment_terminal_input(py)?.is_some() {
                    self.finish_segment_budget_refusal(py, original)
                } else {
                    Err(original)
                }
            }
        }
    }

    fn execute_private_numerical_operations(
        &self,
        py: Python<'_>,
        pending: &Py<Self>,
        branch: &'static str,
    ) -> PyResult<()> {
        if self.intermediate_restore_unfinished(branch)? {
            self.execute_intermediate_restore(py, pending, branch)?;
        }
        if self.private_group()?.is_some() {
            self.execute_private_group(py, branch, None)?;
        }
        loop {
            self.preparation_inputs
                .require_program(py, &self.scientific_owner)?;
            let next = self.scientific_owner.bind(py).getattr("next_operation")?;
            if !next.is_exact_instance_of::<PyTuple>() || next.cast::<PyTuple>()?.is_empty() {
                return Err(invalid(
                    "private numerical sequence lost its original next group",
                ));
            }
            let entries = next.cast::<PyTuple>()?;
            let first = PyTuple::new(py, [entries.get_item(0)?])?;
            let (material, _) = Self::singleton_lifecycle_material(&first)?;
            let material = ColdValue::from_canonical_bytes(&material)?;
            let fields = material.fields(6)?;
            if fields[1].text()? != branch {
                return Err(invalid(
                    "private numerical sequence changed its original branch",
                ));
            }
            if fields[0].text()? == "restore" {
                self.execute_intermediate_restore(py, pending, branch)?;
                continue;
            }
            if !matches!(fields[0].text()?, "proposal" | "update" | "recompute") {
                // The original lifecycle executor consumes the complete final
                // evaluation roster; numerical groups are never shortened.
                return Ok(());
            }
            if self.private_group()?.is_none() {
                self.execute_private_group(py, branch, None)?;
                continue;
            }
            let (selected, ordinal) = self.private_group_successor_input(py, branch)?;
            if fields[2].unsigned()? != ordinal || fields[3].unsigned()? != 0 {
                return Err(invalid(
                    "private numerical sequence changed its contiguous original group position",
                ));
            }
            let session = selected.parent.borrow(py).session.clone_ref(py);
            let successor = PySemanticTransitionRestoredCheckpoint::issue(
                py,
                session,
                selected.controller,
                selected.task,
                selected.parent,
                selected.model,
            )?;
            self.execute_private_group(py, branch, Some((successor, ordinal)))?;
        }
    }

    fn execute_private_group(
        &self,
        py: Python<'_>,
        branch: &'static str,
        input: Option<(Py<PySemanticTransitionRestoredCheckpoint>, u64)>,
    ) -> PyResult<()> {
        if self
            .private_group()?
            .as_ref()
            .is_some_and(|group| group.branch != branch)
        {
            return Err(invalid(
                "private execution cannot replace its original branch group",
            ));
        }
        if let Some((_, ordinal)) = &input {
            let retained = self.private_group()?;
            let original = retained
                .as_ref()
                .ok_or_else(|| invalid("private successor lost its original completed group"))?;
            if original.records_completed != original.kinds.len()
                || !original.records_released
                || original.budget_exceeded
                || original.callback_error.is_some()
                || original.ordinal.checked_add(original.kinds.len() as u64) != Some(*ordinal)
            {
                return Err(invalid(
                    "private successor cannot replace unfinished, refused or unknown original work",
                ));
            }
        }
        if self.private_group()?.is_none() || input.is_some() {
            let (restored, ordinal) = match input {
                Some(input) => input,
                None => self.private_execution_input(py, branch)?,
            };
            self.preparation_inputs
                .require_program(py, &self.scientific_owner)?;
            let next = self.scientific_owner.bind(py).getattr("next_operation")?;
            let entries = next.cast::<PyTuple>()?;
            if !next.is_exact_instance_of::<PyTuple>() || entries.is_empty() {
                return Err(invalid(
                    "private execution requires its original complete numeric group",
                ));
            }
            let mut materials = Vec::with_capacity(entries.len());
            let mut kinds = Vec::with_capacity(entries.len());
            let mut budgets = Vec::with_capacity(entries.len());
            let mut original_instruction = None;
            for (index, entry) in entries.iter().enumerate() {
                let singleton = PyTuple::new(py, [entry])?;
                let (material, instruction) = Self::singleton_lifecycle_material(&singleton)?;
                let fields = ColdValue::from_canonical_bytes(&material)?;
                let fields = fields.fields(6)?;
                let expected = ordinal
                    .checked_add(index as u64)
                    .ok_or_else(|| invalid("private numerical position overflowed"))?;
                if fields[1].text()? != branch
                    || fields[2].unsigned()? != expected
                    || fields[3].unsigned()? != index as u64
                    || original_instruction
                        .as_ref()
                        .is_some_and(|original| original != &instruction)
                {
                    return Err(invalid(
                        "private execution changed its original contiguous branch roster",
                    ));
                }
                let kind = transition_kind(&fields[0])?;
                if kind == SemanticTransitionKind::Drain {
                    return Err(invalid(
                        "private schedule contains an unscheduled numeric drain",
                    ));
                }
                let budget = fields[4].fields(3)?;
                budgets.push([
                    budget[0].unsigned()?,
                    budget[1].unsigned()?,
                    budget[2].unsigned()?,
                ]);
                materials.push(material);
                kinds.push(kind);
                original_instruction = Some(instruction);
            }
            // Retain the completed predecessor until the new original physical
            // interval and shared native tally cover destruction of its owners.
            let previous = self.private_group()?.take().map(Box::new);
            *self.private_group()? = Some(PrivateExecutionGroup {
                previous,
                branch,
                entries: entries.clone().unbind(),
                materials,
                instruction: original_instruction.expect("nonempty group"),
                kinds,
                ordinal,
                budgets,
                restored,
                started: false,
                ready: false,
                preparation_work: None,
                preparation_custody: None,
                preparation_report: None,
                preparation_role: PrivateColdRole::Prefix,
                preparation_reports: Vec::new(),
                preparation_detached: false,
                build_entered: false,
                build_scope: None,
                native_steps: Vec::new(),
                selected_parent: None,
                native_retired: false,
                adoption_work: None,
                adoption_custody: None,
                adoption_report: None,
                retirement_work: None,
                retirement_custody: None,
                retirement_report: None,
                retirement_ready: false,
                retirement_entered: false,
                callback_entered: false,
                callback_pending: false,
                callback_result: None,
                callback_error: None,
                observer_finish_entered: false,
                physical_peak: None,
                record_entered: None,
                records_completed: 0,
                records_released: false,
                budget_exceeded: false,
            });
        }
        let (restored, entries, ordinal, started, entered, pending, returned) = {
            let retained = self.private_group()?;
            let group = retained.as_ref().expect("retained original group");
            Self::require_private_group_entries(py, group)?;
            if let Some(error) = &group.callback_error {
                return Err(error.clone_ref(py));
            }
            (
                group.restored.clone_ref(py),
                group.entries.clone_ref(py),
                group.ordinal,
                group.started,
                group.callback_entered,
                group.callback_pending,
                group
                    .callback_result
                    .as_ref()
                    .map(|result| result.clone_ref(py)),
            )
        };
        let child = restored.borrow(py);
        let session = child.session.borrow(py);
        let task = child.task_use.borrow(py);
        if !started {
            self.private_group()?
                .as_mut()
                .expect("retained original group")
                .started = true;
            self.preparation_inputs
                .resource_observer
                .begin(py, ordinal)?;
            let inner = self
                .source
                .borrow(py)
                .owner()?
                .prepare_cold_model_work(
                    &*self.parent.borrow(py).lease()?,
                    self.preparation_inputs.cold_model_work_capacity,
                    ordinal,
                    self.records()?.confirmed_admission()?,
                )
                .map_err(xlog_err)?;
            let custody = self
                .source
                .borrow(py)
                .owner()?
                .share_cold_native_work(&inner)
                .map_err(xlog_err)?;
            let work = Py::new(
                py,
                PySemanticColdModelWork {
                    parent: child.parent.clone_ref(py),
                    reader: self.parent.clone_ref(py),
                    inner,
                    region: None,
                    active: AtomicBool::new(false),
                },
            )?;
            {
                let mut retained = self.private_group()?;
                let group = retained.as_mut().expect("retained original group");
                group.preparation_work = Some(work);
                group.preparation_custody = Some(custody.clone());
            }
            session
                .owner()?
                .attach_shared_cold_native_work(&*child.parent.borrow(py).lease()?, custody)
                .map_err(xlog_err)?;
            let previous = self
                .private_group()?
                .as_mut()
                .expect("retained successor group")
                .previous
                .take();
            // Original Python finalizers and native deallocations run outside
            // every phase mutex, inside this group's actual preparation interval.
            drop(previous);
            self.private_group()?
                .as_mut()
                .expect("retained original group")
                .ready = true;
        }
        if !self
            .private_group()?
            .as_ref()
            .expect("retained original group")
            .ready
        {
            return Err(invalid(
                "unknown private admission cannot start a replacement interval or callback",
            ));
        }
        let returned = match returned {
            Some(returned) => returned,
            None => {
                if entered && !pending {
                    return Err(invalid(
                        "unknown private numeric callback cannot repeat its original invocation",
                    ));
                }
                let fresh = self.refresh_snapshot.bind(py).call0()?;
                let fresh = AuthoritySnapshot::parse(&ColdValue::read(
                    &fresh,
                    &mut (16 * 1024 * 1024),
                    0,
                )?)?;
                fresh.newer_than(&task.state()?.snapshot)?;
                check_learning_grant(&task, &self.grant_reference, &fresh)?;
                task.state()?.snapshot = fresh;
                let _scope = PrivateTaskCallbackScope::enter(
                    &self.private_execution_active,
                    &task,
                    &session,
                )?;
                self.private_group()?
                    .as_mut()
                    .expect("retained original group")
                    .callback_entered = true;
                let visible_work = {
                    let retained = self.private_group()?;
                    let group = retained.as_ref().expect("retained original group");
                    if group.preparation_report.is_none() {
                        group.preparation_work.as_ref()
                    } else if group.retirement_ready && !group.native_retired {
                        group.retirement_work.as_ref()
                    } else {
                        None
                    }
                    .map(|work| work.clone_ref(py))
                };
                if let Some(work) = visible_work {
                    work.borrow(py).active.store(true, Ordering::Release);
                    *session.active_cold_model_work.lock().map_err(|_| {
                        invalid("private cold callback custody mutex is poisoned")
                    })? = Some(work);
                }
                let callback = self.model_owner(py, PhaseModelOwner::Execute)?;
                let result = callback.bind(py).call1((
                    child.controller.clone_ref(py),
                    child.task_use.clone_ref(py),
                    child.parent.clone_ref(py),
                    child.model.clone_ref(py),
                    entries,
                ));
                let mut retained = self.private_group()?;
                let group = retained.as_mut().expect("retained original group");
                match result {
                    Ok(result) => {
                        group.callback_pending = false;
                        group.callback_result = Some(result.clone().unbind());
                        result.unbind()
                    }
                    Err(error) => {
                        group.callback_pending =
                            error.is_instance_of::<SemanticCompletedSegmentPending>(py);
                        if !group.callback_pending {
                            group.callback_error = Some(error.clone_ref(py));
                        }
                        return Err(error);
                    }
                }
            }
        };
        self.finish_private_group(py, &returned)
    }

    fn finish_private_group(&self, py: Python<'_>, returned: &Py<PyAny>) -> PyResult<()> {
        {
            let retained = self.private_group()?;
            let group = retained.as_ref().expect("retained original group");
            if group.records_completed == group.kinds.len() && group.records_released {
                return if group.budget_exceeded {
                    Err(invalid("private group exceeded an original step budget; preserve actual complete receipts and expenditure without moving budgets"))
                } else {
                    Ok(())
                };
            }
        }
        let returned = returned.bind(py);
        if !returned.is_exact_instance_of::<PyTuple>() || returned.cast::<PyTuple>()?.len() != 3 {
            return Err(invalid(
                "private numeric execution lost its original parent/model/receipt tuple",
            ));
        }
        let returned = returned.cast::<PyTuple>()?;
        let (restored, selected, count) = {
            let retained = self.private_group()?;
            let group = retained.as_ref().expect("retained original group");
            if !group.native_retired || group.preparation_report.is_none() {
                return Err(invalid(
                    "private numerical handoff precedes its original report or graph retirement",
                ));
            }
            (
                group.restored.clone_ref(py),
                group
                    .selected_parent
                    .as_ref()
                    .expect("known selected parent")
                    .clone_ref(py),
                group.kinds.len(),
            )
        };
        if returned.get_item(0)?.as_ptr() != selected.as_ptr() || returned.get_item(1)?.is_none() {
            return Err(invalid(
                "private execution changed its original selected parent/model handoff",
            ));
        }
        let rows = returned.get_item(2)?;
        if !rows.is_exact_instance_of::<PyTuple>() || rows.cast::<PyTuple>()?.len() != count {
            return Err(invalid(
                "private execution changed its original completed receipt roster",
            ));
        }
        let rows = rows.cast::<PyTuple>()?;
        let child = restored.borrow(py);
        let session = child.session.borrow(py);
        let owner = session.owner()?;
        let mut receipts: Vec<Py<PySemanticCompletedExecutionObservation>> =
            Vec::with_capacity(count);
        let retained = self.private_group()?;
        let group = retained.as_ref().expect("retained original group");
        for (index, row) in rows.iter().enumerate() {
            if !row.is_exact_instance_of::<PyTuple>() || row.cast::<PyTuple>()?.len() != 2 {
                return Err(invalid(
                    "private execution requires each original step and issued native receipt",
                ));
            }
            let row = row.cast::<PyTuple>()?;
            let step = row.get_item(0)?;
            if !step.is_exact_instance_of::<PyDict>() || step.cast::<PyDict>()?.len() != 9 {
                return Err(invalid(
                    "private history requires the complete original Worker step",
                ));
            }
            let step = step.cast::<PyDict>()?;
            for name in [
                "step_ordinal",
                "transition",
                "skipped",
                "refusal",
                "bank",
                "completed_model_binding",
                "objective",
                "update_measurements",
                "execution_measurements",
            ] {
                if step.get_item(name)?.is_none() {
                    return Err(invalid("private history changed an original Worker field"));
                }
            }
            let ordinal = step
                .get_item("step_ordinal")?
                .ok_or_else(|| invalid("private step lost its original ordinal"))?;
            if ColdValue::read(&ordinal, &mut 128, 0)?.unsigned()? != index as u64 {
                return Err(invalid(
                    "private receipt roster changed its original step order",
                ));
            }
            let receipt = row
                .get_item(1)?
                .extract::<Py<PySemanticCompletedExecutionObservation>>()?;
            let observation = receipt.borrow(py);
            if !observation.inner.belongs_to(&owner)
                || !observation
                    .inner
                    .belongs_to_step(&group.native_steps[index])
                || observation.inner.requested_transition() != group.kinds[index]
                || receipts
                    .iter()
                    .any(|original| original.borrow(py).inner.same_execution(&observation.inner))
            {
                return Err(invalid("private history changed, duplicated or reordered its actual native execution receipts"));
            }
            let (transition, skipped, bank, refusal) = match observation.inner.outcome() {
                xlog_cuda::SemanticPreparedStepOutcome::Skipped { .. } => (None, true, None, None),
                xlog_cuda::SemanticPreparedStepOutcome::Completed {
                    transition,
                    bank,
                    outcome,
                    ..
                } => {
                    let transition = match transition {
                        SemanticTransitionKind::Proposal => "proposal",
                        SemanticTransitionKind::Update => "update",
                        SemanticTransitionKind::Recompute => "recompute",
                        SemanticTransitionKind::Drain => "drain",
                    };
                    (
                        Some(transition),
                        false,
                        Some(u64::from(*bank)),
                        transition_refusal_report(py, outcome)?,
                    )
                }
            };
            let expected_transition =
                transition.map_or(ColdValue::None, |value| ColdValue::Text(value.to_owned()));
            let expected_bank = bank.map_or(ColdValue::None, |value| {
                ColdValue::Integer(value.to_string())
            });
            if ColdValue::read(
                &step.get_item("transition")?.expect("original field"),
                &mut 128,
                0,
            )? != expected_transition
                || ColdValue::read(
                    &step.get_item("skipped")?.expect("original field"),
                    &mut 128,
                    0,
                )?
                .boolean()?
                    != skipped
                || ColdValue::read(
                    &step.get_item("bank")?.expect("original field"),
                    &mut 128,
                    0,
                )? != expected_bank
            {
                return Err(invalid(
                    "private step changed its native outcome or selected bank",
                ));
            }
            let reported_refusal = step.get_item("refusal")?.expect("original field");
            match refusal {
                None if reported_refusal.is_none() => {}
                Some(original) if reported_refusal.is_exact_instance_of::<PyDict>() => {
                    let reported = reported_refusal.cast::<PyDict>()?;
                    let original = original.bind(py);
                    if reported.len() != original.len() {
                        return Err(invalid("private step changed its complete native refusal"));
                    }
                    for (name, value) in original.iter() {
                        let reported = reported
                            .get_item(name)?
                            .ok_or_else(|| invalid("private step lost a native refusal field"))?;
                        if ColdValue::read(&reported, &mut (16 * 1024 * 1024), 0)?
                            != ColdValue::read(&value, &mut (16 * 1024 * 1024), 0)?
                        {
                            return Err(invalid(
                                "private step changed its original native refusal evidence",
                            ));
                        }
                    }
                }
                _ => return Err(invalid("private step replaced its original native refusal")),
            }
            let binding = step
                .get_item("completed_model_binding")?
                .expect("original field");
            if let Some((generation, geometry, numerical)) = observation.inner.model_binding() {
                if !binding.is_exact_instance_of::<PyDict>() || binding.cast::<PyDict>()?.len() != 5
                {
                    return Err(invalid(
                        "private step lost its complete acquired native model binding",
                    ));
                }
                let hex = |value: Identity256| {
                    value
                        .as_bytes()
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>()
                };
                let identity = |value: xlog_cuda::SemanticPublishedIdentity| {
                    ColdValue::Sequence(vec![
                        ColdValue::Text(hex(value.instance)),
                        ColdValue::Integer(value.word.to_string()),
                        ColdValue::Text(hex(value.logical_digest)),
                        ColdValue::Text(hex(value.state_digest)),
                    ])
                };
                let binding = binding.cast::<PyDict>()?;
                for (name, original) in [
                    (
                        "predecessor",
                        observation
                            .inner
                            .predecessor()
                            .map_or(ColdValue::None, identity),
                    ),
                    (
                        "successor",
                        observation
                            .inner
                            .successor()
                            .map_or(ColdValue::None, identity),
                    ),
                    (
                        "acquired_model_generation",
                        ColdValue::Integer(generation.to_string()),
                    ),
                    ("model_geometry_digest", ColdValue::Text(hex(geometry))),
                    ("model_numerical_digest", ColdValue::Text(hex(numerical))),
                ] {
                    let reported = binding.get_item(name)?.ok_or_else(|| {
                        invalid("private step lost an acquired model binding field")
                    })?;
                    if ColdValue::read(&reported, &mut (16 * 1024 * 1024), 0)? != original {
                        return Err(invalid("private step changed its native predecessor, successor or acquired model"));
                    }
                }
            } else if !binding.is_none() {
                return Err(invalid(
                    "private skipped step invented an acquired model binding",
                ));
            }
            let measured = step
                .get_item("execution_measurements")?
                .ok_or_else(|| invalid("private step lost its actual native measurements"))?;
            if !measured.is_exact_instance_of::<PyDict>() || measured.cast::<PyDict>()?.len() != 4 {
                return Err(invalid(
                    "private step changed its complete native measurement fields",
                ));
            }
            let measured = measured.cast::<PyDict>()?;
            let actual = observation.inner.quantities();
            for (name, original) in ["device_work", "model_work", "native_work", "model_calls"]
                .into_iter()
                .zip(actual)
            {
                let value = measured
                    .get_item(name)?
                    .ok_or_else(|| invalid("private step lost a native measurement"))?;
                if ColdValue::read(&value, &mut 128, 0)?.unsigned()? != original {
                    return Err(invalid(
                        "private step substituted its issued native expenditure",
                    ));
                }
            }
            drop(observation);
            receipts.push(receipt);
        }
        drop(retained);
        drop(owner);
        self.finish_private_cold_report(py, PrivateColdRole::Adoption)?;
        let finish = {
            let mut retained = self.private_group()?;
            let group = retained.as_mut().expect("retained original group");
            let finish = !group.observer_finish_entered;
            group.observer_finish_entered = true;
            finish
        };
        if finish {
            self.preparation_inputs.resource_observer.finish(py)?;
        }
        let peaks = self
            .preparation_inputs
            .resource_observer
            .step_physical_peaks(py, count)?;
        let peak = self
            .preparation_inputs
            .resource_observer
            .physical_peak(py)?;
        self.private_group()?
            .as_mut()
            .expect("retained original group")
            .physical_peak = Some(peak);
        let add = |a: u64, b: u64| {
            a.checked_add(b)
                .ok_or_else(|| invalid("private actual resource sum overflowed"))
        };
        let (instruction, ordinal, preparation, retirement, adoption, budgets, completed, entered) = {
            let retained = self.private_group()?;
            let group = retained.as_ref().expect("retained original group");
            let preparation = group
                .preparation_reports
                .iter()
                .copied()
                .chain(group.preparation_report)
                .try_fold((0u64, 0u64), |(work, calls), report| -> PyResult<_> {
                    Ok((
                        add(work, add(report.native_work, report.model_work)?)?,
                        add(calls, report.model_calls)?,
                    ))
                })?;
            (
                group.instruction.clone(),
                group.ordinal,
                preparation,
                group
                    .retirement_report
                    .expect("known original retirement expenditure"),
                group
                    .adoption_report
                    .expect("known original adoption expenditure"),
                group.budgets.clone(),
                group.records_completed,
                group.record_entered,
            )
        };
        if entered.is_some() {
            return Err(invalid(
                "unknown private history append cannot repeat its original scientific callback",
            ));
        }
        let tail = (
            add(
                add(retirement.native_work, retirement.model_work)?,
                add(adoption.native_work, adoption.model_work)?,
            )?,
            add(retirement.model_calls, adoption.model_calls)?,
        );
        let materials = feedback_materials(
            self.preparation_inputs
                .feedback_interventions
                .bind(py)
                .as_any(),
        )?;
        let mut exceeded = false;
        for index in 0..count {
            let receipt = receipts[index].borrow(py);
            let actual = receipt.inner.quantities();
            let mut work = actual[0];
            let mut calls = actual[3];
            if index == 0 {
                work = add(work, preparation.0)?;
                calls = add(calls, preparation.1)?;
            }
            if index + 1 == count {
                work = add(work, tail.0)?;
                calls = add(calls, tail.1)?;
            }
            exceeded |= work > budgets[index][0]
                || peaks[index] > budgets[index][1]
                || calls > budgets[index][2];
            if index < completed {
                continue;
            }
            let arguments = PyDict::new(py);
            arguments.set_item("instruction_bytes", PyBytes::new(py, &instruction))?;
            arguments.set_item(
                "operation_ordinal",
                ordinal
                    .checked_add(index as u64)
                    .ok_or_else(|| invalid("private history ordinal overflowed"))?,
            )?;
            let branch = self
                .private_group()?
                .as_ref()
                .expect("original private group")
                .branch;
            arguments.set_item("branch", branch)?;
            arguments.set_item(
                "step",
                rows.get_item(index)?.cast::<PyTuple>()?.get_item(0)?,
            )?;
            let intervention = match branch {
                "real" => materials[0].clone(),
                "control" => materials[1].clone(),
                _ => {
                    return Err(invalid(
                        "private history lost its original branch intervention",
                    ))
                }
            };
            arguments.set_item("feedback_intervention_bytes", intervention)?;
            arguments.set_item("resource_usage", (work, peaks[index], calls))?;
            let callback = self.scientific_owner.bind(py).getattr("record_step")?;
            self.private_group()?
                .as_mut()
                .expect("retained original group")
                .record_entered = Some(index);
            callback.call((), Some(&arguments))?;
            let mut retained = self.private_group()?;
            let group = retained.as_mut().expect("retained original group");
            group.records_completed += 1;
            group.record_entered = None;
        }
        self.private_group()?
            .as_mut()
            .expect("retained original group")
            .budget_exceeded = exceeded;
        self.preparation_inputs.resource_observer.release(py)?;
        self.private_group()?
            .as_mut()
            .expect("retained original group")
            .records_released = true;
        if exceeded {
            return Err(invalid("private group exceeded an original step budget; preserve actual complete receipts and expenditure without moving budgets"));
        }
        Ok(())
    }
}
