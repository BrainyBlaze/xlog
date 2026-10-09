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
    replay_children: Vec<Arc<PrivateReplayChildCustody>>,
    replay_model_backings: Arc<xlog_cuda::SemanticReplayModelBackings>,
    non_submission: Option<xlog_cuda::SemanticPreparedSegmentNonSubmission>,
    non_submission_confirmed: bool,
    non_submission_observer_entered: bool,
    selected_parent: Option<Py<PySemanticPublishedParent>>,
    native_retired: bool,
    retirement_work: Option<Py<PySemanticColdModelWork>>,
    retirement_custody: Option<SemanticColdNativeWork>,
    retirement_report: Option<SemanticColdModelWorkResult>,
    retirement_ready: bool,
    retirement_entered: bool,
    retirement_children_ready: bool,
    adoption_work: Option<Py<PySemanticColdModelWork>>,
    adoption_custody: Option<SemanticColdNativeWork>,
    adoption_report: Option<SemanticColdModelWorkResult>,
    callback_entered: bool,
    callback_pending: bool,
    callback_result: Option<Py<PyAny>>,
    callback_error: Option<PyErr>,
    observer_finish_entered: bool,
    backing_peak: Option<u64>,
    record_entered: Option<usize>,
    records_completed: usize,
    records_released: bool,
    budget_exceeded: bool,
}

/// One original replay construction, issued before the target steps exist.
/// The phase retains this custody even if construction or import never returns.
pub(crate) struct PrivateReplayChildCustody {
    phase: Mutex<Option<Py<PySemanticLearningPhaseTransition>>>,
    instruction: std::sync::Weak<SegmentInstructionCustody>,
    session: Py<PySemanticTransitionSession>,
    task: Py<PySemanticTransitionTaskUse>,
    group_ordinal: u64,
    step_index: usize,
    replay_ordinal: u64,
    model_backings: Arc<xlog_cuda::SemanticReplayModelBackings>,
    state: Mutex<PrivateReplayChildState>,
    failed: AtomicBool,
}

#[derive(Default)]
struct PrivateReplayChildState {
    construction_entered: bool,
    observed: bool,
    import_entered: bool,
    import_returned: bool,
    child: Option<Py<PySemanticTransitionSession>>,
    controller: Option<Arc<()>>,
    issued: Option<Py<PySemanticTransitionTaskUse>>,
    parent: Option<Py<PySemanticPublishedParent>>,
    member: Option<Py<PySemanticRetainedReplayMember>>,
    callbacks: Vec<Py<PyAny>>,
    native_work: Option<SemanticColdNativeWork>,
    retirement_entered: bool,
    retirement_attached: bool,
    released: bool,
    source: Option<Arc<VerifiedCheckpointSource>>,
}

impl PrivateReplayChildCustody {
    pub(in crate::semantic_transition) fn prepare_instruction(
        py: Python<'_>,
        instruction: &Arc<SegmentInstructionCustody>,
        phase: Option<Py<PySemanticLearningPhaseTransition>>,
        group_ordinal: u64,
        model_backings: Arc<xlog_cuda::SemanticReplayModelBackings>,
        step_index: usize,
        replay_ordinal: u64,
    ) -> PyResult<Arc<PrivateReplayChildCustody>> {
        instruction.require_owner(py)?;
        let work = {
            let state = instruction.preparation()?;
            if state.build_entered
                || state.result.is_some()
                || state.callback_failed
                || instruction.transitions.get(step_index) != Some(&SemanticTransitionKind::Update)
                || state.replay_children.iter().any(|child| {
                    child.step_index == step_index && child.replay_ordinal == replay_ordinal
                })
            {
                return Err(invalid(
                    "replay child must precede recording for one original Update and row",
                ));
            }
            state
                .work
                .as_ref()
                .map(|work| work.clone_ref(py))
                .ok_or_else(|| {
                    invalid("replay construction lost its original instruction callback")
                })?
        };
        work.borrow(py).check(py)?;
        let issued = instruction.task_use.borrow(py);
        let roster = issued.checkpoint.original_training_roster()?;
        let roster = roster.fields(2)?;
        let objective = read_training_objective(&roster[1])?
            .ok_or_else(|| invalid("replay requires the original frozen training objective"))?;
        let ordinal = usize::try_from(replay_ordinal)
            .map_err(|_| invalid("replay ordinal exceeds native address space"))?;
        let original_row = roster[0]
            .sequence()?
            .get(ordinal)
            .ok_or_else(|| invalid("replay row is absent from its original full roster"))?;
        let original_row = ReplayRow::parse_with_live(original_row, &issued.authority.live)?;
        if !matches!(original_row.basis, ReplayBasis::Episode { .. })
            || !objective.groups.iter().any(|group| {
                matches!(
                    group.kind,
                    SemanticTrainingObjectiveGroupKind::Edit
                        | SemanticTrainingObjectiveGroupKind::ActorCriticCost
                ) && group.row_ordinals.contains(&replay_ordinal)
            })
        {
            return Err(invalid(
                "replay row is not an original frozen Update member",
            ));
        }
        let child = Arc::new(PrivateReplayChildCustody {
            phase: Mutex::new(phase),
            instruction: Arc::downgrade(instruction),
            session: instruction.session.clone_ref(py),
            task: instruction.task_use.clone_ref(py),
            group_ordinal,
            step_index,
            replay_ordinal,
            model_backings,
            state: Mutex::new(PrivateReplayChildState::default()),
            failed: AtomicBool::new(false),
        });
        // The same instruction owns the child before any constructor or import.
        instruction
            .preparation()?
            .replay_children
            .push(Arc::clone(&child));
        Ok(child)
    }
}

#[pyclass(name = "SemanticPrivateReplayChild", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticPrivateReplayChild {
    pub(in crate::semantic_transition) inner: Arc<PrivateReplayChildCustody>,
}

#[pymethods]
impl PySemanticPrivateReplayChild {
    /// Project this selected row's original task, never the target task's factory.
    /// The content projection grants no authority; its ordinary import still
    /// validates current source grants and the complete frozen target roster.
    #[pyo3(signature = (*, refresh_snapshot))]
    fn source_task(
        &self,
        py: Python<'_>,
        refresh_snapshot: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.inner.require_preparation(py)?;
        if !refresh_snapshot.is_callable() {
            return Err(invalid(
                "original replay source requires current authority refresh",
            ));
        }
        let target = self.inner.task.borrow(py);
        let mut snapshot = target.state()?.snapshot.clone();
        let refreshed = refresh_checkpoint_authority(
            &target.authority,
            &mut snapshot,
            refresh_snapshot,
            "training",
            true,
        );
        target.state()?.snapshot = snapshot;
        refreshed?;
        if self.inner.state()?.source.is_none() {
            let roster = target.checkpoint.original_training_roster()?;
            let roster = roster.fields(2)?;
            let ordinal = usize::try_from(self.inner.replay_ordinal)
                .map_err(|_| invalid("original replay source ordinal exceeds this host"))?;
            let row = roster[0]
                .sequence()?
                .get(ordinal)
                .ok_or_else(|| invalid("original replay source row is absent"))?;
            let row = ReplayRow::parse_with_live(row, &target.authority.live)?;
            let referent = row
                .checkpoint_referent()?
                .or(row.pre_action_checkpoint_referent()?)
                .or(row.recovered_prefill_referent()?)
                .ok_or_else(|| {
                    invalid("original replay source has no complete checkpoint referent")
                })?;
            let checkpoint = target
                .checkpoint
                .checkpoint_sources
                .lock()
                .map_err(|_| invalid("checkpoint source owner mutex is poisoned"))?
                .verified
                .get(&referent.checkpoint_digest)
                .map(Arc::clone)
                .ok_or_else(|| {
                    invalid("original replay source was not retained by canonical admission")
                })?;
            referent.verify_source(&checkpoint)?;
            if checkpoint.cold.is_none() {
                return Err(invalid(
                    "original replay source has no canonical executable task admission",
                ));
            }
            self.inner.state()?.source = Some(checkpoint);
        }
        let state = self.inner.state()?;
        let source = state.source.as_ref().expect("retained original source");
        let cold = source
            .cold
            .as_ref()
            .expect("checked canonical cold task source");
        let cold = (
            cold.initial_theory.as_str(),
            cold.input_facts.as_str(),
            cold.observer_source.as_deref(),
            PyTuple::new(py, &cold.statements)?,
            PyTuple::new(py, &cold.query_records)?,
        )
            .into_pyobject(py)?
            .unbind()
            .into_any();
        let source_fields = ColdValue::Sequence(vec![
            ColdValue::Sequence(source.seed.authority.clone()),
            ColdValue::Sequence(source.seed.evaluation.clone()),
            source.seed.training_domain.clone(),
            source.seed.initial_sources.clone(),
            source.seed.source_mapping.clone(),
            source.seed.replay_capacity.clone(),
        ])
        .python_value(py)?;
        Ok(PyTuple::new(
            py,
            [
                PyBytes::new(py, &source.bytes).unbind().into_any(),
                cold,
                source_fields,
            ],
        )?
        .unbind())
    }
}

impl PrivateReplayChildCustody {
    pub(in crate::semantic_transition) fn source_checkpoint(
        &self,
        py: Python<'_>,
    ) -> PyResult<Arc<VerifiedCheckpointSource>> {
        self.require_preparation(py)?;
        self.state()?
            .source
            .as_ref()
            .map(Arc::clone)
            .ok_or_else(|| {
                invalid("private replay requires its retained original source checkpoint")
            })
    }

    pub(in crate::semantic_transition) fn shared_import_owners(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Arc<Mutex<ProposalExpense>>, Arc<Mutex<CheckpointSources>>)> {
        self.require_preparation(py)?;
        let target = self.task.borrow(py);
        Ok((
            Arc::clone(&target.checkpoint.proposal_expense),
            Arc::clone(&target.checkpoint.checkpoint_sources),
        ))
    }

    pub(in crate::semantic_transition) fn original_replay_row(
        &self,
        py: Python<'_>,
    ) -> PyResult<ReplayRow> {
        let target = self.task.borrow(py);
        let roster = target.checkpoint.original_training_roster()?;
        let ordinal = usize::try_from(self.replay_ordinal)
            .map_err(|_| invalid("private replay ordinal exceeds native address space"))?;
        let row = roster.fields(2)?[0]
            .sequence()?
            .get(ordinal)
            .ok_or_else(|| invalid("private replay lost its original target row"))?;
        ReplayRow::parse_with_live(row, &target.authority.live)
    }

    pub(in crate::semantic_transition) fn model_backings(
        &self,
        py: Python<'_>,
    ) -> PyResult<&xlog_cuda::SemanticReplayModelBackings> {
        self.require_preparation(py)?;
        Ok(&self.model_backings)
    }

    pub(in crate::semantic_transition) fn require_access(&self, py: Python<'_>) -> PyResult<()> {
        if self
            .phase
            .lock()
            .map_err(|_| invalid("original replay phase custody is poisoned"))?
            .is_none()
        {
            self.require_preparation(py)?;
            if self.state()?.released {
                return Err(invalid("original replay child has retired"));
            }
            return Ok(());
        }
        let phase = self.phase(py)?;
        let phase = phase.borrow(py);
        let state = self.state()?;
        if self.failed.load(Ordering::Acquire) || state.released {
            return Err(invalid(
                "unknown or retired private replay grants no ordinary native access",
            ));
        }
        let retirement = state.retirement_entered;
        drop(state);
        let retained = phase.private_group()?;
        let group = retained
            .as_ref()
            .ok_or_else(|| invalid("private replay lost its original group"))?;
        // Completed members finish only after their target step has retired.
        // Keep their original callback access until that same retirement work
        // closes; cancelled members still finish before target storage release.
        let completed_final_use = retirement
            && group.native_retired
            && group.non_submission.is_none()
            && group.retirement_report.is_none()
            && phase.private_execution_active.load(Ordering::Acquire);
        if group.ordinal != self.group_ordinal
            || (group.native_retired && !completed_final_use)
            || !group
                .replay_children
                .iter()
                .any(|original| std::ptr::eq(self, original.as_ref()))
            || !(phase.private_execution_active.load(Ordering::Acquire)
                || (retirement
                    && phase.phase_evaluation_active.load(Ordering::Acquire)
                    && self.session.borrow(py).retiring.load(Ordering::Acquire)))
        {
            return Err(invalid(
                "private replay native access requires its original active callback",
            ));
        }
        Ok(())
    }

    fn state(&self) -> PyResult<MutexGuard<'_, PrivateReplayChildState>> {
        self.state
            .lock()
            .map_err(|_| invalid("original private replay child custody is poisoned"))
    }

    fn phase(&self, py: Python<'_>) -> PyResult<Py<PySemanticLearningPhaseTransition>> {
        self.phase
            .lock()
            .map_err(|_| invalid("original private replay phase custody is poisoned"))?
            .as_ref()
            .map(|phase| phase.clone_ref(py))
            .ok_or_else(|| invalid("original private replay child has already retired"))
    }

    pub(in crate::semantic_transition) fn require_preparation(
        &self,
        py: Python<'_>,
    ) -> PyResult<()> {
        if self.failed.load(Ordering::Acquire) {
            return Err(invalid(
                "unknown private replay construction or import cannot be retried",
            ));
        }
        let instruction = self
            .instruction
            .upgrade()
            .ok_or_else(|| invalid("replay lost its original instruction owner"))?;
        instruction.require_owner(py)?;
        let original = instruction.preparation()?;
        if !original
            .replay_children
            .iter()
            .any(|child| std::ptr::eq(self, child.as_ref()))
        {
            return Err(invalid(
                "replay changed its original instruction membership",
            ));
        }
        if self
            .phase
            .lock()
            .map_err(|_| invalid("original replay phase custody is poisoned"))?
            .is_none()
        {
            let session = self.session.borrow(py);
            let active_cold = session
                .active_cold_model_work
                .lock()
                .map_err(|_| invalid("original replay callback custody is poisoned"))?;
            let early = !original.handoff_entered
                && original.result.is_none()
                && !original.callback_failed
                && active_cold
                    .as_ref()
                    .zip(original.work.as_ref())
                    .is_some_and(|(active, retained)| active.as_ptr() == retained.as_ptr());
            let prepared = session
                .prepared_segment
                .lock()
                .map_err(|_| invalid("prepared segment mutex is poisoned"))?;
            let original_scope = prepared.as_ref().is_some_and(|prepared| {
                original
                    .scope
                    .as_ref()
                    .is_some_and(|scope| Arc::ptr_eq(scope, &prepared.scope))
            });
            if !early
                && !(original_scope
                    && (session.recording.load(Ordering::Acquire)
                        || session.retiring.load(Ordering::Acquire)))
            {
                return Err(invalid(
                    "replay access requires its original early or prepared callback",
                ));
            }
            return Ok(());
        }
        drop(original);
        let phase = self.phase(py)?;
        let phase = phase.borrow(py);
        phase.require_private_group_owner(py, &self.session.borrow(py), &self.task.borrow(py))?;
        let retained = phase.private_group()?;
        let group = retained.as_ref().expect("checked original private group");
        if group.ordinal != self.group_ordinal
            || group.native_retired
            || !group
                .replay_children
                .iter()
                .any(|original| std::ptr::eq(self, original.as_ref()))
        {
            return Err(invalid("private replay changed its original phase group"));
        }
        Ok(())
    }

    pub(in crate::semantic_transition) fn allocation(
        &self,
        py: Python<'_>,
    ) -> PyResult<CheckpointAllocationDomain> {
        self.require_preparation(py)?;
        let work = if self
            .phase
            .lock()
            .map_err(|_| invalid("original replay phase custody is poisoned"))?
            .is_none()
        {
            let instruction = self
                .instruction
                .upgrade()
                .ok_or_else(|| invalid("replay lost its original instruction owner"))?;
            let original = instruction.preparation()?;
            if original.build_entered || original.result.is_some() {
                return Err(invalid(
                    "replay allocation must precede original target recording",
                ));
            }
            let work = original
                .work
                .as_ref()
                .ok_or_else(|| invalid("replay lost its original cold work owner"))?
                .borrow(py);
            self.session
                .borrow(py)
                .owner()?
                .share_cold_native_work(&work.inner)
                .map_err(xlog_err)?
        } else {
            let phase = self.phase(py)?;
            let phase = phase.borrow(py);
            let retained = phase.private_group()?;
            let group = retained.as_ref().expect("checked original private group");
            if group.build_entered || group.preparation_detached {
                return Err(invalid(
                    "private replay construction must precede original target recording",
                ));
            }
            group.preparation_custody.as_ref().cloned().ok_or_else(|| {
                invalid("private replay requires its original shared native work owner")
            })?
        };
        {
            let mut state = self.state()?;
            if state.construction_entered {
                return Err(invalid("private replay construction cannot repeat"));
            }
            state.construction_entered = true;
            state.native_work = Some(work.clone());
        }
        let (provider, domain) = self
            .session
            .borrow(py)
            .owner()?
            .checkpoint_allocation_domain()
            .map_err(xlog_err)?;
        Ok(CheckpointAllocationDomain {
            provider,
            domain,
            cold_work: Some(work),
        })
    }

    pub(in crate::semantic_transition) fn retain_session(
        self: &Arc<Self>,
        py: Python<'_>,
        session: &Py<PySemanticTransitionSession>,
    ) -> PyResult<()> {
        self.require_preparation(py)?;
        let mut state = self.state()?;
        if !state.construction_entered || state.child.is_some() {
            return Err(invalid(
                "private replay replaced its original child Session",
            ));
        }
        *session
            .borrow(py)
            .private_replay_child
            .lock()
            .map_err(|_| invalid("private replay Session custody is poisoned"))? =
            Some(Arc::clone(self));
        state.child = Some(session.clone_ref(py));
        Ok(())
    }

    pub(in crate::semantic_transition) fn observed(&self) -> PyResult<()> {
        let mut state = self.state()?;
        if state.child.is_none() || state.observed {
            return Err(invalid("private replay observer lost its original Session"));
        }
        state.observed = true;
        Ok(())
    }

    pub(in crate::semantic_transition) fn retain_controller(
        &self,
        controller: &Arc<()>,
    ) -> PyResult<()> {
        let mut state = self.state()?;
        if !state.observed || state.controller.is_some() {
            return Err(invalid(
                "private replay replaced its original cold Controller",
            ));
        }
        state.controller = Some(Arc::clone(controller));
        Ok(())
    }

    pub(in crate::semantic_transition) fn require_controller(
        &self,
        controller: &Arc<()>,
    ) -> PyResult<()> {
        if self
            .state()?
            .controller
            .as_ref()
            .is_none_or(|original| !Arc::ptr_eq(original, controller))
        {
            return Err(invalid(
                "private replay import requires its original cold Controller",
            ));
        }
        Ok(())
    }

    pub(in crate::semantic_transition) fn begin_import(&self, py: Python<'_>) -> PyResult<()> {
        self.require_preparation(py)?;
        let mut state = self.state()?;
        if !state.observed || state.import_entered {
            return Err(invalid(
                "private replay import requires its original observed child exactly once",
            ));
        }
        state.import_entered = true;
        Ok(())
    }

    pub(in crate::semantic_transition) fn retain_callbacks(
        &self,
        callbacks: Vec<Py<PyAny>>,
    ) -> PyResult<()> {
        let mut state = self.state()?;
        if !state.import_entered || !state.callbacks.is_empty() {
            return Err(invalid(
                "private replay replaced its original import callbacks",
            ));
        }
        state.callbacks = callbacks;
        Ok(())
    }

    pub(in crate::semantic_transition) fn retain_import_task(
        &self,
        py: Python<'_>,
        task: &Py<PySemanticTransitionTaskUse>,
    ) -> PyResult<()> {
        let mut state = self.state()?;
        if !state.import_entered || state.issued.is_some() {
            return Err(invalid(
                "private replay replaced its original imported task",
            ));
        }
        state.issued = Some(task.clone_ref(py));
        Ok(())
    }

    pub(in crate::semantic_transition) fn retain_parent(
        &self,
        py: Python<'_>,
        parent: &Py<PySemanticPublishedParent>,
    ) -> PyResult<()> {
        let mut state = self.state()?;
        if state.issued.is_none() || state.parent.is_some() {
            return Err(invalid(
                "private replay replaced its original import reader",
            ));
        }
        state.parent = Some(parent.clone_ref(py));
        Ok(())
    }

    pub(in crate::semantic_transition) fn retain_member(
        &self,
        py: Python<'_>,
        member: &Py<PySemanticRetainedReplayMember>,
    ) -> PyResult<()> {
        self.require_preparation(py)?;
        let mut state = self.state()?;
        if state.member.is_some()
            || state.parent.is_none()
            || member.borrow(py).replay_ordinal != self.replay_ordinal
        {
            return Err(invalid(
                "private replay replaced its original retained member",
            ));
        }
        state.member = Some(member.clone_ref(py));
        state.import_returned = true;
        Ok(())
    }

    pub(in crate::semantic_transition) fn fail(&self) {
        self.failed.store(true, Ordering::Release);
    }

    pub(in crate::semantic_transition) fn require_projection(
        &self,
        py: Python<'_>,
        selection: &ColdValue,
        objective: &ColdValue,
        source_authority: &[ColdValue],
    ) -> PyResult<(usize, NativeReplayBinding)> {
        self.require_preparation(py)?;
        let source = self.source_checkpoint(py)?;
        self.refresh_current_authority(py)?;
        let task = self.task.borrow(py);
        let roster = task.checkpoint.original_training_roster()?;
        let roster = roster.fields(2)?;
        let ordinal = usize::try_from(self.replay_ordinal)
            .map_err(|_| invalid("private replay ordinal exceeds native address space"))?;
        let selected = selection.fields(2)?;
        let row = self.original_replay_row(py)?;
        if selected[0].unsigned()? != self.replay_ordinal
            || selected[1] != row.identity
            || *objective != roster[1]
            || source_authority != source.seed.authority.as_slice()
        {
            return Err(invalid(
                "private replay changed its source authority or original target objective and row",
            ));
        }
        Ok((ordinal, row.native_replay()?))
    }

    pub(in crate::semantic_transition) fn require_current_authority(
        &self,
        py: Python<'_>,
    ) -> PyResult<()> {
        let task = self.task.borrow(py);
        let state = task.state()?;
        task.authority.check_use("training", &state.snapshot, true)
    }

    pub(in crate::semantic_transition) fn refresh_current_authority(
        &self,
        py: Python<'_>,
    ) -> PyResult<()> {
        if self
            .phase
            .lock()
            .map_err(|_| invalid("original replay phase custody is poisoned"))?
            .is_none()
        {
            // Ordinary preparation uses the same current task snapshot admitted
            // by before_segment; it has no private phase refresh callback.
            return self.require_current_authority(py);
        }
        let phase = self.phase(py)?;
        let refresh = phase.borrow(py).refresh_snapshot.clone_ref(py);
        let task = self.task.borrow(py);
        let mut snapshot = task.state()?.snapshot.clone();
        let refreshed = refresh_checkpoint_authority(
            &task.authority,
            &mut snapshot,
            refresh.bind(py),
            "training",
            true,
        );
        task.state()?.snapshot = snapshot;
        refreshed
    }

    pub(in crate::semantic_transition) fn require_target(
        &self,
        py: Python<'_>,
        target: &PySemanticPreparedStep,
    ) -> PyResult<()> {
        self.require_preparation(py)?;
        let instruction = self
            .instruction
            .upgrade()
            .ok_or_else(|| invalid("replay lost its original instruction owner"))?;
        let original = instruction.preparation()?;
        if target.session.as_ptr() != self.session.as_ptr()
            || target.task_use.as_ptr() != self.task.as_ptr()
            || original
                .scope
                .as_ref()
                .is_none_or(|scope| !Arc::ptr_eq(scope, &target.scope))
            || original
                .native_steps
                .get(self.step_index)
                .is_none_or(|step| !step.same_handle(&target.inner))
        {
            return Err(invalid(
                "private replay delivery changed its original frozen Update",
            ));
        }
        Ok(())
    }

    pub(in crate::semantic_transition) fn detach_preparation_work(
        &self,
        py: Python<'_>,
    ) -> PyResult<()> {
        let (child, work) = {
            let state = self.state()?;
            if state.released {
                return Ok(());
            }
            if self.failed.load(Ordering::Acquire) || !state.import_returned {
                return Err(invalid(
                    "unknown private replay import retains its original native work custody",
                ));
            }
            (
                state.child.as_ref().expect("returned child").clone_ref(py),
                state.native_work.clone(),
            )
        };
        if let Some(work) = work {
            child
                .borrow(py)
                .owner()?
                .detach_shared_cold_native_work(&work)
                .map_err(xlog_err)?;
            let original = self.state()?.native_work.take();
            drop(original);
        }
        Ok(())
    }

    pub(in crate::semantic_transition) fn attach_retirement_work(
        &self,
        py: Python<'_>,
        work: SemanticColdNativeWork,
    ) -> PyResult<()> {
        let (child, parent, work) = {
            let mut state = self.state()?;
            if self.failed.load(Ordering::Acquire)
                || !state.import_returned
                || state.released
            {
                return Err(invalid(
                    "private replay retirement requires its original known import exactly once",
                ));
            }
            let work = if state.retirement_entered {
                let original = state.native_work.as_ref().ok_or_else(|| {
                    invalid("private replay retirement lost its original shared custody")
                })?;
                if !original.same_custody(&work) {
                    return Err(invalid(
                        "private replay retirement changed its original shared custody",
                    ));
                }
                if state.retirement_attached {
                    return Ok(());
                }
                original.clone()
            } else {
                if state.native_work.is_some() {
                    return Err(invalid(
                        "private replay still retains its preparation tally",
                    ));
                }
                state.retirement_entered = true;
                state.native_work = Some(work.clone());
                work
            };
            (
                state.child.as_ref().expect("returned child").clone_ref(py),
                state
                    .parent
                    .as_ref()
                    .expect("returned reader")
                    .clone_ref(py),
                work,
            )
        };
        // This native attachment is metadata-only. Its checked graph setter
        // has no fallible suffix after installing the original borrowed owner.
        child
            .borrow(py)
            .owner()?
            .attach_shared_cold_native_work(&*parent.borrow(py).lease()?, work)
            .map_err(xlog_err)?;
        self.state()?.retirement_attached = true;
        Ok(())
    }

    pub(in crate::semantic_transition) fn require_cancelled_final_use(
        &self,
        py: Python<'_>,
        member: &PySemanticRetainedReplayMember,
    ) -> PyResult<()> {
        let state = self.state()?;
        if self.failed.load(Ordering::Acquire)
            || !state.import_returned
            || !state.retirement_entered
            || state.released
            || state
                .member
                .as_ref()
                .is_none_or(|original| !std::ptr::eq(&*original.borrow(py), member))
        {
            return Err(invalid(
                "cancelled replay final use requires its original active native retirement",
            ));
        }
        drop(state);
        let session = self.session.borrow(py);
        if !session.retiring.load(Ordering::Acquire) {
            return Err(invalid(
                "cancelled replay final use is outside original graph retirement",
            ));
        }
        let phase = self.phase
            .lock()
            .map_err(|_| invalid("original replay phase custody is poisoned"))?
            .as_ref().map(|phase| phase.clone_ref(py));
        if phase.is_none() {
            let instruction = self.instruction.upgrade()
                .ok_or_else(|| invalid("cancelled replay lost its original instruction"))?;
            instruction.require_owner(py)?;
            if instruction.session.as_ptr() != self.session.as_ptr()
                || instruction.task_use.as_ptr() != self.task.as_ptr()
                || !instruction.preparation()?.replay_children.iter()
                    .any(|original| std::ptr::eq(self, original.as_ref()))
            {
                return Err(invalid("cancelled replay changed its original instruction membership"));
            }
            let (target, graph_retired) = {
                let retained = session.prepared_segment.lock()
                    .map_err(|_| invalid("prepared segment mutex is poisoned"))?;
                let retained = retained.as_ref()
                    .ok_or_else(|| invalid("cancelled replay lost its original segment"))?;
                if retained.instruction.as_ref().is_none_or(|original| !Arc::ptr_eq(original, &instruction))
                    || retained.task_use.as_ptr() != self.task.as_ptr()
                {
                    return Err(invalid("cancelled replay changed its original prepared segment"));
                }
                (retained.steps.get(self.step_index)
                    .ok_or_else(|| invalid("cancelled replay lost its original Update"))?
                    .borrow(py).inner.clone(), retained.graph_retired)
            };
            if !graph_retired {
                return Err(invalid("cancelled replay requires actual original graph destruction"));
            }
            let controller = PySemanticTransitionController {
                session: self.session.clone_ref(py),
                identity: Arc::clone(&self.task.borrow(py).controller),
            };
            let proof = controller.original_segment_cancellation(py, &session, &self.task)?
                .ok_or_else(|| invalid("cancelled replay lost its original instruction non-submission proof"))?;
            return session.owner()?.require_cancelled_prepared_step_final_use(&target, &proof)
                .map_err(xlog_err);
        }
        let phase = phase.expect("original private replay phase");
        let phase = phase.borrow(py);
        if !phase.phase_evaluation_active.load(Ordering::Acquire) {
            return Err(invalid("cancelled replay requires its original active phase retirement"));
        }
        let retained = phase.private_group()?;
        let group = retained
            .as_ref()
            .ok_or_else(|| invalid("cancelled replay lost its original group"))?;
        if group.ordinal != self.group_ordinal
            || !group.non_submission_confirmed
            || !group.retirement_entered
            || group.native_retired
            || !group
                .replay_children
                .iter()
                .any(|original| std::ptr::eq(self, original.as_ref()))
        {
            return Err(invalid(
                "cancelled replay final use changed its original unsubmitted group",
            ));
        }
        let proof = group
            .non_submission
            .as_ref()
            .ok_or_else(|| invalid("cancelled replay lost its original native proof"))?;
        let target = group.native_steps.get(self.step_index)
            .ok_or_else(|| invalid("cancelled replay lost its original target Update"))?;
        let result = session.owner()?.require_cancelled_prepared_step_final_use(target, proof)
            .map_err(xlog_err);
        result
    }

    pub(in crate::semantic_transition) fn finish_retained_source(
        &self,
        py: Python<'_>,
        cancelled: bool,
    ) -> PyResult<()> {
        let member = {
            let state = self.state()?;
            if self.failed.load(Ordering::Acquire) || !state.import_returned
                || !state.retirement_entered || state.released
            {
                return Err(invalid("source final use requires its original attached retirement"));
            }
            state.member.as_ref()
                .ok_or_else(|| invalid("source retirement lost its original member"))?
                .clone_ref(py)
        };
        let member = member.borrow(py);
        if member.completed.load(Ordering::Acquire) {
            let mode = member.final_use.try_lock()
                .map_err(|_| invalid("original source final use is already active or poisoned"))?
                .mode.ok_or_else(|| invalid("original final-use acknowledgment has no disposition"))?;
            if (mode == RetainedFinalUseMode::Cancelled) != cancelled {
                return Err(invalid("original source final use changed its cancellation disposition"));
            }
            return member.require_final_use_ack(py, mode, None);
        }
        if cancelled {
            return member.finish_cancelled_private_segment(py);
        }
        let unpublished = {
            let state = member.group_use.lock()
                .map_err(|_| invalid("original retained target use is poisoned"))?;
            match state.completion {
                Some(RetainedTargetCompletion::Published) => None,
                Some(RetainedTargetCompletion::Unpublished) => Some(state.update.as_ref()
                    .ok_or_else(|| invalid("original unpublished target has no Update"))?
                    .clone_ref(py)),
                None => return Err(invalid("source final use requires its actual original target outcome")),
            }
        };
        member.finish_retained_import(py, unpublished)
    }

    pub(in crate::semantic_transition) fn release_session(&self, py: Python<'_>) -> PyResult<()> {
        let (child, parent) = {
            let state = self.state()?;
            if state.released {
                return Ok(());
            }
            if self.failed.load(Ordering::Acquire)
                || !state.retirement_entered
                || state
                    .member
                    .as_ref()
                    .is_none_or(|member| !member.borrow(py).completed.load(Ordering::Acquire))
            {
                return Err(invalid(
                    "private replay Session retains unfinished original final use",
                ));
            }
            (
                state.child.as_ref().expect("returned child").clone_ref(py),
                state
                    .parent
                    .as_ref()
                    .expect("returned reader")
                    .clone_ref(py),
            )
        };
        child
            .borrow(py)
            .release_retired_publication(py, &parent.borrow(py))?;
        let session_custody = child
            .borrow(py)
            .private_replay_child
            .lock()
            .map_err(|_| invalid("retired private replay Session custody is poisoned"))?
            .take();
        let original = std::mem::replace(
            &mut *self.state()?,
            PrivateReplayChildState {
                released: true,
                ..PrivateReplayChildState::default()
            },
        );
        let phase = self
            .phase
            .lock()
            .map_err(|_| invalid("retired private replay phase custody is poisoned"))?
            .take();
        // Original callbacks, member, model aliases and native tally leave only
        // here, outside every Session and custody lock in the same cold region.
        drop((session_custody, original, phase));
        Ok(())
    }
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
    fn replay_children(&self) -> PyResult<Vec<Arc<PrivateReplayChildCustody>>> {
        Ok(self
            .private_group()?
            .as_ref()
            .ok_or_else(|| invalid("private replay lost its original group"))?
            .replay_children
            .clone())
    }

    fn attach_replay_retirement_work(
        &self,
        py: Python<'_>,
        work: &PySemanticColdModelWork,
    ) -> PyResult<()> {
        let children = self.replay_children()?;
        if children.is_empty() {
            return Ok(());
        }
        work.check(py)?;
        let reader = work.reader.borrow(py);
        let source = reader.session.borrow(py);
        for child in children {
            let custody = source
                .owner()?
                .share_cold_native_work(&work.inner)
                .map_err(xlog_err)?;
            child.attach_retirement_work(py, custody)?;
        }
        Ok(())
    }

    fn release_replay_children(&self, py: Python<'_>) -> PyResult<()> {
        for child in self.replay_children()? {
            child.release_session(py)?;
        }
        Ok(())
    }

    pub(in crate::semantic_transition) fn prepare_original_replay_child(
        &self,
        py: Python<'_>,
        phase: &Py<Self>,
        session: &Py<PySemanticTransitionSession>,
        task: &Py<PySemanticTransitionTaskUse>,
        instruction: &Arc<SegmentInstructionCustody>,
        step_index: usize,
        replay_ordinal: u64,
    ) -> PyResult<Arc<PrivateReplayChildCustody>> {
        self.require_private_group_owner(py, &session.borrow(py), &task.borrow(py))?;
        let mut retained = self.private_group()?;
        let group = retained.as_mut().expect("checked original private group");
        if group.build_entered
            || group.preparation_detached
            || group.kinds.get(step_index) != Some(&SemanticTransitionKind::Update)
            || group.replay_children.iter().any(|child| {
                child.step_index == step_index && child.replay_ordinal == replay_ordinal
            })
        {
            return Err(invalid(
                "private replay must bind one frozen Update and replay row before target recording",
            ));
        }
        let ordinal = group.ordinal;
        let backings = Arc::clone(&group.replay_model_backings);
        drop(retained);
        let child = PrivateReplayChildCustody::prepare_instruction(
            py,
            instruction,
            Some(phase.clone_ref(py)),
            ordinal,
            backings,
            step_index,
            replay_ordinal,
        )?;
        self.private_group()?
            .as_mut()
            .expect("retained original group")
            .replay_children
            .push(Arc::clone(&child));
        Ok(child)
    }

    pub(super) fn private_segment_terminal_input(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<super::phase_evaluation::SegmentTerminalInput>> {
        let retained = self.private_group()?;
        let Some(group) = retained
            .as_ref()
            .filter(|group| group.budget_exceeded || group.non_submission_confirmed)
        else {
            return Ok(None);
        };
        if group.non_submission_confirmed {
            Self::require_private_group_entries(py, group)?;
            let proof = group
                .non_submission
                .as_ref()
                .filter(|proof| proof.matches(&group.native_steps))
                .ok_or_else(|| {
                    invalid("cancelled group lost its genuine native whole-roster proof")
                })?;
            let scope = group
                .build_scope
                .as_ref()
                .ok_or_else(|| invalid("cancelled group lost its original construction scope"))?;
            let restored = group.restored.borrow(py);
            let task = restored.task_use.borrow(py);
            if group.native_retired
                || group.callback_error.is_none()
                || group.callback_pending
                || group.callback_result.is_some()
                || group.observer_finish_entered
                || group.record_entered.is_some()
                || group.records_completed != 0
                || group.records_released
                || group.preparation_report.is_none()
                || group.selected_parent.is_some()
                || !matches!(&task.state()?.phase,
                    TaskUsePhase::ArenaPreparing(original) if matches!(original.as_ref(),
                        TaskUsePhase::CancelledPrivate { scope: original, .. } if Arc::ptr_eq(original, scope)))
            {
                return Err(invalid("cancelled terminal requires the unchanged unsubmitted group, failed callback and private task"));
            }
            let preparation = group
                .preparation_reports
                .iter()
                .copied()
                .chain(group.preparation_report)
                .try_fold([0u64; 2], |[work, calls], report| -> PyResult<_> {
                    Ok([
                        report
                            .model_work
                            .checked_add(report.native_work)
                            .and_then(|value| work.checked_add(value))
                            .ok_or_else(|| invalid("cancelled preparation work overflowed"))?,
                        calls
                            .checked_add(report.model_calls)
                            .ok_or_else(|| invalid("cancelled preparation calls overflowed"))?,
                    ])
                })?;
            let parent = restored.parent.borrow(py);
            let identity = original_instruction_parent_identity(py, &parent)?;
            let count = u64::try_from(group.kinds.len())
                .map_err(|_| invalid("cancelled original roster extent overflowed"))?;
            return Ok(Some(super::phase_evaluation::SegmentTerminalInput {
                owners: super::phase_evaluation::EvaluationOwners {
                    controller: restored.controller.clone_ref(py),
                    task: restored.task_use.clone_ref(py),
                    parent: restored.parent.clone_ref(py),
                    model: restored.model.clone_ref(py),
                },
                branch: group.branch,
                instruction: group.instruction.clone(),
                group: super::phase_evaluation::SegmentTerminalGroup {
                    group_ordinal: group.ordinal,
                    count,
                    parent: identity,
                },
                cancellation: Some(super::phase_evaluation::CancelledSegmentTerminal {
                    group: super::phase_evaluation::SegmentTerminalGroup {
                        group_ordinal: group.ordinal,
                        count,
                        parent: identity,
                    },
                    proof: proof.clone(),
                    scope: Arc::clone(scope),
                    preparation,
                }),
            }));
        }
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
            || group.backing_peak.is_none()
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
        let identity = original_instruction_parent_identity(py, &original_parent)?;
        Ok(Some(super::phase_evaluation::SegmentTerminalInput {
            owners: super::phase_evaluation::EvaluationOwners {
                controller: restored.controller.clone_ref(py),
                task: restored.task_use.clone_ref(py),
                parent: parent.clone_ref(py),
                model: returned.get_item(1)?.unbind(),
            },
            branch: group.branch,
            instruction: group.instruction.clone(),
            group: super::phase_evaluation::SegmentTerminalGroup {
                group_ordinal: group.ordinal,
                count: u64::try_from(group.kinds.len())
                    .map_err(|_| invalid("terminal original group extent overflowed"))?,
                parent: identity,
            },
            cancellation: None,
        }))
    }

    pub(super) fn retire_original_cancelled_group(
        &self,
        py: Python<'_>,
        branch: &'static str,
        ordinal: u64,
        count: u64,
        proof: &xlog_cuda::SemanticPreparedSegmentNonSubmission,
        scope: &Arc<()>,
    ) -> PyResult<()> {
        let (restored, steps) = {
            let mut retained = self.private_group()?;
            let group = retained
                .as_mut()
                .ok_or_else(|| invalid("cancelled retirement lost its original group"))?;
            if group.branch != branch
                || group.ordinal != ordinal
                || group.kinds.len() as u64 != count
                || !group.non_submission_confirmed
                || !proof.matches(&group.native_steps)
                || group
                    .build_scope
                    .as_ref()
                    .is_none_or(|original| !Arc::ptr_eq(original, scope))
                || group.records_completed != 0
                || group.records_released
                || group.observer_finish_entered
                || group.selected_parent.is_some()
            {
                return Err(invalid(
                    "cancelled retirement changed its original unsubmitted roster",
                ));
            }
            if group.native_retired {
                return Ok(());
            }
            group.retirement_entered = true;
            (group.restored.clone_ref(py), group.native_steps.clone())
        };
        let restored = restored.borrow(py);
        let session = restored.session.borrow(py);
        session.require_creator()?;
        if session.importing.load(Ordering::Acquire)
            || session.recording.load(Ordering::Acquire)
            || session.retiring.swap(true, Ordering::AcqRel)
        {
            return Err(invalid(
                "cancelled native retirement overlaps another original operation",
            ));
        }
        let _retirement = PreparedRetirementGuard(&session.retiring);
        let streams = self.preparation_inputs.consumer_streams.python_value(py)?;
        let streams = checkpoint_consumer_streams(streams.bind(py), &mut (16 * 1024 * 1024))?;
        let (mut quiesced, graph_retired) = {
            let mut retained = session.prepared_segment.lock()
                .map_err(|_| invalid("cancelled Python segment custody is poisoned"))?;
            let retained = retained.as_mut()
                .ok_or_else(|| invalid("cancelled retirement lost its original Python segment"))?;
            if let Some(original) = &retained.retirement_streams {
                if original != &streams {
                    return Err(invalid("cancelled retirement changed its original consumer roster"));
                }
            } else {
                retained.retirement_streams = Some(streams.clone());
            }
            (retained.retirement_steps_quiesced, retained.graph_retired)
        };
        if !graph_retired {
            while quiesced < steps.len() {
                session.owner()?.quiesce_cancelled_prepared_step(&steps[quiesced], proof, &streams)
                    .map_err(xlog_err)?;
                quiesced += 1;
                session.prepared_segment.lock()
                    .map_err(|_| invalid("cancelled Python segment custody is poisoned"))?
                    .as_mut().ok_or_else(|| invalid("cancelled segment disappeared during final use"))?
                    .retirement_steps_quiesced = quiesced;
            }
            let graph = session.owner()?.take_cancelled_prepared_executable_for_retirement(proof)
                .map_err(xlog_err)?;
            drop(graph);
            session.prepared_segment.lock()
                .map_err(|_| invalid("cancelled Python segment custody is poisoned"))?
                .as_mut().ok_or_else(|| invalid("cancelled segment disappeared during graph retirement"))?
                .graph_retired = true;
        }
        session
            .owner()?
            .require_cancelled_prepared_graph_retirement(proof)
            .map_err(xlog_err)?;
        let work = session
            .active_cold_model_work
            .lock()
            .map_err(|_| invalid("cancelled replay lost its original cold retirement custody"))?
            .as_ref()
            .map(|work| work.clone_ref(py))
            .ok_or_else(|| {
                invalid("cancelled replay requires the original active cold retirement region")
            })?;
        if !self.private_group()?.as_ref().expect("original cancelled group").retirement_children_ready {
            self.attach_replay_retirement_work(py, &work.borrow(py))?;
            self.private_group()?.as_mut().expect("original cancelled group")
                .retirement_children_ready = true;
        }
        let children = self.private_group()?.as_ref().expect("original cancelled group")
            .replay_children.clone();
        let producer_state = {
            let retained = session.prepared_segment.lock()
                .map_err(|_| invalid("cancelled Python segment custody is poisoned"))?;
            let retained = retained.as_ref()
                .ok_or_else(|| invalid("cancelled retirement lost its original Python segment"))?;
            (retained.producer_retirement_entered, retained.producers_retired)
        };
        if !producer_state.1 {
            if producer_state.0 {
                return Err(invalid("original cancelled producer cleanup has no known return and cannot repeat"));
            }
            for child in children {
                child.finish_retained_source(py, true)?;
            }
        }
        let python_owners = {
            let retained = session
                .prepared_segment
                .lock()
                .map_err(|_| invalid("cancelled Python segment custody is poisoned"))?;
            let segment = retained.as_ref().ok_or_else(|| {
                invalid("cancelled retirement lost its original Python segment custody")
            })?;
            if segment.task_use.as_ptr() != restored.task_use.as_ptr()
                || !Arc::ptr_eq(&segment.scope, scope)
            {
                return Err(invalid(
                    "cancelled Python segment differs from its original native proof",
                ));
            }
            let mut after = 0;
            for step in &segment.steps {
                let step = step.borrow(py);
                let offset = steps[after..].iter().position(|original| original.same_handle(&step.inner))
                    .ok_or_else(|| invalid("cancelled retirement changed its remaining ordered steps"))?;
                after += offset + 1;
            }
            (
                segment
                    .steps
                    .iter()
                    .map(|step| step.clone_ref(py))
                    .collect::<Vec<_>>(),
                Arc::clone(&segment.resources),
            )
        };
        {
            let (steps, resources) = &python_owners;
            if !producer_state.1 {
                let prepared = resources.producers.lock()
                    .map_err(|_| invalid("cancelled producer custody is poisoned"))?
                    .get(1).map(|owner| owner.clone_ref(py))
                    .ok_or_else(|| invalid("cancelled retirement lost its original prepared owner"))?;
                let task = restored.task_use.borrow(py);
                let snapshot = task.state()?.snapshot.canonical.clone();
                let check = || -> PyResult<()> {
                    let state = task.state()?;
                    if state.snapshot.canonical != snapshot
                        || !matches!(&state.phase, TaskUsePhase::ArenaPreparing(original)
                            if matches!(original.as_ref(), TaskUsePhase::CancelledPrivate { scope: original, .. }
                                if Arc::ptr_eq(original, scope)))
                    {
                        return Err(invalid("cancelled producer retirement changed its original private task or authority"));
                    }
                    Ok(())
                };
                session.prepared_segment.lock()
                    .map_err(|_| invalid("cancelled Python segment custody is poisoned"))?
                    .as_mut().ok_or_else(|| invalid("cancelled producer owner disappeared"))?
                    .producer_retirement_entered = true;
                let finish =
                    recording_callback(check, || prepared.bind(py).getattr("finish_segment"))?;
                if !recording_callback(check, || finish.call0())?.is_none() {
                    return Err(invalid(
                        "cancelled finish_segment requires its original known None return",
                    ));
                }
                session
                    .prepared_segment
                    .lock()
                    .map_err(|_| invalid("cancelled Python segment custody is poisoned"))?
                    .as_mut()
                    .ok_or_else(|| {
                        invalid("cancelled prepared owner disappeared during finish_segment")
                    })?
                    .producers_retired = true;
            }
            for step in steps {
                step.borrow(py).release_recorded_producer_aliases(py)?;
            }
            let producers = std::mem::take(
                &mut *resources
                    .producers
                    .lock()
                    .map_err(|_| invalid("cancelled producer custody is poisoned"))?,
            );
            drop(producers);
        }
        drain_export_owners();
        self.release_replay_children(py)?;
        for step in &python_owners.0 {
            session
                .owner()?
                .release_cancelled_prepared_step(&step.borrow(py).inner, proof, &streams)
                .map_err(xlog_err)?;
            let removed = {
                let mut retained = session.prepared_segment.lock()
                    .map_err(|_| invalid("cancelled Python segment custody is poisoned"))?;
                let retained = retained.as_mut()
                    .ok_or_else(|| invalid("cancelled segment disappeared during storage retirement"))?;
                let index = retained.steps.iter().position(|original| original.as_ptr() == step.as_ptr())
                    .ok_or_else(|| invalid("cancelled storage changed its original step"))?;
                retained.steps.remove(index)
            };
            drop(removed);
        }
        let resources = session
            .owner()?
            .take_cancelled_prepared_resources_for_retirement(proof)
            .map_err(xlog_err)?;
        drop(resources);
        drain_export_owners();
        // Unknown retirement retains the original Python custody slot. Remove
        // it only after all native storage and consumer joins are known closed.
        let python_segment = session
            .prepared_segment
            .lock()
            .map_err(|_| invalid("cancelled Python segment custody is poisoned"))?
            .take();
        drop(python_segment);
        drop(python_owners);
        self.private_group()?
            .as_mut()
            .expect("retained cancelled group")
            .native_retired = true;
        Ok(())
    }

    pub(super) fn drop_completed_private_execution_owners(
        &self,
        branch: &'static str,
    ) -> PyResult<()> {
        let original = {
            let mut retained = self.private_group()?;
            let mut cursor = retained.as_ref();
            while let Some(group) = cursor {
                let cancelled = group.non_submission_confirmed
                    && self.cancelled_segment_terminal_matches(
                        branch,
                        group.ordinal,
                        group.kinds.len() as u64,
                    )?;
                if group.branch != branch
                    || !group.native_retired
                    || if cancelled {
                        group.records_completed != 0 || group.records_released
                    } else {
                        group.records_completed != group.kinds.len()
                            || !group.records_released
                            || group.callback_error.is_some()
                    }
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
        if group.preparation_report.is_none() {
            self.source
                .borrow(py)
                .owner()?
                .require_closed_cold_model_work(&work.inner)
                .map_err(xlog_err)?;
        }
        // This precedes nominal reservation, allocation and producer callbacks.
        group.build_entered = true;
        Ok(())
    }

    pub(in crate::semantic_transition) fn instruction_preparation_work(
        &self,
        py: Python<'_>,
        instruction: &SegmentInstructionCustody,
    ) -> PyResult<Py<PySemanticColdModelWork>> {
        self.require_private_group_owner(
            py,
            &instruction.session.borrow(py),
            &instruction.task_use.borrow(py),
        )?;
        let retained = self.private_group()?;
        let group = retained.as_ref().expect("retained original group");
        if group.instruction.as_slice() != instruction.instruction.as_ref()
            || group.build_entered
            || group.preparation_role != PrivateColdRole::Prefix
            || group.preparation_report.is_some()
        {
            return Err(invalid(
                "instruction preparation changed its original private prefix",
            ));
        }
        let work = group
            .preparation_work
            .as_ref()
            .ok_or_else(|| invalid("original private prefix lost its work owner"))?
            .clone_ref(py);
        work.borrow(py).check(py)?;
        Ok(work)
    }

    pub(in crate::semantic_transition) fn finish_instruction_preparation(
        &self,
        py: Python<'_>,
    ) -> PyResult<SemanticColdModelWorkResult> {
        self.finish_private_cold_report(py, PrivateColdRole::Prefix)
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
        let parent = group.restored.borrow(py).parent.clone_ref(py);
        drop(retained);
        let instruction = session
            .segment_instruction
            .lock()
            .map_err(|_| invalid("original instruction owner mutex is poisoned"))?
            .clone();
        if let Some(instruction) = instruction {
            instruction.require_owner(py)?;
        } else {
            session
                .owner()?
                .bind_prepared_segment_parent(&*parent.borrow(py).lease()?)
                .map_err(xlog_err)?;
        }
        Ok(())
    }

    pub(in crate::semantic_transition) fn private_group_parent(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
    ) -> PyResult<Py<PySemanticPublishedParent>> {
        self.require_private_group_owner(py, session, task)?;
        let retained = self.private_group()?;
        let group = retained.as_ref().expect("original private group");
        let parent = group.restored.borrow(py).parent.clone_ref(py);
        Ok(parent)
    }

    pub(in crate::semantic_transition) fn quiesce_private_group_parent(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
    ) -> PyResult<()> {
        self.require_private_group_owner(py, session, task)?;
        let instruction = session
            .segment_instruction
            .lock()
            .map_err(|_| invalid("original instruction owner mutex is poisoned"))?
            .clone();
        if let Some(instruction) = instruction {
            instruction.require_owner(py)?;
            if !std::ptr::eq(parent, &*instruction.parent.borrow(py))
                || !instruction.preparation()?.handoff_complete
            {
                return Err(invalid(
                    "private recording lost its original completed parent handoff",
                ));
            }
            return Ok(());
        }
        let streams = self.preparation_inputs.consumer_streams.python_value(py)?;
        let streams = checkpoint_consumer_streams(streams.bind(py), &mut (16 * 1024 * 1024))?;
        session
            .owner()?
            .quiesce_prepared_segment_parent(&*parent.lease()?, &streams)
            .map_err(xlog_err)
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
        for child in self.replay_children()? {
            child.detach_preparation_work(py)?;
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
                xlog_cuda::SemanticColdModelWorkDisposition::Complete,
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
                match role {
                    PrivateColdRole::Prefix => {
                        xlog_cuda::SemanticColdModelWorkPurpose::PrivatePrefix
                    }
                    PrivateColdRole::ModelPreparation => {
                        xlog_cuda::SemanticColdModelWorkPurpose::PrivateModelPreparation
                    }
                    PrivateColdRole::SuccessorPreparation => {
                        xlog_cuda::SemanticColdModelWorkPurpose::PrivateSuccessorPreparation
                    }
                    PrivateColdRole::Retirement => {
                        xlog_cuda::SemanticColdModelWorkPurpose::PrivateRetirement
                    }
                    PrivateColdRole::Adoption => {
                        xlog_cuda::SemanticColdModelWorkPurpose::PrivateAdoption
                    }
                },
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
        let instruction = session
            .segment_instruction
            .lock()
            .map_err(|_| invalid("original instruction owner mutex is poisoned"))?
            .clone();
        let admitted = match instruction {
            Some(instruction)
                if std::ptr::eq(&*instruction.parent.borrow(py), &*parent.borrow(py)) =>
            {
                let complete = instruction.preparation()?.handoff_complete;
                complete.then_some(instruction)
            }
            _ => None,
        };
        if let Some(instruction) = admitted {
            session
                .owner()?
                .attach_shared_admitted_cold_native_work(
                    &instruction.inner,
                    &*parent.borrow(py).lease()?,
                    custody,
                )
                .map_err(xlog_err)?;
        } else {
            session
                .owner()?
                .attach_shared_cold_native_work(&*parent.borrow(py).lease()?, custody)
                .map_err(xlog_err)?;
        }
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

    /// Consume only the native original whole-roster disposition, while its
    /// preparation interval is still open. Retain it before any external
    /// observer handoff; unknown completion cannot issue another cancellation.
    pub(in crate::semantic_transition) fn retain_private_non_submission(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
        scope: &Arc<()>,
        proof: xlog_cuda::SemanticPreparedSegmentNonSubmission,
    ) -> PyResult<()> {
        self.require_private_group_owner(py, session, task)?;
        let (steps, ordinal, detached) = {
            let mut retained = self.private_group()?;
            let group = retained.as_mut().expect("original private group");
            if !proof.matches(&group.native_steps)
                || group.non_submission.as_ref().is_some_and(|original| !original.matches(&group.native_steps))
                || group
                    .build_scope
                    .as_ref()
                    .is_none_or(|original| !Arc::ptr_eq(original, scope))
                || group.observer_finish_entered
                || group.selected_parent.is_some()
                || group.native_retired
            {
                return Err(invalid(
                    "private cancellation changed its original unsubmitted construction",
                ));
            }
            if group.non_submission_confirmed {
                return Ok(());
            }
            if group.non_submission.is_none() {
                group.non_submission = Some(proof.clone());
            }
            (
                group.native_steps.clone(),
                group.ordinal,
                group.preparation_detached,
            )
        };
        if detached {
            self.private_capture_boundary(py, session, task, true)?;
        }
        self.finish_private_group_preparation(py, session, task, scope)?;
        let stream = session
            .owner()?
            .prepared_stream(&steps[0])
            .map_err(xlog_err)?;
        let entered = {
            let mut retained = self.private_group()?;
            let group = retained.as_mut().expect("original cancelled group");
            std::mem::replace(&mut group.non_submission_observer_entered, true)
        };
        let observation = if entered {
            self.preparation_inputs.resource_observer.resolve_step_cancellation(
                py, ordinal, &steps, stream.cu_stream() as u64, &proof,
            )
        } else {
            self.preparation_inputs.resource_observer.cancel_step_captures(
                py, ordinal, &steps, stream.cu_stream() as u64, &proof,
            )
        };
        if let Err(error) = observation {
            return match self.preparation_inputs.resource_observer.step_cancellation_pending(
                py, ordinal, &steps, stream.cu_stream() as u64, &proof,
            ) {
                Ok(true) => Err(super::super::prepared_segment_pending(py, error)),
                Ok(false) => Err(error),
                Err(authority) => super::super::finish_with_cleanup(py, Err(error), Err(authority)),
            };
        }
        self.private_group()?
            .as_mut()
            .expect("original cancelled group")
            .non_submission_confirmed = true;
        Ok(())
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
            || group.native_retired
        {
            return Err(invalid(
                "private retirement requires its original final-use cold owner",
            ));
        }
        if group.retirement_entered && group.retirement_children_ready {
            return Ok(());
        }
        group.retirement_entered = true;
        let work = group
            .retirement_work
            .as_ref()
            .expect("checked original retirement work")
            .clone_ref(py);
        drop(retained);
        self.attach_replay_retirement_work(py, &work.borrow(py))?;
        self.private_group()?
            .as_mut()
            .expect("retained original group")
            .retirement_children_ready = true;
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

    pub(in crate::semantic_transition) fn finish_private_retained_sources(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
    ) -> PyResult<()> {
        self.require_private_group_owner(py, session, task)?;
        let children = {
            let retained = self.private_group()?;
            let group = retained.as_ref().expect("original private group");
            if !group.retirement_entered || !group.retirement_children_ready
                || group.native_retired || group.non_submission_confirmed
            {
                return Err(invalid("source final use requires the original executed private retirement"));
            }
            group.replay_children.clone()
        };
        for child in children {
            child.finish_retained_source(py, false)?;
        }
        Ok(())
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
        // The original Runtime finishes completed source members after
        // retire_segment returns. Release their Sessions only now, before
        // reporting that same retirement interval or admitting adoption work.
        self.release_replay_children(py)?;
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
                    self.finish_segment_terminal_refusal(py, original)
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
                replay_children: Vec::new(),
                replay_model_backings: Arc::new(xlog_cuda::SemanticReplayModelBackings::default()),
                non_submission: None,
                non_submission_confirmed: false,
                non_submission_observer_entered: false,
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
                retirement_children_ready: false,
                callback_entered: false,
                callback_pending: false,
                callback_result: None,
                callback_error: None,
                observer_finish_entered: false,
                backing_peak: None,
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
                    xlog_cuda::SemanticColdModelWorkPurpose::PrivateExecution,
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
            if !measured.is_exact_instance_of::<PyDict>() || measured.cast::<PyDict>()?.len() != 5 {
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
            let transfers = measured
                .get_item("segment_transfers")?
                .ok_or_else(|| invalid("private step lost its original segment transfers"))?;
            if !transfers.is_exact_instance_of::<PyTuple>()
                || transfers.cast::<PyTuple>()?.len() != 6
            {
                return Err(invalid(
                    "private step changed its immutable segment transfers",
                ));
            }
            let transfers = transfers.cast::<PyTuple>()?;
            for (index, width) in [(0, 9), (2, 10), (3, 10), (4, 10), (5, 10)] {
                let field = transfers.get_item(index)?;
                if index >= 3 && field.is_none() {
                    continue;
                }
                if !field.is_exact_instance_of::<PyTuple>()
                    || field.cast::<PyTuple>()?.len() != width
                {
                    return Err(invalid(
                        "private step changed its immutable segment transfer fields",
                    ));
                }
            }
            let original = observation.segment_transfers(py)?;
            if ColdValue::read(transfers.as_any(), &mut 4096, 0)?
                != ColdValue::read(original.bind(py).as_any(), &mut 4096, 0)?
            {
                return Err(invalid(
                    "private step substituted its original segment transfers",
                ));
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
            .step_backing_peaks(py, count)?;
        let peak = self.preparation_inputs.resource_observer.backing_peak(py)?;
        self.private_group()?
            .as_mut()
            .expect("retained original group")
            .backing_peak = Some(peak);
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
            self.preparation_inputs
                .resource_observer
                .observation_arguments(py, &arguments)?;
            let callback = self.scientific_owner.bind(py).getattr("record_step")?;
            self.private_group()?
                .as_mut()
                .expect("retained original group")
                .record_entered = Some(index);
            callback.call((), Some(&arguments))?;
            self.require_scientific_history(py)?;
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
