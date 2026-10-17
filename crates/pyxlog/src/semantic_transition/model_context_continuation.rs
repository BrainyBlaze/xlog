//! Original same-model preparation, dormant capture and acknowledged publication transfer.

use super::source_content_geometry::add;
use super::*;
pub(crate) mod declaration;
pub(crate) mod restored_prefill;

pyo3::create_exception!(
    pyxlog._native,
    SemanticModelContextContinuationPending,
    PyRuntimeError,
    "The original model continuation retains every owner; resolve its original completion."
);

#[pyclass(
    name = "SemanticModelContextContinuation",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticModelContextContinuation {
    pub(super) inner: xlog_cuda::SemanticModelContextContinuation,
    source: ModelContextSource,
    source_controller: Py<PySemanticTransitionController>,
    target_controller: Py<PySemanticTransitionController>,
    target_task: Py<PySemanticTransitionTaskUse>,
    model_owner: Py<PyAny>,
    state: Mutex<ContinuationState>,
    _metadata: Arc<xlog_cuda::SemanticReplayMaterialReservation>,
}

enum ModelContextSource {
    Live(Arc<SegmentInstructionCustody>),
    Restored {
        session: Py<PySemanticTransitionSession>,
        task: Py<PySemanticTransitionTaskUse>,
        parent: Py<PySemanticPublishedParent>,
    },
}

impl ModelContextSource {
    fn session(&self) -> &Py<PySemanticTransitionSession> {
        match self {
            Self::Live(instruction) => &instruction.session,
            Self::Restored { session, .. } => session,
        }
    }

    fn task(&self) -> &Py<PySemanticTransitionTaskUse> {
        match self {
            Self::Live(instruction) => &instruction.task_use,
            Self::Restored { task, .. } => task,
        }
    }

    fn parent(&self) -> &Py<PySemanticPublishedParent> {
        match self {
            Self::Live(instruction) => &instruction.parent,
            Self::Restored { parent, .. } => parent,
        }
    }

    fn live_instruction(&self) -> PyResult<&Arc<SegmentInstructionCustody>> {
        match self {
            Self::Live(instruction) => Ok(instruction),
            Self::Restored { .. } => Err(invalid(
                "restored model context cannot repeat live prefill or parent capture",
            )),
        }
    }
}

pub(super) struct ModelContextPythonCustody {
    source_controller: Py<PySemanticTransitionController>,
    target_controller: Py<PySemanticTransitionController>,
    target_task: Py<PySemanticTransitionTaskUse>,
    model_owner: Py<PyAny>,
    result: Mutex<Option<PyResult<xlog_cuda::SemanticModelContextContinuation>>>,
    cap_metadata: Arc<xlog_cuda::SemanticReplayMaterialReservation>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

impl ModelContextPythonCustody {
    fn require_same(
        &self,
        source: &Py<PySemanticTransitionController>,
        target: &Py<PySemanticTransitionController>,
        task: &Py<PySemanticTransitionTaskUse>,
        model: &Py<PyAny>,
    ) -> PyResult<()> {
        if self.source_controller.as_ptr() != source.as_ptr()
            || self.target_controller.as_ptr() != target.as_ptr()
            || self.target_task.as_ptr() != task.as_ptr()
            || self.model_owner.as_ptr() != model.as_ptr()
        {
            return Err(invalid(
                "original model continuation owners cannot be replaced",
            ));
        }
        Ok(())
    }

    fn project(
        &self,
        py: Python<'_>,
        instruction: &Arc<SegmentInstructionCustody>,
    ) -> PyResult<Py<PySemanticModelContextContinuation>> {
        let inner = match self
            .result
            .lock()
            .map_err(|_| invalid("original model continuation issuance is poisoned"))?
            .as_ref()
        {
            Some(Ok(inner)) => inner.clone(),
            Some(Err(error)) => return Err(error.clone_ref(py)),
            None => return Err(invalid("retain the original model continuation issuance")),
        };
        let context = Py::new(
            py,
            PySemanticModelContextContinuation {
                inner,
                source: ModelContextSource::Live(Arc::clone(instruction)),
                source_controller: self.source_controller.clone_ref(py),
                target_controller: self.target_controller.clone_ref(py),
                target_task: self.target_task.clone_ref(py),
                model_owner: self.model_owner.clone_ref(py),
                state: Mutex::new(ContinuationState::default()),
                _metadata: Arc::clone(&self.cap_metadata),
            },
        )?;
        let mut preparation = instruction.preparation()?;
        preparation.model_context_continuation = Some(context.clone_ref(py));
        preparation.model_context_owner = None;
        Ok(context)
    }
}

#[derive(Default)]
struct ContinuationState {
    restored_prefill_native: Option<xlog_cuda::SemanticRestoredModelContextPrefill>,
    restored_prefill: Option<Py<restored_prefill::PySemanticRestoredModelContextPrefill>>,
    source_alias: Option<SourceModelAlias>,
    prefill_arguments: Option<([u64; 6], Vec<u8>)>,
    prefill_entered: bool,
    prefill_error: Option<PyErr>,
    prefill_layout: Option<(Py<PyTuple>, usize)>,
    prefill_lease: Option<(SemanticInitialPrefillLease, Arc<()>)>,
    prefill: Option<Py<PySemanticInitialPrefillStage>>,
    parent_entered: bool,
    parent_ready: bool,
    parent_error: Option<PyErr>,
    build_entered: bool,
    prepared_import_callback: Option<Py<PyAny>>,
    scope: Option<Arc<()>>,
    steps: Vec<SemanticPreparedStep>,
    consumer_streams: Vec<u64>,
    work: Option<Py<learning_phase::cold_model_work::PySemanticColdModelWork>>,
    report: Option<xlog_cuda::SemanticColdModelWorkResult>,
    preparation_error: Option<PyErr>,
    source_parent: Option<Py<PySemanticPublishedParent>>,
    retired_target_expense: Option<ProposalExpense>,
    activation_callback: Option<Py<PyAny>>,
    activation_work: Option<Py<learning_phase::cold_model_work::PySemanticColdModelWork>>,
    activation_preparation_callback: Option<Py<PyAny>>,
    activation_preparation_entered: bool,
    activation_declaration_returned: bool,
    activation_admission_pending: bool,
    activation_preparation_complete: bool,
    activation_preparation_error: Option<PyErr>,
    activation_streams: Vec<u64>,
    activation_error: Option<PyErr>,
    identity: Option<xlog_cuda::SemanticPublishedIdentity>,
    acquisition: Option<xlog_cuda::SemanticPublishedAcquisition>,
    target_lease: Option<Arc<Mutex<SemanticPublishedLease>>>,
    target_parent: Option<Py<PySemanticPublishedParent>>,
    callback_entered: bool,
    callback_returned: bool,
    callback_complete: bool,
    callback_error: Option<PyErr>,
    instruction_native: Option<SemanticSegmentInstructionAdmission>,
    instruction_custody: Option<Arc<SegmentInstructionCustody>>,
    instruction: Option<Py<PySemanticSegmentInstructionAdmission>>,
}

struct SourceModelAlias {
    parent: Py<PySemanticPublishedParent>,
    model: Py<PyAny>,
    tensors: Py<PyAny>,
    allocations: Py<PyAny>,
    stream: u64,
    producers: Vec<Py<PyAny>>,
    native_entered: bool,
    pending: bool,
    validated: bool,
    error: Option<PyErr>,
}

impl PySemanticModelContextContinuation {
    pub(in crate::semantic_transition) fn checkpoint_target(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<PySemanticTransitionTaskUse>> {
        self.require_target(
            py,
            &self.target_controller.borrow(py),
            &self.target_task.borrow(py),
        )?;
        Ok(self.target_task.clone_ref(py))
    }

    pub(in crate::semantic_transition) fn require_checkpoint_source(
        &self,
        py: Python<'_>,
        source_controller: &PySemanticTransitionController,
        source_task: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
    ) -> PyResult<()> {
        let original_controller = self.source_controller.borrow(py);
        let original_task = self.source.task().borrow(py);
        if !std::ptr::eq(source_controller, &*original_controller)
            || !std::ptr::eq(source_task, &*original_task)
            || source_controller.session.as_ptr() != self.source.session().as_ptr()
            || source_task.session.as_ptr() != self.source.session().as_ptr()
            || parent.session.as_ptr() != self.source.session().as_ptr()
            || !matches!(source_task.state()?.phase, TaskUsePhase::Segment(_))
        {
            return Err(invalid(
                "checkpoint context changed its original completed source owner",
            ));
        }
        source_controller.require_read_issued(source_task)?;
        parent.require_task(py, source_task)?;
        let target_task = self.checkpoint_target(py)?;
        let target = target_task.borrow(py);
        let source_session = self.source.session().borrow(py);
        let target_session = target.session.borrow(py);
        source_session.require_creator()?;
        target_session.require_creator()?;
        let source = source_session.owner()?;
        source_task.require_current(&source)?;
        let target_owner = target_session.owner()?;
        source
            .require_model_context_checkpoint_source(&self.inner, &target_owner, &*parent.lease()?)
            .map_err(xlog_err)
    }

    pub(in crate::semantic_transition) fn project_restored_prefill(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<restored_prefill::PySemanticRestoredModelContextPrefill>> {
        self.restored_prefill(py)?
            .ok_or_else(|| invalid("restored model context has no authenticated completed prefill"))
    }

    pub(in crate::semantic_transition) fn retain_restored_prefill(
        &self,
        py: Python<'_>,
        proof: xlog_cuda::SemanticRestoredModelContextPrefill,
    ) -> PyResult<()> {
        if !matches!(&self.source, ModelContextSource::Restored { .. }) {
            return Err(invalid(
                "restored prefill proof requires its genuine restored source",
            ));
        }
        let target = self.target_task.borrow(py);
        let session = target.session.borrow(py);
        session.require_creator()?;
        session
            .owner()?
            .require_restored_model_context_prefill(&self.inner, &proof)
            .map_err(xlog_err)?;
        let mut state = self.state()?;
        if let Some(original) = &state.restored_prefill_native {
            if !original.same_handle(&proof) {
                return Err(invalid(
                    "restored prefill cannot replace its original native proof",
                ));
            }
        } else {
            state.restored_prefill_native = Some(proof);
        }
        Ok(())
    }

    fn require_source_parent(
        &self,
        py: Python<'_>,
        parent: &Py<PySemanticPublishedParent>,
    ) -> PyResult<()> {
        if !matches!(&self.source, ModelContextSource::Restored { .. }) {
            return Ok(());
        }
        let task = self.source.task().borrow(py);
        let source_controller = self.source_controller.borrow(py);
        let current = parent.borrow(py);
        if current.session.as_ptr() != self.source.session().as_ptr()
            || task.session.as_ptr() != self.source.session().as_ptr()
            || source_controller.session.as_ptr() != self.source.session().as_ptr()
        {
            return Err(invalid(
                "restored source parent changed its original session custody",
            ));
        }
        source_controller.require_read_issued(&task)?;
        current.require_task(py, &task)?;
        let target_task = self.target_task.borrow(py);
        self.require_target(py, &self.target_controller.borrow(py), &target_task)?;
        let source = self.source.session().borrow(py);
        let target = target_task.session.borrow(py);
        source.require_creator()?;
        target.require_creator()?;
        let source_owner = source.owner()?;
        task.require_current(&source_owner)?;
        let target_owner = target.owner()?;
        let result = source_owner
            .require_model_context_source_parent(&self.inner, &target_owner, &*current.lease()?)
            .map_err(xlog_err);
        result
    }

    /// Project the genuine restored SOURCE reader and already installed target.
    /// The native owner authenticates the saved operation and Program mapping.
    #[expect(
        clippy::too_many_arguments,
        reason = "restoration retains the complete original source and target owner set"
    )]
    pub(in crate::semantic_transition) fn issue_restored(
        py: Python<'_>,
        inner: xlog_cuda::SemanticModelContextContinuation,
        source_controller: Py<PySemanticTransitionController>,
        source_task: Py<PySemanticTransitionTaskUse>,
        source_parent: Py<PySemanticPublishedParent>,
        target_controller: Py<PySemanticTransitionController>,
        target_task: Py<PySemanticTransitionTaskUse>,
        model_owner: Py<PyAny>,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
    ) -> PyResult<Py<Self>> {
        let source_control = source_controller.borrow(py);
        let source_task_owner = source_task.borrow(py);
        let target_control = target_controller.borrow(py);
        let target_task_owner = target_task.borrow(py);
        source_control.require_read_issued(&source_task_owner)?;
        target_control.require_read_issued(&target_task_owner)?;
        if source_control.session.as_ptr() != source_task_owner.session.as_ptr()
            || target_control.session.as_ptr() != target_task_owner.session.as_ptr()
            || source_control.session.as_ptr() == target_control.session.as_ptr()
            || !matches!(source_task_owner.state()?.phase, TaskUsePhase::Segment(_))
            || !budget.same_pool(
                &source_task_owner
                    .checkpoint
                    .material_sources
                    .native_budget()?,
            )
            || !budget.same_pool(
                &target_task_owner
                    .checkpoint
                    .material_sources
                    .native_budget()?,
            )
        {
            return Err(invalid(
                "restored context requires its genuine completed source operation and original material pool",
            ));
        }
        let parent = source_parent.borrow(py);
        parent.require_task(py, &source_task_owner)?;
        let source_session = source_task_owner.session.borrow(py);
        let target_session = target_task_owner.session.borrow(py);
        source_session.require_creator()?;
        target_session.require_creator()?;
        {
            let source = source_session.owner()?;
            let target = target_session.owner()?;
            target
                .require_restored_model_context_source_basis(&inner, &source, &*parent.lease()?)
                .map_err(xlog_err)?;
        }
        let extent = add(
            material_bytes::python_instance_extent::<Self>(py)?,
            add(
                2 * std::mem::size_of::<usize>(),
                std::mem::size_of::<xlog_cuda::SemanticReplayMaterialReservation>(),
            )?,
        )?;
        let metadata = Arc::new(material_bytes::reserve_material_metadata(budget, extent)?);
        let session = source_task_owner.session.clone_ref(py);
        target_task_owner
            .frozen_model_context
            .store(true, Ordering::Release);
        drop(target_session);
        drop(source_session);
        drop(parent);
        drop(target_task_owner);
        drop(target_control);
        drop(source_task_owner);
        drop(source_control);
        Py::new(
            py,
            Self {
                inner,
                source: ModelContextSource::Restored {
                    session,
                    task: source_task,
                    parent: source_parent,
                },
                source_controller,
                target_controller,
                target_task,
                model_owner,
                state: Mutex::new(ContinuationState {
                    parent_ready: true,
                    ..ContinuationState::default()
                }),
                _metadata: metadata,
            },
        )
    }

    fn require_ledger_owners(
        source: &PySemanticTransitionSession,
        target: &PySemanticTransitionSession,
        source_task: &PySemanticTransitionTaskUse,
        target_task: &PySemanticTransitionTaskUse,
        source_expense: &ProposalExpense,
        target_expense: &ProposalExpense,
    ) -> PyResult<()> {
        if !Arc::ptr_eq(
            &source.proposal_expense,
            &source_task.checkpoint.proposal_expense,
        ) || !Arc::ptr_eq(
            &target.proposal_expense,
            &target_task.checkpoint.proposal_expense,
        ) || Arc::ptr_eq(&source.proposal_expense, &target.proposal_expense)
            || source_expense.capacity.is_none()
            || source_expense.capacity == Some(0)
            || source_expense.capacity != target_expense.capacity
            || target_expense.spent != 0
            || !source_expense.original_admission_bound
            || !target_expense.original_admission_bound
            || source_expense.original_admission.is_none()
            || source_expense.original_admission != target_expense.original_admission
        {
            return Err(invalid(
                "continuation requires the complete original Program, unused target ledger and actual Session-owned expenses",
            ));
        }
        Ok(())
    }

    fn prepare_checkpoint_union(
        &self,
        source: &CheckpointSources,
        target: &CheckpointSources,
        registry: &Arc<material_source::MaterialSourceRegistry>,
    ) -> PyResult<
        Option<(
            BTreeMap<[u8; 32], Arc<VerifiedCheckpointSource>>,
            xlog_cuda::SemanticReplayMaterialReservation,
        )>,
    > {
        if source.verified.is_empty() {
            return Ok(None);
        }
        let limits = target.original_limits()?.ok_or_else(|| {
            invalid("target checkpoint custody requires its original finite limits")
        })?;
        let source_limits = source
            .original_limits()?
            .ok_or_else(|| invalid("source checkpoint custody has no original finite limits"))?;
        if limits.item_bytes != source_limits.item_bytes
            || limits.total_bytes != source_limits.total_bytes
            || source.resolver.as_ref().map(Py::as_ptr) != target.resolver.as_ref().map(Py::as_ptr)
        {
            return Err(invalid(
                "checkpoint union cannot change its original limits or resolver",
            ));
        }
        let budget = registry.native_budget()?;
        let capacity = target
            .verified
            .len()
            .checked_add(source.verified.len())
            .ok_or_else(|| invalid("checkpoint union roster extent overflow"))?;
        let metadata = budget
            .reserve_ordered_index::<[u8; 32], Arc<VerifiedCheckpointSource>>(capacity)
            .map_err(xlog_err)?;
        let mut verified = target.verified.clone();
        for (digest, original) in &source.verified {
            registry.require_saved(&original.seed.material_sources)?;
            if !budget.same_pool(&original.seed.material_sources.native_budget()?) {
                return Err(invalid(
                    "checkpoint union changed its authenticated source material pool",
                ));
            }
            if let Some(existing) = verified.get(digest) {
                existing.require_referent(&original.referent)?;
            } else {
                verified.insert(*digest, Arc::clone(original));
            }
        }
        let total = verified
            .values()
            .try_fold(0usize, |total, source| {
                let bytes = usize::try_from(source.referent.checkpoint_bytes).ok()?;
                if bytes > limits.item_bytes {
                    return None;
                }
                total.checked_add(bytes)
            })
            .filter(|total| *total <= limits.total_bytes);
        if total.is_none() {
            return Err(invalid(
                "checkpoint union exceeds its original authenticated source limits",
            ));
        }
        Ok(Some((verified, metadata)))
    }
    fn state(&self) -> PyResult<MutexGuard<'_, ContinuationState>> {
        self.state
            .lock()
            .map_err(|_| invalid("original model continuation custody is poisoned"))
    }

    pub(super) fn require_target(
        &self,
        py: Python<'_>,
        controller: &PySemanticTransitionController,
        task: &PySemanticTransitionTaskUse,
    ) -> PyResult<()> {
        let original = self.target_controller.borrow(py);
        let target = self.target_task.borrow(py);
        controller.session.borrow(py).require_creator()?;
        let same_controller = match &self.source {
            ModelContextSource::Live(_) => std::ptr::eq(controller, &*original),
            ModelContextSource::Restored { .. } => {
                Arc::ptr_eq(&controller.identity, &original.identity)
                    && controller.session.as_ptr() == original.session.as_ptr()
            }
        };
        if !same_controller
            || !std::ptr::eq(task, &*target)
            || controller.session.as_ptr() != target.session.as_ptr()
            || original.session.as_ptr() != target.session.as_ptr()
            || (!task.frozen_model_context.load(Ordering::Acquire)
                && !(matches!(&self.source, ModelContextSource::Restored { .. })
                    && self.state()?.callback_complete))
        {
            return Err(invalid(
                "model continuation changed its original target controller or TaskUse",
            ));
        }
        controller.require_read_issued(task)?;
        Ok(())
    }

    pub(super) fn require_build(
        &self,
        py: Python<'_>,
        transitions: &[SemanticTransitionKind],
        capacity: &SemanticSegmentColdCapacity,
    ) -> PyResult<()> {
        self.require_target(
            py,
            &self.target_controller.borrow(py),
            &self.target_task.borrow(py),
        )?;
        let expected = self.inner.segment_cold_capacity().map_err(xlog_err)?;
        let mut state = self.state()?;
        if !state.parent_ready
            || state.build_entered
            || transitions != self.inner.segment_transitions().map_err(xlog_err)?
            || capacity.segment_capacity_bytes != expected.segment_capacity_bytes
            || capacity.other_external_cuda_bytes != expected.other_external_cuda_bytes
        {
            return Err(invalid(
                "future build must consume the complete original frozen instruction once",
            ));
        }
        state.build_entered = true;
        Ok(())
    }

    pub(super) fn prepare_parent_binding(
        &self,
        py: Python<'_>,
        parent: xlog_cuda::SemanticParentBinding,
    ) -> PyResult<()> {
        let instruction = self.source.live_instruction()?;
        let source_work = instruction
            .preparation()?
            .work
            .as_ref()
            .map(|work| work.clone_ref(py))
            .ok_or_else(|| invalid("future binding lost its original source work"))?;
        source_work.borrow(py).check(py)?;
        let source = self.source.session().borrow(py);
        let target_task = self.target_task.borrow(py);
        let target = target_task.session.borrow(py);
        let source_parent = self.source.parent().borrow(py);
        let lease = source_parent.lease()?;
        let source_owner = source.owner()?;
        let mut target_owner = target.owner()?;
        let result = target_owner
            .prepare_model_context_parent(
                &self.inner,
                &source_owner,
                &lease,
                &instruction.inner,
                &source_work.borrow(py).custody(py)?.inner,
                parent,
            )
            .map_err(xlog_err);
        result
    }

    fn project_prefill_stage(&self, py: Python<'_>) -> PyResult<Py<PySemanticInitialPrefillStage>> {
        let (lease, scope, layout, prefix_len) = {
            let state = self.state()?;
            if let Some(stage) = &state.prefill {
                return Ok(stage.clone_ref(py));
            }
            let (lease, scope) = state
                .prefill_lease
                .as_ref()
                .ok_or_else(|| invalid("future prefill lost its original issued lease"))?;
            let (layout, prefix_len) = state
                .prefill_layout
                .as_ref()
                .ok_or_else(|| invalid("future prefill lost its original layout projection"))?;
            (
                lease.clone(),
                Arc::clone(scope),
                layout.clone_ref(py),
                *prefix_len,
            )
        };
        let issued = self.target_task.borrow(py);
        let stage = Py::new(
            py,
            PySemanticInitialPrefillStage {
                session: issued.session.clone_ref(py),
                task_use: self.target_task.clone_ref(py),
                scope,
                lease,
                layout,
                prefix_len,
                model_producers: Mutex::new(Vec::new()),
                input_producer: Mutex::new(None),
                output_producers: Mutex::new(Vec::new()),
                restored_custody: None,
            },
        )?;
        self.state()?.prefill = Some(stage.clone_ref(py));
        Ok(stage)
    }

    fn resolve_source_alias(&self, py: Python<'_>) -> PyResult<()> {
        let parent = {
            let state = self.state()?;
            let alias = state
                .source_alias
                .as_ref()
                .ok_or_else(|| invalid("continuation has no original source model alias"))?;
            if alias.validated {
                return Ok(());
            }
            if let Some(error) = &alias.error {
                return Err(error.clone_ref(py));
            }
            if !alias.pending || !alias.native_entered {
                return Err(invalid(
                    "source model alias has no submitted original completion",
                ));
            }
            alias.parent.clone_ref(py)
        };
        let source = self.source.session().borrow(py);
        let target_task = self.target_task.borrow(py);
        let target = target_task.session.borrow(py);
        let result = {
            let mut source_owner = source.owner()?;
            let target_owner = target.owner()?;
            source_owner.resolve_model_context_source_alias(
                &self.inner,
                &target_owner,
                &*parent.borrow(py).lease()?,
            )
        };
        match result {
            Ok(()) => {
                let mut state = self.state()?;
                let alias = state.source_alias.as_mut().expect("original source alias");
                alias.pending = false;
                alias.validated = true;
                Ok(())
            }
            Err(error) => {
                let pending = target
                    .owner()?
                    .model_context_continuation_pending(&self.inner)
                    .map_err(xlog_err)?;
                let error = xlog_err(error);
                if pending {
                    return Err(self.pending(py, error));
                }
                let mut state = self.state()?;
                let alias = state.source_alias.as_mut().expect("original source alias");
                alias.pending = false;
                alias.error = Some(error.clone_ref(py));
                Err(error)
            }
        }
    }

    pub(super) fn prepare_cold_work(
        &self,
        py: Python<'_>,
        step: &SemanticPreparedStep,
    ) -> PyResult<Py<learning_phase::cold_model_work::PySemanticColdModelWork>> {
        if self.state()?.work.is_some() {
            return Err(invalid("future cold work already has its original owner"));
        }
        let target = self.target_task.borrow(py);
        let session = target.session.borrow(py);
        let scope = session
            .prepared_segment
            .lock()
            .map_err(|_| invalid("prepared segment mutex is poisoned"))?
            .as_ref()
            .map(|segment| Arc::clone(&segment.scope))
            .ok_or_else(|| invalid("future cold work lost its original prepared scope"))?;
        let inner = session
            .owner()?
            .prepare_model_context_cold_work(&self.inner, step)
            .map_err(xlog_err)?;
        let work = Py::new(
            py,
            learning_phase::cold_model_work::PySemanticColdModelWork::new_frozen_context(
                self.source.parent().clone_ref(py),
                inner,
                self.inner.clone(),
                step.clone(),
                self.target_task.clone_ref(py),
                Arc::clone(&scope),
            ),
        )?;
        let mut state = self.state()?;
        state.scope = Some(scope);
        state.work = Some(work.clone_ref(py));
        Ok(work)
    }

    pub(super) fn finish_build(
        &self,
        py: Python<'_>,
        scope: &Arc<()>,
        steps: &[Py<PySemanticPreparedStep>],
        streams: &[u64],
    ) -> PyResult<()> {
        let state = self.state()?;
        if state
            .scope
            .as_ref()
            .is_none_or(|original| !Arc::ptr_eq(original, scope))
            || steps.is_empty()
        {
            return Err(invalid(
                "future capture changed its original prepared scope",
            ));
        }
        let work = state
            .work
            .as_ref()
            .map(|work| work.clone_ref(py))
            .ok_or_else(|| invalid("future capture lost its actual cold work"))?;
        drop(state);
        let target = self.target_task.borrow(py);
        let session = target.session.borrow(py);
        let native_work = work.borrow(py).custody(py)?.inner;
        {
            let mut state = self.state()?;
            state.steps = steps
                .iter()
                .map(|step| step.borrow(py).inner.clone())
                .collect();
            state.consumer_streams = streams.to_vec();
        }
        let mut owner = session.owner()?;
        owner
            .resolve_model_context_prepared_model_rosters(
                &self.inner,
                &steps[0].borrow(py).inner,
                &native_work,
            )
            .map_err(xlog_err)?;
        if !owner
            .cold_model_work_report_cursor(&native_work, None)
            .map_err(xlog_err)?
            .recording_closed
        {
            owner
                .close_cold_model_work(&native_work)
                .map_err(xlog_err)?;
        }
        let result = owner.finish_frozen_context_cold_model_work(
            &self.inner,
            &steps[0].borrow(py).inner,
            &native_work,
            streams,
        );
        let report = match result {
            Ok(report) => report,
            Err(error @ xlog_cuda::SemanticTransitionError::Runtime { .. }) => {
                return Err(self.pending(py, xlog_err(error)));
            }
            Err(error) => {
                let error = xlog_err(error);
                self.state()?.preparation_error = Some(error.clone_ref(py));
                return Err(error);
            }
        };
        let mut state = self.state()?;
        state.steps = steps
            .iter()
            .map(|step| step.borrow(py).inner.clone())
            .collect();
        state.report = Some(report);
        state.consumer_streams = streams.to_vec();
        Ok(())
    }

    fn pending(&self, py: Python<'_>, original: PyErr) -> PyErr {
        let error = SemanticModelContextContinuationPending::new_err(
            "retain this original continuation and resolve its submitted completion",
        );
        error.set_cause(py, Some(original));
        let metadata = (|| -> PyResult<()> {
            let value = error.value(py);
            let context = match &self.source {
                ModelContextSource::Live(instruction) => instruction
                    .preparation()?
                    .model_context_continuation
                    .as_ref()
                    .map(|context| context.clone_ref(py)),
                ModelContextSource::Restored { .. } => None,
            };
            if let Some(context) = context {
                value.setattr("continuation", context)?;
            }
            value.setattr("source_controller", self.source_controller.clone_ref(py))?;
            value.setattr("target_controller", self.target_controller.clone_ref(py))?;
            value.setattr("source_task_use", self.source.task().clone_ref(py))?;
            value.setattr("target_task_use", self.target_task.clone_ref(py))?;
            let parent = {
                let state = self.state()?;
                state
                    .source_parent
                    .as_ref()
                    .or_else(|| state.source_alias.as_ref().map(|alias| &alias.parent))
                    .map(|parent| parent.clone_ref(py))
            };
            if let Some(parent) = parent {
                value.setattr("source_parent", parent)?;
            }
            Ok(())
        })();
        match metadata {
            Ok(()) => error,
            Err(metadata) => {
                metadata.set_cause(py, Some(error));
                metadata
            }
        }
    }

    fn handoff_target_parent(
        &self,
        py: Python<'_>,
        admission: &PySemanticSegmentInstructionAdmission,
        continuation: &Py<Self>,
    ) -> PyResult<()> {
        let instruction = &admission.inner;
        let mut preparation = instruction.preparation()?;
        if preparation.handoff_complete {
            return Ok(());
        }
        let entered = preparation.handoff_entered;
        preparation.handoff_entered = true;
        drop(preparation);
        let streams = self.state()?.consumer_streams.clone();
        let session = instruction.session.borrow(py);
        let result = if entered {
            session.owner()?.resolve_admitted_segment_parent(
                &instruction.inner,
                &mut *instruction.parent.borrow(py).lease()?,
            )
        } else {
            session.owner()?.handoff_admitted_segment_parent(
                &instruction.inner,
                &mut *instruction.parent.borrow(py).lease()?,
                &streams,
            )
        };
        match result {
            Ok(()) => {
                instruction.preparation()?.handoff_complete = true;
                Ok(())
            }
            Err(error @ xlog_cuda::SemanticTransitionError::Runtime { .. }) => {
                let pending = self.pending(py, xlog_err(error));
                match pending
                    .value(py)
                    .setattr("continuation", continuation.clone_ref(py))
                {
                    Ok(()) => Err(pending),
                    Err(error) => {
                        error.set_cause(py, Some(pending));
                        Err(error)
                    }
                }
            }
            Err(error) => Err(xlog_err(error)),
        }
    }
}

#[pymethods]
impl PySemanticSegmentInstructionAdmission {
    #[pyo3(signature = (source_controller, target_controller, target_task_use, model_owner))]
    fn freeze_model_context_continuation(
        &self,
        py: Python<'_>,
        source_controller: Py<PySemanticTransitionController>,
        target_controller: Py<PySemanticTransitionController>,
        target_task_use: Py<PySemanticTransitionTaskUse>,
        model_owner: Py<PyAny>,
    ) -> PyResult<Py<PySemanticModelContextContinuation>> {
        {
            let preparation = self.inner.preparation()?;
            if let Some(original) = &preparation.model_context_continuation {
                let context = original.borrow(py);
                if context.source_controller.as_ptr() != source_controller.as_ptr()
                    || context.target_controller.as_ptr() != target_controller.as_ptr()
                    || context.target_task.as_ptr() != target_task_use.as_ptr()
                    || context.model_owner.as_ptr() != model_owner.as_ptr()
                {
                    return Err(invalid(
                        "original model continuation owners cannot be replaced",
                    ));
                }
                return Ok(original.clone_ref(py));
            }
            if let Some(original) = &preparation.model_context_owner {
                let original = Arc::clone(original);
                original.require_same(
                    &source_controller,
                    &target_controller,
                    &target_task_use,
                    &model_owner,
                )?;
                drop(preparation);
                return original.project(py, &self.inner);
            }
            if !preparation.callback_entered
                || preparation.result.is_some()
                || preparation.callback_failed
                || preparation.build_entered
            {
                return Err(invalid(
                    "model continuation must freeze inside the original pre-draw preparation callback",
                ));
            }
        }
        self.inner.require_owner(py)?;
        let source = source_controller.borrow(py);
        let target = target_controller.borrow(py);
        let source_task = self.inner.task_use.borrow(py);
        let target_task = target_task_use.borrow(py);
        source.require_issued(&source_task)?;
        target.require_issued(&target_task)?;
        if source.session.as_ptr() != self.inner.session.as_ptr()
            || target.session.as_ptr() != target_task.session.as_ptr()
            || target.session.as_ptr() == source.session.as_ptr()
            || !matches!(target_task.state()?.phase, TaskUsePhase::Imported)
        {
            return Err(invalid(
                "model continuation requires its original source and distinct fresh target import",
            ));
        }
        let work = self
            .inner
            .preparation()?
            .work
            .as_ref()
            .map(|work| work.clone_ref(py))
            .ok_or_else(|| invalid("model continuation lost its actual source preparation work"))?;
        work.borrow(py).check(py)?;
        let source_session = source.session.borrow(py);
        let target_session = target.session.borrow(py);
        {
            let source_expense = source_session
                .proposal_expense
                .lock()
                .map_err(|_| invalid("source Proposal expense is poisoned"))?;
            let target_expense = target_session
                .proposal_expense
                .lock()
                .map_err(|_| invalid("target Proposal expense is poisoned"))?;
            PySemanticModelContextContinuation::require_ledger_owners(
                &source_session,
                &target_session,
                &source_task,
                &target_task,
                &source_expense,
                &target_expense,
            )?;
            if source_expense.spent != 0 {
                return Err(invalid(
                    "continuation ledger must freeze before any original Proposal place",
                ));
            }
        }
        let budget = source_task.checkpoint.material_sources.native_budget()?;
        let metadata = budget
            .reserve_auxiliary(
                u64::try_from(add(
                    std::mem::size_of::<ModelContextPythonCustody>(),
                    2 * std::mem::size_of::<usize>(),
                )?)
                .map_err(|_| invalid("model continuation custody metadata exceeds u64"))?,
            )
            .map_err(xlog_err)?;
        let cap_metadata = material_bytes::reserve_python_material::<
            PySemanticModelContextContinuation,
        >(py, &budget)?;
        let original = Arc::new(ModelContextPythonCustody {
            source_controller: source_controller.clone_ref(py),
            target_controller: target_controller.clone_ref(py),
            target_task: target_task_use.clone_ref(py),
            model_owner: model_owner.clone_ref(py),
            result: Mutex::new(None),
            cap_metadata: Arc::new(cap_metadata),
            _metadata: metadata,
        });
        self.inner.preparation()?.model_context_owner = Some(Arc::clone(&original));
        let result = source_session
            .owner()?
            .freeze_model_context_continuation(
                &mut *target_session.owner()?,
                &self.inner.inner,
                &*self.inner.parent.borrow(py).lease()?,
            )
            .map_err(xlog_err);
        if result.is_ok() {
            target_task
                .frozen_model_context
                .store(true, Ordering::Release);
        }
        *original
            .result
            .lock()
            .map_err(|_| invalid("original model continuation issuance is poisoned"))? =
            Some(result);
        drop(target_session);
        drop(source_session);
        drop(target_task);
        drop(source_task);
        drop(target);
        drop(source);
        original.project(py, &self.inner)
    }
}

#[pymethods]
impl PySemanticModelContextContinuation {
    #[getter]
    fn restored_prefill(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<Py<restored_prefill::PySemanticRestoredModelContextPrefill>>> {
        self.require_target(
            py,
            &self.target_controller.borrow(py),
            &self.target_task.borrow(py),
        )?;
        let (proof, model, tensors, allocations) = {
            let state = self.state()?;
            if let Some(original) = &state.restored_prefill {
                let projected = original.clone_ref(py);
                drop(state);
                projected.borrow(py).require_current(py)?;
                return Ok(Some(projected));
            }
            let Some(proof) = &state.restored_prefill_native else {
                return Ok(None);
            };
            let alias = state.source_alias.as_ref()
                .filter(|alias| alias.validated)
                .ok_or_else(|| invalid("restored prefill projection requires its authenticated source model rosters"))?;
            (
                proof.clone(),
                alias.model.clone_ref(py),
                alias.tensors.clone_ref(py),
                alias.allocations.clone_ref(py),
            )
        };
        let projected = restored_prefill::PySemanticRestoredModelContextPrefill::issue(
            py,
            self,
            proof,
            model,
            tensors,
            allocations,
        )?;
        self.state()?.restored_prefill = Some(projected.clone_ref(py));
        Ok(Some(projected))
    }

    #[getter]
    fn source_task_use(&self, py: Python<'_>) -> Py<PySemanticTransitionTaskUse> {
        self.source.task().clone_ref(py)
    }

    #[getter]
    fn target_task_use(&self, py: Python<'_>) -> Py<PySemanticTransitionTaskUse> {
        self.target_task.clone_ref(py)
    }

    #[getter]
    fn source_model_owner(&self, py: Python<'_>) -> Py<PyAny> {
        self.model_owner.clone_ref(py)
    }

    /// The actual already-issued frozen step is a preparation basis, not a
    /// fabricated published parent or permission to draw the next instruction.
    #[getter]
    fn activation_parent(&self, py: Python<'_>) -> PyResult<Py<PySemanticPreparedStep>> {
        let task = self.target_task.borrow(py);
        let session = task.session.borrow(py);
        session.require_creator()?;
        let state = self.state()?;
        let prepared = session
            .prepared_segment
            .lock()
            .map_err(|_| invalid("activation lost its original prepared owner"))?;
        let prepared = prepared
            .as_ref()
            .filter(|prepared| {
                state
                    .scope
                    .as_ref()
                    .is_some_and(|scope| Arc::ptr_eq(scope, &prepared.scope))
            })
            .ok_or_else(|| invalid("activation changed its original frozen scope"))?;
        let step = prepared
            .steps
            .first()
            .ok_or_else(|| invalid("activation lost its original frozen step"))?;
        if state
            .steps
            .first()
            .is_none_or(|actual| !actual.same_handle(&step.borrow(py).inner))
        {
            return Err(invalid(
                "activation cannot replace its original TARGET preparation basis",
            ));
        }
        Ok(step.clone_ref(py))
    }

    fn activation_prefix_layouts(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let task = self.target_task.borrow(py);
        let session = task.session.borrow(py);
        session.require_creator()?;
        let geometry = session
            .owner()?
            .model_context_activation_prefix_geometry(&self.inner)
            .map_err(xlog_err)?;
        let layout =
            |(shape, strides, dtype): &(Vec<i64>, Vec<i64>, (u8, u8))| -> PyResult<Py<PyTuple>> {
                Ok(
                    (PyTuple::new(py, shape)?, PyTuple::new(py, strides)?, *dtype)
                        .into_pyobject(py)?
                        .unbind(),
                )
            };
        Ok((layout(&geometry[0])?, layout(&geometry[1])?)
            .into_pyobject(py)?
            .unbind())
    }

    /// Declare every original native prefill verification before admit_plan.
    /// The compiler supplies layouts/physical extents, not tensor contents.
    fn declare_activation_prefill_content(
        &self,
        py: Python<'_>,
        content_geometry: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let work = self
            .state()?
            .activation_work
            .as_ref()
            .ok_or_else(|| invalid("activation content requires its original report"))?
            .clone_ref(py);
        work.borrow(py).check(py)?;
        let geometry = parse_prefill_geometry(content_geometry)?;
        let task = self.target_task.borrow(py);
        let session = task.session.borrow(py);
        let result = session
            .owner()?
            .freeze_model_context_activation_prefill(
                &self.inner,
                &work.borrow(py).custody(py)?.inner,
                &geometry,
            )
            .map_err(xlog_err);
        result
    }

    #[pyo3(signature = (parent, *, consumer_stream))]
    fn activation_prefix_views(
        &self,
        py: Python<'_>,
        parent: Py<PySemanticPublishedParent>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let work = {
            let state = self.state()?;
            if state
                .target_parent
                .as_ref()
                .is_none_or(|held| held.as_ptr() != parent.as_ptr())
            {
                return Err(invalid(
                    "activation prefix views require the held actual TARGET parent",
                ));
            }
            state
                .activation_work
                .as_ref()
                .ok_or_else(|| invalid("activation lost its original report"))?
                .clone_ref(py)
        };
        work.borrow(py).check(py)?;
        let task = self.target_task.borrow(py);
        parent.borrow(py).require_task(py, &task)?;
        let session = task.session.borrow(py);
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let (prefix, extent, descriptor) = session
            .owner()?
            .model_context_activation_prefix_views(
                &self.inner,
                &*parent.borrow(py).lease()?,
                stream,
            )
            .map_err(xlog_err)?;
        let capsule = |tensor| {
            crate::dlpack_capsule_from_tensor(
                py,
                retain_export_owner(tensor, task.session.clone_ref(py), session.owner_thread)?,
            )
        };
        Ok((
            capsule(prefix)?,
            capsule(extent)?,
            PyBytes::new(py, descriptor.as_bytes()),
        )
            .into_pyobject(py)?
            .unbind())
    }

    #[pyo3(signature = (source_parent, *, prepare_callback, consumer_streams))]
    fn prepare_activation(
        slf: Py<Self>,
        py: Python<'_>,
        source_parent: Py<PySemanticPublishedParent>,
        prepare_callback: Py<PyAny>,
        consumer_streams: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let context = slf.borrow(py);
        context.require_source_parent(py, &source_parent)?;
        if !prepare_callback.bind(py).is_callable() {
            return Err(invalid(
                "activation preparation requires its original complete compiler callback",
            ));
        }
        let streams = checkpoint_consumer_streams(consumer_streams, &mut (16 * 1024 * 1024))?;
        {
            let mut state = context.state()?;
            if state.report.is_none() {
                return Err(invalid(
                    "activation preparation requires known frozen Recompute preparation",
                ));
            }
            if let Some(original) = &state.activation_preparation_callback {
                if original.as_ptr() != prepare_callback.as_ptr()
                    || state.activation_streams != streams
                    || state
                        .source_parent
                        .as_ref()
                        .is_none_or(|parent| parent.as_ptr() != source_parent.as_ptr())
                {
                    return Err(invalid(
                        "activation preparation cannot replace its original compiler, streams or selected SOURCE parent",
                    ));
                }
            } else {
                state.source_parent = Some(source_parent.clone_ref(py));
                state.activation_preparation_callback = Some(prepare_callback.clone_ref(py));
                state.activation_streams = streams;
            }
            if state.activation_preparation_complete {
                return Ok(());
            }
            if let Some(error) = &state.activation_preparation_error {
                return Err(error.clone_ref(py));
            }
        }
        let source_task = context.source.task().borrow(py);
        source_parent.borrow(py).require_task(py, &source_task)?;
        let source = context.source.session().borrow(py);
        let task = context.target_task.borrow(py);
        let target = task.session.borrow(py);
        if context.state()?.activation_work.is_none() {
            let inner = target
                .owner()?
                .prepare_model_context_activation(
                    &*source.owner()?,
                    &context.inner,
                    &*source_parent.borrow(py).lease()?,
                )
                .map_err(xlog_err)?;
            let (step, scope) = {
                let state = context.state()?;
                (
                    state
                        .steps
                        .first()
                        .cloned()
                        .ok_or_else(|| invalid("activation lost its original frozen step"))?,
                    Arc::clone(
                        state
                            .scope
                            .as_ref()
                            .ok_or_else(|| invalid("activation lost its original frozen scope"))?,
                    ),
                )
            };
            let region = target
                .owner()?
                .model_context_activation_model_region(&context.inner, &inner)
                .map_err(xlog_err)?;
            let work = Py::new(
                py,
                learning_phase::cold_model_work::PySemanticColdModelWork::new_activation_context(
                    source_parent.clone_ref(py),
                    inner,
                    context.inner.clone(),
                    step,
                    context.target_task.clone_ref(py),
                    scope,
                    region,
                ),
            )?;
            context.state()?.activation_work = Some(work);
        }
        let work = context
            .state()?
            .activation_work
            .as_ref()
            .expect("original activation work retained")
            .clone_ref(py);
        let declaration_returned = context.state()?.activation_declaration_returned;
        if !declaration_returned {
            let mut state = context.state()?;
            if state.activation_preparation_entered {
                return Err(invalid(
                    "resolve the retained original activation declaration without repeating its compiler",
                ));
            }
            state.activation_preparation_entered = true;
        }
        if !declaration_returned {
            let callback = {
                let work_owner = work.borrow(py);
                let _scope = learning_phase::cold_model_work::ColdCallbackScope::enter(
                    py,
                    &target,
                    &work_owner,
                    work.clone_ref(py),
                )?;
                prepare_callback.bind(py).call1((
                    slf.clone_ref(py),
                    context.target_task.clone_ref(py),
                    work.clone_ref(py),
                ))
            };
            match callback {
                Ok(value) if value.is_none() => (),
                Ok(_) => {
                    let error = invalid("activation declaration callback must return None");
                    context.state()?.activation_preparation_error = Some(error.clone_ref(py));
                    return Err(error);
                }
                Err(error) => {
                    let pending = target
                        .owner()?
                        .model_context_continuation_pending(&context.inner)
                        .map_err(xlog_err)?;
                    let admitted = target
                        .owner()?
                        .cold_model_work_plan_is_admitted(&work.borrow(py).custody(py)?.inner)
                        .map_err(xlog_err)?;
                    if pending && admitted {
                        context.state()?.activation_admission_pending = true;
                        context.state()?.activation_preparation_error = Some(error.clone_ref(py));
                        return Err(context.pending(py, error));
                    }
                    context.state()?.activation_preparation_error = Some(error.clone_ref(py));
                    return Err(error);
                }
            }
            context.state()?.activation_declaration_returned = true;
        }
        let result = target.owner()?.begin_model_context_activation_work(
            &context.inner,
            &work.borrow(py).custody(py)?.inner,
        );
        if let Err(error) = result {
            let pending = target
                .owner()?
                .model_context_continuation_pending(&context.inner)
                .map_err(xlog_err)?;
            let error = xlog_err(error);
            if pending {
                context.state()?.activation_admission_pending = true;
                return Err(context.pending(py, error));
            }
            context.state()?.activation_preparation_error = Some(error.clone_ref(py));
            return Err(error);
        }
        context.state()?.activation_preparation_complete = true;
        Ok(())
    }

    /// Join the report's already-admitted original initialization. This does
    /// not invoke the Python compiler, install a report or repeat its DMA.
    fn resolve_activation_preparation(&self, py: Python<'_>) -> PyResult<()> {
        let work = {
            let state = self.state()?;
            if state.activation_preparation_complete {
                return Ok(());
            }
            if !state.activation_admission_pending {
                return Err(state
                    .activation_preparation_error
                    .as_ref()
                    .map(|error| error.clone_ref(py))
                    .unwrap_or_else(|| {
                        invalid("activation has no submitted original admission to resolve")
                    }));
            }
            state
                .activation_work
                .as_ref()
                .ok_or_else(|| invalid("activation lost its admitted original report"))?
                .clone_ref(py)
        };
        let task = self.target_task.borrow(py);
        let session = task.session.borrow(py);
        let result = session
            .owner()?
            .begin_model_context_activation_work(&self.inner, &work.borrow(py).custody(py)?.inner);
        if let Err(error) = result {
            let pending = session
                .owner()?
                .model_context_continuation_pending(&self.inner)
                .map_err(xlog_err)?;
            let error = xlog_err(error);
            if pending {
                return Err(self.pending(py, error));
            }
            self.state()?.activation_admission_pending = false;
            self.state()?.activation_preparation_error = Some(error.clone_ref(py));
            return Err(error);
        }
        let mut state = self.state()?;
        state.activation_admission_pending = false;
        drop(state);
        work.borrow(py).bind_activation_model_region(py)?;
        let mut state = self.state()?;
        state.activation_preparation_error = None;
        state.activation_preparation_complete = true;
        Ok(())
    }

    fn require_model_basis(
        &self,
        py: Python<'_>,
        model_owner: Py<PyAny>,
        target_task_use: Py<PySemanticTransitionTaskUse>,
        target_parent_or_step: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if target_task_use.as_ptr() != self.target_task.as_ptr() {
            return Err(invalid("model basis requires the original TARGET TaskUse"));
        }
        let state = self.state()?;
        let expected_model = if state.identity.is_some() {
            &state
                .source_alias
                .as_ref()
                .filter(|alias| alias.validated)
                .ok_or_else(|| {
                    invalid("published basis requires its authenticated selected model")
                })?
                .model
        } else {
            &self.model_owner
        };
        if model_owner.as_ptr() != expected_model.as_ptr() {
            return Err(invalid(
                "model basis cannot substitute its original numerical owner",
            ));
        }
        let task = self.target_task.borrow(py);
        let session = task.session.borrow(py);
        if target_parent_or_step.is_exact_instance_of::<PySemanticPreparedStep>() {
            if state.identity.is_some() {
                return Err(invalid(
                    "activated model basis requires its held published parent",
                ));
            }
            let step = target_parent_or_step.extract::<PyRef<'_, PySemanticPreparedStep>>()?;
            step.require_task(py, &task)?;
            session
                .owner()?
                .require_model_context_prepared_model_basis(&self.inner, &step.inner)
                .map_err(xlog_err)
        } else if target_parent_or_step.is_exact_instance_of::<PySemanticPublishedParent>() {
            let parent = target_parent_or_step.extract::<PyRef<'_, PySemanticPublishedParent>>()?;
            if state
                .target_parent
                .as_ref()
                .is_none_or(|held| held.as_ptr() != target_parent_or_step.as_ptr())
            {
                return Err(invalid(
                    "published model basis requires its original acquired TARGET parent",
                ));
            }
            parent.require_task(py, &task)?;
            let result = session
                .owner()?
                .require_model_context_published_model_basis(&self.inner, &*parent.lease()?)
                .map_err(xlog_err);
            result
        } else {
            Err(invalid(
                "model basis requires its genuine frozen step or acquired TARGET parent",
            ))
        }
    }

    fn require_initial_prefill_basis(
        &self,
        py: Python<'_>,
        original_stage: Py<PySemanticInitialPrefillStage>,
        selected_model_owner: Py<PyAny>,
        target_parent: Py<PySemanticPublishedParent>,
    ) -> PyResult<()> {
        self.require_model_basis(
            py,
            selected_model_owner,
            self.target_task.clone_ref(py),
            target_parent.bind(py).as_any(),
        )?;
        if self
            .state()?
            .prefill
            .as_ref()
            .is_none_or(|stage| stage.as_ptr() != original_stage.as_ptr())
        {
            return Err(invalid(
                "prefill basis requires the retained original stage and result",
            ));
        }
        let stage = original_stage.borrow(py);
        let task = self.target_task.borrow(py);
        if stage.task_use.as_ptr() != self.target_task.as_ptr()
            || stage.session.as_ptr() != task.session.as_ptr()
        {
            return Err(invalid(
                "prefill basis changed its original native stage issuer",
            ));
        }
        let session = task.session.borrow(py);
        let result = session
            .owner()?
            .require_model_context_initial_prefill_basis(
                &self.inner,
                &stage.lease,
                &*target_parent.borrow(py).lease()?,
            )
            .map_err(xlog_err);
        result
    }

    #[pyo3(signature = (source_parent, model_owner, *, tensors, model_allocations, consumer_stream))]
    pub(in crate::semantic_transition) fn bind_source_model_alias(
        &self,
        py: Python<'_>,
        source_parent: Py<PySemanticPublishedParent>,
        model_owner: Py<PyAny>,
        tensors: &Bound<'_, PyAny>,
        model_allocations: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.require_source_parent(py, &source_parent)?;
        let restored = matches!(&self.source, ModelContextSource::Restored { .. });
        let restored_proof = if restored {
            if model_owner.as_ptr() != self.model_owner.as_ptr() {
                return Err(invalid(
                    "restored source alias changed its original numerical owner",
                ));
            }
            Some(
                self.state()?
                    .restored_prefill_native
                    .clone()
                    .ok_or_else(|| {
                        invalid("restored source alias requires its known native prefill proof")
                    })?,
            )
        } else {
            None
        };
        let source = self.source.session().borrow(py);
        source.require_creator()?;
        let task = self.source.task().borrow(py);
        source_parent.borrow(py).require_task(py, &task)?;
        if let Some(proof) = &restored_proof {
            let target_task = self.target_task.borrow(py);
            let target_session = target_task.session.borrow(py);
            target_session.require_creator()?;
            let source_owner = source.owner()?;
            let target_owner = target_session.owner()?;
            target_owner
                .require_restored_model_context_source_basis(
                    &self.inner,
                    &source_owner,
                    &*source_parent.borrow(py).lease()?,
                )
                .map_err(xlog_err)?;
            target_owner
                .require_restored_model_context_prefill(&self.inner, proof)
                .map_err(xlog_err)?;
        }
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        {
            let mut state = self.state()?;
            if !restored && state.report.is_none() {
                return Err(invalid(
                    "source model aliases require completed original future preparation",
                ));
            }
            if !restored && !state.activation_preparation_complete {
                return Err(invalid(
                    "selected model aliases require the full original activation plan before producer effects",
                ));
            }
            if state.identity.is_some() || state.activation_callback.is_some() {
                return Err(invalid(
                    "source model aliases must bind before the original activation attempt",
                ));
            }
            if let Some(original) = &state.source_alias {
                if (original.parent.as_ptr() != source_parent.as_ptr()
                    && !(restored && original.validated))
                    || original.model.as_ptr() != model_owner.as_ptr()
                    || original.tensors.as_ptr() != tensors.as_ptr()
                    || original.allocations.as_ptr() != model_allocations.as_ptr()
                    || original.stream != stream
                {
                    return Err(invalid(
                        "source model alias cannot replace its original objects or handoff",
                    ));
                }
                if original.validated {
                    return Ok(());
                }
                if original.pending {
                    drop(state);
                    return self.resolve_source_alias(py);
                }
                return Err(original
                    .error
                    .as_ref()
                    .map(|error| error.clone_ref(py))
                    .unwrap_or_else(|| {
                        invalid("retain the original unresolved source model alias")
                    }));
            }
            state.source_alias = Some(SourceModelAlias {
                parent: source_parent.clone_ref(py),
                model: model_owner,
                tensors: tensors.clone().unbind(),
                allocations: model_allocations.clone().unbind(),
                stream,
                producers: Vec::new(),
                native_entered: false,
                pending: false,
                validated: false,
                error: None,
            });
        }
        let identity = (|| -> PyResult<_> {
            source
                .owner()?
                .published_identity(&*source_parent.borrow(py).lease()?)
                .map_err(xlog_err)
        })();
        let identity = match identity {
            Ok(identity) => identity,
            Err(error) => {
                self.state()?
                    .source_alias
                    .as_mut()
                    .expect("original source alias")
                    .error = Some(error.clone_ref(py));
                return Err(error);
            }
        };
        let check = || -> PyResult<()> {
            let owner = source.owner()?;
            task.require_identity(&owner)?;
            if owner
                .published_identity(&*source_parent.borrow(py).lease()?)
                .map_err(xlog_err)?
                != identity
            {
                return Err(invalid(
                    "source model alias changed its original selected parent during producer handoff",
                ));
            }
            if let Some(proof) = &restored_proof {
                let target_task = self.target_task.borrow(py);
                let target_session = target_task.session.borrow(py);
                let target_owner = target_session.owner()?;
                target_owner
                    .require_restored_model_context_source_basis(
                        &self.inner,
                        &owner,
                        &*source_parent.borrow(py).lease()?,
                    )
                    .map_err(xlog_err)?;
                target_owner
                    .require_restored_model_context_prefill(&self.inner, proof)
                    .map_err(xlog_err)?;
            }
            Ok(())
        };
        let result = (|| -> PyResult<()> {
            let mut budget = 16 * 1024 * 1024;
            let tensors = parse_tensor_inputs_guarded(
                tensors,
                &mut budget,
                source.device_ordinal,
                stream,
                &check,
            )?;
            let allocations = parse_tensor_inputs_guarded(
                model_allocations,
                &mut budget,
                source.device_ordinal,
                stream,
                &check,
            )?;
            {
                let mut state = self.state()?;
                let original = state
                    .source_alias
                    .as_mut()
                    .expect("original source alias retained before callbacks");
                original.producers.extend(tensors.producers);
                original.producers.extend(allocations.producers);
            }
            check()?;
            let target_session = self.target_task.borrow(py).session.clone_ref(py);
            let target_session = target_session.borrow(py);
            let mut source_owner = source.owner()?;
            let mut target_owner = target_session.owner()?;
            self.state()?
                .source_alias
                .as_mut()
                .expect("original source alias")
                .native_entered = true;
            if let Some(proof) = &restored_proof {
                source_owner
                    .require_restored_model_context_source_alias(
                        &self.inner,
                        &mut target_owner,
                        proof,
                        &*source_parent.borrow(py).lease()?,
                        tensors.handoff.into_native(),
                        allocations.handoff.into_native(),
                        stream,
                    )
                    .map_err(xlog_err)?;
            } else {
                source_owner
                    .require_model_context_source_alias(
                        &self.inner,
                        &mut target_owner,
                        &*source_parent.borrow(py).lease()?,
                        tensors.handoff.into_native(),
                        allocations.handoff.into_native(),
                        stream,
                    )
                    .map_err(xlog_err)?;
            }
            self.state()?
                .source_alias
                .as_mut()
                .expect("original source alias")
                .validated = true;
            Ok(())
        })();
        if let Err(error) = result {
            let native_entered = self
                .state()?
                .source_alias
                .as_ref()
                .is_some_and(|alias| alias.native_entered);
            let target_session = self.target_task.borrow(py).session.clone_ref(py);
            let pending = native_entered
                && target_session
                    .borrow(py)
                    .owner()?
                    .model_context_continuation_pending(&self.inner)
                    .map_err(xlog_err)?;
            if pending {
                self.state()?
                    .source_alias
                    .as_mut()
                    .expect("original source alias")
                    .pending = true;
                return Err(self.pending(py, error));
            }
            self.state()?
                .source_alias
                .as_mut()
                .expect("original source alias")
                .error = Some(error.clone_ref(py));
            return Err(error);
        }
        Ok(())
    }

    /// Complete only the original producer handoff, without re-exporting tensors.
    fn resolve_source_model_alias(&self, py: Python<'_>) -> PyResult<()> {
        self.resolve_source_alias(py)
    }

    #[getter]
    fn source_controller(&self, py: Python<'_>) -> Py<PySemanticTransitionController> {
        self.source_controller.clone_ref(py)
    }

    #[getter]
    fn target_controller(&self, py: Python<'_>) -> Py<PySemanticTransitionController> {
        self.target_controller.clone_ref(py)
    }

    #[getter]
    fn preparation_result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let Some(report) = self.state()?.report else {
            return Ok(py.None());
        };
        Ok((
            report.model_work,
            report.operation_count,
            report.work_bound,
            report.model_calls,
            report.native_work,
            PyTuple::new(py, report.native_events)?,
        )
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }

    #[getter]
    fn cold_budget(&self) -> PyResult<[u64; 3]> {
        self.inner.cold_budget().map_err(xlog_err)
    }

    #[getter]
    fn activation_result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let Some(report) = self.inner.activation_cold_result().map_err(xlog_err)? else {
            return Ok(py.None());
        };
        Ok((
            report.model_work,
            report.operation_count,
            report.work_bound,
            report.model_calls,
            report.native_work,
            PyTuple::new(py, report.native_events)?,
        )
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }

    #[pyo3(signature = (*, prefix_capacity, feedback_capacity, max_position, pad_token, provenance_capacity_records, model_generation, content_geometry))]
    #[expect(
        clippy::too_many_arguments,
        reason = "initial prefill retains the complete original layout and content roster"
    )]
    fn begin_initial_prefill(
        &self,
        py: Python<'_>,
        prefix_capacity: u64,
        feedback_capacity: u64,
        max_position: u64,
        pad_token: u64,
        provenance_capacity_records: u64,
        model_generation: u64,
        content_geometry: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PySemanticInitialPrefillStage>> {
        let instruction = self.source.live_instruction()?;
        self.require_target(
            py,
            &self.target_controller.borrow(py),
            &self.target_task.borrow(py),
        )?;
        let arguments = (
            [
                prefix_capacity,
                feedback_capacity,
                max_position,
                pad_token,
                provenance_capacity_records,
                model_generation,
            ],
            ColdValue::read(content_geometry, &mut (16 * 1024 * 1024), 0)?.canonical_bytes(),
        );
        {
            let mut state = self.state()?;
            if let Some(original) = &state.prefill_arguments {
                if original != &arguments {
                    return Err(invalid(
                        "future prefill cannot replace its original layout or occurrence roster",
                    ));
                }
                if let Some(stage) = &state.prefill {
                    return Ok(stage.clone_ref(py));
                }
            } else {
                state.prefill_arguments = Some(arguments);
            }
        }
        if self.state()?.prefill_lease.is_some() {
            return self.project_prefill_stage(py);
        }
        let issued = self.target_task.borrow(py);
        let layout = issued.checked_initial_source_layout(
            py,
            prefix_capacity,
            feedback_capacity,
            max_position,
            pad_token,
            provenance_capacity_records,
        )?;
        let prefix_len = layout.prefill_tokens.len();
        if self.state()?.prefill_layout.is_none() {
            let parts = initial_source_parts(py, layout.clone())?;
            self.state()?.prefill_layout = Some((parts, prefix_len));
        }
        let geometries = parse_prefill_geometry(content_geometry)?;
        if let Some(error) = &self.state()?.prefill_error {
            return Err(error.clone_ref(py));
        }
        let source_work = instruction
            .preparation()?
            .work
            .as_ref()
            .map(|work| work.clone_ref(py))
            .ok_or_else(|| invalid("future prefill lost its actual source work"))?;
        let source = self.source.session().borrow(py);
        let target = issued.session.borrow(py);
        let result = if self.state()?.prefill_entered {
            target.owner()?.resolve_model_context_prefill(&self.inner)
        } else {
            source_work.borrow(py).check(py)?;
            self.state()?.prefill_entered = true;
            let mut source_owner = source.owner()?;
            let mut target_owner = target.owner()?;
            target_owner.begin_model_context_prefill(
                &self.inner,
                &mut source_owner,
                &*self.source.parent().borrow(py).lease()?,
                &instruction.inner,
                &source_work.borrow(py).custody(py)?.inner,
                layout,
                model_generation,
                &geometries,
            )
        };
        let lease = match result {
            Ok(lease) => lease,
            Err(error) => {
                let pending = target
                    .owner()?
                    .model_context_continuation_pending(&self.inner)
                    .map_err(xlog_err)?;
                let error = xlog_err(error);
                if pending {
                    return Err(self.pending(py, error));
                }
                self.state()?.prefill_error = Some(error.clone_ref(py));
                return Err(error);
            }
        };
        let scope = issued.state()?.begin_initial_prefill()?;
        self.state()?.prefill_lease = Some((lease, scope));
        self.project_prefill_stage(py)
    }

    #[pyo3(signature = (*, metadata, source, prefix, records, tensors, model_allocations, model_storages, model_views, prefill_receipt=None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "dormant binding consumes the canonical complete parent metadata"
    )]
    fn bind_parent(
        &self,
        py: Python<'_>,
        metadata: &Bound<'_, PyAny>,
        source: &Bound<'_, PyAny>,
        prefix: &Bound<'_, PyAny>,
        records: &Bound<'_, PyAny>,
        tensors: &Bound<'_, PyAny>,
        model_allocations: &Bound<'_, PyAny>,
        model_storages: &Bound<'_, PyAny>,
        model_views: &Bound<'_, PyAny>,
        prefill_receipt: Option<&PySemanticInitialPrefillReceipt>,
    ) -> PyResult<()> {
        self.source.live_instruction()?;
        {
            let mut state = self.state()?;
            if state.parent_entered {
                return Err(invalid("future parent binding cannot be repeated"));
            }
            state.parent_entered = true;
        }
        let binding = self.target_controller.borrow(py).bind_parent_impl(
            py,
            &self.target_task.borrow(py),
            metadata,
            source,
            prefix,
            records,
            tensors,
            model_allocations,
            model_storages,
            model_views,
            prefill_receipt,
            Some(self),
        );
        if let Err(error) = binding {
            let session = self.target_task.borrow(py).session.clone_ref(py);
            let pending = session
                .borrow(py)
                .owner()?
                .model_context_continuation_pending(&self.inner)
                .map_err(xlog_err)?;
            if pending {
                return Err(self.pending(py, error));
            }
            self.state()?.parent_error = Some(error.clone_ref(py));
            return Err(error);
        }
        self.state()?.parent_ready = true;
        Ok(())
    }

    #[pyo3(signature = (*, producer, execute_prepared_imports, consumer_streams))]
    fn build_segment(
        &self,
        py: Python<'_>,
        producer: Py<PyAny>,
        execute_prepared_imports: Py<PyAny>,
        consumer_streams: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if !execute_prepared_imports.bind(py).is_callable() {
            return Err(invalid(
                "future preparation requires its original import compiler callback",
            ));
        }
        {
            let mut state = self.state()?;
            if let Some(original) = &state.prepared_import_callback {
                if original.as_ptr() != execute_prepared_imports.as_ptr() {
                    return Err(invalid(
                        "future preparation cannot replace its original import compiler",
                    ));
                }
            } else {
                state.prepared_import_callback = Some(execute_prepared_imports.clone_ref(py));
            }
        }
        let transitions = self.inner.segment_transitions().map_err(xlog_err)?;
        let transitions = transitions
            .into_iter()
            .map(|kind| match kind {
                SemanticTransitionKind::Proposal => Ok("proposal"),
                SemanticTransitionKind::Recompute => Ok("recompute"),
                SemanticTransitionKind::Update => Ok("update"),
                SemanticTransitionKind::Drain => Err(invalid(
                    "ordinary frozen instruction cannot request an explicit drain",
                )),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let transitions = PyTuple::new(py, transitions)?;
        let capacity = self.inner.segment_cold_capacity().map_err(xlog_err)?;
        let segment_capacity = capacity.segment_capacity_bytes.into_pyobject(py)?;
        let external_capacity = capacity.other_external_cuda_bytes.into_pyobject(py)?;
        self.target_controller.borrow(py).build_segment_impl(
            py,
            self.target_task.clone_ref(py),
            transitions.as_any(),
            producer,
            segment_capacity.as_any(),
            external_capacity.as_any(),
            None,
            consumer_streams,
            Some(execute_prepared_imports),
            Some(self),
        )
    }

    /// Resolve only the original dormant upload or submitted frozen report.
    fn resolve_preparation(&self, py: Python<'_>) -> PyResult<()> {
        self.require_target(
            py,
            &self.target_controller.borrow(py),
            &self.target_task.borrow(py),
        )?;
        let target = self.target_task.borrow(py);
        let session = target.session.borrow(py);
        {
            let state = self.state()?;
            if let Some(error) = state
                .parent_error
                .as_ref()
                .or(state.preparation_error.as_ref())
            {
                return Err(error.clone_ref(py));
            }
        }
        let parent_pending = {
            let state = self.state()?;
            state.parent_entered && !state.parent_ready
        };
        if parent_pending {
            let result = session.owner()?.resolve_model_context_parent(&self.inner);
            if let Err(error) = result {
                let pending = session
                    .owner()?
                    .model_context_continuation_pending(&self.inner)
                    .map_err(xlog_err)?;
                let error = xlog_err(error);
                if pending {
                    return Err(self.pending(py, error));
                }
                self.state()?.parent_error = Some(error.clone_ref(py));
                return Err(error);
            }
            if let Some(stage) = &self.state()?.prefill {
                target
                    .state()?
                    .commit_initial_prefill(&stage.borrow(py).scope)?;
            }
            self.state()?.parent_ready = true;
        }
        let state = self.state()?;
        if state.report.is_some() || state.steps.is_empty() {
            return Ok(());
        }
        let step = state.steps[0].clone();
        let work = state
            .work
            .as_ref()
            .map(|work| work.clone_ref(py))
            .ok_or_else(|| invalid("original future report lost its work owner"))?;
        drop(state);
        let result = session.owner()?.resolve_frozen_context_cold_model_work(
            &self.inner,
            &step,
            &work.borrow(py).custody(py)?.inner,
        );
        let report = match result {
            Ok(report) => report,
            Err(error @ xlog_cuda::SemanticTransitionError::Runtime { .. }) => {
                return Err(self.pending(py, xlog_err(error)));
            }
            Err(error) => {
                let error = xlog_err(error);
                self.state()?.preparation_error = Some(error.clone_ref(py));
                return Err(error);
            }
        };
        self.state()?.report = Some(report);
        Ok(())
    }

    #[pyo3(signature = (source_parent, *, activate_callback))]
    fn activate(
        slf: Py<Self>,
        py: Python<'_>,
        source_parent: Py<PySemanticPublishedParent>,
        activate_callback: Py<PyAny>,
    ) -> PyResult<Py<PySemanticPublishedParent>> {
        let context = slf.borrow(py);
        context.require_source_parent(py, &source_parent)?;
        if !activate_callback.bind(py).is_callable() {
            return Err(invalid(
                "activation requires its original model binding callback",
            ));
        }
        {
            let mut state = context.state()?;
            if state.report.is_none() {
                return Err(invalid(
                    "activation requires actual completed future preparation",
                ));
            }
            if state.source_alias.as_ref().is_none_or(|alias| {
                !alias.validated
                    || (alias.parent.as_ptr() != source_parent.as_ptr()
                        && !matches!(&context.source, ModelContextSource::Restored { .. }))
            }) {
                return Err(invalid(
                    "activation requires the actual authenticated post-source model alias and selected parent",
                ));
            }
            if let Some(original) = &state.source_parent {
                if original.as_ptr() != source_parent.as_ptr()
                    || state
                        .activation_callback
                        .as_ref()
                        .is_some_and(|original| original.as_ptr() != activate_callback.as_ptr())
                {
                    return Err(invalid(
                        "activation cannot replace its original source parent or callback",
                    ));
                }
            } else {
                state.source_parent = Some(source_parent.clone_ref(py));
            }
            if state.activation_callback.is_none() {
                state.activation_callback = Some(activate_callback.clone_ref(py));
            }
            if state.callback_complete {
                return Ok(state
                    .target_parent
                    .as_ref()
                    .expect("completed target parent")
                    .clone_ref(py));
            }
            if let Some(error) = &state.callback_error {
                return Err(error.clone_ref(py));
            }
            if let Some(error) = &state.activation_error {
                return Err(error.clone_ref(py));
            }
        }
        if context.state()?.identity.is_none() {
            let source_task = context.source.task().borrow(py);
            let target_task = context.target_task.borrow(py);
            source_parent.borrow(py).require_task(py, &source_task)?;
            let source = context.source.session().borrow(py);
            let target = target_task.session.borrow(py);
            if !Arc::ptr_eq(
                &source.checkpoint_sources,
                &source_task.checkpoint.checkpoint_sources,
            ) || !Arc::ptr_eq(
                &target.checkpoint_sources,
                &target_task.checkpoint.checkpoint_sources,
            ) || Arc::ptr_eq(&source.checkpoint_sources, &target.checkpoint_sources)
            {
                return Err(invalid(
                    "continuation checkpoint custody changed its original Session owners",
                ));
            }
            target
                .material_sources
                .require_saved(&source.material_sources)?;
            let result = {
                let mut source_expense = source
                    .proposal_expense
                    .lock()
                    .map_err(|_| invalid("source Proposal expense is poisoned"))?;
                let mut target_expense = target
                    .proposal_expense
                    .lock()
                    .map_err(|_| invalid("target Proposal expense is poisoned"))?;
                Self::require_ledger_owners(
                    &source,
                    &target,
                    &source_task,
                    &target_task,
                    &source_expense,
                    &target_expense,
                )?;
                let source_checkpoints = source
                    .checkpoint_sources
                    .lock()
                    .map_err(|_| invalid("source checkpoint custody is poisoned"))?;
                let mut target_checkpoints = target
                    .checkpoint_sources
                    .lock()
                    .map_err(|_| invalid("target checkpoint custody is poisoned"))?;
                let union = context.prepare_checkpoint_union(
                    &source_checkpoints,
                    &target_checkpoints,
                    &target.material_sources,
                )?;
                let mut task_state = source_task.state()?;
                let mut target_state = target_task.state()?;
                let (target_scope, target_operation) = match &target_state.phase {
                    TaskUsePhase::Prepared { scope, operation } => {
                        (Arc::clone(scope), operation.clone())
                    }
                    _ => {
                        return Err(invalid(
                            "continuation target changed its original captured phase",
                        ));
                    }
                };
                let operation = match &task_state.phase {
                    TaskUsePhase::Segment(operation) => operation.clone(),
                    _ => {
                        return Err(invalid(
                            "continuation source is not the completed original request",
                        ));
                    }
                };
                let parent = source_parent.borrow(py);
                let lease = parent.lease()?;
                let mut source_owner = source.owner()?;
                if source_expense.spent
                    != target
                        .owner()?
                        .continuation_source_proposal(&context.inner, &lease)
                        .map_err(xlog_err)?
                {
                    return Err(invalid(
                        "original Proposal places differ from the selected source parent",
                    ));
                }
                let mut state = context.state()?;
                if state
                    .scope
                    .as_ref()
                    .is_none_or(|scope| !Arc::ptr_eq(scope, &target_scope))
                {
                    return Err(invalid(
                        "continuation target changed its frozen scope before activation",
                    ));
                }
                let result = target.owner()?.activate_model_context_continuation(
                    &mut source_owner,
                    &context.inner,
                    &lease,
                );
                if let Ok(identity) = &result {
                    state.identity = Some(*identity);
                    state.retired_target_expense = Some(std::mem::replace(
                        &mut *target_expense,
                        std::mem::take(&mut *source_expense),
                    ));
                    if let Some((verified, metadata)) = union {
                        target_checkpoints.verified = verified;
                        target_checkpoints._continuation_index_metadata = Some(metadata);
                    }
                    task_state.phase = TaskUsePhase::ContextRetired { operation };
                    target_state.phase = TaskUsePhase::ContextActivated {
                        scope: target_scope,
                        operation: target_operation,
                    };
                }
                result
            };
            match result {
                Ok(_) => {}
                Err(error) => {
                    let pending = target
                        .owner()?
                        .model_context_continuation_pending(&context.inner)
                        .map_err(xlog_err)?;
                    let error = xlog_err(error);
                    return Err(if pending {
                        context.pending(py, error)
                    } else {
                        context.state()?.activation_error = Some(error.clone_ref(py));
                        error
                    });
                }
            }
        }
        if context.state()?.target_parent.is_none() {
            let task = context.target_task.borrow(py);
            let session = task.session.borrow(py);
            if context.state()?.target_lease.is_none() {
                let result = {
                    let mut state = context.state()?;
                    if state.acquisition.is_none() {
                        state.acquisition = Some(
                            session
                                .owner()?
                                .begin_published_acquisition()
                                .map_err(xlog_err)?,
                        );
                    }
                    session.owner()?.resolve_published_acquisition(
                        state
                            .acquisition
                            .as_ref()
                            .expect("original target acquisition"),
                    )
                };
                let lease = match result {
                    Ok(lease) => lease,
                    Err(error @ xlog_cuda::SemanticTransitionError::Runtime { .. }) => {
                        return Err(context.pending(py, xlog_err(error)));
                    }
                    Err(error) => {
                        let error = xlog_err(error);
                        context.state()?.activation_error = Some(error.clone_ref(py));
                        return Err(error);
                    }
                };
                context.state()?.target_lease = Some(Arc::new(Mutex::new(lease)));
            }
            let lease = Arc::clone(
                context
                    .state()?
                    .target_lease
                    .as_ref()
                    .expect("original acquired target lease"),
            );
            let parent = Py::new(
                py,
                PySemanticPublishedParent {
                    session: task.session.clone_ref(py),
                    task_use: context.target_task.clone_ref(py),
                    inner: lease,
                    continuation_producers: Mutex::new(Vec::new()),
                },
            )?;
            context.state()?.target_parent = Some(parent);
        }
        let parent = context
            .state()?
            .target_parent
            .as_ref()
            .expect("acquired target parent")
            .clone_ref(py);
        let callback_returned = context.state()?.callback_returned;
        if !callback_returned {
            let mut state = context.state()?;
            if state.callback_entered {
                return Err(invalid(
                    "resolve the original entered model binding without repeating it",
                ));
            }
            state.callback_entered = true;
        }
        if !callback_returned {
            let target_task = context.target_task.borrow(py);
            let target_session = target_task.session.borrow(py);
            let work = context
                .state()?
                .activation_work
                .as_ref()
                .ok_or_else(|| invalid("activation lost its original callback report"))?
                .clone_ref(py);
            let work_owner = work.borrow(py);
            let callback_scope = learning_phase::cold_model_work::ColdCallbackScope::enter(
                py,
                &target_session,
                &work_owner,
                work.clone_ref(py),
            )?;
            let selected_model = {
                let state = context.state()?;
                state
                    .source_alias
                    .as_ref()
                    .expect("authenticated source model alias")
                    .model
                    .clone_ref(py)
            };
            let result = activate_callback.bind(py).call1((
                slf.clone_ref(py),
                selected_model,
                context.target_task.clone_ref(py),
                parent.clone_ref(py),
            ));
            match result {
                Ok(value) if value.is_none() => {}
                Ok(_) => {
                    let error = invalid("model binding callback must return None");
                    context.state()?.callback_error = Some(error.clone_ref(py));
                    return Err(error);
                }
                Err(error) => {
                    context.state()?.callback_error = Some(error.clone_ref(py));
                    return Err(error);
                }
            }
            drop(callback_scope);
            context.state()?.callback_returned = true;
        }
        let result = {
            let source_session = context.source.session().borrow(py);
            let target_task = context.target_task.borrow(py);
            let target_session = target_task.session.borrow(py);
            let streams = context.state()?.activation_streams.clone();
            let result = target_session
                .owner()?
                .finish_model_context_activation_callback(
                    &mut *source_session.owner()?,
                    &context.inner,
                    &*source_parent.borrow(py).lease()?,
                    &streams,
                );
            result
        };
        if let Err(error) = result {
            let task = context.target_task.borrow(py);
            let session = task.session.borrow(py);
            let pending = session
                .owner()?
                .model_context_continuation_pending(&context.inner)
                .map_err(xlog_err)?;
            let error = xlog_err(error);
            return Err(if pending {
                context.pending(py, error)
            } else {
                context.state()?.activation_error = Some(error.clone_ref(py));
                error
            });
        }
        context
            .target_task
            .borrow(py)
            .frozen_model_context
            .store(false, Ordering::Release);
        context.state()?.callback_complete = true;
        let unused_target_expense = context.state()?.retired_target_expense.take();
        drop(unused_target_expense);
        if let ModelContextSource::Live(instruction) = &context.source {
            instruction.preparation()?.model_context_continuation.take();
        }
        Ok(parent)
    }

    fn admit_instruction(
        slf: Py<Self>,
        py: Python<'_>,
        parent: Py<PySemanticPublishedParent>,
        instruction_bytes: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PySemanticSegmentInstructionAdmission>> {
        let context = slf.borrow(py);
        let parent_matches = {
            let state = context.state()?;
            state.callback_complete
                && state
                    .target_parent
                    .as_ref()
                    .is_some_and(|original| original.as_ptr() == parent.as_ptr())
        };
        if !parent_matches {
            return Err(invalid(
                "live target instruction requires its known activated original parent",
            ));
        }
        let bytes = exact_replay_bytes(instruction_bytes)?;
        if bytes
            != context
                .inner
                .original_instruction_bytes()
                .map_err(xlog_err)?
        {
            return Err(invalid(
                "live target instruction differs from the original frozen mapping",
            ));
        }
        let original = context
            .state()?
            .instruction
            .as_ref()
            .map(|original| original.clone_ref(py));
        if let Some(original) = original {
            context.handoff_target_parent(py, &original.borrow(py), &slf)?;
            return Ok(original);
        }
        let target = context.target_task.borrow(py);
        let session = target.session.borrow(py);
        if context.state()?.instruction_native.is_none() {
            let inner = session
                .owner()?
                .admit_model_context_continuation_instruction(
                    &context.inner,
                    &*parent.borrow(py).lease()?,
                    bytes,
                )
                .map_err(xlog_err)?;
            context.state()?.instruction_native = Some(inner);
        }
        if context.state()?.instruction_custody.is_none() {
            let inner = context
                .state()?
                .instruction_native
                .as_ref()
                .expect("original target native admission retained")
                .clone();
            let owner = session.owner()?;
            let transitions = owner
                .admitted_segment_transitions(&inner)
                .map_err(xlog_err)?;
            let cold_capacity = owner
                .admitted_segment_cold_capacity(&inner)
                .map_err(xlog_err)?;
            let state = context.state()?;
            let custody = Arc::new(SegmentInstructionCustody {
                session: target.session.clone_ref(py),
                task_use: context.target_task.clone_ref(py),
                parent,
                inner,
                instruction: Arc::from(bytes),
                transitions,
                cold_capacity,
                replay_model_backings: Arc::new(xlog_cuda::SemanticReplayModelBackings::default()),
                cancelled_checkpoint: None,
                preparation: Mutex::new(SegmentInstructionPreparation {
                    scope: state.scope.as_ref().map(Arc::clone),
                    native_steps: state.steps.clone(),
                    build_entered: true,
                    ..SegmentInstructionPreparation::default()
                }),
            });
            drop(state);
            drop(owner);
            context.state()?.instruction_custody = Some(Arc::clone(&custody));
        }
        let custody = context
            .state()?
            .instruction_custody
            .as_ref()
            .map(Arc::clone)
            .expect("original target instruction custody retained");
        *session
            .segment_instruction
            .lock()
            .map_err(|_| invalid("original instruction owner mutex is poisoned"))? =
            Some(Arc::clone(&custody));
        session
            .prepared_segment
            .lock()
            .map_err(|_| invalid("prepared segment mutex is poisoned"))?
            .as_mut()
            .ok_or_else(|| invalid("activated target lost its original captured segment"))?
            .instruction = Some(Arc::clone(&custody));
        let admission = Py::new(py, PySemanticSegmentInstructionAdmission { inner: custody })?;
        context.state()?.instruction = Some(admission.clone_ref(py));
        context.handoff_target_parent(py, &admission.borrow(py), &slf)?;
        Ok(admission)
    }
}

fn parse_prefill_geometry(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<xlog_cuda::SemanticModelContextPrefillContentGeometry>> {
    if !value.is_exact_instance_of::<PyTuple>() {
        return Err(invalid(
            "future prefill geometry requires its exact immutable roster",
        ));
    }
    let mut budget = 16 * 1024 * 1024;
    let mut occurrences = Vec::new();
    for occurrence in value.cast::<PyTuple>()?.iter() {
        if !occurrence.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "future prefill occurrences must be exact immutable tuples",
            ));
        }
        let occurrence = occurrence.cast::<PyTuple>()?;
        if occurrence.len() != 2
            || !occurrence.get_item(0)?.is_exact_instance_of::<PyBool>()
            || !occurrence.get_item(1)?.is_exact_instance_of::<PyTuple>()
        {
            return Err(invalid(
                "future prefill occurrence requires verification and original tensor geometries",
            ));
        }
        let mut tensors = Vec::new();
        for tensor in occurrence.get_item(1)?.cast::<PyTuple>()?.iter() {
            if !tensor.is_exact_instance_of::<PyTuple>() {
                return Err(invalid(
                    "future prefill tensors must be exact immutable geometry tuples",
                ));
            }
            let tensor = tensor.cast::<PyTuple>()?;
            if tensor.len() != 4 {
                return Err(invalid(
                    "future tensor geometry requires layout, logical extent and physical bytes",
                ));
            }
            tensors.push(xlog_cuda::SemanticModelContextPrefillTensorGeometry {
                layout: parse_tensor_layout(&ColdValue::read(
                    &tensor.get_item(0)?,
                    &mut budget,
                    0,
                )?)?,
                logical_begin: ColdValue::read(&tensor.get_item(1)?, &mut budget, 0)?.unsigned()?,
                logical_end: ColdValue::read(&tensor.get_item(2)?, &mut budget, 0)?.unsigned()?,
                physical_bytes: usize::try_from(
                    ColdValue::read(&tensor.get_item(3)?, &mut budget, 0)?.unsigned()?,
                )
                .map_err(|_| invalid("future physical extent exceeds address space"))?,
            });
        }
        occurrences.push(xlog_cuda::SemanticModelContextPrefillContentGeometry {
            tensors,
            verify: occurrence.get_item(0)?.extract::<bool>()?,
        });
    }
    Ok(occurrences)
}
