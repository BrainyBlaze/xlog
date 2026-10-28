use super::*;

pyo3::create_exception!(
    pyxlog._native,
    SemanticEvaluationForkPending,
    PyRuntimeError,
    "The original evaluation fork retains unfinished native restoration; resolve the same owner without executing it again."
);

struct EvaluationForkState {
    preparation: xlog_cuda::SemanticEvaluationForkPreparation,
    native_complete: bool,
    parent: Option<Py<PySemanticPublishedParent>>,
    before_model: Option<xlog_cuda::SemanticReplayStateMaterialProof>,
    callback_entered: bool,
    callback_return: Option<Py<PyAny>>,
    callback_error: Option<PyErr>,
    pending_cause: Option<PyErr>,
    model: Option<Py<PyAny>>,
    model_rosters: Option<(Py<PyAny>, Py<PyAny>)>,
    _model_binding: Option<xlog_cuda::SemanticGradientDeliveryBinding>,
    _producers: Vec<Py<PyAny>>,
    source_snapshot: Vec<u8>,
    completed: Option<Py<PySemanticTransitionRestoredCheckpoint>>,
}

/// One independently owned evaluation derived from an authenticated full checkpoint.
/// Native restoration and the original model wrapper are each entered once.
#[pyclass(name = "SemanticEvaluationFork", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticEvaluationFork {
    session: Py<PySemanticTransitionSession>,
    controller: Py<PySemanticTransitionController>,
    task_use: Py<PySemanticTransitionTaskUse>,
    source_custody: Mutex<Option<EvaluationSourceCustody>>,
    source_registry: Arc<material_source::MaterialSourceRegistry>,
    dormant_source: Arc<DormantEvaluationSource>,
    origin: xlog_cuda::SemanticEvaluationForkOrigin,
    restore_model: Py<PyAny>,
    refresh_snapshot: Py<PyAny>,
    active: AtomicBool,
    entered: AtomicBool,
    state: Mutex<EvaluationForkState>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

struct EvaluationSourceCustody {
    _checkpoint: Py<material_bytes::PySemanticMaterialBytes>,
    model: Py<material_bytes::PySemanticMaterialBytes>,
    _source: Arc<VerifiedCheckpointSource>,
    _checkpoint_sources: Arc<Mutex<CheckpointSources>>,
}

fn fork_snapshots(value: &Bound<'_, PyAny>) -> PyResult<(AuthoritySnapshot, AuthoritySnapshot)> {
    let (source, target) = evaluation_authority_snapshots(value)?;
    let mut budget = 16 * 1024 * 1024;
    Ok((
        source,
        AuthoritySnapshot::parse(&ColdValue::read(&target, &mut budget, 0)?)?,
    ))
}

fn snapshot_bytes(bytes: &[u8]) -> PyResult<AuthoritySnapshot> {
    AuthoritySnapshot::parse(&ColdValue::from_canonical_bytes(bytes)?)
}

struct EvaluationPreparationScope {
    task: Py<PySemanticTransitionTaskUse>,
    completed: bool,
}

impl Drop for EvaluationPreparationScope {
    fn drop(&mut self) {
        if !self.completed {
            Python::attach(|py| {
                let task = self.task.borrow(py);
                if let Ok(mut state) = task.state() {
                    if matches!(&state.phase, TaskUsePhase::ArenaPreparing(original)
                        if matches!(original.as_ref(), TaskUsePhase::Imported))
                    {
                        let _ = state.finish_arena_preparation();
                    }
                }
            });
        }
    }
}

impl PySemanticEvaluationFork {
    fn state(&self) -> PyResult<MutexGuard<'_, EvaluationForkState>> {
        self.state.try_lock().map_err(|_| {
            invalid("evaluation fork completion is already active or its custody is poisoned")
        })
    }

    fn require_original(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let controlled = self.controller.borrow(py);
        let task = self.task_use.borrow(py);
        controlled.require_read_issued(&task)?;
        task.require_current(&session.owner()?)?;
        if controlled.session.as_ptr() != self.session.as_ptr()
            || task.session.as_ptr() != self.session.as_ptr()
        {
            return Err(invalid(
                "evaluation fork changed its original controller or task",
            ));
        }
        Ok(())
    }

    fn refresh_authority(&self, py: Python<'_>) -> PyResult<()> {
        self.require_original(py)?;
        let previous_source = snapshot_bytes(&self.state()?.source_snapshot)?;
        let previous_target = {
            let task = self.task_use.borrow(py);
            let state = task.state()?;
            snapshot_bytes(&state.snapshot.canonical)?
        };
        // No native, task-state or completion mutex crosses the authority callback.
        let observed = self.refresh_snapshot.bind(py).call0()?;
        self.require_original(py)?;
        let (source, target) = fork_snapshots(&observed)?;
        source.newer_than(&previous_source)?;
        target.newer_than(&previous_target)?;
        self.dormant_source
            .require_native(Some(&self.origin), &source, &self.source_registry)?;
        let task = self.task_use.borrow(py);
        task.authority.check_use("inference", &target, false)?;
        let mut state = task.state()?;
        if state.snapshot.canonical != previous_target.canonical {
            return Err(invalid(
                "evaluation authority changed during its original refresh",
            ));
        }
        state.snapshot = target;
        drop(state);
        self.state()?.source_snapshot = source.canonical;
        Ok(())
    }

    fn parent(&self, py: Python<'_>) -> PyResult<Py<PySemanticPublishedParent>> {
        if let Some(parent) = &self.state()?.parent {
            return Ok(parent.clone_ref(py));
        }
        let lease = {
            let session = self.session.borrow(py);
            let mut owner = session.owner()?;
            self.task_use.borrow(py).require_current(&owner)?;
            owner.acquire().map_err(xlog_err)?
        };
        let parent = Py::new(
            py,
            PySemanticPublishedParent {
                session: self.session.clone_ref(py),
                task_use: self.task_use.clone_ref(py),
                inner: Arc::new(Mutex::new(lease)),
                continuation_producers: Mutex::new(Vec::new()),
            },
        )?;
        self.state()?.parent = Some(parent.clone_ref(py));
        Ok(parent)
    }

    fn wrap_model(&self, py: Python<'_>, parent: &Py<PySemanticPublishedParent>) -> PyResult<()> {
        let returned = {
            let mut state = self.state()?;
            if let Some(error) = &state.callback_error {
                return Err(error.clone_ref(py));
            }
            if state.model_rosters.is_some() {
                return Ok(());
            }
            if let Some(returned) = &state.callback_return {
                Some(returned.clone_ref(py))
            } else {
                if state.callback_entered {
                    return Err(invalid(
                        "the original evaluation model wrapper has not returned",
                    ));
                }
                state.callback_entered = true;
                None
            }
        };
        let returned = if let Some(returned) = returned {
            returned
        } else {
            let result = (|| -> PyResult<Py<PyAny>> {
                let session = self.session.borrow(py);
                let task = self.task_use.borrow(py);
                let acquired = parent.borrow(py);
                let _reads = ImportReadScope::checkpoint(&session, &task, &acquired, py)?;
                let source_model = self
                    .source_custody
                    .lock()
                    .map_err(|_| invalid("evaluation source custody is poisoned"))?
                    .as_ref()
                    .ok_or_else(|| invalid("evaluation source retired before model handoff"))?
                    .model
                    .clone_ref(py);
                self.restore_model
                    .bind(py)
                    .call1((
                        self.controller.clone_ref(py),
                        self.task_use.clone_ref(py),
                        parent.clone_ref(py),
                        source_model,
                    ))
                    .map(Bound::unbind)
            })();
            match result {
                Ok(returned) => {
                    // Retain the actual factory return before any projection or validation.
                    self.state()?.callback_return = Some(returned.clone_ref(py));
                    returned
                }
                Err(error) => {
                    self.state()?.callback_error = Some(error.clone_ref(py));
                    return Err(error);
                }
            }
        };
        let returned = returned.bind(py);
        if !returned.is_exact_instance_of::<PyTuple>() {
            return Err(invalid("evaluation model wrapper must return its model and complete typed and backing rosters"));
        }
        let fields = returned.cast::<PyTuple>()?;
        if fields.len() != 3 {
            return Err(invalid(
                "evaluation model wrapper requires exactly three original owners",
            ));
        }
        let model = fields.get_item(0)?.unbind();
        let tensors = fields.get_item(1)?;
        let allocations = fields.get_item(2)?;
        if !tensors.is_exact_instance_of::<PyTuple>()
            || !allocations.is_exact_instance_of::<PyTuple>()
        {
            return Err(invalid(
                "evaluation model rosters require exact immutable tuples",
            ));
        }
        let stream = self
            .session
            .borrow(py)
            .owner()?
            .evaluation_fork_consumer_stream()
            .map_err(xlog_err)?;
        let check = || self.require_original(py);
        let device = self.session.borrow(py).device_ordinal;
        let mut budget = 16 * 1024 * 1024;
        let typed = parse_tensor_inputs_guarded(&tensors, &mut budget, device, stream, &check)?;
        let backing =
            parse_tensor_inputs_guarded(&allocations, &mut budget, device, stream, &check)?;
        let mut inputs = typed.handoff.into_native();
        inputs.extend(backing.handoff.into_native());
        {
            let mut state = self.state()?;
            state._producers.extend(typed.producers);
            state._producers.extend(backing.producers);
        }
        self.require_original(py)?;
        let binding = {
            let session = self.session.borrow(py);
            let acquired = parent.borrow(py);
            let mut owner = session.owner()?;
            owner
                .require_evaluation_fork_model_roster(&*acquired.lease()?, inputs, stream)
                .map_err(xlog_err)?
        };
        let mut state = self.state()?;
        state._model_binding = Some(binding);
        state.model = Some(model);
        // These rows retain their actual tensor/export producers after this fork
        // wrapper retires; metadata-only replacement rows fail native validation.
        state.model_rosters = Some((tensors.unbind(), allocations.unbind()));
        Ok(())
    }

    fn complete(
        &self,
        py: Python<'_>,
        resolve: bool,
    ) -> PyResult<Option<Py<PySemanticTransitionRestoredCheckpoint>>> {
        self.require_original(py)?;
        if let Some(completed) = &self.state()?.completed {
            return Ok(Some(completed.clone_ref(py)));
        }
        if !resolve {
            self.refresh_authority(py)?;
        }
        if !self.state()?.native_complete {
            let session = self.session.borrow(py);
            let mut owner = session.owner()?;
            let mut state = self.state()?;
            // Resolve only the exact retained command. This branch never
            // reenters the initial restoration after an unknown completion.
            let published = if resolve {
                owner.resolve_evaluation_fork(&state.preparation)
            } else {
                owner.restore_evaluation_fork(&mut state.preparation, None)
            };
            match published {
                Ok(Some(_)) => state.native_complete = true,
                Ok(None) => return Ok(None),
                Err(error) => {
                    let pending = owner.evaluation_fork_completion_pending(&state.preparation);
                    let error = xlog_err(error);
                    if pending {
                        state.pending_cause = Some(error);
                        return Ok(None);
                    }
                    return Err(error);
                }
            }
        }
        let parent = self.parent(py)?;
        // Joining an entered command is mandatory even if later authority was
        // revoked. Only subsequent reads and model wrapping require a fresh use.
        self.refresh_authority(py)?;
        if self.state()?.before_model.is_none() {
            let proof = {
                let session = self.session.borrow(py);
                let acquired = parent.borrow(py);
                let mut owner = session.owner()?;
                owner
                    .observe_published_state_material_proof(&*acquired.lease()?)
                    .map_err(xlog_err)?
            };
            self.state()?.before_model = Some(proof);
        }
        self.wrap_model(py, &parent)?;
        self.refresh_authority(py)?;
        {
            let session = self.session.borrow(py);
            let acquired = parent.borrow(py);
            let state = self.state()?;
            session
                .owner()?
                .verify_published_loaded_state_material(
                    &*acquired.lease()?,
                    state.before_model.as_ref().ok_or_else(|| {
                        invalid("evaluation wrapper lost its original full publication proof")
                    })?,
                )
                .map_err(xlog_err)?;
        }
        {
            let session = self.session.borrow(py);
            let acquired = parent.borrow(py);
            session
                .owner()?
                .release_evaluation_fork_source_archive(&*acquired.lease()?)
                .map_err(xlog_err)?;
        }
        // Native has joined the original restoration and accepted this exact
        // model publication. Retained fork capsules keep only its durable
        // source reference and authenticated authority, not another full archive.
        let source = self
            .source_custody
            .lock()
            .map_err(|_| invalid("evaluation source custody is poisoned"))?
            .take();
        drop(source);
        let (model, tensors, allocations) = {
            let state = self.state()?;
            let (tensors, allocations) = state
                .model_rosters
                .as_ref()
                .ok_or_else(|| invalid("evaluation wrapper lost its original model rosters"))?;
            (
                state
                    .model
                    .as_ref()
                    .ok_or_else(|| invalid("evaluation wrapper lost its original model"))?
                    .clone_ref(py),
                tensors.clone_ref(py),
                allocations.clone_ref(py),
            )
        };
        let completed = PySemanticTransitionRestoredCheckpoint::issue(
            py,
            self.session.clone_ref(py),
            self.controller.clone_ref(py),
            self.task_use.clone_ref(py),
            parent,
            model,
        )?;
        *completed
            .borrow(py)
            .model_context_rosters
            .lock()
            .map_err(|_| invalid("evaluation model roster custody is poisoned"))? =
            Some((tensors, allocations));
        // Only complete byte and authority verification opens public task use.
        self.task_use
            .borrow(py)
            .state()?
            .finish_arena_preparation()?;
        self.state()?.completed = Some(completed.clone_ref(py));
        Ok(Some(completed))
    }

    fn run(
        slf: Py<Self>,
        py: Python<'_>,
        resolve: bool,
    ) -> PyResult<Py<PySemanticTransitionRestoredCheckpoint>> {
        let owner = slf.borrow(py);
        owner.session.borrow(py).require_creator()?;
        if resolve {
            if !owner.entered.load(Ordering::Acquire) {
                return Err(invalid(
                    "evaluation completion requires its original execution",
                ));
            }
        } else {
            owner
                .entered
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .map_err(|_| {
                    invalid(
                        "evaluation fork execution is single use; resolve its original completion",
                    )
                })?;
        }
        owner
            .active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid("evaluation fork completion cannot be nested"))?;
        let _active = PreparedRetirementGuard(&owner.active);
        let completed = match owner.complete(py, resolve) {
            Ok(completed) => completed,
            Err(error) => {
                // A known refusal before the native command was entered has
                // no numerical prefix to join. Preserve the genuine TaskUse
                // for the existing aborted-task terminal release boundary.
                if !owner.state()?.preparation.started() {
                    owner.task_use.borrow(py).state()?.phase = TaskUsePhase::Refused;
                }
                return Err(error);
            }
        };
        match completed {
            Some(completed) => Ok(completed),
            None => {
                let pending = SemanticEvaluationForkPending::new_err(
                    "retain this evaluation fork and call resolve_completion(); do not execute or replace its original owners");
                let cause = owner
                    .state()?
                    .pending_cause
                    .as_ref()
                    .map(|error| error.clone_ref(py));
                pending.set_cause(py, cause);
                pending.value(py).setattr("fork", slf.bind(py))?;
                pending.value(py).setattr("owner", slf.bind(py))?;
                // This exception is returned only. Caching it would create an
                // owner -> exception -> owner cycle.
                Err(pending)
            }
        }
    }
}

#[pymethods]
impl PySemanticEvaluationFork {
    #[getter]
    fn session(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionSession>> {
        self.session.borrow(py).require_creator()?;
        Ok(self.session.clone_ref(py))
    }

    #[getter]
    fn controller(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionController>> {
        self.session.borrow(py).require_creator()?;
        Ok(self.controller.clone_ref(py))
    }

    #[getter]
    fn task_use(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionTaskUse>> {
        self.session.borrow(py).require_creator()?;
        Ok(self.task_use.clone_ref(py))
    }

    #[getter(parent)]
    fn retained_parent(&self, py: Python<'_>) -> PyResult<Option<Py<PySemanticPublishedParent>>> {
        self.session.borrow(py).require_creator()?;
        Ok(self
            .state()?
            .parent
            .as_ref()
            .map(|parent| parent.clone_ref(py)))
    }

    /// Restore the independent target and wrap its genuine resident model once.
    fn execute(
        slf: Py<Self>,
        py: Python<'_>,
    ) -> PyResult<Py<PySemanticTransitionRestoredCheckpoint>> {
        Self::run(slf, py, false)
    }

    /// Continue the same retained native restoration without replaying the wrapper.
    fn resolve_completion(
        slf: Py<Self>,
        py: Python<'_>,
    ) -> PyResult<Py<PySemanticTransitionRestoredCheckpoint>> {
        Self::run(slf, py, true)
    }
}

#[pymethods]
impl PySemanticTransitionController {
    #[pyo3(signature = (task_use, checkpoint, *, member, intervention, snapshot, restore_model, resolve_checkpoint, max_checkpoint_bytes, max_total_checkpoint_bytes, refresh_snapshot))]
    #[expect(
        clippy::too_many_arguments,
        reason = "a fork retains separate scoped authority, a bounded source resolver and its actual model wrapper"
    )]
    fn prepare_evaluation_fork(
        slf: Bound<'_, Self>,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
        checkpoint: Py<material_bytes::PySemanticMaterialBytes>,
        member: &Bound<'_, PyAny>,
        intervention: &Bound<'_, PyAny>,
        snapshot: &Bound<'_, PyAny>,
        restore_model: Py<PyAny>,
        resolve_checkpoint: Py<PyAny>,
        max_checkpoint_bytes: &Bound<'_, PyAny>,
        max_total_checkpoint_bytes: &Bound<'_, PyAny>,
        refresh_snapshot: Py<PyAny>,
    ) -> PyResult<Py<PySemanticEvaluationFork>> {
        let controlled = slf.borrow();
        let session = controlled.session.borrow(py);
        session.require_creator()?;
        let task = task_use.borrow(py);
        controlled.require_read_issued(&task)?;
        task.require_current(&session.owner()?)?;
        let member = usize::try_from(ColdValue::read(member, &mut 128, 0)?.unsigned()?)
            .map_err(|_| invalid("evaluation catalogue member exceeds host address space"))?;
        if !checkpoint
            .bind(py)
            .is_exact_instance_of::<material_bytes::PySemanticMaterialBytes>()
        {
            return Err(invalid(
                "evaluation fork requires its original guarded checkpoint material",
            ));
        }
        if member < 2
            || !matches!(task.state()?.phase, TaskUsePhase::Imported)
            || !task.authority.replay.is_empty()
            || !task.authority.training.is_empty()
            || task.checkpoint.training_objective != ColdValue::None
            || task.checkpoint.training_domain != ColdValue::None
            || task.checkpoint.replay_capacity != ColdValue::None
            || task.checkpoint.replay_selection != ColdValue::None
            || task.checkpoint.evaluation.get(6) != Some(&ColdValue::Bool(false))
        {
            return Err(invalid("evaluation fork requires a newly imported evaluation member without training or replay admission"));
        }
        if !restore_model.bind(py).is_callable()
            || !refresh_snapshot.bind(py).is_callable()
            || !resolve_checkpoint.bind(py).is_callable()
        {
            return Err(invalid("evaluation fork requires its original model, authority and bounded checkpoint callbacks"));
        }
        let (mut source_snapshot, target_snapshot) = fork_snapshots(snapshot)?;
        target_snapshot.newer_than(&task.state()?.snapshot)?;
        task.authority
            .check_use("inference", &target_snapshot, false)?;
        let source_registry = session.evaluation_source_registry()?;
        let source_budget = source_registry.native_budget()?;
        let target_budget = task.checkpoint.material_sources.native_budget()?;
        let library = session
            .owner()?
            .evaluation_rule_library()
            .ok_or_else(|| invalid("evaluation fork requires its original frozen rule library"))?;
        let intervention =
            super::evaluation_rule_library::intervention(intervention, &library, &target_budget)?;
        let mut scalar_budget = 256;
        let limits = CheckpointSourceLimits::new(
            ColdValue::read(max_checkpoint_bytes, &mut scalar_budget, 0)?.unsigned()?,
            ColdValue::read(max_total_checkpoint_bytes, &mut scalar_budget, 0)?.unsigned()?,
        )?;
        let original = checkpoint.borrow(py);
        let bytes = original.as_bytes();
        if bytes.len() > limits.item_bytes || bytes.len() > limits.total_bytes {
            return Err(invalid(
                "evaluation source exceeds its declared complete checkpoint bounds",
            ));
        }
        let manifest = SemanticCheckpointManifest::decode_borrowed(bytes)?;
        let envelope = original.original_loaded(&source_budget)?;
        let mut native = envelope
            .slice_payload(checkpoint_section_range(bytes, manifest.native)?)
            .map_err(xlog_err)?;
        if native
            .checkpoint_proof()
            .map_err(xlog_err)?
            .evaluation_fork_origin()
            .is_some()
        {
            return Err(invalid(
                "evaluation fork requires its original ordinary donor checkpoint",
            ));
        }
        let referent =
            CheckpointReferent::from_checkpoint_bound_native(bytes, &source_registry, &mut native)?;
        let sources = Arc::new(Mutex::new(CheckpointSources::default()));
        let source = {
            let mut sources = sources
                .lock()
                .map_err(|_| invalid("evaluation checkpoint source custody is poisoned"))?;
            sources.bind(Some((resolve_checkpoint.bind(py), limits)))?;
            sources.retain_verified(&referent, envelope, &source_registry)?
        };
        let (_, saved_snapshot, _, _) =
            TaskCheckpointSeed::decode_bound(manifest.task, &source_registry)?;
        source_snapshot.newer_than(&saved_snapshot)?;
        let mut source_authority = TaskAuthority::parse(&source.seed.authority)?;
        source_authority
            .bind_initial_sources(&source.seed.initial_sources, &source.seed.source_mapping)?;
        source_authority.check_use("inference", &saved_snapshot, false)?;
        source_authority.check_use("inference", &source_snapshot, false)?;
        let source_authority = Arc::new(source_authority);
        let snapshots = Arc::new(Mutex::new((
            source_snapshot.canonical.clone(),
            target_snapshot.canonical.clone(),
        )));
        {
            let mut state = task.state()?;
            if !matches!(state.phase, TaskUsePhase::Imported) {
                return Err(invalid(
                    "evaluation task changed during original checkpoint authentication",
                ));
            }
            target_snapshot.newer_than(&state.snapshot)?;
            state.phase = TaskUsePhase::ArenaPreparing(Box::new(TaskUsePhase::Imported));
        }
        let mut preparation_scope = EvaluationPreparationScope {
            task: task_use.clone_ref(py),
            completed: false,
        };
        let refresh = {
            let callback = refresh_snapshot.clone_ref(py);
            let snapshots = Arc::clone(&snapshots);
            let authority = Arc::clone(&source_authority);
            let target = task_use.clone_ref(py);
            pyo3::types::PyCFunction::new_closure(py, None, None, move |args, kwargs| {
                let py = args.py();
                if !args.is_empty() || kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
                    return Err(invalid("evaluation authority refresh takes no arguments"));
                }
                let observed = callback.bind(py).call0()?;
                let (source, current) = fork_snapshots(&observed)?;
                let previous = snapshots
                    .lock()
                    .map_err(|_| invalid("evaluation authority pair custody is poisoned"))?;
                source.newer_than(&snapshot_bytes(&previous.0)?)?;
                current.newer_than(&snapshot_bytes(&previous.1)?)?;
                drop(previous);
                authority.check_use("inference", &source, false)?;
                let task = target.borrow(py);
                let session = task.session.borrow(py);
                session.require_creator()?;
                task.require_current(&session.owner()?)?;
                task.authority.check_use("inference", &current, false)?;
                *snapshots
                    .lock()
                    .map_err(|_| invalid("evaluation authority pair custody is poisoned"))? =
                    (source.canonical.clone(), current.canonical);
                ColdValue::from_canonical_bytes(&source.canonical)?.python_value(py)
            })?
        };
        resolve_replay_checkpoint_referents(
            py,
            &source_authority,
            &mut source_snapshot,
            Some(resolve_checkpoint.bind(py)),
            refresh.as_any(),
            Some(max_checkpoint_bytes),
            Some(max_total_checkpoint_bytes),
            &sources,
        )?;
        let target_snapshot = {
            let snapshots = snapshots
                .lock()
                .map_err(|_| invalid("evaluation authority pair custody is poisoned"))?;
            snapshot_bytes(&snapshots.1)?
        };
        task.require_current(&session.owner()?)?;
        let dormant = DormantEvaluationSource::from_checkpoint(bytes, &source)?;
        let context = task
            .checkpoint
            .native_model_context
            .clone()
            .ok_or_else(|| invalid("evaluation member lost its original frozen model catalogue"))?;
        let preparation = {
            let owner = session.owner()?;
            let restriction = owner
                .prepare_evaluation_intervention(intervention)
                .map_err(xlog_err)?;
            let records = owner
                .evaluation_fork_target_records(
                    &task.authority.canonical,
                    &target_snapshot.canonical,
                )
                .map_err(xlog_err)?;
            owner
                .prepare_evaluation_fork(native, &context, records, restriction)
                .map_err(xlog_err)?
        };
        let target_task_identity = *preparation.target_task_identity().as_bytes();
        if preparation.origin().member() != member {
            return Err(invalid(
                "evaluation member differs from the original imported catalogue member",
            ));
        }
        dormant.require_native(
            Some(preparation.origin()),
            &source_snapshot,
            &source_registry,
        )?;
        let source_model = Py::new(
            py,
            material_bytes::PySemanticMaterialBytes::from_native(
                py,
                original
                    .original_loaded(&source_budget)?
                    .slice_payload(checkpoint_section_range(bytes, manifest.model)?)
                    .map_err(xlog_err)?,
                &source_budget,
            )?,
        )?;
        let metadata = material_bytes::reserve_python_material::<PySemanticEvaluationFork>(
            py,
            &target_budget,
        )?;
        let mut authority = TaskAuthority::parse(&task.checkpoint.authority)?;
        authority.bind_initial_sources(
            &task.checkpoint.initial_sources,
            &task.checkpoint.source_mapping,
        )?;
        authority.external_reads = task.authority.external_reads.clone();
        if authority.canonical != task.authority.canonical {
            return Err(invalid(
                "evaluation preparation changed its original target authority",
            ));
        }
        let issuance = TaskIssuance::prepare(Arc::clone(&session.issuance))?;
        let activation = issuance.clone();
        let derived_task = Py::new(
            py,
            PySemanticTransitionTaskUse {
                session: controlled.session.clone_ref(py),
                controller: Arc::clone(&controlled.identity),
                issuance,
                importing: Arc::clone(&session.importing),
                task_identity: target_task_identity,
                task_epoch: task.task_epoch,
                authority,
                checkpoint: task
                    .checkpoint
                    .with_dormant_evaluation_source(Arc::clone(&dormant)),
                state: Mutex::new(TaskUseState {
                    phase: TaskUsePhase::ArenaPreparing(Box::new(TaskUsePhase::Imported)),
                    snapshot: target_snapshot,
                }),
                frozen_model_context: AtomicBool::new(false),
                delivery: Mutex::new(None),
                pending_delivery_receiver: Mutex::new(None),
            },
        )?;
        let origin = preparation.origin().clone();
        let fork = Py::new(
            py,
            PySemanticEvaluationFork {
                session: controlled.session.clone_ref(py),
                controller: slf.clone().unbind(),
                task_use: derived_task,
                source_custody: Mutex::new(Some(EvaluationSourceCustody {
                    _checkpoint: checkpoint.clone_ref(py),
                    model: source_model,
                    _source: source,
                    _checkpoint_sources: sources,
                })),
                source_registry,
                dormant_source: dormant,
                origin,
                restore_model,
                refresh_snapshot,
                active: AtomicBool::new(false),
                entered: AtomicBool::new(false),
                state: Mutex::new(EvaluationForkState {
                    preparation,
                    native_complete: false,
                    parent: None,
                    before_model: None,
                    callback_entered: false,
                    callback_return: None,
                    callback_error: None,
                    pending_cause: None,
                    model: None,
                    model_rosters: None,
                    _model_binding: None,
                    _producers: Vec::new(),
                    source_snapshot: source_snapshot.canonical,
                    completed: None,
                }),
                _metadata: metadata,
            },
        )?;
        let mut owner = session.owner()?;
        task.require_current(&owner)?;
        let mut original_state = task.state()?;
        if !matches!(&original_state.phase, TaskUsePhase::ArenaPreparing(original)
            if matches!(original.as_ref(), TaskUsePhase::Imported))
        {
            return Err(invalid(
                "evaluation preparation changed its original task phase",
            ));
        }
        let fork_ref = fork.borrow(py);
        let mut fork_state = fork_ref.state()?;
        owner
            .activate_evaluation_intervention(&mut fork_state.preparation)
            .map_err(xlog_err)?;
        activation.activate()?;
        original_state.phase = TaskUsePhase::Refused;
        preparation_scope.completed = true;
        drop(fork_state);
        drop(fork_ref);
        Ok(fork)
    }
}
