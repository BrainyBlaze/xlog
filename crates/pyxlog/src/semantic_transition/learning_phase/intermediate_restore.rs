//! Original current-state restoration through the canonical full checkpoint path.

use super::phase_evaluation::EvaluationOwners;
use super::*;
use xlog_cuda::{
    SemanticColdModelWork, SemanticColdModelWorkRegion, SemanticColdModelWorkResult,
    SemanticColdNativeWork,
};

#[derive(Default)]
struct RestoreSnapshot {
    entered: bool,
    admitted: bool,
    saved: Option<Py<PyBytes>>,
    error: Option<PyErr>,
}

pub(super) struct IntermediateRestore {
    previous: Option<Box<IntermediateRestore>>,
    branch: &'static str,
    owners: Option<EvaluationOwners>,
    retiring_parent: Py<PySemanticPublishedParent>,
    source_identity: Py<PyTuple>,
    entries: Py<PyTuple>,
    material: Vec<u8>,
    instruction: Vec<u8>,
    ordinal: u64,
    budget: [u64; 3],
    decode_entered: bool,
    decoded: Option<Py<PyAny>>,
    decode_verified: bool,
    recipe: Option<SemanticLearningPhaseTransition>,
    started: bool,
    ready: bool,
    work: Option<SemanticColdModelWork>,
    custody: Option<SemanticColdNativeWork>,
    regions: Vec<SemanticColdModelWorkRegion>,
    callbacks: [Option<Py<PySemanticColdModelWork>>; 4],
    snapshots: [RestoreSnapshot; 2],
    retirement_entered: bool,
    retirement_completed: bool,
    session_release_entered: bool,
    session_released: bool,
    models_dropped: bool,
    restore_entered: bool,
    restored: Option<Py<PySemanticTransitionRestoredCheckpoint>>,
    error: Option<PyErr>,
    child_joined: bool,
    closed: bool,
    report: Option<SemanticColdModelWorkResult>,
    observer_finish_entered: bool,
    backing_peak: Option<u64>,
    record_entered: bool,
    recorded: bool,
    released: bool,
    budget_exceeded: bool,
}

impl PySemanticLearningPhaseTransition {
    fn intermediate(&self) -> PyResult<MutexGuard<'_, Option<IntermediateRestore>>> {
        self.intermediate_restore
            .lock()
            .map_err(|_| invalid("original intermediate restoration custody mutex is poisoned"))
    }

    pub(super) fn intermediate_restore_pending(&self) -> PyResult<bool> {
        Ok(self
            .intermediate()?
            .as_ref()
            .is_some_and(|entry| !entry.recorded))
    }

    pub(super) fn intermediate_restore_retained(&self) -> PyResult<bool> {
        Ok(self.intermediate()?.is_some())
    }

    pub(super) fn intermediate_restore_unfinished(&self, branch: &'static str) -> PyResult<bool> {
        Ok(self
            .intermediate()?
            .as_ref()
            .is_some_and(|entry| entry.branch == branch && !entry.released))
    }

    fn require_intermediate_entry(&self, py: Python<'_>) -> PyResult<()> {
        self.records()?.require_preparation_admission()?;
        self.preparation_inputs
            .require_program(py, &self.scientific_owner)?;
        let retained = self.intermediate()?;
        let current = retained
            .as_ref()
            .ok_or_else(|| invalid("intermediate restore lost its original entry"))?;
        let (material, instruction) = Self::singleton_lifecycle_material(current.entries.bind(py))?;
        if material != current.material || instruction != current.instruction {
            return Err(invalid(
                "intermediate restore changed its complete frozen entry",
            ));
        }
        Ok(())
    }

    fn capture_intermediate_restore(&self, py: Python<'_>, branch: &'static str) -> PyResult<()> {
        if self.intermediate_restore_unfinished(branch)? {
            return self.require_intermediate_entry(py);
        }
        let (owners, ordinal) = self.private_current_restore_input(py, branch)?;
        let next = self.scientific_owner.bind(py).getattr("next_operation")?;
        if !next.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "intermediate restore requires its complete original scheduled entry",
            ));
        }
        let entries = next.cast::<PyTuple>()?;
        let (material, instruction) = Self::singleton_lifecycle_material(entries)?;
        let decoded = ColdValue::from_canonical_bytes(&material)?;
        let fields = decoded.fields(6)?;
        if fields[0].text()? != "restore"
            || fields[1].text()? != branch
            || fields[2].unsigned()? != ordinal
            || fields[3].unsigned()? != 0
            || fields[5] != ColdValue::None
        {
            return Err(invalid(
                "intermediate restore changed its original branch or contiguous position",
            ));
        }
        let budget = fields[4].fields(3)?;
        let budget = [
            budget[0].unsigned()?,
            budget[1].unsigned()?,
            budget[2].unsigned()?,
        ];
        if budget[1] == 0 {
            return Err(invalid(
                "intermediate restore requires its original backing memory limit",
            ));
        }
        let source_identity = owners.parent.borrow(py).identity(py)?;
        let retiring_parent = owners.parent.clone_ref(py);
        let mut retained = self.intermediate()?;
        if retained.as_ref().is_some_and(|entry| {
            entry.branch != branch
                || !entry.recorded
                || !entry.released
                || entry.budget_exceeded
                || entry.error.is_some()
        }) {
            return Err(invalid(
                "intermediate restore cannot replace unknown or refused original work",
            ));
        }
        let previous = retained.take().map(Box::new);
        *retained = Some(IntermediateRestore {
            previous,
            branch,
            owners: Some(owners),
            retiring_parent,
            source_identity,
            entries: entries.clone().unbind(),
            material,
            instruction,
            ordinal,
            budget,
            decode_entered: false,
            decoded: None,
            decode_verified: false,
            recipe: None,
            started: false,
            ready: false,
            work: None,
            custody: None,
            regions: Vec::new(),
            callbacks: std::array::from_fn(|_| None),
            snapshots: std::array::from_fn(|_| RestoreSnapshot::default()),
            retirement_entered: false,
            retirement_completed: false,
            session_release_entered: false,
            session_released: false,
            models_dropped: false,
            restore_entered: false,
            restored: None,
            error: None,
            child_joined: false,
            closed: false,
            report: None,
            observer_finish_entered: false,
            backing_peak: None,
            record_entered: false,
            recorded: false,
            released: false,
            budget_exceeded: false,
        });
        Ok(())
    }

    fn decode_intermediate_restore(&self, py: Python<'_>) -> PyResult<()> {
        self.require_intermediate_entry(py)?;
        let (entered, decoded, entries, verified) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            (
                current.decode_entered,
                current.decoded.as_ref().map(|value| value.clone_ref(py)),
                current.entries.clone_ref(py),
                current.decode_verified,
            )
        };
        if verified {
            return Ok(());
        }
        let decoded = match decoded {
            Some(decoded) => decoded,
            None if entered => {
                return Err(invalid(
                    "unknown intermediate decoder cannot repeat its original callback",
                ))
            }
            None => {
                let decoder = self
                    .store()?
                    .as_ref()
                    .ok_or_else(|| {
                        invalid("intermediate restore lost its original lifecycle decoder")
                    })?
                    .decode
                    .clone_ref(py);
                self.intermediate()?
                    .as_mut()
                    .expect("original intermediate restore")
                    .decode_entered = true;
                let decoded = decoder
                    .bind(py)
                    .call1((entries.bind(py).get_item(0)?,))?
                    .unbind();
                self.intermediate()?
                    .as_mut()
                    .expect("original intermediate restore")
                    .decoded = Some(decoded.clone_ref(py));
                decoded
            }
        };
        if !decoded.bind(py).is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "intermediate restore requires its original closed decoder tuple",
            ));
        }
        let decoded = decoded.bind(py).cast::<PyTuple>()?;
        if decoded.len() != 3
            || ColdValue::read(&decoded.get_item(0)?, &mut 128, 0)?.text()? != "restore"
            || ColdValue::read(&decoded.get_item(1)?, &mut 128, 0)?.text()? != "current"
        {
            return Err(invalid(
                "intermediate restore must use the actual current complete checkpoint",
            ));
        }
        let recipe = decoded.get_item(2)?;
        let recipe = if recipe.is_none() {
            None
        } else {
            let original = recipe.extract::<PyRef<'_, PySemanticLearningPhaseRecipe>>()?;
            let mut native = original.inner.clone();
            if !native.admission.is_empty() {
                return Err(invalid(
                    "intermediate recipe cannot replace its original signed admission",
                ));
            }
            native.admission = self.records()?.confirmed_admission()?.to_vec();
            Some(native)
        };
        let mut retained = self.intermediate()?;
        let current = retained.as_mut().expect("original intermediate restore");
        current.recipe = recipe;
        current.decode_verified = true;
        Ok(())
    }

    pub(super) fn require_intermediate_continuation(&self, py: Python<'_>) -> PyResult<()> {
        self.require_intermediate_entry(py)?;
        let retained = self.intermediate()?;
        let current = retained.as_ref().expect("original intermediate restore");
        if let Some(error) = &current.error {
            return Err(error.clone_ref(py));
        }
        if current.restore_entered && current.restored.is_none() {
            return Err(invalid(
                "unknown intermediate construction cannot repeat its original factory",
            ));
        }
        Ok(())
    }

    pub(super) fn require_intermediate_restore_admission(
        &self,
        py: Python<'_>,
        checkpoint: &[u8],
        transition: Option<&SemanticLearningPhaseTransition>,
    ) -> PyResult<()> {
        self.require_intermediate_entry(py)?;
        let retained = self.intermediate()?;
        let current = retained.as_ref().expect("original intermediate restore");
        if let Some(error) = &current.error {
            return Err(error.clone_ref(py));
        }
        let same_recipe = match (current.recipe.as_ref(), transition) {
            (None, None) => true,
            (Some(original), Some(actual)) => {
                original.source == actual.source
                    && original.target == actual.target
                    && original.phase_index == actual.phase_index
                    && original.completed_updates_index == actual.completed_updates_index
                    && original.recipe_digest() == actual.recipe_digest()
                    && original.admission == actual.admission
            }
            _ => false,
        };
        if !current.decode_verified
            || !current.ready
            || !current.session_released
            || !current.models_dropped
            || !current.restore_entered
            || current.restored.is_some()
            || !same_recipe
            || current.snapshots[0]
                .saved
                .as_ref()
                .is_none_or(|saved| saved.bind(py).as_bytes() != checkpoint)
        {
            return Err(invalid(
                "intermediate allocation changed its actual checkpoint, recipe or known retirement",
            ));
        }
        Ok(())
    }

    pub(super) fn intermediate_restore_native_work(
        &self,
        py: Python<'_>,
    ) -> PyResult<SemanticColdNativeWork> {
        let work = self
            .intermediate()?
            .as_ref()
            .and_then(|current| current.work.clone())
            .ok_or_else(|| {
                invalid("intermediate allocation lost its original pre-allocation report")
            })?;
        self.source
            .borrow(py)
            .owner()?
            .share_cold_native_work(&work)
            .map_err(xlog_err)
    }

    fn intermediate_callback(
        &self,
        py: Python<'_>,
        index: usize,
        parent: &Py<PySemanticPublishedParent>,
    ) -> PyResult<Py<PySemanticColdModelWork>> {
        let mut retained = self.intermediate()?;
        let current = retained.as_mut().expect("original intermediate restore");
        if let Some(original) = &current.callbacks[index] {
            if original.borrow(py).parent.as_ptr() != parent.as_ptr() {
                return Err(invalid("intermediate callback changed its original parent"));
            }
            return Ok(original.clone_ref(py));
        }
        let original = Py::new(
            py,
            PySemanticColdModelWork {
                parent: parent.clone_ref(py),
                reader: self.parent.clone_ref(py),
                inner: current
                    .work
                    .as_ref()
                    .expect("original restore report")
                    .clone(),
                region: Some(current.regions[index].clone()),
                active: AtomicBool::new(false),
            },
        )?;
        current.callbacks[index] = Some(original.clone_ref(py));
        Ok(original)
    }

    pub(super) fn intermediate_factory_work(
        &self,
        py: Python<'_>,
        parent: &Py<PySemanticPublishedParent>,
    ) -> PyResult<Py<PySemanticColdModelWork>> {
        let construction = self.private_restore()?;
        if construction
            .as_ref()
            .and_then(|owners| owners.parent.as_ref())
            .is_none_or(|original| original.as_ptr() != parent.as_ptr())
        {
            return Err(invalid(
                "intermediate factory changed its actual retained child parent",
            ));
        }
        drop(construction);
        if self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .callbacks[2]
            .is_some()
        {
            return Err(invalid(
                "intermediate factory cannot replace its original registrar",
            ));
        }
        self.intermediate_callback(py, 2, parent)
    }

    pub(super) fn intermediate_feedback_work(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<(&'static str, Py<PySemanticColdModelWork>)>> {
        let retained = self.intermediate()?;
        Ok(retained.as_ref().and_then(|current| {
            current.callbacks[2]
                .as_ref()
                .map(|work| (current.branch, work.clone_ref(py)))
        }))
    }

    pub(super) fn require_intermediate_snapshot_save(
        &self,
        py: Python<'_>,
        controller: &PySemanticTransitionController,
        task: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
    ) -> PyResult<()> {
        self.require_intermediate_entry(py)?;
        let mut retained = self.intermediate()?;
        let current = retained.as_mut().expect("original intermediate restore");
        let index = usize::from(current.session_released);
        let same = if index == 0 {
            let owners = current
                .owners
                .as_ref()
                .expect("original current-state owners");
            std::ptr::eq(controller, &*owners.controller.borrow(py))
                && std::ptr::eq(task, &*owners.task.borrow(py))
                && std::ptr::eq(parent, &*owners.parent.borrow(py))
        } else {
            let owners = current
                .restored
                .as_ref()
                .ok_or_else(|| invalid("restored snapshot lost its original complete owners"))?
                .borrow(py);
            let same = std::ptr::eq(controller, &*owners.controller.borrow(py))
                && std::ptr::eq(task, &*owners.task_use.borrow(py))
                && std::ptr::eq(parent, &*owners.parent.borrow(py));
            same
        };
        let snapshot = &mut current.snapshots[index];
        if !current.ready
            || !same
            || !snapshot.entered
            || snapshot.admitted
            || snapshot.saved.is_some()
            || snapshot.error.is_some()
            || !matches!(task.state()?.phase, TaskUsePhase::ArenaPreparing(_))
        {
            return Err(invalid(
                "intermediate snapshot changed or repeated its original save or selected owners",
            ));
        }
        snapshot.admitted = true;
        Ok(())
    }

    fn intermediate_snapshot(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyBytes>> {
        self.require_intermediate_entry(py)?;
        let (entered, owners) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            let snapshot = &current.snapshots[index];
            if let Some(error) = &snapshot.error {
                return Err(error.clone_ref(py));
            }
            if let Some(saved) = &snapshot.saved {
                return Ok(saved.clone_ref(py));
            }
            let owners = if index == 0 {
                let original = current.owners.as_ref().expect("actual pre-restore owners");
                EvaluationOwners {
                    controller: original.controller.clone_ref(py),
                    task: original.task.clone_ref(py),
                    parent: original.parent.clone_ref(py),
                    model: original.model.clone_ref(py),
                }
            } else {
                let original = current
                    .restored
                    .as_ref()
                    .ok_or_else(|| invalid("post-restore save precedes its actual factory return"))?
                    .borrow(py);
                EvaluationOwners {
                    controller: original.controller.clone_ref(py),
                    task: original.task_use.clone_ref(py),
                    parent: original.parent.clone_ref(py),
                    model: original.model.clone_ref(py),
                }
            };
            (snapshot.entered, owners)
        };
        if entered {
            return Err(invalid(
                "unknown intermediate full snapshot cannot repeat its original serializer",
            ));
        }
        let work =
            self.intermediate_callback(py, if index == 0 { 0 } else { 3 }, &owners.parent)?;
        let serializer = self.model_owner(py, PhaseModelOwner::SerializeCandidate)?;
        let selected = owners.model.clone_ref(py);
        let session = owners.parent.borrow(py).session.clone_ref(py);
        let snapshot_model =
            pyo3::types::PyCFunction::new_closure(py, None, None, move |args, kwargs| {
                let py = args.py();
                if !args.is_empty() || kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
                    return Err(invalid(
                        "intermediate serializer accepts its original zero-argument call",
                    ));
                }
                let session = session.borrow(py);
                let original = work.borrow(py);
                let _scope = ColdCallbackScope::enter(py, &session, &original, work.clone_ref(py))?;
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
        self.intermediate()?
            .as_mut()
            .expect("original intermediate restore")
            .snapshots[index]
            .entered = true;
        let result = {
            let _operation = PhaseOperation::begin(&self.intermediate_snapshot_active)?;
            owners.controller.borrow(py).save_checkpoint(
                py,
                &owners.task.borrow(py),
                &owners.parent.borrow(py),
                streams.bind(py),
                &fresh,
                snapshot_model.as_any(),
            )
        };
        let mut retained = self.intermediate()?;
        let snapshot = &mut retained
            .as_mut()
            .expect("original intermediate restore")
            .snapshots[index];
        match result {
            Ok(saved) => {
                snapshot.saved = Some(saved.clone_ref(py));
                Ok(saved)
            }
            Err(error) => {
                snapshot.error = Some(error.clone_ref(py));
                Err(error)
            }
        }
    }

    fn require_intermediate_region_closed(&self, py: Python<'_>, index: usize) -> PyResult<()> {
        let region = self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .regions[index]
            .clone();
        self.source
            .borrow(py)
            .owner()?
            .require_closed_cold_model_work_region(&region)
            .map_err(xlog_err)
    }

    fn retire_intermediate_source(&self, py: Python<'_>) -> PyResult<()> {
        let (completed, parent) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            if current.session_released {
                return Ok(());
            }
            (
                current.retirement_completed,
                current.retiring_parent.clone_ref(py),
            )
        };
        let session = parent.borrow(py).session.clone_ref(py);
        let task = parent.borrow(py).task_use.clone_ref(py);
        if !completed {
            if self
                .intermediate()?
                .as_ref()
                .expect("original intermediate restore")
                .retirement_entered
            {
                return Err(invalid(
                    "unknown intermediate retirement cannot repeat its original model release",
                ));
            }
            let work = self.intermediate_callback(py, 1, &parent)?;
            let model = self
                .intermediate()?
                .as_ref()
                .expect("original intermediate restore")
                .owners
                .as_ref()
                .expect("actual current selected owners")
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
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .retirement_entered = true;
            let result = (|| {
                let published = parent.borrow(py);
                let original_session = session.borrow(py);
                let issued = task.borrow(py);
                let _reads =
                    ImportReadScope::checkpoint(&original_session, &issued, &published, py)?;
                let original = work.borrow(py);
                let _scope =
                    ColdCallbackScope::enter(py, &original_session, &original, work.clone_ref(py))?;
                let callback = self.model_owner(py, PhaseModelOwner::RetirePrivate)?;
                if !callback.bind(py).call1((model,))?.is_none() {
                    return Err(invalid("original intermediate model retirement must return None after known release"));
                }
                Ok(())
            })();
            if let Err(error) = result {
                self.intermediate()?
                    .as_mut()
                    .expect("original intermediate restore")
                    .error = Some(error.clone_ref(py));
                return Err(error);
            }
            self.require_intermediate_region_closed(py, 1)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .retirement_completed = true;
        }
        if self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .session_release_entered
        {
            return Err(invalid(
                "unknown intermediate native deallocation retains its original terminal custody",
            ));
        }
        session
            .borrow(py)
            .owner()?
            .require_retired_publication(&*parent.borrow(py).lease()?)
            .map_err(xlog_err)?;
        let custody = self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .custody
            .clone();
        if let Some(custody) = custody {
            session
                .borrow(py)
                .owner()?
                .detach_shared_cold_native_work(&custody)
                .map_err(xlog_err)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .custody = None;
            drop(custody);
        }
        if !self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .models_dropped
        {
            let branch = self
                .intermediate()?
                .as_ref()
                .expect("original intermediate restore")
                .branch;
            self.drop_completed_private_execution_owners(branch)?;
            self.drop_completed_trajectory_model_references()?;
            let construction = {
                let mut retained = self.private_restore()?;
                if retained
                    .as_ref()
                    .is_none_or(|owners| owners.session.as_ptr() != session.as_ptr())
                {
                    return Err(invalid(
                        "intermediate retirement cannot discard another private construction",
                    ));
                }
                retained.take()
            };
            drop(construction);
            let original = {
                let mut retained = self.intermediate()?;
                let current = retained.as_mut().expect("original intermediate restore");
                (
                    current.owners.take(),
                    current.previous.take(),
                    current.callbacks[0].take(),
                    current.callbacks[1].take(),
                )
            };
            drop(original);
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .models_dropped = true;
        }
        self.intermediate()?
            .as_mut()
            .expect("original intermediate restore")
            .session_release_entered = true;
        if let Err(error) = session
            .borrow(py)
            .release_retired_publication(py, &parent.borrow(py))
        {
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .error = Some(error.clone_ref(py));
            return Err(error);
        }
        task.borrow(py).state()?.phase = TaskUsePhase::Refused;
        {
            let original = session.borrow(py);
            original.learning_preparing.store(false, Ordering::Release);
            let retained = original
                .learning_transition
                .lock()
                .map_err(|_| invalid("retired intermediate Session lost its phase mutex"))?
                .take();
            drop(retained);
        }
        self.intermediate()?
            .as_mut()
            .expect("original intermediate restore")
            .session_released = true;
        Ok(())
    }

    pub(super) fn execute_intermediate_restore(
        &self,
        py: Python<'_>,
        pending: &Py<Self>,
        branch: &'static str,
    ) -> PyResult<()> {
        self.capture_intermediate_restore(py, branch)?;
        self.decode_intermediate_restore(py)?;
        self.require_intermediate_continuation(py)?;
        let completed_native = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            if current.closed {
                current.restored.as_ref().map(|value| value.clone_ref(py))
            } else {
                None
            }
        };
        if let Some(restored) = completed_native {
            // Original native Closed/Submitted/Completed and observer readback
            // are the only remaining work; never re-enter a sealed region.
            return self.finish_intermediate_restore(py, &restored);
        }
        let (started, ready, ordinal) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            (current.started, current.ready, current.ordinal)
        };
        if !started {
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .started = true;
            self.preparation_inputs
                .resource_observer
                .begin(py, ordinal)?;
            let work = self
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
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .work = Some(work.clone());
            let custody = self
                .source
                .borrow(py)
                .owner()?
                .share_cold_native_work(&work)
                .map_err(xlog_err)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .custody = Some(custody.clone());
            let parent = self
                .intermediate()?
                .as_ref()
                .expect("original intermediate restore")
                .owners
                .as_ref()
                .expect("actual current-state owners")
                .parent
                .clone_ref(py);
            parent
                .borrow(py)
                .session
                .borrow(py)
                .owner()?
                .attach_shared_cold_native_work(&*parent.borrow(py).lease()?, custody)
                .map_err(xlog_err)?;
            let regions = self
                .source
                .borrow(py)
                .owner()?
                .prepare_cold_model_work_regions(&work, 4)
                .map_err(xlog_err)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .regions = regions;
            self.source
                .borrow(py)
                .owner()?
                .begin_cold_model_work(&work)
                .map_err(xlog_err)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .ready = true;
        } else if !ready {
            return Err(invalid(
                "unknown intermediate admission cannot start a replacement interval",
            ));
        }
        let checkpoint = self.intermediate_snapshot(py, 0)?;
        self.require_intermediate_region_closed(py, 0)?;
        self.retire_intermediate_source(py)?;
        let (entered, restored, recipe) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            (
                current.restore_entered,
                current.restored.as_ref().map(|value| value.clone_ref(py)),
                current.recipe.clone(),
            )
        };
        let restored = match restored {
            Some(restored) => restored,
            None if entered => {
                return Err(invalid(
                    "unknown intermediate construction cannot enter a replacement factory",
                ))
            }
            None => {
                let snapshot = self.refresh_snapshot.bind(py).call0()?;
                let authority = AuthoritySnapshot::parse(&ColdValue::read(
                    &snapshot,
                    &mut (16 * 1024 * 1024),
                    0,
                )?)?;
                let task = self.task_use.borrow(py);
                authority.newer_than(&task.state()?.snapshot)?;
                check_learning_grant(&task, &self.grant_reference, &authority)?;
                self.require_intermediate_entry(py)?;
                let domain = task.checkpoint.training_domain.python_value(py)?;
                let checkpoint_limit = self
                    .preparation_inputs
                    .max_checkpoint_bytes
                    .as_ref()
                    .map(|value| value.python_value(py))
                    .transpose()?;
                let total_limit = self
                    .preparation_inputs
                    .max_total_checkpoint_bytes
                    .as_ref()
                    .map(|value| value.python_value(py))
                    .transpose()?;
                self.intermediate()?
                    .as_mut()
                    .expect("original intermediate restore")
                    .restore_entered = true;
                let result = PySemanticTransitionSession::restore_checkpoint_impl(
                    py,
                    checkpoint.bind(py).as_any(),
                    self.source.borrow(py).device_ordinal,
                    &snapshot,
                    self.model_owner(py, PhaseModelOwner::Restore)?.bind(py),
                    domain.bind(py),
                    None,
                    recipe.as_ref(),
                    self.preparation_inputs
                        .resolve_checkpoint
                        .as_ref()
                        .map(|callback| callback.bind(py)),
                    checkpoint_limit.as_ref().map(|value| value.bind(py)),
                    total_limit.as_ref().map(|value| value.bind(py)),
                    Some(self.refresh_snapshot.bind(py)),
                    Some(&task.checkpoint.proposal_expense),
                    Some(&task.checkpoint.checkpoint_sources),
                    Some(pending),
                );
                let mut retained = self.intermediate()?;
                let current = retained.as_mut().expect("original intermediate restore");
                match result {
                    Ok(restored) => {
                        current.restored = Some(restored.clone_ref(py));
                        restored
                    }
                    Err(error) => {
                        current.error = Some(error.clone_ref(py));
                        return Err(error);
                    }
                }
            }
        };
        self.require_intermediate_region_closed(py, 2)?;
        self.intermediate_snapshot(py, 1)?;
        self.require_intermediate_region_closed(py, 3)?;
        self.finish_intermediate_restore(py, &restored)
    }

    fn finish_intermediate_restore(
        &self,
        py: Python<'_>,
        restored: &Py<PySemanticTransitionRestoredCheckpoint>,
    ) -> PyResult<()> {
        let streams = self.preparation_inputs.consumer_streams.python_value(py)?;
        let streams = checkpoint_consumer_streams(streams.bind(py), &mut (16 * 1024 * 1024))?;
        let joined = self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .child_joined;
        if !joined {
            let owners = restored.borrow(py);
            owners
                .session
                .borrow(py)
                .owner()?
                .complete_shared_cold_native_work(&*owners.parent.borrow(py).lease()?, &streams)
                .map_err(xlog_err)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .child_joined = true;
        }
        let (work, closed, report) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            (
                current.work.clone().expect("original restore report"),
                current.closed,
                current.report,
            )
        };
        if !closed {
            self.source
                .borrow(py)
                .owner()?
                .close_cold_model_work(&work)
                .map_err(xlog_err)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .closed = true;
        }
        let report = match report {
            Some(report) => report,
            None => {
                let report = self
                    .source
                    .borrow(py)
                    .owner()?
                    .finish_cold_model_work(
                        &*self.parent.borrow(py).lease()?,
                        &work,
                        &streams,
                        xlog_cuda::SemanticColdModelWorkDisposition::Complete,
                    )
                    .map_err(xlog_err)?;
                self.intermediate()?
                    .as_mut()
                    .expect("original intermediate restore")
                    .report = Some(report);
                report
            }
        };
        let finish = {
            let mut retained = self.intermediate()?;
            let current = retained.as_mut().expect("original intermediate restore");
            let finish = !current.observer_finish_entered;
            current.observer_finish_entered = true;
            finish
        };
        if finish {
            self.preparation_inputs.resource_observer.finish(py)?;
        }
        let peak = self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .backing_peak;
        let peak = if let Some(peak) = peak {
            peak
        } else {
            let peak = self.preparation_inputs.resource_observer.backing_peak(py)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .backing_peak = Some(peak);
            peak
        };
        let work = report
            .native_work
            .checked_add(report.model_work)
            .ok_or_else(|| invalid("intermediate observed work overflowed"))?;
        let (
            instruction,
            ordinal,
            branch,
            budget,
            entered,
            recorded,
            source,
            checkpoint,
            restored_checkpoint,
        ) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            (
                current.instruction.clone(),
                current.ordinal,
                current.branch,
                current.budget,
                current.record_entered,
                current.recorded,
                current.source_identity.clone_ref(py),
                current.snapshots[0]
                    .saved
                    .as_ref()
                    .expect("actual complete current checkpoint")
                    .clone_ref(py),
                current.snapshots[1]
                    .saved
                    .as_ref()
                    .expect("actual complete restored checkpoint")
                    .clone_ref(py),
            )
        };
        if !recorded {
            if entered {
                return Err(invalid(
                    "unknown intermediate history append cannot repeat its original callback",
                ));
            }
            let arguments = PyDict::new(py);
            arguments.set_item("instruction_bytes", PyBytes::new(py, &instruction))?;
            arguments.set_item("operation_ordinal", ordinal)?;
            arguments.set_item("branch", branch)?;
            arguments.set_item("source_parent", source.bind(py))?;
            arguments.set_item(
                "restored_parent",
                restored.borrow(py).parent.borrow(py).identity(py)?,
            )?;
            arguments.set_item("source_checkpoint", checkpoint.bind(py))?;
            arguments.set_item("restored_checkpoint", restored_checkpoint.bind(py))?;
            arguments.set_item("resource_usage", (work, peak, report.model_calls))?;
            self.preparation_inputs
                .resource_observer
                .observation_arguments(py, &arguments)?;
            let callback = self.scientific_owner.bind(py).getattr("record_restore")?;
            {
                let mut retained = self.intermediate()?;
                let current = retained.as_mut().expect("original intermediate restore");
                current.record_entered = true;
                current.budget_exceeded =
                    work > budget[0] || peak > budget[1] || report.model_calls > budget[2];
            }
            callback.call((), Some(&arguments))?;
            self.require_scientific_history(py)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .recorded = true;
        }
        if !self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .released
        {
            self.preparation_inputs.resource_observer.release(py)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .released = true;
        }
        if self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .budget_exceeded
        {
            return Err(invalid("intermediate restore exceeded its original budget; retain the actual recorded expenditure"));
        }
        Ok(())
    }

    pub(super) fn intermediate_execution_input(
        &self,
        py: Python<'_>,
        branch: &'static str,
    ) -> PyResult<Option<(Py<PySemanticTransitionRestoredCheckpoint>, u64)>> {
        let retained = self.intermediate()?;
        let Some(current) = retained.as_ref() else {
            return Ok(None);
        };
        if current.branch != branch {
            return Ok(None);
        }
        if !current.recorded
            || !current.released
            || !current.child_joined
            || !current.session_released
            || current.budget_exceeded
            || current.error.is_some()
        {
            return Err(invalid("private numerical continuation requires known complete intermediate restoration and accounting"));
        }
        Ok(Some((
            current
                .restored
                .as_ref()
                .expect("known intermediate restore")
                .clone_ref(py),
            current
                .ordinal
                .checked_add(1)
                .ok_or_else(|| invalid("intermediate successor position overflowed"))?,
        )))
    }

    pub(super) fn drop_completed_intermediate_restore(&self, branch: &'static str) -> PyResult<()> {
        let original = {
            let mut retained = self.intermediate()?;
            if retained.as_ref().is_some_and(|current| {
                current.branch != branch
                    || !current.recorded
                    || !current.released
                    || current.budget_exceeded
                    || current.error.is_some()
                    || current.previous.is_some()
            }) {
                return Err(invalid(
                    "model retirement cannot discard an unfinished intermediate restoration",
                ));
            }
            retained.take()
        };
        drop(original);
        Ok(())
    }
}
