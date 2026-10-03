//! Cold phase ownership through the canonical checkpoint and restoration path.

use super::*;
use pyo3::types::PyCFunction;
use xlog_cuda::{
    SemanticLearningCopyReset, SemanticLearningPhase, SemanticLearningPhaseTransition,
};

fn phase(value: &str) -> PyResult<SemanticLearningPhase> {
    match value {
        "alignment" => Ok(SemanticLearningPhase::Alignment),
        "fast" => Ok(SemanticLearningPhase::Fast),
        "consolidation" => Ok(SemanticLearningPhase::Consolidation),
        _ => Err(invalid(
            "learning phase must be alignment, fast or consolidation",
        )),
    }
}

fn phase_name(value: SemanticLearningPhase) -> &'static str {
    match value {
        SemanticLearningPhase::Alignment => "alignment",
        SemanticLearningPhase::Fast => "fast",
        SemanticLearningPhase::Consolidation => "consolidation",
    }
}

/// Complete producer recipe; constructing it grants no acceptance or execution.
#[pyclass(
    name = "SemanticLearningPhaseRecipe",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticLearningPhaseRecipe {
    inner: SemanticLearningPhaseTransition,
}

#[pymethods]
impl PySemanticLearningPhaseRecipe {
    #[new]
    #[pyo3(signature = (*, source, target, phase_index, completed_updates_index, views))]
    fn new(
        source: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
        phase_index: &Bound<'_, PyAny>,
        completed_updates_index: &Bound<'_, PyAny>,
        views: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let source = phase(ColdValue::read(source, &mut 128, 0)?.text()?)?;
        let target = phase(ColdValue::read(target, &mut 128, 0)?.text()?)?;
        let phase_index = ColdValue::read(phase_index, &mut 128, 0)?.unsigned()?;
        let completed_updates_index =
            ColdValue::read(completed_updates_index, &mut 128, 0)?.unsigned()?;
        let values = ColdValue::read(views, &mut (16 * 1024 * 1024), 0)?;
        let recipe = values
            .sequence()?
            .iter()
            .map(|value| {
                let fields = value.fields(5)?;
                let role = fields[0].unsigned()?;
                let index = fields[1].unsigned()?;
                let source_role = fields[3].unsigned()?;
                let source_index = fields[4].unsigned()?;
                Ok(match fields[2].text()? {
                    "preserve" if source_role == 0 && source_index == 0 => {
                        SemanticLearningCopyReset::Preserve { role, index }
                    }
                    "zero" if source_role == 0 && source_index == 0 => {
                        SemanticLearningCopyReset::Zero { role, index }
                    }
                    "master-from-effective" if role == 21 => {
                        SemanticLearningCopyReset::MasterFromEffective {
                            index,
                            effective_role: source_role,
                            effective_index: source_index,
                        }
                    }
                    "phase"
                        if role == 23
                            && index == phase_index
                            && source_role == 0
                            && source_index == 0 =>
                    {
                        SemanticLearningCopyReset::Phase { index }
                    }
                    _ => {
                        return Err(invalid(
                            "copy/reset view has an invalid operation or unused source fields",
                        ))
                    }
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: SemanticLearningPhaseTransition {
                source,
                target,
                phase_index,
                completed_updates_index,
                recipe,
                acceptance: Vec::new(),
            },
        })
    }

    #[getter]
    fn recipe_digest(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, self.inner.recipe_digest().as_bytes()).unbind()
    }

    #[getter]
    fn source(&self) -> &'static str {
        phase_name(self.inner.source)
    }

    #[getter]
    fn target(&self) -> &'static str {
        phase_name(self.inner.target)
    }

    #[getter]
    fn phase_index(&self) -> u64 {
        self.inner.phase_index
    }

    #[getter]
    fn completed_updates_index(&self) -> u64 {
        self.inner.completed_updates_index
    }

    #[getter]
    fn views(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        PyTuple::new(
            py,
            self.inner.recipe.iter().map(|operation| match *operation {
                SemanticLearningCopyReset::Preserve { role, index } => {
                    (role, index, "preserve", 0, 0)
                }
                SemanticLearningCopyReset::Zero { role, index } => (role, index, "zero", 0, 0),
                SemanticLearningCopyReset::Phase { index } => (23, index, "phase", 0, 0),
                SemanticLearningCopyReset::MasterFromEffective {
                    index,
                    effective_role,
                    effective_index,
                } => (
                    21,
                    index,
                    "master-from-effective",
                    effective_role,
                    effective_index,
                ),
            }),
        )
        .map(Bound::unbind)
    }
}

enum Completion {
    Prepared,
    Unknown {
        _owner: Py<PyAny>,
        resolve: Py<PyAny>,
    },
    Committed,
    Abandoned,
}

/// Native-issued pending owner. No candidate owner escapes before durable commit.
/// The source Session retains this object even if the caller drops its reference.
#[pyclass(
    name = "SemanticLearningPhaseTransition",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticLearningPhaseTransition {
    source: Py<PySemanticTransitionSession>,
    task_use: Py<PySemanticTransitionTaskUse>,
    parent: Py<PySemanticPublishedParent>,
    candidate: Py<PySemanticTransitionRestoredCheckpoint>,
    source_checkpoint: Vec<u8>,
    checkpoint: Vec<u8>,
    _scientific_owner: Py<PyAny>,
    source_serializer: Py<PyAny>,
    candidate_serializer: Py<PyAny>,
    refresh_snapshot: Py<PyAny>,
    grant_reference: String,
    destination: String,
    completion: Mutex<Completion>,
    operating: AtomicBool,
}

struct PhaseOperation<'a>(&'a AtomicBool);

impl<'a> PhaseOperation<'a> {
    fn begin(flag: &'a AtomicBool) -> PyResult<Self> {
        flag.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid("learning-phase completion cannot be nested from a callback"))?;
        Ok(Self(flag))
    }
}

impl Drop for PhaseOperation<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn check_learning_grant<'a>(
    task: &'a PySemanticTransitionTaskUse,
    reference: &str,
    snapshot: &AuthoritySnapshot,
) -> PyResult<&'a ResolvedGrant> {
    let grant = task
        .authority
        .training
        .iter()
        .find(|grant| grant.reference == reference)
        .ok_or_else(|| invalid("phase transition requires an original task training grant"))?;
    task.authority.check_snapshot(snapshot)?;
    if let CheckpointTaskPhase::Segment(operation) = checkpoint_task_phase(&*task.state()?)? {
        task.authority.check_use(&operation, snapshot, false)?;
    }
    task.authority.check_grants(
        &task.authority.publication,
        snapshot,
        snapshot.observed_at.micros,
    )?;
    task.authority.check_grants(
        std::slice::from_ref(grant),
        snapshot,
        snapshot.observed_at.micros,
    )?;
    if !task.authority.sources_allow_learning
        || grant.learning_purpose.as_deref() != Some("durable-slow-consolidation")
    {
        return Err(invalid(
            "durable phase checkpoint requires its original durable-slow-consolidation grant",
        ));
    }
    Ok(grant)
}

impl PySemanticLearningPhaseTransition {
    fn status_lock(&self) -> PyResult<MutexGuard<'_, Completion>> {
        self.completion
            .lock()
            .map_err(|_| invalid("learning-phase completion owner mutex is poisoned"))
    }

    fn verify(&self, py: Python<'_>, include_candidate: bool) -> PyResult<()> {
        let source = self.source.borrow(py);
        source.require_creator()?;
        let refreshed = self.refresh_snapshot.bind(py).call0()?;
        let snapshot =
            AuthoritySnapshot::parse(&ColdValue::read(&refreshed, &mut (16 * 1024 * 1024), 0)?)?;
        let task = self.task_use.borrow(py);
        let parent = self.parent.borrow(py);
        snapshot.newer_than(&task.state()?.snapshot)?;
        check_learning_grant(&task, &self.grant_reference, &snapshot)?;
        let source_manifest = SemanticCheckpointManifest::decode(&self.source_checkpoint)?;
        self.verify_native(py, &task, &parent, &source_manifest.native, true)?;
        let model = self.source_serializer.bind(py).call0()?;
        require_model_bytes(&model, &source_manifest.model)?;
        self.verify_native(py, &task, &parent, &source_manifest.native, true)?;
        if include_candidate {
            let candidate = self.candidate.borrow(py);
            let issued = candidate.task_use.borrow(py);
            snapshot.newer_than(&issued.state()?.snapshot)?;
            check_learning_grant(&issued, &self.grant_reference, &snapshot)?;
            let acquired = candidate.parent.borrow(py);
            let manifest = SemanticCheckpointManifest::decode(&self.checkpoint)?;
            self.verify_native(py, &issued, &acquired, &manifest.native, false)?;
            let model = self
                .candidate_serializer
                .bind(py)
                .call1((candidate.model.clone_ref(py),))?;
            require_model_bytes(&model, &manifest.model)?;
            self.verify_native(py, &issued, &acquired, &manifest.native, false)?;
            issued.state()?.snapshot = snapshot.clone();
        }
        task.state()?.snapshot = snapshot;
        Ok(())
    }

    fn verify_native(
        &self,
        py: Python<'_>,
        task: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
        expected: &[u8],
        recompute: bool,
    ) -> PyResult<()> {
        let session = task.session.borrow(py);
        let mut owner = session.owner()?;
        task.require_current(&owner)?;
        parent.require_task(py, task)?;
        if !matches!(task.state()?.phase, TaskUsePhase::ArenaPreparing(_)) {
            return Err(invalid(
                "learning-phase owner escaped its retained preparation",
            ));
        }
        let lease = parent.lease()?;
        let actual = if recompute {
            owner.current_recompute_state_material(&lease)
        } else {
            owner.published_state_material(&lease)
        }
        .map_err(xlog_err)?;
        if actual != expected {
            return Err(invalid("retained learning-phase native state changed"));
        }
        Ok(())
    }

    fn activate(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionRestoredCheckpoint>> {
        self.verify(py, true)?;
        let candidate = self.candidate.borrow(py);
        let task = self.task_use.borrow(py);
        let issued = candidate.task_use.borrow(py);
        let source = self.source.borrow(py);
        let successor = candidate.session.borrow(py);
        let mut retention = source
            .learning_transition
            .lock()
            .map_err(|_| invalid("learning-phase retention mutex is poisoned"))?;
        let mut completion = self.status_lock()?;
        let mut old = task.state()?;
        let mut new = issued.state()?;
        if !matches!(old.phase, TaskUsePhase::ArenaPreparing(_))
            || !matches!(new.phase, TaskUsePhase::ArenaPreparing(_))
        {
            return Err(invalid("learning-phase activation lost its private owners"));
        }
        let activated = match &new.phase {
            TaskUsePhase::ArenaPreparing(original) => *original.clone(),
            _ => unreachable!("checked private learning successor"),
        };
        TaskIssuance::issue(Arc::clone(&source.issuance))?;
        old.phase = TaskUsePhase::Refused;
        new.phase = activated;
        *completion = Completion::Committed;
        source.learning_preparing.store(false, Ordering::Release);
        successor.learning_preparing.store(false, Ordering::Release);
        let retained = retention.take();
        drop(old);
        drop(new);
        drop(completion);
        drop(retention);
        drop(retained);
        Ok(self.candidate.clone_ref(py))
    }

    fn abandon(&self, py: Python<'_>) -> PyResult<()> {
        self.verify(py, false)?;
        let source = self.source.borrow(py);
        let candidate = self.candidate.borrow(py);
        let successor = candidate.session.borrow(py);
        let task = self.task_use.borrow(py);
        let issued = candidate.task_use.borrow(py);
        let mut retention = source
            .learning_transition
            .lock()
            .map_err(|_| invalid("learning-phase retention mutex is poisoned"))?;
        let mut completion = self.status_lock()?;
        let mut old = task.state()?;
        let mut new = issued.state()?;
        let resumed = match &old.phase {
            TaskUsePhase::ArenaPreparing(original) => *original.clone(),
            _ => return Err(invalid("source lost its retained learning preparation")),
        };
        let mut owner = successor.owner()?;
        owner.abort();
        old.phase = resumed;
        new.phase = TaskUsePhase::Refused;
        *completion = Completion::Abandoned;
        source.learning_preparing.store(false, Ordering::Release);
        successor.learning_preparing.store(false, Ordering::Release);
        let retained = retention.take();
        drop(owner);
        drop(old);
        drop(new);
        drop(completion);
        drop(retention);
        drop(retained);
        Ok(())
    }
}

fn require_model_bytes(model: &Bound<'_, PyAny>, expected: &[u8]) -> PyResult<()> {
    if !model.is_exact_instance_of::<PyBytes>() || model.cast::<PyBytes>()?.as_bytes() != expected {
        return Err(invalid(
            "retained full model state changed during learning-phase preparation",
        ));
    }
    Ok(())
}

#[pymethods]
impl PySemanticLearningPhaseTransition {
    #[getter]
    fn status(&self) -> PyResult<&'static str> {
        Ok(match &*self.status_lock()? {
            Completion::Prepared => "prepared",
            Completion::Unknown { .. } => "unknown",
            Completion::Committed => "committed",
            Completion::Abandoned => "abandoned",
        })
    }

    #[getter]
    fn checkpoint_sha256(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &Sha256::digest(&self.checkpoint)).unbind()
    }

    /// The trusted durable owner implements commit(destination, bytes, sha256)
    /// and resolve(destination, sha256). Only exact durable readback accepts it.
    fn commit_checkpoint(
        &self,
        py: Python<'_>,
        durable_owner: Py<PyAny>,
    ) -> PyResult<Py<PySemanticTransitionRestoredCheckpoint>> {
        let _operation = PhaseOperation::begin(&self.operating)?;
        self.source.borrow(py).require_creator()?;
        if !matches!(*self.status_lock()?, Completion::Prepared) {
            return Err(invalid(
                "phase checkpoint commit is single-attempt; resolve unknown completion instead",
            ));
        }
        self.verify(py, true)?;
        let commit = durable_owner.bind(py).getattr("commit")?;
        let resolve = durable_owner.bind(py).getattr("resolve")?;
        if !commit.is_callable() || !resolve.is_callable() {
            return Err(invalid(
                "phase checkpoint requires the original durable commit and resolution owner",
            ));
        }
        // Mark unknown before entering external code. Neither callback errors nor
        // later authority failure can permit a second write or release an owner.
        *self.status_lock()? = Completion::Unknown {
            _owner: durable_owner.clone_ref(py),
            resolve: resolve.unbind(),
        };
        let readback = commit.call1((
            &self.destination,
            PyBytes::new(py, &self.checkpoint),
            PyBytes::new(py, &Sha256::digest(&self.checkpoint)),
        ))?;
        require_model_bytes(&readback, &self.checkpoint)?;
        self.activate(py)
    }

    /// Resolve through the SAME retained durable owner; never repeats commit.
    /// None is still unknown, False proves absence, exact bytes prove readback.
    fn resolve_checkpoint(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<Py<PySemanticTransitionRestoredCheckpoint>>> {
        let _operation = PhaseOperation::begin(&self.operating)?;
        self.source.borrow(py).require_creator()?;
        let resolve = match &*self.status_lock()? {
            Completion::Unknown { resolve, .. } => resolve.clone_ref(py),
            _ => return Err(invalid("only an unknown phase checkpoint needs resolution")),
        };
        let result = resolve.bind(py).call1((
            &self.destination,
            PyBytes::new(py, &Sha256::digest(&self.checkpoint)),
        ))?;
        if result.is_none() {
            return Ok(None);
        }
        if result.is_exact_instance_of::<PyBool>() && !result.extract::<bool>()? {
            self.abandon(py)?;
            return Ok(None);
        }
        require_model_bytes(&result, &self.checkpoint)?;
        self.activate(py).map(Some)
    }

    fn abandon_prepared(&self, py: Python<'_>) -> PyResult<()> {
        let _operation = PhaseOperation::begin(&self.operating)?;
        self.source.borrow(py).require_creator()?;
        if !matches!(*self.status_lock()?, Completion::Prepared) {
            return Err(invalid(
                "unknown durable completion must be resolved, not abandoned",
            ));
        }
        self.abandon(py)
    }
}

#[pymethods]
impl PySemanticTransitionController {
    /// Recover the same retained pending owner after a lost Python reference.
    fn pending_learning_phase_transition(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<Py<PySemanticLearningPhaseTransition>>> {
        let source = self.session.borrow(py);
        source.require_creator()?;
        let pending = source
            .learning_transition
            .lock()
            .map_err(|_| invalid("learning-phase retention mutex is poisoned"))?
            .as_ref()
            .map(|pending| pending.clone_ref(py));
        Ok(pending)
    }

    #[pyo3(signature = (task_use, *, parent, recipe, consumer_streams, snapshot, scientific_owner,
        learning_grant_ref, checkpoint_destination, snapshot_model_state, restore_model,
        snapshot_restored_model, refresh_snapshot, resolve_checkpoint=None,
        max_checkpoint_bytes=None, max_total_checkpoint_bytes=None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "the original task, scientific acceptance, durable destination and model owners are independent mandatory inputs"
    )]
    fn prepare_learning_phase_transition(
        &self,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
        parent: Py<PySemanticPublishedParent>,
        recipe: Py<PySemanticLearningPhaseRecipe>,
        consumer_streams: &Bound<'_, PyAny>,
        snapshot: &Bound<'_, PyAny>,
        scientific_owner: Py<PyAny>,
        learning_grant_ref: &Bound<'_, PyAny>,
        checkpoint_destination: &Bound<'_, PyAny>,
        snapshot_model_state: &Bound<'_, PyAny>,
        restore_model: &Bound<'_, PyAny>,
        snapshot_restored_model: &Bound<'_, PyAny>,
        refresh_snapshot: &Bound<'_, PyAny>,
        resolve_checkpoint: Option<&Bound<'_, PyAny>>,
        max_checkpoint_bytes: Option<&Bound<'_, PyAny>>,
        max_total_checkpoint_bytes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PySemanticLearningPhaseTransition>> {
        let source = self.session.borrow(py);
        source.require_creator()?;
        let task = task_use.borrow(py);
        let acquired = parent.borrow(py);
        self.require_issued(&task)?;
        acquired.require_task(py, &task)?;
        let grant_value = ColdValue::read(learning_grant_ref, &mut (16 * 1024 * 1024), 0)?;
        let learning_grant_ref = grant_value.text()?;
        let destination_value =
            ColdValue::read(checkpoint_destination, &mut (16 * 1024 * 1024), 0)?;
        let checkpoint_destination = destination_value.text()?;
        if checkpoint_destination.is_empty()
            || !snapshot_model_state.is_callable()
            || !restore_model.is_callable()
            || !snapshot_restored_model.is_callable()
            || !refresh_snapshot.is_callable()
        {
            return Err(invalid("phase preparation requires its original destination, serializers, restore and authority owner"));
        }
        let initial =
            AuthoritySnapshot::parse(&ColdValue::read(snapshot, &mut (16 * 1024 * 1024), 0)?)?;
        initial.newer_than(&task.state()?.snapshot)?;
        let grant = check_learning_grant(&task, learning_grant_ref, &initial)?;
        let grant_value = task.checkpoint.authority[8]
            .sequence()?
            .iter()
            .find(|value| {
                value.fields(8).is_ok_and(|fields| {
                    fields[0]
                        .text()
                        .is_ok_and(|reference| reference == grant.reference)
                })
            })
            .ok_or_else(|| invalid("original learning grant disappeared from its task capsule"))?
            .clone();
        if source.learning_preparing.swap(true, Ordering::AcqRel) {
            return Err(invalid(
                "Session already retains a learning-phase preparation",
            ));
        }
        if let Err(error) = task
            .state()
            .and_then(|mut state| state.begin_arena_preparation())
        {
            source.learning_preparing.store(false, Ordering::Release);
            return Err(error);
        }
        let mut saved_source: Option<Vec<u8>> = None;
        let mut restored: Option<Py<PySemanticTransitionRestoredCheckpoint>> = None;
        let result = (|| -> PyResult<Py<PySemanticLearningPhaseTransition>> {
            let checkpoint = self.save_checkpoint(
                py,
                &task,
                &acquired,
                consumer_streams,
                snapshot,
                snapshot_model_state,
            )?;
            let checkpoint = checkpoint.bind(py).as_bytes().to_vec();
            saved_source = Some(checkpoint.clone());
            let manifest = SemanticCheckpointManifest::decode(&checkpoint)?;
            if source
                .owner()?
                .current_recompute_state_material(&*acquired.lease()?)
                .map_err(xlog_err)?
                != manifest.native
            {
                return Err(invalid(
                    "phase transition requires the exact successful quiescent recompute parent",
                ));
            }
            let acceptance = scientific_owner.bind(py).getattr("accept_learning_phase")?;
            if !acceptance.is_callable() {
                return Err(invalid(
                    "phase preparation requires the original scientific acceptance owner",
                ));
            }
            let accepted = acceptance.call1((
                parent.clone_ref(py),
                recipe.clone_ref(py),
                PyBytes::new(py, &checkpoint),
                grant_value.python_value(py)?,
                checkpoint_destination,
                snapshot,
            ))?;
            if !accepted.is_exact_instance_of::<PyBytes>()
                || accepted.cast::<PyBytes>()?.as_bytes().is_empty()
            {
                return Err(invalid("scientific owner must return its complete original accepted criteria and result, not an acceptance flag"));
            }
            let mut transition = recipe.borrow(py).inner.clone();
            let mut retained = b"xlog.learning-phase.acceptance.v1\0".to_vec();
            retained.extend_from_slice(&Sha256::digest(&checkpoint));
            retained.extend_from_slice(transition.recipe_digest().as_bytes());
            for bytes in [
                task.authority.canonical.as_slice(),
                grant_value.canonical_bytes().as_slice(),
                initial.canonical.as_slice(),
                checkpoint_destination.as_bytes(),
                accepted.cast::<PyBytes>()?.as_bytes(),
            ] {
                retained.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                retained.extend_from_slice(bytes);
            }
            transition.acceptance = retained;
            let fresh = refresh_snapshot.call0()?;
            let current =
                AuthoritySnapshot::parse(&ColdValue::read(&fresh, &mut (16 * 1024 * 1024), 0)?)?;
            current.newer_than(&initial)?;
            check_learning_grant(&task, learning_grant_ref, &current)?;
            task.state()?.snapshot = initial.clone();
            let candidate = PySemanticTransitionSession::restore_checkpoint_impl(
                py,
                PyBytes::new(py, &checkpoint).as_any(),
                source.device_ordinal,
                &fresh,
                restore_model,
                task.checkpoint.training_domain.python_value(py)?.bind(py),
                None,
                Some(&transition),
                resolve_checkpoint,
                max_checkpoint_bytes,
                max_total_checkpoint_bytes,
                Some(refresh_snapshot),
            )?;
            restored = Some(candidate.clone_ref(py));
            let candidate_model = candidate.borrow(py).model.clone_ref(py);
            let serializer = snapshot_restored_model.clone().unbind();
            let snapshot_candidate =
                PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
                    serializer
                        .bind(args.py())
                        .call1((candidate_model.clone_ref(args.py()),))
                        .map(Bound::unbind)
                })?;
            let fresh = refresh_snapshot.call0()?;
            let checkpoint = {
                let new = candidate.borrow(py);
                let blob = new.controller.borrow(py).save_checkpoint(
                    py,
                    &new.task_use.borrow(py),
                    &new.parent.borrow(py),
                    consumer_streams,
                    &fresh,
                    snapshot_candidate.as_any(),
                )?;
                blob.bind(py).as_bytes().to_vec()
            };
            let pending = Py::new(
                py,
                PySemanticLearningPhaseTransition {
                    source: self.session.clone_ref(py),
                    task_use: task_use.clone_ref(py),
                    parent: parent.clone_ref(py),
                    candidate,
                    source_checkpoint: saved_source.clone().expect("saved phase source"),
                    checkpoint,
                    _scientific_owner: scientific_owner.clone_ref(py),
                    source_serializer: snapshot_model_state.clone().unbind(),
                    candidate_serializer: snapshot_restored_model.clone().unbind(),
                    refresh_snapshot: refresh_snapshot.clone().unbind(),
                    grant_reference: learning_grant_ref.to_owned(),
                    destination: checkpoint_destination.to_owned(),
                    completion: Mutex::new(Completion::Prepared),
                    operating: AtomicBool::new(false),
                },
            )?;
            pending.borrow(py).verify(py, true)?;
            *source
                .learning_transition
                .lock()
                .map_err(|_| invalid("learning-phase retention mutex is poisoned"))? =
                Some(pending.clone_ref(py));
            Ok(pending)
        })();
        if result.is_err() {
            if let Some(candidate) = restored {
                if let Ok(mut owner) = candidate.borrow(py).session.borrow(py).owner() {
                    owner.abort();
                }
                candidate.borrow(py).task_use.borrow(py).state()?.phase = TaskUsePhase::Refused;
                candidate
                    .borrow(py)
                    .session
                    .borrow(py)
                    .learning_preparing
                    .store(false, Ordering::Release);
            }
            let safe = (|| -> PyResult<AuthoritySnapshot> {
                let fresh = refresh_snapshot.call0()?;
                let current = AuthoritySnapshot::parse(&ColdValue::read(
                    &fresh,
                    &mut (16 * 1024 * 1024),
                    0,
                )?)?;
                current.newer_than(&initial)?;
                check_learning_grant(&task, learning_grant_ref, &current)?;
                task.require_current(&*source.owner()?)?;
                if let Some(saved) = &saved_source {
                    let manifest = SemanticCheckpointManifest::decode(saved)?;
                    require_model_bytes(&snapshot_model_state.call0()?, &manifest.model)?;
                    if source
                        .owner()?
                        .published_state_material(&*acquired.lease()?)
                        .map_err(xlog_err)?
                        != manifest.native
                    {
                        return Err(invalid("phase failure changed the original native state"));
                    }
                }
                Ok(current)
            })();
            let mut state = task.state()?;
            if let Ok(snapshot) = safe {
                state.snapshot = snapshot;
                state.finish_arena_preparation()?;
            } else {
                state.phase = TaskUsePhase::Refused;
            }
            source.learning_preparing.store(false, Ordering::Release);
        }
        result
    }
}
