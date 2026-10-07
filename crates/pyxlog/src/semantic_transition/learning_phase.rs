//! Cold phase ownership through the canonical checkpoint and restoration path.

use super::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};
use xlog_cuda::{
    SemanticLearningCopyReset, SemanticLearningPhase, SemanticLearningPhaseTransition,
};

mod phase_record;
use phase_record::{PhaseRecords, RecordKind, RecordLimits};
mod resource_observer;
use resource_observer::ResourceObserver;
#[cfg(feature = "semantic-policy")]
pub(crate) mod cold_model_work;
#[cfg(feature = "semantic-policy")]
use cold_model_work::{ColdCallbackScope, PySemanticColdModelWork};
#[cfg(feature = "semantic-policy")]
pub(super) mod phase_evaluation;
#[cfg(feature = "semantic-policy")]
use phase_evaluation::PhaseEvaluation;
#[cfg(feature = "semantic-policy")]
mod private_restore;
#[cfg(feature = "semantic-policy")]
use private_restore::PrivateTrajectoryStart;
#[cfg(feature = "semantic-policy")]
mod private_execution;
#[cfg(feature = "semantic-policy")]
use private_execution::PrivateExecutionGroup;
#[cfg(feature = "semantic-policy")]
mod private_checkpoint;
#[cfg(feature = "semantic-policy")]
use private_checkpoint::PrivateCheckpoint;
#[cfg(feature = "semantic-policy")]
mod control_retirement;
#[cfg(feature = "semantic-policy")]
use control_retirement::ControlRetirement;

struct PhaseRecordStore {
    pin: Py<PyAny>,
    resolve_issuer: Py<PyAny>,
    commit: Py<PyAny>,
    resolve: Py<PyAny>,
    decode: Py<PyAny>,
}

impl PhaseRecordStore {
    fn capture(owner: &Bound<'_, PyAny>) -> PyResult<Self> {
        let callback = |name| -> PyResult<Py<PyAny>> {
            let callback = owner.getattr(name)?;
            if !callback.is_callable() {
                return Err(invalid(
                    "early phase store requires its original issuer pin/readback and phase record commit/readback callbacks",
                ));
            }
            Ok(callback.unbind())
        };
        Ok(Self {
            pin: callback("pin_phase_issuer")?,
            resolve_issuer: callback("resolve_phase_issuer")?,
            commit: callback("commit_phase_record")?,
            resolve: callback("resolve_phase_record")?,
            decode: callback("decode_phase_instruction")?,
        })
    }
}

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
                        ));
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
                admission: Vec::new(),
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
    Preparing,
    PreparationUnknown,
    Prepared,
    Unknown {
        _owner: Py<PyAny>,
        resolve: Py<PyAny>,
        readback_observed: bool,
    },
    RetirementUnknown,
    Retired,
    Committed,
    Abandoned,
}

struct PreparedCandidate {
    owner: Option<Py<PySemanticTransitionRestoredCheckpoint>>,
    checkpoint: Arc<[u8]>,
    preparation_outcome: Option<Arc<[u8]>>,
}

/// An immutable scientific call boundary. Even an exception is an entered
/// attempt, not permission to repeat the original owner's comparison.
struct ScientificAcceptance {
    checkpoint: Arc<[u8]>,
    history: Arc<[u8]>,
    result: Option<Py<PyAny>>,
    error: Option<PyErr>,
}

/// Immutable inputs retained before the first external issuer/store callback.
/// Continuation consumes these same inputs, never a replacement registration.
struct PreparationInputs {
    source_model: Py<PyAny>,
    execute_phase_instruction: Py<PyAny>,
    resource_observer: ResourceObserver,
    feedback_interventions: Py<PyTuple>,
    consumer_streams: ColdValue,
    snapshot: ColdValue,
    prior_snapshot: AuthoritySnapshot,
    grant: ColdValue,
    frozen_program: Vec<u8>,
    cold_model_work_capacity: usize,
    resolve_checkpoint: Option<Py<PyAny>>,
    max_checkpoint_bytes: Option<ColdValue>,
    max_total_checkpoint_bytes: Option<ColdValue>,
}

/// The two original immutable materials, in real/control order. Reading this
/// closed transport does not decode or reconstruct the scientific program.
fn feedback_materials<'py>(value: &Bound<'py, PyAny>) -> PyResult<[Bound<'py, PyBytes>; 2]> {
    const DOMAIN: &[u8] = b"xlog.learning-feedback-intervention.v1\0";
    if !value.is_exact_instance_of::<PyTuple>() {
        return Err(invalid(
            "feedback interventions require the original exact builtin tuple of real and control bytes",
        ));
    }
    let pair = value.cast::<PyTuple>()?;
    if pair.len() != 2 {
        return Err(invalid(
            "feedback interventions require exactly the original real/control pair",
        ));
    }
    let material = |index| -> PyResult<Bound<'py, PyBytes>> {
        let value = pair.get_item(index)?;
        if !value.is_exact_instance_of::<PyBytes>() {
            return Err(invalid(
                "feedback intervention material requires exact builtin bytes",
            ));
        }
        let bytes = value.cast_into::<PyBytes>()?;
        let raw = bytes.as_bytes();
        if raw.len() != DOMAIN.len() + 1
            || !raw.starts_with(DOMAIN)
            || raw[DOMAIN.len()] != index as u8
        {
            return Err(invalid(
                "feedback interventions require the original real identity and control positive-zero materials in that order",
            ));
        }
        Ok(bytes)
    };
    Ok([material(0)?, material(1)?])
}

impl PreparationInputs {
    fn require_program(&self, py: Python<'_>, scientific_owner: &Py<PyAny>) -> PyResult<()> {
        let original_program = scientific_owner.bind(py).getattr("program_bytes")?;
        require_model_bytes(&original_program, &self.frozen_program)
    }

    fn require_execution_inputs(&self, py: Python<'_>) -> PyResult<()> {
        if self.source_model.bind(py).is_none()
            || !self.execute_phase_instruction.bind(py).is_callable()
        {
            return Err(invalid(
                "phase preparation requires its original source model and instruction execution owner",
            ));
        }
        self.resource_observer.require_original(py)?;
        feedback_materials(self.feedback_interventions.bind(py).as_any())?;
        Ok(())
    }
}

/// Retain each actual owner before the next potentially asynchronous operation.
/// An incomplete construction is not a restored checkpoint or a runnable task.
struct PrivateRestoreOwners {
    session: Py<PySemanticTransitionSession>,
    controller: Option<Py<PySemanticTransitionController>>,
    task_use: Option<Py<PySemanticTransitionTaskUse>>,
    parent: Option<Py<PySemanticPublishedParent>>,
    model: Option<Py<PyAny>>,
}

/// The actual source group and its returned model bytes survive late failures.
/// Neither resource readback nor a cold continuation may repeat this work.
struct SourcePreparation {
    entries: Py<PyTuple>,
    entry_material: Vec<u8>,
    instruction: Vec<u8>,
    physical_memory_limit: u64,
    work_limit: u64,
    model_calls_limit: u64,
    #[cfg(feature = "semantic-policy")]
    cold_model_work: Option<Py<PySemanticColdModelWork>>,
    #[cfg(feature = "semantic-policy")]
    model_work_result: Option<xlog_cuda::SemanticColdModelWorkResult>,
    decoded: Option<Py<PyAny>>,
    decoded_verified: bool,
    model_result: Option<Py<PyAny>>,
    resource_finish_entered: bool,
    physical_peak: Option<u64>,
    verified: bool,
    record_entered: bool,
    recorded: bool,
}

/// Native-issued pending owner. No candidate owner escapes before durable commit.
/// The source Session retains this object even if the caller drops its reference.
#[pyclass(
    name = "SemanticLearningPhaseTransition",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticLearningPhaseTransition {
    source_controller: Py<PySemanticTransitionController>,
    source: Py<PySemanticTransitionSession>,
    task_use: Py<PySemanticTransitionTaskUse>,
    parent: Py<PySemanticPublishedParent>,
    candidate: Mutex<Option<PreparedCandidate>>,
    private_restore: Mutex<Option<PrivateRestoreOwners>>,
    source_checkpoint: Vec<u8>,
    scientific_owner: Py<PyAny>,
    accept_learning_phase: Py<PyAny>,
    scientific_acceptance: Mutex<Option<ScientificAcceptance>>,
    recipe: Py<PySemanticLearningPhaseRecipe>,
    restore_model: Py<PyAny>,
    retire_restored_model: Py<PyAny>,
    source_serializer: Py<PyAny>,
    candidate_serializer: Py<PyAny>,
    refresh_snapshot: Py<PyAny>,
    grant_reference: String,
    destination: String,
    phase_record_owner: Py<PyAny>,
    phase_store: Mutex<Option<PhaseRecordStore>>,
    phase_records: Mutex<PhaseRecords>,
    preparation_inputs: PreparationInputs,
    preparation_entered: AtomicBool,
    source_preparation: Mutex<Option<SourcePreparation>>,
    #[cfg(feature = "semantic-policy")]
    phase_evaluations: Mutex<Vec<PhaseEvaluation>>,
    #[cfg(feature = "semantic-policy")]
    phase_evaluation_active: AtomicBool,
    #[cfg(feature = "semantic-policy")]
    control_evaluations_done: AtomicBool,
    #[cfg(feature = "semantic-policy")]
    real_evaluations_done: AtomicBool,
    #[cfg(feature = "semantic-policy")]
    private_checkpoints: Mutex<Vec<PrivateCheckpoint>>,
    #[cfg(feature = "semantic-policy")]
    private_checkpoint_active: AtomicBool,
    #[cfg(feature = "semantic-policy")]
    control_retirement: Mutex<Option<ControlRetirement>>,
    #[cfg(feature = "semantic-policy")]
    private_trajectory_start: Mutex<Option<PrivateTrajectoryStart>>,
    #[cfg(feature = "semantic-policy")]
    private_execution: Mutex<Option<PrivateExecutionGroup>>,
    #[cfg(feature = "semantic-policy")]
    private_execution_active: AtomicBool,
    candidate_entered: AtomicBool,
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
    let task_phase = {
        let state = task.state()?;
        if matches!(state.phase, TaskUsePhase::Evaluating(_)) {
            state
                .content_handoff_binding()?
                .0
                .map(CheckpointTaskPhase::Segment)
                .ok_or_else(|| {
                    invalid("pending phase evaluation lost its original admitted task operation")
                })?
        } else {
            checkpoint_task_phase(&state)?
        }
    };
    if let CheckpointTaskPhase::Segment(operation) = task_phase {
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

fn verify_phase_native(
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

fn verify_phase_checkpoint(
    py: Python<'_>,
    task: &PySemanticTransitionTaskUse,
    parent: &PySemanticPublishedParent,
    manifest: &SemanticCheckpointManifest,
    saved_snapshot: &AuthoritySnapshot,
) -> PyResult<()> {
    verify_phase_native(py, task, parent, &manifest.native, true)?;
    let session = task.session.borrow(py);
    let mut owner = session.owner()?;
    task.require_current(&owner)?;
    let phase = checkpoint_task_phase(&*task.state()?)?;
    let binding = TaskCheckpointBinding::from_owner(&owner)?;
    if task.checkpoint.encode(saved_snapshot, &phase, &binding)? != manifest.task {
        return Err(invalid(
            "phase checkpoint differs from its original live task capsule",
        ));
    }
    let config = SemanticCheckpointSessionConfig {
        capacities: session.capacities,
        admission_limits: session.admission_limits,
        memory_bytes: session.memory_bytes,
    };
    if config.encode()? != manifest.session {
        return Err(invalid(
            "phase checkpoint differs from its original Session configuration",
        ));
    }
    let (initial_prefill, _) = owner
        .checkpoint_initial_prefill_material(&*parent.lease()?)
        .map_err(xlog_err)?;
    if initial_prefill != manifest.initial_prefill {
        return Err(invalid(
            "phase checkpoint differs from its original initial prefill",
        ));
    }
    Ok(())
}

impl PySemanticLearningPhaseTransition {
    fn singleton_lifecycle_material(entries: &Bound<'_, PyTuple>) -> PyResult<(Vec<u8>, Vec<u8>)> {
        if entries.len() != 1 {
            return Err(invalid(
                "phase lifecycle changed its original singleton group",
            ));
        }
        let entry = entries.get_item(0)?;
        if !entry.is_exact_instance_of::<PyDict>() {
            return Err(invalid(
                "phase lifecycle changed its original exact scheduled entry",
            ));
        }
        let entry = entry.cast::<PyDict>()?;
        if entry.len() != 7 {
            return Err(invalid(
                "phase lifecycle changed its original scheduled fields",
            ));
        }
        for (key, _) in entry.iter() {
            if !key.is_exact_instance_of::<PyString>()
                || ![
                    "operation",
                    "branch",
                    "operation_ordinal",
                    "step_ordinal",
                    "budget",
                    "comparison",
                    "instruction",
                ]
                .contains(&key.cast::<PyString>()?.to_str()?)
            {
                return Err(invalid(
                    "phase lifecycle changed its original exact scheduled keys",
                ));
            }
        }
        let fields = [
            "operation",
            "branch",
            "operation_ordinal",
            "step_ordinal",
            "budget",
            "comparison",
        ]
        .into_iter()
        .map(|name| {
            let value = entry
                .get_item(name)?
                .ok_or_else(|| invalid("phase lifecycle lost an original scheduled field"))?;
            ColdValue::read(&value, &mut (16 * 1024 * 1024), 0)
        })
        .collect::<PyResult<Vec<_>>>()?;
        let instruction = entry
            .get_item("instruction")?
            .ok_or_else(|| invalid("phase lifecycle lost its original instruction bytes"))?;
        if !instruction.is_exact_instance_of::<PyBytes>() {
            return Err(invalid(
                "phase lifecycle changed its original exact instruction bytes",
            ));
        }
        Ok((
            ColdValue::Sequence(fields).canonical_bytes(),
            instruction.cast::<PyBytes>()?.as_bytes().to_vec(),
        ))
    }

    fn source_preparation(&self) -> PyResult<MutexGuard<'_, Option<SourcePreparation>>> {
        self.source_preparation
            .lock()
            .map_err(|_| invalid("original source preparation custody mutex is poisoned"))
    }

    fn source_verified(&self) -> PyResult<bool> {
        Ok(self
            .source_preparation()?
            .as_ref()
            .is_some_and(|source| source.verified && source.model_result.is_some()))
    }

    fn capture_source_operation(&self, py: Python<'_>) -> PyResult<()> {
        if self.source_preparation()?.is_some() {
            return self.require_source_operation(py);
        }
        // Scientific owns schedule projection; the original store owns the one
        // canonical instruction decoder. Native never parses program JSON.
        let entries = self.scientific_owner.bind(py).getattr("next_operation")?;
        if !entries.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "source preparation requires its original complete scheduled group",
            ));
        }
        let entries = entries.cast::<PyTuple>()?;
        if entries.len() != 1 {
            return Err(invalid(
                "source preparation requires its sole original lifecycle operation",
            ));
        }
        let entry = entries.get_item(0)?;
        if !entry.is_exact_instance_of::<PyDict>() {
            return Err(invalid(
                "source preparation requires its original exact scheduled entry",
            ));
        }
        let entry = entry.cast::<PyDict>()?;
        let (material, instruction_material) = Self::singleton_lifecycle_material(entries)?;
        let field = |name| -> PyResult<ColdValue> {
            let value = entry
                .get_item(name)?
                .ok_or_else(|| invalid("source lifecycle entry lost an original field"))?;
            ColdValue::read(&value, &mut (16 * 1024 * 1024), 0)
        };
        if field("operation")?.text()? != "prepare"
            || field("branch")?.text()? != "source"
            || field("operation_ordinal")?.unsigned()? != 0
            || field("step_ordinal")?.unsigned()? != 0
        {
            return Err(invalid(
                "source preparation differs from its original first scheduled operation",
            ));
        }
        let instruction = entry
            .get_item("instruction")?
            .ok_or_else(|| invalid("source preparation lost its original instruction bytes"))?;
        if !instruction.is_exact_instance_of::<PyBytes>()
            || instruction.cast::<PyBytes>()?.as_bytes().is_empty()
        {
            return Err(invalid(
                "source preparation requires exact original nonempty instruction bytes",
            ));
        }
        let budget = field("budget")?;
        let limits = budget
            .fields(3)?
            .iter()
            .map(ColdValue::unsigned)
            .collect::<PyResult<Vec<_>>>()?;
        if limits[1] == 0 {
            return Err(invalid(
                "source preparation requires its original positive physical memory limit",
            ));
        }
        *self.source_preparation()? = Some(SourcePreparation {
            entries: entries.clone().unbind(),
            entry_material: material,
            instruction: instruction_material,
            physical_memory_limit: limits[1],
            work_limit: limits[0],
            model_calls_limit: limits[2],
            #[cfg(feature = "semantic-policy")]
            cold_model_work: None,
            #[cfg(feature = "semantic-policy")]
            model_work_result: None,
            decoded: None,
            decoded_verified: false,
            model_result: None,
            resource_finish_entered: false,
            physical_peak: None,
            verified: false,
            record_entered: false,
            recorded: false,
        });
        let decode = self
            .store()?
            .as_ref()
            .ok_or_else(|| invalid("source preparation lost its original lifecycle decoder"))?
            .decode
            .clone_ref(py);
        let decoded = decode.bind(py).call1((entry,))?;
        // Keep the actual return before checking its projection. A replacement
        // instruction owner is never registered to repair a late failure.
        self.source_preparation()?
            .as_mut()
            .expect("retained original source group")
            .decoded = Some(decoded.clone().unbind());
        if !decoded.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "original source decoder requires its exact preparation tuple including cold model capacity",
            ));
        }
        let decoded = decoded.cast::<PyTuple>()?;
        if decoded.len() != 4
            || ColdValue::read(&decoded.get_item(0)?, &mut 128, 0)?.text()? != "prepare"
            || !decoded.get_item(1)?.is_none()
            || !decoded.get_item(2)?.is_none()
            || ColdValue::read(&decoded.get_item(3)?, &mut 128, 0)?.unsigned()?
                != self.preparation_inputs.cold_model_work_capacity as u64
        {
            return Err(invalid(
                "source decoder changed the original preparation lifecycle",
            ));
        }
        self.source_preparation()?
            .as_mut()
            .expect("retained original source group")
            .decoded_verified = true;
        self.require_source_operation(py)
    }

    fn require_source_operation(&self, py: Python<'_>) -> PyResult<()> {
        let retained = self.source_preparation()?;
        let source = retained
            .as_ref()
            .ok_or_else(|| invalid("source preparation lost its original scheduled group"))?;
        let (material, instruction) = Self::singleton_lifecycle_material(source.entries.bind(py))?;
        if source.decoded.is_none()
            || !source.decoded_verified
            || material != source.entry_material
            || instruction != source.instruction
        {
            return Err(invalid(
                "source preparation changed or lost its original decoded group",
            ));
        }
        Ok(())
    }

    /// Admit preparation using the same captured inputs and store. Unknown
    /// native/model execution never grants rights to repeat that execution.
    fn prepare_candidate(&self, py: Python<'_>, pending: &Py<Self>) -> PyResult<()> {
        let source = self.source.borrow(py);
        source.require_creator()?;
        let admission_needed = self.records()?.preparation_admission_needed()?;
        let entered = self.preparation_entered.load(Ordering::Acquire);
        if entered && (!self.source_verified()? || self.candidate_entered.load(Ordering::Acquire)) {
            return Err(invalid(
                "the original preparation has already entered execution; retain and resolve its outcome",
            ));
        }
        let task = self.task_use.borrow(py);
        let acquired = self.parent.borrow(py);
        let checkpoint = self.source_checkpoint.as_slice();
        let manifest = SemanticCheckpointManifest::decode(checkpoint)?;
        let (_, saved_snapshot, _, _) = TaskCheckpointSeed::decode(&manifest.task)?;
        let inputs = &self.preparation_inputs;
        inputs.require_execution_inputs(py)?;
        let stream_values = inputs.consumer_streams.python_value(py)?;
        let consumer_streams = stream_values.bind(py);
        let streams = checkpoint_consumer_streams(consumer_streams, &mut (16 * 1024 * 1024))?;
        let initial = AuthoritySnapshot::parse(&inputs.snapshot)?;
        let prior_snapshot = &inputs.prior_snapshot;
        let scientific_owner = &self.scientific_owner;
        let snapshot_model_state = self.source_serializer.bind(py);
        let refresh_snapshot = self.refresh_snapshot.bind(py);
        let learning_grant_ref = self.grant_reference.as_str();
        // Continuation has a distinct, fresh authority admission. A confirmed
        // old issuer or record cannot extend the original data-use grants.
        let refreshed = refresh_snapshot.call0()?;
        let authority =
            AuthoritySnapshot::parse(&ColdValue::read(&refreshed, &mut (16 * 1024 * 1024), 0)?)?;
        authority.newer_than(&initial)?;
        authority.newer_than(&task.state()?.snapshot)?;
        check_learning_grant(&task, learning_grant_ref, &authority)?;
        inputs.require_program(py, scientific_owner)?;
        if admission_needed {
            let inputs = Arc::clone(&self.records()?.inputs);
            self.append_phase_record(py, RecordKind::Admission, &inputs)?;
        }
        self.records()?.require_preparation_admission()?;
        self.capture_source_operation(py)?;
        // The original store callbacks can fail or change external authority.
        // Durable admission does not waive verification before execution.
        inputs.require_execution_inputs(py)?;
        inputs.require_program(py, scientific_owner)?;
        self.require_source_operation(py)?;
        let refreshed = refresh_snapshot.call0()?;
        let admitted_authority =
            AuthoritySnapshot::parse(&ColdValue::read(&refreshed, &mut (16 * 1024 * 1024), 0)?)?;
        admitted_authority.newer_than(&authority)?;
        admitted_authority.newer_than(&task.state()?.snapshot)?;
        check_learning_grant(&task, learning_grant_ref, &admitted_authority)?;
        // Refresh is external code too: recheck the retained inputs after it,
        // before marking execution entered or touching native/model state.
        inputs.require_execution_inputs(py)?;
        inputs.require_program(py, scientific_owner)?;
        self.require_source_operation(py)?;
        // No native/model preparation is entered before the original signed
        // admission is known durable. This marker is never cleared on failure.
        if !entered {
            self.preparation_entered
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid("the original preparation has already entered execution; retain and resolve its outcome"))?;
            inputs.resource_observer.begin(py, 0)?;
            {
                if source.importing.load(Ordering::Acquire)
                    || source.recording.load(Ordering::Acquire)
                    || source.retiring.load(Ordering::Acquire)
                {
                    return Err(invalid(
                        "phase preparation cannot overlap recording, import or retirement",
                    ));
                }
                let mut owner = source.owner()?;
                task.require_current(&owner)?;
                owner
                    .quiesce_published_reader(&*acquired.lease()?, &streams)
                    .map_err(xlog_err)?;
            }
            #[cfg(feature = "semantic-policy")]
            {
                let admission = self.records()?.confirmed_admission()?;
                let inner = source
                    .owner()?
                    .prepare_cold_model_work(
                        &*acquired.lease()?,
                        inputs.cold_model_work_capacity,
                        0,
                        admission,
                    )
                    .map_err(xlog_err)?;
                let work = Py::new(
                    py,
                    PySemanticColdModelWork {
                        parent: self.parent.clone_ref(py),
                        reader: self.parent.clone_ref(py),
                        inner,
                        region: None,
                        active: AtomicBool::new(false),
                    },
                )?;
                self.source_preparation()?
                    .as_mut()
                    .expect("retained original source group")
                    .cold_model_work = Some(work);
            }
            verify_phase_checkpoint(py, &task, &acquired, &manifest, &saved_snapshot)?;
            {
                let _reads = ImportReadScope::checkpoint(&source, &task, &acquired, py)?;
                #[cfg(feature = "semantic-policy")]
                let callback_work = self
                    .source_preparation()?
                    .as_ref()
                    .expect("retained source group")
                    .cold_model_work
                    .as_ref()
                    .expect("issued original cold work")
                    .clone_ref(py);
                #[cfg(feature = "semantic-policy")]
                let work = callback_work.borrow(py);
                #[cfg(feature = "semantic-policy")]
                let _callback =
                    ColdCallbackScope::enter(py, &source, &work, callback_work.clone_ref(py))?;
                let result = snapshot_model_state.call0();
                #[cfg(feature = "semantic-policy")]
                if result.is_err() {
                    source.owner()?.fail_cold_model_work(&work.inner);
                }
                let result = result?;
                self.source_preparation()?
                    .as_mut()
                    .expect("retained original source group")
                    .model_result = Some(result.clone().unbind());
                require_model_bytes(&result, &manifest.model)?;
            }
            verify_phase_checkpoint(py, &task, &acquired, &manifest, &saved_snapshot)?;
            if task.state()?.snapshot.canonical != prior_snapshot.canonical {
                return Err(invalid(
                    "phase source authority changed during model verification",
                ));
            }
            // Join again after the actual serializer result and all native source
            // verification. finish owns the original end before its cold delivery.
            source
                .owner()?
                .quiesce_published_reader(&*acquired.lease()?, &streams)
                .map_err(xlog_err)?;
            self.source_preparation()?
                .as_mut()
                .expect("retained original source group")
                .verified = true;
        }
        #[cfg(feature = "semantic-policy")]
        {
            let (work, result) = {
                let retained = self.source_preparation()?;
                let source = retained.as_ref().expect("retained original source group");
                (
                    source
                        .cold_model_work
                        .as_ref()
                        .expect("issued original cold work")
                        .clone_ref(py),
                    source.model_work_result,
                )
            };
            let result = if let Some(result) = result {
                result
            } else {
                let mut owner = source.owner()?;
                let lease = acquired.lease()?;
                let work = work.borrow(py);
                // Entering preparation is not proof that this report reached
                // submission. Native dispatches from the original report's
                // actual state without re-entering the model callback.
                owner
                    .finish_cold_model_work(&lease, &work.inner, &streams)
                    .map_err(xlog_err)?
            };
            // Retain the actual components before refusal. Other source S
            // still requires its genuine producers, never substituted zeroes.
            let mut retained = self.source_preparation()?;
            let source = retained.as_mut().expect("retained original source group");
            source.model_work_result = Some(result);
        }
        let finish_needed = {
            let mut retained = self.source_preparation()?;
            let source = retained.as_mut().expect("retained original source group");
            let needed = !source.resource_finish_entered;
            source.resource_finish_entered = true;
            needed
        };
        if finish_needed {
            inputs.resource_observer.finish(py)?;
        }
        // Missing physical producer/baseline/coverage/allocator completion is
        // not zero or a reservation. Keep the completed source result and all
        // original owners; explicit continuation only reads this same interval.
        let saved_peak = self
            .source_preparation()?
            .as_ref()
            .expect("retained original source group")
            .physical_peak;
        let peak = match saved_peak {
            Some(peak) => peak,
            None => {
                let peak = inputs.resource_observer.physical_peak(py)?;
                self.source_preparation()?
                    .as_mut()
                    .expect("retained original source group")
                    .physical_peak = Some(peak);
                peak
            }
        };
        #[cfg(feature = "semantic-policy")]
        {
            let retained = self.source_preparation()?;
            let source = retained.as_ref().expect("retained original source group");
            let result = source
                .model_work_result
                .expect("completed original model component");
            let recorded_work = result
                .model_work
                .checked_add(result.native_work)
                .ok_or_else(|| {
                    invalid("source model and semantic work overflowed its original expense")
                })?;
            if recorded_work > source.work_limit || result.model_calls > source.model_calls_limit {
                return Err(invalid("source model and semantic work exceeded its original operation budget; retain the actual result"));
            }
        }
        let memory_limit = self
            .source_preparation()?
            .as_ref()
            .expect("retained original source group")
            .physical_memory_limit;
        if peak > memory_limit {
            return Err(invalid("source preparation exceeded its original physical memory budget; retain the actual result without admitting private work"));
        }
        #[cfg(feature = "semantic-policy")]
        {
            self.record_source_preparation(py, peak)?;
            self.execute_source_evaluations(py)?;
        }
        #[cfg(feature = "semantic-policy")]
        {
            self.execute_control_branch(py, pending)?;
            self.prepare_private_trajectory(py, pending, "real")?;
            self.execute_private_group(py, "real")?;
            self.execute_private_evaluations(py, "real")?;
            self.execute_private_checkpoint(py, "real")?;
            self.retain_final_checkpoint_candidate(py)?;
            return self.finish_preparation(py);
        }
        #[cfg(not(feature = "semantic-policy"))]
        {
            let _ = pending;
            Err(invalid(
                "private scientific trajectory execution requires semantic-policy",
            ))
        }
    }

    fn preparation_outcome(&self) -> PyResult<Arc<[u8]>> {
        self.candidate
            .lock()
            .map_err(|_| invalid("learning-phase candidate owner mutex is poisoned"))?
            .as_ref()
            .and_then(|candidate| candidate.preparation_outcome.as_ref())
            .map(Arc::clone)
            .ok_or_else(|| {
                invalid("preparation continuation lacks its original complete retained outcome; unknown native/model work cannot be repeated")
            })
    }

    /// Finish only the already known cold result. Its checkpoint and history
    /// are immutable; neither the factory nor scientific acceptance is repeated.
    fn finish_preparation(&self, py: Python<'_>) -> PyResult<()> {
        self.finish_scientific_acceptance(py)?;
        let outcome = self.preparation_outcome()?;
        // An attempted signed write belongs exclusively to its resolver, even
        // when the candidate and the unsigned outcome are already known.
        self.records()?.require_preparation_admission()?;
        self.verify(py, true)?;
        self.append_phase_record(py, RecordKind::PreparationOutcome, &outcome)?;
        // The complete checkpoint now owns all of the actual restored
        // handles. Release only the redundant construction references.
        let construction = self.private_restore()?.take();
        drop(construction);
        *self.status_lock()? = Completion::Prepared;
        Ok(())
    }

    fn acceptance(&self) -> PyResult<MutexGuard<'_, Option<ScientificAcceptance>>> {
        self.scientific_acceptance
            .lock()
            .map_err(|_| invalid("original scientific acceptance custody mutex is poisoned"))
    }

    fn finish_scientific_acceptance(&self, py: Python<'_>) -> PyResult<()> {
        if self.acceptance()?.is_none() {
            self.records()?.require_preparation_admission()?;
            self.preparation_inputs
                .require_program(py, &self.scientific_owner)?;
            // The scientific owner validates its entire measured prefix when
            // projecting the next group. Delivery is the sole remaining
            // operation at acceptance; a prepared candidate is not that proof.
            let next = self.scientific_owner.bind(py).getattr("next_operation")?;
            if !next.is_exact_instance_of::<PyTuple>() {
                return Err(invalid(
                    "final acceptance requires the original complete prefix before delivery",
                ));
            }
            let next = next.cast::<PyTuple>()?;
            if next.is_empty() || next.len() != 1 {
                return Err(invalid("final acceptance cannot precede complete original source, control and real execution"));
            }
            let (next_material, _) = Self::singleton_lifecycle_material(next)?;
            let next_material = ColdValue::from_canonical_bytes(&next_material)?;
            let fields = next_material.fields(6)?;
            if fields[0].text()? != "delivery"
                || fields[1].text()? != "real"
                || fields[3].unsigned()? != 0
            {
                return Err(invalid("final acceptance cannot precede complete original source, control and real execution"));
            }
            self.verify(py, true)?;
            let (candidate, checkpoint) = self.candidate(py)?;
            let history = self.scientific_owner.bind(py).getattr("history_bytes")?;
            if !history.is_exact_instance_of::<PyBytes>()
                || history.cast::<PyBytes>()?.as_bytes().is_empty()
            {
                return Err(invalid(
                    "scientific acceptance requires the original complete history bytes",
                ));
            }
            let history: Arc<[u8]> = Arc::from(history.cast::<PyBytes>()?.as_bytes());
            let latest = self.refresh_snapshot.bind(py).call0()?;
            let snapshot =
                AuthoritySnapshot::parse(&ColdValue::read(&latest, &mut (16 * 1024 * 1024), 0)?)?;
            {
                let task = self.task_use.borrow(py);
                snapshot.newer_than(&task.state()?.snapshot)?;
                check_learning_grant(&task, &self.grant_reference, &snapshot)?;
                let restored = candidate.borrow(py);
                let task = restored.task_use.borrow(py);
                snapshot.newer_than(&task.state()?.snapshot)?;
                check_learning_grant(&task, &self.grant_reference, &snapshot)?;
            }
            self.preparation_inputs
                .require_program(py, &self.scientific_owner)?;
            require_model_bytes(
                &self.scientific_owner.bind(py).getattr("history_bytes")?,
                &history,
            )?;
            let (actual_parent, actual_model) = {
                let restored = candidate.borrow(py);
                (restored.parent.clone_ref(py), restored.model.clone_ref(py))
            };
            let source_manifest = SemanticCheckpointManifest::decode(&self.source_checkpoint)?;
            verify_phase_native(
                py,
                &self.task_use.borrow(py),
                &self.parent.borrow(py),
                &source_manifest.native,
                true,
            )?;
            {
                let restored = candidate.borrow(py);
                let final_manifest = SemanticCheckpointManifest::decode(&checkpoint)?;
                verify_phase_native(
                    py,
                    &restored.task_use.borrow(py),
                    &restored.parent.borrow(py),
                    &final_manifest.native,
                    false,
                )?;
            }
            // Complete all fallible argument projection before marking the
            // call entered. Snapshot is the verified immutable cold value.
            let arguments = (
                self.parent.clone_ref(py),
                self.recipe.clone_ref(py),
                PyBytes::new(py, &self.source_checkpoint),
                self.preparation_inputs.grant.python_value(py)?,
                &self.destination,
                ColdValue::from_canonical_bytes(&snapshot.canonical)?.python_value(py)?,
                actual_parent,
                actual_model,
                PyBytes::new(py, &checkpoint),
                PyBytes::new(py, &self.preparation_inputs.frozen_program),
                PyBytes::new(py, &history),
            )
                .into_pyobject(py)?;
            // Retain the exact checkpoint/history before entering external
            // code. No phase/native/task mutex survives into the owner.
            *self.acceptance()? = Some(ScientificAcceptance {
                checkpoint: Arc::clone(&checkpoint),
                history: Arc::clone(&history),
                result: None,
                error: None,
            });
            let result = self.accept_learning_phase.bind(py).call1(arguments);
            let mut retained = self.acceptance()?;
            let retained = retained.as_mut().expect("retained before scientific entry");
            match result {
                Ok(result) => retained.result = Some(result.unbind()),
                Err(error) => {
                    retained.error = Some(error.clone_ref(py));
                    return Err(error);
                }
            }
        }
        let (checkpoint, history, result) = {
            let retained = self.acceptance()?;
            let retained = retained
                .as_ref()
                .expect("original scientific call retained");
            if let Some(error) = &retained.error {
                return Err(error.clone_ref(py));
            }
            (
                Arc::clone(&retained.checkpoint),
                Arc::clone(&retained.history),
                retained
                    .result
                    .as_ref()
                    .ok_or_else(|| {
                        invalid(
                            "unknown scientific acceptance cannot repeat its original comparison",
                        )
                    })?
                    .clone_ref(py),
            )
        };
        let accepted = result.bind(py);
        if !accepted.is_exact_instance_of::<PyBytes>()
            || accepted.cast::<PyBytes>()?.as_bytes().is_empty()
        {
            return Err(invalid("scientific owner must return its complete original accepted criteria and result, not an acceptance flag"));
        }
        self.preparation_inputs
            .require_program(py, &self.scientific_owner)?;
        require_model_bytes(
            &self.scientific_owner.bind(py).getattr("history_bytes")?,
            &history,
        )?;
        if self.candidate_checkpoint()?.as_ref() != checkpoint.as_ref() {
            return Err(invalid(
                "scientific acceptance lost its original full final checkpoint",
            ));
        }
        let accepted = accepted.cast::<PyBytes>()?.as_bytes();
        let outcome_length = checkpoint
            .len()
            .checked_add(history.len())
            .and_then(|length| length.checked_add(accepted.len()))
            .and_then(|length| length.checked_add(24))
            .ok_or_else(|| invalid("complete phase preparation outcome size overflowed"))?;
        self.records()?.check_payload_length(outcome_length)?;
        let mut outcome = Vec::with_capacity(outcome_length);
        // Acceptance is outside Q and cannot change the bytes it accepted.
        for material in [checkpoint.as_ref(), history.as_ref(), accepted] {
            outcome.extend_from_slice(&(material.len() as u64).to_le_bytes());
            outcome.extend_from_slice(material);
        }
        let mut candidate = self
            .candidate
            .lock()
            .map_err(|_| invalid("learning-phase candidate owner mutex is poisoned"))?;
        let candidate = candidate
            .as_mut()
            .ok_or_else(|| invalid("scientific acceptance lost its original candidate"))?;
        if let Some(original) = &candidate.preparation_outcome {
            if original.as_ref() != outcome.as_slice() {
                return Err(invalid(
                    "scientific acceptance changed its retained complete outcome",
                ));
            }
        } else {
            candidate.preparation_outcome = Some(outcome.into());
        }
        Ok(())
    }

    pub(super) fn require_restore_admission(
        &self,
        py: Python<'_>,
        checkpoint: &[u8],
        transition: Option<&SemanticLearningPhaseTransition>,
    ) -> PyResult<()> {
        self.source.borrow(py).require_creator()?;
        #[cfg(feature = "semantic-policy")]
        return self.require_private_restore_admission(py, checkpoint, transition);
        #[cfg(not(feature = "semantic-policy"))]
        {
            let _ = (checkpoint, transition);
            Err(invalid(
                "private scientific trajectory execution requires semantic-policy",
            ))
        }
    }

    pub(super) fn restore_allocation_domain(
        &self,
        py: Python<'_>,
        device_ordinal: usize,
        memory_bytes: u64,
    ) -> PyResult<CheckpointAllocationDomain> {
        let source = self.source.borrow(py);
        source.require_creator()?;
        if !matches!(*self.status_lock()?, Completion::Preparing)
            || self.private_restore()?.is_some()
            || source.device_ordinal != device_ordinal
            || source.memory_bytes != memory_bytes
        {
            return Err(invalid(
                "private phase allocation requires its original unused preparation and unchanged Session bounds",
            ));
        }
        self.records()?.require_preparation_admission()?;
        let (provider, domain) = source
            .owner()?
            .checkpoint_allocation_domain()
            .map_err(xlog_err)?;
        Ok(CheckpointAllocationDomain {
            provider,
            domain,
            #[cfg(feature = "semantic-policy")]
            cold_work: Some(self.private_restore_native_work(py)?),
        })
    }

    fn finish_preparation_readback(&self, py: Python<'_>) -> PyResult<&'static str> {
        if !self.records()?.preparation_outcome_known {
            return Ok("unknown");
        }
        // The complete candidate was retained and verified before issuing this
        // outcome. Readback never re-enters its factory or preparation callbacks.
        self.candidate(py)?;
        let construction = self.private_restore()?.take();
        drop(construction);
        *self.status_lock()? = Completion::Prepared;
        Ok("prepared")
    }

    /// Read only the same retained attempt. Knowing admission durability alone
    /// never authorizes replay of a preparation whose execution is unknown.
    fn resolve_preparation_record(&self, py: Python<'_>) -> PyResult<&'static str> {
        if !matches!(*self.status_lock()?, Completion::PreparationUnknown) {
            return Err(invalid(
                "only unknown early phase preparation requires phase record resolution",
            ));
        }
        let (phase_id, issuer, pinned, pin_attempted) = {
            let records = self.records()?;
            (
                records.phase_id,
                records.issuer(),
                records.issuer_pinned,
                records.pin_attempted,
            )
        };
        if !pin_attempted {
            return Err(invalid(
                "early phase preparation has no attempted issuer pin",
            ));
        }
        if !pinned {
            let resolve = self
                .store()?
                .as_ref()
                .ok_or_else(|| invalid("phase issuer resolution lost its original store"))?
                .resolve_issuer
                .clone_ref(py);
            let readback = resolve
                .bind(py)
                .call1((&self.destination, PyBytes::new(py, &phase_id)))?;
            if readback.is_none() {
                return Ok("unknown");
            }
            require_model_bytes(&readback, &issuer)?;
            self.records()?.issuer_pinned = true;
        }
        let attempt = self
            .records()?
            .attempt
            .as_ref()
            .map(|attempt| (attempt.ordinal, attempt.digest, attempt.readback_observed));
        let Some((ordinal, digest, observed)) = attempt else {
            // A confirmed pin or a completed admission is not proof of a
            // completed factory/native restoration. Do not resume either owner.
            return self.finish_preparation_readback(py);
        };
        let resolve = self
            .store()?
            .as_ref()
            .ok_or_else(|| invalid("phase record resolution lost its original store"))?
            .resolve
            .clone_ref(py);
        let readback = resolve.bind(py).call1((
            &self.destination,
            PyBytes::new(py, &phase_id),
            ordinal,
            PyBytes::new(py, &digest),
        ))?;
        if readback.is_none() {
            return Ok("unknown");
        }
        if readback.is_exact_instance_of::<PyBool>() && !readback.extract::<bool>()? {
            return Err(invalid(if observed {
                "phase resolver contradicted an observed exact durable record; retain the original pending owners"
            } else {
                "absence of a phase record does not resolve execution; retain the original pending owners"
            }));
        }
        {
            let mut records = self.records()?;
            records.confirm(&readback)?;
            records.advance()?;
        }
        self.finish_preparation_readback(py)
    }

    fn records(&self) -> PyResult<MutexGuard<'_, PhaseRecords>> {
        self.phase_records
            .lock()
            .map_err(|_| invalid("native phase record custody mutex is poisoned"))
    }

    fn store(&self) -> PyResult<MutexGuard<'_, Option<PhaseRecordStore>>> {
        self.phase_store
            .lock()
            .map_err(|_| invalid("original early phase store mutex is poisoned"))
    }

    fn pin_issuer(
        &self,
        py: Python<'_>,
        pending: &Py<PySemanticLearningPhaseTransition>,
    ) -> PyResult<()> {
        let (phase_id, issuer) = {
            let mut records = self.records()?;
            if records.pin_attempted {
                return Err(invalid(
                    "native phase issuer pin is single-attempt; read back the original pin instead",
                ));
            }
            records.pin_attempted = true;
            (records.phase_id, records.issuer())
        };
        let pin = self
            .store()?
            .as_ref()
            .ok_or_else(|| invalid("native phase issuer lost its original early store"))?
            .pin
            .clone_ref(py);
        let readback = pin.bind(py).call1((
            &self.destination,
            PyBytes::new(py, &phase_id),
            pending.clone_ref(py),
        ))?;
        require_model_bytes(&readback, &issuer)?;
        self.records()?.issuer_pinned = true;
        Ok(())
    }

    fn append_phase_record(
        &self,
        py: Python<'_>,
        kind: RecordKind,
        payload: &[u8],
    ) -> PyResult<()> {
        let (phase_id, ordinal, bytes, digest) = {
            let mut records = self.records()?;
            records.begin(kind, payload)?;
            let attempt = records.attempt.as_ref().expect("retained original record");
            (
                records.phase_id,
                attempt.ordinal,
                Arc::clone(&attempt.bytes),
                attempt.digest,
            )
        };
        let commit = self
            .store()?
            .as_ref()
            .ok_or_else(|| invalid("phase record lost its original early store"))?
            .commit
            .clone_ref(py);
        let readback = commit.bind(py).call1((
            &self.destination,
            PyBytes::new(py, &phase_id),
            ordinal,
            PyBytes::new(py, &bytes),
            PyBytes::new(py, &digest),
        ))?;
        let mut records = self.records()?;
        records.confirm(&readback)?;
        records.advance()
    }

    fn status_lock(&self) -> PyResult<MutexGuard<'_, Completion>> {
        self.completion
            .lock()
            .map_err(|_| invalid("learning-phase completion owner mutex is poisoned"))
    }

    fn candidate(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Py<PySemanticTransitionRestoredCheckpoint>, Arc<[u8]>)> {
        let candidate = self
            .candidate
            .lock()
            .map_err(|_| invalid("learning-phase candidate owner mutex is poisoned"))?;
        let candidate = candidate.as_ref().ok_or_else(|| {
            invalid("learning-phase preparation has no complete verified candidate checkpoint")
        })?;
        Ok((
            candidate
                .owner
                .as_ref()
                .ok_or_else(|| invalid("private candidate consumers and owners have retired"))?
                .clone_ref(py),
            Arc::clone(&candidate.checkpoint),
        ))
    }

    fn candidate_checkpoint(&self) -> PyResult<Arc<[u8]>> {
        let candidate = self
            .candidate
            .lock()
            .map_err(|_| invalid("learning-phase candidate owner mutex is poisoned"))?;
        candidate
            .as_ref()
            .map(|candidate| Arc::clone(&candidate.checkpoint))
            .ok_or_else(|| {
                invalid("learning-phase preparation has no complete candidate checkpoint")
            })
    }

    fn private_restore(&self) -> PyResult<MutexGuard<'_, Option<PrivateRestoreOwners>>> {
        self.private_restore
            .lock()
            .map_err(|_| invalid("learning-phase private restore owner mutex is poisoned"))
    }

    pub(super) fn retain_restore_session(
        &self,
        py: Python<'_>,
        session: &Py<PySemanticTransitionSession>,
    ) -> PyResult<()> {
        if !matches!(*self.status_lock()?, Completion::Preparing) {
            return Err(invalid(
                "private restoration lost its original preparing owner",
            ));
        }
        let mut retained = self.private_restore()?;
        if retained.is_some() {
            return Err(invalid("private learning restoration cannot be repeated"));
        }
        *retained = Some(PrivateRestoreOwners {
            session: session.clone_ref(py),
            controller: None,
            task_use: None,
            parent: None,
            model: None,
        });
        Ok(())
    }

    pub(super) fn retain_restore_controller(
        &self,
        py: Python<'_>,
        session: &Py<PySemanticTransitionSession>,
        controller: &Py<PySemanticTransitionController>,
    ) -> PyResult<()> {
        let mut retained = self.private_restore()?;
        let retained = retained
            .as_mut()
            .ok_or_else(|| invalid("private restoration lost its actual retained Session"))?;
        if retained.session.as_ptr() != session.as_ptr()
            || retained.controller.is_some()
            || retained.task_use.is_some()
            || retained.parent.is_some()
        {
            return Err(invalid(
                "private restoration substituted or repeated its actual owners",
            ));
        }
        retained.controller = Some(controller.clone_ref(py));
        Ok(())
    }

    pub(super) fn retain_restore_task(
        &self,
        py: Python<'_>,
        task_use: &Py<PySemanticTransitionTaskUse>,
    ) -> PyResult<()> {
        let mut retained = self.private_restore()?;
        let retained = retained
            .as_mut()
            .ok_or_else(|| invalid("private restoration lost its actual retained Session"))?;
        if retained.controller.is_none() || retained.task_use.is_some() {
            return Err(invalid(
                "private restoration lost or repeated its task owner",
            ));
        }
        retained.task_use = Some(task_use.clone_ref(py));
        Ok(())
    }

    pub(super) fn retain_restore_parent(
        &self,
        py: Python<'_>,
        parent: &Py<PySemanticPublishedParent>,
    ) -> PyResult<()> {
        let mut retained = self.private_restore()?;
        let retained = retained
            .as_mut()
            .ok_or_else(|| invalid("private restoration lost its actual retained Session"))?;
        if retained.task_use.is_none() || retained.parent.is_some() {
            return Err(invalid(
                "private restoration lost or repeated its parent owner",
            ));
        }
        retained.parent = Some(parent.clone_ref(py));
        Ok(())
    }

    pub(super) fn retain_restored_model(&self, py: Python<'_>, model: &Py<PyAny>) -> PyResult<()> {
        let mut retained = self.private_restore()?;
        let retained = retained
            .as_mut()
            .ok_or_else(|| invalid("private restoration lost its actual retained Session"))?;
        if retained.parent.is_none() || retained.model.is_some() {
            return Err(invalid(
                "private restoration lost or repeated its model factory owners",
            ));
        }
        retained.model = Some(model.clone_ref(py));
        Ok(())
    }

    fn retain_durable_readback(&self) -> PyResult<()> {
        let mut completion = self.status_lock()?;
        let Completion::Unknown {
            readback_observed, ..
        } = &mut *completion
        else {
            return Err(invalid(
                "durable phase readback lost its original pending owner",
            ));
        };
        // Exact readback remains known even if subsequent fresh authority or
        // native/model verification fails. A later absence cannot undo it.
        *readback_observed = true;
        Ok(())
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
        verify_phase_native(py, &task, &parent, &source_manifest.native, true)?;
        {
            let _reads = ImportReadScope::checkpoint(&source, &task, &parent, py)?;
            let model = self.source_serializer.bind(py).call0()?;
            require_model_bytes(&model, &source_manifest.model)?;
        }
        verify_phase_native(py, &task, &parent, &source_manifest.native, true)?;
        if include_candidate {
            let (candidate, checkpoint) = self.candidate(py)?;
            let candidate = candidate.borrow(py);
            let issued = candidate.task_use.borrow(py);
            snapshot.newer_than(&issued.state()?.snapshot)?;
            check_learning_grant(&issued, &self.grant_reference, &snapshot)?;
            let acquired = candidate.parent.borrow(py);
            let manifest = SemanticCheckpointManifest::decode(&checkpoint)?;
            verify_phase_native(py, &issued, &acquired, &manifest.native, false)?;
            {
                let successor = candidate.session.borrow(py);
                let _reads = ImportReadScope::checkpoint(&successor, &issued, &acquired, py)?;
                let model = self
                    .candidate_serializer
                    .bind(py)
                    .call1((candidate.model.clone_ref(py),))?;
                require_model_bytes(&model, &manifest.model)?;
            }
            verify_phase_native(py, &issued, &acquired, &manifest.native, false)?;
            issued.state()?.snapshot = snapshot.clone();
        }
        task.state()?.snapshot = snapshot;
        Ok(())
    }

    fn activate(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionRestoredCheckpoint>> {
        self.verify(py, true)?;
        self.preparation_inputs.resource_observer.release(py)?;
        let (candidate_owner, _) = self.candidate(py)?;
        let candidate = candidate_owner.borrow(py);
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
        Ok(candidate_owner.clone_ref(py))
    }

    fn retire_candidate(&self, py: Python<'_>) -> PyResult<()> {
        {
            let completion = self.status_lock()?;
            match &*completion {
                Completion::Retired => return Ok(()),
                Completion::Prepared
                | Completion::Unknown {
                    readback_observed: false,
                    ..
                } => {}
                _ => {
                    return Err(invalid(
                        "private candidate retirement is single-attempt; unknown consumer completion retains its original owners",
                    ));
                }
            }
        }
        let (candidate_owner, _) = self.candidate(py)?;
        {
            let candidate = candidate_owner.borrow(py);
            let successor = candidate.session.borrow(py);
            let issued = candidate.task_use.borrow(py);
            let acquired = candidate.parent.borrow(py);
            let owner = successor.owner()?;
            issued.require_current(&owner)?;
            acquired.require_task(py, &issued)?;
            if !matches!(issued.state()?.phase, TaskUsePhase::ArenaPreparing(_)) {
                return Err(invalid(
                    "private retirement lost its original candidate task",
                ));
            }
            owner
                .published_identity(&*acquired.lease()?)
                .map_err(xlog_err)?;
        }
        // The original model owner must return its aliases and join the actual
        // consumers. No native/task/lease mutex is held across its callback.
        // Mark the attempt before external code: neither a callback exception
        // nor a false durable-absence response authorizes a second retirement.
        *self.status_lock()? = Completion::RetirementUnknown;
        let model = candidate_owner.borrow(py).model.clone_ref(py);
        let result = self.retire_restored_model.bind(py).call1((model,))?;
        if !result.is_none() {
            return Err(invalid(
                "original model retirement must return None after known consumer release",
            ));
        }
        {
            let candidate = candidate_owner.borrow(py);
            let successor = candidate.session.borrow(py);
            let acquired = candidate.parent.borrow(py);
            successor
                .owner()?
                .require_retired_publication(&*acquired.lease()?)
                .map_err(xlog_err)?;
        }
        // Keep known retirement distinct from an unknown callback outcome. A
        // later authority failure may finish source continuation, but must not
        // send the retirement callback or durable resolver again.
        *self.status_lock()? = Completion::Retired;
        Ok(())
    }

    fn abandon(&self, py: Python<'_>) -> PyResult<()> {
        self.verify(py, false)?;
        self.retire_candidate(py)?;
        self.verify(py, false)?;
        let source = self.source.borrow(py);
        let task = self.task_use.borrow(py);
        let resumed = match &task.state()?.phase {
            TaskUsePhase::ArenaPreparing(original) => *original.clone(),
            _ => return Err(invalid("source lost its retained learning preparation")),
        };
        let candidate_owner = self
            .candidate
            .lock()
            .map_err(|_| invalid("learning-phase candidate owner mutex is poisoned"))?
            .as_ref()
            .ok_or_else(|| invalid("private retirement lost its complete outcome"))?
            .owner
            .as_ref()
            .map(|owner| owner.clone_ref(py));
        if let Some(candidate_owner) = candidate_owner {
            let candidate = candidate_owner.borrow(py);
            let successor = candidate.session.borrow(py);
            let issued = candidate.task_use.borrow(py);
            let mut new = issued.state()?;
            match new.phase {
                TaskUsePhase::ArenaPreparing(_) => {
                    successor.owner()?.abort();
                    new.phase = TaskUsePhase::Refused;
                }
                // A later cold-finalization failure may retain an already
                // invalidated owner. Its known retirement is never re-executed.
                TaskUsePhase::Refused => {}
                _ => {
                    return Err(invalid(
                        "private retirement lost its original candidate task",
                    ));
                }
            }
            successor.learning_preparing.store(false, Ordering::Release);
        }
        // Preserve the complete cold outcome, but relinquish actual native/model
        // owners before resuming source use. Python finalizers run without locks.
        let retired = self
            .candidate
            .lock()
            .map_err(|_| invalid("learning-phase candidate owner mutex is poisoned"))?
            .as_mut()
            .ok_or_else(|| invalid("private retirement lost its complete outcome"))?
            .owner
            .take();
        let construction = self.private_restore()?.take();
        drop(retired);
        drop(construction);
        // Released Python owners can run finalizers against the source. Verify
        // its original live checkpoint again before unwrapping its task phase.
        self.verify(py, false)?;
        self.preparation_inputs.resource_observer.release(py)?;
        let mut retention = source
            .learning_transition
            .lock()
            .map_err(|_| invalid("learning-phase retention mutex is poisoned"))?;
        let mut completion = self.status_lock()?;
        let mut old = task.state()?;
        if !matches!(old.phase, TaskUsePhase::ArenaPreparing(_))
            || !matches!(*completion, Completion::Retired)
        {
            return Err(invalid(
                "source continuation lost its known candidate retirement",
            ));
        }
        old.phase = resumed;
        *completion = Completion::Abandoned;
        source.learning_preparing.store(false, Ordering::Release);
        let retained = retention.take();
        drop(old);
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
    fn phase_id(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        Ok(PyBytes::new(py, &self.records()?.phase_id).unbind())
    }

    #[getter]
    fn issuer_anchor(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        Ok(PyBytes::new(py, &self.records()?.issuer()).unbind())
    }

    #[getter]
    fn phase_record_limits(&self) -> PyResult<(u64, u64, u64)> {
        let limits = self.records()?.limits;
        Ok((limits.record_bytes, limits.total_bytes, limits.records))
    }

    #[getter]
    fn status(&self) -> PyResult<&'static str> {
        Ok(match &*self.status_lock()? {
            Completion::Preparing => "preparing",
            Completion::PreparationUnknown => "unknown",
            Completion::Prepared => "prepared",
            Completion::Unknown { .. } => "unknown",
            Completion::RetirementUnknown | Completion::Retired => "unknown",
            Completion::Committed => "committed",
            Completion::Abandoned => "abandoned",
        })
    }

    #[getter]
    fn checkpoint_sha256(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let checkpoint = self.candidate_checkpoint()?;
        Ok(PyBytes::new(py, &Sha256::digest(&checkpoint)).unbind())
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
        if durable_owner.as_ptr() != self.phase_record_owner.as_ptr() {
            return Err(invalid(
                "final phase publication requires the same original early durable store",
            ));
        }
        if !matches!(*self.status_lock()?, Completion::Prepared) {
            return Err(invalid(
                "phase checkpoint commit is single-attempt; resolve unknown completion instead",
            ));
        }
        self.verify(py, true)?;
        let (_, checkpoint) = self.candidate(py)?;
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
            readback_observed: false,
        };
        let readback = commit.call1((
            &self.destination,
            PyBytes::new(py, &checkpoint),
            PyBytes::new(py, &Sha256::digest(&checkpoint)),
        ))?;
        require_model_bytes(&readback, &checkpoint)?;
        self.retain_durable_readback()?;
        self.activate(py)
    }

    /// Native selects the original preparation or final publication readback.
    /// A known preparation stays private; resolution never commits or abandons it.
    /// Final absence permits source continuation only before exact readback.
    fn resolve_checkpoint(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<Py<PySemanticTransitionRestoredCheckpoint>>> {
        let _operation = PhaseOperation::begin(&self.operating)?;
        self.source.borrow(py).require_creator()?;
        let preparation_unknown = matches!(*self.status_lock()?, Completion::PreparationUnknown);
        if preparation_unknown {
            self.resolve_preparation_record(py)?;
            return Ok(None);
        }
        let retirement_known = matches!(*self.status_lock()?, Completion::Retired);
        if retirement_known {
            self.abandon(py)?;
            return Ok(None);
        }
        if matches!(*self.status_lock()?, Completion::RetirementUnknown) {
            return Err(invalid(
                "unknown private consumer retirement must retain the same owners; checkpoint absence cannot resolve or repeat it",
            ));
        }
        let (resolve, readback_observed) = match &*self.status_lock()? {
            Completion::Unknown {
                resolve,
                readback_observed,
                ..
            } => (resolve.clone_ref(py), *readback_observed),
            Completion::Prepared => return Ok(None),
            _ => return Err(invalid("only an unknown phase checkpoint needs resolution")),
        };
        let (_, checkpoint) = self.candidate(py)?;
        let result = resolve.bind(py).call1((
            &self.destination,
            PyBytes::new(py, &Sha256::digest(&checkpoint)),
        ))?;
        if result.is_none() {
            return Ok(None);
        }
        if result.is_exact_instance_of::<PyBool>() && !result.extract::<bool>()? {
            if readback_observed {
                return Err(invalid(
                    "phase resolver contradicted the original exact durable readback; retain the same pending owners",
                ));
            }
            self.abandon(py)?;
            return Ok(None);
        }
        require_model_bytes(&result, &checkpoint)?;
        self.retain_durable_readback()?;
        self.activate(py).map(Some)
    }

    /// Deliberately continue the same live attempt: enter only from proven
    /// nonentry, or finish its retained known cold outcome without re-execution.
    /// Record readback itself never enters preparation or repeats a write.
    fn continue_preparation(slf: Py<Self>, py: Python<'_>) -> PyResult<Py<Self>> {
        let pending = slf.borrow(py);
        let operation = PhaseOperation::begin(&pending.operating)?;
        let source = pending.source.borrow(py);
        source.require_creator()?;
        if !matches!(*pending.status_lock()?, Completion::PreparationUnknown) {
            return Err(invalid(
                "preparation continuation requires the original unknown pending owner",
            ));
        }
        let entered = pending.preparation_entered.load(Ordering::Acquire);
        let candidate_entered = pending.candidate_entered.load(Ordering::Acquire);
        let final_candidate_known = pending
            .candidate
            .lock()
            .map_err(|_| invalid("learning-phase candidate owner mutex is poisoned"))?
            .is_some();
        if entered && candidate_entered {
            #[cfg(feature = "semantic-policy")]
            if !final_candidate_known && !pending.control_retirement_retained()? {
                pending.require_known_private_restore(py)?;
            }
            if final_candidate_known {
                // The actual canonical save proves known complete construction,
                // not full scientific execution or resource use. Only remaining
                // cold verification may continue; a retained scientific return is
                // reused and an entered failed comparison is never called again.
                pending.candidate(py)?;
                pending.records()?.require_preparation_admission()?;
                if let Some(acceptance) = pending.acceptance()?.as_ref() {
                    if let Some(error) = &acceptance.error {
                        return Err(error.clone_ref(py));
                    }
                    if acceptance.result.is_none() {
                        return Err(invalid("unknown scientific comparison cannot be repeated by preparation continuation"));
                    }
                }
            }
        } else if entered {
            if !pending.source_verified()? {
                return Err(invalid(
                    "unknown source native/model work cannot be repeated by resource continuation",
                ));
            }
            pending.require_source_operation(py)?;
            pending.records()?.require_preparation_admission()?;
        } else if pending.private_restore()?.is_some()
            || pending
                .candidate
                .lock()
                .map_err(|_| invalid("learning-phase candidate owner mutex is poisoned"))?
                .is_some()
        {
            return Err(invalid(
                "preparation continuation requires native-proven nonentry; an unknown execution outcome must remain retained",
            ));
        }
        let retained = source
            .learning_transition
            .lock()
            .map_err(|_| invalid("learning-phase retention mutex is poisoned"))?;
        if !source.learning_preparing.load(Ordering::Acquire)
            || retained
                .as_ref()
                .is_none_or(|owner| owner.as_ptr() != slf.as_ptr())
        {
            return Err(invalid(
                "preparation continuation lost its original source custody",
            ));
        }
        drop(retained);
        pending.records()?.preparation_admission_needed()?;
        if pending.store()?.is_none() {
            return Err(invalid(
                "preparation continuation lost its original captured store",
            ));
        }
        *pending.status_lock()? = Completion::Preparing;
        let result = if entered && candidate_entered && final_candidate_known {
            pending.finish_preparation(py)
        } else if entered && candidate_entered {
            #[cfg(feature = "semantic-policy")]
            {
                pending
                    .execute_control_branch(py, &slf)
                    .and_then(|()| pending.prepare_private_trajectory(py, &slf, "real"))
                    .and_then(|()| pending.execute_private_group(py, "real"))
                    .and_then(|()| pending.execute_private_evaluations(py, "real"))
                    .and_then(|()| pending.execute_private_checkpoint(py, "real"))
                    .and_then(|()| pending.retain_final_checkpoint_candidate(py))
                    .and_then(|()| pending.finish_preparation(py))
            }
            #[cfg(not(feature = "semantic-policy"))]
            {
                Err(invalid(
                    "private scientific trajectory execution requires semantic-policy",
                ))
            }
        } else {
            pending.prepare_candidate(py, &slf)
        };
        if let Err(error) = result {
            *pending.status_lock()? = Completion::PreparationUnknown;
            return Err(error);
        }
        drop(source);
        drop(operation);
        drop(pending);
        Ok(slf)
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

    #[pyo3(signature = (task_use, *, parent, recipe, source_checkpoint, consumer_streams, snapshot, scientific_owner,
        source_model, execute_phase_instruction, resource_observer, feedback_interventions,
        learning_grant_ref, checkpoint_destination, snapshot_model_state, restore_model,
        snapshot_restored_model, retire_restored_model, refresh_snapshot, phase_record_owner, phase_record_limits, frozen_program_bytes, cold_model_work_capacity, resolve_checkpoint=None,
        max_checkpoint_bytes=None, max_total_checkpoint_bytes=None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "the original task, scientific acceptance, durable destination and model owners are independent mandatory inputs"
    )]
    fn prepare_learning_phase_transition(
        slf: Py<Self>,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
        parent: Py<PySemanticPublishedParent>,
        recipe: Py<PySemanticLearningPhaseRecipe>,
        source_checkpoint: &Bound<'_, PyAny>,
        consumer_streams: &Bound<'_, PyAny>,
        snapshot: &Bound<'_, PyAny>,
        scientific_owner: Py<PyAny>,
        source_model: Py<PyAny>,
        execute_phase_instruction: &Bound<'_, PyAny>,
        resource_observer: &Bound<'_, PyAny>,
        feedback_interventions: &Bound<'_, PyAny>,
        learning_grant_ref: &Bound<'_, PyAny>,
        checkpoint_destination: &Bound<'_, PyAny>,
        snapshot_model_state: &Bound<'_, PyAny>,
        restore_model: &Bound<'_, PyAny>,
        snapshot_restored_model: &Bound<'_, PyAny>,
        retire_restored_model: &Bound<'_, PyAny>,
        refresh_snapshot: &Bound<'_, PyAny>,
        phase_record_owner: Py<PyAny>,
        phase_record_limits: &Bound<'_, PyAny>,
        frozen_program_bytes: &Bound<'_, PyAny>,
        cold_model_work_capacity: &Bound<'_, PyAny>,
        resolve_checkpoint: Option<&Bound<'_, PyAny>>,
        max_checkpoint_bytes: Option<&Bound<'_, PyAny>>,
        max_total_checkpoint_bytes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PySemanticLearningPhaseTransition>> {
        let controller = slf.borrow(py);
        if !cfg!(feature = "semantic-policy") {
            return Err(invalid(
                "cold model work requires the semantic-policy build",
            ));
        }
        let source = controller.session.borrow(py);
        source.require_creator()?;
        let task = task_use.borrow(py);
        let acquired = parent.borrow(py);
        controller.require_issued(&task)?;
        acquired.require_task(py, &task)?;
        if !source_checkpoint.is_exact_instance_of::<PyBytes>() {
            return Err(invalid(
                "phase preparation requires the exact original full source checkpoint bytes",
            ));
        }
        let checkpoint = source_checkpoint.cast::<PyBytes>()?.as_bytes();
        if !frozen_program_bytes.is_exact_instance_of::<PyBytes>()
            || frozen_program_bytes
                .cast::<PyBytes>()?
                .as_bytes()
                .is_empty()
        {
            return Err(invalid(
                "phase admission requires the complete original frozen program bytes",
            ));
        }
        let record_limits = RecordLimits::read(phase_record_limits)?;
        let cold_capacity_material = ColdValue::read(cold_model_work_capacity, &mut 128, 0)?;
        let cold_model_work_capacity = usize::try_from(cold_capacity_material.unsigned()?)
            .ok().filter(|capacity| *capacity > 0)
            .ok_or_else(|| invalid("cold model work requires an exact positive capacity with bounded allocation arithmetic"))?;
        #[cfg(feature = "semantic-policy")]
        xlog_cuda::SemanticColdModelWork::allocation_bytes(cold_model_work_capacity)
            .map_err(xlog_err)?;
        let feedback_materials = feedback_materials(feedback_interventions)?;
        if source_model.bind(py).is_none() || !execute_phase_instruction.is_callable() {
            return Err(invalid(
                "phase preparation requires its original source model and instruction execution owner",
            ));
        }
        let resource_observer = ResourceObserver::capture(resource_observer.cast::<PyCapsule>()?)?;
        let manifest = SemanticCheckpointManifest::decode(checkpoint)?;
        let (_, saved_snapshot, _, _) = TaskCheckpointSeed::decode(&manifest.task)?;
        task.authority.check_snapshot(&saved_snapshot)?;
        checkpoint_consumer_streams(consumer_streams, &mut (16 * 1024 * 1024))?;
        let grant_value = ColdValue::read(learning_grant_ref, &mut (16 * 1024 * 1024), 0)?;
        let learning_grant_ref = grant_value.text()?;
        let destination_value =
            ColdValue::read(checkpoint_destination, &mut (16 * 1024 * 1024), 0)?;
        let checkpoint_destination = destination_value.text()?;
        if checkpoint_destination.is_empty()
            || !snapshot_model_state.is_callable()
            || !restore_model.is_callable()
            || !snapshot_restored_model.is_callable()
            || !retire_restored_model.is_callable()
            || !refresh_snapshot.is_callable()
        {
            return Err(invalid(
                "phase preparation requires its original destination, serializers, restore and authority owner",
            ));
        }
        let snapshot_value = ColdValue::read(snapshot, &mut (16 * 1024 * 1024), 0)?;
        let initial = AuthoritySnapshot::parse(&snapshot_value)?;
        let prior_snapshot = task.state()?.snapshot.clone();
        initial.newer_than(&prior_snapshot)?;
        initial.newer_than(&saved_snapshot)?;
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
        let recipe_owner = recipe.borrow(py);
        let recipe_values = PyTuple::new(
            py,
            [
                recipe_owner.source().into_pyobject(py)?.into_any(),
                recipe_owner.target().into_pyobject(py)?.into_any(),
                recipe_owner.phase_index().into_pyobject(py)?.into_any(),
                recipe_owner
                    .completed_updates_index()
                    .into_pyobject(py)?
                    .into_any(),
                recipe_owner.views(py)?.into_bound(py).into_any(),
            ],
        )?;
        let recipe_material =
            ColdValue::read(recipe_values.as_any(), &mut (16 * 1024 * 1024), 0)?.canonical_bytes();
        drop(recipe_owner);
        let phase_records = PhaseRecords::new(
            &[
                checkpoint,
                frozen_program_bytes.cast::<PyBytes>()?.as_bytes(),
                &recipe_material,
                &grant_value.canonical_bytes(),
                checkpoint_destination.as_bytes(),
                feedback_materials[0].as_bytes(),
                feedback_materials[1].as_bytes(),
                &cold_capacity_material.canonical_bytes(),
            ],
            record_limits,
        )?;
        let accept_learning_phase = scientific_owner.bind(py).getattr("accept_learning_phase")?;
        if !accept_learning_phase.is_callable() {
            return Err(invalid(
                "phase preparation requires its original final scientific acceptance owner",
            ));
        }
        let preparation_inputs = PreparationInputs {
            source_model,
            execute_phase_instruction: execute_phase_instruction.clone().unbind(),
            resource_observer,
            feedback_interventions: feedback_interventions.cast::<PyTuple>()?.clone().unbind(),
            consumer_streams: ColdValue::read(consumer_streams, &mut (16 * 1024 * 1024), 0)?,
            snapshot: snapshot_value,
            prior_snapshot,
            grant: grant_value,
            frozen_program: frozen_program_bytes.cast::<PyBytes>()?.as_bytes().to_vec(),
            cold_model_work_capacity,
            resolve_checkpoint: resolve_checkpoint.map(|callback| callback.clone().unbind()),
            max_checkpoint_bytes: max_checkpoint_bytes
                .map(|value| ColdValue::read(value, &mut 1024, 0))
                .transpose()?,
            max_total_checkpoint_bytes: max_total_checkpoint_bytes
                .map(|value| ColdValue::read(value, &mut 1024, 0))
                .transpose()?,
        };
        let pending = Py::new(
            py,
            PySemanticLearningPhaseTransition {
                source_controller: slf.clone_ref(py),
                source: controller.session.clone_ref(py),
                task_use: task_use.clone_ref(py),
                parent: parent.clone_ref(py),
                candidate: Mutex::new(None),
                private_restore: Mutex::new(None),
                source_checkpoint: checkpoint.to_vec(),
                scientific_owner: scientific_owner.clone_ref(py),
                accept_learning_phase: accept_learning_phase.unbind(),
                scientific_acceptance: Mutex::new(None),
                recipe: recipe.clone_ref(py),
                restore_model: restore_model.clone().unbind(),
                retire_restored_model: retire_restored_model.clone().unbind(),
                source_serializer: snapshot_model_state.clone().unbind(),
                candidate_serializer: snapshot_restored_model.clone().unbind(),
                refresh_snapshot: refresh_snapshot.clone().unbind(),
                grant_reference: learning_grant_ref.to_owned(),
                destination: checkpoint_destination.to_owned(),
                phase_record_owner,
                phase_store: Mutex::new(None),
                phase_records: Mutex::new(phase_records),
                preparation_inputs,
                preparation_entered: AtomicBool::new(false),
                source_preparation: Mutex::new(None),
                #[cfg(feature = "semantic-policy")]
                phase_evaluations: Mutex::new(Vec::new()),
                #[cfg(feature = "semantic-policy")]
                phase_evaluation_active: AtomicBool::new(false),
                #[cfg(feature = "semantic-policy")]
                control_evaluations_done: AtomicBool::new(false),
                #[cfg(feature = "semantic-policy")]
                real_evaluations_done: AtomicBool::new(false),
                #[cfg(feature = "semantic-policy")]
                private_checkpoints: Mutex::new(Vec::new()),
                #[cfg(feature = "semantic-policy")]
                private_checkpoint_active: AtomicBool::new(false),
                #[cfg(feature = "semantic-policy")]
                control_retirement: Mutex::new(None),
                #[cfg(feature = "semantic-policy")]
                private_trajectory_start: Mutex::new(None),
                #[cfg(feature = "semantic-policy")]
                private_execution: Mutex::new(None),
                #[cfg(feature = "semantic-policy")]
                private_execution_active: AtomicBool::new(false),
                candidate_entered: AtomicBool::new(false),
                completion: Mutex::new(Completion::Preparing),
                operating: AtomicBool::new(false),
            },
        )?;
        let mut retention = source
            .learning_transition
            .lock()
            .map_err(|_| invalid("learning-phase retention mutex is poisoned"))?;
        if retention.is_some() || source.learning_preparing.swap(true, Ordering::AcqRel) {
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
        // Retain the same native-issued owner before serializers, scientific
        // callbacks, native restoration or the model factory can perform work.
        *retention = Some(pending.clone_ref(py));
        drop(retention);
        let retained_owner = pending.borrow(py);
        let operation = PhaseOperation::begin(&retained_owner.operating)?;
        let result = (|| -> PyResult<()> {
            *retained_owner.store()? = Some(PhaseRecordStore::capture(
                retained_owner.phase_record_owner.bind(py),
            )?);
            retained_owner.pin_issuer(py, &pending)?;
            retained_owner.prepare_candidate(py, &pending)
        })();
        if let Err(error) = result {
            // Source equality does not prove that an external operation did
            // not execute. Keep the same callbacks, native/model owners and
            // irreversible expense; never abort or resume by that assumption.
            *retained_owner.status_lock()? = Completion::PreparationUnknown;
            return Err(error);
        }
        drop(operation);
        drop(retained_owner);
        Ok(pending)
    }
}
