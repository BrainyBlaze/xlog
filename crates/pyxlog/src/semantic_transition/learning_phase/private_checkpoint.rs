//! Full private snapshots through the original controller and model serializer.

use super::phase_evaluation::EvaluationOwners;
use super::*;
use xlog_cuda::{SemanticColdModelWork, SemanticColdModelWorkResult, SemanticColdNativeWork};

pub(super) struct PrivateCheckpoint {
    owners: EvaluationOwners,
    branch: &'static str,
    entries: Py<PyTuple>,
    material: Vec<u8>,
    instruction: Vec<u8>,
    ordinal: u64,
    budget: [u64; 3],
    decode_entered: bool,
    decoded: Option<Py<PyAny>>,
    started: bool,
    ready: bool,
    work: Option<SemanticColdModelWork>,
    callback_work: Option<Py<PySemanticColdModelWork>>,
    custody: Option<SemanticColdNativeWork>,
    save_entered: bool,
    save_admitted: bool,
    saved: Option<Py<PyBytes>>,
    error: Option<PyErr>,
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
    fn private_checkpoints(&self) -> PyResult<MutexGuard<'_, Vec<PrivateCheckpoint>>> {
        self.private_checkpoints
            .lock()
            .map_err(|_| invalid("private checkpoint custody mutex is poisoned"))
    }

    pub(super) fn execute_private_checkpoint(
        &self,
        py: Python<'_>,
        branch: &'static str,
    ) -> PyResult<()> {
        let (owners, ordinal) = self.private_checkpoint_input(py, branch)?;
        self.save_private_checkpoint(py, branch, owners, ordinal)
    }

    pub(super) fn retain_final_checkpoint_candidate(&self, py: Python<'_>) -> PyResult<()> {
        self.records()?.require_preparation_admission()?;
        self.preparation_inputs
            .require_program(py, &self.scientific_owner)?;
        let (selected, _) = self.private_checkpoint_input(py, "real")?;
        let (checkpoint, ordinal) = {
            let retained = self.private_checkpoints()?;
            let current = retained
                .last()
                .filter(|entry| entry.branch == "real")
                .ok_or_else(|| invalid("final handoff lost its original full real checkpoint"))?;
            Self::require_private_checkpoint_entry(py, current)?;
            if !current.recorded
                || !current.released
                || current.budget_exceeded
                || current.error.is_some()
                || current.owners.controller.as_ptr() != selected.controller.as_ptr()
                || current.owners.task.as_ptr() != selected.task.as_ptr()
                || current.owners.parent.as_ptr() != selected.parent.as_ptr()
                || current.owners.model.as_ptr() != selected.model.as_ptr()
            {
                return Err(invalid("final handoff precedes known checkpoint accounting or changed its selected owners"));
            }
            (
                Arc::<[u8]>::from(
                    current
                        .saved
                        .as_ref()
                        .ok_or_else(|| invalid("final handoff lost its saved full checkpoint"))?
                        .bind(py)
                        .as_bytes(),
                ),
                current.ordinal,
            )
        };
        let next = self.scientific_owner.bind(py).getattr("next_operation")?;
        if !next.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "final handoff requires the complete original prefix before delivery",
            ));
        }
        let (material, _) = Self::singleton_lifecycle_material(next.cast::<PyTuple>()?)?;
        let material = ColdValue::from_canonical_bytes(&material)?;
        let fields = material.fields(6)?;
        if fields[0].text()? != "delivery"
            || fields[1].text()? != "real"
            || fields[2].unsigned()?
                != ordinal
                    .checked_add(1)
                    .ok_or_else(|| invalid("final handoff position overflowed"))?
            || fields[3].unsigned()? != 0
            || fields[5] != ColdValue::None
        {
            return Err(invalid(
                "final handoff changed its sole contiguous original delivery tail",
            ));
        }
        let session = selected.parent.borrow(py).session.clone_ref(py);
        self.prepare_delivery_expense(py, next.cast::<PyTuple>()?, &selected)?;
        {
            let construction = self.private_restore()?;
            let construction = construction.as_ref().ok_or_else(|| {
                invalid("final handoff lost its original private Session construction")
            })?;
            if construction.session.as_ptr() != session.as_ptr()
                || !session
                    .borrow(py)
                    .learning_preparing
                    .load(Ordering::Acquire)
                || !matches!(*self.status_lock()?, Completion::Preparing)
            {
                return Err(invalid(
                    "final handoff changed its actual private preparation owner",
                ));
            }
        }
        let manifest = SemanticCheckpointManifest::decode(&checkpoint)?;
        let (_, saved_snapshot, _, _) = TaskCheckpointSeed::decode(&manifest.task)?;
        verify_phase_checkpoint(
            py,
            &selected.task.borrow(py),
            &selected.parent.borrow(py),
            &manifest,
            &saved_snapshot,
        )?;
        {
            let retained = self
                .candidate
                .lock()
                .map_err(|_| invalid("learning-phase candidate owner mutex is poisoned"))?;
            if let Some(original) = retained.as_ref() {
                let owner = original
                    .owner
                    .as_ref()
                    .ok_or_else(|| {
                        invalid("final handoff cannot replace retired candidate owners")
                    })?
                    .borrow(py);
                if original.checkpoint.as_ref() != checkpoint.as_ref()
                    || owner.session.as_ptr() != session.as_ptr()
                    || owner.controller.as_ptr() != selected.controller.as_ptr()
                    || owner.task_use.as_ptr() != selected.task.as_ptr()
                    || owner.parent.as_ptr() != selected.parent.as_ptr()
                    || owner.model.as_ptr() != selected.model.as_ptr()
                {
                    return Err(invalid(
                        "final handoff cannot replace its original complete candidate",
                    ));
                }
                return Ok(());
            }
        }
        // This issues the actual selected owners, not another restoration or
        // a wrapper around the initial, now superseded publication.
        let owner = PySemanticTransitionRestoredCheckpoint::issue(
            py,
            session,
            selected.controller,
            selected.task,
            selected.parent,
            selected.model,
        )?;
        let mut retained = self
            .candidate
            .lock()
            .map_err(|_| invalid("learning-phase candidate owner mutex is poisoned"))?;
        if retained.is_some() {
            return Err(invalid(
                "final handoff cannot replace an already retained candidate",
            ));
        }
        *retained = Some(PreparedCandidate {
            owner: Some(owner),
            checkpoint,
            preparation_outcome: None,
        });
        Ok(())
    }

    pub(super) fn control_retirement_input(
        &self,
        py: Python<'_>,
    ) -> PyResult<(EvaluationOwners, Py<PyBytes>, u64)> {
        let retained = self.private_checkpoints()?;
        let current = retained
            .last()
            .filter(|entry| entry.branch == "control")
            .ok_or_else(|| invalid("control retirement lost its original full checkpoint"))?;
        Self::require_private_checkpoint_entry(py, current)?;
        if !current.recorded || !current.released || current.budget_exceeded {
            return Err(invalid(
                "control retirement precedes known checkpoint accounting and release",
            ));
        }
        Ok((
            EvaluationOwners {
                controller: current.owners.controller.clone_ref(py),
                task: current.owners.task.clone_ref(py),
                parent: current.owners.parent.clone_ref(py),
                model: current.owners.model.clone_ref(py),
            },
            current
                .saved
                .as_ref()
                .expect("original full checkpoint")
                .clone_ref(py),
            current
                .ordinal
                .checked_add(1)
                .ok_or_else(|| invalid("control retirement position overflowed"))?,
        ))
    }

    pub(super) fn drop_completed_private_checkpoints(&self, branch: &'static str) -> PyResult<()> {
        let mut retained = self.private_checkpoints()?;
        if retained.iter().any(|entry| {
            entry.branch != branch
                || !entry.recorded
                || !entry.released
                || entry.budget_exceeded
                || entry.error.is_some()
        }) {
            return Err(invalid(
                "model retirement cannot discard unfinished or unrelated private checkpoints",
            ));
        }
        let original = std::mem::take(&mut *retained);
        drop(retained);
        drop(original);
        Ok(())
    }

    fn require_private_checkpoint_entry(
        py: Python<'_>,
        current: &PrivateCheckpoint,
    ) -> PyResult<()> {
        let (material, instruction) = Self::singleton_lifecycle_material(current.entries.bind(py))?;
        if material != current.material || instruction != current.instruction {
            return Err(invalid(
                "private checkpoint changed its original frozen entry",
            ));
        }
        Ok(())
    }

    pub(in crate::semantic_transition) fn require_private_checkpoint_save(
        &self,
        py: Python<'_>,
        controller: &PySemanticTransitionController,
        task: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
    ) -> PyResult<()> {
        if self.intermediate_snapshot_active.load(Ordering::Acquire) {
            return self.require_intermediate_snapshot_save(py, controller, task, parent);
        }
        if self.phase_evaluation_active.load(Ordering::Acquire) {
            return self.require_terminal_private_checkpoint(py, controller, task, parent);
        }
        if !self.private_checkpoint_active.load(Ordering::Acquire) {
            return Err(invalid(
                "private checkpoint save belongs only to its original active operation",
            ));
        }
        self.records()?.require_preparation_admission()?;
        self.preparation_inputs
            .require_program(py, &self.scientific_owner)?;
        let mut retained = self.private_checkpoints()?;
        let current = retained
            .last_mut()
            .ok_or_else(|| invalid("private checkpoint lost its original snapshot owner"))?;
        Self::require_private_checkpoint_entry(py, current)?;
        if !current.ready
            || !current.save_entered
            || current.save_admitted
            || current.saved.is_some()
            || current.error.is_some()
            || !std::ptr::eq(controller, &*current.owners.controller.borrow(py))
            || !std::ptr::eq(task, &*current.owners.task.borrow(py))
            || !std::ptr::eq(parent, &*current.owners.parent.borrow(py))
            || !matches!(task.state()?.phase, TaskUsePhase::ArenaPreparing(_))
        {
            return Err(invalid(
                "private checkpoint changed or repeated its original save, task or parent",
            ));
        }
        // Consume before the canonical save can enter any external serializer.
        current.save_admitted = true;
        Ok(())
    }

    fn save_private_checkpoint(
        &self,
        py: Python<'_>,
        branch: &'static str,
        owners: EvaluationOwners,
        ordinal: u64,
    ) -> PyResult<()> {
        self.records()?.require_preparation_admission()?;
        self.preparation_inputs
            .require_program(py, &self.scientific_owner)?;
        let retained = self.private_checkpoints()?;
        let existing = retained.last().is_some_and(|entry| entry.branch == branch);
        if existing {
            let current = retained.last().expect("original checkpoint");
            if current.ordinal != ordinal
                || current.owners.controller.as_ptr() != owners.controller.as_ptr()
                || current.owners.task.as_ptr() != owners.task.as_ptr()
                || current.owners.parent.as_ptr() != owners.parent.as_ptr()
                || current.owners.model.as_ptr() != owners.model.as_ptr()
            {
                return Err(invalid(
                    "private checkpoint changed its original selected owners or position",
                ));
            }
            Self::require_private_checkpoint_entry(py, current)?;
            if let Some(error) = &current.error {
                return Err(error.clone_ref(py));
            }
            if current.recorded && current.released {
                return if current.budget_exceeded {
                    Err(invalid(
                        "private checkpoint exceeded its original operation budget",
                    ))
                } else {
                    Ok(())
                };
            }
        } else if retained
            .last()
            .is_some_and(|entry| !entry.recorded || !entry.released || entry.budget_exceeded)
        {
            return Err(invalid(
                "private checkpoint cannot replace an unfinished or refused original snapshot",
            ));
        }
        drop(retained);
        if !existing {
            let next = self.scientific_owner.bind(py).getattr("next_operation")?;
            if !next.is_exact_instance_of::<PyTuple>() {
                return Err(invalid(
                    "private checkpoint requires the original complete scheduled group",
                ));
            }
            let entries = next.cast::<PyTuple>()?;
            let (material, instruction) = Self::singleton_lifecycle_material(entries)?;
            let fields = ColdValue::from_canonical_bytes(&material)?;
            let fields = fields.fields(6)?;
            if fields[0].text()? != "checkpoint"
                || fields[1].text()? != branch
                || fields[2].unsigned()? != ordinal
                || fields[3].unsigned()? != 0
                || fields[5] != ColdValue::None
            {
                return Err(invalid(
                    "private checkpoint changed its original lifecycle entry",
                ));
            }
            let budget = fields[4].fields(3)?;
            let budget = [
                budget[0].unsigned()?,
                budget[1].unsigned()?,
                budget[2].unsigned()?,
            ];
            self.private_checkpoints()?.push(PrivateCheckpoint {
                owners,
                branch,
                entries: entries.clone().unbind(),
                material,
                instruction,
                ordinal,
                budget,
                decode_entered: false,
                decoded: None,
                started: false,
                ready: false,
                work: None,
                callback_work: None,
                custody: None,
                save_entered: false,
                save_admitted: false,
                saved: None,
                error: None,
                closed: false,
                report: None,
                observer_finish_entered: false,
                backing_peak: None,
                record_entered: false,
                recorded: false,
                released: false,
                budget_exceeded: false,
            });
        }
        self.decode_private_checkpoint(py)?;
        let (controller, task, parent, model, started, ready, saved, entered, recorded) = {
            let retained = self.private_checkpoints()?;
            let current = retained.last().expect("original checkpoint");
            (
                current.owners.controller.clone_ref(py),
                current.owners.task.clone_ref(py),
                current.owners.parent.clone_ref(py),
                current.owners.model.clone_ref(py),
                current.started,
                current.ready,
                current.saved.as_ref().map(|value| value.clone_ref(py)),
                current.save_entered,
                current.recorded,
            )
        };
        if recorded {
            return self.release_private_checkpoint(py);
        }
        if !started {
            let fresh = self.refresh_snapshot.bind(py).call0()?;
            let snapshot =
                AuthoritySnapshot::parse(&ColdValue::read(&fresh, &mut (16 * 1024 * 1024), 0)?)?;
            {
                let issued = task.borrow(py);
                snapshot.newer_than(&issued.state()?.snapshot)?;
                check_learning_grant(&issued, &self.grant_reference, &snapshot)?;
            }
            self.private_checkpoints()?
                .last_mut()
                .expect("original checkpoint")
                .started = true;
            self.preparation_inputs
                .resource_observer
                .begin(py, ordinal)?;
            let source = self.source.borrow(py);
            let work = source
                .owner()?
                .prepare_cold_model_work(
                    &*self.parent.borrow(py).lease()?,
                    self.preparation_inputs.cold_model_work_capacity,
                    ordinal,
                    self.records()?.confirmed_admission()?,
                )
                .map_err(xlog_err)?;
            self.private_checkpoints()?
                .last_mut()
                .expect("original checkpoint")
                .work = Some(work.clone());
            let callback_work = Py::new(
                py,
                PySemanticColdModelWork {
                    parent: parent.clone_ref(py),
                    reader: self.parent.clone_ref(py),
                    inner: work.clone(),
                    region: None,
                    active: AtomicBool::new(false),
                },
            )?;
            self.private_checkpoints()?
                .last_mut()
                .expect("original checkpoint")
                .callback_work = Some(callback_work);
            let custody = source
                .owner()?
                .share_cold_native_work(&work)
                .map_err(xlog_err)?;
            self.private_checkpoints()?
                .last_mut()
                .expect("original checkpoint")
                .custody = Some(custody.clone());
            parent
                .borrow(py)
                .session
                .borrow(py)
                .owner()?
                .attach_shared_cold_native_work(&*parent.borrow(py).lease()?, custody)
                .map_err(xlog_err)?;
            self.private_checkpoints()?
                .last_mut()
                .expect("original checkpoint")
                .ready = true;
        } else if !ready {
            return Err(invalid("unknown private checkpoint admission retains its original work and cannot serialize again"));
        }
        let saved = if let Some(saved) = saved {
            saved
        } else {
            if entered {
                return Err(invalid(
                    "unknown private checkpoint save cannot repeat its original serializer",
                ));
            }
            let work = self
                .private_checkpoints()?
                .last()
                .expect("original checkpoint")
                .callback_work
                .as_ref()
                .expect("original checkpoint registrar")
                .clone_ref(py);
            let serializer = self.model_owner(py, PhaseModelOwner::SerializeCandidate)?;
            let selected = model.clone_ref(py);
            let session = parent.borrow(py).session.clone_ref(py);
            let snapshot_model = pyo3::types::PyCFunction::new_closure(
                py,
                None,
                None,
                move |args, kwargs| {
                    let py = args.py();
                    if !args.is_empty() || kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
                        return Err(invalid("private checkpoint serializer accepts only its original zero-argument call"));
                    }
                    let session = session.borrow(py);
                    let original = work.borrow(py);
                    // The canonical save owns CheckpointReading before entering this
                    // callback; the original model recorder retains its real owner.
                    let _scope =
                        ColdCallbackScope::enter(py, &session, &original, work.clone_ref(py))?;
                    serializer
                        .bind(py)
                        .call1((selected.clone_ref(py),))
                        .map(Bound::unbind)
                },
            )?;
            let fresh = self.refresh_snapshot.bind(py).call0()?;
            let authority =
                AuthoritySnapshot::parse(&ColdValue::read(&fresh, &mut (16 * 1024 * 1024), 0)?)?;
            {
                let issued = task.borrow(py);
                authority.newer_than(&issued.state()?.snapshot)?;
                check_learning_grant(&issued, &self.grant_reference, &authority)?;
            }
            let streams = self.preparation_inputs.consumer_streams.python_value(py)?;
            self.private_checkpoints()?
                .last_mut()
                .expect("original checkpoint")
                .save_entered = true;
            let result = {
                let _operation = PhaseOperation::begin(&self.private_checkpoint_active)?;
                let original_controller = controller.borrow(py);
                let issued = task.borrow(py);
                let published = parent.borrow(py);
                original_controller.save_checkpoint(
                    py,
                    &issued,
                    &published,
                    streams.bind(py),
                    &fresh,
                    snapshot_model.as_any(),
                )
            };
            let mut retained = self.private_checkpoints()?;
            let current = retained.last_mut().expect("original checkpoint");
            match result {
                Ok(saved) => {
                    current.saved = Some(saved.clone_ref(py));
                    saved
                }
                Err(error) => {
                    current.error = Some(error.clone_ref(py));
                    return Err(error);
                }
            }
        };
        // These are the actual final full snapshots of both frozen branches.
        // Check the complete executed recipe before recording either snapshot
        // or retiring control; trajectory-start alone cannot change the phase.
        self.require_executed_phase_lineage(saved.bind(py).as_bytes())?;
        let (work, closed, cached_report, custody) = {
            let retained = self.private_checkpoints()?;
            let current = retained.last().expect("original checkpoint");
            (
                current.work.clone().expect("original checkpoint work"),
                current.closed,
                current.report,
                current.custody.clone(),
            )
        };
        let source = self.source.borrow(py);
        if !closed {
            source
                .owner()?
                .require_closed_cold_model_work(&work)
                .map_err(xlog_err)?;
            self.private_checkpoints()?
                .last_mut()
                .expect("original checkpoint")
                .closed = true;
        }
        let report = match cached_report {
            Some(report) => report,
            None => {
                if let Some(custody) = custody {
                    parent
                        .borrow(py)
                        .session
                        .borrow(py)
                        .owner()?
                        .detach_shared_cold_native_work(&custody)
                        .map_err(xlog_err)?;
                    self.private_checkpoints()?
                        .last_mut()
                        .expect("original checkpoint")
                        .custody = None;
                    drop(custody);
                }
                let streams = self.preparation_inputs.consumer_streams.python_value(py)?;
                let streams =
                    checkpoint_consumer_streams(streams.bind(py), &mut (16 * 1024 * 1024))?;
                let report = source
                    .owner()?
                    .finish_cold_model_work(&*self.parent.borrow(py).lease()?, &work, &streams)
                    .map_err(xlog_err)?;
                self.private_checkpoints()?
                    .last_mut()
                    .expect("original checkpoint")
                    .report = Some(report);
                report
            }
        };
        let finish = {
            let mut retained = self.private_checkpoints()?;
            let current = retained.last_mut().expect("original checkpoint");
            let finish = !current.observer_finish_entered;
            current.observer_finish_entered = true;
            finish
        };
        if finish {
            self.preparation_inputs.resource_observer.finish(py)?;
        }
        let cached_peak = self
            .private_checkpoints()?
            .last()
            .expect("original checkpoint")
            .backing_peak;
        let peak = match cached_peak {
            Some(peak) => peak,
            None => {
                let peak = self.preparation_inputs.resource_observer.backing_peak(py)?;
                self.private_checkpoints()?
                    .last_mut()
                    .expect("original checkpoint")
                    .backing_peak = Some(peak);
                peak
            }
        };
        let manifest = SemanticCheckpointManifest::decode(saved.bind(py).as_bytes())?;
        let binding = SemanticTransitionSession::state_material_input_projection(&manifest.native)
            .map_err(xlog_err)?;
        let work = report
            .model_work
            .checked_add(report.native_work)
            .ok_or_else(|| invalid("private checkpoint expenditure overflowed"))?;
        let (instruction, budget, entered) = {
            let retained = self.private_checkpoints()?;
            let current = retained.last().expect("original checkpoint");
            (
                current.instruction.clone(),
                current.budget,
                current.record_entered,
            )
        };
        if entered {
            return Err(invalid(
                "unknown private checkpoint history append cannot repeat its original callback",
            ));
        }
        let arguments = PyDict::new(py);
        arguments.set_item("instruction_bytes", PyBytes::new(py, &instruction))?;
        arguments.set_item("operation_ordinal", ordinal)?;
        arguments.set_item("branch", branch)?;
        arguments.set_item("parent", parent.borrow(py).identity(py)?)?;
        arguments.set_item(
            "model_binding",
            (
                binding.model_generation,
                PyBytes::new(py, binding.model_geometry_digest.as_bytes()),
                PyBytes::new(py, binding.model_numerical_digest.as_bytes()),
            ),
        )?;
        arguments.set_item("full_checkpoint", saved.bind(py))?;
        arguments.set_item("resource_usage", (work, peak, report.model_calls))?;
        self.preparation_inputs
            .resource_observer
            .observation_arguments(py, &arguments)?;
        let callback = self
            .scientific_owner
            .bind(py)
            .getattr("record_checkpoint")?;
        {
            let mut retained = self.private_checkpoints()?;
            let current = retained.last_mut().expect("original checkpoint");
            current.budget_exceeded =
                work > budget[0] || peak > budget[1] || report.model_calls > budget[2];
            current.record_entered = true;
        }
        callback.call((), Some(&arguments))?;
        self.require_scientific_history(py)?;
        self.private_checkpoints()?
            .last_mut()
            .expect("original checkpoint")
            .recorded = true;
        self.release_private_checkpoint(py)
    }

    fn decode_private_checkpoint(&self, py: Python<'_>) -> PyResult<()> {
        let (entered, decoded, entries) = {
            let retained = self.private_checkpoints()?;
            let current = retained.last().expect("original checkpoint");
            Self::require_private_checkpoint_entry(py, current)?;
            (
                current.decode_entered,
                current.decoded.as_ref().map(|value| value.clone_ref(py)),
                current.entries.clone_ref(py),
            )
        };
        let decoded = if let Some(decoded) = decoded {
            decoded
        } else {
            if entered {
                return Err(invalid(
                    "unknown private checkpoint decode cannot repeat its original callback",
                ));
            }
            let decoder = self
                .store()?
                .as_ref()
                .ok_or_else(|| invalid("private checkpoint lost its original decoder"))?
                .decode
                .clone_ref(py);
            self.private_checkpoints()?
                .last_mut()
                .expect("original checkpoint")
                .decode_entered = true;
            let decoded = decoder
                .bind(py)
                .call1((entries.bind(py).get_item(0)?,))?
                .unbind();
            self.private_checkpoints()?
                .last_mut()
                .expect("original checkpoint")
                .decoded = Some(decoded.clone_ref(py));
            decoded
        };
        let decoded = decoded.bind(py);
        if !decoded.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "private checkpoint requires its original closed decoder tuple",
            ));
        }
        let decoded = decoded.cast::<PyTuple>()?;
        if decoded.len() != 3
            || ColdValue::read(&decoded.get_item(0)?, &mut 128, 0)?.text()? != "checkpoint"
            || !decoded.get_item(1)?.is_none()
            || !decoded.get_item(2)?.is_none()
        {
            return Err(invalid(
                "private checkpoint decoder changed its original lifecycle operation",
            ));
        }
        Ok(())
    }

    fn release_private_checkpoint(&self, py: Python<'_>) -> PyResult<()> {
        let (recorded, released, refused) = {
            let retained = self.private_checkpoints()?;
            let current = retained.last().expect("original checkpoint");
            (current.recorded, current.released, current.budget_exceeded)
        };
        if !recorded {
            return Err(invalid(
                "private checkpoint release precedes its known original record",
            ));
        }
        if !released {
            self.preparation_inputs.resource_observer.release(py)?;
            self.private_checkpoints()?
                .last_mut()
                .expect("original checkpoint")
                .released = true;
        }
        if refused {
            return Err(invalid(
                "private checkpoint exceeded its original operation budget",
            ));
        }
        Ok(())
    }
}
