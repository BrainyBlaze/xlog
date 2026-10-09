//! Original private trajectories share the canonical decoded source restore.

use super::*;
use xlog_cuda::{SemanticColdModelWork, SemanticColdModelWorkResult, SemanticColdNativeWork};

pub(super) struct PrivateTrajectoryStart {
    branch: &'static str,
    entries: Py<PyTuple>,
    material: Vec<u8>,
    instruction: Vec<u8>,
    ordinal: u64,
    budget: [u64; 3],
    decode_entered: bool,
    decoded: Option<Py<PyAny>>,
    decoded_verified: bool,
    started: bool,
    restore_entered: bool,
    restored: Option<Py<PySemanticTransitionRestoredCheckpoint>>,
    restore_error: Option<PyErr>,
    work: Option<SemanticColdModelWork>,
    callback_work: Option<Py<PySemanticColdModelWork>>,
    child_joined: bool,
    report: Option<SemanticColdModelWorkResult>,
    model_binding: Option<(u64, Identity256, Identity256)>,
    observer_finish_entered: bool,
    backing_peak: Option<u64>,
    record_entered: bool,
    recorded: bool,
    released: bool,
}

impl PySemanticLearningPhaseTransition {
    pub(super) fn drop_completed_private_restore_owners(
        &self,
        branch: &'static str,
    ) -> PyResult<()> {
        let original = {
            let mut retained = self.trajectory_start()?;
            if retained.as_ref().is_some_and(|entry| {
                entry.branch != branch
                    || !entry.recorded
                    || !entry.released
                    || !entry.child_joined
                    || entry.restore_error.is_some()
            }) {
                return Err(invalid(
                    "model retirement cannot discard an unfinished private restore",
                ));
            }
            retained.take()
        };
        drop(original);
        self.drop_completed_intermediate_restore(branch)?;
        Ok(())
    }

    pub(super) fn drop_completed_trajectory_model_references(&self) -> PyResult<()> {
        let original = {
            let mut retained = self.trajectory_start()?;
            let original = retained
                .as_mut()
                .ok_or_else(|| invalid("intermediate restore lost its original trajectory"))?;
            if !original.recorded
                || !original.released
                || !original.child_joined
                || original.restore_error.is_some()
            {
                return Err(invalid(
                    "intermediate restore cannot discard an unfinished trajectory",
                ));
            }
            (original.restored.take(), original.callback_work.take())
        };
        drop(original);
        Ok(())
    }

    fn trajectory_start(&self) -> PyResult<MutexGuard<'_, Option<PrivateTrajectoryStart>>> {
        self.private_trajectory_start
            .lock()
            .map_err(|_| invalid("original private trajectory custody mutex is poisoned"))
    }

    fn require_trajectory_entry(&self, py: Python<'_>) -> PyResult<()> {
        let retained = self.trajectory_start()?;
        let operation = retained
            .as_ref()
            .ok_or_else(|| invalid("private restore lost its original scheduled entry"))?;
        let (material, instruction) =
            Self::singleton_lifecycle_material(operation.entries.bind(py))?;
        if material != operation.material || instruction != operation.instruction {
            return Err(invalid("private restore changed its original frozen entry"));
        }
        Ok(())
    }

    fn capture_private_trajectory(&self, py: Python<'_>, branch: &'static str) -> PyResult<()> {
        let expected = match branch {
            "control" => u64::try_from(self.source_evaluation_count()?)
                .ok()
                .and_then(|count| count.checked_add(1))
                .ok_or_else(|| invalid("private trajectory position overflowed"))?,
            "real" => self.real_trajectory_ordinal()?,
            _ => {
                return Err(invalid(
                    "private source restore requires its original real or control arm",
                ))
            }
        };
        if self
            .trajectory_start()?
            .as_ref()
            .is_some_and(|entry| entry.branch != branch || entry.ordinal != expected)
        {
            return Err(invalid(
                "private trajectory cannot replace its original branch or position",
            ));
        }
        if self.trajectory_start()?.is_none() {
            self.preparation_inputs
                .require_program(py, &self.scientific_owner)?;
            let next = self.scientific_owner.bind(py).getattr("next_operation")?;
            if !next.is_exact_instance_of::<PyTuple>() {
                return Err(invalid(
                    "private trajectory requires the original complete scheduled group",
                ));
            }
            let entries = next.cast::<PyTuple>()?;
            let (material, instruction) = Self::singleton_lifecycle_material(entries)?;
            let fields = ColdValue::from_canonical_bytes(&material)?;
            let fields = fields.fields(6)?;
            let ordinal = fields[2].unsigned()?;
            if fields[0].text()? != "trajectory-start"
                || fields[1].text()? != branch
                || ordinal != expected
                || fields[3].unsigned()? != 0
                || fields[5] != ColdValue::None
            {
                return Err(invalid(
                    "private trajectory differs from its original branch source restore",
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
                    "private trajectory requires its original backing memory limit",
                ));
            }
            *self.trajectory_start()? = Some(PrivateTrajectoryStart {
                branch,
                entries: entries.clone().unbind(),
                material,
                instruction,
                ordinal,
                budget,
                decode_entered: false,
                decoded: None,
                decoded_verified: false,
                started: false,
                restore_entered: false,
                restored: None,
                restore_error: None,
                work: None,
                callback_work: None,
                child_joined: false,
                report: None,
                model_binding: None,
                observer_finish_entered: false,
                backing_peak: None,
                record_entered: false,
                recorded: false,
                released: false,
            });
        }
        self.require_trajectory_entry(py)?;
        let (entered, decoded) = {
            let retained = self.trajectory_start()?;
            let operation = retained.as_ref().expect("retained original trajectory");
            (
                operation.decode_entered,
                operation.decoded.as_ref().map(|value| value.clone_ref(py)),
            )
        };
        let decoded = match decoded {
            Some(decoded) => decoded,
            None if entered => {
                return Err(invalid(
                    "unknown private instruction decode cannot repeat its original callback",
                ))
            }
            None => {
                let entries = self
                    .trajectory_start()?
                    .as_ref()
                    .expect("retained original trajectory")
                    .entries
                    .clone_ref(py);
                let entry = entries.bind(py).get_item(0)?;
                let decode = self
                    .store()?
                    .as_ref()
                    .ok_or_else(|| {
                        invalid("private trajectory lost the original lifecycle decoder")
                    })?
                    .decode
                    .clone_ref(py);
                self.trajectory_start()?
                    .as_mut()
                    .expect("retained original trajectory")
                    .decode_entered = true;
                let result = decode.bind(py).call1((entry,))?.unbind();
                self.trajectory_start()?
                    .as_mut()
                    .expect("retained original trajectory")
                    .decoded = Some(result.clone_ref(py));
                result
            }
        };
        let decoded = decoded.bind(py);
        if !decoded.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "private restore requires its original closed decoder tuple",
            ));
        }
        let decoded = decoded.cast::<PyTuple>()?;
        if decoded.len() != 3
            || ColdValue::read(&decoded.get_item(0)?, &mut 128, 0)?.text()? != "trajectory-start"
            || ColdValue::read(&decoded.get_item(1)?, &mut 128, 0)?.text()? != "source"
            || !decoded.get_item(2)?.is_none()
        {
            return Err(invalid(
                "trajectory-start must restore the original source without copy/reset",
            ));
        }
        self.trajectory_start()?
            .as_mut()
            .expect("retained original trajectory")
            .decoded_verified = true;
        Ok(())
    }

    pub(super) fn require_private_restore_admission(
        &self,
        py: Python<'_>,
        checkpoint: &[u8],
        transition: Option<&SemanticLearningPhaseTransition>,
    ) -> PyResult<()> {
        if self.intermediate_restore_pending()? {
            return self.require_intermediate_restore_admission(py, checkpoint, transition);
        }
        self.records()?.require_preparation_admission()?;
        self.preparation_inputs
            .require_program(py, &self.scientific_owner)?;
        self.require_trajectory_entry(py)?;
        let retained = self.trajectory_start()?;
        let operation = retained.as_ref().expect("retained original trajectory");
        if checkpoint != self.source_checkpoint
            || transition.is_some()
            || !operation.decoded_verified
            || !operation.started
            || !operation.restore_entered
            || operation.restored.is_some()
            || operation.restore_error.is_some()
            || operation.work.is_none()
        {
            return Err(invalid(
                "private restore changed its actual original source, decoder or single invocation",
            ));
        }
        Ok(())
    }

    pub(super) fn private_restore_native_work(
        &self,
        py: Python<'_>,
    ) -> PyResult<SemanticColdNativeWork> {
        if self.intermediate_restore_pending()? {
            return self.intermediate_restore_native_work(py);
        }
        let work = self
            .trajectory_start()?
            .as_ref()
            .and_then(|operation| operation.work.clone())
            .ok_or_else(|| {
                invalid("private construction lost its original pre-allocation work owner")
            })?;
        self.source
            .borrow(py)
            .owner()?
            .share_cold_native_work(&work)
            .map_err(xlog_err)
    }

    pub(in crate::semantic_transition) fn restore_callback_work(
        &self,
        py: Python<'_>,
        parent: &Py<PySemanticPublishedParent>,
    ) -> PyResult<Py<PySemanticColdModelWork>> {
        if self.intermediate_restore_pending()? {
            return self.intermediate_factory_work(py, parent);
        }
        self.require_private_restore_admission(py, &self.source_checkpoint, None)?;
        {
            let retained = self.private_restore()?;
            if retained
                .as_ref()
                .and_then(|owners| owners.parent.as_ref())
                .is_none_or(|original| original.as_ptr() != parent.as_ptr())
            {
                return Err(invalid(
                    "private factory changed its original retained child parent",
                ));
            }
        }
        let mut retained = self.trajectory_start()?;
        let operation = retained.as_mut().expect("retained original trajectory");
        if operation.callback_work.is_some() {
            return Err(invalid(
                "private model factory cannot replace its original registrar",
            ));
        }
        let work = Py::new(
            py,
            PySemanticColdModelWork {
                parent: parent.clone_ref(py),
                reader: self.parent.clone_ref(py),
                inner: operation
                    .work
                    .as_ref()
                    .expect("retained original work")
                    .clone(),
                region: None,
                active: AtomicBool::new(false),
            },
        )?;
        operation.callback_work = Some(work.clone_ref(py));
        Ok(work)
    }

    pub(in crate::semantic_transition) fn private_restore_feedback_projection(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        parent: &Py<PySemanticPublishedParent>,
    ) -> PyResult<Py<PyTuple>> {
        self.records()?.require_preparation_admission()?;
        self.require_trajectory_entry(py)?;
        let retained = self.private_restore()?;
        let owners = retained
            .as_ref()
            .ok_or_else(|| invalid("private feedback lost its actual child owners"))?;
        if !std::ptr::eq(&*owners.session.borrow(py), session)
            || owners
                .parent
                .as_ref()
                .is_none_or(|original| original.as_ptr() != parent.as_ptr())
        {
            return Err(invalid(
                "private feedback changed its original child Session or parent",
            ));
        }
        drop(retained);
        let intermediate = self.intermediate_feedback_work(py)?;
        let (branch, original) = if let Some(original) = intermediate {
            original
        } else {
            let retained = self.trajectory_start()?;
            let operation = retained.as_ref().expect("original private trajectory");
            (
                operation.branch,
                operation
                    .callback_work
                    .as_ref()
                    .ok_or_else(|| {
                        invalid("private feedback lacks its original model factory registrar")
                    })?
                    .clone_ref(py),
            )
        };
        original.borrow(py).check(py)?;
        let materials = feedback_materials(
            self.preparation_inputs
                .feedback_interventions
                .bind(py)
                .as_any(),
        )?;
        // Preserve the actual frozen intervention: real uses the original
        // feedback tensor; control executes the existing positive-zero path.
        let (material, enabled) = match branch {
            "real" => (materials[0].clone(), true),
            "control" => (materials[1].clone(), false),
            _ => return Err(invalid("private feedback lost its original branch")),
        };
        Ok((material, enabled).into_pyobject(py)?.unbind())
    }

    pub(super) fn prepare_private_trajectory(
        &self,
        py: Python<'_>,
        pending: &Py<Self>,
        branch: &'static str,
    ) -> PyResult<()> {
        self.capture_private_trajectory(py, branch)?;
        if self
            .trajectory_start()?
            .as_ref()
            .is_some_and(|original| original.recorded && original.released)
        {
            // The initial scientific record remains immutable after an
            // intermediate retirement drops its superseded model references.
            return self.require_completed_trajectory_budget();
        }
        let (started, entered, restored, error, ordinal) = {
            let retained = self.trajectory_start()?;
            let operation = retained.as_ref().expect("retained original trajectory");
            (
                operation.started,
                operation.restore_entered,
                operation.restored.as_ref().map(|value| value.clone_ref(py)),
                operation
                    .restore_error
                    .as_ref()
                    .map(|error| error.clone_ref(py)),
                operation.ordinal,
            )
        };
        if let Some(error) = error {
            return Err(error);
        }
        if !started {
            self.trajectory_start()?
                .as_mut()
                .expect("retained original trajectory")
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
                    xlog_cuda::SemanticColdModelWorkPurpose::PrivateRestore,
                )
                .map_err(xlog_err)?;
            self.trajectory_start()?
                .as_mut()
                .expect("retained original trajectory")
                .work = Some(work);
        }
        let restored = match restored {
            Some(restored) => restored,
            None if entered => return Err(invalid("unknown private restoration retains its original child and cannot enter a replacement factory")),
            None => {
                let snapshot = self.refresh_snapshot.bind(py).call0()?;
                let latest = AuthoritySnapshot::parse(&ColdValue::read(&snapshot, &mut (16 * 1024 * 1024), 0)?)?;
                let task = self.task_use.borrow(py);
                latest.newer_than(&task.state()?.snapshot)?;
                check_learning_grant(&task, &self.grant_reference, &latest)?;
                self.require_trajectory_entry(py)?;
                self.preparation_inputs.require_program(py, &self.scientific_owner)?;
                let checkpoint = PyBytes::new(py, &self.source_checkpoint);
                let domain = task.checkpoint.training_domain.python_value(py)?;
                let checkpoint_limit = self.preparation_inputs.max_checkpoint_bytes.as_ref()
                    .map(|value| value.python_value(py)).transpose()?;
                let total_limit = self.preparation_inputs.max_total_checkpoint_bytes.as_ref()
                    .map(|value| value.python_value(py)).transpose()?;
                if branch == "control" {
                    self.candidate_entered.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                        .map_err(|_| invalid("private trajectory has already entered its original restore"))?;
                } else if !self.candidate_entered.load(Ordering::Acquire) {
                    return Err(invalid("real restoration lost the original entered phase"));
                }
                self.trajectory_start()?.as_mut().expect("retained original trajectory").restore_entered = true;
                let result = PySemanticTransitionSession::restore_checkpoint_impl(
                    py, checkpoint.as_any(), self.source.borrow(py).device_ordinal, &snapshot,
                    self.model_owner(py, PhaseModelOwner::Restore)?.bind(py), domain.bind(py),
                    task.checkpoint.training_canary_owner.as_ref().map(|owner| owner.bind(py).as_any()), None, None,
                    self.preparation_inputs.resolve_checkpoint.as_ref().map(|callback| callback.bind(py)),
                    checkpoint_limit.as_ref().map(|value| value.bind(py)),
                    total_limit.as_ref().map(|value| value.bind(py)), Some(self.refresh_snapshot.bind(py)),
                    Some(&task.checkpoint.proposal_expense), Some(&task.checkpoint.checkpoint_sources), Some(pending),
                );
                let mut retained = self.trajectory_start()?;
                let operation = retained.as_mut().expect("retained original trajectory");
                match result {
                    Ok(restored) => { operation.restored = Some(restored.clone_ref(py)); restored }
                    Err(error) => { operation.restore_error = Some(error.clone_ref(py)); return Err(error); }
                }
            }
        };
        self.finish_private_trajectory(py, &restored)
    }

    fn finish_private_trajectory(
        &self,
        py: Python<'_>,
        restored: &Py<PySemanticTransitionRestoredCheckpoint>,
    ) -> PyResult<()> {
        {
            let retained = self.trajectory_start()?;
            let operation = retained.as_ref().expect("retained original trajectory");
            if operation.recorded && operation.released {
                let report = operation.report.expect("known original trajectory report");
                let work = report
                    .native_work
                    .checked_add(report.model_work)
                    .ok_or_else(|| invalid("private trajectory actual work overflowed"))?;
                let peak = operation
                    .backing_peak
                    .expect("known original trajectory memory");
                if work > operation.budget[0]
                    || peak > operation.budget[1]
                    || report.model_calls > operation.budget[2]
                {
                    return Err(invalid("private trajectory exceeded its original budget; retain the actual recorded expenditure"));
                }
                // A later operation may now own the observer. Read only this
                // completed original restore, never that later interval.
                return Ok(());
            }
        }
        let streams = checkpoint_consumer_streams(
            self.preparation_inputs
                .consumer_streams
                .python_value(py)?
                .bind(py),
            &mut (16 * 1024 * 1024),
        )?;
        let joined = self
            .trajectory_start()?
            .as_ref()
            .expect("retained original trajectory")
            .child_joined;
        if !joined {
            let restored = restored.borrow(py);
            // Project once while this same tally and whole backing interval
            // are still open. History readback never reserializes native state.
            if self
                .trajectory_start()?
                .as_ref()
                .expect("retained original trajectory")
                .model_binding
                .is_none()
            {
                let native = restored
                    .session
                    .borrow(py)
                    .owner()?
                    .published_state_material(&*restored.parent.borrow(py).lease()?)
                    .map_err(xlog_err)?;
                let model = SemanticTransitionSession::state_material_input_projection(&native)
                    .map_err(xlog_err)?;
                self.trajectory_start()?
                    .as_mut()
                    .expect("retained original trajectory")
                    .model_binding = Some((
                    model.model_generation,
                    model.model_geometry_digest,
                    model.model_numerical_digest,
                ));
            }
            restored
                .session
                .borrow(py)
                .owner()?
                .complete_shared_cold_native_work(&*restored.parent.borrow(py).lease()?, &streams)
                .map_err(xlog_err)?;
            self.trajectory_start()?
                .as_mut()
                .expect("retained original trajectory")
                .child_joined = true;
        }
        let (work, report) = {
            let retained = self.trajectory_start()?;
            let operation = retained.as_ref().expect("retained original trajectory");
            (
                operation
                    .work
                    .as_ref()
                    .expect("original pre-allocation work")
                    .clone(),
                operation.report,
            )
        };
        let report = match report {
            Some(report) => report,
            None => {
                let source = self.source.borrow(py);
                let parent = self.parent.borrow(py);
                // Native dispatches from its actual Closed/Submitted state.
                // A callback attempt flag cannot prove report submission.
                let report = source
                    .owner()?
                    .finish_cold_model_work(
                        &*parent.lease()?,
                        &work,
                        &streams,
                        xlog_cuda::SemanticColdModelWorkDisposition::Complete,
                    )
                    .map_err(xlog_err)?;
                self.trajectory_start()?
                    .as_mut()
                    .expect("retained original trajectory")
                    .report = Some(report);
                report
            }
        };
        let finish = {
            let mut retained = self.trajectory_start()?;
            let operation = retained.as_mut().expect("retained original trajectory");
            let finish = !operation.observer_finish_entered;
            operation.observer_finish_entered = true;
            finish
        };
        if finish {
            self.preparation_inputs.resource_observer.finish(py)?;
        }
        let peak = self.preparation_inputs.resource_observer.backing_peak(py)?;
        self.trajectory_start()?
            .as_mut()
            .expect("retained original trajectory")
            .backing_peak = Some(peak);
        let (instruction, ordinal, budget, entered, recorded) = {
            let retained = self.trajectory_start()?;
            let operation = retained.as_ref().expect("retained original trajectory");
            (
                operation.instruction.clone(),
                operation.ordinal,
                operation.budget,
                operation.record_entered,
                operation.recorded,
            )
        };
        let work = report
            .native_work
            .checked_add(report.model_work)
            .ok_or_else(|| invalid("private trajectory actual work overflowed"))?;
        if !recorded {
            if entered {
                return Err(invalid(
                    "unknown private trajectory history append cannot repeat its original callback",
                ));
            }
            let restored = restored.borrow(py);
            let parent = restored.parent.borrow(py);
            let model = self
                .trajectory_start()?
                .as_ref()
                .expect("retained original trajectory")
                .model_binding
                .ok_or_else(|| {
                    invalid("private trajectory lost its original completed model projection")
                })?;
            let arguments = PyDict::new(py);
            arguments.set_item("instruction_bytes", PyBytes::new(py, &instruction))?;
            arguments.set_item("operation_ordinal", ordinal)?;
            let branch = self
                .trajectory_start()?
                .as_ref()
                .expect("original private trajectory")
                .branch;
            arguments.set_item("branch", branch)?;
            arguments.set_item("restored_parent", parent.identity(py)?)?;
            arguments.set_item(
                "model_binding",
                (
                    model.0,
                    PyBytes::new(py, model.1.as_bytes()),
                    PyBytes::new(py, model.2.as_bytes()),
                ),
            )?;
            arguments.set_item(
                "source_checkpoint",
                PyBytes::new(py, &self.source_checkpoint),
            )?;
            arguments.set_item("resource_usage", (work, peak, report.model_calls))?;
            self.preparation_inputs
                .resource_observer
                .observation_arguments(py, &arguments)?;
            let callback = self
                .scientific_owner
                .bind(py)
                .getattr("record_trajectory_start")?;
            self.trajectory_start()?
                .as_mut()
                .expect("retained original trajectory")
                .record_entered = true;
            callback.call((), Some(&arguments))?;
            self.require_scientific_history(py)?;
            self.trajectory_start()?
                .as_mut()
                .expect("retained original trajectory")
                .recorded = true;
        }
        self.preparation_inputs.resource_observer.release(py)?;
        self.trajectory_start()?
            .as_mut()
            .expect("retained original trajectory")
            .released = true;
        if work > budget[0] || peak > budget[1] || report.model_calls > budget[2] {
            return Err(invalid("private trajectory exceeded its original budget; retain the actual recorded expenditure"));
        }
        Ok(())
    }

    pub(super) fn require_known_private_restore(&self, py: Python<'_>) -> PyResult<()> {
        if self.intermediate_restore_retained()? {
            return self.require_intermediate_continuation(py);
        }
        self.records()?.require_preparation_admission()?;
        self.require_trajectory_entry(py)?;
        let retained = self.trajectory_start()?;
        let operation = retained.as_ref().expect("retained original trajectory");
        if let Some(error) = &operation.restore_error {
            return Err(error.clone_ref(py));
        }
        if operation.restored.is_none() {
            return Err(invalid(
                "unknown private construction cannot repeat the original model factory",
            ));
        }
        Ok(())
    }

    pub(super) fn private_execution_input(
        &self,
        py: Python<'_>,
        branch: &'static str,
    ) -> PyResult<(Py<PySemanticTransitionRestoredCheckpoint>, u64)> {
        if let Some(input) = self.intermediate_execution_input(py, branch)? {
            return Ok(input);
        }
        self.require_known_private_restore(py)?;
        let retained = self.trajectory_start()?;
        let operation = retained.as_ref().expect("retained original trajectory");
        if operation.branch != branch {
            return Err(invalid(
                "private numerical execution requires its original branch restore",
            ));
        }
        if !operation.recorded || operation.report.is_none() || !operation.child_joined {
            return Err(invalid(
                "private numerical execution requires its actual recorded source restore",
            ));
        }
        Ok((
            operation
                .restored
                .as_ref()
                .expect("known restore")
                .clone_ref(py),
            operation
                .ordinal
                .checked_add(1)
                .ok_or_else(|| invalid("private execution position overflowed"))?,
        ))
    }

    fn require_completed_trajectory_budget(&self) -> PyResult<()> {
        let retained = self.trajectory_start()?;
        let original = retained.as_ref().expect("original completed trajectory");
        let report = original
            .report
            .ok_or_else(|| invalid("completed trajectory lost its original report"))?;
        let work = report
            .native_work
            .checked_add(report.model_work)
            .ok_or_else(|| invalid("trajectory work overflowed"))?;
        let peak = original
            .backing_peak
            .ok_or_else(|| invalid("completed trajectory lost its original backing peak"))?;
        if work > original.budget[0]
            || peak > original.budget[1]
            || report.model_calls > original.budget[2]
        {
            return Err(invalid(
                "original trajectory exceeded its recorded operation budget",
            ));
        }
        Ok(())
    }
}
