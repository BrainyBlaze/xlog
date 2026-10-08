//! Complete original control-model and native Session retirement.

use super::phase_evaluation::EvaluationOwners;
use super::*;
use xlog_cuda::{SemanticColdModelWork, SemanticColdModelWorkResult, SemanticColdNativeWork};

pub(super) struct ControlRetirement {
    owners: Option<EvaluationOwners>,
    checkpoint: Py<PyBytes>,
    checkpoint_sha256: [u8; 32],
    parent_identity: Py<PyTuple>,
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
    callback_entered: bool,
    callback_completed: bool,
    error: Option<PyErr>,
    session_release_entered: bool,
    session_released: bool,
    model_owners_dropped: bool,
    owners_dropped: bool,
    report: Option<SemanticColdModelWorkResult>,
    observer_finish_entered: bool,
    physical_peak: Option<u64>,
    record_entered: bool,
    recorded: bool,
    released: bool,
    budget_exceeded: bool,
}

impl PySemanticLearningPhaseTransition {
    fn control_retirement(&self) -> PyResult<MutexGuard<'_, Option<ControlRetirement>>> {
        self.control_retirement
            .lock()
            .map_err(|_| invalid("original control retirement custody mutex is poisoned"))
    }

    pub(super) fn control_retirement_retained(&self) -> PyResult<bool> {
        Ok(self.control_retirement()?.is_some())
    }

    pub(super) fn real_trajectory_ordinal(&self) -> PyResult<u64> {
        let retained = self.control_retirement()?;
        let current = retained
            .as_ref()
            .ok_or_else(|| invalid("real trajectory precedes original control retirement"))?;
        if !current.recorded
            || !current.released
            || !current.session_released
            || !current.model_owners_dropped
            || !current.owners_dropped
            || current.budget_exceeded
            || current.error.is_some()
        {
            return Err(invalid("real trajectory requires known complete original control retirement and accounting"));
        }
        current
            .ordinal
            .checked_add(1)
            .ok_or_else(|| invalid("real trajectory position overflowed"))
    }

    pub(super) fn execute_control_branch(
        &self,
        py: Python<'_>,
        pending: &Py<Self>,
    ) -> PyResult<()> {
        if !self.control_retirement_retained()? {
            self.prepare_private_trajectory(py, pending, "control")?;
            self.execute_private_numerical_sequence(py, pending, "control")?;
            if self.readonly_terminal_refusal_retained()? {
                return Ok(());
            }
            self.execute_private_evaluations(py, "control")?;
            if self.readonly_terminal_refusal_retained()? {
                return Ok(());
            }
            self.execute_private_checkpoint(py, "control")?;
        }
        self.execute_control_retirement(py)
    }

    fn require_control_retirement_entry(
        py: Python<'_>,
        current: &ControlRetirement,
    ) -> PyResult<()> {
        let (material, instruction) = Self::singleton_lifecycle_material(current.entries.bind(py))?;
        if material != current.material || instruction != current.instruction {
            return Err(invalid(
                "control retirement changed its original frozen entry",
            ));
        }
        let checkpoint_sha256: [u8; 32] =
            Sha256::digest(current.checkpoint.bind(py).as_bytes()).into();
        if checkpoint_sha256 != current.checkpoint_sha256 {
            return Err(invalid(
                "control retirement changed its original complete checkpoint",
            ));
        }
        Ok(())
    }

    fn execute_control_retirement(&self, py: Python<'_>) -> PyResult<()> {
        self.records()?.require_preparation_admission()?;
        self.preparation_inputs
            .require_program(py, &self.scientific_owner)?;
        if !self.control_retirement_retained()? {
            let (owners, checkpoint, ordinal) = self.control_retirement_input(py)?;
            let next = self.scientific_owner.bind(py).getattr("next_operation")?;
            if !next.is_exact_instance_of::<PyTuple>() {
                return Err(invalid(
                    "control retirement requires its complete original scheduled group",
                ));
            }
            let entries = next.cast::<PyTuple>()?;
            let (material, instruction) = Self::singleton_lifecycle_material(entries)?;
            let fields = ColdValue::from_canonical_bytes(&material)?;
            let fields = fields.fields(6)?;
            if fields[0].text()? != "retire"
                || fields[1].text()? != "control"
                || fields[2].unsigned()? != ordinal
                || fields[3].unsigned()? != 0
                || fields[5] != ColdValue::None
            {
                return Err(invalid(
                    "control retirement changed its original lifecycle position",
                ));
            }
            let budget = fields[4].fields(3)?;
            let budget = [
                budget[0].unsigned()?,
                budget[1].unsigned()?,
                budget[2].unsigned()?,
            ];
            let parent_identity = owners.parent.borrow(py).identity(py)?;
            let checkpoint_sha256: [u8; 32] = Sha256::digest(checkpoint.bind(py).as_bytes()).into();
            *self.control_retirement()? = Some(ControlRetirement {
                owners: Some(owners),
                checkpoint,
                checkpoint_sha256,
                parent_identity,
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
                callback_entered: false,
                callback_completed: false,
                error: None,
                session_release_entered: false,
                session_released: false,
                model_owners_dropped: false,
                owners_dropped: false,
                report: None,
                observer_finish_entered: false,
                physical_peak: None,
                record_entered: false,
                recorded: false,
                released: false,
                budget_exceeded: false,
            });
        }
        let recorded = {
            let retained = self.control_retirement()?;
            let current = retained.as_ref().expect("original control retirement");
            Self::require_control_retirement_entry(py, current)?;
            if let Some(error) = &current.error {
                return Err(error.clone_ref(py));
            }
            current.recorded
        };
        if recorded {
            return self.release_control_retirement(py);
        }
        self.decode_control_retirement(py)?;
        let (started, ready, ordinal) = {
            let retained = self.control_retirement()?;
            let current = retained.as_ref().expect("original control retirement");
            (current.started, current.ready, current.ordinal)
        };
        if !started {
            let (task, parent) = {
                let retained = self.control_retirement()?;
                let owners = retained
                    .as_ref()
                    .expect("original control retirement")
                    .owners
                    .as_ref()
                    .expect("original control owners");
                (owners.task.clone_ref(py), owners.parent.clone_ref(py))
            };
            let fresh = self.refresh_snapshot.bind(py).call0()?;
            let authority =
                AuthoritySnapshot::parse(&ColdValue::read(&fresh, &mut (16 * 1024 * 1024), 0)?)?;
            {
                let issued = task.borrow(py);
                authority.newer_than(&issued.state()?.snapshot)?;
                check_learning_grant(&issued, &self.grant_reference, &authority)?;
            }
            self.control_retirement()?
                .as_mut()
                .expect("original control retirement")
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
            self.control_retirement()?
                .as_mut()
                .expect("original control retirement")
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
            self.control_retirement()?
                .as_mut()
                .expect("original control retirement")
                .callback_work = Some(callback_work);
            let custody = self
                .source
                .borrow(py)
                .owner()?
                .share_cold_native_work(&work)
                .map_err(xlog_err)?;
            self.control_retirement()?
                .as_mut()
                .expect("original control retirement")
                .custody = Some(custody.clone());
            let original_parent = parent.borrow(py);
            original_parent
                .session
                .borrow(py)
                .owner()?
                .attach_shared_cold_native_work(&*original_parent.lease()?, custody)
                .map_err(xlog_err)?;
            self.control_retirement()?
                .as_mut()
                .expect("original control retirement")
                .ready = true;
        } else if !ready {
            return Err(invalid(
                "unknown control retirement admission retains its original interval and owners",
            ));
        }
        self.retire_original_control_owners(py)?;
        let (work, report) = {
            let retained = self.control_retirement()?;
            let current = retained.as_ref().expect("original control retirement");
            (
                current.work.clone().expect("original retirement work"),
                current.report,
            )
        };
        let report = if let Some(report) = report {
            report
        } else {
            let streams = self.preparation_inputs.consumer_streams.python_value(py)?;
            let streams = checkpoint_consumer_streams(streams.bind(py), &mut (16 * 1024 * 1024))?;
            let report = self
                .source
                .borrow(py)
                .owner()?
                .finish_cold_model_work(&*self.parent.borrow(py).lease()?, &work, &streams)
                .map_err(xlog_err)?;
            self.control_retirement()?
                .as_mut()
                .expect("original control retirement")
                .report = Some(report);
            report
        };
        let finish = {
            let mut retained = self.control_retirement()?;
            let current = retained.as_mut().expect("original control retirement");
            let finish = !current.observer_finish_entered;
            current.observer_finish_entered = true;
            finish
        };
        if finish {
            self.preparation_inputs.resource_observer.finish(py)?;
        }
        let peak = self
            .control_retirement()?
            .as_ref()
            .expect("original control retirement")
            .physical_peak;
        let peak = if let Some(peak) = peak {
            peak
        } else {
            let peak = self
                .preparation_inputs
                .resource_observer
                .physical_peak(py)?;
            self.control_retirement()?
                .as_mut()
                .expect("original control retirement")
                .physical_peak = Some(peak);
            peak
        };
        let work = report
            .native_work
            .checked_add(report.model_work)
            .ok_or_else(|| invalid("control retirement expenditure overflowed"))?;
        let (instruction, parent, budget, entered) = {
            let retained = self.control_retirement()?;
            let current = retained.as_ref().expect("original control retirement");
            (
                current.instruction.clone(),
                current.parent_identity.clone_ref(py),
                current.budget,
                current.record_entered,
            )
        };
        if entered {
            return Err(invalid(
                "unknown control retirement history append cannot repeat its original callback",
            ));
        }
        let arguments = PyDict::new(py);
        arguments.set_item("instruction_bytes", PyBytes::new(py, &instruction))?;
        arguments.set_item("operation_ordinal", ordinal)?;
        arguments.set_item("retired_parent", parent.bind(py))?;
        arguments.set_item("resource_usage", (work, peak, report.model_calls))?;
        let callback = self
            .scientific_owner
            .bind(py)
            .getattr("record_retirement")?;
        {
            let mut retained = self.control_retirement()?;
            let current = retained.as_mut().expect("original control retirement");
            current.record_entered = true;
            current.budget_exceeded =
                work > budget[0] || peak > budget[1] || report.model_calls > budget[2];
        }
        callback.call((), Some(&arguments))?;
        self.control_retirement()?
            .as_mut()
            .expect("original control retirement")
            .recorded = true;
        self.release_control_retirement(py)
    }

    fn decode_control_retirement(&self, py: Python<'_>) -> PyResult<()> {
        let (entered, decoded, entries) = {
            let retained = self.control_retirement()?;
            let current = retained.as_ref().expect("original control retirement");
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
                    "unknown control retirement decode cannot repeat its original callback",
                ));
            }
            let decoder = self
                .store()?
                .as_ref()
                .ok_or_else(|| invalid("control retirement lost its original decoder"))?
                .decode
                .clone_ref(py);
            self.control_retirement()?
                .as_mut()
                .expect("original control retirement")
                .decode_entered = true;
            let decoded = decoder
                .bind(py)
                .call1((entries.bind(py).get_item(0)?,))?
                .unbind();
            self.control_retirement()?
                .as_mut()
                .expect("original control retirement")
                .decoded = Some(decoded.clone_ref(py));
            decoded
        };
        if !decoded.bind(py).is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "control retirement requires its original closed decoder tuple",
            ));
        }
        let decoded = decoded.bind(py).cast::<PyTuple>()?;
        if decoded.len() != 3
            || ColdValue::read(&decoded.get_item(0)?, &mut 128, 0)?.text()? != "retire"
            || !decoded.get_item(1)?.is_none()
            || !decoded.get_item(2)?.is_none()
        {
            return Err(invalid(
                "control retirement decoder changed its original operation",
            ));
        }
        Ok(())
    }

    fn retire_original_control_owners(&self, py: Python<'_>) -> PyResult<()> {
        let (completed, dropped) = {
            let retained = self.control_retirement()?;
            let current = retained.as_ref().expect("original control retirement");
            (current.callback_completed, current.owners_dropped)
        };
        if dropped {
            return Ok(());
        }
        let work = {
            let retained = self.control_retirement()?;
            let current = retained.as_ref().expect("original control retirement");
            current
                .callback_work
                .as_ref()
                .expect("original retirement registrar")
                .clone_ref(py)
        };
        let parent = work.borrow(py).parent.clone_ref(py);
        let task = parent.borrow(py).task_use.clone_ref(py);
        if !completed {
            if self
                .control_retirement()?
                .as_ref()
                .expect("original control retirement")
                .callback_entered
            {
                return Err(invalid(
                    "unknown control retirement cannot repeat its original model release",
                ));
            }
            let fresh = self.refresh_snapshot.bind(py).call0()?;
            let authority =
                AuthoritySnapshot::parse(&ColdValue::read(&fresh, &mut (16 * 1024 * 1024), 0)?)?;
            {
                let issued = task.borrow(py);
                authority.newer_than(&issued.state()?.snapshot)?;
                check_learning_grant(&issued, &self.grant_reference, &authority)?;
            }
            self.control_retirement()?
                .as_mut()
                .expect("original control retirement")
                .callback_entered = true;
            let model = self
                .control_retirement()?
                .as_ref()
                .expect("original control retirement")
                .owners
                .as_ref()
                .expect("original control owners")
                .model
                .clone_ref(py);
            let result = (|| {
                let original_parent = parent.borrow(py);
                let session = original_parent.session.borrow(py);
                let issued = task.borrow(py);
                let _reads = ImportReadScope::checkpoint(&session, &issued, &original_parent, py)?;
                let original_work = work.borrow(py);
                let _cold =
                    ColdCallbackScope::enter(py, &session, &original_work, work.clone_ref(py))?;
                let callback = self.model_owner(py, PhaseModelOwner::RetirePrivate)?;
                let result = callback.bind(py).call1((model,))?;
                if !result.is_none() {
                    return Err(invalid(
                        "original control model retirement must return None after known release",
                    ));
                }
                Ok(())
            })();
            if let Err(error) = result {
                self.control_retirement()?
                    .as_mut()
                    .expect("original control retirement")
                    .error = Some(error.clone_ref(py));
                return Err(error);
            }
            self.control_retirement()?
                .as_mut()
                .expect("original control retirement")
                .callback_completed = true;
        }
        self.source
            .borrow(py)
            .owner()?
            .require_closed_cold_model_work(&work.borrow(py).inner)
            .map_err(xlog_err)?;
        let session = parent.borrow(py).session.clone_ref(py);
        let (released, entered, custody) = {
            let retained = self.control_retirement()?;
            let current = retained.as_ref().expect("original control retirement");
            (
                current.session_released,
                current.session_release_entered,
                current.custody.clone(),
            )
        };
        if !released {
            if entered {
                return Err(invalid(
                    "unknown control native Session release retains its original terminal custody",
                ));
            }
            let original_session = session.borrow(py);
            original_session
                .owner()?
                .require_retired_publication(&*parent.borrow(py).lease()?)
                .map_err(xlog_err)?;
            if let Some(custody) = custody {
                original_session
                    .owner()?
                    .detach_shared_cold_native_work(&custody)
                    .map_err(xlog_err)?;
                self.control_retirement()?
                    .as_mut()
                    .expect("original control retirement")
                    .custody = None;
                drop(custody);
            }
            self.drop_control_model_owners(&session)?;
            self.control_retirement()?
                .as_mut()
                .expect("original control retirement")
                .session_release_entered = true;
            original_session.release_retired_publication(py, &parent.borrow(py))?;
            self.control_retirement()?
                .as_mut()
                .expect("original control retirement")
                .session_released = true;
        }
        task.borrow(py).state()?.phase = TaskUsePhase::Refused;
        let original_session = session.borrow(py);
        original_session
            .learning_preparing
            .store(false, Ordering::Release);
        let retained = original_session
            .learning_transition
            .lock()
            .map_err(|_| invalid("retired control Session lost its original phase mutex"))?
            .take();
        drop(retained);
        drop(original_session);
        drop(session);
        drop(task);
        drop(parent);
        drop(work);
        self.control_retirement()?
            .as_mut()
            .expect("original control retirement")
            .owners_dropped = true;
        Ok(())
    }

    fn drop_control_model_owners(&self, session: &Py<PySemanticTransitionSession>) -> PyResult<()> {
        if self
            .control_retirement()?
            .as_ref()
            .expect("original control retirement")
            .model_owners_dropped
        {
            return Ok(());
        }
        // Publication retirement is already known. Drop every model-bearing
        // frame outside its mutex before the joined Session deallocation drain.
        self.drop_completed_evaluation_owners("control")?;
        self.drop_completed_private_checkpoints("control")?;
        self.drop_completed_private_execution_owners("control")?;
        self.drop_completed_private_restore_owners("control")?;
        let construction = {
            let mut retained = self.private_restore()?;
            if retained
                .as_ref()
                .is_some_and(|owners| owners.session.as_ptr() != session.as_ptr())
            {
                return Err(invalid(
                    "control retirement cannot discard another private construction",
                ));
            }
            retained.take()
        };
        drop(construction);
        let original = self
            .control_retirement()?
            .as_mut()
            .expect("original control retirement")
            .owners
            .take();
        drop(original);
        self.control_retirement()?
            .as_mut()
            .expect("original control retirement")
            .model_owners_dropped = true;
        Ok(())
    }

    fn release_control_retirement(&self, py: Python<'_>) -> PyResult<()> {
        let (released, exceeded) = {
            let retained = self.control_retirement()?;
            let current = retained.as_ref().expect("original control retirement");
            (current.released, current.budget_exceeded)
        };
        if !released {
            self.preparation_inputs.resource_observer.release(py)?;
            self.control_retirement()?
                .as_mut()
                .expect("original control retirement")
                .released = true;
        }
        if exceeded {
            return Err(invalid(
                "control retirement exceeded its original operation budget",
            ));
        }
        Ok(())
    }
}
