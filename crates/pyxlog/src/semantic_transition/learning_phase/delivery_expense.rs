//! Original late verification, publication and source retirement expenditure.

use super::phase_evaluation::EvaluationOwners;
use super::*;
use xlog_cuda::{SemanticColdModelWork, SemanticColdModelWorkResult, SemanticColdNativeWork};

struct DeliveryCall {
    entered: bool,
    result: Option<Py<PyAny>>,
    error: Option<PyErr>,
}

pub(super) struct DeliveryExpense {
    owners: EvaluationOwners,
    entries: Py<PyTuple>,
    material: Vec<u8>,
    instruction: Vec<u8>,
    ordinal: u64,
    budget: [u64; 3],
    started: bool,
    ready: bool,
    work: Option<SemanticColdModelWork>,
    custody: Option<SemanticColdNativeWork>,
    regions: Vec<Py<PySemanticColdModelWork>>,
    calls: Vec<DeliveryCall>,
    source_retired: bool,
    model_owners_dropped: bool,
    source_release_entered: bool,
    source_released: bool,
    candidate_verified: bool,
    work_closed: bool,
    report: Option<SemanticColdModelWorkResult>,
    observer_finish_entered: bool,
    physical_peak: Option<u64>,
    record_entered: bool,
    recorded: bool,
    payload: Option<Arc<[u8]>>,
    released: bool,
    budget_exceeded: bool,
}

impl PySemanticLearningPhaseTransition {
    fn delivery_expense(&self) -> PyResult<MutexGuard<'_, Option<DeliveryExpense>>> {
        self.delivery_expense
            .lock()
            .map_err(|_| invalid("original delivery expense custody mutex is poisoned"))
    }

    fn require_delivery_entry(&self, py: Python<'_>) -> PyResult<()> {
        self.preparation_inputs
            .require_program(py, &self.scientific_owner)?;
        let retained = self.delivery_expense()?;
        let current = retained
            .as_ref()
            .ok_or_else(|| invalid("delivery lost its original frozen tail"))?;
        let (material, instruction) = Self::singleton_lifecycle_material(current.entries.bind(py))?;
        if material != current.material || instruction != current.instruction {
            return Err(invalid("delivery changed its original frozen entry"));
        }
        Ok(())
    }

    pub(super) fn prepare_delivery_expense(
        &self,
        py: Python<'_>,
        entries: &Bound<'_, PyTuple>,
        selected: &EvaluationOwners,
    ) -> PyResult<()> {
        let (material, instruction) = Self::singleton_lifecycle_material(entries)?;
        let value = ColdValue::from_canonical_bytes(&material)?;
        let fields = value.fields(6)?;
        if fields[0].text()? != "delivery"
            || fields[1].text()? != "real"
            || fields[3].unsigned()? != 0
            || fields[5] != ColdValue::None
        {
            return Err(invalid(
                "late expense requires the original sole delivery tail",
            ));
        }
        let ordinal = fields[2].unsigned()?;
        let limits = fields[4].fields(3)?;
        let budget = [
            limits[0].unsigned()?,
            limits[1].unsigned()?,
            limits[2].unsigned()?,
        ];
        {
            let mut retained = self.delivery_expense()?;
            if let Some(original) = retained.as_ref() {
                if original.material != material
                    || original.instruction != instruction
                    || original.owners.controller.as_ptr() != selected.controller.as_ptr()
                    || original.owners.task.as_ptr() != selected.task.as_ptr()
                    || original.owners.parent.as_ptr() != selected.parent.as_ptr()
                    || original.owners.model.as_ptr() != selected.model.as_ptr()
                {
                    return Err(invalid(
                        "late expense cannot replace its original selected owners or entry",
                    ));
                }
            } else {
                *retained = Some(DeliveryExpense {
                    owners: EvaluationOwners {
                        controller: selected.controller.clone_ref(py),
                        task: selected.task.clone_ref(py),
                        parent: selected.parent.clone_ref(py),
                        model: selected.model.clone_ref(py),
                    },
                    entries: entries.clone().unbind(),
                    material,
                    instruction,
                    ordinal,
                    budget,
                    started: false,
                    ready: false,
                    work: None,
                    custody: None,
                    regions: Vec::new(),
                    calls: (0..9)
                        .map(|_| DeliveryCall {
                            entered: false,
                            result: None,
                            error: None,
                        })
                        .collect(),
                    source_retired: false,
                    model_owners_dropped: false,
                    source_release_entered: false,
                    source_released: false,
                    candidate_verified: false,
                    work_closed: false,
                    report: None,
                    observer_finish_entered: false,
                    physical_peak: None,
                    record_entered: false,
                    recorded: false,
                    payload: None,
                    released: false,
                    budget_exceeded: false,
                });
            }
        }
        let started = self
            .delivery_expense()?
            .as_ref()
            .expect("original delivery")
            .started;
        if !started {
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .started = true;
            self.preparation_inputs
                .resource_observer
                .begin(py, ordinal)?;
            let parent = selected.parent.borrow(py);
            let session = parent.session.borrow(py);
            let work = session
                .owner()?
                .prepare_cold_model_work(
                    &*parent.lease()?,
                    self.preparation_inputs.cold_model_work_capacity,
                    ordinal,
                    self.records()?.confirmed_admission()?,
                )
                .map_err(xlog_err)?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .work = Some(work.clone());
            let custody = session
                .owner()?
                .share_cold_native_work(&work)
                .map_err(xlog_err)?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .custody = Some(custody.clone());
            self.source
                .borrow(py)
                .owner()?
                .attach_shared_cold_native_work(&*self.parent.borrow(py).lease()?, custody)
                .map_err(xlog_err)?;
            let native_regions = session
                .owner()?
                .prepare_cold_model_work_regions(&work, 9)
                .map_err(xlog_err)?;
            for (index, region) in native_regions.into_iter().enumerate() {
                let original = Py::new(
                    py,
                    PySemanticColdModelWork {
                        parent: if index % 2 == 0 {
                            self.parent.clone_ref(py)
                        } else {
                            selected.parent.clone_ref(py)
                        },
                        reader: selected.parent.clone_ref(py),
                        inner: work.clone(),
                        region: Some(region),
                        active: AtomicBool::new(false),
                    },
                )?;
                self.delivery_expense()?
                    .as_mut()
                    .expect("original delivery")
                    .regions
                    .push(original);
            }
            session
                .owner()?
                .begin_cold_model_work(&work)
                .map_err(xlog_err)?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .ready = true;
        }
        if !self
            .delivery_expense()?
            .as_ref()
            .expect("original delivery")
            .ready
        {
            return Err(invalid("unknown delivery admission retains its original report and cannot allocate another"));
        }
        Ok(())
    }

    fn delivery_model_call(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        self.require_delivery_entry(py)?;
        let (region, cached) = {
            let retained = self.delivery_expense()?;
            let current = retained.as_ref().expect("original delivery");
            if !current.ready || current.source_retired {
                return Err(invalid(
                    "delivery model callbacks require their original live source and report",
                ));
            }
            let call = &current.calls[index];
            if let Some(error) = &call.error {
                return Err(error.clone_ref(py));
            }
            if call.entered && call.result.is_none() {
                return Err(invalid(
                    "unknown late model work cannot repeat its original callback",
                ));
            }
            (
                current.regions[index].clone_ref(py),
                call.result.as_ref().map(|value| value.clone_ref(py)),
            )
        };
        let result = if let Some(result) = cached {
            result
        } else {
            let source_call = index % 2 == 0;
            let parent = region.borrow(py).parent.clone_ref(py);
            let published = parent.borrow(py);
            let session = published.session.borrow(py);
            let task = published.task_use.borrow(py);
            let callback_kind = if index == 8 {
                PhaseModelOwner::RetireSource
            } else if source_call {
                PhaseModelOwner::SerializeSource
            } else {
                PhaseModelOwner::SerializeCandidate
            };
            let callback = self.model_owner(py, callback_kind)?;
            let arguments = if index == 8 {
                let source = self.model_owner(py, PhaseModelOwner::Source)?;
                let (candidate, _) = self.candidate(py)?;
                Some((source, candidate).into_pyobject(py)?.unbind())
            } else if source_call {
                None
            } else {
                let (candidate, _) = self.candidate(py)?;
                let model = candidate.borrow(py).model.clone_ref(py);
                Some((model,).into_pyobject(py)?.unbind())
            };
            let _reads = ImportReadScope::checkpoint(&session, &task, &published, py)?;
            let original = region.borrow(py);
            let _scope = ColdCallbackScope::enter(py, &session, &original, region.clone_ref(py))?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .calls[index]
                .entered = true;
            let returned = match arguments {
                Some(arguments) => callback.bind(py).call1(arguments.bind(py)),
                None => callback.bind(py).call0(),
            };
            let mut retained = self.delivery_expense()?;
            let call = &mut retained.as_mut().expect("original delivery").calls[index];
            match returned {
                Ok(returned) => {
                    let result = returned.unbind();
                    call.result = Some(result.clone_ref(py));
                    result
                }
                Err(error) => {
                    call.error = Some(error.clone_ref(py));
                    return Err(error);
                }
            }
        };
        let reader = region.borrow(py).reader.clone_ref(py);
        let original = region
            .borrow(py)
            .region
            .clone()
            .expect("original delivery region");
        reader
            .borrow(py)
            .session
            .borrow(py)
            .owner()?
            .require_closed_cold_model_work_region(&original)
            .map_err(xlog_err)?;
        Ok(result)
    }

    pub(super) fn serialize_delivery_model(
        &self,
        py: Python<'_>,
        source: bool,
    ) -> PyResult<Py<PyAny>> {
        let offset = match &*self.status_lock()? {
            Completion::Preparing if self.acceptance()?.is_none() => 0,
            Completion::Preparing => 2,
            Completion::Prepared => 4,
            Completion::Unknown { readback_observed: true, .. } => 6,
            _ => return Err(invalid("late model verification lost its original preparation or known publication boundary")),
        };
        self.delivery_model_call(py, offset + usize::from(!source))
    }

    pub(super) fn delivery_source_retired(&self) -> PyResult<bool> {
        Ok(self
            .delivery_expense()?
            .as_ref()
            .is_some_and(|entry| entry.source_retired))
    }

    pub(super) fn delivery_source_retirement_entered(&self) -> PyResult<bool> {
        Ok(self
            .delivery_expense()?
            .as_ref()
            .is_some_and(|entry| entry.calls[8].entered))
    }

    fn retire_delivery_source(&self, py: Python<'_>) -> PyResult<()> {
        if !self.delivery_source_retired()? {
            let result = self.delivery_model_call(py, 8)?;
            if !result.bind(py).is_none() {
                return Err(invalid(
                    "source retirement must return None after its original known release",
                ));
            }
            self.source
                .borrow(py)
                .owner()?
                .require_retired_publication(&*self.parent.borrow(py).lease()?)
                .map_err(xlog_err)?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .source_retired = true;
        }
        let (released, entered, dropped) = {
            let retained = self.delivery_expense()?;
            let current = retained.as_ref().expect("original delivery");
            (
                current.source_released,
                current.source_release_entered,
                current.model_owners_dropped,
            )
        };
        if released {
            return Ok(());
        }
        if entered {
            return Err(invalid("unknown source native release retains its original delivery and cannot be repeated"));
        }
        if !dropped {
            self.drop_completed_evaluation_owners("source")?;
            self.drop_completed_evaluation_owners("real")?;
            self.drop_completed_private_checkpoints("real")?;
            self.drop_completed_private_execution_owners("real")?;
            self.drop_completed_private_restore_owners("real")?;
            let source_frame = {
                let mut retained = self.source_preparation()?;
                if retained
                    .as_ref()
                    .is_some_and(|source| !source.verified || !source.recorded)
                {
                    return Err(invalid(
                        "delivery cannot drop an unfinished original source frame",
                    ));
                }
                retained.take()
            };
            let callbacks = self
                .model_owners
                .lock()
                .map_err(|_| invalid("original model owner custody mutex is poisoned"))?
                .take();
            // Finalizers can release actual source storage. They run inside the
            // original physical interval, without any phase/native mutex held.
            drop(source_frame);
            drop(callbacks);
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .model_owners_dropped = true;
        }
        let custody = self
            .delivery_expense()?
            .as_ref()
            .expect("original delivery")
            .custody
            .clone();
        if let Some(custody) = custody {
            self.source
                .borrow(py)
                .owner()?
                .detach_shared_cold_native_work(&custody)
                .map_err(xlog_err)?;
            let original = self
                .delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .custody
                .take();
            drop(original);
            drop(custody);
        }
        self.delivery_expense()?
            .as_mut()
            .expect("original delivery")
            .source_release_entered = true;
        self.source
            .borrow(py)
            .release_retired_publication(py, &self.parent.borrow(py))?;
        self.delivery_expense()?
            .as_mut()
            .expect("original delivery")
            .source_released = true;
        Ok(())
    }

    /// Only cold authority metadata is refreshed after measurement or a signed
    /// readback. A retired source is never read or serialized again.
    pub(super) fn refresh_delivery_authority(&self, py: Python<'_>) -> PyResult<AuthoritySnapshot> {
        self.require_delivery_entry(py)?;
        let refreshed = self.refresh_snapshot.bind(py).call0()?;
        let snapshot =
            AuthoritySnapshot::parse(&ColdValue::read(&refreshed, &mut (16 * 1024 * 1024), 0)?)?;
        let (candidate, _) = self.candidate(py)?;
        let source_task = self.task_use.borrow(py);
        let candidate = candidate.borrow(py);
        let issued = candidate.task_use.borrow(py);
        if !matches!(source_task.state()?.phase, TaskUsePhase::ArenaPreparing(_))
            || !matches!(issued.state()?.phase, TaskUsePhase::ArenaPreparing(_))
        {
            return Err(invalid(
                "delivery authority lost its original private task owners",
            ));
        }
        snapshot.newer_than(&source_task.state()?.snapshot)?;
        snapshot.newer_than(&issued.state()?.snapshot)?;
        check_learning_grant(&source_task, &self.grant_reference, &snapshot)?;
        check_learning_grant(&issued, &self.grant_reference, &snapshot)?;
        source_task.state()?.snapshot = snapshot.clone();
        issued.state()?.snapshot = snapshot.clone();
        Ok(snapshot)
    }

    fn resolve_delivery_record(&self, py: Python<'_>) -> PyResult<()> {
        if self.records()?.delivery_known {
            return Ok(());
        }
        let (phase_id, ordinal, digest, observed) = {
            let records = self.records()?;
            let attempt = records
                .attempt
                .as_ref()
                .ok_or_else(|| invalid("delivery resolution lost its original signed write"))?;
            if attempt.kind != RecordKind::Delivery {
                return Err(invalid("delivery cannot resolve another lifecycle record"));
            }
            (
                records.phase_id,
                attempt.ordinal,
                attempt.digest,
                attempt.readback_observed,
            )
        };
        let resolve = self
            .store()?
            .as_ref()
            .ok_or_else(|| invalid("delivery resolution lost its original store"))?
            .resolve
            .clone_ref(py);
        let readback = resolve.bind(py).call1((
            &self.destination,
            PyBytes::new(py, &phase_id),
            ordinal,
            PyBytes::new(py, &digest),
        ))?;
        if readback.is_none() {
            return Err(invalid("delivery signed readback remains unknown; retain the original pending and interval"));
        }
        if readback.is_exact_instance_of::<PyBool>() && !readback.extract::<bool>()? {
            return Err(invalid(if observed {
                "delivery resolver contradicted its original exact signed readback"
            } else {
                "signed delivery absence cannot undo known checkpoint publication or repeat retirement"
            }));
        }
        let mut records = self.records()?;
        records.confirm(&readback)?;
        records.advance()
    }

    pub(super) fn finish_delivery_expense(&self, py: Python<'_>) -> PyResult<()> {
        self.require_delivery_entry(py)?;
        if !matches!(
            *self.status_lock()?,
            Completion::Unknown {
                readback_observed: true,
                ..
            }
        ) {
            return Err(invalid(
                "delivery requires exact durable readback of its original accepted checkpoint",
            ));
        }
        self.retire_delivery_source(py)?;
        let (work, reader, closed, verified, cached_report) = {
            let retained = self.delivery_expense()?;
            let current = retained.as_ref().expect("original delivery");
            (
                current.work.clone().expect("original delivery work"),
                current.owners.parent.clone_ref(py),
                current.work_closed,
                current.candidate_verified,
                current.report,
            )
        };
        let report = if let Some(report) = cached_report {
            report
        } else {
            // These are the last native checks and allocations of the measured
            // tail. The source was verified before its real retirement; only the
            // actual final selected private checkpoint can still be read.
            let parent = reader.borrow(py);
            if !verified {
                let (_, checkpoint) = self.candidate(py)?;
                let manifest = SemanticCheckpointManifest::decode(&checkpoint)?;
                let (_, saved_snapshot, _, _) = TaskCheckpointSeed::decode(&manifest.task)?;
                verify_phase_checkpoint(
                    py,
                    &parent.task_use.borrow(py),
                    &parent,
                    &manifest,
                    &saved_snapshot,
                )?;
                self.delivery_expense()?
                    .as_mut()
                    .expect("original delivery")
                    .candidate_verified = true;
            }
            if !closed {
                parent
                    .session
                    .borrow(py)
                    .owner()?
                    .close_cold_model_work(&work)
                    .map_err(xlog_err)?;
                self.delivery_expense()?
                    .as_mut()
                    .expect("original delivery")
                    .work_closed = true;
            }
            let streams = self.preparation_inputs.consumer_streams.python_value(py)?;
            let streams = checkpoint_consumer_streams(streams.bind(py), &mut (16 * 1024 * 1024))?;
            let report = parent
                .session
                .borrow(py)
                .owner()?
                .finish_cold_model_work(&*parent.lease()?, &work, &streams)
                .map_err(xlog_err)?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .report = Some(report);
            report
        };
        let finish = {
            let mut retained = self.delivery_expense()?;
            let current = retained.as_mut().expect("original delivery");
            let finish = !current.observer_finish_entered;
            current.observer_finish_entered = true;
            finish
        };
        if finish {
            self.preparation_inputs.resource_observer.finish(py)?;
        }
        let peak = self
            .delivery_expense()?
            .as_ref()
            .expect("original delivery")
            .physical_peak;
        let peak = if let Some(peak) = peak {
            peak
        } else {
            let peak = self
                .preparation_inputs
                .resource_observer
                .physical_peak(py)?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .physical_peak = Some(peak);
            peak
        };
        let expenditure = report
            .model_work
            .checked_add(report.native_work)
            .ok_or_else(|| invalid("complete delivery expenditure overflowed"))?;
        let (recorded, entered, instruction, ordinal, budget) = {
            let retained = self.delivery_expense()?;
            let current = retained.as_ref().expect("original delivery");
            (
                current.recorded,
                current.record_entered,
                current.instruction.clone(),
                current.ordinal,
                current.budget,
            )
        };
        if !recorded {
            if entered {
                return Err(invalid(
                    "unknown delivery history append cannot repeat its original callback",
                ));
            }
            let (_, checkpoint) = self.candidate(py)?;
            let identity = reader.borrow(py).identity(py)?;
            let arguments = PyDict::new(py);
            arguments.set_item("instruction_bytes", PyBytes::new(py, &instruction))?;
            arguments.set_item("operation_ordinal", ordinal)?;
            arguments.set_item("published_parent", identity.bind(py))?;
            arguments.set_item("full_checkpoint", PyBytes::new(py, &checkpoint))?;
            arguments.set_item("resource_usage", (expenditure, peak, report.model_calls))?;
            let callback = self.scientific_owner.bind(py).getattr("record_delivery")?;
            {
                let mut retained = self.delivery_expense()?;
                let current = retained.as_mut().expect("original delivery");
                current.record_entered = true;
                current.budget_exceeded =
                    expenditure > budget[0] || peak > budget[1] || report.model_calls > budget[2];
            }
            callback.call((), Some(&arguments))?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .recorded = true;
        }
        if !self.records()?.delivery_known {
            if self.records()?.attempt.is_some() {
                self.resolve_delivery_record(py)?;
            } else {
                let payload = self
                    .delivery_expense()?
                    .as_ref()
                    .expect("original delivery")
                    .payload
                    .as_ref()
                    .map(Arc::clone);
                let payload = if let Some(payload) = payload {
                    payload
                } else {
                    let authority = self.refresh_delivery_authority(py)?;
                    let (_, checkpoint) = self.candidate(py)?;
                    let history = self.scientific_owner.bind(py).getattr("history_bytes")?;
                    if !history.is_exact_instance_of::<PyBytes>()
                        || history.cast::<PyBytes>()?.as_bytes().is_empty()
                    {
                        return Err(invalid(
                            "delivery lacks its original full history after known cleanup",
                        ));
                    }
                    let identity = reader.borrow(py).identity(py)?;
                    let identity =
                        ColdValue::read(identity.bind(py).as_any(), &mut (16 * 1024 * 1024), 0)?
                            .canonical_bytes();
                    let material = self
                        .delivery_expense()?
                        .as_ref()
                        .expect("original delivery")
                        .material
                        .clone();
                    let usage: Vec<u8> = [expenditure, peak, report.model_calls]
                        .into_iter()
                        .flat_map(u64::to_le_bytes)
                        .collect();
                    let mut payload = b"xlog.learning-phase.delivery.v1\0".to_vec();
                    for field in [
                        checkpoint.as_ref(),
                        history.cast::<PyBytes>()?.as_bytes(),
                        material.as_slice(),
                        identity.as_slice(),
                        authority.canonical.as_slice(),
                        usage.as_slice(),
                    ] {
                        payload.extend_from_slice(&(field.len() as u64).to_le_bytes());
                        payload.extend_from_slice(field);
                    }
                    self.records()?.check_payload_length(payload.len())?;
                    let payload: Arc<[u8]> = payload.into();
                    self.delivery_expense()?
                        .as_mut()
                        .expect("original delivery")
                        .payload = Some(Arc::clone(&payload));
                    payload
                };
                self.append_phase_record(py, RecordKind::Delivery, &payload)?;
            }
        }
        let (released, exceeded) = {
            let retained = self.delivery_expense()?;
            let current = retained.as_ref().expect("original delivery");
            (current.released, current.budget_exceeded)
        };
        if !released {
            self.preparation_inputs.resource_observer.release(py)?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .released = true;
        }
        if exceeded {
            return Err(invalid("delivery exceeded its original whole-operation budget; measured history and known checkpoint remain retained without activation"));
        }
        Ok(())
    }
}
