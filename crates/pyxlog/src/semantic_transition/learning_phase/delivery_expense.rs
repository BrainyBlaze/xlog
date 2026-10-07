//! Original late verification, publication and source retirement expenditure.

use super::phase_evaluation::EvaluationOwners;
use super::*;
use xlog_cuda::{SemanticColdModelWork, SemanticColdModelWorkResult, SemanticColdNativeWork};

struct DeliveryCall {
    entered: bool,
    result: Option<Py<PyAny>>,
    error: Option<PyErr>,
}

struct RefusalTail {
    cause: &'static str,
    prefix: Arc<[u8]>,
    result: Arc<[u8]>,
    reason: ColdValue,
    handoff_entered: bool,
    handed_off: bool,
    transferred_regions: Option<Vec<xlog_cuda::SemanticColdModelWorkRegion>>,
    source_verified: [bool; 3],
    retired: bool,
    owners_dropped: bool,
    release_entered: bool,
    native_released: bool,
    history: Option<Arc<[u8]>>,
}

pub(super) struct DeliveryExpense {
    owners: Option<EvaluationOwners>,
    reader: Py<PySemanticPublishedParent>,
    refusal: Option<RefusalTail>,
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
    pub(super) fn terminal_refusal_retained(&self) -> PyResult<bool> {
        Ok(self
            .delivery_expense()?
            .as_ref()
            .is_some_and(|entry| entry.refusal.is_some()))
    }

    pub(in crate::semantic_transition) fn require_terminal_refusal_cold_callback(
        &self,
        py: Python<'_>,
        work: &PySemanticColdModelWork,
    ) -> PyResult<bool> {
        let retained = self.delivery_expense()?;
        let Some(current) = retained.as_ref().filter(|entry| entry.refusal.is_some()) else {
            return Ok(false);
        };
        let tail = current.refusal.as_ref().expect("original refusal");
        if !tail.handed_off
            || tail.retired
            || !current.calls[8].entered
            || !std::ptr::eq(&*current.regions[8].borrow(py), work)
            || current.reader.as_ptr() != work.reader.as_ptr()
        {
            return Err(invalid(
                "terminal private retirement changed its original admitted region",
            ));
        }
        Ok(true)
    }

    fn require_terminal_source(&self, py: Python<'_>, serialize: bool) -> PyResult<()> {
        let source = self.source.borrow(py);
        source.require_creator()?;
        let task = self.task_use.borrow(py);
        if !matches!(task.state()?.phase, TaskUsePhase::ArenaPreparing(_)) {
            return Err(invalid(
                "terminal refusal lost its original held source task",
            ));
        }
        let refreshed = self.refresh_snapshot.bind(py).call0()?;
        let snapshot =
            AuthoritySnapshot::parse(&ColdValue::read(&refreshed, &mut (16 * 1024 * 1024), 0)?)?;
        snapshot.newer_than(&task.state()?.snapshot)?;
        check_learning_grant(&task, &self.grant_reference, &snapshot)?;
        let manifest = SemanticCheckpointManifest::decode(&self.source_checkpoint)?;
        let parent = self.parent.borrow(py);
        verify_phase_native(py, &task, &parent, &manifest.native, true)?;
        if serialize {
            let model = self.delivery_model_call(py, 6)?;
            require_model_bytes(model.bind(py), &manifest.model)?;
            verify_phase_native(py, &task, &parent, &manifest.native, true)?;
        }
        task.state()?.snapshot = snapshot;
        Ok(())
    }

    pub(super) fn finish_terminal_refusal(
        &self,
        py: Python<'_>,
        cause: &'static str,
    ) -> PyResult<()> {
        if matches!(*self.status_lock()?, Completion::Refused) {
            return Ok(());
        }
        if !self.terminal_refusal_retained()? {
            let (prefix, result, reason) = {
                let accepted = self.acceptance()?;
                let accepted = accepted.as_ref().ok_or_else(|| {
                    invalid("terminal refusal lost its original complete comparison")
                })?;
                let result =
                    if cause == "scientific-comparison-refused" {
                        Arc::clone(accepted.refusal.as_ref().ok_or_else(|| {
                            invalid("unknown comparison is not a scientific refusal")
                        })?)
                    } else {
                        Arc::from(
                            accepted
                                .result
                                .as_ref()
                                .ok_or_else(|| {
                                    invalid("private abandonment lost its positive original result")
                                })?
                                .bind(py)
                                .cast::<PyBytes>()?
                                .as_bytes(),
                        )
                    };
                let reason = if cause == "scientific-comparison-refused" {
                    cold_restore::scientific_refusal_reason(
                        &self.preparation_inputs.frozen_program,
                        &accepted.history,
                        &result,
                    )?
                } else {
                    ColdValue::Text(cause.to_owned())
                };
                (Arc::clone(&accepted.history), result, reason)
            };
            let allowed = match &*self.status_lock()? {
                Completion::Preparing => cause == "scientific-comparison-refused",
                Completion::Prepared => cause == "private-candidate-abandoned",
                Completion::Unknown {
                    readback_observed: false,
                    ..
                } => cause == "checkpoint-publication-absent",
                _ => false,
            };
            let mut retained = self.delivery_expense()?;
            let current = retained
                .as_mut()
                .ok_or_else(|| invalid("terminal refusal lost its original measured tail"))?;
            if !allowed
                || current.source_retired
                || current.calls[8].entered
                || self.records()?.attempt.is_some()
                || self.records()?.delivery_known
            {
                return Err(invalid("terminal refusal cannot undo unknown or known accepted publication or source release"));
            }
            current.refusal = Some(RefusalTail {
                cause,
                prefix,
                result,
                reason,
                handoff_entered: false,
                handed_off: false,
                transferred_regions: None,
                source_verified: [false; 3],
                retired: false,
                owners_dropped: false,
                release_entered: false,
                native_released: false,
                history: None,
            });
        }
        self.require_delivery_entry(py)?;
        let verified = self
            .delivery_expense()?
            .as_ref()
            .expect("original delivery")
            .refusal
            .as_ref()
            .expect("original refusal")
            .source_verified[0];
        if !verified {
            self.require_terminal_source(py, false)?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .refusal
                .as_mut()
                .expect("original refusal")
                .source_verified[0] = true;
        }
        self.handoff_terminal_report(py)?;
        self.retire_terminal_candidate(py)?;
        let verified = self
            .delivery_expense()?
            .as_ref()
            .expect("original delivery")
            .refusal
            .as_ref()
            .expect("original refusal")
            .source_verified[2];
        if !verified {
            self.require_terminal_source(py, true)?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .refusal
                .as_mut()
                .expect("original refusal")
                .source_verified[2] = true;
        }
        let (work, closed) = {
            let retained = self.delivery_expense()?;
            let current = retained.as_ref().expect("original delivery");
            (
                current.work.clone().expect("original work"),
                current.work_closed,
            )
        };
        if !closed {
            // Known complete private release proves the remaining callbacks were
            // never entered. Entered/failed/unknown regions cannot be cancelled.
            self.source
                .borrow(py)
                .owner()?
                .cancel_unentered_cold_model_work_regions(&work, 8)
                .map_err(xlog_err)?;
        }
        self.finish_delivery_expense(py)?;
        let source = self.source.borrow(py);
        let task = self.task_use.borrow(py);
        // Only CPU authority freshness follows the measured and signed tail.
        let refreshed = self.refresh_snapshot.bind(py).call0()?;
        let snapshot =
            AuthoritySnapshot::parse(&ColdValue::read(&refreshed, &mut (16 * 1024 * 1024), 0)?)?;
        snapshot.newer_than(&task.state()?.snapshot)?;
        check_learning_grant(&task, &self.grant_reference, &snapshot)?;
        let resumed = match &task.state()?.phase {
            TaskUsePhase::ArenaPreparing(original) => *original.clone(),
            _ => {
                return Err(invalid(
                    "terminal refusal lost its original source continuation",
                ))
            }
        };
        let mut retention = source
            .learning_transition
            .lock()
            .map_err(|_| invalid("source refusal retention mutex is poisoned"))?;
        let mut state = task.state()?;
        if !matches!(state.phase, TaskUsePhase::ArenaPreparing(_)) || !self.records()?.refusal_known
        {
            return Err(invalid(
                "source continuation precedes exact terminal durable readback",
            ));
        }
        state.snapshot = snapshot;
        state.phase = resumed;
        *self.status_lock()? = Completion::Refused;
        source.learning_preparing.store(false, Ordering::Release);
        let original = retention.take();
        drop(state);
        drop(retention);
        drop(original);
        Ok(())
    }

    fn handoff_terminal_report(&self, py: Python<'_>) -> PyResult<()> {
        let (work, entered, handed_off) = {
            let retained = self.delivery_expense()?;
            let current = retained.as_ref().expect("original delivery");
            let tail = current.refusal.as_ref().expect("original refusal");
            (
                current.work.clone().expect("original report"),
                tail.handoff_entered,
                tail.handed_off,
            )
        };
        if !handed_off {
            if entered {
                return Err(invalid(
                    "unknown report handoff cannot move or allocate its original report again",
                ));
            }
            let (candidate, _) = self.candidate(py)?;
            let parent = candidate.borrow(py).parent.clone_ref(py);
            let streams = self.preparation_inputs.consumer_streams.python_value(py)?;
            let streams = checkpoint_consumer_streams(streams.bind(py), &mut (16 * 1024 * 1024))?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .refusal
                .as_mut()
                .expect("original refusal")
                .handoff_entered = true;
            let (transferred, regions) = {
                let parent = parent.borrow(py);
                let private = parent.session.borrow(py);
                let source = self.source.borrow(py);
                let original = self.parent.borrow(py);
                let mut private_owner = private.owner()?;
                let mut source_owner = source.owner()?;
                let private_lease = parent.lease()?;
                let source_lease = original.lease()?;
                let transferred = private_owner
                    .handoff_cold_model_work(
                        &private_lease,
                        &work,
                        &mut source_owner,
                        &source_lease,
                        &streams,
                    )
                    .map_err(xlog_err)?;
                transferred
            };
            let mut retained = self.delivery_expense()?;
            let current = retained.as_mut().expect("original delivery");
            current.work = Some(transferred);
            current.reader = self.parent.clone_ref(py);
            let tail = current.refusal.as_mut().expect("original refusal");
            tail.transferred_regions = Some(regions);
            tail.handed_off = true;
        }
        let rebuild = self
            .delivery_expense()?
            .as_ref()
            .expect("original delivery")
            .refusal
            .as_ref()
            .expect("original refusal")
            .transferred_regions
            .is_some();
        if rebuild {
            let (parent, _) = self.candidate(py)?;
            let parent = parent.borrow(py).parent.clone_ref(py);
            let (work, regions) = {
                let retained = self.delivery_expense()?;
                let current = retained.as_ref().expect("original delivery");
                (
                    current.work.clone().expect("transferred report"),
                    current
                        .refusal
                        .as_ref()
                        .expect("original refusal")
                        .transferred_regions
                        .clone()
                        .expect("original region states"),
                )
            };
            let mut original = Vec::with_capacity(regions.len());
            for (index, region) in regions.into_iter().enumerate() {
                original.push(Py::new(
                    py,
                    PySemanticColdModelWork {
                        parent: if index == 8 || index % 2 == 1 {
                            parent.clone_ref(py)
                        } else {
                            self.parent.clone_ref(py)
                        },
                        reader: self.parent.clone_ref(py),
                        inner: work.clone(),
                        region: Some(region),
                        active: AtomicBool::new(false),
                    },
                )?);
            }
            let replaced = {
                let mut retained = self.delivery_expense()?;
                let current = retained.as_mut().expect("original delivery");
                current
                    .refusal
                    .as_mut()
                    .expect("original refusal")
                    .transferred_regions = None;
                std::mem::replace(&mut current.regions, original)
            };
            drop(replaced);
        }
        Ok(())
    }

    fn retire_terminal_candidate(&self, py: Python<'_>) -> PyResult<()> {
        let retired = self
            .delivery_expense()?
            .as_ref()
            .expect("original delivery")
            .refusal
            .as_ref()
            .expect("original refusal")
            .retired;
        if !retired {
            let result = self.delivery_model_call(py, 8)?;
            if !result.bind(py).is_none() {
                return Err(invalid(
                    "private retirement must return None after known consumer release",
                ));
            }
            let (candidate, _) = self.candidate(py)?;
            let candidate = candidate.borrow(py);
            candidate
                .session
                .borrow(py)
                .owner()?
                .require_retired_publication(&*candidate.parent.borrow(py).lease()?)
                .map_err(xlog_err)?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .refusal
                .as_mut()
                .expect("original refusal")
                .retired = true;
        }
        let dropped = self
            .delivery_expense()?
            .as_ref()
            .expect("original delivery")
            .refusal
            .as_ref()
            .expect("original refusal")
            .owners_dropped;
        if !dropped {
            self.drop_completed_evaluation_owners("real")?;
            self.drop_completed_private_checkpoints("real")?;
            self.drop_completed_private_execution_owners("real")?;
            self.drop_completed_private_restore_owners("real")?;
            let construction = self.private_restore()?.take();
            let owners = self
                .delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .owners
                .take();
            drop(construction);
            drop(owners);
            self.require_terminal_source(py, false)?;
            let mut retained = self.delivery_expense()?;
            let tail = retained
                .as_mut()
                .expect("original delivery")
                .refusal
                .as_mut()
                .expect("original refusal");
            tail.source_verified[1] = true;
            tail.owners_dropped = true;
        }
        let (entered, released) = {
            let retained = self.delivery_expense()?;
            let tail = retained
                .as_ref()
                .expect("original delivery")
                .refusal
                .as_ref()
                .expect("original refusal");
            (tail.release_entered, tail.native_released)
        };
        if !released {
            if entered {
                return Err(invalid(
                    "unknown private native deallocation cannot repeat or release its report",
                ));
            }
            let (candidate, _) = self.candidate(py)?;
            let candidate = candidate.borrow(py);
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .refusal
                .as_mut()
                .expect("original refusal")
                .release_entered = true;
            let private = candidate.session.borrow(py);
            private.release_retired_publication(py, &candidate.parent.borrow(py))?;
            private.learning_preparing.store(false, Ordering::Release);
            let retained = private
                .learning_transition
                .lock()
                .map_err(|_| invalid("private refusal retention mutex is poisoned"))?
                .take();
            drop(retained);
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .refusal
                .as_mut()
                .expect("original refusal")
                .native_released = true;
        }
        let retired = self
            .candidate
            .lock()
            .map_err(|_| invalid("refusal candidate custody mutex is poisoned"))?
            .as_mut()
            .expect("original candidate")
            .owner
            .take();
        let custody = self
            .delivery_expense()?
            .as_mut()
            .expect("original delivery")
            .custody
            .take();
        drop(retired);
        drop(custody);
        Ok(())
    }

    fn record_terminal_tail(&self, py: Python<'_>, usage: [u64; 3]) -> PyResult<()> {
        let (entered, recorded, instruction, ordinal, reason, prefix, result, cause) = {
            let retained = self.delivery_expense()?;
            let current = retained.as_ref().expect("original delivery");
            let tail = current.refusal.as_ref().expect("original refusal");
            (
                current.record_entered,
                current.recorded,
                current.instruction.clone(),
                current.ordinal,
                tail.reason.clone(),
                Arc::clone(&tail.prefix),
                Arc::clone(&tail.result),
                tail.cause,
            )
        };
        if !recorded {
            if entered {
                return Err(invalid(
                    "unknown terminal history append cannot repeat its original callback",
                ));
            }
            let arguments = PyDict::new(py);
            arguments.set_item("instruction_bytes", PyBytes::new(py, &instruction))?;
            arguments.set_item("operation_ordinal", ordinal)?;
            arguments.set_item("branch", "real")?;
            arguments.set_item("resource_usage", (usage[0], usage[1], usage[2]))?;
            arguments.set_item("refusal", reason.python_value(py)?)?;
            let callback = self.scientific_owner.bind(py).getattr("record_refusal")?;
            {
                let mut retained = self.delivery_expense()?;
                let current = retained.as_mut().expect("original delivery");
                current.record_entered = true;
                current.budget_exceeded = usage
                    .into_iter()
                    .zip(current.budget)
                    .any(|(actual, limit)| actual > limit);
            }
            callback.call((), Some(&arguments))?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .recorded = true;
        }
        let history = self
            .delivery_expense()?
            .as_ref()
            .expect("original delivery")
            .refusal
            .as_ref()
            .expect("original refusal")
            .history
            .as_ref()
            .map(Arc::clone);
        let history = if let Some(history) = history {
            history
        } else {
            let history = self.scientific_owner.bind(py).getattr("history_bytes")?;
            if !history.is_exact_instance_of::<PyBytes>() {
                return Err(invalid(
                    "terminal refusal requires its actual full history bytes",
                ));
            }
            let history: Arc<[u8]> = Arc::from(history.cast::<PyBytes>()?.as_bytes());
            cold_restore::require_terminal_history(
                &prefix,
                &history,
                &instruction,
                ordinal,
                &reason,
                usage,
            )?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .refusal
                .as_mut()
                .expect("original refusal")
                .history = Some(Arc::clone(&history));
            history
        };
        if !self.records()?.refusal_known {
            if self.records()?.attempt.is_some() {
                self.resolve_terminal_record(py)?;
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
                    let source = self.source.borrow(py);
                    let task = self.task_use.borrow(py);
                    let snapshot = task.state()?.snapshot.canonical.clone();
                    source.require_creator()?;
                    let checkpoint = self.candidate_checkpoint()?;
                    let material = self
                        .delivery_expense()?
                        .as_ref()
                        .expect("original delivery")
                        .material
                        .clone();
                    let reason = reason.canonical_bytes();
                    let usage: Vec<u8> = usage.into_iter().flat_map(u64::to_le_bytes).collect();
                    let mut payload = b"xlog.learning-phase.refusal.v1\0".to_vec();
                    for field in [
                        cause.as_bytes(),
                        self.source_checkpoint.as_slice(),
                        checkpoint.as_ref(),
                        prefix.as_ref(),
                        result.as_ref(),
                        history.as_ref(),
                        material.as_slice(),
                        reason.as_slice(),
                        snapshot.as_slice(),
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
                self.append_phase_record(py, RecordKind::TerminalRefusal, &payload)?;
            }
        }
        if !self
            .delivery_expense()?
            .as_ref()
            .expect("original delivery")
            .released
        {
            self.preparation_inputs.resource_observer.release(py)?;
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .released = true;
        }
        Ok(())
    }

    fn resolve_terminal_record(&self, py: Python<'_>) -> PyResult<()> {
        let (phase_id, ordinal, digest) = {
            let records = self.records()?;
            let attempt = records
                .attempt
                .as_ref()
                .ok_or_else(|| invalid("terminal resolution lost its original signed attempt"))?;
            if attempt.kind != RecordKind::TerminalRefusal {
                return Err(invalid(
                    "terminal refusal cannot resolve another lifecycle record",
                ));
            }
            (records.phase_id, attempt.ordinal, attempt.digest)
        };
        let resolver = self
            .store()?
            .as_ref()
            .ok_or_else(|| invalid("terminal refusal lost its original durable store"))?
            .resolve
            .clone_ref(py);
        let readback = resolver.bind(py).call1((
            &self.destination,
            PyBytes::new(py, &phase_id),
            ordinal,
            PyBytes::new(py, &digest),
        ))?;
        if readback.is_none() || readback.is_exact_instance_of::<PyBool>() {
            return Err(invalid("terminal record readback remains unknown; retain the original refusal without another write"));
        }
        let mut records = self.records()?;
        records.confirm(&readback)?;
        records.advance()
    }

    pub(super) fn terminal_refusal_outcome(&self, py: Python<'_>) -> PyResult<Option<Py<PyTuple>>> {
        let record = self.records()?.refusal_record.as_ref().map(Arc::clone);
        let Some(record) = record else {
            return Ok(None);
        };
        let retained = self.delivery_expense()?;
        let tail = retained
            .as_ref()
            .expect("original delivery")
            .refusal
            .as_ref()
            .expect("original refusal");
        Ok(Some(
            (
                tail.cause,
                PyBytes::new(py, &tail.result),
                PyBytes::new(py, tail.history.as_ref().expect("known terminal history")),
                PyBytes::new(py, &record),
            )
                .into_pyobject(py)?
                .unbind(),
        ))
    }

    /// Known signed Delivery alone authorizes a checkpoint signer. This cold
    /// CPU custody does not repeat model work, change Q or enter another region.
    pub(super) fn install_delivered_checkpoint_custody(&self, py: Python<'_>) -> PyResult<()> {
        let (phase_id, delivery_digest, admission) = self.records()?.delivered_identity()?;
        let payload = self
            .delivery_expense()?
            .as_ref()
            .and_then(|delivery| delivery.payload.as_ref().map(Arc::clone))
            .ok_or_else(|| invalid("known signed Delivery lost its original payload"))?;
        let mut reader = phase_record::RecordReader(
            payload
                .strip_prefix(b"xlog.learning-phase.delivery.v1\0")
                .ok_or_else(|| invalid("known signed Delivery lost its original domain"))?,
        );
        let checkpoint = reader.field()?;
        let history: Arc<[u8]> = Arc::from(reader.field()?);
        reader.field()?;
        reader.field()?;
        let snapshot =
            AuthoritySnapshot::parse(&ColdValue::from_canonical_bytes(reader.field()?)?)?;
        reader.field()?;
        reader.finish()?;
        let acceptance = self.acceptance()?;
        let accepted = acceptance
            .as_ref()
            .and_then(|accepted| accepted.result.as_ref())
            .ok_or_else(|| invalid("known signed Delivery lost its original accepted result"))?;
        let acceptance = Arc::from(accepted.bind(py).cast::<PyBytes>()?.as_bytes());
        let (candidate, original) = self.candidate(py)?;
        if original.as_ref() != checkpoint {
            return Err(invalid(
                "known signed Delivery changed its original candidate checkpoint",
            ));
        }
        let closure = Arc::new(cold_restore::VerifiedClosure {
            phase_id,
            delivery_digest,
            admission,
            program: Arc::from(self.preparation_inputs.frozen_program.as_slice()),
            history,
            acceptance,
            snapshot,
        });
        let result = candidate
            .borrow(py)
            .session
            .borrow(py)
            .install_checkpoint_custody(closure, Sha256::digest(checkpoint).into());
        result
    }

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
                let owners = original
                    .owners
                    .as_ref()
                    .ok_or_else(|| invalid("late expense selected owners are already retired"))?;
                if original.material != material
                    || original.instruction != instruction
                    || owners.controller.as_ptr() != selected.controller.as_ptr()
                    || owners.task.as_ptr() != selected.task.as_ptr()
                    || owners.parent.as_ptr() != selected.parent.as_ptr()
                    || owners.model.as_ptr() != selected.model.as_ptr()
                {
                    return Err(invalid(
                        "late expense cannot replace its original selected owners or entry",
                    ));
                }
            } else {
                *retained = Some(DeliveryExpense {
                    owners: Some(EvaluationOwners {
                        controller: selected.controller.clone_ref(py),
                        task: selected.task.clone_ref(py),
                        parent: selected.parent.clone_ref(py),
                        model: selected.model.clone_ref(py),
                    }),
                    reader: selected.parent.clone_ref(py),
                    refusal: None,
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
            let refusal_retirement = index == 8
                && self
                    .delivery_expense()?
                    .as_ref()
                    .expect("original delivery")
                    .refusal
                    .is_some();
            let callback_kind = if refusal_retirement {
                PhaseModelOwner::RetirePrivate
            } else if index == 8 {
                PhaseModelOwner::RetireSource
            } else if source_call {
                PhaseModelOwner::SerializeSource
            } else {
                PhaseModelOwner::SerializeCandidate
            };
            let callback = self.model_owner(py, callback_kind)?;
            let arguments = if refusal_retirement {
                let (candidate, _) = self.candidate(py)?;
                let model = candidate.borrow(py).model.clone_ref(py);
                Some((model,).into_pyobject(py)?.unbind())
            } else if index == 8 {
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
            let _reads = if refusal_retirement {
                None
            } else {
                Some(ImportReadScope::checkpoint(
                    &session, &task, &published, py,
                )?)
            };
            self.delivery_expense()?
                .as_mut()
                .expect("original delivery")
                .calls[index]
                .entered = true;
            let original = region.borrow(py);
            let _scope = ColdCallbackScope::enter(py, &session, &original, region.clone_ref(py))?;
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
            .is_some_and(|entry| entry.refusal.is_none() && entry.calls[8].entered))
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
        let refusing = self.terminal_refusal_retained()?;
        if !refusing
            && !matches!(
                *self.status_lock()?,
                Completion::Unknown {
                    readback_observed: true,
                    ..
                }
            )
        {
            return Err(invalid(
                "delivery requires exact durable readback of its original accepted checkpoint",
            ));
        }
        if !refusing {
            self.retire_delivery_source(py)?;
        } else if !self
            .delivery_expense()?
            .as_ref()
            .expect("original delivery")
            .refusal
            .as_ref()
            .is_some_and(|tail| {
                tail.native_released && tail.source_verified.iter().all(|verified| *verified)
            })
        {
            return Err(invalid("terminal accounting precedes known private release and unchanged source verification"));
        }
        let (work, reader, closed, verified, cached_report) = {
            let retained = self.delivery_expense()?;
            let current = retained.as_ref().expect("original delivery");
            (
                current.work.clone().expect("original delivery work"),
                current.reader.clone_ref(py),
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
            if !verified && !refusing {
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
        if refusing {
            return self.record_terminal_tail(py, [expenditure, peak, report.model_calls]);
        }
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
                    // The acquired native identity contains digest bytes. General
                    // metadata transport deliberately does not accept bytes;
                    // retain these exact native-issued fields, not a Python
                    // conversion or a relaxation of that public transport.
                    let identity = identity.bind(py);
                    let digest = |index| -> PyResult<ColdValue> {
                        let field = identity.get_item(index)?;
                        Ok(ColdValue::Bytes(Arc::from(
                            field.cast::<PyBytes>()?.as_bytes(),
                        )))
                    };
                    let identity = ColdValue::Sequence(vec![
                        digest(0)?,
                        ColdValue::read(&identity.get_item(1)?, &mut 128, 0)?,
                        digest(2)?,
                        digest(3)?,
                    ])
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
