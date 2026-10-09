//! Original CURRENT actor children embedded in their admitted Update graph.

pub(super) enum PreparedCaptureSequence<'a> {
    Outer(&'a mut xlog_cuda::SemanticPreparedSegmentCapture),
    Body(&'a mut xlog_cuda::cuda_graph::ConditionalCudaGraphBodySequence),
}

impl PreparedCaptureSequence<'_> {
    fn add_conditional_if<P, F, E>(
        &mut self,
        stream: &cudarc::driver::CudaStream,
        preflight: P,
        body: F,
    ) -> Result<u64, xlog_cuda::cuda_graph::CudaConditionalGraphUnavailable>
    where
        P: FnOnce(u64) -> Result<(), E>,
        E: std::fmt::Display,
        F: FnOnce(
            &xlog_cuda::cuda_graph::ConditionalCudaGraphBody,
        ) -> Result<(), xlog_cuda::cuda_graph::CudaConditionalGraphUnavailable>,
    {
        match self {
            Self::Outer(capture) => capture.add_conditional_if(stream, preflight, body),
            Self::Body(capture) => capture.add_conditional_if(stream, preflight, body),
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the canonical recorder keeps original task, scope, model and graph owners distinct"
)]
pub(super) fn record_prepared_steps(
    sequence: &mut PreparedCaptureSequence<'_>,
    py: Python<'_>,
    session: &PySemanticTransitionSession,
    issued: &PySemanticTransitionTaskUse,
    steps: &[Py<PySemanticPreparedStep>],
    prepared: &Bound<'_, PyAny>,
    stream: &Arc<cudarc::driver::CudaStream>,
    check: &impl Fn() -> PyResult<()>,
    private: Option<&Py<learning_phase::PySemanticLearningPhaseTransition>>,
    actor_refreshes: &[Arc<ActorRefreshCustody>],
) -> PyResult<()> {
    for step in steps {
        check()?;
        let native = step.borrow(py).inner.clone();
        let admission_error = std::cell::RefCell::new(None);
        let admission = sequence.add_conditional_if(
            &stream,
            |handle| -> PyResult<()> {
                let result = (|| -> PyResult<()> {
                    session
                        .owner()?
                        .record_prepared_step_admission(&native, handle)
                        .map_err(xlog_err)?;
                    #[cfg(feature = "semantic-policy")]
                    if let Some(private) = &private {
                        private.borrow(py).bind_private_step_capture(
                            py,
                            &session,
                            &issued,
                            &step.borrow(py),
                            stream.cu_stream() as u64,
                            false,
                        )?;
                    }
                    Ok(())
                })();
                if let Err(value) = &result {
                    *admission_error.borrow_mut() = Some(value.clone_ref(py));
                }
                result
            },
            |body| {
                body.capture_on_stream(&stream, || -> PyResult<()> {
                    let result = session
                        .owner()?
                        .record_prepared_step_inputs(&native)
                        .map_err(xlog_err);
                    if let Err(value) = &result {
                        *admission_error.borrow_mut() = Some(value.clone_ref(py));
                    }
                    result
                })
            },
        );
        if let Err(graph_error) = admission {
            return Err(admission_error
                .into_inner()
                .unwrap_or_else(|| xlog_err(graph_error)));
        }

        for bank in 0..2 {
            let requested_error = std::cell::RefCell::new(None);
            let requested = sequence.add_conditional_if(
                &stream,
                |handle| -> PyResult<()> {
                    let result = session
                        .owner()?
                        .record_prepared_step_requested_bank_gate(&native, bank, handle)
                        .map_err(xlog_err);
                    if let Err(value) = &result {
                        *requested_error.borrow_mut() = Some(value.clone_ref(py));
                    }
                    result
                },
                |body| {
                    let children = actor_refreshes.iter().filter(|child| child.matches_update(py, step)).cloned().collect::<Vec<_>>();
                    if !children.is_empty() {
                        return body.with_sequence(stream, |nested| {
                            nested.capture_segment_on_stream(stream, || -> PyResult<()> {
                                let result = (|| {
                                    session.owner()?.begin_prepared_step_bank_capture(&native, bank).map_err(xlog_err)?;
                                    session.owner()?.record_prepared_bank_model_content(&native, bank).map_err(xlog_err)
                                })();
                                if let Err(error) = &result { *requested_error.borrow_mut() = Some(error.clone_ref(py)); }
                                result
                            })?;
                            for child in &children {
                                child.record_bank(py, bank, nested).map_err(|error| {
                                    *requested_error.borrow_mut() = Some(error.clone_ref(py));
                                    xlog_cuda::cuda_graph::CudaConditionalGraphUnavailable::BodyPopulationFailed { detail: error.to_string() }
                                })?;
                            }
                            nested.capture_segment_on_stream(stream, || -> PyResult<()> {
                                let result = (|| {
                                recording_callback(check, || {
                                    let value = prepared.getattr("enqueue_step")?.call1((step.clone_ref(py), bank))?;
                                    if !value.is_none() { return Err(invalid("enqueue_step must return None, not a host result")); }
                                    Ok(())
                                })?;
                                session.owner()?.enqueue_prepared_transition(&native, bank).map_err(xlog_err)
                                })();
                                if let Err(error) = &result { *requested_error.borrow_mut() = Some(error.clone_ref(py)); }
                                result
                            })
                        });
                    }
                    body.capture_on_stream(&stream, || -> PyResult<()> {
                        let result = (|| {
                            session
                                .owner()?
                                .begin_prepared_step_bank_capture(&native, bank)
                                .map_err(xlog_err)?;
                            session
                                .owner()?
                                .record_prepared_bank_model_content(&native, bank)
                                .map_err(xlog_err)?;
                            let enqueue = recording_callback(check, || {
                                prepared.getattr("enqueue_step")
                            })?;
                            recording_callback(check, || {
                                let value = enqueue.call1((step.clone_ref(py), bank))?;
                                if !value.is_none() {
                                    return Err(invalid(
                                        "enqueue_step must return None, not a host result",
                                    ));
                                }
                                Ok(())
                            })?;
                            session
                                .owner()?
                                .enqueue_prepared_transition(&native, bank)
                                .map_err(xlog_err)
                        })();
                        if let Err(value) = &result {
                            *requested_error.borrow_mut() = Some(value.clone_ref(py));
                        }
                        result
                    })
                },
            );
            if let Err(graph_error) = requested {
                return Err(requested_error
                    .into_inner()
                    .unwrap_or_else(|| xlog_err(graph_error)));
            }
        }

        let drain_error = std::cell::RefCell::new(None);
        let drain = sequence.add_conditional_if(
            &stream,
            |handle| -> PyResult<()> {
                let result = session
                    .owner()?
                    .record_prepared_step_drain_gate(&native, handle)
                    .map_err(xlog_err);
                if let Err(value) = &result {
                    *drain_error.borrow_mut() = Some(value.clone_ref(py));
                }
                result
            },
            |body| {
                body.capture_on_stream(&stream, || -> PyResult<()> {
                    let result = session
                        .owner()?
                        .enqueue_prepared_drain(&native)
                        .map_err(xlog_err);
                    if let Err(value) = &result {
                        *drain_error.borrow_mut() = Some(value.clone_ref(py));
                    }
                    result
                })
            },
        );
        if let Err(graph_error) = drain {
            return Err(drain_error
                .into_inner()
                .unwrap_or_else(|| xlog_err(graph_error)));
        }

        let release_error = std::cell::RefCell::new(None);
        let release = sequence.add_conditional_if(
            &stream,
            |handle| -> PyResult<()> {
                let result = session
                    .owner()?
                    .record_prepared_step_active_gate(&native, handle)
                    .map_err(xlog_err);
                if let Err(value) = &result {
                    *release_error.borrow_mut() = Some(value.clone_ref(py));
                }
                result
            },
            |body| {
                body.capture_on_stream(&stream, || -> PyResult<()> {
                    let result = (|| -> PyResult<()> {
                        session
                            .owner()?
                            .record_prepared_step_release(&native)
                            .map_err(xlog_err)?;
                        #[cfg(feature = "semantic-policy")]
                        if let Some(private) = &private {
                            private.borrow(py).bind_private_step_capture(
                                py,
                                &session,
                                &issued,
                                &step.borrow(py),
                                stream.cu_stream() as u64,
                                true,
                            )?;
                        }
                        Ok(())
                    })();
                    if let Err(value) = &result {
                        *release_error.borrow_mut() = Some(value.clone_ref(py));
                    }
                    result
                })
            },
        );
        if let Err(graph_error) = release {
            return Err(release_error
                .into_inner()
                .unwrap_or_else(|| xlog_err(graph_error)));
        }
        check()?;
    }

    Ok(())
}

use super::*;
use std::sync::Weak;

pub(super) fn finish_actor_refresh_children(
    py: Python<'_>,
    children: &[Arc<ActorRefreshCustody>],
    outer: Option<&xlog_cuda::SemanticPreparedSegmentNonSubmission>,
    streams: &[u64],
) -> PyResult<()> {
    for child in children {
        child.finish_after_parent_completion(py, outer, streams)?;
    }
    Ok(())
}

pub(super) fn retire_actor_refresh_children(
    py: Python<'_>,
    children: &[Arc<ActorRefreshCustody>],
    streams: &[u64],
) -> PyResult<()> {
    for child in children {
        child.retire_after_parent_callback(py, streams)?;
    }
    Ok(())
}

pub(super) struct ActorRefreshSession {
    pub(super) owner: Weak<ActorRefreshCustody>,
    pub(super) bank: usize,
}

pub(super) struct ActorRefreshCustody {
    update: Py<PySemanticPreparedStep>,
    instruction: Arc<SegmentInstructionCustody>,
    member_ordinal: u64,
    row: SemanticTrainingViewRow,
    original_row: ReplayRow,
    replay: NativeReplayBinding,
    source: Arc<VerifiedCheckpointSource>,
    proof: xlog_cuda::SemanticPreparedActorRefresh,
    children: Mutex<[ActorRefreshChild; 2]>,
}

#[derive(Default)]
struct ActorRefreshChild {
    construction_entered: bool,
    allocation: Option<CheckpointAllocationDomain>,
    construction: Option<Arc<Mutex<ActorRefreshConstruction>>>,
    constructor_retired: bool,
    observer: Option<Arc<dyn xlog_cuda::SemanticTaskProgram>>,
    observer_terminal_error: Option<PyErr>,
    cold_content: Option<(xlog_cuda::SemanticTaskContentIdentity, Vec<xlog_cuda::SemanticTruth>)>,
    session: Option<Py<PySemanticTransitionSession>>,
    controller: Option<Py<PySemanticTransitionController>>,
    task: Option<Py<PySemanticTransitionTaskUse>>,
    scope: Option<Arc<()>>,
    steps: Vec<Py<PySemanticPreparedStep>>,
    producer: Option<Py<PyAny>>,
    prepared: Option<Py<PyAny>>,
    original_prepared: Option<Py<PyAny>>,
    resources: Option<Arc<PreparedProducerResources>>,
    cold_work: Option<SemanticColdNativeWork>,
    retirement_work: Option<SemanticColdNativeWork>,
    retirement_attached: bool,
    owner_callback_entered: bool,
    native_steps: Vec<SemanticPreparedStep>,
    checkpoint_phase: Option<CheckpointTaskPhase>,
    preparation_entered: bool,
    context_ready: bool,
    successors_entered: bool,
    successors_ready: bool,
    recording_entered: bool,
    recorded: bool,
    frozen: bool,
    delivery: Option<Py<PySemanticGradientDelivery>>,
    vjp_entered: bool,
    vjp_recorded: bool,
    completion: Option<ActorRefreshCompletion>,
    producer_retirement_entered: bool,
    producer_retired: bool,
    graph_retired: bool,
    resources_retired: bool,
    native_released: bool,
    release_memory: Option<Arc<xlog_cuda::memory::GpuMemoryManager>>,
    retirement_streams: Option<Vec<u64>>,
    retired: bool,
}

impl ActorRefreshChild {
    fn pristine(&self) -> bool {
        !self.construction_entered
            && self.allocation.is_none()
            && self.construction.is_none()
            && !self.constructor_retired
            && self.observer.is_none()
            && self.observer_terminal_error.is_none()
            && self.cold_content.is_none()
            && self.session.is_none()
            && self.controller.is_none()
            && self.task.is_none()
            && self.scope.is_none()
            && self.steps.is_empty()
            && self.producer.is_none()
            && self.prepared.is_none()
            && self.original_prepared.is_none()
            && self.resources.is_none()
            && self.cold_work.is_none()
            && self.retirement_work.is_none()
            && !self.retirement_attached
            && !self.owner_callback_entered
            && self.native_steps.is_empty()
            && self.checkpoint_phase.is_none()
            && !self.preparation_entered
            && !self.context_ready
            && !self.successors_entered
            && !self.successors_ready
            && !self.recording_entered
            && !self.recorded
            && !self.frozen
            && self.delivery.is_none()
            && !self.vjp_entered
            && !self.vjp_recorded
            && self.completion.is_none()
            && !self.producer_retirement_entered
            && !self.producer_retired
            && !self.graph_retired
            && !self.resources_retired
            && !self.native_released
            && self.release_memory.is_none()
            && self.retirement_streams.is_none()
            && !self.retired
    }

    fn constructor_only(&self) -> bool {
        if self.pristine() {
            return true;
        }
        self.observer.is_none()
            && self.observer_terminal_error.is_none()
            && self.cold_content.is_none()
            && self.controller.is_none()
            && self.task.is_none()
            && self.scope.is_none()
            && self.steps.is_empty()
            && self.producer.is_none()
            && self.prepared.is_none()
            && self.original_prepared.is_none()
            && self.resources.is_none()
            && self.retirement_work.is_none()
            && !self.retirement_attached
            && !self.owner_callback_entered
            && self.native_steps.is_empty()
            && self.checkpoint_phase.is_none()
            && !self.preparation_entered
            && !self.context_ready
            && !self.successors_entered
            && !self.successors_ready
            && !self.recording_entered
            && !self.recorded
            && !self.frozen
            && self.delivery.is_none()
            && !self.vjp_entered
            && !self.vjp_recorded
            && self.completion.is_none()
            && !self.producer_retirement_entered
            && !self.producer_retired
            && !self.graph_retired
            && !self.resources_retired
            && !self.native_released
    }
}

struct ActorRefreshConstruction {
    native: xlog_cuda::SemanticTransitionSessionConstruction,
    records: SemanticAdmissionRecords,
    parent: Py<cold_task::PySemanticTransitionFreshParent>,
    task_ground: ColdValue,
    editable: Option<cold_task::EditableTaskSource>,
    capacities: (u32, u32, u32, u32),
    limits: (u32, u32, u32, usize),
    device: usize,
    memory: u64,
    proposal_expense: Arc<Mutex<ProposalExpense>>,
    checkpoint_sources: Arc<Mutex<CheckpointSources>>,
    canary: Option<TrainingCanaryOwner>,
}

enum ActorRefreshCompletion {
    Observed {
        _outcomes: Vec<xlog_cuda::SemanticPreparedStepOutcome>,
    },
    InitializerRefused {
        _certificate: xlog_cuda::SemanticActorRefreshInitializerRefusal,
    },
    Cancelled {
        proof: xlog_cuda::SemanticPreparedSegmentNonSubmission,
    },
}

#[pyclass(
    name = "SemanticPreparedActorRefresh",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticPreparedActorRefresh {
    pub(super) inner: Arc<ActorRefreshCustody>,
}

/// Resolve one original row's canonical SOURCE, never the enclosing task.
pub(super) fn replay_source_checkpoint(
    _py: Python<'_>,
    target: &PySemanticTransitionTaskUse,
    ordinal: u64,
    refresh_snapshot: &Bound<'_, PyAny>,
) -> PyResult<Arc<VerifiedCheckpointSource>> {
    if !refresh_snapshot.is_callable() {
        return Err(invalid(
            "original replay source requires current authority refresh",
        ));
    }
    let mut snapshot = target.state()?.snapshot.clone();
    let refreshed = refresh_checkpoint_authority(
        &target.authority,
        &mut snapshot,
        refresh_snapshot,
        "training",
        true,
    );
    target.state()?.snapshot = snapshot;
    refreshed?;
    let roster = target.checkpoint.original_training_roster()?;
    let ordinal = usize::try_from(ordinal)
        .map_err(|_| invalid("original replay source ordinal exceeds this host"))?;
    let row = roster.fields(2)?[0]
        .sequence()?
        .get(ordinal)
        .ok_or_else(|| invalid("original replay source row is absent"))?;
    let row = ReplayRow::parse_with_live(row, &target.authority.live)?;
    let referent = row
        .checkpoint_referent()?
        .or(row.pre_action_checkpoint_referent()?)
        .or(row.recovered_prefill_referent()?)
        .ok_or_else(|| invalid("original replay source has no complete checkpoint referent"))?;
    let source = target
        .checkpoint
        .checkpoint_sources
        .lock()
        .map_err(|_| invalid("checkpoint source owner mutex is poisoned"))?
        .verified
        .get(&referent.checkpoint_digest)
        .map(Arc::clone)
        .ok_or_else(|| invalid("original replay source was not retained by canonical admission"))?;
    referent.verify_source(&source)?;
    if source.cold.is_none() {
        return Err(invalid(
            "original replay source has no canonical executable task admission",
        ));
    }
    Ok(source)
}

pub(super) fn replay_source_projection(
    py: Python<'_>,
    source: &VerifiedCheckpointSource,
) -> PyResult<Py<PyTuple>> {
    let cold = source
        .cold
        .as_ref()
        .ok_or_else(|| invalid("original source lost its canonical cold task"))?;
    let cold = (
        cold.initial_theory.as_str(),
        cold.input_facts.as_str(),
        cold.observer_source.as_deref(),
        PyTuple::new(py, &cold.statements)?,
        PyTuple::new(py, &cold.query_records)?,
    )
        .into_pyobject(py)?
        .unbind()
        .into_any();
    let fields = ColdValue::Sequence(vec![
        ColdValue::Sequence(source.seed.authority.clone()),
        ColdValue::Sequence(source.seed.evaluation.clone()),
        source.seed.training_domain.clone(),
        source.seed.initial_sources.clone(),
        source.seed.source_mapping.clone(),
        source.seed.replay_capacity.clone(),
    ])
    .python_value(py)?;
    Ok(PyTuple::new(
        py,
        [
            PyBytes::new(py, &source.bytes).unbind().into_any(),
            cold,
            fields,
        ],
    )?
    .unbind())
}

impl ActorRefreshCustody {
    fn original_retirement_work(&self, py: Python<'_>) -> PyResult<SemanticColdNativeWork> {
        self.require_access(py)?;
        let update = self.update.borrow(py);
        let session = update.session.borrow(py);
        let work = session
            .active_cold_model_work
            .lock()
            .map_err(|_| invalid("active cold callback custody mutex is poisoned"))?
            .as_ref()
            .map(|work| work.clone_ref(py))
            .ok_or_else(|| {
                invalid("actor final use requires the original outer retirement report")
            })?;
        let work = work.borrow(py);
        let result = work
            .reader
            .borrow(py)
            .session
            .borrow(py)
            .owner()?
            .share_cold_native_work(&work.inner)
            .map_err(xlog_err);
        result
    }

    fn finish_after_parent_completion(
        &self,
        py: Python<'_>,
        outer: Option<&xlog_cuda::SemanticPreparedSegmentNonSubmission>,
        streams: &[u64],
    ) -> PyResult<()> {
        if let Some(outer) = outer {
            for bank in 0..2 {
                self.retire_cancelled_constructor(py, bank, outer)?;
            }
            if self.children()?.iter().all(|child| child.retired) {
                return Ok(());
            }
        }
        let work = self.original_retirement_work(py)?;
        for bank in 0..2 {
            let (session, prepared) = {
                let children = self.children()?;
                let child = &children[bank];
                if child.retired || child.producer_retired {
                    continue;
                }
                if outer.is_none() && !child.frozen {
                    return Err(invalid(
                        "actor final use requires both original frozen children",
                    ));
                }
                (
                    child
                        .session
                        .as_ref()
                        .ok_or_else(|| {
                            invalid("actor final use retains its unresolved original constructor")
                        })?
                        .clone_ref(py),
                    child
                        .prepared
                        .as_ref()
                        .or(child.original_prepared.as_ref())
                        .ok_or_else(|| invalid("actor final use lost its original producer"))?
                        .clone_ref(py),
                )
            };
            if let Some(outer) = outer {
                if self.children()?[bank].completion.is_none() {
                    let cancellation = {
                        let update = self.update.borrow(py);
                        let target = update.session.borrow(py);
                        let child = session.borrow(py);
                        let target_owner = target.owner()?;
                        let mut child_owner = child.owner()?;
                        target_owner
                            .cancel_prepared_actor_refresh(
                                &update.inner,
                                bank,
                                Some(&mut *child_owner),
                                &self.proof,
                                outer,
                            )
                            .map_err(xlog_err)?
                            .ok_or_else(|| {
                                invalid(
                                    "constructed actor cancellation lost its original child proof",
                                )
                            })?
                    };
                    self.children()?[bank].completion = Some(ActorRefreshCompletion::Cancelled {
                        proof: cancellation,
                    });
                }
                if !matches!(
                    self.children()?[bank].completion,
                    Some(ActorRefreshCompletion::Cancelled { .. })
                ) {
                    return Err(invalid(
                        "actor cancellation cannot replace an observed execution",
                    ));
                }
            } else if matches!(
                self.children()?[bank].completion,
                Some(ActorRefreshCompletion::Cancelled { .. })
            ) {
                return Err(invalid(
                    "cancelled actor cannot become a completed execution",
                ));
            }
            let cancellation = {
                let children = self.children()?;
                match children[bank].completion.as_ref() {
                    Some(ActorRefreshCompletion::Cancelled { proof }) => Some(proof.clone()),
                    _ => None,
                }
            };
            let attached = self.children()?[bank].retirement_attached;
            if !attached {
                {
                    let mut children = self.children()?;
                    let child = &mut children[bank];
                    if let Some(original) = &child.retirement_work {
                        if !original.same_custody(&work) {
                            return Err(invalid("actor final use changed its original report"));
                        }
                    } else {
                        child.retirement_work = Some(work.clone());
                    }
                }
                session
                    .borrow(py)
                    .owner()?
                    .attach_actor_refresh_retirement_work(
                        &self.proof,
                        bank,
                        work.clone(),
                        cancellation.as_ref(),
                    )
                    .map_err(xlog_err)?;
                self.children()?[bank].retirement_attached = true;
            }
            if self.children()?[bank].completion.is_none() {
                let completion = {
                    let source = session.borrow(py);
                    let mut owner = source.owner()?;
                    match owner.complete_prepared_segment() {
                        Ok(outcomes) => ActorRefreshCompletion::Observed { _outcomes: outcomes },
                        Err(xlog_cuda::SemanticTransitionError::PublicationRefused { .. }) => {
                            ActorRefreshCompletion::InitializerRefused { _certificate: owner.actor_refresh_initializer_refusal()
                                .map_err(xlog_err)?.ok_or_else(|| invalid("actor refusal has no original native initializer certificate"))? }
                        }
                        Err(error) => return Err(xlog_err(error)),
                    }
                };
                self.children()?[bank].completion = Some(completion);
            }
            // This graph is the original outer graph. Detaching a child never
            // creates or destroys an independent executable.
            if !self.children()?[bank].graph_retired {
                let graph = {
                    let source = session.borrow(py);
                    let mut owner = source.owner()?;
                    match &cancellation {
                        Some(proof) => {
                            owner.take_cancelled_prepared_executable_for_retirement(proof)
                        }
                        None => owner.take_prepared_executable_for_retirement(),
                    }
                    .map_err(xlog_err)?
                };
                drop(graph);
                {
                    let source = session.borrow(py);
                    let owner = source.owner()?;
                    match &cancellation {
                        Some(proof) => owner.require_cancelled_prepared_graph_retirement(proof),
                        None => owner.require_completed_prepared_graph_retirement(),
                    }
                    .map_err(xlog_err)?;
                }
                self.children()?[bank].graph_retired = true;
            }
            let mut native_steps = self.children()?[bank].native_steps.clone();
            if native_steps.is_empty() {
                native_steps = session
                    .borrow(py)
                    .owner()?
                    .prepared_segment_preparation_handles()
                    .map_err(xlog_err)?;
                self.children()?[bank].native_steps = native_steps.clone();
            }
            for step in &native_steps {
                let source = session.borrow(py);
                let mut owner = source.owner()?;
                match &cancellation {
                    Some(proof) => owner.quiesce_cancelled_prepared_step(step, proof, streams),
                    None => owner.quiesce_prepared_step(step, streams),
                }
                .map_err(xlog_err)?;
            }
            {
                let mut children = self.children()?;
                let child = &mut children[bank];
                if child.producer_retirement_entered {
                    return Err(invalid(
                        "original actor producer retirement cannot repeat after an uncertain callback",
                    ));
                }
                child.producer_retirement_entered = true;
            }
            let check = || self.require_access(py);
            let finished =
                recording_callback(check, || prepared.bind(py).call_method0("finish_segment"))?;
            if !finished.is_none() {
                return Err(invalid(
                    "actor finish_segment must return None after original producer retirement",
                ));
            }
            self.children()?[bank].producer_retired = true;
        }
        Ok(())
    }

    fn retire_cancelled_constructor(
        &self,
        py: Python<'_>,
        bank: usize,
        outer: &xlog_cuda::SemanticPreparedSegmentNonSubmission,
    ) -> PyResult<()> {
        self.require_access(py)?;
        let (construction, retired) = {
            let children = self.children()?;
            let child = &children[bank];
            if child.retired {
                return Ok(());
            }
            if child.observer.is_some() {
                return Err(SemanticRetainedFinalUsePending::new_err(
                    "retain the entered actor observer and constructor; cancellation cannot certify its unfinished observation",
                ));
            }
            if !child.constructor_only() {
                return Ok(());
            }
            (child.construction.as_ref().map(Arc::clone), child.constructor_retired)
        };
        if !retired {
            let update = self.update.borrow(py);
            let target = update.session.borrow(py);
            let target = target.owner()?;
            if let Some(construction) = construction {
                let mut construction = construction.lock().map_err(|_| invalid("original actor constructor mutex is poisoned"))?;
                target.retire_actor_refresh_construction(
                    &update.inner, bank, &self.proof, outer, &mut construction.native,
                ).map_err(xlog_err)?;
            } else {
                // No native constructor claimed this bank. The original outer
                // proof authenticates absence, including allocation-only metadata.
                if target.cancel_prepared_actor_refresh(
                    &update.inner, bank, None, &self.proof, outer,
                ).map_err(xlog_err)?.is_some() {
                    return Err(invalid("unconstructed actor bank acquired a native child proof"));
                }
            }
            let owners = {
                let mut children = self.children()?;
                let child = &mut children[bank];
                child.constructor_retired = true;
                (child.session.take(), child.construction.take(), child.allocation.take(), child.cold_work.take())
            };
            if let Some(session) = &owners.0 {
                let session = session.borrow(py);
                session.issuance.fetch_add(1, Ordering::AcqRel);
                session.actor_refresh_child.lock().map_err(|_| invalid("actor Session custody mutex is poisoned"))?.take();
                session.training_canary_owner.lock().map_err(|_| invalid("symbolic canary owner mutex is poisoned"))?.take();
            }
            drop(owners);
        }
        let memory = self.children()?[bank].release_memory.as_ref().map(Arc::clone);
        if let Some(memory) = memory {
            memory.reap_pending_deallocations().map_err(xlog_err)?;
        }
        let original = {
            let mut children = self.children()?;
            std::mem::replace(&mut children[bank], ActorRefreshChild {
                retired: true,
                ..ActorRefreshChild::default()
            })
        };
        drop(original);
        Ok(())
    }

    fn retire_after_parent_callback(&self, py: Python<'_>, streams: &[u64]) -> PyResult<()> {
        self.require_access(py)?;
        for bank in 0..2 {
            let (session, steps, native_steps, cancellation, resources_done) = {
                let mut children = self.children()?;
                let child = &mut children[bank];
                if child.retired {
                    continue;
                }
                if !child.producer_retired || child.completion.is_none() || !child.graph_retired {
                    return Err(invalid(
                        "actor resource retirement requires actual native completion and original producer cleanup",
                    ));
                }
                if let Some(original) = &child.retirement_streams {
                    if original != streams {
                        return Err(invalid(
                            "actor final use changed its original stream roster",
                        ));
                    }
                } else {
                    child.retirement_streams = Some(streams.to_vec());
                }
                (
                    child
                        .session
                        .as_ref()
                        .expect("retained actor Session")
                        .clone_ref(py),
                    child
                        .steps
                        .iter()
                        .map(|step| step.clone_ref(py))
                        .collect::<Vec<_>>(),
                    child.native_steps.clone(),
                    match child.completion.as_ref() {
                        Some(ActorRefreshCompletion::Cancelled { proof }) => Some(proof.clone()),
                        _ => None,
                    },
                    child.resources_retired,
                )
            };
            if !resources_done {
                // Alias finalizers run only after the caller's cached outer
                // acknowledgement, never under the native Session mutex.
                for step in &steps {
                    step.borrow(py).release_recorded_producer_aliases(py)?;
                }
                drain_export_owners();
                for step in &native_steps {
                    {
                        let source = session.borrow(py);
                        let mut owner = source.owner()?;
                        match &cancellation {
                            Some(proof) => {
                                owner.release_cancelled_prepared_step(step, proof, streams)
                            }
                            None => owner.release_prepared_step(step, streams),
                        }
                        .map_err(xlog_err)?;
                    }
                    let removed = {
                        let mut children = self.children()?;
                        let child = &mut children[bank];
                        let index = child
                            .native_steps
                            .iter()
                            .position(|original| original.same_handle(step))
                            .ok_or_else(|| invalid("actor retirement changed its original step"))?;
                        child.native_steps.remove(index);
                        child
                            .steps
                            .iter()
                            .position(|original| original.borrow(py).inner.same_handle(step))
                            .map(|index| child.steps.remove(index))
                    };
                    let removed_native = session
                        .borrow(py)
                        .prepared_segment
                        .lock()
                        .map_err(|_| invalid("prepared segment mutex is poisoned"))?
                        .as_mut()
                        .and_then(|original| {
                            original
                                .steps
                                .iter()
                                .position(|original| original.borrow(py).inner.same_handle(step))
                                .map(|index| original.steps.remove(index))
                        });
                    drop((removed, removed_native));
                }
                let native = session
                    .borrow(py)
                    .owner()?
                    .take_actor_refresh_resources_for_retirement(
                        &self.proof,
                        bank,
                        cancellation.as_ref(),
                    )
                    .map_err(xlog_err)?;
                drop(native);
                let retained = session
                    .borrow(py)
                    .prepared_segment
                    .lock()
                    .map_err(|_| invalid("prepared segment mutex is poisoned"))?
                    .take();
                drop(retained);
                self.children()?[bank].resources_retired = true;
            }
            let work = self.children()?[bank]
                .retirement_work
                .as_ref()
                .cloned()
                .ok_or_else(|| invalid("actor final use lost its original report custody"))?;
            if self.children()?[bank].retirement_attached {
                session
                    .borrow(py)
                    .owner()?
                    .detach_shared_cold_native_work(&work)
                    .map_err(xlog_err)?;
                self.children()?[bank].retirement_attached = false;
            }
            if !self.children()?[bank].native_released {
                let native = {
                    let source = session.borrow(py);
                    let mut owner = source
                        .inner
                        .lock()
                        .map_err(|_| invalid("native actor Session mutex is poisoned"))?;
                    let live = owner
                        .as_mut()
                        .ok_or_else(|| invalid("original actor Session was already released"))?;
                    live.join_actor_refresh_release(&self.proof, bank, cancellation.as_ref())
                        .map_err(xlog_err)?;
                    let mut children = self.children()?;
                    if children[bank].release_memory.is_none() {
                        return Err(invalid(
                            "actor release lost its original allocation manager",
                        ));
                    }
                    children[bank].native_released = true;
                    source.issuance.fetch_add(1, Ordering::AcqRel);
                    owner.take().expect("joined original actor Session")
                };
                native
                    .finish_retired_publication_release()
                    .map_err(xlog_err)?;
            } else {
                let memory = self.children()?[bank]
                    .release_memory
                    .as_ref()
                    .map(Arc::clone)
                    .ok_or_else(|| invalid("actor release lost its original allocation manager"))?;
                memory.reap_pending_deallocations().map_err(xlog_err)?;
            }
            let original = {
                let mut children = self.children()?;
                std::mem::replace(
                    &mut children[bank],
                    ActorRefreshChild {
                        retired: true,
                        ..ActorRefreshChild::default()
                    },
                )
            };
            let custody = session
                .borrow(py)
                .actor_refresh_child
                .lock()
                .map_err(|_| invalid("actor Session custody mutex is poisoned"))?
                .take();
            let canary = session
                .borrow(py)
                .training_canary_owner
                .lock()
                .map_err(|_| invalid("symbolic canary owner mutex is poisoned"))?
                .take();
            // The child drops its typed initializer-refusal certificate only
            // after canonical final use; the outer owner keeps its own pins.
            drop((original, custody, canary));
            drain_export_owners();
        }
        Ok(())
    }

    pub(super) fn matches_update(
        &self,
        py: Python<'_>,
        update: &Py<PySemanticPreparedStep>,
    ) -> bool {
        self.update.as_ptr() == update.as_ptr()
            && self
                .update
                .borrow(py)
                .inner
                .same_handle(&update.borrow(py).inner)
    }

    pub(super) fn require_update(
        &self,
        py: Python<'_>,
        update: &PySemanticPreparedStep,
        bank: usize,
    ) -> PyResult<()> {
        self.require_access(py)?;
        if bank > 1 || !self.update.borrow(py).inner.same_handle(&update.inner) {
            return Err(invalid(
                "actor delivery changed its original Update or bank",
            ));
        }
        Ok(())
    }

    pub(super) fn bind_delivery(
        &self,
        py: Python<'_>,
        update: &PySemanticPreparedStep,
        bank: usize,
        owner: &mut SemanticTransitionSession,
        tensors: Vec<SemanticTensorInput>,
        stream: u64,
    ) -> PyResult<SemanticGradientDeliveryBinding> {
        self.require_update(py, update, bank)?;
        let (session, proposal) = {
            let children = self.children()?;
            let child = &children[bank];
            if child.delivery.is_some() || child.recording_entered {
                return Err(invalid("actor physical delivery binds once before capture"));
            }
            (
                child
                    .session
                    .as_ref()
                    .map(|owner| owner.clone_ref(py))
                    .ok_or_else(|| invalid("actor delivery lost its original child"))?,
                child
                    .steps
                    .get(1)
                    .map(|step| step.borrow(py).inner.clone())
                    .ok_or_else(|| invalid("actor delivery lost its fresh Proposal"))?,
            )
        };
        let source = session.borrow(py);
        let source = source.owner()?;
        owner
            .bind_prepared_actor_gradient_delivery(
                &update.inner,
                bank,
                &source,
                &proposal,
                &self.proof,
                &self.row,
                tensors,
                stream,
            )
            .map_err(xlog_err)
    }

    pub(super) fn retain_delivery(
        &self,
        py: Python<'_>,
        bank: usize,
        delivery: &Py<PySemanticGradientDelivery>,
    ) -> PyResult<()> {
        let mut children = self.children()?;
        if children[bank].delivery.is_some() {
            return Err(invalid("actor physical delivery cannot be replaced"));
        }
        children[bank].delivery = Some(delivery.clone_ref(py));
        Ok(())
    }

    fn record_bank(
        &self,
        py: Python<'_>,
        bank: usize,
        sequence: &mut xlog_cuda::cuda_graph::ConditionalCudaGraphBodySequence,
    ) -> PyResult<()> {
        let (session, task, prepared, steps) = (|| -> PyResult<_> {
            self.require_access(py)?;
            let mut children = self.children()?;
            let child = &mut children[bank];
            if child.recording_entered || !child.successors_ready || child.delivery.is_none() {
                return Err(invalid(
                    "actor capture requires one prepared child and original physical delivery",
                ));
            }
            child.recording_entered = true;
            Ok((
                child
                    .session
                    .as_ref()
                    .expect("prepared actor Session")
                    .clone_ref(py),
                child
                    .task
                    .as_ref()
                    .expect("prepared actor task")
                    .clone_ref(py),
                child
                    .prepared
                    .as_ref()
                    .expect("prepared actor producer")
                    .clone_ref(py),
                child
                    .steps
                    .iter()
                    .map(|step| step.clone_ref(py))
                    .collect::<Vec<_>>(),
            ))
        })()?;
        let stream = session
            .borrow(py)
            .owner()?
            .prepared_stream(&steps[0].borrow(py).inner)
            .map_err(xlog_err)?;
        let capture_error = std::cell::RefCell::new(None);
        let captured = sequence.capture_segment_on_stream(&stream, || -> PyResult<()> {
            let result = (|| {
                let update = self.update.borrow(py);
                let target = update.session.borrow(py);
                let child = session.borrow(py);
                target
                    .owner()?
                    .record_prepared_actor_refresh(
                        &update.inner,
                        bank,
                        &mut child.owner()?,
                        &self.proof,
                    )
                    .map_err(xlog_err)
            })();
            if let Err(error) = &result {
                *capture_error.borrow_mut() = Some(error.clone_ref(py));
            }
            result
        });
        if let Err(error) = captured {
            return Err(capture_error
                .into_inner()
                .unwrap_or_else(|| xlog_err(error)));
        }
        record_prepared_steps(
            &mut PreparedCaptureSequence::Body(sequence),
            py,
            &session.borrow(py),
            &task.borrow(py),
            &steps,
            prepared.bind(py),
            &stream,
            &|| self.check_child(py, bank),
            None,
            &[],
        )?;
        self.children()?[bank].recorded = true;
        Ok(())
    }

    pub(super) fn finish_recording(&self, py: Python<'_>) -> PyResult<()> {
        for bank in 0..2 {
            let (session, task, scope, steps) = {
                let children = self.children()?;
                let child = &children[bank];
                if child.frozen {
                    continue;
                }
                if !child.recorded || !child.vjp_recorded {
                    return Err(invalid(
                        "actor freeze requires its genuine R/P and target VJP in both banks",
                    ));
                }
                (
                    child
                        .session
                        .as_ref()
                        .expect("recorded actor Session")
                        .clone_ref(py),
                    child
                        .task
                        .as_ref()
                        .expect("recorded actor task")
                        .clone_ref(py),
                    Arc::clone(child.scope.as_ref().expect("recorded actor scope")),
                    child
                        .steps
                        .iter()
                        .map(|step| step.clone_ref(py))
                        .collect::<Vec<_>>(),
                )
            };
            for step in &steps {
                step.borrow(py).finish_gradient_delivery_recording(py)?;
            }
            let update = self.update.borrow(py);
            update
                .session
                .borrow(py)
                .owner()?
                .finish_prepared_actor_refresh(
                    &update.inner,
                    bank,
                    &mut session.borrow(py).owner()?,
                    &self.proof,
                )
                .map_err(xlog_err)?;
            task.borrow(py).state()?.finish_build(&scope)?;
            session.borrow(py).recording.store(false, Ordering::Release);
            self.children()?[bank].frozen = true;
        }
        Ok(())
    }
    fn children(&self) -> PyResult<MutexGuard<'_, [ActorRefreshChild; 2]>> {
        self.children
            .lock()
            .map_err(|_| invalid("original actor child custody is poisoned"))
    }

    pub(super) fn require_access(&self, py: Python<'_>) -> PyResult<()> {
        let update = self.update.borrow(py);
        let session = update.session.borrow(py);
        session.require_creator()?;
        update.require_task(py, &update.task_use.borrow(py))?;
        let stored = session
            .prepared_segment
            .lock()
            .map_err(|_| invalid("prepared segment mutex is poisoned"))?;
        let stored = stored
            .as_ref()
            .ok_or_else(|| invalid("actor child lost its original outer segment"))?;
        if !Arc::ptr_eq(&stored.scope, &update.scope)
            || !stored
                .actor_refreshes
                .iter()
                .any(|original| std::ptr::eq(self, original.as_ref()))
        {
            return Err(invalid(
                "actor child requires its original admitted Update custody",
            ));
        }
        Ok(())
    }

    pub(super) fn allocation(
        self: &Arc<Self>,
        py: Python<'_>,
        bank: usize,
    ) -> PyResult<CheckpointAllocationDomain> {
        self.require_access(py)?;
        if bank > 1 {
            return Err(invalid("actor child bank must be zero or one"));
        }
        if let Some(original) = self.children()?[bank].allocation.as_ref() {
            return Ok(original.clone());
        }
        let update = self.update.borrow(py);
        let session = update.session.borrow(py);
        let work = session
            .active_cold_model_work
            .lock()
            .map_err(|_| invalid("active cold callback custody mutex is poisoned"))?
            .as_ref()
            .map(|work| work.clone_ref(py))
            .ok_or_else(|| {
                invalid("actor construction requires its original Update cold callback")
            })?;
        let work = work.borrow(py);
        let original = prepared_cold_callback_step(py, &session, &work)?
            .ok_or_else(|| invalid("actor construction lost its original prepared cold work"))?;
        if original.as_ptr() != self.update.as_ptr() {
            return Err(invalid(
                "actor construction changed its original Update cold work",
            ));
        }
        let mut children = self.children()?;
        if children[bank].construction_entered {
            return Err(invalid(
                "retain the original actor child constructor without repeating it",
            ));
        }
        let cold_work = work
            .reader
            .borrow(py)
            .session
            .borrow(py)
            .owner()?
            .share_cold_native_work(&work.inner)
            .map_err(xlog_err)?;
        let owner = session.owner()?;
        let (provider, domain) = owner.checkpoint_allocation_domain().map_err(xlog_err)?;
        children[bank].release_memory = Some(Arc::clone(provider.memory()));
        children[bank].cold_work = Some(cold_work.clone());
        children[bank].construction_entered = true;
        let allocation = CheckpointAllocationDomain {
            provider,
            domain,
            cold_work: Some(cold_work),
        };
        children[bank].allocation = Some(allocation.clone());
        Ok(allocation)
    }

    #[expect(clippy::too_many_arguments, reason = "the original constructor freezes the complete native admission and Python ownership inputs")]
    pub(super) fn construct_session(
        self: &Arc<Self>,
        py: Python<'_>,
        bank: usize,
        records: SemanticAdmissionRecords,
        parent: &Py<cold_task::PySemanticTransitionFreshParent>,
        task_ground: &ColdValue,
        capacities: (u32, u32, u32, u32),
        limits: (u32, u32, u32, usize),
        device: usize,
        memory: u64,
        proposal_expense: Arc<Mutex<ProposalExpense>>,
        checkpoint_sources: Arc<Mutex<CheckpointSources>>,
        canary: Option<TrainingCanaryOwner>,
    ) -> PyResult<(Py<PySemanticTransitionSession>, Py<cold_task::PySemanticTransitionFreshParent>)> {
        self.require_access(py)?;
        let original = {
            let mut children = self.children()?;
            let child = children.get_mut(bank).ok_or_else(|| invalid("actor child bank must be zero or one"))?;
            if child.retired {
                return Err(invalid("original actor constructor was retired"));
            }
            if child.construction.is_none() {
                let allocation = child.allocation.as_ref().ok_or_else(|| invalid("actor construction lost its original allocation"))?;
                if allocation.provider.device().ordinal() != device
                    || allocation.provider.memory().budget_limit_bytes() != memory {
                    return Err(invalid("actor constructor changed its original device or memory budget"));
                }
                let editable = cold_task::editable_source_from_admission(&records)?;
                let update = self.update.borrow(py);
                let target = update.session.borrow(py);
                let native = target.owner()?.prepare_actor_refresh_construction(
                    &update.inner, bank, &self.proof, records.clone(),
                    SemanticHypergraphCapacities::try_new(capacities.0, capacities.1, capacities.2, capacities.3).map_err(val_err)?,
                    SemanticAdmissionLimits { max_records: limits.0, max_terms: limits.1, max_references: limits.2, max_utf8_bytes: limits.3 },
                    editable.as_ref().map(|source| Arc::clone(&source.program)),
                    child.cold_work.clone().ok_or_else(|| invalid("actor constructor lost its original cold work"))?,
                ).map_err(xlog_err)?;
                child.construction = Some(Arc::new(Mutex::new(ActorRefreshConstruction {
                    native, records: records.clone(), parent: parent.clone_ref(py),
                    task_ground: task_ground.clone(), editable, capacities, limits, device, memory,
                    proposal_expense: Arc::clone(&proposal_expense), checkpoint_sources: Arc::clone(&checkpoint_sources), canary: canary.clone(),
                })));
            }
            Arc::clone(child.construction.as_ref().expect("original actor constructor"))
        };
        let (storage, original_parent, editable_program, editable_initial_source, editable_observer_source, original_canary) = {
            let mut original = original.lock().map_err(|_| invalid("original actor constructor mutex is poisoned"))?;
            let same_canary = match (&original.canary, &canary) {
                (None, None) => true,
                (Some((source, owner)), Some((other_source, other_owner))) => Arc::ptr_eq(source, other_source) && owner.as_ptr() == other_owner.as_ptr(),
                _ => false,
            };
            if original.records != records || original.parent.borrow(py).ne(&*parent.borrow(py))
                || original.task_ground != *task_ground || original.capacities != capacities
                || original.limits != limits || original.device != device || original.memory != memory
                || !Arc::ptr_eq(&original.proposal_expense, &proposal_expense)
                || !Arc::ptr_eq(&original.checkpoint_sources, &checkpoint_sources) || !same_canary {
                return Err(invalid("actor constructor changed its original admitted inputs"));
            }
            if let Some(session) = self.children()?[bank].session.as_ref() {
                return Ok((session.clone_ref(py), original.parent.clone_ref(py)));
            }
            let update = self.update.borrow(py);
            let target = update.session.borrow(py);
            let storage = target.owner()?.resolve_actor_refresh_construction(
                &update.inner, bank, &self.proof, &mut original.native,
            ).map_err(xlog_err)?;
            if let Some((source, _)) = &original.canary {
                storage.lock().map_err(|_| invalid("original constructed Session mutex is poisoned"))?
                    .as_mut().ok_or_else(|| invalid("original constructed Session was released"))?
                    .bind_training_canary_source(Arc::clone(source)).map_err(xlog_err)?;
            }
            (
                storage, original.parent.clone_ref(py),
                original.editable.as_ref().map(|source| Arc::clone(&source.program)),
                original.editable.as_ref().map(|source| source.initial_source.clone()),
                original.editable.as_ref().and_then(|source| source.observer_source.clone()),
                original.canary.clone(),
            )
        };
        let mut wrapper = PySemanticTransitionSession::from_shared_native(
            storage, None, device, capacities, limits, memory, proposal_expense, checkpoint_sources, original_canary,
        );
        wrapper.editable_program = editable_program;
        wrapper.editable_initial_source = editable_initial_source;
        wrapper.editable_observer_source = editable_observer_source;
        let session = Py::new(py, wrapper)?;
        self.retain_session(py, bank, &session)?;
        Ok((session, original_parent))
    }

    pub(super) fn constructed_content(&self, bank: usize) -> PyResult<Option<(xlog_cuda::SemanticTaskContentIdentity, Vec<xlog_cuda::SemanticTruth>)>> {
        Ok(self.children()?.get(bank).ok_or_else(|| invalid("actor child bank must be zero or one"))?.cold_content.clone())
    }

    pub(super) fn observe_constructed_content(
        &self,
        py: Python<'_>,
        bank: usize,
        program: Arc<dyn xlog_cuda::SemanticTaskProgram>,
        statements: &[u32],
        ground: &xlog_cuda::SemanticTaskGround,
    ) -> PyResult<(xlog_cuda::SemanticTaskContentIdentity, Vec<xlog_cuda::SemanticTruth>)> {
        self.require_access(py)?;
        let session = {
            let mut children = self.children()?;
            let child = children.get_mut(bank).ok_or_else(|| invalid("actor child bank must be zero or one"))?;
            if let Some(content) = &child.cold_content {
                return Ok(content.clone());
            }
            if let Some(error) = &child.observer_terminal_error {
                return Err(error.clone_ref(py));
            }
            if child.observer.is_some() {
                return Err(SemanticPreparedSegmentPending::new_err(
                    "retain the original actor observer and constructor; an entered observer cannot be called again",
                ));
            }
            let session = child.session.as_ref().ok_or_else(|| invalid("actor observer lost its original Session"))?.clone_ref(py);
            session
        };
        let (result, entered, completed) = {
            let session = session.borrow(py);
            let mut native = session.owner()?;
            {
                let mut children = self.children()?;
                // Keep the executable observer alive before its one original
                // call. Only native entry metadata classifies an error.
                children[bank].observer = Some(Arc::clone(&program));
            }
            let result = native.observe_cold_task_content(
                statements, &[], program.as_ref(), ground,
            );
            (result, native.cold_task_observer_entered(), native.cold_task_observer_completed())
        };
        let content = match result {
            Ok(content) => content,
            Err(error) if !entered => {
                self.children()?[bank].observer = None;
                return Err(xlog_err(error));
            }
            Err(error) if completed => {
                let error = xlog_err(error);
                self.children()?[bank].observer_terminal_error = Some(error.clone_ref(py));
                return Err(error);
            }
            Err(error) => {
                let pending = SemanticPreparedSegmentPending::new_err(
                    "retain the original actor observer and constructor; do not replay the observer or retire it as unentered construction",
                );
                pending.set_cause(py, Some(xlog_err(error)));
                return Err(pending);
            }
        };
        let mut children = self.children()?;
        let child = &mut children[bank];
        child.cold_content = Some(content.clone());
        child.observer = None;
        Ok(content)
    }

    pub(super) fn constructed_controller(&self, py: Python<'_>, bank: usize) -> PyResult<Option<Py<PySemanticTransitionController>>> {
        Ok(self.children()?.get(bank).ok_or_else(|| invalid("actor child bank must be zero or one"))?
            .controller.as_ref().map(|controller| controller.clone_ref(py)))
    }

    pub(super) fn shared_import_owners(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Arc<Mutex<ProposalExpense>>, Arc<Mutex<CheckpointSources>>)> {
        self.require_access(py)?;
        let update = self.update.borrow(py);
        let task = update.task_use.borrow(py);
        Ok((
            Arc::clone(&task.checkpoint.proposal_expense),
            Arc::clone(&task.checkpoint.checkpoint_sources),
        ))
    }

    pub(super) fn retain_session(
        self: &Arc<Self>,
        py: Python<'_>,
        bank: usize,
        session: &Py<PySemanticTransitionSession>,
    ) -> PyResult<()> {
        self.require_access(py)?;
        let mut children = self.children()?;
        let child = children
            .get_mut(bank)
            .ok_or_else(|| invalid("actor child bank must be zero or one"))?;
        if !child.construction_entered || child.session.is_some() {
            return Err(invalid("actor child cannot replace its original Session"));
        }
        *session
            .borrow(py)
            .actor_refresh_child
            .lock()
            .map_err(|_| invalid("actor Session custody is poisoned"))? =
            Some(ActorRefreshSession {
                owner: Arc::downgrade(self),
                bank,
            });
        child.session = Some(session.clone_ref(py));
        Ok(())
    }

    pub(super) fn retain_controller(
        &self,
        py: Python<'_>,
        bank: usize,
        controller: &Py<PySemanticTransitionController>,
    ) -> PyResult<()> {
        let mut children = self.children()?;
        let child = children
            .get_mut(bank)
            .ok_or_else(|| invalid("actor child bank must be zero or one"))?;
        if child.session.is_none() || child.controller.as_ref().is_some_and(|original| original.as_ptr() != controller.as_ptr()) {
            return Err(invalid(
                "actor child cannot replace its original Controller",
            ));
        }
        child.controller = Some(controller.clone_ref(py));
        Ok(())
    }

    pub(super) fn source_checkpoint(&self) -> Arc<VerifiedCheckpointSource> {
        Arc::clone(&self.source)
    }
    pub(super) fn original_row(&self) -> &ReplayRow {
        &self.original_row
    }
    pub(super) fn material(&self) -> &xlog_cuda::SemanticReplayMaterial {
        &self.replay.material
    }

    fn outer_memory(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let update = self.update.borrow(py);
        let session = update.session.borrow(py);
        recording_callback(
            || self.require_access(py),
            || session.prepared_memory_scope(py, &update.scope, None),
        )
    }

    fn check_child(&self, py: Python<'_>, bank: usize) -> PyResult<()> {
        self.require_access(py)?;
        let (controller, task, scope) = {
            let children = self.children()?;
            let child = &children[bank];
            (
                child
                    .controller
                    .as_ref()
                    .map(|owner| owner.clone_ref(py))
                    .ok_or_else(|| invalid("actor child lost its original Controller"))?,
                child
                    .task
                    .as_ref()
                    .map(|owner| owner.clone_ref(py))
                    .ok_or_else(|| invalid("actor child lost its original task admission"))?,
                child
                    .scope
                    .as_ref()
                    .map(Arc::clone)
                    .ok_or_else(|| invalid("actor child lost its original recording scope"))?,
            )
        };
        let task = task.borrow(py);
        let snapshot = task.state()?.snapshot.canonical.clone();
        controller
            .borrow(py)
            .check_recording(py, &task, &scope, &snapshot)
    }

    pub(super) fn prepare_successors(&self, py: Python<'_>) -> PyResult<()> {
        for bank in 0..2 {
            let prepared = {
                let mut children = self.children()?;
                let child = &mut children[bank];
                if child.successors_ready {
                    continue;
                }
                if child.successors_entered {
                    return Err(invalid(
                        "retain the original actor successor callback without repeating it",
                    ));
                }
                let prepared = child
                    .prepared
                    .as_ref()
                    .map(|owner| owner.clone_ref(py))
                    .ok_or_else(|| {
                        invalid(
                            "both original actor bank producers must be prepared before capture",
                        )
                    })?;
                child.successors_entered = true;
                prepared
            };
            recording_callback(
                || self.check_child(py, bank),
                || {
                    let value = prepared
                        .bind(py)
                        .getattr("prepare_unpublished_successors")?
                        .call0()?;
                    if !value.is_none() {
                        return Err(invalid("actor successor preparation must return None"));
                    }
                    Ok(())
                },
            )?;
            self.children()?[bank].successors_ready = true;
        }
        Ok(())
    }

    pub(super) fn retain_import_task(
        &self,
        py: Python<'_>,
        bank: usize,
        task: &Py<PySemanticTransitionTaskUse>,
    ) -> PyResult<()> {
        self.require_access(py)?;
        let mut children = self.children()?;
        let child = children
            .get_mut(bank)
            .ok_or_else(|| invalid("actor child bank must be zero or one"))?;
        let issued = task.borrow(py);
        if child.task.is_some()
            || child
                .session
                .as_ref()
                .is_none_or(|session| session.as_ptr() != issued.session.as_ptr())
            || child.controller.as_ref().is_none_or(|controller| {
                !Arc::ptr_eq(&controller.borrow(py).identity, &issued.controller)
            })
        {
            return Err(invalid(
                "actor import changed its original Session or Controller",
            ));
        }
        child.task = Some(task.clone_ref(py));
        Ok(())
    }
}

#[pymethods]
impl PySemanticPreparedActorRefresh {
    #[pyo3(signature = (update_step, bank, *, consumer_stream))]
    fn group_update_vjp(
        &self,
        py: Python<'_>,
        update_step: Py<PySemanticPreparedStep>,
        bank: usize,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let update = update_step.borrow(py);
        self.inner.require_update(py, &update, bank)?;
        let (session, proposal, delivery) = {
            let children = self.inner.children()?;
            let child = &children[bank];
            if !child.recorded || child.vjp_entered {
                return Err(invalid(
                    "actor VJP requires its unused fresh recorded Proposal",
                ));
            }
            (
                child
                    .session
                    .as_ref()
                    .expect("recorded actor Session")
                    .clone_ref(py),
                child
                    .steps
                    .get(1)
                    .expect("recorded actor Proposal")
                    .clone_ref(py),
                child
                    .delivery
                    .as_ref()
                    .ok_or_else(|| invalid("actor VJP lost its original physical delivery"))?
                    .clone_ref(py),
            )
        };
        delivery.borrow(py).require_active_hooks()?;
        if !delivery.borrow(py).activated_once.load(Ordering::Acquire) {
            return Err(invalid(
                "actor physical delivery must be active during its original VJP",
            ));
        }
        let target = update.session.borrow(py);
        let mut target_owner = target.owner()?;
        update.content_binding_with_owner(py, &target_owner)?;
        let source = session.borrow(py);
        self.inner.children()?[bank].vjp_entered = true;
        let gradients = target_owner
            .record_imported_prepared_actor_policy_vjp(
                &update.inner,
                bank,
                &mut source.owner()?,
                &proposal.borrow(py).inner,
                &self.inner.proof,
                &self.inner.row,
                stream,
            )
            .map_err(xlog_err)?;
        let capsules = gradients
            .into_dlpack()
            .map_err(xlog_err)?
            .into_iter()
            .map(|tensor| {
                let tensor =
                    retain_export_owner(tensor, session.clone_ref(py), source.owner_thread)?;
                crate::dlpack_capsule_from_tensor(py, tensor)
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.inner.children()?[bank].vjp_recorded = true;
        Ok(PyTuple::new(py, capsules)?.unbind())
    }
    #[pyo3(signature = (*, refresh_snapshot))]
    fn source_task(
        &self,
        py: Python<'_>,
        refresh_snapshot: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.inner.require_access(py)?;
        let update = self.inner.update.borrow(py);
        let source = replay_source_checkpoint(
            py,
            &update.task_use.borrow(py),
            self.inner.member_ordinal,
            refresh_snapshot,
        )?;
        if source.bytes != self.inner.source.bytes {
            return Err(invalid(
                "actor source changed after its original native binding",
            ));
        }
        replay_source_projection(py, &self.inner.source)
    }

    #[pyo3(signature = (bank, task_use, producer))]
    fn prepare_bank(
        &self,
        py: Python<'_>,
        bank: usize,
        task_use: Py<PySemanticTransitionTaskUse>,
        producer: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.inner.require_access(py)?;
        if bank > 1 {
            return Err(invalid("actor child bank must be zero or one"));
        }
        let (session, controller) = {
            let mut children = self.inner.children()?;
            let child = &mut children[bank];
            if child
                .task
                .as_ref()
                .is_none_or(|original| original.as_ptr() != task_use.as_ptr())
                || child
                    .producer
                    .as_ref()
                    .is_some_and(|original| original.as_ptr() != producer.as_ptr())
            {
                return Err(invalid(
                    "actor preparation changed its original task or producer",
                ));
            }
            child.producer = Some(producer.clone_ref(py));
            (
                child
                    .session
                    .as_ref()
                    .map(|owner| owner.clone_ref(py))
                    .ok_or_else(|| invalid("actor preparation lost its original Session"))?,
                child
                    .controller
                    .as_ref()
                    .map(|owner| owner.clone_ref(py))
                    .ok_or_else(|| invalid("actor preparation lost its original Controller"))?,
            )
        };
        let memory = self.inner.outer_memory(py)?;
        let check_original = || {
            self.inner.require_access(py)?;
            controller
                .borrow(py)
                .require_read_issued(&task_use.borrow(py))
        };
        if self.inner.children()?[bank].original_prepared.is_none() {
            {
                let mut children = self.inner.children()?;
                let child = &mut children[bank];
                if child.owner_callback_entered {
                    return Err(invalid(
                        "original actor producer construction cannot be repeated after an uncertain callback",
                    ));
                }
                child.resources = Some(Arc::new(PreparedProducerResources {
                    producers: Mutex::new(vec![producer.clone_ref(py)]),
                    owner_thread: session.borrow(py).owner_thread,
                }));
                child.owner_callback_entered = true;
            }
            let original = recording_callback(check_original, || {
                producer.bind(py).getattr("prepared_segment_owner")?.call0()
            })?
            .unbind();
            let resources = self.inner.children()?[bank]
                .resources
                .as_ref()
                .map(Arc::clone)
                .expect("original actor resources installed before construction");
            resources
                .producers
                .lock()
                .map_err(|_| invalid("actor producer ownership mutex is poisoned"))?
                .extend([original.clone_ref(py), memory.clone_ref(py)]);
            self.inner.children()?[bank].original_prepared = Some(original);
        }
        if !self.inner.children()?[bank].context_ready {
            if session
                .borrow(py)
                .owner()?
                .actor_refresh_context_started(&self.inner.proof, bank)
                .map_err(xlog_err)?
            {
                session
                    .borrow(py)
                    .owner()?
                    .resolve_actor_refresh_context(&self.inner.proof, bank)
                    .map_err(xlog_err)?;
            } else {
                session
                    .borrow(py)
                    .owner()?
                    .restore_actor_refresh_context(self.inner.material(), &self.inner.proof, bank)
                    .map_err(xlog_err)?;
            }
            self.inner.children()?[bank].context_ready = true;
        }
        let issued = task_use.borrow(py);
        if self.inner.children()?[bank].scope.is_none() {
            controller.borrow(py).require_read_issued(&issued)?;
            let checkpoint_phase = checkpoint_task_phase(&*issued.state()?)?;
            let scope = issued.state()?.begin_build()?;
            let mut children = self.inner.children()?;
            children[bank].scope = Some(scope);
            children[bank].checkpoint_phase = Some(checkpoint_phase);
            session.borrow(py).recording.store(true, Ordering::Release);
        }
        let scope = Arc::clone(
            self.inner.children()?[bank]
                .scope
                .as_ref()
                .expect("original actor recording scope"),
        );
        if self.inner.children()?[bank].native_steps.is_empty() {
            let handles = {
                let source = session.borrow(py);
                let mut owner = source.owner()?;
                if owner.prepared_segment_preparation_started() {
                    owner
                        .resolve_prepared_segment_preparation()
                        .map_err(xlog_err)?
                } else {
                    owner
                        .prepare_segment_steps(
                            [
                                SemanticTransitionKind::Recompute,
                                SemanticTransitionKind::Proposal,
                            ]
                            .into_iter(),
                            self.inner.instruction.cold_capacity,
                        )
                        .map_err(xlog_err)?
                }
            };
            self.inner.children()?[bank].native_steps = handles;
        }
        loop {
            let native = {
                let children = self.inner.children()?;
                let child = &children[bank];
                child.native_steps.get(child.steps.len()).cloned()
            };
            let Some(inner) = native else {
                break;
            };
            let step = Py::new(
                py,
                PySemanticPreparedStep {
                    session: session.clone_ref(py),
                    task_use: task_use.clone_ref(py),
                    scope: Arc::clone(&scope),
                    inner,
                    continuation_producers: Mutex::new(std::array::from_fn(|_| Vec::new())),
                    policy_inputs: Mutex::new(std::array::from_fn(|_| None)),
                    gradient_deliveries: Mutex::new(std::array::from_fn(|_| None)),
                    group_gradient_deliveries: Mutex::new(std::array::from_fn(|_| Vec::new())),
                },
            )?;
            self.inner.children()?[bank].steps.push(step);
        }
        let (steps, resources, original_prepared, checkpoint_phase) = {
            let children = self.inner.children()?;
            let child = &children[bank];
            (
                child
                    .steps
                    .iter()
                    .map(|step| step.clone_ref(py))
                    .collect::<Vec<_>>(),
                Arc::clone(child.resources.as_ref().expect("original actor resources")),
                child
                    .original_prepared
                    .as_ref()
                    .expect("original actor producer")
                    .clone_ref(py),
                child
                    .checkpoint_phase
                    .as_ref()
                    .expect("original actor task phase")
                    .clone(),
            )
        };
        {
            let source = session.borrow(py);
            let mut stored = source
                .prepared_segment
                .lock()
                .map_err(|_| invalid("prepared segment mutex is poisoned"))?;
            if let Some(original) = stored.as_ref() {
                if !Arc::ptr_eq(&original.scope, &scope)
                    || original.task_use.as_ptr() != task_use.as_ptr()
                    || original.steps.len() != steps.len()
                {
                    return Err(invalid(
                        "actor preparation changed its original retained segment",
                    ));
                }
            } else {
                *stored = Some(PreparedPythonSegment {
                    scope: Arc::clone(&scope),
                    instruction: None,
                    construction: None,
                    task_use: task_use.clone_ref(py),
                    steps: steps.iter().map(|step| step.clone_ref(py)).collect(),
                    resources,
                    prepared_owner: Some(original_prepared.clone_ref(py)),
                    memory_scope: Some(memory.clone_ref(py)),
                    checkpoint_phase,
                    producers_retired: false,
                    producer_retirement_entered: false,
                    retirement_streams: None,
                    retirement_steps_quiesced: 0,
                    graph_retired: false,
                    outcomes: None,
                    completion: None,
                    cold_callbacks: Vec::new(),
                    actor_refreshes: Vec::new(),
                });
            }
        }
        if self.inner.children()?[bank].prepared.is_none() {
            if self.inner.children()?[bank].preparation_entered {
                return Err(invalid(
                    "original actor preparation callback cannot be repeated after uncertainty",
                ));
            }
            let stream = session
                .borrow(py)
                .owner()?
                .prepared_stream(&steps[0].borrow(py).inner)
                .map_err(xlog_err)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item(
                "steps",
                PyTuple::new(py, steps.iter().map(|step| step.clone_ref(py)))?,
            )?;
            kwargs.set_item("consumer_stream", stream.cu_stream() as u64)?;
            kwargs.set_item("memory_scope", memory.clone_ref(py))?;
            self.inner.children()?[bank].preparation_entered = true;
            let prepared = recording_callback(
                || self.inner.check_child(py, bank),
                || {
                    producer
                        .bind(py)
                        .getattr("prepare_segment")?
                        .call((task_use.clone_ref(py),), Some(&kwargs))
                },
            )?;
            if prepared.as_ptr() != original_prepared.as_ptr()
                || !prepared.getattr("memory_scope")?.is(memory.bind(py))
            {
                return Err(invalid(
                    "actor preparation must retain its original producer and borrow the sole outer memory scope",
                ));
            }
            self.inner.children()?[bank].prepared = Some(prepared.unbind());
        }
        let cold_work = self.inner.children()?[bank].cold_work.clone();
        if let Some(work) = cold_work {
            session
                .borrow(py)
                .owner()?
                .detach_shared_cold_native_work(&work)
                .map_err(xlog_err)?;
            let retained = self.inner.children()?[bank].cold_work.take();
            drop(retained);
        }
        let result = self.inner.children()?[bank]
            .prepared
            .as_ref()
            .expect("original prepared actor")
            .clone_ref(py);
        Ok(result)
    }
}

#[pymethods]
impl PySemanticPreparedStep {
    #[pyo3(signature = (member_ordinal, *, refresh_snapshot))]
    fn prepare_actor_refresh(
        slf: Py<Self>,
        py: Python<'_>,
        member_ordinal: &Bound<'_, PyAny>,
        refresh_snapshot: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PySemanticPreparedActorRefresh>> {
        let ordinal = ColdValue::read(member_ordinal, &mut 128, 0)?.unsigned()?;
        let update = slf.borrow(py);
        let session = update.session.borrow(py);
        session.require_creator()?;
        update.require_task(py, &update.task_use.borrow(py))?;
        let work = session
            .active_cold_model_work
            .lock()
            .map_err(|_| invalid("active cold callback custody mutex is poisoned"))?
            .as_ref()
            .map(|work| work.clone_ref(py))
            .ok_or_else(|| invalid("actor refresh requires its original Update cold callback"))?;
        let original = prepared_cold_callback_step(py, &session, &work.borrow(py))?
            .ok_or_else(|| invalid("actor refresh lost its original prepared cold work"))?;
        if original.as_ptr() != slf.as_ptr() {
            return Err(invalid(
                "actor refresh changed its original Update cold work",
            ));
        }
        let instruction = {
            let stored = session
                .prepared_segment
                .lock()
                .map_err(|_| invalid("prepared segment mutex is poisoned"))?;
            let stored = stored
                .as_ref()
                .ok_or_else(|| invalid("actor refresh requires its original outer segment"))?;
            if !Arc::ptr_eq(&stored.scope, &update.scope)
                || stored.actor_refreshes.iter().any(|original| {
                    original.member_ordinal == ordinal && original.update.as_ptr() == slf.as_ptr()
                })
            {
                return Err(invalid(
                    "actor refresh retains one original Update member attempt",
                ));
            }
            Arc::clone(stored.instruction.as_ref().ok_or_else(|| {
                invalid("actor refresh requires its original instruction admission")
            })?)
        };
        let source =
            replay_source_checkpoint(py, &update.task_use.borrow(py), ordinal, refresh_snapshot)?;
        let target = update.task_use.borrow(py);
        let roster = target.checkpoint.original_training_roster()?;
        let row = roster.fields(2)?[0]
            .sequence()?
            .get(
                usize::try_from(ordinal)
                    .map_err(|_| invalid("actor member ordinal exceeds this host"))?,
            )
            .ok_or_else(|| invalid("actor member is absent from the original roster"))?;
        let replay = ReplayRow::parse_with_live(row, &target.authority.live)?;
        let native_row = replay.training_view_row()?;
        let material = replay.native_replay()?;
        let (_, batch_identity, batch_bytes) = replay.actor_refresh_episode()?;
        let mut owner = session.owner()?;
        let position = instruction
            .preparation()?
            .native_steps
            .iter()
            .position(|step| step.same_handle(&update.inner))
            .ok_or_else(|| invalid("actor refresh changed its original native Update"))?;
        let program_ordinal = owner
            .admitted_segment_first_program_ordinal(&instruction.inner)
            .map_err(xlog_err)?
            .checked_add(position as u64)
            .ok_or_else(|| invalid("actor Program ordinal overflow"))?;
        let proof = owner
            .prepare_actor_refresh(
                &update.inner,
                &*instruction.parent.borrow(py).lease()?,
                program_ordinal,
                ordinal,
                &material.material,
                batch_identity,
                &batch_bytes,
            )
            .map_err(xlog_err)?;
        drop(owner);
        let custody = Arc::new(ActorRefreshCustody {
            update: slf.clone_ref(py),
            instruction,
            member_ordinal: ordinal,
            row: native_row,
            original_row: replay,
            replay: material,
            source,
            proof,
            children: Mutex::new(std::array::from_fn(|_| ActorRefreshChild::default())),
        });
        session
            .prepared_segment
            .lock()
            .map_err(|_| invalid("prepared segment mutex is poisoned"))?
            .as_mut()
            .ok_or_else(|| invalid("actor refresh lost its original outer segment"))?
            .actor_refreshes
            .push(Arc::clone(&custody));
        Py::new(py, PySemanticPreparedActorRefresh { inner: custody })
    }
}
