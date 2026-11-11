use super::*;
use learning_phase::cold_model_work::{ColdCallbackScope, PySemanticColdModelWork};

#[derive(Clone, Copy, PartialEq, Eq)]
struct InitialPrefillGeometry {
    source: [u64; 5],
    model_generation: u64,
    memory_capacity_bytes: u64,
    other_external_cuda_bytes: u64,
}

pub(super) struct InitialPrefillPreparation {
    task_use: Py<PySemanticTransitionTaskUse>,
    scope: Arc<()>,
    geometry: InitialPrefillGeometry,
    layout: SemanticInitialSourceLayout,
    prepare_callback: Py<PyAny>,
    initialize_callback: Py<PyAny>,
    work: Option<Py<PySemanticColdModelWork>>,
    prepared: Option<Py<PyAny>>,
    stage: Option<Py<PySemanticInitialPrefillStage>>,
    result: Option<Py<PyAny>>,
    report: Option<xlog_cuda::SemanticColdModelWorkResult>,
    executing: bool,
    prepare_entered: bool,
    initialize_entered: bool,
    failed: bool,
}

impl TaskUseState {
    fn begin_initial_prefill_preparation(&mut self, scope: Arc<()>) -> PyResult<Arc<()>> {
        if !matches!(self.phase, TaskUsePhase::Imported) {
            return Err(invalid(
                "initial preparation requires its original imported task",
            ));
        }
        self.phase = TaskUsePhase::InitialPrefillPreparing {
            scope: Arc::clone(&scope),
        };
        Ok(scope)
    }

    fn begin_prepared_initial_prefill(&mut self, scope: &Arc<()>) -> PyResult<()> {
        if !matches!(&self.phase, TaskUsePhase::InitialPrefillPreparing { scope: original }
            if Arc::ptr_eq(original, scope))
        {
            return Err(invalid("initial preparation lost its original task scope"));
        }
        self.phase = TaskUsePhase::InitialPrefill {
            scope: Arc::clone(scope),
        };
        Ok(())
    }
}

impl PySemanticTransitionSession {
    pub(super) fn require_initial_prefill_idle(&self) -> PyResult<()> {
        let retained = self
            .initial_prefill_preparation
            .lock()
            .map_err(|_| invalid("initial prefill custody mutex is poisoned"))?;
        if let Some(original) = retained.as_ref() {
            if original
                .lock()
                .map_err(|_| invalid("initial prefill custody mutex is poisoned"))?
                .executing
            {
                return Err(invalid(
                    "terminal release cannot interrupt original initial preparation",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn require_initial_prefill_joined(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
    ) -> PyResult<()> {
        let retained = self
            .initial_prefill_preparation
            .lock()
            .map_err(|_| invalid("initial prefill custody mutex is poisoned"))?;
        if let Some(original) = retained.as_ref() {
            let original = original
                .lock()
                .map_err(|_| invalid("initial prefill custody mutex is poisoned"))?;
            if std::ptr::eq(&*original.task_use.borrow(py), task_use)
                && (original.report.is_none() || original.executing || original.failed)
            {
                return Err(invalid(
                    "parent binding requires the original completed initial preparation join",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn take_initial_prefill_preparation(
        &self,
        py: Python<'_>,
        issuance: Option<&TaskIssuance>,
    ) -> Option<Arc<Mutex<InitialPrefillPreparation>>> {
        {
            let mut retained = self
                .initial_prefill_preparation
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let same_task = retained.as_ref().is_some_and(|original| {
                let original = original
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .task_use
                    .borrow(py)
                    .issuance
                    .clone();
                issuance.is_none_or(|issuance| original.same_as(issuance))
            });
            if same_task {
                retained.take()
            } else {
                None
            }
        }
    }
}

#[pymethods]
impl PySemanticTransitionTaskUse {
    /// Prepare and initialize one original pre-parent model invocation. Retain
    /// both callback objects across Pending; a retry joins, never calls them again.
    #[pyo3(signature = (prepare_callback, initialize_callback, *, prefix_capacity, feedback_capacity, max_position, pad_token, provenance_capacity_records, model_generation, memory_capacity_bytes, other_external_cuda_bytes))]
    #[expect(
        clippy::too_many_arguments,
        reason = "the original prefill binds source, model and memory geometry"
    )]
    fn prepare_initial_prefill(
        slf: Py<Self>,
        py: Python<'_>,
        prepare_callback: Py<PyAny>,
        initialize_callback: Py<PyAny>,
        prefix_capacity: u64,
        feedback_capacity: u64,
        max_position: u64,
        pad_token: u64,
        provenance_capacity_records: u64,
        model_generation: u64,
        memory_capacity_bytes: u64,
        other_external_cuda_bytes: u64,
    ) -> PyResult<(Py<PySemanticInitialPrefillStage>, Py<PyAny>)> {
        if !prepare_callback.bind(py).is_callable() || !initialize_callback.bind(py).is_callable() {
            return Err(invalid(
                "initial prefill requires two original callable owners",
            ));
        }
        let issued = slf.borrow(py);
        let session = issued.session.borrow(py);
        session.require_creator()?;
        if issued.frozen_model_context.load(Ordering::Acquire) {
            return Err(invalid(
                "context initial prefill requires its original continuation",
            ));
        }
        issued.require_identity(&*session.owner()?)?;
        let geometry = InitialPrefillGeometry {
            source: [
                prefix_capacity,
                feedback_capacity,
                max_position,
                pad_token,
                provenance_capacity_records,
            ],
            model_generation,
            memory_capacity_bytes,
            other_external_cuda_bytes,
        };
        let existing = session
            .initial_prefill_preparation
            .lock()
            .map_err(|_| invalid("initial prefill custody mutex is poisoned"))?
            .as_ref()
            .map(Arc::clone);
        let preparation = if let Some(original) = existing {
            original
        } else {
            // Layout admission drains deferred export owners. Their Python
            // finalizers may reenter; never hold callback custody across it.
            let layout = issued.checked_initial_source_layout(
                py,
                prefix_capacity,
                feedback_capacity,
                max_position,
                pad_token,
                provenance_capacity_records,
            )?;
            if layout.prefill_tokens.is_empty() || model_generation == 0 {
                return Err(invalid(
                    "initial preparation requires a nonempty original prefix and model generation",
                ));
            }
            let mut original = session
                .initial_prefill_preparation
                .lock()
                .map_err(|_| invalid("initial prefill custody mutex is poisoned"))?;
            if original.is_some() {
                return Err(invalid(
                    "initial preparation was entered during source admission",
                ));
            }
            let scope = session.initial_model_scope(&issued)?.ok_or_else(|| {
                invalid("initial preparation requires its registered original model")
            })?;
            let scope = issued.state()?.begin_initial_prefill_preparation(scope)?;
            let retained = Arc::new(Mutex::new(InitialPrefillPreparation {
                task_use: slf.clone_ref(py),
                scope,
                geometry,
                layout,
                prepare_callback: prepare_callback.clone_ref(py),
                initialize_callback: initialize_callback.clone_ref(py),
                work: None,
                prepared: None,
                stage: None,
                result: None,
                report: None,
                executing: false,
                prepare_entered: false,
                initialize_entered: false,
                failed: false,
            }));
            *original = Some(Arc::clone(&retained));
            retained
        };
        let retained = || {
            preparation
                .lock()
                .map_err(|_| invalid("initial prefill custody mutex is poisoned"))
        };
        {
            let mut original = retained()?;
            if original.task_use.as_ptr() != slf.as_ptr()
                || original.geometry != geometry
                || original.prepare_callback.as_ptr() != prepare_callback.as_ptr()
                || original.initialize_callback.as_ptr() != initialize_callback.as_ptr()
                || original.failed
                || original.executing
                || (original.prepare_entered && original.prepared.is_none())
                || (original.initialize_entered && original.result.is_none())
            {
                return Err(invalid("retain the original initial prefill callbacks, arguments and returned owners without reentry"));
            }
            match &issued.state()?.phase {
                TaskUsePhase::InitialPrefillPreparing { scope }
                | TaskUsePhase::InitialPrefill { scope }
                | TaskUsePhase::InitialPrefillReady { scope }
                    if Arc::ptr_eq(scope, &original.scope) => {}
                _ => return Err(invalid("initial preparation lost its original task phase")),
            }
            original.executing = true;
        }
        let mut native_pending = false;
        let result = (|| {
            if retained()?.work.is_none() {
                let inner = session
                    .owner()?
                    .prepare_initial_prefill_cold_model_work(
                        memory_capacity_bytes,
                        other_external_cuda_bytes,
                    )
                    .map_err(xlog_err)?;
                let scope = Arc::clone(&retained()?.scope);
                let work = Py::new(
                    py,
                    PySemanticColdModelWork::new_initial_prefill(
                        issued.session.clone_ref(py),
                        slf.clone_ref(py),
                        scope,
                        inner,
                    ),
                )?;
                retained()?.work = Some(work);
            }
            let work = retained()?
                .work
                .as_ref()
                .map(|work| work.clone_ref(py))
                .ok_or_else(|| invalid("initial preparation lost its original native report"))?;
            if retained()?.prepared.is_none() {
                let work_ref = work.borrow(py);
                let _callback =
                    ColdCallbackScope::enter(py, &session, &work_ref, work.clone_ref(py))?;
                retained()?.prepare_entered = true;
                let prepared = prepare_callback
                    .bind(py)
                    .call1((work.clone_ref(py),))?
                    .unbind();
                retained()?.prepared = Some(prepared);
            }
            if retained()?.stage.is_none() {
                let (scope, layout) = {
                    let original = retained()?;
                    (Arc::clone(&original.scope), original.layout.clone())
                };
                let parts = initial_source_parts(py, layout.clone())?;
                issued.state()?.begin_prepared_initial_prefill(&scope)?;
                let stage = issue_initial_prefill_stage(
                    py,
                    slf.clone_ref(py),
                    scope,
                    layout,
                    parts,
                    model_generation,
                )?;
                retained()?.stage = Some(stage);
            }
            if retained()?.result.is_none() {
                let (stage, prepared) = {
                    let original = retained()?;
                    (
                        original
                            .stage
                            .as_ref()
                            .ok_or_else(|| invalid("initial prefill lost its stage"))?
                            .clone_ref(py),
                        original
                            .prepared
                            .as_ref()
                            .ok_or_else(|| invalid("initial prefill lost its preparation"))?
                            .clone_ref(py),
                    )
                };
                let work_ref = work.borrow(py);
                let _callback =
                    ColdCallbackScope::enter(py, &session, &work_ref, work.clone_ref(py))?;
                retained()?.initialize_entered = true;
                let result = initialize_callback
                    .bind(py)
                    .call1((stage, prepared))?
                    .unbind();
                retained()?.result = Some(result);
            }
            if retained()?.report.is_none() {
                let work_ref = work.borrow(py);
                let custody = work_ref.custody(py)?;
                let mut owner = session.owner()?;
                let stream = owner
                    .cold_model_work_stream(&custody.inner)
                    .map_err(xlog_err)?;
                match owner.finish_initial_prefill_cold_model_work(
                    &custody.inner,
                    &[stream],
                    xlog_cuda::SemanticColdModelWorkDisposition::Complete,
                ) {
                    Ok(report) => retained()?.report = Some(report),
                    Err(error) => {
                        native_pending = owner
                            .initial_prefill_cold_model_work_pending(&custody.inner)
                            .map_err(xlog_err)?;
                        let original = xlog_err(error);
                        if !native_pending {
                            return Err(original);
                        }
                        let pending = SemanticInitialPrefillPending::new_err(
                            "retain task_use and both original callbacks, then repeat prepare_initial_prefill with the unchanged arguments; do not repeat begin, reset or forward",
                        );
                        pending.value(py).setattr("task_use", slf.bind(py))?;
                        pending
                            .value(py)
                            .setattr("prepare_callback", prepare_callback.bind(py))?;
                        pending
                            .value(py)
                            .setattr("initialize_callback", initialize_callback.bind(py))?;
                        pending.set_cause(py, Some(original));
                        return Err(pending);
                    }
                }
            }
            let original = retained()?;
            Ok((
                original
                    .stage
                    .as_ref()
                    .ok_or_else(|| invalid("initial prefill has no original stage"))?
                    .clone_ref(py),
                original
                    .result
                    .as_ref()
                    .ok_or_else(|| invalid("initial prefill has no known original result"))?
                    .clone_ref(py),
            ))
        })();
        {
            let mut original = retained()?;
            original.executing = false;
            if result.is_err() && !native_pending {
                original.failed = true;
            }
        }
        if result.is_err() && !native_pending {
            // A callback or known native failure is terminal, not a permission
            // to repeat any effect. Custody survives until the original abort join.
            if let Ok(mut owner) = session.owner() {
                owner.abort();
            }
        }
        result
    }
}
