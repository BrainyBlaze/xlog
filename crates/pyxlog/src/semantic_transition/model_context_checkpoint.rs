//! One checkpoint frame for the original dormant model context and its prefill.

use super::*;

const MAGIC: &[u8] = b"XLOG-MODEL-CONTEXT-CHECKPOINT\0";
const VERSION: u32 = 1;
const SECTION_COUNT: usize = 4;

pub(super) struct OriginalModelContextRestore {
    continuation: xlog_cuda::SemanticModelContextContinuation,
    target_task: Py<PySemanticTransitionTaskUse>,
    target_controller: Py<PySemanticTransitionController>,
    layout: SemanticInitialSourceLayout,
    work: Py<learning_phase::cold_model_work::PySemanticColdModelWork>,
    declaration: Py<
        model_context_continuation::declaration::PySemanticModelContextPrefillRestoreDeclaration,
    >,
    prepare_callback: Option<Py<PyAny>>,
    callback: Option<Py<PyAny>>,
    preparation_entered: bool,
    preparation: Option<Py<PyAny>>,
    preparation_error: Option<PyErr>,
    cap: Option<Py<model_context_continuation::PySemanticModelContextContinuation>>,
    proof: Option<
        Py<model_context_continuation::restored_prefill::PySemanticRestoredModelContextPrefill>,
    >,
    callback_entered: bool,
    result: Option<Py<PyAny>>,
    callback_error: Option<PyErr>,
    completed: bool,
    _metadata: [xlog_cuda::SemanticReplayMaterialReservation; 2],
}

impl OriginalModelContextRestore {
    pub(super) fn prepare(
        py: Python<'_>,
        checkpoint: &Py<PySemanticTransitionRestoredCheckpoint>,
        target_task: Py<PySemanticTransitionTaskUse>,
        native_source: &mut xlog_cuda::SemanticReplayLoadedMaterial,
        loaded: LoadedModelContextCheckpoint,
        prepare_callback: Py<PyAny>,
        callback: Py<PyAny>,
    ) -> PyResult<Self> {
        let restored = checkpoint.borrow(py);
        let source = restored.session.borrow(py);
        let target = target_task.borrow(py);
        let target_session = target.session.borrow(py);
        source.require_creator()?;
        target_session.require_creator()?;
        if !matches!(target.state()?.phase, TaskUsePhase::Imported)
            || target.session.as_ptr() == restored.session.as_ptr()
            || !Arc::ptr_eq(
                &target.checkpoint.material_sources,
                &restored.task_use.borrow(py).checkpoint.material_sources,
            )
        {
            return Err(invalid("model context restore requires its distinct fresh target in the original material owner"));
        }
        let budget = restored
            .task_use
            .borrow(py)
            .checkpoint
            .material_sources
            .native_budget()?;
        let parent = restored.parent.borrow(py);
        let lease = parent.lease()?;
        let mut source_owner = source.owner()?;
        let mut target_owner = target_session.owner()?;
        let continuation = source_owner
            .freeze_restored_model_context_continuation(
                &mut target_owner,
                &lease,
                native_source,
                &loaded.assignments,
                &loaded.mapping,
            )
            .map_err(xlog_err)?;
        let geometry = target_owner
            .prepare_model_context_checkpoint_attachment_restore(
                &continuation,
                &source_owner,
                &lease,
                native_source,
                loaded.target,
                loaded.prefill,
            )
            .map_err(xlog_err)?;
        // The canonical TaskUse layout check acquires its Target owner. It
        // must not be called while this same native mutex is already held.
        drop(target_owner);
        let layout = target.checked_initial_source_layout(
            py, geometry.0, geometry.1, geometry.2, geometry.3, geometry.4,
        )?;
        let mut target_owner = target_session.owner()?;
        let native_work = target_owner
            .prepare_model_context_checkpoint_attachment_restore_work(
                &continuation,
                &mut source_owner,
                &lease,
                layout.clone(),
            )
            .map_err(xlog_err)?;
        let region = source_owner
            .model_context_material_restore_callback_region(
                &target_owner,
                &continuation,
                &lease,
                &native_work,
            )
            .map_err(xlog_err)?;
        let native_declaration = target_owner
            .model_context_checkpoint_attachment_restore_declaration(
                &continuation,
                &source_owner,
                &lease,
            )
            .map_err(xlog_err)?;
        drop(target_owner);
        drop(source_owner);
        drop(lease);
        drop(parent);
        let work_metadata = material_bytes::reserve_python_material::<
            learning_phase::cold_model_work::PySemanticColdModelWork,
        >(py, &budget)?;
        let work = Py::new(
            py,
            learning_phase::cold_model_work::PySemanticColdModelWork::new_model_context_restore(
                restored.parent.clone_ref(py),
                native_work,
                region,
                continuation.clone(),
                target_task.clone_ref(py),
            ),
        )?;
        let controller_metadata =
            material_bytes::reserve_python_material::<PySemanticTransitionController>(py, &budget)?;
        let target_controller = Py::new(
            py,
            PySemanticTransitionController {
                session: target.session.clone_ref(py),
                identity: Arc::clone(&target.controller),
            },
        )?;
        drop(target_session);
        drop(target);
        drop(source);
        drop(restored);
        let declaration = model_context_continuation::declaration::PySemanticModelContextPrefillRestoreDeclaration::issue(
            py, checkpoint.clone_ref(py), continuation.clone(), target_task.clone_ref(py), native_declaration,
        )?;
        Ok(Self {
            continuation,
            target_task,
            target_controller,
            layout,
            work,
            declaration,
            prepare_callback: Some(prepare_callback),
            callback: Some(callback),
            preparation_entered: false,
            preparation: None,
            preparation_error: None,
            cap: None,
            proof: None,
            callback_entered: false,
            result: None,
            callback_error: None,
            completed: false,
            _metadata: [work_metadata, controller_metadata],
        })
    }
}

fn pending_error(
    py: Python<'_>,
    checkpoint: &Py<PySemanticTransitionRestoredCheckpoint>,
    original: PyErr,
) -> PyErr {
    let error = model_context_continuation::SemanticModelContextContinuationPending::new_err(
        "retain the original restored checkpoint and join its model context completion",
    );
    error.set_cause(py, Some(original));
    if let Err(metadata) = error
        .value(py)
        .setattr("restored_checkpoint", checkpoint.clone_ref(py))
    {
        metadata.set_cause(py, Some(error));
        return metadata;
    }
    error
}

fn finish_model_context_restore(
    py: Python<'_>,
    checkpoint: &Py<PySemanticTransitionRestoredCheckpoint>,
    continuation: &xlog_cuda::SemanticModelContextContinuation,
    target_task: &Py<PySemanticTransitionTaskUse>,
) -> PyResult<()> {
    let restored = checkpoint.borrow(py);
    {
        let source = restored.session.borrow(py);
        let target = target_task.borrow(py);
        let target_session = target.session.borrow(py);
        let mut source_owner = source.owner()?;
        let mut target_owner = target_session.owner()?;
        let parent = restored.parent.borrow(py);
        let lease = parent.lease()?;
        if let Err(error) = target_owner.finish_model_context_checkpoint_attachment_restore(
            continuation,
            &mut source_owner,
            &lease,
        ) {
            let pending = target_owner
                .model_context_continuation_pending(continuation)
                .map_err(xlog_err)?;
            drop(lease);
            drop(parent);
            drop(target_owner);
            drop(source_owner);
            let error = xlog_err(error);
            return Err(if pending {
                pending_error(py, checkpoint, error)
            } else {
                error
            });
        }
    }
    restored
        .model_context_restore
        .lock()
        .map_err(|_| invalid("original model context restore is poisoned"))?
        .as_mut()
        .expect("original model context restore")
        .completed = true;
    Ok(())
}

/// Each callback is entered once and outside every state/native mutex. Native
/// retries join the exact retained command/report rather than re-entering a
/// model factory or replaying the completed prefill.
pub(super) fn resolve_model_context_restore(
    py: Python<'_>,
    checkpoint: &Py<PySemanticTransitionRestoredCheckpoint>,
) -> PyResult<()> {
    let restored = checkpoint.borrow(py);
    restored.session.borrow(py).require_creator()?;
    let (
        continuation,
        target_task,
        target_controller,
        layout,
        work,
        declaration,
        factory_completed,
    ) = {
        let state = restored
            .model_context_restore
            .lock()
            .map_err(|_| invalid("original model context restore is poisoned"))?;
        let Some(state) = state.as_ref() else {
            return Ok(());
        };
        if state.completed {
            return Ok(());
        }
        if let Some(error) = state
            .preparation_error
            .as_ref()
            .or(state.callback_error.as_ref())
        {
            return Err(error.clone_ref(py));
        }
        (
            state.continuation.clone(),
            state.target_task.clone_ref(py),
            state.target_controller.clone_ref(py),
            state.layout.clone(),
            state.work.clone_ref(py),
            state.declaration.clone_ref(py),
            state.result.is_some(),
        )
    };
    if factory_completed {
        // The factory closed its sole region. Only the original report may
        // still be pending; entering an active callback would replay custody
        // checks against a region whose effects have already completed.
        return finish_model_context_restore(py, checkpoint, &continuation, &target_task);
    }
    let prepare_callback =
        {
            let mut state = restored
                .model_context_restore
                .lock()
                .map_err(|_| invalid("original model context restore is poisoned"))?;
            let state = state.as_mut().expect("original model context restore");
            if state.preparation.is_some() {
                None
            } else if state.preparation_entered {
                return Err(invalid(
                    "original model context compiler callback has unresolved completion",
                ));
            } else {
                state.preparation_entered = true;
                Some(state.prepare_callback.take().ok_or_else(|| {
                    invalid("original model context compiler callback disappeared")
                })?)
            }
        };
    if let Some(callback) = prepare_callback {
        let returned = (|| -> PyResult<Py<PyAny>> {
            let session = restored.session.borrow(py);
            let registrar = work.borrow(py);
            let _scope = learning_phase::cold_model_work::ColdCallbackScope::enter(
                py,
                &session,
                &registrar,
                work.clone_ref(py),
            )?;
            let returned = callback.bind(py).call1((
                checkpoint.clone_ref(py),
                declaration.clone_ref(py),
                work.clone_ref(py),
            ))?;
            if !returned.is_exact_instance_of::<PyTuple>() || returned.cast::<PyTuple>()?.len() != 2
            {
                return Err(invalid(
                    "model context preparation must return its original context and full geometry",
                ));
            }
            let fields = returned.cast::<PyTuple>()?;
            declaration
                .borrow(py)
                .bind_geometry(py, &fields.get_item(1)?)?;
            Ok(fields.get_item(0)?.unbind())
        })();
        let mut state = restored
            .model_context_restore
            .lock()
            .map_err(|_| invalid("original model context restore is poisoned"))?;
        let state = state.as_mut().expect("original model context restore");
        match returned {
            Ok(prepared) => state.preparation = Some(prepared),
            Err(error) => {
                state.preparation_error = Some(error.clone_ref(py));
                return Err(error);
            }
        }
    }
    let native = {
        let source = restored.session.borrow(py);
        let target = target_task.borrow(py);
        let target_session = target.session.borrow(py);
        let mut source_owner = source.owner()?;
        let mut target_owner = target_session.owner()?;
        let parent = restored.parent.borrow(py);
        let lease = parent.lease()?;
        match target_owner.restore_model_context_checkpoint_attachment(
            &continuation,
            &mut source_owner,
            &lease,
            layout,
        ) {
            Ok(proof) => proof,
            Err(error) => {
                let pending = target_owner
                    .model_context_continuation_pending(&continuation)
                    .map_err(xlog_err)?;
                drop(lease);
                drop(parent);
                drop(target_owner);
                drop(source_owner);
                let error = xlog_err(error);
                return Err(if pending {
                    pending_error(py, checkpoint, error)
                } else {
                    error
                });
            }
        }
    };
    declaration.borrow(py).require_bound(&native)?;
    let cap = {
        let existing = restored
            .model_context_restore
            .lock()
            .map_err(|_| invalid("original model context restore is poisoned"))?
            .as_ref()
            .and_then(|state| state.cap.as_ref().map(|cap| cap.clone_ref(py)));
        if let Some(cap) = existing {
            cap
        } else {
            let budget = restored
                .task_use
                .borrow(py)
                .checkpoint
                .material_sources
                .native_budget()?;
            let cap =
                model_context_continuation::PySemanticModelContextContinuation::issue_restored(
                    py,
                    continuation.clone(),
                    restored.controller.clone_ref(py),
                    restored.task_use.clone_ref(py),
                    restored.parent.clone_ref(py),
                    target_controller,
                    target_task.clone_ref(py),
                    restored.model.clone_ref(py),
                    &budget,
                )?;
            cap.borrow(py).retain_restored_prefill(py, native)?;
            let mut state = restored
                .model_context_restore
                .lock()
                .map_err(|_| invalid("original model context restore is poisoned"))?;
            state.as_mut().expect("original model context restore").cap = Some(cap.clone_ref(py));
            cap
        }
    };
    let proof = {
        let source = restored.session.borrow(py);
        let registrar = work.borrow(py);
        let _scope = learning_phase::cold_model_work::ColdCallbackScope::enter(
            py,
            &source,
            &registrar,
            work.clone_ref(py),
        )?;
        let stream = registrar.consumer_stream(py)?.into_pyobject(py)?.into_any();
        let (tensors, allocations) = restored
            .model_context_rosters
            .lock()
            .map_err(|_| invalid("restored model context roster custody is poisoned"))?
            .as_ref()
            .map(|(tensors, allocations)| (tensors.clone_ref(py), allocations.clone_ref(py)))
            .ok_or_else(|| invalid("restored model context lost its complete model rosters"))?;
        cap.borrow(py).bind_source_model_alias(
            py,
            restored.parent.clone_ref(py),
            restored.model.clone_ref(py),
            tensors.bind(py),
            allocations.bind(py),
            &stream,
        )?;
        cap.borrow(py).project_restored_prefill(py)?
    };
    let callback =
        {
            let mut state = restored
                .model_context_restore
                .lock()
                .map_err(|_| invalid("original model context restore is poisoned"))?;
            let state = state.as_mut().expect("original model context restore");
            if let Some(original) = &state.proof {
                if original.as_ptr() != proof.as_ptr() {
                    return Err(invalid("model context restore replaced its original proof"));
                }
            } else {
                state.proof = Some(proof.clone_ref(py));
            }
            if state.result.is_some() {
                None
            } else if state.callback_entered {
                return Err(invalid(
                    "original model context factory has unresolved completion",
                ));
            } else {
                state.callback_entered = true;
                Some(state.callback.take().ok_or_else(|| {
                    invalid("original model context factory callback disappeared")
                })?)
            }
        };
    if let Some(callback) = callback {
        let returned = (|| -> PyResult<Py<PyAny>> {
            let source = restored.session.borrow(py);
            let registrar = work.borrow(py);
            let _scope = learning_phase::cold_model_work::ColdCallbackScope::enter(
                py,
                &source,
                &registrar,
                work.clone_ref(py),
            )?;
            callback
                .bind(py)
                .call1((
                    checkpoint.clone_ref(py),
                    cap.clone_ref(py),
                    proof.clone_ref(py),
                ))
                .map(Bound::unbind)
        })();
        let mut state = restored
            .model_context_restore
            .lock()
            .map_err(|_| invalid("original model context restore is poisoned"))?;
        let state = state.as_mut().expect("original model context restore");
        match returned {
            Ok(result) => {
                if state
                    .preparation
                    .as_ref()
                    .is_none_or(|original| original.as_ptr() != result.as_ptr())
                {
                    let error =
                        invalid("model context factory replaced its original preparation owner");
                    state.callback_error = Some(error.clone_ref(py));
                    return Err(error);
                }
                // The successful factory returned the same context. Keep
                // that result once, not a second preparation reference.
                state.preparation = None;
                state.result = Some(result);
            }
            Err(error) => {
                // The Runtime must join a content command inside the same
                // recorded invocation. An unwound callback is never replayed.
                state.callback_error = Some(error.clone_ref(py));
                return Err(error);
            }
        }
    }
    finish_model_context_restore(py, checkpoint, &continuation, &target_task)
}

#[pymethods]
impl PySemanticTransitionRestoredCheckpoint {
    fn resolve_model_context_restore(slf: Py<Self>, py: Python<'_>) -> PyResult<()> {
        resolve_model_context_restore(py, &slf)
    }

    #[getter]
    fn model_context_restore_pending(&self, py: Python<'_>) -> PyResult<bool> {
        self.session.borrow(py).require_creator()?;
        Ok(self
            .model_context_restore
            .lock()
            .map_err(|_| invalid("original model context restore is poisoned"))?
            .as_ref()
            .is_some_and(|state| !state.completed))
    }

    #[getter]
    fn model_context_continuation(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<Py<model_context_continuation::PySemanticModelContextContinuation>>> {
        self.session.borrow(py).require_creator()?;
        Ok(self
            .model_context_restore
            .lock()
            .map_err(|_| invalid("original model context restore is poisoned"))?
            .as_ref()
            .and_then(|state| state.cap.as_ref().map(|cap| cap.clone_ref(py))))
    }

    #[getter]
    fn model_context_prefill(
        &self,
        py: Python<'_>,
    ) -> PyResult<
        Option<
            Py<model_context_continuation::restored_prefill::PySemanticRestoredModelContextPrefill>,
        >,
    > {
        self.session.borrow(py).require_creator()?;
        Ok(self
            .model_context_restore
            .lock()
            .map_err(|_| invalid("original model context restore is poisoned"))?
            .as_ref()
            .and_then(|state| state.proof.as_ref().map(|proof| proof.clone_ref(py))))
    }

    #[getter]
    fn model_context(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.session.borrow(py).require_creator()?;
        let state = self
            .model_context_restore
            .lock()
            .map_err(|_| invalid("original model context restore is poisoned"))?;
        state.as_ref().map(|state| {
            if !state.completed {
                return Err(model_context_continuation::SemanticModelContextContinuationPending::new_err("join the original Source report before using its restored context"));
            }
            state.result.as_ref().map(|result| result.clone_ref(py)).ok_or_else(|| invalid("known restore lost its original context factory result"))
        }).transpose()
    }
}

/// Retain native scopes and the actual target import, without retaining a
/// Python continuation-to-source-Session ownership cycle.
pub(super) struct CheckpointModelContextCustody {
    caller_identity: usize,
    continuation: xlog_cuda::SemanticModelContextContinuation,
    target_task: Py<PySemanticTransitionTaskUse>,
    attachment: Option<xlog_cuda::SemanticModelContextCheckpointAttachment>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

impl CheckpointModelContextCustody {
    pub(super) fn new(
        py: Python<'_>,
        cap: &Bound<'_, model_context_continuation::PySemanticModelContextContinuation>,
        source: &PySemanticTransitionController,
        task: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
    ) -> PyResult<(Box<Self>, usize)> {
        let original = cap.borrow();
        original.require_checkpoint_source(py, source, task, parent)?;
        let target_task = original.checkpoint_target(py)?;
        let target = target_task.borrow(py);
        let target_session = target.session.borrow(py);
        let source_session = source.session.borrow(py);
        let capacity = source_session
            .owner()?
            .model_context_checkpoint_attachment_allocation_bound(
                &original.inner,
                &*target_session.owner()?,
                &*parent.lease()?,
            )
            .map_err(xlog_err)?;
        let metadata =
            material_bytes::reserve_material_metadata(budget, std::mem::size_of::<Self>())?;
        drop(source_session);
        drop(target_session);
        drop(target);
        Ok((
            Box::new(Self {
                caller_identity: cap.as_ptr() as usize,
                continuation: original.inner.clone(),
                target_task,
                attachment: None,
                _metadata: metadata,
            }),
            capacity,
        ))
    }

    pub(super) fn require_request(&self, original: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        if original.is_none_or(|cap| cap.as_ptr() as usize != self.caller_identity) {
            return Err(invalid(
                "checkpoint continuation changed its original model context",
            ));
        }
        Ok(())
    }

    pub(super) fn resolve(
        &mut self,
        py: Python<'_>,
        source: &mut SemanticTransitionSession,
        parent: &SemanticPublishedLease,
        native_source: &mut xlog_cuda::SemanticReplayLoadedMaterial,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
    ) -> PyResult<xlog_cuda::SemanticReplayLoadedMaterial> {
        let target = self.target_task.borrow(py);
        let target_session = target.session.borrow(py);
        let mut target_owner = target_session.owner()?;
        if self.attachment.is_none() {
            let work = source
                .prepare_model_context_checkpoint_work(
                    &self.continuation,
                    &mut target_owner,
                    parent,
                )
                .map_err(xlog_err)?;
            self.attachment = Some(
                source
                    .stage_model_context_checkpoint_attachment(
                        &self.continuation,
                        &mut target_owner,
                        parent,
                        native_source,
                        &work,
                    )
                    .map_err(xlog_err)?,
            );
        }
        let materials = source
            .resolve_model_context_checkpoint_attachment(
                &self.continuation,
                &mut target_owner,
                parent,
                self.attachment
                    .as_mut()
                    .expect("original model context attachment"),
            )
            .map_err(xlog_err)?;
        LoadedModelContextCheckpoint::encode(materials, budget)
    }
}

/// All views retain the original signed checkpoint allocation. The native
/// decoders authenticate their complete Program, assignment and prefill basis.
pub(super) struct LoadedModelContextCheckpoint {
    pub(super) target: xlog_cuda::SemanticReplayLoadedMaterial,
    pub(super) mapping: xlog_cuda::SemanticReplayLoadedMaterial,
    pub(super) assignments: xlog_cuda::SemanticReplayLoadedMaterial,
    pub(super) prefill: xlog_cuda::SemanticReplayLoadedMaterial,
}

impl LoadedModelContextCheckpoint {
    pub(super) fn encode(
        materials: (
            xlog_cuda::SemanticReplayLoadedMaterial,
            xlog_cuda::SemanticReplayLoadedMaterial,
            xlog_cuda::SemanticReplayLoadedMaterial,
            xlog_cuda::SemanticReplayLoadedMaterial,
        ),
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
    ) -> PyResult<xlog_cuda::SemanticReplayLoadedMaterial> {
        let (target, mapping, assignments, prefill) = materials;
        let sections = [&target, &mapping, &assignments, &prefill];
        let extent =
            sections
                .iter()
                .try_fold(MAGIC.len() + 4 + SECTION_COUNT * 8, |total, material| {
                    material.require_budget(budget).map_err(xlog_err)?;
                    if material.as_bytes().is_empty() {
                        return Err(invalid("model context checkpoint lost an original section"));
                    }
                    total
                        .checked_add(material.as_bytes().len())
                        .ok_or_else(|| invalid("model context checkpoint extent overflow"))
                })?;
        let output = budget.prepare_material_output(extent).map_err(xlog_err)?;
        output
            .declare_spans(extent, &[(0, extent)])
            .map_err(xlog_err)?;
        let mut cursor = 0usize;
        let mut write = |part: &[u8]| -> PyResult<()> {
            output.write_model(cursor, part).map_err(xlog_err)?;
            cursor = cursor
                .checked_add(part.len())
                .ok_or_else(|| invalid("model context checkpoint cursor overflow"))?;
            Ok(())
        };
        write(MAGIC)?;
        write(&VERSION.to_le_bytes())?;
        for material in sections {
            let bytes = material.as_bytes();
            write(
                &u64::try_from(bytes.len())
                    .map_err(|_| invalid("model context section exceeds u64"))?
                    .to_le_bytes(),
            )?;
            write(bytes)?;
        }
        if cursor != extent {
            return Err(invalid(
                "model context checkpoint changed its original extent",
            ));
        }
        output.seal_model().map_err(xlog_err)
    }

    pub(super) fn decode(original: &xlog_cuda::SemanticReplayLoadedMaterial) -> PyResult<Self> {
        let raw = original.as_bytes();
        let header = MAGIC.len() + 4;
        if raw.len() < header + SECTION_COUNT * 8
            || &raw[..MAGIC.len()] != MAGIC
            || raw[MAGIC.len()..header] != VERSION.to_le_bytes()
        {
            return Err(invalid(
                "model context checkpoint has another domain or version",
            ));
        }
        let mut cursor = header;
        let mut read = || -> PyResult<xlog_cuda::SemanticReplayLoadedMaterial> {
            let length_end = cursor
                .checked_add(8)
                .filter(|end| *end <= raw.len())
                .ok_or_else(|| invalid("model context checkpoint length is incomplete"))?;
            let length = usize::try_from(u64::from_le_bytes(
                raw[cursor..length_end]
                    .try_into()
                    .expect("checked length extent"),
            ))
            .map_err(|_| invalid("model context checkpoint section exceeds this host"))?;
            let end = length_end
                .checked_add(length)
                .filter(|end| length != 0 && *end <= raw.len())
                .ok_or_else(|| invalid("model context checkpoint section is incomplete"))?;
            let view = original.slice_payload(length_end..end).map_err(xlog_err)?;
            cursor = end;
            Ok(view)
        };
        let decoded = Self {
            target: read()?,
            mapping: read()?,
            assignments: read()?,
            prefill: read()?,
        };
        if cursor != raw.len() {
            return Err(invalid(
                "model context checkpoint contains trailing material",
            ));
        }
        Ok(decoded)
    }
}
