//! Projection of the native model-work registrar during the original cold
//! callback. It grants none of the prepared/evaluation numerical authorities.

use super::*;
use xlog_cuda::{SemanticColdModelWork, SemanticColdModelWorkRegion};

fn cold_region_ends(value: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if !value.is_exact_instance_of::<PyTuple>() {
        return Err(invalid("cold region ends require an exact immutable tuple"));
    }
    value
        .cast::<PyTuple>()?
        .iter()
        .map(|end| {
            usize::try_from(ColdValue::read(&end, &mut 128, 0)?.unsigned()?)
                .map_err(|_| invalid("cold region end exceeds the native event extent"))
        })
        .collect()
}

pub(in crate::semantic_transition) struct ColdModelWorkCustody {
    pub(in crate::semantic_transition) parent: Option<Py<PySemanticPublishedParent>>,
    pub(in crate::semantic_transition) reader: Py<PySemanticPublishedParent>,
    pub(in crate::semantic_transition) inner: SemanticColdModelWork,
    pub(in crate::semantic_transition) region: Option<SemanticColdModelWorkRegion>,
    #[cfg(feature = "semantic-policy")]
    future: Option<FrozenContextColdReader>,
    #[cfg(feature = "semantic-policy")]
    material_restore: Option<MaterialRestoreColdReader>,
}

#[cfg(feature = "semantic-policy")]
struct MaterialRestoreColdReader {
    context: xlog_cuda::SemanticModelContextContinuation,
    target_task: Py<PySemanticTransitionTaskUse>,
}

#[cfg(feature = "semantic-policy")]
struct FrozenContextColdReader {
    context: xlog_cuda::SemanticModelContextContinuation,
    step: SemanticPreparedStep,
    task_use: Py<PySemanticTransitionTaskUse>,
    scope: Arc<()>,
    activation: bool,
}

impl ColdModelWorkCustody {
    fn callback_session(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionSession>> {
        #[cfg(feature = "semantic-policy")]
        if self.future.is_some() {
            return Ok(self.session(py));
        }
        Ok(self.parent()?.borrow(py).session.clone_ref(py))
    }
    fn session(&self, py: Python<'_>) -> Py<PySemanticTransitionSession> {
        #[cfg(feature = "semantic-policy")]
        if let Some(future) = &self.future {
            return future.task_use.borrow(py).session.clone_ref(py);
        }
        self.reader.borrow(py).session.clone_ref(py)
    }

    #[cfg(feature = "semantic-policy")]
    fn require_future(&self, py: Python<'_>) -> PyResult<bool> {
        let Some(future) = &self.future else {
            return Ok(false);
        };
        let task = future.task_use.borrow(py);
        let session = task.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        task.require_identity(&owner)?;
        let phase = task.state()?.phase.clone();
        let phase_matches = if future.activation {
            matches!(&phase, TaskUsePhase::Prepared { scope, .. } | TaskUsePhase::ContextActivated { scope, .. } if Arc::ptr_eq(scope, &future.scope))
                && !session.recording.load(Ordering::Acquire)
        } else {
            session.recording.load(Ordering::Acquire)
                && matches!(&phase, TaskUsePhase::Recording { scope, .. } | TaskUsePhase::Prepared { scope, .. } if Arc::ptr_eq(scope, &future.scope))
        };
        if !task.frozen_model_context.load(Ordering::Acquire) || !phase_matches {
            return Err(invalid(
                "future cold callback changed its original recording scope",
            ));
        }
        owner
            .require_frozen_context_cold_model_work(&future.context, &future.step, &self.inner)
            .map_err(xlog_err)?;
        Ok(true)
    }
    pub(in crate::semantic_transition) fn parent(
        &self,
    ) -> PyResult<&Py<PySemanticPublishedParent>> {
        self.parent
            .as_ref()
            .ok_or_else(|| invalid("future cold region has no authenticated callback parent"))
    }
}

#[pyclass(name = "SemanticColdModelWork", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticColdModelWork {
    custody: Mutex<ColdModelWorkCustody>,
    #[cfg(feature = "semantic-policy")]
    checkpoint_writer: Mutex<Option<Py<material_bytes::PySemanticMaterialWriter>>>,
    #[cfg(feature = "semantic-policy")]
    checkpoint_writer_preparation: Mutex<Option<OriginalCheckpointWriterPreparation>>,
    pub(in crate::semantic_transition) active: AtomicBool,
}

#[cfg(feature = "semantic-policy")]
struct OriginalCheckpointWriterPreparation {
    selected_identity: usize,
    callback_identity: usize,
    selected: Option<Py<PyAny>>,
    callback: Option<Py<PyAny>>,
    result: Option<Py<material_bytes::PySemanticMaterialWriter>>,
    error: Option<PyErr>,
}

impl PySemanticColdModelWork {
    pub(in crate::semantic_transition) fn new(
        parent: Option<Py<PySemanticPublishedParent>>,
        reader: Py<PySemanticPublishedParent>,
        inner: SemanticColdModelWork,
        region: Option<SemanticColdModelWorkRegion>,
        active: bool,
    ) -> Self {
        Self {
            custody: Mutex::new(ColdModelWorkCustody {
                parent,
                reader,
                inner,
                region,
                #[cfg(feature = "semantic-policy")]
                future: None,
                #[cfg(feature = "semantic-policy")]
                material_restore: None,
            }),
            active: AtomicBool::new(active),
            #[cfg(feature = "semantic-policy")]
            checkpoint_writer: Mutex::new(None),
            #[cfg(feature = "semantic-policy")]
            checkpoint_writer_preparation: Mutex::new(None),
        }
    }

    pub(in crate::semantic_transition) fn custody(
        &self,
        py: Python<'_>,
    ) -> PyResult<ColdModelWorkCustody> {
        let original = self
            .custody
            .lock()
            .map_err(|_| invalid("cold callback custody mutex is poisoned"))?;
        Ok(ColdModelWorkCustody {
            parent: original.parent.as_ref().map(|parent| parent.clone_ref(py)),
            reader: original.reader.clone_ref(py),
            inner: original.inner.clone(),
            region: original.region.clone(),
            #[cfg(feature = "semantic-policy")]
            future: original
                .future
                .as_ref()
                .map(|future| FrozenContextColdReader {
                    context: future.context.clone(),
                    step: future.step.clone(),
                    task_use: future.task_use.clone_ref(py),
                    scope: Arc::clone(&future.scope),
                    activation: future.activation,
                }),
            #[cfg(feature = "semantic-policy")]
            material_restore: original.material_restore.as_ref().map(|restore| {
                MaterialRestoreColdReader {
                    context: restore.context.clone(),
                    target_task: restore.target_task.clone_ref(py),
                }
            }),
        })
    }

    #[cfg(feature = "semantic-policy")]
    pub(in crate::semantic_transition) fn new_model_context_restore(
        parent: Py<PySemanticPublishedParent>,
        inner: SemanticColdModelWork,
        region: SemanticColdModelWorkRegion,
        context: xlog_cuda::SemanticModelContextContinuation,
        target_task: Py<PySemanticTransitionTaskUse>,
    ) -> Self {
        let reader = Python::attach(|py| parent.clone_ref(py));
        let mut work = Self::new(Some(parent), reader, inner, Some(region), false);
        work.custody
            .get_mut()
            .expect("new cold custody is unpoisoned")
            .material_restore = Some(MaterialRestoreColdReader {
            context,
            target_task,
        });
        work
    }

    #[cfg(feature = "semantic-policy")]
    pub(in crate::semantic_transition) fn new_frozen_context(
        reader: Py<PySemanticPublishedParent>,
        inner: SemanticColdModelWork,
        context: xlog_cuda::SemanticModelContextContinuation,
        step: SemanticPreparedStep,
        task_use: Py<PySemanticTransitionTaskUse>,
        scope: Arc<()>,
    ) -> Self {
        Self::new_model_context(reader, inner, context, step, task_use, scope, None, false)
    }

    #[cfg(feature = "semantic-policy")]
    pub(in crate::semantic_transition) fn new_activation_context(
        reader: Py<PySemanticPublishedParent>,
        inner: SemanticColdModelWork,
        context: xlog_cuda::SemanticModelContextContinuation,
        step: SemanticPreparedStep,
        task_use: Py<PySemanticTransitionTaskUse>,
        scope: Arc<()>,
        region: SemanticColdModelWorkRegion,
    ) -> Self {
        Self::new_model_context(
            reader,
            inner,
            context,
            step,
            task_use,
            scope,
            Some(region),
            true,
        )
    }

    #[cfg(feature = "semantic-policy")]
    fn new_model_context(
        reader: Py<PySemanticPublishedParent>,
        inner: SemanticColdModelWork,
        context: xlog_cuda::SemanticModelContextContinuation,
        step: SemanticPreparedStep,
        task_use: Py<PySemanticTransitionTaskUse>,
        scope: Arc<()>,
        region: Option<SemanticColdModelWorkRegion>,
        activation: bool,
    ) -> Self {
        let mut work = Self::new(None, reader, inner, region, false);
        work.custody
            .get_mut()
            .expect("new cold custody is unpoisoned")
            .future = Some(FrozenContextColdReader {
            context,
            step,
            task_use,
            scope,
            activation,
        });
        work
    }

    pub(in crate::semantic_transition) fn bind_callback_parent(
        &self,
        py: Python<'_>,
        parent: &Py<PySemanticPublishedParent>,
    ) -> PyResult<()> {
        let original = self.custody(py)?;
        if let Some(bound) = &original.parent {
            if bound.as_ptr() != parent.as_ptr() {
                return Err(invalid("cold callback changed its original bound parent"));
            }
            return Ok(());
        }
        if self.active.load(Ordering::Acquire) {
            return Err(invalid("active cold callback cannot bind another parent"));
        }
        let reader = original.reader.borrow(py);
        let source = reader.session.borrow(py);
        source.require_creator()?;
        let acquired = parent.borrow(py);
        let callback = acquired.session.borrow(py);
        callback.require_creator()?;
        let owner = source.owner()?;
        if reader.session.as_ptr() == acquired.session.as_ptr() {
            owner.require_cold_model_work_callback_parent(
                &original.inner,
                &owner,
                &*acquired.lease()?,
            )
        } else {
            owner.require_cold_model_work_callback_parent(
                &original.inner,
                &*callback.owner()?,
                &*acquired.lease()?,
            )
        }
        .map_err(xlog_err)?;
        drop(owner);
        let mut custody = self
            .custody
            .lock()
            .map_err(|_| invalid("cold callback custody mutex is poisoned"))?;
        if !custody.inner.same_invocation(&original.inner) || custody.parent.is_some() {
            return Err(invalid(
                "future cold callback changed its original binding custody",
            ));
        }
        custody.parent = Some(parent.clone_ref(py));
        Ok(())
    }

    pub(in crate::semantic_transition) fn handoff_custody(
        &self,
        py: Python<'_>,
        parent: Py<PySemanticPublishedParent>,
        reader: Py<PySemanticPublishedParent>,
        inner: SemanticColdModelWork,
        region: Option<SemanticColdModelWorkRegion>,
    ) -> PyResult<()> {
        if self.active.load(Ordering::Acquire) {
            return Err(invalid(
                "active cold callback cannot transfer its original report",
            ));
        }
        let original = self.custody(py)?;
        let acquired = reader.borrow(py);
        let source = acquired.session.borrow(py);
        source.require_creator()?;
        let callback_parent = parent.borrow(py);
        let callback = callback_parent.session.borrow(py);
        callback.require_creator()?;
        let owner = source.owner()?;
        owner
            .require_cold_model_work_handoff(
                &original.inner,
                &inner,
                original.region.as_ref(),
                region.as_ref(),
            )
            .map_err(xlog_err)?;
        if acquired.session.as_ptr() == callback_parent.session.as_ptr() {
            owner.require_cold_model_work_callback_parent(
                &inner,
                &owner,
                &*callback_parent.lease()?,
            )
        } else {
            owner.require_cold_model_work_callback_parent(
                &inner,
                &*callback.owner()?,
                &*callback_parent.lease()?,
            )
        }
        .map_err(xlog_err)?;
        drop(owner);
        drop(callback);
        drop(source);
        drop(callback_parent);
        drop(acquired);
        let mut custody = self
            .custody
            .lock()
            .map_err(|_| invalid("cold callback custody mutex is poisoned"))?;
        if !custody.inner.same_invocation(&original.inner) {
            return Err(invalid(
                "cold handoff changed its original callback custody",
            ));
        }
        *custody = ColdModelWorkCustody {
            parent: Some(parent),
            reader,
            inner,
            region,
            #[cfg(feature = "semantic-policy")]
            future: None,
            #[cfg(feature = "semantic-policy")]
            material_restore: None,
        };
        Ok(())
    }

    pub(in crate::semantic_transition) fn check(&self, py: Python<'_>) -> PyResult<()> {
        self.check_callback(
            py,
            #[cfg(feature = "semantic-policy")]
            ColdCallbackAccess::Active,
        )
    }

    #[cfg(feature = "semantic-policy")]
    pub(in crate::semantic_transition) fn bind_activation_model_region(
        &self,
        py: Python<'_>,
    ) -> PyResult<()> {
        let original = self.custody(py)?;
        let Some(future) = original.future.as_ref().filter(|future| future.activation) else {
            return Ok(());
        };
        let session_handle = original.session(py);
        let region = session_handle
            .borrow(py)
            .owner()?
            .model_context_activation_model_region(&future.context, &original.inner)
            .map_err(xlog_err)?;
        let mut custody = self
            .custody
            .lock()
            .map_err(|_| invalid("activation model-region custody is poisoned"))?;
        if !custody.inner.same_invocation(&original.inner) {
            return Err(invalid("activation region changed its original report"));
        }
        if let Some(held) = &custody.region {
            if held.ordinal() != region.ordinal() {
                return Err(invalid("activation changed its original MODEL span"));
            }
        } else {
            custody.region = Some(region);
        }
        Ok(())
    }

    #[cfg(feature = "semantic-policy")]
    pub(in crate::semantic_transition) fn original_checkpoint_writer_port(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<material_bytes::PySemanticMaterialWriter>> {
        self.custody(py)?
            .reader
            .borrow(py)
            .session
            .borrow(py)
            .require_creator()?;
        self.checkpoint_writer
            .lock()
            .map_err(|_| invalid("cold checkpoint writer custody is poisoned"))?
            .as_ref()
            .map(|port| port.clone_ref(py))
            .ok_or_else(|| {
                invalid("cold checkpoint requires its original pre-admission writer port")
            })
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn prepare_original_checkpoint_writer(
        &self,
        py: Python<'_>,
        selected: &Py<PyAny>,
        callback: &Py<PyAny>,
        original_handle: &Py<Self>,
    ) -> PyResult<Py<material_bytes::PySemanticMaterialWriter>> {
        self.check(py)?;
        {
            let mut retained = self
                .checkpoint_writer_preparation
                .lock()
                .map_err(|_| invalid("checkpoint metadata preparation custody is poisoned"))?;
            if let Some(original) = retained.as_ref() {
                if original.selected_identity != selected.as_ptr() as usize
                    || original.callback_identity != callback.as_ptr() as usize
                {
                    return Err(invalid(
                        "checkpoint metadata preparation changed its original model or callback",
                    ));
                }
                if let Some(error) = &original.error {
                    return Err(error.clone_ref(py));
                }
                if let Some(writer) = &original.result {
                    return Ok(writer.clone_ref(py));
                }
                return Err(invalid(
                    "unknown checkpoint metadata preparation cannot repeat its original callback",
                ));
            }
            *retained = Some(OriginalCheckpointWriterPreparation {
                selected_identity: selected.as_ptr() as usize,
                callback_identity: callback.as_ptr() as usize,
                selected: Some(selected.clone_ref(py)),
                callback: Some(callback.clone_ref(py)),
                result: None,
                error: None,
            });
        }
        let returned = callback
            .bind(py)
            .call1((selected.clone_ref(py), original_handle.clone_ref(py)))
            .and_then(|returned| {
                if !returned.is_exact_instance_of::<material_bytes::PySemanticMaterialWriter>() {
                    return Err(invalid(
                        "checkpoint metadata preparation must return its original material writer",
                    ));
                }
                let writer = self.original_checkpoint_writer_port(py)?;
                if !returned.is(writer.bind(py).as_any()) {
                    return Err(invalid(
                        "checkpoint metadata preparation returned another writer object",
                    ));
                }
                let custody = self.custody(py)?;
                let reader = custody.reader.borrow(py);
                let session = reader.session.borrow(py);
                let budget = reader
                    .task_use
                    .borrow(py)
                    .checkpoint
                    .material_sources
                    .native_budget()?;
                let bounds = writer.borrow(py).original_cold_bounds()?;
                let owner = session.owner()?;
                writer.borrow(py).require_original_cold_port(
                    &budget,
                    &owner,
                    &custody.inner,
                    custody.region.as_ref(),
                    custody.region.as_ref().map_or(0, |region| region.ordinal()),
                    bounds.0,
                    bounds.1,
                )?;
                Ok(writer)
            });
        let mut retained = self
            .checkpoint_writer_preparation
            .lock()
            .map_err(|_| invalid("checkpoint metadata preparation custody is poisoned"))?;
        let original = retained
            .as_mut()
            .expect("retained original metadata preparation");
        match returned {
            Ok(writer) => {
                original.result = Some(writer.clone_ref(py));
                let completed = (original.selected.take(), original.callback.take());
                drop(retained);
                drop(completed);
                Ok(writer)
            }
            Err(error) => {
                original.error = Some(error.clone_ref(py));
                Err(error)
            }
        }
    }

    fn check_callback(
        &self,
        py: Python<'_>,
        #[cfg(feature = "semantic-policy")] access: ColdCallbackAccess,
    ) -> PyResult<()> {
        let custody = self.custody(py)?;
        #[cfg(feature = "semantic-policy")]
        if custody.future.is_some() {
            if !self.active.load(Ordering::Acquire) {
                return Err(invalid(
                    "future cold work requires its original active callback",
                ));
            }
            custody.require_future(py)?;
            return Ok(());
        }
        let parent = custody.parent()?.borrow(py);
        let source = parent.session.borrow(py);
        source.require_creator()?;
        if !self.active.load(Ordering::Acquire) {
            return Err(invalid(
                "cold model work is accessible only inside its original admitted callback",
            ));
        }
        #[cfg(feature = "semantic-policy")]
        if let Some(restore) = &custody.material_restore {
            let task = restore.target_task.borrow(py);
            if task.session.as_ptr() == parent.session.as_ptr() {
                return Err(invalid(
                    "material restore callback requires its distinct original target",
                ));
            }
            let target = task.session.borrow(py);
            target.require_creator()?;
            let source_owner = source.owner()?;
            let target_owner = target.owner()?;
            parent.task_use.borrow(py).require_current(&source_owner)?;
            task.require_identity(&target_owner)?;
            let region = custody.region.as_ref().ok_or_else(|| {
                invalid("material restore callback lost its original report region")
            })?;
            match access {
                ColdCallbackAccess::Active => source_owner
                    .require_model_context_material_restore_callback_work(
                        &target_owner,
                        &restore.context,
                        &*parent.lease()?,
                        &custody.inner,
                        region,
                    ),
                ColdCallbackAccess::OriginalCounter => source_owner
                    .require_model_context_material_restore_counter_read(
                        &target_owner,
                        &restore.context,
                        &*parent.lease()?,
                        &custody.inner,
                        region,
                    ),
            }
            .map_err(xlog_err)?;
            return Ok(());
        }
        #[cfg(feature = "semantic-policy")]
        if match access {
            ColdCallbackAccess::Active => require_admitted_cold_callback(py, &source, self)?,
            ColdCallbackAccess::OriginalCounter => {
                require_original_counter_cold_callback(py, &source, self)?
            }
        } {
            return Ok(());
        }
        #[cfg(feature = "semantic-policy")]
        if match access {
            ColdCallbackAccess::Active => prepared_cold_callback_step(py, &source, self)?,
            ColdCallbackAccess::OriginalCounter => {
                original_counter_cold_callback_step(py, &source, self)?
            }
        }
        .is_some()
        {
            return Ok(());
        }
        #[cfg(feature = "semantic-policy")]
        if match access {
            ColdCallbackAccess::Active => admitted_retirement_cold_callback(py, &source, self)?,
            ColdCallbackAccess::OriginalCounter => {
                original_counter_retirement_cold_callback(py, &source, self)?
            }
        }
        .is_some()
        {
            return Ok(());
        }
        if !source.learning_preparing.load(Ordering::Acquire) {
            return Err(invalid(
                "cold model work is accessible only inside its original admitted callback",
            ));
        }
        if matches!(&parent.task_use.borrow(py).state()?.phase,
            TaskUsePhase::CheckpointReading { original, .. } if matches!(original.as_ref(), TaskUsePhase::ArenaPreparing(_)))
        {
            return Ok(());
        }
        #[cfg(feature = "semantic-policy")]
        {
            let pending = source
                .learning_transition
                .lock()
                .map_err(|_| invalid("private cold callback lost its original phase mutex"))?
                .as_ref()
                .map(|pending| pending.clone_ref(py))
                .ok_or_else(|| {
                    invalid("private cold callback lost its original native phase owner")
                })?;
            if custody.region.is_some() {
                if pending
                    .borrow(py)
                    .require_terminal_refusal_cold_callback(py, self)?
                {
                    return Ok(());
                }
                if pending
                    .borrow(py)
                    .require_delivery_cold_callback(py, self)?
                    || pending
                        .borrow(py)
                        .require_intermediate_cold_callback(py, self)?
                {
                    return Ok(());
                }
                return pending
                    .borrow(py)
                    .require_evaluation_cold_callback(py, self);
            }
            return pending
                .borrow(py)
                .require_private_numeric_cold_callback(py, self);
        }
        #[cfg(not(feature = "semantic-policy"))]
        Err(invalid(
            "private numerical cold work requires semantic-policy",
        ))
    }

    #[cfg(feature = "semantic-policy")]
    pub(in crate::semantic_transition) fn issue_model_work_buffer_port(
        &self,
        py: Python<'_>,
        stream: u64,
    ) -> PyResult<xlog_cuda::SemanticModelWorkBufferPort> {
        self.check(py)?;
        let custody = self.custody(py)?;
        if let Some(future) = &custody.future {
            let session = custody.session(py);
            let result = session
                .borrow(py)
                .owner()?
                .frozen_context_cold_model_work_buffer_port(
                    &future.context,
                    &future.step,
                    &custody.inner,
                    stream,
                    custody.region.as_ref(),
                )
                .map_err(xlog_err);
            return result;
        }
        if let Some(restore) = &custody.material_restore {
            let parent = custody.reader.borrow(py);
            let source = parent.session.borrow(py);
            let task = restore.target_task.borrow(py);
            let target = task.session.borrow(py);
            let owner = source.owner()?;
            let target_owner = target.owner()?;
            return owner
                .model_context_material_restore_buffer_port(
                    &target_owner,
                    &restore.context,
                    &*parent.lease()?,
                    &custody.inner,
                    stream,
                    custody.region.as_ref().ok_or_else(|| {
                        invalid("material restore buffer lost its original report region")
                    })?,
                )
                .map_err(xlog_err);
        }
        let parent = custody.reader.borrow(py);
        let source = parent.session.borrow(py);
        source.require_creator()?;
        let prepared = original_counter_cold_callback_step(py, &source, self)?;
        let retirement = admitted_retirement_cold_callback(py, &source, self)?;
        let owner = source.owner()?;
        let result = if let Some(instruction) = retirement {
            owner.admitted_retirement_cold_model_work_buffer_port(
                &instruction.inner,
                &custody.inner,
                stream,
            )
        } else if let Some(step) = prepared {
            owner.prepared_cold_model_work_buffer_port(
                &step.borrow(py).inner,
                &*parent.lease()?,
                &custody.inner,
                stream,
                custody.region.as_ref(),
            )
        } else {
            owner.cold_model_work_buffer_port(
                &*parent.lease()?,
                &custody.inner,
                stream,
                custody.region.as_ref(),
            )
        }
        .map_err(xlog_err);
        result
    }

    #[cfg(feature = "semantic-policy")]
    pub(in crate::semantic_transition) fn export_model_work_buffer_port(
        &self,
        py: Python<'_>,
        port: &xlog_cuda::SemanticModelWorkBufferPort,
    ) -> PyResult<(
        xlog_cuda::SemanticModelWorkBufferExport,
        Py<PySemanticTransitionSession>,
    )> {
        self.check_callback(py, ColdCallbackAccess::OriginalCounter)?;
        let custody = self.custody(py)?;
        if let Some(future) = &custody.future {
            let session = custody.session(py);
            let export = session
                .borrow(py)
                .owner()?
                .export_frozen_context_cold_model_work_buffer_port(
                    &future.context,
                    &future.step,
                    port,
                    &custody.inner,
                    custody.region.as_ref(),
                )
                .map_err(xlog_err)?;
            return Ok((export, session));
        }
        if let Some(restore) = &custody.material_restore {
            let parent = custody.reader.borrow(py);
            let source = parent.session.borrow(py);
            let task = restore.target_task.borrow(py);
            let target = task.session.borrow(py);
            let mut owner = source.owner()?;
            let target_owner = target.owner()?;
            let export = owner
                .export_model_context_material_restore_buffer_port(
                    &target_owner,
                    &restore.context,
                    &*parent.lease()?,
                    port,
                    &custody.inner,
                    custody.region.as_ref().ok_or_else(|| {
                        invalid("material restore buffer lost its original report region")
                    })?,
                )
                .map_err(xlog_err)?;
            return Ok((export, parent.session.clone_ref(py)));
        }
        let parent = custody.reader.borrow(py);
        let source = parent.session.borrow(py);
        source.require_creator()?;
        let prepared = original_counter_cold_callback_step(py, &source, self)?;
        let retirement = original_counter_retirement_cold_callback(py, &source, self)?;
        let mut owner = source.owner()?;
        let result = if let Some(instruction) = retirement {
            owner.export_admitted_retirement_cold_model_work_buffer_port(
                &instruction.inner,
                port,
                &custody.inner,
            )
        } else if let Some(step) = prepared {
            owner.export_prepared_cold_model_work_buffer_port(
                &step.borrow(py).inner,
                &*parent.lease()?,
                port,
                &custody.inner,
                custody.region.as_ref(),
            )
        } else {
            owner.export_cold_model_work_buffer_port(
                &*parent.lease()?,
                port,
                &custody.inner,
                custody.region.as_ref(),
            )
        }
        .map_err(xlog_err)?;
        Ok((result, parent.session.clone_ref(py)))
    }

    #[cfg(feature = "semantic-policy")]
    pub(in crate::semantic_transition) fn finalized_model_work_extent(
        &self,
        py: Python<'_>,
        port: &xlog_cuda::SemanticModelWorkBufferPort,
    ) -> PyResult<usize> {
        self.check_callback(py, ColdCallbackAccess::OriginalCounter)?;
        let custody = self.custody(py)?;
        if let Some(future) = &custody.future {
            let session = custody.session(py);
            let result = session
                .borrow(py)
                .owner()?
                .frozen_context_cold_model_work_buffer_port_finalized_extent(
                    &future.context,
                    &future.step,
                    port,
                    &custody.inner,
                    custody.region.as_ref(),
                )
                .map_err(xlog_err);
            return result;
        }
        if let Some(restore) = &custody.material_restore {
            let parent = custody.reader.borrow(py);
            let source = parent.session.borrow(py);
            let task = restore.target_task.borrow(py);
            let target = task.session.borrow(py);
            let owner = source.owner()?;
            let target_owner = target.owner()?;
            return owner
                .model_context_material_restore_buffer_port_finalized_extent(
                    &target_owner,
                    &restore.context,
                    &*parent.lease()?,
                    port,
                    &custody.inner,
                    custody.region.as_ref().ok_or_else(|| {
                        invalid("material restore buffer lost its original report region")
                    })?,
                )
                .map_err(xlog_err);
        }
        let parent = custody.reader.borrow(py);
        let source = parent.session.borrow(py);
        source.require_creator()?;
        let prepared = original_counter_cold_callback_step(py, &source, self)?;
        let retirement = original_counter_retirement_cold_callback(py, &source, self)?;
        let owner = source.owner()?;
        let result = if let Some(instruction) = retirement {
            owner.admitted_retirement_cold_model_work_buffer_port_finalized_extent(
                &instruction.inner,
                port,
                &custody.inner,
            )
        } else if let Some(step) = prepared {
            owner.prepared_cold_model_work_buffer_port_finalized_extent(
                &step.borrow(py).inner,
                &*parent.lease()?,
                port,
                &custody.inner,
                custody.region.as_ref(),
            )
        } else {
            owner.cold_model_work_buffer_port_finalized_extent(
                &*parent.lease()?,
                port,
                &custody.inner,
                custody.region.as_ref(),
            )
        }
        .map_err(xlog_err);
        result
    }
}

#[pyclass(
    name = "SemanticColdModelWorkOperation",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticColdModelWorkOperation {
    work: Py<PySemanticColdModelWork>,
    inner: xlog_cuda::SemanticColdModelWorkOperation,
}

#[pymethods]
impl PySemanticColdModelWorkOperation {
    /// The original pre-initialized device slot, not actual expenditure.
    #[getter]
    fn slot(&self, py: Python<'_>) -> PyResult<usize> {
        let work = self.work.borrow(py);
        work.check(py)?;
        let custody = work.custody(py)?;
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        let result = session
            .owner()?
            .cold_model_work_operation_slot(&self.inner)
            .map_err(xlog_err);
        result
    }

    fn enter(&self, py: Python<'_>) -> PyResult<usize> {
        let work = self.work.borrow(py);
        work.check(py)?;
        let custody = work.custody(py)?;
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        let result = session
            .owner()?
            .enter_cold_model_work_operation(&self.inner)
            .map_err(xlog_err);
        result
    }

    fn complete(&self, py: Python<'_>) -> PyResult<usize> {
        let work = self.work.borrow(py);
        work.check(py)?;
        let custody = work.custody(py)?;
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        let result = session
            .owner()?
            .complete_cold_model_work_operation(&self.inner)
            .map_err(xlog_err);
        result
    }

    fn unknown(&self, py: Python<'_>) -> PyResult<()> {
        let work = self.work.borrow(py);
        work.check(py)?;
        let custody = work.custody(py)?;
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        let result = session
            .owner()?
            .retain_unknown_cold_model_work_operation(&self.inner)
            .map_err(xlog_err);
        result
    }

    fn cancel_before_entry(&self, py: Python<'_>) -> PyResult<()> {
        let work = self.work.borrow(py);
        work.check(py)?;
        let custody = work.custody(py)?;
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        let result = session
            .owner()?
            .cancel_unentered_cold_model_work_operation(&self.inner)
            .map_err(xlog_err);
        result
    }
}

#[pymethods]
impl PySemanticColdModelWork {
    /// Original intermediate-restoration transport, before its first snapshot.
    #[cfg(feature = "semantic-policy")]
    fn restore_material_mode(&self, py: Python<'_>) -> PyResult<&'static str> {
        let custody = self.custody(py)?;
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        session.require_creator()?;
        let phase = session
            .learning_transition
            .lock()
            .map_err(|_| invalid("restore metadata lost its original phase mutex"))?
            .as_ref()
            .map(|original| original.clone_ref(py))
            .ok_or_else(|| invalid("restore metadata requires its original phase owner"))?;
        let result = phase
            .borrow(py)
            .intermediate_restore_material_mode(py, self);
        result
    }

    #[cfg(feature = "semantic-policy")]
    fn prepare_resident_model_restoration(
        &self,
        py: Python<'_>,
        memory: &crate::semantic_transition::resident_model_memory::PySemanticPreparedResidentModelMemory,
    ) -> PyResult<Py<crate::semantic_transition::resident_model_memory::PySemanticPreparedResidentModelRestoration>>{
        let custody = self.custody(py)?;
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        session.require_creator()?;
        let phase = session
            .learning_transition
            .lock()
            .map_err(|_| invalid("restore metadata lost its original phase mutex"))?
            .as_ref()
            .map(|phase| phase.clone_ref(py))
            .ok_or_else(|| invalid("restore metadata requires its original phase owner"))?;
        let result = phase
            .borrow(py)
            .prepare_intermediate_resident_model_restoration(py, self, memory);
        result
    }

    /// Native declaration/physical-slot proof, retained after known completion.
    fn report_cursor(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let custody = self.custody(py)?;
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        session.require_creator()?;
        let cursor = if let Some(completed) = custody
            .inner
            .completed_report_cursor(custody.region.as_ref())
            .map_err(xlog_err)?
        {
            completed
        } else {
            let active = self.active.load(Ordering::Acquire);
            if active {
                #[cfg(feature = "semantic-policy")]
                if custody.material_restore.is_some() {
                    self.check_callback(py, ColdCallbackAccess::OriginalCounter)?;
                } else {
                    self.check(py)?;
                }
                #[cfg(not(feature = "semantic-policy"))]
                self.check(py)?;
            }
            let cursor = session
                .owner()?
                .cold_model_work_report_cursor(&custody.inner, custody.region.as_ref())
                .map_err(xlog_err)?;
            if !active && !cursor.recording_closed {
                return Err(invalid(
                    "cold cursor outside its callback requires native recording closure",
                ));
            }
            cursor
        };
        let skipped_spans = PyTuple::new(py, cursor.skipped_declared_spans.iter().copied())?;
        let skipped_regions = PyTuple::new(py, cursor.skipped_region_ordinals.iter().copied())?;
        Ok((
            cursor.region_begin,
            cursor.region_end,
            cursor.declared_cursor,
            cursor.recorded_slots,
            skipped_spans,
            skipped_regions,
            cursor.recording_closed,
            cursor.report_completed,
        )
            .into_pyobject(py)?
            .unbind())
    }

    #[pyo3(signature = (kind, dimensions, *, device_produced = false))]
    fn prepare_operation(
        slf: Py<Self>,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        dimensions: &Bound<'_, PyAny>,
        device_produced: bool,
    ) -> PyResult<Py<PySemanticColdModelWorkOperation>> {
        let work = slf.borrow(py);
        work.check(py)?;
        let custody = work.custody(py)?;
        if !dimensions.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "operation geometry requires its exact immutable tuple",
            ));
        }
        let (kind, values, rank) = parse_model_work(kind, dimensions)?;
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        let inner = session
            .owner()?
            .prepare_cold_model_work_operation(
                &custody.inner,
                custody.region.as_ref(),
                kind,
                &values[..rank],
                device_produced,
            )
            .map_err(xlog_err)?;
        Py::new(
            py,
            PySemanticColdModelWorkOperation {
                work: slf.clone_ref(py),
                inner,
            },
        )
    }

    fn prepare_operations(
        slf: Py<Self>,
        py: Python<'_>,
        operations: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let work = slf.borrow(py);
        work.check(py)?;
        let custody = work.custody(py)?;
        if !operations.is_exact_instance_of::<PyTuple>() {
            return Err(invalid("model operations require an exact immutable tuple"));
        }
        let mut parsed = Vec::new();
        for operation in operations.cast::<PyTuple>()?.iter() {
            if !operation.is_exact_instance_of::<PyTuple>() {
                return Err(invalid(
                    "model operation requires its exact geometry and production triple",
                ));
            }
            let operation = operation.cast::<PyTuple>()?;
            if operation.len() != 3
                || !operation.get_item(1)?.is_exact_instance_of::<PyTuple>()
                || !operation.get_item(2)?.is_exact_instance_of::<PyBool>()
            {
                return Err(invalid(
                    "model operation requires kind, immutable dimensions and exact bool",
                ));
            }
            let (kind, values, rank) =
                parse_model_work(&operation.get_item(0)?, &operation.get_item(1)?)?;
            parsed.push((
                kind,
                values[..rank].to_vec(),
                operation.get_item(2)?.extract::<bool>()?,
            ));
        }
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        let handles = session
            .owner()?
            .prepare_cold_model_work_operations(&custody.inner, custody.region.as_ref(), &parsed)
            .map_err(xlog_err)?;
        let handles = handles
            .into_iter()
            .map(|inner| {
                Py::new(
                    py,
                    PySemanticColdModelWorkOperation {
                        work: slf.clone_ref(py),
                        inner,
                    },
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, handles)?.unbind())
    }

    /// Original native plan state, not a Python callback invocation count.
    #[getter]
    fn plan_admitted(&self, py: Python<'_>) -> PyResult<bool> {
        self.check(py)?;
        let custody = self.custody(py)?;
        let result = custody
            .session(py)
            .borrow(py)
            .owner()?
            .cold_model_work_plan_is_admitted(&custody.inner)
            .map_err(xlog_err);
        result
    }

    /// Fixed native report geometry, not a callback invocation counter.
    #[getter]
    fn region_count(&self, py: Python<'_>) -> PyResult<usize> {
        self.check(py)?;
        let custody = self.custody(py)?;
        let result = custody
            .session(py)
            .borrow(py)
            .owner()?
            .cold_model_work_region_count(&custody.inner)
            .map_err(xlog_err);
        result
    }

    /// Seventeen original source-port layouts; no device export or selection.
    fn evaluation_source_geometry(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<(Vec<i64>, Vec<i64>, (u8, u8))>> {
        self.check(py)?;
        let custody = self.custody(py)?;
        let parent = custody.parent()?.borrow(py);
        let session = parent.session.borrow(py);
        let result = session
            .owner()?
            .evaluation_source_geometry(&*parent.lease()?)
            .map_err(xlog_err);
        result
    }

    /// Retain the complete physical operation recipe before original recording.
    /// Native returns (work bound, event count, call upper), not a Python estimate.
    #[pyo3(
        signature = (operations, *, region_ends = Vec::new(), evaluation_content = None, source_content = None, resident_reads = None),
        text_signature = "($self, operations, *, region_ends=(), evaluation_content=None, source_content=None, resident_reads=None)"
    )]
    fn admit_plan(
        &self,
        py: Python<'_>,
        operations: &Bound<'_, PyAny>,
        #[pyo3(from_py_with = cold_region_ends)] region_ends: Vec<usize>,
        evaluation_content: Option<&Bound<'_, PyAny>>,
        source_content: Option<&Bound<'_, PyAny>>,
        resident_reads: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(u64, u64, u64)> {
        self.check(py)?;
        let custody = self.custody(py)?;
        let mut reads = Vec::new();
        if let Some(values) = resident_reads {
            if !values.is_exact_instance_of::<PyTuple>() {
                return Err(invalid(
                    "resident read admission requires an exact immutable tuple",
                ));
            }
            let values = values.cast::<PyTuple>()?;
            reads
                .try_reserve_exact(values.len())
                .map_err(|_| invalid("resident read declaration reservation failed"))?;
            for entry in values.iter() {
                if !entry.is_exact_instance_of::<PyTuple>() || entry.cast::<PyTuple>()?.len() != 2 {
                    return Err(invalid(
                        "resident read admission requires original region and occurrence pairs",
                    ));
                }
                let entry = entry.cast::<PyTuple>()?;
                let ordinal =
                    usize::try_from(ColdValue::read(&entry.get_item(0)?, &mut 128, 0)?.unsigned()?)
                        .map_err(|_| {
                            invalid("resident read region exceeds the original native roster")
                        })?;
                let read = entry.get_item(1)?;
                if !read.is_exact_instance_of::<crate::semantic_transition::resident_model_memory::PySemanticResidentModelRead>() {
                    return Err(invalid("resident read admission requires its original native occurrence"));
                }
                let read = read.extract::<PyRef<
                    '_,
                    crate::semantic_transition::resident_model_memory::PySemanticResidentModelRead,
                >>()?;
                reads.push((ordinal, read.inner().clone()));
            }
        }
        let mut sources = Vec::new();
        if let Some(values) = source_content {
            if !values.is_exact_instance_of::<PyTuple>() {
                return Err(invalid(
                    "SOURCE admission requires an exact immutable recipe tuple",
                ));
            }
            for value in values.cast::<PyTuple>()?.iter() {
                let original = if value
                    .is_exact_instance_of::<prepared_import::PySemanticPreparedSourceContent>()
                {
                    let value = value
                        .extract::<PyRef<'_, prepared_import::PySemanticPreparedSourceContent>>()?;
                    value.require_original(py)?;
                    value.inner.clone()
                } else if value.is_exact_instance_of::<
                    source_content_geometry::PySemanticPreparedSourceContentGeometry,
                >() {
                    let value = value.extract::<PyRef<
                        '_,
                        source_content_geometry::PySemanticPreparedSourceContentGeometry,
                    >>()?;
                    value.require_original(py)?;
                    value
                        .native()
                        .bound_source_content_for_admission(&custody.inner)
                        .map_err(xlog_err)?
                } else {
                    return Err(invalid(
                        "SOURCE admission requires its original native recipe or historical geometry",
                    ));
                };
                sources.push(original);
            }
        }
        // Invoke original tensor producers before taking either native Session
        // mutex. The actual evaluation owner guards every producer callback.
        let (phase, content) = if let Some(value) = evaluation_content {
            let parent = custody.parent()?.borrow(py);
            let session = parent.session.borrow(py);
            let phase = private_execution_owner(py, &session)?.ok_or_else(|| {
                invalid("evaluation content requires its actual original phase owner")
            })?;
            let evaluation = phase
                .borrow(py)
                .evaluation_cold_content_owner(py, &custody.inner)?;
            let content = evaluation.borrow(py).prepare_cold_content(py, value)?;
            (Some(phase), Some(content))
        } else {
            (None, None)
        };
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        let mut owner = session.owner()?;
        let result = (|| {
            if !operations.is_exact_instance_of::<PyTuple>() {
                return Err(invalid(
                    "cold operation plan requires an exact immutable tuple",
                ));
            }
            let operations = operations.cast::<PyTuple>()?;
            let mut recipe = Vec::new();
            recipe
                .try_reserve_exact(operations.len())
                .map_err(|_| invalid("cold operation plan reservation failed"))?;
            for operation in operations.iter() {
                if !operation.is_exact_instance_of::<PyTuple>()
                    || operation.cast::<PyTuple>()?.len() != 2
                {
                    return Err(invalid(
                        "cold operation plan requires exact kind and geometry pairs",
                    ));
                }
                let operation = operation.cast::<PyTuple>()?;
                let kind = operation.get_item(0)?;
                let dimensions = operation.get_item(1)?;
                if !dimensions.is_exact_instance_of::<PyTuple>() {
                    return Err(invalid(
                        "cold operation geometry requires an exact immutable tuple",
                    ));
                }
                let (kind, values, rank) = parse_model_work(&kind, &dimensions)?;
                recipe.push((kind, values[..rank].to_vec()));
            }
            let quantities = if let Some(region) = &custody.region {
                #[cfg(feature = "semantic-policy")]
                let declaring_restore = custody.material_restore.is_some();
                #[cfg(not(feature = "semantic-policy"))]
                let declaring_restore = false;
                if declaring_restore {
                    owner.declare_cold_model_work_region_plan_with_sources(
                        region,
                        &recipe,
                        &region_ends,
                        content,
                        &sources,
                        &reads,
                    )
                } else {
                    owner.admit_cold_model_work_region_plan_with_sources(
                        region,
                        &recipe,
                        &region_ends,
                        content,
                        &sources,
                        &reads,
                    )
                }
            } else {
                owner.admit_cold_model_work_plan_with_sources(
                    &custody.inner,
                    &recipe,
                    &region_ends,
                    content,
                    &sources,
                    &reads,
                )
            }
            .map_err(xlog_err)?;
            Ok((quantities[0], quantities[1], quantities[2]))
        })();
        if result.is_err()
            && !owner
                .cold_model_work_plan_is_admitted(&custody.inner)
                .unwrap_or(false)
        {
            owner.fail_cold_model_work(&custody.inner);
        }
        drop(owner);
        #[cfg(feature = "semantic-policy")]
        if custody
            .future
            .as_ref()
            .is_some_and(|future| future.activation)
        {
            let admitted = session
                .owner()?
                .cold_model_work_plan_is_admitted(&custody.inner)
                .map_err(xlog_err)?;
            if admitted {
                self.bind_activation_model_region(py)?;
            }
        }
        let quantities = result?;
        if let Some(phase) = phase {
            if let Err(error) = phase
                .borrow(py)
                .admit_evaluation_cold_content(py, &custody.inner)
            {
                session.owner()?.fail_cold_model_work(&custody.inner);
                return Err(error);
            }
        }
        Ok(quantities)
    }

    #[getter]
    pub(in crate::semantic_transition) fn consumer_stream(&self, py: Python<'_>) -> PyResult<u64> {
        self.check(py)?;
        let custody = self.custody(py)?;
        let result = custody
            .session(py)
            .borrow(py)
            .owner()?
            .cold_model_work_stream(&custody.inner)
            .map_err(xlog_err);
        result
    }

    /// The original scratch owner before its sole plan determines the extent.
    #[cfg(feature = "semantic-policy")]
    #[pyo3(signature = (*, region_ordinal, model_material_capacity, model_material_working_bytes, resident_read, codec_staging_bytes))]
    fn prepare_checkpoint_writer(
        slf: Py<Self>,
        py: Python<'_>,
        region_ordinal: usize,
        model_material_capacity: usize,
        model_material_working_bytes: u64,
        resident_read: Py<super::super::resident_model_memory::PySemanticResidentModelRead>,
        codec_staging_bytes: usize,
    ) -> PyResult<Py<material_bytes::PySemanticMaterialWriter>> {
        let work = slf.borrow(py);
        let custody = work.custody(py)?;
        let reader = custody.reader.borrow(py);
        let source = reader.session.borrow(py);
        source.require_creator()?;
        let owner = source.owner()?;
        let count = owner
            .cold_model_work_region_count(&custody.inner)
            .map_err(xlog_err)?
            .max(1);
        if region_ordinal >= count {
            return Err(invalid(
                "checkpoint writer has no such original report region",
            ));
        }
        let budget = reader
            .task_use
            .borrow(py)
            .checkpoint
            .material_sources
            .native_budget()?;
        if custody
            .region
            .as_ref()
            .is_some_and(|region| region.ordinal() == region_ordinal)
            || (count == 1 && custody.region.is_none())
        {
            let cached = work
                .checkpoint_writer
                .lock()
                .map_err(|_| invalid("cold checkpoint writer custody is poisoned"))?;
            if let Some(writer) = &*cached {
                writer.borrow(py).require_original_readback(
                    py,
                    &resident_read,
                    codec_staging_bytes,
                )?;
                writer.borrow(py).require_original_cold_port(
                    &budget,
                    &owner,
                    &custody.inner,
                    custody.region.as_ref(),
                    region_ordinal,
                    model_material_capacity,
                    model_material_working_bytes,
                )?;
                return Ok(writer.clone_ref(py));
            }
        }
        let _roster_metadata = material_bytes::reserve_material_metadata(
            &budget,
            count
                .checked_mul(
                    2 * std::mem::size_of::<Py<Self>>()
                        + std::mem::size_of::<Option<SemanticColdModelWorkRegion>>(),
                )
                .ok_or_else(|| invalid("checkpoint writer roster metadata extent overflow"))?,
        )?;
        drop(owner);
        drop(source);
        drop(reader);
        let mut regions = None;
        if custody
            .region
            .as_ref()
            .is_some_and(|region| region.ordinal() == region_ordinal)
            || (count == 1 && custody.region.is_none())
        {
            regions = Some(vec![slf.clone_ref(py)]);
        } else {
            work.check(py)?;
            let callback = custody.parent()?.borrow(py);
            let callback_session = callback.session.borrow(py);
            if let Some(phase) = private_execution_owner(py, &callback_session)? {
                let phase = phase.borrow(py);
                regions = phase.delivery_cold_report_regions(py, &work)?;
                if regions.is_none() {
                    regions = phase.evaluation_cold_report_regions(py, &work)?;
                }
                if regions.is_none() {
                    regions = phase.intermediate_cold_report_regions(py, &work)?;
                }
            }
        }
        let regions = regions.unwrap_or_else(|| vec![slf.clone_ref(py)]);
        let selected_index = if regions.len() == 1 {
            0
        } else {
            region_ordinal
        };
        if regions.len() != 1 && regions.len() != count {
            return Err(invalid(
                "checkpoint writer changed its original complete report roster",
            ));
        }
        let selected = regions[selected_index].borrow(py);
        let original = selected.custody(py)?;
        if !original.inner.same_invocation(&custody.inner)
            || original
                .region
                .as_ref()
                .map_or(0, |region| region.ordinal())
                != region_ordinal
        {
            return Err(invalid(
                "checkpoint writer changed its original region owner",
            ));
        }
        let mut cached = selected
            .checkpoint_writer
            .lock()
            .map_err(|_| invalid("cold checkpoint writer custody is poisoned"))?;
        let reader = custody.reader.borrow(py);
        let source = reader.session.borrow(py);
        let owner = source.owner()?;
        if let Some(port) = &*cached {
            port.borrow(py)
                .require_original_readback(py, &resident_read, codec_staging_bytes)?;
            port.borrow(py).require_original_cold_port(
                &budget,
                &owner,
                &original.inner,
                original.region.as_ref(),
                region_ordinal,
                model_material_capacity,
                model_material_working_bytes,
            )?;
            return Ok(port.clone_ref(py));
        }
        owner
            .require_cold_checkpoint_writer_preparation(&custody.inner, region_ordinal)
            .map_err(xlog_err)?;
        let port = material_bytes::PySemanticMaterialWriter::prepare_original_cold_port(
            py,
            &budget,
            original.inner,
            original.region,
            region_ordinal,
            model_material_capacity,
            model_material_working_bytes,
            &owner,
            resident_read,
            codec_staging_bytes,
        )?;
        *cached = Some(port.clone_ref(py));
        Ok(port)
    }

    /// The original scratch owner before its sole plan determines the extent.
    #[cfg(feature = "semantic-policy")]
    #[pyo3(signature = (*, consumer_stream))]
    fn model_work_buffer_port(
        slf: Py<Self>,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<crate::semantic_transition::model_work_port::PySemanticModelWorkBufferPort>>
    {
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let port = slf.borrow(py).issue_model_work_buffer_port(py, stream)?;
        Py::new(
            py,
            crate::semantic_transition::model_work_port::PySemanticModelWorkBufferPort::new_cold(
                slf, port,
            ),
        )
    }

    /// Exact materialized scratch geometry; it grants no numerical access.
    #[pyo3(signature = (*, consumer_stream))]
    fn model_work_buffer_layout(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<(
        Py<PyNativeTensorAllocation>,
        Py<PyTuple>,
        Py<PyTuple>,
        (u8, u8),
    )> {
        self.check(py)?;
        let custody = self.custody(py)?;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        #[cfg(feature = "semantic-policy")]
        if let Some(future) = &custody.future {
            let session = custody.session(py);
            let (provenance, layout) = session
                .borrow(py)
                .owner()?
                .frozen_context_cold_model_work_buffer_layout(
                    &future.context,
                    &future.step,
                    &custody.inner,
                    stream,
                    custody.region.as_ref(),
                )
                .map_err(xlog_err)?;
            return Ok((
                Py::new(
                    py,
                    PyNativeTensorAllocation {
                        provenance,
                        policy_layout: None,
                    },
                )?,
                PyTuple::new(py, layout.dimensions[..2].iter().copied())?.unbind(),
                PyTuple::new(
                    py,
                    layout.strides_bytes[..2]
                        .iter()
                        .map(|stride| stride / layout.element_bytes),
                )?
                .unbind(),
                (1, 64),
            ));
        }
        let parent = custody.reader.borrow(py);
        let source = parent.session.borrow(py);
        #[cfg(feature = "semantic-policy")]
        let prepared = prepared_cold_callback_step(py, &source, self)?;
        #[cfg(feature = "semantic-policy")]
        let retirement = admitted_retirement_cold_callback(py, &source, self)?;
        let owner = source.owner()?;
        #[cfg(feature = "semantic-policy")]
        let (provenance, layout) = if let Some(instruction) = retirement {
            owner.admitted_retirement_cold_model_work_buffer_layout(
                &instruction.inner,
                &custody.inner,
                stream,
            )
        } else if let Some(step) = prepared {
            owner.prepared_cold_model_work_buffer_layout(
                &step.borrow(py).inner,
                &*parent.lease()?,
                &custody.inner,
                stream,
                custody.region.as_ref(),
            )
        } else {
            owner.cold_model_work_buffer_layout(
                &*parent.lease()?,
                &custody.inner,
                stream,
                custody.region.as_ref(),
            )
        }
        .map_err(xlog_err)?;
        #[cfg(not(feature = "semantic-policy"))]
        let (provenance, layout) = owner
            .cold_model_work_buffer_layout(
                &*parent.lease()?,
                &custody.inner,
                stream,
                custody.region.as_ref(),
            )
            .map_err(xlog_err)?;
        drop(owner);
        Ok((
            Py::new(
                py,
                PyNativeTensorAllocation {
                    provenance,
                    policy_layout: None,
                },
            )?,
            PyTuple::new(py, layout.dimensions[..2].iter().copied())?.unbind(),
            PyTuple::new(
                py,
                layout.strides_bytes[..2]
                    .iter()
                    .map(|stride| stride / layout.element_bytes),
            )?
            .unbind(),
            (1, 64),
        ))
    }

    /// Native U64[capacity,3] actual-work slots on the original CUDA stream.
    #[pyo3(signature = (*, consumer_stream))]
    fn model_work_buffer(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.check(py)?;
        let custody = self.custody(py)?;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        #[cfg(feature = "semantic-policy")]
        if let Some(future) = &custody.future {
            let session = custody.session(py);
            let owner_thread = session.borrow(py).owner_thread;
            let tensor = session
                .borrow(py)
                .owner()?
                .frozen_context_cold_model_work_buffer(
                    &future.context,
                    &future.step,
                    &custody.inner,
                    stream,
                    custody.region.as_ref(),
                )
                .map_err(xlog_err)?;
            return crate::dlpack_capsule_from_tensor(
                py,
                retain_export_owner(tensor, session, owner_thread)?,
            );
        }
        #[cfg(feature = "semantic-policy")]
        if let Some(restore) = &custody.material_restore {
            let parent = custody.reader.borrow(py);
            let source = parent.session.borrow(py);
            let task = restore.target_task.borrow(py);
            let target = task.session.borrow(py);
            let tensor = {
                let mut owner = source.owner()?;
                let target_owner = target.owner()?;
                owner
                    .model_context_material_restore_buffer(
                        &target_owner,
                        &restore.context,
                        &*parent.lease()?,
                        &custody.inner,
                        stream,
                        custody.region.as_ref().ok_or_else(|| {
                            invalid("material restore buffer lost its original report region")
                        })?,
                    )
                    .map_err(xlog_err)?
            };
            let owner_thread = source.owner_thread;
            drop(target);
            drop(task);
            drop(source);
            drop(parent);
            return crate::dlpack_capsule_from_tensor(
                py,
                retain_export_owner(tensor, custody.parent()?.clone_ref(py), owner_thread)?,
            );
        }
        let parent = custody.reader.borrow(py);
        let source = parent.session.borrow(py);
        #[cfg(feature = "semantic-policy")]
        let prepared = prepared_cold_callback_step(py, &source, self)?;
        #[cfg(feature = "semantic-policy")]
        let retirement = admitted_retirement_cold_callback(py, &source, self)?;
        let mut owner = source.owner()?;
        #[cfg(feature = "semantic-policy")]
        let tensor = if let Some(instruction) = retirement {
            owner.admitted_retirement_cold_model_work_buffer(
                &instruction.inner,
                &custody.inner,
                stream,
            )
        } else if let Some(step) = prepared {
            owner.prepared_cold_model_work_buffer(
                &step.borrow(py).inner,
                &*parent.lease()?,
                &custody.inner,
                stream,
                custody.region.as_ref(),
            )
        } else if let Some(region) = &custody.region {
            owner.cold_model_work_region_buffer(&*parent.lease()?, region, stream)
        } else {
            owner.cold_model_work_buffer(&*parent.lease()?, &custody.inner, stream)
        }
        .map_err(xlog_err)?;
        #[cfg(not(feature = "semantic-policy"))]
        let tensor = if let Some(region) = &custody.region {
            owner.cold_model_work_region_buffer(&*parent.lease()?, region, stream)
        } else {
            owner.cold_model_work_buffer(&*parent.lease()?, &custody.inner, stream)
        }
        .map_err(xlog_err)?;
        let tensor =
            retain_export_owner(tensor, custody.parent()?.clone_ref(py), source.owner_thread)?;
        crate::dlpack_capsule_from_tensor(py, tensor)
    }

    /// Attach the existing recorder once, even for a CPU/meta-only callback.
    fn begin(&self, py: Python<'_>) -> PyResult<()> {
        self.check(py)?;
        let custody = self.custody(py)?;
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        let mut owner = session.owner()?;
        if let Some(region) = &custody.region {
            owner.begin_cold_model_work_region(region)
        } else {
            owner.begin_cold_model_work(&custody.inner)
        }
        .map_err(xlog_err)
    }

    /// Close registration only. Callback return does not certify CUDA or
    /// allocator completion; the enclosing native operation joins those later.
    fn end(&self, py: Python<'_>) -> PyResult<()> {
        self.check(py)?;
        let custody = self.custody(py)?;
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        let mut owner = session.owner()?;
        if let Some(region) = &custody.region {
            owner.close_cold_model_work_region(region)
        } else {
            owner.close_cold_model_work(&custody.inner)
        }
        .map_err(xlog_err)
    }

    fn fail(&self, py: Python<'_>) -> PyResult<()> {
        self.check(py)?;
        let custody = self.custody(py)?;
        custody
            .session(py)
            .borrow(py)
            .owner()?
            .fail_cold_model_work(&custody.inner);
        Ok(())
    }

    fn record_model_invocation(&self, py: Python<'_>) -> PyResult<()> {
        self.check(py)?;
        let custody = self.custody(py)?;
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        let mut owner = session.owner()?;
        if let Some(region) = &custody.region {
            owner.record_cold_model_region_invocation(region)
        } else {
            owner.record_cold_model_invocation(&custody.inner)
        }
        .map_err(xlog_err)
    }

    fn record_model_work(
        &self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        dimensions: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.record(py, kind, dimensions, false).map(|_| ())
    }

    fn record_model_device_work(
        &self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        upper_dimensions: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        self.record(py, kind, upper_dimensions, true)
    }
}

impl PySemanticColdModelWork {
    fn record(
        &self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        dimensions: &Bound<'_, PyAny>,
        device_produced: bool,
    ) -> PyResult<usize> {
        self.check(py)?;
        let custody = self.custody(py)?;
        let session_handle = custody.session(py);
        let source = session_handle.borrow(py);
        let mut owner = source.owner()?;
        let result = (|| {
            let (kind, values, rank) = parse_model_work(kind, dimensions)?;
            if let Some(region) = &custody.region {
                owner.record_cold_model_work_region(region, kind, &values[..rank], device_produced)
            } else {
                owner.record_cold_model_work(&custody.inner, kind, &values[..rank], device_produced)
            }
            .map_err(xlog_err)
        })();
        if result.is_err() {
            owner.fail_cold_model_work(&custody.inner);
        }
        result
    }
}

/// Visibility belongs to the actual callback's Session, not the source phase's
/// first registrar. Private restored Sessions use the same callback boundary.
pub(in crate::semantic_transition) struct ColdCallbackScope<'a> {
    active: &'a AtomicBool,
    session: &'a PySemanticTransitionSession,
}

impl<'a> ColdCallbackScope<'a> {
    pub(in crate::semantic_transition) fn enter(
        py: Python<'_>,
        session: &'a PySemanticTransitionSession,
        work: &'a PySemanticColdModelWork,
        retained: Py<PySemanticColdModelWork>,
    ) -> PyResult<Self> {
        session.require_creator()?;
        let custody = work.custody(py)?;
        let session_handle = custody.callback_session(py)?;
        if !std::ptr::eq(&*retained.borrow(py), work)
            || !std::ptr::eq(&*session_handle.borrow(py), session)
        {
            return Err(invalid(
                "cold callback requires its actual original Session and registrar",
            ));
        }
        #[cfg(feature = "semantic-policy")]
        if custody.future.is_none() {
            custody.parent()?;
        }
        #[cfg(not(feature = "semantic-policy"))]
        custody.parent()?;
        let mut current = session
            .active_cold_model_work
            .lock()
            .map_err(|_| invalid("active cold callback custody mutex is poisoned"))?;
        if current.is_some() {
            return Err(invalid("another cold callback is already active"));
        }
        work.active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid("the original cold registrar is already active"))?;
        *current = Some(retained);
        drop(current);
        let scope = Self {
            active: &work.active,
            session,
        };
        work.check(py)?;
        Ok(scope)
    }
}

impl Drop for ColdCallbackScope<'_> {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
        let retained = self
            .session
            .active_cold_model_work
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        // Dropping visibility cannot release or complete the native work. Its
        // original operation retains the registrar and asynchronous storage.
        drop(retained);
    }
}

fn retained_work(
    py: Python<'_>,
    session: &PySemanticTransitionSession,
) -> PyResult<Py<PySemanticColdModelWork>> {
    session.require_creator()?;
    let work = session
        .active_cold_model_work
        .lock()
        .map_err(|_| invalid("active cold callback custody mutex is poisoned"))?
        .as_ref()
        .map(|work| work.clone_ref(py))
        .ok_or_else(|| invalid("there is no original admitted cold callback in this Session"))?;
    work.borrow(py).check(py)?;
    Ok(work)
}

impl PySemanticTransitionTaskUse {
    pub(in crate::semantic_transition) fn already_issued_cold_model_work(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<PySemanticColdModelWork>> {
        let issuing_session = self.session.borrow(py);
        issuing_session.require_creator()?;
        let pending = issuing_session
            .learning_transition
            .lock()
            .map_err(|_| invalid("cold metadata lost its original phase mutex"))?
            .as_ref()
            .map(|pending| pending.clone_ref(py));
        let active = if let Some(pending) = pending {
            let phase = pending.borrow(py);
            if let Some(work) = phase.already_issued_evaluation_cold_work(py, self)? {
                work
            } else {
                if phase.private_adoption_due()? {
                    return Err(invalid(
                        "cold metadata requires the original adoption report already issued through active_cold_model_work",
                    ));
                }
                retained_work(py, &issuing_session)?
            }
        } else {
            retained_work(py, &issuing_session)?
        };
        let work = active.borrow(py);
        let custody = work.custody(py)?;
        #[cfg(feature = "semantic-policy")]
        if let Some(future) = &custody.future {
            if !std::ptr::eq(&*future.task_use.borrow(py), self) {
                return Err(invalid(
                    "future cold metadata belongs to another original TaskUse",
                ));
            }
            return Ok(active.clone_ref(py));
        }
        let callback = custody.parent()?.borrow(py);
        if !std::ptr::eq(&*callback.task_use.borrow(py), self) {
            return Err(invalid("cold metadata belongs to another original TaskUse"));
        }
        drop(callback);
        drop(work);
        Ok(active)
    }
}

#[pymethods]
impl PySemanticTransitionTaskUse {
    /// Original complete registrar roster and one shared unmaterialized port.
    #[cfg(feature = "semantic-policy")]
    fn cold_model_work_report_layouts(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let active = self.already_issued_cold_model_work(py)?;
        let work = active.borrow(py);
        let custody = work.custody(py)?;
        let callback_session_handle = custody.callback_session(py)?;
        let callback_session = callback_session_handle.borrow(py);
        let mut regions = None;
        #[cfg(feature = "semantic-policy")]
        if let Some(phase) = private_execution_owner(py, &callback_session)? {
            let phase = phase.borrow(py);
            regions = phase.delivery_cold_report_regions(py, &work)?;
            if regions.is_none() {
                regions = phase.evaluation_cold_report_regions(py, &work)?;
            }
            if regions.is_none() {
                regions = phase.intermediate_cold_report_regions(py, &work)?;
            }
        }
        let regions = regions.unwrap_or_else(|| vec![active.clone_ref(py)]);
        let native_regions = regions
            .iter()
            .map(|region| {
                let original = region.borrow(py).custody(py)?;
                if !original.inner.same_invocation(&custody.inner) {
                    return Err(invalid(
                        "cold roster changed its original report invocation",
                    ));
                }
                Ok(original.region)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let session_handle = custody.session(py);
        let session = session_handle.borrow(py);
        let owner = session.owner()?;
        if custody.region.is_some() {
            let original = native_regions
                .into_iter()
                .map(|region| {
                    region.ok_or_else(|| {
                        invalid("segmented cold report lost its original region owner")
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            owner
                .require_cold_model_work_report_regions(&custody.inner, &original)
                .map_err(xlog_err)?;
        } else if regions.len() != 1
            || native_regions[0].is_some()
            || owner
                .cold_model_work_region_count(&custody.inner)
                .map_err(xlog_err)?
                != 0
        {
            return Err(invalid(
                "unsegmented cold callback changed its original report roster",
            ));
        }
        let stream = owner
            .cold_model_work_stream(&custody.inner)
            .map_err(xlog_err)?;
        drop(owner);
        let native_port = work.issue_model_work_buffer_port(py, stream)?;
        drop(work);
        let port = Py::new(
            py,
            crate::semantic_transition::model_work_port::PySemanticModelWorkBufferPort::new_cold(
                active.clone_ref(py),
                native_port,
            ),
        )?;
        let mut rows = Vec::with_capacity(regions.len());
        for region in regions {
            rows.push(
                (region, port.clone_ref(py), stream)
                    .into_pyobject(py)?
                    .unbind(),
            );
        }
        Ok(PyTuple::new(py, rows)?.unbind())
    }

    /// The original owner is available inside the callback, before prepare
    /// returns. Another task or a stale retained handle cannot borrow its rights.
    fn active_cold_model_work(&self, py: Python<'_>) -> PyResult<Py<PySemanticColdModelWork>> {
        #[cfg(feature = "semantic-policy")]
        {
            let source = self.session.borrow(py);
            let pending = source
                .learning_transition
                .lock()
                .map_err(|_| invalid("private cold callback lost its original phase mutex"))?
                .as_ref()
                .map(|pending| pending.clone_ref(py));
            if let Some(pending) = pending {
                if let Some(work) = pending.borrow(py).evaluation_cold_work(py, self)? {
                    return Ok(work);
                }
                if pending.borrow(py).private_adoption_due()? {
                    return pending.borrow(py).private_adoption_work(py, &source, self);
                }
            }
        }
        let work = retained_work(py, &self.session.borrow(py))?;
        #[cfg(feature = "semantic-policy")]
        if let Some(future) = &work.borrow(py).custody(py)?.future {
            if !std::ptr::eq(&*future.task_use.borrow(py), self) {
                return Err(invalid(
                    "future cold work belongs to another original task owner",
                ));
            }
            return Ok(work.clone_ref(py));
        }
        if !std::ptr::eq(
            &*work
                .borrow(py)
                .custody(py)?
                .parent()?
                .borrow(py)
                .task_use
                .borrow(py),
            self,
        ) {
            return Err(invalid(
                "cold model work belongs to another original task owner",
            ));
        }
        Ok(work)
    }
}

#[pymethods]
impl PySemanticTransitionController {
    fn active_cold_model_work(&self, py: Python<'_>) -> PyResult<Py<PySemanticColdModelWork>> {
        let work = retained_work(py, &self.session.borrow(py))?;
        #[cfg(feature = "semantic-policy")]
        if let Some(future) = &work.borrow(py).custody(py)?.future {
            self.require_read_issued(&future.task_use.borrow(py))?;
            if future.task_use.borrow(py).session.as_ptr() != self.session.as_ptr() {
                return Err(invalid(
                    "future cold callback belongs to another controller Session",
                ));
            }
            return Ok(work.clone_ref(py));
        }
        let task = work
            .borrow(py)
            .custody(py)?
            .parent()?
            .borrow(py)
            .task_use
            .clone_ref(py);
        self.require_issued(&task.borrow(py))?;
        let original = task.borrow(py).active_cold_model_work(py);
        original
    }
}
