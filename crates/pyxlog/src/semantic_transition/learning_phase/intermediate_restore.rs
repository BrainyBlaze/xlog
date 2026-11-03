//! Original current-state restoration through the canonical full checkpoint path.

use super::phase_evaluation::EvaluationOwners;
use super::*;
use crate::semantic_transition::resident_model_memory::{
    PySemanticPreparedResidentModelMemory, PySemanticPreparedResidentModelRestoration,
};
use xlog_cuda::{
    SemanticColdModelWork, SemanticColdModelWorkRegion, SemanticColdModelWorkResult,
    SemanticColdNativeWork,
};

#[derive(Default)]
struct RestoreSnapshot {
    entered: bool,
    admitted: bool,
    saved: Option<Py<material_bytes::PySemanticMaterialBytes>>,
    serializer: Option<Py<PyAny>>,
    native_snapshot_bound: bool,
    error: Option<PyErr>,
}

pub(super) struct IntermediateRestore {
    previous: Option<Box<IntermediateRestore>>,
    branch: &'static str,
    owners: Option<EvaluationOwners>,
    retiring_parent: Py<PySemanticPublishedParent>,
    source_identity: Py<PyTuple>,
    entries: Py<PyTuple>,
    material: Vec<u8>,
    instruction: Vec<u8>,
    ordinal: u64,
    budget: [u64; 3],
    decode_entered: bool,
    decoded: Option<Py<PyAny>>,
    decode_verified: bool,
    recipe: Option<SemanticLearningPhaseTransition>,
    started: bool,
    ready: bool,
    work: Option<SemanticColdModelWork>,
    custody: Option<SemanticColdNativeWork>,
    regions: Vec<SemanticColdModelWorkRegion>,
    callbacks: [Option<Py<PySemanticColdModelWork>>; 4],
    resident_restoration_attempt: Option<xlog_cuda::SemanticPreparedResidentModelRestoration>,
    resident_restoration: Option<Py<PySemanticPreparedResidentModelRestoration>>,
    resident_prelude: Option<Arc<xlog_cuda::SemanticColdResidentRestorationPrelude>>,
    snapshots: [RestoreSnapshot; 2],
    retirement_entered: bool,
    retirement_completed: bool,
    session_release_entered: bool,
    session_released: bool,
    models_dropped: bool,
    restore_entered: bool,
    restored: Option<Py<PySemanticTransitionRestoredCheckpoint>>,
    error: Option<PyErr>,
    child_joined: bool,
    closed: bool,
    report: Option<SemanticColdModelWorkResult>,
    observer_finish_entered: bool,
    backing_peak: Option<u64>,
    record_entered: bool,
    recorded: bool,
    released: bool,
    budget_exceeded: bool,
    _recipe_metadata: Option<xlog_cuda::SemanticReplayMaterialReservation>,
}

impl PySemanticLearningPhaseTransition {
    pub(super) fn intermediate_restore_material_mode(
        &self,
        py: Python<'_>,
        work: &PySemanticColdModelWork,
    ) -> PyResult<&'static str> {
        self.require_intermediate_entry(py)?;
        let retained = self.intermediate()?;
        let current = retained
            .as_ref()
            .ok_or_else(|| invalid("restore metadata lost its original intermediate owner"))?;
        let custody = work.custody(py)?;
        if !current.decode_verified
            || !current.started
            || !current.ready
            || current.error.is_some()
            || current.closed
            || current
                .work
                .as_ref()
                .is_none_or(|original| !original.same_invocation(&custody.inner))
            || custody.published_reader()?.as_ptr() != self.parent.as_ptr()
            || !current
                .callbacks
                .iter()
                .flatten()
                .any(|original| std::ptr::eq(&*original.borrow(py), work))
        {
            return Err(invalid(
                "restore metadata requires its original decoded four-region registrar",
            ));
        }
        Ok(if current.recipe.is_some() {
            "phase-reset"
        } else {
            "checkpoint-model"
        })
    }

    pub(super) fn prepare_intermediate_resident_model_restoration(
        &self,
        py: Python<'_>,
        work: &PySemanticColdModelWork,
        memory: &PySemanticPreparedResidentModelMemory,
    ) -> PyResult<Py<PySemanticPreparedResidentModelRestoration>> {
        self.intermediate_restore_material_mode(py, work)?;
        let mut retained = self.intermediate()?;
        let current = retained.as_mut().expect("original intermediate restore");
        let original = current.work.as_ref().expect("original four-region report");
        if current.regions.len() != 4
            || current.callbacks[0]
                .as_ref()
                .is_none_or(|callback| !std::ptr::eq(&*callback.borrow(py), work))
            || current
                .owners
                .as_ref()
                .is_none_or(|owners| owners.parent.as_ptr() != memory.original_binding().0.as_ptr())
            || current
                .snapshots
                .iter()
                .any(|snapshot| snapshot.saved.is_some() || snapshot.error.is_some())
            || current.snapshots[1].entered
            || current.retirement_entered
            || current.session_release_entered
            || current.restore_entered
        {
            return Err(invalid(
                "resident restoration requires its original current snapshot carrier before retirement",
            ));
        }
        self.source
            .borrow(py)
            .owner()?
            .require_cold_model_work_resident_read_preparation(original)
            .map_err(xlog_err)?;
        if current
            .resident_restoration_attempt
            .as_ref()
            .is_some_and(|attempt| !attempt.same_source(memory.original_binding().1))
        {
            return Err(invalid(
                "resident restoration changed its original snapshot carrier",
            ));
        }
        if let Some(cached) = &current.resident_restoration {
            return Ok(cached.clone_ref(py));
        }
        if current.resident_restoration_attempt.is_none() {
            current.resident_restoration_attempt =
                Some(memory.prepare_original_restoration(py, original)?);
        }
        let task = current
            .owners
            .as_ref()
            .expect("original current-state owners")
            .task
            .borrow(py);
        let projection_extent = task.future_model_metadata_allocation_bound(py)?;
        let projected = PySemanticPreparedResidentModelRestoration::new(
            py,
            current
                .resident_restoration_attempt
                .as_ref()
                .expect("original resident restoration issuer")
                .clone(),
            original,
            projection_extent,
            || {
                Ok((
                    task.task_training_domain(py)?,
                    task.task_objective_law(py)?,
                    self.private_branch_feedback_projection(py, current.branch)?,
                ))
            },
        )?;
        drop(task);
        current.resident_restoration = Some(projected.clone_ref(py));
        Ok(projected)
    }

    pub(in crate::semantic_transition) fn require_intermediate_restored_model_metadata(
        &self,
        py: Python<'_>,
        task: &PySemanticTransitionTaskUse,
        parent: &Py<PySemanticPublishedParent>,
    ) -> PyResult<()> {
        let (planned, branch) = {
            let retained = self.intermediate()?;
            let Some(current) = retained.as_ref() else {
                return Ok(());
            };
            let Some(planned) = &current.resident_restoration else {
                return Ok(());
            };
            if !current.restore_entered
                || current.restored.is_some()
                || current
                    .resident_restoration_attempt
                    .as_ref()
                    .is_none_or(|issuer| !issuer.same_issuer(&planned.borrow(py).inner))
            {
                return Err(invalid(
                    "future model metadata lost its original pending restoration",
                ));
            }
            (planned.clone_ref(py), current.branch)
        };
        {
            let retained = self.private_restore()?;
            let owners = retained.as_ref().ok_or_else(|| {
                invalid("future model metadata lacks its original constructed child")
            })?;
            if owners
                .task_use
                .as_ref()
                .is_none_or(|original| !std::ptr::eq(&*original.borrow(py), task))
                || owners
                    .parent
                    .as_ref()
                    .is_none_or(|original| original.as_ptr() != parent.as_ptr())
            {
                return Err(invalid(
                    "future model metadata changed its actual child task or parent",
                ));
            }
        }
        let planned = planned.borrow(py);
        let budget = planned.material_budget(py);
        let _projection = budget
            .reserve_auxiliary(task.future_model_metadata_allocation_bound(py)?)
            .map_err(xlog_err)?;
        let result = planned.require_metadata(
            py,
            task.task_training_domain(py)?.bind(py),
            task.task_objective_law(py)?.bind(py),
            self.private_branch_feedback_projection(py, branch)?
                .bind(py),
        );
        result
    }

    pub(super) fn require_intermediate_cold_callback(
        &self,
        py: Python<'_>,
        work: &PySemanticColdModelWork,
    ) -> PyResult<bool> {
        let retained = self.intermediate()?;
        let Some(current) = retained.as_ref() else {
            return Ok(false);
        };
        let Some(index) = current.callbacks.iter().position(|original| {
            original
                .as_ref()
                .is_some_and(|original| std::ptr::eq(&*original.borrow(py), work))
        }) else {
            return Ok(false);
        };
        let custody = work.custody(py)?;
        let bound = custody.parent()?;
        let expected =
            match index {
                0 => {
                    current
                        .owners
                        .as_ref()
                        .is_some_and(|owners| owners.parent.as_ptr() == bound.as_ptr())
                        && current.snapshots[0].entered
                        && current.snapshots[0].saved.is_none()
                        && current.snapshots[0].error.is_none()
                }
                1 => {
                    current.retiring_parent.as_ptr() == bound.as_ptr()
                        && current.retirement_entered
                        && !current.retirement_completed
                }
                2 => {
                    current.restore_entered
                        && current.restored.is_none()
                        && self
                            .private_restore()?
                            .as_ref()
                            .and_then(|owners| owners.parent.as_ref())
                            .is_some_and(|parent| parent.as_ptr() == bound.as_ptr())
                }
                3 => {
                    current.restored.as_ref().is_some_and(|restored| {
                        restored.borrow(py).parent.as_ptr() == bound.as_ptr()
                    }) && current.snapshots[1].entered
                        && current.snapshots[1].saved.is_none()
                        && current.snapshots[1].error.is_none()
                }
                _ => false,
            };
        if !current.ready
            || current.closed
            || current.error.is_some()
            || !expected
            || custody.published_reader()?.as_ptr() != self.parent.as_ptr()
        {
            return Err(invalid(
                "intermediate cold callback changed its actual original region or owners",
            ));
        }
        Ok(true)
    }

    pub(super) fn intermediate_cold_report_regions(
        &self,
        py: Python<'_>,
        work: &PySemanticColdModelWork,
    ) -> PyResult<Option<Vec<Py<PySemanticColdModelWork>>>> {
        let retained = self.intermediate()?;
        let Some(current) = retained.as_ref() else {
            return Ok(None);
        };
        if !current
            .callbacks
            .iter()
            .flatten()
            .any(|region| std::ptr::eq(&*region.borrow(py), work))
        {
            return Ok(None);
        }
        current
            .callbacks
            .iter()
            .map(|region| {
                region
                    .as_ref()
                    .map(|region| region.clone_ref(py))
                    .ok_or_else(|| {
                        invalid("intermediate report lost its original complete registrar roster")
                    })
            })
            .collect::<PyResult<Vec<_>>>()
            .map(Some)
    }

    fn intermediate(&self) -> PyResult<MutexGuard<'_, Option<IntermediateRestore>>> {
        self.intermediate_restore
            .lock()
            .map_err(|_| invalid("original intermediate restoration custody mutex is poisoned"))
    }

    pub(super) fn intermediate_restore_pending(&self) -> PyResult<bool> {
        Ok(self
            .intermediate()?
            .as_ref()
            .is_some_and(|entry| !entry.recorded))
    }

    pub(super) fn intermediate_restore_retained(&self) -> PyResult<bool> {
        Ok(self.intermediate()?.is_some())
    }

    pub(super) fn intermediate_restore_unfinished(&self, branch: &'static str) -> PyResult<bool> {
        Ok(self
            .intermediate()?
            .as_ref()
            .is_some_and(|entry| entry.branch == branch && !entry.released))
    }

    fn require_intermediate_entry(&self, py: Python<'_>) -> PyResult<()> {
        self.records()?.require_preparation_admission()?;
        self.preparation_inputs
            .require_program(py, &self.scientific_owner)?;
        let retained = self.intermediate()?;
        let current = retained
            .as_ref()
            .ok_or_else(|| invalid("intermediate restore lost its original entry"))?;
        let (material, instruction) = Self::singleton_lifecycle_material(current.entries.bind(py))?;
        if material != current.material || instruction != current.instruction {
            return Err(invalid(
                "intermediate restore changed its complete frozen entry",
            ));
        }
        Ok(())
    }

    fn capture_intermediate_restore(&self, py: Python<'_>, branch: &'static str) -> PyResult<()> {
        if self.intermediate_restore_unfinished(branch)? {
            return self.require_intermediate_entry(py);
        }
        let (owners, ordinal) = self.private_current_restore_input(py, branch)?;
        let next = self.scientific_owner.bind(py).getattr("next_operation")?;
        if !next.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "intermediate restore requires its complete original scheduled entry",
            ));
        }
        let entries = next.cast::<PyTuple>()?;
        let (material, instruction) = Self::singleton_lifecycle_material(entries)?;
        let decoded = ColdValue::from_canonical_bytes(&material)?;
        let fields = decoded.fields(6)?;
        if fields[0].text()? != "restore"
            || fields[1].text()? != branch
            || fields[2].unsigned()? != ordinal
            || fields[3].unsigned()? != 0
            || fields[5] != ColdValue::None
        {
            return Err(invalid(
                "intermediate restore changed its original branch or contiguous position",
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
                "intermediate restore requires its original backing memory limit",
            ));
        }
        let source_identity = owners.parent.borrow(py).identity(py)?;
        let retiring_parent = owners.parent.clone_ref(py);
        let mut retained = self.intermediate()?;
        if retained.as_ref().is_some_and(|entry| {
            entry.branch != branch
                || !entry.recorded
                || !entry.released
                || entry.budget_exceeded
                || entry.error.is_some()
        }) {
            return Err(invalid(
                "intermediate restore cannot replace unknown or refused original work",
            ));
        }
        let previous = retained.take().map(Box::new);
        *retained = Some(IntermediateRestore {
            previous,
            branch,
            owners: Some(owners),
            retiring_parent,
            source_identity,
            entries: entries.clone().unbind(),
            material,
            instruction,
            ordinal,
            budget,
            decode_entered: false,
            decoded: None,
            decode_verified: false,
            recipe: None,
            started: false,
            ready: false,
            work: None,
            custody: None,
            regions: Vec::new(),
            callbacks: std::array::from_fn(|_| None),
            resident_restoration_attempt: None,
            resident_restoration: None,
            resident_prelude: None,
            snapshots: std::array::from_fn(|_| RestoreSnapshot::default()),
            retirement_entered: false,
            retirement_completed: false,
            session_release_entered: false,
            session_released: false,
            models_dropped: false,
            restore_entered: false,
            restored: None,
            error: None,
            child_joined: false,
            closed: false,
            report: None,
            observer_finish_entered: false,
            backing_peak: None,
            record_entered: false,
            recorded: false,
            released: false,
            budget_exceeded: false,
            _recipe_metadata: None,
        });
        Ok(())
    }

    fn decode_intermediate_restore(&self, py: Python<'_>) -> PyResult<()> {
        self.require_intermediate_entry(py)?;
        let (entered, decoded, entries, verified) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            (
                current.decode_entered,
                current.decoded.as_ref().map(|value| value.clone_ref(py)),
                current.entries.clone_ref(py),
                current.decode_verified,
            )
        };
        if verified {
            return Ok(());
        }
        let decoded = match decoded {
            Some(decoded) => decoded,
            None if entered => {
                return Err(invalid(
                    "unknown intermediate decoder cannot repeat its original callback",
                ))
            }
            None => {
                let decoder = self
                    .store()?
                    .as_ref()
                    .ok_or_else(|| {
                        invalid("intermediate restore lost its original lifecycle decoder")
                    })?
                    .decode
                    .clone_ref(py);
                self.intermediate()?
                    .as_mut()
                    .expect("original intermediate restore")
                    .decode_entered = true;
                let decoded = decoder
                    .bind(py)
                    .call1((entries.bind(py).get_item(0)?,))?
                    .unbind();
                self.intermediate()?
                    .as_mut()
                    .expect("original intermediate restore")
                    .decoded = Some(decoded.clone_ref(py));
                decoded
            }
        };
        if !decoded.bind(py).is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "intermediate restore requires its original closed decoder tuple",
            ));
        }
        let decoded = decoded.bind(py).cast::<PyTuple>()?;
        if decoded.len() != 3
            || ColdValue::read(&decoded.get_item(0)?, &mut 128, 0)?.text()? != "restore"
            || ColdValue::read(&decoded.get_item(1)?, &mut 128, 0)?.text()? != "current"
        {
            return Err(invalid(
                "intermediate restore must use the actual current complete checkpoint",
            ));
        }
        let recipe = decoded.get_item(2)?;
        let mut recipe_metadata = None;
        let recipe = if recipe.is_none() {
            None
        } else {
            let original = recipe.extract::<PyRef<'_, PySemanticLearningPhaseRecipe>>()?;
            if original.inner.admission.is_some() {
                return Err(invalid(
                    "intermediate recipe cannot replace its original signed admission",
                ));
            }
            let pool = self
                .task_use
                .borrow(py)
                .checkpoint
                .material_sources
                .native_budget()?;
            let extent = original
                .inner
                .recipe
                .len()
                .checked_mul(std::mem::size_of::<SemanticLearningCopyReset>())
                .ok_or_else(|| invalid("intermediate recipe metadata extent overflow"))?;
            recipe_metadata = Some(
                pool.reserve_auxiliary(u64::try_from(extent).map_err(|_| {
                    invalid("intermediate recipe metadata exceeds its native index")
                })?)
                .map_err(xlog_err)?,
            );
            let mut native = original.inner.clone();
            native.admission = Some(
                xlog_cuda::SemanticLearningPhaseAdmission::from_loaded(
                    self.records()?.confirmed_admission()?,
                    &pool,
                )
                .map_err(xlog_err)?,
            );
            Some(native)
        };
        let mut retained = self.intermediate()?;
        let current = retained.as_mut().expect("original intermediate restore");
        current.recipe = recipe;
        current._recipe_metadata = recipe_metadata;
        current.decode_verified = true;
        Ok(())
    }

    pub(super) fn require_intermediate_continuation(&self, py: Python<'_>) -> PyResult<()> {
        self.require_intermediate_entry(py)?;
        let retained = self.intermediate()?;
        let current = retained.as_ref().expect("original intermediate restore");
        if let Some(error) = &current.error {
            return Err(error.clone_ref(py));
        }
        if current.restore_entered && current.restored.is_none() {
            return Err(invalid(
                "unknown intermediate construction cannot repeat its original factory",
            ));
        }
        Ok(())
    }

    pub(super) fn require_intermediate_restore_admission(
        &self,
        py: Python<'_>,
        checkpoint: &[u8],
        transition: Option<&SemanticLearningPhaseTransition>,
    ) -> PyResult<()> {
        self.require_intermediate_entry(py)?;
        let retained = self.intermediate()?;
        let current = retained.as_ref().expect("original intermediate restore");
        if let Some(error) = &current.error {
            return Err(error.clone_ref(py));
        }
        let same_recipe = match (current.recipe.as_ref(), transition) {
            (None, None) => true,
            (Some(original), Some(actual)) => {
                original.source == actual.source
                    && original.target == actual.target
                    && original.phase_index == actual.phase_index
                    && original.completed_updates_index == actual.completed_updates_index
                    && original.recipe_digest() == actual.recipe_digest()
                    && original.admission == actual.admission
            }
            _ => false,
        };
        if !current.decode_verified
            || !current.ready
            || !current.session_released
            || !current.models_dropped
            || !current.restore_entered
            || current.restored.is_some()
            || !same_recipe
            || current.snapshots[0]
                .saved
                .as_ref()
                .is_none_or(|saved| saved.borrow(py).as_bytes() != checkpoint)
        {
            return Err(invalid(
                "intermediate allocation changed its actual checkpoint, recipe or known retirement",
            ));
        }
        Ok(())
    }

    pub(super) fn intermediate_restore_native_work(
        &self,
        py: Python<'_>,
    ) -> PyResult<SemanticColdNativeWork> {
        let work = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().ok_or_else(|| {
                invalid("intermediate allocation lost its original pre-allocation report")
            })?;
            if let Some(issuer) = &current.resident_restoration_attempt {
                let prelude = current.resident_prelude.as_ref().ok_or_else(|| {
                    invalid("intermediate allocation lost its original admitted native prelude")
                })?;
                prelude.require_restoration(issuer).map_err(xlog_err)?;
                return Ok(prelude.shared_native_work().clone());
            }
            current.work.clone().ok_or_else(|| {
                invalid("intermediate allocation lost its original pre-allocation report")
            })?
        };
        self.source
            .borrow(py)
            .owner()?
            .share_cold_native_work(&work)
            .map_err(xlog_err)
    }

    fn intermediate_callback(
        &self,
        py: Python<'_>,
        index: usize,
        parent: &Py<PySemanticPublishedParent>,
    ) -> PyResult<Py<PySemanticColdModelWork>> {
        let original = self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .callbacks
            .get(index)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                invalid("intermediate callback lost its original preallocated registrar")
            })?
            .clone_ref(py);
        original.borrow(py).bind_callback_parent(py, parent)?;
        Ok(original)
    }

    pub(super) fn intermediate_factory_work(
        &self,
        py: Python<'_>,
        parent: &Py<PySemanticPublishedParent>,
    ) -> PyResult<Py<PySemanticColdModelWork>> {
        let construction = self.private_restore()?;
        if construction
            .as_ref()
            .and_then(|owners| owners.parent.as_ref())
            .is_none_or(|original| original.as_ptr() != parent.as_ptr())
        {
            return Err(invalid(
                "intermediate factory changed its actual retained child parent",
            ));
        }
        drop(construction);
        let original = self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .callbacks[2]
            .as_ref()
            .expect("original factory registrar")
            .clone_ref(py);
        if original.borrow(py).custody(py)?.parent.is_some() {
            return Err(invalid(
                "intermediate factory cannot replace its original registrar",
            ));
        }
        self.intermediate_callback(py, 2, parent)
    }

    pub(super) fn intermediate_feedback_work(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<(&'static str, Py<PySemanticColdModelWork>)>> {
        let retained = self.intermediate()?;
        let Some(current) = retained.as_ref() else {
            return Ok(None);
        };
        let ordinal = if current.snapshots[1].entered
            && current.snapshots[1].saved.is_none()
            && current.snapshots[1].error.is_none()
        {
            3
        } else {
            2
        };
        let Some(work) = current.callbacks[ordinal].as_ref() else {
            return Ok(None);
        };
        if work.borrow(py).custody(py)?.parent.is_none() {
            return Ok(None);
        }
        Ok(Some((current.branch, work.clone_ref(py))))
    }

    pub(super) fn require_intermediate_snapshot_save(
        &self,
        py: Python<'_>,
        controller: &PySemanticTransitionController,
        task: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
    ) -> PyResult<()> {
        self.require_intermediate_entry(py)?;
        let mut retained = self.intermediate()?;
        let current = retained.as_mut().expect("original intermediate restore");
        let index = usize::from(current.session_released);
        let same = if index == 0 {
            let owners = current
                .owners
                .as_ref()
                .expect("original current-state owners");
            std::ptr::eq(controller, &*owners.controller.borrow(py))
                && std::ptr::eq(task, &*owners.task.borrow(py))
                && std::ptr::eq(parent, &*owners.parent.borrow(py))
        } else {
            let owners = current
                .restored
                .as_ref()
                .ok_or_else(|| invalid("restored snapshot lost its original complete owners"))?
                .borrow(py);
            let same = std::ptr::eq(controller, &*owners.controller.borrow(py))
                && std::ptr::eq(task, &*owners.task_use.borrow(py))
                && std::ptr::eq(parent, &*owners.parent.borrow(py));
            same
        };
        let snapshot = &mut current.snapshots[index];
        if !current.ready
            || !same
            || !snapshot.entered
            || snapshot.admitted
            || snapshot.saved.is_some()
            || snapshot.error.is_some()
            || !matches!(task.state()?.phase, TaskUsePhase::ArenaPreparing(_))
        {
            return Err(invalid(
                "intermediate snapshot changed or repeated its original save or selected owners",
            ));
        }
        snapshot.admitted = true;
        Ok(())
    }

    fn intermediate_snapshot(
        &self,
        py: Python<'_>,
        index: usize,
    ) -> PyResult<Py<material_bytes::PySemanticMaterialBytes>> {
        self.require_intermediate_entry(py)?;
        let saved = {
            let retained = self.intermediate()?;
            let snapshot = &retained
                .as_ref()
                .expect("original intermediate restore")
                .snapshots[index];
            if let Some(error) = &snapshot.error {
                return Err(error.clone_ref(py));
            }
            snapshot.saved.as_ref().map(|saved| saved.clone_ref(py))
        };
        if let Some(saved) = saved {
            if index == 0 {
                self.bind_intermediate_resident_snapshot(py)?;
            }
            return Ok(saved);
        }
        let (entered, owners) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            let snapshot = &current.snapshots[index];
            if let Some(error) = &snapshot.error {
                return Err(error.clone_ref(py));
            }
            let owners = if index == 0 {
                let original = current.owners.as_ref().expect("actual pre-restore owners");
                EvaluationOwners {
                    controller: original.controller.clone_ref(py),
                    task: original.task.clone_ref(py),
                    parent: original.parent.clone_ref(py),
                    model: original.model.clone_ref(py),
                }
            } else {
                let original = current
                    .restored
                    .as_ref()
                    .ok_or_else(|| invalid("post-restore save precedes its actual factory return"))?
                    .borrow(py);
                EvaluationOwners {
                    controller: original.controller.clone_ref(py),
                    task: original.task_use.clone_ref(py),
                    parent: original.parent.clone_ref(py),
                    model: original.model.clone_ref(py),
                }
            };
            (snapshot.entered, owners)
        };
        if entered {
            return Err(invalid(
                "unknown intermediate full snapshot cannot repeat its original serializer",
            ));
        }
        let work =
            self.intermediate_callback(py, if index == 0 { 0 } else { 3 }, &owners.parent)?;
        let writer = self.prepare_model_checkpoint_writer(py, &owners.model, &work)?;
        let (capacity, working_bytes) = writer.borrow(py).original_cold_bounds()?;
        let cached = self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .snapshots[index]
            .serializer
            .as_ref()
            .map(|serializer| serializer.clone_ref(py));
        let snapshot_model = if let Some(cached) = cached {
            cached
        } else {
            let serializer = self.model_owner(py, PhaseModelOwner::SerializeCandidate)?;
            let selected = owners.model.clone_ref(py);
            let session = owners.parent.borrow(py).session.clone_ref(py);
            let original_writer = writer.clone_ref(py);
            let callback =
                pyo3::types::PyCFunction::new_closure(py, None, None, move |args, kwargs| {
                    let py = args.py();
                    if args.len() != 1
                        || kwargs.is_some_and(|kwargs| !kwargs.is_empty())
                        || !args.get_item(0)?.is(original_writer.bind(py).as_any())
                    {
                        return Err(invalid(
                            "intermediate serializer requires its original writer object",
                        ));
                    }
                    let session = session.borrow(py);
                    let original = work.borrow(py);
                    let _scope =
                        ColdCallbackScope::enter(py, &session, &original, work.clone_ref(py))?;
                    original_writer
                        .borrow(py)
                        .bind_original_cold_callback(py, &work)?;
                    serializer
                        .bind(py)
                        .call1((selected.clone_ref(py), original_writer.clone_ref(py)))
                        .map(Bound::unbind)
                })?
                .into_any()
                .unbind();
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .snapshots[index]
                .serializer = Some(callback.clone_ref(py));
            callback
        };
        let fresh = self.refresh_snapshot.bind(py).call0()?;
        let authority =
            AuthoritySnapshot::parse(&ColdValue::read(&fresh, &mut (16 * 1024 * 1024), 0)?)?;
        {
            let issued = owners.task.borrow(py);
            authority.newer_than(&issued.state()?.snapshot)?;
            check_learning_grant(&issued, &self.grant_reference, &authority)?;
        }
        let streams = self.preparation_inputs.consumer_streams.python_value(py)?;
        self.intermediate()?
            .as_mut()
            .expect("original intermediate restore")
            .snapshots[index]
            .entered = true;
        let result = {
            let _operation = PhaseOperation::begin(&self.intermediate_snapshot_active)?;
            owners.controller.borrow(py).save_checkpoint(
                py,
                &owners.task.borrow(py),
                owners.parent.bind(py),
                streams.bind(py),
                &fresh,
                snapshot_model.bind(py),
                capacity,
                working_bytes,
                Some(writer.clone_ref(py)),
                // Both snapshots belong to private restored sessions without a future context.
                None,
            )
        };
        let mut retained = self.intermediate()?;
        let snapshot = &mut retained
            .as_mut()
            .expect("original intermediate restore")
            .snapshots[index];
        match result {
            Ok(saved) => {
                snapshot.saved = Some(saved.clone_ref(py));
                drop(retained);
                if index == 0 {
                    self.bind_intermediate_resident_snapshot(py)?;
                }
                Ok(saved)
            }
            Err(error) => {
                snapshot.error = Some(error.clone_ref(py));
                Err(error)
            }
        }
    }

    fn bind_intermediate_resident_snapshot(&self, py: Python<'_>) -> PyResult<()> {
        let (issuer, saved, parent, task) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            if current.snapshots[0].native_snapshot_bound {
                return Ok(());
            }
            let Some(issuer) = &current.resident_restoration_attempt else {
                return Ok(());
            };
            let owners = current.owners.as_ref().ok_or_else(|| {
                invalid("planned resident snapshot lost its original current-state owners")
            })?;
            (
                issuer.clone(),
                current.snapshots[0]
                    .saved
                    .as_ref()
                    .ok_or_else(|| {
                        invalid("planned resident restoration precedes its original saved snapshot")
                    })?
                    .clone_ref(py),
                owners.parent.clone_ref(py),
                owners.task.clone_ref(py),
            )
        };
        let saved = saved.borrow(py);
        let manifest = SemanticCheckpointManifest::decode_borrowed(saved.as_bytes())?;
        let mut material = saved
            .original_loaded(
                &task
                    .borrow(py)
                    .checkpoint
                    .material_sources
                    .native_budget()?,
            )?
            .slice_payload(checkpoint_section_range(saved.as_bytes(), manifest.native)?)
            .map_err(xlog_err)?;
        let proof = material.checkpoint_proof().map_err(xlog_err)?;
        let published = parent.borrow(py);
        let session = published.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        task.borrow(py).require_original_identity(&owner)?;
        issuer
            .bind_original_snapshot(&owner, &*published.lease()?, proof)
            .map_err(xlog_err)?;
        self.intermediate()?
            .as_mut()
            .expect("original intermediate restore")
            .snapshots[0]
            .native_snapshot_bound = true;
        Ok(())
    }

    fn require_intermediate_region_closed(&self, py: Python<'_>, index: usize) -> PyResult<()> {
        let region = self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .regions[index]
            .clone();
        self.source
            .borrow(py)
            .owner()?
            .require_closed_cold_model_work_region(&region)
            .map_err(xlog_err)
    }

    fn retire_intermediate_source(&self, py: Python<'_>) -> PyResult<()> {
        let (completed, parent) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            if current.session_released {
                return Ok(());
            }
            (
                current.retirement_completed,
                current.retiring_parent.clone_ref(py),
            )
        };
        let session = parent.borrow(py).session.clone_ref(py);
        let task = parent.borrow(py).task_use.clone_ref(py);
        if !completed {
            if self
                .intermediate()?
                .as_ref()
                .expect("original intermediate restore")
                .retirement_entered
            {
                return Err(invalid(
                    "unknown intermediate retirement cannot repeat its original model release",
                ));
            }
            let work = self.intermediate_callback(py, 1, &parent)?;
            let model = self
                .intermediate()?
                .as_ref()
                .expect("original intermediate restore")
                .owners
                .as_ref()
                .expect("actual current selected owners")
                .model
                .clone_ref(py);
            let fresh = self.refresh_snapshot.bind(py).call0()?;
            let authority =
                AuthoritySnapshot::parse(&ColdValue::read(&fresh, &mut (16 * 1024 * 1024), 0)?)?;
            {
                let issued = task.borrow(py);
                authority.newer_than(&issued.state()?.snapshot)?;
                check_learning_grant(&issued, &self.grant_reference, &authority)?;
            }
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .retirement_entered = true;
            let result = (|| {
                let published = parent.borrow(py);
                let original_session = session.borrow(py);
                let issued = task.borrow(py);
                let _reads =
                    ImportReadScope::checkpoint(&original_session, &issued, &published, py)?;
                let original = work.borrow(py);
                let _scope =
                    ColdCallbackScope::enter(py, &original_session, &original, work.clone_ref(py))?;
                let callback = self.model_owner(py, PhaseModelOwner::RetirePrivate)?;
                if !callback.bind(py).call1((model,))?.is_none() {
                    return Err(invalid("original intermediate model retirement must return None after known release"));
                }
                Ok(())
            })();
            if let Err(error) = result {
                self.intermediate()?
                    .as_mut()
                    .expect("original intermediate restore")
                    .error = Some(error.clone_ref(py));
                return Err(error);
            }
            self.require_intermediate_region_closed(py, 1)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .retirement_completed = true;
        }
        if self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .session_release_entered
        {
            return Err(invalid(
                "unknown intermediate native deallocation retains its original terminal custody",
            ));
        }
        session
            .borrow(py)
            .owner()?
            .require_retired_publication(&*parent.borrow(py).lease()?)
            .map_err(xlog_err)?;
        let custody = self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .custody
            .clone();
        if let Some(custody) = custody {
            session
                .borrow(py)
                .owner()?
                .detach_shared_cold_native_work(&custody)
                .map_err(xlog_err)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .custody = None;
            drop(custody);
        }
        if !self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .models_dropped
        {
            let branch = self
                .intermediate()?
                .as_ref()
                .expect("original intermediate restore")
                .branch;
            self.drop_completed_private_execution_owners(branch)?;
            self.drop_completed_trajectory_model_references()?;
            let construction = {
                let mut retained = self.private_restore()?;
                if retained
                    .as_ref()
                    .is_none_or(|owners| owners.session.as_ptr() != session.as_ptr())
                {
                    return Err(invalid(
                        "intermediate retirement cannot discard another private construction",
                    ));
                }
                retained.take()
            };
            drop(construction);
            let original = {
                let mut retained = self.intermediate()?;
                let current = retained.as_mut().expect("original intermediate restore");
                (
                    current.owners.take(),
                    current.previous.take(),
                    current.callbacks[0].take(),
                    current.callbacks[1].take(),
                )
            };
            drop(original);
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .models_dropped = true;
        }
        self.intermediate()?
            .as_mut()
            .expect("original intermediate restore")
            .session_release_entered = true;
        if let Err(error) = session
            .borrow(py)
            .release_retired_publication(py, &parent.borrow(py))
        {
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .error = Some(error.clone_ref(py));
            return Err(error);
        }
        task.borrow(py).state()?.phase = TaskUsePhase::Refused;
        {
            let original = session.borrow(py);
            original.learning_preparing.store(false, Ordering::Release);
            let retained = original
                .learning_transition
                .lock()
                .map_err(|_| invalid("retired intermediate Session lost its phase mutex"))?
                .take();
            drop(retained);
        }
        self.intermediate()?
            .as_mut()
            .expect("original intermediate restore")
            .session_released = true;
        Ok(())
    }

    pub(super) fn execute_intermediate_restore(
        &self,
        py: Python<'_>,
        pending: &Py<Self>,
        branch: &'static str,
    ) -> PyResult<()> {
        self.capture_intermediate_restore(py, branch)?;
        self.decode_intermediate_restore(py)?;
        self.require_intermediate_continuation(py)?;
        let completed_native = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            if current.closed {
                current.restored.as_ref().map(|value| value.clone_ref(py))
            } else {
                None
            }
        };
        if let Some(restored) = completed_native {
            // Original native Closed/Submitted/Completed and observer readback
            // are the only remaining work; never re-enter a sealed region.
            return self.finish_intermediate_restore(py, &restored);
        }
        let (started, ready, ordinal) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            (current.started, current.ready, current.ordinal)
        };
        if !started {
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .started = true;
            self.preparation_inputs
                .resource_observer
                .begin(py, ordinal)?;
            let work = self
                .source
                .borrow(py)
                .owner()?
                .prepare_cold_model_work_deferred(
                    &*self.parent.borrow(py).lease()?,
                    ordinal,
                    self.records()?.confirmed_admission()?,
                    xlog_cuda::SemanticColdModelWorkPurpose::IntermediateRestore,
                )
                .map_err(xlog_err)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .work = Some(work.clone());
            let custody = self
                .source
                .borrow(py)
                .owner()?
                .share_cold_native_work(&work)
                .map_err(xlog_err)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .custody = Some(custody.clone());
            let parent = self
                .intermediate()?
                .as_ref()
                .expect("original intermediate restore")
                .owners
                .as_ref()
                .expect("actual current-state owners")
                .parent
                .clone_ref(py);
            parent
                .borrow(py)
                .session
                .borrow(py)
                .owner()?
                .attach_shared_cold_native_work(&*parent.borrow(py).lease()?, custody)
                .map_err(xlog_err)?;
            let regions = self
                .source
                .borrow(py)
                .owner()?
                .prepare_cold_model_work_regions(&work, 4)
                .map_err(xlog_err)?;
            let mut callbacks = [None, None, None, None];
            for (index, region) in regions.iter().enumerate() {
                callbacks[index] = Some(Py::new(
                    py,
                    PySemanticColdModelWork::new(
                        if index < 2 {
                            Some(parent.clone_ref(py))
                        } else {
                            None
                        },
                        self.parent.clone_ref(py),
                        work.clone(),
                        Some(region.clone()),
                        false,
                    ),
                )?);
            }
            {
                let mut retained = self.intermediate()?;
                let current = retained.as_mut().expect("original intermediate restore");
                current.regions = regions;
                current.callbacks = callbacks;
            }
            self.source
                .borrow(py)
                .owner()?
                .begin_cold_model_work(&work)
                .map_err(xlog_err)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .ready = true;
        } else if !ready {
            return Err(invalid(
                "unknown intermediate admission cannot start a replacement interval",
            ));
        }
        let checkpoint = self.intermediate_snapshot(py, 0)?;
        self.require_intermediate_region_closed(py, 0)?;
        self.retire_intermediate_source(py)?;
        {
            let mut retained = self.intermediate()?;
            let current = retained.as_mut().expect("original intermediate restore");
            if let Some(issuer) = &current.resident_restoration_attempt {
                if current.resident_prelude.is_none() {
                    let prelude = self
                        .source
                        .borrow(py)
                        .owner()?
                        .prepare_cold_resident_restoration(
                            current.work.as_ref().expect("original four-region report"),
                            issuer,
                        )
                        .map_err(xlog_err)?;
                    current.resident_prelude = Some(Arc::new(prelude));
                }
                issuer
                    .bind_native_prelude(
                        current
                            .resident_prelude
                            .as_ref()
                            .expect("original native prelude"),
                    )
                    .map_err(xlog_err)?;
            }
        }
        // The temporary recipe data retires before this original copy receipt.
        let _recipe_metadata;
        let (entered, restored, recipe, resident_restoration) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            _recipe_metadata = current
                .recipe
                .as_ref()
                .map(|recipe| {
                    let extent = recipe
                        .recipe
                        .len()
                        .checked_mul(std::mem::size_of::<SemanticLearningCopyReset>())
                        .ok_or_else(|| invalid("intermediate restore recipe extent overflow"))?;
                    self.task_use
                        .borrow(py)
                        .checkpoint
                        .material_sources
                        .native_budget()?
                        .reserve_auxiliary(u64::try_from(extent).map_err(|_| {
                            invalid("intermediate restore recipe exceeds its native index")
                        })?)
                        .map_err(xlog_err)
                })
                .transpose()?;
            (
                current.restore_entered,
                current.restored.as_ref().map(|value| value.clone_ref(py)),
                current.recipe.clone(),
                current.resident_restoration_attempt.clone(),
            )
        };
        let restored = match restored {
            Some(restored) => restored,
            None if entered => {
                return Err(invalid(
                    "unknown intermediate construction cannot enter a replacement factory",
                ))
            }
            None => {
                let snapshot = self.refresh_snapshot.bind(py).call0()?;
                let authority = AuthoritySnapshot::parse(&ColdValue::read(
                    &snapshot,
                    &mut (16 * 1024 * 1024),
                    0,
                )?)?;
                let task = self.task_use.borrow(py);
                authority.newer_than(&task.state()?.snapshot)?;
                check_learning_grant(&task, &self.grant_reference, &authority)?;
                self.require_intermediate_entry(py)?;
                let domain = task.checkpoint.training_domain.python_value(py)?;
                let checkpoint_limit = self
                    .preparation_inputs
                    .max_checkpoint_bytes
                    .as_ref()
                    .map(|value| value.python_value(py))
                    .transpose()?;
                let total_limit = self
                    .preparation_inputs
                    .max_total_checkpoint_bytes
                    .as_ref()
                    .map(|value| value.python_value(py))
                    .transpose()?;
                self.intermediate()?
                    .as_mut()
                    .expect("original intermediate restore")
                    .restore_entered = true;
                let result = PySemanticTransitionSession::restore_checkpoint_impl(
                    py,
                    checkpoint.bind(py).as_any(),
                    self.source.borrow(py).device_ordinal,
                    &snapshot,
                    self.model_owner(py, PhaseModelOwner::Restore)?.bind(py),
                    domain.bind(py),
                    task.checkpoint
                        .training_canary_owner
                        .as_ref()
                        .map(|owner| owner.bind(py).as_any()),
                    None,
                    recipe.as_ref(),
                    self.preparation_inputs
                        .resolve_checkpoint
                        .as_ref()
                        .map(|callback| callback.bind(py)),
                    checkpoint_limit.as_ref().map(|value| value.bind(py)),
                    total_limit.as_ref().map(|value| value.bind(py)),
                    Some(self.refresh_snapshot.bind(py)),
                    Some(&task.checkpoint.proposal_expense),
                    Some(&task.checkpoint.checkpoint_sources),
                    Some(&task.checkpoint.material_sources),
                    Some(pending),
                    resident_restoration.as_ref(),
                    None,
                    task.checkpoint.native_model_context.as_ref(),
                    None,
                );
                let mut retained = self.intermediate()?;
                let current = retained.as_mut().expect("original intermediate restore");
                match result {
                    Ok(restored) => {
                        current.restored = Some(restored.clone_ref(py));
                        restored
                    }
                    Err(error) => {
                        current.error = Some(error.clone_ref(py));
                        return Err(error);
                    }
                }
            }
        };
        self.require_intermediate_region_closed(py, 2)?;
        self.intermediate_snapshot(py, 1)?;
        self.require_intermediate_region_closed(py, 3)?;
        self.finish_intermediate_restore(py, &restored)
    }

    fn finish_intermediate_restore(
        &self,
        py: Python<'_>,
        restored: &Py<PySemanticTransitionRestoredCheckpoint>,
    ) -> PyResult<()> {
        let streams = self.preparation_inputs.consumer_streams.python_value(py)?;
        let streams = checkpoint_consumer_streams(streams.bind(py), &mut (16 * 1024 * 1024))?;
        let joined = self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .child_joined;
        if !joined {
            let owners = restored.borrow(py);
            owners
                .session
                .borrow(py)
                .owner()?
                .complete_shared_cold_native_work(&*owners.parent.borrow(py).lease()?, &streams)
                .map_err(xlog_err)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .child_joined = true;
        }
        let (work, closed, report) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            (
                current.work.clone().expect("original restore report"),
                current.closed,
                current.report,
            )
        };
        if !closed {
            self.source
                .borrow(py)
                .owner()?
                .close_cold_model_work(&work)
                .map_err(xlog_err)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .closed = true;
        }
        let report = match report {
            Some(report) => report,
            None => {
                let report = self
                    .source
                    .borrow(py)
                    .owner()?
                    .finish_cold_model_work(
                        &*self.parent.borrow(py).lease()?,
                        &work,
                        &streams,
                        xlog_cuda::SemanticColdModelWorkDisposition::Complete,
                    )
                    .map_err(xlog_err)?;
                self.intermediate()?
                    .as_mut()
                    .expect("original intermediate restore")
                    .report = Some(report);
                report
            }
        };
        let finish = {
            let mut retained = self.intermediate()?;
            let current = retained.as_mut().expect("original intermediate restore");
            let finish = !current.observer_finish_entered;
            current.observer_finish_entered = true;
            finish
        };
        if finish {
            self.preparation_inputs.resource_observer.finish(py)?;
        }
        let peak = self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .backing_peak;
        let peak = if let Some(peak) = peak {
            peak
        } else {
            let peak = self.preparation_inputs.resource_observer.backing_peak(py)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .backing_peak = Some(peak);
            peak
        };
        let work = report
            .native_work
            .checked_add(report.model_work)
            .ok_or_else(|| invalid("intermediate observed work overflowed"))?;
        let (
            instruction,
            ordinal,
            branch,
            budget,
            entered,
            recorded,
            source,
            checkpoint,
            restored_checkpoint,
        ) = {
            let retained = self.intermediate()?;
            let current = retained.as_ref().expect("original intermediate restore");
            (
                current.instruction.clone(),
                current.ordinal,
                current.branch,
                current.budget,
                current.record_entered,
                current.recorded,
                current.source_identity.clone_ref(py),
                current.snapshots[0]
                    .saved
                    .as_ref()
                    .expect("actual complete current checkpoint")
                    .clone_ref(py),
                current.snapshots[1]
                    .saved
                    .as_ref()
                    .expect("actual complete restored checkpoint")
                    .clone_ref(py),
            )
        };
        if !recorded {
            if entered {
                return Err(invalid(
                    "unknown intermediate history append cannot repeat its original callback",
                ));
            }
            let arguments = PyDict::new(py);
            arguments.set_item("instruction_bytes", PyBytes::new(py, &instruction))?;
            arguments.set_item("operation_ordinal", ordinal)?;
            arguments.set_item("branch", branch)?;
            arguments.set_item("source_parent", source.bind(py))?;
            arguments.set_item(
                "restored_parent",
                restored.borrow(py).parent.borrow(py).identity(py)?,
            )?;
            arguments.set_item("source_checkpoint", checkpoint.bind(py))?;
            arguments.set_item("restored_checkpoint", restored_checkpoint.bind(py))?;
            arguments.set_item("resource_usage", (work, peak, report.model_calls))?;
            self.preparation_inputs
                .resource_observer
                .observation_arguments(py, &arguments)?;
            let callback = self.scientific_owner.bind(py).getattr("record_restore")?;
            {
                let mut retained = self.intermediate()?;
                let current = retained.as_mut().expect("original intermediate restore");
                current.record_entered = true;
                current.budget_exceeded =
                    work > budget[0] || peak > budget[1] || report.model_calls > budget[2];
            }
            callback.call((), Some(&arguments))?;
            self.require_scientific_history(py)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .recorded = true;
        }
        if !self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .released
        {
            self.preparation_inputs.resource_observer.release(py)?;
            self.intermediate()?
                .as_mut()
                .expect("original intermediate restore")
                .released = true;
        }
        if self
            .intermediate()?
            .as_ref()
            .expect("original intermediate restore")
            .budget_exceeded
        {
            return Err(invalid("intermediate restore exceeded its original budget; retain the actual recorded expenditure"));
        }
        Ok(())
    }

    pub(super) fn intermediate_execution_input(
        &self,
        py: Python<'_>,
        branch: &'static str,
    ) -> PyResult<Option<(Py<PySemanticTransitionRestoredCheckpoint>, u64)>> {
        let retained = self.intermediate()?;
        let Some(current) = retained.as_ref() else {
            return Ok(None);
        };
        if current.branch != branch {
            return Ok(None);
        }
        if !current.recorded
            || !current.released
            || !current.child_joined
            || !current.session_released
            || current.budget_exceeded
            || current.error.is_some()
        {
            return Err(invalid("private numerical continuation requires known complete intermediate restoration and accounting"));
        }
        Ok(Some((
            current
                .restored
                .as_ref()
                .expect("known intermediate restore")
                .clone_ref(py),
            current
                .ordinal
                .checked_add(1)
                .ok_or_else(|| invalid("intermediate successor position overflowed"))?,
        )))
    }

    pub(super) fn drop_completed_intermediate_restore(&self, branch: &'static str) -> PyResult<()> {
        let original = {
            let mut retained = self.intermediate()?;
            if retained.as_ref().is_some_and(|current| {
                current.branch != branch
                    || !current.recorded
                    || !current.released
                    || current.budget_exceeded
                    || current.error.is_some()
                    || current.previous.is_some()
            }) {
                return Err(invalid(
                    "model retirement cannot discard an unfinished intermediate restoration",
                ));
            }
            retained.take()
        };
        drop(original);
        Ok(())
    }
}
