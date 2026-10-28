//! Original checkpoint custody for a positively retired, non-submitted instruction.

use super::*;

pyo3::create_exception!(
    pyxlog._native,
    SemanticTransitionCheckpointCancelled,
    PyRuntimeError,
    "The original instruction was cancelled without submission; its charged terminal checkpoint cannot resume execution."
);

pub(super) struct OriginalCancelledCheckpointCustody {
    material: xlog_cuda::SemanticReplayLoadedMaterial,
    native: Mutex<xlog_cuda::SemanticReplayLoadedMaterial>,
    source: OriginalCancelledCheckpointSource,
    seed: TaskCheckpointSeed,
    snapshot: AuthoritySnapshot,
    phase: CheckpointTaskPhase,
    binding: TaskCheckpointBinding,
    current_binding: TaskCheckpointBinding,
    issuance: TaskIssuance,
    parent: xlog_cuda::SemanticPublishedIdentity,
    source_material_digest: Identity256,
    source_checkpoint_digest: Identity256,
    completion: Mutex<CancelledCheckpointCompletion>,
    budget: xlog_cuda::SemanticReplayMaterialBudget,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

enum OriginalCancelledCheckpointSource {
    Ordinary,
    ModelContext(xlog_cuda::SemanticModelContextContinuation),
}

#[derive(Default)]
struct CancelledCheckpointCompletion {
    assignments: Option<xlog_cuda::SemanticReplayLoadedMaterial>,
    sealed: Option<xlog_cuda::SemanticReplayLoadedMaterial>,
    handoff: Option<Py<material_bytes::PySemanticMaterialBytes>>,
}

pub(super) struct CancelledCheckpointTerminal {
    pub(super) first_program_ordinal: u64,
    pub(super) end_program_ordinal: u64,
    pub(super) parent: xlog_cuda::SemanticPublishedIdentity,
    pub(super) proposal_spent: u64,
    pub(super) task_identity: [u8; 32],
    source_checkpoint_digest: Identity256,
    evaluation_intervention: Option<(xlog_cuda::SemanticEvaluationIntervention, Identity256)>,
    budget: xlog_cuda::SemanticReplayMaterialBudget,
}

fn material_digest(bytes: &[u8]) -> Identity256 {
    Identity256::from_bytes(Sha256::digest(bytes).into())
}

pub(super) fn initial_index(admission: &OriginalActorRefreshAdmission) -> PyResult<Identity256> {
    let kit = admission.initial_kit.fields(2)?;
    let ColdValue::Bytes(index) = &kit[0] else {
        return Err(invalid("original replay index bytes are absent"));
    };
    Ok(material_digest(index))
}

fn require_seed_fields(
    original: &ColdValue,
    current: &ColdValue,
    terminal: bool,
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
) -> PyResult<()> {
    let original = original.fields(21)?;
    let current = current.fields(21)?;
    if original[20] != current[20] {
        let prior = original[20].fields(8)?;
        let latest = current[20].fields(8)?;
        if prior[..7] != latest[..7] {
            return Err(invalid(
                "cancelled checkpoint changed its immutable evaluation donor",
            ));
        }
        let (ColdValue::Bytes(prior), ColdValue::Bytes(latest)) = (&prior[7], &latest[7]) else {
            return Err(invalid(
                "cancelled checkpoint lost its accepted donor authority snapshot",
            ));
        };
        let extent = [prior, latest]
            .into_iter()
            .try_fold(0usize, |total, bytes| {
                let (_, owned, nodes) = ColdValue::checked_canonical_bytes(bytes, false)?;
                nodes
                    .checked_mul(std::mem::size_of::<ColdValue>())
                    .and_then(|extent| extent.checked_add(owned))
                    .and_then(|extent| extent.checked_add(bytes.len()))
                    .and_then(|extent| extent.checked_mul(3))
                    .and_then(|extent| extent.checked_add(total))
                    .ok_or_else(|| invalid("cancelled donor snapshot comparison extent overflow"))
            })?;
        let _metadata = budget
            .reserve_auxiliary(
                u64::try_from(extent)
                    .map_err(|_| invalid("cancelled donor snapshot comparison exceeds u64"))?,
            )
            .map_err(xlog_err)?;
        let prior = AuthoritySnapshot::parse(&ColdValue::from_canonical_bytes(prior)?)?;
        let latest = AuthoritySnapshot::parse(&ColdValue::from_canonical_bytes(latest)?)?;
        latest.newer_than(&prior)?;
    }
    if original
        .iter()
        .zip(current)
        .enumerate()
        .any(|(index, (left, right))| {
            index != 20 && !(terminal && matches!(index, 12 | 18)) && left != right
        })
    {
        return Err(invalid(
            "cancelled checkpoint differs from its original task capsule",
        ));
    }
    Ok(())
}

fn require_initial_prefill_source(
    manifest: &impl CheckpointManifestReader,
    phase: &CheckpointTaskPhase,
    binding: TaskCheckpointBinding,
    native: &mut xlog_cuda::SemanticReplayLoadedMaterial,
) -> PyResult<()> {
    let manifest = manifest.borrowed();
    let has_initial_prefill =
        SemanticTransitionSession::verify_checkpoint_loaded_initial_prefill_source(
            manifest.initial_prefill,
            native,
            Identity256::from_bytes(binding.identity),
            binding.epoch,
            binding.content_identity(),
            *phase == CheckpointTaskPhase::InitialPrefillBound,
        )
        .map_err(xlog_err)?;
    if has_initial_prefill != (*phase == CheckpointTaskPhase::InitialPrefillBound)
        && !matches!(phase, CheckpointTaskPhase::Segment(_))
    {
        return Err(invalid(
            "checkpoint initial prefill differs from its task phase",
        ));
    }
    Ok(())
}

fn require_session_checkpoint_custody(
    session: &PySemanticTransitionSession,
    manifest: &impl CheckpointManifestReader,
    bytes: &[u8],
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
) -> PyResult<()> {
    let retained = session.checkpoint_custody()?;
    let custody = retained
        .as_ref()
        .ok_or_else(|| invalid("checkpoint source lost its original Session signing owner"))?;
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    if Some(digest) != custody.origin {
        let _seal_storage = budget
            .reserve_auxiliary(custody.seal_extent() as u64)
            .map_err(xlog_err)?;
        let owner = session.owner()?;
        if custody.seal(manifest, owner.learning_phase_history())? != manifest.borrowed().phase_seal
        {
            return Err(invalid(
                "checkpoint source differs from this Session's authenticated custody",
            ));
        }
    }
    Ok(())
}

impl OriginalCancelledCheckpointCustody {
    pub(super) fn require_original_material(&self, checkpoint: &Bound<'_, PyAny>) -> PyResult<()> {
        if !checkpoint.is_exact_instance_of::<material_bytes::PySemanticMaterialBytes>() {
            return Err(invalid(
                "cancelled checkpoint custody requires original guarded material",
            ));
        }
        let carrier = checkpoint
            .cast::<material_bytes::PySemanticMaterialBytes>()?
            .borrow();
        if !std::ptr::eq(carrier.as_bytes(), self.material.as_bytes()) {
            return Err(invalid(
                "instruction cannot replace its original checkpoint material",
            ));
        }
        Ok(())
    }

    /// Validate the pre-request capsule before the instruction can charge or hand off.
    pub(super) fn prepare(
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
        checkpoint: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        session.require_creator()?;
        parent.require_task(py, task)?;
        if !checkpoint.is_exact_instance_of::<material_bytes::PySemanticMaterialBytes>() {
            return Err(invalid(
                "cancelled checkpoint custody requires original guarded material",
            ));
        }
        let source = {
            let state = task.state()?;
            if let TaskUsePhase::ContextActivated { scope, .. } = &state.phase {
                let prepared = session
                    .prepared_segment
                    .lock()
                    .map_err(|_| invalid("activated checkpoint lost its prepared owner"))?;
                let prepared = prepared
                    .as_ref()
                    .filter(|prepared| Arc::ptr_eq(scope, &prepared.scope))
                    .ok_or_else(|| {
                        invalid("activated checkpoint changed its original frozen scope")
                    })?;
                OriginalCancelledCheckpointSource::ModelContext(
                    prepared.model_context.clone().ok_or_else(|| {
                        invalid("activated checkpoint lost its original continuation")
                    })?,
                )
            } else {
                OriginalCancelledCheckpointSource::Ordinary
            }
        };
        let registry = Arc::clone(&task.checkpoint.material_sources);
        let budget = registry.native_budget()?;
        let metadata = budget
            .reserve_auxiliary(
                (std::mem::size_of::<Self>() + 2 * std::mem::size_of::<usize>()) as u64,
            )
            .map_err(xlog_err)?;
        let carrier = checkpoint
            .cast::<material_bytes::PySemanticMaterialBytes>()?
            .borrow();
        let material = carrier.original_loaded(&budget)?;
        let bytes = material.as_bytes();
        let manifest = SemanticCheckpointManifest::decode_borrowed(bytes)?;
        require_session_checkpoint_custody(session, &manifest, bytes, &budget)?;
        let mut native = material
            .slice_payload(checkpoint_section_range(bytes, manifest.native)?)
            .map_err(xlog_err)?;
        let (seed, snapshot, phase, binding) =
            TaskCheckpointSeed::decode_bound(manifest.task, &registry)?;
        require_initial_prefill_source(&manifest, &phase, binding, &mut native)?;
        let config = SemanticCheckpointSessionConfig::decode(&manifest.session)?;
        if config.capacities != session.capacities
            || config.admission_limits != session.admission_limits
            || config.memory_bytes != session.memory_bytes
        {
            return Err(invalid(
                "cancelled checkpoint has another original Session reservation",
            ));
        }
        let expense = seed.proposal_expense()?;
        if expense.original_admission.is_none() {
            return Err(invalid(
                "cancelled checkpoint requires the original complete Program",
            ));
        }
        if terminal_checkpoint(&manifest, &seed, &phase, binding, &mut native)?.is_some() {
            return Err(invalid(
                "a terminal cancelled checkpoint cannot admit another instruction",
            ));
        }
        let state = task.state()?;
        if checkpoint_task_phase(&state)? != phase {
            return Err(invalid(
                "cancelled checkpoint has another original task phase",
            ));
        }
        if state.snapshot.canonical != snapshot.canonical {
            state.snapshot.newer_than(&snapshot)?;
        }
        drop(state);
        let owner = session.owner()?;
        task.require_current(&owner)?;
        let current_binding = TaskCheckpointBinding::from_owner(&owner)?;
        if !binding.same_task_content(current_binding) {
            return Err(invalid(
                "cancelled checkpoint has another native task binding",
            ));
        }
        // The original restore owner, not Python digest equivalence, authenticates
        // a saved instance when the selected parent was created by fresh restore.
        let lease = parent.lease()?;
        let (source_material_digest, restored_from_exact_source) = match &source {
            OriginalCancelledCheckpointSource::Ordinary => {
                owner.verify_checkpoint_loaded_source_material(&lease, &mut native)
            }
            OriginalCancelledCheckpointSource::ModelContext(continuation) => owner
                .verify_model_context_checkpoint_loaded_source(continuation, &lease, &mut native),
        }
        .map_err(xlog_err)?;
        if binding.epoch != current_binding.epoch && !restored_from_exact_source {
            return Err(invalid(
                "cancelled checkpoint has another original native task epoch",
            ));
        }
        let selected = owner.published_identity(&lease).map_err(xlog_err)?;
        let current_task = task
            .checkpoint
            .encode(&owner, &snapshot, &phase, &binding)?;
        let original_plan =
            checked_checkpoint_cold_value(manifest.task, manifest.task.len(), false, None)?;
        let current_plan =
            checked_checkpoint_cold_value(&current_task, current_task.len(), false, None)?;
        let _values = budget
            .reserve_auxiliary(
                original_plan
                    .owned_bytes
                    .checked_add(current_plan.owned_bytes)
                    .ok_or_else(|| invalid("cancelled checkpoint comparison storage overflow"))?,
            )
            .map_err(xlog_err)?;
        let seed_value = checkpoint_cold_value(manifest.task, manifest.task.len())?;
        require_seed_fields(
            &seed_value,
            &checkpoint_cold_value(&current_task, current_task.len())?,
            false,
            &budget,
        )?;
        let source_checkpoint_digest = material_digest(bytes);
        Ok(Self {
            material,
            native: Mutex::new(native),
            source,
            seed,
            snapshot,
            phase,
            binding,
            current_binding,
            issuance: task.issuance.clone(),
            parent: selected,
            source_material_digest,
            source_checkpoint_digest,
            completion: Mutex::new(CancelledCheckpointCompletion::default()),
            budget,
            _metadata: metadata,
        })
    }

    /// Attach the already validated source to this native issuance before any charge.
    pub(super) fn bind_source(
        &self,
        owner: &mut SemanticTransitionSession,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
    ) -> PyResult<()> {
        if owner.published_identity(parent).map_err(xlog_err)? != self.parent {
            return Err(invalid(
                "cancelled checkpoint source parent changed before admission",
            ));
        }
        let source_expense = self.seed.proposal_expense()?;
        let source_assignments =
            self.seed
                .actor_refresh_assignments
                .as_deref()
                .ok_or_else(|| {
                    invalid("cancelled checkpoint lost its original native source assignments")
                })?;
        let mut native = self
            .native
            .lock()
            .map_err(|_| invalid("cancelled checkpoint native custody is poisoned"))?;
        match &self.source {
            OriginalCancelledCheckpointSource::Ordinary => owner
                .bind_segment_loaded_checkpoint_source(
                    admission,
                    parent,
                    &mut native,
                    self.source_checkpoint_digest,
                    source_expense.spent,
                    source_assignments,
                ),
            OriginalCancelledCheckpointSource::ModelContext(continuation) => owner
                .bind_model_context_checkpoint_loaded_source(
                    continuation,
                    admission,
                    parent,
                    &mut native,
                    self.source_checkpoint_digest,
                    source_expense.spent,
                    source_assignments,
                ),
        }
        .map_err(xlog_err)
    }

    /// Freeze only after the original native cancellation and all final-use reports.
    pub(super) fn seal(
        &self,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
        owner: &mut SemanticTransitionSession,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> PyResult<()> {
        session.require_creator()?;
        let mut completion = self
            .completion
            .lock()
            .map_err(|_| invalid("cancelled checkpoint custody mutex is poisoned"))?;
        if completion.sealed.is_some() {
            return Ok(());
        }
        if !self.issuance.same_as(&task.issuance) {
            return Err(invalid(
                "cancelled checkpoint requires its original issued task",
            ));
        }
        task.require_original_identity(owner)?;
        if TaskCheckpointBinding::from_owner(owner)? != self.current_binding {
            return Err(invalid(
                "cancelled checkpoint native task changed during retirement",
            ));
        }
        if completion.assignments.is_none() {
            let assignments = owner
                .cancelled_segment_loaded_assignment_material(admission)
                .map_err(xlog_err)?;
            // Retain the native terminal disposition before any fallible encoding.
            completion.assignments = Some(assignments);
        }
        let task_bytes =
            task.checkpoint
                .encode(owner, &self.snapshot, &self.phase, &self.binding)?;
        let manifest = SemanticCheckpointManifest::decode_borrowed(self.material.as_bytes())?;
        let original_plan =
            checked_checkpoint_cold_value(manifest.task, manifest.task.len(), false, None)?;
        let current_plan =
            checked_checkpoint_cold_value(&task_bytes, task_bytes.len(), false, None)?;
        let _values = self
            .budget
            .reserve_auxiliary(
                original_plan
                    .owned_bytes
                    .checked_add(current_plan.owned_bytes)
                    .ok_or_else(|| invalid("cancelled checkpoint comparison storage overflow"))?,
            )
            .map_err(xlog_err)?;
        let task_value = checkpoint_cold_value(&task_bytes, task_bytes.len())?;
        require_seed_fields(
            &checkpoint_cold_value(manifest.task, manifest.task.len())?,
            &task_value,
            true,
            &self.budget,
        )?;
        let fields = task_value.fields(21)?;
        let ColdValue::Bytes(assignments) = &fields[18] else {
            return Err(invalid(
                "cancelled checkpoint has no original native assignment material",
            ));
        };
        if completion
            .assignments
            .as_ref()
            .map(|original| original.as_bytes())
            != Some(assignments.as_ref())
        {
            return Err(invalid(
                "cancelled checkpoint native assignment material changed while sealing",
            ));
        }
        let (seed, _, _, binding) =
            TaskCheckpointSeed::decode_bound(&task_bytes, &self.seed.material_sources)?;
        let mut terminal_manifest = BorrowedSemanticCheckpointManifest {
            native: manifest.native,
            model: manifest.model,
            task: &task_bytes,
            session: manifest.session,
            initial_prefill: manifest.initial_prefill,
            model_context: manifest.model_context,
            phase_seal: &[],
        };
        let terminal = terminal_checkpoint(
            &terminal_manifest,
            &seed,
            &self.phase,
            binding,
            &mut *self
                .native
                .lock()
                .map_err(|_| invalid("cancelled checkpoint native custody is poisoned"))?,
        )?
        .ok_or_else(|| invalid("native cancellation has no complete terminal disposition"))?;
        if terminal.parent != self.parent
            || material_digest(&terminal_manifest.native) != self.source_material_digest
            || terminal.source_checkpoint_digest != self.source_checkpoint_digest
        {
            return Err(invalid(
                "cancelled checkpoint lost its original native source custody",
            ));
        }
        let retained = session.checkpoint_custody()?;
        let signing = retained
            .as_ref()
            .ok_or_else(|| invalid("cancelled checkpoint lost its original signing owner"))?;
        let _seal_storage = self
            .budget
            .reserve_auxiliary(signing.seal_extent() as u64)
            .map_err(xlog_err)?;
        let phase_seal = signing.seal(&terminal_manifest, owner.learning_phase_history())?;
        terminal_manifest.phase_seal = &phase_seal;
        let extent = terminal_manifest.encoded_extent()?;
        let output = self
            .budget
            .prepare_material_output(extent)
            .map_err(xlog_err)?;
        output
            .declare_spans(extent, &[(0, extent)])
            .map_err(xlog_err)?;
        let mut offset = 0usize;
        terminal_manifest.write_encoded(|part| {
            if !part.is_empty() {
                output.write_model(offset, part).map_err(xlog_err)?;
                offset = offset
                    .checked_add(part.len())
                    .ok_or_else(|| invalid("cancelled checkpoint output extent overflow"))?;
            }
            Ok(())
        })?;
        if offset != extent {
            return Err(invalid(
                "cancelled checkpoint encoder changed its original extent",
            ));
        }
        completion.sealed = Some(output.seal_model().map_err(xlog_err)?);
        Ok(())
    }

    pub(super) fn sealed_bytes(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<Py<material_bytes::PySemanticMaterialBytes>>> {
        let mut completion = self
            .completion
            .lock()
            .map_err(|_| invalid("cancelled checkpoint custody mutex is poisoned"))?;
        if let Some(handoff) = &completion.handoff {
            return Ok(Some(handoff.clone_ref(py)));
        }
        let Some(sealed) = &completion.sealed else {
            return Ok(None);
        };
        let handoff = Py::new(
            py,
            material_bytes::PySemanticMaterialBytes::from_native(
                py,
                sealed
                    .slice_payload(0..sealed.as_bytes().len())
                    .map_err(xlog_err)?,
                &self.budget,
            )?,
        )?;
        completion.handoff = Some(handoff.clone_ref(py));
        Ok(Some(handoff))
    }
}

/// The normal and lightweight restore entries use this same metadata-only check.
pub(super) fn terminal_checkpoint(
    manifest: &impl CheckpointManifestReader,
    seed: &TaskCheckpointSeed,
    phase: &CheckpointTaskPhase,
    binding: TaskCheckpointBinding,
    native: &mut xlog_cuda::SemanticReplayLoadedMaterial,
) -> PyResult<Option<CancelledCheckpointTerminal>> {
    let manifest = manifest.borrowed();
    let expense = seed.proposal_expense()?;
    let Some(original) = expense.original_admission else {
        return Ok(None);
    };
    let assignments = seed
        .actor_refresh_assignments
        .as_deref()
        .ok_or_else(|| invalid("checkpoint original native assignment material is absent"))?;
    let terminal = SemanticTransitionSession::verify_cancelled_segment_assignment_material(
        &original.program,
        initial_index(&original)?,
        assignments,
    )
    .map_err(xlog_err)?;
    let Some((first, end, parent, proposal_spent, source_digest, source_checkpoint_digest)) =
        terminal
    else {
        return Ok(None);
    };
    if proposal_spent != expense.spent || source_digest != material_digest(&manifest.native) {
        return Err(invalid(
            "cancelled checkpoint source or charged Proposal expense is inconsistent",
        ));
    }
    SemanticTransitionSession::state_material_loaded_projection(native).map_err(xlog_err)?;
    SemanticCheckpointSessionConfig::decode(&manifest.session)?;
    require_initial_prefill_source(&manifest, phase, binding, native)?;
    let proof = native.checkpoint_proof().map_err(xlog_err)?;
    let evaluation_intervention = proof
        .evaluation_fork_origin()
        .map(|origin| {
            let library = proof
                .evaluation_rule_library()
                .ok_or_else(|| invalid("cancelled evaluation lost its original rule library"))?;
            Ok::<_, PyErr>((origin.intervention(), library.digest()))
        })
        .transpose()?;
    Ok(Some(CancelledCheckpointTerminal {
        first_program_ordinal: first,
        end_program_ordinal: end,
        parent,
        proposal_spent,
        task_identity: binding.identity,
        source_checkpoint_digest,
        evaluation_intervention,
        budget: seed.material_sources.native_budget()?,
    }))
}

pub(super) fn terminal_error(
    py: Python<'_>,
    checkpoint: &Bound<'_, PyAny>,
    terminal: &CancelledCheckpointTerminal,
) -> PyResult<PyErr> {
    let error = SemanticTransitionCheckpointCancelled::new_err(
        "the original non-submitted instruction is durably cancelled; no further request can be admitted",
    );
    let value = error.value(py);
    value.setattr("checkpoint", checkpoint)?;
    value.setattr("task_identity", PyBytes::new(py, &terminal.task_identity))?;
    value.setattr("first_program_ordinal", terminal.first_program_ordinal)?;
    value.setattr("end_program_ordinal", terminal.end_program_ordinal)?;
    value.setattr("proposal_spent", terminal.proposal_spent)?;
    let intervention = terminal
        .evaluation_intervention
        .map(|(mode, digest)| {
            evaluation_rule_library::intervention_material(py, mode, digest, &terminal.budget)
                .map(|material| material.into_any())
        })
        .transpose()?
        .unwrap_or_else(|| py.None());
    value.setattr("evaluation_intervention", intervention)?;
    value.setattr(
        "parent_identity",
        (
            PyBytes::new(py, terminal.parent.instance.as_bytes()),
            terminal.parent.word,
            PyBytes::new(py, terminal.parent.logical_digest.as_bytes()),
        ),
    )?;
    Ok(error)
}

#[pyfunction]
#[pyo3(signature = (checkpoint, *, program_bytes, material_source, max_resolved_material_bytes, snapshot=None, learning_phase_issuer=None, learning_phase_records=None, learning_phase_record_limits=None, learning_phase_fence=None, learning_phase_checkpoint_issuer=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "terminal recovery authenticates the original complete lifecycle without native execution"
)]
pub(crate) fn restore_cancelled_checkpoint(
    py: Python<'_>,
    checkpoint: &Bound<'_, PyAny>,
    program_bytes: &Bound<'_, PyAny>,
    material_source: &Bound<'_, PyAny>,
    max_resolved_material_bytes: &Bound<'_, PyAny>,
    snapshot: Option<&Bound<'_, PyAny>>,
    learning_phase_issuer: Option<&Bound<'_, PyAny>>,
    learning_phase_records: Option<&Bound<'_, PyAny>>,
    learning_phase_record_limits: Option<&Bound<'_, PyAny>>,
    learning_phase_fence: Option<&Bound<'_, PyAny>>,
    learning_phase_checkpoint_issuer: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    if !checkpoint.is_exact_instance_of::<material_bytes::PySemanticMaterialBytes>() {
        return Err(invalid(
            "cancelled checkpoint recovery requires original guarded material",
        ));
    }
    let material = checkpoint
        .cast::<material_bytes::PySemanticMaterialBytes>()?
        .borrow();
    let raw = material.as_bytes();
    let program = exact_replay_bytes(program_bytes)?;
    let manifest = SemanticCheckpointManifest::decode_borrowed(raw)?;
    let (saved_store, item_cap, store_cap, saved_resolved_cap) =
        checkpoint_material_source_header(manifest.task)?;
    if !max_resolved_material_bytes.is_exact_instance_of::<PyInt>() {
        return Err(invalid(
            "resolved material cap requires an exact positive integer",
        ));
    }
    let resolved_cap = max_resolved_material_bytes.extract::<u64>()?;
    if resolved_cap == 0 || resolved_cap != saved_resolved_cap {
        return Err(invalid(
            "cancelled checkpoint restore changed its original resolved material cap",
        ));
    }
    let material_sources = material_owner::original_registry(material_source)?;
    if material_sources.saved_caps()? != Some((item_cap, store_cap, resolved_cap)) {
        return Err(invalid(
            "cancelled checkpoint recovery changed its original material source caps",
        ));
    }
    material_sources.require_store(saved_store)?;
    let original_loaded = material.original_loaded(&material_sources.native_budget()?)?;
    let (seed, _, phase, binding) =
        TaskCheckpointSeed::decode_bound(manifest.task, &material_sources)?;
    let mut native = original_loaded
        .slice_payload(checkpoint_section_range(raw, manifest.native)?)
        .map_err(xlog_err)?;
    let expense = seed.proposal_expense()?;
    if expense
        .original_admission
        .as_ref()
        .map(|original| original.program.as_ref())
        != Some(program)
    {
        return Err(invalid(
            "cancelled checkpoint differs from the expected original Program",
        ));
    }
    let target_snapshot = if let Some(source) = &seed.dormant_evaluation_source {
        snapshot
            .map(|pair| -> PyResult<_> {
                let (current_source, target) = evaluation_authority_snapshots(pair)?;
                source.require_current_source_snapshot(
                    &current_source,
                    &material_sources.native_budget()?,
                )?;
                Ok(target)
            })
            .transpose()?
    } else {
        None
    };
    let _closure = learning_phase::cold_restore::read_checkpoint_restore_closure_bound(
        raw,
        target_snapshot.as_ref().or(snapshot),
        learning_phase_issuer,
        learning_phase_records,
        learning_phase_record_limits,
        learning_phase_fence,
        learning_phase_checkpoint_issuer,
        &material_sources,
        &mut native,
    )?;
    let terminal = terminal_checkpoint(&manifest, &seed, &phase, binding, &mut native)?
        .ok_or_else(|| invalid("checkpoint is not a positively completed cancelled terminal"))?;
    Err(terminal_error(py, checkpoint, &terminal)?)
}
