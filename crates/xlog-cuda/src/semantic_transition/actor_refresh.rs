use super::*;

/// Native custody of an original actor row and one Update's actual input owners.
/// This proof does not claim that a pending model generation has executed.
#[derive(Clone)]
pub struct SemanticPreparedActorRefresh {
    pub(super) inner: Arc<PreparedActorRefreshOwner>,
}

pub(super) struct PreparedActorRefreshOwner {
    issuer: Arc<()>,
    scope: Arc<()>,
    step: u64,
    logical_update: u64,
    program: Identity256,
    phase: Identity256,
    rng: SemanticRngBinding,
    member: crate::semantic_training_view::FrozenPolicyGroupMember,
    inputs: Arc<PreparedStepInputs>,
    models: Vec<Arc<ModelGenerationOwner>>,
    provider: Arc<CudaKernelProvider>,
    domain: ResidentExecutionDomain,
    children: Mutex<[Option<Arc<()>>; 2]>,
}

pub(super) struct StagedActorRefresh {
    pub(super) proof: SemanticPreparedActorRefresh,
    pub(super) target_bank: usize,
    pub(super) recorded: bool,
}

pub(super) struct ActorRefreshInitialEpisodes {
    pub(super) index: Identity256,
    pub(super) episodes: Vec<(SemanticTrainingViewOriginRecord, Identity256)>,
    pub(super) actors: Vec<SemanticTrainingViewOriginRecord>,
}

pub(super) fn initial_episode_actor(
    material: &SemanticReplayMaterial,
    batch_identity: Identity256,
    batch_bytes: &[u8],
) -> Result<(SemanticTrainingViewOriginRecord, bool), SemanticTransitionError> {
    if material.kind != SemanticTransitionKind::Proposal
        || batch_bytes.len() != size_of::<SemanticActionBatchReceipt>()
        || Identity256::from_bytes(Sha256::digest(batch_bytes).into()) != batch_identity
    {
        return Err(publication_input_error("initial actor requires its complete original native action batch"));
    }
    // SAFETY: this complete integer-only native ABI has its exact extent above.
    let batch = unsafe { std::ptr::read_unaligned(batch_bytes.as_ptr().cast::<SemanticActionBatchReceipt>()) };
    let header = material.predecessor.bank.header;
    let state = material.predecessor.bank.state;
    let rng = header.rng_binding()?;
    let successor = material.evidence.successor;
    let attempt = material.evidence.range(33)?.attempt()?;
    let next = u64::from(rng.proposal).checked_add(1)
        .filter(|next| *next <= u64::from(u32::MAX))
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    if batch.abi != 1 || batch.actor_eligible > 1
        || batch_identity != attempt.action_receipts_digest
        || batch.proposal != u64::from(rng.proposal)
        || batch.stream_serial != rng.stream_serial
        || batch.family_id != u64::from(rng.family_id)
        || batch.model_generation != u64::from(rng.model_generation)
        || batch.action_law_generation != state.catalogue_generation
        || batch.catalogue_digest != state.catalogue_digest
        || batch.component_count != COMPONENT_COUNT as u64
        || batch.candidate_count != 3 || batch.winner > 2
        || batch.base_word != header.publication_word
        || batch.next_word != successor.word
        || batch.base_logical_digest != header.logical_digest
        || batch.rng_base != u64::from(rng.proposal) * COMPONENT_COUNT as u64
        || batch.rng_span != COMPONENT_COUNT as u64
        || batch.rng_successor != next * COMPONENT_COUNT as u64
        || batch.semantic_receipts_digest != attempt.semantic_receipts_digest
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    Ok((crate::semantic_training_view::origin_record(material.training_view_origin()?),
        batch.actor_eligible == 1))
}

impl SemanticTransitionSession {
    /// Bind every original sealed episode in original index order. The trusted
    /// typed importer verifies complete index coverage before this call; native
    /// batch receipts, not row count or host flags, determine qualified actors.
    pub fn bind_actor_refresh_initial_episodes(
        &mut self,
        index_identity: Identity256,
        initial_episodes: &[(SemanticReplayMaterial, Identity256, Vec<u8>)],
    ) -> Result<u64, SemanticTransitionError> {
        self.ensure_quiescent()?;
        if index_identity == Identity256::default() || self.prepared_segment.is_some() {
            return Err(publication_input_error("initial actor roster must precede original prepared work"));
        }
        let mut episodes = Vec::with_capacity(initial_episodes.len());
        let mut actors = Vec::new();
        for (material, batch_identity, batch_bytes) in initial_episodes {
            let (origin, actor) = initial_episode_actor(material, *batch_identity, batch_bytes)?;
            if episodes.iter().any(|(previous, _)| previous == &origin) {
                return Err(publication_input_error("initial episode roster repeats an original action"));
            }
            episodes.push((origin, *batch_identity));
            if actor { actors.push(origin); }
        }
        let count = u64::try_from(actors.len()).map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        if let Some(original) = &self.actor_refresh_initial {
            if original.index != index_identity || original.episodes != episodes || original.actors != actors {
                return Err(publication_input_error("initial actor roster differs from its original sealed index"));
            }
        } else {
            self.actor_refresh_initial = Some(ActorRefreshInitialEpisodes { index: index_identity, episodes, actors });
        }
        Ok(count)
    }

    /// Retain the real model input owners of a particular original Update.
    /// The released parent authenticates the builder's immutable provenance;
    /// it is never reacquired or used as a future Update publication.
    pub fn prepare_actor_refresh(
        &mut self,
        step: &SemanticPreparedStep,
        released_parent: &SemanticPublishedLease,
        program_ordinal: u64,
        member_ordinal: u64,
        material: &SemanticReplayMaterial,
        batch_identity: Identity256,
        batch_bytes: &[u8],
    ) -> Result<SemanticPreparedActorRefresh, SemanticTransitionError> {
        self.require_prepared_replay_parent(step, released_parent)?;
        if self.actor_refresh_preparations.contains_key(&(step.token, member_ordinal)) {
            return Err(publication_input_error("actor refresh retains one original attempt for this Update member"));
        }
        let address = self.prepared_segment.as_ref().ok_or(SemanticTransitionError::NotBound)?
            .program_steps.get(&step.token).ok_or_else(|| publication_input_error("actor refresh requires the original admitted program address"))?;
        if address.ordinal != program_ordinal { return Err(SemanticTransitionError::ObservationMismatch); }
        let logical_update = address.logical_update.ok_or(SemanticTransitionError::ObservationMismatch)?;
        let phase = address.phase;
        let owner = self.checked_prepared_step(step, false)?;
        let prepared = owner.prepared.as_ref().ok_or(SemanticTransitionError::NotBound)?;
        let training = prepared.training_view.as_ref().ok_or_else(||
            publication_input_error("actor refresh requires the original frozen actor roster"))?;
        let member = *training.actor_group_members().iter()
            .find(|member| member.ordinal == member_ordinal)
            .ok_or_else(|| publication_input_error("actor refresh row is outside the frozen actor group"))?;
        let (actual_origin, actor) = initial_episode_actor(material, batch_identity, batch_bytes)?;
        if !actor || actual_origin != member.origin { return Err(SemanticTransitionError::ObservationMismatch); }
        let inputs = Arc::clone(owner.inputs.as_ref().ok_or(SemanticTransitionError::NotBound)?);
        let models = inputs.model_slots.iter().map(|slots| {
            if slots[0] != slots[1] {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            inputs.storage.allocations.get(slots[0])
                .and_then(PublicationAllocation::model_owner)
                .ok_or(SemanticTransitionError::ObservationMismatch)
        }).collect::<Result<Vec<_>, _>>()?;
        if models.len() != inputs.storage.model_memory.allocation_bytes.len() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let original = self.actor_refresh_initial.as_ref().ok_or_else(|| publication_input_error("actor refresh lacks the complete initial actor roster"))?;
        let program = self.actor_refresh_program.as_ref().ok_or(SemanticTransitionError::NotBound)?;
        let initial_count = u64::try_from(original.actors.len()).map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let actor_slot = if let Some(index) = original.actors.iter().position(|origin| *origin == member.origin) {
            if !original.episodes.iter().any(|(origin, batch)| *origin == member.origin && *batch == batch_identity) {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            index as u64
        } else {
            initial_count.checked_add(program.main_origins.iter()
                .find(|assignment| assignment.origin == member.origin && assignment.batch == batch_identity
                    && assignment.batch_bytes == batch_bytes)
                .map(|assignment| assignment.slot).ok_or_else(|| publication_input_error("actor origin has no authenticated original program assignment"))?)
                .ok_or(SemanticTransitionError::GenerationExhausted)?
        };
        let stride = initial_count.checked_add(program.proposals).ok_or(SemanticTransitionError::GenerationExhausted)?;
        if actor_slot >= stride || logical_update >= program.logical_updates { return Err(SemanticTransitionError::ObservationMismatch); }
        let mut rng = program.original_rng.ok_or(SemanticTransitionError::NotBound)?;
        let end = program.logical_updates.checked_mul(stride)
            .and_then(|span| rng.stream_serial.checked_add(span))
            .filter(|end| *end < 1u64 << 56).ok_or(SemanticTransitionError::GenerationExhausted)?;
        rng.stream_serial = logical_update.checked_mul(stride)
            .and_then(|offset| offset.checked_add(actor_slot))
            .and_then(|offset| rng.stream_serial.checked_add(offset))
            .and_then(|serial| serial.checked_add(1))
            .filter(|serial| *serial <= end).ok_or(SemanticTransitionError::GenerationExhausted)?;
        rng.proposal = 0;
        let proof = SemanticPreparedActorRefresh { inner: Arc::new(PreparedActorRefreshOwner {
            issuer: Arc::clone(&self.publication_issuer),
            scope: Arc::clone(&step.scope),
            step: step.token,
            logical_update,
            program: program.identity,
            phase,
            rng,
            member,
            inputs,
            models,
            provider: Arc::clone(&self.provider),
            domain: self.domain.clone(),
            children: Mutex::new([None, None]),
        }) };
        if let Some(assignment) = self.actor_refresh_program.as_mut()
            .and_then(|program| program.main_origins.iter_mut().find(|assignment| assignment.origin == member.origin)) {
            match assignment.replay {
                actor_refresh_program::AssignmentReplayProof::Unresolved => {
                    assignment.replay = actor_refresh_program::AssignmentReplayProof::Joined;
                }
                actor_refresh_program::AssignmentReplayProof::Joined => (),
            }
        }
        self.actor_refresh_preparations.insert((step.token, member_ordinal), Arc::downgrade(&proof.inner));
        Ok(proof)
    }

    /// Cold-stage original context with the target's retained CURRENT owners.
    /// No numerical model bytes are copied, zeroed, hashed, or observed here.
    /// Actual initialization and basis validation belong to the same captured
    /// Update branch before its genuine Recompute and new Proposal.
    pub fn restore_actor_refresh_context(
        &mut self,
        material: &SemanticReplayMaterial,
        proof: &SemanticPreparedActorRefresh,
        target_bank: usize,
    ) -> Result<(), SemanticTransitionError> {
        if target_bank > 1 || self.actor_refresh.is_some()
            || material.kind != SemanticTransitionKind::Proposal
            || !Arc::ptr_eq(&self.provider.memory, &proof.inner.provider.memory)
            || crate::semantic_training_view::origin_record(material.training_view_origin()?)
                != proof.inner.member.origin
        {
            return Err(publication_input_error("actor context requires its exact original row and current model owner"));
        }
        {
            let mut children = proof.inner.children.lock()
                .map_err(|_| publication_input_error("actor context custody lock is poisoned"))?;
            if children[target_bank].is_some() {
                return Err(publication_input_error("actor context is staged once per original conditional bank"));
            }
            children[target_bank] = Some(Arc::clone(&self.publication_issuer));
        }
        // Keep the proof even when original staging enters an uncertain CUDA
        // operation. A failed attempt cannot be replaced with a fresh child.
        self.actor_refresh = Some(StagedActorRefresh {
            proof: proof.clone(), target_bank, recorded: false,
        });
        self.domain = proof.inner.domain.clone();
        self.stream = Arc::clone(self.domain.execution_stream());
        let staged = self.restore_state_material_inner(
            &material.predecessor.encode()?, None,
            Some(RestoredModelOwners::Current(&proof.inner.models)),
        )?;
        if staged.is_some() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }
}
