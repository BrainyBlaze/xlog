use super::prefill_restoration::{
    dormant_prefill_source_geometry, OriginalDormantPrefillCheckpoint,
    OriginalRestoredModelContextPrefill, SemanticModelContextPrefillRestoreDeclaration,
    SemanticRestoredModelContextPrefill,
};
use super::*;
use crate::semantic_transition::dormant_material::{
    DormantModelMaterialSource, DormantPublicationMaterial,
};

/// The exact original dormant read destinations survive unknown completion.
/// Neither SOURCE's claim nor TARGET's future graph is executed by this owner.
pub struct SemanticModelContextCheckpointAttachment {
    source_issuer: Arc<()>,
    target_issuer: Arc<()>,
    continuation: Arc<ModelContextContinuationOwner>,
    source_parent: SemanticPublishedIdentity,
    storage: Arc<PublicationStorage>,
    source: Arc<DormantModelMaterialSource<PublicationMaterialBytes>>,
    bank: PublicationRead<PublicationBank>,
    control: PublicationRead<PublicationControl>,
    counts: PublicationRead<PublicationRoleCount>,
    terminals: Option<PublicationRead<u64>>,
    ranges: Vec<(PublicationRange, usize, Option<PublicationRead<u8>>)>,
    graph: crate::semantic_hypergraph::SemanticRootExport,
    material: Option<DormantPublicationMaterial>,
    loaded: Option<SemanticReplayLoadedMaterial>,
    mapping: SemanticReplayLoadedMaterial,
    assignments: SemanticReplayLoadedMaterial,
    prefill: OriginalDormantPrefillCheckpoint,
    loaded_prefill: Option<SemanticReplayLoadedMaterial>,
    allowance: Arc<Mutex<native_work_bound::ColdNativeAllowance>>,
    work: SemanticColdModelWork,
    poisoned: bool,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

pub(super) struct OriginalDormantContextRestore {
    source_parent: SemanticPublishedIdentity,
    source_token: u64,
    source_digest: Identity256,
    target_digest: Identity256,
    prefill_digest: Identity256,
    checkpoint_source_parent: SemanticPublishedIdentity,
    source_geometry: (u64, u64, u64, u64, u64),
    material: Option<DormantPublicationMaterial>,
    prefill_section: Option<SemanticReplayLoadedMaterial>,
    prefill: Option<OriginalRestoredModelContextPrefill>,
    allowance: Option<Arc<Mutex<native_work_bound::ColdNativeAllowance>>>,
    work: Option<SemanticColdModelWork>,
    native_ceilings: ([u64; 9], [u64; 9]),
    entered: bool,
    proof: Option<TrackedCudaSlice<PreparedModelContextContinuationDevice>>,
    proof_write: Option<state_restoration::OriginalDeviceWrite>,
    completed: bool,
    report_completed: bool,
    poisoned: bool,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

pub(in crate::semantic_transition) struct OriginalModelContextAcknowledgement {
    continuation: std::sync::Weak<ModelContextContinuationOwner>,
    issuer: Arc<()>,
    storage: Arc<PublicationStorage>,
    base_token: u64,
    base: SemanticPublishedIdentity,
    base_header: PublicationHeader,
    successor: Option<PublicationHeader>,
    model_ranges: Vec<PublicationRange>,
    models: Vec<Arc<ModelGenerationOwner>>,
    receipt: Arc<[u8]>,
    receipt_digest: Identity256,
    recipient: Identity256,
    result_receipt: Option<Arc<[u8]>>,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

fn model_context_header_identity(header: &PublicationHeader) -> SemanticPublishedIdentity {
    SemanticPublishedIdentity {
        instance: header.instance,
        word: header.publication_word,
        logical_digest: header.logical_digest,
        state_digest: header.state_digest,
    }
}

fn model_context_model_header_unchanged(
    original: &PublicationHeader,
    current: &PublicationHeader,
) -> bool {
    original.instance == current.instance
        && original.model_generation == current.model_generation
        && original.neural_bank == current.neural_bank
        && original.neural_generation == current.neural_generation
        && original.cache_generation == current.cache_generation
        && original.authority_generation == current.authority_generation
        && original.model_geometry_digest == current.model_geometry_digest
        && original.model_numerical_digest == current.model_numerical_digest
        && original.proposal == current.proposal
        && original.stream_serial == current.stream_serial
        && original.family_id == current.family_id
        && original.training_cursor == current.training_cursor
        && original.training_rng == current.training_rng
}

fn model_context_frozen_model_unchanged(
    frozen: &PublicationHeader,
    current: &PublicationHeader,
) -> bool {
    frozen.instance == current.instance
        && frozen.model_generation == current.model_generation
        && frozen.neural_bank == current.neural_bank
        && frozen.neural_generation == current.neural_generation
        && frozen.model_geometry_digest == current.model_geometry_digest
        && frozen.model_numerical_digest == current.model_numerical_digest
}

fn model_context_model_range_unchanged(
    original: &PublicationRange,
    current: &PublicationRange,
) -> bool {
    original.role == current.role
        && original.index == current.index
        && original.storage_slot == current.storage_slot
        && original.generation == current.generation
        && original.offset_bytes == current.offset_bytes
        && original.length_bytes == current.length_bytes
        && original.logical_begin == current.logical_begin
        && original.logical_end == current.logical_end
        && original.digest == current.digest
        && original.backing_digest == current.backing_digest
}

impl OriginalDormantContextRestore {
    pub(super) fn completion_pending(&self) -> Result<bool, SemanticTransitionError> {
        Ok(self
            .proof_write
            .as_ref()
            .is_some_and(state_restoration::OriginalDeviceWrite::completion_pending)
            || self
                .prefill
                .as_ref()
                .map(|prefill| prefill.completion_pending())
                .transpose()?
                .unwrap_or(false))
    }

    pub(super) fn known_prefill_proof(
        &self,
    ) -> Result<SemanticRestoredModelContextPrefill, SemanticTransitionError> {
        if !self.completed {
            return Err(SemanticTransitionError::NotBound);
        }
        self.prefill
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .proof()
    }

    pub(super) fn completed_prefill_proof(
        &self,
    ) -> Result<SemanticRestoredModelContextPrefill, SemanticTransitionError> {
        if !self.report_completed {
            return Err(SemanticTransitionError::NotBound);
        }
        self.known_prefill_proof()
    }
}

impl SemanticTransitionSession {
    pub(in crate::semantic_transition) fn retain_original_model_context_source_boundary(
        &self,
        header: PublicationHeader,
    ) -> Result<(), SemanticTransitionError> {
        let Some(owner) = self
            .model_context_continuation
            .as_ref()
            .filter(|owner| Arc::ptr_eq(&owner.source_issuer, &self.publication_issuer))
        else {
            return Ok(());
        };
        if header.abi != 1
            || header.publication_word == owner.source_header.publication_word
            || self
                .publication
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, &owner.source_publication))
            || !model_context_frozen_model_unchanged(&owner.source_header, &header)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        // Live continuations freeze model owners before Source Recompute. Its
        // known canonical result, not that input or a later Acquire, is the
        // one immutable boundary from which acknowledgements may proceed.
        let mut original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original.source_boundary.as_ref().is_some_and(|known| {
            model_context_header_identity(known) != model_context_header_identity(&header)
        }) || !original.source_acknowledgements.is_empty()
            || original.activating
            || original.activated.is_some()
            || original.refused
            || original.checkpoint_restore.is_some()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        original.source_boundary = Some(header);
        Ok(())
    }

    fn require_model_context_source_model_basis(
        &self,
        owner: &Arc<ModelContextContinuationOwner>,
        parent: &SemanticPublishedLease,
        resume_alias: bool,
    ) -> Result<(), SemanticTransitionError> {
        let reader = if resume_alias {
            self.checked_original_consumer_step(parent)?;
            let original = owner
                .original
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?;
            let alias = original
                .source_alias
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            self.require_model_context_alias_resolution_quiescence(owner, parent, alias)?;
            self.readers
                .get(&parent.token)
                .ok_or(SemanticTransitionError::ObservationMismatch)?
        } else {
            self.checked_step(parent)?;
            self.checked_reader(parent)?
        };
        let storage = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let boundary = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .source_boundary
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&self.publication_issuer, &owner.source_issuer)
            || !Arc::ptr_eq(&parent.issuer, &owner.source_issuer)
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
            || !Arc::ptr_eq(storage, &owner.source_publication)
            || reader.acquisition_pending
            || reader.retirement_entered
            || reader.retirement_pending
            || self.pending
            || self.admitted_transition.is_some()
            || self.continuation_base.is_some()
            || !model_context_frozen_model_unchanged(&owner.source_header, &parent.header)
            || !model_context_model_header_unchanged(&boundary, &parent.header)
            || self.next_proposal != parent.header.proposal
            || reader.model_owners.len() != owner.models.len()
            || parent._model_owners.len() != owner.models.len()
            || self.model_owners.len() != owner.models.len()
            || reader
                .model_owners
                .iter()
                .zip(&owner.models)
                .any(|(actual, saved)| !Arc::ptr_eq(actual, saved))
            || parent
                ._model_owners
                .iter()
                .zip(&owner.models)
                .any(|(actual, saved)| !Arc::ptr_eq(actual, saved))
            || self
                .model_owners
                .values()
                .zip(&owner.models)
                .any(|(actual, saved)| !Arc::ptr_eq(actual, saved))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        parent.model_context_header()?;
        let slots = storage.model_slots_for_directory(&parent.directory)?;
        if slots.len() != owner.models.len() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        for (slots, saved) in slots.iter().zip(&owner.models) {
            let actual = storage.allocations[slots[0]]
                .model_owner()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if !Arc::ptr_eq(&actual, saved) || !Arc::ptr_eq(&actual.allocation, &saved.allocation) {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        }
        Ok(())
    }

    fn require_model_context_acknowledgement_chain(
        &self,
        owner: &Arc<ModelContextContinuationOwner>,
        parent: &SemanticPublishedLease,
        resume_alias: bool,
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        self.require_model_context_source_model_basis(owner, parent, resume_alias)?;
        let original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let anchor = model_context_header_identity(
            original
                .source_boundary
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let mut tail = anchor;
        let retained_parent = original.source_parent.map(|(_, identity)| identity);
        let mut retained_parent_known = retained_parent.is_none_or(|identity| identity == anchor);
        if (original.activating && original.source_parent != Some((parent.token, parent.identity)))
            || original.activated.is_some()
            || original.refused
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if let Some(alias) = &original.source_alias {
            self.require_original_model_context_source_alias_basis(owner, parent, alias)?;
        }
        for edge in &original.source_acknowledgements {
            let edge = edge.lock().map_err(|_| SemanticTransitionError::Poisoned)?;
            let successor = edge
                .successor
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if !edge.continuation.ptr_eq(&Arc::downgrade(owner))
                || !Arc::ptr_eq(&edge.issuer, &self.publication_issuer)
                || !Arc::ptr_eq(&edge.storage, &owner.source_publication)
                || edge.base != tail
                || model_context_header_identity(&edge.base_header) != edge.base
                || successor.base_word != edge.base.word
                || successor.publication_word
                    != ((edge.base.word >> 1) + 1) << 1 | ((edge.base.word & 1) ^ 1)
                || successor.sealed_epoch != successor.publication_word >> 1
                || successor.fuel.checked_add(1) != Some(edge.base_header.fuel)
                || !model_context_model_header_unchanged(&edge.base_header, successor)
                || edge.base_header.terminal != u64::from(edge.result_receipt.is_none()) * 2
                || successor.terminal != edge.base_header.terminal
                || Identity256::from_bytes(Sha256::digest(&edge.receipt).into())
                    != edge.receipt_digest
                || edge.receipt.get(57..89) != Some(edge.recipient.as_bytes().as_slice())
                || edge.models.len() != owner.models.len()
                || edge
                    .models
                    .iter()
                    .zip(&owner.models)
                    .any(|(actual, saved)| !Arc::ptr_eq(actual, saved))
                || edge.model_ranges.len()
                    != parent
                        .directory
                        .iter()
                        .filter(|range| matches!(range.role, 18..=25))
                        .count()
                || edge
                    .model_ranges
                    .iter()
                    .zip(
                        parent
                            .directory
                            .iter()
                            .filter(|range| matches!(range.role, 18..=25)),
                    )
                    .any(|(saved, actual)| !model_context_model_range_unchanged(saved, actual))
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            tail = model_context_header_identity(successor);
            retained_parent_known |= retained_parent == Some(tail);
        }
        if tail != parent.identity || !retained_parent_known {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(anchor)
    }

    pub(in crate::semantic_transition) fn require_model_context_acknowledgement_parent(
        &self,
        target: &Self,
        parent: &SemanticPublishedLease,
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        let owner = self
            .model_context_continuation
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&target.publication_issuer, &owner.target_issuer)
            || target
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.require_model_context_acknowledgement_chain(owner, parent, false)
    }

    pub(super) fn require_model_context_source_alias_parent(
        &self,
        target: &Self,
        parent: &SemanticPublishedLease,
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        let owner = self
            .model_context_continuation
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&target.publication_issuer, &owner.target_issuer)
            || target
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.require_model_context_acknowledgement_chain(owner, parent, true)
    }

    pub(in crate::semantic_transition) fn stage_model_context_acknowledgement(
        &self,
        parent: &SemanticPublishedLease,
        receipt: &[u8],
        receipt_digest: Identity256,
        recipient: Identity256,
        result_receipt: Option<&[u8]>,
    ) -> Result<Option<Arc<Mutex<OriginalModelContextAcknowledgement>>>, SemanticTransitionError>
    {
        let Some(owner) = self
            .model_context_continuation
            .as_ref()
            .filter(|owner| Arc::ptr_eq(&owner.source_issuer, &self.publication_issuer))
        else {
            return Ok(None);
        };
        self.require_model_context_acknowledgement_chain(owner, parent, false)?;
        if receipt.len() < 169
            || Identity256::from_bytes(Sha256::digest(receipt).into()) != receipt_digest
            || receipt.get(57..89) != Some(recipient.as_bytes().as_slice())
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original.activating || original.activated.is_some() || original.refused {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let restoration = original.checkpoint_restore.clone();
        drop(original);
        if let Some(restoration) = restoration {
            if !restoration
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?
                .report_completed
            {
                return Err(publication_input_error(
                    "source acknowledgement cannot precede completed context restoration",
                ));
            }
        }
        let count = parent
            .directory
            .iter()
            .filter(|range| matches!(range.role, 18..=25))
            .count();
        let bytes = size_of::<OriginalModelContextAcknowledgement>()
            .checked_add(size_of::<Mutex<OriginalModelContextAcknowledgement>>())
            .and_then(|bytes| bytes.checked_add(7 * size_of::<usize>()))
            .and_then(|bytes| bytes.checked_add(receipt.len()))
            .and_then(|bytes| bytes.checked_add(result_receipt.map_or(0, <[u8]>::len)))
            .and_then(|bytes| bytes.checked_add(count.checked_mul(size_of::<PublicationRange>())?))
            .and_then(|bytes| {
                bytes.checked_add(
                    owner
                        .models
                        .len()
                        .checked_mul(size_of::<Arc<ModelGenerationOwner>>())?,
                )
            })
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let reservation = Arc::new(budget.reserve_auxiliary(bytes)?);
        budget.require_host_allocation(bytes)?;
        let mut model_ranges = Vec::new();
        model_ranges
            .try_reserve_exact(count)
            .map_err(|error| runtime_error("acknowledgement model roster custody", error))?;
        model_ranges.extend(
            parent
                .directory
                .iter()
                .filter(|range| matches!(range.role, 18..=25))
                .copied(),
        );
        let edge = Arc::new(Mutex::new(OriginalModelContextAcknowledgement {
            continuation: Arc::downgrade(owner),
            issuer: Arc::clone(&self.publication_issuer),
            storage: Arc::clone(&owner.source_publication),
            base_token: parent.token,
            base: parent.identity,
            base_header: parent.header,
            successor: None,
            model_ranges,
            models: owner.models.clone(),
            receipt: Arc::from(receipt),
            receipt_digest,
            recipient,
            result_receipt: result_receipt.map(Arc::from),
            _reservation: reservation,
        }));
        let mut original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        original
            .source_acknowledgements
            .try_reserve_exact(1)
            .map_err(|error| runtime_error("acknowledgement lineage custody", error))?;
        original.source_acknowledgements.push(Arc::clone(&edge));
        Ok(Some(edge))
    }

    pub(in crate::semantic_transition) fn complete_model_context_acknowledgement(
        &self,
        parent: &SemanticPublishedLease,
        edge: &Arc<Mutex<OriginalModelContextAcknowledgement>>,
        successor: PublicationHeader,
    ) -> Result<(), SemanticTransitionError> {
        let owner = self
            .model_context_continuation
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !Arc::ptr_eq(&self.publication_issuer, &owner.source_issuer)
            || original
                .source_acknowledgements
                .last()
                .is_none_or(|actual| !Arc::ptr_eq(actual, edge))
            || original.activating
            || original.activated.is_some()
            || original.refused
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let mut edge = edge.lock().map_err(|_| SemanticTransitionError::Poisoned)?;
        if !edge.continuation.ptr_eq(&Arc::downgrade(owner))
            || !Arc::ptr_eq(&edge.issuer, &self.publication_issuer)
            || edge.base_token != parent.token
            || edge.base != parent.identity
            || successor.base_word != edge.base.word
            || successor.publication_word
                != ((edge.base.word >> 1) + 1) << 1 | ((edge.base.word & 1) ^ 1)
            || successor.sealed_epoch != successor.publication_word >> 1
            || successor.fuel.checked_add(1) != Some(edge.base_header.fuel)
            || !model_context_model_header_unchanged(&edge.base_header, &successor)
            || edge.successor.as_ref().is_some_and(|known| {
                model_context_header_identity(known) != model_context_header_identity(&successor)
            })
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        edge.successor = Some(successor);
        Ok(())
    }

    pub(in crate::semantic_transition) fn abandon_model_context_acknowledgement(
        &self,
        edge: &Arc<Mutex<OriginalModelContextAcknowledgement>>,
    ) -> Result<(), SemanticTransitionError> {
        let owner = self
            .model_context_continuation
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let mut original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original
            .source_acknowledgements
            .last()
            .is_none_or(|actual| !Arc::ptr_eq(actual, edge))
            || edge
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?
                .successor
                .is_some()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        original.source_acknowledgements.pop();
        Ok(())
    }

    pub(super) fn retain_original_model_context_continuation(
        &mut self,
        target: &mut Self,
        header: PublicationHeader,
        program: actor_refresh_program::OriginalProgramContextContinuation,
        models: Vec<Arc<ModelGenerationOwner>>,
    ) -> Result<SemanticModelContextContinuation, SemanticTransitionError> {
        let shared_catalogue = matches!(
            (&self.codebooks.model_catalogue,&target.codebooks.model_catalogue),
            (Some((source,0)),Some((future,1))) if Arc::ptr_eq(source,future)
        );
        if !shared_catalogue
            || !Arc::ptr_eq(self.provider.memory(), target.provider.memory())
            || self.domain.stream_id() != target.domain.stream_id()
            || self.model_context_continuation.is_some()
            || target.model_context_continuation.is_some()
            || target.publication.is_some()
            || target.initial_prefill.is_some()
            || target.prepared_segment.is_some()
            || target.task_evaluation_identity().is_none()
            || self.pending_replay_delivery.is_some()
            || models.is_empty()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if target
            .replay_material_budget
            .as_ref()
            .is_none_or(|other| !budget.same_pool(other))
        {
            return Err(publication_input_error(
                "model continuation requires its original shared material pool",
            ));
        }
        let bytes =
            size_of::<ModelContextContinuationOwner>()
                .checked_add(size_of::<OriginalModelContextContinuation>())
                .and_then(|bytes| {
                    bytes.checked_add(
                        size_of::<DormantModelContextParent>()
                            + 3 * size_of::<PreparedModelContextContinuationDevice>()
                            + 2 * size_of::<PublicationHeader>(),
                    )
                })
                .and_then(|bytes| {
                    bytes.checked_add(models.len().checked_mul(
                        size_of::<Arc<ModelGenerationOwner>>() + 2 * size_of::<usize>(),
                    )?)
                })
                .and_then(|bytes| bytes.checked_add(program.instruction().len()))
                .and_then(|bytes| {
                    bytes.checked_add(
                        usize::try_from(storage.contract_value.terminal_token_count)
                            .ok()?
                            .checked_mul(size_of::<u64>())?,
                    )
                })
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = budget.reserve_auxiliary(bytes)?;
        budget.require_host_allocation(bytes)?;
        let instruction = Arc::from(program.instruction());
        let owner = Arc::new(ModelContextContinuationOwner {
            source_issuer: Arc::clone(&self.publication_issuer),
            target_issuer: Arc::clone(&target.publication_issuer),
            source_publication: storage,
            _source_provider: Arc::clone(&self.provider),
            models,
            instruction,
            source_header: header,
            program: Mutex::new(program),
            original: Mutex::new(OriginalModelContextContinuation {
                parent: None,
                command: OriginalNativeCommand::new(&target.domain)?,
                source_parent: None,
                activation_result: None,
                activation_work: None,
                proof_read: None,
                header_read: None,
                observed_proof: None,
                observed_header: None,
                activation_rng: None,
                native_work: None,
                native_claimed: false,
                prefix_guard: None,
                prefix_guard_stream: None,
                source_work_attached: false,
                initial_prefill_scope: None,
                activating: false,
                activated: None,
                refused: false,
                poisoned: false,
                checkpoint_restore: None,
                source_alias: None,
                source_acknowledgements: Vec::new(),
                source_boundary: Some(header),
            }),
            _reservation: Arc::new(reservation),
        });
        self.model_context_continuation = Some(Arc::clone(&owner));
        target.model_context_continuation = Some(Arc::clone(&owner));
        Ok(SemanticModelContextContinuation { inner: owner })
    }

    pub fn freeze_restored_model_context_continuation(
        &mut self,
        target: &mut Self,
        parent: &SemanticPublishedLease,
        native_source: &mut SemanticReplayLoadedMaterial,
        assignments: &SemanticReplayLoadedMaterial,
        mapping: &SemanticReplayLoadedMaterial,
    ) -> Result<SemanticModelContextContinuation, SemanticTransitionError> {
        self.checked_step(parent)?;
        self.checked_reader(parent)?;
        let header = parent.model_context_header()?;
        if self.next_proposal != header.proposal
            || self.prepared_segment.is_some()
            || self.pending_training_materialization.is_some()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let storage = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let slots = storage.model_slots_for_directory(&parent.directory)?;
        let models = slots
            .iter()
            .map(|slots| {
                storage.allocations[slots[0]]
                    .model_owner()
                    .ok_or(SemanticTransitionError::ObservationMismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let program = self.freeze_restored_original_program_context_continuation(
            target,
            parent,
            native_source,
            assignments,
            mapping,
        )?;
        self.retain_original_model_context_continuation(target, header, program, models)
    }

    pub fn prepare_model_context_checkpoint_attachment_restore(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        source: &Self,
        parent: &SemanticPublishedLease,
        native_source: &mut SemanticReplayLoadedMaterial,
        loaded_target: SemanticReplayLoadedMaterial,
        loaded_prefill: SemanticReplayLoadedMaterial,
    ) -> Result<(u64, u64, u64, u64, u64), SemanticTransitionError> {
        let owner = &continuation.inner;
        source.checked_step(parent)?;
        source.checked_original_consumer_step(parent)?;
        if !Arc::ptr_eq(&source.publication_issuer, &owner.source_issuer)
            || !Arc::ptr_eq(&self.publication_issuer, &owner.target_issuer)
            || !parent.active
            || source
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
            || source
                .publication
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, &owner.source_publication))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let target_digest =
            Identity256::from_bytes(Sha256::digest(loaded_target.as_bytes()).into());
        let source_digest =
            Identity256::from_bytes(Sha256::digest(native_source.as_bytes()).into());
        let prefill_digest =
            Identity256::from_bytes(Sha256::digest(loaded_prefill.as_bytes()).into());
        let existing = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .checkpoint_restore
            .clone();
        let restoration = if let Some(existing) = existing {
            existing
        } else {
            owner.require_imported_target(self)?;
            let source_material = Arc::new(DormantModelMaterialSource::authenticate(
                source,
                parent,
                native_source,
            )?);
            let source_header = native_source.complete()?.bank.header;
            let checkpoint_source_parent = SemanticPublishedIdentity {
                instance: source_header.instance,
                word: source_header.publication_word,
                logical_digest: source_header.logical_digest,
                state_digest: source_header.state_digest,
            };
            let budget = self
                .replay_material_budget
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let bytes = (size_of::<OriginalDormantContextRestore>()
                + size_of::<Mutex<OriginalDormantContextRestore>>()
                + 2 * size_of::<usize>()) as u64;
            let reservation = Arc::new(budget.reserve_auxiliary(bytes)?);
            budget.require_host_allocation(bytes)?;
            let material =
                DormantPublicationMaterial::decode(loaded_target, source_material, budget)?;
            loaded_prefill.require_budget(budget)?;
            let source_geometry = dormant_prefill_source_geometry(&material)?;
            let native_ceilings =
                self.dormant_model_context_restore_native_ceilings(&material.material)?;
            let restoration = Arc::new(Mutex::new(OriginalDormantContextRestore {
                source_parent: parent.identity,
                source_token: parent.token,
                source_digest,
                target_digest,
                prefill_digest,
                checkpoint_source_parent,
                source_geometry,
                material: Some(material),
                prefill_section: Some(loaded_prefill),
                prefill: None,
                allowance: None,
                work: None,
                native_ceilings,
                entered: false,
                proof: None,
                proof_write: None,
                completed: false,
                report_completed: false,
                poisoned: false,
                _reservation: reservation,
            }));
            let mut original = owner
                .original
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?;
            if original.checkpoint_restore.is_some()
                || original.parent.is_some()
                || original.activating
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            original.checkpoint_restore = Some(Arc::clone(&restoration));
            restoration
        };
        let restoration = restoration
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if restoration.source_parent != parent.identity
            || restoration.source_token != parent.token
            || restoration.source_digest != source_digest
            || restoration.target_digest != target_digest
            || restoration.prefill_digest != prefill_digest
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(restoration.source_geometry)
    }

    pub fn model_context_checkpoint_attachment_restore_source_geometry(
        &self,
        continuation: &SemanticModelContextContinuation,
    ) -> Result<(u64, u64, u64, u64, u64), SemanticTransitionError> {
        if !Arc::ptr_eq(&self.publication_issuer, &continuation.inner.target_issuer)
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !Arc::ptr_eq(owner, &continuation.inner))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let restoration = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .checkpoint_restore
            .clone()
            .ok_or(SemanticTransitionError::NotBound)?;
        let restoration = restoration
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        Ok(restoration.source_geometry)
    }

    pub fn prepare_model_context_checkpoint_attachment_restore_work(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        source: &mut Self,
        parent: &SemanticPublishedLease,
        source_layout: SemanticInitialSourceLayout,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        let owner = &continuation.inner;
        source.checked_step(parent)?;
        source.checked_original_consumer_step(parent)?;
        if !parent.active
            || !Arc::ptr_eq(&source.publication_issuer, &owner.source_issuer)
            || !Arc::ptr_eq(&self.publication_issuer, &owner.target_issuer)
            || source
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
            || source
                .publication
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, &owner.source_publication))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let restoration = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .checkpoint_restore
            .clone()
            .ok_or(SemanticTransitionError::NotBound)?;
        let mut restoration = restoration
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if restoration.source_parent != parent.identity || restoration.source_token != parent.token
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if let Some(prefill) = &restoration.prefill {
            prefill.require_source_layout(&source_layout)?;
        } else {
            let section = restoration
                .prefill_section
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let prefill = OriginalRestoredModelContextPrefill::prepare(
                continuation,
                self,
                (parent.identity, restoration.checkpoint_source_parent),
                restoration.source_digest,
                restoration.target_digest,
                SemanticReplayLoadedMaterial {
                    bytes: section.bytes.clone(),
                    decoded: None,
                },
                restoration
                    .material
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?,
                source_layout,
            )?;
            let (prefill_dma, prefill_total) = prefill.native_ceilings()?;
            let (mut dma, mut total) = restoration.native_ceilings;
            native_work_bound::add_native_work_ceiling(&mut dma, prefill_dma)?;
            native_work_bound::add_native_work_ceiling(&mut total, prefill_total)?;
            restoration.native_ceilings = (dma, total);
            restoration.prefill = Some(prefill);
            restoration.prefill_section = None;
        }
        if restoration.completed {
            return restoration
                .work
                .clone()
                .ok_or(SemanticTransitionError::NotBound);
        }
        if restoration.work.is_none() {
            let (dma, total) = restoration.native_ceilings;
            let work = source.prepare_model_context_material_work(
                self,
                continuation,
                parent,
                true,
                dma,
                total,
            )?;
            let allowance =
                self.stage_model_context_material_work_allowance(continuation, &work, true)?;
            restoration.allowance = Some(allowance);
            restoration.work = Some(work);
        }
        let work = restoration
            .work
            .as_ref()
            .expect("original dormant restore report")
            .clone();
        Ok(work)
    }

    pub fn model_context_checkpoint_attachment_restore_work(
        &self,
        continuation: &SemanticModelContextContinuation,
        source: &Self,
        parent: &SemanticPublishedLease,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        source.checked_step(parent)?;
        source.checked_original_consumer_step(parent)?;
        let owner = &continuation.inner;
        if !parent.active
            || !Arc::ptr_eq(&self.publication_issuer, &owner.target_issuer)
            || !Arc::ptr_eq(&source.publication_issuer, &owner.source_issuer)
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
            || source
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
            || source
                .publication
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, &owner.source_publication))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let restoration = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .checkpoint_restore
            .clone()
            .ok_or(SemanticTransitionError::NotBound)?;
        let restoration = restoration
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if restoration.source_parent != parent.identity || restoration.source_token != parent.token
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        restoration
            .work
            .clone()
            .ok_or(SemanticTransitionError::NotBound)
    }

    pub fn model_context_checkpoint_attachment_restore_declaration(
        &self,
        continuation: &SemanticModelContextContinuation,
        source: &Self,
        parent: &SemanticPublishedLease,
    ) -> Result<SemanticModelContextPrefillRestoreDeclaration, SemanticTransitionError> {
        self.model_context_checkpoint_attachment_restore_work(continuation, source, parent)?;
        let restoration = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .checkpoint_restore
            .clone()
            .ok_or(SemanticTransitionError::NotBound)?;
        let restoration = restoration
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        restoration
            .prefill
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .declaration()
    }

    pub fn restore_model_context_checkpoint_attachment(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        source: &mut Self,
        parent: &SemanticPublishedLease,
        source_layout: SemanticInitialSourceLayout,
    ) -> Result<SemanticRestoredModelContextPrefill, SemanticTransitionError> {
        let work = self.prepare_model_context_checkpoint_attachment_restore_work(
            continuation,
            source,
            parent,
            source_layout,
        )?;
        let owner = &continuation.inner;
        let restoration = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .checkpoint_restore
            .clone()
            .ok_or(SemanticTransitionError::NotBound)?;
        let mut restoration = restoration
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if restoration.completed {
            let proof = restoration.known_prefill_proof()?;
            drop(restoration);
            self.resolve_restored_model_context_prefill_consumer_commands(continuation, &proof)?;
            return Ok(proof);
        }
        let allowance = Arc::clone(
            restoration
                .allowance
                .as_ref()
                .expect("original dormant restore allowance"),
        );
        restoration
            .prefill
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .require_factory_geometry()?;
        source.initialize_model_context_material_work(self, continuation, parent, &work)?;
        let actual_allowance =
            self.require_dormant_model_context_restore_work(continuation, &work)?;
        if !Arc::ptr_eq(&actual_allowance, &allowance) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if !restoration.entered {
            let material = restoration
                .material
                .take()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let minimum_generation = material.source.generation;
            restoration.entered = true;
            self.restore_dormant_model_context_material(
                material.material,
                restoration.target_digest,
                continuation,
                &owner.models,
                minimum_generation,
                &work,
            )?;
        } else {
            if !self.state_material_restore_started() {
                return Err(publication_input_error(
                    "original dormant restore did not retain its native owner",
                ));
            }
            if self.resolve_state_material_restore()?.is_some() {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        }
        let (bank, rng, terminals, role_counts) = self.dormant_state_material_restore_recipe()?;
        let source_digest = restoration.source_digest;
        let target_digest = restoration.target_digest;
        self.resolve_restored_model_context_prefill(
            continuation,
            parent.identity,
            source_digest,
            target_digest,
            &work,
            restoration
                .prefill
                .as_mut()
                .ok_or(SemanticTransitionError::NotBound)?,
        )?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let program = owner
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let header = parent.model_context_header()?;
        if bank.header.abi != 0
            || bank.header.publication_word != 0
            || bank.header.instance != storage.instance
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if restoration.proof.is_none() {
            let proof = allocate_publication(&self.provider, 1)?;
            let input = PreparedModelContextContinuationDevice {
                abi: 1,
                status: 1,
                source_control: owner.source_publication.control.device_ptr_value(),
                target_control: storage.control.device_ptr_value(),
                minimum_generation: header.model_generation,
                stream_serial: header.stream_serial,
                family_id: header.family_id,
                proposal: program.expected_source_proposal(),
                source_instance: owner.source_publication.instance,
                target_instance: storage.instance,
                model_geometry_digest: header.model_geometry_digest,
                model_numerical_digest: header.model_numerical_digest,
                ..PreparedModelContextContinuationDevice::default()
            };
            let write = state_restoration::OriginalDeviceWrite::new_with_owner(
                &self.stream,
                &[input],
                proof.view(),
                true,
                Some(Arc::clone(&owner._reservation)),
            )?;
            restoration.proof = Some(proof);
            restoration.proof_write = Some(write);
        }
        let OriginalDormantContextRestore {
            proof_write,
            poisoned,
            ..
        } = &mut *restoration;
        proof_write
            .as_mut()
            .expect("original dormant continuation proof")
            .resolve_with_dma(
                &self.domain,
                &self.stream,
                &self.provider,
                poisoned,
                true,
                Some(&allowance),
            )?;
        let mut original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original.parent.is_some() || original.activating || original.refused {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if original
            .source_parent
            .is_some_and(|actual| actual != (parent.token, parent.identity))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let prefill = restoration
            .prefill
            .as_mut()
            .ok_or(SemanticTransitionError::NotBound)?
            .take_verified_stage()?;
        original.initial_prefill_scope = Some(Arc::clone(&prefill.scope));
        original.source_parent = Some((parent.token, parent.identity));
        original.parent = Some(DormantModelContextParent {
            storage,
            bank,
            rng,
            terminals,
            role_counts,
            prefill: Some(prefill),
            proof: restoration
                .proof
                .take()
                .expect("original dormant proof storage"),
            initialized: true,
        });
        restoration.completed = true;
        drop(original);
        drop(program);
        restoration.known_prefill_proof()
    }

    pub fn finish_model_context_checkpoint_attachment_restore(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        source: &mut Self,
        parent: &SemanticPublishedLease,
    ) -> Result<SemanticRestoredModelContextPrefill, SemanticTransitionError> {
        self.model_context_checkpoint_attachment_restore_work(continuation, source, parent)?;
        continuation.inner.require_dormant_parent(self)?;
        let restoration = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .checkpoint_restore
            .clone()
            .ok_or(SemanticTransitionError::NotBound)?;
        let mut restoration = restoration
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if restoration.source_parent != parent.identity || restoration.source_token != parent.token
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let proof = restoration.known_prefill_proof()?;
        self.require_restored_model_context_prefill_consumers_complete(continuation, &proof)?;
        if !restoration.report_completed {
            source.finish_model_context_material_work(
                self,
                continuation,
                parent,
                restoration
                    .work
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?,
            )?;
            restoration.report_completed = true;
        }
        restoration.completed_prefill_proof()
    }

    pub fn require_restored_model_context_source_basis(
        &self,
        continuation: &SemanticModelContextContinuation,
        source: &Self,
        source_parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        let owner = &continuation.inner;
        source.require_model_context_source_parent(continuation, self, source_parent)?;
        let anchor = source.require_model_context_acknowledgement_parent(self, source_parent)?;
        let restoration = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .checkpoint_restore
            .clone()
            .ok_or(SemanticTransitionError::NotBound)?;
        let restoration = restoration
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if restoration.source_parent != anchor {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let proof = restoration.known_prefill_proof()?;
        self.require_restored_model_context_prefill(continuation, &proof)
    }

    pub fn require_restored_model_context_source_alias(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        target: &mut Self,
        proof: &SemanticRestoredModelContextPrefill,
        parent: &SemanticPublishedLease,
        tensors: Vec<SemanticTensorInput>,
        allocations: Vec<SemanticTensorInput>,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        target.require_restored_model_context_source_basis(continuation, self, parent)?;
        target.require_restored_model_context_prefill(continuation, proof)?;
        let restoration = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .checkpoint_restore
            .clone()
            .ok_or(SemanticTransitionError::NotBound)?;
        let restoration = restoration
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let original_proof = restoration.known_prefill_proof()?;
        if !proof.same_handle(&original_proof) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let report_completed = restoration.report_completed;
        if !report_completed {
            target.require_dormant_model_context_restore_work(
                continuation,
                restoration
                    .work
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?,
            )?;
        }
        drop(restoration);
        if report_completed {
            self.require_model_context_source_alias_inputs(
                continuation,
                parent,
                tensors,
                allocations,
                consumer_stream,
            )?;
            return self.resolve_model_context_source_alias(continuation, target, parent);
        }
        self.stage_model_context_source_alias(
            continuation,
            parent,
            tensors,
            allocations,
            consumer_stream,
        )?;
        self.resolve_model_context_source_alias(continuation, target, parent)
    }

    pub(super) fn model_context_checkpoint_native_ceilings(
        &self,
        continuation: &SemanticModelContextContinuation,
    ) -> Result<([u64; 9], [u64; 9]), SemanticTransitionError> {
        continuation.inner.require_dormant_parent(self)?;
        let storage = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let graph = self
            .graph
            .transition_root_export_geometry()
            .map_err(SemanticTransitionError::Semantic)?;
        let mut bytes = size_of::<PublicationBank>()
            .checked_add(size_of::<PublicationControl>())
            .and_then(|bytes| {
                bytes.checked_add(
                    storage
                        .role_counts
                        .len()
                        .checked_mul(size_of::<PublicationRoleCount>())?,
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(storage.terminals.len().checked_mul(size_of::<u64>())?)
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        for range in storage.bank_templates[0]
            .iter()
            .filter(|range| !matches!(range.role, 18..=25))
        {
            bytes = bytes
                .checked_add(
                    usize::try_from(range.length_bytes)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                )
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let mut dma = graph.dma_ceiling;
        let range_reads = storage.bank_templates[0]
            .iter()
            .filter(|range| !matches!(range.role, 18..=25) && range.length_bytes != 0)
            .count();
        dma[0] = dma[0]
            .checked_add(
                3 + u64::from(!storage.terminals.is_empty())
                    + u64::try_from(range_reads)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            )
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        dma[8] = dma[8]
            .checked_add(
                u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            )
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut total = dma;
        native_work_bound::add_native_work_ceiling(&mut total, graph.kernel_ceiling)?;
        let (prefill_dma, prefill_total) =
            self.dormant_prefill_checkpoint_native_ceilings(continuation)?;
        native_work_bound::add_native_work_ceiling(&mut dma, prefill_dma)?;
        native_work_bound::add_native_work_ceiling(&mut total, prefill_total)?;
        Ok((dma, total))
    }

    pub fn prepare_model_context_checkpoint_work(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        target: &mut Self,
        parent: &SemanticPublishedLease,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        let (dma, total) = target.model_context_checkpoint_native_ceilings(continuation)?;
        self.prepare_model_context_material_work(target, continuation, parent, false, dma, total)
    }

    pub fn model_context_checkpoint_attachment_allocation_bound(
        &self,
        continuation: &SemanticModelContextContinuation,
        target: &Self,
        parent: &SemanticPublishedLease,
    ) -> Result<usize, SemanticTransitionError> {
        self.require_model_context_checkpoint_source(continuation, target, parent)?;
        let owner = &continuation.inner;
        let program = owner
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let storage = target
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let mapping = self.original_program_context_continuation_mapping(&program)?;
        let assignments = self.actor_refresh_loaded_assignment_material()?;
        let bound = target
            .publication_material_observation_allocation_bound_core(
                storage,
                storage.bank_templates[0].len(),
                storage.bank_templates[0]
                    .iter()
                    .map(|range| Ok((range.role, range.length_bytes))),
                true,
                &target.learning_phases,
                true,
            )?
            .checked_add(target.dormant_prefill_checkpoint_allocation_bound(continuation)?)
            .and_then(|bytes| {
                bytes.checked_add(size_of::<SemanticModelContextCheckpointAttachment>() as u64)
            })
            .and_then(|bytes| bytes.checked_add(mapping.as_bytes().len() as u64))
            .and_then(|bytes| bytes.checked_add(assignments.as_bytes().len() as u64))
            .and_then(|bytes| bytes.checked_add(4 * size_of::<u64>() as u64))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        usize::try_from(bound).map_err(|_| SemanticTransitionError::GenerationExhausted)
    }

    pub fn require_model_context_checkpoint_source(
        &self,
        continuation: &SemanticModelContextContinuation,
        target: &Self,
        parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.require_model_context_source_parent(continuation, target, parent)
    }

    pub fn require_model_context_checkpoint_attachment_presence(
        &self,
        parent: &SemanticPublishedLease,
        present: bool,
        private: Option<&SemanticPrivateModelContextCheckpointSave>,
        cancellation: Option<&SemanticPreparedSegmentNonSubmission>,
    ) -> Result<(), SemanticTransitionError> {
        if let Some(cancellation) = cancellation {
            self.prepared_segment
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .require_non_submission(cancellation)?;
            self.checked_original_consumer_step(parent)?;
            if !parent.active {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        } else {
            self.checked_step(parent)?;
        }
        self.checked_reader(parent)?;
        let expected = if let Some(private) = private {
            if self.model_context_continuation.is_some() {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            self.require_restored_model_context_checkpoint_boundary(parent, Some(private))?;
            false
        } else if let Some(owner) = &self.model_context_continuation {
            if Arc::ptr_eq(&self.publication_issuer, &owner.source_issuer) {
                self.require_model_context_acknowledgement_chain(owner, parent, false)?;
                let original = owner
                    .original
                    .lock()
                    .map_err(|_| SemanticTransitionError::Poisoned)?;
                if original.activating || original.activated.is_some() || original.refused {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                let restoration = original.checkpoint_restore.clone();
                drop(original);
                if let Some(restoration) = restoration {
                    if !restoration
                        .lock()
                        .map_err(|_| SemanticTransitionError::Poisoned)?
                        .report_completed
                    {
                        return Err(publication_input_error(
                            "checkpoint requires completed original context restoration work",
                        ));
                    }
                }
                let program = owner
                    .program
                    .lock()
                    .map_err(|_| SemanticTransitionError::Poisoned)?;
                self.require_original_program_context_checkpoint(&program, parent)?;
                true
            } else if Arc::ptr_eq(&self.publication_issuer, &owner.target_issuer) {
                let original = owner
                    .original
                    .lock()
                    .map_err(|_| SemanticTransitionError::Poisoned)?;
                if original.activated.is_none() || original.activating || original.refused {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                false
            } else {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        } else {
            match &self.codebooks.model_catalogue {
                None => {}
                Some((_, 0)) => self.require_initial_model_context_checkpoint_boundary(parent)?,
                Some((_, 1)) => {
                    self.require_restored_model_context_checkpoint_boundary(parent, None)?
                }
                Some(_) => return Err(SemanticTransitionError::ObservationMismatch),
            }
            false
        };
        if present != expected {
            return Err(publication_input_error(
                "checkpoint attachment presence differs from its native source boundary",
            ));
        }
        Ok(())
    }

    pub fn require_model_context_source_parent(
        &self,
        continuation: &SemanticModelContextContinuation,
        target: &Self,
        parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        let owner = &continuation.inner;
        owner.require_dormant_parent(target)?;
        target.require_model_context_numerical_contract(continuation)?;
        self.require_model_context_acknowledgement_parent(target, parent)?;
        let program = owner
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        self.require_original_program_context_checkpoint(&program, parent)?;
        target.validate_transfer_original_program_context_continuation(self, &program, parent)?;
        if self
            .model_context_continuation
            .as_ref()
            .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub fn stage_model_context_checkpoint_attachment(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        target: &mut Self,
        parent: &SemanticPublishedLease,
        native_source: &mut SemanticReplayLoadedMaterial,
        work: &SemanticColdModelWork,
    ) -> Result<SemanticModelContextCheckpointAttachment, SemanticTransitionError> {
        let owner = &continuation.inner;
        let program = owner
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        self.require_original_program_context_checkpoint(&program, parent)?;
        target.validate_transfer_original_program_context_continuation(self, &program, parent)?;
        owner.require_dormant_parent(target)?;
        if self
            .model_context_continuation
            .as_ref()
            .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let allowance =
            target.stage_model_context_material_work_allowance(continuation, work, false)?;
        if target
            .graph
            .cold_work_allowance()
            .as_ref()
            .is_none_or(|actual| !Arc::ptr_eq(actual, &allowance))
        {
            return Err(publication_input_error(
                "dormant attachment requires its actual counted graph readback report",
            ));
        }
        let source = Arc::new(DormantModelMaterialSource::authenticate(
            self,
            parent,
            native_source,
        )?);
        let storage = Arc::clone(
            target
                .publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let budget = target
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let prefill_bound = target.dormant_prefill_checkpoint_allocation_bound(continuation)?;
        let requested = target
            .publication_material_observation_allocation_bound_core(
                &storage,
                storage.bank_templates[0].len(),
                storage.bank_templates[0]
                    .iter()
                    .map(|range| Ok((range.role, range.length_bytes))),
                true,
                &target.learning_phases,
                true,
            )?
            .checked_add(size_of::<SemanticModelContextCheckpointAttachment>() as u64)
            .and_then(|bytes| bytes.checked_add(prefill_bound))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        budget.require_host_allocation(requested)?;
        let reservation = Arc::new(budget.reserve_owned(requested, requested)?);
        let mapping = self.original_program_context_continuation_mapping(&program)?;
        let assignments = self.actor_refresh_loaded_assignment_material()?;
        let mut bank = target.stage_publication_read_with_owner(
            storage.banks[0].view(),
            Some(Arc::clone(&reservation)),
        )?;
        let mut control = target.stage_publication_read_with_owner(
            storage.control.view(),
            Some(Arc::clone(&reservation)),
        )?;
        let mut counts = target.stage_publication_read_with_owner(
            storage.role_counts.view(),
            Some(Arc::clone(&reservation)),
        )?;
        bank.dma_allowance = Some(Arc::downgrade(&allowance));
        control.dma_allowance = Some(Arc::downgrade(&allowance));
        counts.dma_allowance = Some(Arc::downgrade(&allowance));
        let terminals = if storage.terminals.is_empty() {
            None
        } else {
            let mut read = target.stage_publication_read_with_owner(
                storage.terminals.view(),
                Some(Arc::clone(&reservation)),
            )?;
            read.dma_allowance = Some(Arc::downgrade(&allowance));
            Some(read)
        };
        let mut ranges = Vec::new();
        ranges
            .try_reserve_exact(storage.bank_templates[0].len())
            .map_err(|error| runtime_error("dormant context readback custody", error))?;
        for range in &storage.bank_templates[0] {
            if matches!(range.role, 18..=25) {
                continue;
            }
            let allocation = storage
                .allocations
                .get(
                    usize::try_from(range.storage_slot)
                        .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
                )
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let begin = usize::try_from(range.offset_bytes)
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
            let end = begin
                .checked_add(
                    usize::try_from(range.length_bytes)
                        .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
                )
                .filter(|end| *end <= allocation.len())
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let read = if begin == end {
                None
            } else {
                let mut read = target.stage_publication_read_with_owner(
                    allocation.slice()?.view().slice(begin..end),
                    Some(Arc::clone(&reservation)),
                )?;
                read.dma_allowance = Some(Arc::downgrade(&allowance));
                Some(read)
            };
            ranges.push((*range, allocation.len() - begin, read));
        }
        let prefill = target.stage_dormant_prefill_checkpoint(
            continuation,
            parent.identity,
            Arc::clone(&reservation),
            Arc::clone(&allowance),
        )?;
        let graph = target
            .graph
            .prepare_transition_root_export_with_owner(target.root, Some(reservation.clone()))
            .map_err(SemanticTransitionError::Semantic)?;
        Ok(SemanticModelContextCheckpointAttachment {
            source_issuer: Arc::clone(&self.publication_issuer),
            target_issuer: Arc::clone(&target.publication_issuer),
            continuation: Arc::clone(owner),
            source_parent: parent.identity,
            storage,
            source,
            bank,
            control,
            counts,
            terminals,
            ranges,
            graph,
            material: None,
            loaded: None,
            mapping,
            assignments,
            prefill,
            loaded_prefill: None,
            allowance,
            work: work.clone(),
            poisoned: false,
            _reservation: reservation,
        })
    }

    pub fn resolve_model_context_checkpoint_attachment(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        target: &mut Self,
        parent: &SemanticPublishedLease,
        original: &mut SemanticModelContextCheckpointAttachment,
    ) -> Result<
        (
            SemanticReplayLoadedMaterial,
            SemanticReplayLoadedMaterial,
            SemanticReplayLoadedMaterial,
            SemanticReplayLoadedMaterial,
        ),
        SemanticTransitionError,
    > {
        if original.loaded.is_some() && original.loaded_prefill.is_some() {
            self.finish_model_context_material_work(target, continuation, parent, &original.work)?;
            let retained = |loaded: &SemanticReplayLoadedMaterial| SemanticReplayLoadedMaterial {
                bytes: loaded.bytes.clone(),
                decoded: None,
            };
            return Ok((
                retained(original.loaded.as_ref().expect("original dormant encoding")),
                retained(&original.mapping),
                retained(&original.assignments),
                retained(
                    original
                        .loaded_prefill
                        .as_ref()
                        .expect("original dormant prefill encoding"),
                ),
            ));
        }
        self.checked_step(parent)?;
        self.checked_original_consumer_step(parent)?;
        if !Arc::ptr_eq(&self.publication_issuer, &original.source_issuer)
            || !Arc::ptr_eq(&target.publication_issuer, &original.target_issuer)
            || !Arc::ptr_eq(&continuation.inner, &original.continuation)
            || parent.identity != original.source_parent
            || target
                .publication
                .as_ref()
                .is_none_or(|storage| !Arc::ptr_eq(storage, &original.storage))
            || target
                .stage_model_context_material_work_allowance(continuation, &original.work, false)
                .as_ref()
                .map_or(true, |allowance| {
                    !Arc::ptr_eq(allowance, &original.allowance)
                })
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let assignments = self.actor_refresh_loaded_assignment_material()?;
        if Sha256::digest(assignments.as_bytes()) != Sha256::digest(original.assignments.as_bytes())
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.initialize_model_context_material_work(target, continuation, parent, &original.work)?;
        if original.material.is_none() {
            let bank = target
                .resolve_publication_read_with_poison(&mut original.bank, &mut original.poisoned)?
                [0];
            let control = target.resolve_publication_read_with_poison(
                &mut original.control,
                &mut original.poisoned,
            )?[0];
            if bank.header.abi != 0
                || control.abi != 1
                || control.word != 0
                || control.instance != original.storage.instance
                || control.reader_counts != [0, 0]
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let counts = target.resolve_publication_read_with_poison(
                &mut original.counts,
                &mut original.poisoned,
            )?;
            let mut role_counts = [0; PUBLICATION_ROLE_COUNT];
            if counts.len() != role_counts.len() {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            for (index, count) in counts.iter().enumerate() {
                if count.role != index as u64 + 1 {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                role_counts[index] = count.count;
            }
            let terminals = if let Some(read) = &mut original.terminals {
                target.resolve_publication_read_with_poison(read, &mut original.poisoned)?
            } else {
                Vec::new()
            };
            let graph = target
                .graph
                .resolve_transition_root_export(&mut original.graph)
                .map_err(SemanticTransitionError::Semantic)?;
            let mut ranges = Vec::new();
            ranges
                .try_reserve_exact(original.storage.bank_templates[0].len())
                .map_err(|error| runtime_error("dormant context material custody", error))?;
            for range in &original.storage.bank_templates[0] {
                let (capacity, bytes) = if matches!(range.role, 18..=25) {
                    let (allocation, offset) = original
                        .source
                        .model_memory
                        .location(range.role, range.index)?;
                    let end = offset
                        .checked_add(
                            usize::try_from(range.length_bytes)
                                .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
                        )
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    let backing = &original.source.model_allocations[allocation];
                    (backing.len() - offset, backing.slice_bytes(offset..end)?)
                } else {
                    let (_, capacity, read) = original
                        .ranges
                        .iter_mut()
                        .find(|(candidate, _, _)| {
                            (candidate.role, candidate.index) == (range.role, range.index)
                        })
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    let bytes = if let Some(read) = read {
                        target.resolve_publication_read_with_poison(read, &mut original.poisoned)?
                    } else {
                        Vec::new()
                    };
                    (
                        *capacity,
                        PublicationMaterialBytes::readback(
                            bytes,
                            Some(Arc::clone(&original._reservation)),
                        ),
                    )
                };
                ranges.push(OwnedPublicationMaterialRange {
                    range: *range,
                    capacity,
                    bytes,
                });
            }
            let mut contract = original.storage.contract_value;
            contract.semantic_owner = 0;
            contract.terminal_tokens = 0;
            contract.role_counts = 0;
            original.material = Some(DormantPublicationMaterial::from_readback(
                OwnedPublicationMaterial {
                    bank,
                    contract,
                    role_counts,
                    terminals,
                    layouts: original.storage.layouts.clone(),
                    model_memory: original.storage.model_memory.clone(),
                    model_allocations: original.source.model_allocations.clone(),
                    ranges,
                    graph,
                    learning_phases: target.learning_phases.clone(),
                },
                Arc::clone(&original.source),
            )?);
        }
        if original.loaded.is_none() {
            original.loaded = Some(
                original
                    .material
                    .as_ref()
                    .expect("original dormant material")
                    .encode(&original._reservation)?,
            );
        }
        if original.loaded_prefill.is_none() {
            let target_digest = Identity256::from_bytes(
                Sha256::digest(
                    original
                        .loaded
                        .as_ref()
                        .expect("original dormant encoding")
                        .as_bytes(),
                )
                .into(),
            );
            original.loaded_prefill = Some(target.resolve_dormant_prefill_checkpoint(
                continuation,
                &mut original.prefill,
                original.source.native_digest,
                target_digest,
            )?);
        }
        self.finish_model_context_material_work(target, continuation, parent, &original.work)?;
        let retained = |loaded: &SemanticReplayLoadedMaterial| SemanticReplayLoadedMaterial {
            bytes: loaded.bytes.clone(),
            decoded: None,
        };
        Ok((
            retained(original.loaded.as_ref().expect("original dormant encoding")),
            retained(&original.mapping),
            retained(&original.assignments),
            retained(
                original
                    .loaded_prefill
                    .as_ref()
                    .expect("original dormant prefill encoding"),
            ),
        ))
    }
}
