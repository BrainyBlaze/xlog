//! Authenticate the original Proposals which supplied a delivered coding interval.

use super::*;

/// Read the original action witness, rather than interpreting the episode's JSON
/// projection as a second source of native execution facts.
fn proposal_receipts(
    replay: &SemanticReplayMaterial,
    original: &SemanticResolvedReplayMaterial,
    action: &SemanticReplayLoadedMaterial,
) -> Result<(DeviceState, [SemanticTransitionReceipt; COMPONENT_COUNT]), SemanticTransitionError> {
    if replay.kind != SemanticTransitionKind::Proposal
        || !Arc::ptr_eq(&replay.proof_owner, &original.proof_owner)
        || original.digest != replay.predecessor.digest
    {
        return Err(publication_input_error(
            "coding contributor requires its original resolved Proposal material",
        ));
    }
    let budget = replay.require_loaded_budget(action)?;
    if original.material.bank.header != replay.predecessor.bank.header
        || !budget.same_pool(
            action
                .bytes
                .budget()
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
        )
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let decoded = super::super::completed_action::decode(action.as_ref())?;
    if decoded.predecessor != replay.predecessor_identity()
        || decoded.successor != replay.successor_identity()
    {
        return Err(publication_input_error(
            "coding contributor action witness changed its original publication pair",
        ));
    }
    let state = decoded.state;
    let receipts = decoded.receipts;
    let codebook = original
        .material
        .ranges
        .iter()
        .find(|range| (range.range.role, range.range.index) == (48, 0))
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let batch = state.action_batch;
    let root = publication_action_batch_root(&batch);
    if state.status != 0
        || state.blocks != COMPONENT_COUNT as u64
        || state.actor_eligible != 0
        || state.proposal != replay.predecessor.bank.header.proposal
        || state.proposal.checked_add(1) != Some(state.next_proposal)
        || batch.abi != ACTION_BATCH_ABI as u64
        || batch.actor_eligible != 0
        || batch.proposal != state.proposal
        || batch.component_count != COMPONENT_COUNT as u64
        || batch.candidate_count != 3
        || batch.winner > 2
        || batch.winner != state.task_evaluation.winner
        || batch.base_word != replay.predecessor_identity().word
        || batch.next_word != replay.successor_identity().word
        || batch.base_logical_digest != replay.predecessor_identity().logical_digest
        || batch.component_receipts_digest
            != publication_action_receipts_digest(codebook, &receipts)?
        || batch.semantic_receipts_digest
            != publication_semantic_receipts_digest(&state.semantic_receipts)
        || root != state.action_batch_root
        || root != replay.evidence.range(33)?.attempt()?.action_receipts_digest
        || receipts.iter().enumerate().any(|(index, receipt)| {
            receipt.proposal != state.proposal || receipt.ordinal as usize != index
        })
    {
        return Err(publication_input_error(
            "coding contributor differs from its original sealed native action batch",
        ));
    }
    Ok((state, receipts))
}

pub(super) fn contributor_indices<B: PublicationByteStorage>(
    contract: PublicationContract,
    header: PublicationHeader,
    intent: &SemanticOutputIntent,
    prefix: &PublicationMaterialRangeStorage<B>,
    provenance: &PublicationMaterialRangeStorage<B>,
    proposals: &[(
        &SemanticReplayMaterial,
        &SemanticResolvedReplayMaterial,
        &SemanticReplayLoadedMaterial,
    )],
) -> Result<Vec<usize>, SemanticTransitionError> {
    let binding = contract.nonterminal_intent_binding()?.ok_or_else(|| {
        publication_input_error("coding delivery has no original candidate interval")
    })?;
    if intent.payload.len() < 48
        || intent.payload[..32] != *binding.plan_digest.as_bytes()
        || intent.payload[32..40] != binding.candidate_prefix_begin.to_le_bytes()
    {
        return Err(publication_input_error(
            "coding intent lost its original candidate frame",
        ));
    }
    let count = u64::from_le_bytes(
        intent.payload[40..48]
            .try_into()
            .expect("checked coding frame"),
    );
    let end = binding
        .candidate_prefix_begin
        .checked_add(count)
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    if end > binding.candidate_prefix_end
        || end > header.prefix_extent
        || (header.terminal == 2
            && (end > header.terminal_position
                || header.terminal_position.checked_add(1) != Some(header.prefix_extent)))
        || count.checked_mul(8).and_then(|n| n.checked_add(48)) != Some(intent.payload.len() as u64)
        || (end != binding.candidate_prefix_end
            && (header.terminal != 2 || end != header.terminal_position))
        || prefix.range.role != 1
        || prefix.range.logical_begin != 0
        || prefix.range.logical_end != header.prefix_extent
        || header
            .prefix_extent
            .checked_mul(size_of::<SourceSlot>() as u64)
            != Some(prefix.bytes.len() as u64)
        || provenance.range.role != 2
        || !provenance
            .bytes
            .len()
            .is_multiple_of(size_of::<TokenProvenance>())
        || proposals.len() as u64 > contract.replay_append_entry_capacity()?
    {
        return Err(publication_input_error(
            "coding intent interval exceeds its original frozen geometry",
        ));
    }
    let mut previous = None;
    for (replay, _, _) in proposals {
        let word = replay.predecessor_identity().word;
        if previous.is_some_and(|prior| prior >= word) {
            return Err(publication_input_error(
                "coding Proposal candidates must retain their unique original order",
            ));
        }
        previous = Some(word);
    }
    let mut used = vec![false; proposals.len()];
    for offset in 0..count as usize {
        let position = binding.candidate_prefix_begin as usize + offset;
        // SAFETY: the sealed prefix extent above bounds this complete scalar record.
        let slot = unsafe {
            prefix.bytes[position * size_of::<SourceSlot>()..]
                .as_ptr()
                .cast::<SourceSlot>()
                .read_unaligned()
        };
        let expected = u64::from_le_bytes(
            intent.payload[48 + offset * 8..56 + offset * 8]
                .try_into()
                .expect("checked coding token extent"),
        );
        let record_offset = usize::try_from(slot.provenance_record)
            .ok()
            .and_then(|index| index.checked_mul(size_of::<TokenProvenance>()))
            .filter(|&offset| {
                provenance
                    .bytes
                    .len()
                    .checked_sub(size_of::<TokenProvenance>())
                    .is_some_and(|last| offset <= last)
            })
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        // SAFETY: the checked scalar record lies in the acquired append-only native provenance arena.
        let record = unsafe {
            provenance.bytes[record_offset..]
                .as_ptr()
                .cast::<TokenProvenance>()
                .read_unaligned()
        };
        if slot.kind != 1
            || slot.valid != 1
            || slot.committed != 1
            || slot.recomputed != 1
            || slot.logical_position != position as u64
            || slot.token != expected
            || record.logical_position != slot.logical_position
            || record.token != slot.token
            || !matches!(slot.provenance, 1 | 2)
        {
            return Err(publication_input_error(
                "coding payload differs from its actual native token provenance",
            ));
        }
        if slot.provenance == 1 {
            if record.action_receipt_digest != Identity256::default()
                || record.successor_logical_digest != Identity256::default()
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            continue;
        }
        let index = proposals
            .iter()
            .position(|(replay, _, _)| {
                replay.predecessor_identity().logical_digest == record.origin_logical_digest
                    && replay.successor_identity().logical_digest == record.successor_logical_digest
            })
            .ok_or_else(|| {
                publication_input_error(
                    "coding contributor roster omits an actual original Proposal",
                )
            })?;
        let (replay, original, action) = proposals[index];
        let (state, receipts) = proposal_receipts(replay, original, action)?;
        let receipt = usize::try_from(record.ordinal)
            .ok()
            .and_then(|ordinal| receipts.get(ordinal))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let closure = original
            .material
            .ranges
            .iter()
            .find(|range| (range.range.role, range.range.index) == (17, 0))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if record.proposal != state.proposal
            || record.source_slot >= 32
            || state.task_evaluation.winner == 0
            || record.ordinal != (state.task_evaluation.winner - 1) * 68 + record.source_slot
            || receipt.choice as u64 != slot.token
            || record.authority_generation != replay.predecessor.bank.header.authority_generation
            || record.authority_closure_origin
                != publication_range_origin(
                    replay.predecessor.bank.header.logical_digest,
                    &closure.range,
                )
            || record.record_digest == Identity256::default()
            || record.action_receipt_digest
                != Identity256::from_bytes(
                    Sha256::digest(publication_abi_bytes(&[*receipt])).into(),
                )
        {
            return Err(publication_input_error(
                "coding token was not sampled by its claimed original Proposal",
            ));
        }
        used[index] = true;
    }
    let mut indices = Vec::with_capacity(proposals.len());
    for (index, used) in used.into_iter().enumerate() {
        if used {
            indices.push(index);
        }
    }
    Ok(indices)
}

impl SemanticTransitionSession {
    /// Return only the original Proposal indices proved by the delivered tokens.
    /// The caller retains the uncovered originals; this getter grants no historical execution authority.
    pub fn coding_intent_contributors(
        &mut self,
        lease: &SemanticPublishedLease,
        stable: Identity256,
        proposals: &[(
            &SemanticReplayMaterial,
            &SemanticResolvedReplayMaterial,
            &SemanticReplayLoadedMaterial,
        )],
    ) -> Result<Vec<usize>, SemanticTransitionError> {
        self.checked_reader(lease)?;
        let contract = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .contract_value;
        if proposals.len() as u64 > contract.replay_append_entry_capacity()? {
            return Err(publication_input_error(
                "coding Proposal roster exceeds its original replay reservation",
            ));
        }
        let length = |role: u64| {
            lease
                .directory
                .iter()
                .find(|range| (range.role, range.index) == (role, 0))
                .map(|range| range.length_bytes)
                .ok_or(SemanticTransitionError::ObservationMismatch)
        };
        let intent_count = length(30)?
            .checked_sub(size_of::<IntentQueueHeader>() as u64)
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            / size_of::<IntentEntry>() as u64;
        let payload_length = length(31)?;
        let mut bytes = intent_count
            .checked_mul(size_of::<SemanticOutputIntent>() as u64)
            .and_then(|bytes| {
                bytes.checked_add(payload_length.checked_mul(intent_count.checked_add(1)?)?)
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    (proposals.len() as u64)
                        .checked_mul((2 * size_of::<usize>() + size_of::<bool>()) as u64)?,
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        for role in [1, 2, 30, 31] {
            bytes = bytes
                .checked_add(
                    length(role)?
                        .checked_mul(2)
                        .ok_or(SemanticTransitionError::GenerationExhausted)?,
                )
                .and_then(|bytes| {
                    bytes.checked_add(
                        (size_of::<PublicationMaterialRange>() + size_of::<PublicationRead<u8>>())
                            as u64,
                    )
                })
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let reservation = Arc::new(budget.reserve_auxiliary(bytes)?);
        budget.require_host_allocation(bytes)?;
        let entries = self.read_published_control_record_with_custody(
            lease,
            30,
            Some(Arc::clone(&reservation)),
            None,
        )?;
        let payload = self.read_published_control_record_with_custody(
            lease,
            31,
            Some(Arc::clone(&reservation)),
            None,
        )?;
        let intents = output_intents_from_ranges(&entries, &payload)?;
        let intent = intents
            .iter()
            .find(|intent| intent.stable_identity == stable)
            .ok_or_else(|| {
                publication_input_error("coding contributor query has no original native intent")
            })?;
        let prefix = self.read_published_control_record_with_custody(
            lease,
            1,
            Some(Arc::clone(&reservation)),
            None,
        )?;
        let provenance =
            self.read_published_control_record_with_custody(lease, 2, Some(reservation), None)?;
        contributor_indices(
            contract,
            lease.header,
            intent,
            &prefix,
            &provenance,
            proposals,
        )
    }
}
