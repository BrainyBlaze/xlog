//! Original publication reads into disjoint ranges of one checkpoint backing.

use super::*;

/// Actual native-selected facts of one held publication, independent of cold
/// admission labels and of a later publication on the same session.
pub struct SemanticSelectedTaskFacts {
    pub parent: SemanticPublishedIdentity,
    pub task_identity: Identity256,
    pub selected_slot: usize,
    pub facts: SemanticTaskFacts,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

/// Original small bank/ground readbacks, retained before either DMA can enter.
pub struct SemanticSelectedTaskFactsObservation {
    issuer: Arc<()>,
    token: u64,
    parent: SemanticPublishedIdentity,
    task_identity: Identity256,
    layout: task_ground::TaskGroundLayout,
    bank: Option<PublicationRead<PublicationBank>>,
    ground: Option<PublicationRead<u8>>,
    result: Option<Arc<SemanticSelectedTaskFacts>>,
    poisoned: bool,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

impl SemanticTransitionSession {
    pub fn prepare_selected_task_facts_observation(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<SemanticSelectedTaskFactsObservation, SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.checked_reader(lease)?;
        let task_identity = self
            .task_evaluation_identity()
            .ok_or(SemanticTransitionError::NotBound)?;
        let layout = self
            .task_ground
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .layout;
        let storage = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let range = lease
            .directory
            .iter()
            .find(|range| (range.role, range.index) == (58, 0))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if range.length_bytes != layout.bytes as u64 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let begin = usize::try_from(range.offset_bytes)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let end = begin
            .checked_add(layout.bytes)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let ground = storage
            .allocations
            .get(
                usize::try_from(range.storage_slot)
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
            )
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .slice()?
            .view()
            .try_slice(begin..end)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let read_bytes = layout
            .bytes
            .checked_add(size_of::<PublicationBank>())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let bytes = read_bytes
            .checked_mul(3)
            .and_then(|bytes| {
                bytes.checked_add(layout.queries.checked_mul(
                    3 * (size_of::<task_ground::TaskQueryRecord>() + 3 * size_of::<u64>()),
                )?)
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    size_of::<SemanticSelectedTaskFactsObservation>()
                        + size_of::<SemanticSelectedTaskFacts>()
                        + 2 * size_of::<PublicationRead<PublicationBank>>()
                        + 8 * size_of::<usize>(),
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = Arc::new(
            self.replay_material_budget
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .reserve_auxiliary(bytes as u64)?,
        );
        let bank = self.stage_publication_read_with_owner(
            storage.banks[(lease.identity.word & 1) as usize].view(),
            Some(Arc::clone(&reservation)),
        )?;
        let ground =
            self.stage_publication_read_with_owner(ground, Some(Arc::clone(&reservation)))?;
        Ok(SemanticSelectedTaskFactsObservation {
            issuer: Arc::clone(&self.publication_issuer),
            token: lease.token,
            parent: lease.identity,
            task_identity,
            layout,
            bank: Some(bank),
            ground: Some(ground),
            result: None,
            poisoned: false,
            _reservation: reservation,
        })
    }

    pub fn selected_task_facts_pending(
        &self,
        original: &SemanticSelectedTaskFactsObservation,
    ) -> bool {
        Arc::ptr_eq(&original.issuer, &self.publication_issuer)
            && original.result.is_none()
            && (original
                .bank
                .as_ref()
                .is_some_and(PublicationRead::completion_pending)
                || original
                    .ground
                    .as_ref()
                    .is_some_and(PublicationRead::completion_pending))
    }

    pub fn resolve_selected_task_facts(
        &mut self,
        lease: &SemanticPublishedLease,
        original: &mut SemanticSelectedTaskFactsObservation,
    ) -> Result<Arc<SemanticSelectedTaskFacts>, SemanticTransitionError> {
        self.checked_reader(lease)?;
        if !Arc::ptr_eq(&original.issuer, &self.publication_issuer)
            || original.token != lease.token
            || original.parent != lease.identity
            || self.task_evaluation_identity() != Some(original.task_identity)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if let Some(result) = &original.result {
            return Ok(Arc::clone(result));
        }
        let bank = self
            .resolve_publication_read_with_poison(
                original
                    .bank
                    .as_mut()
                    .ok_or(SemanticTransitionError::NotBound)?,
                &mut original.poisoned,
            )?
            .first()
            .copied()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if bank.header != lease.header || bank.state.task_evaluation.winner > 2 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let ground = self.resolve_publication_read_with_poison(
            original
                .ground
                .as_mut()
                .ok_or(SemanticTransitionError::NotBound)?,
            &mut original.poisoned,
        )?;
        let binding = &self
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .0;
        let facts = task_ground::decoded_task_facts(
            original.layout,
            &ground,
            &bank.state.task_evaluation,
            binding.expected_truth(),
        )?;
        let selected_slot = bank.state.task_evaluation.winner as usize;
        let facts = facts
            .into_iter()
            .nth(selected_slot)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if facts.truth.len() != original.layout.queries
            || facts.attained.len() != original.layout.queries
            || facts.g > original.layout.queries as u64
            || facts.p > 1
            || facts.eligible > 1
        {
            return Err(publication_input_error(
                "selected task facts omit the actual native query receipts",
            ));
        }
        let result = Arc::new(SemanticSelectedTaskFacts {
            parent: original.parent,
            task_identity: original.task_identity,
            selected_slot,
            facts,
            _reservation: Arc::clone(&original._reservation),
        });
        original.result = Some(Arc::clone(&result));
        original.bank = None;
        original.ground = None;
        original.poisoned = false;
        Ok(result)
    }
}

#[cfg(feature = "semantic-policy")]
#[derive(Clone, Copy)]
pub(super) enum CheckpointReadbackOrigin {
    Selected,
    Restored,
}

/// Captured by the genuine producer Session before the original plan. It
/// retains geometry and its receipt, never a graph or physical reader pin.
#[cfg(feature = "semantic-policy")]
pub(super) struct CheckpointReadbackGeometry {
    graph: crate::semantic_hypergraph::SemanticRootExportGeometry,
    role_count_bytes: u64,
    terminal_bytes: u64,
    nonmodels: Vec<(u64, u64, u64, u64)>,
    metadata_dma: [u64; 9],
    _reservation: SemanticReplayMaterialReservation,
}

#[cfg(feature = "semantic-policy")]
impl CheckpointReadbackGeometry {
    pub(super) fn ceilings(
        &self,
        _origin: CheckpointReadbackOrigin,
    ) -> Result<([u64; 9], [u64; 9]), SemanticTransitionError> {
        let mut dma = self.metadata_dma;
        native_work_bound::add_native_work_ceiling(&mut dma, self.graph.dma_ceiling)?;
        let mut total = dma;
        native_work_bound::add_native_work_ceiling(&mut total, self.graph.kernel_ceiling)?;
        Ok((dma, total))
    }

    pub(super) fn require_actual(
        &self,
        _origin: CheckpointReadbackOrigin,
        session: &SemanticTransitionSession,
        lease: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        let storage = session
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let graph = session
            .graph
            .transition_root_export_geometry()
            .map_err(SemanticTransitionError::Semantic)?;
        if graph != self.graph
            || u64::try_from(
                storage
                    .role_counts
                    .len()
                    .checked_mul(size_of::<PublicationRoleCount>())
                    .ok_or(SemanticTransitionError::GenerationExhausted)?,
            )
            .ok()
                != Some(self.role_count_bytes)
            || u64::try_from(
                storage
                    .terminals
                    .len()
                    .checked_mul(size_of::<u64>())
                    .ok_or(SemanticTransitionError::GenerationExhausted)?,
            )
            .ok()
                != Some(self.terminal_bytes)
        {
            return Err(publication_input_error(
                "checkpoint readback changed its original graph or control layout",
            ));
        }
        let mut nonmodels = lease
            .directory
            .iter()
            .filter(|range| !matches!(range.role, 18..=25));
        for &(role, index, length, capacity) in &self.nonmodels {
            let range = nonmodels
                .next()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let allocation = storage
                .allocations
                .get(
                    usize::try_from(range.storage_slot)
                        .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
                )
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let actual_capacity = u64::try_from(allocation.len())
                .ok()
                .and_then(|n| n.checked_sub(range.offset_bytes))
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if (range.role, range.index, range.length_bytes, actual_capacity)
                != (role, index, length, capacity)
            {
                return Err(publication_input_error(
                    "checkpoint readback changed its original non-model ranges",
                ));
            }
        }
        if nonmodels.next().is_some() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }
}

pub(super) struct OriginalPublicationReadback {
    loan: SemanticPublicationReadbackLoan,
    views: Vec<Option<DeviceMemoryView<u8>>>,
    allocation: usize,
    offset: usize,
    allocation_begin: Option<usize>,
    pending: Option<PublicationRead<u8>>,
    allocations: Vec<PublicationMaterialBytes>,
    metadata: Option<OriginalPublicationMetadata>,
    prefix: Option<Vec<u8>>,
    suffix_written: bool,
    suffix_begin: Option<usize>,
    entitlement_claimed: bool,
}

pub(super) struct OriginalPublicationMetadata {
    pub(super) bank: PublicationBank,
    pub(super) graph: SemanticRootMaterial,
    pub(super) role_counts: [u64; PUBLICATION_ROLE_COUNT],
    pub(super) terminals: Vec<u64>,
}

impl OriginalPublicationReadback {
    pub(super) fn new(
        loan: SemanticPublicationReadbackLoan,
        views: Vec<Option<DeviceMemoryView<u8>>>,
    ) -> Self {
        let count = views.len();
        Self {
            loan,
            views,
            allocation: 0,
            offset: 0,
            allocation_begin: None,
            pending: None,
            allocations: Vec::with_capacity(count),
            metadata: None,
            prefix: None,
            suffix_written: false,
            suffix_begin: None,
            entitlement_claimed: false,
        }
    }
}

impl SemanticTransitionSession {
    #[cfg(feature = "semantic-policy")]
    pub(super) fn checkpoint_readback_geometry(
        &self,
        storage: &PublicationStorage,
        directory: &[PublicationRange],
    ) -> Result<Arc<CheckpointReadbackGeometry>, SemanticTransitionError> {
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let graph = self
            .graph
            .transition_root_export_geometry()
            .map_err(SemanticTransitionError::Semantic)?;
        let count = directory
            .iter()
            .filter(|range| !matches!(range.role, 18..=25))
            .count();
        let extent = size_of::<CheckpointReadbackGeometry>()
            .checked_add(2 * size_of::<usize>())
            .and_then(|n| n.checked_add(count.checked_mul(size_of::<(u64, u64, u64, u64)>())?))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = budget.reserve_auxiliary(
            u64::try_from(extent).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?;
        let mut nonmodels = Vec::new();
        nonmodels.try_reserve_exact(count).map_err(|_| {
            publication_input_error("checkpoint readback range metadata allocation failed")
        })?;
        let mut dma = [0; 9];
        let mut add_read = |bytes: u64| {
            native_work_bound::add_native_work_ceiling(&mut dma, [1, 0, 0, 0, 0, 0, 0, 0, bytes])
        };
        // The original native observation supports a selected-recompute check;
        // at most one control read is reached in that same producer.
        add_read(size_of::<PublicationBank>() as u64)?;
        add_read(size_of::<PublicationControl>() as u64)?;
        let role_count_bytes = u64::try_from(
            storage
                .role_counts
                .len()
                .checked_mul(size_of::<PublicationRoleCount>())
                .ok_or(SemanticTransitionError::GenerationExhausted)?,
        )
        .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let terminal_bytes = u64::try_from(
            storage
                .terminals
                .len()
                .checked_mul(size_of::<u64>())
                .ok_or(SemanticTransitionError::GenerationExhausted)?,
        )
        .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        add_read(role_count_bytes)?;
        if terminal_bytes != 0 {
            add_read(terminal_bytes)?;
        }
        for range in directory
            .iter()
            .filter(|range| !matches!(range.role, 18..=25))
        {
            let allocation = storage
                .allocations
                .get(
                    usize::try_from(range.storage_slot)
                        .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
                )
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let capacity = u64::try_from(allocation.len())
                .ok()
                .and_then(|n| n.checked_sub(range.offset_bytes))
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if range.length_bytes > capacity {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            nonmodels.push((range.role, range.index, range.length_bytes, capacity));
            if range.length_bytes != 0 {
                add_read(range.length_bytes)?;
            }
        }
        Ok(Arc::new(CheckpointReadbackGeometry {
            graph,
            role_count_bytes,
            terminal_bytes,
            nonmodels,
            metadata_dma: dma,
            _reservation: reservation,
        }))
    }
    pub(super) fn require_observation_model_backings(
        &self,
        lease: &SemanticPublishedLease,
        storage: &PublicationStorage,
    ) -> Result<(), SemanticTransitionError> {
        if storage.model_slots.len() != storage.model_memory.allocation_bytes.len()
            || storage.model_generations.len() != storage.model_slots.len()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        for (index, slots) in storage.model_slots.iter().enumerate() {
            let mut selected = None;
            for range in lease
                .directory
                .iter()
                .filter(|range| matches!(range.role, 18..=25))
            {
                let (allocation, _) = storage.model_memory.location(range.role, range.index)?;
                if allocation == index
                    && selected
                        .replace(range.storage_slot as usize)
                        .is_some_and(|prior| prior != range.storage_slot as usize)
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
            }
            let slot = selected.unwrap_or(slots[(lease.identity.word & 1) as usize]);
            if !storage.model_generations[index].contains(&slot) {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let allocation = storage
                .allocations
                .get(slot)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if u64::try_from(allocation.len()).ok()
                != Some(storage.model_memory.allocation_bytes[index])
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        }
        Ok(())
    }

    pub(super) fn resolve_publication_observation_metadata(
        &mut self,
        original: &mut SemanticPublishedStateObservation,
        poisoned: &mut bool,
    ) -> Result<OriginalPublicationMetadata, SemanticTransitionError> {
        let bank = *self
            .resolve_publication_read_with_poison(&mut original.bank, poisoned)?
            .first()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if publication_abi_bytes(&[bank.header]) != publication_abi_bytes(&[original.header])
            || publication_abi_bytes(&bank.source) != publication_abi_bytes(&original.source)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if let Some(control) = &mut original.control {
            let control = *self
                .resolve_publication_read_with_poison(control, poisoned)?
                .first()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if control.abi != 1
                || control.instance != original.identity.instance
                || control.word != original.identity.word
            {
                return Err(publication_input_error(
                    "selected recompute is no longer the current publication",
                ));
            }
        }
        let graph = self
            .graph
            .resolve_transition_root_export(&mut original.graph)
            .map_err(SemanticTransitionError::Semantic)?;
        let counts = self.resolve_publication_read_with_poison(&mut original.counts, poisoned)?;
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
        let terminals = if let Some(terminals) = &mut original.terminals {
            self.resolve_publication_read_with_poison(terminals, poisoned)?
        } else {
            Vec::new()
        };
        Ok(OriginalPublicationMetadata {
            bank,
            graph,
            role_counts,
            terminals,
        })
    }

    pub(super) fn resolve_streamed_publication_in_output(
        &mut self,
        lease: &SemanticPublishedLease,
        original: &mut SemanticPublishedStateObservation,
        output: &SemanticReplayMaterialOutput,
        frame_prefix: &[u8],
    ) -> Result<SemanticReplayLoadedMaterial, SemanticTransitionError> {
        let mut poisoned = self.poisoned;
        let result = self.resolve_streamed_publication_with_poison(
            lease,
            original,
            output,
            frame_prefix,
            &mut poisoned,
        );
        self.poisoned |= poisoned;
        result
    }

    fn resolve_streamed_publication_with_poison(
        &mut self,
        lease: &SemanticPublishedLease,
        original: &mut SemanticPublishedStateObservation,
        output: &SemanticReplayMaterialOutput,
        frame_prefix: &[u8],
        poisoned: &mut bool,
    ) -> Result<SemanticReplayLoadedMaterial, SemanticTransitionError> {
        self.require_published_state_observation(lease, original)?;
        let loan = original
            .readback
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .loan
            .clone();
        loan.bind_observation(&self.publication_issuer, lease.token, output)?;
        loan.require_backing_spans(&original.storage.model_memory.allocation_bytes)?;
        #[cfg(feature = "semantic-policy")]
        if !original
            .readback
            .as_ref()
            .expect("original readback")
            .entitlement_claimed
        {
            let allowance = self.graph.cold_work_allowance().ok_or_else(|| {
                publication_input_error("compiled readback requires its original finite allowance")
            })?;
            allowance
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?
                .claim_checkpoint_readback(&loan)?;
            original
                .readback
                .as_mut()
                .expect("original readback")
                .entitlement_claimed = true;
        }
        if original
            .readback
            .as_ref()
            .expect("original readback")
            .metadata
            .is_none()
        {
            let metadata = self.resolve_publication_observation_metadata(original, poisoned)?;
            original
                .readback
                .as_mut()
                .expect("original readback")
                .metadata = Some(metadata);
        }
        if original.encoded_output.is_none() {
            let readback = original.readback.as_mut().expect("original readback");
            let metadata = readback
                .metadata
                .as_ref()
                .expect("original publication metadata");
            let mut contract = original.storage.contract_value;
            contract.terminal_tokens = 0;
            contract.role_counts = 0;
            contract.semantic_owner = 0;
            let prefix = OwnedPublicationMaterial::encoded_prefix(
                &metadata.bank,
                &contract,
                &metadata.role_counts,
                &metadata.terminals,
                &original.storage.layouts,
                &original.storage.model_memory,
                &metadata.graph,
            )?;
            let mut extent = prefix.len();
            for view in &readback.views {
                let length = view.as_ref().map_or(0, DeviceMemoryView::len);
                u32::try_from(length).map_err(|_| SemanticTransitionError::GenerationExhausted)?;
                extent = extent
                    .checked_add(4)
                    .and_then(|n| n.checked_add(length))
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
            }
            extent = extent
                .checked_add(4)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            for range in &original.directory {
                extent = extent
                    .checked_add(4 + size_of::<PublicationRange>() + 8)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                if !matches!(range.role, 18..=25) {
                    let bytes = usize::try_from(range.length_bytes)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
                    u32::try_from(bytes)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
                    extent = extent
                        .checked_add(4)
                        .and_then(|n| n.checked_add(bytes))
                        .ok_or(SemanticTransitionError::GenerationExhausted)?;
                }
            }
            let mut history = Vec::new();
            learning_phase::encode_history(&original.learning_phases, &mut history)?;
            extent = extent
                .checked_add(history.len())
                .and_then(|n| n.checked_add(frame_prefix.len()))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let frame_begin = output.declare_frame_prefix(extent)?;
            // All fallible framing precedes the first chunk. The original
            // output identity and cursor are retained before any DMA entry.
            readback.prefix = Some(prefix);
            original.encoded_output = Some(OriginalPublicationEncodedOutput {
                output: output.clone(),
                prefix_digest: Identity256::from_bytes(Sha256::digest(frame_prefix).into()),
                prefix_extent: frame_prefix.len(),
                frame_begin,
                cursor: 0,
                native: None,
            });
        }
        let encoded = original
            .encoded_output
            .as_mut()
            .expect("original output cursor");
        let readback = original
            .readback
            .as_mut()
            .expect("original readback cursor");
        if encoded.cursor == 0 {
            let prefix = readback.prefix.as_ref().expect("original canonical prefix");
            encoded
                .output
                .write_frame(encoded.frame_begin, frame_prefix)?;
            encoded.cursor = frame_prefix.len();
            encoded
                .output
                .write_frame(encoded.frame_begin + encoded.cursor, prefix)?;
            encoded.cursor += prefix.len();
        } else if encoded.cursor == frame_prefix.len() {
            let prefix = readback.prefix.as_ref().expect("original canonical prefix");
            encoded
                .output
                .write_frame(encoded.frame_begin + encoded.cursor, prefix)?;
            encoded.cursor += prefix.len();
        }
        // The prefix is now in its sole monotone destination. Do not keep a
        // second encoder prefix while validation hashes the original backing.
        readback.prefix = None;
        while readback.allocation < readback.views.len() {
            let view = &readback.views[readback.allocation];
            let length = view.as_ref().map_or(0, DeviceMemoryView::len);
            if readback.allocation_begin.is_none() {
                let length = u32::try_from(length)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
                encoded
                    .output
                    .write_frame(encoded.frame_begin + encoded.cursor, &length.to_le_bytes())?;
                encoded.cursor += 4;
                readback.allocation_begin = Some(encoded.frame_begin + encoded.cursor);
            }
            if readback.offset < length {
                let end = readback
                    .offset
                    .checked_add(loan.staging_bytes())
                    .map(|end| end.min(length))
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                if readback.pending.is_none() {
                    loan.require_staging_host_allocation(
                        (end - readback.offset)
                            .checked_mul(2)
                            .ok_or(SemanticTransitionError::GenerationExhausted)?,
                    )?;
                    readback.pending = Some(
                        self.stage_publication_read_with_custody(
                            view.as_ref()
                                .expect("original nonempty backing view")
                                .slice(readback.offset..end),
                            None,
                            Some(loan.retained_owner()),
                        )?,
                    );
                }
                let pending = readback.pending.as_mut().expect("original pending chunk");
                // The original pinned or cached extent is already resident on
                // continuation; price only its additional returned buffer.
                loan.require_staging_host_allocation(end - readback.offset)?;
                let bytes = self.resolve_publication_read_with_poison(pending, poisoned)?;
                if bytes.len() != end - readback.offset {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                encoded
                    .output
                    .write_frame(encoded.frame_begin + encoded.cursor, &bytes)?;
                encoded.cursor += bytes.len();
                readback.offset = end;
                drop(bytes);
                pending.read.retire_completed_values();
                readback.pending = None;
            }
            if readback.offset == length {
                let begin = readback
                    .allocation_begin
                    .expect("original full backing output range");
                let loaded = encoded
                    .output
                    .initialized_prefix_range(begin..begin + length)?;
                readback.allocations.push(loaded.bytes);
                readback.allocation += 1;
                readback.offset = 0;
                readback.allocation_begin = None;
            }
        }
        if original.material.is_none() {
            let mut ranges = Vec::with_capacity(original.directory.len());
            for range in &original.directory {
                if matches!(range.role, 18..=25) {
                    let (allocation, offset) = original
                        .storage
                        .model_memory
                        .location(range.role, range.index)?;
                    let bytes = readback
                        .allocations
                        .get(allocation)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    let end = usize::try_from(range.length_bytes)
                        .ok()
                        .and_then(|n| offset.checked_add(n))
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    ranges.push(OwnedPublicationMaterialRange {
                        range: *range,
                        capacity: bytes
                            .len()
                            .checked_sub(offset)
                            .ok_or(SemanticTransitionError::ObservationMismatch)?,
                        bytes: bytes.slice_bytes(offset..end)?,
                    });
                } else {
                    let (_, capacity, pending) = original
                        .ranges
                        .iter_mut()
                        .find(|(candidate, _, _)| {
                            candidate.role == range.role && candidate.index == range.index
                        })
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    let bytes = if let Some(pending) = pending {
                        self.resolve_publication_read_with_poison(pending, poisoned)?
                    } else {
                        Vec::new()
                    };
                    ranges.push(OwnedPublicationMaterialRange {
                        range: *range,
                        capacity: *capacity,
                        bytes: PublicationMaterialBytes::readback(
                            bytes,
                            original.allocation_reservation.clone(),
                        ),
                    });
                }
            }
            let metadata = readback
                .metadata
                .take()
                .expect("original metadata completion");
            let mut contract = original.storage.contract_value;
            contract.terminal_tokens = 0;
            contract.role_counts = 0;
            contract.semantic_owner = 0;
            let material = OwnedPublicationMaterial {
                bank: metadata.bank,
                contract,
                role_counts: metadata.role_counts,
                terminals: metadata.terminals,
                layouts: original.storage.layouts.clone(),
                model_memory: original.storage.model_memory.clone(),
                model_allocations: std::mem::take(&mut readback.allocations),
                ranges,
                graph: metadata.graph,
                learning_phases: original.learning_phases.clone(),
                evaluation_fork: self.evaluation_fork.clone(),
            };
            material.validate()?;
            if original.control.is_some() {
                material.require_successful_recompute()?;
            }
            original.material = Some(Arc::new(material));
        }
        let material = original
            .material
            .as_ref()
            .expect("original validated material");
        if !readback.suffix_written {
            let begin = *readback.suffix_begin.get_or_insert(encoded.cursor);
            let mut offset = begin;
            OwnedPublicationMaterial::write_encoded_suffix(
                material
                    .ranges
                    .iter()
                    .map(|range| (range.range, range.capacity, Some(range.bytes.as_ref()))),
                material.ranges.len(),
                &material.learning_phases,
                material.evaluation_fork.as_ref(),
                |part| {
                    let end = offset
                        .checked_add(part.len())
                        .ok_or(SemanticTransitionError::GenerationExhausted)?;
                    if end > encoded.cursor {
                        if offset != encoded.cursor {
                            return Err(SemanticTransitionError::ObservationMismatch);
                        }
                        encoded
                            .output
                            .write_frame(encoded.frame_begin + offset, part)?;
                        encoded.cursor = end;
                    }
                    offset = end;
                    Ok(())
                },
            )?;
            readback.suffix_written = true;
        }
        let (digest, extent) = material.encoded_identity()?;
        if usize::try_from(extent)
            .ok()
            .and_then(|n| n.checked_add(frame_prefix.len()))
            != Some(encoded.cursor)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let mut sources = original
            .storage
            .material_sources
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let source = &mut sources[(original.identity.word & 1) as usize];
        if source.identity != Some(original.identity) {
            source.identity = Some(original.identity);
            source.digests.clear();
        }
        source.digests.insert(digest);
        drop(sources);
        let native = encoded
            .output
            .seal_prefix()?
            .slice_payload(frame_prefix.len()..encoded.cursor)?;
        encoded.native = Some(native.slice_payload(0..native.as_bytes().len())?);
        original.bank.read.retire_completed_values();
        if let Some(control) = &mut original.control {
            control.read.retire_completed_values();
        }
        original.counts.read.retire_completed_values();
        if let Some(terminals) = &mut original.terminals {
            terminals.read.retire_completed_values();
        }
        for (_, _, pending) in &mut original.ranges {
            if let Some(pending) = pending {
                pending.read.retire_completed_values();
            }
        }
        original.material = None;
        Ok(native)
    }
}
