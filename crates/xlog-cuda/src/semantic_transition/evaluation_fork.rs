//! Independent evaluation contexts and complete immutable source custody.
use super::*;

impl SemanticTransitionSession {
    pub fn evaluation_fork_origin(
        &self,
    ) -> Result<Option<SemanticEvaluationForkOrigin>, SemanticTransitionError> {
        // Immutable custody metadata is available to the same original
        // completion join; this grants no new native submission or reader.
        Ok(self.evaluation_fork.clone())
    }
}

impl SemanticEvaluationForkPreparation {
    pub fn started(&self) -> bool {
        self.started
    }
}

#[cfg(feature = "semantic-policy")]
impl SemanticTransitionSession {
    pub fn evaluation_fork_consumer_stream(&self) -> Result<u64, SemanticTransitionError> {
        self.ensure_quiescent()?;
        if self.evaluation_fork.is_none() {
            return Err(SemanticTransitionError::NotBound);
        }
        let stream = self.stream.cu_stream() as u64;
        dlpack_consumer_stream(stream)?;
        Ok(stream)
    }

    pub fn evaluation_fork_completion_pending(
        &self,
        preparation: &SemanticEvaluationForkPreparation,
    ) -> bool {
        if !Arc::ptr_eq(&preparation.issuer, &self.publication_issuer)
            || self.task_evaluation_identity() != Some(preparation.target_task)
        {
            return false;
        }
        if !preparation.started || self.state_material_restore.is_none() {
            return false;
        }
        self.state_material_restore_completion_pending()
            .is_ok_and(|pending| pending)
    }
}

// The complete selected training checkpoint belongs to the second presealed
// training context. The first context remains the enclosing roster owner.
const SELECTED_TRAINING_CONTEXT: usize = 1;

#[cfg(feature = "semantic-policy")]
impl SemanticTransitionSession {
    /// Authenticate the callback's complete original model and backing owners.
    /// This observes metadata and allocation identity, never tensor values.
    pub fn require_evaluation_fork_model_roster(
        &mut self,
        lease: &SemanticPublishedLease,
        tensors: Vec<SemanticTensorInput>,
        consumer_stream: u64,
    ) -> Result<SemanticGradientDeliveryBinding, SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.checked_reader(lease)?;
        if self.evaluation_fork.is_none() {
            return Err(publication_input_error(
                "model restoration roster requires its original evaluation fork",
            ));
        }
        dlpack_consumer_stream(consumer_stream)?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let count = tensors
            .len()
            .checked_add(lease.directory.len())
            .and_then(|count| count.checked_add(storage.model_memory.allocation_bytes.len()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let bytes = count
            .checked_mul(
                size_of::<PreparedSemanticTensor>() + size_of::<SemanticTensorLayout>() + 512,
            )
            .and_then(|bytes| bytes.checked_add(size_of::<SemanticGradientDeliveryBinding>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .reserve_auxiliary(bytes as u64)?;
        let pending = RetainedTensorAdmission {
            inputs: Some(tensors),
            stream: Arc::clone(&self.stream),
            consumer_stream: Some(consumer_stream),
            ready: false,
        };
        let imported = prepare_retained_semantic_tensors(&self.provider, pending)?;
        model_content_ranges(
            &lease.directory,
            &storage.layouts,
            imported
                .iter()
                .filter(|tensor| matches!(tensor.layout.role, 18..=25))
                .map(|tensor| {
                    (
                        tensor.layout,
                        tensor.logical_begin,
                        tensor.logical_end,
                        tensor.source.as_ref().map_or(0, DeviceMemoryView::len),
                    )
                }),
        )?;
        let slots = storage.model_slots_for_directory(&lease.directory)?;
        let coordinates = BTreeSet::new();
        let model = GradientDeliveryModel::Acquired(&storage, lease, &coordinates);
        let mut backings = BTreeSet::new();
        let mut caches = BTreeSet::new();
        for tensor in &imported {
            match tensor.layout.role {
                18..=25 => {
                    if !model.owns_tensor(tensor)? {
                        return Err(publication_input_error("restored model tensor is not an alias of its original native fork allocation"));
                    }
                }
                0 => {
                    let ordinal = usize::try_from(tensor.layout.index)
                        .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
                    if !backings.insert(ordinal) {
                        return Err(publication_input_error(
                            "restored model repeats a complete backing allocation",
                        ));
                    }
                    let slot = slots
                        .get(ordinal)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?
                        [(lease.identity.word & 1) as usize];
                    let view = storage
                        .allocations
                        .get(slot)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?
                        .slice()?
                        .view();
                    let expected = SemanticTensorLayout {
                        role: 0,
                        index: ordinal as u64,
                        scalar_type: 1,
                        element_bytes: 1,
                        rank: 1,
                        logical_axis: u64::MAX,
                        dimensions: [view.len() as u64, 0, 0, 0],
                        strides_bytes: [1, 0, 0, 0],
                    };
                    if tensor.layout != expected
                        || tensor.logical_begin != 0
                        || tensor.logical_end != 0
                        || tensor.data != *view.device_ptr()
                        || tensor.source.as_ref().map_or(0, DeviceMemoryView::len) != view.len()
                        || !step_input_allocation_matches(tensor, &view)
                    {
                        return Err(publication_input_error(
                            "restored backing is not the complete original native fork allocation",
                        ));
                    }
                }
                4..=13 => {
                    let key = (tensor.layout.role, tensor.layout.index);
                    if !caches.insert(key) {
                        return Err(publication_input_error(
                            "restored model repeats a prefix-cache tensor",
                        ));
                    }
                    let range = *lease
                        .directory
                        .iter()
                        .find(|range| (range.role, range.index) == key)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    let layout = *storage
                        .layouts
                        .get(&key)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    let view = self.publication_tensor_view(lease, range, layout)?;
                    if tensor.layout != layout
                        || (tensor.logical_begin, tensor.logical_end)
                            != (range.logical_begin, range.logical_end)
                        || tensor.data != *view.device_ptr()
                        || tensor.source.as_ref().map_or(0, DeviceMemoryView::len) != view.len()
                        || !step_input_allocation_matches(tensor, &view)
                    {
                        return Err(publication_input_error(
                            "restored prefix cache is not its original native fork view",
                        ));
                    }
                }
                _ => {
                    return Err(publication_input_error(
                        "restored model roster contains a foreign semantic role",
                    ))
                }
            }
        }
        if backings != (0..slots.len()).collect::<BTreeSet<_>>() {
            return Err(publication_input_error(
                "restored model omits a complete original backing allocation",
            ));
        }
        if !caches.is_empty()
            && caches
                != lease
                    .directory
                    .iter()
                    .filter(|range| matches!(range.role, 4..=13))
                    .map(|range| (range.role, range.index))
                    .collect::<BTreeSet<_>>()
        {
            return Err(publication_input_error(
                "restored prefix-cache roster is incomplete",
            ));
        }
        self.steps
            .get_mut(&lease.token)
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .consumer_streams
            .insert(consumer_stream);
        self.order_content_inputs(lease, consumer_stream)?;
        self.order_content_consumers(consumer_stream)?;
        Ok(SemanticGradientDeliveryBinding {
            _tensors: imported,
            _reservation: Some(reservation),
        })
    }
}

/// Immutable cold metadata that can outlive terminal retirement of the source's
/// numerical publication. It retains no hypergraph, tensor, model or reader.
pub struct SemanticDetachedColdTaskContext {
    canary: Arc<crate::SemanticTrainingCanarySource>,
    provider: Arc<CudaKernelProvider>,
    domain: ResidentExecutionDomain,
    model_context: Option<SemanticFrozenModelContext>,
    _statement_reservation: Option<SemanticReplayMaterialReservation>,
}

impl SemanticDetachedColdTaskContext {
    pub fn training_canary_source(&self) -> Arc<crate::SemanticTrainingCanarySource> {
        Arc::clone(&self.canary)
    }
    pub fn allocation_domain(&self) -> (Arc<CudaKernelProvider>, ResidentExecutionDomain) {
        (Arc::clone(&self.provider), self.domain.clone())
    }
    pub fn frozen_model_context(&self) -> Option<SemanticFrozenModelContext> {
        self.model_context.clone()
    }
}

impl SemanticTransitionSession {
    /// Move only completed immutable cold observation custody before the
    /// controller performs the existing full reader join and terminal release.
    /// Ordinary publication retirement still validates every numerical alias.
    pub fn detach_cold_task_context(
        &mut self,
    ) -> Result<Option<SemanticDetachedColdTaskContext>, SemanticTransitionError> {
        self.ensure_quiescent()?;
        if self.cold_canary_source.is_none() {
            return Ok(None);
        }
        if !self.cold_observer_completed
            || self.cold_content.is_none()
            || self
                .model_context_preseal
                .as_ref()
                .is_some_and(|_| self.has_pending_model_context_preseal())
        {
            return Err(publication_input_error(
                "cold context detachment requires its completed original observation and frozen catalogue",
            ));
        }
        let model_context = self.frozen_model_context()?;
        let canary = self.cold_canary_source.take().ok_or_else(|| {
            publication_input_error("original cold observation custody was already detached")
        })?;
        Ok(Some(SemanticDetachedColdTaskContext {
            canary,
            provider: Arc::clone(&self.provider),
            domain: self.domain.clone(),
            model_context,
            _statement_reservation: self.cold_statement_reservation.take(),
        }))
    }
}

#[cfg(feature = "semantic-policy")]
impl SemanticTransitionSession {
    pub fn restore_evaluation_fork(
        &mut self,
        preparation: &mut SemanticEvaluationForkPreparation,
        restoration: Option<&SemanticPreparedResidentModelRestoration>,
    ) -> Result<Option<SemanticPublishedIdentity>, SemanticTransitionError> {
        if !Arc::ptr_eq(&preparation.issuer, &self.publication_issuer)
            || self.task_evaluation_identity() != Some(preparation.target_task)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if preparation.started {
            return self.resolve_evaluation_fork(preparation);
        }
        self.ensure_rebindable()?;
        let material = preparation
            .material
            .take()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let digest = preparation.origin.native_digest;
        preparation.started = true;
        self.restore_decoded_state_material_original(
            material,
            digest,
            None,
            None,
            restoration.cloned(),
            None,
            true,
        )
    }

    pub fn resolve_evaluation_fork(
        &mut self,
        preparation: &SemanticEvaluationForkPreparation,
    ) -> Result<Option<SemanticPublishedIdentity>, SemanticTransitionError> {
        if !preparation.started
            || !Arc::ptr_eq(&preparation.issuer, &self.publication_issuer)
            || self.task_evaluation_identity() != Some(preparation.target_task)
            || self.evaluation_fork.as_ref().is_none_or(|origin| {
                origin.source != preparation.origin.source
                    || origin.member != preparation.origin.member
                    || !origin
                        .material
                        .as_ref()
                        .zip(preparation.origin.material.as_ref())
                        .is_some_and(|(left, right)| Arc::ptr_eq(&left.owner, &right.owner))
            })
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.resolve_state_material_restore()
    }
}

fn retain_evaluation_bytes(
    budget: &SemanticReplayMaterialBudget,
    bytes: Vec<u8>,
) -> Result<PublicationMaterialBytes, SemanticTransitionError> {
    Ok(budget
        .reserve(bytes.capacity() as u64)?
        .retain_owned_payload(bytes)?
        .bytes)
}

fn replace_evaluation_record(
    material: &mut OwnedPublicationMaterial,
    budget: &SemanticReplayMaterialBudget,
    role: SemanticStateRole,
    index: u64,
    bytes: Vec<u8>,
    capacity: usize,
) -> Result<(), SemanticTransitionError> {
    if bytes.len() > capacity {
        return Err(publication_input_error(
            "evaluation material exceeds its original capacity",
        ));
    }
    let payload = retain_evaluation_bytes(budget, bytes)?;
    if let Some(item) = material
        .ranges
        .iter_mut()
        .find(|item| (item.range.role, item.range.index) == (role as u64, index))
    {
        item.range.length_bytes = payload.len() as u64;
        item.range.logical_begin = 0;
        item.range.logical_end = 0;
        item.capacity = capacity;
        item.bytes = payload;
    } else {
        let slot = material
            .ranges
            .iter()
            .map(|item| item.range.storage_slot)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        material.ranges.push(OwnedPublicationMaterialRange {
            range: PublicationRange {
                role: role as u64,
                index,
                storage_slot: slot,
                generation: 1,
                length_bytes: payload.len() as u64,
                ..PublicationRange::default()
            },
            capacity,
            bytes: payload,
        });
    }
    Ok(())
}

impl SemanticTransitionSession {
    pub fn evaluation_fork_target_bound(&self) -> Result<bool, SemanticTransitionError> {
        match &self.evaluation_context {
            None => Ok(false),
            Some(context) => Ok(context
                .target
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?
                .is_some()),
        }
    }

    #[cfg(feature = "semantic-policy")]
    pub fn prepare_evaluation_fork(
        &self,
        loaded: SemanticReplayLoadedMaterial,
        member: &SemanticFrozenModelContext,
        target_records: SemanticEvaluationForkTargetRecords,
    ) -> Result<SemanticEvaluationForkPreparation, SemanticTransitionError> {
        self.ensure_rebindable()?;
        member.require_budget(
            self.replay_material_budget
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        )?;
        let (catalogue, ordinal) = self
            .codebooks
            .model_catalogue
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if member.member < 2
            || member.member != *ordinal
            || !Arc::ptr_eq(&member.catalogue, catalogue)
            || self.publication.is_some()
            || self.initial_prefill.is_some()
            || self.state_material_restore.is_some()
            || self.evaluation_fork.is_some()
            || self.training_views.is_some()
            || self.task_training_domain().is_some()
        {
            return Err(publication_input_error(
                "evaluation fork requires its original newly imported frozen evaluation member",
            ));
        }
        let task = &self
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .0;
        let target_task = task.identity();
        if !Arc::ptr_eq(&target_records.issuer, &self.publication_issuer)
            || target_records.task != target_task
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let SemanticEvaluationForkTargetRecords {
            records: target_records,
            _reservation: _target_record_reservation,
            ..
        } = target_records;
        let context = self
            .evaluation_context
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let target = context
            .target
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let target = &target
            .as_ref()
            .ok_or_else(|| {
                publication_input_error("evaluation member lost its original cold target envelope")
            })?
            .value;
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        loaded.require_budget(budget)?;
        let archive = loaded.bytes.clone();
        let mut material = loaded.into_complete()?;
        if material.evaluation_fork.is_some()
            || material.contract.task_identity == target_task
            || material.graph.records.records.len() != context.target_record_offset as usize
        {
            return Err(publication_input_error(
                "evaluation source must be the complete ordinary source task",
            ));
        }
        let saved = material
            .ranges
            .iter()
            .find(|item| (item.range.role, item.range.index) == (48, 0))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let header_bytes = saved
            .bytes
            .get(..model_catalogue::HEADER_WORDS * 8)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let mut header = [0u64; model_catalogue::HEADER_WORDS];
        for (word, bytes) in header.iter_mut().zip(header_bytes.chunks_exact(8)) {
            *word = u64::from_le_bytes(bytes.try_into().unwrap());
        }
        let original_context = catalogue.context_for_header(&header)?;
        if original_context.member != SELECTED_TRAINING_CONTEXT {
            return Err(publication_input_error(
                "evaluation source archive does not name the frozen source member",
            ));
        }
        let graph = self
            .graph
            .admission()
            .ok_or(SemanticTransitionError::NotBound)?;
        let ground_layout = self
            .task_ground
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .layout;
        let lineage_node = OwnedPublicationMaterial::tree_node_bound(
            size_of::<[u8; 32]>(),
            size_of::<([u8; 32], u64)>(),
        )?
        .checked_add(OwnedPublicationMaterial::tree_node_bound(
            size_of::<([u8; 32], [u8; 32])>(),
            0,
        )?)
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut extent = crate::semantic_hypergraph::semantic_material_allocation_bound(
            graph,
            material.graph.insertions.len(),
        )
        .map_err(SemanticTransitionError::Semantic)?
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(material.graph.admission_allocation_bound().ok()?))
        .and_then(|bytes| {
            bytes.checked_add(material.graph.insertions.len().checked_mul(lineage_node)?)
        })
        .and_then(|bytes| bytes.checked_add(self.codebooks.words.len().checked_mul(16)?))
        .and_then(|bytes| bytes.checked_add(ground_layout.bytes.checked_mul(2)?))
        .and_then(|bytes| bytes.checked_add(size_of::<SemanticEvaluationForkPreparation>() + 1024))
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
        for record in &target_records {
            extent = extent
                .checked_add(record.bytes.capacity())
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let reservation = Arc::new(budget.reserve_auxiliary(extent as u64)?);
        let mode = material.model_numerical_mode()?.to_vec();
        let archive = SemanticReplayLoadedMaterial {
            bytes: archive,
            decoded: None,
        }
        .retain_shared_metadata(Arc::clone(&reservation))?
        .bytes;
        let origin = SemanticEvaluationForkOrigin {
            source: SemanticPublishedIdentity {
                instance: material.bank.header.instance,
                word: material.bank.header.publication_word,
                logical_digest: material.bank.header.logical_digest,
            },
            source_task: material.contract.task_identity,
            catalogue: self.model_catalogue_binding().digest,
            member: member.member,
            native_digest: Identity256::from_bytes(Sha256::digest(&archive).into()),
            native_extent: archive.len() as u64,
            material: Some(archive),
        };
        material.graph = material
            .graph
            .evaluation_context(graph)
            .map_err(SemanticTransitionError::Semantic)?;
        let old_ground = material
            .ranges
            .iter()
            .find(|item| (item.range.role, item.range.index) == (58, 0))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        catalogue.member_words(SELECTED_TRAINING_CONTEXT)?;
        // The selected rule count is explicit in the checked native ground.
        // Original source rules are copied byte-for-byte into the new task bank.
        let source_queries = u64::from_le_bytes(
            old_ground
                .bytes
                .get(16..24)
                .ok_or(SemanticTransitionError::ObservationMismatch)?
                .try_into()
                .unwrap(),
        );
        let source_bindings = u64::from_le_bytes(
            old_ground
                .bytes
                .get(24..32)
                .ok_or(SemanticTransitionError::ObservationMismatch)?
                .try_into()
                .unwrap(),
        );
        let source_rule_count = u64::from_le_bytes(
            old_ground
                .bytes
                .get(48..56)
                .ok_or(SemanticTransitionError::ObservationMismatch)?
                .try_into()
                .unwrap(),
        );
        if source_queries != context.source_queries as u64 {
            return Err(publication_input_error(
                "selected source query geometry changed its original program",
            ));
        }
        let source_layout = task_ground::TaskGroundLayout::new(
            usize::try_from(source_queries)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            usize::try_from(source_bindings)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            context
                .source_program_admission
                .as_ref()
                .map(|program| program.initial_rules.len()),
        )?;
        source_layout.validate(&old_ground.bytes)?;
        let source_rule_offset = source_layout.rule_offset;
        let rule_bytes = usize::try_from(source_rule_count)
            .ok()
            .and_then(|count| count.checked_mul(size_of::<crate::SemanticProgramRule>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if source_rule_count > ground_layout.rule_capacity as u64 {
            return Err(publication_input_error(
                "inherited selected program exceeds the frozen evaluation context",
            ));
        }
        let selected_rules = old_ground
            .bytes
            .get(
                source_rule_offset
                    ..source_rule_offset
                        .checked_add(rule_bytes)
                        .ok_or(SemanticTransitionError::GenerationExhausted)?,
            )
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let mut ground = ground_layout.initial_bytes();
        ground[48..56].copy_from_slice(&source_rule_count.to_le_bytes());
        ground[ground_layout.rule_offset..ground_layout.rule_offset + rule_bytes]
            .copy_from_slice(selected_rules);
        ground_layout.validate(&ground)?;
        replace_evaluation_record(
            &mut material,
            budget,
            SemanticStateRole::TaskGround,
            0,
            ground,
            ground_layout.bytes,
        )?;
        let statements = self.feedback_statement_bytes()?;
        material
            .ranges
            .retain(|item| item.range.role != SemanticStateRole::FeedbackStatements as u64);
        material.role_counts[SemanticStateRole::FeedbackStatements as usize - 1] =
            statements.len() as u64;
        for (index, statement) in statements.into_iter().enumerate() {
            replace_evaluation_record(
                &mut material,
                budget,
                SemanticStateRole::FeedbackStatements,
                index as u64,
                statement.to_vec(),
                statement.len(),
            )?;
        }
        replace_evaluation_record(
            &mut material,
            budget,
            SemanticStateRole::ActionCodebooks,
            0,
            publication_abi_bytes(&self.codebooks.words),
            self.codebooks.words.len() * 8,
        )?;
        let mut seen = BTreeSet::new();
        for record in target_records {
            if !matches!(
                record.role,
                SemanticStateRole::Environment
                    | SemanticStateRole::AuthorityEnvelope
                    | SemanticStateRole::AuthorityDecisions
                    | SemanticStateRole::CheckpointLineage
            ) || record.index != 0
                || !seen.insert(record.role as u64)
            {
                return Err(publication_input_error(
                    "evaluation authority records differ from the bound target envelope",
                ));
            }
            replace_evaluation_record(
                &mut material,
                budget,
                record.role,
                record.index,
                record.bytes,
                record.capacity_bytes,
            )?;
        }
        if seen != BTreeSet::from([37, 38, 39, 40]) {
            return Err(publication_input_error(
                "evaluation fork omits its original target authority or environment",
            ));
        }
        if material.contract.prefix_capacity > target.prefix_capacity
            || material.contract.feedback_capacity > target.feedback_capacity
            || material.bank.header.prefix_extent > target.max_position
            || material.contract.pad_token != target.pad_token
        {
            return Err(publication_input_error(
                "inherited source material exceeds the original evaluation task envelope",
            ));
        }
        let intent = initial_intent_records(
            &target.intent_effect,
            target.intent_entry_capacity,
            target.intent_payload_capacity_bytes,
            target.final_intent_payload_bytes,
        )?;
        let acknowledgement = initial_acknowledgement_record(
            target.intent_entry_capacity,
            target.acknowledgement_payload_capacity_bytes,
            target.intent_payload_capacity_bytes,
            target.intent_effect.len(),
        )?;
        for record in intent
            .into_iter()
            .chain(std::iter::once(acknowledgement))
            .chain(initial_replay_append_records(None, None)?)
        {
            replace_evaluation_record(
                &mut material,
                budget,
                record.role,
                record.index,
                record.bytes,
                record.capacity_bytes,
            )?;
        }
        for role in [
            SemanticStateRole::RawFeedback,
            SemanticStateRole::FeedbackSupportProvenance,
        ] {
            let records = material
                .ranges
                .iter()
                .filter(|item| item.range.role == role as u64)
                .map(|item| (item.range.index, item.bytes.len(), item.capacity))
                .collect::<Vec<_>>();
            for (index, length, capacity) in records {
                replace_evaluation_record(
                    &mut material,
                    budget,
                    role,
                    index,
                    vec![0; length],
                    capacity,
                )?;
            }
        }
        material.contract.task_identity = target_task;
        material.contract.authority_generation = self.task_epoch;
        material.bank.header.authority_generation = self.task_epoch;
        material.contract.final_intent_payload_bytes = target.final_intent_payload_bytes;
        material.contract.max_position = target.max_position;
        material.contract.nonterminal_intent_mode = u64::from(target.nonterminal_intent.is_some());
        material.contract.nonterminal_intent_plan_digest = target
            .nonterminal_intent
            .map_or(Identity256::default(), |binding| binding.plan_digest);
        material.contract.nonterminal_intent_prefix_begin = target
            .nonterminal_intent
            .map_or(0, |binding| binding.candidate_prefix_begin);
        material.contract.nonterminal_intent_prefix_end = target
            .nonterminal_intent
            .map_or(0, |binding| binding.candidate_prefix_end);
        material.contract.replay_row_capacity = 0;
        material.contract.replay_original_slot_boundary = 0;
        material.contract.replay_raw_capacity_bytes = 0;
        material.contract.replay_payload_capacity_bytes = 0;
        material.contract.replay_metadata_bytes = 0;
        material.contract.replay_max_material_bytes = 0;
        material.contract.replay_total_material_bytes = 0;
        material.contract.replay_max_evidence_bytes = 0;
        material.contract.replay_reservation_identity =
            crate::semantic_training_view::replay_reservation_identity(None, None);
        material.terminals = target.terminal_tokens.clone();
        material.contract.terminal_token_count = material.terminals.len() as u64;
        material.bank.header.terminal = 0;
        material.bank.header.terminal_position = 0;
        material.bank.header.fuel = target.fuel;
        material.bank.header.proposal = u64::from(target.rng.proposal);
        material.bank.header.stream_serial = target.rng.stream_serial;
        material.bank.header.family_id = u64::from(target.rng.family_id);
        material.bank.header.training_cursor = 0;
        material.bank.header.training_rng = [0; 4];
        material.bank.state = DeviceState::default();
        material.bank.receipts = [SemanticTransitionReceipt::default(); COMPONENT_COUNT];
        material
            .ranges
            .sort_by_key(|item| (item.range.role, item.range.index));
        material.contract.range_capacity = material.ranges.len() as u64;
        material.bank.header.range_count = material.contract.range_capacity;
        let capacities = material
            .ranges
            .iter()
            .filter(|item| item.range.role != 43)
            .map(|item| (item.range.role, item.range.index, item.capacity))
            .collect::<Vec<_>>();
        let runtime = initial_runtime_contract_record(
            material.contract,
            &material.role_counts,
            &material.terminals,
            &material.layouts,
            capacities,
            &mode,
            &material.model_memory,
        )?;
        replace_evaluation_record(
            &mut material,
            budget,
            runtime.role,
            runtime.index,
            runtime.bytes,
            runtime.capacity_bytes,
        )?;
        material.evaluation_fork = Some(origin.clone());
        Ok(SemanticEvaluationForkPreparation {
            issuer: Arc::clone(&self.publication_issuer),
            target_task,
            origin,
            material: Some(material),
            started: false,
            _reservation: reservation,
        })
    }
}

impl SemanticReplayStateMaterialProof {
    pub fn evaluation_fork_origin(&self) -> Option<&SemanticEvaluationForkOrigin> {
        self.owner.evaluation_fork.as_ref()
    }
}

/// Original cold target envelope, retained before catalogue selection. Model and
/// cache operands still come exclusively from the selected full source archive.
#[derive(Clone)]
pub struct SemanticEvaluationForkTarget {
    pub prefix_capacity: u64,
    pub feedback_capacity: u64,
    pub max_position: u64,
    pub pad_token: u64,
    pub terminal_tokens: Vec<u64>,
    pub final_intent_payload_bytes: u64,
    pub nonterminal_intent: Option<SemanticNonterminalIntentBinding>,
    pub intent_effect: Vec<u8>,
    pub intent_entry_capacity: u64,
    pub intent_payload_capacity_bytes: usize,
    pub acknowledgement_payload_capacity_bytes: usize,
    pub fuel: u64,
    pub rng: SemanticRngBinding,
    pub environment: Vec<u8>,
    pub checkpoint_lineage: Vec<u8>,
    pub authority_decisions_capacity_bytes: usize,
}

/// Original same-pool authority/environment records of the bound target.
pub struct SemanticEvaluationForkTargetRecords {
    issuer: Arc<()>,
    task: Identity256,
    records: Vec<SemanticStateRecord>,
    _reservation: SemanticReplayMaterialReservation,
}

impl SemanticTransitionSession {
    pub fn evaluation_fork_target_records(
        &self,
        authority: &[u8],
        snapshot: &[u8],
    ) -> Result<SemanticEvaluationForkTargetRecords, SemanticTransitionError> {
        self.ensure_rebindable()?;
        if self.publication.is_some()
            || self.initial_prefill.is_some()
            || self.evaluation_fork.is_some()
        {
            return Err(publication_input_error(
                "evaluation target authority must bind before publication or prefill",
            ));
        }
        let task = self
            .task_evaluation_identity()
            .ok_or(SemanticTransitionError::NotBound)?;
        let context = self
            .evaluation_context
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let original = context
            .target
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let original = &original
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .value;
        if authority.is_empty()
            || snapshot.is_empty()
            || snapshot.len() > original.authority_decisions_capacity_bytes
        {
            return Err(publication_input_error(
                "evaluation authority snapshot exceeds its original finite target envelope",
            ));
        }
        let bytes = authority
            .len()
            .checked_add(snapshot.len())
            .and_then(|bytes| bytes.checked_add(original.environment.len()))
            .and_then(|bytes| bytes.checked_add(original.checkpoint_lineage.len()))
            .and_then(|bytes| {
                bytes.checked_add(
                    4 * size_of::<SemanticStateRecord>()
                        + size_of::<SemanticEvaluationForkTargetRecords>(),
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .reserve_auxiliary(bytes as u64)?;
        Ok(SemanticEvaluationForkTargetRecords {
            issuer: Arc::clone(&self.publication_issuer),
            task,
            records: vec![
                SemanticStateRecord {
                    role: SemanticStateRole::Environment,
                    index: 0,
                    bytes: original.environment.clone(),
                    capacity_bytes: original.environment.len(),
                },
                SemanticStateRecord {
                    role: SemanticStateRole::AuthorityEnvelope,
                    index: 0,
                    bytes: authority.to_vec(),
                    capacity_bytes: authority.len(),
                },
                SemanticStateRecord {
                    role: SemanticStateRole::AuthorityDecisions,
                    index: 0,
                    bytes: snapshot.to_vec(),
                    capacity_bytes: original.authority_decisions_capacity_bytes,
                },
                SemanticStateRecord {
                    role: SemanticStateRole::CheckpointLineage,
                    index: 0,
                    bytes: original.checkpoint_lineage.clone(),
                    capacity_bytes: original.checkpoint_lineage.len(),
                },
            ],
            _reservation: reservation,
        })
    }
}

struct BoundEvaluationForkTarget {
    value: SemanticEvaluationForkTarget,
    _reservation: SemanticReplayMaterialReservation,
}

impl SemanticTransitionSession {
    pub fn bind_evaluation_fork_target(
        &mut self,
        target: SemanticEvaluationForkTarget,
    ) -> Result<(), SemanticTransitionError> {
        let context = self
            .evaluation_context
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let preseal = self
            .model_context_preseal
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        self.ensure_rebindable_except_model_context_preseal(preseal)?;
        if self.codebooks.model_catalogue.is_some()
            || self.task.is_some()
            || self.publication.is_some()
            || target.fuel == 0
            || target.prefix_capacity == 0
            || target.feedback_capacity == 0
            || target.max_position == 0
            || target.terminal_tokens.is_empty()
        {
            return Err(publication_input_error(
                "evaluation target envelope must bind to its original cold context before freeze",
            ));
        }
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let bytes = target
            .intent_effect
            .capacity()
            .checked_mul(2)
            .and_then(|bytes| {
                bytes.checked_add(
                    target
                        .terminal_tokens
                        .capacity()
                        .checked_mul(size_of::<u64>())?,
                )
            })
            .and_then(|bytes| bytes.checked_add(size_of::<BoundEvaluationForkTarget>() + 1024))
            .and_then(|bytes| bytes.checked_add(target.environment.capacity()))
            .and_then(|bytes| bytes.checked_add(target.checkpoint_lineage.capacity()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = budget.reserve_auxiliary(bytes as u64)?;
        initial_intent_records(
            &target.intent_effect,
            target.intent_entry_capacity,
            target.intent_payload_capacity_bytes,
            target.final_intent_payload_bytes,
        )?;
        initial_acknowledgement_record(
            target.intent_entry_capacity,
            target.acknowledgement_payload_capacity_bytes,
            target.intent_payload_capacity_bytes,
            target.intent_effect.len(),
        )?;
        let mut original = context
            .target
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original.is_some() {
            return Err(publication_input_error(
                "evaluation target envelope is already bound",
            ));
        }
        *original = Some(BoundEvaluationForkTarget {
            value: target,
            _reservation: reservation,
        });
        Ok(())
    }
}

impl SemanticEvaluationForkOrigin {
    pub(super) fn encoded_bytes(&self) -> Result<usize, SemanticTransitionError> {
        Ok(188)
    }

    pub(super) fn write(
        origin: Option<&Self>,
        mut write: impl FnMut(&[u8]) -> Result<(), SemanticTransitionError>,
    ) -> Result<(), SemanticTransitionError> {
        let Some(origin) = origin else {
            return write(&0u32.to_le_bytes());
        };
        write(&1u32.to_le_bytes())?;
        write(origin.source.instance.as_bytes())?;
        write(&origin.source.word.to_le_bytes())?;
        write(origin.source.logical_digest.as_bytes())?;
        write(origin.source_task.as_bytes())?;
        write(origin.catalogue.as_bytes())?;
        write(&(origin.member as u64).to_le_bytes())?;
        write(origin.native_digest.as_bytes())?;
        write(&origin.native_extent.to_le_bytes())
    }

    pub(super) fn decode(
        reader: &mut SemanticMaterialReader<'_>,
    ) -> Result<Option<Self>, SemanticTransitionError> {
        match reader.u32().map_err(SemanticTransitionError::Semantic)? {
            0 => return Ok(None),
            1 => {}
            _ => {
                return Err(publication_input_error(
                    "invalid evaluation fork source reference",
                ))
            }
        }
        let source = decode_publication_identity(reader)?;
        let mut identity = || -> Result<Identity256, SemanticTransitionError> {
            Ok(Identity256::from_bytes(
                reader
                    .take(32)
                    .map_err(SemanticTransitionError::Semantic)?
                    .try_into()
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
            ))
        };
        let source_task = identity()?;
        let catalogue = identity()?;
        let member = usize::try_from(reader.u64().map_err(SemanticTransitionError::Semantic)?)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let native_digest = Identity256::from_bytes(
            reader
                .take(32)
                .map_err(SemanticTransitionError::Semantic)?
                .try_into()
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
        );
        let native_extent = reader.u64().map_err(SemanticTransitionError::Semantic)?;
        if member < 2
            || native_digest == Identity256::default()
            || native_extent == 0
            || source_task == Identity256::default()
            || catalogue == Identity256::default()
        {
            return Err(publication_input_error(
                "incomplete evaluation fork source reference",
            ));
        }
        Ok(Some(Self {
            source,
            source_task,
            catalogue,
            member,
            native_digest,
            native_extent,
            material: None,
        }))
    }
}

impl SemanticReplayLoadedMaterial {
    /// Bind the exact original guarded donor section through the existing full
    /// decoder. Publication restoration cannot execute an unresolved reference.
    pub fn bind_evaluation_fork_source(
        &mut self,
        mut source: SemanticReplayLoadedMaterial,
    ) -> Result<(), SemanticTransitionError> {
        let budget = self
            .bytes
            .budget()
            .ok_or(SemanticTransitionError::NotBound)?
            .clone();
        source.require_budget(&budget)?;
        let digest = Identity256::from_bytes(Sha256::digest(source.as_bytes()).into());
        let extent = source.as_bytes().len() as u64;
        self.complete()?;
        let current = self
            .decoded
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let origin = current
            .evaluation_fork
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if origin.native_digest != digest || origin.native_extent != extent {
            return Err(publication_input_error(
                "evaluation source differs from its original guarded material reference",
            ));
        }
        let archive = source.bytes.clone();
        let donor = source.complete()?;
        let identity = SemanticPublishedIdentity {
            instance: donor.bank.header.instance,
            word: donor.bank.header.publication_word,
            logical_digest: donor.bank.header.logical_digest,
        };
        if donor.evaluation_fork.is_some()
            || identity != origin.source
            || donor.contract.task_identity != origin.source_task
            || !current
                .graph
                .records
                .predicates
                .starts_with(&donor.graph.records.predicates)
            || !current
                .graph
                .records
                .records
                .starts_with(&donor.graph.records.records)
            || !current
                .graph
                .records
                .supports
                .starts_with(&donor.graph.records.supports)
            || !current.graph.symbols.starts_with(&donor.graph.symbols)
            || !current
                .graph
                .insertions
                .starts_with(&donor.graph.insertions)
        {
            return Err(publication_input_error(
                "evaluation source changed the selected complete donor world or task",
            ));
        }
        let codebook = donor
            .ranges
            .iter()
            .find(|range| (range.range.role, range.range.index) == (48, 0))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let header = codebook
            .bytes
            .get(..model_catalogue::HEADER_WORDS * 8)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let word =
            |index: usize| u64::from_le_bytes(header[index * 8..index * 8 + 8].try_into().unwrap());
        let catalogue =
            Identity256::from_bytes(std::array::from_fn(|index| header[25 * 8 + index]));
        if catalogue != origin.catalogue
            || word(42) != (SELECTED_TRAINING_CONTEXT + 1) as u64
            || word(43) <= origin.member as u64
        {
            return Err(publication_input_error(
                "evaluation source is not the original selected training context",
            ));
        }
        self.decoded
            .as_mut()
            .ok_or(SemanticTransitionError::NotBound)?
            .evaluation_fork
            .as_mut()
            .ok_or(SemanticTransitionError::NotBound)?
            .material = Some(archive);
        Ok(())
    }
}

pub(super) struct ColdEvaluationContext {
    source_issuer: Arc<()>,
    source_program: Option<Identity256>,
    source_program_admission: Option<Arc<crate::SemanticProgramAdmission>>,
    source_queries: usize,
    target_program: Identity256,
    pub(super) program: Arc<crate::SemanticProgramAdmission>,
    pub(super) target_record_offset: u32,
    admission: Mutex<Option<crate::semantic_hypergraph::PreparedEvaluationAdmission>>,
    target: Mutex<Option<BoundEvaluationForkTarget>>,
    _reservation: SemanticReplayMaterialReservation,
}

/// Durable original source custody, distinct from the active evaluation task.
#[derive(Clone)]
pub struct SemanticEvaluationForkOrigin {
    pub(super) source: SemanticPublishedIdentity,
    pub(super) source_task: Identity256,
    pub(super) catalogue: Identity256,
    pub(super) member: usize,
    pub(super) native_digest: Identity256,
    pub(super) native_extent: u64,
    pub(super) material: Option<PublicationMaterialBytes>,
}

impl SemanticEvaluationForkOrigin {
    pub fn source_identity(&self) -> SemanticPublishedIdentity {
        self.source
    }
    pub fn source_task_identity(&self) -> Identity256 {
        self.source_task
    }
    pub fn member(&self) -> usize {
        self.member
    }
    pub fn source_material_reference(&self) -> (Identity256, u64) {
        (self.native_digest, self.native_extent)
    }
    pub fn is_resolved(&self) -> bool {
        self.material.is_some()
    }
    pub fn source_material(&self) -> Result<SemanticReplayLoadedMaterial, SemanticTransitionError> {
        let material = self.material.as_ref().ok_or_else(|| {
            publication_input_error(
                "resolve the original guarded evaluation source before restoration",
            )
        })?;
        let budget = material.budget().ok_or(SemanticTransitionError::NotBound)?;
        let metadata = Arc::new(budget.reserve_auxiliary(
            (size_of::<SemanticReplayLoadedMaterial>() + 4 * size_of::<usize>()) as u64,
        )?);
        SemanticReplayLoadedMaterial {
            bytes: material.slice_bytes(0..material.len())?,
            decoded: None,
        }
        .retain_shared_metadata(metadata)
    }
}

/// One bound target and its original complete derivation, retained across unknown
/// completion. It does not issue a second TaskUse or a partial runnable session.
pub struct SemanticEvaluationForkPreparation {
    issuer: Arc<()>,
    target_task: Identity256,
    pub(super) origin: SemanticEvaluationForkOrigin,
    pub(super) material: Option<OwnedPublicationMaterial>,
    started: bool,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

impl SemanticEvaluationForkPreparation {
    pub fn origin(&self) -> &SemanticEvaluationForkOrigin {
        &self.origin
    }
}

impl SemanticTransitionSession {
    pub fn cold_admission_records(
        &self,
    ) -> Result<&SemanticAdmissionRecords, SemanticTransitionError> {
        self.ensure_quiescent_except_cold_report(None)?;
        Ok(self
            .graph
            .admission()
            .ok_or(SemanticTransitionError::NotBound)?
            .records())
    }

    pub fn observed_cold_task_content(
        &self,
    ) -> Result<(SemanticTaskContentIdentity, Vec<crate::SemanticTruth>), SemanticTransitionError>
    {
        self.ensure_quiescent_except_cold_report(None)?;
        self.cold_content
            .clone()
            .ok_or(SemanticTransitionError::NotBound)
    }

    pub(super) fn require_prepared_evaluation_model_context(
        &self,
        source: &Self,
    ) -> Result<(), SemanticTransitionError> {
        let context = self.evaluation_context.as_ref().ok_or_else(|| {
            publication_input_error("evaluation member has no original composed source context")
        })?;
        if !Arc::ptr_eq(&context.source_issuer, &source.publication_issuer)
            || context
                .admission
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?
                .is_some()
            || self.codebooks.program_binding
                != Some(ActionCodebooks::program_binding(&context.program))
            || self.cold_content.is_none()
            || context
                .target
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?
                .is_none()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub fn prepare_evaluation_model_context(
        &mut self,
        source: &Self,
        source_program: Option<&crate::SemanticProgramAdmission>,
        target_program: &crate::SemanticProgramAdmission,
        limits: SemanticAdmissionLimits,
    ) -> Result<(Arc<crate::SemanticProgramAdmission>, u32), SemanticTransitionError> {
        let preseal = self.require_original_model_context_preseal(source)?;
        let ordinal = preseal
            .member_ordinal(&self.publication_issuer)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        // An entered codebook upload has its own original completion resolver;
        // it may be joined before the enclosing preseal is released.
        if let Some(context) = self.evaluation_context.clone() {
            if !Arc::ptr_eq(&context.source_issuer, &source.publication_issuer)
                || context.source_program != source_program.map(ActionCodebooks::program_binding)
                || ![
                    context.target_program,
                    ActionCodebooks::program_binding(&context.program),
                ]
                .contains(&ActionCodebooks::program_binding(target_program))
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            if self.codebooks.program_binding
                != Some(ActionCodebooks::program_binding(&context.program))
            {
                self.ensure_rebindable_except_model_context_preseal(&preseal)?;
                let original = context
                    .admission
                    .lock()
                    .map_err(|_| SemanticTransitionError::Poisoned)?;
                let admission = original
                    .as_ref()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let codebooks = ActionCodebooks::derive_with_material_budget(
                    &admission.admission,
                    self.graph.transition_arena()[1],
                    Some(&context.program),
                    self.replay_material_budget
                        .as_ref()
                        .ok_or(SemanticTransitionError::NotBound)?,
                )?;
                drop(original);
                model_catalogue::install_cold_action_codebooks(self, codebooks)?;
            } else {
                self.resolve_constructor_codebooks(true)?;
            }
            let prepared = context
                .admission
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?
                .take();
            if let Some(prepared) = prepared {
                self.graph.install_evaluation_admission(prepared);
                self.cold_content = None;
                self.cold_statement_records.clear();
                self.cold_statement_reservation = None;
                self.cold_observer_entered = false;
                self.cold_observer_completed = false;
                self.cold_canary_source = None;
            }
            return Ok((Arc::clone(&context.program), context.target_record_offset));
        }
        self.ensure_rebindable_except_model_context_preseal(&preseal)?;
        source.ensure_rebindable_except_model_context_preseal(&preseal)?;
        for session in [&*self, source] {
            if session.task.is_some()
                || session.publication.is_some()
                || session.initial_prefill.is_some()
                || session.captured.is_some()
                || !session.model_owners.is_empty()
                || session.codebooks.model_catalogue.is_some()
                || session.next_proposal != 0
                || session.text_binding.is_some()
            {
                return Err(publication_input_error(
                    "evaluation context must compose before task import and model use",
                ));
            }
        }
        if self.codebooks.program_binding != Some(ActionCodebooks::program_binding(target_program))
            || source.codebooks.program_binding
                != source_program.map(ActionCodebooks::program_binding)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if source
            .replay_material_budget
            .as_ref()
            .is_none_or(|other| !budget.same_pool(other))
            || !Arc::ptr_eq(self.provider.memory(), source.provider.memory())
            || self.domain.stream_id() != source.domain.stream_id()
        {
            return Err(publication_input_error(
                "evaluation context changed its original allocation domain",
            ));
        }
        let extent = crate::SemanticProgramAdmission::evaluation_context_allocation_bytes_from_optional_source(
            source_program,
            target_program,
        )
        .map_err(publication_input_error)?
        .checked_add(size_of::<ColdEvaluationContext>() + 8 * size_of::<usize>())
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = budget.reserve_auxiliary(extent as u64)?;
        // The scope authenticates the finite presealed member using the same
        // physical-coordinate normalization as the frozen model catalogue.
        // Reconstructing this roster creates the same immutable task meanings.
        let mut scope = Sha256::new();
        scope.update(b"xlog.semantic.evaluation-context.v1\0");
        scope.update((ordinal as u64).to_le_bytes());
        for codebooks in [&source.codebooks, &self.codebooks] {
            scope.update((codebooks.words.len() as u64).to_le_bytes());
            for (index, &word) in codebooks.words.iter().enumerate() {
                let normalized = if matches!(index, 14..=16 | 21..=24) {
                    0
                } else {
                    word
                };
                scope.update(normalized.to_le_bytes());
            }
        }
        let scope = Identity256::from_bytes(scope.finalize().into());
        let admission = self
            .graph
            .prepare_evaluation_admission(&source.graph, limits, scope)
            .map_err(SemanticTransitionError::Semantic)?;
        let target_record_offset = admission.target_record_offset;
        let program = Arc::new(
            crate::SemanticProgramAdmission::evaluation_context_from_optional_source(
                source_program,
                target_program,
                target_record_offset,
            )
            .map_err(publication_input_error)?,
        );
        let codebooks = ActionCodebooks::derive_with_material_budget(
            &admission.admission,
            self.graph.transition_arena()[1],
            Some(&program),
            budget,
        )?;
        let context = Arc::new(ColdEvaluationContext {
            source_issuer: Arc::clone(&source.publication_issuer),
            source_program: source_program.map(ActionCodebooks::program_binding),
            source_program_admission: source_program.map(|program| Arc::new(program.clone())),
            source_queries: source.cold_statement_records.len(),
            target_program: ActionCodebooks::program_binding(target_program),
            program: Arc::clone(&program),
            target_record_offset,
            admission: Mutex::new(Some(admission)),
            target: Mutex::new(None),
            _reservation: reservation,
        });
        self.evaluation_context = Some(Arc::clone(&context));
        // The candidate admission and all host capacity receipts are retained
        // before the canonical installer may submit its first original upload.
        model_catalogue::install_cold_action_codebooks(self, codebooks)?;
        let prepared = context
            .admission
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .take()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        self.graph.install_evaluation_admission(prepared);
        self.cold_content = None;
        self.cold_statement_records.clear();
        self.cold_statement_reservation = None;
        self.cold_observer_entered = false;
        self.cold_observer_completed = false;
        self.cold_canary_source = None;
        Ok((program, target_record_offset))
    }
}
