use super::*;

#[cfg(feature = "semantic-policy")]
mod model_context_checkpoint;
#[cfg(feature = "semantic-policy")]
pub use model_context_checkpoint::{
    SemanticPrivateModelContextCheckpointOrigin, SemanticPrivateModelContextCheckpointSave,
};

impl SemanticTransitionSession {
    #[cfg(feature = "semantic-policy")]
    pub(super) fn admit_original_model_context_preparation(
        &self,
        source: &Self,
        context: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        admission: &SemanticSegmentInstructionAdmission,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        let preparer = self
            .publication_preparation
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let mut original = preparer
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let allowance = source.admit_model_context_publication_dma(
            context,
            self,
            parent,
            admission,
            work,
            preparer,
            &original.memory,
            &original.uploads,
            original.upload_staging_bytes,
            &original.writes,
        )?;
        if original
            .context_dma_allowance
            .as_ref()
            .is_some_and(|prior| !Arc::ptr_eq(prior, &allowance))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        original.context_dma_allowance = Some(allowance);
        Ok(())
    }

    /// Authenticate the exact checkpoint material for an active original reader.
    /// A fresh restored instance is accepted only through its completed native
    /// restore owner, never by logical-digest equivalence alone.
    pub fn verify_checkpoint_source_material(
        &self,
        parent: &SemanticPublishedLease,
        bytes: &[u8],
    ) -> Result<(Identity256, bool), SemanticTransitionError> {
        self.ensure_rebindable()?;
        self.checked_reader(parent)?;
        if self.replay_material_budget.is_some() {
            return Err(publication_input_error(
                "bound checkpoint verification requires its original charged material",
            ));
        }
        let material = OwnedPublicationMaterial::decode(bytes)?;
        self.verify_checkpoint_source_content(parent, &material, bytes)
    }

    /// Authenticate the same charged source without cloning its publication bytes.
    pub fn verify_checkpoint_loaded_source_material(
        &self,
        parent: &SemanticPublishedLease,
        loaded: &mut SemanticReplayLoadedMaterial,
    ) -> Result<(Identity256, bool), SemanticTransitionError> {
        self.ensure_rebindable()?;
        self.checked_reader(parent)?;
        loaded.require_budget(self.replay_material_budget.as_ref().ok_or_else(|| {
            publication_input_error("checkpoint source lost its original capacity pool")
        })?)?;
        let digest = Identity256::from_bytes(Sha256::digest(loaded.as_bytes()).into());
        let header = loaded.complete()?.bank.header;
        self.require_checkpoint_source_digest(
            parent,
            SemanticPublishedIdentity {
                instance: header.instance,
                word: header.publication_word,
                logical_digest: header.logical_digest,
                state_digest: header.state_digest,
            },
            digest,
        )
    }

    fn verify_checkpoint_source_content(
        &self,
        parent: &SemanticPublishedLease,
        material: &OwnedPublicationMaterial,
        bytes: &[u8],
    ) -> Result<(Identity256, bool), SemanticTransitionError> {
        let header = material.bank.header;
        let source = SemanticPublishedIdentity {
            instance: header.instance,
            word: header.publication_word,
            logical_digest: header.logical_digest,
            state_digest: header.state_digest,
        };
        let digest = Identity256::from_bytes(Sha256::digest(bytes).into());
        self.require_checkpoint_source_digest(parent, source, digest)
    }

    pub(super) fn verify_replay_source_material(
        &self,
        parent: &SemanticPublishedLease,
        material: &SemanticReplayMaterial,
    ) -> Result<(Identity256, bool), SemanticTransitionError> {
        self.ensure_rebindable()?;
        self.checked_reader(parent)?;
        self.require_checkpoint_source_digest(
            parent,
            material.predecessor_identity(),
            material.predecessor_material_digest(),
        )
    }

    pub(super) fn require_checkpoint_source_digest(
        &self,
        parent: &SemanticPublishedLease,
        source: SemanticPublishedIdentity,
        digest: Identity256,
    ) -> Result<(Identity256, bool), SemanticTransitionError> {
        if source == parent.identity {
            let storage = self
                .publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let sources = storage
                .material_sources
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?;
            let source = &sources[(parent.identity.word & 1) as usize];
            if source.identity != Some(parent.identity) || !source.digests.contains(&digest) {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            return Ok((digest, false));
        }
        self.require_model_context_checkpoint_anchor_digest(parent.identity, source, digest)?;
        Ok((digest, true))
    }

    /// Authenticate only the immutable fresh restoration anchor against its
    /// original complete SOURCE archive; successor authority is checked elsewhere.
    pub(super) fn require_model_context_checkpoint_anchor_digest(
        &self,
        anchor: SemanticPublishedIdentity,
        source: SemanticPublishedIdentity,
        digest: Identity256,
    ) -> Result<(), SemanticTransitionError> {
        let restoration = self
            .state_material_restore
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if !Arc::ptr_eq(&restoration.issuer, &self.publication_issuer)
            || restoration.pending.load(Ordering::Acquire)
            || restoration.source_digest != digest
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let original = restoration
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original.stage != StateRestoreStage::Completed
            || original.transitioned
            || original.retain_current_models
            || original.result != Some(anchor)
            || source
                != (SemanticPublishedIdentity {
                    instance: original.material.bank.header.instance,
                    word: original.material.bank.header.publication_word,
                    logical_digest: original.material.bank.header.logical_digest,
                    state_digest: original.material.bank.header.state_digest,
                })
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub(super) fn require_restored_replay_append_material(
        &self,
        owner: &Arc<OriginalStateMaterialRestore>,
        material: &RestoredReplayAppendMaterial,
    ) -> Result<(), SemanticTransitionError> {
        if !Arc::ptr_eq(&owner.issuer, &self.publication_issuer)
            || self
                .state_material_restore
                .as_ref()
                .is_none_or(|original| !Arc::ptr_eq(original, owner))
            || !Arc::ptr_eq(&owner.issuance, &material.restoration)
            || !owner.pending.load(Ordering::Acquire)
            || !owner.initialized.load(Ordering::Acquire)
            || self
                .publication
                .as_ref()
                .is_none_or(|publication| publication.instance != owner.instance)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    fn require_state_material_restore_entry(&self) -> Result<(), SemanticTransitionError> {
        #[cfg(feature = "semantic-policy")]
        if let Some(context) = self.actor_refresh_context_issuance() {
            self.ensure_quiescent_except_actor_context(&context)?;
            if self.initial_prefill.is_some() {
                return Err(publication_input_error(
                    "an imported task's initial prefill stage cannot be rebound",
                ));
            }
            return self.ensure_rebinding_ownership();
        }
        self.ensure_rebindable()
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn require_dormant_model_context_restore_entry(
        &self,
        continuation: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
    ) -> Result<Arc<Mutex<ColdNativeAllowance>>, SemanticTransitionError> {
        let allowance = self.require_dormant_model_context_restore_work(continuation, work)?;
        if self.initial_prefill.is_some()
            || self.captured.is_some()
            || self.publication.is_some()
            || self.state_material_restore.is_some()
            || self.actor_refresh_context_issuance().is_some()
        {
            return Err(publication_input_error(
                "dormant restoration requires its original fresh target",
            ));
        }
        self.ensure_rebinding_ownership()?;
        if !self.dormant_model_context_restore_may_submit(continuation, work, &allowance)
            || !self.original_restoration_may_submit(None, None)
        {
            return Err(SemanticTransitionError::Poisoned);
        }
        Ok(allowance)
    }

    pub(super) fn has_pending_state_restoration_except(
        &self,
        restoration: Option<&Arc<OriginalStateMaterialRestore>>,
        preparation: Option<&Arc<OriginalPublicationPreparation>>,
    ) -> bool {
        self.state_material_restore.as_ref().is_some_and(|owner| {
            owner.pending.load(Ordering::Acquire)
                && restoration.is_none_or(|original| {
                    !Arc::ptr_eq(owner, original)
                        || !Arc::ptr_eq(&original.issuer, &self.publication_issuer)
                })
        }) || self.publication_preparation.as_ref().is_some_and(|owner| {
            owner.pending.load(Ordering::Acquire)
                && preparation.is_none_or(|original| {
                    !Arc::ptr_eq(owner, original)
                        || !Arc::ptr_eq(&original.issuer, &self.publication_issuer)
                })
        })
    }

    pub(super) fn original_restoration_may_submit(
        &self,
        restoration: Option<&Arc<OriginalStateMaterialRestore>>,
        preparation: Option<&Arc<OriginalPublicationPreparation>>,
    ) -> bool {
        self.original_restoration_may_submit_with_context(
            restoration,
            preparation,
            restoration.and_then(|owner| owner.context.as_ref()),
        )
    }

    fn original_restoration_may_submit_with_context(
        &self,
        restoration: Option<&Arc<OriginalStateMaterialRestore>>,
        preparation: Option<&Arc<OriginalPublicationPreparation>>,
        context: Option<&Arc<()>>,
    ) -> bool {
        if restoration.is_some_and(|owner| {
            !Arc::ptr_eq(&owner.issuer, &self.publication_issuer)
                || self
                    .state_material_restore
                    .as_ref()
                    .is_none_or(|original| !Arc::ptr_eq(original, owner))
        }) || preparation.is_some_and(|owner| {
            !Arc::ptr_eq(&owner.issuer, &self.publication_issuer)
                || self
                    .publication_preparation
                    .as_ref()
                    .is_none_or(|original| !Arc::ptr_eq(original, owner))
                || match (&owner.restoration, restoration) {
                    (None, None) => false,
                    (Some(issuance), Some(original)) => !Arc::ptr_eq(issuance, &original.issuance),
                    _ => true,
                }
        }) {
            return false;
        }
        #[cfg(feature = "semantic-policy")]
        if context.is_some_and(|context| {
            self.actor_refresh_context_issuance()
                .is_none_or(|original| !Arc::ptr_eq(&original, context))
        }) {
            return false;
        }
        #[cfg(not(feature = "semantic-policy"))]
        if context.is_some() {
            return false;
        }
        #[cfg(feature = "semantic-policy")]
        if restoration
            .and_then(|owner| owner.dormant_context.as_ref())
            .is_some_and(|original| {
                context.is_some()
                    || !self.dormant_model_context_restore_may_submit(
                        &original.continuation,
                        &original.work,
                        &original.allowance,
                    )
            })
        {
            return false;
        }
        self.original_session_completion_may_submit()
            && !self
                .readers
                .values()
                .any(|reader| reader.retirement_pending)
            && !self
                .steps
                .values()
                .any(|step| step.consumer_completion.is_some())
            && !self.has_pending_state_restoration_except(restoration, preparation)
            && if restoration.is_some() {
                self.graph.original_root_restore_may_submit()
            } else {
                self.graph.ensure_not_poisoned().is_ok()
            }
            && {
                #[cfg(feature = "semantic-policy")]
                {
                    !self.has_pending_actor_refresh_context_except(context)
                }
                #[cfg(not(feature = "semantic-policy"))]
                {
                    true
                }
            }
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn ensure_quiescent_except_actor_context(
        &self,
        issuance: &Arc<()>,
    ) -> Result<(), SemanticTransitionError> {
        if self
            .actor_refresh_context_issuance()
            .is_none_or(|original| !Arc::ptr_eq(&original, issuance))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let restoration = self.state_material_restore.as_ref().filter(|owner| {
            owner
                .context
                .as_ref()
                .is_some_and(|context| Arc::ptr_eq(context, issuance))
        });
        let preparation = self.publication_preparation.as_ref().filter(|owner| {
            restoration.is_some_and(|restoration| {
                owner
                    .restoration
                    .as_ref()
                    .is_some_and(|original| Arc::ptr_eq(original, &restoration.issuance))
            })
        });
        if self.original_restoration_may_submit_with_context(
            restoration,
            preparation,
            Some(issuance),
        ) {
            Ok(())
        } else {
            Err(SemanticTransitionError::Poisoned)
        }
    }

    fn require_learning_phase_admission(
        &self,
        transition: Option<&SemanticLearningPhaseTransition>,
    ) -> Result<(), SemanticTransitionError> {
        if let Some(transition) = transition {
            let admission = transition.admission.as_ref().ok_or_else(|| {
                publication_input_error(
                    "learning transition requires its original admitted material",
                )
            })?;
            let budget = self
                .replay_material_budget
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            admission.require_budget(budget)?;
        }
        Ok(())
    }

    pub(super) fn restore_state_material_inner(
        &mut self,
        bytes: &[u8],
        transition: Option<&SemanticLearningPhaseTransition>,
        immutable_models: Option<RestoredModelOwners<'_>>,
    ) -> Result<Option<SemanticPublishedIdentity>, SemanticTransitionError> {
        self.require_state_material_restore_entry()?;
        self.require_learning_phase_admission(transition)?;
        if self.replay_material_budget.is_some() {
            return Err(publication_input_error(
                "bounded state restoration requires its original loaded material",
            ));
        }
        let material = OwnedPublicationMaterial::decode(bytes)?;
        self.restore_decoded_state_material(
            material,
            Identity256::from_bytes(Sha256::digest(bytes).into()),
            transition,
            immutable_models,
        )
    }

    pub fn restore_loaded_state_material(
        &mut self,
        loaded: SemanticReplayLoadedMaterial,
        transition: Option<&SemanticLearningPhaseTransition>,
    ) -> Result<Option<SemanticPublishedIdentity>, SemanticTransitionError> {
        self.restore_loaded_state_material_original(
            loaded,
            transition,
            #[cfg(feature = "semantic-policy")]
            None,
        )
    }

    #[cfg(feature = "semantic-policy")]
    pub fn restore_loaded_state_material_with_resident_restoration(
        &mut self,
        loaded: SemanticReplayLoadedMaterial,
        transition: Option<&SemanticLearningPhaseTransition>,
        restoration: &SemanticPreparedResidentModelRestoration,
    ) -> Result<Option<SemanticPublishedIdentity>, SemanticTransitionError> {
        self.restore_loaded_state_material_original(loaded, transition, Some(restoration.clone()))
    }

    fn restore_loaded_state_material_original(
        &mut self,
        mut loaded: SemanticReplayLoadedMaterial,
        transition: Option<&SemanticLearningPhaseTransition>,
        #[cfg(feature = "semantic-policy")] restoration: Option<
            SemanticPreparedResidentModelRestoration,
        >,
    ) -> Result<Option<SemanticPublishedIdentity>, SemanticTransitionError> {
        self.require_state_material_restore_entry()?;
        self.require_learning_phase_admission(transition)?;
        let budget = self.replay_material_budget.as_ref().ok_or_else(|| {
            publication_input_error("loaded restoration requires its original capacity pool")
        })?;
        if !loaded
            .bytes
            .budget()
            .is_some_and(|loaded_budget| budget.same_pool(loaded_budget))
        {
            return Err(publication_input_error(
                "loaded restoration changes its original capacity pool",
            ));
        }
        let digest = Identity256::from_bytes(Sha256::digest(loaded.as_bytes()).into());
        #[cfg(feature = "semantic-policy")]
        if let Some(original) = &restoration {
            original.require_loaded(&mut loaded)?;
        }
        let material = loaded.into_complete()?;
        self.restore_decoded_state_material_original(
            material,
            digest,
            transition,
            None,
            #[cfg(feature = "semantic-policy")]
            restoration,
            #[cfg(feature = "semantic-policy")]
            None,
        )
    }

    pub(super) fn restore_decoded_state_material(
        &mut self,
        material: OwnedPublicationMaterial,
        source_digest: Identity256,
        transition: Option<&SemanticLearningPhaseTransition>,
        immutable_models: Option<RestoredModelOwners<'_>>,
    ) -> Result<Option<SemanticPublishedIdentity>, SemanticTransitionError> {
        self.restore_decoded_state_material_original(
            material,
            source_digest,
            transition,
            immutable_models,
            #[cfg(feature = "semantic-policy")]
            None,
            #[cfg(feature = "semantic-policy")]
            None,
        )
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn restore_dormant_model_context_material(
        &mut self,
        material: OwnedPublicationMaterial,
        source_digest: Identity256,
        continuation: &SemanticModelContextContinuation,
        owners: &[Arc<ModelGenerationOwner>],
        minimum_generation: u64,
        work: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        let allowance = self.require_dormant_model_context_restore_entry(continuation, work)?;
        if material.bank.header.abi != 0
            || material.bank.header.publication_word != 0
            || minimum_generation == 0
            || minimum_generation > u64::from(u32::MAX)
            || owners.len() != continuation.inner.models.len()
            || owners
                .iter()
                .zip(&continuation.inner.models)
                .any(|(owner, issued)| !Arc::ptr_eq(owner, issued))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let result = self.restore_decoded_state_material_original(
            material,
            source_digest,
            None,
            Some(RestoredModelOwners::Current {
                owners,
                minimum_generation,
            }),
            None,
            Some(DormantStateRestoreCustody {
                continuation: continuation.clone(),
                work: work.clone(),
                allowance,
            }),
        )?;
        if result.is_some() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn dormant_model_context_restore_dma_ceiling(
        &self,
        material: &OwnedPublicationMaterial,
    ) -> Result<[u64; 9], SemanticTransitionError> {
        let geometry = publication_storage_geometry(
            &material.ranges,
            &material.layouts,
            &material.model_memory,
            &material.contract,
            self.replay_material_budget.as_ref(),
        )?;
        let mut commands = 0u64;
        let mut bytes = 0u64;
        let mut add = |count: usize, width: usize| -> Result<(), SemanticTransitionError> {
            let extent = count
                .checked_mul(width)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            if extent != 0 {
                commands = commands
                    .checked_add(1)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                bytes = bytes
                    .checked_add(
                        u64::try_from(extent)
                            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                    )
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
            }
            Ok(())
        };
        let (_, task) = self
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        add(task.len(), size_of::<u64>())?;
        add(self.device_codebooks.len(), size_of::<u64>())?;
        add(self.device_components.len(), size_of::<SemanticComponent>())?;
        // Current models retain their exact numerical allocations. Only the
        // canonical writable publication destinations are zeroed.
        for capacity in &geometry.allocation_bytes[material.model_memory.allocation_bytes.len()..] {
            add(*capacity, 1)?;
        }
        for _ in 0..2 {
            add(1, size_of::<PublicationBank>())?;
        }
        for directory in &geometry.bank_templates {
            add(directory.len(), size_of::<PublicationRange>())?;
        }
        add(
            geometry.allocation_bytes.len(),
            size_of::<PublicationStorageEntry>(),
        )?;
        add(1, size_of::<PublicationContract>())?;
        add(material.terminals.len(), size_of::<u64>())?;
        add(PUBLICATION_ROLE_COUNT, size_of::<PublicationRoleCount>())?;
        add(1, size_of::<PendingContinuation>())?;
        add(
            geometry.continuation_templates.len(),
            size_of::<PublicationRange>(),
        )?;
        add(1, size_of::<PublicationControl>())?;
        add(1, size_of::<DeviceState>())?;
        add(
            1,
            size_of::<model_context_continuation::PreparedModelContextContinuationDevice>(),
        )?;
        let mut upload_bytes = 0u64;
        for range in &material.ranges {
            if matches!(range.range.role, 18..=25) {
                continue;
            }
            let copies = if publication_mutable_role(range.range.role) {
                2u64
            } else {
                1
            };
            upload_bytes = upload_bytes
                .checked_add(
                    u64::try_from(range.bytes.len())
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?
                        .checked_mul(copies)
                        .ok_or(SemanticTransitionError::GenerationExhausted)?,
                )
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        for range in &geometry.continuation_templates {
            if range.role == 55 {
                upload_bytes = upload_bytes
                    .checked_add(range.length_bytes)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
            }
        }
        // The original shared pool selects the available staging extent later.
        // One-byte chunks enclose every legal extent without adding model DMA.
        commands = commands
            .checked_add(upload_bytes)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        bytes = bytes
            .checked_add(upload_bytes)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        Ok([commands, 0, 0, 0, 0, 0, 0, 0, bytes])
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn dormant_model_context_restore_native_ceilings(
        &self,
        material: &OwnedPublicationMaterial,
    ) -> Result<([u64; 9], [u64; 9]), SemanticTransitionError> {
        let publication = self.dormant_model_context_restore_dma_ceiling(material)?;
        let (mut dma, mut total) = self
            .graph
            .root_restore_native_work_ceilings(&material.graph)
            .map_err(SemanticTransitionError::Semantic)?;
        native_work_bound::add_native_work_ceiling(&mut dma, publication)?;
        native_work_bound::add_native_work_ceiling(&mut total, publication)?;
        Ok((dma, total))
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn dormant_state_material_restore_recipe(
        &self,
    ) -> Result<
        (
            PublicationBank,
            SemanticRngBinding,
            Vec<u64>,
            [u64; PUBLICATION_ROLE_COUNT],
        ),
        SemanticTransitionError,
    > {
        let owner = self
            .state_material_restore
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let custody = owner
            .dormant_context
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let allowance =
            self.require_dormant_model_context_restore_work(&custody.continuation, &custody.work)?;
        let preparation = self
            .publication_preparation
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let storage = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&owner.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&allowance, &custody.allowance)
            || owner.pending.load(Ordering::Acquire)
            || owner.initialized.load(Ordering::Acquire)
            || owner.context.is_some()
            || storage.instance != owner.instance
            || !Arc::ptr_eq(&preparation.issuer, &self.publication_issuer)
            || preparation.pending.load(Ordering::Acquire)
            || preparation
                .restoration
                .as_ref()
                .is_none_or(|issuance| !Arc::ptr_eq(issuance, &owner.issuance))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original.stage != StateRestoreStage::Completed
            || !original.retain_current_models
            || original.transitioned
            || original.fold.is_some()
            || original.result.is_some()
            || original.material.bank.header.abi != 0
            || original.material.bank.header.publication_word != 0
            || original.material.bank.header.instance != storage.instance
            || self.rng != Some(original.rng)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok((
            original.material.bank,
            original.rng,
            original.material.terminals.clone(),
            original.material.role_counts,
        ))
    }

    fn restore_decoded_state_material_original(
        &mut self,
        material: OwnedPublicationMaterial,
        source_digest: Identity256,
        transition: Option<&SemanticLearningPhaseTransition>,
        immutable_models: Option<RestoredModelOwners<'_>>,
        #[cfg(feature = "semantic-policy")] resident_restoration: Option<
            SemanticPreparedResidentModelRestoration,
        >,
        #[cfg(feature = "semantic-policy")] dormant_context: Option<DormantStateRestoreCustody>,
    ) -> Result<Option<SemanticPublishedIdentity>, SemanticTransitionError> {
        // Keep the outside-array receipt alive through every local material
        // error drop, then transfer it behind the retained material's data.
        let mut phase_metadata: Option<Arc<SemanticReplayMaterialReservation>> = None;
        let mut material = material;
        if self.state_material_restore.is_some() {
            return Err(publication_input_error(
                "original state restoration is already retained; resolve that owner",
            ));
        }
        #[cfg(feature = "semantic-policy")]
        if let Some(original) = &dormant_context {
            let allowance = self.require_dormant_model_context_restore_entry(
                &original.continuation,
                &original.work,
            )?;
            if !Arc::ptr_eq(&allowance, &original.allowance)
                || resident_restoration.is_some()
                || transition.is_some()
                || !immutable_models.is_some_and(RestoredModelOwners::current)
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        } else {
            self.require_state_material_restore_entry()?;
        }
        #[cfg(not(feature = "semantic-policy"))]
        self.require_state_material_restore_entry()?;
        self.require_learning_phase_admission(transition)?;
        #[cfg(feature = "semantic-policy")]
        let resident_prelude = resident_restoration
            .as_ref()
            .map(|original| {
                let prelude = original.native_prelude()?;
                prelude.require_callback_owner(self)?;
                Ok::<_, SemanticTransitionError>(prelude)
            })
            .transpose()?;
        if let Some(budget) = &self.replay_material_budget {
            if !material
                .budget()
                .is_some_and(|material_budget| budget.same_pool(material_budget))
            {
                return Err(publication_input_error(
                    "restoration changes its original capacity pool",
                ));
            }
        }
        if self.publication.is_some() || self.captured.is_some() || self.task.is_none() {
            return Err(publication_input_error(
                "state restoration requires a fresh uncaptured task-bound Session",
            ));
        }
        let retain_current_models = immutable_models.is_some_and(RestoredModelOwners::current);
        if retain_current_models && transition.is_some() {
            return Err(publication_input_error(
                "actor context cannot apply a learning-phase fold",
            ));
        }
        if let Some(transition) = transition {
            let budget = self
                .replay_material_budget
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let bytes = learning_phase::record_allocation_bound(
                &transition.recipe,
                material.learning_phases.len(),
                material.learning_phases.capacity(),
            )?;
            phase_metadata = Some(Arc::new(budget.reserve_auxiliary(bytes)?));
            material
                .learning_phases
                .try_reserve_exact(1)
                .map_err(|_| publication_input_error("learning history allocation failed"))?;
        } else if let Some(budget) = self.replay_material_budget.as_ref() {
            if !material.learning_phases.is_empty() {
                let bytes = material
                    .learning_phases
                    .len()
                    .checked_mul(size_of::<SemanticLearningPhaseRecord>())
                    .and_then(|bytes| {
                        bytes.checked_add(
                            size_of::<SemanticReplayMaterialReservation>() + 2 * size_of::<usize>(),
                        )
                    })
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                phase_metadata = Some(Arc::new(
                    budget.reserve_auxiliary(
                        u64::try_from(bytes)
                            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                    )?,
                ));
            }
        }
        if retain_current_models {
            let coverage = material
                .ranges
                .iter_mut()
                .find(|range| range.range.role == 14)
                .filter(|range| range.bytes.len() == size_of::<CompletionCoverage>())
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            // SAFETY: the canonical material decoder checked this integer-only
            // record, and its exact complete ABI extent was checked above.
            let mut invalidated = unsafe {
                std::ptr::read_unaligned(coverage.bytes.as_ptr().cast::<CompletionCoverage>())
            };
            invalidated.model_generation = 0;
            coverage.bytes = publication_abi_bytes(&[invalidated]).into();
        }
        let ground = self
            .task_ground
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let saved_ground = material
            .ranges
            .iter()
            .find(|range| {
                range.range.role == SemanticStateRole::TaskGround as u64 && range.range.index == 0
            })
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        ground.layout.validate(&saved_ground.bytes)?;
        self.task_observations_authenticated = !saved_ground.bytes[64..ground.layout.query_offset]
            .chunks_exact(144)
            .any(|observation| observation[..8] != [0; 8]);
        let header = material.bank.header;
        let mut rng = header.rng_binding()?;
        let mut fold = None;
        if let Some(transition) = transition {
            let metadata = phase_metadata
                .as_ref()
                .expect("original learning history allocation");
            let (record, absorption) = transition.apply(&mut material, Arc::clone(metadata))?;
            material.learning_phases.push(record);
            fold = absorption;
        }
        if fold.is_some() {
            let generation = header
                .model_generation
                .checked_add(1)
                .filter(|n| *n <= u64::from(u32::MAX))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            material.bank.header.model_generation = generation;
            material.bank.header.neural_generation = header
                .neural_generation
                .checked_add(1)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            material.bank.state.model_generation = generation as u32;
            rng.model_generation = generation as u32;
        }
        if header.authority_generation == 0 {
            return Err(publication_input_error(
                "restored RNG or task generation exceeds its native domain",
            ));
        }
        // Validate immutable task statements before any graph mutation. Symbols
        // are compared through admitted canonical bytes, never process IDs.
        for (index, statement) in self.feedback_statement_bytes()?.iter().enumerate() {
            let saved = material
                .ranges
                .iter()
                .find(|item| item.range.role == 16 && item.range.index == index as u64)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if saved.bytes.as_slice() != *statement {
                return Err(publication_input_error(
                    "restored feedback statement differs from the actual admitted task",
                ));
            }
        }

        let mut instance = [0; 32];
        getrandom::fill(&mut instance)
            .map_err(|error| runtime_error("restored instance entropy", error))?;
        let instance = Identity256::from_bytes(instance);
        if instance == header.instance {
            return Err(publication_input_error(
                "restoration must create a fresh instance",
            ));
        }
        #[cfg(feature = "semantic-policy")]
        let context = self.actor_refresh_context_issuance();
        #[cfg(not(feature = "semantic-policy"))]
        let context = None;
        let issuance = Arc::new(());
        let append_material = if !retain_current_models && self.training_views.is_some() {
            let record = |role| {
                material
                    .ranges
                    .iter()
                    .find(|record| record.range.role == role && record.range.index == 0)
                    .cloned()
                    .ok_or(SemanticTransitionError::ObservationMismatch)
            };
            Some(RestoredReplayAppendMaterial {
                restoration: Arc::clone(&issuance),
                entries: record(56)?,
                payload: record(57)?,
            })
        } else {
            None
        };
        #[cfg(feature = "semantic-policy")]
        let checkpoint_boundary_metadata = self
            .replay_material_budget
            .as_ref()
            .map(|budget| {
                budget.reserve_auxiliary(
                    u64::try_from(
                        size_of::<Option<model_context_checkpoint::ModelContextCheckpointBoundary>>(
                        ) + size_of::<Option<SemanticReplayMaterialReservation>>(),
                    )
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                )
            })
            .transpose()?;
        let owner = Arc::new(OriginalStateMaterialRestore {
            issuer: Arc::clone(&self.publication_issuer),
            issuance,
            source_digest,
            context,
            instance,
            initialized: AtomicBool::new(false),
            pending: AtomicBool::new(true),
            #[cfg(feature = "semantic-policy")]
            resident_restoration: resident_restoration.clone(),
            #[cfg(feature = "semantic-policy")]
            resident_prelude,
            #[cfg(feature = "semantic-policy")]
            dormant_context,
            original: Mutex::new(StateMaterialRestore {
                material,
                header,
                rng,
                fold,
                transitioned: transition.is_some(),
                immutable_models: immutable_models.map(OwnedRestoredModelOwners::retain),
                plans: None,
                model_payloads: Vec::new(),
                retain_current_models,
                instance,
                stage: StateRestoreStage::Graph,
                root: None,
                catalogue_writes: Vec::new(),
                catalogue_next: 0,
                initialize: None,
                control: None,
                initialized_header: None,
                append_material,
                append_writes: None,
                append_next: 0,
                result: None,
                poisoned: false,
                #[cfg(feature = "semantic-policy")]
                checkpoint_boundary: None,
                #[cfg(feature = "semantic-policy")]
                _checkpoint_boundary_metadata: checkpoint_boundary_metadata,
                _phase_metadata: phase_metadata,
            }),
        });
        #[cfg(feature = "semantic-policy")]
        if let Some(original) = &resident_restoration {
            original.retain_native_restore(&owner)?;
        }
        self.state_material_restore = Some(owner);
        self.resolve_state_material_restore()
    }

    pub(super) fn state_material_restore_started(&self) -> bool {
        self.state_material_restore.is_some()
    }

    pub(super) fn state_material_restore_pending(&self) -> bool {
        self.state_material_restore
            .as_ref()
            .is_some_and(|owner| owner.pending.load(Ordering::Acquire))
    }

    pub(super) fn state_material_restore_completion_pending(
        &self,
    ) -> Result<bool, SemanticTransitionError> {
        let Some(owner) = &self.state_material_restore else {
            return Ok(false);
        };
        if !Arc::ptr_eq(&owner.issuer, &self.publication_issuer) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let read_pending = |read: &PublicationRead<_>| read.completion_pending();
        let control_pending = original.control.as_ref().is_some_and(read_pending);
        let header_pending = original
            .initialized_header
            .as_ref()
            .is_some_and(PublicationRead::completion_pending);
        let preparation_pending =
            self.publication_preparation
                .as_ref()
                .is_some_and(|preparation| {
                    Arc::ptr_eq(&preparation.issuer, &self.publication_issuer)
                        && preparation
                            .restoration
                            .as_ref()
                            .is_some_and(|issuance| Arc::ptr_eq(issuance, &owner.issuance))
                        && preparation.completion_pending()
                });
        Ok(original
            .catalogue_writes
            .get(original.catalogue_next)
            .is_some_and(OriginalDeviceWrite::completion_pending)
            || original
                .initialize
                .as_ref()
                .is_some_and(OriginalNativeCommand::completion_pending)
            || control_pending
            || header_pending
            || original.append_writes.as_ref().is_some_and(|writes| {
                writes
                    .get(original.append_next)
                    .is_some_and(OriginalDeviceWrite::completion_pending)
            })
            || preparation_pending
            || self
                .graph
                .root_restore_completion_pending()
                .map_err(SemanticTransitionError::Semantic)?)
    }

    pub(super) fn resolve_state_material_restore(
        &mut self,
    ) -> Result<Option<SemanticPublishedIdentity>, SemanticTransitionError> {
        let owner = Arc::clone(
            self.state_material_restore
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        if !Arc::ptr_eq(&owner.issuer, &self.publication_issuer) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        #[cfg(feature = "semantic-policy")]
        if let Some(original) = &owner.dormant_context {
            let allowance = self.require_dormant_model_context_restore_work(
                &original.continuation,
                &original.work,
            )?;
            if !Arc::ptr_eq(&allowance, &original.allowance) || owner.context.is_some() {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        }
        #[cfg(feature = "semantic-policy")]
        let dma_allowance = owner.dma_allowance()?;
        let mut original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        loop {
            match original.stage {
                StateRestoreStage::Graph => {
                    let may_submit = self.original_restoration_may_submit(Some(&owner), None);
                    if !may_submit && !self.graph.root_restore_pending() {
                        return Err(SemanticTransitionError::Poisoned);
                    }
                    let root = if self.graph.root_restore_pending() {
                        self.graph.resolve_root_restore_with_submission(may_submit)
                    } else {
                        self.graph.restore_root(&original.material.graph)
                    }
                    .map_err(SemanticTransitionError::Semantic)?;
                    let snapshot = self
                        .graph
                        .restored_root_snapshot(root)
                        .map_err(SemanticTransitionError::Semantic)?;
                    original.root = Some(root);
                    self.root = root;
                    self.base_snapshot = snapshot;
                    original.stage = StateRestoreStage::Catalogue;
                }
                StateRestoreStage::Catalogue => {
                    if !self.original_restoration_may_submit(Some(&owner), None) {
                        return Err(SemanticTransitionError::Poisoned);
                    }
                    let admission = self
                        .graph
                        .admission()
                        .ok_or(SemanticTransitionError::NotBound)?;
                    let mut codebooks = ActionCodebooks::derive(
                        admission,
                        self.graph.transition_arena()[1],
                        self.task
                            .as_ref()
                            .and_then(|(task, _)| task.spec.program.editable_program()),
                    )?;
                    if let Some((catalogue, member)) = &self.codebooks.model_catalogue {
                        codebooks.project_model_catalogue(Arc::clone(catalogue), *member)?;
                    }
                    if codebooks.input_cells != self.codebooks.input_cells
                        || codebooks.words.len() != self.device_codebooks.len()
                    {
                        return Err(publication_input_error(
                            "restored action catalogue changes its allocated shape",
                        ));
                    }
                    let (task, device) = self
                        .task
                        .as_mut()
                        .ok_or(SemanticTransitionError::NotBound)?;
                    task.admission_identity = admission.identity();
                    if task.identity() != original.material.contract.task_identity {
                        return Err(publication_input_error(
                            "restored native task differs from its original complete admission",
                        ));
                    }
                    let words = task.words(self.graph.transition_arena()[1]);
                    let writes = vec![
                        OriginalDeviceWrite::new(&self.stream, &words, device.view(), true)?,
                        OriginalDeviceWrite::new_with_owner(
                            &self.stream,
                            &codebooks.words,
                            self.device_codebooks.view(),
                            true,
                            codebooks
                                .model_projection_reservation
                                .as_ref()
                                .map(Arc::clone),
                        )?,
                        OriginalDeviceWrite::new_with_owner(
                            &self.stream,
                            &codebooks.components,
                            self.device_components.view(),
                            true,
                            codebooks
                                .model_projection_reservation
                                .as_ref()
                                .map(Arc::clone),
                        )?,
                    ];
                    let codebook = original
                        .material
                        .ranges
                        .iter_mut()
                        .find(|item| item.range.role == 48)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    let relocated =
                        relocate_publication_codebooks(&codebook.bytes, &codebooks.words)?;
                    codebook.bytes = relocated.into();
                    self.codebooks = codebooks;
                    self.task_epoch = original.header.authority_generation;
                    self.learning_phases = original.material.learning_phases.clone();
                    original.catalogue_writes = writes;
                    original.stage = StateRestoreStage::CatalogueWrites;
                }
                StateRestoreStage::CatalogueWrites => {
                    while original.catalogue_next < original.catalogue_writes.len() {
                        let may_submit = self.original_restoration_may_submit(Some(&owner), None);
                        let index = original.catalogue_next;
                        let StateMaterialRestore {
                            catalogue_writes,
                            poisoned,
                            ..
                        } = &mut *original;
                        #[cfg(feature = "semantic-policy")]
                        catalogue_writes[index].resolve_with_dma(
                            &self.domain,
                            &self.stream,
                            &self.provider,
                            poisoned,
                            may_submit,
                            dma_allowance,
                        )?;
                        #[cfg(not(feature = "semantic-policy"))]
                        catalogue_writes[index].resolve(
                            &self.domain,
                            &self.stream,
                            &self.provider,
                            poisoned,
                            may_submit,
                        )?;
                        original.catalogue_next += 1;
                    }
                    original.catalogue_writes.clear();
                    original.stage = StateRestoreStage::Allocate;
                }
                StateRestoreStage::Allocate => {
                    if !self.original_restoration_may_submit(Some(&owner), None) {
                        return Err(SemanticTransitionError::Poisoned);
                    }
                    let mut contract = original.material.contract;
                    contract.semantic_owner = self.graph.transition_arena()[1];
                    #[cfg(feature = "semantic-policy")]
                    if let Some(OwnedRestoredModelOwners::Current {
                        minimum_generation, ..
                    }) = original.immutable_models.as_ref()
                    {
                        if *minimum_generation == 0 || *minimum_generation > u64::from(u32::MAX) {
                            return Err(SemanticTransitionError::ObservationMismatch);
                        }
                        contract.model_generation = *minimum_generation;
                    }
                    if original.plans.is_none() {
                        original.plans = Some(
                            std::mem::take(&mut original.material.ranges)
                                .into_iter()
                                .map(|item| PublicationAllocationPlan {
                                    role: item.range.role,
                                    index: item.range.index,
                                    capacity: item.capacity,
                                    length: item.bytes.len(),
                                    logical_begin: item.range.logical_begin,
                                    logical_end: item.range.logical_end,
                                    payload: if matches!(item.range.role, 18..=25) {
                                        PublicationPayload::Uncomputed
                                    } else {
                                        PublicationPayload::Metadata(item.bytes)
                                    },
                                })
                                .collect::<Vec<_>>(),
                        );
                        original.model_payloads =
                            std::mem::take(&mut original.material.model_allocations)
                                .into_iter()
                                .map(PublicationPayload::Metadata)
                                .collect();
                    }
                    let (storage, uploads, model_owners) = PublicationStorage::allocate(
                        &self.provider,
                        original
                            .plans
                            .as_ref()
                            .expect("original restored allocation plans"),
                        original.material.layouts.clone(),
                        original.material.model_memory.clone(),
                        &original.model_payloads,
                        contract,
                        &original.material.terminals,
                        original.instance,
                        original
                            .immutable_models
                            .as_ref()
                            .map(OwnedRestoredModelOwners::borrowed),
                        self.replay_material_budget.as_ref(),
                    )?;
                    let instance = original.instance;
                    let recovered_instance = original.header.instance;
                    let root = original.root.expect("original restored root");
                    let bank = &mut original.material.bank;
                    bank.header.abi = 0;
                    bank.header.instance = instance;
                    bank.header.recovered_instance = recovered_instance;
                    bank.header.sealed_epoch = 0;
                    bank.header.base_word = 0;
                    bank.header.publication_word = 0;
                    bank.header.semantic_owner = contract.semantic_owner;
                    bank.header.semantic_slot = u64::from(root.slot());
                    bank.header.semantic_generation = root.generation();
                    bank.header.neural_bank = 0;
                    self.publication = Some(Arc::new(storage));
                    self.publication_uploads = uploads;
                    self.model_owners = model_owners;
                    original.immutable_models = None;
                    original.plans = None;
                    original.model_payloads.clear();
                    original.stage = StateRestoreStage::Bind;
                }
                StateRestoreStage::Bind => {
                    if !self.original_restoration_may_submit(Some(&owner), None) {
                        return Err(SemanticTransitionError::Poisoned);
                    }
                    self.bind_training_publication()?;
                    original.stage = StateRestoreStage::Prepare;
                }
                StateRestoreStage::Prepare => {
                    if self.publication_preparation.is_none() {
                        if !self.original_restoration_may_submit(Some(&owner), None) {
                            return Err(SemanticTransitionError::Poisoned);
                        }
                        self.publication_preparation = Some(self.stage_publication_preparation(
                            original.material.bank,
                            &original.material.terminals,
                            &original.material.role_counts,
                            original.rng,
                            original.fold.as_ref(),
                            original.retain_current_models,
                            Some(&owner),
                            None,
                        )?);
                    }
                    self.resolve_publication_preparation(Some(&owner))?;
                    if original.retain_current_models {
                        self.publication_uploads.clear();
                        original.plans = None;
                        original.model_payloads = Vec::new();
                        self.rng = Some(original.rng);
                        self.next_proposal = u64::from(original.rng.proposal);
                        original.stage = StateRestoreStage::Completed;
                        owner.pending.store(false, Ordering::Release);
                        return Ok(None);
                    }
                    original.stage = StateRestoreStage::Initialize;
                }
                StateRestoreStage::Initialize => {
                    let storage = Arc::clone(
                        self.publication
                            .as_ref()
                            .ok_or(SemanticTransitionError::NotBound)?,
                    );
                    if original.initialize.is_none() {
                        if !self.original_restoration_may_submit(
                            Some(&owner),
                            self.publication_preparation.as_ref(),
                        ) {
                            return Err(SemanticTransitionError::Poisoned);
                        }
                        // Every original completion owner is retained before
                        // the first initialization command can enter the driver.
                        let command = OriginalNativeCommand::new(&self.domain)?;
                        let control = self.stage_publication_read(storage.control.view())?;
                        let header_view = unsafe { storage.banks[0].view().cast::<u8>() }
                            .ok_or(SemanticTransitionError::ObservationMismatch)?
                            .slice(..size_of::<PublicationHeader>());
                        // SAFETY: the original bank starts with this integer-only
                        // header and has the checked complete ABI extent.
                        let header_view = unsafe { header_view.cast::<PublicationHeader>() }
                            .ok_or(SemanticTransitionError::ObservationMismatch)?;
                        let header = self.stage_publication_read(header_view)?;
                        original.initialize = Some(command);
                        original.control = Some(control);
                        original.initialized_header = Some(header);
                    }
                    let operation = if original.fold.is_some() {
                        7
                    } else if original.transitioned {
                        5
                    } else {
                        4
                    };
                    let may_submit = self.original_restoration_may_submit(
                        Some(&owner),
                        self.publication_preparation.as_ref(),
                    );
                    let StateMaterialRestore {
                        initialize,
                        poisoned,
                        ..
                    } = &mut *original;
                    let command = initialize
                        .as_mut()
                        .expect("original restore initialization");
                    if !command.entered && !may_submit {
                        return Err(SemanticTransitionError::Poisoned);
                    }
                    self.publication_command_with_original(
                        operation,
                        None,
                        Some(command),
                        poisoned,
                    )?;
                    let may_submit = self.original_restoration_may_submit(
                        Some(&owner),
                        self.publication_preparation.as_ref(),
                    );
                    let StateMaterialRestore {
                        control, poisoned, ..
                    } = &mut *original;
                    let control = control.as_mut().expect("original restore control read");
                    if !control.read.entered() && !may_submit {
                        return Err(SemanticTransitionError::Poisoned);
                    }
                    #[cfg(feature = "semantic-policy")]
                    let control =
                        self.resolve_restoration_read_with_dma(control, poisoned, dma_allowance)?
                            [0];
                    #[cfg(not(feature = "semantic-policy"))]
                    let control = self.resolve_publication_read_with_poison(control, poisoned)?[0];
                    if control.refusal != 0 {
                        return Err(SemanticTransitionError::PublicationRefused {
                            status: control.refusal,
                        });
                    }
                    let may_submit = self.original_restoration_may_submit(
                        Some(&owner),
                        self.publication_preparation.as_ref(),
                    );
                    let StateMaterialRestore {
                        initialized_header,
                        poisoned,
                        ..
                    } = &mut *original;
                    let header = initialized_header
                        .as_mut()
                        .expect("original restored header read");
                    if !header.read.entered() && !may_submit {
                        return Err(SemanticTransitionError::Poisoned);
                    }
                    #[cfg(feature = "semantic-policy")]
                    let header =
                        self.resolve_restoration_read_with_dma(header, poisoned, dma_allowance)?[0];
                    #[cfg(not(feature = "semantic-policy"))]
                    let header = self.resolve_publication_read_with_poison(header, poisoned)?[0];
                    let bank = original.material.bank;
                    if header.abi != 1
                        || header.instance != storage.instance
                        || header.publication_word != 0
                        || control.word != 0
                        || header.model_geometry_digest != bank.header.model_geometry_digest
                        || (operation == 4
                            && header.model_numerical_digest != bank.header.model_numerical_digest)
                    {
                        return Err(SemanticTransitionError::ObservationMismatch);
                    }
                    self.publication_uploads.clear();
                    self.rng = Some(original.rng);
                    self.next_proposal = u64::from(original.rng.proposal);
                    original.result = Some(SemanticPublishedIdentity {
                        instance: header.instance,
                        word: header.publication_word,
                        logical_digest: header.logical_digest,
                        state_digest: header.state_digest,
                    });
                    if !original.transitioned
                        && original.result.is_some_and(|restored| {
                            restored.logical_digest != original.header.logical_digest
                                || restored.state_digest != original.header.state_digest
                        })
                    {
                        return Err(SemanticTransitionError::ObservationMismatch);
                    }
                    owner.initialized.store(true, Ordering::Release);
                    original.initialize = None;
                    original.control = None;
                    original.initialized_header = None;
                    original.stage = StateRestoreStage::AppendRows;
                }
                StateRestoreStage::AppendRows => {
                    if let Some(material) = original.append_material.as_ref() {
                        if original.append_writes.is_none() {
                            if !self.original_restoration_may_submit(
                                Some(&owner),
                                self.publication_preparation.as_ref(),
                            ) {
                                return Err(SemanticTransitionError::Poisoned);
                            }
                            original.append_writes =
                                Some(self.stage_restored_replay_append_rows(&owner, material)?);
                        }
                    }
                    while original
                        .append_writes
                        .as_ref()
                        .is_some_and(|writes| original.append_next < writes.len())
                    {
                        let may_submit = self.original_restoration_may_submit(
                            Some(&owner),
                            self.publication_preparation.as_ref(),
                        );
                        let index = original.append_next;
                        let StateMaterialRestore {
                            append_writes,
                            poisoned,
                            ..
                        } = &mut *original;
                        #[cfg(feature = "semantic-policy")]
                        append_writes
                            .as_mut()
                            .expect("original append write roster")[index]
                            .resolve_with_dma(
                                &self.domain,
                                &self.stream,
                                &self.provider,
                                poisoned,
                                may_submit,
                                dma_allowance,
                            )?;
                        #[cfg(not(feature = "semantic-policy"))]
                        append_writes
                            .as_mut()
                            .expect("original append write roster")[index]
                            .resolve(
                                &self.domain,
                                &self.stream,
                                &self.provider,
                                poisoned,
                                may_submit,
                            )?;
                        original.append_next += 1;
                    }
                    original.append_writes = None;
                    original.append_material = None;
                    original.stage = StateRestoreStage::Seal;
                }
                StateRestoreStage::Seal => {
                    let restored = original
                        .result
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    if !original.transitioned
                        && (restored.logical_digest != original.header.logical_digest
                            || restored.state_digest != original.header.state_digest)
                    {
                        return Err(SemanticTransitionError::ObservationMismatch);
                    }
                    for allocation in &self
                        .publication
                        .as_ref()
                        .ok_or(SemanticTransitionError::NotBound)?
                        .allocations
                    {
                        allocation.seal_restoration();
                    }
                    original.stage = StateRestoreStage::Completed;
                    original.poisoned = false;
                    // Every original upload, initialization, append and seal
                    // is known complete. Retain the validated metadata/result,
                    // not the resolved source allocation's byte-view aliases.
                    original.plans = None;
                    original.model_payloads = Vec::new();
                    owner.pending.store(false, Ordering::Release);
                }
                StateRestoreStage::Completed => return Ok(original.result),
            }
        }
    }
}

trait OriginalTypedWrite: Send + Sync {
    fn entered(&self) -> bool;
    fn submitted(&self) -> bool;
    fn completion_pending(&self) -> bool;
    fn known_completion_failure(&self) -> Option<sys::CUresult>;
    fn enqueue(
        &mut self,
        domain: &ResidentExecutionDomain,
        stream: &Arc<CudaStream>,
        poisoned: &mut bool,
    ) -> Result<(), SemanticTransitionError>;
    fn resolve(&mut self, stream: &Arc<CudaStream>) -> Result<(), SemanticTransitionError>;
    fn retire_without_result(
        &mut self,
        stream: &Arc<CudaStream>,
    ) -> Result<(), SemanticTransitionError>;
}

struct TypedOriginalWrite<T: DeviceRepr> {
    destination: DeviceMemoryView<T>,
    write: crate::device::RetainedDeviceWrite<T>,
}

impl<T: DeviceRepr + Send + Sync> OriginalTypedWrite for TypedOriginalWrite<T> {
    fn completion_pending(&self) -> bool {
        self.write.completion_pending()
    }
    fn known_completion_failure(&self) -> Option<sys::CUresult> {
        self.write.known_completion_failure()
    }
    fn entered(&self) -> bool {
        self.write.entered()
    }

    fn submitted(&self) -> bool {
        self.write.submitted()
    }

    fn enqueue(
        &mut self,
        domain: &ResidentExecutionDomain,
        stream: &Arc<CudaStream>,
        poisoned: &mut bool,
    ) -> Result<(), SemanticTransitionError> {
        let mut recorder = domain.new_strict_recorder();
        recorder.write(&self.destination);
        let _ = stream;
        enqueue_recorded(domain, poisoned, recorder, |enqueue| {
            self.write.enqueue_admitted(enqueue)
        })
    }

    fn resolve(&mut self, stream: &Arc<CudaStream>) -> Result<(), SemanticTransitionError> {
        self.write
            .resolve_on(stream)
            .map_err(|error| runtime_error("original device write completion", error))
    }

    fn retire_without_result(
        &mut self,
        stream: &Arc<CudaStream>,
    ) -> Result<(), SemanticTransitionError> {
        self.write
            .retire_without_result(stream)
            .map_err(|error| runtime_error("original device write retirement", error))
    }
}

/// A heterogeneous ledger entry preserving the original typed host staging,
/// destination and admitted transfer classification.
pub(crate) struct OriginalDeviceWrite {
    transfer: Box<dyn OriginalTypedWrite>,
    bytes: usize,
    tracked: bool,
    admitted: bool,
    completed: bool,
    completion_pending: bool,
    retired: bool,
    host_reservation: Option<Arc<SemanticReplayMaterialReservation>>,
    #[cfg(feature = "semantic-policy")]
    dma_recorded: bool,
    #[cfg(feature = "semantic-policy")]
    dma_allowance: Option<std::sync::Weak<Mutex<ColdNativeAllowance>>>,
}

impl OriginalDeviceWrite {
    pub(super) fn dma_bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn metadata_allocation_bytes<T: DeviceRepr + Copy + Send + Sync + 'static>() -> usize
    {
        size_of::<Self>() + size_of::<TypedOriginalWrite<T>>()
    }

    pub(super) fn new<T: DeviceRepr + Copy + Send + Sync + 'static>(
        stream: &CudaStream,
        values: &[T],
        destination: DeviceMemoryView<T>,
        tracked: bool,
    ) -> Result<Self, SemanticTransitionError> {
        Self::new_with_owner(stream, values, destination, tracked, None)
    }

    pub(crate) fn new_with_owner<T: DeviceRepr + Copy + Send + Sync + 'static>(
        stream: &CudaStream,
        values: &[T],
        destination: DeviceMemoryView<T>,
        tracked: bool,
        host_reservation: Option<Arc<SemanticReplayMaterialReservation>>,
    ) -> Result<Self, SemanticTransitionError> {
        let bytes = values
            .len()
            .checked_mul(size_of::<T>())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let owner = host_reservation
            .as_ref()
            .map(|owner| Arc::clone(owner) as Arc<dyn Send + Sync>);
        let write = crate::device::RetainedDeviceWrite::new_with_owner(
            stream,
            values,
            destination.clone(),
            owner,
        )
        .map_err(|error| runtime_error("original device write staging", error))?;
        Ok(Self {
            transfer: Box::new(TypedOriginalWrite { destination, write }),
            bytes,
            tracked,
            admitted: false,
            completed: bytes == 0,
            completion_pending: false,
            retired: false,
            host_reservation,
            #[cfg(feature = "semantic-policy")]
            dma_recorded: false,
            #[cfg(feature = "semantic-policy")]
            dma_allowance: None,
        })
    }

    pub(super) fn entered(&self) -> bool {
        self.transfer.entered()
    }

    pub(super) fn completed(&self) -> bool {
        self.completed && !self.retired
    }

    pub(crate) fn pending(&self) -> bool {
        self.entered() && !self.completed && !self.retired
    }

    /// Result resolution is possible only for a confirmed original submission
    /// whose completion remains unknown. Lifetime custody is broader: an
    /// entered but unconfirmed write still must be retained/retired, not retried.
    pub(crate) fn completion_pending(&self) -> bool {
        self.completion_pending
            && self.pending()
            && self.transfer.completion_pending()
            && self.known_completion_failure().is_none()
    }

    pub(crate) fn known_completion_failure(&self) -> Option<sys::CUresult> {
        self.transfer.known_completion_failure()
    }

    pub(crate) fn resolve(
        &mut self,
        domain: &ResidentExecutionDomain,
        stream: &Arc<CudaStream>,
        provider: &CudaKernelProvider,
        poisoned: &mut bool,
        may_submit: bool,
    ) -> Result<(), SemanticTransitionError> {
        self.resolve_original(
            domain,
            stream,
            provider,
            poisoned,
            may_submit,
            #[cfg(feature = "semantic-policy")]
            None,
        )
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn resolve_with_dma(
        &mut self,
        domain: &ResidentExecutionDomain,
        stream: &Arc<CudaStream>,
        provider: &CudaKernelProvider,
        poisoned: &mut bool,
        may_submit: bool,
        allowance: Option<&Arc<Mutex<ColdNativeAllowance>>>,
    ) -> Result<(), SemanticTransitionError> {
        self.completion_pending = false;
        if self.dma_allowance.as_ref().is_some_and(|original| {
            allowance.is_none_or(|allowance| !original.ptr_eq(&Arc::downgrade(allowance)))
        }) {
            return Err(publication_input_error(
                "original restored upload changed its native report owner",
            ));
        }
        if self.dma_allowance.is_none() && !self.entered() {
            self.dma_allowance = allowance.map(Arc::downgrade);
        }
        self.resolve_original(domain, stream, provider, poisoned, may_submit, allowance)
    }

    pub(super) fn resolve_original(
        &mut self,
        domain: &ResidentExecutionDomain,
        stream: &Arc<CudaStream>,
        provider: &CudaKernelProvider,
        poisoned: &mut bool,
        may_submit: bool,
        #[cfg(feature = "semantic-policy")] allowance: Option<&Arc<Mutex<ColdNativeAllowance>>>,
    ) -> Result<(), SemanticTransitionError> {
        self.completion_pending = false;
        if self.retired {
            return Err(publication_input_error("original device write has retired"));
        }
        if self.completed {
            return Ok(());
        }
        if !self.entered() {
            if !may_submit {
                return Err(SemanticTransitionError::Poisoned);
            }
            #[cfg(feature = "semantic-policy")]
            if let Some(allowance) = allowance {
                allowance
                    .lock()
                    .map_err(|_| publication_input_error("original cold allowance is poisoned"))?
                    .require_submitted_dma(self.bytes, self.dma_recorded)?;
            }
            if !self.admitted {
                if self.tracked {
                    provider.admit_tracked_htod(self.bytes);
                } else {
                    provider.admit_launch_metadata_htod(self.bytes);
                }
                self.admitted = true;
            }
            let result = self.transfer.enqueue(domain, stream, poisoned);
            #[cfg(feature = "semantic-policy")]
            self.record_submitted_dma(allowance)?;
            if let Err(error) = result {
                self.completion_pending = self.transfer.submitted();
                return Err(error);
            }
        }
        #[cfg(feature = "semantic-policy")]
        self.record_submitted_dma(allowance)?;
        self.transfer.resolve(stream).map_err(|error| {
            *poisoned |= self.entered();
            self.completion_pending = self.transfer.submitted();
            error
        })?;
        self.completed = true;
        self.host_reservation = None;
        Ok(())
    }

    #[cfg(feature = "semantic-policy")]
    fn record_submitted_dma(
        &mut self,
        allowance: Option<&Arc<Mutex<ColdNativeAllowance>>>,
    ) -> Result<(), SemanticTransitionError> {
        if self.transfer.submitted() && !self.dma_recorded {
            if let Some(allowance) = allowance {
                allowance
                    .lock()
                    .map_err(|_| publication_input_error("original cold allowance is poisoned"))?
                    .record_submitted_dma(self.bytes, &mut self.dma_recorded)?;
            } else {
                self.dma_recorded = true;
            }
        }
        Ok(())
    }

    pub(super) fn retire_without_result(
        &mut self,
        stream: &Arc<CudaStream>,
    ) -> Result<(), SemanticTransitionError> {
        if self.retired {
            return Ok(());
        }
        if self.entered() && !self.completed {
            self.transfer.retire_without_result(stream)?;
        }
        self.retired = true;
        self.completion_pending = false;
        self.host_reservation = None;
        Ok(())
    }
}

pub(super) struct OriginalPublicationPreparation {
    issuer: Arc<()>,
    restoration: Option<Arc<()>>,
    pending: AtomicBool,
    original: Mutex<PublicationPreparation>,
}

impl OriginalPublicationPreparation {
    pub(super) fn completion_pending(&self) -> bool {
        self.original.lock().is_ok_and(|original| {
            original
                .memory
                .get(original.memory_next)
                .is_some_and(crate::device::RetainedDeviceMemoryCommand::completion_pending)
                || original
                    .upload_write
                    .as_ref()
                    .is_some_and(OriginalDeviceWrite::completion_pending)
                || original
                    .writes
                    .get(original.write_next)
                    .is_some_and(OriginalDeviceWrite::completion_pending)
        })
    }

    pub(super) fn require_issuer(
        &self,
        target: &SemanticTransitionSession,
    ) -> Result<(), SemanticTransitionError> {
        if !Arc::ptr_eq(&self.issuer, &target.publication_issuer) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }
}

struct PublicationPreparation {
    storage: Option<Arc<PublicationStorage>>,
    bank: Vec<u8>,
    terminal_tokens: Vec<u64>,
    role_counts: [u64; PUBLICATION_ROLE_COUNT],
    rng: SemanticRngBinding,
    retain_current_models: bool,
    fold: Option<learning_phase::LearningFoldPlan>,
    fold_work: Option<learning_phase::OriginalLearningFold>,
    fold_completed: bool,
    memory: Vec<crate::device::RetainedDeviceMemoryCommand>,
    memory_next: usize,
    memory_dma_recorded: bool,
    #[cfg(feature = "semantic-policy")]
    context_dma_allowance: Option<Arc<Mutex<ColdNativeAllowance>>>,
    uploads: Vec<PublicationUploadPlan>,
    upload_next: usize,
    upload_offset: usize,
    upload_end: usize,
    upload_write: Option<OriginalDeviceWrite>,
    upload_staging: Option<Arc<SemanticReplayMaterialReservation>>,
    upload_staging_bytes: usize,
    host_reservation: Option<Arc<SemanticReplayMaterialReservation>>,
    writes: Vec<OriginalDeviceWrite>,
    write_next: usize,
    fold_after: usize,
    poisoned: bool,
}

pub(super) struct PublicationUploadPlan {
    pub(super) source: PublicationMaterialBytes,
    pub(super) destination: DeviceMemoryView<u8>,
}

enum OwnedRestoredModelOwners {
    Historical(Vec<Arc<ReplayModelBacking>>),
    #[cfg(feature = "semantic-policy")]
    Continuation(Vec<Arc<ModelGenerationOwner>>),
    #[cfg(feature = "semantic-policy")]
    Current {
        owners: Vec<Arc<ModelGenerationOwner>>,
        minimum_generation: u64,
    },
}

impl OwnedRestoredModelOwners {
    fn retain(owners: RestoredModelOwners<'_>) -> Self {
        match owners {
            RestoredModelOwners::Historical(owners) => Self::Historical(owners.to_vec()),
            #[cfg(feature = "semantic-policy")]
            RestoredModelOwners::Continuation(owners) => Self::Continuation(owners.to_vec()),
            #[cfg(feature = "semantic-policy")]
            RestoredModelOwners::Current {
                owners,
                minimum_generation,
            } => Self::Current {
                owners: owners.to_vec(),
                minimum_generation,
            },
        }
    }

    fn borrowed(&self) -> RestoredModelOwners<'_> {
        match self {
            Self::Historical(owners) => RestoredModelOwners::Historical(owners),
            #[cfg(feature = "semantic-policy")]
            Self::Continuation(owners) => RestoredModelOwners::Continuation(owners),
            #[cfg(feature = "semantic-policy")]
            Self::Current {
                owners,
                minimum_generation,
            } => RestoredModelOwners::Current {
                owners,
                minimum_generation: *minimum_generation,
            },
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StateRestoreStage {
    Graph,
    Catalogue,
    CatalogueWrites,
    Allocate,
    Bind,
    Prepare,
    Initialize,
    AppendRows,
    Seal,
    Completed,
}

#[cfg(feature = "semantic-policy")]
struct DormantStateRestoreCustody {
    continuation: SemanticModelContextContinuation,
    work: SemanticColdModelWork,
    allowance: Arc<Mutex<ColdNativeAllowance>>,
}

pub(super) struct OriginalStateMaterialRestore {
    issuer: Arc<()>,
    issuance: Arc<()>,
    source_digest: Identity256,
    context: Option<Arc<()>>,
    instance: Identity256,
    initialized: AtomicBool,
    pending: AtomicBool,
    #[cfg(feature = "semantic-policy")]
    resident_restoration: Option<SemanticPreparedResidentModelRestoration>,
    #[cfg(feature = "semantic-policy")]
    resident_prelude: Option<Arc<SemanticColdResidentRestorationPrelude>>,
    #[cfg(feature = "semantic-policy")]
    dormant_context: Option<DormantStateRestoreCustody>,
    original: Mutex<StateMaterialRestore>,
}

#[cfg(feature = "semantic-policy")]
impl OriginalStateMaterialRestore {
    pub(super) fn dma_allowance(
        &self,
    ) -> Result<Option<&Arc<Mutex<native_work_bound::ColdNativeAllowance>>>, SemanticTransitionError>
    {
        if let Some(original) = &self.dormant_context {
            if self.resident_restoration.is_some()
                || self.resident_prelude.is_some()
                || self.context.is_some()
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            return Ok(Some(&original.allowance));
        }
        match (&self.resident_restoration, &self.resident_prelude) {
            (Some(issuer), Some(prelude)) => {
                prelude.require_restoration(issuer)?;
                Ok(Some(prelude.allowance()))
            }
            (None, None) => Ok(None),
            _ => Err(publication_input_error(
                "native restoration changed its original admitted DMA owner",
            )),
        }
    }

    pub(super) fn require_resident_read(
        self: &Arc<Self>,
        original: &SemanticPreparedResidentModelRestoration,
        identity: SemanticPublishedIdentity,
    ) -> Result<(), SemanticTransitionError> {
        if self.pending.load(Ordering::Acquire)
            || !self.initialized.load(Ordering::Acquire)
            || self
                .resident_restoration
                .as_ref()
                .is_none_or(|retained| !Arc::ptr_eq(&retained.owner, &original.owner))
        {
            return Err(publication_input_error(
                "resident read requires its original completed native restore",
            ));
        }
        original.require_native_restore(self)?;
        let restore = self
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if restore.stage != StateRestoreStage::Completed
            || restore.retain_current_models
            || restore.result != Some(identity)
        {
            return Err(publication_input_error(
                "resident read replaced its original restored publication",
            ));
        }
        Ok(())
    }
}

pub(super) struct RestoredReplayAppendMaterial {
    restoration: Arc<()>,
    entries: OwnedPublicationMaterialRange,
    payload: OwnedPublicationMaterialRange,
}

impl RestoredReplayAppendMaterial {
    pub(super) fn records(
        &self,
    ) -> (
        &OwnedPublicationMaterialRange,
        &OwnedPublicationMaterialRange,
    ) {
        (&self.entries, &self.payload)
    }
}

struct StateMaterialRestore {
    material: OwnedPublicationMaterial,
    header: PublicationHeader,
    rng: SemanticRngBinding,
    fold: Option<learning_phase::LearningFoldPlan>,
    transitioned: bool,
    immutable_models: Option<OwnedRestoredModelOwners>,
    plans: Option<Vec<PublicationAllocationPlan>>,
    model_payloads: Vec<PublicationPayload>,
    retain_current_models: bool,
    instance: Identity256,
    stage: StateRestoreStage,
    root: Option<SemanticRootHandle>,
    catalogue_writes: Vec<OriginalDeviceWrite>,
    catalogue_next: usize,
    initialize: Option<OriginalNativeCommand>,
    control: Option<PublicationRead<PublicationControl>>,
    initialized_header: Option<PublicationRead<PublicationHeader>>,
    append_material: Option<RestoredReplayAppendMaterial>,
    append_writes: Option<Vec<OriginalDeviceWrite>>,
    append_next: usize,
    result: Option<SemanticPublishedIdentity>,
    poisoned: bool,
    #[cfg(feature = "semantic-policy")]
    checkpoint_boundary: Option<model_context_checkpoint::ModelContextCheckpointBoundary>,
    #[cfg(feature = "semantic-policy")]
    _checkpoint_boundary_metadata: Option<SemanticReplayMaterialReservation>,
    // Data fields, including the grown history array, drop before this receipt.
    _phase_metadata: Option<Arc<SemanticReplayMaterialReservation>>,
}

impl SemanticTransitionSession {
    #[cfg(feature = "semantic-policy")]
    pub(super) fn resolve_restoration_read_with_dma<T: DeviceRepr + Copy>(
        &mut self,
        original: &mut PublicationRead<T>,
        poisoned: &mut bool,
        allowance: Option<&Arc<Mutex<native_work_bound::ColdNativeAllowance>>>,
    ) -> Result<Vec<T>, SemanticTransitionError> {
        if !Arc::ptr_eq(&original.issuer, &self.publication_issuer) {
            return Err(publication_input_error(
                "native restoration read changed its original Session",
            ));
        }
        match (&original.dma_allowance, allowance) {
            (Some(retained), Some(allowance)) if retained.ptr_eq(&Arc::downgrade(allowance)) => {}
            (None, Some(allowance)) if !original.read.entered() => {
                original.dma_allowance = Some(Arc::downgrade(allowance));
            }
            (None, None) => {}
            _ => {
                return Err(publication_input_error(
                    "native restoration read changed its original DMA allowance",
                ));
            }
        }
        self.resolve_publication_read_with_poison(original, poisoned)
    }

    pub(super) fn stage_publication_preparation(
        &self,
        bank: PublicationBank,
        terminal_tokens: &[u64],
        role_counts: &[u64; PUBLICATION_ROLE_COUNT],
        rng: SemanticRngBinding,
        fold: Option<&learning_phase::LearningFoldPlan>,
        retain_current_models: bool,
        restoration: Option<&Arc<OriginalStateMaterialRestore>>,
        extra_write: Option<OriginalDeviceWrite>,
    ) -> Result<Arc<OriginalPublicationPreparation>, SemanticTransitionError> {
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let upload_count = self
            .publication_uploads
            .iter()
            .filter(|(_, payload)| {
                matches!(payload, PublicationPayload::Metadata(bytes) if !bytes.is_empty())
            })
            .count();
        let write_count = 10usize
            .checked_add(usize::from(extra_write.is_some()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?
            .checked_add(usize::from(!terminal_tokens.is_empty()))
            .and_then(|count| {
                count.checked_add(usize::from(!storage.continuation_templates.is_empty()))
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut host_bytes = 0usize;
        let mut add = |count: usize, width: usize| -> Result<(), SemanticTransitionError> {
            host_bytes = host_bytes
                .checked_add(
                    count
                        .checked_mul(width)
                        .ok_or(SemanticTransitionError::GenerationExhausted)?,
                )
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            Ok(())
        };
        add(upload_count, size_of::<PublicationUploadPlan>())?;
        add(write_count, size_of::<OriginalDeviceWrite>())?;
        add(1, size_of::<TypedOriginalWrite<u8>>())?;
        add(2, size_of::<TypedOriginalWrite<PublicationBank>>())?;
        add(2, size_of::<TypedOriginalWrite<PublicationRange>>())?;
        add(1, size_of::<TypedOriginalWrite<PublicationStorageEntry>>())?;
        add(1, size_of::<TypedOriginalWrite<PublicationContract>>())?;
        add(
            usize::from(!terminal_tokens.is_empty()),
            size_of::<TypedOriginalWrite<u64>>(),
        )?;
        add(1, size_of::<TypedOriginalWrite<PublicationRoleCount>>())?;
        add(1, size_of::<TypedOriginalWrite<PendingContinuation>>())?;
        add(
            usize::from(!storage.continuation_templates.is_empty()),
            size_of::<TypedOriginalWrite<PublicationRange>>(),
        )?;
        add(1, size_of::<TypedOriginalWrite<PublicationControl>>())?;
        add(1, size_of::<TypedOriginalWrite<DeviceState>>())?;
        add(1, size_of::<OriginalPublicationPreparation>())?;
        add(1, 2 * size_of::<usize>())?;
        add(
            2,
            size_of::<SemanticReplayMaterialReservation>() + 2 * size_of::<usize>(),
        )?;
        // Retained metadata clones and the original small typed upload pins.
        add(3, size_of::<PublicationBank>())?;
        add(2, std::mem::size_of_val(terminal_tokens))?;
        add(
            storage.allocations.len(),
            2 * size_of::<PublicationStorageEntry>(),
        )?;
        add(role_counts.len(), 2 * size_of::<PublicationRoleCount>())?;
        for directory in &storage.bank_templates {
            add(directory.len(), size_of::<PublicationRange>())?;
        }
        add(1, size_of::<PublicationContract>())?;
        add(1, size_of::<PendingContinuation>())?;
        add(
            storage.continuation_templates.len(),
            size_of::<PublicationRange>(),
        )?;
        add(1, size_of::<PublicationControl>())?;
        add(1, size_of::<DeviceState>())?;
        let host_reservation = self
            .replay_material_budget
            .as_ref()
            .map(|budget| {
                budget
                    .reserve_auxiliary(
                        u64::try_from(host_bytes)
                            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                    )
                    .map(Arc::new)
            })
            .transpose()?;
        let mut uploads = Vec::with_capacity(upload_count);
        let mut maximum_upload = 0usize;
        for (slot, payload) in &self.publication_uploads {
            if let PublicationPayload::Metadata(bytes) = payload {
                if bytes.is_empty() {
                    continue;
                }
                if self.replay_material_budget.as_ref().is_some_and(|budget| {
                    bytes
                        .budget()
                        .is_some_and(|original| !original.same_pool(budget))
                }) {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                let destination = storage.allocations[*slot]
                    .slice()?
                    .view()
                    .try_slice(..bytes.len())
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                maximum_upload = maximum_upload.max(bytes.len());
                uploads.push(PublicationUploadPlan {
                    source: bytes.clone(),
                    destination,
                });
            }
        }
        let upload_staging = if maximum_upload == 0 {
            None
        } else {
            self.replay_material_budget
                .as_ref()
                .map(|budget| {
                    budget
                        .reserve_upload_staging(
                            u64::try_from(maximum_upload)
                                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                        )
                        .map(Arc::new)
                })
                .transpose()?
        };
        let upload_staging_bytes = upload_staging
            .as_ref()
            .map(|owner| usize::try_from(owner.payload_bytes))
            .transpose()
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?
            .unwrap_or(maximum_upload);
        let mut memory = Vec::new();
        for allocation in &storage.allocations {
            if allocation.is_empty()
                || !allocation.initializing()
                || (retain_current_models && allocation.immutable())
            {
                continue;
            }
            memory.push(
                crate::device::RetainedDeviceMemoryCommand::zero(
                    &self.stream,
                    allocation.slice()?.view(),
                )
                .map_err(|error| runtime_error("publication zero staging", error))?,
            );
        }
        for (slot, payload) in &self.publication_uploads {
            if let PublicationPayload::Tensor(tensor) = payload {
                if let Some(source) = &tensor.source {
                    let destination = storage.allocations[*slot].slice()?.view();
                    let layout = if tensor.layout.role == 0 {
                        &tensor.layout
                    } else {
                        &storage.layouts[&(tensor.layout.role, tensor.layout.index)]
                    };
                    for (source_offset, destination_offset, bytes) in
                        tensor_copy_plan(&tensor.layout, layout)?
                    {
                        let source_offset = usize::try_from(source_offset)
                            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
                        let destination_offset = usize::try_from(destination_offset)
                            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
                        let source_end = source_offset
                            .checked_add(bytes)
                            .ok_or(SemanticTransitionError::GenerationExhausted)?;
                        let destination_end = destination_offset
                            .checked_add(bytes)
                            .ok_or(SemanticTransitionError::GenerationExhausted)?;
                        let source = source
                            .try_slice(source_offset..source_end)
                            .ok_or(SemanticTransitionError::ObservationMismatch)?;
                        let destination = destination
                            .try_slice(destination_offset..destination_end)
                            .ok_or(SemanticTransitionError::ObservationMismatch)?;
                        memory.push(
                            crate::device::RetainedDeviceMemoryCommand::copy(
                                &self.stream,
                                source,
                                destination,
                            )
                            .map_err(|error| {
                                runtime_error("publication tensor copy staging", error)
                            })?,
                        );
                    }
                }
            }
        }
        let mut writes = Vec::with_capacity(write_count);
        let fold_after = writes.len();
        for destination in &storage.banks {
            writes.push(OriginalDeviceWrite::new_with_owner(
                &self.stream,
                &[bank],
                destination.view(),
                false,
                host_reservation.clone(),
            )?);
        }
        for index in 0..2 {
            writes.push(OriginalDeviceWrite::new_with_owner(
                &self.stream,
                &storage.bank_templates[index],
                storage.directories[index].view(),
                false,
                host_reservation.clone(),
            )?);
        }
        let entries = storage
            .allocations
            .iter()
            .map(PublicationAllocation::entry)
            .collect::<Vec<_>>();
        writes.push(OriginalDeviceWrite::new_with_owner(
            &self.stream,
            &entries,
            storage.storage.view(),
            false,
            host_reservation.clone(),
        )?);
        writes.push(OriginalDeviceWrite::new_with_owner(
            &self.stream,
            &[storage.contract_value],
            storage.contract.view(),
            false,
            host_reservation.clone(),
        )?);
        if !terminal_tokens.is_empty() {
            writes.push(OriginalDeviceWrite::new_with_owner(
                &self.stream,
                terminal_tokens,
                storage.terminals.view(),
                false,
                host_reservation.clone(),
            )?);
        }
        let counts = role_counts
            .iter()
            .enumerate()
            .map(|(index, &count)| PublicationRoleCount {
                role: index as u64 + 1,
                count,
            })
            .collect::<Vec<_>>();
        writes.push(OriginalDeviceWrite::new_with_owner(
            &self.stream,
            &counts,
            storage.role_counts.view(),
            false,
            host_reservation.clone(),
        )?);
        writes.push(OriginalDeviceWrite::new_with_owner(
            &self.stream,
            &[PendingContinuation {
                abi: 1,
                ranges: storage.continuation_directory.device_ptr_value(),
                range_count: storage.continuation_templates.len() as u64,
                ..PendingContinuation::default()
            }],
            storage.continuation.view(),
            false,
            host_reservation.clone(),
        )?);
        if !storage.continuation_templates.is_empty() {
            writes.push(OriginalDeviceWrite::new_with_owner(
                &self.stream,
                &storage.continuation_templates,
                storage.continuation_directory.view(),
                false,
                host_reservation.clone(),
            )?);
        }
        writes.push(OriginalDeviceWrite::new_with_owner(
            &self.stream,
            &[PublicationControl {
                abi: 1,
                instance: storage.instance,
                banks: storage.banks.each_ref().map(|bank| bank.device_ptr_value()),
                directories: storage
                    .directories
                    .each_ref()
                    .map(|directory| directory.device_ptr_value()),
                storage: storage.storage.device_ptr_value(),
                storage_count: storage.allocations.len() as u64,
                contract: storage.contract.device_ptr_value(),
                continuation: storage.continuation.device_ptr_value(),
                ..PublicationControl::default()
            }],
            storage.control.view(),
            false,
            host_reservation.clone(),
        )?);
        let binding = self.binding();
        writes.push(OriginalDeviceWrite::new_with_owner(
            &self.stream,
            &[DeviceState {
                model_generation: rng.model_generation,
                family_id: u32::from(rng.family_id),
                stream_serial: rng.stream_serial,
                next_proposal: u64::from(rng.proposal),
                catalogue_generation: binding.generation,
                catalogue_digest: Identity256::from_bytes(CATALOGUE_DIGEST),
                binding_digest: binding.digest,
                ..DeviceState::default()
            }],
            self.state.view(),
            true,
            host_reservation.clone(),
        )?);
        let fold_work = fold
            .map(|plan| self.stage_learning_fold(plan, Arc::clone(&storage)))
            .transpose()?;
        if let Some(write) = extra_write {
            writes.push(write);
        }
        Ok(Arc::new(OriginalPublicationPreparation {
            issuer: Arc::clone(&self.publication_issuer),
            restoration: restoration.map(|owner| Arc::clone(&owner.issuance)),
            pending: AtomicBool::new(true),
            original: Mutex::new(PublicationPreparation {
                storage: Some(storage),
                bank: publication_abi_bytes(&[bank]),
                terminal_tokens: terminal_tokens.to_vec(),
                role_counts: *role_counts,
                rng,
                retain_current_models,
                fold: fold.cloned(),
                fold_work,
                fold_completed: false,
                memory,
                memory_next: 0,
                memory_dma_recorded: false,
                #[cfg(feature = "semantic-policy")]
                context_dma_allowance: None,
                uploads,
                upload_next: 0,
                upload_offset: 0,
                upload_end: 0,
                upload_write: None,
                upload_staging,
                upload_staging_bytes,
                host_reservation,
                writes,
                write_next: 0,
                fold_after,
                poisoned: false,
            }),
        }))
    }

    pub(super) fn prepare_publication_storage(
        &mut self,
        bank: PublicationBank,
        terminal_tokens: &[u64],
        role_counts: &[u64; PUBLICATION_ROLE_COUNT],
        rng: SemanticRngBinding,
        fold: Option<&learning_phase::LearningFoldPlan>,
        retain_current_models: bool,
    ) -> Result<(), SemanticTransitionError> {
        if let Some(owner) = &self.publication_preparation {
            let original = owner
                .original
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?;
            if original.bank != publication_abi_bytes(&[bank])
                || original.terminal_tokens != terminal_tokens
                || original.role_counts != *role_counts
                || original.rng != rng
                || original.retain_current_models != retain_current_models
                || original.fold.as_ref() != fold
            {
                return Err(publication_input_error(
                    "publication preparation changed its original inputs",
                ));
            }
        } else {
            let owner = self.stage_publication_preparation(
                bank,
                terminal_tokens,
                role_counts,
                rng,
                fold,
                retain_current_models,
                None,
                None,
            )?;
            self.publication_preparation = Some(owner);
        }
        self.resolve_publication_preparation(None)
    }

    fn resolve_publication_preparation(
        &mut self,
        restoration: Option<&Arc<OriginalStateMaterialRestore>>,
    ) -> Result<(), SemanticTransitionError> {
        #[cfg(feature = "semantic-policy")]
        let dma_allowance = restoration
            .map(|original| original.dma_allowance())
            .transpose()?
            .flatten();
        let owner = Arc::clone(
            self.publication_preparation
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        if !Arc::ptr_eq(&owner.issuer, &self.publication_issuer) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if !owner.pending.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if self.publication.as_ref().is_none_or(|storage| {
            original
                .storage
                .as_ref()
                .is_none_or(|original| !Arc::ptr_eq(storage, original))
        }) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        #[cfg(feature = "semantic-policy")]
        let context_dma_allowance = original.context_dma_allowance.clone();
        #[cfg(feature = "semantic-policy")]
        let dma_allowance = dma_allowance.or(context_dma_allowance.as_ref());
        while original.memory_next < original.memory.len() {
            let may_submit = self.original_restoration_may_submit(restoration, Some(&owner));
            let index = original.memory_next;
            let PublicationPreparation {
                memory,
                memory_dma_recorded,
                ..
            } = &mut *original;
            let command = &mut memory[index];
            #[cfg(feature = "semantic-policy")]
            if let Some(allowance) = dma_allowance {
                allowance
                    .lock()
                    .map_err(|_| SemanticTransitionError::Poisoned)?
                    .require_submitted_dma(command.destination().len(), *memory_dma_recorded)?;
            }
            let mut submission = Ok(());
            if !command.entered() {
                if !may_submit {
                    return Err(SemanticTransitionError::Poisoned);
                }
                let mut recorder = self.domain.new_strict_recorder();
                recorder.write(command.destination());
                if let Some(source) = command.source() {
                    recorder.read(source);
                }
                let mut poisoned = false;
                submission =
                    enqueue_recorded(&self.domain, &mut poisoned, recorder, |_| command.enqueue());
            }
            #[cfg(feature = "semantic-policy")]
            if command.submitted() && !*memory_dma_recorded {
                if let Some(allowance) = dma_allowance {
                    allowance
                        .lock()
                        .map_err(|_| SemanticTransitionError::Poisoned)?
                        .record_submitted_dma(command.destination().len(), memory_dma_recorded)?;
                } else {
                    *memory_dma_recorded = true;
                }
            }
            submission?;
            command
                .resolve()
                .map_err(|error| runtime_error("original publication memory completion", error))?;
            original.memory_next += 1;
            original.memory_dma_recorded = false;
        }
        while original.upload_next < original.uploads.len() {
            let may_submit = self.original_restoration_may_submit(restoration, Some(&owner));
            if original.upload_write.is_none() {
                if !may_submit {
                    return Err(SemanticTransitionError::Poisoned);
                }
                let plan = &original.uploads[original.upload_next];
                let remaining = plan
                    .source
                    .len()
                    .checked_sub(original.upload_offset)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let length = remaining.min(original.upload_staging_bytes);
                if length == 0 {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                let end = original
                    .upload_offset
                    .checked_add(length)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                let destination = plan
                    .destination
                    .try_slice(original.upload_offset..end)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let write = OriginalDeviceWrite::new_with_owner(
                    &self.stream,
                    &plan.source[original.upload_offset..end],
                    destination,
                    false,
                    original.upload_staging.clone(),
                )?;
                original.upload_end = end;
                original.upload_write = Some(write);
            }
            let PublicationPreparation {
                upload_write,
                poisoned,
                ..
            } = &mut *original;
            #[cfg(feature = "semantic-policy")]
            upload_write
                .as_mut()
                .ok_or(SemanticTransitionError::ObservationMismatch)?
                .resolve_with_dma(
                    &self.domain,
                    &self.stream,
                    &self.provider,
                    poisoned,
                    may_submit,
                    dma_allowance,
                )?;
            #[cfg(not(feature = "semantic-policy"))]
            upload_write
                .as_mut()
                .ok_or(SemanticTransitionError::ObservationMismatch)?
                .resolve(
                    &self.domain,
                    &self.stream,
                    &self.provider,
                    poisoned,
                    may_submit,
                )?;
            // Known completion has retired the original pin before another
            // chunk can allocate staging from this same fixed reservation.
            original.upload_offset = original.upload_end;
            original.upload_write = None;
            if original.upload_offset == original.uploads[original.upload_next].source.len() {
                original.upload_next += 1;
                original.upload_offset = 0;
            }
        }
        original.uploads.clear();
        original.upload_staging = None;
        loop {
            if original.write_next == original.fold_after
                && original.fold.is_some()
                && !original.fold_completed
            {
                self.apply_learning_fold(
                    original
                        .fold_work
                        .as_mut()
                        .ok_or(SemanticTransitionError::ObservationMismatch)?,
                    &owner,
                    restoration,
                )?;
                original.fold_completed = true;
                original.fold_work = None;
            }
            if original.write_next == original.writes.len() {
                break;
            }
            let may_submit = self.original_restoration_may_submit(restoration, Some(&owner));
            let index = original.write_next;
            let PublicationPreparation {
                writes, poisoned, ..
            } = &mut *original;
            #[cfg(feature = "semantic-policy")]
            writes[index].resolve_with_dma(
                &self.domain,
                &self.stream,
                &self.provider,
                poisoned,
                may_submit,
                dma_allowance,
            )?;
            #[cfg(not(feature = "semantic-policy"))]
            writes[index].resolve(
                &self.domain,
                &self.stream,
                &self.provider,
                poisoned,
                may_submit,
            )?;
            original.write_next += 1;
        }
        original.poisoned = false;
        original.memory.clear();
        original.writes.clear();
        original.host_reservation = None;
        original.storage = None;
        owner.pending.store(false, Ordering::Release);
        Ok(())
    }
}
