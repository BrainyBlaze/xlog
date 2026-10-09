use super::*;

impl SemanticTransitionSession {
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

    pub(super) fn publication_preparation_read<T: DeviceRepr + Copy>(
        &mut self,
        preparation: &Arc<OriginalPublicationPreparation>,
        source: DeviceMemoryView<T>,
    ) -> Result<Vec<T>, SemanticTransitionError> {
        let restoration = self
            .state_material_restore
            .as_ref()
            .filter(|owner| {
                preparation
                    .restoration
                    .as_ref()
                    .is_some_and(|issuance| Arc::ptr_eq(issuance, &owner.issuance))
            })
            .cloned();
        if !self.original_restoration_may_submit(restoration.as_ref(), Some(preparation)) {
            return Err(SemanticTransitionError::Poisoned);
        }
        let mut original = self.stage_publication_read(source)?;
        self.resolve_publication_read_with_poison(&mut original, &mut false)
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

    fn original_restoration_may_submit(
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

    pub(super) fn restore_state_material_inner(
        &mut self,
        bytes: &[u8],
        transition: Option<&SemanticLearningPhaseTransition>,
        immutable_models: Option<RestoredModelOwners<'_>>,
    ) -> Result<Option<SemanticPublishedIdentity>, SemanticTransitionError> {
        if self.state_material_restore.is_some() {
            return Err(publication_input_error(
                "original state restoration is already retained; resolve that owner",
            ));
        }
        self.require_state_material_restore_entry()?;
        if self.publication.is_some() || self.captured.is_some() || self.task.is_none() {
            return Err(publication_input_error(
                "state restoration requires a fresh uncaptured task-bound Session",
            ));
        }
        let mut material = PublicationMaterial::decode(bytes)?;
        let retain_current_models = immutable_models.is_some_and(RestoredModelOwners::current);
        if retain_current_models && transition.is_some() {
            return Err(publication_input_error(
                "actor context cannot apply a learning-phase fold",
            ));
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
            coverage.bytes = publication_abi_bytes(&[invalidated]);
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
            let (record, absorption) = transition.apply(&mut material)?;
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
            if saved.bytes != *statement {
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
        let owner = Arc::new(OriginalStateMaterialRestore {
            issuer: Arc::clone(&self.publication_issuer),
            issuance,
            context,
            instance,
            initialized: AtomicBool::new(false),
            pending: AtomicBool::new(true),
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
            }),
        });
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
                    let codebooks = ActionCodebooks::derive(
                        admission,
                        self.graph.transition_arena()[1],
                        self.task
                            .as_ref()
                            .and_then(|(task, _)| task.spec.program.editable_program()),
                    )?;
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
                        OriginalDeviceWrite::new(
                            &self.stream,
                            &codebooks.words,
                            self.device_codebooks.view(),
                            true,
                        )?,
                        OriginalDeviceWrite::new(
                            &self.stream,
                            &codebooks.components,
                            self.device_components.view(),
                            true,
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
                    codebook.bytes = relocated;
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
                        )?);
                    }
                    self.resolve_publication_preparation(Some(&owner))?;
                    if original.retain_current_models {
                        self.publication_uploads.clear();
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
                    owner.pending.store(false, Ordering::Release);
                }
                StateRestoreStage::Completed => return Ok(original.result),
            }
        }
    }
}

trait OriginalTypedWrite: Send + Sync {
    fn entered(&self) -> bool;
    fn enqueue(
        &mut self,
        domain: &ResidentExecutionDomain,
        stream: &Arc<CudaStream>,
        poisoned: &mut bool,
    ) -> Result<(), SemanticTransitionError>;
    fn resolve(&mut self) -> Result<(), SemanticTransitionError>;
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
    fn entered(&self) -> bool {
        self.write.entered()
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
            self.write.enqueue(enqueue.stream())
        })
    }

    fn resolve(&mut self) -> Result<(), SemanticTransitionError> {
        self.write
            .resolve()
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
pub(super) struct OriginalDeviceWrite {
    transfer: Box<dyn OriginalTypedWrite>,
    bytes: usize,
    tracked: bool,
    admitted: bool,
    completed: bool,
    retired: bool,
}

impl OriginalDeviceWrite {
    pub(super) fn new<T: DeviceRepr + Copy + Send + Sync + 'static>(
        stream: &CudaStream,
        values: &[T],
        destination: DeviceMemoryView<T>,
        tracked: bool,
    ) -> Result<Self, SemanticTransitionError> {
        let bytes = values
            .len()
            .checked_mul(size_of::<T>())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let write = crate::device::RetainedDeviceWrite::new(stream, values, destination.clone())
            .map_err(|error| runtime_error("original device write staging", error))?;
        Ok(Self {
            transfer: Box::new(TypedOriginalWrite { destination, write }),
            bytes,
            tracked,
            admitted: false,
            completed: bytes == 0,
            retired: false,
        })
    }

    pub(super) fn entered(&self) -> bool {
        self.transfer.entered()
    }

    pub(super) fn pending(&self) -> bool {
        self.entered() && !self.completed && !self.retired
    }

    pub(super) fn resolve(
        &mut self,
        domain: &ResidentExecutionDomain,
        stream: &Arc<CudaStream>,
        provider: &CudaKernelProvider,
        poisoned: &mut bool,
        may_submit: bool,
    ) -> Result<(), SemanticTransitionError> {
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
            if !self.admitted {
                if self.tracked {
                    provider.admit_tracked_htod(self.bytes);
                } else {
                    provider.admit_launch_metadata_htod(self.bytes);
                }
                self.admitted = true;
            }
            self.transfer.enqueue(domain, stream, poisoned)?;
        }
        self.transfer.resolve()?;
        self.completed = true;
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
        Ok(())
    }
}

pub(super) struct OriginalPublicationPreparation {
    issuer: Arc<()>,
    restoration: Option<Arc<()>>,
    pending: AtomicBool,
    original: Mutex<PublicationPreparation>,
}

struct PublicationPreparation {
    storage: Option<Arc<PublicationStorage>>,
    bank: Vec<u8>,
    terminal_tokens: Vec<u64>,
    role_counts: [u64; PUBLICATION_ROLE_COUNT],
    rng: SemanticRngBinding,
    retain_current_models: bool,
    fold: Option<learning_phase::LearningFoldPlan>,
    fold_entered: bool,
    fold_completed: bool,
    memory: Vec<crate::device::RetainedDeviceMemoryCommand>,
    memory_next: usize,
    writes: Vec<OriginalDeviceWrite>,
    write_next: usize,
    fold_after: usize,
    poisoned: bool,
}

enum OwnedRestoredModelOwners {
    Historical(Vec<Arc<ReplayModelBacking>>),
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

pub(super) struct OriginalStateMaterialRestore {
    issuer: Arc<()>,
    issuance: Arc<()>,
    context: Option<Arc<()>>,
    instance: Identity256,
    initialized: AtomicBool,
    pending: AtomicBool,
    original: Mutex<StateMaterialRestore>,
}

pub(super) struct RestoredReplayAppendMaterial {
    restoration: Arc<()>,
    entries: PublicationMaterialRange,
    payload: PublicationMaterialRange,
}

impl RestoredReplayAppendMaterial {
    pub(super) fn records(&self) -> (&PublicationMaterialRange, &PublicationMaterialRange) {
        (&self.entries, &self.payload)
    }
}

struct StateMaterialRestore {
    material: PublicationMaterial,
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
}

impl SemanticTransitionSession {
    fn stage_publication_preparation(
        &self,
        bank: PublicationBank,
        terminal_tokens: &[u64],
        role_counts: &[u64; PUBLICATION_ROLE_COUNT],
        rng: SemanticRngBinding,
        fold: Option<&learning_phase::LearningFoldPlan>,
        retain_current_models: bool,
        restoration: Option<&Arc<OriginalStateMaterialRestore>>,
    ) -> Result<Arc<OriginalPublicationPreparation>, SemanticTransitionError> {
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
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
        let mut writes = Vec::new();
        for (slot, payload) in &self.publication_uploads {
            if let PublicationPayload::Metadata(bytes) = payload {
                if !bytes.is_empty() {
                    writes.push(OriginalDeviceWrite::new(
                        &self.stream,
                        bytes,
                        storage.allocations[*slot]
                            .slice()?
                            .view()
                            .slice(..bytes.len()),
                        false,
                    )?);
                }
            }
        }
        let fold_after = writes.len();
        for destination in &storage.banks {
            writes.push(OriginalDeviceWrite::new(
                &self.stream,
                &[bank],
                destination.view(),
                false,
            )?);
        }
        for index in 0..2 {
            writes.push(OriginalDeviceWrite::new(
                &self.stream,
                &storage.bank_templates[index],
                storage.directories[index].view(),
                false,
            )?);
        }
        let entries = storage
            .allocations
            .iter()
            .map(PublicationAllocation::entry)
            .collect::<Vec<_>>();
        writes.push(OriginalDeviceWrite::new(
            &self.stream,
            &entries,
            storage.storage.view(),
            false,
        )?);
        writes.push(OriginalDeviceWrite::new(
            &self.stream,
            &[storage.contract_value],
            storage.contract.view(),
            false,
        )?);
        if !terminal_tokens.is_empty() {
            writes.push(OriginalDeviceWrite::new(
                &self.stream,
                terminal_tokens,
                storage.terminals.view(),
                false,
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
        writes.push(OriginalDeviceWrite::new(
            &self.stream,
            &counts,
            storage.role_counts.view(),
            false,
        )?);
        writes.push(OriginalDeviceWrite::new(
            &self.stream,
            &[PendingContinuation {
                abi: 1,
                ranges: storage.continuation_directory.device_ptr_value(),
                range_count: storage.continuation_templates.len() as u64,
                ..PendingContinuation::default()
            }],
            storage.continuation.view(),
            false,
        )?);
        if !storage.continuation_templates.is_empty() {
            writes.push(OriginalDeviceWrite::new(
                &self.stream,
                &storage.continuation_templates,
                storage.continuation_directory.view(),
                false,
            )?);
        }
        writes.push(OriginalDeviceWrite::new(
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
        )?);
        let binding = self.binding();
        writes.push(OriginalDeviceWrite::new(
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
        )?);
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
                fold_entered: false,
                fold_completed: false,
                memory,
                memory_next: 0,
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
            )?;
            self.publication_preparation = Some(owner);
        }
        self.resolve_publication_preparation(None)
    }

    fn resolve_publication_preparation(
        &mut self,
        restoration: Option<&Arc<OriginalStateMaterialRestore>>,
    ) -> Result<(), SemanticTransitionError> {
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
        while original.memory_next < original.memory.len() {
            let may_submit = self.original_restoration_may_submit(restoration, Some(&owner));
            let index = original.memory_next;
            let command = &mut original.memory[index];
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
                enqueue_recorded(&self.domain, &mut poisoned, recorder, |_| command.enqueue())?;
            }
            command
                .resolve()
                .map_err(|error| runtime_error("original publication memory completion", error))?;
            original.memory_next += 1;
        }
        loop {
            if original.write_next == original.fold_after
                && original.fold.is_some()
                && !original.fold_completed
            {
                if original.fold_entered {
                    return Err(publication_input_error(
                        "original learning fold completion remains unresolved",
                    ));
                }
                if !self.original_restoration_may_submit(restoration, Some(&owner)) {
                    return Err(SemanticTransitionError::Poisoned);
                }
                original.fold_entered = true;
                self.apply_learning_fold(
                    original.fold.as_ref().expect("original learning fold"),
                    &owner,
                )?;
                original.fold_completed = true;
            }
            if original.write_next == original.writes.len() {
                break;
            }
            let may_submit = self.original_restoration_may_submit(restoration, Some(&owner));
            let index = original.write_next;
            let PublicationPreparation {
                writes, poisoned, ..
            } = &mut *original;
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
        original.storage = None;
        owner.pending.store(false, Ordering::Release);
        Ok(())
    }
}
