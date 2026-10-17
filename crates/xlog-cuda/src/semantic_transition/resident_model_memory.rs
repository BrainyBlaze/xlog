//! Structural aliases and admitted content checks of one resident model owner.

#[cfg(feature = "semantic-policy")]
use super::cold_model_work::{
    SemanticColdNativeWorkActivationIdentity, SemanticColdNativeWorkActivationPermit,
};
use super::replay_verification::OriginalPublicationGuard;
use super::*;

/// One original reader's structural envelope. Preparing this owner performs no
/// device work and grants no numerical access. Its backing capsules are issued
/// once; activation checks their original content on every numerical occurrence.
pub struct SemanticPreparedResidentModelMemory {
    owner: Arc<ResidentModelMemoryOwner>,
    memory: Option<SemanticResidentModelMemory>,
}

pub(super) struct ResidentModelMemoryOwner {
    issuer: Arc<()>,
    token: u64,
    identity: SemanticPublishedIdentity,
    header: PublicationHeader,
    directory: Vec<PublicationRange>,
    consumer_stream: u64,
    reader_pin: Arc<()>,
    storage: Arc<PublicationStorage>,
    keys: Vec<(SemanticStateRole, u64)>,
    exports: Vec<Arc<ResidentModelExportOwner>>,
    model_contract: Vec<u8>,
    model_contract_layout: SemanticModelContractLayout,
    restoration: Option<std::sync::Weak<state_restoration::OriginalStateMaterialRestore>>,
    #[cfg(feature = "semantic-policy")]
    checkpoint_geometry: Option<Arc<publication_readback::CheckpointReadbackGeometry>>,
}

/// An exact numerical read occurrence in one original, still-unmaterialized report.
#[cfg(feature = "semantic-policy")]
#[derive(Clone)]
pub struct SemanticResidentModelRead {
    occurrence: Arc<ResidentModelReadOccurrence>,
}

#[cfg(feature = "semantic-policy")]
struct ResidentModelReadOccurrence {
    work: SemanticColdModelWork,
    region: usize,
    ceiling: [u64; 9],
    origin: ResidentModelReadOrigin,
    readback: std::sync::OnceLock<SemanticPublicationReadbackLoan>,
    _reservation: SemanticReplayMaterialReservation,
}

#[cfg(feature = "semantic-policy")]
enum ResidentModelReadOrigin {
    Selected(std::sync::Weak<ResidentModelMemoryOwner>),
    Restored(std::sync::Weak<ResidentModelRestorationOwner>),
}

/// The original planned snapshot producer, carried into its one native restore.
#[cfg(feature = "semantic-policy")]
#[derive(Clone)]
pub struct SemanticPreparedResidentModelRestoration {
    pub(super) owner: Arc<ResidentModelRestorationOwner>,
}

#[cfg(feature = "semantic-policy")]
pub(super) struct ResidentModelRestorationOwner {
    source: std::sync::Weak<ResidentModelMemoryOwner>,
    source_identity: SemanticPublishedIdentity,
    consumer_stream: u64,
    geometry: ModelMemoryGeometry,
    checkpoint_geometry: Arc<publication_readback::CheckpointReadbackGeometry>,
    keys: Vec<(SemanticStateRole, u64)>,
    layouts: Vec<SemanticTensorLayout>,
    ceiling: [u64; 9],
    dma_ceiling: [u64; 9],
    budget: SemanticReplayMaterialBudget,
    _reservation: SemanticReplayMaterialReservation,
    work: SemanticColdModelWork,
    snapshot: Mutex<Option<SemanticReplayStateMaterialProof>>,
    restoration: Mutex<Option<std::sync::Weak<state_restoration::OriginalStateMaterialRestore>>>,
    native_prelude: std::sync::OnceLock<std::sync::Weak<SemanticColdResidentRestorationPrelude>>,
    reads: [std::sync::OnceLock<SemanticResidentModelRead>; 2],
}

#[cfg(feature = "semantic-policy")]
impl SemanticResidentModelRead {
    /// Bind the original writer workspace before this report's sole admission.
    /// Full backing extents come from the actual selected/restored producer.
    pub fn prepare_checkpoint_readback(
        &self,
        session: &SemanticTransitionSession,
        work: &SemanticColdModelWork,
        region: usize,
        working_bytes: u64,
        codec_staging_bytes: usize,
    ) -> Result<SemanticPublicationReadbackLoan, SemanticTransitionError> {
        self.require_report(work, region)?;
        if let Some(original) = self.occurrence.readback.get() {
            original.require_workspace(working_bytes, codec_staging_bytes)?;
            return Ok(original.clone());
        }
        session.require_cold_checkpoint_writer_preparation(work, region)?;
        let scratch = self
            .material_budget()
            .prepare_material_scratch(working_bytes)?;
        let loan = match &self.occurrence.origin {
            ResidentModelReadOrigin::Selected(original) => {
                let original = original.upgrade().ok_or_else(|| {
                    publication_input_error("checkpoint readback lost its original resident owner")
                })?;
                scratch.prepare_readback_loan(
                    &original.storage.model_memory.allocation_bytes,
                    codec_staging_bytes,
                    Arc::clone(original.checkpoint_geometry.as_ref().ok_or_else(|| {
                        publication_input_error(
                            "checkpoint readback lost its original producer metadata",
                        )
                    })?),
                    publication_readback::CheckpointReadbackOrigin::Selected,
                )?
            }
            ResidentModelReadOrigin::Restored(original) => {
                let original = original.upgrade().ok_or_else(|| {
                    publication_input_error(
                        "checkpoint readback lost its original restoration issuer",
                    )
                })?;
                scratch.prepare_readback_loan(
                    &original.geometry.allocation_bytes,
                    codec_staging_bytes,
                    Arc::clone(&original.checkpoint_geometry),
                    publication_readback::CheckpointReadbackOrigin::Restored,
                )?
            }
        };
        self.occurrence.readback.set(loan).map_err(|_| {
            publication_input_error("checkpoint readback was concurrently prepared")
        })?;
        Ok(self
            .occurrence
            .readback
            .get()
            .expect("original pre-admission readback loan")
            .clone())
    }

    pub(super) fn checkpoint_readback_ceiling(&self) -> [u64; 9] {
        self.occurrence
            .readback
            .get()
            .map_or([0; 9], SemanticPublicationReadbackLoan::native_work_ceiling)
    }

    pub(super) fn checkpoint_readback_loan(&self) -> Option<&SemanticPublicationReadbackLoan> {
        self.occurrence.readback.get()
    }

    pub fn material_budget(&self) -> SemanticReplayMaterialBudget {
        self.occurrence._reservation.budget().clone()
    }

    pub(super) fn restoration_issuer(
        &self,
    ) -> Result<Option<SemanticPreparedResidentModelRestoration>, SemanticTransitionError> {
        if self.occurrence.region != 2 {
            return Ok(None);
        }
        match &self.occurrence.origin {
            ResidentModelReadOrigin::Selected(_) => Ok(None),
            ResidentModelReadOrigin::Restored(original) => {
                Ok(Some(SemanticPreparedResidentModelRestoration {
                    owner: original.upgrade().ok_or_else(|| {
                        publication_input_error("resident restoration lost its original issuer")
                    })?,
                }))
            }
        }
    }

    pub(super) fn native_work_ceiling(&self) -> Result<[u64; 9], SemanticTransitionError> {
        Ok(self.occurrence.ceiling)
    }

    pub(super) fn require_report(
        &self,
        work: &SemanticColdModelWork,
        region: usize,
    ) -> Result<(), SemanticTransitionError> {
        if !self.occurrence.work.same_invocation(work) || self.occurrence.region != region {
            return Err(publication_input_error(
                "resident read changed its original report occurrence",
            ));
        }
        Ok(())
    }

    pub(super) fn same_occurrence(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.occurrence, &other.occurrence)
    }

    pub(super) fn claim_actual(
        &self,
        owner: &Arc<ResidentModelMemoryOwner>,
        ceiling: [u64; 9],
    ) -> Result<(), SemanticTransitionError> {
        if ceiling != self.occurrence.ceiling || owner.native_guard_ceiling()? != ceiling {
            return Err(publication_input_error(
                "resident read changed its complete admitted guard geometry",
            ));
        }
        match &self.occurrence.origin {
            ResidentModelReadOrigin::Selected(original)
                if original
                    .upgrade()
                    .is_some_and(|original| Arc::ptr_eq(&original, owner)) =>
            {
                Ok(())
            }
            ResidentModelReadOrigin::Restored(original) => {
                let original = SemanticPreparedResidentModelRestoration {
                    owner: original.upgrade().ok_or_else(|| {
                        publication_input_error(
                            "future resident read lost its original planned producer",
                        )
                    })?,
                };
                original.require_geometry(owner)?;
                let restore = owner
                    .restoration
                    .as_ref()
                    .and_then(std::sync::Weak::upgrade)
                    .ok_or_else(|| {
                        publication_input_error(
                            "future resident read has no original native restoration",
                        )
                    })?;
                restore.require_resident_read(&original, owner.identity)
            }
            _ => Err(publication_input_error(
                "resident read replaced its original selected owner",
            )),
        }
    }
}

#[cfg(feature = "semantic-policy")]
impl SemanticPreparedResidentModelRestoration {
    pub fn bind_native_prelude(
        &self,
        prelude: &Arc<SemanticColdResidentRestorationPrelude>,
    ) -> Result<(), SemanticTransitionError> {
        prelude.require_restoration(self)?;
        let original = self
            .owner
            .native_prelude
            .get_or_init(|| Arc::downgrade(prelude));
        if !original.ptr_eq(&Arc::downgrade(prelude)) {
            return Err(publication_input_error(
                "resident restoration cannot replace its original native prelude",
            ));
        }
        Ok(())
    }

    pub(super) fn native_prelude(
        &self,
    ) -> Result<Arc<SemanticColdResidentRestorationPrelude>, SemanticTransitionError> {
        let original = self
            .owner
            .native_prelude
            .get()
            .and_then(std::sync::Weak::upgrade)
            .ok_or_else(|| {
                publication_input_error(
                    "resident restoration lost its original admitted native prelude",
                )
            })?;
        original.require_restoration(self)?;
        Ok(original)
    }

    pub fn same_issuer(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
    }

    pub(super) fn require_report(
        &self,
        work: &SemanticColdModelWork,
        region: usize,
    ) -> Result<(), SemanticTransitionError> {
        if region != 2 || !self.owner.work.same_invocation(work) {
            return Err(publication_input_error(
                "resident restoration changed its original report or restore region",
            ));
        }
        Ok(())
    }

    pub(super) fn dma_native_work_ceiling(&self) -> Result<[u64; 9], SemanticTransitionError> {
        Ok(self.owner.dma_ceiling)
    }

    pub fn same_source(&self, memory: &SemanticPreparedResidentModelMemory) -> bool {
        self.owner
            .source
            .upgrade()
            .is_some_and(|original| Arc::ptr_eq(&original, &memory.owner))
    }
    pub(super) fn require_native_restore(
        &self,
        restore: &Arc<state_restoration::OriginalStateMaterialRestore>,
    ) -> Result<(), SemanticTransitionError> {
        let original = self
            .owner
            .restoration
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .is_none_or(|original| !Arc::ptr_eq(&original, restore))
        {
            return Err(publication_input_error(
                "resident read changed its original native restore issuer",
            ));
        }
        Ok(())
    }

    pub fn prepare_read(
        &self,
        work: &SemanticColdModelWork,
        region: usize,
    ) -> Result<SemanticResidentModelRead, SemanticTransitionError> {
        if !self.owner.work.same_invocation(work) || !matches!(region, 2 | 3) {
            return Err(publication_input_error(
                "future resident read requires its original restore or post-restore snapshot region",
            ));
        }
        let slot = &self.owner.reads[region - 2];
        if let Some(original) = slot.get() {
            return Ok(original.clone());
        }
        let read = SemanticResidentModelRead {
            occurrence: Arc::new(ResidentModelReadOccurrence {
                work: work.clone(),
                region,
                ceiling: self.owner.ceiling,
                origin: ResidentModelReadOrigin::Restored(Arc::downgrade(&self.owner)),
                readback: std::sync::OnceLock::new(),
                _reservation: self.owner.budget.reserve_auxiliary(
                    (size_of::<ResidentModelReadOccurrence>() + 2 * size_of::<usize>()) as u64,
                )?,
            }),
        };
        let _ = slot.set(read);
        Ok(slot.get().expect("original planned resident read").clone())
    }

    fn require_geometry(
        &self,
        actual: &ResidentModelMemoryOwner,
    ) -> Result<(), SemanticTransitionError> {
        if self.owner.geometry != actual.storage.model_memory
            || self.owner.keys != actual.keys
            || self.owner.consumer_stream != actual.consumer_stream
            || self
                .owner
                .keys
                .iter()
                .zip(&self.owner.layouts)
                .any(|((role, index), layout)| {
                    let key = (*role as u64, *index);
                    Some(layout) != actual.storage.layouts.get(&key)
                })
        {
            return Err(publication_input_error(
                "restored resident model changed its original planned typed geometry",
            ));
        }
        Ok(())
    }

    pub fn bind_original_snapshot(
        &self,
        session: &SemanticTransitionSession,
        lease: &SemanticPublishedLease,
        proof: SemanticReplayStateMaterialProof,
    ) -> Result<(), SemanticTransitionError> {
        let source = self.owner.source.upgrade().ok_or_else(|| {
            publication_input_error("planned snapshot lost its original resident source")
        })?;
        session.require_resident_model_memory(lease, &source, self.owner.consumer_stream)?;
        session.require_checkpoint_source_digest(
            lease,
            proof.original_identity(),
            proof.source_digest(),
        )?;
        self.require_snapshot_geometry(&proof)?;
        let mut saved = self
            .owner
            .snapshot
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if saved.is_some() {
            return Err(publication_input_error(
                "planned resident restoration already retained its original snapshot",
            ));
        }
        *saved = Some(proof);
        Ok(())
    }

    fn require_snapshot_geometry(
        &self,
        proof: &SemanticReplayStateMaterialProof,
    ) -> Result<(), SemanticTransitionError> {
        let (_, _, geometry, layouts, _) = proof.model_guard_descriptor();
        if proof.original_identity() != self.owner.source_identity
            || geometry != &self.owner.geometry
            || self
                .owner
                .keys
                .iter()
                .zip(&self.owner.layouts)
                .any(|((role, index), layout)| {
                    let key = (*role as u64, *index);
                    layouts.get(&key) != Some(layout)
                })
        {
            return Err(publication_input_error(
                "planned snapshot changed its original resident geometry or source",
            ));
        }
        Ok(())
    }

    pub(super) fn require_loaded(
        &self,
        loaded: &mut SemanticReplayLoadedMaterial,
    ) -> Result<(), SemanticTransitionError> {
        let proof = loaded.checkpoint_proof()?;
        self.require_snapshot_geometry(&proof)?;
        let saved = self
            .owner
            .snapshot
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let original = saved.as_ref().ok_or_else(|| {
            publication_input_error("resident restoration precedes its original planned snapshot")
        })?;
        if original.source_digest() != proof.source_digest() {
            return Err(publication_input_error(
                "resident restoration replaced its original snapshot material",
            ));
        }
        Ok(())
    }

    pub(super) fn retain_native_restore(
        &self,
        restore: &Arc<state_restoration::OriginalStateMaterialRestore>,
    ) -> Result<(), SemanticTransitionError> {
        let mut original = self
            .owner
            .restoration
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original.is_some() {
            return Err(publication_input_error(
                "planned resident restoration cannot replace its original native restore",
            ));
        }
        *original = Some(Arc::downgrade(restore));
        Ok(())
    }
}

#[cfg(feature = "semantic-policy")]
impl ResidentModelMemoryOwner {
    pub(super) fn native_guard_ceiling(&self) -> Result<[u64; 9], SemanticTransitionError> {
        super::native_work_bound::resident_model_content_guard_native_work_ceiling(
            &self.storage,
            &self.directory,
            &self.keys,
        )
    }
}

// Capsules retain only their genuine numerical generation, not the publication
// reader. The carrier separately pins that reader through cold use and Unknown.
struct ResidentModelExportOwner {
    generation: Arc<ModelGenerationOwner>,
    view: DeviceMemoryView<u8>,
    retirement: Mutex<Option<ModelExportUse>>,
}

impl ResidentModelExportOwner {
    fn arm(
        &self,
        stream: &Arc<CudaStream>,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        let mut retirement = self
            .retirement
            .lock()
            .map_err(|_| publication_input_error("resident model export owner poisoned"))?;
        if retirement.is_none() {
            let event = stream.context().new_event(None).map_err(|error| {
                runtime_error("model consumer retirement event allocation", error)
            })?;
            *retirement = Some(ModelExportUse {
                _generation: Some(Arc::clone(&self.generation)),
                _view: self.view.clone(),
                _publication: None,
                _guard: None,
                event,
                stream: Arc::clone(stream),
                consumer_stream,
                _retained_owner: None,
            });
        }
        Ok(())
    }
}

impl Drop for ResidentModelExportOwner {
    fn drop(&mut self) {
        let retirement = self
            .retirement
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        // Reuse the original model-export final-use retirement. An unactivated
        // structural alias has no device use or event to retire.
        drop(ModelExportOwner(retirement));
    }
}

pub(super) struct ResidentModelActivation {
    memory: Arc<ResidentModelMemoryOwner>,
    guard: Option<OriginalPublicationGuard>,
    authority: ResidentModelActivationAuthority,
    #[cfg(feature = "semantic-policy")]
    guard_claimed: bool,
}

enum ResidentModelActivationAuthority {
    Ordinary,
    #[cfg(feature = "semantic-policy")]
    Cold(SemanticColdNativeWorkActivationIdentity),
}

impl ResidentModelActivationAuthority {
    fn same_context(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Ordinary, Self::Ordinary) => true,
            #[cfg(feature = "semantic-policy")]
            (Self::Cold(original), Self::Cold(current)) => original.same_context(current),
            #[cfg(feature = "semantic-policy")]
            _ => false,
        }
    }
}

impl SemanticPreparedResidentModelMemory {
    #[cfg(feature = "semantic-policy")]
    pub fn prepare_resident_read(
        &self,
        session: &SemanticTransitionSession,
        lease: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
        region: usize,
    ) -> Result<SemanticResidentModelRead, SemanticTransitionError> {
        session.require_resident_model_memory(lease, &self.owner, self.owner.consumer_stream)?;
        let budget = session.replay_material_budget.as_ref().ok_or_else(|| {
            publication_input_error("resident read requires its original bound material pool")
        })?;
        let reservation = budget.reserve_auxiliary(
            (size_of::<ResidentModelReadOccurrence>() + 2 * size_of::<usize>()) as u64,
        )?;
        Ok(SemanticResidentModelRead {
            occurrence: Arc::new(ResidentModelReadOccurrence {
                work: work.clone(),
                region,
                ceiling: self.owner.native_guard_ceiling()?,
                origin: ResidentModelReadOrigin::Selected(Arc::downgrade(&self.owner)),
                readback: std::sync::OnceLock::new(),
                _reservation: reservation,
            }),
        })
    }

    #[cfg(feature = "semantic-policy")]
    pub fn prepare_resident_restoration(
        &self,
        session: &SemanticTransitionSession,
        lease: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
    ) -> Result<SemanticPreparedResidentModelRestoration, SemanticTransitionError> {
        session.require_resident_model_memory(lease, &self.owner, self.owner.consumer_stream)?;
        let budget = session.replay_material_budget.as_ref().ok_or_else(|| {
            publication_input_error(
                "resident restoration requires its original bound material pool",
            )
        })?;
        let geometry = &self.owner.storage.model_memory;
        let dma_ceiling =
            native_work_bound::restoration_dma_native_work_ceiling(session, lease, budget)?;
        let mut bytes =
            (size_of::<ResidentModelRestorationOwner>() + 2 * size_of::<usize>()) as u64;
        for (count, width) in [
            (geometry.allocation_bytes.len(), size_of::<u64>()),
            (geometry.storages.len(), size_of::<SemanticModelStorage>()),
            (geometry.views.len(), size_of::<SemanticModelView>()),
            (self.owner.keys.len(), size_of::<(SemanticStateRole, u64)>()),
            (self.owner.keys.len(), size_of::<SemanticTensorLayout>()),
        ] {
            bytes = bytes
                .checked_add(
                    (count as u64)
                        .checked_mul(width as u64)
                        .ok_or(SemanticTransitionError::GenerationExhausted)?,
                )
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let reservation = budget.reserve_auxiliary(bytes)?;
        let layouts = self
            .owner
            .keys
            .iter()
            .map(|(role, index)| {
                let key = (*role as u64, *index);
                self.owner
                    .storage
                    .layouts
                    .get(&key)
                    .copied()
                    .ok_or(SemanticTransitionError::ObservationMismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SemanticPreparedResidentModelRestoration {
            owner: Arc::new(ResidentModelRestorationOwner {
                source: Arc::downgrade(&self.owner),
                source_identity: self.owner.identity,
                consumer_stream: self.owner.consumer_stream,
                geometry: geometry.clone(),
                checkpoint_geometry: Arc::clone(
                    self.owner.checkpoint_geometry.as_ref().ok_or_else(|| {
                        publication_input_error(
                            "resident restoration lost its original producer metadata",
                        )
                    })?,
                ),
                keys: self.owner.keys.clone(),
                layouts,
                ceiling: self.owner.native_guard_ceiling()?,
                dma_ceiling,
                budget: budget.clone(),
                _reservation: reservation,
                work: work.clone(),
                snapshot: Mutex::new(None),
                restoration: Mutex::new(None),
                native_prelude: std::sync::OnceLock::new(),
                reads: std::array::from_fn(|_| std::sync::OnceLock::new()),
            }),
        })
    }
    #[cfg(feature = "semantic-policy")]
    pub(super) fn source_content_owner(
        &self,
        session: &SemanticTransitionSession,
        lease: &SemanticPublishedLease,
        consumer_stream: u64,
    ) -> Result<std::sync::Weak<ResidentModelMemoryOwner>, SemanticTransitionError> {
        session.require_resident_model_memory(lease, &self.owner, consumer_stream)?;
        Ok(Arc::downgrade(&self.owner))
    }

    /// Move the genuine allocation capsules into their sole structural carrier.
    pub fn take_memory(&mut self) -> Result<SemanticResidentModelMemory, SemanticTransitionError> {
        self.memory.take().ok_or_else(|| {
            publication_input_error("resident model memory capsules were already issued")
        })
    }

    /// Complete immutable record bytes from this same original selected parent.
    pub fn model_contract_metadata(&self) -> (&[u8], SemanticModelContractLayout) {
        (&self.owner.model_contract, self.owner.model_contract_layout)
    }
}

impl SemanticTransitionSession {
    /// Retain the selected resident bank and reconstruct its structural aliases
    /// without guards, private-step copies, stream edges or retirement events.
    pub fn prepare_resident_model_memory(
        &self,
        lease: &SemanticPublishedLease,
        consumer_stream: u64,
    ) -> Result<SemanticPreparedResidentModelMemory, SemanticTransitionError> {
        dlpack_consumer_stream(consumer_stream)?;
        let reader_pin = Arc::clone(&self.checked_reader(lease)?.aliases);
        let header = lease.model_context_header()?;
        let (model_contract, model_contract_layout) =
            self.published_model_contract_metadata(lease)?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        storage.model_memory.validate(&storage.layouts)?;
        if storage.model_slots.len() != storage.model_memory.allocation_bytes.len() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let keys = lease
            .directory
            .iter()
            .filter(|range| matches!(range.role, 18..=25))
            .map(|range| {
                SemanticStateRole::from_code(range.role)
                    .map(|role| (role, range.index))
                    .ok_or(SemanticTransitionError::ObservationMismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if keys.len() != storage.model_memory.views.len() {
            return Err(publication_input_error(
                "resident model memory requires the complete selected model roster",
            ));
        }

        let mut allocations = Vec::with_capacity(storage.model_slots.len());
        let mut exports = Vec::with_capacity(storage.model_slots.len());
        for (allocation_index, (&bytes, slots)) in storage
            .model_memory
            .allocation_bytes
            .iter()
            .zip(&storage.model_slots)
            .enumerate()
        {
            let mut selected_slot = None;
            for range in lease
                .directory
                .iter()
                .filter(|range| matches!(range.role, 18..=25))
            {
                let (owner, _) = storage.model_memory.location(range.role, range.index)?;
                if owner == allocation_index
                    && selected_slot
                        .replace(range.storage_slot as usize)
                        .is_some_and(|prior| prior != range.storage_slot as usize)
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
            }
            let slot = selected_slot.unwrap_or(slots[(lease.identity.word & 1) as usize]);
            if !storage.model_generations[allocation_index].contains(&slot) {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let allocation = storage
                .allocations
                .get(slot)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if allocation.len() as u64 != bytes {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let view = allocation.slice()?.view();
            let provenance = view.allocation_provenance().ok_or_else(|| {
                publication_input_error("resident model backing has no native allocation owner")
            })?;
            let generation = allocation
                .model_owner()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let extent =
                i64::try_from(bytes).map_err(|_| SemanticTransitionError::ObservationMismatch)?;
            let export = Arc::new(ResidentModelExportOwner {
                generation,
                view: view.clone(),
                retirement: Mutex::new(None),
            });
            let retained_owner: Arc<dyn Send + Sync> = export.clone();
            let tensor = export_owned_allocation(
                view,
                vec![extent],
                vec![1],
                (1, 8),
                self.provider.device().ordinal() as i32,
                Arc::new(()),
                None,
                None,
                Some(retained_owner),
            );
            allocations.push((provenance, tensor));
            exports.push(export);
        }
        let views = storage
            .model_memory
            .views
            .iter()
            .cloned()
            .map(|view| {
                let layout = storage
                    .layouts
                    .get(&(view.role, view.index))
                    .copied()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                Ok((view, layout))
            })
            .collect::<Result<Vec<_>, SemanticTransitionError>>()?;
        let memory = SemanticResidentModelMemory {
            parent: lease.identity,
            generations: (
                header.model_generation,
                header.neural_bank,
                header.neural_generation,
            ),
            numerical_state: (header.model_geometry_digest, header.model_numerical_digest),
            allocations,
            storages: storage.model_memory.storages.clone(),
            views,
        };
        #[cfg(feature = "semantic-policy")]
        let checkpoint_geometry = if self.replay_material_budget.is_some() {
            Some(self.checkpoint_readback_geometry(&storage, &lease.directory)?)
        } else {
            None
        };
        Ok(SemanticPreparedResidentModelMemory {
            owner: Arc::new(ResidentModelMemoryOwner {
                issuer: Arc::clone(&lease.issuer),
                token: lease.token,
                identity: lease.identity,
                header,
                directory: lease.directory.clone(),
                consumer_stream,
                reader_pin,
                storage,
                keys,
                exports,
                model_contract,
                model_contract_layout,
                restoration: self.state_material_restore.as_ref().map(Arc::downgrade),
                #[cfg(feature = "semantic-policy")]
                checkpoint_geometry,
            }),
            memory: Some(memory),
        })
    }

    /// Check every original typed seal and full backing seal before this
    /// occurrence reads resident content. Unknown retains this exact command
    /// roster in the Session; a later successful occurrence starts fresh guards.
    #[cfg(feature = "semantic-policy")]
    pub fn activate_resident_model_memory(
        &mut self,
        lease: &SemanticPublishedLease,
        memory: &SemanticPreparedResidentModelMemory,
        consumer_stream: u64,
        work: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<(), SemanticTransitionError> {
        let identity = self.require_cold_model_work_activation(work, region, lease)?;
        self.activate_resident_model_memory_original(
            lease,
            memory,
            consumer_stream,
            ResidentModelActivationAuthority::Cold(identity),
        )
    }

    /// Use a fresh borrow of the original recording owner for a child callback.
    #[cfg(feature = "semantic-policy")]
    pub fn activate_resident_model_memory_with_permit(
        &mut self,
        lease: &SemanticPublishedLease,
        memory: &SemanticPreparedResidentModelMemory,
        consumer_stream: u64,
        permit: &SemanticColdNativeWorkActivationPermit<'_>,
    ) -> Result<(), SemanticTransitionError> {
        let identity = permit.identity_for_callback(self, lease)?;
        self.activate_resident_model_memory_original(
            lease,
            memory,
            consumer_stream,
            ResidentModelActivationAuthority::Cold(identity),
        )
    }

    pub(super) fn activate_resident_model_memory_ordinary(
        &mut self,
        lease: &SemanticPublishedLease,
        memory: &SemanticPreparedResidentModelMemory,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        #[cfg(feature = "semantic-policy")]
        if self.cold_native_work(lease.token)?.is_some() {
            return Err(publication_input_error(
                "cold resident model access requires its original admitted recording owner",
            ));
        }
        self.activate_resident_model_memory_original(
            lease,
            memory,
            consumer_stream,
            ResidentModelActivationAuthority::Ordinary,
        )
    }

    fn activate_resident_model_memory_original(
        &mut self,
        lease: &SemanticPublishedLease,
        memory: &SemanticPreparedResidentModelMemory,
        consumer_stream: u64,
        authority: ResidentModelActivationAuthority,
    ) -> Result<(), SemanticTransitionError> {
        let original = &memory.owner;
        self.require_resident_model_memory(lease, original, consumer_stream)?;
        if self.pending
            || self.pending_replay_delivery.is_some()
            || self.pending_training_materialization.is_some()
        {
            return Err(SemanticTransitionError::Poisoned);
        }
        // Exclude only this carrier's pending activation. Independent original
        // commands, reader retirement and cold-report completion still forbid
        // access, including when this exact guard is being resumed.
        if self.has_unresolved_session_native_effect_except_replay_verification()
            || self.replay_verification_pending.load(Ordering::Acquire)
            || self.has_pending_state_restoration_except(None, None)
            || self
                .readers
                .values()
                .any(|reader| reader.retirement_pending || reader.acquisition_pending)
            || self
                .steps
                .values()
                .any(|step| step.consumer_completion.is_some())
        {
            return Err(SemanticTransitionError::Poisoned);
        }
        #[cfg(feature = "semantic-policy")]
        if self.has_pending_actor_refresh_context_except(None)
            || self.has_pending_actor_refresh_construction_except_cancellation(None)
            || self.has_pending_cold_model_work_completion(None)
        {
            return Err(SemanticTransitionError::Poisoned);
        }
        self.graph
            .ensure_not_poisoned()
            .map_err(SemanticTransitionError::Semantic)?;
        if self.resident_model_activation.is_none() {
            self.checked_reader(lease)?;
            // Retain the exact reader, envelope and recording context before
            // constructing commands or events. No caller-owned retry can
            // replace this occurrence, even after a constructor refusal.
            self.resident_model_activation = Some(Arc::new(Mutex::new(ResidentModelActivation {
                memory: Arc::clone(original),
                guard: None,
                authority,
                #[cfg(feature = "semantic-policy")]
                guard_claimed: false,
            })));
        } else {
            let activation = self
                .resident_model_activation
                .as_ref()
                .expect("retained resident model activation")
                .lock()
                .map_err(|_| publication_input_error("resident model activation owner poisoned"))?;
            if !activation.authority.same_context(&authority) {
                return Err(publication_input_error(
                    "resident model activation changed its original admitted recording context",
                ));
            }
        }
        let pending = Arc::clone(
            self.resident_model_activation
                .as_ref()
                .expect("retained resident model activation"),
        );
        let mut activation = pending
            .lock()
            .map_err(|_| publication_input_error("resident model activation owner poisoned"))?;
        if !Arc::ptr_eq(&activation.memory, original) {
            return Err(publication_input_error(
                "resident model activation cannot replace its pending original carrier",
            ));
        }
        if activation.guard.is_none() {
            #[cfg(feature = "semantic-policy")]
            if let ResidentModelActivationAuthority::Cold(identity) = &activation.authority {
                if !activation.guard_claimed {
                    let mut allowance = identity.allowance().lock().map_err(|_| {
                        publication_input_error("original resident guard allowance is poisoned")
                    })?;
                    let ceiling = original.native_guard_ceiling()?;
                    if allowance.has_resident_reads() {
                        allowance.claim_resident_read(
                            original,
                            identity.region_ordinal(),
                            ceiling,
                        )?;
                    } else {
                        allowance.claim_source_resident_guard(original, ceiling)?;
                    }
                    drop(allowance);
                    activation.guard_claimed = true;
                }
            }
            let mut guard =
                self.stage_publication_content_guard(lease, &original.keys, consumer_stream)?;
            #[cfg(feature = "semantic-policy")]
            if let ResidentModelActivationAuthority::Cold(identity) = &activation.authority {
                guard.use_cold_work(identity.work());
            }
            activation.guard = Some(guard);
        }
        self.readers
            .get_mut(&lease.token)
            .expect("validated resident model reader")
            .consumer_streams
            .insert(consumer_stream);
        self.steps
            .get_mut(&lease.token)
            .expect("validated resident model step")
            .consumer_streams
            .insert(consumer_stream);
        for export in &activation.memory.exports {
            export.arm(&self.stream, consumer_stream)?;
        }
        activation
            .guard
            .as_mut()
            .expect("retained resident model publication guard")
            .complete(self, lease)?;
        drop(activation);
        self.resident_model_activation = None;
        Ok(())
    }

    /// Inspect only this carrier's retained activation, including uncertainty
    /// before a driver acknowledgement. This grants no new guard submission.
    pub fn resident_model_memory_activation_pending(
        &self,
        lease: &SemanticPublishedLease,
        memory: &SemanticPreparedResidentModelMemory,
        consumer_stream: u64,
    ) -> Result<bool, SemanticTransitionError> {
        self.require_resident_model_memory(lease, &memory.owner, consumer_stream)?;
        let Some(pending) = &self.resident_model_activation else {
            return Ok(false);
        };
        let activation = pending
            .lock()
            .map_err(|_| publication_input_error("resident model activation owner poisoned"))?;
        if !Arc::ptr_eq(&activation.memory, &memory.owner) {
            return Err(publication_input_error(
                "resident model activation cannot replace its pending original carrier",
            ));
        }
        Ok(true)
    }

    fn require_resident_model_memory(
        &self,
        lease: &SemanticPublishedLease,
        original: &ResidentModelMemoryOwner,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        dlpack_consumer_stream(consumer_stream)?;
        self.checked_original_consumer_step(lease)?;
        if !lease.active
            || !Arc::ptr_eq(&original.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&original.issuer, &lease.issuer)
            || original.token != lease.token
            || original.identity != lease.identity
            || original.consumer_stream != consumer_stream
            || publication_abi_bytes(&[original.header]) != publication_abi_bytes(&[lease.header])
            || publication_abi_bytes(&original.directory) != publication_abi_bytes(&lease.directory)
            || self
                .publication
                .as_ref()
                .is_none_or(|storage| !Arc::ptr_eq(storage, &original.storage))
            || self
                .readers
                .get(&lease.token)
                .is_none_or(|reader| !Arc::ptr_eq(&reader.aliases, &original.reader_pin))
        {
            return Err(publication_input_error(
                "resident model activation changed its original reader, envelope or stream",
            ));
        }
        Ok(())
    }
}
