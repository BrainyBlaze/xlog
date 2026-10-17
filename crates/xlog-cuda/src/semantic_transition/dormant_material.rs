use super::*;

/// The canonical material codec has one explicit non-public mode for a task
/// whose publication has never been initialized. Its model fields refer to the
/// authentic SOURCE checkpoint; they never encode another parameter payload.
pub(in crate::semantic_transition) enum PublicationMaterialMode<'a, B> {
    Published,
    Dormant(&'a DormantModelMaterialSource<B>),
}

pub(in crate::semantic_transition) struct DormantModelMaterialSource<B> {
    pub(in crate::semantic_transition) model_memory: ModelMemoryGeometry,
    pub(in crate::semantic_transition) layouts: BTreeMap<(u64, u64), SemanticTensorLayout>,
    pub(in crate::semantic_transition) model_allocations: Vec<B>,
    pub(in crate::semantic_transition) generation: u64,
    pub(in crate::semantic_transition) geometry_digest: Identity256,
    pub(in crate::semantic_transition) native_digest: Identity256,
    pub(in crate::semantic_transition) model_contract_layout: SemanticModelContractLayout,
    pub(in crate::semantic_transition) model_contract_record: B,
    // Shared byte owners and their metadata drop before this pool receipt.
    _reservation: Option<SemanticReplayMaterialReservation>,
}

impl DormantModelMaterialSource<PublicationMaterialBytes> {
    #[cfg(feature = "semantic-policy")]
    pub(super) fn authenticate(
        source: &SemanticTransitionSession,
        parent: &SemanticPublishedLease,
        loaded: &mut SemanticReplayLoadedMaterial,
    ) -> Result<Self, SemanticTransitionError> {
        let (native_digest, _) = source.verify_checkpoint_loaded_source_material(parent, loaded)?;
        let header = parent.model_context_header()?;
        let material = loaded.complete()?;
        if material.bank.header.model_generation != header.model_generation
            || material.bank.header.model_geometry_digest != header.model_geometry_digest
            || material.model_memory.digest(&material.layouts)? != header.model_geometry_digest
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let budget = source
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let mut bytes = size_of::<Self>() + 2 * size_of::<usize>();
        for (count, width) in [
            (
                material.model_allocations.len(),
                size_of::<PublicationMaterialBytes>(),
            ),
            (
                material.model_memory.allocation_bytes.len(),
                size_of::<u64>(),
            ),
            (
                material.model_memory.storages.len(),
                size_of::<SemanticModelStorage>(),
            ),
            (
                material.model_memory.views.len(),
                size_of::<SemanticModelView>(),
            ),
            (
                material.layouts.len(),
                OwnedPublicationMaterial::tree_node_bound(
                    size_of::<(u64, u64)>(),
                    size_of::<SemanticTensorLayout>(),
                )?,
            ),
        ] {
            bytes = bytes
                .checked_add(
                    count
                        .checked_mul(width)
                        .ok_or(SemanticTransitionError::GenerationExhausted)?,
                )
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let bytes =
            u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let reservation = budget.reserve_auxiliary(bytes)?;
        budget.require_host_allocation(bytes)?;
        let model_contract_record = material
            .ranges
            .iter()
            .find(|item| (item.range.role, item.range.index) == (44, 0))
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .bytes
            .clone();
        Ok(Self {
            model_memory: material.model_memory.clone(),
            layouts: material.layouts.clone(),
            model_allocations: material.model_allocations.clone(),
            generation: header.model_generation,
            geometry_digest: header.model_geometry_digest,
            native_digest,
            model_contract_layout: material.contract.model_contract_layout,
            model_contract_record,
            _reservation: Some(reservation),
        })
    }
}

/// This owner cannot be passed to ordinary publication or restore APIs. Only
/// the continuation's canonical Current-model restore consumes its raw recipe.
#[cfg(feature = "semantic-policy")]
pub(super) struct DormantPublicationMaterial {
    pub(super) material: OwnedPublicationMaterial,
    pub(super) source: Arc<DormantModelMaterialSource<PublicationMaterialBytes>>,
}

#[cfg(feature = "semantic-policy")]
impl DormantPublicationMaterial {
    pub(super) fn from_readback(
        material: OwnedPublicationMaterial,
        source: Arc<DormantModelMaterialSource<PublicationMaterialBytes>>,
    ) -> Result<Self, SemanticTransitionError> {
        material.validate_mode(&PublicationMaterialMode::Dormant(&source))?;
        Ok(Self { material, source })
    }

    pub(super) fn decode(
        loaded: SemanticReplayLoadedMaterial,
        source: Arc<DormantModelMaterialSource<PublicationMaterialBytes>>,
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<Self, SemanticTransitionError> {
        loaded.require_budget(budget)?;
        let material = OwnedPublicationMaterial::decode_buffer_mode(
            loaded.bytes,
            &PublicationMaterialMode::Dormant(&source),
        )?;
        Ok(Self { material, source })
    }

    pub(super) fn encode(
        &self,
        reservation: &Arc<SemanticReplayMaterialReservation>,
    ) -> Result<SemanticReplayLoadedMaterial, SemanticTransitionError> {
        let mode = PublicationMaterialMode::Dormant(&self.source);
        let mut length = 0usize;
        self.material.write_encoded_mode(&mode, |part| {
            length = length
                .checked_add(part.len())
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            Ok(())
        })?;
        if u64::try_from(length).map_err(|_| SemanticTransitionError::GenerationExhausted)?
            > reservation.charged_bytes
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        reservation.budget.require_host_allocation(length as u64)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|error| runtime_error("dormant context encoding custody", error))?;
        self.material.write_encoded_mode(&mode, |part| {
            if bytes
                .len()
                .checked_add(part.len())
                .is_none_or(|end| end > length)
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            bytes.extend_from_slice(part);
            Ok(())
        })?;
        if bytes.len() != length {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(SemanticReplayLoadedMaterial {
            bytes: PublicationMaterialBytes::readback(bytes, Some(Arc::clone(reservation))),
            decoded: None,
        })
    }
}
