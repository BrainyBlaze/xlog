//! Original model ownership across two frozen task contexts.

use super::*;
mod prefill_restoration;
mod restoration;
pub use prefill_restoration::{
    SemanticModelContextPrefillRestoreDeclaration, SemanticRestoredModelContextPrefill,
};
pub(super) use restoration::OriginalModelContextAcknowledgement;
pub use restoration::SemanticModelContextCheckpointAttachment;

/// One original preparation and publication transfer. This capability does not
/// authorize executing its future instruction before the source is completed.
#[derive(Clone)]
pub struct SemanticModelContextContinuation {
    pub(super) inner: Arc<ModelContextContinuationOwner>,
}

pub(super) struct ModelContextContinuationOwner {
    pub(super) source_issuer: Arc<()>,
    pub(super) target_issuer: Arc<()>,
    pub(super) source_publication: Arc<PublicationStorage>,
    _source_provider: Arc<CudaKernelProvider>,
    pub(super) models: Vec<Arc<ModelGenerationOwner>>,
    instruction: Arc<[u8]>,
    source_header: PublicationHeader,
    pub(super) program: Mutex<actor_refresh_program::OriginalProgramContextContinuation>,
    original: Mutex<OriginalModelContextContinuation>,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct PreparedModelContextContinuationDevice {
    abi: u64,
    status: u64,
    source_control: u64,
    source_lease: u64,
    target_control: u64,
    minimum_generation: u64,
    stream_serial: u64,
    family_id: u64,
    proposal: u64,
    source_instance: Identity256,
    target_instance: Identity256,
    model_geometry_digest: Identity256,
    model_numerical_digest: Identity256,
    acquired_generation: u64,
    source_word: u64,
    target_word: u64,
    training_cursor: u64,
    training_rng: [u64; 4],
}

// SAFETY: the original CUDA command consumes the matching 33-word C structure.
unsafe impl DeviceRepr for PreparedModelContextContinuationDevice {}

struct DormantModelContextParent {
    storage: Arc<PublicationStorage>,
    bank: PublicationBank,
    rng: SemanticRngBinding,
    terminals: Vec<u64>,
    role_counts: [u64; PUBLICATION_ROLE_COUNT],
    prefill: Option<InitialPrefillStage>,
    proof: TrackedCudaSlice<PreparedModelContextContinuationDevice>,
    initialized: bool,
}

struct OriginalModelContextContinuation {
    parent: Option<DormantModelContextParent>,
    command: OriginalNativeCommand,
    source_parent: Option<(u64, SemanticPublishedIdentity)>,
    pub(super) source_boundary: Option<PublicationHeader>,
    activation_result: Option<SemanticColdModelWorkResult>,
    activation_work: Option<SemanticColdModelWork>,
    proof_read: Option<PublicationRead<PreparedModelContextContinuationDevice>>,
    header_read: Option<PublicationRead<PublicationHeader>>,
    observed_proof: Option<PreparedModelContextContinuationDevice>,
    observed_header: Option<PublicationHeader>,
    activation_rng: Option<SemanticRngBinding>,
    native_work: Option<DeviceMemoryView<u64>>,
    native_claimed: bool,
    prefix_guard: Option<OriginalNativeCommand>,
    prefix_guard_stream: Option<u64>,
    source_work_attached: bool,
    checkpoint_restore: Option<Arc<Mutex<restoration::OriginalDormantContextRestore>>>,
    source_alias: Option<OriginalModelContextSourceAlias>,
    pub(super) source_acknowledgements:
        Vec<Arc<Mutex<restoration::OriginalModelContextAcknowledgement>>>,
    initial_prefill_scope: Option<Arc<()>>,
    activating: bool,
    activated: Option<SemanticPublishedIdentity>,
    refused: bool,
    poisoned: bool,
}

pub(super) struct OriginalModelContextSourceAlias {
    tensors: Vec<PreparedSemanticTensor>,
    allocations: Vec<PreparedSemanticTensor>,
    consumer_stream: u64,
    command: OriginalNativeCommand,
    _reservation: SemanticReplayMaterialReservation,
}

impl SemanticModelContextContinuation {
    pub fn cold_budget(&self) -> Result<[u64; 3], SemanticTransitionError> {
        self.inner.cold_budget()
    }

    pub fn activation_cold_result(
        &self,
    ) -> Result<Option<SemanticColdModelWorkResult>, SemanticTransitionError> {
        Ok(self
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .activation_result)
    }
    pub fn segment_transitions(
        &self,
    ) -> Result<Vec<SemanticTransitionKind>, SemanticTransitionError> {
        Ok(self
            .inner
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .transitions()
            .collect())
    }

    pub fn segment_cold_capacity(
        &self,
    ) -> Result<SemanticSegmentColdCapacity, SemanticTransitionError> {
        Ok(self
            .inner
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .cold_capacity())
    }

    pub fn original_instruction_bytes(&self) -> Result<Vec<u8>, SemanticTransitionError> {
        Ok(self
            .inner
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .instruction()
            .to_vec())
    }
}

impl ModelContextContinuationOwner {
    pub(super) fn cold_budget(&self) -> Result<[u64; 3], SemanticTransitionError> {
        Ok(self
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .cold_budget())
    }

    pub(super) fn original_program_instruction(
        &self,
    ) -> Result<Arc<[u8]>, SemanticTransitionError> {
        Ok(Arc::clone(&self.instruction))
    }

    pub(super) fn require_imported_target(
        &self,
        target: &SemanticTransitionSession,
    ) -> Result<(), SemanticTransitionError> {
        target.ensure_rebindable()?;
        let original = self
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !Arc::ptr_eq(&self.target_issuer, &target.publication_issuer)
            || target
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !std::ptr::eq(Arc::as_ptr(owner), self))
            || target.task_evaluation_identity().is_none()
            || target.publication.is_some()
            || target.initial_prefill.is_some()
            || original.initial_prefill_scope.is_some()
            || original.parent.is_some()
            || original.activating
            || original.refused
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub(super) fn require_initial_prefill_preparation(
        &self,
        target: &SemanticTransitionSession,
        scope: &Arc<()>,
    ) -> Result<(), SemanticTransitionError> {
        let original = self
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !Arc::ptr_eq(&self.target_issuer, &target.publication_issuer)
            || original
                .initial_prefill_scope
                .as_ref()
                .is_none_or(|issued| !Arc::ptr_eq(issued, scope))
            || target
                .initial_prefill
                .as_ref()
                .is_none_or(|stage| !Arc::ptr_eq(&stage.scope, scope))
            || original.activating
            || original.refused
            || original.parent.is_some()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub(super) fn require_parent_preparation(
        &self,
        target: &SemanticTransitionSession,
    ) -> Result<(), SemanticTransitionError> {
        let original = self
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !Arc::ptr_eq(&target.publication_issuer, &self.target_issuer)
            || target
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !std::ptr::eq(Arc::as_ptr(actual), self))
            || original.activating
            || original.refused
            || original.parent.as_ref().is_none_or(|parent| {
                parent.initialized
                    || target
                        .publication
                        .as_ref()
                        .is_none_or(|storage| !Arc::ptr_eq(storage, &parent.storage))
            })
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub(super) fn require_frozen_preparation(
        &self,
        target: &SemanticTransitionSession,
        scope: &Arc<()>,
    ) -> Result<(), SemanticTransitionError> {
        self.require_dormant_parent(target)?;
        let program = self
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        target.require_frozen_original_program_scope(&program)?;
        let build = target
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&build.scope, scope)
            || build.program_admission.is_some()
            || build.submitted
            || build.completed
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub(super) fn require_activation_report(
        &self,
        source: &SemanticTransitionSession,
        parent: &SemanticPublishedLease,
        target: &SemanticTransitionSession,
        scope: &Arc<()>,
    ) -> Result<(), SemanticTransitionError> {
        {
            let original = self
                .original
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?;
            if original.activated.is_some() {
                if !Arc::ptr_eq(&self.source_issuer, &source.publication_issuer)
                    || !Arc::ptr_eq(&self.target_issuer, &target.publication_issuer)
                    || original.source_parent != Some((parent.token, parent.identity))
                    || original.refused
                    || target
                        .model_context_continuation
                        .as_ref()
                        .is_none_or(|owner| !std::ptr::eq(Arc::as_ptr(owner), self))
                    || target
                        .prepared_segment
                        .as_ref()
                        .is_none_or(|build| !Arc::ptr_eq(&build.scope, scope) || build.submitted)
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                return Ok(());
            }
        }
        if !Arc::ptr_eq(&self.source_issuer, &source.publication_issuer)
            || !Arc::ptr_eq(&self.target_issuer, &target.publication_issuer)
            || source
                .publication
                .as_ref()
                .is_none_or(|storage| !Arc::ptr_eq(storage, &self.source_publication))
            || source.pending_replay_delivery.is_some()
            || source.pending_training_materialization.is_some()
            || source.prepared_segment.is_some()
            || source.readers.len() != 1
            || !parent.active
            || !source.readers.contains_key(&parent.token)
            || source
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !std::ptr::eq(Arc::as_ptr(owner), self))
            || target
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !std::ptr::eq(Arc::as_ptr(owner), self))
            || target
                .prepared_segment
                .as_ref()
                .is_none_or(|build| !Arc::ptr_eq(&build.scope, scope))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let original = self
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if let Some(issued) = original.source_parent {
            if issued != (parent.token, parent.identity) || !original.activating || original.refused
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            return Ok(());
        }
        drop(original);
        self.require_dormant_parent(target)?;
        let program = self
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        target.validate_transfer_original_program_context_continuation(source, &program, parent)
    }
    pub(super) fn pending(&self) -> bool {
        self.original.lock().map_or(true, |original| {
            original.activating && original.activated.is_none() && !original.refused
        })
    }

    pub(super) fn require_activation_scope(
        &self,
        target: &SemanticTransitionSession,
        scope: &Arc<()>,
    ) -> Result<(), SemanticTransitionError> {
        let original = self
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !Arc::ptr_eq(&self.target_issuer, &target.publication_issuer)
            || target
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !std::ptr::eq(Arc::as_ptr(owner), self))
            || original.source_parent.is_none()
            || !original.activating
            || original.refused
            || target
                .prepared_segment
                .as_ref()
                .is_none_or(|build| !Arc::ptr_eq(scope, &build.scope) || build.submitted)
            || original.parent.as_ref().is_none_or(|parent| {
                !parent.initialized
                    || target
                        .publication
                        .as_ref()
                        .is_none_or(|storage| !Arc::ptr_eq(storage, &parent.storage))
            })
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub(super) fn require_initial_content_scope(
        &self,
        target: &SemanticTransitionSession,
        initial_scope: &Arc<()>,
    ) -> Result<bool, SemanticTransitionError> {
        let original = self
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let activating = original.activating;
        if original
            .initial_prefill_scope
            .as_ref()
            .is_none_or(|scope| !Arc::ptr_eq(scope, initial_scope))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        drop(original);
        if activating {
            self.require_activation_scope(
                target,
                &target
                    .prepared_segment
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?
                    .scope,
            )?;
        } else {
            self.require_initial_prefill_preparation(target, initial_scope)?;
        }
        Ok(activating)
    }

    pub(super) fn require_parent_binding(
        &self,
        target: &SemanticTransitionSession,
        parent: &SemanticParentBinding,
        memory: &ModelMemoryGeometry,
        tensors: &[PreparedSemanticTensor],
        allocations: &[PreparedSemanticTensor],
    ) -> Result<(), SemanticTransitionError> {
        let original = self
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !Arc::ptr_eq(&target.publication_issuer, &self.target_issuer)
            || target
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !std::ptr::eq(Arc::as_ptr(owner), self))
            || original.parent.is_some()
            || original.activating
            || original.refused
            || target.publication.is_some()
            || *memory != self.source_publication.model_memory
            || parent.model_generation != self.source_header.model_generation
            || parent.neural_generation != self.source_header.neural_generation
            || parent.training_cursor != self.source_header.training_cursor
            || parent.training_rng != self.source_header.training_rng
            || parent.rng != self.source_header.rng_binding()?
            || parent.prefix_capacity != self.source_publication.contract_value.prefix_capacity
            || parent.feedback_capacity != self.source_publication.contract_value.feedback_capacity
            || parent.max_position != self.source_publication.contract_value.max_position
            || parent.pad_token != self.source_publication.contract_value.pad_token
            || parent.model_contract_layout
                != self.source_publication.contract_value.model_contract_layout
            || parent
                .records
                .iter()
                .filter(|record| record.role == SemanticStateRole::ModelContract)
                .count()
                != 1
            || !parent.records.iter().any(|record| {
                record.role == SemanticStateRole::ModelContract
                    && record.index == 0
                    && record.bytes.as_slice()
                        == self.source_publication.model_contract_record.as_ref()
            })
            || allocations.len() != self.models.len()
            || allocations
                .iter()
                .zip(&self.models)
                .any(|(actual, expected)| {
                    actual.data != expected.allocation.device_ptr_value()
                        || actual
                            .source
                            .as_ref()
                            .is_none_or(|source| source.len() != expected.allocation.len())
                })
            || tensors
                .iter()
                .filter(|tensor| matches!(tensor.layout.role, 18..=25))
                .any(|tensor| {
                    self.source_publication
                        .layouts
                        .get(&(tensor.layout.role, tensor.layout.index))
                        != Some(&tensor.layout)
                })
            || tensors
                .iter()
                .filter(|tensor| matches!(tensor.layout.role, 18..=25))
                .count()
                != self
                    .source_publication
                    .layouts
                    .keys()
                    .filter(|(role, _)| matches!(*role, 18..=25))
                    .count()
        {
            return Err(publication_input_error(
                "future context changed the original model owners or complete publication contract",
            ));
        }
        Ok(())
    }

    pub(super) fn retain_dormant_parent(
        &self,
        target: &mut SemanticTransitionSession,
        bank: PublicationBank,
        rng: SemanticRngBinding,
        terminals: &[u64],
        role_counts: &[u64; PUBLICATION_ROLE_COUNT],
    ) -> Result<(), SemanticTransitionError> {
        let storage = Arc::clone(
            target
                .publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let mut original = self
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original.parent.is_some() || original.activating || original.refused {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let proposal = self
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .expected_source_proposal();
        let input = PreparedModelContextContinuationDevice {
            abi: 1,
            status: 1,
            source_control: self.source_publication.control.device_ptr_value(),
            target_control: storage.control.device_ptr_value(),
            minimum_generation: self.source_header.model_generation,
            stream_serial: self.source_header.stream_serial,
            family_id: self.source_header.family_id,
            proposal,
            source_instance: self.source_publication.instance,
            target_instance: storage.instance,
            model_geometry_digest: self.source_header.model_geometry_digest,
            model_numerical_digest: self.source_header.model_numerical_digest,
            ..PreparedModelContextContinuationDevice::default()
        };
        let proof = allocate_publication(&target.provider, 1)?;
        let proof_write = state_restoration::OriginalDeviceWrite::new(
            &target.stream,
            &[input],
            proof.view(),
            true,
        )?;
        target.publication_preparation = Some(target.stage_publication_preparation(
            bank,
            terminals,
            role_counts,
            rng,
            None,
            true,
            None,
            Some(proof_write),
        )?);
        // The one original write and every destination are retained before DMA.
        original.parent = Some(DormantModelContextParent {
            storage,
            bank,
            rng,
            terminals: terminals.to_vec(),
            role_counts: *role_counts,
            prefill: target.initial_prefill.take(),
            proof,
            initialized: false,
        });
        Ok(())
    }

    fn resolve_dormant_parent(
        target: &mut SemanticTransitionSession,
        original: &mut OriginalModelContextContinuation,
    ) -> Result<(), SemanticTransitionError> {
        let parent = original
            .parent
            .as_mut()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !parent.initialized {
            target.prepare_publication_storage(
                parent.bank,
                &parent.terminals,
                &parent.role_counts,
                parent.rng,
                None,
                true,
            )?;
            parent.initialized = true;
        }
        Ok(())
    }

    pub(super) fn require_dormant_parent(
        &self,
        target: &SemanticTransitionSession,
    ) -> Result<(), SemanticTransitionError> {
        let original = self
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !Arc::ptr_eq(&target.publication_issuer, &self.target_issuer)
            || target
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !std::ptr::eq(Arc::as_ptr(owner), self))
            || original.activating
            || original.refused
            || original.activated.is_some()
            || original.parent.as_ref().is_none_or(|parent| {
                !parent.initialized
                    || target
                        .publication
                        .as_ref()
                        .is_none_or(|actual| !Arc::ptr_eq(actual, &parent.storage))
            })
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }
}

impl SemanticTransitionSession {
    pub fn model_context_continuation_pending(
        &self,
        continuation: &SemanticModelContextContinuation,
    ) -> Result<bool, SemanticTransitionError> {
        let owner = &continuation.inner;
        if !Arc::ptr_eq(&self.publication_issuer, &owner.target_issuer) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let pending = original.command.completion_pending()
            || original
                .prefix_guard
                .as_ref()
                .is_some_and(OriginalNativeCommand::completion_pending)
            || original
                .source_alias
                .as_ref()
                .is_some_and(|alias| alias.command.completion_pending())
            || original
                .proof_read
                .as_ref()
                .is_some_and(PublicationRead::completion_pending)
            || original
                .header_read
                .as_ref()
                .is_some_and(PublicationRead::completion_pending);
        let work = original.activation_work.clone();
        let checkpoint_restore = original.checkpoint_restore.clone();
        drop(original);
        if pending {
            return Ok(true);
        }
        if self.model_context_material_completion_pending(continuation)?
            || self.state_material_restore_completion_pending()?
        {
            return Ok(true);
        }
        if let Some(restoration) = checkpoint_restore {
            if restoration
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?
                .completion_pending()?
            {
                return Ok(true);
            }
        }
        if let Some(stage) = &self.initial_prefill {
            if stage
                .token_write
                .as_ref()
                .is_some_and(state_restoration::OriginalDeviceWrite::completion_pending)
                || stage
                    .input_guard
                    .as_ref()
                    .is_some_and(OriginalNativeCommand::completion_pending)
            {
                return Ok(true);
            }
            for content in stage
                .model_content
                .iter()
                .chain(&stage.output_content)
                .chain(&stage.captured_content)
            {
                let batch = content
                    .original_cold
                    .lock()
                    .map_err(|_| SemanticTransitionError::Poisoned)?;
                if batch.as_ref().is_some_and(|batch| {
                    batch
                        .commands
                        .iter()
                        .any(OriginalNativeCommand::completion_pending)
                }) {
                    return Ok(true);
                }
            }
        }
        if self
            .publication_preparation
            .as_ref()
            .is_some_and(|preparer| preparer.completion_pending())
        {
            return Ok(true);
        }
        if let Some(work) = work {
            return self.model_context_activation_completion_pending(continuation, &work);
        }
        Ok(false)
    }

    /// Authenticate the held publication without admitting or executing its
    /// next instruction. Checkpoint export uses this narrow stable-owner gate.
    pub fn require_activated_model_context_parent(
        &self,
        continuation: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        let owner = &continuation.inner;
        self.checked_reader(parent)?;
        let original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !Arc::ptr_eq(&self.publication_issuer, &owner.target_issuer)
            || original.activated != Some(parent.identity)
            || original.refused
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
            || self.publication.as_ref().is_none_or(|storage| {
                original
                    .parent
                    .as_ref()
                    .is_none_or(|saved| !Arc::ptr_eq(storage, &saved.storage))
            })
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub fn require_activated_model_context_checkpoint(
        &self,
        continuation: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.require_activated_model_context_parent(continuation, parent)?;
        let work = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .activation_work
            .clone()
            .ok_or(SemanticTransitionError::NotBound)?;
        self.completed_model_context_cold_result(continuation, &work)?;
        Ok(())
    }

    fn require_model_context_numerical_contract(
        &self,
        continuation: &SemanticModelContextContinuation,
    ) -> Result<(), SemanticTransitionError> {
        let owner = &continuation.inner;
        let storage = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&self.publication_issuer, &owner.target_issuer)
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
            || storage.model_memory != owner.source_publication.model_memory
            || storage.model_contract_record.as_ref()
                != owner.source_publication.model_contract_record.as_ref()
            || storage.contract_value.model_contract_layout
                != owner
                    .source_publication
                    .contract_value
                    .model_contract_layout
            || self.model_owners.len() != owner.models.len()
            || self
                .model_owners
                .values()
                .zip(&owner.models)
                .any(|(actual, saved)| !Arc::ptr_eq(actual, saved))
            || storage
                .layouts
                .iter()
                .filter(|((role, _), _)| matches!(*role, 18..=25))
                .any(|(key, layout)| owner.source_publication.layouts.get(key) != Some(layout))
            || storage
                .layouts
                .keys()
                .filter(|(role, _)| matches!(*role, 18..=25))
                .count()
                != owner
                    .source_publication
                    .layouts
                    .keys()
                    .filter(|(role, _)| matches!(*role, 18..=25))
                    .count()
        {
            return Err(publication_input_error(
                "model basis changed the complete original contract or allocation owners",
            ));
        }
        for slots in storage.model_slots_for_directory(&storage.bank_templates[0])? {
            let selected = storage.allocations[slots[0]]
                .model_owner()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if !owner
                .models
                .iter()
                .any(|saved| Arc::ptr_eq(saved, &selected))
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        }
        Ok(())
    }

    pub fn require_model_context_prepared_model_basis(
        &self,
        continuation: &SemanticModelContextContinuation,
        step: &SemanticPreparedStep,
    ) -> Result<(), SemanticTransitionError> {
        let activating = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .activating;
        if activating {
            continuation
                .inner
                .require_activation_scope(self, &step.scope)?;
        } else {
            continuation
                .inner
                .require_frozen_preparation(self, &step.scope)?;
        }
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&step.issuer, &self.publication_issuer)
            || !build.tokens.contains(&step.token)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.require_model_context_numerical_contract(continuation)
    }

    pub fn require_model_context_published_model_basis(
        &self,
        continuation: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.require_activated_model_context_parent(continuation, parent)?;
        self.require_model_context_numerical_contract(continuation)?;
        let header = parent.model_context_header()?;
        let source = continuation.inner.source_header;
        if header.model_generation != source.model_generation
            || header.neural_generation != source.neural_generation
            || header.model_geometry_digest != source.model_geometry_digest
            || header.model_numerical_digest != source.model_numerical_digest
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub fn require_model_context_initial_prefill_basis(
        &self,
        continuation: &SemanticModelContextContinuation,
        initial: &SemanticInitialPrefillLease,
        parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.require_model_context_published_model_basis(continuation, parent)?;
        let original = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let retained = self
            .retained_initial_prefill
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&initial.issuer, &self.publication_issuer)
            || original
                .initial_prefill_scope
                .as_ref()
                .is_none_or(|scope| !Arc::ptr_eq(scope, &initial.scope))
            || !Arc::ptr_eq(&retained.scope, &initial.scope)
            || retained.task_epoch != self.task_epoch
            || self.task_evaluation_identity() != Some(retained.task_identity)
            || retained.model_generation != parent.header.model_generation
        {
            return Err(publication_input_error("initial prefill basis changed its original stage, task or complete neural generation"));
        }
        Ok(())
    }

    fn finish_original_model_context_activation_report(
        &mut self,
        source: &mut Self,
        parent: &SemanticPublishedLease,
        continuation: &SemanticModelContextContinuation,
        work: &SemanticColdModelWork,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        let owner = &continuation.inner;
        let original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let proof = original
            .observed_proof
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if proof.abi != 1
            || proof.status == 0
            || !original.command.submitted
            || !original.command.completed
            || original.activated.is_some()
            || original.refused
            || original
                .activation_work
                .as_ref()
                .is_none_or(|actual| !actual.same_invocation(work))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let attached = original.source_work_attached;
        drop(original);
        owner.require_activation_report(
            source,
            parent,
            self,
            &self
                .prepared_segment
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .scope,
        )?;
        if attached {
            source.detach_model_context_activation_work(parent, continuation, self, work)?;
            owner
                .original
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?
                .source_work_attached = false;
        }
        let result = self.finish_model_context_activation_work(
            source,
            parent,
            continuation,
            work,
            &[],
            SemanticColdModelWorkDisposition::KnownRefusal,
        )?;
        let mut original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        original.activation_result = Some(result);
        original.refused = true;
        Ok(result)
    }

    /// Retain the single real activation report before any alias handoff or
    /// publication command. Its original finite instruction budget never resets.
    fn prepare_original_model_context_activation(
        &mut self,
        source: &Self,
        continuation: &SemanticModelContextContinuation,
        current: &SemanticPublishedLease,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        let owner = &continuation.inner;
        let scope = &self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .scope;
        owner.require_activation_report(source, current, self, scope)?;
        let retained = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .activation_work
            .clone();
        let ceiling =
            native_work_bound::model_context_activation_native_work_ceiling(source, current, self)?;
        let activation_rng = current.model_context_header()?.rng_binding()?;
        let preparation = if let Some(work) = retained {
            Ok(work)
        } else {
            self.prepare_model_context_activation_work(source, current, continuation, ceiling)
        };
        // Installation can precede a later allocation error. Recover that
        // actual native handle, never install a second report on a retry.
        let (work, preparation_error) = match preparation {
            Ok(work) => (work, None),
            Err(error) => match self.model_context_activation_handle(continuation)? {
                Some(work) => (work, Some(error)),
                None => return Err(error),
            },
        };
        let (native_work, _) = self.model_context_activation_native_work(continuation, &work)?;
        let mut original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original
            .activation_work
            .as_ref()
            .is_some_and(|actual| !actual.same_invocation(&work))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        original.activation_work = Some(work.clone());
        original.activation_rng = Some(activation_rng);
        original.native_work = Some(native_work);
        original.source_parent = Some((current.token, current.identity));
        original.activating = true;
        let parent = original
            .parent
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let proof_view = parent.proof.view();
        // SAFETY: the fixed header is the first aligned member of the retained
        // original bank. No device value is read while staging its one copy.
        let header_view = unsafe {
            parent.storage.banks[0]
                .view()
                .cast::<u8>()
                .expect("byte-aligned bank ABI")
                .slice(..size_of::<PublicationHeader>())
                .cast::<PublicationHeader>()
                .expect("aligned fixed header ABI")
        };
        if original.proof_read.is_none() {
            original.proof_read = Some(self.stage_publication_read_with_owner(
                proof_view,
                Some(Arc::clone(&owner._reservation)),
            )?);
        }
        if original.header_read.is_none() {
            original.header_read = Some(self.stage_publication_read_with_owner(
                header_view,
                Some(Arc::clone(&owner._reservation)),
            )?);
        }
        if let Some(error) = preparation_error {
            return Err(error);
        }
        Ok(work)
    }

    pub fn prepare_model_context_activation(
        &mut self,
        source: &Self,
        continuation: &SemanticModelContextContinuation,
        current: &SemanticPublishedLease,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.prepare_original_model_context_activation(source, continuation, current)
    }

    /// Metadata comes from the retained original TARGET destinations. This
    /// exposes no tensor, device read, publication lease or model authority.
    pub fn model_context_activation_prefix_geometry(
        &self,
        continuation: &SemanticModelContextContinuation,
    ) -> Result<[(Vec<i64>, Vec<i64>, (u8, u8)); 2], SemanticTransitionError> {
        let scope = &self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .scope;
        continuation.inner.require_activation_scope(self, scope)?;
        let storage = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let range = storage.bank_templates[0]
            .iter()
            .find(|range| {
                range.role == SemanticStateRole::PrefixIdentity as u64 && range.index == 0
            })
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if range.length_bytes != 32 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let (_, shape, strides) = PublicationBankField::PrefixExtent.layout();
        Ok([(vec![32], vec![1], (1, 8)), (shape, strides, (1, 64))])
    }

    /// Guard the selected original prefix once, then export only its two
    /// actual native views. The original report remains open for their reads.
    pub fn model_context_activation_prefix_views(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        consumer_stream: u64,
    ) -> Result<(DlpackManagedTensor, DlpackManagedTensor, Identity256), SemanticTransitionError>
    {
        self.require_activated_model_context_parent(continuation, parent)?;
        self.model_context_activation_prefix_geometry(continuation)?;
        dlpack_consumer_stream(consumer_stream)?;
        let owner = &continuation.inner;
        let mut original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original
            .prefix_guard_stream
            .is_some_and(|stream| stream != consumer_stream)
        {
            return Err(publication_input_error(
                "activation prefix cannot replace its original guarded consumer stream",
            ));
        }
        let work = original
            .activation_work
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        self.model_context_activation_native_work(continuation, work)?;
        if original.prefix_guard.is_none() {
            original.prefix_guard = Some(OriginalNativeCommand::new(&self.domain)?);
            original.prefix_guard_stream = Some(consumer_stream);
        }
        let storage = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let execute = self
            .provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_publication_content_guard",
            )
            .ok_or_else(|| {
                runtime_error("kernel lookup", "publication prefix guard unavailable")
            })?;
        let reader = self
            .readers
            .get(&parent.token)
            .ok_or(SemanticTransitionError::NotBound)?
            .device
            .view();
        let native_work = original
            .native_work
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .clone();
        let OriginalModelContextContinuation {
            prefix_guard,
            poisoned,
            ..
        } = &mut *original;
        storage.enqueue_content_guard(
            &self.domain,
            poisoned,
            reader,
            &execute,
            SemanticStateRole::PrefixIdentity as u64,
            0,
            SemanticTensorLayout::default(),
            Some(&native_work),
            prefix_guard.as_mut(),
        )?;
        drop(original);
        let prefix = self.published_record(
            parent,
            SemanticStateRole::PrefixIdentity,
            0,
            consumer_stream,
        )?;
        let extent = self.export_publication_bank_field(
            parent,
            PublicationBankField::PrefixExtent,
            consumer_stream,
        )?;
        Ok((prefix, extent, parent.header.descriptor_digest))
    }

    pub fn finish_model_context_activation_callback(
        &mut self,
        source: &mut Self,
        continuation: &SemanticModelContextContinuation,
        current: &SemanticPublishedLease,
        streams: &[u64],
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        let original = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if let Some(result) = original.activation_result {
            return Ok(result);
        }
        if original.activated.is_none() || original.refused {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let work = original
            .activation_work
            .clone()
            .ok_or(SemanticTransitionError::NotBound)?;
        let attached = original.source_work_attached;
        drop(original);
        if attached {
            source.detach_model_context_activation_work(current, continuation, self, &work)?;
            continuation
                .inner
                .original
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?
                .source_work_attached = false;
        }
        let result = self.finish_model_context_activation_work(
            source,
            current,
            continuation,
            &work,
            streams,
            SemanticColdModelWorkDisposition::Complete,
        )?;
        continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .activation_result = Some(result);
        Ok(result)
    }

    /// Submit or resolve exactly one native model-context publication. A late
    /// completion keeps the original command, report, reads and all owners.
    pub fn activate_model_context_continuation(
        &mut self,
        source: &mut Self,
        continuation: &SemanticModelContextContinuation,
        current: &SemanticPublishedLease,
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        let owner = Arc::clone(&continuation.inner);
        let retained_refusal = {
            let original = owner
                .original
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?;
            if let Some(identity) = original.activated {
                if !Arc::ptr_eq(&self.publication_issuer, &owner.target_issuer)
                    || original.source_parent != Some((current.token, current.identity))
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                return Ok(identity);
            }
            if original.refused {
                if !Arc::ptr_eq(&source.publication_issuer, &owner.source_issuer)
                    || !Arc::ptr_eq(&self.publication_issuer, &owner.target_issuer)
                    || original.source_parent != Some((current.token, current.identity))
                    || original.activation_result.is_none()
                    || source
                        .model_context_continuation
                        .as_ref()
                        .is_none_or(|actual| !Arc::ptr_eq(actual, &owner))
                    || self
                        .model_context_continuation
                        .as_ref()
                        .is_none_or(|actual| !Arc::ptr_eq(actual, &owner))
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                let proof = original
                    .observed_proof
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if proof.abi != 1 || proof.status == 0 {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                drop(original);
                source.checked_reader(current)?;
                return Err(SemanticTransitionError::PublicationRefused {
                    status: proof.status,
                });
            }
            match original.observed_proof {
                Some(proof) if proof.abi == 1 && proof.status != 0 => Some((
                    original
                        .activation_work
                        .clone()
                        .ok_or(SemanticTransitionError::NotBound)?,
                    proof.status,
                )),
                _ => None,
            }
        };
        if let Some((work, status)) = retained_refusal {
            // The native refusal is already observed. Resolve only this same
            // report, whose admitted MODEL region never entered its callback.
            self.finish_original_model_context_activation_report(
                source,
                current,
                continuation,
                &work,
            )?;
            return Err(SemanticTransitionError::PublicationRefused { status });
        }
        let work = self.prepare_original_model_context_activation(source, continuation, current)?;
        let (native_work, claimed) = {
            let original = owner
                .original
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?;
            (
                original
                    .native_work
                    .clone()
                    .ok_or(SemanticTransitionError::NotBound)?,
                original.native_claimed,
            )
        };
        if !claimed {
            self.claim_model_context_activation_work(continuation, &work)?;
            owner
                .original
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?
                .native_claimed = true;
        }
        let execute = self
            .provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_publication_continue_model_context",
            )
            .ok_or_else(|| {
                runtime_error("kernel lookup", "original model continuation unavailable")
            })?;
        let mut descriptor = self.descriptor();
        let mut recorder = self.kernel_recorder();
        owner.source_publication.record(&mut recorder);
        let source_reader = source
            .readers
            .get(&current.token)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        recorder.read(&source_reader.device);
        recorder.read_write(&native_work);
        let mut original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original
            .source_alias
            .as_ref()
            .is_none_or(|alias| !alias.command.completed)
        {
            return Err(publication_input_error(
                "publication requires known completion of the original selected model alias",
            ));
        }
        let parent = original
            .parent
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        parent.storage.record(&mut recorder);
        recorder.read_write(&parent.proof);
        let mut proof_ptr = parent.proof.device_ptr_value();
        let mut lease_ptr = source_reader.device.device_ptr_value();
        let mut work_ptr = *native_work.device_ptr();
        let OriginalModelContextContinuation {
            command, poisoned, ..
        } = &mut *original;
        command.run(
            &self.domain,
            poisoned,
            recorder,
            |enqueue, entered, submitted| {
                let mut params = [
                    (&mut descriptor as *mut Descriptor).cast(),
                    (&mut proof_ptr as *mut u64).cast(),
                    (&mut lease_ptr as *mut u64).cast(),
                    (&mut work_ptr as *mut u64).cast(),
                ];
                *entered = true;
                // SAFETY: the exact source reader, original destination bank,
                // frozen proof, model backings and tally are all retained/recorded.
                unsafe {
                    execute.launch_raw_in(
                        enqueue,
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &mut params,
                        false,
                    )
                }
                .map_err(|error| XlogError::Kernel(error.to_string()))?;
                *submitted = true;
                Ok(())
            },
        )?;
        if original.observed_proof.is_none() {
            let OriginalModelContextContinuation {
                proof_read,
                poisoned,
                ..
            } = &mut *original;
            let rows = self.resolve_publication_read_with_poison(
                proof_read
                    .as_mut()
                    .ok_or(SemanticTransitionError::NotBound)?,
                poisoned,
            )?;
            if rows.len() != 1 {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            original.observed_proof = Some(rows[0]);
        }
        let proof = original
            .observed_proof
            .expect("original proof observation retained");
        if proof.abi != 1 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if proof.status != 0 {
            drop(original);
            self.finish_original_model_context_activation_report(
                source,
                current,
                continuation,
                &work,
            )?;
            return Err(SemanticTransitionError::PublicationRefused {
                status: proof.status,
            });
        }
        if original.observed_header.is_none() {
            let OriginalModelContextContinuation {
                header_read,
                poisoned,
                ..
            } = &mut *original;
            let rows = self.resolve_publication_read_with_poison(
                header_read
                    .as_mut()
                    .ok_or(SemanticTransitionError::NotBound)?,
                poisoned,
            )?;
            if rows.len() != 1 {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            original.observed_header = Some(rows[0]);
        }
        let header = original
            .observed_header
            .expect("original header observation retained");
        if header.abi != 1
            || header.instance
                != original
                    .parent
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?
                    .storage
                    .instance
            || header.publication_word != proof.target_word
            || proof.source_word != current.identity.word
            || header.model_generation != proof.acquired_generation
            || header.training_cursor != proof.training_cursor
            || header.training_rng != proof.training_rng
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        drop(original);
        let mut program = owner
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        // Every fallible Program/scope/resource check preceded the native seal.
        self.transfer_original_program_context_continuation(source, &mut program);
        self.rng = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .activation_rng;
        self.next_proposal = header.proposal;
        let identity = SemanticPublishedIdentity {
            instance: header.instance,
            word: header.publication_word,
            logical_digest: header.logical_digest,
            state_digest: header.state_digest,
        };
        let mut original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        original.activated = Some(identity);
        if let Some(mut stage) = original
            .parent
            .as_mut()
            .and_then(|parent| parent.prefill.take())
        {
            let task_content = self
                .task
                .as_ref()
                .expect("authenticated original TARGET task")
                .0
                .content_identity();
            self.retained_initial_prefill = Some(RetainedInitialPrefillContent {
                scope: stage.scope,
                task_identity: stage.task_identity,
                task_epoch: stage.task_epoch,
                model_generation: stage.model_generation,
                model: stage
                    .model_content
                    .take()
                    .expect("authenticated original model content"),
                output: stage
                    .output_content
                    .take()
                    .expect("authenticated original output content"),
                record: InitialPrefillRecord {
                    predecessor: identity,
                    semantic_root: header.semantic_digest,
                    task_identity: stage.task_identity,
                    task_epoch: stage.task_epoch,
                    task_content,
                    model_generation: stage.model_generation,
                    tokens: stage.layout.prefill_tokens,
                    mapping: stage.layout.prefill_mapping,
                },
                captured: stage.captured_content,
                context_content_cursor: 0,
            });
        }
        original.poisoned = false;
        original.proof_read = None;
        original.header_read = None;
        Ok(identity)
    }

    pub fn admit_model_context_continuation_instruction(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        instruction: &[u8],
    ) -> Result<SemanticSegmentInstructionAdmission, SemanticTransitionError> {
        let owner = &continuation.inner;
        let original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !Arc::ptr_eq(&self.publication_issuer, &owner.target_issuer)
            || original.activated.is_none()
            || original.refused
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let work = original
            .activation_work
            .clone()
            .ok_or(SemanticTransitionError::NotBound)?;
        drop(original);
        let completed = self.completed_model_context_cold_result(continuation, &work)?;
        let mut program = owner
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if instruction != program.instruction() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let admission =
            self.admit_future_original_program_instruction(&mut program, parent, instruction)?;
        completed.bind(self.instruction_admission_mut(&admission)?);
        Ok(admission)
    }

    pub fn enter_model_context_continuation_execution(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        if !Arc::ptr_eq(&self.publication_issuer, &continuation.inner.target_issuer) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let program = continuation
            .inner
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        self.enter_future_original_program_execution(&program, admission, parent)
    }

    /// Bind the actual selected model wrapper to its original numerical owners.
    /// A checkpoint may project a new wrapper, never a new backing allocation.
    pub fn require_model_context_source_alias(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        target: &mut Self,
        parent: &SemanticPublishedLease,
        tensors: Vec<SemanticTensorInput>,
        allocations: Vec<SemanticTensorInput>,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        let work = target.prepare_original_model_context_activation(self, continuation, parent)?;
        self.attach_model_context_activation_work(parent, continuation, target, &work)?;
        continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .source_work_attached = true;
        self.stage_model_context_source_alias(
            continuation,
            parent,
            tensors,
            allocations,
            consumer_stream,
        )?;
        self.resolve_model_context_source_alias(continuation, target, parent)
    }

    /// Import and retain the exact selected SOURCE topology without entering
    /// an activation plan or a numerical producer.
    pub(super) fn require_model_context_alias_resolution_quiescence(
        &self,
        owner: &Arc<ModelContextContinuationOwner>,
        parent: &SemanticPublishedLease,
        alias: &OriginalModelContextSourceAlias,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_original_consumer_step(parent)?;
        let reader = self
            .readers
            .get(&parent.token)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if !Arc::ptr_eq(&self.publication_issuer, &owner.source_issuer)
            || !Arc::ptr_eq(&parent.issuer, &owner.source_issuer)
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
            || !parent.active
            || reader.acquisition_pending
            || reader.retirement_entered
            || reader.retirement_pending
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if let Some(error) = alias.command.completion_failure {
            return Err(runtime_error(
                "original model alias completion",
                cudarc::driver::DriverError(error),
            ));
        }
        let resume_original = self.poisoned && alias.command.entered && !alias.command.completed;
        self.ensure_quiescent_except_cold_report_with_effect_exclusion(
            None,
            OriginalSessionEffectExclusion {
                allow_session_poison: resume_original,
                delivery_acknowledgement: None,
            },
        )
    }

    fn require_model_context_source_alias_tensor_basis(
        &self,
        owner: &Arc<ModelContextContinuationOwner>,
        parent: &SemanticPublishedLease,
        tensors: &[PreparedSemanticTensor],
        allocations: &[PreparedSemanticTensor],
    ) -> Result<(), SemanticTransitionError> {
        self.checked_original_consumer_step(parent)?;
        let reader = self
            .readers
            .get(&parent.token)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if !parent.active
            || reader.acquisition_pending
            || reader.retirement_entered
            || reader.retirement_pending
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let header = parent.model_context_header()?;
        if !Arc::ptr_eq(&self.publication_issuer, &owner.source_issuer)
            || !Arc::ptr_eq(&parent.issuer, &owner.source_issuer)
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
            || self
                .publication
                .as_ref()
                .is_none_or(|storage| !Arc::ptr_eq(storage, &owner.source_publication))
            || header.instance != owner.source_publication.instance
            || header.model_generation != owner.source_header.model_generation
            || allocations.len() != owner.models.len()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let slots = owner
            .source_publication
            .model_slots_for_directory(&parent.directory)?;
        if slots.len() != owner.models.len() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        for ((tensor, original), slots) in allocations.iter().zip(&owner.models).zip(&slots) {
            let selected = owner.source_publication.allocations[slots[0]]
                .model_owner()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if !Arc::ptr_eq(&selected, original)
                || tensor.data != original.allocation.device_ptr_value()
                || tensor
                    .source
                    .as_ref()
                    .is_none_or(|source| source.len() != original.allocation.len())
            {
                return Err(publication_input_error(
                    "selected model alias changed its original physical owner",
                ));
            }
        }
        let expected_count = owner
            .source_publication
            .layouts
            .keys()
            .filter(|(role, _)| matches!(*role, 18..=25))
            .count();
        let mut distinct = BTreeSet::new();
        if tensors.len() != expected_count {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        for tensor in tensors {
            let key = (tensor.layout.role, tensor.layout.index);
            if !matches!(key.0, 18..=25)
                || !distinct.insert(key)
                || owner.source_publication.layouts.get(&key) != Some(&tensor.layout)
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let (allocation, offset) = owner
                .source_publication
                .model_memory
                .location(key.0, key.1)?;
            let original = owner
                .models
                .get(allocation)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let address = original
                .allocation
                .device_ptr_value()
                .checked_add(offset as u64)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            if tensor.data != address
                || parent
                    .directory
                    .iter()
                    .find(|range| (range.role, range.index) == key)
                    .is_none_or(|range| {
                        range.logical_begin != tensor.logical_begin
                            || range.logical_end != tensor.logical_end
                    })
            {
                return Err(publication_input_error(
                    "selected model view changed its original layout or interval",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn require_original_model_context_source_alias_basis(
        &self,
        owner: &Arc<ModelContextContinuationOwner>,
        parent: &SemanticPublishedLease,
        alias: &OriginalModelContextSourceAlias,
    ) -> Result<(), SemanticTransitionError> {
        dlpack_consumer_stream(alias.consumer_stream)?;
        self.require_model_context_source_alias_tensor_basis(
            owner,
            parent,
            &alias.tensors,
            &alias.allocations,
        )
    }

    pub(super) fn stage_model_context_source_alias(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        tensors: Vec<SemanticTensorInput>,
        allocations: Vec<SemanticTensorInput>,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        if continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .source_alias
            .is_some()
        {
            return Err(publication_input_error(
                "selected model alias already has its original operation",
            ));
        }
        let bytes = tensors
            .len()
            .checked_add(allocations.len())
            .and_then(|count| {
                count.checked_mul(
                    size_of::<PreparedSemanticTensor>() + size_of::<SemanticTensorInput>(),
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    size_of::<OriginalModelContextSourceAlias>()
                        + 2 * size_of::<cudarc::driver::CudaEvent>(),
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .reserve_auxiliary(
                u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            )?;
        let count = tensors.len();
        let mut producers = tensors;
        producers.extend(allocations);
        // Metadata import reads no tensor cells. The single original event
        // command below orders these retained producers before publication.
        let mut tensors = prepare_semantic_tensors(&self.provider, producers)?;
        let allocations = tensors.split_off(count);
        dlpack_consumer_stream(consumer_stream)?;
        self.checked_reader(parent)?;
        self.require_model_context_source_alias_tensor_basis(
            &continuation.inner,
            parent,
            &tensors,
            &allocations,
        )?;
        continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .source_alias = Some(OriginalModelContextSourceAlias {
            tensors,
            allocations,
            consumer_stream,
            command: OriginalNativeCommand::new(&self.domain)?,
            _reservation: reservation,
        });
        Ok(())
    }

    pub(super) fn require_model_context_source_alias_inputs(
        &self,
        continuation: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
        tensors: Vec<SemanticTensorInput>,
        allocations: Vec<SemanticTensorInput>,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        let bytes = tensors
            .len()
            .checked_add(allocations.len())
            .and_then(|count| {
                count.checked_mul(
                    size_of::<PreparedSemanticTensor>() + size_of::<SemanticTensorInput>(),
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    size_of::<OriginalModelContextSourceAlias>()
                        + 2 * size_of::<cudarc::driver::CudaEvent>(),
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let _reservation = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .reserve_auxiliary(
                u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            )?;
        let count = tensors.len();
        let mut producers = tensors;
        producers.extend(allocations);
        // Metadata comparison does not replay the original ordering event.
        let mut tensors = prepare_semantic_tensors(&self.provider, producers)?;
        let allocations = tensors.split_off(count);
        dlpack_consumer_stream(consumer_stream)?;
        self.require_model_context_source_alias_tensor_basis(
            &continuation.inner,
            parent,
            &tensors,
            &allocations,
        )?;
        let original = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let alias = original
            .source_alias
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let same = |actual: &[PreparedSemanticTensor], saved: &[PreparedSemanticTensor]| {
            actual.len() == saved.len()
                && actual.iter().zip(saved).all(|(actual, saved)| {
                    actual.layout == saved.layout
                        && actual.logical_begin == saved.logical_begin
                        && actual.logical_end == saved.logical_end
                        && actual.data == saved.data
                        && actual.source.as_ref().map(|view| view.len())
                            == saved.source.as_ref().map(|view| view.len())
                })
        };
        if alias.consumer_stream != consumer_stream
            || !same(&tensors, &alias.tensors)
            || !same(&allocations, &alias.allocations)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub fn resolve_model_context_source_alias(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        target: &Self,
        parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        let owner = &continuation.inner;
        if !Arc::ptr_eq(&self.publication_issuer, &owner.source_issuer)
            || !Arc::ptr_eq(&target.publication_issuer, &owner.target_issuer)
            || !Arc::ptr_eq(&parent.issuer, &owner.source_issuer)
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
            || target
                .model_context_continuation
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, owner))
            || !parent.active
            || !self.readers.contains_key(&parent.token)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.require_model_context_source_alias_parent(target, parent)?;
        let mut original = owner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original.source_parent.is_none() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let alias = original
            .source_alias
            .as_mut()
            .ok_or(SemanticTransitionError::NotBound)?;
        let mut recorder = self.domain.new_strict_recorder();
        for tensor in alias.tensors.iter().chain(&alias.allocations) {
            if let Some(view) = &tensor.source {
                recorder.read(view);
            }
        }
        self.order_tensor_producer_inputs_with_original(
            alias.consumer_stream,
            Some((&mut alias.command, recorder)),
        )?;
        self.poisoned = false;
        Ok(())
    }

    pub fn continuation_source_proposal(
        &self,
        continuation: &SemanticModelContextContinuation,
        parent: &SemanticPublishedLease,
    ) -> Result<u64, SemanticTransitionError> {
        let original = &continuation.inner;
        let header = parent.model_context_header()?;
        let expected = original
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .expected_source_proposal();
        if !Arc::ptr_eq(&self.publication_issuer, &original.target_issuer)
            || !Arc::ptr_eq(&parent.issuer, &original.source_issuer)
            || !parent.active
            || header.instance != original.source_publication.instance
            || header.model_generation != original.source_header.model_generation
            || header.proposal != expected
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(header.proposal)
    }

    pub fn begin_model_context_prefill(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        source: &mut Self,
        source_parent: &SemanticPublishedLease,
        source_admission: &SemanticSegmentInstructionAdmission,
        source_work: &SemanticColdModelWork,
        layout: SemanticInitialSourceLayout,
        model_generation: u64,
        geometry: &[SemanticModelContextPrefillContentGeometry],
    ) -> Result<SemanticInitialPrefillLease, SemanticTransitionError> {
        continuation.inner.require_imported_target(self)?;
        if model_generation != continuation.inner.source_header.model_generation {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        source.freeze_model_context_prefill_roster(
            continuation,
            self,
            source_parent,
            source_admission,
            source_work,
            geometry,
            layout.prefill_tokens.len(),
        )?;
        self.attach_model_context_prefill_work(
            continuation,
            source,
            source_parent,
            source_admission,
            source_work,
        )?;
        let result = self.begin_initial_prefill(layout, model_generation);
        if let Some(stage) = &self.initial_prefill {
            let mut original = continuation
                .inner
                .original
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?;
            original.initial_prefill_scope = Some(Arc::clone(&stage.scope));
        }
        result
    }

    pub fn resolve_model_context_prefill(
        &mut self,
        continuation: &SemanticModelContextContinuation,
    ) -> Result<SemanticInitialPrefillLease, SemanticTransitionError> {
        let stage = self
            .initial_prefill
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        continuation
            .inner
            .require_initial_prefill_preparation(self, &stage.scope)?;
        let allowance = self.graph.cold_work_allowance();
        let stage = self
            .initial_prefill
            .as_mut()
            .expect("same original prefill stage");
        stage
            .token_write
            .as_mut()
            .ok_or(SemanticTransitionError::NotBound)?
            .resolve_original(
                &self.domain,
                &self.stream,
                &self.provider,
                &mut self.poisoned,
                false,
                allowance.as_ref(),
            )?;
        self.poisoned = false;
        Ok(SemanticInitialPrefillLease {
            issuer: Arc::clone(&self.publication_issuer),
            scope: Arc::clone(&stage.scope),
        })
    }

    pub fn resolve_model_context_parent(
        &mut self,
        continuation: &SemanticModelContextContinuation,
    ) -> Result<(), SemanticTransitionError> {
        if !Arc::ptr_eq(&self.publication_issuer, &continuation.inner.target_issuer) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let mut original = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        ModelContextContinuationOwner::resolve_dormant_parent(self, &mut original)
    }

    /// Freeze the next real instruction and existing numerical owners, without
    /// a TARGET publication, live instruction claim or extra source reader.
    pub fn freeze_model_context_continuation(
        &mut self,
        target: &mut Self,
        instruction: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
    ) -> Result<SemanticModelContextContinuation, SemanticTransitionError> {
        self.require_segment_instruction_admission(instruction, parent)?;
        self.checked_reader(parent)?;
        target.ensure_rebindable()?;
        let header = parent.model_context_header()?;
        let shared_catalogue = matches!(
            (&self.codebooks.model_catalogue, &target.codebooks.model_catalogue),
            (Some((source, 0)), Some((future, 1))) if Arc::ptr_eq(source, future)
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
            || self.next_proposal != header.proposal
            || header.proposal != 0
            || header.training_cursor != 0
            || self.pending_replay_delivery.is_some()
            || self.model_owners.is_empty()
        {
            return Err(publication_input_error("model continuation must freeze once from the original imported contexts before drawing"));
        }
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let next = self
            .admitted_segment_steps(instruction)?
            .last()
            .ok_or(SemanticTransitionError::NotBound)?
            .ordinal
            .checked_add(1)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let instruction_bytes = self
            .actor_refresh_program
            .as_ref()
            .and_then(|program| program.steps.get(next as usize))
            .ok_or(SemanticTransitionError::NotBound)?
            .instruction
            .len();
        let bytes =
            size_of::<ModelContextContinuationOwner>()
                .checked_add(size_of::<OriginalModelContextContinuation>())
                .and_then(|bytes| {
                    bytes.checked_add(
                        size_of::<DormantModelContextParent>()
                            + 2 * size_of::<PreparedModelContextContinuationDevice>(),
                    )
                })
                .and_then(|bytes| {
                    bytes.checked_add(storage.model_slots.len().checked_mul(
                        size_of::<Arc<ModelGenerationOwner>>() + 2 * size_of::<usize>(),
                    )?)
                })
                .and_then(|bytes| bytes.checked_add(instruction_bytes))
                .and_then(|bytes| {
                    bytes.checked_add(
                        usize::try_from(storage.contract_value.terminal_token_count)
                            .ok()?
                            .checked_mul(size_of::<u64>())?,
                    )
                })
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
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
        let reservation = budget.reserve_auxiliary(bytes as u64)?;
        let slots = storage.model_slots_for_directory(&parent.directory)?;
        let models = slots
            .iter()
            .map(|slots| {
                storage.allocations[slots[0]]
                    .model_owner()
                    .filter(|owner| owner.replay.is_none())
                    .ok_or(SemanticTransitionError::ObservationMismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let program = self.freeze_original_program_context_continuation(target, instruction)?;
        let instruction_bytes = Arc::from(program.instruction());
        let original = Arc::new(ModelContextContinuationOwner {
            source_issuer: Arc::clone(&self.publication_issuer),
            target_issuer: Arc::clone(&target.publication_issuer),
            source_publication: storage,
            _source_provider: Arc::clone(&self.provider),
            models,
            instruction: instruction_bytes,
            source_header: header,
            program: Mutex::new(program),
            original: Mutex::new(OriginalModelContextContinuation {
                parent: None,
                command: OriginalNativeCommand::new(&target.domain)?,
                source_parent: None,
                source_boundary: None,
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
                checkpoint_restore: None,
                source_alias: None,
                source_acknowledgements: Vec::new(),
                initial_prefill_scope: None,
                activating: false,
                activated: None,
                refused: false,
                poisoned: false,
            }),
            _reservation: Arc::new(reservation),
        });
        self.model_context_continuation = Some(Arc::clone(&original));
        target.model_context_continuation = Some(Arc::clone(&original));
        Ok(SemanticModelContextContinuation { inner: original })
    }

    /// Prepare only the canonical future publication destinations. The genuine
    /// prefill witnesses still authenticate its own task source and cache.
    pub fn prepare_model_context_parent(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        source: &Self,
        source_parent: &SemanticPublishedLease,
        source_admission: &SemanticSegmentInstructionAdmission,
        source_work: &SemanticColdModelWork,
        parent: SemanticParentBinding,
    ) -> Result<(), SemanticTransitionError> {
        if self
            .bind_parent_with_context(parent, Some(continuation))?
            .is_some()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.admit_original_model_context_preparation(
            source,
            continuation,
            source_parent,
            source_admission,
            source_work,
        )?;
        self.resolve_model_context_parent(continuation)
    }

    pub fn prepare_model_context_segment(
        &mut self,
        continuation: &SemanticModelContextContinuation,
    ) -> Result<Vec<SemanticPreparedStep>, SemanticTransitionError> {
        let mut program = continuation
            .inner
            .program
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let steps = self.prepare_segment_steps_from_source(
            program.transitions(),
            program.cold_capacity(),
            PreparedSegmentSource::DormantModelContext(Arc::clone(&continuation.inner)),
            None,
        )?;
        self.bind_future_original_program_preparation(&mut program, &steps)?;
        Ok(steps)
    }
}
