//! The original numerical model, retained before choosing an initial prefix.

use super::*;

pub(super) struct InitialModelContent {
    task_identity: Identity256,
    task_epoch: u64,
    generation: u64,
    record: SemanticStateRecord,
    layout: SemanticModelContractLayout,
    memory: ModelMemoryGeometry,
    model: Option<InitialPrefillTensorOwners>,
    stream: u64,
}

fn same_roster<'a>(
    original: &InitialPrefillTensorOwners,
    actual: impl Iterator<Item = &'a PreparedSemanticTensor>,
) -> bool {
    let actual = actual.collect::<Vec<_>>();
    original.tensors.len() == actual.len()
        && original
            .tensors
            .iter()
            .zip(actual)
            .all(|(expected, actual)| same_tensor_content_owner(expected, actual))
}

impl SemanticTransitionSession {
    /// Retain one complete, actual model before either empty-prefix binding or
    /// prefill. The first version belongs to these owners, independently of the
    /// optimizer's update count. Publication later finalizes its model identity.
    pub fn bind_initial_model_content(
        &mut self,
        record: SemanticStateRecord,
        layout: SemanticModelContractLayout,
        tensors: Vec<SemanticTensorInput>,
        mut memory: SemanticModelMemory,
        consumer_stream: u64,
    ) -> Result<u64, SemanticTransitionError> {
        let model_count = tensors.len();
        let mut inputs = tensors;
        inputs.append(&mut memory.allocations);
        let inputs = complete_tensor_handoff(&self.provider, inputs)?;
        self.ensure_rebindable_with_prefill()?;
        #[cfg(feature = "semantic-policy")]
        if self.model_context_continuation.is_some() {
            return Err(publication_input_error("model continuation retains its selected source instead of issuing an initial version"));
        }
        dlpack_consumer_stream(consumer_stream)?;
        let task_identity = self
            .task_evaluation_identity()
            .ok_or(SemanticTransitionError::NotBound)?;
        if self.publication.is_some()
            || record.role != SemanticStateRole::ModelContract
            || record.index != 0
            || record.bytes.len() > record.capacity_bytes
            || model_count == 0
            || inputs[..model_count]
                .iter()
                .any(|tensor| !matches!(tensor.layout.role, 18..=25))
        {
            return Err(publication_input_error(
                "initial model requires an unpublished task and its complete model contract and owners",
            ));
        }
        layout.validate(record.bytes.len() as u64)?;
        for (offset, length) in [
            (layout.schema_digest_offset, 32),
            (layout.generation_offset, 8),
            (layout.identity_offset, 32),
        ] {
            if record.bytes[offset as usize..(offset + length) as usize]
                .iter()
                .any(|byte| *byte != 0)
            {
                return Err(publication_input_error(
                    "initial model contract must reserve native digest, generation and identity outputs",
                ));
            }
        }
        let mut prepared = prepare_semantic_tensors(&self.provider, inputs)?;
        let allocations = prepared.split_off(model_count);
        let geometry =
            prepare_model_memory(&allocations, memory.storages, memory.views, &prepared)?;
        prepared.extend(allocations);
        validate_content_coordinates_after(
            prepared
                .iter()
                .map(|tensor| (tensor.layout.role, tensor.layout.index)),
            model_count,
        )?;
        if let Some(original) = &self.initial_model {
            self.require_initial_model_origin(original)?;
            let owners = original.model.as_ref().or_else(|| {
                self.initial_prefill
                    .as_ref()
                    .and_then(|stage| stage.model_content.as_ref())
            });
            if original.record.bytes != record.bytes
                || original.record.capacity_bytes != record.capacity_bytes
                || original.layout != layout
                || original.memory != geometry
                || original.stream != consumer_stream
                || owners.is_none_or(|owners| !same_roster(owners, prepared.iter()))
            {
                return Err(publication_input_error(
                    "initial model cannot replace its original contract or tensor owners",
                ));
            }
            return Ok(original.generation);
        }
        if self.initial_prefill.is_some() || !self.model_owners.is_empty() {
            return Err(publication_input_error(
                "initial model must precede prefill and selected model ownership",
            ));
        }
        let generation = 1;
        self.initial_model = Some(InitialModelContent {
            task_identity,
            task_epoch: self.task_epoch,
            generation,
            record,
            layout,
            memory: geometry,
            model: Some(InitialPrefillTensorOwners { tensors: prepared }),
            stream: consumer_stream,
        });
        let result = self
            .order_tensor_producer_inputs(consumer_stream)
            .and_then(|()| self.order_content_consumers(consumer_stream));
        if result.is_err() {
            self.poisoned = true;
        }
        result.map(|()| generation)
    }

    pub(super) fn require_initial_model_origin(
        &self,
        original: &InitialModelContent,
    ) -> Result<(), SemanticTransitionError> {
        if original.task_epoch != self.task_epoch
            || self.task_evaluation_identity() != Some(original.task_identity)
        {
            return Err(publication_input_error(
                "initial model belongs to another original task",
            ));
        }
        Ok(())
    }

    pub(super) fn require_initial_model_generation(
        &self,
        generation: u64,
    ) -> Result<(), SemanticTransitionError> {
        if let Some(original) = &self.initial_model {
            self.require_initial_model_origin(original)?;
            if original.generation != generation {
                return Err(publication_input_error(
                    "prefill changed its original model generation",
                ));
            }
            return Ok(());
        }
        #[cfg(feature = "semantic-policy")]
        if let Some(original) = &self.model_context_continuation {
            let source = SemanticModelContextContinuation {
                inner: Arc::clone(original),
            };
            if source.model_parent_coordinates(self)?.0 == generation {
                return Ok(());
            }
        }
        Err(publication_input_error(
            "prefill requires its registered original model version",
        ))
    }

    pub(super) fn take_initial_model_for_prefill(
        &mut self,
        actual: &[PreparedSemanticTensor],
        consumer_stream: u64,
    ) -> Result<InitialPrefillTensorOwners, SemanticTransitionError> {
        let Some(original) = &self.initial_model else {
            return Err(publication_input_error(
                "prefill requires its registered original model owners",
            ));
        };
        self.require_initial_model_origin(original)?;
        if original.stream != consumer_stream
            || original
                .model
                .as_ref()
                .is_none_or(|model| !same_roster(model, actual.iter()))
        {
            return Err(publication_input_error(
                "prefill changed its original model stream or tensor owners",
            ));
        }
        self.initial_model
            .as_mut()
            .expect("original model")
            .model
            .take()
            .ok_or_else(|| publication_input_error("initial model owners already joined prefill"))
    }

    pub(super) fn join_initial_model_parent(
        &self,
        parent: &SemanticParentBinding,
        memory: &ModelMemoryGeometry,
        tensors: &[PreparedSemanticTensor],
        allocations: &[PreparedSemanticTensor],
        #[cfg(feature = "semantic-policy")] continuation: Option<&SemanticModelContextContinuation>,
    ) -> Result<(), SemanticTransitionError> {
        let Some(original) = &self.initial_model else {
            #[cfg(feature = "semantic-policy")]
            if let Some(continuation) = continuation {
                return continuation.inner.require_parent_binding(
                    self,
                    parent,
                    memory,
                    tensors,
                    allocations,
                );
            }
            return Err(publication_input_error(
                "fresh parent requires its registered original model",
            ));
        };
        #[cfg(feature = "semantic-policy")]
        if continuation.is_some() {
            return Err(publication_input_error(
                "initial model cannot replace a selected model continuation",
            ));
        }
        self.require_initial_model_origin(original)?;
        let owners = original.model.as_ref().or_else(|| {
            self.initial_prefill
                .as_ref()
                .and_then(|stage| stage.model_content.as_ref())
        });
        let records = parent
            .records
            .iter()
            .filter(|record| record.role == SemanticStateRole::ModelContract)
            .collect::<Vec<_>>();
        if parent.model_generation != original.generation
            || u64::from(parent.rng.model_generation) != original.generation
            || parent.model_contract_layout != original.layout
            || *memory != original.memory
            || records.len() != 1
            || records[0].index != 0
            || records[0].bytes != original.record.bytes
            || records[0].capacity_bytes != original.record.capacity_bytes
            || owners.is_none_or(|model| {
                !same_roster(
                    model,
                    tensors
                        .iter()
                        .filter(|tensor| matches!(tensor.layout.role, 18..=25))
                        .chain(allocations),
                )
            })
        {
            return Err(publication_input_error(
                "parent changed its original model version, contract or tensor owners",
            ));
        }
        Ok(())
    }
}
