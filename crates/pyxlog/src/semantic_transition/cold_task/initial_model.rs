use super::*;
use crate::semantic_transition::{
    parse_model_contract_layout, parse_model_storages, parse_model_views, parse_state_records,
    parse_tensor_inputs_guarded, parse_witness_consumer_stream, ColdValue,
    PySemanticTransitionTaskUse, TaskUsePhase, TensorProducerOperation,
};

impl PySemanticTransitionFreshParent {
    pub(super) fn bind_original_session(
        &self,
        session: &PySemanticTransitionSession,
    ) -> PyResult<()> {
        let original = self
            .original_session
            .get_or_init(|| Arc::downgrade(&session.inner));
        if !original.ptr_eq(&Arc::downgrade(&session.inner)) {
            return Err(invalid("fresh parent belongs to another original Session"));
        }
        Ok(())
    }

    pub(in crate::semantic_transition) fn require_original_session(
        &self,
        session: &PySemanticTransitionSession,
    ) -> PyResult<()> {
        self.require_creator()?;
        session.require_creator()?;
        if self
            .original_session
            .get()
            .and_then(Weak::upgrade)
            .is_none_or(|original| !Arc::ptr_eq(&original, &session.inner))
        {
            return Err(invalid("fresh parent belongs to another original Session"));
        }
        Ok(())
    }

    // Actor construction compares cold inputs before either Session exists.
    // Session identity is checked separately when numerical owners are joined.
    pub(in crate::semantic_transition) fn same_template(&self, other: &Self) -> bool {
        self.owner_thread == other.owner_thread
            && self.task_digest == other.task_digest
            && self.provenance_capacity_records == other.provenance_capacity_records
            && self.prefix_capacity == other.prefix_capacity
            && self.feedback_capacity == other.feedback_capacity
            && self.pad_token == other.pad_token
            && self.terminal_tokens == other.terminal_tokens
            && self.final_intent_payload_bytes == other.final_intent_payload_bytes
            && self.nonterminal_intent == other.nonterminal_intent
            && self.intent_effect == other.intent_effect
            && self.intent_entry_capacity == other.intent_entry_capacity
            && self.intent_payload_capacity_bytes == other.intent_payload_capacity_bytes
            && self.acknowledgement_payload_capacity_bytes
                == other.acknowledgement_payload_capacity_bytes
            && self.generations == other.generations
            && self.training_cursor == other.training_cursor
            && self.training_rng == other.training_rng
            && self.fuel == other.fuel
            && self.rng == other.rng
            && self.role_counts == other.role_counts
            && self.authority_decisions_capacity_bytes == other.authority_decisions_capacity_bytes
            && self.records == other.records
    }

    fn require_initial_model_task(
        &self,
        py: Python<'_>,
        task: &PySemanticTransitionTaskUse,
    ) -> PyResult<()> {
        let session = task.session.borrow(py);
        self.require_original_session(&session)?;
        let owner = session.owner()?;
        task.require_current(&owner)?;
        if !matches!(task.state()?.phase, TaskUsePhase::Imported) {
            return Err(invalid(
                "initial model requires its original imported task before prefill",
            ));
        }
        #[cfg(feature = "semantic-policy")]
        if task
            .frozen_model_context
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(invalid(
                "frozen model context requires its original continuation",
            ));
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "one original model binds its full typed contract and tensor rosters"
    )]
    pub(super) fn bind_initial_model_owner(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        model_record: &Bound<'_, PyAny>,
        model_contract_layout: &Bound<'_, PyAny>,
        tensors: &Bound<'_, PyAny>,
        model_allocations: &Bound<'_, PyAny>,
        model_storages: &Bound<'_, PyAny>,
        model_views: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        self.require_initial_model_task(py, task_use)?;
        let session = task_use.session.borrow(py);
        let _operation = TensorProducerOperation::begin(&session)?;
        let mut budget = 16 * 1024 * 1024;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut budget)?;
        let row = PyTuple::new(py, [model_record])?;
        let record = parse_state_records(row.as_any(), &mut budget)?
            .pop()
            .ok_or_else(|| invalid("initial model requires its actual contract record"))?;
        let layout =
            parse_model_contract_layout(&ColdValue::read(model_contract_layout, &mut budget, 0)?)?;
        let storages = parse_model_storages(model_storages, &mut budget)?;
        let views = parse_model_views(model_views, &mut budget)?;
        let check = || self.require_initial_model_task(py, task_use);
        let model = parse_tensor_inputs_guarded(
            tensors,
            &mut budget,
            session.device_ordinal,
            stream,
            &check,
        )?;
        let allocations = parse_tensor_inputs_guarded(
            model_allocations,
            &mut budget,
            session.device_ordinal,
            stream,
            &check,
        )?;
        check()?;
        let scope = session
            .initial_model_scope(task_use)?
            .unwrap_or_else(|| Arc::new(()));
        let producers =
            session.retain_initial_prefill_stage_producers(task_use, &scope, Vec::new())?;
        {
            let mut retained = producers
                .model
                .lock()
                .map_err(|_| invalid("initial model producer custody is poisoned"))?;
            retained.extend(model.producers);
            retained.extend(allocations.producers);
        }
        let generation = session
            .owner()?
            .bind_initial_model_content(
                record,
                layout,
                model.handoff.into_native(),
                xlog_cuda::SemanticModelMemory {
                    allocations: allocations.handoff.into_native(),
                    storages,
                    views,
                },
                stream,
            )
            .map_err(xlog_err)?;
        let mut projected = self.clone();
        projected.generations.0 = generation;
        projected.rng.0 = u32::try_from(generation)
            .map_err(|_| invalid("initial model generation exceeds the RNG domain"))?;
        projected.refresh_model_records();
        Ok(projected)
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn project_model_context(
        &self,
        py: Python<'_>,
        continuation: &crate::semantic_transition::model_context_continuation::PySemanticModelContextContinuation,
    ) -> PyResult<Self> {
        self.require_creator()?;
        let (generation, neural, cursor, training_rng, rng) =
            continuation.parent_coordinates(py, self)?;
        let mut projected = self.clone();
        projected.generations.0 = generation;
        projected.generations.2 = neural;
        projected.training_cursor = cursor;
        projected.training_rng = training_rng;
        projected.rng = (
            rng.model_generation,
            rng.stream_serial,
            rng.family_id,
            rng.proposal,
        );
        projected.refresh_model_records();
        Ok(projected)
    }

    fn refresh_model_records(&mut self) {
        let [rng, goal] = fresh_model_records(
            self.task_digest,
            self.generations,
            self.training_cursor,
            self.training_rng,
            (self.rng.1, self.rng.2, self.rng.3),
        );
        for replacement in [rng, goal] {
            let original = self
                .records
                .iter_mut()
                .find(|record| record.role == replacement.role)
                .expect("native fresh model record");
            *original = replacement;
        }
    }
}
