//! Native non-learning model work and its canonical instruction receipt.
use super::*;
use serde_json::{json, Value};

const CONTROL_MAGIC: &[u8] = b"XLOG-MODEL-CONTROL\0";
pub(super) const CONTROL_BYTES: usize = CONTROL_MAGIC.len()
    + 6 * 32
    + 21 * 8
    + size_of::<SemanticTrainingViewSelection>()
    + 5 * (size_of::<SemanticTrainingCanaryRecord>()
        + size_of::<SemanticTrainingCanaryResultRecord>())
    + 2 * size_of::<ModelForwardReceipt>()
    + size_of::<SemanticTrainingCanaryRefusalRecord>()
    + size_of::<SemanticCapacityRefusalRecord>();

/// The original terminal receipt. Its bytes are a field of the existing
/// instruction-assignment carrier, never a second model or checkpoint format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticControlObservation {
    pub(super) bytes: Arc<[u8]>,
}

#[derive(PartialEq, Eq)]
pub(super) struct ControlContext {
    pub(super) program_identity: Identity256,
    pub(super) first: u64,
    pub(super) end: u64,
    pub(super) family: (Identity256, u64),
    pub(super) plan_identity: Identity256,
    pub(super) budget: [u64; 3],
}

struct ControlFacts {
    context: ControlContext,
    parent: SemanticPublishedIdentity,
    generation: u64,
    geometry: Identity256,
    status: u64,
    work: [u64; 3],
    calls: Option<u64>,
    copy: Option<u64>,
    cache: Option<u64>,
    selection: SemanticTrainingViewSelection,
    canaries: [SemanticTrainingCanaryRecord; 5],
    results: [SemanticTrainingCanaryResultRecord; 5],
    forwards: [ModelForwardReceipt; 2],
    refusal: Option<SemanticTrainingCanaryRefusalRecord>,
    capacity: Option<SemanticCapacityRefusalRecord>,
}

// This sealed boundary only admits native C records made entirely of integer
// words, byte-array identities and embedded records with the same property.
// None of their inert pointer words becomes a live capability on restoration.
unsafe trait ControlWords: Copy + DeviceRepr {}
// SAFETY: selection and its embedded origin contain only integer words/arrays.
unsafe impl ControlWords for SemanticTrainingViewSelection {}
// SAFETY: the frozen native canary contains only integer words/arrays.
unsafe impl ControlWords for SemanticTrainingCanaryRecord {}
// SAFETY: native measured fields and their identity are integer words/arrays.
unsafe impl ControlWords for SemanticTrainingCanaryResultRecord {}
// SAFETY: receipt fields are integer words and unrestricted byte-array identities.
unsafe impl ControlWords for ModelForwardReceipt {}
// SAFETY: refusal fields are integer words/arrays, without Rust enums or references.
unsafe impl ControlWords for SemanticTrainingCanaryRefusalRecord {}
// SAFETY: capacity refusal is an integer-word native record.
unsafe impl ControlWords for SemanticCapacityRefusalRecord {}

fn read_word_record<T: ControlWords>(
    reader: &mut SemanticMaterialReader<'_>,
) -> Result<T, SemanticTransitionError> {
    let bytes = reader
        .take(size_of::<T>())
        .map_err(SemanticTransitionError::Semantic)?;
    // SAFETY: the sealed bound excludes invalid bit patterns, references and
    // destructors; take proved the complete initialized byte extent. An
    // unaligned copy obtains inert metadata, never borrowed GPU storage.
    Ok(unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<T>()) })
}

fn identity(
    reader: &mut SemanticMaterialReader<'_>,
) -> Result<Identity256, SemanticTransitionError> {
    Ok(Identity256::from_bytes(
        reader
            .take(32)
            .map_err(SemanticTransitionError::Semantic)?
            .try_into()
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
    ))
}

fn word(reader: &mut SemanticMaterialReader<'_>) -> Result<u64, SemanticTransitionError> {
    reader.u64().map_err(SemanticTransitionError::Semantic)
}

fn optional_word(
    reader: &mut SemanticMaterialReader<'_>,
) -> Result<Option<u64>, SemanticTransitionError> {
    match word(reader)? {
        0 => Ok(None),
        1 => Ok(Some(word(reader)?)),
        _ => Err(SemanticTransitionError::ObservationMismatch),
    }
}

fn optional_record<T: ControlWords>(
    reader: &mut SemanticMaterialReader<'_>,
) -> Result<Option<T>, SemanticTransitionError> {
    match word(reader)? {
        0 => Ok(None),
        1 => Ok(Some(read_word_record(reader)?)),
        _ => Err(SemanticTransitionError::ObservationMismatch),
    }
}

fn write_record<T: ControlWords>(bytes: &mut Vec<u8>, value: &T) {
    bytes.extend_from_slice(&publication_abi_bytes(std::slice::from_ref(value)));
}

fn write_optional_word(bytes: &mut Vec<u8>, value: Option<u64>) {
    material_u64(bytes, u64::from(value.is_some()));
    if let Some(value) = value {
        material_u64(bytes, value);
    }
}

fn write_optional_record<T: ControlWords>(bytes: &mut Vec<u8>, value: Option<T>) {
    material_u64(bytes, u64::from(value.is_some()));
    if let Some(value) = value {
        write_record(bytes, &value);
    }
}

impl ControlFacts {
    fn validate(&self) -> Result<(), SemanticTransitionError> {
        if self.context.first.checked_add(1) != Some(self.context.end)
            || self.context.budget[1] == 0
            || !matches!(self.status, 0 | 5 | 11..=18)
            || (self.status != 11 && self.work[1].checked_add(self.work[2]) != Some(self.work[0]))
            || self.copy.is_some() != matches!(self.status, 0 | 18)
            || self.cache.is_some() != matches!(self.status, 0 | 18)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        for (index, (frozen, measured)) in self.canaries.iter().zip(&self.results).enumerate() {
            if frozen.evaluator_abi != 3
                || frozen.kind != index as u64 + 1
                || measured.abi != 1
                || measured.kind != frozen.kind
                || measured.row_ordinal != frozen.row_ordinal
                || measured.identity != frozen.identity
                || measured.availability > 2
                || measured.reason > 9
                || !f64::from_bits(frozen.lower_bound_bits).is_finite()
                || !f64::from_bits(frozen.upper_bound_bits).is_finite()
                || frozen.memory_limit == 0
                || frozen.work_limit == 0
                || (self.status == 0 && (measured.availability != 2 || measured.reason != 0))
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        }
        if let Some(refusal) = self.refusal {
            if decode_canary_refusal(refusal, &self.results)? != Some(refusal) {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        } else if self.results.iter().any(|result| result.reason != 0) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if self.status == 0 && (self.refusal.is_some() || self.capacity.is_some()) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        for (role, receipt) in self.forwards.iter().enumerate() {
            if receipt.availability > 2 {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            if receipt.availability == 2 {
                let bytes = receipt.canonical_bytes();
                if receipt.abi != 2
                    || receipt.role != role as u64
                    || self.generation.checked_add(role as u64) != Some(receipt.generation)
                    || receipt.model_geometry_digest != self.geometry
                    || receipt.identity
                        != Identity256::from_bytes(Sha256::digest(&bytes[..168]).into())
                    || receipt.finite > 1
                    || (self.status == 0 && receipt.finite != 1)
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
            } else {
                let mut absent = *receipt;
                absent.availability = 0;
                if absent.canonical_bytes().iter().any(|&byte| byte != 0) || self.status == 0 {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
            }
        }
        Ok(())
    }

    fn encode(&self) -> Result<SemanticControlObservation, SemanticTransitionError> {
        self.validate()?;
        let mut bytes = CONTROL_MAGIC.to_vec();
        material_u64(&mut bytes, 1);
        bytes.extend_from_slice(self.context.program_identity.as_bytes());
        for value in [self.context.first, self.context.end] {
            material_u64(&mut bytes, value);
        }
        bytes.extend_from_slice(self.context.family.0.as_bytes());
        material_u64(&mut bytes, self.context.family.1);
        bytes.extend_from_slice(self.parent.instance.as_bytes());
        material_u64(&mut bytes, self.parent.word);
        bytes.extend_from_slice(self.parent.logical_digest.as_bytes());
        material_u64(&mut bytes, self.generation);
        bytes.extend_from_slice(self.geometry.as_bytes());
        bytes.extend_from_slice(self.context.plan_identity.as_bytes());
        for value in self
            .context
            .budget
            .into_iter()
            .chain(self.work)
            .chain([self.status])
        {
            material_u64(&mut bytes, value);
        }
        write_optional_word(&mut bytes, self.calls);
        write_optional_word(&mut bytes, self.copy);
        write_optional_word(&mut bytes, self.cache);
        write_record(&mut bytes, &self.selection);
        for record in &self.canaries {
            write_record(&mut bytes, record);
        }
        for record in &self.results {
            write_record(&mut bytes, record);
        }
        for record in &self.forwards {
            write_record(&mut bytes, record);
        }
        write_optional_record(&mut bytes, self.refusal);
        write_optional_record(&mut bytes, self.capacity);
        Ok(SemanticControlObservation {
            bytes: bytes.into(),
        })
    }
}

impl SemanticControlObservation {
    pub(super) fn from_bytes(bytes: Arc<[u8]>) -> Result<Self, SemanticTransitionError> {
        if bytes.len() > CONTROL_BYTES {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let result = Self { bytes };
        result.facts()?.validate()?;
        Ok(result)
    }

    fn facts(&self) -> Result<ControlFacts, SemanticTransitionError> {
        let mut reader = SemanticMaterialReader::new(&self.bytes);
        if reader
            .take(CONTROL_MAGIC.len())
            .map_err(SemanticTransitionError::Semantic)?
            != CONTROL_MAGIC
            || word(&mut reader)? != 1
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let program_identity = identity(&mut reader)?;
        let first = word(&mut reader)?;
        let end = word(&mut reader)?;
        let family = (identity(&mut reader)?, word(&mut reader)?);
        let parent = SemanticPublishedIdentity {
            instance: identity(&mut reader)?,
            word: word(&mut reader)?,
            logical_digest: identity(&mut reader)?,
        };
        let generation = word(&mut reader)?;
        let geometry = identity(&mut reader)?;
        let plan_identity = identity(&mut reader)?;
        let budget = [word(&mut reader)?, word(&mut reader)?, word(&mut reader)?];
        let work = [word(&mut reader)?, word(&mut reader)?, word(&mut reader)?];
        let status = word(&mut reader)?;
        let calls = optional_word(&mut reader)?;
        let copy = optional_word(&mut reader)?;
        let cache = optional_word(&mut reader)?;
        let selection = read_word_record(&mut reader)?;
        let canaries = [
            read_word_record(&mut reader)?,
            read_word_record(&mut reader)?,
            read_word_record(&mut reader)?,
            read_word_record(&mut reader)?,
            read_word_record(&mut reader)?,
        ];
        let results = [
            read_word_record(&mut reader)?,
            read_word_record(&mut reader)?,
            read_word_record(&mut reader)?,
            read_word_record(&mut reader)?,
            read_word_record(&mut reader)?,
        ];
        let forwards = [
            read_word_record(&mut reader)?,
            read_word_record(&mut reader)?,
        ];
        let refusal = optional_record(&mut reader)?;
        let capacity = optional_record(&mut reader)?;
        reader.finish().map_err(SemanticTransitionError::Semantic)?;
        Ok(ControlFacts {
            context: ControlContext {
                program_identity,
                first,
                end,
                family,
                plan_identity,
                budget,
            },
            parent,
            generation,
            geometry,
            status,
            work,
            calls,
            copy,
            cache,
            selection,
            canaries,
            results,
            forwards,
            refusal,
            capacity,
        })
    }

    pub(super) fn selection_material(&self) -> Result<Vec<u8>, SemanticTransitionError> {
        Ok(publication_abi_bytes(&[self.facts()?.selection]))
    }

    pub(super) fn context(&self) -> Result<ControlContext, SemanticTransitionError> {
        Ok(self.facts()?.context)
    }
    pub(super) fn parent(&self) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        Ok(self.facts()?.parent)
    }
    pub(super) fn refused(&self) -> Result<bool, SemanticTransitionError> {
        Ok(self.facts()?.status != 0)
    }

    pub(super) fn projection(
        &self,
        retirement: Option<Vec<u8>>,
    ) -> Result<Value, SemanticTransitionError> {
        let facts = self.facts()?;
        facts.validate()?;
        let hex = |identity: Identity256| {
            identity
                .as_bytes()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        };
        let canaries = facts.canaries.iter().zip(&facts.results).map(|(frozen, measured)| json!({
            "kind": frozen.kind, "row_ordinal": frozen.row_ordinal, "identity": frozen.identity,
            "lower_bound_bits": frozen.lower_bound_bits, "upper_bound_bits": frozen.upper_bound_bits,
            "memory_limit": frozen.memory_limit, "work_limit": frozen.work_limit,
            "frozen_record": publication_abi_bytes(std::slice::from_ref(frozen)),
            "availability": measured.availability, "reason": measured.reason,
            "measurement_bits": (measured.availability == 2).then_some(measured.measurement_bits),
            "memory_used": (measured.availability != 0).then_some(measured.memory_used),
            "work_used": (measured.availability != 0).then_some(measured.work_used)
        })).collect::<Vec<_>>();
        let forwards = facts.forwards.iter().map(|receipt| json!({
            "availability": receipt.availability,
            "material": (receipt.availability == 2).then(|| json!({
                "role": receipt.role, "private_generation": receipt.generation,
                "work_sequence": receipt.work_sequence, "finite": receipt.finite,
                "model_geometry_digest": hex(receipt.model_geometry_digest),
                "logits_digest": hex(receipt.logits_digest), "identity": hex(receipt.identity),
                "canonical_bytes": receipt.canonical_bytes().to_vec()
            }))
        })).collect::<Vec<_>>();
        Ok(json!({
            "program_identity": hex(facts.context.program_identity),
            "first_program_ordinal": facts.context.first, "end_program_ordinal": facts.context.end,
            "family_binding": {"roster_sha256": hex(facts.context.family.0), "member_ordinal": facts.context.family.1},
            "parent": {"instance": hex(facts.parent.instance), "word": facts.parent.word, "logical_digest": hex(facts.parent.logical_digest)},
            "model_generation": facts.generation, "model_geometry_digest": hex(facts.geometry),
            "plan_identity": hex(facts.context.plan_identity), "budget": facts.context.budget,
            "actual_work": {"device_work": facts.work[0], "model_work": facts.work[1], "native_work": facts.work[2],
                "model_calls": facts.calls, "overflow": facts.status == 11},
            "copy": facts.copy.map(|bytes| json!({"verified_bytes": bytes, "unchanged": true})),
            "cache": facts.cache.map(|bytes| json!({"bound_bytes": bytes, "rebuilt": true})),
            "canary": {"selection_identity": facts.selection.identity,
                "selection_record": publication_abi_bytes(std::slice::from_ref(&facts.selection)),
                "checks": canaries, "forward_receipts": forwards},
            "retirement": retirement.map(|bytes| json!({"status": "retired", "native_expense": bytes})),
            "refusal": (facts.status != 0).then(|| json!({"status": facts.status,
                "copy_changed": facts.status == 17,
                "canary": facts.refusal.map(|r| publication_abi_bytes(&[r])),
                "capacity": facts.capacity.map(|r| publication_abi_bytes(&[r]))}))
        }))
    }
}

impl SemanticTransitionSession {
    pub(super) fn reconcile_control(
        &mut self,
        step: &SemanticPreparedStep,
        parent: &PublicationHeader,
        result: &PreparedStepResult,
        state: &DeviceState,
        canary_refusal: Option<SemanticTrainingCanaryRefusalRecord>,
        capacity_refusal: Option<SemanticCapacityRefusalRecord>,
    ) -> Result<SemanticTransitionOutcome, SemanticTransitionError> {
        if result.advanced != 0
            || result.word != parent.publication_word
            || result.header != *parent
            || result.refusal != 0
            || state.blocks != 0
            || state.next_proposal != parent.proposal
            || state.proposal != parent.proposal
            || u64::from(state.model_generation) != parent.model_generation
            || state.stream_serial != parent.stream_serial
            || u64::from(state.family_id) != parent.family_id
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let context = self.control_receipt_context(step, parent)?;
        let bank = usize::try_from(parent.publication_word & 1)
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        let (selection, canaries, results, forwards, actual, model_bytes, cache_bytes) = {
            let prepared = self.steps[&step.token]
                .prepared
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let view = prepared
                .training_view
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let branch = &prepared.branches[bank];
            let update = branch
                .model_update
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let bytes = self
                .publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .model_memory
                .allocation_bytes
                .iter()
                .try_fold(0u64, |sum, &value| {
                    sum.checked_add(value)
                        .ok_or(SemanticTransitionError::GenerationExhausted)
                })?;
            let binding = &branch
                .continuation
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?
                .text_binding;
            let cache = self.steps[&binding._witness.reader_token].content[binding._witness.index]
                .tensors
                .iter()
                .filter(|v| matches!(v.layout.role, 4 | 5))
                .try_fold(0u64, |sum, tensor| {
                    sum.checked_add(tensor_layout_bytes(&tensor.layout)? as u64)
                        .ok_or(SemanticTransitionError::GenerationExhausted)
                })?;
            (
                view.selection(),
                view.canaries(),
                update.canary_results.view(),
                update.forward_receipts.view(),
                prepared
                    .model_work
                    .as_ref()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
                    .actual
                    .view(),
                bytes,
                cache,
            )
        };
        let selection = self.prepared_completion_read(selection, |o| {
            &mut o.steps.entry(step.token).or_default().control_selection
        })?[0];
        let canaries = self
            .prepared_completion_read(canaries, |o| {
                &mut o.steps.entry(step.token).or_default().control_canaries
            })?
            .try_into()
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        let results = self
            .prepared_completion_read(results, |o| {
                &mut o.steps.entry(step.token).or_default().canary_results
            })?
            .try_into()
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        let forwards = self
            .prepared_completion_read(forwards, |o| {
                &mut o.steps.entry(step.token).or_default().control_forwards
            })?
            .try_into()
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        let actual = self.prepared_completion_read(actual, |o| {
            &mut o.steps.entry(step.token).or_default().control_actual
        })?;
        let calls = if state.execution_work.overflow == 0 {
            Some(
                self.steps[&step.token]
                    .prepared
                    .as_ref()
                    .expect("original Control")
                    .model_work
                    .as_ref()
                    .expect("original model work")
                    .completed_model_calls(&actual, true)?,
            )
        } else {
            None
        };
        let refusal = if state.status == 0 {
            None
        } else {
            Some(refused_terminal(
                state,
                parent.rng_binding()?,
                SemanticActionCatalogue::current().binding(),
                self.binding().digest,
                canary_refusal,
                capacity_refusal,
            )?)
        };
        let complete = matches!(state.status, 0 | 18);
        let observation = ControlFacts {
            context,
            parent: SemanticPublishedIdentity {
                instance: parent.instance,
                word: parent.publication_word,
                logical_digest: parent.logical_digest,
            },
            generation: parent.model_generation,
            geometry: parent.model_geometry_digest,
            status: state.status,
            work: [
                state.execution_work.raw,
                state.execution_work.model_once,
                state.execution_work.native_attempt,
            ],
            calls,
            copy: complete.then_some(model_bytes),
            cache: complete.then_some(cache_bytes),
            selection,
            canaries,
            results,
            forwards,
            refusal: canary_refusal,
            capacity: capacity_refusal,
        }
        .encode()?;
        self.retain_control_result(step, observation.clone())?;
        Ok(match refusal {
            Some(refusal) => SemanticTransitionOutcome::Refused(refusal),
            None => SemanticTransitionOutcome::Discarded(observation),
        })
    }

    /// Project the single native receipt; no new execution, model or authority.
    pub fn prepared_control_measurements(
        &mut self,
        step: &SemanticPreparedStep,
    ) -> Result<Option<Value>, SemanticTransitionError> {
        self.checked_prepared_step(step, false)?;
        let Some(observation) = self
            .steps
            .get(&step.token)
            .and_then(|v| v.control_receipt.clone())
        else {
            return Ok(None);
        };
        Ok(Some(self.control_observation_projection(&observation)?))
    }
}

impl SemanticTransitionSession {
    pub(super) fn control_observation_projection(
        &self,
        observation: &SemanticControlObservation,
    ) -> Result<Value, SemanticTransitionError> {
        let mut result = observation.projection(self.control_retirement_material(observation)?)?;
        result["source_cohort"] = self.control_source_projection(observation)?;
        Ok(result)
    }
}
