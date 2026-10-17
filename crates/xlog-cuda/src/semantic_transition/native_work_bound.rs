use super::*;

impl OriginalNativeCommand {
    pub(super) fn run_dma(
        &mut self,
        domain: &ResidentExecutionDomain,
        poisoned: &mut bool,
        recorder: LaunchRecorder,
        allowance: Option<&Arc<Mutex<ColdNativeAllowance>>>,
        bytes: usize,
        operation: impl FnOnce(&CudaEnqueue<'_>, &mut bool, &mut bool) -> Result<(), XlogError>,
    ) -> Result<(), SemanticTransitionError> {
        if !self.entered {
            if let Some(allowance) = allowance {
                allowance
                    .lock()
                    .map_err(|_| publication_input_error("original cold allowance is poisoned"))?
                    .require_submitted_dma(bytes, self.dma_recorded)?;
            }
        }
        let result = self.run(domain, poisoned, recorder, operation);
        // A successful driver submission remains expenditure when the original
        // commit or completion fence subsequently fails. A preflight refusal
        // has submitted=false and cannot manufacture a reached copy.
        if self.submitted && !self.dma_recorded {
            if let Some(allowance) = allowance {
                allowance
                    .lock()
                    .map_err(|_| publication_input_error("original cold allowance is poisoned"))?
                    .record_submitted_dma(bytes, &mut self.dma_recorded)?;
            }
            self.dma_recorded = true;
        }
        result
    }

    pub(super) fn claim_model_contract_guard(
        &mut self,
        allowance: &Arc<Mutex<ColdNativeAllowance>>,
        ceiling: [u64; 9],
        source: Option<&Arc<prepared_source_content::PreparedSourceContentOwner>>,
    ) -> Result<(), SemanticTransitionError> {
        if let Some(original) = &self.allowance_claim {
            if !original.ptr_eq(&Arc::downgrade(allowance)) {
                return Err(publication_input_error(
                    "original native guard changed its admitted report",
                ));
            }
            return Ok(());
        }
        if allowance
            .lock()
            .map_err(|_| publication_input_error("original model guard allowance is poisoned"))?
            .claim_model_contract_guard(ceiling, source)?
        {
            self.allowance_claim = Some(Arc::downgrade(allowance));
        }
        Ok(())
    }
}

pub(super) struct OriginalContentBatch {
    pub(super) verify: bool,
    pub(super) retirement: bool,
    pub(super) cursor: usize,
    pub(super) context_prefill_occurrence: Option<usize>,
    pub(super) commands: Vec<OriginalNativeCommand>,
    // The original report and child graph retain the strong owner. Keeping
    // only its issued identity here avoids a cohort -> batch -> cohort cycle.
    pub(super) allowance: Option<std::sync::Weak<Mutex<ColdNativeAllowance>>>,
    pub(super) source_occurrence: Option<super::source_content_occurrences::SourceContentClaim>,
    pub(super) source_native_guard:
        Option<super::source_content_occurrences::SourceNativeContentClaim>,
}

impl OriginalContentBatch {
    pub(super) fn new(
        domain: &ResidentExecutionDomain,
        verify: bool,
        count: usize,
    ) -> Result<Self, SemanticTransitionError> {
        let commands = (0..count)
            .map(|_| OriginalNativeCommand::new(domain))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            verify,
            retirement: false,
            cursor: 0,
            context_prefill_occurrence: None,
            commands,
            allowance: None,
            source_occurrence: None,
            source_native_guard: None,
        })
    }
}

impl TensorContentBuffers {
    pub(super) fn retain_source_native_guard_batch(
        &self,
        domain: &ResidentExecutionDomain,
        claim: &super::source_content_occurrences::SourceNativeContentClaim,
        allowance: &Arc<Mutex<ColdNativeAllowance>>,
    ) -> Result<(), SemanticTransitionError> {
        if self
            .native_work_ceiling(true)?
            .iter()
            .zip(claim.native_work_ceiling())
            .any(|(actual, upper)| *actual > upper)
        {
            return Err(publication_input_error(
                "SOURCE native consumer exceeded its frozen original producer geometry",
            ));
        }
        let mut original = self.original_cold.lock().map_err(|_| {
            publication_input_error("original SOURCE native guard custody is poisoned")
        })?;
        if let Some(batch) = original.as_ref() {
            if !batch.verify
                || batch.source_occurrence.is_some()
                || batch
                    .source_native_guard
                    .as_ref()
                    .is_none_or(|issued| !claim.same_claim(issued))
            {
                return Err(publication_input_error(
                    "SOURCE native consumer changed its original retained commands",
                ));
            }
            return Ok(());
        }
        let count = self
            .tensors
            .len()
            .checked_add(usize::from(matches!(
                self.seals,
                TensorContentSeals::Model(_)
            )))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut batch = OriginalContentBatch::new(domain, true, count)?;
        batch.retirement = claim.is_retirement();
        batch.source_native_guard = Some(claim.clone());
        batch.allowance = Some(Arc::downgrade(allowance));
        *original = Some(batch);
        Ok(())
    }
    pub(super) fn retain_source_feedback_batch(
        &self,
        domain: &ResidentExecutionDomain,
        owner: &Arc<prepared_source_content::PreparedSourceContentOwner>,
    ) -> Result<(), SemanticTransitionError> {
        owner.require_feedback_content(self)?;
        owner
            .allowance
            .lock()
            .map_err(|_| publication_input_error("original SOURCE allowance is poisoned"))?
            .require_source_content_admitted(owner)?;
        let mut original = self.original_cold.lock().map_err(|_| {
            publication_input_error("original SOURCE feedback command custody is poisoned")
        })?;
        if original.is_none() {
            let mut batch = OriginalContentBatch::new(domain, false, self.tensors.len())?;
            batch.allowance = Some(Arc::downgrade(&owner.allowance));
            *original = Some(batch);
        } else if original.as_ref().is_some_and(|batch| {
            batch.verify
                || batch.source_occurrence.is_some()
                || batch
                    .allowance
                    .as_ref()
                    .is_none_or(|allowance| !allowance.ptr_eq(&Arc::downgrade(&owner.allowance)))
        }) {
            return Err(publication_input_error(
                "SOURCE feedback changed its original retained seal commands",
            ));
        }
        Ok(())
    }

    pub(super) fn retain_source_occurrence_batch(
        &self,
        domain: &ResidentExecutionDomain,
        claim: &super::source_content_occurrences::SourceContentClaim,
        allowance: &Arc<Mutex<ColdNativeAllowance>>,
        verify: bool,
    ) -> Result<(), SemanticTransitionError> {
        if self
            .native_work_ceiling(verify)?
            .iter()
            .zip(claim.native_work_ceiling())
            .any(|(actual, upper)| *actual > upper)
        {
            return Err(publication_input_error(
                "SOURCE content exceeded its frozen original native geometry ceiling",
            ));
        }
        let mut original = self
            .original_cold
            .lock()
            .map_err(|_| publication_input_error("original SOURCE content custody is poisoned"))?;
        if let Some(batch) = original.as_ref() {
            if batch.verify != verify
                || batch
                    .source_occurrence
                    .as_ref()
                    .is_none_or(|issued| !claim.same_claim(issued))
            {
                return Err(publication_input_error(
                    "SOURCE content changed its retained original command claim",
                ));
            }
            return Ok(());
        }
        let count = self
            .tensors
            .len()
            .checked_add(usize::from(matches!(
                self.seals,
                TensorContentSeals::Model(_)
            )))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut batch = OriginalContentBatch::new(domain, verify, count)?;
        batch.source_occurrence = Some(claim.clone());
        batch.allowance = Some(Arc::downgrade(allowance));
        *original = Some(batch);
        Ok(())
    }
}

/// Original finite content occurrences. Other canonical native producers have
/// separate schedules; this subtotal must never stand in for their ceiling.
fn native_tally_initialization_ceiling() -> [u64; 9] {
    [1, 0, 0, 0, 0, 0, 0, 0, 11 * size_of::<u64>() as u64]
}

/// Each admitted row can be reset once and can contain one invocation marker.
/// Initialization also clears all three words of the authentic original rows.
/// These are finite entitlements, not reached-work charges.
pub(super) fn model_scratch_native_work_ceiling(
    capacity: usize,
    initialize: bool,
) -> Result<[u64; 9], SemanticTransitionError> {
    let capacity = u64::try_from(capacity).map_err(|_| native_work_ceiling_overflow())?;
    Ok([
        ceiling_sum([
            ceiling_product(2, capacity)?,
            u64::from(initialize && capacity != 0),
        ])?,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        ceiling_product(
            ceiling_product(if initialize { 9 } else { 6 }, capacity)?,
            size_of::<u64>() as u64,
        )?,
    ])
}

pub(super) fn model_scratch_initialization_ceiling(
    capacity: usize,
) -> Result<[u64; 9], SemanticTransitionError> {
    let capacity = u64::try_from(capacity).map_err(|_| native_work_ceiling_overflow())?;
    Ok([
        u64::from(capacity != 0),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        ceiling_product(ceiling_product(3, capacity)?, size_of::<u64>() as u64)?,
    ])
}

/// Finite geometry of the original fresh graph and its canonical codebooks.
/// These terms are upper bounds, never reached counters or content permission.
pub(super) fn child_constructor_native_work_ceiling(
    capacities: crate::SemanticHypergraphCapacities,
    records: &crate::SemanticAdmissionRecords,
    limits: crate::SemanticAdmissionLimits,
    program: Option<&crate::SemanticProgramAdmission>,
) -> Result<([u64; 9], [u64; 9]), SemanticTransitionError> {
    let (counts, arena_words) = capacities
        .native_work_geometry()
        .map_err(SemanticTransitionError::Semantic)?;
    let [roots, statements, supports, versions] = counts;
    let count = |count: usize| u64::try_from(count).map_err(|_| native_work_ceiling_overflow());
    let predicate_count = count(records.predicates.len())?;
    let record_count = count(records.records.len())?;
    let support_count = count(records.supports.len())?;
    let arguments = records.records.iter().try_fold(0, |total, record| {
        ceiling_sum([total, count(record.arguments.len())?])
    })?;
    let qualifiers = records.records.iter().try_fold(0, |total, record| {
        ceiling_sum([total, count(record.qualifiers.len())?])
    })?;
    let additional = program.map_or(0, |program| u64::from(program.predicate_count));
    // The original derive owner deduplicates these entries; before dedup each
    // argument, statement qualifier bundle and support can contribute once.
    // Every table also contains its genuine NULL row. The empty qualifier
    // bundle is present even when the original admission has no statement.
    let targets = ceiling_sum([predicate_count, additional, 1])?;
    let operands = ceiling_sum([
        arguments,
        additional,
        if program.is_some() { 4 } else { 0 },
        1,
    ])?;
    let qualifier_rows = ceiling_sum([record_count, 2])?;
    let leaves = ceiling_sum([support_count, 1])?;
    // Canonical typed operand encodings fit in sixteen fixed bytes plus their
    // original UTF-8 symbols. The admission's checked UTF-8 limit includes all
    // copied occurrences, so dedup cannot increase this payload extent.
    let payload = ceiling_sum([
        ceiling_product(16, arguments)?,
        count(limits.max_utf8_bytes)?,
        ceiling_product(4, ceiling_sum([record_count, 1])?)?,
        ceiling_product(32, qualifiers)?,
    ])?;
    let words = ceiling_sum([
        model_catalogue::HEADER_WORDS as u64,
        ceiling_product(10, targets)?,
        ceiling_product(6, operands)?,
        ceiling_product(3, qualifier_rows)?,
        ceiling_product(18, leaves)?,
        ceiling_sum([payload, 7])? / 8,
    ])?;
    let catalogue_bytes = ceiling_sum([
        ceiling_product(words, size_of::<u64>() as u64)?,
        ceiling_product(
            COMPONENT_COUNT as u64,
            size_of::<SemanticComponent>() as u64,
        )?,
    ])?;
    let command_bytes = size_of::<crate::semantic_hypergraph::DeviceCommand>() as u64;
    let receipt_bytes =
        size_of::<crate::semantic_hypergraph::SemanticResidentReceiptRecord>() as u64;
    let dma = [
        6,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        ceiling_sum([
            ceiling_product(2, ceiling_sum([command_bytes, receipt_bytes])?)?,
            catalogue_bytes,
        ])?,
    ];
    // Initialize clears the actual arena and fixed receipt/root words, scans
    // every table (including rejected occupied slots), and seals the empty
    // root. The paired admission Snapshot is the existing two-visit projection.
    let mut ceiling = [
        2,
        ceiling_sum([
            ceiling_product(2, roots)?,
            ceiling_product(2, statements)?,
            supports,
            versions,
            3,
        ])?,
        0,
        0,
        0,
        0,
        0,
        2,
        ceiling_sum([
            ceiling_product(arena_words, size_of::<u64>() as u64)?,
            874,
            448,
        ])?,
    ];
    add_native_work_ceiling(&mut ceiling, dma)?;
    native_work_ceiling_units(ceiling)?;
    Ok((ceiling, dma))
}

pub(crate) struct ColdNativeAllowance {
    purpose: SemanticColdModelWorkPurpose,
    content: Option<FrozenColdEvaluationContent>,
    source: Option<FrozenColdSourceContent>,
    resident_reads: Option<FrozenColdResidentReads>,
    submitted_dma: [u64; 9],
    submitted_dma_ceiling: [u64; 9],
    native_initialization_claimed: bool,
    model_input_generations:
        Option<std::sync::Weak<prepared_source_content::OriginalModelInputGenerations>>,
    model_scratch_ceiling: Option<[u64; 9]>,
    prepared_scratch_initialization: [u64; 9],
    model_context_ceiling: Option<[u64; 9]>,
    model_context_prefill: Option<FrozenModelContextPrefillRoster>,
    model_context_prefill_recording: bool,
    model_context_activation_ceiling: Option<[u64; 9]>,
    model_context_activation_claimed: bool,
    model_context_publication_dma: Option<(
        std::sync::Weak<state_restoration::OriginalPublicationPreparation>,
        [u64; 9],
    )>,
    model_context_preparation: Option<FrozenModelContextPreparation>,
    model_context_source_budget: Option<[u64; 3]>,
}

/// Ordered producer geometry frozen before the imported context's first model
/// effect. Actual tensor owners are attached only by that original producer.
pub(super) struct FrozenModelContextPrefillRoster {
    owner: std::sync::Weak<model_context_continuation::ModelContextContinuationOwner>,
    scope: Option<Arc<()>>,
    occurrences: Vec<(Vec<ModelContextPrefillTensorGeometry>, bool, [u64; 9])>,
    cursor: usize,
    input_tokens: usize,
    input_guard_claimed: bool,
    _reservation: SemanticReplayMaterialReservation,
}

#[derive(Clone, Copy)]
pub struct ModelContextPrefillTensorGeometry {
    pub layout: SemanticTensorLayout,
    pub logical_begin: u64,
    pub logical_end: u64,
    pub physical_bytes: usize,
}

pub struct ModelContextPrefillContentGeometry {
    pub tensors: Vec<ModelContextPrefillTensorGeometry>,
    pub verify: bool,
}

/// Structural inference preparation has two original model-roster uploads and
/// one canonical step-input initializer, not an early resident execution.
pub(super) struct FrozenModelContextPreparation {
    pub(super) ceiling: [u64; 9],
    dma: [u64; 9],
    model_roster_bytes: usize,
    model_roster_claimed: u8,
    step_input: Option<([u64; 9], [u64; 9])>,
}

impl FrozenModelContextPreparation {
    pub(super) fn new(
        target: &SemanticTransitionSession,
        step: &SemanticPreparedStep,
    ) -> Result<Self, SemanticTransitionError> {
        let owner = target.checked_prepared_step(step, false)?;
        let storage = target.prepared_publication()?;
        let directory = &storage.bank_templates[0];
        let models = directory
            .iter()
            .filter(|range| matches!(range.role, 18..=20))
            .count();
        let model_roster_bytes = models
            .checked_mul(2 * size_of::<u64>())
            .filter(|bytes| *bytes != 0)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut dma = [
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            u64::try_from(
                model_roster_bytes
                    .checked_mul(2)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?,
            )
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        ];
        let input_pending = match &owner.inputs {
            None => true,
            Some(inputs) => inputs
                .original_preparation
                .lock()
                .map_err(|_| {
                    publication_input_error(
                        "prepared context input initialization custody is poisoned",
                    )
                })?
                .is_none(),
        };
        let step_input = if input_pending {
            let plans = step_input_plans(storage, directory)?;
            Some((
                step_input_producer_native_work_ceiling(storage, &plans, directory)?,
                step_input_dma_ceiling(plans.len())?,
            ))
        } else {
            None
        };
        let mut ceiling = dma;
        if let Some((native, uploads)) = step_input {
            add_native_work_ceiling(&mut ceiling, native)?;
            add_native_work_ceiling(&mut ceiling, uploads)?;
            add_native_work_ceiling(&mut dma, uploads)?;
        }
        native_work_ceiling_units(ceiling)?;
        Ok(Self {
            ceiling,
            dma,
            model_roster_bytes,
            model_roster_claimed: 0,
            step_input,
        })
    }
}

impl FrozenModelContextPrefillRoster {
    pub(super) fn new(
        context: &SemanticModelContextContinuation,
        budget: &SemanticReplayMaterialBudget,
        declarations: &[ModelContextPrefillContentGeometry],
        input_tokens: usize,
    ) -> Result<Self, SemanticTransitionError> {
        let layouts = declarations.iter().try_fold(0usize, |count, declaration| {
            if declaration.tensors.is_empty() {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            for tensor in &declaration.tensors {
                canonical_tensor_layout(tensor.layout)?;
            }
            count
                .checked_add(declaration.tensors.len())
                .ok_or(SemanticTransitionError::GenerationExhausted)
        })?;
        let bytes = declarations
            .len()
            .checked_mul(size_of::<(
                Vec<ModelContextPrefillTensorGeometry>,
                bool,
                [u64; 9],
            )>())
            .and_then(|bytes| {
                layouts
                    .checked_mul(size_of::<ModelContextPrefillTensorGeometry>())
                    .and_then(|layouts| bytes.checked_add(layouts))
            })
            .and_then(|bytes| bytes.checked_add(size_of::<Self>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = budget.reserve_auxiliary(bytes)?;
        let mut occurrences = Vec::new();
        occurrences
            .try_reserve_exact(declarations.len())
            .map_err(|error| runtime_error("context prefill roster allocation", error))?;
        for declaration in declarations {
            let mut ceiling = [0; 9];
            let mut layouts = Vec::new();
            layouts
                .try_reserve_exact(declaration.tensors.len())
                .map_err(|error| {
                    runtime_error("context prefill tensor geometry allocation", error)
                })?;
            for tensor in &declaration.tensors {
                let geometry = ModelContextPrefillTensorGeometry {
                    layout: canonical_tensor_layout(tensor.layout)?,
                    ..*tensor
                };
                add_native_work_ceiling(
                    &mut ceiling,
                    tensor_layout_native_work_ceiling(
                        &geometry.layout,
                        geometry.logical_begin,
                        geometry.logical_end,
                        geometry.physical_bytes,
                        declaration.verify,
                    )?,
                )?;
                layouts.push(geometry);
            }
            occurrences.push((layouts, declaration.verify, ceiling));
        }
        Ok(Self {
            owner: Arc::downgrade(&context.inner),
            scope: None,
            occurrences,
            cursor: 0,
            input_tokens,
            input_guard_claimed: false,
            _reservation: reservation,
        })
    }

    fn ceiling(&self) -> Result<[u64; 9], SemanticTransitionError> {
        let mut total = [0; 9];
        for (_, _, ceiling) in &self.occurrences {
            add_native_work_ceiling(&mut total, *ceiling)?;
        }
        Ok(total)
    }
}

/// Exact original read occurrences, ordered by their enclosing report regions.
/// Geometry remains metadata; only the actual issued producer can claim it.
pub(super) struct FrozenColdResidentReads {
    reads: Vec<(usize, SemanticResidentModelRead, [u64; 9], bool)>,
    readbacks: Vec<Option<(SemanticPublicationReadbackLoan, bool)>>,
    restorations: Vec<(
        SemanticPreparedResidentModelRestoration,
        [u64; 9],
        bool,
        bool,
    )>,
    cursor: usize,
    ceiling: [u64; 9],
    _reservation: Option<SemanticReplayMaterialReservation>,
}

impl FrozenColdResidentReads {
    pub(super) fn new(
        work: &SemanticColdModelWork,
        reads: &[(usize, SemanticResidentModelRead)],
    ) -> Result<Self, SemanticTransitionError> {
        let reservation = if let Some((_, first)) = reads.first() {
            let budget = first.material_budget();
            for (region, read) in reads {
                read.require_report(work, *region)?;
                if !budget.same_pool(&read.material_budget()) {
                    return Err(publication_input_error(
                        "resident read declaration changed its original material pool",
                    ));
                }
            }
            let read_bytes = reads
                .len()
                .checked_mul(
                    size_of::<(usize, SemanticResidentModelRead, [u64; 9], bool)>()
                        + size_of::<Option<(SemanticPublicationReadbackLoan, bool)>>(),
                )
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let restoration_bytes = reads
                .len()
                .checked_mul(size_of::<(
                    SemanticPreparedResidentModelRestoration,
                    [u64; 9],
                    bool,
                    bool,
                )>())
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let bytes = read_bytes
                .checked_add(restoration_bytes)
                .and_then(|bytes| bytes.checked_add(size_of::<Self>()))
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            Some(budget.reserve_auxiliary(bytes)?)
        } else {
            None
        };
        let mut original = Vec::new();
        original
            .try_reserve_exact(reads.len())
            .map_err(|error| runtime_error("resident read declaration reservation", error))?;
        let mut total = [0; 9];
        let mut readbacks = Vec::new();
        readbacks
            .try_reserve_exact(reads.len())
            .map_err(|error| runtime_error("native readback declaration reservation", error))?;
        let mut restorations = Vec::new();
        restorations
            .try_reserve_exact(reads.len())
            .map_err(|error| runtime_error("native restoration declaration reservation", error))?;
        for (index, (region, read)) in reads.iter().enumerate() {
            read.require_report(work, *region)?;
            if index > 0 && reads[index - 1].0 > *region
                || reads[..index]
                    .iter()
                    .any(|(_, earlier)| read.same_occurrence(earlier))
            {
                return Err(publication_input_error(
                    "resident read declaration changed its original order or repeated an occurrence",
                ));
            }
            let ceiling = read.native_work_ceiling()?;
            add_native_work_ceiling(&mut total, ceiling)?;
            add_native_work_ceiling(&mut total, read.checkpoint_readback_ceiling())?;
            readbacks.push(
                read.checkpoint_readback_loan()
                    .map(|loan| (loan.clone(), false)),
            );
            if let Some(issuer) = read.restoration_issuer()? {
                issuer.require_report(work, *region)?;
                if restorations.iter().any(
                    |(original, _, _, _): &(
                        SemanticPreparedResidentModelRestoration,
                        [u64; 9],
                        bool,
                        bool,
                    )| issuer.same_issuer(original),
                ) {
                    return Err(publication_input_error(
                        "native restoration repeated its original issuer",
                    ));
                }
                let dma = issuer.dma_native_work_ceiling()?;
                add_native_work_ceiling(&mut total, dma)?;
                restorations.push((issuer, dma, false, false));
            }
            original.push((*region, read.clone(), ceiling, false));
        }
        native_work_ceiling_units(total)?;
        Ok(Self {
            reads: original,
            readbacks,
            restorations,
            cursor: 0,
            ceiling: total,
            _reservation: reservation,
        })
    }

    fn skip_unentered_region(&mut self, region: usize) -> Result<(), SemanticTransitionError> {
        if region == 2 && self.restorations.iter().any(|(_, _, claimed, _)| *claimed) {
            return Err(publication_input_error(
                "native restoration skip cannot replace an entered original occurrence",
            ));
        }
        if self.reads[..self.cursor]
            .iter()
            .any(|(original, _, _, _)| *original == region)
        {
            return Err(publication_input_error(
                "resident read skip cannot replace an entered original occurrence",
            ));
        }
        for (original, _, _, skipped) in &mut self.reads[self.cursor..] {
            if *original == region {
                *skipped = true;
            }
        }
        if region == 2 {
            for (_, _, _, skipped) in &mut self.restorations {
                *skipped = true;
            }
        }
        self.advance_skipped_prefix();
        Ok(())
    }

    fn advance_skipped_prefix(&mut self) {
        while self
            .reads
            .get(self.cursor)
            .is_some_and(|(_, _, _, skipped)| *skipped)
        {
            self.cursor += 1;
        }
    }

    fn require_complete(&self) -> Result<(), SemanticTransitionError> {
        if self.cursor != self.reads.len()
            || self
                .restorations
                .iter()
                .any(|(_, _, claimed, skipped)| !claimed && !skipped)
        {
            return Err(publication_input_error(
                "successful cold completion requires every non-skipped original resident read",
            ));
        }
        Ok(())
    }
}

/// Enclose original restore copies before their actual input exists. The
/// immutable pool cap bounds payload bytes and the worst legal one-byte chunk;
/// it is never an executed-work charge or a permission to access contents.
pub(super) fn restoration_dma_native_work_ceiling(
    source: &SemanticTransitionSession,
    lease: &SemanticPublishedLease,
    budget: &SemanticReplayMaterialBudget,
) -> Result<[u64; 9], SemanticTransitionError> {
    source.checked_reader(lease)?;
    if source
        .replay_material_budget
        .as_ref()
        .is_none_or(|original| !original.same_pool(budget))
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let storage = source
        .publication
        .as_ref()
        .ok_or(SemanticTransitionError::NotBound)?;
    let (_, task) = source
        .task
        .as_ref()
        .ok_or(SemanticTransitionError::NotBound)?;
    let count = |n: usize| u64::try_from(n).map_err(|_| native_work_ceiling_overflow());
    let bytes = |n: usize, width: usize| ceiling_product(count(n)?, count(width)?);
    let payload = budget.owner.capacity;
    let catalogue = ceiling_sum([
        bytes(task.len(), size_of::<u64>())?,
        bytes(source.device_codebooks.len(), size_of::<u64>())?,
        bytes(
            source.device_components.len(),
            size_of::<SemanticComponent>(),
        )?,
    ])?;
    let queue = lease
        .directory
        .iter()
        .find(|range| range.role == 56 && range.index == 0);
    let append_capacity = queue
        .map(|range| {
            range
                    .length_bytes
                    .checked_sub(size_of::<
                        crate::semantic_training_view::SemanticTrainingReplayAppendHeader,
                    >() as u64)
                    .ok_or(SemanticTransitionError::ObservationMismatch)
                    .map(|bytes| {
                        bytes
                            / size_of::<
                                crate::semantic_training_view::SemanticTrainingReplayAppendEntry,
                            >() as u64
                    })
        })
        .transpose()?
        .unwrap_or(0);
    let mut control_bytes = ceiling_sum([
        2 * size_of::<PublicationBank>() as u64,
        bytes(
            storage.allocations.len(),
            size_of::<PublicationStorageEntry>(),
        )?,
        size_of::<PublicationContract>() as u64,
        bytes(storage.terminals.len(), size_of::<u64>())?,
        bytes(storage.role_counts.len(), size_of::<PublicationRoleCount>())?,
        size_of::<PendingContinuation>() as u64,
        bytes(
            storage.continuation_templates.len(),
            size_of::<PublicationRange>(),
        )?,
        size_of::<PublicationControl>() as u64,
        size_of::<DeviceState>() as u64,
    ])?;
    let mut control_count = 8u64;
    for directory in &storage.bank_templates {
        control_bytes = ceiling_sum([
            control_bytes,
            bytes(directory.len(), size_of::<PublicationRange>())?,
        ])?;
        control_count = ceiling_sum([control_count, u64::from(!directory.is_empty())])?;
    }
    control_count = ceiling_sum([
        control_count,
        u64::from(!storage.terminals.is_empty()),
        u64::from(!storage.continuation_templates.is_empty()),
    ])?;
    let ceiling = [
        ceiling_sum([
            payload,
            3,
            control_count,
            ceiling_product(2, append_capacity)?,
            2,
        ])?,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        ceiling_sum([
            ceiling_product(2, payload)?,
            catalogue,
            control_bytes,
            ceiling_product(
                append_capacity,
                size_of::<crate::semantic_training_view::TrainingViewRowDescriptor>() as u64,
            )?,
            size_of::<PublicationControl>() as u64,
            size_of::<PublicationHeader>() as u64,
        ])?,
    ];
    native_work_ceiling_units(ceiling)?;
    Ok(ceiling)
}

pub(super) struct FrozenColdEvaluationContent {
    invocation: Arc<()>,
    preparation_available: bool,
    pub(super) cohort: Arc<SemanticEvaluationCohort>,
    pub(super) output: Vec<PreparedSemanticTensor>,
    pub(super) objective: Vec<PreparedSemanticTensor>,
    model_ceiling: [u64; 9],
    ceiling: [u64; 9],
    remaining: [[u8; 2]; 4],
    model_snapshot: Option<[u64; 9]>,
    model_contract_guard: Option<[u64; 9]>,
    step_input_preparation: Option<([u64; 9], [u64; 9])>,
}

impl ColdNativeAllowance {
    pub(super) fn new(purpose: SemanticColdModelWorkPurpose) -> Self {
        Self {
            purpose,
            content: None,
            source: None,
            resident_reads: None,
            submitted_dma: [0; 9],
            submitted_dma_ceiling: [0; 9],
            native_initialization_claimed: false,
            model_input_generations: None,
            model_scratch_ceiling: None,
            prepared_scratch_initialization: [0; 9],
            model_context_ceiling: None,
            model_context_prefill: None,
            model_context_prefill_recording: false,
            model_context_activation_ceiling: None,
            model_context_activation_claimed: false,
            model_context_publication_dma: None,
            model_context_preparation: None,
            model_context_source_budget: None,
        }
    }

    pub(super) fn freeze_model_context_native_work(
        &mut self,
        ceiling: [u64; 9],
    ) -> Result<(), SemanticTransitionError> {
        if self.model_context_ceiling.is_some()
            || self.model_scratch_ceiling.is_some()
            || self.native_initialization_claimed
        {
            return Err(publication_input_error(
                "model context native work requires its sole original frozen preparation roster",
            ));
        }
        native_work_ceiling_units(ceiling)?;
        self.model_context_ceiling = Some(ceiling);
        Ok(())
    }

    pub(super) fn freeze_model_context_preparation(
        &mut self,
        roster: FrozenModelContextPreparation,
    ) -> Result<(), SemanticTransitionError> {
        if self.purpose != SemanticColdModelWorkPurpose::PreparedActorRefresh
            || self.model_context_ceiling != Some(roster.ceiling)
            || self.model_context_preparation.is_some()
            || self.native_initialization_claimed
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        add_native_work_ceiling(&mut self.submitted_dma_ceiling, roster.dma)?;
        self.model_context_preparation = Some(roster);
        Ok(())
    }

    pub(super) fn claim_model_context_prepared_model_roster(
        &mut self,
        bank: usize,
        bytes: usize,
    ) -> Result<(), SemanticTransitionError> {
        let roster = self
            .model_context_preparation
            .as_mut()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if bank > 1
            || bytes != roster.model_roster_bytes
            || roster.model_roster_claimed & (1 << bank) != 0
        {
            return Err(publication_input_error(
                "prepared context model roster changed or repeated its original bank upload",
            ));
        }
        roster.model_roster_claimed |= 1 << bank;
        Ok(())
    }

    pub(super) fn freeze_model_context_report_dma(
        &mut self,
        events: usize,
    ) -> Result<(), SemanticTransitionError> {
        if self.model_context_ceiling.is_none() || self.model_scratch_ceiling.is_some() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let bytes = events
            .checked_mul(size_of::<ModelWorkEvent>())
            .ok_or_else(native_work_ceiling_overflow)?;
        let dma = [
            1 + u64::from(events != 0),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            ceiling_sum([
                u64::try_from(bytes).map_err(|_| native_work_ceiling_overflow())?,
                15 * size_of::<u64>() as u64,
            ])?,
        ];
        let mut ceiling = self
            .model_context_ceiling
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        add_native_work_ceiling(&mut ceiling, dma)?;
        add_native_work_ceiling(&mut self.submitted_dma_ceiling, dma)?;
        self.model_context_ceiling = Some(ceiling);
        Ok(())
    }

    pub(super) fn is_model_context_report(&self) -> bool {
        self.model_context_ceiling.is_some()
            || self.model_context_prefill.is_some()
            || self.model_context_preparation.is_some()
            || self.model_context_activation_ceiling.is_some()
    }

    pub(super) fn require_model_context_source_budget(
        &self,
        model: [u64; 3],
    ) -> Result<(), SemanticTransitionError> {
        if let Some(budget) = self.model_context_source_budget {
            let total = native_work_ceiling_units(self.content_ceiling()?)?
                .checked_add(model[0])
                .ok_or_else(native_work_ceiling_overflow)?;
            if total > budget[0] || model[2] > budget[2] {
                return Err(publication_input_error(
                    "context prefill exceeds its original complete SOURCE instruction allowance",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn freeze_model_context_activation_dma(
        &mut self,
        kernel: [u64; 9],
        dma: [u64; 9],
    ) -> Result<(), SemanticTransitionError> {
        if self.purpose != SemanticColdModelWorkPurpose::PrivateModelPreparation
            || self.model_context_ceiling != Some(kernel)
            || self.native_initialization_claimed
            || self.model_context_activation_ceiling.is_some()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let mut total = kernel;
        add_native_work_ceiling(&mut total, dma)?;
        native_work_ceiling_units(total)?;
        add_native_work_ceiling(&mut self.submitted_dma_ceiling, dma)?;
        self.model_context_ceiling = Some(total);
        self.model_context_activation_ceiling = Some(kernel);
        Ok(())
    }

    pub(super) fn freeze_model_context_material_dma(
        &mut self,
        dma: [u64; 9],
    ) -> Result<(), SemanticTransitionError> {
        if !matches!(
            self.purpose,
            SemanticColdModelWorkPurpose::PrivateCheckpoint
                | SemanticColdModelWorkPurpose::PrivateRestore
        ) || self.model_context_ceiling.is_none()
            || self.native_initialization_claimed
            || self.model_scratch_ceiling.is_some()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        native_work_ceiling_units(dma)?;
        add_native_work_ceiling(&mut self.submitted_dma_ceiling, dma)
    }

    pub(super) fn claim_model_context_activation(
        &mut self,
        kernel: [u64; 9],
    ) -> Result<(), SemanticTransitionError> {
        if self.model_context_activation_ceiling != Some(kernel)
            || !self.native_initialization_claimed
            || self.model_context_activation_claimed
        {
            return Err(publication_input_error(
                "context activation changed or repeated its original native seal occurrence",
            ));
        }
        self.model_context_activation_claimed = true;
        Ok(())
    }

    pub(super) fn admit_model_context_publication_dma(
        &mut self,
        context: &SemanticModelContextContinuation,
        preparer: &Arc<state_restoration::OriginalPublicationPreparation>,
        dma: [u64; 9],
        model: [u64; 3],
        budget: [u64; 3],
    ) -> Result<(), SemanticTransitionError> {
        let roster = self
            .model_context_prefill
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if self.purpose != SemanticColdModelWorkPurpose::PrivatePrefix
            || !self.model_context_prefill_recording
            || !roster.owner.ptr_eq(&Arc::downgrade(&context.inner))
            || self.model_context_source_budget != Some(budget)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if let Some((original, geometry)) = &self.model_context_publication_dma {
            return if original.ptr_eq(&Arc::downgrade(preparer)) && *geometry == dma {
                Ok(())
            } else {
                Err(SemanticTransitionError::ObservationMismatch)
            };
        }
        let mut ceiling = self.content_ceiling()?;
        add_native_work_ceiling(&mut ceiling, dma)?;
        let work = native_work_ceiling_units(ceiling)?
            .checked_add(model[0])
            .ok_or_else(native_work_ceiling_overflow)?;
        if work > budget[0] || model[2] > budget[2] {
            return Err(publication_input_error(
                "imported publication preparation exceeds its original SOURCE instruction budget",
            ));
        }
        let mut context_ceiling = self
            .model_context_ceiling
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        add_native_work_ceiling(&mut context_ceiling, dma)?;
        let mut dma_ceiling = self.submitted_dma_ceiling;
        add_native_work_ceiling(&mut dma_ceiling, dma)?;
        self.model_context_ceiling = Some(context_ceiling);
        self.submitted_dma_ceiling = dma_ceiling;
        self.model_context_publication_dma = Some((Arc::downgrade(preparer), dma));
        Ok(())
    }

    pub(super) fn freeze_model_context_prefill(
        &mut self,
        roster: FrozenModelContextPrefillRoster,
        dma: [u64; 9],
        input_guard: [u64; 9],
        budget: [u64; 3],
    ) -> Result<(), SemanticTransitionError> {
        if self.purpose != SemanticColdModelWorkPurpose::PrivatePrefix
            || self.model_context_prefill.is_some()
            || self.model_scratch_ceiling.is_some()
            || self.native_initialization_claimed
        {
            return Err(publication_input_error(
                "context prefill requires its unopened original SOURCE preparation allowance",
            ));
        }
        let mut ceiling = roster.ceiling()?;
        add_native_work_ceiling(&mut ceiling, dma)?;
        add_native_work_ceiling(&mut ceiling, input_guard)?;
        self.freeze_model_context_native_work(ceiling)?;
        add_native_work_ceiling(&mut self.submitted_dma_ceiling, dma)?;
        self.model_context_prefill = Some(roster);
        self.model_context_source_budget = Some(budget);
        Ok(())
    }

    pub(super) fn freeze_model_context_activation_prefill(
        &mut self,
        roster: FrozenModelContextPrefillRoster,
    ) -> Result<(), SemanticTransitionError> {
        if self.purpose != SemanticColdModelWorkPurpose::PrivateModelPreparation
            || self.model_context_activation_ceiling.is_none()
            || self.native_initialization_claimed
            || self.model_context_prefill.is_some()
        {
            return Err(publication_input_error(
                "activation content must precede the original complete plan admission",
            ));
        }
        let mut ceiling = self
            .model_context_ceiling
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        add_native_work_ceiling(&mut ceiling, roster.ceiling()?)?;
        native_work_ceiling_units(ceiling)?;
        self.model_context_ceiling = Some(ceiling);
        self.model_context_prefill = Some(roster);
        Ok(())
    }

    pub(super) fn require_model_context_activation_prefill(
        &self,
    ) -> Result<(), SemanticTransitionError> {
        if self.model_context_prefill.is_none() {
            return Err(publication_input_error(
                "activation requires its complete original prefill verification roster",
            ));
        }
        Ok(())
    }

    pub(super) fn claim_model_context_prefill_content(
        &mut self,
        context: &SemanticModelContextContinuation,
        scope: &Arc<()>,
        occurrence: usize,
        tensors: &TensorContentBuffers,
        verify: bool,
    ) -> Result<(), SemanticTransitionError> {
        if !self.model_context_prefill_recording {
            return Err(publication_input_error(
                "context prefill content requires its initialized original SOURCE recording report",
            ));
        }
        let roster = self.model_context_prefill.as_mut().ok_or_else(|| {
            publication_input_error("context prefill precedes its original frozen producer roster")
        })?;
        let (layouts, original_verify, ceiling) =
            roster.occurrences.get(roster.cursor).ok_or_else(|| {
                publication_input_error("context prefill exceeded its original producer order")
            })?;
        if occurrence != roster.cursor
            || !roster.owner.ptr_eq(&Arc::downgrade(&context.inner))
            || roster
                .scope
                .as_ref()
                .is_some_and(|original| !Arc::ptr_eq(original, scope))
            || *original_verify != verify
            || matches!(tensors.seals, TensorContentSeals::Model(_))
            || layouts.len() != tensors.tensors.len()
            || layouts
                .iter()
                .zip(&tensors.tensors)
                .any(|(geometry, tensor)| {
                    geometry.layout != tensor.layout
                        || geometry.logical_begin != tensor.logical_begin
                        || geometry.logical_end != tensor.logical_end
                        || geometry.physical_bytes
                            != tensor.source.as_ref().map_or(0, DeviceMemoryView::len)
                })
            || tensors
                .native_work_ceiling(verify)?
                .iter()
                .zip(*ceiling)
                .any(|(actual, upper)| *actual > upper)
        {
            return Err(publication_input_error(
                "context prefill changed its original scope, layout, order or native ceiling",
            ));
        }
        if roster.scope.is_none() {
            roster.scope = Some(Arc::clone(scope));
        }
        roster.cursor += 1;
        Ok(())
    }

    pub(super) fn set_model_context_prefill_recording(&mut self, recording: bool) {
        self.model_context_prefill_recording = recording && self.model_context_prefill.is_some();
    }

    pub(super) fn claim_model_context_prefill_input_guard(
        &mut self,
        context: &SemanticModelContextContinuation,
        scope: &Arc<()>,
        input_tokens: usize,
    ) -> Result<(), SemanticTransitionError> {
        let roster = self
            .model_context_prefill
            .as_mut()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if !self.model_context_prefill_recording
            || !roster.owner.ptr_eq(&Arc::downgrade(&context.inner))
            || roster
                .scope
                .as_ref()
                .is_some_and(|original| !Arc::ptr_eq(original, scope))
            || input_tokens != roster.input_tokens
            || roster.input_guard_claimed
        {
            return Err(publication_input_error(
                "context prefill input guard requires its same initialized original report, scope and single occurrence",
            ));
        }
        if roster.scope.is_none() {
            roster.scope = Some(Arc::clone(scope));
        }
        roster.input_guard_claimed = true;
        Ok(())
    }

    pub(super) fn is_evaluation(&self) -> bool {
        matches!(
            self.purpose,
            SemanticColdModelWorkPurpose::SourceEvaluation
                | SemanticColdModelWorkPurpose::PrivateEvaluation
        )
    }

    pub(super) fn register_model_input_generations(
        &mut self,
        original: &Arc<prepared_source_content::OriginalModelInputGenerations>,
    ) -> Result<(), SemanticTransitionError> {
        if self.model_input_generations.is_some() {
            return Err(publication_input_error(
                "original model input generation roster is registered once",
            ));
        }
        self.model_input_generations = Some(Arc::downgrade(original));
        Ok(())
    }

    pub(super) fn model_input_generations(
        &self,
        instruction: &Arc<[u8]>,
    ) -> Result<
        Option<Arc<prepared_source_content::OriginalModelInputGenerations>>,
        SemanticTransitionError,
    > {
        let Some(original) = &self.model_input_generations else {
            return Ok(None);
        };
        let original = original.upgrade().ok_or_else(|| {
            publication_input_error("original model input generation roster has retired")
        })?;
        original.require_instruction(instruction)?;
        Ok(Some(original))
    }

    pub(super) fn freeze_model_scratch(
        &mut self,
        ceiling: [u64; 9],
    ) -> Result<(), SemanticTransitionError> {
        if self.model_scratch_ceiling.is_some() {
            return Err(publication_input_error(
                "model scratch requires its sole original complete cold admission",
            ));
        }
        native_work_ceiling_units(ceiling)?;
        self.model_scratch_ceiling = Some(ceiling);
        Ok(())
    }

    pub(super) fn require_unfrozen_prepared_scratch(&self) -> Result<(), SemanticTransitionError> {
        if self.model_scratch_ceiling.is_some() {
            return Err(publication_input_error(
                "prepared scratch allocation must precede its original complete cold admission",
            ));
        }
        Ok(())
    }

    /// The sole native allocator calls this once for its original installed
    /// work bank, including children sharing this authentic Prefix allowance.
    pub(super) fn register_prepared_scratch_initialization(
        &mut self,
        capacity: usize,
    ) -> Result<(), SemanticTransitionError> {
        self.require_unfrozen_prepared_scratch()?;
        add_native_work_ceiling(
            &mut self.prepared_scratch_initialization,
            model_scratch_initialization_ceiling(capacity)?,
        )
    }

    pub(super) fn is_source_prefix(&self) -> bool {
        self.purpose == SemanticColdModelWorkPurpose::PrivatePrefix
    }

    pub(super) fn require_source_preparation(&self) -> Result<(), SemanticTransitionError> {
        if self.purpose != SemanticColdModelWorkPurpose::PrivatePrefix || self.source.is_some() {
            return Err(publication_input_error(
                "SOURCE structural preparation requires its original Prefix allowance",
            ));
        }
        Ok(())
    }

    pub(super) fn freeze_source_content(
        &mut self,
        source: FrozenColdSourceContent,
    ) -> Result<(), SemanticTransitionError> {
        self.require_source_preparation()?;
        self.source = Some(source);
        Ok(())
    }

    pub(super) fn freeze_resident_reads(
        &mut self,
        reads: FrozenColdResidentReads,
    ) -> Result<(), SemanticTransitionError> {
        if self.resident_reads.is_some() {
            return Err(publication_input_error(
                "resident reads belong to the sole original complete cold admission",
            ));
        }
        self.resident_reads = Some(reads);
        Ok(())
    }

    pub(super) fn has_resident_reads(&self) -> bool {
        self.resident_reads
            .as_ref()
            .is_some_and(|reads| !reads.reads.is_empty())
    }

    /// Called only when the native enclosing report proves this whole region
    /// never entered. Closed cleanup and partial refusal do not satisfy it.
    pub(super) fn skip_unentered_resident_reads(
        &mut self,
        region: usize,
    ) -> Result<(), SemanticTransitionError> {
        if let Some(reads) = &mut self.resident_reads {
            reads.skip_unentered_region(region)?;
        }
        Ok(())
    }

    pub(super) fn claim_resident_read(
        &mut self,
        owner: &Arc<super::resident_model_memory::ResidentModelMemoryOwner>,
        region: usize,
        ceiling: [u64; 9],
    ) -> Result<(), SemanticTransitionError> {
        let reads = self.resident_reads.as_mut().ok_or_else(|| {
            publication_input_error("resident read precedes its original finite cold admission")
        })?;
        let (original_region, read, original_ceiling, _) =
            reads.reads.get(reads.cursor).ok_or_else(|| {
                publication_input_error("resident read exceeded its original ordered roster")
            })?;
        if *original_region != region || *original_ceiling != ceiling {
            return Err(publication_input_error(
                "resident read changed its original region, order or native geometry ceiling",
            ));
        }
        read.claim_actual(owner, ceiling)?;
        reads.cursor += 1;
        reads.advance_skipped_prefix();
        Ok(())
    }

    /// Transfer this exact original read's already frozen DMA entitlement to
    /// its sole retained observation. This is not a second total-work charge.
    pub(super) fn claim_checkpoint_readback(
        &mut self,
        loan: &SemanticPublicationReadbackLoan,
    ) -> Result<(), SemanticTransitionError> {
        let reads = self.resident_reads.as_mut().ok_or_else(|| {
            publication_input_error("checkpoint readback precedes its original finite admission")
        })?;
        let index = reads
            .readbacks
            .iter()
            .position(|entry| {
                entry
                    .as_ref()
                    .is_some_and(|(original, _)| original.same_loan(loan))
            })
            .ok_or_else(|| {
                publication_input_error(
                    "checkpoint readback is outside its original resident roster",
                )
            })?;
        if index >= reads.cursor || reads.reads[index].3 {
            return Err(publication_input_error(
                "checkpoint readback has not entered its original resident occurrence",
            ));
        }
        let (original, claimed) = reads.readbacks[index]
            .as_mut()
            .expect("original admitted readback loan");
        if *claimed || original.native_work_ceiling() != loan.native_work_ceiling() {
            return Err(publication_input_error(
                "checkpoint readback repeated or changed its original DMA entitlement",
            ));
        }
        add_native_work_ceiling(
            &mut self.submitted_dma_ceiling,
            original.submitted_dma_ceiling(),
        )?;
        *claimed = true;
        Ok(())
    }

    /// A caller must first prove the original live report and waiting native
    /// restore prelude. This consumes the occurrence frozen from its real read.
    pub(super) fn claim_resident_restoration(
        &mut self,
        issuer: &SemanticPreparedResidentModelRestoration,
    ) -> Result<(), SemanticTransitionError> {
        let reads = self.resident_reads.as_mut().ok_or_else(|| {
            publication_input_error(
                "native restoration precedes its original finite cold admission",
            )
        })?;
        let (_, region_read, _, _) = reads.reads.get(reads.cursor).ok_or_else(|| {
            publication_input_error("native restoration exceeded its original read order")
        })?;
        let expected = region_read.restoration_issuer()?.ok_or_else(|| {
            publication_input_error(
                "native restoration precedes a different original read occurrence",
            )
        })?;
        if !issuer.same_issuer(&expected) {
            return Err(publication_input_error(
                "native restoration replaced its original issuer",
            ));
        }
        let (original, ceiling, claimed, skipped) = reads
            .restorations
            .iter_mut()
            .find(|(original, _, _, _)| issuer.same_issuer(original))
            .ok_or_else(|| {
                publication_input_error("native restoration is outside its original finite roster")
            })?;
        if *claimed || *skipped || original.dma_native_work_ceiling()? != *ceiling {
            return Err(publication_input_error(
                "native restoration repeated or changed its original finite occurrence",
            ));
        }
        add_native_work_ceiling(&mut self.submitted_dma_ceiling, *ceiling)?;
        *claimed = true;
        Ok(())
    }

    pub(super) fn require_resident_reads_complete(&self) -> Result<(), SemanticTransitionError> {
        if self
            .model_context_preparation
            .as_ref()
            .is_some_and(|roster| roster.model_roster_claimed != 3 || roster.step_input.is_some())
        {
            return Err(publication_input_error(
                "prepared context has not consumed its original structural native producer roster",
            ));
        }
        if self.model_context_activation_ceiling.is_some() && !self.model_context_activation_claimed
        {
            return Err(publication_input_error(
                "context activation has not entered its original native seal occurrence",
            ));
        }
        if self.model_context_prefill.as_ref().is_some_and(|roster| {
            roster.cursor != roster.occurrences.len() || !roster.input_guard_claimed
        }) {
            return Err(publication_input_error(
                "context prefill has not completed its original frozen producer roster",
            ));
        }
        self.resident_reads
            .as_ref()
            .ok_or_else(|| {
                publication_input_error(
                    "cold completion lost its original frozen resident read roster",
                )
            })?
            .require_complete()
    }

    pub(super) fn require_source_content_admitted(
        &self,
        owner: &Arc<prepared_source_content::PreparedSourceContentOwner>,
    ) -> Result<(), SemanticTransitionError> {
        let source = self.source.as_ref().ok_or_else(|| {
            publication_input_error(
                "SOURCE content requires its complete original finite admission",
            )
        })?;
        if !source
            .recipes
            .iter()
            .any(|recipe| recipe.owner.ptr_eq(&Arc::downgrade(owner)) && recipe.preparation_claimed)
        {
            return Err(publication_input_error(
                "SOURCE content has not entered its sole admitted original preparation",
            ));
        }
        Ok(())
    }

    pub(super) fn claim_source_preparation(
        &mut self,
        owner: &Arc<prepared_source_content::PreparedSourceContentOwner>,
    ) -> Result<(), SemanticTransitionError> {
        let source = self.source.as_mut().ok_or_else(|| {
            publication_input_error(
                "SOURCE preparation precedes its complete original finite admission",
            )
        })?;
        let recipe = source
            .recipes
            .iter_mut()
            .find(|recipe| recipe.owner.ptr_eq(&Arc::downgrade(owner)))
            .ok_or_else(|| {
                publication_input_error(
                    "SOURCE preparation is outside its original admitted roster",
                )
            })?;
        if recipe.preparation_claimed {
            return Err(publication_input_error(
                "SOURCE preparation cannot replace its sole original occurrence",
            ));
        }
        add_native_work_ceiling(&mut self.submitted_dma_ceiling, recipe.dma_ceiling)?;
        recipe.preparation_claimed = true;
        Ok(())
    }

    pub(super) fn claim_source_resident_guard(
        &mut self,
        owner: &Arc<super::resident_model_memory::ResidentModelMemoryOwner>,
        ceiling: [u64; 9],
    ) -> Result<(), SemanticTransitionError> {
        let source = self.source.as_mut().ok_or_else(|| {
            publication_input_error(
                "resident SOURCE guard precedes its complete original finite admission",
            )
        })?;
        let recipe = source
            .recipes
            .iter_mut()
            .find(|recipe| recipe.resident.ptr_eq(&Arc::downgrade(owner)))
            .ok_or_else(|| {
                publication_input_error("resident guard is outside its original SOURCE roster")
            })?;
        if !recipe.preparation_claimed
            || recipe.resident_claimed
            || recipe.resident_ceiling != ceiling
        {
            return Err(publication_input_error(
                "resident SOURCE guard changed or repeated its original finite producer",
            ));
        }
        recipe.resident_claimed = true;
        Ok(())
    }

    pub(super) fn freeze_content(
        &mut self,
        content: FrozenColdEvaluationContent,
    ) -> Result<(), SemanticTransitionError> {
        if !self.is_evaluation() || self.content.is_some() {
            return Err(publication_input_error(
                "evaluation content belongs to its sole original cold admission",
            ));
        }
        self.content = Some(content);
        Ok(())
    }

    pub(super) fn content_ceiling(&self) -> Result<[u64; 9], SemanticTransitionError> {
        let mut ceiling = if self.purpose == SemanticColdModelWorkPurpose::PrivatePrefix {
            self.source.as_ref().map(|source| source.ceiling)
                .ok_or_else(|| publication_input_error("Prefix content must include its full original SOURCE roster before effects"))?
        } else {
            match &self.content {
                Some(content) => Ok(content.ceiling),
                None if !self.is_evaluation() => Ok([0; 9]),
                None => Err(publication_input_error(
                    "evaluation content must be frozen before its original native producers",
                )),
            }?
        };
        if let Some(reads) = &self.resident_reads {
            add_native_work_ceiling(&mut ceiling, reads.ceiling)?;
        }
        add_native_work_ceiling(&mut ceiling, native_tally_initialization_ceiling())?;
        if let Some(scratch) = self.model_scratch_ceiling {
            add_native_work_ceiling(&mut ceiling, scratch)?;
        }
        add_native_work_ceiling(&mut ceiling, self.prepared_scratch_initialization)?;
        if let Some(context) = self.model_context_ceiling {
            add_native_work_ceiling(&mut ceiling, context)?;
        }
        native_work_ceiling_units(ceiling)?;
        Ok(ceiling)
    }

    pub(super) fn claim_native_tally_initialization(
        &mut self,
    ) -> Result<(), SemanticTransitionError> {
        if self.native_initialization_claimed {
            return Ok(());
        }
        add_native_work_ceiling(
            &mut self.submitted_dma_ceiling,
            native_tally_initialization_ceiling(),
        )?;
        self.native_initialization_claimed = true;
        Ok(())
    }

    pub(super) fn claim_content(
        &mut self,
        tensors: &TensorContentBuffers,
        verify: bool,
    ) -> Result<(), SemanticTransitionError> {
        if self.purpose == SemanticColdModelWorkPurpose::PrivatePrefix {
            return Err(publication_input_error(
                "Prefix native content requires its original finite producer claim before effects",
            ));
        }
        if !self.is_evaluation() {
            return Ok(());
        }
        let content = self.content.as_mut().ok_or_else(|| {
            publication_input_error(
                "evaluation content must be admitted before its first native producer",
            )
        })?;
        let same = |expected: &[PreparedSemanticTensor], native: bool| {
            expected.len() == tensors.tensors.len()
                && expected
                    .iter()
                    .zip(&tensors.tensors)
                    .all(|(expected, actual)| {
                        if native {
                            tensor_content_identity(expected) == tensor_content_identity(actual)
                        } else {
                            same_tensor_content_owner(expected, actual)
                        }
                    })
        };
        let kind = if matches!(tensors.seals, TensorContentSeals::Model(_)) {
            let actual = tensors.native_work_ceiling(verify)?;
            if !verify
                || actual
                    .iter()
                    .zip(content.model_ceiling)
                    .any(|(actual, upper)| *actual > upper)
            {
                return Err(publication_input_error(
                    "model content changed its original native geometry ceiling",
                ));
            }
            0
        } else if same(content.cohort.content_tensors(), true) {
            1
        } else if same(&content.output, false) {
            2
        } else if same(&content.objective, false) {
            3
        } else {
            return Err(publication_input_error(
                "native content is outside its original evaluation roster",
            ));
        };
        let remaining = &mut content.remaining[kind][usize::from(verify)];
        *remaining = remaining.checked_sub(1).ok_or_else(|| {
            publication_input_error(
                "native content exceeded its original finite evaluation occurrences",
            )
        })?;
        Ok(())
    }

    /// The actual evaluation owner claims its complete original preparation
    /// once, before any selection or source observation can enter the driver.
    pub(super) fn claim_evaluation_preparation(
        &mut self,
        invocation: &Arc<()>,
    ) -> Result<(), SemanticTransitionError> {
        if !self.is_evaluation() {
            return Ok(());
        }
        let content = self.content.as_mut().ok_or_else(|| {
            publication_input_error("evaluation preparation precedes its original admission")
        })?;
        if !Arc::ptr_eq(&content.invocation, invocation) || !content.preparation_available {
            return Err(publication_input_error(
                "evaluation preparation differs from its sole admitted original invocation",
            ));
        }
        content.preparation_available = false;
        Ok(())
    }

    pub(super) fn claim_model_snapshot(
        &mut self,
        rows: usize,
        contract_bytes: usize,
        source: Option<&Arc<prepared_source_content::PreparedSourceContentOwner>>,
    ) -> Result<bool, SemanticTransitionError> {
        if let Some(owner) = source {
            self.require_source_content_admitted(owner)?;
            let recipe = self
                .source
                .as_mut()
                .expect("checked original SOURCE admission")
                .recipes
                .iter_mut()
                .find(|recipe| recipe.owner.ptr_eq(&Arc::downgrade(owner)))
                .expect("checked original SOURCE recipe");
            if recipe.model_snapshot
                != Some(model_snapshot_native_work_ceiling(rows, contract_bytes)?)
            {
                return Err(publication_input_error(
                    "SOURCE model snapshot changed or repeated its original finite producer",
                ));
            }
            recipe.model_snapshot = None;
            return Ok(true);
        }
        if self.is_source_prefix() {
            return Err(publication_input_error(
                "Prefix model snapshot lacks its original finite SOURCE producer",
            ));
        }
        if !self.is_evaluation() {
            return Ok(false);
        }
        let content = self.content.as_mut().ok_or_else(|| {
            publication_input_error("model snapshot precedes its original cold admission")
        })?;
        let expected = content.model_snapshot.ok_or_else(|| {
            publication_input_error("model snapshot exceeded its original finite occurrence")
        })?;
        if expected != model_snapshot_native_work_ceiling(rows, contract_bytes)? {
            return Err(publication_input_error(
                "model snapshot changed its originally admitted native geometry",
            ));
        }
        add_native_work_ceiling(&mut self.submitted_dma_ceiling, expected)?;
        content.model_snapshot = None;
        Ok(true)
    }

    fn claim_model_contract_guard(
        &mut self,
        ceiling: [u64; 9],
        source: Option<&Arc<prepared_source_content::PreparedSourceContentOwner>>,
    ) -> Result<bool, SemanticTransitionError> {
        if let Some(owner) = source {
            self.require_source_content_admitted(owner)?;
            let recipe = self
                .source
                .as_mut()
                .expect("checked original SOURCE admission")
                .recipes
                .iter_mut()
                .find(|recipe| recipe.owner.ptr_eq(&Arc::downgrade(owner)))
                .expect("checked original SOURCE recipe");
            if recipe.model_contract_guard != Some(ceiling) {
                return Err(publication_input_error(
                    "SOURCE model guard changed or repeated its original finite producer",
                ));
            }
            recipe.model_contract_guard = None;
            return Ok(true);
        }
        if self.is_source_prefix() {
            return Err(publication_input_error(
                "Prefix model contract guard lacks its original finite SOURCE producer",
            ));
        }
        if !self.is_evaluation() {
            return Ok(false);
        }
        let content = self.content.as_mut().ok_or_else(|| {
            publication_input_error("model contract guard precedes its original admission")
        })?;
        if content.model_contract_guard != Some(ceiling) {
            return Err(publication_input_error(
                "model contract guard exceeded or changed its original finite producer",
            ));
        }
        content.model_contract_guard = None;
        Ok(true)
    }

    pub(super) fn claim_step_input_preparation(
        &mut self,
        ceiling: [u64; 9],
        dma: [u64; 9],
        inputs: &PreparedStepInputs,
    ) -> Result<bool, SemanticTransitionError> {
        if let Some(roster) = &mut self.model_context_preparation {
            if roster.step_input != Some((ceiling, dma)) {
                return Err(publication_input_error(
                    "prepared context step input changed or repeated its original immutable geometry",
                ));
            }
            roster.step_input = None;
            return Ok(true);
        }
        if self.purpose == SemanticColdModelWorkPurpose::PrivatePrefix {
            let source = self.source.as_mut().ok_or_else(|| {
                publication_input_error(
                    "SOURCE step input preparation precedes its complete native admission",
                )
            })?;
            let recipe = source
                .recipes
                .iter_mut()
                .find(|recipe| {
                    recipe
                        .owner
                        .upgrade()
                        .is_some_and(|owner| owner.matches_inputs(inputs))
                })
                .ok_or_else(|| {
                    publication_input_error(
                        "SOURCE step inputs are outside their original native union",
                    )
                })?;
            if !recipe.preparation_claimed || recipe.step_input_preparation != Some((ceiling, dma))
            {
                return Err(publication_input_error(
                    "SOURCE step inputs changed or repeated their original finite producer",
                ));
            }
            recipe.step_input_preparation = None;
            return Ok(true);
        }
        if !self.is_evaluation() {
            return Ok(false);
        }
        let content = self.content.as_mut().ok_or_else(|| {
            publication_input_error("step input preparation precedes its original admission")
        })?;
        if content.step_input_preparation != Some((ceiling, dma)) {
            return Err(publication_input_error(
                "step input preparation exceeded or changed its original finite producer",
            ));
        }
        add_native_work_ceiling(&mut self.submitted_dma_ceiling, dma)?;
        content.step_input_preparation = None;
        Ok(true)
    }

    pub(crate) fn require_submitted_dma(
        &self,
        bytes: usize,
        recorded: bool,
    ) -> Result<(), SemanticTransitionError> {
        if recorded {
            return Ok(());
        }
        let mut reached = self.submitted_dma;
        add_native_work_ceiling(
            &mut reached,
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                u64::try_from(bytes).map_err(|_| native_work_ceiling_overflow())?,
            ],
        )?;
        if reached
            .iter()
            .zip(self.submitted_dma_ceiling)
            .any(|(actual, upper)| *actual > upper)
        {
            return Err(publication_input_error(
                "native DMA precedes its original finite entitlement",
            ));
        }
        Ok(())
    }

    pub(crate) fn record_submitted_dma(
        &mut self,
        bytes: usize,
        recorded: &mut bool,
    ) -> Result<(), SemanticTransitionError> {
        if *recorded {
            return Ok(());
        }
        let mut reached = self.submitted_dma;
        add_native_work_ceiling(
            &mut reached,
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                u64::try_from(bytes).map_err(|_| native_work_ceiling_overflow())?,
            ],
        )?;
        // A finite-entitlement failure cannot erase a command already submitted
        // to the driver. Retain its reached evidence before refusing completion.
        self.submitted_dma = reached;
        *recorded = true;
        if reached
            .iter()
            .zip(self.submitted_dma_ceiling)
            .any(|(actual, upper)| *actual > upper)
        {
            return Err(publication_input_error(
                "submitted native DMA exceeded its original finite entitlement",
            ));
        }
        Ok(())
    }

    /// Merge only after the original operation's joins. DMA occurrences have
    /// no CUDA tally writer, so this same native owner supplies their reached
    /// subtotal exactly once to the cached final report, never to GPU counters.
    pub(super) fn merge_submitted_dma(
        &self,
        result: &mut SemanticColdModelWorkResult,
    ) -> Result<(), SemanticTransitionError> {
        if native_work_ceiling_units(result.native_events)? != result.native_work
            || self
                .submitted_dma
                .iter()
                .zip(self.submitted_dma_ceiling)
                .any(|(actual, upper)| *actual > upper)
        {
            return Err(publication_input_error(
                "original native DMA report differs from its retained entitlement",
            ));
        }
        let mut events = result.native_events;
        add_native_work_ceiling(&mut events, self.submitted_dma)?;
        if self.model_context_ceiling.is_some()
            && events
                .iter()
                .zip(self.content_ceiling()?)
                .any(|(actual, upper)| *actual > upper)
        {
            return Err(publication_input_error(
                "model context native work exceeded its original frozen producer roster",
            ));
        }
        let units = native_work_ceiling_units(events)?;
        result
            .model_work
            .checked_add(units)
            .ok_or_else(native_work_ceiling_overflow)?;
        result.native_events = events;
        result.native_work = units;
        Ok(())
    }
}

pub(super) struct FrozenColdSourceContent {
    pub(super) recipes: Vec<FrozenSourceRecipe>,
    pub(super) ceiling: [u64; 9],
}

/// Enclose the genuine feedback encoder, lineage projection and six seals.
/// The CUDA launch repeats its acquired-input guard once per encoder block.
pub(super) fn feedback_native_work_ceiling(
    source: &SemanticTransitionSession,
    lease: &SemanticPublishedLease,
    feedback: &FeedbackBuffers,
) -> Result<[u64; 9], SemanticTransitionError> {
    let descriptor = source.descriptor();
    let slots = u64::try_from(feedback.slots).map_err(|_| native_work_ceiling_overflow())?;
    let capacity =
        u64::try_from(feedback.support_capacity).map_err(|_| native_work_ceiling_overflow())?;
    let task = &source
        .task
        .as_ref()
        .ok_or(SemanticTransitionError::NotBound)?
        .0;
    feedback_geometry_native_work_ceiling(
        &lease.directory,
        slots,
        capacity,
        &feedback.schema,
        descriptor.arena[2],
        descriptor.arena[3],
        &task.spec,
        feedback.content.native_work_ceiling(false)?,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "one canonical producer requires its original typed geometry"
)]
pub(super) fn feedback_geometry_native_work_ceiling(
    directory: &[PublicationRange],
    slots: u64,
    capacity: u64,
    schema: &SemanticFeedbackSchema,
    statement_capacity: u64,
    versions: u64,
    task: &SemanticTaskEvaluationSpec,
    output_ceiling: [u64; 9],
) -> Result<[u64; 9], SemanticTransitionError> {
    let ranges = u64::try_from(directory.len()).map_err(|_| native_work_ceiling_overflow())?;
    let records = directory
        .iter()
        .filter(|range| range.role == 15 && range.index == 0);
    let statements = directory.iter().filter(|range| range.role == 16);
    let count =
        u64::try_from(statements.clone().count()).map_err(|_| native_work_ceiling_overflow())?;
    if records.clone().count() != 1 || count == 0 || count > slots {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let guards = ceiling_sum([count, 1])?;
    let mut inputs = [
        0,
        ceiling_sum([
            8,
            ceiling_product(guards, ranges)?,
            ceiling_product(4, guards)?,
        ])?,
        0,
        0,
        0,
        0,
        0,
        0,
        ceiling_product(guards, 21 * 8 + 32)?,
    ];
    for range in records.chain(statements) {
        inputs[7] = ceiling_sum([
            inputs[7],
            ceiling_sum([128, range.length_bytes, 9, 63])? / 64,
        ])?;
    }
    let mut ceiling = [
        2,
        ceiling_product(slots, ranges)?,
        0,
        0,
        0,
        0,
        0,
        0,
        ceiling_product(
            slots,
            ceiling_sum([9, ceiling_product(4, schema.feature_width as u64)?])?,
        )?,
    ];
    repeat_native_ceiling(&mut ceiling, inputs, ceiling_sum([slots.min(65535), 1])?)?;
    let receipt = size_of::<crate::semantic_hypergraph::SemanticResidentReceiptRecord>() as u64;
    add_native_work_ceiling(
        &mut ceiling,
        [
            0,
            ceiling_product(2, slots)?,
            0,
            0,
            0,
            0,
            0,
            0,
            ceiling_sum([
                ceiling_product(slots, ceiling_sum([receipt, 16])?)?,
                8,
                ceiling_product(96, capacity)?,
                ceiling_product(16, slots)?,
                ceiling_product(slots, ceiling_sum([receipt, 8])?)?,
                ceiling_product(8, slots)?,
            ])?,
        ],
    )?;
    let editable = task.program.editable_program().is_some();
    let calls = ceiling_product(
        slots,
        if editable {
            task.task_ground.bindings().len() as u64
        } else {
            1
        },
    )?;
    let truth = [
        1,
        ceiling_sum([ceiling_product(4, statement_capacity)?, 8])?,
        0,
        0,
        0,
        0,
        0,
        0,
        ceiling_sum([ceiling_product(3, receipt)?, 32 * 8])?,
    ];
    repeat_native_ceiling(&mut ceiling, truth, calls)?;
    repeat_native_ceiling(
        &mut ceiling,
        [
            0,
            ceiling_sum([
                1,
                ceiling_product(9, versions)?,
                ceiling_product(versions, capacity)?,
                4,
            ])?,
            versions,
            0,
            0,
            0,
            0,
            ceiling_product(2, versions)?,
            ceiling_sum([ceiling_product(265, versions)?, 7 * 8])?,
        ],
        calls,
    )?;
    if editable {
        // Same fixed PROGRAM_FUEL used by task_program_closure. Every charged
        // pair checks the fuel before increment, including failed prefixes.
        add_native_work_ceiling(
            &mut ceiling,
            [0, ceiling_sum([ranges, 1_000_000])?, 0, 0, 0, 0, 0, 0, 0],
        )?;
    }
    add_native_work_ceiling(&mut ceiling, output_ceiling)?;
    native_work_ceiling_units(ceiling)?;
    Ok(ceiling)
}

pub(super) struct FrozenSourceRecipe {
    pub(super) owner: std::sync::Weak<prepared_source_content::PreparedSourceContentOwner>,
    pub(super) resident: std::sync::Weak<super::resident_model_memory::ResidentModelMemoryOwner>,
    pub(super) resident_ceiling: [u64; 9],
    pub(super) preparation_claimed: bool,
    pub(super) resident_claimed: bool,
    pub(super) dma_ceiling: [u64; 9],
    pub(super) model_snapshot: Option<[u64; 9]>,
    pub(super) model_contract_guard: Option<[u64; 9]>,
    pub(super) step_input_preparation: Option<([u64; 9], [u64; 9])>,
}

impl FrozenColdEvaluationContent {
    #[expect(
        clippy::too_many_arguments,
        reason = "original admitted producer geometry stays bound to its evaluation owner"
    )]
    pub(super) fn new(
        invocation: Arc<()>,
        cohort: Arc<SemanticEvaluationCohort>,
        output: Vec<PreparedSemanticTensor>,
        objective: Vec<PreparedSemanticTensor>,
        model_ceiling: [u64; 9],
        preparation_ceiling: [u64; 9],
        model_snapshot: [u64; 9],
        model_contract_guard: [u64; 9],
        step_input_preparation: Option<([u64; 9], [u64; 9])>,
    ) -> Result<Self, SemanticTransitionError> {
        let original = cohort.content_tensors();
        if original.len() != 17 || output.len() != 3 || objective.len() != 8 {
            return Err(publication_input_error(
                "evaluation content requires its original seventeen, three and eight ports",
            ));
        }
        let check =
            |tensor: &PreparedSemanticTensor, index: u64, scalar: u64, dimensions: &[u64]| {
                let layout = tensor.layout;
                layout.role == 0
                    && layout.index == index
                    && layout.scalar_type == scalar
                    && layout.rank as usize == dimensions.len()
                    && layout.logical_axis == u64::MAX
                    && tensor.logical_begin == 0
                    && tensor.logical_end == 0
                    && &layout.dimensions[..dimensions.len()] == dimensions
            };
        if !check(&output[0], 0, 6, &[6])
            || !check(&output[1], 1, 8, &[1])
            || !check(&output[2], 2, 1, &[25])
            || output[0].data != output[2].data
            || output[1].data
                != output[2]
                    .data
                    .checked_add(24)
                    .ok_or_else(native_work_ceiling_overflow)?
        {
            return Err(publication_input_error(
                "evaluation output changed its original shared twenty-five-byte projection",
            ));
        }
        for (index, tensor) in objective[..5].iter().enumerate() {
            let source = original[index].layout;
            if !check(
                tensor,
                index as u64,
                3,
                &source.dimensions[..source.rank as usize],
            ) {
                return Err(publication_input_error(
                    "objective source copies changed their original selected geometry",
                ));
            }
        }
        let groups = original[3].layout.dimensions[0];
        let rows = original[6].layout.dimensions[0];
        if !check(&objective[5], 5, 6, &[groups, 2])
            || !check(&objective[6], 6, 8, &[groups, rows])
            || !check(&objective[7], 7, 8, &[1])
            || objective[7].data != output[1].data
        {
            return Err(publication_input_error(
                "objective result changed its original selected geometry or output owner",
            ));
        }
        // Actual original producer roster: four model verifications; a fresh
        // cohort seal and external source capture; three cohort guards and two
        // external source verifies; output capture/finish verify; objective
        // capture and two verifies. Captured occurrences remain finite here,
        // although their execution tally is owned by numerical capture.
        let remaining = [[0, 4], [2, 5], [1, 1], [1, 2]];
        let mut ceiling = preparation_ceiling;
        add_native_work_ceiling(&mut ceiling, model_snapshot)?;
        add_native_work_ceiling(&mut ceiling, model_contract_guard)?;
        if let Some((native, dma)) = step_input_preparation {
            add_native_work_ceiling(&mut ceiling, native)?;
            add_native_work_ceiling(&mut ceiling, dma)?;
        }
        let sources = [&output, &objective];
        for (kind, tensors) in sources.into_iter().enumerate() {
            for verify in [false, true] {
                let mut occurrence = [0; 9];
                for tensor in tensors {
                    add_native_work_ceiling(
                        &mut occurrence,
                        tensor_layout_native_work_ceiling(
                            &tensor.layout,
                            tensor.logical_begin,
                            tensor.logical_end,
                            tensor.source.as_ref().map_or(0, DeviceMemoryView::len),
                            verify,
                        )?,
                    )?;
                }
                for _ in 0..remaining[kind + 2][usize::from(verify)] {
                    add_native_work_ceiling(&mut ceiling, occurrence)?;
                }
            }
        }
        for _ in 0..4 {
            add_native_work_ceiling(&mut ceiling, model_ceiling)?;
        }
        for verify in [false, true] {
            let occurrence = cohort.content_native_work_ceiling(verify)?;
            for _ in 0..remaining[1][usize::from(verify)] {
                add_native_work_ceiling(&mut ceiling, occurrence)?;
            }
        }
        Ok(Self {
            invocation,
            preparation_available: true,
            cohort,
            output,
            objective,
            model_ceiling,
            ceiling,
            remaining,
            model_snapshot: Some(model_snapshot),
            model_contract_guard: Some(model_contract_guard),
            step_input_preparation,
        })
    }
}

pub(super) struct OriginalModelSnapshot {
    pub(super) directory: DeviceMemoryView<PublicationRange>,
    pub(super) source: DeviceMemoryView<u8>,
    pub(super) offsets: Vec<usize>,
    pub(super) contract_span: std::ops::Range<usize>,
    pub(super) cursor: usize,
    pub(super) commands: Vec<OriginalNativeCommand>,
    pub(super) allowance: Option<std::sync::Weak<Mutex<ColdNativeAllowance>>>,
}

pub(super) struct OriginalStepInputPreparation {
    pub(super) writes: [crate::device::RetainedDeviceWrite<PublicationStepInput>; 2],
    pub(super) admitted: [bool; 2],
    pub(super) recorded: [bool; 2],
    pub(super) cursor: usize,
    pub(super) command: OriginalNativeCommand,
    pub(super) allowance: Option<std::sync::Weak<Mutex<ColdNativeAllowance>>>,
    pub(super) native_work: Option<DeviceMemoryView<u64>>,
}

pub(super) fn model_snapshot_native_work_ceiling(
    rows: usize,
    contract_bytes: usize,
) -> Result<[u64; 9], SemanticTransitionError> {
    let commands = rows
        .checked_add(1)
        .ok_or_else(native_work_ceiling_overflow)?;
    let bytes = rows
        .checked_mul(size_of::<PublicationRange>())
        .and_then(|bytes| bytes.checked_add(contract_bytes))
        .ok_or_else(native_work_ceiling_overflow)?;
    Ok([
        u64::try_from(commands).map_err(|_| native_work_ceiling_overflow())?,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        u64::try_from(bytes).map_err(|_| native_work_ceiling_overflow())?,
    ])
}

/// The actual publication ModelContract guard authenticates its original bank,
/// every directory seal, the schema/identity, and the selected model seal fold.
/// Its outer guard hashes the raw contract a second time; private retained
/// contract verifications are separate content occurrences.
pub(super) fn model_contract_guard_native_work_ceiling(
    storage: &PublicationStorage,
    directory: &[PublicationRange],
) -> Result<[u64; 9], SemanticTransitionError> {
    model_contract_guard_geometry_native_work_ceiling(&storage.contract_value, directory)
}

pub(super) fn model_contract_guard_geometry_native_work_ceiling(
    contract: &PublicationContract,
    directory: &[PublicationRange],
) -> Result<[u64; 9], SemanticTransitionError> {
    let record = directory
        .iter()
        .find(|range| range.role == 44 && range.index == 0)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let blocks = |bytes: u64| {
        bytes
            .checked_add(9 + 63)
            .map(|bytes| bytes / 64)
            .ok_or_else(native_work_ceiling_overflow)
    };
    let mut ceiling = [1, 0, 0, 0, 0, 0, 0, 0, 752];
    ceiling[7] = blocks(size_of::<PublicationBank>() as u64)?
        .checked_add(
            blocks(
                record
                    .length_bytes
                    .checked_add(128)
                    .ok_or_else(native_work_ceiling_overflow)?,
            )?
            .checked_mul(2)
            .ok_or_else(native_work_ceiling_overflow)?,
        )
        .and_then(|value| value.checked_add(5))
        .ok_or_else(native_work_ceiling_overflow)?;
    ceiling[7] = ceiling[7]
        .checked_add(blocks(contract.model_contract_layout.schema_bytes)?)
        .ok_or_else(native_work_ceiling_overflow)?;
    let range_blocks = blocks(size_of::<PublicationRange>() as u64)?
        .checked_add(2)
        .ok_or_else(native_work_ceiling_overflow)?;
    for range in directory {
        add_native_work_ceiling(&mut ceiling, [0, 4, 0, 0, 0, 0, 0, range_blocks, 144])?;
        if matches!(range.role, 18..=25) {
            add_native_work_ceiling(&mut ceiling, [0, 0, 0, 0, 0, 0, 0, 4, 224])?;
        }
    }
    Ok(ceiling)
}

/// One genuine resident guard hashes every typed range and that range's full
/// selected backing. Shared allocations are deliberately counted again for each
/// view, exactly as semantic_publication_content_guard executes them.
pub(super) fn resident_model_content_guard_native_work_ceiling(
    storage: &PublicationStorage,
    directory: &[PublicationRange],
    keys: &[(SemanticStateRole, u64)],
) -> Result<[u64; 9], SemanticTransitionError> {
    resident_model_content_guard_geometry_native_work_ceiling(
        &storage.model_memory,
        &storage.layouts,
        directory,
        keys,
        |slot| {
            storage
                .allocations
                .get(slot)
                .map(|allocation| allocation.entry().bytes)
                .ok_or(SemanticTransitionError::ObservationMismatch)
        },
    )
}

pub(super) fn resident_model_content_guard_geometry_native_work_ceiling(
    model_memory: &ModelMemoryGeometry,
    layouts: &BTreeMap<(u64, u64), SemanticTensorLayout>,
    directory: &[PublicationRange],
    keys: &[(SemanticStateRole, u64)],
    allocation_bytes: impl Fn(usize) -> Result<u64, SemanticTransitionError>,
) -> Result<[u64; 9], SemanticTransitionError> {
    if keys.len() != model_memory.views.len() {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let mut ceiling = [0; 9];
    let mut seen = BTreeSet::new();
    for &(role, index) in keys {
        if !matches!(role as u64, 18..=25) || !seen.insert((role as u64, index)) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let (position, range) = directory
            .iter()
            .enumerate()
            .find(|(_, range)| range.role == role as u64 && range.index == index)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let layout = layouts
            .get(&(role as u64, index))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        add_native_work_ceiling(
            &mut ceiling,
            tensor_layout_native_work_ceiling(
                layout,
                range.logical_begin,
                range.logical_end,
                usize::try_from(range.length_bytes).map_err(|_| native_work_ceiling_overflow())?,
                true,
            )?,
        )?;
        let blocks = allocation_bytes(
            usize::try_from(range.storage_slot).map_err(|_| native_work_ceiling_overflow())?,
        )?
        .checked_add(9 + 63)
        .ok_or_else(native_work_ceiling_overflow)?
            / 64;
        let visits = u64::try_from(position)
            .ok()
            .and_then(|position| position.checked_add(1))
            .ok_or_else(native_work_ceiling_overflow)?;
        add_native_work_ceiling(&mut ceiling, [0, visits, 0, 0, 0, 0, 0, blocks, 32])?;
    }
    Ok(ceiling)
}

/// Add producer ceilings in the native tally's fixed event order.
pub(super) fn add_native_work_ceiling(
    destination: &mut [u64; 9],
    addition: [u64; 9],
) -> Result<(), SemanticTransitionError> {
    let mut combined = *destination;
    for (total, value) in combined.iter_mut().zip(addition) {
        *total = total
            .checked_add(value)
            .ok_or_else(native_work_ceiling_overflow)?;
    }
    *destination = combined;
    Ok(())
}

/// Native events have unit tariffs, as checked by `ExecutionWork::validate_model`.
pub(super) fn native_work_ceiling_units(ceiling: [u64; 9]) -> Result<u64, SemanticTransitionError> {
    ceiling.into_iter().try_fold(0u64, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(native_work_ceiling_overflow)
    })
}

fn native_work_ceiling_overflow() -> SemanticTransitionError {
    publication_input_error("native producer work ceiling overflowed")
}

fn ceiling_sum(values: impl IntoIterator<Item = u64>) -> Result<u64, SemanticTransitionError> {
    values.into_iter().try_fold(0u64, |sum, value| {
        sum.checked_add(value)
            .ok_or_else(native_work_ceiling_overflow)
    })
}

fn ceiling_product(a: u64, b: u64) -> Result<u64, SemanticTransitionError> {
    a.checked_mul(b).ok_or_else(native_work_ceiling_overflow)
}

/// Finite producer geometry of the original native model-context seal. The
/// acquired SOURCE supplies immutable extents, not observed execution counts.
pub(super) fn model_context_activation_native_work_ceiling(
    source: &SemanticTransitionSession,
    current: &SemanticPublishedLease,
    target: &SemanticTransitionSession,
) -> Result<[u64; 9], SemanticTransitionError> {
    source.checked_reader(current)?;
    let source_storage = source
        .publication
        .as_ref()
        .ok_or(SemanticTransitionError::NotBound)?;
    let target_storage = target
        .publication
        .as_ref()
        .ok_or(SemanticTransitionError::NotBound)?;
    let source_directory = &current.directory;
    let target_directory = &target_storage.bank_templates[0];
    let range = |directory: &[PublicationRange], role| {
        directory
            .iter()
            .find(|range| range.role == role && range.index == 0)
            .copied()
            .ok_or(SemanticTransitionError::ObservationMismatch)
    };
    let count = |value: usize| u64::try_from(value).map_err(|_| native_work_ceiling_overflow());
    let ns = count(source_directory.len())?;
    let nt = count(target_directory.len())?;
    let ss = count(source_storage.allocations.len())?;
    let ts = count(source_storage.layouts.len())?;
    let tt = count(target_storage.layouts.len())?;
    let pairs = |value: u64| ceiling_product(value, value.saturating_sub(1)).map(|value| value / 2);
    let hash = |bytes: u64| -> Result<[u64; 9], SemanticTransitionError> {
        Ok([0, 0, 0, 0, 0, 0, 0, ceiling_sum([bytes, 72])? / 64, 32])
    };
    let digest = |bytes| -> Result<[u64; 9], SemanticTransitionError> {
        let mut result = content_digest_ceiling(bytes, 21, false)?;
        result[0] = 0;
        Ok(result)
    };
    let fold = [0, 0, 0, 0, 0, 0, 0, 2, 112];
    let mut ceiling = super::actor_refresh::actor_initialize_native_work_ceiling(target)?;
    let prefix_position = target_directory
        .iter()
        .position(|range| {
            range.role == SemanticStateRole::PrefixIdentity as u64 && range.index == 0
        })
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let prefix_range = target_directory[prefix_position];
    if prefix_range.length_bytes != 32 {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    add_native_work_ceiling(
        &mut ceiling,
        content_digest_ceiling(prefix_range.length_bytes, 21, false)?,
    )?;
    add_native_work_ceiling(
        &mut ceiling,
        [0, count(prefix_position + 1)?, 0, 0, 0, 0, 0, 0, 0],
    )?;
    // The same report encloses acquiring the sole TARGET reader and releasing
    // the original SOURCE reader after the known seal. Their fixed commands
    // do not re-run model work or initialization.
    add_native_work_ceiling(
        &mut ceiling,
        [
            2,
            12,
            0,
            0,
            0,
            0,
            0,
            0,
            size_of::<PublicationLease>() as u64 + 184,
        ],
    )?;
    add_native_work_ceiling(
        &mut ceiling,
        model_contract_guard_geometry_native_work_ceiling(
            &source_storage.contract_value,
            source_directory,
        )?,
    )?;
    add_native_work_ceiling(
        &mut ceiling,
        [0, ceiling_sum([ss, pairs(ss)?])?, 0, 0, 0, 0, 0, 0, 0],
    )?;
    for directory in &target_storage.bank_templates {
        add_native_work_ceiling(
            &mut ceiling,
            step_input_producer_native_work_ceiling(target_storage, &[], directory)?,
        )?;
    }
    let models = count(
        target_directory
            .iter()
            .filter(|item| matches!(item.role, 18..=25))
            .count(),
    )?;
    let mutable = target_storage
        .bank_templates
        .iter()
        .try_fold(0u64, |total, directory| {
            ceiling_sum([
                total,
                count(
                    directory
                        .iter()
                        .filter(|item| {
                            (publication_mutable_role(item.role) && !matches!(item.role, 18..=25))
                                || item.role == 17
                        })
                        .count(),
                )?,
            ])
        })?;
    // The frozen frame is checked as a subspan of each original codebook.
    let frame_bytes = range(source_directory, 48)?
        .length_bytes
        .min(range(target_directory, 48)?.length_bytes);
    let model_contract_bytes = range(source_directory, 44)?.length_bytes;
    let mut visits = ceiling_sum([
        44,
        2,
        count(target_storage.bank_templates[0].len())?,
        count(target_storage.bank_templates[1].len())?,
        ceiling_product(
            models,
            ceiling_sum([ceiling_product(2, ns)?, ts, nt, tt, 14])?,
        )?,
        ceiling_product(mutable, ss)?,
        ceiling_sum([ns, nt, 15, frame_bytes])?,
        ceiling_sum([ns, nt, 4, model_contract_bytes])?,
    ])?;
    for role in [42, 45, 46, 47, 50] {
        let bytes = range(source_directory, role)?.length_bytes;
        visits = ceiling_sum([visits, 1, ns, nt, 4, bytes])?;
        add_native_work_ceiling(&mut ceiling, digest(bytes)?)?;
    }
    add_native_work_ceiling(
        &mut ceiling,
        digest(range(source_directory, 48)?.length_bytes)?,
    )?;
    add_native_work_ceiling(&mut ceiling, digest(model_contract_bytes)?)?;
    add_native_work_ceiling(
        &mut ceiling,
        [
            0,
            visits,
            0,
            0,
            0,
            0,
            0,
            0,
            ceiling_sum([400, size_of::<DeviceState>() as u64, 24])?,
        ],
    )?;

    // Fresh queries enter the same bounded resident closure. The restored
    // initializer already encloses noneditable truth calls and feedback reset.
    let (task, _) = target
        .task
        .as_ref()
        .ok_or(SemanticTransitionError::NotBound)?;
    let queries = count(task.spec.statement_records.len())?;
    if task.spec.program.editable_program().is_some() {
        let attainment = attainment_native_work_ceiling(&task.spec)?;
        add_native_work_ceiling(
            &mut ceiling,
            [
                0,
                ceiling_sum([1_000_000, attainment[1]])?,
                0,
                0,
                0,
                0,
                0,
                0,
                ceiling_product(
                    queries,
                    size_of::<crate::semantic_hypergraph::SemanticResidentReceiptRecord>() as u64,
                )?,
            ],
        )?;
    }
    add_native_work_ceiling(
        &mut ceiling,
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            ceiling_product(queries, size_of::<RawFeedbackRecord>() as u64)?,
        ],
    )?;
    add_native_work_ceiling(
        &mut ceiling,
        digest(range(target_directory, 1)?.length_bytes)?,
    )?;
    // The initializer's raw typed-hash prefix bound omits these tensor words.
    let tensor_prefix = target_storage
        .layouts
        .values()
        .try_fold(0u64, |bytes, layout| {
            ceiling_sum([bytes, ceiling_product(8, ceiling_sum([4, layout.rank])?)?])
        })?;
    add_native_work_ceiling(&mut ceiling, [0, 0, 0, 0, 0, 0, 0, 0, tensor_prefix])?;
    let coverage_count = target_directory
        .iter()
        .filter(|item| matches!(item.role, 1 | 4..=13))
        .count();
    add_native_work_ceiling(
        &mut ceiling,
        [
            0,
            ceiling_sum([
                ceiling_product(3, nt)?,
                ceiling_product(count(coverage_count)?, ceiling_sum([nt, tt])?)?,
            ])?,
            0,
            0,
            0,
            0,
            0,
            0,
            528,
        ],
    )?;
    for item in target_directory
        .iter()
        .filter(|item| matches!(item.role, 1 | 4..=13))
    {
        let layout = target_storage
            .layouts
            .get(&(item.role, item.index))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let allocation = target_storage
            .allocations
            .get(item.storage_slot as usize)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let mut typed = tensor_layout_native_work_ceiling(
            layout,
            item.logical_begin,
            item.logical_end,
            allocation.len(),
            true,
        )?;
        typed[0] = 0;
        add_native_work_ceiling(&mut ceiling, typed)?;
        add_native_work_ceiling(&mut ceiling, fold)?;
    }
    add_native_work_ceiling(&mut ceiling, digest(360)?)?;

    // Canonical fresh queue construction uses one original entry capacity for
    // intents and acknowledgments. Payload partitions retain their own spans.
    let intents = range(source_directory, 30)?
        .length_bytes
        .checked_sub(size_of::<IntentQueueHeader>() as u64)
        .filter(|bytes| bytes % size_of::<IntentEntry>() as u64 == 0)
        .map(|bytes| bytes / size_of::<IntentEntry>() as u64)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let acknowledgments = intents;
    let intent_payload = range(source_directory, 31)?.length_bytes;
    let ack_payload = range(source_directory, 32)?
        .length_bytes
        .checked_sub(ceiling_sum([
            size_of::<AcknowledgementQueueHeader>() as u64,
            ceiling_product(intents, size_of::<AcknowledgementEntry>() as u64)?,
        ])?)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let ack_backing =
        source_storage
            .bank_templates
            .iter()
            .try_fold(0u64, |maximum, directory| {
                let item = range(directory, 32)?;
                let allocation = source_storage
                    .allocations
                    .get(item.storage_slot as usize)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                Ok::<_, SemanticTransitionError>(maximum.max(count(allocation.len())?))
            })?;
    add_native_work_ceiling(
        &mut ceiling,
        [
            0,
            ceiling_sum([
                ceiling_product(5, ns)?,
                ceiling_product(2, nt)?,
                ceiling_product(21, intents)?,
                ceiling_product(13, acknowledgments)?,
                ceiling_product(5, ceiling_product(acknowledgments, intents)?)?,
                ceiling_product(5, pairs(acknowledgments)?)?,
                16,
            ])?,
            acknowledgments,
            0,
            0,
            0,
            0,
            ceiling_sum([
                ceiling_sum([
                    intent_payload,
                    ceiling_product(72, ceiling_sum([intents, 1])?)?,
                ])? / 64,
                ceiling_product(6, intents)?,
                ceiling_sum([ack_payload, ceiling_product(72, acknowledgments)?])? / 64,
                ceiling_product(4, acknowledgments)?,
            ])?,
            ceiling_sum([
                ceiling_product(32, ceiling_sum([intents, 1])?)?,
                ceiling_product(344, intents)?,
                ceiling_product(32, acknowledgments)?,
                ceiling_product(224, acknowledgments)?,
            ])?,
        ],
    )?;
    add_native_work_ceiling(&mut ceiling, digest(ack_backing)?)?;

    // The final logical and descriptor reseals are distinct from initialization.
    add_native_work_ceiling(
        &mut ceiling,
        [
            0,
            ceiling_product(2, nt)?,
            0,
            0,
            0,
            0,
            0,
            0,
            ceiling_sum([size_of::<PublicationHeader>() as u64, (1 + 8 + 19) * 8])?,
        ],
    )?;
    for bytes in [
        size_of::<PublicationHeader>() as u64,
        (32 * size_of::<SourceSlot>()) as u64,
        (COMPONENT_COUNT * size_of::<SemanticTransitionReceipt>()) as u64,
    ] {
        add_native_work_ceiling(&mut ceiling, hash(bytes)?)?;
    }
    let book_prefix = b"xlog.semantic.logical-codebooks.v1\0".len() as u64 + 32 + 8;
    for _ in 0..2 {
        add_native_work_ceiling(
            &mut ceiling,
            hash(ceiling_sum([
                book_prefix,
                range(target_directory, 48)?.length_bytes,
            ])?)?,
        )?;
        add_native_work_ceiling(&mut ceiling, [0, 0, 0, 0, 0, 0, 0, 0, book_prefix])?;
        add_native_work_ceiling(&mut ceiling, fold)?;
    }
    for item in target_directory
        .iter()
        .filter(|item| !matches!(item.role, 27 | 28 | 30 | 31 | 32 | 33))
    {
        add_native_work_ceiling(&mut ceiling, fold)?;
        if matches!(item.role, 18..=25) {
            add_native_work_ceiling(&mut ceiling, fold)?;
        }
        if matches!(item.role, 2 | 14 | 15) {
            add_native_work_ceiling(&mut ceiling, digest(item.length_bytes)?)?;
        } else if item.role != 48 {
            add_native_work_ceiling(&mut ceiling, [0, 0, 0, 0, 0, 0, 0, 0, 32])?;
        }
    }
    add_native_work_ceiling(
        &mut ceiling,
        [0, ceiling_product(2, nt)?, 0, 0, 0, 0, 0, 0, 32],
    )?;
    for item in target_directory
        .iter()
        .filter(|item| matches!(item.role, 27 | 28 | 30 | 31 | 32 | 33))
    {
        add_native_work_ceiling(&mut ceiling, digest(item.length_bytes)?)?;
        add_native_work_ceiling(&mut ceiling, fold)?;
    }
    add_native_work_ceiling(&mut ceiling, hash(size_of::<PublicationBank>() as u64)?)?;
    for _ in target_directory {
        add_native_work_ceiling(&mut ceiling, hash(size_of::<PublicationRange>() as u64)?)?;
        add_native_work_ceiling(&mut ceiling, fold)?;
    }
    repeat_native_ceiling(&mut ceiling, digest(360)?, 2)?;
    add_native_work_ceiling(&mut ceiling, [0, nt, 0, 0, 0, 0, 0, 0, 0])?;
    native_work_ceiling_units(ceiling)?;
    Ok(ceiling)
}

fn repeat_native_ceiling(
    total: &mut [u64; 9],
    occurrence: [u64; 9],
    count: u64,
) -> Result<(), SemanticTransitionError> {
    let mut addition = [0; 9];
    for (destination, value) in addition.iter_mut().zip(occurrence) {
        *destination = ceiling_product(value, count)?;
    }
    add_native_work_ceiling(total, addition)
}

fn policy_viability_category_ceiling(
    source: &SemanticTransitionSession,
) -> Result<u64, SemanticTransitionError> {
    let components = &source.codebooks.components;
    if components.len() != COMPONENT_COUNT {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let targets = *source
        .codebooks
        .words
        .get(1)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let operands = *source
        .codebooks
        .words
        .get(3)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let qualifier = u64::from(components[43].cardinality);
    let leaf = u64::from(components[49].cardinality);
    // Four target banks, four supported/operand_matches scans per target,
    // and all three action branches, including their rejected categories.
    ceiling_product(
        4,
        ceiling_sum([
            ceiling_product(targets, ceiling_sum([1, ceiling_product(8, operands)?])?)?,
            18,
            ceiling_sum([20, qualifier, leaf, targets])?,
            2,
        ])?,
    )
}

fn rule_prefix_native_work_ceiling(
    predicate_bound: u64,
) -> Result<[u64; 9], SemanticTransitionError> {
    // The arity-aware prefix scans at most three predicate ranges: one head
    // descriptor (17 TableSlot events) and two body descriptors (13 each)
    // per predicate, including admission and context checks.
    // Constant work covers seven trailing NULLs, three optional unary NULLs,
    // 24 variable probes (11 TableSlot / 3 Category each), three 4-by-4
    // atom-mask constructions, and two 15-by-15 coverage scans. Fixed
    // predicates and the three outer position visits fit these same bounds.
    let ceiling = [
        0,
        ceiling_sum([ceiling_product(43, predicate_bound)?, 834])?,
        0,
        ceiling_sum([ceiling_product(6, predicate_bound)?, 566])?,
        0,
        0,
        0,
        0,
        0,
    ];
    native_work_ceiling_units(ceiling)?;
    Ok(ceiling)
}

/// Enclose one original backward recording, not its executed work.
/// `group_member_bound` is the same frozen target's original member-storage
/// extent, not a measured visit count;
/// `snapshot_bytes` is the exact DtoD occurrence retained by that recording.
pub(super) fn policy_vjp_native_work_ceiling(
    source: &SemanticTransitionSession,
    snapshot_bytes: u64,
    group_member_bound: u64,
) -> Result<[u64; 9], SemanticTransitionError> {
    let components = &source.codebooks.components;
    let layout = SemanticPolicyLayout::from_components(components)?;
    let parameters =
        u64::try_from(layout.parameter_cells).map_err(|_| native_work_ceiling_overflow())?;
    let text = ceiling_product(32, TEXT_CARDINALITY as u64)?;
    let recurrent = ceiling_product(ceiling_product(2, 37)?, 128)?;
    let component_count = COMPONENT_COUNT as u64;
    let cells = ceiling_sum(
        components
            .iter()
            .map(|component| u64::from(component.cardinality)),
    )?;
    let mut edit_cells = 0;
    let mut edit_components = 0;
    for component in components.iter().filter(|component| !component.is_text()) {
        edit_cells = ceiling_sum([edit_cells, u64::from(component.cardinality)])?;
        edit_components = ceiling_sum([edit_components, 1])?;
    }
    let targets = *source
        .codebooks
        .words
        .get(1)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let origin_words =
        u64::try_from(size_of::<SemanticTrainingViewOriginRecord>() / size_of::<u64>())
            .map_err(|_| native_work_ceiling_overflow())?;
    let outputs = ceiling_sum([parameters, text, recurrent])?;
    let roots = ceiling_sum([parameters, text, component_count])?;
    // Both original-origin comparisons; fixed coefficients and eight groups;
    // each reached group can scan its bounded member range before refusal.
    let preparation = ceiling_sum([
        ceiling_product(4, origin_words)?,
        9,
        7,
        8,
        ceiling_product(8, group_member_bound)?,
        16,                                   // Goal witness roots.
        2,                                    // Candidate lanes.
        component_count,                      // External coefficient validation.
        component_count,                      // Original importance-weight receipt scan.
        ceiling_product(2, component_count)?, // Critic receipts and baselines.
        1,                                    // Critic term, including reached-refusal clearing.
    ])?;
    // Reached refusal clears four component coordinates; the normal actor path
    // visits at most eleven. Neither path can exceed this one output clear.
    let clear = ceiling_sum([
        ceiling_product(2, outputs)?,
        ceiling_product(11, component_count)?,
    ])?;
    let traversal = ceiling_sum([
        ceiling_product(4, targets)?,
        ceiling_product(4, 3)?,
        ceiling_product(2, component_count)?, // Component and receipt.
        ceiling_product(8, cells)?,
        ceiling_product(2, component_count)?, // Selected support/score.
        ceiling_product(18, edit_components)?, // Retained prefix choices.
    ])?;
    // Hidden/score recomputation and the supplied producer's scalar tally:
    // readout_vjp visits 128*(1+cardinality), plus optional cardinality biases;
    // advance_vjp visits 128*(1+optional selected embedding+128 recurrence).
    let numerical = ceiling_sum([
        ceiling_product(128, edit_components)?,
        ceiling_product(130, edit_cells)?,
        ceiling_product(128, edit_components)?,
        ceiling_product(129, edit_cells)?,
        ceiling_product(ceiling_product(128, 130)?, edit_components)?,
    ])?;
    let legality_calls = ceiling_sum([ceiling_product(2, cells)?, component_count])?;
    let mut ceiling = [
        0,
        ceiling_sum([
            preparation,
            clear,
            traversal,
            numerical,
            ceiling_product(2, policy_viability_category_ceiling(source)?)?,
            ceiling_product(2, legality_calls)?,
            ceiling_product(5, roots)?, // Final actor/full-domain/error packing.
        ])?,
        0,
        ceiling_sum([
            1,
            policy_viability_category_ceiling(source)?,
            ceiling_product(2, legality_calls)?,
        ])?,
        0,
        0,
        0,
        0,
        snapshot_bytes,
    ];
    if source
        .task
        .as_ref()
        .is_some_and(|task| task.0.spec.program.editable_program().is_some())
    {
        // A reached prefix requires predicate_count < books[1]. Four action
        // checks and both scans plus the selected check can enter that owner.
        let prefix_calls = ceiling_sum([4, ceiling_product(2, edit_cells)?, edit_components])?;
        repeat_native_ceiling(
            &mut ceiling,
            rule_prefix_native_work_ceiling(targets)?,
            prefix_calls,
        )?;
    }
    native_work_ceiling_units(ceiling)?;
    Ok(ceiling)
}

/// Enclose the original one-thread ordered critic reduction.
pub(super) fn group_critic_native_work_ceiling(
    member_count: usize,
) -> Result<[u64; 9], SemanticTransitionError> {
    let members = u64::try_from(member_count).map_err(|_| native_work_ceiling_overflow())?;
    // Coefficient, evaluator ABI, group count, up to four identity words,
    // every member term and the one scalar output.
    let ceiling = [0, ceiling_sum([members, 8])?, 0, 0, 0, 0, 0, 0, 0];
    native_work_ceiling_units(ceiling)?;
    Ok(ceiling)
}

/// The protected attainment fold visits actual closure facts and explicit
/// edges. This bounds the same reached TableSlot events, not structural cost.
fn attainment_native_work_ceiling(
    task: &SemanticTaskEvaluationSpec,
) -> Result<[u64; 9], SemanticTransitionError> {
    let mut ceiling = [0; 9];
    if task.task_ground.code() == 1 {
        ceiling[1] = ceiling_sum([
            ceiling_product(
                task.statement_records.len() as u64,
                crate::semantic_program::RESIDENT_PROGRAM_FACT_CAPACITY as u64 + 1,
            )?,
            task.task_ground.mandatory_edge_count()?,
        ])?;
    }
    Ok(ceiling)
}

/// Enclose the reached event counters of the original embedded transitions.
/// These are capacity bounds, not observed work or structural selection cost.
/// The initializer has its own separate finite occurrence and is not included.
pub(super) fn actor_refresh_transition_native_work_ceilings(
    child: &SemanticTransitionSession,
) -> Result<[[u64; 9]; 2], SemanticTransitionError> {
    let build = child
        .prepared_segment
        .as_ref()
        .ok_or(SemanticTransitionError::NotCaptured)?;
    let handles = build.handles()?;
    if build.transitions
        != [
            SemanticTransitionKind::Recompute,
            SemanticTransitionKind::Proposal,
        ]
        || handles.len() != 2
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let storage = child
        .publication
        .as_ref()
        .ok_or(SemanticTransitionError::NotBound)?;
    let task = child
        .task
        .as_ref()
        .ok_or(SemanticTransitionError::NotBound)?;
    let queries = u64::try_from(task.0.spec.statement_records.len())
        .map_err(|_| native_work_ceiling_overflow())?;
    let editable = task.0.spec.program.editable_program().is_some();
    let components = &child.codebooks.components;
    if components.len() != COMPONENT_COUNT {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let targets = *child
        .codebooks
        .words
        .get(1)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    // An acquisition refusal can reach viability even for Recompute before
    // its successful non-Proposal early return. Enclose both actual branches.
    let viability = policy_viability_category_ceiling(child)?;
    let prefix = if editable {
        Some(rule_prefix_native_work_ceiling(targets)?)
    } else {
        None
    };
    let arena = child.graph.transition_arena();
    let roots = arena[2];
    let statements = arena[3];
    let supports = arena[4];
    let versions = arena[5];
    let records = ceiling_sum([statements, supports, versions])?;
    let receipt_bytes =
        std::mem::size_of_val(&task_ground::TaskQueryRecord::default().receipt) as u64;
    // execute clears the output, can copy one source receipt, and emits its
    // bounded command/handle/refusal words. Three receipts plus a whole command
    // also cover every fixed fill_view/fill_statement/fill_version field.
    let fixed_bytes = ceiling_sum([ceiling_product(3, receipt_bytes)?, 32 * 8])?;
    let snapshot = [1, 2, 0, 0, 0, 0, 0, 0, fixed_bytes];
    let truth = [
        1,
        ceiling_sum([ceiling_product(4, statements)?, 8])?,
        0,
        0,
        0,
        0,
        0,
        0,
        fixed_bytes,
    ];
    // retire_root validates and then marks each original root/candidate chain.
    // Each chain link visits version, statement, support, version and support;
    // the three complete record scans follow those two traces.
    let views = ceiling_sum([roots, 1])?;
    let chains = ceiling_product(ceiling_product(views, statements)?, versions)?;
    let trace_tables = ceiling_sum([
        ceiling_product(2, roots)?,
        2,
        ceiling_product(
            views,
            ceiling_sum([
                ceiling_product(2, statements)?,
                ceiling_product(5, ceiling_product(statements, versions)?)?,
            ])?,
        )?,
    ])?;
    let retirement = [
        1,
        ceiling_sum([
            ceiling_product(2, trace_tables)?,
            ceiling_product(3, records)?,
            3,
        ])?,
        ceiling_product(2, chains)?,
        0,
        0,
        0,
        0,
        0,
        ceiling_sum([
            fixed_bytes,
            ceiling_product(
                16,
                ceiling_product(
                    ceiling_product(views, statements)?,
                    ceiling_sum([versions, 1])?,
                )?,
            )?,
            ceiling_product(24, records)?,
            ceiling_product(8, statements)?,
        ])?,
    ];
    let mut result = [[0; 9]; 2];
    for index in 0..2 {
        let io = child.prepared_kernel_io(&handles[index], index)?;
        let descriptor = child.descriptor_with(&io);
        let ceiling = &mut result[index];
        // Base and both executed lanes share one protected graph predicate.
        repeat_native_ceiling(ceiling, attainment_native_work_ceiling(&task.0.spec)?, 3)?;
        // A numerical refusal snapshots the original root; an admitted slot
        // snapshots/preflights it instead. Keeping both covers either outcome.
        repeat_native_ceiling(ceiling, snapshot, 2)?;
        // One inactive-bank retirement and up to two freshly sealed roots.
        // Successful retirements are masked, so final refusal cleanup cannot
        // repeat them; a failed retirement stops that pass immediately.
        repeat_native_ceiling(ceiling, retirement, 4)?;

        let ground_bytes = descriptor.task_ground_bytes;
        let mut ground_digest = content_digest_ceiling(ground_bytes, 21, false)?;
        ground_digest[0] = 0; // Inline digest, not another native command.
        add_native_work_ceiling(ceiling, ground_digest)?;
        add_native_work_ceiling(
            ceiling,
            [
                0,
                storage.bank_templates[0].len() as u64,
                0,
                0,
                0,
                0,
                0,
                0,
                ceiling_product(2, ground_bytes)?,
            ],
        )?;
        // apply_continuation preserves these exact original non-tensor roles,
        // including their unused backing. Both publication banks are possible;
        // select the larger retained physical extent per canonical role/index.
        for original in storage.bank_templates[0].iter().filter(|range| {
            matches!(range.role, 32 | 44 | 56 | 57)
                || range.role == SemanticStateRole::TaskGround as u64
        }) {
            let mut bytes = 0;
            for bank in &storage.bank_templates {
                let range = bank
                    .iter()
                    .find(|range| range.role == original.role && range.index == original.index)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let slot = usize::try_from(range.storage_slot)
                    .map_err(|_| native_work_ceiling_overflow())?;
                let extent = storage
                    .allocations
                    .get(slot)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
                    .entry()
                    .bytes
                    .checked_sub(range.offset_bytes)
                    .ok_or_else(native_work_ceiling_overflow)?;
                bytes = bytes.max(extent);
            }
            let mut digest = content_digest_ceiling(bytes, 21, false)?;
            digest[0] = 0;
            add_native_work_ceiling(ceiling, digest)?;
            add_native_work_ceiling(ceiling, [0, 0, 0, 0, 0, 0, 0, 0, bytes])?;
        }
        if io.replay_custody.is_some() {
            // The original snapshot allocator reserves every copied backing,
            // header, directory and parent arena. Its allocation bound includes
            // both copy occurrences plus metadata which is not copied at all.
            let bytes = PreparedReplayCustody::allocation_bytes(
                storage,
                usize::try_from(arena[6]).map_err(|_| native_work_ceiling_overflow())?,
            )?;
            add_native_work_ceiling(ceiling, [0, 0, 0, 0, 0, 0, 0, 0, bytes])?;
        }
        if index == 0 {
            if io.policy.is_some() {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            add_native_work_ceiling(
                ceiling,
                [
                    0,
                    ceiling_product(2, viability)?,
                    0,
                    viability,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            )?;
            if let Some(prefix) = prefix {
                repeat_native_ceiling(ceiling, prefix, 4)?;
            }
            native_work_ceiling_units(*ceiling)?;
            continue;
        }
        let cells = ceiling_sum(
            components
                .iter()
                .map(|component| u64::from(component.cardinality)),
        )?;
        let edit_cells = ceiling_sum(
            components
                .iter()
                .filter(|component| !component.is_text())
                .map(|component| u64::from(component.cardinality)),
        )?;
        let text_cells = ceiling_sum(
            components
                .iter()
                .filter(|component| component.is_text())
                .map(|component| u64::from(component.cardinality)),
        )?;
        let edit_components = u64::try_from(
            components
                .iter()
                .filter(|component| !component.is_text())
                .count(),
        )
        .map_err(|_| native_work_ceiling_overflow())?;
        let policy = io.policy.ok_or(SemanticTransitionError::NotBound)?;
        // The original forward readout is now metered by the same owner as
        // backward recomputation. Each reached structural component visits its
        // 128 hidden coordinates and at most 130 coordinates per score row.
        add_native_work_ceiling(
            ceiling,
            [
                0,
                ceiling_sum([
                    ceiling_product(128, edit_components)?,
                    ceiling_product(130, edit_cells)?,
                ])?,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        )?;
        if let Some(prefix) = prefix {
            repeat_native_ceiling(
                ceiling,
                prefix,
                ceiling_sum([4, ceiling_product(2, edit_cells)?])?,
            )?;
            // Each of the four final edit fields can decode one original rule:
            // seven null choices, one ten-slot head predicate, two body and
            // six variable operands, one output, and the valid_rule check.
            repeat_native_ceiling(
                ceiling,
                [
                    0,
                    ceiling_sum([7, 12, ceiling_product(8, 8)?, 1])?,
                    0,
                    ceiling_sum([1, 7, 3, 6, 1])?,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                4,
            )?;
        }
        let parameter_cells = policy.buffers.parameters.len() as u64;
        // The same uniform-domain producer contributes its retained Category
        // tally: parameter initialization/export, recurrent initialization,
        // support scan, 128 alternatives in each forward and reverse edit,
        // two-lane text alternatives and public text-domain export. The hot
        // alias check and primal/critic admission then visit their own cells.
        let uniform = ceiling_sum([
            ceiling_product(2, parameter_cells)?,
            2 * 37 * 128,
            ceiling_product(3, COMPONENT_COUNT as u64)?,
            cells,
            ceiling_product(257, edit_cells)?,
            ceiling_product(128, TEXT_CARDINALITY as u64)?,
        ])?;
        let alias = ceiling_sum([
            parameter_cells,
            32 * TEXT_CARDINALITY as u64,
            COMPONENT_COUNT as u64,
        ])?;
        let primal = ceiling_sum([
            cells,
            text_cells,
            parameter_cells,
            32 * TEXT_CARDINALITY as u64,
            ceiling_product(2, COMPONENT_COUNT as u64)?,
        ])?;
        // Four edit viability banks: target_completes visits at most four full
        // operand banks with supported+operand_matches; all three opcodes are
        // checked once. Each category has two legal-category visits and each
        // visit can perform one additional operand_matches.
        let categories = ceiling_sum([
            uniform,
            alias,
            primal,
            viability,
            ceiling_product(4, cells)?,
        ])?;
        let mut comparisons = 0;
        for component in components {
            let width = u64::from(component.cardinality)
                .checked_next_power_of_two()
                .ok_or_else(native_work_ceiling_overflow)?;
            let levels = u64::from(width.trailing_zeros());
            let stages = ceiling_product(levels, ceiling_sum([levels, 1])?)? / 2;
            comparisons = ceiling_sum([comparisons, ceiling_product(width / 2, stages)?])?;
        }
        add_native_work_ceiling(
            ceiling,
            [
                0,
                ceiling_product(
                    2,
                    ceiling_sum([viability, ceiling_product(2, edit_cells)?])?,
                )?,
                0,
                categories,
                comparisons,
                ceiling_product(2, cells)?,
                ceiling_product(10, COMPONENT_COUNT as u64)?,
                0,
                ceiling_product(
                    COMPONENT_COUNT as u64,
                    size_of::<SemanticTransitionReceipt>() as u64,
                )?,
            ],
        )?;

        // One preflight, two forks, four insertions, two terminals and one
        // live-candidate cleanup. Each seal also reserves its internal discard.
        let preflight = [
            1,
            ceiling_sum([3, roots, records, ceiling_product(4, statements)?])?,
            0,
            0,
            0,
            0,
            0,
            0,
            fixed_bytes,
        ];
        let fork = [
            1,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            ceiling_sum([fixed_bytes, ceiling_product(8, statements)?])?,
        ];
        let insertion = [
            1,
            ceiling_sum([
                20,
                ceiling_product(5, statements)?,
                supports,
                ceiling_product(10, versions)?,
            ])?,
            versions,
            0,
            0,
            0,
            0,
            ceiling_sum([8, ceiling_product(2, versions)?])?,
            ceiling_sum([fixed_bytes, 2048, ceiling_product(288, versions)?])?,
        ];
        let terminal = [
            2,
            ceiling_sum([
                12,
                roots,
                ceiling_product(3, records)?,
                ceiling_product(2, statements)?,
            ])?,
            0,
            0,
            0,
            0,
            0,
            0,
            ceiling_sum([
                ceiling_product(2, fixed_bytes)?,
                ceiling_product(32, records)?,
                ceiling_product(24, statements)?,
            ])?,
        ];
        add_native_work_ceiling(ceiling, preflight)?;
        repeat_native_ceiling(ceiling, fork, 2)?;
        repeat_native_ceiling(ceiling, insertion, 4)?;
        repeat_native_ceiling(ceiling, terminal, 3)?;
        if editable {
            // materialize checks charged==fuel before every increment, including
            // failure. Exactly three task_program_queries can reach the owner.
            add_native_work_ceiling(
                ceiling,
                [
                    0,
                    3 * 1_000_000,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    ceiling_product(ceiling_product(3, queries)?, receipt_bytes)?,
                ],
            )?;
            let rules = crate::semantic_program::RESIDENT_PROGRAM_RULE_CAPACITY as u64;
            add_native_work_ceiling(
                ceiling,
                [
                    0,
                    ceiling_product(2, ceiling_sum([rules, 2])?)?,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2 * std::mem::size_of::<crate::SemanticProgramRule>() as u64 + 8,
                ],
            )?;
        } else {
            // Baseline, two candidates and final feedback use the same truth owner.
            repeat_native_ceiling(ceiling, truth, ceiling_product(4, queries)?)?;
        }
        // Original query rows are zeroed once before sampling.
        add_native_work_ceiling(
            ceiling,
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                ceiling_product(
                    ceiling_product(3, queries)?,
                    size_of::<task_ground::TaskQueryRecord>() as u64,
                )?,
            ],
        )?;
        native_work_ceiling_units(*ceiling)?;
    }
    Ok(result)
}

pub(super) fn tensor_layout_native_work_ceiling(
    layout: &SemanticTensorLayout,
    logical_begin: u64,
    logical_end: u64,
    physical_bytes: usize,
    verify: bool,
) -> Result<[u64; 9], SemanticTransitionError> {
    tensor_content_range(layout, logical_begin, logical_end, physical_bytes)?;
    let dimensions = &layout.dimensions[..layout.rank as usize];
    let logical_bytes = if dimensions.contains(&0) {
        0
    } else {
        dimensions
            .iter()
            .try_fold(layout.element_bytes, |bytes, dimension| {
                bytes
                    .checked_mul(*dimension)
                    .ok_or_else(native_work_ceiling_overflow)
            })?
    };
    content_digest_ceiling(logical_bytes, 25 + layout.rank, !verify)
}

pub(super) fn content_digest_ceiling(
    content_bytes: u64,
    prefix_words: u64,
    seal: bool,
) -> Result<[u64; 9], SemanticTransitionError> {
    // `publication_content_digest` hashes its sixteen-word prefix and logical
    // bytes. SHA-256 adds nine padding bytes, rounded up to complete blocks.
    let blocks = content_bytes
        .checked_add(16 * size_of::<u64>() as u64)
        .and_then(|bytes| bytes.checked_add(9 + 63))
        .ok_or_else(native_work_ceiling_overflow)?
        / 64;
    let canonical_bytes = prefix_words
        .checked_mul(size_of::<u64>() as u64)
        .and_then(|bytes| bytes.checked_add(32))
        .and_then(|bytes| bytes.checked_add(if seal { 32 } else { 0 }))
        .ok_or_else(native_work_ceiling_overflow)?;
    // Command, TableSlot, ChainLink, Category, SortComparison, CdfStep,
    // PhiloxRound, ShaBlock, CanonicalByte.
    Ok([1, 0, 0, 0, 0, 0, 0, blocks, canonical_bytes])
}

impl PreparedStepInputs {
    pub(super) fn native_work_ceiling(&self) -> Result<[u64; 9], SemanticTransitionError> {
        Self::basis_native_work_ceiling(&self.storage, &self.plans)
    }

    fn basis_native_work_ceiling(
        storage: &PublicationStorage,
        plans: &[StepInputPlan],
    ) -> Result<[u64; 9], SemanticTransitionError> {
        step_input_basis_geometry_native_work_ceiling(plans, |slot| {
            storage
                .allocations
                .get(slot)
                .map(|allocation| allocation.entry().bytes)
                .ok_or(SemanticTransitionError::ObservationMismatch)
        })
    }
}

pub(super) fn step_input_basis_geometry_native_work_ceiling(
    plans: &[StepInputPlan],
    allocation_bytes: impl Fn(usize) -> Result<u64, SemanticTransitionError>,
) -> Result<[u64; 9], SemanticTransitionError> {
    // One original guard, two metadata seals, and the fixed roster fold.
    let mut ceiling = [1, 0, 0, 0, 0, 0, 0, 2, 112];
    for (bytes, prefix_words) in [
        (size_of::<PublicationHeader>() as u64, 26),
        ((32 * size_of::<SemanticTextSlot>()) as u64, 27),
    ] {
        let mut digest = content_digest_ceiling(bytes, prefix_words, false)?;
        digest[0] = 0;
        add_native_work_ceiling(&mut ceiling, digest)?;
    }
    let count = u64::try_from(plans.len()).map_err(|_| native_work_ceiling_overflow())?;
    let previous = count
        .checked_sub(1)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    ceiling[1] = count
        .checked_mul(previous)
        .map(|visits| visits / 2)
        .and_then(|visits| visits.checked_add(count))
        .ok_or_else(native_work_ceiling_overflow)?;
    for plan in plans {
        let mut digest = if matches!(plan.role, 4..=13 | 18..=25) {
            let end = if plan.layout.logical_axis == u64::MAX {
                0
            } else {
                plan.layout.dimensions[plan.layout.logical_axis as usize]
            };
            tensor_layout_native_work_ceiling(&plan.layout, 0, end, plan.banks[0].span.len(), true)?
        } else {
            content_digest_ceiling(plan.banks[0].span.len() as u64, 21, false)?
        };
        digest[0] = 0;
        add_native_work_ceiling(&mut ceiling, digest)?;
    }
    let mut backing_ceiling = [0; 9];
    for bank in 0..2 {
        let mut selected = [0; 9];
        let mut seen = BTreeSet::new();
        for plan in plans.iter().filter(|plan| matches!(plan.role, 18..=25)) {
            let slot = plan.banks[bank].storage_slot;
            if !seen.insert(slot) {
                continue;
            }
            let blocks = allocation_bytes(slot)?
                .checked_add(9 + 63)
                .ok_or_else(native_work_ceiling_overflow)?
                / 64;
            add_native_work_ceiling(&mut selected, [0, 0, 0, 0, 0, 0, 0, blocks, 32])?;
        }
        for (upper, actual) in backing_ceiling.iter_mut().zip(selected) {
            *upper = (*upper).max(actual);
        }
    }
    add_native_work_ceiling(&mut ceiling, backing_ceiling)?;
    Ok(ceiling)
}

pub(super) fn step_input_plans(
    storage: &PublicationStorage,
    directory: &[PublicationRange],
) -> Result<Vec<StepInputPlan>, SemanticTransitionError> {
    let mut plans = PreparedStepInputs::plan(storage)?;
    let slots = storage.model_slots_for_directory(directory)?;
    for plan in &mut plans {
        if matches!(plan.role, 18..=25) {
            let allocation = storage.model_memory.location(plan.role, plan.index)?.0;
            for bank in 0..2 {
                plan.banks[bank].storage_slot = slots[allocation][bank];
            }
        }
    }
    Ok(plans)
}

pub(super) fn step_input_dma_ceiling(count: usize) -> Result<[u64; 9], SemanticTransitionError> {
    let bytes = count
        .checked_mul(size_of::<PublicationStepInput>())
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(native_work_ceiling_overflow)?;
    Ok([
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        u64::try_from(bytes).map_err(|_| native_work_ceiling_overflow())?,
    ])
}

pub(super) fn step_input_producer_native_work_ceiling(
    storage: &PublicationStorage,
    plans: &[StepInputPlan],
    directory: &[PublicationRange],
) -> Result<[u64; 9], SemanticTransitionError> {
    step_input_producer_geometry_native_work_ceiling(
        &storage.contract_value,
        storage.layouts.len(),
        storage.allocations.len(),
        storage.role_counts.len(),
        plans,
        directory,
        |slot| {
            storage
                .allocations
                .get(slot)
                .map(|allocation| allocation.entry().bytes)
                .ok_or(SemanticTransitionError::ObservationMismatch)
        },
    )
}

pub(super) fn step_input_producer_geometry_native_work_ceiling(
    contract: &PublicationContract,
    layout_count: usize,
    allocation_count: usize,
    role_count: usize,
    plans: &[StepInputPlan],
    directory: &[PublicationRange],
    allocation_bytes: impl Fn(usize) -> Result<u64, SemanticTransitionError>,
) -> Result<[u64; 9], SemanticTransitionError> {
    let mut ceiling = step_input_basis_geometry_native_work_ceiling(plans, allocation_bytes)?;
    // The selected-model check is a strict prefix of the full role44 guard.
    // Retaining that complete guard ceiling is a conservative native bound.
    add_native_work_ceiling(
        &mut ceiling,
        model_contract_guard_geometry_native_work_ceiling(contract, directory)?,
    )?;
    let table = directory
        .iter()
        .find(|range| range.role == 55 && range.index == 0)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let mut table_digest = content_digest_ceiling(table.length_bytes, 21, false)?;
    table_digest[0] = 0;
    add_native_work_ceiling(&mut ceiling, table_digest)?;
    let n = u64::try_from(directory.len()).map_err(|_| native_work_ceiling_overflow())?;
    let t = u64::try_from(layout_count).map_err(|_| native_work_ceiling_overflow())?;
    let k = u64::try_from(plans.len()).map_err(|_| native_work_ceiling_overflow())?;
    let s = u64::try_from(allocation_count).map_err(|_| native_work_ceiling_overflow())?;
    let d = contract
        .window_capacity
        .checked_add(contract.feedback_capacity)
        .ok_or_else(native_work_ceiling_overflow)?;
    let sum = |a: u64, b: u64| a.checked_add(b).ok_or_else(native_work_ceiling_overflow);
    let product = |a: u64, b: u64| a.checked_mul(b).ok_or_else(native_work_ceiling_overflow);
    let pairs = |count: u64| product(count, count.saturating_sub(1)).map(|value| value / 2);
    // Storage validation; role/table/layout validation; original output alias
    // scans; every binding's full directory/layout scans and fixed layout words;
    // mutually exclusive previous-binding scans; final copied range lookups.
    let mut visits = sum(s, pairs(s)?)?;
    visits = sum(
        visits,
        u64::try_from(role_count).map_err(|_| native_work_ceiling_overflow())?,
    )?;
    visits = sum(visits, sum(6, d)?)?;
    visits = sum(visits, sum(n, product(t, sum(n, 1)?)?)?)?;
    visits = sum(visits, pairs(t)?)?;
    visits = sum(visits, product(n, sum(sum(sum(1, n)?, t)?, d)?)?)?;
    visits = sum(visits, 22 + 44 + 10)?;
    visits = sum(visits, product(8, sum(s, 2)?)?)?;
    let words = (size_of::<SemanticTensorLayout>() / 8) as u64;
    visits = sum(
        visits,
        product(k, sum(sum(sum(1, product(2, n)?)?, t)?, words)?)?,
    )?;
    visits = sum(visits, product(pairs(k)?, sum(n, 1)?)?)?;
    visits = sum(visits, product(k, sum(product(2, sum(s, 2)?)?, 4)?)?)?;
    visits = sum(visits, product(k, sum(n, 1)?)?)?;
    add_native_work_ceiling(
        &mut ceiling,
        [0, visits, 0, 0, 0, 0, 0, 0, product(n, 29 * 8)?],
    )?;
    let bytes = sum(
        sum(64, size_of::<PublicationHeader>() as u64)?,
        (32 * size_of::<SourceSlot>()) as u64,
    )?;
    add_native_work_ceiling(
        &mut ceiling,
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            sum(bytes, product(k, size_of::<PublicationRange>() as u64)?)?,
        ],
    )?;
    for plan in plans
        .iter()
        .filter(|plan| !matches!(plan.role, 1 | 4 | 5 | 18..=25))
    {
        add_native_work_ceiling(
            &mut ceiling,
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                u64::try_from(plan.banks[0].span.len())
                    .map_err(|_| native_work_ceiling_overflow())?,
            ],
        )?;
    }
    Ok(ceiling)
}

impl TensorContentBuffers {
    /// Bound one actual enqueue from its retained original geometry and seals.
    pub(super) fn native_work_ceiling(
        &self,
        verify: bool,
    ) -> Result<[u64; 9], SemanticTransitionError> {
        let mut ceiling = [0; 9];
        match &self.seals {
            TensorContentSeals::Model(model) => {
                if !verify || model.ranges.len() != self.tensors.len() + 1 {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                // The contract guard has no tensor layout: sixteen zeroed
                // prefix words plus five original range fields, then a digest.
                add_native_work_ceiling(
                    &mut ceiling,
                    content_digest_ceiling(model.contract.len() as u64, 21, false)?,
                )?;
            }
            TensorContentSeals::Captured(digests) => {
                if digests.len() != self.tensors.len() {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
            }
        }
        for (ordinal, tensor) in self.tensors.iter().enumerate() {
            let verify = match &self.seals {
                TensorContentSeals::Captured(digests) => match &digests[ordinal] {
                    CapturedTensorDigest::Publication(inputs) => {
                        if !digests[..ordinal].iter().any(|digest| matches!(digest,
                            CapturedTensorDigest::Publication(previous) if Arc::ptr_eq(inputs, previous))) {
                            add_native_work_ceiling(&mut ceiling, inputs.native_work_ceiling()?)?;
                        }
                        continue;
                    }
                    CapturedTensorDigest::Tensor {
                        producer_sealed, ..
                    } => verify || *producer_sealed,
                },
                TensorContentSeals::Model(_) => true,
            };
            let physical_bytes = tensor.source.as_ref().map_or(0, |source| source.len());
            // A tensor adds its four layout fields and every logical dimension
            // to the same original range prefix used by publication seals.
            add_native_work_ceiling(
                &mut ceiling,
                tensor_layout_native_work_ceiling(
                    &tensor.layout,
                    tensor.logical_begin,
                    tensor.logical_end,
                    physical_bytes,
                    verify,
                )?,
            )?;
        }
        Ok(ceiling)
    }
}
