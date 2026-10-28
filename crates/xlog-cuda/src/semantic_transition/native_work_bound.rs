pub(super) use super::native_work_report::{
    add_native_work_ceiling, ceiling_product, ceiling_sum, native_work_ceiling_overflow,
    native_work_ceiling_units, publication_hash_ceiling,
    selected_model_origin_geometry_native_work_ceiling, selected_model_origin_native_work_ceiling,
};

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
}

pub(super) fn policy_uniform_native_work_ceiling(
    components: &[SemanticComponent],
    parameter_cells: u64,
) -> Result<[u64; 9], SemanticTransitionError> {
    let cells = ceiling_sum(
        components
            .iter()
            .map(|component| u64::from(component.cardinality)),
    )?;
    let edits = ceiling_sum(
        components
            .iter()
            .filter(|component| !component.is_text())
            .map(|component| u64::from(component.cardinality)),
    )?;
    Ok([
        1,
        0,
        0,
        ceiling_sum([
            ceiling_product(2, parameter_cells)?,
            2 * 37 * 128,
            ceiling_product(3, COMPONENT_COUNT as u64)?,
            cells,
            ceiling_product(257, edits)?,
            ceiling_product(128, TEXT_CARDINALITY as u64)?,
        ])?,
        0,
        0,
        0,
        0,
        0,
    ])
}

pub(super) fn policy_binding_native_work_ceiling(
    components: &[SemanticComponent],
    layouts: &[SemanticTensorLayout; 9],
) -> Result<[u64; 9], SemanticTransitionError> {
    let mut ceiling = policy_uniform_native_work_ceiling(components, layouts[2].dimensions[0])?;
    add_native_work_ceiling(&mut ceiling, [9, 0, 0, 0, 0, 0, 0, 0, 0])?;
    for layout in layouts {
        let bytes = tensor_layout_bytes(layout)?;
        ceiling[8] = ceiling[8]
            .checked_add(bytes as u64)
            .ok_or_else(native_work_ceiling_overflow)?;
    }
    Ok(ceiling)
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

/// The original continuation retains six FP64 bounds per physical block.
pub(super) fn numerical_block_snapshot_native_work_ceiling(
    physical_blocks: usize,
) -> Result<[u64; 9], SemanticTransitionError> {
    let blocks = u64::try_from(physical_blocks).map_err(|_| native_work_ceiling_overflow())?;
    let bytes = ceiling_product(blocks, 6 * size_of::<f64>() as u64)?;
    let ceiling = [1, 0, 0, 0, 0, 0, 0, 0, bytes];
    native_work_ceiling_units(ceiling)?;
    Ok(ceiling)
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
    evaluation_preparation: Option<FrozenEvaluationNativePreparation>,
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
    initial_preparation_budget: Option<[u64; 3]>,
}

/// Ordered producer geometry frozen before the imported context's first model
/// effect. Actual tensor owners are attached only by that original producer.
pub(super) struct FrozenModelContextPrefillRoster {
    owner: std::sync::Weak<model_context_continuation::ModelContextContinuationOwner>,
    _reservation: SemanticReplayMaterialReservation,
}

/// Structural inference preparation has two original model-roster uploads and
/// one canonical step-input initializer, not an early resident execution.
pub(super) struct FrozenModelContextPreparation {
    pub(super) ceiling: [u64; 9],
    dma: [u64; 9],
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
        let mut ceiling = [0; 9];
        let mut dma = [0; 9];
        if let Some((native, uploads)) = step_input {
            add_native_work_ceiling(&mut ceiling, native)?;
            add_native_work_ceiling(&mut ceiling, uploads)?;
            add_native_work_ceiling(&mut dma, uploads)?;
        }
        native_work_ceiling_units(ceiling)?;
        Ok(Self {
            ceiling,
            dma,
            step_input,
        })
    }
}

impl FrozenModelContextPrefillRoster {
    pub(super) fn new(
        context: &SemanticModelContextContinuation,
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<Self, SemanticTransitionError> {
        Ok(Self {
            owner: Arc::downgrade(&context.inner),
            _reservation: budget.reserve_auxiliary(size_of::<Self>() as u64)?,
        })
    }
}

/// Exact original read occurrences, ordered by their enclosing report regions.
/// Geometry remains metadata; only the actual issued producer can claim it.
pub(super) struct FrozenColdResidentReads {
    reads: Vec<(usize, SemanticResidentModelRead, bool)>,
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
                    size_of::<(usize, SemanticResidentModelRead, bool)>()
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
            original.push((*region, read.clone(), false));
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
            .any(|(original, _, _)| *original == region)
        {
            return Err(publication_input_error(
                "resident read skip cannot replace an entered original occurrence",
            ));
        }
        for (original, _, skipped) in &mut self.reads[self.cursor..] {
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
            .is_some_and(|(_, _, skipped)| *skipped)
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

pub(super) struct FrozenEvaluationNativePreparation {
    pub(super) invocation: Arc<()>,
    pub(super) ceiling: [u64; 9],
    pub(super) available: bool,
    pub(super) step_input_preparation: Option<([u64; 9], [u64; 9])>,
}

impl ColdNativeAllowance {
    pub(super) fn freeze_evaluation_preparation(
        &mut self,
        preparation: FrozenEvaluationNativePreparation,
    ) -> Result<(), SemanticTransitionError> {
        if !self.is_evaluation() || self.native_initialization_claimed {
            return Err(publication_input_error(
                "evaluation native preparation must precede its original admitted execution",
            ));
        }
        if let Some(original) = &self.evaluation_preparation {
            if !Arc::ptr_eq(&original.invocation, &preparation.invocation)
                || original.ceiling != preparation.ceiling
                || original.step_input_preparation != preparation.step_input_preparation
            {
                return Err(publication_input_error(
                    "evaluation native preparation changed its original workload owners",
                ));
            }
            return Ok(());
        }
        native_work_ceiling_units(preparation.ceiling)?;
        self.evaluation_preparation = Some(preparation);
        Ok(())
    }

    pub(super) fn new(purpose: SemanticColdModelWorkPurpose) -> Self {
        Self {
            purpose,
            evaluation_preparation: None,
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
            initial_preparation_budget: None,
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

    pub(super) fn freeze_model_context_report_dma(
        &mut self,
        events: usize,
    ) -> Result<(), SemanticTransitionError> {
        if !self.is_model_context_report() || self.model_scratch_ceiling.is_some() {
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
        let mut ceiling = dma;
        if let Some(original) = self.model_context_ceiling {
            add_native_work_ceiling(&mut ceiling, original)?;
        }
        add_native_work_ceiling(&mut self.submitted_dma_ceiling, dma)?;
        self.model_context_ceiling = Some(ceiling);
        Ok(())
    }

    pub(super) fn is_model_context_report(&self) -> bool {
        self.initial_preparation_budget.is_some()
            || self.model_context_ceiling.is_some()
            || self.model_context_prefill.is_some()
            || self.model_context_preparation.is_some()
            || self.model_context_activation_ceiling.is_some()
    }

    pub(super) fn require_model_context_source_budget(
        &self,
        model: [u64; 3],
    ) -> Result<(), SemanticTransitionError> {
        if let Some(budget) = self
            .model_context_source_budget
            .or(self.initial_preparation_budget)
        {
            let total = native_work_ceiling_units(self.native_ceiling()?)?
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

    pub(super) fn freeze_initial_prefill_preparation(
        &mut self,
        budget: [u64; 3],
    ) -> Result<(), SemanticTransitionError> {
        if self.purpose != SemanticColdModelWorkPurpose::PrivateModelPreparation
            || self.initial_preparation_budget.is_some()
            || self.model_scratch_ceiling.is_some()
            || self.native_initialization_claimed
        {
            return Err(publication_input_error(
                "initial preparation requires its original unopened program allowance",
            ));
        }
        self.initial_preparation_budget = Some(budget);
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
        let mut ceiling = self.native_ceiling()?;
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
        self.model_context_prefill = Some(roster);
        self.model_context_source_budget = Some(budget);
        Ok(())
    }

    pub(super) fn set_model_context_prefill_recording(&mut self, recording: bool) {
        self.model_context_prefill_recording = recording && self.model_context_prefill.is_some();
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
    ) -> Result<(), SemanticTransitionError> {
        let reads = self.resident_reads.as_mut().ok_or_else(|| {
            publication_input_error("resident read precedes its original finite cold admission")
        })?;
        let (original_region, read, _) = reads.reads.get(reads.cursor).ok_or_else(|| {
            publication_input_error("resident read exceeded its original ordered roster")
        })?;
        if *original_region != region {
            return Err(publication_input_error(
                "resident read changed its original region or order",
            ));
        }
        read.claim_actual(owner)?;
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
        if index >= reads.cursor || reads.reads[index].2 {
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
        let (_, region_read, _) = reads.reads.get(reads.cursor).ok_or_else(|| {
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
            .is_some_and(|roster| roster.step_input.is_some())
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

    pub(super) fn claim_source_resident_owner(
        &mut self,
        owner: &Arc<super::resident_model_memory::ResidentModelMemoryOwner>,
    ) -> Result<(), SemanticTransitionError> {
        let source = self.source.as_mut().ok_or_else(|| {
            publication_input_error(
                "resident SOURCE owner precedes its complete original finite admission",
            )
        })?;
        let recipe = source
            .recipes
            .iter_mut()
            .find(|recipe| recipe.resident.ptr_eq(&Arc::downgrade(owner)))
            .ok_or_else(|| {
                publication_input_error("resident owner is outside its original SOURCE roster")
            })?;
        if !recipe.preparation_claimed || recipe.resident_claimed {
            return Err(publication_input_error(
                "resident SOURCE guard changed or repeated its original finite producer",
            ));
        }
        recipe.resident_claimed = true;
        Ok(())
    }

    pub(super) fn native_ceiling(&self) -> Result<[u64; 9], SemanticTransitionError> {
        let mut ceiling = if self.purpose == SemanticColdModelWorkPurpose::PrivatePrefix {
            self.source.as_ref().map(|source| source.ceiling)
                .ok_or_else(|| publication_input_error("Prefix content must include its full original SOURCE roster before effects"))?
        } else {
            match &self.evaluation_preparation {
                Some(preparation) => Ok(preparation.ceiling),
                None if !self.is_evaluation() => Ok([0; 9]),
                None => Err(publication_input_error(
                    "evaluation native work must be admitted from its original preparation owners",
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

    /// The actual evaluation owner claims its complete original preparation
    /// once, before any selection or source observation can enter the driver.
    pub(super) fn claim_evaluation_preparation(
        &mut self,
        invocation: &Arc<()>,
    ) -> Result<(), SemanticTransitionError> {
        if !self.is_evaluation() {
            return Ok(());
        }
        let content = self.evaluation_preparation.as_mut().ok_or_else(|| {
            publication_input_error("evaluation preparation precedes its original admission")
        })?;
        if !Arc::ptr_eq(&content.invocation, invocation) || !content.available {
            return Err(publication_input_error(
                "evaluation preparation differs from its sole admitted original invocation",
            ));
        }
        content.available = false;
        Ok(())
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
        let content = self.evaluation_preparation.as_mut().ok_or_else(|| {
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
        native_work_report::record_submitted_native_dma(
            &mut self.submitted_dma,
            self.submitted_dma_ceiling,
            bytes,
            recorded,
        )
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
                .zip(self.native_ceiling()?)
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
    let inputs = [
        0,
        ceiling_sum([
            8,
            ceiling_product(ceiling_sum([count, 1])?, ranges)?,
            ceiling_product(4, ceiling_sum([count, 1])?)?,
        ])?,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];
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
        // pair checks the fuel before increment, including failed prefixes,
        // inherited read-only observation rows, protected-head scans and all
        // inherited provenance words. The limit is unchanged by a fork.
        add_native_work_ceiling(
            &mut ceiling,
            [0, ceiling_sum([ranges, 1_000_000])?, 0, 0, 0, 0, 0, 0, 0],
        )?;
    }
    native_work_ceiling_units(ceiling)?;
    Ok(ceiling)
}

pub(super) struct FrozenSourceRecipe {
    pub(super) owner: std::sync::Weak<prepared_source_content::PreparedSourceContentOwner>,
    pub(super) resident: std::sync::Weak<super::resident_model_memory::ResidentModelMemoryOwner>,
    pub(super) preparation_claimed: bool,
    pub(super) resident_claimed: bool,
    pub(super) dma_ceiling: [u64; 9],
    pub(super) step_input_preparation: Option<([u64; 9], [u64; 9])>,
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
    let mut ceiling = super::actor_refresh::actor_initialize_native_work_ceiling(target)?;
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
        selected_model_origin_geometry_native_work_ceiling(
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
    ])?;
    for role in [42, 45, 46, 47, 50] {
        let bytes = range(source_directory, role)?.length_bytes;
        visits = ceiling_sum([visits, 1, ns, nt, 4, bytes])?;
    }
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
    add_native_work_ceiling(
        &mut ceiling,
        [
            0,
            ceiling_sum([
                ceiling_product(5, ns)?,
                ceiling_product(2, nt)?,
                ceiling_product(21, intents)?,
                ceiling_product(17, acknowledgments)?,
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
                ceiling_sum([ack_payload, ceiling_product(144, acknowledgments)?])? / 64,
                ceiling_product(6, acknowledgments)?,
            ])?,
            ceiling_sum([
                ceiling_product(32, ceiling_sum([intents, 1])?)?,
                ceiling_product(344, intents)?,
                ceiling_product(64, acknowledgments)?,
                ceiling_product(336, acknowledgments)?,
            ])?,
        ],
    )?;

    // Activation derives one branch origin and refreshes its bounded completion receipt.
    add_native_work_ceiling(&mut ceiling, hash(212)?)?;
    add_native_work_ceiling(&mut ceiling, [0, 0, 0, 0, 0, 0, 0, 0, 388])?;
    add_native_work_ceiling(
        &mut ceiling,
        publication_record_identity_native_work_ceiling(size_of::<CompletionCoverage>() as u64)?,
    )?;
    native_work_ceiling_units(ceiling)?;
    Ok(ceiling)
}

pub(super) fn repeat_native_ceiling(
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
        if source
            .actor_refresh
            .as_ref()
            .is_some_and(|staged| staged.proof.is_demonstration())
        {
            // Original choice, current choice, draw, and all retained key/counter/random words.
            ceiling_product(13, component_count)?
        } else {
            0
        },
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

fn text_binding_validation_native_work_ceiling() -> Result<[u64; 9], SemanticTransitionError> {
    // Original source/selection roster, original text rows, and the optional
    // non-Proposal selected-row scan all have the same fixed 32-slot bank.
    Ok([
        0,
        3 * 32,
        0,
        0,
        0,
        0,
        0,
        0,
        32 * size_of::<SemanticTextRow>() as u64,
    ])
}

fn active_origin_validation_native_work_ceiling(
    n: u64,
    feedback: u64,
) -> Result<[u64; 9], SemanticTransitionError> {
    let rows = ceiling_sum([32, feedback])?;
    Ok([
        0,
        ceiling_sum([
            n,                         // Original feedback-directory search.
            ceiling_product(2, rows)?, // Shape and matched active-row visits.
            2 * 32 * 32,               // Each kind's bounded source selection passes.
            feedback,
        ])?,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ])
}

fn continuation_validation_native_work_ceiling(
    storage: &PublicationStorage,
) -> Result<[u64; 9], SemanticTransitionError> {
    continuation_validation_geometry_native_work_ceiling(
        &storage.contract_value,
        storage.bank_templates[0].len(),
        storage.continuation_templates.len(),
        storage.layouts.len(),
        storage.allocations.len(),
    )
}

fn model_update_validation_native_work_ceiling(
    n: u64,
    s: u64,
) -> Result<u64, SemanticTransitionError> {
    let pairs = ceiling_product(n, n.saturating_sub(1))? / 2;
    ceiling_sum([
        n,
        pairs,
        ceiling_product(n, n)?,
        ceiling_product(n, ceiling_sum([3, n, s])?)?,
        pairs,
    ])
}

fn continuation_validation_geometry_native_work_ceiling(
    contract: &PublicationContract,
    directory_count: usize,
    continuation_count: usize,
    layout_count: usize,
    allocation_count: usize,
) -> Result<[u64; 9], SemanticTransitionError> {
    let n = u64::try_from(directory_count).map_err(|_| native_work_ceiling_overflow())?;
    let p = u64::try_from(continuation_count).map_err(|_| native_work_ceiling_overflow())?;
    // Source, authority and layout-table are the original three non-tensor
    // pending roles. The validator itself requires precisely this roster.
    let t = p.checked_sub(3).ok_or_else(native_work_ceiling_overflow)?;
    let b = u64::try_from(layout_count).map_err(|_| native_work_ceiling_overflow())?;
    let s = u64::try_from(allocation_count).map_err(|_| native_work_ceiling_overflow())?;
    let rows = ceiling_sum([32, contract.feedback_capacity])?;
    let pairs = |count: u64| -> Result<u64, SemanticTransitionError> {
        Ok(ceiling_product(count, count.saturating_sub(1))? / 2)
    };
    // A malformed private Update binding count is rejected above n. Include
    // every reached directory, binding, storage and prior-binding visit up to
    // that original directory geometry, not merely the successful allocation count.
    let model_update = model_update_validation_native_work_ceiling(n, s)?;
    let slots = ceiling_sum([
        4 * 4, // Original instance, topology, table and prefix identities.
        model_update,
        n,
        p,
        n, // Prefix search, pending tensor table, expected-directory scan.
        ceiling_product(t, ceiling_sum([1, p])?)?,
        pairs(t)?,
        ceiling_product(p, ceiling_sum([1, n])?)?,
        pairs(p)?,
        1,             // Original authority storage entry.
        32 * (1 + 32), // Source rows and their complete original source search.
        ceiling_product(t, ceiling_sum([p, t, n, b, 4, rows])?)?,
    ])?;
    let mut result = [0, slots, 0, 0, 0, 0, 0, 0, 0];
    add_native_work_ceiling(&mut result, text_binding_validation_native_work_ceiling()?)?;
    add_native_work_ceiling(
        &mut result,
        active_origin_validation_native_work_ceiling(n, contract.feedback_capacity)?,
    )?;
    native_work_ceiling_units(result)?;
    Ok(result)
}

/// Extent validation and the actual snapshot each resolve every original row
/// once. The copy kernel shares that resolution across its byte-copy threads.
pub(super) fn replay_metadata_native_work_ceiling(
    rows: u64,
    directory_count: u64,
) -> Result<[u64; 9], SemanticTransitionError> {
    Ok([
        0,
        ceiling_product(
            2,
            ceiling_product(rows, ceiling_sum([directory_count, 2])?)?,
        )?,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ])
}

pub(super) fn continuation_preparation_native_work_ceiling(
    storage: &PublicationStorage,
) -> Result<[u64; 9], SemanticTransitionError> {
    continuation_preparation_geometry_native_work_ceiling(
        &storage.contract_value,
        &storage.bank_templates[0],
        storage.continuation_templates.len(),
        storage.layouts.len(),
        storage.allocations.len(),
        continuation_copy_capacity_native_work_ceiling(&storage.continuation_templates, |slot| {
            storage
                .allocations
                .get(slot)
                .map(|allocation| allocation.entry().bytes)
                .ok_or(SemanticTransitionError::ObservationMismatch)
        })?,
    )
}

pub(super) fn continuation_copy_capacity_native_work_ceiling(
    ranges: &[PublicationRange],
    allocation_bytes: impl Fn(usize) -> Result<u64, SemanticTransitionError>,
) -> Result<u64, SemanticTransitionError> {
    ranges
        .iter()
        .filter(|range| is_tensor_role(range.role))
        .try_fold(0, |total, range| {
            let slot =
                usize::try_from(range.storage_slot).map_err(|_| native_work_ceiling_overflow())?;
            ceiling_sum([
                total,
                allocation_bytes(slot)?
                    .checked_sub(range.offset_bytes)
                    .ok_or_else(native_work_ceiling_overflow)?,
            ])
        })
}

pub(super) fn continuation_preparation_geometry_native_work_ceiling(
    contract: &PublicationContract,
    directory: &[PublicationRange],
    continuation_count: usize,
    layout_count: usize,
    allocation_count: usize,
    copy_capacity: u64,
) -> Result<[u64; 9], SemanticTransitionError> {
    let mut result = continuation_validation_geometry_native_work_ceiling(
        contract,
        directory.len(),
        continuation_count,
        layout_count,
        allocation_count,
    )?;
    let n = u64::try_from(directory.len()).map_err(|_| native_work_ceiling_overflow())?;
    let p = u64::try_from(continuation_count).map_err(|_| native_work_ceiling_overflow())?;
    let s = u64::try_from(allocation_count).map_err(|_| native_work_ceiling_overflow())?;
    let rows = ceiling_sum([32, contract.feedback_capacity])?;
    // Preparation validates text and active origin before the final complete
    // continuation validator. Admission later has its separate original meter.
    add_native_work_ceiling(&mut result, text_binding_validation_native_work_ceiling()?)?;
    add_native_work_ceiling(
        &mut result,
        active_origin_validation_native_work_ceiling(n, contract.feedback_capacity)?,
    )?;
    add_native_work_ceiling(&mut result, [0, ceiling_sum([n, 32])?, 0, 0, 0, 0, 0, 0, 0])?;
    let mut guard = selected_model_origin_geometry_native_work_ceiling(contract, directory)?;
    guard[0] = 0;
    add_native_work_ceiling(&mut result, guard)?;
    add_native_work_ceiling(
        &mut result,
        [
            0,
            ceiling_sum([
                12,
                // The optional six-word capacity cause is not known until binding.
                6,
                s,
                ceiling_product(s, s.saturating_sub(1))? / 2,
                publication_source_validation_native_work_ceiling(contract)?,
                n,
                ceiling_product(3, p)?,
                rows,
                p,
                32 * 32,
            ])?,
            0,
            0,
            0,
            0,
            0,
            0,
            ceiling_sum([
                64,
                6 * size_of::<u64>() as u64, // Private capacity-cause snapshot.
                2 * size_of::<PublicationRange>() as u64 + 5 * 8,
                ceiling_product(rows, size_of::<SemanticActiveRow>() as u64)?,
                size_of::<PendingContinuation>() as u64 - 3 * 8,
                ceiling_product(p, 2 * 8)?,
                size_of::<PublicationRange>() as u64 + 2 * 8,
                32 * (size_of::<SourceSlot>() as u64 + 2 * 8),
            ])?,
        ],
    )?;
    // Each original nonempty D2D chunk contains at least one byte. Its
    // canonical destination backing therefore bounds both bytes and commands.
    result[0] = ceiling_sum([1, copy_capacity])?;
    result[8] = ceiling_sum([result[8], copy_capacity])?;
    native_work_ceiling_units(result)?;
    Ok(result)
}

fn publication_source_validation_native_work_ceiling(
    contract: &PublicationContract,
) -> Result<u64, SemanticTransitionError> {
    let window = contract.window_capacity;
    ceiling_sum([
        ceiling_product(2, window)?,
        ceiling_product(window, window.saturating_sub(1))? / 2,
        ceiling_product(window, contract.terminal_token_count)?,
        ceiling_product(window, ceiling_sum([window, 1])?)?,
    ])
}

/// Bound reached publication origin, receipt and data-copy work. Numerical
/// tensors contribute their real copies and geometry, never checksum scans.
/// Both banks' capacities enclose reachable success and refusal prefixes.
fn publication_finalization_native_work_ceiling(
    storage: &PublicationStorage,
) -> Result<[u64; 9], SemanticTransitionError> {
    let n = storage.bank_templates[0].len() as u64;
    let t = storage.layouts.len() as u64;
    let contract = storage.contract_value;
    let window = contract.window_capacity;
    let rows = ceiling_sum([window, contract.feedback_capacity])?;
    let mut result = [0; 9];
    add_native_work_ceiling(
        &mut result,
        continuation_validation_native_work_ceiling(storage)?,
    )?;
    add_native_work_ceiling(&mut result, text_binding_validation_native_work_ceiling()?)?;
    add_native_work_ceiling(&mut result, [0, ceiling_sum([n, 32])?, 0, 0, 0, 0, 0, 0, 0])?;
    let p = storage.continuation_templates.len() as u64;
    let pending_layouts = p.checked_sub(3).ok_or_else(native_work_ceiling_overflow)?;
    let preserves = storage.bank_templates[0]
        .iter()
        .filter(|range| {
            matches!(range.role, 32 | 44 | 56 | 57)
                || range.role == SemanticStateRole::TaskGround as u64
        })
        .count() as u64;
    // Applying the continuation repeats Update-binding validation, then walks
    // the original output/destination rosters and resolves each logical view.
    // Payload copies have their separate original backing-capacity enclosure.
    let apply_visits = ceiling_sum([
        model_update_validation_native_work_ceiling(n, storage.allocations.len() as u64)?,
        p,
        n,
        ceiling_product(3, n)?,
        ceiling_product(n, n)?,
        n,
        ceiling_product(p, ceiling_sum([1, n])?)?,
        ceiling_product(
            pending_layouts,
            ceiling_sum([p, pending_layouts, n, t, rows, 1])?,
        )?,
        ceiling_product(3, n)?,
        ceiling_product(preserves, ceiling_sum([2, n, t, 4])?)?,
    ])?;
    let apply_bytes = ceiling_sum([
        ceiling_product(n, 3 * size_of::<PublicationRange>() as u64 + 2 * 8)?,
        ceiling_product(p, 3 * 8)?,
        3 * 8,
        ceiling_product(preserves, size_of::<PublicationRange>() as u64 + 2 * 8)?,
    ])?;
    add_native_work_ceiling(
        &mut result,
        [0, apply_visits, 0, 0, 0, 0, 0, 0, apply_bytes],
    )?;
    let capacity = |role, index| -> Result<u64, SemanticTransitionError> {
        storage.bank_templates.iter().try_fold(0, |bytes, bank| {
            let range = bank
                .iter()
                .find(|range| range.role == role && range.index == index)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let entry = storage
                .allocations
                .get(
                    usize::try_from(range.storage_slot)
                        .map_err(|_| native_work_ceiling_overflow())?,
                )
                .ok_or(SemanticTransitionError::ObservationMismatch)?
                .entry();
            Ok(bytes.max(
                entry
                    .bytes
                    .checked_sub(range.offset_bytes)
                    .ok_or_else(native_work_ceiling_overflow)?,
            ))
        })
    };
    // Admission validates both directories through the original finite
    // producer geometry. Selected model origin binding can occur once before
    // admission and once in admission (or its numerical-refusal branch).
    for directory in &storage.bank_templates {
        let mut validation = step_input_producer_native_work_ceiling(storage, &[], directory)?;
        validation[0] = 0;
        add_native_work_ceiling(&mut result, validation)?;
    }
    let mut origin =
        selected_model_origin_native_work_ceiling(storage, &storage.bank_templates[0])?;
    origin[0] = 0;
    repeat_native_ceiling(&mut result, origin, 2)?;
    // Two lane probes and the final selected source preparation each validate
    // the same window; a refused probe stops at its actually reached prefix.
    let source_visits = publication_source_validation_native_work_ceiling(&contract)?;
    let prepare_visits = ceiling_sum([
        source_visits,
        ceiling_product(2, window)?,
        ceiling_product(window, contract.terminal_token_count)?,
    ])?;
    let prepare_bytes = ceiling_sum([
        size_of::<PublicationHeader>() as u64 + 8,
        ceiling_product(window, 2 * size_of::<SourceSlot>() as u64 + 8)?,
        ceiling_product(16, ceiling_product(window, contract.terminal_token_count)?)?,
    ])?;
    add_native_work_ceiling(
        &mut result,
        [
            0,
            ceiling_sum([
                source_visits,
                ceiling_product(3, prepare_visits)?,
                contract.prefix_capacity,
                window,
            ])?,
            0,
            0,
            0,
            0,
            0,
            0,
            ceiling_product(3, prepare_bytes)?,
        ],
    )?;
    let intents = capacity(30, 0)?
        .checked_sub(size_of::<IntentQueueHeader>() as u64)
        .ok_or_else(native_work_ceiling_overflow)?
        / size_of::<IntentEntry>() as u64;
    // Original intent validation hashes its effect once, each actual payload,
    // and each identity/chain. Partitioning adds at most two SHA padding blocks
    // per original entry above the full-capacity payload enclosure.
    add_native_work_ceiling(&mut result, publication_hash_ceiling(capacity(31, 0)?)?)?;
    let mut intent_payload = publication_hash_ceiling(capacity(31, 0)?)?;
    intent_payload[7] = ceiling_sum([intent_payload[7], ceiling_product(2, intents)?])?;
    intent_payload[8] = ceiling_product(32, intents)?;
    add_native_work_ceiling(&mut result, intent_payload)?;
    repeat_native_ceiling(&mut result, publication_hash_ceiling(25 * 8)?, intents)?;
    repeat_native_ceiling(&mut result, publication_hash_ceiling(80)?, intents)?;
    add_native_work_ceiling(
        &mut result,
        [
            0,
            ceiling_sum([ceiling_product(intents, 1 + 5 * 4)?, 4])?,
            0,
            0,
            0,
            0,
            0,
            0,
            ceiling_product(intents, 25 * 8 + 80)?,
        ],
    )?;
    for range in &storage.bank_templates[0] {
        let role = range.role;
        let coverage = u64::from(role == 1 || (4..=13).contains(&role));
        if coverage != 0 {
            // Coverage groups retain exact producer origins and logical ranges.
            add_native_work_ceiling(&mut result, publication_hash_ceiling(64)?)?;
            add_native_work_ceiling(&mut result, publication_hash_ceiling(80)?)?;
            add_native_work_ceiling(&mut result, [0, 0, 0, 0, 0, 0, 0, 0, 64 + 80])?;
        }
        let receipts =
            u64::from(matches!(role, 14 | 27 | 30 | 32 | 33)) + u64::from(matches!(role, 14 | 33));
        if receipts != 0 {
            // Only bounded control/provenance records enter receipt identities.
            // No tensor or full model backing contributes SHA work.
            let bytes = ceiling_sum([128, capacity(role, range.index)?])?;
            repeat_native_ceiling(&mut result, publication_hash_ceiling(bytes)?, receipts)?;
            add_native_work_ceiling(
                &mut result,
                [0, 0, 0, 0, 0, 0, 0, 0, ceiling_product(receipts, 21 * 8)?],
            )?;
            // Prediction repeats the metadata framing, never its payload hash.
            add_native_work_ceiling(
                &mut result,
                [0, 0, 0, 0, 0, 0, 0, 0, ceiling_product(receipts, 21 * 8)?],
            )?;
        }
    }
    // Directory geometry, coverage and bounded receipt searches are enclosed
    // independently of numerical payload size.
    let searches = ceiling_sum([
        ceiling_product(2, ceiling_product(n, ceiling_sum([n, t])?)?)?,
        ceiling_product(n, n.saturating_sub(1))? / 2,
        ceiling_product(ceiling_sum([window, 27])?, n)?,
        ceiling_product(3, n)?,
        ceiling_product(2, window)?,
    ])?;
    add_native_work_ceiling(
        &mut result,
        [0, ceiling_product(2, searches)?, 0, 0, 0, 0, 0, 0, 0],
    )?;
    // Each selected source creates provenance metadata with its consumed
    // authority origin. Action and attempt receipts use the same codebooks.
    let codebooks = ceiling_sum([window, 2])?;
    let codebook_prefix = b"xlog.semantic.logical-codebooks.v1\0".len() as u64 + 40;
    let mut codebook = publication_hash_ceiling(ceiling_sum([capacity(48, 0)?, codebook_prefix])?)?;
    codebook[8] = ceiling_sum([codebook[8], codebook_prefix])?;
    repeat_native_ceiling(&mut result, codebook, codebooks)?;
    repeat_native_ceiling(
        &mut result,
        publication_hash_ceiling(size_of::<SemanticTransitionReceipt>() as u64)?,
        window,
    )?;
    repeat_native_ceiling(
        &mut result,
        publication_hash_ceiling(size_of::<TokenProvenance>() as u64)?,
        window,
    )?;
    for (bytes, count) in [
        (
            COMPONENT_COUNT as u64 * size_of::<SemanticTransitionReceipt>() as u64,
            1,
        ),
        (
            7 * size_of::<crate::semantic_hypergraph::SemanticResidentReceiptRecord>() as u64,
            2,
        ),
        (size_of::<SemanticActionBatchReceipt>() as u64, 2),
        (80, 4),
        (9 * size_of::<u64>() as u64, 1),
        (25 * size_of::<u64>() as u64, 1),
        (contract.model_contract_layout.schema_bytes, 1),
        (
            b"xlog.semantic.model-identity.v2\0".len() as u64 + 32 + 8 + 32,
            1,
        ),
        (capacity(31, 0)?, 2),
        (220, 1), // Tagged Proposal origin; nondraw records are eight bytes shorter.
        (64, ceiling_sum([window, 1])?), // Authority and consumed lineage origins.
    ] {
        repeat_native_ceiling(&mut result, publication_hash_ceiling(bytes)?, count)?;
    }
    let fixed = ceiling_sum([
        size_of::<DeviceState>() as u64,
        136 * size_of::<SemanticTransitionReceipt>() as u64,
        3 * size_of::<SemanticActionBatchReceipt>() as u64,
        2 * size_of::<ExecutionWork>() as u64,
        3 * size_of::<PublicationHeader>() as u64,
        2 * size_of::<CompletionCoverage>() as u64,
        size_of::<AttemptReceipt>() as u64,
        size_of::<IntentEntry>() as u64,
        12 * size_of::<PublicationRange>() as u64,
        ceiling_product(window, 2 * size_of::<TokenProvenance>() as u64 + 24 * 8)?,
        // Fixed identity copies and scalar writes in action/coverage/attempt,
        // model-contract, intent, equality and publication closure.
        (16 + 4 + 13 + 9 + 25 + 19 + 8 + 5 + 4 + 4 * 5 + 10 + 11) * 8,
        4 * 80,
        404, // Origin construction and copies, excluding its SHA output above.
        ceiling_product(64, ceiling_sum([window, 1])?)?,
        capacity(30, 0)?,
        capacity(15, 0)?,
        capacity(31, 0)?,
    ])?;
    // Execute validates the optional capacity cause once, including its refusal
    // prefix before publication begins. Cold geometry reserves the present case.
    add_native_work_ceiling(&mut result, [0, 11 + 5 + 4 + 6, 0, 0, 0, 0, 0, 0, fixed])?;
    native_work_ceiling_units(result)?;
    Ok(result)
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
        // Continuation copies reach their own byte-writing boundaries. Bound
        // each original output by its fixed logical tensor capacity, including
        // packed active rows; cache preservation and the layout table use the
        // largest retained native backing in the two publication banks.
        let continuation = child.steps[&handles[index].token]
            .prepared
            .as_ref()
            .and_then(|prepared| prepared.branches[index].continuation.as_ref())
            .ok_or(SemanticTransitionError::NotBound)?;
        add_native_work_ceiling(
            ceiling,
            publication_finalization_native_work_ceiling(storage)?,
        )?;
        for source in &continuation.sources {
            let bytes = source.layout.dimensions[..source.layout.rank as usize]
                .iter()
                .try_fold(source.layout.element_bytes, |bytes, dimension| {
                    ceiling_product(bytes, *dimension)
                })?;
            add_native_work_ceiling(ceiling, [0, 0, 0, 0, 0, 0, 0, 0, bytes])?;
        }
        for original in storage.bank_templates[0]
            .iter()
            .filter(|range| matches!(range.role, 4 | 5 | 55))
        {
            let mut bytes = 0;
            for bank in &storage.bank_templates {
                let range = bank
                    .iter()
                    .find(|range| range.role == original.role && range.index == original.index)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let slot = usize::try_from(range.storage_slot)
                    .map_err(|_| native_work_ceiling_overflow())?;
                bytes = bytes.max(
                    storage
                        .allocations
                        .get(slot)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?
                        .entry()
                        .bytes,
                );
            }
            if original.role == 55 {
                bytes = ceiling_sum([bytes, 2 * size_of::<u64>() as u64])?;
            }
            add_native_work_ceiling(ceiling, [0, 0, 0, 0, 0, 0, 0, 0, bytes])?;
        }
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
            add_native_work_ceiling(
                ceiling,
                io.replay_custody
                    .as_ref()
                    .expect("checked original replay custody")
                    .successor_metadata_native_work_ceiling()?,
            )?;
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
        // The immutable support bank is proved binary before publication can
        // read support or retire a root; its later numerical scan is separate.
        add_native_work_ceiling(ceiling, [0, 0, 0, cells, 0, 0, 0, 0, 0])?;
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
        let binding = policy
            .buffers
            .binding_work
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let transferred = binding
            .as_ref()
            .is_some_and(|binding| binding.work.is_some());
        if !transferred {
            add_native_work_ceiling(
                ceiling,
                policy_uniform_native_work_ceiling(components, parameter_cells)?,
            )?;
            let bytes = binding.as_ref().map_or(0, |binding| binding.snapshot_bytes);
            add_native_work_ceiling(
                ceiling,
                [if bytes == 0 { 0 } else { 9 }, 0, 0, 0, 0, 0, 0, 0, bytes],
            )?;
        }
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
        let categories = ceiling_sum([alias, primal, viability, ceiling_product(4, cells)?])?;
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

pub(super) fn publication_record_identity_native_work_ceiling(
    record_bytes: u64,
) -> Result<[u64; 9], SemanticTransitionError> {
    Ok([
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        ceiling_sum([128, record_bytes, 72])? / 64,
        21 * 8 + 32,
    ])
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
    )
}

pub(super) fn step_input_producer_geometry_native_work_ceiling(
    contract: &PublicationContract,
    layout_count: usize,
    allocation_count: usize,
    role_count: usize,
    plans: &[StepInputPlan],
    directory: &[PublicationRange],
) -> Result<[u64; 9], SemanticTransitionError> {
    let mut ceiling = selected_model_origin_geometry_native_work_ceiling(contract, directory)?;
    add_native_work_ceiling(&mut ceiling, [1, 0, 0, 0, 0, 0, 0, 0, 0])?;
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
    visits = sum(visits, 22 + 44 + 6)?;
    visits = sum(visits, product(6, sum(s, 2)?)?)?;
    let words = (size_of::<SemanticTensorLayout>() / 8) as u64;
    visits = sum(
        visits,
        product(k, sum(sum(sum(1, product(2, n)?)?, t)?, words)?)?,
    )?;
    visits = sum(visits, product(pairs(k)?, sum(n, 1)?)?)?;
    visits = sum(visits, product(k, sum(product(2, sum(s, 2)?)?, 4)?)?)?;
    visits = sum(visits, product(k, sum(n, 1)?)?)?;
    add_native_work_ceiling(&mut ceiling, [0, visits, 0, 0, 0, 0, 0, 0, 0])?;
    let bytes = sum(
        size_of::<PublicationHeader>() as u64,
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
