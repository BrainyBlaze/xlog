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
        let result = self.run(domain, poisoned, recorder, operation);
        // A successful driver submission remains expenditure when the original
        // commit or completion fence subsequently fails. A preflight refusal
        // has submitted=false and cannot manufacture a reached copy.
        if self.submitted && !self.dma_recorded {
            if let Some(allowance) = allowance {
                allowance
                    .lock()
                    .map_err(|_| publication_input_error("original cold allowance is poisoned"))?
                    .record_submitted_dma(bytes)?;
            }
            self.dma_recorded = true;
        }
        result
    }

    pub(super) fn claim_model_contract_guard(
        &mut self,
        allowance: &Arc<Mutex<ColdNativeAllowance>>,
        ceiling: [u64; 9],
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
            .claim_model_contract_guard(ceiling)?
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
    pub(super) commands: Vec<OriginalNativeCommand>,
    // The original report and child graph retain the strong owner. Keeping
    // only its issued identity here avoids a cohort -> batch -> cohort cycle.
    pub(super) allowance: Option<std::sync::Weak<Mutex<ColdNativeAllowance>>>,
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
            commands,
            allowance: None,
        })
    }
}

/// Original finite content occurrences. Other canonical native producers have
/// separate schedules; this subtotal must never stand in for their ceiling.
pub(crate) struct ColdNativeAllowance {
    purpose: SemanticColdModelWorkPurpose,
    content: Option<FrozenColdEvaluationContent>,
    submitted_dma: [u64; 9],
    submitted_dma_ceiling: [u64; 9],
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
            submitted_dma: [0; 9],
            submitted_dma_ceiling: [0; 9],
        }
    }

    pub(super) fn is_evaluation(&self) -> bool {
        matches!(
            self.purpose,
            SemanticColdModelWorkPurpose::SourceEvaluation
                | SemanticColdModelWorkPurpose::PrivateEvaluation
        )
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
        match &self.content {
            Some(content) => Ok(content.ceiling),
            None if !self.is_evaluation() => Ok([0; 9]),
            None => Err(publication_input_error(
                "evaluation content must be frozen before its original native producers",
            )),
        }
    }

    pub(super) fn claim_content(
        &mut self,
        tensors: &TensorContentBuffers,
        verify: bool,
    ) -> Result<(), SemanticTransitionError> {
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
    ) -> Result<bool, SemanticTransitionError> {
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
    ) -> Result<bool, SemanticTransitionError> {
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
    ) -> Result<bool, SemanticTransitionError> {
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

    pub(super) fn record_submitted_dma(
        &mut self,
        bytes: usize,
    ) -> Result<(), SemanticTransitionError> {
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
                "submitted native DMA exceeded its original finite entitlement",
            ));
        }
        self.submitted_dma = reached;
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
        .checked_add(blocks(
            storage.contract_value.model_contract_layout.schema_bytes,
        )?)
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
        sum.checked_add(value).ok_or_else(native_work_ceiling_overflow)
    })
}

fn ceiling_product(a: u64, b: u64) -> Result<u64, SemanticTransitionError> {
    a.checked_mul(b).ok_or_else(native_work_ceiling_overflow)
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

/// Enclose the reached event counters of the original embedded transitions.
/// These are capacity bounds, not observed work or structural selection cost.
/// The initializer has its own separate finite occurrence and is not included.
pub(super) fn actor_refresh_transition_native_work_ceilings(
    child: &SemanticTransitionSession,
) -> Result<[[u64; 9]; 2], SemanticTransitionError> {
    let build = child.prepared_segment.as_ref().ok_or(SemanticTransitionError::NotCaptured)?;
    let handles = build.handles()?;
    if build.transitions != [SemanticTransitionKind::Recompute, SemanticTransitionKind::Proposal]
        || handles.len() != 2
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let storage = child.publication.as_ref().ok_or(SemanticTransitionError::NotBound)?;
    let task = child.task.as_ref().ok_or(SemanticTransitionError::NotBound)?;
    let queries = u64::try_from(task.0.spec.statement_records.len())
        .map_err(|_| native_work_ceiling_overflow())?;
    let editable = task.0.spec.program.editable_program().is_some();
    let components = &child.codebooks.components;
    if components.len() != COMPONENT_COUNT {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let targets = *child.codebooks.words.get(1).ok_or(SemanticTransitionError::ObservationMismatch)?;
    let operands = *child.codebooks.words.get(3).ok_or(SemanticTransitionError::ObservationMismatch)?;
    let qualifier = components.get(32 + 11).ok_or(SemanticTransitionError::ObservationMismatch)?.cardinality as u64;
    let leaf = components.get(32 + 17).ok_or(SemanticTransitionError::ObservationMismatch)?.cardinality as u64;
    // An acquisition refusal can reach viability even for Recompute before
    // its successful non-Proposal early return. Enclose both actual branches.
    let viability = ceiling_product(4, ceiling_sum([
        ceiling_product(targets, ceiling_sum([1, ceiling_product(8, operands)?])?)?,
        18, ceiling_sum([20, qualifier, leaf, targets])?, 2,
    ])?)?;
    let arena = child.graph.transition_arena();
    let roots = arena[2];
    let statements = arena[3];
    let supports = arena[4];
    let versions = arena[5];
    let records = ceiling_sum([statements, supports, versions])?;
    let receipt_bytes = std::mem::size_of_val(&task_ground::TaskQueryRecord::default().receipt) as u64;
    // execute clears the output, can copy one source receipt, and emits its
    // bounded command/handle/refusal words. Three receipts plus a whole command
    // also cover every fixed fill_view/fill_statement/fill_version field.
    let fixed_bytes = ceiling_sum([ceiling_product(3, receipt_bytes)?, 32 * 8])?;
    let snapshot = [1, 2, 0, 0, 0, 0, 0, 0, fixed_bytes];
    let truth = [1, ceiling_sum([ceiling_product(4, statements)?, 8])?, 0, 0, 0, 0, 0, 0, fixed_bytes];
    // retire_root validates and then marks each original root/candidate chain.
    // Each chain link visits version, statement, support, version and support;
    // the three complete record scans follow those two traces.
    let views = ceiling_sum([roots, 1])?;
    let chains = ceiling_product(ceiling_product(views, statements)?, versions)?;
    let trace_tables = ceiling_sum([
        ceiling_product(2, roots)?, 2,
        ceiling_product(views, ceiling_sum([
            ceiling_product(2, statements)?,
            ceiling_product(5, ceiling_product(statements, versions)?)?,
        ])?)?,
    ])?;
    let retirement = [
        1,
        ceiling_sum([ceiling_product(2, trace_tables)?, ceiling_product(3, records)?, 3])?,
        ceiling_product(2, chains)?, 0, 0, 0, 0, 0,
        ceiling_sum([
            fixed_bytes,
            ceiling_product(16, ceiling_product(ceiling_product(views, statements)?, ceiling_sum([versions, 1])?)?)?,
            ceiling_product(24, records)?, ceiling_product(8, statements)?,
        ])?,
    ];
    let mut result = [[0; 9]; 2];
    for index in 0..2 {
        let io = child.prepared_kernel_io(&handles[index], index)?;
        let descriptor = child.descriptor_with(&io);
        let ceiling = &mut result[index];
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
        add_native_work_ceiling(ceiling, [0, storage.bank_templates[0].len() as u64, 0, 0, 0, 0, 0, 0,
            ceiling_product(2, ground_bytes)?])?;
        // apply_continuation preserves these exact original non-tensor roles,
        // including their unused backing. Both publication banks are possible;
        // select the larger retained physical extent per canonical role/index.
        for original in storage.bank_templates[0].iter().filter(|range| {
            matches!(range.role, 32 | 44 | 56 | 57) || range.role == SemanticStateRole::TaskGround as u64
        }) {
            let mut bytes = 0;
            for bank in &storage.bank_templates {
                let range = bank.iter().find(|range| range.role == original.role && range.index == original.index)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let slot = usize::try_from(range.storage_slot).map_err(|_| native_work_ceiling_overflow())?;
                let extent = storage.allocations.get(slot).ok_or(SemanticTransitionError::ObservationMismatch)?
                    .entry().bytes.checked_sub(range.offset_bytes).ok_or_else(native_work_ceiling_overflow)?;
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
            let bytes = PreparedReplayCustody::allocation_bytes(storage, usize::try_from(arena[6])
                .map_err(|_| native_work_ceiling_overflow())?)?;
            add_native_work_ceiling(ceiling, [0, 0, 0, 0, 0, 0, 0, 0, bytes])?;
        }
        if index == 0 {
            if io.policy.is_some() {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            add_native_work_ceiling(ceiling, [0, 0, 0, viability, 0, 0, 0, 0, 0])?;
            native_work_ceiling_units(*ceiling)?;
            continue;
        }
        let cells = ceiling_sum(components.iter().map(|component| u64::from(component.cardinality)))?;
        let edit_cells = ceiling_sum(components.iter().filter(|component| !component.is_text())
            .map(|component| u64::from(component.cardinality)))?;
        let text_cells = ceiling_sum(components.iter().filter(|component| component.is_text())
            .map(|component| u64::from(component.cardinality)))?;
        let policy = io.policy.ok_or(SemanticTransitionError::NotBound)?;
        let parameter_cells = policy.buffers.parameters.len() as u64;
        // The same uniform-domain producer contributes its retained Category
        // tally: parameter initialization/export, recurrent initialization,
        // support scan, 128 alternatives in each forward and reverse edit,
        // two-lane text alternatives and public text-domain export. The hot
        // alias check and primal/critic admission then visit their own cells.
        let uniform = ceiling_sum([
            ceiling_product(2, parameter_cells)?, 2 * 37 * 128,
            ceiling_product(3, COMPONENT_COUNT as u64)?, cells,
            ceiling_product(257, edit_cells)?, ceiling_product(128, TEXT_CARDINALITY as u64)?,
        ])?;
        let alias = ceiling_sum([parameter_cells, 32 * TEXT_CARDINALITY as u64, COMPONENT_COUNT as u64])?;
        let primal = ceiling_sum([cells, text_cells, parameter_cells, 32 * TEXT_CARDINALITY as u64,
            ceiling_product(2, COMPONENT_COUNT as u64)?])?;
        // Four edit viability banks: target_completes visits at most four full
        // operand banks with supported+operand_matches; all three opcodes are
        // checked once. Each category has two legal-category visits and each
        // visit can perform one additional operand_matches.
        let categories = ceiling_sum([uniform, alias, primal, viability, ceiling_product(4, cells)?])?;
        let mut comparisons = 0;
        for component in components {
            let width = u64::from(component.cardinality).checked_next_power_of_two()
                .ok_or_else(native_work_ceiling_overflow)?;
            let levels = u64::from(width.trailing_zeros());
            let stages = ceiling_product(levels, ceiling_sum([levels, 1])?)? / 2;
            comparisons = ceiling_sum([comparisons, ceiling_product(width / 2, stages)?])?;
        }
        add_native_work_ceiling(ceiling, [0, 0, 0, categories, comparisons,
            ceiling_product(2, cells)?, ceiling_product(10, COMPONENT_COUNT as u64)?, 0,
            ceiling_product(COMPONENT_COUNT as u64, size_of::<SemanticTransitionReceipt>() as u64)?])?;

        // One preflight, two forks, four insertions, two terminals and one
        // live-candidate cleanup. Each seal also reserves its internal discard.
        let preflight = [1, ceiling_sum([3, roots, records, ceiling_product(4, statements)?])?,
            0, 0, 0, 0, 0, 0, fixed_bytes];
        let fork = [1, 2, 0, 0, 0, 0, 0, 0, ceiling_sum([fixed_bytes, ceiling_product(8, statements)?])?];
        let insertion = [1, ceiling_sum([20, ceiling_product(5, statements)?, supports, ceiling_product(10, versions)?])?,
            versions, 0, 0, 0, 0, ceiling_sum([8, ceiling_product(2, versions)?])?,
            ceiling_sum([fixed_bytes, 2048, ceiling_product(288, versions)?])?];
        let terminal = [2, ceiling_sum([12, roots, ceiling_product(3, records)?, ceiling_product(2, statements)?])?,
            0, 0, 0, 0, 0, 0, ceiling_sum([ceiling_product(2, fixed_bytes)?,
                ceiling_product(32, records)?, ceiling_product(24, statements)?])?];
        add_native_work_ceiling(ceiling, preflight)?;
        repeat_native_ceiling(ceiling, fork, 2)?;
        repeat_native_ceiling(ceiling, insertion, 4)?;
        repeat_native_ceiling(ceiling, terminal, 3)?;
        if editable {
            // materialize checks charged==fuel before every increment, including
            // failure. Exactly three task_program_queries can reach the owner.
            add_native_work_ceiling(ceiling, [0, 3 * 1_000_000, 0, 0, 0, 0, 0, 0,
                ceiling_product(ceiling_product(3, queries)?, receipt_bytes)?])?;
            let rules = crate::semantic_program::RESIDENT_PROGRAM_RULE_CAPACITY as u64;
            add_native_work_ceiling(ceiling, [0, ceiling_product(2, ceiling_sum([rules, 2])?)?,
                0, 0, 0, 0, 0, 0, 2 * 40 + 8])?;
        } else {
            // Baseline, two candidates and final feedback use the same truth owner.
            repeat_native_ceiling(ceiling, truth, ceiling_product(4, queries)?)?;
        }
        // Original query rows are zeroed once before sampling.
        add_native_work_ceiling(ceiling, [0, 0, 0, 0, 0, 0, 0, 0,
            ceiling_product(ceiling_product(3, queries)?, size_of::<task_ground::TaskQueryRecord>() as u64)?])?;
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
    fn native_work_ceiling(&self) -> Result<[u64; 9], SemanticTransitionError> {
        Self::basis_native_work_ceiling(&self.storage, &self.plans)
    }

    fn basis_native_work_ceiling(
        storage: &PublicationStorage,
        plans: &[StepInputPlan],
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
                tensor_layout_native_work_ceiling(
                    &plan.layout,
                    0,
                    end,
                    plan.banks[0].span.len(),
                    true,
                )?
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
                let allocation = storage
                    .allocations
                    .get(slot)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let blocks = allocation
                    .entry()
                    .bytes
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
    let mut ceiling = PreparedStepInputs::basis_native_work_ceiling(storage, plans)?;
    // The selected-model check is a strict prefix of the full role44 guard.
    // Retaining that complete guard ceiling is a conservative native bound.
    add_native_work_ceiling(
        &mut ceiling,
        model_contract_guard_native_work_ceiling(storage, directory)?,
    )?;
    let table = directory
        .iter()
        .find(|range| range.role == 55 && range.index == 0)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let mut table_digest = content_digest_ceiling(table.length_bytes, 21, false)?;
    table_digest[0] = 0;
    add_native_work_ceiling(&mut ceiling, table_digest)?;
    let n = u64::try_from(directory.len()).map_err(|_| native_work_ceiling_overflow())?;
    let t = u64::try_from(storage.layouts.len()).map_err(|_| native_work_ceiling_overflow())?;
    let k = u64::try_from(plans.len()).map_err(|_| native_work_ceiling_overflow())?;
    let s = u64::try_from(storage.allocations.len()).map_err(|_| native_work_ceiling_overflow())?;
    let d = storage
        .contract_value
        .window_capacity
        .checked_add(storage.contract_value.feedback_capacity)
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
        u64::try_from(storage.role_counts.len()).map_err(|_| native_work_ceiling_overflow())?,
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
