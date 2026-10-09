use super::*;
use crate::device::ExecutionCompletion;

/// One command of an original cold producer. Driver entry and successful
/// submission are distinct: a late commit/fence error never authorizes replay.
pub(crate) struct OriginalNativeCommand {
    completion: ExecutionCompletion,
    completion_entered: bool,
    entered: bool,
    submitted: bool,
    completed: bool,
    dma_recorded: bool,
}

impl OriginalNativeCommand {
    pub(crate) fn new(domain: &ResidentExecutionDomain) -> Result<Self, SemanticTransitionError> {
        Ok(Self {
            completion: ExecutionCompletion::new(domain.execution_stream().context())
                .map_err(|error| runtime_error("original content completion allocation", error))?,
            completion_entered: false,
            entered: false,
            submitted: false,
            completed: false,
            dma_recorded: false,
        })
    }

    pub(crate) fn run(
        &mut self,
        domain: &ResidentExecutionDomain,
        poisoned: &mut bool,
        recorder: LaunchRecorder,
        operation: impl FnOnce(&CudaEnqueue<'_>, &mut bool, &mut bool) -> Result<(), XlogError>,
    ) -> Result<(), SemanticTransitionError> {
        if self.completed {
            return Ok(());
        }
        if !self.entered {
            if self.completion_entered {
                self.completion.wait().map_err(|error| {
                    *poisoned = true;
                    runtime_error("original content prelaunch completion", error)
                })?;
                self.completion = ExecutionCompletion::new(domain.execution_stream().context())
                    .map_err(|error| {
                        runtime_error("original content completion allocation", error)
                    })?;
                self.completion_entered = false;
            }
            let mut original_result = None;
            let completion = self.completion.submit(domain.execution_stream(), None, || {
                self.completion_entered = true;
                // SAFETY: the original caller records the exact retained
                // allocations and the operation uses only this bound stream.
                let result = unsafe {
                    domain.enqueue(recorder, |enqueue| {
                        operation(enqueue, &mut self.entered, &mut self.submitted)
                    })
                }
                .map_err(|error| runtime_error("original content enqueue", error))
                .and_then(|enqueued| {
                    enqueued
                        .commit()
                        .map_err(|error| runtime_error("original content commit", error))
                });
                let successful = result.is_ok();
                original_result = Some(result);
                if successful {
                    Ok(())
                } else {
                    Err(cudarc::driver::DriverError(
                        sys::CUresult::CUDA_ERROR_UNKNOWN,
                    ))
                }
            });
            let submitted = original_result.ok_or_else(|| {
                runtime_error(
                    "original content submission",
                    "original command did not enter",
                )
            })?;
            if let Err(error) = submitted {
                *poisoned |= self.entered;
                return Err(error);
            }
            completion.map_err(|error| {
                *poisoned |= self.entered;
                runtime_error("original content completion fence", error)
            })?;
        }
        self.completion.wait().map_err(|error| {
            *poisoned = true;
            runtime_error("original content completion", error)
        })?;
        if !self.submitted {
            *poisoned |= self.entered;
            return Err(runtime_error(
                "original content completion",
                "original driver command has no successful submission",
            ));
        }
        self.completed = true;
        Ok(())
    }

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
}

pub(super) struct OriginalContentBatch {
    pub(super) verify: bool,
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

    fn record_submitted_dma(&mut self, bytes: usize) -> Result<(), SemanticTransitionError> {
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
    pub(super) fn new(
        invocation: Arc<()>,
        cohort: Arc<SemanticEvaluationCohort>,
        output: Vec<PreparedSemanticTensor>,
        objective: Vec<PreparedSemanticTensor>,
        model_ceiling: [u64; 9],
        preparation_ceiling: [u64; 9],
        model_snapshot: [u64; 9],
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
        let count = u64::try_from(self.plans.len()).map_err(|_| native_work_ceiling_overflow())?;
        let previous = count
            .checked_sub(1)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        ceiling[1] = count
            .checked_mul(previous)
            .map(|visits| visits / 2)
            .and_then(|visits| visits.checked_add(count))
            .ok_or_else(native_work_ceiling_overflow)?;
        for plan in &self.plans {
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
            for plan in self
                .plans
                .iter()
                .filter(|plan| matches!(plan.role, 18..=25))
            {
                let slot = plan.banks[bank].storage_slot;
                if !seen.insert(slot) {
                    continue;
                }
                let allocation = self
                    .storage
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
