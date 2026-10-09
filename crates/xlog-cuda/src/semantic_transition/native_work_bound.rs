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
}

pub(super) struct OriginalContentBatch {
    pub(super) verify: bool,
    pub(super) cursor: usize,
    pub(super) commands: Vec<OriginalNativeCommand>,
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

fn content_digest_ceiling(
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
