use super::*;

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
                    // This original publication guard receives no cold tally;
                    // it does not contribute events to this native report.
                    CapturedTensorDigest::Publication(_) => continue,
                    CapturedTensorDigest::Tensor {
                        producer_sealed, ..
                    } => verify || *producer_sealed,
                },
                TensorContentSeals::Model(_) => true,
            };
            let physical_bytes = tensor.source.as_ref().map_or(0, |source| source.len());
            tensor_content_range(
                &tensor.layout,
                tensor.logical_begin,
                tensor.logical_end,
                physical_bytes,
            )?;
            let dimensions = &tensor.layout.dimensions[..tensor.layout.rank as usize];
            let logical_bytes = if dimensions.contains(&0) {
                0
            } else {
                dimensions
                    .iter()
                    .try_fold(tensor.layout.element_bytes, |bytes, dimension| {
                        bytes
                            .checked_mul(*dimension)
                            .ok_or_else(native_work_ceiling_overflow)
                    })?
            };
            // A tensor adds its four layout fields and every logical dimension
            // to the same original range prefix used by publication seals.
            add_native_work_ceiling(
                &mut ceiling,
                content_digest_ceiling(logical_bytes, 25 + tensor.layout.rank, !verify)?,
            )?;
        }
        Ok(ceiling)
    }
}
