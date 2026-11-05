//! Original full model-forward and paired-canary occurrences with bounded logits scratch.
//!
//! Cold preparation allocates only native cursors, row seals and five canary
//! accumulators. Capture records a use immediately before the producer overwrites
//! its scratch. Device nodes repeat every cursor/coverage check on every replay.

use super::*;

const MODEL_STREAM_LEAF_BYTES: u64 = 4096;

/// Independent original counters, merged exactly once by the existing Update
/// canary gate. These are checked execution ceilings, never observed work.
pub(super) struct ModelStreamNativeWorkCeilings {
    pub(super) forwards: [[u64; 9]; 2],
    pub(super) canary: [u64; 9],
    pub(super) comparison: [u64; 9],
}

impl ModelStreamNativeWorkCeilings {
    pub(super) fn combined(&self) -> Result<[u64; 9], SemanticTransitionError> {
        let mut combined = self.comparison;
        for ceiling in self.forwards.into_iter().chain([self.canary]) {
            native_work_bound::add_native_work_ceiling(&mut combined, ceiling)?;
        }
        native_work_bound::native_work_ceiling_units(combined)?;
        Ok(combined)
    }
}

fn stream_work_product(left: u64, right: u64) -> Result<u64, SemanticTransitionError> {
    left.checked_mul(right)
        .ok_or(SemanticTransitionError::GenerationExhausted)
}

fn stream_work_sum(values: &[u64]) -> Result<u64, SemanticTransitionError> {
    values.iter().try_fold(0u64, |sum, value| {
        sum.checked_add(*value)
            .ok_or(SemanticTransitionError::GenerationExhausted)
    })
}

fn stream_work_repeat(
    total: &mut [u64; 9],
    occurrence: [u64; 9],
    count: u64,
) -> Result<(), SemanticTransitionError> {
    native_work_bound::repeat_native_ceiling(total, occurrence, count)
}

fn stream_work_hash(
    total: &mut [u64; 9],
    bytes: u64,
    count: u64,
) -> Result<(), SemanticTransitionError> {
    stream_work_repeat(
        total,
        native_work_bound::publication_hash_ceiling(bytes)?,
        count,
    )
}

// The event positions and unit tariffs remain owned by native_work_bound. This
// helper only supplies this producer's reached command/slot/category/byte counts.
fn stream_work_events(
    total: &mut [u64; 9],
    commands: u64,
    slots: u64,
    categories: u64,
    bytes: u64,
    occurrences: u64,
) -> Result<(), SemanticTransitionError> {
    stream_work_repeat(
        total,
        [commands, slots, 0, categories, 0, 0, 0, 0, bytes],
        occurrences,
    )
}

fn stream_leaf_work_ceiling(
    geometry: StreamGeometry,
    count: u64,
) -> Result<([u64; 9], u64), SemanticTransitionError> {
    let bytes = stream_work_product(
        stream_work_product(count, geometry.vocabulary)?,
        geometry.element_bytes(),
    )?;
    let full_leaves = bytes / MODEL_STREAM_LEAF_BYTES;
    let tail_bytes = bytes % MODEL_STREAM_LEAF_BYTES;
    let leaves = stream_work_sum(&[full_leaves, u64::from(tail_bytes != 0)])?;
    let mut ceiling = [0; 9];
    for (length, repeats) in [
        (MODEL_STREAM_LEAF_BYTES, full_leaves),
        (tail_bytes, u64::from(tail_bytes != 0)),
    ] {
        if repeats == 0 {
            continue;
        }
        // Finite scan reads length bytes, then the authenticated leaf hashes
        // its exact 64-byte prefix and those same original bytes. The local
        // leaf receipt is written once. SHA padding/output uses one canonical
        // implementation, including a short final leaf.
        stream_work_hash(&mut ceiling, stream_work_sum(&[64, length])?, repeats)?;
        stream_work_events(
            &mut ceiling,
            0,
            0,
            length / geometry.element_bytes(),
            stream_work_sum(&[
                stream_work_product(2, length)?,
                64,
                size_of::<ModelStreamLeaf>() as u64,
            ])?,
            repeats,
        )?;
    }
    Ok((ceiling, leaves))
}

fn stream_block_work_ceiling(
    geometry: StreamGeometry,
    count: u64,
    paired: bool,
) -> Result<[u64; 9], SemanticTransitionError> {
    let (leaf, leaves) = stream_leaf_work_ceiling(geometry, count)?;
    let roles = if paired { 2 } else { 1 };
    let mut ceiling = [0; 9];
    stream_work_repeat(&mut ceiling, leaf, roles)?;
    let digest_bytes = stream_work_sum(&[64, stream_work_product(32, leaves)?])?;
    stream_work_hash(&mut ceiling, digest_bytes, roles)?;
    // Each leaf merge visits nine source counters to check their sum and nine
    // more to merge them. The caller then reaches one leaf-table slot. Position
    // merging uses the same rule. All counts are actual source-loop bounds.
    let leaf_slots = stream_work_product(stream_work_product(19, leaves)?, roles)?;
    let position_slots = if paired {
        stream_work_product(19, count)?
    } else {
        0
    };
    let writes = if paired {
        size_of::<ModelCanaryStreamRow>() as u64 + 16
    } else {
        4 * 8
    };
    let position_bytes = if paired {
        stream_work_product(count, size_of::<ModelCanaryStreamPosition>() as u64)?
    } else {
        0
    };
    stream_work_events(
        &mut ceiling,
        if paired { 3 } else { 2 },
        stream_work_sum(&[leaf_slots, position_slots])?,
        0,
        stream_work_sum(&[
            stream_work_product(digest_bytes, roles)?,
            writes,
            position_bytes,
        ])?,
        1,
    )?;
    Ok(ceiling)
}

fn stream_all_blocks_work_ceiling(
    geometry: StreamGeometry,
    rows: u64,
    paired: bool,
) -> Result<[u64; 9], SemanticTransitionError> {
    let mut ceiling = [0; 9];
    let full_blocks = geometry.capacity / geometry.block_positions;
    let tail = geometry.capacity % geometry.block_positions;
    if full_blocks != 0 {
        stream_work_repeat(
            &mut ceiling,
            stream_block_work_ceiling(geometry, geometry.block_positions, paired)?,
            stream_work_product(rows, full_blocks)?,
        )?;
    }
    if tail != 0 {
        stream_work_repeat(
            &mut ceiling,
            stream_block_work_ceiling(geometry, tail, paired)?,
            rows,
        )?;
    }
    Ok(ceiling)
}

fn full_forward_native_work_ceiling(
    storage: &PublicationStorage,
    geometry: StreamGeometry,
    role: usize,
) -> Result<[u64; 9], SemanticTransitionError> {
    let mut ceiling = stream_all_blocks_work_ceiling(geometry, geometry.rows, false)?;
    // Begin and finish each acquire the original publication (two identity
    // comparisons each). Begin zeroes its cursor and row seals on EVERY replay.
    stream_work_events(
        &mut ceiling,
        2,
        16,
        0,
        stream_work_sum(&[
            size_of::<ModelForwardStreamState>() as u64,
            32,
            stream_work_product(32, geometry.rows)?,
            size_of::<ModelForwardReceipt>() as u64,
            32,
            32,
            size_of::<ModelForwardReceipt>() as u64,
            8,
        ])?,
        1,
    )?;
    let mut guard = native_work_bound::selected_model_origin_native_work_ceiling(
        storage,
        &storage.bank_templates[0],
    )?;
    guard[0] = 0;
    native_work_bound::add_native_work_ceiling(&mut ceiling, guard)?;
    // Two initial seeds, then one seed before each remaining row.
    stream_work_hash(&mut ceiling, 64, stream_work_sum(&[geometry.rows, 1])?)?;
    // Every complete row writes its seal and enters the original ordered digest.
    stream_work_hash(&mut ceiling, 80, geometry.rows)?;
    stream_work_events(&mut ceiling, 0, 0, 0, 32 + 80, geometry.rows)?;
    stream_work_hash(
        &mut ceiling,
        size_of::<ModelForwardReceipt>() as u64 - 32,
        1,
    )?;
    if role != 1 {
        stream_work_events(&mut ceiling, 0, 0, 0, 32, 1)?;
    } else {
        let ranges = &storage.bank_templates[0];
        let seals = u64::try_from(
            ranges
                .iter()
                .filter(|range| matches!(range.role, 18..=25))
                .count(),
        )
        .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let folds = stream_work_sum(&[1, stream_work_product(2, seals)?])?;
        stream_work_hash(
            &mut ceiling,
            b"xlog.semantic.model-numerics.v1\0".len() as u64,
            1,
        )?;
        stream_work_hash(&mut ceiling, 80, folds)?;
        stream_work_events(
            &mut ceiling,
            0,
            u64::try_from(ranges.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            0,
            stream_work_product(80, folds)?,
            1,
        )?;
    }
    native_work_bound::native_work_ceiling_units(ceiling)?;
    Ok(ceiling)
}

pub(super) fn evaluation_forward_native_work_ceiling(
    storage: &PublicationStorage,
    geometry: StreamGeometry,
) -> Result<[u64; 9], SemanticTransitionError> {
    full_forward_native_work_ceiling(storage, geometry, 2)
}

/// Authenticate the same completed forward and the three original output seals.
/// These bounds follow the fixed identity/merge loops in the shared GPU owner;
/// hashing uses the canonical padding and tariff helpers above.
pub(super) fn evaluation_forward_binding_native_work_ceiling(
) -> Result<[u64; 9], SemanticTransitionError> {
    Ok([1, 0, 0, 0, 0, 0, 0, 0, 9])
}

fn paired_canary_native_work_ceiling(
    geometry: StreamGeometry,
    task_queries: u64,
    protected_members: u64,
) -> Result<[u64; 9], SemanticTransitionError> {
    let mut ceiling = stream_all_blocks_work_ceiling(geometry, 5, true)?;
    let capacity = geometry.capacity;
    let vocabulary = geometry.vocabulary;
    // The two obligation canaries may share positions. Their native per-position
    // argmax is computed once; the complete query roster still supplies every
    // original obligation, including repeated positions. No labels control this bound.
    let obligation_positions = capacity.min(task_queries);
    let symbolic_scans = stream_work_sum(&[
        stream_work_product(stream_work_product(2, capacity)?, task_queries)?,
        stream_work_product(
            stream_work_product(2, obligation_positions)?,
            stream_work_sum(&[stream_work_product(2, vocabulary)?, 3, task_queries])?,
        )?,
    ])?;
    let retained_scans = stream_work_product(
        capacity,
        stream_work_sum(&[stream_work_product(2, vocabulary)?, protected_members])?,
    )?;
    let drift_scans = stream_work_product(stream_work_product(6, capacity)?, vocabulary)?;
    // Begin verifies both full receipts, original selected canary rows, protected
    // positions and both goal roots. Finish checks all five and seals comparison.
    stream_work_events(
        &mut ceiling,
        2,
        stream_work_sum(&[
            121,
            protected_members,
            40,
            5,
            symbolic_scans,
            retained_scans,
            drift_scans,
        ])?,
        stream_work_product(stream_work_product(2, capacity)?, vocabulary)?,
        stream_work_sum(&[
            size_of::<ModelCanaryStreamState>() as u64,
            stream_work_product(
                2,
                stream_work_sum(&[
                    size_of::<ModelForwardStreamState>() as u64,
                    size_of::<ModelForwardReceipt>() as u64,
                ])?,
            )?,
            32,
            5 * 32,
            size_of::<ModelCanaryStreamReceipt>() as u64,
            stream_work_product(16, capacity)?,
            3 * 32,
            size_of::<ModelCanaryStreamReceipt>() as u64,
            8,
        ])?,
        1,
    )?;
    stream_work_hash(
        &mut ceiling,
        size_of::<ModelForwardReceipt>() as u64 - 32,
        2,
    )?;
    stream_work_hash(&mut ceiling, 64, 10)?;
    stream_work_hash(
        &mut ceiling,
        5 * size_of::<ModelCanaryStreamRow>() as u64,
        1,
    )?;
    stream_work_hash(
        &mut ceiling,
        size_of::<ModelCanaryStreamReceipt>() as u64 - 32,
        1,
    )?;
    native_work_bound::native_work_ceiling_units(ceiling)?;
    Ok(ceiling)
}

fn stream_comparison_native_work_ceiling(
    publication_queries: u64,
    protected_members: u64,
) -> Result<[u64; 9], SemanticTransitionError> {
    // Comparison authenticates its five rows (49 slots); two full receipt checks
    // use 28; finite flags use two; three nine-counter validation/merge pairs use 54.
    // Two selection identities use eight, distinct receipt identities use four,
    // and the five paired row-seal checks use 45 more slots.
    let mut ceiling = [0; 9];
    // The same original gate also acquires its publication (two identities),
    // checks the objective identity, four truth tokens and their six pairs,
    // visits every publication query and protected position, and checks three
    // identities per canary. Its two result passes visit five rows each; final
    // refusal validation visits five results and four identity words.
    let gate_slots = stream_work_sum(&[
        2 * 4,
        4,
        4 + 6,
        publication_queries,
        protected_members,
        5 * 3 * 4,
        5,
        5,
        5 + 4,
    ])?;
    let word = size_of::<u64>() as u64;
    let refusal_bytes = size_of::<SemanticTrainingCanaryRefusalRecord>() as u64;
    // Each existing canary iteration may replace the refusal once in each of
    // two passes. Include all five result-reason writes, the complete result
    // records, and both final work-field writes on the refusal branch.
    let gate_bytes = stream_work_sum(&[
        refusal_bytes + word,
        5 * 3 * size_of::<SemanticTrainingCanaryResultRecord>() as u64,
        2 * 5 * (refusal_bytes - word),
        5 * word,
        refusal_bytes + size_of::<u8>() as u64 + 6 * word,
        6 * word,
    ])?;
    stream_work_events(
        &mut ceiling,
        1,
        stream_work_sum(&[49 + 28 + 2 + 54 + 8 + 4 + 45, gate_slots])?,
        0,
        stream_work_sum(&[
            size_of::<ModelCanaryStreamState>() as u64,
            size_of::<ModelCanaryStreamReceipt>() as u64,
            2 * size_of::<ModelForwardReceipt>() as u64,
            2 * (size_of::<ModelForwardStreamState>() as u64
                + size_of::<ModelForwardReceipt>() as u64),
            2 * 8,
            gate_bytes,
        ])?,
        1,
    )?;
    stream_work_hash(
        &mut ceiling,
        5 * size_of::<ModelCanaryStreamRow>() as u64,
        1,
    )?;
    stream_work_hash(
        &mut ceiling,
        size_of::<ModelCanaryStreamReceipt>() as u64 - 32,
        1,
    )?;
    stream_work_hash(
        &mut ceiling,
        size_of::<ModelForwardReceipt>() as u64 - 32,
        2,
    )?;
    native_work_bound::native_work_ceiling_units(ceiling)?;
    Ok(ceiling)
}

/// Complete finite producer geometry for one original Update bank. The caller
/// supplies only the native admitted task-query/protected-roster bounds, never
/// tile labels, source values, host statistics or purported observed quantities.
/// Paired obligations and publication queries retain their distinct native owners.
pub(super) fn model_stream_native_work_ceilings(
    storage: &PublicationStorage,
    rows: usize,
    capacity: usize,
    vocabulary: u64,
    scalar_type: u64,
    block_positions: u64,
    task_queries: u64,
    publication_queries: u64,
    protected_members: u64,
) -> Result<ModelStreamNativeWorkCeilings, SemanticTransitionError> {
    let geometry = StreamGeometry::new(rows, capacity, vocabulary, scalar_type, block_positions)?;
    let ceilings = ModelStreamNativeWorkCeilings {
        forwards: [
            full_forward_native_work_ceiling(storage, geometry, 0)?,
            full_forward_native_work_ceiling(storage, geometry, 1)?,
        ],
        canary: paired_canary_native_work_ceiling(geometry, task_queries, protected_members)?,
        comparison: stream_comparison_native_work_ceiling(publication_queries, protected_members)?,
    };
    ceilings.combined()?;
    Ok(ceilings)
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ModelStreamLeaf {
    digest: Identity256,
    finite: u64,
    work: [u64; 11],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct ModelForwardStreamState {
    abi: u64,
    phase: u64,
    row: u64,
    position: u64,
    row_count: u64,
    capacity: u64,
    vocabulary: u64,
    scalar_type: u64,
    block_positions: u64,
    role: u64,
    generation: u64,
    publication_word: u64,
    finite: u64,
    blocks: u64,
    row_digest: Identity256,
    logits_digest: Identity256,
    selection_identity: Identity256,
    work: [u64; 11],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct ModelCanaryStreamRow {
    row_ordinal: u64,
    kind: u64,
    reason: u64,
    obligations: u64,
    baseline_correct: u64,
    candidate_correct: u64,
    lost: u64,
    retention_denominator: u64,
    protected_seen: u64,
    protected_lost: u64,
    positions: u64,
    maximum_bits: u64,
    baseline_digest: Identity256,
    candidate_digest: Identity256,
    identity: Identity256,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct ModelCanaryStreamState {
    abi: u64,
    phase: u64,
    canary_index: u64,
    position: u64,
    row_count: u64,
    capacity: u64,
    vocabulary: u64,
    scalar_type: u64,
    block_positions: u64,
    publication_word: u64,
    selection_identity: Identity256,
    rows: [ModelCanaryStreamRow; 5],
    work: [u64; 11],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ModelCanaryStreamPosition {
    metrics: ModelCanaryStreamRow,
    work: [u64; 11],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct ModelCanaryStreamReceipt {
    abi: u64,
    availability: u64,
    row_count: u64,
    capacity: u64,
    vocabulary: u64,
    baseline_identity: Identity256,
    candidate_identity: Identity256,
    selection_identity: Identity256,
    summary_digest: Identity256,
    identity: Identity256,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct ModelForwardStreamInputs {
    pub(super) control: u64,
    pub(super) lease: u64,
    pub(super) selection: u64,
    pub(super) state: u64,
    pub(super) row_seals: u64,
    pub(super) receipt: u64,
    pub(super) row_count: u64,
    pub(super) capacity: u64,
    pub(super) vocabulary: u64,
    pub(super) scalar_type: u64,
    pub(super) block_positions: u64,
    pub(super) role: u64,
    pub(super) row: u64,
    pub(super) position_begin: u64,
    pub(super) count: u64,
    pub(super) logits: u64,
    pub(super) position_stride_bytes: u64,
    pub(super) work_sequence: u64,
    pub(super) leaf_scratch: u64,
    pub(super) leaf_capacity: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ModelCanaryStreamInputs {
    control: u64,
    lease: u64,
    selection: u64,
    roster_rows: u64,
    objective: u64,
    canaries: u64,
    protected_members: u64,
    protected_member_count: u64,
    task: u64,
    task_words: u64,
    retention_labels: u64,
    kinds: u64,
    state: u64,
    receipt: u64,
    baseline_state: u64,
    candidate_state: u64,
    baseline_row_seals: u64,
    candidate_row_seals: u64,
    forward_receipts: u64,
    canary_index: u64,
    position_begin: u64,
    count: u64,
    baseline_logits: u64,
    candidate_logits: u64,
    position_stride_bytes: u64,
    baseline_leaf_scratch: u64,
    candidate_leaf_scratch: u64,
    leaf_capacity: u64,
    position_scratch: u64,
}

// SAFETY: these are pointer-free, fixed repr(C) records mirrored by the CUDA
// header. Their integer addresses are bound only while native owners are retained.
unsafe impl DeviceRepr for ModelForwardStreamState {}
unsafe impl DeviceRepr for ModelCanaryStreamState {}
unsafe impl DeviceRepr for ModelCanaryStreamReceipt {}
unsafe impl DeviceRepr for ModelStreamLeaf {}
unsafe impl DeviceRepr for ModelCanaryStreamPosition {}

macro_rules! stream_kernel_parameter {
    ($visibility:vis $record:ident, $parameter:ident) => {
        $visibility struct $parameter($record);
        impl crate::cuda_compat::KernelParamStorage for $parameter {
            fn as_kernel_param(&self) -> *mut std::ffi::c_void {
                (&self.0 as *const $record).cast_mut().cast()
            }
        }
        impl crate::cuda_compat::IntoKernelParamStorage for $record {
            type Storage = $parameter;
            fn into_kernel_param_storage(self) -> Self::Storage {
                $parameter(self)
            }
        }
    };
}
stream_kernel_parameter!(pub(super) ModelForwardStreamInputs, ModelForwardStreamParameter);
stream_kernel_parameter!(ModelCanaryStreamInputs, ModelCanaryStreamParameter);

const _: () = assert!(size_of::<ModelForwardStreamState>() == 296);
const _: () = assert!(size_of::<ModelForwardStreamInputs>() == 160);
const _: () = assert!(size_of::<ModelStreamLeaf>() == 128);
const _: () = assert!(size_of::<ModelCanaryStreamRow>() == 192);
const _: () = assert!(size_of::<ModelCanaryStreamState>() == 1160);
const _: () = assert!(size_of::<ModelCanaryStreamPosition>() == 280);
const _: () = assert!(size_of::<ModelCanaryStreamReceipt>() == 200);
const _: () = assert!(size_of::<ModelCanaryStreamInputs>() == 232);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct StreamGeometry {
    pub(super) rows: u64,
    pub(super) capacity: u64,
    pub(super) vocabulary: u64,
    pub(super) scalar_type: u64,
    pub(super) block_positions: u64,
}

impl StreamGeometry {
    pub(super) fn new(
        rows: usize,
        capacity: usize,
        vocabulary: u64,
        scalar_type: u64,
        block_positions: u64,
    ) -> Result<Self, SemanticTransitionError> {
        let geometry = Self {
            rows: u64::try_from(rows).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            capacity: u64::try_from(capacity)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            vocabulary,
            scalar_type,
            block_positions,
        };
        if geometry.rows == 0
            || geometry.capacity == 0
            || vocabulary == 0
            || block_positions == 0
            || !matches!(scalar_type, 4..=6)
        {
            return Err(publication_input_error(
                "model stream requires positive complete geometry",
            ));
        }
        geometry
            .rows
            .checked_mul(geometry.capacity)
            .and_then(|positions| positions.checked_mul(vocabulary))
            .and_then(|values| values.checked_mul(geometry.element_bytes()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        geometry.scratch_bytes()?;
        Ok(geometry)
    }

    fn element_bytes(self) -> u64 {
        if self.scalar_type == 6 {
            4
        } else {
            2
        }
    }

    fn scratch_bytes(self) -> Result<u64, SemanticTransitionError> {
        self.block_positions
            .min(self.capacity)
            .checked_mul(self.vocabulary)
            .and_then(|values| values.checked_mul(self.element_bytes()))
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }

    fn leaf_capacity(self) -> Result<usize, SemanticTransitionError> {
        let leaves = self.scratch_bytes()?.div_ceil(MODEL_STREAM_LEAF_BYTES);
        u32::try_from(leaves.div_ceil(128))
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        usize::try_from(leaves).map_err(|_| SemanticTransitionError::GenerationExhausted)
    }

    fn count_at(self, begin: u64) -> Result<u64, SemanticTransitionError> {
        if begin >= self.capacity || begin % self.block_positions != 0 {
            return Err(publication_input_error(
                "model stream block is outside its original partition",
            ));
        }
        Ok((self.capacity - begin).min(self.block_positions))
    }

    fn require_block(
        self,
        tensor: &PreparedSemanticTensor,
        role: usize,
        count: u64,
    ) -> Result<(), SemanticTransitionError> {
        let stride = self
            .vocabulary
            .checked_mul(self.element_bytes())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if tensor.layout
            != (SemanticTensorLayout {
                role: 0,
                index: role as u64,
                element_bytes: self.element_bytes(),
                scalar_type: self.scalar_type,
                rank: 2,
                logical_axis: u64::MAX,
                dimensions: [count, self.vocabulary, 0, 0],
                strides_bytes: [stride, self.element_bytes(), 0, 0],
            })
            || tensor.logical_begin != 0
            || tensor.logical_end != 0
        {
            return Err(publication_input_error(
                "model stream requires the actual contiguous F16/BF16/F32 [count,vocabulary] block",
            ));
        }
        Ok(())
    }
}

#[derive(Default)]
struct StreamRecording {
    phase: u8,
    row: u64,
    position: u64,
    scratch: [Option<PreparedSemanticTensor>; 2],
    candidate_owner: Option<SemanticTensorOwner>,
}

/// An original prepared full forward, not a retained or fabricated dense tensor.
#[derive(Clone)]
pub struct SemanticModelForwardStream {
    issuer: Arc<()>,
    reader_token: u64,
    origin: ModelForwardStreamOrigin,
    role: usize,
    owner: Arc<ModelForwardStreamStorage>,
}

#[derive(Clone)]
pub(super) enum ModelForwardStreamOrigin {
    Update { bank: usize },
    Evaluation(model_evaluation::SemanticModelEvaluation),
}

pub(super) struct ModelForwardStreamStorage {
    state: TrackedCudaSlice<ModelForwardStreamState>,
    row_seals: TrackedCudaSlice<u64>,
    leaves: TrackedCudaSlice<ModelStreamLeaf>,
    // Evaluation has no prepared Update receipt bank. It owns one genuine
    // receipt in the same stream storage, retained through original retirement.
    evaluation_receipt: Option<TrackedCudaSlice<ModelForwardReceipt>>,
    geometry: StreamGeometry,
    native_work_ceiling: [u64; 9],
    kernels: [CudaFunction; 4],
    recording: Mutex<StreamRecording>,
    // Reservation is last: metadata and genuine scratch owners drop first.
    _host_reservation: Option<SemanticReplayMaterialReservation>,
}

impl SemanticModelForwardStream {
    pub fn role(&self) -> usize {
        self.role
    }

    pub fn bank(&self) -> Option<usize> {
        match &self.origin {
            ModelForwardStreamOrigin::Update { bank } => Some(*bank),
            ModelForwardStreamOrigin::Evaluation(_) => None,
        }
    }

    pub fn geometry(&self) -> (u64, u64, u64, u64, u64) {
        let geometry = self.owner.geometry;
        (
            geometry.rows,
            geometry.capacity,
            geometry.vocabulary,
            geometry.scalar_type,
            geometry.block_positions,
        )
    }

    fn update_bank(&self) -> Result<usize, SemanticTransitionError> {
        self.bank().ok_or_else(|| {
            publication_input_error("evaluation forward has no prepared Update bank")
        })
    }

    pub(super) fn belongs_to_evaluation(
        &self,
        handle: &model_evaluation::SemanticModelEvaluation,
    ) -> bool {
        matches!(&self.origin, ModelForwardStreamOrigin::Evaluation(original)
            if original.same_invocation(handle))
    }

    pub(super) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.issuer, &other.issuer)
            && self.reader_token == other.reader_token
            && Arc::ptr_eq(&self.owner, &other.owner)
    }

    pub(super) fn evaluation_receipt_pointer(&self) -> Result<u64, SemanticTransitionError> {
        self.owner
            .evaluation_receipt
            .as_ref()
            .map(TrackedCudaSlice::device_ptr_value)
            .ok_or(SemanticTransitionError::ObservationMismatch)
    }

    pub(super) fn require_finished_scratch(
        &self,
    ) -> Result<PreparedSemanticTensor, SemanticTransitionError> {
        let recording = self
            .owner
            .recording
            .lock()
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        if recording.phase != 2
            || recording.row != self.owner.geometry.rows
            || recording.position != 0
        {
            return Err(publication_input_error(
                "model forward requires its complete original producer finish",
            ));
        }
        recording.scratch[0]
            .clone()
            .ok_or(SemanticTransitionError::ObservationMismatch)
    }

    pub(super) fn state_pointer(&self) -> u64 {
        self.owner.state.device_ptr_value()
    }

    pub(super) fn row_seals_pointer(&self) -> u64 {
        self.owner.row_seals.device_ptr_value()
    }

    pub(super) fn native_work_ceiling(&self) -> [u64; 9] {
        self.owner.native_work_ceiling
    }

    pub(super) fn inputs(
        &self,
        control: u64,
        lease: u64,
        selection: u64,
        receipt: u64,
    ) -> ModelForwardStreamInputs {
        let geometry = self.owner.geometry;
        ModelForwardStreamInputs {
            control,
            lease,
            selection,
            receipt,
            state: self.state_pointer(),
            row_seals: self.row_seals_pointer(),
            row_count: geometry.rows,
            capacity: geometry.capacity,
            vocabulary: geometry.vocabulary,
            scalar_type: geometry.scalar_type,
            block_positions: geometry.block_positions,
            role: self.role as u64,
            leaf_scratch: self.owner.leaves.device_ptr_value(),
            leaf_capacity: self.owner.leaves.len() as u64,
            ..Default::default()
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut LaunchRecorder,
    ) -> Result<(), SemanticTransitionError> {
        self.owner.record(recorder)
    }
}

/// The paired original five canaries, bound to both complete full forwards.
#[derive(Clone)]
pub struct SemanticModelCanaryStream {
    issuer: Arc<()>,
    reader_token: u64,
    bank: usize,
    owner: Arc<ModelCanaryStreamStorage>,
}

pub(super) struct ModelCanaryStreamStorage {
    state: TrackedCudaSlice<ModelCanaryStreamState>,
    receipt: TrackedCudaSlice<ModelCanaryStreamReceipt>,
    leaves: [TrackedCudaSlice<ModelStreamLeaf>; 2],
    positions: TrackedCudaSlice<ModelCanaryStreamPosition>,
    forwards: [SemanticModelForwardStream; 2],
    canary_rows: [u64; 5],
    native_work_ceiling: [u64; 9],
    comparison_work_ceiling: [u64; 9],
    kernels: [CudaFunction; 5],
    recording: Mutex<StreamRecording>,
    _host_reservation: Option<SemanticReplayMaterialReservation>,
}

/// Opaque original comparison; a Python Boolean cannot replace this capability.
#[derive(Clone)]
pub struct SemanticModelCanaryWitness {
    stream: SemanticModelCanaryStream,
}

impl SemanticModelCanaryStream {
    pub fn canary_rows(&self) -> [u64; 5] {
        self.owner.canary_rows
    }

    pub(super) fn state_pointer(&self) -> u64 {
        self.owner.state.device_ptr_value()
    }

    pub(super) fn receipt_pointer(&self) -> u64 {
        self.owner.receipt.device_ptr_value()
    }

    pub(super) fn complete_native_work_ceiling(&self) -> Result<[u64; 9], SemanticTransitionError> {
        ModelStreamNativeWorkCeilings {
            forwards: [
                self.owner.forwards[0].native_work_ceiling(),
                self.owner.forwards[1].native_work_ceiling(),
            ],
            canary: self.owner.native_work_ceiling,
            comparison: self.owner.comparison_work_ceiling,
        }
        .combined()
    }

    pub(super) fn record(
        &self,
        recorder: &mut LaunchRecorder,
    ) -> Result<(), SemanticTransitionError> {
        self.owner.record(recorder)
    }
}

impl SemanticModelCanaryWitness {
    pub(super) fn stream(&self) -> &SemanticModelCanaryStream {
        &self.stream
    }
}

impl ModelForwardStreamStorage {
    pub(super) fn record(
        &self,
        recorder: &mut LaunchRecorder,
    ) -> Result<(), SemanticTransitionError> {
        recorder.read_write(&self.state);
        recorder.read_write(&self.row_seals);
        recorder.read_write(&self.leaves);
        if let Some(receipt) = &self.evaluation_receipt {
            recorder.read_write(receipt);
        }
        for tensor in self
            .recording
            .lock()
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?
            .scratch
            .iter()
            .flatten()
        {
            if let Some(source) = &tensor.source {
                recorder.read(source);
            }
        }
        Ok(())
    }
}

impl ModelCanaryStreamStorage {
    pub(super) fn record(
        &self,
        recorder: &mut LaunchRecorder,
    ) -> Result<(), SemanticTransitionError> {
        recorder.read_write(&self.state);
        recorder.read_write(&self.receipt);
        recorder.read_write(&self.leaves[0]);
        recorder.read_write(&self.leaves[1]);
        recorder.read_write(&self.positions);
        for forward in &self.forwards {
            forward.owner.record(recorder)?;
        }
        for tensor in self
            .recording
            .lock()
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?
            .scratch
            .iter()
            .flatten()
        {
            if let Some(source) = &tensor.source {
                recorder.read(source);
            }
        }
        Ok(())
    }
}

impl SemanticTransitionSession {
    /// Authenticate all three original producer finishes before the existing
    /// output binder records its sole numerical/publication gate. Returned
    /// scratch roots are only four retained allocations, never logical logits.
    pub(super) fn require_streamed_model_update_output(
        &self,
        step: &SemanticPreparedStep,
        bank: usize,
        baseline: &SemanticModelForwardWitness,
        candidate: &SemanticModelForwardWitness,
        comparison: &SemanticModelCanaryWitness,
        output: &SemanticTensorOwner,
        consumer_stream: u64,
    ) -> Result<[PreparedSemanticTensor; 4], SemanticTransitionError> {
        let forwards = [&baseline.stream, &candidate.stream];
        for (role, forward) in forwards.iter().enumerate() {
            self.require_model_forward_stream(step, forward, consumer_stream)?;
            if forward.bank() != Some(bank) || forward.role != role {
                return Err(publication_input_error(
                    "model update changed an original full-forward bank or role",
                ));
            }
            let recording = forward
                .owner
                .recording
                .lock()
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
            if recording.phase != 2 {
                return Err(publication_input_error(
                    "model update requires both complete original full-forward finishes",
                ));
            }
            if role == 1 {
                let original = recording
                    .candidate_owner
                    .as_ref()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if !Arc::ptr_eq(&original.issuer, &output.issuer)
                    || original.reader_token != output.reader_token
                    || original.index != output.index
                {
                    return Err(publication_input_error("candidate forward finish must follow this exact original model output content witness"));
                }
            }
        }
        let canary = &comparison.stream;
        self.require_model_canary_stream(step, canary, consumer_stream)?;
        if canary.bank != bank
            || !Arc::ptr_eq(&canary.owner.forwards[0].owner, &forwards[0].owner)
            || !Arc::ptr_eq(&canary.owner.forwards[1].owner, &forwards[1].owner)
        {
            return Err(publication_input_error(
                "model update comparison belongs to different original full forwards",
            ));
        }
        let paired = canary
            .owner
            .recording
            .lock()
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        if paired.phase != 2 {
            return Err(publication_input_error(
                "model update requires the complete original paired-canary finish",
            ));
        }
        let scratch = |stream: &SemanticModelForwardStream| -> Result<PreparedSemanticTensor, SemanticTransitionError> {
            stream.owner.recording.lock().map_err(|_| SemanticTransitionError::ObservationMismatch)?
                .scratch[0].clone().ok_or(SemanticTransitionError::ObservationMismatch)
        };
        Ok([
            scratch(forwards[0])?,
            scratch(forwards[1])?,
            paired.scratch[0]
                .clone()
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
            paired.scratch[1]
                .clone()
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
        ])
    }

    /// Cold-only: derive R and C from the exact native Update training view.
    pub fn prepare_model_forward_stream(
        &mut self,
        step: &SemanticPreparedStep,
        bank: usize,
        role: usize,
        vocabulary: u64,
        scalar_type: u64,
        block_positions: u64,
    ) -> Result<SemanticModelForwardStream, SemanticTransitionError> {
        self.check_prepared_cold(step)?;
        self.require_prepared_structure_mutable()?;
        if bank > 1 || role > 1 || !self.prepared_transition_kind(step)?.uses_pending_model() {
            return Err(publication_input_error(
                "model forward stream belongs to an original pending-model bank and role",
            ));
        }
        let prepared = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("checked step");
        let update = prepared.branches[bank]
            .model_update
            .as_ref()
            .ok_or_else(|| {
                publication_input_error(
                    "model forward stream has no original prepared model update",
                )
            })?;
        if update.forward_streams[role].is_some() {
            return Err(publication_input_error(
                "model forward stream is prepared once per original bank and role",
            ));
        }
        let view = prepared.training_view.as_ref().ok_or_else(|| {
            publication_input_error("model forward stream has no original native training view")
        })?;
        let geometry = StreamGeometry::new(
            view.roster_rows().len(),
            view.capacity(),
            vocabulary,
            scalar_type,
            block_positions,
        )?;
        // Overflow and the complete native loop geometry are checked before
        // allocating any stream storage. Original Program admission owns the
        // eventual aggregate allowance; a tile cannot supply its own ceiling.
        let native_work_ceiling =
            full_forward_native_work_ceiling(self.prepared_publication()?, geometry, role)?;
        self.require_model_stream_cold_allowance(step)?;
        let stream = self.allocate_model_forward_stream(
            step.token,
            ModelForwardStreamOrigin::Update { bank },
            role,
            geometry,
            native_work_ceiling,
        )?;
        self.steps
            .get_mut(&step.token)
            .expect("checked step")
            .prepared
            .as_mut()
            .expect("checked prepared owner")
            .branches[bank]
            .model_update
            .as_mut()
            .expect("checked original update")
            .forward_streams[role] = Some(stream.clone());
        Ok(stream)
    }

    pub(super) fn allocate_model_forward_stream(
        &mut self,
        reader_token: u64,
        origin: ModelForwardStreamOrigin,
        role: usize,
        geometry: StreamGeometry,
        native_work_ceiling: [u64; 9],
    ) -> Result<SemanticModelForwardStream, SemanticTransitionError> {
        let kernel_names = [
            "semantic_model_forward_stream_begin",
            "semantic_model_forward_stream_block_leaves",
            "semantic_model_forward_stream_block",
            "semantic_model_forward_stream_finish",
        ];
        let [begin, leaves, fold, finish] = kernel_names.map(|name| self.model_stream_kernel(name));
        let kernels = [begin?, leaves?, fold?, finish?];
        let host_bytes = size_of::<ModelForwardStreamStorage>()
            .checked_add(size_of::<SemanticModelForwardStream>())
            .and_then(|bytes| bytes.checked_add(2 * size_of::<PreparedSemanticTensor>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let host = self
            .replay_material_budget
            .as_ref()
            .map(|budget| budget.reserve(host_bytes as u64))
            .transpose()?;
        let evaluation = matches!(&origin, ModelForwardStreamOrigin::Evaluation(_));
        let row_cells = usize::try_from(geometry.rows)
            .ok()
            .and_then(|rows| rows.checked_mul(4))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let bytes = row_cells
            .checked_mul(size_of::<u64>())
            .and_then(|bytes| bytes.checked_add(size_of::<ModelForwardStreamState>()))
            .and_then(|bytes| {
                geometry
                    .leaf_capacity()
                    .ok()
                    .and_then(|leaves| leaves.checked_mul(size_of::<ModelStreamLeaf>()))
                    .and_then(|leaves| bytes.checked_add(leaves))
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let bytes = bytes
            .checked_add(if evaluation {
                size_of::<ModelForwardReceipt>()
            } else {
                0
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut reservation = self
            .provider
            .memory()
            .reserve_bytes(bytes as u64)
            .map_err(|error| runtime_error("model forward stream reservation", error))?;
        let owner = Arc::new(ModelForwardStreamStorage {
            state: reservation
                .alloc(1)
                .map_err(|error| runtime_error("model forward stream cursor allocation", error))?,
            row_seals: reservation
                .alloc(row_cells)
                .map_err(|error| runtime_error("model forward row-seal allocation", error))?,
            leaves: reservation
                .alloc(geometry.leaf_capacity()?)
                .map_err(|error| runtime_error("model forward leaf allocation", error))?,
            evaluation_receipt: if evaluation {
                Some(reservation.alloc(1).map_err(|error| {
                    runtime_error("evaluation forward receipt allocation", error)
                })?)
            } else {
                None
            },
            geometry,
            native_work_ceiling,
            kernels,
            recording: Mutex::new(StreamRecording::default()),
            _host_reservation: host,
        });
        Ok(SemanticModelForwardStream {
            issuer: Arc::clone(&self.publication_issuer),
            reader_token,
            origin,
            role,
            owner,
        })
    }

    pub fn begin_model_forward_stream(
        &mut self,
        step: &SemanticPreparedStep,
        stream: &SemanticModelForwardStream,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.require_model_forward_stream(step, stream, consumer_stream)?;
        let update = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("checked step")
            .branches[stream.update_bank()?]
        .model_update
        .as_ref()
        .expect("checked update");
        if stream.role == 1 {
            let baseline = update.forward_streams[0]
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if baseline
                .owner
                .recording
                .lock()
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?
                .phase
                != 2
            {
                return Err(publication_input_error(
                    "candidate full forward begins after the original baseline finishes",
                ));
            }
        }
        let inputs = self.model_forward_stream_inputs(step, stream)?;
        self.begin_model_forward_stream_inner(stream, inputs)
    }

    pub(super) fn begin_model_forward_stream_inner(
        &mut self,
        stream: &SemanticModelForwardStream,
        inputs: ModelForwardStreamInputs,
    ) -> Result<(), SemanticTransitionError> {
        {
            let mut recording = stream
                .owner
                .recording
                .lock()
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
            if recording.phase != 0 {
                return Err(publication_input_error(
                    "model forward stream begin is recorded exactly once",
                ));
            }
            recording.phase = 1;
        }
        self.enqueue_model_forward_stream(stream, 0, inputs)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the original producer supplies one exact contiguous block occurrence"
    )]
    pub fn record_model_forward_stream_block(
        &mut self,
        step: &SemanticPreparedStep,
        stream: &SemanticModelForwardStream,
        row: u64,
        position_begin: u64,
        count: u64,
        logits: SemanticTensorInput,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.require_model_forward_stream(step, stream, consumer_stream)?;
        let inputs = self.model_forward_stream_inputs(step, stream)?;
        self.record_model_forward_stream_block_inner(
            stream,
            row,
            position_begin,
            count,
            logits,
            inputs,
        )
    }

    pub(super) fn record_model_forward_stream_block_inner(
        &mut self,
        stream: &SemanticModelForwardStream,
        row: u64,
        position_begin: u64,
        count: u64,
        logits: SemanticTensorInput,
        mut inputs: ModelForwardStreamInputs,
    ) -> Result<(), SemanticTransitionError> {
        let mut tensors = prepare_semantic_tensors(&self.provider, vec![logits])?;
        let tensor = tensors
            .pop()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let pointer = tensor.data;
        let result = (|| {
            let mut recording = stream
                .owner
                .recording
                .lock()
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
            // Retain the producer before checking it. At most one rejected owner
            // can exist: any failed record poisons the complete construction.
            let retained = recording.scratch[0].as_ref().map(|original| original.data);
            if retained.is_none() {
                recording.scratch[0] = Some(tensor.clone());
            } else if retained != Some(pointer) {
                recording.scratch[1] = Some(tensor.clone());
            }
            if retained.is_some_and(|original| original != pointer) {
                return Err(publication_input_error(
                    "model forward blocks must reuse their one original scratch allocation",
                ));
            }
            let geometry = stream.owner.geometry;
            if recording.phase != 1
                || row != recording.row
                || position_begin != recording.position
                || row >= geometry.rows
                || count != geometry.count_at(position_begin)?
            {
                return Err(publication_input_error(
                    "model forward blocks must cover every row and position in original order",
                ));
            }
            geometry.require_block(&tensor, stream.role, count)?;
            self.require_stream_scratch_disjoint(stream, &tensor)?;
            recording.position += count;
            if recording.position == geometry.capacity {
                recording.position = 0;
                recording.row += 1;
            }
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result?;
        inputs.row = row;
        inputs.position_begin = position_begin;
        inputs.count = count;
        inputs.logits = pointer;
        inputs.position_stride_bytes = tensor.layout.strides_bytes[0];
        self.enqueue_model_forward_stream(stream, 1, inputs)
    }

    pub fn finish_model_forward_stream(
        &mut self,
        step: &SemanticPreparedStep,
        stream: &SemanticModelForwardStream,
        allocation_owner: Option<&SemanticTensorOwner>,
        consumer_stream: u64,
    ) -> Result<SemanticModelForwardWitness, SemanticTransitionError> {
        self.require_model_forward_stream(step, stream, consumer_stream)?;
        if (stream.role == 0) != allocation_owner.is_none() {
            return Err(publication_input_error(
                "only the candidate full forward requires its retained original output owner",
            ));
        }
        if let Some(witness) = allocation_owner {
            self.checked_prepared_tensor_owner(step, witness)?;
            if self.steps[&step.token].content[witness.index]
                .model_bank
                .is_some()
            {
                return Err(publication_input_error(
                    "candidate full forward requires its original output storage owner",
                ));
            }
        }
        let mut inputs = self.model_forward_stream_inputs(step, stream)?;
        let prepared = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("checked step");
        inputs.work_sequence = prepared
            .model_work
            .as_ref()
            .expect("checked active model work")
            .next_slot()
            .map_err(publication_input_error)? as u64;
        if inputs.work_sequence == 0 {
            return Err(publication_input_error(
                "full forward finish requires its actual producer model work",
            ));
        }
        self.finish_model_forward_stream_inner(stream, allocation_owner, inputs)
    }

    pub(super) fn finish_model_forward_stream_inner(
        &mut self,
        stream: &SemanticModelForwardStream,
        allocation_owner: Option<&SemanticTensorOwner>,
        inputs: ModelForwardStreamInputs,
    ) -> Result<SemanticModelForwardWitness, SemanticTransitionError> {
        if inputs.work_sequence == 0 {
            return Err(publication_input_error(
                "full forward finish requires its actual producer model work",
            ));
        }
        {
            let mut recording = stream
                .owner
                .recording
                .lock()
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
            if recording.phase != 1
                || recording.row != stream.owner.geometry.rows
                || recording.position != 0
            {
                return Err(publication_input_error(
                    "full forward finish requires complete original row and position coverage",
                ));
            }
            recording.candidate_owner = allocation_owner.cloned();
            recording.phase = 2;
        }
        self.enqueue_model_forward_stream(stream, 2, inputs)?;
        Ok(SemanticModelForwardWitness::from_stream(stream.clone()))
    }

    pub fn prepare_model_canary_stream(
        &mut self,
        step: &SemanticPreparedStep,
        bank: usize,
        baseline: &SemanticModelForwardStream,
        candidate: &SemanticModelForwardStream,
    ) -> Result<SemanticModelCanaryStream, SemanticTransitionError> {
        self.check_prepared_cold(step)?;
        self.require_prepared_structure_mutable()?;
        for (role, forward) in [(0, baseline), (1, candidate)] {
            self.require_model_forward_stream_identity(step, forward)?;
            if forward.bank() != Some(bank) || forward.role != role {
                return Err(publication_input_error("paired canaries require the original baseline and candidate streams of this bank"));
            }
        }
        if baseline.owner.geometry != candidate.owner.geometry {
            return Err(publication_input_error(
                "paired canaries require identical complete forward geometry",
            ));
        }
        let prepared = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("checked step");
        let update = prepared.branches[bank]
            .model_update
            .as_ref()
            .expect("checked original update");
        if update.canary_stream.is_some() {
            return Err(publication_input_error(
                "paired canary stream is prepared once per original bank",
            ));
        }
        let canary_rows = prepared
            .training_view
            .as_ref()
            .expect("checked training view")
            .canary_rows();
        let geometry = baseline.owner.geometry;
        let task_queries = u64::try_from(
            self.training_canary_source
                .as_ref()
                .ok_or_else(|| {
                    publication_input_error(
                        "paired canary work requires its original admitted symbolic source",
                    )
                })?
                .statements
                .len(),
        )
        .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let publication_queries = u64::try_from(
            self.task
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .0
                .spec
                .statement_records
                .len(),
        )
        .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let protected_members = u64::try_from(
            prepared
                .training_view
                .as_ref()
                .expect("checked training view")
                .protected_members()
                .len(),
        )
        .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let ceilings = model_stream_native_work_ceilings(
            self.prepared_publication()?,
            geometry.rows as usize,
            geometry.capacity as usize,
            geometry.vocabulary,
            geometry.scalar_type,
            geometry.block_positions,
            task_queries,
            publication_queries,
            protected_members,
        )?;
        self.require_model_stream_cold_allowance(step)?;
        let [begin, leaves, positions, fold, finish] = [
            "semantic_model_canary_stream_begin",
            "semantic_model_canary_stream_block_leaves",
            "semantic_model_canary_stream_block_positions",
            "semantic_model_canary_stream_block",
            "semantic_model_canary_stream_finish",
        ]
        .map(|name| self.model_stream_kernel(name));
        let kernels = [begin?, leaves?, positions?, fold?, finish?];
        let host_bytes = size_of::<ModelCanaryStreamStorage>()
            .checked_add(4 * size_of::<usize>())
            .and_then(|bytes| bytes.checked_add(4 * size_of::<PreparedSemanticTensor>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let host = self
            .replay_material_budget
            .as_ref()
            .map(|budget| budget.reserve(host_bytes as u64))
            .transpose()?;
        let leaf_capacity = geometry.leaf_capacity()?;
        let position_capacity = usize::try_from(geometry.block_positions.min(geometry.capacity))
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let bytes = size_of::<ModelCanaryStreamState>()
            .checked_add(size_of::<ModelCanaryStreamReceipt>())
            .and_then(|bytes| {
                leaf_capacity
                    .checked_mul(2 * size_of::<ModelStreamLeaf>())
                    .and_then(|leaves| bytes.checked_add(leaves))
            })
            .and_then(|bytes| {
                position_capacity
                    .checked_mul(size_of::<ModelCanaryStreamPosition>())
                    .and_then(|positions| bytes.checked_add(positions))
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut reservation = self
            .provider
            .memory()
            .reserve_bytes(bytes as u64)
            .map_err(|error| runtime_error("paired model canary stream reservation", error))?;
        let stream = SemanticModelCanaryStream {
            issuer: Arc::clone(&self.publication_issuer),
            reader_token: step.token,
            bank,
            owner: Arc::new(ModelCanaryStreamStorage {
                state: reservation.alloc(1).map_err(|error| {
                    runtime_error("paired model canary cursor allocation", error)
                })?,
                receipt: reservation.alloc(1).map_err(|error| {
                    runtime_error("paired model canary receipt allocation", error)
                })?,
                leaves: [
                    reservation
                        .alloc(leaf_capacity)
                        .map_err(|error| runtime_error("baseline paired leaf allocation", error))?,
                    reservation.alloc(leaf_capacity).map_err(|error| {
                        runtime_error("candidate paired leaf allocation", error)
                    })?,
                ],
                positions: reservation
                    .alloc(position_capacity)
                    .map_err(|error| runtime_error("paired position allocation", error))?,
                forwards: [baseline.clone(), candidate.clone()],
                canary_rows,
                native_work_ceiling: ceilings.canary,
                comparison_work_ceiling: ceilings.comparison,
                kernels,
                recording: Mutex::new(StreamRecording::default()),
                _host_reservation: host,
            }),
        };
        self.steps
            .get_mut(&step.token)
            .expect("checked step")
            .prepared
            .as_mut()
            .expect("checked prepared owner")
            .branches[bank]
            .model_update
            .as_mut()
            .expect("checked original update")
            .canary_stream = Some(stream.clone());
        Ok(stream)
    }

    pub fn begin_model_canary_stream(
        &mut self,
        step: &SemanticPreparedStep,
        stream: &SemanticModelCanaryStream,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.require_model_canary_stream(step, stream, consumer_stream)?;
        for forward in &stream.owner.forwards {
            if forward
                .owner
                .recording
                .lock()
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?
                .phase
                != 2
            {
                return Err(publication_input_error(
                    "paired canaries begin after both complete full-forward finishes",
                ));
            }
        }
        {
            let mut recording = stream
                .owner
                .recording
                .lock()
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
            if recording.phase != 0 {
                return Err(publication_input_error(
                    "paired canary begin is recorded exactly once",
                ));
            }
            recording.phase = 1;
        }
        let inputs = self.model_canary_stream_inputs(step, stream)?;
        self.enqueue_model_canary_stream(stream, 0, inputs)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the paired original model producers supply one exact block occurrence"
    )]
    pub fn record_model_canary_stream_block(
        &mut self,
        step: &SemanticPreparedStep,
        stream: &SemanticModelCanaryStream,
        canary_index: u64,
        position_begin: u64,
        count: u64,
        baseline_logits: SemanticTensorInput,
        candidate_logits: SemanticTensorInput,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.require_model_canary_stream(step, stream, consumer_stream)?;
        let tensors =
            prepare_semantic_tensors(&self.provider, vec![baseline_logits, candidate_logits])?;
        let result = (|| {
            let mut recording = stream
                .owner
                .recording
                .lock()
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
            let geometry = stream.owner.forwards[0].owner.geometry;
            if recording.phase != 1
                || canary_index != recording.row
                || canary_index >= 5
                || position_begin != recording.position
                || count != geometry.count_at(position_begin)?
            {
                return Err(publication_input_error("paired canary blocks must cover all five original rows and every position in order"));
            }
            for (role, tensor) in tensors.iter().enumerate() {
                geometry.require_block(tensor, role, count)?;
                self.require_stream_scratch_disjoint(&stream.owner.forwards[role], tensor)?;
                if let Some(original) = &recording.scratch[role] {
                    if original.data != tensor.data {
                        return Err(publication_input_error(
                            "paired canaries must reuse their two original scratch allocations",
                        ));
                    }
                } else {
                    recording.scratch[role] = Some(tensor.clone());
                }
            }
            if step_input_overlap(
                tensors[0].data,
                tensor_layout_bytes(&tensors[0].layout)?,
                tensors[1].data,
                tensor_layout_bytes(&tensors[1].layout)?,
            )? {
                return Err(publication_input_error(
                    "paired canary baseline and candidate scratch must not alias",
                ));
            }
            recording.position += count;
            if recording.position == geometry.capacity {
                recording.position = 0;
                recording.row += 1;
            }
            Ok(())
        })();
        if result.is_err() {
            // Preserve all actual producers on invalid recording until the
            // original complete construction is retired; never free in flight.
            self.prepared_resources.push(Arc::new(tensors));
            self.poisoned = true;
            return result;
        }
        let mut inputs = self.model_canary_stream_inputs(step, stream)?;
        inputs.canary_index = canary_index;
        inputs.position_begin = position_begin;
        inputs.count = count;
        inputs.baseline_logits = tensors[0].data;
        inputs.candidate_logits = tensors[1].data;
        inputs.position_stride_bytes = tensors[0].layout.strides_bytes[0];
        self.enqueue_model_canary_stream(stream, 1, inputs)
    }

    pub fn finish_model_canary_stream(
        &mut self,
        step: &SemanticPreparedStep,
        stream: &SemanticModelCanaryStream,
        consumer_stream: u64,
    ) -> Result<SemanticModelCanaryWitness, SemanticTransitionError> {
        self.require_model_canary_stream(step, stream, consumer_stream)?;
        {
            let mut recording = stream
                .owner
                .recording
                .lock()
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
            if recording.phase != 1 || recording.row != 5 || recording.position != 0 {
                return Err(publication_input_error(
                    "paired canary finish requires full coverage of all five original rows",
                ));
            }
            recording.phase = 2;
        }
        let inputs = self.model_canary_stream_inputs(step, stream)?;
        self.enqueue_model_canary_stream(stream, 2, inputs)?;
        Ok(SemanticModelCanaryWitness {
            stream: stream.clone(),
        })
    }

    fn model_stream_kernel(&self, name: &str) -> Result<CudaFunction, SemanticTransitionError> {
        self.provider
            .device()
            .inner()
            .get_func("xlog_semantic_transition", name)
            .ok_or_else(|| {
                runtime_error("model stream kernel lookup", format!("{name} unavailable"))
            })
    }

    fn require_model_stream_cold_allowance(
        &self,
        step: &SemanticPreparedStep,
    ) -> Result<(), SemanticTransitionError> {
        self.require_model_stream_cold_allowance_token(step.token)
    }

    pub(super) fn require_model_stream_cold_allowance_token(
        &self,
        token: u64,
    ) -> Result<(), SemanticTransitionError> {
        let allowance = self.cold_native_allowance(token)?.ok_or_else(|| {
            publication_input_error("model stream allocation requires its original cold allowance")
        })?;
        let allowance = allowance
            .lock()
            .map_err(|_| publication_input_error("original model stream allowance is poisoned"))?;
        allowance.require_unfrozen_prepared_scratch()
    }

    fn require_model_forward_stream_identity(
        &self,
        step: &SemanticPreparedStep,
        stream: &SemanticModelForwardStream,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_step(step, false)?;
        if !Arc::ptr_eq(&stream.issuer, &self.publication_issuer)
            || stream.reader_token != step.token
            || stream.bank().is_none_or(|bank| bank > 1)
            || stream.role > 1
        {
            return Err(publication_input_error(
                "model forward stream belongs to another original prepared Update",
            ));
        }
        let original = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("checked step")
            .branches[stream.update_bank()?]
        .model_update
        .as_ref()
        .and_then(|update| update.forward_streams[stream.role].as_ref());
        if original.is_none_or(|original| !Arc::ptr_eq(&original.owner, &stream.owner)) {
            return Err(publication_input_error(
                "model forward stream is not the original native bank owner",
            ));
        }
        Ok(())
    }

    fn require_model_forward_stream(
        &self,
        step: &SemanticPreparedStep,
        stream: &SemanticModelForwardStream,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.require_model_forward_stream_identity(step, stream)?;
        self.check_prepared_content_stream(step, consumer_stream)?;
        if self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("checked step")
            .model_work
            .as_ref()
            .is_none_or(|work| work.capture_bank != stream.bank())
        {
            return Err(publication_input_error(
                "model stream requires its original active producer bank capture",
            ));
        }
        Ok(())
    }

    fn require_model_canary_stream(
        &self,
        step: &SemanticPreparedStep,
        stream: &SemanticModelCanaryStream,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.require_model_forward_stream(step, &stream.owner.forwards[0], consumer_stream)?;
        self.require_model_forward_stream(step, &stream.owner.forwards[1], consumer_stream)?;
        if !Arc::ptr_eq(&stream.issuer, &self.publication_issuer)
            || stream.reader_token != step.token
        {
            return Err(publication_input_error(
                "paired canary stream belongs to another original prepared Update",
            ));
        }
        let original = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("checked step")
            .branches[stream.bank]
            .model_update
            .as_ref()
            .and_then(|update| update.canary_stream.as_ref());
        if original.is_none_or(|original| !Arc::ptr_eq(&original.owner, &stream.owner)) {
            return Err(publication_input_error(
                "paired canary stream is not the original native bank owner",
            ));
        }
        if self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("checked step")
            .training_view
            .as_ref()
            .is_none_or(|view| view.canary_rows() != stream.owner.canary_rows)
        {
            return Err(publication_input_error(
                "paired canary rows changed from their original native roster",
            ));
        }
        Ok(())
    }

    pub(super) fn require_stream_scratch_disjoint(
        &self,
        stream: &SemanticModelForwardStream,
        tensor: &PreparedSemanticTensor,
    ) -> Result<(), SemanticTransitionError> {
        let bytes = tensor_layout_bytes(&tensor.layout)?;
        for owned in &self.model_forward_stream_publication(stream)?.allocations {
            if step_input_overlap(
                tensor.data,
                bytes,
                owned.entry().pointer,
                owned.entry().bytes as usize,
            )? {
                return Err(publication_input_error(
                    "streamed model logits scratch aliases original publication storage",
                ));
            }
        }
        Ok(())
    }

    fn model_forward_stream_inputs(
        &self,
        step: &SemanticPreparedStep,
        stream: &SemanticModelForwardStream,
    ) -> Result<ModelForwardStreamInputs, SemanticTransitionError> {
        let storage = self.prepared_publication()?;
        let prepared = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("checked step");
        let update = prepared.branches[stream.update_bank()?]
            .model_update
            .as_ref()
            .expect("checked update");
        let geometry = stream.owner.geometry;
        Ok(ModelForwardStreamInputs {
            control: storage.control.device_ptr_value(),
            lease: prepared.reader.device_ptr_value(),
            selection: *prepared
                .training_view
                .as_ref()
                .expect("checked training view")
                .selection()
                .device_ptr(),
            state: stream.owner.state.device_ptr_value(),
            row_seals: stream.owner.row_seals.device_ptr_value(),
            receipt: update.forward_receipts.device_ptr_value()
                + (stream.role * size_of::<ModelForwardReceipt>()) as u64,
            row_count: geometry.rows,
            capacity: geometry.capacity,
            vocabulary: geometry.vocabulary,
            scalar_type: geometry.scalar_type,
            block_positions: geometry.block_positions,
            role: stream.role as u64,
            leaf_scratch: stream.owner.leaves.device_ptr_value(),
            leaf_capacity: stream.owner.leaves.len() as u64,
            ..Default::default()
        })
    }

    fn model_canary_stream_inputs(
        &self,
        step: &SemanticPreparedStep,
        stream: &SemanticModelCanaryStream,
    ) -> Result<ModelCanaryStreamInputs, SemanticTransitionError> {
        let storage = self.prepared_publication()?;
        let prepared = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("checked step");
        let update = prepared.branches[stream.bank]
            .model_update
            .as_ref()
            .expect("checked update");
        let view = prepared
            .training_view
            .as_ref()
            .expect("checked training view");
        let (_, task) = self.task.as_ref().ok_or_else(|| {
            publication_input_error("paired canaries require the original native task")
        })?;
        Ok(ModelCanaryStreamInputs {
            control: storage.control.device_ptr_value(),
            lease: prepared.reader.device_ptr_value(),
            selection: *view.selection().device_ptr(),
            roster_rows: *view.roster_rows().device_ptr(),
            objective: *view.objective().device_ptr(),
            canaries: *view.canaries().device_ptr(),
            protected_members: *view.protected_members().device_ptr(),
            protected_member_count: view.protected_members().len() as u64,
            task: task.device_ptr_value(),
            task_words: task.len() as u64,
            retention_labels: *view.retention_labels().device_ptr(),
            kinds: *view.kinds().device_ptr(),
            state: stream.owner.state.device_ptr_value(),
            receipt: stream.owner.receipt.device_ptr_value(),
            baseline_state: stream.owner.forwards[0].owner.state.device_ptr_value(),
            candidate_state: stream.owner.forwards[1].owner.state.device_ptr_value(),
            baseline_row_seals: stream.owner.forwards[0].owner.row_seals.device_ptr_value(),
            candidate_row_seals: stream.owner.forwards[1].owner.row_seals.device_ptr_value(),
            forward_receipts: update.forward_receipts.device_ptr_value(),
            baseline_leaf_scratch: stream.owner.leaves[0].device_ptr_value(),
            candidate_leaf_scratch: stream.owner.leaves[1].device_ptr_value(),
            leaf_capacity: stream.owner.leaves[0].len() as u64,
            position_scratch: stream.owner.positions.device_ptr_value(),
            ..Default::default()
        })
    }

    fn model_forward_stream_publication(
        &self,
        stream: &SemanticModelForwardStream,
    ) -> Result<&Arc<PublicationStorage>, SemanticTransitionError> {
        match &stream.origin {
            ModelForwardStreamOrigin::Update { .. } => self.prepared_publication(),
            ModelForwardStreamOrigin::Evaluation(_) => self
                .publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound),
        }
    }

    fn enqueue_model_forward_stream(
        &mut self,
        stream: &SemanticModelForwardStream,
        operation: usize,
        inputs: ModelForwardStreamInputs,
    ) -> Result<(), SemanticTransitionError> {
        let mut recorder = self.domain.new_strict_recorder();
        self.model_forward_stream_publication(stream)?
            .record(&mut recorder);
        stream.owner.record(&mut recorder)?;
        match &stream.origin {
            ModelForwardStreamOrigin::Update { .. } => {
                let prepared = self.steps[&stream.reader_token]
                    .prepared
                    .as_ref()
                    .expect("checked step");
                recorder.read(&prepared.reader);
                recorder.read(
                    &prepared
                        .training_view
                        .as_ref()
                        .expect("checked training view")
                        .selection(),
                );
                let update = prepared.branches[stream.update_bank()?]
                    .model_update
                    .as_ref()
                    .expect("checked update");
                recorder.read_write(&update.forward_receipts);
                if operation == 2 && stream.role == 1 {
                    let recording = stream
                        .owner
                        .recording
                        .lock()
                        .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
                    let witness = recording
                        .candidate_owner
                        .as_ref()
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    self.steps[&stream.reader_token].content[witness.index]
                        .record_uses(&mut recorder)?;
                }
            }
            ModelForwardStreamOrigin::Evaluation(handle) => {
                self.record_evaluation_model_stream_dependencies(handle, stream, &mut recorder)?;
            }
        }
        let leaf_grid = if operation == 1 {
            let bytes = inputs
                .count
                .checked_mul(stream.owner.geometry.vocabulary)
                .and_then(|values| values.checked_mul(stream.owner.geometry.element_bytes()))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            u32::try_from(bytes.div_ceil(MODEL_STREAM_LEAF_BYTES).div_ceil(128))
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?
        } else {
            1
        };
        let result = enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            // SAFETY: all native buffers and the single genuine producer scratch
            // are retained by the original invocation through its known wait.
            if operation == 1 {
                unsafe {
                    stream.owner.kernels[1].clone().launch_in(
                        enqueue,
                        LaunchConfig {
                            grid_dim: (leaf_grid, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (inputs,),
                    )
                }
                .map_err(|error| XlogError::Kernel(error.to_string()))?;
            }
            let kernel = match operation {
                0 => 0,
                1 => 2,
                2 => 3,
                _ => unreachable!("native stream operation"),
            };
            unsafe {
                stream.owner.kernels[kernel].clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (inputs,),
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        });
        if result.is_err() {
            self.poisoned = true;
        }
        result.map_err(Into::into)
    }

    fn enqueue_model_canary_stream(
        &mut self,
        stream: &SemanticModelCanaryStream,
        operation: usize,
        inputs: ModelCanaryStreamInputs,
    ) -> Result<(), SemanticTransitionError> {
        let mut recorder = self.domain.new_strict_recorder();
        self.prepared_publication()?.record(&mut recorder);
        stream.owner.record(&mut recorder)?;
        let prepared = self.steps[&stream.reader_token]
            .prepared
            .as_ref()
            .expect("checked step");
        recorder.read(&prepared.reader);
        let view = prepared
            .training_view
            .as_ref()
            .expect("checked training view");
        recorder.read(&view.selection());
        recorder.read(&view.roster_rows());
        recorder.read(&view.objective());
        recorder.read(&view.canaries());
        recorder.read(&view.protected_members());
        recorder.read(&view.retention_labels());
        recorder.read(&view.kinds());
        recorder.read(&self.task.as_ref().expect("checked native task").1);
        recorder.read(
            &prepared.branches[stream.bank]
                .model_update
                .as_ref()
                .expect("checked update")
                .forward_receipts,
        );
        let geometry = stream.owner.forwards[0].owner.geometry;
        let leaf_grid = if operation == 1 {
            let bytes = inputs
                .count
                .checked_mul(geometry.vocabulary)
                .and_then(|values| values.checked_mul(geometry.element_bytes()))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            u32::try_from(bytes.div_ceil(MODEL_STREAM_LEAF_BYTES).div_ceil(128))
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?
        } else {
            1
        };
        let position_grid = if operation == 1 {
            u32::try_from(inputs.count.div_ceil(128))
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?
        } else {
            1
        };
        let result = enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            // SAFETY: both full receipts, original selected rows, native task,
            // paired scratch roots and all cursor storage remain original owners.
            if operation == 1 {
                unsafe {
                    stream.owner.kernels[1].clone().launch_in(
                        enqueue,
                        LaunchConfig {
                            grid_dim: (leaf_grid, 2, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (inputs,),
                    )
                }
                .map_err(|error| XlogError::Kernel(error.to_string()))?;
                unsafe {
                    stream.owner.kernels[2].clone().launch_in(
                        enqueue,
                        LaunchConfig {
                            grid_dim: (position_grid, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (inputs,),
                    )
                }
                .map_err(|error| XlogError::Kernel(error.to_string()))?;
            }
            let kernel = match operation {
                0 => 0,
                1 => 3,
                2 => 4,
                _ => unreachable!("native canary stream operation"),
            };
            unsafe {
                stream.owner.kernels[kernel].clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (inputs,),
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        });
        if result.is_err() {
            self.poisoned = true;
        }
        result.map_err(Into::into)
    }
}
