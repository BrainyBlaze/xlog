use std::mem::size_of;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::launch::LaunchEnqueueError;
use crate::memory::{DeviceMemoryView, TrackedCudaSlice};
use crate::provider::resident_schedule::{validate_execution_domain, ResidentExecutionDomain};
use crate::semantic_transition::{
    Identity256, SemanticPublishedIdentity, SemanticRngBinding, SemanticTransitionError,
    SemanticTransitionKind,
};
use crate::{CudaFunction, CudaKernelProvider, DeviceRepr, LaunchAsync, LaunchConfig};

const MODULE: &str = "xlog_semantic_training_view";
const SELECT_KERNEL: &str = "semantic_training_view_select";
const GATHER_KERNEL: &str = "semantic_training_view_gather";
const TRAINING_VIEW_HEADER_BYTES: usize = 136;
const TRAINING_VIEW_ROW_BYTES: usize = 68;

/// Origin of one authentic replay training view.
#[repr(u64)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticTrainingViewBasis {
    Episode = 1,
    CorpusAnchor = 2,
}

/// Authenticated native execution that produced one episode training view.
/// Corpus anchors have no execution origin and use their independent graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticTrainingViewOrigin {
    pub transition: SemanticTransitionKind,
    pub predecessor: SemanticPublishedIdentity,
    pub successor: SemanticPublishedIdentity,
    pub invocation: SemanticRngBinding,
    pub model_geometry_digest: Identity256,
    pub model_numerical_digest: Identity256,
}

/// One already-admitted training view retained for cold device selection.
pub struct SemanticTrainingViewRow {
    pub basis: SemanticTrainingViewBasis,
    pub content_identity: Identity256,
    pub origin: Option<SemanticTrainingViewOrigin>,
    pub bytes: Vec<u8>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SemanticTrainingViewOriginRecord {
    pub present: u64,
    pub transition: u64,
    pub predecessor_instance: [u64; 4],
    pub predecessor_word: u64,
    pub predecessor_logical: [u64; 4],
    pub predecessor_state: [u64; 4],
    pub successor_instance: [u64; 4],
    pub successor_word: u64,
    pub successor_logical: [u64; 4],
    pub successor_state: [u64; 4],
    pub model_generation: u64,
    pub stream_serial: u64,
    pub family_id: u64,
    pub proposal: u64,
    pub model_geometry_digest: [u64; 4],
    pub model_numerical_digest: [u64; 4],
}

// SAFETY: the fixed CUDA ABI contains only u64 words.
unsafe impl DeviceRepr for SemanticTrainingViewOriginRecord {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TrainingViewRowDescriptor {
    ordinal: u64,
    basis: u64,
    raw_offset: u64,
    raw_bytes: u64,
    window: u64,
    source_length: u64,
    block_size: u64,
    prefix_extent: u64,
    answer_start: u64,
    identity: [u64; 4],
    source_identity: [u64; 4],
    content_identity: [u64; 4],
    origin: SemanticTrainingViewOriginRecord,
}

// SAFETY: the fixed CUDA ABI contains only u64 words.
unsafe impl DeviceRepr for TrainingViewRowDescriptor {}

/// Device-written selection coordinates. Status zero is the only admissible
/// result; downstream Update work must predicate on this resident word.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SemanticTrainingViewSelection {
    pub status: u64,
    pub ordinal: u64,
    pub basis: u64,
    pub window: u64,
    pub source_length: u64,
    pub block_size: u64,
    pub prefix_extent: u64,
    pub answer_start: u64,
    pub identity: [u64; 4],
    pub source_identity: [u64; 4],
    pub content_identity: [u64; 4],
    pub origin: SemanticTrainingViewOriginRecord,
    pub training_rng: [u64; 4],
}

// SAFETY: the fixed CUDA ABI contains only u64 words.
unsafe impl DeviceRepr for SemanticTrainingViewSelection {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TrainingViewLaunch {
    descriptors: u64,
    raw: u64,
    row_count: u64,
    selected_view: u64,
    selected_view_bytes: u64,
    cursor: u64,
    training_rng: [u64; 4],
    selection: u64,
    capacity: u64,
    token_ids: u64,
    mask_labels: u64,
    mask_weights: u64,
    ar_labels: u64,
    retention_labels: u64,
    source_slots: u64,
    logical_positions: u64,
    kinds: u64,
    parents: u64,
}

// SAFETY: the fixed CUDA ABI contains only u64 words.
unsafe impl DeviceRepr for TrainingViewLaunch {}

struct TrainingViewLaunchParam(TrainingViewLaunch);

impl crate::cuda_compat::KernelParamStorage for TrainingViewLaunchParam {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.0 as *const TrainingViewLaunch).cast_mut().cast()
    }
}

impl crate::cuda_compat::IntoKernelParamStorage for TrainingViewLaunch {
    type Storage = TrainingViewLaunchParam;

    fn into_kernel_param_storage(self) -> Self::Storage {
        TrainingViewLaunchParam(self)
    }
}

struct SelectedTrainingViewStorage {
    selection: TrackedCudaSlice<SemanticTrainingViewSelection>,
    token_ids: TrackedCudaSlice<i64>,
    mask_labels: TrackedCudaSlice<i64>,
    mask_weights: TrackedCudaSlice<f32>,
    ar_labels: TrackedCudaSlice<i64>,
    retention_labels: TrackedCudaSlice<i64>,
    source_slots: TrackedCudaSlice<i64>,
    logical_positions: TrackedCudaSlice<i64>,
    kinds: TrackedCudaSlice<i64>,
    parents: TrackedCudaSlice<i64>,
}

/// Retained native result of device selection. The views are fixed-capacity
/// CUDA ports; the resident selection record supplies their logical extent.
pub struct SemanticSelectedTrainingView {
    arena: Arc<SemanticTrainingViewArena>,
    storage: SelectedTrainingViewStorage,
}

impl SemanticSelectedTrainingView {
    pub fn capacity(&self) -> usize {
        self.arena.capacity
    }

    pub fn selection(&self) -> DeviceMemoryView<SemanticTrainingViewSelection> {
        self.storage.selection.view()
    }

    pub fn token_ids(&self) -> DeviceMemoryView<i64> {
        self.storage.token_ids.view()
    }

    pub fn mask_labels(&self) -> DeviceMemoryView<i64> {
        self.storage.mask_labels.view()
    }

    pub fn mask_weights(&self) -> DeviceMemoryView<f32> {
        self.storage.mask_weights.view()
    }

    pub fn ar_labels(&self) -> DeviceMemoryView<i64> {
        self.storage.ar_labels.view()
    }

    pub fn retention_labels(&self) -> DeviceMemoryView<i64> {
        self.storage.retention_labels.view()
    }

    pub fn source_slots(&self) -> DeviceMemoryView<i64> {
        self.storage.source_slots.view()
    }

    pub fn logical_positions(&self) -> DeviceMemoryView<i64> {
        self.storage.logical_positions.view()
    }

    pub fn kinds(&self) -> DeviceMemoryView<i64> {
        self.storage.kinds.view()
    }

    pub fn parents(&self) -> DeviceMemoryView<i64> {
        self.storage.parents.view()
    }
}

/// Immutable cold roster used by the native Update selector.
pub(crate) struct SemanticTrainingViewArena {
    provider: Arc<CudaKernelProvider>,
    domain: ResidentExecutionDomain,
    select: CudaFunction,
    gather: CudaFunction,
    descriptors: TrackedCudaSlice<TrainingViewRowDescriptor>,
    raw: TrackedCudaSlice<u8>,
    row_count: usize,
    capacity: usize,
}

impl SemanticTrainingViewArena {
    pub(crate) fn allocate(
        provider: &Arc<CudaKernelProvider>,
        domain: &ResidentExecutionDomain,
        rows: Vec<SemanticTrainingViewRow>,
    ) -> Result<Arc<Self>, SemanticTransitionError> {
        validate_execution_domain(provider, domain)
            .map_err(|error| runtime_error("training-view domain validation", error))?;
        if rows.is_empty() {
            return Err(input_error(
                "training-view arena requires at least one admitted row",
            ));
        }
        let select = provider
            .device()
            .inner()
            .get_func(MODULE, SELECT_KERNEL)
            .ok_or_else(|| runtime_error("training-view kernel lookup", "selector unavailable"))?;
        let gather = provider
            .device()
            .inner()
            .get_func(MODULE, GATHER_KERNEL)
            .ok_or_else(|| runtime_error("training-view kernel lookup", "gather unavailable"))?;
        let mut raw = Vec::new();
        let mut descriptors = Vec::with_capacity(rows.len());
        let mut capacity = 0usize;
        for (ordinal, row) in rows.into_iter().enumerate() {
            let descriptor = validate_row(
                ordinal,
                row.basis,
                row.content_identity,
                row.origin,
                &row.bytes,
                raw.len(),
            )?;
            capacity = capacity.max(descriptor.window as usize);
            raw.extend_from_slice(&row.bytes);
            descriptors.push(descriptor);
        }
        let bytes = descriptors
            .len()
            .checked_mul(size_of::<TrainingViewRowDescriptor>())
            .and_then(|bytes| bytes.checked_add(raw.len()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut reservation = provider
            .memory()
            .reserve_bytes(
                u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            )
            .map_err(|error| runtime_error("training-view reservation", error))?;
        let mut device_descriptors = reservation
            .alloc::<TrainingViewRowDescriptor>(descriptors.len())
            .map_err(|error| runtime_error("training-view descriptor allocation", error))?;
        let mut device_raw = reservation
            .alloc::<u8>(raw.len())
            .map_err(|error| runtime_error("training-view byte allocation", error))?;
        if reservation.remaining_bytes() != 0 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        provider
            .htod_sync_copy_into_tracked(&descriptors, &mut device_descriptors)
            .map_err(|error| runtime_error("training-view descriptor upload", error))?;
        provider
            .htod_sync_copy_into_tracked(&raw, &mut device_raw)
            .map_err(|error| runtime_error("training-view byte upload", error))?;
        Ok(Arc::new(Self {
            provider: Arc::clone(provider),
            domain: domain.clone(),
            select,
            gather,
            descriptors: device_descriptors,
            raw: device_raw,
            row_count: descriptors.len(),
            capacity,
        }))
    }

    pub(crate) fn enqueue_selection(
        self: &Arc<Self>,
        selected_view: DeviceMemoryView<u8>,
        cursor: u64,
        training_rng: [u64; 4],
    ) -> Result<SemanticSelectedTrainingView, SemanticTransitionError> {
        let output_bytes = size_of::<SemanticTrainingViewSelection>()
            .checked_add(
                self.capacity
                    .checked_mul(TRAINING_VIEW_ROW_BYTES)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?,
            )
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut reservation = self
            .provider
            .memory()
            .reserve_bytes(
                u64::try_from(output_bytes)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            )
            .map_err(|error| runtime_error("selected training-view reservation", error))?;
        let storage = SelectedTrainingViewStorage {
            selection: reservation
                .alloc::<SemanticTrainingViewSelection>(1)
                .map_err(|error| runtime_error("training-view selection allocation", error))?,
            token_ids: allocate_port(&mut reservation, self.capacity)?,
            mask_labels: allocate_port(&mut reservation, self.capacity)?,
            mask_weights: allocate_port(&mut reservation, self.capacity)?,
            ar_labels: allocate_port(&mut reservation, self.capacity)?,
            retention_labels: allocate_port(&mut reservation, self.capacity)?,
            source_slots: allocate_port(&mut reservation, self.capacity)?,
            logical_positions: allocate_port(&mut reservation, self.capacity)?,
            kinds: allocate_port(&mut reservation, self.capacity)?,
            parents: allocate_port(&mut reservation, self.capacity)?,
        };
        if reservation.remaining_bytes() != 0 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let launch = TrainingViewLaunch {
            descriptors: self.descriptors.device_ptr_value(),
            raw: self.raw.device_ptr_value(),
            row_count: u64::try_from(self.row_count)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            selected_view: *selected_view.device_ptr(),
            selected_view_bytes: u64::try_from(selected_view.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            cursor,
            training_rng,
            selection: storage.selection.device_ptr_value(),
            capacity: u64::try_from(self.capacity)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            token_ids: storage.token_ids.device_ptr_value(),
            mask_labels: storage.mask_labels.device_ptr_value(),
            mask_weights: storage.mask_weights.device_ptr_value(),
            ar_labels: storage.ar_labels.device_ptr_value(),
            retention_labels: storage.retention_labels.device_ptr_value(),
            source_slots: storage.source_slots.device_ptr_value(),
            logical_positions: storage.logical_positions.device_ptr_value(),
            kinds: storage.kinds.device_ptr_value(),
            parents: storage.parents.device_ptr_value(),
        };
        let mut recorder = self.domain.new_strict_recorder();
        recorder.read(&self.descriptors);
        recorder.read(&self.raw);
        recorder.read(&selected_view);
        recorder.write(&storage.selection);
        recorder.write(&storage.token_ids);
        recorder.write(&storage.mask_labels);
        recorder.write(&storage.mask_weights);
        recorder.write(&storage.ar_labels);
        recorder.write(&storage.retention_labels);
        recorder.write(&storage.source_slots);
        recorder.write(&storage.logical_positions);
        recorder.write(&storage.kinds);
        recorder.write(&storage.parents);
        let select = self.select.clone();
        let gather = self.gather.clone();
        let enqueued = unsafe {
            self.domain.enqueue(recorder, |stream| {
                select.clone().launch_in(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (launch,),
                )?;
                gather.clone().launch_in(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (launch,),
                )
            })
        }
        .map_err(map_enqueue_error)?;
        enqueued
            .commit()
            .map_err(|error| runtime_error("training-view launch commit", error))?;
        Ok(SemanticSelectedTrainingView {
            arena: Arc::clone(self),
            storage,
        })
    }
}

fn allocate_port<T: DeviceRepr>(
    reservation: &mut crate::memory::GpuMemoryReservation,
    capacity: usize,
) -> Result<TrackedCudaSlice<T>, SemanticTransitionError> {
    reservation
        .alloc::<T>(capacity)
        .map_err(|error| runtime_error("selected training-view port allocation", error))
}

fn validate_row(
    ordinal: usize,
    basis: SemanticTrainingViewBasis,
    content_identity: Identity256,
    origin: Option<SemanticTrainingViewOrigin>,
    bytes: &[u8],
    raw_offset: usize,
) -> Result<TrainingViewRowDescriptor, SemanticTransitionError> {
    let mut schema = [0u8; 32];
    let tag = b"dlm-new/training-view/v1";
    schema[..tag.len()].copy_from_slice(tag);
    if bytes.len() < TRAINING_VIEW_HEADER_BYTES || bytes[..32] != schema {
        return Err(input_error("training-view row has another schema"));
    }
    let word = |offset: usize| {
        u64::from_le_bytes(
            bytes[offset..offset + 8]
                .try_into()
                .expect("bounded header"),
        )
    };
    let window = usize::try_from(word(96))
        .map_err(|_| input_error("training-view window exceeds host address space"))?;
    let expected = window
        .checked_mul(TRAINING_VIEW_ROW_BYTES)
        .and_then(|length| length.checked_add(TRAINING_VIEW_HEADER_BYTES + (window % 2) * 4));
    if window == 0
        || word(104) == 0
        || word(112) == 0
        || word(120) > word(104)
        || word(128) == 0
        || word(128) > word(104)
        || expected != Some(bytes.len())
        || raw_offset % 8 != 0
    {
        return Err(input_error(
            "training-view row has invalid extent or geometry",
        ));
    }
    let padding = TRAINING_VIEW_HEADER_BYTES + window * 20;
    if bytes[padding..padding + (window % 2) * 4]
        .iter()
        .any(|&byte| byte != 0)
    {
        return Err(input_error(
            "training-view row has noncanonical alignment padding",
        ));
    }
    let identity: [u8; 32] = bytes[32..64].try_into().expect("bounded identity");
    let source_identity: [u8; 32] = bytes[64..96].try_into().expect("bounded identity");
    if matches!(basis, SemanticTrainingViewBasis::Episode) != origin.is_some() {
        return Err(input_error(
            "only an episode training view has an authentic execution origin",
        ));
    }
    if Sha256::digest(bytes).as_slice() != identity
        || source_identity == [0; 32]
        || content_identity == Identity256::default()
    {
        return Err(input_error(
            "training-view row identity differs from its bytes, source or replay content",
        ));
    }
    Ok(TrainingViewRowDescriptor {
        ordinal: u64::try_from(ordinal)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        basis: basis as u64,
        raw_offset: u64::try_from(raw_offset)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        raw_bytes: u64::try_from(bytes.len())
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        window: u64::try_from(window).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        source_length: word(104),
        block_size: word(112),
        prefix_extent: word(120),
        answer_start: word(128),
        identity: identity_words(Identity256::from_bytes(identity)),
        source_identity: identity_words(Identity256::from_bytes(source_identity)),
        content_identity: identity_words(content_identity),
        origin: origin.map(origin_record).unwrap_or_default(),
    })
}

fn origin_record(origin: SemanticTrainingViewOrigin) -> SemanticTrainingViewOriginRecord {
    SemanticTrainingViewOriginRecord {
        present: 1,
        transition: match origin.transition {
            SemanticTransitionKind::Proposal => 1,
            SemanticTransitionKind::Recompute => 2,
            SemanticTransitionKind::Drain => 3,
        },
        predecessor_instance: identity_words(origin.predecessor.instance),
        predecessor_word: origin.predecessor.word,
        predecessor_logical: identity_words(origin.predecessor.logical_digest),
        predecessor_state: identity_words(origin.predecessor.state_digest),
        successor_instance: identity_words(origin.successor.instance),
        successor_word: origin.successor.word,
        successor_logical: identity_words(origin.successor.logical_digest),
        successor_state: identity_words(origin.successor.state_digest),
        model_generation: u64::from(origin.invocation.model_generation),
        stream_serial: origin.invocation.stream_serial,
        family_id: u64::from(origin.invocation.family_id),
        proposal: u64::from(origin.invocation.proposal),
        model_geometry_digest: identity_words(origin.model_geometry_digest),
        model_numerical_digest: identity_words(origin.model_numerical_digest),
    }
}

fn identity_words(identity: Identity256) -> [u64; 4] {
    std::array::from_fn(|index| {
        let begin = index * 8;
        u64::from_le_bytes(
            identity.as_bytes()[begin..begin + 8]
                .try_into()
                .expect("identity word"),
        )
    })
}

fn input_error(detail: impl Into<String>) -> SemanticTransitionError {
    SemanticTransitionError::InvalidInput {
        detail: detail.into(),
    }
}

fn runtime_error(
    operation: &'static str,
    error: impl std::fmt::Display,
) -> SemanticTransitionError {
    SemanticTransitionError::Runtime {
        operation,
        detail: error.to_string(),
    }
}

fn map_enqueue_error<E: std::fmt::Display>(
    error: LaunchEnqueueError<E>,
) -> SemanticTransitionError {
    runtime_error("training-view device selection", error)
}
