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

type TrainingViewPort = (DeviceMemoryView<u8>, Vec<i64>, Vec<i64>, (u8, u8));
type ValidatedTrainingObjective = (
    SemanticTrainingObjectiveRecord,
    Vec<SemanticTrainingObjectiveGroupRecord>,
    Vec<u64>,
    Vec<SemanticTrainingCanaryRecord>,
);

const MODULE: &str = "xlog_semantic_training_view";
const SELECT_KERNEL: &str = "semantic_training_view_select";
const GATHER_KERNEL: &str = "semantic_training_view_gather";
const TRAINING_VIEW_HEADER_BYTES: usize = 136;
const TRAINING_VIEW_ROW_BYTES: usize = 68;
const PROPOSAL_TRANSITION: u64 = 1;

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

/// Frozen evaluator term attached to the complete device-resident training roster.
#[repr(u64)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticTrainingObjectiveGroupKind {
    MaskedLanguage = 1,
    AutoregressiveLanguage = 2,
    Semantic = 3,
    Edit = 4,
    Execution = 5,
    RetentionLanguage = 6,
    RetentionSymbolic = 7,
    ActorCriticCost = 8,
}

/// One nonempty reduction group in the frozen training objective.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticTrainingObjectiveGroup {
    pub kind: SemanticTrainingObjectiveGroupKind,
    pub denominator: u64,
    pub row_ordinals: Vec<u64>,
}

/// Mandatory acceptance gate evaluated after a candidate update.
#[repr(u64)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticTrainingCanaryKind {
    SymbolicUtility = 1,
    RetainedBehavior = 2,
    LogitDrift = 3,
    GoalChain = 4,
    ResourceLimits = 5,
}

/// Frozen bounds and resource ceilings for one candidate-update canary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SemanticTrainingCanary {
    pub kind: SemanticTrainingCanaryKind,
    pub row_ordinal: u64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub memory_limit: u64,
    pub fuel_limit: u64,
    pub identity: Identity256,
}

/// Complete frozen objective carried by the canonical replay-roster owner.
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticTrainingObjective {
    pub identity: Identity256,
    pub evaluator_min: f64,
    pub evaluator_max: f64,
    /// Masked-language, autoregressive, semantic, edit, execution, retention,
    /// actor, critic and cost coefficients, in that order.
    pub coefficients: [f32; 9],
    pub cost_unit: Identity256,
    pub cost_cap: u64,
    pub groups: Vec<SemanticTrainingObjectiveGroup>,
    pub canaries: Vec<SemanticTrainingCanary>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SemanticTrainingObjectiveRecord {
    pub identity: [u64; 4],
    pub row_count: u64,
    pub capacity: u64,
    pub group_count: u64,
    pub group_member_count: u64,
    pub canary_count: u64,
    pub evaluator_min_bits: u64,
    pub evaluator_max_bits: u64,
    pub coefficient_bits: [u64; 9],
    pub cost_unit: [u64; 4],
    pub cost_cap: u64,
}

// SAFETY: the fixed CUDA ABI contains only u64 words.
unsafe impl DeviceRepr for SemanticTrainingObjectiveRecord {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SemanticTrainingObjectiveGroupRecord {
    pub kind: u64,
    pub denominator: u64,
    pub member_offset: u64,
    pub member_count: u64,
}

// SAFETY: the fixed CUDA ABI contains only u64 words.
unsafe impl DeviceRepr for SemanticTrainingObjectiveGroupRecord {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SemanticTrainingCanaryRecord {
    pub kind: u64,
    pub row_ordinal: u64,
    pub lower_bound_bits: u64,
    pub upper_bound_bits: u64,
    pub memory_limit: u64,
    pub fuel_limit: u64,
    pub identity: [u64; 4],
}

// SAFETY: the fixed CUDA ABI contains only u64 words.
unsafe impl DeviceRepr for SemanticTrainingCanaryRecord {}

/// Device-produced measurement for one frozen candidate-update canary.
///
/// The result repeats the frozen kind, row and identity so the native update
/// gate can reject a measurement produced for any other roster entry. The
/// measurement is carried as raw FP64 bits; memory and fuel are exact integer
/// tallies checked against the frozen ceilings.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SemanticTrainingCanaryResultRecord {
    pub kind: u64,
    pub row_ordinal: u64,
    pub measurement_bits: u64,
    pub memory_used: u64,
    pub fuel_used: u64,
    pub identity: [u64; 4],
}

// SAFETY: the fixed CUDA ABI contains only u64 words.
unsafe impl DeviceRepr for SemanticTrainingCanaryResultRecord {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SemanticTrainingViewOriginRecord {
    pub present: u64,
    pub transition: u64,
    /// Stable historical predecessor identity. Historical rows use their
    /// original instance; a fresh replay candidate derives it from the restored
    /// predecessor's recovered-instance seal.
    pub lineage_instance: [u64; 4],
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

/// Device-written metadata for every row in the selected objective roster.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SemanticTrainingRosterRow {
    pub ordinal: u64,
    pub basis: u64,
    pub window: u64,
    pub source_length: u64,
    pub block_size: u64,
    pub prefix_extent: u64,
    pub answer_start: u64,
    pub origin_candidate: u64,
    pub identity: [u64; 4],
    pub source_identity: [u64; 4],
    pub content_identity: [u64; 4],
    /// Authenticated historical execution origin for episode rows. The separate
    /// candidate ordinal, when present, names a successful fresh re-execution.
    pub origin: SemanticTrainingViewOriginRecord,
}

// SAFETY: the fixed CUDA ABI contains only u64 words.
unsafe impl DeviceRepr for SemanticTrainingRosterRow {}

/// Device-written selection coordinates. Status zero is the only admissible
/// result; downstream Update work must predicate on this resident word.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SemanticTrainingViewSelection {
    pub status: u64,
    pub row_count: u64,
    pub capacity: u64,
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
    pub origin_candidate: u64,
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
    coordinates: u64,
    origin_candidates: u64,
    origin_candidate_count: u64,
    selection: u64,
    roster_rows: u64,
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
    roster_rows: TrackedCudaSlice<SemanticTrainingRosterRow>,
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
    _origin_candidates: Option<Arc<TrackedCudaSlice<SemanticTrainingViewOriginRecord>>>,
    storage: SelectedTrainingViewStorage,
}

/// Fixed device port of the native-selected complete training roster.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticTrainingViewPort {
    Selection,
    RosterRows,
    Objective,
    ObjectiveGroups,
    ObjectiveGroupMembers,
    Canaries,
    TokenIds,
    MaskLabels,
    MaskWeights,
    AutoregressiveLabels,
    RetentionLabels,
    SourceSlots,
    LogicalPositions,
    Kinds,
    Parents,
}

impl SemanticSelectedTrainingView {
    pub fn capacity(&self) -> usize {
        self.arena.capacity
    }

    pub fn row_count(&self) -> usize {
        self.arena.row_count
    }

    pub fn selection(&self) -> DeviceMemoryView<SemanticTrainingViewSelection> {
        self.storage.selection.view()
    }

    pub fn roster_rows(&self) -> DeviceMemoryView<SemanticTrainingRosterRow> {
        self.storage.roster_rows.view()
    }

    pub(crate) fn objective(&self) -> DeviceMemoryView<SemanticTrainingObjectiveRecord> {
        self.arena.objective.view()
    }

    pub(crate) fn objective_groups(
        &self,
    ) -> DeviceMemoryView<SemanticTrainingObjectiveGroupRecord> {
        self.arena.groups.view()
    }

    pub(crate) fn objective_group_members(&self) -> DeviceMemoryView<u64> {
        self.arena.group_members.view()
    }

    pub(crate) fn canaries(&self) -> DeviceMemoryView<SemanticTrainingCanaryRecord> {
        self.arena.canaries.view()
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

    pub(crate) fn port(
        &self,
        port: SemanticTrainingViewPort,
    ) -> Result<TrainingViewPort, SemanticTransitionError> {
        let words = |bytes: usize| {
            i64::try_from(bytes / size_of::<u64>())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)
        };
        let rows = i64::try_from(self.row_count())
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let capacity = i64::try_from(self.capacity())
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let extent = |length: usize| {
            i64::try_from(length).map_err(|_| SemanticTransitionError::GenerationExhausted)
        };
        let matrix = |view, dtype| (view, vec![rows, capacity], vec![capacity, 1], dtype);
        let result = match port {
            SemanticTrainingViewPort::Selection => (
                unsafe { self.storage.selection.view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                vec![words(size_of::<SemanticTrainingViewSelection>())?],
                vec![1],
                (1, 64),
            ),
            SemanticTrainingViewPort::RosterRows => (
                unsafe { self.storage.roster_rows.view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                vec![rows, words(size_of::<SemanticTrainingRosterRow>())?],
                vec![words(size_of::<SemanticTrainingRosterRow>())?, 1],
                (1, 64),
            ),
            SemanticTrainingViewPort::Objective => (
                unsafe { self.arena.objective.view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                vec![words(size_of::<SemanticTrainingObjectiveRecord>())?],
                vec![1],
                (1, 64),
            ),
            SemanticTrainingViewPort::ObjectiveGroups => {
                let width = words(size_of::<SemanticTrainingObjectiveGroupRecord>())?;
                (
                    unsafe { self.arena.groups.view().cast::<u8>() }
                        .ok_or(SemanticTransitionError::ObservationMismatch)?,
                    vec![extent(self.arena.groups.len())?, width],
                    vec![width, 1],
                    (1, 64),
                )
            }
            SemanticTrainingViewPort::ObjectiveGroupMembers => (
                unsafe { self.arena.group_members.view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                vec![extent(self.arena.group_members.len())?],
                vec![1],
                (1, 64),
            ),
            SemanticTrainingViewPort::Canaries => {
                let width = words(size_of::<SemanticTrainingCanaryRecord>())?;
                (
                    unsafe { self.arena.canaries.view().cast::<u8>() }
                        .ok_or(SemanticTransitionError::ObservationMismatch)?,
                    vec![extent(self.arena.canaries.len())?, width],
                    vec![width, 1],
                    (1, 64),
                )
            }
            SemanticTrainingViewPort::TokenIds => (
                unsafe { self.storage.token_ids.view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                vec![],
                vec![],
                (0, 64),
            ),
            SemanticTrainingViewPort::MaskLabels => (
                unsafe { self.storage.mask_labels.view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                vec![],
                vec![],
                (0, 64),
            ),
            SemanticTrainingViewPort::MaskWeights => (
                unsafe { self.storage.mask_weights.view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                vec![],
                vec![],
                (2, 32),
            ),
            SemanticTrainingViewPort::AutoregressiveLabels => (
                unsafe { self.storage.ar_labels.view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                vec![],
                vec![],
                (0, 64),
            ),
            SemanticTrainingViewPort::RetentionLabels => (
                unsafe { self.storage.retention_labels.view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                vec![],
                vec![],
                (0, 64),
            ),
            SemanticTrainingViewPort::SourceSlots => (
                unsafe { self.storage.source_slots.view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                vec![],
                vec![],
                (0, 64),
            ),
            SemanticTrainingViewPort::LogicalPositions => (
                unsafe { self.storage.logical_positions.view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                vec![],
                vec![],
                (0, 64),
            ),
            SemanticTrainingViewPort::Kinds => (
                unsafe { self.storage.kinds.view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                vec![],
                vec![],
                (0, 64),
            ),
            SemanticTrainingViewPort::Parents => (
                unsafe { self.storage.parents.view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                vec![],
                vec![],
                (0, 64),
            ),
        };
        Ok(match port {
            SemanticTrainingViewPort::TokenIds
            | SemanticTrainingViewPort::MaskLabels
            | SemanticTrainingViewPort::MaskWeights
            | SemanticTrainingViewPort::AutoregressiveLabels
            | SemanticTrainingViewPort::RetentionLabels
            | SemanticTrainingViewPort::SourceSlots
            | SemanticTrainingViewPort::LogicalPositions
            | SemanticTrainingViewPort::Kinds
            | SemanticTrainingViewPort::Parents => matrix(result.0, result.3),
            _ => result,
        })
    }

    pub(crate) fn enqueue_device_selection(
        &self,
        selected_view: DeviceMemoryView<u8>,
        coordinates: DeviceMemoryView<u64>,
    ) -> Result<(), SemanticTransitionError> {
        if coordinates.len() != 5 {
            return Err(input_error(
                "prepared training-view coordinates require cursor and four RNG words",
            ));
        }
        self.enqueue(selected_view, 0, [0; 4], Some(coordinates))
    }

    fn enqueue(
        &self,
        selected_view: DeviceMemoryView<u8>,
        cursor: u64,
        training_rng: [u64; 4],
        coordinates: Option<DeviceMemoryView<u64>>,
    ) -> Result<(), SemanticTransitionError> {
        let (origin_candidate_ptr, origin_candidate_count) =
            if let Some(candidates) = self._origin_candidates.as_ref() {
                (
                    candidates.device_ptr_value(),
                    u64::try_from(candidates.len())
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                )
            } else {
                (0, 0)
            };
        let launch = TrainingViewLaunch {
            descriptors: self.arena.descriptors.device_ptr_value(),
            raw: self.arena.raw.device_ptr_value(),
            row_count: u64::try_from(self.arena.row_count)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            selected_view: *selected_view.device_ptr(),
            selected_view_bytes: u64::try_from(selected_view.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            cursor,
            training_rng,
            coordinates: coordinates.as_ref().map_or(0, |view| *view.device_ptr()),
            origin_candidates: origin_candidate_ptr,
            origin_candidate_count,
            selection: self.storage.selection.device_ptr_value(),
            roster_rows: self.storage.roster_rows.device_ptr_value(),
            capacity: u64::try_from(self.arena.capacity)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            token_ids: self.storage.token_ids.device_ptr_value(),
            mask_labels: self.storage.mask_labels.device_ptr_value(),
            mask_weights: self.storage.mask_weights.device_ptr_value(),
            ar_labels: self.storage.ar_labels.device_ptr_value(),
            retention_labels: self.storage.retention_labels.device_ptr_value(),
            source_slots: self.storage.source_slots.device_ptr_value(),
            logical_positions: self.storage.logical_positions.device_ptr_value(),
            kinds: self.storage.kinds.device_ptr_value(),
            parents: self.storage.parents.device_ptr_value(),
        };
        let mut recorder = self.arena.domain.new_strict_recorder();
        recorder.read(&self.arena.descriptors);
        recorder.read(&self.arena.raw);
        recorder.read(&selected_view);
        if let Some(candidates) = &self._origin_candidates {
            recorder.read(candidates.as_ref());
        }
        if let Some(coordinates) = &coordinates {
            recorder.read(coordinates);
        }
        recorder.write(&self.storage.selection);
        recorder.write(&self.storage.roster_rows);
        recorder.write(&self.storage.token_ids);
        recorder.write(&self.storage.mask_labels);
        recorder.write(&self.storage.mask_weights);
        recorder.write(&self.storage.ar_labels);
        recorder.write(&self.storage.retention_labels);
        recorder.write(&self.storage.source_slots);
        recorder.write(&self.storage.logical_positions);
        recorder.write(&self.storage.kinds);
        recorder.write(&self.storage.parents);
        let select = self.arena.select.clone();
        let gather = self.arena.gather.clone();
        let gather_grid = u32::try_from(self.arena.row_count)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let enqueued = unsafe {
            self.arena.domain.enqueue(recorder, |stream| {
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
                        grid_dim: (gather_grid, 1, 1),
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
            .map_err(|error| runtime_error("training-view launch commit", error))
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
    objective: TrackedCudaSlice<SemanticTrainingObjectiveRecord>,
    groups: TrackedCudaSlice<SemanticTrainingObjectiveGroupRecord>,
    group_members: TrackedCudaSlice<u64>,
    canaries: TrackedCudaSlice<SemanticTrainingCanaryRecord>,
    row_count: usize,
    capacity: usize,
}

impl SemanticTrainingViewArena {
    pub(crate) fn selection_bytes(&self) -> Result<usize, SemanticTransitionError> {
        let roster_bytes = self
            .row_count
            .checked_mul(size_of::<SemanticTrainingRosterRow>())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let port_bytes = self
            .capacity
            .checked_mul(self.row_count)
            .and_then(|cells| cells.checked_mul(TRAINING_VIEW_ROW_BYTES))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        size_of::<SemanticTrainingViewSelection>()
            .checked_add(roster_bytes)
            .and_then(|bytes| bytes.checked_add(port_bytes))
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }

    pub(crate) fn allocate_selection_reserved(
        self: &Arc<Self>,
        reservation: &mut crate::memory::GpuMemoryReservation,
        origin_candidates: Option<Arc<TrackedCudaSlice<SemanticTrainingViewOriginRecord>>>,
    ) -> Result<SemanticSelectedTrainingView, SemanticTransitionError> {
        let storage = SelectedTrainingViewStorage {
            selection: reservation
                .alloc::<SemanticTrainingViewSelection>(1)
                .map_err(|error| runtime_error("training-view selection allocation", error))?,
            roster_rows: reservation
                .alloc::<SemanticTrainingRosterRow>(self.row_count)
                .map_err(|error| runtime_error("training-view roster allocation", error))?,
            token_ids: allocate_port(reservation, self.row_count, self.capacity)?,
            mask_labels: allocate_port(reservation, self.row_count, self.capacity)?,
            mask_weights: allocate_port(reservation, self.row_count, self.capacity)?,
            ar_labels: allocate_port(reservation, self.row_count, self.capacity)?,
            retention_labels: allocate_port(reservation, self.row_count, self.capacity)?,
            source_slots: allocate_port(reservation, self.row_count, self.capacity)?,
            logical_positions: allocate_port(reservation, self.row_count, self.capacity)?,
            kinds: allocate_port(reservation, self.row_count, self.capacity)?,
            parents: allocate_port(reservation, self.row_count, self.capacity)?,
        };
        Ok(SemanticSelectedTrainingView {
            arena: Arc::clone(self),
            _origin_candidates: origin_candidates,
            storage,
        })
    }

    pub(crate) fn allocate(
        provider: &Arc<CudaKernelProvider>,
        domain: &ResidentExecutionDomain,
        rows: Vec<SemanticTrainingViewRow>,
        objective: SemanticTrainingObjective,
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
        let (objective, groups, group_members, canaries) =
            validate_objective(objective, &descriptors, capacity)?;
        let bytes = descriptors
            .len()
            .checked_mul(size_of::<TrainingViewRowDescriptor>())
            .and_then(|bytes| bytes.checked_add(raw.len()))
            .and_then(|bytes| bytes.checked_add(size_of::<SemanticTrainingObjectiveRecord>()))
            .and_then(|bytes| {
                groups
                    .len()
                    .checked_mul(size_of::<SemanticTrainingObjectiveGroupRecord>())
                    .and_then(|group_bytes| bytes.checked_add(group_bytes))
            })
            .and_then(|bytes| {
                group_members
                    .len()
                    .checked_mul(size_of::<u64>())
                    .and_then(|member_bytes| bytes.checked_add(member_bytes))
            })
            .and_then(|bytes| {
                canaries
                    .len()
                    .checked_mul(size_of::<SemanticTrainingCanaryRecord>())
                    .and_then(|canary_bytes| bytes.checked_add(canary_bytes))
            })
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
        let mut device_objective = reservation
            .alloc::<SemanticTrainingObjectiveRecord>(1)
            .map_err(|error| runtime_error("training objective allocation", error))?;
        let mut device_groups = reservation
            .alloc::<SemanticTrainingObjectiveGroupRecord>(groups.len())
            .map_err(|error| runtime_error("training objective group allocation", error))?;
        let mut device_group_members = reservation
            .alloc::<u64>(group_members.len())
            .map_err(|error| runtime_error("training objective member allocation", error))?;
        let mut device_canaries = reservation
            .alloc::<SemanticTrainingCanaryRecord>(canaries.len())
            .map_err(|error| runtime_error("training canary allocation", error))?;
        if reservation.remaining_bytes() != 0 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        provider
            .htod_sync_copy_into_tracked(&descriptors, &mut device_descriptors)
            .map_err(|error| runtime_error("training-view descriptor upload", error))?;
        provider
            .htod_sync_copy_into_tracked(&raw, &mut device_raw)
            .map_err(|error| runtime_error("training-view byte upload", error))?;
        provider
            .htod_sync_copy_into_tracked(&[objective], &mut device_objective)
            .map_err(|error| runtime_error("training objective upload", error))?;
        provider
            .htod_sync_copy_into_tracked(&groups, &mut device_groups)
            .map_err(|error| runtime_error("training objective group upload", error))?;
        provider
            .htod_sync_copy_into_tracked(&group_members, &mut device_group_members)
            .map_err(|error| runtime_error("training objective member upload", error))?;
        provider
            .htod_sync_copy_into_tracked(&canaries, &mut device_canaries)
            .map_err(|error| runtime_error("training canary upload", error))?;
        Ok(Arc::new(Self {
            provider: Arc::clone(provider),
            domain: domain.clone(),
            select,
            gather,
            descriptors: device_descriptors,
            raw: device_raw,
            objective: device_objective,
            groups: device_groups,
            group_members: device_group_members,
            canaries: device_canaries,
            row_count: descriptors.len(),
            capacity,
        }))
    }

    pub(crate) fn enqueue_selection(
        self: &Arc<Self>,
        selected_view: DeviceMemoryView<u8>,
        cursor: u64,
        training_rng: [u64; 4],
        origin_candidates: Option<Arc<TrackedCudaSlice<SemanticTrainingViewOriginRecord>>>,
    ) -> Result<SemanticSelectedTrainingView, SemanticTransitionError> {
        let output_bytes = self.selection_bytes()?;
        let mut reservation = self
            .provider
            .memory()
            .reserve_bytes(
                u64::try_from(output_bytes)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            )
            .map_err(|error| runtime_error("selected training-view reservation", error))?;
        let selected = self.allocate_selection_reserved(&mut reservation, origin_candidates)?;
        if reservation.remaining_bytes() != 0 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        selected.enqueue(selected_view, cursor, training_rng, None)?;
        Ok(selected)
    }
}

fn allocate_port<T: DeviceRepr>(
    reservation: &mut crate::memory::GpuMemoryReservation,
    row_count: usize,
    capacity: usize,
) -> Result<TrackedCudaSlice<T>, SemanticTransitionError> {
    reservation
        .alloc::<T>(
            row_count
                .checked_mul(capacity)
                .ok_or(SemanticTransitionError::GenerationExhausted)?,
        )
        .map_err(|error| runtime_error("selected training-view port allocation", error))
}

fn validate_objective(
    objective: SemanticTrainingObjective,
    rows: &[TrainingViewRowDescriptor],
    capacity: usize,
) -> Result<ValidatedTrainingObjective, SemanticTransitionError> {
    if objective.groups.len() != 8 || objective.canaries.len() != 5 {
        return Err(input_error(
            "training objective requires all eight reduction groups and five canaries",
        ));
    }
    if !objective.evaluator_min.is_finite()
        || !objective.evaluator_max.is_finite()
        || objective.evaluator_min > objective.evaluator_max
        || objective
            .coefficients
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || objective.cost_unit == Identity256::default()
        || objective.cost_cap == 0
    {
        return Err(input_error(
            "training objective has invalid evaluator, coefficient or cost bounds",
        ));
    }
    let mut seen_groups = [false; 8];
    let mut referenced_rows = vec![false; rows.len()];
    let mut groups = Vec::with_capacity(objective.groups.len());
    let mut group_members = Vec::new();
    for group in &objective.groups {
        let index = group.kind as usize - 1;
        if seen_groups[index] || group.denominator == 0 || group.row_ordinals.is_empty() {
            return Err(input_error(
                "training objective groups must be unique, nonempty and have positive denominators",
            ));
        }
        seen_groups[index] = true;
        let member_offset = group_members.len();
        let mut previous = None;
        for &ordinal in &group.row_ordinals {
            let row_ordinal = usize::try_from(ordinal)
                .map_err(|_| input_error("training objective row exceeds host address space"))?;
            let row = rows
                .get(row_ordinal)
                .ok_or_else(|| input_error("training objective names an unknown replay row"))?;
            if previous.is_some_and(|value| value >= ordinal) {
                return Err(input_error(
                    "training objective row ordinals must be strictly increasing",
                ));
            }
            let anchor_group = matches!(
                group.kind,
                SemanticTrainingObjectiveGroupKind::RetentionLanguage
                    | SemanticTrainingObjectiveGroupKind::RetentionSymbolic
            );
            let policy_group = group.kind == SemanticTrainingObjectiveGroupKind::ActorCriticCost;
            if (anchor_group && row.basis != SemanticTrainingViewBasis::CorpusAnchor as u64)
                || (policy_group
                    && (row.basis != SemanticTrainingViewBasis::Episode as u64
                        || row.origin.transition != PROPOSAL_TRANSITION))
            {
                return Err(input_error(
                    "training objective group refers to an incompatible replay basis",
                ));
            }
            referenced_rows[row_ordinal] = true;
            group_members.push(ordinal);
            previous = Some(ordinal);
        }
        groups.push(SemanticTrainingObjectiveGroupRecord {
            kind: group.kind as u64,
            denominator: group.denominator,
            member_offset: u64::try_from(member_offset)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            member_count: u64::try_from(group.row_ordinals.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        });
    }
    if seen_groups.iter().any(|seen| !seen) || referenced_rows.iter().any(|seen| !seen) {
        return Err(input_error(
            "training objective must cover every mandatory group and replay row",
        ));
    }
    let return_scale = objective
        .evaluator_min
        .abs()
        .max(objective.evaluator_max.abs())
        .max(1.0);
    let actor_coefficient = (1.0 / return_scale) as f32;
    let critic_coefficient = (1.0 / (return_scale * return_scale)) as f32;
    if objective.coefficients[6].to_bits() != actor_coefficient.to_bits()
        || objective.coefficients[7].to_bits() != critic_coefficient.to_bits()
    {
        return Err(input_error(
            "actor and critic coefficients differ from the frozen evaluator scale",
        ));
    }
    let mut seen_canaries = [false; 5];
    let mut canaries = Vec::with_capacity(objective.canaries.len());
    for canary in &objective.canaries {
        let index = canary.kind as usize - 1;
        if seen_canaries[index]
            || !canary.lower_bound.is_finite()
            || !canary.upper_bound.is_finite()
            || canary.lower_bound > canary.upper_bound
            || canary.memory_limit == 0
            || canary.fuel_limit == 0
            || canary.identity == Identity256::default()
            || usize::try_from(canary.row_ordinal)
                .ok()
                .is_none_or(|ordinal| ordinal >= rows.len())
        {
            return Err(input_error(
                "training canaries require unique kinds, valid rows, bounds and resource ceilings",
            ));
        }
        seen_canaries[index] = true;
        canaries.push(SemanticTrainingCanaryRecord {
            kind: canary.kind as u64,
            row_ordinal: canary.row_ordinal,
            lower_bound_bits: canary.lower_bound.to_bits(),
            upper_bound_bits: canary.upper_bound.to_bits(),
            memory_limit: canary.memory_limit,
            fuel_limit: canary.fuel_limit,
            identity: identity_words(canary.identity),
        });
    }
    if seen_canaries.iter().any(|seen| !seen)
        || objective_identity(&objective) != objective.identity
    {
        return Err(input_error(
            "training objective is incomplete or differs from its frozen identity",
        ));
    }
    Ok((
        SemanticTrainingObjectiveRecord {
            identity: identity_words(objective.identity),
            row_count: u64::try_from(rows.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            capacity: u64::try_from(capacity)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            group_count: u64::try_from(groups.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            group_member_count: u64::try_from(group_members.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            canary_count: u64::try_from(canaries.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            evaluator_min_bits: objective.evaluator_min.to_bits(),
            evaluator_max_bits: objective.evaluator_max.to_bits(),
            coefficient_bits: objective
                .coefficients
                .map(|value| u64::from(value.to_bits())),
            cost_unit: identity_words(objective.cost_unit),
            cost_cap: objective.cost_cap,
        },
        groups,
        group_members,
        canaries,
    ))
}

fn objective_identity(objective: &SemanticTrainingObjective) -> Identity256 {
    let mut hasher = Sha256::new();
    hasher.update(b"xlog.semantic.training-objective.v1\0");
    hasher.update(objective.evaluator_min.to_bits().to_le_bytes());
    hasher.update(objective.evaluator_max.to_bits().to_le_bytes());
    for coefficient in objective.coefficients {
        hasher.update(coefficient.to_bits().to_le_bytes());
    }
    hasher.update(objective.cost_unit.as_bytes());
    hasher.update(objective.cost_cap.to_le_bytes());
    hasher.update(
        u64::try_from(objective.groups.len())
            .expect("validated training objective group count")
            .to_le_bytes(),
    );
    for group in &objective.groups {
        hasher.update((group.kind as u64).to_le_bytes());
        hasher.update(group.denominator.to_le_bytes());
        hasher.update(
            u64::try_from(group.row_ordinals.len())
                .expect("validated training objective group extent")
                .to_le_bytes(),
        );
        for ordinal in &group.row_ordinals {
            hasher.update(ordinal.to_le_bytes());
        }
    }
    hasher.update(
        u64::try_from(objective.canaries.len())
            .expect("validated training canary count")
            .to_le_bytes(),
    );
    for canary in &objective.canaries {
        hasher.update((canary.kind as u64).to_le_bytes());
        hasher.update(canary.row_ordinal.to_le_bytes());
        hasher.update(canary.lower_bound.to_bits().to_le_bytes());
        hasher.update(canary.upper_bound.to_bits().to_le_bytes());
        hasher.update(canary.memory_limit.to_le_bytes());
        hasher.update(canary.fuel_limit.to_le_bytes());
        hasher.update(canary.identity.as_bytes());
    }
    Identity256::from_bytes(hasher.finalize().into())
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
        || !raw_offset.is_multiple_of(8)
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
            SemanticTransitionKind::Update => 4,
        },
        lineage_instance: identity_words(origin.predecessor.instance),
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
