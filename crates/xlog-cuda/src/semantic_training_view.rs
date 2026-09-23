use std::mem::size_of;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::launch::LaunchEnqueueError;
use crate::memory::{DeviceMemoryView, TrackedCudaSlice};
use crate::provider::resident_schedule::{validate_execution_domain, ResidentExecutionDomain};
use crate::semantic_transition::{
    Identity256, SemanticPublishedIdentity, SemanticRngBinding, SemanticTaskContentIdentity,
    SemanticTransitionError, SemanticTransitionKind,
};
use crate::{
    CudaFunction, CudaKernelProvider, DeviceRepr, LaunchAsync, LaunchConfig, SemanticTruth,
};

type TrainingViewPort = (DeviceMemoryView<u8>, Vec<i64>, Vec<i64>, (u8, u8));
type ValidatedTrainingObjective = (
    SemanticTrainingObjectiveRecord,
    Vec<SemanticTrainingObjectiveGroupRecord>,
    Vec<u64>,
    Vec<SemanticTrainingCanaryRecord>,
    Vec<u64>,
);

const MODULE: &str = "xlog_semantic_training_view";
const SELECT_KERNEL: &str = "semantic_training_view_select";
const GATHER_KERNEL: &str = "semantic_training_view_gather";
const TRAINING_VIEW_HEADER_BYTES: usize = 264;
const TRAINING_VIEW_ROW_BYTES: usize = 84;
const PROPOSAL_TRANSITION: u64 = 1;
pub const SEMANTIC_TRAINING_CANARY_EVALUATOR_ABI: u64 = 3;

/// Origin of one authentic replay training view.
#[repr(u64)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticTrainingViewBasis {
    Episode = 1,
    CorpusLanguageAnchor = 2,
    CorpusSymbolicAnchor = 3,
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
    pub identity: Identity256,
    pub bytes_identity: Identity256,
    pub content_identity: Identity256,
    /// Present only for a symbolic corpus anchor. The three identities must
    /// equal the native task content executed during this cold binding.
    pub task_content: Option<SemanticTaskContentIdentity>,
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

/// Mandatory acceptance gate evaluated from original full-vocabulary logits.
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
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticTrainingCanary {
    pub kind: SemanticTrainingCanaryKind,
    pub row_ordinal: u64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub memory_limit: u64,
    pub work_limit: u64,
    /// Model-logit positions whose next-token predictions must equal the
    /// canonical truth token for each native task result, in query order. Only
    /// the symbolic-utility and goal-chain canaries use these coordinates;
    /// every other kind uses `[u64::MAX; 3]`.
    pub obligation_positions: [u64; 3],
    /// Explicit frozen members whose individual retention is uncompensated.
    /// Only the retained-behavior canary carries this set. Aggregate retention
    /// labels remain the metric denominator and do not imply protection.
    pub protected_positions: Vec<u64>,
}

/// Complete frozen objective carried by the canonical replay-roster owner.
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticTrainingObjective {
    pub evaluator_min: f64,
    pub evaluator_max: f64,
    /// Masked-language, autoregressive, semantic, edit, execution, retention,
    /// actor, critic and cost coefficients, in that order.
    pub coefficients: [f32; 9],
    pub cost_unit: Identity256,
    pub cost_cap: u64,
    /// Vocabulary token representing NEITHER, TRUE, FALSE and BOTH.
    pub truth_tokens: [u64; 4],
    pub groups: Vec<SemanticTrainingObjectiveGroup>,
    pub canaries: Vec<SemanticTrainingCanary>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SemanticTrainingObjectiveRecord {
    pub evaluator_abi: u64,
    pub identity: [u64; 4],
    pub task_identity: [u64; 4],
    pub row_count: u64,
    pub capacity: u64,
    pub group_count: u64,
    pub group_member_count: u64,
    pub canary_count: u64,
    pub protected_member_count: u64,
    pub evaluator_min_bits: u64,
    pub evaluator_max_bits: u64,
    pub coefficient_bits: [u64; 9],
    pub cost_unit: [u64; 4],
    pub cost_cap: u64,
    pub truth_tokens: [u64; 4],
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
    pub evaluator_abi: u64,
    pub kind: u64,
    pub row_ordinal: u64,
    pub lower_bound_bits: u64,
    pub upper_bound_bits: u64,
    pub memory_limit: u64,
    pub work_limit: u64,
    pub obligation_positions: [u64; 3],
    pub protected_member_offset: u64,
    pub protected_member_count: u64,
    pub row_identity: [u64; 4],
    pub row_content_identity: [u64; 4],
    pub task_identity: [u64; 4],
    pub identity: [u64; 4],
}

// SAFETY: the fixed CUDA ABI contains only u64 words.
unsafe impl DeviceRepr for SemanticTrainingCanaryRecord {}

/// Native device-produced measurement for one frozen candidate-update canary.
///
/// The result repeats the frozen kind, row and identity so the native update
/// gate can reject a measurement produced for any other roster entry. The
/// measurement is carried as raw FP64 bits; memory and work are exact integer
/// tallies checked against the frozen ceilings.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SemanticTrainingCanaryResultRecord {
    pub kind: u64,
    pub row_ordinal: u64,
    pub measurement_bits: u64,
    pub memory_used: u64,
    pub work_used: u64,
    pub identity: [u64; 4],
}

// SAFETY: the fixed CUDA ABI contains only u64 words.
unsafe impl DeviceRepr for SemanticTrainingCanaryResultRecord {}

/// Closed device reason for refusing a candidate model update canary.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticTrainingCanaryRefusalReason {
    NonFiniteMeasurement = 1,
    OutsideBounds = 2,
    MemoryLimitExceeded = 3,
    WorkLimitExceeded = 4,
    IncompleteOperands = 5,
    ProtectedRetentionLost = 6,
    GoalWitnessInvalid = 7,
    WorkOverflow = 8,
    GenerationMismatch = 9,
}

/// Device-authored evidence for the highest-precedence failed update canary.
///
/// A successful canary join has ABI one and every other word zero. A refusal
/// retains the exact measured value, frozen comparison bounds, resource use and
/// limits, and both the canary and selected-view identities.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemanticTrainingCanaryRefusalRecord {
    pub abi: u64,
    pub reason: u64,
    pub kind: u64,
    pub row_ordinal: u64,
    pub measurement_bits: u64,
    pub lower_bound_bits: u64,
    pub upper_bound_bits: u64,
    pub memory_used: u64,
    pub memory_limit: u64,
    pub work_used: u64,
    pub work_limit: u64,
    pub identity: [u64; 4],
    pub selection_identity: [u64; 4],
}

impl SemanticTrainingCanaryRefusalRecord {
    pub fn reason(&self) -> Option<SemanticTrainingCanaryRefusalReason> {
        match self.reason {
            1 => Some(SemanticTrainingCanaryRefusalReason::NonFiniteMeasurement),
            2 => Some(SemanticTrainingCanaryRefusalReason::OutsideBounds),
            3 => Some(SemanticTrainingCanaryRefusalReason::MemoryLimitExceeded),
            4 => Some(SemanticTrainingCanaryRefusalReason::WorkLimitExceeded),
            5 => Some(SemanticTrainingCanaryRefusalReason::IncompleteOperands),
            6 => Some(SemanticTrainingCanaryRefusalReason::ProtectedRetentionLost),
            7 => Some(SemanticTrainingCanaryRefusalReason::GoalWitnessInvalid),
            8 => Some(SemanticTrainingCanaryRefusalReason::WorkOverflow),
            9 => Some(SemanticTrainingCanaryRefusalReason::GenerationMismatch),
            _ => None,
        }
    }
}

// SAFETY: the fixed CUDA ABI contains only u64 words.
unsafe impl DeviceRepr for SemanticTrainingCanaryRefusalRecord {}

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
    task_query_identity: [u64; 4],
    task_theory_program_identity: [u64; 4],
    task_result_identity: [u64; 4],
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
    branch_labels: u64,
    branch_ids: u64,
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
    branch_labels: TrackedCudaSlice<i64>,
    branch_ids: TrackedCudaSlice<i64>,
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
    BranchLabels,
    BranchIds,
    SourceSlots,
    LogicalPositions,
    Kinds,
    Parents,
}

impl SemanticSelectedTrainingView {
    pub fn capacity(&self) -> usize {
        self.arena.capacity
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn accounted_allocation_bytes(&self) -> Result<u64, SemanticTransitionError> {
        let mut allocations: Vec<crate::memory::DeviceAllocationProvenance> = Vec::new();
        macro_rules! account {
            ($slice:expr) => {{
                let provenance = $slice
                    .view()
                    .allocation_provenance()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if !allocations
                    .iter()
                    .any(|known| provenance.same_allocation(known))
                {
                    allocations.push(provenance);
                }
            }};
        }
        account!(self.arena.descriptors);
        account!(self.arena.raw);
        account!(self.arena.objective);
        account!(self.arena.groups);
        account!(self.arena.group_members);
        account!(self.arena.canaries);
        account!(self.arena.protected_members);
        account!(self.storage.selection);
        account!(self.storage.roster_rows);
        account!(self.storage.token_ids);
        account!(self.storage.mask_labels);
        account!(self.storage.mask_weights);
        account!(self.storage.ar_labels);
        account!(self.storage.retention_labels);
        account!(self.storage.branch_labels);
        account!(self.storage.branch_ids);
        account!(self.storage.source_slots);
        account!(self.storage.logical_positions);
        account!(self.storage.kinds);
        account!(self.storage.parents);
        allocations.into_iter().try_fold(0u64, |total, allocation| {
            total
                .checked_add(allocation.allocation_bytes())
                .ok_or(SemanticTransitionError::GenerationExhausted)
        })
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

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn objective(&self) -> DeviceMemoryView<SemanticTrainingObjectiveRecord> {
        self.arena.objective.view()
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn objective_groups(
        &self,
    ) -> DeviceMemoryView<SemanticTrainingObjectiveGroupRecord> {
        self.arena.groups.view()
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn objective_group_members(&self) -> DeviceMemoryView<u64> {
        self.arena.group_members.view()
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn canaries(&self) -> DeviceMemoryView<SemanticTrainingCanaryRecord> {
        self.arena.canaries.view()
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn protected_members(&self) -> DeviceMemoryView<u64> {
        self.arena.protected_members.view()
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

    pub fn branch_labels(&self) -> DeviceMemoryView<i64> {
        self.storage.branch_labels.view()
    }

    pub fn branch_ids(&self) -> DeviceMemoryView<i64> {
        self.storage.branch_ids.view()
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
            SemanticTrainingViewPort::BranchLabels => (
                unsafe { self.storage.branch_labels.view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                vec![],
                vec![],
                (0, 64),
            ),
            SemanticTrainingViewPort::BranchIds => (
                unsafe { self.storage.branch_ids.view().cast::<u8>() }
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
            | SemanticTrainingViewPort::BranchLabels
            | SemanticTrainingViewPort::BranchIds
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
            branch_labels: self.storage.branch_labels.device_ptr_value(),
            branch_ids: self.storage.branch_ids.device_ptr_value(),
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
        recorder.write(&self.storage.branch_labels);
        recorder.write(&self.storage.branch_ids);
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
    #[cfg(feature = "semantic-policy")]
    protected_members: TrackedCudaSlice<u64>,
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
            branch_labels: allocate_port(reservation, self.row_count, self.capacity)?,
            branch_ids: allocate_port(reservation, self.row_count, self.capacity)?,
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
        task_identity: Identity256,
        task_content: SemanticTaskContentIdentity,
        expected_truth: [SemanticTruth; 3],
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
            let descriptor = validate_row(ordinal, &row, raw.len())?;
            capacity = capacity.max(descriptor.window as usize);
            raw.extend_from_slice(&row.bytes);
            descriptors.push(descriptor);
        }
        #[cfg(feature = "semantic-policy")]
        let (objective, groups, group_members, canaries, protected_members) = validate_objective(
            objective,
            &descriptors,
            &raw,
            capacity,
            task_identity,
            task_content,
            expected_truth,
        )?;
        #[cfg(not(feature = "semantic-policy"))]
        let (objective, groups, group_members, canaries, _) = validate_objective(
            objective,
            &descriptors,
            &raw,
            capacity,
            task_identity,
            task_content,
            expected_truth,
        )?;
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
            });
        #[cfg(feature = "semantic-policy")]
        let bytes = bytes.and_then(|bytes| {
            protected_members
                .len()
                .checked_mul(size_of::<u64>())
                .and_then(|member_bytes| bytes.checked_add(member_bytes))
        });
        let bytes = bytes.ok_or(SemanticTransitionError::GenerationExhausted)?;
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
        #[cfg(feature = "semantic-policy")]
        let mut device_protected_members = reservation
            .alloc::<u64>(protected_members.len())
            .map_err(|error| runtime_error("protected retention member allocation", error))?;
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
        #[cfg(feature = "semantic-policy")]
        provider
            .htod_sync_copy_into_tracked(&protected_members, &mut device_protected_members)
            .map_err(|error| runtime_error("protected retention member upload", error))?;
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
            #[cfg(feature = "semantic-policy")]
            protected_members: device_protected_members,
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
    raw: &[u8],
    capacity: usize,
    task_identity: Identity256,
    task_content: SemanticTaskContentIdentity,
    expected_truth: [SemanticTruth; 3],
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
        || task_identity == Identity256::default()
        || objective
            .truth_tokens
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != objective.truth_tokens.len()
    {
        return Err(input_error(
            "training objective has invalid evaluator, coefficient or cost bounds",
        ));
    }
    let expected_query = identity_words(task_content.query);
    let expected_theory_program = identity_words(task_content.theory_program);
    let expected_result = identity_words(task_content.result);
    for row in rows {
        let symbolic = row.basis == SemanticTrainingViewBasis::CorpusSymbolicAnchor as u64;
        let bound = row.task_query_identity == expected_query
            && row.task_theory_program_identity == expected_theory_program
            && row.task_result_identity == expected_result;
        let absent = row.task_query_identity == [0; 4]
            && row.task_theory_program_identity == [0; 4]
            && row.task_result_identity == [0; 4];
        if (symbolic && !bound) || (!symbolic && !absent) {
            return Err(input_error(
                "symbolic training view differs from the native query, theory/program or result binding",
            ));
        }
    }
    let mut seen_groups = [false; 8];
    let mut referenced_rows = vec![false; rows.len()];
    let mut groups = Vec::with_capacity(objective.groups.len());
    let mut group_members = Vec::new();
    for group in &objective.groups {
        let index = group.kind as usize - 1;
        if seen_groups[index]
            || group.row_ordinals.is_empty()
            || group.denominator
                != u64::try_from(group.row_ordinals.len())
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?
        {
            return Err(input_error(
                "training objective groups must be unique, nonempty and use their exact member count as denominator",
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
            let policy_group = group.kind == SemanticTrainingObjectiveGroupKind::ActorCriticCost;
            let expected_anchor = match group.kind {
                SemanticTrainingObjectiveGroupKind::RetentionLanguage => {
                    Some(SemanticTrainingViewBasis::CorpusLanguageAnchor)
                }
                SemanticTrainingObjectiveGroupKind::RetentionSymbolic => {
                    Some(SemanticTrainingViewBasis::CorpusSymbolicAnchor)
                }
                _ => None,
            };
            if expected_anchor.is_some_and(|basis| row.basis != basis as u64)
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
    if [0, 1, 2, 3, 4, 5, 8]
        .into_iter()
        .any(|index| objective.coefficients[index].to_bits() != 1.0f32.to_bits())
        || objective.coefficients[6].to_bits() != actor_coefficient.to_bits()
        || objective.coefficients[7].to_bits() != critic_coefficient.to_bits()
    {
        return Err(input_error(
            "training coefficients differ from the frozen primary and evaluator-scaled law",
        ));
    }
    let mut seen_canaries = [false; 5];
    let mut canaries = Vec::with_capacity(objective.canaries.len());
    let mut protected_members = Vec::new();
    for canary in &objective.canaries {
        let index = canary.kind as usize - 1;
        let row = usize::try_from(canary.row_ordinal)
            .ok()
            .and_then(|ordinal| rows.get(ordinal));
        let obligation_kind = matches!(
            canary.kind,
            SemanticTrainingCanaryKind::SymbolicUtility | SemanticTrainingCanaryKind::GoalChain
        );
        let positions_valid = if obligation_kind {
            row.is_some_and(|row| {
                row.basis == SemanticTrainingViewBasis::CorpusSymbolicAnchor as u64
                    && canary
                        .obligation_positions
                        .windows(2)
                        .all(|pair| pair[0] < pair[1])
                    && canary
                        .obligation_positions
                        .iter()
                        .all(|position| *position < row.window)
                    && canary.obligation_positions.iter().zip(expected_truth).all(
                        |(position, truth)| {
                            position
                                .checked_add(1)
                                .filter(|target| {
                                    *target >= row.answer_start
                                        && *target < row.window
                                        && *target < row.source_length
                                })
                                .and_then(|target| {
                                    usize::try_from(row.raw_offset).ok().and_then(|base| {
                                        usize::try_from(target).ok().and_then(|target| {
                                            base.checked_add(TRAINING_VIEW_HEADER_BYTES).and_then(
                                                |offset| {
                                                    target
                                                        .checked_mul(size_of::<i64>())
                                                        .and_then(|bytes| offset.checked_add(bytes))
                                                },
                                            )
                                        })
                                    })
                                })
                                .and_then(|offset| {
                                    offset
                                        .checked_add(size_of::<i64>())
                                        .and_then(|end| raw.get(offset..end))
                                })
                                .map(|bytes| {
                                    u64::from_le_bytes(
                                        bytes.try_into().expect("bounded truth token"),
                                    ) == objective.truth_tokens[truth as usize]
                                })
                                .unwrap_or(false)
                        },
                    )
            })
        } else {
            canary.obligation_positions == [u64::MAX; 3]
        };
        let protected_valid = if canary.kind == SemanticTrainingCanaryKind::RetainedBehavior {
            canary
                .protected_positions
                .windows(2)
                .all(|pair| pair[0] < pair[1])
                && row.is_some_and(|row| {
                    canary
                        .protected_positions
                        .iter()
                        .all(|position| *position < row.window)
                })
        } else {
            canary.protected_positions.is_empty()
        };
        if index != canaries.len()
            || seen_canaries[index]
            || !canary.lower_bound.is_finite()
            || !canary.upper_bound.is_finite()
            || canary.lower_bound > canary.upper_bound
            || canary.memory_limit == 0
            || canary.work_limit == 0
            || row.is_none()
            || !positions_valid
            || !protected_valid
            || (canary.kind == SemanticTrainingCanaryKind::GoalChain
                && (canary.lower_bound.to_bits() != 0.0f64.to_bits()
                    || canary.upper_bound.to_bits() != 0.0f64.to_bits()))
        {
            return Err(input_error(
                "training canaries require canonical kind order, valid rows, bounds and resource ceilings",
            ));
        }
        seen_canaries[index] = true;
        let row = row.expect("checked canary row");
        let identity = canary_identity(&objective, canary, row, task_identity);
        let protected_member_offset = protected_members.len();
        protected_members.extend_from_slice(&canary.protected_positions);
        canaries.push(SemanticTrainingCanaryRecord {
            evaluator_abi: SEMANTIC_TRAINING_CANARY_EVALUATOR_ABI,
            kind: canary.kind as u64,
            row_ordinal: canary.row_ordinal,
            lower_bound_bits: canary.lower_bound.to_bits(),
            upper_bound_bits: canary.upper_bound.to_bits(),
            memory_limit: canary.memory_limit,
            work_limit: canary.work_limit,
            obligation_positions: canary.obligation_positions,
            protected_member_offset: u64::try_from(protected_member_offset)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            protected_member_count: u64::try_from(canary.protected_positions.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            row_identity: row.identity,
            row_content_identity: row.content_identity,
            task_identity: identity_words(task_identity),
            identity: identity_words(identity),
        });
    }
    if seen_canaries.iter().any(|seen| !seen) {
        return Err(input_error("training objective is incomplete"));
    }
    let identity = objective_identity(&objective, rows, task_identity, &canaries);
    Ok((
        SemanticTrainingObjectiveRecord {
            evaluator_abi: SEMANTIC_TRAINING_CANARY_EVALUATOR_ABI,
            identity: identity_words(identity),
            task_identity: identity_words(task_identity),
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
            protected_member_count: u64::try_from(protected_members.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            evaluator_min_bits: objective.evaluator_min.to_bits(),
            evaluator_max_bits: objective.evaluator_max.to_bits(),
            coefficient_bits: objective
                .coefficients
                .map(|value| u64::from(value.to_bits())),
            cost_unit: identity_words(objective.cost_unit),
            cost_cap: objective.cost_cap,
            truth_tokens: objective.truth_tokens,
        },
        groups,
        group_members,
        canaries,
        protected_members,
    ))
}

fn objective_identity(
    objective: &SemanticTrainingObjective,
    rows: &[TrainingViewRowDescriptor],
    task_identity: Identity256,
    canaries: &[SemanticTrainingCanaryRecord],
) -> Identity256 {
    let mut hasher = Sha256::new();
    hasher.update(b"xlog.semantic.training-objective.v3\0");
    hasher.update(SEMANTIC_TRAINING_CANARY_EVALUATOR_ABI.to_le_bytes());
    hasher.update(task_identity.as_bytes());
    hasher.update(objective.evaluator_min.to_bits().to_le_bytes());
    hasher.update(objective.evaluator_max.to_bits().to_le_bytes());
    for coefficient in objective.coefficients {
        hasher.update(coefficient.to_bits().to_le_bytes());
    }
    hasher.update(objective.cost_unit.as_bytes());
    hasher.update(objective.cost_cap.to_le_bytes());
    for token in objective.truth_tokens {
        hasher.update(token.to_le_bytes());
    }
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
            let row = &rows[*ordinal as usize];
            for word in row.identity {
                hasher.update(word.to_le_bytes());
            }
            for word in row.content_identity {
                hasher.update(word.to_le_bytes());
            }
        }
    }
    hasher.update(
        u64::try_from(objective.canaries.len())
            .expect("validated training canary count")
            .to_le_bytes(),
    );
    for canary in canaries {
        for word in canary.identity {
            hasher.update(word.to_le_bytes());
        }
    }
    Identity256::from_bytes(hasher.finalize().into())
}

fn canary_identity(
    objective: &SemanticTrainingObjective,
    canary: &SemanticTrainingCanary,
    row: &TrainingViewRowDescriptor,
    task_identity: Identity256,
) -> Identity256 {
    let mut hasher = Sha256::new();
    hasher.update(b"xlog.semantic.training-canary.v3\0");
    hasher.update(SEMANTIC_TRAINING_CANARY_EVALUATOR_ABI.to_le_bytes());
    hasher.update((canary.kind as u64).to_le_bytes());
    hasher.update(canary.row_ordinal.to_le_bytes());
    hasher.update(canary.lower_bound.to_bits().to_le_bytes());
    hasher.update(canary.upper_bound.to_bits().to_le_bytes());
    hasher.update(canary.memory_limit.to_le_bytes());
    hasher.update(canary.work_limit.to_le_bytes());
    for position in canary.obligation_positions {
        hasher.update(position.to_le_bytes());
    }
    hasher.update(
        u64::try_from(canary.protected_positions.len())
            .expect("validated protected retention member count")
            .to_le_bytes(),
    );
    for position in &canary.protected_positions {
        hasher.update(position.to_le_bytes());
    }
    for word in row.identity {
        hasher.update(word.to_le_bytes());
    }
    for word in row.content_identity {
        hasher.update(word.to_le_bytes());
    }
    hasher.update(task_identity.as_bytes());
    for token in objective.truth_tokens {
        hasher.update(token.to_le_bytes());
    }
    Identity256::from_bytes(hasher.finalize().into())
}

fn validate_row(
    ordinal: usize,
    row: &SemanticTrainingViewRow,
    raw_offset: usize,
) -> Result<TrainingViewRowDescriptor, SemanticTransitionError> {
    let bytes = &row.bytes;
    let mut schema = [0u8; 32];
    let tag = b"dlm-new/training-view/v2";
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
        || word(104) > word(96)
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
    let shared_context_end = word(136);
    if shared_context_end == 0 {
        if (1..16).any(|index| word(136 + index * 8) != 0) {
            return Err(input_error(
                "training-view row has incomplete branch geometry",
            ));
        }
    } else {
        let mut next_begin = word(104)
            .checked_mul(2)
            .ok_or_else(|| input_error("training-view branch extent exceeds address space"))?;
        if shared_context_end > word(128) || next_begin > word(96) {
            return Err(input_error("training-view row has invalid branch geometry"));
        }
        for branch in 0..3 {
            let offset = 144 + branch * 40;
            let tail_begin = word(offset);
            let tail_end = word(offset + 8);
            let answer_begin = word(offset + 16);
            let answer_end = word(offset + 24);
            let answer_start = word(offset + 32);
            if tail_begin != next_begin
                || tail_end <= tail_begin
                || answer_begin != tail_end
                || answer_end <= answer_begin
                || answer_end > word(96)
                || shared_context_end.checked_add(tail_end - tail_begin) != Some(answer_start)
                || answer_start
                    .checked_add(answer_end - answer_begin)
                    .is_none_or(|end| end > word(96))
            {
                return Err(input_error("training-view row has invalid branch geometry"));
            }
            next_begin = answer_end;
        }
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
    if matches!(row.basis, SemanticTrainingViewBasis::Episode) != row.origin.is_some() {
        return Err(input_error(
            "only an episode training view has an authentic execution origin",
        ));
    }
    if matches!(row.basis, SemanticTrainingViewBasis::CorpusSymbolicAnchor)
        != row.task_content.is_some()
    {
        return Err(input_error(
            "only a symbolic corpus training view carries a native task-content binding",
        ));
    }
    if identity != *row.identity.as_bytes()
        || Sha256::digest(bytes).as_slice() != row.bytes_identity.as_bytes()
        || source_identity == [0; 32]
        || row.content_identity == Identity256::default()
    {
        return Err(input_error(
            "training-view row differs from its logical identity, material bytes, source or replay content",
        ));
    }
    Ok(TrainingViewRowDescriptor {
        ordinal: u64::try_from(ordinal)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        basis: row.basis as u64,
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
        content_identity: identity_words(row.content_identity),
        task_query_identity: row
            .task_content
            .map(|binding| identity_words(binding.query))
            .unwrap_or_default(),
        task_theory_program_identity: row
            .task_content
            .map(|binding| identity_words(binding.theory_program))
            .unwrap_or_default(),
        task_result_identity: row
            .task_content
            .map(|binding| identity_words(binding.result))
            .unwrap_or_default(),
        origin: row.origin.map(origin_record).unwrap_or_default(),
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
