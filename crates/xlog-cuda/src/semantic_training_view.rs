use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use sha2::{Digest, Sha256};

#[cfg(feature = "semantic-policy")]
use crate::cuda_compat::{IntoKernelParamStorage, KernelParamStorage};
use crate::launch::LaunchEnqueueError;
use crate::memory::{DeviceMemoryView, TrackedCudaSlice};
use crate::provider::resident_schedule::{validate_execution_domain, ResidentExecutionDomain};
#[cfg(feature = "semantic-policy")]
use crate::semantic_transition::OriginalNativeCommand;
use crate::semantic_transition::{
    Identity256, SemanticPublishedIdentity, SemanticReplayAppendBinding, SemanticRngBinding,
    SemanticTaskContentIdentity, SemanticTransitionError, SemanticTransitionKind,
};
use crate::{
    CudaFunction, CudaKernelProvider, DeviceRepr, LaunchAsync, LaunchConfig, SemanticTruth,
};

type TrainingViewPort = (DeviceMemoryView<u8>, Vec<i64>, Vec<i64>, (u8, u8));
type TrainingViewPortLayout = (Vec<i64>, Vec<i64>, (u8, u8));
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
pub(crate) const SEMANTIC_TRAINING_COMMITTED_APPEND_ORIGIN: u64 = u64::MAX - 1;

/// Validated geometry of the bytes consumed by native training-view selection.
#[derive(Clone, Copy, Debug)]
pub struct SemanticTrainingViewLayout {
    pub window: usize,
    pub source_length: u64,
    pub block_size: u64,
    pub prefix_extent: u64,
    pub answer_start: u64,
    pub branch_words: [u64; 16],
}

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

/// Exact sealed input identity and its four admission caps, in source order:
/// entries, total keys, records per input, bytes per record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticTrainingManifest {
    pub identity: Identity256,
    pub caps: [u64; 4],
}

/// Prospective training domain fixed before any action, not a selected roster
/// or a numerical certificate. The application supplies the original sealed
/// manifests and complete loader geometry; native admission retains them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticTrainingDomain {
    /// Strictly increasing by identity; no duplicate manifest or omitted cap.
    pub manifests: Vec<SemanticTrainingManifest>,
    pub window: u64,
    pub pad_id: u64,
    pub mask_id: u64,
    pub block_size: u64,
    pub batch_size: u64,
    pub mask_permille: u64,
    /// Canonical nonnegative decimal, preserving arbitrary-width source seeds.
    pub training_seed: String,
    pub mask_policy: Identity256,
}

impl SemanticTrainingDomain {
    pub(crate) fn validate(&self) -> Result<(), SemanticTransitionError> {
        let seed = self.training_seed.as_bytes();
        if self.manifests.is_empty()
            || self.manifests.iter().any(|manifest| {
                manifest.identity == Identity256::default() || manifest.caps.contains(&0)
            })
            || self
                .manifests
                .windows(2)
                .any(|pair| pair[0].identity.as_bytes() >= pair[1].identity.as_bytes())
            || self.window == 0
            || self.block_size == 0
            || self.batch_size == 0
            || !(1..=1000).contains(&self.mask_permille)
            || self.mask_policy == Identity256::default()
            || !(seed == b"0"
                || (matches!(seed.first(), Some(b'1'..=b'9'))
                    && seed.iter().all(u8::is_ascii_digit)))
        {
            return Err(input_error(
                "training domain requires exact manifests and loader geometry",
            ));
        }
        Ok(())
    }

    /// Each original episodes/anchors input is independently bounded by this
    /// minimum. It is not derived from arena storage or byte capacities.
    pub fn record_limit(&self) -> u64 {
        self.manifests
            .iter()
            .map(|manifest| manifest.caps[2])
            .min()
            .unwrap_or(0)
    }

    /// Cold metadata identity retained by task/checkpoint ownership, not a grant.
    pub fn identity(&self) -> Identity256 {
        let mut hash = Sha256::new();
        hash.update(b"xlog.semantic.training-domain.v1\0");
        hash.update((self.manifests.len() as u64).to_le_bytes());
        for manifest in &self.manifests {
            hash.update(manifest.identity.as_bytes());
            for cap in manifest.caps {
                hash.update(cap.to_le_bytes());
            }
        }
        for word in [
            self.window,
            self.pad_id,
            self.mask_id,
            self.block_size,
            self.batch_size,
            self.mask_permille,
        ] {
            hash.update(word.to_le_bytes());
        }
        hash.update((self.training_seed.len() as u64).to_le_bytes());
        hash.update(self.training_seed.as_bytes());
        hash.update(self.mask_policy.as_bytes());
        Identity256::from_bytes(hash.finalize().into())
    }
}

/// One already-admitted training view retained for cold device selection.
pub struct SemanticTrainingViewRow {
    pub basis: SemanticTrainingViewBasis,
    pub identity: Identity256,
    pub bytes_identity: Identity256,
    pub content_identity: Identity256,
    /// Original record's training-view identity, validated by its cold owner.
    pub data_manifest: Identity256,
    pub mask_policy: Identity256,
    pub training_seed: String,
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

/// Finite replay storage and non-actor membership fixed before results arrive.
struct SemanticTrainingReplayCapacity {
    pub row_capacity: usize,
    pub raw_byte_capacity: usize,
    pub append_groups: Vec<Vec<SemanticTrainingObjectiveGroupKind>>,
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

/// One coefficient law shared by cold task admission and full arena validation.
pub(crate) fn frozen_training_coefficients(evaluator_min: f64, evaluator_max: f64) -> [f32; 9] {
    let return_scale = evaluator_min.abs().max(evaluator_max.abs()).max(1.0);
    let mut coefficients = [1.0f32; 9];
    coefficients[6] = (1.0 / return_scale) as f32;
    coefficients[7] = (1.0 / (return_scale * return_scale)) as f32;
    coefficients
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
    pub abi: u64,
    pub kind: u64,
    pub row_ordinal: u64,
    /// Zero: not evaluated; one: evaluation started without a complete metric;
    /// two: the complete original metric was obtained, including refused values.
    pub availability: u64,
    pub reason: u64,
    pub measurement_bits: u64,
    pub memory_used: u64,
    pub work_used: u64,
    pub identity: [u64; 4],
}

// SAFETY: the fixed CUDA ABI contains only u64 words.
unsafe impl DeviceRepr for SemanticTrainingCanaryResultRecord {}

const _: () = assert!(size_of::<SemanticTrainingCanaryResultRecord>() == 96);

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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
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
pub(crate) struct TrainingViewRowDescriptor {
    pub ordinal: u64,
    pub basis: u64,
    pub raw_offset: u64,
    pub raw_bytes: u64,
    pub window: u64,
    pub source_length: u64,
    pub block_size: u64,
    pub prefix_extent: u64,
    pub answer_start: u64,
    pub identity: [u64; 4],
    pub source_identity: [u64; 4],
    pub content_identity: [u64; 4],
    pub task_query_identity: [u64; 4],
    pub task_theory_program_identity: [u64; 4],
    pub task_result_identity: [u64; 4],
    pub origin: SemanticTrainingViewOriginRecord,
}

// SAFETY: the fixed CUDA ABI contains only u64 words.
unsafe impl DeviceRepr for TrainingViewRowDescriptor {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct SemanticTrainingReplayAppendHeader {
    pub abi: u64,
    pub count: u64,
    pub capacity: u64,
    pub payload_used_bytes: u64,
    pub payload_capacity_bytes: u64,
    pub eligible_count: u64,
    pub chain_head: Identity256,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct SemanticTrainingReplayAppendEntry {
    pub row_ordinal: u64,
    pub content_identity: [u64; 4],
    pub original_origin: SemanticTrainingViewOriginRecord,
    pub stable_intent: [u64; 4],
    pub result_receipt_digest: [u64; 4],
    pub payload_offset_bytes: u64,
    pub payload_length_bytes: u64,
    pub disposition: u64,
}

// SAFETY: the native append queue ABI contains only integer and identity words.
unsafe impl DeviceRepr for SemanticTrainingReplayAppendHeader {}
unsafe impl DeviceRepr for SemanticTrainingReplayAppendEntry {}

const _: () = assert!(size_of::<SemanticTrainingReplayAppendHeader>() == 80);
const _: () = assert!(size_of::<SemanticTrainingReplayAppendEntry>() == 480);

struct TrainingPublicationBinding {
    word: DeviceMemoryView<u64>,
    directories: [DeviceMemoryView<u64>; 2],
    storage: DeviceMemoryView<u64>,
    entries: [DeviceMemoryView<u8>; 2],
    payloads: [DeviceMemoryView<u8>; 2],
}

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
    /// candidate ordinal names a fresh re-execution or the committed-append marker.
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
    cold_work: u64,
    initial_row_count: u64,
    objective_template: u64,
    groups_template: u64,
    group_members_template: u64,
    objective: u64,
    groups: u64,
    group_members: u64,
    group_member_capacity: u64,
    publication_word: u64,
    directories: [u64; 2],
    directory_count: u64,
    publication_storage: u64,
    publication_storage_count: u64,
    append_entries: [u64; 2],
    append_entries_bytes: [u64; 2],
    append_payloads: [u64; 2],
    append_payload_bytes: [u64; 2],
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
    objective: TrackedCudaSlice<SemanticTrainingObjectiveRecord>,
    groups: TrackedCudaSlice<SemanticTrainingObjectiveGroupRecord>,
    group_members: TrackedCudaSlice<u64>,
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
    #[cfg(feature = "semantic-policy")]
    critic_terms: TrackedCudaSlice<f32>,
    #[cfg(feature = "semantic-policy")]
    critic_total: TrackedCudaSlice<f32>,
}

/// Retained native result of device selection. The views are fixed-capacity
/// CUDA ports; the resident selection record supplies their logical extent.
pub struct SemanticSelectedTrainingView {
    arena: Arc<SemanticTrainingViewArena>,
    _origin_candidates: Option<Arc<TrackedCudaSlice<SemanticTrainingViewOriginRecord>>>,
    storage: SelectedTrainingViewStorage,
}

#[cfg(feature = "semantic-policy")]
#[derive(Clone, Copy)]
pub(crate) struct FrozenPolicyGroupMember {
    pub ordinal: u64,
    pub identity: [u64; 4],
    pub source_identity: [u64; 4],
    pub bytes_identity: Identity256,
    pub content_identity: [u64; 4],
    pub origin: SemanticTrainingViewOriginRecord,
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

impl SemanticTrainingViewPort {
    /// Canonical public order of the complete selected roster.
    pub const ALL: [Self; 17] = [
        Self::Selection,
        Self::RosterRows,
        Self::Objective,
        Self::ObjectiveGroups,
        Self::ObjectiveGroupMembers,
        Self::Canaries,
        Self::TokenIds,
        Self::MaskLabels,
        Self::MaskWeights,
        Self::AutoregressiveLabels,
        Self::RetentionLabels,
        Self::BranchLabels,
        Self::BranchIds,
        Self::SourceSlots,
        Self::LogicalPositions,
        Self::Kinds,
        Self::Parents,
    ];
}

impl SemanticSelectedTrainingView {
    pub fn capacity(&self) -> usize {
        self.arena.capacity
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn account_allocations(
        &self,
        allocations: &mut Vec<crate::memory::DeviceAllocationProvenance>,
    ) -> Result<(), SemanticTransitionError> {
        macro_rules! account_view {
            ($view:expr) => {{
                let provenance = $view
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
        macro_rules! account {
            ($slice:expr) => {
                account_view!($slice.view())
            };
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
        account!(self.storage.objective);
        account!(self.storage.groups);
        account!(self.storage.group_members);
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
        #[cfg(feature = "semantic-policy")]
        account!(self.storage.critic_terms);
        #[cfg(feature = "semantic-policy")]
        account!(self.storage.critic_total);
        if let Some(publication) = &self.arena.publication {
            account_view!(publication.word);
            for directory in &publication.directories {
                account_view!(directory);
            }
            account_view!(publication.storage);
            for entries in &publication.entries {
                account_view!(entries);
            }
            for payload in &publication.payloads {
                account_view!(payload);
            }
        }
        Ok(())
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
        self.storage.objective.view()
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn objective_groups(
        &self,
    ) -> DeviceMemoryView<SemanticTrainingObjectiveGroupRecord> {
        self.storage.groups.view()
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn objective_group_members(&self) -> DeviceMemoryView<u64> {
        self.storage.group_members.view()
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn actor_group_member_count(&self) -> u64 {
        self.arena.actor_group_member_count
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn actor_group_members(&self) -> &[FrozenPolicyGroupMember] {
        &self.arena.actor_group_members
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn edit_group_members(&self) -> &[FrozenPolicyGroupMember] {
        &self.arena.edit_group_members
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn actor_group_member_index(
        &self,
        ordinal: u64,
        row: &SemanticTrainingViewRow,
    ) -> Result<usize, SemanticTransitionError> {
        let descriptor = validate_row(
            usize::try_from(ordinal)
                .map_err(|_| input_error("actor group ordinal exceeds host address space"))?,
            row,
            0,
            &self.arena.training_domain,
        )?;
        self.actor_group_members()
            .iter()
            .position(|member| {
                member.ordinal == ordinal
                    && member.identity == descriptor.identity
                    && member.source_identity == descriptor.source_identity
                    && member.bytes_identity == row.bytes_identity
                    && member.content_identity == descriptor.content_identity
                    && member.origin == descriptor.origin
            })
            .ok_or_else(|| input_error("original replay is not an exact frozen actor group member"))
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn edit_group_member_index(
        &self,
        ordinal: u64,
        row: &SemanticTrainingViewRow,
    ) -> Result<Option<usize>, SemanticTransitionError> {
        let descriptor = validate_row(
            usize::try_from(ordinal)
                .map_err(|_| input_error("edit group ordinal exceeds host address space"))?,
            row,
            0,
            &self.arena.training_domain,
        )?;
        Ok(self.edit_group_members().iter().position(|member| {
            member.ordinal == ordinal
                && member.identity == descriptor.identity
                && member.source_identity == descriptor.source_identity
                && member.bytes_identity == row.bytes_identity
                && member.content_identity == descriptor.content_identity
                && member.origin == descriptor.origin
        }))
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn critic_term(&self, index: usize) -> DeviceMemoryView<f32> {
        self.storage.critic_terms.slice(index..index + 1)
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn critic_terms(&self) -> DeviceMemoryView<f32> {
        self.storage.critic_terms.view()
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn critic_total(&self) -> DeviceMemoryView<f32> {
        self.storage.critic_total.view()
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
        // SAFETY: every retained port below has its original typed allocation;
        // its original scalar layout comes from the same immutable arena.
        let view = unsafe {
            match port {
                SemanticTrainingViewPort::Selection => self.storage.selection.view().cast::<u8>(),
                SemanticTrainingViewPort::RosterRows => {
                    self.storage.roster_rows.view().cast::<u8>()
                }
                SemanticTrainingViewPort::Objective => self.storage.objective.view().cast::<u8>(),
                SemanticTrainingViewPort::ObjectiveGroups => {
                    self.storage.groups.view().cast::<u8>()
                }
                SemanticTrainingViewPort::ObjectiveGroupMembers => {
                    self.storage.group_members.view().cast::<u8>()
                }
                SemanticTrainingViewPort::Canaries => self.arena.canaries.view().cast::<u8>(),
                SemanticTrainingViewPort::TokenIds => self.storage.token_ids.view().cast::<u8>(),
                SemanticTrainingViewPort::MaskLabels => {
                    self.storage.mask_labels.view().cast::<u8>()
                }
                SemanticTrainingViewPort::MaskWeights => {
                    self.storage.mask_weights.view().cast::<u8>()
                }
                SemanticTrainingViewPort::AutoregressiveLabels => {
                    self.storage.ar_labels.view().cast::<u8>()
                }
                SemanticTrainingViewPort::RetentionLabels => {
                    self.storage.retention_labels.view().cast::<u8>()
                }
                SemanticTrainingViewPort::BranchLabels => {
                    self.storage.branch_labels.view().cast::<u8>()
                }
                SemanticTrainingViewPort::BranchIds => self.storage.branch_ids.view().cast::<u8>(),
                SemanticTrainingViewPort::SourceSlots => {
                    self.storage.source_slots.view().cast::<u8>()
                }
                SemanticTrainingViewPort::LogicalPositions => {
                    self.storage.logical_positions.view().cast::<u8>()
                }
                SemanticTrainingViewPort::Kinds => self.storage.kinds.view().cast::<u8>(),
                SemanticTrainingViewPort::Parents => self.storage.parents.view().cast::<u8>(),
            }
        }
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let (shape, strides, dtype) = self.arena.port_layout(port)?;
        Ok((view, shape, strides, dtype))
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
        self.enqueue(selected_view, 0, [0; 4], Some(coordinates), None)
    }

    fn enqueue(
        &self,
        selected_view: DeviceMemoryView<u8>,
        cursor: u64,
        training_rng: [u64; 4],
        coordinates: Option<DeviceMemoryView<u64>>,
        cold_work: Option<&DeviceMemoryView<u64>>,
    ) -> Result<(), SemanticTransitionError> {
        self.enqueue_with_original(
            selected_view,
            cursor,
            training_rng,
            coordinates,
            cold_work,
            #[cfg(feature = "semantic-policy")]
            None,
        )
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn enqueue_original(
        &self,
        selected_view: DeviceMemoryView<u8>,
        cursor: u64,
        training_rng: [u64; 4],
        cold_work: Option<&DeviceMemoryView<u64>>,
        commands: &mut [OriginalNativeCommand; 2],
        poisoned: &mut bool,
    ) -> Result<(), SemanticTransitionError> {
        self.enqueue_with_original(
            selected_view,
            cursor,
            training_rng,
            None,
            cold_work,
            Some((commands, poisoned)),
        )
    }

    fn enqueue_with_original(
        &self,
        selected_view: DeviceMemoryView<u8>,
        cursor: u64,
        training_rng: [u64; 4],
        coordinates: Option<DeviceMemoryView<u64>>,
        cold_work: Option<&DeviceMemoryView<u64>>,
        #[cfg(feature = "semantic-policy")] original: Option<(
            &mut [OriginalNativeCommand; 2],
            &mut bool,
        )>,
    ) -> Result<(), SemanticTransitionError> {
        self.arena.require_authenticated_appends()?;
        let publication = self.arena.publication.as_ref().ok_or_else(|| {
            input_error(
                "training replay requires its original publication binding before selection",
            )
        })?;
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
            cold_work: cold_work.map_or(0, |work| *work.device_ptr()),
            initial_row_count: u64::try_from(self.arena.initial_row_count)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            objective_template: self.arena.objective.device_ptr_value(),
            groups_template: self.arena.groups.device_ptr_value(),
            group_members_template: self.arena.group_members.device_ptr_value(),
            objective: self.storage.objective.device_ptr_value(),
            groups: self.storage.groups.device_ptr_value(),
            group_members: self.storage.group_members.device_ptr_value(),
            group_member_capacity: self.arena.group_members.len() as u64,
            publication_word: *publication.word.device_ptr(),
            directories: [
                *publication.directories[0].device_ptr(),
                *publication.directories[1].device_ptr(),
            ],
            directory_count: (publication.directories[0].len() / 16) as u64,
            publication_storage: *publication.storage.device_ptr(),
            publication_storage_count: (publication.storage.len() / 3) as u64,
            append_entries: [
                *publication.entries[0].device_ptr(),
                *publication.entries[1].device_ptr(),
            ],
            append_entries_bytes: [
                publication.entries[0].len() as u64,
                publication.entries[1].len() as u64,
            ],
            append_payloads: [
                *publication.payloads[0].device_ptr(),
                *publication.payloads[1].device_ptr(),
            ],
            append_payload_bytes: [
                publication.payloads[0].len() as u64,
                publication.payloads[1].len() as u64,
            ],
        };
        let record = || {
            let mut recorder = self.arena.domain.new_strict_recorder();
            recorder.read(&self.arena.descriptors);
            recorder.read(&self.arena.raw);
            recorder.read(&self.arena.objective);
            recorder.read(&self.arena.groups);
            recorder.read(&self.arena.group_members);
            recorder.read(&publication.word);
            recorder.read(&publication.directories[0]);
            recorder.read(&publication.directories[1]);
            recorder.read(&publication.storage);
            for bank in 0..2 {
                recorder.read(&publication.entries[bank]);
                recorder.read(&publication.payloads[bank]);
            }
            recorder.read(&selected_view);
            if let Some(candidates) = &self._origin_candidates {
                recorder.read(candidates.as_ref());
            }
            if let Some(coordinates) = &coordinates {
                recorder.read(coordinates);
            }
            if let Some(work) = cold_work {
                recorder.read_write(work);
            }
            recorder.write(&self.storage.selection);
            recorder.write(&self.storage.roster_rows);
            recorder.write(&self.storage.objective);
            recorder.write(&self.storage.groups);
            recorder.write(&self.storage.group_members);
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
            recorder
        };
        let select = self.arena.select.clone();
        let gather = self.arena.gather.clone();
        let gather_grid = u32::try_from(self.arena.row_count)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        #[cfg(feature = "semantic-policy")]
        if let Some((commands, poisoned)) = original {
            let [select_command, gather_command] = commands;
            for (command, kernel, grid) in [
                (select_command, select, 1),
                (gather_command, gather, gather_grid),
            ] {
                command.run(
                    &self.arena.domain,
                    poisoned,
                    record(),
                    |enqueue, entered, submitted| {
                        let argument = launch.into_kernel_param_storage();
                        let mut parameters = [argument.as_kernel_param()];
                        // SAFETY: this original selection retains every fixed port,
                        // source view and publication allocation registered above.
                        unsafe {
                            kernel.launch_raw_in_original(
                                enqueue,
                                LaunchConfig {
                                    grid_dim: (grid, 1, 1),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                &mut parameters,
                                false,
                                entered,
                                submitted,
                            )
                        }
                        .map_err(|error| xlog_core::XlogError::Kernel(error.to_string()))
                    },
                )?;
            }
            return Ok(());
        }
        let enqueued = unsafe {
            self.arena.domain.enqueue(record(), |stream| {
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

/// Cold-reserved replay storage used by the native Update selector.
pub(crate) struct SemanticTrainingViewArena {
    training_domain: SemanticTrainingDomain,
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
    #[cfg(feature = "semantic-policy")]
    actor_group_member_count: u64,
    #[cfg(feature = "semantic-policy")]
    actor_group_members: Box<[FrozenPolicyGroupMember]>,
    #[cfg(feature = "semantic-policy")]
    edit_group_members: Box<[FrozenPolicyGroupMember]>,
    row_count: usize,
    initial_row_count: usize,
    row_bytes: usize,
    publication: Option<TrainingPublicationBinding>,
    replay_capacity: Option<SemanticReplayAppendBinding>,
    appends_authenticated: AtomicBool,
    capacity: usize,
}

impl SemanticTrainingViewArena {
    pub(crate) fn original_count(&self) -> usize {
        self.initial_row_count
    }

    pub(crate) fn row_capacity(&self) -> usize {
        self.row_count
    }

    pub(crate) fn raw_capacity_bytes(&self) -> usize {
        self.raw.len()
    }

    pub(crate) fn replay_capacity(&self) -> Option<&SemanticReplayAppendBinding> {
        self.replay_capacity.as_ref()
    }

    pub(crate) fn port_layout(
        &self,
        port: SemanticTrainingViewPort,
    ) -> Result<TrainingViewPortLayout, SemanticTransitionError> {
        let extent = |length: usize| {
            i64::try_from(length).map_err(|_| SemanticTransitionError::GenerationExhausted)
        };
        let words = |bytes: usize| extent(bytes / size_of::<u64>());
        let vector = |length| -> Result<TrainingViewPortLayout, SemanticTransitionError> {
            Ok((vec![extent(length)?], vec![1], (1, 64)))
        };
        let records = |length, bytes| -> Result<TrainingViewPortLayout, SemanticTransitionError> {
            let width = words(bytes)?;
            Ok((vec![extent(length)?, width], vec![width, 1], (1, 64)))
        };
        match port {
            SemanticTrainingViewPort::Selection => {
                vector(size_of::<SemanticTrainingViewSelection>() / size_of::<u64>())
            }
            SemanticTrainingViewPort::RosterRows => {
                records(self.row_count, size_of::<SemanticTrainingRosterRow>())
            }
            SemanticTrainingViewPort::Objective => {
                vector(size_of::<SemanticTrainingObjectiveRecord>() / size_of::<u64>())
            }
            SemanticTrainingViewPort::ObjectiveGroups => records(
                self.groups.len(),
                size_of::<SemanticTrainingObjectiveGroupRecord>(),
            ),
            SemanticTrainingViewPort::ObjectiveGroupMembers => vector(self.group_members.len()),
            SemanticTrainingViewPort::Canaries => records(
                self.canaries.len(),
                size_of::<SemanticTrainingCanaryRecord>(),
            ),
            _ => {
                let rows = extent(self.row_count)?;
                let capacity = extent(self.capacity)?;
                let dtype = if port == SemanticTrainingViewPort::MaskWeights {
                    (2, 32)
                } else {
                    (0, 64)
                };
                Ok((vec![rows, capacity], vec![capacity, 1], dtype))
            }
        }
    }

    pub(crate) fn selection_native_work_ceiling(
        &self,
        origin_candidates: Option<&TrackedCudaSlice<SemanticTrainingViewOriginRecord>>,
    ) -> Result<[u64; 9], SemanticTransitionError> {
        let publication = self.publication.as_ref().ok_or_else(|| {
            input_error("training selection work requires its original publication binding")
        })?;
        let extent = |value: usize| {
            u64::try_from(value).map_err(|_| SemanticTransitionError::GenerationExhausted)
        };
        let add = |left: u64, right: u64| {
            left.checked_add(right)
                .ok_or(SemanticTransitionError::GenerationExhausted)
        };
        let multiply = |left: u64, right: u64| {
            left.checked_mul(right)
                .ok_or(SemanticTransitionError::GenerationExhausted)
        };
        let sum = |terms: &[u64]| {
            terms
                .iter()
                .try_fold(0u64, |total, &value| add(total, value))
        };
        let rows = extent(self.row_count)?;
        let original_rows = extent(self.initial_row_count)?;
        let append_slots = rows
            .checked_sub(original_rows)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let members = extent(self.group_members.len())?;
        let candidates = extent(origin_candidates.map_or(0, TrackedCudaSlice::len))?;
        // These are the charged visits of the same select/gather kernels. Each
        // possible branch uses the finite original roster, not the active prefix.
        let table_slots = sum(&[
            extent(publication.directories[0].len() / 16)?,
            members,
            extent(self.row_bytes)?,
            rows,
            multiply(original_rows, candidates)?,
            multiply(append_slots, append_slots)?,
        ])?;
        let canonical_bytes = sum(&[
            multiply(2, extent(size_of::<SemanticTrainingViewSelection>())?)?,
            multiply(
                multiply(2, rows)?,
                extent(size_of::<SemanticTrainingRosterRow>())?,
            )?,
            multiply(members, 2 * size_of::<u64>() as u64)?,
            multiply(
                extent(self.groups.len())?,
                extent(size_of::<SemanticTrainingObjectiveGroupRecord>())?,
            )?,
            extent(size_of::<SemanticTrainingObjectiveRecord>())?,
            multiply(rows, size_of::<u64>() as u64)?,
            2 * size_of::<u64>() as u64,
            multiply(
                multiply(multiply(2, rows)?, extent(self.capacity)?)?,
                TRAINING_VIEW_ROW_BYTES as u64,
            )?,
        ])?;
        // NativeWorkEvent order: Command, TableSlot, ChainLink, Category,
        // SortComparison, CdfStep, PhiloxRound, ShaBlock, CanonicalByte.
        Ok([0, table_slots, 0, 0, 0, 0, 0, 0, canonical_bytes])
    }

    pub(crate) fn require_append_authentication(&self) {
        self.appends_authenticated.store(false, Ordering::Release);
    }

    pub(crate) fn authenticate_appends(&self) {
        self.appends_authenticated.store(true, Ordering::Release);
    }

    fn require_authenticated_appends(&self) -> Result<(), SemanticTransitionError> {
        if !self.appends_authenticated.load(Ordering::Acquire) {
            return Err(input_error(
                "restored replay appends require their original receipts and canonical rows to be authenticated",
            ));
        }
        Ok(())
    }

    pub(crate) fn bind_publication(
        &mut self,
        publication_word: DeviceMemoryView<u64>,
        directories: [DeviceMemoryView<u64>; 2],
        storage: DeviceMemoryView<u64>,
        entries: [DeviceMemoryView<u8>; 2],
        payloads: [DeviceMemoryView<u8>; 2],
    ) -> Result<(), SemanticTransitionError> {
        let queue_bytes = self
            .row_count
            .checked_sub(self.initial_row_count)
            .and_then(|slots| slots.checked_mul(size_of::<SemanticTrainingReplayAppendEntry>()))
            .and_then(|bytes| bytes.checked_add(size_of::<SemanticTrainingReplayAppendHeader>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if self.publication.is_some()
            || publication_word.len() != 1
            || directories[0].len() != directories[1].len()
            || directories[0].is_empty()
            || !directories[0].len().is_multiple_of(16)
            || storage.is_empty()
            || !storage.len().is_multiple_of(3)
            || entries.iter().any(|view| view.len() < queue_bytes)
        {
            return Err(input_error(
                "training replay requires its exact original publication views",
            ));
        }
        self.publication = Some(TrainingPublicationBinding {
            word: publication_word,
            directories,
            storage,
            entries,
            payloads,
        });
        Ok(())
    }

    pub(crate) fn append_storage(
        &self,
    ) -> (
        DeviceMemoryView<TrainingViewRowDescriptor>,
        DeviceMemoryView<u8>,
    ) {
        (self.descriptors.view(), self.raw.view())
    }

    pub(crate) fn prepare_append_row(
        &self,
        ordinal: usize,
        row: &SemanticTrainingViewRow,
    ) -> Result<TrainingViewRowDescriptor, SemanticTransitionError> {
        if ordinal < self.initial_row_count
            || ordinal >= self.row_count
            || row.basis != SemanticTrainingViewBasis::Episode
            || row.bytes.len() != self.row_bytes
        {
            return Err(input_error(
                "replay append row is outside its reserved non-actor episode slot",
            ));
        }
        validate_row(
            ordinal,
            row,
            ordinal
                .checked_mul(self.row_bytes)
                .ok_or(SemanticTransitionError::GenerationExhausted)?,
            &self.training_domain,
        )
    }

    pub(crate) fn prepare_rejected_row(
        &self,
        row: &SemanticTrainingViewRow,
    ) -> Result<TrainingViewRowDescriptor, SemanticTransitionError> {
        if row.basis != SemanticTrainingViewBasis::Episode || row.bytes.len() != self.row_bytes {
            return Err(input_error(
                "rejected replay outcome differs from its original episode geometry",
            ));
        }
        let mut descriptor = validate_row(0, row, 0, &self.training_domain)?;
        descriptor.ordinal = u64::MAX;
        Ok(descriptor)
    }
    #[cfg(feature = "semantic-policy")]
    pub(crate) fn training_domain_identity(&self) -> Identity256 {
        self.training_domain.identity()
    }

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
        let bytes = size_of::<SemanticTrainingViewSelection>()
            .checked_add(roster_bytes)
            .and_then(|bytes| bytes.checked_add(port_bytes))
            .and_then(|bytes| bytes.checked_add(size_of::<SemanticTrainingObjectiveRecord>()))
            .and_then(|bytes| {
                self.groups
                    .len()
                    .checked_mul(size_of::<SemanticTrainingObjectiveGroupRecord>())
                    .and_then(|group_bytes| bytes.checked_add(group_bytes))
            })
            .and_then(|bytes| {
                self.group_members
                    .len()
                    .checked_mul(size_of::<u64>())
                    .and_then(|member_bytes| bytes.checked_add(member_bytes))
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        #[cfg(feature = "semantic-policy")]
        let bytes = bytes
            .checked_add(
                (self.actor_group_members.len() + 1)
                    .checked_mul(size_of::<f32>())
                    .ok_or(SemanticTransitionError::GenerationExhausted)?,
            )
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        Ok(bytes)
    }

    pub(crate) fn allocate_selection_reserved(
        self: &Arc<Self>,
        reservation: &mut crate::memory::GpuMemoryReservation,
        origin_candidates: Option<Arc<TrackedCudaSlice<SemanticTrainingViewOriginRecord>>>,
    ) -> Result<SemanticSelectedTrainingView, SemanticTransitionError> {
        self.require_authenticated_appends()?;
        let storage = SelectedTrainingViewStorage {
            selection: reservation
                .alloc::<SemanticTrainingViewSelection>(1)
                .map_err(|error| runtime_error("training-view selection allocation", error))?,
            roster_rows: reservation
                .alloc::<SemanticTrainingRosterRow>(self.row_count)
                .map_err(|error| runtime_error("training-view roster allocation", error))?,
            objective: reservation
                .alloc::<SemanticTrainingObjectiveRecord>(1)
                .map_err(|error| runtime_error("selected training objective allocation", error))?,
            groups: reservation
                .alloc::<SemanticTrainingObjectiveGroupRecord>(self.groups.len())
                .map_err(|error| runtime_error("selected training group allocation", error))?,
            group_members: reservation
                .alloc::<u64>(self.group_members.len())
                .map_err(|error| runtime_error("selected training member allocation", error))?,
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
            #[cfg(feature = "semantic-policy")]
            critic_terms: reservation
                .alloc::<f32>(self.actor_group_members.len())
                .map_err(|error| runtime_error("group critic term allocation", error))?,
            #[cfg(feature = "semantic-policy")]
            critic_total: reservation
                .alloc::<f32>(1)
                .map_err(|error| runtime_error("group critic scalar allocation", error))?,
        };
        Ok(SemanticSelectedTrainingView {
            arena: Arc::clone(self),
            _origin_candidates: origin_candidates,
            storage,
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "arena retains the task, numerical law and original input domain"
    )]
    pub(crate) fn allocate(
        provider: &Arc<CudaKernelProvider>,
        domain: &ResidentExecutionDomain,
        rows: Vec<SemanticTrainingViewRow>,
        objective: SemanticTrainingObjective,
        training_domain: &SemanticTrainingDomain,
        task_identity: Identity256,
        task_content: SemanticTaskContentIdentity,
        expected_truth: [SemanticTruth; 3],
        replay_binding: Option<SemanticReplayAppendBinding>,
    ) -> Result<Arc<Self>, SemanticTransitionError> {
        validate_execution_domain(provider, domain)
            .map_err(|error| runtime_error("training-view domain validation", error))?;
        if rows.is_empty() {
            return Err(input_error(
                "training-view arena requires at least one admitted row",
            ));
        }
        let replay_capacity = match &replay_binding {
            Some(binding) => SemanticTrainingReplayCapacity {
                row_capacity: binding.row_capacity,
                raw_byte_capacity: binding.raw_byte_capacity,
                append_groups: binding
                    .append_groups
                    .iter()
                    .map(|groups| {
                        groups
                            .iter()
                            .map(|kind| match kind {
                                1 => Ok(SemanticTrainingObjectiveGroupKind::MaskedLanguage),
                                2 => Ok(SemanticTrainingObjectiveGroupKind::AutoregressiveLanguage),
                                3 => Ok(SemanticTrainingObjectiveGroupKind::Semantic),
                                4 => Ok(SemanticTrainingObjectiveGroupKind::Edit),
                                5 => Ok(SemanticTrainingObjectiveGroupKind::Execution),
                                _ => Err(input_error(
                                    "replay append accepts only original non-actor episode groups",
                                )),
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            },
            None => SemanticTrainingReplayCapacity {
                row_capacity: rows.len(),
                raw_byte_capacity: rows.iter().try_fold(0usize, |bytes, row| {
                    bytes
                        .checked_add(row.bytes.len())
                        .ok_or(SemanticTransitionError::GenerationExhausted)
                })?,
                append_groups: Vec::new(),
            },
        };
        training_domain.validate()?;
        let record_limit = training_domain.record_limit();
        let mut episode_count = 0u64;
        let mut anchor_count = 0u64;
        let mut content_identities = std::collections::BTreeSet::new();
        for row in &rows {
            let count = if row.basis == SemanticTrainingViewBasis::Episode {
                &mut episode_count
            } else {
                &mut anchor_count
            };
            *count = count
                .checked_add(1)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            if *count > record_limit || !content_identities.insert(*row.content_identity.as_bytes())
            {
                return Err(input_error(
                    "training rows exceed the original input bound or repeat content",
                ));
            }
        }
        let initial_row_count = rows.len();
        let append_slots = replay_capacity
            .row_capacity
            .checked_sub(initial_row_count)
            .ok_or_else(|| input_error("replay capacity is smaller than its original roster"))?;
        if replay_capacity.append_groups.len() != append_slots
            || episode_count
                .checked_add(
                    u64::try_from(append_slots)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                )
                .is_none_or(|count| count > record_limit)
            || replay_capacity.append_groups.iter().any(|groups| {
                groups.is_empty()
                    || groups.iter().any(|kind| {
                        *kind as u64 > SemanticTrainingObjectiveGroupKind::Execution as u64
                    })
                    || groups
                        .windows(2)
                        .any(|pair| pair[0] as u64 >= pair[1] as u64)
            })
        {
            return Err(input_error(
                "replay append capacity requires bounded original-domain non-actor groups",
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
        #[cfg(feature = "semantic-policy")]
        let mut row_bytes_identities = Vec::with_capacity(rows.len());
        let mut capacity = 0usize;
        for (ordinal, row) in rows.into_iter().enumerate() {
            let descriptor = validate_row(ordinal, &row, raw.len(), training_domain)?;
            #[cfg(feature = "semantic-policy")]
            row_bytes_identities.push(row.bytes_identity);
            capacity = capacity.max(descriptor.window as usize);
            raw.extend_from_slice(&row.bytes);
            descriptors.push(descriptor);
        }
        let row_bytes = raw
            .len()
            .checked_div(initial_row_count)
            .filter(|stride| stride.checked_mul(initial_row_count) == Some(raw.len()))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let required_raw_bytes = replay_capacity
            .row_capacity
            .checked_mul(row_bytes)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if replay_capacity.raw_byte_capacity < required_raw_bytes
            || !replay_capacity.raw_byte_capacity.is_multiple_of(8)
            || descriptors
                .iter()
                .any(|row| row.raw_bytes != row_bytes as u64)
        {
            return Err(input_error(
                "replay byte capacity differs from its fixed original training geometry",
            ));
        }
        #[cfg(feature = "semantic-policy")]
        let (mut objective, mut groups, mut group_members, canaries, protected_members) =
            validate_objective(
                objective,
                &descriptors,
                &raw,
                capacity,
                task_identity,
                task_content,
                expected_truth,
                record_limit,
            )?;
        #[cfg(not(feature = "semantic-policy"))]
        let (mut objective, mut groups, mut group_members, canaries, _) = validate_objective(
            objective,
            &descriptors,
            &raw,
            capacity,
            task_identity,
            task_content,
            expected_truth,
            record_limit,
        )?;
        #[cfg(feature = "semantic-policy")]
        let actor_group = groups
            .iter()
            .find(|group| group.kind == SemanticTrainingObjectiveGroupKind::ActorCriticCost as u64)
            .ok_or_else(|| input_error("training objective has no actor-critic-cost group"))?;
        #[cfg(feature = "semantic-policy")]
        let actor_group_member_count = actor_group.member_count;
        #[cfg(feature = "semantic-policy")]
        let actor_group_members = {
            let start = usize::try_from(actor_group.member_offset)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            let count = usize::try_from(actor_group.member_count)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            group_members[start..start + count]
                .iter()
                .map(|&ordinal| {
                    let row = &descriptors[ordinal as usize];
                    FrozenPolicyGroupMember {
                        ordinal,
                        identity: row.identity,
                        source_identity: row.source_identity,
                        bytes_identity: row_bytes_identities[ordinal as usize],
                        content_identity: row.content_identity,
                        origin: row.origin,
                    }
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        };
        #[cfg(feature = "semantic-policy")]
        let edit_group_members = {
            let group = groups
                .iter()
                .find(|group| group.kind == SemanticTrainingObjectiveGroupKind::Edit as u64)
                .ok_or_else(|| input_error("training objective has no supervised-edit group"))?;
            let start = usize::try_from(group.member_offset)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            let count = usize::try_from(group.member_count)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            group_members[start..start + count]
                .iter()
                .map(|&ordinal| {
                    let row = &descriptors[ordinal as usize];
                    FrozenPolicyGroupMember {
                        ordinal,
                        identity: row.identity,
                        source_identity: row.source_identity,
                        bytes_identity: row_bytes_identities[ordinal as usize],
                        content_identity: row.content_identity,
                        origin: row.origin,
                    }
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        };
        let mut expanded_members = Vec::with_capacity(group_members.len());
        for group in &mut groups {
            let start = usize::try_from(group.member_offset)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            let count = usize::try_from(group.member_count)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            group.member_offset = u64::try_from(expanded_members.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            expanded_members.extend_from_slice(&group_members[start..start + count]);
            for (slot, assignments) in replay_capacity.append_groups.iter().enumerate() {
                if assignments.iter().any(|kind| *kind as u64 == group.kind) {
                    expanded_members.push(
                        u64::try_from(initial_row_count + slot)
                            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                    );
                }
            }
            group.member_count = u64::try_from(expanded_members.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?
                - group.member_offset;
        }
        group_members = expanded_members;
        let mut objective_hash = Sha256::new();
        objective_hash.update(b"xlog.semantic.training-replay-capacity.v1\0");
        for word in objective.identity {
            objective_hash.update(word.to_le_bytes());
        }
        objective_hash.update((replay_capacity.row_capacity as u64).to_le_bytes());
        objective_hash.update((replay_capacity.raw_byte_capacity as u64).to_le_bytes());
        for assignments in &replay_capacity.append_groups {
            objective_hash.update((assignments.len() as u64).to_le_bytes());
            for kind in assignments {
                objective_hash.update((*kind as u64).to_le_bytes());
            }
        }
        objective.identity =
            identity_words(Identity256::from_bytes(objective_hash.finalize().into()));
        descriptors.resize(
            replay_capacity.row_capacity,
            TrainingViewRowDescriptor::default(),
        );
        raw.resize(replay_capacity.raw_byte_capacity, 0);
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
            training_domain: training_domain.clone(),
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
            #[cfg(feature = "semantic-policy")]
            actor_group_member_count,
            #[cfg(feature = "semantic-policy")]
            actor_group_members,
            #[cfg(feature = "semantic-policy")]
            edit_group_members,
            row_count: replay_capacity.row_capacity,
            initial_row_count,
            row_bytes,
            publication: None,
            replay_capacity: replay_binding,
            appends_authenticated: AtomicBool::new(true),
            capacity,
        }))
    }

    pub(crate) fn enqueue_selection(
        self: &Arc<Self>,
        selected_view: DeviceMemoryView<u8>,
        cursor: u64,
        training_rng: [u64; 4],
        origin_candidates: Option<Arc<TrackedCudaSlice<SemanticTrainingViewOriginRecord>>>,
        cold_work: Option<&DeviceMemoryView<u64>>,
    ) -> Result<SemanticSelectedTrainingView, SemanticTransitionError> {
        let selected = self.prepare_selection(origin_candidates)?;
        selected.enqueue(selected_view, cursor, training_rng, None, cold_work)?;
        Ok(selected)
    }

    pub(crate) fn prepare_selection(
        self: &Arc<Self>,
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

#[expect(
    clippy::too_many_arguments,
    reason = "objective admission checks task, truth and original input bound"
)]
fn validate_objective(
    objective: SemanticTrainingObjective,
    rows: &[TrainingViewRowDescriptor],
    raw: &[u8],
    capacity: usize,
    task_identity: Identity256,
    task_content: SemanticTaskContentIdentity,
    expected_truth: [SemanticTruth; 3],
    record_limit: u64,
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
            || group.denominator > record_limit
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
            if expected_anchor.map_or(
                row.basis != SemanticTrainingViewBasis::Episode as u64,
                |basis| row.basis != basis as u64,
            ) || (policy_group
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
    if objective.coefficients.map(f32::to_bits)
        != frozen_training_coefficients(objective.evaluator_min, objective.evaluator_max)
            .map(f32::to_bits)
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

/// Check the original v2 byte layout before either cold admission or device allocation.
pub fn validate_training_view_layout(
    bytes: &[u8],
) -> Result<SemanticTrainingViewLayout, SemanticTransitionError> {
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
                .expect("bounded training-view header"),
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
    {
        return Err(input_error(
            "training-view row has invalid extent or geometry",
        ));
    }
    let branch_words = std::array::from_fn(|index| word(136 + index * 8));
    let shared_context_end = branch_words[0];
    if shared_context_end == 0 {
        if branch_words[1..].iter().any(|&value| value != 0) {
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
            let offset = 1 + branch * 5;
            let [tail_begin, tail_end, answer_begin, answer_end, answer_start] = branch_words
                [offset..offset + 5]
                .try_into()
                .expect("bounded branch words");
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
    Ok(SemanticTrainingViewLayout {
        window,
        source_length: word(104),
        block_size: word(112),
        prefix_extent: word(120),
        answer_start: word(128),
        branch_words,
    })
}

fn validate_row(
    ordinal: usize,
    row: &SemanticTrainingViewRow,
    raw_offset: usize,
    training_domain: &SemanticTrainingDomain,
) -> Result<TrainingViewRowDescriptor, SemanticTransitionError> {
    let bytes = &row.bytes;
    let layout = validate_training_view_layout(bytes)?;
    if layout.window as u64 != training_domain.window
        || layout.block_size != training_domain.block_size
        || layout
            .source_length
            .checked_mul(2)
            .is_none_or(|extent| extent > training_domain.window)
        || row.mask_policy != training_domain.mask_policy
        || row.training_seed != training_domain.training_seed
        || training_domain
            .manifests
            .binary_search_by(|manifest| {
                manifest
                    .identity
                    .as_bytes()
                    .cmp(row.data_manifest.as_bytes())
            })
            .is_err()
    {
        return Err(input_error(
            "training view lies outside its original manifest or loader domain",
        ));
    }
    let kinds_offset = TRAINING_VIEW_HEADER_BYTES + layout.window * 68 + (layout.window % 2) * 4;
    for slot in 0..layout.window {
        let token_offset = TRAINING_VIEW_HEADER_BYTES + slot * 8;
        let kind_offset = kinds_offset + slot * 8;
        let token = i64::from_le_bytes(bytes[token_offset..token_offset + 8].try_into().unwrap());
        let kind = i64::from_le_bytes(bytes[kind_offset..kind_offset + 8].try_into().unwrap());
        if (kind == 0 && u64::try_from(token).ok() != Some(training_domain.pad_id))
            || (kind == 2 && u64::try_from(token).ok() != Some(training_domain.mask_id))
        {
            return Err(input_error(
                "training view changed the original padding or mask token",
            ));
        }
    }
    if !raw_offset.is_multiple_of(8) {
        return Err(input_error("training-view row has invalid raw alignment"));
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
        window: u64::try_from(layout.window)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        source_length: layout.source_length,
        block_size: layout.block_size,
        prefix_extent: layout.prefix_extent,
        answer_start: layout.answer_start,
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
