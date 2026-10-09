use super::*;

/// Native custody of an original actor row and one Update's actual input owners.
/// This proof does not claim that a pending model generation has executed.
#[derive(Clone)]
pub struct SemanticPreparedActorRefresh {
    pub(super) inner: Arc<PreparedActorRefreshOwner>,
}

/// The original outer execution reached this child's initializer, refused it,
/// and authenticated its actual native prefix. No child model outcome is implied.
#[derive(Clone)]
pub struct SemanticActorRefreshInitializerRefusal {
    inner: Arc<ActorRefreshInitializerRefusalOwner>,
}

struct ActorRefreshInitializerRefusalOwner {
    issuer: Arc<()>,
    scope: Arc<()>,
    proof: Arc<PreparedActorRefreshOwner>,
    target_bank: usize,
    observation: ActorRefreshInitializerObservation,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct ActorRefreshInitializerObservation {
    marker: PreparedActorRefreshDevice,
    native_work: [u64; 11],
}

pub(super) struct PreparedActorRefreshOwner {
    pub(super) issuer: Arc<()>,
    pub(super) scope: Arc<()>,
    pub(super) step: u64,
    logical_update: u64,
    program: Identity256,
    phase: Identity256,
    pub(super) minimum_generation: u64,
    pub(super) rng: SemanticRngBinding,
    pub(super) member: crate::semantic_training_view::FrozenPolicyGroupMember,
    pub(super) inputs: Arc<PreparedStepInputs>,
    pub(super) models: Vec<Arc<ModelGenerationOwner>>,
    pub(super) provider: Arc<CudaKernelProvider>,
    pub(super) domain: ResidentExecutionDomain,
    pub(super) execution: Arc<Mutex<ActorRefreshParentExecution>>,
    pub(super) cancelled: Arc<AtomicBool>,
    pub(super) target: Arc<PublicationStorage>,
    work: Arc<ActorRefreshWork>,
    children: Mutex<[Option<Arc<()>>; 2]>,
    construction_pending: [Arc<AtomicBool>; 2],
}

pub(super) struct ActorRefreshConstructionBinding {
    pub(super) proof: SemanticPreparedActorRefresh,
    pub(super) bank: usize,
    pub(super) issuer: Arc<()>,
    pub(super) pending: Arc<AtomicBool>,
}

/// Only the original native segment submission may advance this owner.
/// An uncertain launch retains LaunchEntered until that same launch is joined.
pub(super) enum ActorRefreshParentExecution {
    Constructing,
    Frozen,
    LaunchEntered,
    Completed,
    NeverSubmitted,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ActorRefreshCaptureState {
    Staged,
    Recording,
    Recorded,
    Frozen,
}

/// The target reader and copied header are original prepared-step producers.
/// Acquired basis fields are written only inside the original Update bank.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct PreparedActorRefreshDevice {
    pub(super) abi: u64,
    pub(super) status: u64,
    target_control: u64,
    target_lease: u64,
    target_header: u64,
    pub(super) target_bank: u64,
    child_control: u64,
    member_ordinal: u64,
    minimum_generation: u64,
    stream_serial: u64,
    family_id: u64,
    proposal: u64,
    program: Identity256,
    phase: Identity256,
    original_origin: SemanticTrainingViewOriginRecord,
    pub(super) acquired_generation: u64,
    pub(super) model_geometry_digest: Identity256,
    pub(super) model_numerical_digest: Identity256,
}

// SAFETY: the padding-free C representation contains only initialized integers.
unsafe impl DeviceRepr for PreparedActorRefreshDevice {}

const _: () = assert!(size_of::<PreparedActorRefreshDevice>() == 584);

pub(super) struct StagedActorRefresh {
    pub(super) proof: SemanticPreparedActorRefresh,
    pub(super) target_bank: usize,
    pub(super) state: ActorRefreshCaptureState,
    pub(super) device: Arc<TrackedCudaSlice<PreparedActorRefreshDevice>>,
    pub(super) native_work: Arc<TrackedCudaSlice<u64>>,
    retirement_complete: bool,
    context: Arc<Mutex<ActorRefreshContextPreparation>>,
    context_issuance: Arc<()>,
    context_pending: Arc<AtomicBool>,
    native_ceiling: [u64; 9],
    initializer_observation: Arc<Mutex<Option<ActorRefreshInitializerObservation>>>,
    initializer_refusal: Option<SemanticActorRefreshInitializerRefusal>,
    cancellation: Option<SemanticPreparedSegmentNonSubmission>,
}

struct ActorRefreshContextPreparation {
    material: Vec<u8>,
    zero: crate::device::RetainedDeviceWrite<u64>,
    marker: Option<crate::device::RetainedDeviceWrite<PreparedActorRefreshDevice>>,
    admitted: [bool; 2],
    zero_done: bool,
    restore_done: bool,
    ready: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ActorRefreshWorkInput {
    abi: u64,
    entries: u64,
    count: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ActorRefreshWorkEntry {
    member_ordinal: u64,
    proof: u64,
    native_work: u64,
    states: [u64; 2],
    readers: [u64; 2],
    results: [u64; 2],
    parents: [u64; 2],
    model_work: [ModelWorkInput; 2],
    native_ceiling: [u64; 9],
    transition_ceilings: [[u64; 9]; 2],
}

// SAFETY: both original-work ABIs contain only initialized integer fields.
unsafe impl DeviceRepr for ActorRefreshWorkInput {}
unsafe impl DeviceRepr for ActorRefreshWorkEntry {}
const _: () = assert!(size_of::<ActorRefreshWorkInput>() == 24);
const _: () = assert!(size_of::<ActorRefreshWorkEntry>() == 352);

/// One original Update's complete actor roster, not a caller-supplied count.
/// Both conditional banks retain the same member order and distinct actual
/// producers; only the bank entered by the original graph can charge work.
pub(super) struct ActorRefreshWork {
    members: Vec<u64>,
    inputs: [TrackedCudaSlice<ActorRefreshWorkInput>; 2],
    entries: [TrackedCudaSlice<ActorRefreshWorkEntry>; 2],
    components: Mutex<[Vec<Option<ActorRefreshObservedComponent>>; 2]>,
    frozen: Mutex<[Vec<bool>; 2]>,
    uploads: Mutex<[Option<ActorRefreshMetadataUploads>; 2]>,
}

struct ActorRefreshMetadataUploads {
    entries: crate::device::RetainedDeviceWrite<ActorRefreshWorkEntry>,
    header: crate::device::RetainedDeviceWrite<ActorRefreshWorkInput>,
    admitted: [bool; 2],
    cursor: usize,
}

impl ActorRefreshMetadataUploads {
    fn complete(
        &mut self,
        provider: &CudaKernelProvider,
        stream: &Arc<CudaStream>,
        entry_bytes: usize,
    ) -> Result<(), SemanticTransitionError> {
        while self.cursor < 2 {
            let bytes = if self.cursor == 0 {
                entry_bytes
            } else {
                size_of::<ActorRefreshWorkInput>()
            };
            if !self.admitted[self.cursor] {
                provider.admit_launch_metadata_htod(bytes);
                self.admitted[self.cursor] = true;
            }
            let write = if self.cursor == 0 {
                if !self.entries.entered() {
                    self.entries
                        .enqueue(stream)
                        .map_err(|error| runtime_error("original actor metadata upload", error))?;
                }
                self.entries.resolve()
            } else {
                if !self.header.entered() {
                    self.header.enqueue(stream).map_err(|error| {
                        runtime_error("original actor metadata publication", error)
                    })?;
                }
                self.header.resolve()
            };
            write.map_err(|error| runtime_error("original actor metadata completion", error))?;
            self.cursor += 1;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(super) struct ActorRefreshObservedComponent {
    pub(super) member_ordinal: u64,
    pub(super) instance: Identity256,
    pub(super) marker: DeviceMemoryView<PreparedActorRefreshDevice>,
    pub(super) native_work: DeviceMemoryView<u64>,
    pub(super) states: [DeviceMemoryView<DeviceState>; 2],
    pub(super) readers: [DeviceMemoryView<PublicationLease>; 2],
    pub(super) results: [DeviceMemoryView<PreparedStepResult>; 2],
    pub(super) parents: [DeviceMemoryView<PublicationHeader>; 2],
    pub(super) model_actual: [DeviceMemoryView<u64>; 2],
    pub(super) model_recordings: [crate::semantic_work::FrozenModelWorkRecording; 2],
    pub(super) model_event_records: [Vec<ModelWorkEvent>; 2],
    pub(super) native_ceiling: [u64; 9],
    pub(super) transition_ceilings: [[u64; 9]; 2],
    initializer_observation: Arc<Mutex<Option<ActorRefreshInitializerObservation>>>,
    model_events: [DeviceMemoryView<ModelWorkEvent>; 2],
    model_inputs: [ModelWorkInput; 2],
}

impl ActorRefreshObservedComponent {
    pub(super) fn retain_initializer_refusal(
        &self,
        marker: PreparedActorRefreshDevice,
        native_work: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        if marker.abi != 1 || marker.status != 2 || marker.member_ordinal != self.member_ordinal {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let observation = ActorRefreshInitializerObservation {
            marker,
            native_work: native_work
                .try_into()
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
        };
        let mut original = self.initializer_observation.lock().map_err(|_| {
            publication_input_error("original actor refusal observation lock is poisoned")
        })?;
        if original.is_some_and(|original| original != observation) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        *original = Some(observation);
        Ok(())
    }
}

impl ActorRefreshWork {
    fn allocate(
        provider: &CudaKernelProvider,
        members: Vec<u64>,
    ) -> Result<Self, SemanticTransitionError> {
        let inputs = [
            allocate_publication(provider, 1)?,
            allocate_publication(provider, 1)?,
        ];
        let entries = [
            allocate_publication(provider, members.len())?,
            allocate_publication(provider, members.len())?,
        ];
        let empty = vec![ActorRefreshWorkEntry::default(); members.len()];
        for bank in 0..2 {
            upload_publication(provider, &empty, &entries[bank])?;
            upload_publication(
                provider,
                &[ActorRefreshWorkInput {
                    abi: 0,
                    entries: entries[bank].device_ptr_value(),
                    count: members.len() as u64,
                }],
                &inputs[bank],
            )?;
        }
        let components = [vec![None; members.len()], vec![None; members.len()]];
        let frozen = [vec![false; members.len()], vec![false; members.len()]];
        Ok(Self {
            members,
            inputs,
            entries,
            components: Mutex::new(components),
            frozen: Mutex::new(frozen),
            uploads: Mutex::new([None, None]),
        })
    }

    fn record(
        &self,
        bank: usize,
        recorder: &mut LaunchRecorder,
    ) -> Result<(), SemanticTransitionError> {
        recorder.read(&self.inputs[bank]);
        recorder.read(&self.entries[bank]);
        let components = self
            .components
            .lock()
            .map_err(|_| publication_input_error("original actor work custody lock is poisoned"))?;
        for component in &components[bank] {
            let component = component.as_ref().ok_or_else(|| {
                publication_input_error("original Update must retain every actor refresh producer")
            })?;
            recorder.read(&component.marker);
            recorder.read(&component.native_work);
            for index in 0..2 {
                recorder.read(&component.states[index]);
                recorder.read(&component.readers[index]);
                recorder.read(&component.results[index]);
                recorder.read(&component.parents[index]);
                recorder.read(&component.model_actual[index]);
                recorder.read(&component.model_events[index]);
            }
        }
        Ok(())
    }
}

fn actor_work_add(a: u64, b: u64) -> Result<u64, SemanticTransitionError> {
    a.checked_add(b)
        .ok_or(SemanticTransitionError::GenerationExhausted)
}

fn actor_work_mul(a: u64, b: u64) -> Result<u64, SemanticTransitionError> {
    a.checked_mul(b)
        .ok_or(SemanticTransitionError::GenerationExhausted)
}

fn actor_work_hash(ceiling: &mut [u64; 9], bytes: u64) -> Result<(), SemanticTransitionError> {
    let blocks = actor_work_add(bytes, 9 + 63)? / 64;
    native_work_bound::add_native_work_ceiling(ceiling, [0, 0, 0, 0, 0, 0, 0, blocks, 32])
}

/// Bound the same operation-10 initializer from its retained publication and
/// semantic arena geometry. These are producer loop ceilings, not execution
/// charges or a substitute for the enclosing Update's original allowance.
fn actor_initialize_native_work_ceiling(
    child: &SemanticTransitionSession,
) -> Result<[u64; 9], SemanticTransitionError> {
    let storage = child
        .publication
        .as_ref()
        .ok_or(SemanticTransitionError::NotBound)?;
    let directory = &storage.bank_templates[0];
    let range = |role| {
        directory
            .iter()
            .find(|range| range.role == role && range.index == 0)
            .ok_or(SemanticTransitionError::ObservationMismatch)
    };
    let n = directory.len() as u64;
    let t = storage.layouts.len() as u64;
    let contract = storage.contract_value;
    let rows = actor_work_add(contract.window_capacity, contract.feedback_capacity)?;
    let queries = child
        .task_evaluation
        .as_ref()
        .ok_or(SemanticTransitionError::NotBound)?
        .spec
        .statement_records
        .len() as u64;
    let mut ceiling = [0; 9];
    // Both original directories traverse the same canonical table, layout,
    // alias and storage validators. The complete step-input validator is a
    // conservative superset with an empty copy roster; reuse its counting law.
    let validation =
        native_work_bound::step_input_producer_native_work_ceiling(storage, &[], directory)?;
    native_work_bound::add_native_work_ceiling(&mut ceiling, validation)?;
    native_work_bound::add_native_work_ceiling(&mut ceiling, validation)?;
    let pairs = actor_work_mul(n, n.saturating_sub(1))? / 2;
    let window = contract.window_capacity;
    let source = actor_work_add(
        actor_work_add(
            window,
            actor_work_mul(window, window.saturating_sub(1))? / 2,
        )?,
        actor_work_add(
            actor_work_mul(window, contract.terminal_token_count)?,
            actor_work_add(window, actor_work_mul(window, actor_work_add(window, 1)?)?)?,
        )?,
    )?;
    let mut visits = actor_work_add(source, actor_work_add(contract.prefix_capacity, window)?)?;
    visits = actor_work_add(
        visits,
        actor_work_add(n, actor_work_add(actor_work_mul(n, n)?, pairs)?)?,
    )?;
    // Range/table/layout searches at sealing; same-slot model prefixes; the
    // logical and descriptor folds; source, feedback and model lookups.
    visits = actor_work_add(
        visits,
        actor_work_add(
            actor_work_mul(n, actor_work_add(actor_work_add(n, t)?, rows)?)?,
            pairs,
        )?,
    )?;
    visits = actor_work_add(
        visits,
        actor_work_add(
            actor_work_mul(8, n)?,
            actor_work_mul(queries, actor_work_add(n, 1 + 8 + 42 + 42)?)?,
        )?,
    )?;
    visits = actor_work_add(
        visits,
        actor_work_add(
            range(48)?.length_bytes / 8,
            actor_work_add(size_of::<PublicationHeader>() as u64 / 8, 2 + 3 + 4 * 3)?,
        )?,
    )?;
    let descriptor = child.descriptor();
    let statements = descriptor.arena[3];
    let supports = descriptor.arena[4];
    let versions = descriptor.arena[5];
    // Resident preflight scans each finite arena table and the original heads.
    // Noneditable restored queries perform at most two original truth calls
    // per query (initial query and feedback), each one statement-table scan.
    let truth_calls = if child
        .task_evaluation
        .as_ref()
        .ok_or(SemanticTransitionError::NotBound)?
        .spec
        .program
        .editable_program()
        .is_some()
    {
        0
    } else {
        actor_work_mul(2, queries)?
    };
    visits = actor_work_add(
        visits,
        actor_work_add(
            actor_work_add(statements, supports)?,
            actor_work_add(versions, statements)?,
        )?,
    )?;
    visits = actor_work_add(visits, actor_work_mul(truth_calls, statements)?)?;
    let graph_calls = actor_work_add(1, truth_calls)?;
    let receipt_bytes =
        size_of::<crate::semantic_hypergraph::SemanticResidentReceiptRecord>() as u64;
    // Clear/copy, refusal, view, statement, version, command and handle receipt
    // writes are separately bounded by the owner's fixed receipt extent.
    let graph_bytes = actor_work_mul(graph_calls, actor_work_mul(8, receipt_bytes)?)?;
    let intents = range(30)?
        .length_bytes
        .checked_sub(size_of::<IntentQueueHeader>() as u64)
        .filter(|bytes| bytes % size_of::<IntentEntry>() as u64 == 0)
        .map(|bytes| bytes / size_of::<IntentEntry>() as u64)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    visits = actor_work_add(
        visits,
        actor_work_add(actor_work_mul(intents, 1 + 5 * 4)?, 4)?,
    )?;
    native_work_bound::add_native_work_ceiling(
        &mut ceiling,
        [
            actor_work_add(1, graph_calls)?,
            visits,
            0,
            0,
            0,
            0,
            0,
            0,
            actor_work_add(
                graph_bytes,
                actor_work_add(
                    actor_work_mul(
                        contract.feedback_capacity,
                        size_of::<RawFeedbackRecord>() as u64,
                    )?,
                    actor_work_add(
                        size_of::<DeviceState>() as u64,
                        actor_work_add(
                            actor_work_mul(2, 6 * 8 + 4 + 3 * 8)?,
                            actor_work_add(
                                size_of::<PublicationHeader>() as u64 + (1 + 8 + 19) * 8 + 64 + 16,
                                actor_work_mul(intents, 25 * 8 + 80)?,
                            )?,
                        )?,
                    )?,
                )?,
            )?,
        ],
    )?;
    // Each current typed range is sealed once; directory entries and logical
    // folds are separate hashes. Full model backing hashes include padding and
    // sibling views; summing per view conservatively covers shared backings.
    for item in directory {
        let mut typed = native_work_bound::content_digest_ceiling(item.length_bytes, 21, false)?;
        typed[0] = 0;
        native_work_bound::add_native_work_ceiling(&mut ceiling, typed)?;
        actor_work_hash(&mut ceiling, size_of::<PublicationRange>() as u64)?;
        for _ in 0..2 {
            actor_work_hash(&mut ceiling, 80)?;
        }
        native_work_bound::add_native_work_ceiling(
            &mut ceiling,
            [0, 0, 0, 0, 0, 0, 0, 0, 2 * 80 + 32],
        )?;
        if matches!(item.role, 18..=25) {
            let allocation = storage
                .allocations
                .get(item.storage_slot as usize)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            actor_work_hash(&mut ceiling, allocation.len() as u64)?;
            for _ in 0..3 {
                actor_work_hash(&mut ceiling, 80)?;
            }
            native_work_bound::add_native_work_ceiling(
                &mut ceiling,
                [0, 0, 0, 0, 0, 0, 0, 0, 3 * 80 + 32],
            )?;
        }
        if matches!(item.role, 2 | 14 | 15) {
            let mut logical =
                native_work_bound::content_digest_ceiling(item.length_bytes, 21, false)?;
            logical[0] = 0;
            native_work_bound::add_native_work_ceiling(&mut ceiling, logical)?;
        }
    }
    actor_work_hash(&mut ceiling, size_of::<PublicationBank>() as u64)?;
    actor_work_hash(&mut ceiling, size_of::<PublicationHeader>() as u64)?;
    actor_work_hash(&mut ceiling, (32 * size_of::<SourceSlot>()) as u64)?;
    actor_work_hash(
        &mut ceiling,
        (COMPONENT_COUNT * size_of::<SemanticTransitionReceipt>()) as u64,
    )?;
    // The numerical and logical domains include their terminal NUL, as in the
    // actual native producer. The codebook can be hashed by action fallback and
    // logical publication; retain both original occurrences.
    actor_work_hash(
        &mut ceiling,
        b"xlog.semantic.model-numerics.v1\0".len() as u64,
    )?;
    actor_work_hash(&mut ceiling, 80)?;
    for _ in 0..2 {
        let prefix = b"xlog.semantic.logical-codebooks.v1\0".len() as u64 + 32 + 8;
        actor_work_hash(
            &mut ceiling,
            actor_work_add(prefix, range(48)?.length_bytes)?,
        )?;
        native_work_bound::add_native_work_ceiling(&mut ceiling, [0, 0, 0, 0, 0, 0, 0, 0, prefix])?;
    }
    actor_work_hash(&mut ceiling, contract.model_contract_layout.schema_bytes)?;
    let model_identity_bytes = b"xlog.semantic.model-identity.v1\0".len() as u64 + 32 + 8 + 32;
    actor_work_hash(&mut ceiling, model_identity_bytes)?;
    native_work_bound::add_native_work_ceiling(
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
            model_identity_bytes + 8 + 32 + 8 + 32 + 32 + 80,
        ],
    )?;
    // Partitioned payload SHA padding adds at most one extra block per intent
    // above a single full-payload hash. Identity and chain hashes are distinct.
    actor_work_hash(&mut ceiling, range(31)?.length_bytes)?;
    native_work_bound::add_native_work_ceiling(
        &mut ceiling,
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            actor_work_mul(intents, 1 + 4 + 2)?,
            actor_work_mul(intents, 3 * 32)?,
        ],
    )?;
    native_work_bound::native_work_ceiling_units(ceiling)?;
    Ok(ceiling)
}

pub(super) struct ActorRefreshInitialEpisodes {
    pub(super) index: Identity256,
    pub(super) episodes: Vec<(SemanticTrainingViewOriginRecord, Identity256)>,
    pub(super) actors: Vec<SemanticTrainingViewOriginRecord>,
}

pub(super) fn initial_episode_actor(
    material: &SemanticReplayMaterial,
    batch_identity: Identity256,
    batch_bytes: &[u8],
) -> Result<(SemanticTrainingViewOriginRecord, bool), SemanticTransitionError> {
    if material.kind != SemanticTransitionKind::Proposal
        || batch_bytes.len() != size_of::<SemanticActionBatchReceipt>()
        || Identity256::from_bytes(Sha256::digest(batch_bytes).into()) != batch_identity
    {
        return Err(publication_input_error(
            "initial actor requires its complete original native action batch",
        ));
    }
    // SAFETY: this complete integer-only native ABI has its exact extent above.
    let batch = unsafe {
        std::ptr::read_unaligned(batch_bytes.as_ptr().cast::<SemanticActionBatchReceipt>())
    };
    let header = material.predecessor.bank.header;
    let state = material.predecessor.bank.state;
    let rng = header.rng_binding()?;
    let successor = material.evidence.successor;
    let attempt = material.evidence.range(33)?.attempt()?;
    let next = u64::from(rng.proposal)
        .checked_add(1)
        .filter(|next| *next <= u64::from(u32::MAX))
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    if batch.abi != 1
        || batch.actor_eligible > 1
        || batch_identity != attempt.action_receipts_digest
        || batch.proposal != u64::from(rng.proposal)
        || batch.stream_serial != rng.stream_serial
        || batch.family_id != u64::from(rng.family_id)
        || batch.model_generation != u64::from(rng.model_generation)
        || batch.action_law_generation != state.catalogue_generation
        || batch.catalogue_digest != state.catalogue_digest
        || batch.component_count != COMPONENT_COUNT as u64
        || batch.candidate_count != 3
        || batch.winner > 2
        || batch.base_word != header.publication_word
        || batch.next_word != successor.word
        || batch.base_logical_digest != header.logical_digest
        || batch.rng_base != u64::from(rng.proposal) * COMPONENT_COUNT as u64
        || batch.rng_span != COMPONENT_COUNT as u64
        || batch.rng_successor != next * COMPONENT_COUNT as u64
        || batch.semantic_receipts_digest != attempt.semantic_receipts_digest
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    Ok((
        crate::semantic_training_view::origin_record(material.training_view_origin()?),
        batch.actor_eligible == 1,
    ))
}

impl SemanticTransitionSession {
    /// Bind every original sealed episode in original index order. The trusted
    /// typed importer verifies complete index coverage before this call; native
    /// batch receipts, not row count or host flags, determine qualified actors.
    pub fn bind_actor_refresh_initial_episodes(
        &mut self,
        index_identity: Identity256,
        initial_episodes: &[(SemanticReplayMaterial, Identity256, Vec<u8>)],
    ) -> Result<u64, SemanticTransitionError> {
        self.ensure_quiescent()?;
        if index_identity == Identity256::default() || self.prepared_segment.is_some() {
            return Err(publication_input_error(
                "initial actor roster must precede original prepared work",
            ));
        }
        let mut episodes = Vec::with_capacity(initial_episodes.len());
        let mut actors = Vec::new();
        for (material, batch_identity, batch_bytes) in initial_episodes {
            let (origin, actor) = initial_episode_actor(material, *batch_identity, batch_bytes)?;
            if episodes.iter().any(|(previous, _)| previous == &origin) {
                return Err(publication_input_error(
                    "initial episode roster repeats an original action",
                ));
            }
            episodes.push((origin, *batch_identity));
            if actor {
                actors.push(origin);
            }
        }
        let count = u64::try_from(actors.len())
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        if let Some(original) = &self.actor_refresh_initial {
            if original.index != index_identity
                || original.episodes != episodes
                || original.actors != actors
            {
                return Err(publication_input_error(
                    "initial actor roster differs from its original sealed index",
                ));
            }
        } else {
            self.actor_refresh_initial = Some(ActorRefreshInitialEpisodes {
                index: index_identity,
                episodes,
                actors,
            });
        }
        Ok(count)
    }

    /// Retain the real model input owners of a particular original Update.
    /// The released parent authenticates the builder's immutable provenance;
    /// it is never reacquired or used as a future Update publication.
    pub fn prepare_actor_refresh(
        &mut self,
        step: &SemanticPreparedStep,
        released_parent: &SemanticPublishedLease,
        program_ordinal: u64,
        member_ordinal: u64,
        material: &SemanticReplayMaterial,
        batch_identity: Identity256,
        batch_bytes: &[u8],
    ) -> Result<SemanticPreparedActorRefresh, SemanticTransitionError> {
        self.require_prepared_replay_parent(step, released_parent)?;
        if self
            .actor_refresh_preparations
            .contains_key(&(step.token, member_ordinal))
        {
            return Err(publication_input_error(
                "actor refresh retains one original attempt for this Update member",
            ));
        }
        let address = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .program_steps
            .get(&step.token)
            .ok_or_else(|| {
                publication_input_error(
                    "actor refresh requires the original admitted program address",
                )
            })?;
        if address.ordinal != program_ordinal {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let logical_update = address
            .logical_update
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let phase = address.phase;
        let owner = self.checked_prepared_step(step, false)?;
        let prepared = owner
            .prepared
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let training = prepared.training_view.as_ref().ok_or_else(|| {
            publication_input_error("actor refresh requires the original frozen actor roster")
        })?;
        let member = *training
            .actor_group_members()
            .iter()
            .find(|member| member.ordinal == member_ordinal)
            .ok_or_else(|| {
                publication_input_error("actor refresh row is outside the frozen actor group")
            })?;
        let actor_members = training
            .actor_group_members()
            .iter()
            .map(|member| member.ordinal)
            .collect::<Vec<_>>();
        let (actual_origin, actor) = initial_episode_actor(material, batch_identity, batch_bytes)?;
        if !actor || actual_origin != member.origin {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let inputs = Arc::clone(
            owner
                .inputs
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let models = inputs
            .model_slots
            .iter()
            .map(|slots| {
                if slots[0] != slots[1] {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                inputs
                    .storage
                    .allocations
                    .get(slots[0])
                    .and_then(PublicationAllocation::model_owner)
                    .ok_or(SemanticTransitionError::ObservationMismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if models.len() != inputs.storage.model_memory.allocation_bytes.len() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let original = self.actor_refresh_initial.as_ref().ok_or_else(|| {
            publication_input_error("actor refresh lacks the complete initial actor roster")
        })?;
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let initial_count = u64::try_from(original.actors.len())
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let actor_slot = if let Some(index) = original
            .actors
            .iter()
            .position(|origin| *origin == member.origin)
        {
            if !original
                .episodes
                .iter()
                .any(|(origin, batch)| *origin == member.origin && *batch == batch_identity)
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            index as u64
        } else {
            initial_count
                .checked_add(
                    program
                        .main_origins
                        .iter()
                        .find(|assignment| {
                            assignment.origin == member.origin
                                && assignment.batch == batch_identity
                                && assignment.batch_bytes == batch_bytes
                        })
                        .map(|assignment| assignment.slot)
                        .ok_or_else(|| {
                            publication_input_error(
                                "actor origin has no authenticated original program assignment",
                            )
                        })?,
                )
                .ok_or(SemanticTransitionError::GenerationExhausted)?
        };
        let stride = initial_count
            .checked_add(program.proposals)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if actor_slot >= stride || logical_update >= program.logical_updates {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let mut rng = program
            .original_rng
            .ok_or(SemanticTransitionError::NotBound)?;
        let end = program
            .logical_updates
            .checked_mul(stride)
            .and_then(|span| rng.stream_serial.checked_add(span))
            .filter(|end| *end < 1u64 << 56)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        rng.stream_serial = logical_update
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(actor_slot))
            .and_then(|offset| rng.stream_serial.checked_add(offset))
            .and_then(|serial| serial.checked_add(1))
            .filter(|serial| *serial <= end)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        rng.proposal = 0;
        let work = if let Some(original) = self.actor_refresh_work.get(&step.token) {
            if original.members != actor_members {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            Arc::clone(original)
        } else {
            let work = Arc::new(ActorRefreshWork::allocate(&self.provider, actor_members)?);
            self.actor_refresh_work
                .insert(step.token, Arc::clone(&work));
            work
        };
        let proof = SemanticPreparedActorRefresh {
            inner: Arc::new(PreparedActorRefreshOwner {
                issuer: Arc::clone(&self.publication_issuer),
                scope: Arc::clone(&step.scope),
                step: step.token,
                logical_update,
                program: program.identity,
                phase,
                minimum_generation: released_parent.header.model_generation,
                rng,
                member,
                inputs,
                models,
                provider: Arc::clone(&self.provider),
                domain: self.domain.clone(),
                execution: self.prepared_parent_execution(step)?,
                cancelled: Arc::clone(
                    &self
                        .prepared_segment
                        .as_ref()
                        .ok_or(SemanticTransitionError::NotBound)?
                        .cancelled,
                ),
                target: Arc::clone(
                    self.publication
                        .as_ref()
                        .ok_or(SemanticTransitionError::NotBound)?,
                ),
                work,
                children: Mutex::new([None, None]),
                construction_pending: std::array::from_fn(|_| Arc::new(AtomicBool::new(false))),
            }),
        };
        if let Some(assignment) = self.actor_refresh_program.as_mut().and_then(|program| {
            program
                .main_origins
                .iter_mut()
                .find(|assignment| assignment.origin == member.origin)
        }) {
            match assignment.replay {
                actor_refresh_program::AssignmentReplayProof::Unresolved => {
                    assignment.replay = actor_refresh_program::AssignmentReplayProof::Joined;
                }
                actor_refresh_program::AssignmentReplayProof::Joined => (),
            }
        }
        self.actor_refresh_preparations
            .insert((step.token, member_ordinal), Arc::downgrade(&proof.inner));
        Ok(proof)
    }

    #[expect(clippy::too_many_arguments, reason = "original constructor retains the admitted graph geometry, program and cold operation")]
    pub fn prepare_actor_refresh_construction(
        &self,
        update: &SemanticPreparedStep,
        bank: usize,
        proof: &SemanticPreparedActorRefresh,
        records: crate::SemanticAdmissionRecords,
        capacities: crate::SemanticHypergraphCapacities,
        limits: crate::SemanticAdmissionLimits,
        program: Option<Arc<crate::SemanticProgramAdmission>>,
        work: SemanticColdNativeWork,
    ) -> Result<SemanticTransitionSessionConstruction, SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.require_actor_refresh_construction_target(update, bank, proof)?;
        if proof.inner.cancelled.load(Ordering::Acquire)
            || matches!(*proof.inner.execution.lock().map_err(|_| publication_input_error(
                "original actor submission custody lock is poisoned"
            ))?, ActorRefreshParentExecution::NeverSubmitted)
        {
            return Err(publication_input_error("cancelled actor cannot begin Session construction"));
        }
        let graph = self.provider.prepare_semantic_graph_construction(
            &proof.inner.domain, capacities, records, limits, work,
        ).map_err(SemanticTransitionError::Semantic)?;
        let binding = Arc::new(ActorRefreshConstructionBinding {
            proof: proof.clone(), bank, issuer: Arc::new(()),
            pending: Arc::clone(&proof.inner.construction_pending[bank]),
        });
        {
            let mut children = proof.inner.children.lock().map_err(|_| {
                publication_input_error("original actor child issuance lock is poisoned")
            })?;
            if proof.inner.cancelled.load(Ordering::Acquire) || children[bank].is_some() {
                return Err(publication_input_error("actor construction is issued once per original bank"));
            }
            children[bank] = Some(Arc::clone(&binding.issuer));
            binding.pending.store(true, Ordering::Release);
        }
        Ok(SemanticTransitionSessionConstruction {
            graph: Some(graph), admitted_graph: None, program, session: None,
            completed: None, binding, retired: false,
        })
    }

    pub fn resolve_actor_refresh_construction(
        &self,
        update: &SemanticPreparedStep,
        bank: usize,
        proof: &SemanticPreparedActorRefresh,
        original: &mut SemanticTransitionSessionConstruction,
    ) -> Result<Arc<Mutex<Option<SemanticTransitionSession>>>, SemanticTransitionError> {
        self.require_actor_refresh_construction_target(update, bank, proof)?;
        self.require_actor_refresh_construction_binding(bank, proof, &original.binding)?;
        let cancelled = proof.inner.cancelled.load(Ordering::Acquire)
            || matches!(*proof.inner.execution.lock().map_err(|_| publication_input_error(
                "original actor submission custody lock is poisoned"
            ))?, ActorRefreshParentExecution::NeverSubmitted);
        let may_submit = !cancelled
            && self.original_session_completion_may_submit()
            && self.original_prepared_non_owner_effects_are_clear_except_actor_construction(Some(&original.binding));
        original.resolve(may_submit)
    }

    pub fn retire_actor_refresh_construction(
        &self,
        update: &SemanticPreparedStep,
        bank: usize,
        proof: &SemanticPreparedActorRefresh,
        outer: &SemanticPreparedSegmentNonSubmission,
        original: &mut SemanticTransitionSessionConstruction,
    ) -> Result<(), SemanticTransitionError> {
        self.require_actor_refresh_construction_target(update, bank, proof)?;
        self.require_actor_refresh_construction_binding(bank, proof, &original.binding)?;
        self.prepared_segment.as_ref().ok_or(SemanticTransitionError::NotBound)?.require_non_submission(outer)?;
        self.require_original_prepared_graph_retirement()?;
        if !proof.inner.cancelled.load(Ordering::Acquire)
            || !matches!(*proof.inner.execution.lock().map_err(|_| publication_input_error(
                "original actor submission custody lock is poisoned"
            ))?, ActorRefreshParentExecution::NeverSubmitted)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        original.retire()
    }

    fn require_actor_refresh_construction_target(
        &self,
        update: &SemanticPreparedStep,
        bank: usize,
        proof: &SemanticPreparedActorRefresh,
    ) -> Result<(), SemanticTransitionError> {
        let build = self.prepared_segment.as_ref().ok_or(SemanticTransitionError::NotBound)?;
        build.check_retained(update, &self.publication_issuer)?;
        if bank > 1 || !Arc::ptr_eq(&update.issuer, &proof.inner.issuer)
            || !Arc::ptr_eq(&update.scope, &proof.inner.scope) || update.token != proof.inner.step
            || !Arc::ptr_eq(&self.provider, &proof.inner.provider)
            || !Arc::ptr_eq(&build.actor_refresh_execution, &proof.inner.execution)
            || !Arc::ptr_eq(&build.cancelled, &proof.inner.cancelled)
            || build.tokens.iter().position(|token| *token == update.token)
                .is_none_or(|index| build.transitions[index] != SemanticTransitionKind::Update)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    fn require_actor_refresh_construction_binding(
        &self,
        bank: usize,
        proof: &SemanticPreparedActorRefresh,
        binding: &Arc<ActorRefreshConstructionBinding>,
    ) -> Result<(), SemanticTransitionError> {
        let children = proof.inner.children.lock().map_err(|_| {
            publication_input_error("original actor child issuance lock is poisoned")
        })?;
        if binding.bank != bank || !Arc::ptr_eq(&binding.proof.inner, &proof.inner)
            || !Arc::ptr_eq(&binding.pending, &proof.inner.construction_pending[bank])
            || children[bank].as_ref().is_none_or(|issuer| !Arc::ptr_eq(issuer, &binding.issuer))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub(super) fn has_pending_actor_refresh_construction_except(
        &self,
        original: Option<&Arc<ActorRefreshConstructionBinding>>,
    ) -> bool {
        self.has_pending_actor_refresh_construction_except_owners(original, None)
    }

    pub(super) fn has_pending_actor_refresh_construction_except_cancellation(
        &self,
        cancellation: Option<&PreparedSegmentState>,
    ) -> bool {
        self.has_pending_actor_refresh_construction_except_owners(None, cancellation)
    }

    fn has_pending_actor_refresh_construction_except_owners(
        &self,
        original: Option<&Arc<ActorRefreshConstructionBinding>>,
        cancellation: Option<&PreparedSegmentState>,
    ) -> bool {
        self.actor_refresh_preparations.values().filter_map(std::sync::Weak::upgrade).any(|proof| {
            proof.construction_pending.iter().enumerate().any(|(bank, pending)| {
                pending.load(Ordering::Acquire) && original.is_none_or(|original| {
                    original.bank != bank || !Arc::ptr_eq(&original.proof.inner, &proof)
                        || !Arc::ptr_eq(&original.pending, pending)
                }) && cancellation.is_none_or(|build| {
                    !Arc::ptr_eq(&build.issuer, &self.publication_issuer)
                        || !Arc::ptr_eq(&proof.issuer, &build.issuer)
                        || !Arc::ptr_eq(&proof.scope, &build.scope)
                        || !Arc::ptr_eq(&proof.cancelled, &build.cancelled)
                        || !Arc::ptr_eq(&proof.execution, &build.actor_refresh_execution)
                        || build.tokens.iter().position(|token| *token == proof.step)
                            .is_none_or(|index| build.transitions[index] != SemanticTransitionKind::Update)
                })
            })
        })
    }

    /// Cold-stage original context with the target's retained CURRENT owners.
    /// No numerical model bytes are copied, zeroed, hashed, or observed here.
    /// Actual initialization and basis validation belong to the same captured
    /// Update branch before its genuine Recompute and new Proposal.
    pub fn restore_actor_refresh_context(
        &mut self,
        material: &SemanticReplayMaterial,
        proof: &SemanticPreparedActorRefresh,
        target_bank: usize,
    ) -> Result<(), SemanticTransitionError> {
        if target_bank > 1
            || material.kind != SemanticTransitionKind::Proposal
            || !Arc::ptr_eq(&self.provider, &proof.inner.provider)
            || crate::semantic_training_view::origin_record(material.training_view_origin()?)
                != proof.inner.member.origin
        {
            return Err(publication_input_error(
                "actor context requires its exact original row and current model owner",
            ));
        }
        let material_bytes = material.predecessor.encode()?;
        if let Some(original) = &self.actor_refresh {
            if !Arc::ptr_eq(&original.proof.inner, &proof.inner)
                || original.target_bank != target_bank
                || original.context.lock().map_err(|_| publication_input_error("actor context custody is poisoned"))?.material != material_bytes
            {
                return Err(publication_input_error("actor context changed its original restoration"));
            }
            return self.resolve_actor_refresh_context(proof, target_bank);
        }
        if proof.inner.cancelled.load(Ordering::Acquire)
            || matches!(
                *proof.inner.execution.lock().map_err(|_| publication_input_error(
                    "original actor submission custody lock is poisoned"
                ))?,
                ActorRefreshParentExecution::NeverSubmitted
            )
        {
            return Err(publication_input_error(
                "cancelled actor context cannot begin original staging",
            ));
        }
        self.ensure_quiescent()?;
        let device = Arc::new(allocate_publication(&self.provider, 1)?);
        let native_work = Arc::new(allocate_publication(&self.provider, 11)?);
        let stream = Arc::clone(proof.inner.domain.execution_stream());
        let zero = crate::device::RetainedDeviceWrite::new(&stream, &[0u64; 11], native_work.view())
            .map_err(|error| runtime_error("original actor work staging", error))?;
        let context = Arc::new(Mutex::new(ActorRefreshContextPreparation {
            material: material_bytes,
            zero,
            marker: None,
            admitted: [false; 2],
            zero_done: false,
            restore_done: false,
            ready: false,
        }));
        {
            let mut children = proof.inner.children.lock()
                .map_err(|_| publication_input_error("actor context custody lock is poisoned"))?;
            if children[target_bank].as_ref().is_some_and(|issuer| {
                !Arc::ptr_eq(issuer, &self.publication_issuer)
                    || self.actor_refresh_construction.as_ref().is_none_or(|construction| {
                        !Arc::ptr_eq(&construction.proof.inner, &proof.inner)
                            || construction.bank != target_bank
                            || !Arc::ptr_eq(&construction.issuer, issuer)
                    })
            }) {
                return Err(publication_input_error(
                    "actor context is staged once per original conditional bank",
                ));
            }
            children[target_bank] = Some(Arc::clone(&self.publication_issuer));
        }
        self.domain = proof.inner.domain.clone();
        self.stream = stream;
        self.actor_refresh = Some(StagedActorRefresh {
            proof: proof.clone(),
            target_bank,
            state: ActorRefreshCaptureState::Staged,
            device,
            native_work,
            retirement_complete: false,
            context,
            context_issuance: Arc::new(()),
            context_pending: Arc::new(AtomicBool::new(true)),
            native_ceiling: [0; 9],
            initializer_observation: Arc::new(Mutex::new(None)),
            initializer_refusal: None,
            cancellation: None,
        });
        self.resolve_actor_refresh_context(proof, target_bank)
    }

    /// Continue only the original child context and its retained restoration.
    pub fn resolve_actor_refresh_context(
        &mut self,
        proof: &SemanticPreparedActorRefresh,
        target_bank: usize,
    ) -> Result<(), SemanticTransitionError> {
        let original = self.actor_refresh.as_ref().ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&original.proof.inner, &proof.inner)
            || original.target_bank != target_bank
            || original.state != ActorRefreshCaptureState::Staged
        {
            return Err(publication_input_error("actor context continuation changed its original child"));
        }
        let context = Arc::clone(&original.context);
        let issuance = Arc::clone(&original.context_issuance);
        let pending = Arc::clone(&original.context_pending);
        let device = Arc::clone(&original.device);
        let mut retained = context.lock().map_err(|_| publication_input_error("actor context custody is poisoned"))?;
        if retained.ready { return Ok(()); }
        if !retained.zero_done {
            if !retained.admitted[0] {
                self.provider.admit_launch_metadata_htod(11 * size_of::<u64>());
                retained.admitted[0] = true;
            }
            if !retained.zero.entered() {
                self.ensure_quiescent_except_actor_context(&issuance)?;
                retained.zero.enqueue(&self.stream).map_err(|error| runtime_error("original actor work initialization", error))?;
            }
            retained.zero.resolve().map_err(|error| runtime_error("original actor work completion", error))?;
            retained.zero_done = true;
        }
        if !retained.restore_done {
            let restored = if self.state_material_restore_started() {
                self.resolve_state_material_restore()?
            } else {
                self.ensure_quiescent_except_actor_context(&issuance)?;
                self.restore_state_material_inner(&retained.material, None,
                    Some(RestoredModelOwners::Current {
                        owners: &proof.inner.models,
                        minimum_generation: proof.inner.minimum_generation,
                    }))?
            };
            if restored.is_some() { return Err(SemanticTransitionError::ObservationMismatch); }
            retained.restore_done = true;
        }
        if retained.marker.is_none() {
            self.ensure_quiescent_except_actor_context(&issuance)?;
            let native_ceiling = actor_initialize_native_work_ceiling(self)?;
            self.actor_refresh.as_mut().expect("retained original actor").native_ceiling = native_ceiling;
            let child = self.publication.as_ref().ok_or(SemanticTransitionError::NotBound)?;
            let marker = PreparedActorRefreshDevice {
                abi: 1, status: 1,
                target_control: proof.inner.target.control.device_ptr_value(),
                target_lease: proof.inner.inputs.reader.device_ptr_value(),
                target_header: proof.inner.inputs.header.device_ptr_value(),
                target_bank: target_bank as u64,
                child_control: child.control.device_ptr_value(),
                member_ordinal: proof.inner.member.ordinal,
                minimum_generation: proof.inner.minimum_generation,
                stream_serial: proof.inner.rng.stream_serial,
                family_id: u64::from(proof.inner.rng.family_id),
                proposal: u64::from(proof.inner.rng.proposal),
                program: proof.inner.program, phase: proof.inner.phase,
                original_origin: proof.inner.member.origin, acquired_generation: 0,
                model_geometry_digest: Identity256::default(),
                model_numerical_digest: Identity256::default(),
            };
            retained.marker = Some(crate::device::RetainedDeviceWrite::new(&self.stream, &[marker], device.view())
                .map_err(|error| runtime_error("original actor marker staging", error))?);
        }
        if !retained.admitted[1] {
            self.provider.admit_launch_metadata_htod(size_of::<PreparedActorRefreshDevice>());
            retained.admitted[1] = true;
        }
        let marker = retained.marker.as_mut().expect("original actor marker");
        if !marker.entered() {
            self.ensure_quiescent_except_actor_context(&issuance)?;
            marker.enqueue(&self.stream).map_err(|error| runtime_error("original actor marker initialization", error))?;
        }
        marker.resolve().map_err(|error| runtime_error("original actor marker completion", error))?;
        let mut rng = proof.inner.rng;
        rng.model_generation = u32::try_from(proof.inner.minimum_generation)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        self.rng = Some(rng);
        self.next_proposal = u64::from(rng.proposal);
        retained.ready = true;
        pending.store(false, Ordering::Release);
        Ok(())
    }

    pub(super) fn actor_refresh_context_issuance(&self) -> Option<Arc<()>> {
        self.actor_refresh.as_ref().map(|original| Arc::clone(&original.context_issuance))
    }

    pub(super) fn has_pending_actor_refresh_context_except(&self, except: Option<&Arc<()>>) -> bool {
        self.actor_refresh.as_ref().is_some_and(|original| {
            original.context_pending.load(Ordering::Acquire)
                && except.is_none_or(|issuer| !Arc::ptr_eq(issuer, &original.context_issuance))
        })
    }

    pub fn actor_refresh_context_started(
        &self,
        proof: &SemanticPreparedActorRefresh,
        bank: usize,
    ) -> Result<bool, SemanticTransitionError> {
        if bank > 1 || !Arc::ptr_eq(&self.provider, &proof.inner.provider) {
            return Err(publication_input_error("actor context changed its original provider or bank"));
        }
        match &self.actor_refresh {
            None => Ok(false),
            Some(original) if Arc::ptr_eq(&original.proof.inner, &proof.inner)
                && original.target_bank == bank => Ok(true),
            Some(_) => Err(publication_input_error("actor context changed its original child")),
        }
    }

    /// Record the CURRENT basis bridge inside the original target bank, after
    /// its input producer and before the child's genuine Recompute and Proposal.
    pub fn record_prepared_actor_refresh(
        &mut self,
        update: &SemanticPreparedStep,
        bank: usize,
        child: &mut SemanticTransitionSession,
        proof: &SemanticPreparedActorRefresh,
    ) -> Result<(), SemanticTransitionError> {
        self.require_prepared_actor_refresh(update, bank, proof)?;
        let staged = child
            .actor_refresh
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if staged.target_bank != bank
            || !Arc::ptr_eq(&staged.proof.inner, &proof.inner)
            || staged.state != ActorRefreshCaptureState::Staged
        {
            return Err(publication_input_error(
                "actor bridge records once in its original conditional bank",
            ));
        }
        let retirement = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .graph_retirement
            .clone()
            .ok_or_else(|| {
                publication_input_error(
                    "actor capture requires the original outer graph retirement owner",
                )
            })?;
        let child_build = child
            .prepared_segment
            .as_mut()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&child_build.cancelled, &proof.inner.cancelled)
            || !Arc::ptr_eq(&child_build.actor_refresh_execution, &proof.inner.execution)
            || child_build.graph_retirement.is_some()
        {
            return Err(publication_input_error(
                "actor child must retain its sole original outer executable",
            ));
        }
        child_build.graph_retirement = Some(retirement);
        child_build.capturing = true;
        child.graph.enter_transition();
        let staged = child
            .actor_refresh
            .as_mut()
            .ok_or(SemanticTransitionError::NotBound)?;
        staged.state = ActorRefreshCaptureState::Recording;
        let device = Arc::clone(&staged.device);
        let native_work = Arc::clone(&staged.native_work);
        let mut descriptor = child.descriptor();
        descriptor.publication = PublicationCommand {
            control: child
                .publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .control
                .device_ptr_value(),
            lease: device.device_ptr_value(),
            operation: 10,
            native_work: native_work.device_ptr_value(),
        };
        let mut recorder = child.kernel_recorder();
        proof.inner.target.record(&mut recorder);
        recorder.read(&proof.inner.inputs.reader);
        recorder.read(&proof.inner.inputs.header);
        recorder.read_write(device.as_ref());
        recorder.read_write(native_work.as_ref());
        let execute = child.execute.clone();
        enqueue_recorded(&child.domain, &mut child.poisoned, recorder, |enqueue| {
            let mut params = [(&mut descriptor as *mut Descriptor).cast()];
            // SAFETY: all target, child, model and proof allocations belong to
            // the original outer capture and have been retained and recorded.
            unsafe {
                execute.launch_raw_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut params,
                    false,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })?;
        child
            .actor_refresh
            .as_mut()
            .expect("retained original actor bridge")
            .state = ActorRefreshCaptureState::Recorded;
        Ok(())
    }

    /// Freeze only the metadata of the child already embedded in the outer
    /// executable. This never creates or submits an independent executable.
    pub fn finish_prepared_actor_refresh(
        &self,
        update: &SemanticPreparedStep,
        bank: usize,
        child: &mut SemanticTransitionSession,
        proof: &SemanticPreparedActorRefresh,
    ) -> Result<(), SemanticTransitionError> {
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        build.check_retained(update, &self.publication_issuer)?;
        if build.submitted || build.cancelled.load(Ordering::Acquire) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let staged = child
            .actor_refresh
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if bank != staged.target_bank
            || !Arc::ptr_eq(&staged.proof.inner, &proof.inner)
            || !Arc::ptr_eq(&update.issuer, &proof.inner.issuer)
            || !Arc::ptr_eq(&update.scope, &proof.inner.scope)
            || update.token != proof.inner.step
            || !Arc::ptr_eq(
                &self.prepared_parent_execution(update)?,
                &proof.inner.execution,
            )
            || !matches!(
                staged.state,
                ActorRefreshCaptureState::Recorded | ActorRefreshCaptureState::Frozen
            )
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if staged.state == ActorRefreshCaptureState::Recorded {
            child.freeze_prepared_segment_metadata()?;
            child
                .actor_refresh
                .as_mut()
                .expect("retained original actor bridge")
                .state = ActorRefreshCaptureState::Frozen;
        }
        let work = &proof.inner.work;
        let slot = work
            .members
            .iter()
            .position(|ordinal| *ordinal == proof.inner.member.ordinal)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let mut frozen = work
            .frozen
            .lock()
            .map_err(|_| publication_input_error("original actor work freeze lock is poisoned"))?;
        frozen[bank][slot] = true;
        if frozen[bank].iter().all(|frozen| *frozen) {
            let mut uploads = work.uploads.lock().map_err(|_| {
                publication_input_error("original actor metadata custody lock is poisoned")
            })?;
            if uploads[bank].is_none() {
                let components = work.components.lock().map_err(|_| {
                    publication_input_error("original actor work custody lock is poisoned")
                })?;
                let entries = components[bank]
                    .iter()
                    .map(|component| {
                        let component = component
                            .as_ref()
                            .ok_or(SemanticTransitionError::ObservationMismatch)?;
                        Ok(ActorRefreshWorkEntry {
                            member_ordinal: component.member_ordinal,
                            proof: component.marker.device_ptr_value(),
                            native_work: component.native_work.device_ptr_value(),
                            states: [
                                component.states[0].device_ptr_value(),
                                component.states[1].device_ptr_value(),
                            ],
                            readers: [
                                component.readers[0].device_ptr_value(),
                                component.readers[1].device_ptr_value(),
                            ],
                            results: [
                                component.results[0].device_ptr_value(),
                                component.results[1].device_ptr_value(),
                            ],
                            parents: [
                                component.parents[0].device_ptr_value(),
                                component.parents[1].device_ptr_value(),
                            ],
                            model_work: component.model_inputs,
                            native_ceiling: component.native_ceiling,
                            transition_ceilings: component.transition_ceilings,
                        })
                    })
                    .collect::<Result<Vec<_>, SemanticTransitionError>>()?;
                let entry_write = crate::device::RetainedDeviceWrite::new(
                    &self.stream,
                    &entries,
                    work.entries[bank].view(),
                )
                .map_err(|error| runtime_error("original actor metadata staging", error))?;
                let header_write = crate::device::RetainedDeviceWrite::new(
                    &self.stream,
                    &[ActorRefreshWorkInput {
                        abi: 1,
                        entries: work.entries[bank].device_ptr_value(),
                        count: work.members.len() as u64,
                    }],
                    work.inputs[bank].view(),
                )
                .map_err(|error| runtime_error("original actor metadata staging", error))?;
                uploads[bank] = Some(ActorRefreshMetadataUploads {
                    entries: entry_write,
                    header: header_write,
                    admitted: [false; 2],
                    cursor: 0,
                });
            }
            let bytes = work
                .members
                .len()
                .checked_mul(size_of::<ActorRefreshWorkEntry>())
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            uploads[bank]
                .as_mut()
                .expect("retained original actor metadata")
                .complete(&self.provider, &self.stream, bytes)?;
        }
        Ok(())
    }

    pub(super) fn record_prepared_actor_refresh_work_entry(
        &self,
        proof: &SemanticPreparedActorRefresh,
        recorder: &mut LaunchRecorder,
    ) -> Result<u64, SemanticTransitionError> {
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&build.issuer, &proof.inner.issuer)
            || !Arc::ptr_eq(&build.scope, &proof.inner.scope)
            || !Arc::ptr_eq(&build.cancelled, &proof.inner.cancelled)
            || !self
                .actor_refresh_work
                .get(&proof.inner.step)
                .is_some_and(|work| Arc::ptr_eq(work, &proof.inner.work))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let bank = self
            .steps
            .get(&proof.inner.step)
            .and_then(|owner| owner.prepared.as_ref())
            .and_then(|prepared| prepared.model_work.as_ref())
            .and_then(|work| work.capture_bank)
            .ok_or(SemanticTransitionError::NotBound)?;
        let work = &proof.inner.work;
        let slot = work
            .members
            .iter()
            .position(|ordinal| *ordinal == proof.inner.member.ordinal)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        // These entries are filled only after the same child R/P graph and its
        // original model recordings have frozen. Capture records their stable
        // addresses without reading future execution output.
        recorder.read(&work.entries[bank]);
        let components = work
            .components
            .lock()
            .map_err(|_| publication_input_error("original actor work custody lock is poisoned"))?;
        let component = components[bank][slot]
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        recorder.read(&component.marker);
        for operation in 0..2 {
            recorder.read(&component.states[operation]);
            recorder.read(&component.readers[operation]);
            recorder.read(&component.results[operation]);
            recorder.read(&component.parents[operation]);
        }
        work.entries[bank]
            .device_ptr_value()
            .checked_add(
                slot.checked_mul(size_of::<ActorRefreshWorkEntry>())
                    .ok_or(SemanticTransitionError::GenerationExhausted)? as u64,
            )
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }

    pub(super) fn require_prepared_actor_refresh(
        &self,
        update: &SemanticPreparedStep,
        bank: usize,
        proof: &SemanticPreparedActorRefresh,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_step(update, true)?;
        if bank > 1
            || !Arc::ptr_eq(&update.issuer, &proof.inner.issuer)
            || !Arc::ptr_eq(&update.scope, &proof.inner.scope)
            || update.token != proof.inner.step
            || !Arc::ptr_eq(
                &self.prepared_parent_execution(update)?,
                &proof.inner.execution,
            )
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let owner = &self.steps[&update.token];
        let prepared = owner
            .prepared
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !prepared.inputs_recorded
            || !Arc::ptr_eq(
                owner
                    .inputs
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?,
                &proof.inner.inputs,
            )
            || self.prepared_transition_kind(update)? != SemanticTransitionKind::Update
        {
            return Err(publication_input_error(
                "actor bridge requires this Update's original recorded input producer",
            ));
        }
        Ok(())
    }

    pub(super) fn prepared_actor_refresh_device(
        &self,
        proof: &SemanticPreparedActorRefresh,
    ) -> Result<Arc<TrackedCudaSlice<PreparedActorRefreshDevice>>, SemanticTransitionError> {
        let staged = self
            .actor_refresh
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&staged.proof.inner, &proof.inner)
            || !matches!(
                staged.state,
                ActorRefreshCaptureState::Recorded | ActorRefreshCaptureState::Frozen
            )
            || build.transitions
                != [
                    SemanticTransitionKind::Recompute,
                    SemanticTransitionKind::Proposal,
                ]
            || build.next != 2
            || build.active
        {
            return Err(publication_input_error(
                "actor VJP requires its genuine recorded Recompute and new Proposal",
            ));
        }
        let work = &proof.inner.work;
        let slot = work
            .members
            .iter()
            .position(|ordinal| *ordinal == proof.inner.member.ordinal)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let mut components = work
            .components
            .lock()
            .map_err(|_| publication_input_error("original actor work custody lock is poisoned"))?;
        if components[staged.target_bank][slot].is_none() {
            let recompute_owner = self
                .steps
                .get(&build.tokens[0])
                .ok_or(SemanticTransitionError::NotBound)?;
            let proposal_owner = self
                .steps
                .get(&build.tokens[1])
                .ok_or(SemanticTransitionError::NotBound)?;
            let recompute = recompute_owner
                .prepared
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let proposal = proposal_owner
                .prepared
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let recompute_work = recompute
                .model_work
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let proposal_work = proposal
                .model_work
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            components[staged.target_bank][slot] = Some(ActorRefreshObservedComponent {
                member_ordinal: proof.inner.member.ordinal,
                instance: self
                    .publication
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?
                    .instance,
                marker: staged.device.view(),
                native_work: staged.native_work.view(),
                states: [
                    recompute.branches[0].state.view(),
                    proposal.branches[1].state.view(),
                ],
                readers: [recompute.reader.view(), proposal.reader.view()],
                results: [recompute.result.view(), proposal.result.view()],
                parents: [
                    recompute_owner
                        .inputs
                        .as_ref()
                        .ok_or(SemanticTransitionError::NotBound)?
                        .header
                        .view(),
                    proposal_owner
                        .inputs
                        .as_ref()
                        .ok_or(SemanticTransitionError::NotBound)?
                        .header
                        .view(),
                ],
                model_actual: [recompute_work.actual.view(), proposal_work.actual.view()],
                model_recordings: [
                    recompute_work
                        .recording
                        .frozen_certificate()
                        .map_err(publication_input_error)?,
                    proposal_work
                        .recording
                        .frozen_certificate()
                        .map_err(publication_input_error)?,
                ],
                model_event_records: [
                    recompute_work.recording.events().to_vec(),
                    proposal_work.recording.events().to_vec(),
                ],
                native_ceiling: staged.native_ceiling,
                transition_ceilings:
                    native_work_bound::actor_refresh_transition_native_work_ceilings(self)?,
                initializer_observation: Arc::clone(&staged.initializer_observation),
                model_events: [recompute_work.device.view(), proposal_work.device.view()],
                model_inputs: [recompute_work.descriptor(), proposal_work.descriptor()],
            });
        }
        Ok(Arc::clone(&staged.device))
    }

    /// Attach only the original native-owned full actor roster to the target
    /// transition. The numerical producer cannot substitute an aggregate tally.
    pub(super) fn attach_prepared_actor_refresh_work(
        &self,
        step: &SemanticPreparedStep,
        bank: usize,
        descriptor: &mut Descriptor,
        recorder: &mut LaunchRecorder,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_step(step, true)?;
        if bank > 1 || descriptor.publication.operation != 0 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if let Some(work) = self.actor_refresh_work.get(&step.token) {
            if self.prepared_transition_kind(step)? != SemanticTransitionKind::Update {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            work.record(bank, recorder)?;
            descriptor.publication.native_work = work.inputs[bank].device_ptr_value();
        }
        Ok(())
    }

    pub(super) fn prepared_actor_refresh_work_components(
        &self,
        step: &SemanticPreparedStep,
        bank: usize,
    ) -> Result<Vec<ActorRefreshObservedComponent>, SemanticTransitionError> {
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        build.check_retained(step, &self.publication_issuer)?;
        if !build.submitted
            || build.capturing
            || !build.transfers.as_ref().is_some_and(|transfers| {
                transfers.terminal_wait == SemanticSegmentTerminalWait::Complete
            })
        {
            return Err(publication_input_error(
                "original actor work observation requires its same completed graph wait",
            ));
        }
        if self
            .steps
            .get(&step.token)
            .and_then(|step| step.prepared.as_ref())
            .is_none()
        {
            return Err(SemanticTransitionError::NotBound);
        }
        if bank > 1 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let Some(work) = self.actor_refresh_work.get(&step.token) else {
            return Ok(Vec::new());
        };
        let frozen = work
            .frozen
            .lock()
            .map_err(|_| publication_input_error("original actor work freeze lock is poisoned"))?;
        if !frozen[bank].iter().all(|frozen| *frozen) {
            return Err(publication_input_error(
                "original actor work must be fully frozen before observation",
            ));
        }
        if !work.uploads.lock().map_err(|_| {
            publication_input_error("original actor metadata custody lock is poisoned")
        })?[bank]
            .as_ref()
            .is_some_and(|uploads| uploads.cursor == 2)
        {
            return Err(publication_input_error(
                "original actor metadata upload must complete before observation",
            ));
        }
        let components = work
            .components
            .lock()
            .map_err(|_| publication_input_error("original actor work custody lock is poisoned"))?;
        components[bank]
            .iter()
            .cloned()
            .map(|component| component.ok_or(SemanticTransitionError::ObservationMismatch))
            .collect()
    }

    pub(super) fn require_prepared_actor_refresh_work_frozen(
        &self,
    ) -> Result<(), SemanticTransitionError> {
        for (&token, work) in &self.actor_refresh_work {
            let build = self
                .prepared_segment
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let address = build
                .program_steps
                .get(&token)
                .ok_or(SemanticTransitionError::NotBound)?;
            let prepared = self
                .steps
                .get(&token)
                .and_then(|step| step.prepared.as_ref())
                .ok_or(SemanticTransitionError::NotBound)?;
            let target = prepared
                .model_work
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .recording
                .frozen_certificate()
                .map_err(publication_input_error)?;
            let frozen = work.frozen.lock().map_err(|_| {
                publication_input_error("original actor work freeze lock is poisoned")
            })?;
            let uploads = work.uploads.lock().map_err(|_| {
                publication_input_error("original actor metadata custody lock is poisoned")
            })?;
            let components = work.components.lock().map_err(|_| {
                publication_input_error("original actor work custody lock is poisoned")
            })?;
            for bank in 0..2 {
                if !frozen[bank].iter().all(|frozen| *frozen)
                    || !uploads[bank]
                        .as_ref()
                        .is_some_and(|uploads| uploads.cursor == 2)
                {
                    return Err(publication_input_error(
                        "every original actor child must freeze before its sole parent launch",
                    ));
                }
                let mut model_bound = target.bound();
                let mut model_calls = target.model_call_count();
                let mut native = [0; 9];
                for component in &components[bank] {
                    let component = component
                        .as_ref()
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    for recording in &component.model_recordings {
                        model_bound = actor_work_add(model_bound, recording.bound())?;
                        model_calls = actor_work_add(model_calls, recording.model_call_count())?;
                    }
                    native_work_bound::add_native_work_ceiling(
                        &mut native,
                        component.native_ceiling,
                    )?;
                    for ceiling in component.transition_ceilings {
                        native_work_bound::add_native_work_ceiling(&mut native, ceiling)?;
                    }
                }
                // This necessary admission joins actual frozen MODEL producers
                // and operation-10 native ceilings to the original instruction.
                // The original operation owner additionally admits hot native
                // transition and full cold/lifecycle producers; no budget is
                // reset or replaced with a caller-authored remaining allowance.
                let known_work = actor_work_add(
                    model_bound,
                    native_work_bound::native_work_ceiling_units(native)?,
                )?;
                if known_work > address.budget[0] || model_calls > address.budget[2] {
                    return Err(publication_input_error("original Update allowance cannot admit its retained actor model and initialization producers"));
                }
            }
        }
        Ok(())
    }

    /// Retire one genuine unused child only after its original outer whole
    /// roster was cancelled and the shared executable was destroyed.
    pub fn cancel_prepared_actor_refresh(
        &self,
        update: &SemanticPreparedStep,
        bank: usize,
        child: Option<&mut SemanticTransitionSession>,
        proof: &SemanticPreparedActorRefresh,
        outer: &SemanticPreparedSegmentNonSubmission,
    ) -> Result<Option<SemanticPreparedSegmentNonSubmission>, SemanticTransitionError> {
        self.ensure_quiescent()?;
        let build = self.prepared_segment.as_ref().ok_or(SemanticTransitionError::NotBound)?;
        build.check_retained(update, &self.publication_issuer)?;
        build.require_non_submission(outer)?;
        self.require_original_prepared_graph_retirement()?;
        if bank > 1
            || !Arc::ptr_eq(&update.issuer, &proof.inner.issuer)
            || !Arc::ptr_eq(&update.scope, &proof.inner.scope)
            || update.token != proof.inner.step
            || !Arc::ptr_eq(&build.actor_refresh_execution, &proof.inner.execution)
            || !Arc::ptr_eq(&build.cancelled, &proof.inner.cancelled)
            || build.tokens.iter().position(|token| *token == update.token)
                .is_none_or(|index| build.transitions[index] != SemanticTransitionKind::Update)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if !matches!(*proof.inner.execution.lock().map_err(|_| {
            publication_input_error("original actor submission custody lock is poisoned")
        })?, ActorRefreshParentExecution::NeverSubmitted) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let Some(child) = child else {
            let children = proof.inner.children.lock().map_err(|_| {
                publication_input_error("original actor child issuance lock is poisoned")
            })?;
            if children[bank].is_some() {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            return Ok(None);
        };
        let staged = child.actor_refresh.as_ref().ok_or(SemanticTransitionError::NotBound)?;
        let child_build = child.prepared_segment.as_ref().ok_or(SemanticTransitionError::NotBound)?;
        if bank != staged.target_bank
            || !Arc::ptr_eq(&staged.proof.inner, &proof.inner)
            || !Arc::ptr_eq(&child.provider, &proof.inner.provider)
            || !Arc::ptr_eq(&child_build.actor_refresh_execution, &proof.inner.execution)
            || !Arc::ptr_eq(&child_build.cancelled, &proof.inner.cancelled)
            || child_build.transitions != [SemanticTransitionKind::Recompute, SemanticTransitionKind::Proposal]
            || child_build.submitted
            || child_build.completed
            || child.captured.is_some()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if let Some(original) = &staged.cancellation {
            child_build.require_non_submission(original)?;
            return Ok(Some(original.clone()));
        }
        let handles = child_build.handles()?;
        let cancellation = child.cancel_prepared_segment_before_submission(&handles)?;
        child.actor_refresh.as_mut().expect("authenticated original actor")
            .cancellation = Some(cancellation.clone());
        Ok(Some(cancellation))
    }

    fn require_actor_refresh_final_use(
        &self,
        cancellation: Option<&SemanticPreparedSegmentNonSubmission>,
    ) -> Result<(), SemanticTransitionError> {
        let original = self.actor_refresh.as_ref().ok_or(SemanticTransitionError::NotBound)?;
        if let Some(cancellation) = cancellation {
            let retained = original.cancellation.as_ref().ok_or(SemanticTransitionError::NotBound)?;
            if !retained.matches(&cancellation.steps)
                || !original.proof.inner.cancelled.load(Ordering::Acquire)
                || !matches!(*original.proof.inner.execution.lock().map_err(|_| {
                    publication_input_error("original actor submission custody lock is poisoned")
                })?, ActorRefreshParentExecution::NeverSubmitted)
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        } else {
            if original.cancellation.is_some() {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            self.require_actor_refresh_parent_completion()?;
        }
        Ok(())
    }

    /// Attach the same outer final-use report, never an invented child reader.
    pub fn attach_actor_refresh_retirement_work(
        &mut self,
        proof: &SemanticPreparedActorRefresh,
        bank: usize,
        work: SemanticColdNativeWork,
        cancellation: Option<&SemanticPreparedSegmentNonSubmission>,
    ) -> Result<(), SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.require_actor_refresh_final_use(cancellation)?;
        let original = self.actor_refresh.as_ref().ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&original.proof.inner, &proof.inner)
            || original.target_bank != bank
            || original.retirement_complete
        {
            return Err(publication_input_error("actor retirement changed its original child"));
        }
        self.attach_cold_native_work(work)
    }

    /// Preserve the actual child-completion proof after the original build is
    /// removed by the common resource-retirement implementation.
    pub fn take_actor_refresh_resources_for_retirement(
        &mut self,
        proof: &SemanticPreparedActorRefresh,
        bank: usize,
        cancellation: Option<&SemanticPreparedSegmentNonSubmission>,
    ) -> Result<Vec<Arc<dyn Send + Sync>>, SemanticTransitionError> {
        self.require_actor_refresh_final_use(cancellation)?;
        let original = self.actor_refresh.as_ref().ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&original.proof.inner, &proof.inner)
            || original.target_bank != bank
            || original.retirement_complete
        {
            return Err(publication_input_error("actor resource retirement changed its original child"));
        }
        let resources = match cancellation {
            Some(cancellation) => self.take_cancelled_prepared_resources_for_retirement(cancellation)?,
            None => self.take_prepared_resources_for_retirement()?,
        };
        self.actor_refresh.as_mut().expect("authenticated original actor").retirement_complete = true;
        Ok(resources)
    }

    /// Join only a genuinely retired CURRENT child before the common terminal
    /// allocation release. This path never constructs a publication lease.
    pub fn join_actor_refresh_release(
        &mut self,
        proof: &SemanticPreparedActorRefresh,
        bank: usize,
        cancellation: Option<&SemanticPreparedSegmentNonSubmission>,
    ) -> Result<(), SemanticTransitionError> {
        if !self.has_retired_prepared_initialization() {
            self.ensure_quiescent()?;
        } else if !self.retired_prepared_initialization_may_release() {
            return Err(SemanticTransitionError::Poisoned);
        }
        self.require_actor_refresh_final_use(cancellation)?;
        self.require_closed_evaluations()?;
        self.require_completed_graph_retirements()?;
        let original = self.actor_refresh.as_ref().ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&original.proof.inner, &proof.inner)
            || original.target_bank != bank
            || !original.retirement_complete
            || self.training_canary_source_borrowed()
            || self.graph.borrowed_cold_work().is_some()
            || !self.readers.is_empty()
            || !self.steps.is_empty()
            || self.captured.is_some()
            || self.prepared_segment.is_some()
            || !self.prepared_resources.is_empty()
        {
            return Err(publication_input_error("actor Session retains unfinished original final use"));
        }
        self.stream.context().bind_to_thread()
            .and_then(|_| self.stream.context().synchronize())
            .map_err(|error| runtime_error("original actor allocation completion", error))
    }

    pub(super) fn require_actor_refresh_parent_completion(
        &self,
    ) -> Result<(), SemanticTransitionError> {
        let staged = self
            .actor_refresh
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let execution = staged.proof.inner.execution.lock().map_err(|_| {
            publication_input_error("original actor submission custody lock is poisoned")
        })?;
        if !matches!(*execution, ActorRefreshParentExecution::Completed) {
            return Err(publication_input_error(
                "actor completion requires the same original outer submission to finish",
            ));
        }
        Ok(())
    }

    fn validate_actor_refresh_initializer_refusal(
        &self,
        observation: &ActorRefreshInitializerObservation,
    ) -> Result<(), SemanticTransitionError> {
        self.require_actor_refresh_parent_completion()?;
        let staged = self
            .actor_refresh
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotCaptured)?;
        let proof = &staged.proof.inner;
        let marker = observation.marker;
        if staged.state != ActorRefreshCaptureState::Frozen
            || !build.finished
            || build.submitted
            || build.graph_retirement.is_none()
            || !Arc::ptr_eq(&build.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&build.actor_refresh_execution, &proof.execution)
            || !Arc::ptr_eq(&build.cancelled, &proof.cancelled)
            || build.transitions
                != [
                    SemanticTransitionKind::Recompute,
                    SemanticTransitionKind::Proposal,
                ]
            || build.tokens.len() != 2
            || marker.abi != 1
            || marker.status != 2
            || marker.target_bank != staged.target_bank as u64
            || marker.target_control != proof.target.control.device_ptr_value()
            || marker.target_lease != proof.inputs.reader.device_ptr_value()
            || marker.target_header != proof.inputs.header.device_ptr_value()
            || marker.child_control
                != self
                    .publication
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?
                    .control
                    .device_ptr_value()
            || marker.member_ordinal != proof.member.ordinal
            || marker.minimum_generation != proof.minimum_generation
            || marker.stream_serial != proof.rng.stream_serial
            || marker.family_id != u64::from(proof.rng.family_id)
            || marker.proposal != u64::from(proof.rng.proposal)
            || marker.program != proof.program
            || marker.phase != proof.phase
            || marker.original_origin != proof.member.origin
            || observation.native_work[10] != 0
            || observation.native_work[1..10]
                .iter()
                .try_fold(0u64, |sum, value| sum.checked_add(*value))
                != Some(observation.native_work[0])
            || observation.native_work[1..10]
                .iter()
                .zip(staged.native_ceiling)
                .any(|(actual, ceiling)| *actual > ceiling)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let children = proof.children.lock().map_err(|_| {
            publication_input_error("original actor child custody lock is poisoned")
        })?;
        if children[staged.target_bank]
            .as_ref()
            .is_none_or(|issuer| !Arc::ptr_eq(issuer, &self.publication_issuer))
            || *staged.initializer_observation.lock().map_err(|_| {
                publication_input_error("original actor refusal observation lock is poisoned")
            })? != Some(*observation)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    pub(super) fn retain_actor_refresh_initializer_refusal(
        &mut self,
        marker: PreparedActorRefreshDevice,
    ) -> Result<(), SemanticTransitionError> {
        let staged = self
            .actor_refresh
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let observation = staged
            .initializer_observation
            .lock()
            .map_err(|_| {
                publication_input_error("original actor refusal observation lock is poisoned")
            })?
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if observation.marker != marker {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.validate_actor_refresh_initializer_refusal(&observation)?;
        if self.checked_actor_refresh_initializer_refusal()?.is_some() {
            return Ok(());
        }
        let refusal = SemanticActorRefreshInitializerRefusal {
            inner: Arc::new(ActorRefreshInitializerRefusalOwner {
                issuer: Arc::clone(&self.publication_issuer),
                scope: Arc::clone(
                    &self.prepared_segment.as_ref().expect("original actor child").scope,
                ),
                proof: Arc::clone(&staged.proof.inner),
                target_bank: staged.target_bank,
                observation,
            }),
        };
        self.actor_refresh
            .as_mut()
            .expect("original actor child")
            .initializer_refusal = Some(refusal);
        Ok(())
    }

    pub(super) fn checked_actor_refresh_initializer_refusal(
        &self,
    ) -> Result<Option<&SemanticActorRefreshInitializerRefusal>, SemanticTransitionError> {
        let Some(staged) = self.actor_refresh.as_ref() else {
            return Ok(None);
        };
        let Some(refusal) = staged.initializer_refusal.as_ref() else {
            return Ok(None);
        };
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotCaptured)?;
        if !Arc::ptr_eq(&refusal.inner.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&refusal.inner.scope, &build.scope)
            || !Arc::ptr_eq(&refusal.inner.proof, &staged.proof.inner)
            || refusal.inner.target_bank != staged.target_bank
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.validate_actor_refresh_initializer_refusal(&refusal.inner.observation)?;
        Ok(Some(refusal))
    }

    /// Return only the reached refusal authenticated by this child's original
    /// completion observer. It authorizes final-use retirement, not Recompute
    /// or Proposal outcomes.
    pub fn actor_refresh_initializer_refusal(
        &self,
    ) -> Result<Option<SemanticActorRefreshInitializerRefusal>, SemanticTransitionError> {
        Ok(self.checked_actor_refresh_initializer_refusal()?.cloned())
    }

    /// Retire the genuine fresh Proposal's final-use tape after the same outer
    /// execution and embedded observation. The backward pass was already
    /// recorded in that execution; this invokes only its canonical finisher.
    pub(super) fn finish_actor_refresh_policy_invocations(
        &mut self,
    ) -> Result<(), SemanticTransitionError> {
        self.require_actor_refresh_parent_completion()?;
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !build.completed
            || build.transitions
                != [
                    SemanticTransitionKind::Recompute,
                    SemanticTransitionKind::Proposal,
                ]
            || build.tokens.len() != 2
        {
            return Err(publication_input_error(
                "actor tape retirement requires its actual completed child roster",
            ));
        }
        let proposal = SemanticPreparedStep {
            issuer: Arc::clone(&build.issuer),
            scope: Arc::clone(&build.scope),
            token: build.tokens[1],
        };
        let invocations = self
            .policy_tapes
            .iter()
            .filter(|tape| tape.policy.text_binding._witness.reader_token == proposal.token)
            .map(|tape| tape.invocation)
            .collect::<Vec<_>>();
        if invocations.len() > 1 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        for invocation in invocations {
            self.finish_prepared_policy_invocation(
                &proposal,
                invocation,
                self.stream.cu_stream() as u64,
            )?;
        }
        Ok(())
    }
}
