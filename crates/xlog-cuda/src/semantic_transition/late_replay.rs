//! Cold acquisition of an external result, committed with its native acknowledgement.

use super::acknowledgement_archive::{self, acknowledgement_records};
use super::verification_receipts::OriginalObservationOrigin;
use super::*;
#[path = "intent_lineage.rs"]
mod intent_lineage;
use crate::semantic_hypergraph::{
    SemanticResidentDecodedStatement, SemanticResidentDecodedSupport,
};
use crate::semantic_training_view::{
    SemanticTrainingReplayAppendEntry, SemanticTrainingReplayAppendHeader,
    SemanticTrainingViewBasis, TrainingViewRowDescriptor,
};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PublicationReplayDeliveryRow {
    entry: SemanticTrainingReplayAppendEntry,
    payload: u64,
    payload_bytes: u64,
    row_descriptor: TrainingViewRowDescriptor,
    row_bytes: u64,
    row_bytes_len: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PublicationReplayDeliveryInput {
    delivery: PublicationDeliveryInput,
    receiver_outcome: u64,
    receiver_outcome_bytes: u64,
    receiver_outcome_digest: Identity256,
    receiver_disposition: u64,
    rows: u64,
    row_count: u64,
    arena_descriptors: u64,
    arena_descriptor_count: u64,
    arena_raw: u64,
    arena_raw_bytes: u64,
    observations: u64,
    observation_count: u64,
    statements: u64,
    supports: u64,
    native_work: u64,
}

// SAFETY: the device wire consists exclusively of integral scalar records.
unsafe impl DeviceRepr for PublicationReplayDeliveryInput {}
// SAFETY: each row is a padding-free collection of integer native ABI records.
unsafe impl DeviceRepr for PublicationReplayDeliveryRow {}
const _: () = assert!(size_of::<PublicationReplayDeliveryInput>() == 232);
const _: () = assert!(std::mem::offset_of!(PublicationReplayDeliveryInput, receiver_outcome) == 88);
const _: () = assert!(std::mem::offset_of!(PublicationReplayDeliveryInput, rows) == 144);
const _: () = assert!(std::mem::offset_of!(PublicationReplayDeliveryInput, observations) == 192);
const _: () = assert!(std::mem::offset_of!(PublicationReplayDeliveryInput, native_work) == 224);
const _: () = assert!(size_of::<PublicationReplayDeliveryRow>() == 1072);

type LoadedReplayDeliveryAcquisition<V> = (
    SemanticReplayMaterial,
    SemanticReplayTrainingViewRow,
    Vec<u8>,
    Identity256,
    Identity256,
    Identity256,
    V,
    SemanticResolvedReplayMaterial,
    SemanticReplayLoadedMaterial,
);

type LoadedCodingProposal = (
    SemanticReplayMaterial,
    SemanticResolvedReplayMaterial,
    SemanticReplayLoadedMaterial,
);

struct ReplayUpload<T: DeviceRepr> {
    device: TrackedCudaSlice<T>,
    write: crate::device::RetainedDeviceWrite<T>,
    admitted: bool,
    completed: bool,
}

impl<T: DeviceRepr + 'static> ReplayUpload<T> {
    fn new_with_owner(
        provider: &CudaKernelProvider,
        stream: &Arc<CudaStream>,
        source: &[T],
        material_reservation: Option<&Arc<SemanticReplayMaterialReservation>>,
    ) -> Result<Self, SemanticTransitionError> {
        let device = allocate_publication(provider, source.len())?;
        let owner = material_reservation.map(|owner| Arc::clone(owner) as Arc<dyn Send + Sync>);
        let write = crate::device::RetainedDeviceWrite::new_with_owner(
            stream,
            source,
            device.view(),
            owner,
        )
        .map_err(|error| runtime_error("replay acquisition original upload staging", error))?;
        Ok(Self {
            device,
            write,
            admitted: false,
            completed: false,
        })
    }

    fn complete(
        &mut self,
        provider: &CudaKernelProvider,
        stream: &Arc<CudaStream>,
    ) -> Result<(), SemanticTransitionError> {
        if !self.admitted {
            provider.admit_launch_metadata_htod(size_of::<T>() * self.device.len());
            self.admitted = true;
        }
        if !self.write.entered() {
            self.write
                .enqueue(stream)
                .map_err(|error| runtime_error("replay acquisition original upload", error))?;
        }
        self.write
            .resolve()
            .map_err(|error| runtime_error("replay acquisition upload completion", error))?;
        self.completed = true;
        Ok(())
    }
}

struct ReplayAuthenticationRange {
    range: PublicationRange,
    capacity: usize,
    read: Option<Mutex<PublicationRead<u8>>>,
}

struct ReplayAuthenticationReads {
    ranges: Vec<ReplayAuthenticationRange>,
    control: Mutex<PublicationRead<PublicationControl>>,
    _reservation: Option<Arc<SemanticReplayMaterialReservation>>,
}

pub(super) struct ReplayAuthentication {
    recipient: Identity256,
    key: [u8; 32],
    effect: Vec<u8>,
    task_identity: Identity256,
    task_epoch: u64,
    reader: Option<u64>,
    refusal: Option<u64>,
    lease: Option<Arc<Mutex<SemanticPublishedLease>>>,
    reads: Option<Arc<ReplayAuthenticationReads>>,
    pub(super) verified: bool,
    pub(super) completed: bool,
    _effect_reservation: Option<SemanticReplayMaterialReservation>,
}

pub(super) struct PendingReplayDelivery {
    base: SemanticPublishedIdentity,
    fuel: u64,
    eligible: bool,
    lease_token: u64,
    completion: Option<SemanticPublishedIdentity>,
    training_publication: Option<(
        Arc<SemanticTrainingViewArena>,
        crate::semantic_training_view::PreparedTrainingAppendPublication,
    )>,
    #[cfg(feature = "semantic-policy")]
    context_acknowledgement: Option<
        Arc<std::sync::Mutex<model_context_continuation::OriginalModelContextAcknowledgement>>,
    >,
    receipt: ReplayUpload<u8>,
    receiver_outcome: ReplayUpload<u8>,
    payload: ReplayUpload<u8>,
    rows: ReplayUpload<PublicationReplayDeliveryRow>,
    observations: ReplayUpload<u64>,
    statements: ReplayUpload<SemanticResidentDecodedStatement>,
    supports: ReplayUpload<SemanticResidentDecodedSupport>,
    input: ReplayUpload<PublicationReplayDeliveryInput>,
    support_references: Vec<(Identity256, crate::SemanticRootSupport)>,
    native_work: Option<DeviceMemoryView<u64>>,
    control: Arc<std::sync::Mutex<PublicationRead<PublicationControl>>>,
    header: Arc<std::sync::Mutex<PublicationRead<PublicationHeader>>>,
    kernel_entered: bool,
    kernel_submitted: bool,
    _row_custody: Option<Arc<dyn Send + Sync>>,
    _material_reservation: Option<Arc<SemanticReplayMaterialReservation>>,
    _support_allocation_owner: Option<Arc<SemanticReplayMaterialReservation>>,
    _receipt_allocation_owners: Vec<Arc<SemanticReplayMaterialReservation>>,
}

pub(super) fn receiver_mac_matches(body: &[u8], mac: &[u8], key: &[u8; 32]) -> bool {
    if mac.len() != 32 || key.iter().all(|&byte| byte == 0) {
        return false;
    }
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for (index, &byte) in key.iter().enumerate() {
        ipad[index] ^= byte;
        opad[index] ^= byte;
    }
    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(body);
    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner.finalize());
    let expected: [u8; 32] = outer.finalize().into();
    mac.iter()
        .zip(expected)
        .fold(0u8, |difference, (actual, expected)| {
            difference | (*actual ^ expected)
        })
        == 0
}

fn verify_result_receipt(
    receipt: &[u8],
    stable_intent: Identity256,
    predecessor_logical: Identity256,
    successor_logical: Identity256,
    delivery_digest: Identity256,
    action_identity: Identity256,
    record_digest: Identity256,
    evidence_digest: Identity256,
    disposition: u64,
    key: &[u8; 32],
) -> Result<Identity256, SemanticTransitionError> {
    const DOMAIN: &[u8] = b"xlog.replay.result.v2\0";
    let disposition_offset = DOMAIN.len() + 7 * 32;
    let body_len = disposition_offset + 8;
    if receipt.len() != body_len + 32
        || !receipt.starts_with(DOMAIN)
        || !matches!(disposition, 1 | 2)
        || receipt[disposition_offset..body_len] != disposition.to_le_bytes()
    {
        return Err(publication_input_error(
            "external result receipt has another canonical extent",
        ));
    }
    for (ordinal, expected) in [
        stable_intent,
        delivery_digest,
        predecessor_logical,
        successor_logical,
        action_identity,
        record_digest,
        evidence_digest,
    ]
    .into_iter()
    .enumerate()
    {
        let offset = DOMAIN.len() + ordinal * 32;
        if expected == Identity256::default()
            || receipt[offset..offset + 32] != expected.as_bytes()[..]
        {
            return Err(publication_input_error(
                "external result receipt differs from its original action or complete record",
            ));
        }
    }
    if !receiver_mac_matches(&receipt[..body_len], &receipt[body_len..], key) {
        return Err(publication_input_error(
            "external result receipt MAC does not match the trusted issuer",
        ));
    }
    Ok(Identity256::from_bytes(Sha256::digest(receipt).into()))
}

fn append_header(
    bytes: &[u8],
) -> Result<SemanticTrainingReplayAppendHeader, SemanticTransitionError> {
    if bytes.len() < size_of::<SemanticTrainingReplayAppendHeader>() {
        return Err(publication_input_error(
            "replay append queue has no complete header",
        ));
    }
    // SAFETY: the checked header is an all-bit-valid scalar ABI record.
    let header = unsafe {
        bytes
            .as_ptr()
            .cast::<SemanticTrainingReplayAppendHeader>()
            .read_unaligned()
    };
    let extent = usize::try_from(header.count)
        .ok()
        .and_then(|count| count.checked_mul(size_of::<SemanticTrainingReplayAppendEntry>()))
        .and_then(|size| size.checked_add(size_of::<SemanticTrainingReplayAppendHeader>()));
    if header.abi != 1
        || header.original_stage > 1
        || (header.original_stage == 0
            && (header.original_count != 0
                || header.original_prefix_identity != Identity256::default()
                || header.count != 0))
        || (header.original_stage == 1
            && (header.original_count == 0
                || header.original_prefix_identity == Identity256::default()))
        || header.count > header.capacity
        || header.eligible_count > header.count
        || extent != Some(bytes.len())
    {
        return Err(publication_input_error(
            "replay append queue has inconsistent extents",
        ));
    }
    Ok(header)
}

fn append_payload(
    descriptor: &TrainingViewRowDescriptor,
    raw: &[u8],
    canonical: &[u8],
    result_receipt: &[u8],
) -> Result<Vec<u8>, SemanticTransitionError> {
    let length = 40usize
        .checked_add(size_of::<TrainingViewRowDescriptor>())
        .and_then(|size| size.checked_add(raw.len()))
        .and_then(|size| size.checked_add(canonical.len()))
        .and_then(|size| size.checked_add(result_receipt.len()))
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    let mut bytes = Vec::with_capacity(length);
    for value in [
        1,
        size_of::<TrainingViewRowDescriptor>() as u64,
        raw.len() as u64,
        canonical.len() as u64,
        result_receipt.len() as u64,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    // SAFETY: repr(C) scalar descriptor is fully initialized, including its nested origin.
    bytes.extend_from_slice(unsafe {
        std::slice::from_raw_parts(
            (descriptor as *const TrainingViewRowDescriptor).cast::<u8>(),
            size_of::<TrainingViewRowDescriptor>(),
        )
    });
    bytes.extend_from_slice(raw);
    bytes.extend_from_slice(canonical);
    bytes.extend_from_slice(result_receipt);
    Ok(bytes)
}

struct AppendPayload<'a> {
    entry: SemanticTrainingReplayAppendEntry,
    descriptor: TrainingViewRowDescriptor,
    raw: &'a [u8],
    canonical: &'a [u8],
    result_receipt: &'a [u8],
}

fn append_rows_from_ranges<'a, B: PublicationByteStorage>(
    entries: &'a PublicationMaterialRangeStorage<B>,
    payload: &'a PublicationMaterialRangeStorage<B>,
) -> Result<Vec<AppendPayload<'a>>, SemanticTransitionError> {
    let header = append_header(&entries.bytes)?;
    if header.payload_used_bytes != payload.bytes.len() as u64
        || header.payload_capacity_bytes != payload.capacity as u64
        || header.payload_used_bytes > header.payload_capacity_bytes
    {
        return Err(publication_input_error(
            "append queue lost its complete original payload",
        ));
    }
    let mut cursor = 0usize;
    let mut chain = Identity256::default();
    let mut eligible_count = 0;
    let mut original_pairs = BTreeSet::new();
    let mut closed_intents = BTreeSet::new();
    let mut previous_intent = None;
    let mut rows = Vec::with_capacity(header.count as usize);
    for ordinal in 0..header.count as usize {
        let offset = size_of::<SemanticTrainingReplayAppendHeader>()
            + ordinal * size_of::<SemanticTrainingReplayAppendEntry>();
        let entry_bytes =
            &entries.bytes[offset..offset + size_of::<SemanticTrainingReplayAppendEntry>()];
        // SAFETY: the complete scalar roster was checked by append_header.
        let entry = unsafe {
            entry_bytes
                .as_ptr()
                .cast::<SemanticTrainingReplayAppendEntry>()
                .read_unaligned()
        };
        let end = usize::try_from(entry.payload_length_bytes)
            .ok()
            .and_then(|length| cursor.checked_add(length))
            .filter(|&end| end <= payload.bytes.len())
            .ok_or_else(|| publication_input_error("append payload is incomplete"))?;
        let packed = &payload.bytes[cursor..end];
        if entry.payload_offset_bytes != cursor as u64
            || packed.len() < 40 + size_of::<TrainingViewRowDescriptor>()
        {
            return Err(publication_input_error(
                "append payload has another original extent",
            ));
        }
        let word = |index: usize| {
            u64::from_le_bytes(
                packed[index * 8..index * 8 + 8]
                    .try_into()
                    .expect("checked envelope header"),
            )
        };
        let length = |index: usize| {
            usize::try_from(word(index)).map_err(|_| SemanticTransitionError::GenerationExhausted)
        };
        let raw_begin = 40 + size_of::<TrainingViewRowDescriptor>();
        let raw_end = raw_begin
            .checked_add(length(2)?)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let canonical_end = raw_end
            .checked_add(length(3)?)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let result_len = length(4)?;
        if word(0) != 1
            || word(1) != size_of::<TrainingViewRowDescriptor>() as u64
            || length(3)? == 0
            || canonical_end.checked_add(result_len) != Some(packed.len())
            || result_len != b"xlog.replay.result.v2\0".len() + 8 * 32 + 8
            || !packed[canonical_end..].starts_with(b"xlog.replay.result.v2\0")
        {
            return Err(publication_input_error(
                "append lost its complete row or original signed result",
            ));
        }
        // SAFETY: the fixed initialized descriptor is inside the checked complete envelope.
        let descriptor = unsafe {
            packed[40..]
                .as_ptr()
                .cast::<TrainingViewRowDescriptor>()
                .read_unaligned()
        };
        if !matches!(entry.disposition, 1 | 2)
            || (entry.disposition == 2
                && (entry.row_ordinal != u64::MAX || descriptor.raw_offset != 0))
            || !original_pairs.insert((
                entry.stable_intent,
                entry.original_origin.predecessor_logical,
                entry.original_origin.successor_logical,
            ))
            || descriptor.ordinal != entry.row_ordinal
            || descriptor.basis != 1
            || descriptor.origin.present != 1
            || descriptor.origin != entry.original_origin
            || descriptor.content_identity != entry.content_identity
            || descriptor.raw_bytes != (raw_end - raw_begin) as u64
            || words_identity(entry.stable_intent) == Identity256::default()
            || words_identity(entry.result_receipt_digest)
                != Identity256::from_bytes(Sha256::digest(&packed[canonical_end..]).into())
        {
            return Err(publication_input_error(
                "append row differs from its committed origin or signed result",
            ));
        }
        if previous_intent != Some(entry.stable_intent) {
            if !closed_intents.insert(entry.stable_intent) {
                return Err(publication_input_error(
                    "replay contributor batch is not contiguous",
                ));
            }
            previous_intent = Some(entry.stable_intent);
        }
        chain = publication_fold_digest(
            chain,
            56,
            ordinal as u64,
            Identity256::from_bytes(Sha256::digest(entry_bytes).into()),
        );
        chain = publication_fold_digest(
            chain,
            57,
            ordinal as u64,
            Identity256::from_bytes(Sha256::digest(packed).into()),
        );
        rows.push(AppendPayload {
            entry,
            descriptor,
            raw: &packed[raw_begin..raw_end],
            canonical: &packed[raw_end..canonical_end],
            result_receipt: &packed[canonical_end..],
        });
        eligible_count += u64::from(entry.disposition == 1);
        cursor = end;
    }
    if cursor != payload.bytes.len()
        || chain != header.chain_head
        || eligible_count != header.eligible_count
    {
        return Err(publication_input_error(
            "append chain does not cover its complete original payload",
        ));
    }
    Ok(rows)
}

impl SemanticTransitionSession {
    fn replay_delivery_allocation_bound(
        &self,
        lease: &SemanticPublishedLease,
        replay: Option<&SemanticReplayMaterial>,
        row_bytes: usize,
        canonical_row: &[u8],
        delivery_receipt: &[u8],
        receiver_outcome: &[u8],
        result_receipt: &[u8],
    ) -> Result<u64, SemanticTransitionError> {
        let mut bytes = 0u64;
        let mut add = |count: u64, width: u64| -> Result<(), SemanticTransitionError> {
            bytes = bytes
                .checked_add(
                    count
                        .checked_mul(width)
                        .ok_or(SemanticTransitionError::GenerationExhausted)?,
                )
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            Ok(())
        };
        let length = |role: u64| -> Result<u64, SemanticTransitionError> {
            lease
                .directory
                .iter()
                .find(|range| (range.role, range.index) == (role, 0))
                .map(|range| range.length_bytes)
                .ok_or(SemanticTransitionError::ObservationMismatch)
        };
        for role in [1, 2, 30, 31, 32, 56, 57] {
            add(length(role)?.max(1), 2)?;
            add(
                1,
                (size_of::<PublicationMaterialRange>() + size_of::<PublicationRead<u8>>()) as u64,
            )?;
        }
        let intents = length(30)?
            .checked_sub(size_of::<IntentQueueHeader>() as u64)
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            / size_of::<IntentEntry>() as u64;
        add(intents, size_of::<SemanticOutputIntent>() as u64)?;
        add(intents, length(31)?)?;
        add(length(31)?, 1)?;
        let acknowledgements = length(32)?
            .checked_sub(size_of::<AcknowledgementQueueHeader>() as u64)
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            / size_of::<AcknowledgementEntry>() as u64;
        add(
            acknowledgements,
            size_of::<acknowledgement_archive::ArchivedAcknowledgement>() as u64
                + OwnedPublicationMaterial::tree_node_bound(size_of::<[u8; 32]>(), 0)? as u64,
        )?;
        add(length(32)?, 1)?;
        let appends = length(56)?
            .checked_sub(size_of::<SemanticTrainingReplayAppendHeader>() as u64)
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            / size_of::<SemanticTrainingReplayAppendEntry>() as u64;
        add(
            appends,
            size_of::<AppendPayload<'_>>() as u64
                + OwnedPublicationMaterial::tree_node_bound(
                    size_of::<([u64; 4], [u64; 4], [u64; 4])>(),
                    0,
                )? as u64
                + OwnedPublicationMaterial::tree_node_bound(size_of::<Identity256>(), 0)? as u64
                + size_of::<&AppendPayload<'_>>() as u64,
        )?;
        let packed = 40u64
            .checked_add(size_of::<TrainingViewRowDescriptor>() as u64)
            .and_then(|n| n.checked_add(row_bytes as u64))
            .and_then(|n| n.checked_add(canonical_row.len() as u64))
            .and_then(|n| n.checked_add(result_receipt.len() as u64))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        add(packed.max(1), 2)?;
        add(delivery_receipt.len().max(1) as u64, 1)?;
        add(receiver_outcome.len().max(1) as u64, 1)?;
        // The loaded row already owns its row-sized Arc metadata reservation.
        // Charge the additional receipt field and the receipt's real Arc here.
        add(
            1,
            (size_of::<Arc<SemanticReplayMaterialReservation>>()
                + size_of::<SemanticReplayMaterialReservation>()
                + 2 * size_of::<usize>()) as u64,
        )?;
        let count = self
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .0
            .spec
            .task_ground
            .bindings()
            .len() as u64;
        add(count, 4 * size_of::<[u64; 18]>() as u64)?;
        add(4, size_of::<u64>() as u64)?;
        add(
            count,
            2 * size_of::<SemanticResidentDecodedStatement>() as u64
                + 2 * size_of::<SemanticResidentDecodedSupport>() as u64
                + size_of::<(u32, crate::SemanticRootSupport)>() as u64
                + size_of::<(Identity256, crate::SemanticRootSupport)>() as u64,
        )?;
        add(
            1,
            size_of::<PublicationReplayDeliveryInput>() as u64
                + 4 * size_of::<PublicationControl>() as u64
                + 2 * size_of::<PublicationHeader>() as u64,
        )?;
        add(
            1,
            (size_of::<std::sync::Mutex<PublicationRead<PublicationControl>>>()
                + size_of::<std::sync::Mutex<PublicationRead<PublicationHeader>>>()
                + 4 * size_of::<usize>()) as u64,
        )?;
        let mut evidence = (REPLAY_EVIDENCE_MAGIC.len() + 4 + 32 + 80 + 4) as u64;
        for range in replay
            .into_iter()
            .flat_map(|replay| &replay.evidence.ranges)
        {
            evidence = evidence
                .checked_add((24 + size_of::<PublicationRange>()) as u64)
                .and_then(|n| n.checked_add(range.bytes.len() as u64))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        add(evidence, 2)?;
        add(1, size_of::<PublicationRange>() as u64)?;
        Ok(bytes)
    }
    /// Full original transport bytes, not their metadata/hash-only projection.
    pub fn replay_append_rows(
        &mut self,
        recipient: Identity256,
        key: &[u8; 32],
        effect: &[u8],
    ) -> Result<(Vec<Vec<u8>>, [u64; 4]), SemanticTransitionError> {
        if self.replay_material_budget.is_some() {
            return Err(publication_input_error(
                "bounded replay append reads require their original byte-owned result",
            ));
        }
        let result = self.replay_append_rows_inner(recipient, key, effect)?;
        Ok((result.rows, result.limits))
    }

    /// Read the complete restored archive, or register the trusted scope of a
    /// native-proven original import before its first parent. Early registration
    /// never authenticates subsequently attached history.
    pub fn replay_loaded_append_rows(
        &mut self,
        recipient: Identity256,
        key: &[u8; 32],
        effect: &[u8],
    ) -> Result<SemanticReplayAppendRows, SemanticTransitionError> {
        if self.replay_material_budget.is_none() {
            return Err(publication_input_error(
                "loaded replay append reads require their original capacity pool",
            ));
        }
        self.replay_append_rows_inner(recipient, key, effect)
    }

    fn require_original_receiver_registration(&self) -> Result<(), SemanticTransitionError> {
        self.ensure_quiescent()?;
        // A publication is never removed, and every state restore retains its
        // original owner before submitting work. Together these native owners
        // distinguish an original import from an unactivated restored bank.
        if self.publication.is_some()
            || self.state_material_restore.is_some()
            || self.publication_preparation.is_some()
            || self.task_ground.is_none()
            || !self.task_observations_authenticated
            || !self.graph.authenticated_supports().is_empty()
        {
            return Err(publication_input_error(
                "receiver registration requires the original task without restored history",
            ));
        }
        #[cfg(feature = "semantic-policy")]
        if self.model_context_continuation.is_some() {
            return Err(publication_input_error(
                "receiver registration must precede model context continuation",
            ));
        }
        Ok(())
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn require_unexecuted_receiver_history(
        &self,
        material: &OwnedPublicationMaterial,
    ) -> Result<(), SemanticTransitionError> {
        let range = |role| {
            material
                .ranges
                .iter()
                .find(|range| (range.range.role, range.range.index) == (role, 0))
                .ok_or(SemanticTransitionError::ObservationMismatch)
        };
        let intents = range(30)?;
        let intent_payload = range(31)?;
        let acknowledgements = range(32)?;
        let appends = range(56)?;
        let append_payload = range(57)?;
        let ground = range(58)?;
        let layout = self
            .task_ground
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .layout;
        layout.validate(&ground.bytes)?;
        // Check header-only extents before the canonical parsers: a nonempty
        // hostile roster cannot allocate temporary rows during this proof.
        if intents.bytes.len() != size_of::<IntentQueueHeader>()
            || acknowledgements.bytes.len() != size_of::<AcknowledgementQueueHeader>()
            || appends.bytes.len() != size_of::<SemanticTrainingReplayAppendHeader>()
            || !output_intents_from_ranges(intents, intent_payload)?.is_empty()
            || !acknowledgement_records(&acknowledgements.bytes, acknowledgements.capacity)?
                .is_empty()
            || !append_rows_from_ranges(appends, append_payload)?.is_empty()
            || self
                .replay_authentication
                .as_ref()
                .is_some_and(|registration| {
                    intent_payload.bytes.as_slice() != registration.effect.as_slice()
                })
            || ground.bytes[64..layout.query_offset]
                .iter()
                .any(|&byte| byte != 0)
        {
            return Err(publication_input_error(
                "unactivated model context cannot carry executed receiver history",
            ));
        }
        Ok(())
    }

    fn replay_append_rows_inner(
        &mut self,
        recipient: Identity256,
        key: &[u8; 32],
        effect: &[u8],
    ) -> Result<SemanticReplayAppendRows, SemanticTransitionError> {
        if recipient == Identity256::default()
            || key.iter().all(|&byte| byte == 0)
            || effect.is_empty()
        {
            return Err(publication_input_error(
                "trusted receiver scope must be nonempty",
            ));
        }
        let task_identity = self
            .task_evaluation_identity()
            .ok_or(SemanticTransitionError::NotBound)?;
        let task_epoch = self.task_evaluation_epoch();
        if self.replay_authentication.as_ref().is_some_and(|original| {
            original.completed
                && (original.task_identity != task_identity || original.task_epoch != task_epoch)
        }) {
            self.ensure_quiescent()?;
            self.replay_authentication = None;
        }
        if let Some(original) = &self.replay_authentication {
            if original.recipient != recipient
                || original.key != *key
                || original.effect != effect
                || original.task_identity != task_identity
                || original.task_epoch != task_epoch
            {
                return Err(publication_input_error(
                    "receiver authentication changed its original scope",
                ));
            }
            if let Some(status) = original.refusal {
                return Err(SemanticTransitionError::PublicationRefused { status });
            }
        } else {
            self.ensure_quiescent()?;
            let effect_bytes = effect
                .len()
                .checked_add(size_of::<ReplayAuthentication>())
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let effect_reservation = self
                .replay_material_budget
                .as_ref()
                .map(|budget| {
                    budget.reserve_auxiliary(
                        u64::try_from(effect_bytes)
                            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                    )
                })
                .transpose()?;
            self.replay_authentication = Some(ReplayAuthentication {
                recipient,
                key: *key,
                effect: effect.to_vec(),
                task_identity,
                task_epoch,
                reader: None,
                refusal: None,
                lease: None,
                reads: None,
                verified: false,
                completed: false,
                _effect_reservation: effect_reservation,
            });
        }
        if self.publication.is_none() {
            self.require_original_receiver_registration()?;
            let reservation = self
                .replay_material_budget
                .as_ref()
                .map(|budget| {
                    budget
                        .reserve_auxiliary(
                            (size_of::<SemanticReplayAppendRows>() + 2 * size_of::<usize>()) as u64,
                        )
                        .map(Arc::new)
                })
                .transpose()?;
            return Ok(SemanticReplayAppendRows {
                rows: Vec::new(),
                limits: [0; 4],
                _reservation: reservation,
            });
        }
        if self
            .replay_authentication
            .as_ref()
            .expect("original receiver authentication")
            .lease
            .is_none()
        {
            let token = self
                .replay_authentication
                .as_ref()
                .expect("original receiver authentication")
                .reader;
            let acquired = if let Some(token) = token {
                self.continue_acquire_reader(token)
            } else {
                let token = self
                    .next_reader
                    .checked_add(1)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                self.replay_authentication
                    .as_mut()
                    .expect("original receiver authentication")
                    .reader = Some(token);
                let acquired = self.acquire();
                if acquired.is_err() && !self.readers.contains_key(&token) {
                    self.replay_authentication
                        .as_mut()
                        .expect("original receiver authentication")
                        .reader = None;
                }
                acquired
            };
            if let Err(SemanticTransitionError::PublicationRefused { status }) = &acquired {
                self.replay_authentication
                    .as_mut()
                    .expect("original receiver authentication")
                    .refusal = Some(*status);
            }
            let acquired = acquired?;
            self.replay_authentication
                .as_mut()
                .expect("original receiver authentication")
                .lease = Some(Arc::new(Mutex::new(acquired)));
        }
        let lease = Arc::clone(
            self.replay_authentication
                .as_ref()
                .and_then(|original| original.lease.as_ref())
                .expect("original receiver lease"),
        );
        let lease = lease
            .lock()
            .map_err(|_| publication_input_error("receiver lease owner poisoned"))?;
        if self
            .replay_authentication
            .as_ref()
            .expect("original receiver authentication")
            .reads
            .is_none()
        {
            if !lease.active || !Arc::ptr_eq(&lease.issuer, &self.publication_issuer) {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let storage = Arc::clone(
                self.publication
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?,
            );
            let mut read_bytes = (size_of::<ReplayAuthenticationReads>()
                + 2 * size_of::<usize>()
                + 8 * size_of::<ReplayAuthenticationRange>()
                + 2 * size_of::<PublicationControl>()) as u64;
            for role in [1, 2, 30, 31, 32, 56, 57, 58] {
                let extent = lease
                    .directory
                    .iter()
                    .find(|range| (range.role, range.index) == (role, 0))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
                    .length_bytes;
                read_bytes = read_bytes
                    .checked_add(
                        extent
                            .checked_mul(2)
                            .ok_or(SemanticTransitionError::GenerationExhausted)?,
                    )
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
            }
            let reservation = self
                .replay_material_budget
                .as_ref()
                .map(|budget| budget.reserve_auxiliary(read_bytes).map(Arc::new))
                .transpose()?;
            let mut ranges = Vec::with_capacity(8);
            for role in [1, 2, 30, 31, 32, 56, 57, 58] {
                let range = *lease
                    .directory
                    .iter()
                    .find(|range| range.role == role && range.index == 0)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if role == 57 && storage.contract_value.replay_payload_capacity_bytes == 0 {
                    if range.length_bytes != 0 || range.offset_bytes != 0 {
                        return Err(SemanticTransitionError::ObservationMismatch);
                    }
                    ranges.push(ReplayAuthenticationRange {
                        range,
                        capacity: 0,
                        read: None,
                    });
                    continue;
                }
                let allocation = storage
                    .allocations
                    .get(range.storage_slot as usize)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let begin = usize::try_from(range.offset_bytes)
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
                let end = usize::try_from(range.length_bytes)
                    .ok()
                    .and_then(|length| begin.checked_add(length))
                    .filter(|&end| end <= allocation.len())
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if begin == end {
                    ranges.push(ReplayAuthenticationRange {
                        range,
                        capacity: allocation.len() - begin,
                        read: None,
                    });
                    continue;
                }
                ranges.push(ReplayAuthenticationRange {
                    range,
                    capacity: allocation.len() - begin,
                    read: Some(Mutex::new(self.stage_publication_read_with_owner(
                        allocation.slice()?.view().slice(begin..end),
                        reservation.clone(),
                    )?)),
                });
            }
            let control =
                Mutex::new(self.stage_publication_read_with_owner(
                    storage.control.view(),
                    reservation.clone(),
                )?);
            self.replay_authentication
                .as_mut()
                .expect("original receiver authentication")
                .reads = Some(Arc::new(ReplayAuthenticationReads {
                ranges,
                control,
                _reservation: reservation,
            }));
        }
        drop(lease);
        let entries = self.read_replay_authentication_range(56)?;
        let payload = self.read_replay_authentication_range(57)?;
        let count = entries
            .bytes
            .len()
            .checked_sub(size_of::<SemanticTrainingReplayAppendHeader>())
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            / size_of::<SemanticTrainingReplayAppendEntry>();
        let node = OwnedPublicationMaterial::tree_node_bound(
            size_of::<([u64; 4], [u64; 4], [u64; 4])>(),
            0,
        )?
        .checked_add(OwnedPublicationMaterial::tree_node_bound(
            size_of::<Identity256>(),
            0,
        )?)
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let row_bytes = count
            .checked_mul(size_of::<AppendPayload<'_>>() + size_of::<Vec<u8>>() + node)
            .and_then(|n| n.checked_add(payload.bytes.len()))
            .and_then(|n| {
                n.checked_add(size_of::<SemanticReplayAppendRows>() + 2 * size_of::<usize>())
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = self
            .replay_material_budget
            .as_ref()
            .map(|budget| {
                budget
                    .reserve_auxiliary(
                        u64::try_from(row_bytes)
                            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                    )
                    .map(Arc::new)
            })
            .transpose()?;
        let rows = append_rows_from_ranges(&entries, &payload)?
            .into_iter()
            .map(|row| row.canonical.to_vec())
            .collect::<Vec<_>>();
        let limits = if rows.is_empty() {
            [0; 4]
        } else {
            let binding = self
                .training_views
                .as_ref()
                .and_then(|arena| arena.replay_capacity())
                .ok_or(SemanticTransitionError::NotBound)?;
            [
                binding.metadata_bytes,
                binding.max_material_bytes,
                binding.total_material_bytes,
                binding.max_evidence_bytes,
            ]
        };
        Ok(SemanticReplayAppendRows {
            rows,
            limits,
            _reservation: reservation,
        })
    }

    fn read_replay_authentication_range(
        &mut self,
        role: u64,
    ) -> Result<PublicationMaterialRange, SemanticTransitionError> {
        let reads = Arc::clone(
            self.replay_authentication
                .as_ref()
                .and_then(|original| original.reads.as_ref())
                .ok_or(SemanticTransitionError::NoPendingLaunch)?,
        );
        let original = reads
            .ranges
            .iter()
            .find(|original| original.range.role == role)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let bytes = if let Some(read) = &original.read {
            let mut read = read
                .lock()
                .map_err(|_| publication_input_error("receiver archive read owner poisoned"))?;
            self.resolve_publication_read(&mut read)?
        } else {
            Vec::new()
        };
        let material = PublicationMaterialRange {
            range: original.range,
            capacity: original.capacity,
            bytes,
        };
        Ok(material)
    }

    /// Authenticate all restored append rows at the current trusted receiver ingress.
    /// The caller passes rows decoded by the complete replay parser, never archival grants.
    pub fn authenticate_replay_appends(
        &mut self,
        recipient: Identity256,
        key: &[u8; 32],
        effect: &[u8],
        acquisitions: &[(
            SemanticReplayMaterial,
            SemanticTrainingViewRow,
            Vec<u8>,
            Identity256,
            Identity256,
            Identity256,
            Vec<(Identity256, Vec<u8>, Vec<u8>)>,
            SemanticResolvedReplayMaterial,
            SemanticReplayLoadedMaterial,
        )],
    ) -> Result<(), SemanticTransitionError> {
        if self.replay_material_budget.is_some() {
            return Err(publication_input_error(
                "bounded append authentication requires original loaded rows",
            ));
        }
        self.authenticate_replay_appends_inner(recipient, key, effect, acquisitions)
    }

    pub fn authenticate_loaded_replay_appends<
        V: AsRef<
            [(
                Identity256,
                SemanticReplayLoadedMaterial,
                SemanticReplayLoadedMaterial,
            )],
        >,
    >(
        &mut self,
        recipient: Identity256,
        key: &[u8; 32],
        effect: &[u8],
        acquisitions: &[(
            SemanticReplayMaterial,
            SemanticReplayTrainingViewRow,
            Vec<u8>,
            Identity256,
            Identity256,
            Identity256,
            V,
            SemanticResolvedReplayMaterial,
            SemanticReplayLoadedMaterial,
        )],
    ) -> Result<(), SemanticTransitionError> {
        let budget = self.replay_material_budget.as_ref().ok_or_else(|| {
            publication_input_error(
                "loaded append authentication requires its original material pool",
            )
        })?;
        if acquisitions
            .iter()
            .any(|(proof, row, _, _, _, _, materials, original, action)| {
                row.bytes.require_budget(budget).is_err()
                    || !Arc::ptr_eq(&proof.proof_owner, &original.proof_owner)
                    || action
                        .bytes
                        .budget()
                        .is_none_or(|original| !budget.same_pool(original))
                    || proof
                        .budget
                        .as_ref()
                        .is_none_or(|original| !budget.same_pool(original))
                    || materials.as_ref().iter().any(|(_, material, metadata)| {
                        [material, metadata].iter().any(|loaded| {
                            loaded
                                .bytes
                                .budget()
                                .is_none_or(|original| !budget.same_pool(original))
                        })
                    })
            })
        {
            return Err(publication_input_error(
                "append row changed its original material pool",
            ));
        }
        self.authenticate_replay_appends_inner(recipient, key, effect, acquisitions)
    }

    fn authenticate_replay_appends_inner<R, B, V>(
        &mut self,
        recipient: Identity256,
        key: &[u8; 32],
        effect: &[u8],
        acquisitions: &[(
            SemanticReplayMaterial,
            SemanticTrainingViewRow<R>,
            Vec<u8>,
            Identity256,
            Identity256,
            Identity256,
            V,
            SemanticResolvedReplayMaterial,
            SemanticReplayLoadedMaterial,
        )],
    ) -> Result<(), SemanticTransitionError>
    where
        R: AsRef<[u8]>,
        B: AsRef<[u8]>,
        V: AsRef<[(Identity256, B, B)]>,
    {
        let original = self
            .replay_authentication
            .as_ref()
            .ok_or(SemanticTransitionError::NoPendingLaunch)?;
        if original.recipient != recipient || original.key != *key || original.effect != effect {
            return Err(publication_input_error(
                "receiver authentication changed its original scope",
            ));
        }
        if Some(original.task_identity) != self.task_evaluation_identity()
            || original.task_epoch != self.task_evaluation_epoch()
        {
            return Err(publication_input_error(
                "receiver authentication changed its original task",
            ));
        }
        if original.completed {
            return Ok(());
        }
        if original.lease.is_none() {
            self.require_original_receiver_registration()?;
            if !acquisitions.is_empty() {
                return Err(publication_input_error(
                    "original receiver registration cannot authenticate archived rows",
                ));
            }
            // Scope registration is not historical authentication. A later
            // restore must either verify its archive or prove an unexecuted
            // dormant target at the canonical state-material ingress.
            self.replay_authentication
                .as_mut()
                .expect("original receiver registration")
                .completed = true;
            return Ok(());
        }
        let verified = original.verified;
        let lease = Arc::clone(
            original
                .lease
                .as_ref()
                .ok_or(SemanticTransitionError::NoPendingLaunch)?,
        );
        let mut lease = lease
            .lock()
            .map_err(|_| publication_input_error("receiver lease owner poisoned"))?;
        if !verified {
            self.verify_replay_authentication(&lease, recipient, key, effect, acquisitions)?;
            self.replay_authentication
                .as_mut()
                .expect("original receiver authentication")
                .verified = true;
        }
        // This private reader has never exported content or model aliases.
        // Its original Release alone may continue through historical poison.
        #[cfg(feature = "semantic-policy")]
        self.require_closed_evaluations()?;
        let _ordinary =
            crate::cuda_graph::reserve_uncaptured_stream(&self.stream).map_err(|error| {
                runtime_error("receiver authentication retirement stream admission", error)
            })?;
        self.retire_original_published_reader(&mut lease, &[], false)?;
        self.steps.remove(&lease.token);
        self.replay_authentication
            .as_mut()
            .expect("original receiver authentication")
            .completed = true;
        if let Some(arena) = &self.training_views {
            arena.authenticate_appends();
        }
        self.task_observations_authenticated = true;
        Ok(())
    }

    fn verify_replay_authentication<R, B, V>(
        &mut self,
        lease: &SemanticPublishedLease,
        recipient: Identity256,
        key: &[u8; 32],
        effect: &[u8],
        acquisitions: &[(
            SemanticReplayMaterial,
            SemanticTrainingViewRow<R>,
            Vec<u8>,
            Identity256,
            Identity256,
            Identity256,
            V,
            SemanticResolvedReplayMaterial,
            SemanticReplayLoadedMaterial,
        )],
    ) -> Result<(), SemanticTransitionError>
    where
        R: AsRef<[u8]>,
        B: AsRef<[u8]>,
        V: AsRef<[(Identity256, B, B)]>,
    {
        if !lease.active || !Arc::ptr_eq(&lease.issuer, &self.publication_issuer) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.task_observations_authenticated = false;
        let ground = self
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .0
            .spec
            .task_ground
            .clone();
        let layout = self
            .task_ground
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .layout;
        let entries = self.read_replay_authentication_range(56)?;
        let payload = self.read_replay_authentication_range(57)?;
        let training_header = append_header(&entries.bytes)?;
        let row_count = entries
            .bytes
            .len()
            .checked_sub(size_of::<SemanticTrainingReplayAppendHeader>())
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            / size_of::<SemanticTrainingReplayAppendEntry>();
        let row_parse_width = size_of::<AppendPayload<'_>>()
            .checked_add(OwnedPublicationMaterial::tree_node_bound(
                size_of::<([u64; 4], [u64; 4], [u64; 4])>(),
                0,
            )?)
            .and_then(|width| {
                width.checked_add(
                    OwnedPublicationMaterial::tree_node_bound(size_of::<[u64; 4]>(), 0).ok()?,
                )
            })
            .and_then(|width| {
                width.checked_add(
                    size_of::<(
                        &SemanticReplayMaterial,
                        &SemanticResolvedReplayMaterial,
                        &SemanticReplayLoadedMaterial,
                    )>() + size_of::<usize>()
                        + size_of::<bool>(),
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let _row_parse_owner = self
            .replay_material_budget
            .as_ref()
            .map(|budget| {
                budget.reserve_auxiliary(
                    (row_count as u64)
                        .checked_mul(row_parse_width as u64)
                        .ok_or(SemanticTransitionError::GenerationExhausted)?,
                )
            })
            .transpose()?;
        let rows = append_rows_from_ranges(&entries, &payload)?;
        let observation_only = matches!(&ground, SemanticTaskGround::Coding { .. })
            && self.coding_delivery_is_observation()?;
        if observation_only && (!rows.is_empty() || !acquisitions.is_empty()) {
            return Err(publication_input_error(
                "evaluation coding observations cannot restore training append rows",
            ));
        }
        if rows.len() != acquisitions.len() {
            return Err(publication_input_error(
                "append authentication requires the complete original committed roster",
            ));
        }
        let arena = self.training_views.as_ref().map(Arc::clone);
        if let Some(arena) = &arena {
            arena.require_append_authentication();
        }
        if !rows.is_empty() && arena.is_none() {
            return Err(SemanticTransitionError::NotBound);
        }
        let intent_entries = self.read_replay_authentication_range(30)?;
        let intent_payload = self.read_replay_authentication_range(31)?;
        let intent_count = intent_entries
            .bytes
            .len()
            .checked_sub(size_of::<IntentQueueHeader>())
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            / size_of::<IntentEntry>();
        let intent_parse_bytes = intent_count
            .checked_mul(
                size_of::<SemanticOutputIntent>()
                    .checked_add(intent_payload.bytes.len())
                    .ok_or(SemanticTransitionError::GenerationExhausted)?,
            )
            .and_then(|bytes| bytes.checked_add(intent_payload.bytes.len()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let _intent_parse_owner = self
            .replay_material_budget
            .as_ref()
            .map(|budget| budget.reserve_auxiliary(intent_parse_bytes as u64))
            .transpose()?;
        let intents = output_intents_from_ranges(&intent_entries, &intent_payload)?;
        let acknowledgements = self.read_replay_authentication_range(32)?;
        let archive_bound = acknowledgement_archive::allocation_bound(
            &acknowledgements.bytes,
            acknowledgements.capacity,
        )?;
        let _archive_owner = self
            .replay_material_budget
            .as_ref()
            .map(|budget| budget.reserve_auxiliary(archive_bound))
            .transpose()?;
        let receipts = acknowledgement_records(&acknowledgements.bytes, acknowledgements.capacity)?;
        let prefix = self.read_replay_authentication_range(1)?;
        let provenance = self.read_replay_authentication_range(2)?;
        let contract = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .contract_value;
        let mut begin = 0;
        while begin < rows.len() {
            let stable = rows[begin].entry.stable_intent;
            let end = rows[begin..]
                .iter()
                .position(|row| row.entry.stable_intent != stable)
                .map_or(rows.len(), |offset| begin + offset);
            let intent = intents
                .iter()
                .find(|intent| identity_words(*intent.stable_identity.as_bytes()) == stable)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let proposals = acquisitions[begin..end]
                .iter()
                .map(|(replay, _, _, _, _, _, _, original, action)| (replay, original, action))
                .collect::<Vec<_>>();
            let contributing = intent_lineage::contributor_indices(
                contract,
                lease.header,
                intent,
                &prefix,
                &provenance,
                &proposals,
            )?;
            if contributing.len() != proposals.len()
                || contributing.iter().copied().ne(0..proposals.len())
            {
                return Err(publication_input_error(
                    "restored batch changed the complete native contributor set",
                ));
            }
            begin = end;
        }
        let source_count = rows
            .len()
            .checked_add(receipts.len())
            .and_then(|count| count.checked_mul(ground.bindings().len()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let metadata_bytes = receipts
            .len()
            .checked_mul(size_of::<(
                Identity256,
                Option<super::verification_receipts::AuthenticatedTaskResults>,
            )>())
            .and_then(|bytes| {
                bytes.checked_add(
                    source_count.checked_mul(size_of::<(u32, crate::SemanticRootSupport)>())?,
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    ground
                        .bindings()
                        .len()
                        .checked_mul(size_of::<[u64; 18]>())?,
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let _metadata_owner = self
            .replay_material_budget
            .as_ref()
            .map(|budget| budget.reserve_auxiliary(metadata_bytes as u64))
            .transpose()?;
        let mut eligible_count = 0;
        let mut restored_training_rows = Vec::with_capacity(acquisitions.len());
        let mut observations = vec![[0u64; 18]; ground.bindings().len()];
        let mut sources = Vec::with_capacity(source_count);
        let mut receiver_results = Vec::with_capacity(receipts.len());
        for record in &receipts {
            let intent = intents
                .iter()
                .find(|intent| intent.stable_identity == record.stable_identity)
                .ok_or_else(|| {
                    publication_input_error(
                        "restored acknowledgement has no original native intent",
                    )
                })?;
            if effect.is_empty() || intent.effect != effect {
                return Err(publication_input_error(
                    "restored receiver outcome changed its original effect",
                ));
            }
            let outcome_capacity = acknowledgement_archive::admit_payload(
                &acknowledgements.bytes,
                acknowledgements.capacity,
                record.stable_identity,
                record.delivery_receipt.len(),
                record.receiver_outcome.len(),
            )?;
            verify_delivery_receipt(&record.delivery_receipt, intent, recipient, key)?;
            if matches!(&ground, SemanticTaskGround::Logical) {
                if !record.receiver_outcome.is_empty()
                    || rows.iter().any(|row| {
                        words_identity(row.entry.stable_intent) == record.stable_identity
                    })
                {
                    return Err(publication_input_error(
                        "logical delivery acknowledgement cannot contain coding results",
                    ));
                }
                receiver_results.push((record.stable_identity, None));
                continue;
            }
            let outcome = super::receiver_outcome::verify_receiver_outcome(
                &record.receiver_outcome,
                record.stable_identity,
                recipient,
                &intent.effect,
                &intent.payload,
                &record.delivery_receipt,
                key,
                outcome_capacity,
            )?;
            let batch_rows = rows
                .iter()
                .filter(|row| words_identity(row.entry.stable_intent) == record.stable_identity)
                .count();
            let common =
                if outcome.disposition == super::receiver_outcome::ReceiverDisposition::Completed {
                    if !observation_only
                        && batch_rows == 0
                        && !intent_lineage::contributor_indices(
                            contract,
                            lease.header,
                            intent,
                            &prefix,
                            &provenance,
                            &[],
                        )?
                        .is_empty()
                    {
                        return Err(publication_input_error(
                            "restored empty result omitted native contributors",
                        ));
                    }
                    let common = super::verification_receipts::verify_completed_tool_result(
                        &ground,
                        outcome.content,
                        record.stable_identity,
                        recipient,
                        &intent.effect,
                        &intent.payload,
                        key,
                        self.replay_material_budget.as_ref(),
                    )?;
                    for (ordinal, (binding, &incoming)) in ground
                        .bindings()
                        .iter()
                        .zip(&common.observations)
                        .enumerate()
                    {
                        sources.push((
                            binding.observation_record,
                            crate::SemanticRootSupport::Authenticated {
                                binding_ordinal: ordinal as u64,
                                observation: incoming,
                            },
                        ));
                        let accumulated = observations[ordinal][1] | incoming[1];
                        observations[ordinal] = incoming;
                        observations[ordinal][1] = accumulated;
                    }
                    Some(common)
                } else {
                    if batch_rows != 0 {
                        return Err(publication_input_error(
                            "refused receiver outcome acquired action rows",
                        ));
                    }
                    None
                };
            receiver_results.push((record.stable_identity, common));
        }
        for (
            archived,
            (
                replay,
                row,
                canonical,
                action_identity,
                record_digest,
                evidence_digest,
                materials,
                _,
                _,
            ),
        ) in rows.iter().zip(acquisitions)
        {
            let stable = words_identity(archived.entry.stable_intent);
            let index = intents
                .iter()
                .position(|intent| intent.stable_identity == stable)
                .ok_or_else(|| {
                    publication_input_error("restored append has no original native intent")
                })?;
            let intent = &intents[index];
            let authenticated = super::verification_receipts::verify_acquired_receipts(
                &ground,
                materials
                    .as_ref()
                    .iter()
                    .map(|(identity, material, metadata)| {
                        (*identity, material.as_ref(), Some(metadata.as_ref()))
                    }),
                stable,
                &intent.effect,
                &intent.payload,
                OriginalObservationOrigin::Action(*action_identity),
                key,
                self.replay_material_budget.as_ref(),
            )?;
            let common = receiver_results
                .iter()
                .find(|(identity, _)| *identity == stable)
                .and_then(|(_, result)| result.as_ref())
                .ok_or_else(|| {
                    publication_input_error(
                        "restored action row lacks a completed receiver outcome",
                    )
                })?;
            if authenticated
                .receipts
                .iter()
                .map(|receipt| (receipt.identity, receipt.outcome))
                .ne(common
                    .receipts
                    .iter()
                    .map(|receipt| (receipt.identity, receipt.outcome)))
            {
                return Err(publication_input_error(
                    "restored action row changed its actual completed receipt roster",
                ));
            }
            let disposition = authenticated.disposition;
            if disposition != archived.entry.disposition {
                return Err(publication_input_error(
                    "restored result changed its authenticated learning disposition",
                ));
            }
            let arena = arena.as_ref().ok_or(SemanticTransitionError::NotBound)?;
            let row_ordinal = arena
                .original_count()
                .checked_add(eligible_count)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let descriptor = if disposition == 1 {
                arena.prepare_append_row(row_ordinal, row)?
            } else {
                arena.prepare_rejected_row(row)?
            };
            eligible_count += usize::from(disposition == 1);
            for (ordinal, (binding, &incoming)) in ground
                .bindings()
                .iter()
                .zip(&authenticated.observations)
                .enumerate()
            {
                sources.push((
                    binding.observation_record,
                    crate::SemanticRootSupport::Authenticated {
                        binding_ordinal: ordinal as u64,
                        observation: incoming,
                    },
                ));
            }
            if effect.is_empty()
                || intent.effect != effect
                || canonical.as_slice() != archived.canonical
                || row.bytes.as_ref() != archived.raw
                || row.origin != Some(replay.training_view_origin()?)
                || Identity256::from_bytes(Sha256::digest(replay.evidence.encode()?).into())
                    != *evidence_digest
            {
                return Err(publication_input_error(
                    "restored append differs from its current receiver or complete original replay",
                ));
            }
            // SAFETY: both scalar descriptors have no padding and are fully initialized.
            let descriptor_bytes = unsafe {
                std::slice::from_raw_parts(
                    (&descriptor as *const TrainingViewRowDescriptor).cast::<u8>(),
                    size_of::<TrainingViewRowDescriptor>(),
                )
            };
            let archived_bytes = unsafe {
                std::slice::from_raw_parts(
                    (&archived.descriptor as *const TrainingViewRowDescriptor).cast::<u8>(),
                    size_of::<TrainingViewRowDescriptor>(),
                )
            };
            if descriptor_bytes != archived_bytes {
                return Err(publication_input_error(
                    "restored training row differs from its original committed descriptor",
                ));
            }
            if disposition == 1 {
                restored_training_rows
                    .push(self.retain_training_append(descriptor, row, canonical, None)?);
            }
            let receipt = &receipts
                .iter()
                .find(|record| record.stable_identity == stable)
                .ok_or_else(|| {
                    publication_input_error(
                        "restored append has no original delivery acknowledgement",
                    )
                })?
                .delivery_receipt;
            let delivery_digest = verify_delivery_receipt(receipt, intent, recipient, key)?;
            if verify_result_receipt(
                archived.result_receipt,
                stable,
                replay.predecessor_identity().logical_digest,
                replay.successor_identity().logical_digest,
                delivery_digest,
                *action_identity,
                *record_digest,
                *evidence_digest,
                disposition,
                key,
            )? != words_identity(archived.entry.result_receipt_digest)
            {
                return Err(publication_input_error(
                    "restored result proof differs from its original action pair",
                ));
            }
        }
        let prepared = self
            .graph
            .prepare_authenticated_supports(&sources)
            .map_err(SemanticTransitionError::Semantic)?;
        let retained = self.graph.authenticated_supports();
        if prepared.references.len() != retained.len()
            || prepared
                .references
                .iter()
                .any(|reference| !retained.contains(reference))
        {
            return Err(publication_input_error(
                "restored semantic supports differ from the complete authenticated result archive",
            ));
        }
        let plane = self.read_replay_authentication_range(58)?;
        layout.validate(&plane.bytes)?;
        for (ordinal, observation) in observations.iter().enumerate() {
            let offset = 64 + ordinal * 18 * size_of::<u64>();
            for (word, expected) in observation.iter().enumerate() {
                let start = offset + word * size_of::<u64>();
                let actual = u64::from_le_bytes(
                    plane.bytes[start..start + size_of::<u64>()]
                        .try_into()
                        .expect("validated task observation word"),
                );
                if actual != *expected {
                    return Err(publication_input_error(
                        "restored observation plane differs from the complete authenticated result archive",
                    ));
                }
            }
        }
        let reads = Arc::clone(
            self.replay_authentication
                .as_ref()
                .and_then(|original| original.reads.as_ref())
                .ok_or(SemanticTransitionError::NoPendingLaunch)?,
        );
        let current = {
            let mut control = reads
                .control
                .lock()
                .map_err(|_| publication_input_error("receiver publication read owner poisoned"))?;
            self.resolve_publication_read(&mut control)?[0]
        };
        if current.word != lease.identity.word || current.instance != lease.identity.instance {
            return Err(publication_input_error(
                "append authentication changed its original acquired publication",
            ));
        }
        if let Some(arena) = &arena {
            let prepared = arena.prepare_training_appends(
                0,
                training_header.count,
                training_header.chain_head,
                restored_training_rows,
            )?;
            arena.publish_training_appends(&prepared)?;
        }
        Ok(())
    }

    /// Reconstitute only original committed append slots from the complete archived queue.
    /// No new result, authority, selection, or publication is produced by restoration.
    pub(super) fn stage_restored_replay_append_rows(
        &self,
        restoration: &Arc<state_restoration::OriginalStateMaterialRestore>,
        material: &state_restoration::RestoredReplayAppendMaterial,
    ) -> Result<
        (
            Vec<state_restoration::OriginalDeviceWrite>,
            Option<Arc<SemanticReplayMaterialReservation>>,
        ),
        SemanticTransitionError,
    > {
        self.require_restored_replay_append_material(restoration, material)?;
        let (entries, payload) = material.records();
        let arena = Arc::clone(
            self.training_views
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let header = append_header(&entries.bytes)?;
        if header.capacity != (arena.row_capacity() - arena.original_slot_boundary()) as u64
            || header.original_stage != u64::from(arena.is_materialized())
            || header.original_count != arena.original_count() as u64
            || header.original_prefix_identity != arena.original_prefix_identity()
        {
            return Err(publication_input_error(
                "restored replay queue differs from its original cold arena",
            ));
        }
        let row_metadata_bytes = OwnedPublicationMaterial::tree_node_bound(
            size_of::<([u64; 4], [u64; 4], [u64; 4])>(),
            0,
        )?
        .checked_add(OwnedPublicationMaterial::tree_node_bound(
            size_of::<[u64; 4]>(),
            0,
        )?)
        .ok_or(SemanticTransitionError::GenerationExhausted)?
        .checked_add(size_of::<AppendPayload<'_>>())
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let parse_bytes = usize::try_from(header.count)
            .ok()
            .and_then(|count| count.checked_mul(row_metadata_bytes))
            .and_then(|bytes| {
                bytes.checked_add(
                    size_of::<SemanticReplayMaterialReservation>() + 2 * size_of::<usize>(),
                )
            })
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let _parse_reservation = self
            .replay_material_budget
            .as_ref()
            .map(|budget| budget.reserve_auxiliary(parse_bytes))
            .transpose()?;
        let rows = append_rows_from_ranges(entries, payload)?;
        if !rows.is_empty() {
            arena.require_append_authentication();
        }
        let (descriptors, raw) = arena.append_storage();
        let mut eligible_count = 0;
        let mut write_count = 0usize;
        let mut write_bytes =
            size_of::<SemanticReplayMaterialReservation>() + 2 * size_of::<usize>();
        for row in rows.iter().filter(|row| row.entry.disposition == 1) {
            write_count = write_count
                .checked_add(2)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            for bytes in [
                state_restoration::OriginalDeviceWrite::metadata_allocation_bytes::<u8>(),
                state_restoration::OriginalDeviceWrite::metadata_allocation_bytes::<
                    TrainingViewRowDescriptor,
                >(),
                row.raw.len().max(1),
                size_of::<TrainingViewRowDescriptor>(),
            ] {
                write_bytes = write_bytes
                    .checked_add(bytes)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
            }
        }
        let reservation = self
            .replay_material_budget
            .as_ref()
            .map(|budget| {
                budget
                    .reserve_auxiliary(
                        u64::try_from(write_bytes)
                            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                    )
                    .map(Arc::new)
            })
            .transpose()?;
        let mut writes = Vec::with_capacity(write_count);
        for row in &rows {
            if row.entry.disposition == 2 {
                continue;
            }
            let descriptor = row.descriptor;
            let raw_len = row.raw.len();
            let expected_ordinal = arena
                .original_count()
                .checked_add(eligible_count)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            eligible_count += 1;
            if row.entry.row_ordinal != expected_ordinal as u64
                || descriptor.raw_offset
                    != (arena.physical_ordinal(expected_ordinal)? as u64)
                        .checked_mul(raw_len as u64)
                        .ok_or(SemanticTransitionError::GenerationExhausted)?
                || descriptor
                    .raw_offset
                    .checked_add(descriptor.raw_bytes)
                    .is_none_or(|end| end > raw.len() as u64)
                || expected_ordinal >= descriptors.len()
            {
                return Err(publication_input_error(
                    "restored append row exceeds its original reserved storage",
                ));
            }
            let target =
                raw.slice(descriptor.raw_offset as usize..descriptor.raw_offset as usize + raw_len);
            let physical = arena.physical_ordinal(expected_ordinal)?;
            let target_descriptor = descriptors.slice(physical..physical + 1);
            writes.push(state_restoration::OriginalDeviceWrite::new_with_owner(
                &self.stream,
                row.raw,
                target,
                false,
                reservation.clone(),
            )?);
            writes.push(state_restoration::OriginalDeviceWrite::new_with_owner(
                &self.stream,
                &[descriptor],
                target_descriptor,
                false,
                reservation.clone(),
            )?);
        }
        Ok((writes, reservation))
    }

    /// Only an original evaluation fork can observe coding results without appending training.
    pub fn coding_delivery_is_observation(&self) -> Result<bool, SemanticTransitionError> {
        let Some(origin) = &self.evaluation_fork else {
            return Ok(false);
        };
        let contract = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .contract_value;
        let task = &self
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .0;
        if origin.member() < 2
            || self.training_views.is_some()
            || self.task_training_domain().is_some()
            || !matches!(&task.spec.task_ground, SemanticTaskGround::Coding { .. })
            || contract.nonterminal_intent_mode != 1
            || contract.replay_row_capacity != 0
            || contract.replay_original_slot_boundary != 0
            || contract.replay_raw_capacity_bytes != 0
            || contract.replay_payload_capacity_bytes != 0
            || contract.replay_metadata_bytes != 0
            || contract.replay_max_material_bytes != 0
            || contract.replay_total_material_bytes != 0
            || contract.replay_max_evidence_bytes != 0
            || contract.replay_reservation_identity
                != crate::semantic_training_view::replay_reservation_identity(None, None)
        {
            return Err(publication_input_error(
                "evaluation coding delivery changed its original observation contract",
            ));
        }
        Ok(true)
    }

    /// Original finite transport limits, sealed before any publication or result acquisition.
    pub fn replay_append_limits(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<[u64; 4], SemanticTransitionError> {
        self.checked_reader(lease)?;
        let binding = self
            .training_views
            .as_ref()
            .and_then(|arena| arena.replay_capacity())
            .ok_or_else(|| {
                publication_input_error("parent has no cold replay append reservation")
            })?;
        Ok([
            binding.metadata_bytes,
            binding.max_material_bytes,
            binding.total_material_bytes,
            binding.max_evidence_bytes,
        ])
    }

    /// Acquire the complete non-actor contributor batch and acknowledge its intent in one CAS.
    /// Neither the original action coordinates nor the ordinary training cursor are rewritten.

    pub fn acknowledge_loaded_replay_delivery<V>(
        &mut self,
        lease: &SemanticPublishedLease,
        delivery_receipt: &[u8],
        receiver_outcome: &[u8],
        result_receipts: &[Vec<u8>],
        recipient_id: Identity256,
        verification_key: &[u8; 32],
        expected_effect: &[u8],
        proposals: Vec<LoadedCodingProposal>,
        acquisitions: Vec<LoadedReplayDeliveryAcquisition<V>>,
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError>
    where
        V: AsRef<
                [(
                    Identity256,
                    SemanticReplayLoadedMaterial,
                    SemanticReplayLoadedMaterial,
                )],
            > + Send
            + Sync
            + 'static,
    {
        self.checked_reader(lease)?;
        let observation_only = self.coding_delivery_is_observation()?;
        if observation_only
            && (!proposals.is_empty() || !acquisitions.is_empty() || !result_receipts.is_empty())
        {
            return Err(publication_input_error(
                "evaluation coding observations cannot carry training append material",
            ));
        }
        if acquisitions.len() != result_receipts.len() {
            return Err(publication_input_error(
                "replay delivery requires one complete bounded contributor batch",
            ));
        }
        if !observation_only {
            let arena = self
                .training_views
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            arena
                .replay_capacity()
                .ok_or(SemanticTransitionError::NotBound)?;
            if acquisitions.len() > arena.row_capacity() - arena.original_slot_boundary() {
                return Err(publication_input_error(
                    "replay delivery requires one complete bounded contributor batch",
                ));
            }
        }
        let budget = self.replay_material_budget.as_ref().ok_or_else(|| {
            publication_input_error("loaded replay delivery requires its original material pool")
        })?;
        let mut bytes = (size_of::<(
            Vec<LoadedCodingProposal>,
            Vec<LoadedReplayDeliveryAcquisition<V>>,
        )>() + 2 * size_of::<usize>()) as u64;
        bytes = bytes
            .checked_add(self.replay_delivery_allocation_bound(
                lease,
                None,
                0,
                &[],
                delivery_receipt,
                receiver_outcome,
                &[],
            )?)
            .and_then(|bytes| {
                bytes.checked_add(
                    (proposals.capacity() as u64)
                        .checked_mul(size_of::<LoadedCodingProposal>() as u64)?,
                )
            })
            .and_then(|bytes| {
                bytes.checked_add((proposals.len() as u64).checked_mul(
                    (size_of::<(
                        &SemanticReplayMaterial,
                        &SemanticResolvedReplayMaterial,
                        &SemanticReplayLoadedMaterial,
                    )>() + size_of::<usize>()
                        + size_of::<bool>()) as u64,
                )?)
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    (acquisitions.capacity() as u64)
                        .checked_mul(size_of::<LoadedReplayDeliveryAcquisition<V>>() as u64)?,
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        for (replay, original, action) in &proposals {
            if replay
                .budget
                .as_ref()
                .is_none_or(|original| !budget.same_pool(original))
                || !Arc::ptr_eq(&replay.proof_owner, &original.proof_owner)
                || action
                    .bytes
                    .budget()
                    .is_none_or(|original| !budget.same_pool(original))
            {
                return Err(publication_input_error(
                    "coding Proposal custody changed its original material pool",
                ));
            }
        }
        for ((replay, row, canonical, _, _, _, materials, original, action), result) in
            acquisitions.iter().zip(result_receipts)
        {
            if row.bytes.require_budget(budget).is_err()
                || replay
                    .budget
                    .as_ref()
                    .is_none_or(|original| !budget.same_pool(original))
                || !Arc::ptr_eq(&replay.proof_owner, &original.proof_owner)
                || action
                    .bytes
                    .budget()
                    .is_none_or(|original| !budget.same_pool(original))
                || materials.as_ref().iter().any(|(_, material, metadata)| {
                    [material, metadata].iter().any(|loaded| {
                        loaded
                            .bytes
                            .budget()
                            .is_none_or(|original| !budget.same_pool(original))
                    })
                })
            {
                return Err(publication_input_error(
                    "delivery batch changed its original material pool",
                ));
            }
            bytes = bytes
                .checked_add(self.replay_delivery_allocation_bound(
                    lease,
                    Some(replay),
                    row.bytes.as_ref().len(),
                    canonical,
                    delivery_receipt,
                    receiver_outcome,
                    result,
                )?)
                .and_then(|n| {
                    n.checked_add(
                        (size_of::<PublicationReplayDeliveryRow>()
                            + size_of::<super::verification_receipts::AuthenticatedTaskResults>()
                            + size_of::<Identity256>()) as u64,
                    )
                })
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let reservation = Arc::new(budget.reserve_auxiliary(bytes)?);
        let custody = Arc::new((proposals, acquisitions));
        self.acknowledge_replay_delivery_inner(
            lease,
            delivery_receipt,
            receiver_outcome,
            result_receipts,
            recipient_id,
            verification_key,
            expected_effect,
            &custody.0,
            &custody.1,
            Some(Arc::clone(&custody) as Arc<dyn Send + Sync>),
            Some(reservation),
        )
    }

    fn acknowledge_replay_delivery_inner<V>(
        &mut self,
        lease: &SemanticPublishedLease,
        delivery_receipt: &[u8],
        receiver_outcome: &[u8],
        result_receipts: &[Vec<u8>],
        recipient_id: Identity256,
        verification_key: &[u8; 32],
        expected_effect: &[u8],
        original_proposals: &[LoadedCodingProposal],
        acquisitions: &[LoadedReplayDeliveryAcquisition<V>],
        row_custody: Option<Arc<dyn Send + Sync>>,
        material_reservation: Option<Arc<SemanticReplayMaterialReservation>>,
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError>
    where
        V: AsRef<
            [(
                Identity256,
                SemanticReplayLoadedMaterial,
                SemanticReplayLoadedMaterial,
            )],
        >,
    {
        self.checked_reader(lease)?;
        if self.pending_replay_delivery.is_some()
            || self.pending
            || self.admitted_transition.is_some()
            || self.continuation_base.is_some()
            || !matches!(lease.header.terminal, 0 | 2)
        {
            return Err(publication_input_error(
                "replay acknowledgement requires an idle acquired coding parent",
            ));
        }
        if !self.task_observations_authenticated {
            return Err(publication_input_error(
                "restored observations require current trusted receiver authentication",
            ));
        }
        let observation_only = self.coding_delivery_is_observation()?;
        if observation_only
            && (!original_proposals.is_empty()
                || !acquisitions.is_empty()
                || !result_receipts.is_empty())
        {
            return Err(publication_input_error(
                "evaluation coding observations cannot carry training append material",
            ));
        }
        let arena = if observation_only {
            None
        } else {
            let arena = Arc::clone(
                self.training_views
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?,
            );
            arena
                .replay_capacity()
                .ok_or_else(|| publication_input_error("replay append was not reserved cold"))?;
            if !acquisitions.is_empty() && !arena.is_materialized() {
                return Err(publication_input_error("acquired results remain in durable custody until original training materialization"));
            }
            Some(arena)
        };
        let entries = self.read_published_control_record_with_custody(
            lease,
            30,
            material_reservation.clone(),
            row_custody.clone(),
        )?;
        let payload = self.read_published_control_record_with_custody(
            lease,
            31,
            material_reservation.clone(),
            row_custody.clone(),
        )?;
        let intents = output_intents_from_ranges(&entries, &payload)?;
        let stable = delivery_receipt
            .get(25..57)
            .and_then(|bytes| bytes.try_into().ok())
            .map(Identity256::from_bytes)
            .ok_or_else(|| {
                publication_input_error("delivery receipt has no stable native intent")
            })?;
        let intent = intents
            .iter()
            .find(|intent| intent.stable_identity == stable)
            .ok_or_else(|| {
                publication_input_error("external result has no original native intent")
            })?;
        if expected_effect.is_empty() || intent.effect != expected_effect {
            return Err(publication_input_error(
                "external result effect differs from the currently bound receiver",
            ));
        }
        let acknowledgements = self.read_published_control_record_with_custody(
            lease,
            32,
            material_reservation.clone(),
            row_custody.clone(),
        )?;
        let outcome_capacity = acknowledgement_archive::admit_payload(
            &acknowledgements.bytes,
            acknowledgements.capacity,
            stable,
            delivery_receipt.len(),
            receiver_outcome.len(),
        )?;
        let delivery_digest =
            verify_delivery_receipt(delivery_receipt, intent, recipient_id, verification_key)?;
        let outcome = super::receiver_outcome::verify_receiver_outcome(
            receiver_outcome,
            stable,
            recipient_id,
            &intent.effect,
            &intent.payload,
            delivery_receipt,
            verification_key,
            outcome_capacity,
        )?;
        let proposals = original_proposals
            .iter()
            .map(|(replay, original, action)| (replay, original, action))
            .collect::<Vec<_>>();
        let prefix = self.read_published_control_record_with_custody(
            lease,
            1,
            material_reservation.clone(),
            row_custody.clone(),
        )?;
        let provenance = self.read_published_control_record_with_custody(
            lease,
            2,
            material_reservation.clone(),
            row_custody.clone(),
        )?;
        let contract = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .contract_value;
        let contributing = if observation_only {
            Vec::new()
        } else {
            intent_lineage::contributor_indices(
                contract,
                lease.header,
                intent,
                &prefix,
                &provenance,
                &proposals,
            )?
        };
        let completed =
            outcome.disposition == super::receiver_outcome::ReceiverDisposition::Completed;
        if (completed && contributing.len() != acquisitions.len())
            || (!completed && (!acquisitions.is_empty() || !result_receipts.is_empty()))
        {
            return Err(publication_input_error(
                "replay acknowledgement must cover exactly the complete native contributor set",
            ));
        }
        let ground = &self
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .0
            .spec
            .task_ground;
        let common = if completed {
            Some(super::verification_receipts::verify_completed_tool_result(
                ground,
                outcome.content,
                stable,
                recipient_id,
                &intent.effect,
                &intent.payload,
                verification_key,
                self.replay_material_budget.as_ref(),
            )?)
        } else {
            None
        };
        let mut authenticated = Vec::with_capacity(acquisitions.len());
        let mut result_digests = Vec::with_capacity(acquisitions.len());
        let source_count = acquisitions
            .len()
            .checked_add(usize::from(completed))
            .ok_or(SemanticTransitionError::GenerationExhausted)?
            .checked_mul(ground.bindings().len())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut sources = Vec::with_capacity(source_count);
        if let Some(common) = &common {
            for (ordinal, (binding, &observation)) in ground
                .bindings()
                .iter()
                .zip(&common.observations)
                .enumerate()
            {
                sources.push((
                    binding.observation_record,
                    crate::SemanticRootSupport::Authenticated {
                        binding_ordinal: ordinal as u64,
                        observation,
                    },
                ));
            }
        }
        for (
            ((replay, row, canonical, action, record, evidence, materials, _, _), result),
            &contributor,
        ) in acquisitions.iter().zip(result_receipts).zip(&contributing)
        {
            let arena = arena.as_ref().ok_or(SemanticTransitionError::NotBound)?;
            let capacity = arena
                .replay_capacity()
                .ok_or(SemanticTransitionError::NotBound)?;
            let proposal = &original_proposals[contributor].0;
            if row.basis != SemanticTrainingViewBasis::Episode
                || replay.predecessor_identity().logical_digest
                    != proposal.predecessor_identity().logical_digest
                || replay.successor_identity().logical_digest
                    != proposal.successor_identity().logical_digest
                || row.origin != Some(replay.training_view_origin()?)
                || canonical.is_empty()
                || canonical.len() > capacity.payload_capacity_bytes
                || Identity256::from_bytes(Sha256::digest(replay.evidence.encode()?).into())
                    != *evidence
            {
                return Err(publication_input_error(
                    "replay acknowledgement requires each complete original episode and evidence",
                ));
            }
            let verification = super::verification_receipts::verify_acquired_receipts(
                ground,
                materials
                    .as_ref()
                    .iter()
                    .map(|(identity, material, metadata)| {
                        (*identity, material.as_ref(), Some(metadata.as_ref()))
                    }),
                stable,
                &intent.effect,
                &intent.payload,
                OriginalObservationOrigin::Action(*action),
                verification_key,
                self.replay_material_budget.as_ref(),
            )?;
            let common = common
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if verification
                .receipts
                .iter()
                .map(|receipt| (receipt.identity, receipt.outcome))
                .ne(common
                    .receipts
                    .iter()
                    .map(|receipt| (receipt.identity, receipt.outcome)))
                || verification.canonical_payload_identity != common.canonical_payload_identity
                || verification.file_roster_identity != common.file_roster_identity
            {
                return Err(publication_input_error(
                    "contributor row differs from the receiver's actual completed receipt roster",
                ));
            }
            result_digests.push(verify_result_receipt(
                result,
                stable,
                replay.predecessor_identity().logical_digest,
                replay.successor_identity().logical_digest,
                delivery_digest,
                *action,
                *record,
                *evidence,
                verification.disposition,
                verification_key,
            )?);
            for (ordinal, (binding, &observation)) in ground
                .bindings()
                .iter()
                .zip(&verification.observations)
                .enumerate()
            {
                sources.push((
                    binding.observation_record,
                    crate::SemanticRootSupport::Authenticated {
                        binding_ordinal: ordinal as u64,
                        observation,
                    },
                ));
            }
            authenticated.push(verification);
        }
        let publication = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        self.ensure_quiescent()?;
        let mut current_read = self.stage_publication_read_with_custody(
            publication.control.view(),
            material_reservation.clone(),
            row_custody.clone(),
        )?;
        self.submit_publication_read(&mut current_read)?;
        let current = self.resolve_publication_read(&mut current_read)?[0];
        if current.word != lease.identity.word || current.instance != lease.identity.instance {
            return Err(publication_input_error(
                "external result acknowledgement requires the current acquired parent",
            ));
        }
        let queue = self.read_published_control_record_with_custody(
            lease,
            56,
            material_reservation.clone(),
            row_custody.clone(),
        )?;
        let queue_payload = self.read_published_control_record_with_custody(
            lease,
            57,
            material_reservation.clone(),
            row_custody.clone(),
        )?;
        let archived = append_rows_from_ranges(&queue, &queue_payload)?;
        if let Some(previous) =
            acknowledgement_records(&acknowledgements.bytes, acknowledgements.capacity)?
                .iter()
                .find(|record| record.stable_identity == stable)
        {
            let matching = archived
                .iter()
                .filter(|item| words_identity(item.entry.stable_intent) == stable)
                .collect::<Vec<_>>();
            if previous.delivery_receipt.as_slice() != delivery_receipt
                || previous.receiver_outcome.as_slice() != receiver_outcome
                || matching.len() != acquisitions.len()
            {
                return Err(publication_input_error(
                    "native intent was acknowledged with another complete contributor batch",
                ));
            }
            for (((item, (_, row, canonical, _, _, _, _, _, _)), result), (verification, digest)) in
                matching
                    .iter()
                    .zip(acquisitions)
                    .zip(result_receipts)
                    .zip(authenticated.iter().zip(&result_digests))
            {
                let arena = arena.as_ref().ok_or(SemanticTransitionError::NotBound)?;
                let descriptor = if verification.disposition == 1 {
                    arena.prepare_append_row(
                        usize::try_from(item.entry.row_ordinal)
                            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                        row,
                    )?
                } else {
                    arena.prepare_rejected_row(row)?
                };
                if item.entry.disposition != verification.disposition
                    || item.result_receipt != result.as_slice()
                    || words_identity(item.entry.result_receipt_digest) != *digest
                    || item.canonical != canonical.as_slice()
                    || item.raw != row.bytes.as_ref()
                    || publication_abi_bytes(&[item.descriptor])
                        != publication_abi_bytes(&[descriptor])
                {
                    return Err(publication_input_error(
                        "native intent was acknowledged with a different original replay row",
                    ));
                }
            }
            return Ok(lease.identity);
        }
        let queue_header = append_header(&queue.bytes)?;
        let append_capacity = match &arena {
            Some(arena) => (arena.row_capacity() - arena.original_slot_boundary()) as u64,
            None => 0,
        };
        if lease.header.fuel == 0
            || queue_header
                .count
                .checked_add(acquisitions.len() as u64)
                .is_none_or(|count| count > queue_header.capacity)
            || queue_header.capacity != append_capacity
        {
            return Err(publication_input_error(
                "complete replay batch exceeds the original append or acknowledgement reservation",
            ));
        }
        let mut eligible_count = usize::try_from(queue_header.eligible_count)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let packed_capacity = acquisitions.iter().zip(result_receipts).try_fold(
            0usize,
            |bytes, ((_, row, canonical, _, _, _, _, _, _), result)| {
                bytes
                    .checked_add(40 + size_of::<TrainingViewRowDescriptor>())
                    .and_then(|bytes| bytes.checked_add(row.bytes.as_ref().len()))
                    .and_then(|bytes| bytes.checked_add(canonical.len()))
                    .and_then(|bytes| bytes.checked_add(result.len()))
                    .ok_or(SemanticTransitionError::GenerationExhausted)
            },
        )?;
        let mut packed = Vec::with_capacity(packed_capacity);
        let mut rows = Vec::with_capacity(acquisitions.len());
        let mut training_rows = Vec::with_capacity(acquisitions.len());
        let mut training_chain = queue_header.chain_head;
        for ((((_, row, canonical, _, _, _, _, _, _), result), verification), digest) in
            acquisitions
                .iter()
                .zip(result_receipts)
                .zip(&authenticated)
                .zip(&result_digests)
        {
            let arena = arena.as_ref().ok_or(SemanticTransitionError::NotBound)?;
            let descriptor = if verification.disposition == 1 {
                let ordinal = arena
                    .original_count()
                    .checked_add(eligible_count)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                eligible_count = eligible_count
                    .checked_add(1)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                arena.prepare_append_row(ordinal, row)?
            } else {
                arena.prepare_rejected_row(row)?
            };
            let payload_offset = packed.len() as u64;
            let bytes = append_payload(&descriptor, row.bytes.as_ref(), canonical, result)?;
            if queue_header
                .payload_used_bytes
                .checked_add(payload_offset)
                .and_then(|used| used.checked_add(bytes.len() as u64))
                .is_none_or(|used| used > queue_header.payload_capacity_bytes)
            {
                return Err(publication_input_error(
                    "complete replay batch exceeds its original payload reservation",
                ));
            }
            let entry = SemanticTrainingReplayAppendEntry {
                row_ordinal: descriptor.ordinal,
                content_identity: descriptor.content_identity,
                original_origin: descriptor.origin,
                stable_intent: identity_words(*stable.as_bytes()),
                result_receipt_digest: identity_words(*digest.as_bytes()),
                payload_offset_bytes: queue_header.payload_used_bytes + payload_offset,
                payload_length_bytes: bytes.len() as u64,
                disposition: verification.disposition,
            };
            let archive_ordinal = queue_header.count + rows.len() as u64;
            training_chain = publication_fold_digest(
                training_chain,
                56,
                archive_ordinal,
                Identity256::from_bytes(Sha256::digest(publication_abi_bytes(&[entry])).into()),
            );
            training_chain = publication_fold_digest(
                training_chain,
                57,
                archive_ordinal,
                Identity256::from_bytes(Sha256::digest(&bytes).into()),
            );
            if verification.disposition == 1 {
                training_rows.push(self.retain_training_append(
                    descriptor,
                    row,
                    canonical,
                    row_custody.clone(),
                )?);
            }
            rows.push(PublicationReplayDeliveryRow {
                entry,
                payload: payload_offset,
                payload_bytes: bytes.len() as u64,
                row_descriptor: descriptor,
                row_bytes: payload_offset + 40 + size_of::<TrainingViewRowDescriptor>() as u64,
                row_bytes_len: row.bytes.as_ref().len() as u64,
            });
            packed.extend(bytes);
        }
        let prepared = self
            .graph
            .prepare_authenticated_supports(&sources)
            .map_err(SemanticTransitionError::Semantic)?;
        let receipt = ReplayUpload::new_with_owner(
            &self.provider,
            &self.stream,
            delivery_receipt,
            material_reservation.as_ref(),
        )?;
        let receiver_upload = ReplayUpload::new_with_owner(
            &self.provider,
            &self.stream,
            receiver_outcome,
            material_reservation.as_ref(),
        )?;
        let payload = ReplayUpload::new_with_owner(
            &self.provider,
            &self.stream,
            &packed,
            material_reservation.as_ref(),
        )?;
        for row in &mut rows {
            row.payload += payload.device.device_ptr_value();
            row.row_bytes += payload.device.device_ptr_value();
        }
        let rows = ReplayUpload::new_with_owner(
            &self.provider,
            &self.stream,
            &rows,
            material_reservation.as_ref(),
        )?;
        let mut observation_words = Vec::with_capacity(
            source_count
                .checked_mul(18)
                .ok_or(SemanticTransitionError::GenerationExhausted)?,
        );
        if let Some(common) = &common {
            for observation in &common.observations {
                observation_words.extend_from_slice(observation);
            }
        }
        for result in &authenticated {
            for observation in &result.observations {
                observation_words.extend_from_slice(observation);
            }
        }
        let observations = ReplayUpload::new_with_owner(
            &self.provider,
            &self.stream,
            &observation_words,
            material_reservation.as_ref(),
        )?;
        let statements = ReplayUpload::new_with_owner(
            &self.provider,
            &self.stream,
            &prepared.statements,
            material_reservation.as_ref(),
        )?;
        let supports = ReplayUpload::new_with_owner(
            &self.provider,
            &self.stream,
            &prepared.supports,
            material_reservation.as_ref(),
        )?;
        #[cfg(feature = "semantic-policy")]
        let native_work = self.cold_native_work(lease.token)?;
        #[cfg(not(feature = "semantic-policy"))]
        let native_work: Option<DeviceMemoryView<u64>> = None;
        let (arena_descriptors, arena_descriptor_count, arena_raw, arena_raw_bytes) = match &arena {
            Some(arena) => {
                let (descriptors, raw) = arena.append_storage();
                (
                    *descriptors.device_ptr(),
                    descriptors.len() as u64,
                    *raw.device_ptr(),
                    raw.len() as u64,
                )
            }
            None => (0, 0, 0, 0),
        };
        let input = PublicationReplayDeliveryInput {
            delivery: PublicationDeliveryInput {
                lease: self.readers[&lease.token].device.device_ptr_value(),
                receipt: receipt.device.device_ptr_value(),
                receipt_len: delivery_receipt.len() as u64,
                recipient_id,
                receipt_digest: delivery_digest,
            },
            receiver_outcome: receiver_upload.device.device_ptr_value(),
            receiver_outcome_bytes: receiver_outcome.len() as u64,
            receiver_outcome_digest: outcome.identity,
            receiver_disposition: if completed { 1 } else { 2 },
            rows: rows.device.device_ptr_value(),
            row_count: acquisitions.len() as u64,
            arena_descriptors,
            arena_descriptor_count,
            arena_raw,
            arena_raw_bytes,
            observations: observations.device.device_ptr_value(),
            observation_count: source_count as u64,
            statements: statements.device.device_ptr_value(),
            supports: supports.device.device_ptr_value(),
            native_work: native_work.as_ref().map_or(0, |work| *work.device_ptr()),
        };
        let input = ReplayUpload::new_with_owner(
            &self.provider,
            &self.stream,
            &[input],
            material_reservation.as_ref(),
        )?;
        self.ensure_quiescent()?;
        let control = Arc::new(std::sync::Mutex::new(
            self.stage_publication_read_with_custody(
                publication.control.view(),
                material_reservation.clone(),
                row_custody.clone(),
            )?,
        ));
        let successor_bank = ((lease.identity.word & 1) ^ 1) as usize;
        // SAFETY: the inactive publication bank's fixed header exists for this same retained submission.
        let header_view = unsafe {
            publication.banks[successor_bank]
                .view()
                .cast::<u8>()
                .expect("publication bank bytes")
                .slice(..size_of::<PublicationHeader>())
                .cast::<PublicationHeader>()
                .expect("publication header alignment")
        };
        let header = Arc::new(std::sync::Mutex::new(
            self.stage_publication_read_with_custody(
                header_view,
                material_reservation.clone(),
                row_custody.clone(),
            )?,
        ));
        #[cfg(feature = "semantic-policy")]
        let context_acknowledgement = self.stage_model_context_acknowledgement(
            lease,
            delivery_receipt,
            delivery_digest,
            recipient_id,
            Some((&packed, receiver_outcome)),
        )?;
        let mut receipt_allocation_owners =
            Vec::with_capacity(authenticated.len() + usize::from(completed));
        if let Some(common) = common {
            if let Some(owner) = common.allocation_owner {
                receipt_allocation_owners.push(owner);
            }
        }
        for result in authenticated.iter_mut() {
            if let Some(owner) = result.allocation_owner.take() {
                receipt_allocation_owners.push(owner);
            }
        }
        let eligible = authenticated.iter().any(|result| result.disposition == 1);
        let training_publication = arena
            .map(|arena| {
                let publication = arena.prepare_training_appends(
                    queue_header.count,
                    queue_header.count + acquisitions.len() as u64,
                    training_chain,
                    training_rows,
                )?;
                Ok::<_, SemanticTransitionError>((arena, publication))
            })
            .transpose()?;
        self.pending_replay_delivery = Some(PendingReplayDelivery {
            base: lease.identity,
            fuel: lease.header.fuel,
            eligible,
            lease_token: lease.token,
            completion: None,
            training_publication,
            #[cfg(feature = "semantic-policy")]
            context_acknowledgement,
            receipt,
            receiver_outcome: receiver_upload,
            payload,
            rows,
            observations,
            statements,
            supports,
            input,
            support_references: prepared.references,
            native_work,
            control,
            header,
            kernel_entered: false,
            kernel_submitted: false,
            _row_custody: row_custody,
            _material_reservation: material_reservation,
            _support_allocation_owner: prepared.allocation_owner,
            _receipt_allocation_owners: receipt_allocation_owners,
        });
        let result = self.resolve_replay_delivery(lease);
        if result.is_err()
            && self
                .pending_replay_delivery
                .as_ref()
                .is_some_and(|pending| {
                    !pending.kernel_entered
                        && pending.receipt.completed
                        && pending.receiver_outcome.completed
                        && pending.payload.completed
                        && pending.rows.completed
                        && pending.observations.completed
                        && pending.statements.completed
                        && pending.supports.completed
                        && pending.input.completed
                })
        {
            #[cfg(feature = "semantic-policy")]
            if let Some(edge) = self
                .pending_replay_delivery
                .as_ref()
                .and_then(|pending| pending.context_acknowledgement.as_ref())
            {
                self.abandon_model_context_acknowledgement(edge)?;
            }
            self.pending_replay_delivery = None;
        }
        result
    }

    fn retain_training_append<R: AsRef<[u8]>>(
        &self,
        descriptor: TrainingViewRowDescriptor,
        row: &SemanticTrainingViewRow<R>,
        canonical: &[u8],
        custody: Option<Arc<dyn Send + Sync>>,
    ) -> Result<
        Arc<crate::semantic_training_view::AuthenticatedTrainingAppendRow>,
        SemanticTransitionError,
    > {
        use crate::semantic_training_view::AuthenticatedTrainingAppendRow;
        let bytes = canonical
            .len()
            .checked_add(size_of::<AuthenticatedTrainingAppendRow>())
            .and_then(|bytes| {
                bytes.checked_add(4 * size_of::<Arc<AuthenticatedTrainingAppendRow>>())
            })
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = Arc::new(
            self.replay_material_budget
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .reserve_auxiliary(bytes)?,
        );
        Ok(Arc::new(AuthenticatedTrainingAppendRow {
            descriptor,
            bytes_identity: row.bytes_identity,
            canonical: canonical.to_vec().into(),
            _custody: custody,
            _reservation: reservation,
        }))
    }

    /// Return the same immutable roster registered before this Update's geometry.
    /// A later acknowledgement cannot change this projection or its derivatives.
    pub fn prepare_training_roster(
        &self,
        step: &SemanticPreparedStep,
    ) -> Result<(Vec<Arc<[u8]>>, Vec<(u64, Vec<u64>)>), SemanticTransitionError> {
        if self.prepared_transition_kind(step)? != SemanticTransitionKind::Update {
            return Err(publication_input_error(
                "training roster requires its original Update",
            ));
        }
        self.checked_prepared_step(step, false)?;
        self.steps
            .get(&step.token)
            .and_then(|record| record.prepared.as_ref())
            .and_then(|prepared| prepared.training_view.as_ref())
            .ok_or_else(|| {
                publication_input_error("Update has no original selected training view")
            })?
            .training_roster_projection()
    }

    pub fn replay_delivery_pending(&self) -> bool {
        self.pending_replay_delivery.is_some()
    }

    fn complete_replay_delivery_prefix(&mut self) -> Result<(), SemanticTransitionError> {
        let pending = self
            .pending_replay_delivery
            .as_mut()
            .ok_or(SemanticTransitionError::NoPendingLaunch)?;
        let uploads = (|| {
            pending.receipt.complete(&self.provider, &self.stream)?;
            pending
                .receiver_outcome
                .complete(&self.provider, &self.stream)?;
            pending.payload.complete(&self.provider, &self.stream)?;
            pending.rows.complete(&self.provider, &self.stream)?;
            pending
                .observations
                .complete(&self.provider, &self.stream)?;
            pending.statements.complete(&self.provider, &self.stream)?;
            pending.supports.complete(&self.provider, &self.stream)?;
            pending.input.complete(&self.provider, &self.stream)
        })();
        if let Err(error) = uploads {
            self.poisoned = true;
            return Err(error);
        }
        if pending.kernel_entered {
            if !pending.kernel_submitted {
                return Err(publication_input_error(
                    "original replay delivery has no known successful driver submission",
                ));
            }
            return Ok(());
        }
        let mut descriptor = self.descriptor();
        let pending = self
            .pending_replay_delivery
            .as_ref()
            .expect("original replay delivery");
        descriptor.publication = PublicationCommand {
            control: self
                .publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .control
                .device_ptr_value(),
            lease: pending.input.device.device_ptr_value(),
            operation: 8,
            native_work: 0,
        };
        let mut recorder = self.kernel_recorder()?;
        recorder.read(&pending.receipt.device);
        recorder.read(&pending.receiver_outcome.device);
        recorder.read(&pending.payload.device);
        recorder.read(&pending.rows.device);
        recorder.read(&pending.observations.device);
        recorder.read(&pending.statements.device);
        recorder.read(&pending.supports.device);
        recorder.read(&pending.input.device);
        recorder.read(&self.readers[&pending.lease_token].device);
        self.task_ground
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .record(&mut recorder);
        if let Some(work) = &pending.native_work {
            recorder.read_write(work);
        }
        if pending.eligible {
            let (arena, _) = pending
                .training_publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let (descriptors, raw) = arena.append_storage();
            recorder.write(&descriptors);
            recorder.write(&raw);
        }
        let execute = self.execute.clone();
        let pending = self
            .pending_replay_delivery
            .as_mut()
            .expect("original replay delivery");
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |stream| {
            // SAFETY: strict declarations and the original pending owner retain
            // every indirect input and destination before the driver boundary.
            let mut params = [(&mut descriptor as *mut Descriptor).cast()];
            unsafe {
                execute.launch_raw_in_original(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut params,
                    false,
                    &mut pending.kernel_entered,
                    &mut pending.kernel_submitted,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })
    }

    pub fn validate_replay_delivery_parent(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        let pending = self
            .pending_replay_delivery
            .as_ref()
            .ok_or(SemanticTransitionError::NoPendingLaunch)?;
        if !lease.active
            || !Arc::ptr_eq(&lease.issuer, &self.publication_issuer)
            || !self.readers.contains_key(&lease.token)
            || pending.base != lease.identity
            || pending.lease_token != lease.token
        {
            return Err(publication_input_error(
                "replay result resolver requires its original acquired parent",
            ));
        }
        Ok(())
    }

    /// Release a known original completion only after its consumer has built
    /// the complete returned identity. This never submits or replays work.
    pub fn acknowledge_replay_delivery_completion(
        &mut self,
        lease: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.validate_replay_delivery_parent(lease)?;
        if self
            .pending_replay_delivery
            .as_ref()
            .and_then(|pending| pending.completion)
            .is_none()
        {
            return Err(publication_input_error(
                "replay delivery completion is still unresolved",
            ));
        }
        self.pending_replay_delivery = None;
        Ok(())
    }

    /// Join/read the retained original submission. This never replays its CAS or reacquires authority.
    pub fn resolve_replay_delivery(
        &mut self,
        lease: &SemanticPublishedLease,
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        self.validate_replay_delivery_parent(lease)?;
        let pending = self
            .pending_replay_delivery
            .as_ref()
            .ok_or(SemanticTransitionError::NoPendingLaunch)?;
        if let Some(identity) = pending.completion {
            return Ok(identity);
        }
        self.complete_replay_delivery_prefix()?;
        let pending = self
            .pending_replay_delivery
            .as_ref()
            .ok_or(SemanticTransitionError::NoPendingLaunch)?;
        let base = pending.base;
        let fuel = pending.fuel;
        let control_read = Arc::clone(&pending.control);
        let header_read = Arc::clone(&pending.header);
        wait_on_stream(
            &self.stream,
            &mut self.poisoned,
            &mut self.stream_waits,
            "replay acknowledgement completion",
            CudaStream::synchronize,
        )?;
        let control = {
            let mut read = control_read
                .lock()
                .map_err(|_| publication_input_error("replay completion control owner poisoned"))?;
            self.resolve_publication_read(&mut read)?[0]
        };
        let expected_word = ((base.word >> 1) + 1) << 1 | ((base.word & 1) ^ 1);
        if control.word == base.word && control.refusal != 0 {
            let status = control.refusal;
            #[cfg(feature = "semantic-policy")]
            if let Some(edge) = self
                .pending_replay_delivery
                .as_ref()
                .and_then(|pending| pending.context_acknowledgement.as_ref())
            {
                self.abandon_model_context_acknowledgement(edge)?;
            }
            self.pending_replay_delivery = None;
            return Err(SemanticTransitionError::PublicationRefused { status });
        }
        let header = {
            let mut read = header_read
                .lock()
                .map_err(|_| publication_input_error("replay completion header owner poisoned"))?;
            self.resolve_publication_read(&mut read)?[0]
        };
        if control.word != expected_word
            || header.publication_word != expected_word
            || header.instance != base.instance
            || header.terminal != lease.header.terminal
            || header.fuel.checked_add(1) != Some(fuel)
            || header.training_cursor != lease.header.training_cursor
            || header.training_rng != lease.header.training_rng
        {
            self.poisoned = true;
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let identity = SemanticPublishedIdentity {
            instance: header.instance,
            word: header.publication_word,
            logical_digest: header.logical_digest,
        };
        #[cfg(feature = "semantic-policy")]
        if let Some(edge) = self
            .pending_replay_delivery
            .as_ref()
            .and_then(|pending| pending.context_acknowledgement.as_ref())
        {
            self.complete_model_context_acknowledgement(lease, edge, header)?;
        }
        let references = &self
            .pending_replay_delivery
            .as_ref()
            .ok_or(SemanticTransitionError::NoPendingLaunch)?
            .support_references;
        self.graph.finish_authenticated_supports(references);
        let pending = self
            .pending_replay_delivery
            .as_ref()
            .ok_or(SemanticTransitionError::NoPendingLaunch)?;
        if pending
            .training_publication
            .as_ref()
            .is_some_and(|(arena, publication)| {
                arena.publish_training_appends(publication).is_err()
            })
        {
            self.poisoned = true;
            return Err(SemanticTransitionError::Poisoned);
        }
        self.pending_replay_delivery
            .as_mut()
            .ok_or(SemanticTransitionError::NoPendingLaunch)?
            .completion = Some(identity);
        Ok(identity)
    }
}

fn words_identity(words: [u64; 4]) -> Identity256 {
    let mut bytes = [0u8; 32];
    for (index, word) in words.into_iter().enumerate() {
        bytes[index * 8..index * 8 + 8].copy_from_slice(&word.to_ne_bytes());
    }
    Identity256::from_bytes(bytes)
}
