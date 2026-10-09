//! Cold acquisition of an external result, committed with its native acknowledgement.

use super::*;
use crate::semantic_hypergraph::{
    SemanticResidentDecodedStatement, SemanticResidentDecodedSupport,
};
use crate::semantic_training_view::{
    SemanticTrainingReplayAppendEntry, SemanticTrainingReplayAppendHeader,
    SemanticTrainingViewBasis, TrainingViewRowDescriptor,
};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PublicationReplayDeliveryInput {
    delivery: PublicationDeliveryInput,
    entry: SemanticTrainingReplayAppendEntry,
    payload: u64,
    payload_bytes: u64,
    row_descriptor: TrainingViewRowDescriptor,
    row_bytes: u64,
    row_bytes_len: u64,
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
const _: () = assert!(size_of::<PublicationReplayDeliveryInput>() == 1288);

struct ReplayUpload<T: DeviceRepr> {
    device: TrackedCudaSlice<T>,
    write: crate::device::RetainedDeviceWrite<T>,
    admitted: bool,
    completed: bool,
}

impl<T: DeviceRepr + 'static> ReplayUpload<T> {
    fn new(
        provider: &CudaKernelProvider,
        stream: &Arc<CudaStream>,
        source: &[T],
    ) -> Result<Self, SemanticTransitionError> {
        let device = allocate_publication(provider, source.len())?;
        let write = crate::device::RetainedDeviceWrite::new(stream, source, device.view())
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
    verified: bool,
    completed: bool,
}

pub(super) struct PendingReplayDelivery {
    base: SemanticPublishedIdentity,
    fuel: u64,
    disposition: u64,
    lease_token: u64,
    completion: Option<SemanticPublishedIdentity>,
    receipt: ReplayUpload<u8>,
    payload: ReplayUpload<u8>,
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
    _arena: Arc<SemanticTrainingViewArena>,
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
    intent: &IntentEntry,
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
        intent.stable_identity,
        delivery_digest,
        intent.base_logical,
        intent.result_logical,
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

fn append_rows_from_ranges<'a>(
    entries: &'a PublicationMaterialRange,
    payload: &'a PublicationMaterialRange,
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
    let mut stable_intents = BTreeSet::new();
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
            || !stable_intents.insert(entry.stable_intent)
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
    /// Full original transport bytes, not their metadata/hash-only projection.
    pub fn replay_append_rows(
        &mut self,
        recipient: Identity256,
        key: &[u8; 32],
        effect: &[u8],
    ) -> Result<(Vec<Vec<u8>>, [u64; 4]), SemanticTransitionError> {
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
            let mut ranges = Vec::new();
            for role in [30, 31, 32, 56, 57, 58] {
                let range = *lease
                    .directory
                    .iter()
                    .find(|range| range.role == role && range.index == 0)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if role == 57 && storage.contract.replay_payload_capacity_bytes == 0 {
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
                    read: Some(Mutex::new(self.stage_publication_read(
                        allocation.slice()?.view().slice(begin..end),
                    )?)),
                });
            }
            let control = Mutex::new(self.stage_publication_read(storage.control.view())?);
            self.replay_authentication
                .as_mut()
                .expect("original receiver authentication")
                .reads = Some(Arc::new(ReplayAuthenticationReads { ranges, control }));
        }
        drop(lease);
        let entries = self.read_replay_authentication_range(56)?;
        let payload = self.read_replay_authentication_range(57)?;
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
        Ok((rows, limits))
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
        let bytes =
            if let Some(read) = &original.read {
                self.resolve_publication_read(&mut read.lock().map_err(|_| {
                    publication_input_error("receiver archive read owner poisoned")
                })?)?
            } else {
                Vec::new()
            };
        let material = PublicationMaterialRange {
            range: original.range,
            capacity: original.capacity,
            bytes,
        };
        if material.original_record_digest() != material.range.digest {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
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
        )],
    ) -> Result<(), SemanticTransitionError> {
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
        self.retire_published_reader(&mut lease)?;
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

    fn verify_replay_authentication(
        &mut self,
        lease: &SemanticPublishedLease,
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
        )],
    ) -> Result<(), SemanticTransitionError> {
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
        let rows = append_rows_from_ranges(&entries, &payload)?;
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
        let intents = output_intents_from_ranges(&intent_entries, &intent_payload)?;
        let acknowledgements = self.read_replay_authentication_range(32)?;
        let receipts =
            acknowledgement_receipts(&acknowledgements.bytes, acknowledgements.capacity)?;
        let mut eligible_count = 0;
        let mut observations = vec![[0u64; 18]; ground.bindings().len()];
        let mut sources = Vec::new();
        for (
            archived,
            (replay, row, canonical, action_identity, record_digest, evidence_digest, materials),
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
                materials,
                stable,
                &intent.effect,
                &intent.payload,
                *action_identity,
                key,
            )?;
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
                let accumulated = observations[ordinal][1] | incoming[1];
                observations[ordinal] = incoming;
                observations[ordinal][1] = accumulated;
            }
            if effect.is_empty()
                || intent.effect != effect
                || canonical.as_slice() != archived.canonical
                || row.bytes.as_slice() != archived.raw
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
            let receipt = &receipts
                .iter()
                .find(|(identity, _)| *identity == stable)
                .ok_or_else(|| {
                    publication_input_error(
                        "restored append has no original delivery acknowledgement",
                    )
                })?
                .1;
            let delivery_digest = verify_delivery_receipt(receipt, intent, recipient, key)?;
            let offset = size_of::<IntentQueueHeader>() + index * size_of::<IntentEntry>();
            // SAFETY: the canonical intent decoder checked this complete scalar entry.
            let original = unsafe {
                intent_entries.bytes[offset..]
                    .as_ptr()
                    .cast::<IntentEntry>()
                    .read_unaligned()
            };
            if replay.predecessor_identity().logical_digest != original.base_logical
                || replay.successor_identity().logical_digest != original.result_logical
                || verify_result_receipt(
                    archived.result_receipt,
                    &original,
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
        let current =
            self.resolve_publication_read(&mut reads.control.lock().map_err(|_| {
                publication_input_error("receiver publication read owner poisoned")
            })?)?[0];
        if current.word != lease.identity.word || current.instance != lease.identity.instance {
            return Err(publication_input_error(
                "append authentication changed its original acquired publication",
            ));
        }
        Ok(())
    }

    /// Reconstitute only original committed append slots from the complete archived queue.
    /// No new result, authority, selection, or publication is produced by restoration.
    pub(super) fn stage_restored_replay_append_rows(
        &self,
        restoration: &Arc<state_restoration::OriginalStateMaterialRestore>,
        material: &state_restoration::RestoredReplayAppendMaterial,
    ) -> Result<Vec<state_restoration::OriginalDeviceWrite>, SemanticTransitionError> {
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
        let rows = append_rows_from_ranges(entries, payload)?;
        if !rows.is_empty() {
            arena.require_append_authentication();
        }
        let (descriptors, raw) = arena.append_storage();
        let mut eligible_count = 0;
        let mut writes = Vec::new();
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
            let target = raw.slice(
                descriptor.raw_offset as usize..descriptor.raw_offset as usize + raw_len,
            );
            let physical = arena.physical_ordinal(expected_ordinal)?;
            let target_descriptor = descriptors.slice(physical..physical + 1);
            writes.push(state_restoration::OriginalDeviceWrite::new(
                &self.stream, row.raw, target, false,
            )?);
            writes.push(state_restoration::OriginalDeviceWrite::new(
                &self.stream, &[descriptor], target_descriptor, false,
            )?);
        }
        Ok(writes)
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

    /// Acquire one complete non-actor episode and acknowledge its nonterminal intent in one CAS.
    /// Neither the original action coordinates nor the ordinary training cursor are rewritten.
    pub fn acknowledge_replay_delivery(
        &mut self,
        lease: &SemanticPublishedLease,
        delivery_receipt: &[u8],
        result_receipt: &[u8],
        recipient_id: Identity256,
        verification_key: &[u8; 32],
        expected_effect: &[u8],
        replay: &SemanticReplayMaterial,
        row: SemanticTrainingViewRow,
        canonical_row: &[u8],
        action_identity: Identity256,
        record_digest: Identity256,
        evidence_digest: Identity256,
        verification_materials: &[(Identity256, Vec<u8>, Vec<u8>)],
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        self.checked_reader(lease)?;
        if self.pending_replay_delivery.is_some()
            || self.pending
            || self.admitted_transition.is_some()
            || self.continuation_base.is_some()
            || lease.header.terminal != 0
        {
            return Err(publication_input_error(
                "replay acknowledgement requires an idle acquired nonterminal parent",
            ));
        }
        if !self.task_observations_authenticated {
            return Err(publication_input_error(
                "restored observations require current trusted receiver authentication",
            ));
        }
        let arena = Arc::clone(
            self.training_views
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let capacity = arena
            .replay_capacity()
            .ok_or_else(|| publication_input_error("replay append was not reserved cold"))?;
        if !arena.is_materialized() {
            return Err(publication_input_error("acquired results remain in durable custody until original training materialization"));
        }
        if row.basis != SemanticTrainingViewBasis::Episode
            || row.origin != Some(replay.training_view_origin()?)
            || canonical_row.is_empty()
            || canonical_row.len() > capacity.payload_capacity_bytes
            || Identity256::from_bytes(Sha256::digest(replay.evidence.encode()?).into())
                != evidence_digest
        {
            return Err(publication_input_error(
                "replay acknowledgement requires the complete original episode and evidence",
            ));
        }
        let entries = self.read_published_control_record(lease, 30)?;
        let payload = self.read_published_control_record(lease, 31)?;
        let intents = output_intents_from_ranges(&entries, &payload)?;
        let stable = delivery_receipt
            .get(25..57)
            .and_then(|bytes| bytes.try_into().ok())
            .map(Identity256::from_bytes)
            .ok_or_else(|| {
                publication_input_error("delivery receipt has no stable native intent")
            })?;
        let ordinal = intents
            .iter()
            .position(|intent| intent.stable_identity == stable)
            .ok_or_else(|| {
                publication_input_error("external result has no original native intent")
            })?;
        let intent = &intents[ordinal];
        if expected_effect.is_empty() || intent.effect != expected_effect {
            return Err(publication_input_error(
                "external result effect differs from the currently bound receiver",
            ));
        }
        let delivery_digest =
            verify_delivery_receipt(delivery_receipt, intent, recipient_id, verification_key)?;
        let offset = size_of::<IntentQueueHeader>() + ordinal * size_of::<IntentEntry>();
        // SAFETY: output_intents_from_ranges verified the complete fixed entry roster above.
        let original = unsafe {
            entries.bytes[offset..]
                .as_ptr()
                .cast::<IntentEntry>()
                .read_unaligned()
        };
        if replay.predecessor_identity().logical_digest != original.base_logical
            || replay.successor_identity().logical_digest != original.result_logical
        {
            return Err(publication_input_error(
                "replay original action pair differs from the native intent",
            ));
        }
        let ground = &self
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .0
            .spec
            .task_ground;
        let authenticated = super::verification_receipts::verify_acquired_receipts(
            ground,
            verification_materials,
            stable,
            &intent.effect,
            &intent.payload,
            action_identity,
            verification_key,
        )?;
        let disposition = authenticated.disposition;
        let sources = ground
            .bindings()
            .iter()
            .zip(&authenticated.observations)
            .enumerate()
            .map(|(ordinal, (binding, &observation))| {
                (
                    binding.observation_record,
                    crate::SemanticRootSupport::Authenticated {
                        binding_ordinal: ordinal as u64,
                        observation,
                    },
                )
            })
            .collect::<Vec<_>>();
        let result_digest = verify_result_receipt(
            result_receipt,
            &original,
            delivery_digest,
            action_identity,
            record_digest,
            evidence_digest,
            disposition,
            verification_key,
        )?;
        let publication = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let current = self.publication_read(publication.control.view())?[0];
        if current.word != lease.identity.word || current.instance != lease.identity.instance {
            return Err(publication_input_error(
                "external result acknowledgement requires the current acquired parent",
            ));
        }
        let acknowledgements = self.read_published_control_record(lease, 32)?;
        if let Some((_, previous)) =
            acknowledgement_receipts(&acknowledgements.bytes, acknowledgements.capacity)?
                .iter()
                .find(|(identity, _)| *identity == stable)
        {
            let queue = self.read_published_control_record(lease, 56)?;
            let payload = self.read_published_control_record(lease, 57)?;
            let rows = append_rows_from_ranges(&queue, &payload)?;
            let mut matching = rows
                .iter()
                .filter(|item| words_identity(item.entry.stable_intent) == stable);
            let archived = matching.next().ok_or_else(|| {
                publication_input_error("acknowledged result has no original committed append")
            })?;
            let descriptor = if disposition == 1 {
                let row_ordinal = usize::try_from(archived.entry.row_ordinal)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
                arena.prepare_append_row(row_ordinal, &row)?
            } else {
                arena.prepare_rejected_row(&row)?
            };
            if previous.as_slice() != delivery_receipt
                || archived.entry.disposition != disposition
                || matching.next().is_some()
                || archived.result_receipt != result_receipt
                || words_identity(archived.entry.result_receipt_digest) != result_digest
                || archived.canonical != canonical_row
                || archived.raw != row.bytes.as_slice()
                || publication_abi_bytes(&[archived.descriptor])
                    != publication_abi_bytes(&[descriptor])
            {
                return Err(publication_input_error(
                    "native intent was acknowledged with a different complete replay result",
                ));
            }
            return Ok(lease.identity);
        }
        if lease.header.fuel == 0 {
            return Err(publication_input_error(
                "nonterminal parent has no reserved replay acknowledgement fuel",
            ));
        }
        let queue = self.read_published_control_record(lease, 56)?;
        let queue_header = append_header(&queue.bytes)?;
        if queue_header.count >= queue_header.capacity
            || queue_header.capacity
                != (arena.row_capacity() - arena.original_slot_boundary()) as u64
        {
            return Err(publication_input_error(
                "replay append reservation is exhausted or differs from the original arena",
            ));
        }
        let row_ordinal = arena
            .original_count()
            .checked_add(queue_header.eligible_count as usize)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let row_descriptor = if disposition == 1 {
            arena.prepare_append_row(row_ordinal, &row)?
        } else {
            arena.prepare_rejected_row(&row)?
        };
        let packed = append_payload(&row_descriptor, &row.bytes, canonical_row, result_receipt)?;
        if queue_header
            .payload_used_bytes
            .checked_add(packed.len() as u64)
            .is_none_or(|size| size > queue_header.payload_capacity_bytes)
        {
            return Err(publication_input_error(
                "complete replay append exceeds its original payload reservation",
            ));
        }
        let prepared = self
            .graph
            .prepare_authenticated_supports(&sources)
            .map_err(SemanticTransitionError::Semantic)?;
        let receipt = ReplayUpload::new(&self.provider, &self.stream, delivery_receipt)?;
        let payload = ReplayUpload::new(&self.provider, &self.stream, &packed)?;
        let observation_words = authenticated
            .observations
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let observations = ReplayUpload::new(&self.provider, &self.stream, &observation_words)?;
        let statements = ReplayUpload::new(&self.provider, &self.stream, &prepared.statements)?;
        let supports = ReplayUpload::new(&self.provider, &self.stream, &prepared.supports)?;
        #[cfg(feature = "semantic-policy")]
        let native_work = self.cold_native_work(lease.token)?;
        #[cfg(not(feature = "semantic-policy"))]
        let native_work: Option<DeviceMemoryView<u64>> = None;
        let (descriptors, raw) = arena.append_storage();
        let input = PublicationReplayDeliveryInput {
            delivery: PublicationDeliveryInput {
                lease: self.readers[&lease.token].device.device_ptr_value(),
                receipt: receipt.device.device_ptr_value(),
                receipt_len: delivery_receipt.len() as u64,
                recipient_id,
                receipt_digest: delivery_digest,
            },
            entry: SemanticTrainingReplayAppendEntry {
                row_ordinal: row_descriptor.ordinal,
                content_identity: row_descriptor.content_identity,
                original_origin: row_descriptor.origin,
                stable_intent: std::array::from_fn(|index| {
                    u64::from_ne_bytes(
                        stable.as_bytes()[index * 8..index * 8 + 8]
                            .try_into()
                            .expect("identity word"),
                    )
                }),
                result_receipt_digest: std::array::from_fn(|index| {
                    u64::from_ne_bytes(
                        result_digest.as_bytes()[index * 8..index * 8 + 8]
                            .try_into()
                            .expect("identity word"),
                    )
                }),
                payload_offset_bytes: queue_header.payload_used_bytes,
                payload_length_bytes: packed.len() as u64,
                disposition,
            },
            payload: payload.device.device_ptr_value(),
            payload_bytes: packed.len() as u64,
            row_descriptor,
            row_bytes: payload.device.device_ptr_value()
                + 40
                + size_of::<TrainingViewRowDescriptor>() as u64,
            row_bytes_len: row.bytes.len() as u64,
            arena_descriptors: descriptors.device_ptr_value(),
            arena_descriptor_count: descriptors.len() as u64,
            arena_raw: raw.device_ptr_value(),
            arena_raw_bytes: raw.len() as u64,
            observations: observations.device.device_ptr_value(),
            observation_count: authenticated.observations.len() as u64,
            statements: statements.device.device_ptr_value(),
            supports: supports.device.device_ptr_value(),
            native_work: native_work
                .as_ref()
                .map_or(0, DeviceMemoryView::device_ptr_value),
        };
        let input = ReplayUpload::new(&self.provider, &self.stream, &[input])?;
        let control = Arc::new(std::sync::Mutex::new(
            self.prepare_publication_read(publication.control.view())?,
        ));
        // Stage the exact successor header before any upload or CAS may enter.
        // Resolution joins these same owners after an unknown read completion.
        let successor_bank = ((lease.identity.word & 1) ^ 1) as usize;
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
            self.prepare_publication_read(header_view)?,
        ));
        // All pinned sources, destinations and pre-reserved graph references
        // precede the first transfer or numerical/publication entry boundary.
        self.pending_replay_delivery = Some(PendingReplayDelivery {
            base: lease.identity,
            fuel: lease.header.fuel,
            disposition,
            lease_token: lease.token,
            completion: None,
            receipt,
            payload,
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
            _arena: arena,
        });
        let result = self.resolve_replay_delivery(lease);
        if result.is_err()
            && self
                .pending_replay_delivery
                .as_ref()
                .is_some_and(|pending| {
                    !pending.kernel_entered
                        && pending.receipt.completed
                        && pending.payload.completed
                        && pending.observations.completed
                        && pending.statements.completed
                        && pending.supports.completed
                        && pending.input.completed
                })
        {
            // Every original DMA completed and the driver kernel boundary was
            // provably untouched. No publication can have committed.
            self.pending_replay_delivery = None;
        }
        result
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
            pending.payload.complete(&self.provider, &self.stream)?;
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
        let mut recorder = self.kernel_recorder();
        recorder.read(&pending.receipt.device);
        recorder.read(&pending.payload.device);
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
        if pending.disposition == 1 {
            let (descriptors, raw) = pending._arena.append_storage();
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
        let control =
            self.resolve_publication_read(&mut control_read.lock().map_err(|_| {
                publication_input_error("replay completion control owner poisoned")
            })?)?[0];
        let expected_word = ((base.word >> 1) + 1) << 1 | ((base.word & 1) ^ 1);
        if control.word == base.word && control.refusal != 0 {
            let status = control.refusal;
            self.pending_replay_delivery = None;
            return Err(SemanticTransitionError::PublicationRefused { status });
        }
        let header =
            self.resolve_publication_read(&mut header_read.lock().map_err(|_| {
                publication_input_error("replay completion header owner poisoned")
            })?)?[0];
        if control.word != expected_word
            || header.publication_word != expected_word
            || header.instance != base.instance
            || header.terminal != 0
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
            state_digest: header.state_digest,
        };
        let references = &self
            .pending_replay_delivery
            .as_ref()
            .ok_or(SemanticTransitionError::NoPendingLaunch)?
            .support_references;
        self.graph.finish_authenticated_supports(references);
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
