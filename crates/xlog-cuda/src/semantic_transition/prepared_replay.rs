//! Private original-step custody. Only canonical cold replay material escapes.

use super::*;

const SUCCESSOR_CUSTODY_ROLES: [u64; 9] = [14, 27, 28, 30, 31, 32, 33, 39, 48];

/// Canonical replay roots belonging to one original completed Proposal.
#[cfg(feature = "semantic-policy")]
pub struct SemanticCompletedReplayMaterials {
    pub predecessor: SemanticPublishedIdentity,
    pub successor: SemanticPublishedIdentity,
    pub parent: Vec<u8>,
    pub provenance: Vec<u8>,
    pub evidence: Vec<u8>,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ReplayCopyRow {
    role: u64,
    index: u64,
    model: u64,
    slots: [u64; 2],
    offsets: [u64; 2],
    bytes: [u64; 2],
    destination: u64,
    capacity: u64,
}

// SAFETY: fixed integer-only device ABI; no host references or drop state.
unsafe impl DeviceRepr for ReplayCopyRow {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct ReplayCopyDescriptor {
    rows: u64,
    count: u64,
    bank: u64,
    directory: u64,
    directory_count: u64,
    arena: u64,
    arena_words: u64,
    actual: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct PreparedReplayDescriptor {
    parent: ReplayCopyDescriptor,
    successor: ReplayCopyDescriptor,
}

struct ReplaySnapshot {
    rows: TrackedCudaSlice<ReplayCopyRow>,
    plan: Vec<ReplayCopyRow>,
    bank: TrackedCudaSlice<PublicationBank>,
    directory: TrackedCudaSlice<PublicationRange>,
    backings: Vec<TrackedCudaSlice<u8>>,
    arena: Option<TrackedCudaSlice<u64>>,
    actual: TrackedCudaSlice<u64>,
}

pub(super) struct PreparedReplayCustody {
    parent: ReplaySnapshot,
    successor: ReplaySnapshot,
    copy: CudaFunction,
    #[cfg(feature = "semantic-policy")]
    learning_phases: Vec<SemanticLearningPhaseRecord>,
}

fn replay_plan(
    storage: &PublicationStorage,
    parent: bool,
) -> Result<Vec<ReplayCopyRow>, SemanticTransitionError> {
    let mut rows = Vec::new();
    if parent {
        for (index, slots) in storage.model_slots.iter().enumerate() {
            rows.push(ReplayCopyRow {
                index: index as u64,
                model: 1,
                slots: slots.map(|slot| slot as u64),
                bytes: slots.map(|slot| storage.allocations[slot].len() as u64),
                ..ReplayCopyRow::default()
            });
        }
    }
    for range in &storage.bank_templates[0] {
        if matches!(range.role, 18..=25)
            || (!parent && !SUCCESSOR_CUSTODY_ROLES.contains(&range.role))
        {
            continue;
        }
        let mut row = ReplayCopyRow {
            role: range.role,
            index: range.index,
            ..ReplayCopyRow::default()
        };
        for bank in 0..2 {
            let range = storage.bank_templates[bank]
                .iter()
                .find(|item| (item.role, item.index) == (row.role, row.index))
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let allocation = storage
                .allocations
                .get(range.storage_slot as usize)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let offset = usize::try_from(range.offset_bytes)
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
            row.slots[bank] = range.storage_slot;
            row.offsets[bank] = range.offset_bytes;
            row.bytes[bank] = allocation
                .len()
                .checked_sub(offset)
                .ok_or(SemanticTransitionError::ObservationMismatch)?
                as u64;
        }
        rows.push(row);
    }
    for row in &mut rows {
        row.capacity = row.bytes[0].max(row.bytes[1]);
    }
    Ok(rows)
}

fn snapshot_allocation_bytes(
    rows: &[ReplayCopyRow],
    directory_count: usize,
    arena_words: usize,
) -> Result<u64, SemanticTransitionError> {
    let metadata = rows
        .len()
        .checked_mul(size_of::<ReplayCopyRow>())
        .and_then(|bytes| bytes.checked_add(size_of::<PublicationBank>()))
        .and_then(|bytes| {
            bytes.checked_add(directory_count.checked_mul(size_of::<PublicationRange>())?)
        })
        .and_then(|bytes| bytes.checked_add(arena_words.checked_mul(8)?))
        .and_then(|bytes| bytes.checked_add(3 * 8))
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    rows.iter().try_fold(metadata as u64, |bytes, row| {
        bytes
            .checked_add(row.capacity)
            .ok_or(SemanticTransitionError::GenerationExhausted)
    })
}

impl PreparedReplayCustody {
    pub(super) fn allocation_bytes(
        storage: &PublicationStorage,
        arena_words: usize,
    ) -> Result<u64, SemanticTransitionError> {
        let count = storage.bank_templates[0].len();
        let parent = snapshot_allocation_bytes(&replay_plan(storage, true)?, count, arena_words)?;
        let successor = snapshot_allocation_bytes(&replay_plan(storage, false)?, count, 0)?;
        parent
            .checked_add(successor)
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }

    pub(super) fn allocate(
        provider: &CudaKernelProvider,
        storage: &PublicationStorage,
        arena_words: usize,
        reservation: &mut GpuMemoryReservation,
        learning_phases: &[SemanticLearningPhaseRecord],
    ) -> Result<Self, SemanticTransitionError> {
        let copy = provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_prepared_replay_parent",
            )
            .ok_or_else(|| runtime_error("kernel lookup", "prepared replay custody unavailable"))?;
        #[cfg(not(feature = "semantic-policy"))]
        let _ = learning_phases;
        Ok(Self {
            parent: ReplaySnapshot::allocate(
                provider,
                replay_plan(storage, true)?,
                storage.bank_templates[0].len(),
                arena_words,
                reservation,
            )?,
            successor: ReplaySnapshot::allocate(
                provider,
                replay_plan(storage, false)?,
                storage.bank_templates[0].len(),
                0,
                reservation,
            )?,
            copy,
            #[cfg(feature = "semantic-policy")]
            learning_phases: learning_phases.to_vec(),
        })
    }

    pub(super) fn descriptor(&self) -> PreparedReplayDescriptor {
        PreparedReplayDescriptor {
            parent: self.parent.descriptor(),
            successor: self.successor.descriptor(),
        }
    }

    pub(super) fn copy_work_bound(&self) -> Result<u64, SemanticTransitionError> {
        let bound = |snapshot: &ReplaySnapshot| {
            let metadata = snapshot
                .directory
                .len()
                .checked_mul(size_of::<PublicationRange>())
                .and_then(|bytes| bytes.checked_add(size_of::<PublicationBank>() + 3 * 8))
                .and_then(|bytes| {
                    bytes.checked_add(
                        snapshot
                            .arena
                            .as_ref()
                            .map_or(0, |arena| arena.len())
                            .checked_mul(8)?,
                    )
                })
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            snapshot
                .plan
                .iter()
                .try_fold(metadata as u64, |bytes, row| {
                    bytes
                        .checked_add(row.capacity)
                        .ok_or(SemanticTransitionError::GenerationExhausted)
                })
        };
        bound(&self.parent)?
            .checked_add(bound(&self.successor)?)
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }

    pub(super) fn record(&self, recorder: &mut LaunchRecorder) {
        self.parent.record(recorder);
        self.successor.record(recorder);
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn material_capacity(
        &self,
        storage: &PublicationStorage,
        graph_bytes: usize,
    ) -> Result<(usize, usize), SemanticTransitionError> {
        let mut geometry = Vec::new();
        storage.model_memory.encode_into(&mut geometry)?;
        let mut history = Vec::new();
        learning_phase::encode_history(&self.learning_phases, &mut history)?;
        let mut parent = completed_material_extent(&[
            b"XLOG-PUBLICATION-MATERIAL\0".len(),
            4 + 32,
            4 + graph_bytes,
            4 + size_of::<PublicationBank>(),
            4 + size_of::<PublicationContract>(),
            4 + 55 * 8,
            4 + storage.terminals.len() * 8,
            4 + storage.layouts.len() * (4 + size_of::<SemanticTensorLayout>()),
            geometry.len(),
            history.len(),
            4,
        ])?;
        for row in &self.parent.plan {
            parent = completed_material_extent(&[
                parent,
                if row.model == 1 {
                    4 + usize::try_from(row.capacity)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?
                } else {
                    Self::range_capacity(row)?
                },
            ])?;
        }
        for range in &storage.bank_templates[0] {
            if matches!(range.role, 18..=25) {
                parent =
                    completed_material_extent(&[parent, 4 + size_of::<PublicationRange>() + 8])?;
            }
        }
        let authority = self
            .parent
            .plan
            .iter()
            .find(|row| (row.role, row.index) == (38, 0))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let decision = self
            .successor
            .plan
            .iter()
            .find(|row| (row.role, row.index) == (39, 0))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let provenance = completed_material_extent(&[
            REPLAY_PROVENANCE_MAGIC.len(),
            4 + 32,
            3 * 32 + 8,
            Self::range_capacity(authority)?,
            Self::range_capacity(decision)?,
        ])?;
        Ok((parent, provenance))
    }

    #[cfg(feature = "semantic-policy")]
    fn range_capacity(row: &ReplayCopyRow) -> Result<usize, SemanticTransitionError> {
        completed_material_extent(&[
            4 + size_of::<PublicationRange>() + 8 + 4,
            usize::try_from(row.capacity)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        ])
    }

    pub(super) fn enqueue_parent(
        &self,
        domain: &ResidentExecutionDomain,
        poisoned: &mut bool,
        storage: &PublicationStorage,
        lease: &TrackedCudaSlice<PublicationLease>,
        graph: &SemanticHypergraph,
    ) -> Result<(), SemanticTransitionError> {
        let mut recorder = domain.new_strict_recorder();
        storage.record(&mut recorder);
        graph.record_transition(&mut recorder);
        recorder.read(lease);
        self.parent.record(&mut recorder);
        enqueue_recorded(domain, poisoned, recorder, |enqueue| {
            // SAFETY: source publication/arena and every private destination are
            // retained by the original step. Native selection validates the exact
            // acquired directory, backing spans, and frozen capacity before copying.
            unsafe {
                self.copy.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        storage.control.device_ptr_value(),
                        lease.device_ptr_value(),
                        *graph.transition_arena_view().device_ptr(),
                        self.parent.descriptor(),
                    ),
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })
    }
}

#[cfg(feature = "semantic-policy")]
struct RetainedSnapshot {
    bank: DeviceMemoryView<PublicationBank>,
    directory: DeviceMemoryView<PublicationRange>,
    backings: Vec<DeviceMemoryView<u8>>,
    arena: Option<DeviceMemoryView<u64>>,
    actual: DeviceMemoryView<u64>,
    plan: Vec<ReplayCopyRow>,
}

#[cfg(feature = "semantic-policy")]
struct RetainedPublication {
    bank: PublicationBank,
    directory: Vec<PublicationRange>,
    ranges: Vec<PublicationMaterialRange>,
    models: Vec<Vec<u8>>,
}

#[cfg(feature = "semantic-policy")]
impl RetainedSnapshot {
    fn new(snapshot: &ReplaySnapshot) -> Self {
        Self {
            bank: snapshot.bank.view(),
            directory: snapshot.directory.view(),
            backings: snapshot
                .backings
                .iter()
                .map(TrackedCudaSlice::view)
                .collect(),
            arena: snapshot.arena.as_ref().map(TrackedCudaSlice::view),
            actual: snapshot.actual.view(),
            plan: snapshot.plan.clone(),
        }
    }

    fn read(
        &self,
        session: &mut SemanticTransitionSession,
    ) -> Result<RetainedPublication, SemanticTransitionError> {
        let actual = session.publication_read(self.actual.clone())?;
        let bank = session.publication_read(self.bank.clone())?[0];
        let directory = session.publication_read(self.directory.clone())?;
        if actual[2] != 1
            || actual[1] != bank.header.publication_word
            || bank.header.range_count as usize != directory.len()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let mut ranges = Vec::new();
        let mut models = Vec::new();
        let mut copied = size_of::<PublicationBank>()
            + 3 * size_of::<u64>()
            + directory.len() * size_of::<PublicationRange>()
            + self.arena.as_ref().map_or(0, |arena| arena.len() * 8);
        for (row, backing) in self.plan.iter().zip(&self.backings) {
            let mut selected = None;
            let mut original = None;
            for range in &directory {
                let matches = if row.model == 1 {
                    matches!(range.role, 18..=25) && row.slots.contains(&range.storage_slot)
                } else {
                    (range.role, range.index) == (row.role, row.index)
                };
                if !matches {
                    continue;
                }
                let bank = (0..2)
                    .find(|&bank| {
                        range.storage_slot == row.slots[bank]
                            && (row.model == 1 || range.offset_bytes == row.offsets[bank])
                    })
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if selected
                    .replace(bank)
                    .is_some_and(|previous| previous != bank)
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                original = Some(*range);
            }
            let selected = if row.model == 1 {
                selected.unwrap_or((bank.header.publication_word & 1) as usize)
            } else {
                selected.ok_or(SemanticTransitionError::ObservationMismatch)?
            };
            let capacity = usize::try_from(row.bytes[selected])
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
            copied = copied
                .checked_add(capacity)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let bytes = if capacity == 0 {
                Vec::new()
            } else {
                session.publication_read(
                    backing
                        .try_slice(0..capacity)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?,
                )?
            };
            if row.model == 1 {
                models.push(bytes);
            } else {
                let range = original.ok_or(SemanticTransitionError::ObservationMismatch)?;
                let used = usize::try_from(range.length_bytes)
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
                ranges.push(PublicationMaterialRange {
                    range,
                    capacity,
                    bytes: bytes
                        .get(..used)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?
                        .to_vec(),
                });
            }
        }
        if copied as u64 != actual[0] {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(RetainedPublication {
            bank,
            directory,
            ranges,
            models,
        })
    }
}

impl SemanticTransitionSession {
    /// Decode the actual acquired root from this step's pre-mutation arena copy.
    #[cfg(feature = "semantic-policy")]
    pub(super) fn prepared_parent_root_material(
        &mut self,
        step: &SemanticPreparedStep,
        header: &PublicationHeader,
    ) -> Result<SemanticRootMaterial, SemanticTransitionError> {
        let original = self.checked_prepared_step(step, false)?;
        let prepared = original.prepared.as_ref().expect("checked original owner");
        if !prepared.observed
            || !self
                .prepared_segment
                .as_ref()
                .expect("checked prepared scope")
                .completed
        {
            return Err(publication_input_error(
                "retained root requires this step's known completed execution",
            ));
        }
        let parent = &prepared
            .replay_custody
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .parent;
        let bank = parent.bank.view();
        let actual = parent.actual.view();
        let arena = parent
            .arena
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .view();
        let actual = self.publication_read(actual)?;
        if actual[2] != 1
            || actual[1] != header.publication_word
            || self.publication_read(bank)?[0].header != *header
        {
            self.poisoned = true;
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let arena = self.publication_read(arena)?;
        self.graph
            .export_retained_transition_root(
                &arena,
                header.semantic_owner,
                header.semantic_slot,
                header.semantic_generation,
                *header.semantic_digest.as_bytes(),
                header.semantic_extents,
            )
            .map_err(|error| {
                self.poisoned = true;
                SemanticTransitionError::Semantic(error)
            })
    }

    /// Canonical cold replay tuple for this original, known completed Proposal.
    /// Later publication-bank reuse cannot change these private native snapshots.
    #[cfg(feature = "semantic-policy")]
    pub fn prepared_completed_replay_materials(
        &mut self,
        step: &SemanticPreparedStep,
        consumer_streams: &[u64],
    ) -> Result<Option<SemanticCompletedReplayMaterials>, SemanticTransitionError> {
        if self.completed_prepared_transition_kind(step)? != SemanticTransitionKind::Proposal {
            return Err(publication_input_error(
                "completed replay export requires an original Proposal",
            ));
        }
        let original = self.checked_prepared_step(step, false)?;
        let prepared = original.prepared.as_ref().expect("checked original owner");
        let custody = prepared
            .replay_custody
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let parent = RetainedSnapshot::new(&custody.parent);
        let successor = RetainedSnapshot::new(&custody.successor);
        let history = custody.learning_phases.clone();
        let result_view = prepared.result.view();
        let reader_view = prepared.reader.view();
        let parent_header = original
            .inputs
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .header
            .view();
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("completed replay stream admission", error))?;
        self.complete_step_consumers_by_token(step.token, consumer_streams)?;
        let result = self.publication_read(result_view)?[0];
        let lease = self.publication_read(reader_view)?[0];
        let header = self.publication_read(parent_header)?[0];
        let kind = validate_prepared_completion(&lease, &header, &result, storage.instance)?;
        if result.advanced == 0 {
            return Ok(None);
        }
        if kind != SemanticTransitionKind::Proposal {
            return Err(publication_input_error(
                "completed replay export requires the actual Proposal publication",
            ));
        }
        let observation = (|| {
            let RetainedPublication {
                bank,
                directory,
                mut ranges,
                models: model_allocations,
            } = parent.read(self)?;
            if bank.header != header {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let graph = self.prepared_parent_root_material(step, &header)?;
            for range in &directory {
                if matches!(range.role, 18..=25) {
                    let (allocation, offset) =
                        storage.model_memory.location(range.role, range.index)?;
                    let allocation = model_allocations
                        .get(allocation)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    let used = usize::try_from(range.length_bytes)
                        .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
                    let end = offset
                        .checked_add(used)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    ranges.push(PublicationMaterialRange {
                        range: *range,
                        capacity: allocation
                            .len()
                            .checked_sub(offset)
                            .ok_or(SemanticTransitionError::ObservationMismatch)?,
                        bytes: allocation
                            .get(offset..end)
                            .ok_or(SemanticTransitionError::ObservationMismatch)?
                            .to_vec(),
                    });
                }
            }
            ranges.sort_by_key(|item| {
                directory.iter().position(|range| {
                    (range.role, range.index) == (item.range.role, item.range.index)
                })
            });
            let counts = self.publication_read(storage.role_counts.view())?;
            let mut role_counts = [0; 55];
            for (index, count) in counts.iter().enumerate() {
                if count.role != index as u64 + 1 {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                role_counts[index] = count.count;
            }
            let terminals = if storage.terminals.is_empty() {
                Vec::new()
            } else {
                self.publication_read(storage.terminals.view())?
            };
            let mut contract = storage.contract_value;
            contract.terminal_tokens = 0;
            contract.role_counts = 0;
            contract.semantic_owner = 0;
            let material = PublicationMaterial {
                bank,
                contract,
                role_counts,
                terminals,
                layouts: storage.layouts.clone(),
                model_memory: storage.model_memory.clone(),
                model_allocations,
                ranges,
                graph,
                learning_phases: history,
            };
            let RetainedPublication {
                bank: next,
                ranges: next_ranges,
                ..
            } = successor.read(self)?;
            if next.header != result.header {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let identity = |header: PublicationHeader| SemanticPublishedIdentity {
                instance: header.instance,
                word: header.publication_word,
                logical_digest: header.logical_digest,
                state_digest: header.state_digest,
            };
            let range = |role| {
                next_ranges
                    .iter()
                    .find(|item| (item.range.role, item.range.index) == (role, 0))
                    .ok_or(SemanticTransitionError::ObservationMismatch)
            };
            let evidence = PublicationReplayEvidence {
                successor: identity(next.header),
                ranges: REPLAY_EVIDENCE_ROLES
                    .iter()
                    .map(|&role| range(role).cloned())
                    .collect::<Result<_, _>>()?,
            };
            evidence.validate_observed(&material, &next, range(14)?, range(48)?)?;
            let provenance = PublicationReplayProvenance {
                predecessor: identity(header),
                authority: material
                    .ranges
                    .iter()
                    .find(|item| (item.range.role, item.range.index) == (38, 0))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
                    .clone(),
                decision: range(39)?.clone(),
            };
            provenance.validate(&material)?;
            Ok(Some(SemanticCompletedReplayMaterials {
                predecessor: identity(header),
                successor: identity(next.header),
                parent: material.encode()?,
                provenance: provenance.encode()?,
                evidence: evidence.encode()?,
            }))
        })();
        if observation.is_err() {
            self.poisoned = true;
        }
        observation
    }
}

impl ReplaySnapshot {
    fn allocate(
        provider: &CudaKernelProvider,
        mut plan: Vec<ReplayCopyRow>,
        count: usize,
        arena_words: usize,
        reservation: &mut GpuMemoryReservation,
    ) -> Result<Self, SemanticTransitionError> {
        let mut backings = Vec::with_capacity(plan.len());
        for row in &mut plan {
            let backing = reservation
                .alloc::<u8>(
                    usize::try_from(row.capacity)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                )
                .map_err(|error| runtime_error("replay backing reservation", error))?;
            row.destination = backing.device_ptr_value();
            backings.push(backing);
        }
        let rows = reservation
            .alloc(plan.len())
            .map_err(|error| runtime_error("replay roster reservation", error))?;
        upload_publication(provider, &plan, &rows)?;
        let bank = reservation
            .alloc(1)
            .map_err(|error| runtime_error("replay bank reservation", error))?;
        let directory = reservation
            .alloc(count)
            .map_err(|error| runtime_error("replay directory reservation", error))?;
        let arena = if arena_words == 0 {
            None
        } else {
            Some(
                reservation
                    .alloc(arena_words)
                    .map_err(|error| runtime_error("replay arena reservation", error))?,
            )
        };
        let actual = reservation
            .alloc(3)
            .map_err(|error| runtime_error("replay work reservation", error))?;
        upload_publication(provider, &[0u64; 3], &actual)?;
        Ok(Self {
            rows,
            plan,
            bank,
            directory,
            backings,
            arena,
            actual,
        })
    }

    fn descriptor(&self) -> ReplayCopyDescriptor {
        ReplayCopyDescriptor {
            rows: self.rows.device_ptr_value(),
            count: self.rows.len() as u64,
            bank: self.bank.device_ptr_value(),
            directory: self.directory.device_ptr_value(),
            directory_count: self.directory.len() as u64,
            arena: self
                .arena
                .as_ref()
                .map_or(0, TrackedCudaSlice::device_ptr_value),
            arena_words: self.arena.as_ref().map_or(0, |arena| arena.len() as u64),
            actual: self.actual.device_ptr_value(),
        }
    }

    fn record(&self, recorder: &mut LaunchRecorder) {
        recorder.read(&self.rows);
        recorder.read_write(&self.bank);
        recorder.read_write(&self.directory);
        recorder.read_write(&self.actual);
        for backing in &self.backings {
            recorder.read_write(backing);
        }
        if let Some(arena) = &self.arena {
            recorder.read_write(arena);
        }
    }
}
