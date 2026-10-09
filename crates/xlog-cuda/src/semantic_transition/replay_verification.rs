//! Original cold replay comparison, including every entered guard and read.

use super::*;

/// Capability for one comparison of two original held publication readers.
/// It neither acquires a reader nor issues authority to use replay material.
pub struct SemanticReplayPublicationVerification {
    issuer: Arc<()>,
    owner: Arc<Mutex<OriginalReplayPublicationVerification>>,
}

struct VerificationParent {
    token: u64,
    identity: SemanticPublishedIdentity,
    header: PublicationHeader,
    source: [SemanticTextSlot; 32],
    directory: Vec<PublicationRange>,
}

impl VerificationParent {
    fn new(lease: &SemanticPublishedLease) -> Self {
        Self {
            token: lease.token,
            identity: lease.identity,
            header: lease.header,
            source: lease.source,
            directory: lease.directory.clone(),
        }
    }

    fn matches(&self, lease: &SemanticPublishedLease) -> bool {
        lease.active
            && self.token == lease.token
            && self.identity == lease.identity
            && publication_abi_bytes(&[self.header]) == publication_abi_bytes(&[lease.header])
            && publication_abi_bytes(&self.source) == publication_abi_bytes(&lease.source)
            && publication_abi_bytes(&self.directory) == publication_abi_bytes(&lease.directory)
    }
}

pub(super) struct RetainedMaterialRange {
    pub(super) range: PublicationRange,
    pub(super) capacity: usize,
    pub(super) read: Option<PublicationRead<u8>>,
}

impl RetainedMaterialRange {
    pub(super) fn resolve(
        &mut self,
        session: &mut SemanticTransitionSession,
        poisoned: &mut bool,
    ) -> Result<PublicationMaterialRange, SemanticTransitionError> {
        Ok(PublicationMaterialRange {
            range: self.range,
            capacity: self.capacity,
            bytes: if let Some(read) = &mut self.read {
                session.resolve_publication_read_with_poison(read, poisoned)?
            } else {
                Vec::new()
            },
        })
    }
}

struct OriginalPublicationGuard {
    storage: Arc<PublicationStorage>,
    inputs: Option<Arc<PreparedStepInputs>>,
    input_command: Option<OriginalNativeCommand>,
    ranges: Vec<(u64, u64, SemanticTensorLayout)>,
    commands: Vec<OriginalNativeCommand>,
    edges: [OriginalConsumerEdge; 3],
    execute: CudaFunction,
    cold_work: Option<DeviceMemoryView<u64>>,
    cursor: usize,
    poisoned: bool,
}

impl OriginalPublicationGuard {
    fn complete(
        &mut self,
        session: &mut SemanticTransitionSession,
        lease: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&session.stream)
            .map_err(|error| runtime_error("original replay guard stream admission", error))?;
        let end = self
            .ranges
            .len()
            .checked_add(4)
            .ok_or_else(|| publication_input_error("replay guard cursor overflow"))?;
        while self.cursor < end {
            let result = match self.cursor {
                0 | 1 => self.edges[self.cursor].join(),
                2 => {
                    if let Some(inputs) = &self.inputs {
                        inputs.verify_basis(
                            &session.domain,
                            &mut self.poisoned,
                            None,
                            false,
                            None,
                            self.input_command.as_mut(),
                        )
                    } else {
                        Ok(())
                    }
                }
                ordinal if ordinal == end - 1 => self.edges[2].join(),
                ordinal => {
                    let index = ordinal - 3;
                    let (role, coordinate, layout) = self.ranges[index];
                    self.storage.enqueue_content_guard(
                        &session.domain,
                        &mut self.poisoned,
                        session.readers[&lease.token].device.view(),
                        &self.execute,
                        role,
                        coordinate,
                        layout,
                        self.cold_work.as_ref(),
                        Some(&mut self.commands[index]),
                    )
                }
            };
            if let Err(error) = result {
                self.poisoned |= self.edges.iter().any(edge_unknown);
                return Err(error);
            }
            self.cursor += 1;
        }
        self.poisoned = false;
        Ok(())
    }

    fn pending(&self) -> bool {
        self.poisoned
            && (self.edges.iter().any(edge_unknown)
                || self.input_command.as_ref().is_some_and(command_unknown)
                || self.commands.iter().any(command_unknown))
    }
}

fn edge_unknown(edge: &OriginalConsumerEdge) -> bool {
    (edge.record_entered && !edge.recorded && !edge.source_completed)
        || (edge.wait_entered && !edge.waited)
}

fn command_unknown(command: &OriginalNativeCommand) -> bool {
    !command.completed && (command.completion_entered || command.entered)
}

struct ReplayEvidenceObservation {
    bank: PublicationRead<PublicationBank>,
    ranges: Vec<RetainedMaterialRange>,
    coverage: RetainedMaterialRange,
    codebook: RetainedMaterialRange,
}

impl ReplayEvidenceObservation {
    fn resolve(
        &mut self,
        session: &mut SemanticTransitionSession,
        predecessor: &PublicationMaterial,
        successor: &SemanticPublishedLease,
        poisoned: &mut bool,
    ) -> Result<PublicationReplayEvidence, SemanticTransitionError> {
        let bank = *session
            .resolve_publication_read_with_poison(&mut self.bank, poisoned)?
            .first()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        SemanticTransitionSession::validate_published_bank(successor, &bank)?;
        let mut ranges = Vec::with_capacity(self.ranges.len());
        for range in &mut self.ranges {
            ranges.push(range.resolve(session, poisoned)?);
        }
        let evidence = PublicationReplayEvidence {
            successor: successor.identity,
            ranges,
        };
        let coverage = self.coverage.resolve(session, poisoned)?;
        let codebook = self.codebook.resolve(session, poisoned)?;
        evidence.validate_observed(predecessor, &bank, &coverage, &codebook)?;
        Ok(evidence)
    }

    fn pending(&self) -> bool {
        read_unknown(&self.bank)
            || self
                .ranges
                .iter()
                .chain([&self.coverage, &self.codebook])
                .any(|range| range.read.as_ref().is_some_and(read_unknown))
    }
}

fn read_unknown<T: DeviceRepr + Copy>(read: &PublicationRead<T>) -> bool {
    read.read.entered() && !read.read.completed()
}

pub(super) struct OriginalReplayPublicationVerification {
    predecessor: VerificationParent,
    successor: VerificationParent,
    expected_predecessor: SemanticPublishedIdentity,
    expected_successor: SemanticPublishedIdentity,
    expected_ranges: Vec<Identity256>,
    consumers: Option<[OriginalStepConsumers; 2]>,
    guards: Option<[OriginalPublicationGuard; 2]>,
    predecessor_observation: Option<SemanticPublishedStateObservation>,
    predecessor_material: Option<PublicationMaterial>,
    successor_observation: Option<ReplayEvidenceObservation>,
    cursor: usize,
    poisoned: bool,
    completed: bool,
}

impl SemanticTransitionSession {
    /// Allocate one original comparison owner. No guard, copy or snapshot is
    /// enqueued until the caller retains this capability and resolves it.
    pub fn begin_replay_publication_verification(
        &mut self,
        material: &SemanticReplayMaterial,
        predecessor: &SemanticPublishedLease,
        successor: &SemanticPublishedLease,
    ) -> Result<SemanticReplayPublicationVerification, SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.checked_reader(predecessor)?;
        self.checked_reader(successor)?;
        self.checked_original_consumer_step(predecessor)?;
        self.checked_original_consumer_step(successor)?;
        if predecessor.token == successor.token || self.replay_verification.is_some() {
            return Err(publication_input_error(
                "replay comparison requires two original distinct held readers",
            ));
        }
        let consumers = [
            self.stage_replay_consumers(predecessor)?,
            self.stage_replay_consumers(successor)?,
        ];
        let guards = [
            self.stage_replay_publication_guard(predecessor)?,
            self.stage_replay_publication_guard(successor)?,
        ];
        let storage = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let bank = self
            .stage_publication_read(storage.banks[(successor.identity.word & 1) as usize].view())?;
        let ranges = REPLAY_EVIDENCE_ROLES
            .iter()
            .map(|role| self.stage_published_material_range(successor, *role, 0))
            .collect::<Result<Vec<_>, _>>()?;
        let expected_ranges = material
            .evidence
            .ranges
            .iter()
            .map(PublicationMaterialRange::logical_record_digest)
            .collect::<Result<Vec<_>, _>>()?;
        let owner = Arc::new(Mutex::new(OriginalReplayPublicationVerification {
            predecessor: VerificationParent::new(predecessor),
            successor: VerificationParent::new(successor),
            expected_predecessor: material.predecessor_identity(),
            expected_successor: material.successor_identity(),
            expected_ranges,
            consumers: Some(consumers),
            guards: Some(guards),
            predecessor_observation: None,
            predecessor_material: None,
            successor_observation: Some(ReplayEvidenceObservation {
                bank,
                ranges,
                coverage: self.stage_published_material_range(successor, 14, 0)?,
                codebook: self.stage_published_material_range(successor, 48, 0)?,
            }),
            cursor: 0,
            poisoned: false,
            completed: false,
        }));
        // The Session retains the owner independently of the Python handoff
        // and moves it into its existing quarantine on an unresolved drop.
        self.replay_verification = Some(Arc::clone(&owner));
        self.replay_verification_pending
            .store(true, Ordering::Release);
        Ok(SemanticReplayPublicationVerification {
            issuer: Arc::clone(&self.publication_issuer),
            owner,
        })
    }

    fn stage_replay_consumers(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<OriginalStepConsumers, SemanticTransitionError> {
        let streams = self.readers[&lease.token]
            .consumer_streams
            .iter()
            .chain(&self.steps[&lease.token].consumer_streams)
            .copied()
            .collect::<BTreeSet<_>>();
        OriginalStepConsumers::new(
            &self.stream,
            Some(lease.identity),
            streams.clone(),
            streams,
            OriginalConsumerPurpose::Observation,
        )
    }

    fn stage_replay_publication_guard(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<OriginalPublicationGuard, SemanticTransitionError> {
        let keys = self.published_range_keys(lease)?;
        let ranges = self.publication_content_guard_ranges(lease, &keys)?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let inputs = if keys
            .iter()
            .any(|(role, _)| matches!(*role as u64, 1 | 3..=13))
        {
            self.steps[&lease.token].inputs.as_ref().map(Arc::clone)
        } else {
            None
        };
        let input_command = inputs
            .as_ref()
            .map(|_| OriginalNativeCommand::new(&self.domain))
            .transpose()?;
        let commands = ranges
            .iter()
            .map(|_| OriginalNativeCommand::new(&self.domain))
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(feature = "semantic-policy")]
        let cold_work = self.cold_native_work(lease.token)?;
        #[cfg(not(feature = "semantic-policy"))]
        let cold_work = None;
        let native = self.stream.cu_stream();
        let consumer = dlpack_consumer_stream(1)?;
        Ok(OriginalPublicationGuard {
            storage,
            inputs,
            input_command,
            ranges,
            commands,
            edges: [
                OriginalConsumerEdge::new(&self.stream, std::ptr::null_mut(), native)?,
                OriginalConsumerEdge::new(&self.stream, consumer, native)?,
                OriginalConsumerEdge::new(&self.stream, native, consumer)?,
            ],
            execute: self
                .provider
                .device()
                .inner()
                .get_func(
                    "xlog_semantic_transition",
                    "semantic_publication_content_guard",
                )
                .ok_or_else(|| {
                    runtime_error("kernel lookup", "publication content guard unavailable")
                })?,
            cold_work,
            cursor: 0,
            poisoned: false,
        })
    }

    fn require_replay_verification(
        &self,
        capability: &SemanticReplayPublicationVerification,
        original: &OriginalReplayPublicationVerification,
        predecessor: &SemanticPublishedLease,
        successor: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_original_consumer_step(predecessor)?;
        self.checked_original_consumer_step(successor)?;
        if !Arc::ptr_eq(&capability.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&predecessor.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&successor.issuer, &self.publication_issuer)
            || !original.predecessor.matches(predecessor)
            || !original.successor.matches(successor)
            || self
                .replay_verification
                .as_ref()
                .is_some_and(|owner| !Arc::ptr_eq(owner, &capability.owner))
            || (!original.completed && self.replay_verification.is_none())
        {
            return Err(publication_input_error(
                "replay comparison changed its original readers or owner",
            ));
        }
        if self.has_unresolved_session_native_effect_except_replay_verification()
            || self.pending
            || self.pending_replay_delivery.is_some()
            || self.pending_training_materialization.is_some()
            || self
                .readers
                .values()
                .any(|reader| reader.retirement_pending || reader.acquisition_pending)
            || self
                .steps
                .values()
                .any(|step| step.consumer_completion.is_some())
        {
            return Err(SemanticTransitionError::Poisoned);
        }
        #[cfg(feature = "semantic-policy")]
        if self.has_pending_cold_model_work_completion(None) {
            return Err(SemanticTransitionError::Poisoned);
        }
        if let Some(observation) = &original.predecessor_observation {
            self.graph
                .require_root_export_continuation(&observation.graph)
                .map_err(SemanticTransitionError::Semantic)?;
        } else {
            self.graph
                .ensure_not_poisoned()
                .map_err(SemanticTransitionError::Semantic)?;
        }
        Ok(())
    }

    /// Continue only the original guard/read roster; no entered command,
    /// snapshot or copy can be enqueued a second time.
    pub fn resolve_replay_publication_verification(
        &mut self,
        capability: &mut SemanticReplayPublicationVerification,
        predecessor: &SemanticPublishedLease,
        successor: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        let owner = Arc::clone(&capability.owner);
        let mut original = owner
            .lock()
            .map_err(|_| publication_input_error("original replay comparison owner poisoned"))?;
        self.require_replay_verification(capability, &original, predecessor, successor)?;
        if original.completed {
            return Ok(());
        }
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("original replay comparison stream admission", error))?;
        let result = (|| {
            while original.cursor < 6 {
                let cursor = original.cursor;
                match cursor {
                    0 | 1 => original
                        .consumers
                        .as_mut()
                        .expect("retained original comparison consumers")[cursor]
                        .complete(&self.stream, &mut self.stream_waits, true)?,
                    2 | 3 => {
                        let lease = if cursor == 2 { predecessor } else { successor };
                        self.readers
                            .get_mut(&lease.token)
                            .expect("retained comparison reader")
                            .consumer_streams
                            .insert(1);
                        self.steps
                            .get_mut(&lease.token)
                            .expect("retained comparison step")
                            .consumer_streams
                            .insert(1);
                        original
                            .guards
                            .as_mut()
                            .expect("retained original comparison guards")[cursor - 2]
                            .complete(self, lease)?;
                    }
                    4 => {
                        if original.predecessor_observation.is_none() {
                            original.predecessor_observation =
                                Some(self.stage_published_state_observation(predecessor, false)?);
                        }
                        let OriginalReplayPublicationVerification {
                            predecessor_observation,
                            poisoned,
                            ..
                        } = &mut *original;
                        let bytes = self.resolve_published_state_observation_with_poison(
                            predecessor,
                            predecessor_observation
                                .as_mut()
                                .expect("original predecessor observation"),
                            poisoned,
                        )?;
                        let material = PublicationMaterial::decode(&bytes)?;
                        let header = material.bank.header;
                        let expected = original.expected_predecessor;
                        if header.instance == expected.instance
                            || header.recovered_instance != expected.instance
                            || header.publication_word != 0
                            || header.logical_digest != expected.logical_digest
                            || header.state_digest != expected.state_digest
                        {
                            return Err(publication_input_error("replay comparison requires the freshly restored complete predecessor"));
                        }
                        original.predecessor_material = Some(material);
                        original.poisoned = false;
                    }
                    5 => {
                        let OriginalReplayPublicationVerification {
                            predecessor_material,
                            successor_observation,
                            poisoned,
                            ..
                        } = &mut *original;
                        let actual = successor_observation
                            .as_mut()
                            .expect("retained original successor observation")
                            .resolve(
                                self,
                                predecessor_material
                                    .as_ref()
                                    .expect("complete original predecessor"),
                                successor,
                                poisoned,
                            )?;
                        if actual.successor.logical_digest
                            != original.expected_successor.logical_digest
                            || actual.successor.state_digest
                                != original.expected_successor.state_digest
                        {
                            return Err(publication_input_error("genuine replay successor differs from the expected complete native state"));
                        }
                        if actual.ranges.len() != original.expected_ranges.len() {
                            return Err(SemanticTransitionError::ObservationMismatch);
                        }
                        for (actual, expected) in
                            actual.ranges.iter().zip(&original.expected_ranges)
                        {
                            if actual.logical_record_digest()? != *expected {
                                return Err(publication_input_error("genuine replay differs from the expected normalized publication records"));
                            }
                        }
                        original.poisoned = false;
                    }
                    _ => unreachable!("bounded replay comparison cursor"),
                }
                original.cursor += 1;
            }
            Ok(())
        })();
        if let Err(error) = result {
            if !matches!(
                error,
                SemanticTransitionError::Runtime { .. }
                    | SemanticTransitionError::Semantic(SemanticHypergraphError::Runtime { .. })
            ) {
                self.poisoned = true;
            }
            return Err(error);
        }
        // Every original command/read has positive completion. Keep only the
        // immutable comparison identities and result; retaining device views
        // here would prevent the subsequent original reader retirement.
        original.consumers = None;
        original.guards = None;
        original.predecessor_observation = None;
        original.predecessor_material = None;
        original.successor_observation = None;
        original.completed = true;
        self.replay_verification_pending
            .store(false, Ordering::Release);
        self.replay_verification = None;
        Ok(())
    }

    /// Metadata-only classification of uncertainty retained by this exact
    /// comparison. Allocation failures and semantic mismatches are not pending.
    pub fn replay_publication_verification_pending(
        &self,
        capability: &SemanticReplayPublicationVerification,
        predecessor: &SemanticPublishedLease,
        successor: &SemanticPublishedLease,
    ) -> Result<bool, SemanticTransitionError> {
        let original = capability
            .owner
            .lock()
            .map_err(|_| publication_input_error("original replay comparison owner poisoned"))?;
        self.require_replay_verification(capability, &original, predecessor, successor)?;
        if original.completed {
            return Ok(false);
        }
        if original.consumers.as_ref().is_some_and(|owners| {
            owners.iter().any(|owner| {
                !owner.completed
                    && owner.poisoned
                    && (owner.cursor != 0 || owner.edges.iter().any(edge_unknown))
            })
        }) || original
            .guards
            .as_ref()
            .is_some_and(|guards| guards.iter().any(OriginalPublicationGuard::pending))
        {
            return Ok(true);
        }
        if let Some(observation) = &original.predecessor_observation {
            if self
                .graph
                .root_export_pending(&observation.graph)
                .map_err(SemanticTransitionError::Semantic)?
            {
                return Ok(true);
            }
            if original.poisoned
                && (read_unknown(&observation.bank)
                    || read_unknown(&observation.counts)
                    || observation.control.as_ref().is_some_and(read_unknown)
                    || observation.terminals.as_ref().is_some_and(read_unknown)
                    || observation.models.iter().flatten().any(read_unknown)
                    || observation
                        .ranges
                        .iter()
                        .filter_map(|(_, _, read)| read.as_ref())
                        .any(read_unknown))
            {
                return Ok(true);
            }
        }
        Ok(original.poisoned
            && original
                .successor_observation
                .as_ref()
                .is_some_and(ReplayEvidenceObservation::pending))
    }
}
