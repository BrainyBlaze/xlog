use super::*;

// The first 25 words still identify the context's actual semantic bytes. The
// remaining header binds model rows, owned category ranges and the cold frame.
pub(super) const HEADER_WORDS: usize = 44;
const FRAME_MAGIC: u64 = 0x584c4f474d4f444c;
const ROW_WIDTHS: [usize; 4] = [10, 6, 3, 18];

/// Process-local authority derived only from a completed native cold domain.
/// Fresh restoration reissues it from the complete ordered cold roster.
#[derive(Clone)]
pub struct SemanticFrozenModelContext {
    pub(super) catalogue: Arc<FrozenModelCatalogue>,
    pub(super) member: usize,
}

impl SemanticFrozenModelContext {
    pub fn require_budget(
        &self,
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<(), SemanticTransitionError> {
        if !self.catalogue._reservation.budget.same_pool(budget) {
            return Err(publication_input_error(
                "frozen model context requires its original host capacity pool",
            ));
        }
        Ok(())
    }
}

impl SemanticReplayStateMaterialProof {
    pub fn require_frozen_model_context(
        &self,
        context: Option<&SemanticFrozenModelContext>,
    ) -> Result<(), SemanticTransitionError> {
        let header = &self.owner.codebook_header;
        match (header[40] != 0, context) {
            (false, None) => Ok(()),
            (true, Some(context)) if context.catalogue.matches_header(context.member, header) => {
                Ok(())
            }
            _ => Err(publication_input_error(
                "checkpoint model rows require the same freshly frozen original cold domain",
            )),
        }
    }
}

struct CatalogueMember {
    issuer: Arc<()>,
    content: SemanticTaskContentIdentity,
    words: Vec<u64>,
    program: Option<Identity256>,
}

/// One original finite roster, claimed before any cold composition effect.
pub(super) struct ModelContextPreseal {
    issuers: Vec<Arc<()>>,
    pending: AtomicBool,
    _reservation: SemanticReplayMaterialReservation,
}

impl ModelContextPreseal {
    fn requires_roster(&self, contexts: &[&SemanticTransitionSession]) -> bool {
        self.issuers.len() == contexts.len()
            && self
                .issuers
                .iter()
                .zip(contexts)
                .all(|(issuer, context)| Arc::ptr_eq(issuer, &context.publication_issuer))
    }
}

/// All original cold recipes, in their immutable model-row order. It grants
/// no task authority: each projection retains only its own descriptor meanings.
pub(super) struct FrozenModelCatalogue {
    members: Vec<CatalogueMember>,
    frame: Vec<u64>,
    binding: SemanticCatalogueBinding,
    statement_bytes: usize,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

fn normalized(words: &[u64]) -> impl Iterator<Item = u64> + '_ {
    words.iter().enumerate().map(|(index, &word)| {
        if matches!(index, 14..=16 | 21..=24) {
            0
        } else {
            word
        }
    })
}

impl FrozenModelCatalogue {
    pub(super) fn member_count(&self) -> usize {
        self.members.len()
    }

    pub(super) fn member_words(&self, member: usize) -> Result<&[u64], SemanticTransitionError> {
        self.members
            .get(member)
            .map(|member| member.words.as_slice())
            .ok_or(SemanticTransitionError::ObservationMismatch)
    }

    fn category_range(
        &self,
        table: usize,
        member: usize,
    ) -> Result<(u64, u64, u64), SemanticTransitionError> {
        if member >= self.members.len() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let count_word = 2 * table + 1;
        let mut total = 1u64;
        let mut begin = 1;
        let mut end = 1;
        for (ordinal, context) in self.members.iter().enumerate() {
            if ordinal == member {
                begin = total;
            }
            total = context.words[count_word]
                .checked_sub(1)
                .and_then(|count| total.checked_add(count))
                .filter(|&count| count <= 65536)
                .ok_or_else(|| {
                    publication_input_error("immutable model field exceeds 65536 categories")
                })?;
            if ordinal == member {
                end = total;
            }
        }
        Ok((total, begin, end))
    }

    fn matches_header(&self, member: usize, header: &[u64; HEADER_WORDS]) -> bool {
        let Some(selected) = self.members.get(member) else {
            return false;
        };
        header[40] != 0
            && words_identity(header, 25) == *self.binding.digest.as_bytes()
            && header[37] == self.statement_bytes as u64
            && header[38] == u64::from(self.members.iter().any(|member| member.program.is_some()))
            && header[39] == u64::from(selected.program.is_some())
            && header[41] == (self.frame.len() * size_of::<u64>()) as u64
            && header[42] == member as u64 + 1
            && header[43] == self.members.len() as u64
            && header[10..14] == selected.words[10..14]
            && (0..ROW_WIDTHS.len()).all(|table| {
                self.category_range(table, member)
                    .is_ok_and(|(count, begin, end)| {
                        header[2 * table + 1] == count
                            && header[29 + 2 * table] == begin
                            && header[30 + 2 * table] == end
                    })
            })
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn context_for_header(
        self: &Arc<Self>,
        header: &[u64; HEADER_WORDS],
    ) -> Result<SemanticFrozenModelContext, SemanticTransitionError> {
        let member = header[42]
            .checked_sub(1)
            .and_then(|member| usize::try_from(member).ok())
            .filter(|&member| self.matches_header(member, header));
        if let Some(member) = member {
            return Ok(SemanticFrozenModelContext {
                catalogue: Arc::clone(self),
                member,
            });
        }
        Err(publication_input_error(
            "original source geometry changed its context-owned model row resolution",
        ))
    }

    fn new(
        sessions: &[&mut SemanticTransitionSession],
        statement_bytes: usize,
        reservation: SemanticReplayMaterialReservation,
    ) -> Result<Arc<Self>, SemanticTransitionError> {
        let mut members = Vec::new();
        members
            .try_reserve_exact(sessions.len())
            .map_err(|error| runtime_error("model context roster allocation", error))?;
        let mut frame_words = 4usize;
        for session in sessions {
            let mut words = Vec::new();
            words
                .try_reserve_exact(session.codebooks.words.len())
                .map_err(|error| runtime_error("model context recipe allocation", error))?;
            words.extend(normalized(&session.codebooks.words));
            frame_words = frame_words
                .checked_add(18)
                .and_then(|size| size.checked_add(words.len()))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            members.push(CatalogueMember {
                issuer: Arc::clone(&session.publication_issuer),
                content: session
                    .cold_content
                    .as_ref()
                    .expect("checked cold content")
                    .0,
                words,
                program: session.codebooks.program_binding,
            });
        }
        let mut frame = Vec::new();
        frame
            .try_reserve_exact(frame_words)
            .map_err(|error| runtime_error("model context frame allocation", error))?;
        frame.extend([FRAME_MAGIC, 2, statement_bytes as u64, members.len() as u64]);
        for member in &members {
            for identity in [
                member.content.query,
                member.content.theory_program,
                member.content.result,
            ] {
                frame.extend(identity_words(*identity.as_bytes()));
            }
            frame.push(u64::from(member.program.is_some()));
            frame.extend(identity_words(
                member
                    .program
                    .map_or([0; 32], |identity| *identity.as_bytes()),
            ));
            frame.push(member.words.len() as u64);
            frame.extend_from_slice(&member.words);
        }
        let mut hash = Sha256::new();
        hash.update(b"xlog.semantic.immutable-model-catalogue.v2\0");
        hash.update(CATALOGUE_DIGEST);
        for &word in &frame {
            hash.update(word.to_le_bytes());
        }
        let binding = SemanticCatalogueBinding {
            generation: CATALOGUE_GENERATION,
            digest: Identity256::from_bytes(hash.finalize().into()),
        };
        Ok(Arc::new(Self {
            members,
            frame,
            binding,
            statement_bytes,
            _reservation: Arc::new(reservation),
        }))
    }

    fn project(
        &self,
        original_words: &[u64],
        original_program: Option<Identity256>,
        original_binding: SemanticCatalogueBinding,
        member: usize,
    ) -> Result<Vec<u64>, SemanticTransitionError> {
        let selected = self
            .members
            .get(member)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if selected.program != original_program
            || !normalized(original_words).eq(selected.words.iter().copied())
        {
            return Err(publication_input_error(
                "model catalogue requires its original cold context recipe",
            ));
        }
        let word_bound = self.members.iter().try_fold(
            self.frame
                .len()
                .checked_add(HEADER_WORDS)
                .ok_or(SemanticTransitionError::GenerationExhausted)?,
            |bound, member| {
                bound
                    .checked_add(member.words.len())
                    .ok_or(SemanticTransitionError::GenerationExhausted)
            },
        )?;
        let mut words = Vec::new();
        words
            .try_reserve_exact(word_bound)
            .map_err(|error| runtime_error("model context projection allocation", error))?;
        words.resize(HEADER_WORDS, 0);
        words[..25].copy_from_slice(&original_words[..25]);
        words[21..25].fill(0);
        words[25..29].copy_from_slice(&identity_words(*self.binding.digest.as_bytes()));
        words[37] = self.statement_bytes as u64;
        words[38] = u64::from(self.members.iter().any(|member| member.program.is_some()));
        words[39] = u64::from(selected.program.is_some());
        words[42] = member as u64 + 1;
        words[43] = self.members.len() as u64;
        for (table, width) in ROW_WIDTHS.into_iter().enumerate() {
            let count_word = 2 * table + 1;
            let (count, begin, end) = self.category_range(table, member)?;
            words[2 * table] = words.len() as u64;
            words[count_word] = count;
            words[29 + 2 * table] = begin;
            words[30 + 2 * table] = end;
            // One intrinsic NULL, followed by every ordered occurrence roster.
            words.extend(std::iter::repeat_n(0, width));
            let mut payload_offset = 0u64;
            for context in &self.members {
                let start = context.words[2 * table] as usize;
                for category in 1..context.words[count_word] as usize {
                    let row = words.len();
                    words.extend_from_slice(
                        &context.words[start + width * category..start + width * (category + 1)],
                    );
                    if table == 2 || table == 1 && words[row + 2] != 0 {
                        words[row] = words[row]
                            .checked_add(payload_offset)
                            .ok_or(SemanticTransitionError::GenerationExhausted)?;
                    }
                }
                payload_offset = (context.words.len() as u64)
                    .checked_mul(size_of::<u64>() as u64)
                    .and_then(|length| length.checked_sub(context.words[8]))
                    .and_then(|length| payload_offset.checked_add(length))
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
            }
        }
        words[8] = (words.len() * 8) as u64;
        for context in &self.members {
            words.extend_from_slice(&context.words[context.words[8] as usize / 8..]);
        }
        words[40] = (words.len() * 8) as u64;
        words[41] = (self.frame.len() * 8) as u64;
        words.extend_from_slice(&self.frame);
        words[9] = (words.len() * 8) as u64 - words[8];
        let mut hash = Sha256::new();
        hash.update(b"xlog.semantic.model-context-resolution.v1\0");
        hash.update(original_binding.digest.as_bytes());
        for &word in &words {
            hash.update(word.to_le_bytes());
        }
        words[21..25].copy_from_slice(&identity_words(hash.finalize().into()));
        Ok(words)
    }
}

impl ActionCodebooks {
    pub(super) fn bind_admission_words(
        admission: &SemanticAdmission,
        owner: u64,
        program: Option<Identity256>,
        words: &mut [u64],
    ) -> Result<SemanticCatalogueBinding, SemanticTransitionError> {
        if words.len() < HEADER_WORDS
            || words[10..14] != identity_words(*admission.schema_generation().as_bytes())
        {
            return Err(publication_input_error(
                "action catalogue changes its original schema",
            ));
        }
        words[14] = owner;
        words[15] = admission.base().slot() as u64;
        words[16] = admission.base().generation();
        words[17..21].copy_from_slice(&identity_words(
            *admission.base_snapshot().digest().as_bytes(),
        ));
        words[21..25].fill(0);
        let mut hash = Sha256::new();
        hash.update(b"xlog.semantic.action-binding.v5\0");
        hash.update(CATALOGUE_DIGEST);
        hash.update(admission.identity().as_bytes());
        if let Some(identity) = program {
            hash.update(identity.as_bytes());
        }
        for &word in words.iter() {
            hash.update(word.to_le_bytes());
        }
        let digest = Identity256::from_bytes(hash.finalize().into());
        words[21..25].copy_from_slice(&identity_words(*digest.as_bytes()));
        Ok(SemanticCatalogueBinding {
            generation: CATALOGUE_GENERATION,
            digest,
        })
    }

    /// Rebind runtime root coordinates without rebuilding immutable meanings.
    /// The original projection reservation already encloses the retained old
    /// words, one local recipe, its replacement projection and both component
    /// rosters. The restore caller owns the unprojected and pinned-copy claim.
    pub(super) fn rebind_restored_root(
        &mut self,
        admission: &SemanticAdmission,
        owner: u64,
        program: Option<&crate::SemanticProgramAdmission>,
    ) -> Result<(), SemanticTransitionError> {
        let program_binding = program.map(Self::program_binding);
        if program_binding != self.program_binding {
            return Err(publication_input_error(
                "restored catalogue changes its original program",
            ));
        }
        let original = match &self.model_catalogue {
            Some((catalogue, member)) => {
                &catalogue
                    .members
                    .get(*member)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
                    .words
            }
            None => &self.words,
        };
        let mut words = Vec::new();
        words
            .try_reserve_exact(original.len())
            .map_err(|error| runtime_error("restored catalogue allocation", error))?;
        words.extend_from_slice(original);
        let local_binding =
            Self::bind_admission_words(admission, owner, program_binding, &mut words)?;
        if let Some((catalogue, member)) = &self.model_catalogue {
            words = catalogue.project(&words, program_binding, local_binding, *member)?;
        }
        let (components, input_cells) = Self::component_geometry(&words, program.is_some())?;
        if words.len() != self.words.len() || input_cells != self.input_cells {
            return Err(publication_input_error(
                "restored catalogue changes its allocated geometry",
            ));
        }
        self.binding.digest = Identity256::from_bytes(words_identity(&words, 21));
        self.words = words;
        self.components = components.into_iter().collect();
        Ok(())
    }

    pub(super) fn project_model_catalogue(
        &mut self,
        catalogue: Arc<FrozenModelCatalogue>,
        member: usize,
    ) -> Result<(), SemanticTransitionError> {
        let frame_bound = catalogue
            .frame
            .len()
            .checked_add(HEADER_WORDS)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let word_bound = catalogue
            .members
            .iter()
            .try_fold(frame_bound, |bound, member| {
                bound.checked_add(member.words.len())
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let bytes = word_bound
            .checked_mul(3 * size_of::<u64>())
            .and_then(|bytes| {
                bytes.checked_add(2 * COMPONENT_COUNT * size_of::<SemanticComponent>())
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = catalogue
            ._reservation
            .budget
            .reserve_auxiliary(bytes as u64)?;
        let words = catalogue.project(&self.words, self.program_binding, self.binding, member)?;
        let (components, cells) = Self::component_geometry(&words, self.program_binding.is_some())?;
        self.binding.digest = Identity256::from_bytes(words_identity(&words, 21));
        self.words = words;
        self.components = components.into_iter().collect();
        self.input_cells = cells;
        self.model_catalogue = Some((catalogue, member));
        self.model_projection_reservation = Some(Arc::new(reservation));
        Ok(())
    }
}

fn words_identity(words: &[u64], begin: usize) -> [u8; 32] {
    let mut identity = [0; 32];
    for (bytes, word) in identity.chunks_exact_mut(8).zip(&words[begin..begin + 4]) {
        bytes.copy_from_slice(&word.to_le_bytes());
    }
    identity
}

struct ContextBuffers {
    words: Vec<u64>,
    components: Vec<SemanticComponent>,
    cells: usize,
    logits: TrackedCudaSlice<f32>,
    support: TrackedCudaSlice<u8>,
    codebooks: TrackedCudaSlice<u64>,
    device_components: TrackedCudaSlice<SemanticComponent>,
}

impl ContextBuffers {
    fn allocate(
        session: &SemanticTransitionSession,
        catalogue: &FrozenModelCatalogue,
        member: usize,
    ) -> Result<Self, SemanticTransitionError> {
        let words = catalogue.project(
            &session.codebooks.words,
            session.codebooks.program_binding,
            session.codebooks.binding,
            member,
        )?;
        Self::allocate_words(session, words, session.codebooks.program_binding.is_some())
    }

    fn allocate_words(
        session: &SemanticTransitionSession,
        words: Vec<u64>,
        has_program: bool,
    ) -> Result<Self, SemanticTransitionError> {
        let (components, cells) = ActionCodebooks::component_geometry(&words, has_program)?;
        let components: Vec<_> = components.into_iter().collect();
        let bytes = cells
            .checked_mul(size_of::<f32>() + size_of::<u8>())
            .and_then(|bytes| {
                words
                    .len()
                    .checked_mul(8)
                    .and_then(|words| bytes.checked_add(words))
            })
            .and_then(|bytes| bytes.checked_add(COMPONENT_COUNT * size_of::<SemanticComponent>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut reservation = session
            .provider
            .memory()
            .reserve_bytes(bytes as u64)
            .map_err(|error| runtime_error("model context allocation", error))?;
        Ok(Self {
            logits: reservation
                .alloc::<f32>(cells)
                .map_err(|error| runtime_error("model context logits", error))?,
            support: reservation
                .alloc::<u8>(cells)
                .map_err(|error| runtime_error("model context support", error))?,
            codebooks: reservation
                .alloc::<u64>(words.len())
                .map_err(|error| runtime_error("model context codebooks", error))?,
            device_components: reservation
                .alloc::<SemanticComponent>(COMPONENT_COUNT)
                .map_err(|error| runtime_error("model context components", error))?,
            words,
            components,
            cells,
        })
    }

    fn install(
        self,
        session: &mut SemanticTransitionSession,
        catalogue: Arc<FrozenModelCatalogue>,
        member: usize,
    ) {
        session.codebooks.model_catalogue = Some((catalogue, member));
        self.install_storage(session);
        session.codebooks.model_projection_reservation = None;
    }

    fn install_storage(self, session: &mut SemanticTransitionSession) {
        session.codebooks.binding.digest = Identity256::from_bytes(words_identity(&self.words, 21));
        session.codebooks.words = self.words;
        session.codebooks.components = self.components;
        session.codebooks.input_cells = self.cells;
        session.logits = self.logits;
        session.support = self.support;
        session.device_codebooks = self.codebooks;
        session.device_components = self.device_components;
    }
}

/// Replace the descriptor buffers of the same unimported cold Session. The
/// caller retains the original admission/program composition across Unknown.
pub(super) fn install_cold_action_codebooks(
    session: &mut SemanticTransitionSession,
    mut codebooks: ActionCodebooks,
) -> Result<(), SemanticTransitionError> {
    let bytes = codebooks
        .words
        .len()
        .checked_mul(2 * size_of::<u64>())
        .and_then(|bytes| {
            bytes.checked_add(
                2 * COMPONENT_COUNT * size_of::<SemanticComponent>()
                    + size_of::<ContextBuffers>()
                    + size_of::<OriginalSessionCodebooks>()
                    + state_restoration::OriginalDeviceWrite::metadata_allocation_bytes::<u64>()
                    + state_restoration::OriginalDeviceWrite::metadata_allocation_bytes::<
                        SemanticComponent,
                    >()
                    + 8 * size_of::<usize>(),
            )
        })
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    let reservation = Arc::new(
        session
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .reserve_auxiliary(bytes as u64)?,
    );
    let buffers = ContextBuffers::allocate_words(
        session,
        std::mem::take(&mut codebooks.words),
        codebooks.program_binding.is_some(),
    )?;
    let mut writes = Vec::new();
    writes
        .try_reserve_exact(2)
        .map_err(|error| runtime_error("cold context upload roster allocation", error))?;
    writes.push(state_restoration::OriginalDeviceWrite::new_with_owner(
        &session.stream,
        &buffers.words,
        buffers.codebooks.view(),
        true,
        Some(Arc::clone(&reservation)),
    )?);
    writes.push(state_restoration::OriginalDeviceWrite::new_with_owner(
        &session.stream,
        &buffers.components,
        buffers.device_components.view(),
        true,
        Some(Arc::clone(&reservation)),
    )?);
    let original = Arc::new(Mutex::new(OriginalSessionCodebooks {
        writes,
        cursor: 0,
        poisoned: false,
        complete: false,
        retired: false,
        _model_catalogue: None,
    }));
    codebooks.model_projection_reservation = Some(reservation);
    session.codebooks = codebooks;
    buffers.install_storage(session);
    session.construction_codebooks = Some(original);
    session.construction_pending = Arc::new(AtomicBool::new(true));
    session.resolve_constructor_codebooks(true)
}

impl SemanticTransitionSession {
    pub(super) fn has_pending_model_context_preseal(&self) -> bool {
        self.model_context_preseal
            .as_ref()
            .is_some_and(|preseal| preseal.pending.load(Ordering::Acquire))
    }

    pub(super) fn require_original_model_context_preseal(
        &self,
        source: &Self,
    ) -> Result<Arc<ModelContextPreseal>, SemanticTransitionError> {
        let preseal = self
            .model_context_preseal
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !preseal.pending.load(Ordering::Acquire)
            || source
                .model_context_preseal
                .as_ref()
                .is_none_or(|original| !Arc::ptr_eq(preseal, original))
            || preseal
                .issuers
                .first()
                .is_none_or(|issuer| !Arc::ptr_eq(issuer, &source.publication_issuer))
            || !preseal
                .issuers
                .iter()
                .skip(2)
                .any(|issuer| Arc::ptr_eq(issuer, &self.publication_issuer))
        {
            return Err(publication_input_error(
                "evaluation context requires its original ordered cold roster",
            ));
        }
        Ok(Arc::clone(preseal))
    }

    fn require_cold_model_contexts(
        contexts: &[&mut Self],
    ) -> Result<SemanticReplayMaterialBudget, SemanticTransitionError> {
        for context in contexts {
            if let Some(preseal) = &context.model_context_preseal {
                context.ensure_rebindable_except_model_context_preseal(preseal)?;
            } else {
                context.ensure_rebindable()?;
            }
            #[cfg(feature = "semantic-policy")]
            if context.policy.is_some()
                || context.pending_policy.is_some()
                || context.prepared_policy.is_some()
            {
                return Err(publication_input_error(
                    "model context domain cannot change an allocated policy owner",
                ));
            }
            if context.cold_content.is_none()
                || context.cold_statement_records.is_empty()
                || context.task.is_some()
                || context.publication.is_some()
                || context.captured.is_some()
                || context.codebooks.model_catalogue.is_some()
                || !context.model_owners.is_empty()
                || context.text_binding.is_some()
                || context.next_proposal != 0
            {
                return Err(publication_input_error("model context domain must freeze once from the complete original cold roster before model use"));
            }
        }
        let first = &*contexts[0];
        let budget = first
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        for context in contexts.iter().skip(1) {
            if !Arc::ptr_eq(first.provider.memory(), context.provider.memory())
                || first.domain.stream_id() != context.domain.stream_id()
            {
                return Err(publication_input_error(
                    "model contexts require the original allocator and serial stream",
                ));
            }
            if context
                .replay_material_budget
                .as_ref()
                .is_none_or(|other| !budget.same_pool(other))
            {
                return Err(publication_input_error(
                    "model contexts require one original host material capacity pool",
                ));
            }
        }
        Ok(budget.clone())
    }

    pub fn begin_model_contexts(
        contexts: &mut [&mut Self],
    ) -> Result<bool, SemanticTransitionError> {
        if contexts.len() < 2 {
            return Err(publication_input_error(
                "model contexts require an ordered source, target and optional evaluation roster",
            ));
        }
        for (member, context) in contexts.iter().enumerate() {
            if contexts[..member]
                .iter()
                .any(|prior| Arc::ptr_eq(&prior.publication_issuer, &context.publication_issuer))
            {
                return Err(publication_input_error(
                    "model contexts require distinct original cold owners",
                ));
            }
        }
        if contexts
            .iter()
            .any(|context| context.model_context_preseal.is_some())
        {
            let preseal = contexts[0].model_context_preseal.as_ref().ok_or_else(|| {
                publication_input_error("model contexts cannot replace an original cold owner")
            })?;
            if preseal.issuers.len() != contexts.len()
                || contexts.iter().enumerate().any(|(member, context)| {
                    !Arc::ptr_eq(&preseal.issuers[member], &context.publication_issuer)
                        || context
                            .model_context_preseal
                            .as_ref()
                            .is_none_or(|other| !Arc::ptr_eq(preseal, other))
                })
            {
                return Err(publication_input_error(
                    "model contexts must retain every original owner in its frozen order",
                ));
            }
            return Ok(contexts
                .iter()
                .any(|context| context.codebooks.model_catalogue.is_some()));
        }
        let budget = Self::require_cold_model_contexts(contexts)?;
        let bytes = contexts
            .len()
            .checked_mul(size_of::<Arc<()>>())
            .and_then(|bytes| {
                bytes.checked_add(size_of::<ModelContextPreseal>() + 2 * size_of::<usize>())
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = budget.reserve_auxiliary(bytes as u64)?;
        let mut issuers = Vec::new();
        issuers
            .try_reserve_exact(contexts.len())
            .map_err(|error| runtime_error("model context original roster allocation", error))?;
        issuers.extend(
            contexts
                .iter()
                .map(|context| Arc::clone(&context.publication_issuer)),
        );
        let preseal = Arc::new(ModelContextPreseal {
            issuers,
            pending: AtomicBool::new(true),
            _reservation: reservation,
        });
        for context in contexts {
            context.model_context_preseal = Some(Arc::clone(&preseal));
        }
        Ok(false)
    }

    pub fn frozen_model_context(
        &self,
    ) -> Result<Option<SemanticFrozenModelContext>, SemanticTransitionError> {
        self.ensure_quiescent()?;
        Ok(self
            .codebooks
            .model_catalogue
            .as_ref()
            .map(|(catalogue, member)| SemanticFrozenModelContext {
                catalogue: Arc::clone(catalogue),
                member: *member,
            }))
    }

    /// Freeze the complete ordered cold roster before task import, model
    /// allocation, capture or drawing. Unknown retains one shared DMA operation.
    pub fn freeze_model_contexts(
        contexts: &mut [&mut Self],
    ) -> Result<SemanticCatalogueBinding, SemanticTransitionError> {
        Self::begin_model_contexts(contexts)?;
        if contexts
            .iter()
            .any(|context| context.codebooks.model_catalogue.is_some())
        {
            let (catalogue, _) =
                contexts[0]
                    .codebooks
                    .model_catalogue
                    .as_ref()
                    .ok_or_else(|| {
                        publication_input_error(
                            "model contexts cannot replace an original cold owner",
                        )
                    })?;
            if catalogue.members.len() != contexts.len()
                || contexts.iter().enumerate().any(|(member, context)| {
                    context
                        .codebooks
                        .model_catalogue
                        .as_ref()
                        .is_none_or(|(other, ordinal)| {
                            !Arc::ptr_eq(catalogue, other)
                                || *ordinal != member
                                || !Arc::ptr_eq(
                                    &catalogue.members[member].issuer,
                                    &context.publication_issuer,
                                )
                        })
                        || !Arc::ptr_eq(
                            &contexts[0].construction_pending,
                            &context.construction_pending,
                        )
                        || match (
                            &contexts[0].construction_codebooks,
                            &context.construction_codebooks,
                        ) {
                            (Some(first), Some(other)) => !Arc::ptr_eq(first, other),
                            _ => true,
                        }
                })
            {
                return Err(publication_input_error(
                    "model contexts must retain every original owner in its frozen order",
                ));
            }
            let binding = catalogue.binding;
            contexts[0].resolve_constructor_codebooks(true)?;
            contexts[0]
                .model_context_preseal
                .as_ref()
                .expect("original cold roster")
                .pending
                .store(false, Ordering::Release);
            return Ok(binding);
        }
        let budget = Self::require_cold_model_contexts(contexts)?;
        for context in contexts.iter().skip(2) {
            context.require_prepared_evaluation_model_context(contexts[0])?;
        }
        for count_word in [1, 3, 5, 7] {
            contexts.iter().try_fold(1u64, |total, context| {
                context.codebooks.words[count_word]
                    .checked_sub(1)
                    .and_then(|count| total.checked_add(count))
                    .filter(|&count| count <= 65536)
                    .ok_or_else(|| {
                        publication_input_error("immutable model field exceeds 65536 categories")
                    })
            })?;
        }
        let word_count = contexts
            .iter()
            .try_fold(0usize, |count, context| {
                count.checked_add(context.codebooks.words.len())
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let frame_words = contexts
            .len()
            .checked_mul(18)
            .and_then(|count| count.checked_add(4))
            .and_then(|count| count.checked_add(word_count))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let projection_words = frame_words
            .checked_add(word_count)
            .and_then(|count| count.checked_add(HEADER_WORDS))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        // Every recipe, frame, projection and overlapping pinned upload is
        // charged before allocation. The common learned banks are not copied.
        let per_context_bytes = projection_words
            .checked_mul(2 * size_of::<u64>())
            .and_then(|bytes| {
                bytes.checked_add(
                    2 * COMPONENT_COUNT * size_of::<SemanticComponent>()
                        + size_of::<CatalogueMember>()
                        + size_of::<ContextBuffers>()
                        + state_restoration::OriginalDeviceWrite::metadata_allocation_bytes::<u64>(
                        )
                        + state_restoration::OriginalDeviceWrite::metadata_allocation_bytes::<
                            SemanticComponent,
                        >(),
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let bytes = word_count
            .checked_add(frame_words)
            .and_then(|count| count.checked_mul(size_of::<u64>()))
            .and_then(|bytes| {
                contexts
                    .len()
                    .checked_mul(per_context_bytes)
                    .and_then(|contexts| bytes.checked_add(contexts))
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    size_of::<FrozenModelCatalogue>()
                        + size_of::<OriginalSessionCodebooks>()
                        + 8 * size_of::<usize>(),
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = budget.reserve_auxiliary(bytes as u64)?;
        // Reserve the canonical encoder's largest temporary before copying any
        // learned payload. Repeated qualifier occurrences remain real bytes.
        let mut payload_bound = 0usize;
        for context in contexts.iter() {
            let admission = context
                .graph
                .admission()
                .ok_or(SemanticTransitionError::NotBound)?;
            for &record in &context.cold_statement_records {
                let statement = admission
                    .records()
                    .records
                    .get(record as usize)
                    .ok_or(SemanticTransitionError::InvalidTaskBank)?;
                let mut bound = 4usize;
                for index in std::iter::once(record).chain(statement.qualifiers.iter().copied()) {
                    let atom = admission
                        .encoded_records
                        .get(index as usize)
                        .ok_or(SemanticTransitionError::InvalidTaskBank)?;
                    bound = bound
                        .checked_add(8)
                        .and_then(|bound| bound.checked_add(atom.bytes.len()))
                        .ok_or(SemanticTransitionError::GenerationExhausted)?;
                }
                payload_bound = payload_bound.max(bound);
            }
        }
        let payload_reservation = budget.reserve_auxiliary(
            payload_bound
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(size_of::<Vec<u8>>()))
                .ok_or(SemanticTransitionError::GenerationExhausted)? as u64,
        )?;
        let mut statement_bytes = 0usize;
        for context in contexts.iter() {
            let admission = context
                .graph
                .admission()
                .ok_or(SemanticTransitionError::NotBound)?;
            for &record in &context.cold_statement_records {
                let payload = feedback_statement_payload(
                    admission.records(),
                    &admission.encoded_records,
                    record,
                )?;
                statement_bytes = statement_bytes.max(payload.len());
            }
        }
        SemanticFeedbackSchema::from_statement_lengths([statement_bytes])?;
        drop(payload_reservation);
        let catalogue = FrozenModelCatalogue::new(contexts, statement_bytes, reservation)?;
        let mut buffers = Vec::new();
        buffers
            .try_reserve_exact(contexts.len())
            .map_err(|error| runtime_error("model context buffer roster allocation", error))?;
        for (member, context) in contexts.iter().enumerate() {
            buffers.push(ContextBuffers::allocate(context, &catalogue, member)?);
        }
        let write_count = contexts
            .len()
            .checked_mul(2)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut writes = Vec::new();
        writes
            .try_reserve_exact(write_count)
            .map_err(|error| runtime_error("model context upload roster allocation", error))?;
        for buffers in &buffers {
            writes.push(state_restoration::OriginalDeviceWrite::new_with_owner(
                &contexts[0].stream,
                &buffers.words,
                buffers.codebooks.view(),
                true,
                Some(Arc::clone(&catalogue._reservation)),
            )?);
            writes.push(state_restoration::OriginalDeviceWrite::new_with_owner(
                &contexts[0].stream,
                &buffers.components,
                buffers.device_components.view(),
                true,
                Some(Arc::clone(&catalogue._reservation)),
            )?);
        }
        let original = Arc::new(Mutex::new(OriginalSessionCodebooks {
            writes,
            cursor: 0,
            poisoned: false,
            complete: false,
            retired: false,
            _model_catalogue: Some(Arc::clone(&catalogue)),
        }));
        let pending = Arc::new(AtomicBool::new(true));
        let binding = catalogue.binding;
        // Install every actual owner before the first effect. All Sessions stay
        // unusable until the one shared original upload ledger is complete.
        for (member, (context, buffers)) in contexts.iter_mut().zip(buffers).enumerate() {
            buffers.install(context, Arc::clone(&catalogue), member);
            context.construction_codebooks = Some(Arc::clone(&original));
            context.construction_pending = Arc::clone(&pending);
        }
        contexts[0].resolve_constructor_codebooks(true)?;
        contexts[0]
            .model_context_preseal
            .as_ref()
            .expect("original cold roster")
            .pending
            .store(false, Ordering::Release);
        Ok(binding)
    }

    /// Whether this exact ordered roster can resolve an entered original upload
    /// with unknown completion. Known refusal cannot create a retry loop merely
    /// because installation retained the original catalogue/allocations.
    pub fn model_contexts_pending(contexts: &[&Self]) -> bool {
        let Some(first) = contexts.first() else {
            return false;
        };
        let Some(preseal) = &first.model_context_preseal else {
            return false;
        };
        if !preseal.requires_roster(contexts)
            || contexts.iter().any(|context| {
                context
                    .model_context_preseal
                    .as_ref()
                    .is_none_or(|other| !Arc::ptr_eq(preseal, other))
            })
            || !preseal.pending.load(Ordering::Acquire)
        {
            return false;
        }
        if first.codebooks.model_catalogue.is_none() {
            return contexts.iter().any(|context| {
                context.construction_pending.load(Ordering::Acquire)
                    && context
                        .construction_codebooks
                        .as_ref()
                        .is_some_and(|owner| {
                            owner.lock().is_ok_and(|original| {
                                !original.complete
                                    && !original.retired
                                    && original
                                        .writes
                                        .get(original.cursor)
                                        .is_some_and(|write| write.completion_pending())
                            })
                        })
            });
        }
        let Some((catalogue, 0)) = &first.codebooks.model_catalogue else {
            return false;
        };
        catalogue.members.len() == contexts.len()
            && contexts.iter().enumerate().all(|(member, context)| {
                matches!(&context.codebooks.model_catalogue,
                    Some((other, ordinal)) if Arc::ptr_eq(catalogue, other) && *ordinal == member)
                    && Arc::ptr_eq(
                        &catalogue.members[member].issuer,
                        &context.publication_issuer,
                    )
                    && Arc::ptr_eq(&first.construction_pending, &context.construction_pending)
                    && match (
                        &first.construction_codebooks,
                        &context.construction_codebooks,
                    ) {
                        (Some(first), Some(other)) => Arc::ptr_eq(first, other),
                        _ => false,
                    }
            })
            && first.construction_pending.load(Ordering::Acquire)
            && first.construction_codebooks.as_ref().is_some_and(|owner| {
                owner.lock().is_ok_and(|original| {
                    !original.complete
                        && !original.retired
                        && original
                            .writes
                            .get(original.cursor)
                            .is_some_and(|write| write.completion_pending())
                })
            })
    }

    /// Model row identity is distinct from this context's authenticated bytes.
    pub fn model_catalogue_binding(&self) -> SemanticCatalogueBinding {
        self.codebooks
            .model_catalogue
            .as_ref()
            .map_or(self.codebooks.binding, |(catalogue, _)| catalogue.binding)
    }

    pub fn feedback_schema(&self) -> Result<SemanticFeedbackSchema, SemanticTransitionError> {
        let mut schema = SemanticFeedbackSchema::new(self.feedback_statement_bytes()?)?;
        if let Some((catalogue, _)) = &self.codebooks.model_catalogue {
            if schema.statement_bytes > catalogue.statement_bytes {
                return Err(publication_input_error(
                    "task feedback exceeds its original immutable model domain",
                ));
            }
            schema = SemanticFeedbackSchema::from_statement_lengths([catalogue.statement_bytes])?;
        }
        Ok(schema)
    }
}
