//! Frozen original cohorts for independent learning trajectories.
use super::*;
use serde_json::{json, Value};
use std::sync::OnceLock;

const MAGIC: &[u8] = b"XLOG-FROZEN-EVALUATION-COHORT\0";
const PORTS: usize = SemanticTrainingViewPort::ALL.len();

pub(super) struct FrozenFamilyCohort {
    pub(super) coordinates: [u64; 5],
    pub(super) saved: OnceLock<Arc<[u8]>>,
    preparation: Mutex<CohortPreparation>,
}

#[derive(Default)]
struct CohortPreparation {
    live: Option<Arc<SemanticEvaluationCohort>>,
    selection: Option<(DeviceMemoryView<u8>, [OriginalNativeCommand; 2])>,
    selected: bool,
    next_port: usize,
    encoding: Vec<u8>,
    read: Option<PublicationRead<u8>>,
    write: Option<state_restoration::OriginalDeviceWrite>,
    reservation: Option<Arc<SemanticReplayMaterialReservation>>,
}

struct SavedCohort<'a> {
    coordinates: [u64; 5],
    domain: Identity256,
    task: SemanticTaskContentIdentity,
    prefix: Identity256,
    initial_count: u64,
    archive_count: u64,
    eligible_count: u64,
    chain: Identity256,
    ports: [&'a [u8]; PORTS],
    digest: Identity256,
}

fn word(reader: &mut SemanticMaterialReader<'_>) -> Result<u64, SemanticTransitionError> {
    reader.u64().map_err(SemanticTransitionError::Semantic)
}

fn identity(
    reader: &mut SemanticMaterialReader<'_>,
) -> Result<Identity256, SemanticTransitionError> {
    Ok(Identity256::from_bytes(
        reader
            .take(32)
            .map_err(SemanticTransitionError::Semantic)?
            .try_into()
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
    ))
}

impl FrozenFamilyCohort {
    pub(super) fn pending(coordinates: [u64; 5]) -> Self {
        Self {
            coordinates,
            saved: OnceLock::new(),
            preparation: Mutex::new(CohortPreparation::default()),
        }
    }

    pub(super) fn restored(
        coordinates: [u64; 5],
        material: Option<Arc<[u8]>>,
    ) -> Result<Self, SemanticTransitionError> {
        let result = Self::pending(coordinates);
        if let Some(bytes) = material {
            let saved = decode(&bytes)?;
            if saved.coordinates != coordinates {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            result
                .saved
                .set(bytes)
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        }
        Ok(result)
    }

    pub(super) fn live(&self) -> Result<Arc<SemanticEvaluationCohort>, SemanticTransitionError> {
        if self.saved.get().is_none() {
            return Err(publication_input_error(
                "retain the original unfinished cohort preparation",
            ));
        }
        let original = self
            .preparation
            .lock()
            .map_err(|_| publication_input_error("original cohort owner is poisoned"))?;
        if !original.selected || original.next_port != PORTS {
            return Err(publication_input_error(
                "retain the original unfinished cohort restoration",
            ));
        }
        original.live.as_ref().map(Arc::clone).ok_or_else(|| {
            publication_input_error(
                "restore the saved cohort through its original cold preparation",
            )
        })
    }

    pub(super) fn hydration_dma_ceiling(
        &self,
    ) -> Result<(Identity256, [u64; 9]), SemanticTransitionError> {
        let bytes = self.saved.get().ok_or_else(|| {
            publication_input_error("freeze the original cohort before cold admission")
        })?;
        let saved = decode(bytes)?;
        let mut dma = [0; 9];
        dma[0] = PORTS as u64;
        dma[8] = saved.ports.iter().try_fold(0u64, |total, port| {
            total
                .checked_add(
                    u64::try_from(port.len())
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                )
                .ok_or(SemanticTransitionError::GenerationExhausted)
        })?;
        Ok((saved.digest, dma))
    }

    pub(super) fn selection_material(&self) -> Result<&[u8], SemanticTransitionError> {
        let bytes = self
            .saved
            .get()
            .ok_or_else(|| publication_input_error("cohort preparation is unfinished"))?;
        Ok(decode(bytes)?.ports[0])
    }

    pub(super) fn projection(&self) -> Result<Value, SemanticTransitionError> {
        let bytes = self
            .saved
            .get()
            .ok_or_else(|| publication_input_error("cohort preparation is unfinished"))?;
        let saved = decode(bytes)?;
        let selection = selection(saved.ports[0])?;
        let hex = |id: Identity256| {
            id.as_bytes()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        };
        Ok(json!({
            "cursor": saved.coordinates[0], "rng": &saved.coordinates[1..],
            "source_sha256": hex(saved.digest), "training_domain": hex(saved.domain),
            "task": {"query": hex(saved.task.query), "theory_program": hex(saved.task.theory_program), "result": hex(saved.task.result)},
            "original_source_prefix": hex(saved.prefix), "initial_count": saved.initial_count, "archive_count": saved.archive_count,
            "eligible_count": saved.eligible_count, "archive_chain_head": hex(saved.chain),
            "selection_identity": selection.identity, "source_identity": selection.source_identity,
            "content_identity": selection.content_identity, "row_count": selection.row_count,
            "selection_record": saved.ports[0],
        }))
    }
}

fn selection(bytes: &[u8]) -> Result<SemanticTrainingViewSelection, SemanticTransitionError> {
    if bytes.len() != size_of::<SemanticTrainingViewSelection>() {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    // SAFETY: this exact dense native ABI contains only integer words, arrays
    // and embedded records with the same unrestricted bit representations.
    let value =
        unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<SemanticTrainingViewSelection>()) };
    if value.status != 0 || value.row_count == 0 {
        return Err(publication_input_error(
            "original cohort selection did not complete successfully",
        ));
    }
    Ok(value)
}

fn decode(bytes: &[u8]) -> Result<SavedCohort<'_>, SemanticTransitionError> {
    let body = bytes
        .len()
        .checked_sub(32)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let mut reader = SemanticMaterialReader::new(&bytes[..body]);
    if reader
        .take(MAGIC.len())
        .map_err(SemanticTransitionError::Semantic)?
        != MAGIC
        || word(&mut reader)? != 1
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let mut coordinates = [0; 5];
    for value in &mut coordinates {
        *value = word(&mut reader)?;
    }
    let domain = identity(&mut reader)?;
    let task = SemanticTaskContentIdentity {
        query: identity(&mut reader)?,
        theory_program: identity(&mut reader)?,
        result: identity(&mut reader)?,
    };
    let prefix = identity(&mut reader)?;
    let initial_count = word(&mut reader)?;
    let archive_count = word(&mut reader)?;
    let eligible_count = word(&mut reader)?;
    let chain = identity(&mut reader)?;
    if eligible_count > archive_count || word(&mut reader)? != PORTS as u64 {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let mut ports = [&[][..]; PORTS];
    for port in &mut ports {
        let length = usize::try_from(word(&mut reader)?)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        *port = reader
            .take(length)
            .map_err(SemanticTransitionError::Semantic)?;
    }
    reader.finish().map_err(SemanticTransitionError::Semantic)?;
    let digest = Identity256::from_bytes(Sha256::digest(&bytes[..body]).into());
    if digest.as_bytes() != &bytes[body..] {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let selected = selection(ports[0])?;
    if selected.ordinal != coordinates[0]
        || initial_count.checked_add(eligible_count) != Some(selected.row_count)
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    Ok(SavedCohort {
        coordinates,
        domain,
        task,
        prefix,
        initial_count,
        archive_count,
        eligible_count,
        chain,
        ports,
        digest,
    })
}

fn encoding_extent(cohort: &SemanticEvaluationCohort) -> Result<usize, SemanticTransitionError> {
    let header = MAGIC.len() + (1 + 5 + 3 + 1) * size_of::<u64>() + (1 + 3 + 1 + 1) * 32;
    SemanticTrainingViewPort::ALL.into_iter().try_fold(
        header + PORTS * size_of::<u64>() + 32,
        |bytes, port| {
            bytes
                .checked_add(cohort.selected.port(port)?.0.len())
                .ok_or(SemanticTransitionError::GenerationExhausted)
        },
    )
}

impl SemanticTransitionSession {
    /// Freeze only the original Program's supplied coordinates, before learning.
    /// The original admitted cold report owns every selection, transfer and wait.
    pub fn prepare_family_cohorts(
        &mut self,
        parent: &SemanticPublishedLease,
        admission: &SemanticSegmentInstructionAdmission,
        work: &SemanticColdModelWork,
    ) -> Result<Option<Value>, SemanticTransitionError> {
        self.require_admitted_cold_model_work(admission, parent, work)?;
        let current = self.instruction_family_cohort_coordinates(admission)?;
        let native_work = self.cold_native_work(parent.token)?.ok_or_else(|| {
            publication_input_error("cohort preparation requires its original cold work scope")
        })?;
        self.prepare_original_family_cohorts(parent, current, native_work)
    }

    pub(super) fn prepare_original_family_cohorts(
        &mut self,
        parent: &SemanticPublishedLease,
        current: Option<[u64; 5]>,
        native_work: DeviceMemoryView<u64>,
    ) -> Result<Option<Value>, SemanticTransitionError> {
        self.checked_reader(parent)?;
        let Some(mut coordinates) = self.required_family_cohorts()? else {
            return Ok(None);
        };
        // Initial selections are independent original inputs. Keeping the
        // current member last leaves only its required data allocation live.
        coordinates.sort_by_key(|key| (Some(*key) == current, *key));
        for key in &coordinates {
            if Some(*key) != current {
                if let Some(frame) = self.original_family_cohort(*key)? {
                    let mut original = frame.preparation.lock().map_err(|_| {
                        publication_input_error("original cohort owner is poisoned")
                    })?;
                    if frame.saved.get().is_some()
                        && original.selected
                        && original.read.is_none()
                        && original.write.is_none()
                    {
                        original.live = None;
                        original.selected = false;
                        original.next_port = 0;
                    }
                }
            }
        }
        let arena = Arc::clone(
            self.training_views
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let mut results = BTreeMap::new();
        for coordinates in coordinates {
            let frame =
                match self.original_family_cohort(coordinates)? {
                    Some(frame) => frame,
                    None => {
                        self.family_cohort_freeze_anchor()?;
                        let (source, cohort) =
                            self.prepare_evaluation_cohort(parent, coordinates)?;
                        let (prefix, archive_count, eligible_count, chain) =
                            cohort.selected.frozen_source_metadata()?;
                        let encoded_bytes = encoding_extent(&cohort)?;
                        let reservation = self.reserve_cohort_material(encoded_bytes)?;
                        let frame = Arc::new(FrozenFamilyCohort::pending(coordinates));
                        {
                            let mut original = frame.preparation.lock().map_err(|_| {
                                publication_input_error("original cohort owner is poisoned")
                            })?;
                            original.reservation = Some(reservation);
                            // Allocate the complete immutable frame before its first
                            // byte. Pending reads never grow or replace this buffer.
                            original.encoding.try_reserve_exact(encoded_bytes).map_err(
                                |error| runtime_error("original cohort encoding allocation", error),
                            )?;
                            original.live = Some(cohort);
                            original.selection = Some((
                                source,
                                [
                                    OriginalNativeCommand::new(&self.domain)?,
                                    OriginalNativeCommand::new(&self.domain)?,
                                ],
                            ));
                            original.encoding.extend_from_slice(MAGIC);
                            material_u64(&mut original.encoding, 1);
                            for value in coordinates {
                                material_u64(&mut original.encoding, value);
                            }
                            original
                                .encoding
                                .extend_from_slice(arena.training_domain_identity().as_bytes());
                            let task = original
                                .live
                                .as_ref()
                                .expect("original cohort")
                                .source_task();
                            for identity in [task.query, task.theory_program, task.result] {
                                original.encoding.extend_from_slice(identity.as_bytes());
                            }
                            original.encoding.extend_from_slice(prefix.as_bytes());
                            material_u64(&mut original.encoding, arena.original_source_count());
                            material_u64(&mut original.encoding, archive_count);
                            material_u64(&mut original.encoding, eligible_count);
                            original.encoding.extend_from_slice(chain.as_bytes());
                            material_u64(&mut original.encoding, PORTS as u64);
                        }
                        self.install_original_family_cohort(Arc::clone(&frame))?;
                        frame
                    }
                };
            let mut original = frame
                .preparation
                .lock()
                .map_err(|_| publication_input_error("original cohort owner is poisoned"))?;
            if original.reservation.is_none() {
                let bytes = frame
                    .saved
                    .get()
                    .ok_or_else(|| {
                        publication_input_error("retain the original unresolved cohort preparation")
                    })?
                    .len();
                original.reservation = Some(self.reserve_cohort_material(bytes)?);
            }
            if let Some(bytes) = frame.saved.get() {
                if current != Some(coordinates) {
                    drop(original);
                    results.insert(coordinates, frame.projection()?);
                    continue;
                }
                let saved = decode(bytes)?;
                let task = self
                    .task
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?
                    .0
                    .content_identity();
                if saved.domain != arena.training_domain_identity()
                    || saved.task != task
                    || saved.prefix != arena.original_source_prefix()
                {
                    return Err(publication_input_error(
                        "saved cohort changed its complete original source binding",
                    ));
                }
                if original.live.is_none() {
                    original.selected = false;
                    original.next_port = 0;
                    let selected = Arc::new(arena.prepare_selection(None)?);
                    original.live = Some(self.evaluation_cohort_from_selected(parent, selected)?);
                }
                let cohort = Arc::clone(original.live.as_ref().expect("retained restored cohort"));
                for (index, port) in SemanticTrainingViewPort::ALL.into_iter().enumerate() {
                    if cohort.selected.port(port)?.0.len() != saved.ports[index].len() {
                        return Err(SemanticTransitionError::ObservationMismatch);
                    }
                }
                while original.next_port < PORTS {
                    let index = original.next_port;
                    let port = SemanticTrainingViewPort::ALL[index];
                    let view = cohort.selected.port(port)?.0;
                    if view.len() != saved.ports[index].len() {
                        return Err(SemanticTransitionError::ObservationMismatch);
                    }
                    if matches!(port, SemanticTrainingViewPort::Canaries) {
                        let data = self.read_family_cohort_port(&mut original, view)?;
                        if data != saved.ports[index] {
                            return Err(SemanticTransitionError::ObservationMismatch);
                        }
                    } else {
                        if original.write.is_none() {
                            original.write =
                                Some(state_restoration::OriginalDeviceWrite::new_with_owner(
                                    &self.stream,
                                    saved.ports[index],
                                    view,
                                    true,
                                    original.reservation.clone(),
                                )?);
                        }
                        let allowance = self.graph.cold_work_allowance();
                        original
                            .write
                            .as_mut()
                            .expect("retained original cohort write")
                            .resolve_with_dma(
                                &self.domain,
                                &self.stream,
                                &self.provider,
                                &mut self.poisoned,
                                true,
                                allowance.as_ref(),
                            )?;
                        original.write = None;
                    }
                    original.next_port += 1;
                }
                original.selected = true;
            } else {
                let cohort = Arc::clone(original.live.as_ref().ok_or_else(|| {
                    publication_input_error("retain the original unresolved cohort preparation")
                })?);
                if !original.selected {
                    let (source, commands) = original.selection.as_mut().ok_or_else(|| {
                        publication_input_error(
                            "restored unresolved cohort cannot be selected again",
                        )
                    })?;
                    cohort.selected.enqueue_original(
                        source.clone(),
                        coordinates[0],
                        [
                            coordinates[1],
                            coordinates[2],
                            coordinates[3],
                            coordinates[4],
                        ],
                        Some(&native_work),
                        commands,
                        &mut self.poisoned,
                    )?;
                    original.selected = true;
                }
                while original.next_port < PORTS {
                    let index = original.next_port;
                    let view = cohort
                        .selected
                        .port(SemanticTrainingViewPort::ALL[index])?
                        .0;
                    let data = self.read_family_cohort_port(&mut original, view)?;
                    if index == 0 {
                        selection(&data)?;
                    }
                    material_u64(&mut original.encoding, data.len() as u64);
                    original.encoding.extend_from_slice(&data);
                    original.next_port += 1;
                }
                let digest: [u8; 32] = Sha256::digest(&original.encoding).into();
                original.encoding.extend_from_slice(&digest);
                let bytes: Arc<[u8]> = std::mem::take(&mut original.encoding).into();
                decode(&bytes)?;
                frame
                    .saved
                    .set(bytes)
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
                original.selection = None;
            }
            if current != Some(coordinates) {
                original.live = None;
                original.selected = false;
                original.next_port = 0;
            }
            drop(original);
            results.insert(coordinates, frame.projection()?);
        }
        Ok(Some(
            json!({"cohorts": results.into_values().collect::<Vec<_>>()}),
        ))
    }

    /// Enclose the actual original select/gather and seventeen retained reads.
    /// Input quantities come from the admitted Program and arena reservation.
    pub(super) fn family_cohort_preparation_ceiling(
        &self,
    ) -> Result<[[u64; 9]; 2], SemanticTransitionError> {
        let mut ceilings = [[0; 9]; 2];
        let Some(keys) = self.required_family_cohorts()? else {
            return Ok(ceilings);
        };
        let arena = self
            .training_views
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let mut selection =
            arena.selection_native_work_ceiling(self.training_origins.as_deref())?;
        selection[0] = selection[0]
            .checked_add(2)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut dma = [0; 9];
        dma[0] = PORTS as u64;
        let (canary_shape, _, _) = arena.port_layout(SemanticTrainingViewPort::Canaries)?;
        let canary_bytes = canary_shape.into_iter().try_fold(8u64, |bytes, extent| {
            let extent =
                u64::try_from(extent).map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            bytes
                .checked_mul(extent)
                .ok_or(SemanticTransitionError::GenerationExhausted)
        })?;
        dma[8] = u64::try_from(arena.selection_bytes()?)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?
            .checked_add(canary_bytes)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        for _ in keys {
            native_work_report::add_native_work_ceiling(&mut ceilings[0], selection)?;
            native_work_report::add_native_work_ceiling(&mut ceilings[1], dma)?;
        }
        Ok(ceilings)
    }

    fn reserve_cohort_material(
        &self,
        bytes: usize,
    ) -> Result<Arc<SemanticReplayMaterialReservation>, SemanticTransitionError> {
        // Exact encoding backing plus either pinned/read values or the final
        // Vec-to-Arc conversion. The native read retires its pin before cloning.
        // Both the frame and its saved byte slice own an Arc allocation header.
        let bound = u64::try_from(bytes)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?
            .checked_mul(3)
            .and_then(|bytes| {
                bytes.checked_add(
                    size_of::<FrozenFamilyCohort>() as u64 + 4 * size_of::<usize>() as u64,
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        Ok(Arc::new(
            self.replay_material_budget
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .reserve_auxiliary(bound)?,
        ))
    }

    fn read_family_cohort_port(
        &mut self,
        original: &mut CohortPreparation,
        view: DeviceMemoryView<u8>,
    ) -> Result<Vec<u8>, SemanticTransitionError> {
        if original.read.is_none() {
            original.read =
                Some(self.stage_publication_read_with_owner(view, original.reservation.clone())?);
        }
        let mut poisoned = self.poisoned;
        let result = self.resolve_publication_read_with_poison(
            original
                .read
                .as_mut()
                .expect("retained original cohort read"),
            &mut poisoned,
        );
        self.poisoned |= poisoned;
        let result = result?;
        original.read = None;
        Ok(result)
    }

    pub(super) fn prepared_family_cohort(
        &self,
        coordinates: [u64; 5],
    ) -> Result<Arc<SemanticEvaluationCohort>, SemanticTransitionError> {
        self.original_family_cohort(coordinates)?
            .ok_or_else(|| publication_input_error("freeze the original cohort before learning"))?
            .live()
    }
}
