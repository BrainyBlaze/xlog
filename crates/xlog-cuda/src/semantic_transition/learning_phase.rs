//! Cold learning-phase transitions over the complete native model allocation map.
//!
//! The privileged native owner supplies its durably confirmed Admission and the
//! complete copy/reset recipe. Scientific acceptance follows private execution
//! and belongs outside its immutable checkpoint. This module owns phase writes,
//! validates alias effects, and retains admission and ancestry; it never infers
//! a phase from a tensor role or grant.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticLearningPhase {
    Alignment = 0,
    Fast = 1,
    Consolidation = 2,
}

impl SemanticLearningPhase {
    pub fn from_code(code: u64) -> Result<Self, SemanticTransitionError> {
        match code {
            0 => Ok(Self::Alignment),
            1 => Ok(Self::Fast),
            2 => Ok(Self::Consolidation),
            _ => Err(publication_input_error("unknown learning phase")),
        }
    }
}

/// One operation on an explicitly identified typed view. Full backing allocations,
/// including padding, are copied first. No operation may silently change another
/// view, even when the producer deliberately shares its physical storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticLearningCopyReset {
    Preserve {
        role: u64,
        index: u64,
    },
    Zero {
        role: u64,
        index: u64,
    },
    MasterFromEffective {
        index: u64,
        effective_role: u64,
        effective_index: u64,
    },
    Phase {
        index: u64,
    },
}

impl SemanticLearningCopyReset {
    fn key(&self) -> (u64, u64) {
        match *self {
            Self::Preserve { role, index } | Self::Zero { role, index } => (role, index),
            Self::MasterFromEffective { index, .. } => (21, index),
            Self::Phase { index } => (23, index),
        }
    }
}

/// A transition request, not an assertion that a scientific criterion passed.
/// The privileged Controller supplies its original signed, durably read-back
/// Admission, separately from checking original data-use grants. This is permission
/// for private work, never a claim that the scientific comparison passed.
#[derive(Clone, Debug)]
pub struct SemanticLearningPhaseTransition {
    pub source: SemanticLearningPhase,
    pub target: SemanticLearningPhase,
    pub phase_index: u64,
    pub completed_updates_index: u64,
    pub recipe: Vec<SemanticLearningCopyReset>,
    pub admission: Option<SemanticLearningPhaseAdmission>,
}

/// Retained evidence for one actual native cold phase change. Ordinary Update and
/// restore preserve this history; neither can reset the cumulative progress.
#[derive(Clone)]
pub struct SemanticLearningPhaseRecord {
    pub source: SemanticLearningPhase,
    pub target: SemanticLearningPhase,
    pub phase_index: u64,
    pub completed_updates_index: u64,
    pub predecessor: SemanticPublishedIdentity,
    pub model_generation: u64,
    pub completed_updates: u64,
    pub recipe_digest: Identity256,
    recipe: Arc<[SemanticLearningCopyReset]>,
    admission: SemanticLearningPhaseAdmission,
    _metadata: Option<Arc<SemanticReplayMaterialReservation>>,
}

impl SemanticLearningPhaseRecord {
    pub(super) fn detach_admission_carrier(&mut self) -> Result<(), SemanticTransitionError> {
        self.admission =
            SemanticLearningPhaseAdmission::from_decoded(self.admission.bytes.copy_owned()?)?;
        Ok(())
    }

    pub fn recipe(&self) -> &[SemanticLearningCopyReset] {
        &self.recipe
    }

    pub fn admission(&self) -> &SemanticLearningPhaseAdmission {
        &self.admission
    }
}

impl std::fmt::Debug for SemanticLearningPhaseRecord {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SemanticLearningPhaseRecord")
            .field("source", &self.source)
            .field("target", &self.target)
            .field("phase_index", &self.phase_index)
            .field("completed_updates_index", &self.completed_updates_index)
            .field("predecessor", &self.predecessor)
            .field("model_generation", &self.model_generation)
            .field("completed_updates", &self.completed_updates)
            .field("recipe_digest", &self.recipe_digest)
            .field("recipe", &self.recipe())
            .field("admission", &self.admission)
            .finish()
    }
}

impl PartialEq for SemanticLearningPhaseRecord {
    fn eq(&self, other: &Self) -> bool {
        (
            self.source,
            self.target,
            self.phase_index,
            self.completed_updates_index,
            self.predecessor,
            self.model_generation,
            self.completed_updates,
            self.recipe_digest,
            self.recipe(),
            self.admission.as_bytes(),
        ) == (
            other.source,
            other.target,
            other.phase_index,
            other.completed_updates_index,
            other.predecessor,
            other.model_generation,
            other.completed_updates,
            other.recipe_digest,
            other.recipe(),
            other.admission.as_bytes(),
        )
    }
}

impl Eq for SemanticLearningPhaseRecord {}

pub(super) fn record_allocation_bound(
    recipe: &[SemanticLearningCopyReset],
    history_length: usize,
    history_capacity: usize,
    model_allocation_metadata_bytes: usize,
) -> Result<u64, SemanticTransitionError> {
    let mut bytes = recipe
        .len()
        .checked_mul(size_of::<SemanticLearningCopyReset>())
        .and_then(|bytes| bytes.checked_add(2 * size_of::<usize>()))
        .and_then(|bytes| {
            bytes.checked_add(
                size_of::<SemanticReplayMaterialReservation>() + 2 * size_of::<usize>(),
            )
        })
        .and_then(|bytes| bytes.checked_add(model_allocation_metadata_bytes))
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    let history_bytes = history_length
        .checked_add(1)
        .and_then(|count| count.checked_mul(size_of::<SemanticLearningPhaseRecord>()))
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    // The restored Session owns a distinct record array, even though each
    // record shares its original recipe, admission and allocation receipt.
    bytes = bytes
        .checked_add(history_bytes)
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    if history_length == history_capacity {
        bytes = bytes
            .checked_add(history_bytes)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
    }
    u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)
}

fn learning_recipe_digest(
    source: SemanticLearningPhase,
    target: SemanticLearningPhase,
    phase_index: u64,
    completed_updates_index: u64,
    recipe: &[SemanticLearningCopyReset],
) -> Identity256 {
    let mut digest = Sha256::new();
    digest.update(b"xlog.learning-phase.copy-reset.v1\0");
    for value in [
        source as u64,
        target as u64,
        phase_index,
        completed_updates_index,
        recipe.len() as u64,
    ] {
        digest.update(value.to_le_bytes());
    }
    for operation in recipe {
        let (role, index) = operation.key();
        let (code, source_role, source_index) = match *operation {
            SemanticLearningCopyReset::Preserve { .. } => (0, 0, 0),
            SemanticLearningCopyReset::Zero { .. } => (1, 0, 0),
            SemanticLearningCopyReset::MasterFromEffective {
                effective_role,
                effective_index,
                ..
            } => (2, effective_role, effective_index),
            SemanticLearningCopyReset::Phase { .. } => (3, 0, 0),
        };
        for value in [role, index, code, source_role, source_index] {
            digest.update(value.to_le_bytes());
        }
    }
    Identity256::from_bytes(digest.finalize().into())
}

impl SemanticLearningPhaseTransition {
    pub fn recipe_digest(&self) -> Identity256 {
        learning_recipe_digest(
            self.source,
            self.target,
            self.phase_index,
            self.completed_updates_index,
            &self.recipe,
        )
    }

    pub(super) fn apply<B: PublicationByteStorage>(
        &self,
        material: &mut PublicationMaterialStorage<B>,
        metadata: Arc<SemanticReplayMaterialReservation>,
    ) -> Result<(SemanticLearningPhaseRecord, Option<LearningFoldPlan>), SemanticTransitionError>
    {
        if !matches!(
            (self.source, self.target),
            (
                SemanticLearningPhase::Alignment,
                SemanticLearningPhase::Fast
            ) | (
                SemanticLearningPhase::Fast,
                SemanticLearningPhase::Consolidation
            ) | (
                SemanticLearningPhase::Consolidation,
                SemanticLearningPhase::Fast
            )
        ) || self
            .admission
            .as_ref()
            .is_none_or(SemanticLearningPhaseAdmission::is_empty)
            || self.phase_index == self.completed_updates_index
        {
            return Err(publication_input_error(
                "learning transition lacks its permitted boundary or original native admission",
            ));
        }
        material.require_successful_recompute()?;
        let phase = scalar_i64(material, (23, self.phase_index))?;
        let updates = scalar_i64(material, (23, self.completed_updates_index))?;
        if phase != self.source as i64 || updates < 0 {
            return Err(publication_input_error(
                "learning transition differs from the selected phase or cumulative progress",
            ));
        }
        if let Some(previous) = material.learning_phases.last() {
            if previous.target != self.source
                || previous.phase_index != self.phase_index
                || previous.completed_updates_index != self.completed_updates_index
                || previous.completed_updates > updates as u64
            {
                return Err(publication_input_error(
                    "learning transition changed its retained phase lineage",
                ));
            }
        } else if self.source != SemanticLearningPhase::Alignment {
            return Err(publication_input_error(
                "learning lineage must begin with admitted alignment",
            ));
        }
        let node = OwnedPublicationMaterial::tree_node_bound(size_of::<(u64, u64)>(), 0)?;
        let mut workspace_bytes = material
            .layouts
            .len()
            .checked_add(self.recipe.len())
            .and_then(|count| count.checked_mul(node))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        for (count, width) in [
            (self.recipe.len(), size_of::<(u64, u64)>()),
            (material.model_allocations.len(), size_of::<B>()),
        ] {
            workspace_bytes = count
                .checked_mul(width)
                .and_then(|bytes| workspace_bytes.checked_add(bytes))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let _workspace_reservation = metadata.budget.reserve_auxiliary(
            u64::try_from(workspace_bytes)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?;
        let expected = material
            .layouts
            .keys()
            .filter(|(role, _)| (18..=25).contains(role))
            .copied()
            .collect::<BTreeSet<_>>();
        let keys = self
            .recipe
            .iter()
            .map(SemanticLearningCopyReset::key)
            .collect::<Vec<_>>();
        if keys.windows(2).any(|pair| pair[0] >= pair[1])
            || keys.iter().copied().collect::<BTreeSet<_>>() != expected
            || !self.recipe.contains(&SemanticLearningCopyReset::Phase {
                index: self.phase_index,
            })
            || !self.recipe.contains(&SemanticLearningCopyReset::Preserve {
                role: 23,
                index: self.completed_updates_index,
            })
        {
            return Err(publication_input_error("copy/reset recipe must cover every model and learning view exactly once in native order"));
        }
        let fold = if self.source == SemanticLearningPhase::Fast
            && self.target == SemanticLearningPhase::Consolidation
        {
            Some(LearningFoldPlan::prepare(material, self, &metadata.budget)?)
        } else {
            None
        };
        // Absorption reads the original effective factors on the device. In
        // particular, its masters must not be reset from pre-absorption values.
        if fold.is_none() {
            let original = &material.model_allocations;
            let mut allocations = Vec::with_capacity(original.len());
            for allocation in original {
                allocations.push(allocation.copy_owned()?);
            }
            for operation in &self.recipe {
                if matches!(operation, SemanticLearningCopyReset::Preserve { .. }) {
                    continue;
                }
                visit_expected_values(
                    material,
                    operation,
                    self.target,
                    self.phase_index,
                    |allocation, offset, value| {
                        allocations[allocation].writable_bytes()?[offset..offset + value.len()]
                            .copy_from_slice(value);
                        Ok(())
                    },
                )?;
            }
            for operation in &self.recipe {
                visit_expected_values(
                    material,
                    operation,
                    self.target,
                    self.phase_index,
                    |allocation, offset, value| {
                        if allocations[allocation][offset..offset + value.len()] != *value {
                            return Err(publication_input_error(
                            "copy/reset recipe has conflicting effects on shared physical views",
                        ));
                        }
                        Ok(())
                    },
                )?;
            }
            material.model_allocations = allocations;
            for range in &mut material.ranges {
                if matches!(range.range.role, 18..=25) {
                    let (allocation, offset) = material
                        .model_memory
                        .location(range.range.role, range.range.index)?;
                    range.bytes = material.model_allocations[allocation]
                        .slice_bytes(offset..offset + range.bytes.len())?;
                }
            }
        }
        let header = material.bank.header;
        Ok((
            SemanticLearningPhaseRecord {
                source: self.source,
                target: self.target,
                phase_index: self.phase_index,
                completed_updates_index: self.completed_updates_index,
                predecessor: SemanticPublishedIdentity {
                    instance: header.instance,
                    word: header.publication_word,
                    logical_digest: header.logical_digest,
                },
                model_generation: header.model_generation,
                completed_updates: updates as u64,
                recipe_digest: self.recipe_digest(),
                recipe: Arc::from(self.recipe.as_slice()),
                admission: self
                    .admission
                    .as_ref()
                    .expect("checked original learning admission")
                    .clone(),
                _metadata: Some(metadata),
            },
            fold,
        ))
    }
}

fn scalar_i64<B: PublicationByteStorage>(
    material: &PublicationMaterialStorage<B>,
    key: (u64, u64),
) -> Result<i64, SemanticTransitionError> {
    let layout = material
        .layouts
        .get(&key)
        .ok_or_else(|| publication_input_error("learning scalar is absent"))?;
    if layout.rank != 0 || layout.scalar_type != 7 || layout.element_bytes != 8 {
        return Err(publication_input_error(
            "learning phase and progress need their actual I64 scalar views",
        ));
    }
    let (allocation, offset) = material.model_memory.location(key.0, key.1)?;
    let bytes = material.model_allocations[allocation]
        .get(offset..offset + 8)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    Ok(i64::from_le_bytes(bytes.try_into().map_err(|_| {
        SemanticTransitionError::ObservationMismatch
    })?))
}

fn element_offsets<'a, B: PublicationByteStorage>(
    material: &'a PublicationMaterialStorage<B>,
    key: (u64, u64),
) -> Result<
    (
        usize,
        impl Iterator<Item = Result<usize, SemanticTransitionError>> + 'a,
    ),
    SemanticTransitionError,
> {
    let layout = material
        .layouts
        .get(&key)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let (allocation, base) = material.model_memory.location(key.0, key.1)?;
    let rank =
        usize::try_from(layout.rank).map_err(|_| SemanticTransitionError::ObservationMismatch)?;
    if rank > 4 {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let count = layout.dimensions[..rank]
        .iter()
        .try_fold(1u64, |n, &dimension| {
            n.checked_mul(dimension)
                .ok_or(SemanticTransitionError::GenerationExhausted)
        })?;
    let offsets = (0..count).map(move |mut position| {
        let mut offset = base as u64;
        for axis in (0..rank).rev() {
            let coordinate = position % layout.dimensions[axis];
            position /= layout.dimensions[axis];
            offset = coordinate
                .checked_mul(layout.strides_bytes[axis])
                .and_then(|stride| offset.checked_add(stride))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let end = offset
            .checked_add(layout.element_bytes)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if end > material.model_memory.allocation_bytes[allocation] {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        usize::try_from(offset).map_err(|_| SemanticTransitionError::GenerationExhausted)
    });
    Ok((allocation, offsets))
}

fn visit_expected_values<B: PublicationByteStorage>(
    material: &PublicationMaterialStorage<B>,
    operation: &SemanticLearningCopyReset,
    target: SemanticLearningPhase,
    phase_index: u64,
    mut visit: impl FnMut(usize, usize, &[u8]) -> Result<(), SemanticTransitionError>,
) -> Result<(), SemanticTransitionError> {
    let key = operation.key();
    let width = material.layouts[&key].element_bytes as usize;
    if width == 0 || width > 8 {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let (allocation, offsets) = element_offsets(material, key)?;
    let mut effective_offsets = None;
    match *operation {
        SemanticLearningCopyReset::Zero { role, .. } if !matches!(role, 21 | 24 | 25) => {
            return Err(publication_input_error(
                "phase reset cannot zero effective weights, masks, rates or loss scale",
            ))
        }
        SemanticLearningCopyReset::Phase { index } if index != phase_index => {
            return Err(publication_input_error(
                "recipe writes another learning phase",
            ))
        }
        SemanticLearningCopyReset::MasterFromEffective {
            effective_role,
            effective_index,
            ..
        } => {
            let effective_key = (effective_role, effective_index);
            let layout = &material.layouts[&key];
            let effective = material.layouts.get(&effective_key).ok_or_else(|| {
                publication_input_error("master reset has no original effective parameter")
            })?;
            if !(18..=20).contains(&effective_role)
                || effective.scalar_type != 5
                || effective.element_bytes != 2
                || layout.scalar_type != 6
                || width != 4
                || effective.rank != layout.rank
                || effective.dimensions != layout.dimensions
            {
                return Err(publication_input_error("master reset requires the actual matching BF16 effective and FP32 master views"));
            }
            effective_offsets = Some(element_offsets(material, effective_key)?);
        }
        _ => {}
    }
    let phase = (target as i64).to_le_bytes();
    for offset in offsets {
        let offset = offset?;
        let mut converted = [0u8; 4];
        let expected = match *operation {
            SemanticLearningCopyReset::Preserve { .. } => {
                &material.model_allocations[allocation][offset..offset + width]
            }
            SemanticLearningCopyReset::Zero { .. } => &[0u8; 8][..width],
            SemanticLearningCopyReset::Phase { .. } => &phase,
            SemanticLearningCopyReset::MasterFromEffective { .. } => {
                let (source_allocation, offsets) = effective_offsets
                    .as_mut()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let offset = offsets
                    .next()
                    .ok_or(SemanticTransitionError::ObservationMismatch)??;
                let bytes = &material.model_allocations[*source_allocation][offset..offset + 2];
                let bits = u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16;
                if !f32::from_bits(bits).is_finite() {
                    return Err(publication_input_error(
                        "master reset encountered a nonfinite accepted effective weight",
                    ));
                }
                converted.copy_from_slice(&bits.to_le_bytes());
                &converted
            }
        };
        visit(allocation, offset, expected)?;
    }
    Ok(())
}

#[derive(Clone, PartialEq, Eq)]
struct FoldAssignment {
    key: (u64, u64),
    mode: u64,
    source: Option<(u64, u64)>,
    factors: Option<((u64, u64), (u64, u64))>,
}

/// Derived only from the complete original ModelContract and native memory map.
/// This is an ephemeral launch plan, not a new producer roster or wire format.
#[derive(Clone)]
pub(super) struct LearningFoldPlan {
    assignments: Vec<FoldAssignment>,
    scale_bits: u32,
    phase: u64,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

impl PartialEq for LearningFoldPlan {
    fn eq(&self, other: &Self) -> bool {
        self.assignments == other.assignments
            && self.scale_bits == other.scale_bits
            && self.phase == other.phase
    }
}

impl Eq for LearningFoldPlan {}

fn fold_error() -> SemanticTransitionError {
    publication_input_error(
        "adapter absorption differs from its original model contract or physical views",
    )
}

fn fold_target_name(
    prefix: &str,
    name: &str,
    suffix: &str,
) -> Result<String, SemanticTransitionError> {
    let bytes = prefix
        .len()
        .checked_add(name.len())
        .and_then(|bytes| bytes.checked_add(suffix.len()))
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    let mut result = String::with_capacity(bytes);
    result.push_str(prefix);
    result.push_str(name);
    result.push_str(suffix);
    Ok(result)
}

fn schema_array(
    value: &serde_json::Value,
) -> Result<&Vec<serde_json::Value>, SemanticTransitionError> {
    value.as_array().ok_or_else(fold_error)
}

fn tagged<'a>(
    value: &'a serde_json::Value,
    tag: &str,
) -> Result<&'a serde_json::Value, SemanticTransitionError> {
    let pair = schema_array(value)?;
    if pair.len() != 2 || pair[0].as_str() != Some(tag) {
        return Err(fold_error());
    }
    Ok(&pair[1])
}

fn original_float(value: &serde_json::Value) -> Result<f32, SemanticTransitionError> {
    // Python's canonical float.hex encoding, not a caller-provided decimal or
    // a replacement scale inferred from target names.
    let text = tagged(value, "float")?.as_str().ok_or_else(fold_error)?;
    let (negative, text) = text.strip_prefix('-').map_or((false, text), |s| (true, s));
    let (mantissa, exponent) = text
        .strip_prefix("0x")
        .and_then(|s| s.split_once('p'))
        .ok_or_else(fold_error)?;
    let (whole, fraction) = mantissa.split_once('.').ok_or_else(fold_error)?;
    if whole.len() != 1 || fraction.is_empty() || fraction.len() > 13 {
        return Err(fold_error());
    }
    let significand =
        u64::from_str_radix(&format!("{whole}{fraction}"), 16).map_err(|_| fold_error())?;
    let exponent = exponent.parse::<i32>().map_err(|_| fold_error())?;
    if !(-1022..=1023).contains(&exponent) || significand > (1u64 << 53) - 1 {
        return Err(fold_error());
    }
    let value = (significand as f64 / (1u64 << (4 * fraction.len())) as f64) * 2f64.powi(exponent);
    let value = if negative { -value } else { value } as f32;
    if !value.is_finite() {
        return Err(fold_error());
    }
    Ok(value)
}

fn view_key<B: PublicationByteStorage>(
    value: &serde_json::Value,
    material: &PublicationMaterialStorage<B>,
) -> Result<(u64, u64), SemanticTransitionError> {
    let fields = schema_array(&value["layout"])?;
    if fields.len() != 8 {
        return Err(fold_error());
    }
    let word = |i: usize| fields[i].as_u64().ok_or_else(fold_error);
    let key = (word(0)?, word(1)?);
    let layout = material.layouts.get(&key).ok_or_else(fold_error)?;
    if [
        layout.role,
        layout.index,
        layout.element_bytes,
        layout.scalar_type,
        layout.rank,
        layout.logical_axis,
    ] != [word(0)?, word(1)?, word(2)?, word(3)?, word(4)?, word(5)?]
        || !matches!(key.0, 18..=25)
    {
        return Err(fold_error());
    }
    for (field, expected) in [(6, layout.dimensions), (7, layout.strides_bytes)] {
        let actual = schema_array(&fields[field])?;
        if actual.len() != 4
            || actual
                .iter()
                .zip(expected)
                .any(|(v, n)| v.as_u64() != Some(n))
        {
            return Err(fold_error());
        }
    }
    Ok(key)
}

impl LearningFoldPlan {
    pub(super) fn clone_allocation_bytes(&self) -> Result<usize, SemanticTransitionError> {
        self.assignments
            .len()
            .checked_mul(size_of::<FoldAssignment>())
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }

    fn prepare<B: PublicationByteStorage>(
        material: &PublicationMaterialStorage<B>,
        transition: &SemanticLearningPhaseTransition,
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<Self, SemanticTransitionError> {
        let record = material
            .ranges
            .iter()
            .find(|r| (r.range.role, r.range.index) == (44, 0))
            .ok_or_else(fold_error)?;
        let schema_layout = material.contract.model_contract_layout;
        schema_layout.validate(record.bytes.len() as u64)?;
        let schema_begin = usize::try_from(schema_layout.schema_begin)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let schema_bytes = usize::try_from(schema_layout.schema_bytes)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let _schema_reservation =
            budget.reserve_json_parse(&record.bytes[schema_begin..schema_begin + schema_bytes])?;
        let schema = retained_model_schema(&record.bytes, material.contract.model_contract_layout)?
            .ok_or_else(fold_error)?;
        let model = &schema["model"];
        let physical = model["physical"].as_object().ok_or_else(fold_error)?;
        let learning = &model["learning"];
        let views = schema_array(&learning["views"])?;
        let leaves = schema_array(&learning["leaves"])?;
        let buffers = schema_array(&learning["buffers"])?;
        let state = schema_array(&model["roots"]["adapter"]["modules"][""][1])?;
        let target_names = state
            .iter()
            .find_map(|field| {
                let pair = field.as_array()?;
                (pair.first()?.as_str()? == "target_names")
                    .then(|| pair.get(1))
                    .flatten()
            })
            .ok_or_else(fold_error)?;
        let target_names = schema_array(tagged(target_names, "tuple")?)?;
        let assignment_capacity = target_names
            .len()
            .checked_mul(3)
            .and_then(|targets| targets.checked_add(transition.recipe.len()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut alias_count = 0usize;
        let mut alias_bytes = 0usize;
        for item in leaves.iter().chain(buffers) {
            for name in schema_array(&item["aliases"])? {
                alias_count = alias_count
                    .checked_add(1)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                alias_bytes = alias_bytes
                    .checked_add(name.as_str().ok_or_else(fold_error)?.len())
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
            }
        }
        let mut generated_bytes = 0usize;
        for (index, target) in target_names.iter().enumerate() {
            let target = tagged(target, "str")?.as_str().ok_or_else(fold_error)?;
            let mut digits = 1usize;
            let mut ordinal = index;
            while ordinal >= 10 {
                ordinal /= 10;
                digits += 1;
            }
            for (prefix, suffix, width, copies) in [
                ("model.", ".weight", target.len(), 2usize),
                ("adapter.residuals.", ".down", digits, 2),
                ("adapter.residuals.", ".up", digits, 2),
                ("adapter.residuals.", ".neutral_down", digits, 1),
            ] {
                generated_bytes = prefix
                    .len()
                    .checked_add(suffix.len())
                    .and_then(|bytes| bytes.checked_add(width))
                    .and_then(|bytes| bytes.checked_mul(copies))
                    .and_then(|bytes| generated_bytes.checked_add(bytes))
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
            }
        }
        let node = OwnedPublicationMaterial::tree_node_bound;
        let mut workspace_bytes = alias_bytes
            .checked_add(generated_bytes)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        for (count, width) in [
            (views.len(), size_of::<(u64, u64)>()),
            (material.layouts.len(), node(size_of::<(u64, u64)>(), 0)?),
            (views.len(), node(size_of::<(u64, u64)>(), 0)?),
            (views.len(), 2 * node(size_of::<&str>(), size_of::<u64>())?),
            (views.len(), 2 * node(size_of::<u64>(), 0)?),
            (
                alias_count,
                node(size_of::<String>(), size_of::<(u64, u64)>())?,
            ),
            (
                leaves.len(),
                node(
                    size_of::<(u64, u64)>(),
                    size_of::<&Vec<serde_json::Value>>(),
                )?,
            ),
            (
                leaves.len(),
                node(size_of::<(u64, u64)>(), size_of::<(u64, u64)>())?,
            ),
            (leaves.len(), node(size_of::<&str>(), 0)?),
            (buffers.len(), node(size_of::<&str>(), 0)?),
            (physical.len(), node(size_of::<&str>(), 0)?),
            (
                state.len(),
                node(size_of::<&str>(), size_of::<&serde_json::Value>())?,
            ),
            (target_names.len(), 3 * node(size_of::<(u64, u64)>(), 0)?),
            (target_names.len(), 3 * node(size_of::<String>(), 0)?),
            (4, size_of::<(u64, u64)>()),
            (2, size_of::<u64>()),
        ] {
            workspace_bytes = count
                .checked_mul(width)
                .and_then(|bytes| workspace_bytes.checked_add(bytes))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let _workspace_reservation = budget.reserve_auxiliary(
            u64::try_from(workspace_bytes)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?;
        let assignment_bytes = assignment_capacity
            .checked_mul(size_of::<FoldAssignment>())
            .and_then(|bytes| {
                bytes.checked_add(
                    size_of::<SemanticReplayMaterialReservation>() + 2 * size_of::<usize>(),
                )
            })
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = Arc::new(budget.reserve_auxiliary(assignment_bytes)?);
        let mut keys = Vec::with_capacity(views.len());
        for view in views {
            keys.push(view_key(view, material)?);
        }
        let expected = material
            .layouts
            .keys()
            .filter(|(role, _)| matches!(role, 18..=25))
            .copied()
            .collect::<BTreeSet<_>>();
        if keys.iter().copied().collect::<BTreeSet<_>>() != expected || keys.len() != expected.len()
        {
            return Err(fold_error());
        }
        let mut allocation_classes = BTreeMap::new();
        let mut storage_classes = BTreeMap::new();
        for (view, key) in views.iter().zip(&keys) {
            let geometry = &view["geometry"];
            let layout = material.layouts[key];
            let native_view = material
                .model_memory
                .views
                .iter()
                .find(|v| (v.role, v.index) == *key)
                .ok_or_else(fold_error)?;
            let storage = &material.model_memory.storages[native_view.storage as usize];
            let word = |name: &str| geometry[name].as_u64().ok_or_else(fold_error);
            let allocation = geometry["allocation"].as_str().ok_or_else(fold_error)?;
            let storage_class = geometry["storage"].as_str().ok_or_else(fold_error)?;
            let dimensions = schema_array(&geometry["shape"])?;
            let strides = schema_array(&geometry["stride"])?;
            let dtype = match geometry["dtype"].as_str() {
                Some("torch.uint8") => (1, 1),
                Some("torch.uint32") => (2, 4),
                Some("torch.uint64") => (3, 8),
                Some("torch.float16") => (4, 2),
                Some("torch.bfloat16") => (5, 2),
                Some("torch.float32") => (6, 4),
                Some("torch.int64") => (7, 8),
                Some("torch.bool") => (8, 1),
                _ => return Err(fold_error()),
            };
            if dimensions.len() != layout.rank as usize
                || dtype != (layout.scalar_type, layout.element_bytes)
                || strides.len() != dimensions.len()
                || dimensions
                    .iter()
                    .zip(layout.dimensions)
                    .any(|(d, n)| d.as_u64() != Some(n))
                || strides.iter().zip(layout.strides_bytes).any(|(s, n)| {
                    s.as_u64().and_then(|s| s.checked_mul(layout.element_bytes)) != Some(n)
                })
                || word("offset")?.checked_mul(layout.element_bytes)
                    != Some(native_view.byte_offset)
                || word("span_bytes")? != storage.span_bytes
                || word("allocation_byte_offset")? != storage.byte_offset
                || word("allocation_span_bytes")?
                    != material.model_memory.allocation_bytes[storage.allocation as usize]
                || allocation_classes
                    .insert(allocation, storage.allocation)
                    .is_some_and(|a| a != storage.allocation)
                || storage_classes
                    .insert(storage_class, native_view.storage)
                    .is_some_and(|s| s != native_view.storage)
            {
                return Err(fold_error());
            }
        }
        if allocation_classes
            .values()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != allocation_classes.len()
            || storage_classes
                .values()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != storage_classes.len()
        {
            return Err(fold_error());
        }
        let key_at = |value: &serde_json::Value| {
            value
                .as_u64()
                .and_then(|i| usize::try_from(i).ok())
                .and_then(|i| keys.get(i))
                .copied()
                .ok_or_else(fold_error)
        };
        let mut aliases = BTreeMap::new();
        let mut owner_aliases = BTreeMap::new();
        let mut masters = BTreeMap::new();
        let mut owners = BTreeSet::new();
        if key_at(&learning["shared"]["phase"])? != (23, transition.phase_index)
            || key_at(&learning["shared"]["completed_updates"])?
                != (23, transition.completed_updates_index)
        {
            return Err(fold_error());
        }
        for leaf in schema_array(&learning["leaves"])? {
            let key = key_at(&leaf["effective"])?;
            let view = &views[leaf["effective"].as_u64().ok_or_else(fold_error)? as usize];
            let owner = view["owner"].as_str().ok_or_else(fold_error)?;
            let entry = physical.get(owner).ok_or_else(fold_error)?;
            if !owners.insert(owner) {
                return Err(fold_error());
            }
            if entry["kind"].as_str() != Some("parameter")
                || entry["geometry"] != view["geometry"]
                || entry["aliases"] != leaf["aliases"]
                || !(18..=20).contains(&key.0)
            {
                return Err(fold_error());
            }
            let names = schema_array(&leaf["aliases"])?;
            if names.is_empty() || owner_aliases.insert(key, names).is_some() {
                return Err(fold_error());
            }
            for name in names {
                if aliases
                    .insert(name.as_str().ok_or_else(fold_error)?.to_owned(), key)
                    .is_some()
                {
                    return Err(fold_error());
                }
            }
            if !leaf["master"].is_null() && masters.insert(key_at(&leaf["master"])?, key).is_some()
            {
                return Err(fold_error());
            }
        }
        for buffer in schema_array(&learning["buffers"])? {
            let key = key_at(&buffer["tensor"])?;
            let view = &views[buffer["tensor"].as_u64().ok_or_else(fold_error)? as usize];
            let entry = physical
                .get(view["owner"].as_str().ok_or_else(fold_error)?)
                .ok_or_else(fold_error)?;
            if !owners.insert(view["owner"].as_str().ok_or_else(fold_error)?) {
                return Err(fold_error());
            }
            if entry["kind"].as_str() != Some("buffer")
                || entry["geometry"] != view["geometry"]
                || entry["aliases"] != buffer["aliases"]
            {
                return Err(fold_error());
            }
            for name in schema_array(&buffer["aliases"])? {
                if aliases
                    .insert(name.as_str().ok_or_else(fold_error)?.to_owned(), key)
                    .is_some()
                {
                    return Err(fold_error());
                }
            }
        }
        if owners != physical.keys().map(String::as_str).collect::<BTreeSet<_>>() {
            return Err(fold_error());
        }
        let mut fields = BTreeMap::new();
        for field in state {
            let field = schema_array(field)?;
            if field.len() != 2
                || fields
                    .insert(field[0].as_str().ok_or_else(fold_error)?, &field[1])
                    .is_some()
            {
                return Err(fold_error());
            }
        }
        let field = |name: &str| fields.get(name).copied().ok_or_else(fold_error);
        let targets = schema_array(tagged(field("target_names")?, "tuple")?)?;
        let shapes = schema_array(tagged(field("target_shapes")?, "tuple")?)?;
        let rank = tagged(field("rank")?, "int")?
            .as_u64()
            .filter(|r| *r > 0)
            .ok_or_else(fold_error)?;
        let scale = original_float(field("scale")?)?;
        if targets.is_empty()
            || targets.len() != shapes.len()
            || original_float(field("dropout")?)? != 0.0
            || tagged(field("initialization_identity")?, "str")?
                .as_str()
                .is_none_or(str::is_empty)
        {
            return Err(fold_error());
        }
        let mut assignments = Vec::with_capacity(assignment_capacity);
        for operation in &transition.recipe {
            let key = operation.key();
            let source = match *operation {
                SemanticLearningCopyReset::MasterFromEffective {
                    effective_role,
                    effective_index,
                    ..
                } => {
                    let effective = (effective_role, effective_index);
                    if masters.get(&key) != Some(&effective) {
                        return Err(fold_error());
                    }
                    let layout = material.layouts[&key];
                    let original = material.layouts.get(&effective).ok_or_else(fold_error)?;
                    if key.0 != 21
                        || layout.scalar_type != 6
                        || layout.element_bytes != 4
                        || original.scalar_type != 5
                        || original.element_bytes != 2
                        || layout.rank != original.rank
                        || layout.dimensions != original.dimensions
                    {
                        return Err(fold_error());
                    }
                    Some(effective)
                }
                SemanticLearningCopyReset::Zero { role, .. } if !matches!(role, 21 | 24 | 25) => {
                    return Err(fold_error())
                }
                SemanticLearningCopyReset::Phase { index } if index != transition.phase_index => {
                    return Err(fold_error())
                }
                _ => None,
            };
            assignments.push(FoldAssignment {
                key,
                mode: match operation {
                    SemanticLearningCopyReset::Preserve { .. } => 0,
                    SemanticLearningCopyReset::Zero { .. } => 1,
                    SemanticLearningCopyReset::MasterFromEffective { .. } => 2,
                    SemanticLearningCopyReset::Phase { .. } => 3,
                },
                source,
                factors: None,
            });
        }
        if masters
            .keys()
            .any(|key| !assignments.iter().any(|a| a.key == *key && a.mode == 2))
        {
            return Err(fold_error());
        }
        for leaf in schema_array(&learning["leaves"])? {
            for field in ["m", "v", "t", "gradient", "presence"] {
                if !leaf[field].is_null() {
                    let key = key_at(&leaf[field])?;
                    if !assignments.iter().any(|a| a.key == key && a.mode == 1) {
                        return Err(fold_error());
                    }
                }
            }
        }
        let accumulation = key_at(&learning["shared"]["accumulation"])?;
        if !assignments
            .iter()
            .any(|a| a.key == accumulation && a.mode == 1)
        {
            return Err(fold_error());
        }
        let mut replaced = BTreeSet::new();
        let mut adapted_aliases = BTreeSet::new();
        let mut previous = None;
        for (index, (target, shape)) in targets.iter().zip(shapes).enumerate() {
            let target = tagged(target, "str")?.as_str().ok_or_else(fold_error)?;
            if previous.is_some_and(|p| p >= target) {
                return Err(fold_error());
            }
            previous = Some(target);
            let shape = schema_array(tagged(shape, "tuple")?)?;
            if shape.len() != 2 {
                return Err(fold_error());
            }
            let rows = tagged(&shape[0], "int")?
                .as_u64()
                .filter(|n| *n > 0)
                .ok_or_else(fold_error)?;
            let columns = tagged(&shape[1], "int")?
                .as_u64()
                .filter(|n| *n > 0)
                .ok_or_else(fold_error)?;
            let mut ordinal = index;
            let mut decimal = [b'0'; usize::BITS as usize];
            let mut begin = decimal.len() - 1;
            while ordinal >= 10 {
                decimal[begin] = b'0' + (ordinal % 10) as u8;
                ordinal /= 10;
                begin -= 1;
            }
            decimal[begin] = b'0' + ordinal as u8;
            let index = std::str::from_utf8(&decimal[begin..])
                .expect("original ordinal contains decimal digits");
            let names = [
                fold_target_name("model.", target, ".weight")?,
                fold_target_name("adapter.residuals.", index, ".down")?,
                fold_target_name("adapter.residuals.", index, ".up")?,
                fold_target_name("adapter.residuals.", index, ".neutral_down")?,
            ];
            let actual = names
                .iter()
                .map(|name| aliases.get(name).copied().ok_or_else(fold_error))
                .collect::<Result<Vec<_>, _>>()?;
            let [weight, down, up, neutral] = actual.as_slice() else {
                return Err(fold_error());
            };
            for (key, dims, role) in [
                (*weight, [rows, columns, 0, 0], 18),
                (*down, [rank, columns, 0, 0], 19),
                (*up, [rows, rank, 0, 0], 19),
                (*neutral, [rank, columns, 0, 0], 19),
            ] {
                let layout = material.layouts[&key];
                if key.0 != role
                    || layout.rank != 2
                    || layout.dimensions != dims
                    || !matches!((layout.scalar_type, layout.element_bytes), (5, 2) | (6, 4))
                    || (key == *neutral && layout.scalar_type != material.layouts[down].scalar_type)
                    || !transition
                        .recipe
                        .contains(&SemanticLearningCopyReset::Preserve {
                            role: key.0,
                            index: key.1,
                        })
                {
                    return Err(fold_error());
                }
            }
            for key in [*weight, *down, *up] {
                replaced.insert(key);
            }
            for name in &names[..3] {
                adapted_aliases.insert(name.clone());
            }
            assignments.push(FoldAssignment {
                key: *weight,
                mode: 4,
                source: None,
                factors: Some((*up, *down)),
            });
            assignments.push(FoldAssignment {
                key: *down,
                mode: 5,
                source: Some(*neutral),
                factors: None,
            });
            assignments.push(FoldAssignment {
                key: *up,
                mode: 1,
                source: None,
                factors: None,
            });
        }
        assignments.retain(|a| {
            a.mode != 0
                || !replaced.contains(&a.key)
                || owner_aliases.get(&a.key).is_some_and(|names| {
                    names
                        .iter()
                        .any(|n| n.as_str().is_none_or(|n| !adapted_aliases.contains(n)))
                })
        });
        Ok(Self {
            assignments,
            scale_bits: scale.to_bits(),
            phase: transition.target as u64,
            _reservation: reservation,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct FoldDeviceAssignment {
    original: u64,
    scratch: u64,
    expected: u64,
    source: u64,
    up: u64,
    down: u64,
    count: u64,
    mode: u64,
    phase: u64,
    scale_bits: u64,
    serial_scatter: u64,
    layout: SemanticTensorLayout,
    source_layout: SemanticTensorLayout,
    up_layout: SemanticTensorLayout,
    down_layout: SemanticTensorLayout,
}

unsafe impl DeviceRepr for FoldDeviceAssignment {}
const _: () = assert!(size_of::<FoldDeviceAssignment>() == 536);

fn packed_layout(
    mut layout: SemanticTensorLayout,
) -> Result<SemanticTensorLayout, SemanticTransitionError> {
    let mut stride = layout.element_bytes;
    for axis in (0..layout.rank as usize).rev() {
        layout.strides_bytes[axis] = stride;
        stride = stride
            .checked_mul(layout.dimensions[axis])
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
    }
    Ok(layout)
}

/// The original cold fold's allocations and ordered, single-entry commands.
/// Nothing in this owner is recreated after a device effect has entered.
pub(super) struct OriginalLearningFold {
    storage: Arc<PublicationStorage>,
    scratch: Vec<TrackedCudaSlice<u8>>,
    outputs: Vec<Option<TrackedCudaSlice<u8>>>,
    status: TrackedCudaSlice<u64>,
    status_write: state_restoration::OriginalDeviceWrite,
    status_read: PublicationRead<u64>,
    preserve: Vec<crate::device::RetainedDeviceMemoryCommand>,
    preserve_next: usize,
    kernels: Vec<LearningFoldKernel>,
    kernel_next: usize,
    execute: CudaFunction,
    checked: bool,
    assign: Vec<crate::device::RetainedDeviceMemoryCommand>,
    assign_next: usize,
    poisoned: bool,
    _reservation: Option<Arc<SemanticReplayMaterialReservation>>,
}

struct LearningFoldKernel {
    descriptor: FoldDeviceAssignment,
    stage: u64,
    command: OriginalNativeCommand,
}

fn fold_kernel_stage_reached(mode: u64, count: u64, stage: u64) -> bool {
    count != 0
        && !(stage == 0 && matches!(mode, 0 | 2))
        && !(stage == 1 && mode != 2)
        && !(stage == 2 && mode == 0)
}

fn resolve_fold_memory_command(
    domain: &ResidentExecutionDomain,
    command: &mut crate::device::RetainedDeviceMemoryCommand,
    poisoned: &mut bool,
    may_submit: bool,
) -> Result<(), SemanticTransitionError> {
    if !command.entered() {
        if !may_submit {
            return Err(SemanticTransitionError::Poisoned);
        }
        let mut recorder = domain.new_strict_recorder();
        recorder.write(command.destination());
        if let Some(source) = command.source() {
            recorder.read(source);
        }
        enqueue_recorded(domain, poisoned, recorder, |_| command.enqueue())?;
    }
    command.resolve().map_err(|error| {
        *poisoned = true;
        runtime_error("original learning fold copy completion", error)
    })
}

impl SemanticTransitionSession {
    pub(super) fn stage_learning_fold(
        &self,
        plan: &LearningFoldPlan,
        storage: Arc<PublicationStorage>,
    ) -> Result<OriginalLearningFold, SemanticTransitionError> {
        let mut kernel_count = 0usize;
        for assignment in &plan.assignments {
            let layout = storage.layouts[&assignment.key];
            let count = layout.dimensions[..layout.rank as usize].iter().try_fold(
                1u64,
                |count, &dimension| {
                    count
                        .checked_mul(dimension)
                        .ok_or(SemanticTransitionError::GenerationExhausted)
                },
            )?;
            for stage in 0..4 {
                kernel_count = kernel_count
                    .checked_add(usize::from(fold_kernel_stage_reached(
                        assignment.mode,
                        count,
                        stage,
                    )))
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
            }
        }
        let copy_count = storage
            .model_memory
            .allocation_bytes
            .iter()
            .filter(|&&bytes| bytes != 0)
            .count();
        let command_count = copy_count
            .checked_mul(2)
            .and_then(|copies| copies.checked_add(kernel_count))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let completion_bytes = usize::try_from(
            crate::device::ExecutionCompletion::single_stream_allocation_bytes()
                .map_err(|error| runtime_error("fold completion allocation bound", error))?,
        )
        .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let mut host_bytes = size_of::<OriginalLearningFold>()
            + size_of::<SemanticReplayMaterialReservation>()
            + 2 * size_of::<usize>();
        for (count, width) in [
            (
                storage.model_memory.allocation_bytes.len(),
                size_of::<TrackedCudaSlice<u8>>(),
            ),
            (
                plan.assignments.len(),
                size_of::<Option<TrackedCudaSlice<u8>>>(),
            ),
            (plan.assignments.len(), size_of::<FoldDeviceAssignment>()),
            (kernel_count, size_of::<LearningFoldKernel>()),
            (
                copy_count,
                2 * size_of::<crate::device::RetainedDeviceMemoryCommand>(),
            ),
            (command_count, completion_bytes),
            (
                plan.assignments
                    .iter()
                    .filter(|assignment| matches!(assignment.mode, 1 | 4 | 5))
                    .count(),
                OwnedPublicationMaterial::tree_node_bound(
                    size_of::<(u64, u64)>(),
                    size_of::<u64>(),
                )?,
            ),
            (
                1,
                state_restoration::OriginalDeviceWrite::metadata_allocation_bytes::<u64>(),
            ),
            (3, size_of::<u64>()),
        ] {
            host_bytes = count
                .checked_mul(width)
                .and_then(|bytes| host_bytes.checked_add(bytes))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let reservation = self
            .replay_material_budget
            .as_ref()
            .map(|budget| {
                budget
                    .reserve_auxiliary(
                        u64::try_from(host_bytes)
                            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                    )
                    .map(Arc::new)
            })
            .transpose()?;
        // A cold fold has a genuine private writer, not a second persistent
        // model bank. Preserve all bytes (including alias padding) before the
        // unchanged compute/scatter/all-view-check law and sole assignment.
        let mut scratch = Vec::with_capacity(storage.model_memory.allocation_bytes.len());
        for &bytes in &storage.model_memory.allocation_bytes {
            scratch.push(allocate_publication::<u8>(
                &self.provider,
                usize::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            )?);
        }
        let pointer = |key: (u64, u64), bank: usize| {
            let (allocation, offset) = storage.model_memory.location(key.0, key.1)?;
            let base = if bank == 0 {
                storage.allocations[storage.model_slots[allocation][0]]
                    .entry()
                    .pointer
            } else {
                scratch[allocation].device_ptr_value()
            };
            base.checked_add(offset as u64)
                .ok_or(SemanticTransitionError::GenerationExhausted)
        };
        let mut outputs = Vec::with_capacity(plan.assignments.len());
        let mut effective_outputs = BTreeMap::new();
        for assignment in &plan.assignments {
            let layout = storage.layouts[&assignment.key];
            let count =
                layout.dimensions[..layout.rank as usize]
                    .iter()
                    .try_fold(1u64, |n, &d| {
                        n.checked_mul(d)
                            .ok_or(SemanticTransitionError::GenerationExhausted)
                    })?;
            let bytes = count
                .checked_mul(layout.element_bytes)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let output = if assignment.mode == 0 || bytes == 0 {
                None
            } else {
                Some(allocate_publication::<u8>(&self.provider, bytes)?)
            };
            if matches!(assignment.mode, 1 | 4 | 5) {
                effective_outputs.entry(assignment.key).or_insert_with(|| {
                    output
                        .as_ref()
                        .map_or(0, TrackedCudaSlice::device_ptr_value)
                });
            }
            outputs.push(output);
        }
        let mut descriptors = Vec::with_capacity(plan.assignments.len());
        for (assignment, output) in plan.assignments.iter().zip(&outputs) {
            let layout = storage.layouts[&assignment.key];
            let mut descriptor = FoldDeviceAssignment {
                original: pointer(assignment.key, 0)?,
                scratch: pointer(assignment.key, 1)?,
                expected: output
                    .as_ref()
                    .map_or(0, TrackedCudaSlice::device_ptr_value),
                mode: assignment.mode,
                phase: plan.phase,
                scale_bits: u64::from(plan.scale_bits),
                layout,
                ..Default::default()
            };
            descriptor.count =
                layout.dimensions[..layout.rank as usize]
                    .iter()
                    .try_fold(1u64, |n, &d| {
                        n.checked_mul(d)
                            .ok_or(SemanticTransitionError::GenerationExhausted)
                    })?;
            if let Some(source) = assignment.source {
                descriptor.source_layout = storage.layouts[&source];
                if assignment.mode == 2 && effective_outputs.contains_key(&source) {
                    descriptor.source = effective_outputs[&source];
                    descriptor.source_layout = packed_layout(descriptor.source_layout)?;
                } else {
                    descriptor.source = pointer(source, 0)?;
                }
            }
            if let Some((up, down)) = assignment.factors {
                descriptor.up = pointer(up, 0)?;
                descriptor.down = pointer(down, 0)?;
                descriptor.up_layout = storage.layouts[&up];
                descriptor.down_layout = storage.layouts[&down];
            }
            // Parallel scatter is safe only for provably disjoint cells. A
            // deliberately overlapping stride uses the same byte-exact law in
            // serial order; the subsequent all-view check still detects conflict.
            let mut axes = [0usize; 4];
            let mut axis_count = 0;
            for axis in 0..layout.rank as usize {
                if layout.dimensions[axis] > 1 {
                    axes[axis_count] = axis;
                    axis_count += 1;
                }
            }
            let axes = &mut axes[..axis_count];
            axes.sort_unstable_by_key(|&i| layout.strides_bytes[i]);
            let mut span = layout.element_bytes;
            for &axis in axes.iter() {
                if layout.strides_bytes[axis] < span {
                    descriptor.serial_scatter = 1;
                }
                span = span
                    .checked_add(
                        (layout.dimensions[axis] - 1)
                            .checked_mul(layout.strides_bytes[axis])
                            .ok_or(SemanticTransitionError::GenerationExhausted)?,
                    )
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
            }
            descriptors.push(descriptor);
        }
        let status = allocate_publication::<u64>(&self.provider, 1)?;
        let status_write = state_restoration::OriginalDeviceWrite::new_with_owner(
            &self.stream,
            &[0u64],
            status.view(),
            false,
            reservation.clone(),
        )?;
        let status_read =
            self.stage_publication_read_with_owner(status.view(), reservation.clone())?;
        let execute = self
            .provider
            .device()
            .inner()
            .get_func("xlog_semantic_transition", "semantic_learning_fold")
            .ok_or_else(|| runtime_error("kernel lookup", "learning absorption unavailable"))?;
        let mut kernels = Vec::with_capacity(kernel_count);
        // Freeze the same complete compute/scatter/all-view-check sequence.
        // Masters consume already-rounded effective outputs, never an unrounded
        // accumulator, and no assignment enters before every check completes.
        for stage in 0..4u64 {
            for descriptor in &descriptors {
                if !fold_kernel_stage_reached(descriptor.mode, descriptor.count, stage) {
                    continue;
                }
                kernels.push(LearningFoldKernel {
                    descriptor: *descriptor,
                    stage,
                    command: OriginalNativeCommand::new(&self.domain)?,
                });
            }
        }
        let mut preserve = Vec::with_capacity(copy_count);
        let mut assign = Vec::with_capacity(copy_count);
        for (index, allocation) in scratch.iter().enumerate() {
            if allocation.is_empty() {
                continue;
            }
            let original = storage.allocations[storage.model_slots[index][0]]
                .slice()?
                .view();
            preserve.push(
                crate::device::RetainedDeviceMemoryCommand::copy(
                    &self.stream,
                    original.clone(),
                    allocation.view(),
                )
                .map_err(|error| runtime_error("original fold preservation staging", error))?,
            );
            assign.push(
                crate::device::RetainedDeviceMemoryCommand::copy(
                    &self.stream,
                    allocation.view(),
                    original,
                )
                .map_err(|error| runtime_error("original fold assignment staging", error))?,
            );
        }
        Ok(OriginalLearningFold {
            storage,
            scratch,
            outputs,
            status,
            status_write,
            status_read,
            preserve,
            preserve_next: 0,
            kernels,
            kernel_next: 0,
            execute,
            checked: false,
            assign,
            assign_next: 0,
            poisoned: false,
            _reservation: reservation,
        })
    }

    pub(super) fn apply_learning_fold(
        &mut self,
        original: &mut OriginalLearningFold,
        preparation: &Arc<state_restoration::OriginalPublicationPreparation>,
        restoration: Option<&Arc<state_restoration::OriginalStateMaterialRestore>>,
    ) -> Result<(), SemanticTransitionError> {
        if self
            .publication
            .as_ref()
            .is_none_or(|storage| !Arc::ptr_eq(storage, &original.storage))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        // Preserve the original effect order: full-backing copies, status
        // initialization, four kernel stages, status check, sole assignment.
        while original.preserve_next < original.preserve.len() {
            let may_submit = self.original_restoration_may_submit(restoration, Some(preparation));
            resolve_fold_memory_command(
                &self.domain,
                &mut original.preserve[original.preserve_next],
                &mut original.poisoned,
                may_submit,
            )?;
            original.preserve_next += 1;
        }
        let may_submit = self.original_restoration_may_submit(restoration, Some(preparation));
        original.status_write.resolve(
            &self.domain,
            &self.stream,
            &self.provider,
            &mut original.poisoned,
            may_submit,
        )?;
        while original.kernel_next < original.kernels.len() {
            let kernel = &mut original.kernels[original.kernel_next];
            if !kernel.command.resolve_entered(&mut original.poisoned)? {
                if !self.original_restoration_may_submit(restoration, Some(preparation)) {
                    return Err(SemanticTransitionError::Poisoned);
                }
                let mut recorder = self.domain.new_strict_recorder();
                original.storage.record(&mut recorder);
                for allocation in &original.scratch {
                    recorder.read_write(allocation);
                }
                recorder.read_write(&original.status);
                for output in original.outputs.iter().flatten() {
                    recorder.read_write(output);
                }
                let descriptor = kernel.descriptor;
                let stage = kernel.stage;
                let status = original.status.device_ptr_value();
                let execute = original.execute.clone();
                let blocks = descriptor.count.div_ceil(256).min(65535) as u32;
                kernel.command.run(
                    &self.domain,
                    &mut original.poisoned,
                    recorder,
                    |enqueue, entered, submitted| {
                        // SAFETY: the frozen descriptor, scratch, expectations
                        // and factor spans have their original retained owners
                        // in this exact command's strict recorder.
                        *entered = true;
                        unsafe {
                            execute.launch_in(
                                enqueue,
                                LaunchConfig {
                                    grid_dim: (blocks, 1, 1),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                (descriptor, stage, status),
                            )
                        }
                        .map_err(|error| XlogError::Kernel(error.to_string()))?;
                        *submitted = true;
                        Ok(())
                    },
                )?;
            }
            original.kernel_next += 1;
        }
        if !original.checked {
            let may_submit = self.original_restoration_may_submit(restoration, Some(preparation));
            let status = self.resolve_acquisition_read(
                &mut original.status_read,
                may_submit,
                &mut original.poisoned,
            )?;
            if status.as_slice() != [0] {
                return Err(publication_input_error("adapter absorption encountered nonfinite numerics or conflicting physical alias bytes"));
            }
            original.checked = true;
        }
        while original.assign_next < original.assign.len() {
            let may_submit = self.original_restoration_may_submit(restoration, Some(preparation));
            resolve_fold_memory_command(
                &self.domain,
                &mut original.assign[original.assign_next],
                &mut original.poisoned,
                may_submit,
            )?;
            original.assign_next += 1;
        }
        original.poisoned = false;
        Ok(())
    }
}

pub(super) fn encode_history(
    history: &[SemanticLearningPhaseRecord],
    bytes: &mut Vec<u8>,
) -> Result<(), SemanticTransitionError> {
    material_u32(
        bytes,
        u32::try_from(history.len()).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
    );
    for record in history {
        for value in [
            record.source as u64,
            record.target as u64,
            record.phase_index,
            record.completed_updates_index,
            record.predecessor.word,
            record.model_generation,
            record.completed_updates,
        ] {
            material_u64(bytes, value);
        }
        for digest in [
            record.predecessor.instance,
            record.predecessor.logical_digest,
            record.recipe_digest,
        ] {
            bytes.extend_from_slice(digest.as_bytes());
        }
        material_u32(
            bytes,
            u32::try_from(record.recipe().len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        );
        for operation in record.recipe() {
            let (role, index) = operation.key();
            let (code, source_role, source_index) = match *operation {
                SemanticLearningCopyReset::Preserve { .. } => (0, 0, 0),
                SemanticLearningCopyReset::Zero { .. } => (1, 0, 0),
                SemanticLearningCopyReset::MasterFromEffective {
                    effective_role,
                    effective_index,
                    ..
                } => (2, effective_role, effective_index),
                SemanticLearningCopyReset::Phase { .. } => (3, 0, 0),
            };
            for value in [role, index, code, source_role, source_index] {
                material_u64(bytes, value);
            }
        }
        material_bytes(bytes, record.admission.as_bytes())
            .map_err(SemanticTransitionError::Semantic)?;
    }
    Ok(())
}

pub(super) fn decode_history(
    reader: &mut SemanticMaterialReader<'_>,
    mut admission_owner: impl FnMut(
        &[u8],
    )
        -> Result<SemanticLearningPhaseAdmission, SemanticTransitionError>,
) -> Result<Vec<SemanticLearningPhaseRecord>, SemanticTransitionError> {
    let count = reader
        .count(7 * 8 + 3 * 32 + 4)
        .map_err(SemanticTransitionError::Semantic)?;
    let mut history = Vec::with_capacity(count);
    for _ in 0..count {
        let mut next = || reader.u64().map_err(SemanticTransitionError::Semantic);
        let source = SemanticLearningPhase::from_code(next()?)?;
        let target = SemanticLearningPhase::from_code(next()?)?;
        let phase_index = next()?;
        let completed_updates_index = next()?;
        let word = next()?;
        let model_generation = next()?;
        let completed_updates = next()?;
        let mut identity = || -> Result<Identity256, SemanticTransitionError> {
            Ok(Identity256::from_bytes(
                reader
                    .take(32)
                    .map_err(SemanticTransitionError::Semantic)?
                    .try_into()
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
            ))
        };
        let instance = identity()?;
        let logical_digest = identity()?;
        let recipe_digest = identity()?;
        let recipe_count = reader
            .count(5 * 8)
            .map_err(SemanticTransitionError::Semantic)?;
        let mut recipe = Vec::with_capacity(recipe_count);
        for _ in 0..recipe_count {
            let role = reader.u64().map_err(SemanticTransitionError::Semantic)?;
            let index = reader.u64().map_err(SemanticTransitionError::Semantic)?;
            let code = reader.u64().map_err(SemanticTransitionError::Semantic)?;
            let source_role = reader.u64().map_err(SemanticTransitionError::Semantic)?;
            let source_index = reader.u64().map_err(SemanticTransitionError::Semantic)?;
            recipe.push(match code {
                0 if source_role == 0 && source_index == 0 => {
                    SemanticLearningCopyReset::Preserve { role, index }
                }
                1 if source_role == 0 && source_index == 0 => {
                    SemanticLearningCopyReset::Zero { role, index }
                }
                2 if role == 21 => SemanticLearningCopyReset::MasterFromEffective {
                    index,
                    effective_role: source_role,
                    effective_index: source_index,
                },
                3 if role == 23 && source_role == 0 && source_index == 0 => {
                    SemanticLearningCopyReset::Phase { index }
                }
                _ => {
                    return Err(publication_input_error(
                        "checkpoint learning recipe has an invalid operation",
                    ))
                }
            });
        }
        let admission =
            admission_owner(reader.bytes().map_err(SemanticTransitionError::Semantic)?)?;
        history.push(SemanticLearningPhaseRecord {
            source,
            target,
            phase_index,
            completed_updates_index,
            predecessor: SemanticPublishedIdentity {
                instance,
                word,
                logical_digest,
            },
            model_generation,
            completed_updates,
            recipe_digest,
            recipe: Arc::from(recipe),
            admission,
            _metadata: None,
        });
    }
    Ok(history)
}

pub(super) fn validate_history<B: PublicationByteStorage>(
    material: &PublicationMaterialStorage<B>,
) -> Result<(), SemanticTransitionError> {
    let mut previous: Option<&SemanticLearningPhaseRecord> = None;
    for record in &material.learning_phases {
        let keys = record
            .recipe
            .iter()
            .map(SemanticLearningCopyReset::key)
            .collect::<Vec<_>>();
        if record.admission.is_empty()
            || record.phase_index == record.completed_updates_index
            || keys.windows(2).any(|pair| pair[0] >= pair[1])
            || keys.iter().any(|(role, _)| !(18..=25).contains(role))
            || !record.recipe.contains(&SemanticLearningCopyReset::Phase {
                index: record.phase_index,
            })
            || !record
                .recipe
                .contains(&SemanticLearningCopyReset::Preserve {
                    role: 23,
                    index: record.completed_updates_index,
                })
            || record.recipe.iter().any(|operation| match operation {
                SemanticLearningCopyReset::Zero { role, .. } => !matches!(role, 21 | 24 | 25),
                SemanticLearningCopyReset::MasterFromEffective { effective_role, .. } => {
                    !(18..=20).contains(effective_role)
                }
                SemanticLearningCopyReset::Phase { index } => *index != record.phase_index,
                SemanticLearningCopyReset::Preserve { .. } => false,
            })
            || record.completed_updates > i64::MAX as u64
            || record.model_generation == 0
            || record.model_generation > material.bank.header.model_generation
            || record.predecessor.instance == Identity256::default()
            || record.recipe_digest == Identity256::default()
            || record.recipe_digest
                != learning_recipe_digest(
                    record.source,
                    record.target,
                    record.phase_index,
                    record.completed_updates_index,
                    record.recipe(),
                )
            || !matches!(
                (record.source, record.target),
                (
                    SemanticLearningPhase::Alignment,
                    SemanticLearningPhase::Fast
                ) | (
                    SemanticLearningPhase::Fast,
                    SemanticLearningPhase::Consolidation
                ) | (
                    SemanticLearningPhase::Consolidation,
                    SemanticLearningPhase::Fast
                )
            )
            || previous.is_none() && record.source != SemanticLearningPhase::Alignment
            || previous.is_some_and(|prior| {
                prior.target != record.source
                    || prior.phase_index != record.phase_index
                    || prior.completed_updates_index != record.completed_updates_index
                    || prior.completed_updates > record.completed_updates
                    || prior.model_generation > record.model_generation
            })
        {
            return Err(publication_input_error(
                "checkpoint learning-phase lineage is inconsistent",
            ));
        }
        previous = Some(record);
    }
    if let Some(record) = previous {
        if scalar_i64(material, (23, record.phase_index))? != record.target as i64
            || scalar_i64(material, (23, record.completed_updates_index))?
                < record.completed_updates as i64
        {
            return Err(publication_input_error(
                "selected learning values differ from their retained phase lineage",
            ));
        }
    }
    Ok(())
}
