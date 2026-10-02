//! Cold learning-phase transitions over the complete native model allocation map.
//!
//! The trusted scientific owner supplies acceptance and the complete copy/reset
//! recipe. This module owns phase writes, validates alias effects, and retains
//! acceptance and ancestry; it never infers a phase from a tensor role or grant.

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
/// Acceptance is obtained from the original trusted scientific owner at the public
/// Controller boundary, separately from checking the original data-use grants.
#[derive(Clone, Debug)]
pub struct SemanticLearningPhaseTransition {
    pub source: SemanticLearningPhase,
    pub target: SemanticLearningPhase,
    pub phase_index: u64,
    pub completed_updates_index: u64,
    pub recipe: Vec<SemanticLearningCopyReset>,
    pub acceptance: Vec<u8>,
}

/// Retained evidence for one actual native cold phase change. Ordinary Update and
/// restore preserve this history; neither can reset the cumulative progress.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticLearningPhaseRecord {
    pub source: SemanticLearningPhase,
    pub target: SemanticLearningPhase,
    pub phase_index: u64,
    pub completed_updates_index: u64,
    pub predecessor: SemanticPublishedIdentity,
    pub model_generation: u64,
    pub completed_updates: u64,
    pub recipe_digest: Identity256,
    pub recipe: Vec<SemanticLearningCopyReset>,
    pub acceptance: Vec<u8>,
}

impl SemanticLearningPhaseTransition {
    pub fn recipe_digest(&self) -> Identity256 {
        let mut digest = Sha256::new();
        digest.update(b"xlog.learning-phase.copy-reset.v1\0");
        for value in [
            self.source as u64,
            self.target as u64,
            self.phase_index,
            self.completed_updates_index,
            self.recipe.len() as u64,
        ] {
            digest.update(value.to_le_bytes());
        }
        for operation in &self.recipe {
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

    pub(super) fn apply(
        &self,
        material: &mut PublicationMaterial,
    ) -> Result<SemanticLearningPhaseRecord, SemanticTransitionError> {
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
        ) || self.acceptance.is_empty()
            || self.phase_index == self.completed_updates_index
        {
            return Err(publication_input_error(
                "learning transition lacks its permitted boundary or original acceptance",
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
                "learning lineage must begin with accepted alignment",
            ));
        }
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
        let original = &material.model_allocations;
        let mut allocations = original.clone();
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
                    allocations[allocation][offset..offset + value.len()].copy_from_slice(value);
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
                    [offset..offset + range.bytes.len()]
                    .to_vec();
            }
        }
        let header = material.bank.header;
        Ok(SemanticLearningPhaseRecord {
            source: self.source,
            target: self.target,
            phase_index: self.phase_index,
            completed_updates_index: self.completed_updates_index,
            predecessor: SemanticPublishedIdentity {
                instance: header.instance,
                word: header.publication_word,
                logical_digest: header.logical_digest,
                state_digest: header.state_digest,
            },
            model_generation: header.model_generation,
            completed_updates: updates as u64,
            recipe_digest: self.recipe_digest(),
            recipe: self.recipe.clone(),
            acceptance: self.acceptance.clone(),
        })
    }
}

fn scalar_i64(
    material: &PublicationMaterial,
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

fn element_offsets<'a>(
    material: &'a PublicationMaterial,
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

fn visit_expected_values(
    material: &PublicationMaterial,
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
            record.predecessor.state_digest,
            record.recipe_digest,
        ] {
            bytes.extend_from_slice(digest.as_bytes());
        }
        material_u32(
            bytes,
            u32::try_from(record.recipe.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        );
        for operation in &record.recipe {
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
        material_bytes(bytes, &record.acceptance).map_err(SemanticTransitionError::Semantic)?;
    }
    Ok(())
}

pub(super) fn decode_history(
    reader: &mut SemanticMaterialReader<'_>,
) -> Result<Vec<SemanticLearningPhaseRecord>, SemanticTransitionError> {
    let count = reader
        .count(7 * 8 + 4 * 32 + 4)
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
        let state_digest = identity()?;
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
        let acceptance = reader
            .bytes()
            .map_err(SemanticTransitionError::Semantic)?
            .to_vec();
        history.push(SemanticLearningPhaseRecord {
            source,
            target,
            phase_index,
            completed_updates_index,
            predecessor: SemanticPublishedIdentity {
                instance,
                word,
                logical_digest,
                state_digest,
            },
            model_generation,
            completed_updates,
            recipe_digest,
            recipe,
            acceptance,
        });
    }
    Ok(history)
}

pub(super) fn validate_history(
    material: &PublicationMaterial,
) -> Result<(), SemanticTransitionError> {
    let mut previous: Option<&SemanticLearningPhaseRecord> = None;
    for record in &material.learning_phases {
        let keys = record
            .recipe
            .iter()
            .map(SemanticLearningCopyReset::key)
            .collect::<Vec<_>>();
        let recipe = SemanticLearningPhaseTransition {
            source: record.source,
            target: record.target,
            phase_index: record.phase_index,
            completed_updates_index: record.completed_updates_index,
            recipe: record.recipe.clone(),
            acceptance: record.acceptance.clone(),
        };
        if record.acceptance.is_empty()
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
            || record.recipe_digest != recipe.recipe_digest()
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
