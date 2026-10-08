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
    pub admission: Vec<u8>,
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
    pub admission: Vec<u8>,
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
        ) || self.admission.is_empty()
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
            Some(LearningFoldPlan::prepare(material, self)?)
        } else {
            None
        };
        // Absorption reads the original effective factors on the device. In
        // particular, its masters must not be reset from pre-absorption values.
        if fold.is_none() {
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
                        allocations[allocation][offset..offset + value.len()]
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
                        [offset..offset + range.bytes.len()]
                        .to_vec();
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
                    state_digest: header.state_digest,
                },
                model_generation: header.model_generation,
                completed_updates: updates as u64,
                recipe_digest: self.recipe_digest(),
                recipe: self.recipe.clone(),
                admission: self.admission.clone(),
            },
            fold,
        ))
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

struct FoldAssignment {
    key: (u64, u64),
    mode: u64,
    source: Option<(u64, u64)>,
    factors: Option<((u64, u64), (u64, u64))>,
}

/// Derived only from the complete original ModelContract and native memory map.
/// This is an ephemeral launch plan, not a new producer roster or wire format.
pub(super) struct LearningFoldPlan {
    assignments: Vec<FoldAssignment>,
    scale_bits: u32,
    phase: u64,
}

fn fold_error() -> SemanticTransitionError {
    publication_input_error(
        "adapter absorption differs from its original model contract or physical views",
    )
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

fn view_key(
    value: &serde_json::Value,
    material: &PublicationMaterial,
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
    fn prepare(
        material: &PublicationMaterial,
        transition: &SemanticLearningPhaseTransition,
    ) -> Result<Self, SemanticTransitionError> {
        let record = material
            .ranges
            .iter()
            .find(|r| (r.range.role, r.range.index) == (44, 0))
            .ok_or_else(fold_error)?;
        let schema = retained_model_schema(&record.bytes, material.contract.model_contract_layout)?
            .ok_or_else(fold_error)?;
        let model = &schema["model"];
        let physical = model["physical"].as_object().ok_or_else(fold_error)?;
        let learning = &model["learning"];
        let views = schema_array(&learning["views"])?;
        let keys = views
            .iter()
            .map(|v| view_key(v, material))
            .collect::<Result<Vec<_>, _>>()?;
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
            if names.is_empty() || owner_aliases.insert(key, names.clone()).is_some() {
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
        let state = schema_array(&model["roots"]["adapter"]["modules"][""][1])?;
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
        let mut assignments = Vec::new();
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
            let names = [
                format!("model.{target}.weight"),
                format!("adapter.residuals.{index}.down"),
                format!("adapter.residuals.{index}.up"),
                format!("adapter.residuals.{index}.neutral_down"),
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

impl SemanticTransitionSession {
    pub(super) fn apply_learning_fold(
        &mut self,
        plan: &LearningFoldPlan,
    ) -> Result<(), SemanticTransitionError> {
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        // A cold fold has a genuine private writer, not a second persistent
        // model bank. Preserve all bytes (including alias padding) before the
        // unchanged compute/scatter/all-view-check law and sole assignment.
        let scratch = storage
            .model_memory
            .allocation_bytes
            .iter()
            .map(|&bytes| allocate_publication::<u8>(&self.provider, bytes as usize))
            .collect::<Result<Vec<_>, _>>()?;
        let mut recorder = self.domain.new_strict_recorder();
        for (index, destination) in scratch.iter().enumerate() {
            recorder.read(&storage.allocations[storage.model_slots[index][0]].slice()?);
            recorder.write(destination);
        }
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            for (index, destination) in scratch.iter().enumerate() {
                if destination.is_empty() {
                    continue;
                }
                // SAFETY: full source/scratch allocations are disjoint, retained
                // and recorded; no descriptor names this cold fold's writer.
                unsafe {
                    sys::cuMemcpyDtoDAsync_v2(
                        destination.device_ptr_value(),
                        storage.allocations[storage.model_slots[index][0]]
                            .entry()
                            .pointer,
                        destination.len(),
                        enqueue.stream().cu_stream(),
                    )
                }
                .result()
                .map_err(|error| XlogError::Kernel(error.to_string()))?;
            }
            Ok::<(), XlogError>(())
        })?;
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
        let mut outputs = Vec::new();
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
        let mut descriptors = Vec::new();
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
            let mut axes = (0..layout.rank as usize)
                .filter(|&i| layout.dimensions[i] > 1)
                .collect::<Vec<_>>();
            axes.sort_unstable_by_key(|&i| layout.strides_bytes[i]);
            let mut span = layout.element_bytes;
            for axis in axes {
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
        upload_publication(&self.provider, &[0u64], &status)?;
        let execute = self
            .provider
            .device()
            .inner()
            .get_func("xlog_semantic_transition", "semantic_learning_fold")
            .ok_or_else(|| runtime_error("kernel lookup", "learning absorption unavailable"))?;
        let mut recorder = self.domain.new_strict_recorder();
        storage.record(&mut recorder);
        for allocation in &scratch {
            recorder.read_write(allocation);
        }
        recorder.read_write(&status);
        for output in outputs.iter().flatten() {
            recorder.read_write(output);
        }
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            // Every expectation is computed before scatter. Masters consume the
            // already-rounded effective outputs, never an unrounded accumulator.
            for stage in 0..4u64 {
                for descriptor in &descriptors {
                    if descriptor.count == 0
                        || (stage == 0 && matches!(descriptor.mode, 0 | 2))
                        || (stage == 1 && descriptor.mode != 2)
                        || (stage == 2 && descriptor.mode == 0)
                    {
                        continue;
                    }
                    let blocks = descriptor.count.div_ceil(256).min(65535) as u32;
                    // SAFETY: original, scratch, expectation and all factor
                    // spans are derived from the complete retained native map;
                    // their owners were recorded before this enqueue boundary.
                    unsafe {
                        execute.clone().launch_in(
                            enqueue,
                            LaunchConfig {
                                grid_dim: (blocks, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (*descriptor, stage, status.device_ptr_value()),
                        )
                    }
                    .map_err(|error| XlogError::Kernel(error.to_string()))?;
                }
            }
            Ok::<(), XlogError>(())
        })?;
        wait_on_stream(
            &self.stream,
            &mut self.poisoned,
            &mut self.stream_waits,
            "cold adapter absorption and physical alias check",
            CudaStream::synchronize,
        )?;
        if self.publication_read(status.view())?[0] != 0 {
            return Err(publication_input_error("adapter absorption encountered nonfinite numerics or conflicting physical alias bytes"));
        }
        let mut recorder = self.domain.new_strict_recorder();
        storage.record(&mut recorder);
        for (index, allocation) in scratch.iter().enumerate() {
            recorder.read(allocation);
            recorder.write(&storage.allocations[storage.model_slots[index][0]].slice()?);
        }
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            for (index, slots) in storage.model_slots.iter().enumerate() {
                let destination = &storage.allocations[slots[0]];
                if destination.is_empty() {
                    continue;
                }
                // SAFETY: the complete unsealed scratch allocation passed every
                // original view before this first candidate-bank assignment.
                unsafe {
                    sys::cuMemcpyDtoDAsync_v2(
                        destination.entry().pointer,
                        scratch[index].device_ptr_value(),
                        destination.len(),
                        enqueue.stream().cu_stream(),
                    )
                }
                .result()
                .map_err(|error| XlogError::Kernel(error.to_string()))?;
            }
            Ok::<(), XlogError>(())
        })?;
        wait_on_stream(
            &self.stream,
            &mut self.poisoned,
            &mut self.stream_waits,
            "cold absorbed candidate assignment",
            CudaStream::synchronize,
        )
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
        material_bytes(bytes, &record.admission).map_err(SemanticTransitionError::Semantic)?;
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
        let admission = reader
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
            admission,
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
            admission: record.admission.clone(),
        };
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
