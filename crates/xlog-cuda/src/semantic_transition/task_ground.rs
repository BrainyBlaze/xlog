use super::{publication_input_error, Identity256, SemanticTransitionError};
use crate::{SemanticAdmissionRecords, SemanticRecordRole};
use std::collections::BTreeSet;

/// Original verifier family, distinct from its observed outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SemanticTaskReceiptFamily {
    Build,
    Test,
    Proof,
    Measurement,
    Environment,
}

impl SemanticTaskReceiptFamily {
    pub const fn code(self) -> u64 {
        match self {
            Self::Build => 1,
            Self::Test => 2,
            Self::Proof => 3,
            Self::Measurement => 4,
            Self::Environment => 5,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Test => "test",
            Self::Proof => "proof",
            Self::Measurement => "measurement",
            Self::Environment => "environment",
        }
    }
}

/// Identities of the original frozen measurement descriptors and environment.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SemanticTaskMeasurementBinding {
    pub metric_identity: Identity256,
    pub comparator_identity: Identity256,
    pub tolerance_identity: Identity256,
    pub environment_identity: Identity256,
}

/// One original ordered verification obligation. The query is a desired goal;
/// the observation record is the separately protected statement the tool can
/// actually support. Neither identifies a reference answer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SemanticTaskVerificationBinding {
    pub query: u32,
    pub observation_record: u32,
    pub obligation_identity: Identity256,
    pub requirement_identity: Identity256,
    pub verifier_plan_identity: Identity256,
    pub acceptance_condition_identity: Identity256,
    pub receipt_family: SemanticTaskReceiptFamily,
    pub measurement: Option<SemanticTaskMeasurementBinding>,
}

/// Explicit immutable prerequisites for one query's attainment. Observation
/// ordinals refer to the original verification roster, not to its goal labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticTaskAttainmentRequirement {
    pub query: u32,
    pub required_goal_ordinals: Vec<u32>,
    pub required_binding_ordinals: Vec<u32>,
}

/// The single task-ground carrier retained from cold admission through restore.
/// Logical tasks retain their original reference semantics. Coding tasks have
/// no reference result: each external observation is absent until acquired.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticTaskGround {
    Logical,
    Coding {
        bindings: Vec<SemanticTaskVerificationBinding>,
        requirements: Vec<SemanticTaskAttainmentRequirement>,
    },
}

impl SemanticTaskGround {
    pub const fn code(&self) -> u64 {
        match self {
            Self::Logical => 0,
            Self::Coding { .. } => 1,
        }
    }

    pub fn bindings(&self) -> &[SemanticTaskVerificationBinding] {
        match self {
            Self::Logical => &[],
            Self::Coding { bindings, .. } => bindings,
        }
    }

    pub fn requirements(&self) -> &[SemanticTaskAttainmentRequirement] {
        match self {
            Self::Logical => &[],
            Self::Coding { requirements, .. } => requirements,
        }
    }

    /// One canonical framing shared by content commitments, goal witnesses and
    /// completed material. No consumer invents a second graph representation.
    pub fn visit_requirement_words(&self, mut visit: impl FnMut(u64)) {
        visit(self.requirements().len() as u64);
        for row in self.requirements() {
            visit(u64::from(row.query));
            visit(row.required_goal_ordinals.len() as u64);
            visit(row.required_binding_ordinals.len() as u64);
            for &index in row
                .required_goal_ordinals
                .iter()
                .chain(&row.required_binding_ordinals)
            {
                visit(u64::from(index));
            }
        }
    }

    pub fn mandatory_edge_count(&self) -> Result<u64, SemanticTransitionError> {
        self.requirements().iter().try_fold(0u64, |count, row| {
            count
                .checked_add(row.required_goal_ordinals.len() as u64)
                .and_then(|count| count.checked_add(row.required_binding_ordinals.len() as u64))
                .ok_or(SemanticTransitionError::GenerationExhausted)
        })
    }

    pub(super) fn requirement_word_count(&self) -> Result<usize, SemanticTransitionError> {
        self.requirements().iter().try_fold(1usize, |count, row| {
            count
                .checked_add(3)
                .and_then(|count| count.checked_add(row.required_goal_ordinals.len()))
                .and_then(|count| count.checked_add(row.required_binding_ordinals.len()))
                .ok_or(SemanticTransitionError::GenerationExhausted)
        })
    }

    pub(super) fn requirement_allocation_bytes(&self) -> Result<u64, SemanticTransitionError> {
        let rows = self.requirements();
        (rows.len() as u64)
            .checked_mul(std::mem::size_of::<SemanticTaskAttainmentRequirement>() as u64)
            .and_then(|bytes| bytes.checked_add(self.mandatory_edge_count().ok()?.checked_mul(4)?))
            .and_then(|bytes| {
                bytes.checked_add(
                    std::mem::size_of::<Vec<SemanticTaskAttainmentRequirement>>() as u64,
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }

    pub(super) fn requirement_workspace_bytes(&self) -> Result<u64, SemanticTransitionError> {
        let words = self.device_requirement_word_count()?;
        // Device-bank staging and the containing task-word vector overlap.
        (words as u64)
            .checked_mul(2 * std::mem::size_of::<u64>() as u64)
            .and_then(|bytes| {
                bytes.checked_add((self.requirements().len() as u64).checked_mul(
                    (std::mem::size_of::<u32>() + std::mem::size_of::<bool>()) as u64,
                )?)
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }

    fn topological_queries(&self) -> Result<Vec<u32>, SemanticTransitionError> {
        let rows = self.requirements();
        let mut complete = vec![false; rows.len()];
        let mut order = Vec::with_capacity(rows.len());
        for _ in rows {
            let next = rows
                .iter()
                .enumerate()
                .find(|(query, row)| {
                    !complete[*query]
                        && row
                            .required_goal_ordinals
                            .iter()
                            .all(|&dependency| complete.get(dependency as usize) == Some(&true))
                })
                .map(|(query, _)| query);
            let Some(query) = next else {
                return Err(publication_input_error(
                    "task attainment dependencies contain a cycle",
                ));
            };
            complete[query] = true;
            order.push(query as u32);
        }
        Ok(order)
    }

    /// Bounded device bank derived from the original graph. Offsets and the
    /// stable topological schedule are accelerators, never independent custody.
    pub(super) fn device_requirement_word_count(&self) -> Result<usize, SemanticTransitionError> {
        if self.requirements().is_empty() {
            return Ok(0);
        }
        self.requirement_word_count()?
            .checked_add(
                self.requirements()
                    .len()
                    .checked_mul(2)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?,
            )
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }

    pub(super) fn device_requirement_words(&self) -> Result<Vec<u64>, SemanticTransitionError> {
        if self.requirements().is_empty() {
            return Ok(Vec::new());
        }
        let rows = self.requirements();
        let count = self.device_requirement_word_count()?;
        let order = self.topological_queries()?;
        let mut words = Vec::with_capacity(count);
        words.push(rows.len() as u64);
        let mut offset = 1 + 2 * rows.len();
        for row in rows {
            words.push(offset as u64);
            offset += 3 + row.required_goal_ordinals.len() + row.required_binding_ordinals.len();
        }
        words.extend(order.into_iter().map(u64::from));
        let mut header = true;
        self.visit_requirement_words(|word| {
            if header {
                header = false;
            } else {
                words.push(word);
            }
        });
        Ok(words)
    }

    pub(super) fn validate(
        &self,
        records: &SemanticAdmissionRecords,
        query_records: &[u32],
    ) -> Result<(), SemanticTransitionError> {
        let Self::Coding {
            bindings,
            requirements,
        } = self
        else {
            return Ok(());
        };
        let query_count = query_records.len();
        if bindings.is_empty() || bindings.len() > u32::MAX as usize {
            return Err(publication_input_error(
                "coding task requires a finite nonempty verification roster",
            ));
        }
        let mut unique = BTreeSet::new();
        let mut families = [false; 5];
        for binding in bindings {
            // The complete row is the uniqueness key. Several verifiers may
            // share a goal, obligation or protected statement without being
            // independent evidence or acquiring an implicit all/any meaning.
            let encoded = binding.words();
            if !unique.insert(encoded)
                || binding.query as usize >= query_count
                || [
                    binding.obligation_identity,
                    binding.requirement_identity,
                    binding.verifier_plan_identity,
                    binding.acceptance_condition_identity,
                ]
                .contains(&Identity256::default())
            {
                return Err(publication_input_error(
                    "coding verification binding is duplicated or outside its original task",
                ));
            }
            let statement = records
                .records
                .get(binding.observation_record as usize)
                .ok_or_else(|| publication_input_error("observation target is not admitted"))?;
            if !records.predicates.iter().any(|predicate| {
                predicate.predicate == statement.predicate
                    && predicate.role == SemanticRecordRole::Statement
            }) {
                return Err(publication_input_error(
                    "observation target is not an admitted statement",
                ));
            }
            match (&binding.receipt_family, &binding.measurement) {
                (SemanticTaskReceiptFamily::Measurement, Some(measurement))
                    if ![
                        measurement.metric_identity,
                        measurement.comparator_identity,
                        measurement.tolerance_identity,
                        measurement.environment_identity,
                    ]
                    .contains(&Identity256::default()) => {}
                (SemanticTaskReceiptFamily::Measurement, _) | (_, Some(_)) => {
                    return Err(publication_input_error(
                        "measurement binding differs from its original descriptor roster",
                    ));
                }
                (_, None) => {}
            }
            families[binding.receipt_family.code() as usize - 1] = true;
        }
        if families.iter().any(|present| !present) {
            return Err(publication_input_error(
                "coding verification roster must cover all five original families",
            ));
        }
        if requirements.len() != query_count || query_count > u32::MAX as usize {
            return Err(publication_input_error(
                "coding attainment requires the complete ordered query graph",
            ));
        }
        for (query, row) in requirements.iter().enumerate() {
            if row.query as usize != query
                || (row.required_goal_ordinals.is_empty()
                    && row.required_binding_ordinals.is_empty())
                || row
                    .required_goal_ordinals
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                || row
                    .required_binding_ordinals
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                || row
                    .required_goal_ordinals
                    .iter()
                    .any(|&index| index as usize >= query_count || index as usize == query)
                || row.required_binding_ordinals.iter().any(|&index| {
                    bindings
                        .get(index as usize)
                        .is_none_or(|binding| binding.observation_record != query_records[query])
                })
            {
                return Err(publication_input_error("task attainment prerequisites differ from their original query or verification roster"));
            }
        }
        self.mandatory_edge_count()?;
        self.requirement_word_count()?;
        self.topological_queries()?;
        Ok(())
    }
}

impl SemanticTaskVerificationBinding {
    /// Exact fixed-width device representation of the original eight fields.
    /// The presence word distinguishes an absent measurement from any identity.
    pub(super) fn words(&self) -> [u64; 36] {
        let mut words = [0; 36];
        words[..2].copy_from_slice(&[u64::from(self.query), u64::from(self.observation_record)]);
        for (index, identity) in [
            self.obligation_identity,
            self.requirement_identity,
            self.verifier_plan_identity,
            self.acceptance_condition_identity,
        ]
        .into_iter()
        .enumerate()
        {
            words[2 + 4 * index..6 + 4 * index]
                .copy_from_slice(&super::identity_words(*identity.as_bytes()));
        }
        words[18] = self.receipt_family.code();
        words[19] = u64::from(self.measurement.is_some());
        if let Some(measurement) = &self.measurement {
            for (index, identity) in [
                measurement.metric_identity,
                measurement.comparator_identity,
                measurement.tolerance_identity,
                measurement.environment_identity,
            ]
            .into_iter()
            .enumerate()
            {
                words[20 + 4 * index..24 + 4 * index]
                    .copy_from_slice(&super::identity_words(*identity.as_bytes()));
            }
        }
        words
    }
}

pub(super) const TASK_GROUND_MAGIC: u64 = 0x584c4f4754475244;
pub(super) const TASK_GROUND_VERSION: u64 = 6;
pub(super) const INHERITED_OBSERVATIONS_MAGIC: u64 = 0x584c4f47494e484f;
pub(super) const INHERITED_OBSERVATION_BYTES: usize = 176;

/// One result from the original numerical candidate/query invocation.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TaskQueryRecord {
    pub receipt: [u64; 42],
    pub truth: u64,
    pub attained: u64,
}

impl Default for TaskQueryRecord {
    fn default() -> Self {
        Self {
            receipt: [0; 42],
            truth: 0,
            attained: 0,
        }
    }
}

/// A separately authenticated tool observation. All-zero storage represents
/// absence, not a fifth truth value and not an observation of Neither.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TaskObservationRecord {
    pub present: u64,
    pub truth: u64,
    pub receipt_identity: Identity256,
    pub action_identity: Identity256,
    pub intent_identity: Identity256,
    pub support_identity: Identity256,
}

/// Finite original query/binding geometry, never a caller-authored work cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TaskGroundLayout {
    pub queries: usize,
    pub bindings: usize,
    pub query_offset: usize,
    pub image_offset: usize,
    pub snapshot_offset: usize,
    pub image: super::program_image::ProgramImageLayout,
    pub rule_offset: usize,
    pub rule_capacity: usize,
    pub inherited_capacity: usize,
    pub inherited_offset: usize,
    pub activity_offset: usize,
    pub library_count: usize,
    pub bytes: usize,
}

impl TaskGroundLayout {
    pub fn material_geometry(bytes: &[u8]) -> Result<Self, SemanticTransitionError> {
        let footer_bytes = super::program_image::IMAGE_GEOMETRY_WORDS * 8;
        if bytes.len() < 64 + footer_bytes {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let word =
            |index: usize| u64::from_le_bytes(bytes[index * 8..index * 8 + 8].try_into().unwrap());
        let image = super::program_image::ProgramImageLayout::from_footer(
            &bytes[bytes.len() - footer_bytes..],
        )?;
        let layout = Self::geometry(
            usize::try_from(word(2)).map_err(|_| SemanticTransitionError::ObservationMismatch)?,
            usize::try_from(word(3)).map_err(|_| SemanticTransitionError::ObservationMismatch)?,
            image,
        )?;
        let capacity = usize::try_from(word(bytes.len() / 8 - 1))
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        let layout = layout.with_inherited_observations(capacity)?;
        if layout.bytes != bytes.len() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(layout)
    }

    pub fn from_material(bytes: &[u8]) -> Result<Self, SemanticTransitionError> {
        let layout = Self::material_geometry(bytes)?;
        layout.validate(bytes)?;
        Ok(layout)
    }

    pub fn new(
        queries: usize,
        bindings: usize,
        program: Option<&crate::SemanticProgramAdmission>,
    ) -> Result<Self, SemanticTransitionError> {
        Self::geometry(
            queries,
            bindings,
            super::program_image::ProgramImageLayout::new(program)?,
        )
    }

    fn geometry(
        queries: usize,
        bindings: usize,
        image: super::program_image::ProgramImageLayout,
    ) -> Result<Self, SemanticTransitionError> {
        let overflow =
            || publication_input_error("task ground geometry exceeds addressable storage");
        if queries == 0 || queries > u32::MAX as usize || bindings > u32::MAX as usize {
            return Err(overflow());
        }
        let query_offset = bindings
            .checked_mul(std::mem::size_of::<TaskObservationRecord>())
            .and_then(|bytes| bytes.checked_add(64))
            .ok_or_else(overflow)?;
        let image_offset = queries
            .checked_mul(3)
            .and_then(|cells| cells.checked_mul(std::mem::size_of::<TaskQueryRecord>()))
            .and_then(|bytes| bytes.checked_add(query_offset))
            .ok_or_else(overflow)?;
        let rule_offset = image_offset
            .checked_add(image.rule_offset)
            .ok_or_else(overflow)?;
        let snapshot_offset = image
            .frame_bytes
            .checked_mul(4)
            .and_then(|bytes| image_offset.checked_add(bytes))
            .ok_or_else(overflow)?;
        let inherited_offset = image
            .statements
            .checked_mul(42 * 8)
            .and_then(|bytes| bytes.checked_add(image.entry_supports.checked_mul(13 * 8)?))
            .and_then(|bytes| bytes.checked_add(bindings.checked_mul(144)?))
            .and_then(|bytes| bytes.checked_add(32 * 8))
            .and_then(|bytes| snapshot_offset.checked_add(bytes))
            .and_then(|bytes| bytes.checked_add(16))
            .ok_or_else(overflow)?;
        Ok(Self {
            queries,
            bindings,
            query_offset,
            image_offset,
            snapshot_offset,
            image,
            rule_offset,
            rule_capacity: image.mutations,
            inherited_capacity: 0,
            inherited_offset,
            activity_offset: inherited_offset,
            library_count: image.library,
            bytes: inherited_offset
                .checked_add(super::program_image::IMAGE_GEOMETRY_WORDS * 8)
                .ok_or_else(overflow)?,
        })
    }

    pub fn frame(self, slot: usize) -> Result<std::ops::Range<usize>, SemanticTransitionError> {
        if slot >= 4 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let begin = self.image_offset + slot * self.image.frame_bytes;
        Ok(begin..begin + self.image.frame_bytes)
    }

    pub fn with_inherited_observations(
        mut self,
        capacity: usize,
    ) -> Result<Self, SemanticTransitionError> {
        if capacity > u32::MAX as usize
            || self
                .bindings
                .checked_add(capacity)
                .and_then(|count| count.checked_add(self.image.statements))
                .is_none_or(|count| count > u32::MAX as usize / 2)
        {
            return Err(publication_input_error(
                "inherited observation roster exceeds its finite native domain",
            ));
        }
        self.inherited_offset = self
            .snapshot_offset
            .checked_add(32 * 8)
            .and_then(|bytes| bytes.checked_add(self.bindings.checked_mul(144)?))
            .and_then(|bytes| bytes.checked_add(capacity.checked_mul(INHERITED_OBSERVATION_BYTES)?))
            .and_then(|bytes| bytes.checked_add(self.image.statements.checked_mul(42 * 8)?))
            .and_then(|bytes| bytes.checked_add(self.image.entry_supports.checked_mul(13 * 8)?))
            .and_then(|bytes| bytes.checked_add(16))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        self.activity_offset = capacity
            .checked_mul(INHERITED_OBSERVATION_BYTES)
            .and_then(|bytes| self.inherited_offset.checked_add(bytes))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        self.bytes = self
            .activity_offset
            .checked_add(super::program_image::IMAGE_GEOMETRY_WORDS * 8)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        self.inherited_capacity = capacity;
        Ok(self)
    }

    pub fn with_entry_supports(mut self, capacity: usize) -> Result<Self, SemanticTransitionError> {
        self.image = self.image.with_entry_supports(capacity)?;
        self.with_inherited_observations(self.inherited_capacity)
    }

    pub fn require_program_workspace(
        self,
        program: &crate::SemanticProgramAdmission,
    ) -> Result<(), SemanticTransitionError> {
        let rules = program
            .initial_rules
            .len()
            .checked_add(self.image.mutations)
            .and_then(|count| count.checked_add(self.image.library))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let stack = self
            .image
            .facts
            .checked_mul(5 * std::mem::size_of::<crate::SemanticProgramFact>())
            .and_then(|bytes| {
                bytes.checked_add(
                    rules.checked_mul(std::mem::size_of::<crate::SemanticProgramRule>())?,
                )
            })
            .and_then(|bytes| bytes.checked_add(self.image.nodes.checked_mul(2)?))
            .and_then(|bytes| bytes.checked_add(3))
            .map(|bytes| bytes & !3)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let bytes = self
            .image
            .nodes
            .checked_mul(4)
            .and_then(|bytes| stack.checked_add(bytes))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if rules > u32::MAX as usize || bytes > super::SCRATCH_BYTES - 8 * 1024 * 1024 {
            return Err(publication_input_error(
                "program forest workspace exceeds the original native scratch",
            ));
        }
        Ok(())
    }

    pub fn support_binding_capacity(self) -> usize {
        self.bindings + self.inherited_capacity + self.image.statements
    }

    pub fn with_rule_library(mut self, count: usize) -> Result<Self, SemanticTransitionError> {
        self.image = self.image.with_library(count)?;
        self.library_count = count;
        Ok(self)
    }

    pub fn block_activity(
        self,
        bytes: &[u8],
        slot: usize,
        block: usize,
    ) -> Result<u64, SemanticTransitionError> {
        if bytes.len() != self.bytes || slot >= 3 || block >= self.image.blocks {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let begin = self.frame(slot)?.start + self.image.activity_offset + block * 8;
        Ok(u64::from_le_bytes(
            bytes[begin..begin + 8].try_into().unwrap(),
        ))
    }

    pub fn rule_activity(self, bytes: &[u8], slot: usize) -> Result<u64, SemanticTransitionError> {
        (0..self.image.blocks).try_fold(0, |mask, block| {
            Ok(mask | self.block_activity(bytes, slot, block)?)
        })
    }

    pub fn inherited_row(
        self,
        binding: usize,
    ) -> Result<std::ops::Range<usize>, SemanticTransitionError> {
        if binding >= self.inherited_capacity {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let begin = self.inherited_offset + binding * INHERITED_OBSERVATION_BYTES;
        Ok(begin..begin + INHERITED_OBSERVATION_BYTES)
    }

    pub fn initial_bytes(
        self,
        program: Option<&crate::SemanticProgramAdmission>,
        owner: Option<[u64; 4]>,
    ) -> Result<Vec<u8>, SemanticTransitionError> {
        let mut bytes = vec![0; self.bytes];
        for (index, word) in [
            TASK_GROUND_MAGIC,
            TASK_GROUND_VERSION,
            self.queries as u64,
            self.bindings as u64,
            44,
            18,
            0,
            self.rule_capacity as u64,
        ]
        .into_iter()
        .enumerate()
        {
            bytes[index * 8..index * 8 + 8].copy_from_slice(&word.to_le_bytes());
        }
        bytes[self.inherited_offset - 16..self.inherited_offset - 8]
            .copy_from_slice(&INHERITED_OBSERVATIONS_MAGIC.to_le_bytes());
        bytes[self.inherited_offset - 8..self.inherited_offset]
            .copy_from_slice(&(self.inherited_capacity as u64).to_le_bytes());
        let frame = self.image.initial_frame(program, owner)?;
        for slot in 0..4 {
            bytes[self.frame(slot)?].copy_from_slice(&frame);
        }
        let mut footer = self.image.footer();
        footer[31] = self.inherited_capacity as u64;
        for (index, value) in footer.into_iter().enumerate() {
            let begin = self.activity_offset + index * 8;
            bytes[begin..begin + 8].copy_from_slice(&value.to_le_bytes());
        }
        Ok(bytes)
    }

    pub fn validate(self, bytes: &[u8]) -> Result<(), SemanticTransitionError> {
        if bytes.len() != self.bytes {
            return Err(publication_input_error(
                "task ground storage differs from its original finite geometry",
            ));
        }
        for (actual, expected) in bytes[..64].chunks_exact(8).zip([
            TASK_GROUND_MAGIC,
            TASK_GROUND_VERSION,
            self.queries as u64,
            self.bindings as u64,
            44,
            18,
        ]) {
            if actual != expected.to_le_bytes() {
                return Err(publication_input_error(
                    "task ground header differs from its original finite geometry",
                ));
            }
        }
        let rule_count = u64::from_le_bytes(bytes[48..56].try_into().expect("rule count"));
        let rule_capacity = u64::from_le_bytes(bytes[56..64].try_into().expect("rule capacity"));
        if rule_capacity != self.rule_capacity as u64 || rule_count > rule_capacity {
            return Err(publication_input_error(
                "selected rules differ from their original finite bank",
            ));
        }
        for slot in 0..4 {
            let frame = &bytes[self.frame(slot)?];
            self.image.validate_frame(frame)?;
            let count = u64::from_le_bytes(frame[32..40].try_into().unwrap());
            if slot == 0 && count != rule_count {
                return Err(publication_input_error(
                    "selected image and task rule counts disagree",
                ));
            }
        }
        let snapshot = &bytes[self.snapshot_offset..self.inherited_offset - 16];
        let snapshot_word = |index: usize| {
            u64::from_le_bytes(snapshot[index * 8..index * 8 + 8].try_into().unwrap())
        };
        if snapshot_word(0) == 0 {
            if snapshot.iter().any(|&value| value != 0) {
                return Err(publication_input_error(
                    "unmaterialized native entry view has nonzero data",
                ));
            }
        } else {
            if snapshot_word(0) != 0x584c4f47534e4150
                || snapshot_word(1) != 1
                || snapshot_word(22) != 1
                || snapshot[2 * 8..5 * 8].iter().any(|&value| value != 0)
                || snapshot[24 * 8..32 * 8].iter().any(|&value| value != 0)
                || snapshot_word(23) > self.image.entry_supports as u64
                || snapshot[5 * 8..9 * 8].iter().all(|&value| value == 0)
            {
                return Err(publication_input_error(
                    "native entry view is incomplete or lacks its root binding",
                ));
            }
            let frame = &bytes[self.frame(3)?];
            if snapshot[9 * 8..10 * 8] != frame[2 * 8..3 * 8]
                || snapshot[10 * 8..14 * 8] != frame[10 * 8..14 * 8]
            {
                return Err(publication_input_error(
                    "native entry view changed its acquired image generation",
                ));
            }
            let observed_end = 32 * 8
                + self.bindings * 144
                + self.inherited_capacity * INHERITED_OBSERVATION_BYTES;
            use sha2::{Digest, Sha256};
            let mut observed = Sha256::new();
            observed.update((self.bindings as u64).to_le_bytes());
            observed.update((self.inherited_capacity as u64).to_le_bytes());
            observed.update(&snapshot[32 * 8..observed_end]);
            let observed: [u8; 32] = observed.finalize().into();
            if snapshot[14 * 8..18 * 8] != observed {
                return Err(publication_input_error(
                    "native entry view changed its authenticated observations",
                ));
            }
            let query_end = observed_end + self.image.statements * 42 * 8;
            for row in snapshot[observed_end..query_end].chunks_exact(42 * 8) {
                if u64::from_le_bytes(row[..8].try_into().unwrap()) != 0
                    || u64::from_le_bytes(row[16..24].try_into().unwrap()) > 3
                    || row[16 * 8..20 * 8] != snapshot[5 * 8..9 * 8]
                    || row[3 * 8..13 * 8]
                        .iter()
                        .chain(&row[34 * 8..42 * 8])
                        .any(|&value| value != 0)
                {
                    return Err(publication_input_error(
                        "native entry query differs from its bound world root",
                    ));
                }
            }
            let count = snapshot_word(23) as usize;
            for row in snapshot[query_end..query_end + count * 13 * 8].chunks_exact(13 * 8) {
                let word = |index: usize| {
                    u64::from_le_bytes(row[index * 8..index * 8 + 8].try_into().unwrap())
                };
                if word(0) >= self.image.statements as u64
                    || word(1) > u32::MAX as u64
                    || word(2) > u32::MAX as u64
                    || !(1..=2).contains(&word(8))
                    || row[3 * 8..7 * 8].iter().any(|&value| value != 0)
                    || row[9 * 8..13 * 8].iter().all(|&value| value == 0)
                {
                    return Err(publication_input_error(
                        "native entry support changed its canonical provenance",
                    ));
                }
            }
            if snapshot[query_end + count * 13 * 8..]
                .iter()
                .any(|&value| value != 0)
            {
                return Err(publication_input_error(
                    "native entry support tail is not canonical",
                ));
            }
            let mut view = Sha256::new();
            view.update(&snapshot[..18 * 8]);
            view.update(&snapshot[23 * 8..24 * 8]);
            view.update(&snapshot[32 * 8..]);
            let view: [u8; 32] = view.finalize().into();
            if snapshot[18 * 8..22 * 8] != view {
                return Err(publication_input_error(
                    "native entry derived view changed its complete material",
                ));
            }
        }
        if bytes[self.inherited_offset - 16..self.inherited_offset - 8]
            != INHERITED_OBSERVATIONS_MAGIC.to_le_bytes()
            || bytes[self.inherited_offset - 8..self.inherited_offset]
                != (self.inherited_capacity as u64).to_le_bytes()
        {
            return Err(publication_input_error(
                "inherited observation storage changes its frozen capacity",
            ));
        }
        if super::program_image::ProgramImageLayout::from_footer(&bytes[self.activity_offset..])?
            != self.image
        {
            return Err(publication_input_error(
                "selected program image changes its frozen geometry",
            ));
        }
        for (binding, row) in bytes[self.inherited_offset..self.activity_offset]
            .chunks_exact(INHERITED_OBSERVATION_BYTES)
            .enumerate()
        {
            let word = |offset| u64::from_le_bytes(row[offset..offset + 8].try_into().unwrap());
            let arity = u32::from_le_bytes(row[20..24].try_into().unwrap());
            if word(0) != binding as u64
                || word(8) > u32::MAX as u64
                || !(1..=2).contains(&arity)
                || (arity == 1 && row[28..32] != [0; 4])
            {
                return Err(publication_input_error(
                    "inherited observation changes its frozen fact or original binding ordinal",
                ));
            }
        }
        for observation in bytes[64..self.query_offset].chunks_exact(144).chain(
            bytes[self.inherited_offset..self.activity_offset]
                .chunks_exact(INHERITED_OBSERVATION_BYTES)
                .map(|row| &row[32..]),
        ) {
            let present = u64::from_le_bytes(observation[..8].try_into().expect("presence word"));
            let truth = u64::from_le_bytes(observation[8..16].try_into().expect("truth word"));
            if present > 2
                || (present == 0 && observation.iter().any(|byte| *byte != 0))
                || (present != 0
                    && (truth > 3
                        || (present == 2 && observation[48..80] != observation[80..112])
                        || observation[16..]
                            .chunks_exact(32)
                            .any(|identity| identity.iter().all(|byte| *byte == 0))))
            {
                return Err(publication_input_error(
                    "task ground observation has invalid original presence or provenance",
                ));
            }
        }
        Ok(())
    }

    pub fn query(
        self,
        bytes: &[u8],
        candidate: usize,
        query: usize,
    ) -> Result<TaskQueryRecord, SemanticTransitionError> {
        if bytes.len() != self.bytes || candidate >= 3 || query >= self.queries {
            return Err(publication_input_error(
                "task query is outside its original ground",
            ));
        }
        let offset = self.query_offset + (candidate * self.queries + query) * 352;
        let record = &bytes[offset..offset + 352];
        let word = |index: usize| {
            u64::from_le_bytes(
                record[index * 8..index * 8 + 8]
                    .try_into()
                    .expect("query record word"),
            )
        };
        let receipt = std::array::from_fn(word);
        Ok(TaskQueryRecord {
            receipt,
            truth: word(42),
            attained: word(43),
        })
    }
}

const _: () = {
    assert!(std::mem::size_of::<TaskQueryRecord>() == 352);
    assert!(std::mem::size_of::<TaskObservationRecord>() == 144);
};

/// Invocation-owned output and least-fixpoint support scratch. The publication
/// owns the durable observation plane; each original invocation owns its own
/// query receipts and derivations even after either publication bank is reused.
pub(super) struct TaskGroundStorage {
    pub layout: TaskGroundLayout,
    pub device: super::TrackedCudaSlice<u8>,
    pub support_truths: super::TrackedCudaSlice<u8>,
    pub support_lineage: super::TrackedCudaSlice<u64>,
    pub measure_geometry: super::world_state::MeasureGeometry,
    pub measure: super::TrackedCudaSlice<u64>,
}

impl TaskGroundStorage {
    pub fn allocation_bytes(
        layout: TaskGroundLayout,
        measure: super::world_state::MeasureGeometry,
    ) -> Result<usize, SemanticTransitionError> {
        let facts = layout.image.facts;
        let words = (2 * layout.support_binding_capacity()).div_ceil(64) + 1;
        facts
            .checked_mul(5)
            .and_then(|cells| cells.checked_mul(2))
            .and_then(|cells| cells.checked_mul(words))
            .and_then(|cells| cells.checked_mul(8))
            .and_then(|bytes| bytes.checked_add(5 * facts))
            .and_then(|bytes| bytes.checked_add(layout.bytes))
            .and_then(|bytes| bytes.checked_add(measure.words.checked_mul(8)?))
            .ok_or_else(|| {
                publication_input_error("task derivation geometry exceeds addressable storage")
            })
    }

    pub fn allocate(
        layout: TaskGroundLayout,
        measure_geometry: super::world_state::MeasureGeometry,
        reservation: &mut super::GpuMemoryReservation,
    ) -> Result<Self, SemanticTransitionError> {
        let facts = layout.image.facts;
        let words = (2 * layout.support_binding_capacity()).div_ceil(64) + 1;
        let cells = facts
            .checked_mul(5)
            .and_then(|count| count.checked_mul(2))
            .and_then(|count| count.checked_mul(words))
            .ok_or_else(|| {
                publication_input_error("task derivation geometry exceeds addressable storage")
            })?;
        Self::allocation_bytes(layout, measure_geometry)?;
        Ok(Self {
            layout,
            device: reservation
                .alloc::<u8>(layout.bytes)
                .map_err(|error| super::runtime_error("task ground allocation", error))?,
            support_truths: reservation
                .alloc::<u8>(5 * facts)
                .map_err(|error| super::runtime_error("task support allocation", error))?,
            support_lineage: reservation
                .alloc::<u64>(cells)
                .map_err(|error| super::runtime_error("task provenance allocation", error))?,
            measure_geometry,
            measure: reservation
                .alloc::<u64>(measure_geometry.words)
                .map_err(|error| {
                    super::runtime_error("original joint measure allocation", error)
                })?,
        })
    }

    pub fn record(&self, recorder: &mut super::LaunchRecorder) {
        recorder.read_write(&self.device);
        recorder.write(&self.support_truths);
        recorder.write(&self.support_lineage);
        recorder.read_write(&self.measure);
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn export_measure(
        &self,
        device: i32,
        aliases: std::sync::Arc<()>,
        publication: std::sync::Arc<super::PublicationStorage>,
    ) -> Result<[super::DlpackManagedTensor; 5], SemanticTransitionError> {
        let geometry = self.measure_geometry;
        if self.measure.len() != geometry.words {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let queries = geometry.queries;
        let factors = geometry.factors;
        let views = [
            (0, 32, vec![32], vec![1], (1, 64)),
            (
                geometry.masses,
                12 * queries,
                vec![3, queries as i64, 4],
                vec![4 * queries as i64, 4, 1],
                (2, 64),
            ),
            (14, 9, vec![3, 3], vec![3, 1], (2, 64)),
            (
                geometry.factor_gradients,
                6 * factors,
                vec![3, factors as i64, 2],
                vec![2 * factors as i64, 2, 1],
                (2, 64),
            ),
            (
                geometry.log_potentials,
                2 * factors,
                vec![factors as i64, 2],
                vec![2, 1],
                (2, 64),
            ),
        ];
        let tensors = views
            .into_iter()
            .map(|(begin, count, shape, strides, dtype)| {
                let end = begin
                    .checked_add(count)
                    .filter(|end| *end <= geometry.words)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                // SAFETY: all views are exact aligned spans of the invocation's
                // original u64 arena. Only FP64 result regions are labelled FP64;
                // the status/header view remains unsigned integer storage.
                let view = unsafe { self.measure.view().slice(begin..end).cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                Ok(super::export_owned_allocation(
                    view,
                    shape,
                    strides,
                    dtype,
                    device,
                    std::sync::Arc::clone(&aliases),
                    Some(std::sync::Arc::clone(&publication)),
                    None,
                    None,
                ))
            })
            .collect::<Result<Vec<_>, SemanticTransitionError>>()?;
        tensors
            .try_into()
            .map_err(|_| SemanticTransitionError::ObservationMismatch)
    }
}

pub(super) fn decoded_task_facts(
    layout: TaskGroundLayout,
    bytes: &[u8],
    state: &super::DeviceTaskEvaluation,
    expected: Option<&[crate::SemanticTruth]>,
) -> Result<[super::SemanticTaskFacts; 3], SemanticTransitionError> {
    layout.validate(bytes)?;
    if expected.is_some_and(|values| values.len() != layout.queries) {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let mut candidates = Vec::with_capacity(3);
    let mut query_count = 0u64;
    for candidate in 0..3 {
        let records = (0..layout.queries)
            .map(|query| layout.query(bytes, candidate, query))
            .collect::<Result<Vec<_>, _>>()?;
        let executed = records
            .iter()
            .any(|record| record.receipt.iter().any(|word| *word != 0));
        let mut truth = Vec::new();
        let mut attained = Vec::new();
        if executed {
            for record in &records {
                if record.truth > 3
                    || record.attained > 1
                    || record.receipt.iter().all(|word| *word == 0)
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                truth.push(record.truth);
                attained.push(record.attained);
            }
            query_count = query_count
                .checked_add(layout.queries as u64)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        } else if records
            .iter()
            .any(|record| record.truth != 0 || record.attained != 0)
            || state.facts[candidate].g != 0
            || state.facts[candidate].p != 0
            || state.facts[candidate].eligible != 0
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let correct = expected.map(|values| {
            truth
                .iter()
                .zip(values)
                .map(|(actual, expected)| u64::from(*actual == *expected as u64))
                .collect()
        });
        let scalar = state.facts[candidate];
        candidates.push(super::SemanticTaskFacts {
            truth,
            correct,
            attained,
            g: scalar.g,
            p: scalar.p,
            c: scalar.c,
            v: scalar.v,
            eligible: scalar.eligible,
        });
    }
    if query_count != state.query_count {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    candidates
        .try_into()
        .map_err(|_| SemanticTransitionError::ObservationMismatch)
}
