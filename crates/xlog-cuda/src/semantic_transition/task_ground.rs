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

/// The single task-ground carrier retained from cold admission through restore.
/// Logical tasks retain their original reference semantics. Coding tasks have
/// no reference result: each external observation is absent until acquired.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticTaskGround {
    Logical,
    Coding(Vec<SemanticTaskVerificationBinding>),
}

impl SemanticTaskGround {
    pub const fn code(&self) -> u64 {
        match self {
            Self::Logical => 0,
            Self::Coding(_) => 1,
        }
    }

    pub fn bindings(&self) -> &[SemanticTaskVerificationBinding] {
        match self {
            Self::Logical => &[],
            Self::Coding(bindings) => bindings,
        }
    }

    pub(super) fn validate(
        &self,
        records: &SemanticAdmissionRecords,
        query_count: usize,
    ) -> Result<(), SemanticTransitionError> {
        let Self::Coding(bindings) = self else {
            return Ok(());
        };
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
        Ok(())
    }
}

impl SemanticTaskVerificationBinding {
    /// Exact fixed-width device representation of the original eight fields.
    /// The presence word distinguishes an absent measurement from any identity.
    pub(super) fn words(&self) -> Vec<u64> {
        let mut words = vec![u64::from(self.query), u64::from(self.observation_record)];
        for identity in [
            self.obligation_identity,
            self.requirement_identity,
            self.verifier_plan_identity,
            self.acceptance_condition_identity,
        ] {
            words.extend(super::identity_words(*identity.as_bytes()));
        }
        words.push(self.receipt_family.code());
        words.push(u64::from(self.measurement.is_some()));
        if let Some(measurement) = &self.measurement {
            for identity in [
                measurement.metric_identity,
                measurement.comparator_identity,
                measurement.tolerance_identity,
                measurement.environment_identity,
            ] {
                words.extend(super::identity_words(*identity.as_bytes()));
            }
        } else {
            words.extend([0; 16]);
        }
        words
    }
}

pub(super) const TASK_GROUND_MAGIC: u64 = 0x584c4f4754475244;
pub(super) const TASK_GROUND_VERSION: u64 = 1;

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
        Self { receipt: [0; 42], truth: 0, attained: 0 }
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
    pub rule_offset: usize,
    pub rule_capacity: usize,
    pub bytes: usize,
}

impl TaskGroundLayout {
    pub fn new(queries: usize, bindings: usize, initial_rules: Option<usize>) -> Result<Self, SemanticTransitionError> {
        let overflow = || publication_input_error("task ground geometry exceeds addressable storage");
        if queries == 0 || queries > u32::MAX as usize || bindings > u32::MAX as usize {
            return Err(overflow());
        }
        let query_offset = bindings
            .checked_mul(std::mem::size_of::<TaskObservationRecord>())
            .and_then(|bytes| bytes.checked_add(64))
            .ok_or_else(overflow)?;
        let rule_offset = queries
            .checked_mul(3)
            .and_then(|cells| cells.checked_mul(std::mem::size_of::<TaskQueryRecord>()))
            .and_then(|bytes| bytes.checked_add(query_offset))
            .ok_or_else(overflow)?;
        let rule_capacity = initial_rules.map(|count| crate::semantic_program::RESIDENT_PROGRAM_RULE_CAPACITY.checked_sub(count)
            .ok_or_else(overflow)).transpose()?.unwrap_or(0);
        let bytes = rule_capacity.checked_mul(std::mem::size_of::<crate::SemanticProgramRule>())
            .and_then(|bytes| bytes.checked_add(rule_offset)).ok_or_else(overflow)?;
        Ok(Self {
            queries,
            bindings,
            query_offset,
            rule_offset,
            rule_capacity,
            bytes,
        })
    }

    pub fn initial_bytes(self) -> Vec<u8> {
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
        bytes
    }

    pub fn validate(self, bytes: &[u8]) -> Result<(), SemanticTransitionError> {
        if bytes.len() != self.bytes {
            return Err(publication_input_error(
                "task ground storage differs from its original finite geometry",
            ));
        }
        for (actual, expected) in bytes[..64].chunks_exact(8).zip([
            TASK_GROUND_MAGIC, TASK_GROUND_VERSION, self.queries as u64,
            self.bindings as u64, 44, 18,
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
            return Err(publication_input_error("selected rules differ from their original finite bank"));
        }
        let used_rules = usize::try_from(rule_count).ok()
            .and_then(|count| count.checked_mul(std::mem::size_of::<crate::SemanticProgramRule>()))
            .and_then(|bytes| self.rule_offset.checked_add(bytes))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if bytes[used_rules..].iter().any(|byte| *byte != 0) {
            return Err(publication_input_error("unoccupied selected rule storage is not canonical"));
        }
        let mut selected = BTreeSet::new();
        for rule in bytes[self.rule_offset..used_rules].chunks_exact(std::mem::size_of::<crate::SemanticProgramRule>()) {
            if !selected.insert(rule) {
                return Err(publication_input_error("selected rule storage repeats an authored rule"));
            }
        }
        for observation in bytes[64..self.query_offset].chunks_exact(144) {
            let present = u64::from_le_bytes(observation[..8].try_into().expect("presence word"));
            let truth = u64::from_le_bytes(observation[8..16].try_into().expect("truth word"));
            if present > 1
                || (present == 0 && observation.iter().any(|byte| *byte != 0))
                || (present == 1
                    && (truth > 3
                        || observation[16..].chunks_exact(32).any(|identity| {
                            identity.iter().all(|byte| *byte == 0)
                        })))
            {
                return Err(publication_input_error(
                    "task ground observation has invalid original presence or provenance",
                ));
            }
        }
        Ok(())
    }

    pub fn query(self, bytes: &[u8], candidate: usize, query: usize) -> Result<TaskQueryRecord, SemanticTransitionError> {
        if bytes.len() != self.bytes || candidate >= 3 || query >= self.queries {
            return Err(publication_input_error("task query is outside its original ground"));
        }
        let offset = self.query_offset + (candidate * self.queries + query) * 352;
        let record = &bytes[offset..offset + 352];
        let word = |index: usize| {
            u64::from_le_bytes(record[index * 8..index * 8 + 8].try_into().expect("query record word"))
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
}

impl TaskGroundStorage {
    pub fn allocation_bytes(layout: TaskGroundLayout) -> Result<usize, SemanticTransitionError> {
        let facts = crate::semantic_program::RESIDENT_PROGRAM_FACT_CAPACITY;
        let words = layout.bindings.div_ceil(64);
        facts.checked_mul(3)
            .and_then(|cells| cells.checked_mul(2))
            .and_then(|cells| cells.checked_mul(words))
            .and_then(|cells| cells.checked_mul(8))
            .and_then(|bytes| bytes.checked_add(3 * facts))
            .and_then(|bytes| bytes.checked_add(layout.bytes))
            .ok_or_else(|| publication_input_error("task derivation geometry exceeds addressable storage"))
    }

    pub fn allocate(layout: TaskGroundLayout, reservation: &mut super::GpuMemoryReservation)
        -> Result<Self, SemanticTransitionError> {
        let facts = crate::semantic_program::RESIDENT_PROGRAM_FACT_CAPACITY;
        let words = layout.bindings.div_ceil(64);
        let cells = facts.checked_mul(3).and_then(|count| count.checked_mul(2))
            .and_then(|count| count.checked_mul(words))
            .ok_or_else(|| publication_input_error("task derivation geometry exceeds addressable storage"))?;
        Self::allocation_bytes(layout)?;
        Ok(Self {
            layout,
            device: reservation.alloc::<u8>(layout.bytes)
                .map_err(|error| super::runtime_error("task ground allocation", error))?,
            support_truths: reservation.alloc::<u8>(3 * facts)
                .map_err(|error| super::runtime_error("task support allocation", error))?,
            support_lineage: reservation.alloc::<u64>(cells)
                .map_err(|error| super::runtime_error("task provenance allocation", error))?,
        })
    }

    pub fn record(&self, recorder: &mut super::LaunchRecorder) {
        recorder.read_write(&self.device);
        recorder.write(&self.support_truths);
        recorder.write(&self.support_lineage);
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
        let executed = records.iter().any(|record| record.receipt.iter().any(|word| *word != 0));
        let mut truth = Vec::new();
        let mut attained = Vec::new();
        if executed {
            for record in &records {
                if record.truth > 3 || record.attained > 1 || record.receipt.iter().all(|word| *word == 0) {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                truth.push(record.truth);
                attained.push(record.attained);
            }
            query_count = query_count.checked_add(layout.queries as u64)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        } else if records.iter().any(|record| record.truth != 0 || record.attained != 0)
            || state.facts[candidate].g != 0 || state.facts[candidate].p != 0
            || state.facts[candidate].eligible != 0
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let correct = expected.map(|values| truth.iter().zip(values)
            .map(|(actual, expected)| u64::from(*actual == *expected as u64)).collect());
        let scalar = state.facts[candidate];
        candidates.push(super::SemanticTaskFacts {
            truth, correct, attained,
            g: scalar.g, p: scalar.p, c: scalar.c, v: scalar.v, eligible: scalar.eligible,
        });
    }
    if query_count != state.query_count {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    candidates.try_into().map_err(|_| SemanticTransitionError::ObservationMismatch)
}
